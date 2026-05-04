import logging
import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
from abc import ABC, abstractmethod
from collections import deque
from hydra.utils import instantiate
from il_lib.utils.array_tensor_utils import any_concat
from il_lib.utils.convert_utils import any_to_torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from omnigibson.learning.utils.eval_utils import (
    ACTION_QPOS_INDICES,
    PROPRIOCEPTION_INDICES,
    PROPRIO_QPOS_INDICES,
    JOINT_RANGE,
    ROBOT_CAMERA_NAMES,
    CAMERA_INTRINSICS,
    EEF_POSITION_RANGE,
)
from omnigibson.learning.utils.obs_utils import (
    create_video_writer, 
    process_fused_point_cloud,
    MIN_DEPTH,
    MAX_DEPTH,
)
from omnigibson.macros import gm
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from typing import Any, Dict, List, Optional


logger = logging.getLogger("BasePolicy")


class BasePolicy(LightningModule, ABC):
    """
    Base class for policies that is used for training and rollout
    """

    def __init__(
        self, 
        *args,
        online_eval: Optional[DictConfig] = None, 
        policy_wrapper: Optional[DictConfig] = None, 
        robot_type: str = "R1Pro",
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        # require evaluator for online testing
        self.online_eval_config = online_eval
        self.policy_wrapper_config = policy_wrapper
        if self.online_eval_config is not None:
            OmegaConf.resolve(self.online_eval_config)
            assert self.policy_wrapper_config is not None, "policy_wrapper config must be provided for online evaluation!"
            OmegaConf.resolve(self.policy_wrapper_config)
        else:
            logger.info("No evaluation config provided, online evaluation will not be performed during training.")
        self.evaluator = None
        self.test_id = 0
        self.robot_type = robot_type

    @abstractmethod
    def forward(self, obs: dict, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass of the policy.
        This is used for inference and should return the action.
        """
        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def act(self, obs, policy_state, deterministic=None) -> torch.Tensor:
        """
        Args:
            obs: dict of (B, L=1, ...)
            policy_state: (h_0, c_0) or h_0
            deterministic: whether to use deterministic action or not
        Returns:
            action: (B, L=1, A) where A is the action dimension
        """
        raise NotImplementedError
    
    @abstractmethod
    def reset(self) -> None:
        """
        Reset the policy
        """
        raise NotImplementedError

    @abstractmethod
    def policy_training_step(self, batch, batch_idx) -> Any:
        raise NotImplementedError

    @abstractmethod
    def policy_evaluation_step(self, batch, batch_idx) -> Any:
        raise NotImplementedError

    @abstractmethod
    def configure_optimizers(self) -> OptimizerLRScheduler:
        """
        Get optimizers, which are subsequently used to train.
        """
        raise NotImplementedError

    def training_step(self, *args, **kwargs):
        loss, log_dict, batch_size = self.policy_training_step(*args, **kwargs)
        log_dict = {f"train/{k}": v for k, v in log_dict.items()}
        log_dict["train/loss"] = loss
        self.log_dict(
            log_dict,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        return loss

    def validation_step(self, *args, **kwargs):
        loss, log_dict, real_batch_size = self.policy_evaluation_step(*args, **kwargs)
        log_dict = {f"val/{k}": v for k, v in log_dict.items()}
        log_dict["val/loss"] = loss
        self.log_dict(
            log_dict,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=real_batch_size,
            sync_dist=True,
        )
        return log_dict

    def test_step(self, *args, **kwargs):
        logger.info("Skipping test step.")

    def on_validation_epoch_end(self):
        # only run test for global zero rank
        if self.trainer.is_global_zero:
            if self.online_eval_config is not None:
                # evaluator for online evaluation should only be created once
                if self.evaluator is None:
                    self.evaluator = self.create_evaluator()
                if not self.trainer.sanity_checking:
                    self.log_dict(self.run_online_evaluation())
        # Synchronize all processes to prevent timeout
        if dist.is_initialized():
            dist.barrier()

    def create_evaluator(self):
        """
        Create a evaluator parameter config containing vectorized distributed envs.
        This will be used to spawn the OmniGibson environments for online evaluation
        """
        # For performance optimization
        gm.DEFAULT_VIEWER_WIDTH = 128
        gm.DEFAULT_VIEWER_HEIGHT = 128
        gm.HEADLESS = self.online_eval_config.cfg.headless

        # update parameters with policy cfg file
        assert self.online_eval_config is not None, "online_eval_config must be provided to create evaluator!"
        evaluator = instantiate(self.online_eval_config, _recursive_=False)
        # instantiate policy wrapper and set the policy 
        policy_wrapper = instantiate(self.policy_wrapper_config)
        policy_wrapper.policy = self
        evaluator.policy.policy = policy_wrapper
        return evaluator

    def run_online_evaluation(self):
        """
        Run online evaluation using the evaluator.
        """
        assert self.evaluator is not None, "evaluator is not created!"
        self.evaluator.reset()
        self.evaluator.env._current_episode = 0
        if self.online_eval_config.cfg.write_video:
            video_name = f"videos/test_{self.test_id}.mp4"
            os.makedirs("videos", exist_ok=True)
            self.evaluator.video_writer = create_video_writer(
                fpath=video_name,
                resolution=(224, 448),
            )
        done = False
        while not done:
            terminated, truncated = self.evaluator.step()
            if self.online_eval_config.cfg.write_video:
                self.evaluator._write_video()
            if terminated:
                self.evaluator.env.reset()
            if truncated:
                done = True
        if self.online_eval_config.cfg.write_video:
            self.evaluator.video_writer = None
        self.test_id += 1
        results = {"eval/success_rate": self.evaluator.n_success_trials / self.evaluator.n_trials}
        return results
    
    def _denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """
        Denormalize the action from [-1, 1] to [min, max] range.
        Also, rectify gripper actions to either -1 or 1.
        Args:
            action: (B, L, A) where A is the action dimension
        Returns:
            unnormalized_action: (B, L, A)
        """
        # rectify gripper actions
        for k, v in ACTION_QPOS_INDICES[self.robot_type].items():
            if "gripper" in k:
                action[..., v] = torch.where(action[..., v] > 0, 1.0, -1.0)
            else:
                action[..., v] = (action[..., v] + 1) / 2 * (
                    JOINT_RANGE[self.robot_type][k][1] - JOINT_RANGE[self.robot_type][k][0]
                ) + JOINT_RANGE[self.robot_type][k][0]
        return action
    

class PolicyWrapper:
    """
    A Wrapper for handling policy observations and actions
    """

    def __init__(
        self,
        *args,
        # ====== policy model ======
        deployed_action_steps: int,
        obs_window_size: int = 1,
        multi_view_cameras: Dict[str, Any],
        visual_obs_types: List[str],
        use_task_info: bool = False,
        task_info_range: Optional[ListConfig] = None,
        pcd_range: Optional[List[float]] = None,
        robot_type: str = "R1Pro",
        # ====== other args for base class ======
        **kwargs,
    ) -> None:
        self.policy = None # to be filled
        self.robot_type = robot_type
        # move all tensor to self.device
        self._post_processing_fn = lambda x: x.to(self.policy.device)
        assert set(visual_obs_types).issubset(
            {"rgb", "depth_linear", "seg_instance_id", "pcd"}
        ), "visual_obs_types must be a subset of {'rgb', 'depth_linear', 'seg_instance_id', 'pcd'}!"
        self.visual_obs_types = visual_obs_types
        self._use_task_info = use_task_info
        self._task_info_range = (
            torch.tensor(OmegaConf.to_container(task_info_range)) if task_info_range is not None else None
        )
        if "pcd" in visual_obs_types:
            # store camera intrinsics
            self.camera_intrinsics = dict()
            for camera_id, camera_name in ROBOT_CAMERA_NAMES[self.robot_type].items():
                scale_factor = 3.0 if camera_id == "head" else 2.0
                camera_intrinsics = torch.from_numpy(CAMERA_INTRINSICS[self.robot_type][camera_id]) / scale_factor
                camera_intrinsics[-1, -1] = 1.0  # make it homogeneous
                self.camera_intrinsics[camera_name] = camera_intrinsics
        self._pcd_range = tuple(pcd_range) if pcd_range is not None else None
        # action steps for deployed policy
        self.deployed_action_steps = deployed_action_steps
        self.obs_window_size = obs_window_size
        self.obs_output_size = {k: tuple(v["resolution"]) for k, v in multi_view_cameras.items()}
        self._obs_history = deque(maxlen=obs_window_size)
        self._action_traj_pred = None
        self._action_idx = 0
        self._robot_name = None
        self.joint_range = JOINT_RANGE[self.robot_type]

    def act(self, obs: dict, *args, **kwargs) -> torch.Tensor:
        obs = any_to_torch(obs, device="cpu")
        obs = self.process_obs(obs=obs)
        if len(self._obs_history) == 0:
            for _ in range(self.obs_window_size):
                self._obs_history.append(obs)
        else:
            self._obs_history.append(obs)
        obs = any_concat(self._obs_history, dim=1)

        need_inference = self._action_idx % self.deployed_action_steps == 0
        if need_inference:
            self._action_traj_pred = self.policy.act({"obs": obs}).squeeze(0)  # (T_A, A)
            self._action_idx = 0
        action = self._action_traj_pred[self._action_idx]
        self._action_idx += 1
        return action

    def reset(self) -> None:
        if self.policy is not None:
            self.policy.reset()
        self._obs_history = deque(maxlen=self.obs_window_size)
        self._action_traj_pred = None
        self._action_idx = 0

    def process_obs(self, obs: dict) -> dict:
        # Expand twice to get B and T_A dimensions
        processed_obs = {"qpos": dict()}
        if self._robot_name is None:
            for key in obs:
                if "proprio" in key:
                    self._robot_name = key.split("::")[0]
                    break
        proprio = obs[f"{self._robot_name}::proprio"].unsqueeze(0).unsqueeze(0)
        if "base_qvel" in PROPRIOCEPTION_INDICES[self.robot_type]:
            processed_obs["odom"] = {
                "base_velocity": self._post_processing_fn(
                    2
                    * (proprio[..., PROPRIOCEPTION_INDICES[self.robot_type]["base_qvel"]] - self.joint_range["base"][0])
                    / (self.joint_range["base"][1] - self.joint_range["base"][0])
                    - 1
                ),
            }
        for key in PROPRIO_QPOS_INDICES[self.robot_type]:
            if "gripper" in key:
                # rectify gripper actions to {-1, 1}
                processed_obs["qpos"][key] = torch.mean(
                    proprio[..., PROPRIO_QPOS_INDICES[self.robot_type][key]], dim=-1, keepdim=True
                )
                processed_obs["qpos"][key] = self._post_processing_fn(
                    torch.where(
                        processed_obs["qpos"][key]
                        > (JOINT_RANGE[self.robot_type][key][0] + JOINT_RANGE[self.robot_type][key][1]) * 0.8,
                        1.0,
                        -1.0,
                    )
                )
            else:
                # normalize the qpos to [-1, 1]
                processed_obs["qpos"][key] = self._post_processing_fn(
                    2
                    * (proprio[..., PROPRIO_QPOS_INDICES[self.robot_type][key]] - JOINT_RANGE[self.robot_type][key][0])
                    / (JOINT_RANGE[self.robot_type][key][1] - JOINT_RANGE[self.robot_type][key][0])
                    - 1.0
                )
        if self.robot_type in EEF_POSITION_RANGE:
            processed_obs["eef"] = dict()
            for key in EEF_POSITION_RANGE[self.robot_type]:
                processed_obs["eef"][f"{key}_pos"] = self._post_processing_fn(
                    2
                    * (
                        proprio[..., PROPRIOCEPTION_INDICES[self.robot_type][f"eef_{key}_pos"]]
                        - EEF_POSITION_RANGE[self.robot_type][key][0]
                    )
                    / (EEF_POSITION_RANGE[self.robot_type][key][1] - EEF_POSITION_RANGE[self.robot_type][key][0])
                    - 1.0
                )
                # don't normalize the eef orientation
                processed_obs["eef"][f"{key}_quat"] = self._post_processing_fn(
                    proprio[..., PROPRIOCEPTION_INDICES[self.robot_type][f"eef_{key}_quat"]]
                )
        if "pcd" in self.visual_obs_types:
            pcd_obs = dict()
        for camera_id, camera in ROBOT_CAMERA_NAMES[self.robot_type].items():
            if "rgb" in self.visual_obs_types or "pcd" in self.visual_obs_types:
                rgb_obs = F.interpolate(
                    obs[f"{camera}::rgb"][..., :3].unsqueeze(0).movedim(-1, -3).to(torch.float32),
                    self.obs_output_size[camera_id],
                    mode="nearest-exact",
                ).unsqueeze(0)
                if "pcd" in self.visual_obs_types:
                    # move rgb dim back
                    pcd_obs[f"{camera}::rgb"] = rgb_obs.movedim(-3, -1).to(self.policy.device)
                else:
                    processed_obs[f"{camera}::rgb"] = self._post_processing_fn(rgb_obs)
            if "depth_linear" in self.visual_obs_types or "pcd" in self.visual_obs_types:
                depth_obs = F.interpolate(
                    obs[f"{camera}::depth_linear"].unsqueeze(0).unsqueeze(0).to(torch.float32),
                    self.obs_output_size[camera_id],
                    mode="nearest-exact",
                )
                # clamp depth to [MIN_DEPTH, MAX_DEPTH]
                depth_obs = torch.clamp(depth_obs, MIN_DEPTH, MAX_DEPTH)
                if "pcd" in self.visual_obs_types:
                    pcd_obs[f"{camera}::depth_linear"] = depth_obs.to(self.policy.device)
                else:
                    processed_obs[f"{camera}::depth_linear"] = self._post_processing_fn(depth_obs)
            if "seg_instance_id" in self.visual_obs_types:
                processed_obs[f"{camera}::seg_instance_id"] = self._post_processing_fn(
                    F.interpolate(
                        obs[f"{camera}::seg_instance_id"].unsqueeze(0).unsqueeze(0).to(torch.float32),
                        self.obs_output_size[camera_id],
                        mode="nearest-exact",
                    )
                )
        if "pcd" in self.visual_obs_types:
            pcd_obs["cam_rel_poses"] = (
                obs["robot_r1::cam_rel_poses"].unsqueeze(0).unsqueeze(0).to(torch.float32).to(self.policy.device)
            )
            processed_obs["pcd"] = self._post_processing_fn(
                process_fused_point_cloud(
                    obs=pcd_obs,
                    camera_intrinsics=self.camera_intrinsics,
                    pcd_range=self._pcd_range,
                    pcd_num_points=4096,
                    use_fps=True,
                )
            )
        if self._use_task_info:
            for key in obs:
                if key.startswith("task::"):
                    if self._task_info_range is not None:
                        # Normalize task info to [-1, 1]
                        processed_obs["task"] = (
                            self._post_processing_fn(
                                2
                                * (obs[key] - self._task_info_range[0])
                                / (self._task_info_range[1] - self._task_info_range[0])
                                - 1.0
                            )
                            .unsqueeze(0)
                            .unsqueeze(0)
                            .to(torch.float32)
                        )
                    else:
                        # If no range is provided, just use the raw data
                        processed_obs["task"] = self._post_processing_fn(
                            obs[key].unsqueeze(0).unsqueeze(0).to(torch.float32)
                        )
                    break
        return processed_obs


class ResidualPolicyWrapper(PolicyWrapper):
    """
    A specialized wrapper for ResidualPolicy that manages both base policy and residual policy
    with different action execution frequencies.
    
    Base policy: Predicts action chunks (e.g., 16 actions every 16 steps)
    Residual policy: Predicts per-step corrections (1 correction every step)
    """

    def __init__(
        self,
        *args,
        base_deployed_action_steps: int,  # Base policy's action chunk size
        residual_deployed_action_steps: int = 1,  # Residual policy's action step (usually 1)
        action_history_len: int = 1,
        gate_threshold: float = 0.5,
        residual_scale: float = 1.0,
        **kwargs,
    ) -> None:
        # Remove deployed_action_steps from kwargs if present to avoid duplicate
        kwargs.pop("deployed_action_steps", None)
        # Initialize parent with residual policy's deployment frequency
        super().__init__(*args, deployed_action_steps=residual_deployed_action_steps, **kwargs)
        self._action_history_len = action_history_len
        self._base_action_history = deque(maxlen=action_history_len)
        self._gate_threshold = gate_threshold
        self._residual_scale = residual_scale
        self._oracle_blend_alpha = kwargs.pop("oracle_blend_alpha", 1.0)
        
        # Base policy specific attributes
        self.base_deployed_action_steps = base_deployed_action_steps
        self._base_action_buffer = None  # Will store (T_A, A) from base policy
        self._base_action_idx = 0
        self.base_policy = None  # Will be set to the base policy from residual_policy.base_policy
        self._base_obs_history = None  # Separate obs history for base policy
        self._base_obs_window_size = None
        # Per-step stats tracking for eval reporting
        self._stats = {
            "interventions": 0,
            "total_steps": 0,
            "residual_mags": [],
            "gate_decisions": [],       # 1=intervened, 0=passthrough
            "residual_norms": [],       # L2 norm of residual per step
            "base_action_norms": [],    # L2 norm of base action per step
        }
        # Per-step action vectors for detailed analysis
        self._episode_traces = []   # list of per-episode dicts
        self._current_episode = None
    
    def act(self, obs: dict, *args, **kwargs) -> torch.Tensor:
        """
        Coordinated action generation:
        1. Get base action from buffer (refresh every base_deployed_action_steps)
        2. Get residual correction from residual policy (every step)
        3. Combine: final_action = base_action + residual_correction
        """
        obs = any_to_torch(obs, device="cpu")
        obs = self.process_obs(obs=obs)
        
        # Maintain observation history
        if len(self._obs_history) == 0:
            for _ in range(self.obs_window_size):
                self._obs_history.append(obs)
        else:
            self._obs_history.append(obs)
        obs_stacked = any_concat(self._obs_history, dim=1)  # (B=1, T_obs, ...)

        # ===== Base Policy: Action Chunking =====
        if self.base_policy is None and hasattr(self.policy, 'base_policy'):
            self.base_policy = self.policy.base_policy
        if self.base_policy is None:
            raise ValueError("base_policy not found in residual policy!")

        # Maintain separate obs history for base policy (may need different window)
        if self._base_obs_history is None:
            self._base_obs_window_size = getattr(self.base_policy, '_num_latest_obs', 2)
            if hasattr(self.base_policy, 'num_latest_obs'):
                self._base_obs_window_size = self.base_policy.num_latest_obs
            self._base_obs_history = deque(maxlen=self._base_obs_window_size)
        if len(self._base_obs_history) == 0:
            for _ in range(self._base_obs_window_size):
                self._base_obs_history.append(obs)
        else:
            self._base_obs_history.append(obs)

        need_base_inference = self._base_action_idx % self.base_deployed_action_steps == 0
        if need_base_inference:
            base_obs_stacked = any_concat(self._base_obs_history, dim=1)
            self._base_action_buffer = self.base_policy.act({"obs": base_obs_stacked}).squeeze(0)
            self._base_action_idx = 0
            if hasattr(self.base_policy, '_last_obs_embedding'):
                self._base_obs_embedding = self.base_policy._last_obs_embedding
        
        # Get current base action from buffer (raw radians, denormalized by base policy)
        base_action = self._base_action_buffer[self._base_action_idx]  # (A,)
        self._base_action_idx += 1

        # Normalize base action back to [-1, 1] so it matches the residual's training space
        base_action_normalized = self._normalize_action(base_action.clone())

        # Maintain action history for temporal context
        self._base_action_history.append(base_action_normalized.clone())
        while len(self._base_action_history) < self._action_history_len:
            self._base_action_history.appendleft(self._base_action_history[0])

        # ===== Residual Policy: Per-step Correction =====
        if self._action_history_len > 1:
            action_hist_flat = torch.cat(list(self._base_action_history), dim=-1)
            obs_for_residual = {"obs": obs_stacked, "base_action": action_hist_flat.unsqueeze(0).unsqueeze(0)}
        else:
            obs_for_residual = {"obs": obs_stacked, "base_action": base_action_normalized.unsqueeze(0).unsqueeze(0)}
        if hasattr(self, '_base_obs_embedding') and self._base_obs_embedding is not None:
            obs_for_residual["base_embedding"] = self._base_obs_embedding

        # Get residual correction (intervention decision + action correction)
        result = self.policy.act(obs_for_residual)
        if isinstance(result, tuple) and len(result) == 2:
            residual_action = result[0].squeeze().cpu()
            intervention = result[1].squeeze().cpu()
        else:
            residual_action = result.squeeze().cpu()
            intervention = torch.tensor(1.0)

        intervened = intervention >= self._gate_threshold
        if intervened:
            predict_oracle = getattr(self.policy, '_predict_oracle_action', False)
            if predict_oracle:
                # Use MLP arm prediction but keep base gripper
                oracle_pred = residual_action.clone()
                oracle_pred[..., -1] = base_action_normalized[..., -1]
                # Blend: mix oracle prediction with base using oracle_blend_alpha
                alpha = self._oracle_blend_alpha
                blended = alpha * oracle_pred + (1 - alpha) * base_action_normalized
                blended[..., -1] = base_action_normalized[..., -1]  # always keep base gripper
                final_action = self._denormalize_action(blended)
            else:
                scaled_residual = residual_action * self._residual_scale
                combined_normalized = base_action_normalized + scaled_residual
                final_action = self._denormalize_action(combined_normalized)
        else:
            final_action = base_action

        # Track per-step stats
        self._stats["total_steps"] += 1
        if intervened:
            self._stats["interventions"] += 1
        self._stats["gate_decisions"].append(1 if intervened else 0)
        self._stats["residual_mags"].append(float(residual_action.abs().mean()))
        self._stats["residual_norms"].append(float(residual_action.norm()))
        self._stats["base_action_norms"].append(float(base_action_normalized.norm()))

        # Log per-step action vectors
        if self._current_episode is None:
            self._current_episode = {"base": [], "residual": [], "final": [], "gate": []}
        self._current_episode["base"].append(base_action.detach().cpu().numpy().tolist())
        self._current_episode["residual"].append(residual_action.detach().cpu().numpy().tolist())
        self._current_episode["final"].append(final_action.detach().cpu().numpy().tolist())
        self._current_episode["gate"].append(1 if intervened else 0)

        return final_action

    def _normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Normalize action from raw joint space to [-1, 1]."""
        for k, v in ACTION_QPOS_INDICES[self.robot_type].items():
            if "gripper" not in k:
                action[..., v] = (
                    2 * (action[..., v] - JOINT_RANGE[self.robot_type][k][0])
                    / (JOINT_RANGE[self.robot_type][k][1] - JOINT_RANGE[self.robot_type][k][0])
                    - 1.0
                )
        return action

    def _denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Denormalize action from [-1, 1] to raw joint space."""
        for k, v in ACTION_QPOS_INDICES[self.robot_type].items():
            if "gripper" in k:
                action[..., v] = torch.where(action[..., v] > 0, 1.0, -1.0)
            else:
                action[..., v] = (action[..., v] + 1) / 2 * (
                    JOINT_RANGE[self.robot_type][k][1] - JOINT_RANGE[self.robot_type][k][0]
                ) + JOINT_RANGE[self.robot_type][k][0]
        return action

    def get_eval_stats(self):
        """Return summary stats from inference run."""
        import numpy as np
        total = self._stats["total_steps"]
        ints = self._stats["interventions"]
        mags = np.array(self._stats["residual_mags"]) if self._stats["residual_mags"] else np.array([0.0])
        norms = np.array(self._stats["residual_norms"]) if self._stats["residual_norms"] else np.array([0.0])
        gates = np.array(self._stats["gate_decisions"]) if self._stats["gate_decisions"] else np.array([0])
        base_norms = np.array(self._stats["base_action_norms"]) if self._stats["base_action_norms"] else np.array([0.0])

        # Split residual stats by gate decision
        intervened_mask = gates == 1
        passthrough_mask = gates == 0

        stats = {
            "total_steps": total,
            "intervention_steps": ints,
            "passthrough_steps": total - ints,
            "intervention_rate": float(ints / max(total, 1)),
            # Residual magnitude (all steps)
            "residual_mag_mean": float(mags.mean()),
            "residual_mag_std": float(mags.std()),
            "residual_mag_p50": float(np.median(mags)),
            "residual_mag_p95": float(np.percentile(mags, 95)),
            "residual_mag_max": float(mags.max()),
            # Residual L2 norm
            "residual_norm_mean": float(norms.mean()),
            # Base action norm (for scale reference)
            "base_action_norm_mean": float(base_norms.mean()),
            # Residual as % of base action
            "residual_pct_of_base": float(norms.mean() / max(base_norms.mean(), 1e-8) * 100),
        }

        # Stats split by gate decision
        if intervened_mask.sum() > 0:
            stats["residual_mag_when_intervened"] = float(mags[intervened_mask].mean())
            stats["residual_norm_when_intervened"] = float(norms[intervened_mask].mean())
        if passthrough_mask.sum() > 0:
            stats["residual_mag_when_passthrough"] = float(mags[passthrough_mask].mean())
            stats["residual_norm_when_passthrough"] = float(norms[passthrough_mask].mean())

        return stats

    def reset(self) -> None:
        """Reset both policies and their states"""
        if self._current_episode is not None and len(self._current_episode.get("base", [])) > 0:
            self._episode_traces.append(self._current_episode)
        self._current_episode = None
        self._base_action_history = deque(maxlen=self._action_history_len)
        super().reset()
        self._base_action_buffer = None
        self._base_action_idx = 0
        self._base_obs_history = None
        if self.base_policy is not None:
            self.base_policy.reset()

    def save_traces(self, path: str):
        """Save per-episode action traces to JSON for analysis."""
        import json
        if self._current_episode is not None and len(self._current_episode.get("base", [])) > 0:
            self._episode_traces.append(self._current_episode)
            self._current_episode = None
        with open(path, "w") as f:
            json.dump(self._episode_traces, f)
        print(f"[ResidualPolicyWrapper] Saved {len(self._episode_traces)} episode traces to {path}")