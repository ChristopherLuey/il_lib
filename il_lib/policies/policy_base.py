import json
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
        self._cached_obs_embedding = None

    @property
    def last_obs_embedding(self):
        if self._cached_obs_embedding is None:
            return None
        return self._cached_obs_embedding.detach().cpu().numpy()

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
            self._cached_obs_embedding = getattr(self.policy, '_cached_obs_feature', None)
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
        self._cached_obs_embedding = None

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
        intervention_blend_mode: str = "binary",
        residual_horizon: int = 0,  # >0 activates chunked residual inference
        **kwargs,
    ) -> None:
        # base_config injects policy_wrapper.deployed_action_steps by default.
        # ResidualPolicyWrapper manages its own deployment cadence, so drop the inherited
        # field to avoid passing deployed_action_steps twice into PolicyWrapper.
        kwargs.pop("deployed_action_steps", None)
        # In chunked mode, deploy at the base cadence; otherwise per-step.
        effective_deployed = base_deployed_action_steps if residual_horizon > 0 else residual_deployed_action_steps
        super().__init__(*args, deployed_action_steps=effective_deployed, **kwargs)

        # Base policy specific attributes
        self.base_deployed_action_steps = base_deployed_action_steps
        self.intervention_blend_mode = intervention_blend_mode
        self._base_action_buffer = None  # Will store (T_A, A) from base policy
        self._base_action_idx = 0
        self.base_policy = None  # Will be set to the base policy from residual_policy.base_policy
        self.base_obs_window_size = 1
        self._base_obs_history = deque(maxlen=self.base_obs_window_size)
        self._trace_path = os.environ.get("IIIL_RESIDUAL_TRACE_PATH")
        self._trace_step_idx = 0
        self._residual_horizon = residual_horizon
        self._combined_action_buffer = None
        self._combined_action_idx = 0
        if self._trace_path:
            os.makedirs(os.path.dirname(self._trace_path), exist_ok=True)
            # Start each traced rollout with a clean file.
            with open(self._trace_path, "w"):
                pass
    
    def act(self, obs: dict, *args, **kwargs) -> torch.Tensor:
        if self._residual_horizon > 0:
            return self._act_chunked(obs, *args, **kwargs)
        return self._act_per_step(obs, *args, **kwargs)

    def _act_per_step(self, obs: dict, *args, **kwargs) -> torch.Tensor:
        """
        Per-step residual: base predicts chunk, residual corrects each step.
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
        need_base_inference = self._base_action_idx % self.base_deployed_action_steps == 0
        if need_base_inference:
            # Get base policy from residual policy
            if self.base_policy is None and hasattr(self.policy, 'base_policy'):
                self.base_policy = self.policy.base_policy
                self.base_obs_window_size = getattr(self.base_policy, "num_latest_obs", 1)
                self._base_obs_history = deque(maxlen=self.base_obs_window_size)

            if len(self._base_obs_history) == 0:
                for _ in range(self.base_obs_window_size):
                    self._base_obs_history.append(obs)
            else:
                self._base_obs_history.append(obs)

            if self.base_policy is not None:
                # Base policy predicts action chunk
                # DiffusionPolicy.act expects the runtime payload under the "obs" key.
                base_obs_stacked = any_concat(self._base_obs_history, dim=1)
                self._base_action_buffer = self.base_policy.act({"obs": base_obs_stacked}).squeeze(0)  # (T_A, A)
                self._base_action_idx = 0
            else:
                raise ValueError("base_policy not found in residual policy!")

        # Get current base action from buffer (raw radians, denormalized by base policy)
        base_action = self._base_action_buffer[self._base_action_idx]  # (A,)
        self._base_action_idx += 1

        # Keep the deployment interface CPU-facing, but run residual math on the residual policy device.
        base_action_device = base_action.to(self.policy.device)
        # Normalize base action back to [-1, 1] so it matches the residual's training space
        base_action_normalized = self._normalize_action(base_action_device.clone())

        # ===== Residual Policy: Per-step Correction =====
        # Match the exact feature structure ResidualPolicy.process_data() builds at training time.
        obs_for_residual = self._build_residual_obs(obs_stacked, base_action_normalized)

        # Get residual correction plus intervention state / weight.
        if hasattr(self.policy, "act_bundle"):
            residual_action, intervention, intervention_weight = self.policy.act_bundle(obs_for_residual)
        else:
            residual_action, intervention = self.policy.act(obs_for_residual)
            intervention_weight = intervention.to(dtype=base_action_normalized.dtype)
        residual_action = residual_action.squeeze()  # (A,)
        intervention = intervention.squeeze()  # scalar
        intervention_weight = intervention_weight.squeeze()

        # ===== Combine Actions =====
        if self.intervention_blend_mode == "always_on":
            blend_weight = torch.ones_like(
                intervention_weight,
                device=base_action_normalized.device,
                dtype=base_action_normalized.dtype,
            )
        elif self.intervention_blend_mode == "weighted":
            blend_weight = intervention_weight.to(
                device=base_action_normalized.device,
                dtype=base_action_normalized.dtype,
            )
        else:
            blend_weight = (intervention >= 0.5).to(
                device=base_action_normalized.device,
                dtype=base_action_normalized.dtype,
            )

        combined_normalized = (
            base_action_normalized
            + blend_weight * residual_action.to(base_action_normalized.device)
        )
        final_action = self._denormalize_action(combined_normalized).cpu()

        if self._trace_path:
            trace_row = {
                "step": self._trace_step_idx,
                "intervention": float(intervention.detach().cpu().item()),
                "intervention_weight": float(intervention_weight.detach().cpu().item()),
                "blend_weight": float(blend_weight.detach().cpu().item()),
                "intervention_blend_mode": self.intervention_blend_mode,
                "base_action": base_action.detach().cpu().view(-1).tolist(),
                "final_action": final_action.detach().cpu().view(-1).tolist(),
                "residual_action_normalized": residual_action.detach().cpu().view(-1).tolist(),
                "base_action_normalized": base_action_normalized.detach().cpu().view(-1).tolist(),
            }
            with open(self._trace_path, "a") as f:
                f.write(json.dumps(trace_row) + "\n")
            self._trace_step_idx += 1

        return final_action

    def _act_chunked(self, obs: dict, *args, **kwargs) -> torch.Tensor:
        """
        Chunked residual: both base and residual predict action chunks at the
        same cadence.  Every base_deployed_action_steps steps we run both
        models, element-wise add their outputs, and buffer the result.
        """
        obs = any_to_torch(obs, device="cpu")
        obs = self.process_obs(obs=obs)

        if len(self._obs_history) == 0:
            for _ in range(self.obs_window_size):
                self._obs_history.append(obs)
        else:
            self._obs_history.append(obs)
        obs_stacked = any_concat(self._obs_history, dim=1)  # (1, T_obs, ...)

        need_inference = (
            self._combined_action_buffer is None
            or self._combined_action_idx >= self.base_deployed_action_steps
        )
        if need_inference:
            # --- init base policy ref ---
            if self.base_policy is None and hasattr(self.policy, "base_policy"):
                self.base_policy = self.policy.base_policy
                self.base_obs_window_size = getattr(self.base_policy, "num_latest_obs", 1)
                self._base_obs_history = deque(maxlen=self.base_obs_window_size)

            if len(self._base_obs_history) == 0:
                for _ in range(self.base_obs_window_size):
                    self._base_obs_history.append(obs)
            else:
                self._base_obs_history.append(obs)

            # --- base policy chunk ---
            base_obs_stacked = any_concat(self._base_obs_history, dim=1)
            base_chunk_raw = self.base_policy.act({"obs": base_obs_stacked}).squeeze(0)  # (T_A, A)
            T_A = base_chunk_raw.shape[0]

            # Pad to residual_horizon if base returns fewer (e.g. 15 from horizon=16, obs=2)
            H = self._residual_horizon
            if T_A < H:
                pad = base_chunk_raw[-1:].expand(H - T_A, -1)
                base_chunk_raw = torch.cat([base_chunk_raw, pad], dim=0)
            base_chunk_raw = base_chunk_raw[:H]  # (H, A)

            # Normalize full chunk to [-1, 1]
            base_chunk_norm = self._normalize_action(
                base_chunk_raw.to(self.policy.device).clone()
            )  # (H, A)

            # --- residual policy chunk ---
            obs_for_residual = self._build_residual_obs_chunked(obs_stacked, base_chunk_norm)

            if hasattr(self.policy, "act_bundle"):
                residual_chunk, intervention, intervention_weight = self.policy.act_bundle(obs_for_residual)
            else:
                residual_chunk, intervention = self.policy.act(obs_for_residual)
                intervention_weight = intervention.to(dtype=base_chunk_norm.dtype)
            residual_chunk = residual_chunk.squeeze(0)  # (H, A)

            # --- blend ---
            if self.intervention_blend_mode == "always_on":
                blend = 1.0
            elif self.intervention_blend_mode == "weighted":
                blend = intervention_weight.squeeze(0).unsqueeze(-1).to(
                    device=base_chunk_norm.device, dtype=base_chunk_norm.dtype
                )
            else:
                blend = (intervention.squeeze(0) >= 0.5).unsqueeze(-1).to(
                    device=base_chunk_norm.device, dtype=base_chunk_norm.dtype
                )

            combined_norm = base_chunk_norm + blend * residual_chunk.to(base_chunk_norm.device)

            # Denormalize each step and move to cpu
            combined_raw = torch.stack([
                self._denormalize_action(combined_norm[i].clone())
                for i in range(combined_norm.shape[0])
            ]).cpu()  # (H, A)

            self._combined_action_buffer = combined_raw
            self._combined_action_idx = 0

            # Store for tracing
            self._base_chunk_raw_for_trace = base_chunk_raw.detach().cpu()
            self._base_chunk_norm_for_trace = base_chunk_norm.detach().cpu()
            self._residual_chunk_for_trace = residual_chunk.detach().cpu()

        # Pop next action from buffer
        action = self._combined_action_buffer[self._combined_action_idx]
        step_idx = self._combined_action_idx
        self._combined_action_idx += 1

        if self._trace_path:
            trace_row = {
                "step": self._trace_step_idx,
                "intervention": 1,
                "intervention_weight": 1.0,
                "blend_weight": 1.0,
                "intervention_blend_mode": self.intervention_blend_mode,
                "base_action": self._base_chunk_raw_for_trace[step_idx].view(-1).tolist(),
                "final_action": action.detach().view(-1).tolist(),
                "residual_action_normalized": self._residual_chunk_for_trace[step_idx].view(-1).tolist(),
                "base_action_normalized": self._base_chunk_norm_for_trace[step_idx].view(-1).tolist(),
            }
            with open(self._trace_path, "a") as f:
                f.write(json.dumps(trace_row) + "\n")
            self._trace_step_idx += 1

        return action

    def _build_residual_obs_chunked(
        self, obs_stacked: dict, base_chunk_norm: torch.Tensor
    ) -> dict:
        """Build obs dict for chunked residual: obs at T_obs, base_action at H."""
        obs_for_residual = {
            "qpos": obs_stacked["qpos"],
            "base_action": base_chunk_norm.unsqueeze(0),  # (1, H, A)
        }
        if "eef" in obs_stacked:
            obs_for_residual["eef"] = obs_stacked["eef"]
        if "odom" in obs_stacked:
            obs_for_residual["odom"] = obs_stacked["odom"]
        if hasattr(self.policy, "_features") and "rgb" in self.policy._features:
            obs_for_residual["rgb"] = {
                k.rsplit("::", 1)[0]: obs_stacked[k].float() / 255.0
                for k in obs_stacked
                if "rgb" in k
            }
        if hasattr(self.policy, "_features") and "task" in self.policy._features and "task" in obs_stacked:
            obs_for_residual["task"] = obs_stacked["task"]
        return any_to_torch(obs_for_residual, device=self.policy.device)

    def _build_residual_obs(self, obs_stacked: dict, base_action_normalized: torch.Tensor) -> dict:
        # base_action is (A,); expand to (1, T, A) to match obs time dimension
        T = getattr(self.policy, "num_latest_obs", 1)
        base_action_obs = base_action_normalized.unsqueeze(0).unsqueeze(0).expand(-1, T, -1)
        obs_for_residual = {
            "qpos": obs_stacked["qpos"],
            "base_action": base_action_obs,
        }
        if "eef" in obs_stacked:
            obs_for_residual["eef"] = obs_stacked["eef"]
        if "odom" in obs_stacked:
            obs_for_residual["odom"] = obs_stacked["odom"]
        if hasattr(self.policy, "_features") and "rgb" in self.policy._features:
            obs_for_residual["rgb"] = {
                k.rsplit("::", 1)[0]: obs_stacked[k].float() / 255.0
                for k in obs_stacked
                if "rgb" in k
            }
        if hasattr(self.policy, "_features") and "task" in self.policy._features and "task" in obs_stacked:
            obs_for_residual["task"] = obs_stacked["task"]
        return any_to_torch(obs_for_residual, device=self.policy.device)

    def _normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Normalize action from raw joint space to [-1, 1]."""
        for k, v in ACTION_QPOS_INDICES[self.robot_type].items():
            if "gripper" not in k:
                lower = torch.as_tensor(
                    JOINT_RANGE[self.robot_type][k][0],
                    device=action.device,
                    dtype=action.dtype,
                )
                upper = torch.as_tensor(
                    JOINT_RANGE[self.robot_type][k][1],
                    device=action.device,
                    dtype=action.dtype,
                )
                action[..., v] = (
                    2 * (action[..., v] - lower)
                    / (upper - lower)
                    - 1.0
                )
        return action

    def _denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Denormalize action from [-1, 1] to raw joint space."""
        for k, v in ACTION_QPOS_INDICES[self.robot_type].items():
            if "gripper" in k:
                action[..., v] = torch.where(action[..., v] > 0, 1.0, -1.0)
            else:
                lower = torch.as_tensor(
                    JOINT_RANGE[self.robot_type][k][0],
                    device=action.device,
                    dtype=action.dtype,
                )
                upper = torch.as_tensor(
                    JOINT_RANGE[self.robot_type][k][1],
                    device=action.device,
                    dtype=action.dtype,
                )
                action[..., v] = (action[..., v] + 1) / 2 * (upper - lower) + lower
        return action

    def reset(self) -> None:
        """Reset both policies and their states"""
        super().reset()
        self._base_action_buffer = None
        self._base_action_idx = 0
        self._combined_action_buffer = None
        self._combined_action_idx = 0
        self.base_obs_window_size = 1
        self._base_obs_history = deque(maxlen=self.base_obs_window_size)
        self._trace_step_idx = 0
        if self.base_policy is not None:
            self.base_policy.reset()
        self.base_policy = None
