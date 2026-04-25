"""
Simple FCN Residual Policy — MSE regression, no GMM, optional intervention head.

Architecture:
    [proprioception, task_info, base_action] → concat → MLP → residual_action (7D)

Design rationale:
- With only ~20 correction episodes, a simple MLP with MSE loss is sufficient
- No multimodality to capture → GMM is overkill
- Zero-residual supervision on non-intervention timesteps prevents jitter
- Optional intervention head can be disabled (always_on mode)
"""

import os
import sys
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from il_lib.policies.policy_base import BasePolicy
from il_lib.nn.features import SimpleFeatureFusion
from il_lib.nn.distributions import CategoricalNet
from il_lib.optim import CosineScheduleFunction
from il_lib.utils.training_utils import freeze_params, load_state_dict, load_torch
from il_lib.utils.config_utils import register_omegaconf_resolvers
from omegaconf import DictConfig, OmegaConf
from typing import Any, Dict, List, Optional


class SimpleResidualPolicy(BasePolicy):
    """
    Simple FCN residual policy with MSE loss.
    
    Outputs a deterministic residual action via MLP regression.
    Optionally includes an intervention head for gating.
    """

    def __init__(
        self,
        *args,
        prop_dim: int,
        prop_keys: List[str],
        # ====== Feature Extractors ======
        feature_extractors: Dict[str, DictConfig],
        feature_fusion_hidden_depth: int = 1,
        feature_fusion_hidden_dim: int = 256,
        feature_fusion_output_dim: int = 256,
        feature_fusion_activation: str = "relu",
        feature_fusion_add_input_activation: bool = False,
        feature_fusion_add_output_activation: bool = False,
        # ====== Action ======
        action_dim: int = 7,
        action_net_hidden_dim: int = 128,
        action_net_hidden_depth: int = 3,
        action_net_activation: str = "relu",
        # ====== Intervention ======
        use_intervention_head: bool = False,
        intervention_head_hidden_dim: int = 128,
        intervention_head_hidden_depth: int = 3,
        intervention_head_activation: str = "relu",
        intervention_loss_weight: float = 1.0,
        # ====== Regularization ======
        supervise_zero_residual_off_intervention: bool = True,
        zero_residual_loss_weight: float = 1.0,
        off_intervention_residual_l1_weight: float = 0.5,
        # ====== Base Policy ======
        base_policy: str,
        base_policy_ckpt_path: str,
        base_policy_overrides: Optional[List[str]] = None,
        # ====== Learning ======
        lr: float = 1e-4,
        use_cosine_lr: bool = True,
        lr_warmup_steps: Optional[int] = None,
        lr_cosine_steps: Optional[int] = None,
        lr_cosine_min: Optional[float] = None,
        optimizer: str = "adam",
        weight_decay: float = 0.0,
        lr_layer_decay: float = 1.0,
        # ====== Two-Stage Training ======
        stage2_ckpt_path: Optional[str] = None,  # If set, freeze backbone+action, train intervention only
        freeze_action_head: bool = False,  # Train intervention head only (reversed stage 1)
        reversed_stage2_ckpt_path: Optional[str] = None,  # Load ckpt, freeze backbone+intervention, train action only
        dropout: float = 0.0,
        **kwargs,
    ):
        # Filter out kwargs that LightningModule doesn't accept
        _allowed = {"online_eval", "policy_wrapper", "robot_type"}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in _allowed}
        super().__init__(*args, **filtered_kwargs)

        self._prop_dim = prop_dim
        self._prop_keys = prop_keys
        self._action_dim = action_dim

        # Feature extraction + fusion
        self._features = set(feature_extractors.keys())
        self.feature_extractor = SimpleFeatureFusion(
            extractors={
                k: instantiate(v) for k, v in feature_extractors.items()
            },
            hidden_depth=feature_fusion_hidden_depth,
            hidden_dim=feature_fusion_hidden_dim,
            output_dim=feature_fusion_output_dim,
            activation=feature_fusion_activation,
            add_input_activation=feature_fusion_add_input_activation,
            add_output_activation=feature_fusion_add_output_activation,
        )

        # Simple MLP action head → deterministic residual
        activation_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[action_net_activation]
        layers = []
        in_dim = feature_fusion_output_dim
        for _ in range(action_net_hidden_depth):
            layers.extend([nn.Linear(in_dim, action_net_hidden_dim), activation_fn()])
            in_dim = action_net_hidden_dim
        layers.append(nn.Linear(in_dim, action_dim))
        layers.append(nn.Tanh())  # constrain output to [-1, 1]
        self.action_net = nn.Sequential(*layers)

        # Optional intervention head
        self._use_intervention_head = use_intervention_head
        if use_intervention_head:
            self.intervention_head = CategoricalNet(
                feature_fusion_output_dim,
                action_dim=2,
                hidden_dim=intervention_head_hidden_dim,
                hidden_depth=intervention_head_hidden_depth,
                activation=intervention_head_activation,
            )
        else:
            self.intervention_head = None

        # Loss config
        self._intervention_loss_weight = intervention_loss_weight
        self._supervise_zero_residual_off_intervention = supervise_zero_residual_off_intervention
        self._zero_residual_loss_weight = zero_residual_loss_weight
        self._off_intervention_residual_l1_weight = off_intervention_residual_l1_weight

        # Load frozen base policy
        assert base_policy_ckpt_path is not None, "Must provide base_policy_ckpt_path!"
        self.base_policy = self._load_base_policy(base_policy, base_policy_ckpt_path, base_policy_overrides)

        # Learning params
        self.lr = lr
        self.use_cosine_lr = use_cosine_lr
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_cosine_steps = lr_cosine_steps
        self.lr_cosine_min = lr_cosine_min
        self.optimizer_name = optimizer
        self.weight_decay = weight_decay

        # Stage 2: load stage 1 checkpoint, freeze backbone + action head, train intervention only
        self._stage2 = stage2_ckpt_path is not None
        if self._stage2:
            assert use_intervention_head, "Stage 2 requires use_intervention_head=true"
            print(f"[SimpleResidualPolicy] Stage 2: loading backbone from {stage2_ckpt_path}")
            ckpt = load_torch(stage2_ckpt_path, map_location="cpu")
            # Load everything except intervention head
            state = ckpt["state_dict"]
            own_state = self.state_dict()
            for k, v in state.items():
                if k in own_state and "intervention_head" not in k and "base_policy" not in k:
                    own_state[k] = v
            self.load_state_dict(own_state, strict=False)
            # Freeze feature extractor + action net
            freeze_params(self.feature_extractor)
            freeze_params(self.action_net)
            print(f"[SimpleResidualPolicy] Frozen: feature_extractor, action_net. Training: intervention_head only.")

        # Reversed stage 1: freeze action head, train backbone + intervention head
        self._freeze_action_head = freeze_action_head
        if freeze_action_head:
            assert use_intervention_head, "freeze_action_head requires use_intervention_head=true"
            freeze_params(self.action_net)
            print(f"[SimpleResidualPolicy] Reversed stage 1: frozen action_net. Training: feature_extractor + intervention_head.")

        # Reversed stage 2: load ckpt with trained intervention head, freeze it + backbone, train action only
        self._reversed_stage2 = reversed_stage2_ckpt_path is not None
        if self._reversed_stage2:
            print(f"[SimpleResidualPolicy] Reversed stage 2: loading from {reversed_stage2_ckpt_path}")
            ckpt = load_torch(reversed_stage2_ckpt_path, map_location="cpu")
            state = ckpt["state_dict"]
            own_state = self.state_dict()
            for k, v in state.items():
                if k in own_state and "action_net" not in k and "base_policy" not in k:
                    own_state[k] = v
            self.load_state_dict(own_state, strict=False)
            freeze_params(self.feature_extractor)
            if self.intervention_head is not None:
                freeze_params(self.intervention_head)
            print(f"[SimpleResidualPolicy] Frozen: feature_extractor, intervention_head. Training: action_net only.")

    def _load_base_policy(self, base_policy_name: str, ckpt_path: str, extra_overrides: Optional[List[str]] = None):
        """Load and freeze the base policy."""
        overrides = [f"arch={base_policy_name}"]

        # Collect CLI overrides (same logic as ResidualPolicy)
        def _skip(o: str) -> bool:
            return o.startswith("module.") or o.startswith("+module.") or o.startswith("++module.")

        cli_overrides = []
        if GlobalHydra.instance().is_initialized():
            try:
                hydra_cfg = HydraConfig.get()
                cli_overrides = [o for o in hydra_cfg.overrides.task if not o.startswith("arch=") and not _skip(o)]
            except Exception:
                cli_overrides = []

        if not any(o.startswith("robot=") for o in cli_overrides) or not any(o.startswith("task=") for o in cli_overrides):
            argv_overrides = [o for o in sys.argv[1:] if not o.startswith("arch=") and not _skip(o)]
            for o in argv_overrides:
                if o not in cli_overrides:
                    cli_overrides.append(o)

        overrides.extend(cli_overrides)
        if extra_overrides:
            overrides.extend(extra_overrides)

        if GlobalHydra.instance().is_initialized():
            cfg = compose(config_name="base_config", overrides=overrides).module
        else:
            config_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs"))
            with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
                cfg = compose(config_name="base_config", overrides=overrides).module

        register_omegaconf_resolvers()
        OmegaConf.resolve(cfg)
        policy = instantiate(cfg, _recursive_=False)

        ckpt = load_torch(ckpt_path, map_location="cpu")
        load_state_dict(policy, ckpt["state_dict"], strict=True)
        policy = policy.to("cuda")
        policy.eval()
        freeze_params(policy)
        return policy

    def forward(self, obs):
        """Forward pass: extract features, predict residual action."""
        # Build proprioception
        prop_obs = []
        for key in self._prop_keys:
            if "/" in key:
                group, k = key.split("/")
                prop_obs.append(obs[group][k])
            else:
                prop_obs.append(obs[key])
        prop_obs = torch.cat(prop_obs, dim=-1)
        obs["proprioception"] = prop_obs
        obs = {k: obs[k] for k in self._features}

        features = self.feature_extractor(obs)  # (B, T, D)
        residual_action = self.action_net(features)  # (B, T, action_dim)
        intervention_dist = self.intervention_head(features) if self._use_intervention_head else None
        return residual_action, intervention_dist

    @torch.no_grad()
    def act_bundle(self, obs, deterministic=None):
        """Inference: return (residual_action, intervention, intervention_weight)."""
        residual_action, intervention_dist = self.forward(obs)

        if intervention_dist is None:
            # No intervention head → always intervene (always apply residual)
            intervention = torch.ones(
                residual_action.shape[:-1], device=residual_action.device, dtype=torch.long
            )
            intervention_weight = torch.ones(
                residual_action.shape[:-1], device=residual_action.device, dtype=residual_action.dtype
            )
        else:
            intervention = intervention_dist.mode()
            intervention_weight = intervention_dist.probs[..., 1]

        return residual_action, intervention, intervention_weight

    @torch.no_grad()
    def act(self, obs, deterministic=None):
        residual_action, intervention, _ = self.act_bundle(obs, deterministic=deterministic)
        return residual_action, intervention

    def reset(self) -> None:
        pass

    def configure_optimizers(self):
        if self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        if self.use_cosine_lr:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=optimizer,
                lr_lambda=CosineScheduleFunction(
                    base_value=1.0,
                    final_value=self.lr_cosine_min / self.lr,
                    epochs=self.lr_cosine_steps,
                    warmup_start_value=self.lr_cosine_min / self.lr,
                    warmup_epochs=self.lr_warmup_steps,
                    steps_per_epoch=1,
                ),
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        return optimizer

    def policy_training_step(self, batch, batch_idx):
        return self._compute_loss(batch, batch_idx, is_train=True)

    def policy_evaluation_step(self, batch, batch_idx):
        return self._compute_loss(batch, batch_idx, is_train=False)

    def _compute_loss(self, batch, batch_idx, is_train: bool):
        """MSE-based residual loss with zero-residual supervision."""
        batch = self.process_data(batch, extract_action=True)

        pad_mask = batch.pop("masks")
        intervention_mask = batch.pop("int_state") == 2
        action_valid_mask = intervention_mask & pad_mask
        non_intervention_mask = (~intervention_mask) & pad_mask

        # Compute residual targets
        base_action = batch["base_action"]
        oracle_action = batch["oracle_action"]

        # Split arm vs gripper
        base_arm, base_grip = base_action[..., :-1], base_action[..., -1:]
        oracle_arm, oracle_grip = oracle_action[..., :-1], oracle_action[..., -1:]

        # Binarize gripper
        base_grip = torch.where(base_grip >= 0, 1.0, 0.0)
        oracle_grip = torch.where(oracle_grip >= 0, 1.0, 0.0)

        residual_arm = oracle_arm - base_arm
        residual_grip = oracle_grip - base_grip
        target_action = torch.cat([residual_arm, residual_grip], dim=-1)

        # Zero-residual on non-intervention timesteps
        if self._supervise_zero_residual_off_intervention:
            target_action = torch.where(intervention_mask.unsqueeze(-1), target_action, torch.zeros_like(target_action))
            action_valid_mask = pad_mask  # supervise all timesteps

        # Forward
        pred_residual, intervention_dist = self.forward(batch)

        # MSE loss on residual action
        mse = ((pred_residual - target_action) ** 2).sum(dim=-1)  # (B, T)

        intervention_mse = (mse * (intervention_mask & pad_mask)).sum() / (intervention_mask & pad_mask).sum().clamp_min(1)

        if self._supervise_zero_residual_off_intervention:
            non_int_mse = (mse * non_intervention_mask).sum() / non_intervention_mask.sum().clamp_min(1)
            denom = 1.0 + (self._zero_residual_loss_weight if non_intervention_mask.sum().item() > 0 else 0.0)
            action_loss = (intervention_mse + self._zero_residual_loss_weight * non_int_mse) / denom
        else:
            non_int_mse = torch.zeros_like(intervention_mse)
            action_loss = intervention_mse

        # L1 regularization on non-intervention residuals
        raw_l1 = pred_residual.abs().mean(dim=-1)
        non_int_l1 = (raw_l1 * non_intervention_mask).sum() / non_intervention_mask.sum().clamp_min(1)

        # Intervention head loss
        if intervention_dist is not None:
            raw_int_loss = intervention_dist.imitation_loss(intervention_mask.long(), reduction="none").reshape(pad_mask.shape)
            int_loss = (raw_int_loss * pad_mask).sum() / pad_mask.sum()
            int_acc = intervention_dist.imitation_accuracy(intervention_mask.long(), mask=pad_mask)
        else:
            int_loss = torch.tensor(0.0, device=mse.device)
            int_acc = torch.tensor(1.0, device=mse.device)

        loss = (
            action_loss
            + self._intervention_loss_weight * int_loss
            + self._off_intervention_residual_l1_weight * non_int_l1
        )

        intervention_steps = (intervention_mask & pad_mask).sum()
        log_dict = {
            "action_loss": action_loss,
            "intervention_mse": intervention_mse,
            "non_intervention_mse": non_int_mse,
            "non_intervention_l1": non_int_l1,
            "intervention_loss": int_loss,
            "intervention_acc": int_acc,
            "intervention_rate": intervention_steps.float() / pad_mask.sum().clamp_min(1),
        }
        if not is_train:
            log_dict["l1"] = action_loss + int_loss

        return loss, log_dict, action_valid_mask.sum()

    def process_data(self, data_batch: dict, extract_action: bool = False) -> Any:
        """Process observation data — same as ResidualPolicy for compatibility."""
        data = {"qpos": data_batch["obs"]["qpos"], "eef": data_batch["obs"]["eef"]}
        if "odom" in data_batch["obs"]:
            data["odom"] = data_batch["obs"]["odom"]
        if "task" in self._features:
            data["task"] = data_batch["obs"]["task"]
        if extract_action:
            data.update({
                "int_state": data_batch["policy"]["int_state"],
                "base_action": data_batch["policy"]["base_action"],
                "oracle_action": data_batch["policy"]["oracle_action"],
                "masks": data_batch["masks"],
            })
        return data
