from __future__ import annotations

from collections import deque
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from il_lib.optim import CosineScheduleFunction
from il_lib.policies.policy_base import BasePolicy


def _make_mlp(
    input_dim: int,
    hidden_dim: int,
    hidden_depth: int,
    activation: str,
    dropout: float,
) -> tuple[nn.Module, int]:
    assert input_dim > 0
    assert hidden_depth >= 0
    act_layer = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
    }[activation]

    layers: list[nn.Module] = []
    in_dim = input_dim
    for _ in range(hidden_depth):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(act_layer())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_dim = hidden_dim
    return (nn.Sequential(*layers) if layers else nn.Identity(), in_dim)


class SimpleCorrectorMLPSplit(BasePolicy):
    """
    Split takeover/oracle corrector.

    Contract:
        takeover branch input:
            base_action_history, shape (B, history_steps, action_dim),
            optionally concatenated with current or historical actual proprio
            state and a scalar Mahalanobis/OOD feature.
        oracle branch input:
            current proprio/task/dp embedding, optionally cross-conditioned on
            the takeover latent.
        outputs:
            oracle actions, shape (B, prediction_horizon, action_dim)
            takeover logits, shape (B, prediction_horizon)

    The takeover loss only updates the takeover branch. If cross_condition is
    enabled with detach_cross_condition=true, the oracle branch can read the
    takeover latent without sending oracle gradients back through it.
    """

    is_sequence_policy = True

    def __init__(
        self,
        *args,
        action_dim: int = 7,
        history_steps: int = 16,
        prediction_horizon: int = 16,
        takeover_prediction_horizon: Optional[int] = None,
        prop_dim: int = 14,
        prop_keys: Optional[Sequence[str]] = None,
        task_dim: int = 6,
        use_task_input: bool = True,
        takeover_prop_dim: int = 0,
        takeover_proprio_history_steps: int = 1,
        takeover_include_action_deltas: bool = False,
        takeover_use_time_progress: bool = False,
        takeover_use_mahalanobis: bool = False,
        takeover_mahalanobis_history_steps: int = 1,
        mahalanobis_calibration_path: Optional[str] = None,
        mahalanobis_transform: str = "raw",
        mahalanobis_log_mean: float = 0.0,
        mahalanobis_log_std: float = 1.0,
        mahalanobis_scale: float = 25.0,
        mahalanobis_clip: float = 100.0,
        takeover_hidden_dim: int = 256,
        takeover_hidden_depth: int = 3,
        oracle_hidden_dim: int = 256,
        oracle_hidden_depth: int = 3,
        activation: str = "relu",
        dropout: float = 0.1,
        cross_condition: bool = True,
        detach_cross_condition: bool = True,
        takeover_state: int = 2,
        takeover_states: Optional[Sequence[int]] = None,
        takeover_pos_weight: float = 1.0,
        hard_negative_weight: float = 1.0,
        hard_positive_weight: float = 1.0,
        action_loss_weight: float = 1.0,
        takeover_loss_weight: float = 1.0,
        action_loss_on_all_steps: bool = False,
        action_loss_type: str = "smooth_l1",
        gate_threshold: float = 0.5,
        gate_on_threshold: Optional[float] = None,
        gate_off_threshold: Optional[float] = None,
        takeover_prob_smoothing_window: int = 1,
        takeover_release_patience_steps: int = 0,
        lr: float = 1e-4,
        use_cosine_lr: bool = True,
        lr_warmup_steps: Optional[int] = None,
        lr_cosine_steps: Optional[int] = None,
        lr_cosine_min: Optional[float] = None,
        lr_layer_decay: float = 1.0,
        optimizer: str = "adamw",
        weight_decay: float = 0.0,
        **kwargs,
    ):
        allowed = {"online_eval", "policy_wrapper", "robot_type"}
        super().__init__(*args, **{k: v for k, v in kwargs.items() if k in allowed})

        assert action_dim > 0
        assert history_steps > 0
        assert prediction_horizon > 0
        if takeover_prediction_horizon is None:
            takeover_prediction_horizon = prediction_horizon
        assert takeover_prediction_horizon > 0
        assert prop_dim >= 0
        assert task_dim >= 0
        assert takeover_prop_dim >= 0
        assert takeover_prop_dim <= prop_dim
        assert takeover_proprio_history_steps >= 1
        assert takeover_mahalanobis_history_steps >= 1

        self.action_dim = action_dim
        self.history_steps = history_steps
        self.prediction_horizon = prediction_horizon
        self.takeover_prediction_horizon = takeover_prediction_horizon
        self.prop_dim = prop_dim
        self.prop_keys = list(prop_keys or [])
        self.task_dim = task_dim
        self.use_task_input = use_task_input
        self.takeover_prop_dim = takeover_prop_dim
        self.takeover_proprio_history_steps = takeover_proprio_history_steps
        self.takeover_include_action_deltas = takeover_include_action_deltas
        self.takeover_use_time_progress = takeover_use_time_progress
        self.takeover_use_mahalanobis = takeover_use_mahalanobis
        self.takeover_mahalanobis_history_steps = takeover_mahalanobis_history_steps
        self.mahalanobis_transform = str(mahalanobis_transform)
        if self.mahalanobis_transform not in {"raw", "log1p", "zlog1p"}:
            raise ValueError(
                "mahalanobis_transform must be one of {'raw', 'log1p', 'zlog1p'}, "
                f"got {self.mahalanobis_transform!r}"
            )
        self.mahalanobis_log_mean = float(mahalanobis_log_mean)
        self.mahalanobis_log_std = max(float(mahalanobis_log_std), 1e-6)
        self.mahalanobis_scale = float(mahalanobis_scale)
        self.mahalanobis_clip = float(mahalanobis_clip)
        self.cross_condition = cross_condition
        self.detach_cross_condition = detach_cross_condition
        self._takeover_state = takeover_state
        self._takeover_states = tuple(
            int(state) for state in (takeover_states if takeover_states is not None else [takeover_state])
        )
        self._takeover_pos_weight = takeover_pos_weight
        self._hard_negative_weight = hard_negative_weight
        self._hard_positive_weight = hard_positive_weight
        self._action_loss_weight = action_loss_weight
        self._takeover_loss_weight = takeover_loss_weight
        self._action_loss_on_all_steps = action_loss_on_all_steps
        self._action_loss_type = action_loss_type
        self._gate_threshold = gate_threshold
        self._gate_on_threshold = gate_threshold if gate_on_threshold is None else float(gate_on_threshold)
        self._gate_off_threshold = gate_threshold if gate_off_threshold is None else float(gate_off_threshold)
        self._takeover_prob_smoothing_window = max(int(takeover_prob_smoothing_window), 1)
        self._takeover_release_patience_steps = max(int(takeover_release_patience_steps), 0)
        self._takeover_prob_history: list[torch.Tensor] = []
        self._takeover_active: Optional[torch.Tensor] = None
        self._takeover_release_count: Optional[torch.Tensor] = None
        self._runtime_proprio_history: deque[torch.Tensor] = deque(
            maxlen=takeover_proprio_history_steps
        )
        self._runtime_mahalanobis_history: deque[torch.Tensor] = deque(
            maxlen=takeover_mahalanobis_history_steps
        )

        takeover_input_dim = history_steps * action_dim
        if takeover_prop_dim > 0:
            takeover_input_dim += takeover_prop_dim * takeover_proprio_history_steps
        if takeover_include_action_deltas:
            takeover_input_dim += (history_steps - 1) * action_dim
        if takeover_use_time_progress:
            takeover_input_dim += 3
        if takeover_use_mahalanobis:
            takeover_input_dim += takeover_mahalanobis_history_steps
        self.takeover_trunk, takeover_latent_dim = _make_mlp(
            input_dim=takeover_input_dim,
            hidden_dim=takeover_hidden_dim,
            hidden_depth=takeover_hidden_depth,
            activation=activation,
            dropout=dropout,
        )
        self.takeover_head = nn.Linear(takeover_latent_dim, takeover_prediction_horizon)

        oracle_input_dim = prop_dim
        if self.use_task_input:
            oracle_input_dim += task_dim
        if cross_condition:
            oracle_input_dim += takeover_latent_dim

        self.oracle_trunk, oracle_latent_dim = _make_mlp(
            input_dim=oracle_input_dim,
            hidden_dim=oracle_hidden_dim,
            hidden_depth=oracle_hidden_depth,
            activation=activation,
            dropout=dropout,
        )
        self.oracle_head = nn.Linear(oracle_latent_dim, prediction_horizon * action_dim)

        self._has_mahalanobis_calibration = False
        if mahalanobis_calibration_path:
            calibration = np.load(mahalanobis_calibration_path)
            self.register_buffer(
                "_mahalanobis_mean",
                torch.as_tensor(calibration["mean"], dtype=torch.float32),
            )
            self.register_buffer(
                "_mahalanobis_precision",
                torch.as_tensor(calibration["precision"], dtype=torch.float32),
            )
            self._has_mahalanobis_calibration = True

        self.lr = lr
        self.use_cosine_lr = use_cosine_lr
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_cosine_steps = lr_cosine_steps
        self.lr_cosine_min = lr_cosine_min
        self.lr_layer_decay = lr_layer_decay
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.save_hyperparameters()

    def forward(
        self,
        base_action_history: torch.Tensor,
        proprio: torch.Tensor,
        task: Optional[torch.Tensor] = None,
        time_progress: Optional[torch.Tensor] = None,
        mahalanobis_distance: Optional[torch.Tensor] = None,
        proprio_history: Optional[torch.Tensor] = None,
        mahalanobis_history: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._check_base_action_shape(base_action_history)

        proprio_flat = self._flatten_context(proprio, self.prop_dim, "proprio")
        takeover_inputs = [base_action_history.reshape(base_action_history.shape[0], -1)]
        if self.takeover_include_action_deltas:
            action_deltas = base_action_history[:, 1:] - base_action_history[:, :-1]
            takeover_inputs.append(action_deltas.reshape(action_deltas.shape[0], -1))
        if self.takeover_prop_dim > 0:
            takeover_inputs.append(
                self._takeover_proprio_context(proprio_flat, proprio_history)
            )
        if self.takeover_use_time_progress:
            takeover_inputs.append(self._time_progress_features(time_progress, proprio_flat))
        if self.takeover_use_mahalanobis:
            takeover_inputs.append(
                self._mahalanobis_feature(
                    mahalanobis_distance,
                    proprio_flat,
                    mahalanobis_history=mahalanobis_history,
                )
            )
        x_takeover = torch.cat(takeover_inputs, dim=-1)
        z_takeover = self.takeover_trunk(x_takeover)
        takeover_logits = self.takeover_head(z_takeover).reshape(
            -1, self.takeover_prediction_horizon
        )

        oracle_inputs = [proprio_flat]
        if self.use_task_input:
            oracle_inputs.append(self._prepare_optional_context(task, self.task_dim, "task", proprio))
        if self.cross_condition:
            oracle_inputs.append(
                z_takeover.detach() if self.detach_cross_condition else z_takeover
            )

        z_oracle = self.oracle_trunk(torch.cat(oracle_inputs, dim=-1))
        oracle_actions = torch.tanh(self.oracle_head(z_oracle)).reshape(
            -1, self.prediction_horizon, self.action_dim
        )
        return oracle_actions, takeover_logits

    @torch.no_grad()
    def act(self, obs: dict, deterministic=None) -> tuple[torch.Tensor, torch.Tensor]:
        batch = self.process_data(obs, extract_action=False)
        oracle_actions, takeover_logits = self.forward(
            base_action_history=batch["base_action_history"],
            proprio=batch["proprio"],
            task=batch.get("task"),
            time_progress=batch.get("time_progress"),
            mahalanobis_distance=batch.get("mahalanobis_distance"),
            proprio_history=batch.get("proprio_history"),
            mahalanobis_history=batch.get("mahalanobis_history"),
        )
        takeover_prob = torch.sigmoid(takeover_logits)
        if self._takeover_prob_smoothing_window > 1:
            if (
                self._takeover_prob_history
                and self._takeover_prob_history[-1].shape != takeover_prob.shape
            ):
                self._takeover_prob_history.clear()
            self._takeover_prob_history.append(takeover_prob.detach())
            self._takeover_prob_history = self._takeover_prob_history[-self._takeover_prob_smoothing_window :]
            takeover_prob = torch.stack(self._takeover_prob_history, dim=0).mean(dim=0)
        takeover = self._apply_gate_hysteresis(takeover_prob)
        return oracle_actions, takeover

    def reset(self) -> None:
        self._takeover_prob_history.clear()
        self._takeover_active = None
        self._takeover_release_count = None
        self._runtime_proprio_history.clear()
        self._runtime_mahalanobis_history.clear()

    def policy_training_step(self, batch, batch_idx):
        return self._compute_loss(batch, is_train=True)

    def policy_evaluation_step(self, batch, batch_idx):
        return self._compute_loss(batch, is_train=False)

    def _compute_loss(self, batch, is_train: bool):
        batch = self.process_data(batch, extract_action=True)
        oracle_action = batch["oracle_action"]
        takeover_target = batch["takeover_target"]
        pad_mask = batch["masks"]

        pred_oracle_action, takeover_logits = self.forward(
            base_action_history=batch["base_action_history"],
            proprio=batch["proprio"],
            task=batch.get("task"),
            time_progress=batch.get("time_progress"),
            mahalanobis_distance=batch.get("mahalanobis_distance"),
            proprio_history=batch.get("proprio_history"),
            mahalanobis_history=batch.get("mahalanobis_history"),
        )

        action_mask = pad_mask
        if not self._action_loss_on_all_steps:
            action_mask = action_mask & takeover_target

        if self._action_loss_type == "smooth_l1":
            action_err = F.smooth_l1_loss(
                pred_oracle_action, oracle_action, reduction="none"
            ).mean(dim=-1)
        elif self._action_loss_type == "mse":
            action_err = F.mse_loss(
                pred_oracle_action, oracle_action, reduction="none"
            ).mean(dim=-1)
        else:
            raise ValueError(f"Unknown action_loss_type: {self._action_loss_type}")
        action_loss = (action_err * action_mask).sum() / action_mask.sum().clamp_min(1)

        takeover_steps = min(self.takeover_prediction_horizon, takeover_target.shape[1])
        takeover_target_for_loss = takeover_target[:, :takeover_steps]
        takeover_mask = pad_mask[:, :takeover_steps]
        takeover_logits_for_loss = takeover_logits[:, :takeover_steps]

        pos_weight = torch.tensor(
            self._takeover_pos_weight,
            device=takeover_logits_for_loss.device,
            dtype=takeover_logits_for_loss.dtype,
        )
        takeover_bce = F.binary_cross_entropy_with_logits(
            takeover_logits_for_loss,
            takeover_target_for_loss.to(takeover_logits_for_loss.dtype),
            reduction="none",
            pos_weight=pos_weight,
        )
        takeover_weights = torch.ones_like(takeover_bce)
        hard_negative_rate = takeover_bce.new_tensor(0.0)
        hard_positive_rate = takeover_bce.new_tensor(0.0)
        if self._hard_negative_weight != 1.0:
            hard_negative = batch.get("hard_negative")
            if hard_negative is not None:
                hard_negative = hard_negative[:, :takeover_steps].bool()
                hard_negative = hard_negative & ~takeover_target_for_loss & takeover_mask.bool()
                takeover_weights = torch.where(
                    hard_negative,
                    takeover_weights.new_full((), self._hard_negative_weight),
                    takeover_weights,
                )
                hard_negative_rate = hard_negative.sum().float() / takeover_mask.sum().clamp_min(1)
        if self._hard_positive_weight != 1.0:
            hard_positive = batch.get("hard_positive")
            if hard_positive is not None:
                hard_positive = hard_positive[:, :takeover_steps].bool()
                hard_positive = hard_positive & takeover_target_for_loss & takeover_mask.bool()
                takeover_weights = torch.where(
                    hard_positive,
                    takeover_weights.new_full((), self._hard_positive_weight),
                    takeover_weights,
                )
                hard_positive_rate = hard_positive.sum().float() / takeover_mask.sum().clamp_min(1)

        weighted_takeover_mask = takeover_mask * takeover_weights
        takeover_loss = (takeover_bce * weighted_takeover_mask).sum() / weighted_takeover_mask.sum().clamp_min(1)

        loss = (
            self._action_loss_weight * action_loss
            + self._takeover_loss_weight * takeover_loss
        )

        takeover_pred = takeover_logits_for_loss > 0
        valid = takeover_mask.bool()
        positives = takeover_target_for_loss & valid
        predicted_positives = takeover_pred & valid
        true_positives = takeover_pred & positives
        takeover_acc = (
            takeover_pred[valid] == takeover_target_for_loss[valid]
        ).float().mean()
        takeover_recall = (
            true_positives.sum().float() / positives.sum().clamp_min(1)
        )
        takeover_precision = (
            true_positives.sum().float() / predicted_positives.sum().clamp_min(1)
        )

        log_dict = {
            "action_loss": action_loss,
            "takeover_loss": takeover_loss,
            "takeover_acc": takeover_acc,
            "takeover_precision": takeover_precision,
            "takeover_recall": takeover_recall,
            "takeover_rate": positives.sum().float() / valid.sum().clamp_min(1),
            "hard_negative_rate": hard_negative_rate,
            "hard_positive_rate": hard_positive_rate,
        }
        if not is_train:
            log_dict["l1"] = action_loss + takeover_loss
        return loss, log_dict, pad_mask.sum()

    def process_data(self, data_batch: dict, extract_action: bool = False) -> dict[str, Any]:
        data = {
            "base_action_history": self._extract_base_action_history(data_batch),
            "proprio": self._extract_proprio(data_batch),
        }
        if self.use_task_input:
            data["task"] = self._extract_task(data_batch, reference=data["proprio"])
        if self.takeover_use_mahalanobis:
            data["mahalanobis_distance"] = self._extract_mahalanobis_distance(
                data_batch,
                reference=data["proprio"],
            )
            if self.takeover_mahalanobis_history_steps > 1:
                data["mahalanobis_history"] = self._extract_mahalanobis_history(
                    data_batch,
                    current_distance=data["mahalanobis_distance"],
                    reference=data["proprio"],
                    update_runtime=not extract_action,
                )
        if self.takeover_prop_dim > 0 and self.takeover_proprio_history_steps > 1:
            data["proprio_history"] = self._extract_proprio_history(
                data_batch,
                reference=data["proprio"],
                update_runtime=not extract_action,
            )

        if extract_action:
            policy = data_batch["policy"]
            data["oracle_action"] = policy["oracle_action"]
            data["takeover_target"] = self._make_takeover_target(policy["int_state"])
            data["hard_negative"] = policy.get(
                "hard_negative",
                torch.zeros_like(policy["int_state"], dtype=torch.bool),
            ).bool()
            data["hard_positive"] = policy.get(
                "hard_positive",
                torch.zeros_like(policy["int_state"], dtype=torch.bool),
            ).bool()
            data["masks"] = data_batch["masks"]
            if self.takeover_use_time_progress:
                data["time_progress"] = self._extract_time_progress(policy, reference=data["proprio"])
            self._check_target_shapes(data)
        elif self.takeover_use_time_progress:
            policy = data_batch.get("policy", {})
            data["time_progress"] = self._extract_time_progress(policy, reference=data["proprio"])
        return data

    def _make_takeover_target(self, int_state: torch.Tensor) -> torch.Tensor:
        takeover_target = torch.zeros_like(int_state, dtype=torch.bool)
        for state in self._takeover_states:
            takeover_target = takeover_target | (int_state == state)
        return takeover_target

    def configure_optimizers(self):
        if self.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

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

    def _extract_base_action_history(self, data_batch: dict) -> torch.Tensor:
        if "base_action_history" in data_batch:
            base_action_history = data_batch["base_action_history"]
            if base_action_history.ndim == 2:
                base_action_history = base_action_history.reshape(
                    base_action_history.shape[0], self.history_steps, self.action_dim
                )
            return base_action_history

        if "base_action" in data_batch:
            base_action_history = data_batch["base_action"]
        elif "policy" in data_batch and "base_action" in data_batch["policy"]:
            base_action_history = data_batch["policy"]["base_action"]
        else:
            raise KeyError(
                "SimpleCorrectorMLPSplit requires base_action_history, base_action, "
                "or policy/base_action in the batch."
            )

        if base_action_history.ndim == 2:
            base_action_history = base_action_history.unsqueeze(0)
        return base_action_history

    def _extract_proprio(self, data_batch: dict) -> torch.Tensor:
        if "proprio" in data_batch:
            return data_batch["proprio"]
        if "obs" not in data_batch:
            raise KeyError("SimpleCorrectorMLPSplit requires obs or proprio in the batch.")

        obs = data_batch["obs"]
        prop_obs = []
        for prop_key in self.prop_keys:
            if "/" in prop_key:
                group, key = prop_key.split("/", 1)
                prop_obs.append(obs[group][key])
            else:
                prop_obs.append(obs[prop_key])
        if not prop_obs:
            raise KeyError("prop_keys is empty; cannot build proprio input.")
        return torch.cat(prop_obs, dim=-1)

    def _extract_task(self, data_batch: dict, reference: torch.Tensor) -> torch.Tensor:
        if "task" in data_batch:
            return data_batch["task"]
        if "obs" in data_batch and "task" in data_batch["obs"]:
            return data_batch["obs"]["task"]
        return self._zeros_like_context(reference, self.task_dim)

    def _prepare_optional_context(
        self,
        value: Optional[torch.Tensor],
        dim: int,
        name: str,
        reference: torch.Tensor,
        required: bool = False,
    ) -> torch.Tensor:
        if dim == 0:
            return reference.new_zeros((reference.shape[0], 0))
        if value is None:
            if required:
                raise KeyError(f"{name} is required but missing.")
            return self._zeros_like_context(reference, dim)
        return self._flatten_context(value, dim, name)

    def _flatten_context(self, value: torch.Tensor, dim: int, name: str) -> torch.Tensor:
        if dim == 0:
            return value.new_zeros((value.shape[0], 0))
        if value.ndim == 3:
            value = value[:, -1]
        elif value.ndim > 3:
            value = value.reshape(value.shape[0], -1)
        if value.ndim != 2:
            raise ValueError(f"{name} must be (B, D) or (B, T, D); got {tuple(value.shape)}")
        if value.shape[-1] != dim:
            raise ValueError(f"{name} dim must be {dim}; got {value.shape[-1]}")
        return value

    def _takeover_proprio_context(
        self,
        proprio: torch.Tensor,
        proprio_history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.takeover_prop_dim == 0:
            return proprio.new_zeros((proprio.shape[0], 0))
        if self.takeover_proprio_history_steps > 1:
            history = self._format_proprio_history(
                proprio_history,
                proprio,
                required_steps=self.takeover_proprio_history_steps,
            )
            return history[..., : self.takeover_prop_dim].reshape(proprio.shape[0], -1)
        if proprio.shape[-1] < self.takeover_prop_dim:
            raise ValueError(
                "proprio dim must be at least takeover_prop_dim="
                f"{self.takeover_prop_dim}; got {proprio.shape[-1]}"
            )
        return proprio[..., : self.takeover_prop_dim]

    def _extract_proprio_history(
        self,
        data_batch: dict,
        reference: torch.Tensor,
        update_runtime: bool,
    ) -> torch.Tensor:
        value = data_batch.get("proprio_history")
        if value is not None:
            return self._format_proprio_history(
                value.to(reference.device, dtype=reference.dtype),
                reference,
                required_steps=self.takeover_proprio_history_steps,
            )

        if "obs" in data_batch:
            obs = data_batch["obs"]
            prop_obs = []
            for prop_key in self.prop_keys:
                if "/" in prop_key:
                    group, key = prop_key.split("/", 1)
                    prop_obs.append(obs[group][key])
                else:
                    prop_obs.append(obs[prop_key])
            if prop_obs:
                obs_proprio = torch.cat(prop_obs, dim=-1)
                if obs_proprio.ndim == 3 and obs_proprio.shape[1] >= self.takeover_proprio_history_steps:
                    return self._format_proprio_history(
                        obs_proprio[:, -self.takeover_proprio_history_steps :],
                        reference,
                        required_steps=self.takeover_proprio_history_steps,
                    )

        if update_runtime:
            return self._update_runtime_proprio_history(reference)

        raise KeyError(
            "takeover_proprio_history_steps > 1 requires proprio_history in the batch."
        )

    def _format_proprio_history(
        self,
        value: Optional[torch.Tensor],
        reference: torch.Tensor,
        required_steps: int,
    ) -> torch.Tensor:
        if value is None:
            value = reference.unsqueeze(1).expand(-1, required_steps, -1)
        if value.ndim == 2:
            if value.shape[-1] % required_steps != 0:
                raise ValueError(
                    "flat proprio_history width must be divisible by "
                    f"{required_steps}; got {value.shape[-1]}"
                )
            value = value.reshape(value.shape[0], required_steps, -1)
        elif value.ndim != 3:
            raise ValueError(
                "proprio_history must be (B, H, D) or flat (B, H*D); "
                f"got {tuple(value.shape)}"
            )
        if value.shape[1] != required_steps:
            if value.shape[1] == 1:
                value = value.expand(-1, required_steps, -1)
            else:
                raise ValueError(
                    f"proprio_history needs {required_steps} steps; got {value.shape[1]}"
                )
        if value.shape[-1] < self.takeover_prop_dim:
            raise ValueError(
                "proprio_history dim must be at least takeover_prop_dim="
                f"{self.takeover_prop_dim}; got {value.shape[-1]}"
            )
        return value

    def _update_runtime_proprio_history(self, proprio: torch.Tensor) -> torch.Tensor:
        current = proprio[..., : self.takeover_prop_dim].detach().clone()
        if (
            self._runtime_proprio_history
            and self._runtime_proprio_history[-1].shape != current.shape
        ):
            self._runtime_proprio_history.clear()
        self._runtime_proprio_history.append(current)
        while len(self._runtime_proprio_history) < self.takeover_proprio_history_steps:
            self._runtime_proprio_history.appendleft(self._runtime_proprio_history[0])
        return torch.stack(list(self._runtime_proprio_history), dim=1).to(
            device=proprio.device,
            dtype=proprio.dtype,
        )

    def _extract_time_progress(self, policy: dict, reference: torch.Tensor) -> torch.Tensor:
        progress = policy.get("time_progress")
        if progress is None:
            return reference.new_zeros((reference.shape[0], 1))
        if progress.ndim == 3:
            progress = progress[:, 0]
        elif progress.ndim == 1:
            progress = progress.unsqueeze(-1)
        return self._flatten_context(progress, 1, "time_progress").clamp(0.0, 1.0)

    def _extract_mahalanobis_distance(self, data_batch: dict, reference: torch.Tensor) -> torch.Tensor:
        value = data_batch.get("mahalanobis_distance")
        if value is None and "policy" in data_batch:
            value = data_batch["policy"].get("mahalanobis_distance")
        if value is not None:
            if value.ndim == 3:
                value = value[:, 0]
            elif value.ndim == 2 and value.shape[-1] != 1:
                value = value[:, :1]
            elif value.ndim == 1:
                value = value.unsqueeze(-1)
            return self._flatten_context(value.to(reference.device, dtype=reference.dtype), 1, "mahalanobis_distance")

        embedding = data_batch.get("base_embedding")
        if embedding is None:
            embedding = data_batch.get("dp_embedding")
        if embedding is not None and self._has_mahalanobis_calibration:
            return self._compute_mahalanobis_from_embedding(embedding.to(reference.device, dtype=reference.dtype))
        return self._zeros_like_context(reference, 1)

    def _extract_mahalanobis_history(
        self,
        data_batch: dict,
        current_distance: torch.Tensor,
        reference: torch.Tensor,
        update_runtime: bool,
    ) -> torch.Tensor:
        value = data_batch.get("mahalanobis_history")
        if value is not None:
            return self._format_mahalanobis_history(
                value.to(reference.device, dtype=reference.dtype),
                reference,
                required_steps=self.takeover_mahalanobis_history_steps,
            )
        if update_runtime:
            return self._update_runtime_mahalanobis_history(current_distance)
        raise KeyError(
            "takeover_mahalanobis_history_steps > 1 requires mahalanobis_history "
            "in the training batch."
        )

    def _compute_mahalanobis_from_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        while embedding.ndim > 3 and embedding.shape[1] == 1:
            embedding = embedding.squeeze(1)
        if embedding.ndim == 3:
            embedding = embedding.mean(dim=1)
        elif embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)
        if embedding.ndim != 2:
            embedding = embedding.reshape(embedding.shape[0], -1)
        delta = embedding - self._mahalanobis_mean.to(device=embedding.device, dtype=embedding.dtype)
        precision = self._mahalanobis_precision.to(device=embedding.device, dtype=embedding.dtype)
        dist_sq = torch.einsum("bi,ij,bj->b", delta, precision, delta).clamp_min(0.0)
        return torch.sqrt(dist_sq).unsqueeze(-1)

    def _format_mahalanobis_history(
        self,
        value: torch.Tensor,
        reference: torch.Tensor,
        required_steps: int,
    ) -> torch.Tensor:
        if value.ndim == 3:
            if value.shape[-1] != 1:
                value = value[..., :1]
            value = value.squeeze(-1)
        elif value.ndim == 2:
            if value.shape[-1] == 1 and required_steps > 1:
                value = value.expand(-1, required_steps)
            elif value.shape[-1] != required_steps:
                value = value.reshape(value.shape[0], -1)
        elif value.ndim == 1:
            value = value.unsqueeze(-1)
        else:
            value = value.reshape(value.shape[0], -1)
        if value.ndim != 2:
            raise ValueError(
                "mahalanobis_history must be (B, H), (B, H, 1), or flat; "
                f"got {tuple(value.shape)}"
            )
        if value.shape[-1] != required_steps:
            raise ValueError(
                f"mahalanobis_history needs {required_steps} steps; got {value.shape[-1]}"
            )
        return value.to(reference.device, dtype=reference.dtype)

    def _update_runtime_mahalanobis_history(self, distance: torch.Tensor) -> torch.Tensor:
        current = self._flatten_context(distance, 1, "mahalanobis_distance").detach().clone()
        if (
            self._runtime_mahalanobis_history
            and self._runtime_mahalanobis_history[-1].shape != current.shape
        ):
            self._runtime_mahalanobis_history.clear()
        self._runtime_mahalanobis_history.append(current)
        while len(self._runtime_mahalanobis_history) < self.takeover_mahalanobis_history_steps:
            self._runtime_mahalanobis_history.appendleft(self._runtime_mahalanobis_history[0])
        return torch.cat(list(self._runtime_mahalanobis_history), dim=-1).to(
            device=distance.device,
            dtype=distance.dtype,
        )

    def _time_progress_features(
        self,
        time_progress: Optional[torch.Tensor],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if time_progress is None:
            progress = reference.new_zeros((reference.shape[0], 1))
        else:
            progress = self._flatten_context(time_progress, 1, "time_progress").clamp(0.0, 1.0)
        return torch.cat(
            [
                progress,
                torch.sin(torch.pi * progress),
                torch.cos(torch.pi * progress),
            ],
            dim=-1,
        )

    def _mahalanobis_feature(
        self,
        mahalanobis_distance: Optional[torch.Tensor],
        reference: torch.Tensor,
        mahalanobis_history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.takeover_mahalanobis_history_steps > 1:
            if mahalanobis_history is None:
                if mahalanobis_distance is None:
                    history = reference.new_zeros(
                        (reference.shape[0], self.takeover_mahalanobis_history_steps)
                    )
                else:
                    current = self._flatten_context(
                        mahalanobis_distance,
                        1,
                        "mahalanobis_distance",
                    )
                    history = current.expand(-1, self.takeover_mahalanobis_history_steps)
            else:
                history = self._format_mahalanobis_history(
                    mahalanobis_history,
                    reference,
                    required_steps=self.takeover_mahalanobis_history_steps,
                )
            distance = history
        else:
            if mahalanobis_distance is None:
                distance = reference.new_zeros((reference.shape[0], 1))
            else:
                distance = self._flatten_context(mahalanobis_distance, 1, "mahalanobis_distance")
        distance = distance.clamp(0.0, self.mahalanobis_clip)
        if self.mahalanobis_transform == "raw":
            feature = distance
        elif self.mahalanobis_transform == "log1p":
            feature = torch.log1p(distance)
        elif self.mahalanobis_transform == "zlog1p":
            feature = (torch.log1p(distance) - self.mahalanobis_log_mean) / self.mahalanobis_log_std
        else:
            raise RuntimeError(f"Unhandled mahalanobis_transform={self.mahalanobis_transform!r}")
        return feature / max(self.mahalanobis_scale, 1e-6)

    def _apply_gate_hysteresis(self, takeover_prob: torch.Tensor) -> torch.Tensor:
        on = self._gate_on_threshold
        off = self._gate_off_threshold
        patience = self._takeover_release_patience_steps
        if patience == 0 and on == off:
            return takeover_prob >= on

        if (
            self._takeover_active is None
            or self._takeover_active.shape != takeover_prob.shape
            or self._takeover_active.device != takeover_prob.device
        ):
            self._takeover_active = torch.zeros_like(takeover_prob, dtype=torch.bool)
            self._takeover_release_count = torch.zeros_like(takeover_prob, dtype=torch.long)

        active = self._takeover_active
        release_count = self._takeover_release_count
        assert release_count is not None

        start = (~active) & (takeover_prob >= on)
        stay_confident = active & (takeover_prob >= off)
        below_release = active & (takeover_prob < off)
        release_count = torch.where(
            stay_confident | start,
            torch.zeros_like(release_count),
            torch.where(below_release, release_count + 1, release_count),
        )
        active = start | stay_confident | (below_release & (release_count <= patience))

        self._takeover_active = active.detach()
        self._takeover_release_count = release_count.detach()
        return active

    def _zeros_like_context(self, reference: torch.Tensor, dim: int) -> torch.Tensor:
        if reference.ndim == 3:
            shape = (reference.shape[0], reference.shape[1], dim)
        else:
            shape = (reference.shape[0], dim)
        return reference.new_zeros(shape)

    def _check_base_action_shape(self, base_action_history: torch.Tensor) -> None:
        if base_action_history.ndim != 3:
            raise ValueError(
                f"base_action_history must be (B, {self.history_steps}, {self.action_dim}); "
                f"got shape {tuple(base_action_history.shape)}"
            )
        if base_action_history.shape[1] != self.history_steps:
            raise ValueError(
                f"expected {self.history_steps} history steps, got {base_action_history.shape[1]}"
            )
        if base_action_history.shape[2] != self.action_dim:
            raise ValueError(
                f"expected action_dim={self.action_dim}, got {base_action_history.shape[2]}"
            )

    def _check_target_shapes(self, data: dict[str, torch.Tensor]) -> None:
        if data["oracle_action"].shape[-2:] != (
            self.prediction_horizon,
            self.action_dim,
        ):
            raise ValueError(
                "oracle_action must have shape "
                f"(B, {self.prediction_horizon}, {self.action_dim}); "
                f"got {tuple(data['oracle_action'].shape)}"
            )
        if data["takeover_target"].shape[-1] != self.prediction_horizon:
            raise ValueError(
                f"takeover target must have horizon {self.prediction_horizon}; "
                f"got {tuple(data['takeover_target'].shape)}"
            )
        if data["hard_negative"].shape[-1] != self.prediction_horizon:
            raise ValueError(
                f"hard_negative must have horizon {self.prediction_horizon}; "
                f"got {tuple(data['hard_negative'].shape)}"
            )
        if data["hard_positive"].shape[-1] != self.prediction_horizon:
            raise ValueError(
                f"hard_positive must have horizon {self.prediction_horizon}; "
                f"got {tuple(data['hard_positive'].shape)}"
            )
        if data["masks"].shape[-1] != self.prediction_horizon:
            raise ValueError(
                f"mask must have horizon {self.prediction_horizon}; "
                f"got {tuple(data['masks'].shape)}"
            )
