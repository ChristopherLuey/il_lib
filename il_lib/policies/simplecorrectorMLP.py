import torch
import torch.nn as nn
import torch.nn.functional as F
from il_lib.optim import CosineScheduleFunction
from il_lib.policies.policy_base import BasePolicy
from typing import Any, Optional, Sequence


class SimpleCorrectorMLP(BasePolicy):
    """
    Lightweight takeover corrector.

    Contract:
        input:  16 normalized base-policy actions, shape (B, 16, A)
        output: 16 normalized oracle actions, shape (B, 16, A)
                16 takeover logits, shape (B, 16)

    Unlike SimpleResidualPolicy, this model does not learn oracle - base and
    the deployment semantics should not add its action output to the base action.
    If takeover is active, execute the predicted oracle action directly.
    """

    is_sequence_policy = True

    def __init__(
        self,
        *args,
        action_dim: int = 7,
        history_steps: int = 16,
        prediction_horizon: int = 16,
        hidden_dim: int = 256,
        hidden_depth: int = 3,
        activation: str = "relu",
        dropout: float = 0.1,
        takeover_state: int = 2,
        takeover_states: Optional[Sequence[int]] = None,
        takeover_pos_weight: float = 1.0,
        hard_negative_weight: float = 1.0,
        hard_positive_weight: float = 1.0,
        action_loss_weight: float = 1.0,
        takeover_loss_weight: float = 1.0,
        action_loss_on_all_steps: bool = False,
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
        assert hidden_depth >= 0

        self.action_dim = action_dim
        self.history_steps = history_steps
        self.prediction_horizon = prediction_horizon
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

        act_layer = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
            "silu": nn.SiLU,
        }[activation]

        input_dim = history_steps * action_dim
        layers = []
        in_dim = input_dim
        for _ in range(hidden_depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act_layer())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers) if layers else nn.Identity()
        trunk_dim = hidden_dim if hidden_depth > 0 else input_dim

        self.oracle_action_head = nn.Linear(
            trunk_dim, prediction_horizon * action_dim
        )
        self.takeover_head = nn.Linear(trunk_dim, prediction_horizon)

        self.lr = lr
        self.use_cosine_lr = use_cosine_lr
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_cosine_steps = lr_cosine_steps
        self.lr_cosine_min = lr_cosine_min
        self.lr_layer_decay = lr_layer_decay
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.save_hyperparameters()

    def forward(self, base_action_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            base_action_history: normalized base actions, (B, 16, A)

        Returns:
            oracle_actions: normalized absolute oracle action predictions, (B, 16, A)
            takeover_logits: binary takeover logits, (B, 16)
        """
        self._check_base_action_shape(base_action_history)
        x = base_action_history.reshape(base_action_history.shape[0], -1)
        z = self.trunk(x)
        oracle_actions = torch.tanh(self.oracle_action_head(z)).reshape(
            -1, self.prediction_horizon, self.action_dim
        )
        takeover_logits = self.takeover_head(z).reshape(
            -1, self.prediction_horizon
        )
        return oracle_actions, takeover_logits

    @torch.no_grad()
    def act(self, obs: dict, deterministic=None) -> tuple[torch.Tensor, torch.Tensor]:
        batch = self.process_data(obs, extract_action=False)
        oracle_actions, takeover_logits = self.forward(batch["base_action_history"])
        takeover = takeover_logits > 0
        return oracle_actions, takeover

    def reset(self) -> None:
        pass

    def policy_training_step(self, batch, batch_idx):
        return self._compute_loss(batch, is_train=True)

    def policy_evaluation_step(self, batch, batch_idx):
        return self._compute_loss(batch, is_train=False)

    def _compute_loss(self, batch, is_train: bool):
        batch = self.process_data(batch, extract_action=True)
        base_action_history = batch["base_action_history"]
        oracle_action = batch["oracle_action"]
        takeover_target = batch["takeover_target"]
        pad_mask = batch["masks"]

        pred_oracle_action, takeover_logits = self.forward(base_action_history)

        action_mask = pad_mask
        if not self._action_loss_on_all_steps:
            action_mask = action_mask & takeover_target

        action_l1 = F.smooth_l1_loss(
            pred_oracle_action, oracle_action, reduction="none"
        ).mean(dim=-1)
        action_loss = (action_l1 * action_mask).sum() / action_mask.sum().clamp_min(1)

        pos_weight = torch.tensor(
            self._takeover_pos_weight,
            device=takeover_logits.device,
            dtype=takeover_logits.dtype,
        )
        takeover_bce = F.binary_cross_entropy_with_logits(
            takeover_logits,
            takeover_target.to(takeover_logits.dtype),
            reduction="none",
            pos_weight=pos_weight,
        )
        takeover_weights = torch.ones_like(takeover_bce)
        hard_negative_rate = takeover_bce.new_tensor(0.0)
        hard_positive_rate = takeover_bce.new_tensor(0.0)
        if self._hard_negative_weight != 1.0:
            hard_negative = batch.get("hard_negative")
            if hard_negative is not None:
                hard_negative = hard_negative.bool() & ~takeover_target & pad_mask.bool()
                takeover_weights = torch.where(
                    hard_negative,
                    takeover_weights.new_full((), self._hard_negative_weight),
                    takeover_weights,
                )
                hard_negative_rate = hard_negative.sum().float() / pad_mask.sum().clamp_min(1)
        if self._hard_positive_weight != 1.0:
            hard_positive = batch.get("hard_positive")
            if hard_positive is not None:
                hard_positive = hard_positive.bool() & takeover_target & pad_mask.bool()
                takeover_weights = torch.where(
                    hard_positive,
                    takeover_weights.new_full((), self._hard_positive_weight),
                    takeover_weights,
                )
                hard_positive_rate = hard_positive.sum().float() / pad_mask.sum().clamp_min(1)

        weighted_pad_mask = pad_mask * takeover_weights
        takeover_loss = (takeover_bce * weighted_pad_mask).sum() / weighted_pad_mask.sum().clamp_min(1)

        loss = (
            self._action_loss_weight * action_loss
            + self._takeover_loss_weight * takeover_loss
        )

        takeover_pred = takeover_logits > 0
        valid = pad_mask.bool()
        positives = takeover_target & valid
        predicted_positives = takeover_pred & valid
        true_positives = takeover_pred & positives
        takeover_acc = (takeover_pred[valid] == takeover_target[valid]).float().mean()
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
        }
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
            self._check_target_shapes(data)
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
                "SimpleCorrectorMLP requires base_action_history, base_action, "
                "or policy/base_action in the batch."
            )

        if base_action_history.ndim == 2:
            base_action_history = base_action_history.unsqueeze(0)
        return base_action_history

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
