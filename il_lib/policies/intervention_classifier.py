"""Intervention classifier: predicts whether the base policy needs correction.

A lightweight MLP trained on (proprio + base_action + action_history) features
to predict upcoming intervention (int_state==1) vs normal operation (int_state==3).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, Precision, Recall


class InterventionClassifier(pl.LightningModule):
    """Binary classifier for intervention prediction."""

    def __init__(
        self,
        input_dim: int = 42,
        hidden_dims: list = [256, 128, 64],
        dropout: float = 0.1,
        class_weights: list = [1.0, 8.0],
        lr: float = 1e-3,
        action_history_len: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.action_history_len = action_history_len

        # Build MLP
        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hdim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, 2))
        self.net = nn.Sequential(*layers)

        # Loss with class weighting
        self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))

        # Metrics
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.val_f1 = F1Score(task="binary")
        self.val_precision = Precision(task="binary")
        self.val_recall = Recall(task="binary")

    def forward(self, x):
        return self.net(x)

    def predict_proba(self, x):
        """Return probability of intervention (class 1)."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)[:, 1]

    def training_step(self, batch, batch_idx):
        features = batch["features"]
        labels = batch["label"]
        logits = self.forward(features)
        loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        preds = logits.argmax(dim=-1)
        self.train_acc(preds, labels)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        features = batch["features"]
        labels = batch["label"]
        logits = self.forward(features)
        loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        preds = logits.argmax(dim=-1)
        self.val_acc(preds, labels)
        self.val_f1(preds, labels)
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True)
        self.log("val/precision", self.val_precision, on_step=False, on_epoch=True)
        self.log("val/recall", self.val_recall, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
