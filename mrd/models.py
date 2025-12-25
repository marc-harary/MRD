import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
import torch
from torch import nn
from torch.nn import functional as F
from typing import Literal

Task = Literal["regression", "classification"]


# ----------------------------
# Model
# ----------------------------
class MLP(nn.Module):
    def __init__(
        self,
        width: int,
        depth: int,
        in_dim: int = 28 * 28,
        out_dim: int = 10,
        use_layernorm: bool = True,
        activation: str = "gelu",
    ):
        super().__init__()
        act = nn.GELU() if activation.lower() == "gelu" else nn.Tanh()

        layers: List[nn.Module] = [nn.Flatten(), nn.Linear(in_dim, width)]
        if use_layernorm:
            layers += [nn.LayerNorm(width)]
        layers += [act]

        for _ in range(depth - 1):
            layers += [nn.Linear(width, width)]
            if use_layernorm:
                layers += [nn.LayerNorm(width)]
            layers += [act]

        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ----------------------------
# LightningModule (task-general)
# ----------------------------
class MRDTaskModule(pl.LightningModule):
    """
    Use this for BOTH tasks.

    For classification:
      - model(x) -> logits (B,C)
      - y -> int64 (B,)
      - loss = cross_entropy

    For regression:
      - model(x) -> (B,) or (B,1)
      - y -> float (B,) or (B,1)
      - loss = 0.5 * mse
    """

    def __init__(
        self,
        model: nn.Module,
        task: Task,
        lr: float,
        momentum: float,
        weight_decay: float = 0.0,
        nesterov: bool = True,
        num_classes: int = 10,  # only used when task="classification"
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.task: Task = task
        self.num_classes = int(num_classes) if task == "classification" else 1

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.task == "classification":
            y = y.to(torch.long)
            logits = self(x)
            loss = F.cross_entropy(logits, y)
        else:
            y = y.to(torch.float32)
            pred = self(x)
            pred = pred.squeeze(-1) if pred.ndim == 2 and pred.shape[1] == 1 else pred
            y = y.squeeze(-1) if y.ndim == 2 and y.shape[1] == 1 else y
            loss = 0.5 * F.mse_loss(pred, y, reduction="mean")

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        if self.task == "classification":
            y = y.to(torch.long)
            logits = self(x)
            acc = (logits.argmax(dim=1) == y).float().mean()
            loss = F.cross_entropy(logits, y)
            self.log("test/acc", acc, prog_bar=True, on_step=False, on_epoch=True)
            self.log("test/loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        else:
            y = y.to(torch.float32)
            pred = self(x)
            pred = pred.squeeze(-1) if pred.ndim == 2 and pred.shape[1] == 1 else pred
            y = y.squeeze(-1) if y.ndim == 2 and y.shape[1] == 1 else y
            loss = 0.5 * F.mse_loss(pred, y, reduction="mean")
            self.log("test/loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        opt = torch.optim.SGD(
            self.parameters(),
            lr=float(self.hparams.lr),
            momentum=float(self.hparams.momentum),
            weight_decay=float(self.hparams.weight_decay),
            nesterov=bool(self.hparams.nesterov),
        )
        return opt
