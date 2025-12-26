import torch
import torch.nn.functional as F
import pandas as pd
from typing import Optional, Dict, List
from lightning.pytorch.callbacks import Callback
import lightning as pl

# assumes run_mrd_diagnostics is in scope
from mrd.core import run_mrd_diagnostics
from mrd.core_exact import exact_eig_extrema_HGR_Hztheta


class MRDDiagnosticsCallback(Callback):
    """
    Logs MRD probes + loss on the same diagnostic batch every `every_n_steps`.

    Notes:
      - Uses a single forward for loss (no-grad) separate from MRD (grad-enabled).
      - Leaves pl_module/model training mode unchanged (restores previous mode).
      - For regression, logs MSE (not 0.5*MSE unless you change it).
    """

    def __init__(self, every_n_steps: int, diag_bs: int, K: int):
        super().__init__()
        self.every_n_steps = int(every_n_steps)
        self.diag_bs = int(diag_bs)
        self.K = int(K)
        self.snapshot: Optional[Dict[str, torch.Tensor]] = None
        self.rows: List[dict] = []

    @torch.no_grad()
    def _compute_loss(self, pl_module, xb: torch.Tensor, yb: torch.Tensor) -> float:
        was_training = pl_module.model.training
        pl_module.model.eval()

        z = pl_module.model(xb)

        if pl_module.task == "classification":
            # z: (B,C), y: (B,) int
            loss = F.cross_entropy(z, yb.to(torch.long), reduction="mean")
        else:
            # regression: allow (B,) or (B,1)
            if z.ndim == 2 and z.shape[1] == 1:
                z = z[:, 0]
            y = yb
            if y.ndim == 2 and y.shape[1] == 1:
                y = y[:, 0]
            y = y.to(z.dtype)
            loss = F.mse_loss(z, y, reduction="mean")

        pl_module.model.train(was_training)
        return float(loss.detach().cpu())

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if step == 0 or (step % self.every_n_steps) != 0:
            return

        # --- get a diagnostic batch
        dm = trainer.datamodule
        loader = dm.train_dataloader()
        xb, yb = next(iter(loader))
        xb = xb[: self.diag_bs].to(pl_module.device, non_blocking=True)
        yb = yb[: self.diag_bs].to(pl_module.device, non_blocking=True)

        # --- loss on same diag batch (no grad)
        loss_val = self._compute_loss(pl_module, xb, yb)
        loss_name = "cross_entropy" if pl_module.task == "classification" else "mse"

        # --- MRD probes (need grads)
        yb_probe = yb.to(torch.long) if pl_module.task == "classification" else yb
        with torch.enable_grad():
            diag, self.snapshot = run_mrd_diagnostics(
                pl_module.model,
                xb,
                yb_probe,
                K=self.K,
                prev_snapshot=self.snapshot,
                task=pl_module.task,
                num_classes=int(getattr(pl_module, "num_classes", 10)),
                add_loss=False,  # avoid double-computing loss inside run_mrd_diagnostics
            )

        row = dict(step=step, loss=float(loss_val), loss_name=loss_name, **diag)
        self.rows.append(row)

        # --- log MRD scalars
        for k, v in diag.items():
            if isinstance(v, (float, int)):
                pl_module.log(
                    f"mrd/{k}", float(v), prog_bar=False, on_step=True, on_epoch=False
                )

        # --- log loss too
        pl_module.log(
            "mrd/loss", float(loss_val), prog_bar=False, on_step=True, on_epoch=False
        )

    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


class ExactCurvatureCallback(Callback):
    """
    Exact (batch-level) curvature diagnostics.

    For each trigger step, computes exact eigen-extrema of:
      - H        = ∂²_θ L
      - G        = Jᵀ (∂²_z L) J
      - R        = H − G
      - Hztheta  = ∂²_θ s(theta),  s = mean(vec(z))

    Stores everything in a single table with columns:
      stage, step, batch_idx, H_lmin, H_lmax, G_lmin, ...
    """

    def __init__(
        self,
        every_n_steps: int = 100,
        max_batches: Optional[int] = None,   # limit cost
    ):
        super().__init__()
        self.every_n_steps = int(every_n_steps)
        self.max_batches = max_batches
        self.rows: List[Dict] = []

    # -------------------------------------------------
    # shared driver
    # -------------------------------------------------
    def _run(
        self,
        stage: str,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch,
        batch_idx: int,
    ):
        step = int(trainer.global_step)

        if stage == "train":
            if step == 0 or step % self.every_n_steps != 0:
                return
        else:
            if self.max_batches is not None and batch_idx >= self.max_batches:
                return

        xb, yb = batch
        xb = xb.to(pl_module.device)
        yb = yb.to(pl_module.device)

        with torch.enable_grad():
            stats = exact_eig_extrema_HGR_Hztheta(
                model=pl_module.model,
                inputs=xb,
                targets=yb,
                loss_fn=pl_module.loss_fn,
            )

        row = dict(
            stage=stage,
            step=step,
            batch_idx=int(batch_idx),
            **stats,
        )
        self.rows.append(row)

        # optional Lightning logging
        for k, v in stats.items():
            if isinstance(v, float):
                pl_module.log(
                    f"curv/{k}",
                    v,
                    on_step=(stage == "train"),
                    on_epoch=(stage != "train"),
                    prog_bar=False,
                )

    # -------------------------------------------------
    # hooks
    # -------------------------------------------------
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._run("train", trainer, pl_module, batch, batch_idx)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._run("val", trainer, pl_module, batch, batch_idx)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._run("test", trainer, pl_module, batch, batch_idx)

    # -------------------------------------------------
    # export
    # -------------------------------------------------
    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)
