from __future__ import annotations

from typing import Dict, List, Optional

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
import pandas as pd
import torch
import torch.nn.functional as F

from mrd.core import run_mrd_diagnostics


class MRDDiagnosticsCallback(Callback):
    def __init__(self, every_n_steps: int, diag_bs: int, K: int):
        super().__init__()
        self.every_n_steps = int(every_n_steps)
        self.diag_bs = int(diag_bs)
        self.K = int(K)

        self.snapshot: Optional[Dict[str, torch.Tensor]] = None
        self.rows: List[dict] = []

    @torch.no_grad()
    def _compute_loss(self, pl_module, xb: torch.Tensor, yb: torch.Tensor) -> float:
        pl_module.model.eval()
        z = pl_module.model(xb)

        if pl_module.task == "classification":
            # z: (B,C), y: (B,) int64
            loss = F.cross_entropy(z, yb.to(torch.long))
        else:
            # regression: z: (B,) or (B,1), y: (B,) or (B,1) float
            if z.ndim == 2 and z.shape[1] == 1:
                z = z[:, 0]
            y = yb
            if y.ndim == 2 and y.shape[1] == 1:
                y = y[:, 0]
            loss = F.mse_loss(z, y.to(z.dtype))

        pl_module.model.train()
        return float(loss.detach().cpu())

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if step == 0 or (step % self.every_n_steps) != 0:
            return

        dm = trainer.datamodule
        loader = dm.train_dataloader()
        xb, yb = next(iter(loader))
        xb = xb[: self.diag_bs].to(pl_module.device, non_blocking=True)
        yb = yb[: self.diag_bs].to(pl_module.device, non_blocking=True)

        # loss on the same diag batch (no-grad)
        loss_val = self._compute_loss(pl_module, xb, yb)
        loss_name = "cross_entropy" if pl_module.task == "classification" else "mse"

        # Ensure correct dtype for CE targets for probes
        if pl_module.task == "classification":
            yb = yb.to(torch.long)

        # MRD probes need grads enabled
        with torch.enable_grad():
            diag, self.snapshot = run_mrd_diagnostics(
                pl_module.model,
                xb,
                yb,
                K=self.K,
                prev_snapshot=self.snapshot,
                task=pl_module.task,
                num_classes=getattr(pl_module, "num_classes", 10),
            )

        row = dict(step=step, loss=loss_val, loss_name=loss_name, **diag)
        self.rows.append(row)

        # log MRD scalars
        for k, v in diag.items():
            if isinstance(v, (float, int)):
                pl_module.log(
                    f"mrd/{k}", float(v), prog_bar=False, on_step=True, on_epoch=False
                )

        # log loss too
        pl_module.log(
            "mrd/loss", float(loss_val), prog_bar=False, on_step=True, on_epoch=False
        )

    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)
