import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback


# ----------------------------
# MRD diagnostics callback
# ----------------------------
class MRDDiagnosticsCallback(Callback):
    def __init__(self, every_n_steps: int, diag_bs: int, K: int):
        super().__init__()
        self.every_n_steps = int(every_n_steps)
        self.diag_bs = int(diag_bs)
        self.K = int(K)

        self.snapshot: Optional[Dict[str, torch.Tensor]] = None
        self.rows: List[dict] = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if step == 0 or (step % self.every_n_steps) != 0:
            return

        dm = trainer.datamodule
        loader = dm.train_dataloader()
        xb, yb = next(iter(loader))
        xb = xb[: self.diag_bs].to(pl_module.device, non_blocking=True)
        yb = yb[: self.diag_bs].to(pl_module.device, non_blocking=True)

        # Ensure correct dtype for CE targets
        if pl_module.task == "classification":
            yb = yb.to(torch.long)

        with torch.enable_grad():
            diag, self.snapshot = run_mrd_diagnostics(
                pl_module.model,
                xb,
                yb,
                K=self.K,
                prev_snapshot=self.snapshot,
                task=pl_module.task,
                num_classes=pl_module.num_classes,
            )

        row = dict(step=step, **diag)
        self.rows.append(row)

        for k, v in diag.items():
            if isinstance(v, (float, int)):
                pl_module.log(
                    f"mrd/{k}", float(v), prog_bar=False, on_step=True, on_epoch=False
                )

    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)
