from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Lightning (new namespace)
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
# from lightning.pytorch.utilities.seed import seed_everything

from torch.func import functional_call, jvp, vjp, grad

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset, TensorDataset

import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# DataModule
# ----------------------------
class MNISTData(pl.LightningDataModule):
    def __init__(self, data_root: str, train_bs: int, test_bs: int, num_workers: int = 4):
        super().__init__()
        self.data_root = data_root
        self.train_bs = train_bs
        self.test_bs = test_bs
        self.num_workers = num_workers

        self.tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def prepare_data(self):
        datasets.MNIST(self.data_root, train=True, download=True)
        datasets.MNIST(self.data_root, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        full = datasets.MNIST(self.data_root, train=True, download=False, transform=self.tfm)
        test = datasets.MNIST(self.data_root, train=False, download=False, transform=self.tfm)
        # small validation split
        n_val = 5000
        n_train = len(full) - n_val
        self.train_ds, self.val_ds = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(SEED))
        self.test_ds = test

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.train_bs, shuffle=True,
            num_workers=self.num_workers if ACCELERATOR == "gpu" else 0,
            pin_memory=(ACCELERATOR == "gpu"), drop_last=True, persistent_workers=(ACCELERATOR == "gpu"),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.test_bs, shuffle=False,
            num_workers=self.num_workers if ACCELERATOR == "gpu" else 0,
            pin_memory=(ACCELERATOR == "gpu"), persistent_workers=(ACCELERATOR == "gpu"),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=self.test_bs, shuffle=False,
            num_workers=self.num_workers if ACCELERATOR == "gpu" else 0,
            pin_memory=(ACCELERATOR == "gpu"), persistent_workers=(ACCELERATOR == "gpu"),
        )


# ----------------------------
# Analytic targets (pick one)
# ----------------------------

def target_sin_linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    # x: (B,d), w: (d,)
    return torch.sin(x @ w)

def target_sin_mix(x: torch.Tensor, w: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    # y = sin(w^T x) + 0.3 sin( (Ax)^T w )  (still analytic)
    return torch.sin(x @ w) + 0.3 * torch.sin((x @ A.T) @ w)

def target_rbf_bumps(x: torch.Tensor, centers: torch.Tensor, scales: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    # analytic: sum_j w_j exp(-||x-c_j||^2 / (2 s_j^2))
    # centers: (m,d), scales: (m,), weights: (m,)
    # returns (B,)
    x2 = (x[:, None, :] - centers[None, :, :]).pow(2).sum(dim=-1)  # (B,m)
    return (weights[None, :] * torch.exp(-0.5 * x2 / (scales[None, :] ** 2))).sum(dim=1)

def target_polynomial(x: torch.Tensor, W2: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # analytic (actually polynomial): y = sum_i b_i x_i + sum_{i,j} W2_{ij} x_i x_j
    lin = x @ b
    quad = (x @ W2 * x).sum(dim=1)  # x^T W2 x
    return lin + 0.5 * quad


# ----------------------------
# Fixed analytic dataset
# ----------------------------

class AnalyticRegressionDataset(Dataset):
    """
    Fixed synthetic regression dataset:
      x ~ N(0, I) or Uniform[-a,a]^d
      y = f*(x) + noise
    """
    def __init__(
        self,
        n: int,
        d: int,
        target: Callable[[torch.Tensor], torch.Tensor],
        noise_std: float = 0.0,
        x_dist: str = "normal",            # "normal" or "uniform"
        uniform_a: float = 1.0,
        seed: int = 0,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        super().__init__()
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)

        if x_dist == "normal":
            X = torch.randn(n, d, generator=g, dtype=dtype)
        elif x_dist == "uniform":
            X = (2 * uniform_a) * torch.rand(n, d) - uniform_a #, generator=g, dtype=dtype) - uniform_a
        else:
            raise ValueError(f"Unknown x_dist={x_dist}")

        with torch.no_grad():
            y = target(X).to(dtype)
            if y.ndim != 1:
                y = y.view(-1)
            if noise_std > 0:
                y = y + noise_std * torch.randn_like(y)#, generator=g)

        self.X = X.to(device)
        self.y = y.to(device)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ----------------------------
# Optional: infinite-stream dataset
# ----------------------------

class InfiniteAnalyticRegressionDataset(Dataset):
    """
    Infinite stream via __getitem__ drawing fresh x each time.
    Lightning DataLoader will sample indices; we ignore idx.
    """
    def __init__(
        self,
        d: int,
        target: Callable[[torch.Tensor], torch.Tensor],
        noise_std: float = 0.0,
        x_dist: str = "normal",
        uniform_a: float = 1.0,
        seed: int = 0,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        super().__init__()
        self.d = d
        self.target = target
        self.noise_std = float(noise_std)
        self.x_dist = x_dist
        self.uniform_a = float(uniform_a)
        self.dtype = dtype
        self.device = device

        # one RNG per worker will be handled by worker_init_fn; for notebook simplicity
        self.base_seed = int(seed)

    def __len__(self) -> int:
        return 10**12  # effectively infinite

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # deterministic per idx if you want reproducibility:
        g = torch.Generator(device="cpu")
        g.manual_seed(self.base_seed + int(idx))

        if self.x_dist == "normal":
            x = torch.randn(self.d, generator=g, dtype=self.dtype)
        elif self.x_dist == "uniform":
            a = self.uniform_a
            x = (2 * a) * torch.rand(self.d) - a #, generator=g, dtype=self.dtype) - a
        else:
            raise ValueError(f"Unknown x_dist={self.x_dist}")

        with torch.no_grad():
            y = self.target(x.view(1, -1)).view(-1)[0]
            if self.noise_std > 0:
                y = y + self.noise_std * torch.randn(())#, generator=g, dtype=self.dtype)

        return x.to(self.device), y.to(self.device)


# ----------------------------
# DataModule
# ----------------------------

@dataclass
class AnalyticDMConfig:
    d: int = 3
    n_train: int = 50_000
    n_test: int = 10_000
    train_bs: int = 512
    test_bs: int = 1024
    num_workers: int = 0

    noise_std: float = 0.0
    x_dist: str = "normal"         # "normal" or "uniform"
    uniform_a: float = 1.0
    seed: int = 0

    fixed_dataset: bool = True     # True: fixed tensors; False: infinite stream
    pin_memory: bool = True

    # optional input standardization based on train set stats (fixed_dataset only)
    standardize_x: bool = True


class AnalyticRegressionData(pl.LightningDataModule):
    """
    Produces (x,y) with y = f*(x) + noise, where f* is analytic.

    Key knobs:
      - config.fixed_dataset=True gives a stable finite dataset.
      - config.standardize_x=True removes anisotropy in x (helps isolate model curvature effects).
    """
    def __init__(
        self,
        config: AnalyticDMConfig,
        target_fn: Callable[[torch.Tensor], torch.Tensor],
    ):
        super().__init__()
        self.cfg = config
        self.target_fn = target_fn

        self.train_ds: Optional[Dataset] = None
        self.test_ds: Optional[Dataset] = None
        self._x_mean: Optional[torch.Tensor] = None
        self._x_std: Optional[torch.Tensor] = None

    def setup(self, stage: Optional[str] = None):
        cfg = self.cfg

        if cfg.fixed_dataset:
            train = AnalyticRegressionDataset(
                n=cfg.n_train, d=cfg.d, target=self.target_fn,
                noise_std=cfg.noise_std, x_dist=cfg.x_dist, uniform_a=cfg.uniform_a,
                seed=cfg.seed, dtype=torch.float32, device="cpu"
            )
            test = AnalyticRegressionDataset(
                n=cfg.n_test, d=cfg.d, target=self.target_fn,
                noise_std=cfg.noise_std, x_dist=cfg.x_dist, uniform_a=cfg.uniform_a,
                seed=cfg.seed + 12345, dtype=torch.float32, device="cpu"
            )

            if cfg.standardize_x:
                Xtr = train.X
                mean = Xtr.mean(dim=0, keepdim=True)
                std = Xtr.std(dim=0, keepdim=True).clamp_min(1e-6)
                self._x_mean = mean
                self._x_std = std

                train = TensorDataset((train.X - mean) / std, train.y)
                test  = TensorDataset((test.X  - mean) / std, test.y)

            self.train_ds, self.test_ds = train, test

        else:
            # infinite stream: no standardization (unless you want to fix it analytically)
            self.train_ds = InfiniteAnalyticRegressionDataset(
                d=cfg.d, target=self.target_fn,
                noise_std=cfg.noise_std, x_dist=cfg.x_dist, uniform_a=cfg.uniform_a,
                seed=cfg.seed, dtype=torch.float32, device="cpu"
            )
            self.test_ds = AnalyticRegressionDataset(
                n=cfg.n_test, d=cfg.d, target=self.target_fn,
                noise_std=cfg.noise_std, x_dist=cfg.x_dist, uniform_a=cfg.uniform_a,
                seed=cfg.seed + 12345, dtype=torch.float32, device="cpu"
            )

    def train_dataloader(self):
        cfg = self.cfg
        return DataLoader(
            self.train_ds,
            batch_size=cfg.train_bs,
            shuffle=cfg.fixed_dataset,  # stream can't shuffle meaningfully
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=True,
        )

    def test_dataloader(self):
        cfg = self.cfg
        return DataLoader(
            self.test_ds,
            batch_size=cfg.test_bs,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )
