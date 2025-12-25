# ============================================================
# mrd_lightning_file.py  (paste into a Jupyter cell)
#
# This is YOUR file rewritten to support BOTH:
#   (A) regression (MSE) with float targets
#   (B) classification (cross-entropy) with int64 targets
#
# Key change: all MRD probes are parameterized by `task` and `num_classes`,
# and the LightningModule exposes those so the callback can call probes
# correctly.
#
# Assumptions:
#   - model(x) returns:
#       * regression: shape (B,) or (B,1) (float)
#       * classification: shape (B,C) logits (float)
#   - y dtype:
#       * regression: float, shape (B,) or (B,1)
#       * classification: long, shape (B,)
# ============================================================

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd

from torch.func import functional_call, jvp, vjp, grad


Task = Literal["regression", "classification"]


# ----------------------------
# MRD / GGN matrix-free probes
# ----------------------------
def params_from_model(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v for k, v in model.named_parameters()}


def param_names(model: nn.Module) -> List[str]:
    return [k for k, _ in model.named_parameters()]


def flatten_pytree(pytree: Dict[str, torch.Tensor], keys: List[str]) -> torch.Tensor:
    return torch.cat([pytree[k].reshape(-1) for k in keys], dim=0)


@torch.no_grad()
def copy_params(params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in params.items()}


def rademacher_like(
    params: Dict[str, torch.Tensor], keys: List[str]
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k in keys:
        t = params[k]
        r = (
            torch.randint(0, 2, t.shape, device=t.device, dtype=torch.int8) * 2 - 1
        ).to(t.dtype)
        out[k] = r
    return out


def _ensure_2d_outputs(z: torch.Tensor, task: Task) -> torch.Tensor:
    """
    Standardize model outputs for probes:
      - classification: keep (B,C)
      - regression: make (B,1)
    """
    if task == "classification":
        if z.ndim != 2:
            raise ValueError(
                f"classification expects logits (B,C), got {tuple(z.shape)}"
            )
        return z
    # regression
    if z.ndim == 1:
        return z[:, None]
    if z.ndim == 2 and z.shape[1] == 1:
        return z
    raise ValueError(f"regression expects (B,) or (B,1), got {tuple(z.shape)}")


def logits_fn_factory(model: nn.Module, Xb: torch.Tensor, task: Task):
    def _logits(p: Dict[str, torch.Tensor]) -> torch.Tensor:
        z = functional_call(model, p, (Xb,))
        return _ensure_2d_outputs(z, task)

    return _logits


def apply_G(
    model: nn.Module,
    keys: List[str],
    params: Dict[str, torch.Tensor],
    Xb: torch.Tensor,
    v_params: Dict[str, torch.Tensor],
    task: Task,
):
    """
    Matrix-free metric curvature:
      - classification (CE):   G v = J^T W J v, W = softmax Fisher block (frozen at current logits)
      - regression (MSE):      G v = J^T J v (since W = I on scalar output)
    Returns: (Gv_tree, v^T G v)
    """
    logits_fn = logits_fn_factory(model, Xb, task)

    z, s = jvp(logits_fn, (params,), (v_params,))  # z: (B, out), s: (B, out)

    B = Xb.shape[0]

    if task == "classification":
        with torch.no_grad():
            p = torch.softmax(z, dim=1)  # (B,C)
        # W s = (diag(p) - p p^T) s = p*s - p*(p^T s)
        dot = (p * s).sum(dim=1, keepdim=True)  # (B,1)
        t = p * s - p * dot  # (B,C)
    else:
        # regression: W = I (scalar output) => t = s
        t = s  # (B,1)

    # Average over batch to match (1/B) sum conventions.
    t = t / B

    _, pullback = vjp(logits_fn, params)
    Gv = pullback(t)[0]  # J^T t

    vdotGv = (flatten_pytree(v_params, keys) * flatten_pytree(Gv, keys)).sum()
    return Gv, vdotGv


def apply_R(
    model: nn.Module,
    keys: List[str],
    params: Dict[str, torch.Tensor],
    Xb: torch.Tensor,
    yb: torch.Tensor,
    v_params: Dict[str, torch.Tensor],
    task: Task,
    num_classes: int = 10,
    a_override: Optional[torch.Tensor] = None,
):
    """
    Matrix-free residual curvature:
      - classification (CE):
          R v = (1/B) sum_i sum_k a_{ik} ∇^2 z_{ik} v
          with a = softmax(z) - onehot(y) treated constant
      - regression (MSE):
          R v = (1/B) sum_i e_i ∇^2 f_i v
          with e = (f - y) treated constant
    Implement via Hessian-vector product of:
      s(p) = (1/B) sum_i,k a_{ik} z_{ik}(p)   (classification)
      s(p) = (1/B) sum_i   e_i    f_i(p)     (regression)
    """
    B = Xb.shape[0]
    logits_fn = logits_fn_factory(model, Xb, task)

    if task == "classification":
        C = int(num_classes)

        if a_override is None:
            z0 = logits_fn(params)  # (B,C)
            with torch.no_grad():
                p0 = torch.softmax(z0, dim=1)
                y_oh = F.one_hot(yb.to(torch.long), num_classes=C).to(p0.dtype)
                a = p0 - y_oh  # (B,C)
        else:
            a = a_override  # (B,C)

        def s_fn(p: Dict[str, torch.Tensor]) -> torch.Tensor:
            z = logits_fn(p)  # (B,C)
            return (a * z).sum() / B

    else:
        # regression
        if a_override is None:
            f0 = logits_fn(params)  # (B,1)
            with torch.no_grad():
                y = yb
                if y.ndim == 2 and y.shape[1] == 1:
                    y = y[:, 0]
                y = y.to(f0.dtype).view(-1)  # (B,)
                e = (f0[:, 0] - y).detach()  # (B,)
        else:
            e = a_override.view(-1).detach()  # (B,)

        def s_fn(p: Dict[str, torch.Tensor]) -> torch.Tensor:
            f = logits_fn(p)[:, 0]  # (B,)
            return (e * f).sum() / B

    grad_s = grad(s_fn)
    _, Rv = jvp(grad_s, (params,), (v_params,))
    return Rv


def frob_margin_from_trace_estimates(trG: float, GF: float, RF: float, P: int):
    """
    m_F = mu - alpha*A - ||R||_F
      mu = tr(G)/P
      A  = ||G - mu I||_F  (computed from (trG, ||G||_F))
      alpha = sqrt((P-1)/P)
    """
    mu = trG / P
    alpha = math.sqrt((P - 1) / P)
    A2 = max(GF * GF - 2 * mu * trG + (mu * mu) * P, 0.0)
    A = math.sqrt(A2)
    mF = mu - alpha * A - RF
    return mF, mu, A, alpha


@torch.no_grad()
def run_mrd_diagnostics(
    model: nn.Module,
    Xb: torch.Tensor,
    yb: torch.Tensor,
    K: int,
    prev_snapshot: Optional[Dict[str, torch.Tensor]],
    task: Task,
    num_classes: int = 10,
):
    """
    Returns:
      G_F, trG, R_F, m_F, mu, A, alpha,
      plus FD channel norms if prev_snapshot exists:
        Phi_align_F_fd, Phi_damp_F_fd, Phi_trans_F_fd
    """
    model.eval()
    keys = param_names(model)
    params = params_from_model(model)
    P = sum(params[k].numel() for k in keys)

    # ---- ||G||_F and tr(G) via Hutchinson
    GF2 = 0.0
    trG_acc = 0.0
    for _ in range(K):
        v = rademacher_like(params, keys)
        Gv_tree, vTAv = apply_G(model, keys, params, Xb, v, task=task)
        Gv = flatten_pytree(Gv_tree, keys)
        GF2 += float((Gv * Gv).sum().detach().cpu())
        trG_acc += float(vTAv.detach().cpu())
    G_F = math.sqrt(GF2 / K)
    trG = trG_acc / K

    # ---- ||R||_F via Hutchinson
    RF2 = 0.0
    for _ in range(K):
        v = rademacher_like(params, keys)
        Rv_tree = apply_R(
            model, keys, params, Xb, yb, v, task=task, num_classes=num_classes
        )
        Rv = flatten_pytree(Rv_tree, keys)
        RF2 += float((Rv * Rv).sum().detach().cpu())
    R_F = math.sqrt(RF2 / K)

    m_F, mu, A, alpha = frob_margin_from_trace_estimates(trG, G_F, R_F, P)
    out = dict(
        P=P,
        batch=int(Xb.shape[0]),
        G_F=G_F,
        trG=trG,
        R_F=R_F,
        m_F=m_F,
        mu=mu,
        A=A,
        alpha=alpha,
    )

    # ---- Finite-difference channel norms
    if prev_snapshot is not None and "params_prev" in prev_snapshot:
        params_prev = prev_snapshot["params_prev"]

        # Phi_align v ≈ G_now v - G_prev v
        def Phi_align_Av(v):
            Gv_now, _ = apply_G(model, keys, params, Xb, v, task=task)
            Gv_prev, _ = apply_G(model, keys, params_prev, Xb, v, task=task)
            return {k: (Gv_now[k] - Gv_prev[k]) for k in keys}

        # Build weights for damp/trans splits
        logits_fn_now = logits_fn_factory(model, Xb, task)
        z_now = logits_fn_now(params)
        z_prev = logits_fn_now(params_prev)

        if task == "classification":
            C = int(num_classes)
            p_now = torch.softmax(z_now, dim=1)
            p_prev = torch.softmax(z_prev, dim=1)
            y_oh = F.one_hot(yb.to(torch.long), num_classes=C).to(p_now.dtype)
            a_now = p_now - y_oh
            a_prev = p_prev - y_oh
            da = a_now - a_prev

            # Phi_damp v ≈ H_{da}(params_now) v
            def Phi_damp_Av(v):
                return apply_R(
                    model,
                    keys,
                    params,
                    Xb,
                    yb,
                    v,
                    task=task,
                    num_classes=C,
                    a_override=da,
                )

            # Phi_trans v ≈ H_{a_prev}(params_now) v - H_{a_prev}(params_prev) v
            def Phi_trans_Av(v):
                Rv_now = apply_R(
                    model,
                    keys,
                    params,
                    Xb,
                    yb,
                    v,
                    task=task,
                    num_classes=C,
                    a_override=a_prev,
                )
                Rv_prev = apply_R(
                    model,
                    keys,
                    params_prev,
                    Xb,
                    yb,
                    v,
                    task=task,
                    num_classes=C,
                    a_override=a_prev,
                )
                return {k: (Rv_now[k] - Rv_prev[k]) for k in keys}

        else:
            # regression: weights are residuals e = f - y
            y = yb
            if y.ndim == 2 and y.shape[1] == 1:
                y = y[:, 0]
            y = y.to(z_now.dtype).view(-1)
            e_now = (z_now[:, 0] - y).detach()
            e_prev = (z_prev[:, 0] - y).detach()
            de = e_now - e_prev  # damping weights

            def Phi_damp_Av(v):
                return apply_R(model, keys, params, Xb, yb, v, task=task, a_override=de)

            def Phi_trans_Av(v):
                Rv_now = apply_R(
                    model, keys, params, Xb, yb, v, task=task, a_override=e_prev
                )
                Rv_prev = apply_R(
                    model, keys, params_prev, Xb, yb, v, task=task, a_override=e_prev
                )
                return {k: (Rv_now[k] - Rv_prev[k]) for k in keys}

        Pa2 = 0.0
        Pd2 = 0.0
        Pt2 = 0.0
        for _ in range(K):
            v = rademacher_like(params, keys)
            Pav = flatten_pytree(Phi_align_Av(v), keys)
            Pdv = flatten_pytree(Phi_damp_Av(v), keys)
            Ptv = flatten_pytree(Phi_trans_Av(v), keys)
            Pa2 += float((Pav * Pav).sum().detach().cpu())
            Pd2 += float((Pdv * Pdv).sum().detach().cpu())
            Pt2 += float((Ptv * Ptv).sum().detach().cpu())

        out["Phi_align_F_fd"] = math.sqrt(Pa2 / K)
        out["Phi_damp_F_fd"] = math.sqrt(Pd2 / K)
        out["Phi_trans_F_fd"] = math.sqrt(Pt2 / K)

    snap = dict(params_prev=copy_params(params))
    model.train()
    return out, snap
