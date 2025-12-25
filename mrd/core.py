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


def _rademacher_vec(shape, device, dtype):
    return (torch.randint(0, 2, shape, device=device, dtype=torch.int8) * 2 - 1).to(
        dtype
    )


def apply_model_H_mix(
    model: nn.Module,
    keys: List[str],
    params: Dict[str, torch.Tensor],
    Xb: torch.Tensor,
    v_params: Dict[str, torch.Tensor],
    task: Task,
    num_classes: int = 10,
    # b weights select “which outputs” you’re mixing (logits or scalar f)
    b_override: Optional[torch.Tensor] = None,
):
    """
    Returns a pytree representing:
        Hv = ∇^2_θ s(θ) v
    where
        s(θ) = (1/B) Σ_n [ b_n f_n(θ) ]                      (regression)
        s(θ) = (1/B) Σ_{n,k} [ b_{n,k} z_{n,k}(θ) ]          (classification)
    with b being Rademacher weights (default).
    This is *pure model curvature* (no residuals, no Fisher).
    """
    B = Xb.shape[0]
    logits_fn = logits_fn_factory(model, Xb, task)

    if task == "classification":
        C = int(num_classes)
        if b_override is None:
            # b: (B,C) ±1
            with torch.no_grad():
                b = _rademacher_vec((B, C), Xb.device, logits_fn(params).dtype)
        else:
            b = b_override

        def s_fn(p: Dict[str, torch.Tensor]) -> torch.Tensor:
            z = logits_fn(p)  # (B,C)
            return (b * z).sum() / B

    else:
        if b_override is None:
            # b: (B,) ±1
            with torch.no_grad():
                z0 = logits_fn(params)  # (B,1)
                b = _rademacher_vec((B,), Xb.device, z0.dtype)
        else:
            b = b_override.view(-1)

        def s_fn(p: Dict[str, torch.Tensor]) -> torch.Tensor:
            f = logits_fn(p)[:, 0]  # (B,)
            return (b * f).sum() / B

    grad_s = grad(s_fn)  # ∇ s
    _, Hv = jvp(grad_s, (params,), (v_params,))  # ∇^2 s · v
    return Hv


def estimate_model_H_frob_mix(
    model: nn.Module,
    Xb: torch.Tensor,
    task: Task,
    K: int,
    num_classes: int = 10,
) -> float:
    """
    Hutchinson-style estimator of a *model curvature scale*:
        H_mix_F ≈ sqrt( E_{b,v} || (∇^2 s_b) v ||^2 )
    where s_b mixes samplewise (and logitwise) Hessians with random ±1 weights.

    Interpretation:
      - regression: probes typical size of samplewise Hessians H_n acting on random v
      - classification: probes typical size of logit Hessians H_{n,k} acting on random v

    This is not exactly ||H||_F for any single matrix; it’s a stable scalar
    that tracks “how large model second derivatives are” in the same units as R_F.
    """
    keys = param_names(model)
    params = params_from_model(model)

    H2 = 0.0
    for _ in range(K):
        v = rademacher_like(params, keys)  # v ~ Rademacher in parameter space
        Hv_tree = apply_model_H_mix(
            model, keys, params, Xb, v, task=task, num_classes=num_classes
        )
        Hv = flatten_pytree(Hv_tree, keys)
        H2 += float((Hv * Hv).sum().detach().cpu())
    return math.sqrt(H2 / K)


def estimate_T_action_frob(
    model: nn.Module,
    Xb: torch.Tensor,
    yb: torch.Tensor,
    task: Task,
    K: int,
    num_classes: int = 10,
) -> float:
    """
    Optional: estimate || T[∇L] ||_F in a Hutchinson sense.

    We compute (for random v) the vector:
        w(v) = (d/dε)|_{0} [ H(θ + ε v) · g ]    where g = ∇L(θ)
             = (∇^3 L)[v] · g   (a contraction producing a vector in parameter space)

    Implementation: define hvp(theta, g) = ∇^2 L(theta) · g,
    then take a JVP of hvp in direction v.

    Cost: ~2–3x heavier than H_mix_F.
    """
    keys = param_names(model)
    params = params_from_model(model)

    # build loss function
    logits_fn = logits_fn_factory(model, Xb, task)

    def loss_fn(p: Dict[str, torch.Tensor]) -> torch.Tensor:
        z = logits_fn(p)
        if task == "classification":
            return F.cross_entropy(z, yb.to(torch.long), reduction="mean")
        else:
            y = yb
            if y.ndim == 2 and y.shape[1] == 1:
                y = y[:, 0]
            y = y.to(z.dtype).view(-1)
            f = z[:, 0]
            return 0.5 * ((f - y) ** 2).mean()

    # gradient of loss
    g_tree = grad(loss_fn)(params)
    g_flat = flatten_pytree(g_tree, keys)

    def hvp_of_loss(p: Dict[str, torch.Tensor]) -> torch.Tensor:
        # returns H(p) * g as a flat vector
        def dot_grad(p2):
            gg = grad(loss_fn)(p2)
            return (flatten_pytree(gg, keys) * g_flat).sum()

        Hv = grad(dot_grad)(p)  # ∇ [ <∇L, g> ] = H g
        return flatten_pytree(Hv, keys)

    # Hutchinson over v: E || JVP(hvp, v) ||^2
    T2 = 0.0
    for _ in range(K):
        v = rademacher_like(params, keys)
        v_flat = flatten_pytree(v, keys)
        _, jvp_out = jvp(
            hvp_of_loss, (params,), (v,)
        )  # directional derivative of (H g)
        T2 += float((jvp_out * jvp_out).sum().detach().cpu())

    return math.sqrt(T2 / K)


def estimate_model_curvature_F(
    model: nn.Module,
    params: Dict[str, torch.Tensor],
    keys: List[str],
    Xb: torch.Tensor,
    K: int,
    task: Task,
    num_classes: int = 10,
):
    """
    Estimates sqrt( (1/B) sum_n ||H_n||_F^2 )
    via Hutchinson.
    """
    B = Xb.shape[0]
    H2_acc = 0.0

    logits_fn = logits_fn_factory(model, Xb, task)

    for _ in range(K):
        v = rademacher_like(params, keys)

        if task == "classification":
            # random contraction over classes
            u = torch.randn(B, num_classes, device=Xb.device) / math.sqrt(num_classes)

            def s_fn(p):
                z = logits_fn(p)  # (B,C)
                return (u * z).sum() / B

        else:
            # regression
            u = torch.randn(B, device=Xb.device)

            def s_fn(p):
                f = logits_fn(p)[:, 0]  # (B,)
                return (u * f).sum() / B

        grad_s = grad(s_fn)
        _, Hv = jvp(grad_s, (params,), (v,))
        Hv_flat = flatten_pytree(Hv, keys)

        H2_acc += float((Hv_flat * Hv_flat).sum().detach().cpu())

    return math.sqrt(H2_acc / K)


def _loss_from_logits(z, yb, task: str):
    if task == "classification":
        return F.cross_entropy(z, yb.to(torch.long), reduction="mean")
    else:
        # regression
        if z.ndim == 2 and z.shape[1] == 1:
            z = z[:, 0]
        y = yb
        if y.ndim == 2 and y.shape[1] == 1:
            y = y[:, 0]
        return F.mse_loss(z, y.to(z.dtype), reduction="mean")


def apply_H_loss(
    model, keys, params, Xb, yb, v_params, task: str, num_classes: int = 10
):
    """
    Hv where H = ∂^2_θ L(θ) on batch (Xb,yb).
    Works on older torch versions (no materialize_grads).
    Returns a pytree dict matching `params`.
    """
    with torch.enable_grad():

        def loss_fn(p):
            z = functional_call(model, p, (Xb,))
            if task == "classification":
                # z: (B,C), y: (B,) long
                return F.cross_entropy(z, yb.to(torch.long), reduction="mean")
            else:
                # regression
                if z.ndim == 2 and z.shape[1] == 1:
                    z = z[:, 0]
                y = yb
                if y.ndim == 2 and y.shape[1] == 1:
                    y = y[:, 0]
                return F.mse_loss(z, y.to(z.dtype), reduction="mean")

        def grad_fn(p):
            L = loss_fn(p)
            inputs = tuple(p[k] for k in keys)

            grads = torch.autograd.grad(
                L,
                inputs,
                create_graph=True,  # needed for HVP via jvp
                retain_graph=True,  # safer inside repeated probes
                allow_unused=True,  # unused params -> None
            )

            # materialize Nones as real zeros (mutable tensors)
            out = {}
            for k, pk, gk in zip(keys, inputs, grads):
                out[k] = torch.zeros_like(pk) if gk is None else gk
            return out

        _, Hv = jvp(grad_fn, (params,), (v_params,))
        return Hv


def estimate_H_loss_F(model, Xb, yb, K, task: str, num_classes: int = 10):
    with torch.enable_grad():
        model.eval()
        keys = [k for k, _ in model.named_parameters()]
        params = {k: v for k, v in model.named_parameters()}

        H2 = 0.0
        trH = 0.0
        for _ in range(K):
            v = {}
            for k in keys:
                t = params[k]
                r = (
                    torch.randint(0, 2, t.shape, device=t.device, dtype=torch.int8) * 2
                    - 1
                ).to(t.dtype)
                v[k] = r

            Hv = apply_H_loss(
                model, keys, params, Xb, yb, v, task=task, num_classes=num_classes
            )

            vflat = torch.cat([v[k].reshape(-1) for k in keys])
            Hvflat = torch.cat([Hv[k].reshape(-1) for k in keys])

            H2 += float((Hvflat * Hvflat).sum().detach().cpu())
            trH += float((vflat * Hvflat).sum().detach().cpu())

        model.train()
        return math.sqrt(H2 / K), (trH / K)


def run_mrd_diagnostics(
    model: nn.Module,
    Xb: torch.Tensor,
    yb: torch.Tensor,
    K: int,
    prev_snapshot: Optional[Dict[str, torch.Tensor]],
    task: Task,
    num_classes: int = 10,
    # ---- extras
    add_loss: bool = True,
    estimate_model_curvature: bool = True,
    estimate_third_deriv: bool = False,
):
    """
    Full MRD diagnostics (task-aware) + optional model-curvature probes.

    Requires these symbols already defined in your file:
      - params_from_model, param_names, flatten_pytree, copy_params, rademacher_like
      - logits_fn_factory, apply_G, apply_R
      - frob_margin_from_trace_estimates
      - apply_model_H_mix, estimate_model_H_frob_mix, estimate_T_action_frob
        (the model-curvature code I gave you)

    Returns:
      out: dict with (always) P,batch,G_F,trG,R_F,m_F,mu,A,alpha and (optional) loss,H_mix_F,T_action_F
           and (optional, if prev_snapshot) Phi_align_F_fd,Phi_damp_F_fd,Phi_trans_F_fd
      snap: dict(params_prev=copy_params(params))
    """
    model.eval()
    keys = param_names(model)
    params = params_from_model(model)
    P = sum(params[k].numel() for k in keys)
    B = int(Xb.shape[0])

    # ---- task-aware forward for loss / FD weights
    logits_fn = logits_fn_factory(model, Xb, task)

    out: Dict[str, float] = dict(P=P, batch=B)

    # ============================================================
    # (0) Loss on this probe batch
    # ============================================================
    if add_loss:
        z = logits_fn(params)
        if task == "classification":
            loss = F.cross_entropy(z, yb.to(torch.long), reduction="mean")
        else:
            y = yb
            if y.ndim == 2 and y.shape[1] == 1:
                y = y[:, 0]
            y = y.to(z.dtype).view(-1)
            f = z[:, 0]
            loss = 0.5 * ((f - y) ** 2).mean()
        out["loss"] = float(loss.detach().cpu())

    # ============================================================
    # (1) Metric curvature: ||G||_F and tr(G) via Hutchinson
    # ============================================================
    GF2 = 0.0
    trG_acc = 0.0
    for _ in range(int(K)):
        v = rademacher_like(params, keys)
        Gv_tree, vTAv = apply_G(model, keys, params, Xb, v, task=task)
        Gv = flatten_pytree(Gv_tree, keys)
        GF2 += float((Gv * Gv).sum().detach().cpu())
        trG_acc += float(vTAv.detach().cpu())
    G_F = math.sqrt(GF2 / K)
    trG = trG_acc / K
    out["G_F"] = G_F
    out["trG"] = trG

    # ============================================================
    # (2) Residual curvature: ||R||_F via Hutchinson
    # ============================================================
    RF2 = 0.0
    for _ in range(int(K)):
        v = rademacher_like(params, keys)
        Rv_tree = apply_R(
            model, keys, params, Xb, yb, v, task=task, num_classes=num_classes
        )
        Rv = flatten_pytree(Rv_tree, keys)
        RF2 += float((Rv * Rv).sum().detach().cpu())
    R_F = math.sqrt(RF2 / K)
    out["R_F"] = R_F

    # ============================================================
    # (3) Frobenius margin m_F and components
    # ============================================================
    m_F, mu, A, alpha = frob_margin_from_trace_estimates(trG, G_F, R_F, P)
    out["m_F"] = float(m_F)
    out["mu"] = float(mu)
    out["A"] = float(A)
    out["alpha"] = float(alpha)

    # ============================================================
    # (4) Optional: "model curvature" probes (pure output-Hessian scale)
    # ============================================================
    if estimate_model_curvature:
        H_mix_F = estimate_model_H_frob_mix(
            model, Xb, task=task, K=K, num_classes=num_classes
        )
        out["H_mix_F"] = float(H_mix_F)

    if estimate_third_deriv:
        T_action_F = estimate_T_action_frob(
            model, Xb, yb, task=task, K=K, num_classes=num_classes
        )
        out["T_action_F"] = float(T_action_F)

    # ============================================================
    # (5) Optional: "model curvature" probes (pure output-Hessian scale)
    # ============================================================
    # out["H_loss_F"], out["trH_est"] = estimate_H_loss_F(model, Xb, yb, K, task, num_classes)
    # H2 = 0.0
    # trH = 0.0
    # for _ in range(K):
    #     v = rademacher_like(params, keys)
    #     Hv = apply_H_loss(model, keys, params, Xb, yb, v, task, num_classes)
    #     vflat  = flatten_pytree(v, keys)
    #     Hvflat = flatten_pytree(Hv, keys)
    #     H2  += float((Hvflat*Hvflat).sum())
    #     trH += float((vflat*Hvflat).sum())
    # out["H_loss_F"] = torch.sqrt(H2 / K)
    # out["trH_est"]  = trH / K

    # ============================================================
    # (6) Optional: finite-difference channel norms (requires snapshot)
    # ============================================================
    if prev_snapshot is not None and "params_prev" in prev_snapshot:
        params_prev = prev_snapshot["params_prev"]

        # ---- Phi_align v ≈ G_now v - G_prev v
        def Phi_align_Av(v_params):
            Gv_now, _ = apply_G(model, keys, params, Xb, v_params, task=task)
            Gv_prev, _ = apply_G(model, keys, params_prev, Xb, v_params, task=task)
            return {k: (Gv_now[k] - Gv_prev[k]) for k in keys}

        # ---- weights for damping/transport splits
        z_now = logits_fn(params)
        z_prev = logits_fn(params_prev)

        if task == "classification":
            C = int(num_classes)
            y_long = yb.to(torch.long)

            with torch.no_grad():
                p_now = torch.softmax(z_now, dim=1)
                p_prev = torch.softmax(z_prev, dim=1)
                y_oh = F.one_hot(y_long, num_classes=C).to(p_now.dtype)
                a_now = p_now - y_oh
                a_prev = p_prev - y_oh
                da = a_now - a_prev

            # Phi_damp v ≈ H_{da}(params_now) v
            def Phi_damp_Av(v_params):
                return apply_R(
                    model,
                    keys,
                    params,
                    Xb,
                    y_long,
                    v_params,
                    task=task,
                    num_classes=C,
                    a_override=da,
                )

            # Phi_trans v ≈ H_{a_prev}(params_now) v - H_{a_prev}(params_prev) v
            def Phi_trans_Av(v_params):
                Rv_now = apply_R(
                    model,
                    keys,
                    params,
                    Xb,
                    y_long,
                    v_params,
                    task=task,
                    num_classes=C,
                    a_override=a_prev,
                )
                Rv_prev = apply_R(
                    model,
                    keys,
                    params_prev,
                    Xb,
                    y_long,
                    v_params,
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

            with torch.no_grad():
                e_now = (z_now[:, 0] - y).detach()
                e_prev = (z_prev[:, 0] - y).detach()
                de = e_now - e_prev

            def Phi_damp_Av(v_params):
                return apply_R(
                    model,
                    keys,
                    params,
                    Xb,
                    yb,
                    v_params,
                    task=task,
                    a_override=de,
                )

            def Phi_trans_Av(v_params):
                Rv_now = apply_R(
                    model, keys, params, Xb, yb, v_params, task=task, a_override=e_prev
                )
                Rv_prev = apply_R(
                    model,
                    keys,
                    params_prev,
                    Xb,
                    yb,
                    v_params,
                    task=task,
                    a_override=e_prev,
                )
                return {k: (Rv_now[k] - Rv_prev[k]) for k in keys}

        Pa2 = 0.0
        Pd2 = 0.0
        Pt2 = 0.0
        for _ in range(int(K)):
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
