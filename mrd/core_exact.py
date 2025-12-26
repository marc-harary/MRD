from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.func import functional_call, jacrev, hessian


# ----------------------------
# pack/unpack model params to a flat vector
# ----------------------------
def _named_params(model: nn.Module) -> Tuple[List[str], List[torch.Tensor]]:
    names, tensors = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            names.append(n)
            tensors.append(p)
    return names, tensors


def _pack_params(
    names: List[str], tensors: List[torch.Tensor]
) -> Tuple[torch.Tensor, List[Tuple[int, int, torch.Size, torch.dtype, torch.device]]]:
    flats = []
    meta = []
    off = 0
    for t in tensors:
        n = t.numel()
        flats.append(t.reshape(-1))
        meta.append((off, n, t.shape, t.dtype, t.device))
        off += n
    return torch.cat(flats, dim=0), meta


def _unpack_params(theta_flat: torch.Tensor, names: List[str], meta) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for name, (off, n, shape, dtype, device) in zip(names, meta):
        out[name] = theta_flat[off : off + n].view(shape).to(dtype=dtype, device=device)
    return out


def _eig_extrema_sym(M: torch.Tensor, tag: str) -> Dict[str, float]:
    # assumes symmetric (all matrices here are symmetric by construction)
    evals = torch.linalg.eigvalsh(M)
    return {f"{tag}_lmin": float(evals[0].cpu()), f"{tag}_lmax": float(evals[-1].cpu())}


# ----------------------------
# exact eigen-extrema for H, G, R, and a model-Hessian proxy
# ----------------------------
def _eig_extrema_and_morse_sym(A: torch.Tensor, prefix: str, zero_tol: float = 1e-10) -> Dict[str, float]:
    """
    For symmetric A: returns lmin/lmax and Morse index (# strictly negative eigenvalues).
    `zero_tol` is only used to avoid counting tiny numerical noise as negative.
    """
    # exact symmetric eigvals
    evals = torch.linalg.eigvalsh(A)  # (P,)
    lmin = evals[0]
    lmax = evals[-1]
    morse = (evals < -zero_tol).sum()
    return {
        f"{prefix}_lmin": float(lmin.detach().cpu()),
        f"{prefix}_lmax": float(lmax.detach().cpu()),
        f"{prefix}_morse": float(morse.detach().cpu()),
    }


@torch.no_grad()
def exact_eig_extrema_HGR_Hztheta(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: nn.Module,
    zero_tol: float = 1e-10,
) -> Dict[str, float]:
    """
    Exact on the given batch for L(theta)=loss_fn(model(theta,x), y):

      H := ∂^2_theta L
      G := J^T (∂^2_z L) J                  (generalized Gauss–Newton / GGN)
      R := H - G                            (model-curvature remainder)

    Fourth matrix (to make “∂^2_theta z” a square matrix with eigenvalues):
      Hzθ := ∂^2_theta s(theta),  s(theta) := mean(vec(z(theta)))  (scalarized output)

    Returns:
      - loss
      - lmin/lmax + Morse index for H, G, R, Hzθ
      - P, D_out
    """
    model.eval()

    names, params = _named_params(model)
    theta0, meta = _pack_params(names, params)
    theta0 = theta0.detach().clone().requires_grad_(True)

    x = inputs.detach()
    y = targets.detach()

    with torch.enable_grad():
        # z_flat(theta) = vec(model(theta, x))
        def z_flat(theta: torch.Tensor) -> torch.Tensor:
            p = _unpack_params(theta, names, meta)
            z = functional_call(model, p, (x,))
            return z.reshape(-1)

        # L(theta)
        def L(theta: torch.Tensor) -> torch.Tensor:
            p = _unpack_params(theta, names, meta)
            z = functional_call(model, p, (x,))
            return loss_fn(z, y)

        # scalarized output s(theta) := mean(z_flat(theta))
        def s(theta: torch.Tensor) -> torch.Tensor:
            return z_flat(theta).mean()

        # ---- loss value at current theta
        loss_val = L(theta0).detach()

        # ---- exact Hessian H
        H = hessian(L)(theta0).detach()  # (P,P)

        # ---- exact G = J^T Hz J, where Hz = ∂^2_z L evaluated at current z
        z0 = z_flat(theta0).detach()
        z0_req = z0.clone().requires_grad_(True)

        z_shape = functional_call(model, _unpack_params(theta0, names, meta), (x,)).shape

        def L_of_z(z_flat_vec: torch.Tensor) -> torch.Tensor:
            z = z_flat_vec.view(z_shape)
            return loss_fn(z, y)

        Hz = hessian(L_of_z)(z0_req).detach()   # (D,D)
        J  = jacrev(z_flat)(theta0).detach()    # (D,P)
        G  = (J.T @ (Hz @ J)).detach()          # (P,P)

        # ---- exact remainder R
        R = (H - G).detach()

        # ---- exact “∂^2_theta z” as a square matrix via scalarization
        Hztheta = hessian(s)(theta0).detach()   # (P,P)

    out: Dict[str, float] = {}
    out["loss"] = float(loss_val.cpu())
    out.update(_eig_extrema_and_morse_sym(H, "H", zero_tol=zero_tol))
    out.update(_eig_extrema_and_morse_sym(G, "G", zero_tol=zero_tol))
    out.update(_eig_extrema_and_morse_sym(R, "R", zero_tol=zero_tol))
    out.update(_eig_extrema_and_morse_sym(Hztheta, "Hztheta", zero_tol=zero_tol))
    out["P"] = float(theta0.numel())
    out["D_out"] = float(z0.numel())

    model.train()
    return out
