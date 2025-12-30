import math
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Callable

from tqdm import tqdm
import torch


# ----------------------------
# Config
# ----------------------------
@dataclass
class SNEConfig:
    d_out: int = 2
    perplexity: float = 30.0
    n_steps: int = 1000
    lr: float = 0.5
    init: str = "pca"  # "pca" or "randn"
    center_X: bool = True
    center_Y_each_step: bool = True
    verbose_every: int = 100

    # Early exaggeration (t-SNE-style heuristic; affects attractive forces)
    early_exaggeration: float = 1.0          # e.g. 4.0, 12.0; 1.0 disables
    early_exaggeration_steps: int = 0        # e.g. 250; 0 disables

    # Exact eig tracking (VERY expensive; meant for very small N)
    track_every: int = 50
    max_N_for_exact: int = 256  # guardrail (still likely too high for dense Hz/J)
    eps: float = 1e-12


# ----------------------------
# f-divergence generator(s)
# ----------------------------
def kl_generator(u: torch.Tensor) -> torch.Tensor:
    """KL generator: f(u) = u log u (with u > 0)."""
    return u * u.log()


# ----------------------------
# Main class
# ----------------------------
class SNEGD:
    """
    Single-class SNE with Gaussian joint affinities (P) and global-softmax Q(Y).

    theta := vec(Y)
    z(theta) := vec(logits_ij) with logits_ij = -||y_i - y_j||^2 (diag masked)
    L(theta) := sum_{ij} Q_ij * f(P_ij / Q_ij)    (f-divergence form)

    H := ∂^2_theta L
    Hz := ∂^2_z L
    J := ∂_theta z
    G := J^T Hz J
    R := H - G

    WARNING: Exact J and Hz in dense form are enormous; intended only for tiny N.
    """

    def __init__(
        self,
        X: torch.Tensor,
        cfg: Optional[SNEConfig] = None,
        seed: int = 0,
        f_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        assert X.ndim == 2
        self.cfg = cfg or SNEConfig()
        self.device = X.device
        self.seed = int(seed)

        self.f_func = f_func if f_func is not None else kl_generator

        X = X.to(self.device)
        if self.cfg.center_X:
            X = X - X.mean(dim=0, keepdim=True)
        self.X = X

        self.N, self.D = X.shape

        self.sigmas: Optional[torch.Tensor] = None
        self.P_base: Optional[torch.Tensor] = None  # fixed symmetrized joint affinities
        self.Y: Optional[torch.Tensor] = None

        self.history: List[Dict[str, float]] = []

    # ----------------------------
    # Distances / probabilities
    # ----------------------------
    @staticmethod
    def pairwise_sq_dists(X: torch.Tensor) -> torch.Tensor:
        XX = (X * X).sum(dim=1, keepdim=True)
        return (XX + XX.T - 2.0 * (X @ X.T)).clamp_min(0.0)

    @torch.no_grad()
    def _conditional_gaussian_P(self, D2: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
        """
        Row-conditional Gaussian affinities P_{j|i} via softmax over j.
        """
        eps = self.cfg.eps
        N = D2.shape[0]
        denom = 2.0 * (sigmas * sigmas).view(N, 1) + eps
        logits = -D2 / denom
        logits.fill_diagonal_(-float("inf"))
        P = torch.softmax(logits, dim=1)
        # redundant but harmless normalization guard:
        P = P / (P.sum(dim=1, keepdim=True) + eps)
        P.fill_diagonal_(0.0)
        return P

    @torch.no_grad()
    def fit_sigmas_perplexity(
        self,
        max_iter: int = 60,
        tol: float = 1e-3,
        sigma_min: float = 1e-4,
        sigma_max: float = 1e4,
    ) -> torch.Tensor:
        """
        Binary-search per-point sigmas so Perp(P_{.|i}) ~= perplexity.
        """
        D2 = self.pairwise_sq_dists(self.X)
        target = float(self.cfg.perplexity)

        lo = torch.full((self.N,), math.log(sigma_min), device=self.device)
        hi = torch.full((self.N,), math.log(sigma_max), device=self.device)
        mid = (lo + hi) / 2.0

        for _ in tqdm(range(max_iter)):
            sig = mid.exp()
            P = self._conditional_gaussian_P(D2, sig)

            eps = self.cfg.eps
            p = P.clamp_min(eps)
            # base-2 entropy
            H = -(p * (p.log() / math.log(2.0))).sum(dim=1)
            perp = 2.0 ** H
            err = perp - target

            lo = torch.where(err < 0, mid, lo)
            hi = torch.where(err > 0, mid, hi)

            mid_new = (lo + hi) / 2.0
            if float(err.abs().max().item()) < tol:
                mid = mid_new
                break
            mid = mid_new

        self.sigmas = mid.exp()
        return self.sigmas

    @torch.no_grad()
    def build_P(self) -> torch.Tensor:
        """
        Symmetrized joint:
          P_ij = (P_{j|i}+P_{i|j})/(2N), diag=0, sum(P)=1.
        Stored as self.P_base.
        """
        if self.sigmas is None:
            self.fit_sigmas_perplexity()

        D2 = self.pairwise_sq_dists(self.X)
        Pcond = self._conditional_gaussian_P(D2, self.sigmas)

        eps = self.cfg.eps
        P = (Pcond + Pcond.T) / (2.0 * self.N)
        P.fill_diagonal_(0.0)
        P = P / (P.sum() + eps)

        self.P_base = P
        return P

    # ----------------------------
    # Low-dim model: logits and Q
    # ----------------------------
    @staticmethod
    def _logits_from_Y(Y: torch.Tensor) -> torch.Tensor:
        """
        logits_ij = -||y_i - y_j||^2, diagonal masked to -inf.
        """
        D2 = SNEGD.pairwise_sq_dists(Y)
        logits = -D2
        eye = torch.eye(Y.shape[0], device=Y.device, dtype=torch.bool)
        logits = logits.masked_fill(eye, -float("inf"))
        return logits

    def Q_from_Y(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Global-softmax Q over all (i,j), diagonal excluded by -inf.
        Numerically stable (no exp overflow + no inf*0 nan).
        """
        logits = self._logits_from_Y(Y)
        q = torch.softmax(logits.reshape(-1), dim=0).view(self.N, self.N)
        return q

    # ----------------------------
    # Objective
    # ----------------------------
    def _effective_P(self, t: int) -> torch.Tensor:
        """
        Early exaggeration heuristic: scale P by alpha for first T steps,
        then revert to base P.

        Note: If alpha != 1, P no longer sums to 1; this is intentional as a
        force-scaling heuristic (mirrors common t-SNE practice).
        """
        assert self.P_base is not None
        alpha = float(self.cfg.early_exaggeration)
        T = int(self.cfg.early_exaggeration_steps)

        if alpha != 1.0 and T > 0 and t <= T:
            return self.P_base * alpha
        return self.P_base

    def loss_from_Y(self, Y: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        """
        General f-divergence form:
          L = sum_{ij} Q_ij * f(P_ij / Q_ij)
        For KL, f(u)=u log u gives L = KL(P||Q) when P sums to 1.
        """
        eps = self.cfg.eps
        Q = self.Q_from_Y(Y)

        P_ = P.clamp_min(eps)
        Q_ = Q.clamp_min(eps)
        u = P_ / Q_
        return (Q_ * self.f_func(u)).sum()

    # ----------------------------
    # Init / run
    # ----------------------------
    @torch.no_grad()
    def init_Y(self) -> torch.Tensor:
        g = torch.Generator(device=self.device).manual_seed(self.seed)

        if self.cfg.init == "randn":
            Y = 1e-4 * torch.randn(self.N, self.cfg.d_out, generator=g, device=self.device)
        elif self.cfg.init == "pca":
            # PCA init via SVD
            U, S, Vh = torch.linalg.svd(self.X, full_matrices=False)
            Y = (self.X @ Vh.T[:, : self.cfg.d_out]).contiguous()
            Y = Y / (Y.std(dim=0, keepdim=True).clamp_min(1e-8))
            Y = 1e-2 * Y
        else:
            raise ValueError("init must be 'pca' or 'randn'")

        Y = Y - Y.mean(dim=0, keepdim=True)
        self.Y = Y.requires_grad_(True)
        return self.Y

    # ----------------------------
    # Exact eig tracking for G and R (dense)
    # ----------------------------
    @torch.no_grad()
    def _eig_minmax_sym(self, A: torch.Tensor) -> Tuple[float, float]:
        A = 0.5 * (A + A.T)  # enforce symmetry numerically
        evals = torch.linalg.eigvalsh(A)
        return float(evals[0].item()), float(evals[-1].item())

    def compute_GR_eigs_exact(self, Pmat: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Exact min/max eigenvalues of G and R wrt theta=vec(Y).

        WARNING: Dense jacobian and Hessians; intended for tiny N only.
        """
        assert self.Y is not None
        if self.P_base is None:
            raise RuntimeError("Call build_P() first.")

        if self.N > self.cfg.max_N_for_exact:
            raise RuntimeError(
                f"Exact G/R eigs are too expensive for N={self.N}; "
                f"set max_N_for_exact higher only if you know what you're doing."
            )

        eps = self.cfg.eps
        Y0 = self.Y
        Pmat = (Pmat if Pmat is not None else self.P_base).detach()

        theta0 = Y0.reshape(-1).detach().clone().requires_grad_(True)

        # z(theta): flattened logits (N*N)
        def z_flat(theta: torch.Tensor) -> torch.Tensor:
            Y = theta.view(self.N, self.cfg.d_out)
            logits = self._logits_from_Y(Y)  # (N,N) with -inf diag
            return logits.reshape(-1)

        # L(theta)
        def L(theta: torch.Tensor) -> torch.Tensor:
            Y = theta.view(self.N, self.cfg.d_out)
            Q = self.Q_from_Y(Y)
            P_ = Pmat.clamp_min(eps)
            Q_ = Q.clamp_min(eps)
            u = P_ / Q_
            return (Q_ * self.f_func(u)).sum()

        # H = ∂^2_theta L
        H = torch.func.hessian(L)(theta0)

        # Hz = ∂^2_z L(z) at z0 (stable masked-softmax)
        z0 = z_flat(theta0).detach()
        z0_req = z0.clone().requires_grad_(True)

        def L_of_z(zvec: torch.Tensor) -> torch.Tensor:
            logits = zvec.view(self.N, self.N)
            eye = torch.eye(self.N, device=logits.device, dtype=torch.bool)
            logits = logits.masked_fill(eye, -float("inf"))  # mask BEFORE softmax

            Q = torch.softmax(logits.reshape(-1), dim=0).view(self.N, self.N)

            P_ = Pmat.clamp_min(eps)
            Q_ = Q.clamp_min(eps)
            u = P_ / Q_
            return (Q_ * self.f_func(u)).sum()

        Hz = torch.func.hessian(L_of_z)(z0_req)

        # J = ∂_theta z
        J = torch.func.jacrev(z_flat)(theta0)

        G = J.T @ (Hz @ J)
        R = H - G

        G_lmin, G_lmax = self._eig_minmax_sym(G)
        R_lmin, R_lmax = self._eig_minmax_sym(R)

        return {"G_lmin": G_lmin, "G_lmax": G_lmax, "R_lmin": R_lmin, "R_lmax": R_lmax}

    # ----------------------------
    # Training loop
    # ----------------------------
    def step(self, opt: torch.optim.Optimizer, t: int) -> float:
        assert self.Y is not None
        if self.P_base is None:
            raise RuntimeError("Call build_P() first.")

        P_eff = self._effective_P(t)

        opt.zero_grad(set_to_none=True)
        loss = self.loss_from_Y(self.Y, P=P_eff)
        loss.backward()
        opt.step()

        if self.cfg.center_Y_each_step:
            with torch.no_grad():
                self.Y -= self.Y.mean(dim=0, keepdim=True)

        return float(loss.detach().cpu().item())

    def run(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          Y (N,d_out), loss history tensor at logged steps.
        """
        self.build_P()
        self.init_Y()
        opt = torch.optim.SGD([self.Y], lr=self.cfg.lr)

        loss_log: List[float] = []

        for t in range(1, self.cfg.n_steps + 1):
            loss_val = self.step(opt, t)

            record = (t == 1) or (t == self.cfg.n_steps)
            if self.cfg.verbose_every and (t % self.cfg.verbose_every == 0):
                record = True

            if record:
                loss_log.append(loss_val)
                print(f"[{t:5d}/{self.cfg.n_steps}] L = {loss_val:.6f}")

            if self.cfg.track_every and (t % self.cfg.track_every == 0):
                with torch.enable_grad():
                    P_eff = self._effective_P(t)
                    gr = self.compute_GR_eigs_exact(Pmat=P_eff)
                self.history.append({"step": float(t), "loss": float(loss_val), **gr})

        return self.Y.detach(), torch.tensor(loss_log)

    def history_dicts(self) -> List[Dict[str, float]]:
        return list(self.history)


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    X = torch.randn(64, 20, device="cuda")

    cfg = SNEConfig(
        d_out=2,
        perplexity=30.0,
        n_steps=1000,
        lr=0.5,
        init="pca",
        verbose_every=100,
        early_exaggeration=12.0,      # set to 1.0 to disable
        early_exaggeration_steps=250, # set to 0 to disable
        track_every=50,
        max_N_for_exact=64,
    )

    sne = SNEGD(X, cfg=cfg, seed=0)
    Y, loss_hist = sne.run()

    rows = sne.history_dicts()
    print(rows[:3], "...", rows[-1])
