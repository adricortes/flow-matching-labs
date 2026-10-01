"""common/coupling.py — affine coupling flows from scratch (artifact: `coupling-layer-scratch`).

Built in U1.L1. This file is the course's REFERENCE implementation; the lab
notebook rebuilds `AffineCoupling` and `Flow` cell by cell (TODOs 2.1-3.1) and
then imports this module to cross-check itself. U1.L2 inherits the affine
algebra from here; PS1.4 is verified numerically against it.

THE CONTRACT (U1.L1 notes section 2; L3-U1-L1 section 2 block 2), frozen:

    AffineCoupling(d, mask, hidden=(64, 64), s_cap=2.0)
        forward(z) -> (x, logdet_fwd)      # psi   : source -> data
        inverse(x) -> (z, logdet_inv)      # psi^-1: data -> source
        # logdet_inv == -logdet_fwd, PER EXAMPLE, at matching points

    Permute(perm)
        forward / inverse as above; both log-dets are exactly 0

    Flow(layers)
        forward(z) -> (x, logdet_fwd)      # layers in order, log-dets summed
        inverse(x) -> (z, logdet_inv)      # layers reversed, each .inverse
        log_prob(x) -> (B,)                # log p_0(z) + logdet_inv
        sample(n)   -> (n, d)              # z ~ p_0, then forward
    Flow.realnvp(d=2, n_layers=6, hidden=(64, 64))   # alternating masks

DIRECTION. The course's generative map is psi: z -> x (U1.T1: t = 0 noise,
t = 1 data; `z` here is z_0 = x_0 of the T1 chain). `forward` IS psi.
Density evaluation runs `inverse`. Libraries such as Zuko expose the opposite
object, the density transform f: x -> z = psi^-1; U1.L1 block 5 translates.

THE SIGN, WRITTEN ONCE (U1.T1 eq. cov-model and Exercise 3):

    log p_theta(x) = log p_0(z) - log|det J_psi(z)|          z = psi^-1(x)
                   = log p_0(z) + log|det J_{psi^-1}(x)|
                   = log p_0(z) + logdet_inv

so `log_prob` ADDS the inverse log-det, which is MINUS the forward one. For
an affine coupling layer log|det J_psi| = sum_j s_j(x_A), read off the
conditioner pass; the inverse pass reads the same sum with the opposite sign,
because x_A = y_A is available on both sides.

NUMERICAL POLICY (U1.T1 section 5, "a numerical aside, for the lab"). The
raw scale output is BOUNDED before exponentiation,
    s = s_cap * tanh(raw / s_cap),   s in (-s_cap, s_cap),
so exp(s) can neither overflow nor divide the inverse by ~0 early in
training, when the conditioner is still a random network. `s_cap = 2`
allows local volume factors between e^-2 ~ 0.14 and e^2 ~ 7.4 per layer,
which is ample for the 2D toys. (Zuko's `MonotonicAffineTransform` bounds
the same quantity by a soft clip to |s| < -log(1e-3) ~ 6.9; block 5 reads it.)

INITIALIZATION, AND WHY THE HEAD IS NOT ZERO. The recipe card
(`optimizer-schedule-defaults`) zero-inits a last layer where "output ~ 0 at
start" is sensible, and an identity-at-start coupling layer IS sensible.
But a zero head makes every check in this session vacuous: analytic log-det
0, autodiff log-det 0, a perfect agreement measured on nothing (U0.L3 and
U0.L4 both hit this). The head therefore starts SMALL and NONZERO (default
init scaled by HEAD_INIT_SCALE): near the identity, and with a log-det the
checks can actually see. The lab asserts |logdet| > 0 before trusting any
agreement.

Run the self-check:  conda run -n torch-gpu python common/coupling.py
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn

HEAD_INIT_SCALE = 0.1
LOG_2PI = math.log(2.0 * math.pi)


def log_p0(z: torch.Tensor) -> torch.Tensor:
    """log N(z; 0, I_d) per example, shape (B,). p_0 is the course's fixed source."""
    d = z.shape[-1]
    return -0.5 * (z * z).sum(dim=-1) - 0.5 * d * LOG_2PI


def _mlp(d_in: int, d_out: int, hidden: Sequence[int]) -> nn.Sequential:
    """Conditioner: d_in -> hidden ... -> d_out, SiLU inside, small nonzero head."""
    dims = [d_in, *hidden]
    layers: list[nn.Module] = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b), nn.SiLU()]
    head = nn.Linear(dims[-1], d_out)
    with torch.no_grad():
        head.weight.mul_(HEAD_INIT_SCALE)
        head.bias.zero_()
    layers.append(head)
    return nn.Sequential(*layers)


class AffineCoupling(nn.Module):
    """One RealNVP affine coupling layer, in the GENERATIVE direction.

    `mask` is a boolean vector of length d: True marks the half x_A that the
    conditioner reads and the layer leaves untouched; False marks the half
    x_B that is scaled and shifted. Both halves must be non-empty.

        forward:  y_A = x_A,   y_B = x_B * exp(s(x_A)) + t(x_A)
        inverse:  x_A = y_A,   x_B = (y_B - t(y_A)) * exp(-s(y_A))
        logdet_fwd = sum_j s_j(x_A),   logdet_inv = -logdet_fwd
    """

    def __init__(
        self,
        d: int,
        mask: Sequence[bool] | torch.Tensor,
        hidden: Sequence[int] = (64, 64),
        s_cap: float | None = 2.0,
    ) -> None:
        super().__init__()
        mask_t = torch.as_tensor(mask, dtype=torch.bool)
        if mask_t.shape != (d,):
            raise ValueError(f"mask must have shape ({d},), got {tuple(mask_t.shape)}")
        if not (mask_t.any() and (~mask_t).any()):
            raise ValueError("mask must leave both halves non-empty")
        self.register_buffer("mask", mask_t)
        self.d_a = int(mask_t.sum())
        self.d_b = d - self.d_a
        # s_cap=None switches the bound OFF (raw log-scale). It exists so the
        # lab can measure what the bound buys; it is not a setting to train with.
        self.s_cap = None if s_cap is None else float(s_cap)
        self.cond = _mlp(self.d_a, 2 * self.d_b, hidden)   # two heads: raw s, t

    def _heads(self, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The conditioner on the untouched half: bounded log-scale and shift."""
        raw_s, t = self.cond(a).chunk(2, dim=-1)
        if self.s_cap is None:
            return raw_s, t
        s = self.s_cap * torch.tanh(raw_s / self.s_cap)
        return s, t

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """psi on one layer: (B, d) -> (x, logdet_fwd) with logdet_fwd of shape (B,)."""
        a, b = z[:, self.mask], z[:, ~self.mask]
        s, t = self._heads(a)
        x = z.new_empty(z.shape)
        x[:, self.mask] = a
        x[:, ~self.mask] = b * torch.exp(s) + t
        return x, s.sum(dim=-1)

    def inverse(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """psi^-1 on one layer: (B, d) -> (z, logdet_inv) with logdet_inv = -sum s."""
        a, b = x[:, self.mask], x[:, ~self.mask]
        s, t = self._heads(a)
        z = x.new_empty(x.shape)
        z[:, self.mask] = a
        z[:, ~self.mask] = (b - t) * torch.exp(-s)
        return z, -s.sum(dim=-1)


class Permute(nn.Module):
    """A fixed permutation of the coordinates. |det| = 1, so both log-dets are 0."""

    def __init__(self, perm: Sequence[int] | torch.Tensor) -> None:
        super().__init__()
        perm_t = torch.as_tensor(perm, dtype=torch.long)
        self.register_buffer("perm", perm_t)
        self.register_buffer("inv_perm", torch.argsort(perm_t))

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return z[:, self.perm], z.new_zeros(z.shape[0])

    def inverse(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x[:, self.inv_perm], x.new_zeros(x.shape[0])


class Flow(nn.Module):
    """A composition psi = psi_K o ... o psi_1 of layers with the contract above."""

    def __init__(self, layers: Sequence[nn.Module], d: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.d = d

    @classmethod
    def realnvp(
        cls,
        d: int = 2,
        n_layers: int = 6,
        hidden: Sequence[int] = (64, 64),
        s_cap: float = 2.0,
    ) -> Flow:
        """K affine coupling layers with ALTERNATING masks (layer k reads the
        half layer k-1 moved), so that every coordinate moves and, from K >= 3,
        every coordinate can influence every other (U1.T1 section 6)."""
        base = torch.arange(d) % 2 == 0
        layers = [AffineCoupling(d, base if k % 2 == 0 else ~base, hidden, s_cap)
                  for k in range(n_layers)]
        return cls(layers, d)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """psi: run the layers in order; log-dets add (U1.T1 eq. telescope)."""
        logdet = z.new_zeros(z.shape[0])
        x = z
        for layer in self.layers:
            x, ld = layer(x)
            logdet = logdet + ld
        return x, logdet

    def inverse(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """psi^-1: undo the layers in REVERSE order, each with its own inverse."""
        logdet = x.new_zeros(x.shape[0])
        z = x
        for layer in reversed(self.layers):
            z, ld = layer.inverse(z)
            logdet = logdet + ld
        return z, logdet

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """log p_theta(x) = log p_0(psi^-1(x)) + log|det J_{psi^-1}(x)|, per example."""
        z, logdet_inv = self.inverse(x)
        return log_p0(z) + logdet_inv

    @torch.no_grad()
    def sample(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """Draw z ~ p_0 = N(0, I_d) and push it through psi."""
        device = next(self.parameters()).device
        z = torch.randn(n, self.d, generator=generator, device=device)
        x, _ = self.forward(z)
        return x


# ---------------------------------------------------------------------------
# Independent witness: a PER-EXAMPLE Jacobian of psi, assembled by autodiff,
# in float64. `torch.func.jacrev` on the single-example map, batched with
# `vmap`, gives the (B, d, d) stack; `slogdet` of each is the number the
# analytic sum of scales must match. NEVER take one determinant of
# `torch.autograd.functional.jacobian(f, batch)`: that is the (B*d) x (B*d)
# Jacobian of the batched map, a different object (block-diagonal here, so
# its determinant is the PRODUCT over the batch — a sum of the per-example
# log-dets, which hides which example is wrong and overflows for large B).
# ---------------------------------------------------------------------------
def per_example_logdet(
    forward_fn, z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, d) -> (logabsdet (B,), sign (B,)) of the per-example Jacobian of `forward_fn`.

    `forward_fn` maps a single example (d,) -> (d,). Runs in the dtype of `z`;
    call it with a float64 tensor and a `.double()` copy of the model.
    """
    jac = torch.func.vmap(torch.func.jacrev(forward_fn))(z)      # (B, d, d)
    sign, logabsdet = torch.linalg.slogdet(jac)
    return logabsdet, sign


if __name__ == "__main__":
    torch.manual_seed(0)
    flow = Flow.realnvp(d=2, n_layers=6).double()
    z = torch.randn(64, 2, dtype=torch.float64)
    x, ld_f = flow(z)
    x, ld_f = x.detach(), ld_f.detach()
    z_back, ld_i = flow.inverse(x)
    assert torch.allclose(z_back, z, atol=1e-10), "round trip z -> x -> z"
    x_back, _ = flow(flow.inverse(x)[0])
    assert torch.allclose(x_back, x, atol=1e-10), "round trip x -> z -> x"
    assert torch.allclose(ld_f + ld_i, torch.zeros_like(ld_f), atol=1e-12), "cancellation"
    assert float(ld_f.abs().max()) > 1e-3, "log-det is ~0: the checks below are vacuous"

    def single(zi: torch.Tensor) -> torch.Tensor:
        return flow(zi[None])[0][0]

    ld_ad, sign = per_example_logdet(single, z)
    ld_ad = ld_ad.detach()
    assert bool((sign > 0).all()), "a coupling flow has positive determinant"
    gap = float((ld_ad - ld_f).abs().max())
    tol = 1e-10 + 1e-8 * float(ld_f.abs().max())
    assert gap <= tol, f"analytic vs autodiff log-det: {gap:.3e} > {tol:.3e}"
    lp = flow.log_prob(x).detach()
    assert torch.allclose(lp, log_p0(z) - ld_f, atol=1e-10), "log_prob sign"
    perm = Permute([1, 0]).double()
    y, ld_p = perm(x)
    assert float(ld_p.abs().max()) == 0.0 and torch.equal(perm.inverse(y)[0], x)
    print(f"coupling self-check OK — max|logdet| {float(ld_f.abs().max()):.4f}, "
          f"analytic vs autodiff gap {gap:.2e} (tol {tol:.2e})")
