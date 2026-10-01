"""common/multiscale_flow.py — the multi-scale image shell (U1.L2).

Built in U1.L2. The affine coupling ALGEBRA is U1.L1's (`common/coupling.py`,
artifact `coupling-layer-scratch`): same forward/inverse contract, same
tanh-bounded log-scale, same small nonzero head. What this module adds is the
IMAGE-SCALE ORCHESTRATION of RealNVP sections 3.4 and 3.6 and Glow section 3: a
convolutional conditioner, a squeeze, levels of coupling steps, and the
factor-out of half the state at the end of each level. Plus the logit
preprocessing of RealNVP section 4.1, written as a flow layer with the same
contract, so that its Jacobian can never be dropped by accident.

THE CONTRACT, inherited from U1.L1 and applied to every layer here:

    forward(z)  -> (x, logdet_fwd)      # psi   : source -> data (generative)
    inverse(x)  -> (z, logdet_inv)      # psi^-1: data -> source (density)
    # logdet_inv == -logdet_fwd, PER EXAMPLE, at matching points

    Logit(lam=1e-6)
        inverse(x) = (y, ld)   y = logit(lam + (1 - 2 lam) x),
                               ld = sum_j [log(1-2lam) - log a_j - log(1-a_j)]
        forward(y) = (x, -ld)  x = (sigmoid(y) - lam) / (1 - 2 lam)
    Squeeze2d(factor=2)
        inverse(x): (B,C,H,W) -> (B,4C,H/2,W/2)      density direction SQUEEZES
        forward(z): the exact inverse reshape          both log-dets are 0
    ImageAffineCoupling(channels, mask, hidden=64, s_cap=2.0)
        the U1.L1 layer with a boolean mask over CHANNELS and a conv conditioner
    MultiscaleFlow(in_shape, levels, steps, hidden, s_cap)
        inverse(y) -> ([z_1, ..., z_L], logdet_inv)   squeeze -> steps -> factor-out
        forward(zs) -> (y, logdet_fwd)
        log_prob(y) -> (B,)          sum_l log p_0(z_l) + logdet_inv
        sample(n, tau=1.0) -> y      z_l = tau * eps_l; tau != 1 is a HEURISTIC
        bookkeeping(y) -> rows       per-level table that reconciles with log_prob

TWO BOOKKEEPING FACTS the module makes executable (U1.T2 section 2, U1.L2
block 2). (1) A squeeze is a reshape: it permutes coordinates and changes no
volume, so its log-determinant is exactly zero, not approximately. (2) Factor-
out is a PARTITION of the current state, not a transform: half the channels
are sent to the latent collection as they are, the other half continue. It
contributes no Jacobian term; what it contributes is a base log-probability,
log p_0(z_l), summed over the factors at the end. The `bookkeeping` table has
one row per level and its rows add up to `log_prob` per example, to floating
point precision; the self-check asserts this.

THE INPUT OF THE FLOW IS y, THE LOGIT-SPACE TENSOR. `log_prob(y)` is log p_Y.
The density of the DATA x in [0,1)^D is log p_X(x) = log p_Y(T(x)) + ld_T(x),
with (y, ld_T) = Logit.inverse(x); the caller adds the Jacobian (U1.L2 TODO
4.1 does it by hand, and the cut package for PS1.4(d) hands out frozen
log p_Y values for the same reason). `bpd` in common/eval.py then divides by
D log 2, and the +8 of the frozen `bpd-protocol` is the caller's too.

DIRECTION AND NAMES. `forward` IS psi (source -> data), as in every course
module. The papers draw the density direction f = psi^-1, and so do the names
"squeeze" and "factor-out": the density pass squeezes and factors out; the
generative pass unsqueezes and merges. `Squeeze2d.inverse` therefore squeezes.

Run the self-check (from the repository root, as a module so that `common`
resolves):  conda run -n torch-gpu python -m common.multiscale_flow
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.coupling import HEAD_INIT_SCALE

LOG_2PI = math.log(2.0 * math.pi)

__all__ = [
    "log_p0_flat",
    "Logit",
    "Squeeze2d",
    "ImageAffineCoupling",
    "MultiscaleFlow",
    "count_params",
]


def log_p0_flat(z: torch.Tensor) -> torch.Tensor:
    """log N(z; 0, I) per example for a tensor of shape (B, ...); returns (B,)."""
    flat = z.flatten(1)
    return -0.5 * (flat * flat).sum(dim=1) - 0.5 * flat.shape[1] * LOG_2PI


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _numel(shape: Sequence[int]) -> int:
    n = 1
    for v in shape:
        n *= int(v)
    return n


# ===========================================================================
# Logit preprocessing, as a layer (RealNVP section 4.1; U1.L2 block 1)
# ===========================================================================


class Logit(nn.Module):
    """The logit preprocessing T_lambda, with the course contract.

    Density direction (`inverse`, x -> y):
        a = lam + (1 - 2 lam) x,   y = log a - log(1 - a),
        log|det J_T(x)| = sum_j [ log(1 - 2 lam) - log a_j - log(1 - a_j) ].
    Generative direction (`forward`, y -> x): x = (sigmoid(y) - lam) / (1 - 2 lam),
    with the opposite log-determinant, written through logsigmoid so that a
    large |y| never produces log(0).

    x must lie in [0, 1]; with x = (k + u) / 256 and u in [0, 1) it lies in
    [0, 1), and a in [lam, 1 - lam). lam = 1e-6 is the course constant.
    """

    def __init__(self, lam: float = 1e-6) -> None:
        super().__init__()
        if not 0.0 < lam < 0.5:
            raise ValueError(f"lam must be in (0, 0.5), got {lam}")
        self.lam = float(lam)
        self.log_scale = math.log(1.0 - 2.0 * self.lam)

    def inverse(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x in [0,1] -> (y, logdet_inv), logdet_inv of shape (B,)."""
        a = self.lam + (1.0 - 2.0 * self.lam) * x
        y = torch.log(a) - torch.log1p(-a)
        ld = self.log_scale - torch.log(a) - torch.log1p(-a)
        return y, ld.flatten(1).sum(dim=1)

    def forward(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """y -> (x, logdet_fwd), logdet_fwd = -logdet_inv at matching points."""
        a = torch.sigmoid(y)
        x = (a - self.lam) / (1.0 - 2.0 * self.lam)
        # log a + log(1 - a) = logsigmoid(y) + logsigmoid(-y): stable for any y.
        ld = F.logsigmoid(y) + F.logsigmoid(-y) - self.log_scale
        return x, ld.flatten(1).sum(dim=1)


# ===========================================================================
# Squeeze (RealNVP section 3.6; Glow section 3): a reshape, log-det 0
# ===========================================================================


class Squeeze2d(nn.Module):
    """Space-to-depth by `factor`: (B, C, H, W) <-> (B, C f^2, H/f, W/f).

    The density direction (`inverse`) squeezes; the generative direction
    (`forward`) unsqueezes. Both are permutations of the coordinates, so both
    log-determinants are exactly zero.

    Channel order after a squeeze with f = 2: for each input channel c, the
    four sub-pixels (0,0), (0,1), (1,0), (1,1) of every 2x2 block become the
    channels 4c, 4c+1, 4c+2, 4c+3. So on a one-channel image, a channel mask
    {0, 1} vs {2, 3} is a SPATIAL mask on the original pixels: it reads the top
    row of every 2x2 block and moves the bottom row.
    """

    def __init__(self, factor: int = 2) -> None:
        super().__init__()
        if factor < 2:
            raise ValueError("factor must be >= 2")
        self.factor = int(factor)

    def squeeze(self, x: torch.Tensor) -> torch.Tensor:
        f = self.factor
        b, c, h, w = x.shape
        if h % f or w % f:
            raise ValueError(f"spatial size {(h, w)} is not divisible by {f}")
        x = x.view(b, c, h // f, f, w // f, f)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        return x.view(b, c * f * f, h // f, w // f)

    def unsqueeze(self, z: torch.Tensor) -> torch.Tensor:
        f = self.factor
        b, c4, h, w = z.shape
        if c4 % (f * f):
            raise ValueError(f"{c4} channels are not a multiple of {f * f}")
        c = c4 // (f * f)
        z = z.view(b, c, f, f, h, w)
        z = z.permute(0, 1, 4, 2, 5, 3).contiguous()
        return z.view(b, c, h * f, w * f)

    def inverse(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Density direction: squeeze. log-det exactly 0."""
        return self.squeeze(x), x.new_zeros(x.shape[0])

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Generative direction: unsqueeze. log-det exactly 0."""
        return self.unsqueeze(z), z.new_zeros(z.shape[0])


# ===========================================================================
# Affine coupling on feature maps: U1.L1's layer, mask over channels
# ===========================================================================


def _conv_conditioner(c_in: int, c_out: int, hidden: int) -> nn.Sequential:
    """3x3 -> 1x1 -> 3x3 convolutions, SiLU inside, small NONZERO head.

    The head is scaled by HEAD_INIT_SCALE for the reason written in
    common/coupling.py: a head at exactly zero makes every log-det check
    vacuous (s = 0 identically) and the whole stack the identity.
    """
    body = nn.Sequential(
        nn.Conv2d(c_in, hidden, 3, padding=1), nn.SiLU(),
        nn.Conv2d(hidden, hidden, 1), nn.SiLU(),
    )
    head = nn.Conv2d(hidden, c_out, 3, padding=1)
    with torch.no_grad():
        head.weight.mul_(HEAD_INIT_SCALE)
        head.bias.zero_()
    return nn.Sequential(body, head)


class ImageAffineCoupling(nn.Module):
    """One affine coupling layer on (B, C, H, W), generative direction.

    `mask` is a boolean vector of length C: True marks the channels x_A the
    conditioner reads and leaves untouched; False marks the channels x_B that
    are scaled and shifted, at every spatial position. Same algebra as
    `common.coupling.AffineCoupling`; the conditioner is convolutional so that
    s and t at a position depend on a neighbourhood of x_A, not on all of it.

        forward:  y_A = x_A,   y_B = x_B * exp(s(x_A)) + t(x_A)
        inverse:  x_A = y_A,   x_B = (y_B - t(y_A)) * exp(-s(y_A))
        logdet_fwd = sum over (channels_B, H, W) of s,   logdet_inv = -logdet_fwd
    """

    def __init__(
        self,
        channels: int,
        mask: Sequence[bool] | torch.Tensor,
        hidden: int = 64,
        s_cap: float | None = 2.0,
    ) -> None:
        super().__init__()
        mask_t = torch.as_tensor(mask, dtype=torch.bool)
        if mask_t.shape != (channels,):
            raise ValueError(f"mask must have shape ({channels},), got {tuple(mask_t.shape)}")
        if not (mask_t.any() and (~mask_t).any()):
            raise ValueError("mask must leave both channel halves non-empty")
        self.register_buffer("mask", mask_t)
        self.c_a = int(mask_t.sum())
        self.c_b = channels - self.c_a
        self.s_cap = None if s_cap is None else float(s_cap)
        self.cond = _conv_conditioner(self.c_a, 2 * self.c_b, hidden)

    def _heads(self, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw_s, t = self.cond(a).chunk(2, dim=1)
        if self.s_cap is None:
            return raw_s, t
        return self.s_cap * torch.tanh(raw_s / self.s_cap), t

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a, b = z[:, self.mask], z[:, ~self.mask]
        s, t = self._heads(a)
        x = z.new_empty(z.shape)
        x[:, self.mask] = a
        x[:, ~self.mask] = b * torch.exp(s) + t
        return x, s.flatten(1).sum(dim=1)

    def inverse(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a, b = x[:, self.mask], x[:, ~self.mask]
        s, t = self._heads(a)
        z = x.new_empty(x.shape)
        z[:, self.mask] = a
        z[:, ~self.mask] = (b - t) * torch.exp(-s)
        return z, -s.flatten(1).sum(dim=1)


# ===========================================================================
# The multi-scale shell: squeeze -> coupling steps -> factor-out, per level
# ===========================================================================


class MultiscaleFlow(nn.Module):
    """RealNVP-style multi-scale flow on images, in logit space.

    Level l (density direction): squeeze, then `steps` coupling layers with
    alternating channel masks, then factor out the FIRST half of the channels
    as z_l (levels 1..L-1). The last level keeps everything as z_L. The latent
    z = (z_1, ..., z_L) has exactly as many entries as the input.

    `levels=1` is the one-level baseline of U1.L2 block 4: one squeeze, all
    `steps` at 14x14, no factor-out.
    """

    def __init__(
        self,
        in_shape: tuple[int, int, int] = (1, 28, 28),
        levels: int = 2,
        steps: int = 6,
        hidden: int = 64,
        s_cap: float | None = 2.0,
    ) -> None:
        super().__init__()
        if levels < 1 or steps < 1:
            raise ValueError("levels and steps must be >= 1")
        self.in_shape = tuple(int(v) for v in in_shape)
        self.n_levels = int(levels)
        self.steps = int(steps)
        self.squeeze = Squeeze2d(2)
        self.levels = nn.ModuleList()
        self.active_shapes: list[tuple[int, int, int]] = []   # after the squeeze
        self.latent_shapes: list[tuple[int, int, int]] = []   # what each level emits
        c, h, w = self.in_shape
        for lvl in range(self.n_levels):
            c, h, w = 4 * c, h // 2, w // 2
            self.active_shapes.append((c, h, w))
            base = torch.arange(c) < c // 2
            self.levels.append(nn.ModuleList(
                ImageAffineCoupling(c, base if k % 2 == 0 else ~base, hidden, s_cap)
                for k in range(self.steps)
            ))
            if lvl < self.n_levels - 1:
                self.latent_shapes.append((c // 2, h, w))
                c = c // 2                      # the second half continues
            else:
                self.latent_shapes.append((c, h, w))
        total = sum(cc * hh * ww for cc, hh, ww in self.latent_shapes)
        if total != _numel(self.in_shape):
            raise RuntimeError("latent size does not match the input size")

    # -- density direction ---------------------------------------------------
    def inverse(self, y: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        """y (B, C, H, W) -> ([z_1, ..., z_L], logdet_inv)."""
        h = y
        logdet = y.new_zeros(y.shape[0])
        zs: list[torch.Tensor] = []
        for lvl, steps in enumerate(self.levels):
            h, ld0 = self.squeeze.inverse(h)          # ld0 == 0, kept for the sum
            logdet = logdet + ld0
            for layer in steps:
                h, ld = layer.inverse(h)
                logdet = logdet + ld
            if lvl < self.n_levels - 1:
                z_l, h = h.chunk(2, dim=1)            # factor-out: a PARTITION
                zs.append(z_l)
            else:
                zs.append(h)
        return zs, logdet

    # -- generative direction ------------------------------------------------
    def forward(self, zs: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """[z_1, ..., z_L] -> (y, logdet_fwd). Undo the levels in reverse order."""
        if len(zs) != self.n_levels:
            raise ValueError(f"expected {self.n_levels} latent factors, got {len(zs)}")
        h = zs[-1]
        logdet = h.new_zeros(h.shape[0])
        for lvl in range(self.n_levels - 1, -1, -1):
            if lvl < self.n_levels - 1:
                h = torch.cat([zs[lvl], h], dim=1)    # merge: the partition, undone
            for layer in reversed(self.levels[lvl]):
                h, ld = layer(h)
                logdet = logdet + ld
            h, ld0 = self.squeeze(h)
            logdet = logdet + ld0
        return h, logdet

    def log_prob(self, y: torch.Tensor) -> torch.Tensor:
        """log p_Y(y) = sum_l log p_0(z_l) + logdet_inv, per example, shape (B,)."""
        zs, logdet_inv = self.inverse(y)
        base = torch.stack([log_p0_flat(z) for z in zs], dim=0).sum(dim=0)
        return base + logdet_inv

    @torch.no_grad()
    def sample(
        self, n: int, tau: float = 1.0, generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """y = psi(tau * eps), eps ~ N(0, I) per factor. tau = 1 samples the model;
        tau != 1 samples from the modified base N(0, tau^2 I): a heuristic."""
        ref = next(self.parameters())
        zs = [tau * torch.randn(n, *shape, generator=generator,
                                device=ref.device, dtype=ref.dtype)
              for shape in self.latent_shapes]
        y, _ = self.forward(zs)
        return y

    # -- the per-level table (U1.L2 TODO 2.3) --------------------------------
    @torch.no_grad()
    def bookkeeping(self, y: torch.Tensor) -> list[dict]:
        """One row per level, PER EXAMPLE tensors inside; rows sum to log_prob.

        Keys: level, active_shape, coupling_logdet (B,), latent_shape,
        base_logprob (B,). Squeeze rows are not listed: their log-det is 0.
        """
        rows: list[dict] = []
        h = y
        for lvl, steps in enumerate(self.levels):
            h = self.squeeze.squeeze(h)
            ld_level = y.new_zeros(y.shape[0])
            for layer in steps:
                h, ld = layer.inverse(h)
                ld_level = ld_level + ld
            if lvl < self.n_levels - 1:
                z_l, h = h.chunk(2, dim=1)
            else:
                z_l = h
            rows.append({
                "level": lvl + 1,
                "active_shape": self.active_shapes[lvl],
                "coupling_logdet": ld_level,
                "latent_shape": tuple(z_l.shape[1:]),
                "base_logprob": log_p0_flat(z_l),
            })
        return rows



# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    dt = torch.float64

    # 1. Squeeze: exact round trip, exact zero log-det, the channel order claim.
    sq = Squeeze2d(2)
    x = torch.randn(3, 1, 8, 8, dtype=dt)
    z, ld_s = sq.inverse(x)
    assert z.shape == (3, 4, 4, 4)
    assert torch.equal(sq.forward(z)[0], x), "squeeze round trip"
    assert float(ld_s.abs().max()) == 0.0 and float(sq.forward(z)[1].abs().max()) == 0.0
    assert torch.equal(z[:, 0], x[:, 0, 0::2, 0::2]) and torch.equal(z[:, 3], x[:, 0, 1::2, 1::2])

    # 2. Logit: round trip, cancellation, closed form vs autodiff on one pixel.
    lg = Logit(1e-6)
    xx = torch.rand(4, 1, 2, 2, dtype=dt)
    yy, ld_i = lg.inverse(xx)
    xb, ld_f = lg.forward(yy)
    assert float((xb - xx).abs().max()) < 1e-12 and float((ld_i + ld_f).abs().max()) < 1e-9
    one = torch.tensor([[0.3]], dtype=dt, requires_grad=True)
    yv = lg.inverse(one.view(1, 1, 1, 1))[0]
    dy = torch.autograd.grad(yv.sum(), one)[0]
    assert abs(float(torch.log(dy.abs())) - float(lg.inverse(one.detach().view(1, 1, 1, 1))[1])) < 1e-10

    # 3. One coupling layer: untouched channels, round trips, per-example
    #    autodiff witness in float64 on a 4x4x4 map (64 dims).
    layer = ImageAffineCoupling(4, [True, True, False, False], hidden=16).double()
    z4 = torch.randn(5, 4, 4, 4, dtype=dt)
    x4, ld4 = layer(z4)
    x4, ld4 = x4.detach(), ld4.detach()
    assert torch.equal(x4[:, :2], z4[:, :2]), "x_A must pass through unchanged"
    assert float(ld4.abs().max()) > 1e-3, "log-det ~0: the witness would be vacuous"
    with torch.no_grad():
        zb, ldb = layer.inverse(x4)
    assert float((zb - z4).abs().max()) < 1e-12 and float((ld4 + ldb).abs().max()) < 1e-12

    def single(v: torch.Tensor) -> torch.Tensor:
        """ONE example as a flat vector (64,) -> (64,), for jacrev."""
        return layer(v.view(1, 4, 4, 4))[0][0].flatten()

    jac = torch.func.vmap(torch.func.jacrev(single))(z4.flatten(1))   # (5, 64, 64)
    sign, logabs = torch.linalg.slogdet(jac)
    assert bool((sign > 0).all())
    gap = float((logabs.detach() - ld4.detach()).abs().max())
    assert gap < 1e-10 + 1e-8 * float(ld4.abs().max()), gap

    # 4. The shell: latent sizes add up, both round trips, and the per-level
    #    table reconciles with log_prob per example.
    flow = MultiscaleFlow((1, 8, 8), levels=2, steps=2, hidden=16).double()
    assert [s for s in flow.latent_shapes] == [(2, 4, 4), (8, 2, 2)]
    y = torch.randn(6, 1, 8, 8, dtype=dt)
    with torch.no_grad():
        zs, ld_inv = flow.inverse(y)
        assert sum(zz[0].numel() for zz in zs) == 64
        y_back, ld_fwd = flow.forward(zs)
        lp = flow.log_prob(y)
    assert float((y_back - y).abs().max()) < 1e-12 and float((ld_inv + ld_fwd).abs().max()) < 1e-12
    rows = flow.bookkeeping(y)
    total = sum(r["coupling_logdet"] + r["base_logprob"] for r in rows)
    assert float((total - lp).abs().max()) < 1e-10, "bookkeeping rows must sum to log_prob"
    ys = flow.sample(4, tau=0.7)
    assert ys.shape == (4, 1, 8, 8)
    one_level = MultiscaleFlow((1, 8, 8), levels=1, steps=4, hidden=16)
    assert one_level.latent_shapes == [(4, 4, 4)]
    print(f"multiscale_flow self-check OK — witness gap {gap:.1e}; "
          f"table vs log_prob {float((total - lp).abs().max()):.1e}; "
          f"{count_params(flow)} params in the 8x8 toy shell")
