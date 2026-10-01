"""common/toy2d.py — the 2D toy datasets and their pictures (artifact: `toy2d-datasets`).

Built in U1.L1. **THE GENERATORS BELOW ARE FROZEN.** U2.L1, U2.L2, U3.L1 and
U3.L3 import this module by name and compare models trained on *these* samples;
the before/after comparisons of those sessions are only meaningful if the data
generator and the seed contract are identical. Downstream sessions never
redefine a generator. A change after the U1.L1 freeze is a breaking change and
needs a dated CHANGELOG note.

THE CONTRACT (U1.L1 notes section 1; L3-U1-L1 section 2 block 1):

    two_moons(n, seed)     -> Tensor (n, 2), float32
    checkerboard(n, seed)  -> Tensor (n, 2), float32
    rings(n, seed)         -> Tensor (n, 2), float32
    EXTENT = (-4.0, 4.0)   # default plotting extent, both axes: [-4, 4]^2

    density_heatmap(log_prob_fn, extent=EXTENT, n=200, ax=None)
        -> (xs, ys, logp)  # logp is an (n, n) numpy array of log p on the grid
    warped_grid(generative_fn, extent=EXTENT, ax=None, ...)
        -> list of (n_pts, 2) numpy arrays, one per drawn line

THREE PROPERTIES THE CONTRACT PROMISES:

1. **Pure, seeded functions.** Every generator owns a
   `numpy.random.default_rng(seed)`; nothing here reads or advances a global
   RNG. "Bit-identical" means: same `seed`, same pinned environment
   (`environment.yml`), same tensor. A U3 lab drawing `two_moons(4096, 0)`
   gets exactly the points U1.L1 trained on.

2. **Float32 outputs**, shape `(n, 2)`. The flows train in float32; the
   *checks* promote to float64 themselves (U0.L1's rule: train in float32,
   promote the check).

3. **The checkerboard has literal zero-density gaps.** It is a mixture of
   EIGHT separated squares of side `2 - CHECKER_GAP`, centred in the black
   cells of a 4x4 board on [-4, 4]^2, with NO Gaussian jitter. Two same-colour
   squares are diagonal neighbours and their closest corners are
   `CHECKER_GAP * sqrt(2)` apart; the white cells carry no mass at all. The
   support of the data distribution is therefore a union of eight disjoint
   closed squares. That hard-zero support is part of the mathematical
   contract: U1.T2 makes a statement about literal support, and this
   generator is its witness. U1.L1 only observes what a fitted density does
   in the gaps.

Run the self-check:  conda run -n torch-gpu python common/toy2d.py
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

EXTENT: tuple[float, float] = (-4.0, 4.0)

# --- two moons: the sklearn `make_moons` geometry, placed on the course extent
MOONS_RADIUS = 2.4       # raw make_moons has radius 1; scaled to fill [-4, 4]^2
MOONS_NOISE = 0.12       # isotropic Gaussian jitter in extent units (0.05 raw)

# --- checkerboard: 4x4 board on [-4, 4]^2; the 8 black cells carry the mass
CHECKER_CELL = 2.0       # side of one board cell
CHECKER_GAP = 0.5        # fixed nonzero gap: each square has side 2 - 0.5 = 1.5
CHECKER_CENTRES = tuple(
    (-3.0 + CHECKER_CELL * i, -3.0 + CHECKER_CELL * j)
    for i in range(4) for j in range(4) if (i + j) % 2 == 0
)   # exactly 8 centres, the black cells of the board

# --- rings: three concentric circles with a thin Gaussian radial profile
RINGS_RADII = (1.0, 2.2, 3.4)
RINGS_NOISE = 0.08

# --- house palette for the two picture helpers (matches course/_theme/)
UFRJ_BLUE = "#003366"
UFRJ_GRAY = "#6b7280"


def _as_float32(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))


def two_moons(n: int, seed: int) -> torch.Tensor:
    """Two interleaving half-circles, `n // 2` on the upper moon, the rest below.

    Same geometry as `sklearn.datasets.make_moons`, without the dependency,
    scaled by MOONS_RADIUS and centred so the two moons sit inside the extent:
    upper moon (R cos th - R/2, R sin th - R/4), lower moon its point
    reflection, th ~ Unif[0, pi], plus N(0, MOONS_NOISE^2 I) jitter.
    """
    rng = np.random.default_rng(seed)
    n_up = n // 2
    n_lo = n - n_up
    th_up = rng.uniform(0.0, np.pi, n_up)
    th_lo = rng.uniform(0.0, np.pi, n_lo)
    r = MOONS_RADIUS
    up = np.stack([r * np.cos(th_up) - r / 2, r * np.sin(th_up) - r / 4], axis=1)
    lo = np.stack([r / 2 - r * np.cos(th_lo), r / 4 - r * np.sin(th_lo)], axis=1)
    pts = np.concatenate([up, lo], axis=0)
    pts += rng.normal(0.0, MOONS_NOISE, size=pts.shape)
    return _as_float32(pts)


def checkerboard(n: int, seed: int) -> torch.Tensor:
    """Eight separated squares with hard-zero gaps — the U1.T2 witness.

    Each point picks one of the 8 squares uniformly (an equal-weight mixture)
    and is then uniform inside that square. There is no jitter of any kind:
    the density is exactly zero outside the union of the eight squares.
    """
    rng = np.random.default_rng(seed)
    which = rng.integers(0, len(CHECKER_CENTRES), size=n)
    centres = np.asarray(CHECKER_CENTRES, dtype=np.float64)[which]
    half = (CHECKER_CELL - CHECKER_GAP) / 2
    offs = rng.uniform(-half, half, size=(n, 2))
    return _as_float32(centres + offs)


def rings(n: int, seed: int) -> torch.Tensor:
    """Three concentric rings (equal-weight mixture), thin Gaussian radial jitter."""
    rng = np.random.default_rng(seed)
    which = rng.integers(0, len(RINGS_RADII), size=n)
    radius = np.asarray(RINGS_RADII, dtype=np.float64)[which]
    radius = radius + rng.normal(0.0, RINGS_NOISE, size=n)
    th = rng.uniform(0.0, 2 * np.pi, n)
    return _as_float32(np.stack([radius * np.cos(th), radius * np.sin(th)], axis=1))


GENERATORS: dict[str, Callable[[int, int], torch.Tensor]] = {
    "two_moons": two_moons,
    "checkerboard": checkerboard,
    "rings": rings,
}


def in_checkerboard_support(x: torch.Tensor) -> torch.Tensor:
    """Boolean (B,) mask: is each point inside one of the eight squares?

    Provided so that a session can measure what a model does IN THE GAPS
    (U1.L1 block 4) against the literal support, rather than by eye.
    """
    half = (CHECKER_CELL - CHECKER_GAP) / 2
    centres = torch.as_tensor(CHECKER_CENTRES, dtype=x.dtype, device=x.device)
    d = (x[:, None, :] - centres[None, :, :]).abs()          # (B, 8, 2)
    return (d <= half).all(dim=2).any(dim=1)


# ---------------------------------------------------------------------------
# Pictures. Both helpers evaluate a callable on a grid under no_grad and, when
# an axes object is given, draw on it in the house style. They return the raw
# arrays so a caller can assert on them (the U1.L1 figure script does).
# ---------------------------------------------------------------------------
@torch.no_grad()
def density_heatmap(
    log_prob_fn: Callable[[torch.Tensor], torch.Tensor],
    extent: tuple[float, float] = EXTENT,
    n: int = 200,
    ax: Any = None,
    chunk: int = 8192,
    **imshow_kw: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate `log_prob_fn` on an n x n grid over extent^2; draw exp(log p).

    `log_prob_fn` maps a float32 tensor (B, 2) to a tensor (B,). The grid is
    evaluated in chunks so that a 400 x 400 grid does not build one enormous
    graph-free tensor at once. Drawing uses `imshow` (one raster in the SVG),
    never `pcolormesh(shading="gouraud")`, whose thousands of gradients render
    differently in browsers and in PDF (invariants gate, check 4).
    """
    lo, hi = extent
    xs = np.linspace(lo, hi, n)
    ys = np.linspace(lo, hi, n)
    gx, gy = np.meshgrid(xs, ys)                          # gy varies down rows
    pts = torch.from_numpy(np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32))
    out = []
    for i in range(0, len(pts), chunk):
        out.append(log_prob_fn(pts[i:i + chunk]).detach().cpu().double())
    logp = torch.cat(out).numpy().reshape(n, n)
    if ax is not None:
        kw = {"cmap": "Blues", "interpolation": "nearest"}
        kw.update(imshow_kw)
        ax.imshow(np.exp(logp), origin="lower", extent=(lo, hi, lo, hi), **kw)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    return xs, ys, logp


@torch.no_grad()
def warped_grid(
    generative_fn: Callable[[torch.Tensor], torch.Tensor],
    extent: tuple[float, float] = EXTENT,
    ax: Any = None,
    n_lines: int = 17,
    n_pts: int = 200,
    source_box: float = 3.0,
    color: str = UFRJ_BLUE,
    linewidth: float = 0.7,
    alpha: float = 0.8,
) -> list[np.ndarray]:
    """Push a regular grid of the SOURCE space through psi and draw the image.

    `generative_fn` is the course map psi: source -> data, (B, 2) -> (B, 2).
    The grid covers [-source_box, source_box]^2 in the source coordinates
    (three standard deviations of p_0 on each side); the image is drawn on the
    data extent. This is the visual twin of U1.T1's local-volume picture: a
    cell that comes out large is a place where the density was divided by a
    large |det J|.
    """
    lo, hi = extent
    levels = np.linspace(-source_box, source_box, n_lines)
    ts = np.linspace(-source_box, source_box, n_pts)
    lines: list[np.ndarray] = []
    for c in levels:
        for line in (np.stack([np.full_like(ts, c), ts], 1),     # constant first coord
                     np.stack([ts, np.full_like(ts, c)], 1)):    # constant second coord
            img = generative_fn(torch.from_numpy(line.astype(np.float32)))
            lines.append(img.detach().cpu().double().numpy())
    if ax is not None:
        for img in lines:
            ax.plot(img[:, 0], img[:, 1], color=color, lw=linewidth, alpha=alpha)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    return lines


# ---------------------------------------------------------------------------
# Self-check: the contract's three properties, executed.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for name, gen in GENERATORS.items():
        a, b = gen(4096, 0), gen(4096, 0)
        assert a.dtype == torch.float32 and a.shape == (4096, 2), name
        assert torch.equal(a, b), f"{name}: not bit-identical across calls"
        assert not torch.equal(a, gen(4096, 1)), f"{name}: seed ignored"
        inside = ((a >= EXTENT[0]) & (a <= EXTENT[1])).all(dim=1).float().mean()
        print(f"{name:13s} mean {a.mean(0).tolist()}  inside extent {inside:.4f}")
    cb = checkerboard(100_000, 0)
    assert bool(in_checkerboard_support(cb).all()), "checkerboard leaked outside"
    assert len(CHECKER_CENTRES) == 8
    counts = torch.bincount(
        ((cb[:, 0] + 4) // 2).long() * 4 + ((cb[:, 1] + 4) // 2).long(), minlength=16
    )
    assert int((counts > 0).sum()) == 8, "mass in a white cell"
    print("checkerboard: 8 occupied cells, 0 points outside the squares")
    print("toy2d self-check OK")
