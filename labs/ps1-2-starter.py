"""PS1.2 starter — the projection Lemma, in closed form and on a machine.

Self-contained by design (PESC parity, L3-UP-T1 §2F): this file needs no
notebook, no `common/` module and nothing from the U0 bootcamp — only torch,
numpy and matplotlib. It is intentionally lightweight and is designed to run
on a laptop CPU. The architecture and the optimizer are FIXED so the exercise
probes the theorem, not framework literacy. Change the settings only where a
part of PS1.2 asks you to.

The setting (UP.A §2.6, upgraded to the notation of UP.T1 §6):

    X ~ N(mu0, sigma0^2),   Y = X + sigma * eps,   eps ~ N(0, 1), eps ⊥ X,

and the closed-form regression function

    E[X | Y = y] = (sigma0^2 * y + sigma^2 * mu0) / (sigma0^2 + sigma^2).

Two regression targets are available: `x` (the target X itself) and `z`
(the surrogate Z = X + eta, eta ~ N(0, tau^2), eta ⊥ (X, Y)), which has the
same conditional mean given Y. Corollary (b) of the projection Lemma says
the two POPULATION risks differ by a constant that does not depend on theta
(here exactly tau^2 = 1). It says nothing about two finite training runs;
part (c) measures that.

One seed fixes the data draw AND the initialization. The X-run and the Z-run
of one seed share the same (x_i, y_i) sample and the same initial network;
the Z-run adds the independent eta_i. (The network is built BEFORE the data
are drawn, from the same generator, so that the extra eta draw of the Z-run
cannot move the initialization. The first version built it after, and the two
runs of one seed silently started from different networks — measured
2026-09-17, max parameter difference 1.94.) Two runs with DIFFERENT seeds
change both the data and the initialization, so their gap is a baseline for
ordinary run-to-run variability, not a control for sampling noise alone.

Curve discrepancies are measured on a dense grid between the empirical 0.5%
and 99.5% quantiles of the sampled y_i (601 points), so that the range is
defined by the data and reproducible from the seed.

Usage:
    python ps1-2-starter.py --target x --n 20000 --seed 0 --plot fit.png
    python ps1-2-starter.py --sweep            # part (c): seeds x sizes x targets
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch
import torch.nn as nn

# --- fixed by the assignment: the theorem is under test, not these choices ---
MU0, SIGMA0, SIGMA, TAU = 0.0, 1.0, 0.5, 1.0
HIDDEN, STEPS, LR = 64, 1000, 1e-3
GRID_Q = (0.005, 0.995)  # evaluation grid: empirical quantiles of the sampled y
N_GRID = 601


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def conditional_mean(y: torch.Tensor) -> torch.Tensor:
    """E[X | Y = y], PS1.2 part (a) in d = 1 (toolkit item 9)."""
    w = SIGMA0**2 / (SIGMA0**2 + SIGMA**2)
    return w * y + (1.0 - w) * MU0


def sample(n: int, target: str, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw n pairs (y, t) where t is the regression target X or Z."""
    x = MU0 + SIGMA0 * torch.randn(n, 1, generator=gen)
    y = x + SIGMA * torch.randn(n, 1, generator=gen)
    if target == "x":
        t = x
    elif target == "z":
        t = x + TAU * torch.randn(n, 1, generator=gen)  # eta ⊥ (X, Y)
    else:
        raise ValueError(f"target must be 'x' or 'z', got {target!r}")
    return y, t


def make_net(gen: torch.Generator) -> nn.Module:
    """Three Linear layers with Tanh between them, 64 hidden units."""
    torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=gen)))
    return nn.Sequential(
        nn.Linear(1, HIDDEN), nn.Tanh(),
        nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
        nn.Linear(HIDDEN, 1),
    )


def train(net: nn.Module, y: torch.Tensor, t: torch.Tensor) -> float:
    """Full-batch Adam on the square loss. Returns the final loss value."""
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    loss = torch.tensor(float("nan"))
    for _ in range(STEPS):
        opt.zero_grad()
        loss = ((net(y) - t) ** 2).mean()
        loss.backward()
        opt.step()
    return float(loss.detach())


def eval_grid(y: torch.Tensor) -> torch.Tensor:
    """Dense grid between the empirical 0.5% and 99.5% quantiles of y."""
    lo, hi = torch.quantile(y.flatten(), torch.tensor(GRID_Q)).tolist()
    return torch.linspace(lo, hi, N_GRID).reshape(-1, 1)


def max_gap(f: nn.Module, g, ys: torch.Tensor) -> float:
    """Largest |f - g| on the grid ys; g may be a closed form or another net."""
    with torch.no_grad():
        return float((f(ys) - g(ys)).abs().max())


def run_one(target: str, n: int, seed: int) -> tuple[nn.Module, float, torch.Tensor]:
    """One training run. Same seed => same (x, y) draw and same initialization
    for both targets; the network is built first so the Z-run's extra eta draw
    cannot change it. Returns the net, the final empirical loss, and the y
    sample (for the evaluation grid)."""
    set_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    net = make_net(gen)
    y, t = sample(n, target, gen)
    final_loss = train(net, y, t)
    return net, final_loss, y


def _summary(v: list[float]) -> str:
    return f"mean {np.mean(v):.4f}  median {np.median(v):.4f}  min {np.min(v):.4f}  max {np.max(v):.4f}"


def sweep(seeds: list[int], sizes: list[int]) -> None:
    """Part (c): the same architecture against X and against Z, several seeds,
    two sample sizes. Per size, prints across seeds: the gap between the X-fit
    and the Z-fit of the same seed (same data, same initialization; only the
    eta_i differ), the gap of each fit to the closed form, the gap between two
    X-fits of different seeds (a baseline for ordinary run-to-run variability:
    data AND initialization change), and the empirical loss difference Z - X
    of each seed beside the population offset tau^2. Nothing here asserts
    equality or a fixed ordering, because none is promised for one seed."""
    for n in sizes:
        fits_x, fits_z, loss_x, loss_z, grids = [], [], [], [], []
        for s in seeds:
            fx, lx, y = run_one("x", n, s)
            fz, lz, _ = run_one("z", n, s)
            fits_x.append(fx); fits_z.append(fz); loss_x.append(lx); loss_z.append(lz)
            grids.append(eval_grid(y))
        xz = [max_gap(fx, fz, g) for fx, fz, g in zip(fits_x, fits_z, grids)]
        x_truth = [max_gap(fx, conditional_mean, g) for fx, g in zip(fits_x, grids)]
        z_truth = [max_gap(fz, conditional_mean, g) for fz, g in zip(fits_z, grids)]
        # Cross-seed baseline: the two seeds have different y samples, so the
        # grid is the overlap of their two quantile ranges.
        xx = []
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                lo = max(float(grids[i][0]), float(grids[j][0]))
                hi = min(float(grids[i][-1]), float(grids[j][-1]))
                g = torch.linspace(lo, hi, N_GRID).reshape(-1, 1)
                xx.append(max_gap(fits_x[i], fits_x[j], g))
        diff = [lz - lx for lx, lz in zip(loss_x, loss_z)]
        print(f"n = {n:6d}   seeds {seeds}")
        print(f"  empirical final loss   X-target: {_summary(loss_x)}")
        print(f"                         Z-target: {_summary(loss_z)}")
        print(f"  loss difference Z - X per seed:  {[round(d, 4) for d in diff]}"
              f"   population offset tau^2 = {TAU**2:.2f}")
        print(f"  max gap X-fit vs Z-fit (same seed, same init):   {_summary(xz)}")
        print(f"  max gap X-fit vs closed form:                    {_summary(x_truth)}")
        print(f"  max gap Z-fit vs closed form:                    {_summary(z_truth)}")
        print(f"  max gap X-fit vs X-fit (other seed; baseline):   {_summary(xx)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--target", choices=["x", "z"], default="x")
    parser.add_argument("--n", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", default="", help="save the fit against the closed form")
    parser.add_argument("--sweep", action="store_true", help="part (c) protocol")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--sizes", default="2000,20000")
    args = parser.parse_args()

    if args.sweep:
        sweep([int(s) for s in args.seeds.split(",")],
              [int(s) for s in args.sizes.split(",")])
        return

    net, final_loss, y = run_one(args.target, args.n, args.seed)
    ys = eval_grid(y)
    gap = max_gap(net, conditional_mean, ys)
    print(f"target {args.target}  n {args.n}  seed {args.seed}  "
          f"final loss {final_loss:.4f}  max gap to closed form {gap:.4f}  "
          f"on y in [{float(ys[0]):.3f}, {float(ys[-1]):.3f}]")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        with torch.no_grad():
            fig, ax = plt.subplots(figsize=(6, 3.6))
            ax.plot(ys, conditional_mean(ys), "k--", label="closed form $E[X\\mid Y=y]$")
            ax.plot(ys, net(ys), label=f"trained $f_\\theta$ (target {args.target})")
        ax.set_xlabel("$y$"); ax.set_ylabel("$f(y)$"); ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(args.plot, dpi=150)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
