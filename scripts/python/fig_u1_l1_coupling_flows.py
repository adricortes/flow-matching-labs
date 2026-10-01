"""fig_u1_l1_coupling_flows.py — U1.L1 figures: the reference solution, measured.

Trains the lab's reference RealNVP (common/coupling.py) on the frozen toy
datasets (common/toy2d.py) on CPU, with the course training stack
(common/train.py, the U0.T2 optimizer defaults), and draws what the lab asks
students to look at. Every number the notes, the deck and the notebook quote
is measured here and written to the JSON; nothing is hand-drawn.

Produces, into course/Figures/ (extension-less includes: html picks .svg,
beamer picks .pdf):

  u1-l1-toy2d            the three frozen generators, one panel each
  u1-l1-two-moons-fit    data vs model samples | density heatmap | warped grid
  u1-l1-checkerboard-fit same three panels; the eight support squares outlined
  u1-l1-nll-curves       validation NLL per epoch, both datasets; the exact
                         optimum log(18) drawn for the checkerboard
  u1-l1-loss-bound       training loss with and without the tanh bound on the
                         log-scale: (a) at the course defaults, (b) at lr 1e-2
  u1-l1-zuko-check       scratch density vs Zuko RealNVP density, same data,
                         same budget, same Trainer

  _outputs/u1l1_coupling_flows.json — the numbers.

STAGES AND --only. The full run trains two reference models, six Zuko/bound
sweep models and twelve short sweep runs (about five minutes on CPU).
`--only witness,moons` re-runs those stages, re-measures their keys, and
MERGES them into the existing JSON (MEMORY: a partial run must never
overwrite numbers it did not re-measure). Stage names: data, witness, moons,
checker, bound, zuko, jax, ps14.

HONESTY CHECKS (asserted, then recorded):
  1. per-example autodiff log-det (float64, jacrev+vmap) equals the analytic
     sum of scales on 64 validation points, for the untrained AND the trained
     flow, under atol + rtol; the untrained check first asserts the log-det is
     not ~0 (a zero head makes the check vacuous — U0.L3, U0.L4);
  2. both round trips and the per-example cancellation of the two log-dets;
  3. the batched-Jacobian pitfall, measured: slogdet of the (B*d)x(B*d)
     Jacobian of the batched map equals the SUM of the per-example log-dets;
  4. training decreases the validation NLL by more than 1 nat on both
     datasets (the notebook's relative checkpoint is pre-validated here);
  5. the checkerboard model puts a measured, nonzero fraction of its mass in
     the literal zero-density gaps — the OBSERVATION U1.T2 explains; the
     text must not call it a theorem;
  6. Zuko's transform runs x -> z: log_prob(x) == log p_0(T(x)) + ladj(x),
     asserted numerically, so the direction table is checked, not asserted;
  7. the JAX mirror's per-example log-det agrees with torch's on the same
     points, same weights.

FONT SIZES are large on purpose (14-16 pt): a figure included at the width of
a 16:9 Beamer frame lands at about half scale (U1.T1 qa-quarto M5).

BOUNDARY. No Glow, no autoregressive flow, no continuous time, and no
explanation of what the gaps show. The checkerboard numbers are recorded as
observations.

Run:  conda run -n torch-gpu python scripts/python/fig_u1_l1_coupling_flows.py
      conda run -n torch-gpu python scripts/python/fig_u1_l1_coupling_flows.py --only moons
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: identical output in CI, SSH, containers
matplotlib.rcParams["svg.hashsalt"] = "u1l1"  # deterministic SVG element ids
# Type 42 (TrueType) glyph outlines, NOT matplotlib's Type 3 default: Type 3
# fonts carry no hinting, so PDF viewers render small axis labels mushy.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.coupling import (  # noqa: E402
    AffineCoupling,
    Flow,
    Permute,
    log_p0,
    per_example_logdet,
)
from common.toy2d import (  # noqa: E402
    CHECKER_CELL,
    CHECKER_CENTRES,
    CHECKER_GAP,
    EXTENT,
    GENERATORS,
    checkerboard,
    density_heatmap,
    in_checkerboard_support,
    two_moons,
    warped_grid,
)
from common.train import Trainer, set_seed  # noqa: E402

# --- UFRJ palette (keep in sync with the theme surfaces in course/_theme/;
# --- verify with ./scripts/check-palette-sync.sh) ---
UFRJ_BLUE = "#003366"
UFRJ_BLUE_LIGHT = "#00447f"
UFRJ_ACCENT = "#d97a1f"
UFRJ_ACCENT_DARK = "#a3540c"
UFRJ_GRAY = "#6b7280"
UFRJ_GRAY_LIGHT = "#9aa2aa"
INK = "#26262a"
BLUES = LinearSegmentedColormap.from_list("ufrj_blues", ["#ffffff", "#b7c7dc", UFRJ_BLUE])

SEED = 0
OUT_DIR = ROOT / "course" / "Figures"
JSON_OUT = Path(__file__).resolve().parent / "_outputs" / "u1l1_coupling_flows.json"
RUN_DIR = Path(__file__).resolve().parent / "_outputs" / "u1l1-runs"   # gitignored

# --- the lab's training configuration: K = 6, the U0.T2 defaults, 8000 steps
CONFIG = {
    "n_layers": 6, "hidden": [64, 64], "s_cap": 2.0,
    "n_train": 4096, "n_val": 1024, "seed_train": 0, "seed_val": 1,
    "batch_size": 256, "epochs": 500, "lr": 3e-4, "weight_decay": 0.01,
    "warmup_frac": 0.05,
}
N_WITNESS = 64          # validation points for the per-example Jacobian check
ATOL, RTOL = 1e-10, 1e-8
SWEEP_LR = 1e-2         # where the bound is measured to matter
SWEEP_EPOCHS = 100
SWEEP_SEEDS = (0, 1, 2)
Y_TOP = 8.0             # y-limit of the bound figure; dead runs are marked at the top
FS = {"title": 16, "label": 15, "tick": 13, "legend": 13, "annot": 14}

torch.set_num_threads(4)


def set_all_seeds(seed: int) -> None:
    """Seed every RNG stream this script can touch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)                        # the course helper: torch + CUDA again


def apply_house_style(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(UFRJ_GRAY)
    ax.spines["bottom"].set_color(UFRJ_GRAY)
    ax.tick_params(colors=INK, labelsize=FS["tick"])
    ax.grid(axis=grid_axis, color="#DDDDDD", linewidth=0.6)
    ax.set_axisbelow(True)


def square_axes(ax: plt.Axes, title: str) -> None:
    ax.set_xlim(*EXTENT)
    ax.set_ylim(*EXTENT)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=FS["title"], color=INK)
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_yticks([-4, -2, 0, 2, 4])
    apply_house_style(ax)


def save_pair(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf"):
        path = OUT_DIR / f"{name}.{ext}"
        fig.savefig(path, transparent=True, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# The training closure the lab's TODO 3.2 asks for: the model returns
# log p_theta(x) and the loss is minus its mean. Everything else is the
# given Trainer (common/train.py v2), unchanged since U0.L2.
# ---------------------------------------------------------------------------
class LogProbModel(nn.Module):
    """Wraps any object with .log_prob so the Trainer's model(x) is log p(x)."""

    def __init__(self, flow: nn.Module) -> None:
        super().__init__()
        self.flow = flow

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.flow.log_prob(x)


class ZukoLogProb(nn.Module):
    """Same wrapper for a Zuko flow: flow() is the distribution, then log_prob."""

    def __init__(self, flow: nn.Module) -> None:
        super().__init__()
        self.flow = flow

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.flow().log_prob(x)


class NLL(nn.Module):
    """L(theta) = -E_{x_1 ~ q}[log p_theta(x_1)], the minibatch mean."""

    def forward(self, logp: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return -logp.mean()


def warmup_cosine(step: int, total: int, warmup: int) -> float:
    """Multiplier on the base LR — the U0.T2 default, stepped PER STEP."""
    if step < warmup:
        return step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * p))


def loaders(gen, cfg: dict, seed: int) -> tuple[DataLoader, DataLoader]:
    xtr = gen(cfg["n_train"], cfg["seed_train"])
    xva = gen(cfg["n_val"], cfg["seed_val"])
    # The Trainer was frozen for (x, y) batches in U0.L1; a flow has no labels,
    # so y is a float placeholder (and val_acc is nan, by design).
    tr = DataLoader(TensorDataset(xtr, torch.zeros(len(xtr))), batch_size=cfg["batch_size"],
                    shuffle=True, drop_last=True,
                    generator=torch.Generator().manual_seed(seed))
    va = DataLoader(TensorDataset(xva, torch.zeros(len(xva))), batch_size=1024)
    return tr, va


def train(model: nn.Module, gen, cfg: dict, seed: int, tag: str) -> dict:
    """The lab's training call: AdamW + warmup-cosine per step, given Trainer."""
    set_all_seeds(seed)
    tr, va = loaders(gen, cfg, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total = cfg["epochs"] * len(tr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: warmup_cosine(s, total, int(cfg["warmup_frac"] * total)))
    trainer = Trainer(model, opt, NLL(), "cpu", ckpt_dir=RUN_DIR / tag,
                      log_every=10**9, sched=sched)
    t0 = time.perf_counter()
    hist = trainer.fit(tr, va, cfg["epochs"])
    hist["seconds"] = time.perf_counter() - t0
    hist["steps"] = total
    return hist


def grid_points(n: int = 200) -> torch.Tensor:
    xs = torch.linspace(EXTENT[0], EXTENT[1], n)
    gx, gy = torch.meshgrid(xs, xs, indexing="xy")
    return torch.stack([gx.ravel(), gy.ravel()], 1)


# ---------------------------------------------------------------------------
# Stage: data
# ---------------------------------------------------------------------------
def stage_data() -> dict:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    out = {}
    for ax, (name, gen) in zip(axes, GENERATORS.items()):
        x = gen(4096, 0).numpy()
        ax.scatter(x[:, 0], x[:, 1], s=4, color=UFRJ_BLUE, alpha=0.5, edgecolors="none", rasterized=True)
        square_axes(ax, f"{name}(4096, seed=0)")
        out[name] = {"mean": x.mean(0).round(4).tolist(), "std": x.std(0).round(4).tolist()}
    # the checkerboard's literal support, outlined
    side = CHECKER_CELL - CHECKER_GAP
    for cx, cy in CHECKER_CENTRES:
        axes[1].add_patch(Rectangle((cx - side / 2, cy - side / 2), side, side,
                                    fill=False, lw=1.0, ls="--", ec=UFRJ_ACCENT_DARK))
    cb = checkerboard(100_000, 0)
    out["checkerboard"]["n_components"] = len(CHECKER_CENTRES)
    out["checkerboard"]["square_side"] = side
    out["checkerboard"]["gap"] = CHECKER_GAP
    out["checkerboard"]["corner_gap"] = round(CHECKER_GAP * math.sqrt(2), 6)
    out["checkerboard"]["points_outside_support_of_100000"] = int(
        (~in_checkerboard_support(cb)).sum())
    out["checkerboard"]["exact_nll_of_data"] = round(math.log(8 * side**2), 6)
    assert out["checkerboard"]["points_outside_support_of_100000"] == 0
    fig.tight_layout()
    save_pair(fig, "u1-l1-toy2d")
    return {"data": out}


# ---------------------------------------------------------------------------
# Stage: witness (untrained flow) — the per-example Jacobian ritual
# ---------------------------------------------------------------------------
def witness(flow: Flow, x: torch.Tensor) -> dict:
    """Analytic vs autodiff per-example log-det, both round trips, cancellation.

    `flow` must be float64 already; `x` are data-space points (B, 2) float64.
    """
    with torch.no_grad():
        z, ld_inv = flow.inverse(x)
        x_back, ld_fwd = flow(z)
        z_back, _ = flow.inverse(x_back)
    assert float(ld_fwd.abs().max()) > 1e-3, "log-det ~ 0: the witness would be vacuous"

    def single(zi: torch.Tensor) -> torch.Tensor:
        return flow(zi[None])[0][0]

    ld_ad, sign = per_example_logdet(single, z)
    ld_ad = ld_ad.detach()
    gap = float((ld_ad - ld_fwd).abs().max())
    tol = ATOL + RTOL * float(ld_fwd.abs().max())
    assert bool((sign > 0).all()) and gap <= tol, (gap, tol)
    rt_x = float((x_back - x).abs().max())
    rt_z = float((z_back - z).abs().max())
    cancel = float((ld_fwd + ld_inv).abs().max())
    assert rt_x < 1e-9 and rt_z < 1e-9 and cancel < 1e-12
    return {
        "n_points": int(len(x)), "max_abs_logdet": round(float(ld_fwd.abs().max()), 6),
        "autodiff_vs_analytic_gap": gap, "tolerance": tol,
        "round_trip_x": rt_x, "round_trip_z": rt_z, "logdet_cancellation": cancel,
    }


def batched_pitfall(flow: Flow, z: torch.Tensor) -> dict:
    """The (B*d)x(B*d) Jacobian of the batched map: one number, the SUM."""
    B, d = z.shape

    def batched(zz: torch.Tensor) -> torch.Tensor:
        return flow(zz)[0]

    J = torch.autograd.functional.jacobian(batched, z)          # (B, d, B, d)
    J2 = J.reshape(B * d, B * d)
    sign, logabsdet = torch.linalg.slogdet(J2)
    off_diag = J.clone()
    for i in range(B):
        off_diag[i, :, i, :] = 0.0
    with torch.no_grad():
        _, ld_fwd = flow(z)
    return {
        "B": B, "shape": list(J.shape),
        "max_abs_off_diagonal_block": float(off_diag.abs().max()),
        "slogdet_of_batched_map": float(logabsdet),
        "sum_of_per_example_logdets": float(ld_fwd.sum()),
        "det_of_batched_map": float(torch.linalg.det(J2)),
    }


def stage_witness() -> dict:
    set_all_seeds(SEED)
    flow = Flow.realnvp(2, CONFIG["n_layers"], CONFIG["hidden"], CONFIG["s_cap"]).double()
    x = two_moons(CONFIG["n_val"], CONFIG["seed_val"])[:N_WITNESS].double()
    out = {"untrained": witness(flow, x)}
    with torch.no_grad():
        z, _ = flow.inverse(x)
    out["batched_pitfall_B64"] = batched_pitfall(flow, z)
    out["batched_pitfall_B1024"] = batched_pitfall(
        flow, flow.inverse(two_moons(CONFIG["n_val"], CONFIG["seed_val"]).double())[0].detach())
    pit = out["batched_pitfall_B64"]
    assert pit["max_abs_off_diagonal_block"] == 0.0
    assert abs(pit["slogdet_of_batched_map"] - pit["sum_of_per_example_logdets"]) < 1e-8
    # a permutation is free, and undoes itself
    perm = Permute([1, 0]).double()
    y, ld_p = perm(x)
    assert float(ld_p.abs().max()) == 0.0 and torch.equal(perm.inverse(y)[0], x)
    out["permutation_logdet"] = 0.0
    return {"witness": out}


# ---------------------------------------------------------------------------
# Stages: moons, checker — train the reference, draw the three panels
# ---------------------------------------------------------------------------
def fit_panels(flow: Flow, gen, name: str, title: str, outline: bool) -> dict:
    x = gen(2000, 7).numpy()
    g = torch.Generator().manual_seed(11)
    xm = flow.sample(2000, generator=g).numpy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    axes[0].scatter(x[:, 0], x[:, 1], s=5, color=UFRJ_GRAY, alpha=0.45, edgecolors="none", rasterized=True,
                    label="data $x_1 \\sim q$")
    axes[0].scatter(xm[:, 0], xm[:, 1], s=5, color=UFRJ_ACCENT, alpha=0.6, edgecolors="none", rasterized=True,
                    label="model $\\psi^{\\theta}(x_0)$")
    # the legend sits BELOW the panel: the data fill every corner of the box
    axes[0].legend(fontsize=FS["legend"], loc="upper center", bbox_to_anchor=(0.5, -0.12),
                   ncol=2, frameon=False, markerscale=3, handletextpad=0.2)
    square_axes(axes[0], "samples")
    xs, ys, logp = density_heatmap(flow.log_prob, ax=axes[1], cmap=BLUES)
    square_axes(axes[1], "$p_\\theta$ on the grid")
    lines = warped_grid(lambda zz: flow(zz)[0], ax=axes[2])
    square_axes(axes[2], "the source grid under $\\psi^{\\theta}$")
    if outline:
        side = CHECKER_CELL - CHECKER_GAP
        for ax in axes[1:]:
            for cx, cy in CHECKER_CENTRES:
                ax.add_patch(Rectangle((cx - side / 2, cy - side / 2), side, side,
                                       fill=False, lw=1.0, ls="--", ec=UFRJ_ACCENT_DARK))
    fig.suptitle(title, fontsize=FS["title"], color=INK, y=1.02)
    fig.tight_layout()
    save_pair(fig, name)
    # normalisation on the extent: the model's mass inside the box
    cell = (xs[1] - xs[0]) * (ys[1] - ys[0])
    return {"mass_on_extent": float(np.exp(logp).sum() * cell), "n_grid_lines": len(lines)}


def stage_fit(name: str) -> dict:
    gen = two_moons if name == "moons" else checkerboard
    set_all_seeds(SEED)
    flow = Flow.realnvp(2, CONFIG["n_layers"], CONFIG["hidden"], CONFIG["s_cap"])
    n_params = sum(p.numel() for p in flow.parameters())
    hist = train(LogProbModel(flow), gen, CONFIG, SEED, f"ref-{name}")
    v = hist["val_loss"]
    out = {
        "config": CONFIG, "n_params": n_params, "steps": hist["steps"],
        "seconds": round(hist["seconds"], 1),
        "val_nll_first_epoch": round(v[0], 4), "val_nll_last": round(v[-1], 4),
        "val_nll_best": round(min(v), 4), "val_nll_curve": [round(a, 4) for a in v],
        "improvement_nats": round(v[0] - v[-1], 4),
    }
    assert out["improvement_nats"] > 1.0, out          # the notebook's relative checkpoint
    flow.eval()
    # the witness again, on the TRAINED flow, in float64
    flow64 = Flow.realnvp(2, CONFIG["n_layers"], CONFIG["hidden"], CONFIG["s_cap"]).double()
    flow64.load_state_dict(flow.state_dict())
    out["witness_trained"] = witness(
        flow64, gen(CONFIG["n_val"], CONFIG["seed_val"])[:N_WITNESS].double())
    title = ("two-moons, after 8000 steps" if name == "moons"
             else "gapped checkerboard, after 8000 steps")
    out.update(fit_panels(flow, gen, f"u1-l1-{'two-moons' if name == 'moons' else 'checkerboard'}-fit",
                          title, outline=(name == "checker")))
    if name == "checker":
        with torch.no_grad():
            g = grid_points(400)
            p = flow.log_prob(g).exp()
            inside = in_checkerboard_support(g)
            mass_gaps = float(p[~inside].sum() / p.sum())
            ratio = float(p[~inside].mean() / p[inside].mean())
            # the density along the segment between two diagonal squares'
            # closest corners, through the gap: the bridge, measured
            a = torch.tensor([-2.25, -2.25]); b = torch.tensor([-1.75, -1.75])
            seg = a + (b - a) * torch.linspace(0, 1, 101)[:, None]
            p_seg = flow.log_prob(seg).exp()
        out["gap_observation"] = {
            "mass_fraction_in_gaps": round(mass_gaps, 4),
            "mean_density_ratio_gaps_over_support": round(ratio, 4),
            "min_density_on_corner_to_corner_segment": float(p_seg.min()),
            "density_at_segment_ends": [float(p_seg[0]), float(p_seg[-1])],
            "uniform_density_on_support": round(1.0 / (8 * (CHECKER_CELL - CHECKER_GAP) ** 2), 6),
            "exact_nll_of_data": round(math.log(8 * (CHECKER_CELL - CHECKER_GAP) ** 2), 6),
            "nll_gap_to_optimum": round(v[-1] - math.log(8 * (CHECKER_CELL - CHECKER_GAP) ** 2), 4),
        }
        assert mass_gaps > 0.0 and float(p_seg.min()) > 0.0     # observed, not explained
    torch.save(flow.state_dict(), RUN_DIR / f"ref-{name}" / "flow_state.pt")
    return {name: out}


def stage_curves(data: dict) -> None:
    """Needs both fits in `data`; draws the two validation curves."""
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for key, label, color in (("moons", "two-moons", UFRJ_BLUE),
                              ("checker", "gapped checkerboard", UFRJ_ACCENT)):
        v = data[key]["val_nll_curve"]
        ax.plot(range(1, len(v) + 1), v, color=color, lw=2.0, label=label)
    opt = data["checker"]["gap_observation"]["exact_nll_of_data"]
    ax.axhline(opt, color=UFRJ_ACCENT_DARK, lw=1.2, ls="--")
    ax.text(len(v) * 0.99, opt + 0.05, "$\\log 18 = 2.890$: the NLL of $q$ itself (checkerboard)",
            ha="right", va="bottom", fontsize=FS["annot"], color=UFRJ_ACCENT_DARK)
    ax.set_xlabel("epoch (16 steps each)", fontsize=FS["label"])
    ax.set_ylabel("validation NLL (nats)", fontsize=FS["label"])
    ax.set_ylim(1.8, 7.2)
    ax.legend(fontsize=FS["legend"], frameon=False)
    apply_house_style(ax, "y")
    fig.tight_layout()
    save_pair(fig, "u1-l1-nll-curves")


# ---------------------------------------------------------------------------
# Stage: bound — what forgetting the tanh bound does, measured
# ---------------------------------------------------------------------------
def stage_bound() -> dict:
    out = {"sweep_lr": SWEEP_LR, "sweep_epochs": SWEEP_EPOCHS, "seeds": list(SWEEP_SEEDS)}
    runs: dict[str, dict] = {}
    for lr_name, lr in (("default", CONFIG["lr"]), ("high", SWEEP_LR)):
        for bound in (True, False):
            for seed in SWEEP_SEEDS:
                cfg = dict(CONFIG, lr=lr, epochs=SWEEP_EPOCHS)
                set_all_seeds(seed)
                flow = Flow.realnvp(2, cfg["n_layers"], cfg["hidden"],
                                    cfg["s_cap"] if bound else None)
                hist = train(LogProbModel(flow), two_moons, cfg, seed,
                             f"bound-{lr_name}-{'tanh' if bound else 'raw'}-{seed}")
                t = hist["train_loss"]
                nan_epoch = next((i + 1 for i, a in enumerate(t) if a != a), None)
                runs[f"{lr_name}/{'tanh' if bound else 'raw'}/{seed}"] = {
                    "train_curve": [None if a != a else round(a, 4) for a in t],
                    "first_nan_epoch": nan_epoch,
                    "val_nll_best": round(min(a for a in hist["val_loss"] if a == a), 4),
                }
    out["runs"] = {k: {kk: vv for kk, vv in v.items() if kk != "train_curve"}
                   for k, v in runs.items()}
    for lr_name in ("default", "high"):
        for kind in ("tanh", "raw"):
            n_nan = sum(runs[f"{lr_name}/{kind}/{s}"]["first_nan_epoch"] is not None
                        for s in SWEEP_SEEDS)
            out[f"{lr_name}_{kind}_runs_with_nan"] = n_nan
    # the figure: (a) course defaults, (b) lr = 1e-2
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    for ax, lr_name, lr in ((axes[0], "default", CONFIG["lr"]), (axes[1], "high", SWEEP_LR)):
        for kind, color, label in (("tanh", UFRJ_BLUE, "bounded: $s = 2\\tanh(\\cdot/2)$"),
                                   ("raw", UFRJ_ACCENT, "unbounded: $s$ = raw output")):
            for i, seed in enumerate(SWEEP_SEEDS):
                t = runs[f"{lr_name}/{kind}/{seed}"]["train_curve"]
                ep = [j + 1 for j, a in enumerate(t) if a is not None]
                vals = [a for a in t if a is not None]
                ax.plot(ep, vals, color=color, lw=1.6, alpha=0.85,
                        label=label if i == 0 else None)
                if len(ep) < len(t):      # the run died: mark where (clipped to the axes)
                    ax.plot(ep[-1], min(vals[-1], Y_TOP - 0.25), "x", color=color, ms=11, mew=2.5)
        ax.set_title(f"learning rate {lr:g}" + (" (the course default)" if lr_name == "default" else ""),
                     fontsize=FS["title"], color=INK)
        ax.set_xlabel("epoch", fontsize=FS["label"])
        apply_house_style(ax, "y")
    axes[0].set_ylabel("training NLL (nats)", fontsize=FS["label"])
    axes[0].set_ylim(1.8, Y_TOP)
    axes[0].legend(fontsize=FS["legend"], frameon=False)
    axes[1].text(0.98, 0.95, "$\\times$ = the run produced NaN and stopped",
                 transform=axes[1].transAxes, ha="right", va="top", fontsize=FS["annot"], color=INK)
    fig.tight_layout()
    save_pair(fig, "u1-l1-loss-bound")
    return {"bound": out}


# ---------------------------------------------------------------------------
# Stage: zuko — the library cross-check, same data, same budget, same Trainer
# ---------------------------------------------------------------------------
def stage_zuko(ref_state: Path) -> dict:
    import zuko

    version = zuko.__version__
    set_all_seeds(SEED)
    zflow = zuko.flows.RealNVP(features=2, transforms=CONFIG["n_layers"],
                               hidden_features=tuple(CONFIG["hidden"]))
    hist = train(ZukoLogProb(zflow), two_moons, CONFIG, SEED, "zuko-moons")
    zflow.eval()
    v = hist["val_loss"]
    # the direction, asserted: T = flow.transform() runs x -> z
    xv = two_moons(CONFIG["n_val"], CONFIG["seed_val"])
    with torch.no_grad():
        T = zflow.transform()
        z = T(xv)
        ladj = T.log_abs_det_jacobian(xv, z)
        lp_manual = log_p0(z) + ladj
        lp_lib = zflow().log_prob(xv)
        x_back = T.inv(z)
    dialect_gap = float((lp_manual - lp_lib).abs().max())
    assert dialect_gap < 1e-4, dialect_gap
    # scratch reference on the same validation set
    flow = Flow.realnvp(2, CONFIG["n_layers"], CONFIG["hidden"], CONFIG["s_cap"])
    flow.load_state_dict(torch.load(ref_state))
    flow.eval()
    with torch.no_grad():
        nll_scratch = float(-flow.log_prob(xv).mean())
        nll_zuko = float(-lp_lib.mean())
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2))
    density_heatmap(flow.log_prob, ax=axes[0], cmap=BLUES)
    square_axes(axes[0], f"scratch  (val NLL {nll_scratch:.3f})")
    density_heatmap(lambda pts: zflow().log_prob(pts), ax=axes[1], cmap=BLUES)
    square_axes(axes[1], f"Zuko RealNVP  (val NLL {nll_zuko:.3f})")
    fig.tight_layout()
    save_pair(fig, "u1-l1-zuko-check")
    n_params = sum(p.numel() for p in zflow.parameters())
    return {"zuko": {
        "version": version, "n_params": n_params, "seconds": round(hist["seconds"], 1),
        "val_nll_first_epoch": round(v[0], 4), "val_nll_last": round(v[-1], 4),
        "val_nll_scratch_same_val": round(nll_scratch, 4), "val_nll_zuko_same_val": round(nll_zuko, 4),
        "dialect_gap": dialect_gap, "inv_round_trip": float((x_back - xv).abs().max()),
        "slope": 1e-3, "log_scale_bound": round(-math.log(1e-3), 4),
    }}


# ---------------------------------------------------------------------------
# Stage: jax — the same coupling map in JAX, per-example log-det by jacfwd+vmap
# ---------------------------------------------------------------------------
def stage_jax(ref_state: Path) -> dict:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    flow = Flow.realnvp(2, CONFIG["n_layers"], CONFIG["hidden"], CONFIG["s_cap"]).double()
    flow.load_state_dict(torch.load(ref_state))
    params = []
    for layer in flow.layers:
        ws = [(jnp.asarray(m.weight.detach().numpy()), jnp.asarray(m.bias.detach().numpy()))
              for m in layer.cond if isinstance(m, nn.Linear)]
        params.append((bool(layer.mask[0]), ws, layer.s_cap))

    def mlp(ws, a):
        h = a
        for i, (w, b) in enumerate(ws):
            h = h @ w.T + b
            if i < len(ws) - 1:
                h = jax.nn.silu(h)
        return h

    def psi(z):                       # one example, (2,) -> (2,)
        x = z
        for first_is_a, ws, s_cap in params:
            ia, ib = (0, 1) if first_is_a else (1, 0)
            raw = mlp(ws, x[ia:ia + 1])
            s = s_cap * jnp.tanh(raw[0] / s_cap)
            t = raw[1]
            xb = x[ib] * jnp.exp(s) + t
            x = x.at[ib].set(xb)
        return x

    x = two_moons(CONFIG["n_val"], CONFIG["seed_val"])[:N_WITNESS].double()
    with torch.no_grad():
        z, _ = flow.inverse(x)
        _, ld_torch = flow(z)
    zj = jnp.asarray(z.numpy())
    J = jax.vmap(jax.jacfwd(psi))(zj)                       # (B, 2, 2)
    _, ld_jax = jnp.linalg.slogdet(J)
    xj = jax.vmap(psi)(zj)
    gap = float(jnp.abs(ld_jax - jnp.asarray(ld_torch.numpy())).max())
    fwd_gap = float(jnp.abs(xj - jnp.asarray(x.numpy())).max())
    assert gap < 1e-9 and fwd_gap < 1e-9, (gap, fwd_gap)
    return {"jax": {"version": jax.__version__, "n_points": N_WITNESS,
                    "logdet_gap_vs_torch": gap, "forward_gap_vs_torch": fwd_gap}}


# ---------------------------------------------------------------------------
# Stage: ps14 — PS1.4(b) written by hand, checked against the module (d = 4)
# ---------------------------------------------------------------------------
def stage_ps14() -> dict:
    set_all_seeds(3)
    mA = torch.tensor([True, True, False, False])
    layers = [AffineCoupling(4, mA), AffineCoupling(4, ~mA), AffineCoupling(4, mA)]
    flow = Flow(layers, 4).double()
    x1 = torch.randn(16, 4, dtype=torch.float64)
    with torch.no_grad():
        # the inverse chain, by hand, in the order it is computed
        z2, _ = layers[2].inverse(x1)
        z1, _ = layers[1].inverse(z2)
        z0, _ = layers[0].inverse(z1)
        s3, _ = layers[2]._heads(z2[:, mA])       # psi_3 reads (xi_1, xi_2) of z_2
        s2, _ = layers[1]._heads(z1[:, ~mA])      # psi_2 reads (xi_3, xi_4) of z_1
        s1, _ = layers[0]._heads(z0[:, mA])       # psi_1 reads (xi_1, xi_2) of z_0
        by_hand = log_p0(z0) - s1.sum(1) - s2.sum(1) - s3.sum(1)
        by_module = flow.log_prob(x1)
    gap = float((by_hand - by_module).abs().max())
    assert gap < 1e-12, gap
    return {"ps14": {"d": 4, "n_points": 16, "gap_by_hand_vs_module": gap}}


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated stage names to re-run")
    args = ap.parse_args()
    all_stages = ["data", "witness", "moons", "checker", "bound", "zuko", "jax", "ps14"]
    stages = [s.strip() for s in args.only.split(",") if s.strip()] or all_stages
    unknown = set(stages) - set(all_stages)
    if unknown:
        raise SystemExit(f"unknown stage(s): {sorted(unknown)}")

    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(JSON_OUT.read_text()) if JSON_OUT.exists() else {}
    ref_state = RUN_DIR / "ref-moons" / "flow_state.pt"

    t0 = time.perf_counter()
    for stage in stages:
        print(f"--- stage {stage}")
        if stage == "data":
            data.update(stage_data())
        elif stage == "witness":
            data.update(stage_witness())
        elif stage in ("moons", "checker"):
            data.update(stage_fit(stage))
        elif stage == "bound":
            data.update(stage_bound())
        elif stage == "zuko":
            data.update(stage_zuko(ref_state))
        elif stage == "jax":
            data.update(stage_jax(ref_state))
        elif stage == "ps14":
            data.update(stage_ps14())
        JSON_OUT.write_text(json.dumps(data, indent=2))   # merge after every stage
    if "moons" in data and "checker" in data:
        stage_curves(data)
    data["_meta"] = {"seed": SEED, "torch": torch.__version__,
                     "wall_clock_s_this_run": round(time.perf_counter() - t0, 1),
                     "stages_this_run": stages}
    JSON_OUT.write_text(json.dumps(data, indent=2))
    print(f"wrote {JSON_OUT}")


if __name__ == "__main__":
    main()
