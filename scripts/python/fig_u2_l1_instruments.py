"""fig_u2_l1_instruments.py - U2.L1 figures and numbers: the instruments, measured.

Trains the lab's reference Neural ODE classifier on the frozen two-moons
generator (common/toy2d.py) and measures, with each instrument the lab
installs, what the notes, the deck and the notebook quote. Nothing is
hand-drawn and no number is typed: every value is computed here, asserted
where the text makes a claim about it, and written to the JSON.

Produces, into course/Figures/ (extension-less includes: html picks .svg,
beamer picks .pdf):

  u2-l1-trajectories    (a) input space and decision regions; (b) the ODE
                        state in R^2 along t, and the head's line at t = 1
  u2-l1-accuracy-nfe    (a) test accuracy against NFE; (b) terminal-state
                        error against NFE, for euler / rk4 / dopri5
  u2-l1-nfe-tolerance   dopri5 NFE against rtol (atol = rtol * 1e-2)
  u2-l1-nfe-drift       forward NFE per training step, three seeds, with the
                        training loss underneath
  u2-l1-memory          peak GPU memory of one forward-and-backward pass against
                        the NFE of the forward solve: direct backprop vs the adjoint

  common/weights/u2l1-node-classifier.{pt,json}  the rescue checkpoint
  _outputs/u2l1_instruments.json                 the numbers

STAGES AND --only. Stage names: train, trajectories, crosscheck, sweep, drift,
memory, fidelity, jax, init, replot, init-summary (re-classifies the recorded init rows,
no training). `train` writes the checkpoint that every later stage loads, so
it runs first in a full run. `--only sweep,fidelity` re-runs those stages and
MERGES their keys into the existing JSON (MEMORY 2026-09-14: a partial run
must never overwrite numbers it did not re-measure; wall-clock values are the
ones that move between runs).

HONESTY CHECKS (asserted, then recorded):
  1. the labels read from the row order of two_moons(n, seed) are the moons:
     the upper moon (label 0) has the larger mean second coordinate;
  2. the reference model reaches >= 0.99 test accuracy at the dopri5
     defaults, so the notebook's checkpoint is pre-validated;
  3. the NFE counter is checked against the solver's own arithmetic: euler
     spends N calls and torchdiffeq's rk4 spends 4N for N fixed steps, and
     direct backprop makes ZERO calls of f in the backward pass;
  4. torchdiffeq's rk4 is the 3/8-rule tableau, not the classical one of
     common/odesolve.py: both are order 4, and the two differ by a measured,
     nonzero amount on the same grid;
  5. equal-time separation: over every pair of the 2048 test points and 41
     times, the ratio |x_i(t) - x_j(t)| / |x_i(0) - x_j(0)| stays above a
     measured positive floor. Paths drawn in the plane may still cross,
     because a crossing of paths happens at DIFFERENT times; the count of
     such crossings is recorded, not hidden;
  6. memory: direct backprop grows with the number of solver steps and the
     adjoint does not (ratios asserted on the GPU measurement only; the CPU
     RSS fallback is recorded as coarse and asserted on nothing);
  7. gradient fidelity: direct autodiff agrees with the central difference
     of the SAME discrete forward computation under the U0.L1 mixed
     tolerance; the continuous adjoint does not, and its gap is recorded;
     both approach the continuous-time gradient, computed two ways that
     must agree;
  8. the JAX port computes the same terminal states as torch at tight
     tolerance, and RecursiveCheckpointAdjoint agrees with the central
     difference of its own discrete program while BacksolveAdjoint does not.

NO MONOTONIC LAW IS ASSERTED for NFE drift in training, for accuracy
against tolerance, or for fidelity against step size (L3-U2-L1 v0.2 review):
those are measured and described.

FONT SIZES are large on purpose (14-16 pt): a figure included at the width of
a 16:9 Beamer frame lands at about half scale (U1.T1 qa-quarto M5).

BOUNDARY (L3-U2-L1 section 5). Classifiers only. No density, no divergence,
no generative model of any kind.

Run:  conda run -n torch-gpu python scripts/python/fig_u2_l1_instruments.py
      conda run -n torch-gpu python scripts/python/fig_u2_l1_instruments.py --only sweep
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: identical output in CI, SSH, containers
matplotlib.rcParams["svg.hashsalt"] = "u2l1"  # deterministic SVG element ids
# Type 42 (TrueType) glyph outlines, NOT matplotlib's Type 3 default: Type 3
# fonts carry no hinting, so PDF viewers render small axis labels mushy.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from torchdiffeq import odeint, odeint_adjoint  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common import odesolve  # noqa: E402
from common.toy2d import EXTENT, two_moons  # noqa: E402

# --- UFRJ palette (keep in sync with the theme surfaces in course/_theme/;
# --- verify with ./scripts/check-palette-sync.sh) ---
UFRJ_BLUE = "#003366"
UFRJ_ACCENT = "#d97a1f"
UFRJ_GRAY = "#6b7280"
UFRJ_GRAY_LIGHT = "#9aa2aa"
INK = "#26262a"
CLASS_COLORS = (UFRJ_BLUE, UFRJ_ACCENT)             # label 0 = upper moon
REGION_CMAP = ListedColormap(["#dbe4ef", "#f7e3cf"])  # light blue / light orange

SEED = 0
OUT_DIR = ROOT / "course" / "Figures"
JSON_OUT = Path(__file__).resolve().parent / "_outputs" / "u2l1_instruments.json"
WEIGHTS = ROOT / "common" / "weights" / "u2l1-node-classifier.pt"
WEIGHTS_META = WEIGHTS.with_suffix(".json")

# --- the lab's configuration (Pre-Flight D2/D3, 2026-09-28) -----------------
CONFIG = {
    "hidden": 32, "n_train": 512, "n_test": 2048, "seed_train": 0, "seed_test": 1,
    "steps": 300, "lr": 3e-3, "weight_decay": 0.01, "warmup_frac": 0.03,
    "method": "dopri5", "rtol": 1e-7, "atol": 1e-9,   # torchdiffeq's defaults
}
CARD_LR = 3e-4            # the U0.T2 recipe card's base rate: the deviation's witness
CARD_STEPS = 1500
DRIFT_SEEDS = (0, 1, 2)
SWEEP_STEPS = (4, 8, 16, 32)
SWEEP_RTOLS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
REF_TOL = {"rtol": 1e-12, "atol": 1e-14}          # float64 reference solves
CROSS_STEPS = (8, 16, 32, 64, 128)
MEM_STEPS = (8, 16, 32, 64, 128, 256)
FID_STEPS = (8, 32)
FID_BATCH = 256
FD_EPS = 1e-6
RITUAL_ATOL, RITUAL_RTOL = 1e-8, 1e-6              # U0.L1, the mixed tolerance
N_SINGLE = 64                                      # single-point solves for the batch-NFE check
FS = {"title": 16, "label": 15, "tick": 14, "legend": 14, "annot": 14}   # >= 7 pt at ~0.5x in Beamer

torch.set_num_threads(1)   # tiny ops: one thread is fastest, and timings stay comparable


def set_all_seeds(seed: int) -> None:
    """Seed every RNG stream this script can touch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def apply_house_style(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(UFRJ_GRAY)
    ax.spines["bottom"].set_color(UFRJ_GRAY)
    ax.tick_params(colors=INK, labelsize=FS["tick"])
    ax.grid(axis=grid_axis, color="#DDDDDD", linewidth=0.6)
    ax.set_axisbelow(True)


def save_pair(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf"):
        path = OUT_DIR / f"{name}.{ext}"
        fig.savefig(path, transparent=True, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# The lab's model. The notebook generator copies these three classes by
# `inspect.getsource`, so the notebook and this script cannot drift apart.
# ---------------------------------------------------------------------------
class ODEField(nn.Module):
    """The right-hand side f_theta(t, x): R^2 -> R^2 of the Neural ODE.

    The solver calls it as f(t, x), time first (U0.T5), with t a scalar
    tensor. The network sees t as a (B,) column beside x: the course
    convention for time in a network signature. tanh is smooth with
    |tanh'| <= 1, so the field is C^1 in (t, x, theta) and globally Lipschitz
    in x: both tiers of U2.T1's regularity contract.
    """

    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2),
        )
        # Recipe card item 3: zero-init the last layer, so f = 0 at the start
        # and the flow map starts at the identity.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        tb = t.to(x.dtype).expand(x.shape[0])                  # (B,)
        return self.net(torch.cat([x, tb[:, None]], dim=1))


class NFECounter(nn.Module):
    """Wraps a field and counts its calls: the NFE, the course's cost currency."""

    def __init__(self, f: nn.Module) -> None:
        super().__init__()
        self.f = f
        self.nfe = 0

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        return self.f(t, x)


class NODEClassifier(nn.Module):
    """x(0) is the input point, x(1) = psi_1(x(0)), and the logits are W x(1) + b."""

    def __init__(self, hidden: int = 32, zero_head: bool = True) -> None:
        super().__init__()
        self.field = NFECounter(ODEField(hidden))
        self.head = nn.Linear(2, 2)
        if zero_head:
            # Recipe card item 3 again: logits = 0 at the start, a uniform first
            # prediction (loss log 2). With PyTorch's default head, some seeds
            # stall at every learning rate tried (measured: stage `init` of
            # scripts/python/fig_u2_l1_instruments.py).
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def flow(self, x0: torch.Tensor, ts: torch.Tensor | None = None,
             adjoint: bool = False, **solver) -> torch.Tensor:
        """The states x(t) at the times ts (default: t = 0 and t = 1)."""
        if ts is None:
            ts = torch.tensor([0.0, 1.0], dtype=x0.dtype, device=x0.device)
        solve = odeint_adjoint if adjoint else odeint
        return solve(self.field, x0, ts, **solver)             # (len(ts), B, 2)

    def forward(self, x0: torch.Tensor, adjoint: bool = False, **solver) -> torch.Tensor:
        return self.head(self.flow(x0, adjoint=adjoint, **solver)[-1])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def moons_with_labels(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """two_moons plus its labels. The toy2d contract puts the first n // 2 rows
    on the upper moon, so the label IS the row order: no edit of the frozen file."""
    x = two_moons(n, seed)
    y = (torch.arange(n) >= n // 2).long()
    return x, y


def warmup_cosine(step: int, total: int, warmup: int) -> float:
    """Multiplier on the base LR: the U0.T2 schedule, stepped PER STEP."""
    if step < warmup:
        return step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * p))


def dopri5_defaults() -> dict:
    return {"method": CONFIG["method"], "rtol": CONFIG["rtol"], "atol": CONFIG["atol"]}


def fixed(method: str, n_steps: int) -> dict:
    return {"method": method, "options": {"step_size": 1.0 / n_steps}}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_reference(dtype: torch.dtype = torch.float32, device: str = "cpu") -> NODEClassifier:
    model = NODEClassifier(CONFIG["hidden"])
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    return model.to(device=device, dtype=dtype).eval()


def train_run(seed: int, lr: float, steps: int,
              zero_head: bool = True) -> tuple[NODEClassifier, dict]:
    """The lab's TODO 1.2: full batch, AdamW + warmup-cosine, dopri5 defaults.

    Logs, per step: loss, training accuracy, forward NFE, backward NFE (calls
    of f during loss.backward(), which direct backprop never makes) and the
    wall-clock of the step.
    """
    set_all_seeds(seed)
    model = NODEClassifier(CONFIG["hidden"], zero_head)   # built BEFORE the data (MEMORY 2026-09-17)
    x, y = moons_with_labels(CONFIG["n_train"], CONFIG["seed_train"])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=CONFIG["weight_decay"])
    warm = max(1, int(CONFIG["warmup_frac"] * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_cosine(s, steps, warm))
    log = {"loss": [], "acc": [], "nfe_fwd": [], "nfe_bwd": [], "sec": []}
    for _ in range(steps):
        t0 = time.perf_counter()
        model.field.nfe = 0
        logits = model(x, **dopri5_defaults())
        nfe_fwd = model.field.nfe
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        log["sec"].append(time.perf_counter() - t0)
        log["loss"].append(float(loss.detach()))
        log["acc"].append(float((logits.argmax(1) == y).float().mean()))
        log["nfe_fwd"].append(nfe_fwd)
        log["nfe_bwd"].append(model.field.nfe - nfe_fwd)
    return model, log


@torch.no_grad()
def evaluate(model: NODEClassifier, x: torch.Tensor, y: torch.Tensor, **solver) -> dict:
    model.field.nfe = 0
    t0 = time.perf_counter()
    z1 = model.flow(x, **solver)[-1]
    sec = time.perf_counter() - t0
    logits = model.head(z1)
    return {
        "acc": float((logits.argmax(1) == y).float().mean()),
        "ce": float(F.cross_entropy(logits, y)),
        "nfe": model.field.nfe, "sec": sec, "z1": z1,
    }


# ---------------------------------------------------------------------------
# Stage: train. The reference model, the card-rate witness, the checkpoint.
# ---------------------------------------------------------------------------
def stage_train() -> dict:
    xtr, ytr = moons_with_labels(CONFIG["n_train"], CONFIG["seed_train"])
    upper = float(xtr[ytr == 0, 1].mean())
    lower = float(xtr[ytr == 1, 1].mean())
    assert upper > lower + 1.0, (upper, lower)            # check 1: the labels are the moons

    model, log = train_run(SEED, CONFIG["lr"], CONFIG["steps"])
    assert all(b == 0 for b in log["nfe_bwd"]), "direct backprop called f in backward"  # check 3
    xte, yte = moons_with_labels(CONFIG["n_test"], CONFIG["seed_test"])
    ev = evaluate(model.eval(), xte, yte, **dopri5_defaults())
    assert ev["acc"] >= 0.99, ev["acc"]                     # check 2

    # the linear head alone, on the raw inputs: how much work the flow does
    set_all_seeds(SEED)
    lin = nn.Linear(2, 2)
    opt = torch.optim.AdamW(lin.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    for _ in range(2000):
        loss = F.cross_entropy(lin(xtr), ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        lin_acc = float((lin(xte).argmax(1) == yte).float().mean())

    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS)
    reloaded = load_reference()
    ev2 = evaluate(reloaded, xte, yte, **dopri5_defaults())
    assert ev2["acc"] == ev["acc"] and ev2["nfe"] == ev["nfe"]

    # the card's rate, same seed, same model: the reason for the deviation, measured
    _, card = train_run(SEED, CARD_LR, CARD_STEPS)
    final = log["loss"][-1]
    reach = next((k + 1 for k, v in enumerate(card["loss"]) if v <= final), None)
    card_300 = card["loss"][CONFIG["steps"] - 1]

    n_params = sum(p.numel() for p in model.parameters())
    meta = {
        "sha256": sha256(WEIGHTS),
        "built_by": "scripts/python/fig_u2_l1_instruments.py (stage train)",
        "loads_into": "NODEClassifier(hidden=32) of the U2.L1 notebook",
        "data": "two_moons(512, 0) train / two_moons(2048, 1) test; label = row >= n // 2",
        "config": CONFIG, "seed": SEED, "n_params": n_params,
        "test_acc_dopri5_defaults": ev["acc"], "test_nfe_dopri5_defaults": ev["nfe"],
        "purpose": "rescue checkpoint for Blocks 2-4 of the U2.L1 notebook",
    }
    WEIGHTS_META.write_text(json.dumps(meta, indent=2) + "\n")
    med_ms = 1e3 * statistics.median(log["sec"])
    print(f"train: acc {ev['acc']:.4f} NFE {ev['nfe']} params {n_params} "
          f"{sum(log['sec']):.1f}s ({med_ms:.1f} ms/step); card lr reaches {final:.4g} at {reach}")
    return {"train": {
        "config": CONFIG, "n_params": n_params,
        "label_check_mean_y": {"upper_label0": upper, "lower_label1": lower},
        "final_train_loss": final, "final_train_acc": log["acc"][-1],
        "test_acc": ev["acc"], "test_ce": ev["ce"], "test_nfe": ev["nfe"],
        "train_seconds": sum(log["sec"]), "median_ms_per_step": med_ms,
        "nfe_fwd_first": log["nfe_fwd"][0], "nfe_fwd_last": log["nfe_fwd"][-1],
        "nfe_fwd_max": max(log["nfe_fwd"]), "nfe_bwd_max": max(log["nfe_bwd"]),
        "linear_head_only_test_acc": lin_acc,
        "card_rate": {"lr": CARD_LR, "steps": CARD_STEPS, "loss_at_300": card_300,
                      "acc_at_300": card["acc"][CONFIG["steps"] - 1],
                      "steps_to_reach_final_loss": reach,
                      "loss_at_end": card["loss"][-1], "acc_at_end": card["acc"][-1],
                      "seconds_total": sum(card["sec"])},
        "checkpoint_sha256": meta["sha256"],
        "cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(),
    }, "_drift_seed0": log}


# ---------------------------------------------------------------------------
# Stage: init. Why the head starts at zero: a seed study, measured.
# ---------------------------------------------------------------------------
INIT_RANDOM_HEAD = ((3e-3, 300), (1e-3, 600), (3e-4, 1500))   # (lr, steps)
INIT_SEEDS_RANDOM = (0, 1, 2, 3, 4)
INIT_SEEDS_ZERO = tuple(range(10))


def stage_init() -> dict:
    xte, yte = moons_with_labels(CONFIG["n_test"], CONFIG["seed_test"])
    rows = []

    def one(zero_head: bool, lr: float, steps: int, seed: int) -> None:
        model, log = train_run(seed, lr, steps, zero_head)
        ev = evaluate(model.eval(), xte, yte, **dopri5_defaults())
        rows.append({"head": "zero" if zero_head else "pytorch-default", "lr": lr, "steps": steps,
                     "seed": seed, "loss_first": log["loss"][0], "loss_last": log["loss"][-1],
                     "test_acc": ev["acc"], "test_nfe": ev["nfe"], "nfe_fwd_max": max(log["nfe_fwd"])})
        r = rows[-1]
        print(f"init {r['head']:15s} lr {lr:.0e} steps {steps:4d} seed {seed}: "
              f"loss {r['loss_first']:.3f} -> {r['loss_last']:.4f} acc {r['test_acc']:.4f}")

    for lr, steps in INIT_RANDOM_HEAD:
        for s in INIT_SEEDS_RANDOM:
            one(False, lr, steps, s)
    for s in INIT_SEEDS_ZERO:
        one(True, CONFIG["lr"], CONFIG["steps"], s)
    return {"init": summarize_init(rows)}


STALL_LOSS = 0.1   # a run "stalls" when its FINAL TRAINING loss stays above this


def summarize_init(rows: list[dict]) -> dict:
    """Classify the init study on the TRAINING loss, never on the test split (INV-13;
    qa-quarto round 6). Test accuracies are kept as a description only."""
    zero = [r for r in rows if r["head"] == "zero"]
    assert all(r["loss_last"] < STALL_LOSS for r in zero), "a zero-head seed stalled"
    stalled = {(r["lr"], r["seed"]) for r in rows if r["head"] != "zero" and r["loss_last"] > STALL_LOSS}
    stalled_seeds = sorted({s for _, s in stalled})
    # the claim the notes make: the SAME seeds stall at EVERY learning rate tried
    for s in stalled_seeds:
        assert all((lr, s) in stalled for lr, _ in INIT_RANDOM_HEAD), (s, stalled)
    others = [r for r in rows if r["head"] != "zero" and r["seed"] not in stalled_seeds]
    return {"rows": rows, "stalled_seeds_random_head": stalled_seeds,
            "stall_criterion": f"final training loss > {STALL_LOSS}",
            "stalled_loss_min": min(r["loss_last"] for r in rows
                                    if r["head"] != "zero" and r["seed"] in stalled_seeds),
            "others_loss_max": max(r["loss_last"] for r in others + zero),
            "zero_head_min_acc": min(r["test_acc"] for r in zero),
            "zero_head_max_final_loss": max(r["loss_last"] for r in zero)}


# ---------------------------------------------------------------------------
# Stage: crosscheck. The optional warmup: common/odesolve RK4 vs torchdiffeq.
# ---------------------------------------------------------------------------
@torch.no_grad()
def stage_crosscheck() -> dict:
    model = load_reference(torch.float64)
    xte, _ = moons_with_labels(CONFIG["n_test"], CONFIG["seed_test"])
    xte = xte.double()
    ref = model.flow(xte, method="dopri5", **REF_TOL)[-1]
    rows = []
    for n in CROSS_STEPS:
        ours = odesolve.odeint(model.field.f, xte, 0.0, 1.0, method="rk4", n_steps=n)
        model.field.nfe = 0
        lib = model.flow(xte, **fixed("rk4", n))[-1]
        lib_nfe = model.field.nfe
        rows.append({
            "n_steps": n, "nfe_odesolve": int(ours.nfe), "nfe_torchdiffeq": lib_nfe,
            "err_odesolve_vs_ref": float((ours - ref).abs().max()),
            "err_torchdiffeq_rk4_vs_ref": float((lib - ref).abs().max()),
            "classical_vs_38rule": float((ours - lib).abs().max()),
        })
        assert rows[-1]["nfe_odesolve"] == 4 * n and lib_nfe == 4 * n        # check 3
    assert all(r["classical_vs_38rule"] > 0 for r in rows)                    # check 4
    at64 = next(r for r in rows if r["n_steps"] == 64)
    # the notebook's threshold: ten times the measured error, rounded up to a power of ten
    threshold = 10.0 ** math.ceil(math.log10(10 * at64["err_odesolve_vs_ref"]))
    for r in rows:
        print(f"crosscheck N={r['n_steps']:4d}: odesolve {r['err_odesolve_vs_ref']:.2e} "
              f"torchdiffeq {r['err_torchdiffeq_rk4_vs_ref']:.2e} diff {r['classical_vs_38rule']:.2e}")
    return {"crosscheck": {"rows": rows, "notebook_n_steps": 64,
                           "notebook_threshold": threshold, "reference": REF_TOL}}


# ---------------------------------------------------------------------------
# Stage: sweep (Block 2). Evaluation only, on the checkpoint, in float64.
# ---------------------------------------------------------------------------
@torch.no_grad()
def stage_sweep() -> dict:
    model = load_reference(torch.float64)
    xte, yte = moons_with_labels(CONFIG["n_test"], CONFIG["seed_test"])
    xte = xte.double()
    ref = evaluate(model, xte, yte, method="dopri5", **REF_TOL)
    rows = []

    def run(family: str, setting: float, **solver) -> None:
        secs = []
        for _ in range(3):
            ev = evaluate(model, xte, yte, **solver)
            secs.append(ev["sec"])
        err = (ev["z1"] - ref["z1"]).norm(dim=1)
        rows.append({"family": family, "setting": setting, "nfe": ev["nfe"],
                     "acc": ev["acc"], "ce": ev["ce"],
                     "err_rms": float(err.pow(2).mean().sqrt()), "err_max": float(err.max()),
                     "ms": 1e3 * statistics.median(secs)})

    for method in ("euler", "rk4"):
        for n in SWEEP_STEPS:
            run(method, n, **fixed(method, n))
            assert rows[-1]["nfe"] == (n if method == "euler" else 4 * n)   # check 3
    for rtol in SWEEP_RTOLS:
        run("dopri5", rtol, method="dopri5", rtol=rtol, atol=rtol * 1e-2)

    # the NFE of a batched adaptive solve belongs to the batch: the error norm
    # is an RMS over the WHOLE state tensor (torchdiffeq _rms_norm)
    batch_nfe = evaluate(model, xte, yte, **dopri5_defaults())["nfe"]
    # spread over BOTH moons: rows 0..1023 are all upper moon (toy2d contract), so
    # the first 64 rows would be one class only (domain lens, 2026-09-28)
    single_idx = list(range(0, CONFIG["n_test"], CONFIG["n_test"] // N_SINGLE))
    assert len(single_idx) == N_SINGLE and len({int(yte[i]) for i in single_idx}) == 2
    singles = []
    for i in single_idx:
        singles.append(evaluate(model, xte[i:i + 1], yte[i:i + 1], **dopri5_defaults())["nfe"])

    dop = [r for r in rows if r["family"] == "dopri5"]
    fit = [r for r in dop if r["setting"] <= 1e-3]
    slope = float(np.polyfit(np.log10([r["setting"] for r in fit]),
                             np.log10([r["nfe"] for r in fit]), 1)[0])
    for r in rows:
        print(f"sweep {r['family']:6s} {r['setting']:8.0e} NFE {r['nfe']:4d} acc {r['acc']:.4f} "
              f"err {r['err_rms']:.2e} {r['ms']:.1f} ms")
    print(f"batch NFE {batch_nfe}; single-point NFE min/med/max "
          f"{min(singles)}/{statistics.median(singles)}/{max(singles)}; slope {slope:.3f}")
    out = {"sweep": {"rows": rows, "ref_acc": ref["acc"], "ref_nfe": ref["nfe"],
                     "reference": REF_TOL, "dtype": "float64",
                     "dopri5_loglog_slope_rtol_le_1e-3": slope},
           "batch_vs_single": {"batch_nfe": batch_nfe, "n_single": N_SINGLE,
                               "single_min": min(singles), "single_median": statistics.median(singles),
                               "single_max": max(singles),
                               "n_single_above_batch": sum(1 for v in singles if v > batch_nfe),
                               "n_single_below_batch": sum(1 for v in singles if v < batch_nfe),
                               "single_rows": "range(0, 2048, 32): both moons"}}
    fig_accuracy_nfe(rows)
    fig_nfe_tolerance(dop)
    return out


def fig_accuracy_nfe(rows: list[dict]) -> None:
    # sized for a 16:9 Beamer frame at full text width: scale ~0.54, ticks >= 7 pt
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    styles = {"euler": (UFRJ_GRAY, "s", r"euler, $N = 4, \ldots, 32$"),
              "rk4": (UFRJ_BLUE, "o", r"rk4, $N = 4, \ldots, 32$"),
              "dopri5": (UFRJ_ACCENT, "^", r"dopri5, rtol$\,= 10^{-1}, \ldots, 10^{-7}$")}
    for fam, (c, m, lab) in styles.items():
        rr = sorted((r for r in rows if r["family"] == fam), key=lambda r: r["nfe"])
        nfe = [r["nfe"] for r in rr]
        axes[0].plot(nfe, [100 * r["acc"] for r in rr], marker=m, color=c, lw=1.8, ms=7, label=lab)
        axes[1].plot(nfe, [r["err_rms"] for r in rr], marker=m, color=c, lw=1.8, ms=7, label=lab)
    axes[0].set_ylabel("test accuracy (%)", fontsize=FS["label"], color=INK)
    axes[0].set_ylim(94.5, 100.5)                  # accuracy cannot exceed 100: no tick above it
    axes[0].set_yticks([95, 96, 97, 98, 99, 100])
    axes[0].set_title("(a) the task metric", fontsize=FS["title"], color=INK)
    axes[1].set_yscale("log")
    axes[1].set_ylabel(r"RMS error of $x(1)$", fontsize=FS["label"], color=INK)
    axes[1].set_title("(b) the terminal state", fontsize=FS["title"], color=INK)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("NFE of the forward solve", fontsize=FS["label"], color=INK)
        apply_house_style(ax)
    # panel (a) is empty below 99.5 %: the legend goes there, off the data of (b)
    axes[0].legend(fontsize=FS["legend"], frameon=False, loc="lower right")
    # rtol 1e-1 and 1e-2 cost the same NFE: label them, or the vertical segment
    # between them reads as an arrow (qa-quarto round 2)
    for r in rows:
        if r["family"] == "dopri5" and r["setting"] in (1e-1, 1e-2):
            exp = int(round(math.log10(r["setting"])))
            axes[1].annotate(rf"rtol $10^{{{exp}}}$", (r["nfe"], r["err_rms"]),
                             textcoords="offset points", xytext=(8, 0), va="center",
                             fontsize=FS["annot"], color=UFRJ_ACCENT)
    fig.tight_layout()
    save_pair(fig, "u2-l1-accuracy-nfe")


def fig_nfe_tolerance(dop: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    rr = sorted(dop, key=lambda r: -r["setting"])
    ax.plot([r["setting"] for r in rr], [r["nfe"] for r in rr], marker="^", color=UFRJ_ACCENT,
            lw=1.8, ms=8)
    for r in rr:
        ax.annotate(str(r["nfe"]), (r["setting"], r["nfe"]), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=FS["annot"], color=INK)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_ylim(0, 1.15 * max(r["nfe"] for r in rr))   # headroom for the top label
    ax.set_xlabel(r"rtol (atol $=$ rtol$\,\times 10^{-2}$)", fontsize=FS["label"], color=INK)
    ax.set_ylabel("NFE of the forward solve", fontsize=FS["label"], color=INK)
    apply_house_style(ax)
    fig.tight_layout()
    save_pair(fig, "u2-l1-nfe-tolerance")


# ---------------------------------------------------------------------------
# Stage: drift (TODO 2.3). Forward NFE per training step, three seeds.
# ---------------------------------------------------------------------------
def stage_drift(seed0_log: dict | None) -> dict:
    logs = {}
    for s in DRIFT_SEEDS:
        if s == SEED and seed0_log is not None:
            logs[s] = seed0_log
        else:
            _, logs[s] = train_run(s, CONFIG["lr"], CONFIG["steps"])
    # Steps 1-3 of every seed cost 44: the field is exactly zero (card item 3), so
    # torchdiffeq's initial-step rule falls back to h = 1e-6 (misc._select_initial_step,
    # d1 < 1e-5), every error estimate is exactly zero, and the controller multiplies
    # h by its maximum factor, 10 (misc._optimal_step_size, ifactor). Seven steps
    # cover [0, 1]: 2 start-up calls + 6 x 7.
    zero = NODEClassifier(CONFIG["hidden"])
    times: list[float] = []
    inner = zero.field.f

    class Recorder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.f = inner

        def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            times.append(float(t))
            return self.f(t, x)

    zero.field.f = Recorder()
    xtr, _ = moons_with_labels(CONFIG["n_train"], CONFIG["seed_train"])
    with torch.no_grad():
        zero.field.nfe = 0
        zero(xtr, **dopri5_defaults())
    starts = sorted({round(t, 12) for t in times})
    assert zero.field.nfe == 44 == 2 + 6 * 7, zero.field.nfe
    summary = {"zero_field_first_solve": {"nfe": zero.field.nfe, "first_step": 1e-6,
                                          "growth_per_step": 10, "steps": 7,
                                          "distinct_call_times": len(starts)}}
    for s, lg in logs.items():
        nf = lg["nfe_fwd"]
        # the field stays exactly zero for three solves: step 1 has learning rate 0
        # (the warmup starts at 0) and step 2 moves only the head (W = 0 blocks the
        # field's gradient); step 4 is the first solve of a nonzero field
        assert nf[:3] == [44, 44, 44] and nf[3] < 44, nf[:4]
        k_max = int(np.argmax(nf))
        summary[str(s)] = {"first": nf[0], "at_50": nf[49], "at_100": nf[99], "last": nf[-1],
                           "loss_first": lg["loss"][0], "loss_at_argmax": lg["loss"][k_max],
                           "max": max(nf), "argmax": int(np.argmax(nf)) + 1,
                           "min_first_30": min(nf[:30]), "argmin_first_30": int(np.argmin(nf[:30])) + 1,
                           "first_10": nf[:10],
                           "min_last_100": min(nf[-100:]), "max_last_100": max(nf[-100:]),
                           "final_loss": lg["loss"][-1]}
        print(f"drift seed {s}: NFE {nf[0]} -> {nf[99]} (step 100) -> {nf[-1]} (max {max(nf)})")
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6), sharex=True)
    colors = (UFRJ_BLUE, UFRJ_ACCENT, UFRJ_GRAY)
    steps = np.arange(1, CONFIG["steps"] + 1)
    # seed 0 (the reference run) is drawn last, on top: the text quotes its values
    for (s, lg), c in reversed(list(zip(logs.items(), colors))):
        z = 3 if s == SEED else 2
        ls = "--" if s == DRIFT_SEEDS[-1] else "-"     # seed 2's loss lies under seed 0's
        axes[0].plot(steps, lg["nfe_fwd"], color=c, lw=1.6, ls=ls, drawstyle="steps-post",
                     label=f"seed {s}", zorder=z)
        axes[1].plot(steps, lg["loss"], color=c, lw=1.6, ls=ls, zorder=z)
    axes[0].set_ylabel("forward NFE", fontsize=FS["label"], color=INK)
    axes[0].set_title("(a) the cost of one forward solve", fontsize=FS["title"], color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda k: labels[k])
    axes[0].legend([handles[k] for k in order], [labels[k] for k in order],
                   fontsize=FS["legend"], frameon=False, loc="lower right")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("training loss", fontsize=FS["label"], color=INK)
    axes[1].set_title("(b) the training loss", fontsize=FS["title"], color=INK)
    for ax in axes:
        ax.set_xlabel("optimizer step", fontsize=FS["label"], color=INK)
    for ax in axes:
        apply_house_style(ax)
    fig.tight_layout()
    save_pair(fig, "u2-l1-nfe-drift")
    return {"drift": {"seeds": list(DRIFT_SEEDS), "summary": summary}}


# ---------------------------------------------------------------------------
# Stage: trajectories (part of train; needs the checkpoint). Figure + check 5.
# ---------------------------------------------------------------------------
@torch.no_grad()
def stage_trajectories() -> dict:
    model = load_reference(torch.float64)
    xte, yte = moons_with_labels(CONFIG["n_test"], CONFIG["seed_test"])
    xte = xte.double()
    ts = torch.linspace(0.0, 1.0, 41, dtype=torch.float64)
    traj = model.flow(xte, ts=ts, **dopri5_defaults())            # (41, 2048, 2)

    d0 = torch.cdist(traj[0], traj[0])
    off = ~torch.eye(len(xte), dtype=torch.bool)
    ratio_min, ratio_max = float("inf"), 0.0
    for k in range(len(ts)):
        r = torch.cdist(traj[k], traj[k])[off] / d0[off]
        ratio_min = min(ratio_min, float(r.min()))
        ratio_max = max(ratio_max, float(r.max()))
    # check 5: a stated floor, not "> 0" (a float ratio of exactly 0 needs an exact
    # coincidence, so "> 0" could never fire). Measured: 0.41.
    assert ratio_min > 0.1, ratio_min

    # the 40 drawn paths: 20 per class, evenly spread along each moon
    idx = torch.cat([torch.arange(0, 1024, 52)[:20], torch.arange(1024, 2048, 52)[:20]])
    paths = traj[:, idx].numpy()                                      # (41, 40, 2)
    crossings = count_path_crossings(paths)

    # decision regions in input space: solve from every grid point
    g = torch.linspace(EXTENT[0], EXTENT[1], 161, dtype=torch.float64)
    gx, gy = torch.meshgrid(g, g, indexing="xy")
    grid = torch.stack([gx.ravel(), gy.ravel()], 1)
    pred = model.head(model.flow(grid, **dopri5_defaults())[-1]).argmax(1).reshape(gx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
    ax = axes[0]
    ax.contourf(gx.numpy(), gy.numpy(), pred.numpy(), levels=[-0.5, 0.5, 1.5], cmap=REGION_CMAP)
    sub = torch.arange(0, len(xte), 5)
    for c in (0, 1):
        m = sub[yte[sub] == c]
        ax.scatter(xte[m, 0], xte[m, 1], s=7, color=CLASS_COLORS[c], alpha=0.8, lw=0)
    ax.set_xlim(*EXTENT)
    ax.set_ylim(*EXTENT)
    ax.set_aspect("equal")
    ax.set_xlabel("first coordinate", fontsize=FS["label"], color=INK)
    ax.set_ylabel("second coordinate", fontsize=FS["label"], color=INK)
    ax.set_title(r"(a) inputs $x(0)$, decision regions", fontsize=FS["title"], color=INK)
    apply_house_style(ax)
    ax.grid(False)

    ax = axes[1]
    for j in range(paths.shape[1]):
        c = CLASS_COLORS[0 if j < 20 else 1]
        ax.plot(paths[:, j, 0], paths[:, j, 1], color=c, lw=1.1, alpha=0.85)
        ax.scatter(paths[0, j, 0], paths[0, j, 1], s=16, facecolor="white", edgecolor=c, lw=1.0, zorder=3)
        ax.scatter(paths[-1, j, 0], paths[-1, j, 1], s=22, color=c, zorder=3)
    w = (model.head.weight[0] - model.head.weight[1]).numpy()
    b = float(model.head.bias[0] - model.head.bias[1])
    lo = paths.reshape(-1, 2).min(0) - 0.6
    hi = paths.reshape(-1, 2).max(0) + 0.6
    xs = np.linspace(lo[0], hi[0], 2)
    ax.plot(xs, -(w[0] * xs + b) / w[1], color=INK, lw=1.6, ls="--")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.set_xlabel("first coordinate", fontsize=FS["label"], color=INK)
    ax.set_ylabel("second coordinate", fontsize=FS["label"], color=INK)
    ax.set_title(r"(b) states $x(t)$; the head at $t=1$", fontsize=FS["title"], color=INK)
    apply_house_style(ax)
    fig.tight_layout()
    save_pair(fig, "u2-l1-trajectories")
    print(f"trajectories: equal-time distance ratio in [{ratio_min:.3f}, {ratio_max:.3f}]; "
          f"{crossings} crossings among the 40 drawn paths")
    return {"trajectories": {"n_points": len(xte), "n_times": len(ts),
                             "equal_time_ratio_min": ratio_min, "equal_time_ratio_max": ratio_max,
                             "drawn_paths": int(paths.shape[1]),
                             "drawn_path_crossings_at_different_times": crossings,
                             "head_normal": w.tolist(), "head_offset": b}}


def count_path_crossings(paths: np.ndarray) -> int:
    """Pairs of drawn paths whose polylines intersect in the plane (any times)."""
    def seg_cross(p1, p2, q1, q2) -> bool:
        def orient(a, b, c):
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        o1, o2 = orient(p1, p2, q1), orient(p1, p2, q2)
        o3, o4 = orient(q1, q2, p1), orient(q1, q2, p2)
        return (o1 * o2 < 0) and (o3 * o4 < 0)

    n = paths.shape[1]
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            hit = False
            for a in range(paths.shape[0] - 1):
                for b_ in range(paths.shape[0] - 1):
                    if seg_cross(paths[a, i], paths[a + 1, i], paths[b_, j], paths[b_ + 1, j]):
                        hit = True
                        break
                if hit:
                    break
            count += hit
    return count


# ---------------------------------------------------------------------------
# Stage: memory (Block 3). GPU canonical; CPU RSS recorded as coarse.
# ---------------------------------------------------------------------------
def _one_backward(model: NODEClassifier, x: torch.Tensor, y: torch.Tensor,
                  n_steps: int, adjoint: bool) -> tuple[int, int]:
    model.zero_grad(set_to_none=True)
    model.field.nfe = 0
    logits = model(x, adjoint=adjoint, **fixed("rk4", n_steps))
    nfe_fwd = model.field.nfe
    F.cross_entropy(logits, y).backward()
    return nfe_fwd, model.field.nfe - nfe_fwd


def gpu_peak(model, x, y, n_steps, adjoint) -> tuple[int, int, int, float]:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    nf, nb = _one_backward(model, x, y, n_steps, adjoint)
    torch.cuda.synchronize()
    sec = time.perf_counter() - t0
    return torch.cuda.max_memory_allocated() - base, nf, nb, sec


def cpu_rss_peak(model, x, y, n_steps, adjoint) -> int:
    """The notebook's CPU fallback: sample the process RSS every millisecond.
    Coarse by construction (allocator reuse, sampling), recorded, never asserted."""
    import psutil
    proc = psutil.Process()
    base = proc.memory_info().rss
    peak = [base]
    stop = threading.Event()

    def sample() -> None:
        while not stop.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss)
            time.sleep(0.001)

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    _one_backward(model, x, y, n_steps, adjoint)
    stop.set()
    th.join()
    return peak[0] - base


def stage_memory() -> dict:
    assert torch.cuda.is_available(), "the canonical memory measurement needs a GPU"
    x, y = moons_with_labels(CONFIG["n_train"], CONFIG["seed_train"])
    model = load_reference(torch.float32, "cuda")
    xg, yg = x.cuda(), y.cuda()
    _one_backward(model, xg, yg, 8, False)            # warm the CUDA kernels once
    _one_backward(model, xg, yg, 8, True)
    rows = []
    for n in MEM_STEPS:
        for adjoint in (False, True):
            reps = [gpu_peak(model, xg, yg, n, adjoint) for _ in range(3)]
            peaks = [r[0] for r in reps]
            rows.append({"n_steps": n, "strategy": "adjoint" if adjoint else "direct",
                         "peak_bytes": int(statistics.median(peaks)),
                         "peak_bytes_spread": int(max(peaks) - min(peaks)),
                         "nfe_fwd": reps[0][1], "nfe_bwd": reps[0][2],
                         "ms": 1e3 * statistics.median(r[3] for r in reps)})
            print(f"memory N={n:4d} {rows[-1]['strategy']:8s} peak {rows[-1]['peak_bytes'] / 2**20:8.2f} MiB "
                  f"NFE fwd {rows[-1]['nfe_fwd']} bwd {rows[-1]['nfe_bwd']}")
    direct = {r["n_steps"]: r for r in rows if r["strategy"] == "direct"}
    adj = {r["n_steps"]: r for r in rows if r["strategy"] == "adjoint"}
    lo, hi = MEM_STEPS[0], MEM_STEPS[-1]
    ratio_direct = direct[hi]["peak_bytes"] / direct[lo]["peak_bytes"]
    ratio_adj = adj[hi]["peak_bytes"] / adj[lo]["peak_bytes"]
    assert ratio_direct > 10.0, ratio_direct                             # check 6
    assert ratio_adj < 1.5, ratio_adj
    assert all(r["nfe_bwd"] == 0 for r in direct.values())
    assert all(r["nfe_bwd"] == r["nfe_fwd"] for r in adj.values()), "adjoint backward NFE"
    per_nfe = np.polyfit([4 * n for n in MEM_STEPS],
                         [direct[n]["peak_bytes"] for n in MEM_STEPS], 1)[0]

    # the CPU fallback, same configurations, in-process, in the notebook's order
    mcpu = load_reference(torch.float32, "cpu")
    cpu_rows = []
    for n in MEM_STEPS:
        for adjoint in (False, True):
            cpu_rows.append({"n_steps": n, "strategy": "adjoint" if adjoint else "direct",
                             "rss_bytes": int(cpu_rss_peak(mcpu, x, y, n, adjoint))})
    fig_memory(direct, adj)
    return {"memory": {"device": torch.cuda.get_device_name(0), "dtype": "float32",
                       "batch": CONFIG["n_train"], "method": "rk4 (torchdiffeq, 3/8 rule)",
                       "rows": rows, "ratio_direct_256_over_8": ratio_direct,
                       "ratio_adjoint_256_over_8": ratio_adj,
                       "direct_bytes_per_nfe_fit": float(per_nfe),
                       "cpu_rss_fallback": {"rows": cpu_rows,
                                            "note": "psutil RSS sampled at 1 ms; coarse, not asserted"}}}


def fig_memory(direct: dict, adj: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.9))
    ns = sorted(direct)
    nfe = [4 * n for n in ns]
    mib = 2.0 ** 20
    ax.plot(nfe, [direct[n]["peak_bytes"] / mib for n in ns], marker="o", color=UFRJ_BLUE, lw=2.0,
            ms=7, label="direct backprop, odeint")
    ax.plot(nfe, [adj[n]["peak_bytes"] / mib for n in ns], marker="s", color=UFRJ_ACCENT, lw=2.0,
            ms=7, label="adjoint, odeint_adjoint")
    for n in ns:
        ax.annotate(f"N={n}", (4 * n, direct[n]["peak_bytes"] / mib), textcoords="offset points",
                    xytext=(6, -14), ha="left", fontsize=FS["annot"], color=UFRJ_GRAY)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("NFE of the forward solve (rk4, 4 per step)", fontsize=FS["label"], color=INK)
    ax.set_ylabel("peak memory (MiB)", fontsize=FS["label"], color=INK)
    ax.legend(fontsize=FS["legend"], frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.14))
    apply_house_style(ax)
    fig.tight_layout()
    save_pair(fig, "u2-l1-memory")


# ---------------------------------------------------------------------------
# Stage: fidelity (Block 3, TODO 3.2). float64, one fixed direction v.
# ---------------------------------------------------------------------------
def field_params(model: NODEClassifier) -> list[torch.Tensor]:
    return list(model.field.parameters())


def fixed_direction(model: NODEClassifier, seed: int = 2026) -> list[torch.Tensor]:
    """One unit direction over the parameters of f_theta (the head's gradient
    does not pass through a solve, so it is left out of v)."""
    g = torch.Generator().manual_seed(seed)
    v = [torch.randn(p.shape, generator=g, dtype=torch.float64) for p in field_params(model)]
    norm = torch.sqrt(sum((t ** 2).sum() for t in v))
    return [t / norm for t in v]


def directional(model, x, y, v, adjoint: bool, **solver) -> float:
    model.zero_grad(set_to_none=True)
    F.cross_entropy(model(x, adjoint=adjoint, **solver), y).backward()
    return float(sum((p.grad * t).sum() for p, t in zip(field_params(model), v)))


@torch.no_grad()
def central_difference(model, x, y, v, eps: float, **solver) -> float:
    ps = field_params(model)
    base = [p.detach().clone() for p in ps]

    def loss_at(sign: float) -> float:
        for p, b_, t in zip(ps, base, v):
            p.copy_(b_ + sign * eps * t)
        return float(F.cross_entropy(model(x, **solver), y))

    hi, lo = loss_at(+1.0), loss_at(-1.0)
    for p, b_ in zip(ps, base):
        p.copy_(b_)
    return (hi - lo) / (2 * eps)


def stage_fidelity() -> dict:
    model = load_reference(torch.float64)
    xte, yte = moons_with_labels(CONFIG["n_test"], CONFIG["seed_test"])
    idx = torch.arange(0, CONFIG["n_test"], CONFIG["n_test"] // FID_BATCH)[:FID_BATCH]
    x, y = xte[idx].double(), yte[idx]
    v = fixed_direction(model)
    d_cont_rk4 = directional(model, x, y, v, False, **fixed("rk4", 1024))
    d_cont_dop = directional(model, x, y, v, False, method="dopri5", **REF_TOL)
    cont_gap = abs(d_cont_rk4 - d_cont_dop)
    assert cont_gap <= 1e-10 + 1e-8 * abs(d_cont_dop), cont_gap     # check 7: two routes agree
    d_cont = d_cont_dop
    rows = []
    for n in FID_STEPS:
        d_fd = central_difference(model, x, y, v, FD_EPS, **fixed("rk4", n))
        d_ad = directional(model, x, y, v, False, **fixed("rk4", n))
        d_adj = directional(model, x, y, v, True, **fixed("rk4", n))
        tol = RITUAL_ATOL + RITUAL_RTOL * max(abs(d_ad), abs(d_fd))
        assert abs(d_ad - d_fd) <= tol, (n, d_ad, d_fd)                  # check 7
        assert abs(d_adj - d_fd) > abs(d_ad - d_fd), (n, d_adj, d_ad, d_fd)
        rows.append({"n_steps": n, "h": 1.0 / n, "d_fd": d_fd, "d_ad": d_ad, "d_adj": d_adj,
                     "abs_ad_fd": abs(d_ad - d_fd), "rel_ad_fd": abs(d_ad - d_fd) / abs(d_fd),
                     "abs_adj_fd": abs(d_adj - d_fd), "rel_adj_fd": abs(d_adj - d_fd) / abs(d_fd),
                     "abs_ad_cont": abs(d_ad - d_cont), "abs_adj_cont": abs(d_adj - d_cont),
                     "ritual_tol": tol})
        r = rows[-1]
        print(f"fidelity N={n}: FD {d_fd:.12f} AD-FD {r['abs_ad_fd']:.2e} ADJ-FD {r['abs_adj_fd']:.2e} "
              f"AD-cont {r['abs_ad_cont']:.2e} ADJ-cont {r['abs_adj_cont']:.2e}")
    return {"fidelity": {"batch": FID_BATCH, "dtype": "float64", "eps": FD_EPS,
                         "direction": "unit, over the parameters of f_theta, Generator seed 2026",
                         "ritual": {"atol": RITUAL_ATOL, "rtol": RITUAL_RTOL},
                         "d_cont": d_cont, "d_cont_rk4_1024": d_cont_rk4,
                         "d_cont_two_routes_gap": cont_gap, "rows": rows}}


# ---------------------------------------------------------------------------
# Stage: jax (Block 4). The diffrax mirror on CPU, float64.
# ---------------------------------------------------------------------------
def stage_jax() -> dict:
    import jax
    jax.config.update("jax_enable_x64", True)
    import diffrax
    import jax.numpy as jnp

    model = load_reference(torch.float64)
    net = model.field.f.net
    p = {"W1": net[0].weight, "b1": net[0].bias, "W2": net[2].weight, "b2": net[2].bias,
         "W3": net[4].weight, "b3": net[4].bias}
    p = {k: jnp.asarray(t.detach().numpy()) for k, t in p.items()}
    head = {"W": jnp.asarray(model.head.weight.detach().numpy()),
            "b": jnp.asarray(model.head.bias.detach().numpy())}
    xte, yte = moons_with_labels(CONFIG["n_test"], CONFIG["seed_test"])
    idx = torch.arange(0, CONFIG["n_test"], CONFIG["n_test"] // FID_BATCH)[:FID_BATCH]
    x_t, y_t = xte[idx].double(), yte[idx]
    x0, y = jnp.asarray(x_t.numpy()), jnp.asarray(y_t.numpy())

    def field(t, x, prm):
        tb = jnp.full((x.shape[0], 1), t)
        h = jnp.tanh(jnp.concatenate([x, tb], axis=1) @ prm["W1"].T + prm["b1"])
        h = jnp.tanh(h @ prm["W2"].T + prm["b2"])
        return h @ prm["W3"].T + prm["b3"]

    term = diffrax.ODETerm(field)

    def terminal(prm, x, n_steps, adjoint):
        sol = diffrax.diffeqsolve(term, diffrax.Dopri5(), t0=0.0, t1=1.0, dt0=1.0 / n_steps, y0=x,
                                  args=prm, stepsize_controller=diffrax.ConstantStepSize(),
                                  saveat=diffrax.SaveAt(t1=True), adjoint=adjoint,
                                  max_steps=4 * n_steps)
        return sol.ys[-1], sol.stats["num_steps"]

    def loss(prm, x, yy, n_steps, adjoint):
        z1, _ = terminal(prm, x, n_steps, adjoint)
        logits = z1 @ head["W"].T + head["b"]
        return -jnp.mean(jax.nn.log_softmax(logits)[jnp.arange(yy.shape[0]), yy])

    # the port computes what torch computes (tight adaptive solves, both sides)
    sol = diffrax.diffeqsolve(term, diffrax.Dopri5(), t0=0.0, t1=1.0, dt0=None, y0=x0, args=p,
                              stepsize_controller=diffrax.PIDController(rtol=1e-12, atol=1e-14),
                              saveat=diffrax.SaveAt(t1=True), max_steps=100_000)
    with torch.no_grad():
        z_torch = model.flow(x_t, method="dopri5", **REF_TOL)[-1].numpy()
    port_gap = float(np.abs(np.asarray(sol.ys[-1]) - z_torch).max())
    assert port_gap < 1e-9, port_gap                                      # check 8

    # evaluations per Dopri5 step, counted by a host callback (not assumed)
    calls = [0]

    def counted(t, x, prm):
        jax.debug.callback(lambda: calls.__setitem__(0, calls[0] + 1))
        return field(t, x, prm)

    n_count = 8
    sol_c = diffrax.diffeqsolve(diffrax.ODETerm(counted), diffrax.Dopri5(), t0=0.0, t1=1.0,
                                dt0=1.0 / n_count, y0=x0, args=p,
                                stepsize_controller=diffrax.ConstantStepSize(),
                                saveat=diffrax.SaveAt(t1=True), max_steps=4 * n_count)
    jax.block_until_ready(sol_c.ys)
    steps_c = int(sol_c.stats["num_steps"])

    # the same direction v as the torch fidelity stage, moved into JAX
    v_t = fixed_direction(model)
    names = ["W1", "b1", "W2", "b2", "W3", "b3"]
    v = {k: jnp.asarray(t.numpy()) for k, t in zip(names, v_t)}

    def dot(a, b_):
        return float(sum(jnp.vdot(a[k], b_[k]) for k in names))

    d_cont = json.loads(JSON_OUT.read_text())["fidelity"]["d_cont"] if JSON_OUT.exists() else None
    adjoints = {"RecursiveCheckpointAdjoint": diffrax.RecursiveCheckpointAdjoint(),
                "BacksolveAdjoint": diffrax.BacksolveAdjoint()}
    rows = []
    for n in FID_STEPS:
        fd_fn = jax.jit(lambda prm: loss(prm, x0, y, n, adjoints["RecursiveCheckpointAdjoint"]))
        hi = float(fd_fn({k: p[k] + FD_EPS * v[k] for k in names}))
        lo = float(fd_fn({k: p[k] - FD_EPS * v[k] for k in names}))
        d_fd = (hi - lo) / (2 * FD_EPS)
        for name, adj in adjoints.items():
            vg = jax.jit(jax.value_and_grad(lambda prm, a=adj: loss(prm, x0, y, n, a)))
            # the up-front price is tracing + lowering + XLA compilation, the whole
            # first-call overhead of the U0.L5 rule (qa-quarto round 4: timing only
            # .compile() left tracing and lowering out)
            t0 = time.perf_counter()
            lowered = vg.lower(p)
            lower_s = time.perf_counter() - t0
            compiled = lowered.compile()
            setup_s = time.perf_counter() - t0
            compile_s = setup_s - lower_s
            jax.block_until_ready(compiled(p))                   # one warm call
            samples = []
            for _ in range(20):
                t0 = time.perf_counter()
                val, grad = compiled(p)
                jax.block_until_ready(grad)
                samples.append(time.perf_counter() - t0)
            d = dot(grad, v)
            rows.append({"n_steps": n, "adjoint": name, "d": d, "d_fd": d_fd,
                         "abs_d_fd": abs(d - d_fd), "rel_d_fd": abs(d - d_fd) / abs(d_fd),
                         "abs_d_cont": abs(d - d_cont) if d_cont is not None else None,
                         "lower_s": lower_s, "compile_s": compile_s, "setup_s": setup_s,
                         "steady_ms": 1e3 * statistics.median(samples)})
            r = rows[-1]
            print(f"jax N={n} {name:26s} |d-FD| {r['abs_d_fd']:.2e} setup {setup_s:.2f}s "
                  f"steady {r['steady_ms']:.2f} ms")
        rc = next(r for r in rows if r["n_steps"] == n and r["adjoint"] == "RecursiveCheckpointAdjoint")
        bs = next(r for r in rows if r["n_steps"] == n and r["adjoint"] == "BacksolveAdjoint")
        tol = RITUAL_ATOL + RITUAL_RTOL * max(abs(rc["d"]), abs(d_fd))
        assert rc["abs_d_fd"] <= tol, rc                                  # check 8
        assert bs["abs_d_fd"] > rc["abs_d_fd"], (bs, rc)

    # the torch side of the same call, for a descriptive (never ranked) comparison
    tmodel = load_reference(torch.float64)
    secs = []
    for _ in range(20):
        t0 = time.perf_counter()
        tmodel.zero_grad(set_to_none=True)
        F.cross_entropy(tmodel(x_t, **fixed("rk4", 8)), y_t).backward()
        secs.append(time.perf_counter() - t0)
    return {"jax": {"device": str(jax.devices()[0]), "jax": jax.__version__,
                    "diffrax": diffrax.__version__, "dtype": "float64", "solver": "Dopri5, ConstantStepSize",
                    "port_gap_vs_torch": port_gap,
                    "dopri5_calls_for_8_steps": calls[0], "dopri5_steps_counted": steps_c,
                    "rows": rows, "torch_direct_rk4_8_ms": 1e3 * statistics.median(secs)}}


# ---------------------------------------------------------------------------
# Stage: replot. Redraws the three figures whose data live in the JSON, with
# no new measurement (the memory numbers move by a few KiB between runs).
# ---------------------------------------------------------------------------
def stage_replot(data: dict) -> dict:
    rows = data["sweep"]["rows"]
    fig_accuracy_nfe(rows)
    fig_nfe_tolerance([r for r in rows if r["family"] == "dopri5"])
    mem = data["memory"]["rows"]
    fig_memory({r["n_steps"]: r for r in mem if r["strategy"] == "direct"},
               {r["n_steps"]: r for r in mem if r["strategy"] == "adjoint"})
    return {}


def cpu_model() -> str:
    """The CPU model name, for the record of every wall-clock number (Linux: lscpu)."""
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return platform.processor() or "unknown"
    for line in out.splitlines():
        if line.startswith("Model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated stage names to re-run")
    args = ap.parse_args()
    all_stages = ["train", "trajectories", "crosscheck", "sweep", "drift", "memory",
                  "fidelity", "jax", "init", "replot", "init-summary"]
    stages = [s.strip() for s in args.only.split(",") if s.strip()] or all_stages
    unknown = set(stages) - set(all_stages)
    if unknown:
        raise SystemExit(f"unknown stage(s): {sorted(unknown)}")

    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(JSON_OUT.read_text()) if JSON_OUT.exists() else {}
    seed0_log = None
    t0 = time.perf_counter()
    for stage in stages:
        print(f"--- stage {stage}")
        if stage == "train":
            out = stage_train()
            seed0_log = out.pop("_drift_seed0")
            data.update(out)
        elif stage == "trajectories":
            data.update(stage_trajectories())
        elif stage == "crosscheck":
            data.update(stage_crosscheck())
        elif stage == "sweep":
            data.update(stage_sweep())
        elif stage == "drift":
            data.update(stage_drift(seed0_log))
        elif stage == "memory":
            data.update(stage_memory())
        elif stage == "fidelity":
            data.update(stage_fidelity())
        elif stage == "jax":
            data.update(stage_jax())
        elif stage == "init":
            data.update(stage_init())
        elif stage == "replot":
            data.update(stage_replot(data))
        elif stage == "init-summary":
            data["init"] = summarize_init(data["init"]["rows"])
        JSON_OUT.write_text(json.dumps(data, indent=2))   # merge after every stage
    data["_meta"] = {"seed": SEED, "torch": torch.__version__, "cpu_model": cpu_model(),
                     "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                  capture_output=True, text=True).stdout.strip(),
                     "git_dirty": bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                                      capture_output=True, text=True).stdout.strip()),
                     "wall_clock_s_this_run": round(time.perf_counter() - t0, 1),
                     "stages_this_run": stages}
    JSON_OUT.write_text(json.dumps(data, indent=2))
    print(f"wrote {JSON_OUT}")


if __name__ == "__main__":
    main()
