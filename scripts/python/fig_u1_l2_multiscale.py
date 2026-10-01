"""fig_u1_l2_multiscale.py — U1.L2 figures, checkpoints and numbers, measured.

Trains the lab's two reference models (common/multiscale_flow.py) on raw
uint8 MNIST with the course training stack (common/train.py, the U0.T2
optimizer defaults), evaluates them under the FROZEN `bpd-protocol`, draws what
the lab asks students to look at, and writes the checkpoint package. Every
number the notes, the deck and the notebook quote is measured here and written
to the JSON; nothing is hand-drawn.

Produces, into course/Figures/ (extension-less includes: html picks .svg,
beamer picks .pdf):

  u1-l2-observation-model   pixel-value histogram of the test set (atoms), the
                            logit map T_lambda and its per-pixel log-Jacobian
  u1-l2-atoms-spike         a continuous density on atoms: mean log-density at
                            the atoms grows without bound as the width shrinks
  u1-l2-bpd-curves          validation bits per dimension (dequantized) per
                            epoch, two-level model vs one-level baseline
  u1-l2-mc-fluctuation      eight one-sample dequantization banks vs the
                            sixteen-sample average on a 1,000-image subset
  u1-l2-samples             reference checkpoint, fixed noise bank, tau = 1.0
                            (the model) and tau = 0.7 (a heuristic), plus the
                            one-level baseline at tau = 1.0

  common/weights/u1l2-multiscale.{pt,json}, u1l2-onelevel.{pt,json}
                            the checkpoint package: state_dict + resolved config
                            + seed + git commit + dirty flag + per-epoch curves
                            + the protocol constants + the measured BPD
  labs/ps1-4d-bundle.npz    the CPU-only cut package for PS1.4(d): 16 raw
                            uint8 test images, their fixed noise, the frozen
                            log p_Y values of the reference checkpoint
  _outputs/u1l2_multiscale.json — the numbers.

STAGES AND --only. Stage names: data, train, protocol, samples, live, zuko,
ps14d. `--only protocol,samples` re-runs those stages, re-measures their keys,
and MERGES them into the existing JSON (MEMORY: a partial run must never
overwrite numbers it did not re-measure). `train` runs on the GPU when one is
present (about 20 minutes for both models on an RTX 4070 Laptop) and its
wall-clock is recorded as GPU time; every other stage runs on CPU, because the
notebook does.

HONESTY CHECKS (asserted, then recorded):
  1. the raw loader preserves uint8: the test set is integer-valued in
     {0, ..., 255}, and the 256 factor is applied ONCE, in the dequantization
     cell — `ToTensor()` divides by 255 and would hide it;
  2. squeeze round trip is exact and both log-dets are exactly 0; factor-out
     partitions: latent sizes add up to 784; the per-level bookkeeping table
     reconciles with `log_prob` per example (multiscale_flow self-check, run
     again here on real data);
  3. manual arithmetic and harness arithmetic agree on a fixed 16-image batch:
     sum_l log p_0(z_l) + coupling log-dets + logit log-det, then
     bpd = -mean / (784 log 2), then +8;
  4. training decreases the validation BPD (both models; the notebook's live
     run is pre-validated in stage `live` on CPU with the same criterion);
  5. the one-sample estimate fluctuates around the K-average; the spread is
     recorded so the text can say "not an exact bound" with a number;
  6. the two models are evaluated under IDENTICAL protocol constants — the
     comparison the L3 allows students to interpret directly;
  7. Zuko's SigmoidTransform and CouplingTransform reproduce the course
     Logit and ImageAffineCoupling arithmetic, given the SAME conditioner,
     to float precision — so the block 3 worksheet ("which lines are
     mathematics, which orchestration, which API") is checked, not asserted;
  8. samples at tau = 0.7 are NOT samples from the model: their mean
     log-density under the model differs from the tau = 1 samples' by a
     measured amount, recorded so the caption can say so.

FONT SIZES are large on purpose (14-16 pt): a figure included at the width of
a 16:9 Beamer frame lands at about half scale (U1.T1 qa-quarto M5).

BOUNDARY. No FID, no variational dequantization, no continuous time, no
architecture tuning for sample quality. The GPU run is not bitwise
reproducible (cuDNN autotuning is left on) and the JSON says so; the CPU
stages are.

Run:  conda run -n torch-gpu python scripts/python/fig_u1_l2_multiscale.py
      conda run -n torch-gpu python scripts/python/fig_u1_l2_multiscale.py --only protocol
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: identical output in CI, SSH, containers
matplotlib.rcParams["svg.hashsalt"] = "u1l2"  # deterministic SVG element ids
# Type 42 (TrueType) glyph outlines, NOT matplotlib's Type 3 default: Type 3
# fonts carry no hinting, so PDF viewers render small axis labels mushy.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.eval import bpd, panel  # noqa: E402
from common.multiscale_flow import (  # noqa: E402
    ImageAffineCoupling,
    Logit,
    MultiscaleFlow,
    Squeeze2d,
    count_params,
    log_p0_flat,
)
from common.train import Trainer, set_seed  # noqa: E402

FIG_DIR = ROOT / "course" / "Figures"
OUT_DIR = ROOT / "scripts" / "python" / "_outputs"
RUN_DIR = OUT_DIR / "u1l2-runs"          # gitignored; the JSON is the artifact
WEIGHTS = ROOT / "common" / "weights"
LABS = ROOT / "labs"
DATA_DIR = ROOT / "scripts" / "python" / "data"
JSON_PATH = OUT_DIR / "u1l2_multiscale.json"

SEED = 0

# --- UFRJ palette (keep in sync with the three theme surfaces in
# --- course/_theme/ — INV-1; verify with ./scripts/check-palette-sync.sh) ---
UFRJ_BLUE = "#003366"
UFRJ_BLUE_LIGHT = "#00447f"
UFRJ_ACCENT = "#d97a1f"
UFRJ_ACCENT_DARK = "#a3540c"
UFRJ_GRAY = "#6b7280"
INK = "#26262a"

# --- the frozen bpd-protocol constants (L3-U1-L2 block 1; the notes' box) ---
PROTOCOL = {
    "dataset": "MNIST", "split": "test", "N": 10_000, "D": 784,
    "dequantization": "uniform, x = (k + u) / 256, u ~ Unif[0,1)^D",
    "K_deq": 1, "deq_seed": 2026, "lam": 1e-6,
    "logit_jacobian": "included in log p_X before eval_harness.bpd",
    "reported": "eval_harness.bpd(log_prob_x, 784) + 8",
}

# --- the two reference configurations and the shared training settings ---
CFG_MS = dict(name="u1l2-multiscale", levels=2, steps=8, hidden=96, s_cap=2.0)
CFG_1L = dict(name="u1l2-onelevel", levels=1, steps=16, hidden=96, s_cap=2.0)
TRAIN = dict(n_train=55_000, n_val=5_000, split_seed=0, batch_size=128, epochs=30,
             lr=3e-4, weight_decay=0.01, warmup_frac=0.05, seed=SEED)
# The notebook's live run (TODO 2.4): CPU, a subset, a few epochs.
LIVE = dict(n_train=10_000, n_val=2_000, batch_size=128, epochs=3, lr=3e-4,
            weight_decay=0.01, warmup_frac=0.05, seed=SEED, levels=2, steps=4, hidden=48)


def set_global_seed(seed: int) -> None:
    """All three streams, plus CUDA: `set_seed` does it, and it is repeated
    here explicitly so the file itself says which streams it seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def apply_house_style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors="#1A1A1A", labelsize=13)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, name: str) -> None:
    for ext in ("svg", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", transparent=True, bbox_inches="tight")
    plt.close(fig)


def git_state() -> dict:
    """Commit SHA and dirty flag: which code produced the checkpoint."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        sha, dirty = "unknown", True
    return {"git_commit": sha, "git_dirty": dirty}


def sha256_of(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().cpu().numpy().tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ===========================================================================
# Data: raw uint8, and the fixed dequantization bank
# ===========================================================================


def load_raw(train: bool) -> torch.Tensor:
    """MNIST as it is stored: uint8 in {0, ..., 255}, shape (N, 1, 28, 28).

    `datasets.MNIST(...).data` is the raw tensor. No transform, no
    `ToTensor()`: that one divides by 255 (not 256) and would hide the factor
    whose logarithm becomes the +8.
    """
    ds = datasets.MNIST(str(DATA_DIR), train=train, download=False)
    k = ds.data.clone().unsqueeze(1)
    assert k.dtype == torch.uint8 and k.shape[1:] == (1, 28, 28)
    return k


def deq_bank(n: int, seed: int) -> torch.Tensor:
    """The frozen noise bank: u ~ Unif[0,1)^D, one draw per image, CPU, seeded."""
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 1, 28, 28, generator=g)


def dequantize(k: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    return (k.float() + u) / 256.0


def train_val_split(k: torch.Tensor, n_train: int, n_val: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(k.shape[0], generator=g)
    return k[perm[:n_train]], k[perm[n_train:n_train + n_val]]


# ===========================================================================
# The training closure: dequantize with FRESH noise, logit, log p_X
# ===========================================================================


class LogProbModel(nn.Module):
    """The Trainer calls model(k) on a raw uint8 batch; return log p_X(x).

    x = (k + u) / 256 with u drawn fresh at every call (training AND the
    Trainer's per-epoch validation, which is therefore a Monte Carlo estimate
    of the same expectation). The FIXED bank of the frozen protocol is used
    only in `evaluate_protocol`, on the test set, once.
    """

    def __init__(self, flow: MultiscaleFlow, lam: float) -> None:
        super().__init__()
        self.flow = flow
        self.logit = Logit(lam)

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        u = torch.rand(k.shape, device=k.device)
        y, ld = self.logit.inverse(dequantize(k, u))
        return self.flow.log_prob(y) + ld


class NLL(nn.Module):
    def forward(self, logp: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return -logp.mean()


def warmup_cosine(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * p))


def nats_to_bpd_deq(nll_nats: float) -> float:
    return nll_nats / (784 * math.log(2.0)) + 8.0


def fit(model: LogProbModel, k_tr: torch.Tensor, k_va: torch.Tensor, cfg: dict,
        device: str, run_name: str, log_every: int = 5) -> dict:
    """The lab's training call: given Trainer, course defaults, no tracker."""
    set_global_seed(cfg["seed"])
    tr = DataLoader(TensorDataset(k_tr, torch.zeros(len(k_tr))), batch_size=cfg["batch_size"],
                    shuffle=True, drop_last=True,
                    generator=torch.Generator().manual_seed(cfg["seed"]))
    va = DataLoader(TensorDataset(k_va, torch.zeros(len(k_va))), batch_size=500)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total = cfg["epochs"] * len(tr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: warmup_cosine(s, total, int(cfg["warmup_frac"] * total)))
    trainer = Trainer(model, opt, NLL(), device, ckpt_dir=RUN_DIR / run_name,
                      log_every=log_every, sched=sched, config=cfg)
    t0 = time.perf_counter()
    hist = trainer.fit(tr, va, cfg["epochs"])
    hist["seconds"] = time.perf_counter() - t0
    hist["val_bpd_deq"] = [nats_to_bpd_deq(v) for v in hist["val_loss"]]
    hist["train_bpd_deq"] = [nats_to_bpd_deq(v) for v in hist["train_loss"]]
    return hist


# ===========================================================================
# The frozen protocol, executed
# ===========================================================================


@torch.no_grad()
def evaluate_protocol(flow: MultiscaleFlow, k_test: torch.Tensor, seed: int = PROTOCOL["deq_seed"],
                      lam: float = PROTOCOL["lam"], chunk: int = 500) -> dict:
    """bpd-protocol on the full test set, on CPU, with the fixed bank."""
    flow.eval()
    logit = Logit(lam)
    u = deq_bank(k_test.shape[0], seed)
    t0 = time.perf_counter()
    logp_x, logp_y, ld_t = [], [], []
    for i in range(0, k_test.shape[0], chunk):
        y, ld = logit.inverse(dequantize(k_test[i:i + chunk], u[i:i + chunk]))
        lp_y = flow.log_prob(y)
        logp_y.append(lp_y)
        ld_t.append(ld)
        logp_x.append(lp_y + ld)
    lp_x = torch.cat(logp_x).double()
    seconds = time.perf_counter() - t0
    b_cont = bpd(lp_x.numpy(), PROTOCOL["D"])
    return {
        "N": int(k_test.shape[0]), "bank_sha256": sha256_of(u),
        "bpd_cont": b_cont, "bpd_deq": b_cont + 8.0,
        "mean_logp_x_nats": float(lp_x.mean()), "mean_logp_y_nats": float(torch.cat(logp_y).double().mean()),
        "mean_logit_logdet_nats": float(torch.cat(ld_t).double().mean()),
        "cpu_seconds": seconds,
        "_lp_x": lp_x,
    }


def load_checkpoint(name: str, device: str = "cpu") -> tuple[MultiscaleFlow, dict]:
    payload = torch.load(WEIGHTS / f"{name}.pt", map_location=device, weights_only=False)
    c = payload["config"]
    flow = MultiscaleFlow((1, 28, 28), levels=c["levels"], steps=c["steps"],
                          hidden=c["hidden"], s_cap=c["s_cap"]).to(device)
    flow.load_state_dict(payload["state_dict"])
    flow.eval()
    return flow, payload


# ===========================================================================
# Stages
# ===========================================================================


def stage_data(rec: dict) -> None:
    """Honesty check 1, the observation-model figure, the atoms figure."""
    set_global_seed(SEED)
    k_test = load_raw(False)
    assert k_test.dtype == torch.uint8
    flat = k_test.flatten()
    counts = torch.bincount(flat.long(), minlength=256).double()
    frac0, frac255 = float(counts[0] / flat.numel()), float(counts[255] / flat.numel())
    n_distinct = int((counts > 0).sum())
    rec["data"] = {
        "test_dtype": str(k_test.dtype), "test_shape": list(k_test.shape),
        "test_min": int(flat.min()), "test_max": int(flat.max()),
        "n_distinct_values": n_distinct,
        "frac_pixels_at_0": frac0, "frac_pixels_at_255": frac255,
        "to_tensor_divides_by": 255,
        "bank_sha256_seed2026_N10000": sha256_of(deq_bank(10_000, PROTOCOL["deq_seed"])),
    }

    # --- u1-l2-observation-model: atoms | T_lambda | its log-Jacobian ---
    lam = PROTOCOL["lam"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    ax.bar(np.arange(256), counts.numpy() / flat.numel(), width=1.0, color=UFRJ_BLUE)
    ax.set_yscale("log")
    ax.set_xlabel("raw pixel value $k$", fontsize=14)
    ax.set_ylabel("fraction of test pixels", fontsize=14)
    ax.set_title(f"the observations are atoms: {100 * frac0:.0f}% at 0, {100 * frac255:.1f}% at 255",
                 fontsize=14)
    apply_house_style(ax)
    xs = np.linspace(1e-4, 1 - 1e-4, 2000)
    a = lam + (1 - 2 * lam) * xs
    ax = axes[1]
    ax.plot(xs, np.log(a) - np.log1p(-a), color=UFRJ_BLUE, lw=2.2)
    ax.set_xlabel("$x = (k+u)/256$", fontsize=14)
    ax.set_ylabel(r"$y = T_\lambda(x)$", fontsize=14)
    ax.set_title(r"logit preprocessing, $\lambda = 10^{-6}$", fontsize=14)
    apply_house_style(ax)
    ax = axes[2]
    ax.plot(xs, np.log(1 - 2 * lam) - np.log(a) - np.log1p(-a), color=UFRJ_ACCENT, lw=2.2)
    ax.set_xlabel("$x$", fontsize=14)
    ax.set_ylabel(r"$\log |T_\lambda'(x)|$ per pixel", fontsize=14)
    ax.set_title("its log-Jacobian: large near 0 and 1", fontsize=14)
    apply_house_style(ax)
    fig.tight_layout()
    save(fig, "u1-l2-observation-model")

    # --- u1-l2-atoms-spike: continuous MLE on atoms is ill-posed ---
    # A mixture of Gaussians centred at the integers, evaluated AT the integers:
    # mean log-density = -log(sigma sqrt(2 pi)) + log(1/256) + ..., unbounded as
    # sigma -> 0. Computed, not sketched: one pixel, 256 atoms, equal weights.
    sigmas = np.logspace(0, -4, 9)
    atoms = np.arange(256, dtype=np.float64)
    mean_logp = []
    for s in sigmas:
        d = (atoms[:, None] - atoms[None, :]) / s
        logp = -0.5 * d ** 2 - math.log(s * math.sqrt(2 * math.pi)) - math.log(256)
        m = logp.max(axis=1, keepdims=True)
        mean_logp.append(float((m[:, 0] + np.log(np.exp(logp - m).sum(axis=1))).mean()))
    mean_logp = np.array(mean_logp)
    assert mean_logp[-1] > mean_logp[0] + 5.0, "the spike argument must show growth"
    rec["data"]["atoms_sigma"] = sigmas.tolist()
    rec["data"]["atoms_mean_logp"] = mean_logp.tolist()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(sigmas, mean_logp, "o-", color=UFRJ_ACCENT, lw=2.0, ms=6)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(r"width $\sigma$ of a Gaussian at each atom  (narrower $\rightarrow$)", fontsize=14)
    ax.set_ylabel("mean log-density at the atoms", fontsize=14)
    ax.set_title("a continuous density evaluated at integer pixels: no maximum", fontsize=14)
    apply_house_style(ax)
    fig.tight_layout()
    save(fig, "u1-l2-atoms-spike")


def stage_train(rec: dict) -> None:
    """Train the two references on the GPU when present; write the package."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    k_all = load_raw(True)
    k_tr, k_va = train_val_split(k_all, TRAIN["n_train"], TRAIN["n_val"], TRAIN["split_seed"])
    rec["train"] = {"device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
                    "bitwise_reproducible": device == "cpu",
                    "settings": {**TRAIN}, "models": {}}
    for cfg in (CFG_MS, CFG_1L):
        set_global_seed(SEED)
        flow = MultiscaleFlow((1, 28, 28), levels=cfg["levels"], steps=cfg["steps"],
                              hidden=cfg["hidden"], s_cap=cfg["s_cap"])
        model = LogProbModel(flow, PROTOCOL["lam"])
        full_cfg = {**cfg, **TRAIN, "lam": PROTOCOL["lam"]}
        hist = fit(model, k_tr, k_va, full_cfg, device, cfg["name"])
        v = hist["val_bpd_deq"]
        assert v[-1] < v[0] - 0.05, f"{cfg['name']}: validation BPD did not decrease"
        payload = {
            "state_dict": {k: t.detach().cpu() for k, t in flow.state_dict().items()},
            "config": {k: cfg[k] for k in ("levels", "steps", "hidden", "s_cap")},
            "train": full_cfg, "seed": SEED, **git_state(),
            "protocol": PROTOCOL, "n_params": count_params(flow),
            "device": device, "seconds": hist["seconds"],
            "curves": {"train_bpd_deq": hist["train_bpd_deq"], "val_bpd_deq": v},
            "selection": "last epoch (no test-set selection; val curve recorded)",
        }
        WEIGHTS.mkdir(parents=True, exist_ok=True)
        path = WEIGHTS / f"{cfg['name']}.pt"
        torch.save(payload, path)
        side = {k: payload[k] for k in ("config", "train", "seed", "git_commit", "git_dirty",
                                        "n_params", "device", "seconds", "selection")}
        side["sha256"] = file_sha256(path)
        side["built_by"] = "scripts/python/fig_u1_l2_multiscale.py (stage train)"
        side["loads_into"] = (f"common.multiscale_flow.MultiscaleFlow((1, 28, 28), levels={cfg['levels']}, "
                              f"steps={cfg['steps']}, hidden={cfg['hidden']}, s_cap={cfg['s_cap']})")
        side["val_bpd_deq_first"], side["val_bpd_deq_last"] = v[0], v[-1]
        (WEIGHTS / f"{cfg['name']}.json").write_text(json.dumps(side, indent=2) + "\n")
        rec["train"]["models"][cfg["name"]] = {
            "config": payload["config"], "n_params": payload["n_params"],
            "seconds": hist["seconds"], "epochs": TRAIN["epochs"],
            "val_bpd_deq": v, "train_bpd_deq": hist["train_bpd_deq"],
            "val_bpd_deq_first": v[0], "val_bpd_deq_last": v[-1], "val_bpd_deq_best": min(v),
            "checkpoint_sha256": side["sha256"],
        }
        print(f"{cfg['name']}: {payload['n_params']} params, {hist['seconds']:.0f} s on {device}, "
              f"val bpd {v[0]:.3f} -> {v[-1]:.3f}")

    # --- u1-l2-bpd-curves ---
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for cfg, color, label in ((CFG_MS, UFRJ_BLUE, "two levels (reference)"),
                              (CFG_1L, UFRJ_ACCENT, "one level (baseline)")):
        v = rec["train"]["models"][cfg["name"]]["val_bpd_deq"]
        ax.plot(range(1, len(v) + 1), v, color=color, lw=2.2, label=label)
    ax.set_xlabel("epoch", fontsize=14)
    ax.set_ylabel("validation bits / dim (dequantized, +8)", fontsize=14)
    ax.legend(fontsize=13, frameon=False)
    apply_house_style(ax)
    fig.tight_layout()
    save(fig, "u1-l2-bpd-curves")


def stage_protocol(rec: dict) -> None:
    """Honesty checks 2, 3, 5, 6: the frozen protocol on both checkpoints."""
    set_global_seed(SEED)
    torch.set_num_threads(4)
    k_test = load_raw(False)
    rec["protocol"] = {"constants": PROTOCOL, "models": {}}
    lp_cache: dict[str, torch.Tensor] = {}
    for name in (CFG_MS["name"], CFG_1L["name"]):
        flow, payload = load_checkpoint(name)
        res = evaluate_protocol(flow, k_test)
        lp_cache[name] = res.pop("_lp_x")
        assert res["bank_sha256"] == rec["data"]["bank_sha256_seed2026_N10000"] if "data" in rec else True
        rec["protocol"]["models"][name] = {**res, "n_params": payload["n_params"]}
        print(f"{name}: bpd_cont {res['bpd_cont']:.4f}  bpd_deq {res['bpd_deq']:.4f}  "
              f"({res['cpu_seconds']:.1f} s CPU for N = {res['N']})")
    ms, ol = CFG_MS["name"], CFG_1L["name"]
    # Compute per image for one density evaluation (log_prob of a batch of 64,
    # divided by 64), counted by torch's FLOP counter: "same number of coupling
    # steps" is not "same compute" when half the steps run at 7x7.
    from torch.utils.flop_counter import FlopCounterMode

    rec["protocol"]["flops_per_image_log_prob"] = {}
    for name in (ms, ol):
        flow, _ = load_checkpoint(name)
        y_probe = Logit(PROTOCOL["lam"]).inverse(dequantize(k_test[:64], deq_bank(64, 1)))[0]
        counter = FlopCounterMode(display=False)
        with counter, torch.no_grad():
            flow.log_prob(y_probe)
        rec["protocol"]["flops_per_image_log_prob"][name] = counter.get_total_flops() / 64
    fl = rec["protocol"]["flops_per_image_log_prob"]
    rec["protocol"]["flops_ratio_onelevel_over_multiscale"] = fl[ol] / fl[ms]
    rec["protocol"]["delta_bpd_deq_onelevel_minus_multiscale"] = (
        rec["protocol"]["models"][ol]["bpd_deq"] - rec["protocol"]["models"][ms]["bpd_deq"])
    # Paired bootstrap over the shared test images (the same u, the same k):
    # the difference is a paired comparison, and its interval says whether the
    # gap between the two models is larger than test-set sampling noise.
    # Per-image difference in BITS PER DIMENSION, one-level minus two-level
    # (the sign of `delta_bpd_deq_onelevel_minus_multiscale`): a higher
    # log-density is a lower bpd, hence the minus.
    d = -(lp_cache[ol] - lp_cache[ms]).numpy() / (784 * math.log(2.0))
    rng = np.random.default_rng(SEED)
    boots = np.array([d[rng.integers(0, d.size, d.size)].mean() for _ in range(2000)])
    rec["protocol"]["paired_bootstrap_delta_ci95"] = [float(np.percentile(boots, 2.5)),
                                                      float(np.percentile(boots, 97.5))]

    # --- check 3: manual vs harness on a fixed 16-image batch, reference model ---
    flow, _ = load_checkpoint(ms)
    logit = Logit(PROTOCOL["lam"])
    u = deq_bank(k_test.shape[0], PROTOCOL["deq_seed"])
    idx = torch.arange(16)
    with torch.no_grad():
        x16 = dequantize(k_test[idx], u[idx])
        y16, ld_t = logit.inverse(x16)
        rows = flow.bookkeeping(y16)
        zs, ld_inv = flow.inverse(y16)
        base = sum(log_p0_flat(z) for z in zs)
        lp_y = flow.log_prob(y16)
        lp_x = lp_y + ld_t
    manual_total = sum(r["coupling_logdet"] + r["base_logprob"] for r in rows)
    # float32 on terms of size 1e3: rounding is ~1e-3; the reference measured 1.2e-4
    assert float((manual_total - lp_y).abs().max()) < 1e-2, "bookkeeping rows must reconcile"
    assert float((base + ld_inv - lp_y).abs().max()) < 1e-2
    assert sum(math.prod(r["latent_shape"]) for r in rows) == 784
    manual_bpd_cont = -float(lp_x.double().mean()) / (784 * math.log(2.0))
    harness_bpd_cont = bpd(lp_x.double().numpy(), 784)
    assert abs(manual_bpd_cont - harness_bpd_cont) < 1e-9
    # squeeze exactness on real data, again
    sq = Squeeze2d(2)
    z_sq, ld_sq = sq.inverse(y16)
    assert torch.equal(sq.forward(z_sq)[0], y16) and float(ld_sq.abs().max()) == 0.0
    rec["protocol"]["batch16"] = {
        "indices": "test[0:16]",
        "rows": [{"level": r["level"], "active_shape": list(r["active_shape"]),
                  "coupling_logdet_mean": float(r["coupling_logdet"].mean()),
                  "latent_shape": list(r["latent_shape"]), "latent_numel": math.prod(r["latent_shape"]),
                  "base_logprob_mean": float(r["base_logprob"].mean())} for r in rows],
        "mean_logp_y": float(lp_y.mean()), "mean_logit_logdet": float(ld_t.mean()),
        "mean_logp_x": float(lp_x.mean()),
        "bpd_cont_manual": manual_bpd_cont, "bpd_cont_harness": harness_bpd_cont,
        "bpd_deq": harness_bpd_cont + 8.0,
        "max_abs_reconciliation_gap": float((manual_total - lp_y).abs().max()),
    }

    # --- check 5: one-sample banks fluctuate around the K-average (1,000 images) ---
    sub = k_test[:1000]
    with torch.no_grad():
        singles = []
        for s in range(8):
            r = evaluate_protocol(flow, sub, seed=PROTOCOL["deq_seed"] + s)
            singles.append(r["bpd_deq"])
        lps = []
        for s in range(16):
            uu = deq_bank(sub.shape[0], 5000 + s)
            y, ld = logit.inverse(dequantize(sub, uu))
            lps.append((flow.log_prob(y) + ld).double())
        lps_t = torch.stack(lps)                          # (16, 1000)
        mean_of_logs = float(lps_t.mean())
        # log of the mean over u (Jensen: >= mean of logs) -> a tighter bound on
        # log mass, named in the notes as the publication-grade variant.
        log_of_means = float(torch.logsumexp(lps_t, dim=0).sub(math.log(16)).mean())
    avg16 = -mean_of_logs / (784 * math.log(2.0)) + 8.0
    iw16 = -log_of_means / (784 * math.log(2.0)) + 8.0
    assert iw16 <= avg16 + 1e-12, "log-mean must give a lower or equal BPD (Jensen)"
    rec["protocol"]["fluctuation"] = {
        "N_subset": 1000, "single_bank_bpd_deq": singles,
        "single_spread": float(max(singles) - min(singles)),
        "single_std": float(np.std(singles)),
        "avg16_bpd_deq": avg16, "logmean16_bpd_deq": iw16,
        "n_singles_below_avg16": int(sum(s < avg16 for s in singles)),
    }
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.scatter(range(1, 9), singles, s=60, color=UFRJ_BLUE, zorder=3,
               label="one bank each (seeds 2026–2033)")
    ax.axhline(avg16, color=UFRJ_ACCENT, lw=2.0, label="average of logs over 16 banks")
    ax.axhline(iw16, color=UFRJ_GRAY, lw=1.6, ls="--", label="log of the mean over 16 banks")
    ax.set_xlabel("dequantization bank", fontsize=14)
    ax.set_ylabel("bits / dim (dequantized), 1,000 images", fontsize=14)
    ax.legend(fontsize=12, frameon=False, loc="best")
    apply_house_style(ax)
    fig.tight_layout()
    save(fig, "u1-l2-mc-fluctuation")


def stage_samples(rec: dict) -> None:
    """Honesty check 8, and the sample panels with a fixed noise bank."""
    set_global_seed(SEED)
    flow, _ = load_checkpoint(CFG_MS["name"])
    base, _ = load_checkpoint(CFG_1L["name"])
    logit = Logit(PROTOCOL["lam"])
    g = torch.Generator().manual_seed(11)
    eps = [torch.randn(64, *s, generator=g) for s in flow.latent_shapes]
    eps_1l = [torch.randn(64, *s, generator=torch.Generator().manual_seed(11)) for s in base.latent_shapes]

    def make_sampler(model: MultiscaleFlow, tau: float, bank: list[torch.Tensor]):
        def sample_fn(_unused: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                y, _ = model.forward([tau * e for e in bank])
                return logit.forward(y)[0]
        return sample_fn

    rec["samples"] = {"n": 64, "eps_seed": 11, "panels": {}}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, (model, tau, bank, key, title) in zip(axes, (
        (flow, 1.0, eps, "multiscale_tau1.0", r"reference, $\tau = 1$ (the model)"),
        (flow, 0.7, eps, "multiscale_tau0.7", r"reference, $\tau = 0.7$ (a heuristic)"),
        (base, 1.0, eps_1l, "onelevel_tau1.0", r"one-level baseline, $\tau = 1$"),
    )):
        fn = make_sampler(model, tau, bank)
        with torch.no_grad():
            xs = fn(torch.empty(0)).clamp(0, 1)
            y_s, ld_s = logit.inverse(xs.clamp(1e-6, 1 - 1e-6))
            lp = (model.log_prob(y_s) + ld_s).double()
        mean_bpd = -float(lp.mean()) / (784 * math.log(2.0)) + 8.0
        rec["samples"]["panels"][key] = {"tau": tau, "mean_bpd_deq_of_samples": mean_bpd,
                                        "frac_pixels_clamped": float(((xs <= 0) | (xs >= 1)).float().mean())}
        canvas = np.ones((8 * 28, 8 * 28), dtype=np.float32)
        for i in range(64):
            r, c = divmod(i, 8)
            canvas[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28] = xs[i, 0].numpy()
        ax.imshow(canvas, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=14)
    fig.tight_layout()
    save(fig, "u1-l2-samples")
    p = rec["samples"]["panels"]
    assert p["multiscale_tau0.7"]["mean_bpd_deq_of_samples"] != p["multiscale_tau1.0"]["mean_bpd_deq_of_samples"]
    rec["samples"]["tau07_minus_tau1_bpd_of_samples"] = (
        p["multiscale_tau0.7"]["mean_bpd_deq_of_samples"] - p["multiscale_tau1.0"]["mean_bpd_deq_of_samples"])
    # The harness's own panel helper, for the notebook's TODO 5.1 cross-check.
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    panel(make_sampler(flow, 1.0, eps), torch.empty(64), RUN_DIR / "panel-check.png", title="tau = 1")


def stage_live(rec: dict) -> None:
    """Pre-validate the notebook's live run (TODO 2.4) on CPU: time and trend."""
    set_global_seed(SEED)
    torch.set_num_threads(4)
    k_all = load_raw(True)
    k_tr, k_va = train_val_split(k_all, LIVE["n_train"], LIVE["n_val"], TRAIN["split_seed"])
    flow = MultiscaleFlow((1, 28, 28), levels=LIVE["levels"], steps=LIVE["steps"], hidden=LIVE["hidden"])
    model = LogProbModel(flow, PROTOCOL["lam"])
    hist = fit(model, k_tr, k_va, {**LIVE, "lam": PROTOCOL["lam"]}, "cpu", "u1l2-live", log_every=1)
    v = hist["val_bpd_deq"]
    assert v[-1] < v[0], "the live run must show a downward held-out BPD trend"
    rec["live"] = {"settings": LIVE, "n_params": count_params(flow), "cpu_seconds": hist["seconds"],
                   "val_bpd_deq": v, "train_bpd_deq": hist["train_bpd_deq"],
                   "val_bpd_deq_first": v[0], "val_bpd_deq_last": v[-1]}
    print(f"live config: {count_params(flow)} params, {hist['seconds']:.0f} s CPU, val bpd {v[0]:.3f} -> {v[-1]:.3f}")


def stage_zuko(rec: dict) -> None:
    """Honesty check 7: Zuko reproduces the course arithmetic, same conditioner."""
    import zuko
    import zuko.transforms as zt

    set_global_seed(SEED)
    lam = PROTOCOL["lam"]
    logit = Logit(lam)
    k = load_raw(False)[:8]
    x = dequantize(k, deq_bank(8, PROTOCOL["deq_seed"])).double()   # float64: the gap is arithmetic, not rounding
    y, ld_course = logit.inverse(x)
    # Zuko/torch SigmoidTransform: _call = sigmoid (generative side of ours),
    # .inv = logit. Its ladj is for x -> sigmoid(x); ours is for the OPPOSITE
    # direction plus the affine squeeze into [lam, 1-lam]: minus, plus D log(1-2lam).
    a = lam + (1 - 2 * lam) * x
    sig = zt.SigmoidTransform()
    y_lib = sig.inv(a)
    ld_lib = -sig.log_abs_det_jacobian(y_lib, a).flatten(1).sum(1) + 784 * math.log(1 - 2 * lam)
    gap_logit_y = float((y_lib - y).abs().max())
    gap_logit_ld = float((ld_lib - ld_course).abs().max())
    assert gap_logit_y < 1e-9 and gap_logit_ld < 1e-8, (gap_logit_y, gap_logit_ld)

    # Coupling: Zuko's CouplingTransform with the COURSE conv conditioner
    # plugged in as `meta`, on the flattened squeezed tensor. Zuko's _call is
    # the density direction, so meta must return the inverse affine map.
    sq = Squeeze2d(2)
    h = sq.squeeze(y)                                   # (8, 4, 14, 14)
    layer = ImageAffineCoupling(4, [True, True, False, False], hidden=16).double()
    hd = h.double()
    mask_vec = torch.zeros(4, 14, 14, dtype=torch.bool)
    mask_vec[:2] = True

    def meta(a_flat: torch.Tensor):
        a_img = a_flat.view(-1, 2, 14, 14)
        s, t = layer._heads(a_img)
        return zt.AffineTransform(loc=(-t * torch.exp(-s)).flatten(1), scale=torch.exp(-s).flatten(1),
                                  event_dim=1)

    ct = zt.CouplingTransform(meta, mask_vec.flatten())
    with torch.no_grad():
        z_lib = ct(hd.flatten(1))
        ladj_lib = ct.log_abs_det_jacobian(hd.flatten(1), z_lib)
        z_course, ld_course_c = layer.inverse(hd)
        back = ct.inv(z_lib)
    gap_coupling = float((z_lib.view_as(hd) - z_course).abs().max())
    gap_coupling_ld = float((ladj_lib - ld_course_c).abs().max())
    gap_back = float((back - hd.flatten(1)).abs().max())
    assert gap_coupling < 1e-10 and gap_coupling_ld < 1e-8 and gap_back < 1e-10
    has_squeeze = any("squeeze" in n.lower() for n in dir(zt)) or any("squeeze" in n.lower() for n in dir(zuko.flows))
    rec["zuko"] = {"version": zuko.__version__,
                   "logit_vs_SigmoidTransform_gap_y": gap_logit_y, "logit_vs_SigmoidTransform_gap_logdet": gap_logit_ld,
                   "coupling_vs_CouplingTransform_gap_z": gap_coupling,
                   "coupling_vs_CouplingTransform_gap_logdet": gap_coupling_ld,
                   "coupling_roundtrip_gap": gap_back,
                   "zuko_has_squeeze_or_factor_out": has_squeeze,
                   "zuko_has_ReshapeTransform": hasattr(zt, "ReshapeTransform")}
    assert not has_squeeze, "the L3 says Zuko supplies no squeeze/factor-out; check the pin"
    print(f"zuko {zuko.__version__}: logit gap {gap_logit_ld:.1e}, coupling gap {gap_coupling_ld:.1e}")


def stage_ps14d(rec: dict) -> None:
    """The cut package for PS1.4(d): CPU-only bookkeeping on frozen outputs."""
    set_global_seed(SEED)
    k_test = load_raw(False)
    flow, payload = load_checkpoint(CFG_MS["name"])
    logit = Logit(PROTOCOL["lam"])
    u_all = deq_bank(k_test.shape[0], PROTOCOL["deq_seed"])
    idx = torch.arange(16)
    k16, u16 = k_test[idx], u_all[idx]
    with torch.no_grad():
        x16 = dequantize(k16, u16)
        y16, ld16 = logit.inverse(x16)
        lp_y = flow.log_prob(y16).double()
    lp_x = lp_y + ld16.double()
    b_cont = bpd(lp_x.numpy(), 784)
    LABS.mkdir(parents=True, exist_ok=True)
    np.savez(LABS / "ps1-4d-bundle.npz",
             k_uint8=k16[:, 0].numpy(), u=u16[:, 0].numpy().astype(np.float32),
             log_prob_y=lp_y.numpy(),
             lam=np.array(PROTOCOL["lam"]), D=np.array(784), deq_seed=np.array(PROTOCOL["deq_seed"]),
             checkpoint_sha256=np.array(file_sha256(WEIGHTS / f"{CFG_MS['name']}.pt")),
             reference_bpd_cont=np.array(b_cont), reference_bpd_deq=np.array(b_cont + 8.0))
    rec["ps14d"] = {"file": "labs/ps1-4d-bundle.npz", "n_images": 16, "indices": "test[0:16]",
                    "mean_log_prob_y": float(lp_y.mean()), "mean_logit_logdet": float(ld16.double().mean()),
                    "mean_log_prob_x": float(lp_x.mean()),
                    "bpd_cont": b_cont, "bpd_deq": b_cont + 8.0,
                    "bundle_sha256": file_sha256(LABS / "ps1-4d-bundle.npz")}
    print(f"PS1.4(d) bundle: 16 images, bpd_deq {b_cont + 8.0:.4f}")


STAGES = {"data": stage_data, "train": stage_train, "protocol": stage_protocol,
          "samples": stage_samples, "live": stage_live, "zuko": stage_zuko, "ps14d": stage_ps14d}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated stage names; merged into the JSON")
    args = ap.parse_args()
    wanted = [s for s in args.only.split(",") if s] or list(STAGES)
    unknown = [s for s in wanted if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stage(s): {unknown}; choose from {list(STAGES)}")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rec: dict = json.loads(JSON_PATH.read_text()) if JSON_PATH.exists() else {}
    for name in wanted:
        t0 = time.perf_counter()
        STAGES[name](rec)
        rec.setdefault("_stage_seconds", {})[name] = time.perf_counter() - t0
        JSON_PATH.write_text(json.dumps(rec, indent=2) + "\n")   # after EVERY stage
        print(f"[{name}] done in {rec['_stage_seconds'][name]:.0f} s")
    rec["_meta"] = {"torch": torch.__version__, "stages_run": wanted, **git_state()}
    JSON_PATH.write_text(json.dumps(rec, indent=2) + "\n")
    print(f"wrote {JSON_PATH}")


if __name__ == "__main__":
    main()
