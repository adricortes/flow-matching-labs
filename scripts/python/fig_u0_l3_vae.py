"""fig_u0_l3_vae.py — U0.L3 session figures + the numbers the notes quote.

Trains the session's VAEs on MNIST for real and writes every figure and every
number the U0.L3 notes/slides cite. Nothing here is illustrative: the latent
scatter is an encoded validation set, the collapsed-dimension count is counted,
and the reduction-convention trap is a run that actually collapsed.

Runs (all seeded, all with the U0.T2 recipe-card defaults):

  dz2       d_z = 2,  the session's live run — latent pictures come from here
  dz16      d_z = 16, the contrast run — sample quality up, latent unreadable,
            and the named NEGATIVE result: nothing collapses at this width
  trap      d_z = 2,  the PLANTED LESSON: reconstruction reduced by MEAN over
            pixels instead of SUM. Arithmetically that is beta = 784 on the KL,
            so the latent is expected to die. We do not assert HOW dead in
            advance — the script measures it and the notes quote the measurement.
  sweep     d_z = 32, 64, 128 plus a decoder-widened d_z = 64 — the capacity
            sweep that carries the posterior-collapse exhibit

Figures written into course/Figures/ (extension-less includes: html/revealjs
pick .svg, beamer picks .pdf):

  vae-tug-of-war       recon and KL per epoch, both runs — T3's anatomy slide,
                       now measured: recon falls, KL rises from ~0
  vae-latent-scatter   d_z = 2: encoded validation means, colored by digit
  vae-manifold-sheet   d_z = 2: decoder over a Gaussian-quantile lattice
  vae-interpolation    slerp vs lerp between two encoded digits + the norm
                       profile that is the actual argument for slerp
  vae-samples-dz       prior samples, d_z = 2 vs d_z = 16, shared epsilon bank
  vae-perdim-kl        per-dimension KL, d_z = 16 vs d_z = 64, with the
                       collapse threshold
  vae-capacity-sweep   total KL and dead-dimension share against d_z, plus the
                       decoder-widening point that isolates the second lever
  vae-reduction-trap   what the wrong reduction does, measured

Numbers land in scripts/python/_outputs/u0l3_vae.json. The notes cite that file
and nothing else.

HONESTY CONTRACT (see quality_reports/manim/ + the U0.T3 precedent). Three
claims in this session are quantitative, and each is asserted from the run, not
from expectation:
  1. "several dimensions of the d_z = 16 model collapse" — counted at the
     KL < 0.01 nat threshold. MEASURED ANSWER: zero. d_z = 16 wastes nothing,
     so the notes say so, the d_z = 16 run ships as a named negative result,
     and the phenomenon is exhibited on the over-provisioned d_z = 64 run plus
     the capacity sweep below (L3-U0-L3 v0.3 section 2F).
  2. "the wrong reduction kills the latent" — measured as the trap run's total
     KL against the correct run's.
  3. "slerp keeps you on the typical set" — a norm profile is computed for both
     paths in BOTH latent dimensions, because the argument is a concentration
     argument and concentration is weak at d_z = 2.

Run:  conda run -n torch-gpu python scripts/python/fig_u0_l3_vae.py
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: identical output in CI, SSH, containers
matplotlib.rcParams["svg.hashsalt"] = "u0l3"  # deterministic SVG element ids
# Type 42 (TrueType) glyph outlines, NOT matplotlib's Type 3 default: Type 3
# fonts carry no hinting, so Acrobat / Preview / browser PDF viewers render
# small axis labels mushy in the Beamer deck. Vector either way; readable only
# this way. (Same reason IEEE/ACM reject Type 3.)
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm as scipy_norm
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# --- UFRJ palette (keep in sync with the three theme surfaces in
# --- course/_theme/ — INV-1; verify with ./scripts/check-palette-sync.sh) ---
UFRJ_BLUE = "#003366"
UFRJ_BLUE_LIGHT = "#00447f"
UFRJ_ACCENT = "#d97a1f"
UFRJ_ACCENT_DARK = "#a3540c"
UFRJ_GRAY = "#6b7280"
UFRJ_GRAY_LIGHT = "#9aa2aa"
INK = "#26262a"

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "course" / "Figures"
DATA_DIR = Path(__file__).resolve().parent / "data"
JSON_OUT = Path(__file__).resolve().parent / "_outputs" / "u0l3_vae.json"

SEED = 0
N_VAL = 6_000          # carved out of the 60k train split; the 10k test split
                       # is never touched here (INV-13: test is read once, by
                       # the U0.L4 evaluation harness)
COLLAPSE_NATS = 0.01   # the literature's usual "this dimension is unused" line
HEAD_WEIGHT_STD = 1e-3  # mu/logvar head weights at init (L3-U0-L3 section 2B)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """Seed every RNG stream this script can touch (python, numpy, torch+CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def apply_house_style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(UFRJ_GRAY)
    ax.spines["bottom"].set_color(UFRJ_GRAY)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.set_axisbelow(True)


def save_pair(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf"):
        path = OUT_DIR / f"{name}.{ext}"
        fig.savefig(path, transparent=True, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


# =========================================================================
# Configuration — an experiment is data, not code (hygiene stack, U0.L2)
# =========================================================================


@dataclass
class Config:
    """One run, fully described. Everything that can vary between runs is here."""

    run_name: str
    d_z: int = 2
    dec_width: int = 64          # base decoder channel count
    enc_width: int = 32          # base encoder channel count
    epochs: int = 15
    batch_size: int = 256
    lr: float = 3.0e-4           # U0.T2 recipe card
    weight_decay: float = 0.01
    warmup_frac: float = 0.03
    seed: int = SEED
    recon_reduction: str = "sum"  # "sum" = correct; "mean" = the planted trap
    tags: dict[str, str] = field(default_factory=dict)


# =========================================================================
# Model — small conv encoder, mirrored deconv decoder, Bernoulli likelihood
# =========================================================================


class Encoder(nn.Module):
    """x -> (mu_phi, logvar_phi). Returns LOG-VARIANCE, never sigma: the head is
    then unconstrained (any real is a valid output) and exp() is the only place
    positivity is imposed."""

    def __init__(self, d_z: int, width: int) -> None:
        super().__init__()
        w = width
        self.body = nn.Sequential(
            nn.Conv2d(1, w, 4, stride=2, padding=1),        # 28 -> 14
            nn.GroupNorm(8, w), nn.SiLU(),
            nn.Conv2d(w, 2 * w, 4, stride=2, padding=1),    # 14 -> 7
            nn.GroupNorm(8, 2 * w), nn.SiLU(),
            nn.Conv2d(2 * w, 4 * w, 3, stride=2, padding=1),  # 7 -> 4
            nn.GroupNorm(8, 4 * w), nn.SiLU(),
            nn.Flatten(),
        )
        self.mu = nn.Linear(4 * w * 16, d_z)
        self.logvar = nn.Linear(4 * w * 16, d_z)
        # Head init (L3-U0-L3 section 2B). ZERO BIASES with SMALL NONZERO
        # WEIGHTS, not a fully zeroed head. At initialization mu ~ 0 and
        # logvar ~ 0, so z is standard normal to within the head's own scale --
        # which is what the lab's checkpoint (mean ~ 0, std ~ 1) checks.
        #
        # Why not zero the weights too, as the recipe card's clause 3 does for
        # an output head: with W = 0 the gradient reaching the encoder TRUNK is
        # exactly zero on the first optimizer step (measured: 0.000000e+00
        # against 4.102861e-03 here), which is also what made the gradient-check
        # ritual below unfalsifiable at step 0.
        for head in (self.mu, self.logvar):
            nn.init.normal_(head.weight, std=HEAD_WEIGHT_STD)
            nn.init.zeros_(head.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    """z -> Bernoulli LOGITS over the 784 pixels (never probabilities: the loss
    is a with-logits binary cross entropy, which is the stable form)."""

    def __init__(self, d_z: int, width: int) -> None:
        super().__init__()
        w = width
        self.proj = nn.Linear(d_z, 4 * w * 16)
        self.w = w
        self.body = nn.Sequential(
            nn.GroupNorm(8, 4 * w), nn.SiLU(),
            nn.ConvTranspose2d(4 * w, 2 * w, 3, stride=2, padding=1),  # 4 -> 7
            nn.GroupNorm(8, 2 * w), nn.SiLU(),
            nn.ConvTranspose2d(2 * w, w, 4, stride=2, padding=1),      # 7 -> 14
            nn.GroupNorm(8, w), nn.SiLU(),
            nn.ConvTranspose2d(w, 1, 4, stride=2, padding=1),          # 14 -> 28
        )
        nn.init.zeros_(self.body[-1].weight)   # zero-init last layer (recipe card)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.proj(z).view(z.shape[0], 4 * self.w, 4, 4)
        return self.body(h)


class VAE(nn.Module):
    """The `vae-mnist-scratch` interface contract, frozen for PS0:

        encode(x) -> (mu, logvar)          [PS0 rebuilds the BODY of this from the
                                            U0.L4 U-Net down-path blocks, via
                                            VAEEncoderAdapter; the contract holds]
        reparameterize(mu, logvar) -> z
        decode(z) -> logits
        loss(x) -> {"loss", "recon", "kl"} [composite losses log their parts]
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg.d_z, cfg.enc_width)
        self.decoder = Decoder(cfg.d_z, cfg.dec_width)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = mu + exp(logvar/2) * eps, eps ~ N(0, I) — U0.T3's @eq-reparam."""
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def loss(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)

        # Reconstruction: SUM over pixels, MEAN over batch. The reduction is the
        # session's planted lesson — "mean over pixels" rescales this term by
        # 1/784 relative to the KL, which is a beta-VAE nobody asked for.
        per_pixel = F.binary_cross_entropy_with_logits(logits, x, reduction="none")
        if self.cfg.recon_reduction == "sum":
            recon = per_pixel.flatten(1).sum(dim=1).mean()
        else:
            recon = per_pixel.flatten(1).mean(dim=1).mean()

        kl_per_dim = kl_diag_gaussian(mu, logvar)      # (B, d_z), kept per-dim
        kl = kl_per_dim.sum(dim=1).mean()
        return {"loss": recon + kl, "recon": recon, "kl": kl,
                "kl_per_dim": kl_per_dim.mean(dim=0)}


def kl_diag_gaussian(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(N(mu, diag(sigma^2)) || N(0, I)), per dimension, per example.

    The closed form quoted in U0.T3 (@eq-gauss-kl) and derived in UP.A:
        1/2 * (mu_j^2 + sigma_j^2 - 1 - 2 log sigma_j),  sigma_j^2 = exp(logvar_j)
    Kept per-dimension rather than summed: the posterior-collapse block reads
    exactly this tensor, dimension by dimension.
    """
    return 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)


# =========================================================================
# Data — dynamic binarization, and a val split carved out of train
# =========================================================================


def make_loaders(cfg: Config) -> tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
    """Train/val loaders plus a FIXED validation batch and its labels.

    The fixed batch (and the fixed epsilon bank in the caller) is what makes
    per-epoch panels comparable across epochs and across runs — the same trick
    the U3 labs use for sample grids.
    """
    tf = transforms.ToTensor()   # [0, 1] grayscale; binarization happens per draw
    full = datasets.MNIST(DATA_DIR, train=True, download=False, transform=tf)

    g = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(len(full), generator=g)
    train_idx, val_idx = perm[:-N_VAL].tolist(), perm[-N_VAL:].tolist()

    def seed_worker(_: int) -> None:
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)

    train_loader = DataLoader(
        Subset(full, train_idx), batch_size=cfg.batch_size, shuffle=True,
        drop_last=True, num_workers=2, worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    val_loader = DataLoader(
        Subset(full, val_idx), batch_size=512, shuffle=False, num_workers=2,
    )

    fixed_x = torch.stack([full[i][0] for i in val_idx[:64]]).to(DEVICE)
    fixed_y = torch.tensor([full[i][1] for i in val_idx[:64]])
    return train_loader, val_loader, fixed_x, fixed_y


def binarize(x: torch.Tensor) -> torch.Tensor:
    """Dynamic binarization: resample Bernoulli(x) every time x is drawn.

    This is the decoder-likelihood decision made honest. A Bernoulli decoder
    models binary pixels, so the data it is fed must be binary; feeding
    grayscale to a Bernoulli likelihood is the classic silent pairing bug.
    """
    return torch.bernoulli(x)


# =========================================================================
# Training — the hygiene stack's shape (config in, per-term metrics out)
# =========================================================================


def warmup_cosine(step: int, total: int, warmup: int) -> float:
    """Multiplier on the base LR — the U0.T2 default schedule, stepped per step."""
    if step < warmup:
        return step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * p))


@torch.no_grad()
def evaluate(model: VAE, loader: DataLoader) -> dict[str, float | np.ndarray]:
    """Per-term validation metrics, including the per-dimension KL vector."""
    model.eval()
    tot = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
    kl_dim = torch.zeros(model.cfg.d_z, device=DEVICE)
    n_batches = 0
    for x, _ in loader:
        x = binarize(x.to(DEVICE))
        out = model.loss(x)
        for k in tot:
            tot[k] += float(out[k])
        kl_dim += out["kl_per_dim"]
        n_batches += 1
    model.train()
    return {**{k: v / n_batches for k, v in tot.items()},
            "kl_per_dim": (kl_dim / n_batches).cpu().numpy()}


def train(cfg: Config, on_capture: Callable[[VAE, str], None] | None = None,
          capture_steps: set[int] | None = None) -> dict:
    """Train one VAE. Returns the model, its history, and its validation metrics.

    `on_capture(model, label)` is an optional observer called at initialization,
    at the given global steps of epoch 1, and after every epoch's evaluation.
    It exists so the Manim scene data is produced by THIS loop rather than by a
    copy of it: a re-implemented loop drifts from the committed run the moment
    either side changes how much randomness it consumes (which is exactly how
    the first version of `u0_l3_scene_data.py` ended up 0.05 nats away). The
    observer must not consume global randomness — see that file's `capture`.
    """
    set_seed(cfg.seed)
    train_loader, val_loader, fixed_x, fixed_y = make_loaders(cfg)
    model = VAE(cfg).to(DEVICE)

    # Checkpoint assert from the lab's Part B. With zero biases and small
    # nonzero head weights the latent is standard normal TO WITHIN THE HEAD'S
    # OWN SCALE — approximately, not exactly, which is what L3-U0-L3 section 2B
    # asks for ("mean ~ 0, std ~ 1"). If this ever fails, the callback in the
    # notes is a lie — so it is an assert, not a print.
    with torch.no_grad():
        mu0, logvar0 = model.encode(binarize(fixed_x))
        z0 = model.reparameterize(mu0, logvar0)
    assert abs(float(mu0.abs().max())) < 0.5, "init mu head is not near zero"
    assert abs(float(logvar0.abs().max())) < 0.5, "init logvar head is not near zero"
    init_z_mean, init_z_std = float(z0.mean()), float(z0.std())

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * len(train_loader)
    warmup_steps = int(cfg.warmup_frac * total_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: warmup_cosine(s, total_steps, warmup_steps)
    )

    if on_capture is not None:
        on_capture(model, "init")

    history: dict[str, list[float]] = {"epoch": [], "recon": [], "kl": [], "loss": []}
    step = 0
    for epoch in range(1, cfg.epochs + 1):
        for x, _ in train_loader:
            x = binarize(x.to(DEVICE))
            out = model.loss(x)
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            opt.step()
            sched.step()
            step += 1
            if on_capture is not None and capture_steps and epoch == 1 \
                    and step in capture_steps:
                on_capture(model, f"step {step}")
        val = evaluate(model, val_loader)
        if on_capture is not None:
            on_capture(model, f"epoch {epoch}")
        history["epoch"].append(epoch)
        for k in ("recon", "kl", "loss"):
            history[k].append(float(val[k]))
        print(f"[{cfg.run_name}] epoch {epoch:2d}  "
              f"loss {val['loss']:8.3f}  recon {val['recon']:8.3f}  kl {val['kl']:7.3f}")

    final = evaluate(model, val_loader)
    return {"cfg": cfg, "model": model, "history": history, "final": final,
            "val_loader": val_loader, "fixed_x": fixed_x, "fixed_y": fixed_y,
            "init_z_mean": init_z_mean, "init_z_std": init_z_std}


def gradient_check(cfg: Config, corrupt: bool = False) -> dict[str, float]:
    """The gradient-check ritual (U0.L1), applied to the hand-written ELBO.

    Finite differences against autograd on one encoder tensor, in float64 with
    the sampling noise HELD FIXED — otherwise the two evaluations are of two
    different random functions and the check compares noise to noise. That
    subtlety is the whole reason the ritual is taught on a stochastic loss.

    The check is DIRECTIONAL, in U0.L1's form: one fixed unit direction v,
    autograd's <grad, v> against the central difference of the loss along v.
    Two forward evaluations, whatever the tensor's size. The elementwise scan
    is kept below as the optional cross-check U0.L1 also keeps.

    TWO TRAPS this function exists to avoid, both of which we walked into:

    1. AT INITIALIZATION THE CHECK CANNOT FAIL — it used to. With the heads'
       weights zeroed the gradient reaching any encoder BODY tensor is exactly
       zero, so the check reported a perfect score from a measurement of
       nothing. The heads now carry small nonzero weights (section 2B), which
       removes the cause; we still take a few optimizer steps first and ASSERT
       the analytic gradient is nonzero before trusting any agreement.
    2. A CHECK THAT HAS NEVER FIRED IS NOT KNOWN TO WORK. `corrupt=True`
       differentiates a WRONG ELBO (the KL's 1/2 dropped) against the finite
       differences of the correct one; the ritual must report a large error
       there. Both numbers ship in the JSON, so the notes can quote a check
       that passes AND the same check failing on a bug.
    """
    set_seed(cfg.seed)
    torch.set_default_dtype(torch.float64)
    model = VAE(cfg).to("cpu").double()
    x = torch.bernoulli(torch.rand(8, 1, 28, 28, dtype=torch.float64))
    eps = torch.randn(8, cfg.d_z, dtype=torch.float64)   # FIXED noise

    # Walk off the zero-gradient initialization (trap 1).
    warm_opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(30):
        out = model.loss(x)
        warm_opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        warm_opt.step()

    def elbo_loss(params_flat: torch.Tensor, param: nn.Parameter,
                  kl_scale: float = 0.5) -> torch.Tensor:
        with torch.no_grad():
            param.copy_(params_flat.view_as(param))
        mu, logvar = model.encode(x)
        z = mu + torch.exp(0.5 * logvar) * eps
        per_pixel = F.binary_cross_entropy_with_logits(model.decode(z), x, reduction="none")
        recon = per_pixel.flatten(1).sum(dim=1).mean()
        # kl_scale = 0.5 is the correct closed form; corrupt=True passes 1.0,
        # i.e. the classic "dropped the 1/2" bug.
        kl = (2.0 * kl_scale) * kl_diag_gaussian(mu, logvar).sum(dim=1).mean()
        return recon + kl

    param = model.encoder.body[0].weight
    base = param.detach().clone().flatten()
    model.zero_grad(set_to_none=True)
    elbo_loss(base, param, kl_scale=1.0 if corrupt else 0.5).backward()
    analytic = param.grad.detach().flatten().clone()
    analytic_norm = float(analytic.norm())
    assert analytic_norm > 1e-8, (
        "analytic gradient is zero — the check would pass vacuously "
        "(this is what zero-init heads do at step 0)"
    )

    h = 1e-6

    # --- the ritual proper: one fixed unit direction, two forward evaluations
    v = torch.randn_like(base)
    v /= v.norm()
    d_ad = float(analytic @ v)
    with torch.no_grad():       # finite differences always on the CORRECT loss
        lo = float(elbo_loss(base - h * v, param))
        hi = float(elbo_loss(base + h * v, param))
    d_fd = (hi - lo) / (2 * h)
    atol, rtol = 1e-6, 1e-6     # atol floors the float64 cancellation noise,
    tol = atol + rtol * max(abs(d_ad), abs(d_fd))   # which is ~1e-8 at |f| ~ 150
    accepted = abs(d_ad - d_fd) <= tol

    # --- optional cross-check: the same comparison, coordinate by coordinate
    idx = list(range(0, base.numel(), max(1, base.numel() // 12)))[:12]
    numeric = []
    for i in idx:
        plus, minus = base.clone(), base.clone()
        plus[i] += h
        minus[i] -= h
        with torch.no_grad():
            numeric.append(
                float((elbo_loss(plus, param) - elbo_loss(minus, param)) / (2 * h))
            )
    numeric_arr = np.array(numeric)
    analytic_arr = analytic[idx].numpy()
    denom = np.maximum(np.abs(numeric_arr) + np.abs(analytic_arr), 1e-12)
    rel = float(np.max(np.abs(numeric_arr - analytic_arr) / denom))

    with torch.no_grad():       # leave the model as we found it
        param.copy_(base.view_as(param))
    torch.set_default_dtype(torch.float32)
    return {"d_ad": d_ad, "d_fd": d_fd, "gap": abs(d_ad - d_fd), "tol": tol,
            "accepted": bool(accepted), "atol": atol, "rtol": rtol,
            "max_relative_error": rel, "n_entries_checked": len(idx), "h": h,
            "analytic_grad_norm": analytic_norm, "warmup_steps": 30}


# =========================================================================
# Latent-space instruments
# =========================================================================


@torch.no_grad()
def encode_split(model: VAE, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """Encode a whole split to its posterior MEANS mu_phi(x), with labels."""
    model.eval()
    mus, ys = [], []
    for x, y in loader:
        mu, _ = model.encode(binarize(x.to(DEVICE)))
        mus.append(mu.cpu().numpy())
        ys.append(y.numpy())
    model.train()
    return np.concatenate(mus), np.concatenate(ys)


@torch.no_grad()
def decode_grid(model: VAE, zs: torch.Tensor) -> np.ndarray:
    """Decode latents to Bernoulli MEANS (sigmoid of the logits), as images."""
    model.eval()
    imgs = torch.sigmoid(model.decode(zs.to(DEVICE))).cpu().numpy()
    model.train()
    return imgs[:, 0]


def slerp(z0: np.ndarray, z1: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """Spherical linear interpolation between two latents."""
    n0, n1 = np.linalg.norm(z0), np.linalg.norm(z1)
    omega = np.arccos(np.clip(np.dot(z0, z1) / (n0 * n1), -1.0, 1.0))
    if omega < 1e-8:
        return np.outer(1 - ts, z0) + np.outer(ts, z1)
    s = np.sin(omega)
    return (np.outer(np.sin((1 - ts) * omega) / s, z0)
            + np.outer(np.sin(ts * omega) / s, z1))


def lerp(z0: np.ndarray, z1: np.ndarray, ts: np.ndarray) -> np.ndarray:
    return np.outer(1 - ts, z0) + np.outer(ts, z1)


def tile(images: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Lay (N, 28, 28) out as one (rows*28, cols*28) canvas."""
    h = w = images.shape[-1]
    canvas = np.ones((rows * h, cols * w))
    for k in range(min(len(images), rows * cols)):
        r, c = divmod(k, cols)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = images[k]
    return canvas


def imshow_grid(ax: plt.Axes, canvas: np.ndarray, title: str = "") -> None:
    ax.imshow(1.0 - canvas, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color(UFRJ_GRAY_LIGHT)
        side.set_linewidth(0.8)
    if title:
        ax.set_title(title, fontsize=10, color=INK, pad=6)


# =========================================================================
# Figures
# =========================================================================


def fig_tug_of_war(run2: dict, run16: dict) -> None:
    """T3's anatomy slide, measured: recon falls while KL rises from ~0."""
    fig, (ax_r, ax_k) = plt.subplots(1, 2, figsize=(9.6, 3.4))
    for run, color, label in ((run2, UFRJ_BLUE, "$d_z = 2$"),
                              (run16, UFRJ_ACCENT, "$d_z = 16$")):
        h = run["history"]
        ax_r.plot(h["epoch"], h["recon"], color=color, lw=1.8, label=label)
        ax_k.plot(h["epoch"], h["kl"], color=color, lw=1.8, label=label)
    ax_r.set_title("Reconstruction term (nats/image)", fontsize=10, color=INK)
    ax_k.set_title("KL term (nats/image)", fontsize=10, color=INK)
    for ax in (ax_r, ax_k):
        apply_house_style(ax)
        ax.set_xlabel("epoch", fontsize=9, color=INK)
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    save_pair(fig, "vae-tug-of-war")


def fig_latent_scatter(mu2: np.ndarray, y2: np.ndarray) -> None:
    """The payoff picture: clusters that no label ever produced."""
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    cmap = plt.get_cmap("tab10")
    for digit in range(10):
        m = y2 == digit
        ax.scatter(mu2[m, 0], mu2[m, 1], s=3.0, alpha=0.55,
                   color=cmap(digit), linewidths=0, label=str(digit))
    theta = np.linspace(0, 2 * np.pi, 200)
    for r, style in ((1.0, "-"), (2.0, "--")):
        ax.plot(r * np.cos(theta), r * np.sin(theta), style,
                color=UFRJ_GRAY, lw=1.0, alpha=0.8)
    ax.annotate(r"prior $1\sigma$, $2\sigma$",
                xy=(2.0 * np.cos(3.9), 2.0 * np.sin(3.9)), xytext=(-3.6, -3.1),
                fontsize=8, color=UFRJ_GRAY,
                arrowprops={"arrowstyle": "-", "color": UFRJ_GRAY, "lw": 0.7})
    apply_house_style(ax)
    ax.set_xlabel(r"$\mu_\varphi(x)_1$", fontsize=10, color=INK)
    ax.set_ylabel(r"$\mu_\varphi(x)_2$", fontsize=10, color=INK)
    ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=8, markerscale=3.0, ncol=2,
              loc="center left", bbox_to_anchor=(1.01, 0.5), title="digit")
    fig.tight_layout()
    save_pair(fig, "vae-latent-scatter")


def fig_manifold_sheet(model: VAE, n: int = 16) -> None:
    """The decoder over a Gaussian-quantile lattice — the classic manifold sheet."""
    qs = scipy_norm.ppf(np.linspace(0.02, 0.98, n))
    zz = np.stack(np.meshgrid(qs, -qs, indexing="xy"), axis=-1).reshape(-1, 2)
    imgs = decode_grid(model, torch.tensor(zz, dtype=torch.float32))
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    imshow_grid(ax, tile(imgs, n, n))
    ax.set_xlabel(r"$z_1$ (Gaussian quantiles)", fontsize=9, color=INK)
    ax.set_ylabel(r"$z_2$ (Gaussian quantiles)", fontsize=9, color=INK)
    fig.tight_layout()
    save_pair(fig, "vae-manifold-sheet")


DIGIT_FROM, DIGIT_TO = 0, 1   # two digits a d_z = 2 decoder renders unambiguously


def fig_interpolation(model: VAE, mu2: np.ndarray, y2: np.ndarray,
                      norms: dict) -> dict:
    """slerp vs lerp, decoded, plus the norm profile that is the real argument.

    The right panel plots the norm deficit in units of the PRIOR'S OWN norm
    spread, not in raw nats or raw norm. That is the honest version of the
    folklore: the relative size of lerp's midpoint dip is set by the ANGLE
    between the endpoints and is not larger in high dimension (measured: the
    relative dip here is bigger at d_z = 2 than at d_z = 16). What grows with
    d_z is how many standard deviations of the prior's norm distribution that
    same dip represents — which is what "off the typical set" actually means.
    """
    i0 = int(np.where(y2 == DIGIT_FROM)[0][0])
    i1 = int(np.where(y2 == DIGIT_TO)[0][0])
    z0, z1 = mu2[i0].astype(np.float64), mu2[i1].astype(np.float64)
    ts = np.linspace(0.0, 1.0, 9)
    path_s, path_l = slerp(z0, z1, ts), lerp(z0, z1, ts)
    img_s = decode_grid(model, torch.tensor(path_s, dtype=torch.float32))
    img_l = decode_grid(model, torch.tensor(path_l, dtype=torch.float32))

    fig = plt.figure(figsize=(9.8, 3.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.4, 1.15], hspace=0.45, wspace=0.24)
    ax_s = fig.add_subplot(gs[0, 0])
    ax_l = fig.add_subplot(gs[1, 0])
    ax_n = fig.add_subplot(gs[:, 1])
    imshow_grid(ax_s, tile(img_s, 1, 9), "spherical interpolation (slerp)")
    imshow_grid(ax_l, tile(img_l, 1, 9), "straight-line interpolation (lerp)")

    for tag, d, style in (("dz2", 2, ":"), ("dz16", 16, "-")):
        mean_r, sd_r = chi_moments(d)
        for key, color in (("slerp", UFRJ_BLUE), ("lerp", UFRJ_ACCENT)):
            dev = (np.array(norms[f"{tag}_{key}"]) - mean_r) / sd_r
            ax_n.plot(norms["ts"], dev, style, color=color, lw=1.6,
                      label=f"{key}, $d_z={d}$")
    ax_n.axhline(0.0, color=UFRJ_GRAY, ls="--", lw=1.0)
    apply_house_style(ax_n)
    ax_n.set_title(r"$\|z\|$ vs the prior's typical norm", fontsize=9, color=INK)
    ax_n.set_ylabel("standard deviations", fontsize=8, color=INK)
    ax_n.set_xlabel("interpolation parameter", fontsize=9, color=INK)
    ax_n.legend(frameon=False, fontsize=7, ncol=2, loc="lower center")
    fig.tight_layout()
    save_pair(fig, "vae-interpolation")

    mid = len(norms["ts"]) // 2
    out = {"digit_from": int(y2[i0]), "digit_to": int(y2[i1])}
    for tag, d in (("dz2", 2), ("dz16", 16)):
        mean_r, sd_r = chi_moments(d)
        out[f"{tag}_prior_norm_mean"] = mean_r
        out[f"{tag}_prior_norm_sd"] = sd_r
        out[f"{tag}_lerp_midpoint_sd_below"] = (mean_r - norms[f"{tag}_lerp"][mid]) / sd_r
        out[f"{tag}_slerp_midpoint_sd_below"] = (mean_r - norms[f"{tag}_slerp"][mid]) / sd_r
    return out


def chi_moments(d: int) -> tuple[float, float]:
    """Mean and standard deviation of ||z|| for z ~ N(0, I_d) (a chi_d variable).

    E||z|| = sqrt(2) * Gamma((d+1)/2) / Gamma(d/2);  Var = d - (E||z||)^2.
    The standard deviation tends to 1/sqrt(2) as d grows while the mean grows
    like sqrt(d) — which is the concentration that makes the slerp argument bite.
    """
    mean = math.sqrt(2.0) * math.exp(math.lgamma((d + 1) / 2) - math.lgamma(d / 2))
    return mean, math.sqrt(max(d - mean**2, 1e-12))


def fig_samples(run2: dict, run16: dict, n: int = 8) -> None:
    """Prior samples from both models, drawn from ONE shared epsilon bank."""
    set_seed(SEED + 1)
    eps16 = torch.randn(n * n, 16)
    eps2 = eps16[:, :2].contiguous()   # same bank, truncated: panels are paired
    img2 = decode_grid(run2["model"], eps2)
    img16 = decode_grid(run16["model"], eps16)
    fig, (a2, a16) = plt.subplots(1, 2, figsize=(9.0, 4.6))
    imshow_grid(a2, tile(img2, n, n), r"$d_z = 2$")
    imshow_grid(a16, tile(img16, n, n), r"$d_z = 16$")
    fig.tight_layout()
    save_pair(fig, "vae-samples-dz")


def fig_perdim_kl(run16: dict, run64: dict) -> dict:
    """The instrument the posterior-collapse block reads. Sorted, thresholded.

    THE HONEST PAIRING. The L3 pre-plan expected the d_z = 16 run to exhibit
    collapse; it does not — every one of its 16 dimensions carries more than a
    nat. So the exhibit is a CONTRAST: the session's own contrast run (nothing
    dead) beside a deliberately over-provisioned one (half the dimensions
    unused). The phenomenon is real; it needs capacity the L3 did not give it.
    """
    kl16 = np.sort(run16["final"]["kl_per_dim"])[::-1]
    kl64 = np.sort(run64["final"]["kl_per_dim"])[::-1]
    dead16 = int((kl16 < COLLAPSE_NATS).sum())
    dead64 = int((kl64 < COLLAPSE_NATS).sum())
    weak64 = int((kl64 < 0.1).sum())

    fig, (a16, a64) = plt.subplots(1, 2, figsize=(9.8, 3.5),
                                   gridspec_kw={"width_ratios": [1.0, 3.0]})
    for ax, kl, title in ((a16, kl16, r"$d_z = 16$ — the session's run"),
                          (a64, kl64, r"$d_z = 64$ — over-provisioned")):
        colors = [UFRJ_GRAY_LIGHT if v < COLLAPSE_NATS else UFRJ_BLUE for v in kl]
        ax.bar(np.arange(len(kl)), np.maximum(kl, 1e-4), color=colors, width=0.8)
        ax.axhline(COLLAPSE_NATS, color=UFRJ_ACCENT, ls="--", lw=1.2)
        apply_house_style(ax)
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 5.0)
        ax.set_title(title, fontsize=10, color=INK)
        ax.set_xlabel("latent dimension (sorted)", fontsize=9, color=INK)
    a16.set_ylabel("KL (nats)", fontsize=9, color=INK)
    a16.text(len(kl16) - 0.5, 3.1, f"none below {COLLAPSE_NATS} nats",
             fontsize=8, color=UFRJ_ACCENT_DARK, ha="right")
    a64.text(len(kl64) - 0.5, 3.1,
             f"{dead64} below {COLLAPSE_NATS} nats · {weak64} below 0.1",
             fontsize=8, color=UFRJ_ACCENT_DARK, ha="right")
    fig.tight_layout()
    save_pair(fig, "vae-perdim-kl")
    return {"dead_dz16": dead16, "dead_dz64": dead64, "weak_dz64": weak64,
            "kl_sorted_dz16": kl16.tolist(), "kl_sorted_dz64": kl64.tolist()}


def fig_capacity_sweep(sweep: list[dict]) -> None:
    """Where collapse comes from: capacity in excess of the information budget.

    Left: total KL against d_z — the model's information budget, which
    saturates, so extra dimensions carry nothing. Right: how many dimensions
    are dead or nearly dead. The lone orange point holds d_z fixed and widens
    the DECODER, isolating the second lever (a stronger decoder can afford to
    ignore more of z).
    """
    base = [s for s in sweep if s["dec_width"] == 64]
    wide = [s for s in sweep if s["dec_width"] != 64]
    dz = [s["d_z"] for s in base]

    fig, (ax_kl, ax_dead) = plt.subplots(1, 2, figsize=(9.6, 3.4))
    ax_kl.plot(dz, [s["total_kl"] for s in base], "o-", color=UFRJ_BLUE,
               lw=1.8, ms=5, label="total KL (nats)")
    ax_kl.plot(dz, [s["recon"] for s in base], "s--", color=UFRJ_GRAY,
               lw=1.4, ms=4, label="reconstruction (nats)")
    apply_house_style(ax_kl)
    ax_kl.set_xscale("log", base=2)
    ax_kl.set_xticks(dz)
    ax_kl.set_xticklabels([str(d) for d in dz])
    ax_kl.set_xlabel(r"latent dimension $d_z$", fontsize=9, color=INK)
    ax_kl.set_ylabel("nats per image", fontsize=9, color=INK)
    ax_kl.set_title("What the model actually uses", fontsize=10, color=INK)
    ax_kl.legend(frameon=False, fontsize=8, loc="center right")

    ax_dead.plot(dz, [100.0 * s["weak"] / s["d_z"] for s in base], "o-",
                 color=UFRJ_BLUE, lw=1.8, ms=5, label="below 0.1 nats")
    ax_dead.plot(dz, [100.0 * s["dead"] / s["d_z"] for s in base], "^-",
                 color=UFRJ_BLUE_LIGHT, lw=1.4, ms=5,
                 label=f"below {COLLAPSE_NATS} nats")
    for s in wide:
        ax_dead.plot([s["d_z"]], [100.0 * s["dead"] / s["d_z"]], "*",
                     color=UFRJ_ACCENT, ms=13,
                     label=f"same $d_z$, decoder {s['dec_width']}/64 wide")
    apply_house_style(ax_dead)
    ax_dead.set_xscale("log", base=2)
    ax_dead.set_xticks(dz)
    ax_dead.set_xticklabels([str(d) for d in dz])
    ax_dead.set_xlabel(r"latent dimension $d_z$", fontsize=9, color=INK)
    ax_dead.set_ylabel("% of dimensions", fontsize=9, color=INK)
    ax_dead.set_title("Unused dimensions", fontsize=10, color=INK)
    ax_dead.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    save_pair(fig, "vae-capacity-sweep")


def fig_reduction_trap(run2: dict, trap: dict) -> None:
    """What the wrong reduction actually does — measured, not threatened."""
    fig = plt.figure(figsize=(9.6, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.4, 1.0, 1.0], wspace=0.28)
    ax_kl = fig.add_subplot(gs[0, 0])
    ax_ok = fig.add_subplot(gs[0, 1])
    ax_bad = fig.add_subplot(gs[0, 2])

    for run, color, label in ((run2, UFRJ_BLUE, "sum over pixels (correct)"),
                              (trap, UFRJ_ACCENT, "mean over pixels (the trap)")):
        h = run["history"]
        ax_kl.plot(h["epoch"], np.maximum(h["kl"], 1e-6), color=color, lw=1.8, label=label)
    apply_house_style(ax_kl)
    ax_kl.set_yscale("log")
    ax_kl.set_title("KL term (nats/image)", fontsize=10, color=INK)
    ax_kl.set_xlabel("epoch", fontsize=9, color=INK)
    ax_kl.legend(frameon=False, fontsize=8)

    set_seed(SEED + 2)
    eps = torch.randn(16, 2)
    imshow_grid(ax_ok, tile(decode_grid(run2["model"], eps), 4, 4), "samples, correct")
    imshow_grid(ax_bad, tile(decode_grid(trap["model"], eps), 4, 4), "samples, trap")
    fig.tight_layout()
    save_pair(fig, "vae-reduction-trap")


def norm_profiles(mu2: np.ndarray, y2: np.ndarray,
                  mu16: np.ndarray, y16: np.ndarray) -> dict:
    """Norm along slerp and lerp paths in BOTH latent dimensions.

    The typical-set argument for slerp is a concentration argument: draws from
    N(0, I_d) have norm concentrating at sqrt(d), and the straight line between
    two of them passes through a shorter-norm region the model never saw. The
    effect therefore GROWS with d_z, and at d_z = 2 it is mild. We measure both
    so the notes can say which it is instead of asserting the folklore.
    """
    ts = np.linspace(0.0, 1.0, 41)
    out: dict[str, list[float]] = {"ts": ts.tolist()}
    for tag, mu, y in (("dz2", mu2, y2), ("dz16", mu16, y16)):
        i0 = int(np.where(y == DIGIT_FROM)[0][0])
        i1 = int(np.where(y == DIGIT_TO)[0][0])
        z0, z1 = mu[i0].astype(np.float64), mu[i1].astype(np.float64)
        out[f"{tag}_slerp"] = np.linalg.norm(slerp(z0, z1, ts), axis=1).tolist()
        out[f"{tag}_lerp"] = np.linalg.norm(lerp(z0, z1, ts), axis=1).tolist()
    return out


# =========================================================================
# Main
# =========================================================================


def main() -> None:
    print(f"device: {DEVICE}")
    cfg2 = Config(run_name="U0L3-dz2-s0", d_z=2, tags={"session": "U0.L3"})
    cfg16 = Config(run_name="U0L3-dz16-s0", d_z=16, tags={"session": "U0.L3"})
    cfg_trap = Config(run_name="U0L3-trap-s0", d_z=2, recon_reduction="mean",
                      tags={"session": "U0.L3", "role": "planted-lesson"})
    # The capacity sweep behind the posterior-collapse block. d_z = 16 does NOT
    # collapse (measured), so the exhibit needs the over-provisioned runs; the
    # last entry holds d_z fixed and widens the decoder instead.
    sweep_cfgs = [
        Config(run_name="U0L3-dz32-s0", d_z=32, tags={"role": "capacity-sweep"}),
        Config(run_name="U0L3-dz64-s0", d_z=64, tags={"role": "capacity-sweep"}),
        Config(run_name="U0L3-dz128-s0", d_z=128, tags={"role": "capacity-sweep"}),
        Config(run_name="U0L3-dz64-wide-s0", d_z=64, dec_width=160,
               tags={"role": "capacity-sweep-decoder-power"}),
    ]

    grad_ok = gradient_check(cfg2)
    grad_bad = gradient_check(cfg2, corrupt=True)
    print(f"gradient check (correct loss): d_AD {grad_ok['d_ad']:.10e} "
          f"d_FD {grad_ok['d_fd']:.10e} gap {grad_ok['gap']:.2e} "
          f"(tol {grad_ok['tol']:.2e}) accepted={grad_ok['accepted']}")
    print(f"gradient check (KL's 1/2 dropped): gap {grad_bad['gap']:.2e} "
          f"(tol {grad_bad['tol']:.2e}) accepted={grad_bad['accepted']}"
          "  <- the ritual must REJECT here")
    assert grad_ok["accepted"], "directional ritual failed on the correct loss"
    assert not grad_bad["accepted"], (
        "directional ritual ACCEPTED a deliberately wrong loss — it detects nothing"
    )
    assert grad_ok["max_relative_error"] < 1e-5, "cross-check failed on the correct loss"
    assert grad_bad["max_relative_error"] > 1e-2, (
        "cross-check PASSED on a deliberately wrong loss"
    )

    run2 = train(cfg2)
    run16 = train(cfg16)
    trap = train(cfg_trap)

    sweep: list[dict] = []
    sweep_runs: dict[int, dict] = {}
    for cfg in sweep_cfgs:
        run = train(cfg)
        kl = run["final"]["kl_per_dim"]
        sweep.append({
            "d_z": cfg.d_z, "dec_width": cfg.dec_width,
            "dead": int((kl < COLLAPSE_NATS).sum()),
            "weak": int((kl < 0.1).sum()),
            "total_kl": float(kl.sum()), "recon": float(run["final"]["recon"]),
            "min_kl": float(kl.min()), "max_kl": float(kl.max()),
        })
        if cfg.dec_width == 64:
            sweep_runs[cfg.d_z] = run
    # The session's own d_z = 16 run is the sweep's first point.
    kl16 = run16["final"]["kl_per_dim"]
    sweep.insert(0, {
        "d_z": 16, "dec_width": 64,
        "dead": int((kl16 < COLLAPSE_NATS).sum()),
        "weak": int((kl16 < 0.1).sum()),
        "total_kl": float(kl16.sum()), "recon": float(run16["final"]["recon"]),
        "min_kl": float(kl16.min()), "max_kl": float(kl16.max()),
    })

    mu2, y2 = encode_split(run2["model"], run2["val_loader"])
    mu16, y16 = encode_split(run16["model"], run16["val_loader"])
    norms = norm_profiles(mu2, y2, mu16, y16)

    fig_tug_of_war(run2, run16)
    fig_latent_scatter(mu2, y2)
    fig_manifold_sheet(run2["model"])
    interp = fig_interpolation(run2["model"], mu2, y2, norms)
    fig_samples(run2, run16)
    collapse = fig_perdim_kl(run16, sweep_runs[64])
    fig_capacity_sweep(sweep)
    fig_reduction_trap(run2, trap)

    mid = len(norms["ts"]) // 2
    payload = {
        "_provenance": {
            "script": "scripts/python/fig_u0_l3_vae.py",
            "device": str(DEVICE),
            "seed": SEED,
            "torch": torch.__version__,
            "val_split_size": N_VAL,
            "note": "test split untouched — read once, by the U0.L4 harness",
        },
        "configs": {c.run_name: asdict(c) for c in
                    [cfg2, cfg16, cfg_trap, *sweep_cfgs]},
        "gradient_check": {"correct_loss": grad_ok, "corrupted_loss": grad_bad},
        "init_latent_sanity": {
            "z_mean": run2["init_z_mean"], "z_std": run2["init_z_std"],
        },
        "final_val": {
            tag: {k: float(run["final"][k]) for k in ("loss", "recon", "kl")}
            for tag, run in (("dz2", run2), ("dz16", run16), ("trap", trap))
        },
        "history": {tag: run["history"] for tag, run in
                    (("dz2", run2), ("dz16", run16), ("trap", trap))},
        "capacity_sweep": sweep,
        "collapse": {
            **collapse,
            "threshold_nats": COLLAPSE_NATS,
            "min_kl_dz16": float(kl16.min()),
            "active_dims_dz64": int(len(collapse["kl_sorted_dz64"])
                                    - collapse["dead_dz64"]),
        },
        "interpolation": {
            **interp,
            "dz2_lerp_min_norm": float(min(norms["dz2_lerp"])),
            "dz2_slerp_min_norm": float(min(norms["dz2_slerp"])),
            "dz16_lerp_min_norm": float(min(norms["dz16_lerp"])),
            "dz16_slerp_min_norm": float(min(norms["dz16_slerp"])),
            "dz16_lerp_midpoint_norm": float(norms["dz16_lerp"][mid]),
            "dz16_slerp_midpoint_norm": float(norms["dz16_slerp"][mid]),
            "sqrt_dz16": math.sqrt(16),
        },
        "reduction_trap": {
            "correct_final_kl": float(run2["final"]["kl"]),
            "trap_final_kl": float(trap["final"]["kl"]),
            "correct_final_recon_per_pixel": float(run2["final"]["recon"]) / 784.0,
            "trap_final_recon_per_pixel": float(trap["final"]["recon"]),
            "implied_beta": 784.0,
        },
    }
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(payload, indent=2))
    print(f"wrote {JSON_OUT}")

    # --- The honesty contract, enforced ---------------------------------
    print("\n=== claims the notes may make, checked ===")
    print(f"d_z=16 dead dims: {collapse['dead_dz16']} of 16 "
          f"(min KL {kl16.min():.3f} nats)")
    if collapse["dead_dz16"] == 0:
        print("   -> the notes MUST NOT say the session's own run collapses.")
    print(f"d_z=64 dead dims: {collapse['dead_dz64']} of 64, "
          f"weak(<0.1): {collapse['weak_dz64']}")
    assert collapse["dead_dz64"] > 0 or collapse["weak_dz64"] > 0, (
        "no collapse anywhere in the sweep — the block has no phenomenon to show"
    )
    print("total KL by d_z: " + ", ".join(
        f"{s['d_z']}:{s['total_kl']:.1f}" for s in sweep if s["dec_width"] == 64))
    print(f"trap KL {trap['final']['kl']:.4f} vs correct KL {run2['final']['kl']:.4f}")
    print(f"d_z=16 lerp midpoint norm {norms['dz16_lerp'][mid]:.3f} vs "
          f"slerp {norms['dz16_slerp'][mid]:.3f} (sqrt(d_z) = 4.000)")
    print(f"d_z=2  lerp min norm {min(norms['dz2_lerp']):.3f} vs "
          f"slerp {min(norms['dz2_slerp']):.3f} (sqrt(d_z) = 1.414)")


if __name__ == "__main__":
    main()
