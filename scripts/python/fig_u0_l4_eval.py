"""fig_u0_l4_eval.py — U0.L4 session figures and measured numbers.

Runs the session's two artifacts for real — `common/unet.py` (`unet-skeleton`)
and `common/eval.py` (`eval-harness`) — and writes three svg+pdf pairs into
course/Figures/ plus every number the notes and the deck quote, into
_outputs/u0l4_eval.json.

  1. unet-conditioning-rituals — Part C. The three "is my conditioning alive?"
     checks, measured. The left panel is the one worth the space: at
     INITIALIZATION the network is exactly the zero function, so the
     t-sensitivity ritual cannot pass; it passes only after the network has
     moved off zero. Rituals 2 and 1 are therefore run at different moments,
     and the figure is why.

  2. fid-harness-calibration — Part E. The bias-vs-n curve of the split test,
     with its fitted log-log slope, and the calibration ladder: split test,
     the two VAEs, and the garbage test on one log axis, so the dynamic range
     of the instrument is visible before anyone trusts a number from it.

  3. fid-vae-payoff — Part E. Sample grids of the U0.L3 VAEs at d_z = 2 and
     d_z = 16, each with its measured FID. The eye test of U0.L3 becomes a
     number, measured at EQUAL sample count, which is the only way the
     comparison means anything (caveat (i)).

HONESTY CONTRACT (the U0.T3 rule, applied here). Every quantitative claim the
session makes is computed and asserted below, never drawn by hand. The claims
and what would falsify each:

  * "the untrained U-Net is exactly the zero field" — asserted to be zero to
    machine precision. If a later edit removes a zero-init, this fails.
  * "the t-sensitivity ritual cannot fail at initialization" — asserted: the
    spread of ||f(x,t)|| over t is exactly 0 before the wake-up and strictly
    positive after it. This is a claim ABOUT A DEFECT in the L3 plan, so it is
    the one most worth checking.
  * "the split test is small but nonzero" — asserted strictly positive and
    below the garbage test by a large factor. The figure must NOT show it as
    zero, because it is not.
  * "the bias falls like 1/n" — the log-log slope is fitted and asserted near
    -1. If it comes out elsewhere, the script fails rather than drawing a law
    that does not hold.
  * "the d_z = 16 VAE scores better than the d_z = 2 VAE" — NOT assumed. The
    two are measured at equal n and the ordering is recorded; the assert below
    pins whatever was measured, so a later edit that changes it fails loudly
    instead of quietly contradicting the notes.

Run:  conda run -n torch-gpu python scripts/python/fig_u0_l4_eval.py
"""

from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")  # headless: identical output in CI, SSH, containers
matplotlib.rcParams["svg.hashsalt"] = "u0l4"  # deterministic SVG element ids
# Type 42 (TrueType) outlines, not matplotlib's Type 3 default: Type 3 carries
# no hinting, so small axis labels render mushy in every serious PDF viewer.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402
from torchvision import datasets, transforms  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "python"))

from common.eval import (  # noqa: E402
    MNIST_FID_SAMPLES,
    MNISTFeatures,
    RefStats,
    fid,
    panel,
    ref_stats,
)
from common.unet import (  # noqa: E402
    UNet,
    num_groups,
    timestep_embedding,
)

# --- UFRJ palette (keep in sync with the three theme surfaces in
# --- course/_theme/ — INV-1; verify with ./scripts/check-palette-sync.sh) ---
UFRJ_BLUE = "#003366"
UFRJ_BLUE_LIGHT = "#00447f"
UFRJ_ACCENT = "#d97a1f"
UFRJ_ACCENT_DARK = "#a3540c"
UFRJ_GRAY = "#6b7280"
UFRJ_GRAY_LIGHT = "#9aa2aa"
INK = "#26262a"

SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = ROOT / "course" / "Figures"
DATA_DIR = ROOT / "scripts" / "python" / "data"
JSON_OUT = ROOT / "scripts" / "python" / "_outputs" / "u0l4_eval.json"

# The session's reference U-Net configuration, quoted in the deck's contract
# slide and in the notebook's checkpoint cell.
UNET_CFG = dict(in_ch=1, base_ch=64, ch_mults=(1, 2, 2), attn_resolutions=(7,))

# Sample counts for the bias curve. The three the L3 names (500, 2k, 10k) plus
# enough points on either side to FIT a slope rather than eyeball one.
BIAS_COUNTS = (250, 500, 1000, 2000, 5000, 10000)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def new_figure(ncols: int, figsize: tuple[float, float]) -> tuple[Figure, list[Axes]]:
    fig, axes = plt.subplots(1, ncols, figsize=figsize)
    return cast(Figure, fig), [cast(Axes, ax) for ax in np.atleast_1d(axes)]


def apply_house_style(ax: Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig: Figure, name: str) -> None:
    """Write the svg+pdf pair the extension-less Quarto include expects."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", transparent=True, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# 1. The U-Net, and the rituals of Part C
# ===========================================================================


def wake_up(model: UNet, steps: int = 30, seed: int = SEED) -> None:
    """Move the network off the zero function. **This is not training.**

    The recipe card's zero-initialization makes the untrained network the zero
    field, exactly and for every t. That is the property ritual 2 asserts —
    and it is also what makes ritual 1 (t sensitivity) VACUOUS at
    initialization: a network whose output is identically zero cannot
    distinguish two times, so an assert that the outputs differ must fail.

    The two rituals therefore run at different moments, and something has to
    move the parameters in between. Fitting a few steps of random targets does
    that without teaching the network anything: there is no dataset, no
    generative objective, and the result is discarded. Training the U-Net is
    U3.L2's job by design, and this is not it.

    (Same shape as the U0.L3 correction: the gradient-check ritual there also
    could not fail at initialization, and for the same reason.)
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(steps):
        x = torch.randn(8, 1, 28, 28, generator=g).to(DEVICE)
        t = torch.rand(8, generator=g).to(DEVICE)
        target = torch.randn(8, 1, 28, 28, generator=g).to(DEVICE)
        loss = ((model(x, t) - target) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()


@torch.no_grad()
def film_modulation(model: UNet, ts: np.ndarray, batch: int,
                    label: int | None = None) -> np.ndarray:
    """The (gamma, beta) a chosen ResBlock receives, for each t in `ts`.

    Rituals 2 and 3 are WIRING tests (L3-U0-L4 section 2C): they ask whether
    the conditioning signal reaches the inside of the network, and they read
    the modulation directly rather than the output. That is what lets them
    run AT INITIALIZATION, where the output is the zero field by law. The
    FiLM projection is small-but-nonzero at init exactly so this is possible.
    """
    with torch.no_grad():
        rows = []
        for t in ts:
            emb = timestep_embedding(
                torch.full((batch,), float(t), device=DEVICE), model.base_ch
            )
            if label is not None:
                assert model.label_emb is not None
                emb = emb + model.label_emb(
                    torch.full((batch,), label, dtype=torch.long, device=DEVICE)
                )
            rows.append(model.mid_block1.film(model.emb_mlp(emb))[0].cpu().numpy())
    return np.stack(rows)


def post_update_sensitivity(model: UNet, x: torch.Tensor, *, y_on: bool,
                            seed: int = SEED) -> dict[str, float]:
    """Ritual 4, the canonical recipe (L3-U0-L4 section 2C, v0.3).

    Deep-copy the model; draw ONE fixed random target g ~ N(0, I) of the
    output shape under a fixed seed; take ONE plain SGD step (no momentum,
    lr = 1e-2) on 1/2 ||f(x,t,y) - g||^2 at a fixed non-degenerate (x,t,y);
    then compare the copy's outputs at fixed x for two values of t, and for
    two labels y when the class pathway is on. The float64 norm of each
    difference must exceed 1e-6. The copy is discarded, so the zero-init
    contract of the model that will actually be trained is untouched.

    One recipe, fixed, because U3.L2 inherits this ritual and students'
    numbers have to be comparable.
    """
    probe = copy.deepcopy(model)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    g = torch.randn(x.shape, generator=gen).to(DEVICE)
    b = x.shape[0]
    t_step = torch.full((b,), 0.5, device=DEVICE)
    y_step = (torch.zeros(b, dtype=torch.long, device=DEVICE) if y_on else None)

    opt = torch.optim.SGD(probe.parameters(), lr=1e-2, momentum=0.0)
    loss = 0.5 * (probe(x, t_step, y_step) - g).pow(2).sum()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    probe.eval()
    with torch.no_grad():
        f0 = probe(x, torch.zeros(b, device=DEVICE), y_step).double()
        f1 = probe(x, torch.ones(b, device=DEVICE), y_step).double()
        out = {"t_difference_norm": float((f1 - f0).norm()), "tol": 1e-6}
        if y_on:
            ta = torch.full((b,), 0.5, device=DEVICE)
            fa = probe(x, ta, torch.zeros(b, dtype=torch.long, device=DEVICE)).double()
            fb = probe(x, ta, torch.ones(b, dtype=torch.long, device=DEVICE)).double()
            out["y_difference_norm"] = float((fb - fa).norm())
    del probe
    return out


def output_norm_vs_t(model: UNet, x: torch.Tensor, ts: np.ndarray) -> np.ndarray:
    """||f(x, t)|| for one fixed x, over a sweep of t. Ritual 1's raw material."""
    out = []
    for t_val in ts:
        t = torch.full((x.shape[0],), float(t_val), device=x.device)
        out.append(float(model(x, t).norm()))
    return np.asarray(out)


@torch.no_grad()
def label_distance_matrix(
    model: UNet, x: torch.Tensor, t_val: float, n_classes: int
) -> np.ndarray:
    """Pairwise ||f(x,t,y_i) - f(x,t,y_j)||. Ritual 3's raw material."""
    outs = []
    for label in range(n_classes):
        t = torch.full((x.shape[0],), t_val, device=x.device)
        y = torch.full((x.shape[0],), label, dtype=torch.long, device=x.device)
        outs.append(model(x, t, y).flatten())
    stack = torch.stack(outs)
    return torch.cdist(stack[None], stack[None])[0].cpu().numpy()


def gradient_check(model: UNet, seed: int = SEED) -> dict[str, float]:
    """Finite differences against autograd on one FiLM projection weight.

    Run in float64 and on a scalar reduction of the output, exactly as in
    U0.L1 and U0.L3. Also self-tested: the same check is re-run against a
    deliberately corrupted gradient, and must FAIL there. A check that has
    never failed is not known to work.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    # float64 throughout, as in U0.L1 and U0.L3. In float32 a central
    # difference over a sum of ~1500 activations loses every digit it needs:
    # the first version of this check reported a relative error of 155, which
    # is the arithmetic failing, not the gradient.
    model = model.double()
    x = torch.randn(2, 1, 28, 28, generator=g).double().to(DEVICE)
    t = torch.rand(2, generator=g).double().to(DEVICE)

    # The first ResBlock's FiLM projection: the parameter through which the
    # conditioning signal reaches the computation at all.
    film = None
    for module in model.modules():
        if hasattr(module, "film"):
            film = module.film
            break
    assert film is not None, "no FiLM projection found — did TODO 3 wire it?"
    weight = film.weight

    def scalar() -> torch.Tensor:
        return model(x, t).sum()

    model.zero_grad(set_to_none=True)
    scalar().backward()
    analytic = weight.grad.detach().clone()
    assert float(analytic.abs().max()) > 0, (
        "the analytic gradient into the FiLM projection is exactly zero, so "
        "this check would measure nothing. Run wake_up first — at "
        "initialization the zero-init makes it vacuous (U0.L3 precedent)."
    )

    eps = 1e-6
    flat = int(analytic.abs().argmax())
    idx = tuple(int(i) for i in np.unravel_index(flat, tuple(analytic.shape)))
    with torch.no_grad():
        original = weight[idx].item()
        weight[idx] = original + eps
        plus = float(scalar())
        weight[idx] = original - eps
        minus = float(scalar())
        weight[idx] = original
    numeric = (plus - minus) / (2 * eps)
    a = float(analytic[idx])
    rel = abs(numeric - a) / (abs(a) + 1e-12)

    # Self-test: the same comparison against a gradient we KNOW is wrong.
    corrupt_rel = abs(numeric - 2.0 * a) / (abs(2.0 * a) + 1e-12)
    assert corrupt_rel > 0.1, (
        "the gradient check does not fail on a deliberately doubled gradient, "
        "so it is not measuring what it claims to measure"
    )
    model.float()
    return {"analytic": a, "numeric": numeric, "rel_error": rel,
            "self_test_rel_error": corrupt_rel}


def fig_conditioning_rituals(res: dict) -> None:
    fig, axes = new_figure(3, (11.4, 3.2))

    ax = axes[0]
    ts = np.asarray(res["ts"])
    ax.plot(ts, res["norm_init"], color=UFRJ_GRAY, lw=2.0,
            label="at initialization")
    ax.plot(ts, res["norm_awake"], color=UFRJ_BLUE, lw=2.0,
            label="after 30 steps")
    apply_house_style(ax)
    ax.set_xlabel("$t$", fontsize=10)
    ax.set_ylabel(r"$\|f(x,t)\|$", fontsize=10)
    ax.set_title("Ritual 1 — sensitivity to $t$", fontsize=10, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="center right")
    ax.annotate(
        "identically zero:\nthe ritual is vacuous here",
        xy=(0.5, 0.0), xytext=(0.06, 0.20), textcoords="axes fraction",
        fontsize=8, color=UFRJ_ACCENT_DARK,
        arrowprops=dict(arrowstyle="->", color=UFRJ_ACCENT_DARK, lw=1.0),
    )

    ax = axes[1]
    # A bar of height zero on a log axis is invisible, and an empty panel
    # communicates nothing. The message is a CONTRAST: most layers are
    # initialized normally, and four named ones are exactly zero. So plot
    # both, on a linear axis, and label the zeros.
    ordinary = [("input conv", res["ordinary_init"]["conv_in"]),
                ("res conv 1", res["ordinary_init"]["resblock_conv1"]),
                ("attn qkv", res["ordinary_init"]["attn_qkv"]),
                ("FiLM proj", res["ordinary_init"]["film_proj"]),
                ("res conv 2", res["ordinary_init"]["resblock_conv2"])]
    zeroed = [("output conv", res["zero_init"]["conv_out"]),
              ("attn proj", res["zero_init"]["attn_proj"])]
    names = [n for n, _ in ordinary + zeroed]
    vals = [v for _, v in ordinary + zeroed]
    colors = [UFRJ_GRAY_LIGHT] * len(ordinary) + [UFRJ_BLUE] * len(zeroed)
    ax.bar(names, vals, color=colors, width=0.62)
    apply_house_style(ax)
    ax.set_ylabel("parameter norm at init", fontsize=10)
    ax.set_title("Ritual 2 — the zero-init law", fontsize=10, color=INK)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=38, ha="right")
    top = max(vals) * 1.35
    ax.set_ylim(0, top)
    for i in range(len(ordinary), len(names)):
        ax.text(i, top * 0.035, "0", ha="center", fontsize=9,
                color=UFRJ_BLUE, fontweight="bold")
    ax.text(0.5, 0.90, r"four layers are exactly $0$"
            "\n" r"$\Rightarrow\ \|f(x,t)\| = 0$ for every $t$",
            transform=ax.transAxes, ha="center", fontsize=8.5,
            color=UFRJ_ACCENT_DARK)

    ax = axes[2]
    d = np.asarray(res["label_distances"])
    im = ax.imshow(d, cmap="Blues", interpolation="nearest")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.tick_params(colors=INK, labelsize=7)
    ax.set_xlabel("$y'$", fontsize=10)
    ax.set_ylabel("$y$", fontsize=10)
    ax.set_title("Ritual 3 — sensitivity to $y$", fontsize=10, color=INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7)
    cb.set_label(r"$\|f(x,t,y) - f(x,t,y')\|$", fontsize=8)

    fig.tight_layout()
    save(fig, "unet-conditioning-rituals")


# ===========================================================================
# 2. The harness, calibrated before it is trusted
# ===========================================================================


def mnist_tensors(train: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """The whole split as one (N,1,28,28) float tensor in [0,1], plus labels."""
    ds = datasets.MNIST(
        DATA_DIR, train=train, download=False, transform=transforms.ToTensor()
    )
    loader = DataLoader(ds, batch_size=2048, shuffle=False)
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    return torch.cat(xs), torch.cat(ys)


def stats_from_tensor(
    x: torch.Tensor, extractor, dataset: str, cache_dir: Path
) -> RefStats:
    """`ref_stats` over a plain tensor, batched through a trivial loader."""
    loader = DataLoader(TensorDataset(x), batch_size=1024)
    return ref_stats(loader, extractor, dataset=dataset, cache_dir=cache_dir)


def fig_calibration(res: dict) -> None:
    fig, axes = new_figure(2, (10.2, 3.5))

    ax = axes[0]
    ns = np.asarray(res["bias_counts"], dtype=float)
    vals = np.asarray(res["bias_values"], dtype=float)
    ax.loglog(ns, vals, "o-", color=UFRJ_BLUE, lw=2.0, ms=5,
              label="measured split test")
    slope, intercept = res["bias_slope"], res["bias_intercept"]
    ax.loglog(ns, np.exp(intercept) * ns**slope, "--", color=UFRJ_ACCENT,
              lw=1.6, label=rf"fit: slope ${slope:.2f}$")
    ax.axhline(0.0, color=UFRJ_GRAY_LIGHT, lw=0.8)
    apply_house_style(ax)
    ax.grid(which="both", color="#e6e6e6", linewidth=0.5)
    ax.set_xlabel("sample count $n$", fontsize=10)
    ax.set_ylabel("FID of a split against itself", fontsize=10)
    ax.set_title("The bias is real, and it falls like $1/n$", fontsize=10,
                 color=INK)
    ax.legend(fontsize=8, frameon=False)
    ax.axvline(MNIST_FID_SAMPLES, color=UFRJ_GRAY, lw=0.9, ls=":")
    ax.text(MNIST_FID_SAMPLES * 0.94, vals.max() * 0.55, "course standard\n$n=10^4$",
            fontsize=7.5, color=UFRJ_GRAY, ha="right")

    ax = axes[1]
    labels = ["split test\n(true FID $=0$)", "VAE $d_z=16$", "VAE $d_z=2$",
              "garbage test\n(noise images)"]
    values = [res["split_test"], res["fid_vae_16"], res["fid_vae_2"],
              res["garbage_test"]]
    colors = [UFRJ_GRAY, UFRJ_BLUE, UFRJ_BLUE_LIGHT, UFRJ_ACCENT]
    bars = ax.barh(labels, values, color=colors, height=0.6)
    ax.set_xscale("log")
    apply_house_style(ax)
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"FID, all at $n = {res['equal_n']:,}$", fontsize=10)
    ax.set_title("The calibration ladder", fontsize=10, color=INK)
    ax.tick_params(axis="y", labelsize=8)
    for bar, v in zip(bars, values):
        # One notation for the whole column: 7.03e+03 beside 0.603 makes the
        # reader convert in their head to compare, which is the one thing this
        # panel exists to make easy.
        label = f"{v:,.0f}" if v >= 100 else f"{v:.3g}"
        ax.text(v * 1.15, bar.get_y() + bar.get_height() / 2, label,
                va="center", fontsize=8, color=INK)
    ax.set_xlim(min(values) * 0.4, max(values) * 6)

    fig.tight_layout()
    save(fig, "fid-harness-calibration")


def fig_vae_payoff(grids: dict, res: dict) -> None:
    fig, axes = new_figure(2, (8.0, 4.3))
    for ax, key, dz in zip(axes, ("g2", "g16"), (2, 16)):
        ax.imshow(grids[key], cmap="gray", vmin=0.0, vmax=1.0,
                  interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_color(UFRJ_GRAY_LIGHT)
        value = res[f"fid_vae_{dz}"]
        ax.set_title(f"$d_z = {dz}$   ·   FID $= {value:.2f}$", fontsize=11,
                     color=INK)
    fig.suptitle(
        f"Both measured with {res['extractor']} at n = {res['equal_n']:,} — "
        "the only way the two numbers may be compared",
        fontsize=8.5, color=UFRJ_GRAY, y=0.04,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fid-vae-payoff")


# ===========================================================================
# main
# ===========================================================================


def redraw_only() -> None:
    """Redraw the three figures from the recorded numbers.

    Exists because a figure is edited far more often than it is measured, and
    a layout fix should not cost two VAE trainings. The numbers come from the
    committed JSON, so a redraw can never invent one.
    """
    payload = json.loads(JSON_OUT.read_text())
    fig_conditioning_rituals(payload["rituals"])
    fig_calibration(payload["calibration"])
    print("redrew unet-conditioning-rituals and fid-harness-calibration "
          "(fid-vae-payoff needs the sample grids — rerun in full for it)")


def main() -> None:
    if "--figures-only" in sys.argv:
        redraw_only()
        return
    set_seed(SEED)
    payload: dict = {"seed": SEED, "device": DEVICE}

    # -- 1. The U-Net: structure, then the three rituals -------------------
    model = UNet(**UNET_CFG).to(DEVICE)  # type: ignore[arg-type]
    model_cond = UNet(**UNET_CFG, num_classes=10).to(DEVICE)  # type: ignore[arg-type]
    payload["unet"] = {
        "describe": model.describe(),
        "params": sum(p.numel() for p in model.parameters()),
        "params_conditional": sum(p.numel() for p in model_cond.parameters()),
        "d_emb": model.d_emb,
        # The checkpoint question: GroupNorm's group count is not always 32.
        "groups": {str(c): num_groups(c) for c in (64, 128, 1)},
    }

    x_fixed = torch.randn(4, 1, 28, 28, generator=torch.Generator().manual_seed(1))
    x_fixed = x_fixed.to(DEVICE)
    ts = np.linspace(0.0, 1.0, 41)

    model.eval()
    model_cond.eval()

    # --- Ritual 1: the zero-field check, AT INITIALIZATION ----------------
    # Exact equality against 0.0 is deliberate and is NOT the float-
    # comparison mistake: `conv_out` is zero-initialized, so the output is
    # identically zero bit for bit, and "approximately zero" would hide a
    # lost zero-init.
    norm_init = output_norm_vs_t(model, x_fixed, ts)
    assert float(np.max(np.abs(norm_init))) == 0.0, (
        "the untrained U-Net is not the zero function — a zero-init was lost"
    )

    # --- Ritual 2: t WIRING sensitivity, also at initialization -----------
    # Read the modulation, not the output. This is why the FiLM projection
    # is small-nonzero rather than zero (common/unet.py, L3-U0-L4 2A): the
    # field is still exactly zero, but gamma(t), beta(t) already move with t,
    # so the wiring test is not vacuous at step 0.
    mod_t = film_modulation(model, ts, x_fixed.shape[0])
    t_wiring_spread = float(np.max(np.ptp(mod_t, axis=0)))
    assert t_wiring_spread > 0.0, (
        "the FiLM modulation does not move with t — the conditioning path "
        "is wired to nothing (TODO 3)"
    )

    # --- Ritual 3: y WIRING sensitivity, at initialization ----------------
    mod_y = np.stack([
        film_modulation(model_cond, np.array([0.5]), x_fixed.shape[0], label=k)[0]
        for k in range(10)
    ])
    y_wiring_spread = float(np.max(np.ptp(mod_y, axis=0)))
    assert y_wiring_spread > 0.0, (
        "the FiLM modulation does not move with the label — the class "
        "pathway is dead"
    )

    # --- Ritual 4: post-update OUTPUT sensitivity, canonical recipe -------
    r4 = post_update_sensitivity(model, x_fixed, y_on=False)
    r4_cond = post_update_sensitivity(model_cond, x_fixed, y_on=True)
    r4["y_difference_norm"] = r4_cond["y_difference_norm"]
    assert r4["t_difference_norm"] > r4["tol"], (
        "after one SGD step the outputs at t=0 and t=1 are still identical"
    )
    assert r4["y_difference_norm"] > r4["tol"], (
        "after one SGD step two labels still give identical outputs"
    )
    # The copies are discarded, so the models below are still the zero field.
    assert float(np.max(np.abs(output_norm_vs_t(model, x_fixed, ts)))) == 0.0, (
        "ritual 4 leaked back into the model it was supposed to leave alone"
    )

    # The left panel of the figure contrasts the zero field with a network
    # that has been moved off it. Thirty steps against random targets, result
    # discarded: the U-Net is NOT trained today (first training is U3.L2).
    wake_up(model, steps=30)
    norm_awake = output_norm_vs_t(model, x_fixed, ts)
    spread = float(np.ptp(norm_awake) / (np.mean(norm_awake) + 1e-12))
    assert np.ptp(norm_awake) > 0.0, (
        "after the wake-up the network still cannot tell two times apart"
    )

    ordinary_init = {
        "conv_in": float(model_cond.conv_in.weight.detach().norm()),
        "resblock_conv1": float(model_cond.mid_block1.conv1.weight.detach().norm()),
        "attn_qkv": float(model_cond.mid_attn.qkv.weight.detach().norm()),
    }
    assert min(ordinary_init.values()) > 0.0, (
        "an ordinary layer is zero at init — the contrast panel would lie"
    )
    # ONE exact-zero interior law (L3-U0-L4 section 2A), and it carries the
    # whole zero-field property on its own.
    zero_init = {
        "conv_out": float(model_cond.conv_out.weight.detach().norm()),
        "attn_proj": float(model_cond.mid_attn.proj.weight.detach().norm()),
    }
    assert max(zero_init.values()) == 0.0, "a zero-init was lost"
    # These two are deliberately SMALL rather than zero; see SMALL_INIT_STD.
    small_init = {
        "film_proj": float(model_cond.mid_block1.film.weight.detach().norm()),
        "resblock_conv2": float(model_cond.mid_block1.conv2.weight.detach().norm()),
    }
    assert min(small_init.values()) > 0.0, (
        "a small-init layer is exactly zero — rituals 2, 3 and 4 would be "
        "vacuous (L3-U0-L4 section 2A)"
    )
    ordinary_init.update(small_init)

    wake_up(model_cond, steps=30, seed=SEED + 1)
    label_d = label_distance_matrix(model_cond, x_fixed, 0.5, 10)
    off_diag = label_d[~np.eye(10, dtype=bool)]
    assert off_diag.min() > 0.0, (
        "two labels produce identical output — the class pathway is dead"
    )
    grad = gradient_check(model_cond)
    assert grad["rel_error"] < 1e-2, (
        f"gradient check failed: relative error {grad['rel_error']:.3g}"
    )

    rituals = {
        "ts": ts.tolist(),
        "norm_init": norm_init.tolist(),
        "norm_awake": norm_awake.tolist(),
        "t_spread_relative": spread,
        "t_wiring_spread_at_init": t_wiring_spread,
        "y_wiring_spread_at_init": y_wiring_spread,
        "ritual4_post_update": r4,
        "zero_init": zero_init,
        "ordinary_init": ordinary_init,
        "label_distances": label_d.tolist(),
        "label_distance_min": float(off_diag.min()),
        "label_distance_max": float(off_diag.max()),
        "gradient_check": grad,
    }
    payload["rituals"] = rituals
    fig_conditioning_rituals(rituals)
    print(f"[rituals] wiring at init: t {t_wiring_spread:.3e}, "
          f"y {y_wiring_spread:.3e} | ritual 4 post-update: "
          f"t {r4['t_difference_norm']:.3e}, y {r4['y_difference_norm']:.3e} "
          f"(tol {r4['tol']:.0e}) | t-spread after wake-up: {spread:.3f} "
          f"relative | grad rel err {grad['rel_error']:.2e}")

    # -- 2. Train the U0.L3 VAEs, for the Part E payoff --------------------
    from fig_u0_l3_vae import Config, train  # noqa: E402  (heavy import)

    cache = ROOT / "scripts" / "python" / "_outputs" / "refstats-u0l4"
    cache.mkdir(parents=True, exist_ok=True)
    extractor = MNISTFeatures(device=DEVICE)
    payload["extractor"] = {
        "name": extractor.name,
        "preprocessing": extractor.preprocessing,
        "sha256": extractor.sha256,
        "val_acc": extractor.val_acc,
    }

    x_train, _ = mnist_tensors(train=True)
    x_test, _ = mnist_tensors(train=False)
    n_eq = MNIST_FID_SAMPLES

    # The reference side: MNIST test, the whole split, computed once.
    ref = stats_from_tensor(x_test, extractor, "mnist-test", cache)

    # Ritual 1 of Part E — the split test. Two halves of the TRAIN split, so
    # the two sides are genuinely independent draws from one distribution and
    # the true FID is exactly zero. Everything measured is bias.
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(x_train.shape[0], generator=g)
    half_a, half_b = x_train[perm[:30000]], x_train[perm[30000:]]
    ref_a = stats_from_tensor(half_a, extractor, "mnist-train-halfA", cache)

    bias_values = []
    for n in BIAS_COUNTS:
        bias_values.append(float(fid(half_b[:n], ref_a, extractor)))
    assert all(
        bias_values[i] > bias_values[i + 1] for i in range(len(bias_values) - 1)
    ), f"the split-test bias is not monotone in n: {bias_values}"
    assert min(bias_values) > 0.0, "a split test came out at exactly zero"
    slope, intercept = np.polyfit(
        np.log(np.asarray(BIAS_COUNTS, dtype=float)), np.log(bias_values), 1
    )
    assert -1.4 < slope < -0.6, (
        f"the bias does not follow a 1/n law here (slope {slope:.2f}); do not "
        "draw the law"
    )
    split_test = float(fid(half_b[:n_eq], ref_a, extractor))
    print(f"[calibration] split test at n={n_eq}: {split_test:.4f}  "
          f"slope {slope:.2f}")

    # Ritual 3 — the garbage test. Uniform noise images, same protocol.
    noise = torch.rand(n_eq, 1, 28, 28, generator=g)
    garbage = float(fid(noise, ref, extractor))
    assert garbage > 100 * split_test, (
        f"the garbage test ({garbage:.2f}) is not far above the split test "
        f"({split_test:.4f}) — the instrument does not discriminate"
    )
    print(f"[calibration] garbage test: {garbage:.2f}")

    # -- 3. The payoff: the U0.L3 VAEs, measured -------------------------
    fids: dict[int, float] = {}
    grids: dict[str, np.ndarray] = {}
    for d_z in (2, 16):
        run = train(Config(run_name=f"u0l4-fid-dz{d_z}", d_z=d_z))
        vae = run["model"].eval()
        with torch.no_grad():
            samples = []
            for start in range(0, n_eq, 1000):
                k = min(1000, n_eq - start)
                z = torch.randn(k, d_z, device=DEVICE)
                # Bernoulli MEANS, not draws: the reference side is grayscale
                # MNIST in [0,1], and the extractor's preprocessing is that.
                # Thresholding the samples would compare a binary set against
                # a grayscale one — a preprocessing mismatch, caveat (i).
                samples.append(torch.sigmoid(vae.decode(z)).cpu())
            gen = torch.cat(samples)
        value = fid(gen, ref, extractor)
        fids[d_z] = float(value)
        print(f"[payoff] VAE d_z={d_z}: {value!r}")

        # The panel comes from the harness itself (TODO 7), driven by a
        # FIXED noise bank so the two grids differ because the model differs.
        eps_bank = torch.randn(64, d_z, generator=torch.Generator().manual_seed(7))
        fig_p = panel(
            lambda e: torch.sigmoid(vae.decode(e.to(DEVICE))).cpu(),
            eps_bank,
            nrow=8,
        )
        grids[f"g{d_z}"] = np.asarray(fig_p.axes[0].images[0].get_array())
        plt.close(fig_p)

    # The ordering was NOT assumed. It is measured above, recorded here, and
    # pinned so that a later edit which changes it fails instead of quietly
    # contradicting the notes. Measured 2026-08-28: d_z=16 scores lower.
    assert fids[16] < fids[2], (
        f"the measured ordering changed: d_z=16 -> {fids[16]:.3f}, "
        f"d_z=2 -> {fids[2]:.3f}. The notes state the opposite — fix the "
        "notes, do not weaken this assert."
    )

    calib = {
        "bias_counts": list(BIAS_COUNTS),
        "bias_values": bias_values,
        "bias_slope": float(slope),
        "bias_intercept": float(intercept),
        "split_test": split_test,
        "garbage_test": garbage,
        "fid_vae_2": fids[2],
        "fid_vae_16": fids[16],
        "equal_n": n_eq,
        "extractor": extractor.name,
        "ref_n": ref.n_samples,
    }
    payload["calibration"] = calib
    fig_calibration(calib)
    fig_vae_payoff(grids, calib)

    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {JSON_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
