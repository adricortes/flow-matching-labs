"""fig_u0_l5_solvers.py — U0.L5 session figures (four svg+pdf pairs).

Produces, into course/Figures/ (extension-less includes: html/revealjs pick
.svg, beamer picks .pdf):

  1. l5-tolerance-sweep    — Block C. The adaptive solver's tolerance swept
                             from 1e-1 to 1e-8 on the harmonic oscillator.
                             Panel (a) NFE against the error actually
                             achieved; panel (b) the achieved error against
                             the error requested, with the diagonal. The two
                             failure smells of U0.T5 are located on the
                             students' own plot: the loose end is cheap and
                             wrong, the tight end is expensive and no better
                             than the arithmetic allows.

  2. l5-per-sample-grads   — Block D, the argument for `vmap`. Per-sample
                             gradient norms of a tiny MLP loss over one
                             batch, computed with vmap(grad(.)), against the
                             norm of the batch-mean gradient. The spread is
                             the point: the number the optimizer sees is not
                             a typical sample's number, and getting the
                             per-sample quantity in eager PyTorch is awkward
                             enough that people simply do not look.

  3. l5-jit-timing         — Block E, the honesty rule made into a picture.
                             Wall-clock of the same JAX function eager, on
                             its first jitted call (compile included), and in
                             steady state. Compile time is reported
                             SEPARATELY, never amortised into the speedup.

  4. l5-diffrax-crosscheck — Block E, the humility figure. Their adaptive
                             Heun(2)/Euler(1) against diffrax's Tsit5 on the
                             same problem over the same tolerance sweep, NFE
                             against achieved error. A real solver is not a
                             little better. It is orders of magnitude better,
                             and the gap is the order of the method.

WHAT IS ASSERTED, AND WHY. Every quantitative claim the notes or the deck
make about these pictures is computed here and asserted here. The claims:
NFE rises monotonically as the tolerance tightens; the achieved error tracks
the request; the norm of the mean per-sample gradient is far below the mean
of the norms; the first jitted call costs much more than the steady state;
and Tsit5 reaches a matched accuracy at a small fraction of the NFE. If any
of them stops being true, this script fails rather than drawing a falsehood.

  (no figure)  adaptive_vs_fixed — Block B, checkpoint 2. The adaptive Heun
                             against a fixed-step Heun at MATCHED achieved
                             error, on two problems: the harmonic oscillator
                             (uniform difficulty) and the same oscillator
                             with a burst of fast rotation in a short window
                             (nonuniform difficulty). L3-U0-L5 v0.2 §2B asks
                             for this as a measurement on a problem chosen to
                             have nonuniform difficulty, and forbids asserting
                             a universal theorem: on the uniform problem the
                             saving is ~1x, on the burst problem it is large.
                             Both directions are asserted.

THE MEASURED VALUES ARE WRITTEN TO scripts/python/_outputs/u0l5_solvers.json
so the notes quote numbers that were produced, not remembered.

Run:  conda run -n torch-gpu python scripts/python/fig_u0_l5_solvers.py
      ... --only adaptive_vs_fixed,diffrax     # re-measure a subset; the
                                               # other keys of the JSON are
                                               # kept (the jit timing is the
                                               # one non-reproducible number,
                                               # and a partial re-run must
                                               # not silently move it)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: identical output in CI, SSH, containers
matplotlib.rcParams["svg.hashsalt"] = "u0l5"  # deterministic SVG element ids
# Type 42 (TrueType) glyph outlines, NOT matplotlib's Type 3 default: Type 3
# fonts carry no hinting, so Acrobat / Preview / browser PDF viewers render
# small axis labels mushy in the Beamer deck.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.odesolve import odeint  # noqa: E402

# --- UFRJ palette (keep in sync with the theme surfaces in course/_theme/ —
# --- verify with ./scripts/check-palette-sync.sh) ---
UFRJ_BLUE = "#003366"
UFRJ_BLUE_LIGHT = "#00447f"
UFRJ_ACCENT = "#d97a1f"
UFRJ_ACCENT_DARK = "#a3540c"
UFRJ_GRAY = "#6b7280"
UFRJ_GRAY_LIGHT = "#9aa2aa"
INK = "#26262a"

SEED = 0
OUT_DIR = ROOT / "course" / "Figures"
JSON_OUT = ROOT / "scripts" / "python" / "_outputs" / "u0l5_solvers.json"

RESULTS: dict[str, object] = {}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def apply_house_style(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(UFRJ_GRAY)
    ax.spines["bottom"].set_color(UFRJ_GRAY)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis=grid_axis, color="#DDDDDD", linewidth=0.6)
    ax.set_axisbelow(True)


def save_pair(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf"):
        path = OUT_DIR / f"{name}.{ext}"
        fig.savefig(path, transparent=True, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


# =========================================================================
# The test problem: the harmonic oscillator of U0.T5. Its exact orbit is the
# unit circle, so the global error at t = 2*pi is the distance back to the
# starting point — no reference solver, no reference error.
# =========================================================================

T_END = 2.0 * np.pi
X0 = torch.tensor([[1.0, 0.0]], dtype=torch.float64)


def harmonic(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """dx/dt = v, dv/dt = -x. Time first, batched state (B, 2)."""
    return torch.stack([x[:, 1], -x[:, 0]], dim=1)


def global_error(x1: torch.Tensor) -> float:
    return float((x1 - X0).norm())


# The burst problem: the same rotation, with the angular speed raised from 1
# to 1 + A in a window of width ~W around T_STAR. Outside the window it IS
# the harmonic oscillator; inside it the field is forty times stiffer to
# resolve. The exact solution is a rotation by the integrated angular speed,
# which is an erf, so the global error is computable without a reference
# solver -- the same reason the harmonic oscillator was chosen.
T_STAR, W, A = float(np.pi), 0.1, 40.0


def omega(t: torch.Tensor) -> torch.Tensor:
    return 1.0 + A * torch.exp(-((t - T_STAR) / W) ** 2)


def burst(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """dx/dt = omega(t) v, dv/dt = -omega(t) x. Time first, batched (B, 2)."""
    return omega(t) * torch.stack([x[:, 1], -x[:, 0]], dim=1)


def burst_exact(t: float) -> torch.Tensor:
    tt = torch.as_tensor(t, dtype=torch.float64)
    theta = tt + A * W * np.sqrt(np.pi) / 2 * (
        torch.erf((tt - T_STAR) / W) + torch.erf(torch.tensor(T_STAR / W))
    )
    return torch.stack([torch.cos(theta), -torch.sin(theta)]).reshape(1, 2)


def adaptive_vs_fixed() -> None:
    """Checkpoint 2 of the lab, measured here so the notes can quote it.

    For each problem and each rtol: run the adaptive Heun, read the error it
    achieved, then find the smallest fixed-step Heun (geometric search, factor
    1.25) whose error is at most that, and compare NFE. The saving is a
    property of the PROBLEM, not of adaptivity: on the uniform problem the
    fixed step is already the right step everywhere, so there is nothing to
    save; on the burst problem the fixed step must resolve the burst over the
    whole interval, and the adaptive one pays for it only inside the window.
    """
    problems = {
        "harmonic": (harmonic, X0),
        "burst": (burst, burst_exact(T_END)),
    }
    rows = []
    for name, (f, exact) in problems.items():
        for rtol in (1e-4, 1e-6):
            ad = odeint(f, X0, 0.0, T_END, method="adaptive_heun",
                        rtol=rtol, atol=rtol * 1e-2)
            err_ad = float((ad - exact).norm())
            n = 16
            while True:
                fx = odeint(f, X0, 0.0, T_END, method="heun", n_steps=n)
                err_fx = float((fx - exact).norm())
                if err_fx <= err_ad:
                    break
                n = int(n * 1.25) + 1
            rows.append({
                "problem": name, "rtol": rtol,
                "adaptive_nfe": int(ad.nfe), "adaptive_accepted": ad.accepted,
                "adaptive_rejected": ad.rejected, "adaptive_error": err_ad,
                "fixed_steps": n, "fixed_nfe": int(fx.nfe), "fixed_error": err_fx,
                "nfe_ratio_fixed_over_adaptive": fx.nfe / ad.nfe,
            })
            print(f"  {name:8s} rtol={rtol:.0e}: adaptive NFE {ad.nfe:6d} "
                  f"(err {err_ad:.2e}) vs fixed NFE {fx.nfe:6d} "
                  f"(err {err_fx:.2e}) -> {fx.nfe / ad.nfe:.2f}x")
    # Both directions are claims the notes make, so both are asserted.
    for r in rows:
        if r["problem"] == "burst":
            assert r["nfe_ratio_fixed_over_adaptive"] > 3.0, (
                f"burst saving only {r['nfe_ratio_fixed_over_adaptive']:.2f}x; "
                "the notes say adaptivity pays where difficulty is nonuniform"
            )
        else:
            assert r["nfe_ratio_fixed_over_adaptive"] < 1.5, (
                f"harmonic saving {r['nfe_ratio_fixed_over_adaptive']:.2f}x; "
                "the notes say there is nothing to save on a uniform problem"
            )
    RESULTS["adaptive_vs_fixed"] = {
        "burst": {"t_star": T_STAR, "w": W, "a": A},
        "rows": rows,
    }


# =========================================================================
# 1. The equal-budget table and the tolerance sweep (Block C)
# =========================================================================


def equal_budget_table() -> None:
    """The NFE accounting the notes print: method x n_steps -> (error, NFE).

    Recorded, not plotted. The decision question in the notes ("which row
    would you pick at an error budget of 1e-4?") is answered off this table.
    """
    rows = []
    for method, per_step in (("euler", 1), ("heun", 2), ("rk4", 4)):
        for n in (25, 50, 100, 200, 400, 800, 1600):
            x1 = odeint(harmonic, X0, 0.0, T_END, method=method, n_steps=n)
            rows.append(
                {
                    "method": method,
                    "n_steps": n,
                    "h": T_END / n,
                    "error": global_error(x1),
                    "nfe": int(x1.nfe),
                }
            )
            assert int(x1.nfe) == n * per_step, "NFE bookkeeping is wrong"

    # The equal-budget comparison quoted in the notes: NFE = 4000 for all
    # three, reached with 4000 / 2000 / 1000 steps.
    equal = {}
    for method, n in (("euler", 4000), ("heun", 2000), ("rk4", 1000)):
        x1 = odeint(harmonic, X0, 0.0, T_END, method=method, n_steps=n)
        assert int(x1.nfe) == 4000
        equal[method] = {"n_steps": n, "error": global_error(x1), "nfe": 4000}

    assert equal["euler"]["error"] > equal["heun"]["error"] > equal["rk4"]["error"], (
        "the equal-budget ordering is the claim; it did not hold"
    )

    RESULTS["accounting"] = rows
    RESULTS["equal_budget_nfe4000"] = equal
    print("  equal budget, NFE = 4000: " + ", ".join(
        f"{m} {v['error']:.2e}" for m, v in equal.items()
    ))


def fig_tolerance_sweep() -> None:
    rtols = np.logspace(-1, -8, 8)
    nfes, errors, accepted, rejected = [], [], [], []

    for rtol in rtols:
        x1 = odeint(
            harmonic, X0, 0.0, T_END,
            method="adaptive_heun", rtol=float(rtol), atol=float(rtol) * 1e-2,
        )
        nfes.append(int(x1.nfe))
        errors.append(global_error(x1))
        accepted.append(int(x1.accepted))
        rejected.append(int(x1.rejected))
        print(f"  rtol {rtol:.0e}: NFE {x1.nfe:>7d}  error {global_error(x1):.3e}")

    nfes_a = np.array(nfes, dtype=float)
    errors_a = np.array(errors, dtype=float)

    # --- the claims this figure makes ---
    assert np.all(np.diff(nfes_a) > 0), (
        "NFE must rise monotonically as the tolerance tightens; it did not"
    )
    assert np.all(np.diff(errors_a) < 0), (
        "the achieved error must fall as the tolerance tightens; it did not"
    )
    # Cost against accuracy: a second-order method's NFE scales like
    # error^(-1/2). Fit it and record the slope actually measured.
    slope = float(np.polyfit(np.log(errors_a), np.log(nfes_a), 1)[0])
    assert -0.75 < slope < -0.35, (
        f"NFE-against-error slope {slope:.3f} is not near the -1/2 an "
        "order-2 method predicts"
    )
    cost_ratio = nfes_a[-1] / nfes_a[0]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))

    ax = axes[0]
    ax.loglog(errors_a, nfes_a, "o-", color=UFRJ_BLUE, lw=1.8, ms=5,
              label="adaptive Heun(2)/Euler(1)")
    ax.annotate(
        f"rtol $=10^{{-1}}$\n{nfes[0]} NFE",
        xy=(errors_a[0], nfes_a[0]), xytext=(-6, 26), textcoords="offset points",
        fontsize=8.5, color=UFRJ_ACCENT_DARK, ha="right",
        arrowprops=dict(arrowstyle="-", color=UFRJ_ACCENT, lw=0.9),
    )
    ax.annotate(
        f"rtol $=10^{{-8}}$\n{nfes[-1]:,} NFE",
        xy=(errors_a[-1], nfes_a[-1]), xytext=(30, -14), textcoords="offset points",
        fontsize=8.5, color=UFRJ_ACCENT_DARK,
        arrowprops=dict(arrowstyle="-", color=UFRJ_ACCENT, lw=0.9),
    )
    ax.set_xlabel("global error achieved at $t = 2\\pi$", fontsize=10, color=INK)
    ax.set_ylabel("NFE", fontsize=10, color=INK)
    ax.set_title(f"(a)  cost against accuracy — fitted slope {slope:.2f}",
                 fontsize=10.5, color=INK, loc="left")
    apply_house_style(ax)

    ax = axes[1]
    lo = min(errors_a.min(), rtols.min()) * 0.3
    hi = max(errors_a.max(), rtols.max()) * 3
    ax.loglog([lo, hi], [lo, hi], "--", color=UFRJ_GRAY_LIGHT, lw=1.1,
              label="achieved = requested")
    ax.loglog(rtols, errors_a, "o-", color=UFRJ_ACCENT, lw=1.8, ms=5,
              label="measured")
    ax.set_xlim(hi, lo)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("rtol requested  (tighter $\\rightarrow$)", fontsize=10, color=INK)
    ax.set_ylabel("global error achieved", fontsize=10, color=INK)
    ax.set_title("(b)  the request and the outcome", fontsize=10.5, color=INK,
                 loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    apply_house_style(ax)

    fig.tight_layout()
    save_pair(fig, "l5-tolerance-sweep")

    RESULTS["tolerance_sweep"] = {
        "rtol": [float(r) for r in rtols],
        "atol": [float(r) * 1e-2 for r in rtols],
        "nfe": nfes,
        "error": errors,
        "accepted": accepted,
        "rejected": rejected,
        "nfe_vs_error_slope": slope,
        "cost_ratio_loose_to_tight": float(cost_ratio),
    }


# =========================================================================
# 2. Per-sample gradients with vmap (Block D)
# =========================================================================


def fig_per_sample_grads() -> None:
    import jax
    import jax.numpy as jnp

    key = jax.random.key(SEED)
    key, k_w1, k_b1, k_w2, k_b2, k_x, k_y = jax.random.split(key, 7)

    d_in, d_hidden, batch = 8, 32, 512
    params = {
        "w1": jax.random.normal(k_w1, (d_in, d_hidden)) / jnp.sqrt(d_in),
        "b1": jnp.zeros(d_hidden),
        "w2": jax.random.normal(k_w2, (d_hidden, 1)) / jnp.sqrt(d_hidden),
        "b2": jnp.zeros(1),
    }
    xs = jax.random.normal(k_x, (batch, d_in))
    ys = jax.random.normal(k_y, (batch, 1))

    def loss_one(params, x, y):
        """Squared error on ONE sample. No batch dimension in the body."""
        h = jnp.tanh(x @ params["w1"] + params["b1"])
        pred = h @ params["w2"] + params["b2"]
        return jnp.sum((pred - y) ** 2)

    grad_one = jax.grad(loss_one)
    per_sample = jax.vmap(grad_one, in_axes=(None, 0, 0))(params, xs, ys)

    def tree_norm(tree) -> float:
        leaves = jax.tree.leaves(tree)
        return float(jnp.sqrt(sum(jnp.sum(leaf**2) for leaf in leaves)))

    norms = np.array([
        float(jnp.sqrt(sum(
            jnp.sum(per_sample[k][i] ** 2) for k in per_sample
        )))
        for i in range(batch)
    ])

    mean_grad = jax.tree.map(lambda g: g.mean(axis=0), per_sample)
    norm_of_mean = tree_norm(mean_grad)
    mean_of_norms = float(norms.mean())

    # --- claim 1: vmap(grad) agrees with a loop over grad, to machine
    # precision. This is the correctness check, and it is the reason to
    # believe the rest of the figure.
    loop_g0 = grad_one(params, xs[0], ys[0])
    max_dev = max(
        float(jnp.max(jnp.abs(loop_g0[k] - per_sample[k][0]))) for k in loop_g0
    )
    assert max_dev < 1e-5, f"vmap(grad) disagreed with grad by {max_dev:.2e}"

    # --- claim 2: the batch gradient is far smaller than a typical sample's.
    ratio = mean_of_norms / norm_of_mean
    assert ratio > 2.0, (
        f"the spread claim needs the ratio to be substantial; got {ratio:.2f}. "
        "Do NOT draw this figure as an argument about cancellation if the "
        "cancellation is not there."
    )

    fig, ax = plt.subplots(figsize=(7.6, 3.7))
    ax.hist(norms, bins=48, color=UFRJ_BLUE_LIGHT, edgecolor="white", lw=0.4,
            alpha=0.92, label="per-sample $\\|\\nabla_\\theta \\ell_i\\|$")
    ax.axvline(mean_of_norms, color=UFRJ_GRAY, lw=1.6, ls=":",
               label=f"mean of the norms = {mean_of_norms:.2f}")
    ax.axvline(norm_of_mean, color=UFRJ_ACCENT, lw=2.0,
               label=f"norm of the mean = {norm_of_mean:.2f}")
    ax.set_xlabel("gradient norm", fontsize=10, color=INK)
    ax.set_ylabel("samples", fontsize=10, color=INK)
    ax.set_title(
        f"one batch of {batch}; the optimizer sees the orange line, "
        f"{ratio:.0f}$\\times$ smaller",
        fontsize=10.5, color=INK, loc="left",
    )
    ax.legend(frameon=False, fontsize=8.8)
    apply_house_style(ax, grid_axis="y")
    fig.tight_layout()
    save_pair(fig, "l5-per-sample-grads")

    RESULTS["per_sample_grads"] = {
        "batch": batch,
        "norm_of_mean": norm_of_mean,
        "mean_of_norms": mean_of_norms,
        "ratio": ratio,
        "max_norm": float(norms.max()),
        "min_norm": float(norms.min()),
        "vmap_vs_loop_max_deviation": max_dev,
    }
    print(f"  per-sample grads: mean of norms {mean_of_norms:.3f}, "
          f"norm of mean {norm_of_mean:.3f}, ratio {ratio:.1f}")


# =========================================================================
# 3. Honest jit timing (Block E)
# =========================================================================


def fig_jit_timing() -> None:
    import jax
    import jax.numpy as jnp

    key = jax.random.key(SEED + 1)
    n = 4096
    a = jax.random.normal(key, (n, 64))

    def work(a):
        """Enough arithmetic that the timing is not all dispatch overhead."""
        h = a
        for _ in range(12):
            h = jnp.tanh(h @ (h.T @ h) / n)
        return jnp.sum(h**2)

    def time_it(fn, arg, repeats: int) -> float:
        """Median of `repeats` calls. The MEDIAN, not the mean: one scheduler
        hiccup on a shared laptop moves a mean and does not move a median.
        Wall-clock timing is still machine- and run-dependent, and the notes
        say so — what reproduces is the shape, not the millisecond."""
        samples = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            jax.block_until_ready(fn(arg))
            samples.append(time.perf_counter() - t0)
        return float(np.median(samples))

    jax.block_until_ready(work(a))  # warm the eager path too — fair is fair
    eager = time_it(work, a, repeats=15)

    jitted = jax.jit(work)
    t0 = time.perf_counter()
    jax.block_until_ready(jitted(a))
    first_call = time.perf_counter() - t0

    steady = time_it(jitted, a, repeats=41)
    compile_only = first_call - steady

    # --- the claims ---
    assert first_call > 3 * steady, (
        f"the first call ({first_call*1e3:.1f} ms) is not dominated by "
        f"compilation against the steady state ({steady*1e3:.1f} ms); the "
        "figure's whole point would be false"
    )
    speedup = eager / steady
    breakeven = compile_only / (eager - steady) if eager > steady else float("nan")
    # Honesty: the notes may claim a speedup ONLY if there is one. If jit is
    # not faster here, the figure must still be drawn and the text must say
    # so — this assert exists to force that conversation, not to hide it.
    assert speedup > 1.0, (
        f"jit was NOT faster than eager here (speedup {speedup:.2f}). Do not "
        "ship a speedup claim; rewrite the block around the real result."
    )

    fig, ax = plt.subplots(figsize=(7.8, 3.7))
    labels = ["eager\n(no jit)", "first jitted call\n(compile + run)",
              "steady state\n(jitted)"]
    values = [eager * 1e3, first_call * 1e3, steady * 1e3]

    # LINEAR axis, on purpose. A log axis on a bar chart makes the bar
    # LENGTHS stop being proportional to the times, and the whole reason
    # this figure exists is that the proportion is the argument.
    ax.bar([0], [eager * 1e3], color=UFRJ_GRAY, width=0.58)
    ax.bar([1], [steady * 1e3], color=UFRJ_BLUE, width=0.58)
    ax.bar([1], [compile_only * 1e3], bottom=[steady * 1e3],
           color=UFRJ_ACCENT, width=0.58)
    ax.bar([2], [steady * 1e3], color=UFRJ_BLUE, width=0.58)

    ax.text(1, steady * 1e3 / 2, f"run  {steady*1e3:.0f} ms", ha="center",
            va="center", fontsize=8.6, color="white")
    ax.text(1, steady * 1e3 + compile_only * 1e3 / 2,
            f"compile  {compile_only*1e3:.0f} ms", ha="center", va="center",
            fontsize=8.6, color="white")

    for i, v in enumerate(values):
        ax.text(i, v + 0.03 * max(values), f"{v:.1f} ms", ha="center",
                va="bottom", fontsize=9.5, color=INK)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(labels, fontsize=9.5, color=INK)
    ax.set_ylim(0, max(values) * 1.16)
    ax.set_ylabel("wall clock (ms)", fontsize=10, color=INK)
    ax.set_title(
        f"steady state is {speedup:.1f}$\\times$ eager; the compile costs "
        f"{compile_only*1e3:.0f} ms and is reported separately, never amortised",
        fontsize=10.5, color=INK, loc="left",
    )
    apply_house_style(ax, grid_axis="y")
    fig.tight_layout()
    save_pair(fig, "l5-jit-timing")

    RESULTS["jit_timing"] = {
        "eager_s": eager,
        "first_call_s": first_call,
        "steady_s": steady,
        "compile_only_s": compile_only,
        "speedup": speedup,
        "breakeven_calls": breakeven,
        "jax_version": jax.__version__,
        "device": str(jax.devices()[0]),
    }
    print(f"  jit: eager {eager*1e3:.1f} ms, first {first_call*1e3:.1f} ms, "
          f"steady {steady*1e3:.1f} ms, speedup {speedup:.1f}x, "
          f"breakeven {breakeven:.0f} calls")


# =========================================================================
# 4. Their adaptive Heun against diffrax's Tsit5 (Block E)
# =========================================================================


def fig_diffrax_crosscheck() -> None:
    import diffrax
    import jax
    import jax.numpy as jnp

    # NFE is INSTRUMENTED, not inferred. diffrax exposes num_steps /
    # num_accepted_steps / num_rejected_steps and no evaluation count, and
    # "6 per step" is a fact about Tsit5's tableau (7 stages, first-same-as-
    # last) that the reader has to take on trust. A host callback inside the
    # field fires once per real evaluation, under jit and inside the solver's
    # loop. Measured: it is 6 * num_steps + 1 -- the estimate was one short,
    # the initial evaluation. (L3-U0-L5 v0.2 §2E.)
    calls = {"n": 0}

    def _count() -> None:
        calls["n"] += 1

    def harmonic_jax(t, y, args):
        """The same field, in diffrax's signature: f(t, y, args)."""
        jax.debug.callback(_count)
        return jnp.array([y[1], -y[0]])

    term = diffrax.ODETerm(harmonic_jax)
    y0 = jnp.array([1.0, 0.0], dtype=jnp.float64)
    exact = np.array([1.0, 0.0])

    def solve(rtol: float, atol: float):
        calls["n"] = 0
        sol = diffrax.diffeqsolve(
            term, diffrax.Tsit5(), t0=0.0, t1=float(T_END), dt0=0.01, y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
            max_steps=1_000_000,
        )
        stats = {k: int(sol.stats[k]) for k in
                 ("num_steps", "num_accepted_steps", "num_rejected_steps")}
        return np.asarray(sol.ys[-1], dtype=float), calls["n"], stats

    rtols = np.logspace(-3, -10, 8)
    dfx_nfe, dfx_err, dfx_stats = [], [], []
    for rtol in rtols:
        y1, nfe, stats = solve(float(rtol), float(rtol) * 1e-2)
        assert nfe == 6 * stats["num_steps"] + 1, (nfe, stats)
        dfx_nfe.append(nfe)
        dfx_stats.append(stats)
        dfx_err.append(float(np.linalg.norm(y1 - exact)))

    ours_rtols = np.logspace(-3, -8, 6)
    our_nfe, our_err = [], []
    for rtol in ours_rtols:
        x1 = odeint(harmonic, X0, 0.0, T_END, method="adaptive_heun",
                    rtol=float(rtol), atol=float(rtol) * 1e-2)
        our_nfe.append(int(x1.nfe))
        our_err.append(global_error(x1))

    # --- claim 1: BOTH solvers against the ANALYTIC solution, then against
    # each other. The exact orbit returns to (1, 0) at t = 2 pi, so each
    # solver has a true error, and agreement between two wrong solvers is
    # not what is being checked. (L3-U0-L5 v0.2 §2E.)
    dfx_tight, tight_nfe, tight_stats = solve(1e-12, 1e-14)
    ours_rk4 = odeint(harmonic, X0, 0.0, T_END, method="rk4", n_steps=4000)
    tsit5_tight_error = float(np.linalg.norm(dfx_tight - exact))
    rk4_4000_error = global_error(ours_rk4)
    budget = 1e-9
    assert tsit5_tight_error < budget, tsit5_tight_error
    assert rk4_4000_error < budget, rk4_4000_error
    agreement = float(np.linalg.norm(dfx_tight - ours_rk4.numpy()[0]))
    assert agreement < budget, (
        f"diffrax and our RK4 disagree by {agreement:.2e}; one of them is "
        "wrong and the cross-check is the whole point of this block"
    )

    # --- claim 2: at a matched accuracy, Tsit5 is far cheaper. Compare at
    # the tightest error our method reached.
    target = our_err[-1]
    dfx_at_target = float(np.interp(
        np.log(target), np.log(dfx_err[::-1]), np.log(dfx_nfe[::-1])
    ))
    dfx_at_target = float(np.exp(dfx_at_target))
    saving = our_nfe[-1] / dfx_at_target
    assert saving > 5.0, (
        f"Tsit5 saved only {saving:.1f}x at matched error; the humility "
        "framing of this block assumes a large gap"
    )

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.loglog(our_err, our_nfe, "o-", color=UFRJ_ACCENT, lw=1.9, ms=5.5,
              label="ours — adaptive Heun(2)/Euler(1)")
    ax.loglog(dfx_err, dfx_nfe, "s-", color=UFRJ_BLUE, lw=1.9, ms=5.0,
              label="diffrax — Tsit5 (order 5)")
    ax.axhline(dfx_at_target, color=UFRJ_GRAY_LIGHT, lw=1.0, ls="--")
    ax.axvline(target, color=UFRJ_GRAY_LIGHT, lw=1.0, ls="--")
    ax.annotate(
        f"same accuracy,\n{saving:.0f}$\\times$ fewer evaluations",
        xy=(target, dfx_at_target), xytext=(46, 16), textcoords="offset points",
        fontsize=9, color=UFRJ_BLUE,
        arrowprops=dict(arrowstyle="->", color=UFRJ_BLUE, lw=1.0),
    )
    ax.set_xlabel("global error achieved at $t = 2\\pi$", fontsize=10, color=INK)
    ax.set_ylabel("NFE", fontsize=10, color=INK)
    ax.set_title("the same problem, the same tolerances, a better tableau",
                 fontsize=10.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    apply_house_style(ax)
    fig.tight_layout()
    save_pair(fig, "l5-diffrax-crosscheck")

    RESULTS["diffrax_crosscheck"] = {
        "diffrax_version": diffrax.__version__,
        "jax_version": jax.__version__,
        "diffrax_rtol": [float(r) for r in rtols],
        "diffrax_nfe": dfx_nfe,
        "diffrax_nfe_how": "instrumented: jax.debug.callback in the field, "
                           "one call per evaluation; equals 6*num_steps + 1",
        "diffrax_stats": dfx_stats,
        "diffrax_error": dfx_err,
        "ours_rtol": [float(r) for r in ours_rtols],
        "ours_nfe": our_nfe,
        "ours_error": our_err,
        "tsit5_tight_error_vs_exact": tsit5_tight_error,
        "tsit5_tight_nfe": tight_nfe,
        "tsit5_tight_stats": tight_stats,
        "rk4_4000_error_vs_exact": rk4_4000_error,
        "agreement_rk4_vs_tsit5": agreement,
        "matched_error": target,
        "tsit5_nfe_at_matched_error": dfx_at_target,
        "nfe_saving_factor": saving,
    }
    print(f"  diffrax: Tsit5 tight {tsit5_tight_error:.2e} vs exact, RK4 "
          f"{rk4_4000_error:.2e} vs exact, agreement {agreement:.2e}; "
          f"Tsit5 saves {saving:.0f}x at error {target:.2e}")


STAGES = {
    "accounting": equal_budget_table,
    "adaptive_vs_fixed": adaptive_vs_fixed,
    "tolerance": fig_tolerance_sweep,
    "per_sample_grads": fig_per_sample_grads,
    "jit": fig_jit_timing,
    "diffrax": fig_diffrax_crosscheck,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", default="",
                        help="comma-separated subset of: " + ", ".join(STAGES))
    args = parser.parse_args()
    wanted = [k for k in args.only.split(",") if k] or list(STAGES)
    unknown = set(wanted) - set(STAGES)
    assert not unknown, f"unknown stage(s) {sorted(unknown)}"

    import jax
    jax.config.update("jax_enable_x64", True)

    # A partial run keeps every key it did not re-measure.
    if args.only and JSON_OUT.exists():
        RESULTS.update(json.loads(JSON_OUT.read_text()))

    set_seed(SEED)
    for i, key in enumerate(wanted, 1):
        print(f"{i}/{len(wanted)} {key}")
        STAGES[key]()

    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(RESULTS, indent=2) + "\n")
    print(f"wrote {JSON_OUT}")


if __name__ == "__main__":
    main()
