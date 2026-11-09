"""fig_u3_l1_payoff.py - U3.L1 reference runs and numbers: CFM from scratch, and the payoff pair.

Trains the reference conditional flow matching (CFM) field on the two frozen generators of
common/toy2d.py under the FROZEN BUDGET of U2.L2 (configs/u2l2-budget.yaml, inherited field by
field), on the SAME HOST as the U2.L2 reference CNF run, so that the two runs form the reference
pair `ref-<machine> + ref-U2L2-B1 + U2L2-B1`. Every number the notes, the deck and the notebook
quote is computed here, asserted where the text makes a claim about it, and written to the JSON.

THE CODE THE NOTEBOOK COPIES. `sample_batch`, `cfm_loss`, `make_field`, `train_cfm`,
`move_head`, `directional_check`, `sample_cfm`, `cfm_tolerance_rows`, `write_cfm_evidence`,
`torchcfm_batch_fn` and `jax_cfm` are the solutions of the notebook's TODOs and its given cells.
make_lab_u3l1.py copies them with `inspect.getsource`. The U2.L2 objects (`Field`,
`AugmentedDynamics`, `evaluate_nll`, `warmup_cosine`, `make_log_prob`) are imported from the U2.L2
reference script: the CFM field IS the U2.L2 field class, by design.

Produces:
  labs/u3l1/reference/cfm_evidence/<dataset>/   the CFM half of the reference pair
  labs/u3l1/reference/mlflow_export/<dataset>/  sanitized MLflow export of the reference runs
  common/weights/u3l1-<dataset>.{pt,json}       reference CFM checkpoints
  course/Figures/u3-l1-*.{svg,pdf}              the figures of the notes and the deck
  _outputs/u3l1_payoff.json                     the numbers

STAGES AND --only. ritual, train, sample, tolerance, crosscheck, jax, longer, probes, evidence, figures.
`--only a,b` re-runs those stages and MERGES their keys into the existing JSON (MEMORY
2026-09-14). `train` runs ALONE: it records wall-clock time.

HONESTY CHECKS (asserted, then recorded):
  1. the loss at initialization (zero head, u = 0) matches its population value
     E||x_1||^2 + d inside a 4-standard-error band, over many minibatches;
  2. the directional derivative of the loss by autograd agrees with a central difference in
     float64, on a field whose head was moved by a fixed, stated amount; on the UNMOVED field
     (zero head) the same check reads exactly 0 in the first layer (the vacuous case it guards);
  3. a loss that draws fresh randomness between the +h and -h evaluations FAILS the check;
  4. the CFM evidence records 0 solver NFE in training and passes the painplots validator, and
     the reference CNF and CFM directories form a valid pair under common/payoff.py;
  5. with the SAME (t, x_0, x_1), TorchCFM's ConditionalFlowMatcher gives the same x_t and target
     as the scratch code, to float32 rounding, and the two training runs agree step by step;
  6. no monotonic law is asserted for any NFE or wall-clock curve: they are measured.

BOUNDARY (L3-U3-L1 section 5). No proof or gradient argument relating L_CFM to L_FM; no score,
no noise schedule other than CondOT, no stochastic sampler, no guidance.

Run:  conda run -n torch-gpu python scripts/python/fig_u3_l1_payoff.py --only train
      conda run -n torch-gpu python scripts/python/fig_u3_l1_payoff.py
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import shutil
import statistics
import sys
import time
from pathlib import Path

import fig_u2_l2_pain as u2   # sets Agg, Type 42 fonts and one torch thread (the U2.L2 conventions)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import yaml  # noqa: E402
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common import odesolve, painplots, payoff  # noqa: E402
from common.toy2d import EXTENT, density_heatmap, in_checkerboard_support, warped_grid  # noqa: E402

Field, AugmentedDynamics, evaluate_nll = u2.Field, u2.AugmentedDynamics, u2.evaluate_nll
warmup_cosine, make_log_prob, GENERATORS = u2.warmup_cosine, u2.make_log_prob, u2.GENERATORS
BUDGET = u2.BUDGET


def set_all_seeds(seed: int) -> None:
    """Seed every stream the run touches: random, NumPy and torch keep independent state (INV-9)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
UFRJ_BLUE, UFRJ_ACCENT, UFRJ_GRAY, INK, FS = u2.UFRJ_BLUE, u2.UFRJ_ACCENT, u2.UFRJ_GRAY, u2.INK, u2.FS

JSON_OUT = Path(__file__).resolve().parent / "_outputs" / "u3l1_payoff.json"
WEIGHTS_DIR = ROOT / "common" / "weights"
REF_DIR = ROOT / "labs" / "u3l1" / "reference"
CNF_REF = ROOT / "labs" / "u2l2" / "reference" / "pain_evidence"
SCRATCH = ROOT / ".scratch" / "u3l1"
MLFLOW_URI = f"sqlite:///{SCRATCH / 'mlflow.db'}"
SESSION = "U3.L1"
HEAD_SCALE = 0.1          # the fixed, stated amount by which the ritual moves the zero head
N_SAMPLES = 4096
LONG_ITERS = 30000      # the second receipt: ten times the budget's iterations


# ---------------------------------------------------------------------------
# The solutions of the notebook's TODOs and its given cells. Copied by
# inspect.getsource: edit them here, never in the notebook.
# ---------------------------------------------------------------------------
def sample_batch(train_x: torch.Tensor, batch: int, gen: torch.Generator):
    """One CFM minibatch on the CondOT path (TODO 1.1). Returns (t, x0, x1, xt, target).

    t ~ Unif[0, 1] has shape (B,): the code-level FM arrow, t = 0 noise and t = 1 data.
    x1 ~ q is a row of the training set, x0 ~ N(0, I_2), both (B, 2). Every draw comes from `gen`.
    """
    idx = torch.randint(0, train_x.shape[0], (batch,), generator=gen)
    x1 = train_x[idx]                                                # (B, 2), data
    x0 = torch.randn(x1.shape, generator=gen, dtype=x1.dtype)        # (B, 2), noise
    t = torch.rand(batch, generator=gen, dtype=x1.dtype)             # (B,)
    xt = t[:, None] * x1 + (1 - t[:, None]) * x0                     # (B,1) against (B,2)
    target = x1 - x0                                                 # the conditional velocity
    return t, x0, x1, xt, target


def cfm_loss(field: nn.Module, t: torch.Tensor, xt: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The minibatch estimate of L_CFM (TODO 1.2): squared norm over coordinates, mean over the batch."""
    u = field(t, xt)                                                 # u_t^theta(x_t), (B, 2)
    return ((u - target) ** 2).sum(dim=1).mean()


def make_field(cfg: dict) -> nn.Module:
    """The field of the inherited budget (TODO 1.3): the U2.L2 `Field` class with the budget's width and depth, tanh, zero head."""
    m = cfg["model"]
    assert m["activation"] == "tanh", "the budget fixes tanh, and Field implements tanh"
    return Field(m["hidden"], m["depth"], m["zero_init_head"])


def train_cfm(cfg: dict, dataset: str, *, seed: int | None = None, iters: int | None = None,
              batch_fn=None, verbose: bool = True):
    """Train under the inherited budget, instrumented from iteration 0. Returns (field, log).

    The same seed, minibatch generator, optimizer and schedule as the U2.L2 `train_run`. The step
    time covers the minibatch draw, the loss, the backward pass and the update. The two NFE columns
    record 0: no ODE solver runs inside this loss. The one direct call of the field per step is
    `field_evals_train`, logged elsewhere under its own name.
    """
    seed = cfg["seed"] if seed is None else seed
    iters = cfg["train"]["iters"] if iters is None else iters
    batch_fn = sample_batch if batch_fn is None else batch_fn
    set_all_seeds(seed)
    train_x = GENERATORS[dataset](cfg["data"]["n_train"], cfg["data"]["seed_train"])
    field = make_field(cfg)
    opt = torch.optim.AdamW(field.parameters(), lr=cfg["opt"]["lr"], betas=tuple(cfg["opt"]["betas"]),
                            weight_decay=cfg["opt"]["weight_decay"])
    warm = max(1, round(cfg["sched"]["warmup_frac"] * iters))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: warmup_cosine(i, iters, warm))
    batch_gen = torch.Generator().manual_seed(seed)
    log = painplots.TrainLog()
    for it in range(1, iters + 1):
        t0 = time.perf_counter()
        t, x0, x1, xt, target = batch_fn(train_x, cfg["train"]["batch"], batch_gen)
        loss = cfm_loss(field, t, xt, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        step_s = time.perf_counter() - t0
        if not math.isfinite(loss.item()):
            raise FloatingPointError(f"loss is {loss.item()} at iteration {it}")
        log.add(step_s, 0, 0, loss.item())               # solver NFE: 0 and 0, there is no solve
        if verbose and it % 500 == 0:
            print(f"  [{dataset}] it {it:5d}  loss {statistics.fmean(log.loss[-100:]):.4f}  "
                  f"s/it {statistics.fmean(log.step_s[-100:]):.5f}", flush=True)
    return field, log


@torch.no_grad()
def move_head(field: nn.Module, scale: float = HEAD_SCALE, seed: int = 0) -> nn.Module:
    """A copy of `field` whose zero head is replaced by scale * N(0, 1) entries (a fixed, stated move).

    With the head at zero every parameter upstream of it has gradient exactly 0, and a gradient
    check there compares 0 with 0 (a course lesson: a zero head makes every upstream check vacuous). Moving the
    head first makes it a test.
    """
    g = copy.deepcopy(field)
    gen = torch.Generator().manual_seed(seed)
    head = g.net[-1]
    head.weight.copy_(scale * torch.randn(head.weight.shape, generator=gen))
    head.bias.copy_(scale * torch.randn(head.bias.shape, generator=gen))
    return g


def directional_check(field: nn.Module, batch: tuple, *, h: float = 1e-6, seed: int = 0) -> dict:
    """Autograd against a central difference, in float64, along one fixed direction of net[0].weight.

    `batch` is ONE frozen minibatch (t, x0, x1, xt, target): nothing is drawn between the +h and
    the -h evaluations. The direction is a unit vector of fixed seed in the FIRST layer, the
    tensor farthest from the head.
    """
    f = copy.deepcopy(field).double()
    t, _, _, xt, target = (b.double() for b in batch)
    p = f.net[0].weight
    v = torch.randn(p.shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    v = v / v.norm()
    grad = torch.autograd.grad(cfm_loss(f, t, xt, target), p)[0]
    autodiff = float((grad * v).sum())
    with torch.no_grad():
        p.add_(h * v)
        plus = float(cfm_loss(f, t, xt, target))
        p.add_(-2 * h * v)
        minus = float(cfm_loss(f, t, xt, target))
        p.add_(h * v)
    central = (plus - minus) / (2 * h)
    rel = abs(autodiff - central) / max(abs(autodiff), abs(central), 1e-300)
    return {"autodiff": autodiff, "central": central, "rel_err": rel}


@torch.no_grad()
def sample_cfm(field: nn.Module, n: int, *, seed: int, method: str = "adaptive_heun",
               n_steps: int | None = None, rtol: float | None = None, atol: float | None = None):
    """Draw n samples: x_0 ~ N(0, I_2), then integrate dx/dt = u_t^theta(x) from t = 0 to t = 1 (TODO 3.1).

    Uses the given `common/odesolve.py`. Returns the SolverOutput, which carries .nfe, .method,
    .n_steps, .accepted and .rejected: read them from this value, before any arithmetic.
    """
    x0 = torch.randn(n, 2, generator=torch.Generator().manual_seed(seed))
    rhs = lambda t, x: field(t, x)    # noqa: E731   f(t, x), time first
    return odesolve.odeint(rhs, x0, 0.0, 1.0, method=method, n_steps=n_steps, rtol=rtol, atol=atol)


def cfm_tolerance_rows(field: nn.Module, dataset: str, cfg: dict) -> list[dict]:
    """The evaluation sweep of the budget's tolerance grid (TODO 3.2): test NLL by the augmented ODE.

    The same grid, test set, solver (dopri5) and exact divergence as the U2.L2 sweep, so panel (c)
    follows one protocol (the two fields differ in quality). Likelihood is not in the CFM training loss; it is recovered here.
    """
    x_test = GENERATORS[dataset](cfg["data"]["n_test"], cfg["data"]["seed_test"])
    grid = cfg["tolerance_grid"]
    rows = []
    for rtol in grid["rtol"]:
        r = evaluate_nll(field, x_test, rtol=rtol, atol=rtol * grid["atol_factor"])
        rows.append({"rtol": rtol, "atol": rtol * grid["atol_factor"], "nfe_eval": r["nfe"],
                     "nll_test": r["nll"], "wallclock_eval_s": r["wall_s"]})
    return rows


def write_cfm_evidence(out_dir: Path, log: painplots.TrainLog, tol_rows: list[dict], meta: dict,
                       cfg: dict) -> list[str]:
    """Write one CFM evidence directory in the painplots v1.1 layout; return the validator's problems."""
    shutil.rmtree(out_dir, ignore_errors=True)
    painplots.pain_wallclock(log, out_dir)
    painplots.pain_nfe_growth(log, out_dir)           # two columns of zeros, by the NFE rule
    painplots.pain_tolerance(tol_rows, out_dir)
    painplots.write_meta(meta, out_dir)
    (out_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True))
    return painplots.validate_evidence_dir(out_dir, expected_session="U3.L1")


def torchcfm_batch_fn(matcher):
    """A `batch_fn` for `train_cfm` that hands OUR (t, x0, x1) to TorchCFM and uses its x_t and target."""
    def batch_fn(train_x: torch.Tensor, batch: int, gen: torch.Generator):
        t, x0, x1, _, _ = sample_batch(train_x, batch, gen)
        t, xt, target = matcher.sample_location_and_conditional_flow(x0, x1, t=t)
        return t, x0, x1, xt, target
    return batch_fn


def jax_cfm(cfg: dict, train_x: np.ndarray, seed: int, iters: int | None = None) -> dict:
    """The same loop in plain JAX: the budget's MLP, tanh, a zero head, AdamW and warmup-cosine by hand.

    Honest timing (U0.L5): the up-front price is measured from BEFORE `lower`, and lowering and XLA
    compilation are recorded separately; the steady state is a median over calls, each closed by
    `block_until_ready`.
    """
    iters = cfg["train"]["iters"] if iters is None else iters
    m, o, b = cfg["model"], cfg["opt"], cfg["train"]["batch"]
    key = jax.random.key(seed)
    sizes = [3] + [m["hidden"]] * m["depth"] + [2]
    params = []
    for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        key, k1, k2 = jax.random.split(key, 3)
        bound = 1.0 / math.sqrt(n_in)            # the PyTorch nn.Linear default, U(-1/sqrt(n), 1/sqrt(n))
        w = jax.random.uniform(k1, (n_in, n_out), minval=-bound, maxval=bound)
        bias = jax.random.uniform(k2, (n_out,), minval=-bound, maxval=bound)
        if i == len(sizes) - 2:
            w, bias = jnp.zeros_like(w), jnp.zeros_like(bias)     # the zero head of the budget
        params.append((w, bias))

    def field(params, t, x):                       # u_t^theta(x), t of shape (B,)
        h = jnp.concatenate([x, t[:, None]], axis=1)
        for w, bias in params[:-1]:
            h = jnp.tanh(h @ w + bias)
        w, bias = params[-1]
        return h @ w + bias

    def loss_fn(params, t, x0, x1):
        xt = t[:, None] * x1 + (1 - t[:, None]) * x0
        return jnp.mean(jnp.sum((field(params, t, xt) - (x1 - x0)) ** 2, axis=1))

    warm = max(1, round(cfg["sched"]["warmup_frac"] * iters))
    b1, b2, eps, wd, lr0 = o["betas"][0], o["betas"][1], 1e-8, o["weight_decay"], o["lr"]

    def lr_at(step):                               # warmup-cosine on the step index, as in PyTorch
        p = (step - warm) / max(1, iters - warm)
        return lr0 * jnp.where(step < warm, step / warm, 0.5 * (1 + jnp.cos(jnp.pi * p)))

    def step_fn(params, mom, vel, step, key, data):
        k1, k2, k3, key = jax.random.split(key, 4)
        idx = jax.random.randint(k1, (b,), 0, data.shape[0])
        x1 = data[idx]
        x0 = jax.random.normal(k2, x1.shape)
        t = jax.random.uniform(k3, (b,))
        loss, grads = jax.value_and_grad(loss_fn)(params, t, x0, x1)
        lr = lr_at(step)
        n = step + 1
        mom = jax.tree.map(lambda m_, g: b1 * m_ + (1 - b1) * g, mom, grads)
        vel = jax.tree.map(lambda v_, g: b2 * v_ + (1 - b2) * g * g, vel, grads)
        params = jax.tree.map(                     # AdamW: decoupled weight decay, then the Adam step
            lambda p_, m_, v_: p_ * (1 - lr * wd) - lr * (m_ / (1 - b1 ** n)) / (jnp.sqrt(v_ / (1 - b2 ** n)) + eps),
            params, mom, vel)
        return params, mom, vel, key, loss

    data = jnp.asarray(train_x)
    mom = jax.tree.map(jnp.zeros_like, params)
    vel = jax.tree.map(jnp.zeros_like, params)
    t0 = time.perf_counter()
    lowered = jax.jit(step_fn).lower(params, mom, vel, 0, key, data)
    t1 = time.perf_counter()
    compiled = lowered.compile()
    t2 = time.perf_counter()
    losses, times = [], []
    for step in range(iters):
        s0 = time.perf_counter()
        params, mom, vel, key, loss = compiled(params, mom, vel, step, key, data)
        loss.block_until_ready()
        times.append(time.perf_counter() - s0)
        losses.append(float(loss))
    return {"lower_s": t1 - t0, "compile_s": t2 - t1, "upfront_s": t2 - t0,
            "steady_s_per_iter": statistics.median(times[10:]), "losses": losses}


# ---------------------------------------------------------------------------
# Helpers that are only the reference script's
# ---------------------------------------------------------------------------
def load_cfm_field(dataset: str, cfg: dict = BUDGET) -> nn.Module:
    ck = torch.load(WEIGHTS_DIR / f"u3l1-{dataset}.pt", map_location="cpu")
    f = make_field(cfg)
    f.load_state_dict(ck["state_dict"])
    return f.eval()


def windows(values: list[float], n: int) -> dict:
    return {"first300": statistics.fmean(values[:300]), "last300": statistics.fmean(values[n - 300:n]),
            "middle300": statistics.fmean(values[n // 2 - 150:n // 2 + 150])}


def save_fig(fig, name: str) -> None:
    u2.OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf"):
        path = u2.OUT_DIR / f"{name}.{ext}"
        fig.savefig(path, transparent=True, bbox_inches="tight")
        print(f"wrote {path}")


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def stage_ritual() -> dict:
    cfg = BUDGET
    out: dict = {}
    # 1. the loss at initialization, against E||x1||^2 + d
    for dataset in cfg["datasets"]:
        train_x = GENERATORS[dataset](cfg["data"]["n_train"], cfg["data"]["seed_train"])
        pop = float((train_x.double() ** 2).sum(1).mean()) + 2.0
        f = make_field(cfg)
        gen = torch.Generator().manual_seed(cfg["seed"])
        vals = []
        for _ in range(200):
            t, _, _, xt, target = sample_batch(train_x, cfg["train"]["batch"], gen)
            vals.append(float(cfm_loss(f, t, xt, target)))
        mean, se = statistics.fmean(vals), statistics.stdev(vals) / math.sqrt(len(vals))
        assert abs(mean - pop) < 4 * se, (dataset, mean, pop, se)
        one = vals[0]
        sd1 = statistics.stdev(vals)
        out[dataset] = {"loss_init_population": pop, "loss_init_mean200": mean, "loss_init_se200": se,
                        "loss_init_sd_one_batch": sd1, "window_lo": pop - 4 * sd1, "window_hi": pop + 4 * sd1,
                        "loss_init_first_batch": one}
    # 2. the directional check, moved head against zero head
    train_x = GENERATORS["two-moons"](cfg["data"]["n_train"], cfg["data"]["seed_train"])
    batch = sample_batch(train_x, 64, torch.Generator().manual_seed(7))
    torch.manual_seed(cfg["seed"])
    f = make_field(cfg)
    zero = directional_check(f, batch)
    moved = directional_check(move_head(f), batch)
    assert zero["autodiff"] == 0.0, "with the zero head the first layer's gradient must be exactly 0"
    assert abs(moved["autodiff"]) > 1e-3 and moved["rel_err"] < 1e-6, moved
    # 3. planted defect: fresh randomness between +h and -h
    fm = move_head(f).double()
    p = fm.net[0].weight
    v = torch.randn(p.shape, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    v = v / v.norm()
    redraw = torch.Generator().manual_seed(8)

    def fresh_loss():
        tb, _, _, xb, yb = (z.double() for z in sample_batch(train_x, 64, redraw))
        return float(cfm_loss(fm, tb, xb, yb))
    with torch.no_grad():
        p.add_(1e-6 * v)
        plus = fresh_loss()
        p.add_(-2e-6 * v)
        minus = fresh_loss()
    bad_central = (plus - minus) / 2e-6
    bad_rel = abs(moved["autodiff"] - bad_central) / abs(moved["autodiff"])
    assert bad_rel > 1e3 * moved["rel_err"] and bad_rel > 1e-2, bad_rel
    # 4. one training step on the CPU
    fs = make_field(cfg)
    opt = torch.optim.AdamW(fs.parameters(), lr=cfg["opt"]["lr"])
    gen = torch.Generator().manual_seed(0)
    times = []
    for _ in range(60):
        s0 = time.perf_counter()
        tb, _, _, xb, yb = sample_batch(train_x, cfg["train"]["batch"], gen)
        loss = cfm_loss(fs, tb, xb, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        times.append(time.perf_counter() - s0)
    step_ms = 1e3 * statistics.median(times[10:])
    assert step_ms < 50, step_ms
    n_cfm = u2.n_params(make_field(cfg))
    n_cnf = u2.n_params(u2.load_field("two-moons"))
    assert n_cfm == n_cnf
    return {"ritual": {"loss_init": out, "zero_head": zero, "moved_head": moved, "head_scale": HEAD_SCALE,
                       "fresh_randomness_rel_err": bad_rel, "fd_h": 1e-6, "batch": 64,
                       "step_ms_median": step_ms, "n_params": n_cfm, "n_params_cnf": n_cnf}}


def stage_train() -> dict:
    cfg = BUDGET
    out: dict = {}
    SCRATCH.mkdir(parents=True, exist_ok=True)
    for dataset in cfg["datasets"]:
        print(f"--- reference CFM run: {dataset}", flush=True)
        field, log = train_cfm(cfg, dataset)
        n = len(log)
        assert all(math.isfinite(v) for v in log.loss)
        ck = WEIGHTS_DIR / f"u3l1-{dataset}.pt"
        torch.save({"state_dict": field.state_dict(), "dataset": dataset, "seed": cfg["seed"]}, ck)
        np.save(SCRATCH / f"log_{dataset}.npy", np.array([log.step_s, log.nfe_fwd, log.nfe_bwd, log.loss]))
        lw, sw = windows(log.loss, n), windows(log.step_s, n)
        assert lw["first300"] > lw["last300"], "the reference CFM run did not train"
        out[dataset] = {"iters": n, "train_wall_s": sum(log.step_s), "loss_windows": lw, "step_s_windows": sw,
                        "step_ms_median": 1e3 * statistics.median(log.step_s),
                        "loss_curve": [statistics.fmean(log.loss[i:i + 10]) for i in range(0, n, 10)],
                        "checkpoint_sha256": u2.sha256(ck)}
        (ck.with_suffix(".json")).write_text(json.dumps({
            "sha256": out[dataset]["checkpoint_sha256"],
            "built_by": "scripts/python/fig_u3_l1_payoff.py, stage train",
            "loads_into": "make_field(cfg) of the U3.L1 notebook (the U2.L2 Field class, MLP 3x64)",
            "budget_version": cfg["budget_version"], "seed": cfg["seed"], "dataset": dataset,
            "purpose": "reference CFM field: notes figures and rescue of the sampling and payoff parts",
        }, indent=2) + "\n")
    return {"train": out}


def stage_sample() -> dict:
    cfg = BUDGET
    out: dict = {}
    for dataset in cfg["datasets"]:
        field = load_cfm_field(dataset)
        with torch.no_grad():
            ref = u2.make_generator(field)(torch.randn(N_SAMPLES, 2, generator=torch.Generator().manual_seed(11)))
        rows = {}
        for name, kw in (("rk4_8", {"method": "rk4", "n_steps": 8}), ("rk4_32", {"method": "rk4", "n_steps": 32}),
                         ("adaptive_heun", {"method": "adaptive_heun", "rtol": 1e-5, "atol": 1e-7})):
            x = sample_cfm(field, N_SAMPLES, seed=11, **kw)
            rows[name] = {"nfe": x.nfe, "method": x.method, "n_steps": x.n_steps, "accepted": x.accepted,
                          "rejected": x.rejected,
                          "max_dev_from_dopri5": float((x.as_subclass(torch.Tensor) - ref).abs().max())}
        out[dataset] = rows
        if dataset == "checkerboard":
            x = sample_cfm(field, N_SAMPLES, seed=11, method="adaptive_heun", rtol=1e-5, atol=1e-7)
            cfm_out = 1 - float(in_checkerboard_support(x.as_subclass(torch.Tensor)).float().mean())
            cnf = u2.load_field("checkerboard")
            with torch.no_grad():
                xc = u2.make_generator(cnf)(torch.randn(N_SAMPLES, 2, generator=torch.Generator().manual_seed(11)))
            cnf_out = 1 - float(in_checkerboard_support(xc).float().mean())
            out["gap"] = {"cfm_frac_outside_support": cfm_out, "cnf_frac_outside_support": cnf_out,
                          "n": N_SAMPLES}
            np.save(SCRATCH / "checker_samples.npy", x.as_subclass(torch.Tensor).numpy())
    return {"sample": out}


def stage_tolerance() -> dict:
    cfg = BUDGET
    out: dict = {}
    for dataset in cfg["datasets"]:
        rows = cfm_tolerance_rows(load_cfm_field(dataset), dataset, cfg)
        assert all(math.isfinite(r["nll_test"]) for r in rows)
        out[dataset] = {"rows": rows, "tightest_nll": rows[-1]["nll_test"],
                        "max_gap": max(abs(r["nll_test"] - rows[-1]["nll_test"]) for r in rows)}
        (SCRATCH / f"tol_{dataset}.json").write_text(json.dumps(rows))
    return {"tolerance": out}


def stage_crosscheck() -> dict:
    cfg = BUDGET
    matcher = ConditionalFlowMatcher(sigma=0.0)
    train_x = GENERATORS["two-moons"](cfg["data"]["n_train"], cfg["data"]["seed_train"])
    t, x0, x1, xt, target = sample_batch(train_x, 256, torch.Generator().manual_seed(3))
    tt, xt2, ut2 = matcher.sample_location_and_conditional_flow(x0, x1, t=t)
    dx, du = float((xt - xt2).abs().max()), float((target - ut2).abs().max())
    assert torch.equal(tt, t) and dx < 1e-6 and du < 1e-6, (dx, du)
    _, log_s = train_cfm(cfg, "two-moons", verbose=False)
    _, log_t = train_cfm(cfg, "two-moons", batch_fn=torchcfm_batch_fn(matcher), verbose=False)
    diff = max(abs(a - b) for a, b in zip(log_s.loss, log_t.loss))
    assert diff < 1e-4, diff
    return {"crosscheck": {"max_abs_xt": dx, "max_abs_target": du, "max_abs_loss_diff_3000": diff,
                           "torchcfm_version": "1.0.7",
                           "scratch_last300": statistics.fmean(log_s.loss[-300:]),
                           "torchcfm_last300": statistics.fmean(log_t.loss[-300:])}}


def stage_jax() -> dict:
    cfg = BUDGET
    train_x = GENERATORS["two-moons"](cfg["data"]["n_train"], cfg["data"]["seed_train"]).numpy()
    r = jax_cfm(cfg, train_x, cfg["seed"])
    n = len(r["losses"])
    torch_last = statistics.fmean(np.load(SCRATCH / "log_two-moons.npy")[3][-300:])
    jl = r.pop("losses")
    r.update({"first300": statistics.fmean(jl[:300]), "last300": statistics.fmean(jl[n - 300:]),
              "torch_last300": float(torch_last), "jax_version": jax.__version__})
    assert r["first300"] > r["last300"], "the JAX loop did not train"
    return {"jax": r}


def longer_run(cfg: dict, dataset: str, iters: int, seed: int) -> dict:
    """One CFM run outside the budget: only the iteration count changes (the schedule scales with it).

    Test NLL as in the U2.L2 seeds study (dopri5, rtol 1e-6, exact divergence). For the checkerboard,
    the fraction of samples outside the support (dopri5 flow map, the U2.L2 sampler, rtol 1e-5).
    """
    t0 = time.perf_counter()
    field, log = train_cfm(cfg, dataset, seed=seed, iters=iters, verbose=False)
    wall = sum(log.step_s)
    x_test = GENERATORS[dataset](cfg["data"]["n_test"], cfg["data"]["seed_test"])
    row = {"seed": seed, "iters": iters, "train_wall_s": wall, "total_s": time.perf_counter() - t0,
           "loss_last300": statistics.fmean(log.loss[-300:]),
           "test_nll_rtol1e-6": evaluate_nll(field.eval(), x_test, rtol=1e-6, atol=1e-8)["nll"]}
    if dataset == "checkerboard":
        with torch.no_grad():
            x = u2.make_generator(field)(torch.randn(N_SAMPLES, 2, generator=torch.Generator().manual_seed(11)))
        row["frac_outside_support"] = 1 - float(in_checkerboard_support(x).float().mean())
    return row


def stage_longer() -> dict:
    """The second receipt: CFM at the budget's 3000 iterations and at 30000, five seeds each,
    against the five-seed CNF study of U2.L2 (same seeds, same test metric)."""
    cfg = BUDGET
    cnf = json.loads(u2.JSON_OUT.read_text())["seeds"]
    out: dict = {"iters_long": LONG_ITERS, "seeds": list(u2.SEEDS_STUDY)}
    for dataset in cfg["datasets"]:
        res = {}
        for iters in (cfg["train"]["iters"], LONG_ITERS):
            rows = [longer_run(cfg, dataset, iters, seed) for seed in u2.SEEDS_STUDY]
            nll = [r["test_nll_rtol1e-6"] for r in rows]
            last = [r["loss_last300"] for r in rows]
            res[str(iters)] = {"rows": rows, "nll_min": min(nll), "nll_max": max(nll),
                               "loss_last300_min": min(last), "loss_last300_max": max(last),
                               "nll_mean": statistics.fmean(nll), "nll_sd": statistics.stdev(nll),
                               "wall_max_s": max(r["train_wall_s"] for r in rows),
                               "wall_mean_s": statistics.fmean(r["train_wall_s"] for r in rows)}
            if dataset == "checkerboard":
                fr = [r["frac_outside_support"] for r in rows]
                res[str(iters)].update({"outside_min": min(fr), "outside_max": max(fr)})
            print(f"  [{dataset}] {iters} iters: NLL {min(nll):.3f}..{max(nll):.3f}", flush=True)
        c = cnf[dataset]
        cnf_wall = [r["wall_s"] for r in c["rows"] if r["finite"]]
        res["cnf"] = {"nll_min": c["test_nll_min"], "nll_max": c["test_nll_max"],
                      "wall_min_s": min(cnf_wall), "wall_mean_s": statistics.fmean(cnf_wall),
                      "n_finite": c["n_finite"]}
        res["long_wall_over_cnf_wall"] = res[str(LONG_ITERS)]["wall_mean_s"] / res["cnf"]["wall_mean_s"]
        out[dataset] = res
    return {"longer": out}


def stage_probes() -> dict:
    """Fixed outputs of the reference functions on fixed inputs: the checkpoints call the STUDENT's
    function on the same inputs and compare (MEMORY 2026-10-04: a checkpoint must test the student's code)."""
    cfg = BUDGET
    train_x = GENERATORS["two-moons"](cfg["data"]["n_train"], cfg["data"]["seed_train"])
    t, x0, x1, xt, target = sample_batch(train_x, 4, torch.Generator().manual_seed(0))
    field = load_cfm_field("two-moons")
    tb, _, _, xb, yb = sample_batch(train_x, 256, torch.Generator().manual_seed(5))
    with torch.no_grad():
        loss_ref = float(cfm_loss(field, tb, xb, yb))
        loss_flip = float(cfm_loss(field, tb, xb, -yb))
        loss_coord_mean = float(((field(tb, xb) - yb) ** 2).mean())
    assert loss_flip > 2 * loss_ref and abs(loss_coord_mean - loss_ref / 2) < 1e-5
    xs = sample_cfm(field, 4, seed=11, method="rk4", n_steps=8)
    return {"probes": {
        "batch": {"t": t.tolist(), "x0": x0.tolist(), "x1": x1.tolist(), "xt": xt.tolist(), "target": target.tolist()},
        "loss": {"seed": 5, "batch": 256, "value": loss_ref, "flipped_target": loss_flip},
        "sample": {"seed": 11, "n": 4, "method": "rk4", "n_steps": 8, "nfe": xs.nfe,
                   "x": xs.as_subclass(torch.Tensor).tolist()}}}


def stage_evidence() -> dict:
    cfg = BUDGET
    out: dict = {}
    pair = painplots.mint_pair_id(reference=True)
    for dataset in cfg["datasets"]:
        s, f, b, l = np.load(SCRATCH / f"log_{dataset}.npy")
        log = painplots.TrainLog()
        for i in range(len(s)):
            log.add(s[i], f[i], b[i], l[i])
        rows = json.loads((SCRATCH / f"tol_{dataset}.json").read_text())
        resolved = {**cfg, "dataset": dataset}
        meta = painplots.make_meta(dataset, cfg["seed"], resolved, pair_id=pair,
                                   budget_version=cfg["budget_version"], evidence_origin="reference_pair",
                                   divergence_mode="exact", session_id=SESSION)
        d = REF_DIR / "cfm_evidence" / dataset
        problems = write_cfm_evidence(d, log, rows, meta, resolved)
        assert not problems, problems
        p = payoff.find_pair(CNF_REF / dataset, d)
        ratios = payoff.paired_ratios(p)
        run_id = painplots.log_run(d, run_name=f"u3l1-{dataset}-reference", log=log, tracking_uri=MLFLOW_URI,
                                   experiment="U3",
                                   tags={"session": SESSION, "dataset": dataset, "hardware": "cpu",
                                         "machine_id": meta["machine_id"], "pair_id": pair,
                                         "evidence_origin": "reference_pair"},
                                   params={"budget_version": cfg["budget_version"], "seed": cfg["seed"],
                                           "lr": cfg["opt"]["lr"], "iters": cfg["train"]["iters"],
                                           "hidden": cfg["model"]["hidden"], "config_hash": meta["config_hash"]})
        mlflow.tracking.MlflowClient().log_metric(run_id, "field_evals_train", payoff.FIELD_EVALS_TRAIN)
        painplots.export_run(run_id, REF_DIR / "mlflow_export" / dataset, tracking_uri=MLFLOW_URI)
        out[dataset] = {"machine_id": meta["machine_id"], "pair_id": pair, "validator": "PASS",
                        "pairing": "PASS", "ratios": ratios, "device_name": meta["device_name"]}
    return {"evidence": out}


def stage_figures(data: dict) -> dict:
    cfg = BUDGET
    for dataset in cfg["datasets"]:
        p = payoff.find_pair(CNF_REF / dataset, REF_DIR / "cfm_evidence" / dataset)
        fig = payoff.plot_payoff(p, title=False)
        save_fig(fig, f"u3-l1-payoff-{dataset}")
    # the learned two-moons field at the budget's iterations and at ten times them (seed 2026, the same
    # run as the first row of the `longer` stage): density by the augmented ODE, warped grid by the flow map
    long_field, _ = train_cfm(cfg, "two-moons", iters=LONG_ITERS, verbose=False)
    fig, axes = plt.subplots(2, 2, figsize=(5.4, 5.6))   # small canvas: the text stays legible in a slide column
    for row, (field, label) in enumerate(((load_cfm_field("two-moons"), f"{cfg['train']['iters']} iterations"),
                                          (long_field.eval(), f"{LONG_ITERS} iterations"))):
        density_heatmap(make_log_prob(field), ax=axes[row, 0], n=160)
        warped_grid(u2.make_generator(field), ax=axes[row, 1], color=UFRJ_BLUE)
        axes[row, 0].set_ylabel(label.replace(" iterations", " it."), fontsize=FS["label"], color=INK)
    axes[0, 0].set_title("density", fontsize=FS["title"], color=INK)
    axes[0, 1].set_title("flow map of a grid", fontsize=FS["title"], color=INK)
    for ax in axes.ravel():
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_aspect("equal")
        for side in ax.spines.values():
            side.set_color(UFRJ_GRAY)
    fig.tight_layout()
    save_fig(fig, "u3-l1-learned-two-moons")
    plt.close(fig)
    # the loss curves of the two datasets (windows of 10 iterations)
    fig, ax = plt.subplots(figsize=(6.6, 3.5))   # read at slide size (QA round 1)
    for dataset, color in (("two-moons", UFRJ_BLUE), ("checkerboard", UFRJ_ACCENT)):
        curve = data["train"][dataset]["loss_curve"]
        ax.plot(range(10, 10 * len(curve) + 1, 10), curve, color=color, lw=1.4, label=dataset)
        pop = data["ritual"]["loss_init"][dataset]["loss_init_population"]
        ax.axhline(pop, color=color, lw=0.9, ls=":")
    ax.set_xlabel("training iteration", fontsize=FS["label"])
    ax.set_ylabel(r"minibatch $\mathcal{L}_{\mathrm{CFM}}$ (mean of 10)", fontsize=FS["label"])
    ax.legend(fontsize=FS["legend"], frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.62))
    u2.apply_house_style(ax)
    fig.tight_layout()
    save_fig(fig, "u3-l1-loss")
    plt.close(fig)
    return {"figures": {"written": True}}


STAGES = ("ritual", "train", "sample", "tolerance", "crosscheck", "jax", "longer", "probes", "evidence", "figures")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(STAGES))
    only = ap.parse_args().only.split(",")
    unknown = [s for s in only if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stages {unknown}; the stages are {STAGES}")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(JSON_OUT.read_text()) if JSON_OUT.exists() else {}
    u2.set_all_seeds(BUDGET["seed"])
    for stage in STAGES:
        if stage not in only:
            continue
        print(f"=== stage {stage}", flush=True)
        t0 = time.perf_counter()
        res = stage_figures(data) if stage == "figures" else globals()[f"stage_{stage}"]()
        data.update(res)
        data.setdefault("_wall_s", {})[stage] = time.perf_counter() - t0
        data["_provenance"] = {**u2.git_state(), "torch": torch.__version__, "device": u2.cpu_model(),
                               "machine_id_ref": painplots.machine_id(reference=True), "threads": torch.get_num_threads()}
        JSON_OUT.write_text(json.dumps(data, indent=2) + "\n")
        print(f"=== stage {stage} done in {data['_wall_s'][stage]:.1f} s", flush=True)


if __name__ == "__main__":
    main()
