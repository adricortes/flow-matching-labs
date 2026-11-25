"""U3.L2 reference implementation and measured stages: diffusion as a special case.

ONE SOURCE FOR THE CODE. The functions in the section "solutions and given code" are the
notebook's solutions and given cells: `make_lab_u3l2.py` copies them with `inspect.getsource` and
derives each TODO stub from its solution. Edit them here, never in the notebook.

THE THREE CHANGES (L3-U3-L2 section 2, Block 1). Against `cfm-scratch` (U3.L1), the edit changes
    path        CondOT x_t = t x_1 + (1 - t) x_0   ->  a discrete DDPM VP grid (alpha_k, sigma_k, t_k)
    coordinate  regress u_t on x_1 - x_0           ->  regress eps on x_0 (the noise sample)
    weight      the CFM loss                        ->  DDPM-simple, or the induced weight B_t^2
and nothing else: the network, the optimizer and the loop are the inherited infrastructure.

TIME. Every tensor in this file is on the FM arrow, t = 0 noise and t = 1 data. The native DDPM
index k (k = K noisiest) and the native continuous time tau = 1 - t appear only inside the
schedule constructor and inside the Azula adapter, which is the Rosetta translation s = 1 - t.

THE SCHEDULE. The DDPM grid of Ho et al. 2020, literally: K = 1000 betas linear from 1e-4 to 0.02,
alpha_bar_k = prod_{j <= k} (1 - beta_j), at FM times t_k = 1 - k / K. The ODE sampler and the
weighting check need derivatives in t, which a grid does not have. The smooth schedule is the
Gamma-function extension of that product to real kappa = K (1 - t): with Delta the step of the beta
line and c = (1 - beta_min) / Delta, 1 - beta_j = Delta (c - j + 1), so
    alpha_bar(kappa) = Delta^kappa Gamma(c + 1) / Gamma(c + 1 - kappa),
which equals the product at every integer kappa. It passes through every node of the grid, and its
derivative is a digamma function. Both are computed with the Stirling series (see alpha_bar_fm).
Stage `schedule` asserts the agreement with the grid and the derivative.

Run:  conda run -n torch-gpu python scripts/python/fig_u3_l2_ddpm.py [--only stage,stage]
"""

from __future__ import annotations

import argparse
import copy
import difflib
import inspect
import json
import math
import re
import sys
import hashlib
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import azula.denoise  # noqa: E402
import azula.noise  # noqa: E402
import azula.sample  # noqa: E402
from common import odesolve  # noqa: E402
from common import painplots  # noqa: E402
from common.eval import MNIST_FID_SAMPLES, MNISTFeatures, fid, ref_stats  # noqa: E402
from common.painplots import config_hash  # noqa: E402
from common.unet import UNet  # noqa: E402

JSON_OUT = Path(__file__).resolve().parent / "_outputs" / "u3l2_ddpm.json"
DATA_DIR = Path(__file__).resolve().parent / "data"
WEIGHTS_DIR = ROOT / "common" / "weights"
REF_DIR = ROOT / "labs" / "u3l2" / "reference"
FIG_DIR = ROOT / "course" / "Figures"
SCRATCH = ROOT / ".scratch" / "u3l2"
MLFLOW_URI = f"sqlite:///{SCRATCH / 'mlflow.db'}"
SESSION = "U3.L2"
CKPT_NAME = "u3l2-cond-mnist"


# ===========================================================================
# Solutions and given code (copied into the notebook by make_lab_u3l2.py)
# ===========================================================================

K = 1000            # grid size: the DDPM chain has K steps
BETA_MIN = 1e-4     # the first beta of Ho et al. (2020)
BETA_MAX = 2e-2     # the last beta of Ho et al. (2020)
NULL_LABEL = 10     # the extra row of the label table (used by cfg_dropout)


class VPGrid:
    """The DDPM VP path on the FM arrow (TODO 1.1). All tensors float64, indexed by k = 0..K.

    Native DDPM numbers the chain k = 0 (data) .. K (noisiest). This class translates it once:
    t[k] = 1 - k/K is the FM time, alpha[k] and sigma[k] the coefficients of
    X_{t_k} = alpha_k X_1 + sigma_k X_0. Row k = 0 is the data end (alpha = 1, sigma = 0) and is
    never a training time; rows 1..K are. Training code reads (t, alpha, sigma) and never k's arrow.
    """

    def __init__(self, n_steps: int = K, beta_min: float = BETA_MIN, beta_max: float = BETA_MAX):
        self.n_steps, self.beta_min, self.beta_max = n_steps, beta_min, beta_max
        self.delta = (beta_max - beta_min) / (n_steps - 1)           # step of the beta line
        self.c = (1.0 - beta_min) / self.delta                       # 1 - beta_j = delta (c - j + 1)
        k = torch.arange(n_steps + 1, dtype=torch.float64)
        self.t = 1.0 - k / n_steps                                   # FM time of each grid row
        self.beta = torch.zeros_like(k)                              # beta[0] = 0: no step into row 0
        self.beta[1:] = torch.linspace(beta_min, beta_max, n_steps, dtype=torch.float64)
        self.alpha_bar = torch.cumprod(1.0 - self.beta, 0)           # alpha_bar[0] = 1 at the data end
        self.alpha = self.alpha_bar.sqrt()
        self.sigma = (1.0 - self.alpha_bar).sqrt()

    def alpha_bar_fm(self, t: torch.Tensor) -> torch.Tensor:
        """alpha_bar at FM time t: the smooth schedule, equal to the grid's product at every node.

        alpha_bar(kappa) = Delta^kappa Gamma(c + 1) / Gamma(c + 1 - kappa), kappa = K (1 - t), written
        with the Stirling series of log Gamma: two lgamma values near 5e4 cancel to 1e-10, this to 1e-13.
        The native position kappa stays inside this method and rate().
        """
        kappa = self.n_steps * (1.0 - t)
        z1, z2 = self.c + 1.0, self.c + 1.0 - kappa
        r = lambda z: 1 / (12 * z) - 1 / (360 * z**3) + 1 / (1260 * z**5)    # noqa: E731
        log_ab = (kappa * torch.log(self.delta * z2) - kappa - (self.c + 0.5) * torch.log1p(-kappa / z1)
                  + r(z1) - r(z2))
        return torch.exp(log_ab)

    def rate(self, t: torch.Tensor) -> torch.Tensor:
        """d/dt log alpha_bar of the smooth schedule, positive on [0, 1]: -K (log Delta + digamma(c + 1 - kappa))."""
        z2 = self.c + 1.0 - self.n_steps * (1.0 - t)
        dlog = torch.log(self.delta * z2) - 1 / (2 * z2) - 1 / (12 * z2**2) + 1 / (120 * z2**4) - 1 / (252 * z2**6)
        return -self.n_steps * dlog

    def coeffs(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """(alpha_t, sigma_t, dalpha_t, dsigma_t) of the smooth schedule (TODO 1.1, the interpolation).

        Valid on the interior domain only, where sigma_t > 0: dsigma_t divides by sigma_t.
        """
        # 1e-6: a float32 time (the ODE state's dtype) rounds t_1 = 0.999 up to 0.99900001; sigma is ~0.01 there
        assert float(t.max()) <= self.t[1] + 1e-6, "interior domain: t <= t_1 = 1 - 1/K (sigma_t > 0)"
        ab = self.alpha_bar_fm(t)
        alpha, sigma = ab.sqrt(), (1.0 - ab).sqrt()
        dalpha = 0.5 * self.rate(t) * alpha                          # d/dt sqrt(alpha_bar) = sqrt(alpha_bar) rate / 2
        dsigma = -alpha * dalpha / sigma                             # from alpha^2 + sigma^2 = 1
        return alpha, sigma, dalpha, dsigma

    def at(self, k: torch.Tensor, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(t_k, alpha_k, sigma_k) for (B,) grid rows k, as (B,), (B,1,1,1), (B,1,1,1) in like's dtype (TODO 1.1)."""
        col = (-1,) + (1,) * (like.ndim - 1)
        return (self.t[k].to(like), self.alpha[k].to(like).view(col), self.sigma[k].to(like).view(col))



def time_weight(grid: VPGrid, k: torch.Tensor, loss_mode: str) -> torch.Tensor:
    """The (B,) time weight of each loss mode at grid rows k (the third change of the diff).

    ddpm_simple: 1 at every row. cfm_equiv_eps: B_{t_k}^2, the weight that the velocity loss puts
    on the epsilon residual (TODO 1.3 writes b_coeff and checks the identity).
    """
    if loss_mode == "ddpm_simple":
        return torch.ones(k.shape, dtype=torch.float64)
    if loss_mode == "cfm_equiv_eps":
        return b_coeff(grid, grid.t[k]) ** 2
    raise ValueError(f"unknown loss_mode {loss_mode!r}")


def sample_batch(train_x: torch.Tensor, train_y: torch.Tensor, batch: int, gen: torch.Generator,
                 grid: VPGrid):
    """One minibatch on the VP grid (TODO 1.2). Returns (k, t, x0, x1, y, xt, target), on the CPU.

    The cfm-scratch function of U3.L1 with two lines changed (path) and one (coordinate).
    k ~ Unif{1, ..., K} is the course DDPM convention: one grid row per example, the data row
    k = 0 excluded. t = t_k is the (B,) FM time the network sees; x0 ~ N(0, I) is the noise.
    """
    idx = torch.randint(0, train_x.shape[0], (batch,), generator=gen)
    x1 = train_x[idx]                                                # (B, 1, 28, 28), data
    y = train_y[idx]                                                 # labels: the class pathway
    x0 = torch.randn(x1.shape, generator=gen, dtype=x1.dtype)        # (B, 1, 28, 28), noise
    k = torch.randint(1, grid.n_steps + 1, (batch,), generator=gen)  # path: a grid row, not t ~ Unif[0, 1]
    t, a, s = grid.at(k, x1)                                         # path: (B,), (B,1,1,1), (B,1,1,1)
    xt = a * x1 + s * x0                                             # path: X_{t_k} = alpha_k X_1 + sigma_k X_0
    target = x0                                                      # coordinate: the noise, not x1 - x0
    return k, t, x0, x1, y, xt, target


def drop_labels(y: torch.Tensor, p: float, gen: torch.Generator) -> torch.Tensor:
    """Replace each label by NULL_LABEL with probability p (the `cfg_dropout` field)."""
    mask = torch.rand(y.shape, generator=gen) < p
    return torch.where(mask, torch.full_like(y, NULL_LABEL), y)


def eps_loss(net: nn.Module, grid: VPGrid, k: torch.Tensor, t: torch.Tensor, xt: torch.Tensor,
             target: torch.Tensor, y: torch.Tensor, loss_mode: str) -> torch.Tensor:
    """The two epsilon losses (TODO 1.2 and 1.3): weighted squared norm over pixels, mean over the batch.

    ddpm_simple     E_{k, X_0, (X_1, Y), Y~} || eps_theta(X_{t_k}, t_k, Y~) - X_0 ||^2            (weight 1)
    cfm_equiv_eps   E_{k, X_0, (X_1, Y), Y~} B_{t_k}^2 || eps_theta(X_{t_k}, t_k, Y~) - X_0 ||^2  (induced by CFM)
    with (X_1, Y) an image and its label, and Y~ the label after cfg_dropout (the y argument).
    """
    eps_hat = net(xt, t, y)                                          # coordinate: eps_theta, not u_t^theta
    w = time_weight(grid, k, loss_mode).to(xt)                       # weight: 1, or B_{t_k}^2
    return (w * ((eps_hat - target) ** 2).flatten(1).sum(dim=1)).mean()  # weight: per example, then the mean


def b_coeff(grid: VPGrid, t: torch.Tensor) -> torch.Tensor:
    """B_t = dsigma_t - dalpha_t sigma_t / alpha_t, the noise coefficient of U3.T2 (TODO 1.3).

    Interior domain only (coeffs asserts t <= t_1). For any VP schedule B_t = -rate(t) / (2 sigma_t).
    """
    alpha, sigma, dalpha, dsigma = grid.coeffs(t)
    return dsigma - dalpha * sigma / alpha


def velocity_from_eps(grid: VPGrid, t: torch.Tensor, x: torch.Tensor, eps_hat: torch.Tensor) -> torch.Tensor:
    """u_t^theta = (dalpha_t / alpha_t) x + B_t eps_hat (TODO 1.4), on the interior domain.

    The U3.T2 dictionary, with the network's eps_hat in place of eps_bar_t(x) = E[X_0 | X_t = x].
    """
    alpha, _, dalpha, _ = grid.coeffs(t.double())
    c = (dalpha / alpha).to(x).view(-1, 1, 1, 1)                     # the coefficient of x
    b = b_coeff(grid, t.double()).to(x).view(-1, 1, 1, 1)            # the coefficient of eps
    return c * x + b * eps_hat


def x1_from_eps(alpha: torch.Tensor, sigma: torch.Tensor, x: torch.Tensor, eps_hat: torch.Tensor) -> torch.Tensor:
    """x_1-prediction (x - sigma_t eps) / alpha_t; needs alpha_t > 0, which holds on the whole grid (TODO 1.4)."""
    assert float(alpha.min()) > 0.0, "x1_from_eps divides by alpha_t"
    a, s = alpha.to(x).view(-1, 1, 1, 1), sigma.to(x).view(-1, 1, 1, 1)
    return (x - s * eps_hat) / a


def score_from_eps(sigma: torch.Tensor, eps_hat: torch.Tensor) -> torch.Tensor:
    """score -eps / sigma_t; needs sigma_t > 0, so never at the data end t = 1 (TODO 1.4)."""
    assert float(sigma.min()) > 0.0, "score_from_eps divides by sigma_t"
    return -eps_hat / sigma.to(eps_hat).view(-1, 1, 1, 1)


@torch.no_grad()
def sample_ode(net: nn.Module, grid: VPGrid, x0: torch.Tensor, y: torch.Tensor, *,
               method: str = "adaptive_heun", n_steps: int | None = None,
               rtol: float | None = None, atol: float | None = None):
    """Deterministic ODE/PF sampler (TODO 3.1): integrate u from t = 0 to t_1 = 1 - 1/K.

    The field is the continuous model induced by the smooth schedule and the network, not the
    finite DDPM chain. It stops at the last grid time t_1, where sigma_t > 0, and returns x_{t_1}
    (no division at the data end). Returns (SolverOutput, images): read .nfe from the first, before
    any arithmetic. Starting from N(0, I) at t = 0 is the terminal approximation: the forward
    marginal there is the law of alpha_K X_1 + sigma_K X_0, whose mean alpha_K E[X_1] is small, not zero.
    """
    shape = x0.shape                                                 # (B, 1, 28, 28); odeint wants (B, d)

    def rhs(t, x):                                                   # f(t, x), time first
        tb = t.expand(x.shape[0]).to(x.dtype)                        # (B,) for the network
        img = x.view(shape)
        return velocity_from_eps(grid, tb, img, net(img, tb, y)).flatten(1)
    sol = odesolve.odeint(rhs, x0.flatten(1), 0.0, float(grid.t[1]), method=method, n_steps=n_steps,
                          rtol=rtol, atol=atol)
    return sol, sol.as_subclass(torch.Tensor).view(shape)


def substeps(grid: VPGrid, n_steps: int) -> list[int]:
    """n_steps + 1 grid rows from K (noisiest) down to 0 (data), evenly spaced in k."""
    rows = np.linspace(grid.n_steps, 0, n_steps + 1).round().astype(int)
    assert len(set(rows)) == n_steps + 1, "n_steps must be at most K"
    return [int(r) for r in rows]


def posterior_step(grid: VPGrid, x: torch.Tensor, eps_hat: torch.Tensor, j: int, i: int,
                   z: torch.Tensor | None) -> torch.Tensor:
    """One move from grid row j to the less noisy row i < j (TODO 3.2 and the DDIM cell of 3.3).

    With z given: the DDPM ancestral update, a draw from the Gaussian posterior
    q(x_{t_i} | x_{t_j}, X_1 = x1_hat) of the forward chain, with x1_hat from eps_hat. For i = j - 1
    this is the update of Ho et al. 2020. With z = None: the DDIM update (eta = 0) of Song et al.
    2021, x_{t_i} = alpha_i x1_hat + sigma_i eps_hat.
    """
    a_j, s_j, a_i, s_i = (grid.alpha[j], grid.sigma[j], grid.alpha[i], grid.sigma[i])
    x1_hat = x1_from_eps(a_j.view(1), s_j.view(1), x, eps_hat)
    if z is None:
        return a_i.to(x) * x1_hat + s_i.to(x) * eps_hat
    a_ji = a_j / a_i                                                 # signal ratio from row i to row j
    var_ji = s_j**2 - a_ji**2 * s_i**2                               # variance of the forward move i -> j
    mean = (a_i * var_ji / s_j**2).to(x) * x1_hat + (a_ji * s_i**2 / s_j**2).to(x) * x
    var = (s_i**2 * var_ji / s_j**2).to(x)                           # posterior variance, 0 at i = 0
    return mean + var.sqrt() * z


@torch.no_grad()
def sample_chain(net: nn.Module, grid: VPGrid, x0: torch.Tensor, y: torch.Tensor, n_steps: int, *,
                 stochastic: bool, gen: torch.Generator | None = None) -> dict:
    """The ancestral chain (stochastic=True) or DDIM (False) on n_steps grid moves from row K to 0.

    Starts from x0 ~ N(0, I) at row K: the terminal approximation (alpha_K is small, not zero).
    Returns the samples and the two cost counters of a chain: sampler_steps and model_forwards.
    """
    rows = substeps(grid, n_steps)
    x = x0
    forwards = 0
    for j, i in zip(rows[:-1], rows[1:]):
        t = grid.t[j].to(x).expand(x.shape[0])
        eps_hat = net(x, t, y)
        forwards += 1
        noisy = stochastic and i > 0                                 # the last move (i = 0) adds no noise
        z = torch.randn(x.shape, generator=gen).to(x) if noisy else None
        x = posterior_step(grid, x, eps_hat, j, i, z)
    return {"x": x, "sampler_steps": n_steps, "model_forwards": forwards}


class CourseSchedule(azula.noise.Schedule):
    """The course VP schedule in Azula's dialect (TODO 4.1): native time s = 1 - t, clean -> noisy.

    Azula asks for (alpha_s, sigma_s) at native time s. The adapter answers with the course
    coefficients at FM time t = 1 - s. This is the only place where native time exists.
    """

    def __init__(self, grid: VPGrid):
        self.grid = grid

    def __call__(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ab = self.grid.alpha_bar_fm(1.0 - s.double())                # Rosetta: t = 1 - s
        return ab.sqrt().to(s.dtype), (1.0 - ab).sqrt().to(s.dtype)


class EpsDenoiser(azula.denoise.Denoiser):
    """The course epsilon network as an Azula denoiser (TODO 4.2): Azula expects E[X | x_s], the clean image."""

    def __init__(self, net: nn.Module, grid: VPGrid):
        super().__init__()
        self.net, self.schedule = net, CourseSchedule(grid)

    def forward(self, x_s: torch.Tensor, s: torch.Tensor, y: torch.Tensor | None = None, **kwargs):
        s = s.expand(x_s.shape[0]) if s.ndim == 0 else s
        t = (1.0 - s).to(x_s.dtype)                                  # back to the FM arrow for the network
        alpha, sigma = self.schedule(s)
        eps_hat = self.net(x_s, t, y)
        return azula.denoise.DiracPosterior(mean=x1_from_eps(alpha, sigma, x_s, eps_hat))


def move_head(net: nn.Module, scale: float = 0.1, seed: int = 0) -> nn.Module:
    """A copy of the network whose zero-initialized head is moved by a fixed, stated amount.

    With conv_out at exactly zero, every gradient upstream of it is exactly zero, and a check
    there measures nothing (MEMORY, U0.L4). The ritual therefore moves the head first.
    """
    out = copy.deepcopy(net)
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in out.conv_out.parameters():
            p.add_(scale * torch.randn(p.shape, generator=gen, dtype=p.dtype))
    return out


def directional_check(net: nn.Module, grid: VPGrid, batch: tuple, loss_mode: str, *, h: float = 1e-6,
                      seed: int = 0) -> dict:
    """Autograd against a central difference, in float64, along one fixed direction of conv_in.weight.

    `batch` is ONE frozen minibatch (k, t, x0, xt, y_in), labels already dropped: nothing is drawn
    between the +h and the -h evaluations. conv_in is the tensor farthest from the head.
    """
    f = copy.deepcopy(net).double()
    k, t, x0, xt, y_in = batch
    t, x0, xt = t.double(), x0.double(), xt.double()
    p = f.conv_in.weight
    v = torch.randn(p.shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    v = v / v.norm()
    grad = torch.autograd.grad(eps_loss(f, grid, k, t, xt, x0, y_in, loss_mode), p)[0]
    autodiff = float((grad * v).sum())
    with torch.no_grad():
        p.add_(h * v)
        plus = float(eps_loss(f, grid, k, t, xt, x0, y_in, loss_mode))
        p.add_(-2 * h * v)
        minus = float(eps_loss(f, grid, k, t, xt, x0, y_in, loss_mode))
        p.add_(h * v)
    central = (plus - minus) / (2 * h)
    rel = abs(autodiff - central) / max(abs(autodiff), abs(central), 1e-300)
    return {"autodiff": autodiff, "central": central, "rel_err": rel}


def code_lines(src: str) -> list[tuple[str, str]]:
    """(code, tag) for each code line of a source string: docstrings, comments and blank lines removed.

    The tag is the first word of a trailing comment when it is one of path, coordinate, weight, labels.
    """
    quotes = ('"' * 3, "'" * 3)
    out, in_doc = [], False
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith(quotes):
            if sum(stripped.count(q) for q in quotes) == 1:
                in_doc = not in_doc
            continue
        if in_doc or not stripped or stripped.startswith("#"):    # full-line comments are not code
            continue
        code_part, _, comment = line.partition("  #")
        tag = re.match(r"\s*(path|coordinate|weight|labels)\b", comment)
        out.append((code_part.rstrip(), tag.group(1) if tag else ""))
    return out


def thesis_diff(old_src: str, new_src: str) -> list[str]:
    """The unified diff of two sources' code lines (no context lines); each added line shows its tag."""
    old, new = code_lines(old_src), code_lines(new_src)
    tags = {c: t for c, t in new}
    lines = []
    for line in difflib.unified_diff([c for c, _ in old], [c for c, _ in new], "cfm-scratch (U3.L1)",
                                     "ddpm_simple (U3.L2)", lineterm="", n=0):
        if line.startswith("+") and not line.startswith("+++") and tags.get(line[1:]):
            line = f"{line}      [{tags[line[1:]]}]"
        lines.append(line)
    return lines


def make_unet(cfg: dict) -> UNet:
    """The given `unet-skeleton` at the lab's configuration, with one label row per digit plus the null label."""
    m = cfg["model"]
    return UNet(in_ch=1, base_ch=m["base_ch"], ch_mults=tuple(m["ch_mults"]),
                attn_resolutions=tuple(m["attn_resolutions"]), num_classes=m["num_classes"] + 1)


def warmup_cosine(step: int, total: int, warmup: int) -> float:
    """The recipe-card learning-rate factor: linear warmup, then cosine decay to zero."""
    if step < warmup:
        return step / max(1, warmup)
    return 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def train_step(net: nn.Module, opt: torch.optim.Optimizer, grid: VPGrid, train_x: torch.Tensor,
               train_y: torch.Tensor, cfg: dict, gen: torch.Generator) -> float:
    """One optimizer step: the minibatch on the CPU (as in U3.L1), then the step on the network's device."""
    dev = next(net.parameters()).device
    tr = cfg["train"]
    k, t, x0, x1, y, xt, target = sample_batch(train_x, train_y, tr["batch"], gen, grid)
    y_in = drop_labels(y, tr["cfg_dropout"], gen)                    # the cfg_dropout field
    loss = eps_loss(net, grid, k, t.to(dev), xt.to(dev), target.to(dev), y_in.to(dev), tr["loss_mode"])
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return float(loss.detach())


# ===========================================================================
# Configuration, data and the measured stages (not copied into the notebook)
# ===========================================================================

CONFIG = {
    "session": SESSION,
    "seed": 0,
    "data": {"name": "mnist", "split": "train[perm(seed 0)][:50000]", "val": "train[perm][50000:]",
             "scale": "[-1, 1]"},
    "grid": {"n_steps": K, "beta_min": BETA_MIN, "beta_max": BETA_MAX, "form": "ddpm-linear"},
    "model": {"base_ch": 32, "ch_mults": [1, 2, 2], "attn_resolutions": [7], "num_classes": 10},
    "train": {"iters": None, "batch": 128, "lr": 1e-3, "warmup": 500, "weight_decay": 0.0,
              "loss_mode": "ddpm_simple", "cfg_dropout": 0.1},
}


N_TRAIN = 50_000    # the rest of the MNIST training split is the validation set


def load_mnist(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """MNIST as (N, 1, 28, 28) float32 in [-1, 1] and (N,) long labels.

    split: "train" (50 000 rows of a seeded permutation of the training split), "val" (the other
    10 000; used for every choice of this build), or "test" (read once, by the reported FID).
    """
    ds = datasets.MNIST(str(DATA_DIR), train=(split != "test"), download=not (DATA_DIR / "MNIST").exists())
    x = ds.data.float().div(255.0).unsqueeze(1) * 2.0 - 1.0
    y = ds.targets.long()
    if split == "test":
        return x, y
    perm = torch.randperm(x.shape[0], generator=torch.Generator().manual_seed(0))
    rows = perm[:N_TRAIN] if split == "train" else perm[N_TRAIN:]
    return x[rows], y[rows]


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_run(cfg: dict, *, iters: int, seed: int, log_every: int = 100, on_log=None) -> tuple[nn.Module, list[dict], float]:
    """Train one epsilon model. Returns the network, the loss log and the wall-clock in seconds."""
    dev = device()
    torch.manual_seed(seed)
    net = make_unet(cfg).to(dev)                                     # built first: same init per seed
    x_all, y_all = load_mnist("train")                               # stays on the CPU, as in U3.L1
    grid = VPGrid(cfg["grid"]["n_steps"], cfg["grid"]["beta_min"], cfg["grid"]["beta_max"])
    gen = torch.Generator().manual_seed(seed + 1)
    tr = cfg["train"]
    opt = torch.optim.AdamW(net.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_cosine(s, iters, tr["warmup"]))
    log, run_loss = [], 0.0
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(1, iters + 1):
        loss = train_step(net, opt, grid, x_all, y_all, cfg, gen)
        if not math.isfinite(loss):
            raise FloatingPointError(f"non-finite loss {loss} at step {step}")
        sched.step()
        run_loss += loss
        if step % log_every == 0:
            row = {"step": step, "loss": run_loss / log_every, "lr": opt.param_groups[0]["lr"],
                   "wall_s": time.perf_counter() - t0}
            log.append(row)
            run_loss = 0.0
            if on_log:
                on_log(row)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return net, log, time.perf_counter() - t0


def stage_speed() -> dict:
    """Throughput of the training step on this machine, before any budget is chosen."""
    cfg = copy.deepcopy(CONFIG)
    out = {}
    for base in (32, 64):
        cfg["model"]["base_ch"] = base
        _, _, wall = train_run(cfg, iters=50, seed=0)               # warm-up of kernels
        _, _, wall = train_run(cfg, iters=300, seed=0)
        out[str(base)] = {"it_per_s": 300 / wall}
        print(f"base_ch={base}: {300 / wall:.1f} it/s", flush=True)
    return {"speed": {"device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                      "batch": cfg["train"]["batch"], "per_base_ch": out}}


_EXTRACTOR = None
_REFS: dict = {}


def extractor():
    global _EXTRACTOR
    if _EXTRACTOR is None:
        _EXTRACTOR = MNISTFeatures(device=device())
    return _EXTRACTOR


def ref_for(split: str):
    """Cached reference statistics of the pinned extractor on the val or the test split."""
    if split not in _REFS:
        x, _ = load_mnist(split)
        loader = DataLoader(TensorDataset((x + 1) / 2), batch_size=1024)
        _REFS[split] = ref_stats(loader, extractor(), dataset=f"mnist-u3l2-{split}", split=split,
                                 cache_dir=SCRATCH / "refstats")
    return _REFS[split]


def to_unit(x: torch.Tensor) -> torch.Tensor:
    """Model space [-1, 1] to the extractor's [0, 1], clamped."""
    return ((x + 1) / 2).clamp(0, 1)


@torch.no_grad()
def generate(net: nn.Module, grid: VPGrid, n: int, *, sampler: str, seed: int, batch: int = 1000,
             **kw) -> tuple[torch.Tensor, dict]:
    """n samples with balanced labels (n / 10 per digit). Returns samples in [0, 1] and the cost counters."""
    dev = device()
    gen = torch.Generator().manual_seed(seed)
    out, cost = [], {"nfe_per_solve": [], "model_forwards": 0, "sampler_steps": 0}
    labels = torch.arange(n) % 10
    for i in range(0, n, batch):
        y = labels[i:i + batch].to(dev)
        x0 = torch.randn(len(y), 1, 28, 28, generator=gen).to(dev)
        if sampler == "ode":
            sol, img = sample_ode(net, grid, x0, y, **kw)
            cost["nfe_per_solve"].append(sol.nfe)                    # one solve per batch
            out.append(to_unit(img).cpu())
        else:
            res = sample_chain(net, grid, x0, y, kw["n_steps"], stochastic=(sampler == "ancestral"), gen=gen)
            cost["model_forwards"], cost["sampler_steps"] = res["model_forwards"], res["sampler_steps"]
            out.append(to_unit(res["x"]).cpu())
    return torch.cat(out), cost


def fid_on(samples: torch.Tensor, split: str) -> float:
    return float(fid(samples, ref_for(split), extractor()))


def stage_probe() -> dict:
    """Five seeds at two learning rates, short runs, scored on VAL (FID at n = 2000, DDIM 50 steps)."""
    rows = []
    grid = VPGrid()
    for lr in PROBE_LRS:
        for seed in range(5):
            cfg = copy.deepcopy(CONFIG)
            cfg["train"]["lr"] = lr
            net, log, wall = train_run(cfg, iters=PROBE_ITERS, seed=seed)
            net.eval()
            samples, _ = generate(net, grid, 2000, sampler="ddim", seed=100 + seed, n_steps=50)
            row = {"lr": lr, "seed": seed, "final_loss": log[-1]["loss"], "fid_val_2000": fid_on(samples, "val"),
                   "wall_s": wall}
            rows.append(row)
            print(row, flush=True)
    return {"probe": {"iters": PROBE_ITERS, "rows": rows, **probe_floor()}}


def probe_floor() -> dict:
    """The real-data reference of the probe: 2000 TRAINING images against the validation statistics.

    The test split stays unread until stage fid. (A first version used 2000 test images; QA round 1, M5.)
    """
    x, _ = load_mnist("train")
    idx = torch.randperm(x.shape[0], generator=torch.Generator().manual_seed(3))[:2000]
    return {"real_train_2000_vs_val": fid_on((x[idx] + 1) / 2, "val")}


TINY = {"base_ch": 16, "ch_mults": [1, 2, 2], "attn_resolutions": [7], "num_classes": 10}


def frozen_batch(grid: VPGrid, n: int = 8, seed: int = 0) -> tuple:
    """One frozen float64 minibatch from the training split, labels dropped once: (k, t, x0, xt, y_in, x1)."""
    x, y = load_mnist("train")
    gen = torch.Generator().manual_seed(seed)
    k, t, x0, x1, y, xt, target = sample_batch(x.double(), y, n, gen, grid)
    y_in = drop_labels(y, 0.5, gen)                                  # p = 0.5 so the batch holds null labels too
    return k, t, x0, xt, y_in, x1


def tiny_net(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return move_head(make_unet({"model": TINY}).double(), scale=0.1, seed=seed)


def stage_schedule() -> dict:
    """Facts of the grid and of the smooth schedule, printed by the notebook in TODO 1.1."""
    g = VPGrid()
    ab_smooth = g.alpha_bar_fm(g.t)                                  # the smooth schedule at the nodes
    rows = torch.arange(1, K + 1)
    b = b_coeff(g, g.t[rows])
    tt = torch.tensor([0.0, 0.25, 0.5, 0.75, 0.9, float(g.t[1])], dtype=torch.float64)
    alpha, sigma, dalpha, dsigma = g.coeffs(tt)
    hstep = 1e-4      # central differences at interior points; lgamma near 5e4 carries ~1e-10 absolute error
    t_in = tt[1:-1]
    a_p, s_p, _, _ = g.coeffs(t_in + hstep)
    a_m, s_m, _, _ = g.coeffs(t_in - hstep)
    _, _, da_in, ds_in = g.coeffs(t_in)
    fd_a, fd_s = (a_p - a_m) / (2 * hstep), (s_p - s_m) / (2 * hstep)
    az = azula.noise.VPSchedule()                                    # Azula's own VP family, defaults
    az_a, az_s = az(1.0 - tt.float())
    out = {
        "K": K, "beta_min": BETA_MIN, "beta_max": BETA_MAX,
        "alpha_noise": float(g.alpha[K]), "sigma_noise": float(g.sigma[K]),
        "snr_noise": float(g.alpha[K] ** 2 / g.sigma[K] ** 2), "alpha_noise_is_zero": bool(g.alpha[K] == 0),
        "alpha_t1": float(g.alpha[1]), "sigma_t1": float(g.sigma[1]), "t1": float(g.t[1]),
        "alpha_bar_noise": float(g.alpha_bar[K]),
        "smooth_vs_grid_max_rel": float(((ab_smooth - g.alpha_bar).abs() / g.alpha_bar).max()),
        "b_min": float(b.min()), "b_max": float(b.max()),
        "b2_min": float((b**2).min()), "b2_max": float((b**2).max()),
        "b2_ratio": float((b**2).max() / (b**2).min()),
        "b_at_t1": float(b[0]), "b_at_t0": float(b[-1]),
        "b2_at_t1": float(b[0] ** 2), "t_b2_min": float(g.t[rows][int((b**2).argmin())]),
        "deriv_fd_max_rel": float(max(((fd_a - da_in).abs() / da_in.abs()).max(),
                                      ((fd_s - ds_in).abs() / ds_in.abs()).max())),
        "table": [{"t": float(t_), "alpha": float(a_), "sigma": float(s_), "azula_vp_alpha": float(x_),
                   "azula_vp_sigma": float(y_)} for t_, a_, s_, x_, y_ in zip(tt, alpha, sigma, az_a, az_s)],
    }
    assert not out["alpha_noise_is_zero"], "the finite grid has alpha_K > 0: the N(0, I) start is an approximation"
    assert out["deriv_fd_max_rel"] < 1e-6, out["deriv_fd_max_rel"]
    assert out["smooth_vs_grid_max_rel"] < 1e-12, "the smooth schedule must pass through every grid node"
    assert out["b_max"] < 0, "B_t < 0 on the whole grid (sigma decreases as t grows)"
    print(json.dumps({k_: v for k_, v in out.items() if k_ != "table"}, indent=1))
    return {"schedule": out}


def stage_weighting() -> dict:
    """TODO 1.3 on a frozen batch: velocity CFM in epsilon coordinates is B_t^2 times the epsilon residual."""
    g = VPGrid()
    net = tiny_net(0)
    k, t, x0, _, y_in, x1 = frozen_batch(g)
    alpha, sigma, dalpha, dsigma = g.coeffs(t)
    xt = alpha.view(-1, 1, 1, 1) * x1 + sigma.view(-1, 1, 1, 1) * x0  # the same (alpha, sigma) as the conversion
    with torch.no_grad():
        eps_hat = net(xt, t, y_in)
    u_hat = velocity_from_eps(g, t, xt, eps_hat)
    u_target = dalpha.view(-1, 1, 1, 1) * x1 + dsigma.view(-1, 1, 1, 1) * x0   # the conditional velocity
    vel = ((u_hat - u_target) ** 2).flatten(1).sum(1)
    weighted = b_coeff(g, t) ** 2 * ((eps_hat - x0) ** 2).flatten(1).sum(1)
    rel = float(((vel - weighted).abs() / weighted).max())
    # The same population target does not give the same gradient on a finite network.
    grads = {}
    for mode in ("ddpm_simple", "cfm_equiv_eps"):
        net.zero_grad()
        eps_loss(net, g, k, t, xt, x0, y_in, mode).backward()
        grads[mode] = torch.cat([p.grad.flatten() for p in net.parameters() if p.grad is not None])
    cos = float(torch.nn.functional.cosine_similarity(grads["ddpm_simple"], grads["cfm_equiv_eps"], dim=0))
    w = b_coeff(g, t) ** 2
    out = {"max_rel_identity": rel, "grad_cosine": cos, "batch_t": [float(v) for v in t],
           "batch_b2": [float(v) for v in w], "n": int(t.shape[0])}
    assert rel < 1e-12, rel
    assert cos < 0.999, f"the two gradients came out parallel on this batch ({cos}); the claim needs another batch"
    print(out)
    return {"weighting": out}


def stage_ritual() -> dict:
    """The gradient-check ritual on both loss modes, plus the zero-head control."""
    g = VPGrid()
    k, t, x0, xt, y_in, _ = frozen_batch(g)
    batch = (k, t, x0, xt, y_in)
    out = {}
    for mode in ("ddpm_simple", "cfm_equiv_eps"):
        out[mode] = directional_check(tiny_net(0), g, batch, mode)
        assert out[mode]["rel_err"] < 1e-6 and abs(out[mode]["autodiff"]) > 1e-8, out[mode]
    torch.manual_seed(0)
    zero = make_unet({"model": TINY}).double()                       # head at exactly zero
    out["zero_head"] = directional_check(zero, g, batch, "ddpm_simple")
    assert out["zero_head"]["autodiff"] == 0.0, "with a zero head the upstream gradient is exactly zero"
    print(out)
    return {"ritual": out}


def stage_azula() -> dict:
    """TODO 4.1 and 4.2: Azula's DDPM and DDIM steps against the course steps, under frozen noise."""
    g = VPGrid()
    net = tiny_net(0)
    _, _, _, _, y_in, _ = frozen_batch(g)
    den = EpsDenoiser(net, g)
    x = torch.randn(8, 1, 28, 28, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    out = {"azula_version": azula.__version__ if hasattr(azula, "__version__") else "0.11.1"}
    # schedule through the adapter, at grid nodes
    sched = CourseSchedule(g)
    s_nodes = 1.0 - g.t
    a_az, s_az = sched(s_nodes)
    out["schedule_max_abs"] = float(max((a_az - g.alpha).abs().max(), (s_az - g.sigma).abs().max()))
    # one step, many pairs (j -> i)
    diffs = {"ddpm": 0.0, "ddim": 0.0}
    pairs = ((K, K - 1), (500, 499), (2, 1), (1, 0), (K, 980), (600, 400), (50, 0))
    out["n_pairs"] = len(pairs)
    for j, i in pairs:
        t_j = g.t[j].expand(8)
        eps_hat = net(x, t_j, y_in)
        torch.manual_seed(7)
        z = torch.randn_like(x)
        ours = posterior_step(g, x, eps_hat, j, i, z if i > 0 else None)
        torch.manual_seed(7)
        theirs = azula.sample.DDPMSampler(den).step(x, s_nodes[j], s_nodes[i], y=y_in)
        diffs["ddpm"] = max(diffs["ddpm"], float((ours - theirs).abs().max()))
        ours = posterior_step(g, x, eps_hat, j, i, None)
        theirs = azula.sample.DDIMSampler(den, eta=0.0).step(x, s_nodes[j], s_nodes[i], y=y_in)
        diffs["ddim"] = max(diffs["ddim"], float((ours - theirs).abs().max()))
    out["step_max_abs"] = diffs
    # a whole DDIM chain of 50 steps
    ours = sample_chain(net, g, x, y_in, 50, stochastic=False)["x"]
    samp = azula.sample.DDIMSampler(den, eta=0.0, steps=50, silent=True, dtype=torch.float64)
    theirs = samp(x, y=y_in)
    out["ddim_chain50_max_abs"] = float((ours - theirs).abs().max())
    # representation round trip on interior times
    t_mid = g.t[torch.tensor([1, 250, 500, 750, K])]
    xs = x[:5]
    e = net(xs, t_mid, y_in[:5])
    x1h = x1_from_eps(g.alpha[[1, 250, 500, 750, K]], g.sigma[[1, 250, 500, 750, K]], xs, e)
    a5, s5 = g.alpha[[1, 250, 500, 750, K]].view(-1, 1, 1, 1), g.sigma[[1, 250, 500, 750, K]].view(-1, 1, 1, 1)
    out["roundtrip_max_abs"] = float(((xs - a5 * x1h) / s5 - e).abs().max())
    for key, val in (("schedule_max_abs", out["schedule_max_abs"]), ("ddpm", diffs["ddpm"]),
                     ("ddim", diffs["ddim"]), ("ddim_chain50", out["ddim_chain50_max_abs"]),
                     ("roundtrip", out["roundtrip_max_abs"])):
        assert val < 1e-9, (key, val)
    print(out)
    return {"azula": out}


def ckpt_paths() -> tuple[Path, Path]:
    return WEIGHTS_DIR / f"{CKPT_NAME}.pt", WEIGHTS_DIR / f"{CKPT_NAME}.json"


def resolved_config() -> dict:
    cfg = copy.deepcopy(CONFIG)
    cfg["train"]["iters"] = FULL_ITERS
    cfg["train"]["lr"] = FULL_LR
    return cfg


def git_state() -> dict:
    def run(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"git_commit": run("rev-parse", "HEAD"), "git_dirty": bool(run("status", "--porcelain"))}


def stage_train() -> dict:
    """The course checkpoint: full budget, seed 0, ddpm_simple, cfg_dropout 0.1, logged to MLflow."""
    import mlflow
    cfg = resolved_config()
    chash = config_hash(cfg)
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("u3l2-course-checkpoint")
    with mlflow.start_run(run_name=f"{CKPT_NAME}-seed{cfg['seed']}") as run:
        mlflow.set_tags({"session": SESSION, "config_hash": chash, "loss_mode": cfg["train"]["loss_mode"],
                         "seed": str(cfg["seed"]), "device": torch.cuda.get_device_name(0)})
        mlflow.log_params({"lr": cfg["train"]["lr"], "iters": cfg["train"]["iters"],
                           "batch": cfg["train"]["batch"], "base_ch": cfg["model"]["base_ch"],
                           "cfg_dropout": cfg["train"]["cfg_dropout"], "K": cfg["grid"]["n_steps"]})
        net, log, wall = train_run(cfg, iters=cfg["train"]["iters"], seed=cfg["seed"],
                                   on_log=lambda r: mlflow.log_metrics({"loss": r["loss"], "lr": r["lr"]},
                                                                       step=r["step"]))
        mlflow.log_metric("wall_s", wall)
        run_id = run.info.run_id
    pt, js = ckpt_paths()
    torch.save({"state_dict": net.state_dict(), "config": cfg, "config_hash": chash}, pt)
    sha = hashlib.sha256(pt.read_bytes()).hexdigest()
    meta = {"name": CKPT_NAME, "session": SESSION, "config": cfg, "config_hash": chash, "seed": cfg["seed"],
            "weights_sha256": sha, "wall_s": wall, "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__, **git_state(), "final_loss": log[-1]["loss"],
            "selection": "final iterate (no checkpoint selection)"}
    js.write_text(json.dumps(meta, indent=2) + "\n")
    painplots.export_run(run_id, REF_DIR / "mlflow_export", tracking_uri=MLFLOW_URI)
    return {"train": {"config_hash": chash, "wall_s": wall, "iters": cfg["train"]["iters"],
                      "final_loss": log[-1]["loss"], "loss_log": log, "weights_sha256": sha,
                      "device": meta["device"], "params_M": sum(p.numel() for p in net.parameters()) / 1e6}}


def load_course_net() -> nn.Module:
    pt, js = ckpt_paths()
    payload = torch.load(pt, map_location="cpu", weights_only=False)
    net = make_unet(payload["config"])
    net.load_state_dict(payload["state_dict"])
    return net.to(device()).eval()


def stage_floor() -> dict:
    """Recompute only the probe's real-data reference, without re-running the ten probe trainings."""
    data = json.loads(JSON_OUT.read_text())
    return {"probe": {**data["probe"], **probe_floor()}}


def stage_diff() -> dict:
    """The reference output of the diff cell: cfm-scratch (U3.L1) against today's two functions."""
    import fig_u3_l1_payoff as l1
    old = inspect.getsource(l1.sample_batch) + "\n" + inspect.getsource(l1.cfm_loss)
    new = inspect.getsource(sample_batch) + "\n" + inspect.getsource(eps_loss)
    lines = thesis_diff(old, new)
    changed = [l_ for l_ in lines if l_[:1] in "+-" and l_[:3] not in ("+++", "---")]
    tags = [l_.rsplit("[", 1)[1].rstrip("]") for l_ in changed if l_.endswith("]")]
    out = {"lines": lines, "n_changed": len(changed), "n_added": sum(1 for l_ in changed if l_[0] == "+"),
           "tags": {t: tags.count(t) for t in ("path", "coordinate", "weight", "labels")}}
    print("\n".join(lines))
    return {"diff": out}


def stage_live() -> dict:
    """The small live run of Block 2 (notebook configuration): time per step on GPU and on CPU."""
    out = {}
    cfg = copy.deepcopy(CONFIG)
    cfg["model"]["base_ch"] = LIVE_BASE_CH
    cfg["train"]["lr"] = FULL_LR
    _, log, wall = train_run(cfg, iters=LIVE_ITERS, seed=0)
    out["gpu"] = {"iters": LIVE_ITERS, "wall_s": wall, "final_loss": log[-1]["loss"]}
    print(out, flush=True)
    return {"live": {"base_ch": LIVE_BASE_CH, "iters": LIVE_ITERS, **out}}


def stage_live_cpu() -> dict:
    """The CPU fallback of the live run: 50 steps, timed, extrapolated to the live budget."""
    cfg = copy.deepcopy(CONFIG)
    cfg["model"]["base_ch"] = LIVE_BASE_CH
    global device
    saved = device
    device = lambda: torch.device("cpu")  # noqa: E731
    try:
        _, _, wall = train_run(cfg, iters=50, seed=0, log_every=50)
    finally:
        device = saved
    return {"live_cpu": {"iters": 50, "wall_s": wall, "s_per_iter": wall / 50,
                         "threads": torch.get_num_threads()}}


def stage_panels() -> dict:
    """TODO 3.3: one noise bank, fixed labels, three samplers (figures for notes and deck)."""
    net, g = load_course_net(), VPGrid()
    dev = device()
    bank = torch.randn(40, 1, 28, 28, generator=torch.Generator().manual_seed(2026)).to(dev)
    y = (torch.arange(40) % 10).to(dev)
    sol, ode_x = sample_ode(net, g, bank, y, method="adaptive_heun", rtol=ODE_TOL, atol=ODE_TOL)
    anc = sample_chain(net, g, bank, y, K, stochastic=True, gen=torch.Generator().manual_seed(5))
    ddim = sample_chain(net, g, bank, y, 50, stochastic=False)
    imgs = {"ode": to_unit(ode_x).cpu(), "ancestral": to_unit(anc["x"]).cpu(), "ddim": to_unit(ddim["x"]).cpu()}
    torch.save(imgs, SCRATCH / "panels.pt")
    l2 = {"ode_vs_ddim": float((imgs["ode"] - imgs["ddim"]).flatten(1).norm(dim=1).mean()),
          "ode_vs_ancestral": float((imgs["ode"] - imgs["ancestral"]).flatten(1).norm(dim=1).mean())}
    return {"panels": {"ode": {"nfe": sol.nfe, "method": sol.method, "accepted": sol.accepted,
                               "rejected": sol.rejected, "tol": ODE_TOL},
                       "ancestral": {k_: v for k_, v in anc.items() if k_ != "x"},
                       "ddim": {k_: v for k_, v in ddim.items() if k_ != "x"}, "mean_l2": l2}}


def stage_fid() -> dict:
    """The reported numbers: FID on the TEST split, n = MNIST_FID_SAMPLES, read once here."""
    net, g = load_course_net(), VPGrid()
    n = MNIST_FID_SAMPLES
    rows = []

    def one(sampler: str, **kw) -> None:
        if device().type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        x, cost = generate(net, g, n, sampler=sampler, seed=FID_SEED, **kw)
        if device().type == "cuda":
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        row = {"sampler": sampler, **{k_: v for k_, v in kw.items()}, **cost, "wall_s": wall,
               "fid": fid_on(x, "test")}
        if sampler == "ode":
            row["nfe"] = int(np.median(cost["nfe_per_solve"]))
            row["nfe_min"], row["nfe_max"] = min(cost["nfe_per_solve"]), max(cost["nfe_per_solve"])
            row["model_forwards"] = row["nfe"]                       # one network call per RHS call
        rows.append(row)
        print(row, flush=True)
    for nfe in NFE_SWEEP:
        one("ode", method="euler", n_steps=nfe)
    one("ode", method="adaptive_heun", rtol=ODE_TOL, atol=ODE_TOL)
    for steps in NFE_SWEEP:
        one("ddim", n_steps=steps)
    for steps in ANC_STEPS:
        one("ancestral", n_steps=steps)
    x_val, _ = load_mnist("val")
    floor = fid_on(((x_val[:n] + 1) / 2), "test")                    # real val images against the test ref
    by_key = {}
    for r_ in rows:
        key = r_["sampler"] + (f"-{r_['method']}" if r_["sampler"] == "ode" else "")
        key += f"-{r_['n_steps']}" if r_.get("n_steps") else ""
        by_key[key] = r_
    REF_DIR.mkdir(parents=True, exist_ok=True)
    (REF_DIR / "fid_table.json").write_text(json.dumps(
        {"n": n, "split": "test", "extractor": extractor().name, "rows": by_key,
         "checkpoint": CKPT_NAME, "device": torch.cuda.get_device_name(0)}, indent=2) + "\n")
    return {"fid": {"n": n, "seed": FID_SEED, "rows": rows, "by_key": by_key, "real_val_vs_test": floor,
                    "extractor": extractor().name}}


UFRJ_BLUE, UFRJ_ACCENT, UFRJ_GRAY, INK = "#003366", "#d97a1f", "#6b7280", "#26262a"
FS = {"label": 16, "tick": 15, "legend": 14, "title": 16}       # read at ~0.5x on a 16:9 slide


def apply_house_style(ax, grid_axis: str = "both") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(UFRJ_GRAY)
    ax.spines["bottom"].set_color(UFRJ_GRAY)
    ax.tick_params(colors=INK, labelsize=FS["tick"])
    ax.grid(axis=grid_axis, color="#DDDDDD", linewidth=0.6)
    ax.set_axisbelow(True)


def save_fig(fig, name: str) -> None:
    for ext in ("svg", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", transparent=True, bbox_inches="tight")


def stage_figures() -> dict:
    """The three figures of the notes and the deck, from the stages above (no new measurement)."""
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["svg.fonttype"] = "path"
    import matplotlib.pyplot as plt
    data = json.loads(JSON_OUT.read_text())
    g = VPGrid()
    # (1) the schedule and the induced weight
    t = torch.linspace(0.0, float(g.t[1]), 801, dtype=torch.float64)
    alpha, sigma, _, _ = g.coeffs(t)
    b2 = b_coeff(g, t) ** 2
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    ax.plot(t, alpha, color=UFRJ_BLUE, lw=2.5, label=r"$\alpha_t$")
    ax.plot(t, sigma, color=UFRJ_ACCENT, lw=2.5, label=r"$\sigma_t$")
    ax.scatter([0.0], [float(g.alpha[K])], color=UFRJ_BLUE, zorder=5, s=40)
    ax.annotate(rf"$\alpha_K = {float(g.alpha[K]):.4f}$", (0.0, float(g.alpha[K])), xytext=(0.12, 0.18),
                fontsize=FS["legend"], color=INK, arrowprops={"arrowstyle": "->", "color": UFRJ_GRAY})
    ax.set_xlabel(r"FM time $t$ (0 noise, 1 data)", fontsize=FS["label"])
    ax.set_title("(a) the DDPM grid on the FM arrow", fontsize=FS["title"], loc="left")
    ax.legend(fontsize=FS["legend"], frameon=False, loc="center right")
    apply_house_style(ax)
    ax = axes[1]
    ax.plot(t, b2, color=UFRJ_BLUE, lw=2.5, label=r"cfm_equiv_eps: $B_t^2$")
    ax.plot(t, torch.ones_like(t), color=UFRJ_ACCENT, lw=2.5, ls="--", label="ddpm_simple: 1")
    ax.set_yscale("log")
    ax.set_xlabel(r"FM time $t$", fontsize=FS["label"])
    ax.set_title(r"(b) the weight on $\Vert\hat\varepsilon - X_0\Vert^2$", fontsize=FS["title"], loc="left")
    ax.legend(fontsize=FS["legend"], frameon=False, loc="center left")
    apply_house_style(ax)
    fig.tight_layout()
    save_fig(fig, "u3-l2-schedule")
    plt.close(fig)
    # (2) the panels: one noise bank, three samplers
    imgs = torch.load(SCRATCH / "panels.pt")
    rows = [("ODE/PF", imgs["ode"]), ("ancestral", imgs["ancestral"]), ("DDIM", imgs["ddim"])]
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 4.6))
    for ax, (name, im) in zip(axes, rows):
        strip = torch.cat(list(im[:10, 0]), dim=1).numpy()
        ax.imshow(strip, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_ylabel(name, fontsize=FS["label"], rotation=0, ha="right", va="center", color=INK)
    fig.tight_layout(h_pad=0.4)
    save_fig(fig, "u3-l2-panels")
    plt.close(fig)
    # (3) FID against model forwards. Sized to be read in Beamer at ~0.65x (QA round 1, M2): 6.4 x 4.0 in, 15-18 pt.
    rows = data["fid"]["by_key"]
    fs = {"label": 18, "tick": 16, "legend": 14}
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    series = (("ODE/PF, Euler", [rows[f"ode-euler-{n}"] for n in NFE_SWEEP], UFRJ_BLUE, "o-"),
              ("DDIM", [rows[f"ddim-{n}"] for n in NFE_SWEEP], UFRJ_ACCENT, "s--"),
              ("ancestral", [rows[f"ancestral-{n}"] for n in ANC_STEPS], UFRJ_GRAY, "^:"))
    for name, rs, col, sty in series:
        ax.plot([r_["model_forwards"] for r_ in rs], [r_["fid"] for r_ in rs], sty, color=col, lw=2.4,
                ms=8, label=name)
    ad = rows["ode-adaptive_heun"]
    ax.scatter([ad["model_forwards"]], [ad["fid"]], marker="*", s=300, color=UFRJ_BLUE, zorder=6,
               label="ODE/PF, adaptive Heun")
    floor = data["fid"]["real_val_vs_test"]
    ax.axhline(floor, color=UFRJ_GRAY, lw=1.2, ls="-.")
    ax.text(5.5, floor * 1.08, "real images", color=UFRJ_GRAY, fontsize=fs["legend"], ha="left", va="bottom")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_yticks([2, 3, 5, 10, 30, 100, 300], ["2", "3", "5", "10", "30", "100", "300"])
    ax.minorticks_off()
    ax.set_ylim(1.4, 400)
    ax.set_xlabel("model forwards per sample (ODE: NFE)", fontsize=fs["label"])
    n_txt = f"{data['fid']['n']:,}".replace(",", "\u2009")            # thin space as the thousands separator
    ax.set_ylabel(f"FID, n = {n_txt}", fontsize=fs["label"])
    ax.legend(fontsize=fs["legend"], frameon=False, loc="upper right")
    apply_house_style(ax)
    ax.tick_params(labelsize=fs["tick"])
    fig.tight_layout()
    save_fig(fig, "u3-l2-nfe-fid")
    plt.close(fig)
    return {"figures": {"written": ["u3-l2-schedule", "u3-l2-panels", "u3-l2-nfe-fid"]}}


PROBE_ITERS = 3000
PROBE_LRS = (2e-4, 1e-3)
FULL_ITERS = 20_000     # from the probe: at 3000 steps lr 1e-3 gives val FID 17-29 on all five seeds
FULL_LR = 1e-3          # from the probe: lr 2e-4 is 50x worse at 3000 steps, all five seeds
LIVE_BASE_CH = 16
LIVE_ITERS = 300
ODE_TOL = 1e-3
FID_SEED = 12
NFE_SWEEP = (5, 10, 20, 50)
ANC_STEPS = (10, 20, 50, 1000)
STAGES = ("speed", "schedule", "weighting", "ritual", "azula", "train", "live", "live_cpu",
          "panels", "fid", "figures", "probe", "diff", "floor")


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
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    for stage in STAGES:
        if stage not in only:
            continue
        print(f"=== stage {stage}", flush=True)
        t0 = time.perf_counter()
        data.update(globals()[f"stage_{stage}"]())
        data.setdefault("_wall_s", {})[stage] = time.perf_counter() - t0
        JSON_OUT.write_text(json.dumps(data, indent=2) + "\n")
        print(f"=== stage {stage} done in {data['_wall_s'][stage]:.1f} s", flush=True)


if __name__ == "__main__":
    main()
