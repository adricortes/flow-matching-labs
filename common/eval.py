"""common/eval.py — the evaluation harness (artifact: `eval-harness`).

Built in U0.L4. **The interface below is FROZEN.** U1.L2 and PS1.4 import
`bpd`, U3.L2 and U3.L3 import `fid`, `ref_stats` and `panel`, and PS0 imports
all of them. A change after the U0.L4 freeze is a breaking change and needs a
dated CHANGELOG note.

THE CONTRACT (U0.L4 notes §D; L3-U0-L4 §2A):

    FeatureExtractor                      # protocol: name, preprocessing,
                                          #   dim, __call__(x) -> (N, dim)
    MNISTFeatures                         # course classifier, pinned weights
    InceptionV3Features                   # torchvision, for CIFAR and beyond
    ref_stats(loader, extractor, dataset=, split=)   # cached to disk;
                                  -> RefStats        #   key = the five
                                                     #   comparability fields
    fid(samples, ref, extractor)  -> float
    bpd(log_prob_nats_per_example, num_dims) -> float   # PER-EXAMPLE nats
    panel(model_sample_fn, eps_bank, path)   -> figure

THE COMPARABILITY RULE, which this module enforces rather than states:

    An FID is not a property of a model. It is a property of a model AND a
    measurement protocol. Two FID values are comparable only when the
    extractor, the preprocessing and the sample count all agree.

That is caveat (i) of `fid-definition` (U0.T4). A warning in a docstring does
not survive contact with a results table, so the harness makes the number
carry its own protocol: `fid` returns a `FIDValue`, which IS a float and
behaves as one everywhere, and which additionally records the three fields
and refuses a comparison across protocols (`FIDValue.check_comparable`). The
`ref_stats` disk cache is keyed by the same triple, so a cache hit across
different preprocessing is impossible by construction.

Caveat (ii) — the estimator is biased upward and the bias falls like 1/n —
is not enforceable by code, because the bias is real and correct. It is
*measured* instead, in U0.L4 Part E, before the harness is trusted.

Caveat (iii) — the extractor decides what "similar" means — is settled for
this course: MNIST uses `MNISTFeatures`, a small classifier with pinned
weights trained by `scripts/python/train_mnist_features.py`; the ImageNet
Inception extractor is reserved for CIFAR and beyond.

TWO NUMERICAL RULES, both of which are real bug classes and neither of which
announces itself:
  * covariance accumulates in float64 (§ `ref_stats`);
  * the matrix square root returns a tiny imaginary part, which is truncated
    only after a tolerance check (§ `fid`).

In the student template the regions between SOLUTION markers are stripped to
TODO stubs (TODO 4-7 of the U0.L4 notebook); this file is the course's
reference implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, cast, runtime_checkable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "FeatureExtractor",
    "MNISTFeatures",
    "InceptionV3Features",
    "FIDValue",
    "RefStats",
    "ref_stats",
    "fid",
    "bpd",
    "panel",
]

# Course standard for MNIST, fixed here so no call site has to remember it.
# Part E measures the bias curve that justifies the choice of 10 000.
MNIST_FID_SAMPLES = 10_000


# ===========================================================================
# The feature extractor protocol
# ===========================================================================


@runtime_checkable
class FeatureExtractor(Protocol):
    """What the harness needs from a feature extractor.

    `name` and `preprocessing` are not decoration: they are two thirds of the
    comparability key, so an extractor that does not name itself honestly
    breaks the rule the harness exists to enforce.
    """

    name: str
    preprocessing: str
    dim: int

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """(N,C,H,W) in [0,1] -> (N, dim) float features."""
        ...


class _MNISTClassifier(nn.Module):
    """The course's small MNIST classifier. Weights are pinned, not trained here.

    Deliberately plain, and deliberately fixed: the whole point of a feature
    extractor is that it does not move. The penultimate layer (128 units) is
    the feature space FID is computed in.

    **BatchNorm here, GroupNorm in the U-Net, and the difference is not
    inconsistency.** The recipe card bans BatchNorm from a generative model
    because that model is sampled one example at a time, so batch statistics
    would make one sample depend on the others. A classifier used only in
    `eval()` runs on its frozen running statistics, which depend on no batch
    at all. The law is about the failure, not about the layer.
    """

    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                              # 14
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                              # 7
            nn.Flatten(),                                                 # 3136
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
        )
        self.head = nn.Linear(128, 10)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))


class MNISTFeatures:
    """The course MNIST extractor: pinned classifier, penultimate layer.

    Preprocessing is fixed and named: images arrive as (N,1,28,28) floats in
    [0,1] and are used as they are. No resize, no ImageNet normalization —
    and the name says so, because `preprocessing` is part of the key that
    decides whether two FID values may be compared.
    """

    name = "mnist-cnn-v1"
    preprocessing = "01-float-28x28"
    dim = 128

    def __init__(
        self,
        weights: str | Path | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        root = Path(__file__).resolve().parents[1]
        path = Path(weights) if weights is not None else (
            root / "common" / "weights" / "mnist_features_v1.pt"
        )
        if not path.exists():
            raise FileNotFoundError(
                f"pinned extractor weights not found at {path}. Build them "
                "with:  conda run -n torch-gpu python "
                "scripts/python/train_mnist_features.py"
            )
        payload = torch.load(path, map_location="cpu", weights_only=True)
        self.model = _MNISTClassifier()
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval().to(device)
        self.device = device
        # Provenance travels with the extractor: the number it produces is
        # only as pinned as the weights that produced it.
        self.sha256 = payload.get("sha256", "unknown")
        self.val_acc = payload.get("val_acc", float("nan"))

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.features(x.to(self.device)).float().cpu()


class InceptionV3Features:
    """The ImageNet extractor, for CIFAR and beyond. NOT exercised in U0.

    U0 evaluates MNIST only, and this class is here so the contract is
    complete when U3 moves to a larger dataset. Its weights are a download,
    which the course machine's build sandbox does not permit, so **this path
    has never been run**. It is written to the same protocol and it says so
    out loud rather than presenting itself as tested.
    """

    name = "inception-v3-pool3"
    preprocessing = "imagenet-299x299"
    dim = 2048

    def __init__(self, device: str | torch.device = "cpu") -> None:
        from torchvision.models import Inception_V3_Weights, inception_v3

        net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        net.fc = nn.Identity()
        self.model = net.eval().to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(
            x, size=(299, 299), mode="bilinear", align_corners=False
        )
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = (x - mean.to(x)) / std.to(x)
        return self.model(x.to(self.device)).float().cpu()


# ===========================================================================
# The value type — caveat (i), enforced
# ===========================================================================


class FIDValue(float):
    """An FID that carries its measurement protocol.

    It is a `float`: it prints, compares, sorts and serializes like one, so
    every call site written against the frozen `-> float` contract works. It
    additionally knows the extractor, the preprocessing and the sample count
    that produced it, and `check_comparable` refuses a comparison across a
    different protocol. That turns caveat (i) from something a student must
    remember into something the code will not let them get wrong quietly.
    """

    extractor: str
    preprocessing: str
    n_samples: int

    def __new__(
        cls, value: float, *, extractor: str, preprocessing: str, n_samples: int
    ) -> "FIDValue":
        self = super().__new__(cls, value)
        self.extractor = extractor
        self.preprocessing = preprocessing
        self.n_samples = n_samples
        return self

    @property
    def protocol(self) -> tuple[str, str, int]:
        return (self.extractor, self.preprocessing, self.n_samples)

    def check_comparable(self, other: "FIDValue") -> None:
        """Raise unless the two values were measured the same way."""
        if self.protocol != other.protocol:
            raise ValueError(
                "these two FID values are not comparable — caveat (i).\n"
                f"  this:  {self.protocol}\n"
                f"  other: {other.protocol}\n"
                "An FID is a property of a model AND a protocol. Re-measure "
                "both under one protocol before you rank them."
            )

    def __repr__(self) -> str:
        return (
            f"FID {float(self):.4f} "
            f"[{self.extractor} · {self.preprocessing} · n={self.n_samples}]"
        )


@dataclass(frozen=True)
class RefStats:
    """Reference Gaussian, with the key that says what it describes."""

    mu: np.ndarray
    sigma: np.ndarray
    extractor: str
    preprocessing: str
    n_samples: int
    dataset: str


# ===========================================================================
# TODO 4 — reference statistics
# ===========================================================================


def _cache_path(
    dataset: str,
    extractor: FeatureExtractor,
    cache_dir: Path,
    *,
    split: str = "unnamed",
    n_samples: int | None = None,
) -> Path:
    """The cache key IS caveat (i), written as a filename.

    U0.T4 caveat (i) lists five things an internally comparable Frechet
    number depends on, and all five are in this name:

        dataset + split/version, extractor NAME, extractor pinned-WEIGHT
        HASH, preprocessing signature, feature dimension, reference sample
        count.

    A key that omitted any of them would serve statistics from a different
    protocol on a cache hit — silently, permanently, and exactly as the
    caveat warns. The weight hash is the one people forget: a different
    checkpoint of the same architecture is a different feature space, under
    the same `name`.
    """
    weights = str(getattr(extractor, "sha256", "no-weights"))[:12]
    count = "all" if n_samples is None else str(n_samples)
    return cache_dir / (
        f"refstats__{dataset}--{split}__{extractor.name}--{weights}"
        f"__{extractor.preprocessing}__d{extractor.dim}__n{count}.npz"
    )


def ref_stats(
    loader: Iterable,
    extractor: FeatureExtractor,
    *,
    dataset: str = "unnamed",
    split: str = "unnamed",
    cache_dir: str | Path | None = None,
    max_samples: int | None = None,
) -> RefStats:
    """Mean and covariance of `extractor`'s features over `loader`.

    Computed once over a large reference set and cached, because the
    reference side of an FID does not change between models.

    **The accumulation runs in float64, and that is not fastidiousness.** A
    covariance is a sum of products of numbers whose magnitudes differ by
    orders of magnitude; in float32 the small contributions are lost to
    rounding against a growing sum, and the resulting matrix can fail to be
    positive semi-definite. The square root in `fid` then returns something
    with a large imaginary part, and the failure surfaces far from its cause.
    """
    cache = Path(cache_dir) if cache_dir is not None else (
        Path(__file__).resolve().parents[1] / "scripts" / "python" / "_outputs"
        / "refstats"
    )
    cache.mkdir(parents=True, exist_ok=True)
    path = _cache_path(dataset, extractor, cache,
                       split=split, n_samples=max_samples)
    if path.exists():
        z = np.load(path)
        return RefStats(
            mu=z["mu"],
            sigma=z["sigma"],
            extractor=str(z["extractor"]),
            preprocessing=str(z["preprocessing"]),
            n_samples=int(z["n_samples"]),
            dataset=str(z["dataset"]),
        )

    # --- SOLUTION: TODO 4 (stripped to a stub in the student notebook) ---
    total = 0
    s1: np.ndarray | None = None   # running sum of features
    s2: np.ndarray | None = None   # running sum of outer products
    for batch in loader:
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        if max_samples is not None and total >= max_samples:
            break
        if max_samples is not None:
            x = x[: max_samples - total]
        f = extractor(x).numpy().astype(np.float64)   # float64 from here on
        if s1 is None:
            s1 = np.zeros(f.shape[1], dtype=np.float64)
            s2 = np.zeros((f.shape[1], f.shape[1]), dtype=np.float64)
        s1 += f.sum(axis=0)
        s2 += f.T @ f
        total += f.shape[0]
    if s1 is None or s2 is None or total < 2:
        raise ValueError("ref_stats needs at least 2 samples")
    mu = s1 / total
    # Unbiased covariance from the running sums: E[ff'] - mu mu', rescaled.
    sigma = (s2 - total * np.outer(mu, mu)) / (total - 1)
    # --- END SOLUTION (TODO 4) ---

    stats = RefStats(
        mu=mu,
        sigma=sigma,
        extractor=extractor.name,
        preprocessing=extractor.preprocessing,
        n_samples=total,
        dataset=dataset,
    )
    np.savez(
        path,
        mu=stats.mu,
        sigma=stats.sigma,
        extractor=stats.extractor,
        preprocessing=stats.preprocessing,
        n_samples=stats.n_samples,
        dataset=stats.dataset,
    )
    return stats


# ===========================================================================
# TODO 5 — the Fréchet distance
# ===========================================================================


def _sqrtm_psd(a: np.ndarray, tol: float = 1e-3) -> np.ndarray:
    """Matrix square root of a PSD matrix, with the imaginary-part guard.

    `scipy.linalg.sqrtm` works on general matrices and returns a complex
    array. On a product of two covariance matrices the true root is real, but
    floating-point error puts a tiny imaginary component on it. The classic
    gotcha is to call `.real` unconditionally: that silently discards a LARGE
    imaginary part too, and a large one means the input was not PSD — a
    genuine bug upstream, usually a float32 covariance.

    So: truncate, but only after checking the size of what is truncated.
    """
    from scipy.linalg import sqrtm

    root = cast(Any, sqrtm(a))
    if np.iscomplexobj(root):
        imag = float(np.abs(root.imag).max())
        scale = float(np.abs(root.real).max()) + 1e-12
        if imag / scale > tol:
            raise ValueError(
                f"matrix square root has a large imaginary part "
                f"({imag:.3g}, {100 * imag / scale:.2g}% of the real scale). "
                "The input is not positive semi-definite — check that the "
                "covariances were accumulated in float64."
            )
        root = root.real
    return np.asarray(root, dtype=np.float64)


def fid(
    samples: torch.Tensor,
    ref: RefStats,
    extractor: FeatureExtractor,
) -> FIDValue:
    """FID between `samples` and the reference Gaussian `ref`.

    Implements U0.T4 @eq-fid:

        ||mu_r - mu_g||^2 + Tr(Sigma_r + Sigma_g - 2 (Sigma_r Sigma_g)^(1/2))

    `samples` is (N,C,H,W) in [0,1]. The returned value is a float that also
    records the protocol it was measured under; see `FIDValue`.
    """
    if ref.extractor != extractor.name or ref.preprocessing != extractor.preprocessing:
        raise ValueError(
            "the reference statistics were computed with a different "
            f"protocol ({ref.extractor} · {ref.preprocessing}) than the "
            f"extractor given ({extractor.name} · {extractor.preprocessing}). "
            "Caveat (i): re-run ref_stats with this extractor."
        )

    # --- SOLUTION: TODO 5 (stripped to a stub in the student notebook) ---
    feats = extractor(samples).numpy().astype(np.float64)
    n = feats.shape[0]
    if n < 2:
        raise ValueError("fid needs at least 2 samples")
    mu_g = feats.mean(axis=0)
    sigma_g = np.cov(feats, rowvar=False)

    diff = ref.mu - mu_g
    covmean = _sqrtm_psd(ref.sigma @ sigma_g)
    value = float(
        diff @ diff + np.trace(ref.sigma + sigma_g - 2.0 * covmean)
    )
    # --- END SOLUTION (TODO 5) ---

    return FIDValue(
        value,
        extractor=extractor.name,
        preprocessing=extractor.preprocessing,
        n_samples=n,
    )


# ===========================================================================
# TODO 6 — bits per dimension
# ===========================================================================


def bpd(log_prob_nats_per_example, num_dims: int) -> float:
    """Bits per dimension from PER-EXAMPLE log-probabilities in nats.

        BPD = - mean(log_prob_nats_per_example) / (d * log 2)

    `log_prob_nats_per_example` is a VECTOR: one log-probability per example,
    in nats. `num_dims` is d, the dimension of one example (784 for MNIST).
    Lower is better. The quantity is a compression rate: the average number
    of bits per dimension needed to encode a test example under a code built
    from the model.

    **Why a vector and not a scalar.** The earlier signature took "the total
    log likelihood", which is ambiguous between a sum over the batch and a
    mean, and the two differ by a factor of the batch size. A caller that
    read it the wrong way got a number that was wrong by that factor and
    looked plausible. Taking the per-example vector and averaging here moves
    the decision into one place where it is written down (L3-U0-L4 section
    2D, TODO 6). A scalar is still accepted, and is read as the mean.

    **The dequantization term is not here, and that is deliberate.** A
    likelihood on continuous space compared against discrete pixel data
    needs a correction term before the number means anything across papers.
    It is added when the course first *uses* BPD, in U1. Today the function
    is mechanical, and this note is the marker.
    """
    if num_dims <= 0:
        raise ValueError(f"num_dims must be positive, got {num_dims}")
    # --- SOLUTION: TODO 6 (stripped to a stub in the student notebook) ---
    values = np.atleast_1d(np.asarray(log_prob_nats_per_example, dtype=np.float64))
    if values.ndim != 1:
        raise ValueError(
            "log_prob_nats_per_example must be a vector of per-example log "
            f"probabilities (or a scalar mean), got shape {values.shape}"
        )
    return -float(values.mean()) / (num_dims * float(np.log(2.0)))
    # --- END SOLUTION (TODO 6) ---


# ===========================================================================
# TODO 7 — the sample panel
# ===========================================================================


def panel(
    model_sample_fn: Callable[[torch.Tensor], torch.Tensor],
    eps_bank: torch.Tensor,
    path: str | Path | None = None,
    *,
    nrow: int = 8,
    title: str = "",
):
    """A fixed-noise sample grid, the U0.L3 trick made official.

    `eps_bank` is a FIXED bank of noise drawn once and reused at every
    evaluation, so two panels from two checkpoints differ because the model
    changed and not because the noise did. Without it a visual comparison
    across epochs is a comparison of two unrelated draws, and small real
    changes are invisible under the sampling noise.

    Returns the matplotlib figure; writes `path` when one is given.
    """
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    # --- SOLUTION: TODO 7 (stripped to a stub in the student notebook) ---
    with torch.no_grad():
        imgs = model_sample_fn(eps_bank).detach().cpu().float()
    imgs = imgs.clamp(0.0, 1.0)
    n, c, h, w = imgs.shape
    ncol = nrow
    nrows = (n + ncol - 1) // ncol
    canvas = np.ones((nrows * h, ncol * w), dtype=np.float32)
    for i in range(n):
        r, col = divmod(i, ncol)
        canvas[r * h : (r + 1) * h, col * w : (col + 1) * w] = (
            imgs[i].mean(dim=0).numpy() if c > 1 else imgs[i, 0].numpy()
        )

    fig, ax = plt.subplots(figsize=(ncol * 0.6, nrows * 0.6))
    ax = cast(Any, ax)
    ax.imshow(canvas, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)
    fig.tight_layout(pad=0.2)
    # --- END SOLUTION (TODO 7) ---

    if path is not None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=200, transparent=True, bbox_inches="tight")
    return fig
