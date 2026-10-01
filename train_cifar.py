"""train_cifar.py — the U0.L2 lab launcher (one run = one config = one MLflow run).

Launch (U0.L2 notes §A):
    python train_cifar.py --config configs/baseline.yaml

Gates:
    python train_cifar.py --config configs/smoke.yaml                  # short run
    python train_cifar.py --config configs/smoke.yaml --selftest-resume
        # TODO C.2's round-trip assert: train 1 epoch -> save -> load into
        # a fresh Trainer -> evaluate -> val loss identical (<1e-6)

TODO map (ids match the U0.L2 notes and deck):
    A.1 (config part)     — load_config + build(cfg), here
    B.1 (tracking part)   — log_metric cadences, in common/train.py fit
    C.1 (schedule part)   — sched stepped per STEP, in common/train.py fit
    C.2 (checkpoint part) — resume_from_ckpt, in common/train.py
    D.1 (repro part)      — no code: open the MLflow UI, look for
                            mlflow.source.git.commit among your run's tags.
                            You will not find it (script runs get none) —
                            the explicit git_commit tag below is the ONLY
                            record, which is why it is unconditional.

In the student template repo the regions between SOLUTION markers are
stripped to TODO stubs; this file is the course's reference implementation.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from common.train import Trainer, seed_worker, set_seed

TRACKING_URI = "sqlite:///mlflow.db"     # the course store (U0.L2 notes §B)
EXPERIMENT = "fbgm-2026"                 # ONE experiment per course phase

# Standard CIFAR-10 per-channel statistics (train split).
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


# =========================================================================
# Config — the whole experiment, as data (TODO A.1, config part)
# =========================================================================

@dataclass
class DataCfg:
    dataset: str = "cifar10"
    root: str = "./data"
    normalize: bool = True
    batch_size: int = 256


@dataclass
class ModelCfg:
    arch: str = "resnet9"
    norm: str = "batch"          # classifier => BN allowed (U0.T2 law)


@dataclass
class OptCfg:
    name: str = "adamw"
    lr: float = 3.0e-4
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.01


@dataclass
class SchedCfg:
    name: str = "warmup_cosine"
    warmup_frac: float = 0.03


@dataclass
class Config:
    run_name: str = "U0L2-baseline-s0"
    seed: int = 0
    epochs: int = 8
    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    opt: OptCfg = field(default_factory=OptCfg)
    sched: SchedCfg = field(default_factory=SchedCfg)

    @property
    def tag(self) -> str:
        """The middle segment of {session}-{tag}-s{seed} run naming."""
        parts = self.run_name.split("-")
        if len(parts) < 3 or not parts[-1].startswith("s"):
            raise ValueError(                      # validate config at startup
                f"run_name {self.run_name!r} must follow "
                "{{session}}-{{tag}}-s{{seed}} (e.g. U0L2-baseline-s0)"
            )
        return "-".join(parts[1:-1])


def load_config(path: str | Path) -> Config:
    """yaml -> typed Config; unknown keys fail loudly (no silent typos)."""
    # --- SOLUTION: TODO A.1 (config part) ------------------------------
    raw = yaml.safe_load(Path(path).read_text())
    sections = {"data": DataCfg, "model": ModelCfg,
                "opt": OptCfg, "sched": SchedCfg}
    kwargs: dict = {}
    for key, value in raw.items():
        if key in sections:
            if not isinstance(value, dict):
                raise TypeError(f"config section {key!r} must be a mapping")
            if key == "opt" and "betas" in value:
                value["betas"] = tuple(value["betas"])
            kwargs[key] = sections[key](**value)   # unknown keys -> TypeError
        elif key in ("run_name", "seed", "epochs"):
            kwargs[key] = value
        else:
            raise KeyError(f"unknown config key: {key!r}")
    cfg = Config(**kwargs)
    _ = cfg.tag  # validate run naming now, not after 40 minutes of training
    return cfg
    # --- END SOLUTION (A.1 load) ---


def flatten(cfg: Config) -> dict[str, object]:
    """Nested config -> dotted keys for mlflow.log_params."""
    flat: dict[str, object] = {}
    for key, value in asdict(cfg).items():
        if isinstance(value, dict):
            flat.update({f"{key}.{k}": v for k, v in value.items()})
        else:
            flat[key] = value
    return flat


# =========================================================================
# Model — given (a small ResNet; ~9 weight layers, BN per the config)
# =========================================================================

def conv_block(c_in: int, c_out: int, pool: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class Residual(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.a, self.b = conv_block(c, c), conv_block(c, c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.b(self.a(x))


class ResNet9(nn.Module):
    """The lab classifier (same architecture as the session's prep runs)."""

    def __init__(self, n_classes: int = 10) -> None:
        super().__init__()
        self.stem = conv_block(3, 64)
        self.body = nn.Sequential(
            conv_block(64, 128, pool=True), Residual(128),
            conv_block(128, 256, pool=True),
            conv_block(256, 512, pool=True), Residual(512),
        )
        self.head = nn.Sequential(
            nn.AdaptiveMaxPool2d(1), nn.Flatten(), nn.Linear(512, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(self.stem(x)))


# =========================================================================
# Build — config -> model/opt/sched/loaders (TODO A.1, config part)
# =========================================================================

def make_loaders(
    cfg: DataCfg, seed: int, workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """CIFAR-10 loaders, worker-seeded (U0.L2 notes §D)."""
    if cfg.dataset != "cifar10":
        raise ValueError(f"unknown dataset: {cfg.dataset!r}")  # fail fast
    tfs: list = [transforms.ToTensor()]
    if cfg.normalize:
        tfs.append(transforms.Normalize(CIFAR_MEAN, CIFAR_STD))
    tf = transforms.Compose(tfs)
    train = datasets.CIFAR10(cfg.root, train=True, download=True, transform=tf)
    val = datasets.CIFAR10(cfg.root, train=False, download=True, transform=tf)
    train_loader = DataLoader(
        train, batch_size=cfg.batch_size, shuffle=True, num_workers=workers,
        generator=torch.Generator().manual_seed(seed),
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val, batch_size=cfg.batch_size, shuffle=False, num_workers=workers,
    )
    return train_loader, val_loader


def build(cfg: Config, total_steps: int) -> tuple[nn.Module, torch.optim.Optimizer, LambdaLR]:
    """The factory: reads cfg, returns model/opt/sched. No globals."""
    # --- SOLUTION: TODO A.1 (config part) ------------------------------
    if cfg.model.arch != "resnet9":
        raise ValueError(f"unknown arch: {cfg.model.arch!r}")
    if cfg.opt.name != "adamw" or cfg.sched.name != "warmup_cosine":
        raise ValueError("the recipe card is the default; deviations are "
                         "config-visible AND justified in writing (U0.T2)")
    model = ResNet9()

    # Recipe card clause 1: norm parameters (and biases) are EXCLUDED from
    # weight decay — shrinking a scale parameter toward zero is sabotage.
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.opt.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.opt.lr, betas=cfg.opt.betas,
    )

    warmup_steps = max(1, int(cfg.sched.warmup_frac * total_steps))

    def warmup_cosine(step: int) -> float:       # multiplier on base LR
        if step < warmup_steps:
            return step / warmup_steps
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * p))

    sched = LambdaLR(opt, warmup_cosine)
    return model, opt, sched
    # --- END SOLUTION (A.1 build) ---


# =========================================================================
# Provenance — the run directory stands alone (conventions §2)
# =========================================================================

def git_commit_and_dirty() -> tuple[str, bool]:
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"]).decode().strip()
    dirty = bool(subprocess.run(
        ["git", "diff", "--quiet", "HEAD"]).returncode)
    return sha, dirty


# =========================================================================
# Main
# =========================================================================

def run(cfg: Config, config_path: Path, epochs: int | None = None) -> Trainer:
    """One tracked training run under the full hygiene stack."""
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader = make_loaders(cfg.data, cfg.seed)
    n_epochs = epochs if epochs is not None else cfg.epochs
    total_steps = n_epochs * len(train_loader)
    model, opt, sched = build(cfg, total_steps)

    run_dir = Path("runs") / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, run_dir / "config.yaml")   # the RESOLVED config

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=cfg.run_name):
        mlflow.log_params(flatten(cfg))                 # config, at start
        sha, dirty = git_commit_and_dirty()
        mlflow.set_tags({
            "session": "U0.L2", "tag": cfg.tag, "seed": cfg.seed,
            "git_commit": sha, "git_dirty": str(dirty).lower(),
        })
        trainer = Trainer(
            model, opt, nn.CrossEntropyLoss(), device,
            ckpt_dir=run_dir, log_every=1,
            sched=sched, tracker="mlflow", config=asdict(cfg),
        )
        trainer.fit(train_loader, val_loader, n_epochs)
    return trainer


def selftest_resume(cfg: Config, config_path: Path) -> None:
    """TODO C.2's gate: the checkpoint round trip must be exact."""
    # --- SOLUTION: TODO C.2 gate (checkpoint part) ----------------------
    trainer = run(cfg, config_path, epochs=1)
    device = trainer.device
    _, val_loader = make_loaders(cfg.data, cfg.seed)
    val_before = trainer.evaluate(val_loader)["loss"]

    # A FRESH process in miniature: new model/opt/sched, then restore.
    # Scheduler exists BEFORE optimizer state loads (order is load-bearing
    # — see resume_from_ckpt).
    model, opt, sched = build(cfg, total_steps=1 * 10**9)
    fresh = Trainer(
        model, opt, nn.CrossEntropyLoss(), device,
        ckpt_dir=Path("runs") / cfg.run_name, sched=sched,
        config=asdict(cfg),
    )
    fresh.resume_from_ckpt("last")
    val_after = fresh.evaluate(val_loader)["loss"]

    delta = abs(val_before - val_after)
    assert delta < 1e-6, f"round trip broke: |delta| = {delta:.3e}"
    print(f"resume round-trip PASS: |delta| = {delta:.3e} "
          f"(epoch {fresh.epoch} restored)")
    # --- END SOLUTION (C.2 gate) ---


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="configs/<run>.yaml")
    parser.add_argument("--selftest-resume", action="store_true",
                        help="run TODO C.2's round-trip assert and exit")
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = load_config(config_path)
    if args.selftest_resume:
        selftest_resume(cfg, config_path)
    else:
        run(cfg, config_path)


if __name__ == "__main__":
    main()
