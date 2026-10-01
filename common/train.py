"""common/train.py — the training-loop template, v2 (artifact: hygiene-stack).

v1 built in U0.L1 (training-loop block; artifact: training-loop-template).
v2 built in U0.L2 (hygiene lab): MLflow tracking, LR schedule support,
checkpoint discipline (last/best + resume), dataloader worker seeding.

THE INTERFACE IS PRESERVED — v2 adds only keyword-only optional arguments,
so every v1 call site (including the U0.L1 notebook) still runs unchanged.
This is the contract discipline of every well-maintained library: extend in
place, never rewrite. (U0.L2 TODO A.1 wires a config INTO this interface;
this file never reads a config itself.)

Contract v2 (U0.L2 notes §A-§C; v1 contract unchanged underneath):
    set_seed(seed)                          # + seed_worker for DataLoaders
    class Trainer:
      __init__(model, opt, loss_fn, device, ckpt_dir, log_every,
               *, sched=None, tracker=None, config=None)
      fit(train_loader, val_loader, epochs)
          # per step:  forward -> loss -> zero_grad -> backward -> step
          #            -> sched.step()          (stepped per STEP: TODO C.1)
          #            logs train loss + lr     (TODO B.1)
          # per epoch: val loss/acc, grad norm (global L2), weight norm,
          #            throughput (img/s)       (TODO B.1)
          #            last.pt always; best.pt on val-loss improvement
      evaluate(loader)                          # runs under no_grad
      save_ckpt(tag) / load_ckpt(tag)           # v2 payload: model, opt,
                                                #   sched, epoch, config,
                                                #   mlflow_run_id
      resume_from_ckpt(tag)                     # TODO C.2 + round-trip assert
    CSV/stdout logger KEPT alongside MLflow: the run directory must stand
    alone without the dashboard (ml-experiment conventions §5).

In the student template repo the regions between SOLUTION markers are
stripped to TODO stubs; this file is the course's reference implementation.
"""

from __future__ import annotations

import csv
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    """Seed all three RNG streams (python, numpy, torch incl. CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn: derive per-worker seeds from torch's.

    Without this (and an explicit generator on the DataLoader), the SAME
    config run with num_workers 0 vs 4 produces different batch orders —
    reproducibility that depends on a performance setting is not
    reproducibility (U0.L2 notes §D).
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def global_grad_norm(model: nn.Module) -> float:
    """Global L2 norm over all parameter gradients (checklist sensor #3)."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().norm()) ** 2
    return total**0.5


def global_weight_norm(model: nn.Module) -> float:
    """Global L2 norm over all parameters (checklist sensor #4)."""
    return float(
        torch.sqrt(sum(p.detach().norm() ** 2 for p in model.parameters()))
    )


class Trainer:
    """The canonical training loop: frozen at U0.L1, extended at U0.L2.

    v1 requirements (unchanged): seeding is the caller's job via set_seed;
    batches are moved to the device INSIDE the loop; evaluation runs under
    no_grad; checkpoints round-trip exactly.

    v2 additions are keyword-only and default to off — a v1 call site gets
    exactly v1 behavior.
    """

    def __init__(
        self,
        model: nn.Module,
        opt: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: str | torch.device,
        ckpt_dir: str | Path = "runs",
        log_every: int = 10,
        *,
        sched: torch.optim.lr_scheduler.LRScheduler | None = None,
        tracker: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.model = model.to(device)
        self.opt = opt
        self.loss_fn = loss_fn
        self.device = device
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_every = log_every
        self.sched = sched
        self.config = config
        self.epoch = 0                  # advances in fit; stored in ckpts
        self.best_val_loss = float("inf")

        if tracker not in (None, "mlflow"):
            raise ValueError(f"unknown tracker: {tracker!r}")  # fail fast
        self.tracker = tracker
        self.mlflow_run_id: str | None = None
        if tracker == "mlflow":
            import mlflow  # optional dependency behind an explicit guard

            self._mlflow = mlflow
            active = mlflow.active_run()
            if active is None:
                raise RuntimeError(
                    "tracker='mlflow' requires an active run — wrap fit in "
                    "`with mlflow.start_run(...):` (U0.L2 notes §B)."
                )
            self.mlflow_run_id = active.info.run_id

    # ------------------------------------------------------------------
    # fit — v1: notebook TODO E.2 · v2: U0.L2 TODO B.1 (tracking part)
    #       and TODO C.1 (schedule part)
    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int,
    ) -> dict[str, list[float]]:
        """Train for `epochs` epochs; return the full history.

        The five-step ostinato per batch:
        forward -> loss -> zero_grad -> backward -> step (-> sched.step()).
        """
        history: dict[str, list[float]] = {
            "train_loss": [], "val_loss": [], "val_acc": [],
        }
        log_path = self.ckpt_dir / "log.csv"
        # --- SOLUTION (stripped to a TODO stub in the student template) ---
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "train_loss", "val_loss", "val_acc",
                 "grad_norm", "weight_norm", "img_per_s"]
            )

        step = self.epoch * len(train_loader)  # continues correctly on resume
        # The epoch this call ends on, fixed BEFORE the loop: the print
        # condition below used to compare against self.epoch, which the loop
        # sets to the current epoch first, so every epoch printed and
        # log_every never applied (found at the U1.L1 build, 2026-09-15).
        final_epoch = self.epoch + epochs
        for epoch in range(self.epoch + 1, final_epoch + 1):
            self.model.train()
            batch_losses: list[float] = []
            grad_norms: list[float] = []
            n_images = 0
            t0 = time.perf_counter()
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                loss = self.loss_fn(self.model(x), y)  # forward
                self.opt.zero_grad()                   # clear
                loss.backward()                        # VJP sweep
                grad_norms.append(global_grad_norm(self.model))
                self.opt.step()                        # update
                # --- SOLUTION: TODO C.1 (schedule part) — per STEP, never
                # --- per epoch; the logged LR curve is the test.
                if self.sched is not None:
                    self.sched.step()
                # --- END SOLUTION (C.1) ---
                batch_losses.append(float(loss.detach()))
                n_images += len(x)
                # --- SOLUTION: TODO B.1 (tracking part), per-step cadence.
                # --- Log the LR the optimizer actually USED (param group),
                # --- not what the config asked for.
                if self.tracker == "mlflow":
                    self._mlflow.log_metric(
                        "train_loss", batch_losses[-1], step=step)
                    self._mlflow.log_metric(
                        "lr", self.opt.param_groups[0]["lr"], step=step)
                # --- END SOLUTION (B.1 per-step) ---
                step += 1

            train_loss = float(np.mean(batch_losses))
            val = self.evaluate(val_loader)            # no_grad
            grad_norm = float(np.mean(grad_norms))
            weight_norm = global_weight_norm(self.model)
            img_per_s = n_images / (time.perf_counter() - t0)

            self.epoch = epoch
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val["loss"])
            history["val_acc"].append(val["acc"])
            # --- SOLUTION: TODO B.1 (tracking part), per-epoch cadence ---
            if self.tracker == "mlflow":
                self._mlflow.log_metrics(
                    {"val_loss": val["loss"], "val_acc": val["acc"],
                     "grad_norm": grad_norm, "weight_norm": weight_norm,
                     "img_per_s": img_per_s},
                    step=step,
                )
            # --- END SOLUTION (B.1 per-epoch) ---

            # Checkpoint discipline (U0.L2 notes §C): last.pt every epoch,
            # unconditionally (the resume point); best.pt only on val-loss
            # improvement (the deliverable; selection on val, never test).
            self.save_ckpt("last")
            if val["loss"] < self.best_val_loss:
                self.best_val_loss = val["loss"]
                self.save_ckpt("best")

            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [epoch, f"{train_loss:.6f}", f"{val['loss']:.6f}",
                     f"{val['acc']:.4f}", f"{grad_norm:.4f}",
                     f"{weight_norm:.2f}", f"{img_per_s:.0f}"]
                )
            if epoch % self.log_every == 0 or epoch == final_epoch:
                print(
                    f"epoch {epoch:4d} | train {train_loss:.4f} "
                    f"| val {val['loss']:.4f} | val acc {val['acc']:.3f} "
                    f"| {img_per_s:.0f} img/s"
                )
        # --- END SOLUTION ---
        return history

    # ------------------------------------------------------------------
    # evaluate — v1: notebook TODO E.3 (unchanged in v2)
    # ------------------------------------------------------------------
    @torch.no_grad()  # no backward will ever be called -> build no graph
    def evaluate(self, loader: torch.utils.data.DataLoader) -> dict[str, float]:
        """Mean loss (and accuracy, when labels are integer classes)."""
        self.model.eval()
        # --- SOLUTION (stripped to a TODO stub in the student template) ---
        total_loss, correct, n = 0.0, 0, 0
        classification = False
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            out = self.model(x)
            total_loss += float(self.loss_fn(out, y)) * len(x)
            n += len(x)
            if out.ndim == 2 and not torch.is_floating_point(y):
                classification = True
                correct += int((out.argmax(dim=1) == y).sum())
        acc = correct / n if classification else float("nan")
        return {"loss": total_loss / n, "acc": acc}
        # --- END SOLUTION ---

    # ------------------------------------------------------------------
    # checkpoints — v1: notebook TODO E.4 · v2 payload per the U0.L2
    #               contract box; resume is TODO C.2 (checkpoint part)
    # ------------------------------------------------------------------
    def save_ckpt(self, tag: str) -> Path:
        """Save the v2 checkpoint contract (plain tensors, portable).

        A checkpoint that cannot RESUME is just a weights file — hence the
        optimizer state (Adam's moments are unreconstructable), the
        scheduler state (or the LR restarts into warmup), the epoch, the
        config that produced the run, and the MLflow run id tying the file
        back to its curves.
        """
        path = self.ckpt_dir / f"{tag}.pt"
        # --- SOLUTION (stripped to a TODO stub in the student template) ---
        torch.save(
            {"model": self.model.state_dict(),
             "opt": self.opt.state_dict(),
             "sched": self.sched.state_dict() if self.sched else None,
             "epoch": self.epoch,
             "config": self.config,
             "mlflow_run_id": self.mlflow_run_id,
             "val_loss": self.best_val_loss if tag == "best" else None},
            path,
        )
        # --- END SOLUTION ---
        return path

    def load_ckpt(self, tag: str) -> None:
        """v1 behavior, kept verbatim: load model + optimizer states."""
        payload = torch.load(
            self.ckpt_dir / f"{tag}.pt", map_location=self.device,
            weights_only=False,
        )
        self.model.load_state_dict(payload["model"])
        self.opt.load_state_dict(payload["opt"])

    def resume_from_ckpt(self, tag: str) -> None:
        """Restore the FULL v2 payload so training continues the same run.

        Prove it with the round-trip assert (U0.L2 notes §C): train one
        epoch -> save -> load into a fresh Trainer -> evaluate -> the val
        loss must come back identical (|delta| < 1e-6, fixed seed, same
        machine). `python train_cifar.py --selftest-resume` runs exactly
        that gate.
        """
        # --- SOLUTION: TODO C.2 (checkpoint part) --------------------------
        # ORDER IS LOAD-BEARING. The scheduler must exist BEFORE the
        # optimizer state is loaded: LambdaLR construction runs step 0 and
        # overwrites the param-group LR with f(0) (warmup start, typically
        # 0); loading the optimizer state afterwards restores the true LR,
        # then the scheduler state restores the step counter. Constructed
        # the other way round, your first resumed batch silently trains at
        # LR 0. (Found empirically while animating this lab — the caller
        # constructs opt and sched, so here we only need the load order.)
        payload = torch.load(
            self.ckpt_dir / f"{tag}.pt", map_location=self.device,
            weights_only=False,
        )
        self.model.load_state_dict(payload["model"])
        self.opt.load_state_dict(payload["opt"])
        if self.sched is not None and payload["sched"] is not None:
            self.sched.load_state_dict(payload["sched"])
        self.epoch = payload["epoch"]
        # --- END SOLUTION (C.2) ---
