"""common/painplots.py — the measurement helper (artifact: `painplots-helper`, v1.1).

Built in U2.L2. **THE CONTRACT BELOW IS FROZEN.** U3.L1 reads the evidence
directories this module writes and re-plots them from the CSV files, on shared
axes, beside a run of its own. A change after
the U2.L2 freeze is a breaking change and needs a dated CHANGELOG note here.

DESIGN RULE. **The CSV schema is the contract; the PNG is a courtesy.** The PNGs
autoscale and may look different on every machine. Nothing downstream reads them.

THE DIRECTORY LAYOUT (one per dataset)::

    pain_evidence/<dataset>/
        pain_wallclock.csv    pain_wallclock.png
        pain_nfe_growth.csv   pain_nfe_growth.png
        pain_tolerance.csv    pain_tolerance.png
        meta.json
    pain_evidence/pair.json   # minted once per machine, see `mint_pair_id`
    (a `resolved_config.yaml` may sit beside the files: a record, not part of the contract)

    <dataset> is "two-moons" or "checkerboard".

THE THREE SCHEMAS (columns, in this order)::

    pain_wallclock.csv    iter, wallclock_s_per_iter, cum_wallclock_s
    pain_nfe_growth.csv   iter, nfe_fwd, nfe_bwd
    pain_tolerance.csv    rtol, atol, nfe_eval, nll_test, wallclock_eval_s

ROW SEMANTICS (v1.1 states what v1.0 left open).

* A row is written every `LOG_EVERY` = 10 iterations. The row `iter = k`
  summarizes iterations k-9, ..., k: `wallclock_s_per_iter`, `nfe_fwd` and
  `nfe_bwd` are the MEANS over that window.
* `cum_wallclock_s` is the running sum of the step times up to and including
  iteration k. A step time is the solve, the backward pass and the optimizer
  update. Logging and evaluation are not in it.
* Time is in seconds. NFE is the number of calls of the vector field made by the
  solver. A Hutchinson vector-Jacobian product made INSIDE one call is not
  another NFE; its cost shows in the wall-clock column, and `divergence_mode` records
  which estimator the run used.
* `nfe_fwd` is the primal (likelihood) solve and `nfe_bwd` is the adjoint
  (sensitivity) solve. The names are historical, kept for downstream
  compatibility; they do not name a direction in time.

`meta.json` carries exactly these thirteen fields::

    session_id, dataset, seed, config_hash, machine_id, device_name,
    evidence_origin, pair_id, torch_version, timestamp, budget_version,
    painplots_version, divergence_mode

* `evidence_origin` is one of the literal strings "student" and
  "reference_pair". U3.L1's loader matches on them.
* `machine_id` is a pseudonymous, stable identifier of the computer (a salted
  hash, 12 hex digits). `hardware=cpu|gpu` alone is NOT a pairing key.
* THE PAIRING KEY is `machine_id + pair_id + budget_version`. A comparison of
  wall-clock time across two runs is valid only for runs that share it. Class
  wide claims use within-pair ratios.
* Reference evidence (`evidence_origin == "reference_pair"`) carries a
  `machine_id` and a `pair_id` that start with "ref-". Student evidence never
  does. The validator checks both directions.
* `budget_version` is `BUDGET_VERSION` for the frozen budget and
  `BUDGET_VERSION_RESCUE` for the rescue budget.

Run the self-test:  conda run -n torch-gpu python common/painplots.py

CHANGELOG
    v1.1 (U2.L2, first release). The PNG helper draws on its own `Figure` and never switches the matplotlib
    backend. Schemas of v1.0 unchanged. Added: the pairing
    metadata (`machine_id`, `pair_id`, `evidence_origin`, `painplots_version`,
    `budget_version`), the window semantics above, and the NFE definition
    (calls of the field, not calls plus products).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import secrets
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

PAINPLOTS_VERSION = "1.1"
SESSION_ID = "U2.L2"
LOG_EVERY = 10

BUDGET_VERSION = "U2L2-B1"          # MLP 3x64, 3000 iterations
BUDGET_VERSION_RESCUE = "U2L2-B1R"  # MLP 3x32, 1500 iterations
BUDGET_VERSIONS = (BUDGET_VERSION, BUDGET_VERSION_RESCUE)

DATASET_NAMES = ("two-moons", "checkerboard")
EVIDENCE_ORIGINS = ("student", "reference_pair")
DIVERGENCE_MODES = ("hutchinson_rademacher", "exact")
REFERENCE_PREFIX = "ref-"
REFERENCE_PAIR_ID = "ref-U2L2-B1"

SCHEMAS: dict[str, tuple[str, ...]] = {
    "pain_wallclock": ("iter", "wallclock_s_per_iter", "cum_wallclock_s"),
    "pain_nfe_growth": ("iter", "nfe_fwd", "nfe_bwd"),
    "pain_tolerance": ("rtol", "atol", "nfe_eval", "nll_test", "wallclock_eval_s"),
}
META_FIELDS: tuple[str, ...] = (
    "session_id", "dataset", "seed", "config_hash", "machine_id", "device_name",
    "evidence_origin", "pair_id", "torch_version", "timestamp", "budget_version",
    "painplots_version", "divergence_mode",
)

_SALT = "flow-matching-ufrj|u2l2|machine"
_UFRJ_BLUE, _UFRJ_ORANGE = "#003366", "#D97A1F"


# --- identity ---------------------------------------------------------------

def _hardware_token() -> str:
    """The installation id of Linux when readable, else the hardware address."""
    try:
        text = (Path(os.sep) / "etc" / "machine-id").read_text().strip()
        if text:
            return text
    except OSError:
        pass
    return f"{uuid.getnode():012x}"


def machine_id(reference: bool = False) -> str:
    """A pseudonymous, stable identifier of this computer: 12 hex digits.

    A salted hash of the host name and a stable identifier of the computer: the
    installation id of Linux (`machine-id`) when it can be read, else the
    hardware address (which some virtual machines change at every start). It is
    stable across runs on one machine. It is a label, not a secret: whoever
    knows those two values can recompute it.
    With `reference=True` the prefix "ref-" marks the course's own reference
    machine (the instructor's host, where the reference evidence was measured).
    """
    raw = f"{_SALT}|{platform.node()}|{_hardware_token()}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:12]
    return (REFERENCE_PREFIX if reference else "") + digest


def hardware_id() -> str:
    """The machine id without the host name: a network that renames the computer does not change it."""
    return hashlib.sha256(f"{_SALT}|{_hardware_token()}".encode()).hexdigest()[:12]


def device_name(device: str = "cpu") -> str:
    """The name of the device the run USED: the accelerator for "cuda", else the CPU model.

    The argument is what the run declares, never what the machine happens to have: a
    computer with a GPU that trains on its CPU is a CPU run.
    """
    if str(device).startswith("cuda"):
        try:
            import torch
            return torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001 - a missing torch must not stop a log
            pass
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True, check=True,
                             env={**os.environ, "LC_ALL": "C"}).stdout      # English field names, in any locale
        for line in out.splitlines():
            if line.startswith("Model name"):
                return line.split(":", 1)[1].strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return platform.processor() or platform.machine()


def mint_pair_id(root: str | Path = "pain_evidence", *, reference: bool = False) -> str:
    """The id that ties the U2.L2 evidence to the U3.L1 evidence of one machine.

    Minted once, here, and stored in `<root>/pair.json`; U3.L1 reads the same
    file and reuses it. The file records the machine it was minted on; a copy
    carried to another computer mints a new id (the pair is a property of the
    machine, not of the folder).
    """
    if reference:
        return REFERENCE_PAIR_ID
    path = Path(root) / "pair.json"
    here = machine_id()
    if path.exists():
        data = json.loads(path.read_text())
        if data.get("machine_id") == here and data.get("pair_id"):
            return data["pair_id"]
    if path.exists():
        print(f"pair.json was minted on another machine id: minting a new pair_id (the machine id is {here})")
    path.parent.mkdir(parents=True, exist_ok=True)
    pair = secrets.token_hex(4)
    path.write_text(json.dumps({"pair_id": pair, "machine_id": here}, indent=2) + "\n")
    return pair


def config_hash(config: dict) -> str:
    """Hash of the RESOLVED config (sorted keys), 12 hex digits."""
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


# --- the training log -------------------------------------------------------

class TrainLog:
    """Per-iteration instrumentation, from iteration 0.

    Call `add(step_s, nfe_fwd, nfe_bwd, loss)` once per iteration (iterations
    are counted from 1). `rows()` returns the windowed rows of the contract.
    """

    def __init__(self) -> None:
        self.step_s: list[float] = []
        self.nfe_fwd: list[float] = []
        self.nfe_bwd: list[float] = []
        self.loss: list[float] = []

    def add(self, step_s: float, nfe_fwd: float, nfe_bwd: float, loss: float) -> None:
        self.step_s.append(float(step_s))
        self.nfe_fwd.append(float(nfe_fwd))
        self.nfe_bwd.append(float(nfe_bwd))
        self.loss.append(float(loss))

    def __len__(self) -> int:
        return len(self.step_s)

    def rows(self) -> tuple[list[dict], list[dict]]:
        """(wallclock rows, nfe rows), one per window of LOG_EVERY iterations."""
        n = len(self)
        if n == 0 or n % LOG_EVERY:
            raise ValueError(f"{n} iterations logged; the contract needs a multiple of {LOG_EVERY}")
        wall, nfe, cum = [], [], 0.0
        for k in range(LOG_EVERY, n + 1, LOG_EVERY):
            w = slice(k - LOG_EVERY, k)
            cum = sum(self.step_s[:k])
            mean = lambda v: sum(v[w]) / LOG_EVERY  # noqa: E731
            wall.append({"iter": k, "wallclock_s_per_iter": mean(self.step_s), "cum_wallclock_s": cum})
            nfe.append({"iter": k, "nfe_fwd": mean(self.nfe_fwd), "nfe_bwd": mean(self.nfe_bwd)})
        return wall, nfe


# --- writers ----------------------------------------------------------------

def _fmt(x) -> str:
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return str(x)
    return f"{float(x):.9g}"


def _write_csv(path: Path, name: str, rows: Iterable[dict]) -> None:
    cols = SCHEMAS[name]
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([_fmt(r[c]) for c in cols])


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="") as fh:
        rows = list(csv.reader(fh))
    return (rows[0], rows[1:]) if rows else ([], [])


def _plot(path: Path, draw) -> None:
    """Draw one courtesy PNG WITHOUT pyplot and WITHOUT touching the backend.

    The first version called `matplotlib.use("Agg")` here: after the first `painplots` call every later
    `plt.show()` of the notebook drew nothing. A `Figure` of its own has no side effect."""
    import matplotlib
    from matplotlib.figure import Figure
    with matplotlib.rc_context({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10}):
        fig = Figure(figsize=(5.2, 3.4), dpi=150)
        ax = fig.subplots()
        draw(ax)
        fig.tight_layout()
        fig.savefig(path)


def pain_wallclock(log: TrainLog, out_dir: str | Path) -> Path:
    """Write `pain_wallclock.csv` (the contract) and `.png` (a courtesy)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wall, _ = log.rows()
    _write_csv(out / "pain_wallclock.csv", "pain_wallclock", wall)

    def draw(ax):
        ax.plot([r["iter"] for r in wall], [r["wallclock_s_per_iter"] for r in wall], color=_UFRJ_BLUE)
        ax.set_xlabel("training iteration")
        ax.set_ylabel("wall-clock seconds per iteration")
    _plot(out / "pain_wallclock.png", draw)
    return out / "pain_wallclock.csv"


def pain_nfe_growth(log: TrainLog, out_dir: str | Path) -> Path:
    """Write `pain_nfe_growth.csv` and `.png`: primal and adjoint NFE against iteration."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _, nfe = log.rows()
    _write_csv(out / "pain_nfe_growth.csv", "pain_nfe_growth", nfe)

    def draw(ax):
        it = [r["iter"] for r in nfe]
        ax.plot(it, [r["nfe_fwd"] for r in nfe], color=_UFRJ_BLUE, label="primal solve (nfe_fwd)")
        ax.plot(it, [r["nfe_bwd"] for r in nfe], color=_UFRJ_ORANGE, label="adjoint solve (nfe_bwd)")
        ax.set_xlabel("training iteration")
        ax.set_ylabel("NFE per solve")
        ax.legend(frameon=False)
    _plot(out / "pain_nfe_growth.png", draw)
    return out / "pain_nfe_growth.csv"


def pain_tolerance(rows: Sequence[dict], out_dir: str | Path) -> Path:
    """Write `pain_tolerance.csv` and `.png` from one dict per tolerance.

    Each dict has the keys of the schema: rtol, atol, nfe_eval, nll_test,
    wallclock_eval_s. The NLL must come from the EXACT divergence (see
    `meta.json: divergence_mode`), so that the sweep isolates solver error.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: -float(r["rtol"]))
    _write_csv(out / "pain_tolerance.csv", "pain_tolerance", rows)

    def draw(ax):
        rt = [float(r["rtol"]) for r in rows]
        ax.plot(rt, [r["nll_test"] for r in rows], "o-", color=_UFRJ_BLUE)
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("rtol")
        ax.set_ylabel("test NLL (exact divergence)", color=_UFRJ_BLUE)
        ax2 = ax.twinx()
        ax2.plot(rt, [r["nfe_eval"] for r in rows], "s--", color=_UFRJ_ORANGE)
        ax2.set_ylabel("NFE of the evaluation solve", color=_UFRJ_ORANGE)
        ax2.spines["right"].set_visible(True)
    _plot(out / "pain_tolerance.png", draw)
    return out / "pain_tolerance.csv"


def make_meta(dataset: str, seed: int, config: dict, *, pair_id: str,
              budget_version: str = BUDGET_VERSION,
              evidence_origin: str = "student",
              divergence_mode: str = "hutchinson_rademacher",
              session_id: str = SESSION_ID, device: str = "cpu") -> dict:
    """Assemble the thirteen fields of `meta.json` for this machine."""
    import torch
    ref = evidence_origin == "reference_pair"
    return {
        "session_id": session_id,
        "dataset": dataset,
        "seed": int(seed),
        "config_hash": config_hash(config),
        "machine_id": machine_id(reference=ref),
        "device_name": device_name(device),
        "evidence_origin": evidence_origin,
        "pair_id": pair_id,
        "torch_version": torch.__version__,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "budget_version": budget_version,
        "painplots_version": PAINPLOTS_VERSION,
        "divergence_mode": divergence_mode,
    }


def write_meta(meta: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps({k: meta[k] for k in META_FIELDS}, indent=2) + "\n")
    return out / "meta.json"


# --- reading, pairing, validation --------------------------------------------

def read_evidence(dataset_dir: str | Path) -> dict:
    """Read one evidence directory into plain lists, keyed by file stem, plus `meta`."""
    d = Path(dataset_dir)
    out: dict = {"meta": json.loads((d / "meta.json").read_text())}
    for name in SCHEMAS:
        header, rows = _read_csv(d / f"{name}.csv")
        out[name] = {c: [float(r[i]) for r in rows] for i, c in enumerate(header)}
    return out


def pairing_key(meta: dict) -> tuple[str, str, str]:
    """`machine_id + pair_id + budget_version`: two runs are comparable iff equal."""
    return (meta["machine_id"], meta["pair_id"], meta["budget_version"])


def validate_evidence_dir(dataset_dir: str | Path, *, expected_session: str | None = None) -> list[str]:
    """Return a list of problems; an empty list means the directory is valid."""
    d = Path(dataset_dir)
    bad: list[str] = []
    if not d.is_dir():
        return [f"{d} is not a directory"]
    for name in SCHEMAS:
        for ext in ("csv", "png"):
            if not (d / f"{name}.{ext}").is_file():
                bad.append(f"missing {name}.{ext}")
    if not (d / "meta.json").is_file():
        bad.append("missing meta.json")
    for name, cols in SCHEMAS.items():
        p = d / f"{name}.csv"
        if not p.is_file():
            continue
        header, rows = _read_csv(p)
        if tuple(header) != cols:
            bad.append(f"{name}.csv header is {header}, the contract says {list(cols)}")
            continue
        if not rows:
            bad.append(f"{name}.csv has no rows")
            continue
        try:
            vals = [[float(x) for x in r] for r in rows]
        except ValueError:
            bad.append(f"{name}.csv has a value that is not a number")
            continue
        if any(not math.isfinite(x) for r in vals for x in r):
            bad.append(f"{name}.csv has a NaN or an infinity")
        if name in ("pain_wallclock", "pain_nfe_growth"):
            it = [r[0] for r in vals]
            if any(b - a != LOG_EVERY for a, b in zip(it, it[1:])) or it[0] != LOG_EVERY:
                bad.append(f"{name}.csv: `iter` must run {LOG_EVERY}, {2 * LOG_EVERY}, ... in steps of {LOG_EVERY}")
        if name == "pain_wallclock":
            cum = [r[2] for r in vals]
            if any(b < a for a, b in zip(cum, cum[1:])):
                bad.append("pain_wallclock.csv: `cum_wallclock_s` must not decrease")
        if name == "pain_tolerance" and any(r[0] <= 0 or r[1] <= 0 for r in vals):
            bad.append("pain_tolerance.csv: rtol and atol must be positive")
    mp = d / "meta.json"
    if mp.is_file():
        try:
            meta = json.loads(mp.read_text())
        except json.JSONDecodeError:
            return bad + ["meta.json is not valid JSON"]
        missing = [k for k in META_FIELDS if k not in meta]
        extra = [k for k in meta if k not in META_FIELDS]
        if missing:
            bad.append(f"meta.json lacks {missing}")
        if extra:
            bad.append(f"meta.json has fields outside the contract: {extra}")
        if not missing:
            if meta["evidence_origin"] not in EVIDENCE_ORIGINS:
                bad.append(f"evidence_origin is {meta['evidence_origin']!r}, the literals are {list(EVIDENCE_ORIGINS)}")
            else:
                ref = meta["evidence_origin"] == "reference_pair"
                for k in ("machine_id", "pair_id"):
                    if str(meta[k]).startswith(REFERENCE_PREFIX) != ref:
                        bad.append(f"{k} {'must' if ref else 'must not'} start with {REFERENCE_PREFIX!r} "
                                   f"for evidence_origin={meta['evidence_origin']}")
            if meta["painplots_version"] != PAINPLOTS_VERSION:
                bad.append(f"painplots_version is {meta['painplots_version']!r}, expected {PAINPLOTS_VERSION!r}")
            if meta["budget_version"] not in BUDGET_VERSIONS:
                bad.append(f"budget_version is {meta['budget_version']!r}, expected one of {list(BUDGET_VERSIONS)}")
            if meta["divergence_mode"] not in DIVERGENCE_MODES:
                bad.append(f"divergence_mode is {meta['divergence_mode']!r}, expected one of {list(DIVERGENCE_MODES)}")
            if meta["dataset"] not in DATASET_NAMES:
                bad.append(f"dataset is {meta['dataset']!r}, expected one of {list(DATASET_NAMES)}")
            if d.name in DATASET_NAMES and meta["dataset"] != d.name:
                bad.append(f"meta.json says dataset {meta['dataset']!r} inside the folder {d.name!r}")
            if expected_session and meta["session_id"] != expected_session:
                bad.append(f"session_id is {meta['session_id']!r}, expected {expected_session!r}")
            for k in ("machine_id", "pair_id", "config_hash", "timestamp", "torch_version", "device_name"):
                if not str(meta[k]).strip():
                    bad.append(f"meta.json: {k} is empty")
    return bad


def print_validation(dataset_dir: str | Path, *, expected_session: str | None = None,
                     label: str | Path | None = None) -> bool:
    """The validator cell of PS2.5(b): prints PASS or the list of problems.

    `label` is what is printed in place of the path, so that the output you submit holds no
    absolute path of your computer."""
    problems = validate_evidence_dir(dataset_dir, expected_session=expected_session)
    shown = dataset_dir if label is None else label
    if problems:
        print(f"FAIL  {shown}")
        for p in problems:
            print(f"  - {p}")
        return False
    print(f"PASS  {shown}")
    return True


# --- MLflow convenience (not part of the frozen contract) --------------------

_PRIVATE_TAGS = ("mlflow.user", "mlflow.source.name", "mlflow.source.type", "mlflow.log-model.history")
_PRIVATE_PREFIXES = ("mlflow.source.git.",)    # repository URL, branch and commit of whoever ran it


def log_run(dataset_dir: str | Path | None, *, run_name: str, tags: dict, params: dict,
            log: TrainLog | None = None, experiment: str = "U2",
            tracking_uri: str | None = None, extra_artifacts: Sequence[str | Path] = ()) -> str:
    """One MLflow run for one training run, logged AFTER the fit; returns the run id.

    The metrics go in as one batch per 1000 values: a toy step is under a
    second, and a database write per step would be a visible share of it
    (U1.L1). The evidence directory is logged as an artifact, so the run folder
    and the evidence stay together.
    """
    import mlflow
    from mlflow.entities import Metric

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({k: str(v) for k, v in tags.items()})
        mlflow.log_params({k: str(v) for k, v in params.items()})
        if log is not None:
            batch = [Metric(name, float(v), 0, step + 1)
                     for name, values in (("loss", log.loss), ("step_s", log.step_s),
                                          ("nfe_fwd", log.nfe_fwd), ("nfe_bwd", log.nfe_bwd))
                     for step, v in enumerate(values)]
            client = mlflow.tracking.MlflowClient()
            for i in range(0, len(batch), 1000):
                client.log_batch(run.info.run_id, metrics=batch[i:i + 1000])
        if dataset_dir is not None:
            mlflow.log_artifacts(str(dataset_dir), artifact_path="pain_evidence")
        for extra in extra_artifacts:
            mlflow.log_artifact(str(extra))
        return run.info.run_id


def export_run(run_id: str, out_dir: str | Path, *, tracking_uri: str | None = None) -> Path:
    """Write a portable, sanitized copy of one run: `params.json`, `tags.json`, `metrics.csv`.

    The tracking database holds absolute paths and the user name, so the
    database itself is never what you submit; this export is (PS2.5 item (e)).
    """
    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tags = {k: v for k, v in run.data.tags.items()
            if k not in _PRIVATE_TAGS and not k.startswith(_PRIVATE_PREFIXES)}
    (out / "params.json").write_text(json.dumps(run.data.params, indent=2, sort_keys=True) + "\n")
    (out / "tags.json").write_text(json.dumps(tags, indent=2, sort_keys=True) + "\n")
    with (out / "metrics.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "step", "value"])
        for key in sorted(run.data.metrics):
            for m in client.get_metric_history(run_id, key):
                w.writerow([key, m.step, f"{m.value:.9g}"])
    return out


# --- self-test --------------------------------------------------------------

def _self_test() -> None:
    import random
    import shutil
    import tempfile

    rng = random.Random(0)
    log = TrainLog()
    for i in range(3000):
        log.add(0.1 + 1e-4 * i + 1e-3 * rng.random(), 20 + i // 50, 26 + i // 40, 4.0 - i / 1000)
    wall, nfe = log.rows()
    assert len(wall) == len(nfe) == 300 and wall[0]["iter"] == 10 and wall[-1]["iter"] == 3000
    assert abs(wall[-1]["cum_wallclock_s"] - sum(log.step_s)) < 1e-9, "cum is the running sum"
    assert abs(wall[0]["wallclock_s_per_iter"] - sum(log.step_s[:10]) / 10) < 1e-12, "window mean"

    import matplotlib
    matplotlib.use("pdf")                  # not Agg: a helper that switches to Agg would then be caught
    backend_before = matplotlib.get_backend()
    tmp = Path(tempfile.mkdtemp())
    try:
        good = tmp / "two-moons"
        pair = mint_pair_id(tmp)
        assert mint_pair_id(tmp) == pair, "the pair id is stable on one machine"
        assert hardware_id() == hardware_id() and hardware_id() != machine_id(), "the hardware id leaves the host name out"
        pain_wallclock(log, good)
        pain_nfe_growth(log, good)
        pain_tolerance([{"rtol": r, "atol": r * 1e-2, "nfe_eval": 40 + i, "nll_test": 2.2, "wallclock_eval_s": 1.0}
                        for i, r in enumerate((1e-2, 1e-3, 1e-4, 1e-5, 1e-6))], good)
        cfg = {"hidden": 64, "iters": 3000}
        write_meta(make_meta("two-moons", 2026, cfg, pair_id=pair), good)
        assert validate_evidence_dir(good) == [], validate_evidence_dir(good)
        assert print_validation(good)

        # The validator must FAIL on each known-bad input (rule: test the gate on the bad input).
        def broken(label: str, edit, folder: str = "two-moons-bad") -> None:
            bad = tmp / "bad"
            shutil.rmtree(bad, ignore_errors=True)
            shutil.copytree(good, bad)
            edit(bad)
            # keep the folder name aligned with the dataset so only the planted defect fires
            target = tmp / folder
            shutil.rmtree(target, ignore_errors=True)
            shutil.move(bad, target)
            problems = validate_evidence_dir(target)
            shutil.rmtree(target, ignore_errors=True)
            assert problems, f"validator passed the planted defect: {label}"

        def meta_edit(**kw):
            def f(d: Path) -> None:
                m = json.loads((d / "meta.json").read_text())
                m.update(kw)
                (d / "meta.json").write_text(json.dumps(m))
            return f

        broken("missing file", lambda d: (d / "pain_nfe_growth.csv").unlink())
        broken("renamed column", lambda d: (d / "pain_nfe_growth.csv").write_text(
            "iter,nfe_forward,nfe_bwd\n10,1,1\n"))
        broken("iter gap", lambda d: (d / "pain_wallclock.csv").write_text(
            "iter,wallclock_s_per_iter,cum_wallclock_s\n10,0.1,1\n30,0.1,2\n"))
        broken("NaN", lambda d: (d / "pain_nfe_growth.csv").write_text(
            "iter,nfe_fwd,nfe_bwd\n10,nan,1\n"))
        broken("cum decreases", lambda d: (d / "pain_wallclock.csv").write_text(
            "iter,wallclock_s_per_iter,cum_wallclock_s\n10,0.1,2\n20,0.1,1\n"))
        broken("old literal", meta_edit(evidence_origin="reference"))
        broken("student with a ref- id", meta_edit(machine_id="ref-abc"))
        broken("reference without ref- ids", meta_edit(evidence_origin="reference_pair"))
        broken("wrong version", meta_edit(painplots_version="1.0"))
        broken("unknown budget", meta_edit(budget_version="B1"))
        broken("extra field", meta_edit(cpu_count=4))
        broken("dataset folder mismatch", lambda d: None, folder="checkerboard")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    assert matplotlib.get_backend() == backend_before, "painplots must not switch the matplotlib backend"
    assert machine_id() == machine_id() and not machine_id().startswith(REFERENCE_PREFIX)
    assert machine_id(reference=True).startswith(REFERENCE_PREFIX)
    assert pairing_key({"machine_id": "a", "pair_id": "b", "budget_version": "c"}) == ("a", "b", "c")
    print("painplots self-test passed: schemas, window means, pairing, and 12 planted defects rejected")


if __name__ == "__main__":
    _self_test()
