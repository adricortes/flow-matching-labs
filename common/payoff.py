"""common/payoff.py — the payoff comparison (artifact: `payoff-helper`, local, given complete).

Built in U3.L1. It reads two evidence directories in the `painplots` v1.1 layout,
one written by the U2.L2 likelihood-trained CNF (`pain_evidence/<dataset>/`) and one
written by the U3.L1 CFM run (`cfm_evidence/<dataset>/`), checks that they form a
valid PAIR, and draws the two receipts on shared axes. U3.L2 and U3.L3 reuse its
plotting conventions (L2-U3 artifact registry).

THE PAIRING RULE (painplots v1.1, frozen in U2.L2). Two runs are comparable in
wall-clock time only when they share `machine_id + pair_id + budget_version`.
`find_pair` refuses any other combination and says which field differs. Reference
evidence (`evidence_origin == "reference_pair"`) pairs only with reference evidence.

THE NFE RULE (L3-U3-L1 section 0). NFE is the number of calls of the vector field
made by a numerical ODE SOLVER. CFM training runs no solver, so its NFE columns are
0. The one direct call of the network per training step is NOT an NFE: it is shown
beside panel (b) as `field evaluations per step`, under its own name.

NO LAW IS ASSERTED. The CNF curves are drawn as measured. Nothing here states that a
curve grows, falls or is monotonic; the figure draws the CSV rows and nothing else.

THE THREE PANELS (one figure per dataset):
    (a) wall-clock per training iteration, both runs, log scale;
    (b) training solver NFE per iteration: CNF primal + adjoint solves, CFM = 0;
    (c) evaluation: solver NFE of the test-NLL solve against rtol, both models.

The figure is drawn on its own `Figure` object: this module never switches the
matplotlib backend (MEMORY 2026-10-04, the painplots lesson).

Run the self-test:  conda run -n torch-gpu python common/payoff.py
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path

from matplotlib.figure import Figure

try:                                   # inside the repository: common/ is a package
    from common import painplots
except ImportError:                    # run as a script: common/ is the script's directory
    import painplots                   # type: ignore[no-redef]

PAYOFF_VERSION = "1.0"
CNF_SESSION, CFM_SESSION = "U2.L2", "U3.L1"
FIELD_EVALS_TRAIN = 1                  # direct calls of u_t^theta per CFM training step

_BLUE, _ORANGE, _GRAY, _INK = "#003366", "#D97A1F", "#6b7280", "#26262a"
_FS = {"title": 16, "label": 16, "tick": 15, "legend": 14, "annot": 15}   # read at ~0.5x on a 16:9 slide


class PairingError(ValueError):
    """The two evidence directories do not form a valid pair."""


@dataclass
class Pair:
    """Two validated evidence directories of one dataset that share the pairing key."""

    dataset: str
    cnf_dir: Path
    cfm_dir: Path
    cnf: dict
    cfm: dict
    key: tuple[str, str, str]

    @property
    def origin(self) -> str:
        return self.cnf["meta"]["evidence_origin"]


def find_pair(cnf_dir: str | Path, cfm_dir: str | Path) -> Pair:
    """Validate both directories and check the pairing key; raise `PairingError` otherwise."""
    cnf_dir, cfm_dir = Path(cnf_dir), Path(cfm_dir)
    for d, session in ((cnf_dir, CNF_SESSION), (cfm_dir, CFM_SESSION)):
        label = f"{d.parent.name}/{d.name}"          # never an absolute path: it would print a home directory
        problems = [q.replace(str(d), label) for q in painplots.validate_evidence_dir(d, expected_session=session)]
        if problems:
            raise PairingError(f"{label} is not valid {session} evidence: " + "; ".join(problems))
    cnf, cfm = painplots.read_evidence(cnf_dir), painplots.read_evidence(cfm_dir)
    if cnf["meta"]["dataset"] != cfm["meta"]["dataset"]:
        raise PairingError(f"datasets differ: {cnf['meta']['dataset']} against {cfm['meta']['dataset']}")
    names = ("machine_id", "pair_id", "budget_version")
    k1, k2 = painplots.pairing_key(cnf["meta"]), painplots.pairing_key(cfm["meta"])
    differ = [f"{n}: {a!r} against {b!r}" for n, a, b in zip(names, k1, k2) if a != b]
    if differ:
        raise PairingError("not a pair (machine_id + pair_id + budget_version must agree): "
                           + "; ".join(differ))
    if any(v != 0 for col in ("nfe_fwd", "nfe_bwd") for v in cfm["pain_nfe_growth"][col]):
        raise PairingError("the CFM evidence records a nonzero training solver NFE: CFM training runs no solver")
    return Pair(cnf["meta"]["dataset"], cnf_dir, cfm_dir, cnf, cfm, k1)


def choose_pair(dataset: str, *, own_cnf: str | Path, own_cfm: str | Path,
                ref_cnf: str | Path, ref_cfm: str | Path) -> tuple[str, Pair]:
    """The comparison mode: ("own", pair) when the student's two runs pair, else ("reference", pair).

    Prints the reason, so the notebook cell output states which rule applied.
    """
    try:
        pair = find_pair(Path(own_cnf) / dataset, Path(own_cfm) / dataset)
        print(f"[{dataset}] mode: own pair  key = {pair.key}")
        _warn_devices(pair)
        return "own", pair
    except PairingError as err:
        print(f"[{dataset}] your two runs do not pair: {err}")
    pair = find_pair(Path(ref_cnf) / dataset, Path(ref_cfm) / dataset)
    print(f"[{dataset}] mode: reference pair  key = {pair.key}")
    print(f"[{dataset}] the wall-clock panel compares the two REFERENCE runs only; "
          "your own CFM run is not drawn against a CNF from another machine")
    return "reference", pair


def _warn_devices(pair: Pair) -> None:
    """The frozen key does not compare devices; a pair that spans two of them is printed, not refused."""
    a, b = pair.cnf["meta"]["device_name"], pair.cfm["meta"]["device_name"]
    if a != b:
        print(f"[{pair.dataset}] WARNING: the two runs used different devices "
              f"({a!r} for the CNF run, {b!r} for the CFM run). The pair is valid under the frozen key, "
              "but the wall-clock ratio then compares devices as well as methods.")


def paired_ratios(pair: Pair) -> dict:
    """Within-pair ratios: the only form of a class-wide wall-clock claim the rule allows."""
    a = pair.cnf["pain_wallclock"]["wallclock_s_per_iter"]
    b = pair.cfm["pain_wallclock"]["wallclock_s_per_iter"]
    if len(a) != len(b):
        raise PairingError(f"the two runs logged {len(a)} and {len(b)} windows: not the same budget")
    per_window = [x / y for x, y in zip(a, b)]
    return {
        "median_s_per_iter_ratio": statistics.median(per_window),
        "cum_wallclock_ratio": pair.cnf["pain_wallclock"]["cum_wallclock_s"][-1]
        / pair.cfm["pain_wallclock"]["cum_wallclock_s"][-1],
        "cnf_cum_s": pair.cnf["pain_wallclock"]["cum_wallclock_s"][-1],
        "cfm_cum_s": pair.cfm["pain_wallclock"]["cum_wallclock_s"][-1],
    }


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRAY)
    ax.tick_params(colors=_INK, labelsize=_FS["tick"])
    ax.grid(color="#DDDDDD", linewidth=0.6)
    ax.set_axisbelow(True)


def plot_payoff(pair: Pair, *, title: bool = True) -> Figure:
    """The three panels of one pair, on one new `Figure`. Draws the CSV rows; asserts no trend."""
    fig = Figure(figsize=(12.0, 4.6))
    axes = fig.subplots(1, 3)
    cw, fw = pair.cnf["pain_wallclock"], pair.cfm["pain_wallclock"]
    cn, fn = pair.cnf["pain_nfe_growth"], pair.cfm["pain_nfe_growth"]
    ct, ft = pair.cnf["pain_tolerance"], pair.cfm["pain_tolerance"]

    ax = axes[0]
    ax.plot(cw["iter"], cw["wallclock_s_per_iter"], color=_BLUE, lw=1.6, label="CNF (U2.L2)")
    ax.plot(fw["iter"], fw["wallclock_s_per_iter"], color=_ORANGE, lw=1.6, label="CFM (U3.L1)")
    ax.set_yscale("log")
    ax.set_xlabel("iteration", fontsize=_FS["label"])
    ax.set_ylabel("seconds per iteration", fontsize=_FS["label"])
    ax.set_title("(a) wall-clock per iteration", fontsize=_FS["title"], loc="left")
    r = paired_ratios(pair)
    lo = min(min(fw["wallclock_s_per_iter"]), min(cw["wallclock_s_per_iter"]))
    hi = max(max(fw["wallclock_s_per_iter"]), max(cw["wallclock_s_per_iter"]))
    ax.set_ylim(lo / 3, hi * 3)                  # room above and below the two bands
    ax.text(0.97, 0.30, f"median ratio\n{r['median_s_per_iter_ratio']:.0f}×",
            transform=ax.transAxes, ha="right", va="center", fontsize=_FS["annot"], color=_INK)
    ax.legend(fontsize=_FS["legend"], frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.52))

    ax = axes[1]
    ax.plot(cn["iter"], cn["nfe_fwd"], color=_BLUE, lw=1.6, label="CNF, likelihood")
    ax.plot(cn["iter"], cn["nfe_bwd"], color=_BLUE, lw=1.6, ls="--", label="CNF, adjoint")
    ax.plot(fn["iter"], [a + b for a, b in zip(fn["nfe_fwd"], fn["nfe_bwd"])], color=_ORANGE, lw=2.2,
            label="CFM: no solve")
    ax.set_ylim(-5, 1.65 * max(max(cn["nfe_fwd"]), max(cn["nfe_bwd"])))   # headroom: the legend sits above the curves
    ax.set_xlabel("iteration", fontsize=_FS["label"])
    ax.set_ylabel("solver NFE per iteration", fontsize=_FS["label"])
    ax.set_title("(b) training: solver NFE", fontsize=_FS["title"], loc="left")
    ax.text(0.97, 0.10, f"CFM: {FIELD_EVALS_TRAIN} field call\nper step (not NFE)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=_FS["annot"], color=_ORANGE)
    ax.legend(fontsize=_FS["legend"], frameon=False, loc="upper left")

    ax = axes[2]
    ax.plot(ct["rtol"], ct["nfe_eval"], "o-", color=_BLUE, lw=1.6, label="CNF")
    ax.plot(ft["rtol"], ft["nfe_eval"], "s-", color=_ORANGE, lw=1.6, label="CFM")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("rtol (tighter to the right)", fontsize=_FS["label"])
    ax.set_ylabel("solver NFE, NLL solve", fontsize=_FS["label"])
    ax.set_title("(c) evaluation: solver NFE", fontsize=_FS["title"], loc="left")
    ax.legend(fontsize=_FS["legend"], frameon=False, loc="upper left")

    for ax in axes:
        _style(ax)
    if title:
        fig.suptitle(f"{pair.dataset}: one pair, key {pair.key[1]} / {pair.key[2]}", fontsize=_FS["title"],
                     color=_INK)
    fig.tight_layout()
    return fig


def tolerance_table(pair: Pair) -> str:
    """The evaluation sweep of both models as plain text: rtol, NFE and test NLL side by side."""
    ct, ft = pair.cnf["pain_tolerance"], pair.cfm["pain_tolerance"]
    lines = [f"{'rtol':>8} | {'CNF NFE':>7} {'CNF NLL':>8} | {'CFM NFE':>7} {'CFM NLL':>8}"]
    for i, rtol in enumerate(ct["rtol"]):
        j = ft["rtol"].index(rtol)
        lines.append(f"{rtol:8.0e} | {ct['nfe_eval'][i]:7.0f} {ct['nll_test'][i]:8.4f} | "
                     f"{ft['nfe_eval'][j]:7.0f} {ft['nll_test'][j]:8.4f}")
    return "\n".join(lines)


def _self_test() -> None:
    """Pair the shipped reference evidence; refuse a mismatched key and a nonzero CFM NFE."""
    import json
    import shutil
    import tempfile

    root = Path(__file__).resolve().parents[1]
    cnf_root = root / "labs" / "u2l2" / "reference" / "pain_evidence"
    cfm_root = root / "labs" / "u3l1" / "reference" / "cfm_evidence"
    if not cfm_root.is_dir():
        print("no reference CFM evidence yet: run scripts/python/fig_u3_l1_payoff.py")
        return
    for ds in painplots.DATASET_NAMES:
        pair = find_pair(cnf_root / ds, cfm_root / ds)
        assert pair.origin == "reference_pair"
        plot_payoff(pair)
        print(ds, pair.key, {k: round(v, 2) for k, v in paired_ratios(pair).items()})
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "two-moons"
        shutil.copytree(cfm_root / "two-moons", bad)
        meta = json.loads((bad / "meta.json").read_text())
        meta["pair_id"] = "ref-another"
        (bad / "meta.json").write_text(json.dumps(meta))
        try:
            find_pair(cnf_root / "two-moons", bad)
        except PairingError as err:
            assert "pair_id" in str(err), err
        else:
            raise AssertionError("a mismatched pair_id was accepted")
        meta["pair_id"] = painplots.REFERENCE_PAIR_ID      # a valid key again; now break the NFE column
        (bad / "meta.json").write_text(json.dumps(meta))
        nfe = (bad / "pain_nfe_growth.csv").read_text().splitlines()
        nfe[1] = nfe[1].rsplit(",", 1)[0] + ",1"
        (bad / "pain_nfe_growth.csv").write_text("\n".join(nfe) + "\n")
        try:
            find_pair(cnf_root / "two-moons", bad)
        except PairingError as err:
            assert "nonzero" in str(err), err
        else:
            raise AssertionError("a nonzero CFM training NFE was accepted")
        # the device warning: silent on one device, printed (with both names) on two
        import contextlib
        import io
        same = io.StringIO()
        with contextlib.redirect_stdout(same):
            _warn_devices(find_pair(cnf_root / "two-moons", cfm_root / "two-moons"))
        assert "WARNING" not in same.getvalue(), same.getvalue()
        gpu = Path(tmp) / "gpu" / "two-moons"
        shutil.copytree(cfm_root / "two-moons", gpu)
        meta = json.loads((gpu / "meta.json").read_text())
        meta["device_name"] = "Test GPU"
        (gpu / "meta.json").write_text(json.dumps(meta))
        other = io.StringIO()
        with contextlib.redirect_stdout(other):
            _warn_devices(find_pair(cnf_root / "two-moons", gpu))
        assert "WARNING" in other.getvalue() and "Test GPU" in other.getvalue(), other.getvalue()
    print("payoff self-test: PASS")


if __name__ == "__main__":
    _self_test()
