# U2.L2 — the week's package

This folder belongs to the asynchronous week of **U2.L2** (FFJORD on 2D). The notebook is
`labs/lab-u2l2.ipynb`. Clone the public repository (<https://github.com/adricortes/flow-matching-labs>) and open the
notebook from `labs/`: the file alone does not have `common/` and `configs/`. The reasoning behind every part is in the lecture notes of the course site
(<https://adricortes.github.io/flow-matching-lectures/U2/U2-L2-notes.html>).

## Pacing

Estimated active time: 2.5 to 3.5 hours, plus training time that needs nothing from you (two runs of
about 396 s (two-moons) and 365 s (checkerboard) on the reference computer, one CPU thread).

1. Watch the proof clinic first. The release message of the week and the pinned message of the thread link the recording and its timestamped index.
2. Parts 1 to 3 in the first half of this week. Part 1 is the longest.
3. Parts 4 to 7 in the second half. Start the checkerboard (Part 5) only after the tolerance table of
   two-moons (Part 4) has printed; write the prompt cells of Parts 3 and 4 while the checkerboard trains.

The dates of this week are on the calendar of the course site (<https://adricortes.github.io/flow-matching-lectures/org/course-organization-notes.html#planned-absences>).

## Where to ask

One thread for the week. Tag a question `#lab` (this notebook), `#clinic` (the recording) or `#blocked`
(you cannot go on; answered within a day). The thread is in the channel that the course uses for announcements.
Its pinned message holds the links. Two check-in messages appear in the thread: one after the first half of the
week, one before the due date.

## The rescue ladder

Set `TIER`, run the settings cell, then run the training cell of that dataset.

| Tier | When | What to do |
|---|---|---|
| 1 | the training stops with a non-finite loss or `underflow in dt` | `TIER = 1`: the learning rate becomes `fallback_lr` (`configs/u2l2-budget.yaml`) |
| 2 | one run needs more than 45 minutes | `TIER = 2`: `configs/u2l2-rescue.yaml` (width 32, 1500 iterations, `budget_version` `U2L2-B1R`) |
| 3 | the training is unusable | `TIER = 3`: the reference checkpoint and the reference evidence; your MLflow run is tagged `rescue=true`; Part 1 must still pass |

## Files

| Path | What |
|---|---|
| `labs/lab-u2l2.ipynb` | the notebook |
| `common/painplots.py` | the measurement helper, given complete, contract frozen (v1.1) |
| `common/toy2d.py` | the datasets of U1.L1, unchanged |
| `configs/u2l2-budget.yaml`, `configs/u2l2-rescue.yaml` | the budgets |
| `common/weights/u2l2-two-moons.pt`, `u2l2-checkerboard.pt` (+ `.json`) | reference checkpoints |
| `labs/u2l2/reference/pain_evidence/<dataset>/` | the reference evidence pair (`evidence_origin=reference_pair`) |
| `labs/u2l2/reference/mlflow_export/<dataset>/` | the reference MLflow export |

## What you submit (PS2.5)

The notebook with its outputs; `pain_evidence/two-moons/` and `pain_evidence/checkerboard/`;
`mlflow_export/`; your written answers (two to four sentences for each of the prompts of Parts 3, 4 and 5); and, if you ran
Part 6, `rnode_overlay.png`.
Send these items as one ZIP file to the instructor's institutional e-mail. Do not add `mlflow.db`, `runs/` or
the rest of your clone. Use the subject `PS2.5 <your name>`. The pinned message of the thread states the address
and the due date. Keep the folders in your clone.
U3.L1 reads the two `pain_evidence/` directories and `pain_evidence/pair.json` again. A tier applies to one dataset at a time.

`pain_evidence/`, `mlflow_export/`, `runs/` and `rnode_overlay.png` are written in the repository root, the parent
of `labs/`; `mlflow.db` is written next to the notebook. Each trained run is saved in `runs/`, so a restart of the
kernel does not lose it: the notebook reloads it.
