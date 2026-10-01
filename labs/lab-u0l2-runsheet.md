# U0.L2 Run-Sheet — Training Hygiene on CIFAR-10

**No notebook today — deliberately.** Hygiene is about *runs*: things you
launch, tag, compare, and resume from a terminal. Notebooks return in U0.L3.
(The full rationale is in the session notes, §"Format shift".)

Files you work in: `common/train.py` (template v1 → **v2**, extended in
place) and `train_cifar.py` (the launcher). Configs live in `configs/`;
your tracked runs land in `mlflow.db` + `runs/<run_name>/`.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db     # keep open all session
```

| Block | TODO (id · part) | Where | Gate — *if it passes, continue* |
|---|---|---|---|
| A | A.1 · config | `train_cifar.py` (`load_config`, `build`) | `python train_cifar.py --config configs/smoke.yaml` launches and logs |
| B | B.1 · tracking | `common/train.py` (`fit`, both cadences) | run appears in the UI with train loss, lr, val metrics, grad/weight norms, img/s |
| B | demo | — | run `baseline.yaml` and `baseline-s1.yaml`; Compare view: curves wiggle apart, land together; params differ in `seed` only |
| C | C.1 · schedule | `common/train.py` (`fit`) | the logged LR traces the warmup–cosine curve — **the plot is the test** (flat = you stepped per epoch) |
| C | C.2 · checkpoint | `common/train.py` (`resume_from_ckpt`) | `python train_cifar.py --config configs/smoke.yaml --selftest-resume` prints PASS |
| D | D.1 · repro | the MLflow UI | find `mlflow.source.git.commit` among your run's tags — you **will**, and it matches `git rev-parse HEAD`; now look for a dirty indicator beside it — there is none. That gap is why the template logs `git_commit` **and** `git_dirty` itself |
| E | the mystery run | `mystery-runs.zip` (distributed in class) | `mlflow ui --backend-store-uri sqlite:///mystery.db` — curves and metrics only. Fast finishers: `U0L2-mystery2` is a second pathology in the same store |
| F | freeze | — | commit `common/train.py` v2: this is `hygiene-stack`, required verbatim by PS0 |

**The five-step task for the mystery-run block (E)** (curves alone, in pairs): describe the
pathology → name ≥2 candidate causes → rank them → propose the single
cheapest discriminating experiment → run it. The bug's name is not the
deliverable; the discipline is.
