# Flow-Based Generative Models — lab code

Student code for the UFRJ course **Flow-Based Generative Models: From
Normalizing Flows to Flow Matching** (PGMAT / PESC, 2026.2).

**The lecture notes and slides are at https://adricortes.github.io/flow-matching-lectures/.** This repository holds what the
notes ask you to run: the lab notebooks, the `common/` package every lab
imports, the checkpoints, and the scripts behind the figures.

> This repository is generated from the course's source repository. Please
> report problems as an issue here; pull requests cannot be merged back.

## Layout

| Path | What it is |
|---|---|
| `labs/` | One notebook per lab (`lab-u1l1.ipynb`, …), with TODOs for you to fill in |
| `labs/PACKET-README.md` | Start here if you are a PESC student: the self-study packet |
| `common/` | The course package: training loop, solvers, flows, evaluation |
| `common/weights/` | Checkpoints the labs load, each with a JSON provenance record |
| `configs/`, `train_cifar.py` | The U0.L2 training-hygiene lab (scripts, not a notebook) |
| `scripts/python/` | The scripts that produced the figures in the notes, and their recorded measurements |

Open the notebooks **from `labs/`**: they add the repository root to the path so
that `import common` works.

## Setup

GPU (the full lab environment, CUDA 12.8):

```bash
conda env create -f environment.yml
conda activate torch-gpu
```

CPU only, for the PESC packet (`lab-u0l5.ipynb` and `jax-primer.ipynb`):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-packet.txt
```

The other labs also import packages that this file does not pin (MLflow among
them); `environment.yml` lists them all.

## Solutions

The instructor releases each lab's solutions notebook. The two notebooks of
the PESC self-study packet (`lab-u0l5-solutions.ipynb`,
`jax-primer-solutions.ipynb`) are here from the start, because the packet is
studied alone.

## Licence

MIT — see `LICENSE`.
