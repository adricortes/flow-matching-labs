# PESC Self-Study Packet — Flow-Based Generative Models (UFRJ 2026.2)

You are joining a course that assumes some deep-learning practice its PGMAT
students built in a five-week bootcamp you did not attend. This packet is the
part of that bootcamp you actually need, and nothing else.

**No joint session of this course requires having attended U0.** That is a rule
of the course, not a courtesy. This packet and the probability primer are the
sanctioned way in; everything else stands without them.

## What is here

Two kinds of thing: **pages you read online**, and **notebooks in the course's
lab repository**, <https://github.com/adricortes/flow-matching-labs>.

### Read online

| Page | What it is |
|---|---|
| [Numerical ODE Solvers](https://adricortes.github.io/flow-matching-lectures/U0/U0-T5-notes.html) | The theory, written standalone-first — complete prose, four exercises **with answers** |
| [Solvers by Hand and a JAX Crash Course](https://adricortes.github.io/flow-matching-lectures/U0/U0-L5-notes.html) | The session these notebooks belong to |
| [Probability for Generative Models](https://adricortes.github.io/flow-matching-lectures/UP/UP-A-notes.html) | The probability the joint sessions assume |
| [Gaussian Algebra Toolkit](https://adricortes.github.io/flow-matching-lectures/UP/UP-A-toolkit.html) | The boxed formulas alone, two pages, laid out for your browser's print command |

The pages are the current version, always. They are not copied into this
repository on purpose: a copied page is a page that goes stale, and you would have
no way of knowing when.

### In the lab repository

| File | What it is |
|---|---|
| `lab-u0l5.ipynb` | Build Euler, Heun, RK4 and one adaptive solver from scratch. 6 TODOs |
| `jax-primer.ipynb` | JAX in five moves, then the same ODE in diffrax. 6 TODOs |
| `lab-u0l5-solutions.ipynb` | The same notebook, filled in |
| `jax-primer-solutions.ipynb` | The same |
| `../common/odesolve.py` | The reference implementation of what you build in evening 2 |
| `../requirements-packet.txt` | The exact dependency pins |

The notebooks are in `labs/`, beside this file; the other two are one level up,
at the repository root.

**You get the solutions immediately.** PGMAT students get theirs after the
session; you get yours now, because unsupervised struggle with no recourse
serves nobody. Try each TODO first — the checks after them are written to tell
you *what* is wrong, not merely *that* something is.

## Everything runs on CPU

This is a hard requirement, not a convenience. Nothing here needs a GPU.

```bash
git clone https://github.com/adricortes/flow-matching-labs.git
cd flow-matching-labs
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-packet.txt
cd labs && jupyter notebook
```

The `torch` line in that file installs the **CPU** build. A plain
`pip install torch` would hand you about 2.5 GB of CUDA you cannot use.
Verified from a clean environment: 116 packages, zero NVIDIA packages.

Windows and macOS work with the same lines. On Apple silicon, torch's CPU
wheel is native arm64.

## A suggested pacing: three evenings

Not a schedule — a shape. Each evening is roughly two hours and ends somewhere
sensible.

### Evening 1 — the theory

Read [Numerical ODE Solvers](https://adricortes.github.io/flow-matching-lectures/U0/U0-T5-notes.html) end to end, and do the
four exercises **before** reading their answers.

Exercise 2 asks you to *predict* the three convergence slopes. Evening 2 has
you measure them, and the prediction is what makes the measurement mean
something. Predicting after you have seen the plot teaches nothing.

Stop after "The equation that moves the density". That section is a teaser for
a later unit: read it once, and leave it alone.

### Evening 2 — build the solvers

`lab-u0l5.ipynb`. Six TODOs. The first three are transcription from evening
1; the fourth, the adaptive method, is the one with a decision in it.

Three checkpoints assert *numbers* rather than show pictures, and that is
deliberate: a hand-written solver that draws a plausible-looking wrong curve is
the failure mode here, and the eye does not catch it.

What you produce is imported later in the course to draw the first samples of
its first Flow Matching model.

### Evening 3 — JAX and diffrax

`jax-primer.ipynb`. Five moves and a capstone. It is deliberately small:
enough to *read* the JAX in this literature and to follow the course's JAX
moments. PyTorch stays the course's primary language, and the JAX moments are
always mirrors, never the only path.

One cell is a **planted error**. Run it, read the message, then fix it. That is
on purpose — an error message you have met once is a different thing from one
you have read about.

## The maths you will need, and where to find it

[UP.A](https://adricortes.github.io/flow-matching-lectures/UP/UP-A-notes.html) is released in the same window as this packet and
is where the joint sessions get their probability. Read at least these.

| If a session assumes… | Read |
|---|---|
| joint, marginal, conditional; the tower property | [§1](https://adricortes.github.io/flow-matching-lectures/UP/UP-A-notes.html#sec-jmc), through "Expectations" |
| **the subscript convention on every expectation** | [§1 Expectations](https://adricortes.github.io/flow-matching-lectures/UP/UP-A-notes.html#sec-expectations) — not optional. The course's losses are unreadable without it |
| Gaussian algebra: linear maps, sums, completing the square | [§2.1](https://adricortes.github.io/flow-matching-lectures/UP/UP-A-notes.html#sec-gauss-def)–[§2.4](https://adricortes.github.io/flow-matching-lectures/UP/UP-A-notes.html#sec-square) |
| conditionals and marginals of a joint Gaussian | [§2.5](https://adricortes.github.io/flow-matching-lectures/UP/UP-A-notes.html#sec-jointgauss) and the worked example [§2.6](https://adricortes.github.io/flow-matching-lectures/UP/UP-A-notes.html#sec-noisyobs) |
| KL divergence, its asymmetry, KL between Gaussians | [§3](https://adricortes.github.io/flow-matching-lectures/UP/UP-A-notes.html#sec-kldef) |

Keep the [toolkit](https://adricortes.github.io/flow-matching-lectures/UP/UP-A-toolkit.html) beside you — it is §2's boxed
formulas alone, and it prints to two pages.

## What this packet does *not* cover

The bootcamp also built a training loop, a hygiene stack, a VAE, a U-Net and an
evaluation harness. **You do not need them.** Where a joint session uses one,
it arrives as given code with a described contract, and the session stands
without it.

## If something does not work

The notebooks' checks are meant to be diagnostic. If one fails and its message
does not tell you enough to act, **that is a defect in the packet** — say so,
and it gets fixed for the next reader.
