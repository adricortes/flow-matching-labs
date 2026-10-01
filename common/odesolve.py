"""common/odesolve.py — ODE solvers from scratch (artifact: `ode-solvers-scratch`).

Built in U0.L5. **The interface below is FROZEN.** U3.L1 draws the first samples
of the course's first Flow Matching model by calling `odeint` from this file, so
a change after the U0.L5 freeze is a breaking change and needs a dated CHANGELOG
note.

THE CONTRACT (U0.L5 notes section B; L3-U0-L5 section 2 part B):

    odeint(f, x0, t0, t1, method="rk4", n_steps=None,
           rtol=None, atol=None, return_traj=False)
        -> x1                     # a SolverOutput: a Tensor carrying .nfe
        -> (ts, xs)               # if return_traj=True; xs carries .nfe

    METHODS = {"euler", "heun", "rk4", "adaptive_heun"}

THREE CONVENTIONS THIS FILE ENFORCES, each recorded in U0.T5:

1. **`f(t, x)` — time first.** This matches `torchdiffeq` and `diffrax`, so the
   right-hand side a student writes here is the one the professional libraries
   accept in U2.L1. `t` is a scalar tensor; `x` has the shape of `x0`.

2. **NFE is a first-class output, not a diagnostic.** Every call reports the
   number of evaluations of `f` it spent. NFE is this course's cost currency:
   U2 watches it grow during training, U3 is about making it small. A solver
   that hides its NFE cannot be compared with another one.

3. **The state is batched by construction**, `x0` of shape `(B, d)`. The step
   loop is an ordinary Python loop over steps; the batch dimension is carried
   by the tensor ops. There is no `vmap` here and none is needed.

WHY `SolverOutput` IS A TENSOR SUBCLASS. The same trick as `FIDValue` in
`common/eval.py`: the value IS a `torch.Tensor`, so every call site written
against the frozen `-> x1` contract works unchanged, and it additionally
carries the cost that produced it. Making the cost travel with the number is
what stops NFE from being forgotten at the moment it matters.

DELIBERATELY SIMPLE, AND THE SCOPE IS THE POINT. This module integrates
forward. It does not differentiate *through* the integration, and it has no
adjoint — that is U2.T1's subject, and building it here would teach the wrong
lesson about what a solver is. Sampling is inference: no gradient is needed to
draw a sample. Nothing below blocks autograd (every operation is a plain torch
op, so a gradient will flow if you ask for one), but the memory cost of that
is exactly the problem U2.T1 exists to solve.

Run the self-test:  conda run -n torch-gpu python common/odesolve.py
"""

from __future__ import annotations

from typing import Callable

import torch

Field = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

METHODS = ("euler", "heun", "rk4", "adaptive_heun")

# Controller constants (U0.T5 notes, the step-size rule). The safety factor
# makes the next step slightly more conservative than the estimate suggests,
# so a marginal step is not immediately rejected; the clips stop one freak
# estimate from moving the step size by orders of magnitude.
SAFETY = 0.9
MIN_FACTOR = 0.2
MAX_FACTOR = 5.0


class SolverOutput(torch.Tensor):
    """A solver result that carries the cost that produced it.

    It is a `torch.Tensor`: it prints, indexes, broadcasts and serializes like
    one. It additionally knows how many times the field was evaluated, which
    method spent them, and — for an adaptive method — how the steps divided
    into accepted and rejected.

    The attributes do not survive an arithmetic operation, exactly as
    `FIDValue`'s do not. Read `.nfe` from the value the solver returned.
    """

    nfe: int
    method: str
    n_steps: int
    accepted: int | None
    rejected: int | None

    @staticmethod
    def __new__(
        cls,
        data: torch.Tensor,
        *,
        nfe: int,
        method: str,
        n_steps: int,
        accepted: int | None = None,
        rejected: int | None = None,
    ) -> "SolverOutput":
        self = torch.Tensor._make_subclass(cls, data, data.requires_grad)
        self.nfe = nfe
        self.method = method
        self.n_steps = n_steps
        self.accepted = accepted
        self.rejected = rejected
        return self

    def report(self) -> str:
        """One line naming the method and what it cost."""
        if self.accepted is None:
            return f"{self.method}: {self.n_steps} steps, NFE = {self.nfe}"
        return (
            f"{self.method}: {self.accepted} accepted + {self.rejected} "
            f"rejected steps, NFE = {self.nfe}"
        )


# --------------------------------------------------------------------------
# One step of each fixed-step method.
#
# Each stepper takes (f, t, x, h) and returns (x_next, k_evaluations_used).
# The tableaux are transcribed from U0.T5; nothing here is re-derived.
# --------------------------------------------------------------------------


def _step_euler(f: Field, t: torch.Tensor, x: torch.Tensor, h: torch.Tensor):
    """x_{n+1} = x_n + h f(t_n, x_n).  1 NFE."""
    k1 = f(t, x)
    return x + h * k1, 1


def _step_heun(f: Field, t: torch.Tensor, x: torch.Tensor, h: torch.Tensor):
    """Predict with Euler, then correct with the averaged velocity.  2 NFE."""
    k1 = f(t, x)
    k2 = f(t + h, x + h * k1)
    return x + (h / 2) * (k1 + k2), 2


def _step_rk4(f: Field, t: torch.Tensor, x: torch.Tensor, h: torch.Tensor):
    """Classical Runge-Kutta. The weights 1/6, 2/6, 2/6, 1/6 are Simpson.  4 NFE."""
    k1 = f(t, x)
    k2 = f(t + h / 2, x + (h / 2) * k1)
    k3 = f(t + h / 2, x + (h / 2) * k2)
    k4 = f(t + h, x + h * k3)
    return x + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4), 4


_FIXED_STEPPERS = {
    "euler": _step_euler,
    "heun": _step_heun,
    "rk4": _step_rk4,
}


def _fixed_step_solve(
    f: Field,
    x0: torch.Tensor,
    t0: float,
    t1: float,
    method: str,
    n_steps: int,
    return_traj: bool,
):
    """Integrate with a constant step size, counting every evaluation."""
    step = _FIXED_STEPPERS[method]
    h = torch.as_tensor((t1 - t0) / n_steps, dtype=x0.dtype, device=x0.device)

    x = x0
    nfe = 0
    traj = [x0] if return_traj else None
    times = [t0] if return_traj else None

    for n in range(n_steps):
        t = torch.as_tensor(t0 + n * (t1 - t0) / n_steps, dtype=x0.dtype,
                            device=x0.device)
        x, used = step(f, t, x, h)
        nfe += used
        if return_traj:
            traj.append(x)
            times.append(t0 + (n + 1) * (t1 - t0) / n_steps)

    return x, nfe, n_steps, traj, times


def _error_norm(
    err: torch.Tensor, x: torch.Tensor, x_new: torch.Tensor, rtol: float, atol: float
) -> torch.Tensor:
    """The normalised error of U0.T5: RMS of the error over its own scale.

    The scale is built component by component, sc = atol + rtol * max(|x|,
    |x_new|), so that `rtol` governs where the solution is large and `atol`
    governs where it passes through zero.

    ONE STEP SIZE FOR THE WHOLE BATCH. The RMS runs over every element of the
    batched state, so the controller serves the hardest sample in the batch.
    That is what `torchdiffeq` and `diffrax` do too, and it is the honest
    choice: a per-sample step size would make NFE a per-sample quantity and
    the batch would stop being a batch.
    """
    scale = atol + rtol * torch.maximum(x.abs(), x_new.abs())
    return torch.sqrt(torch.mean((err / scale) ** 2))


def _adaptive_heun_solve(
    f: Field,
    x0: torch.Tensor,
    t0: float,
    t1: float,
    rtol: float,
    atol: float,
    return_traj: bool,
    max_steps: int = 1_000_000,
):
    """Embedded Heun(2)/Euler(1), with the damped-and-clipped controller.

    Both estimates come from the SAME two evaluations: k1 alone is the Euler
    step, k1 and k2 together are the Heun step. Their difference estimates the
    local error of the lower-order one, and the higher-order value is the one
    propagated (local extrapolation).
    """
    direction = 1.0 if t1 >= t0 else -1.0
    span = abs(t1 - t0)

    x = x0
    t = float(t0)
    h = direction * span / 100.0  # a starting guess; the controller fixes it
    nfe = 0
    accepted = 0
    rejected = 0

    traj = [x0] if return_traj else None
    times = [t0] if return_traj else None

    while direction * (t1 - t) > 0:
        # Never step past the endpoint.
        if direction * (t + h - t1) > 0:
            h = t1 - t

        t_t = torch.as_tensor(t, dtype=x0.dtype, device=x0.device)
        h_t = torch.as_tensor(h, dtype=x0.dtype, device=x0.device)

        k1 = f(t_t, x)
        k2 = f(t_t + h_t, x + h_t * k1)
        nfe += 2

        x_low = x + h_t * k1                 # Euler,  order 1
        x_high = x + (h_t / 2) * (k1 + k2)   # Heun,   order 2
        err = _error_norm(x_high - x_low, x, x_high, rtol, atol)

        if err <= 1.0:
            t = t + h
            x = x_high                        # local extrapolation
            accepted += 1
            if return_traj:
                traj.append(x)
                times.append(t)
        else:
            rejected += 1

        # The same rule whether the step was accepted or rejected. p = 1 is
        # the order of the estimate being controlled, so the exponent is
        # -1/(p+1) = -1/2.
        err_val = float(err)
        if err_val == 0.0:
            factor = MAX_FACTOR
        else:
            factor = SAFETY * err_val ** (-0.5)
        factor = min(MAX_FACTOR, max(MIN_FACTOR, factor))
        h = h * factor

        if accepted + rejected > max_steps:
            raise RuntimeError(
                f"adaptive_heun exceeded {max_steps} steps at t = {t:.6g}. "
                "The tolerances are probably tighter than the problem allows."
            )

    return x, nfe, accepted, rejected, traj, times


def odeint(
    f: Field,
    x0: torch.Tensor,
    t0: float,
    t1: float,
    method: str = "rk4",
    n_steps: int | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    return_traj: bool = False,
):
    """Integrate dx/dt = f(t, x) from t0 to t1. FROZEN CONTRACT (U0.L5).

    Parameters
    ----------
    f
        The right-hand side, **time first**: `f(t, x)` with `t` a scalar
        tensor and `x` of the shape of `x0`. This is the U0.T5 convention and
        the signature `torchdiffeq` and `diffrax` expect.
    x0
        Initial state, shape `(B, d)`. A batch of `B` independent initial
        conditions integrated under one field.
    t0, t1
        Start and end time. `t1 < t0` integrates backwards.
    method
        One of `"euler"`, `"heun"`, `"rk4"` (fixed step, need `n_steps`) or
        `"adaptive_heun"` (needs `rtol` and `atol`).
    n_steps
        Number of steps, for a fixed-step method.
    rtol, atol
        Tolerances, for the adaptive method. `rtol` governs where the solution
        is large, `atol` where it passes through zero.
    return_traj
        When True, return `(ts, xs)` instead of the endpoint alone.

    Returns
    -------
    SolverOutput
        The state at `t1`, a Tensor of the shape of `x0` carrying `.nfe`.
    (ts, xs)
        When `return_traj=True`: `ts` of shape `(N+1,)` and `xs` of shape
        `(N+1, B, d)`, with `xs` carrying `.nfe`.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    if x0.dim() != 2:
        raise ValueError(
            f"x0 must be (B, d), got shape {tuple(x0.shape)}. A single initial "
            "condition is a batch of one: use x0.unsqueeze(0)."
        )

    if method == "adaptive_heun":
        if rtol is None or atol is None:
            raise ValueError("adaptive_heun needs both rtol and atol")
        if n_steps is not None:
            raise ValueError(
                "adaptive_heun chooses its own steps; n_steps is not accepted"
            )
        x, nfe, accepted, rejected, traj, times = _adaptive_heun_solve(
            f, x0, t0, t1, rtol, atol, return_traj
        )
        n_used = accepted
    else:
        if n_steps is None or n_steps < 1:
            raise ValueError(f"{method} needs a positive n_steps")
        if rtol is not None or atol is not None:
            raise ValueError(
                f"{method} is a fixed-step method; it ignores rtol and atol, "
                "so passing them is refused rather than silently obeyed"
            )
        x, nfe, n_used, traj, times = _fixed_step_solve(
            f, x0, t0, t1, method, n_steps, return_traj
        )
        accepted = rejected = None

    if not return_traj:
        return SolverOutput(
            x, nfe=nfe, method=method, n_steps=n_used,
            accepted=accepted, rejected=rejected,
        )

    ts = torch.as_tensor(times, dtype=x0.dtype, device=x0.device)
    xs = torch.stack(traj, dim=0)
    return ts, SolverOutput(
        xs, nfe=nfe, method=method, n_steps=n_used,
        accepted=accepted, rejected=rejected,
    )


if __name__ == "__main__":
    # Self-test on the harmonic oscillator of U0.T5: the exact orbit is the
    # unit circle, so the error is known without a reference solver.
    def harmonic(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([x[:, 1], -x[:, 0]], dim=1)

    x0 = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    T = 2 * torch.pi

    for method, n in (("euler", 4000), ("heun", 2000), ("rk4", 1000)):
        x1 = odeint(harmonic, x0, 0.0, float(T), method=method, n_steps=n)
        err = float((x1 - x0).norm())
        print(f"{x1.report():<44s} error = {err:.3e}")

    x1 = odeint(harmonic, x0, 0.0, float(T), method="adaptive_heun",
                rtol=1e-8, atol=1e-10)
    err = float((x1 - x0).norm())
    print(f"{x1.report():<44s} error = {err:.3e}")
