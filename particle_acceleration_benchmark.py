"""Numerical utilities for the convex softmax-metric acceleration positive control."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Trajectory:
    times: np.ndarray
    objective: np.ndarray
    total_energy: np.ndarray
    states: np.ndarray | None = None


def _kernel(X: np.ndarray, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Batched exact kernel for X with shape (S,N,d)."""

    scores = np.matmul(np.matmul(X, A.T), np.swapaxes(X, -1, -2))
    M = np.exp(scores)
    z = M.sum(axis=-1)
    if not np.isfinite(M).all() or not np.isfinite(z).all() or np.any(z <= 0):
        raise FloatingPointError("invalid particle kernel")
    return M, z


def quadratic_objective(X: np.ndarray, H: np.ndarray) -> np.ndarray:
    return 0.5 * np.sum(X * np.matmul(X, H.T), axis=(-1, -2))


def metric_gradient_rhs(X: np.ndarray, A: np.ndarray, B: np.ndarray, H: np.ndarray) -> np.ndarray:
    _, z = _kernel(X, A)
    grad = np.matmul(X, H.T)
    return -float(X.shape[-2]) * np.matmul(grad, B.T) / z[..., None]


def metric_hamiltonian_rhs(
    X: np.ndarray,
    Y: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    H: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    M, z = _kernel(X, A)
    n = X.shape[-2]
    BY = np.matmul(Y, B.T)
    r = np.sum(Y * BY, axis=-1) / np.square(z)
    pair = M * (r[..., :, None] + r[..., None, :])
    kinetic_force = 0.5 * float(n) * np.matmul(np.matmul(pair, X), A.T)
    dX = float(n) * BY / z[..., None]
    dY = -alpha * Y + kinetic_force - np.matmul(X, H.T)
    return dX, dY


def hamiltonian_energy(
    X: np.ndarray,
    Y: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    _, z = _kernel(X, A)
    q = np.sum(Y * np.matmul(Y, B.T), axis=-1)
    kinetic = 0.5 * float(X.shape[-2]) * np.sum(q / z, axis=-1)
    return kinetic + quadratic_objective(X, H)


def _rk4_gradient(X: np.ndarray, dt: float, A: np.ndarray, B: np.ndarray, H: np.ndarray) -> np.ndarray:
    f = lambda q: metric_gradient_rhs(q, A, B, H)
    k1 = f(X)
    k2 = f(X + 0.5 * dt * k1)
    k3 = f(X + 0.5 * dt * k2)
    k4 = f(X + dt * k3)
    return X + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _rk4_hamiltonian(
    X: np.ndarray,
    Y: np.ndarray,
    dt: float,
    A: np.ndarray,
    B: np.ndarray,
    H: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    def f(q: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return metric_hamiltonian_rhs(q, p, A, B, H, alpha)

    k1x, k1y = f(X, Y)
    k2x, k2y = f(X + 0.5 * dt * k1x, Y + 0.5 * dt * k1y)
    k3x, k3y = f(X + 0.5 * dt * k2x, Y + 0.5 * dt * k2y)
    k4x, k4y = f(X + dt * k3x, Y + dt * k3y)
    return (
        X + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x),
        Y + (dt / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y),
    )


def integrate_gradient(
    X0: np.ndarray,
    *,
    A: np.ndarray,
    B: np.ndarray,
    H: np.ndarray,
    dt: float,
    horizon: float,
    keep_states: bool = False,
) -> Trajectory:
    steps = round(horizon / dt)
    if not np.isclose(steps * dt, horizon):
        raise ValueError("horizon must be an integer multiple of dt")
    X = X0.copy()
    times = np.arange(steps + 1, dtype=np.float64) * dt
    objective = np.empty((steps + 1, X.shape[0]), dtype=np.float64)
    states = np.empty((steps + 1,) + X.shape, dtype=np.float64) if keep_states else None
    objective[0] = quadratic_objective(X, H)
    if states is not None:
        states[0] = X
    for k in range(steps):
        X = _rk4_gradient(X, dt, A, B, H)
        objective[k + 1] = quadratic_objective(X, H)
        if states is not None:
            states[k + 1] = X
    return Trajectory(times=times, objective=objective, total_energy=objective.copy(), states=states)


def integrate_hamiltonian(
    X0: np.ndarray,
    *,
    A: np.ndarray,
    B: np.ndarray,
    H: np.ndarray,
    alpha: float,
    dt: float,
    horizon: float,
    keep_states: bool = False,
) -> Trajectory:
    steps = round(horizon / dt)
    if not np.isclose(steps * dt, horizon):
        raise ValueError("horizon must be an integer multiple of dt")
    X, Y = X0.copy(), np.zeros_like(X0)
    times = np.arange(steps + 1, dtype=np.float64) * dt
    objective = np.empty((steps + 1, X.shape[0]), dtype=np.float64)
    total = np.empty_like(objective)
    states = np.empty((steps + 1,) + X.shape, dtype=np.float64) if keep_states else None
    objective[0] = quadratic_objective(X, H)
    total[0] = hamiltonian_energy(X, Y, A, B, H)
    if states is not None:
        states[0] = X
    for k in range(steps):
        X, Y = _rk4_hamiltonian(X, Y, dt, A, B, H, alpha)
        objective[k + 1] = quadratic_objective(X, H)
        total[k + 1] = hamiltonian_energy(X, Y, A, B, H)
        if states is not None:
            states[k + 1] = X
    return Trajectory(times=times, objective=objective, total_energy=total, states=states)


def sustained_hitting_time(times: np.ndarray, ratios: np.ndarray, threshold: float) -> np.ndarray:
    """First sampled time after which every later ratio stays below threshold."""

    if ratios.ndim != 2 or ratios.shape[0] != times.size:
        raise ValueError("ratios must have shape (time,seeds)")
    out = np.full(ratios.shape[1], np.inf, dtype=np.float64)
    for seed_index in range(ratios.shape[1]):
        below = ratios[:, seed_index] <= threshold
        sustained = np.logical_and.accumulate(below[::-1])[::-1]
        indices = np.flatnonzero(sustained)
        if indices.size:
            out[seed_index] = times[indices[0]]
    return out
