"""Standalone interacting-particle schemes for theoretical softmax attention.

These functions implement the global systems in arXiv_v1.tex, equations
``eq:attention`` and ``eq:acc_softmax_particles``. They intentionally do not
apply a causal mask, LayerNorm, an MLP, or learned per-layer projections.
Consequently they are theory/diagnostic utilities, not headline decoder blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union

import torch

Tensor = torch.Tensor
Alpha = Union[float, Tensor, Callable[[float], Union[float, Tensor]]]


@dataclass(frozen=True)
class SympFormerParticleRHS:
    """Right-hand side and reusable kernel data for the accelerated system."""

    dX: Tensor
    dY: Tensor
    kernel: Tensor
    normalizer: Tensor
    kinetic_ratio: Tensor
    B: Tensor


@dataclass(frozen=True)
class MetricHamiltonianParticleRHS:
    """RHS for a general potential in the finite softmax particle co-metric."""

    dX: Tensor
    dY: Tensor
    kernel: Tensor
    normalizer: Tensor
    kinetic_ratio: Tensor
    kinetic_force: Tensor


def _check_particle_inputs(X: Tensor, A: Tensor, V: Tensor, Y: Tensor | None = None) -> None:
    if X.ndim != 2:
        raise ValueError(f"X must have shape (N,d), got {tuple(X.shape)}")
    n, d = X.shape
    if n < 1 or d < 1:
        raise ValueError("X must contain at least one particle and one feature")
    if A.shape != (d, d) or V.shape != (d, d):
        raise ValueError(f"A and V must both have shape {(d, d)}")
    if Y is not None and Y.shape != X.shape:
        raise ValueError(f"Y must have shape {tuple(X.shape)}, got {tuple(Y.shape)}")
    if A.device != X.device or V.device != X.device or (Y is not None and Y.device != X.device):
        raise ValueError("X, Y, A, and V must be on the same device")
    if A.dtype != X.dtype or V.dtype != X.dtype or (Y is not None and Y.dtype != X.dtype):
        raise ValueError("X, Y, A, and V must have the same dtype")


def softmax_particle_kernel(X: Tensor, A: Tensor, *, check_finite: bool = True) -> tuple[Tensor, Tensor]:
    """Return ``M_ij=exp(X_i^T A X_j)`` and row sums ``z_i``.

    The exact unshifted exponential is required by the accelerated Hamiltonian
    system: unlike ordinary row-softmax, its full right-hand side is not
    invariant under arbitrary rowwise shifts of the scores.
    """

    if X.ndim != 2 or A.shape != (X.shape[-1], X.shape[-1]):
        raise ValueError("expected X with shape (N,d) and A with shape (d,d)")
    scores = (X @ A.T) @ X.T
    M = torch.exp(scores)
    z = M.sum(dim=-1)
    if check_finite and (not torch.isfinite(M).all() or not torch.isfinite(z).all() or (z <= 0).any()):
        raise FloatingPointError("nonfinite or nonpositive softmax particle kernel; rescale X or A")
    return M, z


def vanilla_softmax_particle_rhs(
    X: Tensor,
    A: Tensor,
    V: Tensor,
    *,
    check_finite: bool = True,
) -> Tensor:
    """Vanilla softmax interacting-particle velocity ``Gamma(X)``.

    In row notation, ``Gamma = diag(M 1)^(-1) M X V^T``.
    """

    _check_particle_inputs(X, A, V)
    M, z = softmax_particle_kernel(X, A, check_finite=check_finite)
    return (M / z.unsqueeze(-1)) @ (X @ V.T)


def vanilla_softmax_euler_step(
    X: Tensor,
    A: Tensor,
    V: Tensor,
    h: float | Tensor,
    *,
    check_finite: bool = True,
) -> Tensor:
    """Forward Euler for vanilla particles, i.e. residual softmax attention."""

    return X + torch.as_tensor(h, device=X.device, dtype=X.dtype) * vanilla_softmax_particle_rhs(
        X, A, V, check_finite=check_finite
    )


def softmax_particle_B(A: Tensor, V: Tensor) -> Tensor:
    """Compute ``B=V A^{-1}`` without explicitly forming ``A^{-1}``."""

    if A.ndim != 2 or A.shape[0] != A.shape[1] or V.shape != A.shape:
        raise ValueError("A and V must be square matrices with the same shape")
    return torch.linalg.solve(A.T, V.T).T


def validate_sympformer_particle_geometry(
    A: Tensor,
    V: Tensor,
    *,
    atol: float = 1e-8,
) -> Tensor:
    """Validate the paper assumptions ``A=A^T`` and ``B=VA^-1=B^T>0``."""

    B = softmax_particle_B(A, V)
    if not torch.allclose(A, A.T, atol=atol, rtol=atol):
        raise ValueError("the theoretical softmax particle system requires symmetric A")
    if not torch.allclose(B, B.T, atol=atol, rtol=atol):
        raise ValueError("the theoretical softmax particle system requires symmetric B=V A^{-1}")
    eig_min = torch.linalg.eigvalsh(B).min()
    if not bool(eig_min > 0):
        raise ValueError(f"B=V A^(-1) must be positive definite; min eigenvalue={eig_min.item():.3e}")
    return B


def _alpha_value(alpha: Alpha, t: float, reference: Tensor) -> Tensor:
    value = alpha(t) if callable(alpha) else alpha
    value = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if value.numel() != 1:
        raise ValueError("alpha(t) must be scalar")
    return value


def sympformer_softmax_particle_rhs(
    X: Tensor,
    Y: Tensor,
    A: Tensor,
    V: Tensor,
    *,
    alpha: Alpha = 0.0,
    t: float = 0.0,
    validate_geometry: bool = True,
    check_finite: bool = True,
) -> SympFormerParticleRHS:
    """The theoretical accelerated softmax interacting-particle system.

    This is Proposition ``lemma:acc_softmax_particles`` in row-matrix notation.
    The empirical-measure convention in the paper produces the explicit factors
    ``N`` retained here.
    """

    _check_particle_inputs(X, A, V, Y)
    B = validate_sympformer_particle_geometry(A, V) if validate_geometry else softmax_particle_B(A, V)
    M, z = softmax_particle_kernel(X, A, check_finite=check_finite)
    n = X.shape[0]

    BY = Y @ B.T
    kinetic_ratio = (Y * BY).sum(dim=-1) / z.square()
    pair_factor = kinetic_ratio.unsqueeze(-1) + kinetic_ratio.unsqueeze(0) + 2.0
    weighted_kernel = M * pair_factor

    dX = float(n) * BY / z.unsqueeze(-1)
    conservative_force = (0.5 * float(n)) * ((weighted_kernel @ X) @ A.T)
    dY = -_alpha_value(alpha, t, Y) * Y + conservative_force
    if check_finite and (not torch.isfinite(dX).all() or not torch.isfinite(dY).all()):
        raise FloatingPointError("nonfinite SympFormer particle right-hand side")
    return SympFormerParticleRHS(dX=dX, dY=dY, kernel=M, normalizer=z, kinetic_ratio=kinetic_ratio, B=B)


def sympformer_softmax_euler_step(
    X: Tensor,
    Y: Tensor,
    A: Tensor,
    V: Tensor,
    h: float | Tensor,
    *,
    alpha: Alpha = 0.0,
    t: float = 0.0,
    validate_geometry: bool = True,
    check_finite: bool = True,
) -> tuple[Tensor, Tensor]:
    """Vanilla simultaneous forward Euler for the accelerated particle ODE."""

    rhs = sympformer_softmax_particle_rhs(
        X, Y, A, V, alpha=alpha, t=t,
        validate_geometry=validate_geometry, check_finite=check_finite,
    )
    h_t = torch.as_tensor(h, device=X.device, dtype=X.dtype)
    return X + h_t * rhs.dX, Y + h_t * rhs.dY


def sympformer_softmax_kick_drift_step(
    X: Tensor,
    Y: Tensor,
    A: Tensor,
    V: Tensor,
    h: float | Tensor,
    *,
    alpha: Alpha = 0.0,
    t: float = 0.0,
    validate_geometry: bool = True,
    check_finite: bool = True,
) -> tuple[Tensor, Tensor]:
    """One-force kick--drift diagnostic for the nonseparable particle system.

    ``Y`` is first advanced by old-state forward Euler; ``X`` then uses the new
    momentum but the old position/kernel. This explicit map is useful for
    comparison with the practical kick--drift block, but is not claimed to be a
    symplectic Euler discretization of the nonseparable Hamiltonian.
    """

    rhs = sympformer_softmax_particle_rhs(
        X, Y, A, V, alpha=alpha, t=t,
        validate_geometry=validate_geometry, check_finite=check_finite,
    )
    h_t = torch.as_tensor(h, device=X.device, dtype=X.dtype)
    Y_new = Y + h_t * rhs.dY
    dX_new_momentum = float(X.shape[0]) * (Y_new @ rhs.B.T) / rhs.normalizer.unsqueeze(-1)
    return X + h_t * dX_new_momentum, Y_new


def softmax_metric_gradient_rhs(
    X: Tensor,
    A: Tensor,
    B: Tensor,
    potential_gradient: Tensor,
    *,
    check_finite: bool = True,
) -> Tensor:
    """Negative metric gradient for a general finite-particle potential.

    The block-diagonal co-metric is ``K_i(X)=(N/z_i(X)) B``. Therefore this
    returns ``-K(X) grad U(X)`` for a supplied Euclidean row gradient.
    ``A=0`` is accepted as the constant-kernel Euclidean limit.
    """

    _check_particle_inputs(X, A, B)
    if potential_gradient.shape != X.shape:
        raise ValueError("potential_gradient must have the same shape as X")
    if not torch.allclose(A, A.T, atol=1e-8, rtol=1e-8):
        raise ValueError("the finite softmax co-metric requires symmetric A")
    if not torch.allclose(B, B.T, atol=1e-8, rtol=1e-8) or not bool(torch.linalg.eigvalsh(B).min() > 0):
        raise ValueError("B must be symmetric positive definite")
    _, z = softmax_particle_kernel(X, A, check_finite=check_finite)
    dX = -float(X.shape[0]) * (potential_gradient @ B.T) / z.unsqueeze(-1)
    if check_finite and not torch.isfinite(dX).all():
        raise FloatingPointError("nonfinite softmax metric-gradient right-hand side")
    return dX


def softmax_metric_hamiltonian_rhs(
    X: Tensor,
    Y: Tensor,
    A: Tensor,
    B: Tensor,
    potential_gradient: Tensor,
    *,
    alpha: Alpha = 0.0,
    t: float = 0.0,
    check_finite: bool = True,
) -> MetricHamiltonianParticleRHS:
    """Damped Hamiltonian lift for a general potential in the softmax metric.

    Its Hamiltonian is

    ``H(X,Y) = (N/2) sum_i <Y_i,BY_i>/z_i(X) + U(X)``.

    This routine supplies the exact kinetic geometry force and subtracts the
    caller-provided Euclidean gradient of ``U``. It is the controlled surrogate
    used to test acceleration independently of the unbounded attention energy.
    """

    _check_particle_inputs(X, A, B, Y)
    if potential_gradient.shape != X.shape:
        raise ValueError("potential_gradient must have the same shape as X")
    if not torch.allclose(A, A.T, atol=1e-8, rtol=1e-8):
        raise ValueError("the finite softmax co-metric requires symmetric A")
    if not torch.allclose(B, B.T, atol=1e-8, rtol=1e-8) or not bool(torch.linalg.eigvalsh(B).min() > 0):
        raise ValueError("B must be symmetric positive definite")

    M, z = softmax_particle_kernel(X, A, check_finite=check_finite)
    n = X.shape[0]
    BY = Y @ B.T
    kinetic_ratio = (Y * BY).sum(dim=-1) / z.square()
    kinetic_pair = M * (kinetic_ratio.unsqueeze(-1) + kinetic_ratio.unsqueeze(0))
    kinetic_force = (0.5 * float(n)) * ((kinetic_pair @ X) @ A.T)
    dX = float(n) * BY / z.unsqueeze(-1)
    dY = -_alpha_value(alpha, t, Y) * Y + kinetic_force - potential_gradient
    if check_finite and (not torch.isfinite(dX).all() or not torch.isfinite(dY).all()):
        raise FloatingPointError("nonfinite metric-Hamiltonian right-hand side")
    return MetricHamiltonianParticleRHS(
        dX=dX,
        dY=dY,
        kernel=M,
        normalizer=z,
        kinetic_ratio=kinetic_ratio,
        kinetic_force=kinetic_force,
    )


def sympformer_initial_acceleration(
    X: Tensor,
    A: Tensor,
    V: Tensor,
    *,
    validate_geometry: bool = True,
    check_finite: bool = True,
) -> Tensor:
    """Return ``d^2 X/dt^2`` at ``Y=0`` for the theoretical system."""

    Y0 = torch.zeros_like(X)
    rhs0 = sympformer_softmax_particle_rhs(
        X, Y0, A, V, alpha=0.0, t=0.0,
        validate_geometry=validate_geometry, check_finite=check_finite,
    )
    # At Y=0, dX/dt=0, so derivatives through X, z, and B Y vanish.
    return float(X.shape[0]) * (rhs0.dY @ rhs0.B.T) / rhs0.normalizer.unsqueeze(-1)
