#!/usr/bin/env python3
"""Verify vanilla softmax and theoretical SympFormer particle schemes."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from particle_schemes import (  # noqa: E402
    softmax_particle_kernel,
    sympformer_initial_acceleration,
    sympformer_softmax_euler_step,
    sympformer_softmax_kick_drift_step,
    sympformer_softmax_particle_rhs,
    validate_sympformer_particle_geometry,
    vanilla_softmax_euler_step,
    vanilla_softmax_particle_rhs,
)

DTYPE = torch.float64
ATOL = 2e-10
RTOL = 2e-10


def spd(d: int, generator: torch.Generator, floor: float = 0.5) -> torch.Tensor:
    C = torch.randn(d, d, generator=generator, dtype=DTYPE)
    return C @ C.T / d + floor * torch.eye(d, dtype=DTYPE)


def direct_vanilla(X: torch.Tensor, A: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    n, d = X.shape
    out = torch.zeros_like(X)
    for i in range(n):
        weights = torch.stack([torch.exp(X[i] @ A @ X[j]) for j in range(n)])
        weights = weights / weights.sum()
        for j in range(n):
            out[i] += weights[j] * (V @ X[j])
    return out


def direct_symp(
    X: torch.Tensor, Y: torch.Tensor, A: torch.Tensor, V: torch.Tensor, alpha: float
) -> tuple[torch.Tensor, torch.Tensor]:
    n, _ = X.shape
    B = V @ torch.linalg.inv(A)
    M = torch.empty(n, n, dtype=X.dtype)
    for i in range(n):
        for j in range(n):
            M[i, j] = torch.exp(X[i] @ A @ X[j])
    z = M.sum(dim=1)
    r = torch.stack([(Y[i] @ B @ Y[i]) / z[i].square() for i in range(n)])
    dX = torch.empty_like(X)
    dY = torch.empty_like(Y)
    for i in range(n):
        dX[i] = n * (B @ Y[i]) / z[i]
        force = torch.zeros_like(X[i])
        for j in range(n):
            force += M[i, j] * (r[i] + r[j] + 2.0) * (A @ X[j])
        dY[i] = -alpha * Y[i] + 0.5 * n * force
    return dX, dY


def assert_close(a: torch.Tensor, b: torch.Tensor, label: str, atol: float = ATOL, rtol: float = RTOL) -> None:
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        err = (a - b).abs().max().item()
        raise AssertionError(f"{label}: max error {err:.3e}")


def main() -> None:
    g = torch.Generator().manual_seed(20260806)
    max_loop_error = 0.0
    max_accel_error = 0.0

    for n, d in [(1, 1), (2, 3), (5, 4), (8, 2)]:
        A = spd(d, g)
        B = spd(d, g)
        V = B @ A  # ensures B = V A^{-1}, with A and B SPD
        X = 0.12 * torch.randn(n, d, generator=g, dtype=DTYPE)
        Y = 0.09 * torch.randn(n, d, generator=g, dtype=DTYPE)
        alpha = 0.37

        recovered_B = validate_sympformer_particle_geometry(A, V)
        assert_close(recovered_B, B, f"B recovery N={n}")

        vanilla = vanilla_softmax_particle_rhs(X, A, V)
        vanilla_loop = direct_vanilla(X, A, V)
        max_loop_error = max(max_loop_error, (vanilla - vanilla_loop).abs().max().item())
        assert_close(vanilla, vanilla_loop, f"vanilla loop N={n}")

        M, _ = softmax_particle_kernel(X, A)
        standard_attention = torch.softmax((X @ A.T) @ X.T, dim=-1) @ (X @ V.T)
        assert_close(vanilla, standard_attention, f"standard attention N={n}")
        h = torch.tensor(0.013, dtype=DTYPE)
        assert_close(vanilla_softmax_euler_step(X, A, V, h), X + h * standard_attention, f"vanilla Euler N={n}")

        rhs = sympformer_softmax_particle_rhs(X, Y, A, V, alpha=alpha)
        dX_loop, dY_loop = direct_symp(X, Y, A, V, alpha)
        max_loop_error = max(
            max_loop_error,
            (rhs.dX - dX_loop).abs().max().item(),
            (rhs.dY - dY_loop).abs().max().item(),
        )
        assert_close(rhs.dX, dX_loop, f"Symp dX loop N={n}")
        assert_close(rhs.dY, dY_loop, f"Symp dY loop N={n}")

        Xe, Ye = sympformer_softmax_euler_step(X, Y, A, V, h, alpha=alpha)
        assert_close(Xe, X + h * rhs.dX, f"Symp Euler X N={n}")
        assert_close(Ye, Y + h * rhs.dY, f"Symp Euler Y N={n}")

        # Global all-particle systems must be permutation equivariant.
        perm = torch.randperm(n, generator=g)
        inv = torch.argsort(perm)
        vp = vanilla_softmax_particle_rhs(X[perm], A, V)[inv]
        assert_close(vp, vanilla, f"vanilla permutation N={n}")
        rp = sympformer_softmax_particle_rhs(X[perm], Y[perm], A, V, alpha=alpha)
        assert_close(rp.dX[inv], rhs.dX, f"Symp dX permutation N={n}")
        assert_close(rp.dY[inv], rhs.dY, f"Symp dY permutation N={n}")
        assert_close(rp.kernel[inv][:, inv], M, f"kernel permutation N={n}")

        # Exact structural acceleration identity at zero momentum.
        acceleration = sympformer_initial_acceleration(X, A, V)
        expected = float(n * n) * vanilla
        max_accel_error = max(max_accel_error, (acceleration - expected).abs().max().item())
        assert_close(acceleration, expected, f"initial acceleration N={n}", atol=5e-10, rtol=5e-10)

        # Independent JVP of dX along the full ODE direction at Y=0.
        Y0 = torch.zeros_like(X)
        rhs0 = sympformer_softmax_particle_rhs(X, Y0, A, V, alpha=alpha)
        def position_rhs(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return sympformer_softmax_particle_rhs(x, y, A, V, alpha=alpha).dX
        _, accel_jvp = torch.func.jvp(position_rhs, (X, Y0), (rhs0.dX, rhs0.dY))
        assert_close(accel_jvp, expected, f"JVP acceleration N={n}", atol=5e-10, rtol=5e-10)

        # The kick--drift displacement exposes the same identity discretely.
        Xkd, _ = sympformer_softmax_kick_drift_step(X, Y0, A, V, h, alpha=alpha)
        assert_close((Xkd - X) / h.square(), expected, f"kick-drift acceleration N={n}", atol=2e-9, rtol=2e-9)

    # Differentiability through both vector fields.
    n, d = 4, 3
    A = spd(d, g)
    B = spd(d, g)
    V = B @ A
    X = (0.1 * torch.randn(n, d, generator=g, dtype=DTYPE)).requires_grad_(True)
    Y = (0.1 * torch.randn(n, d, generator=g, dtype=DTYPE)).requires_grad_(True)
    vanilla = vanilla_softmax_particle_rhs(X, A, V)
    rhs = sympformer_softmax_particle_rhs(X, Y, A, V, alpha=lambda t: 0.2 + 0.1 * t, t=0.7)
    loss = vanilla.square().sum() + rhs.dX.square().sum() + rhs.dY.square().sum()
    gx, gy = torch.autograd.grad(loss, (X, Y))
    assert torch.isfinite(gx).all() and torch.isfinite(gy).all()
    assert gx.norm().item() > 0 and gy.norm().item() > 0

    # Falsify the stronger claim that acceleration is only a time change of
    # vanilla flow: arbitrary nonzero momentum gives a different direction.
    gamma = vanilla.detach().flatten()
    inertial_direction = rhs.dX.detach().flatten()
    cosine = torch.dot(gamma, inertial_direction) / (gamma.norm() * inertial_direction.norm())
    if abs(cosine.item()) > 0.95:
        raise AssertionError(f"chosen falsification case is nearly collinear: cosine={cosine.item():.6f}")

    # Geometry assumptions fail loudly.
    bad_A = A.clone(); bad_A[0, 1] += 0.2
    try:
        validate_sympformer_particle_geometry(bad_A, V)
    except ValueError:
        pass
    else:
        raise AssertionError("nonsymmetric A should fail geometry validation")

    print("PASS: vanilla and SympFormer softmax particle schemes")
    print(f"max direct-loop error={max_loop_error:.3e}")
    print(f"max initial-acceleration error={max_accel_error:.3e}")
    print(f"nonzero-momentum direction cosine={cosine.item():.6f}")
    print(f"gradient norms: X={gx.norm().item():.6e}, Y={gy.norm().item():.6e}")


if __name__ == "__main__":
    main()
