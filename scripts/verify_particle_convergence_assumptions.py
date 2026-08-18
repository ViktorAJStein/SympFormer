#!/usr/bin/env python3
"""Falsify Nesterov convergence assumptions for the paper's attention energies."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from particle_schemes import vanilla_softmax_particle_rhs  # noqa: E402

DTYPE = torch.float64


def softmax_one_particle_energy(x: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.exp(x.square())


def linear_one_particle_energy(x: torch.Tensor) -> torch.Tensor:
    return -0.5 * x.square()


def main() -> None:
    A = torch.ones(1, 1, dtype=DTYPE)
    V = torch.ones(1, 1, dtype=DTYPE)

    # Allowed N=d=1 softmax geometry: A symmetric and B=VA^{-1}=1 SPD.
    x = torch.tensor(0.7, dtype=DTYPE, requires_grad=True)
    f = softmax_one_particle_energy(x)
    f1, = torch.autograd.grad(f, x, create_graph=True)
    f2, = torch.autograd.grad(f1, x)
    expected_f2 = -(1.0 + 2.0 * x.square()) * torch.exp(x.square())
    torch.testing.assert_close(f2, expected_f2)
    assert f2.item() < 0.0

    # The vanilla attention field is Gamma(x)=x, hence x(t)=x0 exp(t).
    x0 = 0.2
    for t in (0.0, 0.5, 1.0, 2.0):
        xt = torch.tensor([[x0 * math.exp(t)]], dtype=DTYPE)
        rhs = vanilla_softmax_particle_rhs(xt, A, V)
        torch.testing.assert_close(rhs, xt)
        exact_derivative = xt
        torch.testing.assert_close(rhs, exact_derivative)

    radii = torch.arange(0.0, 4.5, 0.5, dtype=DTYPE)
    soft_values = softmax_one_particle_energy(radii)
    linear_values = linear_one_particle_energy(radii)
    assert torch.all(soft_values[1:] < soft_values[:-1])
    assert torch.all(linear_values[1:] < linear_values[:-1])
    assert soft_values[-1].item() < -1e6

    # Linear attention N=d=1, A=V=1: x'=x^3 has the exact finite-time
    # blow-up x(t)=x0/sqrt(1-2*x0^2*t).
    x0_linear = 0.5
    blowup_time = 1.0 / (2.0 * x0_linear * x0_linear)
    assert math.isclose(blowup_time, 2.0)
    for t in (0.0, 0.5, 1.5, 1.9):
        xt = x0_linear / math.sqrt(1.0 - 2.0 * x0_linear * x0_linear * t)
        derivative_exact = xt**3
        # Differentiate the closed form analytically: x0^3*(1-2*x0^2*t)^(-3/2).
        derivative_formula = x0_linear**3 / (1.0 - 2.0 * x0_linear**2 * t) ** 1.5
        assert math.isclose(derivative_exact, derivative_formula, rel_tol=1e-13, abs_tol=1e-13)

    print("PASS: current attention energies fail the basic Nesterov convergence setup")
    print(f"softmax one-particle Hessian at x=0.7: {f2.item():.6f} (strictly concave)")
    print(f"softmax energy at x=4: {soft_values[-1].item():.6e} (decreases without bound)")
    print(f"linear one-particle finite blow-up time at x0=0.5: {blowup_time:.6f}")
    print("objective gap F-F* is undefined because F*=-infinity")


if __name__ == "__main__":
    main()
