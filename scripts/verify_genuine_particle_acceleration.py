#!/usr/bin/env python3
"""Verify E021 geometry formulas, integration, artifacts, and frozen gate."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from particle_acceleration_benchmark import (  # noqa: E402
    integrate_gradient,
    integrate_hamiltonian,
    metric_gradient_rhs,
    metric_hamiltonian_rhs,
)
from particle_schemes import (  # noqa: E402
    softmax_metric_gradient_rhs,
    softmax_metric_hamiltonian_rhs,
)


def verify_formulas() -> None:
    torch.manual_seed(20260806)
    dtype = torch.float64
    n, d = 5, 3
    C = torch.randn(d, d, dtype=dtype)
    A = 0.08 * (C @ C.T / d)
    D = torch.randn(d, d, dtype=dtype)
    B = D @ D.T / d + 0.5 * torch.eye(d, dtype=dtype)
    E = torch.randn(d, d, dtype=dtype)
    H = E @ E.T / d + 0.02 * torch.eye(d, dtype=dtype)
    X = (0.15 * torch.randn(n, d, dtype=dtype)).requires_grad_(True)
    Y = (0.1 * torch.randn(n, d, dtype=dtype)).requires_grad_(True)
    grad_u = X @ H.T

    gf = softmax_metric_gradient_rhs(X, A, B, grad_u)
    M = torch.exp((X @ A.T) @ X.T)
    z = M.sum(dim=-1)
    expected_gf = -float(n) * (grad_u @ B.T) / z[:, None]
    torch.testing.assert_close(gf, expected_gf, atol=2e-11, rtol=2e-11)

    rhs = softmax_metric_hamiltonian_rhs(X, Y, A, B, grad_u, alpha=0.2)
    kinetic = 0.5 * float(n) * torch.sum(torch.sum(Y * (Y @ B.T), dim=-1) / z)
    potential = 0.5 * torch.sum(X * grad_u)
    dH_dX, dH_dY = torch.autograd.grad(kinetic + potential, (X, Y), create_graph=True)
    torch.testing.assert_close(rhs.dX, dH_dY, atol=2e-11, rtol=2e-11)
    torch.testing.assert_close(rhs.dY, -dH_dX - 0.2 * Y, atol=2e-11, rtol=2e-11)

    # At Y=0, acceleration is exactly the metric gradient-flow direction.
    Y0 = torch.zeros_like(X)
    rhs0 = softmax_metric_hamiltonian_rhs(X, Y0, A, B, grad_u, alpha=0.2)
    def position_rhs(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return softmax_metric_hamiltonian_rhs(x, y, A, B, x @ H.T, alpha=0.2).dX
    _, acceleration = torch.func.jvp(position_rhs, (X, Y0), (rhs0.dX, rhs0.dY))
    torch.testing.assert_close(acceleration, gf, atol=3e-11, rtol=3e-11)

    # Independent NumPy implementation agrees with torch.
    xn, yn, an, bn, hn = (q.detach().numpy() for q in (X, Y, A, B, H))
    gf_np = metric_gradient_rhs(xn[None], an, bn, hn)[0]
    dx_np, dy_np = metric_hamiltonian_rhs(xn[None], yn[None], an, bn, hn, 0.2)
    np.testing.assert_allclose(gf_np, gf.detach().numpy(), atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(dx_np[0], rhs.dX.detach().numpy(), atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(dy_np[0], rhs.dY.detach().numpy(), atol=2e-11, rtol=2e-11)


def verify_tier1_integration() -> None:
    rng = np.random.default_rng(7)
    X0 = rng.normal(size=(2, 4, 2)) * np.array([0.2, 0.02])
    A = np.zeros((2, 2))
    B = np.eye(2)
    H = np.diag([0.01, 1.0])
    dt, horizon, alpha = 0.01, 2.0, 0.2
    gf = integrate_gradient(X0, A=A, B=B, H=H, dt=dt, horizon=horizon, keep_states=True)
    acc = integrate_hamiltonian(X0, A=A, B=B, H=H, alpha=alpha, dt=dt, horizon=horizon, keep_states=True)
    times = gf.times[:, None, None, None]
    exact_gf = X0[None] * np.exp(-times * np.array([0.01, 1.0]))
    np.testing.assert_allclose(gf.states, exact_gf, atol=2e-10, rtol=2e-10)

    slow = X0[..., 0][None] * (1.0 + 0.1 * gf.times[:, None, None]) * np.exp(-0.1 * gf.times[:, None, None])
    omega = np.sqrt(1.0 - 0.1**2)
    fast_factor = np.exp(-0.1 * gf.times) * (
        np.cos(omega * gf.times) + (0.1 / omega) * np.sin(omega * gf.times)
    )
    exact_acc = np.empty_like(acc.states)
    exact_acc[..., 0] = slow
    exact_acc[..., 1] = X0[None, ..., 1] * fast_factor[:, None, None]
    np.testing.assert_allclose(acc.states, exact_acc, atol=3e-9, rtol=3e-9)


def verify_artifacts(artifact_dir: Path) -> None:
    summary = json.loads((artifact_dir / "summary.json").read_text())
    rows = list(csv.DictReader((artifact_dir / "results.csv").open()))
    assert len(rows) == 3 * 2 * 2 * 5
    assert summary["success_gate"] is True, summary
    assert summary["status"] == "pass"
    for item in summary["summaries"]:
        if item["epsilon"] > 0:
            assert item["paired_wins"] == 5
            assert item["mean_reduction"] >= 0.25
            assert item["max_refinement_difference"] <= 0.25
    for row in rows:
        assert float(row["max_relative_energy_increase"]) <= 2e-10, row
        assert np.isfinite(float(row["hit_1em03"]))
    print(
        "PASS: E021 artifacts; "
        + "; ".join(
            f"eps={item['epsilon']:.2f}: {item['paired_wins']}/5 wins, "
            f"reduction={item['mean_reduction']:.1%}"
            for item in summary["summaries"]
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact_dir", type=Path,
        default=ROOT / "artifacts" / "genuine_particle_acceleration",
    )
    parser.add_argument("--formulas_only", action="store_true")
    args = parser.parse_args()
    verify_formulas()
    verify_tier1_integration()
    print("PASS: E021 formulas, initial acceleration, NumPy parity, and analytic RK4 control")
    if not args.formulas_only:
        verify_artifacts(args.artifact_dir)


if __name__ == "__main__":
    main()
