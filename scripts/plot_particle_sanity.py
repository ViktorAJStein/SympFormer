#!/usr/bin/env python3
"""Plot acceleration-identity error and nonzero-momentum direction cosines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from particle_schemes import (  # noqa: E402
    sympformer_initial_acceleration,
    sympformer_softmax_particle_rhs,
    vanilla_softmax_particle_rhs,
)


def spd(d: int, g: torch.Generator) -> torch.Tensor:
    C = torch.randn(d, d, generator=g, dtype=torch.float64)
    return C @ C.T / d + 0.5 * torch.eye(d, dtype=torch.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("images/e019_particle_acceleration"))
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()

    g = torch.Generator().manual_seed(20260806)
    ns = [1, 2, 4, 8, 16]
    errors: list[tuple[int, float]] = []
    cosines: list[tuple[int, float]] = []
    for n in ns:
        for _ in range(args.trials):
            d = 4
            A, B = spd(d, g), spd(d, g)
            V = B @ A
            X = 0.1 * torch.randn(n, d, generator=g, dtype=torch.float64)
            Y = 0.1 * torch.randn(n, d, generator=g, dtype=torch.float64)
            gamma = vanilla_softmax_particle_rhs(X, A, V)
            acceleration = sympformer_initial_acceleration(X, A, V)
            expected = float(n * n) * gamma
            rel = (acceleration - expected).norm() / expected.norm().clamp_min(1e-30)
            errors.append((n, rel.item()))
            dX = sympformer_softmax_particle_rhs(X, Y, A, V, alpha=0.3).dX
            cosine = torch.nn.functional.cosine_similarity(gamma.flatten(), dX.flatten(), dim=0)
            cosines.append((n, cosine.item()))

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    for n in ns:
        vals = [value for nn, value in errors if nn == n]
        axes[0].scatter([n] * len(vals), vals, alpha=0.55, s=16)
    axes[0].set_yscale("log")
    axes[0].set_xticks(ns)
    axes[0].set_xlabel("Particle count $N$")
    axes[0].set_ylabel(r"Relative error in $\ddot X(0)=N^2\Gamma(X_0)$")
    axes[0].set_title("Zero-momentum acceleration identity")
    axes[0].grid(alpha=0.25)

    rng = np.random.default_rng(20260806)
    for n in ns:
        vals = [value for nn, value in cosines if nn == n]
        jitter = rng.normal(0.0, 0.045, len(vals))
        axes[1].scatter(np.asarray([n] * len(vals)) + jitter, vals, alpha=0.55, s=16)
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1, label="same direction")
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].set_xticks(ns)
    axes[1].set_xlabel("Particle count $N$")
    axes[1].set_ylabel(r"cosine$(\dot X_{\rm Symp},\Gamma)$")
    axes[1].set_title("Arbitrary nonzero momentum")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, loc="lower right")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=180, bbox_inches="tight")
    print(f"wrote {args.out.with_suffix('.pdf')} and {args.out.with_suffix('.png')}")
    print(f"max relative identity error={max(value for _, value in errors):.3e}")
    print(f"nonzero-momentum cosine range=[{min(value for _, value in cosines):.3f}, {max(value for _, value in cosines):.3f}]")


if __name__ == "__main__":
    main()
