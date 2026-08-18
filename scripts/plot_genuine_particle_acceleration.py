#!/usr/bin/env python3
"""Plot E021 objective trajectories and paired sustained hitting times."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("images/e021_genuine_particle_acceleration"))
    args = parser.parse_args()

    data = np.load(args.artifact_dir / "trajectories.npz")
    rows = list(csv.DictReader((args.artifact_dir / "results.csv").open()))
    epsilons = [0.0, 0.05, 0.20]
    labels = {"gradient": "Metric gradient flow", "accelerated": "Damped Hamiltonian flow"}
    colors = {"gradient": "tab:blue", "accelerated": "tab:orange"}

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.0))
    for ax, epsilon in zip(axes.flat[:3], epsilons):
        for method in ("gradient", "accelerated"):
            tag = f"eps{epsilon:.2f}_{method}".replace(".", "p")
            times = data[f"times_{tag}"]
            ratios = data[f"objective_ratio_{tag}"]
            mean = ratios.mean(axis=1)
            ax.semilogy(times, mean, color=colors[method], label=labels[method])
            ax.fill_between(times, ratios.min(axis=1), ratios.max(axis=1), color=colors[method], alpha=0.15)
        ax.axhline(1e-3, color="black", linestyle="--", linewidth=1, label=r"$10^{-3}$ threshold")
        ax.set_xlim(0, 360)
        ax.set_ylim(1e-8, 1.2)
        ax.set_title(rf"Softmax metric strength $\varepsilon={epsilon:.2f}$")
        ax.set_xlabel("Continuous time")
        ax.set_ylabel(r"Objective ratio $U(X(t))/U(X(0))$")
        ax.grid(alpha=0.25)
    axes.flat[0].legend(frameon=False, fontsize=8)

    ax = axes.flat[3]
    reference = [row for row in rows if float(row["dt"]) == 0.025]
    x_positions = np.arange(len(epsilons), dtype=float)
    width = 0.28
    for offset, method in [(-width / 2, "gradient"), (width / 2, "accelerated")]:
        values_by_eps = []
        for epsilon in epsilons:
            selected = sorted(
                [row for row in reference if float(row["epsilon"]) == epsilon and row["method"] == method],
                key=lambda row: int(row["seed"]),
            )
            values = np.array([float(row["hit_1em03"]) for row in selected])
            values_by_eps.append(values)
        means = [values.mean() for values in values_by_eps]
        ax.bar(x_positions + offset, means, width=width, color=colors[method], alpha=0.8, label=labels[method])
        for index, values in enumerate(values_by_eps):
            ax.scatter(np.full(values.size, x_positions[index] + offset), values, color="black", s=12, zorder=3)
    ax.set_xticks(x_positions, [f"{epsilon:.2f}" for epsilon in epsilons])
    ax.set_xlabel(r"Softmax metric strength $\varepsilon$")
    ax.set_ylabel(r"Sustained $10^{-3}$ hitting time")
    ax.set_title("Paired five-seed acceleration")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=180, bbox_inches="tight")
    print(f"wrote {args.out.with_suffix('.pdf')} and {args.out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
