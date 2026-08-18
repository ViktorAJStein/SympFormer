#!/usr/bin/env python3
"""Plot quantified mechanisms that break ideal acceleration in the decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("trajectories", type=Path)
    parser.add_argument("--out", type=Path, default=Path("images/e022_acceleration_loss_mechanisms"))
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    data = np.load(args.trajectories)

    times = data["times_eps0p20_gradient"]
    gf = data["objective_ratio_eps0p20_gradient"].mean(axis=1)
    acc = data["objective_ratio_eps0p20_accelerated"].mean(axis=1)
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 3.7))

    ax = axes[0]
    ax.plot(times, gf, label="Gradient flow", color="tab:blue")
    ax.plot(times, acc, label="Accelerated flow", color="tab:orange")
    ax.axvspan(*summary["h001"]["duration_range"], color="tab:green", alpha=0.18, label="H001 depth-time")
    ax.axvspan(*summary["h004"]["duration_range"], color="tab:purple", alpha=0.18, label="H004 depth-time")
    cross = summary["positive_control_short_horizon"]["first_mean_crossover"]
    ax.axvline(cross, color="black", linestyle="--", linewidth=1, label="positive-control crossover")
    ax.set_xlim(0, 5)
    ax.set_ylim(0.35, 1.02)
    ax.set_xlabel("Depth-time")
    ax.set_ylabel("Mean objective ratio")
    ax.set_title("Cold-start horizon mismatch")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    campaigns = ["H001", "H004"]
    retention = [100 * summary[name.lower()]["nominal_cumulative_damping_mean"] for name in campaigns]
    ax.bar(campaigns, retention, color=["tab:green", "tab:purple"], alpha=0.8)
    ax.set_ylabel("Nominal momentum retained (%)")
    ax.set_ylim(0, 20)
    ax.set_title("Damping across eight layers")
    for i, value in enumerate(retention):
        ax.text(i, value + 0.5, f"{value:.1f}%", ha="center", fontsize=8)
    raw = 100 * summary["layernorm_scale_witness"]["raw_relative_change"]
    post = 100 * summary["layernorm_scale_witness"]["post_layernorm_relative_change"]
    ax.text(0.5, 2.0, f"Per-layer scalar change: {raw:.1f}%\nAfter LayerNorm: {post:.4f}%", ha="center", fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[2]
    jac = np.asarray(summary["causal_curl_witness"]["jacobian"])
    image = ax.imshow(jac, cmap="coolwarm", vmin=-np.abs(jac).max(), vmax=np.abs(jac).max())
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{jac[i,j]:.3f}", ha="center", va="center", color="black")
    ax.set_xticks([0, 1], [r"$x_1$", r"$x_2$"])
    ax.set_yticks([0, 1], [r"$G_1$", r"$G_2$"])
    ax.set_title("Causal field Jacobian is asymmetric")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=180, bbox_inches="tight")
    print(f"wrote {args.out.with_suffix('.pdf')} and {args.out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
