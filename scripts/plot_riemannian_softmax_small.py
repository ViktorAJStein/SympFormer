#!/usr/bin/env python3
"""Plot E025 causal softmax endpoint and throughput diagnostics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LABELS = {
    "baseline": "Baseline",
    "yurii_lt": "Yurii",
    "causal_symp_fe": "Causal FE",
    "causal_symp_pe": "Causal PE",
    "causal_symp_exp_pe": "Causal ExpPE",
    "causal_symp_ab2": "Causal AB2",
    "causal_riem_nag_noconn": "Riem NAG no-conn",
    "causal_riem_nag": "Riem NAG + conn",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis_dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("images/e025_riemannian_softmax_small"))
    args = parser.parse_args()
    runs = list(csv.DictReader((args.analysis_dir / "runs.csv").open()))
    methods = list(LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for index, method in enumerate(methods):
        selected = [row for row in runs if row["arch"] == method and row["status"] == "valid"]
        vals = np.array([float(row["final_val"]) for row in selected])
        speed = np.array([float(row["tokens_per_second"]) for row in selected])
        axes[0].scatter([index] * len(vals), vals, color="tab:blue", zorder=3)
        axes[0].plot([index - 0.25, index + 0.25], [vals.mean(), vals.mean()], color="black")
        axes[1].bar(index, speed.mean(), color="tab:orange", alpha=0.8)
        axes[1].scatter([index] * len(speed), speed, color="black", s=14, zorder=3)
    labels = [LABELS[method] for method in methods]
    for ax in axes:
        ax.set_xticks(range(len(methods)), labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Final validation NLL")
    axes[0].set_title("Strictly causal endpoint")
    axes[1].set_ylabel("Training tokens/s")
    axes[1].set_title("Throughput")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=180, bbox_inches="tight")
    print(f"wrote {args.out.with_suffix('.pdf')} and {args.out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
