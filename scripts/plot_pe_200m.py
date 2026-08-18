#!/usr/bin/env python3
"""Generate PDF+PNG diagnostics for the 200M paired PE confirmation."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_pe_200m import CONFIRM_THRESHOLD, analyze
from analyze_pe_tuning import read_metrics


LABELS = {
    "pe200_anchor": "PE anchor",
    "pe200_adam0p001": "AdamW LR 0.001",
    "pe200_adam0p0015": "AdamW LR 0.0015",
    "pe200_muon0p04": "Muon LR 0.04",
}


def save(fig, base):
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {base.with_suffix('.png')} and {base.with_suffix('.pdf')}")
    plt.close(fig)


def plot_curves(runs, prefix):
    grouped = defaultdict(list)
    for row in runs:
        if row["status"] != "valid" or not row["run_dir"]:
            continue
        curve = read_metrics(Path(row["run_dir"]) / "metrics.csv")
        if curve:
            grouped[row["candidate"]].append(curve)
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    plotted = 0
    for index, candidate in enumerate(LABELS):
        curves = grouped.get(candidate, [])
        if not curves:
            continue
        shared = set(row["tokens"] for row in curves[0])
        for curve in curves[1:]:
            shared &= {row["tokens"] for row in curve}
        tokens = sorted(shared)
        if not tokens:
            continue
        values = np.asarray(
            [[{row['tokens']: row['val'] for row in curve}[token] for token in tokens] for curve in curves]
        )
        x = np.asarray(tokens) / 1e6
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        color = "black" if candidate == "pe200_anchor" else plt.get_cmap("tab10")(index)
        ax.plot(x, mean, lw=2.2, color=color, label=f"{LABELS[candidate]} (n={len(curves)})")
        if len(curves) > 1:
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.14)
        plotted += 1
    if not plotted:
        plt.close(fig)
        print("skipped curves: no valid runs")
        return
    ax.set_xlabel("Training tokens (millions)")
    ax.set_ylabel("Validation NLL (nats/token)")
    ax.set_title("200M causal-PE confirmation across fresh seeds")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save(fig, f"{prefix}_curves")


def plot_paired(pairs, aggregate, prefix):
    available = [row for row in aggregate if row["n_valid"]]
    if not available:
        print("skipped paired plot: no valid pairs")
        return
    available.sort(key=lambda row: float(row["mean_delta_final_val"]))
    by_candidate = defaultdict(list)
    for row in pairs:
        if row["valid_pair"]:
            by_candidate[row["candidate"]].append(float(row["delta_final_val"]))
    y = np.arange(len(available))
    means = np.asarray([float(row["mean_delta_final_val"]) for row in available])
    colors = ["#238b45" if row["confirms_at_200m"] else "#3182bd" for row in available]
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.barh(y, means, color=colors, alpha=0.8)
    for yi, row in zip(y, available):
        values = by_candidate[row["candidate"]]
        ax.scatter(values, np.full(len(values), yi), color="black", s=28, zorder=3)
    ax.axvline(0, color="black", lw=1)
    ax.axvline(CONFIRM_THRESHOLD, color="#cb181d", lw=1.2, ls="--", label="confirmation threshold")
    ax.set_yticks(y, [LABELS[row["candidate"]] for row in available])
    ax.invert_yaxis()
    ax.set_xlabel("Paired final-NLL difference vs PE anchor (lower is better)")
    ax.set_title("200M fresh-seed paired confirmation")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save(fig, f"{prefix}_paired")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--out_prefix", type=Path)
    args = parser.parse_args()
    prefix = args.out_prefix or args.root / "analysis" / "pe200"
    runs, pairs, aggregate = analyze(args.root)
    plot_curves(runs, prefix)
    plot_paired(pairs, aggregate, prefix)


if __name__ == "__main__":
    main()
