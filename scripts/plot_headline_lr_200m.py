#!/usr/bin/env python3
"""Plot learning curves and locked means for the headline LR campaign."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_headline_lr_200m import analyze
from analyze_pe_tuning import read_metrics
from make_headline_lr_200m_specs import METHOD_LRS


METHOD_LABELS = {
    "baseline": "Baseline",
    "yurii_lt": "YuriiFormer",
    "causal_symp_pe": "Causal PE",
}


def save(fig, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {base.with_suffix('.png')} and {base.with_suffix('.pdf')}")
    plt.close(fig)


def plot_curves(runs: list[dict], prefix: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), sharey=True)
    grouped = defaultdict(list)
    for row in runs:
        if row.get("status") != "valid" or not row.get("run_dir"):
            continue
        curve = read_metrics(Path(row["run_dir"]) / "metrics.csv")
        if curve:
            grouped[(row["method"], float(row["peak_lr"]))].append(curve)
    plotted = 0
    for ax, method in zip(axes, METHOD_LRS):
        for index, peak_lr in enumerate(METHOD_LRS[method]):
            curves = grouped.get((method, peak_lr), [])
            if not curves:
                continue
            shared = set(row["tokens"] for row in curves[0])
            for curve in curves[1:]:
                shared &= {row["tokens"] for row in curve}
            tokens = sorted(shared)
            values = np.asarray(
                [[{row['tokens']: row['val'] for row in curve}[token] for token in tokens] for curve in curves]
            )
            x = np.asarray(tokens) / 1e6
            mean = values.mean(axis=0)
            std = values.std(axis=0)
            color = plt.get_cmap("viridis")(index / max(1, len(METHOD_LRS[method]) - 1))
            ax.plot(x, mean, lw=2, color=color, label=f"LR {peak_lr:g}")
            if len(curves) > 1:
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.13)
            plotted += 1
        ax.set_title(METHOD_LABELS[method])
        ax.set_xlabel("Training tokens (millions)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Validation NLL (nats/token)")
    fig.suptitle("Equal-budget 200M learning-rate selection")
    fig.tight_layout()
    if plotted:
        save(fig, Path(f"{prefix}_curves"))
    else:
        plt.close(fig)
        print("skipped curves: no valid runs")


def plot_means(aggregate: list[dict], prefix: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    plotted = 0
    for ax, method in zip(axes, METHOD_LRS):
        rows = [row for row in aggregate if row["method"] == method and row["n_valid"]]
        rows.sort(key=lambda row: float(row["peak_lr"]))
        if rows:
            x = np.arange(len(rows))
            means = [float(row["mean_final_val"]) for row in rows]
            stds = [float(row["std_final_val"]) for row in rows]
            colors = ["#238b45" if row["selected"] else "#3182bd" for row in rows]
            ax.bar(x, means, yerr=stds, color=colors, alpha=0.82, capsize=4)
            ax.set_xticks(x, [f"{float(row['peak_lr']):g}" for row in rows])
            plotted += 1
        ax.set_title(METHOD_LABELS[method])
        ax.set_xlabel("AdamW-group peak LR")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean final validation NLL")
    fig.suptitle("Locked LR is green; error bars show two-seed sample SD")
    fig.tight_layout()
    if plotted:
        save(fig, Path(f"{prefix}_means"))
    else:
        plt.close(fig)
        print("skipped means: no valid runs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--out_prefix", type=Path)
    args = parser.parse_args()
    prefix = args.out_prefix or args.root / "analysis" / "headline_lr200"
    runs, aggregate, _ = analyze(args.root)
    plot_curves(runs, prefix)
    plot_means(aggregate, prefix)


if __name__ == "__main__":
    main()
