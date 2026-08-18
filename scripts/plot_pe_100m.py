#!/usr/bin/env python3
"""Generate PDF+PNG diagnostics for the 100M PE backfill screen."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_pe_100m import ADVANCE_THRESHOLD, analyze
from analyze_pe_tuning import finite, read_metrics


PANELS = [
    (
        "Integrator step initialization",
        ["pe100_anchor", "pe100_h0p03", "pe100_h0p2", "pe100_h0p3"],
        ["anchor h=0.1", "h=0.03", "h=0.2", "h=0.3"],
    ),
    (
        "Log-damping initialization",
        ["pe100_anchor", "pe100_clog1", "pe100_clog2", "pe100_clog4"],
        ["anchor c_log=3", "c_log=1", "c_log=2", "c_log=4"],
    ),
    (
        "Learned-scalar LR multiplier",
        ["pe100_anchor", "pe100_scalar1", "pe100_scalar10", "pe100_scalar20"],
        ["anchor x5", "x1", "x10", "x20"],
    ),
    (
        "Muon matrix LR",
        ["pe100_anchor", "pe100_muon0p005", "pe100_muon0p01", "pe100_muon0p04"],
        ["anchor 0.02", "0.005", "0.01", "0.04"],
    ),
    (
        "AdamW-group LR (scalar LR fixed)",
        ["pe100_anchor", "pe100_adam0p0003", "pe100_adam0p001", "pe100_adam0p0015"],
        ["anchor 0.0006", "0.0003", "0.001", "0.0015"],
    ),
    (
        "Structural controls",
        ["pe100_anchor", "pe100_lnp_none", "pe100_h_fixed", "pe100_eta_fixed"],
        ["anchor", "no momentum LN", "fixed h", "fixed damping"],
    ),
]


def save(fig, base):
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {base.with_suffix('.png')} and {base.with_suffix('.pdf')}")
    plt.close(fig)


def curves_by_candidate(runs):
    curves = {}
    for row in runs:
        if row["status"] != "valid" or not row["run_dir"]:
            continue
        curve = read_metrics(Path(row["run_dir"]) / "metrics.csv")
        if curve:
            curves[row["candidate"]] = curve
    return curves


def plot_curves(runs, prefix):
    curves = curves_by_candidate(runs)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True, sharey=True)
    colors = plt.get_cmap("tab10")
    plotted = 0
    for ax, (title, candidates, labels) in zip(axes.flat, PANELS):
        for index, (candidate, label) in enumerate(zip(candidates, labels)):
            curve = curves.get(candidate)
            if not curve:
                continue
            x = np.asarray([row["tokens"] for row in curve]) / 1e6
            y = np.asarray([row["val"] for row in curve])
            color = "black" if candidate == "pe100_anchor" else colors(index)
            width = 2.2 if candidate == "pe100_anchor" else 1.7
            ax.plot(x, y, color=color, lw=width, marker="o", ms=3, label=label)
            plotted += 1
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Training tokens (millions)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Validation NLL (nats/token)")
    fig.suptitle("Single-seed 100M causal-PE pruning screen", y=1.01)
    fig.tight_layout()
    if plotted:
        save(fig, f"{prefix}_curves")
    else:
        plt.close(fig)
        print("skipped curves: no valid runs")


def plot_paired(pairs, prefix):
    available = [row for row in pairs if row["valid_pair"]]
    if not available:
        print("skipped paired plot: no valid PE pairs")
        return
    available.sort(key=lambda row: float(row["delta_final_val"]))
    y = np.arange(len(available))
    values = np.asarray([float(row["delta_final_val"]) for row in available])
    colors = [
        "#238b45" if row["advance_to_200m"] else "#3182bd"
        for row in available
    ]
    fig, ax = plt.subplots(figsize=(9.5, max(5.5, 0.42 * len(available))))
    ax.barh(y, values, color=colors, alpha=0.8)
    ax.axvline(0, color="black", lw=1)
    ax.axvline(
        ADVANCE_THRESHOLD,
        color="#cb181d",
        lw=1.2,
        ls="--",
        label="advance threshold",
    )
    ax.set_yticks(
        y,
        [row["candidate"].removeprefix("pe100_") for row in available],
    )
    ax.invert_yaxis()
    ax.set_xlabel("Final-NLL difference vs PE anchor (lower is better)")
    ax.set_title("100M PE pruning results; green advances to 200M")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save(fig, f"{prefix}_paired")


def plot_controls(controls, prefix):
    available = [
        row
        for row in controls
        if row["status"] == "valid" and finite(row["final_val"])
    ]
    if not available:
        print("skipped controls plot: no valid control results")
        return
    labels = [row["candidate"].removeprefix("pe100_") for row in available]
    values = [float(row["final_val"]) for row in available]
    colors = ["#636363", "#756bb1", "#3182bd"][: len(available)]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0, max(values) + 0.10)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.005,
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )
    ax.set_ylabel("Final validation NLL (nats/token)")
    ax.set_title("100M single-seed controls")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save(fig, f"{prefix}_controls")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--out_prefix", type=Path)
    args = parser.parse_args()
    prefix = args.out_prefix or args.root / "analysis" / "pe100"
    runs, pairs, controls = analyze(args.root)
    plot_curves(runs, prefix)
    plot_paired(pairs, prefix)
    plot_controls(controls, prefix)


if __name__ == "__main__":
    main()
