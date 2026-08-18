#!/usr/bin/env python3
"""Create publication-ready diagnostics for the causal-PE tuning campaign."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_pe_tuning import analyze, finite, read_metrics


PANELS = [
    (
        "Integrator step initialization",
        ["pe350_anchor", "pe350_h0p03", "pe350_h0p2", "pe350_h0p3"],
        {"pe350_anchor": "anchor h=0.1", "pe350_h0p03": "h=0.03",
         "pe350_h0p2": "h=0.2", "pe350_h0p3": "h=0.3"},
    ),
    (
        "Log-damping initialization",
        ["pe350_anchor", "pe350_clog1", "pe350_clog2", "pe350_clog4"],
        {"pe350_anchor": "anchor c_log=3", "pe350_clog1": "c_log=1",
         "pe350_clog2": "c_log=2", "pe350_clog4": "c_log=4"},
    ),
    (
        "Learned-scalar LR multiplier",
        ["pe350_anchor", "pe350_scalar1", "pe350_scalar10", "pe350_scalar20"],
        {"pe350_anchor": "anchor x5", "pe350_scalar1": "x1",
         "pe350_scalar10": "x10", "pe350_scalar20": "x20"},
    ),
    (
        "Muon matrix LR",
        ["pe350_anchor", "pe350_muon0p005", "pe350_muon0p01", "pe350_muon0p04"],
        {"pe350_anchor": "anchor 0.02", "pe350_muon0p005": "0.005",
         "pe350_muon0p01": "0.01", "pe350_muon0p04": "0.04"},
    ),
    (
        "AdamW-group LR (scalar LR fixed)",
        ["pe350_anchor", "pe350_adam0p0003", "pe350_adam0p001", "pe350_adam0p0015"],
        {"pe350_anchor": "anchor 0.0006", "pe350_adam0p0003": "0.0003",
         "pe350_adam0p001": "0.001", "pe350_adam0p0015": "0.0015"},
    ),
    (
        "Structural controls",
        ["pe350_anchor", "pe350_lnp_none", "pe350_h_fixed", "pe350_eta_fixed"],
        {"pe350_anchor": "anchor", "pe350_lnp_none": "no momentum LN",
         "pe350_h_fixed": "fixed h", "pe350_eta_fixed": "fixed damping"},
    ),
]


def curve_groups(runs):
    groups = defaultdict(list)
    for row in runs:
        if row["status"] != "valid" or not row["run_dir"]:
            continue
        curve = read_metrics(Path(row["run_dir"]) / "metrics.csv")
        if curve:
            groups[row["candidate"]].append(curve)
    return groups


def mean_curve(curves):
    shared = set(row["tokens"] for row in curves[0])
    for curve in curves[1:]:
        shared &= {row["tokens"] for row in curve}
    tokens = sorted(shared)
    if not tokens:
        return np.array([]), np.array([]), np.array([])
    values = []
    for curve in curves:
        by_token = {row["tokens"]: row["val"] for row in curve}
        values.append([by_token[token] for token in tokens])
    array = np.asarray(values, dtype=float)
    return np.asarray(tokens) / 1e6, array.mean(axis=0), array.std(axis=0)


def save(fig, base: Path):
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {base.with_suffix('.png')} and {base.with_suffix('.pdf')}")
    plt.close(fig)


def plot_curves(runs, prefix: Path):
    groups = curve_groups(runs)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True, sharey=True)
    colors = plt.get_cmap("tab10")
    plotted = 0
    for ax, (title, candidates, labels) in zip(axes.flat, PANELS):
        for index, candidate in enumerate(candidates):
            curves = groups.get(candidate, [])
            if not curves:
                continue
            tokens, mean, std = mean_curve(curves)
            if not len(tokens):
                continue
            color = "black" if candidate == "pe350_anchor" else colors(index)
            width = 2.2 if candidate == "pe350_anchor" else 1.7
            ax.plot(tokens, mean, color=color, lw=width, label=f"{labels[candidate]} (n={len(curves)})")
            if len(curves) > 1:
                ax.fill_between(tokens, mean - std, mean + std, color=color, alpha=0.12)
            plotted += 1
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Training tokens (millions)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Validation NLL (nats/token)")
    fig.suptitle("Causal presymplectic-Euler tuning: mean across predeclared seeds", y=1.01)
    fig.tight_layout()
    if plotted:
        save(fig, Path(f"{prefix}_curves"))
    else:
        plt.close(fig)
        print("skipped curves: no valid 350M runs")


def plot_paired(pairs, aggregate, prefix: Path):
    available = [row for row in aggregate if row["n_valid"]]
    if not available:
        print("skipped paired deltas: no valid 350M pairs")
        return
    available.sort(key=lambda row: row["mean_delta_final_val"])
    by_candidate = defaultdict(list)
    for row in pairs:
        if row["valid_pair"]:
            by_candidate[row["candidate"]].append(float(row["delta_final_val"]))
    height = max(5.5, 0.42 * len(available))
    fig, ax = plt.subplots(figsize=(9.5, height))
    y = np.arange(len(available))
    means = np.array([row["mean_delta_final_val"] for row in available])
    ax.barh(y, means, color=["#2b8cbe" if value < 0 else "#d95f0e" for value in means], alpha=0.75)
    for yi, row in zip(y, available):
        values = by_candidate[row["candidate"]]
        ax.scatter(values, np.full(len(values), yi), color="black", s=22, zorder=3)
        if row["qualifies_for_confirmation"]:
            ax.text(max(0.0005, row["mean_delta_final_val"] + 0.0005), yi, "qualifies",
                    va="center", fontsize=8, fontweight="bold")
    ax.axvline(0, color="black", lw=1)
    ax.axvline(-0.005, color="#238b45", lw=1.2, ls="--", label="mean qualification threshold")
    ax.set_yticks(y, [row["candidate"].removeprefix("pe350_") for row in available])
    ax.invert_yaxis()
    ax.set_xlabel("Paired final-NLL difference vs same-seed PE anchor (lower is better)")
    ax.set_title("350M-token causal-PE screen")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, Path(f"{prefix}_paired"))


def plot_directional(rows, prefix: Path):
    available = [row for row in rows if row["status"] == "valid" and finite(row["final_val"])]
    if not available:
        print("skipped directional bars: no valid 1B runs")
        return
    labels = [row["candidate"].removeprefix("pe1b_") for row in available]
    values = [row["final_val"] for row in available]
    baseline = next(
        (row["final_val"] for row in available if row["candidate"] == "pe1b_baseline"),
        None,
    )
    colors = ["#636363" if label == "baseline" else "#3182bd" for label in labels]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    bars = ax.bar(labels, values, color=colors)
    if baseline is not None:
        ax.axhline(baseline, color="black", lw=1, ls="--", label="baseline")
    # Bars start at zero so the absolute NLL scale is not visually exaggerated.
    ax.set_ylim(0, max(values) + 0.10)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.001, f"{value:.4f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Final validation NLL (nats/token)")
    ax.set_title("1B-token directional check, one predeclared seed")
    ax.grid(axis="y", alpha=0.25)
    if baseline is not None:
        ax.legend()
    fig.tight_layout()
    save(fig, Path(f"{prefix}_directional"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--out_prefix", type=Path)
    args = parser.parse_args()
    prefix = args.out_prefix or args.root / "analysis" / "pe_tuning"
    runs, pairs, aggregate, directional = analyze(args.root)
    plot_curves(runs, prefix)
    plot_paired(pairs, aggregate, prefix)
    plot_directional(directional, prefix)


if __name__ == "__main__":
    main()
