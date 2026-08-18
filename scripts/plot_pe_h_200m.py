#!/usr/bin/env python3
"""Plot curves and paired differences for the PE initial-step bracket."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_pe_h_200m import INCUMBENT_H, REPLACE_THRESHOLD, analyze
from analyze_pe_tuning import read_metrics
from make_pe_h_200m_specs import H_VALUES, read_lr_lock


def save(fig, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {base.with_suffix('.png')} and {base.with_suffix('.pdf')}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--lock_file", type=Path)
    parser.add_argument("--out_prefix", type=Path)
    args = parser.parse_args()
    lock = read_lr_lock(args.lock_file or args.root / "locked_lrs_input.json")
    prefix = args.out_prefix or args.root / "analysis" / "pe_h200"
    runs, pairs, aggregate, _ = analyze(args.root, lock)

    grouped = defaultdict(list)
    for row in runs:
        if row.get("status") != "valid" or not row.get("run_dir") or row.get("h_init") == "":
            continue
        curve = read_metrics(Path(row["run_dir"]) / "metrics.csv")
        if curve:
            grouped[float(row["h_init"])].append(curve)
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    plotted = 0
    for index, h_init in enumerate(H_VALUES):
        curves = grouped.get(h_init, [])
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
        color = "black" if h_init == INCUMBENT_H else plt.get_cmap("tab10")(index)
        ax.plot(x, mean, lw=2.2, color=color, label=f"h={h_init:g}")
        if len(curves) > 1:
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.14)
        plotted += 1
    ax.set_xlabel("Training tokens (millions)")
    ax.set_ylabel("Validation NLL (nats/token)")
    ax.set_title("200M PE initial-step bracket at locked optimizer settings")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    if plotted:
        save(fig, Path(f"{prefix}_curves"))
    else:
        plt.close(fig)

    available = [row for row in aggregate if row["n_valid"]]
    if available:
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        y = np.arange(len(available))
        means = [float(row["mean_delta_final_val"]) for row in available]
        colors = ["#238b45" if row["selected"] else "#3182bd" for row in available]
        ax.barh(y, means, color=colors, alpha=0.82)
        by_h = defaultdict(list)
        for row in pairs:
            if row["valid_pair"]:
                by_h[float(row["h_init"])].append(float(row["delta_final_val"]))
        for yi, row in zip(y, available):
            values = by_h[float(row["h_init"])]
            ax.scatter(values, np.full(len(values), yi), color="black", s=28, zorder=3)
        ax.axvline(0, color="black", lw=1)
        ax.axvline(REPLACE_THRESHOLD, color="#cb181d", ls="--", lw=1.2, label="replacement threshold")
        ax.set_yticks(y, [f"h={float(row['h_init']):g}" for row in available])
        ax.invert_yaxis()
        ax.set_xlabel("Paired final-NLL difference vs h=0.1")
        ax.set_title("PE initial-step replacement decision")
        ax.grid(axis="x", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        save(fig, Path(f"{prefix}_paired"))


if __name__ == "__main__":
    main()
