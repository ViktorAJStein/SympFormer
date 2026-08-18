#!/usr/bin/env python3
"""Plot learning curves and paired decisions for one PE damping stage."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_pe_damping_200m import analyze
from analyze_pe_tuning import read_metrics
from make_pe_damping_200m_specs import (
    INCUMBENTS,
    REPLACE_THRESHOLD,
    STAGES,
    read_stage_lock,
    stage_values,
)


def save(fig, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {base.with_suffix('.png')} and {base.with_suffix('.pdf')}")
    plt.close(fig)


def key(value) -> str:
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--lock_file", type=Path, required=True)
    parser.add_argument("--out_prefix", type=Path, required=True)
    args = parser.parse_args()
    lock = read_stage_lock(args.lock_file, args.stage)
    runs, pairs, aggregate, _ = analyze(args.root, args.stage, lock)

    grouped = defaultdict(list)
    for row in runs:
        if row.get("status") != "valid" or not row.get("run_dir") or row.get("setting") == "":
            continue
        curve = read_metrics(Path(row["run_dir"]) / "metrics.csv")
        if curve:
            grouped[key(row["setting"])].append(curve)
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    plotted = 0
    incumbent = INCUMBENTS[args.stage]
    for index, setting in enumerate(stage_values(args.stage)):
        curves = grouped.get(key(setting), [])
        if not curves:
            continue
        shared = set(row["tokens"] for row in curves[0])
        for curve in curves[1:]:
            shared &= {row["tokens"] for row in curve}
        tokens = sorted(shared)
        values = np.asarray(
            [
                [{row['tokens']: row['val'] for row in curve}[token] for token in tokens]
                for curve in curves
            ]
        )
        x = np.asarray(tokens) / 1e6
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        color = "black" if key(setting) == key(incumbent) else plt.get_cmap("tab10")(index)
        ax.plot(x, mean, lw=2.2, color=color, label=str(setting))
        if len(curves) > 1:
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.14)
        plotted += 1
    ax.set_xlabel("Training tokens (millions)")
    ax.set_ylabel("Validation NLL (nats/token)")
    ax.set_title(f"PE damping stage {args.stage}: token-matched learning curves")
    ax.grid(alpha=0.25)
    ax.legend(title="Setting")
    fig.tight_layout()
    if plotted:
        save(fig, Path(f"{args.out_prefix}_curves"))
    else:
        plt.close(fig)

    available = [row for row in aggregate if row["n_valid"]]
    if available:
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        y = np.arange(len(available))
        means = [float(row["mean_delta_final_val"]) for row in available]
        colors = ["#238b45" if row["selected"] else "#3182bd" for row in available]
        ax.barh(y, means, color=colors, alpha=0.82)
        by_setting = defaultdict(list)
        for row in pairs:
            if row["valid_pair"]:
                by_setting[key(row["setting"])].append(float(row["delta_final_val"]))
        for yi, row in zip(y, available):
            points = by_setting[key(row["setting"])]
            ax.scatter(points, np.full(len(points), yi), color="black", s=28, zorder=3)
        ax.axvline(0, color="black", lw=1)
        ax.axvline(
            REPLACE_THRESHOLD,
            color="#cb181d",
            ls="--",
            lw=1.2,
            label="replacement threshold",
        )
        ax.set_yticks(y, [str(row["setting"]) for row in available])
        ax.invert_yaxis()
        ax.set_xlabel("Paired final-NLL difference vs incumbent")
        ax.set_title(f"PE damping stage {args.stage}: replacement decision")
        ax.grid(axis="x", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        save(fig, Path(f"{args.out_prefix}_paired"))


if __name__ == "__main__":
    main()
