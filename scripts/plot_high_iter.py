#!/usr/bin/env python3
"""Plot high-iteration endpoint and trajectory paired differences."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"missing CSV: {path}")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def finite_float(value: str) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def save(fig, prefix: Path, suffix: str) -> None:
    png = prefix.with_name(prefix.name + suffix + ".png")
    pdf = prefix.with_name(prefix.name + suffix + ".pdf")
    fig.tight_layout()
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    print(f"wrote {png} and {pdf}")


def endpoint_plot(rows: list[dict], prefix: Path) -> None:
    valid = [row for row in rows if row.get("valid_pair") == "1"]
    comps = sorted({row["comparison"] for row in valid})
    fig, ax = plt.subplots(figsize=(7, 4))
    positions = []
    labels = []
    values_by_pos = []
    pos = 0
    for comp in comps:
        vals = [finite_float(row["delta_final_val"]) for row in valid if row["comparison"] == comp]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        positions.append(pos)
        labels.append(comp.replace("pe_minus_", "PE - "))
        values_by_pos.append(vals)
        ax.scatter([pos] * len(vals), vals, alpha=0.8)
        mean = sum(vals) / len(vals)
        ax.plot([pos - 0.25, pos + 0.25], [mean, mean], color="black", lw=2)
        pos += 1
    ax.axhline(0, color="gray", lw=1, ls="--")
    ax.set_xticks(positions, labels, rotation=15, ha="right")
    ax.set_ylabel("Final validation NLL difference (negative favors PE)")
    ax.set_title("High-iteration paired endpoint differences")
    save(fig, prefix, "_endpoint")
    plt.close(fig)


def trajectory_plot(rows: list[dict], prefix: Path) -> None:
    valid = []
    for row in rows:
        delta = finite_float(row.get("delta_val", ""))
        if delta is not None:
            valid.append((row["comparison"], int(row["target_tokens"]), delta))
    by_comp_target: dict[tuple[str, int], list[float]] = defaultdict(list)
    for comp, target, delta in valid:
        by_comp_target[(comp, target)].append(delta)
    comps = sorted({comp for comp, _ in by_comp_target})
    fig, ax = plt.subplots(figsize=(7, 4))
    for comp in comps:
        xs = []
        ys = []
        for (_, target), vals in sorted(by_comp_target.items()):
            if _ != comp:
                continue
            xs.append(target / 1e6)
            ys.append(sum(vals) / len(vals))
        if xs:
            ax.plot(xs, ys, marker="o", label=comp.replace("pe_minus_", "PE - "))
    ax.axhline(0, color="gray", lw=1, ls="--")
    ax.set_xlabel("Training tokens (millions)")
    ax.set_ylabel("Mean validation NLL difference (negative favors PE)")
    ax.set_title("High-iteration trajectory/crossover diagnostic")
    ax.legend()
    save(fig, prefix, "_trajectory")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis_dir", type=Path)
    parser.add_argument("--out_prefix", type=Path)
    args = parser.parse_args()
    prefix = args.out_prefix or args.analysis_dir / "high_iter"
    endpoint_plot(read_csv(args.analysis_dir / "endpoint_pairs.csv"), prefix)
    trajectory_plot(read_csv(args.analysis_dir / "trajectory_pairs.csv"), prefix)


if __name__ == "__main__":
    main()
