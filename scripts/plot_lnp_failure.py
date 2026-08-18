#!/usr/bin/env python3
"""Plot completion/failure points for the momentum-LayerNorm screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

EXPECTED_TOKENS = 99_876_864


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("failure_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    failures = {}
    for path in sorted(args.failure_dir.glob("failure_s*.json")):
        record = json.loads(path.read_text())
        assert record["status"] == "failed_nonfinite"
        failures[int(record["seed"])] = record
    if len(failures) != 2:
        raise SystemExit(f"expected two failures, found {len(failures)}")

    seeds = sorted(failures)
    x = range(len(seeds))
    complete = [100.0] * len(seeds)
    failed = [100.0 * failures[seed]["tokens_completed"] / EXPECTED_TOKENS for seed in seeds]

    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    width = 0.34
    ax.bar([i - width / 2 for i in x], complete, width, label="Momentum LN at end", color="#4C78A8")
    ax.bar([i + width / 2 for i in x], failed, width, label="No momentum LN", color="#E45756")
    for i, value in enumerate(failed):
        ax.text(i + width / 2, value + 2, f"{value:.1f}%\nNaN", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(x), [f"seed {seed}" for seed in seeds])
    ax.set_ylim(0, 112)
    ax.set_ylabel("Fraction of 100M-token budget reached (%)")
    ax.set_title("Momentum normalization screen: training stability")
    ax.legend(frameon=False, loc="lower left")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(args.out.with_suffix(f".{suffix}"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
