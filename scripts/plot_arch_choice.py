#!/usr/bin/env python3
"""Plot strict architecture-screen endpoint results from analyzer CSV output."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs_csv", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="Output stem; writes PDF and PNG")
    parser.add_argument("--title", default="Architecture screen at 100M tokens")
    args = parser.parse_args()

    values: dict[str, dict[int, float]] = defaultdict(dict)
    with args.runs_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["status"] == "valid":
                values[row["candidate"]][int(row["seed"])] = float(row["final_val"])
    if not values:
        raise SystemExit("no valid runs")

    preferred = ["arch100_mlp_attnvel", "arch100_mlp_stdmlp", "arch100_mlp_pvel",
                 "arch100_lnp_lnpend", "arch100_lnp_lnpnone",
                 "arch100_disc_pe", "arch100_disc_exppe", "arch100_disc_halfdamp",
                 "arch100_lookahead_none", "arch100_lookahead_mu01",
                 "arch100_lookahead_mu05", "arch100_lookahead_mu09",
                 "arch100_ab2_pe", "arch100_ab2_ab2"]
    candidates = [name for name in preferred if name in values]
    candidates += sorted(set(values) - set(candidates))
    seeds = sorted(set.intersection(*(set(values[name]) for name in candidates)))
    if not seeds:
        raise SystemExit("no common paired seeds")

    labels = {
        "arch100_mlp_attnvel": "Attention velocity\n(incumbent)",
        "arch100_mlp_stdmlp": "Separate MLP\nvelocity",
        "arch100_mlp_pvel": "P momentum",
        "arch100_lnp_lnpend": "Momentum LN\n(incumbent)",
        "arch100_lnp_lnpnone": "No momentum LN",
        "arch100_disc_pe": "Incumbent PE",
        "arch100_disc_exppe": "Integrating-factor PE",
        "arch100_disc_halfdamp": "Half-damping PE",
        "arch100_lookahead_none": "No lookahead",
        "arch100_lookahead_mu01": "Lookahead μ=0.1",
        "arch100_lookahead_mu05": "Lookahead μ=0.5",
        "arch100_lookahead_mu09": "Lookahead μ=0.9",
        "arch100_ab2_pe": "Causal PE",
        "arch100_ab2_ab2": "Causal AB2",
    }
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    xs = list(range(len(candidates)))
    for seed in seeds:
        ys = [values[name][seed] for name in candidates]
        ax.plot(xs, ys, marker="o", linewidth=1.4, alpha=0.8, label=f"seed {seed}")
    means = [sum(values[name][seed] for seed in seeds) / len(seeds) for name in candidates]
    ax.scatter(xs, means, marker="D", s=60, color="black", zorder=4, label="mean")
    ax.set_xticks(xs, [labels.get(name, name) for name in candidates])
    ax.set_ylabel("Final validation NLL (lower is better)")
    ax.set_title(args.title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(args.out.with_suffix(f".{suffix}"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
