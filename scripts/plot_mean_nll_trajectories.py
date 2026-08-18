#!/usr/bin/env python3
"""Plot seed-mean validation NLL versus optimizer iteration for architecture screens.

Only completed runs are included by default. Curves use the exact intersection
of validation steps across seeds; no interpolation or smoothing is applied.
The shaded region is the seed range, which is transparent and meaningful even
for the usual two-seed screens.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import shlex
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

LABELS = {
    "baseline": "Baseline",
    "baseline_capacity": "Baseline capacity",
    "yurii_lt": "YuriiFormer",
    "causal_symp_fe": "Causal FE",
    "causal_symp_pe": "Causal PE",
    "causal_symp_exp_pe": "Causal ExpPE",
    "causal_symp_halfdamp_pe": "Causal half-damp PE",
    "causal_symp_ab2": "Causal AB2",
    "causal_riem_nag_noconn": "Riem NAG, no connection",
    "causal_riem_nag": "Riem NAG + connection",
    "lin_baseline": "Linear baseline",
    "lin_yurii": "Linear YuriiFormer",
    "lin_euler": "Linear Euler",
    "lin_presymp": "Linear kick--damp--drift",
    "lin_exp_euler": "Linear ExpEuler",
    "lin_ab2": "Linear AB2",
    "lin_etd_ab2": "Linear ETD--AB2",
    "lin_reduced_exp_mid": "Prefix-reduced ExpMid",
    "lin_reduced_ab2": "Prefix-reduced AB2",
    "arch100_mlp_attnvel": "Attention velocity (incumbent)",
    "arch100_mlp_stdmlp": "Separate MLP velocity",
    "arch100_mlp_pvel": "P momentum",
    "arch100_lnp_lnpend": "Momentum LN (incumbent)",
    "arch100_lnp_lnpnone": "No momentum LN",
    "arch100_disc_pe": "Incumbent PE",
    "arch100_disc_exppe": "Integrating-factor PE",
    "arch100_disc_halfdamp": "Half-damping PE",
    "arch100_lookahead_none": "No lookahead",
    "arch100_lookahead_mu01": "Lookahead $\\mu=0.1$",
    "arch100_lookahead_mu05": "Lookahead $\\mu=0.5$",
    "arch100_lookahead_mu09": "Lookahead $\\mu=0.9$",
    "arch100_ab2_pe": "Causal PE",
    "arch100_ab2_ab2": "Causal AB2",
}


def read_spec(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        dataset, arch, config, seed, run_name, flags = line.split("\t")
        candidate = re.sub(r"_s[0-9]+$", "", run_name)
        rows.append({"arch": arch, "candidate": candidate, "seed": int(seed), "run_name": run_name, "flags": shlex.split(flags)})
    return rows


def validation_rows(path: Path) -> dict[int, tuple[int, float]]:
    values = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if not row.get("val_loss"):
                continue
            step = int(row["step"])
            value = float(row["val_loss"])
            if math.isfinite(value):
                values[step] = (int(row["tokens_cum"]), value)
    return values


def aggregate(runs_dir: Path, specs: list[dict], allow_incomplete: bool, group_by: str = "arch") -> list[dict]:
    by_arch: dict[str, list[dict[int, tuple[int, float]]]] = defaultdict(list)
    for spec in specs:
        run_dir = runs_dir / f"{spec['arch']}_{spec['run_name']}"
        if not allow_incomplete and not (run_dir / "summary.json").is_file():
            continue
        metrics = run_dir / "metrics.csv"
        if metrics.is_file():
            rows = validation_rows(metrics)
            if rows:
                by_arch[spec[group_by]].append(rows)

    expected_seed_counts = {
        group: len({spec["seed"] for spec in specs if spec[group_by] == group})
        for group in dict.fromkeys(spec[group_by] for spec in specs)
    }
    output = []
    for arch in dict.fromkeys(spec[group_by] for spec in specs):
        seeds = by_arch.get(arch, [])
        if not seeds or len(seeds) != expected_seed_counts[arch]:
            continue
        common_steps = set(seeds[0])
        for seed in seeds[1:]:
            common_steps &= set(seed)
        for step in sorted(common_steps):
            vals = np.asarray([seed[step][1] for seed in seeds], dtype=float)
            tokens = [seed[step][0] for seed in seeds]
            if len(set(tokens)) != 1:
                raise AssertionError(f"{arch} step {step}: token counts differ across seeds")
            output.append({
                "arch": arch, "step": step, "tokens": tokens[0], "n": len(vals),
                "mean_nll": float(vals.mean()), "min_nll": float(vals.min()),
                "max_nll": float(vals.max()), "std_nll": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            })
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["arch", "step", "tokens", "n", "mean_nll", "min_nll", "max_nll", "std_nll"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs_dir", type=Path)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--allow_incomplete", action="store_true")
    parser.add_argument("--group_by", choices=("arch", "candidate"), default="arch")
    parser.add_argument("--yscale", choices=("linear", "log"), default="linear")
    args = parser.parse_args()

    specs = read_spec(args.spec)
    rows = aggregate(args.runs_dir, specs, args.allow_incomplete, args.group_by)
    if not rows:
        raise SystemExit("no validation trajectories found")
    write_csv(args.out.with_suffix(".csv"), rows)

    methods = list(dict.fromkeys(row["arch"] for row in rows))
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    for index, method in enumerate(methods):
        selected = [row for row in rows if row["arch"] == method]
        steps = np.asarray([row["step"] for row in selected])
        mean = np.asarray([row["mean_nll"] for row in selected])
        low = np.asarray([row["min_nll"] for row in selected])
        high = np.asarray([row["max_nll"] for row in selected])
        n = max(row["n"] for row in selected)
        color = cmap(index % 20)
        label = f"{LABELS.get(method, method)} (n={n})"
        ax.plot(steps, mean, linewidth=2.0, color=color, label=label)
        if n > 1:
            ax.fill_between(steps, low, high, color=color, alpha=0.12, linewidth=0)
    ax.set_xlabel("Optimizer iteration")
    ax.set_ylabel("Mean validation NLL" + (" (log scale)" if args.yscale == "log" else ""))
    ax.set_yscale(args.yscale)
    if args.yscale == "log":
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(ScalarFormatter())
    ax.set_title(args.title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8.5, ncol=2)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {args.out.with_suffix('.pdf')}, {args.out.with_suffix('.png')}, and {args.out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
