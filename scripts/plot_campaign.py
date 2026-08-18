#!/usr/bin/env python3
"""Publication-ready seed-band and compute-quality plots for one campaign slice."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {
    "baseline": "Transformer",
    "baseline_capacity": "Transformer + capacity adapter",
    "yurii_lt": "YuriiFormer LT",
    "causal_symp_fe": "Causal SympFormer FE",
    "causal_symp_pe": "Causal SympFormer PE",
}
COLORS = {
    "baseline": "#2f4b7c",
    "baseline_capacity": "#7a5195",
    "yurii_lt": "#ef5675",
    "causal_symp_fe": "#2ca02c",
    "causal_symp_pe": "#ffa600",
}
T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
       8: 2.365, 9: 2.306, 10: 2.262}


def read_curve(path):
    vals = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("val_loss", ""):
                vals[int(row["tokens_cum"])] = (float(row["val_loss"]), float(row["wall_cum_s"]))
    return vals


def load_runs(root, dataset, n_layer, run_tag):
    runs = []
    for path in sorted(root.rglob("run_manifest.json")):
        manifest = json.loads(path.read_text())
        args = manifest["args"]
        cfg = manifest["model_config"]
        if dataset and args["dataset"] != dataset:
            continue
        if n_layer and int(cfg["n_layer"]) != n_layer:
            continue
        if run_tag and run_tag not in args.get("run_name", ""):
            continue
        summary_path = path.parent / "summary.json"
        metrics_path = path.parent / "metrics.csv"
        if not summary_path.exists() or not metrics_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        curve = read_curve(metrics_path)
        if not curve or not math.isfinite(float(summary.get("final_val", float("nan")))):
            continue
        runs.append((manifest, summary, curve))
    return runs


def ci_band(matrix):
    n = matrix.shape[0]
    mean = matrix.mean(axis=0)
    if n < 2:
        return mean, mean, mean
    sem = matrix.std(axis=0, ddof=1) / math.sqrt(n)
    half = T95.get(n, 1.96) * sem
    return mean, mean - half, mean + half


def save_both(fig, prefix):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(prefix) + ".pdf", bbox_inches="tight")
    fig.savefig(str(prefix) + ".png", dpi=240, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--dataset", default="")
    ap.add_argument("--n_layer", type=int, default=0)
    ap.add_argument("--run_tag", default="")
    ap.add_argument("--out_prefix", type=Path, default=Path("images/campaign"))
    args = ap.parse_args()
    runs = load_runs(args.root, args.dataset, args.n_layer, args.run_tag)
    if not runs:
        raise SystemExit("No completed runs match the requested slice")
    scenarios = {(m["args"]["dataset"], m["model_config"]["n_layer"], m["model_config"]["n_embd"], m["model_config"]["block_size"]) for m, _, _ in runs}
    if len(scenarios) != 1:
        raise SystemExit(f"Filters select multiple dataset/model scenarios: {sorted(scenarios)}")
    dataset, layers, width, block = next(iter(scenarios))
    groups = defaultdict(list)
    for manifest, summary, curve in runs:
        groups[manifest["args"]["arch"]].append((manifest, summary, curve))

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.22, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    for arch, items in sorted(groups.items()):
        common = set.intersection(*(set(curve) for _, _, curve in items))
        if not common:
            continue
        tokens = np.asarray(sorted(common), dtype=float)
        losses = np.asarray([[curve[int(t)][0] for t in tokens] for _, _, curve in items])
        mean, low, high = ci_band(losses)
        color = COLORS.get(arch)
        ax.plot(tokens / 1e9, mean, label=f"{LABELS.get(arch, arch)} (n={len(items)})", color=color, linewidth=2)
        ax.fill_between(tokens / 1e9, low, high, color=color, alpha=0.17, linewidth=0)
    ax.set_xlabel("Training tokens (billions)")
    ax.set_ylabel("Validation cross-entropy (nats/token)")
    ax.set_title(f"{dataset}: {layers} layers, width {width}, context {block}")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_both(fig, Path(str(args.out_prefix) + "_curves"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    for arch, items in sorted(groups.items()):
        x = np.asarray([summary["wall_cum_s"] / 3600.0 for _, summary, _ in items])
        y = np.asarray([summary["final_val"] for _, summary, _ in items])
        color = COLORS.get(arch)
        xerr = x.std(ddof=1) / math.sqrt(len(x)) * T95.get(len(x), 1.96) if len(x) > 1 else 0.0
        yerr = y.std(ddof=1) / math.sqrt(len(y)) * T95.get(len(y), 1.96) if len(y) > 1 else 0.0
        ax.errorbar(x.mean(), y.mean(), xerr=xerr, yerr=yerr, fmt="o", capsize=3, markersize=7,
                    color=color, label=f"{LABELS.get(arch, arch)} (n={len(items)})")
    ax.set_xlabel("Single-GPU wall time (hours)")
    ax.set_ylabel("Final validation cross-entropy (nats/token)")
    ax.set_title("Compute-quality comparison (mean and 95% t-CI)")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_both(fig, Path(str(args.out_prefix) + "_pareto"))
    plt.close(fig)
    print(f"saved {args.out_prefix}_curves.[pdf,png] and {args.out_prefix}_pareto.[pdf,png]")


if __name__ == "__main__":
    main()
