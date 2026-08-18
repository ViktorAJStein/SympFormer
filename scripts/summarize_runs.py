#!/usr/bin/env python3
"""Aggregate completed training directories without silently dropping failures."""

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
       8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201, 13: 2.179,
       14: 2.160, 15: 2.145, 16: 2.131, 17: 2.120, 18: 2.110, 19: 2.101,
       20: 2.093, 21: 2.086, 22: 2.080, 23: 2.074, 24: 2.069, 25: 2.064,
       26: 2.060, 27: 2.056, 28: 2.052, 29: 2.048, 30: 2.045}


def t_critical(n):
    if n < 2:
        return float("nan")
    return T95.get(n, 1.96)


def mean_std_ci(values):
    a = np.asarray(values, dtype=float)
    n = len(a)
    mean = float(a.mean())
    std = float(a.std(ddof=1)) if n > 1 else 0.0
    ci = t_critical(n) * std / math.sqrt(n) if n > 1 else float("nan")
    return mean, std, ci


def read_val_rows(path):
    rows = []
    if not path.exists():
        return rows
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            value = row.get("val_loss", "")
            if value == "":
                continue
            rows.append((int(row["step"]), int(row["tokens_cum"]), float(row["wall_cum_s"]), float(value)))
    rows.sort(key=lambda x: (x[1], x[0]))
    dedup = {}
    for row in rows:
        dedup[row[1]] = row
    return [dedup[k] for k in sorted(dedup)]


def normalized_auc(val_rows):
    if len(val_rows) < 2:
        return val_rows[-1][3] if val_rows else float("nan")
    x = np.asarray([r[1] for r in val_rows], dtype=float)
    y = np.asarray([r[3] for r in val_rows], dtype=float)
    width = x[-1] - x[0]
    return float(np.trapezoid(y, x) / width) if width > 0 else float(y[-1])


def run_tag(name):
    return re.sub(r"_s\d+$", "", name or "default")


def load_run(manifest_path):
    run_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    args = manifest["args"]
    cfg = manifest["model_config"]
    summary_path = run_dir / "summary.json"
    vals = read_val_rows(run_dir / "metrics.csv")
    summary_present = summary_path.exists()
    summary = json.loads(summary_path.read_text()) if summary_present else {}
    final_val = summary.get("final_val", vals[-1][3] if vals else float("nan"))
    wall = summary.get("wall_cum_s", vals[-1][2] if vals else float("nan"))
    tokens = summary.get("tokens", vals[-1][1] if vals else 0)
    completed = bool(
        summary_present and vals and int(tokens) > 0
        and math.isfinite(float(final_val)) and math.isfinite(float(wall))
    )
    return {
        "run_dir": str(run_dir),
        "status": "complete" if completed else "incomplete_or_invalid",
        "dataset": args["dataset"],
        "arch": args["arch"],
        "seed": int(args["seed"]),
        "run_name": args.get("run_name", ""),
        "run_tag": run_tag(args.get("run_name", "")),
        "n_layer": int(cfg["n_layer"]),
        "n_head": int(cfg["n_head"]),
        "n_embd": int(cfg["n_embd"]),
        "block_size": int(cfg["block_size"]),
        "target_max_tokens": manifest.get("target_max_tokens", args.get("max_tokens", 0)),
        "tokens": int(tokens),
        "final_val": float(final_val),
        "best_val": float(summary.get("best_val", min((r[3] for r in vals), default=float("nan")))),
        "last3_val": float(np.mean([r[3] for r in vals[-3:]])) if vals else float("nan"),
        "val_auc": normalized_auc(vals),
        "wall_cum_s": float(wall),
        "gpu_hours": float(wall) / 3600.0,
        "tokens_per_second": float(tokens) / float(wall) if wall and math.isfinite(float(wall)) else float("nan"),
        "trainable_parameters": int(summary.get("trainable_parameters", manifest["trainable_parameters"])),
        "peak_memory_mb": float(summary.get("peak_memory_mb", float("nan"))),
        "gpu": manifest.get("gpu") or "CPU",
        "source_revision": manifest.get("source_revision") or "unversioned",
    }


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(completed):
    key_fields = ["dataset", "n_layer", "n_head", "n_embd", "block_size", "target_max_tokens", "run_tag", "arch"]
    groups = defaultdict(list)
    for row in completed:
        groups[tuple(row[k] for k in key_fields)].append(row)
    out = []
    metrics = ["final_val", "last3_val", "val_auc", "gpu_hours", "tokens_per_second", "peak_memory_mb"]
    for key, rows in sorted(groups.items(), key=str):
        item = dict(zip(key_fields, key))
        item["n"] = len(rows)
        item["seeds"] = ";".join(str(r["seed"]) for r in sorted(rows, key=lambda r: r["seed"]))
        item["tokens"] = rows[0]["tokens"] if len({r["tokens"] for r in rows}) == 1 else "mixed"
        item["trainable_parameters"] = rows[0]["trainable_parameters"]
        for metric in metrics:
            mean, std, ci = mean_std_ci([r[metric] for r in rows])
            item[f"{metric}_mean"] = mean
            item[f"{metric}_std"] = std
            item[f"{metric}_ci95"] = ci
        out.append(item)
    return out


def paired_differences(completed):
    pair_fields = ["dataset", "n_layer", "n_head", "n_embd", "block_size", "tokens", "run_name", "seed"]
    by_pair = defaultdict(dict)
    for row in completed:
        by_pair[tuple(row[k] for k in pair_fields)][row["arch"]] = row
    deltas = defaultdict(list)
    for key, methods in by_pair.items():
        if "baseline" not in methods:
            continue
        base = methods["baseline"]
        for arch, row in methods.items():
            if arch == "baseline":
                continue
            group = key[:6] + (run_tag(key[6]), arch)
            deltas[group].append((row["seed"], row["final_val"] - base["final_val"]))
    fields = ["dataset", "n_layer", "n_head", "n_embd", "block_size", "tokens", "run_tag", "arch"]
    out = []
    for key, values in sorted(deltas.items(), key=str):
        ds = [v for _, v in values]
        mean, std, ci = mean_std_ci(ds)
        item = dict(zip(fields, key))
        item.update(n=len(ds), seeds=";".join(str(s) for s, _ in sorted(values)), delta_final_val_mean=mean,
                    delta_final_val_std=std, delta_final_val_ci95=ci)
        out.append(item)
    return out


def paired_ablations(completed):
    """Pair focused variants against the correct same-seed reference."""
    scenario_fields = ["dataset", "n_layer", "n_head", "n_embd", "block_size", "tokens", "seed"]
    by_scenario = defaultdict(dict)
    for row in completed:
        if not row["run_tag"].startswith("abl_"):
            continue
        by_scenario[tuple(row[k] for k in scenario_fields)][(row["arch"], row["run_tag"])] = row
    deltas = defaultdict(list)
    for key, rows in by_scenario.items():
        seed = key[-1]
        causal_ref = rows.get(("causal_symp_pe", "abl_reference"))
        baseline_ref = rows.get(("baseline", "abl_reference"))
        for (arch, tag), row in rows.items():
            reference = None
            reference_name = ""
            if arch == "causal_symp_pe" and tag != "abl_reference" and causal_ref is not None:
                reference, reference_name = causal_ref, "causal_symp_pe:abl_reference"
            elif arch == "baseline_capacity" and baseline_ref is not None:
                reference, reference_name = baseline_ref, "baseline:abl_reference"
            if reference is None:
                continue
            group = key[:-1] + (f"{arch}:{tag}", reference_name)
            deltas[group].append((seed, row["final_val"] - reference["final_val"]))
    fields = ["dataset", "n_layer", "n_head", "n_embd", "block_size", "tokens", "variant", "reference"]
    out = []
    for key, values in sorted(deltas.items(), key=str):
        ds = [v for _, v in values]
        mean, std, ci = mean_std_ci(ds)
        item = dict(zip(fields, key))
        item.update(n=len(ds), seeds=";".join(str(seed) for seed, _ in sorted(values)),
                    delta_final_val_mean=mean, delta_final_val_std=std, delta_final_val_ci95=ci)
        out.append(item)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="campaign output directory")
    ap.add_argument("--out_dir", type=Path, default=Path("artifacts/campaign_summary"))
    args = ap.parse_args()
    manifests = sorted(args.root.rglob("run_manifest.json"))
    if not manifests:
        raise SystemExit(f"No run_manifest.json files found below {args.root}")
    runs = [load_run(path) for path in manifests]
    completed = [r for r in runs if r["status"] == "complete"]
    fields = list(runs[0])
    write_csv(args.out_dir / "runs.csv", runs, fields)
    grouped = aggregate(completed)
    if grouped:
        write_csv(args.out_dir / "grouped.csv", grouped, list(grouped[0]))
    paired = paired_differences(completed)
    if paired:
        write_csv(args.out_dir / "paired_vs_baseline.csv", paired, list(paired[0]))
    ablations = paired_ablations(completed)
    if ablations:
        write_csv(args.out_dir / "paired_ablations.csv", ablations, list(ablations[0]))
    print(f"found={len(runs)} complete={len(completed)} incomplete={len(runs)-len(completed)}")
    print(f"wrote {args.out_dir / 'runs.csv'}")
    if len(completed) != len(runs):
        print("WARNING: incomplete runs are retained in runs.csv and excluded from aggregate claims")


if __name__ == "__main__":
    main()
