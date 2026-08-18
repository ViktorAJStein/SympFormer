#!/usr/bin/env python3
"""Strict endpoint and trajectory audit for high-iteration headline runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from analyze_pe_tuning import RUN_FIELDS, load_run, missing_run, parse_spec, read_metrics, write_csv

METHODS = ("baseline", "yurii_lt", "causal_symp_pe")


def expected_runs(spec: Path) -> list[dict]:
    return [parse_spec(line) for line in spec.read_text().splitlines() if line.strip()]


def discover_runs(root: Path, expected: list[dict]) -> list[dict]:
    manifests: dict[str, Path] = {}
    for path in sorted(root.rglob("run_manifest.json")):
        try:
            name = json.loads(path.read_text())["args"]["run_name"]
        except (KeyError, json.JSONDecodeError):
            continue
        if name in manifests:
            raise SystemExit(f"duplicate run_name below {root}: {name}")
        manifests[name] = path
    rows = []
    for item in expected:
        path = manifests.pop(item["run_name"], None)
        rows.append(load_run(path, item) if path else missing_run(item))
    for name in sorted(manifests):
        rows.append({**{field: "" for field in RUN_FIELDS}, "run_name": name, "status": "unexpected", "reasons": "not_in_predeclared_specs"})
    return rows


def provenance_key(row: dict) -> tuple:
    return tuple(row.get(field, "") for field in ("source_revision", "train_sha256", "val_sha256", "gpu"))


def endpoint_pairs(runs: list[dict], expected_count: int) -> list[dict]:
    expected = runs[:expected_count]
    by_seed_arch = {(int(row["seed"]), row["arch"]): row for row in expected}
    seeds = sorted({int(row["seed"]) for row in expected})
    pairs = []
    for seed in seeds:
        pe = by_seed_arch.get((seed, "causal_symp_pe"))
        for ref in ("baseline", "yurii_lt"):
            base = by_seed_arch.get((seed, ref))
            valid_pair = bool(pe and base and pe["status"] == "valid" and base["status"] == "valid" and provenance_key(pe) == provenance_key(base))
            pairs.append(
                {
                    "seed": seed,
                    "comparison": f"pe_minus_{ref}",
                    "pe_status": pe["status"] if pe else "missing",
                    "ref_status": base["status"] if base else "missing",
                    "provenance_match": int(valid_pair),
                    "pe_final_val": pe.get("final_val", math.nan) if pe else math.nan,
                    "ref_final_val": base.get("final_val", math.nan) if base else math.nan,
                    "delta_final_val": (float(pe["final_val"]) - float(base["final_val"])) if valid_pair else math.nan,
                    "delta_last3_val": (float(pe["last3_val"]) - float(base["last3_val"])) if valid_pair else math.nan,
                    "delta_val_auc": (float(pe["val_auc"]) - float(base["val_auc"])) if valid_pair else math.nan,
                    "valid_pair": int(valid_pair),
                }
            )
    return pairs


def aggregate_pairs(pairs: list[dict]) -> list[dict]:
    out = []
    by_comp: dict[str, list[dict]] = defaultdict(list)
    for row in pairs:
        by_comp[row["comparison"]].append(row)
    for comp, rows in sorted(by_comp.items()):
        valid = [row for row in rows if row["valid_pair"]]
        vals = [float(row["delta_final_val"]) for row in valid]
        out.append(
            {
                "comparison": comp,
                "n_expected": len(rows),
                "n_valid": len(valid),
                "wins": sum(v < 0 for v in vals),
                "mean_delta_final_val": statistics.fmean(vals) if vals else math.nan,
                "std_delta_final_val": statistics.stdev(vals) if len(vals) > 1 else math.nan,
                "mean_delta_last3_val": statistics.fmean(float(row["delta_last3_val"]) for row in valid) if valid else math.nan,
                "mean_delta_val_auc": statistics.fmean(float(row["delta_val_auc"]) for row in valid) if valid else math.nan,
            }
        )
    return out


def nearest_metric(run_dir: Path, targets: list[int]) -> list[dict]:
    metrics = read_metrics(run_dir / "metrics.csv")
    if not metrics:
        return []
    out = []
    for target in targets:
        row = min(metrics, key=lambda item: abs(int(item["tokens"]) - target))
        out.append({"target_tokens": target, "tokens": row["tokens"], "val": row["val"]})
    return out


def trajectory_rows(runs: list[dict], expected_count: int, targets: list[int]) -> list[dict]:
    expected = runs[:expected_count]
    valid = [row for row in expected if row["status"] == "valid"]
    by_seed_arch = {(int(row["seed"]), row["arch"]): row for row in valid}
    rows = []
    for seed in sorted({int(row["seed"]) for row in expected}):
        pe = by_seed_arch.get((seed, "causal_symp_pe"))
        if not pe:
            continue
        pe_metrics = {row["target_tokens"]: row for row in nearest_metric(Path(pe["run_dir"]), targets)}
        for ref in ("baseline", "yurii_lt"):
            base = by_seed_arch.get((seed, ref))
            if not base or provenance_key(pe) != provenance_key(base):
                continue
            base_metrics = {row["target_tokens"]: row for row in nearest_metric(Path(base["run_dir"]), targets)}
            for target in targets:
                if target not in pe_metrics or target not in base_metrics:
                    continue
                rows.append(
                    {
                        "seed": seed,
                        "comparison": f"pe_minus_{ref}",
                        "target_tokens": target,
                        "pe_tokens": pe_metrics[target]["tokens"],
                        "ref_tokens": base_metrics[target]["tokens"],
                        "pe_val": pe_metrics[target]["val"],
                        "ref_val": base_metrics[target]["val"],
                        "delta_val": pe_metrics[target]["val"] - base_metrics[target]["val"],
                    }
                )
    return rows


def print_summary(runs: list[dict], expected_count: int, aggregate: list[dict]) -> None:
    expected = runs[:expected_count]
    valid = sum(row["status"] == "valid" for row in expected)
    missing = sum(row["status"] == "missing" for row in expected)
    invalid = sum(row["status"] == "invalid" for row in expected)
    print(f"expected={expected_count} valid={valid} missing={missing} invalid={invalid}")
    print("\nHigh-iteration endpoint pairs (negative favors PE):")
    print(f"{'COMPARISON':24s} {'N':>3s} {'WINS':>5s} {'MEAN_DNLL':>11s} {'STD':>11s}")
    for row in aggregate:
        mean = row["mean_delta_final_val"]
        std = row["std_delta_final_val"]
        print(
            f"{row['comparison']:24s} {row['n_valid']:3d} {row['wins']:5d} "
            f"{mean:11.6f} {std:11.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path)
    parser.add_argument("--trajectory_tokens", default="200000000,500000000,750000000,1000000000")
    args = parser.parse_args()
    expected = expected_runs(args.spec)
    runs = discover_runs(args.root, expected)
    pairs = endpoint_pairs(runs, len(expected))
    aggregate = aggregate_pairs(pairs)
    targets = [int(x) for x in args.trajectory_tokens.split(",") if x]
    traj = trajectory_rows(runs, len(expected), targets)
    out_dir = args.out_dir or args.root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "runs.csv", runs, RUN_FIELDS)
    write_csv(out_dir / "endpoint_pairs.csv", pairs)
    write_csv(out_dir / "endpoint_aggregate.csv", aggregate)
    write_csv(out_dir / "trajectory_pairs.csv", traj)
    print_summary(runs, len(expected), aggregate)
    print(f"\nwrote analysis tables to {out_dir}")


if __name__ == "__main__":
    main()
