#!/usr/bin/env python3
"""Strict paired analysis for the fresh-seed 200M PE confirmation."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from analyze_pe_tuning import (
    RUN_FIELDS,
    canonical_name,
    finite,
    load_run,
    missing_run,
    parse_spec,
    write_csv,
)
from make_pe_200m_specs import confirmation_rows


CONFIRM_THRESHOLD = -0.010


def candidate_family(candidate):
    if candidate == "pe200_anchor":
        return "anchor"
    if candidate.startswith("pe200_adam"):
        return "adam_group_lr"
    if candidate.startswith("pe200_muon"):
        return "muon_lr"
    return "unexpected"


def expected_runs():
    expected = []
    for raw in confirmation_rows():
        item = parse_spec(raw)
        item["family"] = candidate_family(item["candidate"])
        expected.append(item)
    return expected


def discover_runs(root):
    expected = expected_runs()
    manifests = {}
    for path in sorted(root.rglob("run_manifest.json")):
        try:
            manifest = json.loads(path.read_text())
            name = manifest["args"]["run_name"]
        except (KeyError, json.JSONDecodeError):
            continue
        if name in manifests:
            raise SystemExit(f"Duplicate run_name below {root}: {name}")
        manifests[name] = path

    runs = []
    for item in expected:
        path = manifests.pop(item["run_name"], None)
        runs.append(load_run(path, item) if path else missing_run(item))

    for name, path in sorted(manifests.items()):
        manifest = json.loads(path.read_text())
        args = manifest["args"]
        item = {
            "run_name": name,
            "candidate": canonical_name(name),
            "family": "unexpected",
            "arch": args["arch"],
            "seed": int(args["seed"]),
            "dataset": args.get("dataset"),
            "config": args.get("config"),
            "expected_tokens": int(manifest.get("actual_max_tokens", 0)),
            "expected_args": {},
        }
        row = load_run(path, item)
        row["status"] = "unexpected"
        row["reasons"] = "not_in_predeclared_specs"
        runs.append(row)
    return runs


def provenance_match(left, right):
    return bool(
        left
        and right
        and all(
            left[field] == right[field]
            for field in ("source_revision", "train_sha256", "val_sha256", "gpu")
        )
    )


def paired_results(runs):
    anchors = {
        row["seed"]: row
        for row in runs
        if row["candidate"] == "pe200_anchor"
    }
    pairs = []
    for row in runs:
        if not row["candidate"].startswith("pe200_") or row["candidate"] == "pe200_anchor":
            continue
        anchor = anchors.get(row["seed"])
        same_provenance = provenance_match(row, anchor)
        valid = bool(
            anchor
            and anchor["status"] == "valid"
            and row["status"] == "valid"
            and same_provenance
        )
        pairs.append(
            {
                "candidate": row["candidate"],
                "family": row["family"],
                "seed": row["seed"],
                "candidate_status": row["status"],
                "anchor_status": anchor["status"] if anchor else "missing",
                "provenance_match": int(same_provenance),
                "candidate_final_val": row["final_val"],
                "anchor_final_val": anchor["final_val"] if anchor else "",
                "delta_final_val": row["final_val"] - anchor["final_val"] if valid else "",
                "delta_last3_val": row["last3_val"] - anchor["last3_val"] if valid else "",
                "valid_pair": int(valid),
            }
        )
    return sorted(pairs, key=lambda row: (row["candidate"], row["seed"]))


def aggregate_results(pairs):
    groups = defaultdict(list)
    for row in pairs:
        groups[row["candidate"]].append(row)
    aggregate = []
    for candidate, rows in groups.items():
        valid = [row for row in rows if row["valid_pair"]]
        deltas = [float(row["delta_final_val"]) for row in valid]
        last3 = [float(row["delta_last3_val"]) for row in valid]
        mean_delta = statistics.fmean(deltas) if deltas else math.nan
        wins = sum(delta < 0 for delta in deltas)
        confirms = (
            len(rows) == 2
            and len(valid) == 2
            and wins == 2
            and mean_delta <= CONFIRM_THRESHOLD
        )
        aggregate.append(
            {
                "candidate": candidate,
                "family": rows[0]["family"],
                "n_expected": len(rows),
                "n_valid": len(valid),
                "wins_vs_anchor": wins,
                "mean_delta_final_val": mean_delta,
                "std_delta_final_val": (
                    statistics.stdev(deltas) if len(deltas) > 1 else math.nan
                ),
                "mean_delta_last3_val": (
                    statistics.fmean(last3) if last3 else math.nan
                ),
                "confirms_at_200m": int(confirms),
            }
        )
    return sorted(
        aggregate,
        key=lambda row: (
            -row["confirms_at_200m"],
            row["mean_delta_final_val"] if finite(row["mean_delta_final_val"]) else math.inf,
            row["candidate"],
        ),
    )


def analyze(root):
    runs = discover_runs(root)
    pairs = paired_results(runs)
    aggregate = aggregate_results(pairs)
    return runs, pairs, aggregate


def print_compact(runs, aggregate):
    expected = runs[: len(expected_runs())]
    valid = sum(row["status"] == "valid" for row in expected)
    missing = sum(row["status"] == "missing" for row in expected)
    invalid = sum(row["status"] == "invalid" for row in expected)
    print(f"expected=8 valid={valid} missing={missing} invalid={invalid}")
    print("\n200M fresh-seed confirmation (negative is better than the paired PE anchor):")
    print(f"{'CANDIDATE':22s} {'N':>3s} {'WINS':>4s} {'MEAN_DNLL':>11s} {'CONFIRM':>8s}")
    for row in aggregate:
        delta = row["mean_delta_final_val"]
        rendered = f"{delta:+.6f}" if finite(delta) else "NA"
        print(
            f"{row['candidate']:22s} {row['n_valid']:3d} "
            f"{row['wins_vs_anchor']:4d} {rendered:>11s} "
            f"{'yes' if row['confirms_at_200m'] else 'no':>8s}"
        )
    print("\nConfirmation requires 2/2 valid paired wins and mean delta <= -0.010.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--out_dir", type=Path)
    args = parser.parse_args()
    out_dir = args.out_dir or args.root / "analysis"
    runs, pairs, aggregate = analyze(args.root)
    write_csv(out_dir / "runs.csv", runs, RUN_FIELDS)
    write_csv(out_dir / "paired_vs_anchor.csv", pairs)
    write_csv(out_dir / "aggregate.csv", aggregate)
    print_compact(runs, aggregate)
    print(f"\nwrote analysis tables to {out_dir}")
    if any(row["status"] != "valid" for row in runs[: len(expected_runs())]):
        print("WARNING: missing/invalid runs are retained and cannot confirm.")


if __name__ == "__main__":
    main()
