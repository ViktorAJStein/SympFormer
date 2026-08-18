#!/usr/bin/env python3
"""Strict audit and lock-file selection for the equal-budget 200M LR campaign."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from analyze_pe_tuning import RUN_FIELDS, load_run, missing_run, parse_spec, write_csv
from make_headline_lr_200m_specs import METHOD_LRS, SCALAR_PEAK_LR, TUNING_SEEDS, lr_rows


TIE_TOLERANCE = 0.005


def expected_runs() -> list[dict]:
    expected = []
    for raw in lr_rows():
        item = parse_spec(raw)
        item["method"] = item["arch"]
        item["peak_lr"] = float(item["expected_args"]["peak_lr"])
        item["family"] = "headline_lr"
        expected.append(item)
    return expected


def discover_runs(root: Path) -> list[dict]:
    manifests = {}
    for path in sorted(root.rglob("run_manifest.json")):
        try:
            name = json.loads(path.read_text())["args"]["run_name"]
        except (KeyError, json.JSONDecodeError):
            continue
        if name in manifests:
            raise SystemExit(f"duplicate run_name below {root}: {name}")
        manifests[name] = path
    rows = []
    for item in expected_runs():
        path = manifests.pop(item["run_name"], None)
        row = load_run(path, item) if path else missing_run(item)
        row["method"] = item["method"]
        row["peak_lr"] = item["peak_lr"]
        rows.append(row)
    for name in sorted(manifests):
        rows.append(
            {
                **{field: "" for field in RUN_FIELDS},
                "run_name": name,
                "status": "unexpected",
                "reasons": "not_in_predeclared_specs",
                "method": "unexpected",
                "peak_lr": "",
            }
        )
    return rows


def provenance_key(row: dict) -> tuple:
    return tuple(row[field] for field in ("source_revision", "train_sha256", "val_sha256", "gpu"))


def aggregate_results(runs: list[dict]) -> tuple[list[dict], bool]:
    expected = runs[: len(expected_runs())]
    valid_rows = [row for row in expected if row["status"] == "valid"]
    provenance_ok = len({provenance_key(row) for row in valid_rows}) == 1
    groups = defaultdict(list)
    for row in expected:
        groups[(row["method"], float(row["peak_lr"]))].append(row)
    aggregate = []
    for (method, peak_lr), rows in groups.items():
        valid = [row for row in rows if row["status"] == "valid"]
        values = [float(row["final_val"]) for row in valid]
        aggregate.append(
            {
                "method": method,
                "peak_lr": peak_lr,
                "n_expected": len(rows),
                "n_valid": len(valid),
                "mean_final_val": statistics.fmean(values) if values else math.nan,
                "std_final_val": statistics.stdev(values) if len(values) > 1 else math.nan,
                "mean_last3_val": statistics.fmean(float(row["last3_val"]) for row in valid) if valid else math.nan,
                "selection_eligible": int(len(valid) == len(TUNING_SEEDS) and provenance_ok),
                "selected": 0,
            }
        )
    complete = len(valid_rows) == len(expected) and provenance_ok
    if complete:
        for method in METHOD_LRS:
            choices = [row for row in aggregate if row["method"] == method and row["selection_eligible"]]
            best = min(float(row["mean_final_val"]) for row in choices)
            selected = min(
                (row for row in choices if float(row["mean_final_val"]) <= best + TIE_TOLERANCE),
                key=lambda row: float(row["peak_lr"]),
            )
            selected["selected"] = 1
    return sorted(aggregate, key=lambda row: (row["method"], float(row["peak_lr"]))), complete


def lock_payload(aggregate: list[dict], runs: list[dict], complete: bool) -> dict | None:
    if not complete:
        return None
    methods = {}
    for row in aggregate:
        if not row["selected"]:
            continue
        peak_lr = float(row["peak_lr"])
        methods[row["method"]] = {
            "peak_lr": peak_lr,
            "scalar_lr_mult": None if row["method"] == "baseline" else SCALAR_PEAK_LR / peak_lr,
            "mean_final_val": float(row["mean_final_val"]),
            "std_final_val": float(row["std_final_val"]),
        }
    first = runs[0]
    return {
        "format_version": 1,
        "campaign": "headline_lr_200m",
        "selection_complete": True,
        "selection_rule": "minimum mean final NLL; choose lower LR within 0.005",
        "seeds": list(TUNING_SEEDS),
        "actual_tokens_per_run": int(first["expected_tokens"]),
        "scalar_peak_lr": SCALAR_PEAK_LR,
        "source_revision": first["source_revision"],
        "train_sha256": first["train_sha256"],
        "val_sha256": first["val_sha256"],
        "gpu": first["gpu"],
        "methods": methods,
    }


def write_lock(path: Path, payload: dict | None) -> None:
    if payload is None:
        if path.exists():
            raise SystemExit(f"audit is incomplete but a stale lock exists: {path}")
        return
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != rendered:
        raise SystemExit(f"refusing to overwrite a different lock: {path}")
    path.write_text(rendered)


def analyze(root: Path) -> tuple[list[dict], list[dict], bool]:
    runs = discover_runs(root)
    aggregate, complete = aggregate_results(runs)
    return runs, aggregate, complete


def print_compact(runs: list[dict], aggregate: list[dict], complete: bool) -> None:
    expected = runs[: len(expected_runs())]
    valid = sum(row["status"] == "valid" for row in expected)
    missing = sum(row["status"] == "missing" for row in expected)
    invalid = sum(row["status"] == "invalid" for row in expected)
    print(f"expected=24 valid={valid} missing={missing} invalid={invalid}")
    print("\nEqual-budget 200M LR selection:")
    print(f"{'METHOD':18s} {'PEAK_LR':>9s} {'N':>3s} {'MEAN_NLL':>11s} {'LOCK':>6s}")
    for row in aggregate:
        mean = row["mean_final_val"]
        rendered = f"{mean:.6f}" if math.isfinite(mean) else "NA"
        print(
            f"{row['method']:18s} {float(row['peak_lr']):9.6f} {row['n_valid']:3d} "
            f"{rendered:>11s} {'yes' if row['selected'] else 'no':>6s}"
        )
    print(f"\nselection_complete={'yes' if complete else 'no'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--out_dir", type=Path)
    parser.add_argument("--lock_out", type=Path)
    args = parser.parse_args()
    out_dir = args.out_dir or args.root / "analysis"
    lock_out = args.lock_out or out_dir / "locked_lrs.json"
    runs, aggregate, complete = analyze(args.root)
    run_fields = RUN_FIELDS + ["method", "peak_lr"]
    write_csv(out_dir / "runs.csv", runs, run_fields)
    write_csv(out_dir / "aggregate.csv", aggregate)
    write_lock(lock_out, lock_payload(aggregate, runs, complete))
    print_compact(runs, aggregate, complete)
    print(f"\nwrote analysis tables to {out_dir}")
    if complete:
        print(f"wrote immutable LR lock to {lock_out}")
    else:
        print("WARNING: no lock written; all 24 runs and one provenance cohort are required.")


if __name__ == "__main__":
    main()
