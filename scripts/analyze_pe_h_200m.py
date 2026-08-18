#!/usr/bin/env python3
"""Strict paired analysis and final PE-config lock for the 200M h bracket."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from analyze_pe_tuning import RUN_FIELDS, load_run, missing_run, parse_spec, write_csv
from make_pe_h_200m_specs import H_SEEDS, H_VALUES, h_rows, read_lr_lock


INCUMBENT_H = 0.10
REPLACE_THRESHOLD = -0.005


def expected_runs(lock: dict) -> list[dict]:
    expected = []
    for raw in h_rows(lock):
        item = parse_spec(raw)
        item["h_init"] = float(item["expected_args"]["presymp_h"])
        item["family"] = "pe_h_init"
        expected.append(item)
    return expected


def discover_runs(root: Path, lock: dict) -> list[dict]:
    manifests = {}
    for path in sorted(root.rglob("run_manifest.json")):
        try:
            name = json.loads(path.read_text())["args"]["run_name"]
        except (KeyError, json.JSONDecodeError):
            continue
        if name in manifests:
            raise SystemExit(f"duplicate run_name below {root}: {name}")
        manifests[name] = path
    expected = expected_runs(lock)
    rows = []
    for item in expected:
        path = manifests.pop(item["run_name"], None)
        row = load_run(path, item) if path else missing_run(item)
        row["h_init"] = item["h_init"]
        rows.append(row)
    for name in sorted(manifests):
        rows.append(
            {
                **{field: "" for field in RUN_FIELDS},
                "run_name": name,
                "status": "unexpected",
                "reasons": "not_in_predeclared_specs",
                "h_init": "",
            }
        )
    return rows


def provenance_match(left: dict, right: dict) -> bool:
    return bool(
        left
        and right
        and all(
            left[field] == right[field]
            for field in ("source_revision", "train_sha256", "val_sha256", "gpu")
        )
    )


def paired_results(runs: list[dict], expected_count: int) -> list[dict]:
    expected = runs[:expected_count]
    anchors = {row["seed"]: row for row in expected if math.isclose(float(row["h_init"]), INCUMBENT_H)}
    pairs = []
    for row in expected:
        if math.isclose(float(row["h_init"]), INCUMBENT_H):
            continue
        anchor = anchors.get(row["seed"])
        same_provenance = provenance_match(row, anchor)
        valid = bool(anchor and row["status"] == "valid" and anchor["status"] == "valid" and same_provenance)
        pairs.append(
            {
                "h_init": float(row["h_init"]),
                "seed": row["seed"],
                "candidate_status": row["status"],
                "anchor_status": anchor["status"] if anchor else "missing",
                "provenance_match": int(same_provenance),
                "candidate_final_val": row["final_val"],
                "anchor_final_val": anchor["final_val"] if anchor else "",
                "delta_final_val": float(row["final_val"]) - float(anchor["final_val"]) if valid else "",
                "delta_last3_val": float(row["last3_val"]) - float(anchor["last3_val"]) if valid else "",
                "valid_pair": int(valid),
            }
        )
    return sorted(pairs, key=lambda row: (float(row["h_init"]), int(row["seed"])))


def aggregate_results(pairs: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in pairs:
        groups[float(row["h_init"])].append(row)
    aggregate = []
    for h_init, rows in groups.items():
        valid = [row for row in rows if row["valid_pair"]]
        deltas = [float(row["delta_final_val"]) for row in valid]
        mean = statistics.fmean(deltas) if deltas else math.nan
        wins = sum(delta < 0 for delta in deltas)
        qualifies = len(valid) == len(H_SEEDS) and wins == len(H_SEEDS) and mean <= REPLACE_THRESHOLD
        aggregate.append(
            {
                "h_init": h_init,
                "n_expected": len(rows),
                "n_valid": len(valid),
                "wins_vs_h0p1": wins,
                "mean_delta_final_val": mean,
                "std_delta_final_val": statistics.stdev(deltas) if len(deltas) > 1 else math.nan,
                "mean_delta_last3_val": statistics.fmean(float(row["delta_last3_val"]) for row in valid) if valid else math.nan,
                "qualifies": int(qualifies),
                "selected": 0,
            }
        )
    qualifying = [row for row in aggregate if row["qualifies"]]
    if qualifying:
        min(qualifying, key=lambda row: (float(row["mean_delta_final_val"]), float(row["h_init"])))["selected"] = 1
    return sorted(aggregate, key=lambda row: float(row["h_init"]))


def complete_audit(runs: list[dict], lock: dict) -> bool:
    expected = runs[: len(expected_runs(lock))]
    provenance_fields = ("source_revision", "train_sha256", "val_sha256", "gpu")
    return (
        len(expected) == 8
        and all(row["status"] == "valid" for row in expected)
        and len({tuple(row[field] for field in provenance_fields) for row in expected}) == 1
        and all(row[field] == lock[field] for row in expected for field in provenance_fields)
    )


def final_lock_payload(lock: dict, aggregate: list[dict], runs: list[dict], complete: bool) -> dict | None:
    if not complete:
        return None
    selected = next((row for row in aggregate if row["selected"]), None)
    selected_h = float(selected["h_init"]) if selected else INCUMBENT_H
    pe = lock["methods"]["causal_symp_pe"]
    first = runs[0]
    return {
        "format_version": 1,
        "campaign": "pe_h_200m",
        "selection_complete": True,
        "selection_rule": "replace h=0.1 only for 2/2 wins and mean paired delta <= -0.005",
        "seeds": list(H_SEEDS),
        "actual_tokens_per_run": int(first["expected_tokens"]),
        "peak_lr": float(pe["peak_lr"]),
        "scalar_lr_mult": float(pe["scalar_lr_mult"]),
        "presymp_h": selected_h,
        "source_revision": first["source_revision"],
        "train_sha256": first["train_sha256"],
        "val_sha256": first["val_sha256"],
        "gpu": first["gpu"],
    }


def write_lock(path: Path, payload: dict | None) -> None:
    if payload is None:
        if path.exists():
            raise SystemExit(f"audit is incomplete but a stale final lock exists: {path}")
        return
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != rendered:
        raise SystemExit(f"refusing to overwrite a different final PE lock: {path}")
    path.write_text(rendered)


def analyze(root: Path, lock: dict) -> tuple[list[dict], list[dict], list[dict], bool]:
    runs = discover_runs(root, lock)
    pairs = paired_results(runs, len(expected_runs(lock)))
    aggregate = aggregate_results(pairs)
    return runs, pairs, aggregate, complete_audit(runs, lock)


def print_compact(runs: list[dict], pairs: list[dict], aggregate: list[dict], complete: bool, lock: dict) -> None:
    expected = runs[: len(expected_runs(lock))]
    valid = sum(row["status"] == "valid" for row in expected)
    missing = sum(row["status"] == "missing" for row in expected)
    invalid = sum(row["status"] == "invalid" for row in expected)
    print(f"expected=8 valid={valid} missing={missing} invalid={invalid}")
    print("\n200M PE initial-step bracket (negative is better than h=0.1):")
    print(f"{'H_INIT':>8s} {'N':>3s} {'WINS':>4s} {'MEAN_DNLL':>11s} {'LOCK':>6s}")
    for row in aggregate:
        mean = row["mean_delta_final_val"]
        rendered = f"{mean:+.6f}" if math.isfinite(mean) else "NA"
        print(
            f"{float(row['h_init']):8.3f} {row['n_valid']:3d} {row['wins_vs_h0p1']:4d} "
            f"{rendered:>11s} {'yes' if row['selected'] else 'no':>6s}"
        )
    locked_h = next((float(row["h_init"]) for row in aggregate if row["selected"]), INCUMBENT_H)
    print(f"\nselection_complete={'yes' if complete else 'no'} locked_h={locked_h:g}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--lock_file", type=Path)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--out_dir", type=Path)
    parser.add_argument("--final_lock_out", type=Path)
    args = parser.parse_args()
    lock_file = args.lock_file or args.root / "locked_lrs_input.json"
    spec = args.spec or args.root / "pe_h_200m.tsv"
    lock = read_lr_lock(lock_file)
    expected_text = "\n".join(h_rows(lock)) + "\n"
    if not spec.is_file() or spec.read_text() != expected_text:
        raise SystemExit(f"PE-h specification does not match {lock_file}: {spec}")
    out_dir = args.out_dir or args.root / "analysis"
    final_lock_out = args.final_lock_out or out_dir / "locked_pe_config.json"
    runs, pairs, aggregate, complete = analyze(args.root, lock)
    write_csv(out_dir / "runs.csv", runs, RUN_FIELDS + ["h_init"])
    write_csv(out_dir / "paired_vs_h0p1.csv", pairs)
    write_csv(out_dir / "aggregate.csv", aggregate)
    write_lock(final_lock_out, final_lock_payload(lock, aggregate, runs, complete))
    print_compact(runs, pairs, aggregate, complete, lock)
    print(f"\nwrote analysis tables to {out_dir}")
    if complete:
        print(f"wrote immutable final PE lock to {final_lock_out}")
    else:
        print("WARNING: no final PE lock written; all eight runs and one provenance cohort are required.")


if __name__ == "__main__":
    main()
