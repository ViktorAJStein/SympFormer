#!/usr/bin/env python3
"""Strict paired audit and immutable locks for staged PE damping tuning."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from analyze_pe_tuning import RUN_FIELDS, load_run, missing_run, parse_spec, write_csv
from make_pe_damping_200m_specs import (
    INCUMBENTS,
    OUTPUT_CAMPAIGNS,
    REPLACE_THRESHOLD,
    STAGES,
    STAGE_SEEDS,
    read_stage_lock,
    stage_rows,
    stage_values,
)


LOCK_NAMES = {
    "d0": "locked_damping_mode.json",
    "d1": "locked_damping_log.json",
    "d2": "locked_pe_damping_config.json",
}


def expected_runs(stage: str, lock: dict) -> list[dict]:
    expected = []
    values = stage_values(stage)
    for raw, value in zip(stage_rows(stage, lock), values * len(STAGE_SEEDS[stage])):
        item = parse_spec(raw)
        item["setting"] = value
        item["family"] = f"pe_damping_{stage}"
        expected.append(item)
    return expected


def discover_runs(root: Path, stage: str, lock: dict) -> list[dict]:
    manifests = {}
    for path in sorted(root.rglob("run_manifest.json")):
        try:
            name = json.loads(path.read_text())["args"]["run_name"]
        except (KeyError, json.JSONDecodeError):
            continue
        if name in manifests:
            raise SystemExit(f"duplicate run_name below {root}: {name}")
        manifests[name] = path
    expected = expected_runs(stage, lock)
    rows = []
    for item in expected:
        path = manifests.pop(item["run_name"], None)
        result = load_run(path, item) if path else missing_run(item)
        result["setting"] = item["setting"]
        rows.append(result)
    for name in sorted(manifests):
        rows.append(
            {
                **{field: "" for field in RUN_FIELDS},
                "run_name": name,
                "status": "unexpected",
                "reasons": "not_in_predeclared_specs",
                "setting": "",
            }
        )
    return rows


def provenance_match(left: dict, right: dict) -> bool:
    fields = ("source_revision", "train_sha256", "val_sha256", "gpu")
    return bool(left and right and all(left[field] == right[field] for field in fields))


def same_setting(left, right) -> bool:
    if isinstance(left, str) or isinstance(right, str):
        return str(left) == str(right)
    return math.isclose(float(left), float(right), rel_tol=0, abs_tol=1e-12)


def paired_results(runs: list[dict], stage: str, expected_count: int) -> list[dict]:
    expected = runs[:expected_count]
    incumbent = INCUMBENTS[stage]
    anchors = {row["seed"]: row for row in expected if same_setting(row["setting"], incumbent)}
    pairs = []
    for row in expected:
        if same_setting(row["setting"], incumbent):
            continue
        anchor = anchors.get(row["seed"])
        same_provenance = provenance_match(row, anchor)
        valid = bool(
            anchor
            and row["status"] == "valid"
            and anchor["status"] == "valid"
            and same_provenance
        )
        pairs.append(
            {
                "setting": row["setting"],
                "seed": row["seed"],
                "candidate_status": row["status"],
                "anchor_status": anchor["status"] if anchor else "missing",
                "provenance_match": int(same_provenance),
                "candidate_final_val": row["final_val"],
                "anchor_final_val": anchor["final_val"] if anchor else "",
                "delta_final_val": (
                    float(row["final_val"]) - float(anchor["final_val"]) if valid else ""
                ),
                "delta_last3_val": (
                    float(row["last3_val"]) - float(anchor["last3_val"]) if valid else ""
                ),
                "delta_val_auc": (
                    float(row["val_auc"]) - float(anchor["val_auc"]) if valid else ""
                ),
                "valid_pair": int(valid),
            }
        )
    return sorted(pairs, key=lambda row: (str(row["setting"]), int(row["seed"])))


def aggregate_results(pairs: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in pairs:
        groups[str(row["setting"])].append(row)
    aggregate = []
    for _, rows in groups.items():
        setting = rows[0]["setting"]
        valid = [row for row in rows if row["valid_pair"]]
        deltas = [float(row["delta_final_val"]) for row in valid]
        mean = statistics.fmean(deltas) if deltas else math.nan
        wins = sum(delta < 0 for delta in deltas)
        qualifies = (
            len(valid) == 2 and wins == 2 and mean <= REPLACE_THRESHOLD
        )
        aggregate.append(
            {
                "setting": setting,
                "n_expected": len(rows),
                "n_valid": len(valid),
                "wins_vs_incumbent": wins,
                "mean_delta_final_val": mean,
                "std_delta_final_val": (
                    statistics.stdev(deltas) if len(deltas) > 1 else math.nan
                ),
                "mean_delta_last3_val": (
                    statistics.fmean(float(row["delta_last3_val"]) for row in valid)
                    if valid
                    else math.nan
                ),
                "mean_delta_val_auc": (
                    statistics.fmean(float(row["delta_val_auc"]) for row in valid)
                    if valid
                    else math.nan
                ),
                "qualifies": int(qualifies),
                "selected": 0,
            }
        )
    qualifying = [row for row in aggregate if row["qualifies"]]
    if qualifying:
        min(
            qualifying,
            key=lambda row: (float(row["mean_delta_final_val"]), str(row["setting"])),
        )["selected"] = 1
    return sorted(aggregate, key=lambda row: str(row["setting"]))


def complete_audit(runs: list[dict], stage: str, lock: dict) -> bool:
    expected = runs[: len(expected_runs(stage, lock))]
    fields = ("source_revision", "train_sha256", "val_sha256", "gpu")
    return (
        len(expected) == len(stage_rows(stage, lock))
        and all(row["status"] == "valid" for row in expected)
        and len({tuple(row[field] for field in fields) for row in expected}) == 1
        and all(row[field] == lock[field] for row in expected for field in fields)
    )


def locked_setting(stage: str, aggregate: list[dict]):
    selected = next((row["setting"] for row in aggregate if row["selected"]), None)
    return INCUMBENTS[stage] if selected is None else selected


def lock_payload(
    stage: str, lock: dict, aggregate: list[dict], runs: list[dict], complete: bool
) -> dict | None:
    if not complete:
        return None
    selected = locked_setting(stage, aggregate)
    payload = {
        "format_version": 1,
        "campaign": OUTPUT_CAMPAIGNS[stage],
        "selection_complete": True,
        "selection_rule": (
            "replace incumbent only for 2/2 paired wins and mean final-NLL delta <= -0.005"
        ),
        "stage": stage,
        "seeds": list(STAGE_SEEDS[stage]),
        "actual_tokens_per_run": int(runs[0]["expected_tokens"]),
        "peak_lr": float(lock["peak_lr"]),
        "scalar_lr_mult": float(lock["scalar_lr_mult"]),
        "presymp_h": float(lock["presymp_h"]),
        "source_revision": runs[0]["source_revision"],
        "train_sha256": runs[0]["train_sha256"],
        "val_sha256": runs[0]["val_sha256"],
        "gpu": runs[0]["gpu"],
    }
    if stage == "d0":
        payload.update(
            damping_learnable=str(selected) == "learned",
            eta_log=3.0,
            eta_lin=0.0001,
        )
    elif stage == "d1":
        payload.update(
            damping_learnable=bool(lock["damping_learnable"]),
            eta_log=float(selected),
            eta_lin=0.0001,
        )
    else:
        payload.update(
            damping_learnable=bool(lock["damping_learnable"]),
            eta_log=float(lock["eta_log"]),
            eta_lin=float(selected),
        )
    return payload


def write_lock(path: Path, payload: dict | None) -> None:
    if payload is None:
        if path.exists():
            raise SystemExit(f"audit is incomplete but a stale damping lock exists: {path}")
        return
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != rendered:
        raise SystemExit(f"refusing to overwrite a different damping lock: {path}")
    path.write_text(rendered)


def analyze(root: Path, stage: str, lock: dict):
    runs = discover_runs(root, stage, lock)
    pairs = paired_results(runs, stage, len(expected_runs(stage, lock)))
    aggregate = aggregate_results(pairs)
    complete = complete_audit(runs, stage, lock)
    return runs, pairs, aggregate, complete


def print_compact(runs, aggregate, complete: bool, stage: str, lock: dict) -> None:
    expected = runs[: len(expected_runs(stage, lock))]
    valid = sum(row["status"] == "valid" for row in expected)
    missing = sum(row["status"] == "missing" for row in expected)
    invalid = sum(row["status"] == "invalid" for row in expected)
    print(f"expected={len(expected)} valid={valid} missing={missing} invalid={invalid}")
    print(f"\n200M PE damping {stage} (negative is better than incumbent):")
    print(f"{'SETTING':>12s} {'N':>3s} {'WINS':>4s} {'MEAN_DNLL':>11s} {'LOCK':>6s}")
    for row in aggregate:
        mean = row["mean_delta_final_val"]
        rendered = f"{mean:+.6f}" if math.isfinite(mean) else "NA"
        print(
            f"{str(row['setting']):>12s} {row['n_valid']:3d} "
            f"{row['wins_vs_incumbent']:4d} {rendered:>11s} "
            f"{'yes' if row['selected'] else 'no':>6s}"
        )
    print(
        f"\nselection_complete={'yes' if complete else 'no'} "
        f"locked_setting={locked_setting(stage, aggregate)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--lock_file", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path)
    parser.add_argument("--lock_out", type=Path)
    args = parser.parse_args()
    lock = read_stage_lock(args.lock_file, args.stage)
    expected_text = "\n".join(stage_rows(args.stage, lock)) + "\n"
    if not args.spec.is_file() or args.spec.read_text() != expected_text:
        raise SystemExit("damping specification differs from the upstream lock")
    out_dir = args.out_dir or args.root / "analysis"
    lock_out = args.lock_out or out_dir / LOCK_NAMES[args.stage]
    runs, pairs, aggregate, complete = analyze(args.root, args.stage, lock)
    write_csv(out_dir / "runs.csv", runs, RUN_FIELDS + ["setting"])
    write_csv(out_dir / "paired_vs_incumbent.csv", pairs)
    write_csv(out_dir / "aggregate.csv", aggregate)
    write_lock(lock_out, lock_payload(args.stage, lock, aggregate, runs, complete))
    print_compact(runs, aggregate, complete, args.stage, lock)
    print(f"\nwrote analysis tables to {out_dir}")
    if complete:
        print(f"wrote immutable damping lock to {lock_out}")
    else:
        print("WARNING: no lock written; all stage runs and one provenance cohort are required.")


if __name__ == "__main__":
    main()
