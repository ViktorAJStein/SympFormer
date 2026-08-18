#!/usr/bin/env python3
"""Strict audit for architecture-choice pruning screens."""

from __future__ import annotations

import argparse
import math
import statistics
from collections import defaultdict
from pathlib import Path

from analyze_pe_tuning import RUN_FIELDS, load_run, missing_run, parse_spec, write_csv

INCUMBENT = {"mlp": "arch100_mlp_attnvel", "lnp": "arch100_lnp_lnpend", "disc": "arch100_disc_pe", "lookahead": "arch100_lookahead_none", "ab2": "arch100_ab2_pe"}


def expected_runs(spec: Path) -> list[dict]:
    return [parse_spec(line) for line in spec.read_text().splitlines() if line.strip()]


def discover(root: Path, expected: list[dict]) -> list[dict]:
    import json
    manifests = {}
    for path in sorted(root.rglob("run_manifest.json")):
        try:
            name = json.loads(path.read_text())["args"]["run_name"]
        except Exception:
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


def provenance(row: dict) -> tuple:
    return tuple(row.get(k, "") for k in ("source_revision", "train_sha256", "val_sha256", "gpu"))


def paired(runs: list[dict], expected_count: int, screen: str) -> list[dict]:
    expected = runs[:expected_count]
    inc = INCUMBENT[screen]
    by_seed_candidate = {(int(r["seed"]), r["candidate"]): r for r in expected}
    seeds = sorted({int(r["seed"]) for r in expected})
    candidates = sorted({r["candidate"] for r in expected if r["candidate"] != inc})
    out = []
    for seed in seeds:
        anchor = by_seed_candidate.get((seed, inc))
        for cand in candidates:
            row = by_seed_candidate.get((seed, cand))
            valid = bool(row and anchor and row["status"] == "valid" and anchor["status"] == "valid" and provenance(row) == provenance(anchor))
            out.append({
                "seed": seed,
                "candidate": cand,
                "candidate_status": row["status"] if row else "missing",
                "anchor_status": anchor["status"] if anchor else "missing",
                "provenance_match": int(valid),
                "candidate_final_val": row.get("final_val", math.nan) if row else math.nan,
                "anchor_final_val": anchor.get("final_val", math.nan) if anchor else math.nan,
                "delta_final_val": float(row["final_val"]) - float(anchor["final_val"]) if valid else math.nan,
                "delta_last3_val": float(row["last3_val"]) - float(anchor["last3_val"]) if valid else math.nan,
                "delta_val_auc": float(row["val_auc"]) - float(anchor["val_auc"]) if valid else math.nan,
                "valid_pair": int(valid),
            })
    return out


def aggregate(pairs: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in pairs:
        groups[row["candidate"]].append(row)
    out = []
    for cand, rows in sorted(groups.items()):
        valid = [r for r in rows if r["valid_pair"]]
        vals = [float(r["delta_final_val"]) for r in valid]
        out.append({
            "candidate": cand,
            "n_expected": len(rows),
            "n_valid": len(valid),
            "wins": sum(v < 0 for v in vals),
            "mean_delta_final_val": statistics.fmean(vals) if vals else math.nan,
            "std_delta_final_val": statistics.stdev(vals) if len(vals) > 1 else math.nan,
            "mean_delta_last3_val": statistics.fmean(float(r["delta_last3_val"]) for r in valid) if valid else math.nan,
            "mean_delta_val_auc": statistics.fmean(float(r["delta_val_auc"]) for r in valid) if valid else math.nan,
            "advance_100m_pruning": int(len(valid) == len(rows) and sum(v < 0 for v in vals) == len(vals) and vals and statistics.fmean(vals) <= -0.010),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--screen", choices=["mlp", "lnp", "disc", "lookahead", "ab2"], required=True)
    ap.add_argument("--out_dir", type=Path)
    args = ap.parse_args()
    expected = expected_runs(args.spec)
    runs = discover(args.root, expected)
    pairs = paired(runs, len(expected), args.screen)
    agg = aggregate(pairs)
    out = args.out_dir or args.root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "runs.csv", runs, RUN_FIELDS)
    write_csv(out / "paired_vs_incumbent.csv", pairs)
    write_csv(out / "aggregate.csv", agg)
    valid = sum(r["status"] == "valid" for r in runs[: len(expected)])
    print(f"expected={len(expected)} valid={valid} missing={sum(r['status']=='missing' for r in runs[:len(expected)])} invalid={sum(r['status']=='invalid' for r in runs[:len(expected)])}")
    print("Architecture screen (negative improves over incumbent):")
    for r in agg:
        print(f"{r['candidate']:24s} n={r['n_valid']} wins={r['wins']} mean={float(r['mean_delta_final_val']):+.6f} advance={r['advance_100m_pruning']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
