#!/usr/bin/env python3
"""Strict audit for the leak-matched E023 linear-v3 small screen."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_pe_tuning import (  # noqa: E402
    RUN_FIELDS, data_sha256, missing_run, normalized_auc, parse_spec,
    read_metrics, read_slurm_status, source_identity, values_match, write_csv,
)
from make_linear_v3_specs import METHODS  # noqa: E402


def load_linear_run(path: Path, expected: dict) -> dict:
    run_dir = path.parent
    manifest = json.loads(path.read_text())
    run_args = manifest["args"]
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    metrics = read_metrics(run_dir / "metrics.csv")
    tokens = int(summary.get("tokens", metrics[-1]["tokens"] if metrics else 0))
    final_val = float(summary.get("final_val", metrics[-1]["val"] if metrics else math.nan))
    wall = float(summary.get("wall_cum_s", metrics[-1]["wall"] if metrics else math.nan))
    source, source_ok = source_identity(manifest)
    train_sha, val_sha = data_sha256(manifest, "train"), data_sha256(manifest, "val")
    state, exit_code = read_slurm_status(run_dir / "slurm_status.tsv")
    cfg = manifest.get("model_config") or {}
    tokens_per_step = int(run_args["batch_size"]) * int(cfg["block_size"]) * int(run_args["grad_accum_steps"])
    expected_tokens = (int(run_args["max_tokens"]) // tokens_per_step) * tokens_per_step
    reasons = []
    if not summary_path.is_file(): reasons.append("missing_summary")
    if not metrics: reasons.append("missing_validation_rows")
    if tokens != expected_tokens: reasons.append("token_mismatch")
    if state != "complete" or exit_code != 0: reasons.append("slurm_not_complete")
    if not source_ok or not train_sha or not val_sha: reasons.append("missing_provenance")
    if any(not values_match(run_args.get(key), value) for key, value in expected["expected_args"].items()):
        reasons.append("spec_mismatch")
    device = str(run_args.get("device", ""))
    if (
        run_args.get("dataset") != expected["dataset"]
        or run_args.get("arch") != expected["arch"]
        or int(run_args.get("seed", -1)) != expected["seed"]
        or run_args.get("config") != expected["config"]
        or int(cfg.get("n_layer", -1)) != 4
        or int(cfg.get("n_head", -1)) != 4
        or int(cfg.get("n_embd", -1)) != 64
        or int(cfg.get("block_size", -1)) != 128
        or not (device == "cpu" or device.startswith("cuda"))
        or run_args.get("amp_dtype") not in ("float16", "bfloat16")
        or (device.startswith("cuda") and not manifest.get("gpu"))
    ):
        reasons.append("protocol_mismatch")
    last3 = statistics.fmean(row["val"] for row in metrics[-3:]) if metrics else math.nan
    gpu_record = manifest.get("gpu")
    gpu_name = gpu_record.get("name", "UNKNOWN") if isinstance(gpu_record, dict) else (gpu_record or "CPU")
    return {
        "run_name": run_args.get("run_name", expected["run_name"]),
        "candidate": expected["candidate"], "family": expected["family"],
        "arch": run_args.get("arch", expected["arch"]), "seed": run_args.get("seed", expected["seed"]),
        "status": "valid" if not reasons else "invalid", "reasons": ";".join(reasons),
        "tokens": tokens, "expected_tokens": expected_tokens, "final_val": final_val,
        "best_val": summary.get("best_val", math.nan), "last3_val": last3,
        "val_auc": normalized_auc(metrics), "gpu_hours": wall / 3600.0,
        "tokens_per_second": tokens / wall if wall > 0 else math.nan,
        "peak_memory_mb": summary.get("peak_memory_mb", 0.0),
        "trainable_parameters": manifest.get("trainable_parameters", math.nan),
        "max_leak_warnings": max((row["leaks"] for row in metrics), default=0),
        "slurm_exit_code": exit_code, "gpu": gpu_name, "source_revision": source,
        "train_sha256": train_sha, "val_sha256": val_sha, "run_dir": str(run_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    expected = [parse_spec(line) for line in args.spec.read_text().splitlines() if line.strip()]
    manifests = {}
    for path in args.root.rglob("run_manifest.json"):
        run_name = json.loads(path.read_text())["args"]["run_name"]
        if run_name in manifests:
            raise SystemExit(f"duplicate run_name {run_name}")
        manifests[run_name] = path
    runs = []
    for item in expected:
        path = manifests.pop(item["run_name"], None)
        runs.append(load_linear_run(path, item) if path else missing_run(item))
    if manifests:
        raise SystemExit(f"unexpected runs: {sorted(manifests)}")

    provenance = {
        (row.get("source_revision", ""), row.get("train_sha256", ""), row.get("val_sha256", ""), row.get("gpu", ""))
        for row in runs if row["status"] == "valid"
    }
    aggregates = []
    pairs = []
    by_method: dict[str, list[dict]] = defaultdict(list)
    for row in runs:
        by_method[row["arch"]].append(row)
    for method in METHODS:
        rows = by_method[method]
        valid = [row for row in rows if row["status"] == "valid"]
        values = [float(row["final_val"]) for row in valid]
        aggregates.append({
            "arch": method,
            "n_expected": len(rows),
            "n_valid": len(valid),
            "mean_final_val": statistics.fmean(values) if values else math.nan,
            "std_final_val": statistics.stdev(values) if len(values) > 1 else math.nan,
            "mean_tokens_per_second": statistics.fmean(float(row["tokens_per_second"]) for row in valid) if valid else math.nan,
            "mean_peak_memory_mb": statistics.fmean(float(row["peak_memory_mb"]) for row in valid) if valid else math.nan,
            "mean_parameters": statistics.fmean(float(row["trainable_parameters"]) for row in valid) if valid else math.nan,
        })
    by_seed_arch = {(int(row["seed"]), row["arch"]): row for row in runs}
    for seed in sorted({int(row["seed"]) for row in runs}):
        baseline = by_seed_arch[(seed, "lin_baseline")]
        for method in METHODS:
            row = by_seed_arch[(seed, method)]
            valid = row["status"] == baseline["status"] == "valid"
            pairs.append({
                "seed": seed,
                "arch": method,
                "valid_pair": int(valid),
                "final_val": row.get("final_val", math.nan),
                "baseline_final_val": baseline.get("final_val", math.nan),
                "delta_vs_lin_baseline": float(row["final_val"]) - float(baseline["final_val"]) if valid else math.nan,
            })
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "runs.csv", runs, RUN_FIELDS)
    write_csv(args.out_dir / "aggregate.csv", aggregates)
    write_csv(args.out_dir / "pairs.csv", pairs)
    valid_count = sum(row["status"] == "valid" for row in runs)
    print(f"expected={len(runs)} valid={valid_count} provenance_cohorts={len(provenance)}")
    for row in sorted(aggregates, key=lambda item: float(item["mean_final_val"])):
        print(f"{row['arch']:24s} n={row['n_valid']} final={float(row['mean_final_val']):.6f} tok/s={float(row['mean_tokens_per_second']):.1f}")


if __name__ == "__main__":
    main()
