#!/usr/bin/env python3
"""Verify the two-seed no-momentum-LayerNorm instability result."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

EXPECTED_TOKENS = 99_876_864
EXPECTED_SEEDS = {20306, 20307}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis_dir", type=Path)
    args = parser.parse_args()

    with (args.analysis_dir / "runs.csv").open(newline="") as handle:
        runs = list(csv.DictReader(handle))
    assert len(runs) == 4
    incumbent = [row for row in runs if row["candidate"] == "arch100_lnp_lnpend"]
    candidate = [row for row in runs if row["candidate"] == "arch100_lnp_lnpnone"]
    assert {int(row["seed"]) for row in incumbent} == EXPECTED_SEEDS
    assert {int(row["seed"]) for row in candidate} == EXPECTED_SEEDS
    assert all(row["status"] == "valid" and int(row["tokens"]) == EXPECTED_TOKENS for row in incumbent)
    assert all(row["status"] == "invalid" and "slurm_not_complete" in row["reasons"] for row in candidate)

    records = [json.loads(path.read_text()) for path in sorted((args.analysis_dir / "failures").glob("failure_s*.json"))]
    assert len(records) == 2
    assert {record["seed"] for record in records} == EXPECTED_SEEDS
    assert all(record["status"] == "failed_nonfinite" for record in records)
    assert all(record["failure_type"] == "training_loss" and record["value"] == "nan" for record in records)
    assert all(not record["checkpoint_resumable"] for record in records)
    assert all(0 < record["tokens_completed"] < EXPECTED_TOKENS for record in records)
    reached = {record["seed"]: record["tokens_completed"] / EXPECTED_TOKENS for record in records}
    print("PASS: incumbents completed; no-LN failed with NaN on both seeds; completion fractions=" +
          ", ".join(f"{seed}:{reached[seed]:.3f}" for seed in sorted(reached)))


if __name__ == "__main__":
    main()
