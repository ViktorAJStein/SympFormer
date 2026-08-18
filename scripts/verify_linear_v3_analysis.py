#!/usr/bin/env python3
"""Verify consistency of E023 linear-v3 analysis tables."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


def read(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open()))


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis_dir", type=Path)
    parser.add_argument("--expected", type=int, default=18)
    args = parser.parse_args()
    runs, aggregates, pairs = (
        read(args.analysis_dir / "runs.csv"),
        read(args.analysis_dir / "aggregate.csv"),
        read(args.analysis_dir / "pairs.csv"),
    )
    assert len(runs) == args.expected
    assert len(aggregates) == 9
    assert len(pairs) == args.expected
    assert all(row["status"] == "valid" for row in runs)
    by_arch = {}
    for row in runs:
        by_arch.setdefault(row["arch"], []).append(row)
    for aggregate in aggregates:
        valid = by_arch[aggregate["arch"]]
        assert int(aggregate["n_valid"]) == len(valid) == 2
        mean = statistics.fmean(float(row["final_val"]) for row in valid)
        assert close(float(aggregate["mean_final_val"]), mean)
    by_seed_arch = {(int(row["seed"]), row["arch"]): row for row in runs}
    for pair in pairs:
        seed, arch = int(pair["seed"]), pair["arch"]
        expected_delta = float(by_seed_arch[(seed, arch)]["final_val"]) - float(
            by_seed_arch[(seed, "lin_baseline")]["final_val"]
        )
        assert int(pair["valid_pair"]) == 1
        assert close(float(pair["delta_vs_lin_baseline"]), expected_delta)
    provenance = {
        (row["source_revision"], row["train_sha256"], row["val_sha256"], row["gpu"])
        for row in runs
    }
    assert len(provenance) == 1, provenance
    print(f"PASS: {len(runs)} valid E023 runs, 9 methods, 2 seeds, one provenance cohort")


if __name__ == "__main__":
    main()
