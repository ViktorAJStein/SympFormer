#!/usr/bin/env python3
"""Verify architecture analyzer outputs are internally consistent."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis_dir", type=Path)
    parser.add_argument("--expected-runs", type=int, required=True)
    parser.add_argument("--expected-valid", type=int)
    args = parser.parse_args()

    runs = read_rows(args.analysis_dir / "runs.csv")
    pairs = read_rows(args.analysis_dir / "paired_vs_incumbent.csv")
    aggregate = read_rows(args.analysis_dir / "aggregate.csv")
    assert len(runs) == args.expected_runs
    expected_valid = args.expected_runs if args.expected_valid is None else args.expected_valid
    valid_runs = [row for row in runs if row["status"] == "valid"]
    assert len(valid_runs) == expected_valid

    grouped: dict[str, list[float]] = defaultdict(list)
    for row in pairs:
        if row["valid_pair"] == "1":
            assert row["provenance_match"] == "1"
            grouped[row["candidate"]].append(float(row["delta_final_val"]))
        else:
            assert row["candidate_status"] != "valid" or row["anchor_status"] != "valid" or row["provenance_match"] != "1"
    assert set(grouped).issubset({row["candidate"] for row in aggregate})
    for row in aggregate:
        deltas = grouped[row["candidate"]]
        assert int(row["n_valid"]) == len(deltas)
        assert int(row["wins"]) == sum(value < 0 for value in deltas)
        if deltas:
            mean = sum(deltas) / len(deltas)
            assert math.isclose(float(row["mean_delta_final_val"]), mean, rel_tol=0, abs_tol=1e-12)
            expected_advance = int(all(value < 0 for value in deltas) and mean <= -0.010)
        else:
            assert math.isnan(float(row["mean_delta_final_val"]))
            expected_advance = 0
        assert int(row["advance_100m_pruning"]) == expected_advance
    print(f"PASS: {len(runs)} audited runs, {len(valid_runs)} valid, {sum(len(x) for x in grouped.values())} valid pairs; aggregates consistent")


if __name__ == "__main__":
    main()
