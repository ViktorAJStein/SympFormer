#!/usr/bin/env python3
"""Verify exact-step seed averaging used by architecture trajectory plots."""
from __future__ import annotations
import csv
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from plot_mean_nll_trajectories import aggregate  # noqa: E402


def write_metrics(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "tokens_cum", "val_loss"])
        writer.writeheader(); writer.writerows(rows)
    (path.parent / "summary.json").write_text("{}")


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        specs = [
            {"arch": "baseline", "seed": 1, "run_name": "b1"},
            {"arch": "baseline", "seed": 2, "run_name": "b2"},
            {"arch": "candidate", "seed": 1, "run_name": "c1"},
        ]
        write_metrics(root / "baseline_b1/metrics.csv", [
            {"step": 10, "tokens_cum": 100, "val_loss": 3.0},
            {"step": 20, "tokens_cum": 200, "val_loss": 2.0},
            {"step": 30, "tokens_cum": 300, "val_loss": 1.5},
        ])
        write_metrics(root / "baseline_b2/metrics.csv", [
            {"step": 10, "tokens_cum": 100, "val_loss": 5.0},
            {"step": 30, "tokens_cum": 300, "val_loss": 2.5},
        ])
        write_metrics(root / "candidate_c1/metrics.csv", [
            {"step": 10, "tokens_cum": 100, "val_loss": 4.5},
        ])
        rows = aggregate(root, specs, allow_incomplete=False)
        baseline = [row for row in rows if row["arch"] == "baseline"]
        assert [row["step"] for row in baseline] == [10, 30]
        assert [row["mean_nll"] for row in baseline] == [4.0, 2.0]
        assert [row["min_nll"] for row in baseline] == [3.0, 1.5]
        assert [row["max_nll"] for row in baseline] == [5.0, 2.5]
        assert all(row["n"] == 2 for row in baseline)
        candidate = [row for row in rows if row["arch"] == "candidate"]
        assert len(candidate) == 1 and candidate[0]["n"] == 1
    print("PASS: trajectory plots use exact common iterations and unsmoothed seed means/ranges")


if __name__ == "__main__":
    main()
