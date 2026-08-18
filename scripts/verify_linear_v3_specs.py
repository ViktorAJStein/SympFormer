#!/usr/bin/env python3
"""Verify E024 causal linear-v3 screen factor matching and routing."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_pe_tuning import parse_spec  # noqa: E402
from make_linear_v3_specs import METHODS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    parser.add_argument("--tokens", type=int, default=20_000_000)
    parser.add_argument("--expected_seeds", type=int, default=2)
    parser.add_argument("--amp_dtype", default="float16")
    args = parser.parse_args()
    rows = [parse_spec(line) for line in args.spec.read_text().splitlines() if line.strip()]
    groups: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        groups[int(row["seed"])].append(row)
        assert row["dataset"] == "tinystories"
        assert row["config"] == "configs/linear_v3_small.json"
        expected = row["expected_args"]
        assert expected.get("lin_noncausal", False) is False
        assert expected.get("no_v0_init") is True
        assert expected.get("max_tokens") == args.tokens
        assert expected.get("batch_size") == 8
        assert expected.get("presymp_h") == 0.1
        assert expected.get("eta_mode") == "log"
        assert expected.get("eta_learnable") is False
        assert expected.get("eta_log_coef") == 3
        assert expected.get("amp_dtype") == args.amp_dtype
    assert len(groups) == args.expected_seeds
    for seed, items in groups.items():
        assert sorted(row["arch"] for row in items) == sorted(METHODS), seed
    assert len({row["run_name"] for row in rows}) == len(rows)
    print(f"PASS: {len(rows)} causal rows, {len(groups)} seeds, {len(METHODS)} linear methods")


if __name__ == "__main__":
    main()
