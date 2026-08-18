#!/usr/bin/env python3
"""Verify factor matching in the E025 causal softmax screen."""
from __future__ import annotations
import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_pe_tuning import parse_spec  # noqa: E402
from make_riemannian_softmax_specs import METHODS  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    parser.add_argument("--tokens", type=int, default=20_000_000)
    parser.add_argument("--expected_seeds", type=int, default=2)
    parser.add_argument("--amp_dtype", default="float16")
    args = parser.parse_args()
    rows = [parse_spec(line) for line in args.spec.read_text().splitlines() if line]
    groups = defaultdict(list)
    for row in rows:
        groups[row["seed"]].append(row)
        assert row["dataset"] == "tinystories"
        assert row["config"] == "configs/linear_v3_small.json"
        expected = row["expected_args"]
        assert expected["max_tokens"] == args.tokens
        assert expected["amp_dtype"] == args.amp_dtype
        assert expected["batch_size"] == 8
        assert expected["presymp_h"] == 0.1
        assert expected["presymp_t0"] == 1
        assert expected["eta_mode"] == "log"
        assert expected["eta_log_coef"] == 3
        assert expected["eta_lin_coef"] == 0
        assert expected["eta_learnable"] is False
        assert expected["no_v0_init"] is True
    assert len(groups) == args.expected_seeds
    for seed, group in groups.items():
        assert sorted(item["arch"] for item in group) == sorted(METHODS), seed
    assert len({item["run_name"] for item in rows}) == len(rows)
    print(f"PASS: {len(rows)} causal rows, {len(groups)} seeds, {len(METHODS)} methods")


if __name__ == "__main__":
    main()
