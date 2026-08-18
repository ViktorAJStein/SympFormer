#!/usr/bin/env python3
"""Verify high-iteration headline comparison TSVs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from analyze_pe_tuning import parse_spec

EXPECTED_ARCHES = ("baseline", "yurii_lt", "causal_symp_pe")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--expected_seeds", type=int, default=None)
    parser.add_argument("--dataset", default="tinystories")
    args = parser.parse_args()

    raw_rows = [line for line in args.spec.read_text().splitlines() if line.strip()]
    parsed = [parse_spec(line) for line in raw_rows]
    if len({row["run_name"] for row in parsed}) != len(parsed):
        raise SystemExit("duplicate run_name in high-iteration spec")
    groups: dict[int, list[dict]] = defaultdict(list)
    for row in parsed:
        groups[int(row["seed"])].append(row)
        if row["dataset"] != args.dataset:
            raise SystemExit(f"unexpected dataset for {row['run_name']}: {row['dataset']}")
        if row["config"] != "configs/core_50m.json":
            raise SystemExit(f"unexpected config for {row['run_name']}: {row['config']}")
        if int(row["expected_args"].get("max_tokens", -1)) != args.tokens:
            raise SystemExit(f"wrong token cap for {row['run_name']}")
        if int(row["expected_args"].get("batch_size", -1)) != 8:
            raise SystemExit(f"wrong batch_size for {row['run_name']}")
        expected = row["expected_args"]
        if expected.get("sample_final") is not True or expected.get("sample_do_sample") != 0:
            raise SystemExit(f"{row['run_name']} lacks deterministic final sampling")
        if expected.get("sample_max_new_tokens") != 256 or expected.get("sample_prefix_tokens") != 64:
            raise SystemExit(f"{row['run_name']} has wrong sample length/prompt settings")
    if args.expected_seeds is not None and len(groups) != args.expected_seeds:
        raise SystemExit(f"expected {args.expected_seeds} seeds, found {len(groups)}")
    for seed, rows in sorted(groups.items()):
        arches = sorted(row["arch"] for row in rows)
        if arches != sorted(EXPECTED_ARCHES):
            raise SystemExit(f"seed {seed} has arches {arches}, expected {EXPECTED_ARCHES}")
        by_arch = {row["arch"]: row for row in rows}
        if "scalar_lr_mult" in by_arch["baseline"]["expected_args"]:
            raise SystemExit("baseline should not use scalar_lr_mult")
        if by_arch["yurii_lt"]["expected_args"].get("no_v0_init") is not True:
            raise SystemExit("YuriiFormer row missing --no_v0_init")
        pe = by_arch["causal_symp_pe"]
        for key in ("presymp_h", "eta_mode", "eta_log_init", "eta_lin_init"):
            if key not in pe["expected_args"]:
                raise SystemExit(f"PE row missing {key}")
        if pe["expected_args"].get("presymp_mlp_use_attn_vel") is not True:
            raise SystemExit("PE row missing --presymp_mlp_use_attn_vel")
    print(
        f"PASS: {len(parsed)} high-iteration rows, {len(groups)} paired seeds, "
        f"dataset={args.dataset}, tokens={args.tokens}"
    )


if __name__ == "__main__":
    main()
