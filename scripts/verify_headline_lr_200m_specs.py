#!/usr/bin/env python3
"""Verify the equal-budget, one-factor 200M LR matrix."""

from __future__ import annotations

import argparse
import math
import shlex
from collections import Counter
from pathlib import Path

from make_headline_lr_200m_specs import (
    CONFIG,
    METHOD_LRS,
    SCALAR_PEAK_LR,
    TOKENS_200M,
    TUNING_SEEDS,
    lr_rows,
)


EXPECTED_ACTUAL_TOKENS = 199_753_728


def parse(raw: str) -> dict:
    fields = raw.split("\t")
    if len(fields) != 6:
        raise AssertionError(f"expected six TSV fields: {raw}")
    dataset, arch, config, seed, run_name, flags = fields
    return {
        "dataset": dataset,
        "arch": arch,
        "config": config,
        "seed": int(seed),
        "run_name": run_name,
        "argv": shlex.split(flags),
    }


def flag_value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=Path("jobs/headline_lr_200m.tsv"))
    args = parser.parse_args()
    expected_text = "\n".join(lr_rows()) + "\n"
    if not args.spec.is_file() or args.spec.read_text() != expected_text:
        raise AssertionError(f"{args.spec} is missing or differs from the generator")
    rows = [parse(raw) for raw in lr_rows()]
    expected_count = len(TUNING_SEEDS) * sum(len(values) for values in METHOD_LRS.values())
    if len(rows) != expected_count or len({row["run_name"] for row in rows}) != expected_count:
        raise AssertionError("LR matrix cardinality or run-name uniqueness changed")
    if Counter(row["seed"] for row in rows) != Counter({seed: 12 for seed in TUNING_SEEDS}):
        raise AssertionError("seed allocation is not balanced")
    for arch, rates in METHOD_LRS.items():
        for seed in TUNING_SEEDS:
            observed = {
                float(flag_value(row["argv"], "--peak_lr"))
                for row in rows
                if row["arch"] == arch and row["seed"] == seed
            }
            if observed != set(rates):
                raise AssertionError(f"LR grid changed for {arch}, seed {seed}")
    for row in rows:
        if row["dataset"] != "tinystories" or row["config"] != CONFIG:
            raise AssertionError("dataset or model config changed")
        observed_flags = {token for token in row["argv"] if token.startswith("--")}
        allowed_flags = {
            "baseline": {"--batch_size", "--max_tokens", "--peak_lr"},
            "yurii_lt": {"--no_v0_init", "--batch_size", "--max_tokens", "--peak_lr", "--scalar_lr_mult"},
            "causal_symp_pe": {"--no_v0_init", "--presymp_mlp_use_attn_vel", "--batch_size", "--max_tokens", "--peak_lr", "--scalar_lr_mult"},
        }[row["arch"]]
        if observed_flags != allowed_flags:
            raise AssertionError(f"unexpected flags for {row['arch']}: {observed_flags}")
        if int(flag_value(row["argv"], "--max_tokens")) != TOKENS_200M:
            raise AssertionError("token cap changed")
        if int(flag_value(row["argv"], "--batch_size")) != 8:
            raise AssertionError("microbatch size changed")
        peak = float(flag_value(row["argv"], "--peak_lr"))
        if row["arch"] != "baseline":
            scalar = float(flag_value(row["argv"], "--scalar_lr_mult"))
            if not math.isclose(peak * scalar, SCALAR_PEAK_LR, rel_tol=0, abs_tol=1e-12):
                raise AssertionError("learned-scalar peak LR is not fixed")
        if row["arch"] == "causal_symp_pe" and "--presymp_mlp_use_attn_vel" not in row["argv"]:
            raise AssertionError("headline PE architecture flags changed")
    if (TOKENS_200M // 262_144) * 262_144 != EXPECTED_ACTUAL_TOKENS:
        raise AssertionError("whole-step token arithmetic changed")
    print(
        "PASS: 24 equal-budget LR runs, three headline methods, two fresh seeds, "
        "fixed scalar peak LR, and 199,753,728 actual tokens"
    )


if __name__ == "__main__":
    main()
