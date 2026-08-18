#!/usr/bin/env python3
"""Verify cardinality, locks, and one-factor isolation for a damping stage."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

from analyze_pe_tuning import parse_spec
from make_pe_damping_200m_specs import (
    ACTUAL_TOKENS_200M,
    CONFIG,
    INCUMBENTS,
    STAGES,
    STAGE_SEEDS,
    semantic_setting,
    stage_rows,
    stage_values,
    read_stage_lock,
)


def semantic_args(item: dict) -> tuple[bool, float, float]:
    args = item["expected_args"]
    learnable = bool(args["eta_learnable"])
    if learnable:
        return learnable, float(args["eta_log_init"]), float(args["eta_lin_init"])
    return learnable, float(args["eta_log_coef"]), float(args["eta_lin_coef"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--lock_file", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    lock = read_stage_lock(args.lock_file, args.stage)
    rows = stage_rows(args.stage, lock)
    expected_text = "\n".join(rows) + "\n"
    if not args.spec.is_file() or args.spec.read_text() != expected_text:
        raise AssertionError("damping specification differs from its locked generator")
    parsed = [parse_spec(raw) for raw in rows]
    values = stage_values(args.stage)
    expected_count = len(STAGE_SEEDS[args.stage]) * len(values)
    if len(parsed) != expected_count or len({item["run_name"] for item in parsed}) != expected_count:
        raise AssertionError("damping matrix cardinality or run-name uniqueness changed")
    if Counter(item["seed"] for item in parsed) != Counter(
        {seed: len(values) for seed in STAGE_SEEDS[args.stage]}
    ):
        raise AssertionError("damping seed allocation changed")
    peak = float(lock["peak_lr"])
    scalar = float(lock["scalar_lr_mult"])
    h_init = float(lock["presymp_h"])
    observed_settings = set()
    for raw, item in zip(rows, parsed):
        if item["dataset"] != "tinystories" or item["arch"] != "causal_symp_pe":
            raise AssertionError("dataset or architecture changed")
        if item["config"] != CONFIG or item["expected_tokens"] != ACTUAL_TOKENS_200M:
            raise AssertionError("model config or exact token budget changed")
        expected_args = item["expected_args"]
        for key, expected in (
            ("batch_size", 8),
            ("peak_lr", peak),
            ("scalar_lr_mult", scalar),
            ("presymp_h", h_init),
        ):
            if not math.isclose(float(expected_args[key]), expected, rel_tol=0, abs_tol=1e-12):
                raise AssertionError(f"locked PE field changed: {key}")
        if expected_args.get("eta_mode") != "loglin":
            raise AssertionError("damping schedule family changed")
        observed_settings.add(semantic_args(item))
    expected_settings = {semantic_setting(args.stage, value, lock) for value in values}
    if observed_settings != expected_settings:
        raise AssertionError("the intended one-factor damping settings changed")
    incumbent = semantic_setting(args.stage, INCUMBENTS[args.stage], lock)
    if incumbent not in observed_settings:
        raise AssertionError("paired incumbent is missing")
    print(
        f"PASS: {args.stage} has {expected_count} unique paired runs, exact tokens, "
        "locked PE settings, and one damping factor"
    )


if __name__ == "__main__":
    main()
