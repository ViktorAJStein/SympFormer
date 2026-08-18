#!/usr/bin/env python3
"""Verify the conditional PE initial-step bracket."""

from __future__ import annotations

import argparse
import math
import shlex
from collections import Counter
from pathlib import Path

from make_pe_h_200m_specs import CONFIG, H_SEEDS, H_VALUES, TOKENS_200M, h_rows, read_lr_lock


def parse(raw: str) -> dict:
    dataset, arch, config, seed, run_name, flags = raw.split("\t")
    return {
        "dataset": dataset,
        "arch": arch,
        "config": config,
        "seed": int(seed),
        "run_name": run_name,
        "argv": shlex.split(flags),
    }


def value(argv: list[str], flag: str) -> float:
    return float(argv[argv.index(flag) + 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock_file", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    lock = read_lr_lock(args.lock_file)
    expected = "\n".join(h_rows(lock)) + "\n"
    if not args.spec.is_file() or args.spec.read_text() != expected:
        raise AssertionError("PE-h specification differs from the locked generator")
    rows = [parse(raw) for raw in h_rows(lock)]
    if len(rows) != 8 or len({row["run_name"] for row in rows}) != 8:
        raise AssertionError("expected eight unique PE-h runs")
    if Counter(row["seed"] for row in rows) != Counter({seed: 4 for seed in H_SEEDS}):
        raise AssertionError("PE-h seed allocation changed")
    locked_peak = float(lock["methods"]["causal_symp_pe"]["peak_lr"])
    locked_scalar = float(lock["methods"]["causal_symp_pe"]["scalar_lr_mult"])
    for seed in H_SEEDS:
        observed = {value(row["argv"], "--presymp_h") for row in rows if row["seed"] == seed}
        if observed != set(H_VALUES):
            raise AssertionError(f"h grid changed for seed {seed}")
    fixed_argv = None
    for row in rows:
        if row["dataset"] != "tinystories" or row["config"] != CONFIG:
            raise AssertionError("dataset or model config changed")
        if row["arch"] != "causal_symp_pe":
            raise AssertionError("PE-h campaign contains a non-PE architecture")
        if int(value(row["argv"], "--max_tokens")) != TOKENS_200M:
            raise AssertionError("token cap changed")
        if int(value(row["argv"], "--batch_size")) != 8:
            raise AssertionError("microbatch size changed")
        if not {"--no_v0_init", "--presymp_mlp_use_attn_vel"} <= set(row["argv"]):
            raise AssertionError("headline PE architecture flags changed")
        if row["argv"].count("--presymp_h") != 1:
            raise AssertionError("initial-step flag is not unique")
        index = row["argv"].index("--presymp_h")
        invariant_argv = row["argv"][:index] + row["argv"][index + 2 :]
        if fixed_argv is None:
            fixed_argv = invariant_argv
        elif invariant_argv != fixed_argv:
            raise AssertionError("a factor other than the initial PE step changed")
        if not math.isclose(value(row["argv"], "--peak_lr"), locked_peak, rel_tol=0, abs_tol=1e-12):
            raise AssertionError("locked AdamW-group LR changed")
        if not math.isclose(value(row["argv"], "--scalar_lr_mult"), locked_scalar, rel_tol=0, abs_tol=1e-12):
            raise AssertionError("locked scalar multiplier changed")
    print("PASS: eight paired PE-h runs, four initial steps, two fresh seeds, and locked optimizer settings")


if __name__ == "__main__":
    main()
