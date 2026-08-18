#!/usr/bin/env python3
"""Verify PE tuning row counts, isolation, and output uniqueness."""

import shlex
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_pe_tuning_specs import (
    CONFIRMATION_SEED,
    PE_FLAGS,
    TOKENS_1B,
    TOKENS_350M,
    TUNING_SEEDS,
    directional_rows,
    dynamics_rows,
    optimizer_rows,
)


def parse(row):
    fields = row.split("\t")
    if len(fields) != 6:
        raise AssertionError(f"expected six TSV fields, got {len(fields)}: {row}")
    dataset, arch, config, seed, run_name, flags = fields
    return {
        "dataset": dataset,
        "arch": arch,
        "config": config,
        "seed": int(seed),
        "run_name": run_name,
        "argv": shlex.split(flags),
    }


def flag_value(argv, flag):
    index = argv.index(flag)
    return argv[index + 1]


def extra_flags(argv):
    """Remove the invariant causal-PE and budget arguments."""
    common = shlex.split(PE_FLAGS)
    result = list(argv)
    for token in common:
        result.remove(token)
    index = result.index("--max_tokens")
    del result[index : index + 2]
    return result


def main():
    groups = {
        "dynamics": [parse(value) for value in dynamics_rows()],
        "optimizer": [parse(value) for value in optimizer_rows()],
        "directional": [parse(value) for value in directional_rows()],
    }
    expected_counts = {"dynamics": 30, "optimizer": 27, "directional": 6}
    for name, rows in groups.items():
        if len(rows) != expected_counts[name]:
            raise AssertionError(f"{name}: expected {expected_counts[name]} rows")
        keys = {(row["arch"], row["run_name"]) for row in rows}
        if len(keys) != len(rows):
            raise AssertionError(f"{name}: output directory collision")

    for name in ("dynamics", "optimizer"):
        rows = groups[name]
        if {row["seed"] for row in rows} != set(TUNING_SEEDS):
            raise AssertionError(f"{name}: wrong tuning seeds")
        counts = Counter(row["seed"] for row in rows)
        per_seed = 10 if name == "dynamics" else 9
        if set(counts.values()) != {per_seed}:
            raise AssertionError(f"{name}: unbalanced seed allocation")
        for row in rows:
            if row["arch"] != "causal_symp_pe":
                raise AssertionError(f"{name}: non-PE architecture")
            if int(flag_value(row["argv"], "--max_tokens")) != TOKENS_350M:
                raise AssertionError(f"{name}: wrong token cap")
            if flag_value(row["argv"], "--batch_size") != "8":
                raise AssertionError(f"{name}: wrong microbatch")

    anchors = [
        row for row in groups["dynamics"] if row["run_name"].startswith("pe350_anchor_")
    ]
    if len(anchors) != len(TUNING_SEEDS):
        raise AssertionError("expected one dynamics anchor per tuning seed")
    if any(extra_flags(row["argv"]) for row in anchors):
        raise AssertionError("anchor unexpectedly overrides a factor")

    dynamics_allowed = {
        (),
        ("--presymp_h", "0.03"),
        ("--presymp_h", "0.2"),
        ("--presymp_h", "0.3"),
        ("--eta_log_init", "1"),
        ("--eta_log_init", "2"),
        ("--eta_log_init", "4"),
        ("--presymp_lnp", "none"),
        ("--learn_h", "0"),
        (
            "--no-eta_learnable",
            "--eta_log_coef",
            "3",
            "--eta_lin_coef",
            "0.0001",
        ),
    }
    for row in groups["dynamics"]:
        if tuple(extra_flags(row["argv"])) not in dynamics_allowed:
            raise AssertionError(f"unexpected dynamics override: {row['run_name']}")

    optimizer_allowed_flags = {
        "--scalar_lr_mult",
        "--muon_lr",
        "--peak_lr",
    }
    for row in groups["optimizer"]:
        extras = extra_flags(row["argv"])
        if any(
            token.startswith("--") and token not in optimizer_allowed_flags
            for token in extras
        ):
            raise AssertionError(f"unexpected optimizer override: {row['run_name']}")

    adam_rows = [
        row for row in groups["optimizer"] if "_adam" in row["run_name"]
    ]
    for row in adam_rows:
        peak = float(flag_value(row["argv"], "--peak_lr"))
        scalar = float(flag_value(row["argv"], "--scalar_lr_mult"))
        if abs(peak * scalar - 0.003) > 1e-12:
            raise AssertionError("AdamW screen failed to hold scalar peak LR fixed")

    directional = groups["directional"]
    if {row["seed"] for row in directional} != {CONFIRMATION_SEED}:
        raise AssertionError("directional seed mismatch")
    if [row["arch"] for row in directional[:2]] != ["baseline", "yurii_lt"]:
        raise AssertionError("directional controls missing or reordered")
    if [row["run_name"].rsplit("_s", 1)[0] for row in directional] != [
        "pe1b_baseline",
        "pe1b_yurii",
        "pe1b_anchor",
        "pe1b_h0p2",
        "pe1b_clog2",
        "pe1b_scalar10",
    ]:
        raise AssertionError("directional variants changed")
    for row in directional:
        if int(flag_value(row["argv"], "--max_tokens")) != TOKENS_1B:
            raise AssertionError("directional token cap mismatch")

    steps_350 = TOKENS_350M // 262_144
    steps_1b = TOKENS_1B // 262_144
    if steps_350 * 262_144 != 349_962_240:
        raise AssertionError("350M whole-step accounting changed")
    if steps_1b * 262_144 != 999_817_216:
        raise AssertionError("1B whole-step accounting changed")

    print(
        "PASS: PE campaign has 30/27/6 unique runs, balanced seeds, "
        "isolated factors, fixed scalar LR in the AdamW screen, and exact "
        "whole-step token accounting."
    )


if __name__ == "__main__":
    main()
