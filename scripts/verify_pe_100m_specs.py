#!/usr/bin/env python3
"""Verify the deterministic, one-factor 100M PE screen."""

import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_pe_100m_specs import (
    SCREEN_SEED,
    TOKENS_100M,
    control_rows,
    dynamics_rows,
    optimizer_rows,
)
from make_pe_tuning_specs import PE_FLAGS


EXPECTED_ACTUAL_TOKENS = 99_876_864


def parse(row):
    fields = row.split("\t")
    if len(fields) != 6:
        raise AssertionError(f"expected six TSV fields: {row}")
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


def pe_extras(argv):
    result = list(argv)
    for token in shlex.split(PE_FLAGS):
        result.remove(token)
    index = result.index("--max_tokens")
    del result[index : index + 2]
    return tuple(result)


def main():
    groups = {
        "dynamics": [parse(row) for row in dynamics_rows()],
        "optimizer": [parse(row) for row in optimizer_rows()],
        "controls": [parse(row) for row in control_rows()],
    }
    if {name: len(rows) for name, rows in groups.items()} != {
        "dynamics": 10,
        "optimizer": 9,
        "controls": 2,
    }:
        raise AssertionError("100M row counts changed")
    all_rows = sum(groups.values(), [])
    if len({row["run_name"] for row in all_rows}) != 21:
        raise AssertionError("run-name collision")
    if {row["seed"] for row in all_rows} != {SCREEN_SEED}:
        raise AssertionError("screen seed changed")
    for row in all_rows:
        if row["dataset"] != "tinystories":
            raise AssertionError("dataset changed")
        if int(flag_value(row["argv"], "--max_tokens")) != TOKENS_100M:
            raise AssertionError("token cap changed")
        if flag_value(row["argv"], "--batch_size") != "8":
            raise AssertionError("microbatch changed")

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
    if {pe_extras(row["argv"]) for row in groups["dynamics"]} != dynamics_allowed:
        raise AssertionError("dynamics factors changed")

    allowed_optimizer_flags = {"--scalar_lr_mult", "--muon_lr", "--peak_lr"}
    for row in groups["optimizer"]:
        extras = pe_extras(row["argv"])
        if any(
            token.startswith("--") and token not in allowed_optimizer_flags
            for token in extras
        ):
            raise AssertionError(f"unexpected optimizer factor: {row['run_name']}")
        if "_adam" in row["run_name"]:
            peak = float(flag_value(row["argv"], "--peak_lr"))
            scalar = float(flag_value(row["argv"], "--scalar_lr_mult"))
            if abs(peak * scalar - 0.003) > 1e-12:
                raise AssertionError("scalar peak LR is not fixed")

    if [row["arch"] for row in groups["controls"]] != ["baseline", "yurii_lt"]:
        raise AssertionError("control architectures changed")
    if (TOKENS_100M // 262_144) * 262_144 != EXPECTED_ACTUAL_TOKENS:
        raise AssertionError("whole-step token arithmetic changed")
    print(
        "PASS: 10/9/2 unique 100M rows, one seed, isolated factors, "
        "and 99,876,864 exact tokens"
    )


if __name__ == "__main__":
    main()
