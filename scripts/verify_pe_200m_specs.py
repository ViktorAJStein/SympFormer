#!/usr/bin/env python3
"""Verify the paired, one-factor 200M PE confirmation matrix."""

import shlex
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_pe_200m_specs import (
    CANDIDATES,
    CONFIRMATION_SEEDS,
    TOKENS_200M,
    confirmation_rows,
)
from make_pe_tuning_specs import PE_FLAGS


EXPECTED_ACTUAL_TOKENS = 199_753_728


def parse(raw):
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


def flag_value(argv, flag):
    index = argv.index(flag)
    return argv[index + 1]


def extras(argv):
    remaining = list(argv)
    for token in shlex.split(PE_FLAGS):
        remaining.remove(token)
    index = remaining.index("--max_tokens")
    del remaining[index : index + 2]
    return tuple(remaining)


def main():
    rows = [parse(raw) for raw in confirmation_rows()]
    if len(rows) != 8 or len({row["run_name"] for row in rows}) != 8:
        raise AssertionError("expected eight unique confirmation runs")
    if {row["seed"] for row in rows} != set(CONFIRMATION_SEEDS):
        raise AssertionError("fresh confirmation seeds changed")
    if Counter(row["seed"] for row in rows) != Counter({20274: 4, 20275: 4}):
        raise AssertionError("seed allocation is not balanced")
    if any(
        row["dataset"] != "tinystories"
        or row["arch"] != "causal_symp_pe"
        or row["config"] != "configs/core_50m.json"
        for row in rows
    ):
        raise AssertionError("model protocol changed")
    if any(int(flag_value(row["argv"], "--max_tokens")) != TOKENS_200M for row in rows):
        raise AssertionError("token cap changed")

    expected_extras = {tuple(shlex.split(extra)) for _, extra in CANDIDATES}
    for seed in CONFIRMATION_SEEDS:
        observed = {extras(row["argv"]) for row in rows if row["seed"] == seed}
        if observed != expected_extras:
            raise AssertionError(f"candidate factors changed for seed {seed}")
    for row in rows:
        if "_adam" in row["run_name"]:
            peak = float(flag_value(row["argv"], "--peak_lr"))
            scalar = float(flag_value(row["argv"], "--scalar_lr_mult"))
            if abs(peak * scalar - 0.003) > 1e-12:
                raise AssertionError("learned-scalar peak LR is not fixed")
    if (TOKENS_200M // 262_144) * 262_144 != EXPECTED_ACTUAL_TOKENS:
        raise AssertionError("whole-step token arithmetic changed")
    print(
        "PASS: eight paired 200M runs, two fresh seeds, isolated optimizer "
        "factors, and 199,753,728 exact tokens"
    )


if __name__ == "__main__":
    main()
