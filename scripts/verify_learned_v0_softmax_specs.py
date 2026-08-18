#!/usr/bin/env python3
"""Verify E027 learned-v0 softmax comparison specifications."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_learned_v0_softmax_specs import learned_v0_rows

ARCHES = ("baseline", "yurii_lt", "causal_symp_pe")
TOKENS_PER_STEP = 262_144


def parse(line: str) -> dict:
    dataset, arch, config, seed, run_name, flags = line.split("\t")
    argv = shlex.split(flags)
    values: dict[str, str | bool] = {}
    index = 0
    while index < len(argv):
        token = argv[index]
        if token.startswith("--"):
            if index + 1 < len(argv) and not argv[index + 1].startswith("--"):
                values[token] = argv[index + 1]
                index += 2
            else:
                values[token] = True
                index += 1
        else:
            raise AssertionError(f"unexpected argument: {token}")
    return {"dataset": dataset, "arch": arch, "config": config, "seed": int(seed), "run_name": run_name, "args": values}


def verify(rows: list[dict], *, tokens: int, expected_seeds: int) -> None:
    assert len(rows) == expected_seeds * len(ARCHES)
    assert len({row["run_name"] for row in rows}) == len(rows)
    seeds = sorted({row["seed"] for row in rows})
    assert len(seeds) == expected_seeds
    actual_tokens = (tokens // TOKENS_PER_STEP) * TOKENS_PER_STEP
    assert actual_tokens > 0
    for seed in seeds:
        group = [row for row in rows if row["seed"] == seed]
        assert tuple(row["arch"] for row in group) == ARCHES
        for row in group:
            args = row["args"]
            assert row["dataset"] == "tinystories"
            assert row["config"] == "configs/core_50m.json"
            assert int(args["--max_tokens"]) == tokens
            assert int(args["--batch_size"]) == 8
            assert "--no_v0_init" not in args
            if row["arch"] == "baseline":
                assert "--learned_v0_init" not in args
            else:
                assert args.get("--learned_v0_init") is True
            if row["arch"] == "causal_symp_pe":
                assert args.get("--presymp_mlp_mode") == "attn_vel"
    train_source = Path("train.py").read_text()
    assert '"--learned_v0_init"' in train_source
    assert "use_v0_init=(not args.no_v0_init)" in train_source


def synthetic_locks() -> tuple[dict, dict]:
    lr = {
        "selection_complete": True,
        "methods": {
            "baseline": {"peak_lr": 0.0015, "scalar_lr_mult": 1.0},
            "yurii_lt": {"peak_lr": 0.0015, "scalar_lr_mult": 2.0},
            "causal_symp_pe": {"peak_lr": 0.0025, "scalar_lr_mult": 1.2},
        },
    }
    pe = {
        "selection_complete": True,
        "presymp_h": 0.1,
        "damping_learnable": True,
        "eta_log": 3.0,
        "eta_lin": 0.0001,
    }
    return lr, pe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", nargs="?", type=Path)
    parser.add_argument("--tokens", type=int, default=20_000_000)
    parser.add_argument("--expected_seeds", type=int, default=2)
    args = parser.parse_args()
    if args.spec is None:
        lr, pe = synthetic_locks()
        seeds = tuple(range(20320, 20320 + args.expected_seeds))
        lines = learned_v0_rows(lr, pe, tokens=args.tokens, seeds=seeds, tag="verify")
    else:
        lines = [line for line in args.spec.read_text().splitlines() if line.strip()]
    rows = [parse(line) for line in lines]
    verify(rows, tokens=args.tokens, expected_seeds=args.expected_seeds)
    actual = (args.tokens // TOKENS_PER_STEP) * TOKENS_PER_STEP
    print(f"PASS: {len(rows)} rows, {args.expected_seeds} paired seeds, actual_tokens={actual}")


if __name__ == "__main__":
    main()
