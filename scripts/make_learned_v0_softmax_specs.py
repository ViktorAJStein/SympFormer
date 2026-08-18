#!/usr/bin/env python3
"""Generate the E027 learned-v0 headline softmax comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from make_pe_tuning_specs import row
from make_high_iter_specs import SAMPLE_FLAGS, fmt, pe_damping_flags

DEFAULT_TIER2_SEEDS = (20320, 20321)
DEFAULT_TIER3_SEEDS = (20322, 20323, 20324, 20325, 20326)


def load_lock(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise SystemExit(f"missing lock: {path}") from exc
    if not value.get("selection_complete"):
        raise SystemExit(f"lock is not selection_complete: {path}")
    return value


def baseline_flags(method: dict, tokens: int) -> str:
    return f"--batch_size 8 --max_tokens {tokens} --peak_lr {fmt(method['peak_lr'])} {SAMPLE_FLAGS}"


def yurii_flags(method: dict, tokens: int) -> str:
    return " ".join((
        "--learned_v0_init",
        baseline_flags(method, tokens),
        f"--scalar_lr_mult {fmt(method['scalar_lr_mult'])}",
    ))


def pe_flags(method: dict, pe_lock: dict, tokens: int) -> str:
    return " ".join((
        "--learned_v0_init",
        "--presymp_mlp_mode attn_vel",
        "--batch_size 8",
        f"--max_tokens {tokens}",
        f"--peak_lr {fmt(method['peak_lr'])}",
        f"--scalar_lr_mult {fmt(method['scalar_lr_mult'])}",
        f"--presymp_h {fmt(pe_lock['presymp_h'])}",
        pe_damping_flags(pe_lock),
        SAMPLE_FLAGS,
    ))


def learned_v0_rows(
    lr_lock: dict,
    pe_lock: dict,
    *,
    tokens: int,
    seeds: tuple[int, ...],
    tag: str,
    dataset: str = "tinystories",
) -> list[str]:
    methods = lr_lock["methods"]
    rows: list[str] = []
    for seed in seeds:
        rows.extend((
            row(dataset, "baseline", seed, f"{tag}_baseline_s{seed}", baseline_flags(methods["baseline"], tokens)),
            row(dataset, "yurii_lt", seed, f"{tag}_yurii_learnedv0_s{seed}", yurii_flags(methods["yurii_lt"], tokens)),
            row(dataset, "causal_symp_pe", seed, f"{tag}_pe_learnedv0_s{seed}", pe_flags(methods["causal_symp_pe"], pe_lock, tokens)),
        ))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr_lock", type=Path, required=True)
    parser.add_argument("--pe_lock", type=Path, required=True)
    parser.add_argument("--out_path", type=Path, required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--dataset", default="tinystories")
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    if not seeds:
        raise SystemExit("at least one seed is required")
    rows = learned_v0_rows(
        load_lock(args.lr_lock),
        load_lock(args.pe_lock),
        tokens=args.tokens,
        seeds=seeds,
        tag=args.tag,
        dataset=args.dataset,
    )
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text("\n".join(rows) + "\n")
    print(f"wrote {args.out_path} ({len(rows)} rows, {len(seeds)} paired seeds)")


if __name__ == "__main__":
    main()
