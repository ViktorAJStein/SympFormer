#!/usr/bin/env python3
"""Generate lock-conditioned high-iteration headline comparison matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from make_pe_tuning_specs import CONFIG, PE_FLAGS, row, slug

DEFAULT_SEEDS = (20290, 20291, 20292, 20293, 20294)
DEFAULT_TOKENS = 1_000_000_000
SAMPLE_FLAGS = "--sample_final --sample_max_new_tokens 256 --sample_prefix_tokens 64 --sample_do_sample 0"


def fmt(value: float) -> str:
    return f"{value:.12g}"


def require_complete(lock: dict, path: Path) -> None:
    if not lock.get("selection_complete"):
        raise SystemExit(f"lock is not selection_complete: {path}")


def load_lock(path: Path) -> dict:
    try:
        lock = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise SystemExit(f"missing lock file: {path}") from exc
    require_complete(lock, path)
    return lock


def baseline_flags(method_lock: dict, tokens: int) -> str:
    return f"--batch_size 8 --max_tokens {tokens} --peak_lr {fmt(method_lock['peak_lr'])} {SAMPLE_FLAGS}"


def yurii_flags(method_lock: dict, tokens: int) -> str:
    return " ".join(
        (
            "--no_v0_init",
            baseline_flags(method_lock, tokens),
            f"--scalar_lr_mult {fmt(method_lock['scalar_lr_mult'])}",
        )
    )


def pe_damping_flags(pe_lock: dict) -> str:
    eta_log = fmt(pe_lock["eta_log"])
    eta_lin = fmt(pe_lock["eta_lin"])
    if pe_lock.get("damping_learnable", True):
        return f"--eta_mode loglin --eta_learnable --eta_log_init {eta_log} --eta_lin_init {eta_lin}"
    return f"--eta_mode loglin --no-eta_learnable --eta_log_coef {eta_log} --eta_lin_coef {eta_lin}"


def pe_flags(method_lock: dict, pe_lock: dict, tokens: int) -> str:
    return " ".join(
        (
            PE_FLAGS,
            f"--max_tokens {tokens}",
            f"--peak_lr {fmt(method_lock['peak_lr'])}",
            f"--scalar_lr_mult {fmt(method_lock['scalar_lr_mult'])}",
            f"--presymp_h {fmt(pe_lock['presymp_h'])}",
            pe_damping_flags(pe_lock),
            SAMPLE_FLAGS,
        )
    )


def high_iter_rows(lr_lock: dict, pe_lock: dict, *, dataset: str, tokens: int, seeds: tuple[int, ...], tag: str) -> list[str]:
    methods = lr_lock["methods"]
    rows: list[str] = []
    for seed in seeds:
        rows.append(
            row(
                dataset,
                "baseline",
                seed,
                f"{tag}_baseline_s{seed}",
                baseline_flags(methods["baseline"], tokens),
            )
        )
        rows.append(
            row(
                dataset,
                "yurii_lt",
                seed,
                f"{tag}_yurii_s{seed}",
                yurii_flags(methods["yurii_lt"], tokens),
            )
        )
        rows.append(
            row(
                dataset,
                "causal_symp_pe",
                seed,
                f"{tag}_pe_s{seed}",
                pe_flags(methods["causal_symp_pe"], pe_lock, tokens),
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr_lock", type=Path, required=True, help="locked_lrs.json from E008")
    parser.add_argument("--pe_lock", type=Path, required=True, help="final PE damping/config lock")
    parser.add_argument("--out_path", type=Path, default=Path("jobs/high_iter_1b.tsv"))
    parser.add_argument("--dataset", default="tinystories")
    parser.add_argument("--tokens", type=int, default=DEFAULT_TOKENS)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    seeds = tuple(int(x) for x in args.seeds.split(",") if x)
    tag = args.tag or f"hi{slug(args.tokens / 1_000_000_000)}b"
    lr_lock = load_lock(args.lr_lock)
    pe_lock = load_lock(args.pe_lock)
    rows = high_iter_rows(lr_lock, pe_lock, dataset=args.dataset, tokens=args.tokens, seeds=seeds, tag=tag)
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text("\n".join(rows) + "\n")
    print(f"wrote {args.out_path} ({len(rows)} runs, {len(seeds)} seeds, tokens={args.tokens})")


if __name__ == "__main__":
    main()
