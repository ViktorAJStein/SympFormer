#!/usr/bin/env python3
"""Generate a strictly causal E024 linear-attention screen."""

from __future__ import annotations

import argparse
from pathlib import Path

METHODS = (
    "lin_baseline",
    "lin_yurii",
    "lin_euler",
    "lin_presymp",
    "lin_exp_euler",
    "lin_ab2",
    "lin_etd_ab2",
    "lin_reduced_exp_mid",
    "lin_reduced_ab2",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", default="31020,31021")
    parser.add_argument("--tokens", type=int, default=20_000_000)
    parser.add_argument("--amp_dtype", choices=("float16", "bfloat16"), default="float16")
    args = parser.parse_args()
    seeds = [int(item) for item in args.seeds.split(",") if item]
    flags = (
        f"--no_v0_init --batch_size 8 --max_tokens {args.tokens} "
        "--presymp_h 0.1 --eta_mode log --no-eta_learnable --eta_log_coef 3 "
        f"--sample_interval 0 --amp_dtype {args.amp_dtype}"
    )
    rows = []
    for seed in seeds:
        for method in METHODS:
            run_name = f"linv3_{method.removeprefix('lin_')}_s{seed}"
            rows.append(
                "\t".join(("tinystories", method, "configs/linear_v3_small.json", str(seed), run_name, flags))
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(rows) + "\n")
    print(f"wrote {args.out} ({len(rows)} rows, {len(seeds)} seeds, {len(METHODS)} methods)")


if __name__ == "__main__":
    main()
