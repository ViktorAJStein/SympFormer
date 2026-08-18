#!/usr/bin/env python3
"""Generate the E025 causal softmax discretization comparison."""
from __future__ import annotations
import argparse
from pathlib import Path

METHODS = (
    "baseline", "yurii_lt", "causal_symp_fe", "causal_symp_pe",
    "causal_symp_exp_pe", "causal_symp_ab2",
    "causal_riem_nag_noconn", "causal_riem_nag",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=20_000_000)
    parser.add_argument("--seeds", default="32020,32021")
    parser.add_argument("--amp_dtype", choices=("float16", "bfloat16"), default="float16")
    args = parser.parse_args()
    flags = (
        f"--no_v0_init --batch_size 8 --max_tokens {args.tokens} "
        "--presymp_h 0.1 --presymp_t0 1 --eta_mode log --no-eta_learnable "
        f"--eta_log_coef 3 --eta_lin_coef 0 --sample_interval 0 --amp_dtype {args.amp_dtype}"
    )
    rows = []
    for seed in (int(x) for x in args.seeds.split(",") if x):
        for method in METHODS:
            rows.append("\t".join((
                "tinystories", method, "configs/linear_v3_small.json", str(seed),
                f"e025_{method}_s{seed}", flags,
            )))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(rows) + "\n")
    print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
