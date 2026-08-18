#!/usr/bin/env python3
"""Generate architecture-choice pruning matrices for tuned causal PE."""

from __future__ import annotations

import argparse
from pathlib import Path

from make_pe_tuning_specs import CONFIG, row

TOKENS_100M = 100_000_000
DEFAULT_SEEDS = (20300, 20301)
BASE_PE = (
    f"--no_v0_init --batch_size 8 --max_tokens {TOKENS_100M} --peak_lr 0.0025 --scalar_lr_mult 1.2 "
    "--presymp_h 0.1 --eta_mode loglin --eta_learnable --eta_log_init 3 --eta_lin_init 0.0001"
)

MLP_VARIANTS = {
    "attnvel": "--presymp_mlp_mode attn_vel",
    "stdmlp": "--presymp_mlp_mode separate_vel",
    "pvel": "--presymp_mlp_mode p_vel",
}

# Causal PE has one momentum update per block, so `end` and
# `each_substep` currently execute the same normalization code. Only the
# meaningful normalized-vs-unnormalized comparison is generated.
LNP_VARIANTS = {
    "lnpend": "--presymp_mlp_mode attn_vel --presymp_lnp end",
    "lnpnone": "--presymp_mlp_mode attn_vel --presymp_lnp none",
}

DISCRETIZATION_VARIANTS = {
    "pe": "causal_symp_pe",
    "exppe": "causal_symp_exp_pe",
    "halfdamp": "causal_symp_halfdamp_pe",
}

AB2_VARIANTS = {
    "pe": "causal_symp_pe",
    "ab2": "causal_symp_ab2",
}

LOOKAHEAD_VARIANTS = {
    "none": "",
    "mu01": "--presymp_lookahead --presymp_lookahead_init 0.1",
    "mu05": "--presymp_lookahead --presymp_lookahead_init 0.5",
    "mu09": "--presymp_lookahead --presymp_lookahead_init 0.9",
}


def rows_for_screen(screen: str, seeds: tuple[int, ...]) -> list[str]:
    if screen == "mlp":
        variants = MLP_VARIANTS
    elif screen == "lnp":
        variants = LNP_VARIANTS
    elif screen == "disc":
        variants = DISCRETIZATION_VARIANTS
    elif screen == "lookahead":
        variants = LOOKAHEAD_VARIANTS
    elif screen == "ab2":
        variants = AB2_VARIANTS
    else:
        raise ValueError(f"unknown screen: {screen}")
    rows: list[str] = []
    for seed in seeds:
        for name, extra in variants.items():
            if screen in {"disc", "ab2"}:
                arch = extra
                flags = " ".join((BASE_PE, "--presymp_mlp_mode attn_vel --presymp_lnp end"))
            else:
                arch = "causal_symp_pe"
                fixed = "--presymp_mlp_mode attn_vel --presymp_lnp end" if screen == "lookahead" else ""
                flags = " ".join(part for part in (BASE_PE, fixed, extra) if part)
            rows.append(row("tinystories", arch, seed, f"arch100_{screen}_{name}_s{seed}", flags))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", choices=["mlp", "lnp", "disc", "lookahead", "ab2"], required=True)
    parser.add_argument("--out_path", type=Path, required=True)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    args = parser.parse_args()
    seeds = tuple(int(x) for x in args.seeds.split(",") if x)
    rows = rows_for_screen(args.screen, seeds)
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text("\n".join(rows) + "\n")
    print(f"wrote {args.out_path} ({len(rows)} runs, screen={args.screen}, seeds={len(seeds)})")


if __name__ == "__main__":
    main()
