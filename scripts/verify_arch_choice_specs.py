#!/usr/bin/env python3
"""Verify architecture-choice pruning matrices."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from analyze_pe_tuning import parse_spec

INCUMBENT = {"mlp": "arch100_mlp_attnvel", "lnp": "arch100_lnp_lnpend", "disc": "arch100_disc_pe", "lookahead": "arch100_lookahead_none", "ab2": "arch100_ab2_pe"}
EXPECTED_VARIANTS = {
    "mlp": {"arch100_mlp_attnvel", "arch100_mlp_stdmlp", "arch100_mlp_pvel"},
    "lnp": {"arch100_lnp_lnpend", "arch100_lnp_lnpnone"},
    "disc": {"arch100_disc_pe", "arch100_disc_exppe", "arch100_disc_halfdamp"},
    "lookahead": {"arch100_lookahead_none", "arch100_lookahead_mu01", "arch100_lookahead_mu05", "arch100_lookahead_mu09"},
    "ab2": {"arch100_ab2_pe", "arch100_ab2_ab2"},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    parser.add_argument("--screen", choices=["mlp", "lnp", "disc", "lookahead", "ab2"], required=True)
    parser.add_argument("--expected_seeds", type=int, default=2)
    args = parser.parse_args()
    rows = [parse_spec(line) for line in args.spec.read_text().splitlines() if line.strip()]
    if len({row["run_name"] for row in rows}) != len(rows):
        raise SystemExit("duplicate run_name")
    by_seed: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_seed[int(row["seed"])].append(row)
        if args.screen == "disc":
            allowed_arches = {"causal_symp_pe", "causal_symp_exp_pe", "causal_symp_halfdamp_pe"}
        elif args.screen == "ab2":
            allowed_arches = {"causal_symp_pe", "causal_symp_ab2"}
        else:
            allowed_arches = {"causal_symp_pe"}
        if row["dataset"] != "tinystories" or row["arch"] not in allowed_arches or row["config"] != "configs/core_50m.json":
            raise SystemExit(f"wrong dataset/arch/config for {row['run_name']}")
        exp = row["expected_args"]
        for key, value in {
            "max_tokens": 100_000_000,
            "batch_size": 8,
            "peak_lr": 0.0025,
            "scalar_lr_mult": 1.2,
            "presymp_h": 0.1,
            "eta_mode": "loglin",
            "eta_log_init": 3,
            "eta_lin_init": 0.0001,
        }.items():
            if exp.get(key) != value:
                raise SystemExit(f"{row['run_name']} has {key}={exp.get(key)}, expected {value}")
        if exp.get("no_v0_init") is not True or exp.get("eta_learnable") is not True:
            raise SystemExit(f"missing safe PE flags for {row['run_name']}")
    if len(by_seed) != args.expected_seeds:
        raise SystemExit(f"expected {args.expected_seeds} seeds, found {len(by_seed)}")
    expected = EXPECTED_VARIANTS[args.screen]
    for seed, seed_rows in sorted(by_seed.items()):
        variants = {row["candidate"] for row in seed_rows}
        if variants != expected:
            raise SystemExit(f"seed {seed} variants {variants}, expected {expected}")
        by_candidate = {row["candidate"]: row for row in seed_rows}
        if args.screen == "mlp":
            expected_modes = {
                "arch100_mlp_attnvel": "attn_vel",
                "arch100_mlp_stdmlp": "separate_vel",
                "arch100_mlp_pvel": "p_vel",
            }
            for candidate, mode in expected_modes.items():
                if by_candidate[candidate]["expected_args"].get("presymp_mlp_mode") != mode:
                    raise SystemExit(f"{candidate} does not select explicit mode {mode}")
        elif args.screen == "lnp":
            expected_lnp = {
                "arch100_lnp_lnpend": "end",
                "arch100_lnp_lnpnone": "none",
            }
            for candidate, placement in expected_lnp.items():
                exp = by_candidate[candidate]["expected_args"]
                if exp.get("presymp_mlp_mode") != "attn_vel" or exp.get("presymp_lnp") != placement:
                    raise SystemExit(f"{candidate} does not isolate momentum LayerNorm placement {placement}")
        elif args.screen == "disc":
            expected_arch = {
                "arch100_disc_pe": "causal_symp_pe",
                "arch100_disc_exppe": "causal_symp_exp_pe",
                "arch100_disc_halfdamp": "causal_symp_halfdamp_pe",
            }
            for candidate, arch in expected_arch.items():
                row = by_candidate[candidate]
                exp = row["expected_args"]
                if row["arch"] != arch or exp.get("presymp_mlp_mode") != "attn_vel" or exp.get("presymp_lnp") != "end":
                    raise SystemExit(f"{candidate} does not isolate discretization {arch}")
        elif args.screen == "lookahead":
            expected_mu = {
                "arch100_lookahead_none": None,
                "arch100_lookahead_mu01": 0.1,
                "arch100_lookahead_mu05": 0.5,
                "arch100_lookahead_mu09": 0.9,
            }
            for candidate, mu in expected_mu.items():
                exp = by_candidate[candidate]["expected_args"]
                if exp.get("presymp_mlp_mode") != "attn_vel" or exp.get("presymp_lnp") != "end":
                    raise SystemExit(f"{candidate} changes a locked block choice")
                if mu is None:
                    if exp.get("presymp_lookahead") is True:
                        raise SystemExit("incumbent unexpectedly enables lookahead")
                elif exp.get("presymp_lookahead") is not True or exp.get("presymp_lookahead_init") != mu:
                    raise SystemExit(f"{candidate} does not select lookahead init {mu}")
        else:
            expected_arch = {
                "arch100_ab2_pe": "causal_symp_pe",
                "arch100_ab2_ab2": "causal_symp_ab2",
            }
            for candidate, arch in expected_arch.items():
                row = by_candidate[candidate]
                exp = row["expected_args"]
                if row["arch"] != arch or exp.get("presymp_mlp_mode") != "attn_vel" or exp.get("presymp_lnp") != "end":
                    raise SystemExit(f"{candidate} does not isolate AB2 discretization {arch}")
                if exp.get("presymp_lookahead") is True:
                    raise SystemExit(f"{candidate} unexpectedly enables lookahead")
    print(f"PASS: {len(rows)} architecture-choice rows, screen={args.screen}, seeds={len(by_seed)}, incumbent={INCUMBENT[args.screen]}")


if __name__ == "__main__":
    main()
