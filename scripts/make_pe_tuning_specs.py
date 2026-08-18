#!/usr/bin/env python3
"""Generate the predeclared causal-PE tuning and directional arrays."""

import argparse
from pathlib import Path


CONFIG = "configs/core_50m.json"
TUNING_SEEDS = (1337, 20272, 20273)
CONFIRMATION_SEED = 20272
PE_FLAGS = "--no_v0_init --presymp_mlp_use_attn_vel --batch_size 8"
TOKENS_350M = 350_000_000
TOKENS_1B = 1_000_000_000


def slug(value):
    return f"{value:g}".replace(".", "p").replace("-", "m")


def row(dataset, arch, seed, run_name, flags):
    return "\t".join((dataset, arch, CONFIG, str(seed), run_name, flags))


def pe_row(seed, run_name, extra="", max_tokens=TOKENS_350M):
    flags = " ".join(
        part for part in (PE_FLAGS, f"--max_tokens {max_tokens}", extra) if part
    )
    return row("tinystories", "causal_symp_pe", seed, run_name, flags)


def dynamics_rows():
    rows = []
    for seed in TUNING_SEEDS:
        rows.append(pe_row(seed, f"pe350_anchor_s{seed}"))
        for value in (0.03, 0.2, 0.3):
            rows.append(
                pe_row(
                    seed,
                    f"pe350_h{slug(value)}_s{seed}",
                    f"--presymp_h {value:g}",
                )
            )
        for value in (1.0, 2.0, 4.0):
            rows.append(
                pe_row(
                    seed,
                    f"pe350_clog{slug(value)}_s{seed}",
                    f"--eta_log_init {value:g}",
                )
            )
        rows.extend(
            (
                pe_row(
                    seed,
                    f"pe350_lnp_none_s{seed}",
                    "--presymp_lnp none",
                ),
                pe_row(
                    seed,
                    f"pe350_h_fixed_s{seed}",
                    "--learn_h 0",
                ),
                pe_row(
                    seed,
                    f"pe350_eta_fixed_s{seed}",
                    "--no-eta_learnable --eta_log_coef 3 --eta_lin_coef 0.0001",
                ),
            )
        )
    return rows


def optimizer_rows():
    rows = []
    for seed in TUNING_SEEDS:
        for value in (1.0, 10.0, 20.0):
            rows.append(
                pe_row(
                    seed,
                    f"pe350_scalar{slug(value)}_s{seed}",
                    f"--scalar_lr_mult {value:g}",
                )
            )
        for value in (0.005, 0.01, 0.04):
            rows.append(
                pe_row(
                    seed,
                    f"pe350_muon{slug(value)}_s{seed}",
                    f"--muon_lr {value:g}",
                )
            )
        # Change the AdamW-group peak LR while keeping the learned-scalar
        # peak LR fixed at 0.003.
        for peak_lr, scalar_mult in (
            (0.0003, 10.0),
            (0.001, 3.0),
            (0.0015, 2.0),
        ):
            rows.append(
                pe_row(
                    seed,
                    f"pe350_adam{slug(peak_lr)}_s{seed}",
                    f"--peak_lr {peak_lr:g} --scalar_lr_mult {scalar_mult:g}",
                )
            )
    return rows


def directional_rows():
    seed = CONFIRMATION_SEED
    common = f"--batch_size 8 --max_tokens {TOKENS_1B}"
    return [
        row(
            "tinystories",
            "baseline",
            seed,
            f"pe1b_baseline_s{seed}",
            common,
        ),
        row(
            "tinystories",
            "yurii_lt",
            seed,
            f"pe1b_yurii_s{seed}",
            f"--no_v0_init {common}",
        ),
        pe_row(seed, f"pe1b_anchor_s{seed}", max_tokens=TOKENS_1B),
        pe_row(
            seed,
            f"pe1b_h0p2_s{seed}",
            "--presymp_h 0.2",
            max_tokens=TOKENS_1B,
        ),
        pe_row(
            seed,
            f"pe1b_clog2_s{seed}",
            "--eta_log_init 2",
            max_tokens=TOKENS_1B,
        ),
        pe_row(
            seed,
            f"pe1b_scalar10_s{seed}",
            "--scalar_lr_mult 10",
            max_tokens=TOKENS_1B,
        ),
    ]


def write_rows(path, rows):
    path.write_text("\n".join(rows) + "\n")
    print(f"wrote {path} ({len(rows)} runs)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, default=Path("jobs"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.out_dir / "pe_tune_dynamics_350m.tsv", dynamics_rows())
    write_rows(args.out_dir / "pe_tune_optimizer_350m.tsv", optimizer_rows())
    write_rows(args.out_dir / "pe_directional_1b.tsv", directional_rows())


if __name__ == "__main__":
    main()
