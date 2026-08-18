#!/usr/bin/env python3
"""Generate the single-seed 100M-token PE backfill screen."""

import argparse
from pathlib import Path

from make_pe_tuning_specs import PE_FLAGS, row, slug


SCREEN_SEED = 20272
TOKENS_100M = 100_000_000


def pe_row(run_name, extra=""):
    flags = " ".join(
        part
        for part in (PE_FLAGS, f"--max_tokens {TOKENS_100M}", extra)
        if part
    )
    return row("tinystories", "causal_symp_pe", SCREEN_SEED, run_name, flags)


def dynamics_rows():
    seed = SCREEN_SEED
    rows = [pe_row(f"pe100_anchor_s{seed}")]
    for value in (0.03, 0.2, 0.3):
        rows.append(
            pe_row(
                f"pe100_h{slug(value)}_s{seed}",
                f"--presymp_h {value:g}",
            )
        )
    for value in (1.0, 2.0, 4.0):
        rows.append(
            pe_row(
                f"pe100_clog{slug(value)}_s{seed}",
                f"--eta_log_init {value:g}",
            )
        )
    rows.extend(
        (
            pe_row(f"pe100_lnp_none_s{seed}", "--presymp_lnp none"),
            pe_row(f"pe100_h_fixed_s{seed}", "--learn_h 0"),
            pe_row(
                f"pe100_eta_fixed_s{seed}",
                "--no-eta_learnable --eta_log_coef 3 --eta_lin_coef 0.0001",
            ),
        )
    )
    return rows


def optimizer_rows():
    seed = SCREEN_SEED
    rows = []
    for value in (1.0, 10.0, 20.0):
        rows.append(
            pe_row(
                f"pe100_scalar{slug(value)}_s{seed}",
                f"--scalar_lr_mult {value:g}",
            )
        )
    for value in (0.005, 0.01, 0.04):
        rows.append(
            pe_row(
                f"pe100_muon{slug(value)}_s{seed}",
                f"--muon_lr {value:g}",
            )
        )
    # Hold the learned-scalar peak LR at 0.003 while changing the AdamW
    # groups' peak LR.
    for peak_lr, scalar_mult in (
        (0.0003, 10.0),
        (0.001, 3.0),
        (0.0015, 2.0),
    ):
        rows.append(
            pe_row(
                f"pe100_adam{slug(peak_lr)}_s{seed}",
                f"--peak_lr {peak_lr:g} --scalar_lr_mult {scalar_mult:g}",
            )
        )
    return rows


def control_rows():
    seed = SCREEN_SEED
    common = f"--batch_size 8 --max_tokens {TOKENS_100M}"
    return [
        row(
            "tinystories",
            "baseline",
            seed,
            f"pe100_baseline_s{seed}",
            common,
        ),
        row(
            "tinystories",
            "yurii_lt",
            seed,
            f"pe100_yurii_s{seed}",
            f"--no_v0_init {common}",
        ),
    ]


def write_rows(path, rows):
    path.write_text("\n".join(rows) + "\n")
    print(f"wrote {path} ({len(rows)} runs)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=Path("jobs"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.out_dir / "pe_screen_dynamics_100m.tsv", dynamics_rows())
    write_rows(args.out_dir / "pe_screen_optimizer_100m.tsv", optimizer_rows())
    write_rows(args.out_dir / "pe_screen_controls_100m.tsv", control_rows())


if __name__ == "__main__":
    main()
