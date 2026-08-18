#!/usr/bin/env python3
"""Generate the paired fresh-seed 200M PE confirmation matrix."""

import argparse
from pathlib import Path

from make_pe_tuning_specs import PE_FLAGS, row


CONFIRMATION_SEEDS = (20274, 20275)
TOKENS_200M = 200_000_000
CANDIDATES = (
    ("anchor", ""),
    ("adam0p001", "--peak_lr 0.001 --scalar_lr_mult 3"),
    ("adam0p0015", "--peak_lr 0.0015 --scalar_lr_mult 2"),
    ("muon0p04", "--muon_lr 0.04"),
)


def confirmation_rows():
    rows = []
    for seed in CONFIRMATION_SEEDS:
        for candidate, extra in CANDIDATES:
            flags = " ".join(
                part
                for part in (PE_FLAGS, f"--max_tokens {TOKENS_200M}", extra)
                if part
            )
            rows.append(
                row(
                    "tinystories",
                    "causal_symp_pe",
                    seed,
                    f"pe200_{candidate}_s{seed}",
                    flags,
                )
            )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=Path("jobs"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "pe_confirm_200m.tsv"
    path.write_text("\n".join(confirmation_rows()) + "\n")
    print(f"wrote {path} ({len(confirmation_rows())} runs)")


if __name__ == "__main__":
    main()
