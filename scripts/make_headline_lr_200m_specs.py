#!/usr/bin/env python3
"""Generate the equal-budget 200M LR-selection matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

from make_pe_tuning_specs import PE_FLAGS, row, slug


CONFIG = "configs/core_50m.json"
TUNING_SEEDS = (20276, 20277)
TOKENS_200M = 200_000_000
SCALAR_PEAK_LR = 0.003
METHOD_LRS = {
    "baseline": (0.0003, 0.0006, 0.0010, 0.0015),
    "yurii_lt": (0.0003, 0.0006, 0.0010, 0.0015),
    "causal_symp_pe": (0.0015, 0.0020, 0.0025, 0.0030),
}
METHOD_TAGS = {
    "baseline": "baseline",
    "yurii_lt": "yurii",
    "causal_symp_pe": "pe",
}


def scalar_multiplier(peak_lr: float) -> float:
    return SCALAR_PEAK_LR / peak_lr


def format_number(value: float) -> str:
    return f"{value:.12g}"


def method_flags(arch: str, peak_lr: float) -> str:
    common = f"--batch_size 8 --max_tokens {TOKENS_200M} --peak_lr {format_number(peak_lr)}"
    if arch == "baseline":
        return common
    scalar = f"--scalar_lr_mult {format_number(scalar_multiplier(peak_lr))}"
    if arch == "yurii_lt":
        return f"--no_v0_init {common} {scalar}"
    if arch == "causal_symp_pe":
        return f"{PE_FLAGS} --max_tokens {TOKENS_200M} --peak_lr {format_number(peak_lr)} {scalar}"
    raise ValueError(f"unsupported architecture: {arch}")


def lr_rows() -> list[str]:
    rows = []
    for seed in TUNING_SEEDS:
        for arch, rates in METHOD_LRS.items():
            for peak_lr in rates:
                run_name = f"hlr200_{METHOD_TAGS[arch]}_lr{slug(peak_lr)}_s{seed}"
                rows.append(
                    row(
                        "tinystories",
                        arch,
                        seed,
                        run_name,
                        method_flags(arch, peak_lr),
                    )
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_path", type=Path, default=Path("jobs/headline_lr_200m.tsv"))
    args = parser.parse_args()
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = lr_rows()
    args.out_path.write_text("\n".join(rows) + "\n")
    print(f"wrote {args.out_path} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
