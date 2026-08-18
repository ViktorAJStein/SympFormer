#!/usr/bin/env python3
"""Generate the four-way 350M-token TinyStories debugging comparison."""

import argparse
from pathlib import Path


TOKEN_CAP = 350_000_000
TOKENS_PER_STEP = 262_144
SEED = 20271
CONFIG = "configs/core_50m.json"
METHODS = {
    "baseline": "",
    "yurii_lt": "--no_v0_init",
    "causal_symp_fe": "--no_v0_init --presymp_mlp_use_attn_vel",
    "causal_symp_pe": "--no_v0_init --presymp_mlp_use_attn_vel",
}


def rows():
    common = f"--batch_size 8 --max_tokens {TOKEN_CAP}"
    for arch, method_flags in METHODS.items():
        flags = " ".join(part for part in (method_flags, common) if part)
        yield "\t".join(
            (
                "tinystories",
                arch,
                CONFIG,
                str(SEED),
                f"short350m_s{SEED}",
                flags,
            )
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("jobs/short_350m.tsv"))
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(rows()) + "\n")
    steps = TOKEN_CAP // TOKENS_PER_STEP
    print(
        f"wrote {args.output} (4 runs, {steps} steps, "
        f"{steps * TOKENS_PER_STEP} actual tokens each)"
    )


if __name__ == "__main__":
    main()
