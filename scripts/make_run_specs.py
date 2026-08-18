#!/usr/bin/env python3
"""Generate predeclared Slurm-array TSVs for pilots, ablations, and core runs."""

import argparse
from pathlib import Path

HEADLINE_METHODS = {
    "baseline": "",
    "yurii_lt": "--no_v0_init",
    "causal_symp_pe": "--no_v0_init --presymp_mlp_use_attn_vel",
}
ALL_METHODS = {**HEADLINE_METHODS, "baseline_capacity": ""}
# After the staged pilots, place any method-specific locked choices here and
# regenerate the TSVs before core submission. Empty strings use the shared
# provisional anchor in configs/core_*.json.
LOCKED_OVERRIDES = {arch: "" for arch in HEADLINE_METHODS}


def row(dataset, arch, config, seed, run_name, extra=""):
    flags = " ".join(x for x in (ALL_METHODS[arch], extra) if x)
    return "\t".join((dataset, arch, config, str(seed), run_name, flags))


def generate_core():
    rows = []
    for seed in range(20271, 20276):
        for arch in HEADLINE_METHODS:
            rows.append(row("tinystories", arch, "configs/core_50m.json", seed, f"ts_50m_s{seed}", f"--batch_size 8 {LOCKED_OVERRIDES[arch]}".strip()))
    for seed in range(20271, 20274):
        for arch in HEADLINE_METHODS:
            rows.append(row("tinystories", arch, "configs/core_124m.json", seed, f"ts_124m_s{seed}", f"--batch_size 2 {LOCKED_OVERRIDES[arch]}".strip()))
        for arch in HEADLINE_METHODS:
            rows.append(row("openwebtext", arch, "configs/core_50m.json", seed, f"owt_50m_s{seed}", f"--batch_size 8 {LOCKED_OVERRIDES[arch]}".strip()))
    return rows


def generate_pilot_lr():
    return [
        row("tinystories", arch, "configs/pilot_tinystories.json", 1337, f"lr{lr:g}", f"--peak_lr {lr:g}")
        for arch in HEADLINE_METHODS
        for lr in (3e-4, 6e-4, 1e-3)
    ]


def generate_symp_stage(flag, values, tag):
    return [
        row("tinystories", "causal_symp_pe", "configs/pilot_tinystories.json", 1337, f"{tag}{value:g}", f"{flag} {value:g}")
        for value in values
    ]


def generate_damping_pilot():
    return [
        row("tinystories", "causal_symp_pe", "configs/pilot_tinystories.json", 1337, "eta_learned"),
        row(
            "tinystories", "causal_symp_pe", "configs/pilot_tinystories.json", 1337,
            "eta_fixed", "--no-eta_learnable --eta_log_coef 3 --eta_lin_coef 0.0001",
        ),
    ]


def generate_noise_pilot():
    return [
        row(
            "tinystories", "causal_symp_pe", "configs/pilot_tinystories.json", 1337,
            f"noise{eta:g}", f"--presymp_noise_eta {eta:g}",
        )
        for eta in (0.0, 1e-5, 1e-4)
    ]


def generate_ablation():
    variants = [
        ("causal_symp_pe", "reference", ""),
        ("causal_symp_pe", "fixed_eta", "--no-eta_learnable --eta_log_coef 3 --eta_lin_coef 0.0001"),
        ("causal_symp_pe", "no_momentum_ln", "--presymp_lnp none"),
        ("causal_symp_pe", "fixed_h", "--learn_h 0"),
        ("causal_symp_pe", "momentum_noise", "--presymp_noise_eta 1e-5"),
        ("baseline", "reference", ""),
        ("baseline_capacity", "capacity_control", ""),
    ]
    return [
        row("tinystories", arch, "configs/core_50m.json", seed, f"abl_{tag}_s{seed}", f"--batch_size 8 {extra}".strip())
        for seed in range(20271, 20274)
        for arch, tag, extra in variants
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="jobs")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    specs = {
        "pilot_runs.tsv": [row("tinystories", arch, "configs/pilot_tinystories.json", 1337, "pilot") for arch in HEADLINE_METHODS],
        "pilot_lr.tsv": generate_pilot_lr(),
        "pilot_h.tsv": generate_symp_stage("--presymp_h", (0.03, 0.1, 0.3), "h"),
        "pilot_scalar.tsv": generate_symp_stage("--scalar_lr_mult", (1, 5, 10), "scalar"),
        "pilot_damping.tsv": generate_damping_pilot(),
        "pilot_noise.tsv": generate_noise_pilot(),
        "ablation_50m.tsv": generate_ablation(),
        "core_runs.tsv": generate_core(),
    }
    for name, rows in specs.items():
        path = out / name
        path.write_text("\n".join(rows) + "\n")
        print(f"wrote {path} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
