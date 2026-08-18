#!/usr/bin/env python3
"""Verify paired headline and focused-ablation statistics on synthetic rows."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from summarize_runs import paired_ablations, paired_differences


def row(seed, arch, tag, value):
    return {
        "dataset": "synthetic", "n_layer": 8, "n_head": 8, "n_embd": 512,
        "block_size": 512, "tokens": 1000, "seed": seed, "arch": arch,
        "run_name": f"{tag}_s{seed}", "run_tag": tag, "final_val": value,
    }


def main():
    rows = []
    for seed in (1, 2, 3):
        baseline = 4.0 + 0.01 * seed
        causal = baseline - 0.10
        rows.extend([
            row(seed, "baseline", "core", baseline),
            row(seed, "causal_symp_pe", "core", causal),
            row(seed, "baseline", "abl_reference", baseline),
            row(seed, "baseline_capacity", "abl_capacity_control", baseline - 0.02),
            row(seed, "causal_symp_pe", "abl_reference", causal),
            row(seed, "causal_symp_pe", "abl_fixed_eta", causal + 0.03),
        ])
    headline = paired_differences(rows)
    core = [x for x in headline if x["arch"] == "causal_symp_pe" and x["run_tag"] == "core" and x["n"] == 3]
    if len(core) != 1 or not math.isclose(core[0]["delta_final_val_mean"], -0.10, abs_tol=1e-12):
        raise AssertionError(f"bad headline pairing: {core}")
    ablations = paired_ablations(rows)
    got = {x["variant"]: x["delta_final_val_mean"] for x in ablations}
    if not math.isclose(got["baseline_capacity:abl_capacity_control"], -0.02, abs_tol=1e-12):
        raise AssertionError(f"bad capacity pairing: {got}")
    if not math.isclose(got["causal_symp_pe:abl_fixed_eta"], 0.03, abs_tol=1e-12):
        raise AssertionError(f"bad causal ablation pairing: {got}")
    print("PASS: headline and focused-ablation same-seed contrasts use the intended references.")


if __name__ == "__main__":
    main()
