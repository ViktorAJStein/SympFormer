#!/usr/bin/env python3
"""Synthetic end-to-end verification of the 100M backfill campaign."""

import csv
import json
import math
import tempfile
from pathlib import Path

from analyze_pe_100m import analyze, expected_runs
from analyze_pe_tuning import RUN_FIELDS, write_csv
from plot_pe_100m import plot_controls, plot_curves, plot_paired


ROOT = Path(__file__).resolve().parents[1]


def final_loss(item):
    candidate = item["candidate"]
    if candidate == "pe100_baseline":
        return 1.45
    if candidate == "pe100_yurii":
        return 1.44
    anchor = 1.50
    deltas = {
        "pe100_h0p2": -0.020,
        "pe100_clog2": -0.015,
        "pe100_scalar10": -0.011,
        "pe100_muon0p01": -0.010,
    }
    return anchor + deltas.get(candidate, 0.0 if candidate == "pe100_anchor" else 0.001)


def write_fake_run(root, item):
    run_dir = root / item["family"] / f"{item['arch']}_{item['run_name']}"
    run_dir.mkdir(parents=True)
    final = final_loss(item)
    wall = 2400.0
    manifest = {
        "args": {
            "run_name": item["run_name"],
            "dataset": item["dataset"],
            "arch": item["arch"],
            "seed": item["seed"],
            "config": item["config"],
            "global_tokens_per_step": 262_144,
            "eval_batches": 160,
            "amp_dtype": "bfloat16",
            "device": "cuda",
            **item["expected_args"],
        },
        "model_config": {
            "n_layer": 8,
            "n_head": 8,
            "n_embd": 512,
            "block_size": 512,
        },
        "trainable_parameters": 53_297_728,
        "actual_max_tokens": item["expected_tokens"],
        "gpu": "NVIDIA RTX A6000",
        "source_revision": "synthetic",
        "source_fingerprint": {"algorithm": "sha256", "sha256": "source-hash"},
        "train_data": {"manifest": {"sha256": "train-hash"}},
        "val_data": {"manifest": {"sha256": "val-hash"}},
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest))
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "best_val": final,
                "final_val": final,
                "tokens": item["expected_tokens"],
                "wall_cum_s": wall,
                "trainable_parameters": 53_297_728,
                "peak_memory_mb": 4900,
            }
        )
    )
    fields = [
        "step",
        "train_loss",
        "val_loss",
        "lr",
        "wall_dt_s",
        "wall_cum_s",
        "tokens_step",
        "tokens_cum",
        "h_mean",
        "hY_mean",
        "xi_mean",
        "rX",
        "rP",
        "c_log_mean",
        "c_lin_mean",
        "leak_warnings",
        "sched_t_start",
        "sched_t_end",
    ]
    with (run_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, fraction in enumerate((0.5, 0.75, 1.0), start=1):
            writer.writerow(
                {
                    "step": index,
                    "train_loss": final + 0.05,
                    "val_loss": final + (3 - index) * 0.02,
                    "lr": 0.0006,
                    "wall_dt_s": wall / 3,
                    "wall_cum_s": wall * index / 3,
                    "tokens_step": 262_144,
                    "tokens_cum": int(item["expected_tokens"] * fraction),
                    "h_mean": 0.1,
                    "hY_mean": 0.1,
                    "xi_mean": 1.0,
                    "rX": 0.0,
                    "rP": 0.0,
                    "c_log_mean": 3.0,
                    "c_lin_mean": 0.0001,
                    "leak_warnings": 0,
                    "sched_t_start": 1.0,
                    "sched_t_end": 2.0,
                }
            )
    (run_dir / "slurm_status.tsv").write_text(
        "state\tjob_id\tarray_id\tstart\tend\texit_code\n"
        "running\t1\t0\tstart\t\t\n"
        "complete\t1\t0\t\tend\t0\n"
    )


def main():
    launcher = (ROOT / "jobs" / "pe_screen_100m.sbatch").read_text()
    for directive in (
        "#SBATCH --cpus-per-task=4",
        "#SBATCH --mem=24G",
        "#SBATCH --time=01:30:00",
        "#SBATCH --gres=gpu:rtx_a6000:1",
    ):
        assert directive in launcher
    submitter = (ROOT / "jobs" / "submit_pe_100m.sh").read_text()
    for array in ("--array=0-9%4", "--array=0-8%4", "--array=0-1%2"):
        assert array in submitter

    with tempfile.TemporaryDirectory(prefix="pe100-analysis-") as directory:
        root = Path(directory)
        for item in expected_runs():
            write_fake_run(root, item)
        runs, pairs, controls = analyze(root)
        assert len(runs) == 21
        assert len(pairs) == 18
        assert len(controls) == 3
        assert all(row["status"] == "valid" for row in runs)
        selected = [row["candidate"] for row in pairs if row["advance_to_200m"]]
        assert selected == ["pe100_h0p2", "pe100_clog2", "pe100_scalar10"]
        muon = next(row for row in pairs if row["candidate"] == "pe100_muon0p01")
        assert math.isclose(float(muon["delta_final_val"]), -0.010, abs_tol=1e-12)
        assert muon["passes_effect_threshold"] == 1
        assert muon["advance_to_200m"] == 0

        out_dir = root / "analysis"
        write_csv(out_dir / "runs.csv", runs, RUN_FIELDS)
        write_csv(out_dir / "paired_vs_pe_anchor.csv", pairs)
        write_csv(out_dir / "controls.csv", controls)
        prefix = out_dir / "pe100"
        plot_curves(runs, prefix)
        plot_paired(pairs, prefix)
        plot_controls(controls, prefix)
        figures = [
            out_dir / f"pe100_{name}.{suffix}"
            for name in ("curves", "paired", "controls")
            for suffix in ("png", "pdf")
        ]
        assert all(path.is_file() and path.stat().st_size > 0 for path in figures)
    print(
        "PASS: 21-run audit, maximum-three pruning, reduced resources, "
        "CSV export, and six figure files"
    )


if __name__ == "__main__":
    main()
