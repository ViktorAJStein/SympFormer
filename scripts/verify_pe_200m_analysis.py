#!/usr/bin/env python3
"""Synthetic end-to-end verification of the 200M PE confirmation."""

import csv
import json
import math
import tempfile
from pathlib import Path

from analyze_pe_200m import analyze, expected_runs
from analyze_pe_tuning import RUN_FIELDS, write_csv
from make_pe_200m_specs import confirmation_rows
from plot_pe_200m import plot_curves, plot_paired


ROOT = Path(__file__).resolve().parents[1]


def delta(item):
    values = {
        "pe200_adam0p001": {20274: -0.012, 20275: 0.002},
        "pe200_adam0p0015": {20274: -0.030, 20275: -0.025},
        "pe200_muon0p04": {20274: -0.008, 20275: -0.009},
    }
    return values.get(item["candidate"], {}).get(item["seed"], 0.0)


def write_fake_run(root, item):
    run_dir = root / "confirmation_200m" / f"{item['arch']}_{item['run_name']}"
    run_dir.mkdir(parents=True)
    anchor = 1.60 + (item["seed"] - 20274) * 0.01
    final = anchor + delta(item)
    wall = 3300.0
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
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
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
        "step", "train_loss", "val_loss", "lr", "wall_dt_s", "wall_cum_s",
        "tokens_step", "tokens_cum", "h_mean", "hY_mean", "xi_mean", "rX",
        "rP", "c_log_mean", "c_lin_mean", "leak_warnings", "sched_t_start",
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
                    "xi_mean": 0.0,
                    "rX": 0.0,
                    "rP": 0.0,
                    "c_log_mean": 3.0,
                    "c_lin_mean": 0.0001,
                    "leak_warnings": 0,
                    "sched_t_start": 1.0,
                    "sched_t_end": 1.8,
                }
            )
    (run_dir / "slurm_status.tsv").write_text(
        "state\tjob_id\tarray_id\tstart\tend\texit_code\n"
        "running\t1\t0\tstart\t\t\n"
        "complete\t1\t0\t\tend\t0\n"
    )
    return manifest_path


def main():
    expected_tsv = "\n".join(confirmation_rows()) + "\n"
    assert (ROOT / "jobs" / "pe_confirm_200m.tsv").read_text() == expected_tsv

    launcher = (ROOT / "jobs" / "pe_confirm_200m.sbatch").read_text()
    for directive in (
        "#SBATCH --cpus-per-task=4",
        "#SBATCH --partition=gpusmpshort",
        "#SBATCH --mem=16G",
        "#SBATCH --time=01:30:00",
        "#SBATCH --gres=gpu:rtx_a6000:1",
    ):
        assert directive in launcher
    assert '"${run_dir}/failure.json"' in launcher
    submitter = (ROOT / "jobs" / "submit_pe_200m.sh").read_text()
    assert "--array=0-7%4" in submitter

    with tempfile.TemporaryDirectory(prefix="pe200-analysis-") as directory:
        root = Path(directory)
        manifests = {}
        for item in expected_runs():
            manifests[item["run_name"]] = write_fake_run(root, item)

        runs, pairs, aggregate = analyze(root)
        assert len(runs) == 8
        assert len(pairs) == 6
        assert len(aggregate) == 3
        assert all(row["status"] == "valid" for row in runs)
        by_candidate = {row["candidate"]: row for row in aggregate}
        winner = by_candidate["pe200_adam0p0015"]
        assert winner["n_valid"] == 2
        assert winner["wins_vs_anchor"] == 2
        assert math.isclose(winner["mean_delta_final_val"], -0.0275, abs_tol=1e-12)
        assert winner["confirms_at_200m"] == 1
        assert by_candidate["pe200_adam0p001"]["confirms_at_200m"] == 0
        assert by_candidate["pe200_muon0p04"]["confirms_at_200m"] == 0

        # A GPU provenance mismatch must invalidate the pair without hiding the run.
        mismatch_path = manifests["pe200_adam0p0015_s20274"]
        mismatch = json.loads(mismatch_path.read_text())
        mismatch["gpu"] = "different GPU"
        mismatch_path.write_text(json.dumps(mismatch))
        mismatch_runs, mismatch_pairs, mismatch_aggregate = analyze(root)
        assert len(mismatch_runs) == 8
        affected = next(
            row
            for row in mismatch_pairs
            if row["candidate"] == "pe200_adam0p0015" and row["seed"] == 20274
        )
        assert affected["valid_pair"] == 0
        assert affected["provenance_match"] == 0
        assert next(
            row for row in mismatch_aggregate if row["candidate"] == "pe200_adam0p0015"
        )["confirms_at_200m"] == 0
        mismatch["gpu"] = "NVIDIA RTX A6000"
        mismatch_path.write_text(json.dumps(mismatch))

        runs, pairs, aggregate = analyze(root)
        out_dir = root / "analysis"
        write_csv(out_dir / "runs.csv", runs, RUN_FIELDS)
        write_csv(out_dir / "paired_vs_anchor.csv", pairs)
        write_csv(out_dir / "aggregate.csv", aggregate)
        prefix = out_dir / "pe200"
        plot_curves(runs, prefix)
        plot_paired(pairs, aggregate, prefix)
        figures = [
            out_dir / f"pe200_{name}.{suffix}"
            for name in ("curves", "paired")
            for suffix in ("png", "pdf")
        ]
        assert all(path.is_file() and path.stat().st_size > 0 for path in figures)
    print(
        "PASS: eight-run strict audit, paired two-seed gate, provenance rejection, "
        "90-minute resources, CSV export, and four figure files"
    )


if __name__ == "__main__":
    main()
