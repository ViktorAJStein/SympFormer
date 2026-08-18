#!/usr/bin/env python3
"""End-to-end synthetic verification of PE campaign analysis and figures."""

import csv
import json
import math
import tempfile
from pathlib import Path

from analyze_pe_tuning import (
    RUN_FIELDS,
    analyze,
    expected_runs,
    write_csv,
)
from plot_pe_tuning import plot_curves, plot_directional, plot_paired


def final_loss(item):
    seed_offset = {1337: 0.0, 20272: 0.002, 20273: -0.001}[item["seed"]]
    candidate = item["candidate"]
    if candidate.startswith("pe350_"):
        anchor = 1.40 + seed_offset
        if candidate == "pe350_anchor":
            return anchor
        if candidate == "pe350_h0p2":
            return anchor - 0.006
        return anchor + 0.001
    values = {
        "pe1b_baseline": 1.27,
        "pe1b_yurii": 1.26,
        "pe1b_anchor": 1.30,
        "pe1b_h0p2": 1.28,
        "pe1b_clog2": 1.29,
        "pe1b_scalar10": 1.285,
    }
    return values[candidate]


def write_fake_run(root, item):
    run_dir = root / item["family"] / f"{item['arch']}_{item['run_name']}"
    run_dir.mkdir(parents=True)
    final = final_loss(item)
    wall = 4 * 3600
    manifest = {
        "args": {
            "run_name": item["run_name"],
            "dataset": "tinystories",
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
        "gpu": "Synthetic A6000",
        "source_revision": "synthetic",
        "source_fingerprint": {"algorithm": "sha256", "sha256": "source-hash"},
        "train_data": {"manifest": {"sha256": "train-hash"}},
        "val_data": {"manifest": {"sha256": "val-hash"}},
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest))
    summary = {
        "best_val": final - 0.0002,
        "final_val": final,
        "tokens": item["expected_tokens"],
        "wall_cum_s": wall,
        "trainable_parameters": 53_297_728,
        "peak_memory_mb": 4552,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary))
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
    with tempfile.TemporaryDirectory(prefix="pe-analysis-") as directory:
        root = Path(directory)
        for item in expected_runs():
            write_fake_run(root, item)
        runs, pairs, aggregate, directional = analyze(root)
        assert len(runs) == 63
        assert len(pairs) == 54
        assert len(aggregate) == 18
        assert len(directional) == 6
        assert all(row["status"] == "valid" for row in runs)
        qualified = [
            row["candidate"]
            for row in aggregate
            if row["qualifies_for_confirmation"]
        ]
        assert qualified == ["pe350_h0p2"], qualified
        h_row = next(row for row in aggregate if row["candidate"] == "pe350_h0p2")
        assert math.isclose(h_row["mean_delta_final_val"], -0.006, abs_tol=1e-12)

        out_dir = root / "analysis"
        write_csv(out_dir / "runs.csv", runs, RUN_FIELDS)
        write_csv(out_dir / "paired_350m.csv", pairs)
        write_csv(out_dir / "aggregate_350m.csv", aggregate)
        write_csv(out_dir / "directional_1b.csv", directional)
        prefix = out_dir / "pe_tuning"
        plot_curves(runs, prefix)
        plot_paired(pairs, aggregate, prefix)
        plot_directional(directional, prefix)
        expected_figures = [
            out_dir / f"pe_tuning_{name}.{suffix}"
            for name in ("curves", "paired", "directional")
            for suffix in ("png", "pdf")
        ]
        assert all(path.is_file() and path.stat().st_size > 0 for path in expected_figures)

        # A mislabeled effective hyperparameter must invalidate the run.
        wrong = next(row for row in runs if row["run_name"] == "pe350_h0p3_s1337")
        wrong_manifest_path = Path(wrong["run_dir"]) / "run_manifest.json"
        wrong_manifest = json.loads(wrong_manifest_path.read_text())
        wrong_manifest["args"]["presymp_h"] = 0.25
        wrong_manifest_path.write_text(json.dumps(wrong_manifest))
        wrong_runs, _, _, _ = analyze(root)
        audited_wrong = next(
            row for row in wrong_runs if row["run_name"] == "pe350_h0p3_s1337"
        )
        assert audited_wrong["status"] == "invalid"
        assert "spec_mismatch" in audited_wrong["reasons"]
        wrong_manifest["args"]["presymp_h"] = 0.3
        wrong_manifest_path.write_text(json.dumps(wrong_manifest))

        # A source mismatch leaves the task visible but invalidates its pair.
        drift = next(row for row in runs if row["run_name"] == "pe350_h0p2_s1337")
        drift_manifest_path = Path(drift["run_dir"]) / "run_manifest.json"
        drift_manifest = json.loads(drift_manifest_path.read_text())
        drift_manifest["source_fingerprint"]["sha256"] = "different-source"
        drift_manifest_path.write_text(json.dumps(drift_manifest))
        _, drift_pairs, drift_aggregate, _ = analyze(root)
        drift_pair = next(
            row
            for row in drift_pairs
            if row["candidate"] == "pe350_h0p2" and row["seed"] == 1337
        )
        assert drift_pair["provenance_match"] == 0
        drift_result = next(
            row for row in drift_aggregate if row["candidate"] == "pe350_h0p2"
        )
        assert drift_result["n_valid"] == 2
        assert drift_result["qualifies_for_confirmation"] == 0
    print(
        "PASS: 63-run audit, strict spec/provenance rejection, qualification "
        "gate, CSV export, and six figure files"
    )


if __name__ == "__main__":
    main()
