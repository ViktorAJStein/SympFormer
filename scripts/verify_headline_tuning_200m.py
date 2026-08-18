#!/usr/bin/env python3
"""Synthetic end-to-end verification for both staged 200M tuning campaigns."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

from analyze_headline_lr_200m import analyze as analyze_lr
from analyze_headline_lr_200m import lock_payload as lr_lock_payload
from analyze_headline_lr_200m import write_lock as write_lr_lock
from analyze_pe_h_200m import analyze as analyze_h
from analyze_pe_h_200m import final_lock_payload, write_lock as write_h_lock
from make_headline_lr_200m_specs import METHOD_LRS, lr_rows
from make_pe_h_200m_specs import h_rows
from analyze_pe_tuning import parse_spec
from plot_headline_lr_200m import plot_curves as plot_lr_curves, plot_means as plot_lr_means
from verify_pe_h_lock_context import current_source_identity, verify_context


ROOT = Path(__file__).resolve().parents[1]


def lr_effect(method: str, peak_lr: float) -> float:
    tables = {
        "baseline": {0.0003: 0.030, 0.0006: 0.000, 0.0010: -0.003, 0.0015: 0.020},
        "yurii_lt": {0.0003: 0.025, 0.0006: 0.006, 0.0010: -0.020, 0.0015: -0.010},
        "causal_symp_pe": {0.0015: 0.000, 0.0020: -0.012, 0.0025: -0.024, 0.0030: -0.018},
    }
    return tables[method][peak_lr]


def fake_manifest_item(raw: str, family: str, **extra) -> dict:
    item = parse_spec(raw)
    item["family"] = family
    item.update(extra)
    return item


def write_fake_run(root: Path, item: dict, final: float) -> Path:
    run_dir = root / f"{item['arch']}_{item['run_name']}"
    run_dir.mkdir(parents=True)
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
        "model_config": {"n_layer": 8, "n_head": 8, "n_embd": 512, "block_size": 512},
        "trainable_parameters": 53_000_000,
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
                "trainable_parameters": 53_000_000,
                "peak_memory_mb": 4800,
            }
        )
    )
    fields = [
        "step", "train_loss", "val_loss", "lr", "wall_dt_s", "wall_cum_s",
        "tokens_step", "tokens_cum", "h_mean", "hY_mean", "xi_mean", "rX", "rP",
        "c_log_mean", "c_lin_mean", "leak_warnings", "sched_t_start", "sched_t_end",
    ]
    with (run_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, fraction in enumerate((0.5, 0.75, 1.0), start=1):
            writer.writerow(
                {
                    "step": index,
                    "train_loss": final + 0.04,
                    "val_loss": final + (3 - index) * 0.01,
                    "lr": item["expected_args"].get("peak_lr", 0.0006),
                    "wall_dt_s": wall / 3,
                    "wall_cum_s": wall * index / 3,
                    "tokens_step": 262_144,
                    "tokens_cum": int(item["expected_tokens"] * fraction),
                    "h_mean": item["expected_args"].get("presymp_h", 0.1),
                    "hY_mean": item["expected_args"].get("presymp_h", 0.1),
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


def main() -> None:
    launcher = (ROOT / "jobs" / "headline_tuning_200m.sbatch").read_text()
    for directive in (
        "#SBATCH --cpus-per-task=4",
        "#SBATCH --partition=gpusmpshort",
        "#SBATCH --mem=16G",
        "#SBATCH --time=01:30:00",
        "#SBATCH --gres=gpu:rtx_a6000:1",
    ):
        assert directive in launcher
    assert '"${run_dir}/failure.json"' in launcher
    assert "--array=0-23%4" in (ROOT / "jobs" / "submit_headline_lr_200m.sh").read_text()
    assert "--array=0-7%4" in (ROOT / "jobs" / "submit_pe_h_200m.sh").read_text()
    assert "verify_pe_h_lock_context.py" in (ROOT / "jobs" / "submit_pe_h_200m.sh").read_text()

    with tempfile.TemporaryDirectory(prefix="headline-tuning-") as directory:
        root = Path(directory)
        lr_root = root / "lr"
        manifests = {}
        for raw in lr_rows():
            parsed = parse_spec(raw)
            peak_lr = float(parsed["expected_args"]["peak_lr"])
            item = fake_manifest_item(raw, "headline_lr", method=parsed["arch"], peak_lr=peak_lr)
            base = {"baseline": 1.60, "yurii_lt": 1.58, "causal_symp_pe": 1.62}[item["arch"]]
            final = base + lr_effect(item["arch"], peak_lr) + (item["seed"] - 20276) * 0.01
            manifests[item["run_name"]] = write_fake_run(lr_root, item, final)

        runs, aggregate, complete = analyze_lr(lr_root)
        assert len(runs) == 24 and len(aggregate) == 12 and complete
        selected = {row["method"]: float(row["peak_lr"]) for row in aggregate if row["selected"]}
        assert selected == {"baseline": 0.0006, "yurii_lt": 0.001, "causal_symp_pe": 0.0025}
        lock = lr_lock_payload(aggregate, runs, complete)
        assert lock is not None
        lock_path = root / "locked_lrs.json"
        write_lr_lock(lock_path, lock)

        h_spec_path = root / "pe_h_200m.tsv"
        h_spec_path.write_text("\n".join(h_rows(lock)) + "\n")
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "verify_pe_h_200m_specs.py"),
                "--lock_file",
                str(lock_path),
                "--spec",
                str(h_spec_path),
            ],
            check=True,
        )

        context_data = root / "data"
        context_data.mkdir()
        context_lock = dict(lock)
        context_lock["source_revision"] = current_source_identity("configs/core_50m.json")
        context_lock["train_sha256"] = "a" * 64
        context_lock["val_sha256"] = "b" * 64
        context_lock_path = root / "context_lock.json"
        context_lock_path.write_text(json.dumps(context_lock))
        for split, digest in (("train", "a" * 64), ("val", "b" * 64)):
            (context_data / f"tinystories_{split}.bin.manifest.json").write_text(
                json.dumps({"sha256": digest})
            )
        verify_context(context_lock_path, context_data, "configs/core_50m.json")
        (context_data / "tinystories_val.bin.manifest.json").write_text(
            json.dumps({"sha256": "c" * 64})
        )
        try:
            verify_context(context_lock_path, context_data, "configs/core_50m.json")
        except SystemExit:
            pass
        else:
            raise AssertionError("Stage-2 context preflight accepted a changed dataset")

        mismatch_path = manifests["hlr200_pe_lr0p0025_s20276"]
        mismatch = json.loads(mismatch_path.read_text())
        mismatch["gpu"] = "different GPU"
        mismatch_path.write_text(json.dumps(mismatch))
        _, mismatch_aggregate, mismatch_complete = analyze_lr(lr_root)
        assert not mismatch_complete and not any(row["selected"] for row in mismatch_aggregate)
        mismatch["gpu"] = "NVIDIA RTX A6000"
        mismatch_path.write_text(json.dumps(mismatch))

        runs, aggregate, complete = analyze_lr(lr_root)
        plot_lr_curves(runs, root / "figures" / "lr")
        plot_lr_means(aggregate, root / "figures" / "lr")

        h_root = root / "h"
        for raw in h_rows(lock):
            parsed = parse_spec(raw)
            h_init = float(parsed["expected_args"]["presymp_h"])
            item = fake_manifest_item(raw, "pe_h_init", h_init=h_init)
            effects = {0.015: -0.002, 0.03: -0.012, 0.06: -0.004, 0.10: 0.0}
            final = 1.55 + effects[h_init] + (item["seed"] - 20278) * 0.01
            write_fake_run(h_root, item, final)
        h_runs, pairs, h_aggregate, h_complete = analyze_h(h_root, lock)
        assert len(h_runs) == 8 and len(pairs) == 6 and h_complete
        mismatched_lock = dict(lock)
        mismatched_lock["gpu"] = "different GPU"
        assert not analyze_h(h_root, mismatched_lock)[3]
        chosen = next(row for row in h_aggregate if row["selected"])
        assert math.isclose(float(chosen["h_init"]), 0.03)
        final_lock = final_lock_payload(lock, h_aggregate, h_runs, h_complete)
        assert final_lock is not None and math.isclose(final_lock["presymp_h"], 0.03)
        write_h_lock(root / "locked_pe_config.json", final_lock)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "plot_pe_h_200m.py"),
                str(h_root),
                "--lock_file",
                str(lock_path),
                "--out_prefix",
                str(root / "figures" / "h"),
            ],
            check=True,
        )

        figures = [
            root / "figures" / f"lr_{kind}.{suffix}"
            for kind in ("curves", "means")
            for suffix in ("png", "pdf")
        ] + [
            root / "figures" / f"h_{kind}.{suffix}"
            for kind in ("curves", "paired")
            for suffix in ("png", "pdf")
        ]
        assert all(path.is_file() and path.stat().st_size > 0 for path in figures)

    print(
        "PASS: staged 24-run LR selection, conservative tie rule, immutable LR lock, "
        "provenance rejection, Stage-2 source/data preflight, conditional eight-run h gate, final PE lock, resources, and plots"
    )


if __name__ == "__main__":
    main()
