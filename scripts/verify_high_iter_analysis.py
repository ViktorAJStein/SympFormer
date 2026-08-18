#!/usr/bin/env python3
"""Synthetic end-to-end verification for high-iteration analysis."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_high_iter_specs import high_iter_rows


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_run(root: Path, *, dataset: str, arch: str, config: str, seed: int, run_name: str, flags: str, final: float) -> None:
    run_dir = root / f"{arch}_{run_name}"
    run_dir.mkdir(parents=True)
    args = {"dataset": dataset, "arch": arch, "config": config, "seed": seed, "run_name": run_name, "device": "cuda", "global_tokens_per_step": 262144, "eval_batches": 160, "amp_dtype": "bfloat16"}
    parts = flags.split()
    i = 0
    while i < len(parts):
        flag = parts[i]
        if flag in ("--no_v0_init", "--presymp_mlp_use_attn_vel", "--eta_learnable", "--sample_final"):
            args[flag[2:].replace("-", "_")] = True
            i += 1
        elif flag == "--no-eta_learnable":
            args["eta_learnable"] = False
            i += 1
        else:
            key = flag[2:].replace("-", "_")
            raw = parts[i + 1]
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
            args[key] = value
            i += 2
    tokens = (int(args["max_tokens"]) // 262144) * 262144
    write_json(
        run_dir / "run_manifest.json",
        {
            "args": args,
            "model_config": {"n_layer": 8, "n_head": 8, "n_embd": 512, "block_size": 512},
            "gpu": "NVIDIA RTX A6000",
            "source_revision": "NO_GIT",
            "source_fingerprint": {"sha256": "abc"},
            "train_data": {"manifest": {"sha256": "train"}},
            "val_data": {"manifest": {"sha256": "val"}},
            "trainable_parameters": 123,
        },
    )
    write_json(run_dir / "summary.json", {"tokens": tokens, "final_val": final, "best_val": final - 0.01, "wall_cum_s": 100.0, "peak_memory_mb": 1000})
    with (run_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "train_loss", "val_loss", "lr", "wall_dt_s", "wall_cum_s", "tokens_step", "tokens_cum", "h_mean", "hY_mean", "xi_mean", "rX", "rP", "c_log_mean", "c_lin_mean", "leak_warnings", "sched_t_start", "sched_t_end"])
        writer.writeheader()
        for j, frac in enumerate((0.2, 0.5, 0.75, 1.0), start=1):
            writer.writerow({"step": j, "train_loss": final + 0.2 * (1 - frac), "val_loss": final + 0.1 * (1 - frac), "lr": 0.0, "wall_dt_s": 1.0, "wall_cum_s": 25.0 * j, "tokens_step": 262144, "tokens_cum": int(tokens * frac), "h_mean": "", "hY_mean": "", "xi_mean": "", "rX": "", "rP": "", "c_log_mean": "", "c_lin_mean": "", "leak_warnings": 0, "sched_t_start": "", "sched_t_end": ""})
    (run_dir / "slurm_status.tsv").write_text("state\tjob_id\tarray_id\tstart\tend\texit_code\nrunning\t1\t0\tstart\t\t\ncomplete\t1\t0\t\tend\t0\n")


def main() -> None:
    lr_lock = {
        "selection_complete": True,
        "methods": {
            "baseline": {"peak_lr": 0.0015, "scalar_lr_mult": None},
            "yurii_lt": {"peak_lr": 0.0015, "scalar_lr_mult": 2.0},
            "causal_symp_pe": {"peak_lr": 0.0025, "scalar_lr_mult": 1.2},
        },
    }
    pe_lock = {"selection_complete": True, "presymp_h": 0.1, "damping_learnable": True, "eta_log": 3.0, "eta_lin": 0.0001}
    with tempfile.TemporaryDirectory(prefix="high-iter-verify-") as tmp:
        base = Path(tmp)
        spec = base / "spec.tsv"
        rows = high_iter_rows(lr_lock, pe_lock, dataset="tinystories", tokens=1_000_000_000, seeds=(1, 2), tag="verify1b")
        spec.write_text("\n".join(rows) + "\n")
        root = base / "runs"
        finals = {"baseline": {1: 1.50, 2: 1.52}, "yurii_lt": {1: 1.49, 2: 1.53}, "causal_symp_pe": {1: 1.48, 2: 1.50}}
        for line in rows:
            dataset, arch, config, seed, run_name, flags = line.split("\t")
            write_run(root, dataset=dataset, arch=arch, config=config, seed=int(seed), run_name=run_name, flags=flags, final=finals[arch][int(seed)])
        out = base / "analysis"
        subprocess.run([sys.executable, "scripts/analyze_high_iter.py", str(root), "--spec", str(spec), "--out_dir", str(out)], check=True)
        agg = list(csv.DictReader((out / "endpoint_aggregate.csv").open()))
        got = {row["comparison"]: float(row["mean_delta_final_val"]) for row in agg}
        assert abs(got["pe_minus_baseline"] + 0.02) < 1e-12, got
        assert abs(got["pe_minus_yurii_lt"] + 0.02) < 1e-12, got
    print("PASS: synthetic high-iteration endpoint and trajectory analysis")


if __name__ == "__main__":
    main()
