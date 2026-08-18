#!/usr/bin/env python3
"""Verify that an end-to-end training run records its source fingerprint."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import source_fingerprint


def main():
    with tempfile.TemporaryDirectory(prefix="manifest-fingerprint-") as directory:
        tmp = Path(directory)
        data_dir = tmp / "data"
        out_dir = tmp / "out"
        data_dir.mkdir()
        tokens = (np.arange(4096, dtype=np.uint16) % 64).astype(np.uint16)
        tokens.tofile(data_dir / "tinystories_train.bin")
        tokens[::-1].tofile(data_dir / "tinystories_val.bin")
        command = [
            sys.executable,
            "train.py",
            "--data_dir",
            str(data_dir),
            "--out_dir",
            str(out_dir),
            "--dataset",
            "tinystories",
            "--arch",
            "baseline",
            "--run_name",
            "fingerprint",
            "--device",
            "cpu",
            "--n_layer",
            "1",
            "--n_head",
            "1",
            "--n_embd",
            "8",
            "--block_size",
            "8",
            "--vocab_size",
            "64",
            "--max_steps",
            "1",
            "--warmup_steps",
            "0",
            "--global_tokens_per_step",
            "0",
            "--batch_size",
            "1",
            "--grad_accum_steps",
            "1",
            "--eval_interval",
            "1",
            "--eval_batches",
            "1",
            "--log_interval",
            "1",
            "--optimizer",
            "adamw",
        ]
        subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        manifest_path = out_dir / "baseline_fingerprint" / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        recorded = manifest["source_fingerprint"]
        expected = source_fingerprint("")
        assert recorded == expected
        assert len(recorded["sha256"]) == 64
        assert (out_dir / "baseline_fingerprint" / "summary.json").is_file()
    print("PASS: end-to-end training manifest contains the deterministic source hash")


if __name__ == "__main__":
    main()
