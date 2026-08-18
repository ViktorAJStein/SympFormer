#!/usr/bin/env python3
"""Verify deterministic non-Git source fingerprints for run manifests."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import source_fingerprint


def main():
    with tempfile.TemporaryDirectory(prefix="source-fingerprint-") as directory:
        config = Path(directory) / "config.json"
        config.write_text('{"value": 1}\n')
        first = source_fingerprint(str(config))
        second = source_fingerprint(str(config))
        assert first == second
        assert first["algorithm"] == "sha256"
        assert len(first["sha256"]) == 64
        labels = {record["path"] for record in first["files"]}
        assert {"train.py", "model.py", "data.py", "pyproject.toml", "uv.lock"} <= labels

        config.write_text('{"value": 2}\n')
        changed = source_fingerprint(str(config))
        assert changed["sha256"] != first["sha256"]
    print("PASS: source hash is stable and changes when the effective config changes")


if __name__ == "__main__":
    main()
