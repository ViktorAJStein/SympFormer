#!/usr/bin/env python3
"""Verify that current source and dataset manifests match an immutable lock."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import source_fingerprint, source_revision


def source_identity(config: str) -> str:
    revision = source_revision() or "NO_GIT"
    digest = source_fingerprint(config)["sha256"]
    return f"git:{revision};sha256:{digest}"


def manifest_sha(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(f"missing dataset manifest: {path}")
    try:
        digest = json.loads(path.read_text())["sha256"]
    except (KeyError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid dataset manifest: {path}: {exc}") from exc
    if not isinstance(digest, str) or len(digest) != 64:
        raise SystemExit(f"invalid dataset SHA-256 in {path}")
    return digest


def verify_context(lock_file: Path, data_dir: Path, config: str) -> None:
    if not lock_file.is_file():
        raise SystemExit(f"missing campaign lock: {lock_file}")
    lock = json.loads(lock_file.read_text())
    if lock.get("selection_complete") is not True:
        raise SystemExit("campaign lock is incomplete")
    if source_identity(config) != lock.get("source_revision"):
        raise SystemExit("current executable source differs from the upstream lock")
    for split in ("train", "val"):
        observed = manifest_sha(data_dir / f"tinystories_{split}.bin.manifest.json")
        if observed != lock.get(f"{split}_sha256"):
            raise SystemExit(f"current {split} dataset differs from the upstream lock")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock_file", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--config", default="configs/core_50m.json")
    args = parser.parse_args()
    verify_context(args.lock_file, args.data_dir, args.config)
    print("PASS: current source and TinyStories manifests match the upstream lock")


if __name__ == "__main__":
    main()
