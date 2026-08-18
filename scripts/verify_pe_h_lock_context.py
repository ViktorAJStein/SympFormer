#!/usr/bin/env python3
"""Refuse Stage 2 when local source or dataset differs from the Stage-1 lock."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import source_fingerprint, source_revision

from make_pe_h_200m_specs import read_lr_lock


def current_source_identity(config: str) -> str:
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
    lock = read_lr_lock(lock_file)
    observed_source = current_source_identity(config)
    if observed_source != lock.get("source_revision"):
        raise SystemExit(
            "current executable source differs from the Stage-1 lock; "
            "do not submit Stage 2"
        )
    for split in ("train", "val"):
        observed = manifest_sha(data_dir / f"tinystories_{split}.bin.manifest.json")
        expected = lock.get(f"{split}_sha256")
        if observed != expected:
            raise SystemExit(
                f"current {split} dataset differs from the Stage-1 lock; "
                "do not submit Stage 2"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock_file", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--config", default="configs/core_50m.json")
    args = parser.parse_args()
    verify_context(args.lock_file, args.data_dir, args.config)
    print("PASS: Stage-2 source and TinyStories manifests match the Stage-1 lock")


if __name__ == "__main__":
    main()
