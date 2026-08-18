#!/usr/bin/env python3
"""Verify persisted deterministic generation examples for a comparison cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--require_arches", default="")
    args = parser.parse_args()

    records = []
    for path in sorted(args.root.rglob("samples.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                record["_path"] = str(path)
                records.append(record)
    if len(records) != args.expected:
        raise SystemExit(f"expected {args.expected} sample records, found {len(records)}")
    required = {item for item in args.require_arches.split(",") if item}
    arches = {record["arch"] for record in records}
    if required and not required.issubset(arches):
        raise SystemExit(f"missing required architectures: {sorted(required - arches)}")
    for record in records:
        assert record["do_sample"] is False, record["_path"]
        assert record["prompt_token_ids"], record["_path"]
        assert record["continuation_token_ids"], record["_path"]
        assert len(record["continuation_token_ids"]) <= 256, record["_path"]
        if record["prompt_text"] is not None:
            assert isinstance(record["continuation_text"], str) and record["continuation_text"], record["_path"]
    prompts = {tuple(record["prompt_token_ids"]) for record in records}
    if len(prompts) != 1:
        raise SystemExit(f"comparison does not use one common prompt: {len(prompts)} prompts")
    print(f"PASS: {len(records)} deterministic samples, common prompt, arches={sorted(arches)}")


if __name__ == "__main__":
    main()
