#!/usr/bin/env python3
"""Generate the conditional PE initial-step bracket from an immutable LR lock."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from make_headline_lr_200m_specs import SCALAR_PEAK_LR, format_number
from make_pe_tuning_specs import PE_FLAGS, row, slug


CONFIG = "configs/core_50m.json"
H_SEEDS = (20278, 20279)
H_VALUES = (0.015, 0.03, 0.06, 0.10)
TOKENS_200M = 200_000_000


def read_lr_lock(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"missing LR lock: {path}")
    lock = json.loads(path.read_text())
    if (
        lock.get("format_version") != 1
        or lock.get("campaign") != "headline_lr_200m"
        or lock.get("selection_complete") is not True
        or set((lock.get("methods") or {})) != {"baseline", "yurii_lt", "causal_symp_pe"}
    ):
        raise SystemExit("LR lock is incomplete or from the wrong campaign")
    for field in ("source_revision", "train_sha256", "val_sha256", "gpu"):
        if not isinstance(lock.get(field), str) or not lock[field]:
            raise SystemExit(f"LR lock is missing provenance field: {field}")
    if int(lock.get("actual_tokens_per_run", -1)) != 199_753_728:
        raise SystemExit("LR lock has the wrong per-run token count")
    pe = lock["methods"]["causal_symp_pe"]
    peak = float(pe["peak_lr"])
    scalar = float(pe["scalar_lr_mult"])
    if not math.isclose(peak * scalar, SCALAR_PEAK_LR, rel_tol=0, abs_tol=1e-12):
        raise SystemExit("locked PE scalar peak LR is not 0.003")
    return lock


def h_rows(lock: dict) -> list[str]:
    pe = lock["methods"]["causal_symp_pe"]
    peak = float(pe["peak_lr"])
    scalar = float(pe["scalar_lr_mult"])
    rows = []
    for seed in H_SEEDS:
        for h in H_VALUES:
            flags = (
                f"{PE_FLAGS} --max_tokens {TOKENS_200M} "
                f"--peak_lr {format_number(peak)} "
                f"--scalar_lr_mult {format_number(scalar)} "
                f"--presymp_h {format_number(h)}"
            )
            rows.append(
                row(
                    "tinystories",
                    "causal_symp_pe",
                    seed,
                    f"peh200_h{slug(h)}_s{seed}",
                    flags,
                )
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock_file", type=Path, required=True)
    parser.add_argument("--out_path", type=Path, required=True)
    args = parser.parse_args()
    lock = read_lr_lock(args.lock_file)
    rows = h_rows(lock)
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = "\n".join(rows) + "\n"
    if args.out_path.exists() and args.out_path.read_text() != rendered:
        raise SystemExit(f"refusing to replace a different PE-h specification: {args.out_path}")
    args.out_path.write_text(rendered)
    print(f"wrote {args.out_path} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
