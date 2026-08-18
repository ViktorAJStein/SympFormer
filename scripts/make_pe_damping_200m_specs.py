#!/usr/bin/env python3
"""Generate the lock-dependent 200M PE damping campaign stages."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from make_headline_lr_200m_specs import SCALAR_PEAK_LR, format_number
from make_pe_tuning_specs import PE_FLAGS, row, slug


CONFIG = "configs/core_50m.json"
TOKENS_200M = 200_000_000
ACTUAL_TOKENS_200M = 199_753_728
REPLACE_THRESHOLD = -0.005

STAGES = ("d0", "d1", "d2")
STAGE_SEEDS = {
    "d0": (20280, 20281),
    "d1": (20282, 20283),
    "d2": (20284, 20285),
}
D0_VALUES = ("learned", "fixed")
D1_VALUES = (0.5, 1.0, 2.0, 3.0)
D2_VALUES = (0.0001, 0.03, 0.10, 0.30)
INCUMBENTS = {"d0": "learned", "d1": 3.0, "d2": 0.0001}
INPUT_CAMPAIGNS = {
    "d0": "pe_h_200m",
    "d1": "pe_damping_mode_200m",
    "d2": "pe_damping_log_200m",
}
OUTPUT_CAMPAIGNS = {
    "d0": "pe_damping_mode_200m",
    "d1": "pe_damping_log_200m",
    "d2": "pe_damping_200m",
}


def read_stage_lock(path: Path, stage: str) -> dict:
    if stage not in STAGES:
        raise SystemExit(f"unknown damping stage: {stage}")
    if not path.is_file():
        raise SystemExit(f"missing upstream lock: {path}")
    lock = json.loads(path.read_text())
    if (
        lock.get("format_version") != 1
        or lock.get("campaign") != INPUT_CAMPAIGNS[stage]
        or lock.get("selection_complete") is not True
    ):
        raise SystemExit(f"upstream lock is incomplete or wrong for {stage}: {path}")
    for field in ("source_revision", "train_sha256", "val_sha256", "gpu"):
        if not isinstance(lock.get(field), str) or not lock[field]:
            raise SystemExit(f"upstream lock is missing provenance field: {field}")
    if int(lock.get("actual_tokens_per_run", -1)) != ACTUAL_TOKENS_200M:
        raise SystemExit("upstream lock has the wrong per-run token count")
    peak = float(lock.get("peak_lr", math.nan))
    scalar = float(lock.get("scalar_lr_mult", math.nan))
    h_init = float(lock.get("presymp_h", math.nan))
    if not math.isclose(peak * scalar, SCALAR_PEAK_LR, rel_tol=0, abs_tol=1e-12):
        raise SystemExit("upstream lock does not preserve scalar peak LR 0.003")
    if not all(math.isfinite(value) and value > 0 for value in (peak, scalar, h_init)):
        raise SystemExit("upstream PE optimizer or step lock is invalid")
    if stage in ("d1", "d2") and not isinstance(lock.get("damping_learnable"), bool):
        raise SystemExit("upstream damping-mode decision is missing")
    if stage == "d2":
        eta_log = float(lock.get("eta_log", math.nan))
        if not math.isfinite(eta_log) or eta_log <= 0:
            raise SystemExit("upstream log-damping decision is missing")
    return lock


def shared_flags(lock: dict) -> str:
    return (
        f"{PE_FLAGS} --max_tokens {TOKENS_200M} "
        f"--peak_lr {format_number(float(lock['peak_lr']))} "
        f"--scalar_lr_mult {format_number(float(lock['scalar_lr_mult']))} "
        f"--presymp_h {format_number(float(lock['presymp_h']))} "
        "--eta_mode loglin"
    )


def damping_flags(learnable: bool, eta_log: float, eta_lin: float) -> str:
    if learnable:
        return (
            f"--eta_learnable --eta_log_init {format_number(eta_log)} "
            f"--eta_lin_init {format_number(eta_lin)}"
        )
    return (
        f"--no-eta_learnable --eta_log_coef {format_number(eta_log)} "
        f"--eta_lin_coef {format_number(eta_lin)}"
    )


def stage_values(stage: str):
    return {"d0": D0_VALUES, "d1": D1_VALUES, "d2": D2_VALUES}[stage]


def semantic_setting(stage: str, value, lock: dict) -> tuple[bool, float, float]:
    if stage == "d0":
        return value == "learned", 3.0, 0.0001
    learnable = bool(lock["damping_learnable"])
    if stage == "d1":
        return learnable, float(value), 0.0001
    return learnable, float(lock["eta_log"]), float(value)


def value_tag(stage: str, value) -> str:
    if stage == "d0":
        return str(value)
    prefix = "clog" if stage == "d1" else "clin"
    return f"{prefix}{slug(float(value))}"


def stage_rows(stage: str, lock: dict) -> list[str]:
    values = stage_values(stage)
    rows = []
    for seed in STAGE_SEEDS[stage]:
        for value in values:
            learnable, eta_log, eta_lin = semantic_setting(stage, value, lock)
            flags = f"{shared_flags(lock)} {damping_flags(learnable, eta_log, eta_lin)}"
            rows.append(
                row(
                    "tinystories",
                    "causal_symp_pe",
                    seed,
                    f"pe{stage}_200_{value_tag(stage, value)}_s{seed}",
                    flags,
                )
            )
    return rows


def write_spec(path: Path, rows: list[str]) -> None:
    rendered = "\n".join(rows) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != rendered:
        raise SystemExit(f"refusing to replace a different damping specification: {path}")
    path.write_text(rendered)
    print(f"wrote {path} ({len(rows)} runs)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--lock_file", type=Path, required=True)
    parser.add_argument("--out_path", type=Path, required=True)
    args = parser.parse_args()
    lock = read_stage_lock(args.lock_file, args.stage)
    write_spec(args.out_path, stage_rows(args.stage, lock))


if __name__ == "__main__":
    main()
