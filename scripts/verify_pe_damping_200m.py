#!/usr/bin/env python3
"""Synthetic end-to-end verification of D0, D1, and D2 damping stages."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

from analyze_pe_damping_200m import analyze, lock_payload, write_lock
from analyze_pe_tuning import parse_spec
from make_pe_damping_200m_specs import STAGES, stage_rows
from verify_campaign_lock_context import source_identity, verify_context
from verify_headline_tuning_200m import fake_manifest_item, write_fake_run


ROOT = Path(__file__).resolve().parents[1]


def initial_lock() -> dict:
    return {
        "format_version": 1,
        "campaign": "pe_h_200m",
        "selection_complete": True,
        "actual_tokens_per_run": 199_753_728,
        "peak_lr": 0.0025,
        "scalar_lr_mult": 1.2,
        "presymp_h": 0.1,
        "source_revision": "git:synthetic;sha256:source-hash",
        "train_sha256": "train-hash",
        "val_sha256": "val-hash",
        "gpu": "NVIDIA RTX A6000",
    }


def effect(stage: str, setting) -> float:
    tables = {
        "d0": {"learned": 0.0, "fixed": -0.012},
        "d1": {0.5: -0.002, 1.0: -0.012, 2.0: -0.004, 3.0: 0.0},
        "d2": {0.0001: 0.0, 0.03: -0.002, 0.1: -0.012, 0.3: 0.002},
    }
    return tables[stage][setting]


def expected_selected(stage: str):
    return {"d0": "fixed", "d1": 1.0, "d2": 0.1}[stage]


def run_stage(root: Path, stage: str, upstream: dict) -> tuple[dict, Path, Path]:
    stage_root = root / f"{stage}_runs"
    lock_path = root / f"{stage}_upstream.json"
    lock_path.write_text(json.dumps(upstream))
    rows = stage_rows(stage, upstream)
    spec_path = root / f"{stage}.tsv"
    spec_path.write_text("\n".join(rows) + "\n")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "verify_pe_damping_200m_specs.py"),
            "--stage",
            stage,
            "--lock_file",
            str(lock_path),
            "--spec",
            str(spec_path),
        ],
        check=True,
    )
    for raw in rows:
        parsed = parse_spec(raw)
        name = parsed["run_name"]
        if stage == "d0":
            setting = "fixed" if "_fixed_" in name else "learned"
        elif stage == "d1":
            setting = float(parsed["expected_args"].get("eta_log_init", parsed["expected_args"].get("eta_log_coef")))
        else:
            setting = float(parsed["expected_args"].get("eta_lin_init", parsed["expected_args"].get("eta_lin_coef")))
        item = fake_manifest_item(raw, f"pe_damping_{stage}", setting=setting)
        final = 1.52 + effect(stage, setting) + (item["seed"] % 2) * 0.004
        write_fake_run(stage_root, item, final)
    runs, pairs, aggregate, complete = analyze(stage_root, stage, upstream)
    assert complete
    assert len(runs) == len(rows)
    assert len(pairs) == len(rows) - 2
    chosen = next(row for row in aggregate if row["selected"])
    expected = expected_selected(stage)
    if isinstance(expected, str):
        assert chosen["setting"] == expected
    else:
        assert math.isclose(float(chosen["setting"]), expected)
    payload = lock_payload(stage, upstream, aggregate, runs, complete)
    assert payload is not None
    fallback_rows = [{**row, "selected": 0} for row in aggregate]
    fallback = lock_payload(stage, upstream, fallback_rows, runs, complete)
    assert fallback is not None
    if stage == "d0":
        assert fallback["damping_learnable"] is True
    elif stage == "d1":
        assert math.isclose(fallback["eta_log"], 3.0)
    else:
        assert math.isclose(fallback["eta_lin"], 0.0001)
    output_lock = root / f"{stage}_lock.json"
    write_lock(output_lock, payload)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "plot_pe_damping_200m.py"),
            str(stage_root),
            "--stage",
            stage,
            "--lock_file",
            str(lock_path),
            "--out_prefix",
            str(root / "figures" / stage),
        ],
        check=True,
    )
    return payload, stage_root, output_lock


def main() -> None:
    launcher = (ROOT / "jobs" / "headline_tuning_200m.sbatch").read_text()
    submit = (ROOT / "jobs" / "submit_pe_damping_200m.sh").read_text()
    for directive in (
        "#SBATCH --cpus-per-task=4",
        "#SBATCH --partition=gpusmpshort",
        "#SBATCH --mem=16G",
        "#SBATCH --time=01:30:00",
        "#SBATCH --gres=gpu:rtx_a6000:1",
    ):
        assert directive in launcher
    for required in (
        "d0) task_last=3",
        "d1|d2) task_last=7",
        "verify_campaign_lock_context.py",
        "verify_pe_damping_200m_specs.py",
    ):
        assert required in submit

    with tempfile.TemporaryDirectory(prefix="pe-damping-") as directory:
        root = Path(directory)
        current_lock = initial_lock()
        produced = {}
        stage_roots = {}
        for stage in STAGES:
            current_lock, stage_root, lock_path = run_stage(root, stage, current_lock)
            produced[stage] = lock_path
            stage_roots[stage] = stage_root

        assert current_lock["damping_learnable"] is False
        assert math.isclose(current_lock["eta_log"], 1.0)
        assert math.isclose(current_lock["eta_lin"], 0.1)
        assert math.isclose(current_lock["peak_lr"], 0.0025)
        assert math.isclose(current_lock["presymp_h"], 0.1)

        # A provenance mismatch must make a complete stage ineligible.
        manifest = next(stage_roots["d2"].rglob("run_manifest.json"))
        changed = json.loads(manifest.read_text())
        changed["gpu"] = "different GPU"
        manifest.write_text(json.dumps(changed))
        d1_lock = json.loads(produced["d1"].read_text())
        assert not analyze(stage_roots["d2"], "d2", d1_lock)[3]

        # The pre-submission context check must reject changed data.
        data = root / "data"
        data.mkdir()
        context_lock = initial_lock()
        context_lock["source_revision"] = source_identity("configs/core_50m.json")
        context_lock["train_sha256"] = "a" * 64
        context_lock["val_sha256"] = "b" * 64
        context_path = root / "context.json"
        context_path.write_text(json.dumps(context_lock))
        for split, digest in (("train", "a" * 64), ("val", "b" * 64)):
            (data / f"tinystories_{split}.bin.manifest.json").write_text(
                json.dumps({"sha256": digest})
            )
        verify_context(context_path, data, "configs/core_50m.json")
        (data / "tinystories_val.bin.manifest.json").write_text(
            json.dumps({"sha256": "c" * 64})
        )
        try:
            verify_context(context_path, data, "configs/core_50m.json")
        except SystemExit:
            pass
        else:
            raise AssertionError("context check accepted a changed dataset")

        figures = [
            root / "figures" / f"{stage}_{kind}.{suffix}"
            for stage in STAGES
            for kind in ("curves", "paired")
            for suffix in ("png", "pdf")
        ]
        assert all(path.is_file() and path.stat().st_size > 0 for path in figures)

    print(
        "PASS: D0/D1/D2 specs, one-factor locks, paired gates, provenance rejection, "
        "source/data preflight, resources, and twelve figure files"
    )


if __name__ == "__main__":
    main()
