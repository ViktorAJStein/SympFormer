#!/usr/bin/env python3
"""Strict audit and pruning analysis for the single-seed 100M PE screen."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from analyze_pe_tuning import (
    RUN_FIELDS,
    canonical_name,
    finite,
    load_run,
    missing_run,
    parse_spec,
    write_csv,
)
from make_pe_100m_specs import control_rows, dynamics_rows, optimizer_rows


ADVANCE_THRESHOLD = -0.010
MAX_ADVANCING_CANDIDATES = 3


def screen_family(candidate):
    if candidate == "pe100_anchor":
        return "anchor"
    if candidate.startswith("pe100_h0"):
        return "step_size"
    if candidate.startswith("pe100_clog"):
        return "damping_init"
    if candidate.startswith("pe100_scalar"):
        return "scalar_lr"
    if candidate.startswith("pe100_muon"):
        return "muon_lr"
    if candidate.startswith("pe100_adam"):
        return "adam_group_lr"
    if candidate.startswith(("pe100_lnp_", "pe100_h_fixed", "pe100_eta_fixed")):
        return "structural"
    if candidate in {"pe100_baseline", "pe100_yurii"}:
        return "controls"
    return "unexpected"


def expected_runs():
    expected = []
    for raw in dynamics_rows() + optimizer_rows() + control_rows():
        item = parse_spec(raw)
        item["family"] = screen_family(item["candidate"])
        expected.append(item)
    return expected


def discover_runs(root):
    expected = expected_runs()
    manifests = {}
    for path in sorted(root.rglob("run_manifest.json")):
        try:
            manifest = json.loads(path.read_text())
            name = manifest["args"]["run_name"]
        except (KeyError, json.JSONDecodeError):
            continue
        if name in manifests:
            raise SystemExit(f"Duplicate run_name below {root}: {name}")
        manifests[name] = path

    runs = []
    for item in expected:
        path = manifests.pop(item["run_name"], None)
        runs.append(load_run(path, item) if path else missing_run(item))

    for name, path in sorted(manifests.items()):
        manifest = json.loads(path.read_text())
        args = manifest["args"]
        item = {
            "run_name": name,
            "candidate": canonical_name(name),
            "family": "unexpected",
            "arch": args["arch"],
            "seed": int(args["seed"]),
            "dataset": args.get("dataset"),
            "config": args.get("config"),
            "expected_tokens": int(manifest.get("actual_max_tokens", 0)),
            "expected_args": {},
        }
        row = load_run(path, item)
        row["status"] = "unexpected"
        row["reasons"] = "not_in_predeclared_specs"
        runs.append(row)
    return runs


def provenance_match(left, right):
    return bool(
        left
        and right
        and all(
            left[field] == right[field]
            for field in ("source_revision", "train_sha256", "val_sha256", "gpu")
        )
    )


def paired_screen(runs):
    anchor = next(
        (row for row in runs if row["candidate"] == "pe100_anchor"),
        None,
    )
    pairs = []
    for row in runs:
        if (
            not row["candidate"].startswith("pe100_")
            or row["candidate"] in {
                "pe100_anchor",
                "pe100_baseline",
                "pe100_yurii",
            }
        ):
            continue
        same_provenance = provenance_match(row, anchor)
        valid = bool(
            anchor
            and anchor["status"] == "valid"
            and row["status"] == "valid"
            and same_provenance
        )
        delta = row["final_val"] - anchor["final_val"] if valid else ""
        pairs.append(
            {
                "candidate": row["candidate"],
                "family": row["family"],
                "seed": row["seed"],
                "candidate_status": row["status"],
                "anchor_status": anchor["status"] if anchor else "missing",
                "provenance_match": int(same_provenance),
                "candidate_final_val": row["final_val"],
                "anchor_final_val": anchor["final_val"] if anchor else "",
                "delta_final_val": delta,
                "delta_last3_val": (
                    row["last3_val"] - anchor["last3_val"] if valid else ""
                ),
                "passes_effect_threshold": int(
                    valid and float(delta) <= ADVANCE_THRESHOLD
                ),
                "advance_to_200m": 0,
                "valid_pair": int(valid),
            }
        )

    eligible = sorted(
        (
            row
            for row in pairs
            if row["valid_pair"] and row["passes_effect_threshold"]
        ),
        key=lambda row: (float(row["delta_final_val"]), row["candidate"]),
    )
    selected = {
        row["candidate"]
        for row in eligible[:MAX_ADVANCING_CANDIDATES]
    }
    for row in pairs:
        row["advance_to_200m"] = int(row["candidate"] in selected)
    return sorted(
        pairs,
        key=lambda row: (
            float(row["delta_final_val"])
            if row["valid_pair"]
            else math.inf,
            row["candidate"],
        ),
    )


def controls_table(runs):
    candidates = {
        row["candidate"]: row
        for row in runs
        if row["candidate"] in {
            "pe100_baseline",
            "pe100_yurii",
            "pe100_anchor",
        }
    }
    baseline = candidates.get("pe100_baseline")
    output = []
    for name in ("pe100_baseline", "pe100_yurii", "pe100_anchor"):
        row = candidates.get(name)
        if row is None:
            continue
        valid_pair = bool(
            baseline
            and baseline["status"] == "valid"
            and row["status"] == "valid"
            and provenance_match(row, baseline)
        )
        output.append(
            {
                "candidate": name,
                "arch": row["arch"],
                "seed": row["seed"],
                "status": row["status"],
                "tokens": row["tokens"],
                "final_val": row["final_val"],
                "last3_val": row["last3_val"],
                "delta_vs_baseline": (
                    row["final_val"] - baseline["final_val"] if valid_pair else ""
                ),
                "gpu_hours": row["gpu_hours"],
                "tokens_per_second": row["tokens_per_second"],
                "peak_memory_mb": row["peak_memory_mb"],
            }
        )
    return output


def analyze(root):
    runs = discover_runs(root)
    pairs = paired_screen(runs)
    controls = controls_table(runs)
    return runs, pairs, controls


def print_compact(runs, pairs, controls):
    expected = runs[: len(expected_runs())]
    valid = sum(row["status"] == "valid" for row in expected)
    missing = sum(row["status"] == "missing" for row in expected)
    invalid = sum(row["status"] == "invalid" for row in expected)
    print(f"expected=21 valid={valid} missing={missing} invalid={invalid}")
    print("\n100M PE pruning screen (negative is better than the PE anchor):")
    print(f"{'CANDIDATE':23s} {'STATUS':>9s} {'D_FINAL':>10s} {'ADVANCE':>8s}")
    for row in pairs:
        delta = row["delta_final_val"]
        rendered = f"{float(delta):+.6f}" if delta != "" and finite(delta) else "NA"
        print(
            f"{row['candidate']:23s} {row['candidate_status']:>9s} "
            f"{rendered:>10s} "
            f"{'yes' if row['advance_to_200m'] else 'no':>8s}"
        )
    print("\n100M controls:")
    print(f"{'METHOD':23s} {'STATUS':>9s} {'FINAL_NLL':>10s} {'D_BASE':>10s}")
    for row in controls:
        final = f"{float(row['final_val']):.6f}" if finite(row["final_val"]) else "NA"
        delta = row["delta_vs_baseline"]
        rendered = f"{float(delta):+.6f}" if delta != "" and finite(delta) else "NA"
        print(
            f"{row['candidate']:23s} {row['status']:>9s} "
            f"{final:>10s} {rendered:>10s}"
        )
    print(
        "\nPruning only: at most three candidates with delta <= -0.010 "
        "advance to fresh-seed 200M confirmation."
    )


def self_test():
    anchor = {
        "candidate": "pe100_anchor",
        "status": "valid",
        "final_val": 1.4,
        "last3_val": 1.41,
        "source_revision": "source",
        "train_sha256": "train",
        "val_sha256": "val",
        "gpu": "gpu",
    }
    runs = [anchor]
    for index, delta in enumerate((-0.020, -0.015, -0.011, -0.010, -0.009)):
        runs.append(
            {
                **anchor,
                "candidate": f"pe100_test{index}",
                "family": "test",
                "seed": 20272,
                "final_val": anchor["final_val"] + delta,
                "last3_val": anchor["last3_val"] + delta,
            }
        )
    pairs = paired_screen(runs)
    selected = [row for row in pairs if row["advance_to_200m"]]
    assert len(selected) == 3
    assert {row["candidate"] for row in selected} == {
        "pe100_test0",
        "pe100_test1",
        "pe100_test2",
    }
    print("PASS: threshold and maximum-three pruning rule")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path)
    parser.add_argument("--out_dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.root is None:
        parser.error("root is required unless --self-test is used")
    out_dir = args.out_dir or args.root / "analysis"
    runs, pairs, controls = analyze(args.root)
    write_csv(out_dir / "runs.csv", runs, RUN_FIELDS)
    write_csv(out_dir / "paired_vs_pe_anchor.csv", pairs)
    write_csv(out_dir / "controls.csv", controls)
    print_compact(runs, pairs, controls)
    print(f"\nwrote analysis tables to {out_dir}")
    if any(
        row["status"] != "valid"
        for row in runs[: len(expected_runs())]
    ):
        print("WARNING: missing/invalid runs are retained and cannot advance.")


if __name__ == "__main__":
    main()
