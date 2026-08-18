#!/usr/bin/env python3
"""Audit and summarize the predeclared causal-PE tuning campaign."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shlex
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_pe_tuning_specs import directional_rows, dynamics_rows, optimizer_rows


SEED_SUFFIX = re.compile(r"_s\d+$")
RUN_FIELDS = [
    "run_name",
    "candidate",
    "family",
    "arch",
    "seed",
    "status",
    "reasons",
    "tokens",
    "expected_tokens",
    "final_val",
    "best_val",
    "last3_val",
    "val_auc",
    "gpu_hours",
    "tokens_per_second",
    "peak_memory_mb",
    "trainable_parameters",
    "max_leak_warnings",
    "slurm_exit_code",
    "gpu",
    "source_revision",
    "train_sha256",
    "val_sha256",
    "run_dir",
]


def canonical_name(run_name: str) -> str:
    return SEED_SUFFIX.sub("", run_name)


def family(candidate: str) -> str:
    if candidate.startswith("pe1b_"):
        return "directional_1b"
    for prefix, label in (
        ("pe350_h0", "step_size"),
        ("pe350_clog", "damping_init"),
        ("pe350_scalar", "scalar_lr"),
        ("pe350_muon", "muon_lr"),
        ("pe350_adam", "adam_group_lr"),
    ):
        if candidate.startswith(prefix):
            return label
    if candidate == "pe350_anchor":
        return "anchor"
    if candidate.startswith(("pe350_lnp_", "pe350_h_fixed", "pe350_eta_fixed")):
        return "structural"
    return "other"


TRUE_FLAGS = {"--no_v0_init", "--presymp_mlp_use_attn_vel", "--presymp_mlp_use_p_vel", "--presymp_lookahead", "--sample_final", "--no_mlp", "--eta_learnable", "--lin_noncausal"}


def parse_expected_args(argv: list[str]) -> dict:
    expected = {}
    index = 0
    while index < len(argv):
        flag = argv[index]
        if flag in TRUE_FLAGS:
            expected[flag[2:]] = True
            index += 1
            continue
        if flag == "--no-eta_learnable":
            expected["eta_learnable"] = False
            index += 1
            continue
        if not flag.startswith("--") or index + 1 >= len(argv):
            raise AssertionError(f"cannot parse specification flag sequence: {argv}")
        key = flag[2:].replace("-", "_")
        raw = argv[index + 1]
        try:
            value = int(raw)
        except ValueError:
            try:
                value = float(raw)
            except ValueError:
                value = raw
        expected[key] = value
        index += 2
    return expected


def values_match(actual, expected) -> bool:
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return actual == expected


def parse_spec(row: str) -> dict:
    dataset, arch, config, seed, run_name, flags = row.split("\t")
    argv = shlex.split(flags)
    max_tokens = int(argv[argv.index("--max_tokens") + 1])
    tokens_per_step = 262_144
    expected_tokens = (max_tokens // tokens_per_step) * tokens_per_step
    return {
        "dataset": dataset,
        "arch": arch,
        "config": config,
        "seed": int(seed),
        "run_name": run_name,
        "candidate": canonical_name(run_name),
        "family": family(canonical_name(run_name)),
        "expected_tokens": expected_tokens,
        "expected_args": parse_expected_args(argv),
    }


def expected_runs() -> list[dict]:
    rows = dynamics_rows() + optimizer_rows() + directional_rows()
    return [parse_spec(row) for row in rows]


def read_metrics(path: Path) -> list[dict]:
    rows = []
    if not path.is_file():
        return rows
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            if raw.get("val_loss", "") == "":
                continue
            try:
                row = {
                    "step": int(raw["step"]),
                    "tokens": int(raw["tokens_cum"]),
                    "wall": float(raw["wall_cum_s"]),
                    "val": float(raw["val_loss"]),
                    "leaks": int(float(raw.get("leak_warnings") or 0)),
                }
            except (KeyError, TypeError, ValueError):
                continue
            rows.append(row)
    # Resumed runs can repeat evaluation token counts. Keep the last record.
    dedup = {row["tokens"]: row for row in rows}
    return [dedup[token] for token in sorted(dedup)]


def normalized_auc(rows: list[dict]) -> float:
    if not rows:
        return math.nan
    if len(rows) == 1:
        return rows[0]["val"]
    area = 0.0
    for left, right in zip(rows, rows[1:]):
        area += (right["tokens"] - left["tokens"]) * (left["val"] + right["val"]) / 2
    width = rows[-1]["tokens"] - rows[0]["tokens"]
    return area / width if width > 0 else rows[-1]["val"]


def read_slurm_status(path: Path) -> tuple[str, int | None]:
    if not path.is_file():
        return "missing", None
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        return "missing", None
    final = rows[-1]
    exit_code = final.get("exit_code", "")
    try:
        parsed_exit = int(exit_code) if exit_code != "" else None
    except ValueError:
        parsed_exit = None
    return final.get("state", "missing"), parsed_exit


def finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def source_identity(manifest: dict) -> tuple[str, bool]:
    revision = manifest.get("source_revision") or "NO_GIT"
    fingerprint = manifest.get("source_fingerprint") or {}
    digest = fingerprint.get("sha256", "")
    if digest:
        return f"git:{revision};sha256:{digest}", True
    return f"git:{revision};sha256:MISSING", False


def data_sha256(manifest: dict, split: str) -> str:
    record = manifest.get(f"{split}_data") or {}
    metadata = record.get("manifest") or {}
    return metadata.get("sha256", "")


def load_run(manifest_path: Path, expected: dict) -> dict:
    run_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    args = manifest["args"]
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    metrics = read_metrics(run_dir / "metrics.csv")
    tokens = int(summary.get("tokens", metrics[-1]["tokens"] if metrics else 0))
    final_val = summary.get("final_val", metrics[-1]["val"] if metrics else math.nan)
    wall = float(summary.get("wall_cum_s", metrics[-1]["wall"] if metrics else math.nan))
    last3 = statistics.fmean(row["val"] for row in metrics[-3:]) if metrics else math.nan
    leaks = max((row["leaks"] for row in metrics), default=0)
    slurm_state, exit_code = read_slurm_status(run_dir / "slurm_status.tsv")
    source, source_ok = source_identity(manifest)
    train_sha = data_sha256(manifest, "train")
    val_sha = data_sha256(manifest, "val")
    model_config = manifest.get("model_config") or {}

    reasons = []
    if not summary_path.is_file():
        reasons.append("missing_summary")
    if not metrics:
        reasons.append("missing_validation_rows")
    if tokens != expected["expected_tokens"]:
        reasons.append("token_mismatch")
    if not finite(final_val) or not finite(last3) or not finite(wall):
        reasons.append("nonfinite_metric")
    if leaks != 0:
        reasons.append("leak_warning")
    if slurm_state != "complete" or exit_code != 0:
        reasons.append("slurm_not_complete")
    if not source_ok:
        reasons.append("missing_source_fingerprint")
    if not train_sha or not val_sha:
        reasons.append("missing_data_sha256")
    if (
        args.get("dataset") != expected["dataset"]
        or args.get("arch") != expected["arch"]
        or int(args.get("seed", -1)) != expected["seed"]
        or args.get("config") != expected["config"]
        or any(
            not values_match(args.get(key), value)
            for key, value in expected.get("expected_args", {}).items()
        )
    ):
        reasons.append("spec_mismatch")
    if (
        int(model_config.get("n_layer", -1)) != 8
        or int(model_config.get("n_head", -1)) != 8
        or int(model_config.get("n_embd", -1)) != 512
        or int(model_config.get("block_size", -1)) != 512
        or int(args.get("global_tokens_per_step", -1)) != 262_144
        or int(args.get("eval_batches", -1)) != 160
        or args.get("amp_dtype") != "bfloat16"
        or not str(args.get("device", "")).startswith("cuda")
        or not manifest.get("gpu")
    ):
        reasons.append("protocol_mismatch")

    # The launcher status is the auditable proof that no task-level error was
    # swallowed. Older/manual runs remain visible but cannot qualify.
    status = "valid" if not reasons else "invalid"
    return {
        "run_name": args.get("run_name", expected["run_name"]),
        "candidate": canonical_name(args.get("run_name", expected["run_name"])),
        "family": expected["family"],
        "arch": args.get("arch", expected["arch"]),
        "seed": int(args.get("seed", expected["seed"])),
        "status": status,
        "reasons": ";".join(reasons),
        "tokens": tokens,
        "expected_tokens": expected["expected_tokens"],
        "final_val": float(final_val),
        "best_val": float(summary.get("best_val", math.nan)),
        "last3_val": last3,
        "val_auc": normalized_auc(metrics),
        "gpu_hours": wall / 3600 if finite(wall) else math.nan,
        "tokens_per_second": tokens / wall if finite(wall) and wall > 0 else math.nan,
        "peak_memory_mb": float(summary.get("peak_memory_mb", math.nan)),
        "trainable_parameters": int(
            summary.get("trainable_parameters", manifest.get("trainable_parameters", 0))
        ),
        "max_leak_warnings": leaks,
        "slurm_exit_code": exit_code if exit_code is not None else "",
        "gpu": manifest.get("gpu") or "UNKNOWN",
        "source_revision": source,
        "train_sha256": train_sha,
        "val_sha256": val_sha,
        "run_dir": str(run_dir),
    }


def missing_run(expected: dict) -> dict:
    row = {field: "" for field in RUN_FIELDS}
    row.update(
        run_name=expected["run_name"],
        candidate=expected["candidate"],
        family=expected["family"],
        arch=expected["arch"],
        seed=expected["seed"],
        status="missing",
        reasons="missing_manifest",
        tokens=0,
        expected_tokens=expected["expected_tokens"],
    )
    return row


def discover_runs(root: Path) -> list[dict]:
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
    rows = []
    for item in expected:
        path = manifests.pop(item["run_name"], None)
        rows.append(load_run(path, item) if path else missing_run(item))
    # Preserve unexpected runs in the audit by synthesizing their expectation
    # from their own manifest. They never enter the predeclared aggregates.
    for name, path in sorted(manifests.items()):
        manifest = json.loads(path.read_text())
        args = manifest["args"]
        expected_tokens = int(manifest.get("actual_max_tokens", 0))
        item = {
            "run_name": name,
            "candidate": canonical_name(name),
            "family": "unexpected",
            "arch": args["arch"],
            "seed": int(args["seed"]),
            "dataset": args.get("dataset"),
            "config": args.get("config"),
            "expected_tokens": expected_tokens,
            "expected_args": {},
        }
        row = load_run(path, item)
        row["status"] = "unexpected"
        row["reasons"] = "not_in_predeclared_specs"
        rows.append(row)
    return rows


def paired_350m(runs: list[dict]) -> list[dict]:
    anchors = {
        row["seed"]: row
        for row in runs
        if row["candidate"] == "pe350_anchor"
    }
    pairs = []
    for row in runs:
        if not row["candidate"].startswith("pe350_") or row["candidate"] == "pe350_anchor":
            continue
        anchor = anchors.get(row["seed"])
        provenance_match = bool(
            anchor
            and all(
                row[field] == anchor[field]
                for field in ("source_revision", "train_sha256", "val_sha256", "gpu")
            )
        )
        valid = bool(
            anchor
            and anchor["status"] == "valid"
            and row["status"] == "valid"
            and provenance_match
        )
        pairs.append(
            {
                "candidate": row["candidate"],
                "family": row["family"],
                "seed": row["seed"],
                "candidate_status": row["status"],
                "anchor_status": anchor["status"] if anchor else "missing",
                "provenance_match": int(provenance_match),
                "delta_final_val": row["final_val"] - anchor["final_val"] if valid else "",
                "delta_last3_val": row["last3_val"] - anchor["last3_val"] if valid else "",
                "candidate_final_val": row["final_val"],
                "anchor_final_val": anchor["final_val"] if anchor else "",
                "valid_pair": int(valid),
            }
        )
    return pairs


def aggregate_pairs(pairs: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in pairs:
        groups[row["candidate"]].append(row)
    aggregate = []
    for candidate, rows in groups.items():
        valid = [row for row in rows if row["valid_pair"]]
        deltas = [float(row["delta_final_val"]) for row in valid]
        last3 = [float(row["delta_last3_val"]) for row in valid]
        mean_delta = statistics.fmean(deltas) if deltas else math.nan
        std_delta = statistics.stdev(deltas) if len(deltas) > 1 else 0.0 if deltas else math.nan
        mean_last3 = statistics.fmean(last3) if last3 else math.nan
        wins = sum(delta < 0 for delta in deltas)
        qualifies = (
            len(rows) == 3
            and len(valid) == 3
            and wins >= 2
            and mean_delta <= -0.005
        )
        aggregate.append(
            {
                "candidate": candidate,
                "family": rows[0]["family"],
                "n_expected": len(rows),
                "n_valid": len(valid),
                "wins_vs_anchor": wins,
                "mean_delta_final_val": mean_delta,
                "std_delta_final_val": std_delta,
                "mean_delta_last3_val": mean_last3,
                "qualifies_for_confirmation": int(qualifies),
            }
        )
    return sorted(
        aggregate,
        key=lambda row: (
            -row["qualifies_for_confirmation"],
            row["mean_delta_final_val"]
            if finite(row["mean_delta_final_val"])
            else math.inf,
            row["candidate"],
        ),
    )


def directional_1b(runs: list[dict]) -> list[dict]:
    rows = [row for row in runs if row["candidate"].startswith("pe1b_")]
    baseline = next((row for row in rows if row["candidate"] == "pe1b_baseline"), None)
    out = []
    for row in rows:
        valid_pair = bool(
            baseline
            and baseline["status"] == "valid"
            and row["status"] == "valid"
        )
        out.append(
            {
                "candidate": row["candidate"],
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
    return out


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else [])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(root: Path):
    runs = discover_runs(root)
    pairs = paired_350m(runs)
    aggregate = aggregate_pairs(pairs)
    directional = directional_1b(runs)
    return runs, pairs, aggregate, directional


def self_test() -> None:
    rows = []
    for name, deltas in (
        ("good", (-0.007, -0.006, -0.005)),
        ("too_small", (-0.006, -0.004, -0.001)),
        ("incomplete", (-0.010, -0.010)),
    ):
        for seed, delta in enumerate(deltas):
            rows.append(
                {
                    "candidate": name,
                    "family": "test",
                    "seed": seed,
                    "valid_pair": 1,
                    "delta_final_val": delta,
                    "delta_last3_val": delta,
                }
            )
    result = {row["candidate"]: row for row in aggregate_pairs(rows)}
    assert result["good"]["qualifies_for_confirmation"] == 1
    assert result["too_small"]["qualifies_for_confirmation"] == 0
    assert result["incomplete"]["qualifies_for_confirmation"] == 0
    assert canonical_name("pe350_h0p2_s20272") == "pe350_h0p2"
    print("PASS: qualification gate and seed canonicalization")


def print_compact(runs, aggregate, directional) -> None:
    valid = sum(row["status"] == "valid" for row in runs)
    missing = sum(row["status"] == "missing" for row in runs)
    invalid = sum(row["status"] == "invalid" for row in runs)
    print(f"expected={len(expected_runs())} valid={valid} missing={missing} invalid={invalid}")
    print("\n350M paired ranking (negative is better than the same-seed PE anchor):")
    print(f"{'CANDIDATE':24s} {'N':>3s} {'WINS':>4s} {'MEAN_DNLL':>11s} {'QUALIFY':>8s}")
    for row in aggregate:
        delta = row["mean_delta_final_val"]
        rendered = f"{delta:+.6f}" if finite(delta) else "NA"
        print(
            f"{row['candidate']:24s} {row['n_valid']:3d} "
            f"{row['wins_vs_anchor']:4d} {rendered:>11s} "
            f"{'yes' if row['qualifies_for_confirmation'] else 'no':>8s}"
        )
    print("\n1B directional check (not a multi-seed confirmation):")
    print(f"{'CANDIDATE':24s} {'STATUS':>9s} {'FINAL_NLL':>11s} {'D_BASE':>11s}")
    for row in directional:
        final = f"{row['final_val']:.6f}" if finite(row["final_val"]) else "NA"
        delta = row["delta_vs_baseline"]
        rendered = f"{float(delta):+.6f}" if delta != "" and finite(delta) else "NA"
        print(f"{row['candidate']:24s} {row['status']:>9s} {final:>11s} {rendered:>11s}")


def main() -> None:
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
    runs, pairs, aggregate, directional = analyze(args.root)
    write_csv(out_dir / "runs.csv", runs, RUN_FIELDS)
    write_csv(out_dir / "paired_350m.csv", pairs)
    write_csv(out_dir / "aggregate_350m.csv", aggregate)
    write_csv(out_dir / "directional_1b.csv", directional)
    print_compact(runs, aggregate, directional)
    print(f"\nwrote analysis tables to {out_dir}")
    if any(row["status"] != "valid" for row in runs[: len(expected_runs())]):
        print("WARNING: missing/invalid runs are retained and cannot qualify.")


if __name__ == "__main__":
    main()
