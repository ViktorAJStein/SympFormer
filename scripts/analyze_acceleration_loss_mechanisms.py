#!/usr/bin/env python3
"""Quantify short-horizon, damping/LN, and causal-curl acceleration obstructions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def last_metric(path: Path) -> dict[str, str]:
    rows = list(csv.DictReader(path.open()))
    validation = [row for row in rows if row.get("val_loss")]
    if not validation:
        raise ValueError(f"no validation row in {path}")
    return validation[-1]


def campaign_summary(root: Path) -> dict:
    rows = [last_metric(path) for path in sorted(root.glob("*_metrics.csv"))]
    if not rows:
        raise ValueError(f"no metrics files below {root}")
    durations, sigmas, hs, hys = [], [], [], []
    for row in rows:
        t0, t1 = float(row["sched_t_start"]), float(row["sched_t_end"])
        c_log, c_lin = float(row["c_log_mean"]), float(row["c_lin_mean"])
        durations.append(t1 - t0)
        sigmas.append(math.exp(-(c_log * math.log(t1 / t0) + c_lin * (t1 - t0))))
        hs.append(float(row["h_mean"]))
        hys.append(float(row["hY_mean"]))
    return {
        "n": len(rows),
        "duration_mean": float(np.mean(durations)),
        "duration_range": [float(np.min(durations)), float(np.max(durations))],
        "h_mean": float(np.mean(hs)),
        "hY_mean": float(np.mean(hys)),
        "nominal_cumulative_damping_mean": float(np.mean(sigmas)),
        "nominal_cumulative_damping_range": [float(np.min(sigmas)), float(np.max(sigmas))],
    }


def causal_curl_witness() -> dict:
    # Two scalar particles with standard prefix-softmax value averaging.
    x = torch.tensor([0.4, -0.7], dtype=torch.float64, requires_grad=True)
    def field(q: torch.Tensor) -> torch.Tensor:
        g0 = q[0]
        scores = q[1] * q
        weights = torch.softmax(scores, dim=0)
        g1 = torch.dot(weights, q)
        return torch.stack((g0, g1))
    jac = torch.autograd.functional.jacobian(field, x)
    asym = jac - jac.T
    return {
        "jacobian": jac.detach().numpy().tolist(),
        "future_derivative_dg0_dx1": float(jac[0, 1]),
        "past_derivative_dg1_dx0": float(jac[1, 0]),
        "max_jacobian_asymmetry": float(asym.abs().max()),
    }


def layernorm_scale_witness(scale: float) -> dict:
    generator = torch.Generator().manual_seed(20260806)
    p = torch.randn(128, 64, generator=generator, dtype=torch.float64)
    a = F.layer_norm(p, (64,), eps=1e-5)
    b = F.layer_norm(scale * p, (64,), eps=1e-5)
    relative = (a - b).norm() / a.norm()
    raw_relative = (p - scale * p).norm() / p.norm()
    return {
        "scale": scale,
        "raw_relative_change": float(raw_relative),
        "post_layernorm_relative_change": float(relative),
        "fraction_removed_by_layernorm": float(1.0 - relative / raw_relative),
    }


def positive_control_summary(npz_path: Path, duration: float) -> dict:
    data = np.load(npz_path)
    tag = "eps0p20"
    times = data[f"times_{tag}_gradient"]
    gf = data[f"objective_ratio_{tag}_gradient"].mean(axis=1)
    acc = data[f"objective_ratio_{tag}_accelerated"].mean(axis=1)
    crossover_indices = np.flatnonzero(acc < gf)
    crossover = float(times[crossover_indices[0]]) if crossover_indices.size else math.inf
    gf_at = float(np.interp(duration, times, gf))
    acc_at = float(np.interp(duration, times, acc))
    return {
        "first_mean_crossover": crossover,
        "comparison_duration": duration,
        "gradient_ratio_at_duration": gf_at,
        "accelerated_ratio_at_duration": acc_at,
        "accelerated_minus_gradient_at_duration": acc_at - gf_at,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/acceleration_loss_mechanisms"))
    parser.add_argument(
        "--positive_control",
        type=Path,
        default=Path("artifacts/genuine_particle_acceleration/trajectories.npz"),
    )
    args = parser.parse_args()
    h001 = campaign_summary(args.root / "h001")
    h004 = campaign_summary(args.root / "h004")
    duration = 0.5 * (h001["duration_mean"] + h004["duration_mean"])
    # Use the geometric-mean nominal per-layer scale for an eight-layer witness.
    cumulative = 0.5 * (
        h001["nominal_cumulative_damping_mean"] + h004["nominal_cumulative_damping_mean"]
    )
    per_layer_scale = cumulative ** (1.0 / 8.0)
    result = {
        "h001": h001,
        "h004": h004,
        "positive_control_short_horizon": positive_control_summary(args.positive_control, duration),
        "layernorm_scale_witness": layernorm_scale_witness(per_layer_scale),
        "causal_curl_witness": causal_curl_witness(),
    }
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
