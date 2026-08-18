#!/usr/bin/env python3
"""Run E021 convex softmax-metric gradient versus Hamiltonian acceleration."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from particle_acceleration_benchmark import (  # noqa: E402
    integrate_gradient,
    integrate_hamiltonian,
    sustained_hitting_time,
)

SEEDS = [31001, 31002, 31003, 31004, 31005]
EPSILONS = [0.0, 0.05, 0.20]
DTS = [0.05, 0.025]
THRESHOLDS = [1e-2, 1e-3, 1e-4]
N, D = 16, 2
MU, L = 0.01, 1.0
ALPHA = 2.0 * np.sqrt(MU)
HORIZON = 500.0
FIXED_TIMES = [25.0, 50.0, 100.0, 200.0, 500.0]


def initial_states() -> np.ndarray:
    states = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((N, D))
        x = np.empty_like(z)
        x[:, 0] = 0.5 * z[:, 0]
        x[:, 1] = 0.5 * np.sqrt(MU / L) * z[:, 1]
        states.append(x)
    return np.stack(states)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "artifacts" / "genuine_particle_acceleration"
    out_dir.mkdir(parents=True, exist_ok=True)
    X0 = initial_states()
    B = np.eye(D)
    H = np.diag([MU, L])
    rows: list[dict[str, object]] = []
    saved: dict[str, np.ndarray] = {}
    wall_start = time.perf_counter()

    for epsilon in EPSILONS:
        A = epsilon * np.eye(D)
        for dt in DTS:
            run_start = time.perf_counter()
            trajectories = {
                "gradient": integrate_gradient(X0, A=A, B=B, H=H, dt=dt, horizon=HORIZON),
                "accelerated": integrate_hamiltonian(
                    X0, A=A, B=B, H=H, alpha=ALPHA, dt=dt, horizon=HORIZON
                ),
            }
            elapsed = time.perf_counter() - run_start
            for method, trajectory in trajectories.items():
                ratios = trajectory.objective / trajectory.objective[0][None, :]
                total_ratios = trajectory.total_energy / trajectory.total_energy[0][None, :]
                hits = {
                    threshold: sustained_hitting_time(trajectory.times, ratios, threshold)
                    for threshold in THRESHOLDS
                }
                diffs = np.diff(trajectory.total_energy, axis=0)
                scale = np.maximum(trajectory.total_energy[:-1], 1e-300)
                max_energy_increase = np.max(diffs / scale, axis=0)
                for seed_index, seed in enumerate(SEEDS):
                    row: dict[str, object] = {
                        "epsilon": epsilon,
                        "dt": dt,
                        "seed": seed,
                        "method": method,
                        "initial_objective": trajectory.objective[0, seed_index],
                        "final_objective_ratio": ratios[-1, seed_index],
                        "max_relative_energy_increase": max_energy_increase[seed_index],
                        "wall_seconds_for_batched_pair": elapsed,
                    }
                    for threshold in THRESHOLDS:
                        key = f"hit_{threshold:.0e}".replace("-", "m")
                        row[key] = hits[threshold][seed_index]
                        row[f"rhs_calls_{threshold:.0e}".replace("-", "m")] = (
                            np.inf if not np.isfinite(hits[threshold][seed_index])
                            else int(round(hits[threshold][seed_index] / dt)) * 4
                        )
                    for fixed_time in FIXED_TIMES:
                        index = int(round(fixed_time / dt))
                        row[f"ratio_t{int(fixed_time)}"] = ratios[index, seed_index]
                    rows.append(row)
                if dt == min(DTS):
                    tag = f"eps{epsilon:.2f}_{method}".replace(".", "p")
                    saved[f"times_{tag}"] = trajectory.times
                    saved[f"objective_ratio_{tag}"] = ratios
                    saved[f"total_energy_ratio_{tag}"] = total_ratios

    fieldnames = list(rows[0])
    with (out_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(out_dir / "trajectories.npz", **saved)

    # Strict audit and frozen gate on reference trajectories.
    reference = [row for row in rows if row["dt"] == min(DTS)]
    main_step = [row for row in rows if row["dt"] == max(DTS)]
    summaries = []
    gate = True
    for epsilon in EPSILONS:
        by_method = {
            method: sorted(
                [r for r in reference if r["epsilon"] == epsilon and r["method"] == method],
                key=lambda r: int(r["seed"]),
            )
            for method in ("gradient", "accelerated")
        }
        coarse_by_method = {
            method: sorted(
                [r for r in main_step if r["epsilon"] == epsilon and r["method"] == method],
                key=lambda r: int(r["seed"]),
            )
            for method in ("gradient", "accelerated")
        }
        gf = np.array([float(r["hit_1em03"]) for r in by_method["gradient"]])
        acc = np.array([float(r["hit_1em03"]) for r in by_method["accelerated"]])
        refinement = max(
            abs(float(ref["hit_1em03"]) - float(coarse["hit_1em03"]))
            for method in ("gradient", "accelerated")
            for ref, coarse in zip(by_method[method], coarse_by_method[method])
        )
        wins = int(np.sum(acc < gf))
        reduction = float((gf.mean() - acc.mean()) / gf.mean())
        item = {
            "epsilon": epsilon,
            "gradient_mean_hit_1e-3": float(gf.mean()),
            "accelerated_mean_hit_1e-3": float(acc.mean()),
            "paired_wins": wins,
            "mean_reduction": reduction,
            "max_refinement_difference": float(refinement),
        }
        summaries.append(item)
        if epsilon > 0:
            gate = gate and wins == len(SEEDS) and reduction >= 0.25 and refinement <= 0.25

    summary = {
        "status": "pass" if gate else "fail",
        "success_gate": bool(gate),
        "seeds": SEEDS,
        "epsilons": EPSILONS,
        "dts": DTS,
        "N": N,
        "d": D,
        "mu": MU,
        "L": L,
        "condition_number": L / MU,
        "alpha": ALPHA,
        "horizon": HORIZON,
        "primary_threshold": 1e-3,
        "summaries": summaries,
        "wall_seconds": time.perf_counter() - wall_start,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
