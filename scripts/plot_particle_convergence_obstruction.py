#!/usr/bin/env python3
"""Plot unbounded attention energies and one-particle escape trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("images/e020_particle_convergence_obstruction"))
    args = parser.parse_args()

    x = np.linspace(0.0, 3.0, 500)
    minus_f_soft = 0.5 * np.exp(x**2)
    minus_f_linear = 0.5 * x**2

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    axes[0].semilogy(x, minus_f_soft, label=r"softmax: $-F=\frac{1}{2}e^{x^2}$")
    axes[0].semilogy(x[1:], minus_f_linear[1:], label=r"linear: $-F=\frac{1}{2}x^2$")
    axes[0].set_xlabel(r"particle magnitude $|x|$")
    axes[0].set_ylabel(r"$-F(\delta_x)$ (log scale)")
    axes[0].set_title("Both objectives decrease without bound")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    t_soft = np.linspace(0.0, 2.0, 400)
    x_soft = 0.2 * np.exp(t_soft)
    t_linear = np.linspace(0.0, 1.98, 400)
    x_linear = 0.5 / np.sqrt(1.0 - 0.5 * t_linear)
    axes[1].plot(t_soft, x_soft, label=r"softmax: $x'=x$")
    axes[1].plot(t_linear, x_linear, label=r"linear: $x'=x^3$")
    axes[1].axvline(2.0, color="tab:red", linestyle="--", linewidth=1, label="linear blow-up")
    axes[1].set_xlabel("continuous time")
    axes[1].set_ylabel(r"particle magnitude $x(t)$")
    axes[1].set_title("Energy descent is particle escape")
    axes[1].set_ylim(0.0, 5.2)
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=180, bbox_inches="tight")
    print(f"wrote {args.out.with_suffix('.pdf')} and {args.out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
