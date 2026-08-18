#!/usr/bin/env python3
"""Verify variable-step AB2 order and causal PE scalar gradients."""

import math

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from model import (
    CausalPrefixPresymplecticEulerAttention,
    ModelConfig,
    variable_step_ab2,
)


def integrate_exp(n):
    # Alternating nonuniform positive steps, normalized to end exactly at t=1.
    raw = torch.tensor([1.0 if k % 2 == 0 else 1.7 for k in range(n)], dtype=torch.float64)
    hs = raw / raw.sum()
    y_prev = torch.tensor(1.0, dtype=torch.float64)
    f_prev = y_prev.clone()
    y = y_prev + hs[0] * f_prev  # Euler startup
    for k in range(1, n):
        f = y  # y'=y
        f_eff = variable_step_ab2(f, f_prev, hs[k], hs[k - 1])
        y_new = y + hs[k] * f_eff
        y_prev, y, f_prev = y, y_new, f
    return abs(float(y) - math.e)


def verify_ab2_order():
    ns = [20, 40, 80, 160]
    errors = [integrate_exp(n) for n in ns]
    orders = [math.log(errors[i] / errors[i + 1], 2.0) for i in range(len(errors) - 1)]
    print("AB2 errors:", dict(zip(ns, errors)))
    print("AB2 observed orders:", orders)
    if min(orders[-2:]) < 1.8:
        raise AssertionError(f"Variable-step AB2 did not recover second order: {orders}")


def verify_causal_pe_gradients():
    torch.manual_seed(7)
    cfg = ModelConfig(vocab_size=32, block_size=5, n_layer=1, n_head=1, n_embd=4)
    layer = CausalPrefixPresymplecticEulerAttention(
        cfg,
        h=0.1,
        eta_learnable=True,
        eta_mode="loglin",
        eta_log_init=3.0,
        eta_lin_init=1e-3,
        presymp_lnp="none",
        lookahead=False,
    )
    X = torch.randn(2, 5, 4, requires_grad=True)
    P = torch.randn(2, 5, 4, requires_grad=True)
    X1, P1 = layer.step(X, P, 1.0)
    loss = X1.square().mean() + P1.square().mean()
    loss.backward()
    named = dict(layer.named_parameters())
    required = ["theta_hX", "theta_hY", "sched.c_log.raw", "sched.c_lin.raw"]
    for name in required:
        grad = named[name].grad
        value = float("nan") if grad is None else float(grad.abs().max())
        print(f"gradient {name}: {value:.3e}")
        if grad is None or not torch.isfinite(grad).all() or not bool((grad != 0).any()):
            raise AssertionError(f"Missing/nonfinite/zero gradient for {name}")

    # Direct prefix-state perturbation test at the oracle level.
    cut = 3
    X_alt = X.detach().clone()
    X_alt[:, cut:] += 5.0
    F, G, _ = layer._prefix_oracle(X.detach(), P.detach())
    F_alt, G_alt, _ = layer._prefix_oracle(X_alt, P.detach())
    diff = max(
        float((F[:, :cut] - F_alt[:, :cut]).abs().max().detach()),
        float((G[:, :cut] - G_alt[:, :cut]).abs().max().detach()),
    )
    print(f"prefix oracle perturbation difference: {diff:.3e}")
    if diff > 1e-6:
        raise AssertionError("Prefix-local oracle depends on suffix state")


if __name__ == "__main__":
    verify_ab2_order()
    verify_causal_pe_gradients()
    print("PASS: integrator order, gradients, and prefix locality verified.")
