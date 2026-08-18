#!/usr/bin/env python3
"""Verify new one-oracle causal PE damping discretizations."""

from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import (
    CausalPrefixExponentialPresymplecticEulerAttention,
    CausalPrefixHalfDampPresymplecticEulerAttention,
    CausalPrefixPresymplecticEulerAttention,
    ModelConfig,
    PresympModel,
)


def assert_close(x: torch.Tensor, y: torch.Tensor, tol: float = 2e-6) -> None:
    err = (x - y).abs().max().item()
    assert err <= tol, err


def attention_formula_checks() -> None:
    torch.manual_seed(7)
    cfg = ModelConfig(vocab_size=64, block_size=8, n_layer=1, n_head=2, n_embd=8, dropout=0.0, bias=False)
    kwargs = dict(h=0.1, eta_mode="linear", eta_mu=2.0, eta_learnable=False,
                  presymp_lnp="none", causal=True, lookahead=False)
    classes = [CausalPrefixPresymplecticEulerAttention,
               CausalPrefixExponentialPresymplecticEulerAttention,
               CausalPrefixHalfDampPresymplecticEulerAttention]
    modules = [cls(cfg, **kwargs).double() for cls in classes]
    state = modules[0].state_dict()
    for module in modules[1:]:
        module.load_state_dict(state)

    X = torch.randn(2, 6, 8, dtype=torch.float64)
    P = 0.2 * torch.randn(2, 6, 8, dtype=torch.float64)
    outputs = []
    for module in modules:
        calls = 0
        original = module._prefix_oracle

        def counted(self, *args, _original=original):
            nonlocal calls
            calls += 1
            return _original(*args)

        module._prefix_oracle = types.MethodType(counted, module)
        outputs.append(module.step(X, P, 1.0))
        assert calls == 1

    hX = modules[0].hX(dtype=X.dtype)
    hY = modules[0].hY(dtype=X.dtype)
    a = modules[0].sched.delta_eta_tensor(1.0, hY, X.device, X.dtype)
    sigma = torch.exp(-a)

    F0, G0, z0 = modules[0]._prefix_oracle(X, P)
    p_current = sigma * (P + hY * G0)
    x_current = X + hX * modules[0].c_B(p_current) / z0.unsqueeze(-1)
    assert_close(outputs[0][0], x_current)
    assert_close(outputs[0][1], p_current)

    phi1 = -torch.expm1(-a) / a
    p_exp = sigma * P + hY * phi1 * G0
    x_exp = X + hX * modules[1].c_B(p_exp) / z0.unsqueeze(-1)
    assert_close(outputs[1][0], x_exp)
    assert_close(outputs[1][1], p_exp)

    rho = torch.exp(-0.5 * a)
    p_half = rho * P
    _Fh, Gh, zh = modules[2]._prefix_oracle(X, p_half)
    p_kick = p_half + hY * Gh
    p_halfdamp = rho * p_kick
    x_halfdamp = X + hX * modules[2].c_B(p_kick) / zh.unsqueeze(-1)
    assert_close(outputs[2][0], x_halfdamp)
    assert_close(outputs[2][1], p_halfdamp)

    assert not torch.equal(outputs[0][0], outputs[1][0])
    assert not torch.equal(outputs[0][0], outputs[2][0])


def damping_weight_bound() -> None:
    # For frozen G and a=alpha*h, exact IF coefficient divided by h is
    # (1-exp(-a))/a. Half damping uses exp(-a/2), while incumbent uses exp(-a).
    a = torch.linspace(1e-5, 0.5, 10000, dtype=torch.float64)
    exact = -torch.expm1(-a) / a
    current = torch.exp(-a)
    half = torch.exp(-0.5 * a)
    assert torch.all((half - exact).abs() < (current - exact).abs())
    # Relevant initial range: alpha ~= 3/t and h=0.1 gives a in [0.176,0.300].
    for value in (0.176, 0.300):
        exact_v = -math.expm1(-value) / value
        current_underweight = 1.0 - math.exp(-value) / exact_v
        half_rel_error = abs(math.exp(-value / 2) / exact_v - 1.0)
        assert current_underweight > 0.08
        assert half_rel_error < 0.004


def model_causality_gradient_checks() -> None:
    torch.manual_seed(11)
    cfg = ModelConfig(vocab_size=71, block_size=8, n_layer=2, n_head=2, n_embd=8, dropout=0.0, bias=False)
    schemes = ["causal_pe", "causal_exp_pe", "causal_halfdamp_pe"]
    models = [PresympModel(cfg, attn_scheme=scheme, h=0.1, eta_mode="loglin",
                           eta_learnable=True, eta_log_init=3.0, eta_lin_init=1e-4,
                           use_v0_init=False, presymp_lnp="end", mlp_use_attn_vel=True).eval()
              for scheme in schemes]
    state = models[0].state_dict()
    for model in models[1:]:
        model.load_state_dict(state)
    assert len({sum(p.numel() for p in model.parameters()) for model in models}) == 1

    ids = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    changed = ids.clone()
    changed[:, 5:] = torch.randint(0, cfg.vocab_size, changed[:, 5:].shape)
    logits = []
    for model in models:
        out, _ = model(ids)
        perturbed, _ = model(changed)
        assert_close(out[:, :5], perturbed[:, :5], tol=0.0)
        loss = out.square().mean()
        loss.backward()
        for block in model.blocks:
            assert block.attn.theta_hX.grad is not None and torch.isfinite(block.attn.theta_hX.grad)
            assert block.attn.theta_hY.grad is not None and torch.isfinite(block.attn.theta_hY.grad)
        logits.append(out.detach())
    assert (logits[0] - logits[1]).abs().max().item() > 1e-8
    assert (logits[0] - logits[2]).abs().max().item() > 1e-8


def main() -> None:
    attention_formula_checks()
    damping_weight_bound()
    model_causality_gradient_checks()
    print("PASS: formulas, one-oracle budget, damping-weight bound, distinct outputs, gradients, parameter equality, suffix causality")


if __name__ == "__main__":
    main()
