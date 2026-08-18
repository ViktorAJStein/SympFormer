#!/usr/bin/env python3
"""Verify one-oracle Nesterov lookahead for headline causal PE."""

from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import CausalPrefixPresymplecticEulerAttention, ModelConfig, PresympModel


def build(*, active: bool, mu: float) -> PresympModel:
    torch.manual_seed(29)
    cfg = ModelConfig(vocab_size=79, block_size=8, n_layer=3, n_head=2, n_embd=16,
                      dropout=0.0, bias=False)
    return PresympModel(
        cfg,
        attn_scheme="causal_pe",
        h=0.1,
        eta_mode="loglin",
        eta_learnable=True,
        eta_log_init=3.0,
        eta_lin_init=1e-4,
        use_v0_init=False,
        presymp_lnp="end",
        mlp_use_attn_vel=True,
        lookahead=active,
        lookahead_init=mu,
    )


def direct_formula_and_call_count() -> None:
    torch.manual_seed(31)
    cfg = ModelConfig(vocab_size=32, block_size=6, n_layer=1, n_head=2, n_embd=8,
                      dropout=0.0, bias=False)
    module = CausalPrefixPresymplecticEulerAttention(
        cfg, h=0.1, eta_mode="linear", eta_mu=2.0, presymp_lnp="none",
        lookahead=True, lookahead_init=0.5,
    ).double()
    X = torch.randn(2, 6, 8, dtype=torch.float64)
    P = 0.2 * torch.randn(2, 6, 8, dtype=torch.float64)

    calls = 0
    original = module._prefix_oracle

    def counted(self, *args):
        nonlocal calls
        calls += 1
        return original(*args)

    module._prefix_oracle = types.MethodType(counted, module)
    X1, P1 = module.step(X, P, 1.0)
    assert calls == 1

    mu = module.mu_la().to(dtype=X.dtype)
    # _prefix_oracle applies LN to X+mu*P internally.
    _F, G, z = original(X, P)
    hX = module.hX(dtype=X.dtype)
    hY = module.hY(dtype=X.dtype)
    a = module.sched.delta_eta_tensor(1.0, hY, X.device, X.dtype)
    sigma = torch.exp(-a)
    expected_p = sigma * (P + hY * G)
    expected_x = X + hX * module.c_B(expected_p) / z.unsqueeze(-1)
    assert torch.allclose(P1, expected_p, atol=2e-6, rtol=0)
    assert torch.allclose(X1, expected_x, atol=2e-6, rtol=0)
    shift = mu * P
    assert shift.norm() < P.norm()
    assert math.isclose(mu.item(), 0.5, rel_tol=0, abs_tol=1e-6)


def full_model_checks() -> None:
    variants = [(False, 0.001), (True, 0.1), (True, 0.5), (True, 0.9)]
    models = [build(active=active, mu=mu).eval() for active, mu in variants]
    counts = {sum(parameter.numel() for parameter in model.parameters()) for model in models}
    assert len(counts) == 1

    ids = torch.randint(0, 79, (2, 8))
    suffix = ids.clone()
    suffix[:, 5:] = torch.randint(0, 79, suffix[:, 5:].shape)
    outputs = []
    for (active, mu), model in zip(variants, models):
        oracle_calls = [0 for _ in model.blocks]
        for index, block in enumerate(model.blocks):
            original = block.attn._prefix_oracle

            def counted(self, *args, _original=original, _index=index):
                oracle_calls[_index] += 1
                return _original(*args)

            block.attn._prefix_oracle = types.MethodType(counted, block.attn)

        logits, _ = model(ids)
        assert oracle_calls == [1] * len(model.blocks)
        changed, _ = model(suffix)
        assert torch.equal(logits[:, :5], changed[:, :5])
        logits.square().mean().backward()
        mu_grads = [block.attn.mu_la.raw.grad for block in model.blocks]
        if active:
            assert any(grad is not None and torch.isfinite(grad) and grad.abs().item() > 0 for grad in mu_grads)
            for block in model.blocks:
                assert math.isclose(block.attn.mu_la().item(), mu, rel_tol=0, abs_tol=2e-6)
        else:
            assert all(grad is None for grad in mu_grads)
        outputs.append(logits.detach())

    for index in range(1, len(outputs)):
        assert (outputs[0] - outputs[index]).abs().max().item() > 1e-8
    assert (outputs[1] - outputs[2]).abs().max().item() > 1e-8
    assert (outputs[2] - outputs[3]).abs().max().item() > 1e-8


def source_routing_check() -> None:
    source = (Path(__file__).resolve().parent.parent / "train.py").read_text()
    assert "lookahead=args.presymp_lookahead" in source
    assert "lookahead_init=args.presymp_lookahead_init" in source
    assert '"--presymp_lookahead_init"' in source
    assert "lookahead=False" not in source


def main() -> None:
    source_routing_check()
    direct_formula_and_call_count()
    full_model_checks()
    print("PASS: causal lookahead routing, formula, one-oracle budget, bound, gradients, equal parameters, distinct outputs, suffix causality")


if __name__ == "__main__":
    main()
