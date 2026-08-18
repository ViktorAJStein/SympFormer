#!/usr/bin/env python3
"""Exhaustive causality and prefix-reduction checks for all linear CLI methods."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import (  # noqa: E402
    LinAttnAB2Model, LinAttnETDAB2Model, LinAttnEulerModel, LinAttnModel,
    LinAttnPresympModel, LinAttnReducedModel, LinAttnYuriiModel, ModelConfig,
    ReducedLinearAttentionLayer,
)


def assert_close(a, b, label, atol=1e-7, rtol=1e-7):
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        raise AssertionError(f"{label}: max error {(a-b).abs().max().item():.3e}")


def build_models(cfg):
    common = dict(h=0.1, no_mlp=False)
    return {
        "lin_baseline": LinAttnModel(cfg, **common),
        "lin_yurii": LinAttnYuriiModel(cfg, use_v0_init=False, **common),
        "lin_euler": LinAttnEulerModel(cfg, use_v0_init=False, **common),
        "lin_presymp": LinAttnPresympModel(cfg, use_v0_init=False, eta_log_coef=3.0, **common),
        "lin_exp_euler": LinAttnPresympModel(cfg, use_v0_init=False, eta_log_coef=3.0, attn_cls="exp_euler", **common),
        "lin_ab2": LinAttnAB2Model(cfg, use_v0_init=False, eta_log_coef=3.0, **common),
        "lin_etd_ab2": LinAttnETDAB2Model(cfg, use_v0_init=False, eta_log_coef=3.0, **common),
        "lin_reduced_exp_mid": LinAttnReducedModel(cfg, scheme="exp_mid", noncausal=False, eta_log_coef=3.0, **common),
        "lin_reduced_ab2": LinAttnReducedModel(cfg, scheme="ab2", noncausal=False, eta_log_coef=3.0, **common),
    }


def verify_prefix_moments():
    cfg = ModelConfig(vocab_size=17, block_size=5, n_layer=1, n_head=1, n_embd=4, dropout=0.0)
    layer = ReducedLinearAttentionLayer(cfg, causal=True).double()
    X = torch.randn(2, 5, 4, generator=torch.Generator().manual_seed(7), dtype=torch.float64)
    Xn = layer.ln(X)
    got = layer._moments(Xn)
    expected = torch.empty_like(got)
    for i in range(5):
        expected[:, i] = Xn[:, : i + 1].transpose(-1, -2) @ Xn[:, : i + 1] / 5.0
    assert_close(got, expected, "prefix moments", atol=1e-12, rtol=1e-12)
    P = torch.randn(2, 5, 4, 4, generator=torch.Generator().manual_seed(8), dtype=torch.float64)
    A = torch.randn(4, 4, generator=torch.Generator().manual_seed(9), dtype=torch.float64)
    drift = layer._row_drift(Xn, A, got, P)
    rows = []
    for i in range(5):
        value = Xn[:, i] @ A
        value = torch.bmm(value.unsqueeze(1), expected[:, i])
        value = torch.bmm(value, P[:, i]).squeeze(1)
        rows.append(value)
    direct = torch.stack(rows, dim=1)
    assert_close(drift, direct, "prefix row drift", atol=1e-12, rtol=1e-12)


def main():
    torch.manual_seed(12)
    cfg = ModelConfig(vocab_size=43, block_size=8, n_layer=3, n_head=1, n_embd=6, dropout=0.0)
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    changed = idx.clone(); changed[:, 4:] = (changed[:, 4:] + 11) % cfg.vocab_size
    targets = torch.randint(0, cfg.vocab_size, idx.shape)
    for name, model in build_models(cfg).items():
        model.eval()
        prefix = model(idx)[0][:, :4]
        changed_prefix = model(changed)[0][:, :4]
        assert_close(prefix, changed_prefix, f"{name} suffix causality")
        model.train()
        loss = model(idx, targets)[1]
        if not torch.isfinite(loss):
            raise AssertionError(f"{name}: nonfinite loss")
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise AssertionError(f"{name}: missing/nonfinite active gradients")
    verify_prefix_moments()
    source = (ROOT / "train.py").read_text()
    assert "--lin_noncausal is disabled for decoder training" in source
    print("PASS: all 9 linear variants are suffix-causal with finite gradients")
    print("PASS: prefix moments/drifts equal direct loops; noncausal CLI is disabled")


if __name__ == "__main__":
    main()
