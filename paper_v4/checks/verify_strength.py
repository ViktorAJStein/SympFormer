#!/usr/bin/env python3
"""Tier-1 verification for the E054 learnable connection-strength implementation."""
import argparse
import math
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--training-source', type=Path, required=True)
    a = ap.parse_args()
    root = a.training_source.resolve()
    sys.path.insert(0, str(root))

    import torch
    from model import ModelConfig
    from normalized_hb import NormalizedHBConfig, NormalizedHBModel

    cfg = ModelConfig(vocab_size=64, block_size=8, n_layer=3, n_head=2, n_embd=16,
                      dropout=0.0, bias=False, yurii_force_gain_floor=1e-6)

    for mode, version in [('H1', 'e046_identity_v1'), ('R2', 'e048_r2_dual_ln_v1')]:
        torch.manual_seed(12345)
        fixed = NormalizedHBModel(cfg, NormalizedHBConfig(mode=mode, version=version))
        torch.manual_seed(12345)
        learned = NormalizedHBModel(cfg, NormalizedHBConfig(
            mode=mode, version=version, eta_learnable=True, eta_init=1.0))

        fs = fixed.state_dict(); ls = learned.state_dict()
        eta_keys = [k for k in ls if k.endswith('eta_connection.raw')]
        assert len(eta_keys) == cfg.n_layer, (mode, eta_keys)
        shared = sorted(set(fs).intersection(ls))
        assert shared
        for key in shared:
            assert torch.equal(fs[key], ls[key]), (mode, key, (fs[key]-ls[key]).abs().max().item())

        eta_ids = [id(block.eta_connection.raw) for block in learned.blocks]
        assert len(set(eta_ids)) == cfg.n_layer
        for block in learned.blocks:
            eta = block.eta_connection()
            assert float(eta.detach()) == 1.0, (mode, float(eta.detach()))

        g = torch.Generator(device='cpu'); g.manual_seed(77)
        x = torch.randn(2, 7, cfg.n_embd, generator=g)
        v = torch.randn(2, 7, cfg.n_embd, generator=g)
        fixed.train(); learned.train()
        xf, vf = x.clone(), v.clone()
        xl, vl = x.clone(), v.clone()
        for bf, bl in zip(fixed.blocks, learned.blocks):
            xf, vf = bf(xf, vf)
            xl, vl = bl(xl, vl)
        max_x = float((xf-xl).abs().max().detach())
        max_v = float((vf-vl).abs().max().detach())
        assert max_x <= 2e-6 and max_v <= 2e-6, (mode, max_x, max_v)

        loss = xl.square().mean() + vl.square().mean()
        loss.backward()
        grads = [float(b.eta_connection.raw.grad.detach()) for b in learned.blocks]
        assert all(math.isfinite(x) for x in grads), (mode, grads)
        assert any(abs(x) > 0 for x in grads), (mode, grads)

        rows = learned.hb_dynamics()
        assert len(rows) == cfg.n_layer
        assert all(r['eta_learnable'] == 1 and abs(r['eta_connection']-1.0) < 1e-12 for r in rows)
        assert all(math.isfinite(r['eta_connection_gradient_abs']) for r in rows)
        print('PASS:%s fixed-vs-learnable eta=1 max_x=%.3e max_v=%.3e grads=%s' %
              (mode, max_x, max_v, ','.join('%.3e' % q for q in grads)))

    # Fail-closed config checks.
    try:
        NormalizedHBConfig(mode='H0', version='e046_identity_v1', eta_learnable=True)
    except ValueError:
        pass
    else:
        raise AssertionError('H0 must reject learnable eta because no connection is active')
    try:
        NormalizedHBConfig(mode='H1', version='e046_identity_v1', eta_learnable=True, eta_init=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError('nonpositive eta_init must be rejected')

    print('PASS:E054 learnable-eta implementation verification')


if __name__ == '__main__':
    main()
