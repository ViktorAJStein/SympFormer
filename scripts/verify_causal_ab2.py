#!/usr/bin/env python3
"""Verify causal variable-step AB2 against direct formulas and invariants."""

from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import ModelConfig, PresympModel, variable_step_ab2


def inv_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def build(scheme: str, *, no_mlp: bool = False, lnp: str = "end") -> PresympModel:
    torch.manual_seed(41)
    return PresympModel(
        ModelConfig(vocab_size=83, block_size=8, n_layer=3, n_head=2, n_embd=16,
                    dropout=0.0, bias=False),
        attn_scheme=scheme,
        h=0.1,
        eta_mode="loglin",
        eta_learnable=True,
        eta_log_init=3.0,
        eta_lin_init=1e-4,
        use_v0_init=False,
        presymp_lnp=lnp,
        mlp_use_attn_vel=True,
        no_mlp=no_mlp,
        lookahead=False,
    )


def aligned_initialization_and_capacity() -> None:
    pe = build("causal_pe")
    ab2 = build("causal_ab2")
    assert sum(p.numel() for p in pe.parameters()) == sum(p.numel() for p in ab2.parameters())
    pe_state, ab2_state = pe.state_dict(), ab2.state_dict()
    assert pe_state.keys() == ab2_state.keys()
    for key in pe_state:
        assert torch.equal(pe_state[key], ab2_state[key]), key


def direct_formula_and_step_histories() -> None:
    model = build("causal_ab2", no_mlp=True, lnp="none").double().eval()
    hx = (0.08, 0.11, 0.07)
    hy = (0.09, 0.05, 0.12)
    calls = [0, 0, 0]
    for index, (block, sx, sy) in enumerate(zip(model.blocks, hx, hy)):
        block.attn.theta_hX.data.fill_(inv_softplus(sx))
        block.attn.theta_hY.data.fill_(inv_softplus(sy))

        def oracle(self, X, P, _index=index):
            calls[_index] += 1
            F = 0.2 * X + 0.3 * P
            G = -0.1 * X + 0.4 * P
            z = torch.ones(X.shape[:2], device=X.device, dtype=X.dtype)
            return F, G, z

        block.attn._prefix_oracle = types.MethodType(oracle, block.attn)

    idx = torch.randint(0, model.cfg.vocab_size, (2, model.cfg.block_size))
    logits, _ = model(idx)
    assert calls == [1, 1, 1]

    pos = torch.arange(model.cfg.block_size)
    X = model.tok_emb(idx) + model.pos_emb(pos)[None, :, :]
    P = torch.zeros_like(X)
    F_prev = R_prev = None
    hX_prev = hY_prev = None
    t = float(model.blocks[0].attn.sched.t0)
    for block in model.blocks:
        hX = block.attn.hX(device=X.device, dtype=X.dtype)
        hY = block.attn.hY(device=X.device, dtype=X.dtype)
        F = 0.2 * X + 0.3 * P
        G = -0.1 * X + 0.4 * P
        R = G - block.attn.sched.alpha(t, X.device, X.dtype) * P
        if F_prev is None:
            F_eff, R_eff = F, R
        else:
            F_eff = variable_step_ab2(F, F_prev, hX, hX_prev)
            R_eff = variable_step_ab2(R, R_prev, hY, hY_prev)
        X = X + hX * F_eff
        P = P + hY * R_eff
        F_prev, R_prev = F, R
        hX_prev, hY_prev = hX.detach(), hY.detach()
        t += hX.detach().item()
    expected = model.lm_head(model.ln_f(X))
    assert torch.allclose(logits, expected, atol=2e-10, rtol=0)


def causality_gradients_and_distinction() -> None:
    pe = build("causal_pe").eval()
    ab2 = build("causal_ab2").eval()
    ids = torch.randint(0, pe.cfg.vocab_size, (2, pe.cfg.block_size))
    changed = ids.clone()
    changed[:, 5:] = torch.randint(0, pe.cfg.vocab_size, changed[:, 5:].shape)

    outputs = []
    for model in (pe, ab2):
        logits, _ = model(ids)
        perturbed, _ = model(changed)
        assert torch.equal(logits[:, :5], perturbed[:, :5])
        logits.square().mean().backward()
        for block in model.blocks:
            for parameter in (block.attn.theta_hX, block.attn.theta_hY, block.attn.sched.c_log.raw):
                if parameter.grad is not None:
                    assert torch.isfinite(parameter.grad).all()
        assert any(block.attn.theta_hX.grad is not None and block.attn.theta_hX.grad.abs().item() > 0 for block in model.blocks)
        assert any(block.attn.theta_hY.grad is not None and block.attn.theta_hY.grad.abs().item() > 0 for block in model.blocks)
        if model.attn_scheme == "causal_ab2":
            # With an X-only language-model readout, the terminal P update has
            # no downstream consumer in simultaneous AB2; this is the expected
            # endpoint adjoint, not a disconnected earlier dynamics path.
            assert model.blocks[-1].attn.theta_hY.grad is None
            assert model.blocks[-1].attn.sched.c_log.raw.grad is None
        outputs.append(logits.detach())
    assert (outputs[0] - outputs[1]).abs().max().item() > 1e-8


def source_routing_check() -> None:
    train = (Path(__file__).resolve().parent.parent / "train.py").read_text()
    model = (Path(__file__).resolve().parent.parent / "model.py").read_text()
    assert '"causal_symp_ab2": "causal_ab2"' in train
    assert 'self.attn_scheme == "causal_ab2"' in model


def main() -> None:
    source_routing_check()
    aligned_initialization_and_capacity()
    direct_formula_and_step_histories()
    causality_gradients_and_distinction()
    print("PASS: causal AB2 routing, bootstrap/formulas, separate step histories, one-oracle budget, aligned initialization, equal parameters, gradients, distinct output, suffix causality")


if __name__ == "__main__":
    main()
