#!/usr/bin/env python3
"""Verify that proposed causal-PE architecture choices reach distinct code paths."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import ModelConfig, PresympModel


def build(*, mode: str, lnp: str = "end", no_mlp: bool = False) -> PresympModel:
    torch.manual_seed(17)
    return PresympModel(
        ModelConfig(vocab_size=64, block_size=8, n_layer=2, n_head=2, n_embd=16, dropout=0.0, bias=False),
        attn_scheme="causal_pe",
        h=0.1,
        t0=1.0,
        eta_log_init=3.0,
        eta_lin_init=0.0001,
        eta_learnable=True,
        eta_mode="loglin",
        use_v0_init=False,
        presymp_lnp=lnp,
        mlp_use_attn_vel=mode == "attn_vel",
        mlp_use_p_vel=mode == "p_vel",
        no_mlp=no_mlp,
    )


def logits_and_grads(model: PresympModel, idx: torch.Tensor) -> tuple[torch.Tensor, dict[str, bool]]:
    model.train()
    model.zero_grad(set_to_none=True)
    logits, loss = model(idx, idx, global_step=0)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    block = model.blocks[0]
    grads = {
        "mu": block.mu_mlp.raw.grad is not None,
        "beta": block.beta_mlp.raw.grad is not None,
        "gamma": block.gamma_mlp.raw.grad is not None,
    }
    return logits.detach(), grads


def suffix_invariance(model: PresympModel, idx: torch.Tensor, cut: int = 3) -> float:
    model.eval()
    alt = idx.clone()
    alt[:, cut + 1 :] = (alt[:, cut + 1 :] + 7) % model.cfg.vocab_size
    with torch.no_grad():
        a, _ = model(idx, global_step=0)
        b, _ = model(alt, global_step=0)
    return float((a[:, : cut + 1] - b[:, : cut + 1]).abs().max())


def main() -> None:
    train_source = Path("train.py").read_text()
    required = (
        "mlp_use_attn_vel=args.presymp_mlp_use_attn_vel",
        "mlp_use_p_vel=args.presymp_mlp_use_p_vel",
        'choices=["attn_vel", "p_vel", "separate_vel"]',
    )
    for needle in required:
        assert needle in train_source, f"train.py does not route explicit MLP choice: {needle}"

    idx = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]) % 64
    models = {mode: build(mode=mode) for mode in ("attn_vel", "separate_vel", "p_vel")}
    state = models["attn_vel"].state_dict()
    for mode in ("separate_vel", "p_vel"):
        models[mode].load_state_dict(state, strict=True)

    outputs = {}
    gradients = {}
    for mode, model in models.items():
        assert all(block.mlp_use_attn_vel == (mode == "attn_vel") for block in model.blocks)
        assert all(block.mlp_use_p_vel == (mode == "p_vel") for block in model.blocks)
        outputs[mode], gradients[mode] = logits_and_grads(model, idx)
        leak = suffix_invariance(model, idx)
        assert leak == 0.0, (mode, leak)

    for a, b in (("attn_vel", "separate_vel"), ("attn_vel", "p_vel"), ("separate_vel", "p_vel")):
        delta = float((outputs[a] - outputs[b]).abs().max())
        assert delta > 1e-7, f"MLP modes {a} and {b} collapse to identical outputs"

    assert gradients["attn_vel"] == {"mu": True, "beta": False, "gamma": True}
    assert gradients["separate_vel"] == {"mu": True, "beta": True, "gamma": True}
    assert gradients["p_vel"] == {"mu": True, "beta": True, "gamma": True}

    no_mlp = build(mode="attn_vel", no_mlp=True)
    no_mlp.load_state_dict(state, strict=True)
    no_mlp_out, _ = logits_and_grads(no_mlp, idx)
    assert float((outputs["attn_vel"] - no_mlp_out).abs().max()) > 1e-7

    # Code audit: causal PE applies momentum LN once at the end for every
    # non-none value. Thus end and each_substep are aliases, not distinct
    # architecture choices; only none versus normalized is currently meaningful.
    end = build(mode="attn_vel", lnp="end")
    each = build(mode="attn_vel", lnp="each_substep")
    none = build(mode="attn_vel", lnp="none")
    each.load_state_dict(end.state_dict(), strict=True)
    # Identity removes ln_p parameters, so load only shared parameters.
    none.load_state_dict(end.state_dict(), strict=False)
    end.eval(); each.eval(); none.eval()
    with torch.no_grad():
        out_end, _ = end(idx)
        out_each, _ = each(idx)
        out_none, _ = none(idx)
    assert torch.equal(out_end, out_each), "expected causal-PE end/each_substep alias changed"
    assert float((out_end - out_none).abs().max()) > 1e-7

    print("PASS: explicit MLP modes route to distinct outputs and expected gradient paths")
    print("PASS: all proposed MLP modes are suffix-causal")
    print("PASS: no-MLP is a distinct diagnostic")
    print("AUDIT: causal-PE presymp_lnp=end and each_substep are exact aliases; compare only none vs normalized")


if __name__ == "__main__":
    main()
