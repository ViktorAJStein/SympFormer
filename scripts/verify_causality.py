#!/usr/bin/env python3
"""Perturb suffix token ids and verify that prefix logits are invariant."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from model import GPTModel, ModelConfig, PresympModel, YuriiFormerModel


def max_prefix_change(model, idx, cut):
    changed = idx.clone()
    vocab = model.cfg.vocab_size
    changed[:, cut:] = (changed[:, cut:] + 17) % vocab
    logits_a, _ = model(idx)
    logits_b, _ = model(changed)
    return float((logits_a[:, :cut] - logits_b[:, :cut]).abs().max().item())


def main():
    torch.manual_seed(20260727)
    cfg = ModelConfig(
        vocab_size=97,
        block_size=8,
        n_layer=2,
        n_head=2,
        n_embd=8,
        dropout=0.0,
        bias=False,
    )
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    headline = {
        "baseline": GPTModel(cfg),
        "baseline_capacity": GPTModel(cfg, capacity_control=True),
        "yurii_lt_zero_v0": YuriiFormerModel(cfg, use_v0_init=False),
        "causal_symp_fe": PresympModel(
            cfg,
            attn_scheme="causal_fe",
            h=0.1,
            eta_learnable=True,
            eta_mode="loglin",
            eta_log_init=3.0,
            eta_lin_init=1e-4,
            use_v0_init=False,
            mlp_use_attn_vel=True,
            presymp_lnp="end",
        ),
        "causal_symp_pe": PresympModel(
            cfg,
            attn_scheme="causal_pe",
            h=0.1,
            eta_learnable=True,
            eta_mode="loglin",
            eta_log_init=3.0,
            eta_lin_init=1e-4,
            use_v0_init=False,
            mlp_use_attn_vel=True,
            presymp_lnp="end",
        ),
    }
    tolerance = 1e-6
    for name, model in headline.items():
        model.eval()
        worst = max(max_prefix_change(model, idx, cut) for cut in range(1, cfg.block_size))
        print(f"{name}: max_prefix_change={worst:.3e}")
        if worst > tolerance:
            raise AssertionError(f"{name} is noncausal: {worst} > {tolerance}")

    # This is an explicit regression witness for the legacy global-Hamiltonian
    # force. Passing this check means the witness detected the known problem.
    legacy = PresympModel(
        cfg,
        attn_scheme="euler",
        h=0.1,
        eta_learnable=True,
        eta_mode="loglin",
        eta_log_init=3.0,
        eta_lin_init=1e-4,
        use_v0_init=False,
        mlp_use_attn_vel=True,
        presymp_lnp="end",
    )
    legacy.eval()
    legacy_worst = max(max_prefix_change(legacy, idx, cut) for cut in range(1, cfg.block_size))
    print(f"legacy_global_hamiltonian: max_prefix_change={legacy_worst:.3e} (expected nonzero)")
    if legacy_worst <= tolerance:
        raise AssertionError("Legacy noncausality witness did not trigger")
    print("PASS: headline decoders are causal and the legacy regression witness triggers.")


if __name__ == "__main__":
    main()
