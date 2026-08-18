#!/usr/bin/env python3
"""Verify equation-level YuriiFormer alignment and the evaluated v0 deviation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from model import GPTModel, ModelConfig, YuriiFormerLieTrotterBlock, YuriiFormerModel


def max_prefix_change(model: YuriiFormerModel, idx: torch.Tensor, cut: int) -> float:
    changed = idx.clone()
    changed[:, cut:] = (changed[:, cut:] + 17) % model.cfg.vocab_size
    logits_a, _ = model(idx)
    logits_b, _ = model(changed)
    return float((logits_a[:, :cut] - logits_b[:, :cut]).abs().max().item())


def verify_block_formula() -> tuple[float, float]:
    torch.manual_seed(260123236)
    cfg = ModelConfig(
        vocab_size=97,
        block_size=8,
        n_layer=2,
        n_head=2,
        n_embd=8,
        dropout=0.0,
        bias=False,
    )
    block = YuriiFormerLieTrotterBlock(cfg).eval()
    x = torch.randn(2, cfg.block_size, cfg.n_embd)
    v = torch.randn_like(x)
    with torch.no_grad():
        x_in = x + block.mu1() * v
        attn_direction = block.attn(block.ln_x_attn(x_in))
        v_half = block.ln_v(block.beta1() * v + block.gamma1() * attn_direction)
        x_half = x + v_half
        x_in_half = x_half + block.mu2() * v_half
        mlp_direction = block.mlp(block.ln_x_mlp(x_in_half))
        v_next = block.ln_v(block.beta2() * v_half + block.gamma2() * mlp_direction)
        x_next = x_half + v_next
        observed_x, observed_v = block(x, v)
    x_error = float((observed_x - x_next).abs().max())
    v_error = float((observed_v - v_next).abs().max())
    assert x_error == 0.0 and v_error == 0.0
    return x_error, v_error


def verify_causality() -> dict[bool, float]:
    torch.manual_seed(20260818)
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
    results = {}
    for use_v0 in (False, True):
        model = YuriiFormerModel(cfg, use_v0_init=use_v0).eval()
        worst = max(max_prefix_change(model, idx, cut) for cut in range(1, cfg.block_size))
        assert worst <= 1e-6
        results[use_v0] = worst
    return results


def parameter_counts() -> dict[str, int]:
    cfg = ModelConfig(
        vocab_size=50304,
        block_size=512,
        n_layer=8,
        n_head=8,
        n_embd=512,
        dropout=0.0,
        bias=False,
    )
    counts = {
        "baseline": sum(p.numel() for p in GPTModel(cfg).parameters()),
        "yurii_zero_v0": sum(
            p.numel() for p in YuriiFormerModel(cfg, use_v0_init=False).parameters()
        ),
        "yurii_learned_v0": sum(
            p.numel() for p in YuriiFormerModel(cfg, use_v0_init=True).parameters()
        ),
    }
    expected_v0 = (cfg.vocab_size + cfg.block_size) * cfg.n_embd
    assert counts["yurii_learned_v0"] - counts["yurii_zero_v0"] == expected_v0
    return counts


def verify_evaluated_specs() -> int:
    generator = Path("scripts/make_high_iter_specs.py").read_text()
    checked_in = Path("jobs/headline_lr_200m.tsv").read_text().splitlines()
    assert '"--no_v0_init"' in generator
    yurii_rows = [line for line in checked_in if "\tyurii_lt\t" in line]
    assert len(yurii_rows) == 8
    assert all("--no_v0_init" in line for line in yurii_rows)
    return len(yurii_rows)


def main() -> None:
    x_error, v_error = verify_block_formula()
    causal = verify_causality()
    counts = parameter_counts()
    tuning_rows = verify_evaluated_specs()
    print(f"paper Lie--Trotter formula residuals: X={x_error:.3e}, V={v_error:.3e}")
    print(
        "suffix-causality residuals: "
        f"zero-v0={causal[False]:.3e}, learned-v0={causal[True]:.3e}"
    )
    print("parameter counts:", counts)
    print(f"checked YuriiFormer tuning rows with --no_v0_init: {tuning_rows}")
    print("PASS: equations align; evaluated runs intentionally use the zero-v0 variant.")


if __name__ == "__main__":
    main()
