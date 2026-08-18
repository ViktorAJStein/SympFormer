#!/usr/bin/env python3
"""Verify learned velocity initialization, gradients, capacity, and causality."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import GPTModel, ModelConfig, PresympModel, YuriiFormerModel
from train import initialize_learned_v0_tables


SCHEMES = (
    "causal_fe",
    "causal_pe",
    "causal_exp_pe",
    "causal_halfdamp_pe",
    "causal_ab2",
)


def config(*, full_scale: bool = False) -> ModelConfig:
    if full_scale:
        return ModelConfig(
            vocab_size=50304,
            block_size=512,
            n_layer=8,
            n_head=8,
            n_embd=512,
            dropout=0.0,
            bias=False,
        )
    return ModelConfig(
        vocab_size=97,
        block_size=8,
        n_layer=2,
        n_head=2,
        n_embd=8,
        dropout=0.0,
        bias=False,
    )


def sympformer(cfg: ModelConfig, scheme: str, *, learned: bool) -> PresympModel:
    return PresympModel(
        cfg,
        attn_scheme=scheme,
        h=0.1,
        eta_learnable=True,
        eta_mode="loglin",
        eta_log_init=3.0,
        eta_lin_init=1e-4,
        use_v0_init=learned,
        mlp_use_attn_vel=True,
        presymp_lnp="end",
    )


def max_suffix_residual(model, idx: torch.Tensor) -> float:
    model.eval()
    worst = 0.0
    with torch.no_grad():
        reference, _ = model(idx)
        for cut in range(1, idx.shape[1]):
            changed = idx.clone()
            changed[:, cut:] = (changed[:, cut:] + 17) % model.cfg.vocab_size
            observed, _ = model(changed)
            worst = max(
                worst,
                float((reference[:, :cut] - observed[:, :cut]).abs().max()),
            )
    return worst


def embedding_gradients(model, idx: torch.Tensor) -> dict[str, float]:
    model.train()
    model.zero_grad(set_to_none=True)
    _, loss = model(idx, idx, global_step=0)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    values = {}
    for name, parameter in model.named_parameters():
        if name in ("tok_v0_emb.weight", "pos_v0_emb.weight"):
            assert parameter.grad is not None, f"missing gradient: {name}"
            assert torch.isfinite(parameter.grad).all(), f"nonfinite gradient: {name}"
            norm = float(parameter.grad.norm())
            assert norm > 0.0, f"zero gradient: {name}"
            values[name] = norm
    assert set(values) == {"tok_v0_emb.weight", "pos_v0_emb.weight"}
    return values


def verify_small_models() -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    cfg = config()
    residuals: dict[str, float] = {}
    gradients: dict[str, dict[str, float]] = {}
    for trial in range(4):
        torch.manual_seed(20260819 + trial)
        idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
        models = {"yurii_lt": YuriiFormerModel(cfg, use_v0_init=True)}
        models.update({name: sympformer(cfg, name, learned=True) for name in SCHEMES})
        for name, model in models.items():
            residual = max_suffix_residual(model, idx)
            residuals[name] = max(residuals.get(name, 0.0), residual)
            assert residual <= 1e-6, (trial, name, residual)
            if trial == 0:
                gradients[name] = embedding_gradients(model, idx)
                names = set(dict(model.named_parameters()))
                assert "tok_v0_emb_mlp.weight" not in names
                assert "pos_v0_emb_mlp.weight" not in names
    return residuals, gradients


def verify_matched_initial_tables() -> float:
    cfg = config()
    seed = 260123236
    torch.manual_seed(seed)
    yurii = YuriiFormerModel(cfg, use_v0_init=True)
    torch.manual_seed(seed)
    pe = sympformer(cfg, "causal_pe", learned=True)
    initialize_learned_v0_tables(yurii, seed)
    initialize_learned_v0_tables(pe, seed)
    tensors = (
        (yurii.tok_v0_emb.weight, pe.tok_v0_emb.weight),
        (yurii.pos_v0_emb.weight, pe.pos_v0_emb.weight),
    )
    residual = max(float((left - right).detach().abs().max()) for left, right in tensors)
    assert residual == 0.0
    return residual


def verify_parameter_overhead() -> dict[str, int]:
    cfg = config(full_scale=True)
    expected = (cfg.vocab_size + cfg.block_size) * cfg.n_embd
    torch.manual_seed(1)
    baseline = sum(p.numel() for p in GPTModel(cfg).parameters())
    yurii_zero = sum(p.numel() for p in YuriiFormerModel(cfg, use_v0_init=False).parameters())
    yurii_learned = sum(p.numel() for p in YuriiFormerModel(cfg, use_v0_init=True).parameters())
    pe_zero = sum(p.numel() for p in sympformer(cfg, "causal_pe", learned=False).parameters())
    pe_learned = sum(p.numel() for p in sympformer(cfg, "causal_pe", learned=True).parameters())
    assert yurii_learned - yurii_zero == expected
    assert pe_learned - pe_zero == expected
    return {
        "baseline": baseline,
        "yurii_zero_v0": yurii_zero,
        "yurii_learned_v0": yurii_learned,
        "pe_zero_v0": pe_zero,
        "pe_learned_v0": pe_learned,
        "per_model_v0_overhead": expected,
    }


def verify_cli_route() -> None:
    source = Path("train.py").read_text()
    required = (
        '"--learned_v0_init"',
        'if args.no_v0_init and args.learned_v0_init:',
        'use_v0_init=(not args.no_v0_init)',
        'initialize_learned_v0_tables(model, args.seed)',
    )
    for text in required:
        assert text in source, f"missing CLI route: {text}"
    causal_constructor = source[source.index('if args.arch in {"causal_symp_fe"'):source.index('elif args.arch == "presymp"')]
    assert "use_v0_init=(not args.no_v0_init)" in causal_constructor


def main() -> None:
    residuals, gradients = verify_small_models()
    table_residual = verify_matched_initial_tables()
    counts = verify_parameter_overhead()
    verify_cli_route()
    print("suffix-causality residuals:", residuals)
    print("learned-v0 gradient norms:", gradients)
    print(f"same-seed v0 table residual (Yurii vs PE): {table_residual:.3e}")
    print("parameter counts:", counts)
    print("PASS: learned token/position velocity initialization is active, matched, and suffix-causal")


if __name__ == "__main__":
    main()
