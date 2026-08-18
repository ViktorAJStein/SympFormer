#!/usr/bin/env python3
"""Forward/backward and finite-value smoke tests for headline architectures."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from model import GPTModel, ModelConfig, PresympModel, YuriiFormerModel


def builders(cfg):
    return {
        "baseline": lambda: GPTModel(cfg),
        "baseline_capacity": lambda: GPTModel(cfg, capacity_control=True),
        "yurii_lt_zero_v0": lambda: YuriiFormerModel(cfg, use_v0_init=False),
        "causal_symp_pe": lambda: PresympModel(
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


def main():
    cfg = ModelConfig(vocab_size=64, block_size=8, n_layer=2, n_head=2, n_embd=8)
    torch.manual_seed(20260727)
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    targets = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    for dtype in (torch.float32, torch.bfloat16):
        for name, build in builders(cfg).items():
            torch.manual_seed(11)
            model = build()
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=(dtype == torch.bfloat16)):
                logits, loss = model(idx, targets)
            if not torch.isfinite(logits).all() or loss is None or not torch.isfinite(loss):
                raise AssertionError(f"{name} produced nonfinite values in {dtype}")
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads):
                raise AssertionError(f"{name} has missing/nonfinite gradients in {dtype}")
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"{name} dtype={dtype} params={params} loss={float(loss.detach()):.6f}")
    # The capacity adapter is a functional no-op when the shared baseline
    # weights are copied, while adding roughly the same C-by-C matrix per layer
    # as the causal method.
    torch.manual_seed(23)
    baseline = GPTModel(cfg).eval()
    capacity = GPTModel(cfg, capacity_control=True).eval()
    missing, unexpected = capacity.load_state_dict(baseline.state_dict(), strict=False)
    if unexpected or any("capacity_adapter.weight" not in name for name in missing):
        raise AssertionError(f"unexpected capacity-control state mismatch: {missing}, {unexpected}")
    logits_base, _ = baseline(idx)
    logits_capacity, _ = capacity(idx)
    if not torch.equal(logits_base, logits_capacity):
        raise AssertionError("zero-initialized capacity control changed the baseline function")

    # Noise is reproducible under a fixed RNG seed, active only in training,
    # and disabled in evaluation.
    initialized = PresympModel(
        cfg, attn_scheme="causal_pe", h=0.1, use_v0_init=False,
        mlp_use_attn_vel=True,
    )
    eye = torch.eye(cfg.n_embd)
    for layer, block in enumerate(initialized.blocks):
        if not torch.equal(block.attn.c_B.weight.detach(), eye):
            raise AssertionError(f"c_B identity initialization was overwritten in layer {layer}")

    noisy = PresympModel(
        cfg, attn_scheme="causal_pe", h=0.1, use_v0_init=False,
        mlp_use_attn_vel=True, noise_eta=1e-4, noise_gamma=0.55,
    )
    noisy.train()
    torch.manual_seed(41)
    logits_a, _ = noisy(idx, global_step=5)
    torch.manual_seed(41)
    logits_b, _ = noisy(idx, global_step=5)
    if not torch.equal(logits_a, logits_b):
        raise AssertionError("momentum noise is not reproducible under fixed RNG")
    torch.manual_seed(42)
    logits_c, _ = noisy(idx, global_step=5)
    if torch.equal(logits_a, logits_c):
        raise AssertionError("momentum-noise ablation had no training-time effect")
    noisy.eval()
    torch.manual_seed(43)
    logits_d, _ = noisy(idx, global_step=5)
    torch.manual_seed(44)
    logits_e, _ = noisy(idx, global_step=5)
    if not torch.equal(logits_d, logits_e):
        raise AssertionError("momentum noise remained active during evaluation")
    print("PASS: headline/control architectures, capacity equality, and noise gating verified.")


if __name__ == "__main__":
    main()
