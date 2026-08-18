#!/usr/bin/env python3
"""Verify the controlled causal forward-Euler/presymplectic-Euler pair."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from model import (
    CausalPrefixForwardEulerAttention,
    CausalPrefixPresymplecticEulerAttention,
    ModelConfig,
    PresympModel,
)


def build_layer(cls):
    cfg = ModelConfig(
        vocab_size=32,
        block_size=5,
        n_layer=1,
        n_head=1,
        n_embd=4,
        dropout=0.0,
    )
    return cls(
        cfg,
        h=0.1,
        eta_learnable=True,
        eta_mode="loglin",
        eta_log_init=3.0,
        eta_lin_init=1e-3,
        presymp_lnp="none",
        lookahead=False,
    )


def verify_one_step_formulas_and_gradients():
    torch.manual_seed(7)
    X = torch.randn(2, 5, 4)
    P = torch.randn(2, 5, 4)
    layers = {
        "causal_fe": build_layer(CausalPrefixForwardEulerAttention),
        "causal_pe": build_layer(CausalPrefixPresymplecticEulerAttention),
    }
    layers["causal_pe"].load_state_dict(layers["causal_fe"].state_dict())

    outputs = {}
    for label, layer in layers.items():
        Xi = X.detach().clone().requires_grad_(True)
        Pi = P.detach().clone().requires_grad_(True)
        X1, P1 = layer.step(Xi, Pi, 1.0)
        outputs[label] = (X1.detach(), P1.detach())
        loss = X1.square().mean() + P1.square().mean()
        loss.backward()
        named = dict(layer.named_parameters())
        required = ["theta_hX", "theta_hY", "sched.c_log.raw", "sched.c_lin.raw"]
        for name in required:
            grad = named[name].grad
            value = float("nan") if grad is None else float(grad.abs().max())
            print(f"{label} gradient {name}: {value:.3e}")
            if grad is None or not torch.isfinite(grad).all() or not bool((grad != 0).any()):
                raise AssertionError(f"{label}: missing/nonfinite/zero gradient for {name}")

    X_fe, P_fe = outputs["causal_fe"]
    X_pe, P_pe = outputs["causal_pe"]
    if not torch.equal(P_fe, P_pe):
        raise AssertionError("Causal FE and PE momentum updates are not identical")

    position_difference = float((X_fe - X_pe).abs().max())
    print(f"causal FE/PE position difference: {position_difference:.3e}")
    if position_difference <= 1e-8:
        raise AssertionError("Causal FE and PE position updates unexpectedly coincide")

    layer = layers["causal_fe"]
    with torch.no_grad():
        F_old, G_old, z = layer._prefix_oracle(X, P)
        hX = layer.hX(device=X.device, dtype=X.dtype)
        hY = layer.hY(device=X.device, dtype=X.dtype)
        d_eta = layer.sched.delta_eta_tensor(1.0, hY, X.device, X.dtype)
        sigma = torch.exp(-d_eta)
        expected_p = sigma * (P + hY * G_old)
        expected_fe_x = X + hX * F_old
        expected_pe_x = X + hX * (layer.c_B(expected_p) / z.unsqueeze(-1))

    checks = {
        "causal FE position": (X_fe, expected_fe_x),
        "causal PE position": (X_pe, expected_pe_x),
        "shared momentum": (P_fe, expected_p),
    }
    for label, (actual, expected) in checks.items():
        error = float((actual - expected).abs().max())
        print(f"{label} formula error: {error:.3e}")
        if not torch.allclose(actual, expected, rtol=0.0, atol=1e-7):
            raise AssertionError(f"{label} formula mismatch")

    cut = 3
    X_alt = X.clone()
    X_alt[:, cut:] += 5.0
    F, G, _ = layer._prefix_oracle(X, P)
    F_alt, G_alt, _ = layer._prefix_oracle(X_alt, P)
    prefix_difference = max(
        float((F[:, :cut] - F_alt[:, :cut]).abs().max().detach()),
        float((G[:, :cut] - G_alt[:, :cut]).abs().max().detach()),
    )
    print(f"prefix oracle perturbation difference: {prefix_difference:.3e}")
    if prefix_difference > 1e-6:
        raise AssertionError("Prefix-local oracle depends on suffix state")


def build_model(cfg, scheme):
    return PresympModel(
        cfg,
        attn_scheme=scheme,
        h=0.1,
        eta_learnable=True,
        eta_mode="loglin",
        eta_log_init=3.0,
        eta_lin_init=1e-4,
        use_v0_init=False,
        mlp_use_attn_vel=True,
        presymp_lnp="end",
    )


def verify_model_pair():
    cfg = ModelConfig(
        vocab_size=64,
        block_size=8,
        n_layer=2,
        n_head=2,
        n_embd=8,
        dropout=0.0,
    )
    torch.manual_seed(23)
    fe = build_model(cfg, "causal_fe")
    torch.manual_seed(23)
    pe = build_model(cfg, "causal_pe")

    fe_shapes = {name: tuple(value.shape) for name, value in fe.state_dict().items()}
    pe_shapes = {name: tuple(value.shape) for name, value in pe.state_dict().items()}
    if fe_shapes != pe_shapes:
        raise AssertionError("Causal FE and PE do not have identical state shapes")

    fe_params = sum(p.numel() for p in fe.parameters() if p.requires_grad)
    pe_params = sum(p.numel() for p in pe.parameters() if p.requires_grad)
    print(f"causal FE parameters: {fe_params}")
    print(f"causal PE parameters: {pe_params}")
    if fe_params != pe_params:
        raise AssertionError("Causal FE and PE parameter counts differ")

    torch.manual_seed(29)
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    targets = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    for dtype in (torch.float32, torch.bfloat16):
        for label, model in (("causal_fe", fe), ("causal_pe", pe)):
            model.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cpu",
                dtype=torch.bfloat16,
                enabled=(dtype == torch.bfloat16),
            ):
                logits, loss = model(idx, targets)
            if loss is None or not torch.isfinite(loss) or not torch.isfinite(logits).all():
                raise AssertionError(f"{label} produced nonfinite values in {dtype}")
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
            if not grads or not all(torch.isfinite(grad).all() for grad in grads):
                raise AssertionError(f"{label} has missing/nonfinite gradients in {dtype}")
            print(f"{label} dtype={dtype} loss={float(loss.detach()):.6f}")


if __name__ == "__main__":
    verify_one_step_formulas_and_gradients()
    verify_model_pair()
    print("PASS: causal FE/PE formulas, controlled difference, gradients, locality, and model smoke verified.")
