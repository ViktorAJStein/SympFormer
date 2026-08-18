#!/usr/bin/env python3
"""Verify arXiv-v3 reduced linear-attention closure and discretizations."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import (  # noqa: E402
    EtaSchedule,
    LinAttnPresympModel,
    LinAttnReducedModel,
    ModelConfig,
    ReducedLinearAttentionLayer,
    _lin_FG,
    variable_step_ab2,
)

DTYPE = torch.float64


def symmetric(d: int, generator: torch.Generator, scale: float = 0.2) -> torch.Tensor:
    raw = torch.randn(d, d, generator=generator, dtype=DTYPE)
    return scale * (raw + raw.T)


def assert_close(actual: torch.Tensor, expected: torch.Tensor, label: str, atol=2e-10, rtol=2e-10) -> None:
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        raise AssertionError(f"{label}: max error={(actual - expected).abs().max().item():.3e}")


def verify_exact_closure() -> None:
    g = torch.Generator().manual_seed(20260817)
    for n, d in [(3, 2), (7, 4), (9, 3)]:
        X = 0.3 * torch.randn(n, d, generator=g, dtype=DTYPE)
        A, V, P = symmetric(d, g), symmetric(d, g), symmetric(d, g)
        Y = X @ P
        alpha = torch.tensor(0.37, dtype=DTYPE)
        F_token, G_token = _lin_FG(X.unsqueeze(0), Y.unsqueeze(0), A, V, causal=False)
        F_token, G_token = F_token[0], G_token[0] - alpha * Y
        S = X.T @ X / float(n)
        dX = X @ A @ S @ P
        dP = -alpha * P - A @ S @ (P @ P) - (P @ P) @ S @ A + V
        reconstructed_dY = dX @ P + X @ dP
        assert_close(dX, F_token, f"closure dX n={n},d={d}")
        assert_close(reconstructed_dY, G_token, f"closure dY n={n},d={d}")
        assert_close(dP, dP.T, f"P symmetry n={n},d={d}")


def verify_damping_integral() -> None:
    dt = torch.tensor(0.23, dtype=DTYPE, requires_grad=True)
    t = 1.4
    m = 0.7
    linear = EtaSchedule(t0=1.0, mode="linear", lin_coef=m, learnable=False)
    got_linear = linear.damping_integral_tensor(t, dt, dt.device, dt.dtype)
    expected_linear = (1.0 - torch.exp(-m * dt)) / m
    assert_close(got_linear, expected_linear, "constant-damping I_k", atol=2e-14, rtol=2e-14)

    r = 3.0
    log = EtaSchedule(t0=1.0, mode="log", log_coef=r, learnable=False)
    got_log = log.damping_integral_tensor(t, dt, dt.device, dt.dtype)
    t1 = t + dt
    expected_log = t1 / (r + 1.0) * (1.0 - (t / t1) ** (r + 1.0))
    assert_close(got_log, expected_log, "log-damping I_k", atol=2e-14, rtol=2e-14)

    loglin = EtaSchedule(t0=1.0, mode="loglin", log_coef=2.7, lin_coef=0.13, learnable=False)
    got_loglin = loglin.damping_integral_tensor(t, dt, dt.device, dt.dtype)
    grid = torch.linspace(0.0, 1.0, 20001, dtype=DTYPE)
    s = t + grid * dt.detach()
    t1_detached = t + dt.detach()
    integrand = torch.exp(-(2.7 * torch.log(t1_detached / s) + 0.13 * (t1_detached - s)))
    expected_loglin = torch.trapezoid(integrand, s)
    assert_close(got_loglin.detach(), expected_loglin, "log-linear quadrature", atol=1e-11, rtol=1e-11)
    got_loglin.backward()
    assert dt.grad is not None and torch.isfinite(dt.grad) and abs(dt.grad.item()) > 0


def verify_corrected_exp_mid() -> None:
    cfg = ModelConfig(vocab_size=32, block_size=5, n_layer=1, n_head=1, n_embd=3, dropout=0.0)
    layer = ReducedLinearAttentionLayer(
        cfg, h=0.17, t0=1.0, eta_mode="linear", eta_lin_coef=0.6,
        eta_learnable=False, causal=False,
    ).to(dtype=DTYPE)
    with torch.no_grad():
        layer.c_A.weight.copy_(torch.tensor([[0.4, 0.1, 0.0], [0.1, 0.3, -0.05], [0.0, -0.05, 0.2]], dtype=DTYPE))
        layer.c_V.weight.copy_(torch.tensor([[0.2, 0.02, 0.0], [0.02, -0.1, 0.03], [0.0, 0.03, 0.15]], dtype=DTYPE))
    X = torch.randn(2, 5, 3, generator=torch.Generator().manual_seed(3), dtype=DTYPE)
    P = symmetric(3, torch.Generator().manual_seed(4), scale=0.08).expand(2, -1, -1).clone()
    tk = 1.2
    X_new, P_new = layer.exp_mid_step(X, P, tk)
    h = layer.h(dtype=DTYPE)
    Xn = layer.ln(X)
    S = Xn.transpose(-1, -2) @ Xn / 5.0
    A, V = layer.matrices()
    R = -A @ S @ (P @ P) - (P @ P) @ S @ A + V
    sigma = torch.exp(-layer.sched.delta_eta_tensor(tk, h, X.device, X.dtype))
    I_k = layer.sched.damping_integral_tensor(tk, h, X.device, X.dtype)
    expected_P = sigma * P + I_k * R
    expected_P = 0.5 * (expected_P + expected_P.transpose(-1, -2))
    expected_X = X + h * (Xn @ A @ S @ (0.5 * (P + expected_P)))
    assert_close(P_new, expected_P, "corrected exponential momentum")
    assert_close(X_new, expected_X, "averaged-momentum drift")

    old_double_damped = sigma * P + I_k * (-0.6 * P + R)
    if torch.allclose(old_double_damped, expected_P, atol=1e-7, rtol=1e-7):
        raise AssertionError("old draft formula did not expose double damping")

    P0 = torch.zeros_like(P)
    X_first, P_first = layer.exp_mid_step(X, P0, tk)
    assert (X_first - X).norm().item() > 0.0
    assert_close(P_first, P_first.transpose(-1, -2), "exp-mid symmetry")


def verify_ab2_and_models() -> None:
    current = torch.tensor([1.2, -0.4], dtype=DTYPE)
    previous = torch.tensor([-0.3, 0.7], dtype=DTYPE)
    h, hp = torch.tensor(0.2, dtype=DTYPE), torch.tensor(0.125, dtype=DTYPE)
    expected = (1.0 + 0.5 * h / hp) * current - 0.5 * h / hp * previous
    assert_close(variable_step_ab2(current, previous, h, hp), expected, "variable-step AB2")

    cfg = ModelConfig(vocab_size=41, block_size=8, n_layer=3, n_head=1, n_embd=6, dropout=0.0)
    torch.manual_seed(11)
    exp_model = LinAttnReducedModel(
        cfg, scheme="exp_mid", h=0.1, eta_mode="log", eta_log_coef=3.0,
        no_mlp=False, noncausal=False,
    )
    torch.manual_seed(11)
    ab2_model = LinAttnReducedModel(
        cfg, scheme="ab2", h=0.1, eta_mode="log", eta_log_coef=3.0,
        no_mlp=False, noncausal=False,
    )
    assert sum(p.numel() for p in exp_model.parameters()) == sum(p.numel() for p in ab2_model.parameters())
    idx = torch.randint(0, cfg.vocab_size, (2, 8), generator=torch.Generator().manual_seed(9))
    targets = torch.randint(0, cfg.vocab_size, (2, 8), generator=torch.Generator().manual_seed(10))
    logits_exp, loss_exp = exp_model(idx, targets)
    logits_ab2, loss_ab2 = ab2_model(idx, targets)
    assert torch.isfinite(loss_exp) and torch.isfinite(loss_ab2)
    assert not torch.allclose(logits_exp, logits_ab2)
    (loss_exp + loss_ab2).backward()
    for name, model in (("exp_mid", exp_model), ("ab2", ab2_model)):
        active = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert active and all(torch.isfinite(grad).all() for grad in active), name
    # AB2 updates X from the old/full RHS and leaves the terminal P update
    # unread by the X-only language-model head. Its final V force therefore has
    # no gradient, while earlier V forces and ExpMid's same-layer Pbar drift do.
    assert ab2_model.attn[-1].c_V.weight.grad is None
    assert ab2_model.attn[-2].c_V.weight.grad is not None
    assert exp_model.attn[-1].c_V.weight.grad is not None

    # Prefix-local matrix momenta are strictly causal: suffix changes cannot
    # alter any prefix logit. This is an approximation, not the global closure.
    changed = idx.clone()
    changed[:, 4:] = (changed[:, 4:] + 7) % cfg.vocab_size
    for name, model in (("exp_mid", exp_model), ("ab2", ab2_model)):
        prefix_a = model(idx)[0][:, :4]
        prefix_b = model(changed)[0][:, :4]
        assert_close(prefix_a, prefix_b, f"{name} suffix causality", atol=1e-8, rtol=1e-8)
    assert all(layer.causal for model in (exp_model, ab2_model) for layer in model.attn)


def verify_existing_schedule_start() -> None:
    """Regression: linear PE must start at sched.t0, never singular t=0."""
    cfg = ModelConfig(vocab_size=31, block_size=8, n_layer=2, n_head=1, n_embd=6, dropout=0.0)
    for attn_cls in ("presymp", "exp_euler"):
        model = LinAttnPresympModel(
            cfg, h=0.1, t0=1.0, eta_mode="log", eta_log_coef=3.0,
            eta_learnable=False, use_v0_init=False, attn_cls=attn_cls,
        )
        for layer in model.attn:
            layer.causal = False
        idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size), generator=torch.Generator().manual_seed(44))
        loss = model(idx, idx)[1]
        loss.backward()
        first_grad = model.attn[0].theta_hY.grad
        assert first_grad is not None and torch.isfinite(first_grad), attn_cls
        assert model.last_t_start == 1.0 and model.last_t_end > model.last_t_start


def verify_draft_text() -> None:
    draft = ROOT / "arXiv_v3.tex"
    if not draft.is_file():
        return
    text = draft.read_text()
    if "I_k \\big(-\\alpha(t_k) P^{(k)}" in text:
        raise AssertionError("draft still double-counts damping")
    assert "A_k S_k" not in text
    assert "\\mathbf Y \\in \\R^{N \\times d}" in text
    assert "Scope of the exact closure" in text


def main() -> None:
    verify_exact_closure()
    verify_damping_integral()
    verify_corrected_exp_mid()
    verify_ab2_and_models()
    verify_existing_schedule_start()
    verify_draft_text()
    print("PASS: arXiv-v3 reduced linear closure and new discretizations")
    print("PASS: exact global identities plus causal prefix reductions, gradients, and suffix invariance")


if __name__ == "__main__":
    main()
