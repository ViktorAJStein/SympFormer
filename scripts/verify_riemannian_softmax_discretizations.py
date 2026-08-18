#!/usr/bin/env python3
"""Verify E025 metric identities, local connection, model gradients, and causality."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import CausalRiemannianNAGLayer, CausalRiemannianNAGModel, ModelConfig, causal_mask  # noqa: E402
from train import _parameter_buckets  # noqa: E402

DTYPE = torch.float64


def close(a, b, label, atol=2e-10, rtol=2e-10):
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        raise AssertionError(f"{label}: max error {(a-b).abs().max().item():.3e}")


def verify_metric_dual():
    g = torch.Generator().manual_seed(2)
    n, d = 4, 3
    X = torch.randn(n, d, generator=g, dtype=DTYPE)
    raw = torch.randn(d, d, generator=g, dtype=DTYPE)
    B = raw @ raw.T + 0.4 * torch.eye(d, dtype=DTYPE)
    A = torch.eye(d, dtype=DTYPE)
    M = torch.exp(X @ A @ X.T)
    z = M.sum(dim=-1)
    velocity = torch.randn(n, d, generator=g, dtype=DTYPE)
    momentum = (z[:, None] * velocity @ torch.linalg.inv(B)) / n
    recovered = n * momentum @ B / z[:, None]
    close(recovered, velocity, "metric dual")


def verify_local_connection():
    g = torch.Generator().manual_seed(3)
    d, prefix = 3, 4
    x = torch.randn(d, generator=g, dtype=DTYPE, requires_grad=True)
    fixed = torch.randn(prefix - 1, d, generator=g, dtype=DTYPE)
    raw = torch.randn(d, d, generator=g, dtype=DTYPE)
    A = raw @ raw.T + 0.3 * torch.eye(d, dtype=DTYPE)
    raw = torch.randn(d, d, generator=g, dtype=DTYPE)
    B = raw @ raw.T + 0.5 * torch.eye(d, dtype=DTYPE)
    u = torch.randn(d, generator=g, dtype=DTYPE)

    keys = torch.cat((fixed, x[None]), dim=0)
    z = torch.exp((keys @ A @ x)).sum()
    grad_z = torch.autograd.grad(z, x, create_graph=True)[0]
    a = grad_z / z
    Binv = torch.linalg.inv(B)
    closed = torch.dot(u, a) * u - 0.5 * torch.dot(u, Binv @ u) * (B @ a)

    # Independent coordinate formula G^{-1}[(DG[u])u - 1/2 grad(u^TGu)].
    G = (z / prefix) * Binv
    directional_z = torch.dot(grad_z, u)
    first = (directional_z / prefix) * (Binv @ u)
    kinetic = torch.dot(u, G @ u)
    grad_kinetic = torch.autograd.grad(kinetic, x)[0]
    direct = torch.linalg.solve(G, first - 0.5 * grad_kinetic)
    close(closed, direct, "local Christoffel contraction")


def verify_layernorm_partition_gradient():
    cfg = ModelConfig(vocab_size=17, block_size=4, n_layer=1, n_head=1, n_embd=3, dropout=0.0)
    layer = CausalRiemannianNAGLayer(cfg).to(dtype=DTYPE)
    Q = torch.randn(1, 4, 3, generator=torch.Generator().manual_seed(13), dtype=DTYPE, requires_grad=True)
    Qn = layer.ln1(Q)
    A, _B = layer.matrices()
    QA = Qn @ A
    scores = (QA @ Qn.transpose(-1, -2)) * layer.scale
    scores = scores.masked_fill(causal_mask(4, Q.device).unsqueeze(0), float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    analytic = layer._grad_log_partition(Q, weights, QA)
    direct_rows = []
    for i in range(4):
        log_z_i = torch.logsumexp(scores[:, i, : i + 1], dim=-1).sum()
        full_grad = torch.autograd.grad(log_z_i, Q, retain_graph=True)[0]
        direct_rows.append(full_grad[:, i])
    direct = torch.stack(direct_rows, dim=1)
    close(analytic, direct, "LayerNorm-chained prefix partition gradient", atol=3e-10, rtol=3e-10)


def verify_models():
    cfg = ModelConfig(vocab_size=37, block_size=8, n_layer=3, n_head=1, n_embd=6, dropout=0.0)
    torch.manual_seed(5)
    no_conn = CausalRiemannianNAGModel(cfg, include_connection=False)
    torch.manual_seed(5)
    conn = CausalRiemannianNAGModel(cfg, include_connection=True)
    conn.load_state_dict(no_conn.state_dict())
    idx = torch.randint(0, cfg.vocab_size, (2, 8), generator=torch.Generator().manual_seed(6))
    changed = idx.clone(); changed[:, 4:] = (changed[:, 4:] + 9) % cfg.vocab_size
    targets = torch.randint(0, cfg.vocab_size, idx.shape, generator=torch.Generator().manual_seed(7))
    for name, model in (("no_connection", no_conn), ("connection", conn)):
        model.eval()
        close(model(idx)[0][:, :4], model(changed)[0][:, :4], f"{name} suffix causality", atol=1e-8, rtol=1e-8)
        model.train()
        loss = model(idx, targets)[1]
        assert torch.isfinite(loss)
        loss.backward()
        active = [p.grad for p in model.parameters() if p.grad is not None]
        assert active and all(torch.isfinite(item).all() for item in active)
        for layer in model.layers:
            A, B = layer.matrices()
            assert torch.linalg.eigvalsh(A).min() > 0
            assert torch.linalg.eigvalsh(B).min() > 0
    if torch.allclose(conn(idx)[0], no_conn(idx)[0], atol=1e-8, rtol=1e-8):
        raise AssertionError("connection and no-connection routes are not distinct")
    _embeddings, scalar_norm, matrices = _parameter_buckets(conn)
    scalar_ids = {id(param) for _name, param in scalar_norm}
    matrix_ids = {id(param) for param in matrices}
    assert id(conn.layers[0].theta_tau) in scalar_ids
    assert id(conn.layers[0].raw_A) in matrix_ids and id(conn.layers[0].raw_B) in matrix_ids


def verify_draft():
    text = (ROOT / "arXiv_v3.tex").read_text()
    required = [
        r"\delta_{X_j(t)}", r"g_{\mathbf X}^{\flat}(\dot{\mathbf{X}})",
        r"1 - \tau \alpha(t_k)", r"\beta_k=\exp",
        r"\mathbf{Q}^{(k)} = \mathbf{X}^{(k)} + \beta_k",
    ]
    for token in required:
        assert token in text, token
    assert r"{\color{red}$\beta_k = ??$}" not in text
    assert r"{\color{red}TODO: unfinished}" not in text


def main():
    verify_metric_dual()
    verify_local_connection()
    verify_layernorm_partition_gradient()
    verify_models()
    verify_draft()
    print("PASS: metric dual, local Christoffel contraction, SPD geometry, gradients, and causality")
    print("PASS: corrected softmax Euler/ExpEuler/Nesterov draft formulas")


if __name__ == "__main__":
    main()
