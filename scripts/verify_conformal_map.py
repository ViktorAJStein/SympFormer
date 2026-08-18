#!/usr/bin/env python3
"""Verify the ideal kick-damp-drift conformal-symplectic identity.

This deliberately excludes LayerNorm, MLPs, and the row-local learned oracle.
It verifies the structural identity that motivates the discretization, not an
exact identity for the full language-model block.
"""

import torch


def main():
    torch.manual_seed(20260727)
    dtype = torch.float64
    d = 4
    raw_m = torch.randn(d, d, dtype=dtype)
    raw_k = torch.randn(d, d, dtype=dtype)
    mass_inv = raw_m.T @ raw_m + 0.2 * torch.eye(d, dtype=dtype)
    stiffness = raw_k.T @ raw_k + 0.2 * torch.eye(d, dtype=dtype)
    h = torch.tensor(0.13, dtype=dtype)
    sigma = torch.tensor(0.81, dtype=dtype)

    def step(z):
        q, p = z[:d], z[d:]
        p_next = sigma * (p - h * (stiffness @ q))
        q_next = q + h * (mass_inv @ p_next)
        return torch.cat((q_next, p_next))

    z = torch.randn(2 * d, dtype=dtype, requires_grad=True)
    jac = torch.autograd.functional.jacobian(step, z)
    eye = torch.eye(d, dtype=dtype)
    zero = torch.zeros_like(eye)
    omega = torch.cat((torch.cat((zero, eye), dim=1), torch.cat((-eye, zero), dim=1)), dim=0)
    residual = jac.T @ omega @ jac - sigma * omega
    max_error = float(residual.abs().max())
    det_error = abs(float(torch.linalg.det(jac)) - float(sigma**d))
    print(f"max_pullback_error={max_error:.3e}")
    print(f"determinant_error={det_error:.3e}")
    if max_error > 1e-11 or det_error > 1e-11:
        raise AssertionError("ideal kick-damp-drift identity failed")
    print("PASS: ideal map is conformally symplectic; no such claim is made for the full normalized LM block.")


if __name__ == "__main__":
    main()
