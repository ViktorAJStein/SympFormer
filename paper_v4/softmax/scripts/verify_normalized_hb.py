"""Pure independent head/row reference, extracted without arithmetic changes."""
import torch
from torch.nn import functional as F

def reference_attention(attn, z, v):
    b, n, d = z.shape
    e, m = attn.head_dim, attn.n_head
    w = attn.c_attn.weight
    outputs, corrections = [], []
    for i in range(n):
        force = torch.zeros_like(v[:, i])
        weighted = torch.zeros_like(force)
        drift = torch.zeros(b, device=z.device, dtype=z.dtype)
        for r in range(m):
            sl = slice(r*e, (r+1)*e)
            q = F.linear(z[:, i], w[sl])
            k = F.linear(z[:, :i+1], w[d+r*e:d+(r+1)*e])
            values = F.linear(z[:, :i+1], w[2*d+r*e:2*d+(r+1)*e])
            dq = F.linear(v[:, i], w[sl])
            dk = F.linear(v[:, :i+1], w[d+r*e:d+(r+1)*e])
            p = torch.softmax(torch.einsum('be,bje->bj', q, k)*attn.scale, dim=-1)
            dr = (torch.einsum('be,bje->bj', dq, k) + torch.einsum('be,bje->bj', q, dk))*attn.scale
            drift = drift + (p*dr).sum(-1)/m
            head = torch.einsum('bj,bje->be', p, values)
            force = force + F.linear(head, attn.c_proj.weight[:, sl])
            weighted_head = torch.einsum('bj,bj,bje->be', p, v[:, :i+1].square().sum(-1), values)
            weighted = weighted + F.linear(weighted_head, attn.c_proj.weight[:, sl])
        outputs.append(force)
        corrections.append(drift[:, None]*v[:, i] - .5*(v[:, i].square().sum(-1, keepdim=True)*force + weighted))
    return torch.stack(outputs, 1), torch.stack(corrections, 1)
