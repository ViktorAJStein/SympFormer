"""E062 isolated signed-linear HB, fixed factorwise-ridge connection proxy.

Requires the frozen native model.py scaffold. No canonical-momentum closure,
LN pullback or literal multihead Levi--Civita connection is asserted.
"""
from dataclasses import dataclass, replace
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from model import GPTModel, YuriiFormerModel

def make_model(cfg, mode, seed, chunk=32):
    """One native Yurii donor defines every common initial tensor, not RNG order."""
    if mode not in MODES: raise ValueError(mode)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        donor = YuriiFormerModel(replace(cfg, yurii_force_gain_floor=1e-6), use_v0_init=True)
        torch.manual_seed(seed)
        if mode in ('B_S', 'B_L'):
            m = GPTModel(cfg) if mode == 'B_S' else LinearGPTModel(cfg, chunk=chunk)
        else:
            m = LinearHBModel(cfg, LinearHBConfig(mode=mode, chunk=chunk))
        source = dict(donor.named_parameters())
        with torch.no_grad():
            for name, value in m.named_parameters():
                mapped = name.replace('.ln_1.', '.ln_x_attn.').replace('.ln_2.', '.ln_x_mlp.')
                assert mapped in source and source[mapped].shape == value.shape, name
                value.copy_(source[mapped])
    m.e062_mode = mode
    return m


VERSION = 'e062_prefix_ridge_v1'
MODES = ('B_S', 'Y_S', 'B_L', 'Y_L', 'H0_L', 'H1_L', 'R2_L')


@dataclass(frozen=True)
class LinearHBConfig:
    mode: str = 'H0_L'
    version: str = VERSION
    ridge: float = 1e-3
    chunk: int = 32
    checkpoint_chunks: bool = True

    def __post_init__(self):
        if self.mode not in MODES[1:2]+MODES[3:] or self.version != VERSION:
            raise ValueError('Unsupported linear HB mode/version')
        if self.ridge != 1e-3 or self.chunk < 1:
            raise ValueError('Frozen E062 ridge1e-3 and positive execution chunk required')


def project(attn, z):
    b, n, d = z.shape; h, r = attn.n_head, attn.head_dim
    q, k, values = attn.c_attn(z).split(d, dim=-1)
    split = lambda t: t.reshape(b, n, h, r).transpose(1, 2)
    return split(q)/r**.25, split(k)/r**.25, split(values)


def merge(t):
    return t.transpose(1, 2).contiguous().flatten(2)


def linear_force(attn, q, k, values, chunk=32, state=None, offset=0):
    """Prefix sufficient statistics; state is per sequence AND logical layer."""
    history = torch.zeros_like(k[..., 0, :, None]*values[..., 0, None, :]) if state is None else state
    outputs = []
    for start in range(0, q.shape[2], chunk):
        stop = min(start+chunk, q.shape[2])
        sums = (k[:, :, start:stop, :, None]*values[:, :, start:stop, None, :]).cumsum(2)+history[:, :, None]
        result = torch.matmul(q[:, :, start:stop, None, :], sums).squeeze(-2)
        count = torch.arange(offset+start+1, offset+stop+1, device=q.device, dtype=q.dtype)
        outputs.append(result/count[None, None, :, None]); history = sums[:, :, -1]
    return attn.c_proj(merge(torch.cat(outputs, 2))), history


def _proxy_chunk(q, k, u, dq, dk, *history, ridge):
    # Five independent statistics: G_Q,G_K,Q^T U,K^T dotK,K^T Q.
    terms = (q[..., :, None]*q[..., None, :], k[..., :, None]*k[..., None, :],
             q[..., :, None]*u[..., None, :], k[..., :, None]*dk[..., None, :],
             k[..., :, None]*q[..., None, :])
    gq, gk, qu, kd, kq = [t.cumsum(2)+old[:, :, None] for t, old in zip(terms, history)]
    eye = torch.eye(q.shape[-1], dtype=q.dtype, device=q.device)
    cq = torch.linalg.cholesky(gq+ridge*eye); ck = torch.linalg.cholesky(gk+ridge*eye)
    e = torch.cholesky_solve(qu, cq)
    solved = torch.cholesky_solve(torch.cat((kd, kq, dq[..., None]), -1), ck)
    r = q.shape[-1]; j, t, kdq = solved.split((r, r, 1), -1)
    # dq L = dq (K^T K) P_K; using the symmetric solve avoids explicitly
    # inverting anything and retains the ridge factor omitted in the old note.
    dq_l = torch.matmul(gk, kdq).transpose(-1, -2)
    first = -torch.matmul(dq_l, e).squeeze(-2)
    second = -torch.matmul(torch.matmul(q[..., None, :], j.transpose(-1, -2)), e).squeeze(-2)
    third_head = torch.matmul(torch.matmul(u[..., None, :], e.transpose(-1, -2)), t).squeeze(-2)
    return first+second, third_head, *[s[:, :, -1] for s in (gq, gk, qu, kd, kq)]


def linear_proxy(attn, q, k, direction, hb, state=None, capture=False):
    b, n, d = direction.shape; h, r = attn.n_head, attn.head_dim
    projected = F.linear(direction, attn.c_attn.weight[:2*d])
    dq, dk = [t.reshape(b, n, h, r).transpose(1, 2)/r**.25 for t in projected.split(d, -1)]
    u = direction[:, None].expand(-1, h, -1, -1)
    if state is None:
        shapes = ((r, r), (r, r), (r, d), (r, r), (r, r))
        state = tuple(q.new_zeros(b, h, *shape) for shape in shapes)
    wk = attn.c_attn.weight[d:2*d].reshape(h, r, d)/r**.25
    results, spectra = [], []
    for start in range(0, n, hb.chunk):
        stop = min(start+hb.chunk, n)
        inputs = (q[:, :, start:stop], k[:, :, start:stop], u[:, :, start:stop],
                  dq[:, :, start:stop], dk[:, :, start:stop], *state)
        fn = lambda *xs: _proxy_chunk(*xs, ridge=hb.ridge)
        if hb.checkpoint_chunks and torch.is_grad_enabled() and not capture:
            out = checkpoint(fn, *inputs, use_reentrant=False)
        else:
            out = fn(*inputs)
        two, third, *next_state = out
        results.append(two+torch.einsum('bhnr,hrd->bhnd', third, wk))
        if capture:
            with torch.no_grad():
                qpart, kpart = inputs[:2]
                gq = (qpart[..., :, None]*qpart[..., None, :]).cumsum(2)+state[0][:, :, None]
                gk = (kpart[..., :, None]*kpart[..., None, :]).cumsum(2)+state[1][:, :, None]
                spectra.append(torch.stack((torch.linalg.eigvalsh(gq), torch.linalg.eigvalsh(gk)), -1).detach().cpu())
        state = tuple(next_state)
    connection = torch.cat(results, 2).mean(1)
    return connection, state, torch.cat(spectra, 2) if spectra else None


class LinearAttention(nn.Module):
    def __init__(self, donor, chunk=32):
        super().__init__(); self.chunk = chunk
        self.n_head, self.head_dim = donor.n_head, donor.head_dim
        for name, module in donor.named_children():
            self.add_module(name, module)

    def forward(self, z):
        return linear_force(self, *project(self, z), chunk=self.chunk)[0]


class LinearGPTModel(GPTModel):
    def __init__(self, cfg, *, chunk=32, **kwargs):
        if cfg.bias or cfg.dropout or kwargs.get('share_layers') or kwargs.get('no_mlp') or kwargs.get('capacity_control'):
            raise ValueError('E062 linear baseline requires native unshared residual blocks, MLP, no dropout/bias')
        super().__init__(cfg, **kwargs)
        for block in self.blocks:
            block.attn = LinearAttention(block.attn, chunk=chunk)

    @torch.no_grad()
    def stream_step(self, token, state=None):
        state = dict(pos=0, layers=[None]*len(self.blocks)) if state is None else state
        pos = state['pos']
        if token.ndim != 2 or token.shape[1] != 1: raise ValueError('Streaming requires one token per sequence')
        if state.get('batch', token.shape[0]) != token.shape[0]: raise ValueError('Streaming batch changed: reset state')
        if not 0 <= pos < self.cfg.block_size: raise ValueError('Streaming positional table exhausted')
        x = self.tok_emb(token)+self.pos_emb.weight[pos][None, None]
        caches = []
        for block, cache in zip(self.blocks, state['layers']):
            q, k, values = project(block.attn, block.ln_1(x))
            force, cache = linear_force(block.attn, q, k, values, state=cache, offset=pos)
            x = x+force; x = x+block.mlp(block.ln_2(x)); caches.append(cache)
        return self.lm_head(self.ln_f(x)), dict(pos=pos+1, layers=caches, batch=token.shape[0])


class LinearHBBlock(nn.Module):
    def __init__(self, donor, hb):
        super().__init__(); self.hb = hb; self.mode = hb.mode
        self.force_gain_floor = donor.force_gain_floor
        for name, module in donor.named_children():
            if name == 'mu1' and hb.mode not in ('Y_S', 'Y_L'): continue
            self.add_module(name, module)
        self.capture_diagnostics = False; self.capture_states = False
        self.last_diagnostics = {}; self.last_states = {}; self.last_spectra = None
        self.work_counts = dict(attention_calls=0, connection_calls=0, velocity_norm_calls=0)
        self.connection_enabled = hb.mode in ('H1_L', 'R2_L')

    def half(self, x, v, cache=None, offset=0, capture=False):
        z = self.ln_x_attn(x+self.mu1()*v) if self.mode in ('Y_S', 'Y_L') else self.ln_x_attn(x)
        if self.mode == 'Y_S':
            force = self.attn(z); force_state = None; q = k = None
        else:
            q, k, values = project(self.attn, z)
            force, force_state = linear_force(self.attn, q, k, values, self.hb.chunk,
                                              None if cache is None else cache['force'], offset)
        proposal = self.beta1()*v+(self.gamma1()+self.force_gain_floor)*force
        if self.connection_enabled:
            c, proxy_state, spectra = linear_proxy(self.attn, q, k, proposal if self.mode == 'R2_L' else v,
                self.hb, None if cache is None else cache['proxy'], capture=capture)
        else:
            c = torch.zeros_like(proposal); proxy_state = spectra = None
        raw_v = proposal-c; raw_x = proposal-.5*c if self.mode == 'R2_L' else raw_v
        va = self.ln_v(raw_v)
        position = self.ln_v(raw_x) if self.mode == 'R2_L' else va
        return x+position, va, proposal, c, force, raw_x, raw_v, dict(force=force_state, proxy=proxy_state), spectra

    def mlp_half(self, xa, va):
        f = self.mlp(self.ln_x_mlp(xa+self.mu2()*va))
        vn = self.ln_v(self.beta2()*va+(self.gamma2()+self.force_gain_floor)*f)
        return xa+vn, vn

    def forward(self, x, v, noise_std=0., noise_loc='v'):
        if noise_std != 0 or noise_loc not in ('v', 'dx', 'xin'): raise ValueError('No E062 noise')
        xa, va, w, c, force, raw_x, raw_v, _, spectra = self.half(x, v, capture=self.capture_diagnostics)
        xn, vn = self.mlp_half(xa, va)
        self.work_counts['attention_calls'] += 1
        self.work_counts['connection_calls'] += int(self.connection_enabled)
        self.work_counts['velocity_norm_calls'] += 3 if self.mode == 'R2_L' else 2
        if self.capture_states:
            self.last_states = {s: (a.detach().cpu(), b.detach().cpu()) for s, a, b in
                [('entry', x, v), ('raw_attention_proposal', x+raw_x, raw_v),
                 ('post_attention', xa, va), ('post_mlp', xn, vn)]}
        if self.capture_diagnostics:
            norm = lambda t: float(t.detach().double().norm(dim=-1).mean())
            off = self.ln_v(w)
            force_size=norm(force)
            self.last_diagnostics = dict(force_norm=force_size, proposal_norm=norm(w), connection_norm=norm(c),
                force_ratio_undefined=int(force_size==0),
                connection_force_ratio=norm(c)/force_size if force_size else float('nan'),
                post_ln_correction_norm=norm(va-off), position_transport_gap=norm((xa-x)-va))
            self.last_spectra = spectra
        return xn, vn


class LinearHBModel(YuriiFormerModel):
    def __init__(self, cfg, hb):
        if cfg.bias or cfg.dropout or cfg.yurii_force_gain_floor != 1e-6:
            raise ValueError('E062 inertial models require biasFalse/dropout0/floor1e-6')
        super().__init__(cfg, use_v0_init=True)
        self.hb = hb
        self.blocks = nn.ModuleList([LinearHBBlock(block, hb) for block in self.blocks])

    def work_counts(self):
        return {k: sum(b.work_counts[k] for b in self.blocks) for k in self.blocks[0].work_counts}

    def hb_dynamics(self):
        result = []
        for layer, b in enumerate(self.blocks):
            ba, bm = float(b.beta1().detach()), float(b.beta2().detach())
            row = dict(layer=layer, mode=self.hb.mode, version=self.hb.version,
                beta_attn=ba, beta_mlp=bm, gain_attn=float((b.gamma1()+b.force_gain_floor).detach()),
                gain_mlp=float((b.gamma2()+b.force_gain_floor).detach()),
                mu_attn=float(b.mu1().detach()) if hasattr(b, 'mu1') else 0., mu_mlp=float(b.mu2().detach()),
                forward_step=1., depth_time=layer+1, nominal_alpha_attn=-math.log(ba),
                nominal_alpha_mlp=-math.log(bm), nesterov_reference=3./(layer+1),
                connection_enabled=int(b.connection_enabled), ridge=self.hb.ridge,
                velocity_units='displacement', proxy_base='position_LN_raw_direction',
                diagnostics_available=int(bool(b.last_diagnostics)),force_ratio_undefined=0,
                force_norm=float('nan'),proposal_norm=float('nan'),connection_norm=float('nan'),connection_force_ratio=float('nan'),
                post_ln_correction_norm=float('nan'),position_transport_gap=float('nan'))
            row.update(b.last_diagnostics); result.append(row)
        return result

    @torch.no_grad()
    def stream_step(self, token, state=None):
        if self.hb.mode == 'Y_S': raise ValueError('Native softmax Y is not a linear recurrent decoder')
        state = dict(pos=0, layers=[None]*len(self.blocks)) if state is None else state
        pos = state['pos']
        if token.ndim != 2 or token.shape[1] != 1: raise ValueError('Streaming requires one token per sequence')
        if state.get('batch', token.shape[0]) != token.shape[0]: raise ValueError('Streaming batch changed: reset state')
        if not 0 <= pos < self.cfg.block_size: raise ValueError('Streaming positional table exhausted')
        x = self.tok_emb(token)+self.pos_emb.weight[pos][None, None]
        v = self.tok_v0_emb(token)+self.pos_v0_emb.weight[pos][None, None]
        caches = []
        for b, cache in zip(self.blocks, state['layers']):
            xa, va, *other = b.half(x, v, cache=cache, offset=pos)
            caches.append(other[-2]); x, v = b.mlp_half(xa, va)
        return self.lm_head(self.ln_f(x)), dict(pos=pos+1, layers=caches, batch=token.shape[0])
