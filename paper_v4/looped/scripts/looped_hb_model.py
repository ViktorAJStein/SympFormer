"""E063 two-block depth recurrence. Frozen native B/Y/H0/dual-LN R2 math.

No new attention, integration, normalization or velocity-reset rule. This module
must be imported alongside the frozen donor model/normalized_hb/train modules.
"""
import copy
from dataclasses import replace
import math

import torch
from torch import nn
from torch.nn import functional as F

from model import GPTModel
from normalized_hb import NormalizedHBConfig, NormalizedHBModel, attention_connection
from train import initialize_learned_v0_tables

METHODS = ('B', 'Y', 'H0', 'R2')
LAYOUTS = ('tied', 'untied')
INFERENCE_LOOPS = (1, 2, 4, 8, 16)
VERSION = 'e063_two_block_recurrence_v1'


def _norm(x):
    return float(x.detach().double().norm(dim=-1).mean())


def _cosine(x, y):
    x, y = x.detach().double(), y.detach().double()
    denom = x.norm(dim=-1) * y.norm(dim=-1)
    valid = denom > 0
    return (float(((x*y).sum(-1)[valid]/denom[valid]).mean()) if valid.any() else None,
            int(valid.sum()))


class LoopedDecoder(nn.Module):
    def __init__(self, cfg, method, layout, seed):
        super().__init__()
        if method not in METHODS or layout not in LAYOUTS:
            raise ValueError('Unknown E063 method/layout')
        if cfg.n_layer != 8 or cfg.dropout != 0 or cfg.bias:
            raise ValueError('E063 requires eight logical blocks, dropout0, biasFalse')
        self.cfg, self.method, self.layout = cfg, method, layout
        self.version = VERSION
        core_cfg = replace(cfg, n_layer=2)
        # Dedicated construction seed: no caller RNG mutation.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            if method == 'B':
                self.net = GPTModel(core_cfg)
            else:
                version = 'e048_r2_dual_ln_v1' if method == 'R2' else 'e046_identity_v1'
                self.net = NormalizedHBModel(core_cfg, NormalizedHBConfig(method, version))
                initialize_learned_v0_tables(self.net, seed)
        if layout == 'untied':
            core = list(self.net.blocks)
            self.net.blocks = nn.ModuleList([copy.deepcopy(core[j % 2]) for j in range(8)])
        self.net.cfg = cfg
        self.last_trace = []

    def check_coefficients(self):
        if self.method != 'B':
            for block in self.net.blocks:
                for name in ('beta1', 'beta2'):
                    value = float(getattr(block, name)().detach())
                    if not 0 < value < 1:
                        raise FloatingPointError('Strict E063 beta gate: '+name)
                if not all(torch.isfinite(p).all() for p in block.parameters()):
                    raise FloatingPointError('Nonfinite block parameter')

    @torch.no_grad()
    def _diagnostic(self, block, x, v, previous_force, j):
        row = dict(layer=j, loop=j//2, core_slot=j%2,
                   unique_block=j%2 if self.layout == 'tied' else j,
                   t_start=j+1., t_end=j+2., forward_step=1., state_norm=_norm(x),
                   damping_applicable=self.method != 'B')
        if self.method == 'B':
            force = block.attn(block.ln_1(x))
            row.update(force_norm=_norm(force))
        else:
            z = block.ln_x_attn(x + block.mu1()*v) if self.method == 'Y' else block.ln_x_attn(x)
            beta, gain = block.beta1(), block.gamma1()+block.force_gain_floor
            if self.method == 'R2':
                force, correction, proposal = attention_connection(block.attn, z, v, proposal=(beta, gain))
                off = block.ln_v(proposal)
                va = block.ln_v(proposal-correction)
                position = block.ln_v(proposal-.5*correction)
            else:
                force = block.attn(z)
                proposal = beta*v + gain*force
                correction = torch.zeros_like(v)
                off = va = position = block.ln_v(proposal)
            row.update(beta_attn=float(beta), beta_mlp=float(block.beta2()),
                       gain_attn=float(gain), gain_mlp=float(block.gamma2()+block.force_gain_floor),
                       mu_attn=float(block.mu1()) if self.method == 'Y' else 0.,
                       mu_mlp=float(block.mu2()),
                       nominal_rate_attn=-math.log(float(beta)),
                       nominal_rate_mlp=-math.log(float(block.beta2())),
                       nesterov_reference=3./(j+1.), velocity_norm=_norm(v),
                       force_norm=_norm(force), correction_norm=_norm(correction),
                       post_ln_velocity_correction_norm=_norm(va-off),
                       post_ln_position_correction_norm=_norm(position-off),
                       post_ln_position_cosine=_cosine(position, off)[0],
                       post_ln_position_cosine_valid=_cosine(position, off)[1],
                       velocity_force_cosine=_cosine(v, force)[0],
                       velocity_force_cosine_valid=_cosine(v, force)[1])
        row['successive_force_cosine'] = None if previous_force is None else _cosine(previous_force, force)[0]
        row['successive_force_cosine_valid'] = 0 if previous_force is None else _cosine(previous_force, force)[1]
        return row, force.detach()

    def forward(self, idx, targets=None, global_step=None, *, loops=4, capture=False):
        if type(loops) is not int or loops not in INFERENCE_LOOPS:
            raise ValueError('Undeclared loop count')
        if self.layout == 'untied' and loops != 4:
            raise ValueError('Untied model has exactly eight trained blocks')
        if self.training and loops != 4:
            raise ValueError('Training depth is fixed at four loops')
        if idx.ndim != 2 or idx.shape[1] > self.cfg.block_size or idx.shape[1] == 0:
            raise ValueError('Invalid sequence shape')
        if capture and torch.is_grad_enabled():
            raise ValueError('Diagnostics require no_grad')
        pos = torch.arange(idx.shape[1], device=idx.device)
        x = self.net.drop(self.net.tok_emb(idx)+self.net.pos_emb(pos)[None])
        v = None if self.method == 'B' else self.net.drop(self.net.tok_v0_emb(idx)+self.net.pos_v0_emb(pos)[None])
        self.last_trace = []
        previous_force = None
        for j in range(2*loops):
            block = self.net.blocks[j % 2 if self.layout == 'tied' else j]
            if capture:
                row, previous_force = self._diagnostic(block, x, v, previous_force, j)
            if self.method == 'B':
                x = block(x)
            else:
                x, v = block(x, v)
            if capture:
                row['next_state_norm'] = _norm(x)
                row['next_velocity_norm'] = None if v is None else _norm(v)
                self.last_trace.append(row)
        logits = self.net.lm_head(self.net.ln_f(x))
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss
