"""Native-scaffold normalized HB; explicit causal, inverse-free connection proxy.

The proxy is NOT the Levi-Civita connection of arbitrary masked multihead
attention. Version e046_identity_v1 fixes H=I, mean directional score drift,
and summed native projected values. No extra trainable metric parameters.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from model import ModelConfig, YuriiFormerModel, causal_mask


@dataclass(frozen=True)
class NormalizedHBConfig:
    mode: str = 'Y'
    version: str = 'e046_identity_v1'

    def __post_init__(self):
        valid = (self.mode in ('Y', 'H0', 'H1') and self.version == 'e046_identity_v1') or (
            self.mode == 'R2' and self.version == 'e048_r2_dual_ln_v1')
        if not valid:
            raise ValueError('Unsupported normalized HB mode/version')


def attention_connection(attn, z, velocity, *, proposal=None):
    """Native force and fixed identity-metric proxy, reusing a single softmax.

For head r, a_r=sum_j P_r,ij D_r,ij, s_i=||velocity_i||^2.
Gamma_i=mean_r(a_r,i)*velocity_i - (s_i*F_i + sum_r P_r(s*w_r)O_r)/2.
Zero dropout/bias are required by this version's explicit definition.
"""
    batch, length, width = z.shape
    heads, dim = attn.n_head, attn.head_dim
    q, k, w = attn.c_attn(z).split(width, dim=2)
    split = lambda x: x.view(batch, length, heads, dim).transpose(1, 2)
    q, k, w = map(split, (q, k, w))
    score = (q @ k.transpose(-2, -1)) * attn.scale
    p = F.softmax(score.masked_fill(causal_mask(length, z.device)[None, None], -torch.inf), dim=-1)
    merge = lambda x: x.transpose(1, 2).contiguous().view(batch, length, width)
    force = attn.c_proj(merge(p @ w))
    if proposal is not None:
        # R2 uses the proposed displacement, reusing the SAME force/softmax graph.
        beta, gain = proposal
        velocity = beta * velocity + gain * force
    # No derivative through LN: the displayed architecture applies raw V at Z.
    dq, dk = F.linear(velocity, attn.c_attn.weight[:2 * width]).split(width, dim=2)
    dq, dk = map(split, (dq, dk))
    directional = (dq @ k.transpose(-2, -1) + q @ dk.transpose(-2, -1)) * attn.scale
    drift = (p * directional).sum(-1).mean(1)
    s = velocity.square().sum(-1)
    weighted_force = attn.c_proj(merge(p @ (w * s[:, None, :, None])))
    connection = drift[..., None] * velocity - .5 * (s[..., None] * force + weighted_force)
    return (force, connection, velocity) if proposal is not None else (force, connection)


class NormalizedHBBlock(nn.Module):
    def __init__(self, donor, mode):
        super().__init__()
        self.mode = mode
        self.no_mlp = False
        self.force_gain_floor = donor.force_gain_floor
        # Transfer the already initialized native modules; no extra RNG draws.
        for name, module in donor.named_children():
            if name == 'mu1' and mode != 'Y':
                continue
            self.add_module(name, module)
        self.capture_diagnostics = False
        self.last_diagnostics = {}
        self.work_counts = dict(attention_calls=0, score_calls=0, connection_calls=0,
                                directional_score_calls=0, extra_value_aggregations=0,
                                recompute_calls=0)
        if mode == 'R2':
            self.work_counts['velocity_norm_calls'] = 0

    def forward(self, x, v, noise_std=0., noise_loc='v'):
        if noise_std != 0 or noise_loc not in ('dx', 'v', 'xin'):
            raise ValueError('Normalized HB version does not permit noise')
        z = self.ln_x_attn(x + self.mu1() * v) if self.mode == 'Y' else self.ln_x_attn(x)
        beta, gain = self.beta1(), self.gamma1() + self.force_gain_floor
        proposal = None
        if self.mode in ('H1', 'R2'):
            if self.mode == 'R2':
                force, connection, proposal = attention_connection(self.attn, z, v, proposal=(beta, gain))
                raw = proposal - connection
            else:
                force, connection = attention_connection(self.attn, z, v)
                raw = beta * v + gain * force - connection
            for key in ('connection_calls', 'directional_score_calls', 'extra_value_aggregations'):
                self.work_counts[key] += 1
        else:
            force = self.attn(z)
            connection = None
            raw = beta * v + gain * force
        self.work_counts['attention_calls'] += 1
        self.work_counts['score_calls'] += 1
        va = self.ln_v(raw)
        # R2 retains distinct half-corrected position and fully transported velocity.
        # This is an explicit dual-LN architecture, not a second-order ODE solver.
        position_delta = self.ln_v(proposal - .5 * connection) if self.mode == 'R2' else va
        xa = x + position_delta
        # Unchanged Yurii MLP and the SAME velocity LayerNorm module.
        fm = self.mlp(self.ln_x_mlp(xa + self.mu2() * va))
        vn = self.ln_v(self.beta2() * va + (self.gamma2() + self.force_gain_floor) * fm)
        xn = xa + vn
        if self.mode == 'R2':
            self.work_counts['velocity_norm_calls'] += 3
        if self.capture_diagnostics:
            with torch.no_grad():
                norm = lambda t: float(t.detach().norm(dim=-1).mean())
                fn = norm(gain * force)
                cn = 0. if connection is None else norm(connection)
                self.last_diagnostics = dict(incoming_velocity_norm=norm(v), force_norm=norm(force),
                    gained_force_norm=fn, connection_norm=cn, connection_force_ratio=cn / max(fn, 1e-30),
                    retained_velocity_norm=norm(beta * v), raw_attention_velocity_norm=norm(raw),
                    attention_velocity_norm=norm(va), mlp_force_norm=norm(fm), stored_velocity_norm=norm(vn),
                    evaluated_beta_attn=float(beta), evaluated_gain_attn=float(gain),
                    position_increment_norm=norm(position_delta),
                    position_velocity_gap_norm=norm(position_delta-va),
                    velocity_norm_calls_per_block=3 if self.mode == 'R2' else 2)
        return xn, vn


class NormalizedHBModel(YuriiFormerModel):
    def __init__(self, cfg: ModelConfig, hb: NormalizedHBConfig):
        if cfg.dropout != 0 or cfg.bias or cfg.yurii_force_gain_floor != 1e-6:
            raise ValueError('Normalized HB requires dropout0, bias=False and Yurii gain floor1e-6')
        super().__init__(cfg, use_v0_init=True)
        self.hb = hb
        self.blocks = nn.ModuleList([NormalizedHBBlock(block, hb.mode) for block in self.blocks])

    def work_counts(self):
        return {key: sum(block.work_counts[key] for block in self.blocks)
                for key in self.blocks[0].work_counts}

    def hb_dynamics(self):
        rows = []
        for layer, block in enumerate(self.blocks):
            beta_a, beta_m = float(block.beta1().detach()), float(block.beta2().detach())
            row = dict(layer=layer, mode=self.hb.mode, version=self.hb.version,
                beta_attn=beta_a, beta_mlp=beta_m,
                gain_attn=float((block.gamma1() + block.force_gain_floor).detach()),
                gain_mlp=float((block.gamma2() + block.force_gain_floor).detach()),
                mu_attn=float(block.mu1().detach()) if self.hb.mode == 'Y' else 0.,
                mu_mlp=float(block.mu2().detach()), connection_enabled=int(self.hb.mode in ('H1', 'R2')),
                depth_time=layer+1, log_retention_attn=-math.log(beta_a),
                log_retention_mlp=-math.log(beta_m), nesterov_depth_reference=3./(layer+1),
                quadratic_form='identity_unscaled', velocity_units='displacement')
            row.update(block.last_diagnostics)
            for name in ('beta1', 'gamma1', 'mu2', 'beta2', 'gamma2'):
                grad = getattr(block, name).raw.grad
                row[name + '_gradient_abs'] = float(grad.detach().abs()) if grad is not None else 0.
            rows.append(row)
        return rows
