"""E061 full-attention-look-ahead extension of frozen native normalized HB.

Old Y/H0/H1/R2 paths delegate without modification. The new modes reuse the
same causal force/proxy graph at LN_A(X+mu_A V); neither LN nor the look-ahead
map is differentiated inside the identity-form connection proxy.
"""
from dataclasses import dataclass
import torch
from torch import nn
from model import YuriiFormerModel
from normalized_hb import (
    NormalizedHBConfig as LegacyConfig, NormalizedHBModel as LegacyModel,
    NormalizedHBBlock as LegacyBlock, attention_connection,
)

VERSION = 'e061_full_attention_lookahead_v1'
NEW_MODES = ('H1_LA', 'R2_LA')


@dataclass(frozen=True)
class NormalizedHBConfig(LegacyConfig):
    def __post_init__(self):
        if self.mode in NEW_MODES:
            if self.version != VERSION:
                raise ValueError('Unsupported normalized HB mode/version')
        else:
            super().__post_init__()


class LookaheadBlock(LegacyBlock):
    def __init__(self, donor, mode):
        super().__init__(donor, 'Y')  # Retain the already initialized native mu1.
        if mode not in NEW_MODES:
            raise ValueError(mode)
        self.mode = mode
        self.work_counts['velocity_norm_calls'] = 0

    def forward(self, x, v, noise_std=0., noise_loc='v'):
        if noise_std != 0 or noise_loc not in ('dx', 'v', 'xin'):
            raise ValueError('Normalized HB version does not permit noise')
        z = self.ln_x_attn(x + self.mu1()*v)
        beta, gain = self.beta1(), self.gamma1()+self.force_gain_floor
        if self.mode == 'R2_LA':
            force, connection, W = attention_connection(self.attn, z, v, proposal=(beta, gain))
        else:
            force, connection = attention_connection(self.attn, z, v)
            W = beta*v+gain*force
        va = self.ln_v(W-connection)
        position = self.ln_v(W-.5*connection) if self.mode == 'R2_LA' else va
        xa = x+position
        fm = self.mlp(self.ln_x_mlp(xa+self.mu2()*va))
        vn = self.ln_v(self.beta2()*va+(self.gamma2()+self.force_gain_floor)*fm)
        xn = xa+vn
        for key in ('attention_calls', 'score_calls', 'connection_calls',
                    'directional_score_calls', 'extra_value_aggregations'):
            self.work_counts[key] += 1
        self.work_counts['velocity_norm_calls'] += 3 if self.mode == 'R2_LA' else 2
        if self.capture_diagnostics:
            with torch.no_grad():
                norm = lambda t: float(t.detach().norm(dim=-1).mean())
                fn, cn = norm(gain*force), norm(connection)
                off = self.ln_v(W)
                self.last_diagnostics = dict(
                    incoming_velocity_norm=norm(v), force_norm=norm(force), gained_force_norm=fn,
                    connection_norm=cn, connection_force_ratio=cn/max(fn, 1e-30),
                    retained_velocity_norm=norm(beta*v), raw_attention_velocity_norm=norm(W-connection),
                    attention_velocity_norm=norm(va), mlp_force_norm=norm(fm), stored_velocity_norm=norm(vn),
                    evaluated_beta_attn=float(beta), evaluated_gain_attn=float(gain),
                    position_increment_norm=norm(position), position_velocity_gap_norm=norm(position-va),
                    velocity_norm_calls_per_block=3 if self.mode == 'R2_LA' else 2,
                    post_ln_velocity_correction_norm=norm(va-off),
                    post_ln_position_correction_norm=norm(position-off),
                    attention_lookahead_displacement_norm=norm(self.mu1()*v),
                )
        return xn, vn


class NormalizedHBModel(LegacyModel):
    def __init__(self, cfg, hb):
        if hb.mode not in NEW_MODES:
            super().__init__(cfg, hb)
            return
        if cfg.dropout != 0 or cfg.bias or cfg.yurii_force_gain_floor != 1e-6:
            raise ValueError('Normalized HB requires dropout0, bias=False and Yurii gain floor1e-6')
        YuriiFormerModel.__init__(self, cfg, use_v0_init=True)
        self.hb = hb
        self.blocks = nn.ModuleList([LookaheadBlock(b, hb.mode) for b in self.blocks])

    def hb_dynamics(self):
        rows = super().hb_dynamics()
        if self.hb.mode in NEW_MODES:
            for row, block in zip(rows, self.blocks):
                row.update(mu_attn=float(block.mu1().detach()), connection_enabled=1,
                           forward_step=1., attention_lookahead_enabled=1,
                           proxy_evaluation='lookahead_position_ln',
                           velocity_norm_calls_per_block=3 if self.hb.mode == 'R2_LA' else 2)
                grad = block.mu1.raw.grad
                row['mu1_gradient_abs'] = float(grad.detach().abs()) if grad is not None else 0.
        return rows
