"""Corrected softmax geometry with persistent independent prefix ensembles.

One ensemble per predicted token, unmasked INSIDE that ensemble. Padding is not
an extra particle. Only each ensemble's last particle reaches the vocabulary
head. N1 is explicitly an oracle-normalization approximation, not a pullback.
"""
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from model import LayerNorm, MLP, ModelConfig


@dataclass(frozen=True)
class PrefixGeometryConfig:
    solver: str = 'cc'
    oracle_norm: str = 'raw'
    h: float = math.sqrt(.1)
    temperature: float = 1.
    damping_r: float = 3.
    t0: float = 1.
    prefix_chunk_size: int = 16
    checkpoint_chunks: bool = False
    initial_ln: bool = True
    norm_eps: float = 1e-5
    no_mlp: bool = False
    learned_v0: bool = False
    coefficient_policy: str = 'fixed'

    def __post_init__(self):
        if self.solver not in ('gd', 'hb', 'cc', 'r2'):
            raise ValueError('solver must be gd, hb, cc or r2')
        if self.oracle_norm not in ('raw', 'ln'):
            raise ValueError('oracle_norm must be raw or ln')
        if self.coefficient_policy not in ('fixed','shared_r','free_beta','free_gain','free_both','constant'):
            raise ValueError('invalid coefficient_policy')
        if self.coefficient_policy != 'fixed' and self.solver != 'cc':
            raise ValueError('coefficient policies currently require solver=cc')
        if self.coefficient_policy in ('shared_r','free_beta','free_both') and self.damping_r <= 0:
            raise ValueError('learned retention requires positive initial damping_r')
        if self.learned_v0 and self.solver == 'gd':
            raise ValueError('Geo-GD must not register inactive velocity tables')
        for key in ('h', 'temperature', 't0', 'norm_eps'):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if not math.isfinite(self.damping_r) or self.damping_r < 0:
            raise ValueError('damping_r must be finite and nonnegative')
        if any(type(getattr(self, key)) is not bool for key in ('checkpoint_chunks', 'initial_ln', 'no_mlp', 'learned_v0')):
            raise ValueError('checkpoint_chunks, initial_ln and no_mlp must be booleans')
        if type(self.prefix_chunk_size) is not int or self.prefix_chunk_size < 1:
            raise ValueError('prefix_chunk_size must be a positive integer')


DIAGNOSTIC_FIELDS = (
    'tau', 'dt', 'velocity_norm', 'attention_norm', 'connection_norm',
    'connection_update_norm', 'connection_velocity_update_norm', 'attention_update_norm',
    'total_displacement_norm', 'connection_attention_ratio', 'connection_displacement_ratio',
    'lookahead_displacement_norm', 'mlp_norm', 'momentum_force_cosine',
    'attention_self_mass', 'attention_entropy', 'weighted_velocity_sq',
    'beta_used', 'force_gain_used', 'connection_force_update_ratio',
)


def spd_factor(packed, indices, dimension):
    raw = packed.new_zeros((dimension, dimension)).index_put(tuple(indices), packed)
    return torch.tril(raw, diagonal=-1) + torch.diag_embed(F.softplus(raw.diagonal()) + 1e-4)


def geometry_oracle(Q, A, B, valid):
    """Batched full (symmetric, unmasked except padding) kernel, no raw exp."""
    QA = Q @ A
    scores = QA @ Q.transpose(-1, -2)
    scores = scores.masked_fill(~valid.unsqueeze(-2), -torch.inf)
    P = torch.softmax(scores, dim=-1)
    P = P.masked_fill(~valid.unsqueeze(-1), 0.)
    values = QA @ B
    return P @ values, P, QA, values, scores


def contracted_connection(Q, H, A, factor_B, P, QA, values):
    """Full contracted Levi--Civita connection; stable row-normalized identity."""
    solved = torch.linalg.solve_triangular(factor_B, H.transpose(-1, -2), upper=False)
    s = solved.square().sum(dim=-2)
    dscore = (H @ A) @ Q.transpose(-1, -2) + QA @ H.transpose(-1, -2)
    a = (P * dscore).sum(dim=-1)
    return a.unsqueeze(-1) * H - .5 * (s.unsqueeze(-1) * (P @ values) + P @ (s.unsqueeze(-1) * values))


class PrefixGeometryLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, geo: PrefixGeometryConfig):
        super().__init__()
        self.geo = geo
        self.d = cfg.n_embd
        initial = math.log(math.expm1(1. - 1e-4))
        indices = torch.tril_indices(self.d, self.d)
        self.register_buffer('factor_indices', indices, persistent=False)
        packed = torch.zeros(2, indices.shape[1])
        packed[:, indices[0] == indices[1]] = initial
        # Two packed triangular factors; no dormant upper-triangle parameters.
        # Keep a 2-D parameter so the existing optimizer uses its matrix group.
        self.metric_factors = nn.Parameter(packed)
        self.depth = cfg.n_layer
        self.theta_beta = nn.Parameter(torch.zeros(())) if geo.coefficient_policy in ('free_beta','free_both') else None
        self.theta_gain = nn.Parameter(torch.zeros(())) if geo.coefficient_policy in ('free_gain','free_both') else None
        if not geo.no_mlp:
            self.ln_mlp = LayerNorm(self.d, bias=cfg.bias)
            self.mlp = MLP(cfg)
        self.capture_diagnostics = False
        self.recomputing = False
        self.work_counts = dict(force_calls=0, connection_calls=0,
                                recompute_force_calls=0, recompute_connection_calls=0)
        self.reset_diagnostics()

    def reset_diagnostics(self):
        self._sums = {}
        self._samples = 0
        self.last_geometry = {}

    def matrices(self):
        LA = spd_factor(self.metric_factors[0], self.factor_indices, self.d)
        LB = spd_factor(self.metric_factors[1], self.factor_indices, self.d) * math.sqrt(self.d / self.geo.temperature)
        A = (self.geo.temperature / self.d) * (LA @ LA.T)
        return A, LB @ LB.T, LB

    def coefficients(self, layer, shared_r=None):
        geo=self.geo;h=geo.h;t=geo.t0+layer*h
        beta=(t/(t+h))**geo.damping_r
        if geo.coefficient_policy == 'constant':
            beta=(geo.t0/(geo.t0+self.depth*h))**(geo.damping_r/self.depth)
        elif geo.coefficient_policy == 'shared_r':
            if shared_r is None:raise ValueError('shared_r must be supplied by the model')
            beta=torch.exp(-shared_r*math.log1p(h/t))
        elif self.theta_beta is not None:
            beta=torch.sigmoid(math.log(beta/(1-beta))+self.theta_beta)
        gain=h*h
        if self.theta_gain is not None:
            c0=self.theta_gain.new_tensor(math.log(math.expm1(1.)))
            gain=gain*F.softplus(c0+self.theta_gain)/F.softplus(c0)
        return beta,gain

    def forward(self, X, U, valid, lengths, layer, shared_r=None):
        geo = self.geo
        h = geo.h
        beta,gain = self.coefficients(layer,shared_r)
        Q = F.layer_norm(X, (self.d,), eps=geo.norm_eps) if geo.oracle_norm == 'ln' else X
        A, B, LB = self.matrices()
        key = 'recompute_force_calls' if self.recomputing else 'force_calls'
        self.work_counts[key] += 1
        G, P, QA, values, scores = geometry_oracle(Q, A, B, valid)
        correction = torch.zeros_like(X)
        if geo.solver == 'gd':
            Xattn = X + h * h * G
            Unext = torch.zeros_like(U)
        else:
            V = h * U
            W = beta * V + gain * G
            if geo.solver in ('cc', 'r2'):
                key = 'recompute_connection_calls' if self.recomputing else 'connection_calls'
                self.work_counts[key] += 1
                argument = V if geo.solver == 'cc' else W
                correction = contracted_connection(Q, argument, A, LB, P, QA, values)
            Xattn = X + W - (.5 if geo.solver == 'r2' else 1.) * correction
            Unext = (W - correction) / h
        mlp = self.mlp(self.ln_mlp(Xattn)) if not geo.no_mlp else torch.zeros_like(Xattn)
        Xout = (Xattn + mlp).masked_fill(~valid.unsqueeze(-1), 0.)
        Unext = Unext.masked_fill(~valid.unsqueeze(-1), 0.)
        if self.capture_diagnostics and not self.recomputing:
            self._capture(X, U, G, correction, Xout, mlp, P, scores, LB, valid, lengths, beta, gain)
        return Xout, Unext

    @torch.no_grad()
    def _capture(self, X, U, G, correction, Xout, mlp, P, scores, LB, valid, lengths, beta, gain):
        # Diagnostics average READOUT particles, not all duplicated internal rows.
        indices = torch.arange(len(lengths), device=X.device)
        last = lambda v: v[:, indices, lengths - 1]
        norms = lambda v: last(v).norm(dim=-1)
        h = self.geo.h
        pos_correction = correction * (.5 if self.geo.solver == 'r2' else 1.)
        force, gamma = norms(G), norms(correction / (h*h))
        total = norms(Xout-X)
        momentum_force = (last(U)*last(G)).sum(-1) / (norms(U)*force).clamp_min(1e-30)
        velocity2 = torch.linalg.solve_triangular(LB, U.transpose(-1,-2), upper=False).square().sum(-2)
        logz = torch.logsumexp(scores, -1).masked_fill(~valid, -torch.inf)
        weighted = (torch.softmax(logz, -1)*velocity2).sum(-1)
        pred_weights = last(P)
        data = dict(velocity_norm=norms(U), attention_norm=force, connection_norm=gamma,
                    connection_update_norm=norms(pos_correction),
                    connection_velocity_update_norm=norms(correction),
                    attention_update_norm=float(gain)*force, total_displacement_norm=total,
                    connection_attention_ratio=gamma/force.clamp_min(1e-30),
                    connection_displacement_ratio=norms(pos_correction)/total.clamp_min(1e-30),
                    lookahead_displacement_norm=torch.zeros_like(total),
                    mlp_norm=norms(mlp), momentum_force_cosine=momentum_force,
                    attention_self_mass=last(P.diagonal(dim1=-2,dim2=-1)),
                    attention_entropy=-(pred_weights*pred_weights.clamp_min(1e-30).log()).sum(-1),
                    weighted_velocity_sq=weighted,
                    beta_used=torch.full_like(force,float(beta)),
                    force_gain_used=torch.full_like(force,float(gain)),
                    connection_force_update_ratio=norms(pos_correction)/(float(gain)*force).clamp_min(1e-30))
        for key, values in data.items():
            self._sums[key] = self._sums.get(key, 0.) + values.double().sum().item()
        self._samples += X.shape[0] * len(lengths)
        self.last_geometry = {k: v/self._samples for k,v in self._sums.items()}
        self.last_geometry.update(tau=h*h, dt=h*h if self.geo.solver=='gd' else h)


class RiemannianPrefixModel(nn.Module):
    """One persistent, entire-prefix ensemble per next-token output."""
    def __init__(self, cfg: ModelConfig, geo: PrefixGeometryConfig):
        super().__init__()
        if cfg.n_head != 1:
            raise ValueError('riem_prefix has exactly one full-width kernel; set n_head=1')
        if cfg.dropout != 0:
            raise ValueError('prefix batching currently requires dropout=0 for reproducibility')
        if min(cfg.n_layer, cfg.n_embd, cfg.block_size) < 1:
            raise ValueError('positive layer, width and context sizes required')
        self.cfg, self.geo = cfg, geo
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.layers = nn.ModuleList([PrefixGeometryLayer(cfg, geo) for _ in range(cfg.n_layer)])
        self.ln_f = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)
        self.theta_shared_r = (nn.Parameter(torch.zeros(()))
                               if geo.coefficient_policy == 'shared_r' else None)
        self.tok_v0_emb = self.pos_v0_emb = None
        if geo.learned_v0:
            # Preserve common core weights AND the caller's global RNG stream.
            # The trainer subsequently uses its dedicated paired initializer.
            with torch.random.fork_rng(devices=[]):
                self.tok_v0_emb=nn.Embedding(cfg.vocab_size,cfg.n_embd)
                self.pos_v0_emb=nn.Embedding(cfg.block_size,cfg.n_embd)
                self.tok_v0_emb.apply(self._init_weights)
                self.pos_v0_emb.apply(self._init_weights)
        self.last_leak_warnings = 0
        self.last_h_mean = geo.h**2 if geo.solver=='gd' else geo.h
        self.last_hY_mean = math.nan
        self.last_c_log_mean = math.nan if geo.solver=='gd' else geo.damping_r
        self.last_c_lin_mean = math.nan if geo.solver=='gd' else 0.
        if geo.coefficient_policy in ('free_beta','free_both'):
            self.last_c_log_mean=self.last_c_lin_mean=math.nan
        elif geo.coefficient_policy=='constant':
            self.last_c_log_mean=0.
            self.last_c_lin_mean=geo.damping_r*math.log1p(cfg.n_layer*geo.h/geo.t0)/(cfg.n_layer*geo.h)
        self.last_t_start = geo.t0
        self.last_t_end = geo.t0 + cfg.n_layer*self.last_h_mean

    def shared_r_value(self):
        if self.theta_shared_r is None:return None
        c0=self.theta_shared_r.new_tensor(math.log(math.expm1(self.geo.damping_r)))
        return self.geo.damping_r*F.softplus(c0+self.theta_shared_r)/F.softplus(c0)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Embedding, nn.Linear)):
            nn.init.normal_(module.weight, mean=0., std=.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    @contextmanager
    def _recompute_context(self):
        for layer in self.layers:
            layer.recomputing = True
        try:
            yield
        finally:
            for layer in self.layers:
                layer.recomputing = False

    def evolve_prefix_batch(self, initial, lengths, initial_velocity=None):
        """initial[B,P,N,d]; lengths[P]. Padding has no effect on valid states."""
        N = initial.shape[-2]
        valid = (torch.arange(N, device=initial.device)[None,:] < lengths[:,None])[None]
        X = initial.masked_fill(~valid.unsqueeze(-1), 0.)
        if initial_velocity is None:
            if self.geo.learned_v0:raise ValueError('learned-v0 ensembles require explicit initial_velocity')
            U=torch.zeros_like(X)
        else:
            if initial_velocity.shape != initial.shape:raise ValueError('initial velocity shape mismatch')
            U=initial_velocity.masked_fill(~valid.unsqueeze(-1),0.)
        shared_r=self.shared_r_value()
        for i,layer in enumerate(self.layers):
            X,U = layer(X,U,valid,lengths,i,shared_r)
        return X,U

    def _read_prefix_chunk(self, initial, lengths, initial_velocity=None):
        X,_ = self.evolve_prefix_batch(initial, lengths, initial_velocity)
        return X[:,torch.arange(len(lengths), device=X.device),lengths-1]

    def forward_features(self, idx, *, last_only=False):
        if idx.ndim != 2 or idx.shape[0] < 1 or not 0 < idx.shape[1] <= self.cfg.block_size:
            raise ValueError('expected nonempty [batch,time] within configured context')
        if torch.is_autocast_enabled(idx.device.type):
            raise ValueError('riem_prefix currently requires FP32/FP64 without autocast')
        if self.tok_emb.weight.dtype not in (torch.float32, torch.float64):
            raise ValueError('riem_prefix currently supports only FP32/FP64')
        for layer in self.layers:
            layer.reset_diagnostics()
        if self.theta_shared_r is not None and any(layer.capture_diagnostics for layer in self.layers):
            self.last_c_log_mean=float(self.shared_r_value().detach())
        T = idx.shape[1]
        X0 = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))[None]
        if self.geo.initial_ln:
            X0 = F.layer_norm(X0, (self.cfg.n_embd,), eps=self.geo.norm_eps)
        U0=(self.tok_v0_emb(idx)+self.pos_v0_emb(torch.arange(T,device=idx.device))[None]
            if self.geo.learned_v0 else None)
        result = []
        starts = [T] if last_only else range(1,T+1,self.geo.prefix_chunk_size)
        for start in starts:
            stop = T+1 if last_only else min(start+self.geo.prefix_chunk_size,T+1)
            lengths = torch.arange(start,stop,device=idx.device)
            initial = X0[:,:stop-1].unsqueeze(1).expand(-1,stop-start,-1,-1)
            velocity=None if U0 is None else U0[:,:stop-1].unsqueeze(1).expand(-1,stop-start,-1,-1)
            if self.geo.checkpoint_chunks and torch.is_grad_enabled():
                out = checkpoint(self._read_prefix_chunk,initial,lengths,velocity,use_reentrant=False,
                                 context_fn=lambda:(nullcontext(),self._recompute_context()),
                                 preserve_rng_state=True)
            else:
                out = self._read_prefix_chunk(initial,lengths,velocity)
            result.append(out)
        return torch.cat(result,dim=1)

    def forward(self, idx, targets=None, global_step=None):
        logits = self.lm_head(self.ln_f(self.forward_features(idx)))
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1,logits.shape[-1]),targets.reshape(-1))
        return logits,loss

    def forward_last(self, idx):
        """Generation without evolving unused smaller prefixes; NOT a KV cache."""
        return self.lm_head(self.ln_f(self.forward_features(idx,last_only=True)))[:,0]

    def reset_work_counts(self):
        for layer in self.layers:
            for key in layer.work_counts:
                layer.work_counts[key] = 0

    def work_counts(self):
        return {k:sum(layer.work_counts[k] for layer in self.layers) for k in self.layers[0].work_counts}

    @torch.no_grad()
    def layer_dynamics(self):
        rows=[]
        policy=self.geo.coefficient_policy
        shared_r=self.shared_r_value()
        gradnorm=lambda p: float(p.grad.norm()) if p is not None and p.grad is not None else math.nan
        vgrad=[gradnorm(m.weight) for m in (self.tok_v0_emb,self.pos_v0_emb) if m is not None]
        velocity_grad=math.sqrt(sum(g*g for g in vgrad)) if vgrad else math.nan
        for i,layer in enumerate(self.layers):
            h=self.last_h_mean; t=self.geo.t0+i*h; gd=self.geo.solver=='gd'
            beta,gain=layer.coefficients(i,shared_r);beta=float(beta);gain=float(gain)
            r=float(shared_r) if shared_r is not None else self.geo.damping_r
            effective=-math.log(beta)/h if beta>0 else math.inf
            shaped=policy in ('fixed','shared_r','free_gain')
            rows.append(dict(layer=i,shared=0,module_type=type(layer).__name__,
                             c_log=(r if shaped else (0. if policy=='constant' else math.nan)) if not gd else math.nan,
                             c_lin=(effective if policy=='constant' else (0. if shaped else math.nan)) if not gd else math.nan,hX=h,hY=math.nan,
                             beta_attn=math.nan if gd else beta,
                             beta_mlp=math.nan,t_start=t,t_end=t+h,
                             alpha_start=math.nan if gd else (r/t if shaped else effective),
                             alpha_end=math.nan if gd else (r/(t+h) if shaped else effective),
                             **{key:layer.last_geometry.get(key,math.nan) for key in DIAGNOSTIC_FIELDS},
                             solver=self.geo.solver,oracle_norm=self.geo.oracle_norm,
                             diagnostic_scope='prefix_readout_particles',effective_heads=1,
                             damping_learned=int(policy in ('shared_r','free_beta','free_both')),
                             coefficient_policy=policy, learned_v0=int(self.geo.learned_v0),
                             force_gain=gain, force_gain_learned=int(layer.theta_gain is not None),
                             force_retention_ratio=gain/beta if beta>0 else math.inf,
                             interval_effective_damping=math.nan if gd else effective,
                             beta_gradient=gradnorm(layer.theta_beta),gain_gradient=gradnorm(layer.theta_gain),
                             shared_r_gradient=gradnorm(self.theta_shared_r),velocity_embedding_grad_norm=velocity_grad,
                             parameter_gradient_scope='last_training_backward_after_clipping',
                             coefficient_snapshot='current_parameters; beta_used/gain_used are captured forward values'))
        return rows
