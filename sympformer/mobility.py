"""Reported normalized-sigmoid and native-L2 SPD prefix proxies only."""
from dataclasses import dataclass,replace
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .core import GPTModel,YuriiFormerModel
from .softmax import NormalizedHBModel,NormalizedHBConfig
VERSION="e068_mobility_prefix_v1"
MODES=("B","Y","H0","H1","R2")
FLAT=frozenset()
FAMILIES=("sigmoid_normalized","l2_native_tensor_proxy")
REFERENCE_FAMILIES=()

@dataclass(frozen=True)
class MobilityConfig:
    family: str = 'softmax'
    mode: str = 'H0'
    version: str = VERSION
    sink: float = 1.
    tensor_delta: float = 1e-3
    sinkhorn_iterations: int = 12
    prefix_chunk: int = 4
    checkpoint_prefixes: bool = True

    def __post_init__(self):
        if self.family not in FAMILIES + REFERENCE_FAMILIES or self.mode not in MODES:
            raise ValueError('Unknown attention family or acceleration mode')
        if self.version != VERSION or self.sink != 1. or self.tensor_delta != 1e-3:
            raise ValueError('Frozen E068 version/sink/tensor regularization')
        if self.sinkhorn_iterations != 12 or self.prefix_chunk < 1:
            raise ValueError('Twelve balancing iterations and positive prefix chunk required')

    @property
    def canonical_mode(self):
        return 'H0' if self.family in FLAT and self.mode in ('H1', 'R2') else self.mode


def merge(x):
    return x.transpose(1, 2).contiguous().flatten(2)


def split(x, heads):
    return x.reshape(*x.shape[:2], heads, x.shape[-1] // heads).transpose(1, 2)


def project(attn, z):
    return tuple(split(x, attn.n_head) for x in attn.c_attn(z).chunk(3, -1))


def direction_project(attn, u):
    # Bias-free version only; directional derivatives must not add a bias.
    return tuple(split(x, attn.n_head) for x in F.linear(u, attn.c_attn.weight).chunk(3, -1))


def geometry(attn, z, u=None):
    q, k, w = project(attn, z)
    positive = attn.spec.family == 'bilinear_normalized_positive'
    raw_q, raw_k = q, k
    if positive:
        q, k = F.softplus(q), F.softplus(k)
    distance = attn.spec.family.startswith('l2_')
    if distance:
        logits = -.5 * attn.scale * (q.square().sum(-1)[..., :, None]
            + k.square().sum(-1)[..., None, :] - 2 * (q @ k.transpose(-2, -1)))
    else:
        logits = attn.scale * (q @ k.transpose(-2, -1))
    if u is None:
        return q, k, w, logits
    dq, dk, dw = direction_project(attn, u)
    if positive:
        dq, dk = dq * torch.sigmoid(raw_q), dk * torch.sigmoid(raw_k)
    if distance:
        directional = -attn.scale * ((q*dq).sum(-1)[..., :, None]
            + (k*dk).sum(-1)[..., None, :] - dq @ k.transpose(-2, -1)
            - q @ dk.transpose(-2, -1))
    else:
        directional = attn.scale * (dq @ k.transpose(-2, -1) + q @ dk.transpose(-2, -1))
    return q, k, w, logits, directional, dw


def weights(logits, spec):
    """Return p and phi'/D; normalized sigmoid computed in log space.

    Sink c=1 is in the SUM denominator. No hidden positivity/clipping epsilon.
    ReLU's derivative at its kink is defined as0, as in torch.relu.
    """
    n = logits.shape[-1]
    valid = torch.ones(n, n, dtype=torch.bool, device=logits.device).tril()
    f = spec.family
    count = torch.arange(1, n+1, device=logits.device, dtype=logits.dtype)
    if f in ('softmax', 'l2_mean_shift', 'l2_native_tensor_proxy'):
        p = F.softmax(logits.masked_fill(~valid, -torch.inf), -1)
        return p, p
    if f == 'softmax_sink':
        scores = logits.masked_fill(~valid, -torch.inf)
        denominator = torch.logaddexp(torch.logsumexp(scores, -1, keepdim=True),
                                     torch.zeros_like(scores[..., :1]))
        p = torch.exp(scores - denominator)
        return p, p
    if f == 'sigmoid_normalized':
        log_phi = F.logsigmoid(logits).masked_fill(~valid, -torch.inf)
        p = F.softmax(log_phi, -1)
        return p, p * torch.sigmoid(-logits)
    if f == 'relu_sink':
        phi = F.relu(logits).masked_fill(~valid, 0)
        denominator = phi.sum(-1, keepdim=True) + spec.sink
        return phi/denominator, ((logits > 0) & valid).to(logits.dtype)/denominator
    if f in ('bilinear_normalized', 'bilinear_normalized_positive'):
        phi = logits.masked_fill(~valid, 0)
        denominator = phi.sum(-1, keepdim=True)
        if not bool((torch.isfinite(denominator) & (denominator > 0)).all()):
            raise ValueError('Normalized bilinear outside finite positive-normalizer domain')
        return phi/denominator, valid.to(logits.dtype)/denominator
    # Mask BEFORE exp; invalid future scores must never overflow via 0*inf.
    if f == 'exp_unnormalized':
        phi = torch.exp(logits.masked_fill(~valid, -torch.inf))
    elif f == 'sigmoid_unnormalized':
        phi = torch.sigmoid(logits).masked_fill(~valid, 0)
    elif f == 'relu_unnormalized':
        phi = F.relu(logits).masked_fill(~valid, 0)
    elif f == 'affine_unnormalized':
        phi = (1+logits).masked_fill(~valid, 0)
    elif f == 'bilinear_otto':
        phi = logits.masked_fill(~valid, 0)
    else:
        raise ValueError(f'No scalar kernel weights for {f}')
    return phi/count[..., None], None


def tensor_metric(force, delta=1e-3):
    """Inverse-free declared proxy g=C^-1, C=I+ff^T/a; NOT Ce=f."""
    a = F.softplus(force[..., 0]) + delta
    t = a + force.square().sum(-1)
    eye = torch.eye(force.shape[-1], device=force.device, dtype=force.dtype)
    return eye-force[..., :, None]*force[..., None, :]/t[..., None, None]


def native_l2_connection(attn, z, u, force, p):
    q, k, w, _, ds, dw = geometry(attn, z, u)
    dp = p*(ds-(p*ds).sum(-1, keepdim=True))
    df = attn.c_proj(merge(dp @ w + p @ dw))
    a = F.softplus(force[..., 0])+attn.spec.tensor_delta
    t = a+force.square().sum(-1)
    b = (force*u).sum(-1)
    da = torch.zeros_like(force)
    da[..., 0] = torch.sigmoid(force[..., 0])
    dt = ((da+2*force)*df).sum(-1)
    dg_u = -(df*b[..., None]+force*(df*u).sum(-1, keepdim=True))/t[..., None]
    dg_u = dg_u+force*(b*dt/t.square())[..., None]
    h = -2*b[..., None]*u/t[..., None] + (b/t).square()[..., None]*(da+2*force)
    # Self-only raw-coordinate VJP. A global row sum would introduce suffix
    # adjoints; this is the last component of EACH separate prefix geometry.
    projected = split(F.linear(h, attn.c_proj.weight.T), attn.n_head)
    scalar = projected @ w.transpose(-2, -1)
    center = p*(scalar-(p*scalar).sum(-1, keepdim=True))
    diag = center.diagonal(dim1=-2, dim2=-1)
    dq = -attn.scale*(q*center.sum(-1, keepdim=True)-center @ k)
    dk = attn.scale*diag[..., None]*(q-k)
    dv = p.diagonal(dim1=-2, dim2=-1)[..., None]*projected
    width = z.shape[-1]
    wq, wk, wv = attn.c_attn.weight.split(width)
    self_vjp = (F.linear(merge(dq), wq.T)+F.linear(merge(dk), wk.T)
                + F.linear(merge(dv), wv.T))
    covector = dg_u-.5*self_vjp
    return covector+force*(force*covector).sum(-1, keepdim=True)/a[..., None]


class MobilityAttention(nn.Module):
    def __init__(self, donor, spec):
        super().__init__()
        self.spec = spec
        self.n_head, self.head_dim, self.scale = donor.n_head, donor.head_dim, donor.scale
        if donor.c_attn.bias is not None or donor.c_proj.bias is not None:
            raise ValueError('Bias-free force/derivative definition required')
        for name, module in donor.named_children():
            self.add_module(name, module)
        self.last_balance_residual = None

    def force_and_connection(self, z, u=None, *, proposal=None):
        q, k, w, logits = geometry(self, z)
        if self.spec.family == 'sinkhorn_prefix':
            force = sinkhorn_force(self, logits, w)
        else:
            p, edge = weights(logits, self.spec)
            aggregated = p @ w
            if self.spec.family == 'l2_mean_shift':
                aggregated = aggregated-w
            force = self.c_proj(merge(aggregated))
        if proposal is not None:
            beta, gain = proposal
            u = beta*u+gain*force
        if u is None or self.spec.family in FLAT:
            connection = torch.zeros_like(force)
        elif self.spec.family == 'l2_native_tensor_proxy':
            connection = native_l2_connection(self, z, u, force, p)
        else:
            directional = geometry(self, z, u)[4]
            drift = (edge*directional).sum(-1).mean(1)
            speed = u.square().sum(-1)
            first = edge @ w
            second = edge @ (w*speed[:, None, :, None])
            if self.spec.family == 'l2_mean_shift':
                first = first-edge.sum(-1, keepdim=True)*w
                second = second-(edge*speed[:, None, None, :]).sum(-1, keepdim=True)*w
            connection = drift[..., None]*u-.5*(speed[..., None]*self.c_proj(merge(first))
                                                 + self.c_proj(merge(second)))
        return force, connection, u

    def forward(self, z):
        return self.force_and_connection(z)[0]


class MobilityBlock(nn.Module):
    def __init__(self, donor, spec):
        super().__init__()
        self.spec, self.mode = spec, spec.canonical_mode
        self.force_gain_floor = donor.force_gain_floor
        for name, module in donor.named_children():
            if name == 'mu1' and self.mode != 'Y':
                continue
            self.add_module(name, module)
        self.attn = MobilityAttention(self.attn, spec)
        self.capture_diagnostics = False
        self.last_diagnostics = {}

    def forward(self, x, v, noise_std=0., noise_loc='v'):
        if noise_std != 0 or noise_loc not in ('dx', 'v', 'xin'):
            raise ValueError('No E068 training noise')
        z = self.ln_x_attn(x+self.mu1()*v) if self.mode == 'Y' else self.ln_x_attn(x)
        beta, gain = self.beta1(), self.gamma1()+self.force_gain_floor
        active = self.mode in ('H1', 'R2')
        force, gamma, direction = self.attn.force_and_connection(z, v if active else None,
            proposal=(beta, gain) if self.mode == 'R2' else None)
        proposal = direction if self.mode == 'R2' else beta*v+gain*force
        raw_v = proposal-gamma
        va = self.ln_v(raw_v)
        position = self.ln_v(proposal-.5*gamma) if self.mode == 'R2' else va
        xa = x+position
        fm = self.mlp(self.ln_x_mlp(xa+self.mu2()*va))
        vn = self.ln_v(self.beta2()*va+(self.gamma2()+self.force_gain_floor)*fm)
        if self.capture_diagnostics:
            with torch.no_grad():
                norm = lambda a: float(a.double().norm(dim=-1).mean())
                self.last_diagnostics = dict(force_norm=norm(force), connection_norm=norm(gamma),
                    proposal_norm=norm(proposal), position_velocity_gap=norm(position-va))
        return xa+vn, vn


class MobilityModel(YuriiFormerModel):
    def __init__(self, cfg, spec):
        super().__init__(cfg, use_v0_init=True)
        self.spec = spec
        self.blocks = nn.ModuleList(MobilityBlock(b, spec) for b in self.blocks)


def dynamics(model):
    """All logical layers, including B=not applicable, not invented betas."""
    rows = []
    spec = model.mobility_spec
    for layer, b in enumerate(model.blocks):
        row = dict(layer=layer, family=spec.family, mode=spec.mode,
                   canonical_mode=spec.canonical_mode, version=VERSION,
                   learned_damping=spec.mode != 'B', forward_step=1., internal_time=layer+1,
                   connection_active=spec.canonical_mode in ('H1', 'R2'),
                   nesterov_reference=3./(layer+1))
        if spec.mode != 'B':
            ba, bm = float(b.beta1().detach()), float(b.beta2().detach())
            if not (0 < ba < 1 and 0 < bm < 1):
                raise ValueError('Strict positive finite damping domain violated')
            row.update(beta_attn=ba, beta_mlp=bm, effective_damping_attn=-math.log(ba),
                effective_damping_mlp=-math.log(bm),
                gain_attn=float((b.gamma1()+b.force_gain_floor).detach()),
                gain_mlp=float((b.gamma2()+b.force_gain_floor).detach()),
                mu_attn=float(b.mu1().detach()) if hasattr(b, 'mu1') else 0.,
                mu_mlp=float(b.mu2().detach()))
            row.update(getattr(b, 'last_diagnostics', {}))
        residual = getattr(b.attn, 'last_balance_residual', None)
        if residual is not None:
            row['sinkhorn_balance_residual'] = residual
        rows.append(row)
    return rows


def make_model(cfg, spec, seed):
    """Native Yurii donor initializes identical common tensors in every arm.

    Flat aliases use canonical H0 execution and parameterization bit-for-bit.
    The signed bilinear Stein block retains its frozen E062 ridge proxy; softmax
    retains existing E046/E048 update semantics. No donor/source modification.
    """
    if cfg.bias or cfg.dropout or cfg.yurii_force_gain_floor != 1e-6:
        raise ValueError('Requires biasFalse/dropout0/Yurii floor1e-6')
    if spec.family == 'bilinear_stein':
        mapping = dict(B='B_L', Y='Y_L', H0='H0_L', H1='H1_L', R2='R2_L')
        model = make_linear_model(cfg, mapping[spec.mode], seed, chunk=512)
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            donor = YuriiFormerModel(cfg, use_v0_init=True)
            torch.manual_seed(seed)
            if spec.mode == 'B':
                model = GPTModel(cfg)
                for b in model.blocks:
                    b.attn = MobilityAttention(b.attn, spec)
            elif spec.family == 'softmax':
                version = 'e048_r2_dual_ln_v1' if spec.mode == 'R2' else 'e046_identity_v1'
                model = NormalizedHBModel(cfg, NormalizedHBConfig(spec.mode, version))
            else:
                model = MobilityModel(cfg, spec)
            source = dict(donor.named_parameters())
            with torch.no_grad():
                for name, value in model.named_parameters():
                    mapped = name.replace('.ln_1.', '.ln_x_attn.').replace('.ln_2.', '.ln_x_mlp.')
                    if mapped not in source or source[mapped].shape != value.shape:
                        raise ValueError(f'Donor parameter mismatch: {name}')
                    value.copy_(source[mapped])
    model.mobility_spec = spec
    return model
