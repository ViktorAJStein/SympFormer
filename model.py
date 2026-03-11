### 10.03.26 00:05
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- helper transforms for constrained scalars ----
def inv_softplus(y: float) -> float:
    """Inverse of softplus for y>0."""
    y = float(y)
    if y <= 0:
        return -20.0
    if y < 20:
        return math.log(math.expm1(y))
    return y

def inv_sigmoid(p: float) -> float:
    """logit(p) for p in (0,1)."""
    p = float(p)
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p) - math.log(1 - p)


@dataclass
class ModelConfig:
    vocab_size: int = 50304
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False

    # Presymp Variant A: use attention-induced velocity for MLP lookahead
    presymp_mlp_use_attn_vel: bool = False


def causal_mask(T: int, device: torch.device) -> torch.Tensor:
    # True where j > i (masked)
    return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)


class LayerNorm(nn.Module):
    '''LayerNorm with optional bias (nanoGPT style).'''
    def __init__(self, n_embd: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.bias = nn.Parameter(torch.zeros(n_embd)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, x.shape[-1:], self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * self.scale  # (B, nh, T, T)
        m = causal_mask(T, x.device)
        att = att.masked_fill(m.unsqueeze(0).unsqueeze(0), float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v  # (B, nh, T, hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = 4 * cfg.n_embd
        self.fc1 = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.fc2 = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return self.drop(x)


class GPTBlock(nn.Module):
    '''Baseline block: x += Attn(LN(x)); x += MLP(LN(x)).'''
    def __init__(self, cfg: ModelConfig, no_mlp: bool = False):
        super().__init__()
        self.no_mlp = bool(no_mlp)
        self.ln_1 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        if not self.no_mlp:
            x = x + self.mlp(self.ln_2(x))
        return x


class ConstrainedScalar(nn.Module):
    '''Unconstrained scalar mapped to (0,1) via sigmoid or to (0,inf) via softplus.'''
    def __init__(self, init: float, kind: str):
        super().__init__()
        self.kind = kind
        if kind == "unit":
            init = min(max(init, 1e-4), 1 - 1e-4)
            p = math.log(init / (1 - init))
        elif kind == "pos":
            p = math.log(math.expm1(max(init, 1e-8)))
        else:
            raise ValueError("kind must be 'unit' or 'pos'")
        self.raw = nn.Parameter(torch.tensor(p, dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        if self.kind == "unit":
            return torch.sigmoid(self.raw)
        return F.softplus(self.raw)


class YuriiFormerLieTrotterBlock(nn.Module):
    '''Nesterov + Lie-Trotter splitting (attention then MLP), with velocity LayerNorm after each velocity update.

    Optional: inject (annealed) Gaussian noise into the Nesterov dynamics, either into
    - the oracle outputs (dx),
    - the velocity stream (v), or
    - the lookahead states (x_in).
    '''
    def __init__(self, cfg: ModelConfig, no_mlp: bool = False):
        super().__init__()
        self.no_mlp = bool(no_mlp)
        self.ln_x_attn = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_x_mlp = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

        self.ln_v = LayerNorm(cfg.n_embd, bias=cfg.bias)

        # learned scalars for the two substeps
        self.mu1 = ConstrainedScalar(0.9, "unit")
        self.beta1 = ConstrainedScalar(0.9, "unit")
        self.gamma1 = ConstrainedScalar(1.0, "pos")

        self.mu2 = ConstrainedScalar(0.9, "unit")
        self.beta2 = ConstrainedScalar(0.9, "unit")
        self.gamma2 = ConstrainedScalar(1.0, "pos")

    def forward(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        noise_std: float = 0.0,
        noise_loc: str = "v",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise_loc not in ("dx", "v", "xin"):
            raise ValueError("noise_loc must be one of {'dx','v','xin'}")

        def _noise_like(t: torch.Tensor) -> torch.Tensor:
            if noise_std <= 0.0:
                return torch.zeros_like(t)
            n = torch.randn_like(t, dtype=torch.float32)
            return (noise_std * n).to(dtype=t.dtype)
        # attention substep
        mu1 = self.mu1()
        beta1 = self.beta1()
        gamma1 = self.gamma1()

        x_in = x + mu1 * v
        if noise_loc == "xin":
            x_in = x_in + _noise_like(x_in)
        dx_attn = self.attn(self.ln_x_attn(x_in))
        if noise_loc == "dx":
            dx_attn = dx_attn + _noise_like(dx_attn)
        v_half = beta1 * v + gamma1 * dx_attn
        if noise_loc == "v":
            v_half = v_half + _noise_like(v_half)
        v_half = self.ln_v(v_half)
        x_half = x + v_half

        # When --no_mlp is set, skip the MLP substep entirely.
        if self.no_mlp:
            return x_half, v_half

        # mlp substep
        mu2 = self.mu2()
        beta2 = self.beta2()
        gamma2 = self.gamma2()

        x_in2 = x_half + mu2 * v_half
        if noise_loc == "xin":
            x_in2 = x_in2 + _noise_like(x_in2)
        dx_mlp = self.mlp(self.ln_x_mlp(x_in2))
        if noise_loc == "dx":
            dx_mlp = dx_mlp + _noise_like(dx_mlp)
        v_next = beta2 * v_half + gamma2 * dx_mlp
        if noise_loc == "v":
            v_next = v_next + _noise_like(v_next)
        v_next = self.ln_v(v_next)
        x_next = x_half + v_next
        return x_next, v_next


class EtaSchedule(nn.Module):
    """Eta schedule used for conformal scaling.

    We use eta(t)=∫ alpha(s) ds and expose several parameterizations.

    Fixed (learnable=False):
      * mode='log'    : eta(t)=c_log*log(t/t0) with c_log defaulting to 3.0
      * mode='linear' : eta(t)=c_lin*t         (set via eta_mu or eta_lin_coef)
      * mode='loglin' : eta(t)=c_log*log(t/t0) + c_lin*t

    Learnable (learnable=True):
      * mode='log'    : eta(t)=c_log*log(t/t0), with c_log>0 learned
      * mode='linear' : eta(t)=c_lin*t,         with c_lin>0 learned
      * mode='loglin' : eta(t)=c_log*log(t/t0) + c_lin*t, both >0 learned

    Note: we clamp eta(t) to [-eta_clip, eta_clip] before exponentiation.
    """

    def __init__(
        self,
        t0: float = 1.0,
        mu: Optional[float] = None,
        *,
        log_coef: Optional[float] = None,
        lin_coef: Optional[float] = None,
        learnable: bool = False,
        mode: str = 'log',
        init: Optional[float] = None,
        init_log: Optional[float] = None,
        init_lin: Optional[float] = None,
        eta_clip: float = 50.0,
    ):
        super().__init__()
        if t0 <= 0:
            raise ValueError('t0 must be > 0')
        self.t0 = float(t0)
        self.learnable = bool(learnable)
        self.mode = str(mode)
        self.eta_clip = float(eta_clip)

        if self.mode not in ('log', 'linear', 'loglin'):
            raise ValueError("eta mode must be one of {'log','linear','loglin'}")

        # Store fixed coefficients (used when learnable=False).
        # We interpret:
        #   - log_coef: coefficient of log(t/t0)
        #   - lin_coef: coefficient of t
        # For backward compatibility:
        #   - mu is treated as lin_coef when lin_coef is not provided.
        if not self.learnable:
            if self.mode == 'log':
                self.c_log_const = 3.0 if log_coef is None else float(log_coef)
                self.c_lin_const = 0.0
            elif self.mode == 'linear':
                c_lin = float(mu) if (lin_coef is None and mu is not None) else (0.0 if lin_coef is None else float(lin_coef))
                self.c_log_const = 0.0
                self.c_lin_const = c_lin
            else:  # loglin
                self.c_log_const = 3.0 if log_coef is None else float(log_coef)
                if lin_coef is None:
                    self.c_lin_const = 0.0 if mu is None else float(mu)
                else:
                    self.c_lin_const = float(lin_coef)

            self.c_log = None
            self.c_lin = None
            return

        # Learnable coefficients (both constrained to be positive via softplus).
        if self.mode == 'log':
            c0 = 3.0 if init is None else float(init)
            if init_log is not None:
                c0 = float(init_log)
            self.c_log = ConstrainedScalar(c0, kind='pos')
            self.c_lin = None
        elif self.mode == 'linear':
            mu0 = float(mu) if (init is None and mu is not None) else (0.1 if init is None else float(init))
            if init_lin is not None:
                mu0 = float(init_lin)
            self.c_lin = ConstrainedScalar(mu0, kind='pos')
            self.c_log = None
        else:  # loglin
            c0 = 3.0
            m0 = 0.0
            if init is not None:
                c0 = float(init)
            if init_log is not None:
                c0 = float(init_log)
            if init_lin is not None:
                m0 = float(init_lin)
            # Backward compatibility: if user provided mu and no init_lin, use mu as initial linear coefficient
            if init_lin is None and mu is not None:
                m0 = float(mu)
            self.c_log = ConstrainedScalar(c0, kind='pos')
            self.c_lin = ConstrainedScalar(max(m0, 1e-8), kind='pos')

    def _eta_unclipped(self, t: float, device, dtype) -> torch.Tensor:
        tt = torch.tensor(t, device=device, dtype=dtype)
        if self.learnable:
            if self.mode == 'log':
                return self.c_log() * torch.log(tt / self.t0)
            if self.mode == 'linear':
                return self.c_lin() * tt
            return self.c_log() * torch.log(tt / self.t0) + self.c_lin() * tt

        if self.mode == 'log':
            return torch.tensor(self.c_log_const, device=device, dtype=dtype) * torch.log(tt / self.t0)
        if self.mode == 'linear':
            return tt * torch.tensor(self.c_lin_const, device=device, dtype=dtype)
        return (
            torch.tensor(self.c_log_const, device=device, dtype=dtype) * torch.log(tt / self.t0)
            + tt * torch.tensor(self.c_lin_const, device=device, dtype=dtype)
        )

    def eta(self, t: float, device, dtype) -> torch.Tensor:
        e = self._eta_unclipped(t, device, dtype)
        return torch.clamp(e, -self.eta_clip, self.eta_clip)

    def exp_eta(self, t: float, device, dtype) -> torch.Tensor:
        return torch.exp(self.eta(t, device, dtype))

    def exp_minus_eta(self, t: float, device, dtype) -> torch.Tensor:
        return torch.exp(-self.eta(t, device, dtype))

    def alpha(self, t: float, device, dtype) -> torch.Tensor:
        """Return alpha(t)=d/dt eta(t) for the supported schedule families."""
        if t <= 0:
            t = 1e-8
        tt = torch.tensor(t, device=device, dtype=dtype)

        if self.learnable:
            if self.mode == 'log':
                return self.c_log() / tt
            if self.mode == 'linear':
                return self.c_lin().to(device=device, dtype=dtype)
            return self.c_log() / tt + self.c_lin().to(device=device, dtype=dtype)

        if self.mode == 'log':
            return torch.tensor(self.c_log_const, device=device, dtype=dtype) / tt
        if self.mode == 'linear':
            return torch.tensor(self.c_lin_const, device=device, dtype=dtype)
        return torch.tensor(self.c_log_const, device=device, dtype=dtype) / tt + torch.tensor(self.c_lin_const, device=device, dtype=dtype)

    def delta_eta(self, t: float, dt: float, device, dtype) -> torch.Tensor:
        """Compute eta(t+dt)-eta(t), clamped to [-eta_clip, eta_clip].

        Used for exact friction factors exp(-(eta(t+dt)-eta(t))).
        """
        e1 = self._eta_unclipped(t + dt, device, dtype)
        e0 = self._eta_unclipped(t, device, dtype)
        d = e1 - e0
        return torch.clamp(d, -self.eta_clip, self.eta_clip)

    def delta_eta_tensor(self, t: float, dt: torch.Tensor, device, dtype) -> torch.Tensor:
        """Like delta_eta but dt is a live tensor, so gradients flow through hY.

        This enables sigma = exp(-delta_eta_tensor(...)) to backpropagate into
        theta_hY, giving the correct full gradient for the exponential-Euler
        momentum update  Pk1 = sigma*Pk + w*Gv.

        Using the detached float version (delta_eta) severs ~84% of the true
        d(Pk1)/d(hY) gradient at initialization because d(sigma)/d(hY)*Pk
        is never computed.
        """
        tt0 = torch.tensor(t, device=device, dtype=dtype)
        tt1 = tt0 + dt                    # tt1 carries gradient w.r.t. dt
        if self.learnable:
            if self.mode == 'log':
                d = self.c_log() * torch.log(tt1 / tt0)
            elif self.mode == 'linear':
                d = self.c_lin().to(device=device, dtype=dtype) * dt
            else:  # loglin
                d = self.c_log() * torch.log(tt1 / tt0) + self.c_lin().to(device=device, dtype=dtype) * dt
        else:
            tt0_c = torch.tensor(self.c_log_const, device=device, dtype=dtype)
            tt1_c = torch.tensor(self.c_lin_const, device=device, dtype=dtype)
            if self.mode == 'log':
                d = tt0_c * torch.log(tt1 / tt0)
            elif self.mode == 'linear':
                d = tt1_c * dt
            else:  # loglin
                d = tt0_c * torch.log(tt1 / tt0) + tt1_c * dt
        return torch.clamp(d, -self.eta_clip, self.eta_clip)

def _get_eta_coefs(sched):
    """Return (c_log, c_lin) as Python floats from an EtaSchedule instance."""
    if sched.learnable:
        c_log = float(sched.c_log().detach().cpu().item()) if sched.c_log is not None else 0.0
        c_lin = float(sched.c_lin().detach().cpu().item()) if sched.c_lin is not None else 0.0
    else:
        c_log = getattr(sched, "c_log_const", 0.0)
        c_lin = getattr(sched, "c_lin_const", 0.0)
    return c_log, c_lin


class PresymplecticSoftmaxAttention(nn.Module):
    '''Explicit 2nd-order presymplectic integrator (variable doubling) using standard causal self-attention projections (Q,K,V). Note: Q and K are not forced equal; symmetry assumptions from the theory are not enforced here.'''
    def __init__(
        self,
        cfg: ModelConfig,
        h: float = 1.0,
        xi: float = 1.0,
        # Data-driven tuning of xi based on mismatch of doubled variables
        # xi_adapt: bool = False,
        r_thresh: float = 1e-2,
        r_low: float = 1e-4,
        xi_mult_up: float = 1.25,
        xi_mult_down: float = 0.5,
        xi_min: float = 1e-4,
        xi_max: float = 100.0,
        theta_max: float = 1.0,
        # xi_adapt_warmup: int = 10,
        # xi_adapt_every: int = 1,
        t0: float = 1.0,
        eta_mu: Optional[float] = None,
        eta_log_coef: Optional[float] = None,
        eta_lin_coef: Optional[float] = None,
        eta_log_init: Optional[float] = None,
        eta_lin_init: Optional[float] = None,
        eta_learnable: bool = False,
        eta_mode: str = "log",
        eta_init: Optional[float] = None,
        eta_clip: float = 50.0,
        eps: float = 1e-8,
        causal: bool = True,
        presymp_lnp: str = "end",
        lookahead: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.lookahead = bool(lookahead)

        # Separate learned step sizes for position (X) and momentum (P):
        #   hX = softplus(theta_hX) > 0  — governs position updates  dot X = F(X,P)
        #   hY = softplus(theta_hY) > 0  — governs momentum updates  dot P = G(X,P) - alpha P
        # Both initialised to h so behaviour is unchanged when h_X = h_Y.
        # h() is kept as an alias for hX() for backward-compat with logging/AB2 code.
        #   xi = (theta_max/(2*hX)) * sigmoid(theta_xi_raw)
        self.theta_hX = nn.Parameter(torch.tensor(inv_softplus(h), dtype=torch.float32))
        self.theta_hY = nn.Parameter(torch.tensor(inv_softplus(h), dtype=torch.float32))
        # Legacy alias so that external callers of .theta_h still work.
        self.theta_h = self.theta_hX
        self.theta_xi_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))  # initialized below
        # self.xi_adapt = bool(xi_adapt)
        self.r_thresh = float(r_thresh)
        self.r_low = float(r_low)
        self.xi_mult_up = float(xi_mult_up)
        self.xi_mult_down = float(xi_mult_down)
        self.xi_min = float(xi_min)
        self.xi_max = float(xi_max)
        self.theta_max = float(theta_max)
        # self.xi_adapt_warmup = int(max(0, xi_adapt_warmup))

        # initialize theta_xi_raw so that xi(h0)=xi0 (up to cap)
        h0 = float(h) if float(h) > 0 else 1.0
        xi0 = float(xi)
        if self.theta_max > 0:
            xi_cap0 = self.theta_max / (2.0 * h0)
            frac = xi0 / max(xi_cap0, 1e-12)
            self.theta_xi_raw.data = torch.tensor(inv_sigmoid(frac), dtype=torch.float32)
        else:
            self.theta_xi_raw.data = torch.tensor(inv_softplus(xi0), dtype=torch.float32)

        # self.xi_adapt_every = int(max(1, xi_adapt_every))
        # self._xi_adapt_ctr = 0
        # last diagnostics (max over batch)
        self.last_rX = 0.0
        self.last_rP = 0.0
        self.last_h = float(F.softplus(self.theta_h).detach().cpu().item())
        self.eps = float(eps)
        self.causal = bool(causal)
        # Track the effective xi after any initial capping.
        self.last_xi = float(self.xi(h=torch.tensor(float(h0), dtype=torch.float32)).detach().cpu().item())
        self.last_xi_changed = False

        self.ln = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.presymp_lnp = str(presymp_lnp)
        if self.presymp_lnp not in ("none", "end", "each_substep"):
            raise ValueError(f"presymp_lnp must be one of none|end|each_substep, got {self.presymp_lnp}")
        # LayerNorm on momentum stream (LNp). Applied either at end of the step or after each substep.
        self.ln_p = LayerNorm(cfg.n_embd, bias=cfg.bias) if self.presymp_lnp != "none" else nn.Identity()
        self.sched = EtaSchedule(t0=t0, mu=eta_mu, log_coef=eta_log_coef, lin_coef=eta_lin_coef, learnable=eta_learnable, mode=eta_mode, init=eta_init, init_log=eta_log_init, init_lin=eta_lin_init, eta_clip=eta_clip)

        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Standard attention projections (not enforcing Q=K symmetry)
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        # B matrix: F-oracle is  dot X_i = B P_i / z_i
        # Theory: B = Sym(V A^{-1}).  We parametrise freely; identity init = previous behaviour.
        self.c_B = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        nn.init.eye_(self.c_B.weight)
        # Lookahead coefficient: oracle evaluated at X + mu_la * P.  Init near 0 = no-op.
        self.mu_la = ConstrainedScalar(0.001, "unit")
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def _qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Q,K,V with standard projections.
        Input x: (B, T, C). Returns q,k,v in (B, nh, T, hd).
        """
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        return q, k, v

    def hX(self, device=None, dtype=None):
        """Step size for position updates  X^{k+1} = X^k + hX * vel."""
        h = F.softplus(self.theta_hX)
        if device is not None or dtype is not None:
            h = h.to(device=device if device is not None else h.device,
                     dtype=dtype if dtype is not None else h.dtype)
        return h

    def hY(self, device=None, dtype=None):
        """Step size for momentum updates  P^{k+1} = P^k + hY * force."""
        h = F.softplus(self.theta_hY)
        if device is not None or dtype is not None:
            h = h.to(device=device if device is not None else h.device,
                     dtype=dtype if dtype is not None else h.dtype)
        return h

    def h(self, device=None, dtype=None):
        """Backward-compat alias for hX() — used by logging and schedule indexing."""
        return self.hX(device=device, dtype=dtype)

    def xi(self, h, device=None, dtype=None):
        if not torch.is_tensor(h):
            h = torch.tensor(float(h), device=device, dtype=dtype)
        h = h.clamp_min(self.eps)
        if self.theta_max > 0:
            xi_max = (self.theta_max / (2.0 * h))
            xi = xi_max * torch.sigmoid(self.theta_xi_raw)
        else:
            xi = F.softplus(self.theta_xi_raw)
        xi = torch.clamp(xi, min=self.xi_min, max=self.xi_max)
        if device is not None or dtype is not None:
            xi = xi.to(device=device if device is not None else xi.device,
                       dtype=dtype if dtype is not None else xi.dtype)
        return xi


    def _kernel_E_z(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute E_ij = exp(<q_i, k_j>/sqrt(d)) and z_i = mean_j E_ij.
        We aggregate heads by averaging the score matrices over heads, yielding a single (B,T,T) kernel.
        """
        B, nh, T, hd = q.shape
        S = (q @ k.transpose(-2, -1)) * self.scale  # (B, nh, T, T)

        if self.causal:
            m = causal_mask(T, q.device)  # compute mask once
            S = S.masked_fill(m.unsqueeze(0).unsqueeze(0), float("-inf"))

        S = S.mean(dim=1)  # (B, T, T)

        # clamp for numerical safety
        S = torch.clamp(S, min=-60.0, max=60.0)
        E = torch.exp(S)
        if self.causal:
            E = E.masked_fill(m.unsqueeze(0), 0.0)  # reuse m

        z = E.sum(dim=-1) / float(T)
        z = z.clamp_min(self.eps)
        return E, z

    def _apply_lnp(self, P: torch.Tensor) -> torch.Tensor:
        """Apply LayerNorm to a momentum-like tensor (B,T,C).

        Uses float32 internally for stability (especially under bf16).
        """
        if self.presymp_lnp == "none":
            return P
        out = self.ln_p(P.float())
        return out.to(dtype=P.dtype)


    def _la_input(self, X: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Return oracle evaluation point: X + mu_la*p when lookahead is on, else X."""
        if not self.lookahead:
            return X
        return X + self.mu_la() * p

    def _vel(self, t: float, X: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
        device, dtype = X.device, X.dtype
        lam = self.sched.exp_minus_eta(t, device, dtype)
        p = lam * Pi

        x_ln = self.ln(self._la_input(X, p))
        q, k, _ = self._qkv(x_ln)
        _, z = self._kernel_E_z(q, k)
        return self.c_B(p) / z.unsqueeze(-1)

    def _force(self, t: float, X: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
        device, dtype = X.device, X.dtype
        Lam = self.sched.exp_eta(t, device, dtype)
        lam = self.sched.exp_minus_eta(t, device, dtype)
        p = lam * Pi  # physical momentum

        x_ln = self.ln(self._la_input(X, p))
        q, k, v = self._qkv(x_ln)
        E, z = self._kernel_E_z(q, k)

        a = (p * p).sum(dim=-1)  # (B, T)
        s = a / (z * z + self.eps)
        M = E * (s.unsqueeze(-1) + s.unsqueeze(-2) - 2.0)  # (B, T, T)

        # Use standard V projection as the "values" being aggregated, then output-projection.
        Bsz, T, C = X.shape
        v_merge = v.transpose(1, 2).contiguous().view(Bsz, T, C)  # (B, T, C)
        core = torch.matmul(M, v_merge) / (2.0 * float(T))
        core = self.resid_drop(self.c_proj(core))
        return Lam * core

    def _oracle(self, t: float, X: torch.Tensor, Pi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Combined oracle: returns (vel, force) at (t, X, Pi) with a single kernel evaluation.

        Replaces calling _vel and _force separately on the same (X, Pi), which would
        compute ln(X), QKV, and the kernel twice. The four substep pairs in step() all
        share identical (X, Pi) arguments within each pair, so this halves oracle calls
        from 8 to 4.
        """
        device, dtype = X.device, X.dtype
        Lam = self.sched.exp_eta(t, device, dtype)
        lam = self.sched.exp_minus_eta(t, device, dtype)
        p = lam * Pi  # physical momentum

        x_ln = self.ln(self._la_input(X, p))
        q, k, v = self._qkv(x_ln)
        E, z = self._kernel_E_z(q, k)

        # velocity: dot X_i = B p_i / z_i  (B = c_B; lookahead shifts evaluation point)
        vel = self.c_B(p) / z.unsqueeze(-1)

        # force: dot Pi = Lam * G(X, p)
        a = (p * p).sum(dim=-1)
        s = a / (z * z + self.eps)
        M = E * (s.unsqueeze(-1) + s.unsqueeze(-2) - 2.0)
        B, T, C = X.shape
        v_merge = v.transpose(1, 2).contiguous().view(B, T, C)
        core = torch.matmul(M, v_merge) / (2.0 * float(T))
        core = self.resid_drop(self.c_proj(core))
        force = Lam * core

        return vel, force

    def step(self, Xk: torch.Tensor, Pk: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        device, dtype = Xk.device, Xk.dtype
        h_x = self.hX(device=device, dtype=dtype)      # position step size tensor
        h_y = self.hY(device=device, dtype=dtype)      # momentum step size tensor
        # Use hX for schedule time-indexing and for the phi_C coupling angle.
        h_f = h_x.detach().cpu().item()
        tau_x = 0.5 * h_x                              # position substep
        tau_y = 0.5 * h_y                              # momentum substep
        t0 = self.sched.t0
        tk = t0 + float(k) * h_f
        tk1 = tk + h_f

        Lam_tk = self.sched.exp_eta(tk, device, dtype)
        Pi = Lam_tk * Pk

        t = tk
        X = Xk
        bar_t = tk
        bar_X = Xk
        bar_Pi = Pi

        # coupling: xi and rotation angle use hX (the position scale)
        xi_t = self.xi(h=h_x, device=device, dtype=dtype)  # tensor
        theta = 2.0 * xi_t * h_x                           # tensor
        c = torch.cos(theta)                                # tensor (scalar)
        s = torch.sin(theta)                                # tensor (scalar)

        # phi_A^{tau}: single oracle at (t, X, bar_Pi) — momentum half-step
        vel, force = self._oracle(t, X, bar_Pi)
        Pi = Pi + tau_y * force
        if self.presymp_lnp == "each_substep":
            Pi = self._apply_lnp(Pi)
        bar_t = bar_t + h_f * 0.5
        bar_X = bar_X + tau_x * vel

        # phi_B^{tau}: single oracle at (bar_t, bar_X, Pi) — position half-step
        t = t + h_f * 0.5
        vel, force = self._oracle(bar_t, bar_X, Pi)
        X = X + tau_x * vel
        bar_Pi = bar_Pi + tau_y * force
        if self.presymp_lnp == "each_substep":
            bar_Pi = self._apply_lnp(bar_Pi)

        # phi_C^{h}
        dX = X - bar_X
        dPi = Pi - bar_Pi
        sX = X + bar_X
        sPi = Pi + bar_Pi

        X_new = 0.5 * (sX + c * dX + s * dPi)
        Pi_new = 0.5 * (sPi - s * dX + c * dPi)
        bar_X_new = 0.5 * (sX - c * dX - s * dPi)
        bar_Pi_new = 0.5 * (sPi + s * dX - c * dPi)
        X, Pi, bar_X, bar_Pi = X_new, Pi_new, bar_X_new, bar_Pi_new
        if self.presymp_lnp == "each_substep":
            Pi = self._apply_lnp(Pi)
            bar_Pi = self._apply_lnp(bar_Pi)

        # phi_B^{tau}: single oracle at (bar_t, bar_X, Pi) — post-phi_C states
        t = t + h_f * 0.5
        vel, force = self._oracle(bar_t, bar_X, Pi)
        X = X + tau_x * vel
        bar_Pi = bar_Pi + tau_y * force

        # phi_A^{tau}: single oracle at (t, X, bar_Pi) — post-phi_B states
        vel, force = self._oracle(t, X, bar_Pi)
        Pi = Pi + tau_y * force
        bar_t = bar_t + h_f * 0.5
        bar_X = bar_X + tau_x * vel

        lam_tk1 = self.sched.exp_minus_eta(tk1, device, dtype)
        Pk1_raw = lam_tk1 * Pi
        bar_Pk1_raw = lam_tk1 * bar_Pi
        if self.presymp_lnp in ("end", "each_substep"):
            Pk1 = self._apply_lnp(Pk1_raw)
            bar_Pk1 = self._apply_lnp(bar_Pk1_raw)
        else:
            Pk1, bar_Pk1 = Pk1_raw, bar_Pk1_raw

        self.last_h = h_f
        self.last_xi = xi_t.detach().cpu().item()  # use already-computed tensor
        self.last_xi_changed = False

        return X, Pk1


class DampedEulerAttention(nn.Module):
    """
    Explicit Euler discretization of the damped finite-dimensional system.

    We integrate
        dot X = F(X,P),\quad \dot P = G(X,P) - alpha(t) P
    by
        X_{k+1} = X_k + h F(X_k,P_k),
        P_{k+1} = P_k + h (G(X_k,P_k) - alpha(t_k) P_k).

    This is meant as a simple baseline discretization of the same vector field used
    inside the presymplectic block (but without any geometric structure preservation).
    """

    def __init__(
        self,
        cfg: ModelConfig,
        h: float = 1.0,
        t0: float = 1.0,
        eta_mu: Optional[float] = None,
        eta_log_coef: Optional[float] = None,
        eta_lin_coef: Optional[float] = None,
        eta_log_init: Optional[float] = None,
        eta_lin_init: Optional[float] = None,
        eta_learnable: bool = False,
        eta_mode: str = "log",
        eta_init: Optional[float] = None,
        eta_clip: float = 50.0,
        eps: float = 1e-8,
        causal: bool = True,
        presymp_lnp: str = "end",
        lookahead: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.lookahead = bool(lookahead)
        # Separate learned step sizes for position (X) and momentum (P).
        self.theta_hX = nn.Parameter(torch.tensor(inv_softplus(h), dtype=torch.float32))
        self.theta_hY = nn.Parameter(torch.tensor(inv_softplus(h), dtype=torch.float32))
        # Legacy alias: keeps .theta_h attribute intact for any external code.
        self.theta_h = self.theta_hX
        self.eps = float(eps)
        self.causal = bool(causal)

        self.ln = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.presymp_lnp = str(presymp_lnp)
        if self.presymp_lnp not in ("none", "end", "each_substep"):
            raise ValueError(f"presymp_lnp must be one of none|end|each_substep, got {self.presymp_lnp}")
        self.ln_p = LayerNorm(cfg.n_embd, bias=cfg.bias) if self.presymp_lnp != "none" else nn.Identity()
        self.sched = EtaSchedule(t0=t0, mu=eta_mu, log_coef=eta_log_coef, lin_coef=eta_lin_coef, learnable=eta_learnable, mode=eta_mode, init=eta_init, init_log=eta_log_init, init_lin=eta_lin_init, eta_clip=eta_clip)

        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.c_B = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        nn.init.eye_(self.c_B.weight)
        self.mu_la = ConstrainedScalar(0.001, "unit")
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        # diagnostics for uniform logging
        self.last_rX = 0.0
        self.last_rP = 0.0
        self.last_h = float(F.softplus(self.theta_h).detach().cpu().item())
        self.last_hY = float(F.softplus(self.theta_hY).detach().cpu().item())
        self.last_xi = 0.0

    def _qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        return q, k, v

    def hX(self, device=None, dtype=None):
        """Step size for position updates  X^{k+1} = X^k + hX * F(X,P)."""
        h = F.softplus(self.theta_hX)
        if device is not None or dtype is not None:
            h = h.to(device=device if device is not None else h.device,
                     dtype=dtype if dtype is not None else h.dtype)
        return h

    def hY(self, device=None, dtype=None):
        """Step size for momentum updates  P^{k+1} = P^k + hY * (...)."""
        h = F.softplus(self.theta_hY)
        if device is not None or dtype is not None:
            h = h.to(device=device if device is not None else h.device,
                     dtype=dtype if dtype is not None else h.dtype)
        return h

    def h(self, device=None, dtype=None):
        """Backward-compat alias for hX()."""
        return self.hX(device=device, dtype=dtype)

    def _kernel_E_z(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute E_ij = exp(<q_i, k_j>/sqrt(d)) and z_i = mean_j E_ij.
        We aggregate heads by averaging the score matrices over heads, yielding a single (B,T,T) kernel.
        """
        B, nh, T, _ = q.shape
        S = (q @ k.transpose(-2, -1)) * self.scale  # (B, nh, T, T)
        if self.causal:
            m = causal_mask(T, q.device)  # compute mask once
            S = S.masked_fill(m.unsqueeze(0).unsqueeze(0), float("-inf"))
        S = S.mean(dim=1)  # (B,T,T)
        S = torch.clamp(S, min=-60.0, max=60.0)
        E = torch.exp(S)
        if self.causal:
            E = E.masked_fill(m.unsqueeze(0), 0.0)  # reuse m
        z = E.sum(dim=-1) / float(T)
        z = z.clamp_min(self.eps)
        return E, z

    def _apply_lnp(self, P: torch.Tensor) -> torch.Tensor:
        if self.presymp_lnp == "none":
            return P
        out = self.ln_p(P.float())
        return out.to(dtype=P.dtype)
    def _la_input(self, X: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        """Return oracle evaluation point: X + mu_la*P when lookahead is on, else X."""
        if not self.lookahead:
            return X
        return X + self.mu_la() * P

    def _F_E_z_vmerge(self, X: torch.Tensor, P: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Returns (F, E, z, v_merge). When lookahead=True, QKV and hence
        # z and G are evaluated at X + mu_la*P instead of X.
        x_ln = self.ln(self._la_input(X, P))
        q, k, v = self._qkv(x_ln)
        E, z = self._kernel_E_z(q, k)
        B, T, C = X.shape
        v_merge = v.transpose(1, 2).contiguous().view(B, T, C)
        Fv = self.c_B(P) / z.unsqueeze(-1)
        return Fv, E, z, v_merge

    def _G_from_cache(self, P: torch.Tensor, E: torch.Tensor, z: torch.Tensor, v_merge: torch.Tensor) -> torch.Tensor:
        # Conservative force term G (without damping), using cached (E,z,v_merge).
        a = (P * P).sum(dim=-1)
        s = a / (z * z + self.eps)
        M = E * (s.unsqueeze(-1) + s.unsqueeze(-2) - 2.0)
        core = torch.matmul(M, v_merge) / (2.0 * float(P.shape[1]))
        core = self.resid_drop(self.c_proj(core))
        return core

    def FG_alpha(self, X: torch.Tensor, P: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Return (F, G, z, alpha) at (X,P) for layer index k.
        device, dtype = X.device, X.dtype
        h = self.h().item()
        tk = self.sched.t0 + float(k) * h
        Fv, E, z, v_merge = self._F_E_z_vmerge(X, P)
        Gv = self._G_from_cache(P, E=E, z=z, v_merge=v_merge)
        alpha = self.sched.alpha(tk, device, dtype)
        return Fv, Gv, z, alpha

    def rhs(self, X: torch.Tensor, P: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Right-hand side of the damped system: (dX, dP) with dP including -alpha P.
        Fv, Gv, _z, alpha = self.FG_alpha(X, P, k)
        return Fv, (Gv - alpha * P)


    def step(self, Xk: torch.Tensor, Pk: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        device, dtype = Xk.device, Xk.dtype
        hX_t = self.hX()               # tensor for position update — keeps gradient
        hY_t = self.hY()               # tensor for momentum update
        hX_f = hX_t.detach().item()    # float — schedule time indexing only

        Fv, Gv, z, alpha = self.FG_alpha(Xk, Pk, k)
        Xk1 = Xk + hX_t * Fv
        Pk1 = Pk + hY_t * (Gv - alpha * Pk)
        if self.presymp_lnp != "none":
            Pk1 = self._apply_lnp(Pk1)
        self.last_h = hX_f
        self.last_hY = hY_t.detach().item()
        return Xk1, Pk1




class DampedExpEulerAttention(DampedEulerAttention):
    # Integrating-factor / exponential Euler discretization of the damped system.
    # Treat damping exactly over one step using sigma = exp(-(eta(t+h)-eta(t))).
    # hX governs the position update; hY governs the momentum update (and sigma).

    def step(self, Xk: torch.Tensor, Pk: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        device, dtype = Xk.device, Xk.dtype
        hX_t = self.hX()               # position step size (tensor)
        hY_t = self.hY()               # momentum step size (tensor)
        hX_f = hX_t.detach().item()    # float — schedule time indexing only

        tk = self.sched.t0 + float(k) * hX_f

        # one oracle call at (Xk, Pk); Fv already incorporates c_B and the lookahead point.
        Fv, Gv, z, _alpha = self.FG_alpha(Xk, Pk, k)

        # Exact discrete damping over interval hY.
        # Use delta_eta_tensor (hY_t live) so that sigma = exp(-d_eta) backprops
        # into theta_hY.  The previously used hY_f = hY_t.detach().item() severed
        # the d(sigma)/d(hY)*Pk term, which is ~84% of the true gradient at init.
        d_eta = self.sched.delta_eta_tensor(tk, hY_t, device, dtype)
        sigma = torch.exp(-d_eta)
        alpha_eff = d_eta / hY_t.clamp(min=1e-8)
        eps = torch.tensor(1e-8, device=device, dtype=dtype)
        w = torch.where(alpha_eff.abs() > eps, (1.0 - sigma) / alpha_eff, hY_t)

        Pk1 = sigma * Pk + w * Gv
        # Position uses old momentum (exponential Euler: p^k, not p^{k+1}).
        Xk1 = Xk + hX_t * Fv
        if self.presymp_lnp != "none":
            Pk1 = self._apply_lnp(Pk1)
        self.last_h = hX_f
        self.last_hY = hY_t.detach().item()
        return Xk1, Pk1

class PlainEulerAttention(DampedEulerAttention):
    """Non-symplectic plain explicit Euler discretization of the damped Hamiltonian.

    Eq. (plain_explicit_Euler) in the paper:
        q^{k+1} = q^k + h  F(q^k, p^k)
        p^{k+1} = alpha_k * p^k + h G(q^k, p^k)

    alpha_k is a learned per-layer scalar in (0,1) (ConstrainedScalar "unit").
    Unlike the conformally-symplectic Euler, the position update here uses p^k
    rather than p^{k+1}, so the step has NO geometric structure preservation.
    """

    def __init__(self, *args, alpha_init: float = 0.9, **kwargs):
        super().__init__(*args, **kwargs)
        # Learned multiplicative damping coefficient in (0,1).
        self.alpha_plain = ConstrainedScalar(alpha_init, "unit")

    def step(self, Xk: torch.Tensor, Pk: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        device, dtype = Xk.device, Xk.dtype
        hX_t = self.hX()
        hY_t = self.hY()
        hX_f = hX_t.detach().item()

        Fv, Gv, _z, _alpha = self.FG_alpha(Xk, Pk, k)

        # Plain Euler: both position and momentum use state at step k (no geometric structure)
        Xk1 = Xk + hX_t * Fv
        alpha_k = self.alpha_plain()      # learned scalar in (0,1)
        Pk1 = alpha_k * Pk + hY_t * Gv

        if self.presymp_lnp != "none":
            Pk1 = self._apply_lnp(Pk1)
        self.last_h = hX_f
        self.last_hY = hY_t.detach().item()
        return Xk1, Pk1
    """Half-damping Strang splitting for the damped system.

    One step is
      P <- sigma^{1/2} P
      (X,P) <- Psi_h(X,P)   [conservative Hamiltonian step]
      P <- sigma^{1/2} P

    Here Psi_h is implemented via the *explicit* variable-doubling method for
    nonseparable Hamiltonians (same map structure as in FJV §6.2), but applied to
    the conservative vector field (no conformal time-dependence).
    """

    def __init__(
        self,
        cfg: ModelConfig,
        h: float = 1.0,
        xi: float = 1.0,
        theta_max: float = 1.0,
        t0: float = 1.0,
        eta_mu: Optional[float] = None,
        eta_log_coef: Optional[float] = None,
        eta_lin_coef: Optional[float] = None,
        eta_log_init: Optional[float] = None,
        eta_lin_init: Optional[float] = None,
        eta_learnable: bool = False,
        eta_mode: str = "log",
        eta_init: Optional[float] = None,
        eta_clip: float = 50.0,
        eps: float = 1e-8,
        causal: bool = True,
        presymp_lnp: str = "end",
        lookahead: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.lookahead = bool(lookahead)

        self.theta_h = nn.Parameter(torch.tensor(inv_softplus(h), dtype=torch.float32))
        self.theta_xi_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.theta_max = float(theta_max)
        self.eps = float(eps)
        self.causal = bool(causal)

        h0 = float(h) if float(h) > 0 else 1.0
        xi0 = float(xi)
        if self.theta_max > 0:
            xi_cap0 = self.theta_max / (2.0 * h0)
            frac = xi0 / max(xi_cap0, 1e-12)
            self.theta_xi_raw.data = torch.tensor(inv_sigmoid(frac), dtype=torch.float32)
        else:
            self.theta_xi_raw.data = torch.tensor(inv_softplus(xi0), dtype=torch.float32)

        self.ln = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.presymp_lnp = str(presymp_lnp)
        if self.presymp_lnp not in ("none", "end", "each_substep"):
            raise ValueError(f"presymp_lnp must be one of none|end|each_substep, got {self.presymp_lnp}")
        self.ln_p = LayerNorm(cfg.n_embd, bias=cfg.bias) if self.presymp_lnp != "none" else nn.Identity()
        self.sched = EtaSchedule(t0=t0, mu=eta_mu, log_coef=eta_log_coef, lin_coef=eta_lin_coef, learnable=eta_learnable, mode=eta_mode, init=eta_init, init_log=eta_log_init, init_lin=eta_lin_init, eta_clip=eta_clip)

        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.c_B = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        nn.init.eye_(self.c_B.weight)
        self.mu_la = ConstrainedScalar(0.001, "unit")
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        # diagnostics for uniform logging
        self.last_rX = 0.0
        self.last_rP = 0.0
        self.last_h = float(F.softplus(self.theta_h).detach().cpu().item())
        self.last_xi = float(self.xi(h=torch.tensor(float(h0), dtype=torch.float32)).detach().cpu().item())

    def _qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        return q, k, v

    def h(self, device=None, dtype=None):
        h = F.softplus(self.theta_h)
        if device is not None or dtype is not None:
            h = h.to(device=device if device is not None else h.device,
                     dtype=dtype if dtype is not None else h.dtype)
        return h

    def xi(self, h, device=None, dtype=None):
        if not torch.is_tensor(h):
            h = torch.tensor(float(h), device=device, dtype=dtype)
        h = h.clamp_min(self.eps)
        if self.theta_max > 0:
            xi_max = (self.theta_max / (2.0 * h))
            xi = xi_max * torch.sigmoid(self.theta_xi_raw)
        else:
            xi = F.softplus(self.theta_xi_raw)
        if device is not None or dtype is not None:
            xi = xi.to(device=device if device is not None else xi.device,
                       dtype=dtype if dtype is not None else xi.dtype)
        return xi


    def _kernel_E_z(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, nh, T, _ = q.shape
        S = (q @ k.transpose(-2, -1)) * self.scale
        if self.causal:
            m = causal_mask(T, q.device)  # compute mask once
            S = S.masked_fill(m.unsqueeze(0).unsqueeze(0), float("-inf"))
        S = S.mean(dim=1)
        S = torch.clamp(S, min=-60.0, max=60.0)
        E = torch.exp(S)
        if self.causal:
            E = E.masked_fill(m.unsqueeze(0), 0.0)  # reuse m
        z = E.sum(dim=-1) / float(T)
        z = z.clamp_min(self.eps)
        return E, z

    def _apply_lnp(self, P: torch.Tensor) -> torch.Tensor:
        if self.presymp_lnp == "none":
            return P
        out = self.ln_p(P.float())
        return out.to(dtype=P.dtype)


    def _la_input(self, X: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        """Return oracle evaluation point: X + mu_la*P when lookahead is on, else X."""
        if not self.lookahead:
            return X
        return X + self.mu_la() * P

    def _velH(self, X: torch.Tensor, P: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (F, E, z, v_merge) for conservative dynamics.
        When lookahead=True, QKV is evaluated at X + mu_la*P.
        """
        x_ln = self.ln(self._la_input(X, P))
        q, k, v = self._qkv(x_ln)
        E, z = self._kernel_E_z(q, k)
        B, T, C = X.shape
        v_merge = v.transpose(1, 2).contiguous().view(B, T, C)
        Fv = self.c_B(P) / z.unsqueeze(-1)
        return Fv, E, z, v_merge

    def _forH(self, X: torch.Tensor, P: torch.Tensor, E: torch.Tensor, z: torch.Tensor, v_merge: torch.Tensor) -> torch.Tensor:
        a = (P * P).sum(dim=-1)
        s = a / (z * z + self.eps)
        M = E * (s.unsqueeze(-1) + s.unsqueeze(-2) - 2.0)
        core = torch.matmul(M, v_merge) / (2.0 * float(X.shape[1]))
        core = self.resid_drop(self.c_proj(core))
        return core

    def _conservative_doubling_step(self, X0: torch.Tensor, P0: torch.Tensor, h_t: torch.Tensor, xi_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Explicit 2nd-order doubling-trick integrator for the conservative part.
        h_t and xi_t are passed in as tensors from step() so gradients reach theta_h/theta_xi_raw.
        """
        h_f = h_t.detach().item()  # float, only for non-differentiable schedule lookups
        tau = 0.5 * h_t            # tensor
        theta = 2.0 * xi_t * h_t  # tensor
        c = torch.cos(theta)       # tensor (scalar)
        s = torch.sin(theta)       # tensor (scalar)

        X = X0
        P = P0
        bar_X = X0
        bar_P = P0

        # phi_A^{tau}
        Fv, E, z, v_merge = self._velH(X, bar_P)
        P = P + tau * self._forH(X, bar_P, E=E, z=z, v_merge=v_merge)
        if self.presymp_lnp == "each_substep":
            P = self._apply_lnp(P)
        bar_X = bar_X + tau * Fv

        # phi_B^{tau}
        Fv2, E2, z2, v_merge2 = self._velH(bar_X, P)
        X = X + tau * Fv2
        bar_P = bar_P + tau * self._forH(bar_X, P, E=E2, z=z2, v_merge=v_merge2)
        if self.presymp_lnp == "each_substep":
            bar_P = self._apply_lnp(bar_P)

        # phi_C^{h}
        dX = X - bar_X
        dP = P - bar_P
        sX = X + bar_X
        sP = P + bar_P
        X_new = 0.5 * (sX + c * dX + s * dP)
        P_new = 0.5 * (sP - s * dX + c * dP)
        bar_X_new = 0.5 * (sX - c * dX - s * dP)
        bar_P_new = 0.5 * (sP + s * dX - c * dP)
        X, P, bar_X, bar_P = X_new, P_new, bar_X_new, bar_P_new
        if self.presymp_lnp == "each_substep":
            P = self._apply_lnp(P)
            bar_P = self._apply_lnp(bar_P)

        # phi_B^{tau}
        Fv3, E3, z3, v_merge3 = self._velH(bar_X, P)
        X = X + tau * Fv3
        bar_P = bar_P + tau * self._forH(bar_X, P, E=E3, z=z3, v_merge=v_merge3)
        if self.presymp_lnp == "each_substep":
            bar_P = self._apply_lnp(bar_P)

        # phi_A^{tau}
        Fv4, E4, z4, v_merge4 = self._velH(X, bar_P)
        P = P + tau * self._forH(X, bar_P, E=E4, z=z4, v_merge=v_merge4)
        if self.presymp_lnp == "each_substep":
            P = self._apply_lnp(P)
        bar_X = bar_X + tau * Fv4

        return X, P

    def step(self, Xk: torch.Tensor, Pk: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        device, dtype = Xk.device, Xk.dtype
        h_t = self.h(device=device, dtype=dtype)                      # tensor — stays in graph
        h_f = h_t.detach().cpu().item()                               # float — only for schedule time index
        xi_t = self.xi(h=h_t, device=device, dtype=dtype)            # tensor — stays in graph
        tk = self.sched.t0 + float(k) * h_f

        # exact half damping
        d_eta = self.sched.delta_eta(tk, 0.5 * h_f, device, dtype)
        sigma_half = torch.exp(-d_eta)  # scalar tensor
        P_half = sigma_half * Pk

        # conservative step — pass tensors so gradients reach theta_h and theta_xi_raw
        X1, P1 = self._conservative_doubling_step(Xk, P_half, h_t=h_t, xi_t=xi_t)

        # exact half damping
        Pk1 = sigma_half * P1
        if self.presymp_lnp != "none":
            Pk1 = self._apply_lnp(Pk1)
        self.last_h = h_f
        self.last_xi = xi_t.detach().cpu().item()  # use already-computed tensor
        return X1, Pk1



class PresympGPTBlock(nn.Module):
    '''Replace attention with presymplectic step; accelerate MLP with a Nesterov-style velocity stream.'''
    def __init__(
        self,
        cfg: ModelConfig,
        mlp_use_attn_vel: bool = False,
        mlp_use_p_vel: bool = False,
        no_mlp: bool = False,
        attn_scheme: str = "presymp",
        lookahead: bool = False,
        h: float = 1.0,
        xi: float = 1.0,
        t0: float = 1.0,
        eta_mu: Optional[float] = None,
        eta_log_coef: Optional[float] = None,
        eta_lin_coef: Optional[float] = None,
        eta_log_init: Optional[float] = None,
        eta_lin_init: Optional[float] = None,
        eta_learnable: bool = False,
        eta_mode: str = "log",
        eta_init: Optional[float] = None,
        eta_clip: float = 50.0,
        presymp_lnp: str = "end",
        # xi adaptation options
        # xi_adapt: bool = False,
        r_thresh: float = 1e-2,
        r_low: float = 1e-4,
        xi_mult_up: float = 2.0,
        xi_mult_down: float = 0.5,
        xi_min: float = 1e-4,
        xi_max: float = 100.0,
        theta_max: float = 1.0,
        # xi_adapt_warmup: int = 10,
        # xi_adapt_every: int = 1,
    ):
        super().__init__()
        if mlp_use_attn_vel and mlp_use_p_vel:
            raise ValueError("mlp_use_attn_vel and mlp_use_p_vel are mutually exclusive")
        self.mlp_use_attn_vel = bool(mlp_use_attn_vel)
        self.mlp_use_p_vel = bool(mlp_use_p_vel)
        self.no_mlp = bool(no_mlp)
        self.lookahead = bool(lookahead)
        attn_scheme = str(attn_scheme)
        if attn_scheme == "presymp":
            self.attn = PresymplecticSoftmaxAttention(
                cfg,
                h=h,
                xi=xi,
                # xi_adapt=xi_adapt,
                r_thresh=r_thresh,
                r_low=r_low,
                xi_mult_up=xi_mult_up,
                xi_mult_down=xi_mult_down,
                xi_min=xi_min,
                xi_max=xi_max,
                theta_max=theta_max,
                # xi_adapt_warmup=xi_adapt_warmup,
                # xi_adapt_every=xi_adapt_every,
                t0=t0,
                eta_mu=eta_mu,
                eta_log_coef=eta_log_coef,
                eta_lin_coef=eta_lin_coef,
                eta_log_init=eta_log_init,
                eta_lin_init=eta_lin_init,
                eta_learnable=eta_learnable,
                eta_mode=eta_mode,
                eta_init=eta_init,
                eta_clip=eta_clip,
                presymp_lnp=presymp_lnp,
                lookahead=lookahead,
            )
        elif attn_scheme == "euler":
            self.attn = DampedEulerAttention(
                cfg,
                h=h,
                t0=t0,
                eta_mu=eta_mu,
                eta_log_coef=eta_log_coef,
                eta_lin_coef=eta_lin_coef,
                eta_log_init=eta_log_init,
                eta_lin_init=eta_lin_init,
                eta_learnable=eta_learnable,
                eta_mode=eta_mode,
                eta_init=eta_init,
                eta_clip=eta_clip,
                presymp_lnp=presymp_lnp,
                lookahead=lookahead,
            )
        elif attn_scheme == "exp_euler":
            self.attn = DampedExpEulerAttention(
                cfg,
                h=h,
                t0=t0,
                eta_mu=eta_mu,
                eta_log_coef=eta_log_coef,
                eta_lin_coef=eta_lin_coef,
                eta_log_init=eta_log_init,
                eta_lin_init=eta_lin_init,
                eta_learnable=eta_learnable,
                eta_mode=eta_mode,
                eta_init=eta_init,
                eta_clip=eta_clip,
                presymp_lnp=presymp_lnp,
                lookahead=lookahead,
            )
        elif attn_scheme == "strang":
            self.attn = HalfDampStrangAttention(
                cfg,
                h=h,
                xi=xi,
                theta_max=theta_max,
                t0=t0,
                eta_mu=eta_mu,
                eta_log_coef=eta_log_coef,
                eta_lin_coef=eta_lin_coef,
                eta_log_init=eta_log_init,
                eta_lin_init=eta_lin_init,
                eta_learnable=eta_learnable,
                eta_mode=eta_mode,
                eta_init=eta_init,
                eta_clip=eta_clip,
                presymp_lnp=presymp_lnp,
                lookahead=lookahead,
            )
        elif attn_scheme == "plain_euler":
            self.attn = PlainEulerAttention(
                cfg,
                h=h,
                t0=t0,
                eta_mu=eta_mu,
                eta_log_coef=eta_log_coef,
                eta_lin_coef=eta_lin_coef,
                eta_log_init=eta_log_init,
                eta_lin_init=eta_lin_init,
                eta_learnable=eta_learnable,
                eta_mode=eta_mode,
                eta_init=eta_init,
                eta_clip=eta_clip,
                presymp_lnp=presymp_lnp,
                lookahead=lookahead,
            )
        else:
            raise ValueError("attn_scheme must be one of {'presymp','euler','exp_euler','strang','plain_euler'}")

        # MLP substep (accelerated)
        self.ln_x_mlp = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)
        self.ln_v_mlp = LayerNorm(cfg.n_embd, bias=cfg.bias)

        # learned scalars for the MLP Nesterov update
        self.mu_mlp = ConstrainedScalar(0.9, "unit")
        self.beta_mlp = ConstrainedScalar(0.9, "unit")
        self.gamma_mlp = ConstrainedScalar(1.0, "pos")

    def forward(
        self,
        x: torch.Tensor,
        p: torch.Tensor,
        v: torch.Tensor,
        k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # --- Attention step (updates x and its cotangent momentum p) ---
        x0 = x
        x, p = self.attn.step(x, p, k)

        # When --no_mlp is set, skip the MLP substep entirely.
        if self.no_mlp:
            return x, p, v

        # --- MLP step ---
        mu = self.mu_mlp()
        gamma = self.gamma_mlp()

        if self.mlp_use_attn_vel:
            # Variant A: use attention-induced velocity for MLP lookahead
            # v_attn ≈ (x_after_attn - x_before_attn)/h
            h = self.attn.h().detach().item()
            if h == 0.0:
                h = 1.0
            v_attn = (x - x0) / h
            x_in = x + mu * v_attn
            dx = self.mlp(self.ln_x_mlp(x_in))
            x = x + gamma * dx
            # expose the used velocity as v for logging/inspection
            v = v_attn
        elif self.mlp_use_p_vel:
            # Variant B (YuriiFormer-style Lie-Trotter): P from the attention step
            # is used directly as the MLP velocity.  The MLP updates P in-place via
            # a Nesterov step; the updated P is what the next layer's attention sees.
            # This eliminates the independent v stream entirely.
            beta = self.beta_mlp()
            x_in = x + mu * p
            dx = self.mlp(self.ln_x_mlp(x_in))
            p = beta * p + gamma * dx
            p = self.ln_v_mlp(p)
            x = x + p
            # v is unused in this mode; pass through unchanged
        else:
            # Default: Nesterov-accelerated MLP velocity stream (learned)
            beta = self.beta_mlp()
            x_in = x + mu * v
            dx = self.mlp(self.ln_x_mlp(x_in))
            v = beta * v + gamma * dx
            v = self.ln_v_mlp(v)
            x = x + v

        return x, p, v


class PresympMLPSubstep(nn.Module):
    # MLP substep used by presymp-family models.
    # Supports:
    #  - learned Nesterov velocity stream (mu,beta,gamma learned)
    #  - Variant A (mlp_use_attn_vel=True): use attention-induced velocity for lookahead

    def __init__(self, cfg: ModelConfig, mlp_use_attn_vel: bool = False):
        super().__init__()
        self.mlp_use_attn_vel = bool(mlp_use_attn_vel)
        self.ln_x_mlp = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)
        self.ln_v_mlp = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mu_mlp = ConstrainedScalar(0.9, "unit")
        self.beta_mlp = ConstrainedScalar(0.9, "unit")
        self.gamma_mlp = ConstrainedScalar(1.0, "pos")

    def forward(self, x: torch.Tensor, v: torch.Tensor, v_attn: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.mu_mlp()
        gamma = self.gamma_mlp()

        if self.mlp_use_attn_vel:
            if v_attn is None:
                raise ValueError('mlp_use_attn_vel=True requires v_attn')
            x_in = x + mu * v_attn
            dx = self.mlp(self.ln_x_mlp(x_in))
            x = x + gamma * dx
            v = v_attn
            return x, v

        beta = self.beta_mlp()
        x_in = x + mu * v
        dx = self.mlp(self.ln_x_mlp(x_in))
        v = beta * v + gamma * dx
        v = self.ln_v_mlp(v)
        x = x + v
        return x, v


class PresympModelAB2(nn.Module):
    # Presymp-family architecture with an Adams-Bashforth 2 (AB2) attention update.
    # One new attention RHS evaluation per layer; reuses the previous RHS.

    def __init__(
        self,
        cfg: ModelConfig,
        h: float = 1.0,
        t0: float = 1.0,
        eta_mu: Optional[float] = None,
        eta_log_coef: Optional[float] = None,
        eta_lin_coef: Optional[float] = None,
        eta_log_init: Optional[float] = None,
        eta_lin_init: Optional[float] = None,
        eta_learnable: bool = False,
        eta_mode: str = "log",
        eta_init: Optional[float] = None,
        eta_clip: float = 50.0,
        presymp_lnp: str = "end",
        use_v0_init: bool = True,
        mlp_use_attn_vel: bool = False,
        mlp_use_p_vel: bool = False,
        no_mlp: bool = False,
        lookahead: bool = False,
    ):
        super().__init__()
        if mlp_use_attn_vel and mlp_use_p_vel:
            raise ValueError("mlp_use_attn_vel and mlp_use_p_vel are mutually exclusive")
        self.cfg = cfg
        self.h = float(h)
        self.use_v0_init = bool(use_v0_init)
        self.mlp_use_attn_vel = bool(mlp_use_attn_vel)
        self.mlp_use_p_vel = bool(mlp_use_p_vel)
        self.no_mlp = bool(no_mlp)
        self.lookahead = bool(lookahead)

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.tok_v0_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_v0_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.tok_v0_emb_mlp = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_v0_emb_mlp = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

        self.attn = nn.ModuleList([
            DampedEulerAttention(
                cfg,
                h=h,
                t0=t0,
                eta_mu=eta_mu,
                eta_log_coef=eta_log_coef,
                eta_lin_coef=eta_lin_coef,
                eta_log_init=eta_log_init,
                eta_lin_init=eta_lin_init,
                eta_learnable=eta_learnable,
                eta_mode=eta_mode,
                eta_init=eta_init,
                eta_clip=eta_clip,
                presymp_lnp=presymp_lnp,
                lookahead=lookahead,
            )
            for _ in range(cfg.n_layer)
        ])
        self.mlp_steps = nn.ModuleList([
            PresympMLPSubstep(cfg, mlp_use_attn_vel=self.mlp_use_attn_vel)
            for _ in range(cfg.n_layer)
        ])

        self.ln_f = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self.last_rX_max = 0.0
        self.last_rP_max = 0.0
        self.last_xi_mean = 0.0
        self.last_h_mean = 0.0
        self.last_c_log_mean = float('nan')
        self.last_c_lin_mean = float('nan')

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, global_step: Optional[int] = None):
        B, T = idx.shape
        assert T <= self.cfg.block_size
        pos = torch.arange(0, T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)

        if self.use_v0_init:
            p = self.drop(self.tok_v0_emb(idx) + self.pos_v0_emb(pos)[None, :, :])
            if self.mlp_use_attn_vel or self.mlp_use_p_vel:
                v = torch.zeros_like(x)
            else:
                v = self.drop(self.tok_v0_emb_mlp(idx) + self.pos_v0_emb_mlp(pos)[None, :, :])
        else:
            p = torch.zeros_like(x)
            v = torch.zeros_like(x)

        dx_prev = None
        dp_prev = None

        for k in range(self.cfg.n_layer):
            hX_k = self.attn[k].hX()  # position step size (tensor)
            hY_k = self.attn[k].hY()  # momentum step size (tensor)
            dx_k, dp_k = self.attn[k].rhs(x, p, k)
            if k == 0 or dx_prev is None:
                dx_eff, dp_eff = dx_k, dp_k
            else:
                dx_eff = 1.5 * dx_k - 0.5 * dx_prev
                dp_eff = 1.5 * dp_k - 0.5 * dp_prev

            x_new = x + hX_k * dx_eff
            p_new = p + hY_k * dp_eff
            if self.attn[k].presymp_lnp != "none":
                p_new = self.attn[k]._apply_lnp(p_new)

            v_attn = (x_new - x) / hX_k.clamp(min=1e-8)
            x, p = x_new, p_new
            dx_prev, dp_prev = dx_k, dp_k

            if self.mlp_use_p_vel:
                # Variant B: P is the shared velocity; MLP updates it in-place.
                # The updated P flows to the next layer's attention step.
                if not self.no_mlp:
                    x, p = self.mlp_steps[k](x, p)
            else:
                if not self.no_mlp:
                    x, v = self.mlp_steps[k](x, v, v_attn=v_attn)

        h_vals = [self.attn[k].h().detach().cpu().item() for k in range(self.cfg.n_layer)]
        self.last_h_mean = sum(h_vals) / len(h_vals)
        # xi is not used in AB2 (DampedEulerAttention has no theta_xi_raw)
        self.last_xi_mean = float('nan')

        # Collect eta schedule coefficients (mean across layers)
        c_log_sum = 0.0; c_lin_sum = 0.0; c_cnt = 0
        for k in range(self.cfg.n_layer):
            cl, cm = _get_eta_coefs(self.attn[k].sched)
            c_log_sum += cl; c_lin_sum += cm; c_cnt += 1
        self.last_c_log_mean = (c_log_sum / c_cnt) if c_cnt > 0 else float('nan')
        self.last_c_lin_mean = (c_lin_sum / c_cnt) if c_cnt > 0 else float('nan')

        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


class PresympModelETDAB2(PresympModelAB2):
    # ETD-AB2 variant: AB2 on the integrating-factor momentum Pi=exp(eta) P.

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, global_step: Optional[int] = None):
        B, T = idx.shape
        assert T <= self.cfg.block_size
        pos = torch.arange(0, T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)

        if self.use_v0_init:
            p = self.drop(self.tok_v0_emb(idx) + self.pos_v0_emb(pos)[None, :, :])
            if self.mlp_use_attn_vel or self.mlp_use_p_vel:
                v = torch.zeros_like(x)
            else:
                v = self.drop(self.tok_v0_emb_mlp(idx) + self.pos_v0_emb_mlp(pos)[None, :, :])
        else:
            p = torch.zeros_like(x)
            v = torch.zeros_like(x)

        H_prev = None

        for k in range(self.cfg.n_layer):
            hX_k = self.attn[k].hX()           # position step size (tensor)
            hY_k = self.attn[k].hY()           # momentum step size (tensor)
            hX_k_f = hX_k.detach().item()      # float — time index and Lam interval
            Fv, Gv, z, _alpha = self.attn[k].FG_alpha(x, p, k)
            tk = self.attn[k].sched.t0 + float(k) * hX_k_f
            device, dtype = x.device, x.dtype
            Lam_k = self.attn[k].sched.exp_eta(tk, device, dtype)
            Lam_k1 = self.attn[k].sched.exp_eta(tk + hX_k_f, device, dtype)

            Pi = Lam_k * p
            H_k = Lam_k * Gv

            if k == 0 or H_prev is None:
                H_eff = H_k
            else:
                H_eff = 1.5 * H_k - 0.5 * H_prev

            Pi_new = Pi + hY_k * H_eff         # momentum update uses hY
            p_new = Pi_new / Lam_k1

            x_new = x + hX_k * (self.attn[k].c_B(p_new) / z.unsqueeze(-1))  # position uses hX
            if self.attn[k].presymp_lnp != "none":
                p_new = self.attn[k]._apply_lnp(p_new)

            v_attn = (x_new - x) / hX_k.clamp(min=1e-8)
            x, p = x_new, p_new
            H_prev = H_k

            if self.mlp_use_p_vel:
                # Variant B: P is the shared velocity; MLP updates it in-place.
                # The updated P flows to the next layer's attention step.
                if not self.no_mlp:
                    x, p = self.mlp_steps[k](x, p)
            else:
                if not self.no_mlp:
                    x, v = self.mlp_steps[k](x, v, v_attn=v_attn)

        h_vals = [self.attn[k].h().detach().cpu().item() for k in range(self.cfg.n_layer)]
        self.last_h_mean = sum(h_vals) / len(h_vals)
        self.last_xi_mean = float('nan')

        # Collect eta schedule coefficients (mean across layers)
        c_log_sum = 0.0; c_lin_sum = 0.0; c_cnt = 0
        for k in range(self.cfg.n_layer):
            cl, cm = _get_eta_coefs(self.attn[k].sched)
            c_log_sum += cl; c_lin_sum += cm; c_cnt += 1
        self.last_c_log_mean = (c_log_sum / c_cnt) if c_cnt > 0 else float('nan')
        self.last_c_lin_mean = (c_lin_sum / c_cnt) if c_cnt > 0 else float('nan')

        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss



# ─────────────────────────────────────────────────────────────────────────────
# Linear-attention helpers and architectures
# Particle system (Proposition prop:add, eq. acc_linear_matrix):
#   dot X = (1/N) X A X^T Y
#   dot Y = -alpha Y  - (1/N) Y Y^T X A  +  X V^T
# Causal masking: zero the strict upper triangle of each T×T Gram matrix.
# ─────────────────────────────────────────────────────────────────────────────

def _lin_FG(X: torch.Tensor, Y: torch.Tensor,
            A: torch.Tensor, V: torch.Tensor,
            causal: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute F(X,Y) and G(X,Y) for the linear-attention Hamiltonian system.

    X, Y : (B, T, d)
    A, V : (d, d)   — weight matrices (linear maps on token dimension)
    Returns (F, G) each of shape (B, T, d).
    """
    B, T, d = X.shape
    # Gram matrices  (B, T, T)
    XA = X @ A.T          # (B,T,d)  — X mapped by A^T
    GAM_XAX = XA @ X.transpose(-1, -2)   # (B,T,T):  (XA) X^T = X A X^T
    GAM_YY  = Y  @ Y.transpose(-1, -2)   # (B,T,T):  Y Y^T

    if causal:
        m = causal_mask(T, X.device)          # (T,T) upper-triangular mask
        GAM_XAX = GAM_XAX.masked_fill(m.unsqueeze(0), 0.0)
        GAM_YY  = GAM_YY .masked_fill(m.unsqueeze(0), 0.0)

    inv_T = 1.0 / float(T)
    F = inv_T * (GAM_XAX @ Y)              # (B,T,d)  : (X A X^T) Y / T
    G = -inv_T * (GAM_YY @ XA) + X @ V.T  # (B,T,d)  : -(Y Y^T X A)/T + X V^T
    return F, G


class LinearAttentionBaseline(nn.Module):
    """Plain (gradient-flow / first-order) linear-attention step.

    One layer:  X^{k+1} = X^k + h * F_lin(X^k, -)

    This is just the standard attention update with linear kernel kappa(x,y) = y^T A x,
    value matrix V, evaluated at the *position* stream only (no momentum).
    There is no momentum variable — this is the direct linear-attention analogue
    of the GPT baseline.

    Layer-norm is applied to X before computing QKV (pre-norm convention).
    """

    def __init__(self, cfg: ModelConfig, h: float = 1.0, causal: bool = True):
        super().__init__()
        self.causal = causal
        # Learned A and V (both d×d)
        self.c_A = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.c_V = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.ln = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.theta_h = nn.Parameter(torch.tensor(inv_softplus(h), dtype=torch.float32))

    def h(self, device=None, dtype=None):
        hv = F.softplus(self.theta_h)
        if device is not None or dtype is not None:
            hv = hv.to(device=device if device is not None else hv.device,
                       dtype=dtype if dtype is not None else hv.dtype)
        return hv

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, T, d = X.shape
        device, dtype = X.device, X.dtype
        h_t = self.h(device=device, dtype=dtype)
        Xn = self.ln(X)
        A = self.c_A.weight   # (d,d)
        V = self.c_V.weight   # (d,d)
        XA = Xn @ A.T
        GAM = XA @ Xn.transpose(-1, -2)  # (B,T,T): X A X^T
        if self.causal:
            m = causal_mask(T, X.device)
            GAM = GAM.masked_fill(m.unsqueeze(0), 0.0)
        # dx = (1/T) (X A X^T) V X  — standard linear-attention update
        VX = Xn @ V.T               # (B,T,d)
        dx = (1.0 / float(T)) * (GAM @ VX)
        dx = self.resid_drop(dx)
        return X + h_t * dx


class LinearAttentionEuler(nn.Module):
    """Plain (non-symplectic) Euler for the linear-attention Hamiltonian system.

    Eq. (plain_explicit_Euler) adapted for the linear oracle:
        X^{k+1} = X^k + h * F_lin(X^k, Y^k)
        Y^{k+1} = alpha * Y^k + h * G_lin(X^k, Y^k)

    alpha is a learned ConstrainedScalar in (0,1).
    """

    def __init__(self, cfg: ModelConfig, h: float = 1.0, alpha_init: float = 0.9,
                 causal: bool = True, presymp_lnp: str = "end"):
        super().__init__()
        self.causal = causal
        self.presymp_lnp = str(presymp_lnp)
        self.c_A = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.c_V = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.ln = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.ln_p = LayerNorm(cfg.n_embd, bias=cfg.bias) if presymp_lnp != "none" else nn.Identity()
        # Separate step sizes: hX for position  X^{k+1}=X+hX*F,  hY for momentum  Y^{k+1}=alpha*Y+hY*G
        self.theta_hX = nn.Parameter(torch.tensor(inv_softplus(h), dtype=torch.float32))
        self.theta_hY = nn.Parameter(torch.tensor(inv_softplus(h), dtype=torch.float32))
        self.theta_h = self.theta_hX   # backward-compat alias
        self.alpha_plain = ConstrainedScalar(alpha_init, "unit")
        # diagnostic
        self.last_h = float(F.softplus(self.theta_hX).detach().item())

    def hX(self, device=None, dtype=None):
        hv = F.softplus(self.theta_hX)
        if device is not None or dtype is not None:
            hv = hv.to(device=device if device is not None else hv.device,
                       dtype=dtype if dtype is not None else hv.dtype)
        return hv

    def hY(self, device=None, dtype=None):
        hv = F.softplus(self.theta_hY)
        if device is not None or dtype is not None:
            hv = hv.to(device=device if device is not None else hv.device,
                       dtype=dtype if dtype is not None else hv.dtype)
        return hv

    def h(self, device=None, dtype=None):
        """Backward-compat alias for hX()."""
        return self.hX(device=device, dtype=dtype)

    def step(self, Xk: torch.Tensor, Yk: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device, dtype = Xk.device, Xk.dtype
        hX_t = self.hX(device=device, dtype=dtype)
        hY_t = self.hY(device=device, dtype=dtype)
        Xn = self.ln(Xk)
        F_val, G_val = _lin_FG(Xn, Yk, self.c_A.weight, self.c_V.weight, self.causal)
        F_val = self.resid_drop(F_val)
        alpha_k = self.alpha_plain()
        Xk1 = Xk + hX_t * F_val
        Yk1 = alpha_k * Yk + hY_t * G_val
        if self.presymp_lnp != "none":
            Yk1 = self.ln_p(Yk1.float()).to(dtype=Yk1.dtype)
        self.last_h = hX_t.detach().item()
        return Xk1, Yk1


class LinearAttentionPresymp(nn.Module):
    """Conformally-symplectic (presymplectic) Euler for the linear-attention system.

    Eq. (presymp_Euler) adapted for the linear oracle:
        Y^{k+1} = sigma_k * ( Y^k + hY * G_lin(X^k, Y^k) )   [kick then damp]
        X^{k+1} = X^k + hX * F_lin(X^k, Y^{k+1})

    sigma_k = exp(-delta_eta) is the exact discrete damping factor.
    hX (position) and hY (momentum) are separately learned; both and the eta
    schedule coefficients are optimised end-to-end.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        h: float = 1.0,
        t0: float = 1.0,
        eta_mu: Optional[float] = None,
        eta_log_coef: Optional[float] = None,
        eta_lin_coef: Optional[float] = None,
        eta_log_init: Optional[float] = None,
        eta_lin_init: Optional[float] = None,
        eta_learnable: bool = False,
        eta_mode: str = "log",
        eta_init: Optional[float] = None,
        eta_clip: float = 50.0,
        causal: bool = True,
        presymp_lnp: str = "end",
    ):
        super().__init__()
        self.causal = causal
        self.presymp_lnp = str(presymp_lnp)
        self.c_A = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.c_V = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.ln = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.ln_p = LayerNorm(cfg.n_embd, bias=cfg.bias) if presymp_lnp != "none" else nn.Identity()
        # Separate step sizes for position and momentum.
        self.theta_hX = nn.Parameter(torch.tensor(inv_softplus(h), dtype=torch.float32))
        self.theta_hY = nn.Parameter(torch.tensor(inv_softplus(h), dtype=torch.float32))
        self.theta_h = self.theta_hX   # backward-compat alias
        self.sched = EtaSchedule(
            t0=t0, mu=eta_mu, log_coef=eta_log_coef, lin_coef=eta_lin_coef,
            learnable=eta_learnable, mode=eta_mode, init=eta_init,
            init_log=eta_log_init, init_lin=eta_lin_init, eta_clip=eta_clip,
        )
        # diagnostics
        self.last_h = float(F.softplus(self.theta_hX).detach().item())
        self.last_xi = 0.0

    def hX(self, device=None, dtype=None):
        hv = F.softplus(self.theta_hX)
        if device is not None or dtype is not None:
            hv = hv.to(device=device if device is not None else hv.device,
                       dtype=dtype if dtype is not None else hv.dtype)
        return hv

    def hY(self, device=None, dtype=None):
        hv = F.softplus(self.theta_hY)
        if device is not None or dtype is not None:
            hv = hv.to(device=device if device is not None else hv.device,
                       dtype=dtype if dtype is not None else hv.dtype)
        return hv

    def h(self, device=None, dtype=None):
        """Backward-compat alias for hX()."""
        return self.hX(device=device, dtype=dtype)

    def step(self, Xk: torch.Tensor, Yk: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        device, dtype = Xk.device, Xk.dtype
        hX_t = self.hX(device=device, dtype=dtype)
        hY_t = self.hY(device=device, dtype=dtype)
        hX_f = hX_t.detach().item()
        tk = self.sched.t0 + float(k) * hX_f

        Xn = self.ln(Xk)
        A = self.c_A.weight
        V = self.c_V.weight
        F_val, G_val = _lin_FG(Xn, Yk, A, V, self.causal)
        F_val = self.resid_drop(F_val)

        # Exact discrete damping over hY interval — use live hY_t so sigma
        # backprops into theta_hY (same fix as DampedExpEulerAttention).
        d_eta = self.sched.delta_eta_tensor(tk, hY_t, device, dtype)
        sigma = torch.exp(-d_eta)

        # Kick-then-damp: Y^{k+1} = sigma * (Y^k + hY * G(X^k, Y^k))
        Yk1 = sigma * (Yk + hY_t * G_val)
        # Symplectic position update uses updated momentum Y^{k+1}
        F_new = self.resid_drop(_lin_FG(Xn, Yk1, A, V, self.causal)[0])
        Xk1 = Xk + hX_t * F_new

        if self.presymp_lnp != "none":
            Yk1 = self.ln_p(Yk1.float()).to(dtype=Yk1.dtype)
        self.last_h = hX_f
        self.last_hY = hY_t.detach().item()
        return Xk1, Yk1


class LinearAttentionExpEuler(LinearAttentionPresymp):
    """Presymplectic exponential Euler for the linear-attention Hamiltonian system.

    Eq. (Presympl_exp_euler) adapted for the linear oracle:
        Y^{k+1} = sigma_k * Y^k + w_k * G_lin(X^k, Y^k)      [exact damping, old Y]
        X^{k+1} = X^k + hX * F_lin(X^k, Y^k)                 [position uses *old* Y]

    where sigma_k = exp(-delta_eta(t_k, hY)) and
    w_k = (1 - sigma_k) / alpha_eff  (= hY when alpha_eff ≈ 0).

    Unlike the presymplectic Euler (kick-then-damp), the position update here
    uses the *old* momentum Y^k rather than the updated Y^{k+1}.
    hX and hY are independently learned.
    """

    def step(self, Xk: torch.Tensor, Yk: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        device, dtype = Xk.device, Xk.dtype
        hX_t = self.hX(device=device, dtype=dtype)
        hY_t = self.hY(device=device, dtype=dtype)
        hX_f = hX_t.detach().item()
        tk = self.sched.t0 + float(k) * hX_f

        Xn = self.ln(Xk)
        A = self.c_A.weight
        V = self.c_V.weight
        F_val, G_val = _lin_FG(Xn, Yk, A, V, self.causal)
        F_val = self.resid_drop(F_val)

        # Exact discrete damping over hY interval — live hY_t so sigma
        # backprops into theta_hY (same fix as DampedExpEulerAttention).
        d_eta = self.sched.delta_eta_tensor(tk, hY_t, device, dtype)
        sigma = torch.exp(-d_eta)
        alpha_eff = d_eta / hY_t.clamp(min=1e-8)
        eps_t = torch.tensor(1e-8, device=device, dtype=dtype)
        w = torch.where(alpha_eff.abs() > eps_t, (1.0 - sigma) / alpha_eff, hY_t)

        # Exponential Euler: momentum damped exactly, position uses OLD Y
        Yk1 = sigma * Yk + w * G_val
        Xk1 = Xk + hX_t * F_val   # F_val was computed with Yk (old momentum)

        if self.presymp_lnp != "none":
            Yk1 = self.ln_p(Yk1.float()).to(dtype=Yk1.dtype)
        self.last_h = hX_f
        self.last_hY = hY_t.detach().item()
        return Xk1, Yk1


class LinAttnAB2Model(nn.Module):
    """Adams-Bashforth 2 (AB2) discretization of the linear-attention Hamiltonian system.

    Reuses one oracle per layer (the previous RHS), achieving order-2 accuracy at the
    same oracle cost as the first-order methods.

    Layer k update (AB2 on the full damped RHS):
        dX_k = F_lin(X^k, Y^k)
        dY_k = G_lin(X^k, Y^k) - alpha(t_k) * Y^k
        X^{k+1} = X^k + hX * (1.5 dX_k - 0.5 dX_{k-1})
        Y^{k+1} = Y^k + hY * (1.5 dY_k - 0.5 dY_{k-1})
    with plain Euler (AB1) for k=0.

    hX, hY, and the eta schedule are all learned per layer.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        h: float = 1.0,
        t0: float = 1.0,
        eta_mu: Optional[float] = None,
        eta_log_coef: Optional[float] = None,
        eta_lin_coef: Optional[float] = None,
        eta_log_init: Optional[float] = None,
        eta_lin_init: Optional[float] = None,
        eta_learnable: bool = False,
        eta_mode: str = "log",
        eta_init: Optional[float] = None,
        eta_clip: float = 50.0,
        presymp_lnp: str = "end",
        use_v0_init: bool = True,
        no_mlp: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.no_mlp = bool(no_mlp)
        self.use_v0_init = bool(use_v0_init)

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.tok_v0_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_v0_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

        self.attn = nn.ModuleList([
            LinearAttentionPresymp(
                cfg, h=h, t0=t0,
                eta_mu=eta_mu, eta_log_coef=eta_log_coef, eta_lin_coef=eta_lin_coef,
                eta_log_init=eta_log_init, eta_lin_init=eta_lin_init,
                eta_learnable=eta_learnable, eta_mode=eta_mode, eta_init=eta_init,
                eta_clip=eta_clip, presymp_lnp=presymp_lnp,
            )
            for _ in range(cfg.n_layer)
        ])
        if not self.no_mlp:
            self.mlp_steps = nn.ModuleList([PresympMLPSubstep(cfg) for _ in range(cfg.n_layer)])

        self.ln_f = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        # logging stubs
        self.last_rX_max = 0.0; self.last_rP_max = 0.0
        self.last_xi_mean = float("nan"); self.last_c_log_mean = float("nan")
        self.last_c_lin_mean = float("nan")
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    @property
    def last_h_mean(self):
        vals = [a.last_h for a in self.attn]
        return sum(vals) / len(vals) if vals else float("nan")

    def forward(self, idx, targets=None, global_step=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos)[None])
        if self.use_v0_init:
            y = self.drop(self.tok_v0_emb(idx) + self.pos_v0_emb(pos)[None])
        else:
            y = torch.zeros_like(x)
        v = torch.zeros_like(x)   # MLP velocity stream

        dX_prev = None; dY_prev = None
        for k in range(self.cfg.n_layer):
            a = self.attn[k]
            hX_k = a.hX(); hY_k = a.hY()
            hX_f = hX_k.detach().item()
            tk = a.sched.t0 + float(k) * hX_f
            device, dtype = x.device, x.dtype

            Xn = a.ln(x)
            F_k, G_k = _lin_FG(Xn, y, a.c_A.weight, a.c_V.weight, a.causal)
            F_k = a.resid_drop(F_k)
            alpha_k = a.sched.alpha(tk, device, dtype)
            dX_k = F_k
            dY_k = G_k - alpha_k * y

            if k == 0 or dX_prev is None:
                dX_eff, dY_eff = dX_k, dY_k
            else:
                dX_eff = 1.5 * dX_k - 0.5 * dX_prev
                dY_eff = 1.5 * dY_k - 0.5 * dY_prev

            x_new = x + hX_k * dX_eff
            y_new = y + hY_k * dY_eff
            if a.presymp_lnp != "none":
                y_new = a.ln_p(y_new.float()).to(dtype=y_new.dtype)

            v_attn = (x_new - x) / hX_k.clamp(min=1e-8)
            x, y = x_new, y_new
            dX_prev, dY_prev = dX_k, dY_k
            a.last_h = hX_f

            if not self.no_mlp:
                x, v = self.mlp_steps[k](x, v, v_attn=v_attn)

        # Update logged eta-schedule diagnostics
        c_log_sum = 0.0; c_lin_sum = 0.0; c_cnt = 0
        for a in self.attn:
            cl, cm = _get_eta_coefs(a.sched)
            c_log_sum += cl; c_lin_sum += cm; c_cnt += 1
        self.last_c_log_mean = (c_log_sum / c_cnt) if c_cnt > 0 else float('nan')
        self.last_c_lin_mean = (c_lin_sum / c_cnt) if c_cnt > 0 else float('nan')

        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


class LinAttnETDAB2Model(nn.Module):
    """ETD-AB2 discretization of the linear-attention Hamiltonian system.

    Uses the integrating-factor (IF) change of variables Pi = exp(eta(t)) * Y to
    remove the linear damping term, then applies AB2 to the remaining autonomous
    conservative force H_k = exp(eta(t_k)) * G_lin(X^k, Y^k).

    Layer k update:
        H_k   = exp(eta(t_k)) * G_lin(X^k, Y^k)
        H_eff = 1.5 H_k - 0.5 H_{k-1}          (AB2; plain H_k for k=0)
        Pi_k  = exp(eta(t_k)) * Y^k
        Pi_{k+1} = Pi_k + hY * H_eff
        Y^{k+1}  = Pi_{k+1} / exp(eta(t_{k+1}))
        X^{k+1}  = X^k + hX * F_lin(X^k, Y^{k+1})   [uses updated Y]

    hX, hY and the eta schedule are all learned per layer.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        h: float = 1.0,
        t0: float = 1.0,
        eta_mu: Optional[float] = None,
        eta_log_coef: Optional[float] = None,
        eta_lin_coef: Optional[float] = None,
        eta_log_init: Optional[float] = None,
        eta_lin_init: Optional[float] = None,
        eta_learnable: bool = False,
        eta_mode: str = "log",
        eta_init: Optional[float] = None,
        eta_clip: float = 50.0,
        presymp_lnp: str = "end",
        use_v0_init: bool = True,
        no_mlp: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.no_mlp = bool(no_mlp)
        self.use_v0_init = bool(use_v0_init)

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.tok_v0_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_v0_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

        self.attn = nn.ModuleList([
            LinearAttentionPresymp(
                cfg, h=h, t0=t0,
                eta_mu=eta_mu, eta_log_coef=eta_log_coef, eta_lin_coef=eta_lin_coef,
                eta_log_init=eta_log_init, eta_lin_init=eta_lin_init,
                eta_learnable=eta_learnable, eta_mode=eta_mode, eta_init=eta_init,
                eta_clip=eta_clip, presymp_lnp=presymp_lnp,
            )
            for _ in range(cfg.n_layer)
        ])
        if not self.no_mlp:
            self.mlp_steps = nn.ModuleList([PresympMLPSubstep(cfg) for _ in range(cfg.n_layer)])

        self.ln_f = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        # logging stubs
        self.last_rX_max = 0.0; self.last_rP_max = 0.0
        self.last_xi_mean = float("nan"); self.last_c_log_mean = float("nan")
        self.last_c_lin_mean = float("nan")
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    @property
    def last_h_mean(self):
        vals = [a.last_h for a in self.attn]
        return sum(vals) / len(vals) if vals else float("nan")

    def forward(self, idx, targets=None, global_step=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos)[None])
        if self.use_v0_init:
            y = self.drop(self.tok_v0_emb(idx) + self.pos_v0_emb(pos)[None])
        else:
            y = torch.zeros_like(x)
        v = torch.zeros_like(x)   # MLP velocity stream

        H_prev = None
        for k in range(self.cfg.n_layer):
            a = self.attn[k]
            hX_k = a.hX(); hY_k = a.hY()
            hX_f = hX_k.detach().item()
            tk = a.sched.t0 + float(k) * hX_f
            device, dtype = x.device, x.dtype

            Lam_k  = a.sched.exp_eta(tk,           device, dtype)
            Lam_k1 = a.sched.exp_eta(tk + hX_f,    device, dtype)

            Xn = a.ln(x)
            A = a.c_A.weight; V_mat = a.c_V.weight
            _F_old, G_k = _lin_FG(Xn, y, A, V_mat, a.causal)

            H_k = Lam_k * G_k
            H_eff = H_k if (k == 0 or H_prev is None) else (1.5 * H_k - 0.5 * H_prev)

            Pi_new = Lam_k * y + hY_k * H_eff      # integrating-factor momentum update
            y_new  = Pi_new / Lam_k1

            # Position update uses updated Y^{k+1} (same as presymp Euler pattern)
            F_new, _ = _lin_FG(Xn, y_new, A, V_mat, a.causal)
            F_new = a.resid_drop(F_new)
            x_new = x + hX_k * F_new

            if a.presymp_lnp != "none":
                y_new = a.ln_p(y_new.float()).to(dtype=y_new.dtype)

            v_attn = (x_new - x) / hX_k.clamp(min=1e-8)
            x, y = x_new, y_new
            H_prev = H_k
            a.last_h = hX_f

            if not self.no_mlp:
                x, v = self.mlp_steps[k](x, v, v_attn=v_attn)

        # Update logged eta-schedule diagnostics
        c_log_sum = 0.0; c_lin_sum = 0.0; c_cnt = 0
        for a in self.attn:
            cl, cm = _get_eta_coefs(a.sched)
            c_log_sum += cl; c_lin_sum += cm; c_cnt += 1
        self.last_c_log_mean = (c_log_sum / c_cnt) if c_cnt > 0 else float('nan')
        self.last_c_lin_mean = (c_lin_sum / c_cnt) if c_cnt > 0 else float('nan')

        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


class GPTModel(nn.Module):
    def __init__(self, cfg: ModelConfig, no_mlp: bool = False):
        super().__init__()
        self.cfg = cfg
        self.no_mlp = bool(no_mlp)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([GPTBlock(cfg, no_mlp=self.no_mlp) for _ in range(cfg.n_layer)])
        self.ln_f = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self.last_rX_max = 0.0
        self.last_rP_max = 0.0
        self.last_xi_mean = 0.0
        self.last_h_mean = 0.0

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, global_step: Optional[int] = None):
        B, T = idx.shape
        assert T <= self.cfg.block_size
        pos = torch.arange(0, T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


class YuriiFormerModel(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        use_v0_init: bool = True,
        noise_eta: float = 0.0,
        noise_gamma: float = 0.55,
        noise_loc: str = "v",
        restart_mode: str = "none",
        restart_min_layer: int = 1,
        no_mlp: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.noise_eta = float(noise_eta)
        self.noise_gamma = float(noise_gamma)
        self.noise_loc = str(noise_loc)
        self.restart_mode = str(restart_mode)
        self.restart_min_layer = int(restart_min_layer)
        self.no_mlp = bool(no_mlp)

        if self.noise_loc not in ("dx", "v", "xin"):
            raise ValueError("noise_loc must be one of {'dx','v','xin'}")
        if self.restart_mode not in ("none", "speed", "loss"):
            raise ValueError("restart_mode must be one of {'none','speed','loss'}")
        # Main token/position embeddings (as in baseline)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)

        # v0 initialization embeddings for momentum variants (Appendix A.1)
        # "we initialize v0 using token and positional embedding tables separate from
        #  the main token and positional embeddings." fileciteturn23file11
        self.use_v0_init = bool(use_v0_init)
        self.tok_v0_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_v0_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([YuriiFormerLieTrotterBlock(cfg, no_mlp=self.no_mlp) for _ in range(cfg.n_layer)])
        self.ln_f = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        # for logging/debug
        self.last_restart_count = 0

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, global_step: Optional[int] = None):
        B, T = idx.shape
        assert T <= self.cfg.block_size
        pos = torch.arange(0, T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)

        # annealed gradient-noise schedule (Neelakantan et al.): sigma_t^2 = eta / (1+t)^gamma
        # We use global_step as t (training step). If not provided, fall back to 0.
        t_step = int(global_step) if global_step is not None else 0
        noise_std = 0.0
        if self.training and self.noise_eta > 0.0:
            noise_var = self.noise_eta / ((1.0 + float(t_step)) ** self.noise_gamma)
            noise_std = math.sqrt(max(noise_var, 0.0))

        if self.use_v0_init:
            # v0 token+pos embeddings (separate tables)
            v = self.tok_v0_emb(idx) + self.pos_v0_emb(pos)[None, :, :]
            v = self.drop(v)
        else:
            v = torch.zeros_like(x)
        x_prev = None
        restarts = 0

        def _lm_loss(h: torch.Tensor) -> torch.Tensor:
            assert targets is not None
            hh = self.ln_f(h)
            logits = self.lm_head(hh)
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        for layer_idx, blk in enumerate(self.blocks):
            x_cand, v_cand = blk(x, v, noise_std=noise_std, noise_loc=self.noise_loc)

            do_restart = None
            if self.restart_mode == "speed" and layer_idx >= self.restart_min_layer and x_prev is not None:
                # speed restart: restart at first time ||x_{k+1}-x_k|| < ||x_k-x_{k-1}||
                dcur = x_cand - x
                dprev = x - x_prev
                ncur = torch.linalg.vector_norm(dcur.reshape(B, -1), ord=2, dim=1)
                nprev = torch.linalg.vector_norm(dprev.reshape(B, -1), ord=2, dim=1)
                do_restart = ncur < nprev

            elif self.restart_mode == "loss" and layer_idx >= self.restart_min_layer and targets is not None:
                # function-value restart: restart when f(x_{k+1}) > f(x_k)
                with torch.no_grad():
                    f_x = _lm_loss(x)
                    f_xcand = _lm_loss(x_cand)
                do_restart = torch.tensor([bool(f_xcand > f_x)] * B, device=x.device)

            if do_restart is not None and bool(do_restart.any()):
                mask = do_restart
                x0, v0 = blk(x[mask], torch.zeros_like(v[mask]), noise_std=noise_std, noise_loc=self.noise_loc)
                x_cand = x_cand.clone()
                v_cand = v_cand.clone()
                x_cand[mask] = x0
                v_cand[mask] = v0
                restarts += int(mask.sum().item())

            x_prev = x
            x, v = x_cand, v_cand

        self.last_restart_count = restarts

        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


class PresympModel(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        attn_scheme: str = "presymp",
        h: float = 1.0,
        xi: float = 1.0,
        t0: float = 1.0,
        eta_mu: Optional[float] = None,
        eta_log_coef: Optional[float] = None,
        eta_lin_coef: Optional[float] = None,
        eta_log_init: Optional[float] = None,
        eta_lin_init: Optional[float] = None,
        eta_learnable: bool = False,
        eta_mode: str = "log",
        eta_init: Optional[float] = None,
        eta_clip: float = 50.0,
        presymp_lnp: str = "end",
        use_v0_init: bool = True,
        # xi adaptation options
        # xi_adapt: bool = False,
        r_thresh: float = 1e-2,
        r_low: float = 1e-4,
        xi_mult_up: float = 2.0,
        xi_mult_down: float = 0.5,
        xi_min: float = 1e-4,
        xi_max: float = 100.0,
        theta_max: float = 1.0,
        # xi_adapt_warmup: int = 10,
        # xi_adapt_every: int = 1,
        mlp_use_attn_vel: bool = False,
        mlp_use_p_vel: bool = False,
        no_mlp: bool = False,
        lookahead: bool = False,
    ):
        super().__init__()
        if mlp_use_attn_vel and mlp_use_p_vel:
            raise ValueError("mlp_use_attn_vel and mlp_use_p_vel are mutually exclusive")
        self.cfg = cfg
        self.mlp_use_attn_vel = bool(mlp_use_attn_vel)
        self.mlp_use_p_vel = bool(mlp_use_p_vel)
        self.no_mlp = bool(no_mlp)
        self.lookahead = bool(lookahead)
        self.use_v0_init = bool(use_v0_init)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)

        # Momentum init embeddings (same idea as YuriiFormer v0): separate token+pos tables
        # We reuse the same naming (tok_v0_emb/pos_v0_emb) so the optimizer grouping
        # treats them as embeddings with wd=0.1.
        self.tok_v0_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_v0_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        # Separate velocity-init embeddings for the MLP velocity stream (v)
        self.tok_v0_emb_mlp = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_v0_emb_mlp = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([
            PresympGPTBlock(
                cfg,
                mlp_use_attn_vel=self.mlp_use_attn_vel,
                mlp_use_p_vel=self.mlp_use_p_vel,
                no_mlp=self.no_mlp,
                attn_scheme=attn_scheme,
                h=h,
                xi=xi,
                t0=t0,
                eta_mu=eta_mu,
                eta_log_coef=eta_log_coef,
                eta_lin_coef=eta_lin_coef,
                eta_log_init=eta_log_init,
                eta_lin_init=eta_lin_init,
                eta_learnable=eta_learnable,
            eta_mode=eta_mode,
            eta_init=eta_init,
            eta_clip=eta_clip,
                presymp_lnp=presymp_lnp,
                lookahead=lookahead,
                # xi_adapt=xi_adapt,
                r_thresh=r_thresh,
                r_low=r_low,
                xi_mult_up=xi_mult_up,
                xi_mult_down=xi_mult_down,
                xi_min=xi_min,
                xi_max=xi_max,
            theta_max=theta_max,
            # xi_adapt_warmup=xi_adapt_warmup,
                # xi_adapt_every=xi_adapt_every,
            )
            for _ in range(cfg.n_layer)
        ])
        self.ln_f = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self.last_rX_max = 0.0
        self.last_rP_max = 0.0
        self.last_xi_mean = float('nan')
        self.last_h_mean = float('nan')
        self.last_c_log_mean = float('nan')
        self.last_c_lin_mean = float('nan')

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, global_step: Optional[int] = None):
        B, T = idx.shape
        assert T <= self.cfg.block_size
        pos = torch.arange(0, T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)

        if self.use_v0_init:
            init_p = self.tok_v0_emb(idx) + self.pos_v0_emb(pos)[None, :, :]
            init_p = self.drop(init_p)
            p = init_p
            if self.mlp_use_attn_vel or self.mlp_use_p_vel:
                # Variant A and B both discard the separate MLP velocity stream.
                v = torch.zeros_like(x)
            else:
                init_v = self.tok_v0_emb_mlp(idx) + self.pos_v0_emb_mlp(pos)[None, :, :]
                init_v = self.drop(init_v)
                v = init_v
        else:
            p = torch.zeros_like(x)
            v = torch.zeros_like(x)
        rX_max = 0.0
        rP_max = 0.0
        xi_sum = 0.0
        xi_cnt = 0
        h_sum = 0.0
        hY_sum = 0.0
        h_cnt = 0
        for k, blk in enumerate(self.blocks):
            x, p, v = blk(x, p, v, k)
            attn = getattr(blk, 'attn', None)
            if attn is not None and hasattr(attn, 'last_rX'):
                rX_max = max(rX_max, float(attn.last_rX))
                rP_max = max(rP_max, float(attn.last_rP))
                xi_sum += float(attn.last_xi)
                xi_cnt += 1
            if attn is not None and hasattr(attn, 'last_h'):
                h_sum += float(attn.last_h)
                h_cnt += 1
            if attn is not None and hasattr(attn, 'last_hY'):
                hY_sum += float(attn.last_hY)

        self.last_rX_max = rX_max
        self.last_rP_max = rP_max
        self.last_xi_mean = (xi_sum / xi_cnt) if xi_cnt > 0 else float('nan')
        self.last_h_mean = (h_sum / h_cnt) if h_cnt > 0 else float('nan')
        self.last_hY_mean = (hY_sum / h_cnt) if h_cnt > 0 else float('nan')

        # Collect eta schedule coefficients (mean across layers that have a sched)
        c_log_sum = 0.0; c_lin_sum = 0.0; c_cnt = 0
        for blk in self.blocks:
            attn = getattr(blk, 'attn', None)
            if attn is not None and hasattr(attn, 'sched'):
                cl, cm = _get_eta_coefs(attn.sched)
                c_log_sum += cl; c_lin_sum += cm; c_cnt += 1
        self.last_c_log_mean = (c_log_sum / c_cnt) if c_cnt > 0 else float('nan')
        self.last_c_lin_mean = (c_lin_sum / c_cnt) if c_cnt > 0 else float('nan')

        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


class LinAttnModel(nn.Module):
    """Baseline model with linear self-attention instead of softmax.

    Each layer:  X^{k+1} = LN(X) -> linear_attn(X) -> X + h*dx
    No momentum; direct analogue of GPTModel.
    """

    def __init__(self, cfg: ModelConfig, h: float = 1.0, no_mlp: bool = False):
        super().__init__()
        self.cfg = cfg
        self.no_mlp = bool(no_mlp)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.attn_layers = nn.ModuleList([
            LinearAttentionBaseline(cfg, h=h) for _ in range(cfg.n_layer)
        ])
        if not self.no_mlp:
            self.mlp_lns = nn.ModuleList([LayerNorm(cfg.n_embd, bias=cfg.bias) for _ in range(cfg.n_layer)])
            self.mlps    = nn.ModuleList([MLP(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        # logging stubs
        self.last_rX_max = 0.0; self.last_rP_max = 0.0
        self.last_xi_mean = float("nan"); self.last_h_mean = float("nan")
        self.last_c_log_mean = float("nan"); self.last_c_lin_mean = float("nan")
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None, global_step=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos)[None])
        for k in range(self.cfg.n_layer):
            x = self.attn_layers[k](x)
            if not self.no_mlp:
                x = x + self.mlps[k](self.mlp_lns[k](x))
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


class LinAttnYuriiModel(nn.Module):
    """YuriiFormer-style model with linear self-attention instead of softmax.

    Each layer uses the Nesterov-Lie-Trotter block structure from YuriiFormerLieTrotterBlock,
    but the oracle is linear attention: dx = (1/T)(X A X^T)(V X) / T
    i.e. the standard linear-attention update dx evaluated at the lookahead point x + mu*v.
    """

    def __init__(self, cfg: ModelConfig, h: float = 1.0, no_mlp: bool = False,
                 use_v0_init: bool = True):
        super().__init__()
        self.cfg = cfg
        self.no_mlp = bool(no_mlp)
        self.use_v0_init = bool(use_v0_init)
        self.tok_emb  = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb  = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.tok_v0_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_v0_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        # Per-layer linear attention + MLP + learned scalars
        self.attn_layers = nn.ModuleList([LinearAttentionBaseline(cfg, h=h) for _ in range(cfg.n_layer)])
        self.mlp_lns = nn.ModuleList([LayerNorm(cfg.n_embd, bias=cfg.bias) for _ in range(cfg.n_layer)])
        self.mlps    = nn.ModuleList([MLP(cfg) for _ in range(cfg.n_layer)])
        self.ln_v    = nn.ModuleList([LayerNorm(cfg.n_embd, bias=cfg.bias) for _ in range(cfg.n_layer)])
        # Learned scalars per layer (mu, beta, gamma for attn; mu2, beta2, gamma2 for MLP)
        self.mu    = nn.ModuleList([ConstrainedScalar(0.9, "unit") for _ in range(cfg.n_layer)])
        self.beta  = nn.ModuleList([ConstrainedScalar(0.9, "unit") for _ in range(cfg.n_layer)])
        self.gamma = nn.ModuleList([ConstrainedScalar(1.0, "pos")  for _ in range(cfg.n_layer)])
        self.mu2    = nn.ModuleList([ConstrainedScalar(0.9, "unit") for _ in range(cfg.n_layer)])
        self.beta2  = nn.ModuleList([ConstrainedScalar(0.9, "unit") for _ in range(cfg.n_layer)])
        self.gamma2 = nn.ModuleList([ConstrainedScalar(1.0, "pos")  for _ in range(cfg.n_layer)])
        self.ln_f = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        # logging stubs
        self.last_rX_max = 0.0; self.last_rP_max = 0.0
        self.last_xi_mean = float("nan"); self.last_h_mean = float("nan")
        self.last_c_log_mean = float("nan"); self.last_c_lin_mean = float("nan")
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None, global_step=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos)[None])
        if self.use_v0_init:
            v = self.drop(self.tok_v0_emb(idx) + self.pos_v0_emb(pos)[None])
        else:
            v = torch.zeros_like(x)

        for k in range(self.cfg.n_layer):
            mu    = self.mu[k]()
            beta  = self.beta[k]()
            gamma = self.gamma[k]()
            # Attention substep with Nesterov lookahead
            x_in = x + mu * v
            # Reuse LinearAttentionBaseline.forward but on the lookahead point
            attn_lyr = self.attn_layers[k]
            dx = attn_lyr(x_in) - x_in   # delta from the layer (layer returns x_in + h*dx)
            v = self.ln_v[k](beta * v + gamma * dx)
            x = x + v

            if not self.no_mlp:
                mu2    = self.mu2[k]()
                beta2  = self.beta2[k]()
                gamma2 = self.gamma2[k]()
                x_in2 = x + mu2 * v
                dx2 = self.mlps[k](self.mlp_lns[k](x_in2))
                v = self.ln_v[k](beta2 * v + gamma2 * dx2)
                x = x + v

        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


class LinAttnEulerModel(nn.Module):
    """Plain (non-symplectic) Euler model with linear attention Hamiltonian system."""

    def __init__(
        self,
        cfg: ModelConfig,
        h: float = 1.0,
        alpha_init: float = 0.9,
        presymp_lnp: str = "end",
        use_v0_init: bool = True,
        no_mlp: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.no_mlp = bool(no_mlp)
        self.use_v0_init = bool(use_v0_init)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.tok_v0_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_v0_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.attn = nn.ModuleList([
            LinearAttentionEuler(cfg, h=h, alpha_init=alpha_init, presymp_lnp=presymp_lnp)
            for _ in range(cfg.n_layer)
        ])
        if not self.no_mlp:
            self.mlp_steps = nn.ModuleList([
                PresympMLPSubstep(cfg) for _ in range(cfg.n_layer)
            ])
        self.ln_f = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        # logging stubs
        self.last_rX_max = 0.0; self.last_rP_max = 0.0
        self.last_xi_mean = float("nan"); self.last_c_log_mean = float("nan"); self.last_c_lin_mean = float("nan")
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    @property
    def last_h_mean(self):
        vals = [a.last_h for a in self.attn]
        return sum(vals) / len(vals) if vals else float("nan")

    def forward(self, idx, targets=None, global_step=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos)[None])
        if self.use_v0_init:
            y = self.drop(self.tok_v0_emb(idx) + self.pos_v0_emb(pos)[None])
        else:
            y = torch.zeros_like(x)
        v = torch.zeros_like(x)

        for k in range(self.cfg.n_layer):
            x_new, y_new = self.attn[k].step(x, y)
            if not self.no_mlp:
                v_attn = (x_new - x) / self.attn[k].h().clamp(min=1e-8)
                x_new, v = self.mlp_steps[k](x_new, v, v_attn=v_attn)
            x, y = x_new, y_new

        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


class LinAttnPresympModel(nn.Module):
    """Conformally-symplectic (presymplectic) Euler model with linear attention.

    attn_cls controls the layer integrator:
        "presymp"   (default) — kick-then-damp Euler, position uses updated Y^{k+1}
        "exp_euler"           — exponential Euler, position uses old Y^k
    """

    def __init__(
        self,
        cfg: ModelConfig,
        h: float = 1.0,
        t0: float = 1.0,
        eta_mu: Optional[float] = None,
        eta_log_coef: Optional[float] = None,
        eta_lin_coef: Optional[float] = None,
        eta_log_init: Optional[float] = None,
        eta_lin_init: Optional[float] = None,
        eta_learnable: bool = False,
        eta_mode: str = "log",
        eta_init: Optional[float] = None,
        eta_clip: float = 50.0,
        presymp_lnp: str = "end",
        use_v0_init: bool = True,
        no_mlp: bool = False,
        attn_cls: str = "presymp",
    ):
        super().__init__()
        self.cfg = cfg
        self.no_mlp = bool(no_mlp)
        self.use_v0_init = bool(use_v0_init)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.tok_v0_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_v0_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

        _attn_cls_map = {"presymp": LinearAttentionPresymp, "exp_euler": LinearAttentionExpEuler}
        if attn_cls not in _attn_cls_map:
            raise ValueError(f"attn_cls must be one of {list(_attn_cls_map)}, got {attn_cls!r}")
        _AttnCls = _attn_cls_map[attn_cls]

        self.attn = nn.ModuleList([
            _AttnCls(
                cfg, h=h, t0=t0,
                eta_mu=eta_mu, eta_log_coef=eta_log_coef, eta_lin_coef=eta_lin_coef,
                eta_log_init=eta_log_init, eta_lin_init=eta_lin_init,
                eta_learnable=eta_learnable, eta_mode=eta_mode, eta_init=eta_init,
                eta_clip=eta_clip, presymp_lnp=presymp_lnp,
            )
            for _ in range(cfg.n_layer)
        ])
        if not self.no_mlp:
            self.mlp_steps = nn.ModuleList([
                PresympMLPSubstep(cfg) for _ in range(cfg.n_layer)
            ])
        self.ln_f = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        # logging stubs
        self.last_rX_max = 0.0; self.last_rP_max = 0.0
        self.last_xi_mean = float("nan"); self.last_c_log_mean = float("nan"); self.last_c_lin_mean = float("nan")
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    @property
    def last_h_mean(self):
        vals = [a.last_h for a in self.attn]
        return sum(vals) / len(vals) if vals else float("nan")

    def forward(self, idx, targets=None, global_step=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos)[None])
        if self.use_v0_init:
            y = self.drop(self.tok_v0_emb(idx) + self.pos_v0_emb(pos)[None])
        else:
            y = torch.zeros_like(x)
        v = torch.zeros_like(x)

        for k in range(self.cfg.n_layer):
            x_new, y_new = self.attn[k].step(x, y, k)
            if not self.no_mlp:
                v_attn = (x_new - x) / self.attn[k].h().clamp(min=1e-8)
                x_new, v = self.mlp_steps[k](x_new, v, v_attn=v_attn)
            x, y = x_new, y_new

        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
