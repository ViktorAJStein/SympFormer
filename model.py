
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
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln_1 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
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
    def __init__(self, cfg: ModelConfig):
        super().__init__()
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
    ):
        super().__init__()
        self.cfg = cfg

        # Learned step size and coupling (positive), with smooth theta_max cap for xi:
        #   h = softplus(theta_h) > 0
        #   xi = (theta_max/(2h)) * sigmoid(theta_xi_raw)  in (0, theta_max/(2h)) if theta_max>0
        self.theta_h = nn.Parameter(torch.tensor(inv_softplus(h), dtype=torch.float32))
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
            m = causal_mask(T, q.device)
            S = S.masked_fill(m.unsqueeze(0).unsqueeze(0), float("-inf"))

        S = S.mean(dim=1)  # (B, T, T)

        # clamp for numerical safety
        S = torch.clamp(S, min=-60.0, max=60.0)
        E = torch.exp(S)
        if self.causal:
            m = causal_mask(T, q.device)
            E = E.masked_fill(m.unsqueeze(0), 0.0)

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


    def _vel(self, t: float, X: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
        device, dtype = X.device, X.dtype
        lam = self.sched.exp_minus_eta(t, device, dtype)
        p = lam * Pi

        x_ln = self.ln(X)
        q, k, _ = self._qkv(x_ln)
        _, z = self._kernel_E_z(q, k)
        return p / z.unsqueeze(-1)

    def _force(self, t: float, X: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
        device, dtype = X.device, X.dtype
        Lam = self.sched.exp_eta(t, device, dtype)
        lam = self.sched.exp_minus_eta(t, device, dtype)
        p = lam * Pi  # physical momentum

        x_ln = self.ln(X)
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

    def step(self, Xk: torch.Tensor, Pk: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        device, dtype = Xk.device, Xk.dtype
        h = float(self.h(device=device, dtype=dtype).detach().cpu().item())
        tau = 0.5 * h
        t0 = self.sched.t0
        tk = t0 + float(k) * h
        tk1 = tk + h

        Lam_tk = self.sched.exp_eta(tk, device, dtype)
        Pi = Lam_tk * Pk

        t = tk
        X = Xk
        bar_t = tk
        bar_X = Xk
        bar_Pi = Pi

        # coupling (possibly learnable): enforce theta_max via xi(h) reparameterization
        xi_eff = float(self.xi(h=torch.tensor(h, device=device, dtype=dtype), device=device, dtype=dtype).detach().cpu().item())
        theta = 2.0 * xi_eff * h
        c = math.cos(theta)
        s = math.sin(theta)

        # phi_A^{tau}
        Pi = Pi + tau * self._force(t, X, bar_Pi)
        if self.presymp_lnp == "each_substep":
            Pi = self._apply_lnp(Pi)
        if self.presymp_lnp == "each_substep":
            Pi = self._apply_lnp(Pi)
        bar_t = bar_t + tau
        bar_X = bar_X + tau * self._vel(t, X, bar_Pi)

        # phi_B^{tau}
        t = t + tau
        X = X + tau * self._vel(bar_t, bar_X, Pi)
        bar_Pi = bar_Pi + tau * self._force(bar_t, bar_X, Pi)
        if self.presymp_lnp == "each_substep":
            bar_Pi = self._apply_lnp(bar_Pi)
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

        # phi_B^{tau}
        t = t + tau
        X = X + tau * self._vel(bar_t, bar_X, Pi)
        bar_Pi = bar_Pi + tau * self._force(bar_t, bar_X, Pi)

        # phi_A^{tau}
        Pi = Pi + tau * self._force(t, X, bar_Pi)
        bar_t = bar_t + tau
        bar_X = bar_X + tau * self._vel(t, X, bar_Pi)

        lam_tk1 = self.sched.exp_minus_eta(tk1, device, dtype)
        Pk1_raw = lam_tk1 * Pi
        bar_Pk1_raw = lam_tk1 * bar_Pi
        if self.presymp_lnp in ("end", "each_substep"):
            Pk1 = self._apply_lnp(Pk1_raw)
            bar_Pk1 = self._apply_lnp(bar_Pk1_raw)
        else:
            Pk1, bar_Pk1 = Pk1_raw, bar_Pk1_raw

        self.last_h = h
        self.last_xi = float(self.xi(h=torch.tensor(h, dtype=torch.float32)).detach().cpu().item())
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
    ):
        super().__init__()
        self.cfg = cfg
        self.theta_h = nn.Parameter(torch.tensor(inv_softplus(h), dtype=torch.float32))
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
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        # diagnostics for uniform logging
        self.last_rX = 0.0
        self.last_rP = 0.0
        self.last_h = float(F.softplus(self.theta_h).detach().cpu().item())
        self.last_xi = 0.0

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


    def _kernel_E_z(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, nh, T, _ = q.shape
        S = (q @ k.transpose(-2, -1)) * self.scale  # (B, nh, T, T)
        if self.causal:
            m = causal_mask(T, q.device)
            S = S.masked_fill(m.unsqueeze(0).unsqueeze(0), float("-inf"))
        S = S.mean(dim=1)  # (B,T,T)
        S = torch.clamp(S, min=-60.0, max=60.0)
        E = torch.exp(S)
        if self.causal:
            m = causal_mask(T, q.device)
            E = E.masked_fill(m.unsqueeze(0), 0.0)
        z = E.sum(dim=-1) / float(T)
        z = z.clamp_min(self.eps)
        return E, z

    def _apply_lnp(self, P: torch.Tensor) -> torch.Tensor:
        if self.presymp_lnp == "none":
            return P
        out = self.ln_p(P.float())
        return out.to(dtype=P.dtype)
    def _F_E_z_vmerge(self, X: torch.Tensor, P: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Return (F, E, z, v_merge) at (X,P), where F=dot X, E=exp(score), z is row normalizer.
        # This computes QKV and the kernel once.
        x_ln = self.ln(X)
        q, k, v = self._qkv(x_ln)
        E, z = self._kernel_E_z(q, k)
        B, T, C = X.shape
        v_merge = v.transpose(1, 2).contiguous().view(B, T, C)
        Fv = P / z.unsqueeze(-1)
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
        # note: we use a scalar h for the schedule evaluation here
        h = self.h().item()
        tk = self.sched.t0 + float(k) * h

        Fv, Gv, z, alpha = self.FG_alpha(Xk, Pk, k)
        Xk1 = Xk + h * Fv
        Pk1 = Pk + h * (Gv - alpha * Pk)
        if self.presymp_lnp != "none":
            Pk1 = self._apply_lnp(Pk1)
        self.last_h = h
        return Xk1, Pk1




class DampedExpEulerAttention(DampedEulerAttention):
    # Integrating-factor / exponential Euler discretization of the damped system.
    # Treat damping exactly over one step using sigma = exp(-(eta(t+h)-eta(t))).

    def step(self, Xk: torch.Tensor, Pk: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        device, dtype = Xk.device, Xk.dtype
        h = self.h().item()
        tk = self.sched.t0 + float(k) * h

        # one oracle call at (Xk,Pk)
        _Fv, Gv, z, _alpha = self.FG_alpha(Xk, Pk, k)

        d_eta = self.sched.delta_eta(tk, h, device, dtype)
        sigma = torch.exp(-d_eta)
        denom = torch.tensor(h if h != 0.0 else 1.0, device=device, dtype=dtype)
        alpha_eff = d_eta / denom
        eps = torch.tensor(1e-8, device=device, dtype=dtype)
        w = torch.where(alpha_eff.abs() > eps, (1.0 - sigma) / alpha_eff, torch.tensor(h, device=device, dtype=dtype))

        Pk1 = sigma * Pk + w * Gv
        Xk1 = Xk + h * (Pk1 / z.unsqueeze(-1))
        if self.presymp_lnp != "none":
            Pk1 = self._apply_lnp(Pk1)
        self.last_h = h
        return Xk1, Pk1
class HalfDampStrangAttention(nn.Module):
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
    ):
        super().__init__()
        self.cfg = cfg

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
            m = causal_mask(T, q.device)
            S = S.masked_fill(m.unsqueeze(0).unsqueeze(0), float("-inf"))
        S = S.mean(dim=1)
        S = torch.clamp(S, min=-60.0, max=60.0)
        E = torch.exp(S)
        if self.causal:
            m = causal_mask(T, q.device)
            E = E.masked_fill(m.unsqueeze(0), 0.0)
        z = E.sum(dim=-1) / float(T)
        z = z.clamp_min(self.eps)
        return E, z

    def _apply_lnp(self, P: torch.Tensor) -> torch.Tensor:
        if self.presymp_lnp == "none":
            return P
        out = self.ln_p(P.float())
        return out.to(dtype=P.dtype)


    def _velH(self, X: torch.Tensor, P: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (F, E, z, v_merge) for conservative dynamics."""
        x_ln = self.ln(X)
        q, k, v = self._qkv(x_ln)
        E, z = self._kernel_E_z(q, k)
        B, T, C = X.shape
        v_merge = v.transpose(1, 2).contiguous().view(B, T, C)
        Fv = P / z.unsqueeze(-1)
        return Fv, E, z, v_merge

    def _forH(self, X: torch.Tensor, P: torch.Tensor, E: torch.Tensor, z: torch.Tensor, v_merge: torch.Tensor) -> torch.Tensor:
        a = (P * P).sum(dim=-1)
        s = a / (z * z + self.eps)
        M = E * (s.unsqueeze(-1) + s.unsqueeze(-2) - 2.0)
        core = torch.matmul(M, v_merge) / (2.0 * float(X.shape[1]))
        core = self.resid_drop(self.c_proj(core))
        return core

    def _conservative_doubling_step(self, X0: torch.Tensor, P0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Explicit 2nd-order doubling-trick integrator for the conservative part."""
        device, dtype = X0.device, X0.dtype
        h = float(self.h(device=device, dtype=dtype).detach().cpu().item())
        tau = 0.5 * h
        xi_eff = float(self.xi(h=torch.tensor(h, device=device, dtype=dtype), device=device, dtype=dtype).detach().cpu().item())
        theta = 2.0 * xi_eff * h
        c = math.cos(theta)
        s = math.sin(theta)

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
        h = float(self.h(device=device, dtype=dtype).detach().cpu().item())
        tk = self.sched.t0 + float(k) * h

        # exact half damping
        d_eta = self.sched.delta_eta(tk, 0.5 * h, device, dtype)
        sigma_half = torch.exp(-d_eta)  # scalar tensor
        P_half = sigma_half * Pk

        # conservative step
        X1, P1 = self._conservative_doubling_step(Xk, P_half)

        # exact half damping
        Pk1 = sigma_half * P1
        if self.presymp_lnp != "none":
            Pk1 = self._apply_lnp(Pk1)
        self.last_h = h
        self.last_xi = float(self.xi(h=torch.tensor(h, dtype=torch.float32)).detach().cpu().item())
        return X1, Pk1



class PresympGPTBlock(nn.Module):
    '''Replace attention with presymplectic step; accelerate MLP with a Nesterov-style velocity stream.'''
    def __init__(
        self,
        cfg: ModelConfig,
        mlp_use_attn_vel: bool = False,
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
        self.mlp_use_attn_vel = bool(mlp_use_attn_vel)
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
            )
        else:
            raise ValueError("attn_scheme must be one of {'presymp','euler','exp_euler','strang'}")

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
    ):
        super().__init__()
        self.cfg = cfg
        self.h = float(h)
        self.use_v0_init = bool(use_v0_init)
        self.mlp_use_attn_vel = bool(mlp_use_attn_vel)

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
            if self.mlp_use_attn_vel:
                v = torch.zeros_like(x)
            else:
                v = self.drop(self.tok_v0_emb_mlp(idx) + self.pos_v0_emb_mlp(pos)[None, :, :])
        else:
            p = torch.zeros_like(x)
            v = torch.zeros_like(x)

        dx_prev = None
        dp_prev = None

        for k in range(self.cfg.n_layer):
            h_k = self.attn[k].h()  # learned scalar tensor, kept in graph
            dx_k, dp_k = self.attn[k].rhs(x, p, k)
            if k == 0 or dx_prev is None:
                dx_eff, dp_eff = dx_k, dp_k
            else:
                dx_eff = 1.5 * dx_k - 0.5 * dx_prev
                dp_eff = 1.5 * dp_k - 0.5 * dp_prev

            x_new = x + h_k * dx_eff
            p_new = p + h_k * dp_eff
            if self.attn[k].presymp_lnp != "none":
                p_new = self.attn[k]._apply_lnp(p_new)

            v_attn = (x_new - x) / h_k.clamp(min=1e-8)
            x, p = x_new, p_new
            dx_prev, dp_prev = dx_k, dp_k

            x, v = self.mlp_steps[k](x, v, v_attn=v_attn)

        h_vals = [self.attn[k].h().detach().cpu().item() for k in range(self.cfg.n_layer)]
        self.last_h_mean = sum(h_vals) / len(h_vals)
        # xi is not used in AB2 (DampedEulerAttention has no theta_xi_raw)
        self.last_xi_mean = float('nan')

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
            if self.mlp_use_attn_vel:
                v = torch.zeros_like(x)
            else:
                v = self.drop(self.tok_v0_emb_mlp(idx) + self.pos_v0_emb_mlp(pos)[None, :, :])
        else:
            p = torch.zeros_like(x)
            v = torch.zeros_like(x)

        H_prev = None

        for k in range(self.cfg.n_layer):
            h_k = self.attn[k].h()  # learned scalar tensor, kept in graph
            h_k_f = h_k.detach().item()  # Python float for time index only
            Fv, Gv, z, _alpha = self.attn[k].FG_alpha(x, p, k)
            tk = self.attn[k].sched.t0 + float(k) * h_k_f
            device, dtype = x.device, x.dtype
            Lam_k = self.attn[k].sched.exp_eta(tk, device, dtype)
            Lam_k1 = self.attn[k].sched.exp_eta(tk + h_k_f, device, dtype)

            Pi = Lam_k * p
            H_k = Lam_k * Gv

            if k == 0 or H_prev is None:
                H_eff = H_k
            else:
                H_eff = 1.5 * H_k - 0.5 * H_prev

            Pi_new = Pi + h_k * H_eff
            p_new = Pi_new / Lam_k1

            x_new = x + h_k * (p_new / z.unsqueeze(-1))
            if self.attn[k].presymp_lnp != "none":
                p_new = self.attn[k]._apply_lnp(p_new)

            v_attn = (x_new - x) / h_k.clamp(min=1e-8)
            x, p = x_new, p_new
            H_prev = H_k

            x, v = self.mlp_steps[k](x, v, v_attn=v_attn)

        h_vals = [self.attn[k].h().detach().cpu().item() for k in range(self.cfg.n_layer)]
        self.last_h_mean = sum(h_vals) / len(h_vals)
        self.last_xi_mean = float('nan')

        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


class GPTModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([GPTBlock(cfg) for _ in range(cfg.n_layer)])
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
    ):
        super().__init__()
        self.cfg = cfg
        self.noise_eta = float(noise_eta)
        self.noise_gamma = float(noise_gamma)
        self.noise_loc = str(noise_loc)
        self.restart_mode = str(restart_mode)
        self.restart_min_layer = int(restart_min_layer)

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

        self.blocks = nn.ModuleList([YuriiFormerLieTrotterBlock(cfg) for _ in range(cfg.n_layer)])
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
    ):
        super().__init__()
        self.cfg = cfg
        self.mlp_use_attn_vel = bool(mlp_use_attn_vel)
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
            if self.mlp_use_attn_vel:
                # Variant A ignores the learned MLP velocity state.
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

        self.last_rX_max = rX_max
        self.last_rP_max = rP_max
        self.last_xi_mean = (xi_sum / xi_cnt) if xi_cnt > 0 else float('nan')
        self.last_h_mean = (h_sum / h_cnt) if h_cnt > 0 else float('nan')

        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
