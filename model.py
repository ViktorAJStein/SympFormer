
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 50304
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False


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

    Default (mu is None): alpha(t)=3/t (Nesterov) => eta(t)=3*log(t/t0).
    Linear (mu provided): eta(t)=mu*t.

    Notes:
      * t0 is still used as the time-origin in the integrator (t_k = t0 + k*h).
      * For the default schedule, t0 must be > 0. For the linear schedule, t0 is
        not used inside eta(t) but is kept for consistency of time indexing.
    """

    def __init__(self, t0: float = 1.0, mu: Optional[float] = None):
        super().__init__()
        if t0 <= 0:
            raise ValueError("t0 must be > 0")
        self.t0 = float(t0)
        self.mu = None if mu is None else float(mu)

    def eta(self, t: float, device, dtype) -> torch.Tensor:
        if self.mu is None:
            tt = torch.tensor(t / self.t0, device=device, dtype=dtype)
            return 3.0 * torch.log(tt)
        else:
            tt = torch.tensor(t, device=device, dtype=dtype)
            return tt * self.mu

    def exp_eta(self, t: float, device, dtype) -> torch.Tensor:
        return torch.exp(self.eta(t, device, dtype))

    def exp_minus_eta(self, t: float, device, dtype) -> torch.Tensor:
        return torch.exp(-self.eta(t, device, dtype))


class PresymplecticSoftmaxAttention(nn.Module):
    '''Explicit 2nd-order presymplectic integrator (variable doubling) for a symmetric softmax kernel.'''
    def __init__(
        self,
        cfg: ModelConfig,
        h: float = 1.0,
        xi: float = 1.0,
        # Data-driven tuning of xi based on mismatch of doubled variables
        xi_adapt: bool = False,
        r_thresh: float = 1e-2,
        r_low: float = 1e-4,
        xi_mult_up: float = 2.0,
        xi_mult_down: float = 0.5,
        xi_min: float = 1e-4,
        xi_max: float = 100.0,
        xi_adapt_every: int = 1,
        t0: float = 1.0,
        eta_mu: Optional[float] = None,
        eps: float = 1e-8,
        causal: bool = True,
        use_v_ln: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.h = float(h)
        self.xi = float(xi)
        self.xi_adapt = bool(xi_adapt)
        self.r_thresh = float(r_thresh)
        self.r_low = float(r_low)
        self.xi_mult_up = float(xi_mult_up)
        self.xi_mult_down = float(xi_mult_down)
        self.xi_min = float(xi_min)
        self.xi_max = float(xi_max)
        self.xi_adapt_every = int(max(1, xi_adapt_every))
        self._xi_adapt_ctr = 0
        # last diagnostics (max over batch)
        self.last_rX = 0.0
        self.last_rP = 0.0
        self.last_xi = float(xi)
        self.last_xi_changed = False
        self.eps = float(eps)
        self.causal = bool(causal)

        self.ln = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.ln_v = LayerNorm(cfg.n_embd, bias=cfg.bias) if use_v_ln else nn.Identity()
        self.sched = EtaSchedule(t0=t0, mu=eta_mu)

        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def _kernel_E_z(self, U: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, nh, hd = U.shape
        S = torch.einsum("bthd,bshd->bhts", U, U) * self.scale
        S = S.mean(dim=1)  # (B, T, T)

        if self.causal:
            m = causal_mask(T, U.device)
            S = S.masked_fill(m.unsqueeze(0), float("-inf"))

        # clamp for numerical safety
        S = torch.clamp(S, min=-60.0, max=60.0)
        E = torch.exp(S)
        if self.causal:
            m = causal_mask(T, U.device)
            E = E.masked_fill(m.unsqueeze(0), 0.0)

        z = E.sum(dim=-1) / float(T)
        z = z.clamp_min(self.eps)
        return E, z

    def _vel(self, t: float, X: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
        device, dtype = X.device, X.dtype
        lam = self.sched.exp_minus_eta(t, device, dtype)
        p = lam * Pi
        U = self.ln(X).view(X.shape[0], X.shape[1], self.n_head, self.head_dim)
        _, z = self._kernel_E_z(U)
        return p / z.unsqueeze(-1)

    def _force(self, t: float, X: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
        device, dtype = X.device, X.dtype
        Lam = self.sched.exp_eta(t, device, dtype)
        lam = self.sched.exp_minus_eta(t, device, dtype)
        p = lam * Pi

        U = self.ln(X).view(X.shape[0], X.shape[1], self.n_head, self.head_dim)
        E, z = self._kernel_E_z(U)

        a = (p * p).sum(dim=-1)  # (B,T)
        s = a / (z * z + self.eps)
        M = E * (s.unsqueeze(-1) + s.unsqueeze(-2) - 2.0)
        core = torch.matmul(M, self.ln(X)) / (2.0 * float(X.shape[1]))
        return Lam * core

    def step(self, Xk: torch.Tensor, Pk: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.h
        tau = 0.5 * h
        t0 = self.sched.t0
        tk = t0 + float(k) * h
        tk1 = tk + h

        device, dtype = Xk.device, Xk.dtype
        Lam_tk = self.sched.exp_eta(tk, device, dtype)
        Pi = Lam_tk * Pk

        t = tk
        X = Xk
        bar_t = tk
        bar_X = Xk
        bar_Pi = Pi

        theta = 2.0 * self.xi * h
        c = math.cos(theta)
        s = math.sin(theta)

        # phi_A^{tau}
        Pi = Pi + tau * self._force(t, X, bar_Pi)
        bar_t = bar_t + tau
        bar_X = bar_X + tau * self._vel(t, X, bar_Pi)

        # phi_B^{tau}
        t = t + tau
        X = X + tau * self._vel(bar_t, bar_X, Pi)
        bar_Pi = bar_Pi + tau * self._force(bar_t, bar_X, Pi)

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
        Pk1 = self.ln_v(Pk1_raw)
        bar_Pk1 = self.ln_v(bar_Pk1_raw)

        # Data-driven xi tuning based on mismatch of the doubled variables.
        # r_X := ||X - X_bar|| / (||X|| + eps), r_P := ||P - P_bar|| / (||P|| + eps)
        if self.training and self.xi_adapt:
            with torch.no_grad():
                Bsz = X.shape[0]
                dX = X - bar_X
                nX = torch.linalg.vector_norm(X.reshape(Bsz, -1), ord=2, dim=1)
                ndX = torch.linalg.vector_norm(dX.reshape(Bsz, -1), ord=2, dim=1)
                rX = ndX / (nX + self.eps)

                dP = Pk1 - bar_Pk1
                nP = torch.linalg.vector_norm(Pk1.reshape(Bsz, -1), ord=2, dim=1)
                ndP = torch.linalg.vector_norm(dP.reshape(Bsz, -1), ord=2, dim=1)
                rP = ndP / (nP + self.eps)

                rX_max = float(rX.max().item())
                rP_max = float(rP.max().item())
                self.last_rX = rX_max
                self.last_rP = rP_max

                self._xi_adapt_ctr += 1
                xi_old = self.xi
                self.last_xi_changed = False
                if (self._xi_adapt_ctr % self.xi_adapt_every) == 0:
                    r = max(rX_max, rP_max)
                    if r > self.r_thresh:
                        self.xi = min(self.xi * self.xi_mult_up, self.xi_max)
                    elif r < self.r_low:
                        self.xi = max(self.xi * self.xi_mult_down, self.xi_min)
                    self.last_xi_changed = (self.xi != xi_old)
                self.last_xi = float(self.xi)
        else:
            self.last_xi = float(self.xi)
            self.last_xi_changed = False

        return X, Pk1



class PresympGPTBlock(nn.Module):
    '''Replace attention with presymplectic step; keep MLP sublayer.'''
    def __init__(
        self,
        cfg: ModelConfig,
        h: float = 1.0,
        xi: float = 1.0,
        t0: float = 1.0,
        eta_mu: Optional[float] = None,
        # xi adaptation options
        xi_adapt: bool = False,
        r_thresh: float = 1e-2,
        r_low: float = 1e-4,
        xi_mult_up: float = 2.0,
        xi_mult_down: float = 0.5,
        xi_min: float = 1e-4,
        xi_max: float = 100.0,
        xi_adapt_every: int = 1,
    ):
        super().__init__()
        self.attn = PresymplecticSoftmaxAttention(
            cfg,
            h=h,
            xi=xi,
            xi_adapt=xi_adapt,
            r_thresh=r_thresh,
            r_low=r_low,
            xi_mult_up=xi_mult_up,
            xi_mult_down=xi_mult_down,
            xi_min=xi_min,
            xi_max=xi_max,
            xi_adapt_every=xi_adapt_every,
            t0=t0,
            eta_mu=eta_mu,
            use_v_ln=True,
        )
        self.ln_2 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, p: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, p = self.attn.step(x, p, k)
        x = x + self.mlp(self.ln_2(x))
        return x, p


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

        # last diagnostics for xi adaptation
        self.last_rX_max = 0.0
        self.last_rP_max = 0.0
        self.last_xi_mean = float(xi)

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
        h: float = 1.0,
        xi: float = 1.0,
        t0: float = 1.0,
        eta_mu: Optional[float] = None,
        use_v0_init: bool = True,
        # xi adaptation options
        xi_adapt: bool = False,
        r_thresh: float = 1e-2,
        r_low: float = 1e-4,
        xi_mult_up: float = 2.0,
        xi_mult_down: float = 0.5,
        xi_min: float = 1e-4,
        xi_max: float = 100.0,
        xi_adapt_every: int = 1,
    ):
        super().__init__()
        self.cfg = cfg
        self.use_v0_init = bool(use_v0_init)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)

        # Momentum init embeddings (same idea as YuriiFormer v0): separate token+pos tables
        # We reuse the same naming (tok_v0_emb/pos_v0_emb) so the optimizer grouping
        # treats them as embeddings with wd=0.1.
        self.tok_v0_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_v0_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([
            PresympGPTBlock(
                cfg,
                h=h,
                xi=xi,
                t0=t0,
                eta_mu=eta_mu,
                xi_adapt=xi_adapt,
                r_thresh=r_thresh,
                r_low=r_low,
                xi_mult_up=xi_mult_up,
                xi_mult_down=xi_mult_down,
                xi_min=xi_min,
                xi_max=xi_max,
                xi_adapt_every=xi_adapt_every,
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
            p = self.tok_v0_emb(idx) + self.pos_v0_emb(pos)[None, :, :]
            p = self.drop(p)
        else:
            p = torch.zeros_like(x)
        rX_max = 0.0
        rP_max = 0.0
        xi_sum = 0.0
        xi_cnt = 0
        for k, blk in enumerate(self.blocks):
            x, p = blk(x, p, k)
            attn = getattr(blk, 'attn', None)
            if attn is not None and hasattr(attn, 'last_rX'):
                rX_max = max(rX_max, float(attn.last_rX))
                rP_max = max(rP_max, float(attn.last_rP))
                xi_sum += float(attn.last_xi)
                xi_cnt += 1

        self.last_rX_max = rX_max
        self.last_rP_max = rP_max
        self.last_xi_mean = (xi_sum / xi_cnt) if xi_cnt > 0 else float('nan')

        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
