"""Native causal scaffold, extracted unchanged from the paper experiment source."""
import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    yurii_force_gain_floor: float = 0.0

    # Presymp Variant A: use attention-induced velocity for MLP lookahead
    presymp_mlp_use_attn_vel: bool = False
    # Exact causal linear-oracle implementation.
    lin_oracle_impl: str = "gram"
    lin_prefix_chunk_size: int = 64


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


class CapacityControlGPTBlock(GPTBlock):
    """GPT block with one zero-initialized C-by-C residual adapter.

    This matches the dominant extra per-layer parameter matrix in the causal
    presymplectic model while preserving the baseline function at
    initialization. It is a capacity control, not a headline method.
    """

    def __init__(self, cfg: ModelConfig, no_mlp: bool = False):
        super().__init__(cfg, no_mlp=no_mlp)
        self.capacity_adapter = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln_1(x)
        x = x + self.attn(h) + self.capacity_adapter(h)
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
        self.force_gain_floor = float(cfg.yurii_force_gain_floor)
        if not math.isfinite(self.force_gain_floor) or not 0 <= self.force_gain_floor < 1:
            raise ValueError('yurii_force_gain_floor must be finite in [0,1)')
        self.ln_x_attn = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_x_mlp = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

        self.ln_v = LayerNorm(cfg.n_embd, bias=cfg.bias)

        # learned scalars for the two substeps
        self.mu1 = ConstrainedScalar(0.9, "unit")
        self.beta1 = ConstrainedScalar(0.9, "unit")
        self.gamma1 = ConstrainedScalar(1.0 - self.force_gain_floor, "pos")

        self.mu2 = ConstrainedScalar(0.9, "unit")
        self.beta2 = ConstrainedScalar(0.9, "unit")
        self.gamma2 = ConstrainedScalar(1.0 - self.force_gain_floor, "pos")

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
        gamma1 = self.gamma1() + self.force_gain_floor

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
        gamma2 = self.gamma2() + self.force_gain_floor

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


class GPTModel(nn.Module):
    def __init__(self, cfg: ModelConfig, no_mlp: bool = False, capacity_control: bool = False, share_layers: bool = False):
        super().__init__()
        self.cfg = cfg
        self.no_mlp = bool(no_mlp)
        self.capacity_control = bool(capacity_control)
        self.share_layers = bool(share_layers)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

        block_cls = CapacityControlGPTBlock if self.capacity_control else GPTBlock
        if self.share_layers:
            shared_block = block_cls(cfg, no_mlp=self.no_mlp)
            self.blocks = nn.ModuleList([shared_block for _ in range(cfg.n_layer)])
        else:
            self.blocks = nn.ModuleList([block_cls(cfg, no_mlp=self.no_mlp) for _ in range(cfg.n_layer)])
        self.ln_f = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self.last_rX_max = 0.0
        self.last_rP_max = 0.0
        self.last_xi_mean = 0.0
        self.last_h_mean = 0.0

        self.apply(self._init_weights)
        if self.capacity_control:
            # Preserve exact functional equality with the baseline at step zero.
            for block in self.blocks:
                nn.init.zeros_(block.capacity_adapter.weight)

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
        share_layers: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.noise_eta = float(noise_eta)
        self.noise_gamma = float(noise_gamma)
        self.noise_loc = str(noise_loc)
        self.restart_mode = str(restart_mode)
        self.restart_min_layer = int(restart_min_layer)
        self.no_mlp = bool(no_mlp)
        self.share_layers = bool(share_layers)

        if self.noise_loc not in ("dx", "v", "xin"):
            raise ValueError("noise_loc must be one of {'dx','v','xin'}")
        if self.restart_mode not in ("none", "speed", "loss"):
            raise ValueError("restart_mode must be one of {'none','speed','loss'}")
        # Main token/position embeddings (as in baseline)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)

        # v0 initialization embeddings for momentum variants (Appendix A.1)
        # "we initialize v0 using token and positional embedding tables separate from
        #  the main token and positional embeddings."
        self.use_v0_init = bool(use_v0_init)
        if self.use_v0_init:
            self.tok_v0_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
            self.pos_v0_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        else:
            self.tok_v0_emb = None
            self.pos_v0_emb = None
        self.drop = nn.Dropout(cfg.dropout)

        if self.share_layers:
            shared_block = YuriiFormerLieTrotterBlock(cfg, no_mlp=self.no_mlp)
            self.blocks = nn.ModuleList([shared_block for _ in range(cfg.n_layer)])
        else:
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
