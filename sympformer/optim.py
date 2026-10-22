"""Paper FP32, initialized velocity tables, schedule and embedding-decay optimizer."""
import math
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW

def configure_precision(name, device):
    """Explicit IEEE FP32 mode; preserve all legacy autocast defaults."""
    if name == "float32":
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return getattr(torch, name) if device.startswith("cuda") else torch.float32


def cosine_lr(step: int, warmup_steps: int, total_steps: int, peak: float, min_ratio: float = 0.1) -> float:
    if step < warmup_steps:
        return peak * (step / max(1, warmup_steps))
    if step >= total_steps:
        return peak * min_ratio
    # cosine from peak -> peak*min_ratio
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
    return peak * (min_ratio + (1.0 - min_ratio) * cosine)


class MixedOptimizer:
    """Minimal common interface for one Muon and one AdamW optimizer."""

    def __init__(self, muon, adamw):
        self.muon = muon
        self.adamw = adamw
        self.param_groups = muon.param_groups + adamw.param_groups

    def zero_grad(self, set_to_none=True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.muon.step()
        self.adamw.step()

    def state_dict(self):
        return {"kind": "muon_adamw", "muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

    def load_state_dict(self, state):
        if state.get("kind") != "muon_adamw":
            raise ValueError("Checkpoint optimizer does not contain mixed Muon+AdamW state")
        self.muon.load_state_dict(state["muon"])
        self.adamw.load_state_dict(state["adamw"])


def initialize_learned_v0_tables(model: nn.Module, seed: int) -> int:
    """Initialize explicit learned-v0 tables identically across architectures.

    Model constructors consume architecture-dependent amounts of random state.
    A dedicated CPU generator makes the token/position velocity tables an
    exactly paired initialization for runs sharing ``seed`` without changing
    any core-model parameters or the global training RNG stream.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 10_000_019)
    initialized = 0
    with torch.no_grad():
        for name in ("tok_v0_emb", "pos_v0_emb", "tok_v0_emb_mlp", "pos_v0_emb_mlp"):
            module = getattr(model, name, None)
            if module is not None:
                module.weight.normal_(mean=0.0, std=0.02, generator=generator)
                initialized += module.weight.numel()
    if initialized == 0:
        raise ValueError("--learned_v0_init was requested for a model without velocity embedding tables")
    return initialized


def _parameter_buckets(model):
    embeddings, norms_scalars, matrices = [], [], []
    scalar_names = ("theta_h", "theta_hX", "theta_hY", "theta_tau", "theta_xi_raw")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_embedding = any(token in name for token in ("tok_emb", "pos_emb", "tok_v0_emb", "pos_v0_emb"))
        is_scalar = name.endswith(".raw") or any(token in name for token in scalar_names)
        is_norm = "ln_" in name or ".ln" in name or "ln_f" in name or "ln_v" in name
        if is_embedding:
            embeddings.append(param)
        elif is_scalar or is_norm or param.ndim < 2:
            norms_scalars.append((name, param))
        else:
            matrices.append(param)
    return embeddings, norms_scalars, matrices


def build_optimizer(
    model: nn.Module,
    peak_lr: float,
    betas=(0.9, 0.95),
    scalar_lr_mult: float = 5.0,
    optimizer_name: str = "muon_adamw",
    muon_lr: float = 0.02,
):
    embeddings, norms_scalars, matrices = _parameter_buckets(model)
    scalar_params = [p for name, p in norms_scalars if ".raw" in name or "theta_" in name]
    other_adamw = [p for name, p in norms_scalars if not (".raw" in name or "theta_" in name)]

    if optimizer_name == "adamw":
        groups = []
        if matrices:
            groups.append({"params": matrices, "lr": peak_lr, "weight_decay": 0.0})
        if embeddings:
            groups.append({"params": embeddings, "lr": peak_lr, "weight_decay": 0.1})
        if other_adamw:
            groups.append({"params": other_adamw, "lr": peak_lr, "weight_decay": 0.0})
        if scalar_params:
            groups.append({"params": scalar_params, "lr": peak_lr * scalar_lr_mult, "weight_decay": 0.0, "lr_mult": scalar_lr_mult})
        return AdamW(groups, betas=betas)

    if optimizer_name != "muon_adamw":
        raise ValueError(f"Unknown optimizer {optimizer_name!r}")
    if not matrices:
        raise ValueError("Muon requires at least one non-embedding matrix parameter")

    muon_mult = muon_lr / peak_lr
    muon = torch.optim.Muon(
        [{"params": matrices, "lr": muon_lr, "weight_decay": 0.0, "lr_mult": muon_mult}],
        lr=muon_lr,
        momentum=0.95,
        weight_decay=0.0,
    )
    adam_groups = []
    if embeddings:
        adam_groups.append({"params": embeddings, "lr": peak_lr, "weight_decay": 0.1})
    if other_adamw:
        adam_groups.append({"params": other_adamw, "lr": peak_lr, "weight_decay": 0.0})
    if scalar_params:
        adam_groups.append({"params": scalar_params, "lr": peak_lr * scalar_lr_mult, "weight_decay": 0.0, "lr_mult": scalar_lr_mult})
    adamw = AdamW(adam_groups, betas=betas)
    return MixedOptimizer(muon, adamw)
