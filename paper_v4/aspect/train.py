
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import warnings
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from data import DataConfig, BlockEpochIterator, load_bin
from aspect_ratio_protocol import SharedTargetIterator
from model import ModelConfig, GPTModel, YuriiFormerModel, PresympModel, PresympModelAB2, PresympModelETDAB2, LinAttnModel, LinAttnYuriiModel, LinAttnEulerModel, LinAttnPresympModel, LinAttnAB2Model, LinAttnETDAB2Model, LinAttnReducedModel, CausalRiemannianNAGModel


from riemannian_prefix import PrefixGeometryConfig, RiemannianPrefixModel
from matched_prefix import MatchedPrefixConfig, MatchedPrefixModel
from normalized_hb import NormalizedHBConfig, NormalizedHBModel


class NonFiniteTrainingError(RuntimeError):
    """Structured failure raised before a nonfinite update can propagate."""

    def __init__(self, kind, step, micro_step=None, value=None, detail=""):
        self.kind = str(kind)
        self.step = int(step)
        self.micro_step = None if micro_step is None else int(micro_step)
        self.value = None if value is None else str(value)
        self.detail = str(detail)
        location = f"step={self.step}"
        if self.micro_step is not None:
            location += f" micro_step={self.micro_step}"
        rendered = f"Nonfinite {self.kind} at {location}"
        if self.value is not None:
            rendered += f" value={self.value}"
        if self.detail:
            rendered += f": {self.detail}"
        super().__init__(rendered)


def require_finite_scalar(value, *, kind, step, micro_step=None):
    """Return a scalar value or raise a structured nonfinite failure."""
    numeric = float(value.detach().item()) if torch.is_tensor(value) else float(value)
    if not math.isfinite(numeric):
        raise NonFiniteTrainingError(
            kind,
            step,
            micro_step=micro_step,
            value=repr(numeric),
        )
    return numeric


def clip_grad_norm_finite(parameters, max_norm, *, step):
    """Clip gradients and reject a NaN/Inf total norm before the optimizer step."""
    effective_max_norm = float(max_norm) if float(max_norm) > 0 else float("inf")
    try:
        total_norm = torch.nn.utils.clip_grad_norm_(
            list(parameters),
            effective_max_norm,
            error_if_nonfinite=True,
        )
    except RuntimeError as exc:
        raise NonFiniteTrainingError(
            "gradient_norm",
            step,
            detail=str(exc),
        ) from exc
    return require_finite_scalar(total_norm, kind="gradient_norm", step=step)

def maybe_get_tokenizer():
    try:
        import tiktoken
    except Exception as e:
        print(f"[sample] tiktoken unavailable ({e}); decoded text samples disabled")
        return None
    try:
        return tiktoken.get_encoding("gpt2")
    except Exception as e:
        print(f"[sample] failed to load GPT-2 tokenizer ({e}); decoded text samples disabled")
        return None


def _find_story_start(tokens: np.ndarray, eot_token: int = 50256) -> int:
    if len(tokens) == 0:
        return 0
    hit = np.flatnonzero(tokens == eot_token)
    if hit.size == 0:
        return 0
    start = int(hit[0]) + 1
    return min(start, max(0, len(tokens) - 1))


def build_prompt_tokens(args, dataset_tokens: np.ndarray, enc):
    if args.sample_prompt:
        if enc is None:
            raise RuntimeError("A text prompt requires tiktoken to be installed.")
        toks = enc.encode_ordinary(args.sample_prompt)
        if len(toks) == 0:
            raise RuntimeError("The provided sample prompt tokenized to an empty sequence.")
        toks = toks[-args.block_size:]
        return torch.tensor(toks, dtype=torch.long).unsqueeze(0), "prompt"

    if len(dataset_tokens) < 2:
        raise RuntimeError("Not enough dataset tokens to build a sample prompt.")

    start = _find_story_start(dataset_tokens)
    n_pref = max(1, int(args.sample_prefix_tokens))
    end = min(start + n_pref, len(dataset_tokens) - 1)
    if end <= start:
        start = 0
        end = min(n_pref, len(dataset_tokens) - 1)
    toks = dataset_tokens[start:end].astype(np.int64)
    return torch.from_numpy(toks).unsqueeze(0), "val_prefix"


@torch.no_grad()
def print_sample(
    model: nn.Module,
    dataset_tokens: np.ndarray,
    device: str,
    args,
    global_step: int,
    *,
    run_dir: str,
    force: bool = False,
):
    if int(args.sample_interval) <= 0 and not force:
        return None

    enc = maybe_get_tokenizer()
    prompt_cpu, prompt_kind = build_prompt_tokens(args, dataset_tokens, enc)
    prompt = prompt_cpu.to(device)

    was_training = model.training
    model.eval()
    out = model.generate(
        prompt,
        max_new_tokens=int(args.sample_max_new_tokens),
        temperature=float(args.sample_temperature),
        top_k=(None if int(args.sample_top_k) <= 0 else int(args.sample_top_k)),
        do_sample=bool(int(args.sample_do_sample)),
        eos_token_id=(None if int(args.sample_eos_token_id) < 0 else int(args.sample_eos_token_id)),
        global_step=global_step,
    )
    if was_training:
        model.train()
    out_cpu = out[0].detach().cpu().tolist()
    prompt_len = prompt_cpu.shape[1]

    prompt_text = gen_text = full_text = None
    if enc is not None:
        prompt_text = enc.decode(out_cpu[:prompt_len])
        gen_text = enc.decode(out_cpu[prompt_len:])
        full_text = enc.decode(out_cpu)
        print(f"[sample][step {global_step}] source={prompt_kind}")
        print("[sample][prompt]")
        print(prompt_text)
        print("[sample][continuation]")
        print(gen_text)
        print("[sample][full]")
        print(full_text)
    else:
        print(f"[sample][step {global_step}] source={prompt_kind} (token ids only)")
        print("[sample][prompt_ids]")
        print(out_cpu[:prompt_len])
        print("[sample][continuation_ids]")
        print(out_cpu[prompt_len:])

    record = {
        "arch": args.arch,
        "dataset": args.dataset,
        "run_name": args.run_name,
        "seed": args.seed,
        "global_step": global_step,
        "source": prompt_kind,
        "do_sample": bool(int(args.sample_do_sample)),
        "temperature": float(args.sample_temperature),
        "top_k": None if int(args.sample_top_k) <= 0 else int(args.sample_top_k),
        "prompt_token_ids": out_cpu[:prompt_len],
        "continuation_token_ids": out_cpu[prompt_len:],
        "prompt_text": prompt_text,
        "continuation_text": gen_text,
        "full_text": full_text,
    }
    with open(os.path.join(run_dir, "samples.jsonl"), "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    return record



def ensure_csv_header(path: str, header):
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    if not exists:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)


def append_csv_row(path: str, row):
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow(row)


GEOMETRY_FIELDS = [
    "tau", "dt", "velocity_norm", "attention_norm", "connection_norm",
    "connection_update_norm", "attention_update_norm", "lookahead_displacement_norm",
    "total_displacement_norm", "connection_attention_ratio", "connection_displacement_ratio",
]

DYNAMICS_FIELDS = [
    "step", "tokens_cum", "phase", "layer", "shared", "module_type",
    "c_log", "c_lin", "hX", "hY", "beta_attn", "beta_mlp",
    "t_start", "t_end", "alpha_start", "alpha_end",
] + GEOMETRY_FIELDS


def set_geometry_capture(model, enabled):
    if isinstance(model, NormalizedHBModel):
        for block in model.blocks:
            block.capture_diagnostics = bool(enabled)
    for layer in getattr(model, "layers", []):
        if hasattr(layer, "capture_diagnostics"):
            layer.capture_diagnostics = bool(enabled)


def configure_precision(name, device):
    """Explicit IEEE FP32 mode; preserve all legacy autocast defaults."""
    if name == "float32":
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return getattr(torch, name) if device.startswith("cuda") else torch.float32


def _module_scalar(module, name=None):
    value = module if name is None else getattr(module, name, None)
    if value is None:
        return math.nan
    try:
        value = value() if callable(value) else value
    except TypeError:
        return math.nan
    if torch.is_tensor(value):
        value = value.detach().cpu().item()
    return float(value)


def _schedule_coefficients(sched):
    if sched is None:
        return math.nan, math.nan
    if getattr(sched, "learnable", False):
        c_log = _module_scalar(sched, "c_log") if getattr(sched, "c_log", None) is not None else 0.0
        c_lin = _module_scalar(sched, "c_lin") if getattr(sched, "c_lin", None) is not None else 0.0
    else:
        c_log = float(getattr(sched, "c_log_const", 0.0))
        c_lin = float(getattr(sched, "c_lin_const", 0.0))
    return c_log, c_lin


def collect_layer_dynamics(model):
    """Return post-update damping/step diagnostics for each logical layer.

    Shared blocks deliberately produce one row per application depth, with the
    ``shared`` field marking repeated object identity. This distinguishes
    logical forward time from the number of unique parameter sets.
    """
    if isinstance(model, RiemannianPrefixModel):
        return model.layer_dynamics()
    logical = []
    blocks = getattr(model, "blocks", None)
    if blocks is not None:
        for layer, block in enumerate(blocks):
            attn = getattr(block, "attn", None)
            if attn is not None or hasattr(block, "beta1"):
                logical.append((layer, block, attn))
    if not logical:
        attention = getattr(model, "attn", None)
        if attention is not None and hasattr(attention, "__iter__"):
            for layer, attn in enumerate(attention):
                logical.append((layer, None, attn))
    if not logical:
        attention = getattr(model, "attn_layers", None)
        if attention is not None and hasattr(attention, "__iter__"):
            for layer, attn in enumerate(attention):
                logical.append((layer, None, attn))
    if not logical:
        for layer, target in enumerate(getattr(model, "layers", [])):
            if hasattr(target, "theta_tau"):
                logical.append((layer, None, target))
    if not logical:
        return []

    identities = [id(attn if attn is not None else block) for _, block, attn in logical]
    t_cur = float(getattr(model, "last_t_start", 1.0))
    rows = []
    for layer, block, attn in logical:
        target = attn if attn is not None else block
        sched = getattr(target, "sched", None)
        beta_attn = _module_scalar(block, "beta1") if block is not None else math.nan
        beta_mlp = _module_scalar(block, "beta2") if block is not None else math.nan
        if block is None:
            model_beta = getattr(model, "beta", None)
            model_beta2 = getattr(model, "beta2", None)
            if model_beta is not None and layer < len(model_beta):
                beta_attn = _module_scalar(model_beta[layer])
            elif hasattr(target, "alpha_plain"):
                beta_attn = _module_scalar(target, "alpha_plain")
            if model_beta2 is not None and layer < len(model_beta2):
                beta_mlp = _module_scalar(model_beta2[layer])
            else:
                mlp_steps = getattr(model, "mlp_steps", None)
                if mlp_steps is not None and layer < len(mlp_steps):
                    beta_mlp = _module_scalar(mlp_steps[layer], "beta_mlp")
        c_log, c_lin = _schedule_coefficients(sched)
        hX = float(getattr(target, "last_h", math.nan))
        hY = float(getattr(target, "last_hY", math.nan))
        if not math.isfinite(hX):
            hX = _module_scalar(target, "hX")
        if not math.isfinite(hX):
            hX = _module_scalar(target, "h")
        if not math.isfinite(hY):
            hY = hX
        if hasattr(target, "theta_tau"):
            # last_h is tau, NOT the Riemannian layer's physical clock step.
            hX = math.sqrt(max(target.last_h, 1e-12))
            hY = math.nan  # no independent momentum-kick step in this route
            beta_attn = float(target.last_beta)
        if sched is None and not (math.isfinite(beta_attn) or math.isfinite(beta_mlp) or math.isfinite(hX)):
            continue
        t_end = t_cur + hX if math.isfinite(hX) else math.nan
        alpha_start = c_log / t_cur + c_lin if sched is not None and t_cur > 0 else math.nan
        alpha_end = c_log / t_end + c_lin if sched is not None and t_end > 0 else math.nan
        if sched is None and math.isfinite(beta_attn) and beta_attn > 0 and math.isfinite(hY) and hY > 0:
            # Effective continuous rate corresponding to the learned discrete retention.
            alpha_start = alpha_end = -math.log(beta_attn) / hY
        rows.append({
            "layer": layer,
            "shared": int(identities.count(id(target)) > 1),
            "module_type": type(target).__name__,
            "c_log": c_log,
            "c_lin": c_lin,
            "hX": hX,
            "hY": hY,
            "beta_attn": beta_attn,
            "beta_mlp": beta_mlp,
            "t_start": t_cur,
            "t_end": t_end,
            "alpha_start": alpha_start,
            "alpha_end": alpha_end,
            **{key: getattr(target, "last_geometry", {}).get(key, math.nan)
               for key in GEOMETRY_FIELDS},
        })
        if math.isfinite(t_end):
            t_cur = t_end
    return rows


def append_layer_dynamics(path, model, *, step, tokens_cum, phase):
    rows = collect_layer_dynamics(model)
    for row in rows:
        append_csv_row(path, [
            step, tokens_cum, phase, row["layer"], row["shared"], row["module_type"],
            row["c_log"], row["c_lin"], row["hX"], row["hY"],
            row["beta_attn"], row["beta_mlp"], row["t_start"], row["t_end"],
            row["alpha_start"], row["alpha_end"],
            *[row[key] for key in GEOMETRY_FIELDS],
        ])
    if isinstance(model, RiemannianPrefixModel) and rows:
        extra_path = os.path.join(os.path.dirname(path), 'prefix_dynamics.csv')
        keys = list(rows[0])
        ensure_csv_header(extra_path, ['step', 'tokens_cum', 'phase'] + keys)
        for row in rows:
            append_csv_row(extra_path, [step, tokens_cum, phase] + [row[k] for k in keys])
    if isinstance(model, NormalizedHBModel):
        extra = model.hb_dynamics()
        extra_path = os.path.join(os.path.dirname(path), 'hb_dynamics.csv')
        keys = list(extra[0])
        ensure_csv_header(extra_path, ['step', 'tokens_cum', 'phase'] + keys)
        for row in extra:
            append_csv_row(extra_path, [step, tokens_cum, phase] + [row[k] for k in keys])
    return len(rows)


def plot_layer_dynamics_csv(csv_path, out_prefix, title):
    """Plot per-layer learned damping over optimizer training time."""
    if not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot] matplotlib not available ({exc}); skipping damping plot")
        return
    with open(csv_path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("layer", "") != ""]
    if not rows:
        return
    layers = sorted({int(row["layer"]) for row in rows})
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 8.0))
    for field, ax, ylabel in (
        ("c_log", axes[0, 0], r"$c_{\log}$"),
        ("c_lin", axes[0, 1], r"$c_{\mathrm{lin}}$"),
        ("beta_attn", axes[0, 2], r"attention retention $\beta$"),
        ("hX", axes[1, 0], r"position step $h_X$"),
        ("hY", axes[1, 1], r"momentum step $h_Y$"),
    ):
        for layer in layers:
            selected = [row for row in rows if int(row["layer"]) == layer and row[field] not in ("", "nan")]
            if selected:
                ax.plot([float(row["tokens_cum"]) / 1e6 for row in selected],
                        [float(row[field]) for row in selected], lw=1.3, label=f"L{layer}")
        if field == "c_log":
            ax.axhline(3.0, color="black", ls="--", lw=1.1, label=r"Nesterov $r=3$")
        ax.set_xlabel("Training tokens (millions)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    terminal_step = max(int(row["step"]) for row in rows)
    terminal = [row for row in rows if int(row["step"]) == terminal_step and row["alpha_start"] not in ("", "nan")]
    ax = axes[1, 2]
    if terminal:
        xs = [int(row["layer"]) for row in terminal]
        alpha = [float(row["alpha_start"]) for row in terminal]
        ax.plot(xs, alpha, marker="o", label=r"implemented effective rate")
        scheduled = [row for row in terminal if row["c_log"] not in ("", "nan")]
        if scheduled:
            ax.plot([int(row["layer"]) for row in scheduled],
                    [3.0 / float(row["t_start"]) for row in scheduled],
                    marker="s", ls="--", label=r"reference $3/t$")
    ax.set_xlabel("Logical layer")
    ax.set_ylabel(r"terminal damping rate")
    ax.grid(alpha=0.25)
    for legend_ax in axes.flat:
        handles, labels = legend_ax.get_legend_handles_labels()
        if handles:
            legend_ax.legend(handles, labels, fontsize=7, ncol=2)
    fig.suptitle(title)
    fig.tight_layout()
    for suffix, kwargs in ((".pdf", {}), (".png", {"dpi": 190})):
        path = out_prefix + suffix
        fig.savefig(path, bbox_inches="tight", **kwargs)
        print(f"[plot] saved {path}")
    plt.close(fig)


def plot_metrics_csv(csv_path: str, out_png: str, title: str):
    """Single-run plot: train loss curve + val loss points (x-axis = step)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] matplotlib not available ({e}); skipping plot")
        return

    steps_train, loss_train = [], []
    steps_val, loss_val = [], []
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            step = int(row["step"])
            tl = row.get("train_loss", "")
            vl = row.get("val_loss", "")
            if tl != "":
                steps_train.append(step)
                loss_train.append(float(tl))
            if vl != "":
                steps_val.append(step)
                loss_val.append(float(vl))

    if not steps_train and not steps_val:
        print("[plot] no data in metrics csv; skipping plot")
        return

    plt.figure(figsize=(7, 4))
    if steps_train:
        plt.plot(steps_train, loss_train, label="train")
    if steps_val:
        plt.scatter(steps_val, loss_val, label="val", s=20)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[plot] saved {out_png}")


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


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    it: BlockEpochIterator,
    device: str,
    eval_batches: int,
    amp_dtype: torch.dtype,
    global_step: int,
    capture_geometry: bool = False,
):
    """Evaluate the same fixed validation batches without advancing ``it``."""
    eval_started = time.perf_counter()
    iterator_state = it.state_dict()
    was_training = model.training
    model.eval()
    losses = []
    try:
        for batch in range(eval_batches):
            set_geometry_capture(model, capture_geometry and batch == eval_batches - 1)
            xb, yb = next(it)
            xb = xb.to(device)
            yb = yb.to(device)
            with torch.autocast(
                device_type=device.split(':')[0],
                dtype=amp_dtype,
                enabled=device.startswith("cuda") and amp_dtype != torch.float32,
            ):
                _, loss = model(xb, yb, global_step=global_step)
            losses.append(loss.item())
    finally:
        it.load_state_dict(iterator_state)
        model.train(was_training)
        set_geometry_capture(model, False)
        model.last_eval_wall_s = time.perf_counter() - eval_started
        model.evaluation_forward_calls = getattr(model, "evaluation_forward_calls", 0) + len(losses)
    return float(np.mean(losses))


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def source_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def source_fingerprint(config_path: str):
    """Hash the executable source/config even when the checkout has no Git metadata."""
    root = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(root, "train.py"),
        os.path.join(root, "model.py"),
        os.path.join(root, "riemannian_prefix.py"),
        os.path.join(root, "matched_prefix.py"),
        os.path.join(root, "normalized_hb.py"),
        os.path.join(root, "data.py"),
        os.path.join(root, "aspect_ratio_protocol.py"),
        os.path.join(root, "pyproject.toml"),
        os.path.join(root, "uv.lock"),
    ]
    if config_path:
        candidates.append(
            config_path if os.path.isabs(config_path) else os.path.join(root, config_path)
        )
    combined = hashlib.sha256()
    records = []
    seen = set()
    for path in candidates:
        normalized = os.path.abspath(path)
        if normalized in seen or not os.path.isfile(normalized):
            continue
        seen.add(normalized)
        with open(normalized, "rb") as handle:
            payload = handle.read()
        try:
            label = os.path.relpath(normalized, root)
        except ValueError:
            label = os.path.basename(normalized)
        digest = hashlib.sha256(payload).hexdigest()
        records.append({"path": label, "sha256": digest, "bytes": len(payload)})
        combined.update(label.encode("utf-8"))
        combined.update(b"\0")
        combined.update(payload)
        combined.update(b"\0")
    return {"algorithm": "sha256", "sha256": combined.hexdigest(), "files": records}


def data_record(path: str, tokens: np.ndarray, require_manifest: bool):
    """Return validated dataset provenance for the run manifest."""
    record = {"path": path, "bytes": os.path.getsize(path), "tokens": len(tokens)}
    sidecar = path + ".manifest.json"
    if not os.path.exists(sidecar):
        if require_manifest:
            raise SystemExit(f"Missing required dataset manifest: {sidecar}")
        record["manifest"] = None
        return record
    with open(sidecar, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    if int(metadata.get("bytes", -1)) != record["bytes"]:
        raise SystemExit(f"Dataset byte count disagrees with {sidecar}")
    if int(metadata.get("tokens", -1)) != record["tokens"]:
        raise SystemExit(f"Dataset token count disagrees with {sidecar}")
    if metadata.get("dtype") != "uint16":
        raise SystemExit(f"Unsupported dataset dtype in {sidecar}: {metadata.get('dtype')!r}")
    record["manifest_path"] = sidecar
    record["manifest"] = metadata
    return record


def make_checkpoint(model, opt, next_step, best_val, mcfg, args, train_it, val_it, wall_cum_s):
    return {
        "format_version": 2,
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "next_step": int(next_step),
        "best_val": float(best_val),
        "cfg": asdict(mcfg),
        "args": vars(args),
        "train_iterator": train_it.state_dict(),
        "val_iterator": val_it.state_dict(),
        "rng_state": capture_rng_state(),
        "wall_cum_s": float(wall_cum_s),
        "geometry_work_by_layer": [dict(layer.work_counts) for layer in model.layers] if isinstance(model, RiemannianPrefixModel) else None,
        "evaluation_forward_calls": getattr(model, 'evaluation_forward_calls', 0),
        "hb_work_by_layer": [dict(block.work_counts) for block in model.blocks] if isinstance(model, NormalizedHBModel) else None,
    }


def write_nonfinite_failure(
    run_dir,
    error,
    *,
    model,
    opt,
    best_val,
    mcfg,
    args,
    train_it,
    val_it,
    wall_cum_s,
    tokens_per_step,
    lr,
):
    """Persist an auditable diagnostic and a non-resumable forensic checkpoint."""
    failure_path = os.path.join(run_dir, "failure.json")
    checkpoint_path = os.path.join(run_dir, f"failure_{args.arch}.pt")
    payload = {
        "format_version": 1,
        "status": "failed_nonfinite",
        "failure_type": error.kind,
        "message": str(error),
        "step": error.step,
        "micro_step": error.micro_step,
        "value": error.value,
        "detail": error.detail,
        "tokens_completed": int(error.step) * int(tokens_per_step),
        "wall_cum_s": float(wall_cum_s),
        "lr": float(lr),
        "run_name": args.run_name,
        "arch": args.arch,
        "seed": int(args.seed),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "checkpoint_resumable": False,
    }
    with open(failure_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)

    try:
        checkpoint = make_checkpoint(
            model,
            opt,
            error.step,
            best_val,
            mcfg,
            args,
            train_it,
            val_it,
            wall_cum_s,
        )
        checkpoint["failure"] = dict(payload)
        checkpoint["resumable"] = False
        torch.save(checkpoint, checkpoint_path)
        payload["forensic_checkpoint"] = checkpoint_path
    except Exception as exc:  # Preserve the primary failure if storage is full.
        payload["forensic_checkpoint_error"] = repr(exc)

    with open(failure_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
    return failure_path, payload.get("forensic_checkpoint")

def main():
    config_probe = argparse.ArgumentParser(add_help=False)
    config_probe.add_argument("--config", type=str, default="")
    config_args, _ = config_probe.parse_known_args()
    ap = argparse.ArgumentParser(parents=[config_probe])
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--dataset", type=str, default="tinystories", choices=["tinystories", "openwebtext"])
    ap.add_argument(
        "--arch",
        type=str,
        default="yurii_lt",
        choices=["baseline", "baseline_capacity", "yurii_lt", "riem_prefix", "matched_prefix", "normalized_hb", "causal_symp_fe", "causal_symp_pe", "causal_symp_exp_pe", "causal_symp_halfdamp_pe", "causal_symp_ab2", "causal_riem_nag_noconn", "causal_riem_nag", "presymp", "presymp_euler", "presymp_exp_euler", "presymp_ab2", "presymp_etd_ab2", "presymp_strang", "plain_euler", "lin_baseline", "lin_yurii", "lin_euler", "lin_presymp", "lin_exp_euler", "lin_ab2", "lin_etd_ab2", "lin_reduced_exp_mid", "lin_reduced_ab2"],
        help="model architecture / attention discretization",
    )

    ap.add_argument(
        "--lin_noncausal",
        action="store_true",
        help="Use global unmasked linear attention. Required by reduced matrix-momentum v3 schemes; invalid for headline next-token quality comparisons.",
    )
    ap.add_argument(
        "--no_mlp",
        action="store_true",
        help="Skip the MLP substep entirely in all architectures. "
             "Use this to isolate the effect of the different attention blocks.",
    )
    ap.add_argument(
        "--lin_mlp_mode",
        choices=("auto", "standard", "momentum"),
        default="auto",
        help="MLP route for linear-attention factorial attribution. 'auto' preserves "
             "the historical baseline=standard and token-momentum=momentum routes.",
    )
    ap.add_argument(
        "--lin_oracle_impl", choices=("gram", "prefix"), default="gram",
        help="Exact causal linear oracle: quadratic Gram matrix or chunk-checkpointed prefix scan.",
    )
    ap.add_argument("--lin_prefix_chunk_size", type=int, default=64)
    ap.add_argument('--hb_mode', choices=('Y', 'H0', 'H1', 'R2'), default='Y')
    ap.add_argument('--hb_version', default='e046_identity_v1')
    ap.add_argument('--hb_eta_learnable', action=argparse.BooleanOptionalAction, default=False,
                    help='learn one strictly positive connection-strength eta per layer for H1/R2')
    ap.add_argument('--hb_eta_init', type=float, default=1.0,
                    help='initial per-layer connection strength for --hb_eta_learnable')
    ap.add_argument('--matched_attention', choices=('yurii','hb_fixed','hb_free'), default='yurii')
    ap.add_argument('--matched_mlp', choices=('standard','yurii'), default='yurii')
    ap.add_argument('--matched_version', default='e045_displacement_v1')
    ap.add_argument('--matched_h', type=float, default=math.sqrt(.1))
    ap.add_argument('--matched_chunk_size', type=int, default=16)
    ap.add_argument('--matched_checkpoint', action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument('--riem_solver', choices=('gd','hb','cc','r2'), default='cc')
    ap.add_argument('--riem_coefficient_policy', choices=('fixed','shared_r','free_beta','free_gain','free_both','constant'), default='fixed')
    ap.add_argument('--yurii_force_gain_floor', type=float, default=0.)
    ap.add_argument('--riem_oracle_norm', choices=('raw','ln'), default='raw')
    ap.add_argument('--riem_h', type=float, default=math.sqrt(.1), help='Physical step; Geo-GD uses its square as the first-order step.')
    ap.add_argument('--riem_temperature', type=float, default=1.)
    ap.add_argument('--riem_damping_r', type=float, default=3.)
    ap.add_argument('--riem_t0', type=float, default=1.)
    ap.add_argument('--riem_prefix_chunk_size', type=int, default=16)
    ap.add_argument('--riem_checkpoint_chunks', action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument('--riem_initial_ln', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--share_layers",
        action="store_true",
        help="Reuse one block object at every forward-depth step (headline softmax architectures only).",
    )
    ap.add_argument(
        "--scalar_lr_mult",
        type=float,
        default=10.0,
        help="LR multiplier for learned scalar parameters (ConstrainedScalar .raw, theta_h, theta_xi_raw). "
             "Higher values let the scalars update faster and diverge more across layers. Default: 10.0",
    )

    # ---- learned integrator scalars toggles ----
    ap.add_argument("--learn_h", type=int, default=1,
                    help="If 1, learn the integrator step size h (theta_h is trainable). If 0, keep it fixed.")
    ap.add_argument("--learn_xi", type=int, default=1,
                    help="If 1, learn the coupling xi (theta_xi_raw is trainable). If 0, keep it fixed.")


    # Paper-like defaults (TinyStories small)
    ap.add_argument("--n_layer", type=int, default=12)
    ap.add_argument("--n_head", type=int, default=12)
    ap.add_argument("--n_embd", type=int, default=768)
    ap.add_argument("--block_size", type=int, default=1024)
    ap.add_argument("--vocab_size", type=int, default=50304)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--bias", action="store_true", help="paper uses no bias; keep default False")

    # Presymplectic params
    ap.add_argument("--presymp_h", type=float, default=1.0)
    ap.add_argument("--presymp_xi", type=float, default=1.0)
    ap.add_argument("--presymp_t0", type=float, default=1.0)
    ap.add_argument("--eta_mu", type=float, default=None, help="if set (and --eta_learnable is not used), use fixed linear eta(t)=mu*t instead of eta(t)=3*log(t/t0)")
    ap.add_argument("--eta_learnable", action=argparse.BooleanOptionalAction, default=False, help="make eta schedule coefficient(s) learnable; use --no-eta_learnable for a fixed schedule")
    ap.add_argument(
        "--eta_mode",
        type=str,
        default="log",
        choices=["log", "linear", "loglin"],
        help="eta schedule family: log=c_log*log(t/t0); linear=c_lin*t; loglin=c_log*log(t/t0)+c_lin*t",
    )
    # Fixed coefficients (when --eta_learnable is NOT set)
    ap.add_argument("--eta_log_coef", type=float, default=None, help="fixed c_log for log(t/t0) term (eta_mode=log or loglin)")
    ap.add_argument("--eta_lin_coef", type=float, default=None, help="fixed c_lin for t term (eta_mode=linear or loglin). If unset, falls back to --eta_mu for linear part")
    # Learnable initializations (when --eta_learnable is set)
    ap.add_argument("--eta_init", type=float, default=None, help="backward-compatible init: for log/linear; for loglin initializes log coefficient unless --eta_log_init is set")
    ap.add_argument("--eta_log_init", type=float, default=None, help="initial value for learnable c_log (eta_mode=log or loglin)")
    ap.add_argument("--eta_lin_init", type=float, default=None, help="initial value for learnable c_lin (eta_mode=linear or loglin)")
    ap.add_argument("--eta_clip", type=float, default=50.0, help="clamp eta(t) to [-eta_clip, eta_clip] before exponentiation")

    # Presymplectic xi adaptation (data-driven)
    # ap.add_argument("--presymp_xi_adapt", action="store_true", help="adapt xi online based on r_X and r_P thresholds (breaks exact presymplecticity)")
    ap.add_argument("--presymp_r_thresh", type=float, default=1e-2, help="increase xi if max(r_X,r_P) exceeds this")
    ap.add_argument("--presymp_r_low", type=float, default=1e-4, help="decrease xi if max(r_X,r_P) goes below this")
    ap.add_argument("--presymp_xi_mult_up", type=float, default=1.25, help="multiplier when increasing xi")
    ap.add_argument("--presymp_xi_mult_down", type=float, default=0.5, help="multiplier when decreasing xi")
    ap.add_argument("--presymp_xi_min", type=float, default=1e-4, help="lower bound for xi during adaptation")
    ap.add_argument("--presymp_xi_max", type=float, default=100.0, help="upper bound for xi during adaptation (also capped by theta_max/(2h))")
    ap.add_argument("--presymp_theta_max", type=float, default=1.0, help="cap the coupling rotation angle theta=2*xi*h to at most theta_max by enforcing xi<=theta_max/(2h)")
    # ap.add_argument("--presymp_xi_adapt_warmup", type=int, default=10, help="do not adapt xi for the first this many presymp steps")
    # ap.add_argument("--presymp_xi_adapt_every", type=int, default=1, help="update xi every N presymp steps (after warmup)")
    ap.add_argument(
        "--presymp_lookahead",
        action="store_true",
        default=False,
        help=(
            "Evaluate the presymp oracle at X + mu_la*P instead of X "
            "(Nesterov lookahead inside the symplectic step)."
        ),
    )
    ap.add_argument(
        "--presymp_lookahead_init",
        type=float,
        default=0.001,
        help="Initial sigmoid-constrained lookahead coefficient mu_la in (0,1).",
    )
    ap.add_argument("--presymp_lnp", type=str, default="end", choices=["none","end","each_substep"], help="LayerNorm on presymplectic attention momentum P/Pi: none|end|each_substep")

    # Variant A: use attention-induced velocity for the MLP lookahead (drop separate MLP velocity dynamics)
    ap.add_argument(
        "--presymp_mlp_use_attn_vel",
        action="store_true",
        help="Presymp only: use v_attn ≈ (X_after_attn - X_before_attn)/h as the velocity in the MLP lookahead (Variant A).",
    )
    # Variant B: use P (symplectic momentum) as the shared MLP velocity (YuriiFormer Lie-Trotter style)
    ap.add_argument(
        "--presymp_mlp_use_p_vel",
        action="store_true",
        help="Presymp only: use P from the attention step as velocity for the MLP substep (Variant B). "
             "The MLP updates P in-place; updated P flows to the next layer's attention. "
             "Mutually exclusive with --presymp_mlp_use_attn_vel.",
    )
    ap.add_argument(
        "--presymp_mlp_mode",
        type=str,
        default=None,
        choices=["attn_vel", "p_vel", "separate_vel"],
        help="Explicit MLP coupling for presymplectic-family models. Prefer this over the legacy boolean flags in architecture comparisons.",
    )


    # v0 initialization embeddings for momentum variants (YuriiFormer Appendix A.1)
    ap.add_argument(
        "--no_v0_init",
        action="store_true",
        help="disable separate token/position v0 embeddings for momentum variants",
    )
    ap.add_argument(
        "--learned_v0_init",
        action="store_true",
        help="explicitly enable the causal learned token/position v0 embedding initialization for momentum variants",
    )

    # YuriiFormer noise + restart (applied across depth)
    ap.add_argument("--yurii_noise_eta", type=float, default=0.0, help="noise variance scale eta in sigma_t^2 = eta/(1+t)^gamma")
    ap.add_argument("--yurii_noise_gamma", type=float, default=0.55, help="noise decay exponent gamma in sigma_t^2 = eta/(1+t)^gamma")
    ap.add_argument("--yurii_noise_loc", type=str, default="v", choices=["dx", "v", "xin"], help="inject noise into dx, v, or lookahead xin")
    ap.add_argument("--yurii_restart", type=str, default="none", choices=["none", "speed", "loss"], help="restart criterion")
    ap.add_argument("--yurii_restart_min_layer", type=int, default=1, help="start checking restart conditions at this layer index")
    ap.add_argument("--presymp_noise_eta", type=float, default=0.0, help="causal presymplectic momentum-noise variance scale")
    ap.add_argument("--presymp_noise_gamma", type=float, default=0.55, help="decay exponent in eta/(1+global_step)^gamma")

    # Training hyperparams (paper: 10k steps, warmup 1k, peak AdamW LR 6e-4, bf16, clip 1.0)
    ap.add_argument("--max_steps", type=int, default=10_000)
    ap.add_argument("--max_tokens", type=int, default=0, help="If positive, use the largest whole-step token budget not exceeding this cap.")
    ap.add_argument("--global_tokens_per_step", type=int, default=0, help="If positive, derive gradient accumulation from this fixed global token batch.")
    ap.add_argument("--warmup_steps", type=int, default=1_000)
    ap.add_argument("--warmup_ratio", type=float, default=None, help="If set, override warmup_steps by this fraction of max_steps.")
    ap.add_argument("--peak_lr", type=float, default=6e-4)
    ap.add_argument("--min_lr_ratio", type=float, default=0.1)
    ap.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--optimizer", type=str, default="muon_adamw", choices=["muon_adamw", "adamw"])
    ap.add_argument("--muon_lr", type=float, default=0.02)

    # Batch / accumulation
    ap.add_argument("--batch_size", type=int, default=2, help="microbatch size (sequences) per iteration")
    ap.add_argument("--grad_accum_steps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1337)

    # Eval / logging / ckpt
    ap.add_argument("--eval_interval", type=int, default=100)
    ap.add_argument("--eval_batches", type=int, default=40, help="paper uses 160; reduce for speed")
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--out_dir", type=str, default="out")
    ap.add_argument("--run_name", type=str, default="", help="optional suffix for outputs")
    ap.add_argument("--plot", action="store_true", help="save loss-vs-step plot PNG to out_dir")
    ap.add_argument("--resume", type=str, default="", help="path to checkpoint.pt")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--amp_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"], help="CUDA autocast dtype; float32 disables autocast and TF32.")
    ap.add_argument("--eval_only", action="store_true", help="Evaluate --resume on the fixed validation batches and exit.")
    ap.add_argument("--require_data_manifest", action=argparse.BooleanOptionalAction, default=False, help="require and validate preprocessing sidecars for both binary datasets")

    # Text generation / inspection
    ap.add_argument("--sample_interval", type=int, default=0,
                    help="If >0, print and persist a decoded text sample every this many training steps at evaluation.")
    ap.add_argument("--sample_final", action=argparse.BooleanOptionalAction, default=False,
                    help="Persist one final decoded example to samples.jsonl after training.")
    ap.add_argument("--sample_max_new_tokens", type=int, default=128,
                    help="Number of new tokens to generate for each sample.")
    ap.add_argument("--sample_prefix_tokens", type=int, default=64,
                    help="Length of validation-prefix context when --sample_prompt is empty.")
    ap.add_argument("--sample_prompt", type=str, default="",
                    help="Optional text prompt to seed generation. Requires tiktoken.")
    ap.add_argument("--sample_temperature", type=float, default=0.8,
                    help="Sampling temperature. Set <=0 for greedy decoding.")
    ap.add_argument("--sample_top_k", type=int, default=40,
                    help="Top-k truncation for sampling. Use <=0 to disable.")
    ap.add_argument("--sample_do_sample", type=int, default=1,
                    help="1 = sample from the distribution, 0 = greedy argmax.")
    ap.add_argument("--sample_eos_token_id", type=int, default=50256,
                    help="Stop generation once all batch elements emit this token. Use -1 to disable early stop.")

    config_defaults = {}
    if config_args.config:
        with open(config_args.config, "r", encoding="utf-8") as f:
            config_defaults = json.load(f)
        valid_dests = {action.dest for action in ap._actions}
        unknown = sorted(set(config_defaults) - valid_dests)
        if unknown:
            raise SystemExit(f"Unknown keys in {config_args.config}: {unknown}")
        ap.set_defaults(**config_defaults)
    args = ap.parse_args()

    base_tokens_per_micro = args.batch_size * args.block_size
    if args.global_tokens_per_step > 0:
        if args.global_tokens_per_step % base_tokens_per_micro != 0:
            raise SystemExit("global_tokens_per_step must be divisible by batch_size*block_size")
        args.grad_accum_steps = args.global_tokens_per_step // base_tokens_per_micro
    tokens_per_step_config = args.batch_size * args.block_size * args.grad_accum_steps
    if args.max_tokens > 0:
        args.max_steps = args.max_tokens // tokens_per_step_config
        if args.max_steps < 1:
            raise SystemExit("max_tokens must cover at least one optimizer step")
    if args.warmup_ratio is not None:
        if not 0.0 <= args.warmup_ratio < 1.0:
            raise SystemExit("warmup_ratio must lie in [0,1)")
        args.warmup_steps = round(args.warmup_ratio * args.max_steps)
    if not 0.0 < args.presymp_lookahead_init < 1.0:
        raise SystemExit("presymp_lookahead_init must lie strictly between 0 and 1")

    presymp_softmax_arches = {"causal_symp_fe", "causal_symp_pe", "causal_symp_exp_pe", "causal_symp_halfdamp_pe", "causal_symp_ab2", "presymp", "presymp_euler", "presymp_exp_euler", "presymp_ab2", "presymp_etd_ab2", "presymp_strang", "plain_euler"}
    if args.no_v0_init and args.learned_v0_init:
        raise SystemExit("--no_v0_init and --learned_v0_init are mutually exclusive")
    explicit_options = {arg.split('=')[0] for arg in sys.argv[1:] if arg.startswith('--')}
    provided = set(config_defaults) | {action.dest for action in ap._actions
                                       if explicit_options.intersection(action.option_strings)}
    if args.arch != 'riem_prefix' and any(key.startswith('riem_') for key in provided):
        raise SystemExit('riem_* options are only valid for --arch riem_prefix')
    if args.arch == 'riem_prefix':
        unsupported = sorted(key for key in provided if key.startswith(('presymp_', 'eta_', 'lin_', 'noise_', 'restart_'))
                             or key in ('learn_h', 'learn_xi'))
        if unsupported:
            raise SystemExit(f'Legacy options do not configure riem_prefix: {unsupported}')
        if args.n_head != 1 or args.dropout != 0 or args.amp_dtype != 'float32':
            raise SystemExit('riem_prefix requires n_head=1, dropout=0 and amp_dtype=float32')
        if args.presymp_lookahead or args.eta_learnable or args.presymp_mlp_mode is not None:
            raise SystemExit('Legacy initialization/lookahead/damping/MLP flags do not configure riem_prefix')
        args.no_v0_init = not args.learned_v0_init
    if args.arch != 'matched_prefix' and any(key.startswith('matched_') for key in provided):
        raise SystemExit('matched_* options require matched_prefix')
    if args.arch == 'matched_prefix':
        unsupported=sorted(k for k in provided if k.startswith(('riem_','presymp_','eta_','lin_')) or k in ('learn_h','learn_xi'))
        if unsupported:raise SystemExit(f'Legacy options do not configure matched_prefix: {unsupported}')
        if args.n_head!=4 or args.dropout!=0 or args.amp_dtype!='float32' or args.no_mlp or not args.learned_v0_init:
            raise SystemExit('matched_prefix requires4 heads,dropout0,FP32,MLP and learned displacement-velocity tables')
        if args.yurii_force_gain_floor!=1e-6 or args.yurii_noise_eta!=0 or args.yurii_restart!='none':
            raise SystemExit('matched_prefix requires author floor1e-6 and no augmentation/restarts')
    if args.arch != 'normalized_hb' and any(k.startswith('hb_') for k in provided):
        raise SystemExit('hb_* options require normalized_hb')
    if args.arch == 'normalized_hb':
        unsupported = sorted(k for k in provided if k.startswith(('riem_', 'matched_', 'presymp_', 'eta_', 'lin_')) or k in ('learn_h', 'learn_xi'))
        if unsupported:
            raise SystemExit(f'Legacy options do not configure normalized_hb: {unsupported}')
        if args.dropout != 0 or args.bias or args.amp_dtype != 'float32' or args.no_mlp or not args.learned_v0_init:
            raise SystemExit('normalized_hb requires dropout0,bias=False,FP32,MLP and learned velocity')
        if args.yurii_force_gain_floor != 1e-6 or args.yurii_noise_eta != 0 or args.yurii_restart != 'none':
            raise SystemExit('normalized_hb requires floor1e-6 and no augmentation/restarts')
        NormalizedHBConfig(mode=args.hb_mode, version=args.hb_version, eta_learnable=args.hb_eta_learnable, eta_init=args.hb_eta_init)
    if args.yurii_force_gain_floor != 0 and args.arch not in ('yurii_lt','matched_prefix','normalized_hb'):
        raise SystemExit('yurii_force_gain_floor requires a supported Yurii-derived architecture')
    if not 0 <= args.yurii_force_gain_floor < 1 or not math.isfinite(args.yurii_force_gain_floor):
        raise SystemExit('yurii_force_gain_floor must be finite in [0,1)')
    if args.share_layers and args.arch not in {"baseline", "baseline_capacity", "yurii_lt", "causal_symp_pe"}:
        raise SystemExit("--share_layers is implemented only for headline softmax architectures")
    # Preserve all previous causal-SympFormer specifications: they remain at
    # zero momentum unless learned v0 is requested explicitly.  The learned
    # token/position stream is prefix-local and is covered by the exhaustive
    # suffix-causality verifier.
    if args.arch in presymp_softmax_arches and not args.learned_v0_init:
        args.no_v0_init = True

    if args.presymp_mlp_mode is not None:
        if args.presymp_mlp_use_attn_vel or args.presymp_mlp_use_p_vel:
            raise SystemExit("--presymp_mlp_mode is mutually exclusive with the legacy MLP velocity flags")
        args.presymp_mlp_use_attn_vel = args.presymp_mlp_mode == "attn_vel"
        args.presymp_mlp_use_p_vel = args.presymp_mlp_mode == "p_vel"
    elif args.arch in presymp_softmax_arches and not args.presymp_mlp_use_attn_vel and not args.presymp_mlp_use_p_vel:
        # Backward-compatible default. Architecture comparisons must use the
        # explicit --presymp_mlp_mode flag so separate_vel is representable.
        args.presymp_mlp_use_attn_vel = True

    # Use per-run directory to avoid collisions when running multiple arch variants.
    run_dir = os.path.join(args.out_dir, args.arch)
    if args.run_name:
        run_dir = os.path.join(args.out_dir, f"{args.arch}_{args.run_name}")
    if args.arch in ('riem_prefix','matched_prefix','normalized_hb') and not args.resume and os.path.exists(os.path.join(run_dir, 'run_manifest.json')):
        raise SystemExit(f'Refusing to overwrite existing riem_prefix run: {run_dir}')
    os.makedirs(run_dir, exist_ok=True)

    # Seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")
    amp_dtype = configure_precision(args.amp_dtype, device)

    # Load data
    train_path = os.path.join(args.data_dir, f"{args.dataset}_train.bin")
    val_path = os.path.join(args.data_dir, f"{args.dataset}_val.bin")
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise SystemExit(f"Missing dataset bins: {train_path} / {val_path}")

    train_tokens = load_bin(train_path)
    val_tokens = load_bin(val_path)
    train_data_info = data_record(train_path, train_tokens, args.require_data_manifest)
    val_data_info = data_record(val_path, val_tokens, args.require_data_manifest)

    dcfg = DataConfig(block_size=args.block_size, batch_size=args.batch_size, grad_accum_steps=args.grad_accum_steps, seed=args.seed, device=device)
    train_it = SharedTargetIterator(BlockEpochIterator, train_tokens, dcfg, split="train")
    val_it = SharedTargetIterator(BlockEpochIterator, val_tokens, dcfg, split="val")

    # Build model
    mcfg = ModelConfig(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        bias=args.bias,
        lin_oracle_impl=args.lin_oracle_impl,
        lin_prefix_chunk_size=args.lin_prefix_chunk_size,
        yurii_force_gain_floor=args.yurii_force_gain_floor,
    )

    if args.arch == 'normalized_hb':
        model = NormalizedHBModel(mcfg, NormalizedHBConfig(mode=args.hb_mode, version=args.hb_version, eta_learnable=args.hb_eta_learnable, eta_init=args.hb_eta_init))
    elif args.arch == 'matched_prefix':
        model=MatchedPrefixModel(mcfg,PrefixGeometryConfig(initial_ln=False,learned_v0=True,
            h=args.matched_h,prefix_chunk_size=args.matched_chunk_size,checkpoint_chunks=args.matched_checkpoint,
            coefficient_policy='free_both' if args.matched_attention=='hb_free' else 'fixed'),
            MatchedPrefixConfig(attention=args.matched_attention,mlp=args.matched_mlp,version=args.matched_version))
    elif args.arch == 'riem_prefix':
        model = RiemannianPrefixModel(mcfg, PrefixGeometryConfig(
            solver=args.riem_solver, oracle_norm=args.riem_oracle_norm,
            h=args.riem_h, temperature=args.riem_temperature,
            damping_r=args.riem_damping_r, t0=args.riem_t0,
            prefix_chunk_size=args.riem_prefix_chunk_size,
            checkpoint_chunks=args.riem_checkpoint_chunks,
            initial_ln=args.riem_initial_ln, no_mlp=args.no_mlp,
            learned_v0=args.learned_v0_init, coefficient_policy=args.riem_coefficient_policy,
        ))
    elif args.arch == "baseline":
        model = GPTModel(mcfg, no_mlp=args.no_mlp, share_layers=args.share_layers)
    elif args.arch == "baseline_capacity":
        model = GPTModel(mcfg, no_mlp=args.no_mlp, capacity_control=True, share_layers=args.share_layers)
    elif args.arch == "yurii_lt":
        model = YuriiFormerModel(
            mcfg,
            use_v0_init=(not args.no_v0_init),
            noise_eta=args.yurii_noise_eta,
            noise_gamma=args.yurii_noise_gamma,
            noise_loc=args.yurii_noise_loc,
            restart_mode=args.yurii_restart,
            restart_min_layer=args.yurii_restart_min_layer,
            no_mlp=args.no_mlp,
            share_layers=args.share_layers,
        )
    else:
        # Presymp family: same overall architecture, different attention discretization
        if args.arch in {"causal_symp_fe", "causal_symp_pe", "causal_symp_exp_pe", "causal_symp_halfdamp_pe", "causal_symp_ab2"}:
            causal_scheme = {
                "causal_symp_fe": "causal_fe",
                "causal_symp_pe": "causal_pe",
                "causal_symp_exp_pe": "causal_exp_pe",
                "causal_symp_halfdamp_pe": "causal_halfdamp_pe",
                "causal_symp_ab2": "causal_ab2",
            }[args.arch]
            model = PresympModel(
                mcfg,
                attn_scheme=causal_scheme,
                h=args.presymp_h,
                xi=args.presymp_xi,
                t0=args.presymp_t0,
                eta_mu=args.eta_mu,
                eta_log_coef=args.eta_log_coef,
                eta_lin_coef=args.eta_lin_coef,
                eta_log_init=args.eta_log_init,
                eta_lin_init=args.eta_lin_init,
                eta_learnable=args.eta_learnable,
                eta_mode=args.eta_mode,
                eta_init=args.eta_init,
                eta_clip=args.eta_clip,
                use_v0_init=(not args.no_v0_init),
                presymp_lnp=args.presymp_lnp,
                mlp_use_attn_vel=args.presymp_mlp_use_attn_vel,
                mlp_use_p_vel=args.presymp_mlp_use_p_vel,
                no_mlp=args.no_mlp,
                lookahead=args.presymp_lookahead,
                lookahead_init=args.presymp_lookahead_init,
                noise_eta=args.presymp_noise_eta,
                noise_gamma=args.presymp_noise_gamma,
                share_layers=args.share_layers,
            )
        elif args.arch == "presymp":
            model = PresympModel(
                mcfg,
                attn_scheme="presymp",
                h=args.presymp_h,
                xi=args.presymp_xi,
                t0=args.presymp_t0,
                eta_mu=args.eta_mu,
                eta_log_coef=args.eta_log_coef,
                eta_lin_coef=args.eta_lin_coef,
                eta_log_init=args.eta_log_init,
                eta_lin_init=args.eta_lin_init,
                eta_learnable=args.eta_learnable,
                eta_mode=args.eta_mode,
                eta_init=args.eta_init,
                eta_clip=args.eta_clip,
                use_v0_init=(not args.no_v0_init),
                # xi_adapt=args.presymp_xi_adapt,
                r_thresh=args.presymp_r_thresh,
                r_low=args.presymp_r_low,
                xi_mult_up=args.presymp_xi_mult_up,
                xi_mult_down=args.presymp_xi_mult_down,
                xi_min=args.presymp_xi_min,
                xi_max=args.presymp_xi_max,
                theta_max=args.presymp_theta_max,
                presymp_lnp=args.presymp_lnp,
                # xi_adapt_warmup=args.presymp_xi_adapt_warmup,
                # xi_adapt_every=args.presymp_xi_adapt_every,
                mlp_use_attn_vel=args.presymp_mlp_use_attn_vel,
                mlp_use_p_vel=args.presymp_mlp_use_p_vel,
                no_mlp=args.no_mlp,
                lookahead=args.presymp_lookahead,
            )
        elif args.arch == "presymp_euler":
            model = PresympModel(
                mcfg,
                attn_scheme="euler",
                h=args.presymp_h,
                xi=args.presymp_xi,
                t0=args.presymp_t0,
                eta_mu=args.eta_mu,
                eta_log_coef=args.eta_log_coef,
                eta_lin_coef=args.eta_lin_coef,
                eta_log_init=args.eta_log_init,
                eta_lin_init=args.eta_lin_init,
                eta_learnable=args.eta_learnable,
                eta_mode=args.eta_mode,
                eta_init=args.eta_init,
                eta_clip=args.eta_clip,
                use_v0_init=(not args.no_v0_init),
                # xi_adapt=args.presymp_xi_adapt,
                r_thresh=args.presymp_r_thresh,
                r_low=args.presymp_r_low,
                xi_mult_up=args.presymp_xi_mult_up,
                xi_mult_down=args.presymp_xi_mult_down,
                xi_min=args.presymp_xi_min,
                xi_max=args.presymp_xi_max,
                theta_max=args.presymp_theta_max,
                presymp_lnp=args.presymp_lnp,
                # xi_adapt_warmup=args.presymp_xi_adapt_warmup,
                # xi_adapt_every=args.presymp_xi_adapt_every,
                mlp_use_attn_vel=args.presymp_mlp_use_attn_vel,
                mlp_use_p_vel=args.presymp_mlp_use_p_vel,
                no_mlp=args.no_mlp,
                lookahead=args.presymp_lookahead,
            )
        elif args.arch == "presymp_exp_euler":
            model = PresympModel(
                mcfg,
                attn_scheme="exp_euler",
                h=args.presymp_h,
                xi=args.presymp_xi,
                t0=args.presymp_t0,
                eta_mu=args.eta_mu,
                eta_log_coef=args.eta_log_coef,
                eta_lin_coef=args.eta_lin_coef,
                eta_log_init=args.eta_log_init,
                eta_lin_init=args.eta_lin_init,
                eta_learnable=args.eta_learnable,
                eta_mode=args.eta_mode,
                eta_init=args.eta_init,
                eta_clip=args.eta_clip,
                use_v0_init=(not args.no_v0_init),
                # xi_adapt=args.presymp_xi_adapt,
                r_thresh=args.presymp_r_thresh,
                r_low=args.presymp_r_low,
                xi_mult_up=args.presymp_xi_mult_up,
                xi_mult_down=args.presymp_xi_mult_down,
                xi_min=args.presymp_xi_min,
                xi_max=args.presymp_xi_max,
                theta_max=args.presymp_theta_max,
                presymp_lnp=args.presymp_lnp,
                # xi_adapt_warmup=args.presymp_xi_adapt_warmup,
                # xi_adapt_every=args.presymp_xi_adapt_every,
                mlp_use_attn_vel=args.presymp_mlp_use_attn_vel,
                mlp_use_p_vel=args.presymp_mlp_use_p_vel,
                no_mlp=args.no_mlp,
                lookahead=args.presymp_lookahead,
            )
        elif args.arch == "presymp_ab2":
            model = PresympModelAB2(
                mcfg,
                h=args.presymp_h,
                t0=args.presymp_t0,
                eta_mu=args.eta_mu,
                eta_log_coef=args.eta_log_coef,
                eta_lin_coef=args.eta_lin_coef,
                eta_log_init=args.eta_log_init,
                eta_lin_init=args.eta_lin_init,
                eta_learnable=args.eta_learnable,
                eta_mode=args.eta_mode,
                eta_init=args.eta_init,
                eta_clip=args.eta_clip,
                presymp_lnp=args.presymp_lnp,
                use_v0_init=(not args.no_v0_init),
                mlp_use_attn_vel=args.presymp_mlp_use_attn_vel,
                mlp_use_p_vel=args.presymp_mlp_use_p_vel,
                no_mlp=args.no_mlp,
                lookahead=args.presymp_lookahead,
            )
        elif args.arch == "presymp_etd_ab2":
            model = PresympModelETDAB2(
                mcfg,
                h=args.presymp_h,
                t0=args.presymp_t0,
                eta_mu=args.eta_mu,
                eta_log_coef=args.eta_log_coef,
                eta_lin_coef=args.eta_lin_coef,
                eta_log_init=args.eta_log_init,
                eta_lin_init=args.eta_lin_init,
                eta_learnable=args.eta_learnable,
                eta_mode=args.eta_mode,
                eta_init=args.eta_init,
                eta_clip=args.eta_clip,
                presymp_lnp=args.presymp_lnp,
                use_v0_init=(not args.no_v0_init),
                mlp_use_attn_vel=args.presymp_mlp_use_attn_vel,
                mlp_use_p_vel=args.presymp_mlp_use_p_vel,
                no_mlp=args.no_mlp,
                lookahead=args.presymp_lookahead,
            )
        elif args.arch == "presymp_strang":
            model = PresympModel(
                mcfg,
                attn_scheme="strang",
                h=args.presymp_h,
                xi=args.presymp_xi,
                t0=args.presymp_t0,
                eta_mu=args.eta_mu,
                eta_log_coef=args.eta_log_coef,
                eta_lin_coef=args.eta_lin_coef,
                eta_log_init=args.eta_log_init,
                eta_lin_init=args.eta_lin_init,
                eta_learnable=args.eta_learnable,
                eta_mode=args.eta_mode,
                eta_init=args.eta_init,
                eta_clip=args.eta_clip,
                use_v0_init=(not args.no_v0_init),
                # xi_adapt=args.presymp_xi_adapt,
                r_thresh=args.presymp_r_thresh,
                r_low=args.presymp_r_low,
                xi_mult_up=args.presymp_xi_mult_up,
                xi_mult_down=args.presymp_xi_mult_down,
                xi_min=args.presymp_xi_min,
                xi_max=args.presymp_xi_max,
                theta_max=args.presymp_theta_max,
                presymp_lnp=args.presymp_lnp,
                # xi_adapt_warmup=args.presymp_xi_adapt_warmup,
                # xi_adapt_every=args.presymp_xi_adapt_every,
                mlp_use_attn_vel=args.presymp_mlp_use_attn_vel,
                mlp_use_p_vel=args.presymp_mlp_use_p_vel,
                no_mlp=args.no_mlp,
                lookahead=args.presymp_lookahead,
            )

        elif args.arch == "plain_euler":
            model = PresympModel(
                mcfg,
                attn_scheme="plain_euler",
                h=args.presymp_h,
                xi=args.presymp_xi,
                t0=args.presymp_t0,
                eta_mu=args.eta_mu,
                eta_log_coef=args.eta_log_coef,
                eta_lin_coef=args.eta_lin_coef,
                eta_log_init=args.eta_log_init,
                eta_lin_init=args.eta_lin_init,
                eta_learnable=args.eta_learnable,
                eta_mode=args.eta_mode,
                eta_init=args.eta_init,
                eta_clip=args.eta_clip,
                use_v0_init=(not args.no_v0_init),
                r_thresh=args.presymp_r_thresh,
                r_low=args.presymp_r_low,
                xi_mult_up=args.presymp_xi_mult_up,
                xi_mult_down=args.presymp_xi_mult_down,
                xi_min=args.presymp_xi_min,
                xi_max=args.presymp_xi_max,
                theta_max=args.presymp_theta_max,
                presymp_lnp=args.presymp_lnp,
                mlp_use_attn_vel=args.presymp_mlp_use_attn_vel,
                mlp_use_p_vel=args.presymp_mlp_use_p_vel,
                no_mlp=args.no_mlp,
                lookahead=args.presymp_lookahead,
            )
        elif args.arch in ("causal_riem_nag_noconn", "causal_riem_nag"):
            model = CausalRiemannianNAGModel(
                mcfg,
                h=args.presymp_h,
                t0=args.presymp_t0,
                eta_log_coef=args.eta_log_coef if args.eta_log_coef is not None else 3.0,
                eta_lin_coef=args.eta_lin_coef if args.eta_lin_coef is not None else 0.0,
                eta_learnable=args.eta_learnable,
                eta_clip=args.eta_clip,
                include_connection=(args.arch == "causal_riem_nag"),
                no_mlp=args.no_mlp,
            )
        elif args.arch == "lin_baseline":
            model = LinAttnModel(
                mcfg,
                h=args.presymp_h,
                no_mlp=args.no_mlp,
                mlp_mode="standard" if args.lin_mlp_mode == "auto" else args.lin_mlp_mode,
            )
        elif args.arch == "lin_yurii":
            model = LinAttnYuriiModel(
                mcfg,
                h=args.presymp_h,
                no_mlp=args.no_mlp,
                use_v0_init=(not args.no_v0_init),
            )
        elif args.arch == "lin_euler":
            model = LinAttnEulerModel(
                mcfg,
                h=args.presymp_h,
                alpha_init=0.9,
                presymp_lnp=args.presymp_lnp,
                use_v0_init=(not args.no_v0_init),
                no_mlp=args.no_mlp,
                mlp_mode="momentum" if args.lin_mlp_mode == "auto" else args.lin_mlp_mode,
            )
        elif args.arch == "lin_presymp":
            model = LinAttnPresympModel(
                mcfg,
                h=args.presymp_h,
                t0=args.presymp_t0,
                eta_mu=args.eta_mu,
                eta_log_coef=args.eta_log_coef,
                eta_lin_coef=args.eta_lin_coef,
                eta_log_init=args.eta_log_init,
                eta_lin_init=args.eta_lin_init,
                eta_learnable=args.eta_learnable,
                eta_mode=args.eta_mode,
                eta_init=args.eta_init,
                eta_clip=args.eta_clip,
                presymp_lnp=args.presymp_lnp,
                use_v0_init=(not args.no_v0_init),
                no_mlp=args.no_mlp,
                mlp_mode="momentum" if args.lin_mlp_mode == "auto" else args.lin_mlp_mode,
            )
        elif args.arch == "lin_exp_euler":
            # Presymplectic exponential Euler for the linear-attention Hamiltonian.
            # Like lin_presymp but position update uses old Y^k (not updated Y^{k+1}).
            model = LinAttnPresympModel(
                mcfg,
                h=args.presymp_h,
                t0=args.presymp_t0,
                eta_mu=args.eta_mu,
                eta_log_coef=args.eta_log_coef,
                eta_lin_coef=args.eta_lin_coef,
                eta_log_init=args.eta_log_init,
                eta_lin_init=args.eta_lin_init,
                eta_learnable=args.eta_learnable,
                eta_mode=args.eta_mode,
                eta_init=args.eta_init,
                eta_clip=args.eta_clip,
                presymp_lnp=args.presymp_lnp,
                use_v0_init=(not args.no_v0_init),
                no_mlp=args.no_mlp,
                attn_cls="exp_euler",
                mlp_mode="momentum" if args.lin_mlp_mode == "auto" else args.lin_mlp_mode,
            )
        elif args.arch == "lin_ab2":
            model = LinAttnAB2Model(
                mcfg,
                h=args.presymp_h,
                t0=args.presymp_t0,
                eta_mu=args.eta_mu,
                eta_log_coef=args.eta_log_coef,
                eta_lin_coef=args.eta_lin_coef,
                eta_log_init=args.eta_log_init,
                eta_lin_init=args.eta_lin_init,
                eta_learnable=args.eta_learnable,
                eta_mode=args.eta_mode,
                eta_init=args.eta_init,
                eta_clip=args.eta_clip,
                presymp_lnp=args.presymp_lnp,
                use_v0_init=(not args.no_v0_init),
                no_mlp=args.no_mlp,
                mlp_mode="momentum" if args.lin_mlp_mode == "auto" else args.lin_mlp_mode,
            )
        elif args.arch == "lin_etd_ab2":
            model = LinAttnETDAB2Model(
                mcfg,
                h=args.presymp_h,
                t0=args.presymp_t0,
                eta_mu=args.eta_mu,
                eta_log_coef=args.eta_log_coef,
                eta_lin_coef=args.eta_lin_coef,
                eta_log_init=args.eta_log_init,
                eta_lin_init=args.eta_lin_init,
                eta_learnable=args.eta_learnable,
                eta_mode=args.eta_mode,
                eta_init=args.eta_init,
                eta_clip=args.eta_clip,
                presymp_lnp=args.presymp_lnp,
                use_v0_init=(not args.no_v0_init),
                no_mlp=args.no_mlp,
            )
        elif args.arch in ("lin_reduced_exp_mid", "lin_reduced_ab2"):
            model = LinAttnReducedModel(
                mcfg,
                scheme="exp_mid" if args.arch == "lin_reduced_exp_mid" else "ab2",
                h=args.presymp_h,
                t0=args.presymp_t0,
                eta_mu=args.eta_mu,
                eta_log_coef=args.eta_log_coef,
                eta_lin_coef=args.eta_lin_coef,
                eta_log_init=args.eta_log_init,
                eta_lin_init=args.eta_lin_init,
                eta_learnable=args.eta_learnable,
                eta_mode=args.eta_mode,
                eta_init=args.eta_init,
                eta_clip=args.eta_clip,
                no_mlp=args.no_mlp,
                noncausal=False,
            )
        else:
            raise ValueError(f"Unknown arch: {args.arch}")

    if args.learned_v0_init:
        initialize_learned_v0_tables(model, args.seed)

    if args.lin_noncausal:
        raise ValueError(
            "--lin_noncausal is disabled for decoder training: every linear-attention CLI route is strictly causal"
        )

    model.to(device)
    # Optionally freeze learned integrator scalars (h, xi) so they stay fixed.
    if args.learn_h == 0:
        for n, p in model.named_parameters():
            if "theta_h" in n or "theta_hX" in n or "theta_hY" in n or "theta_tau" in n:
                p.requires_grad_(False)
    if args.learn_xi == 0:
        for n, p in model.named_parameters():
            if "theta_xi_raw" in n:
                p.requires_grad_(False)
        # disable heuristic xi adaptation if present
        # if hasattr(args, "presymp_xi_adapt"):
        #     args.presymp_xi_adapt = False


    opt = build_optimizer(
        model,
        peak_lr=args.peak_lr,
        betas=tuple(args.betas),
        scalar_lr_mult=args.scalar_lr_mult,
        optimizer_name=args.optimizer,
        muon_lr=args.muon_lr,
    )

    prefix_resume_ckpt = None
    if args.resume and isinstance(model, NormalizedHBModel):
        prefix_resume_ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        old_args = prefix_resume_ckpt.get('args', {})
        for key in ('arch','hb_mode','hb_version','seed','learned_v0_init','no_v0_init','no_mlp',
                    'n_layer','n_head','n_embd','block_size','vocab_size','bias','dropout','amp_dtype',
                    'batch_size','grad_accum_steps','optimizer','peak_lr','scalar_lr_mult','betas',
                    'warmup_steps','max_steps','grad_clip','min_lr_ratio','eval_interval','eval_batches',
                    'dataset','yurii_force_gain_floor','yurii_noise_eta','yurii_restart'):
            if old_args.get(key) != getattr(args, key):
                raise ValueError(f'Incompatible normalized_hb resume setting: {key}')
        for key, default in (('hb_eta_learnable', False), ('hb_eta_init', 1.0)):
            if old_args.get(key, default) != getattr(args, key):
                raise ValueError(f'Incompatible normalized_hb resume setting: {key}')
        if prefix_resume_ckpt.get('hb_work_by_layer') is None:
            raise ValueError('Missing normalized HB work accounting')
    elif args.resume and isinstance(model, RiemannianPrefixModel):
        # Validate before writing over any existing provenance. Trusted local checkpoints only.
        prefix_resume_ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        old_args = prefix_resume_ckpt.get('args', {})
        if isinstance(model,MatchedPrefixModel):
            for key in ('matched_attention','matched_mlp','matched_version','matched_h','matched_chunk_size','matched_checkpoint','yurii_force_gain_floor','seed'):
                if old_args.get(key)!=getattr(args,key):raise ValueError(f'Incompatible matched_prefix resume setting: {key}')
        for key,default in (('riem_coefficient_policy','fixed'),('learned_v0_init',False)):
            if old_args.get(key,default) != getattr(args,key):
                raise ValueError(f'Incompatible riem_prefix resume setting: {key}')
        for key in ('arch', 'riem_solver', 'riem_oracle_norm', 'riem_h', 'riem_temperature',
                    'riem_damping_r', 'riem_t0', 'riem_initial_ln', 'no_mlp', 'n_layer',
                    'n_embd', 'n_head', 'block_size', 'vocab_size', 'bias', 'amp_dtype',
                    'batch_size', 'grad_accum_steps', 'optimizer', 'peak_lr', 'scalar_lr_mult',
                    'betas', 'warmup_steps', 'max_steps', 'grad_clip', 'min_lr_ratio'):
            if old_args.get(key) != getattr(args, key):
                raise ValueError(f'Incompatible riem_prefix resume setting: {key}')
        if prefix_resume_ckpt.get('geometry_work_by_layer') is None:
            raise ValueError('Prefix resume checkpoint is missing work accounting')
    elif args.resume and args.arch == 'yurii_lt':
        prefix_resume_ckpt=torch.load(args.resume,map_location='cpu',weights_only=False)
        if prefix_resume_ckpt.get('args',{}).get('yurii_force_gain_floor',0.) != args.yurii_force_gain_floor:
            raise ValueError('Incompatible YuriiFormer resume force-gain floor')
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    names_by_id = {id(param): name for name, param in model.named_parameters()}
    manifest = {
        "args": vars(args),
        "optimizer_groups": [dict(parameters=[names_by_id[id(p)] for p in group['params']],
                                  lr=group['lr'], weight_decay=group['weight_decay'],
                                  betas=list(group.get('betas', ()))) for group in opt.param_groups],
        "model_config": asdict(mcfg),
        "geometry_config": asdict(model.geo) if isinstance(model, RiemannianPrefixModel) else None,
        "matched_config": asdict(model.matched) if isinstance(model,MatchedPrefixModel) else None,
        "hb_config": asdict(model.hb) if isinstance(model, NormalizedHBModel) else None,
        "trainable_parameters": trainable_params,
        "tokens_per_step": tokens_per_step_config,
        "data_protocol": dict(schema="shared_macro512_v1", macro_length=512),
        "source_revision": source_revision(),
        "source_fingerprint": source_fingerprint(args.config),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
        "target_max_tokens": args.max_tokens if args.max_tokens > 0 else None,
        "actual_max_tokens": args.max_steps * tokens_per_step_config,
        "precision": {"autocast": device.startswith("cuda") and amp_dtype != torch.float32,
                      "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                      "cudnn_tf32": torch.backends.cudnn.allow_tf32},
        "train_data": train_data_info,
        "val_data": val_data_info,
    }
    if not (isinstance(model, (RiemannianPrefixModel, NormalizedHBModel)) and args.eval_only):
        with open(os.path.join(run_dir, "run_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)

    start_step = 0
    best_val = float("inf")
    prior_wall_cum = 0.0

    if args.resume:
        # Resume checkpoints are trusted local training artifacts and include
        # Python/NumPy RNG state, which is intentionally outside the restricted
        # weights-only unpickler. Never use --resume on an untrusted file.
        ckpt = prefix_resume_ckpt if prefix_resume_ckpt is not None else torch.load(args.resume, map_location="cpu", weights_only=False)
        if isinstance(model, NormalizedHBModel):
            counts = ckpt['hb_work_by_layer']
            if len(counts) != len(model.blocks):
                raise ValueError('Normalized HB work-accounting layer mismatch')
            for block, work in zip(model.blocks, counts):
                block.work_counts.clear()
                block.work_counts.update(work)
            model.evaluation_forward_calls = ckpt['evaluation_forward_calls']
        if isinstance(model, RiemannianPrefixModel):
            counts = ckpt['geometry_work_by_layer']
            if len(counts) != len(model.layers):
                raise ValueError('Prefix resume work-accounting layer count mismatch')
            for layer, work in zip(model.layers, counts):
                # Preserve aliases used by nested attention adapters.
                layer.work_counts.clear()
                layer.work_counts.update(work)
            model.evaluation_forward_calls = ckpt['evaluation_forward_calls']
        model.load_state_dict(ckpt["model"])
        model.evaluation_forward_calls = ckpt["evaluation_forward_calls"]
        if not args.eval_only:
            opt.load_state_dict(ckpt["opt"])
        start_step = ckpt.get("next_step", ckpt.get("step", 0))
        best_val = ckpt.get("best_val", float("inf"))
        prior_wall_cum = float(ckpt.get("wall_cum_s", 0.0))
        if "train_iterator" in ckpt:
            train_it.load_state_dict(ckpt["train_iterator"])
        if "val_iterator" in ckpt:
            val_it.load_state_dict(ckpt["val_iterator"])
        if "rng_state" in ckpt:
            restore_rng_state(ckpt["rng_state"])
        else:
            warnings.warn("Legacy checkpoint has no RNG/iterator state; exact resume is impossible.")
        print(f"Resumed from {args.resume} at step {start_step}, best_val={best_val}")

    if args.eval_only:
        if not args.resume:
            raise SystemExit("--eval_only requires --resume")
        val_loss = estimate_loss(model, val_it, device, args.eval_batches, amp_dtype, global_step=start_step)
        print(json.dumps({"checkpoint": args.resume, "step": start_step, "val_loss": val_loss}, sort_keys=True))
        if args.sample_final:
            print_sample(model, val_tokens, device, args, global_step=start_step, run_dir=run_dir, force=True)
        return

    metrics_path = os.path.join(run_dir, "metrics.csv")
    dynamics_path = os.path.join(run_dir, "dynamics.csv")
    timing_path = os.path.join(run_dir, "timing.csv")
    ensure_csv_header(timing_path, ["step", "phase", "seconds", "forward_calls"])
    ensure_csv_header(dynamics_path, DYNAMICS_FIELDS)
    # wall_dt_s: time since previous log print (train rows)
    # wall_cum_s: cumulative wall time since start of run
    # tokens_step: tokens processed per optimizer step
    # tokens_cum: cumulative tokens processed since step 0
    # sched_t_start / sched_t_end: cumulative schedule clock reported by the model (when available)
    ensure_csv_header(
        metrics_path,
        ["step", "train_loss", "val_loss", "lr", "wall_dt_s", "wall_cum_s", "tokens_step", "tokens_cum", "h_mean", "hY_mean", "xi_mean", "rX", "rP", "c_log_mean", "c_lin_mean", "leak_warnings", "sched_t_start", "sched_t_end"],
    )
    plot_path = os.path.join(run_dir, "loss.png")

    # Training loop
    model.train()
    t0_wall = time.time()          # for wall_dt_s
    t_start = time.time() - prior_wall_cum  # preserve cumulative wall time across resume
    def abort_nonfinite(error, lr):
        failure_path, checkpoint_path = write_nonfinite_failure(
            run_dir,
            error,
            model=model,
            opt=opt,
            best_val=best_val,
            mcfg=mcfg,
            args=args,
            train_it=train_it,
            val_it=val_it,
            wall_cum_s=time.time() - t_start,
            tokens_per_step=tokens_per_step_config,
            lr=lr,
        )
        print(f"[fatal] {error}", flush=True)
        print(f"[fatal] diagnostic={failure_path}", flush=True)
        if checkpoint_path:
            print(
                f"[fatal] forensic_checkpoint={checkpoint_path} (not resumable)",
                flush=True,
            )
        raise error
    training_forward_calls = backward_calls = (start_step * args.grad_accum_steps if isinstance(model, (RiemannianPrefixModel, NormalizedHBModel)) else 0)
    for step in range(start_step, args.max_steps):
        # update learning rates
        lr = cosine_lr(step, args.warmup_steps, args.max_steps, args.peak_lr, args.min_lr_ratio)
        for pg in opt.param_groups:
            mult = pg.get("lr_mult", 1.0)
            pg["lr"] = lr * mult

        opt.zero_grad(set_to_none=True)

        loss_accum = 0.0
        restarts_accum = 0
        for micro in range(args.grad_accum_steps):
            xb, yb = next(train_it)
            xb = xb.to(device)
            yb = yb.to(device)

            set_geometry_capture(model, step % args.log_interval == 0 and micro == args.grad_accum_steps - 1)
            with torch.autocast(device_type=device.split(':')[0], dtype=amp_dtype, enabled=(device.startswith("cuda") and amp_dtype != torch.float32)):
                _, loss = model(xb, yb, global_step=step)
                training_forward_calls += 1
                loss = loss / args.grad_accum_steps
            try:
                loss_value = require_finite_scalar(
                    loss,
                    kind="training_loss",
                    step=step,
                    micro_step=micro,
                )
            except NonFiniteTrainingError as error:
                abort_nonfinite(error, lr)
            loss.backward()
            backward_calls += 1
            loss_accum += loss_value
            if hasattr(model, "last_restart_count"):
                restarts_accum += int(getattr(model, "last_restart_count", 0))

        # Check the total norm even when clipping is disabled. This catches
        # nonfinite gradients before they can contaminate model parameters.
        clip_params = [p for p in model.parameters() if p.requires_grad]
        try:
            clip_grad_norm_finite(clip_params, args.grad_clip, step=step)
        except NonFiniteTrainingError as error:
            abort_nonfinite(error, lr)
        opt.step()

        toks_per_step = args.batch_size * args.block_size * args.grad_accum_steps
        toks_cum = (step + 1) * toks_per_step
        wall_cum = time.time() - t_start

        if step % args.log_interval == 0:
            dt = time.time() - t0_wall
            t0_wall = time.time()
            extra = ""
            if args.arch == "yurii_lt" and args.yurii_restart != "none":
                extra = f" | restarts {restarts_accum}"
            if (args.arch.startswith("presymp") or args.arch.startswith("causal_symp_") or args.arch in ("plain_euler", "lin_presymp", "lin_exp_euler", "lin_ab2", "lin_etd_ab2", "lin_reduced_exp_mid", "lin_reduced_ab2")) and hasattr(model, "last_xi_mean"):
                extra += f" | hX {getattr(model, 'last_h_mean', float('nan')):.4g} | hY {getattr(model, 'last_hY_mean', float('nan')):.4g} | xi_mean {getattr(model, 'last_xi_mean', float('nan')):.3g} | rX {getattr(model, 'last_rX_max', float('nan')):.2e} | rP {getattr(model, 'last_rP_max', float('nan')):.2e} | c_log {getattr(model, 'last_c_log_mean', float('nan')):.4g} | c_lin {getattr(model, 'last_c_lin_mean', float('nan')):.4g} | leak_warn {int(getattr(model, 'last_leak_warnings', 0))}"
                if hasattr(model, 'last_t_start') or hasattr(model, 'last_t_end'):
                    extra += f" | t_sched0 {getattr(model, 'last_t_start', float('nan')):.4g} | t_sched1 {getattr(model, 'last_t_end', float('nan')):.4g}"
            print(
                f"[{args.arch}] step {step:6d} | loss {loss_accum:.4f} | lr {lr:.2e} | "
                f"toks/step {toks_per_step} | wall_dt {dt:.2f}s | wall {wall_cum:.1f}s{extra}"
            )
            append_csv_row(
                metrics_path,
                [
                    step,
                    f"{loss_accum:.6f}",
                    "",
                    f"{lr:.8e}",
                    f"{dt:.6f}",
                    f"{wall_cum:.6f}",
                    str(toks_per_step),
                    str(toks_cum),
                    f"{getattr(model, 'last_h_mean', '')}",
                    f"{getattr(model, 'last_hY_mean', '')}",
                    f"{getattr(model, 'last_xi_mean', '')}",
                    f"{getattr(model, 'last_rX_max', '')}",
                    f"{getattr(model, 'last_rP_max', '')}",
                    f"{getattr(model, 'last_c_log_mean', '')}",
                    f"{getattr(model, 'last_c_lin_mean', '')}",
                    f"{getattr(model, 'last_leak_warnings', '')}",
                    f"{getattr(model, 'last_t_start', '')}",
                    f"{getattr(model, 'last_t_end', '')}",
                ],
            )
            append_layer_dynamics(
                dynamics_path, model, step=step, tokens_cum=toks_cum, phase="train"
            )

        if step % args.eval_interval == 0 and step > 0:
            val_loss = estimate_loss(model, val_it, device, args.eval_batches, amp_dtype, global_step=step)
            append_csv_row(timing_path, [step, "eval", model.last_eval_wall_s, args.eval_batches])
            try:
                val_loss = require_finite_scalar(
                    val_loss,
                    kind="validation_loss",
                    step=step,
                )
            except NonFiniteTrainingError as error:
                abort_nonfinite(error, lr)
            print(f"[{args.arch}][eval] step {step:6d} | val_loss {val_loss:.4f}")
            if args.sample_interval > 0 and step % args.sample_interval == 0:
                print_sample(model, val_tokens, device, args, global_step=step, run_dir=run_dir)
            append_csv_row(
                metrics_path,
                [
                    step,
                    "",
                    f"{val_loss:.6f}",
                    f"{lr:.8e}",
                    "",
                    f"{wall_cum:.6f}",
                    str(toks_per_step),
                    str(toks_cum),
                    f"{getattr(model, 'last_h_mean', '')}",
                    f"{getattr(model, 'last_hY_mean', '')}",
                    f"{getattr(model, 'last_xi_mean', '')}",
                    f"{getattr(model, 'last_rX_max', '')}",
                    f"{getattr(model, 'last_rP_max', '')}",
                    f"{getattr(model, 'last_c_log_mean', '')}",
                    f"{getattr(model, 'last_c_lin_mean', '')}",
                    f"{getattr(model, 'last_leak_warnings', '')}",
                    f"{getattr(model, 'last_t_start', '')}",
                    f"{getattr(model, 'last_t_end', '')}",
                ],
            )
            # checkpoint best
            if val_loss < best_val:
                best_val = val_loss
                ckpt_path = os.path.join(run_dir, f"best_{args.arch}.pt")
                torch.save(
                    make_checkpoint(model, opt, step + 1, best_val, mcfg, args, train_it, val_it, wall_cum),
                    ckpt_path,
                )
                print(f"  saved best checkpoint -> {ckpt_path}")

    # A fixed final evaluation is mandatory even when max_steps is not aligned
    # with eval_interval; it defines the primary paper metric.
    final_wall_cum = time.time() - t_start
    final_val = estimate_loss(
        model, val_it, device, args.eval_batches, amp_dtype, global_step=args.max_steps,
        capture_geometry=True,
    )
    append_csv_row(timing_path, [args.max_steps, "final", model.last_eval_wall_s, args.eval_batches])
    final_lr = cosine_lr(args.max_steps, args.warmup_steps, args.max_steps, args.peak_lr, args.min_lr_ratio)
    try:
        final_val = require_finite_scalar(
            final_val,
            kind="final_validation_loss",
            step=args.max_steps,
        )
    except NonFiniteTrainingError as error:
        abort_nonfinite(error, final_lr)
    final_tokens = args.max_steps * tokens_per_step_config
    append_csv_row(
        metrics_path,
        [
            args.max_steps, "", f"{final_val:.6f}", f"{final_lr:.8e}", "",
            f"{final_wall_cum:.6f}", str(tokens_per_step_config), str(final_tokens),
            f"{getattr(model, 'last_h_mean', '')}", f"{getattr(model, 'last_hY_mean', '')}",
            f"{getattr(model, 'last_xi_mean', '')}", f"{getattr(model, 'last_rX_max', '')}",
            f"{getattr(model, 'last_rP_max', '')}", f"{getattr(model, 'last_c_log_mean', '')}",
            f"{getattr(model, 'last_c_lin_mean', '')}", f"{getattr(model, 'last_leak_warnings', '')}",
            f"{getattr(model, 'last_t_start', '')}", f"{getattr(model, 'last_t_end', '')}",
        ],
    )
    final_dynamics_count = append_layer_dynamics(
        dynamics_path, model, step=args.max_steps, tokens_cum=final_tokens, phase="final"
    )
    expected_dynamics_count = (
        len(getattr(model, "blocks", []))
        or len(getattr(model, "attn", []))
        or len(getattr(model, "attn_layers", []))
        or len(getattr(model, "layers", []))
    )
    has_damping = math.isfinite(float(getattr(model, "last_c_log_mean", math.nan)))
    dynamics_required = has_damping or args.arch.startswith("lin_") or args.arch in ("yurii_lt", "riem_prefix")
    if dynamics_required and final_dynamics_count != expected_dynamics_count:
        raise RuntimeError(
            f"malformed per-layer damping trace: expected {expected_dynamics_count}, "
            f"wrote {final_dynamics_count} rows"
        )
    print(f"[{args.arch}][final] step {args.max_steps:6d} | val_loss {final_val:.4f}")
    if final_val < best_val:
        best_val = final_val
        best_path = os.path.join(run_dir, f"best_{args.arch}.pt")
        torch.save(
            make_checkpoint(model, opt, args.max_steps, best_val, mcfg, args, train_it, val_it, final_wall_cum),
            best_path,
        )
        print(f"  saved best checkpoint -> {best_path}")

    # final checkpoint
    ckpt_path = os.path.join(run_dir, f"final_{args.arch}.pt")
    torch.save(
        make_checkpoint(model, opt, args.max_steps, best_val, mcfg, args, train_it, val_it, final_wall_cum),
        ckpt_path,
    )
    summary = {
        "best_val": best_val,
        "final_val": final_val,
        "final_step": args.max_steps,
        "tokens": args.max_steps * tokens_per_step_config,
        "wall_cum_s": final_wall_cum,
        "training_forward_calls": training_forward_calls,
        "backward_calls": backward_calls,
        "evaluation_forward_calls": getattr(model, "evaluation_forward_calls", 0),
        "geometry_work": model.work_counts() if isinstance(model, RiemannianPrefixModel) else None,
        "hb_work": model.work_counts() if isinstance(model, NormalizedHBModel) else None,
        "trainable_parameters": trainable_params,
        "peak_memory_mb": (torch.cuda.max_memory_allocated() / 2**20) if device.startswith("cuda") else 0.0,
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"saved final checkpoint -> {ckpt_path}")
    print(f"[{args.arch}] SUMMARY best_val={best_val:.6f} run_dir={run_dir}")

    if args.sample_interval > 0 or args.sample_final:
        print_sample(model, val_tokens, device, args, global_step=args.max_steps, run_dir=run_dir, force=True)

    if args.plot:
        plot_metrics_csv(metrics_path, plot_path, title=f"{args.arch} loss")
    damping_prefix = os.path.join(run_dir, "damping_coefficients")
    plot_layer_dynamics_csv(
        dynamics_path,
        damping_prefix,
        title=f"{args.arch} learned damping and forward steps",
    )
    if final_dynamics_count:
        for suffix in (".pdf", ".png"):
            artifact = damping_prefix + suffix
            if not os.path.isfile(artifact) or os.path.getsize(artifact) == 0:
                raise RuntimeError(f"missing mandatory damping artifact: {artifact}")


if __name__ == "__main__":
    main()
