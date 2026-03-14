
import argparse
import os
import time
import csv
import warnings
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from data import DataConfig, BlockEpochIterator, load_bin
from model import ModelConfig, GPTModel, YuriiFormerModel, PresympModel, PresympModelAB2, PresympModelETDAB2, LinAttnModel, LinAttnYuriiModel, LinAttnEulerModel, LinAttnPresympModel, LinAttnAB2Model, LinAttnETDAB2Model


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
):
    if int(args.sample_interval) <= 0:
        return

    enc = maybe_get_tokenizer()
    prompt_cpu, prompt_kind = build_prompt_tokens(args, dataset_tokens, enc)
    prompt = prompt_cpu.to(device)

    out = model.generate(
        prompt,
        max_new_tokens=int(args.sample_max_new_tokens),
        temperature=float(args.sample_temperature),
        top_k=(None if int(args.sample_top_k) <= 0 else int(args.sample_top_k)),
        do_sample=bool(int(args.sample_do_sample)),
        eos_token_id=(None if int(args.sample_eos_token_id) < 0 else int(args.sample_eos_token_id)),
        global_step=global_step,
    )
    out_cpu = out[0].detach().cpu().tolist()
    prompt_len = prompt_cpu.shape[1]

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


def build_optimizer(model: nn.Module, peak_lr: float, betas=(0.9, 0.95), scalar_lr_mult: float = 10.0):
    # Parameter grouping following YuriiFormer Appendix A.3 (AdamW side):
    # - embeddings: weight decay 0.1
    # - norms: weight decay 0
    # - learned scalar update-rule params: weight decay 0, lr multiplier 5x
    # - everything else: weight decay 0 (Muon would handle matrix weights in the paper; here we keep AdamW wd=0)
    decay_emb = 0.1
    scalar_mult = scalar_lr_mult

    emb_params = []
    norm_params = []
    scalar_params = []
    integrator_params = []  # theta_h/theta_xi_raw
    other_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ("tok_emb" in name) or ("pos_emb" in name) or ("tok_v0_emb" in name) or ("pos_v0_emb" in name):
            emb_params.append(p)
        elif "theta_h" in name or "theta_hX" in name or "theta_hY" in name or "theta_xi_raw" in name:
            integrator_params.append(p)
        elif ".raw" in name:  # ConstrainedScalar raw parameters
            scalar_params.append(p)
        elif "ln_" in name or ".ln" in name or "ln_f" in name or "ln_v" in name or "LayerNorm" in name:
            norm_params.append(p)
        else:
            other_params.append(p)

    param_groups = []
    if other_params:
        param_groups.append({"params": other_params, "lr": peak_lr, "weight_decay": 0.0})
    if emb_params:
        param_groups.append({"params": emb_params, "lr": peak_lr, "weight_decay": decay_emb})
    if norm_params:
        param_groups.append({"params": norm_params, "lr": peak_lr, "weight_decay": 0.0})
    if scalar_params:
        param_groups.append({"params": scalar_params, "lr": peak_lr * scalar_mult, "weight_decay": 0.0, "lr_mult": scalar_mult})
    if integrator_params:
        param_groups.append({"params": integrator_params, "lr": peak_lr * scalar_mult, "weight_decay": 0.0, "lr_mult": scalar_mult})

    opt = AdamW(param_groups, betas=betas)
    return opt


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    it: BlockEpochIterator,
    device: str,
    eval_batches: int,
    amp_dtype: torch.dtype,
    global_step: int,
):
    model.eval()
    losses = []
    for _ in range(eval_batches):
        xb, yb = next(it)
        xb = xb.to(device)
        yb = yb.to(device)
        with torch.autocast(device_type=device.split(':')[0], dtype=amp_dtype, enabled=(device.startswith("cuda"))):
            _, loss = model(xb, yb, global_step=global_step)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--dataset", type=str, default="tinystories", choices=["tinystories", "openwebtext"])
    ap.add_argument(
        "--arch",
        type=str,
        default="yurii_lt",
        choices=["baseline", "yurii_lt", "presymp", "presymp_euler", "presymp_exp_euler", "presymp_ab2", "presymp_etd_ab2", "presymp_strang", "plain_euler", "lin_baseline", "lin_yurii", "lin_euler", "lin_presymp", "lin_exp_euler", "lin_ab2", "lin_etd_ab2"],
        help="model architecture / attention discretization",
    )

    ap.add_argument(
        "--no_mlp",
        action="store_true",
        help="Skip the MLP substep entirely in all architectures. "
             "Use this to isolate the effect of the different attention blocks.",
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
    ap.add_argument("--eta_learnable", action="store_true", help="make eta schedule coefficient(s) learnable")
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
            "(Nesterov lookahead inside the symplectic step). "
            "mu_la is a per-layer learned scalar initialised near 0 (no-op start)."
        ),
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


    # v0 initialization embeddings for momentum variants (YuriiFormer Appendix A.1)
    ap.add_argument(
        "--no_v0_init",
        action="store_true",
        help="disable separate token/pos v0 embeddings for initializing velocity/momentum (momentum variants only)",
    )
    ap.add_argument(
        "--allow_token_conditioned_v0_init",
        action="store_true",
        help="opt back into token-conditioned v0 initialisation. Unsafe for leak-free autoregressive evaluation and therefore disabled by default for presymplectic softmax models.",
    )

    # YuriiFormer noise + restart (applied across depth)
    ap.add_argument("--yurii_noise_eta", type=float, default=0.0, help="noise variance scale eta in sigma_t^2 = eta/(1+t)^gamma")
    ap.add_argument("--yurii_noise_gamma", type=float, default=0.55, help="noise decay exponent gamma in sigma_t^2 = eta/(1+t)^gamma")
    ap.add_argument("--yurii_noise_loc", type=str, default="v", choices=["dx", "v", "xin"], help="inject noise into dx, v, or lookahead xin")
    ap.add_argument("--yurii_restart", type=str, default="none", choices=["none", "speed", "loss"], help="restart criterion")
    ap.add_argument("--yurii_restart_min_layer", type=int, default=1, help="start checking restart conditions at this layer index")

    # Training hyperparams (paper: 10k steps, warmup 1k, peak AdamW LR 6e-4, bf16, clip 1.0)
    ap.add_argument("--max_steps", type=int, default=10_000)
    ap.add_argument("--warmup_steps", type=int, default=1_000)
    ap.add_argument("--peak_lr", type=float, default=6e-4)
    ap.add_argument("--min_lr_ratio", type=float, default=0.1)
    ap.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    ap.add_argument("--grad_clip", type=float, default=1.0)

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

    # Text generation / inspection
    ap.add_argument("--sample_interval", type=int, default=0,
                    help="If >0, print a decoded text sample every this many training steps (typically at eval time).")
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

    args = ap.parse_args()

    presymp_softmax_arches = {"presymp", "presymp_euler", "presymp_exp_euler", "presymp_ab2", "presymp_etd_ab2", "presymp_strang", "plain_euler"}
    if args.arch in presymp_softmax_arches and not args.allow_token_conditioned_v0_init:
        args.no_v0_init = True
    elif args.arch in presymp_softmax_arches and args.allow_token_conditioned_v0_init:
        warnings.warn(
            "Token-conditioned v0 initialisation was explicitly re-enabled. This can create an autoregressive shortcut in the momentum stream; leak checks will warn at runtime if it becomes unsafe.",
            RuntimeWarning,
        )

    if args.arch in presymp_softmax_arches and not args.presymp_mlp_use_attn_vel and not args.presymp_mlp_use_p_vel:
        args.presymp_mlp_use_attn_vel = True

    # Use per-run directory to avoid collisions when running multiple arch variants.
    run_dir = os.path.join(args.out_dir, args.arch)
    if args.run_name:
        run_dir = os.path.join(args.out_dir, f"{args.arch}_{args.run_name}")
    os.makedirs(run_dir, exist_ok=True)

    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")
    amp_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    # Load data
    train_path = os.path.join(args.data_dir, f"{args.dataset}_train.bin")
    val_path = os.path.join(args.data_dir, f"{args.dataset}_val.bin")
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise SystemExit(f"Missing dataset bins: {train_path} / {val_path}")

    train_tokens = load_bin(train_path)
    val_tokens = load_bin(val_path)

    dcfg = DataConfig(block_size=args.block_size, batch_size=args.batch_size, grad_accum_steps=args.grad_accum_steps, seed=args.seed, device=device)
    train_it = BlockEpochIterator(train_tokens, dcfg, split="train")
    val_it = BlockEpochIterator(val_tokens, dcfg, split="val")

    # Build model
    mcfg = ModelConfig(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        bias=args.bias,
    )

    if args.arch == "baseline":
        model = GPTModel(mcfg, no_mlp=args.no_mlp)
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
        )
    else:
        # Presymp family: same overall architecture, different attention discretization
        if args.arch == "presymp":
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
        elif args.arch == "lin_baseline":
            model = LinAttnModel(
                mcfg,
                h=args.presymp_h,
                no_mlp=args.no_mlp,
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
        else:
            raise ValueError(f"Unknown arch: {args.arch}")

    model.to(device)
    # Optionally freeze learned integrator scalars (h, xi) so they stay fixed.
    if args.learn_h == 0:
        for n, p in model.named_parameters():
            if "theta_h" in n or "theta_hX" in n or "theta_hY" in n:
                p.requires_grad_(False)
    if args.learn_xi == 0:
        for n, p in model.named_parameters():
            if "theta_xi_raw" in n:
                p.requires_grad_(False)
        # disable heuristic xi adaptation if present
        # if hasattr(args, "presymp_xi_adapt"):
        #     args.presymp_xi_adapt = False


    opt = build_optimizer(model, peak_lr=args.peak_lr, betas=tuple(args.betas), scalar_lr_mult=args.scalar_lr_mult)

    start_step = 0
    best_val = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start_step = ckpt.get("step", 0)
        best_val = ckpt.get("best_val", float("inf"))
        print(f"Resumed from {args.resume} at step {start_step}, best_val={best_val}")

    metrics_path = os.path.join(run_dir, "metrics.csv")
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
    t_start = time.time()          # for wall_cum_s
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

            with torch.autocast(device_type=device.split(':')[0], dtype=amp_dtype, enabled=(device.startswith("cuda"))):
                _, loss = model(xb, yb, global_step=step)
                loss = loss / args.grad_accum_steps
            loss.backward()
            loss_accum += loss.item()
            if hasattr(model, "last_restart_count"):
                restarts_accum += int(getattr(model, "last_restart_count", 0))

        # clip
        if args.grad_clip > 0:
            clip_params = [p for (n,p) in model.named_parameters() if p.requires_grad and ('theta_h' not in n and 'theta_hX' not in n and 'theta_hY' not in n and 'theta_xi_raw' not in n)]
            torch.nn.utils.clip_grad_norm_(clip_params, args.grad_clip)

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
            if (args.arch.startswith("presymp") or args.arch in ("plain_euler", "lin_presymp", "lin_exp_euler", "lin_ab2", "lin_etd_ab2")) and hasattr(model, "last_xi_mean"):
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

        if step % args.eval_interval == 0 and step > 0:
            val_loss = estimate_loss(model, val_it, device, args.eval_batches, amp_dtype, global_step=step)
            print(f"[{args.arch}][eval] step {step:6d} | val_loss {val_loss:.4f}")
            if args.sample_interval > 0 and step % args.sample_interval == 0:
                print_sample(model, val_tokens, device, args, global_step=step)
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
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "best_val": best_val, "cfg": asdict(mcfg), "args": vars(args)}, ckpt_path)
                print(f"  saved best checkpoint -> {ckpt_path}")

    # final checkpoint
    ckpt_path = os.path.join(run_dir, f"final_{args.arch}.pt")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": args.max_steps, "best_val": best_val, "cfg": asdict(mcfg), "args": vars(args)}, ckpt_path)
    print(f"saved final checkpoint -> {ckpt_path}")
    print(f"[{args.arch}] SUMMARY best_val={best_val:.6f} run_dir={run_dir}")

    if args.sample_interval > 0:
        print_sample(model, val_tokens, device, args, global_step=args.max_steps)

    if args.plot:
        plot_metrics_csv(metrics_path, plot_path, title=f"{args.arch} loss")


if __name__ == "__main__":
    main()
