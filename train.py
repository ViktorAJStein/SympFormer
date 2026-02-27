
import argparse
import os
import time
import csv
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from data import DataConfig, BlockEpochIterator, load_bin
from model import ModelConfig, GPTModel, YuriiFormerModel, PresympModel


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


def build_optimizer(model: nn.Module, peak_lr: float, betas=(0.9, 0.95)):
    # Parameter grouping following YuriiFormer Appendix A.3 (AdamW side):
    # - embeddings: weight decay 0.1
    # - norms: weight decay 0
    # - learned scalar update-rule params: weight decay 0, lr multiplier 5x
    # - everything else: weight decay 0 (Muon would handle matrix weights in the paper; here we keep AdamW wd=0)
    decay_emb = 0.1
    scalar_mult = 5.0

    emb_params = []
    norm_params = []
    scalar_params = []
    other_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ("tok_emb" in name) or ("pos_emb" in name) or ("tok_v0_emb" in name) or ("pos_v0_emb" in name):
            emb_params.append(p)
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
    ap.add_argument("--dataset", type=str, default="tinystories", choices=["tinystories"])
    ap.add_argument(
        "--arch",
        type=str,
        default="yurii_lt",
        choices=["baseline", "yurii_lt", "presymp", "presymp_euler", "presymp_strang"],
        help="model architecture / attention discretization",
    )

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
    ap.add_argument("--presymp_xi_adapt", action="store_true", help="adapt xi online based on r_X and r_P thresholds (breaks exact presymplecticity)")
    ap.add_argument("--presymp_r_thresh", type=float, default=1e-2, help="increase xi if max(r_X,r_P) exceeds this")
    ap.add_argument("--presymp_r_low", type=float, default=1e-4, help="decrease xi if max(r_X,r_P) goes below this")
    ap.add_argument("--presymp_xi_mult_up", type=float, default=1.25, help="multiplier when increasing xi")
    ap.add_argument("--presymp_xi_mult_down", type=float, default=0.5, help="multiplier when decreasing xi")
    ap.add_argument("--presymp_xi_min", type=float, default=1e-4, help="lower bound for xi during adaptation")
    ap.add_argument("--presymp_xi_max", type=float, default=100.0, help="upper bound for xi during adaptation (also capped by theta_max/(2h))")
    ap.add_argument("--presymp_theta_max", type=float, default=1.0, help="cap the coupling rotation angle theta=2*xi*h to at most theta_max by enforcing xi<=theta_max/(2h)")
    ap.add_argument("--presymp_xi_adapt_warmup", type=int, default=10, help="do not adapt xi for the first this many presymp steps")
    ap.add_argument("--presymp_xi_adapt_every", type=int, default=1, help="update xi every N presymp steps (after warmup)")
    ap.add_argument("--presymp_lnp", type=str, default="end", choices=["none","end","each_substep"], help="LayerNorm on presymplectic attention momentum P/Pi: none|end|each_substep")

    # Variant A: use attention-induced velocity for the MLP lookahead (drop separate MLP velocity dynamics)
    ap.add_argument(
        "--presymp_mlp_use_attn_vel",
        action="store_true",
        help="Presymp only: use v_attn ≈ (X_after_attn - X_before_attn)/h as the velocity in the MLP lookahead (Variant A).",
    )


    # v0 initialization embeddings for momentum variants (YuriiFormer Appendix A.1)
    ap.add_argument(
        "--no_v0_init",
        action="store_true",
        help="disable separate token/pos v0 embeddings for initializing velocity/momentum (momentum variants only)",
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

    args = ap.parse_args()

    # Use per-run directory to avoid collisions when running multiple arch variants.
    run_dir = os.path.join(args.out_dir, args.arch)
    if args.run_name:
        run_dir = os.path.join(args.out_dir, f"{args.arch}_{args.run_name}")
    os.makedirs(run_dir, exist_ok=True)

    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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
        model = GPTModel(mcfg)
    elif args.arch == "yurii_lt":
        model = YuriiFormerModel(
            mcfg,
            use_v0_init=(not args.no_v0_init),
            noise_eta=args.yurii_noise_eta,
            noise_gamma=args.yurii_noise_gamma,
            noise_loc=args.yurii_noise_loc,
            restart_mode=args.yurii_restart,
            restart_min_layer=args.yurii_restart_min_layer,
        )
    else:
        # Presymp family: same overall architecture, different attention discretization
        if args.arch == "presymp":
            attn_scheme = "presymp"
        elif args.arch == "presymp_euler":
            attn_scheme = "euler"
        elif args.arch == "presymp_strang":
            attn_scheme = "strang"
        else:
            raise ValueError(f"Unknown arch: {args.arch}")

        model = PresympModel(
            mcfg,
            attn_scheme=attn_scheme,
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
            xi_adapt=args.presymp_xi_adapt,
            r_thresh=args.presymp_r_thresh,
            r_low=args.presymp_r_low,
            xi_mult_up=args.presymp_xi_mult_up,
            xi_mult_down=args.presymp_xi_mult_down,
            xi_min=args.presymp_xi_min,
            xi_max=args.presymp_xi_max,
            theta_max=args.presymp_theta_max,
            presymp_lnp=args.presymp_lnp,
            xi_adapt_warmup=args.presymp_xi_adapt_warmup,
            xi_adapt_every=args.presymp_xi_adapt_every,
            mlp_use_attn_vel=args.presymp_mlp_use_attn_vel,
        )

    model.to(device)

    opt = build_optimizer(model, peak_lr=args.peak_lr, betas=tuple(args.betas))

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
    ensure_csv_header(
        metrics_path,
        ["step", "train_loss", "val_loss", "lr", "wall_dt_s", "wall_cum_s", "tokens_step", "tokens_cum", "xi", "rX", "rP"],
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

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
            if args.arch.startswith("presymp") and hasattr(model, "last_xi_mean"):
                extra += f" | xi_mean {getattr(model, 'last_xi_mean', float('nan')):.3g} | rX {getattr(model, 'last_rX_max', float('nan')):.2e} | rP {getattr(model, 'last_rP_max', float('nan')):.2e}"
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
                    f"{getattr(model, 'last_xi_mean', '')}",
                    f"{getattr(model, 'last_rX_max', '')}",
                    f"{getattr(model, 'last_rP_max', '')}",
                ],
            )

        if step % args.eval_interval == 0 and step > 0:
            val_loss = estimate_loss(model, val_it, device, args.eval_batches, amp_dtype, global_step=step)
            print(f"[{args.arch}][eval] step {step:6d} | val_loss {val_loss:.4f}")
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
                    f"{getattr(model, 'last_xi_mean', '')}",
                    f"{getattr(model, 'last_rX_max', '')}",
                    f"{getattr(model, 'last_rP_max', '')}",
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

    if args.plot:
        plot_metrics_csv(metrics_path, plot_path, title=f"{args.arch} loss")


if __name__ == "__main__":
    main()
