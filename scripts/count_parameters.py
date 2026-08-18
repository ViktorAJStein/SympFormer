#!/usr/bin/env python3
"""Report trainable parameter counts for the predeclared headline/control models."""

import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import GPTModel, ModelConfig, PresympModel, YuriiFormerModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    args = ap.parse_args()
    raw = json.loads(args.config.read_text())
    cfg = ModelConfig(
        vocab_size=int(raw.get("vocab_size", 50304)),
        block_size=int(raw["block_size"]),
        n_layer=int(raw["n_layer"]),
        n_head=int(raw["n_head"]),
        n_embd=int(raw["n_embd"]),
        dropout=float(raw.get("dropout", 0.0)),
        bias=bool(raw.get("bias", False)),
    )
    builders = {
        "baseline": lambda: GPTModel(cfg),
        "baseline_capacity": lambda: GPTModel(cfg, capacity_control=True),
        "yurii_lt": lambda: YuriiFormerModel(cfg, use_v0_init=False),
        "causal_symp_fe": lambda: PresympModel(
            cfg, attn_scheme="causal_fe", h=float(raw.get("presymp_h", 0.1)),
            eta_learnable=bool(raw.get("eta_learnable", True)),
            eta_mode=str(raw.get("eta_mode", "loglin")),
            eta_log_init=float(raw.get("eta_log_init", 3.0)),
            eta_lin_init=float(raw.get("eta_lin_init", 1e-4)),
            use_v0_init=False, mlp_use_attn_vel=True,
            presymp_lnp=str(raw.get("presymp_lnp", "end")),
        ),
        "causal_symp_pe": lambda: PresympModel(
            cfg, attn_scheme="causal_pe", h=float(raw.get("presymp_h", 0.1)),
            eta_learnable=bool(raw.get("eta_learnable", True)),
            eta_mode=str(raw.get("eta_mode", "loglin")),
            eta_log_init=float(raw.get("eta_log_init", 3.0)),
            eta_lin_init=float(raw.get("eta_lin_init", 1e-4)),
            use_v0_init=False, mlp_use_attn_vel=True,
            presymp_lnp=str(raw.get("presymp_lnp", "end")),
        ),
    }
    for name, build in builders.items():
        model = build()
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{name}\t{count}")
        del model
        gc.collect()


if __name__ == "__main__":
    main()
