#!/usr/bin/env python3
"""Verify exact training resume and non-advancing validation evaluation."""

import copy
import pickle
import random

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.optim import AdamW

from data import BlockEpochIterator, DataConfig
from model import GPTModel, ModelConfig
from train import capture_rng_state, estimate_loss, restore_rng_state


def train_step(model, opt, iterator):
    model.train()
    x, y = next(iterator)
    opt.zero_grad(set_to_none=True)
    _, loss = model(x, y)
    loss.backward()
    opt.step()
    return float(loss.detach())


def same_state(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


def main():
    seed = 314159
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tokens = (np.arange(400, dtype=np.uint16) * 7) % 31
    dcfg = DataConfig(block_size=8, batch_size=2, seed=seed, device="cpu")
    cfg = ModelConfig(vocab_size=32, block_size=8, n_layer=2, n_head=2, n_embd=8, dropout=0.1)

    iterator_a = BlockEpochIterator(tokens, dcfg, split="train")
    model_a = GPTModel(cfg)
    opt_a = AdamW(model_a.parameters(), lr=1e-3)
    train_step(model_a, opt_a, iterator_a)
    train_step(model_a, opt_a, iterator_a)
    checkpoint = {
        "model": copy.deepcopy(model_a.state_dict()),
        "opt": copy.deepcopy(opt_a.state_dict()),
        "iterator": copy.deepcopy(iterator_a.state_dict()),
        "rng": copy.deepcopy(capture_rng_state()),
    }
    expected_loss = train_step(model_a, opt_a, iterator_a)
    expected_state = copy.deepcopy(model_a.state_dict())

    # Constructor calls intentionally consume RNG before checkpoint restoration.
    iterator_b = BlockEpochIterator(tokens, dcfg, split="train")
    model_b = GPTModel(cfg)
    opt_b = AdamW(model_b.parameters(), lr=1e-3)
    model_b.load_state_dict(checkpoint["model"])
    opt_b.load_state_dict(checkpoint["opt"])
    iterator_b.load_state_dict(checkpoint["iterator"])
    restore_rng_state(checkpoint["rng"])
    resumed_loss = train_step(model_b, opt_b, iterator_b)
    print(f"expected_loss={expected_loss:.12f} resumed_loss={resumed_loss:.12f}")
    if expected_loss != resumed_loss or not same_state(expected_state, model_b.state_dict()):
        raise AssertionError("Resumed step is not bitwise identical on CPU")

    val_it = BlockEpochIterator(tokens, dcfg, split="val")
    before = pickle.dumps(val_it.state_dict())
    estimate_loss(model_b, val_it, "cpu", 3, torch.float32, global_step=3)
    after = pickle.dumps(val_it.state_dict())
    if before != after:
        raise AssertionError("Validation evaluation advanced iterator state")
    print("PASS: exact CPU resume and fixed validation iterator verified.")


if __name__ == "__main__":
    main()
