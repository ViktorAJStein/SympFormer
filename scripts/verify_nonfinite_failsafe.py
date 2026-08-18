#!/usr/bin/env python3
"""Verify loss/gradient fail-fast checks and their forensic artifacts."""

import json
import math
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import ModelConfig
from train import (
    NonFiniteTrainingError,
    clip_grad_norm_finite,
    require_finite_scalar,
    write_nonfinite_failure,
)


class DummyIterator:
    def __init__(self, position):
        self.position = int(position)

    def state_dict(self):
        return {"position": self.position}


def expect_nonfinite_scalar(value, expected):
    try:
        require_finite_scalar(value, kind="training_loss", step=7, micro_step=3)
    except NonFiniteTrainingError as error:
        assert error.kind == "training_loss"
        assert error.step == 7
        assert error.micro_step == 3
        assert error.value == expected
        return error
    raise AssertionError(f"Expected {value!r} to be rejected")


def main():
    assert math.isclose(
        require_finite_scalar(torch.tensor(1.25), kind="training_loss", step=0),
        1.25,
    )
    nan_error = expect_nonfinite_scalar(torch.tensor(float("nan")), "nan")
    expect_nonfinite_scalar(float("inf"), "inf")

    parameter = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    parameter.grad = torch.tensor([3.0, 4.0])
    norm = clip_grad_norm_finite([parameter], 1.0, step=5)
    assert math.isclose(norm, 5.0, rel_tol=0, abs_tol=1e-6)
    assert float(parameter.grad.norm()) <= 1.000001

    parameter.grad = torch.tensor([float("nan"), 1.0])
    try:
        clip_grad_norm_finite([parameter], 1.0, step=6)
    except NonFiniteTrainingError as error:
        assert error.kind == "gradient_norm"
        assert error.step == 6
        assert "non-finite" in error.detail.lower()
    else:
        raise AssertionError("A NaN gradient was not rejected")

    with tempfile.TemporaryDirectory(prefix="nonfinite-failsafe-") as directory:
        run_dir = Path(directory)
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        args = SimpleNamespace(arch="baseline", run_name="synthetic_failure", seed=17)
        config = ModelConfig(
            vocab_size=16,
            block_size=4,
            n_layer=1,
            n_head=1,
            n_embd=4,
            dropout=0.0,
            bias=False,
        )
        failure_path, checkpoint_path = write_nonfinite_failure(
            run_dir,
            nan_error,
            model=model,
            opt=optimizer,
            best_val=1.5,
            mcfg=config,
            args=args,
            train_it=DummyIterator(11),
            val_it=DummyIterator(13),
            wall_cum_s=12.5,
            tokens_per_step=32,
            lr=6e-4,
        )
        payload = json.loads(Path(failure_path).read_text())
        assert payload["status"] == "failed_nonfinite"
        assert payload["failure_type"] == "training_loss"
        assert payload["tokens_completed"] == 7 * 32
        assert payload["checkpoint_resumable"] is False
        assert Path(checkpoint_path).is_file()

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert checkpoint["resumable"] is False
        assert checkpoint["failure"]["failure_type"] == "training_loss"
        assert checkpoint["train_iterator"] == {"position": 11}
        assert checkpoint["val_iterator"] == {"position": 13}

    source = (ROOT / "train.py").read_text()
    assert "error_if_nonfinite=True" in source
    assert 'kind="validation_loss"' in source
    assert 'kind="final_validation_loss"' in source
    assert "raise error" in source
    print(
        "PASS: finite values proceed; NaN/Inf losses and gradients are rejected; "
        "failure JSON and a non-resumable forensic checkpoint are written"
    )


if __name__ == "__main__":
    main()
