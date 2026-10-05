
import os
from dataclasses import dataclass
from typing import Iterator, Tuple, Optional

import numpy as np
import torch


@dataclass
class DataConfig:
    block_size: int = 1024
    batch_size: int = 4          # microbatch per step (per GPU)
    grad_accum_steps: int = 1
    seed: int = 1337
    device: str = "cuda"


class BlockEpochIterator:
    '''
    Implements the deterministic epoch protocol described in YuriiFormer Appendix A.2:
      - tokenize into a long sequence
      - train in epochs over non-overlapping T-token blocks
      - each epoch visits every block exactly once
      - between epochs: shift block boundaries by a seeded offset and reshuffle block order
    '''
    def __init__(self, tokens: np.ndarray, cfg: DataConfig, split: str):
        assert tokens.ndim == 1
        self.tokens = tokens
        self.cfg = cfg
        self.split = split
        self.rng = np.random.default_rng(cfg.seed + (0 if split == "train" else 1))

        self.T = cfg.block_size
        # number of full blocks
        self.n_blocks = (len(tokens) - 1) // self.T  # -1 because we need y = x shifted by 1
        if self.n_blocks <= 0:
            raise ValueError("Not enough tokens for one block")

        self.epoch = 0
        self._starts = None
        self._pos = 0
        self._prepare_epoch()

    def _prepare_epoch(self):
        # seeded offset in [0, T-1]
        offset = int(self.rng.integers(low=0, high=self.T))
        starts = offset + np.arange(self.n_blocks, dtype=np.int64) * self.T
        # ensure within bounds for x of length T and y shifted by 1
        max_start = len(self.tokens) - (self.T + 1)
        starts = starts[starts <= max_start]
        self.rng.shuffle(starts)
        if len(starts) < self.cfg.batch_size:
            raise ValueError(
                f"Split has only {len(starts)} complete blocks after offset, "
                f"fewer than batch_size={self.cfg.batch_size}"
            )
        self._starts = starts
        self._pos = 0
        self.epoch += 1

    def state_dict(self):
        """Return all state needed to resume the exact batch sequence."""
        return {
            "rng_state": self.rng.bit_generator.state,
            "epoch": int(self.epoch),
            "starts": self._starts.copy(),
            "pos": int(self._pos),
        }

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state["rng_state"]
        self.epoch = int(state["epoch"])
        self._starts = np.asarray(state["starts"], dtype=np.int64).copy()
        self._pos = int(state["pos"])
        if not 0 <= self._pos <= len(self._starts):
            raise ValueError("Invalid iterator position in checkpoint")

    def __iter__(self) -> "BlockEpochIterator":
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        bs = self.cfg.batch_size
        if self._pos + bs > len(self._starts):
            # start new epoch
            self._prepare_epoch()

        batch_starts = self._starts[self._pos:self._pos + bs]
        self._pos += bs

        T = self.T
        x = np.stack([self.tokens[s:s+T] for s in batch_starts], axis=0)
        y = np.stack([self.tokens[s+1:s+T+1] for s in batch_starts], axis=0)

        x = torch.from_numpy(x.astype(np.int64))
        y = torch.from_numpy(y.astype(np.int64))
        return x, y


def load_bin(path: str) -> np.ndarray:
    """Memory-map uint16 tokens so OpenWebText is not copied into host RAM."""
    return np.memmap(path, dtype=np.uint16, mode="r")
