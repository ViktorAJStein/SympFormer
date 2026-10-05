"""Proposed aspect study constants and paired-target sampler (not a job launcher)."""
from dataclasses import replace

METHODS = ('B', 'Y', 'H0', 'H1', 'R2')
SHAPES = (
    (8, 128, 8), (16, 128, 8), (32, 128, 8), (8, 256, 8),
    (8, 512, 8), (8, 128, 16), (32, 128, 16), (8, 512, 16),
)
TOKENS_PER_UPDATE = 16384
VALIDATION_TOKENS = 32768
MACRO_LENGTH = 512


class SharedTargetIterator:
    """Reshape a context-independent stream of whole512-token macroblocks.

    parent_class must be the explicitly imported frozen BlockEpochIterator.
    No modification of that sampler or any historic data protocol is performed.
    """
    def __init__(self, parent_class, tokens, cfg, split):
        if cfg.block_size not in (128, 256, 512):
            raise ValueError('Unsupported aspect-study context')
        if cfg.batch_size <= 0 or cfg.batch_size * cfg.block_size % MACRO_LENGTH:
            raise ValueError('Microbatch must contain whole macroblocks')
        if split not in ('train', 'val'):
            raise ValueError('Unknown split')
        self.cfg = cfg
        self.layout = dict(schema='shared_macro512_v1', context=cfg.block_size,
                           batch_size=cfg.batch_size, seed=cfg.seed, split=split,
                           dataset_length=len(tokens), macro_length=MACRO_LENGTH)
        self.parent = parent_class(tokens, replace(cfg, block_size=MACRO_LENGTH, batch_size=1), split)
        self.macros_per_batch = cfg.batch_size * cfg.block_size // MACRO_LENGTH

    def __iter__(self):
        return self

    def __next__(self):
        import torch
        pairs = [next(self.parent) for _ in range(self.macros_per_batch)]
        return tuple(torch.cat([pair[i] for pair in pairs], dim=0).reshape(
            self.cfg.batch_size, self.cfg.block_size) for i in (0, 1))

    def state_dict(self):
        return dict(layout=dict(self.layout), parent=self.parent.state_dict())

    def load_state_dict(self, state):
        if state['layout'] != self.layout:
            raise ValueError('Incompatible shared-target iterator layout')
        self.parent.load_state_dict(state['parent'])
