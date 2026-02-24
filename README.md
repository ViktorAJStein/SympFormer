# YuriiFormer vs Presymplectic Softmax Attention (PyTorch)

This is a minimal, reproducible training setup to compare:

1) **nanoGPT-style baseline** (GD + Lie–Trotter): `x <- x + Attn(LN(x)); x <- x + MLP(LN(x))`
2) **YuriiFormer (Nesterov + Lie–Trotter)**: 2-stream (state + velocity) update from the paper (attention then MLP).
3) **Our idea**: replace the attention sublayer by the **explicit, 2nd-order presymplectic nonseparable integrator**
   applied to a **symmetric softmax kernel** `E_ij = exp( <u_i, u_j> / sqrt(d_head) )` (causal masked).
   The MLP sublayer is kept as in the baseline.

This repo is intentionally small and has no external dependencies beyond PyTorch + NumPy + (optional) `tiktoken` + (optional) `datasets`.

## What matches the YuriiFormer paper

From Appendix A of the YuriiFormer PDF:
- Decoder-only causal LM, context length `T=1024`
- Pre-norm: LN before attention and LN before MLP
- MLP: 2-layer GELU with 4× expansion
- Dropout 0, no bias terms
- GPT-2 BPE tokenizer (tiktoken), vocab padded to `V=50304`
- Global grad-norm clip at 1.0
- bfloat16 training
- Cosine LR schedule with linear warmup to peak LR, min LR = 0.1×peak
- TinyStories small model: 12 layers / 12 heads / d=768, 10k steps, warmup 1k, peak AdamW LR 6e-4
- Global batch in the paper: 30 sequences with 16-step grad accumulation at T=1024 (~491,520 tokens/opt step).
  On smaller GPUs you will likely need a smaller microbatch and/or more accumulation steps.

The paper uses a mixed Muon+AdamW optimizer. This code implements **AdamW parameter-grouping** matching Appendix A.3
(weight decay 0.1 on embeddings, 0 on norms/scalars; scalar LR multiplier 5×). It does **not** implement Muon.

## Quickstart

### 1) Preprocess TinyStories (requires internet once)
This downloads TinyStories via HuggingFace `datasets`, tokenizes with `tiktoken` GPT-2 BPE, and writes
`data/tinystories_train.bin` and `data/tinystories_val.bin` (uint16 tokens).

```bash
python preprocess_tinystories.py --out_dir data --max_docs_train 0 --max_docs_val 0
```

Set `--max_docs_*` to a positive integer for a quick smoke test.

### 2) Train baseline / YuriiFormer / presymplectic

Baseline (GD+Lie–Trotter):
```bash
python train.py --data_dir data --dataset tinystories --arch baseline
```

YuriiFormer (Nesterov+Lie–Trotter):
```bash
python train.py --data_dir data --dataset tinystories --arch yurii_lt
```

Presymplectic attention replacement:
```bash
python train.py --data_dir data --dataset tinystories --arch presymp
```

Recommended: start with small microbatch and increase `--grad_accum_steps` until stable.

## Notes on the presymplectic block
- It is compute-heavier than standard attention because it evaluates the attention-kernel multiple times per layer.
- It uses a **symmetric** softmax kernel (shared Q=K) to match the SPD-score assumption used in the derivation.
- The MLP sublayer is applied **after** the presymplectic attention step; the momentum state is **not** updated by the MLP
  (this is the simplest "replace attention only" variant).

## Files
- `model.py` models/blocks
- `data.py` deterministic block-epoch token iterator (matches paper's "shift boundaries + reshuffle")
- `train.py` training loop
- `preprocess_tinystories.py` dataset preprocessing

