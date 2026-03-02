# SymplectFormer

Requirements
* PyTorch
* Numpy

For installing data sets using this code, one needs the following packages
* tiktoken
* datasets

## Quickstart

### 1) Preprocess TinyStories (requires internet once)
This downloads TinyStories via HuggingFace `datasets`, tokenizes with `tiktoken` GPT-2 BPE, and writes
`data/tinystories_train.bin` and `data/tinystories_val.bin` (uint16 tokens).

```bash
python preprocess_tinystories.py --out_dir data --max_docs_train 0 --max_docs_val 0
```

Set `--max_docs_*` to a positive integer for a quick smoke test.

### 2) Train 

...

## Files
- `model.py` models/blocks
- `data.py` deterministic block-epoch token iterator (shift boundaries + reshuffle")
- `train.py` training loop
- `preprocess_tinystories.py` dataset preprocessing

