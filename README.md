# SympFormer

Code for training and comparing transformer variants with classical, accelerated, and presymplectic-style attention updates.

The repository currently contains:
- a baseline GPT-style causal language model,
- an implementation of the [Yuriiformer](https://arxiv.org/abs/2601.23236) (Zimin et al, 2026) architecture, specifically the Lie-Trotter Nesterov acceleration,
- several softmax-attention presymplectic / Euler / higher-order variants,
- several linear-attention analogues,
- dataset preprocessing scripts for TinyStories and OpenWebText,
- a training script with checkpointing, metric logging, and optional text sampling,
- plotting and batch-job helper scripts.


---

## Repository layout

```text
README.md
model.py                  # model definitions and attention/update-rule variants
train.py                  # training loop, evaluation, checkpointing, text sampling
data.py                   # dataset loading + deterministic token block iterator
preprocess_tinystories.py # downloads/tokenizes TinyStories into .bin files
process_openwebtext.py    # downloads/tokenizes OpenWebText into .bin files
plot_compare.py           # compare runs from their metrics.csv files
job.job                   # example SLURM job for softmax-attention variants
job_lin.job               # example SLURM job for linear-attention variants
```

---

## Requirements

At minimum:
- `Python 3.10+`
- `PyTorch`
- `NumPy`

For dataset preprocessing and decoded text sampling:
- `tiktoken`
- `datasets`

For plotting:
- `matplotlib`

A minimal setup is:

```bash
pip install torch numpy tiktoken datasets matplotlib
```

If you train on a GPU, install a PyTorch build compatible with your CUDA setup.

---

## Data format

Training expects tokenized binary files in `uint16` format:

```text
data/<dataset>_train.bin
data/<dataset>_val.bin
```

The training script constructs paths as
`<data_dir>/<dataset>_train.bin` and `<data_dir>/<dataset>_val.bin`,
so the dataset name passed to `--dataset` must match the filename prefix.

Supported dataset names in `train.py` are currently:
- `tinystories`
- `openwebtext`

---

## Preprocessing

### TinyStories

This downloads `roneneldan/TinyStories` via Hugging Face `datasets`, tokenizes with the GPT-2 BPE from `tiktoken`, appends the end-of-text token, and writes:

```text
data/tinystories_train.bin
data/tinystories_val.bin
```

Run:

```bash
python preprocess_tinystories.py --out_dir data
```

For a smaller smoke test:

```bash
python preprocess_tinystories.py \
  --out_dir data \
  --max_docs_train 1000 \
  --max_docs_val 200
```

### OpenWebText

This downloads the single `train` split of `Skylion007/openwebtext`, uses a deterministic validation split controlled by `--val_fraction`, tokenizes with GPT-2 BPE, and writes:

```text
data/openwebtext_train.bin
data/openwebtext_val.bin
```

Run:

```bash
python process_openwebtext.py --out_dir data --val_fraction 0.005
```

For a smaller smoke test:

```bash
python process_openwebtext.py \
  --out_dir data \
  --val_fraction 0.005 \
  --max_docs_train 5000 \
  --max_docs_val 500
```

---

## Training

### Minimal example

After preprocessing TinyStories:

```bash
python train.py \
  --data_dir data \
  --dataset tinystories \
  --arch baseline \
  --out_dir out \
  --plot
```

This creates a run directory inside `out/`, logs losses to `metrics.csv`, and saves checkpoints.

### Example: accelerated / presymplectic variant

```bash
python train.py \
  --data_dir data \
  --dataset tinystories \
  --arch presymp \
  --out_dir out \
  --n_layer 4 \
  --n_head 4 \
  --n_embd 64 \
  --block_size 128 \
  --batch_size 6 \
  --grad_accum_steps 12 \
  --max_steps 1000 \
  --eval_interval 100 \
  --eval_batches 20 \
  --peak_lr 3e-4 \
  --presymp_h 0.3 \
  --learn_h 1 \
  --learn_xi 1 \
  --eta_learnable \
  --eta_mode loglin \
  --eta_log_init 3 \
  --eta_lin_init 1e-4 \
  --eta_clip 12 \
  --plot
```

### Example: linear-attention variant

```bash
python train.py \
  --data_dir data \
  --dataset openwebtext \
  --arch lin_presymp \
  --out_dir out \
  --n_layer 4 \
  --n_head 4 \
  --n_embd 64 \
  --block_size 128 \
  --batch_size 6 \
  --grad_accum_steps 12 \
  --max_steps 1000 \
  --eval_interval 100 \
  --eval_batches 20 \
  --peak_lr 3e-4 \
  --presymp_h 0.3 \
  --eta_learnable \
  --eta_mode loglin \
  --eta_log_init 3 \
  --eta_lin_init 1e-4 \
  --eta_clip 12 \
  --plot
```

---

## Architecture options

The `--arch` flag in `train.py` currently supports:

### Softmax-attention family
- `baseline`
- `yurii_lt`
- `presymp`
- `presymp_euler`
- `presymp_exp_euler`
- `presymp_ab2`
- `presymp_etd_ab2`
- `presymp_strang`
- `plain_euler`

### Linear-attention family
- `lin_baseline`
- `lin_yurii`
- `lin_euler`
- `lin_presymp`
- `lin_exp_euler`
- `lin_ab2`
- `lin_etd_ab2`

The baseline path instantiates a GPT-style model. The remaining options select alternative attention / update-rule blocks implemented in `model.py`.

---

## Important training arguments

Common model-size arguments:

```bash
--n_layer
--n_head
--n_embd
--block_size
--vocab_size
--dropout
--bias
```

Common optimization/logging arguments:

```bash
--batch_size
--grad_accum_steps
--max_steps
--warmup_steps
--peak_lr
--min_lr_ratio
--eval_interval
--eval_batches
--log_interval
--out_dir
--run_name
--resume
--device
--plot
```

Sampling / qualitative inspection:

```bash
--sample_interval
--sample_max_new_tokens
--sample_prefix_tokens
--sample_prompt
--sample_temperature
--sample_top_k
--sample_do_sample
--sample_eos_token_id
```

Presymplectic / accelerated settings include, among others:

```bash
--presymp_h
--presymp_xi
--presymp_t0
--learn_h
--learn_xi
--eta_learnable
--eta_mode
--eta_log_init
--eta_lin_init
--eta_clip
--scalar_lr_mult
--no_mlp
```

For the exact and most up-to-date list, run:

```bash
python train.py --help
```

---

## Outputs

Each run is written to a per-architecture subdirectory:

```text
<out_dir>/<arch>/
```

or, if `--run_name` is set:

```text
<out_dir>/<arch>_<run_name>/
```

Typical outputs are:

```text
metrics.csv
best_<arch>.pt
final_<arch>.pt
loss.png            # when --plot is enabled
```

`metrics.csv` contains training and validation losses, along with additional run metadata such as learning rate, wall-clock time, cumulative tokens, and, for some architectures, learned step-size/damping statistics.

---

## Comparing runs

Use `plot_compare.py` to compare several training runs via their `metrics.csv` files.

Example:

```bash
python plot_compare.py \
  --runs out/baseline out/yurii_lt out/presymp \
  --labels baseline yurii presymp \
  --xaxis step \
  --out compare_loss.png
```

Available x-axes are:
- `step`
- `wall`
- `tokens`

The script can also emit a LaTeX summary table:

```bash
python plot_compare.py \
  --runs out/baseline out/presymp \
  --labels baseline presymp \
  --xaxis wall \
  --out compare_wall.png \
  --latex_table \
  --latex_caption "Validation loss comparison."
```

---

## SLURM scripts

The repository includes:
- `job.job` for softmax-attention comparisons,
- `job_lin.job` for linear-attention comparisons.

These are examples, not generic launchers. You will likely need to adapt:
- working directories,
- Conda environment name,
- dataset choice,
- resource requests,
- output directory names,
- model and optimizer hyperparameters.

---


## Acknowledgements

This code uses GPT-2 tokenization via `tiktoken` and dataset loading via Hugging Face `datasets`.
