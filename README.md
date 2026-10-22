# SympFormer — submitted-paper experiments

Minimal code to repeat the **three numerical tables in the submitted
`arxiv_v4.tex`**, identified by Overleaf commit
`c9ce8f42152e942b8ed80f55f480677ab5c0a04e` (October 22, 2026).
The manuscript is maintained separately; this repository contains no manuscript,
private research records, cluster controllers, results or checkpoints.

| Study (`--study`) | Methods | Datasets | Seeds | Cells |
|---|---|---|---:|---:|
| `softmax` | B / Y / H0 / H1 / R2 | TinyStories + OpenWebText | 5 each | 50 |
| `sharing` | Same five, tied + untied | Both | 10 each | 200 |
| `linear` | Same five, signed prefix-linear attention | Both | 3 each | 30 |
| `sigmoid` | Same five, normalized sigmoid | Both | 3 each | 30 |
| `native_l2` | Same five, native-L2 SPD proxy | Both | 3 each | 30 |

Every cell trains **99,991,552 tokens**. The deferred 1B-token study and unreported
look-ahead/width/strength/shape/particle experiments are deliberately excluded.
**H0 is not vanilla B.** These are causal, geometry-inspired decoders, not exact
Hamiltonian solvers. Benefit and cost are conditional; no universal correction,
sharing-amplification or equal-compute advantage is implied.

- [Exact protocols and reproduction limits](docs/REPRODUCIBILITY.md)
- [Connection proxies: softmax, sigmoid, signed linear and native-L2](docs/CONNECTION_PROXIES.md)
- [Numerical source lineage](docs/SOURCE_PROVENANCE.json)

## Setup and data

Use `uv`; the checked-in lock supplies dependencies, including PyTorch2.13.0.
The historical CUDA cohort additionally used Python3.11.13 and
PyTorch2.13.0+cu130; see the protocol document before comparing GPU timings.

```bash
uv sync --frozen --python 3.11
uv run python -m unittest discover -s tests -v

# Use storage outside this checkout. These commands download full datasets.
uv run python preprocess_tinystories.py --out_dir "$DATA_DIR" --seed 1337
uv run python process_openwebtext.py --out_dir "$DATA_DIR" --seed 1337 --val_fraction 0.005
```

Training verifies the paper's four corpus SHA-256/size records in
`configs/data_fingerprints.json` plus preprocessing manifest sidecars. A changed
upstream dataset is a mismatch, not permission to silently use a new split.
Data are not redistributed. This repository does not promise that a mutable
upstream dataset revision will always regenerate the historical bytes.

## Enumerate, train and resume

```bash
# Exactly340 cells, no scheduler actions:
uv run python run.py --list
uv run python run.py --study sharing --index 0       # protocol preview only

# Explicit execution; one cell on one allocated GPU:
uv run python run.py --study sharing --index 0 --execute \
  --data-dir "$DATA_DIR" --out-dir "$RUNS/sharing/0" --device cuda

# Resume only a trusted own checkpoint into a NEW directory, with the SAME preset:
uv run python run.py --study sharing --index 0 --execute \
  --data-dir "$DATA_DIR" --out-dir "$RUNS/sharing/0-resumed" --device cuda \
  --resume "$RUNS/sharing/0/checkpoint.pt"
```

Choose the proper number of cells from the table above. There are no cluster
paths, resource submissions or automatic retries in the code. Run scheduling and
compute authorization are your responsibility. Use paired hardware/software
cohorts and one GPU per independent cell; do not split methods across GPU types.
Do not include both an interrupted and its resumed replacement as two seeds.
Only a complete full-horizon run supplies a final NLL; best checkpoints do not.

Outputs: `run_manifest.json`, `metrics.csv`, `damping.csv`, `summary.json` and
trusted model/optimizer/RNG/iterator checkpoint state. Incomplete/failed runs are
not silently averaged. CPU fixtures are correctness tests, not paper results.

## Aggregate and plot

```bash
# Each directory must contain the COMPLETE study, both datasets and all seeds:
uv run python analyze.py --study sharing --runs "$RUNS/sharing" --out "$ANALYSIS/sharing"
uv run python plot.py --study sharing --runs "$RUNS/sharing" --out "$ANALYSIS/sharing-plots"
```

Analysis reports mean/sample SD, parameters and peak memory. GPU hours require
an external per-run `allocation.json` containing `gpu_count`, `allocated_seconds`
and `gpu` (the GPU name in the run manifest). **Trainer elapsed time is not a
substitute for allocated GPU time.** Without complete allocation records, GPUh
is reported as unavailable. No table result is hardcoded as training output.

Plots use exact common evaluation iterations, log-y NLL, seed variability and
all logical layers' coefficients/gains/unit steps/nominal damping. They export
PDF+PNG and raw/plotted CSV, with descriptive fitted `r/t+c` versus `3/t`.
Native B damping is marked not applicable. No smoothing, interpolation, best-seed
selection or post-hoc significance tests are performed.

## Repository scope

`configs/` contains the five exact scientific matrices and corpus fingerprints;
`sympformer/` contains only the required causal models and training primitives;
`tests/` checks equations, gradients, causality, parameter counts and exact tiny
CPU interruption/resume; the two preprocessing utilities prepare the data.
`run.py`, `analyze.py`, and `plot.py` are portable public orchestration, not the
private historical scheduler/auditor. Original model/proxy definitions are
preserved; file schemas and resource bookkeeping are new. A fresh full GPU
reproduction has **not** been performed for this release. Timing remains
hardware/storage-dependent and must be measured, not copied from the paper.
