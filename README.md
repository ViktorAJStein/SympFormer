# SympFormer

Reproducible code for geometry-inspired, strictly causal language-model updates.

## Current paper experiments (arxiv_v4)

**Start with [`paper_v4/README.md`](paper_v4/README.md).** This is the implementation
map for the current manuscript's *Discretizations and numerical results* and
*Implementation details* sections (synchronized October 5, 2026).

The paper compares:

- **B:** native residual attention/MLP Transformer, not a momentum control;
- **Y:** Yuriiformer with learned initial velocity and attention/MLP look-ahead;
- **H0:** attention look-ahead disabled, MLP look-ahead retained;
- **H1:** coordinate connection-proxy correction;
- **R2:** half-corrected position and fully corrected velocity, both using the
  shared velocity LayerNorm;
- the explicitly separate attention-look-ahead variants and signed-linear
  attention counterparts.

The implemented connection is a **causal multi-head proxy**, not an established
Levi--Civita connection of the complete normalized decoder. R2 does not imply
second-order accuracy of the full layer or exact symplecticity.

### Quick start

Use [uv](https://docs.astral.sh/uv/) for the environment:

```bash
uv sync --frozen
uv run python paper_v4/verify.py --out /tmp/sympformer-v4-checks
uv run python paper_v4/run.py --list
```

Generate GPT-2-tokenized `uint16` data with manifest sidecars, outside the repo:

```bash
uv run python preprocess_tinystories.py --out_dir "$DATA_DIR" --seed 1337
uv run python process_openwebtext.py --out_dir "$DATA_DIR" --seed 1337 --val_fraction 0.005
```

Preview a paper run, then add `--execute` to train in your allocated environment:

```bash
uv run python paper_v4/run.py \
  --config configs/softmax_primary/R2_tinystories_s70020.json \
  --data-dir "$DATA_DIR" --out-dir "$OUTPUT_DIR" --device cuda
```

Presets use FP32 without TF32 and AdamW, not the older root-level bfloat16/Muon
campaign defaults. All paper studies here are approximately **100M-token
scale-local diagnostics**. Equal tokens do not mean equal parameters or compute.
The completed width256/100M tied-versus-untied looping study is included in the
latest v4 section. The prospective 1B-token campaign is separate and is not
presented as a completed paper result here.

## Results and interpretation

At width64/context128/five seeds, softmax R2 has the lowest descriptive mean NLL
on both datasets. Its matched H0 differences are **not significant under the
reported paired tests**; improvement over full Y alone is not isolated correction
attribution. In the context512/three-seed linear study, R2 improves on the linear
residual baseline on OpenWebText, but neither correction passes the original
Holm20 family against H0. See the protocol-specific tables and limitations in
[`paper_v4/README.md`](paper_v4/README.md); never pool these studies.

## Repository layout

| Path | Purpose |
|---|---|
| `paper_v4/` | Reviewed frozen runtimes, explicit presets, correctness checks and plotting for current paper experiments |
| `model.py`, `train.py`, `data.py` | Earlier general experimental implementations; not the v4 protocol entry point |
| `scripts/` | Earlier verification/analysis/particle utilities |
| `configs/` | Earlier pilot/core presets; not interchangeable with `paper_v4/configs/` |
| `particle_schemes.py`, `particle_acceleration_benchmark.py` | Particle-system diagnostics, not decoder benchmarks |

The v4 runtimes are isolated so older checkpoints and experimental paths remain
available without silently changing their meaning. The snapshots share byte-identical
core modules through relative symlinks. Use a symlink-capable checkout (Linux/macOS).
Historical global-Hamiltonian, PE and old linear-discretization routes are distinct
from the architectures and protocols documented for v4.

Checkpoints are trusted pickle inputs: load only your own or independently trusted
files. Datasets, checkpoints, raw runs, logs, research notes, manuscripts and job
scripts are not distributed as part of this update. No cluster credentials or
cluster-specific launch instructions are required by the paper entry point.
