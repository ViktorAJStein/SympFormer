# Reproducing the arXiv paper

These protocols reproduce the numerical section and implementation appendix of
[SympFormer, arXiv:2610.31600v1](https://arxiv.org/abs/2610.31600v1), submitted
October 22, 2026, by Viktor Stein, Levin Maier, Wuchen Li and Gabriele Steidl.
The numerical source snapshot is Overleaf commit
`c9ce8f42152e942b8ed80f55f480677ab5c0a04e`; its provenance is retained in
[`SOURCE_PROVENANCE.json`](SOURCE_PROVENANCE.json).
The 1B-token addition is deferred and absent from arXiv v1.
Configuration indices are local to the selected JSON matrix; `run.py --list`
shows the exact method/dataset/seed/layout mapping. There are340 distinct cells,
not a newly authorized campaign.

## Table coverage

| Manuscript table label | Presets | Model shape | Training seeds | Validation seed |
|---|---|---|---|---|
| `tab:softmax_fiveway_uncertainty` | `configs/softmax.json`,50cells | 4layers,4heads,64width,128context | TS70020–70024; OWT71020–71024 | run seed (iterator adds split offset1) |
| `tab:separate-vs-shared` | `configs/sharing.json`,200cells | 8logical applications,8heads,256width,512context | 105120–105129, both datasets | 105119 (iterator adds1) |
| `tab:other_attention_endpoints`, linear rows | `configs/linear.json`,30cells | 4layers,4heads,64width,512context | 95020–95022 | 95019 (iterator adds1) |
| same table, sigmoid rows | `configs/sigmoid.json`,30cells | same 4/4/64/512 | 100120–100122 | 100119 (direct random-start generator) |
| same table, native-L2 SPD rows | `configs/native_l2.json`,30cells | same 4/4/64/512 | 100120–100122 | 100119 (direct random-start generator) |

Each study contains B,Y,H0,H1,R2 on TinyStories and OpenWebText. Sharing has
both layouts. It is the **ten-seed five-arm** study, not the old three-seed
four-arm looping experiment. Independent matrices, seeds and attention families
must not be pooled or replaced by whichever historical source is easiest to run.
No results from E064/1B, higher-depth/width/learned-strength/look-ahead, global
noncausal or particle diagnostics belong to these tables.

## Models and initialization

All projections/MLPs are bias-free, with zero dropout. Width-d MLPs have hidden
width4d and GELU. Position LN, shared attention/MLP velocity LN, final token LN
and tied token-embedding/output-head weights use epsilon10^-5. Learned scalar
parameterizations and initialization are the extracted native definitions:
beta=sigmoid(raw), gain=softplus(raw)+10^-6, enabled look-ahead mu=sigmoid(raw);
initial beta=.9, gain=1, mu=.9. H0/H1/R2 remove the attention-look-ahead parameter,
retain the MLP look-ahead and have equal parameter counts. B is a native
residual Transformer and has no velocity embedding/stream. H1/R2 do not correct
its architecture in place.

Momentum models have separate learned token+position velocity embeddings. They
are initialized in that order from a dedicated CPU generator seeded with
`run_seed + 10_000_019`, standard deviation.02, without consuming the global
training RNG. Other embedding/linear weights have standard deviation.02 and
LayerNorm scales start at1. The original factory/donor construction order is
preserved: softmax constructs the native model directly; linear and nonlinear
factories copy matched native Yurii donor tensors; sharing constructs two donor
blocks and deep-copies them into2 or8 unique slots. Do not unify those factory
orders: that would change historical initialization.

Both sharing layouts execute exactly eight logical blocks, with continuous
state/velocity streams and no repeated input injection or detaching. Tied
modules receive accumulated gradients from repeated uses. The untied blocks
start as alternating copies of the same two initialized donors, then train
independently. Embeddings, final LN and output head are not duplicated.

## Sampling, update arithmetic and evaluation

All cells use6103 optimizer updates ×16384 loss-bearing tokens =99,991,552
actual tokens, not exactly100,000,000. Primary softmax uses batch8 ×context128
×accumulation16; other tables use batch4 ×context512 ×accumulation8.
Validation uses32768 fixed targets/event, with32 or16 batches respectively.

**Block-epoch studies:** `BlockEpochIterator` visits shuffled non-overlapping
blocks of the concatenated uint16 stream. A seed-specific random boundary
offset is drawn each epoch; invalid final windows are excluded. Its independent
NumPy generator uses the configured seed plus1 for validation. Validation
restores iterator state so the same targets recur at every evaluation.
The linear and sharing studies have initial update0, then100,200,…,6100 and
final6103:63events. The historical primary softmax loop evaluated after updates
101,201,…,6101, then6103:62events, **no initial evaluation**. Old logs labeled
those intermediate events100,200,…,6100; public CSV labels the actual completed
update count. This label correction is not interpolation or a changed schedule.

**Sigmoid/native-L2 studies:** training starts are sampled with replacement by
`default_rng(SeedSequence([run_seed, one_based_update, 1068]))`; draw
batch×accumulation valid starts, then split into consecutive microbatches.
Validation starts use a separate direct `default_rng(validation_seed)` draw,
unchanged across events. Evaluation is at0,100,…,6100,6103. Keep this sampler
separate from the block-epoch studies.

AdamW uses betas(.9,.95), peak LR.0006, scalar-group multiplier10, embedding-only
decay.1, no decay for matrices/LayerNorm/scalars, global gradient clipping1,
warmup610 updates and cosine floor.1. The original block-epoch runners use
zero-based LR steps0…6102, while the stateless nonlinear runner uses one-based
steps1…6103; this convention is explicit in each preset. Training autograd is
active through all proxy operations; no extra geometry parameters are optimized.

`evaluation_updates` retains these distinct conventions. Best checkpoints are
never selected as endpoints. Interrupting/resuming retains the full LR horizon,
RNG states, batch iterator positions and histories; a shortened-horizon run
cannot be resumed as if it had followed the original schedule.

## Data identity

Preprocessing is GPT-2 BPE via tiktoken `encode_ordinary`; append EOT50256 per
document; padded vocabulary50304; store uint16. Windows can cross document
boundaries, with no extra document mask. TinyStories uses supplied train/val
splits. OpenWebText uses99.5%/0.5% shuffled document split, preprocessing seed1337.
The preprocessing utilities create manifest sidecars. Historical corpus tokens:
TS473,992,236 train /4,765,918 val; OWT8,995,237,948 train /44,779,147 val.
The corresponding four exact sizes/digests are shipped in
[`data_fingerprints.json`](../configs/data_fingerprints.json).

No corpus is redistributed. Upstream Hugging Face content/API availability can
change; neither a guessed dataset revision nor a silent replacement is provided.
Reproducing the historical corpus requires matching these digests. Training
fails on mismatched data instead of silently changing the evaluation.

## Hardware, costs and public-orchestration boundary

Historical softmax, sharing and linear cells used RTX A6000. For sigmoid and
native-L2, TS used A100 and OWT RTX A6000. Python3.11.13,
PyTorch2.13.0+cu130, IEEE FP32, TF32 disabled; paired blocks use one matched
hardware/software cohort. The portable uv lock provides the dependencies, not
an assertion that every OS/CUDA-driver combination is bitwise equivalent.
Original driver versions/cohort context were610.57.04 on TU A6000 and
580.173.02 on the E068 A100 route; do not treat these as a portable requirement
that can simply be substituted for a supported local driver.

Public orchestration is intentionally independent of private scheduler budgets,
keys, leases, source-adoption controllers and historical absolute paths. It
preserves the numerical model/optimizer/sampler/schedule, but its telemetry,
checkpoint schema and validation gates are new. This release was checked with
bounded CPU fixtures and frozen-source parity, **not a new340-run GPU study**.

The public runner retains historical expensive checkpoint event patterns:
softmax/linear save improved validation checkpoints plus the fixed final state;
sharing checkpoints every1000 updates and final, with physical readback;
nonlinear studies checkpoint at every validation event, including0/final, with
physical readback. It retains one latest portable resume state; it does not
recreate every historical file layout, growing observer payload or Slurm/I/O
wrapper. These differences and storage/driver/load affect timing and peak memory.
Therefore GPUh/memory must be measured anew; identical paper timings are not
promised. `elapsed_seconds` is trainer wall time, **not** allocated GPU time.

For analysis, provide each run's real scheduler `allocation.json`, for example
`{"gpu_count":1,"allocated_seconds":1234,"gpu":"NVIDIA RTX A6000"}`.
Do not infer this number from the final loss, a forecast or device-independent
FLOPs. Missing allocation evidence yields unavailable GPUh, not a fabricated
value. Any failed/retried attempts and technical qualification costs should be
reported separately as well as the paper's per-completed-run convention.

`analyze.py` requires all declared cells for a study and rejects duplicate,
changed, incomplete, failed, mixed-data/source/cohort or smoke-test records.
It reports full-precision outputs and rounds only printed NLL/SD/GPUh to four
decimals. `plot.py` requires exact shared evaluation rows, log-y NLL and seed
variability; all logical layers, forward steps and nominal rates are retained.
Inverse-time fits are descriptive and raw CSV is kept. These descriptive tables
do not authorize redefining any historical Holm family over displayed subsets;
no new post-hoc significance tests are included in the minimal release.

## Validation commands and limits

```
uv run python -m unittest discover -s tests -v
uv run python run.py --list
```

The public tests check the complete340-cell matrix, actual reported parameter
counts,25 tiny model/gradient/causality routes, softmax/sigmoid formulas,
linear-ridge prefix contractions, the native-L2 tensor derivative/self-only VJP,
five tiny exact interruption/resume routes and adverse input/data domains.
These are finite CPU checks, not a mathematical convergence theorem, independent
scientific replication or evidence about unreported model variants.
