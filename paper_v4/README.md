# Reproducing the arxiv_v4 numerical section

This directory matches the synchronized manuscript's numerical and implementation
sections as of October 5, 2026 (Overleaf commit `b7cd2ff`). It contains **code and
configurations**, not manuscript sources, raw results or cluster job definitions.
Runtimes are frozen historical implementations, with identical shared core bytes;
relative symlinks remove duplication. `protocols.json` records their SHA-256 values
and every preset. Source/config checks fail if these reviewed bytes change.

## Architecture mapping

| Paper | Runtime route | Meaning |
|---|---|---|
| B / GPT equation | `softmax`: `baseline` | Native residual attention and MLP, no velocity |
| Y / Yuriiformer equation | `softmax`: `normalized_hb`, Y | Learned token/position initial velocity; both look-aheads |
| H0 | `softmax`: H0 | No attention look-ahead; unchanged look-ahead MLP |
| H1 / coordinate equation | `softmax`: H1 | Proxy evaluated on incoming displacement |
| R2 / dual-LN equation | `softmax`: R2 | Proxy evaluated on proposed displacement; coefficients 1/2 and 1 |
| H1-LA, R2-LA | `softmax`: H1_LA, R2_LA | Attention/proxy at LN(X + mu_A V), raw direction |
| Learned correction strength | `strength`: H1/R2 with `hb_eta_learnable` | Positive per-layer eta, initially exactly 1; separate diagnostic |
| Shared/separate B,Y,H0,R2 | `looped`: tied/untied | Two unique blocks repeated four times versus eight alternating independently trainable copies; continuous velocity |
| B_lin, Y_lin, H0_lin, H1_lin, R2_lin | `linear`: B_L/Y_L/H0_L/H1_L/R2_L | Signed causal prefix-linear force and fixed-ridge proxy |

All momentum variants retain the learned initial velocity, unchanged MLP rule and
shared velocity LayerNorm. H0 is **not** B. Raw-position R2 is **not** dual-LN R2.
The root `yurii_lt` zero-initial-velocity route is not a substitute for paper Y.
The look-ahead, correction-strength and model-shape studies remain separate.

The added velocity embeddings dominate the parameter difference, but are not its
only contribution. Let d denote width, T maximum context length, and V vocabulary
size. At the primary64d/context128 shape, d(T+V)=3,227,648 velocity
embedding parameters are accompanied by280 (Y) or276 (H0/H1/R2) additional
velocity-LayerNorm/scalar parameters. Total counts are B3,424,832; Y6,652,760;
H0/H1/R2 each6,652,756. These small extra terms are included in all reported
parameter counts.

For softmax, `attention_connection` implements the manuscript's head-averaged
score-direction contraction and weighted-value term, with B=I. It uses the raw
direction at the normalized base point: no LN/look-ahead Jacobian is inserted in
the proxy. Autodiff still differentiates the actual model. R2 applies the same
velocity LN to W-C/2 for position and W-C for carried velocity; H1 uses C(V,V).
This is geometry-inspired, not an exact Hamiltonian/symplectic decoder or an
established full-model Levi--Civita connection.

For linear attention, q and k are scaled by head_dimension^(-1/4), and the output
at token i uses the signed prefix sum divided by i. It is **not** softmax or a
positive-feature normalization. The proxy uses factorwise ridge lambda=0.001,
prefix Gram Cholesky solves and the retained factor L=I-lambda*(K^T K+lambda I)^(-1).
Chunk512 and exact checkpoint recomputation change execution, not the equations.
Fits are sequence/layer/place-local; there is no assumed canonical-momentum
closure or LN pullback. `linear/scripts/linear_hb_model.py` is the implementation.

## Exact protocol presets

`uv run python paper_v4/run.py --list` enumerates every cell.

| Preset directory | Cells | Shape / seeds | Purpose |
|---|---:|---|---|
| `softmax_primary` | 50 | 4L/4H/64d/context128; five seeds/dataset | B,Y,H0,H1,R2 primary tables |
| `softmax_lookahead` | 32 | 4L/4H,64d or256d/context128; five or three seeds | H1-LA/R2-LA; separate Holm16 family |
| `softmax_width256` | 24 | 4L/4H/256d/context128; three seeds/dataset | B,Y,H0,R2 controls; adverse correction evidence retained |
| `softmax_strength` | 12 | 4L/4H/256d/context128; three seeds/dataset | H1/R2 learned eta; not fixed-strength outcomes |
| `softmax_shapes` | 120 | Width512; eight head/context/depth shapes; three OWT seeds | B,Y,H0,H1,R2; separate utility Holm16 and interaction Holm14 |
| `softmax_looped` | 48 | 8logical applications/8H/256d/context512; three seeds/dataset | B,Y,H0,R2 × tied/untied; original Holm14 |
| `linear_primary` | 30 | 4L/4H/64d/context512; three seeds/dataset | Five within-linear methods |
| `linear_statistical_controls` | 12 | Same linear-study shape/seeds/validation | B_S/Y_S controls needed for the original Holm20 family |

Looped training uses two **unique** blocks applied four times, not eight unique
trainable layers. Untied controls start as eight alternating copies of those same
two blocks, then train independently. Embeddings/readout remain shared within the
model; velocity is initialized once and is never reset between loops. Gradients
flow through all eight applications and accumulate onto shared parameters. No
training-depth adaptation, detached recurrence or checkpoint/depth selection is used.

Every preset trains 6,103 updates / **99,991,552 actual tokens** with 16,384
loss-bearing tokens/update. Nominal 100M is rounded down to whole updates. AdamW:
peak LR0.0006, scalar multiplier10, betas(0.9,0.95), inherited embedding decay0.1,
global clipping1, 10% warmup, cosine floor0.1; dropout0, biasFalse, vocabulary50304.
The recorded cohort is RTX A6000, Python3.11.13/Torch2.13.0+cu130, FP32/TF32 off.
The repository uv environment supports correctness/running the code; reproducing
historical GPU numerics/timings additionally requires that recorded cohort/data.

**Do not unify the samplers or evaluation indices.** Historical softmax validates
at logged step100,200,...,6100, then final6103 (62events, no initial event). Those
intermediate labels describe 101,201,... completed updates, so plotting uses
`tokens_cum/tokens_step`. Primary contexts128 use batch8/accum16/eval32. The
shape study uses the shared 512-token macro sampler to match flattened targets
across contexts128/256/512; its batch8 accum16/8/4 and eval32/16/8 are explicit.
Linear uses batch4/accum8/eval16, initial0 then completed100,200,...,6100,6103
(63events), with common validation seed95019 (iterator's split offset retained).
Changing these historical conventions changes the experiment.

Primary softmax seeds are70020--70024 (TinyStories) and71020--71024 (OWT).
Width256/strength use the first three respective seeds; shapes use94020--94022;
linear uses95020--95022. Paired methods share data/batch and validation policies.
The historical look-ahead family contains16 tests, the shape baseline family16,
and linear20 (including eight softmax utility contrasts **not displayed** in the
paper's within-linear table). The twelve `linear_statistical_controls` presets
supply these B_S/Y_S runs; they have context512 and are not replacements for the
primary context128 softmax study. **Do not recompute Holm over only displayed rows.**
The softmax primary paired table reports nominal two-sided Student-t tests;
significance against Y does not establish correction attribution against H0.

## Running

```bash
uv run python paper_v4/verify.py --out /tmp/sympformer-v4-checks

# Dry-run command preview (no allocation/submission):
uv run python paper_v4/run.py \
  --config configs/softmax_primary/R2_tinystories_s70020.json \
  --data-dir "$DATA_DIR" --out-dir "$OUTPUT_DIR" --device cuda

# Explicit execution in your allocated single-GPU environment:
uv run python paper_v4/run.py \
  --config configs/linear_primary/e062_R2_L_openwebtext_s95020.json \
  --data-dir "$DATA_DIR" --out-dir "$OUTPUT_DIR" --device cuda --execute
```

Data files are `<dataset>_{train,val}.bin` (`uint16`) with validated manifest
sidecars. Preprocessed token counts: TinyStories473,992,236 train /4,765,918 val;
OWT8,995,237,948 train /44,779,147 val. These are corpus sizes, not training budgets.
Use the root preprocessing utilities; retain exact data fingerprints and splits.
Data and outputs belong outside the source checkout. Final checkpoint evaluation,
not best validation or seed selection, defines every result. `--resume` is for
trusted checkpoints with the same full-horizon preset, never a shortened schedule.

## Verified paper summary (not new training)

Primary softmax final NLL, mean (sample SD), five seeds:

| Method | TinyStories | OpenWebText |
|---|---:|---:|
| B |2.659274 (0.043485)|5.394921 (0.037273)|
| Y |2.574789 (0.042433)|5.267365 (0.040739)|
| H0 |2.578203 (0.041753)|5.257956 (0.042541)|
| H1 |2.572989 (0.039143)|5.258599 (0.040714)|
| R2 |2.571951 (0.035781)|5.255397 (0.039610)|

R2-H0: Tiny−0.006253 (nominal p0.187), OWT−0.002559 (p0.515); no isolated
correction discovery. R2-Y on OWT−0.011969 (nominal p2.09e-4). Corrected methods
match H0's parameter count, **not** its cost: H1 costs roughly44--51% more,
R2 roughly57--66% more. No equal-compute advantage is claimed.

Within-linear final NLL, mean (sample SD), three seeds:

| Method | TinyStories | OpenWebText |
|---|---:|---:|
| B_L |3.085549 (0.013943)|5.701846 (0.001999)|
| Y_L |2.954004 (0.010361)|5.571286 (0.000705)|
| H0_L |2.962801 (0.030644)|5.574589 (0.004144)|
| H1_L |2.968827 (0.031537)|5.573876 (0.023693)|
| R2_L |2.939952 (0.035512)|5.553947 (0.007287)|

R2_L-B_L on OWT−0.147898,3/3wins, original Holm20 p0.018086. Neither correction
beats H0_L under Holm20; R2_L-H0_L OWT p0.220922 despite its nominal CI excluding
zero. R2_L costs roughly4.3--4.5x H0_L at equal tokens. Different context lengths
and seed counts prohibit pooling primary softmax and linear results. At width256,
fixed R2 loses all six paired cases to no-attention-look-ahead Y (H0); learned
strength intervals contain zero. Shape-study correction/B differences do not pass
Holm16. These adverse outcomes must not disappear from an overview.

These are scale-local100M diagnostics, not general1B/large-model conclusions.
The newly added looping table is the completed **256d/100M** diagnostic, not the
prospective1B campaign. Its three-seed means are:

| Method | Tiny separate /looped | OWT separate /looped |
|---|---:|---:|
| B |1.7978 /1.9922|4.6987 /4.9845|
| Y |1.8084 /1.9752|4.7104 /4.8744|
| H0 |1.8186 /1.9936|4.7159 /4.8750|
| R2 |1.7921 /1.9548|4.6642 /4.8398|

All24paired tying changes worsen NLL. Only tied OWT R2-H0, R2-B and R2-Y survive
the original Holm14 family (p0.044675/0.000951/0.020372). There is no established
tying interaction, TinyStories/untied discovery or tying-superiority claim. Tied
R2/H0 each have27,592,458parameters; native B tied has14,583,040. Equal tokens
again do not mean equal parameters or cost. Training/primary reporting always use
four loops; any frozen-checkpoint extra-depth diagnostic is secondary, never a
selected endpoint. Looped validation has63events0,100,...,6100,6103 and seed97018;
training seeds97020--97022. The separate1B campaign remains excluded. The latest
draft's temporal-coherence/recurrent-compute-efficiency ideas are hypotheses,
not demonstrated acceleration or beneficial extra-depth results.

## Files and outputs

`common/` holds identical frozen core code and its original dependency fingerprints;
`softmax/`, `aspect/`, `strength/`, `linear/` hold distinct frozen trainer/extension
versions. `looped/` has the unchanged frozen model and training-update body, with
only the private Slurm authorization gate removed for standalone public execution,
physical manifest verification added, and the unchanged passive Observer extracted.
Legacy `quality_run`/approval telemetry indicates private release authority, not a
public authorization. Resume verification excludes only positive observer optimizer
wall-clock samples, never model/optimizer/RNG/iterator/metric/trace/sample-chain state. `checks/` implements independent equation/gradient/null/causality tests.
The LS helper in linear is extracted without changing its numerical functions;
private checkpoint-discovery plumbing is excluded.

Each run writes `run_manifest.json`, `metrics.csv`, `summary.json`, trusted
checkpoints and (where applicable) `hb_dynamics.csv` / sequence-local diagnostics.
For unsmoothed log-y NLL and all-layer damping/reference plots with PDF+PNG and
raw plotted CSV:

```bash
# PLOT_RUNS contains only one paired dataset/shape block:
uv run python paper_v4/plot.py --runs "$PLOT_RUNS"/* --out "$ANALYSIS_DIR/figures"
# Complete42-run linear family, never only the30displayed cells:
uv run --with scipy python paper_v4/statistics.py --family linear \
  --runs "$RUNS"/* --out "$ANALYSIS_DIR/linear_statistics.json"
# Complete48-run looping family (separate directory):
uv run --with scipy python paper_v4/statistics.py --family softmax_looped \
  --runs "$LOOPED_RUNS"/* --out "$ANALYSIS_DIR/looped_statistics.json"
# Complete50-run primary softmax family:
uv run --with scipy python paper_v4/statistics.py --family softmax_primary \
  --runs "$PRIMARY_RUNS"/* --out "$ANALYSIS_DIR/softmax_statistics.json"
```

The plot input must be one paired dataset/shape/protocol block; do not pass all
studies or both datasets. Statistical input must instead contain the complete
specified family across both datasets. `statistics.py` recomputes trusted audited
endpoint numbers, not a fresh strict scheduler/checkpoint audit. It intentionally
has no arbitrary family-size override or partial-study inference. Other look-ahead/shape/strength diagnostic
families remain separate and are not silently reranked by this utility. No checkpoints,
raw run archives, generated research images, cluster scripts or manuscript files
are shipped here. This update does not modify the older root runtime/checkpoints.
