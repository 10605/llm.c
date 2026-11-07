# Learning-rate transfer with μP

The exercise asks two different questions: does a learning rate selected on a
narrow transformer work on a wider transformer, and does its preferred value
change when training lasts longer? Use **final held-out validation loss** to
select the best tested LR. Keep training losses as diagnostics.

## Setup

```sh
git clone --branch mup https://github.com/10605/llm.c.git
cd llm.c/HW5_Code
python3 -m venv .venv
source .venv/bin/activate
python -m pip install modal matplotlib 'numpy<2' requests
git fetch https://github.com/GatiAher/llm.c.git 706b1a5fc94de4bc16a4a3684eafaad6b9e0dada
```

The pinned baseline is needed by the comparison tests and Modal source packager.
The exercise code runs from `HW5_Code/`; both standard parametrization and μP
use the same trainer. Local data, credentials, checkpoints and experimental
results are not included in this release.

The Modal commands below launch paid GPU work in your account after `modal setup`.
Agree on a class compute budget before launching. They build CUDA remotely,
download GPT-2-tokenized FineWeb and the tokenizer, and retain data/source hashes,
commands and raw logs. A sweep uses four independent A10 workers. The horizon
experiment uses five A100-40GB workers. Run one experiment at a time.

For a local NVIDIA GPU, build the same training executable with:

```sh
make train_gpt2cu PRECISION=BF16 USE_CUDNN=0 NO_MULTI_GPU=1
./train_gpt2cu --help
```

## Course preset and student knobs

The CUDA course preset is on by default (`-cp 1`). **Supply `-l LR` on every
invocation, including resume. There is no course LR default.** Students choose
their expert-training LR; the seed's historical LR is not built into the trainer.
Fresh models use 12 layers, width 768 and 12 heads. μP reference width 192 and
attention multiplier 4 are automatic; saved checkpoints restore their geometry
and parametrization. No μP flags are needed in ordinary student commands.

| Setting | Flag | Course default / use |
| --- | --- | --- |
| Peak base LR | `-l` | Required, chosen by the user |
| Output / checkpoint directory | `-o` | Unset; use a separate directory per trial |
| Full schedule in updates | `-x` | 21838; explicit `-x -1` uses one dataset epoch |
| Absolute stopping update | `-es` | -1 = finish; preserves the full schedule |
| Full checkpoint resume | `-y` | 0; use 1 to resume the latest complete checkpoint in `-o` |
| Reset training-data loader on resume | `-rd` | 0; use 1 only when switching datasets |
| Width (channels) / depth / heads | `-1c` / `-1d` / `-1h` | 768 / 12 / 12 for fresh models |
| Global tokens per update | `-d` | 114688 |
| Per-GPU microbatch / context | `-b` / `-t` | 8 sequences / 2048 tokens |
| Schedule / warmup / final fraction | `-k` / `-u` / `-q` | linear / -1 (automatic 10%) / 0 |
| AdamW timescale | `-wt` | 0.1; decay = 1 / (peak base LR × full updates × timescale) |
| Optional fixed decay | `-c` | Unset; replaces automatic decay; cannot combine with explicit `-wt` |
| Checkpoint interval / retention window | `-n` / `-nk` | 1000 / 2 intervals; requires `-o`; `-nk 0` keeps all |
| Validation interval / max microbatches | `-v` / `-m` | 1000 / 72; also validates at the stopping point |
| Initialization seed | `-se` | 42; ignored for loaded weights |
| Sampling / HellaSwag | `-s` / `-h` | 0 / 0 (disabled) |

Default paths, relative to the trainer's working directory:

- Training: `data/pretraining/climbmix_students_train_*.bin` (`-id`).
- Pretraining-mix validation: `data/pretraining/climbmix_students_val_*.bin` (`-iv`), supplied by HF and included in the AMI.
- Optional second validation: `-iv2 PATTERN`; expert runs use the cluster validation with `-iv` and pretraining-mix validation with `-iv2`. Both use the same `-v` interval and `-m` batch limit, sampled across the supplied HF validation split.
- Tokenizer: `data/gpt2_tokenizer.bin` (`-it`).

Prepare or download these files first. Quote shard globs. Set `-id` to your
assigned cluster; this does **not** change validation data. For specialization,
use the same held-out cluster validation file and token count across candidates.
At default batching, 72 validation batches cover up to 1,179,648 tokens.

Global batch must be divisible by `microbatch × context × GPU count`; the
single-GPU default accumulates seven microbatches per optimizer update. Keep
global batch fixed on resume. The full default schedule processes 2,504,556,544
tokens. It is a token budget, not necessarily one epoch of the selected dataset.
Warmup is 2183 updates; `-u -1` recomputes 10% (rounded down) for a different
full horizon, or zero for constant LR. Early stopping never shortens that horizon.
Explicit `-c` holds base decay fixed; otherwise changing LR or the full horizon
also changes derived decay to hold the timescale fixed.

### Minimal seed and expert commands

With the prepared default data paths, set `SEED_LR` to the instructor's chosen
peak LR and run:

```sh
mkdir -p runs
./train_gpt2cu -l "$SEED_LR" -o runs/seed -es 10919
```

This saves the halfway seed after 1,252,278,272 tokens, while keeping the full
21838-update LR schedule. An early stop writes a checkpoint after training when
`-n > 0` and `-o` is set, even between regular checkpoint intervals.

For an expert trial, set `LR` to the student's chosen peak LR and `TRAIN_GLOB`
to their assigned cluster (e.g. `data/cluster_01/climbmix_students_train_*.bin`).
`SEED_DIR` must contain the supplied complete seed; copy it to a **new** trial
directory, preserving the original seed:

```sh
mkdir -p runs/expert
cp "$SEED_DIR/model_00010919.bin" \
   "$SEED_DIR/state_00010919_00000.bin" \
   "$SEED_DIR/DONE_00010919" runs/expert/
./train_gpt2cu -l "$LR" -o runs/expert -y 1 -rd 1 -id "$TRAIN_GLOB"
```

Resume restores the saved step, weights, optimizer and RNG. `-rd 1` resets only
the training loader for the new dataset. The chosen LR scales the remaining
portion of the original schedule; warmup does not restart. Omit `-rd` when
resuming this same expert trial later. Repeat your LR and all nondefault schedule,
batch, decay and data options on every invocation; they are not restored from
checkpoint metadata. Defaults remain in effect when no override is supplied.

### Four experimental strategies

1. **Smaller models:** add `-1d 6 -1c 384 -1h 6`, for example, to a fresh run.
   Width and channels are the same setting. Channels must divide evenly by heads
   and be a multiple of 8 for BF16; backend and memory constraints also apply.
   These flags cannot resize the supplied seed. μP reference/attention settings
   remain automatic; the default head count stays 12 unless explicitly changed.
2. **Fewer updates, original schedule:** use `-x 21838 -es 2000`. This retains
   2183 warmup updates and stops during warmup. Stops are absolute: 1000 additional
   updates from the halfway seed means `-es 11919`.
3. **Smaller dataset, shorter complete schedule:** prepare a subset of shards
   and pass its glob with `-id`, then `-x -1 -u -1 -k linear -q 0`. There is no
   dataset-fraction flag. One epoch defines the horizon, with 10% warmup and decay
   to zero. Alternatively `-x 2000` defines a 2000-update schedule with 200 warmup
   updates. These examples start fresh: independently restarting the schedule
   clock while retaining the seed optimizer's counter is not implemented.
4. **Hyperband / successive halving:** keep the full `-x` fixed. Run each candidate
   to `-es 1000`, evaluate, then promote selected trials using the same directory,
   LR and recipe with `-y 1 -es 3000`. That adds 2000 updates. Promotion decisions
   are external to the trainer. For seed-based rungs, add each cumulative budget
   to 10919; use `-rd 1` only on the first switch to the cluster.

```sh
# Fresh trial, then its promoted continuation. LR and TRAIN_GLOB are user choices.
./train_gpt2cu -l "$LR" -id "$TRAIN_GLOB" -o runs/trial_A -x 21838 -es 1000
./train_gpt2cu -l "$LR" -id "$TRAIN_GLOB" -o runs/trial_A -x 21838 -es 3000 -y 1
```

Select by held-out validation loss at matched budgets, not training loss. For
short trials, `-v` may be lowered; validation always runs at the stopping point.
Checkpoint retention is step-based; off-grid checkpoints can leave extra files.

### Additional controls

`-nt TOKENS` can replace a nonnegative `-x`; `-et TOKENS` can replace `-es`.
Both token limits round up to whole global batches and are absolute, including
pre-resume updates. With neither horizon flag, course mode uses 21838 updates;
explicit `-x -1` requests one dataset epoch. `-e` selects weights only or a fresh
model descriptor; `-ss` sets a weights-only starting counter, **not** a separate
schedule clock. Neither substitutes for full resume with `-y 1`.

μP controls, GPU diagnostics and distributed options remain in `--help` for
advanced use. Saved metadata overrides implicit architecture/μP defaults;
conflicting explicit settings fail. Legacy checkpoints need explicit
parametrization declarations. `-cp 0` retains legacy defaults for parity tests.
Changing architecture, horizon, global batch or schedule can change the preferred
LR; transfer is an empirical question, not a guarantee.

## Batch extrapolation, LR transfer and decay timescale

The preset matches the batch-study context and schedule while retaining our
12-layer, fixed-12-head width-transfer family (reference width 192, attention
multiplier 4). The batch study used a different family, 1–3 layers with reference
64 and attention multiplier 8. Adopting its recipe does not establish that its
batch optimum transfers to this family or to ClimbMix.

The purple discovery fit was `batch_sequences = 0.006130498046614271 * D^0.4199436828783816`,
in units of 2048-token sequences, where `D` counts all processed training tokens.
At roughly 2.50B tokens it predicts about 111,100 tokens/update. The course batch
114688 is the nearest multiple of 16384 (current microbatch 8 × context 2048). This is
an exploratory extrapolation of three small-model, seed-42 minima, not a validated
large-model optimum. Batch is a fixed preset, not recalculated when changing
`-nt` or proxy width. Compatible single-GPU microbatches include 4, 7, 8, 14,
28 and 56; microbatch 16 or 32 would require a different global batch, such as
131072. Benchmark throughput before choosing a larger microbatch.

The expanded positional table makes the target 125,226,240 parameters excluding
unused vocabulary padding (125,262,336 allocated). Its 20-TPP budget is
2,504,524,800 processed tokens, rounded up by the trainer to 2,504,556,544.
Keep this **target** horizon and batch for every skinny proxy; do not use 20 times
the proxy's parameter count. Repeated dataset passes count toward this budget.

Following the peak-LR convention in [Power Lines](https://arxiv.org/html/2505.13738v2#S2.SS2),
`tau = B_tokens / (peak_base_lr * base_decay * D_actual)`.
Thus `base_decay = 1 / (peak_base_lr * full_schedule_updates * tau)`.
The default `-wt 0.1` is the timescale used by all three purple-line discovery
winners; it is a provisional recipe choice, not the paper's universal optimum.
At the target budget, LR 0.002 gives base decay about 0.229; LR 0.004 gives about
0.1145. Each LR candidate gets its own decay to preserve the same timescale.
μP's hidden-matrix LR division and decay multiplication cancel in their product,
so this timescale is also preserved across widths. Biases and normalization
parameters remain exempt from weight decay.

The derived decay stays constant during a run; it does **not** divide by the
instantaneous decaying LR. Timescale is defined at the peak, so schedules must
match in shape. Explicit `-c` selects the old fixed-decay mode; combining it with
explicit `-wt` is rejected. Very short smoke schedules with `tau * updates < 1`
must use a longer horizon, a larger timescale, or explicit fixed decay.

The paper finds optimal timescale changes with tokens per parameter. Holding
`tau=0.1` across skinny and target models here is a controlled width-transfer
experiment, not a claim that every proxy has its independently optimal decay.
Keep the chosen target timescale fixed during the LR sweep and transfer the
winning **base** LR unchanged. Our prior short cosine/context-256 sweeps do not
yet supply the winning LR for this linear/context-2048/full-budget recipe.

Prepare five candidate commands without starting training:

```sh
python3 dev/test/prepare_course_recipe.py \
  --train-pattern 'data/climbmix_train_*.bin' \
  --val-pattern 'data/climbmix_val_*.bin' \
  --tokenizer data/gpt2_tokenizer.bin --output runs/course-lr-plan.json
```

Use local shard paths matching your downloaded dataset. Select by final held-out
validation loss, using the same validation tokens across candidates. If the best
candidate lies at a grid boundary, extend the grid before transferring it.

## 1. Inspect and check the parametrization

Read `gpt_build_from_descriptor`, `gpt2_backward` and `gpt2_update` in
`train_gpt2.cu`. Compare `-me 0` (standard) and `-me 1` (μP). For this experiment,
use reference width `-mb 192` and attention multiplier `-ma 4`; hidden matrix
LRs are the base LR divided by `width / 192`. Query matrices start at zero,
tied token embeddings remain random, and final-layernorm backward scaling
matches the forward scaling. Inspect the corresponding weight-decay scaling.

Host checks require a C/C++ compiler and do not launch GPU jobs:

```sh
python dev/test/test_training_controls.py
python dev/test/test_model_initialization.py
python dev/test/test_sweep_recovery.py
python dev/test/test_source_packaging.py
```

An additional Python initialization check requires PyTorch:

```sh
python -m pip install torch
python dev/test/test_python_mup_initialization.py
```

GPU correctness drivers and their scope are documented in
[`dev/test/MODAL.md`](../../dev/test/MODAL.md). Existing legacy SP numerical
reference failures are documented in [hyperparameters.md](hyperparameters.md);
they are not reported as passes. Provenance retains the imported training-core
hashes used for those experiments and records the later preset changes separately.
The preset exposes the existing initialization seed and changes CLI defaults;
it does not change μP kernels or optimizer scaling rules.

## 2. Transfer LR across widths with a constant schedule

Freeze depth at 12 layers, head count at 12, sequence length at 256, global batch
at 4,096 tokens, and seed 42. Sweep widths 48, 96, 192, 384 and 768. Every run
receives 500 updates, or 2.048M tokens. The LR grid below is 2^-14 through 2^-6,
inclusive: 90 trials across five widths and both parametrizations.

```sh
modal setup
modal run --detach dev/test/modal_gpu_tests.py --suite sweep \
  --min-lr-exponent -14 --max-lr-exponent -6 \
  --scheduler constant --warmup-steps 0 --final-lr-fraction 1 --weight-decay 0
```

Save the result directory printed by the command. Plot it locally:

```sh
python dev/test/plot_lr_transfer.py /absolute/path/to/constant-results
```

The coordinator records narrow-model LR selections before running the width-768
target trials. Measure the target's loss penalty when using the proxy-selected
LR, as well as the displacement of the sampled minimum. A boundary minimum
requires a wider search. Nonfinite trials are marked explicitly.

Questions: how far do SP and μP minima move with width? How broad are the minima?
Does the loss penalty support transfer even when two widths select adjacent LRs?
Widths are different architectures, not independent statistical replicates.

## 3. Repeat with warmup, cosine decay and weight decay

```sh
modal run --detach dev/test/modal_gpu_tests.py --suite sweep \
  --min-lr-exponent -14 --max-lr-exponent -6 \
  --scheduler cosine --warmup-steps 50 --final-lr-fraction 0 --weight-decay 0.1
```

```sh
python dev/test/plot_lr_transfer.py /absolute/path/to/cosine-results
python dev/test/compare_lr_schedules.py \
  --constant /absolute/path/to/constant-results \
  --decay /absolute/path/to/cosine-results \
  --output ../modal-results/schedule-comparison
```

LR now means **peak base LR**. Every width uses the same normalized schedule.
The comparison changes warmup, schedule and weight decay together; it cannot
attribute an effect to only one of them. A cosine/zero-decay control separates
the added weight decay. Final validation loss remains the primary metric.

## 4. Optional: vary the training horizon at fixed widths

This separate, matched experiment uses μP only at widths 48, 96 and 192:

| Budget | Updates | Training tokens | Warmup updates |
| --- | ---: | ---: | ---: |
| 1x | 500 | 2.048M | 50 |
| 8x | 4,000 | 16.384M | 400 |
| 16x | 8,000 | 32.768M | 800 |

```sh
modal run --detach dev/test/modal_mup_horizon_sweep.py
python dev/test/plot_mup_horizons.py /absolute/path/to/horizon-results
python dev/test/plot_mup_lr_vs_tokens.py /absolute/path/to/horizon-results
```

It runs 81 trials: three widths, three budgets and nine LRs. It uses physical
microbatch 16, nested training-data prefixes, and the same 1,048,576 held-out
tokens evaluated at initialization and completion. These controls differ from
the earlier width sweep's microbatch 4 and 32,768-token validation. Compare
horizons within this matched experiment, not absolute losses across protocols.

To refine the critical log2-LR region [-10, -6] from spacing 1 to spacing 0.25,
reuse that completed run by its directory name, not its absolute path:

```sh
modal run --detach dev/test/modal_mup_horizon_sweep.py \
  --refine-from YOUR_COMPLETED_RUN_NAME
```

This adds 108 trials and reuses the original 81 after source/data checks. The
original run must still exist in your Modal results volume. Re-run both plotters
on the new result directory. The log-log plot summarizes the median across
widths; it also retains individual-width curves. A straight fit through three
budgets is descriptive, especially with one seed and broad minima.

Longer horizons also lengthen warmup/decay and change accumulated regularization.
This experiment measures horizon-and-recipe dependence, not an isolated effect
of data volume. Training minima are unsuitable substitutes for held-out loss:
they can select easy batches, and longer runs have more chances to find them.

## Interpretation and reporting

Report the full LR curves, sampled minima, proxy-to-target loss penalties, token
budgets, schedule, batch, reference width, initialization and source/data hashes.
Discuss near-optimal ranges and failed trials. Repeat selected settings with
independent initialization seeds before treating small differences as reliable.
The bundled sweep drivers still use seed 42; direct trainer runs support `-se`.

The existing results support approximate LR transfer across width at a fixed
recipe, with preferred LRs changing across training horizons. They do not prove
exact invariance, establish a data-size power law, or validate full 20-TPP
training. See [lr-transfer.md](lr-transfer.md) for results and limitations.

For the current 124,475,904-parameter target, 20 TPP is approximately **2.49B
tokens**. A clean width-transfer test gives the skinny proxy that same token
budget, rather than 20 times its own parameter count. The short drivers above
do not launch this full-budget experiment or prepare enough unique data for it.

## Provenance

This course release imports integration snapshot
`a6d85cb052cfe4d7aa6d7cd2719332f8852648bf`, derived from
[`GatiAher/llm.c@mup`](https://github.com/GatiAher/llm.c/tree/mup), with the
10605 training controls and subsequent μP fixes. Original attribution and the
MIT license are retained. `source-provenance.json` records the upstream base,
course base, imported training-core hashes, and current course-preset hashes. Historical
generated logs/plots and unrelated unfinished experiments are excluded.
