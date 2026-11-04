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
they are not reported as passes. The imported training-core hashes match the
code used for the completed corrected-μP gradient/optimizer checks and sweeps.

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
independent seeds before treating small differences as reliable; these sweep
drivers currently fix seed 42 and do not supply a seed CLI option.

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
course base and SHA-256 hashes of the unchanged training core. Historical
generated logs/plots and unrelated unfinished experiments are excluded.
