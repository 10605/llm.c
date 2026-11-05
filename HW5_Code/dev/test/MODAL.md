# Modal GPU integration tests

## Course-preset regression checks

```sh
python3 dev/test/test_training_controls.py
python3 dev/test/test_model_initialization.py
modal run --detach dev/test/modal_gpu_tests.py --suite course_preset
```

The short GPU suite checks implicit fixed-head μP at widths 48/96/192/768,
the default 124M architecture, byte-identical trained weights against the same
recipe supplied explicitly in legacy mode, distinct initialization seeds,
exact model/optimizer/data/RNG checkpoint resumption, metadata precedence,
token-budget rounding, microbatch accumulation, and full-horizon warmup.
The token-stop regression additionally compares trained weights and optimizer
state between `-et` and equivalent legacy `-es` runs, then checks exact resumption
to a later absolute token limit while retaining the original LR schedule.
Run `20261104T083447Z-course_preset` passed all ten groups on one A10,
including automatic timescale decay at two LRs and batches, exact resumption,
and a 2048-context update with the 114688-token batch. Automatic and explicit
decay produced byte-identical model/optimizer state. Its app
`ap-pXpdXpMkvMcFb7n8EP1brr` stopped with zero tasks. This is a short correctness
check, not a full-budget LR-transfer experiment.
Run `20261104T075450Z-course_preset` passed the previous eight groups,
including the token-stop regression. The preceding preset-only run
`20261104T074041Z-course_preset` passed its seven groups and remains a historical artifact.
Historical integration/parity cases explicitly use `-cp 0` to preserve their
original defaults. Existing numerical-reference failures remain documented below.

## Existing suites and source packaging

The sweep coordinator saves worker call IDs before waiting for a phase, so a restarted coordinator reconnects to its existing workers. `python3 dev/test/test_sweep_recovery.py` checks this without cloud calls and rejects mismatched source/recipe metadata. A restart between spawning workers and committing their IDs remains a small unprotected window; worker-level preemption can still retry an individual worker. Source manifests include the local source commit, dirty status, and individual file hashes.

`modal_gpu_tests.py` uploads the current local source (including uncommitted changes) and the pinned original μP baseline. It runs disposable GPU workers, downloads text/JSON reports, and persists reports and downloaded test fixtures in the Modal volume `llmc-integration-test-results`. It does not upload the git directory, credentials, or existing weights/logs.

The image pins CUDA 12.4.1, cuDNN 9.2.1.18 and cuDNN frontend v1.5.2, and installs MPI/NCCL. Correctness-test GPU workers are limited to 20 minutes. The single suite requests one A10; the multi suite requests two. Workers scale down after completing their jobs.

From the trainer source root (`HW5_Code/` in the course repository), with a
current Modal CLI installed and authenticated, first fetch the pinned public
baseline required by source packaging and initialization tests:

```sh
git fetch https://github.com/GatiAher/llm.c.git 706b1a5fc94de4bc16a4a3684eafaad6b9e0dada
```

Then run:

```sh
modal run dev/test/modal_gpu_tests.py --suite single
modal run dev/test/modal_gpu_tests.py --suite multi
```

Authenticate your own Modal account with `modal setup`. Keep credentials outside
version control. The original experiment workspace used a separate virtual
environment and `MODAL_CONFIG_PATH`; those local files are not part of this repo.

Results are saved outside the repository under `../modal-results/<UTC-run-id>/`. Each run contains:

- `source_manifest.json`: uploaded source hashes and baseline commit.
- `summary.json`: named checks, status, timings and diagnostic details.
- Numbered `.log` files: exact commands, exit status and output.
- `environment.txt`: GPU and CUDA compiler information.

The single suite covers the CUDA executable and existing GPU test build; CLI parsing; descriptor and custom SP/μP training; all schedules; early stopping; bit-exact resumed model/optimizer/data/RNG state; retention; coordinate output at six widths; original-branch training parity; existing device/file I/O tests; FP32; and cuDNN. The multi suite covers SP and μP on two GPUs with ZeRO stages 0 and 1, including exact resumption.

An existing test failure is reported as a failure even if the original branch has the same issue. Compare against the baseline before changing numerical tolerances or test expectations. Passing smoke tests does not prove all training configurations correct.

## FineWeb learning-rate transfer experiment

After reviewing correctness results, run a 24-update throughput and width-compatibility benchmark:

```sh
modal run dev/test/modal_gpu_tests.py --suite benchmark
```

Launch the bounded pilot on four independent A10 workers:

```sh
modal run --detach dev/test/modal_gpu_tests.py --suite sweep
```

A remote CPU coordinator dispatches 80 narrow-model trials, saves the selected learning rates, then dispatches 20 target-model trials. At most four sweep GPUs run concurrently. Each of eight worker calls has a 45-minute timeout; their combined GPU execution is bounded to six hours, plus container startup/scale-down. The coordinator also has a timeout. No function retries are requested. A trial failure stops progression to the target phase and is retained in the reports.

The dataset is the GPT-2-tokenized FineWeb linked by the repository's `dev/data/fineweb.sh`. The runner resolves and pins a dataset revision, caches one training shard and the separate validation shard, validates their binary headers and lengths, and records SHA-256 hashes. Downloads, source hashes, exact commands, training/validation logs and per-trial summaries are retained in the results volume. The local client downloads those reports when the coordinator returns; the detached coordinator can continue through both phases if the client disconnects.

Plot a downloaded sweep with `python dev/test/plot_lr_transfer.py /absolute/path/to/results`. This writes PNG/PDF curves and `analysis.json`; boundary grid minima and failed trials remain explicit. Matplotlib is needed only for local plotting.

Diagnostic suites `baseline`, `focused`, and `cudnn_diagnostic` reproduce original-branch numerical-reference failures and inspect cuDNN checkpoint resumption. Failed legacy tests remain failures; their tolerances are not changed.

### Matched schedule and weight-decay comparisons

The sweep defaults remain constant LR and zero decay. Explicit options select a new recipe without changing the training core:

```sh
modal run --detach dev/test/modal_gpu_tests.py --suite sweep \
  --min-lr-exponent -16 --max-lr-exponent 0 \
  --scheduler cosine --warmup-steps 50 --final-lr-fraction 0 --weight-decay 0

modal run --detach dev/test/modal_gpu_tests.py --suite sweep \
  --min-lr-exponent -16 --max-lr-exponent 0 \
  --scheduler cosine --warmup-steps 50 --final-lr-fraction 0 --weight-decay 0.1
```

Run these sequentially to retain the four-GPU concurrency limit. Each contains 170 fresh trials at 500 updates/2.048M tokens. The warmup lasts 50 updates at all widths; the remaining updates use the same normalized cosine schedule. The swept LR is the peak base LR. The existing μP matrix learning-rate and weight-decay multipliers remain active. The final fraction is the scheduler's horizon endpoint: the last executed update (index 499) is very close to zero, with the mathematical zero at index 500.

Exact settings are saved in `experiment.json`, each worker's `plan.json`, and the per-trial commands and `run_parameters.txt`. Combining prior runs with a different schedule/decay is rejected. Source and data hashes permit checking that only the intended controls changed. Use `compare_lr_schedules.py --constant <previous-run> --cosine <schedule-run> --decay <decay-run> --output <directory>` to create the matched validation-loss plots and numerical comparison. This plotting command verifies equal grids, horizons, architecture and optimizer controls and labels each recipe explicitly.


### Corrected μP gradient and constant-LR diagnosis

```sh
modal run --detach dev/test/modal_gpu_tests.py --suite mup_gradient
modal run --detach dev/test/modal_gpu_tests.py --suite mup_fix_training
modal run --detach dev/test/modal_gpu_tests.py --suite sweep \
  --min-lr-exponent -16 --max-lr-exponent 0 \
  --scheduler constant --warmup-steps 0 --final-lr-fraction 1 --weight-decay 0
```

The gradient suite compares actual before/fixed CUDA full-model derivatives with independent float64 finite differences in FP32 and BF16. All μP multipliers (.25, .5, 1, 2, 4) and SP controls are included; fixtures have nonzero tied embeddings/query weights. `gradient-checks.json` retains each sampled numerical derivative. The fixture uses 64 tokens and vocabulary 8192 to satisfy the inherited kernels' size constraints. Optional `MUP_GRADIENT_TRACE` enables synchronized stage tracing when compiling the CUDA harness.

The training suite checks exact μP checkpoint resumption at multiplier 2, plus SP and reference-width μP trained-weight parity against the pinned original. Original parity at other μP multipliers is intentionally superseded by the derivative correction; the full single suite now labels and tests reference-multiplier parity explicitly.

Run a fresh sweep after correcting training math. Combining prior trials is rejected when training-core hashes differ. `verify_lr_sweep.py <fixed-run> --baseline <old-run> --expected-source-change <manifest.json>` accepts only the exact declared before/after file hashes. After that audit, `compare_mup_fix.py --before <old-run> --fixed <fixed-run> --output <directory>` checks identical controls and creates before/after validation curves, target loss-gap plots, and numerical control comparisons. Historical results are preserved.

### Official optimizer and initialization diagnosis

`modal_mup_diagnosis.py` provides separate, bounded phases. Run them sequentially to keep at most four GPUs active:

```sh
modal run --detach dev/test/modal_mup_diagnosis.py --phase parity

modal run --detach dev/test/modal_mup_diagnosis.py --phase ablation \
  --parity-run /absolute/path/to/successful-mup-parity
python dev/test/analyze_mup_init_ablation.py /absolute/path/to/mup-ablation

modal run --detach dev/test/modal_mup_diagnosis.py --phase target \
  --parity-run /absolute/path/to/successful-mup-parity \
  --reference-run /absolute/path/to/mup-ablation

modal run --detach dev/test/modal_mup_diagnosis.py --phase queries \
  --parity-run /absolute/path/to/successful-mup-parity \
  --reference-run /absolute/path/to/mup-ablation
python dev/test/analyze_mup_target_diagnosis.py /absolute/path/to/mup-target \
  --queries /absolute/path/to/mup-queries
```

The parity worker is bounded to ten minutes. It pins CPU PyTorch 2.5.1 and Microsoft μP commit `19814971934ef91dd546f88e913fc963e096d11c`; optional package dependencies are suppressed to prevent upgrading the reference runtime. The independent PyTorch transformer uses the official readout and optimizer. Full gradient/forward comparisons synchronize weights each step; optimizer replay uses identical CUDA gradients, and an additional trajectory evolves independently. Original strict tolerances, per-tensor comparisons, official float64 optimizer replay and any failures are recorded. See `doc/mup/lr-transfer.md` for the retained intermittent CPU-FP32 outliers and scope limitations.

Training workers are bounded to twelve minutes each, with at most four per phase. The ablation has 78 width-192 trials, the target phase has 15 width-768 trials plus six width-192 boundary checks, and the query follow-up has six width-768 trials. Each trains for 500 updates/2.048M tokens, using the same cached FineWeb data and constant-LR/zero-decay recipe. Three initialization seeds are paired across variants. The test driver modifies only a disposable trainer copy to select seed and zero/nonzero embedding/query initialization, and retains the exact patch plus initial-weight hashes. The production trainer is not modified.

The target candidate and LR are frozen from the reference-width means before dispatch. Changing only embeddings is preferred when within 0.01 nats of the best initialization arm; a meaningful improvement and interior candidate minimum are required. The query follow-up also uses reference-selected LRs. Neighboring target LRs serve as a secondary transfer check. Plotting/audit scripts verify dataset and source hashes, horizons, LR histories and initialization controls; reports distinguish preset-LR results from full-sweep optima.
