# 10605 training controls on the μP branch

Integration branch: `integrate/10605-hyperparameters`.

Compared before editing:

- Target `GatiAher/llm.c`, branch `mup`, commit `706b1a5fc94de4bc16a4a3684eafaad6b9e0dada`.
- Reference `10605/llm.c`, `HW5_Code/`, commit `2a9c18ce34c99408aa5ce6a67149b428884452de`.

## Comparison and port decisions

| Source change | Integration |
| --- | --- |
| Independent depth, channels and heads | Apply `-1d`, `-1c`, `-1h` to the target's descriptor configuration **before** allocation and μP initialization. |
| `-ss` starting step | Start the training loop, LR lookup and Adam bias-correction counter at this absolute step. A resumed optimizer checkpoint takes precedence. |
| `-es` early stop | Stop at an absolute step; retain the full `-x`/one-epoch LR horizon. This is a step limit, not validation-loss patience. |
| Four input path flags | Add `-id`, `-iv`, `-it`, `-ih`; preserve `-i`, `-j`. |
| Run-parameter file | Append effective configuration and invocation to `run_parameters.txt`, only on rank zero and only with `-o`. |
| Training time in logs | Append `time (ms):...` to training records. Keep the original logger entry point for other executables. |
| Optimizer allocation fallback | Port device-OOM fallback to managed memory for Adam moments and master weights, using the existing per-rank shard sizes. Report affected ranks. |
| `FORCE_NVCC_O` build control | Add the optimization-level override, default 3. |
| Scheduler implementations | All four implementations (`cosine`, `linear`, `constant`, `wsd`) are identical; retain the target header. |
| Checkpoint frequency, retention, completion markers | Already present and effectively identical. Retain `-n`, `-nk`, `-nm`, per-rank state and DONE barriers. Final early-stop iteration goes through evaluation and checkpointing. |
| Source-only defaults | The initial integration kept target defaults. The later course preset below selects the student defaults. |
| Attention, encoder, layernorm, GELU, matmul and cuDNN differences | The relevant target differences implement μP and coordinate instrumentation. Retain those kernels and signatures. |
| Source `cuda_common.h` rename | Adapt the allocation helper to the target error-checking function; no rename needed. |
| Source inference executable and greedy tokenizer encoding | Inspected; separate prompt-inference functionality is outside this training integration. It duplicates SP model code, and its encoder can fail to advance on unmatched text. No dependency from the training controls requires either. |
| Source Makefile removals | Retain precision selection, CPU/GPU tests, profiling, NCCL discovery, and existing cuDNN default. |
| Other `utils.h`/tokenizer differences | Diagnostic-text changes only for training; retain target diagnostics. |

The integration retained the target's AdamW, matrix-group LR and decay scaling, CUDA/cuDNN attention scaling, and pre-logit scaling. Subsequent user-authorized changes correct the final layernorm backward scale and keep the tied token embedding/readout random rather than zero. Query initialization remains zero. See the validation updates below.

## Architecture and μP reference width

The course preset (`-cp 1`, the default) initializes fresh weights from `d12` and fixes the head count at 12. Descriptors `d12`, `gpt2:d12`, `gpt3:c768`, etc. retain their layer/channel settings; `-1d` and `-1c` override them. An explicit `-1h` overrides the course head count for advanced experiments. With a checkpoint, saved dimensions take precedence over implicit defaults and supplied dimensions must match. Under `-cp 0`, descriptor head counts and the old `gpt2_124M_bf16.bin` default are retained; architecture flags without `-e` still select random initialization based on `d12`.

Channels must be divisible by heads and aligned to 128-bit kernel loads: multiples of 8 for BF16 or 4 for FP32. Existing CUDA attention also requires sequence lengths divisible by 4. Memory and backend constraints still apply.

`-mb N` specifies a reference width. The effective multiplier is `channels / N`. Fresh course μP models default to reference width 192, so students only need to set `-1c`. If `-mm` is also explicit, it must agree. An explicit `-mm` without `-mb` defines the reference `channels / multiplier`, which must be a positive integer. An explicit multiplier is never silently replaced. Under `-cp 0`, custom μP channel counts require explicit `-mb` or `-mm`, and descriptor-only runs retain the original multiplier default of 1. Saved checkpoint metadata takes precedence over implicit course defaults.

`-ma` defaults to 4 in the course preset and 1 under `-cp 0`. Keep this value fixed across widths in a transfer experiment. The course value corresponds to sqrt(192/12); it is not recomputed from each model's channels or head count. For example, an advanced 128-channel reference with two heads may explicitly use `-ma 8` (sqrt(128/2)).

Coordinate checks keep `-mc` and `-mw`, default to two layers/two heads, and accept explicit architecture overrides. Conflicting `-mw`/`-1c` values fail. Checks collect up to four updates using a buffer sized for the effective depth and indices relative to the run's starting step. The existing filenames and default 76-float layout are preserved. Only rank zero writes the file. Separate GELU statistics are enforced with recomputation/fusion disabled. As before, coordinate checks require a non-cuDNN build and one microbatch per optimizer step. Custom depths require a corresponding layout when interpreting the binary with the existing visualization script.

## Steps, continuation and checkpoints

`-x` defines the full schedule in optimizer updates (course default 21838); `-es` stops at an absolute update without changing that schedule. The equivalent token interface uses `-nt TOKENS` for the full training horizon and optional `-et TOKENS` for an earlier stopping point. Both round up to whole global batches: `ceil(TOKENS / -d)` updates. `-et` is an absolute token count, including updates before checkpoint resumption, and must not exceed the requested `-nt` (or the effective full schedule if `-nt` is omitted). The resolved stop must lie between the starting/resumed step and the full schedule. If both rounded token limits select the same update, the full schedule completes. Requested and effective token limits are recorded in `run_parameters.txt`.

The course default `-u -1` resolves to 10% of the full schedule (rounded down), or zero for a constant scheduler. Explicit `-u 0` disables warmup. Early stopping changes only the execution boundary; it does not shorten warmup or decay. Use `-y 1 -o DIR` for full continuation and repeat the original schedule/batch/data settings. Remove `-et` or increase it to a later absolute token count. Keep global batch fixed on resume, since checkpoint progress is stored in optimizer steps.

`-x` and `-es` express the same controls in optimizer steps. They cannot be combined with `-nt` and `-et`, respectively, when their values are nonnegative. For `-x 100 -ss 20 -es 30`, updates use steps 20 through 29, Adam times 21 through 30, and the original 100-step LR curve. Validation/checkpointing runs at step 30. `-ss` alone does not restore moments, RNG or data position. A saved step overrides `-ss`. With neither a token nor step horizon, the horizon is one dataset epoch. Token-based early stopping also works with that default horizon or an explicit legacy `-x`.

`-se SEED` selects fresh-weight initialization, with values 0 through 2147483647 and default 42. It does not change the data shuffle or stochastic-rounding seeds. Loaded checkpoint weights ignore it; optimizer resumption restores saved RNG state.

New model and optimizer checkpoint headers store μP enablement, width multiplier, attention multiplier, and reference width in previously unused padding. Optimizer headers additionally store ZeRO stage. Existing payload layouts and file versions are unchanged. This implementation restores the saved parametrization and rejects conflicting explicit flags, model/state mismatches, and changes of ZeRO stage. Older readers can read the payload but do not understand the μP metadata; use this integration to recover it automatically.

Legacy checkpoints lack reliable parametrization metadata. For `-y 1`, explicitly provide `-me 0` for SP, or `-me 1 -mm MULT -ma MULT` for μP. These values are the caller's declaration of the legacy settings. Loading legacy weights for μP also requires those explicit settings. To load legacy SP weights under the course preset, specify `-me 0`; ordinary legacy SP weight loading under `-cp 0` keeps working. Schedule, batch/data and optimizer controls are not restored from metadata: repeat the original values when exact continuation is desired. Process-count/rank checks remain in the existing state loader.

No optimizer checkpoint is written before the first optimizer initialization, including zero-update runs and runs started at a checkpoint boundary with `-ss`. This avoids serializing uninitialized moments/master weights. Loading a master-weight state with `-w 0` now skips the stored master payload before reading loader state. Timing averages use the actual iterations executed, excluding the first iteration in this process.

Retention retains the target's step-based policy: delete `step - nk*n`, unless major. It is not a strict directory-wide count; off-grid early-stop checkpoints can leave additional files. `-nm` major checkpoints remain exempt.

Full resume with `-rd 1` starts the training loader at the beginning of a new dataset while preserving weights, optimizer, RNG and step. Default `-rd 0` preserves the loader position; omit `-rd` on later resumes of the same trial. It does not restart the LR schedule. The [student guide](exercise.md#four-experimental-strategies) maps flags to architecture, horizon, dataset-size and Hyperband experiments.

## Defaults

After the November 3 initialization diagnosis, newly initialized μP models keep random tied token embeddings (standard deviation 0.02) and zero query matrices. Existing checkpoints load their stored weights unchanged. The Python initializer now dispatches hidden matrices and query slices by explicit module markers instead of incorrectly treating `_get_name()` as a parameter path. This aligns its initialization rules with CUDA; the independent official-reference tests remain separate from the Python training program.

The CUDA trainer's course preset defaults to fresh 12-layer/768-channel μP,
12 heads, reference width 192, attention multiplier 4, initialization seed 42,
microbatch 8, context/positional capacity 2048 and 114688 global tokens/update.
**`-l` is required, including on resume; no seed LR is embedded in the preset.**
The full default schedule is 21838 updates. Explicit `-x -1` uses one dataset
epoch; explicit `-nt` supplies a token budget instead. Linear decay, automatic
10% warmup (`-u -1`, normally 2183 updates) and final fraction zero are default.
AdamW timescale `-wt 0.1` derives base decay as
`1/(peak_base_lr * full_schedule_updates * timescale)`. Explicit `-c` selects
fixed decay instead and cannot be combined with explicit `-wt`. Early stopping
preserves the full-horizon warmup and decay calculation.

Logging requires `-o`. Checkpointing defaults to interval 1000 and a step-based
retention window of 2 intervals (not a strict file count); `-n 0` disables saving
and `-nk 0` keeps all checkpoints. Validation defaults to every 1000 updates and
up to 72 microbatches, plus the stopping point. Sampling and HellaSwag remain
disabled. ZeRO 0, BF16 and non-cuDNN remain the build/runtime defaults; backend
selection is an explicit build choice. Training kernels are unchanged by these
CLI-default updates.

Course paths are `data/pretraining/climbmix_students_train_*.bin`,
`data/seed-validation-v1/balanced_*.bin` and `data/gpt2_tokenizer.bin`.
`-cp 0` preserves legacy defaults: SP, checkpoint input, TinyShakespeare paths,
tokenizer `gpt2_tokenizer.bin`, microbatch 4, automatic global batch, one-epoch
horizon, optional LR default 3e-4, no warmup, final fraction 1, fixed decay 0,
checkpointing disabled, validation 20/20 and sampling every 20 updates.
Explicit flags override either preset, independent of selector position.

Both presets retain `-ss 0`, `-es -1` and HellaSwag path `dev/data/hellaswag/hellaswag_val.bin`. CLI names are matched exactly; unknown options, malformed numeric values and invalid combinations fail. `--help` succeeds before CUDA startup. Help names the existing `-pi`/`-pf` NCCL initialization options. See [the student guide](exercise.md#course-preset-and-student-knobs) for the normal course commands.

## Examples

From the trainer directory, after building and preparing the datasets. These explicit legacy examples preserve the original smoke-test recipe with `-cp 0`:

```sh
make train_gpt2cu USE_CUDNN=0

# Standard parametrization, custom architecture
./train_gpt2cu -cp 0 -e d12 -1d 2 -1c 256 -1h 4 -me 0 \
  -id 'dev/data/tinyshakespeare/tiny_shakespeare_train.bin' \
  -iv 'dev/data/tinyshakespeare/tiny_shakespeare_val.bin' \
  -it gpt2_tokenizer.bin -b 2 -t 64 -x 20 -es 4 -m 2 -s 0 -o runs/sp -n 2

# μP, width 256 relative to reference width 128
./train_gpt2cu -cp 0 -e d12 -1d 2 -1c 256 -1h 4 -me 1 -mb 128 \
  -id 'dev/data/tinyshakespeare/tiny_shakespeare_train.bin' \
  -iv 'dev/data/tinyshakespeare/tiny_shakespeare_val.bin' \
  -it gpt2_tokenizer.bin -b 2 -t 64 -x 20 -es 4 -m 2 -s 0 -o runs/mup -n 2

# Resume μP from step 4, with settings recovered from the checkpoint
./train_gpt2cu -cp 0 -y 1 -o runs/mup -b 2 -t 64 -x 20 -es 8 -m 2 -s 0 -n 2
```

The short sequence is useful for smoke tests; full training normally uses the descriptor's full positional context. Create the parent `runs` directory before using these examples.

## Verification on this machine

Successful commands, from the repository root:

```sh
python3 dev/test/test_training_controls.py
python3 dev/test/test_model_initialization.py
cc -O3 -I llmc -o /private/tmp/llmc_test_dataloader dev/test/test_dataloader.c -lm
(cd /private/tmp && ./llmc_test_dataloader)
cc -O3 -I llmc -o /private/tmp/llmc_test_outlier dev/test/test_outlier_detector.c -lm
/private/tmp/llmc_test_outlier
git diff --check
```

- 17 host test groups cover actual CLI/configuration code, course/legacy defaults and override precedence, full/early-stop token rounding, absolute token stops on resume, unchanged full-horizon warmup, initialization seeds, flags and malformed inputs, descriptor/custom SP and μP configurations, all six coordinate widths, scheduler values, absolute stop/start bounds, metadata recovery/conflicts, and rank-aware logging. The test's resumed-step fixture is not an optimizer-resume execution.
- Initialization test executes the actual host initialization loops and model save/load functions with host memory transfers. At widths 48/96/192/384/768, SP is byte-identical to the pinned target base; μP differs only by retaining the SP random token embedding. It verifies zero queries, equivalent custom architecture, and model header/payload round trips. It requires the pinned base commit in local git history. `test_python_mup_initialization.py` separately checks the Python trainer's tied embeddings, query zeros, and hidden-matrix scales at all five widths using PyTorch.
- The initialization harness additionally verifies course head/reference defaults, identical initialization with repeated seed 42 and distinct weights with seed 43. Both host harnesses compile with AddressSanitizer and UndefinedBehaviorSanitizer.
- Existing dataloader tests pass for sequential, shuffled and simulated multi-process data partitioning. Existing outlier-detector test passes. Dataloader compilation emits two existing variable-length-array warnings under Apple Clang.

## Modal GPU validation (2026-11-03)

Local compilation still requires a CUDA host. GPU validation was run on Modal A10 workers using CUDA 12.4.1, cuDNN 9.2.1.18/frontend v1.5.2, and MPI/NCCL. Source hashes and the pinned original `mup` commit are saved with every run. See `dev/test/MODAL.md` for setup and result layout.

Successful checks:

- BF16 CUDA, FP32 CUDA, BF16 cuDNN, and two-GPU MPI/NCCL builds.
- Exact CLI help/options and malformed-option rejection.
- Default and custom SP/μP training, four LR schedules, absolute start/early-stop behavior, checkpoint conflicts and retention.
- Bit-identical trained model payloads against the original `mup` branch for BF16 SP/μP and FP32 SP.
- Bit-identical resumed model, optimizer, loader, and RNG state in CUDA SP/μP and two-GPU SP/μP with ZeRO 0/1.
- Finite coordinate outputs at widths 64/128/256/512/1024/2048 in SP and μP (this alone does not establish width invariance).
- Existing GPU device/file I/O test.
- FineWeb training at widths 48/96/192/384/768 with 12 layers and 12 heads, in SP and μP.
- cuDNN exact resumption after fixing the previously uninitialized `mt19937_state.seed_` metadata field in `manual_seed`. Before the fix, the two state files differed by one byte inside that field; weights, moments, and active RNG state matched. A host regression test first failed with distinct prefilled memory, then passed after the fix, including the original documented random sequence. Dataloader tests and initialization parity also passed again.

Commands (from the repository root with authenticated Modal):

```sh
modal run dev/test/modal_gpu_tests.py --suite single
modal run dev/test/modal_gpu_tests.py --suite multi
modal run dev/test/modal_gpu_tests.py --suite baseline
modal run dev/test/modal_gpu_tests.py --suite focused
modal run dev/test/modal_gpu_tests.py --suite cudnn_diagnostic
modal run dev/test/modal_gpu_tests.py --suite benchmark
cc -O2 -I llmc dev/test/test_rng_state.c -lm -o /tmp/test_rng_state
/tmp/test_rng_state
```

**Unresolved numerical-reference failures:** Existing `test_gpt2cu` fails its gradient checks on this A10 environment in BF16, FP32, and BF16 cuDNN. The four BF16 variants (default, recompute 0/2, and no master weights) and FP32 have identical printed numerical diagnostics on the integration and untouched original branch. Training-loss trajectory checks and the test's own resumed-loss comparisons pass. The cuDNN reference failure also has identical numerical diagnostics on the original branch. Disabling GELU fusion does not resolve the gradient-reference failures. These remain failed tests; tolerances were not relaxed. Integration parity demonstrates preservation of behavior, not independent proof that every inherited kernel is numerically correct.

**Inherited μP gradient issue corrected after the original integration tests:** Final normalized activations are scaled by `1 / width_mult` in forward, but the original backward call multiplied upstream gradients by `width_mult`. The user authorized changing that call to the reciprocal factor. The previous contribution differed by width_mult squared (16× at width 768 with reference width 192). Run `20261103T133213Z-mup_gradient` now passes 14 independent CUDA full-model finite-difference cases, with 460 sampled gradient checks across FP32/BF16 and all five sweep widths. The isolated old version fails at each non-reference μP width; SP and multiplier-one controls remain byte-identical. Run `20261103T133301Z-mup_fix_training` passes corrected-μP exact checkpoint resumption and SP/reference-μP trained-payload parity. The original constant and cosine/decay sweeps remain pre-fix artifacts. The fresh 170-trial constant-LR/zero-decay run `20261103T133353Z-sweep` passes its log/provenance audit and improves proxy LR alignment, but retains a 0.1842-nat best-loss gap at 124M (SP 6.4085, μP 6.5927). Earlier original-branch parity claims above describe the integration before this intentional correction. Resuming a pre-fix μP checkpoint with this code uses the corrected derivative; it intentionally does not reproduce the old non-reference trajectory. This fix is separate from the unresolved SP reference-test failures. See `lr-transfer.md` for scope and results.

Reports outside the repository: `../modal-results/20261103T115256Z-single`, `20261103T115507Z-multi`, `20261103T115504Z-baseline`, `20261103T115604Z-focused`, `20261103T115936Z-cudnn_diagnostic`, and `20261103T115522Z-benchmark`. The focused run caught the cuDNN library-link setup problem's resolution and the RNG metadata issue; the later diagnostic run verifies the metadata fix. The earlier reports are retained without rewriting their outcomes.

Managed-memory fallback under a real out-of-memory condition and multi-node execution have not been exercised. The two-GPU tests use one node. The FineWeb LR-transfer pilot is an empirical test of early training, with its scope and limitations documented in `lr-transfer.md`.

## Files changed

- Training/build: `train_gpt2.cu`, `Makefile`, `llmc/train_config.h`, `llmc/cuda_utils.cuh`, `llmc/logger.h`, `llmc/rand.h`.
- Documentation: this file, `doc/mup/mup.md`, `doc/mup/lr-transfer.md`, `dev/test/MODAL.md`.
- Verification: `dev/test/test_mup_gradients.cu`, `dev/test/run_mup_gradient_test.py`, `dev/test/test_training_controls.py`, `dev/test/test_model_initialization.py`, `dev/test/test_rng_state.c`, `dev/test/modal_gpu_tests.py`, `dev/test/run_gpu_integration.py`.
- Experiment: `dev/test/prepare_lr_transfer.py`, `dev/test/run_fineweb_sweep.py`, `dev/test/plot_lr_transfer.py`.

No CUDA kernel bodies were modified by the training-control integration. The
course snapshot and its setup are documented in [exercise.md](exercise.md), with
the imported revision and training-core hashes in `source-provenance.json`.
