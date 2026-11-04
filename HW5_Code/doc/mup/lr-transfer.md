# FineWeb experiment: learning-rate transfer to a 124M model

Status: **completed: all 100 trials succeeded on four A10 workers** (run `20261103T115752Z-sweep`). Results below are from a single-seed, early-training pilot.

**Gradient correction, 2026-11-03:** The original constant-LR and both cosine/decay sweeps below used an inherited final-layernorm gradient mismatch. Forward scales by `1 / width_mult`, while the old backward call passed `width_mult`. The user subsequently authorized correcting that argument and rerunning the original constant-LR, zero-decay experiment. The correction now passes independent CUDA full-model finite-difference checks in FP32 and BF16 at all five widths, with unchanged SP/reference-width controls. Historical curves remain labeled as pre-correction results; the new comparison is documented separately below. This establishes the targeted derivative correction, not universal correctness of all inherited kernels.

The supplied figure contrasts a width-dependent optimum under standard parametrization with an approximately stable optimum under μP. The relevant empirical question is whether a learning rate selected using a narrow model remains near-optimal for the 124M target. A single successful training run cannot answer that question. See [Tensor Programs V](https://arxiv.org/abs/2203.03466).

## Controlled comparison

| Setting | Choice |
| --- | --- |
| Target | 12 layers, 768 channels, 12 heads; 124,475,904 allocated parameters including vocabulary padding |
| Narrow models | Same 12 layers/12 heads, widths 48 (~2.8M), 96 (~6.3M), 192 (~15M), and 384 (~41M) |
| μP reference | Width 192, `-mb 192`; multipliers 0.25, 0.5, 1, 2, 4 |
| Attention multiplier | `-ma 4` for every μP run, sqrt(reference head dimension 192/12) |
| SP control | Identical widths/depth/heads with `-me 0` |
| Learning-rate grid | Powers of 2, exponents -16 through -7 (10 rates) |
| Schedule | Constant LR, no warmup; same number of steps and tokens at every width |
| Pilot workload | 500 updates × 4096 tokens = 2,048,000 tokens per trial; 100 trials total (204.8M training tokens across trials) |
| Sequence/microbatch | 256 tokens, microbatch 4; accumulate to 4096 tokens |
| Other controls | AdamW beta1=.9, beta2=.95, eps=1e-8; decay 0; clip norm 1; initialization seed 42; no fusion or recomputation |

The x-axis must be `log2(-l)`, the **base** learning rate. μP scales the hidden matrix learning rate by `192 / width`; that effective matrix LR is also recorded in the manifest. Weight decay is zero here to avoid confusing learning-rate transfer with its width scaling. Fixed heads means that head dimension grows with model width. Varying head count instead would be a different experiment.

Use exactly the same tokenized training/validation data and token order for all trials, train from scratch, and keep `-es` disabled. The 500-step experiment probes early training and cannot establish the optimal LR for a full training budget. Use GPT-2-tokenized FineWeb from the dataset linked by `dev/data/fineweb.sh`: one 100M-token training shard and the separate validation shard. Record the resolved dataset revision and file hashes. Every trial sees the same 2.048M training tokens and held-out validation batches. This is a sampled FineWeb pilot, not a pass over the full dataset.

## Selection and reporting

1. Require integration-specific correctness, original-branch training parity, exact resumption, and coordinate checks first. Investigate numerical failures against the untouched baseline and retain any inherited failures in the report.
2. Choose the best base LR from the width-48 μP **validation** curve (primary), plus width-96 and width-192 curves as predeclared secondary checks before examining the target results.
3. Evaluate that chosen LR on the 124M model without retuning it. Use the rest of the 124M LR curve only to quantify transfer regret: selected-LR validation loss minus the best measured target loss.
4. Plot training-loss versus log2 LR as in the supplied reference (mean of the last 20 training losses), and plot final held-out validation loss separately. Display SP and μP side by side, one curve per width.
5. Report the optimum at each width, shifts in log2 LR, target loss regret, and divergence/NaN runs. If the optimum lies at a grid boundary, extend the grid before interpreting it. Do not hide failed runs or interpolate through them as successful measurements.
6. Stable μP minima and smaller transfer regret than SP support width transfer for this setup. Shifted μP minima are evidence to investigate; code matching the original branch is not proof of the μP claim. Confirm promising results across multiple seeds and a longer token budget. The current executable fixes seed 42, so multi-seed confirmation needs explicit seed support or pre-generated initialization fixtures.

## Prepare a manifest

`dev/test/prepare_lr_transfer.py` writes the exact 100 commands and metadata. It has no execution or Modal launch path:

```sh
python3 dev/test/prepare_lr_transfer.py \
  --train-pattern '/data/train_*.bin' \
  --val-pattern '/data/val_*.bin' \
  --tokenizer /data/gpt2_tokenizer.bin \
  --output /tmp/lr-transfer-plan.json
```

The user authorized running the FineWeb experiment after correctness testing. Benchmark all widths first. Run four A10 workers, with all 80 narrow-model trials completing and LR selections saved before dispatching the 20 target trials. Each of the eight worker calls has a 45-minute timeout, bounding combined worker execution to six A10 hours (about $10.41 at the configured 8 CPU cores/32 GiB, before credits and setup). Stop rather than expand the workload automatically if the grid needs extension or the timing estimate exceeds that bound. Keep all comparison runs on the same GPU type, precision and attention backend. Create the parent output directory `lr-transfer` before executing the generated commands.

## Measured pilot timing

The FineWeb benchmark on 2026-11-03 completed 24-update runs for both SP and μP at every proposed width. Extrapolating the measured step times with validation overhead estimates 0.75–0.97 total A10 worker-hours ($1.30–$1.68 at listed resource rates). Allow 15–25 minutes elapsed and $2–$3 with four workers, two phases, compilation and startup. This is an estimate, not a billing receipt. The benchmark report is `../modal-results/20261103T115522Z-benchmark/summary.json`.

The completed pilot used the source snapshot recorded in its `source_manifest.json`. A subsequent one-line fix initializes serialized RNG seed metadata; it does not change random draws or model initialization, and the pilot does not save/resume model checkpoints. GPU cuDNN exact-state resumption was verified after that fix. Numerical reference-test failures reproduced on the untouched original branch are documented in `hyperparameters.md`.

## Results

The run took **11 minutes 26 seconds elapsed** (2026-11-03 11:57:51–12:09:17 UTC), including provisioning and both phases. Combined worker process time was **39.53 minutes**. At [Modal list rates](https://modal.com/pricing), that is **$1.14 estimated worker compute**, before credits and excluding small startup/coordinator overheads and earlier correctness tests. This is not a billing receipt. Modal confirmed all jobs stopped and zero active containers after completion.

Every trial completed 500 updates, 2,048,000 training tokens, and final validation. All eight worker calls used identical dataset revision/file hashes. The reference width was 192 for every μP run, including the proxies narrower than that reference.

| Width | Parameters | SP best base LR | μP best base LR |
| --- | ---: | ---: | ---: |
| 48 | 2,803,104 | 0.001953125 | 0.0009765625 |
| 96 | 6,269,760 | 0.0009765625 | 0.00390625 |
| 192 | 15,193,728 | 0.00048828125 | 0.001953125 |
| 384 | 41,004,288 | 0.000244140625 | 0.001953125 |
| 768 | 124,475,904 | 0.000244140625 | 0.001953125 |

These are minima of the final validation-loss grid. All lie inside the grid. Training-loss minima agree except SP width 384, which prefers 0.00048828125 by the mean-of-last-20-updates metric.

The smaller-model choices were saved before target trials started. Transfer regret is the target validation loss at that preselected LR minus the best measured target validation loss **within the same parametrization**:

| Proxy width | SP transfer regret | μP transfer regret |
| --- | ---: | ---: |
| 48 (primary) | +0.3069 | +0.0902 |
| 96 (secondary) | +0.1390 | +0.0246 |
| 192 (secondary) | +0.0580 | 0.0000 |

μP shows better LR transfer in this pilot: its width-192 choice is also the best measured LR at widths 384 and 768, and its width-48 transfer penalty is about 71% smaller than SP's. The smallest proxies do not have identical optimal LRs: width 48 prefers half the target LR and width 96 prefers twice it. A flat minimum and single-seed noise limit how precisely these minima should be interpreted.

This result concerns transfer. The best absolute target validation loss was **6.4085 for SP** and **6.5739 for μP** in this short experiment; μP did not beat SP on that metric. The pilot supports the expected transfer behavior for this implementation/configuration, but it cannot establish the best LR for long pretraining or seed-robust conclusions.

Artifacts: `../modal-results/20261103T115752Z-sweep/summary.json` contains all trial metrics; `selections_before_target.json` preserves proxy choices; `execution.json` records runtime and cost assumptions; `analysis.json` and `lr-transfer.png`/`.pdf` provide minima and plots. The exact source snapshot, dataset hashes, commands, and logs accompany them. GPU correctness evidence is in `../modal-results/gpu-correctness-report.json` and summarized in `hyperparameters.md`.

## Comparison with the supplied homework handout

The supplied `10_605_F25_HW5_v2.pdf` was reviewed after the first pilot. Pages 6–7 specify the defaults; page 11 says the final run retains them except microbatch 32 and the selected maximum LR. The pilot deliberately used a different, cheap width-transfer setup. Its optimal LRs should not be treated as validated choices for the homework's final recipe.

| Control | Handout / starter code | Width-transfer pilot |
| --- | --- | --- |
| Effective tokens per optimizer update | 524,288 | 4,096 (128× smaller) |
| Microbatch | 64; final run 32 | 4 |
| Sequence length (starter code) | 1,024 | 256 |
| Schedule | Cosine | Constant |
| Warmup | 470 of 4,725 steps; Question 2 uses ~10% | 0 |
| Final LR fraction | 0 | 1 (no decay) |
| Weight decay | 0.1 | 0 |
| Full target horizon | ~2.5B tokens, 4,725 updates | 2.048M tokens, 500 updates |
| Validation interval | 250 updates | 50 |
| HellaSwag evaluation | Enabled | Disabled |
| Target architecture | 12 layers / 768 channels / 12 heads | Same |
| Proxy architectures | Examples vary depth, channels, heads | Channels only; 12 layers / 12 heads |
| Parametrization | Standard starter-code training | Separate standard and μP comparisons |

AdamW beta1=.9, beta2=.95, epsilon=1e-8 and gradient clipping at 1 match the starter code. The pilot preserves the base μP implementation and its defaults in the repository; the experiment overrides are explicit command-line settings.

For a homework-aligned LR choice, a subsequent experiment must use or explicitly model transfer to the actual batch, schedule, warmup, weight decay and token horizon. In particular, a μP base LR cannot be plugged into a standard-parametrization final run without a validated conversion/transfer method. The existing 20-TPP estimate of ~15 hours/$26 for one 124M trial was extrapolated from the pilot's throughput, not a benchmark of the handout's configuration. No 20-TPP training was launched.

## 20-TPP runtime estimate (not launched)

At 20 tokens per parameter for each model, measured pilot throughput extrapolates as follows. These estimates retain the pilot's batch/sequence/backend settings, use typical steady-state step times, and exclude validation/checkpoint overhead. They do not benchmark the handout recipe; the handout additionally prescribes a worst-case timing estimate from steps 10–19.

| Width | Training tokens at 20 TPP | Time per run on A10 | Estimated worker compute |
| --- | ---: | ---: | ---: |
| 48 | 56.1M | 5 min | $0.14 |
| 96 | 125.4M | 12 min | $0.34 |
| 192 | 303.9M | 35 min | $1.02 |
| 384 | 820.1M | 2.4 hours | $4.18 |
| 768 | 2.490B | 15 hours | $25.96 |

The original 100-trial design at these width-specific token budgets would require about 365 A10 worker-hours, roughly four days on four workers and $633 before overhead/credits. Unlike the current equal-token pilot, 20 TPP at each width also changes training token count with width. More FineWeb training shards would be required to avoid repeating the current 100M-token shard. No 20-TPP run was started.

## Larger-LR extension

The user requested a larger range after the initial result. The follow-up keeps all pilot settings fixed and adds LR powers of two from 2^-6 through 2^0: 0.015625 to 1.0. This is 70 new trials, combined with the previous 100 points without rerunning or overwriting them. Four GPU workers run the same narrow-then-target phases. Nonfinite losses or gradient norms are recorded as divergent outcomes; incomplete runs and infrastructure failures remain separate failures. This range check cannot prove a global optimum over all real-valued LRs, different seeds, or longer training horizons.

Launch command:

```sh
modal run --detach dev/test/modal_gpu_tests.py --suite sweep \
  --min-lr-exponent -6 --max-lr-exponent 0 \
  --prior-run 20261103T115752Z-sweep
```

The extension completed as `20261103T121924Z-sweep`: 70 new trials, comprising 30 finite results and 40 nonfinite-loss/gradient outcomes, with zero infrastructure failures. All new trials executed 500 updates. Combined with the original run, there are 170 grid points covering LR 1.52587890625e-5 through 1.0. **All ten width/mode validation minima are unchanged.** The 124M best sampled LR remains 0.000244140625 for SP and 0.001953125 for μP under the pilot recipe.

Elapsed time was 8 minutes 23 seconds, with 28.50 minutes combined A10 worker time and $0.82 estimated worker compute before overhead/credits. All workers stopped. Dataset hashes match the initial run. The only changed core training source is the previously documented initialization of unused serialized RNG seed metadata; random draws and model initialization were verified unchanged.

Updated artifacts are under `../modal-results/20261103T121924Z-sweep/`: `lr-transfer.png` shows the full finite-loss range (logarithmic loss axes), `lr-transfer-zoom.png` shows the region near the minima, and `analysis.json`/`summary.json` preserve results and all divergent outcomes. Crosses in the top strip mark nonfinite runs; curves are not interpolated through them. The prior target results were already observed before this extension, so it is a follow-up range check, not a new blind transfer experiment.

## Schedule and weight-decay follow-up plan

The user requested an equal-token comparison using an LR schedule, explicitly selected zero decay to isolate that change, and then requested another plot with weight decay. Run two fresh sweeps in sequence on four A10 workers:

| Recipe | Warmup | Schedule after warmup | Final LR fraction | Base weight decay |
| --- | ---: | --- | ---: | ---: |
| Existing baseline | 0 | Constant | 1 | 0 |
| Schedule-only trial | 50 updates | Cosine | 0 | 0 |
| Schedule + decay trial | 50 updates | Cosine | 0 | 0.1 |

Both new sweeps retain all 170 grid points (SP and μP; widths 48/96/192/384/768; peak base LR 2^-16 through 2^0), 500 updates, 4096 tokens per update, the same FineWeb shards, seed 42, 12 layers/12 heads, sequence length 256, microbatch 4, validation batches, and all other optimizer controls. HellaSwag remains disabled. Each model sees exactly 2.048M training tokens, independent of width. μP retains reference width 192 and attention multiplier 4. Its matrix LR is base LR divided by width/192; its matrix decay coefficient is base decay multiplied by width/192.

The normalized LR schedule is identical at every width and peak LR. It ramps from 0.02 of peak at update index 0 to peak at indices 49/50, then reaches approximately 0.00001219 of peak at the last executed update (499). The scheduler's zero endpoint is index 500. The schedule-only comparison jointly introduces warmup and decay; it does not separate their individual effects. X-axes show the peak base LR, not average or integrated LR, so the most effective peak may shift under cosine decay.

Select proxy LRs before launching each new target phase. Preserve divergent outcomes. Compare the validation curves and best sampled losses, report transfer regret, and check actual logs for the expected LR sequence, token counts, data hashes and source identity. These remain single-seed early-training results, not validation at 20 TPP. The estimated combined runtime is 35–45 minutes and compute cost $4–$6 based on the preceding sweeps; actual worker time and cost estimates will be recorded.

### Completed schedule-only sweep

Run `20261103T123843Z-sweep` completed all 170 trials: 143 finite and 27 with nonfinite loss/gradient norm; zero infrastructure failures. All trials executed 500 updates. Auditing the actual logs confirmed that each of the 17 peak-LR candidates used an identical base-LR history across all widths and both parametrizations. All eight worker dataset records match the earlier sweep, as do all 26 training-core source hashes. See the run's `verification.json`.

| Width | SP best peak LR | SP validation loss | μP best peak LR | μP validation loss |
| ---: | ---: | ---: | ---: | ---: |
| 48 | 0.00390625 | 6.6245 | 0.00390625 | 6.8693 |
| 96 | 0.00390625 | 6.4328 | 0.0078125 | 6.6369 |
| 192 | 0.001953125 | 6.4360 | 0.00390625 | 6.6183 |
| 384 | 0.0009765625 | 6.3531 | 0.0078125 | 6.6260 |
| 768 | 0.00048828125 | 6.3849 | 0.00390625 | 6.6371 |

The primary width-48 μP LR choice transfers with zero measured target regret on this grid; the corresponding SP regret is 0.2811. Width-96/192 μP regret is 0.0833/0.0000. However, the best absolute μP target loss is worse than the constant-LR baseline (6.6371 vs 6.5739). SP's best target loss improves (6.3849 vs 6.4085). Transfer stability and absolute target quality are separate observations. All sampled minima are interior to the tested range.

Worker process time was 1.12779 A10 worker-hours, or $1.9563 estimated worker compute before credits and startup/controller overhead. The zero-decay plots are saved under this run's result directory. The follow-up decay-0.1 sweep is `20261103T125700Z-sweep` and uses the same schedule and grid.

### Investigation of the SP–μP loss gap

The user questioned μP's worse absolute loss. At 124M, the schedule-only best losses differ by 0.2522 nats (6.6371 versus 6.3849), approximately 28.7% higher perplexity for the existing μP implementation. There is no evidence that 20-TPP training will eliminate that gap.

The audit found that `gpt2_forward` passes `1 / model->mup_width_mult` into the final fused residual/layernorm, whose kernel multiplies the normalized output by that factor. `gpt2_backward` instead passes `model->mup_width_mult` into the final `layernorm_backward`, whose kernel multiplies `dout` by the passed factor in both input and affine-parameter gradients. Chain-rule differentiation requires the same reciprocal factor as forward. The discrepancy is width_mult squared: 1/16, 1/4, 1, 4 and 16 for the five sweep widths. The direct output-projection gradient follows a separate path, so this must not be described as a uniform learning-rate multiplier for all parameters. Adam normalization, global gradient clipping and the tied embedding/input path complicate its training effect.

The mismatch is present in the pinned original branch (original backward call at line 879) and was preserved by the hyperparameter integration. The Python forward also divides by width_mult and relies on ordinary autograd. A separate float64 finite-difference calculation of the extracted layernorm formulas confirms the reciprocal derivative to absolute error below 2e-10; the existing formula differs by the predicted squared multiplier. The diagnostic is saved outside the repository at `../modal-results/mup-output-gradient-diagnostic.json`. This is a formula-level CPU check, not a CUDA kernel test or proof that fixing this issue will close the loss gap.

There is also a loss gap at reference width 192, where the erroneous factor equals one. Therefore this discrepancy cannot by itself explain every SP–μP difference. The inherited μP initialization additionally zeroes the tied token embedding/readout and the query matrices, whereas SP does not. A proper diagnosis should independently verify gradients, then compare matched initialization and scaling controls. Preserve these existing runs as an unchanged baseline; independently tested corrected runs must be labeled separately.

### Completed decay-0.1 sweep and matched plots

Run `20261103T125700Z-sweep` completed all 170 trials: 146 finite and 24 with nonfinite loss/gradient norm, with zero infrastructure failures. The actual-log audit again confirms 500 updates and 2.048M tokens per trial, identical base-LR histories across widths for all 17 candidates, matching data/tokenizer records for all eight workers, and unchanged training-core source. Its compute estimate is $1.9136 from 1.10319 worker-hours.

| Width | SP best peak LR | SP validation loss | μP best peak LR | μP validation loss |
| ---: | ---: | ---: | ---: | ---: |
| 48 | 0.03125 | 6.5445 | 0.0078125 | 6.8428 |
| 96 | 0.00390625 | 6.4715 | 0.0078125 | 6.6467 |
| 192 | 0.001953125 | 6.3900 | 0.0078125 | 6.5803 |
| 384 | 0.0009765625 | 6.3527 | 0.00390625 | 6.5717 |
| 768 | 0.0009765625 | 6.3816 | 0.00390625 | 6.6353 |

All three frozen μP proxies select 0.0078125; each incurs target regret 0.0133 relative to the best sampled target peak of 0.00390625. SP's width-48/96/192 regrets are 1.1533/0.2787/0.1170. All sampled minima are interior to the range. Adding base decay 0.1 changes the best μP target loss from 6.6371 to 6.6353 and SP from 6.3849 to 6.3816. These small changes in a single-seed pilot do not resolve the SP–μP quality gap or demonstrate what will happen at 20 TPP.

| Target recipe | SP best loss | Existing μP best loss |
| --- | ---: | ---: |
| Constant LR, decay 0 | 6.4085 | 6.5739 |
| Cosine + warmup, decay 0 | 6.3849 | 6.6371 |
| Cosine + warmup, decay 0.1 | 6.3816 | 6.6353 |

The new runs together took 36 minutes 55 seconds elapsed (07:38:41–08:15:36 EST on 2026-11-03, including a 14-second gap). Schedule-only runtime was 18:04; decay-0.1 runtime was 18:37. Total worker process time was 2.23098 hours and estimated worker compute was $3.8699 before credits and startup/controller overhead. Modal confirmed both apps stopped with zero tasks. No 20-TPP run was started.

The matched plots and numerical comparison are saved in `../modal-results/20261103-schedule-decay-comparison/`: `schedule-decay-comparison.png` compares validation curves for all widths and recipes; `schedule-decay-comparison-full.png` retains the full finite-loss range on a logarithmic axis; `target-schedule-decay-comparison.png` overlays the three recipes for 124M; `comparison.json` preserves numerical minima, curves and limitations. Each new run also has its own training/validation plots. The new decay and comparison plots explicitly label the inherited gradient mismatch. They describe this implementation, not a verified correct μP implementation.


### Corrected-gradient constant-LR rerun

The production change is one argument in the final `layernorm_backward` call: `width_mult` becomes `1.0f / width_mult`, matching the forward chain rule. Initialization, forward computation, attention scaling, AdamW and matrix LR/decay rules remain unchanged. `../modal-results/mup-gradient-fix-source-change.json` records the exact before/after trainer hashes and patch; all other training-core files match the extension baseline. The original 100 points predate the already documented unused RNG seed-metadata initialization; the source-change manifest also pins that exact historical exception, whose random draws and model initialization were tested unchanged.

Regression run `20261103T133213Z-mup_gradient` passed all 14 full-model cases and 460 sampled parameter-gradient checks against an independent float64 NumPy central-difference reference. It covers FP32/BF16, μP widths 48/96/192/384/768, and SP controls at 192/768. The before/fixed model receives identical nonzero weights, including tied embeddings and queries, so zero initialization cannot mask the derivative error. Forward logits are byte-identical before/after. The old code fails at each non-reference μP width in both precisions (265 failed components overall); the corrected code passes all components. SP and multiplier-one gradient arrays are byte-identical. Tolerances were fixed before the GPU checks and were not relaxed: FP32 `2e-5 + .002*abs(reference)`; BF16 `1e-3 + .08*abs(reference)`.

The one-layer fixture uses 12 heads, 64 tokens and vocabulary 8192. Initial harness attempts exposed a missing NVML link, the GELU kernel's minimum vectorized batch requirement, and a classifier unsigned-tail loop that stalls for tiny vocabularies. The fixture now meets the inherited kernels' constraints; GPT-2's production vocabulary already does. No classifier or GELU kernel was changed. The failed harness attempts are retained in the result directories.

Run `20261103T133301Z-mup_fix_training` also passes all six checks, including exact corrected-μP checkpoint resumption at multiplier 2 and unchanged trained model payloads against the original branch for SP and μP multiplier 1. Non-reference μP intentionally no longer has original-branch training parity; its replacement correctness check is the independent finite-difference regression. Earlier legacy numerical-reference test failures remain unresolved and are not reclassified.

The new sweep runs all 170 points fresh on four A10 workers: 500 updates, 2.048M tokens, constant LR, no warmup, zero weight decay, widths 48/96/192/384/768, and base LRs 2^-16 through 1. FineWeb revision, data order, initialization seed 42, validation protocol, and optimizer/architecture controls match the original run. It includes fresh SP controls and freezes narrow-model LR choices before the new target phase. HellaSwag remains disabled. This is an early-training diagnosis, not a 20-TPP experiment. The completed results follow.


#### Completed corrected-gradient sweep

Run `20261103T133353Z-sweep` completed all 170 fresh trials: 126 finite and 44 nonfinite-loss/gradient outcomes, with zero infrastructure failures. The log audit confirms 500 updates and 2.048M tokens for every point, all 17 identical constant-LR histories across widths/modes, eight matching worker data/tokenizer records, and the exact declared source change. It also verifies the original 100 rows were preserved in the combined baseline, including the previously tested RNG metadata-only source exception.

| Width | SP best LR | SP loss | Corrected μP best LR | Corrected μP loss | Pre-fix μP loss |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 48 | 0.001953125 | 6.5315 | 0.001953125 | 6.6077 | 6.7353 |
| 96 | 0.0009765625 | 6.4767 | 0.001953125 | 6.5877 | 6.6631 |
| 192 | 0.00048828125 | 6.4568 | 0.001953125 | 6.6196 | 6.6196 |
| 384 | 0.00048828125 | 6.4237 | 0.00390625 | 6.5982 | 6.5514 |
| 768 | 0.000244140625 | 6.4085 | 0.001953125 | 6.5927 | 6.5739 |

All three frozen μP proxies (48/96/192) now select 0.001953125, the best sampled 124M μP LR, giving zero measured target regret on this grid. Before correction, their target regrets were 0.0902/0.0246/0.0000. Width 384's new best μP LR is one grid step higher, 0.00390625; the result is not identical optimal LR at every width. Fresh SP proxy regrets are 0.3069/0.1390/0.0580. All sampled minima are interior to the grid.

The correction improves the best μP loss at widths 48 and 96 by 0.1276 and 0.0754, leaves the reference-width best unchanged, and worsens the measured best at 384/768 by 0.0468/0.0188. At 124M, SP again reaches 6.4085, while corrected μP reaches 6.5927 versus 6.5739 before. The gap is 0.1842 nats after versus 0.1654 before. Thus this verified gradient correction does not close the early-training quality gap, although proxy LR transfer is cleaner.

The 102 unchanged-math control points have 98 matching finite/diverged statuses. Of 76 pairs finite in both runs, 51 have identical logged training trajectories, 52 identical validation trajectories, and 54 identical final losses. Large-LR trajectories can vary substantially (maximum final-loss difference 10.6001), so these are not fully bit-reproducible 500-step sweeps. Best SP losses match exactly at widths 48/96/192/768; width 384 differs by 0.0014, with its grid minimum shifting between two close candidates. The best reference-width μP loss and LR match exactly. Interpret the small 124M μP change as a single-run observation rather than a statistically established degradation; the large residual gap remains the salient result.

At reference width 192, output and matrix-LR multipliers are one, and both attention scales are 0.25 (μP 4/16, SP 1/sqrt(16)). With matched synthetic weights, the new FP32 and BF16 reference-width sampled gradients are identical across SP/μP. Source inspection identifies zero initialization of the tied token embedding/readout and query matrices as the remaining initialization differences in this recipe. The unchanged 0.1628 reference-width loss gap makes an independent 2×2 initialization ablation a useful next diagnosis. That ablation has not been run. Neither this experiment nor the correction establishes whether a 20-TPP gap will persist.

The sweep took 17 minutes 27 seconds elapsed (08:33:52–08:51:19 EST), with 1.10634 worker-hours and $1.9191 estimated worker compute before credits/startup/controller overhead. Modal confirms the app stopped with zero tasks. New plots are in `../modal-results/20261103-gradient-fix-comparison/`: `width-gradient-fix-comparison.png`, its `-full.png` variant, and `target-gradient-fix-comparison.png`. `comparison.json` retains minima and per-trial control differences; the run's `verification.json` and `execution.json` retain the provenance audit and cost assumptions. Historical results were not overwritten.

#### Comparison against Microsoft's official μP implementations (2026-11-03)

This audit compares the corrected CUDA path actually used by the FineWeb sweeps against `microsoft/mup` at `19814971934ef91dd546f88e913fc963e096d11c` and the GPT-2 implementation in `microsoft/mutransformers` at `480287ce7b18a07a3432e8f2fbc0f0e5b71e2599`. It finds agreement in the inspected width-scaling rules. It does not establish complete forward/optimizer trajectory parity or explain the remaining loss gap.

Let `m = channels / 192` and `d = channels / 12`. Depth and head count remain fixed; both MLP dimensions grow with channels.

| Component | Official behavior | CUDA sweep behavior | Finding |
| --- | --- | --- | --- |
| Hidden matrix initialization | Normal initialization standard deviation scales as `1/sqrt(m)` for two varying dimensions | QKV, attention projection and both MLP matrices use that factor | Matches |
| Residual projection initialization | GPT-2 adds `1/sqrt(2*layers)` | Same for attention and MLP residual projections | Matches |
| Adam matrix learning rate | `base_lr/m` | Same four hidden matrix families | Matches |
| Embedding, norm and bias LR | Base LR for parameters with at most one varying dimension | Base LR | Matches |
| Matrix weight decay | Default MuAdamW multiplies the supplied coefficient by `m` | Same; embeddings retain base decay, norms and biases have zero decay | Matches with equivalent decay/no-decay groups |
| Readout | `MuReadout` applies `output_mult/m` to its input | Final normalized activation is divided by `m`, with implicit output multiplier 1 | Matches this setting |
| Readout backward | Autograd differentiates the same reciprocal factor | Corrected final layernorm backward applies `1/m` | Matches the chain rule; original code did not |
| Attention | Fixed `attn_mult/d` for μP | `4/d` in forward and backward | Matches the rule; the constant is a choice |

Sources: [official initialization](https://github.com/microsoft/mup/blob/19814971934ef91dd546f88e913fc963e096d11c/mup/init.py#L22), [optimizer grouping](https://github.com/microsoft/mup/blob/19814971934ef91dd546f88e913fc963e096d11c/mup/optim.py#L38), [readout](https://github.com/microsoft/mup/blob/19814971934ef91dd546f88e913fc963e096d11c/mup/layer.py#L54), [GPT-2 attention](https://github.com/microsoft/mutransformers/blob/480287ce7b18a07a3432e8f2fbc0f0e5b71e2599/mutransformers/models/gpt2/modeling_gpt2.py#L200), and [GPT-2 initialization](https://github.com/microsoft/mutransformers/blob/480287ce7b18a07a3432e8f2fbc0f0e5b71e2599/mutransformers/models/gpt2/modeling_gpt2.py#L475).

The official library derives groups from base-shape metadata; CUDA hard-codes these GPT-2 tensor families. The grouping is equivalent for this width-only architecture. It would need another audit if dimensions changed independently. Official MuAdamW's default `decoupled_wd=False` still multiplies matrix decay by `m`; AdamW's name alone is not a reason to remove this scaling. Its wrapper does not rescale Adam epsilon. Our beta values, clipping, decay exclusions and BF16 rounding are training/numerical choices to match explicitly in any trajectory comparison.

The attention constant 4 is deliberate: base head dimension 16 gives `4/16 = 1/sqrt(16)`. Microsoft's examples often use 8 to match SP at head dimension 64. Those constants do not indicate different μP width exponents. Likewise, output multiplier 1 is a valid setting, but this sweep has not tuned output or initialization multipliers.

Zero initialization needs a correction to the earlier diagnosis. The [official README](https://github.com/microsoft/mup/blob/19814971934ef91dd546f88e913fc963e096d11c/README.md#L219) recommends zero readout and query initialization to remove initialization transients in coordinate checks. The [official GPT-2 coordinate-check setup](https://github.com/microsoft/mutransformers/blob/480287ce7b18a07a3432e8f2fbc0f0e5b71e2599/tests/coordcheck/coordcheck.py#L97) enables both flags and reinitializes the model after assigning base shapes. Our CUDA implementation also zeros both, including the tied token embedding/readout. These choices are not evidence of an incorrect μP implementation. The GPT-2 initializer's default arguments allow nonzero versions, while the separate [Transformer example](https://github.com/microsoft/mup/blob/19814971934ef91dd546f88e913fc963e096d11c/examples/Transformer/model.py#L93) treats tied embeddings differently. There is no single mandatory initialization recipe across these examples.

With a zero tied embedding, token IDs initially contribute no embedding signal; learned position embeddings remain nonzero. The readout receives gradients on the first update, while the zero readout blocks the gradient into hidden activations on that update. This makes nonzero tied-embedding initialization a useful performance ablation, but does not prove it explains the 500-update gap. Zero queries are also a supported choice; changing both simultaneously would obscure which choice mattered.

There is a separate, reproducible bug in `train_gpt2.py:158` and `train_gpt2.py:169`. `_init_weights` tests `module._get_name()` against parameter paths and against `c_attn`; an `nn.Linear` reports `Linear`, so neither predicate succeeds. Consequently Python skips hidden-matrix width scaling and query zeroing. At `m=4`, its QKV standard deviation is 0.02 instead of 0.01. The tied-readout zeroing branch still executes. The FineWeb sweep builds `train_gpt2cu` and initializes inside CUDA host code, so this Python defect does not explain those results. It must be corrected before this Python file can serve as a μP reference. This audit does not change either trainer.

Verification artifacts are in `../modal-results/20261103-official-mup-audit/`. `check_scaling.py` executes the pinned official metadata/grouping functions, compiles the actual CUDA host scaling expressions, and compares all 16 tensor families at five widths: **80 optimizer groups and 30 sampled initialization-scale cases pass**. It also reproduces the Python dispatch defect using recording stand-ins. `comparison.json` records the results and local source hashes; `reference/` preserves the inspected upstream files. These are host-level formula/dispatch checks, not execution of a full official PyTorch model or optimizer tensor update. The earlier independent CUDA finite-difference tests remain separate evidence.

The next stronger check is a small FP32 CUDA-versus-official-MuAdamW comparison with identical initial tensors, inputs, loss reduction, optimizer settings and clipping, comparing logits, per-tensor gradients and updates for a few steps. Then a width-192 initialization ablation can isolate token-embedding and query choices cheaply. At width 192 all current μP width factors equal the SP factors, so matched initialization should remove the parametrization difference. This audit provides no basis for claiming the observed 124M loss gap will disappear at 20 TPP.

#### Loss-gap diagnosis: official optimizer and initialization ablations

The user authorized this diagnosis after reviewing the source comparison. `dev/test/modal_mup_diagnosis.py` runs bounded A10 workers; the numerical-reference environment pins CPU PyTorch 2.5.1 and the same official μP commit as above. The reference transformer uses independent PyTorch operations and the official `MuReadout`/`MuAdamW`; it deliberately does not use the defective local Python initializer. The CUDA harness includes the actual trainer. It compares full logits and every gradient component, then replays CUDA gradients through the official optimizer to isolate optimizer math. A separate reference evolves independently for three updates to measure trajectory drift.

Successful numerical run `20261103T141326Z-mup-parity` covers eight cases and 24 step comparisons: all five μP widths with two layers, an SP reference-width control, and 12-layer reference-width cases with random or zero tied embeddings/queries. FP32/TF32-off fixtures use vocabulary 8192 and 64 tokens. Maximum errors are 2.39e-6 for loss, 2.38e-6 for logits, 1.07e-6 for gradients, 1.19e-7 for official FP32 optimizer replay, and 5.99e-8 for official float64 replay. All comparisons pass the declared tolerances; independent short-trajectory losses differ by at most 2.87e-6. This is not BF16 end-to-end parity, production-vocabulary coverage, or a claim about every inherited kernel.

Two preceding runs (`20261103T141015Z-mup-parity` and `20261103T141139Z-mup-parity`) failed the strict CPU-FP32 optimizer comparison at width 48's first update, with maximum discrepancies 2.45e-6 and 2.33e-6. The 16 largest retained CUDA updates agree with an independent analytic first-step AdamW calculation within 1.82e-9; the CPU-FP32 values differ by up to 2.33e-6. A subsequent run adding official float64 replay passes both the original FP32 threshold and float64 checks. The precise cause of the intermittent CPU-FP32 discrepancy remains unresolved; no tolerances were relaxed and the earlier failures are retained. Setup attempts with unintended dependency upgrades and a missing uploaded helper were stopped and corrected before accepting reference results.

Reference-width experiment `20261103T141427Z-mup-ablation` completed all 78 trials: four embedding/query initialization combinations × six LRs (2^-13 through 2^-8) × three initialization seeds (42, 43, 44), plus six fresh SP controls. Every trial uses 12 layers, 12 heads, width 192, 500 updates/2.048M FineWeb tokens, BF16, constant LR, zero decay, identical token order, and the previous clipping/batch/validation settings. HellaSwag remains disabled. Only a disposable trainer copy receives initialization switches and a seed override; the production trainer is unchanged by this diagnosis. Initial-weight SHA-256 hashes and zero/nonzero checks are saved for every trial. Host checks additionally verify all 12 initialization combinations and byte-identical SP/random-random initial weights for all three seeds.

| Width-192 initialization | Best sampled mean LR | Mean validation loss | Across-seed standard deviation |
| --- | ---: | ---: | ---: |
| Zero tied embedding, zero query (current) | 0.00390625 | 6.6097 | 0.0561 |
| Random tied embedding, zero query | 0.00048828125 | 6.4611 | 0.0049 |
| Zero tied embedding, random query | 0.001953125 | 6.5699 | 0.0267 |
| Random tied embedding, random query | 0.00048828125 | 6.4583 | 0.0041 |

The current-initialization minimum is at the upper edge of this diagnostic grid, so two larger LRs are checked in the follow-up before interpreting its optimum. The random-embedding minima are interior. At their selected LR, random/random μP and SP have identical logged training trajectories and final losses for all three seeds. At the larger control LR 2^-9, two seeds differ substantially despite identical initial weights, consistent with the previously observed sensitivity/nondeterminism of larger-LR BF16 runs; those controls are retained rather than declared exact parity. Four of six seed-42 current-initialization final losses reproduce the earlier corrected sweep exactly; the other two differ by 0.0107 and 0.0245.

Randomizing only the tied embedding recovers almost all the measured reference-width loss difference relative to SP initialization, while randomizing only queries helps much less. The mechanism is consistent with removing an initially zero token representation: a zero shared readout also makes the input token embedding zero and blocks the first backward signal into hidden activations. This observation identifies an initialization choice with a large early-training effect; it does not make zero readout/query initialization an invalid μP rule.

The candidate-selection rule was saved before the reference experiment completed: prefer changing only the embedding if its best mean loss is within 0.01 nats of the best arm, otherwise choose the best arm; require an improvement exceeding 0.03 and an interior candidate minimum. It selects random embedding / zero query and freezes LR 2^-11 before target training. The follow-up uses three paired seeds at 124M: candidate LRs 2^-12, 2^-11 and 2^-10, current μP at its preceding-sweep LR 2^-9, and SP at its preceding-sweep LR 2^-12. Six additional width-192 current-initialization trials at LRs 2^-7 and 2^-6 check the reference boundary. The primary target comparison uses the frozen reference-selected LR; neighboring target LRs are secondary diagnostics, not a replacement selection.

Reference plots and the detailed audit are in `../modal-results/20261103T141427Z-mup-ablation/initialization-diagnosis.png` and `analysis.json`. The reference experiment used 0.35374 worker-hours, approximately $0.6136 of worker compute.

The follow-up `20261103T142142Z-mup-target` completed all 21 trials, followed by six query controls in `20261103T142644Z-mup-queries`. The extra reference-width baseline LRs 2^-7 and 2^-6 give mean losses 6.6848 and 7.0046, respectively, so the previous minimum at 2^-8 is now bracketed. Target controls use the earlier full-sweep LRs; candidate/query variants use their reference-width LRs. All trials completed 500 updates, and data/source/initialization/control audits pass.

| 124M configuration | Preset LR | Mean validation loss | Across-seed standard deviation |
| --- | ---: | ---: | ---: |
| Current μP: zero embedding, zero query | 0.001953125 | 6.5753 | 0.0153 |
| μP: zero embedding, random query | 0.001953125 | 6.6031 | 0.0635 |
| μP: random embedding, zero query | 0.00048828125 | 6.4339 | 0.0070 |
| μP: random embedding, random query | 0.00048828125 | 6.4323 | 0.0075 |
| SP control | 0.000244140625 | 6.4025 | 0.0103 |

Randomizing the tied embedding and using the proxy-selected LR reduces the measured target gap from 0.17277 to 0.03137 nats, an **81.84% reduction**. It improves all three paired seeds by 0.1586, 0.1286 and 0.1370 nats. The remaining gaps to SP are 0.0256, 0.0323 and 0.0362. The frozen LR 2^-11 also gives the best three-seed mean among the candidate target bracket (2^-12: 6.5133; 2^-11: 6.4339; 2^-10: 6.4641). This supports transfer from width 192 to 768 for this recipe, not a new full-width/global-LR sweep. Query randomization adds only 0.00157 nats of mean improvement once the embedding is random, and query randomization alone fails to recover the gap at its proxy-selected LR.

The recommended next training recipe is to initialize tied token embeddings with the existing constant standard deviation 0.02, keep query matrices zero, retain all corrected μP width rules, and use the newly tuned LR rather than automatically reusing the old zero-embedding optimum. The diagnostic builds implement this choice; the production trainer still has its prior initialization default. A 0.03-nat residual remains unexplained. The new initializer has not been checked across all five widths, with cosine/weight decay 0.1, or at 20 TPP. Neither the small optimizer discrepancies nor this ablation establishes whether that residual disappears with longer training.

The combined figure is `../modal-results/20261103T142142Z-mup-target/diagnosis-summary.png` (also PDF); `analysis.json` contains paired losses, all tested target LRs, the boundary extension and query controls. `completion.json` records all 105 training trials, the three numerical-check runs (including failed comparisons), unchanged production-trainer hash, cost assumptions and stopped app IDs. Measured worker compute across the diagnosis is approximately **$1.2461**, excluding startup, image-build, failed-startup and client overhead. Modal confirms all diagnostic apps stopped with zero tasks. No production CUDA math or initialization default was changed by this diagnosis.

#### Promotion and fresh full-width confirmation

After the diagnosis, the user authorized promoting random tied embeddings to the production default and rerunning both full-width experiments. The integration and historical diagnosis were committed as `5e75ca3`. The subsequent initializer change retains query zeros, all corrected μP scales, and standard-parametrization behavior. The Python initializer's hidden-matrix/query dispatch is also corrected. Historical zero-embedding experiments above remain unchanged artifacts; references to the previous default describe those runs.

The confirmation protocol repeats 170 fresh trials per recipe: SP and μP, widths 48/96/192/384/768, and base learning rates 2^-16 through 2^0. Both use 500 updates, 2.048M FineWeb tokens, 12 layers/12 heads, seed 42, identical data and validation, and HellaSwag disabled. Recipe one uses constant LR with no warmup or decay. Recipe two uses 50-step warmup, cosine decay to zero, and weight decay 0.1. Each recipe uses four A10 workers; proxy selections are saved before target trials. The recipes initially ran sequentially, then four cosine workers overlapped the single delayed constant-LR target worker, staying within the user's requested maximum of five GPU workers. Source commit and dirty status accompany file hashes. Minima and target transfer regret are reported as measured, including any failure to transfer.

Pre-sweep checks pass: 11 host control tests; byte-level initialization/checkpoint checks at all five widths; actual Python initializer checks at all five widths; 24 CUDA/official-μP numerical step comparisons in `20261103T144132Z-mup-parity`; and six GPU integration checks in `20261103T144132Z-mup_fix_training`. These include exact μP checkpoint resumption, unchanged SP trained payloads, and reference-width μP trained-payload equality against the original trainer with only its embedding initializer matched. The latter intentionally adjusts the reference initialization; it does not claim equality to original zero-embedding training.

The 20-TPP experiment is a subsequent step, not part of these reruns. At 124M, 20 TPP means approximately 2.49B tokens, roughly 1,216 times this pilot's horizon. The working hypotheses are that random embeddings improve early optimization, μP remains useful for width transfer at a fixed horizon, and the longer horizon may require a lower tuned LR. The residual loss gap may persist. Running every width at its own 20 TPP changes both width and token horizon; this must be distinguished from fixed-token width transfer. A small width-by-horizon pilot should precede a target run. This caution is consistent with [Bjorck et al., Scaling Optimal LR Across Token Horizons (ICLR 2025)](https://arxiv.org/abs/2409.19913), which finds horizon-dependent optimal LRs even with μP.

The promoted initializer is committed as `42ac74b`. Constant-LR confirmation `20261103T144242Z-sweep` completed 170 accepted trials (129 finite, 41 nonfinite, zero execution failures). The log/provenance audit passes for all trials, including 500 updates each, exact data identity, identical base-LR histories across widths, and the declared embedding-only change to the CUDA training core. All sampled minima are interior to the grid.

| Width | SP best LR | SP validation loss | μP best LR | μP validation loss |
| --- | ---: | ---: | ---: | ---: |
| 48 | 0.001953125 | 6.5315 | 0.00048828125 | 6.5565 |
| 96 | 0.0009765625 | 6.4767 | 0.0009765625 | 6.4610 |
| 192 | 0.00048828125 | 6.4568 | 0.00048828125 | 6.4609 |
| 384 | 0.00048828125 | 6.4237 | 0.00048828125 | 6.4369 |
| 768 | 0.000244140625 | 6.4073 | 0.00048828125 | 6.4341 |

The primary width-48 μP selection has zero target regret, as does width 192. Width 96 selects one grid step higher and has 0.0534 nats of target regret. SP proxy regrets are 0.3081, 0.1402 and 0.0509 for widths 48, 96 and 192. Thus the best μP LRs span a factor of two, versus eight for SP; this is useful approximate transfer, not exact invariance at every width. The target best-loss gap is 0.0268 nats, versus 0.1842 in the corrected zero-embedding sweep, an 85.45% reduction in this one-seed comparison. The earlier three-seed diagnosis remains the stronger variance check; this confirmation is a broad one-seed grid, not evidence that the residual is statistically zero.

Modal preempted the first constant-LR CPU coordinator. Its retry repeated the proxy phase while the original GPU workers finished; only the retry's unique records enter the accepted 170-point grid. The accepted worker time costs approximately $1.9729, and the discarded first-phase durations recorded in `recovery-event.json` add approximately $1.1056. These are worker-compute estimates, excluding startup, coordinator and billing overhead. A later target worker also started late because of allocation delay. Commit `04cb2a5` persists worker call IDs before waiting so a retried coordinator can reconnect; three host recovery tests pass. The cosine run uses this coordinator change with all 27 training-source files identical to the constant run. Plots and the accepted audit are in the constant run's `lr-transfer-zoom.png`, `lr-transfer.png`, `lr-transfer.pdf`, `analysis.json`, and `verification.json`; the clearly labeled `preview/` is an earlier partial snapshot and must not replace the accepted results.

Cosine/decay confirmation `20261103T151130Z-sweep` completed all 170 trials (145 finite, 25 nonfinite, zero execution failures). Its audit verifies the same data, 500 updates/2.048M tokens per trial, identical scheduled base LRs across widths, and no changes to the training core relative to the new constant-LR sweep. Both runs were launched from clean commits. The saved call journals contain four unique workers per phase with matching source hashes.

| Width | SP best peak LR | SP validation loss | μP best peak LR | μP validation loss |
| --- | ---: | ---: | ---: | ---: |
| 48 | 0.03125 | 6.5180 | 0.001953125 | 6.5034 |
| 96 | 0.00390625 | 6.4715 | 0.001953125 | 6.3965 |
| 192 | 0.001953125 | 6.3921 | 0.001953125 | 6.4316 |
| 384 | 0.0009765625 | 6.3540 | 0.001953125 | 6.3707 |
| 768 | 0.00048828125 | 6.3831 | 0.001953125 | 6.3929 |

All five μP widths select the same interior peak LR, 2^-9. The three frozen proxy selections therefore have zero target regret. SP proxy regrets are 1.1518, 0.2772 and 0.1155 for widths 48, 96 and 192; its sampled optimum shifts 64-fold from the smallest width to 124M. The target μP/SP best-loss gap is only 0.0098 nats. Together these runs support LR transfer after the initializer change for both requested recipes. They do not establish transfer of the weight-decay hyperparameter itself, a continuous global optimum, statistical equality of losses, or performance at 20 TPP. The two-recipe comparison changes warmup/schedule and decay together and cannot isolate their separate effects.

The combined comparison is in `../modal-results/20261103-random-embedding-confirmation/`: `schedule-decay-comparison.png` and `.pdf`, `schedule-decay-comparison-full.png`, `target-schedule-decay-comparison.png` and `.pdf`, and `comparison.json`. Each run also has its own training/validation plots and audit. All 340 accepted trials use seed 42; high-rate nonfinite trials are marked explicitly. The cosine workers cost approximately $1.9514. Total measured sweep worker compute, including the discarded constant proxy phase, is approximately **$5.0299**, excluding correctness checks, startup, coordinator, and billing overhead. Modal confirms both sweep apps and both preceding correctness apps stopped with zero tasks. `completion.json` in the combined directory records results, source hashes, cost scope, and stopped app IDs.

Before a future 20-TPP run, test a few longer horizons on small widths to separate width transfer from horizon-dependent LR tuning. Do not automatically reuse the 500-step peak LR or assume the remaining gap disappears. A 124M run needs about 2.49B tokens; use sufficient fresh training shards rather than repeatedly cycling this pilot's single 100M-token shard, and enable checkpoint/resume for long jobs. Batch size and the normalized schedule also need to be fixed for that comparison. No 20-TPP training was launched as part of this confirmation.

#### A10/A100 microbatch benchmark for the full target horizon

The user corrected the proposed target horizon to **20 TPP, not 2 TPP**.
For the current 124,475,904-parameter target this means 2,489,518,080 tokens
(607,793 whole 4,096-token updates). To isolate width transfer, each skinny
proxy would process that same target token budget, rather than its own 20 TPP.

`20261103T161531Z-microbatch-benchmark` completed all 24 short cases:
A10 and A100-40GB × widths 48/96/192/768 × microbatches 4/8/16. Each case
uses 120 updates, discards the first 20 timing observations, and keeps context
256, global batch 4,096, 12 layers/12 heads, BF16, constant base LR 2^-11,
zero decay, and the confirmed μP initialization/scalings. HellaSwag, sampling,
checkpointing, cuDNN, recomputation and outlier skipping are disabled. The
25 production training files (CUDA trainer and llmc headers) match the confirmed
cosine sweep; dataset revision and hashes match on both GPUs. All cases have
complete finite logs and no managed-memory fallback or skipped updates.

| Width | A10 microbatch 4 | A10 microbatch 16 | A100 microbatch 16 | A100 cost per full-target-token trial |
| --- | ---: | ---: | ---: | ---: |
| 48 | 3.72 h | 2.66 h | 1.65 h | $4.52 |
| 96 | 3.90 h | 2.84 h | 1.70 h | $4.65 |
| 192 | 4.71 h | 3.73 h | 1.95 h | $5.33 |
| 768 / 124M | 14.52 h | 13.05 h | 4.64 h | $12.67 |

These are **training-only extrapolations**, not completed 20-TPP runs. Rates
from https://modal.com/pricing on 2026-11-03 include the requested 8 CPU cores
and 32 GiB: $1.734624/hour for A10, $2.731824/hour for A100-40GB. Allow roughly
15–25% for validation, logging, checkpoints, startup and interruptions; longer
runs may have different sustained throughput. The 24-case benchmark used about
$0.077 of measured worker compute, excluding startup and billing overhead.
The app `ap-7Xfzqf91OYDOApS5om5UUA` is stopped with zero tasks.

A100/microbatch 16 is 2.25–2.41× faster than the former A10/microbatch-4 setup
for the proxies, and 3.13× faster for 124M. Against an A10 also using microbatch
16, A100 saves only 2–6% on widths 48/96 (too small to call a robust cost win
from one short measurement), 18% on width 192, and 44% on 124M. Its wall-time
advantage is larger. Allocated GPU memory for 124M/microbatch 16 was about
4.0–4.2 GiB; memory capacity is not the current microbatch limit. At fixed
global batch 4,096 and context 256, microbatch 16 already has one accumulation
step. Larger microbatches require a different global-batch recipe.

Native data shuffling depends on the physical microbatch, so training examples
are not paired across microbatch sizes. Validation uses the same 32,768-token
prefix. Finite short-run losses provide a sanity check, not numerical parity
or proof of unchanged long-run LR optima. Fix the selected microbatch across
widths before the transfer experiment.

With A100/microbatch 16, an adaptive five-trial proxy plan (three LRs at width
48, then the selected LR and a neighbor at width 96) is about 3.35 hours and
$22.85 before overhead: budget **4–5 hours and $25–30**, expanding if the
minimum remains unbracketed. Five LRs at each of widths 48/96/192 require
26.52 GPU-hours and $72.46 before overhead: roughly **8–9 hours with four
workers**, or **6–7 hours with five**, and **$85–95** allowing overhead.
These estimates exclude target validation runs; one 124M trial adds roughly
5–6 hours and $15–16. No full-horizon training was launched.

Reproduction: `dev/test/modal_microbatch_benchmark.py` launches two bounded
15-minute workers using `dev/test/run_microbatch_benchmark.py`. Raw logs,
source manifest, execution call IDs, device details, timings and the audited
`analysis.json` are retained under `../modal-results/20261103T161531Z-microbatch-benchmark/`.

#### Ninety-run cosine/decay budget on five A100 workers

The proposed scope is nine base LRs (2^-14 through 2^-6 inclusive) × all five
widths × SP/μP, using only cosine with 10% warmup, final fraction zero and
decay 0.1. Every trial receives the same 2.49B target tokens. This is 90 runs,
including 18 target-model trials; it is not the earlier 15-run proxy plan.

Follow-up `20261103T163903Z-microbatch-benchmark` measures all ten width/mode
combinations on one A100-40GB with microbatch 16 under this recipe, again using
120 updates and the final 100 timings. All ten cases have finite complete logs,
no skipped updates or managed-memory fallback, and unchanged production CUDA
source. SP and μP timings are nearly identical. Estimated hours per full run
are 1.58, 1.61, 1.86, 2.59 and 4.74 for widths 48, 96, 192, 384 and 768.

The full 90-run estimate is **222.77 GPU-hours / $608.58 before overhead**.
Five workers need about 45.06 hours with a balanced queue, or 47.17 hours when
all proxy results are frozen before the target phase. A 15–25% allowance gives
54.25–58.97 hours and $699.87–760.72; a practical rounded budget is therefore
**55–60 hours and $700–800**. This assumes every trial runs its full budget;
no savings from stopping divergent trials are included. Rates include the same
CPU/RAM reservation and are estimates, not an invoice or guaranteed runtime.

The short timing check cost about $0.028 of measured worker compute and its
app `ap-ly2F3H52zJBvdcVMzKThko` is stopped with zero tasks. No full sweep was
launched. Raw evidence and calculations are in that run's `90-run-estimate.json`.
Use `--profile cosine` with `modal_microbatch_benchmark.py` to reproduce the
single-worker, ten-case timing check; the default profile remains unchanged.

#### Matched muP-only training-horizon comparison: 1x, 8x, 16x

After narrowing the scope to muP only, the user authorized the clean comparison
at 1x/8x/16x the original token budget. Run `20261103T165112Z-mup-horizons`
completed **81/81 finite trials**, with no execution failures, skipped updates
or managed-memory fallback: widths 48/96/192 × nine peak base LRs 2^-14 through
2^-6 × 500/4,000/8,000 updates (2.048M/16.384M/32.768M tokens).

All runs use A100-40GB, BF16, physical batch 16, context 256, global batch 4,096,
12 layers/12 heads, seed 42, the corrected random-embedding/zero-query muP
initialization, reference width 192 and attention multiplier 4. Cosine decay
to zero, 10% warmup, and weight decay 0.1 are held fixed as a normalized recipe.
HellaSwag, sampling and checkpointing are disabled. Each trial starts fresh;
longer runs do not resume a checkpoint with a different schedule.

The baseline was rerun with the same microbatch as the longer runs. All three
horizons use the identical pinned 100M-token training shard. An executable
including the production C++ loader independently verifies that the shuffled
500/4,000/8,000-update sequences are nested prefixes without repeated batches.
The data hash, training source, actual recipe and every logged LR were audited.
Initial validation losses match exactly within each width across all trials.

Final validation is the same **1,048,576-token held-out prefix** for every trial,
evaluated at start and finish. This larger evaluation differs from the old
32,768-token plots, so compare these freshly matched horizons to one another;
their absolute losses should not be mixed with the older figures.

| Width | Best log2 peak LR: 1x | 8x | 16x | 16x loss penalty when reusing the 1x LR |
| --- | ---: | ---: | ---: | ---: |
| 48 | -9 | -8 | -8 | 0.0203 |
| 96 | -9 | -8 | -7 | 0.0280 |
| 192 | -9 | -7 | -7 | 0.0598 |

All sampled minima are interior. The best tested LR rises by 2–4x between 1x
and the longer budgets under this recipe. Minima are broad: at width 96/16x,
2^-8 is only 0.0013 nats worse than the winning 2^-7. The descriptive range of
sampled LRs within 0.01 nats of best is saved and plotted; it is not a confidence
interval. There is one initialization seed and only powers-of-two LR resolution.
This is a training-horizon effect for the chosen normalized schedule and fixed
decay coefficient, including their longer duration and cumulative regularization;
it does not isolate an independent data-size law or predict the 2.49B-token optimum.

Width transfer remains useful among these three proxies: choosing the LR at
width 48 costs at most 0.0223 nats at width 192 across the three budgets (0 at
1x, 0.0223 at 8x, 0.0186 at 16x). This does not test transfer to the 124M model.

Five workers finished in about **14 minutes 47 seconds** end to end. Measured
worker compute is approximately **$2.9988**, excluding startup/coordinator and
billing overhead. App `ap-H0tNUFqRqX5MKDHE8Dubge` is stopped with zero tasks.
The coordinator journals worker call IDs for reconnects; the host recovery check
passes. No full 20-TPP target run was launched.

Reproduce with `dev/test/modal_mup_horizon_sweep.py`; analyze with
`dev/test/plot_mup_horizons.py RESULTS_DIRECTORY`. Artifacts under
`../modal-results/20261103T165112Z-mup-horizons/` include the complete source/plan
manifests, data-order audit and indices, raw logs, `analysis.json`,
`mup-horizon-comparison.png`/PDF, the full-range PNG, and
`mup-optimal-lr-shift.png` with descriptive near-optimal ranges.

The requested log-log summary is `mup-optimal-lr-vs-tokens-loglog.png` (also PDF
and SVG), generated from the accepted analysis by `plot_mup_lr_vs_tokens.py`.
It takes the median best-tested LR across widths 48/96/192, all at depth 12:
0.001953125, 0.00390625 and 0.0078125 at 2.048M, 16.384M and 32.768M tokens.
Faint individual-width curves preserve the underlying variation. Unweighted
least squares in log space gives the descriptive fit
`LR = 0.0018517139 * (D / 2,048,000)^0.46153846`, drawn only within the measured
range. Three budgets, one seed and the twofold LR grid do not establish a scaling
law. Widths are distinct architectures, not statistical replicates. Numerical
inputs, fit specification and source-analysis hash are retained in the adjacent
`mup-optimal-lr-vs-tokens-loglog.json`. This plot uses existing results only.

#### Fourfold LR-grid refinement near the horizon-specific minima

Run `20261103T171930Z-mup-horizons-refined` adds quarter-step log2 LRs from
-10 through -6: adjacent rates differ by 2^0.25, approximately 1.189x, instead
of 2x. This gives 17 sampled rates in the critical interval. The 81 prior runs
are reused unchanged, and 108 new runs fill the gaps across the same three
widths and three horizons. Each curve has 21 total rates when the four older
low-LR points outside the refined interval are included. All 189 trials are
finite and pass the recipe, source, data-order and raw-metrics audits; all nine
sampled minima are strictly inside the refined interval.

The refinement preserves the matched cosine/decay recipe, initialization seed,
microbatch, token budgets and held-out prefix described above. Before reuse,
the coordinator verifies the original plan and dataset exactly and matches
the training CUDA, Makefile and llmc header hashes. Complete original evidence
is copied under `coarse/` inside the new run; original artifacts are preserved.

| Width | Best log2 peak LR: 1x | 8x | 16x | Final validation loss: 1x / 8x / 16x |
| --- | ---: | ---: | ---: | --- |
| 48 | -8.75 | -8.25 | -7.75 | 6.7647 / 6.0162 / 5.8130 |
| 96 | -8.50 | -7.75 | -7.25 | 6.7041 / 5.8269 / 5.5660 |
| 192 | -9.25 | -7.25 | -7.50 | 6.7078 / 5.7131 / 5.4000 |

Median best-tested LRs across widths are now **0.0023226701, 0.0046453403,
0.0055242717** at 2.048M, 16.384M and 32.768M tokens. The same unweighted
log-space fitting method gives
`LR = 0.0023382022 * (D / 2,048,000)^0.31730769`.
Descriptive log2 residual RMSE falls from 0.22646 on the coarse grid to 0.02831
on the refined grid. This is a fit to three measured medians, not an independent
prediction test. The finer grid changes the estimated optima without forcing
individual-width monotonicity: width 192's best tested LR decreases slightly
from 8x to 16x, while its 16x near-optimal range spans log2 LR -7.75 through -7.
Broad minima, one seed, different architectures and the schedule/decay horizon
coupling still limit interpretation. No extrapolation to full 20-TPP training
is supported by this refinement alone.

Choosing the width-48 optimum incurs validation-loss penalties at width 192
of 0.0095, 0.0456 and 0.0096 nats for 1x/8x/16x. These refined-grid comparisons
replace the coarse-grid point estimates above; near-optimal ranges and the
complete underlying curves remain available for assessing their sensitivity.

Five A100-40GB workers finished in **19 minutes 36 seconds** end to end.
Incremental measured worker compute is approximately **$3.9187**; the original
and refined runs together total **$6.9175**, excluding startup/coordinator and
billing overhead. App `ap-LAQCSy5UarhhJ94pvHM7MC` is stopped with zero tasks.

Reproduce with `dev/test/modal_mup_horizon_sweep.py --refine-from
20261103T165112Z-mup-horizons`. Analyze the new results directory with the same
two plotting scripts. Artifacts under
`../modal-results/20261103T171930Z-mup-horizons-refined/` include the audited
189-run analysis, original evidence, critical-region curves, full-range curves,
near-optimal-range plots and the updated log-log PNG/PDF/SVG. The adjacent
log-log JSON includes the coarse-grid comparison and source hashes.
