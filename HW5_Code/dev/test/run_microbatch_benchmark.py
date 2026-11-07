"""Short throughput measurements; never execute a full 20-TPP trial."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import subprocess
import time

from prepare_lr_transfer import make_plan, parameter_count
from run_fineweb_sweep import command, data_files, read_metrics

WIDTHS = (48, 96, 192, 768)
MICROBATCHES = (4, 8, 16)
STEPS = 120
WARMUP = 20
GLOBAL_BATCH = 4096
CONTEXT = 256
VALIDATION_TOKENS = 32768
RATES = {'A10': 1.734624, 'A100-40GB': 2.731824}
REVISION = 'cf916e6e01482048eeba8d72726d9cd4a7dba932'


def cases(train, val, tokenizer, output, profile='microbatch'):
    if profile not in {'microbatch', 'cosine'}:
        raise ValueError('Unknown benchmark profile')
    cosine = profile == 'cosine'
    widths = (48, 96, 192, 384, 768) if cosine else WIDTHS
    batches = (16,) if cosine else MICROBATCHES
    plan = make_plan(str(train), str(val), str(tokenizer), steps=STEPS,
                     min_lr_exponent=-11, max_lr_exponent=-11,
                     scheduler='cosine' if cosine else 'constant',
                     warmup_steps=12 if cosine else 0,
                     final_lr_fraction=0.0 if cosine else 1.0,
                     weight_decay=0.1 if cosine else 0.0)
    rows = []
    for trial in plan['trials']:
        if (not cosine and trial['mode'] != 'mup') or trial['width'] not in widths:
            continue
        for batch in batches:
            name = (f"{trial['mode']}-" if cosine else '') + f"c{trial['width']}-b{batch}"
            cmd = trial['command'].copy()
            for flag, value in {'-b': batch, '-o': str(output/name), '-v': STEPS,
                                '-m': VALIDATION_TOKENS//(batch*CONTEXT)}.items():
                cmd[cmd.index(flag)+1] = str(value)
            cmd += ['-sl', '0', '-sg', '0']
            rows.append(dict(name=name, mode=trial['mode'], width=trial['width'], microbatch=batch,
                             accumulation=GLOBAL_BATCH//(batch*CONTEXT), command=cmd))
    random.Random(42).shuffle(rows)
    return rows


def parse_case(out, trial):
    result = read_metrics(out, STEPS)
    rows = result['training_log']
    if [int(r[0]) for r in rows] != list(range(STEPS)):
        raise ValueError('Missing or duplicated updates')
    times = [float(r[4]) for r in rows[WARMUP:]]
    if not all(math.isfinite(t) and t > 0 for t in times):
        raise ValueError('Invalid timing')
    actual = (out/'run_parameters.txt').read_text()
    expected = (f"batch={trial['microbatch']} sequence={CONTEXT} total_batch={GLOBAL_BATCH} "
                f"accumulation={trial['accumulation']} ranks=1")
    if expected not in actual or f"channels={trial['width']} heads=12" not in actual:
        raise ValueError('Actual architecture/batch disagrees with benchmark')
    mean = statistics.mean(times)
    result.update(mean_step_ms=mean, median_step_ms=statistics.median(times),
                  min_step_ms=min(times), max_step_ms=max(times),
                  measured_steps=len(times), tokens_per_second=GLOBAL_BATCH*1000/mean)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ['root', 'results', 'cache']:
        parser.add_argument('--'+field, type=Path, required=True)
    parser.add_argument('--gpu', choices=list(RATES), required=True)
    parser.add_argument('--profile', choices=['microbatch', 'cosine'], default='microbatch')
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started+780
    data, tokenizer, provenance = data_files(args.cache)
    if provenance['revision'] != REVISION:
        raise ValueError('Dataset revision differs from confirmed transfer sweeps')
    (args.results/'dataset.json').write_text(json.dumps(provenance, indent=2))
    (args.results/'environment.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
    cap = subprocess.check_output(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
                                  text=True).splitlines()[0].replace('.', '').strip()
    code, build_seconds = command(['make', 'train_gpt2cu', 'PRECISION=BF16',
        f'GPU_COMPUTE_CAPABILITY={cap}', 'FORCE_NVCC_O=3', 'USE_CUDNN=0',
        'NO_MULTI_GPU=1', 'NO_USE_MPI=1'], args.root, args.results/'build.log', 300)
    if code:
        raise RuntimeError('Build failed; see build.log')
    trials = cases(data/'fineweb_train_000001.bin', data/'fineweb_val_000000.bin', tokenizer, args.results, args.profile)
    summary = dict(gpu=args.gpu, hourly_rate_usd=RATES[args.gpu], build_seconds=build_seconds,
        profile=args.profile,
        source_sha256=hashlib.sha256((args.root/'train_gpt2.cu').read_bytes()).hexdigest(),
        context=CONTEXT, total_batch=GLOBAL_BATCH, steps=STEPS, discarded_timing_steps=WARMUP,
        validation_tokens=VALIDATION_TOKENS, target_20tpp_tokens=20*parameter_count(768),
        training_order_note='Native shuffle depends on microbatch; cross-batch losses are sanity checks, not numerical parity.',
        records=[])
    for i, trial in enumerate(trials):
        if time.monotonic() > deadline-60:
            break
        print(f"{args.gpu} {i+1}/{len(trials)} {trial['name']}", flush=True)
        log = args.results/(trial['name']+'.log')
        code, seconds = command(trial['command'], args.root, log,
                                min(90, max(1, deadline-time.monotonic()-15)))
        row = dict(trial, exit_code=code, seconds=seconds)
        try:
            if code:
                raise RuntimeError(f'Trainer exited with {code}')
            row.update(parse_case(args.results/trial['name'], trial), status='completed')
            row['cost_per_million_tokens_usd'] = RATES[args.gpu]/3600*1e6/row['tokens_per_second']
            row['full_target_20tpp_training_hours'] = summary['target_20tpp_tokens']/row['tokens_per_second']/3600
        except Exception as exc:
            oom = bool(re.search(r'out of memory|cudaErrorMemoryAllocation', log.read_text(), re.I))
            row.update(status='out_of_memory' if oom else 'failed', error=str(exc))
        summary['records'].append(row)
        summary.update(worker_seconds=time.monotonic()-started, status='running')
        (args.results/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    summary['status'] = ('completed' if len(summary['records']) == len(trials)
                         and all(r['status'] in {'completed', 'out_of_memory'} for r in summary['records'])
                         else 'failed')
    summary['worker_seconds'] = time.monotonic()-started
    summary['estimated_worker_compute_usd'] = summary['worker_seconds']/3600*RATES[args.gpu]
    (args.results/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    if summary['status'] != 'completed':
        raise RuntimeError('Benchmark incomplete or failed; retained all logs')


if __name__ == '__main__':
    main()
