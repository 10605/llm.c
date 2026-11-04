"""Matched 1x/8x/16x muP learning-rate sweeps, with fixed data and microbatch."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import time

from prepare_lr_transfer import make_plan
from run_fineweb_sweep import command, data_files, read_metrics, NumericalDivergence

WIDTHS = (48, 96, 192)
HORIZONS = (1, 8, 16)
RATE = 2.731824
REVISION = 'cf916e6e01482048eeba8d72726d9cd4a7dba932'
VALIDATION_TOKENS = 1048576
STEP_MS = {48: 9.346, 96: 9.529, 192: 11.0173}


def make_trials(train, val, tokenizer):
    rows = []
    for horizon in HORIZONS:
        steps = 500*horizon
        plan = make_plan(str(train), str(val), str(tokenizer), steps=steps,
            min_lr_exponent=-14, max_lr_exponent=-6, scheduler='cosine',
            warmup_steps=steps//10, final_lr_fraction=0.0, weight_decay=0.1)
        for row in plan['trials']:
            if row['mode'] != 'mup' or row['width'] not in WIDTHS:
                continue
            row = {k:v for k,v in row.items() if k != 'shell_command'}
            row['name'] = f"h{horizon}-"+row['name']
            row.update(horizon=horizon, steps=steps, warmup_steps=steps//10,
                       microbatch=16, context=256, total_batch=4096)
            cmd = row['command']
            # Evaluate the same larger held-out prefix, only at start and end.
            for flag, value in {'-b':16, '-m':VALIDATION_TOKENS//4096,
                                '-v':steps, '-o':row['name']}.items():
                cmd[cmd.index(flag)+1] = str(value)
            cmd += ['-sl', '0', '-sg', '0']
            rows.append(row)
    return rows


def make_refinement_trials(prior_trials):
    """Fill quarter-octave intervals from 2^-10 through 2^-6, reusing integers."""
    rows = []
    for horizon in HORIZONS:
        for width in WIDTHS:
            group = [r for r in prior_trials if r['horizon']==horizon and r['width']==width]
            assert len(group)==9 and len({r['base_learning_rate'] for r in group})==9
            existing = {r['base_learning_rate'] for r in group}
            for quarter in range(-40, -23):
                lr = 2.0**(quarter/4)
                if lr in existing:
                    continue
                row = copy.deepcopy(group[0])
                row.update(name=f'h{horizon}-mup-c{width}-lrq{quarter}',
                    base_learning_rate=lr, matrix_learning_rate=lr/(width/192),
                    lr_exponent=quarter/4, refinement=True)
                row['command'][row['command'].index('-l')+1] = repr(lr)
                row['command'][row['command'].index('-o')+1] = row['name']
                rows.append(row)
    assert len(rows)==108 and len({r['name'] for r in rows})==108
    return rows


def partition(trials, workers=5):
    groups, loads = [[] for _ in range(workers)], [0.0]*workers
    for trial in sorted(trials, key=lambda r:r['steps']*STEP_MS[r['width']], reverse=True):
        i = loads.index(min(loads))
        groups[i].append(trial)
        loads[i] += trial['steps']*STEP_MS[trial['width']]/1000
    # Finish the matched baseline first within each worker, without altering loads.
    for group in groups:
        group.sort(key=lambda r:(r['horizon'], r['width'], r['base_learning_rate']))
    return groups, loads


def audit_data_order(root, train, output):
    """Execute the production C++ loader initialization independently per horizon."""
    source = output/'data_order_check.cpp'
    source.write_text('''#include <cmath>
#include "llmc/dataloader.h"
int main(int argc, char** argv) {
    DataLoader loader;
    dataloader_init(&loader, argv[1], 16, 256, 0, 1, 1);
    int steps = atoi(argv[2]);
    if ((size_t)steps > loader.shard_num_samples) return 2;
    for (int i=0; i<steps; i++) printf("%d\\n", loader.intra_shard_indices[i]);
    dataloader_free(&loader);
}
''')
    binary = output/'data_order_check'
    subprocess.run(['g++','-std=c++17','-O2','-I',str(root),str(source),'-o',str(binary)],
                   check=True, capture_output=True, timeout=60)
    orders = {}
    for h in HORIZONS:
        raw = subprocess.check_output([str(binary), str(train), str(h*500)], timeout=60)
        values = [int(v) for v in raw.splitlines()]
        assert len(values) == h*500 and len(set(values)) == len(values)
        orders[h] = values
        (output/f'data-order-{h}x.txt').write_bytes(raw)
    assert orders[1] == orders[8][:500] == orders[16][:500]
    assert orders[8] == orders[16][:4000]
    report = dict(status='passed', microbatch=16, context=256, shuffle_seed=42,
        nested_training_prefixes=True, no_training_batch_reuse=True,
        prefix_sha256={str(h):hashlib.sha256((output/f'data-order-{h}x.txt').read_bytes()).hexdigest()
                       for h in HORIZONS},
        source_sha256=hashlib.sha256((root/'llmc/dataloader.h').read_bytes()).hexdigest())
    (output/'data-order-audit.json').write_text(json.dumps(report, indent=2))
    return report


def verify_metrics(out, trial):
    metrics = read_metrics(out, trial['steps'])
    if [int(r[0]) for r in metrics['training_log']] != list(range(trial['steps'])):
        raise ValueError('Missing or duplicate updates')
    params = (out/'run_parameters.txt').read_text()
    expected = ['batch=16 sequence=256 total_batch=4096 accumulation=1 ranks=1',
                f"layers=12 channels={trial['width']} heads=12", 'mup=1 ',
                'scheduler=cosine', f"warmup={trial['warmup_steps']} final_fraction=0 weight_decay=0.100000001",
                'val_batches=256 sample_every=0', 'lossz=0 gradz=0 overfit=0']
    if not all(value in params for value in expected):
        raise ValueError('Actual recipe disagrees with plan')
    # The native text log rounds LR to six decimal places.
    peak = trial['base_learning_rate']
    for row in metrics['training_log']:
        step, lr = int(row[0]), float(row[2])
        warm = trial['warmup_steps']
        expected_lr = peak*(step+1)/warm if step < warm else peak*.5*(1+math.cos(math.pi*(step-warm)/(trial['steps']-warm)))
        if abs(lr-expected_lr) > 5.2e-7:
            raise ValueError('Logged LR schedule differs from declared cosine schedule')
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['root','results','plan']:
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    plan = json.loads(args.plan.read_text())
    started = time.monotonic()
    max_seconds = plan.get('max_seconds',1120)
    assert 60 <= max_seconds <= 1400
    deadline = started+max_seconds
    data, tokenizer, provenance = data_files(Path('/results/cache'))
    if provenance != plan['dataset'] or provenance['revision'] != REVISION:
        raise ValueError('Worker data differs from frozen dataset')
    (args.results/'dataset.json').write_text(json.dumps(provenance, indent=2))
    cap = subprocess.check_output(['nvidia-smi','--query-gpu=compute_cap','--format=csv,noheader'],
                                 text=True).splitlines()[0].replace('.','').strip()
    code, _ = command(['make','train_gpt2cu','PRECISION=BF16',f'GPU_COMPUTE_CAPABILITY={cap}',
        'FORCE_NVCC_O=3','USE_CUDNN=0','NO_MULTI_GPU=1','NO_USE_MPI=1'],
        args.root, args.results/'build.log', 300)
    if code:
        raise RuntimeError('Build failed')
    (args.results/'environment.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
    summary = dict(status='running', records=[], dataset=provenance,
                   source_sha256=hashlib.sha256((args.root/'train_gpt2.cu').read_bytes()).hexdigest())
    for i, trial in enumerate(plan['trials']):
        if time.monotonic() > deadline-30:
            break
        out = args.results/trial['name']
        cmd = trial['command'].copy()
        cmd[cmd.index('-o')+1] = str(out)
        print(f"{args.results.name}: {i+1}/{len(plan['trials'])} {trial['name']}", flush=True)
        code, seconds = command(cmd, args.root, args.results/(trial['name']+'.log'),
                                min(200, max(1, deadline-time.monotonic()-15)))
        row = {k:v for k,v in trial.items() if k != 'command'}
        row.update(command=cmd, exit_code=code, seconds=seconds)
        try:
            if code:
                raise RuntimeError(f'Trainer exit {code}')
            row.update(verify_metrics(out, trial), status='completed')
        except NumericalDivergence as exc:
            row.update(status='diverged', error=str(exc))
        except Exception as exc:
            row.update(status='failed', error=str(exc))
        text = (out/'main.log').read_text() if (out/'main.log').exists() else ''
        row['observed_steps'] = len(re.findall(r'^s:\d+ trl:', text, re.M))
        # Even divergent cases must have completed their allocated budget here.
        if row['observed_steps'] != trial['steps']:
            row.update(status='failed', error='Incomplete trial')
        summary['records'].append(row)
        summary.update(worker_seconds=time.monotonic()-started)
        (args.results/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
        print(f"{trial['name']}: {row['status']} val={row.get('final_validation_loss')} ({seconds:.1f}s)", flush=True)
    summary.update(status='completed' if len(summary['records'])==len(plan['trials'])
                   and all(r['status']!='failed' for r in summary['records']) else 'failed',
                   worker_seconds=time.monotonic()-started)
    summary['estimated_compute_usd'] = summary['worker_seconds']/3600*RATE
    (args.results/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    if summary['status'] != 'completed':
        raise RuntimeError('Worker incomplete or failed; results retained')


if __name__ == '__main__':
    main()
