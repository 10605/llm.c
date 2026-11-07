"""Bounded FineWeb LR-transfer pilot, invoked only by the Modal test runner."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import re
import statistics
import struct
import subprocess
import time
from prepare_lr_transfer import make_plan

DATASET = 'chrisdryden/FineWebTokenizedGPT2'
HOURLY_RATE = 1.734624  # A10 + 8 CPU cores + 32 GiB; Modal list rates 2026-11-03.


class NumericalDivergence(ValueError):
    pass


def download(url, dest):
    import requests
    if not dest.exists():
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        temp = dest.with_suffix('.partial')
        with temp.open('wb') as f:
            for block in r.iter_content(4 * 1024 * 1024):
                f.write(block)
        temp.replace(dest)
    digest = hashlib.sha256()
    with dest.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return {'url': url, 'bytes': dest.stat().st_size, 'sha256': digest.hexdigest()}


def data_files(cache):
    import requests
    metadata = cache / 'fineweb-revision.json'
    if not metadata.exists():
        response = requests.get(f'https://huggingface.co/api/datasets/{DATASET}', timeout=60)
        response.raise_for_status()
        revision = response.json()['sha']
        metadata.write_text(json.dumps({'dataset': DATASET, 'revision': revision}))
    info = json.loads(metadata.read_text())
    revision = info['revision']
    dest = cache / 'fineweb' / revision
    dest.mkdir(parents=True, exist_ok=True)
    info['files'] = {}
    for name in ['fineweb_train_000001.bin', 'fineweb_val_000000.bin']:
        path = dest / name
        info['files'][name] = download(f'https://huggingface.co/datasets/{DATASET}/resolve/{revision}/{name}', path)
        with path.open('rb') as f:
            header = struct.unpack('<256i', f.read(1024))
        if header[:2] != (20240520, 1) or path.stat().st_size != 1024 + 2 * header[2]:
            raise ValueError(f'Invalid llm.c token shard: {path}')
        info['files'][name]['tokens'] = header[2]
    tokenizer = cache / 'gpt2_tokenizer.bin'
    info['tokenizer'] = download('https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main/gpt2_tokenizer.bin', tokenizer)
    return dest, tokenizer, info


def command(cmd, root, log, timeout):
    start = time.monotonic()
    with log.open('w') as f:
        f.write(json.dumps({'command': cmd}) + '\n')
        f.flush()
        try:
            p = subprocess.run(cmd, cwd=root, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
            code = p.returncode
        except subprocess.TimeoutExpired:
            code = -999
    return code, time.monotonic() - start


def read_metrics(out, expected_steps):
    text = (out / 'main.log').read_text()
    train = re.findall(r's:(\d+) trl:([^ ]+) lr:([^ ]+) norm:([^ ]+) time \(ms\):([^\n]+)', text)
    val = re.findall(r's:(\d+) tel:([^\n]+)', text)
    values = [float(row[1]) for row in train] + [float(row[3]) for row in train] + [float(row[1]) for row in val]
    if not all(math.isfinite(v) for v in values):
        raise NumericalDivergence('Nonfinite loss or gradient norm; see retained training log')
    if len(train) != expected_steps or not val or int(val[-1][0]) != expected_steps:
        raise ValueError('Incomplete training or final validation log')
    times = [float(row[4]) for row in train]
    return {'final_validation_loss': float(val[-1][1]),
            'last20_training_loss': statistics.mean(float(row[1]) for row in train[-20:]),
            'median_step_ms': statistics.median(times[4:]),
            'training_seconds': sum(times) / 1000, 'training_log': train, 'validation_log': val}


def freeze_selections(rows, results):
    picks = []
    for mode in ['sp', 'mup']:
        for width in [48, 96, 192]:
            valid = [r for r in rows if r['mode'] == mode and r['width'] == width and r['status'] == 'completed']
            if valid:
                best = min(valid, key=lambda r: r['final_validation_loss'])
                picks.append({k: best[k] for k in ['mode', 'width', 'base_learning_rate', 'final_validation_loss']})
    (results / 'selections_before_target.json').write_text(json.dumps(picks, indent=2))
    return picks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', choices=['benchmark', 'sweep'], required=True)
    parser.add_argument('--phase', choices=['all', 'narrow', 'target'], default='all')
    parser.add_argument('--worker', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--min-lr-exponent', type=int, default=-16)
    parser.add_argument('--max-lr-exponent', type=int, default=-7)
    parser.add_argument('--scheduler', choices=['constant', 'cosine', 'linear', 'wsd'], default='constant')
    parser.add_argument('--warmup-steps', type=int, default=0)
    parser.add_argument('--final-lr-fraction', type=float, default=1.0)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    for field in ['work', 'results', 'cache']:
        parser.add_argument('--' + field, type=Path, required=True)
    args = parser.parse_args()
    root = args.work / 'integration'
    args.results.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + (1000 if args.suite == 'benchmark' else 2500)
    data, tokenizer, provenance = data_files(args.cache)
    (args.results / 'dataset.json').write_text(json.dumps(provenance, indent=2))
    plan = make_plan(str(data/'fineweb_train_000001.bin'), str(data/'fineweb_val_000000.bin'), str(tokenizer),
                     min_lr_exponent=args.min_lr_exponent, max_lr_exponent=args.max_lr_exponent,
                     scheduler=args.scheduler, warmup_steps=args.warmup_steps,
                     final_lr_fraction=args.final_lr_fraction, weight_decay=args.weight_decay)
    plan['dataset'] = provenance
    plan['status'] = 'benchmarking' if args.suite == 'benchmark' else 'running'
    (args.results / 'plan.json').write_text(json.dumps(plan, indent=2))
    cap = subprocess.check_output(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'], text=True).splitlines()[0].replace('.', '').strip()
    build = ['make', 'train_gpt2cu', 'PRECISION=BF16', f'GPU_COMPUTE_CAPABILITY={cap}', 'FORCE_NVCC_O=3', 'USE_CUDNN=0', 'NO_MULTI_GPU=1', 'NO_USE_MPI=1']
    code, _ = command(build, root, args.results/'build.log', 300)
    if code:
        raise RuntimeError('Sweep build failed; see build.log')
    env = subprocess.check_output(['nvidia-smi'], text=True)
    (args.results/'environment.txt').write_text(env)
    trials = sorted(plan['trials'], key=lambda r: (r['width'] == 768, r['width'], r['mode'], r['base_learning_rate']))
    if args.suite == 'benchmark':
        trials = [r for r in trials if r['base_learning_rate'] == 2**-11]
    else:
        if args.phase != 'all':
            trials = [r for r in trials if (r['width'] == 768) == (args.phase == 'target')]
        trials = trials[args.worker::args.workers]
    records, selections = [], None
    summary = {'suite': args.suite, 'experiment': plan['experiment'], 'status': 'running', 'records': records}
    for index, trial in enumerate(trials):
        if time.monotonic() >= deadline - 60:
            break
        if args.suite == 'sweep' and args.phase == 'all' and trial['width'] == 768 and selections is None:
            selections = freeze_selections(records, args.results)
        cmd = trial['command'].copy()
        out = args.results / trial['name']
        cmd[cmd.index('-o')+1] = str(out)
        steps = 24 if args.suite == 'benchmark' else plan['training_steps']
        if args.suite == 'benchmark':
            cmd[cmd.index('-x')+1] = str(steps)
            cmd[cmd.index('-v')+1] = '500'
            cmd[cmd.index('-u')+1] = str(round(args.warmup_steps / plan['training_steps'] * steps))
        # No outlier-based skipped optimizer steps in the controlled experiment.
        cmd += ['-sl', '0', '-sg', '0']
        print(f'{args.suite.upper()} {index+1}/{len(trials)} {trial["name"]}', flush=True)
        code, seconds = command(cmd, root, args.results/(trial['name']+'.log'), max(1, min(1800, deadline-time.monotonic()-30)))
        record = {k: v for k, v in trial.items() if k not in {'command', 'shell_command'}}
        record.update({'command': cmd, 'steps': steps, 'tokens': steps*4096, 'seconds': seconds, 'exit_code': code})
        try:
            record.update(read_metrics(out, steps))
            if code:
                raise RuntimeError(f'Trainer exit {code}')
            record['status'] = 'completed'
        except NumericalDivergence as exc:
            record.update(status='diverged', error=str(exc))
        except Exception as exc:
            record.update(status='failed', error=str(exc))
        log_path = out / 'main.log'
        if log_path.exists():
            record['observed_steps'] = len(re.findall(r'^s:\d+ trl:', log_path.read_text(), re.MULTILINE))
            record['tokens'] = record['observed_steps'] * 4096
        records.append(record)
        summary = {'suite': args.suite, 'experiment': plan['experiment'], 'status': 'running', 'records': records}
        (args.results/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
        print(f'{record["status"]}: {seconds:.1f}s; val={record.get("final_validation_loss")}', flush=True)
        if args.suite == 'benchmark' and record['status'] != 'completed':
            break
    summary['status'] = 'completed' if len(records) == len(trials) else 'stopped'
    summary['seconds'] = time.monotonic()-started
    summary['estimated_compute_cost_usd'] = summary['seconds']/3600*HOURLY_RATE
    if args.suite == 'benchmark' and all(r['status'] == 'completed' for r in records) and len(records) == 10:
        # Scale actual per-step time; conservatively count startup as validation overhead.
        estimate = sum(10 * (r['median_step_ms']/1000*500 + max(0, r['seconds']-r['training_seconds'])*5.5) for r in records)
        summary['sweep_estimate_hours'] = [estimate/3600, estimate*1.3/3600]
        summary['sweep_estimate_usd'] = [estimate/3600*HOURLY_RATE, estimate*1.3/3600*HOURLY_RATE]
    if selections:
        for pick in selections:
            targets = [r for r in records if r['mode']==pick['mode'] and r['width']==768 and r['status']=='completed']
            transferred = next((r for r in targets if r['base_learning_rate']==pick['base_learning_rate']), None)
            if transferred:
                pick['target_loss_regret'] = transferred['final_validation_loss']-min(r['final_validation_loss'] for r in targets)
        summary['transfer'] = selections
    (args.results/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in summary.items() if k != 'records'}), flush=True)
    return int(summary['status'] != 'completed' or any(r['status'] == 'failed' for r in records))


if __name__ == '__main__':
    raise SystemExit(main())
