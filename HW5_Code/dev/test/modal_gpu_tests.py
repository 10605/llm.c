"""Bounded Modal GPU tests of the local working tree, with a pinned mup baseline.
Run from repo: modal run dev/test/modal_gpu_tests.py --suite single
Then: modal run dev/test/modal_gpu_tests.py --suite multi
Only source files are uploaded. Credentials, git data and existing artifacts stay local.
"""
from pathlib import Path
import datetime
import hashlib
import io
import json
import math
import subprocess
import tarfile
import modal

BASE = '706b1a5fc94de4bc16a4a3684eafaad6b9e0dada'
ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path('/tmp/llmc-tests/integration')
app = modal.App('llmc-mup-integration-tests')
volume = modal.Volume.from_name('llmc-integration-test-results', create_if_missing=True)
image = (
    modal.Image.from_registry('nvidia/cuda:12.4.1-devel-ubuntu22.04', add_python='3.11')
    .entrypoint([])
    .apt_install('build-essential', 'git', 'curl', 'openmpi-bin', 'libopenmpi-dev')
    .pip_install('numpy<2', 'requests', 'nvidia-cudnn-cu12==9.2.1.18')
    .run_commands(
        'git clone --depth 1 --branch v1.5.2 https://github.com/NVIDIA/cudnn-frontend.git /root/cudnn-frontend',
        'python -c "import nvidia.cudnn, pathlib; p=pathlib.Path(next(iter(nvidia.cudnn.__path__))); pathlib.Path(\'/usr/local/cudnn\').symlink_to(p); [pathlib.Path(\'/usr/local/lib\', f.name).symlink_to(f) for f in (p/\'lib\').glob(\'*.so*\')]"',
        'ldconfig',
    )
    .env({'OMP_NUM_THREADS': '4', 'OMPI_ALLOW_RUN_AS_ROOT': '1', 'OMPI_ALLOW_RUN_AS_ROOT_CONFIRM': '1'})
)


def execute(source, baseline, suite, run_id, execution_timeout=1100, extra_args=()):
    import traceback
    work = Path('/tmp/llmc-tests')
    work.mkdir(exist_ok=True)
    for name, data in [('integration', source), ('baseline', baseline)]:
        dest = work / name
        dest.mkdir(exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            archive.extractall(dest, filter='data')
    results = Path('/results') / run_id
    results.mkdir(parents=True, exist_ok=True)
    driver = ('run_fineweb_sweep.py' if suite in {'benchmark', 'sweep'} else
              'run_mup_gradient_test.py' if suite == 'mup_gradient' else 'run_gpu_integration.py')
    cmd = ['python', str(work / 'integration/dev/test' / driver), '--suite', suite,
           '--work', str(work), '--results', str(results), '--cache', '/results/cache']
    cmd += list(extra_args)
    try:
        p = subprocess.run(cmd, timeout=execution_timeout)
        code = p.returncode
    except Exception:
        (results / 'runner_error.txt').write_text(traceback.format_exc())
        code = 1
    finally:
        volume.commit()
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode='w:gz') as archive:
        for path in results.rglob('*'):
            if path.is_file() and path.suffix in {'.json', '.log', '.txt'}:
                archive.add(path, arcname=str(path.relative_to(results)))
    return code, output.getvalue()


@app.function(image=image, gpu='A10', cpu=8, memory=32768, timeout=1200, scaledown_window=2, volumes={'/results': volume})
def single(source: bytes, baseline: bytes, run_id: str, suite: str = 'single'):
    return execute(source, baseline, suite, run_id)


@app.function(image=image, gpu='A10:2', cpu=8, memory=32768, timeout=1200, scaledown_window=2, volumes={'/results': volume})
def multi(source: bytes, baseline: bytes, run_id: str):
    return execute(source, baseline, 'multi', run_id)


@app.function(image=image, gpu='A10', cpu=8, memory=32768, timeout=2700, max_containers=4, scaledown_window=2, volumes={'/results': volume})
def sweep(source: bytes, baseline: bytes, run_id: str, phase: str, worker: int,
          min_lr_exponent: int = -16, max_lr_exponent: int = -7, experiment: dict = None):
    settings = []
    for key, value in (experiment or {}).items():
        settings += ['--' + key.replace('_', '-'), str(value)]
    return execute(source, baseline, 'sweep', run_id, execution_timeout=2640,
                   extra_args=['--phase', phase, '--worker', str(worker), '--workers', '4',
                               '--min-lr-exponent', str(min_lr_exponent), '--max-lr-exponent', str(max_lr_exponent)] + settings)


def unpack(data, dest):
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        archive.extractall(dest, filter='data')


def durable_phase_calls(source, baseline, run_id, dest, phase, min_lr_exponent, max_lr_exponent, experiment):
    """Reconnect to existing workers when the CPU coordinator is restarted."""
    path = dest / f'{phase}-calls.json'
    spec = {'run_id': run_id, 'phase': phase, 'min_lr_exponent': min_lr_exponent,
            'max_lr_exponent': max_lr_exponent, 'experiment': experiment,
            'source_sha256': hashlib.sha256(source).hexdigest(),
            'baseline_sha256': hashlib.sha256(baseline).hexdigest()}
    if path.exists():
        saved = json.loads(path.read_text())
        if saved['spec'] != spec or len(saved['call_ids']) != 4 or len(set(saved['call_ids'])) != 4:
            raise ValueError('Saved worker calls do not match this experiment')
        return [modal.FunctionCall.from_id(call_id) for call_id in saved['call_ids']]
    calls = [sweep.spawn(source, baseline, f'{run_id}-{phase}-{i}', phase, i,
                         min_lr_exponent, max_lr_exponent, experiment) for i in range(4)]
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps({'spec': spec, 'call_ids': [call.object_id for call in calls]}, indent=2))
    pending.replace(path)
    volume.commit()
    return calls


def parallel_sweep(source, baseline, run_id, dest, min_lr_exponent=-16, max_lr_exponent=-7, prior_run='', experiment=None):
    from run_fineweb_sweep import freeze_selections
    records = []
    prior_seconds = 0
    if prior_run:
        if Path(prior_run).name != prior_run:
            raise ValueError('Prior run must be a run directory name')
        previous = json.loads((Path('/results')/prior_run/'summary.json').read_text())
        if previous['status'] != 'completed':
            raise ValueError('Prior sweep did not complete')
        records = previous['records']
        for record in records:
            cmd = record['command']
            previous_experiment = {
                'scheduler': cmd[cmd.index('-k')+1],
                'warmup_steps': int(cmd[cmd.index('-u')+1]),
                'final_lr_fraction': float(cmd[cmd.index('-q')+1]) if '-q' in cmd else 1.0,
                'weight_decay': float(cmd[cmd.index('-c')+1]),
            }
            if previous_experiment != experiment:
                raise ValueError('Cannot merge sweeps with different schedules or weight decay')
        prior_seconds = previous['worker_seconds'] + previous.get('prior_worker_seconds', 0)
        if any(min_lr_exponent <= round(math.log2(r['base_learning_rate'])) <= max_lr_exponent for r in records):
            raise ValueError('New LR range overlaps prior trials')
        (dest/'prior_run.json').write_text(json.dumps({'run':prior_run, 'trials':len(records),
            'note':'Original target results were already observed before this extension; this is a follow-up range check.'}, indent=2))
    prior_count = len(records)
    for phase in ['narrow', 'target']:
        calls = durable_phase_calls(source, baseline, run_id, dest, phase,
                                    min_lr_exponent, max_lr_exponent, experiment)
        failed = False
        for i, call in enumerate(calls):
            code, data = call.get()
            child = dest / f'{phase}-{i}'
            unpack(data, child)
            failed |= bool(code)
            if (child/'summary.json').exists():
                records += json.loads((child/'summary.json').read_text())['records']
        (dest/'summary.json').write_text(json.dumps({'status': 'failed' if failed else 'running',
                                                   'experiment': experiment, 'records': records}, indent=2))
        if failed:
            raise RuntimeError(f'{phase} phase contains failed trials; results preserved at {dest}')
        if phase == 'narrow':
            freeze_selections(records, dest)
            if not modal.is_local():
                volume.commit()
            print('Narrow-model learning rates saved before starting target trials.', flush=True)
    selected = json.loads((dest/'selections_before_target.json').read_text())
    for pick in selected:
        target = [r for r in records if r['mode']==pick['mode'] and r['width']==768 and r['status']=='completed']
        if not any(r['base_learning_rate']==pick['base_learning_rate'] for r in target):
            pick['target_loss_regret'] = None
            pick['target_status'] = 'Selected learning rate did not produce a finite target result'
            continue
        transferred = next(r for r in target if r['base_learning_rate']==pick['base_learning_rate'])
        pick['target_loss_regret'] = transferred['final_validation_loss'] - min(r['final_validation_loss'] for r in target)
    seconds = sum(json.loads((p/'summary.json').read_text())['seconds'] for p in dest.iterdir() if p.is_dir())
    summary = {'status': 'completed', 'workers': 4, 'worker_seconds': seconds,
               'experiment': experiment,
               'estimated_compute_cost_usd': seconds/3600*1.734624, 'prior_worker_seconds':prior_seconds,
               'prior_run':prior_run, 'new_trial_count':len(records)-prior_count,
               'transfer': selected, 'records': records}
    (dest/'summary.json').write_text(json.dumps(summary, indent=2))
    print(f'Completed {len(records)} trials; results: {dest}', flush=True)


@app.function(image=image, cpu=0.25, memory=1024, timeout=5500, scaledown_window=2, volumes={'/results': volume})
def coordinate_sweep(source: bytes, baseline: bytes, run_id: str, manifest: dict,
                     min_lr_exponent: int = -16, max_lr_exponent: int = -7, prior_run: str = '', experiment: dict = None):
    """Keep both phases running remotely even if the local client disconnects."""
    import sys
    import traceback
    checkout = Path('/tmp/sweep-controller')
    unpack(source, checkout)
    sys.path.insert(0, str(checkout/'dev/test'))
    dest = Path('/results') / run_id
    dest.mkdir(parents=True, exist_ok=True)
    (dest/'source_manifest.json').write_text(json.dumps(manifest, indent=2))
    (dest/'experiment.json').write_text(json.dumps(experiment, indent=2))
    code = 0
    try:
        if prior_run:
            previous_source = json.loads((Path('/results')/prior_run/'source_manifest.json').read_text())['files']
            core = [p for p in previous_source if p.startswith('llmc/') or p in {'train_gpt2.cu','Makefile'}]
            if any(manifest['files'].get(p) != previous_source[p] for p in core):
                raise ValueError('Cannot merge prior trials from a different training-core source revision')
        parallel_sweep(source, baseline, run_id, dest, min_lr_exponent, max_lr_exponent, prior_run, experiment)
    except Exception:
        code = 1
        (dest/'runner_error.txt').write_text(traceback.format_exc())
    finally:
        volume.commit()
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode='w:gz') as archive:
        for path in dest.rglob('*'):
            if path.is_file():
                archive.add(path, arcname=str(path.relative_to(dest)))
    return code, output.getvalue()

def package_sources():
    allowed = {'.c', '.cu', '.cuh', '.h', '.cpp', '.py', '.sh'}
    paths = [p for p in ROOT.iterdir() if p.is_file() and (p.suffix in allowed or p.name == 'Makefile')]
    for directory in ['llmc', 'dev/test', 'dev/data']:
        paths.extend(p for p in (ROOT / directory).rglob('*') if p.is_file() and p.suffix in allowed and '__pycache__' not in p.parts)
    out = io.BytesIO()
    manifest = {}
    with tarfile.open(fileobj=out, mode='w:gz') as archive:
        for path in sorted(set(paths)):
            name = str(path.relative_to(ROOT))
            archive.add(path, arcname=name)
            manifest[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    # The course checkout nests this source under HW5_Code/, while the pinned
    # upstream baseline stores it at the git root. Archive from that root.
    git_root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], cwd=ROOT, text=True).strip()
    baseline = subprocess.check_output(['git', 'archive', BASE, 'Makefile', 'llmc', 'train_gpt2.cu', 'test_gpt2.cu', 'dev/test'], cwd=git_root)
    return out.getvalue(), baseline, manifest


@app.local_entrypoint()
def main(suite: str = 'single', min_lr_exponent: int = -16, max_lr_exponent: int = -7, prior_run: str = '',
         scheduler: str = 'constant', warmup_steps: int = 0, final_lr_fraction: float = 1.0, weight_decay: float = 0.0):
    if suite not in {'single', 'multi', 'baseline', 'focused', 'cudnn_diagnostic', 'benchmark', 'sweep', 'mup_gradient', 'mup_fix_training'}:
        raise ValueError('Invalid suite')
    if not -30 <= min_lr_exponent <= max_lr_exponent <= 4:
        raise ValueError('Invalid learning-rate range')
    from prepare_lr_transfer import make_plan
    experiment = make_plan('train', 'val', 'tokenizer', scheduler=scheduler, warmup_steps=warmup_steps,
                           final_lr_fraction=final_lr_fraction, weight_decay=weight_decay)['experiment']
    if suite != 'sweep' and experiment != {'scheduler':'constant', 'warmup_steps':0, 'final_lr_fraction':1.0, 'weight_decay':0.0}:
        raise ValueError('Custom schedule and decay options currently apply only to the sweep suite')
    source, baseline, manifest = package_sources()
    provenance = {'base_commit': BASE, 'files': manifest,
                  'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  'source_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip())}
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + suite
    dest = ROOT.parent / 'modal-results' / run_id
    dest.mkdir(parents=True)
    (dest / 'source_manifest.json').write_text(json.dumps(provenance, indent=2))
    (dest / 'experiment.json').write_text(json.dumps(experiment, indent=2))
    print(f'Uploading {len(manifest)} source files ({len(source)} bytes); results: {dest}', flush=True)
    if suite == 'multi':
        code, data = multi.remote(source, baseline, run_id)
    elif suite == 'sweep':
        code, data = coordinate_sweep.remote(source, baseline, run_id, provenance,
                                            min_lr_exponent, max_lr_exponent, prior_run, experiment)
    else:
        code, data = single.remote(source, baseline, run_id, suite)
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        archive.extractall(dest, filter='data')
    print(f'Results saved to {dest}; suite exit={code}', flush=True)
    if code:
        raise RuntimeError(f'GPU suite reported failures; see {dest}')
