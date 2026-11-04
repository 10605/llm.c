"""GPU integration checks. Run in an isolated directory; invoked by modal_gpu_tests.py."""
from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
import traceback
import numpy as np
import requests

parser = argparse.ArgumentParser()
parser.add_argument('--suite', choices=['single', 'multi', 'baseline', 'focused', 'cudnn_diagnostic', 'mup_fix_training'], required=True)
parser.add_argument('--work', type=Path, required=True)
parser.add_argument('--results', type=Path, required=True)
parser.add_argument('--cache', type=Path, required=True)
args = parser.parse_args()
ROOT = args.work / 'integration'
BASE = args.work / 'baseline'
RESULTS = args.results
RESULTS.mkdir(parents=True, exist_ok=True)
args.cache.mkdir(parents=True, exist_ok=True)
records = []
command_index = 0


def run(cmd, cwd=ROOT, expected=0, timeout=180):
    global command_index
    command_index += 1
    label = f'{command_index:03d}'
    print(f'COMMAND {label}: {cmd}', flush=True)
    begin = time.monotonic()
    p = subprocess.run(list(map(str, cmd)), cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    output = p.stdout
    (RESULTS / f'{label}.log').write_text(f'cwd: {cwd}\ncommand: {cmd}\nexit: {p.returncode}\nseconds: {time.monotonic()-begin:.2f}\n\n' + output)
    if (expected == 0 and p.returncode != 0) or (expected != 0 and p.returncode == 0):
        print(output[-1600:], flush=True)
        raise AssertionError(f'Command {label} exit={p.returncode}, expected {expected}; see {label}.log')
    print(f'COMMAND {label}: exit={p.returncode} ({time.monotonic()-begin:.1f}s)', flush=True)
    return output


def test(name, fn):
    print(f'TEST {name}', flush=True)
    start = time.monotonic()
    try:
        detail = fn()
        row = {'name': name, 'status': 'passed', 'detail': detail}
    except Exception:
        row = {'name': name, 'status': 'failed', 'detail': traceback.format_exc()}
        print(row['detail'], flush=True)
    row['seconds'] = time.monotonic() - start
    records.append(row)
    (RESULTS / 'summary.json').write_text(json.dumps({'suite': args.suite, 'tests': records}, indent=2))
    print(f'{row["status"].upper()}: {name}', flush=True)
    return row['status'] == 'passed'


def download(name):
    dest = args.cache / name
    if not dest.exists():
        url = 'https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main/' + name
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        temp = dest.with_suffix('.partial')
        with temp.open('wb') as f:
            for chunk in response.iter_content(1024 * 1024):
                f.write(chunk)
        temp.replace(dest)
    assert dest.stat().st_size > 1024
    return dest


def fixtures():
    names = ['tiny_shakespeare_train.bin', 'tiny_shakespeare_val.bin', 'gpt2_tokenizer.bin']
    if args.suite in {'single', 'baseline', 'focused', 'cudnn_diagnostic'}:
        names += ['gpt2_124M.bin', 'gpt2_124M_bf16.bin', 'gpt2_124M_debug_state.bin']
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(download, names))
    for root in [ROOT, BASE]:
        for path in files:
            dest = root / path.name
            if path.name.startswith('tiny_shakespeare'):
                dest = root / 'dev/data/tinyshakespeare' / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                dest.symlink_to(path)
    return {p.name: p.stat().st_size for p in files}


CAP = None

def build(root=ROOT, precision='BF16', cudnn=False, multi=False):
    opts = [f'PRECISION={precision}', f'GPU_COMPUTE_CAPABILITY={CAP}', 'FORCE_NVCC_O=1',
            f'USE_CUDNN={int(cudnn)}', f'NO_MULTI_GPU={0 if multi else 1}', f'NO_USE_MPI={0 if multi else 1}']
    if cudnn:
        link = Path('/usr/local/lib/libcudnn.so')
        if not link.exists():
            link.symlink_to('/usr/local/cudnn/lib/libcudnn.so.9')
        opts += ['NVCC_INCLUDES=-I/root/cudnn-frontend/include -I/usr/local/cudnn/include -L/usr/local/lib']
    targets = ['train_gpt2cu'] if multi else ['train_gpt2cu', 'test_gpt2cu']
    run(['make', *targets, *opts], cwd=root, timeout=300)
    suffix = f'{precision.lower()}-{"cudnn" if cudnn else "cuda"}-{"multi" if multi else "single"}'
    for target in targets:
        shutil.copy2(root / target, root / (target + '-' + suffix))
    return suffix


def train(name, extra=(), exe=None, root=ROOT, multi=False, old_paths=False):
    out = root / 'runs' / name
    out.parent.mkdir(exist_ok=True)
    exe = exe or root / 'train_gpt2cu'
    cmd = (["mpirun", "--allow-run-as-root", "-np", "2"] if multi else []) + [str(exe)]
    cmd += ['-b', '2', '-t', '64', '-d', '256' if multi else '128', '-x', '6', '-m', '2', '-v', '20', '-s', '0', '-n', '3', '-o', str(out), '-f', '0']
    if old_paths:
        cmd += ['-i', str(args.cache/'tiny_shakespeare_train.bin'), '-j', str(args.cache/'tiny_shakespeare_val.bin')]
    else:
        cmd += ['-id', str(args.cache/'tiny_shakespeare_train.bin'), '-iv', str(args.cache/'tiny_shakespeare_val.bin'), '-it', str(args.cache/'gpt2_tokenizer.bin')]
    cmd += list(map(str, extra))
    text = run(cmd, cwd=root)
    assert not re.search(r'\bloss\s+(?:nan|inf)\b|\bnorm\s+(?:nan|inf)\b', text, re.I), text[-3000:]
    return out


def payload_hash(file):
    sha = hashlib.sha256()
    with file.open('rb') as f:
        f.seek(1024)
        for block in iter(lambda: f.read(4*1024*1024), b''):
            sha.update(block)
    return sha.hexdigest()


def compare_models(a, b, step=6):
    name = f'model_{step:08d}.bin'
    ha, hb = payload_hash(a/name), payload_hash(b/name)
    assert ha == hb, f'Model payloads differ: {a} vs {b}'
    return {'model_payload_sha256': ha}


ARCH = ['-e', 'd12', '-1d', '2', '-1c', '128', '-1h', '2']


def resume_case(me, multi=False, zero=0, suffix=''):
    extra = ARCH + ['-me', str(me), '-mb', '64', '-ma', '8', '-k', 'linear', '-u', '1', '-q', '0.1', '-z', str(zero)]
    prefix = f'resume-{me}-{zero}{suffix}'
    full = train(prefix+'-full', extra, multi=multi)
    split = train(prefix+'-split', extra + ['-es', '3'], multi=multi)
    assert (split/'DONE_00000003').exists() and not (split/'DONE_00000006').exists()
    # No architecture or muP flags on resume: recover metadata.
    train(prefix+'-split', ['-y', '1', '-ss', '1', '-k', 'linear', '-u', '1', '-q', '0.1', '-z', str(zero)], multi=multi)
    detail = compare_models(full, split)
    for rank in range(2 if multi else 1):
        name = f'state_00000006_{rank:05d}.bin'
        left, right = (full/name).read_bytes(), (split/name).read_bytes()
        if left != right:
            differing = np.flatnonzero(np.frombuffer(left, dtype=np.uint8) != np.frombuffer(right, dtype=np.uint8))
            # Locate whether differences concern headers, optimizer payload, or serialized loader state.
            diagnostics = {'bytes': len(left), 'different_bytes': int(len(differing)),
                           'first_differing_offsets': differing[:30].tolist(), 'last_differing_offsets': differing[-30:].tolist(),
                           'header_identical': left[:1024] == right[:1024]}
            (RESULTS/f'{prefix}-state-difference-{rank}.json').write_text(json.dumps(diagnostics, indent=2))
            assert False, f'Optimizer/data/RNG state mismatch for rank {rank}: {diagnostics}'
    return detail


def parity_case(me, precision='BF16', cudnn=False):
    suffix = f'{precision.lower()}-{"cudnn" if cudnn else "cuda"}-single'
    # The corrected backward intentionally differs from the original for muP
    # multipliers other than one. Preserve original-branch parity as a control
    # at the reference multiplier; non-reference gradients have an independent
    # before/fixed finite-difference regression in the mup_gradient suite.
    flags = ['-e', 'd6', '-me', str(me), '-mm', '1' if me else '2', '-ma', '8', '-r', '0', '-ge', '0']
    reference = BASE
    if me:
        # The promoted embedding default intentionally differs from the original.
        # Match only that initializer in an isolated reference; its optimizer and
        # kernels remain original, and multiplier one removes the gradient fix.
        reference = args.work / 'baseline-random-embedding'
        if not reference.exists():
            shutil.copytree(BASE, reference, symlinks=True)
            source = (reference/'train_gpt2.cu').read_text()
            pattern = r'                if \(model->use_mup && i == 0\) \{  // set readout/embedding to all 0\.\n.*?\n                \}\n'
            source, count = re.subn(pattern, '', source, flags=re.DOTALL)
            assert count == 1
            (reference/'train_gpt2.cu').write_text(source)
        if not (reference/('init-matched-'+suffix)).exists():
            build(reference, precision=precision, cudnn=cudnn)
            (reference/('init-matched-'+suffix)).touch()
    a = train(f'parity-{me}-{suffix}', flags, exe=ROOT/('train_gpt2cu-'+suffix), old_paths=True)
    b = train(f'parity-{me}-{suffix}', flags, exe=reference/('train_gpt2cu-'+suffix), root=reference, old_paths=True)
    return compare_models(a, b)


def coordinates():
    result = []
    for me in [0, 1]:
        for width in [64, 128, 256, 512, 1024, 2048]:
            train(f'coord-{me}-{width}', ['-mc', '1', '-mw', str(width), '-me', str(me), '-mm', str(width/256), '-ma', '11.313', '-n', '0'])
            data = np.fromfile(ROOT/f'usemup={me}_width={width}_coord_check_data.bin', dtype=np.float32)
            assert data.size == 76 and np.isfinite(data).all()
            result.append({'mup': me, 'width': width, 'max_activation': float(data.max())})
    return result


def schedules():
    for name in ['constant', 'linear', 'cosine', 'wsd']:
        out = train('schedule-'+name, ARCH + ['-k', name, '-l', '0.001', '-x', '10', '-ss', '2', '-es', '6', '-u', '2', '-q', '0.1'])
        lines = [x for x in (out/'main.log').read_text().splitlines() if 'trl:' in x]
        assert len(lines) == 4
        for step, line in zip(range(2, 6), lines):
            assert line.startswith(f's:{step} ')
            lr = float(re.search(r'lr:([\d.e+-]+)', line)[1])
            ratio = (step-2)/8
            expected = {'constant': .001, 'linear': .001-.0009*ratio, 'cosine': .0001+.0009*.5*(1+math.cos(math.pi*ratio)), 'wsd': .001}[name]
            assert abs(lr-expected) <= 0.0000006, (name, step, lr, expected)
        assert (out/'DONE_00000006').exists()
        assert 'time (ms):' in lines[0]


def cli():
    text = run(['./train_gpt2cu', '--help'])
    for flag in ['-1d', '-1c', '-1h', '-ss', '-es', '-id', '-iv', '-it', '-ih', '-me', '-mm', '-ma', '-mc', '-mw', '-mb']:
        assert flag in text
    for flags in [['-mx', '1'], ['-me'], ['-mm', 'nan'], ['-1c', '127', '-1h', '2'], ['-me', '1', '-1c', '128', '-1h', '2', '-mb', '64', '-mm', '1']]:
        run(['./train_gpt2cu', *flags], expected=1)


def checkpoint_checks():
    out = ROOT/'runs/resume-1-0-split'
    for flags in [['-me', '0'], ['-mm', '1'], ['-ma', '1'], ['-1c', '256']]:
        run(['./train_gpt2cu', '-y', '1', '-o', str(out), '-x', '8', *flags], expected=1)
    retained = train('retention', ARCH + ['-n', '2', '-nk', '1', '-nm', '4'])
    assert sorted(p.name for p in retained.glob('DONE_*')) == ['DONE_00000004', 'DONE_00000006']


def main():
    global CAP
    info = run(['nvidia-smi'])
    cap = run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader']).splitlines()[0].strip().replace('.', '')
    CAP = cap
    (RESULTS/'environment.txt').write_text(info + run(['nvcc', '--version']))
    if not test('download upstream test fixtures', fixtures):
        return
    if args.suite == 'mup_fix_training':
        if test('build corrected BF16 trainer', build):
            test('corrected muP multiplier=2 exact checkpoint resume', lambda: resume_case(1))
            if test('build original BF16 trainer', lambda: build(BASE)):
                test('SP training payload parity with original', lambda: parity_case(0))
                test('muP multiplier=1 training parity with initializer-matched original', lambda: parity_case(1))
        return
    if args.suite == 'cudnn_diagnostic':
        if test('build cuDNN diagnostic trainer', lambda: build(cudnn=True)):
            for me in [0, 1]:
                test(f'cuDNN detailed state comparison mode={me}', lambda me=me: resume_case(me, suffix='-cudnn-diagnostic'))
        if test('build original cuDNN reference', lambda: build(BASE, cudnn=True)):
            test('original cuDNN reference', lambda: run(['./test_gpt2cu'], cwd=BASE, timeout=240)[-1800:])
        return
    if args.suite == 'focused':
        for precision in ['BF16', 'FP32']:
            if test(f'build {precision} reference', lambda precision=precision: build(precision=precision)):
                test(f'{precision} reference without GELU fusion', lambda: run(['./test_gpt2cu', '-ge', '0', '-r', '0'], timeout=240)[-1800:])
        if test('build BF16 cuDNN trainer and reference', lambda: build(cudnn=True)):
            for me in [0, 1]:
                test(f'cuDNN architecture/resume mode={me}', lambda me=me: resume_case(me, suffix='-cudnn'))
            test('cuDNN reference', lambda: run(['./test_gpt2cu'], timeout=240)[-1800:])
        return
    if args.suite == 'baseline':
        if test('build original BF16 trainer and reference test', lambda: build(BASE)):
            for flags in [[], ['-r', '0'], ['-r', '2'], ['-w', '0']]:
                test(f'original branch BF16 reference {flags}', lambda flags=flags: run(['./test_gpt2cu', *flags], cwd=BASE, timeout=240)[-1800:])
        if test('build original FP32 trainer and reference test', lambda: build(BASE, precision='FP32')):
            test('original branch FP32 reference', lambda: run(['./test_gpt2cu'], cwd=BASE, timeout=240)[-1800:])
        return
    if args.suite == 'multi':
        if not test('build BF16 MPI/NCCL trainer', lambda: build(multi=True)):
            return
        for zero in [0, 1]:
            for me in [0, 1]:
                test(f'two GPUs SP/muP={me} ZeRO={zero} exact resume', lambda me=me, zero=zero: resume_case(me, multi=True, zero=zero))
        return
    if not test('build BF16 CUDA trainer and existing test', build):
        return
    test('CLI help and invalid arguments', cli)
    test('custom SP architecture', lambda: str(train('custom-sp', ARCH)))
    test('custom muP architecture', lambda: str(train('custom-mup', ARCH + ['-me', '1', '-mb', '64', '-ma', '8'])))
    for me in [0, 1]:
        test(f'default descriptor SP/muP={me}', lambda me=me: str(train(f'default-{me}', ['-e', 'd12', '-me', str(me), '-x', '2'])))
        test(f'exact checkpoint resume SP/muP={me}', lambda me=me: resume_case(me))
    test('four LR schedules and absolute start/early stop', schedules)
    test('checkpoint conflicts and retention', checkpoint_checks)
    test('coordinate checks at six widths for SP and muP', coordinates)
    for flags in [[], ['-r', '0'], ['-r', '2'], ['-w', '0']]:
        test(f'existing BF16 test_gpt2cu {flags}', lambda flags=flags: run(['./test_gpt2cu', *flags], timeout=240)[-1800:])
    if test('build original mup BF16 baseline', lambda: build(BASE)):
        for me in [0, 1]:
            test(f'BF16 original training parity (initializer matched for muP), mode={me}, multiplier={1 if me else 2}', lambda me=me: parity_case(me))
    test('existing device/file I/O test', lambda: (run(['nvcc', '-O1', '-o', '/tmp/test_device_file_io', 'dev/test/device_file_io.cu']), run(['/tmp/test_device_file_io']))[-1][-1000:])
    if test('build FP32 CUDA trainer and existing test', lambda: build(precision='FP32')):
        test('existing FP32 test_gpt2cu', lambda: run(['./test_gpt2cu'], timeout=240)[-1800:])
        test('FP32 custom muP smoke', lambda: str(train('fp32-mup', ARCH + ['-me', '1', '-mb', '64'])))
        if test('build original mup FP32 baseline', lambda: build(BASE, precision='FP32')):
            test('FP32 SP training parity', lambda: parity_case(0, precision='FP32'))
    if test('build BF16 cuDNN trainer and existing test', lambda: build(cudnn=True)):
        for me in [0, 1]:
            test(f'cuDNN custom architecture/resume mode={me}', lambda me=me: resume_case(me, suffix='-cudnn'))
        test('existing cuDNN test_gpt2cu', lambda: run(['./test_gpt2cu'], timeout=240)[-1800:])


try:
    main()
except Exception:
    test('runner exception', lambda: (_ for _ in ()).throw(RuntimeError(traceback.format_exc())))
finally:
    (RESULTS/'summary.json').write_text(json.dumps({'suite': args.suite, 'tests': records}, indent=2))
    print(json.dumps({'suite': args.suite, 'passed': sum(r['status']=='passed' for r in records), 'failed': sum(r['status']=='failed' for r in records)}), flush=True)
raise SystemExit(any(r['status'] == 'failed' for r in records))
