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
parser.add_argument('--suite', choices=['single', 'multi', 'baseline', 'focused', 'cudnn_diagnostic', 'mup_fix_training', 'course_preset'], required=True)
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
    if root == ROOT:
        cmd += ['-cp', '0']  # historical integration/parity cases retain their original recipe
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


def course_run(name, extra=(), steps=10, tiny=True, fixed_decay=True):
    out=ROOT/'runs'/name
    out.parent.mkdir(exist_ok=True)
    cmd=['./train_gpt2cu','-l','.0006','-nk','0','-id',str(args.cache/'tiny_shakespeare_train.bin'),
         '-iv',str(args.cache/'tiny_shakespeare_val.bin'),'-it',str(args.cache/'gpt2_tokenizer.bin'),
         '-b','2','-t','64','-d','256','-v','5','-m','2','-s','0','-n','5','-o',str(out),'-f','0']
    if tiny: cmd+=['-1d','2','-1c','48']
    if fixed_decay: cmd+=['-c','.1']  # very short correctness cases use explicit decay
    if steps is not None: cmd+=['-x',str(steps)]
    cmd+=list(map(str,extra))
    output=run(cmd)
    assert not re.search(r'\bloss\s+(?:nan|inf)\b|\bnorm\s+(?:nan|inf)\b',output,re.I)
    params=dict(re.findall(r'\b(\w+)=([^\s]+)',(out/'run_parameters.txt').read_text()))
    return out,params


def course_widths():
    for width in [48,96,192,768]:
        out,p=course_run(f'course-width-{width}',['-1c',width],steps=1)
        assert (int(p['heads']),int(p['mup']),float(p['reference_width']),float(p['attention_mult']))==(12,1,192,4)
        assert float(p['width_mult'])==width/192
        assert p['scheduler']=='linear' and p['initialization_seed']=='42'
        assert p['max_seq_len']=='2048'
        assert math.isclose(float(p['weight_decay']),.1,rel_tol=1e-6) and float(p['final_fraction'])==0
    _,p=course_run('course-default-architecture',steps=1,tiny=False)
    assert (int(p['layers']),int(p['channels']),int(p['heads']))==(12,768,12)
    return 'Implicit muP settings at four widths; default 124M architecture trains.'


def course_equivalence():
    a,p=course_run('course-implicit')
    assert p['warmup']=='1' and abs(float(p['lr'])-.0006)<1e-9
    b,_=course_run('course-explicit',['-cp',0,'-me',1,'-1h',12,'-mb',192,'-ma',4,
        '-e','gpt3:c768','-k','linear','-l','.0006','-u',1,'-q',0,'-c','.1','-se',42])
    return compare_models(a,b,step=10)


def course_seed_and_resume():
    full,p=course_run('course-seed43',['-se',43])
    assert p['initialization_seed']=='43'
    seed42=ROOT/'runs/course-implicit/model_00000010.bin'
    assert payload_hash(full/'model_00000010.bin')!=payload_hash(seed42)
    split,_=course_run('course-seed43-split',['-se',43,'-es',5])
    # No architecture/parametrization flags on resume; -se must not replace weights.
    _,p=course_run('course-seed43-split',['-y',1,'-se',99],tiny=False)
    assert p['initialization_seed']=='checkpoint' and p['resume']=='1'
    compare_models(full,split,step=10)
    state='state_00000010_00000.bin'
    assert (full/state).read_bytes()==(split/state).read_bytes()
    return 'Distinct seed changes weights; model/optimizer/data/RNG resume is byte-identical.'


def course_checkpoint_defaults():
    for me in [0,1]:
        name=f'course-custom-checkpoint-{me}'
        full,_=course_run(name,['-me',me,'-1c',64,'-1h',2,'-mb',64,'-ma',8])
        _,p=course_run(name,['-y',1],tiny=False)
        assert (int(p['mup']),int(p['heads']),int(p['channels']))==(me,2,64)
        assert float(p['reference_width'])==64 and float(p['attention_mult'])==8
        assert p['initialization_seed']=='checkpoint'
    return 'SP and custom-reference muP checkpoints override implicit course defaults.'


def course_budgets():
    _,p=course_run('course-token-rounding',['-nt',2561],steps=None)
    assert (p['schedule_steps'],p['requested_tokens'],p['schedule_tokens'],p['warmup'])==('11','2561','2816','1')
    _,p=course_run('course-small-microbatch',['-b',1],steps=1)
    assert (p['total_batch'],p['accumulation'])==('256','4')
    _,p=course_run('course-early-stop',['-es',5],steps=100)
    assert (p['stop_step'],p['schedule_steps'],p['warmup'])==('5','100','10')
    _,p=course_run('course-constant',['-k','constant','-c',0],steps=10)
    assert p['warmup']=='0'
    return 'Whole-batch token rounding, fixed global batch, full-horizon warmup and constant override.'


def course_token_early_stop():
    # Stop inside the long warmup. Truncating the schedule would change weights.
    split,p=course_run('course-token-stop',['-nt',25600,'-et',1281],steps=None)
    assert (p['schedule_steps'],p['warmup'],p['stop_step'],p['stop_tokens'])==('100','10','6','1536')
    assert p['requested_early_tokens']=='1281'
    legacy,_=course_run('course-step-stop',['-es',6],steps=100)
    compare_models(split,legacy,step=6)
    state='state_00000006_00000.bin'
    assert (split/state).read_bytes()==(legacy/state).read_bytes()
    # A later token stop on resume is an absolute position, not extra work.
    _,p=course_run('course-token-stop',['-nt',25600,'-et',2304,'-y',1],steps=None,tiny=False)
    assert (p['start_step'],p['stop_step'],p['schedule_steps'],p['warmup'])==('6','9','100','10')
    full,_=course_run('course-token-uninterrupted',['-nt',25600,'-et',2304],steps=None)
    compare_models(split,full,step=9)
    state='state_00000009_00000.bin'
    assert (split/state).read_bytes()==(full/state).read_bytes()
    return 'Token/legacy early-stop weights and state match; token-based resume is exact with unchanged long schedule.'


def course_timescale():
    for index,(lr,batch) in enumerate([(.001,256),(.002,256),(.001,512)]):
        flags=['-nt',256000,'-et',batch*3,'-d',batch,'-l',lr]
        auto,p=course_run(f'course-tau-{index}',flags,steps=None,fixed_decay=False)
        assert p['decay_mode']=='timescale'
        tau=1/(float(p['lr'])*float(p['weight_decay'])*int(p['schedule_steps']))
        assert math.isclose(tau,.1,rel_tol=1e-6)
        manual,_=course_run(f'course-tau-explicit-{index}',flags+['-c',p['weight_decay']],steps=None)
        compare_models(auto,manual,step=3)
        state='state_00000003_00000.bin'
        assert (auto/state).read_bytes()==(manual/state).read_bytes()
    # Changing only the absolute stopping point on resume retains tau and state.
    flags=['-nt',256000,'-et',1280,'-d',256,'-l','.001']
    split,p=course_run('course-tau-0',flags+['-y',1],steps=None,tiny=False,fixed_decay=False)
    full,_=course_run('course-tau-uninterrupted',flags,steps=None,fixed_decay=False)
    compare_models(split,full,step=5)
    state='state_00000005_00000.bin'
    assert (split/state).read_bytes()==(full/state).read_bytes()
    return 'Automatic decay equals explicit decay at two LRs and batches; exact resumed optimizer state with fixed full-horizon tau.'


def course_context():
    _,p=course_run('course-context2048',['-b',4,'-t',2048,'-d',114688,
                                      '-nt',2505246720,'-et',114688],steps=None,fixed_decay=False)
    assert (p['sequence'],p['max_seq_len'],p['total_batch'],p['accumulation'])==('2048','2048','114688','14')
    assert p['stop_step']=='1' and p['decay_mode']=='timescale'
    return '2048-token context with 114688 global tokens trains using 14 accumulation passes and the full target decay horizon.'


def main():
    global CAP
    info = run(['nvidia-smi'])
    cap = run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader']).splitlines()[0].strip().replace('.', '')
    CAP = cap
    (RESULTS/'environment.txt').write_text(info + run(['nvcc', '--version']))
    if not test('download upstream test fixtures', fixtures):
        return
    if args.suite == 'course_preset':
        if test('build course BF16 trainer', build):
            test('course fixed heads and width scaling', course_widths)
            if test('implicit preset equals explicit legacy recipe', course_equivalence):
                test('initialization seed and exact checkpoint resumption', course_seed_and_resume)
            test('saved parametrization overrides course defaults', course_checkpoint_defaults)
            test('course token budgets, microbatch and warmup', course_budgets)
            test('token early stop preserves schedule and exact resumption', course_token_early_stop)
            test('AdamW timescale derivation and exact resumption', course_timescale)
            test('2048-token course context and target batch', course_context)
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
