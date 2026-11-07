"""Independent NumPy finite differences against actual CUDA full-model gradients.

Uses nonzero tied embeddings and queries so initialization cannot mask a broken
backward scale. The current trainer and an isolated one-line pre-fix version
receive identical weights and batches. No original sweep artifacts are changed.
"""
from pathlib import Path
import argparse
import json
import math
import os
import shutil
import struct
import subprocess
import time
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np

NAMES = ['wte', 'wpe', 'ln1w', 'ln1b', 'qkvw', 'qkvb', 'attprojw', 'attprojb',
         'ln2w', 'ln2b', 'fcw', 'fcb', 'fcprojw', 'fcprojb', 'lnfw', 'lnfb']


def shapes(c, layers=1, vocab=8192, length=64):
    return [(vocab,c),(length,c),(layers,c),(layers,c),(layers,3*c,c),(layers,3*c),
            (layers,c,c),(layers,c),(layers,c),(layers,c),(layers,4*c,c),(layers,4*c),
            (layers,c,4*c),(layers,c),(c,),(c,)]


def split(values, dims):
    result = {}; offset = 0
    for name, shape in zip(NAMES, dims):
        count = math.prod(shape)
        result[name] = values[offset:offset+count].reshape(shape)
        offset += count
    assert offset == len(values)
    return result


def reference(values, dims, inputs, targets, mup, multiplier):
    p = split(values, dims)
    def norm(x, weight, bias):
        centered = x - x.mean(-1, keepdims=True)
        return centered / np.sqrt((centered**2).mean(-1, keepdims=True) + 1e-5) * weight + bias
    def linear(x, weight, bias):
        return x @ weight.T + bias
    b, t = inputs.shape; c = dims[0][1]; heads = 12; d = c//heads
    x = p['wte'][inputs] + p['wpe'][np.arange(t)]
    for layer in range(dims[2][0]):
        qkv = linear(norm(x, p['ln1w'][layer], p['ln1b'][layer]), p['qkvw'][layer], p['qkvb'][layer])
        q, k, v = [a.reshape(b,t,heads,d).transpose(0,2,1,3) for a in np.split(qkv,3,axis=-1)]
        scores = q @ k.swapaxes(-1,-2) * (4/d if mup else 1/math.sqrt(d))
        scores = np.where(np.triu(np.ones((t,t), dtype=bool), 1), -np.inf, scores)
        attn = np.exp(scores - scores.max(-1, keepdims=True)); attn /= attn.sum(-1, keepdims=True)
        y = (attn @ v).transpose(0,2,1,3).reshape(b,t,c)
        x = x + linear(y, p['attprojw'][layer], p['attprojb'][layer])
        h = linear(norm(x, p['ln2w'][layer], p['ln2b'][layer]), p['fcw'][layer], p['fcb'][layer])
        h = .5*h*(1+np.tanh(math.sqrt(2/math.pi)*(h+.044715*h**3)))
        x = x + linear(h, p['fcprojw'][layer], p['fcprojb'][layer])
    x = norm(x, p['lnfw'], p['lnfb']) / (multiplier if mup else 1)
    logits = x @ p['wte'].T
    maximum = logits.max(-1, keepdims=True)
    logsumexp = maximum[...,0] + np.log(np.exp(logits-maximum).sum(-1))
    loss = np.mean(logsumexp - np.take_along_axis(logits,targets[...,None],axis=-1)[...,0])
    return float(loss), logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--suite', default='mup_gradient')
    args = parser.parse_args()
    root = args.work/'integration'; results = args.results
    results.mkdir(parents=True, exist_ok=True)
    started = time.monotonic(); commands = 0
    def run(command, cwd=root):
        nonlocal commands
        commands += 1
        before = time.monotonic()
        log = results/f'{commands:03d}.log'
        with log.open('w') as file:
            file.write(json.dumps({'command':list(map(str,command)), 'cwd':str(cwd)})+'\n')
            file.flush()
            proc = subprocess.run(list(map(str, command)), cwd=cwd, text=True, stdout=file,
                                  stderr=subprocess.STDOUT, timeout=300 if command[0]=='nvcc' else 60)
        stdout = log.read_text().split('\n',1)[1]
        (results/f'{commands:03d}.log').write_text(json.dumps({'command':list(map(str,command)),
            'cwd':str(cwd),'exit_code':proc.returncode,'seconds':time.monotonic()-before})+'\n'+stdout)
        if proc.returncode:
            raise RuntimeError(f'Command failed; see {commands:03d}.log: {stdout[-1000:]}')
        return stdout
    (results/'environment.txt').write_text(run(['nvidia-smi']) + run(['nvcc','--version']))
    cap = run(['nvidia-smi','--query-gpu=compute_cap','--format=csv,noheader']).splitlines()[0].replace('.','').strip()
    before_root = args.work/'before-gradient-fix'
    shutil.copytree(root, before_root, dirs_exist_ok=True)
    source = (root/'train_gpt2.cu').read_text()
    needle = 'model->use_mup, 1.0f / model->mup_width_mult, main_stream);'
    assert source.count(needle) == 1, 'Expected precisely one corrected final backward call'
    (before_root/'train_gpt2.cu').write_text(source.replace(needle, 'model->use_mup, model->mup_width_mult, main_stream);'))
    records = []
    for precision in ['FP32', 'BF16']:
        binaries = {}
        for variant, directory in [('before',before_root), ('fixed',root)]:
            binary = directory/f'mup-gradient-{precision.lower()}'
            run(['nvcc','-O1','--use_fast_math','-std=c++17',f'-DENABLE_{precision}',f'-arch=sm_{cap}',
                 'dev/test/test_mup_gradients.cu','-lcublas','-lcublasLt','-lnvidia-ml','-o',binary], cwd=directory)
            binaries[variant] = binary
        for mup, width in [(1,48),(1,96),(1,192),(1,384),(1,768),(0,192),(0,768)]:
            multiplier = width/192
            dims = shapes(width)
            rng = np.random.default_rng(1234 + width)
            chunks = []
            for name, shape in zip(NAMES,dims):
                if name in {'ln1w','ln2w','lnfw'}:
                    value = 1 + .05*rng.normal(size=shape)
                elif name.endswith('b'):
                    value = .01*rng.normal(size=shape)
                else:
                    value = .04*rng.normal(size=shape) / (math.sqrt(multiplier) if name in {'qkvw','attprojw','fcw','fcprojw'} else 1)
                chunks.append(value.astype(np.float32).ravel())
            params = np.concatenate(chunks)
            # 64 tokens satisfy the production GELU kernel's vectorized block
            # size at every tested width in both FP32 and BF16.
            inputs = np.tile(np.array([[3,5,3,7,11,5,13,17]],dtype=np.int32), (1,8))
            targets = np.tile(np.array([[5,3,7,11,5,13,17,19]],dtype=np.int32), (1,8))
            name = f'{precision.lower()}-mup{mup}-c{width}'
            fixture = args.work/f'{name}.bin'
            with fixture.open('wb') as file:
                # The inherited classifier assumes at least 1024*x128::size
                # vocabulary entries; smaller V underflows its unsigned tail loop.
                # GPT-2's real vocabulary meets this constraint. Keep the test
                # fixture valid in both precisions without changing that kernel.
                file.write(struct.pack('<8i2f',20261103,width,1,12,8192,1,64,mup,multiplier,4.0))
                inputs.tofile(file); targets.tofile(file); params.tofile(file)
            outputs = {}
            for variant, binary in binaries.items():
                prefix = args.work/f'{name}-{variant}'
                run([binary,fixture,prefix])
                outputs[variant] = {key:np.fromfile(str(prefix)+f'.{key}.bin', dtype=np.float32)
                                    for key in ['params','logits','grads']}
            assert np.array_equal(outputs['before']['params'], outputs['fixed']['params'])
            assert np.array_equal(outputs['before']['logits'], outputs['fixed']['logits'])
            actual_params = outputs['fixed']['params'].astype(np.float64)
            loss, logits = reference(actual_params,dims,inputs,targets,mup,multiplier)
            max_logit_error = float(np.max(abs(logits.ravel()-outputs['fixed']['logits'])))
            assert max_logit_error < (2e-4 if precision=='FP32' else .08), (name, max_logit_error)
            offset = 0; checks = []
            for tensor, shape in zip(NAMES,dims):
                count = math.prod(shape)
                # Highest-magnitude components give a meaningful signal at mixed precision.
                candidates = set()
                for variant in ['before','fixed']:
                    grad = outputs[variant]['grads'][offset:offset+count]
                    candidates.update((np.argsort(abs(grad))[-2:]+offset).tolist())
                for index in sorted(candidates):
                    original = actual_params[index]; epsilon = 1e-4
                    actual_params[index] = original + epsilon
                    plus = reference(actual_params,dims,inputs,targets,mup,multiplier)[0]
                    actual_params[index] = original - epsilon
                    minus = reference(actual_params,dims,inputs,targets,mup,multiplier)[0]
                    actual_params[index] = original
                    numerical = (plus-minus)/(2*epsilon)
                    check = {'tensor':tensor,'index':index,'finite_difference':numerical}
                    for variant in ['before','fixed']:
                        value = float(outputs[variant]['grads'][index])
                        tolerance = (2e-5 + .002*abs(numerical)) if precision=='FP32' else (1e-3 + .08*abs(numerical))
                        check[variant] = value
                        check[variant+'_passes'] = abs(value-numerical) <= tolerance
                    checks.append(check)
                offset += count
            row = {'name':name,'precision':precision,'width':width,'mup':bool(mup),'multiplier':multiplier,
                   'reference_loss':loss,'maximum_logit_error':max_logit_error,'checks':checks,
                   'fixed_passes':all(x['fixed_passes'] for x in checks),
                   'before_passes':all(x['before_passes'] for x in checks),
                   'before_fixed_gradients_identical':bool(np.array_equal(outputs['before']['grads'],outputs['fixed']['grads']))}
            records.append(row)
            (results/'gradient-checks.json').write_text(json.dumps(records,indent=2)+'\n')
            print(name, 'fixed:',row['fixed_passes'],'before:',row['before_passes'],flush=True)
            # Keep binary diagnostics out of persisted reports; exact parameters are reproducible from this fixture generator.
            for path in args.work.glob(name+'*.bin'):
                path.unlink()
    failed = [r['name'] for r in records if not r['fixed_passes'] or
              (r['mup'] and r['multiplier'] != 1 and r['before_passes']) or
              ((not r['mup'] or r['multiplier']==1) and not r['before_fixed_gradients_identical'])]
    result = {'status':'passed' if not failed else 'failed','suite':'mup_gradient','failed_cases':failed,
              'case_count':len(records),'seconds':time.monotonic()-started,
              'method':'Actual CUDA full-model gradients vs float64 independent NumPy central differences; nonzero tied embeddings/query weights; TF32 disabled.',
              'precision_tolerances':{'FP32':'2e-5 + 0.002*abs(reference)','BF16':'1e-3 + 0.08*abs(reference)'},
              'cases':[{k:v for k,v in r.items() if k!='checks'} for r in records]}
    (results/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    return int(bool(failed))


if __name__ == '__main__':
    raise SystemExit(main())
