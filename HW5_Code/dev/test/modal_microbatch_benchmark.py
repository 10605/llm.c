"""Two bounded GPU workers comparing microbatches at fixed optimizer batch."""
import datetime
import io
import json
from pathlib import Path
import subprocess
import tarfile
import traceback

import modal
from modal_gpu_tests import ROOT, image, volume, package_sources, unpack

app = modal.App('llmc-mup-microbatch-benchmark')
image = image.add_local_python_source('modal_gpu_tests', copy=True)


def execute(source, run_id, gpu, profile='microbatch'):
    root = Path('/tmp/llmc-tests/integration')
    unpack(source, root)
    dest = Path('/results')/run_id/gpu
    dest.mkdir(parents=True, exist_ok=True)
    code = 1
    try:
        code = subprocess.run(['python', str(root/'dev/test/run_microbatch_benchmark.py'),
            '--root', str(root), '--results', str(dest), '--cache', '/results/cache',
            '--gpu', gpu, '--profile', profile], timeout=840).returncode
    except Exception:
        (dest/'runner_error.txt').write_text(traceback.format_exc())
    finally:
        volume.commit()
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode='w:gz') as archive:
        for path in dest.rglob('*'):
            if path.is_file() and path.suffix in {'.json', '.txt', '.log'}:
                archive.add(path, arcname=str(path.relative_to(dest)))
    return code, output.getvalue()


@app.function(image=image, gpu='A10', cpu=8, memory=32768, timeout=900,
              retries=0, max_containers=1, scaledown_window=2, volumes={'/results': volume})
def a10(source: bytes, run_id: str):
    return execute(source, run_id, 'A10')


@app.function(image=image, gpu='A100-40GB', cpu=8, memory=32768, timeout=900,
              retries=0, max_containers=1, scaledown_window=2, volumes={'/results': volume})
def a100(source: bytes, run_id: str, profile: str = 'microbatch'):
    return execute(source, run_id, 'A100-40GB', profile)


@app.local_entrypoint()
def main(profile: str = 'microbatch'):
    if profile not in {'microbatch', 'cosine'}:
        raise ValueError('Unknown benchmark profile')
    source, _, files = package_sources()
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-microbatch-benchmark'
    dest = ROOT.parent/'modal-results'/run_id
    dest.mkdir(parents=True)
    (dest/'source_manifest.json').write_text(json.dumps(dict(files=files,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()), indent=2))
    print('Results:', dest, flush=True)
    calls = ([('A100-40GB', a100.spawn(source, run_id, profile))] if profile == 'cosine'
             else [('A10', a10.spawn(source, run_id)), ('A100-40GB', a100.spawn(source, run_id))])
    (dest/'execution.json').write_text(json.dumps(dict(run_id=run_id,
        calls={gpu: call.object_id for gpu, call in calls}, timeout_per_gpu_seconds=900,
        profile=profile,
        max_requested_worker_compute_usd=900/3600*(2.731824 if profile == 'cosine' else 1.734624+2.731824),
        scope=('10 short SP/muP cosine throughput cases' if profile == 'cosine' else '24 short throughput cases')
              + '; no full 20-TPP training'), indent=2))
    failed = False
    for gpu, call in calls:
        code, output = call.get()
        unpack(output, dest/gpu)
        failed |= bool(code)
    print('Finished:', dest, flush=True)
    if failed:
        raise RuntimeError('One or more benchmarks failed; inspect retained results')
