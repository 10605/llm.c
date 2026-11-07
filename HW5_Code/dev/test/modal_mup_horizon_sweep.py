"""Five bounded A100 workers for the authorized muP-only horizon comparison."""
import datetime
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import traceback

import modal
from modal_gpu_tests import ROOT, image, volume, package_sources, unpack

app = modal.App('llmc-mup-horizon-transfer')
image = image.add_local_python_source('modal_gpu_tests', copy=True)


def archive_results(dest):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode='w:gz') as archive:
        for path in dest.rglob('*'):
            if path.is_file() and path.suffix in {'.json','.txt','.log','.cpp'}:
                archive.add(path, arcname=str(path.relative_to(dest)))
    return output.getvalue()


@app.function(image=image, gpu='A100-40GB', cpu=8, memory=32768, timeout=1500,
              retries=0, max_containers=5, scaledown_window=2, volumes={'/results':volume})
def worker(source: bytes, run_id: str, index: int, plan: dict):
    volume.reload()
    root = Path('/tmp/llmc-tests/integration')
    unpack(source, root)
    dest = Path('/results')/run_id/f'worker{index}'
    dest.mkdir(parents=True, exist_ok=True)
    (dest/'plan.json').write_text(json.dumps(plan, indent=2))
    code = 1
    try:
        code = subprocess.run(['python', str(root/'dev/test/run_mup_horizon_sweep.py'),
            '--root',str(root),'--results',str(dest),'--plan',str(dest/'plan.json')], timeout=1460).returncode
    except Exception:
        (dest/'runner_error.txt').write_text(traceback.format_exc())
    finally:
        volume.commit()
    return code, archive_results(dest)


def durable_calls(source, run_id, dest, plans):
    spec = dict(source_sha256=hashlib.sha256(source).hexdigest(), plans=plans)
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    journal = dest/'worker-calls.json'
    if journal.exists():
        saved = json.loads(journal.read_text())
        if saved['spec_sha256'] != digest or len(saved['call_ids']) != 5 or len(set(saved['call_ids'])) != 5:
            raise ValueError('Saved workers do not match this run')
        return [modal.FunctionCall.from_id(i) for i in saved['call_ids']]
    calls = [worker.spawn(source,run_id,i,p) for i,p in enumerate(plans)]
    pending = journal.with_suffix('.pending')
    pending.write_text(json.dumps(dict(spec_sha256=digest,call_ids=[c.object_id for c in calls]),indent=2))
    pending.replace(journal)
    volume.commit()
    return calls


@app.function(image=image, cpu=0.25, memory=1024, timeout=2100, retries=1,
              scaledown_window=2, volumes={'/results':volume})
def coordinate(source: bytes, run_id: str, manifest: dict, prior_run: str = ''):
    root = Path('/tmp/horizon-coordinator')
    unpack(source, root)
    sys.path.insert(0,str(root/'dev/test'))
    from run_mup_horizon_sweep import make_trials, make_refinement_trials, partition, audit_data_order, RATE, REVISION
    from run_fineweb_sweep import data_files
    dest = Path('/results')/run_id
    dest.mkdir(parents=True,exist_ok=True)
    (dest/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
    cache = Path('/results/cache'); cache.mkdir(exist_ok=True)
    code = 1
    try:
        data, tokenizer, provenance = data_files(cache)
        assert provenance['revision'] == REVISION
        assert provenance['files']['fineweb_train_000001.bin']['tokens'] > 32768000+1
        audit_data_order(root, data/'fineweb_train_000001.bin', dest)
        original_trials = make_trials(data/'fineweb_train_000001.bin',data/'fineweb_val_000000.bin',tokenizer)
        prior_rows, prior_seconds, prior_cost = [], 0, 0
        if prior_run:
            assert Path(prior_run).name==prior_run
            previous = Path('/results')/prior_run
            old_manifest = json.loads((previous/'source_manifest.json').read_text())['files']
            core = [p for p in old_manifest if p.startswith('llmc/') or p in {'train_gpt2.cu','Makefile'}]
            assert all(manifest['files'][p]==old_manifest[p] for p in core), 'Training source changed'
            old_plan = json.loads((previous/'plan.json').read_text())
            old_summary = json.loads((previous/'summary.json').read_text())
            assert old_plan['dataset']==provenance and old_summary['status']=='completed'
            assert old_plan['trials']==original_trials, 'Prior recipe or grid differs'
            assert len(old_summary['records'])==81
            prior_rows = old_summary['records']
            prior_seconds = old_summary['worker_seconds']
            prior_cost = old_summary['estimated_compute_cost_usd']
            # Keep the complete old evidence separate; no original results are overwritten.
            unpack(archive_results(previous), dest/'coarse')
            trials = make_refinement_trials(original_trials)
        else:
            trials = original_trials
        groups, loads = partition(trials)
        plans = [dict(dataset=provenance,trials=group,max_seconds=1400 if prior_run else 1120) for group in groups]
        plan = dict(trials=original_trials+trials if prior_run else trials,dataset=provenance,workers=5,
            prior_run=prior_run,prior_trial_count=len(prior_rows),new_trial_count=len(trials),
            expected_worker_training_seconds=loads,
            lr_grid_note='1.19× LR spacing near minima' if prior_run else '2× LR grid',
            dense_lr_interval_log2=[-10,-6] if prior_run else None,
            dense_lr_step_log2=.25 if prior_run else 1,
            max_requested_gpu_worker_compute_usd=5*1500/3600*RATE,
            recipe='muP only; cosine; 10% warmup; final LR 0; weight decay .1; 12 layers/heads; seed42; B16 T256 global4096',
            validation_tokens=1048576,primary_metric='final held-out loss at each prescribed horizon',
            interpretation='Horizon-specific LR optimum for this schedule/decay recipe; one seed and discrete LR grid, not a continuous global optimum.')
        (dest/'plan.json').write_text(json.dumps(plan,indent=2))
        volume.commit()
        calls = durable_calls(source,run_id,dest,plans)
        rows, seconds, code = list(prior_rows), 0, 0
        for i, call in enumerate(calls):
            child_code, output = call.get()
            child = dest/f'worker{i}'
            unpack(output,child)
            code |= child_code
            summary = json.loads((child/'summary.json').read_text())
            rows.extend(summary['records']); seconds += summary['worker_seconds']
        assert len(rows)==len(plan['trials']) and len({r['name'] for r in rows})==len(plan['trials'])
        summary = dict(status='completed' if code==0 else 'failed',records=rows,
            worker_seconds=seconds,estimated_compute_cost_usd=seconds/3600*RATE,
            prior_worker_seconds=prior_seconds,prior_compute_cost_usd=prior_cost,
            cumulative_compute_cost_usd=prior_cost+seconds/3600*RATE,
            prior_trial_count=len(prior_rows),new_trial_count=len(trials))
        (dest/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    except Exception:
        code=1
        (dest/'coordinator_error.txt').write_text(traceback.format_exc())
    finally:
        volume.commit()
    return code, archive_results(dest)


@app.local_entrypoint()
def main(refine_from: str = ''):
    if refine_from and Path(refine_from).name!=refine_from:
        raise ValueError('Use a prior run name, not a path')
    source,_,files=package_sources()
    run_id=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+('-mup-horizons-refined' if refine_from else '-mup-horizons')
    dest=ROOT.parent/'modal-results'/run_id; dest.mkdir(parents=True)
    manifest=dict(files=files,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    (dest/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Results:',dest,flush=True)
    call=coordinate.spawn(source,run_id,manifest,refine_from)
    (dest/'execution.json').write_text(json.dumps(dict(coordinator_call_id=call.object_id,
        gpu_workers=5,timeout_seconds_per_gpu_worker=1500,max_requested_gpu_worker_compute_usd=5.6913,
        prior_run=refine_from),indent=2))
    code,output=call.get(); unpack(output,dest)
    print('Finished:',dest,flush=True)
    if code:
        raise RuntimeError('Horizon comparison incomplete or failed; inspect retained results')
