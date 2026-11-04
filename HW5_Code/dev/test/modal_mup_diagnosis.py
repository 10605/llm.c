"""Bounded official-optimizer parity and reference-width initialization diagnosis."""
import datetime
import io
import json
from pathlib import Path
import subprocess
import tarfile
import traceback
import modal
from modal_gpu_tests import image, volume, package_sources, unpack, ROOT

app=modal.App('llmc-mup-loss-diagnosis')
image=image.add_local_python_source('modal_gpu_tests',copy=True)
parity_image=(image.pip_install('torch==2.5.1',index_url='https://download.pytorch.org/whl/cpu')
              .pip_install('pyyaml==6.0.2')
              .pip_install('mup @ git+https://github.com/microsoft/mup.git@19814971934ef91dd546f88e913fc963e096d11c',extra_options='--no-deps'))

def execute(source,run_id,phase,worker=0,selection=None):
    root=Path('/tmp/llmc-tests/integration'); unpack(source,root)
    dest=Path('/results')/run_id; dest.mkdir(parents=True,exist_ok=True)
    script='run_mup_optimizer_test.py' if phase=='parity' else 'run_mup_init_ablation.py'
    cmd=['python',str(root/'dev/test'/script),'--root',str(root),'--results',str(dest)]
    if phase!='parity': cmd+=['--cache','/results/cache','--worker',str(worker),'--workers','4']
    if selection:
        (dest/'target-selection.json').write_text(json.dumps(selection,indent=2))
        cmd+=['--target-plan',str(dest/'target-selection.json')]
    code=1
    try: code=subprocess.run(cmd,timeout=540 if phase=='parity' else 660).returncode
    except Exception: (dest/'runner_error.txt').write_text(traceback.format_exc())
    finally: volume.commit()
    output=io.BytesIO()
    with tarfile.open(fileobj=output,mode='w:gz') as archive:
        for p in dest.rglob('*'):
            if p.is_file() and p.suffix in {'.json','.log','.txt','.patch'}: archive.add(p,arcname=str(p.relative_to(dest)))
    return code,output.getvalue()

@app.function(image=parity_image,gpu='A10',cpu=8,memory=32768,timeout=600,scaledown_window=2,volumes={'/results':volume})
def parity(source:bytes,run_id:str): return execute(source,run_id,'parity')

@app.function(image=image,gpu='A10',cpu=8,memory=32768,timeout=720,max_containers=4,scaledown_window=2,volumes={'/results':volume})
def ablation(source:bytes,run_id:str,worker:int,selection:dict=None): return execute(source,run_id,'ablation',worker,selection)

@app.local_entrypoint()
def main(phase:str='parity',parity_run:str='',reference_run:str=''):
    if phase not in {'parity','ablation','target','queries'}: raise ValueError(phase)
    source,_,manifest=package_sources()
    selection=None
    if phase!='parity':
        previous=Path(parity_run)
        assert json.loads((previous/'summary.json').read_text())['status']=='passed','Require successful parity first'
        prior=json.loads((previous/'source_manifest.json').read_text())['files']
        assert all(manifest[p]==h for p,h in prior.items() if p=='train_gpt2.cu' or p.startswith('llmc/'))
    if phase in {'target','queries'}:
        analysis=json.loads((Path(reference_run)/'analysis.json').read_text()); assert analysis['status']=='passed'
        best=min(analysis['minima'],key=lambda x:x['mean'])
        embedding_only=next(x for x in analysis['minima'] if x['arm']=='randomE-zeroQ')
        if embedding_only['mean']<=best['mean']+.01:
            best=embedding_only
        assert best['arm']!='zeroE-zeroQ' and not best['boundary']
        assert best['mean_loss_change_from_current']<-.03,'Require a meaningful reference-width improvement'
        selection=dict(candidate_arm=best['arm'],selected_exponent=best['exponent'],reference_run=reference_run,
                       reference_mean_loss=best['mean'],reference_loss_change=best['mean_loss_change_from_current'],
                       selection_rule='Prefer changing only the tied embedding when its best mean loss is within 0.01 of the best arm; otherwise choose the best arm. Freeze its LR before target training.',
                       target_exponents=[best['exponent']-1,best['exponent'],best['exponent']+1],
                       reference_current_initialization_extension_exponents=[-7,-6],
                       current_mup_control_exponent=-9,sp_control_exponent=-12)
        if phase=='queries':
            selection['query_controls']={name:next(x['exponent'] for x in analysis['minima'] if x['arm']==name)
                                         for name in ['randomE-randomQ','zeroE-randomQ']}
    run_id=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-mup-'+phase
    dest=ROOT.parent/'modal-results'/run_id; dest.mkdir(parents=True)
    (dest/'source_manifest.json').write_text(json.dumps({'files':manifest},indent=2))
    (dest/'execution.json').write_text(json.dumps({'phase':phase,'parity_run':parity_run,'workers':1 if phase=='parity' else 4,'timeout_seconds_per_worker':600 if phase=='parity' else 720},indent=2))
    if selection: (dest/'selection_before_target.json').write_text(json.dumps(selection,indent=2))
    print('Results:',dest,flush=True)
    if phase=='parity':
        code,data=parity.remote(source,run_id); unpack(data,dest)
    else:
        calls=[ablation.spawn(source,f'{run_id}-worker{i}',i,selection) for i in range(4)]
        code=0; records=[]; seconds=0
        for i,call in enumerate(calls):
            child_code,data=call.get(); unpack(data,dest/f'worker{i}'); code|=child_code
            summary_path=dest/f'worker{i}'/'summary.json'
            if summary_path.exists():
                child=json.loads(summary_path.read_text()); records+=child['records']; seconds+=child['seconds']
            else:
                code=1
        (dest/'summary.json').write_text(json.dumps({'status':'completed' if code==0 else 'failed','records':records,'worker_seconds':seconds,'estimated_compute_cost_usd':seconds/3600*1.734624},indent=2))
    print('Finished:',dest,'exit:',code,flush=True)
    if code: raise RuntimeError('Diagnostic reported failures; inspect retained reports')
