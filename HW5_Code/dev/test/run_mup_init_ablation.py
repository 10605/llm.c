"""Three-seed 2x2 initialization ablation at reference width 192 only.

An isolated trainer copy gets diagnostic initialization switches. Production
source and historical results are untouched. Every trial consumes the same
2.048M FineWeb tokens as the preceding constant-LR, zero-decay experiment.
"""
import argparse
import difflib
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from prepare_lr_transfer import make_plan
from run_fineweb_sweep import data_files, read_metrics, NumericalDivergence, HOURLY_RATE

ARMS=[('zeroE-zeroQ',False,False),('randomE-zeroQ',True,False),('zeroE-randomQ',False,True),('randomE-randomQ',True,True)]

def patched_source(source):
    # Support both historical zero-WTE source and the promoted random-WTE default.
    if 'if (model->use_mup && i == 0) {  // set readout/embedding to all 0.' not in source:
        marker='                // Keep the tied token embedding/readout random with std 0.02.'
        assert source.count(marker)==1
        legacy='''                if (model->use_mup && i == 0) {  // set readout/embedding to all 0.
                    for (size_t j = 0; j < n; j++) {
                        params_memory_cpu[offset + layer_offset + j] = 0.f;
                    }
                }
'''
        source=source.replace(marker,legacy+marker)
    changes=[('manual_seed(&init_rng, 42);','manual_seed(&init_rng, getenv("LLMC_DIAG_SEED") ? atoi(getenv("LLMC_DIAG_SEED")) : 42);'),
             ('if (model->use_mup && i == 4) {  // qkv weights, init q with 0.0',
              'if (model->use_mup && i == 4 && !getenv("LLMC_DIAG_KEEP_QUERY")) {  // diagnostic query choice'),
             ('if (model->use_mup && i == 0) {  // set readout/embedding to all 0.',
              'if (model->use_mup && i == 0 && !getenv("LLMC_DIAG_KEEP_EMBEDDING")) {  // diagnostic tied embedding choice'),
             ('    // copy them to GPU\n    cudaCheck(cudaMemcpy(model->params_memory, params_memory_cpu,',
              '    if (const char* dump = getenv("LLMC_DIAG_INIT_DUMP")) {\n'
              '        FILE* f = fopenCheck(dump, "wb");\n'
              '        fwriteCheck(params_memory_cpu, 1, model->num_parameters_bytes, f);\n'
              '        fcloseCheck(f);\n    }\n'
              '    // copy them to GPU\n    cudaCheck(cudaMemcpy(model->params_memory, params_memory_cpu,')]
    for a,b in changes:
        assert source.count(a)==1,a
        source=source.replace(a,b)
    return source

def main():
    parser=argparse.ArgumentParser()
    for name in ['root','results','cache']: parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--worker',type=int,required=True); parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--target-plan',type=Path)
    args=parser.parse_args(); out=args.results; out.mkdir(parents=True,exist_ok=True); started=time.monotonic()
    work=args.root.parent/'init-ablation'; shutil.copytree(args.root,work,dirs_exist_ok=True)
    original=(work/'train_gpt2.cu').read_text(); modified=patched_source(original)
    (work/'train_gpt2.cu').write_text(modified)
    (out/'diagnostic.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),modified.splitlines(True),fromfile='production/train_gpt2.cu',tofile='diagnostic/train_gpt2.cu')))
    (out/'source.json').write_text(json.dumps({'production_sha256':hashlib.sha256(original.encode()).hexdigest(),'diagnostic_sha256':hashlib.sha256(modified.encode()).hexdigest()},indent=2))
    cap=subprocess.check_output(['nvidia-smi','--query-gpu=compute_cap','--format=csv,noheader'],text=True).splitlines()[0].strip().replace('.','')
    cmd=['make','train_gpt2cu','PRECISION=BF16',f'GPU_COMPUTE_CAPABILITY={cap}','FORCE_NVCC_O=3','USE_CUDNN=0','NO_MULTI_GPU=1','NO_USE_MPI=1']
    with (out/'build.log').open('w') as f: subprocess.run(cmd,cwd=work,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=240)
    (out/'environment.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    data,tokenizer,provenance=data_files(args.cache)
    (out/'dataset.json').write_text(json.dumps(provenance,indent=2))
    plan=make_plan(str(data/'fineweb_train_000001.bin'),str(data/'fineweb_val_000000.bin'),str(tokenizer),min_lr_exponent=-13,max_lr_exponent=-8)
    trials=[]; width=192
    if args.target_plan:
        selection=json.loads(args.target_plan.read_text()); width=768
        arm,keep_e,keep_q=next(a for a in ARMS if a[0]==selection['candidate_arm'])
        exponent=selection['selected_exponent']
        plan=make_plan(str(data/'fineweb_train_000001.bin'),str(data/'fineweb_val_000000.bin'),str(tokenizer),min_lr_exponent=min(exponent-1,-12),max_lr_exponent=max(exponent+1,-6))
        for seed in [42,43,44]:
            for trial in plan['trials']:
                if trial['width']!=768: continue
                if trial['mode']=='mup':
                    if round(math.log2(trial['base_learning_rate'])) in [exponent-1,exponent,exponent+1]:
                        trials.append(dict(trial,seed=seed,arm=arm,keep_embedding=keep_e,keep_query=keep_q))
                    if trial['base_learning_rate']==2**-9:
                        trials.append(dict(trial,seed=seed,arm='zeroE-zeroQ',keep_embedding=False,keep_query=False))
                elif trial['base_learning_rate']==2**-12:
                    trials.append(dict(trial,seed=seed,arm='SP-control',keep_embedding=True,keep_query=True))
        assert len(trials)==15
        for seed in [42,43,44]:
            for trial in plan['trials']:
                if trial['width']==192 and trial['mode']=='mup' and trial['base_learning_rate'] in {2**-7,2**-6}:
                    trials.append(dict(trial,seed=seed,arm='zeroE-zeroQ',keep_embedding=False,keep_query=False))
        assert len(trials)==21
        if selection.get('query_controls'):
            trials=[]
            for seed in [42,43,44]:
                for name,query_exponent in selection['query_controls'].items():
                    for trial in plan['trials']:
                        if trial['width']==768 and trial['mode']=='mup' and trial['base_learning_rate']==2**query_exponent:
                            trials.append(dict(trial,seed=seed,arm=name,keep_embedding=name.startswith('randomE'),keep_query=True))
            assert len(trials)==6
    else:
        for seed in [42,43,44]:
            for arm,keep_e,keep_q in ARMS:
                for trial in plan['trials']:
                    if trial['width']==192 and trial['mode']=='mup':
                        trials.append(dict(trial,seed=seed,arm=arm,keep_embedding=keep_e,keep_query=keep_q))
            for trial in plan['trials']:
                if trial['width']==192 and trial['mode']=='sp' and trial['base_learning_rate'] in {2**-11,2**-9}:
                    trials.append(dict(trial,seed=seed,arm='SP-control',keep_embedding=True,keep_query=True))
        assert len(trials)==78
    selected=trials[args.worker::args.workers]
    (out/'plan.json').write_text(json.dumps({'total_trials':len(trials),'seeds':[42,43,44],'width':width,'steps':500,'tokens_per_trial':2048000,'worker':args.worker,'trials':selected},indent=2))
    records=[]
    for trial in selected:
        name=f"c{trial['width']}-s{trial['seed']}-{trial['arm']}-lr{round(math.log2(trial['base_learning_rate']))}"
        print('TRIAL',name,flush=True); before=time.monotonic()
        cmd=trial['command'].copy(); trial_out=out/name; cmd[cmd.index('-o')+1]=str(trial_out); cmd+=['-sl','0','-sg','0']
        env=os.environ.copy()
        for key in ['LLMC_DIAG_KEEP_EMBEDDING','LLMC_DIAG_KEEP_QUERY']: env.pop(key,None)
        env['LLMC_DIAG_SEED']=str(trial['seed']); dump=work/'initialization.bin'; env['LLMC_DIAG_INIT_DUMP']=str(dump)
        if trial['keep_embedding']: env['LLMC_DIAG_KEEP_EMBEDDING']='1'
        if trial['keep_query']: env['LLMC_DIAG_KEEP_QUERY']='1'
        with (out/(name+'.log')).open('w') as f:
            f.write(json.dumps({'command':cmd,'initialization':{k:v for k,v in env.items() if k.startswith('LLMC_DIAG_')}})+'\n'); f.flush()
            proc=subprocess.run(cmd,cwd=work,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=120)
        record={k:v for k,v in trial.items() if k not in {'command','shell_command','name'}}
        record.update(name=name,command=cmd,exit_code=proc.returncode,seconds=time.monotonic()-before)
        try:
            if proc.returncode: raise RuntimeError('Trainer failed')
            record.update(read_metrics(trial_out,500)); record['status']='completed'
        except NumericalDivergence as exc: record.update(status='diverged',error=str(exc))
        except Exception as exc: record.update(status='failed',error=str(exc))
        if dump.exists():
            raw=dump.read_bytes(); record['initialization_sha256']=hashlib.sha256(raw).hexdigest()
            bits=np.frombuffer(raw,dtype=np.uint16); c=trial['width']; l=12
            e=bits[:50257*c]; q_offset=50304*c+1024*c+2*l*c
            q=bits[q_offset:q_offset+l*3*c*c].reshape(l,3*c,c)[:,:c,:]
            record['embedding_all_zero']=bool(np.all((e&0x7fff)==0)); record['queries_all_zero']=bool(np.all((q&0x7fff)==0))
            assert record['embedding_all_zero']==(not trial['keep_embedding'])
            assert record['queries_all_zero']==(not trial['keep_query'])
            dump.unlink()
        else: raise RuntimeError('Missing initialization audit')
        records.append(record)
        (out/'summary.json').write_text(json.dumps({'status':'running','records':records,'seconds':time.monotonic()-started},indent=2,allow_nan=False))
        print(name,record['status'],record.get('final_validation_loss'),flush=True)
    seconds=time.monotonic()-started
    status='completed' if len(records)==len(selected) and all(r['status']!='failed' for r in records) else 'failed'
    (out/'summary.json').write_text(json.dumps({'status':status,'records':records,'seconds':seconds,'estimated_compute_cost_usd':seconds/3600*HOURLY_RATE},indent=2,allow_nan=False))
    return int(status!='completed')

if __name__=='__main__': raise SystemExit(main())
