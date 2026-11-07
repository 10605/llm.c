"""Compare actual CUDA derivatives and updates with official MuReadout/MuAdamW.

The reference transformer uses independent PyTorch primitives, not train_gpt2.py.
Forward/backward checks synchronize parameters before each of three steps.
Optimizer replay uses CUDA gradients to isolate update math from derivative noise.
An additional independent trajectory records, but does not gate on, drift caused
by Adam's sensitivity to nearly-zero gradients. Tolerances are declared below.
"""
import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import re
import struct
import subprocess
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from mup import MuReadout, MuAdamW, set_base_shapes
from run_mup_gradient_test import NAMES, shapes, split

PIN='19814971934ef91dd546f88e913fc963e096d11c'
TOLERANCES={'logits':(2e-4,1e-4),'gradients':(2e-5,.002),'updated':(2e-7,2e-5),
            'moment1':(2e-7,2e-5),'moment2':(2e-7,2e-5)}

class Reference(nn.Module):
    def __init__(self, values, dims, mup):
        super().__init__()
        self.p=nn.ParameterDict({name:nn.Parameter(torch.from_numpy(a.copy())) for name,a in split(values,dims).items()})
        self.head=MuReadout(dims[0][1],dims[0][0],bias=False)
        self.head.weight=self.p['wte']
        self.mup=mup; self.width=dims[0][1]; self.layers=dims[2][0]
        if mup:
            base=dict(zip(('p.'+n for n in NAMES),shapes(192,self.layers,dims[0][0],dims[1][0])))
            delta=dict(zip(('p.'+n for n in NAMES),shapes(384,self.layers,dims[0][0],dims[1][0])))
            set_base_shapes(self,base,delta=delta,rescale_params=False)
        else:
            set_base_shapes(self,None,rescale_params=False)
    def forward(self,ids):
        p=self.p; b,t=ids.shape; c=self.width; d=c//12
        norm=lambda x,w,bias:F.layer_norm(x,(c,),w,bias,eps=1e-5)
        x=F.embedding(ids,p['wte'])+p['wpe'][:t]
        for i in range(self.layers):
            qkv=F.linear(norm(x,p['ln1w'][i],p['ln1b'][i]),p['qkvw'][i],p['qkvb'][i])
            q,k,v=[a.reshape(b,t,12,d).transpose(1,2) for a in qkv.chunk(3,dim=-1)]
            scores=(q@k.transpose(-1,-2))*(4/d if self.mup else 1/math.sqrt(d))
            scores=scores.masked_fill(torch.ones(t,t,dtype=torch.bool).triu(1),float('-inf'))
            y=(scores.softmax(-1)@v).transpose(1,2).reshape(b,t,c)
            x=x+F.linear(y,p['attprojw'][i],p['attprojb'][i])
            h=F.linear(norm(x,p['ln2w'][i],p['ln2b'][i]),p['fcw'][i],p['fcb'][i])
            x=x+F.linear(F.gelu(h,approximate='tanh'),p['fcprojw'][i],p['fcprojb'][i])
        return self.head(norm(x,p['lnfw'],p['lnfb']))
    def load_flat(self,values,dims):
        with torch.no_grad():
            for n,a in split(values,dims).items(): self.p[n].copy_(torch.from_numpy(a))

def optimizer(model,wd):
    groups=[{'params':[model.p[n] for n in NAMES if n in {'wte','wpe','qkvw','attprojw','fcw','fcprojw'}],'weight_decay':wd},
            {'params':[model.p[n] for n in NAMES if n not in {'wte','wpe','qkvw','attprojw','fcw','fcprojw'}],'weight_decay':0}]
    return MuAdamW(groups,lr=2**-9,betas=(.9,.95),eps=1e-8,foreach=False)

def flat(model,kind='params',opt=None):
    if kind=='params': values=[model.p[n] for n in NAMES]
    elif kind=='grads': values=[model.p[n].grad for n in NAMES]
    else: values=[opt.state[model.p[n]][kind] for n in NAMES]
    return np.concatenate([a.detach().numpy().ravel() for a in values])

def error(actual,expected,atol,rtol):
    assert np.isfinite(actual).all() and np.isfinite(expected).all(),'Nonfinite comparison values'
    diff=abs(actual.astype(np.float64)-expected.astype(np.float64))
    return dict(max_absolute=float(diff.max()),relative_l2=float(np.linalg.norm(diff)/(np.linalg.norm(expected)+1e-30)),
                failures=int(np.count_nonzero(diff>atol+rtol*abs(expected))),count=int(diff.size))

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True); parser.add_argument('--results',type=Path,required=True)
    args=parser.parse_args(); out=args.results; out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic(); torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with (out/'python-initialization.log').open('w') as file:
        subprocess.run(['python',str(args.root/'dev/test/test_python_mup_initialization.py')],
                       stdout=file,stderr=subprocess.STDOUT,check=True,timeout=90)
    assert torch.__version__=='2.5.1+cpu',torch.__version__
    distribution=importlib.metadata.distribution('mup'); direct=json.loads(distribution.read_text('direct_url.json'))
    assert direct['vcs_info']['commit_id']==PIN,direct
    (out/'official.json').write_text(json.dumps({'mup':direct,'torch':torch.__version__,'tolerances':TOLERANCES},indent=2))
    cap=subprocess.check_output(['nvidia-smi','--query-gpu=compute_cap','--format=csv,noheader'],text=True).splitlines()[0].strip().replace('.','')
    binary=args.root/'mup-optimizer-parity'
    cmd=['nvcc','-O1','--use_fast_math','-std=c++17','-DENABLE_FP32',f'-arch=sm_{cap}',
         'dev/test/test_mup_optimizer.cu','-lcublas','-lcublasLt','-lnvidia-ml','-o',str(binary)]
    with (out/'build.log').open('w') as f: subprocess.run(cmd,cwd=args.root,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=240)
    records=[]
    cases=[(c,2,1,False,.1) for c in [48,96,192,384,768]]+[(192,2,0,False,.1),(192,12,1,False,0),(192,12,1,True,0)]
    for c,l,mup,zero,wd in cases:
        name=f'c{c}-l{l}-mup{mup}-zero{int(zero)}'; dims=shapes(c,l)
        rng=np.random.default_rng(1234+c); chunks=[]
        for n,s in zip(NAMES,dims):
            if n in {'ln1w','ln2w','lnfw'}: a=1+.05*rng.normal(size=s)
            elif n.endswith('b'): a=.01*rng.normal(size=s)
            else: a=.02*rng.normal(size=s)/(math.sqrt(c/192) if n in {'qkvw','attprojw','fcw','fcprojw'} else 1)
            if zero and (n=='wte' or n.endswith('b')): a=np.zeros(s)
            if zero and n=='qkvw': a[:,:c,:]=0
            chunks.append(a.astype(np.float32).ravel())
        values=np.concatenate(chunks)
        ids=np.tile(np.array([[3,5,3,7,11,5,13,17]],dtype=np.int32),(1,8)); targets=(ids+2)%8192
        fixture=args.root/(name+'.bin'); prefix=args.root/name
        with fixture.open('wb') as f:
            f.write(struct.pack('<8i2f',20261103,c,l,12,8192,1,64,mup,c/192,4.0)); ids.tofile(f); targets.tofile(f); values.tofile(f)
        log=out/(name+'.log')
        with log.open('w') as f: subprocess.run([str(binary),str(fixture),str(prefix),'3',str(2**-9),str(wd)],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=90)
        stats={int(a):(float(b),float(d),float(e)) for a,b,d,e in re.findall(r'PARITY step=(\d+) loss=(\S+) norm=(\S+) scale=(\S+)',log.read_text())}
        ref=Reference(values,dims,bool(mup)); replay=Reference(values,dims,bool(mup)); opt=optimizer(replay,wd)
        replay64=Reference(values,dims,bool(mup)).double(); opt64=optimizer(replay64,wd)
        trajectory=Reference(values,dims,bool(mup)); trajectory_opt=optimizer(trajectory,wd)
        x=torch.from_numpy(ids.astype(np.int64)); y=torch.from_numpy(targets.astype(np.int64)).ravel()
        for step in range(3):
            data={k:np.fromfile(str(prefix)+f'-{step}.{k}.bin',dtype=np.float32) for k in ['params','logits','grads','updated','moment1','moment2']}
            ref.load_flat(data['params'],dims); ref.zero_grad(set_to_none=True)
            logits=ref(x); loss=F.cross_entropy(logits.reshape(-1,8192),y); loss.backward()
            row=dict(case=name,step=step,loss_cuda=stats[step][0],loss_reference=loss.item(),checks={})
            row['checks']['logits']=error(data['logits'],logits.detach().numpy().ravel(),*TOLERANCES['logits'])
            grads=flat(ref,'grads'); offset=0
            for n,s in zip(NAMES,dims):
                size=math.prod(s); row['checks']['gradient_'+n]=error(data['grads'][offset:offset+size],grads[offset:offset+size],*TOLERANCES['gradients']); offset+=size
            row['grad_norm_reference']=float(np.linalg.norm(grads.astype(np.float64)))
            row['grad_norm_cuda']=stats[step][1]
            replay.load_flat(data['params'],dims)
            for n,a in split(data['grads'],dims).items(): replay.p[n].grad=torch.from_numpy(a.copy())*stats[step][2]
            opt.step()
            replay64.load_flat(data['params'],dims)
            for n,a in split(data['grads'],dims).items(): replay64.p[n].grad=(torch.from_numpy(a.copy())*stats[step][2]).double()
            opt64.step()
            row['official_float64_update']=error(data['updated'],flat(replay64),*TOLERANCES['updated'])
            for k,kind in [('updated','params'),('moment1','exp_avg'),('moment2','exp_avg_sq')]:
                row['checks'][k]=error(data[k],flat(replay,kind,opt),*TOLERANCES[k])
            if row['checks']['updated']['failures']:
                expected=flat(replay); mref=flat(replay,'exp_avg',opt); vref=flat(replay,'exp_avg_sq',opt)
                offsets=np.cumsum([0]+[math.prod(s) for s in dims]); details=[]
                for index in np.argsort(abs(data['updated']-expected))[-16:]:
                    tensor=NAMES[int(np.searchsorted(offsets,index,side='right')-1)]
                    details.append(dict(index=int(index),tensor=tensor,old=float(data['params'][index]),
                        gradient=float(data['grads'][index]),scale=stats[step][2],actual=float(data['updated'][index]),expected=float(expected[index]),
                        moment1_cuda=float(data['moment1'][index]),moment1_reference=float(mref[index]),
                        moment2_cuda=float(data['moment2'][index]),moment2_reference=float(vref[index])))
                row['update_failure_details']=details
                if step==0:
                    analytic=data['params'].astype(np.float64).copy(); offset=0
                    for n,s in zip(NAMES,dims):
                        size=math.prod(s); sl=slice(offset,offset+size); matrix=n in {'qkvw','attprojw','fcw','fcprojw'}
                        factor=c/192 if mup and matrix else 1; local_lr=2**-9/factor
                        local_wd=wd*factor if n in {'wte','wpe','qkvw','attprojw','fcw','fcprojw'} else 0
                        g=(data['grads'][sl]*np.float32(stats[step][2])).astype(np.float64)
                        analytic[sl]-=local_lr*(g/(abs(g)+1e-8)+local_wd*analytic[sl]); offset+=size
                    row['first_step_analytic_cuda']=error(data['updated'],analytic,*TOLERANCES['updated'])
                    row['first_step_analytic_official_fp32']=error(expected,analytic,*TOLERANCES['updated'])
            trajectory.zero_grad(set_to_none=True); tl=trajectory(x); tloss=F.cross_entropy(tl.reshape(-1,8192),y); tloss.backward()
            norm=torch.sqrt(sum(p.grad.square().sum() for p in trajectory.parameters())).item()
            for p in trajectory.parameters(): p.grad.mul_(1/norm if norm>1 else 1)
            trajectory_opt.step()
            row['independent_trajectory']=dict(loss=tloss.item(),loss_difference=tloss.item()-stats[step][0],parameters=error(data['updated'],flat(trajectory),2e-7,2e-5))
            row['passed']=(all(v['failures']==0 for v in row['checks'].values())
                           and row['official_float64_update']['failures']==0
                           and abs(row['loss_cuda']-row['loss_reference'])<=2e-5)
            records.append(row); print(name,step,'PASS' if row['passed'] else 'FAIL',flush=True)
            (out/'checks.json').write_text(json.dumps(records,indent=2))
        for f in args.root.glob(name+'*.bin'): f.unlink()
    summary=dict(status='passed' if all(r['passed'] for r in records) else 'failed',case_count=len(cases),step_checks=len(records),
                 seconds=time.monotonic()-started,failed=[{'case':r['case'],'step':r['step'],'checks':{k:v for k,v in r['checks'].items() if v['failures']}} for r in records if not r['passed']])
    (out/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary),flush=True)
    return int(summary['status']!='passed')

if __name__=='__main__': raise SystemExit(main())
