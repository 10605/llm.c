"""Audit and plot the width-192 initialization diagnosis, retaining all seeds."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics

ARMS=['zeroE-zeroQ','randomE-zeroQ','zeroE-randomQ','randomE-randomQ']
LABELS=['Zero embedding / zero query (current)','Random embedding / zero query','Zero embedding / random query','Random embedding / random query (SP initialization)']

def main():
    p=argparse.ArgumentParser(); p.add_argument('run',type=Path); a=p.parse_args(); run=a.run
    report=json.loads((run/'summary.json').read_text()); rows=report['records']
    assert report['status']=='completed' and len(rows)==78
    idx={(r['seed'],r['arm'],r['base_learning_rate']):r for r in rows}; assert len(idx)==78
    data=[json.loads((run/f'worker{i}'/'dataset.json').read_text()) for i in range(4)]; assert all(x==data[0] for x in data)
    source=[json.loads((run/f'worker{i}'/'source.json').read_text()) for i in range(4)]; assert all(x==source[0] for x in source)
    hashes={}
    for r in rows:
        key=(r['seed'],r['arm']); hashes.setdefault(key,set()).add(r['initialization_sha256'])
        assert r['width']==192 and r['tokens']==2048000
        if r['status']=='completed':
            assert len(r['training_log'])==500
            assert all(float(t[2])==float(f"{r['base_learning_rate']:.6f}") for t in r['training_log'])
    assert all(len(h)==1 for h in hashes.values())
    controls=[]
    for seed in [42,43,44]:
        assert hashes[seed,'SP-control']==hashes[seed,'randomE-randomQ']
        for lr in [2**-11,2**-9]:
            sp=idx[seed,'SP-control',lr]; mu=idx[seed,'randomE-randomQ',lr]
            controls.append(dict(seed=seed,lr=lr,initialization_identical=True,
                final_loss_difference=mu['final_validation_loss']-sp['final_validation_loss'],
                max_training_loss_difference=max(abs(float(x[1])-float(y[1])) for x,y in zip(sp['training_log'],mu['training_log']))))
    curves=[]; minima=[]
    for arm in ARMS:
        for e in range(-13,-7):
            group=[idx[s,arm,2**e] for s in [42,43,44]]; valid=[r for r in group if r['status']=='completed']
            losses=[r['final_validation_loss'] for r in valid]
            curves.append(dict(arm=arm,exponent=e,lr=2**e,finite_seeds=len(valid),
                mean=statistics.mean(losses) if losses else None,std=statistics.stdev(losses) if len(losses)>1 else None,
                losses={str(r['seed']):r.get('final_validation_loss') for r in group}))
        candidates=[r for r in curves if r['arm']==arm and r['finite_seeds']==3]
        best=min(candidates,key=lambda r:r['mean']); minima.append(dict(best,boundary=best['exponent'] in {-13,-8}))
    base=next(r for r in minima if r['arm']=='zeroE-zeroQ')
    for best in minima:
        best['paired_loss_change_from_current']={s:best['losses'][s]-base['losses'][s] for s in base['losses']}
        best['mean_loss_change_from_current']=best['mean']-base['mean']
    analysis=dict(status='passed',scope='Reference width 192; three seeds; 500 updates / 2.048M tokens; constant LR / zero decay',
        minima=minima,curves=curves,sp_controls=controls,data=data[0],source=source[0],
        worker_seconds=report['worker_seconds'],estimated_compute_cost_usd=report['estimated_compute_cost_usd'],
        limitations=['LR selected using mean validation loss over these same three seeds; this is a diagnostic, not an independent generalization estimate.',
                     'Reference-width results do not establish 124M loss or width-transfer performance.',
                     'Only initialization seed changes; token order and stochastic-rounding seed remain fixed.'])
    (run/'analysis.json').write_text(json.dumps(analysis,indent=2))
    os.environ.setdefault('MPLCONFIGDIR','/private/tmp/llmc-mpl-cache')
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#aa3377','#228833','#ee7733','#4477aa']
    fig,axes=plt.subplots(1,2,figsize=(13,5.7),layout='constrained')
    for arm,label,color in zip(ARMS,LABELS,colors):
        curve=[c for c in curves if c['arm']==arm]; xs=[c['exponent'] for c in curve]
        ys=[c['mean'] if c['finite_seeds']==3 else float('nan') for c in curve]
        es=[c['std'] if c['finite_seeds']==3 else 0 for c in curve]
        axes[0].errorbar(xs,ys,yerr=es,marker='o',capsize=3,color=color,label=label,linewidth=1.8)
        best=next(c for c in minima if c['arm']==arm)
        selected=[idx[s,arm,best['lr']] for s in [42,43,44]]
        steps=[int(v[0]) for v in selected[0]['validation_log']]
        avg=[statistics.mean(float(r['validation_log'][i][1]) for r in selected) for i in range(len(steps))]
        visible=[(s,v) for s,v in zip(steps,avg) if s>=50]
        axes[1].plot([s for s,v in visible],[v for s,v in visible],color=color,marker='.',label=f'{label}: LR 2^{best["exponent"]}')
    axes[0].set_xlabel('log₂ learning rate'); axes[0].set_ylabel('Final validation loss'); axes[0].set_xticks(range(-13,-7))
    axes[0].set_title('LR sweep · mean ± one seed standard deviation')
    axes[1].set_xlabel('Training updates'); axes[1].set_ylabel('Validation loss'); axes[1].set_title('Updates 50–500 at each arm’s selected LR')
    for ax in axes: ax.grid(alpha=.2)
    axes[1].set_ylim(bottom=min(m['mean'] for m in minima)-.07,top=8.1)
    handles,labels=axes[0].get_legend_handles_labels(); fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
    fig.suptitle('Diagnosing the μP loss gap at reference width 192\nFineWeb · 12 layers / 12 heads · 2.048M tokens · seeds 42, 43, 44',fontweight='bold')
    for suffix in ['png','pdf']: fig.savefig(run/('initialization-diagnosis.'+suffix),dpi=180,bbox_inches='tight')
    print(json.dumps({'minima':minima,'sp_controls':controls,'cost':report['estimated_compute_cost_usd']},indent=2))

if __name__=='__main__': main()
