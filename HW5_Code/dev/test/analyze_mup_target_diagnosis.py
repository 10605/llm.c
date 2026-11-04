"""Report frozen-LR target transfer and separate secondary target-grid results."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics

def main():
    p=argparse.ArgumentParser(); p.add_argument('run',type=Path); p.add_argument('--queries',type=Path); a=p.parse_args(); run=a.run
    summary=json.loads((run/'summary.json').read_text()); all_rows=summary['records']; assert summary['status']=='completed' and len(all_rows)==21
    rows=[r for r in all_rows if r['width']==768]; extensions=[r for r in all_rows if r['width']==192]
    assert len(rows)==15 and len(extensions)==6
    selection=json.loads((run/'selection_before_target.json').read_text()); base=Path(selection['reference_run'])
    ref=json.loads((base/'analysis.json').read_text()); arm=selection['candidate_arm']; lr=2**selection['selected_exponent']
    data=[json.loads((run/f'worker{i}'/'dataset.json').read_text()) for i in range(4)]
    assert all(d==ref['data'] for d in data)
    sources=[json.loads((run/f'worker{i}'/'source.json').read_text()) for i in range(4)]; assert all(s==ref['source'] for s in sources)
    idx={(r['seed'],r['arm'],r['base_learning_rate']):r for r in rows}; assert len(idx)==15
    for r in rows:
        assert r['width']==768 and r['tokens']==2048000 and r['status']=='completed'
        assert len(r['training_log'])==500
        assert all(float(t[2])==float(f"{r['base_learning_rate']:.6f}") for t in r['training_log'])
    def group(name,rate):
        rs=[idx[s,name,rate] for s in [42,43,44]]; losses=[r['final_validation_loss'] for r in rs]
        return dict(arm=name,lr=rate,mean=statistics.mean(losses),std=statistics.stdev(losses),losses=dict(zip(['42','43','44'],losses)))
    current=group('zeroE-zeroQ',2**-9); candidate=group(arm,lr); standard=group('SP-control',2**-12)
    candidates=[group(arm,2**e) for e in selection['target_exponents']]; best=min(candidates,key=lambda x:x['mean'])
    paired=[dict(seed=s,current_gap=current['losses'][s]-standard['losses'][s],candidate_gap=candidate['losses'][s]-standard['losses'][s],candidate_minus_current=candidate['losses'][s]-current['losses'][s]) for s in ['42','43','44']]
    oldgap=current['mean']-standard['mean']; newgap=candidate['mean']-standard['mean']
    extended_curve=[]
    for e in [-7,-6]:
        group_rows=[r for r in extensions if r['base_learning_rate']==2**e]
        valid=[r for r in group_rows if r['status']=='completed']; losses=[r['final_validation_loss'] for r in valid]
        extended_curve.append(dict(arm='zeroE-zeroQ',exponent=e,lr=2**e,finite_seeds=len(valid),
            mean=statistics.mean(losses) if losses else None,std=statistics.stdev(losses) if len(losses)>1 else None,
            losses={str(r['seed']):r.get('final_validation_loss') for r in group_rows}))
    reference_current_best=min([r for r in ref['curves']+extended_curve if r['arm']=='zeroE-zeroQ' and r['finite_seeds']==3],key=lambda x:x['mean'])
    report=dict(status='passed',selection=selection,current=current,candidate_at_frozen_lr=candidate,sp_control=standard,
                candidate_grid=candidates,secondary_best_target_lr=best,paired=paired,
                old_mean_gap=oldgap,new_mean_gap=newgap,gap_reduction_fraction=(oldgap-newgap)/oldgap,
                reference_extension=extended_curve,reference_current_best_with_extension=reference_current_best,
                transferred_lr_regret_within_bracket=candidate['mean']-best['mean'],
                worker_seconds=summary['worker_seconds'],estimated_compute_cost_usd=summary['estimated_compute_cost_usd'],
                limitations=['500-update, 2.048M-token pilot; not a 20-TPP result.','Three initialization seeds with identical data order.',
                             'SP and current-muP controls use LRs chosen by the preceding full sweep; no fresh complete LR grid for them.',
                             'Only reference width 192 to target width 768 tested for the candidate; small-width transfer is not established.'])
    query_groups={}
    if a.queries:
        qs=json.loads((a.queries/'summary.json').read_text()); assert qs['status']=='completed' and len(qs['records'])==6
        qselection=json.loads((a.queries/'selection_before_target.json').read_text())
        for i in range(4):
            assert json.loads((a.queries/f'worker{i}'/'dataset.json').read_text())==ref['data']
            assert json.loads((a.queries/f'worker{i}'/'source.json').read_text())==ref['source']
        for name,e in qselection['query_controls'].items():
            group_rows=sorted([r for r in qs['records'] if r['arm']==name],key=lambda r:r['seed'])
            assert len(group_rows)==3
            for r in group_rows:
                assert r['width']==768 and r['base_learning_rate']==2**e and r['status']=='completed'
                assert len(r['training_log'])==500 and r['tokens']==2048000
            losses={str(r['seed']):r['final_validation_loss'] for r in group_rows}
            query_groups[name]=dict(arm=name,lr=2**e,mean=statistics.mean(losses.values()),std=statistics.stdev(losses.values()),losses=losses)
        report['query_followup']=dict(run=str(a.queries),groups=query_groups,estimated_compute_cost_usd=qs['estimated_compute_cost_usd'])
    (run/'analysis.json').write_text(json.dumps(report,indent=2))
    os.environ.setdefault('MPLCONFIGDIR','/private/tmp/llmc-mpl-cache')
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(15.5 if query_groups else 12.8,5.5),layout='constrained')
    colors=['#aa3377','#228833','#ee7733','#4477aa']
    arms=['zeroE-zeroQ','randomE-zeroQ','zeroE-randomQ','randomE-randomQ']
    labels=['Zero embedding / zero query','Random embedding / zero query','Zero embedding / random query','Random embedding / random query']
    for label,name,color in zip(labels,arms,colors):
        curve=[r for r in ref['curves']+extended_curve if r['arm']==name and r['finite_seeds']==3]
        axes[0].errorbar([math.log2(r['lr']) for r in curve],[r['mean'] for r in curve],yerr=[r['std'] for r in curve],color=color,marker='o',capsize=3,label=label)
    axes[0].set_title('15M reference model: isolate initialization'); axes[0].set_xlabel('log₂ learning rate'); axes[0].set_ylabel('Final validation loss'); axes[0].set_xticks(range(-13,-5))
    axes[0].legend(fontsize=8,frameon=False)
    target_groups=[current,candidate,standard]; target_colors=['#aa3377','#228833','#4477aa']
    ticklabels=['Current μP\nLR 2⁻⁹',f'Random embedding\nLR 2^{selection["selected_exponent"]}','SP control\nLR 2⁻¹²']
    if query_groups:
        target_groups=[current,query_groups['zeroE-randomQ'],candidate,query_groups['randomE-randomQ'],standard]
        target_colors=['#aa3377','#ee7733','#228833','#4477aa','#333333']
        ticklabels=['Current μP\nLR 2⁻⁹','Random query\nLR 2⁻⁹','Random embedding\nLR 2⁻¹¹','Both random\nLR 2⁻¹¹','SP control\nLR 2⁻¹²']
    positions=list(range(len(target_groups)))
    for j,seed in enumerate(['42','43','44']):
        axes[1].plot([i+(j-1)*.09 for i in positions],[g['losses'][seed] for g in target_groups],color='#bbbbbb',linewidth=.8,zorder=1)
    for i,(g,color) in enumerate(zip(target_groups,target_colors)):
        axes[1].scatter([i-.09,i,i+.09],list(g['losses'].values()),s=25,alpha=.55,color=color,zorder=3)
        axes[1].errorbar([i],[g['mean']],yerr=[g['std']],fmt='D',markersize=8,color=color,capsize=5,zorder=4)
        axes[1].annotate(f"{g['mean']:.4f}",(i,g['mean']),xytext=(0,15),textcoords='offset points',ha='center',fontweight='bold',color=color)
    axes[1].set_xticks(positions,ticklabels)
    axes[1].set_ylabel('Final validation loss'); axes[1].set_title('124M target: preset LRs, paired seeds\nDots: seeds · diamonds: mean ± standard deviation')
    axes[1].set_xlim(-.4,len(target_groups)-.6); axes[1].margins(y=.3)
    for ax in axes: ax.grid(alpha=.18,axis='y')
    fig.suptitle('Diagnosing the μP loss gap\nFineWeb · 500 updates / 2.048M tokens · constant LR · decay 0 · three seeds',fontweight='bold')
    for suffix in ['png','pdf']: fig.savefig(run/('diagnosis-summary.'+suffix),dpi=180,bbox_inches='tight')
    print(json.dumps({k:v for k,v in report.items() if k not in {'selection','limitations','candidate_grid'}},indent=2))

if __name__=='__main__': main()
