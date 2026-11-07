"""Audit and plot the matched muP-only horizon comparison."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from run_mup_horizon_sweep import WIDTHS, HORIZONS, verify_metrics


def analyze(root):
    summary = json.loads((root/'summary.json').read_text())
    plan = json.loads((root/'plan.json').read_text())
    rows = summary['records']
    expected = {r['name']:r for r in plan['trials']}
    assert summary['status']=='completed' and len(rows)==len(expected)
    assert len({r['name'] for r in rows})==len(expected) and {r['name'] for r in rows}==set(expected)
    manifest = json.loads((root/'source_manifest.json').read_text())['files']
    order = json.loads((root/'data-order-audit.json').read_text())
    assert order['status']=='passed' and order['nested_training_prefixes']
    assert manifest['llmc/dataloader.h']==order['source_sha256']
    locations = {}
    initial_losses = {w:[] for w in WIDTHS}
    for summary_file in sorted(root.glob('**/worker[0-9]*/summary.json')):
        worker = summary_file.parent
        child = json.loads(summary_file.read_text())
        assert child['status']=='completed' and child['dataset']==plan['dataset']
        assert child['source_sha256']==manifest['train_gpt2.cu']
        for r in child['records']:
            assert r['name'] not in locations
            locations[r['name']] = worker/r['name']
            params = (worker/r['name']/'run_parameters.txt').read_text()
            for text in ['batch=16 sequence=256 total_batch=4096 accumulation=1 ranks=1',
                         'mup=1 ', 'scheduler=cosine', 'weight_decay=0.100000001',
                         'val_batches=256 sample_every=0']:
                assert text in params
            log = (worker/(r['name']+'.log')).read_text()
            assert 'skipping update' not in log and 'optimizer state uses managed memory' not in log
    assert set(locations)==set(expected)
    minima = []
    for r in rows:
        assert r['mode']=='mup' and r['status'] in {'completed','diverged'}
        assert r['exit_code']==0 and r['observed_steps']==r['steps']==expected[r['name']]['steps']
        assert r['tokens']==r['steps']*4096
        if r['status']=='completed':
            metrics=verify_metrics(locations[r['name']],r)
            assert metrics['final_validation_loss']==r['final_validation_loss']
            initial_losses[r['width']].append(float(metrics['validation_log'][0][1]))
    for w, losses in initial_losses.items():
        assert max(losses)-min(losses) <= 0.0002, (w, losses)
    for h in HORIZONS:
        for w in WIDTHS:
            group=sorted([r for r in rows if r['horizon']==h and r['width']==w],key=lambda r:r['base_learning_rate'])
            assert len(group)==sum(t['horizon']==h and t['width']==w for t in plan['trials'])
            assert len({r['base_learning_rate'] for r in group})==len(group)
            if plan.get('dense_lr_interval_log2'):
                lo,hi=plan['dense_lr_interval_log2']
                step=plan['dense_lr_step_log2']
                dense=sorted(r['base_learning_rate'] for r in group if 2**lo <= r['base_learning_rate'] <= 2**hi)
                assert dense==[2**(lo+i*step) for i in range(round((hi-lo)/step)+1)]
            finite=[r for r in group if r['status']=='completed']
            if not finite:
                minima.append(dict(horizon=h,width=w,status='no_finite_trials'))
                continue
            best=min(finite,key=lambda r:r['final_validation_loss'])
            minima.append(dict(horizon=h,width=w,status='completed',lr=best['base_learning_rate'],
                log2_lr=round(math.log2(best['base_learning_rate']),8),loss=best['final_validation_loss'],
                boundary=best['base_learning_rate'] in [group[0]['base_learning_rate'],group[-1]['base_learning_rate']],
                dense_interval_edge=(math.log2(best['base_learning_rate']) <= plan['dense_lr_interval_log2'][0]
                    or math.log2(best['base_learning_rate']) >= plan['dense_lr_interval_log2'][1]) if plan.get('dense_lr_interval_log2') else False,
                within_001_nats=[round(math.log2(r['base_learning_rate']),8) for r in finite
                                 if r['final_validation_loss']-best['final_validation_loss']<=.01]))
    for m in minima:
        if m['status']!='completed': continue
        baseline=next(x for x in minima if x['horizon']==1 and x['width']==m['width'])
        reused=next(r for r in rows if r['horizon']==m['horizon'] and r['width']==m['width']
                    and r['base_learning_rate']==baseline['lr'])
        m['shift_in_doublings_from_1x']=m['log2_lr']-baseline['log2_lr']
        m['regret_reusing_1x_lr']=reused['final_validation_loss']-m['loss'] if reused['status']=='completed' else None
    width_transfer=[]
    for h in HORIZONS:
        small=next(m for m in minima if m['horizon']==h and m['width']==48)
        for w in [96,192]:
            best=next(m for m in minima if m['horizon']==h and m['width']==w)
            reused=next(r for r in rows if r['horizon']==h and r['width']==w and r['base_learning_rate']==small['lr'])
            width_transfer.append(dict(horizon=h,width=w,lr_selected_at_width48=small['lr'],
                loss_regret=reused['final_validation_loss']-best['loss'] if reused['status']=='completed' else None))
    report=dict(status='passed',minima=minima,completed=sum(r['status']=='completed' for r in rows),
        diverged=sum(r['status']=='diverged' for r in rows),source_sha256=manifest['train_gpt2.cu'],
        width_transfer=width_transfer,
        initial_validation_loss_ranges={str(w):[min(x),max(x)] for w,x in initial_losses.items()},
        data_order_audit=order,estimated_worker_compute_cost_usd=summary['estimated_compute_cost_usd'],
        prior_compute_cost_usd=summary.get('prior_compute_cost_usd',0),
        new_trial_count=summary.get('new_trial_count',len(rows)),prior_trial_count=summary.get('prior_trial_count',0),
        lr_grid_note=plan.get('lr_grid_note','2× LR grid'),
        limitation='One initialization seed; discrete LR grid. The .01-nat band is descriptive, not a confidence interval. Horizon changes include the normalized schedule duration and cumulative weight decay.')
    (root/'analysis.json').write_text(json.dumps(report,indent=2))
    return rows,report


def plot(root, rows, report):
    os.environ.setdefault('MPLCONFIGDIR',str(root/'.mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    colors={48:'#6a3d9a',96:'#26828e',192:'#35b779'}
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(15,8.5),sharex=True,sharey='row',layout='constrained')
    for col,h in enumerate(HORIZONS):
        axes[0,col].set_title(f'{h}×: {2.048*h:g}M tokens\n{500*h:,} updates',fontweight='bold')
        for wi,w in enumerate(WIDTHS):
            group=sorted([r for r in rows if r['horizon']==h and r['width']==w],key=lambda r:r['base_learning_rate'])
            m=next(x for x in report['minima'] if x['horizon']==h and x['width']==w)
            x=[math.log2(r['base_learning_rate']) for r in group]
            y=[r.get('final_validation_loss',float('nan')) if r['status']=='completed' else float('nan') for r in group]
            for row in [0,1]:
                values=y if row==0 else [v-m['loss'] for v in y]
                axes[row,col].plot(x,values,'o-',color=colors[w],lw=1.8,ms=4,label=f'Width {w}')
                axes[row,col].scatter([m['log2_lr']],[m['loss'] if row==0 else 0],s=110,
                    facecolors='none',edgecolors=colors[w],linewidths=2)
                bad=[math.log2(r['base_learning_rate']) for r in group if r['status']=='diverged']
                axes[row,col].plot(bad,[.97-.04*wi]*len(bad),'x',color=colors[w],
                                  transform=axes[row,col].get_xaxis_transform())
                axes[row,col].grid(alpha=.2)
            axes[1,col].set_xlabel('log₂ peak base learning rate')
        axes[1,col].set_xticks(range(-14,-5,2))
        axes[1,col].axhline(.01,ls=':',color='gray',lw=1)
    axes[0,0].set_ylabel('Final held-out validation loss')
    axes[1,0].set_ylabel('Loss above best sampled LR\nwithin this width and budget')
    axes[0,2].legend(frameon=False)
    finite=[r['final_validation_loss'] for r in rows if r['status']=='completed']
    for col in range(3):
        axes[0,col].set_ylim(min(finite)-.12,8.5)
        axes[1,col].set_ylim(-.025,.6)
    plan=json.loads((root/'plan.json').read_text())
    dense_interval=plan.get('dense_lr_interval_log2')
    if dense_interval:
        from matplotlib.ticker import MultipleLocator
        lo,hi=dense_interval
        for ax in axes.flat:
            ax.set_xlim(lo-.12,hi+.12)
            ax.xaxis.set_major_locator(MultipleLocator(1))
            ax.xaxis.set_minor_locator(MultipleLocator(.25))
    fig.suptitle('FineWeb: does the μP learning-rate optimum shift with training length?\n'
        'Cosine · 10% warmup · weight decay 0.1 · 12 layers / 12 heads · seed 42\n'
        'Matched microbatch 16, 4,096 tokens/update, nested training data, 1M held-out tokens\n'
        'Circles: best sampled LR; lower row zoomed near minima (full range saved separately)',fontsize=13)
    fig.savefig(root/'mup-horizon-comparison.png',dpi=180)
    fig.savefig(root/'mup-horizon-comparison.pdf')
    if dense_interval:
        for ax in axes.flat:
            ax.set_xlim(min(math.log2(r['base_learning_rate']) for r in rows)-.2,
                        max(math.log2(r['base_learning_rate']) for r in rows)+.2)
            ax.xaxis.set_major_locator(MultipleLocator(2))
    for col in range(3):
        axes[0,col].set_ylim(min(finite)-.12,max(finite)+.1)
        axes[1,col].set_ylim(-.025,max(r['final_validation_loss']-next(m['loss'] for m in report['minima']
            if m['width']==r['width'] and m['horizon']==r['horizon']) for r in rows if r['status']=='completed')+.1)
    fig.savefig(root/'mup-horizon-comparison-full.png',dpi=150)
    plt.close(fig)
    fig,shift_axes=plt.subplots(1,3,figsize=(12,4.5),sharey=True,layout='constrained')
    for ax,w in zip(shift_axes,WIDTHS):
        group=sorted([m for m in report['minima'] if m['width']==w],key=lambda m:m['horizon'])
        x=np.array([math.log2(m['horizon']) for m in group])
        y=np.array([m['log2_lr'] for m in group])
        low=np.array([min(m['within_001_nats']) for m in group])
        high=np.array([max(m['within_001_nats']) for m in group])
        ax.errorbar(x,y,yerr=[y-low,high-y],fmt='o-',color=colors[w],capsize=5,lw=2)
        ax.set_xticks([0,3,4],['1×','8×','16×'])
        ax.set_yticks(range(-14,-5))
        selected=[e for m in report['minima'] for e in m['within_001_nats']]
        ax.set_ylim(min(selected)-.6,max(selected)+.6)
        ax.set_title(f'Width {w}',fontweight='bold')
        ax.grid(alpha=.2)
    shift_axes[0].set_ylabel('log₂ best sampled peak base LR')
    fig.supxlabel('Training budget (1× = 2.048M tokens)',fontsize=11)
    fig.suptitle('μP: learning-rate preference across training budgets\n'
        'Points: best sampled LR; bars: span of sampled LRs within 0.01 nats of best\n'
        'One seed; bars are descriptive ranges, not confidence intervals',fontsize=12)
    fig.savefig(root/'mup-optimal-lr-shift.png',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',type=Path)
    args=parser.parse_args()
    rows,report=analyze(args.results)
    plot(args.results,rows,report)
    print(json.dumps(report,indent=2))
