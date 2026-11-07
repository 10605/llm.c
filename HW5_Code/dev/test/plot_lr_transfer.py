"""Plot completed trials and report grid minima without treating boundary minima as optima."""
from pathlib import Path
import argparse
import json
import math
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results', type=Path)
    parser.add_argument('--png-only', action='store_true')
    parser.add_argument('--implementation-note', default='')
    args = parser.parse_args()
    os.environ.setdefault('MPLCONFIGDIR', str(args.results / '.mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary = json.loads((args.results/'summary.json').read_text())
    rows = summary['records']
    experiment = summary.get('experiment') or {'scheduler':'constant', 'warmup_steps':0,
                                              'final_lr_fraction':1.0, 'weight_decay':0.0}
    widths = sorted({r['width'] for r in rows})
    minima = []
    for mode in ['sp', 'mup']:
        for width in widths:
            group = sorted([r for r in rows if r['mode']==mode and r['width']==width], key=lambda r:r['base_learning_rate'])
            valid = [r for r in group if r['status']=='completed']
            if not valid:
                continue
            best = min(valid, key=lambda r:r['final_validation_loss'])
            minima.append({'mode':mode, 'width':width, 'parameters':best['parameters'],
                           'base_learning_rate':best['base_learning_rate'],
                           'log2_lr':math.log2(best['base_learning_rate']),
                           'validation_loss':best['final_validation_loss'],
                           'grid_boundary':best in [group[0], group[-1]]})
    report = {'minima': minima, 'transfer':summary.get('transfer'), 'experiment':experiment,
              'completed':sum(r['status']=='completed' for r in rows),
              'diverged':sum(r['status']=='diverged' for r in rows),
              'failed':sum(r['status']=='failed' for r in rows),
              'new_trials':summary.get('new_trial_count',len(rows)),
              'prior_run':summary.get('prior_run',''),
              'lr_range':[min(r['base_learning_rate'] for r in rows), max(r['base_learning_rate'] for r in rows)],
              'worker_hours':summary.get('worker_seconds',0)/3600,
              'estimated_compute_cost_usd':summary.get('estimated_compute_cost_usd'),
              'limitation':'One initialization seed, 2.048M tokens per trial. Grid minima measure early training; boundary minima need a wider search.',
              'implementation_note':args.implementation_note}
    (args.results/'analysis.json').write_text(json.dumps(report, indent=2))
    plt.rcParams.update({'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(2,2,figsize=(12,8),sharex=True,sharey='row',layout='constrained')
    colors = ['#6a3d9a','#26828e','#35b779','#e8a330','#cf4a4a']
    for col, mode in enumerate(['sp','mup']):
        best_target = min([r for r in rows if r['mode']==mode and r['width']==768 and r['status']=='completed'],
                          key=lambda r:r['final_validation_loss'])
        heading = 'Standard parametrization' if mode=='sp' else 'μP'
        axes[0,col].set_title(f'{heading}\n124M best sampled LR: {best_target["base_learning_rate"]:.6g}',fontweight='bold')
        for row in range(2):
            axes[row,col].axvline(math.log2(best_target['base_learning_rate']),color='#cf4a4a',ls='--',lw=1,alpha=.5)
        for color_index, (width, color) in enumerate(zip(widths,colors)):
            group = sorted([r for r in rows if r['mode']==mode and r['width']==width],key=lambda r:r['base_learning_rate'])
            x = [math.log2(r['base_learning_rate']) for r in group]
            for row, metric in enumerate(['last20_training_loss','final_validation_loss']):
                y = [r.get(metric,float('nan')) if r['status']=='completed' else float('nan') for r in group]
                axes[row,col].plot(x,y,'o-',ms=4,lw=1.6,color=color,label=f'{width} ({group[0]["parameters"]/1e6:.1f}M)')
                divergent = [math.log2(r['base_learning_rate']) for r in group if r['status']=='diverged']
                axes[row,col].plot(divergent,[.97-.035*color_index]*len(divergent),'x',ms=5,color=color,
                                   transform=axes[row,col].get_xaxis_transform())
                axes[row,col].grid(alpha=.18)
        axes[1,col].set_xlabel('log₂ peak base learning rate')
    axes[0,0].set_ylabel('Training loss\n(mean of last 20 updates)')
    axes[1,0].set_ylabel('Final held-out validation loss')
    axes[0,1].legend(title='Width (parameters)',fontsize=9,frameon=False)
    title = 'FineWeb: learning-rate transfer across widths\n500 updates · 2.048M tokens per trial · 12 layers / 12 heads · seed 42'
    title += (f'\n{experiment["scheduler"]} schedule · {experiment["warmup_steps"]} warmup updates'
              f' · final LR fraction {experiment["final_lr_fraction"]:g} · base weight decay {experiment["weight_decay"]:g}')
    if args.implementation_note:
        title += '\n' + args.implementation_note
    if report['diverged']:
        title += '\n× in top strip: nonfinite loss or gradient norm (no finite point plotted)'
    fig.suptitle(title,fontsize=12)
    for row, metric in enumerate(['last20_training_loss','final_validation_loss']):
        finite = [r[metric] for r in rows if r['status']=='completed']
        if max(finite) > 2*min(finite):
            for col in range(2):
                axes[row,col].set_yscale('log')
    fig.savefig(args.results/'lr-transfer.png',dpi=180)
    if not args.png_only:
        fig.savefig(args.results/'lr-transfer.pdf')
    for row, metric in enumerate(['last20_training_loss','final_validation_loss']):
        low = min(r[metric] for r in rows if r['status']=='completed') - .1
        for col in range(2):
            axes[row,col].set_yscale('linear')
            axes[row,col].set_ylim(low,8)
    fig.suptitle(title+'\nZoom near minima: finite losses above 8 are outside this view; full range saved separately',fontsize=11)
    fig.savefig(args.results/'lr-transfer-zoom.png',dpi=180)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
