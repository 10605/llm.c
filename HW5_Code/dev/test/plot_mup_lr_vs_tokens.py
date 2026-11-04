"""Plot median best sampled muP LR against training tokens on logarithmic axes."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main(root):
    os.environ.setdefault('MPLCONFIGDIR', str(root/'.mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    import numpy as np

    analysis_path = root/'analysis.json'
    analysis = json.loads(analysis_path.read_text())
    plan = json.loads((root/'plan.json').read_text())
    grid_note = plan.get('lr_grid_note','2× LR grid')
    assert analysis['status']=='passed'
    minima = analysis['minima']
    widths = sorted({m['width'] for m in minima})
    horizons = sorted({m['horizon'] for m in minima})
    tokens = []
    for h in horizons:
        budgets = {t['tokens'] for t in plan['trials'] if t['horizon']==h}
        assert len(budgets)==1
        tokens.append(budgets.pop())
    d = np.asarray(tokens, dtype=float)
    series = []
    for w in widths:
        group = [next(m for m in minima if m['width']==w and m['horizon']==h) for h in horizons]
        assert all(m['status']=='completed' and not m['boundary'] for m in group)
        series.append([m['lr'] for m in group])
    series = np.asarray(series)
    assert np.all(series > 0) and np.all(d > 0)
    median = np.median(series, axis=0)
    # Fit the medians in log space, within the measured interval only.
    exponent, intercept = np.polyfit(np.log(d/d[0]), np.log(median), 1)
    coefficient = np.exp(intercept)
    fitted = coefficient*(d/d[0])**exponent
    report = dict(training_tokens=tokens, widths=widths, depth_layers=12,
        best_tested_lrs_by_width={str(w):values.tolist() for w,values in zip(widths,series)},
        median_best_tested_lr=median.tolist(),
        descriptive_fit=dict(formula='lr = coefficient * (training_tokens / reference_tokens)**exponent',
            coefficient=float(coefficient),reference_tokens=tokens[0],exponent=float(exponent),
            log2_rmse=float(np.sqrt(np.mean(np.log2(median/fitted)**2))),
            domain_tokens=[tokens[0],tokens[-1]],method='Unweighted least squares of log median LR on log token budget'),
        source_analysis_sha256=hashlib.sha256(analysis_path.read_bytes()).hexdigest(),
        lr_grid_note=grid_note,
        limitations=['Three measured budgets, one initialization seed, discrete LR grid.',
            'Widths are different architectures, not statistical replicates.',
            'The median summarizes grid minima; broad near-optimal ranges remain in analysis.json.',
            'The straight log-space fit is descriptive; no extrapolation or established scaling law is claimed.'])
    coarse_path=root/'coarse'/'summary.json'
    if coarse_path.exists():
        coarse_rows=json.loads(coarse_path.read_text())['records']
        coarse_series=[[min((r for r in coarse_rows if r['width']==w and r['horizon']==h
                             and r['status']=='completed'),key=lambda r:r['final_validation_loss'])['base_learning_rate']
                        for h in horizons] for w in widths]
        coarse_median=np.median(coarse_series,axis=0)
        old_exponent,old_intercept=np.polyfit(np.log(d/d[0]),np.log(coarse_median),1)
        old_fitted=np.exp(old_intercept)*(d/d[0])**old_exponent
        report['coarse_grid_comparison']=dict(median_best_tested_lr=coarse_median.tolist(),
            exponent=float(old_exponent),log2_rmse=float(np.sqrt(np.mean(np.log2(coarse_median/old_fitted)**2))),
            source_summary_sha256=hashlib.sha256(coarse_path.read_bytes()).hexdigest())
    (root/'mup-optimal-lr-vs-tokens-loglog.json').write_text(json.dumps(report,indent=2)+'\n')

    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(10,6.8))
    fig.subplots_adjust(left=.12,right=.97,bottom=.2,top=.80)
    ax.set_xscale('log',base=2)
    ax.set_yscale('log',base=2)
    colors=['#6a3d9a','#26828e','#35b779']
    markers=['s','^','D']
    for w,values,color,marker in zip(widths,series,colors,markers):
        ax.plot(d,values,color=color,alpha=.45,lw=1.3,marker=marker,ms=6,
                label=f'Width {w}',zorder=2)
    dense=np.geomspace(d[0],d[-1],200)
    ax.plot(dense,coefficient*(dense/d[0])**exponent,'--',color='#636363',lw=1.8,
            label=f'Log–log fit: LR ∝ tokens$^{{{exponent:.2f}}}$',zorder=3)
    ax.plot(d,median,'o-',color='#172b4d',lw=2.8,ms=9,label='Median across widths',zorder=4)
    for x,y in zip(d,median):
        ax.annotate(f'{y:.5f}',(x,y),xytext=(0,13),textcoords='offset points',
                    ha='center',fontsize=10,color='#172b4d',fontweight='bold',zorder=5)
    x_ticks=d[0]*2.0**np.arange(0,round(np.log2(d[-1]/d[0]))+1)
    ax.xaxis.set_major_locator(FixedLocator(x_ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([f'{v/1e6:g}M' for v in x_ticks]))
    ax.xaxis.set_minor_locator(NullLocator())
    y_ticks=2.0**np.arange(np.floor(np.log2(series.min())),np.ceil(np.log2(series.max()))+1)
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    ax.yaxis.set_major_formatter(FixedFormatter([f'{v:.5f}' for v in y_ticks]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlim(d[0]/1.22,d[-1]*1.22)
    ax.set_ylim(series.min()/1.3,series.max()*1.55)
    ax.grid(which='major',alpha=.2)
    ax.set_xlabel('Training tokens (log scale)',labelpad=10)
    ax.set_ylabel('Best tested peak base learning rate (log scale)',labelpad=10)
    handles,labels=ax.get_legend_handles_labels()
    order=[4,3,0,1,2]
    ax.legend([handles[i] for i in order],[labels[i] for i in order],
              loc='upper left',frameon=True,facecolor='white',edgecolor='#eeeeee',fontsize=9.5)
    fig.suptitle('μP: optimal learning rate vs. training-data budget',fontsize=17,y=.965)
    fig.text(.5,.90,'Median over widths 48, 96, 192 · all 12 layers\n'
             'Cosine + 10% warmup · weight decay 0.1 · matched data and microbatch',
             ha='center',va='center',fontsize=11)
    fig.text(.5,.065,f'Three budgets · one seed · {grid_note}\n'
             'Dashed line is a descriptive fit over the measured range.',
             ha='center',fontsize=10,color='#555555')
    for extension in ['png','pdf','svg']:
        fig.savefig(root/f'mup-optimal-lr-vs-tokens-loglog.{extension}',dpi=200)
    plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',type=Path)
    main(parser.parse_args().results)
