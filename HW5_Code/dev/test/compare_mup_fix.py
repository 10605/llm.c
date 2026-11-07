"""Compare the original and corrected constant-LR, zero-decay FineWeb sweeps."""
from pathlib import Path
import argparse
import json
import math
import os
from compare_lr_schedules import load_run


def compare(before, fixed, output):
    runs = [load_run(before), load_run(fixed)]
    expected = dict(scheduler='constant', warmup_steps=0, final_lr_fraction=1.0, weight_decay=0.0)
    assert all(r['experiment'] == expected for r in runs)
    verification = json.loads((fixed/'verification.json').read_text())
    assert verification['status'] == 'passed'
    assert verification['baseline'] == str(before.resolve())
    assert set(verification['expected_source_changes']) == {'train_gpt2.cu'}
    indexed = [{(r['mode'],r['width'],r['base_learning_rate']):r for r in run['rows']} for run in runs]
    assert indexed[0].keys() == indexed[1].keys()
    def command_settings(row):
        settings = dict(zip(row['command'][1::2], row['command'][2::2]))
        settings.pop('-o')
        settings['-q'] = str(float(settings.get('-q', 1)))
        for flag in ['-c','-l','-mm','-mb','-ma']:
            if flag in settings:
                settings[flag] = str(float(settings[flag]))
        return settings
    for key in indexed[0]:
        assert command_settings(indexed[0][key]) == command_settings(indexed[1][key]), key
    minima = []
    widths = sorted({r['width'] for r in runs[0]['rows']})
    for label, run in zip(['before','fixed'], runs):
        for mode in ['sp','mup']:
            for width in widths:
                group = [r for r in run['rows'] if r['mode']==mode and r['width']==width]
                valid = [r for r in group if r['status']=='completed']
                best = min(valid, key=lambda r:r['final_validation_loss'])
                minima.append(dict(version=label, mode=mode, width=width,
                    base_learning_rate=best['base_learning_rate'], validation_loss=best['final_validation_loss'],
                    grid_boundary=best['base_learning_rate'] in
                    [min(r['base_learning_rate'] for r in group), max(r['base_learning_rate'] for r in group)]))
    def minimum(version, mode, width):
        return next(m for m in minima if (m['version'],m['mode'],m['width'])==(version,mode,width))
    gaps = [dict(width=w, before=minimum('before','mup',w)['validation_loss']-minimum('before','sp',w)['validation_loss'],
                 fixed=minimum('fixed','mup',w)['validation_loss']-minimum('fixed','sp',w)['validation_loss']) for w in widths]
    controls = []
    for key, a in indexed[0].items():
        if key[0]!='sp' and key[1]!=192:
            continue
        b = indexed[1][key]
        row = dict(mode=key[0], width=key[1], base_learning_rate=key[2],
                   before_status=a['status'], fixed_status=b['status'], status_matches=a['status']==b['status'])
        if a['status']==b['status']=='completed':
            # Time is intentionally excluded; these are logged numerical trajectories.
            ta = [r[:4] for r in a['training_log']]
            tb = [r[:4] for r in b['training_log']]
            row.update(training_trajectory_identical=ta==tb,
                validation_trajectory_identical=a['validation_log']==b['validation_log'],
                final_validation_difference=b['final_validation_loss']-a['final_validation_loss'],
                max_training_loss_difference=max(abs(float(x[1])-float(y[1])) for x,y in zip(ta,tb)))
        controls.append(row)
    finite_controls = [r for r in controls if 'final_validation_difference' in r]
    control_summary = dict(trials=len(controls), status_matches=sum(r['status_matches'] for r in controls),
        finite_pairs=len(finite_controls),
        identical_training_trajectories=sum(r['training_trajectory_identical'] for r in finite_controls),
        identical_validation_trajectories=sum(r['validation_trajectory_identical'] for r in finite_controls),
        identical_final_losses=sum(r['final_validation_difference']==0 for r in finite_controls),
        maximum_absolute_final_loss_difference=max(abs(r['final_validation_difference']) for r in finite_controls))
    report = dict(runs=[dict(path=r['path'],experiment=r['experiment'],transfer=r['summary']['transfer'],
                    completed=sum(x['status']=='completed' for x in r['rows']),
                    diverged=sum(x['status']=='diverged' for x in r['rows']),
                    failed=sum(x['status']=='failed' for x in r['rows'])) for r in runs],
                  minima=minima, best_sampled_mup_minus_sp_loss=gaps, unchanged_math_controls=controls,
                  control_summary=control_summary,
                  limitations=['One seed, 500 updates and 2.048M tokens per model; this measures early training.',
                               'All results use constant LR and zero decay; earlier cosine runs are not corrected.',
                               'Unchanged-math controls can vary across reruns; consult per-trial controls, especially at unstable large LRs.',
                               'Sampled grid minima do not establish a continuous global optimum.'])
    output.mkdir(parents=True, exist_ok=True)
    (output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    os.environ.setdefault('MPLCONFIGDIR','/private/tmp/llmc-mpl-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    colors = ['#6a3d9a','#26828e','#35b779','#e8a330','#cf4a4a']
    fig, axes = plt.subplots(1,3,figsize=(15,5.4),sharex=True,sharey=True,layout='constrained')
    for ax, run, mode, label in zip(axes,[runs[0],runs[1],runs[1]],['mup','mup','sp'],
                                  ['μP before correction','μP with corrected gradient','Standard control (rerun)']):
        ax.set_title(label,fontweight='bold')
        for index,(width,color) in enumerate(zip(widths,colors)):
            group = sorted([r for r in run['rows'] if r['mode']==mode and r['width']==width],key=lambda r:r['base_learning_rate'])
            ax.plot([math.log2(r['base_learning_rate']) for r in group],
                    [r['final_validation_loss'] if r['status']=='completed' else float('nan') for r in group],
                    'o-',color=color,ms=4,label=f'{width} ({group[0]["parameters"]/1e6:.1f}M)')
            divergent=[math.log2(r['base_learning_rate']) for r in group if r['status']=='diverged']
            ax.plot(divergent,[.97-.035*index]*len(divergent),'x',color=color,ms=5,transform=ax.get_xaxis_transform())
            valid=[r for r in group if r['status']=='completed']
            best=min(valid,key=lambda r:r['final_validation_loss'])
            ax.plot(math.log2(best['base_learning_rate']),best['final_validation_loss'],'o',ms=9,mfc='none',mec=color,mew=1.5)
        ax.grid(alpha=.18); ax.set_xlabel('log₂ base learning rate'); ax.set_xticks(range(-16,1,2))
    axes[0].set_ylabel('Final validation loss (lower is better)')
    handles, labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,title='Width (parameters)',loc='outside lower center',ncol=5,frameon=False)
    title='FineWeb: correcting the μP output gradient\nConstant LR · decay 0 · 500 updates / 2.048M tokens · seed 42'
    finite=[r['final_validation_loss'] for run in runs for r in run['rows'] if r['status']=='completed']
    for ax in axes: ax.set_ylim(min(finite)-.08,8)
    fig.suptitle(title+'\nLarge rings mark sampled minima; losses above 8 clipped; × marks divergence',fontsize=12)
    fig.savefig(output/'width-gradient-fix-comparison.png',dpi=180,bbox_inches='tight',pad_inches=.1)
    for ax in axes: ax.set_yscale('log'); ax.set_ylim(min(finite)*.95,max(finite)*1.1)
    fig.suptitle(title+'\nFull finite loss range (log scale); × marks divergence',fontsize=12)
    fig.savefig(output/'width-gradient-fix-comparison-full.png',dpi=180,bbox_inches='tight',pad_inches=.1)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for run,mode,label,color,style in [(runs[0],'mup','μP before','#b75730','--'),
            (runs[1],'mup','μP corrected','#227c9d','-'),(runs[1],'sp','SP rerun','#555555','-')]:
        group=sorted([r for r in run['rows'] if r['mode']==mode and r['width']==768],key=lambda r:r['base_learning_rate'])
        axes[0].plot([math.log2(r['base_learning_rate']) for r in group],
            [r['final_validation_loss'] if r['status']=='completed' else float('nan') for r in group],
            marker='o',ls=style,color=color,label=label,ms=4)
    axes[0].set_title('124M: validation loss versus LR',fontweight='bold')
    axes[0].set_ylim(min(finite)-.08,7.5); axes[0].set_xlim(-15,-3)
    axes[0].set_xlabel('log₂ base learning rate'); axes[0].set_ylabel('Validation loss')
    axes[0].legend(frameon=False)
    for version,label,color,style in [('before','Before','#b75730','--'),('fixed','Corrected','#227c9d','-')]:
        axes[1].plot(range(len(widths)),[r[version] for r in gaps],marker='o',ls=style,color=color,label=label)
    axes[1].axhline(0,color='#555555',lw=1)
    axes[1].set_xticks(range(len(widths)),widths); axes[1].set_xlabel('Model width')
    axes[1].set_ylabel('Best μP loss − best SP loss')
    axes[1].set_title('Loss gap after tuning LR separately',fontweight='bold'); axes[1].legend(frameon=False)
    for ax in axes: ax.grid(alpha=.18)
    fig.suptitle(title+'\nTarget panel zooms near the minima; positive gap means μP has higher loss',fontsize=12)
    fig.savefig(output/'target-gradient-fix-comparison.png',dpi=180,bbox_inches='tight',pad_inches=.1)
    plt.close(fig)
    print(json.dumps({'minima':minima,'gaps':gaps},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before',type=Path,required=True)
    parser.add_argument('--fixed',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    compare(args.before,args.fixed,args.output)
