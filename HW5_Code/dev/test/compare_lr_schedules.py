"""Compare measured equal-token LR sweeps across schedule/decay recipes."""
from pathlib import Path
import argparse
import json
import math
import os


def flag(command, name, default=None):
    return command[command.index(name) + 1] if name in command else default


def load_run(path):
    summary = json.loads((path / 'summary.json').read_text())
    if summary['status'] != 'completed':
        raise ValueError(f'Incomplete sweep: {path}')
    rows = summary['records']
    keys = [(r['mode'], r['width'], r['base_learning_rate']) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f'Duplicate trials: {path}')
    first = rows[0]['command']
    config = {
        'scheduler': flag(first, '-k'), 'warmup_steps': int(flag(first, '-u')),
        'final_lr_fraction': float(flag(first, '-q', '1')),
        'weight_decay': float(flag(first, '-c')),
    }
    for row in rows:
        cmd = row['command']
        if row['tokens'] != 2_048_000 or row.get('observed_steps', row['steps']) != 500:
            raise ValueError(f'Training horizon differs: {path}/{row["name"]}')
        if (flag(cmd, '-k') != config['scheduler'] or int(flag(cmd, '-u')) != config['warmup_steps']
                or float(flag(cmd, '-q', '1')) != config['final_lr_fraction']
                or float(flag(cmd, '-c')) != config['weight_decay']):
            raise ValueError(f'Mixed recipes: {path}')
    return {'path': str(path.resolve()), 'summary': summary, 'rows': rows, 'experiment': config}


def compare(runs, output, implementation_note=''):
    os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/llmc-mpl-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    expected = [
        {'scheduler':'constant', 'warmup_steps':0, 'final_lr_fraction':1.0, 'weight_decay':0.0},
        {'scheduler':'cosine', 'warmup_steps':50, 'final_lr_fraction':0.0, 'weight_decay':0.0},
        {'scheduler':'cosine', 'warmup_steps':50, 'final_lr_fraction':0.0, 'weight_decay':0.1},
    ]
    labels = ['Constant LR / decay 0', 'Cosine + warmup / decay 0', 'Cosine + warmup / decay 0.1']
    if len(runs) == 2:
        expected = [expected[0], expected[2]]
        labels = [labels[0], labels[2]]
    if [run['experiment'] for run in runs] != expected:
        raise ValueError('Expected constant/zero-decay, optional cosine/zero-decay, then cosine/0.1-decay')

    common_flags = ['-e', '-1d', '-1c', '-1h', '-me', '-b', '-t', '-d', '-x',
                    '-v', '-m', '-s', '-h', '-ge', '-r', '-n', '-mb', '-ma', '-sl', '-sg']
    indexed = [{(r['mode'], r['width'], r['base_learning_rate']): r for r in run['rows']} for run in runs]
    for candidate in indexed[1:]:
        if candidate.keys() != indexed[0].keys():
            raise ValueError('Runs must use the same mode/width/LR grid')
        for key, row in candidate.items():
            for name in common_flags:
                if flag(row['command'], name) != flag(indexed[0][key]['command'], name):
                    raise ValueError(f'Comparison control differs: {key}, {name}')
    output.mkdir(parents=True, exist_ok=True)
    widths = sorted({r['width'] for r in runs[0]['rows']})
    colors = ['#6a3d9a', '#26828e', '#35b779', '#e8a330', '#cf4a4a']
    minima, target_comparison = [], []
    for run, label in zip(runs, labels):
        for mode in ['sp', 'mup']:
            for width in widths:
                group = [r for r in run['rows'] if r['mode'] == mode and r['width'] == width]
                valid = [r for r in group if r['status'] == 'completed']
                if not valid:
                    continue
                best = min(valid, key=lambda r: r['final_validation_loss'])
                minima.append({'recipe': label, 'mode': mode, 'width': width,
                               'peak_base_lr': best['base_learning_rate'],
                               'validation_loss': best['final_validation_loss'],
                               'grid_boundary': best['base_learning_rate'] in
                                   [min(r['base_learning_rate'] for r in group), max(r['base_learning_rate'] for r in group)]})
        for row in run['rows']:
            if row['width'] == 768:
                target_comparison.append({'recipe': label, 'mode': row['mode'],
                    'peak_base_lr': row['base_learning_rate'], 'status': row['status'],
                    'validation_loss': row.get('final_validation_loss')})
    report = {'runs': [{'path': run['path'], 'recipe': label, 'experiment': run['experiment'],
                       'completed': sum(r['status'] == 'completed' for r in run['rows']),
                       'diverged': sum(r['status'] == 'diverged' for r in run['rows']),
                       'failed': sum(r['status'] == 'failed' for r in run['rows']),
                       'transfer': run['summary'].get('transfer'),
                       'estimated_compute_cost_usd': run['summary'].get('estimated_compute_cost_usd')}
                      for run, label in zip(runs, labels)],
              'minima': minima, 'target_curves': target_comparison, 'implementation_note': implementation_note,
              'limitations': ['One seed and 2.048M tokens per trial; conclusions concern early training.',
                              'X-axis is peak base LR, not average LR or integrated LR.',
                              'Cosine comparison also introduces 50 warmup updates; it measures both together.',
                              'A grid minimum does not establish a global continuous optimum.']}
    if len(runs) == 2:
        report['limitations'].append('The two-recipe comparison changes schedule and decay together; it does not isolate their individual effects.')
    (output / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, len(runs), figsize=(5 * len(runs), 8), sharex=True, sharey=True, layout='constrained')
    for col, (run, label) in enumerate(zip(runs, labels)):
        axes[0, col].set_title(label, fontweight='bold')
        for row_index, mode in enumerate(['sp', 'mup']):
            ax = axes[row_index, col]
            for color_index, (width, color) in enumerate(zip(widths, colors)):
                group = sorted([r for r in run['rows'] if r['mode'] == mode and r['width'] == width],
                               key=lambda r: r['base_learning_rate'])
                xs = [math.log2(r['base_learning_rate']) for r in group]
                ys = [r['final_validation_loss'] if r['status'] == 'completed' else float('nan') for r in group]
                ax.plot(xs, ys, 'o-', ms=3.5, lw=1.5, color=color,
                        label=f'{width} ({group[0]["parameters"] / 1e6:.1f}M)')
                divergent = [math.log2(r['base_learning_rate']) for r in group if r['status'] == 'diverged']
                ax.plot(divergent, [.97 - .035 * color_index] * len(divergent), 'x', ms=4, color=color,
                        transform=ax.get_xaxis_transform())
                finite = [r for r in group if r['status'] == 'completed']
                if finite:
                    best = min(finite, key=lambda r: r['final_validation_loss'])
                    ax.plot(math.log2(best['base_learning_rate']), best['final_validation_loss'],
                            'o', ms=8, mfc='none', mec=color, mew=1.4)
            ax.grid(alpha=.18)
            ax.set_xticks(range(-16, 1, 2))
            if row_index == 1:
                ax.set_xlabel('log₂ peak base learning rate')
    axes[0, 0].set_ylabel('Standard parametrization\nFinal validation loss')
    axes[1, 0].set_ylabel('μP\nFinal validation loss')
    width_handles, width_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(width_handles, width_labels, title='Width (parameters)', fontsize=9,
               frameon=False, loc='outside lower center', ncol=5)
    title = ('FineWeb: comparing schedule and weight-decay recipes\n'
             '500 updates · 2.048M tokens · 12 layers / 12 heads · seed 42 · circles mark sampled minima')
    if implementation_note:
        title += '\n' + implementation_note
    low = min(r['final_validation_loss'] for run in runs for r in run['rows'] if r['status'] == 'completed') - .08
    for ax in axes.flat:
        ax.set_ylim(low, 8)
    fig.suptitle(title + '\nLosses above 8 are clipped; × in top strip marks divergence', fontsize=12)
    fig.savefig(output / 'schedule-decay-comparison.png', dpi=180)
    fig.savefig(output / 'schedule-decay-comparison.pdf')
    finite_losses = [r['final_validation_loss'] for run in runs for r in run['rows'] if r['status'] == 'completed']
    for ax in axes.flat:
        ax.set_yscale('log')
        ax.set_ylim(min(finite_losses) * .95, max(finite_losses) * 1.1)
    fig.suptitle(title + '\nFull finite loss range (log scale); × in top strip marks divergence', fontsize=12)
    fig.savefig(output / 'schedule-decay-comparison-full.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True, layout='constrained')
    for ax, mode in zip(axes, ['sp', 'mup']):
        for run, label, color in zip(runs, labels, ['#697386', '#277da1', '#d45c2b']):
            group = sorted([r for r in run['rows'] if r['width'] == 768 and r['mode'] == mode],
                           key=lambda r: r['base_learning_rate'])
            xs = [math.log2(r['base_learning_rate']) for r in group]
            ys = [r['final_validation_loss'] if r['status'] == 'completed' else float('nan') for r in group]
            ax.plot(xs, ys, 'o-', color=color, label=label, ms=4)
        ax.set_title('Standard parametrization' if mode == 'sp' else 'μP', fontweight='bold')
        ax.set_xlabel('log₂ peak base learning rate')
        ax.set_xticks(range(-14, -3, 2))
        ax.set_xlim(-14.3, -3.7)
        ax.set_ylim(low, 7.2)
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Final held-out validation loss')
    axes[1].legend(frameon=False, fontsize=9)
    target_title = '124M target: schedule and decay comparison\n2.048M tokens per run · zoom near minima; omitted points remain in full plots'
    if implementation_note:
        target_title += '\n' + implementation_note
    fig.suptitle(target_title, fontsize=12)
    fig.savefig(output / 'target-schedule-decay-comparison.png', dpi=180)
    fig.savefig(output / 'target-schedule-decay-comparison.pdf')
    print(json.dumps({'output': str(output.resolve()), 'minima': minima}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--constant', required=True, type=Path)
    parser.add_argument('--cosine', type=Path, help='Optional zero-decay cosine control to isolate schedule from decay')
    parser.add_argument('--decay', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--implementation-note', default='')
    args = parser.parse_args()
    paths = [args.constant] + ([args.cosine] if args.cosine else []) + [args.decay]
    compare([load_run(path) for path in paths], args.output, args.implementation_note)
