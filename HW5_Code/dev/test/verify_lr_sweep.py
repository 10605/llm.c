"""Audit actual equal-token sweep logs, schedule values, and comparison provenance."""
from pathlib import Path
import argparse
import json
import math
import re
from compare_lr_schedules import flag, load_run


def audit(directory, baseline, expected_source_change=None):
    run = load_run(directory)
    prior = load_run(baseline)
    expected_keys = {(r['mode'], r['width'], r['base_learning_rate']) for r in prior['rows']}
    actual_keys = {(r['mode'], r['width'], r['base_learning_rate']) for r in run['rows']}
    assert actual_keys == expected_keys, 'Different LR/width grid'
    assert not any(r['status'] == 'failed' for r in run['rows']), 'Infrastructure or incomplete-trial failure'
    old_source = json.loads((baseline / 'source_manifest.json').read_text())['files']
    source = json.loads((directory / 'source_manifest.json').read_text())['files']
    core = [p for p in old_source if p.startswith('llmc/') or p in ['train_gpt2.cu', 'Makefile']]
    changed = {p for p in core if source[p] != old_source[p]}
    declaration = json.loads(expected_source_change.read_text()) if expected_source_change else {}
    expected_changes = declaration.get('files', {})
    assert changed == set(expected_changes), 'Unexpected changes to the training core'
    for name, hashes in expected_changes.items():
        assert hashes['before'] == old_source[name] and hashes['after'] == source[name], 'Source-change hash mismatch'
    ancestor_reports = []
    # A combined baseline may contain trials made before its newest manifest.
    # Audit declared historical metadata-only changes separately from this fix.
    for run_id, declared in declaration.get('baseline_ancestors', {}).items():
        assert Path(run_id).name == run_id
        ancestor = baseline.parent/run_id
        ancestry = json.loads((baseline/'prior_run.json').read_text())
        assert ancestry['run'] == run_id
        ancestor_source = json.loads((ancestor/'source_manifest.json').read_text())['files']
        changed_ancestor = {p for p in core if ancestor_source[p] != old_source[p]}
        assert changed_ancestor == set(declared['files']), 'Unexpected historical source difference'
        for name, hashes in declared['files'].items():
            assert hashes['before'] == ancestor_source[name] and hashes['after'] == old_source[name]
        ancestor_rows = load_run(ancestor)['rows']
        prior_by_name = {r['name']:r for r in prior['rows']}
        assert len(ancestor_rows) == ancestry['trials']
        assert all(prior_by_name[r['name']] == r for r in ancestor_rows), 'Historical trials were altered'
        ancestor_reports.append(dict(run=run_id, preserved_trials=len(ancestor_rows), **declared))
    expected_data = json.loads(next(baseline.glob('*/dataset.json')).read_text())
    datasets = list(directory.glob('*/dataset.json'))
    assert len(datasets) == 8, 'Expected all eight phase/worker provenance records'
    assert all(json.loads(p.read_text()) == expected_data for p in datasets), 'Different data or tokenizer'
    histories = {}
    max_lr_error = 0
    experiment = run['experiment']
    for row in run['rows']:
        paths = list(directory.glob(f'*/{row["name"]}/main.log'))
        assert len(paths) == 1, (row['name'], 'Missing/duplicate log')
        log = paths[0].read_text()
        training = re.findall(r'^s:(\d+) trl:([^ ]+) lr:([^ ]+) norm:([^ ]+) time \(ms\):([^\n]+)', log, re.MULTILINE)
        validation = re.findall(r'^s:(\d+) tel:([^\n]+)', log, re.MULTILINE)
        assert len(training) == 500 and [int(x[0]) for x in training] == list(range(500)), row['name']
        assert validation and int(validation[-1][0]) == 500, row['name']
        actual_lr = [float(x[2]) for x in training]
        peak = row['base_learning_rate']
        expected_lr = []
        for step in range(500):
            if experiment['scheduler'] == 'constant':
                factor = 1.0
            elif experiment['scheduler'] == 'cosine':
                warmup = experiment['warmup_steps']
                if step < warmup:
                    factor = (step + 1) / warmup
                else:
                    factor = experiment['final_lr_fraction'] + (1 - experiment['final_lr_fraction']) * .5 * (
                        1 + math.cos(math.pi * (step - warmup) / (500 - warmup)))
            else:
                raise ValueError('Audit currently supports constant and cosine schedules')
            expected_lr.append(peak * factor)
        error = max(abs(a - b) for a, b in zip(actual_lr, expected_lr))
        # Trainer logs six decimal places; allow float arithmetic plus print rounding.
        assert error < 8e-7, (row['name'], error)
        max_lr_error = max(error, max_lr_error)
        if peak in histories:
            assert histories[peak] == actual_lr, 'Schedule differs across widths or parametrizations'
        histories[peak] = actual_lr
        params = (paths[0].parent / 'run_parameters.txt').read_text()
        setting_line = next(line for line in params.splitlines() if line.startswith('scheduler='))
        actual = dict(re.findall(r'(\w+)=([^ ]+)', setting_line))
        assert actual['scheduler'] == experiment['scheduler']
        assert int(actual['warmup']) == experiment['warmup_steps']
        assert abs(float(actual['weight_decay']) - experiment['weight_decay']) < 1e-7
        assert float(actual['final_fraction']) == experiment['final_lr_fraction']
        assert 'stop_step=500 schedule_steps=500 early_stop=-1 resume=0' in params
        assert 'batch=4 sequence=256 total_batch=4096 accumulation=4' in params
        assert flag(row['command'], '-h') == '0'
        if row['mode'] == 'mup':
            assert flag(row['command'], '-mb') == '192' and flag(row['command'], '-ma') == '4'
    report = {'status':'passed', 'run':str(directory.resolve()), 'baseline':str(baseline.resolve()),
              'experiment':experiment, 'verified_trials':len(run['rows']), 'updates_per_trial':500,
              'tokens_per_trial':2_048_000, 'identical_base_lr_histories_across_widths':len(histories),
              'max_logged_lr_absolute_error':max_lr_error, 'matching_training_core_files':len(core)-len(changed),
              'expected_source_changes':expected_changes,
              'verified_baseline_ancestors':ancestor_reports,
              'matching_worker_dataset_records':len(datasets), 'dataset_revision':expected_data['revision']}
    (directory / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--expected-source-change', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.results, args.baseline, args.expected_source_change), indent=2))
