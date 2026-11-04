"""Prepare, but never execute, an SP versus muP learning-rate transfer sweep."""
from pathlib import Path
import argparse
import json
import math
import shlex


def parameter_count(width, layers=12):
    # Matches allocated CUDA tensors, including padded tied vocabulary.
    return 50304 * width + 1024 * width + layers * (12 * width * width + 13 * width) + 2 * width


def make_plan(train_pattern, val_pattern, tokenizer, steps=500, total_batch=4096,
              min_lr_exponent=-16, max_lr_exponent=-7, *, scheduler='constant',
              warmup_steps=0, final_lr_fraction=1.0, weight_decay=0.0):
    if steps <= 0 or total_batch <= 0 or total_batch % (4 * 256):
        raise ValueError('Steps must be positive; total batch must be a positive multiple of 1024')
    widths = [48, 96, 192, 384, 768]
    if not -30 <= min_lr_exponent <= max_lr_exponent <= 4:
        raise ValueError('LR exponent range must be ordered and within [-30, 4]')
    if scheduler not in {'constant', 'cosine', 'linear', 'wsd'}:
        raise ValueError('Unknown LR scheduler')
    if not 0 <= warmup_steps < steps or not 0 <= final_lr_fraction <= 1:
        raise ValueError('Warmup must be shorter than training; final LR fraction must be within [0, 1]')
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError('Weight decay must be finite and nonnegative')
    if scheduler == 'constant' and (warmup_steps or final_lr_fraction != 1):
        raise ValueError('Constant schedule ignores warmup and final fraction; use 0 and 1 respectively')
    experiment = {'scheduler': scheduler, 'warmup_steps': warmup_steps,
                  'final_lr_fraction': final_lr_fraction, 'weight_decay': weight_decay}
    rates = [2.0 ** exponent for exponent in range(min_lr_exponent, max_lr_exponent + 1)]
    trials = []
    for mode in ['sp', 'mup']:
        for width in widths:
            for lr in rates:
                name = f'{mode}-c{width}-lr2pow{round(math.log2(lr))}'
                command = ['./train_gpt2cu', '-e', 'd12', '-1d', '12', '-1c', str(width), '-1h', '12',
                           '-me', str(int(mode == 'mup')), '-id', train_pattern, '-iv', val_pattern,
                           '-it', tokenizer, '-b', '4', '-t', '256', '-d', str(total_batch),
                           '-x', str(steps), '-k', scheduler, '-l', repr(lr), '-u', str(warmup_steps),
                           '-q', str(final_lr_fraction), '-c', str(weight_decay),
                           '-v', str(max(1, steps//10)), '-m', '32', '-s', '0', '-h', '0',
                           '-ge', '0', '-r', '0', '-o', f'lr-transfer/{name}', '-n', '0']
                if mode == 'mup':
                    command += ['-mb', '192', '-ma', '4']
                trials.append({'name': name, 'mode': mode, 'width': width, 'parameters': parameter_count(width),
                               'base_learning_rate': lr, 'matrix_learning_rate': lr/(width/192) if mode == 'mup' else lr,
                               'tokens': steps*total_batch, 'command': command, 'shell_command': shlex.join(command)})
    return {'status': 'prepared_only_not_run', 'reference_width': 192, 'reference_heads': 12,
            'attention_multiplier': 4, 'layers': 12, 'heads': 12, 'widths': widths,
            'training_steps': steps, 'tokens_per_trial': steps*total_batch,
            'experiment': experiment,
            'min_lr_exponent': min_lr_exponent, 'max_lr_exponent': max_lr_exponent,
            'initialization_seed': 42, 'seed_note': 'The current trainer uses fixed initialization seed 42; repeated-seed confirmation needs additional seed support or explicit weight fixtures.',
            'primary_metric': 'validation loss at equal final token count',
            'reference_figure_metric': 'mean of last 20 training losses',
            'selection': 'Choose LR from the width-48 muP validation curve before running width-768 trials; also freeze width-96 and width-192 selections as secondary transfer checks.',
            'diagnostics': 'Report target loss regret relative to the best measured target LR; compare shifts of SP and muP minima on the log2 LR grid. Boundary minima or divergent curves require a revised grid.',
            'trials': trials}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-pattern', required=True)
    parser.add_argument('--val-pattern', required=True)
    parser.add_argument('--tokenizer', default='gpt2_tokenizer.bin')
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--total-batch', type=int, default=4096)
    parser.add_argument('--min-lr-exponent', type=int, default=-16)
    parser.add_argument('--max-lr-exponent', type=int, default=-7)
    parser.add_argument('--scheduler', choices=['constant', 'cosine', 'linear', 'wsd'], default='constant')
    parser.add_argument('--warmup-steps', type=int, default=0)
    parser.add_argument('--final-lr-fraction', type=float, default=1.0)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = make_plan(args.train_pattern, args.val_pattern, args.tokenizer, args.steps, args.total_batch,
                     args.min_lr_exponent, args.max_lr_exponent, scheduler=args.scheduler,
                     warmup_steps=args.warmup_steps, final_lr_fraction=args.final_lr_fraction,
                     weight_decay=args.weight_decay)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + '\n')
    print(f'Prepared {len(plan["trials"])} trials in {args.output}. No training or Modal jobs were launched.')
