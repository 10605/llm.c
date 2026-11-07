"""Prepare a full-horizon fixed-head muP LR sweep; never launches training.

The batch rule is the exploratory seed-42 fit from the course's small-model
batch study, not a validated large-model law. All trials share the TARGET data
horizon, global batch, schedule and decay timescale.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import struct

ROOT = Path(__file__).resolve().parents[2]
CONTEXT = 2048
FIT_COEFFICIENT = 0.006130498046614271
FIT_EXPONENT = 0.4199436828783816
TARGET_PARAMETERS = (50257 + CONTEXT) * 768 + 12 * (12 * 768**2 + 13 * 768) + 2 * 768
DEFAULT_TOKENS = 20 * TARGET_PARAMETERS


def f32(value):
    return struct.unpack('f', struct.pack('f', value))[0]


def positive_f32(value):
    try:
        result = f32(value)
    except (OverflowError, struct.error) as error:
        raise ValueError('Value must fit a finite positive float32') from error
    if not math.isfinite(result) or result < 2**-126:
        raise ValueError('Value must fit a finite positive normal float32')
    return result


def make_plan(train_pattern, val_pattern, tokenizer, *, tokens=DEFAULT_TOKENS,
              width=96, microbatch=4, batch=None, timescale=.1,
              rates=(.0012, .0024, .0048, .0096, .0192)):
    if tokens <= 0 or tokens > 2**63-1 or width <= 0 or width % 24 or microbatch <= 0:
        raise ValueError('Require a positive int64 token budget, BF16 width divisible by 24, and positive microbatch')
    predicted = CONTEXT * FIT_COEFFICIENT * tokens**FIT_EXPONENT
    # Fixed 8192-token grid: changing microbatch must not silently change the
    # optimization experiment. Override batch explicitly if divisibility needs it.
    batch = batch if batch is not None else max(8192, math.floor(predicted/8192 + .5)*8192)
    if batch <= 0 or batch > 2**31-1 or batch % (microbatch*CONTEXT):
        raise ValueError('Global batch must fit int32 and divide by microbatch * 2048; choose compatible microbatch or explicit batch')
    steps = (tokens-1)//batch + 1
    if steps >= 2**31-1:
        raise ValueError('Training horizon exceeds supported update count')
    tau32 = positive_f32(timescale)
    if tau32*steps < 1:
        raise ValueError('Require positive finite timescale with tau * updates >= 1')
    rates = list(rates)
    if not rates:
        raise ValueError('Require positive finite candidate learning rates')
    rates32 = [positive_f32(lr) for lr in rates]
    if len(set(rates32)) != len(rates):
        raise ValueError('Duplicate candidate learning rates')
    # Fixed held-out token prefix, independent of width/microbatch.
    validation_tokens = 1048576
    if validation_tokens % (microbatch*CONTEXT):
        raise ValueError('Microbatch must also divide the fixed 1,048,576-token validation budget')
    trials = []
    for lr,lr32 in zip(rates,rates32):
        decay = f32(1/(lr32*steps*tau32))
        if not math.isfinite(decay) or decay <= 0:
            raise ValueError('Derived decay must fit a positive float32')
        name = f'mup-c{width}-lr{lr:g}'
        command = ['./train_gpt2cu', '-cp', '1', '-1d', '12', '-1c', str(width),
                   '-1h', '12', '-me', '1', '-mb', '192', '-ma', '4', '-se', '42',
                   '-b', str(microbatch), '-t', str(CONTEXT), '-d', str(batch),
                   '-nt', str(tokens), '-l', repr(lr), '-wt', repr(timescale),
                   '-k', 'linear', '-u', '-1', '-q', '0',
                   '-id', train_pattern, '-iv', val_pattern, '-it', tokenizer,
                   '-h', '0', '-s', '0', '-v', str(steps),
                   '-m', str(validation_tokens//(microbatch*CONTEXT)),
                   '-n', '0', '-o', f'runs/{name}']
        trials.append(dict(name=name, base_lr=lr, base_weight_decay=decay,
                           command=command, shell_command=shlex.join(command)))
    return dict(status='prepared_only_not_run', target_parameters=TARGET_PARAMETERS,
                parameter_count_convention='Unpadded tied vocabulary; 2048 positional embeddings',
                requested_tokens=tokens, actual_tokens=steps*batch, updates=steps,
                warmup_updates=steps//10, context=CONTEXT, proxy_width=width,
                batch_tokens=batch, predicted_batch_tokens=predicted,
                batch_fit=dict(coefficient=FIT_COEFFICIENT, exponent=FIT_EXPONENT,
                               units='2048-token sequences', exploratory=True),
                timescale=timescale, validation_tokens=validation_tokens,
                trials=trials, selected_lr=None,
                selection='Lowest final held-out validation loss; extend grid if the minimum is at a boundary',
                transfer='Keep all recipe flags and the selected base LR; change only width to 768 and output directory',
                source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                               for p in ('train_gpt2.cu', 'llmc/train_config.h')},
                limitations=['Batch fit and tau=0.1 are provisional choices from a different small-model family.',
                             'Candidate LRs are a bracket, not a previously validated optimum for this recipe.',
                             'Same full target horizon and timescale apply to all proxy widths.',
                             'Pin dataset revision and identical data/validation order before launching.',
                             'For ClimbMix at the default target budget, repeated dataset passes are intentional.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-pattern', required=True)
    parser.add_argument('--val-pattern', required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--tokens', type=int, default=DEFAULT_TOKENS)
    parser.add_argument('--width', type=int, default=96)
    parser.add_argument('--microbatch', type=int, default=4)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--timescale', type=float, default=.1)
    parser.add_argument('--lrs', type=float, nargs='+', default=[.0012,.0024,.0048,.0096,.0192])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = make_plan(args.train_pattern,args.val_pattern,args.tokenizer,tokens=args.tokens,
                     width=args.width,microbatch=args.microbatch,batch=args.batch,
                     timescale=args.timescale,rates=args.lrs)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(plan,indent=2)+'\n')
    print(f'Prepared {len(plan["trials"])} trials, {plan["updates"]:,} updates each, '
          f'{plan["batch_tokens"]:,} tokens/update. No training launched. Plan: {args.output}')


if __name__ == '__main__':
    main()
