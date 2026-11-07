"""
Chat supervised fine-tuning (SFT) of an llm.c GPT-2 checkpoint, adapted from
nanochat's scripts/chat_sft.py.

Reads a model .bin written by train_gpt2cu, trains on SmolTalk conversations with
the loss restricted to assistant tokens, and writes the result back as an llm.c .bin
plus a main.log / run_parameters.txt that vislog.ipynb can plot.

    python sft/chat_sft.py --model runs/expert/model_00021838.bin \
        --out-dir runs/sft --lr 0.0001 --num-iterations 100
"""

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict

import numpy as np
import torch
from datasets import Features, Sequence, Value, load_dataset

from chat_tokenizer import ChatTokenizer
from gpt2 import adamw_groups, load_llmc, save_llmc

parser = argparse.ArgumentParser(description="Chat SFT for llm.c GPT-2 checkpoints")
parser.add_argument("--model", type=str, required=True, help="llm.c model .bin to start from")
parser.add_argument("--out-dir", type=str, default="logSFT")
parser.add_argument("--device", type=str, default="", help="cuda|mps|cpu (empty = autodetect)")
parser.add_argument("--compile", type=int, default=1, help="torch.compile the model (cuda only)")
# data
parser.add_argument("--dataset", type=str, default="HuggingFaceTB/smol-smoltalk")
parser.add_argument("--train-file", help="optional local JSON/JSONL training conversations (messages field)")
parser.add_argument("--val-file", help="local validation conversations; required with --train-file")
parser.add_argument("--data-workers", type=int, default=min(4, os.cpu_count() or 1))
parser.add_argument("--num-train-convs", type=int, default=100000, help="use only the first N training conversations (-1 = all)")
parser.add_argument("--max-seq-len", type=int, default=2048)
parser.add_argument("--long-convs", type=str, default="drop", choices=["drop", "truncate"],
                    help="conversations longer than max-seq-len+1 tokens: drop them, or truncate (loses <|assistant_end|>)")
# batch sizes
parser.add_argument("--device-batch-size", type=int, default=16)
parser.add_argument("--total-batch-size", type=int, default=65536, help="tokens per optimizer step")
# optimization
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight-decay", type=float, default=0.0)
parser.add_argument("--grad-clip", type=float, default=1.0)
parser.add_argument("--warmup-ratio", type=float, default=0.05)
parser.add_argument("--warmdown-ratio", type=float, default=0.5)
parser.add_argument("--final-lr-frac", type=float, default=0.0)
parser.add_argument("--num-iterations", type=int, default=-1, help="optimizer steps (-1 = one epoch)")
parser.add_argument("--special-init", type=str, default="mean", choices=["mean", "zero"],
                    help="init of the chat-token embedding rows (they are all zero after pretraining)")
# evaluation
parser.add_argument("--eval-every", type=int, default=100)
parser.add_argument("--eval-rows", type=int, default=256, help="packed val rows per evaluation")
parser.add_argument("--sample-every", type=int, default=200)
parser.add_argument("--sample-prompt", default="Explain why the sky is blue.")
parser.add_argument("--sample-tokens", type=int, default=128)
parser.add_argument("--dollars-per-hour", type=float, default=1.006, help="for the cost estimate printout")
parser.add_argument("--seed", type=int, default=1337)
args = parser.parse_args()
if bool(args.train_file) != bool(args.val_file):
    parser.error("--train-file and --val-file must be supplied together")
if min(args.data_workers, args.device_batch_size, args.total_batch_size, args.max_seq_len,
       args.eval_every, args.eval_rows, args.sample_tokens) <= 0 or args.sample_every < 0:
    parser.error("batch/data/evaluation sizes must be positive; sample-every must be nonnegative")

# -----------------------------------------------------------------------------
# Device

if args.device:
    device = args.device
elif torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
device_type = "cuda" if device.startswith("cuda") else device
autocast = torch.autocast(device_type, dtype=torch.bfloat16) if device_type in ("cuda", "mps") else torch.autocast("cpu", enabled=False)
synchronize = torch.cuda.synchronize if device_type == "cuda" else (torch.mps.synchronize if device_type == "mps" else lambda: None)
torch.manual_seed(args.seed)
random.seed(args.seed)
if device_type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device_type == "cuda" else ""))

# -----------------------------------------------------------------------------
# Model and tokenizer

tokenizer = ChatTokenizer()
model = load_llmc(args.model)
cfg = model.config
assert tokenizer.n_vocab <= cfg.padded_vocab_size, "not enough padded rows for the chat tokens"
assert args.max_seq_len <= cfg.max_seq_len

# The padded rows of wte are zero after pretraining (they never receive gradient), so
# every chat token would start out with an identical input embedding. "mean" instead
# starts each one at the average real-token embedding plus noise of the same scale.
with torch.no_grad():
    real = model.wte.weight[: cfg.vocab_size]
    new = model.wte.weight[cfg.vocab_size: tokenizer.n_vocab]
    if args.special_init == "mean":
        new.copy_(real.mean(0) + torch.randn_like(new) * real.std(0))
    else:
        new.zero_()
model.n_active = tokenizer.n_vocab
model.to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"loaded {args.model}: L={cfg.n_layer} C={cfg.n_embd} H={cfg.n_head} params={num_params:,}")
print(f"muP={cfg.use_mup}, width multiplier={cfg.mup_width_mult}, attention multiplier={cfg.mup_base_attn_mult}")

orig_model = model
if args.compile and device_type == "cuda":
    model = torch.compile(model)

# -----------------------------------------------------------------------------
# Data: render every conversation to (ids, mask) up front

_tok = None


def render(example):
    global _tok
    if _tok is None:
        _tok = ChatTokenizer()
    max_tokens = args.max_seq_len + 1 if args.long_convs == "truncate" else 10**9
    ids, mask = _tok.render_conversation(example["messages"], max_tokens=max_tokens)
    return {"ids": ids, "mask": mask}


class Conversations:
    """All rendered conversations stored as flat arrays (python lists of ints would need ~10GB of RAM)."""

    def __init__(self, ds):
        ids_col = ds.data.column("ids").combine_chunks()
        mask_col = ds.data.column("mask").combine_chunks()
        self.ids = ids_col.values.to_numpy(zero_copy_only=False).astype(np.uint16)
        self.mask = mask_col.values.to_numpy(zero_copy_only=False).astype(np.bool_)
        self.offsets = ids_col.offsets.to_numpy().astype(np.int64)

    def __len__(self):
        return len(self.offsets) - 1

    def length(self, i):
        return int(self.offsets[i + 1] - self.offsets[i])

    def get(self, i):
        a, b = self.offsets[i], self.offsets[i + 1]
        return self.ids[a:b], self.mask[a:b]


def load_split(split, limit):
    """The first `limit` conversations of the split that fit in one row (all of them in truncate mode)."""
    ds = (load_dataset("json", data_files={"train": args.train_file, "test": args.val_file}, split=split)
          if args.train_file else load_dataset(args.dataset, split=split))
    if limit > 0 and args.long_convs == "truncate":
        ds = ds.select(range(min(limit, len(ds))))
    features = Features({"ids": Sequence(Value("uint16")), "mask": Sequence(Value("bool"))})
    num_proc = args.data_workers if args.data_workers > 1 else None
    ds = ds.map(render, remove_columns=ds.column_names, features=features,
                num_proc=num_proc, desc=f"rendering {split}")
    if args.long_convs == "drop":
        n = len(ds)
        ds = ds.filter(lambda ids: len(ids) <= args.max_seq_len + 1, input_columns="ids", num_proc=num_proc)
        print(f"{split}: kept {len(ds):,} of {n:,} conversations that fit in {args.max_seq_len + 1} tokens")
        if limit > 0:
            ds = ds.select(range(min(limit, len(ds))))
    # Prompt-only rows have no supervised targets, which would produce a NaN mean loss.
    ds = ds.filter(lambda mask: any(mask[1:]), input_columns="mask", num_proc=num_proc)
    if len(ds) == 0:
        raise ValueError(f"{split}: no conversations with assistant targets fit the sequence length")
    return Conversations(ds.flatten_indices())  # filter/select only record indices; ds.data is the full table


t0 = time.time()
train_convs = load_split("train", args.num_train_convs)
val_convs = load_split("test", 4 * args.eval_rows)
n_train_tokens = len(train_convs.ids)
print(f"train: {len(train_convs):,} conversations, {n_train_tokens:,} tokens "
      f"({train_convs.mask.mean():.0%} assistant), val: {len(val_convs):,} conversations, "
      f"data prep {time.time() - t0:.0f}s")


def packed_batches(convs, batch_size, shuffle, buffer_size=100):
    """
    nanochat-style best-fit packing. Each row starts at a conversation boundary; we
    keep adding the longest buffered conversation that still fits, and pad the rest
    of the row (masked out) once nothing fits. Yields (inputs, targets, epoch_progress).
    """
    row_capacity = args.max_seq_len + 1
    order = list(range(len(convs)))
    if shuffle:
        random.shuffle(order)
    cursor, consumed, buffer = 0, 0, []  # buffer holds (conversation index, length)
    while True:
        rows = np.full((batch_size, row_capacity), tokenizer.bos, dtype=np.int64)
        masks = np.zeros((batch_size, row_capacity), dtype=np.bool_)
        for r in range(batch_size):
            pos = 0
            while pos < row_capacity:
                while len(buffer) < buffer_size:
                    i = order[cursor % len(order)]
                    buffer.append((i, convs.length(i)))
                    cursor += 1
                remaining = row_capacity - pos
                best = max((j for j in range(len(buffer)) if buffer[j][1] <= remaining),
                           key=lambda j: buffer[j][1], default=-1)
                if best < 0:
                    break  # nothing fits: the rest of the row stays as masked-out padding
                i, n = buffer.pop(best)
                ids, mask = convs.get(i)
                rows[r, pos:pos + n] = ids
                masks[r, pos:pos + n] = mask
                pos += n
                consumed += 1
        batch = torch.from_numpy(rows)
        mask = torch.from_numpy(masks)
        inputs = batch[:, :-1]
        targets = batch[:, 1:].clone()
        targets[~mask[:, 1:]] = -1  # only train on assistant tokens
        yield inputs.to(device), targets.to(device), consumed / len(convs)


# a fixed set of validation rows so every evaluation sees the same data
_val_iter = packed_batches(val_convs, args.device_batch_size, shuffle=False)
val_batches = [next(_val_iter)[:2] for _ in range(max(1, args.eval_rows // args.device_batch_size))]


@torch.no_grad()
def evaluate():
    model.eval()
    total, count = 0.0, 0
    for x, y in val_batches:
        with autocast:
            loss = model(x, y, loss_reduction="sum")
        total += loss.item()
        count += (y >= 0).sum().item()
    model.train()
    return total / count


def sample(step, f):
    orig_model.eval()
    f.write(f"===== step {step} =====\n")
    prompt = args.sample_prompt
    ids = tokenizer.render_for_completion([{"role": "user", "content": prompt}])
    remaining = cfg.max_seq_len - len(ids)
    if remaining <= 0:
        raise ValueError("sample prompt exceeds the checkpoint context")
    with autocast:
        out = orig_model.generate(ids, max_new_tokens=min(args.sample_tokens, remaining),
                                  stop_token=tokenizer.special("<|assistant_end|>"),
                                  temperature=0.6, top_k=50)
    reply = tokenizer.decode([t for t in out if t < tokenizer.base_vocab_size])
    f.write(f"USER: {prompt}\nASSISTANT: {reply}\n\n")
    print(f"  USER: {prompt}\n  ASSISTANT: {reply[:200]!r}")
    f.flush()
    orig_model.train()


# -----------------------------------------------------------------------------
# Optimizer and schedule

tokens_per_micro = args.device_batch_size * args.max_seq_len
assert args.total_batch_size % tokens_per_micro == 0, "total batch size must be a multiple of B*T"
grad_accum = args.total_batch_size // tokens_per_micro
optimizer = torch.optim.AdamW(
    adamw_groups(orig_model, args.lr, args.weight_decay),
    lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=(device_type == "cuda"),
)
approx_steps = args.num_iterations if args.num_iterations > 0 else math.ceil(n_train_tokens / args.total_batch_size)
print(f"grad accum {grad_accum}, ~{approx_steps} steps for "
      f"{'the requested iterations' if args.num_iterations > 0 else 'one epoch'}")


def lr_multiplier(progress):
    if progress < args.warmup_ratio:
        return (progress + 1e-8) / args.warmup_ratio
    if progress <= 1.0 - args.warmdown_ratio:
        return 1.0
    frac = (progress - (1.0 - args.warmdown_ratio)) / args.warmdown_ratio
    return (1 - frac) + frac * args.final_lr_frac


# -----------------------------------------------------------------------------
# Logging in llm.c's main.log format so vislog.ipynb works unchanged

os.makedirs(args.out_dir, exist_ok=True)
logfile = open(os.path.join(args.out_dir, "main.log"), "w")
samplefile = open(os.path.join(args.out_dir, "samples.txt"), "w")
with open(os.path.join(args.out_dir, "run_parameters.txt"), "w") as f:
    rows = {"init model": args.model, "dataset": args.dataset, "learning rate (LR)": f"{args.lr:e}",
            "total batch size": args.total_batch_size, "micro batch size B": args.device_batch_size,
            "sequence length T": args.max_seq_len, "long conversations": args.long_convs,
            "special token init": args.special_init,
            "num_parameters": num_params, "train conversations": len(train_convs)}
    for k, v in rows.items():
        f.write(f"| {k:<21} | {str(v):<50} |\n")
with open(os.path.join(args.out_dir, "args.json"), "w") as f:
    json.dump(vars(args), f, indent=2)
with open(os.path.join(args.out_dir, "model_config.json"), "w") as f:
    json.dump(asdict(cfg), f, indent=2)

# -----------------------------------------------------------------------------
# Training loop

train_iter = packed_batches(train_convs, args.device_batch_size, shuffle=True)
x, y, progress = next(train_iter)
step, train_time, min_val = 0, 0.0, float("inf")
model.train()
while True:
    if args.num_iterations > 0:
        progress = step / args.num_iterations
    last_step = progress >= 1.0

    if last_step or step % args.eval_every == 0:
        val_loss = evaluate()
        min_val = min(min_val, val_loss)
        print(f"step {step:5d} | val loss {val_loss:.4f}")
        logfile.write(f"s:{step} tel:{val_loss:.4f}\n")
        logfile.flush()
    if args.sample_every > 0 and (last_step or step % args.sample_every == 0):
        sample(step, samplefile)
    if last_step:
        break

    synchronize()
    t0 = time.time()
    lrm = lr_multiplier(progress)
    for group in optimizer.param_groups:
        group["lr"] = args.lr * lrm * group["lr_scale"]
    train_loss = 0.0
    for _ in range(grad_accum):
        with autocast:
            loss = model(x, y)
        (loss / grad_accum).backward()
        train_loss += loss.item() / grad_accum
        x, y, data_progress = next(train_iter)
        if args.num_iterations <= 0:
            progress = max(progress, data_progress)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    synchronize()
    dt = time.time() - t0
    step += 1
    if step > 3:  # skip compile / warmup steps
        train_time += dt
    tok_per_sec = args.total_batch_size / dt
    print(f"step {step:5d} ({100 * progress:5.1f}%) | loss {train_loss:.4f} | lr {args.lr * lrm:.2e} | "
          f"norm {norm:.2f} | {dt * 1000:.0f}ms | {tok_per_sec:,.0f} tok/s")
    logfile.write(f"s:{step} trl:{train_loss:.4f} lr:{args.lr * lrm:.6f} norm:{norm:.2f} time (ms):{dt * 1000:.2f}\n")

hours = train_time / 3600
print(f"done: {step} steps, train time {train_time / 60:.1f} min, min val loss {min_val:.4f}, "
      f"est. cost ${hours * args.dollars_per_hour:.2f} at ${args.dollars_per_hour}/hr")
if device_type == "cuda":
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
out_path = os.path.join(args.out_dir, f"model_sft_{step:05d}.bin")
save_llmc(orig_model, out_path)
print(f"saved {out_path}")
logfile.close()
samplefile.close()
