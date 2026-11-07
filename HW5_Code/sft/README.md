# SFT and single-sequence generation

Run these commands from `HW5_Code/`. This code adapts the supplied HW5 `code.zip`
(SHA-256 `df52c09048cf24b6208aafb842b6fa54ea0bb9331c017a5845da70b3230388c3`).
The original archive and seed are unchanged.

## Environment

```bash
conda create -n hw5-sft --override-channels -c conda-forge python=3.11 pip -y
conda activate hw5-sft
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r sft/requirements.txt
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

The CUDA toolkit used for `train_gpt2cu` is separate. Do not add its private cuDNN
directory to a global `LD_LIBRARY_PATH`; PyTorch provides its own CUDA dependencies.
See the [instructor AMI guide](../doc/mup/aws-ami.md).

## Generate once

```bash
python sft/generate.py --model /opt/hw5/seed/model_00010919.bin \
  --mode completion --prompt "The experiment showed that" \
  --max-new-tokens 256 --temperature 0.6 --seed 42
```

The generator requires one prompt, produces one sequence, and exits. It never
reads terminal input or keeps conversation history. For an SFT checkpoint:

```bash
python sft/generate.py --model runs/sft/model_sft_00100.bin \
  --mode chat --prompt "Explain why the sky is blue." \
  --max-new-tokens 256 --temperature 0.6 --seed 42
```

`--mode chat` selects the instruction-token format; it is still a single response.
Use `--temperature 0` for greedy decoding. Prompt plus requested output must fit
the checkpoint context (2048 for the seed); overlong requests give an error.
An end token can stop generation earlier than the maximum. `--mode auto` infers
SFT from nonzero chat-token embeddings; explicit modes are preferable for comparisons.

## SFT

```bash
python sft/chat_sft.py --model runs/expert/model_00021838.bin \
  --out-dir runs/sft --device cuda \
  --num-iterations 100 --num-train-convs 10000 \
  --eval-every 20 --sample-every 0
```

SFT starts a fresh AdamW optimizer. It does not load pretraining optimizer state.
The SFT learning rate defaults to `0.0001`; no LR choice is needed for this exercise.
This example uses 10,000 conversations from SmolTalk for a short 100-update
exercise. Defaults are microbatch 16 and 65,536 tokens/update. SFT sequence
length defaults to 2048 and cannot exceed the checkpoint context. The total batch
size must be divisible by microbatch times sequence length.

At each sampling checkpoint, SFT generates one sequence from `--sample-prompt`
(default: `Explain why the sky is blue.`). `--sample-tokens` controls the maximum
output length. `--sample-every 0` disables all sampling. `--compile 0` is useful
for short checks. To use local data, supply both `--train-file` and `--val-file`;
each JSONL row contains a `messages` list with `role` and `content` fields.

The output model is an llm.c weights file with logs, CLI arguments and saved model
configuration beside it. SFT optimizer state for exact pause/resume is not saved.

## Compatibility and checks

The loader reads `MUP1` checkpoint metadata automatically. The forward pass uses
the saved attention multiplier/head dimension and inverse width scaling before
the output projection. SFT scales the four hidden matrix types' LR and decay
according to the CUDA trainer's muP rule. The writer retains muP metadata and
other header padding. Legacy checkpoints with no metadata remain standard
parametrization; no seed LR is embedded in the code.

```bash
python sft/test_sft.py --checkpoint /opt/hw5/seed/model_00010919.bin -v
```

CPU checks cover independent logits/gradients, checkpoint round trips, optimizer
groups, one-prompt generation/context errors, and two SFT updates followed by
save/load/generation. With `--checkpoint`, the actual seed must agree with the
independent forward calculation and survive a byte-identical BF16 round trip.
The protected seed is never modified. AMI-specific GPU acceptance results are
recorded under `/opt/hw5/release`; `plot_logs.py runs/sft` reads its training logs.
