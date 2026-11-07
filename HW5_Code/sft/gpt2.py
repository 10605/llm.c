"""
Minimal PyTorch GPT-2 that reads and writes llm.c model checkpoints (.bin).

llm.c checkpoint layout (see gpt2_write_to_checkpoint in train_gpt2.cu):
    int32 header[256]: magic=20240326, version (3=fp32, 5=bf16), maxT, V, L, NH, C, Vp
    followed by 16 parameter tensors, concatenated in the order of PARAM_SHAPES below.
Header slots 8..13 hold the optional MUP1 parametrization extension. Loading and
saving preserve it; legacy zero-filled headers retain standard parametrization.
The token embedding wte is tied with the output head, and has Vp >= V rows. The
padded rows (V..Vp-1) are never used in pretraining, so we use them for chat tokens.
"""

from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LLMC_MAGIC = 20240326
LLMC_VERSION_FP32 = 3
LLMC_VERSION_BF16 = 5
MUP_MAGIC = 0x4D555031  # MUP1, matching llmc/train_config.h


@dataclass
class GPTConfig:
    max_seq_len: int = 1024
    vocab_size: int = 50257
    padded_vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    use_mup: bool = False
    mup_width_mult: float = 1.0
    mup_base_attn_mult: float = 1.0
    mup_reference_width: float = 0.0

    def __post_init__(self):
        if min(self.max_seq_len, self.vocab_size, self.n_layer, self.n_head, self.n_embd) <= 0:
            raise ValueError("model dimensions must be positive")
        if self.n_embd % self.n_head or self.padded_vocab_size < self.vocab_size:
            raise ValueError("invalid head dimensions or padded vocabulary")
        if not all(math.isfinite(v) and v > 0 for v in
                   (self.mup_width_mult, self.mup_base_attn_mult)):
            raise ValueError("invalid muP multipliers")
        if self.mup_reference_width == 0:
            self.mup_reference_width = self.n_embd / self.mup_width_mult
        if not math.isfinite(self.mup_reference_width) or self.mup_reference_width <= 0:
            raise ValueError("invalid muP reference width")


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        C = config.n_embd
        self.n_head = config.n_head
        head_size = C // self.n_head
        self.attention_scale = (config.mup_base_attn_mult / head_size if config.use_mup
                                else 1.0 / math.sqrt(head_size))
        self.ln_1 = nn.LayerNorm(C, eps=1e-5)
        self.c_attn = nn.Linear(C, 3 * C)
        self.attn_proj = nn.Linear(C, C)
        self.ln_2 = nn.LayerNorm(C, eps=1e-5)
        self.c_fc = nn.Linear(C, 4 * C)
        self.mlp_proj = nn.Linear(4 * C, C)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(self.ln_1(x)).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.attention_scale)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.attn_proj(y)
        x = x + self.mlp_proj(F.gelu(self.c_fc(self.ln_2(x)), approximate="tanh"))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.padded_vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.max_seq_len, config.n_embd)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=1e-5)
        # only the first n_active rows of wte can be predicted (the rest are unused padding)
        self.n_active = config.vocab_size

    def forward(self, idx, targets=None, loss_reduction="mean"):
        B, T = idx.size()
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        if self.config.use_mup:
            x = x / self.config.mup_width_mult
        logits = F.linear(x, self.wte.weight[: self.n_active])  # tied output head
        if targets is None:
            return logits
        # targets == -1 are ignored (prompt tokens, padding)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1),
                               ignore_index=-1, reduction=loss_reduction)
        return loss

    @torch.no_grad()
    def generate_stream(self, ids, max_new_tokens, stop_token=None, temperature=0.0, top_k=50):
        """Naive (no KV cache) sampling for a single sequence, yielding one token at a time."""
        if not ids or max_new_tokens < 0 or temperature < 0 or top_k < 0:
            raise ValueError("generation needs a nonempty prompt and nonnegative sampling parameters")
        out = list(ids)
        device = self.wte.weight.device
        for _ in range(max_new_tokens):
            ctx = torch.tensor([out[-self.config.max_seq_len:]], dtype=torch.long, device=device)
            logits = self(ctx)[0, -1].float()
            if temperature == 0.0:
                nxt = int(logits.argmax())
            else:
                # same as nanochat's sample_next_token: top-k, then temperature
                if top_k:
                    vals, idx = torch.topk(logits, min(top_k, logits.size(-1)))
                    nxt = int(idx[torch.multinomial(F.softmax(vals / temperature, dim=-1), 1)])
                else:
                    nxt = int(torch.multinomial(F.softmax(logits / temperature, dim=-1), 1))
            out.append(nxt)
            yield nxt
            if nxt == stop_token:
                break

    def generate(self, ids, max_new_tokens, stop_token=None, temperature=0.0, top_k=50):
        return list(self.generate_stream(ids, max_new_tokens, stop_token, temperature, top_k))


# ---------------------------------------------------------------------------
# llm.c checkpoint I/O

def adamw_groups(model, base_lr, weight_decay):
    """Apply the trainer's muP LR/decay scaling to the four hidden weight matrices."""
    grouped = {}
    hidden_suffixes = (".c_attn.weight", ".attn_proj.weight", ".c_fc.weight", ".mlp_proj.weight")
    for name, param in model.named_parameters():
        hidden = model.config.use_mup and name.endswith(hidden_suffixes)
        lr_scale = 1.0 / model.config.mup_width_mult if hidden else 1.0
        decay = weight_decay if param.ndim >= 2 else 0.0
        if hidden:
            decay *= model.config.mup_width_mult
        grouped.setdefault((lr_scale, decay), []).append(param)
    return [{"params": params, "lr": base_lr * scale, "lr_scale": scale, "weight_decay": decay}
            for (scale, decay), params in grouped.items()]

def _param_shapes(cfg):
    Vp, C, T, L = cfg.padded_vocab_size, cfg.n_embd, cfg.max_seq_len, cfg.n_layer
    return [
        ("wte", (Vp, C)), ("wpe", (T, C)),
        ("ln1w", (L, C)), ("ln1b", (L, C)),
        ("qkvw", (L, 3 * C, C)), ("qkvb", (L, 3 * C)),
        ("attprojw", (L, C, C)), ("attprojb", (L, C)),
        ("ln2w", (L, C)), ("ln2b", (L, C)),
        ("fcw", (L, 4 * C, C)), ("fcb", (L, 4 * C)),
        ("fcprojw", (L, C, 4 * C)), ("fcprojb", (L, C)),
        ("lnfw", (C,)), ("lnfb", (C,)),
    ]


def _llmc_to_module(model):
    """Map each llm.c tensor name to a function giving the (per-layer) torch parameters."""
    h = model.h
    return {
        "wte": lambda: [model.wte.weight], "wpe": lambda: [model.wpe.weight],
        "ln1w": lambda: [b.ln_1.weight for b in h], "ln1b": lambda: [b.ln_1.bias for b in h],
        "qkvw": lambda: [b.c_attn.weight for b in h], "qkvb": lambda: [b.c_attn.bias for b in h],
        "attprojw": lambda: [b.attn_proj.weight for b in h], "attprojb": lambda: [b.attn_proj.bias for b in h],
        "ln2w": lambda: [b.ln_2.weight for b in h], "ln2b": lambda: [b.ln_2.bias for b in h],
        "fcw": lambda: [b.c_fc.weight for b in h], "fcb": lambda: [b.c_fc.bias for b in h],
        "fcprojw": lambda: [b.mlp_proj.weight for b in h], "fcprojb": lambda: [b.mlp_proj.bias for b in h],
        "lnfw": lambda: [model.ln_f.weight], "lnfb": lambda: [model.ln_f.bias],
    }


def load_llmc(path, device="cpu"):
    header = np.fromfile(path, dtype="<i4", count=256)
    if len(header) != 256:
        raise ValueError(f"truncated checkpoint header: {path}")
    assert header[0] == LLMC_MAGIC, f"bad magic number in {path}"
    version = int(header[1])
    assert version in (LLMC_VERSION_FP32, LLMC_VERSION_BF16), f"unsupported version {version}"
    parametrization = {}
    if header[8] != 0:
        if header[8] != MUP_MAGIC or header[9] != 1 or header[10] not in (0, 1):
            raise ValueError("unsupported checkpoint parametrization metadata")
        mult, attn, reference = header[11:14].view("<f4")
        if not all(np.isfinite(v) and v > 0 for v in (mult, attn, reference)):
            raise ValueError("invalid checkpoint muP metadata")
        parametrization = dict(use_mup=bool(header[10]), mup_width_mult=float(mult),
                               mup_base_attn_mult=float(attn), mup_reference_width=float(reference))
    cfg = GPTConfig(max_seq_len=int(header[2]), vocab_size=int(header[3]), n_layer=int(header[4]),
                    n_head=int(header[5]), n_embd=int(header[6]), padded_vocab_size=int(header[7]),
                    **parametrization)
    if version == LLMC_VERSION_BF16:
        raw = torch.from_numpy(np.fromfile(path, dtype=np.int16, offset=256 * 4))
        flat = raw.view(torch.bfloat16).float()
    else:
        flat = torch.from_numpy(np.fromfile(path, dtype=np.float32, offset=256 * 4))
    shapes = _param_shapes(cfg)
    expected = sum(int(np.prod(s)) for _, s in shapes)
    assert flat.numel() == expected, f"{path}: {flat.numel()} params, expected {expected}"

    model = GPT(cfg)
    model._checkpoint_header = header.copy()
    targets = _llmc_to_module(model)
    offset = 0
    with torch.no_grad():
        for name, shape in shapes:
            n = int(np.prod(shape))
            t = flat[offset:offset + n].view(shape)
            offset += n
            params = targets[name]()
            if len(params) == 1 and name in ("wte", "wpe", "lnfw", "lnfb"):
                params[0].copy_(t)
            else:
                for layer, p in enumerate(params):
                    p.copy_(t[layer])
    return model.to(device)


def save_llmc(model, path, bf16=True):
    cfg = model.config
    # Preserve extension fields when reading/writing checkpoints from the C trainer.
    header = getattr(model, "_checkpoint_header", np.zeros(256, dtype="<i4")).copy()
    header[:8] = [LLMC_MAGIC, LLMC_VERSION_BF16 if bf16 else LLMC_VERSION_FP32, cfg.max_seq_len,
                  cfg.vocab_size, cfg.n_layer, cfg.n_head, cfg.n_embd, cfg.padded_vocab_size]
    if cfg.use_mup or header[8] == MUP_MAGIC:
        header[8:11] = [MUP_MAGIC, 1, int(cfg.use_mup)]
        header[11:14] = np.array([cfg.mup_width_mult, cfg.mup_base_attn_mult,
                                 cfg.mup_reference_width], dtype="<f4").view("<i4")
    targets = _llmc_to_module(model)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        for name, _ in _param_shapes(cfg):
            t = torch.stack([p.detach().float().cpu() for p in targets[name]()])
            if name in ("wte", "wpe", "lnfw", "lnfb"):
                t = t[0]
            if bf16:
                f.write(t.to(torch.bfloat16).view(torch.int16).numpy().tobytes())
            else:
                f.write(t.numpy().tobytes())
