"""CPU checks; optionally include the protected seed with --checkpoint PATH."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from gpt2 import GPT, GPTConfig, MUP_MAGIC, adamw_groups, load_llmc, save_llmc

HERE = Path(__file__).resolve().parent
REAL_CHECKPOINT = None
torch.set_num_threads(2)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def reference(model, tokens):
    """Explicit attention/readout equations matching the C trainer; no SDPA or Block.forward."""
    cfg = model.config
    x = F.embedding(tokens, model.wte.weight) + F.embedding(
        torch.arange(tokens.shape[1]), model.wpe.weight)
    for b in model.h:
        z = F.layer_norm(x, (cfg.n_embd,), b.ln_1.weight, b.ln_1.bias, 1e-5)
        q, k, v = F.linear(z, b.c_attn.weight, b.c_attn.bias).chunk(3, dim=-1)
        B, T, C = q.shape
        q, k, v = [t.reshape(B, T, cfg.n_head, C // cfg.n_head).transpose(1, 2) for t in (q, k, v)]
        scale = cfg.mup_base_attn_mult / (C // cfg.n_head) if cfg.use_mup else (C // cfg.n_head) ** -0.5
        scores = (q @ k.transpose(-1, -2)) * scale
        mask = torch.ones(T, T, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(mask, float("-inf"))
        att = (scores.softmax(-1) @ v).transpose(1, 2).reshape(B, T, C)
        x = x + F.linear(att, b.attn_proj.weight, b.attn_proj.bias)
        z = F.layer_norm(x, (C,), b.ln_2.weight, b.ln_2.bias, 1e-5)
        z = F.gelu(F.linear(z, b.c_fc.weight, b.c_fc.bias), approximate="tanh")
        x = x + F.linear(z, b.mlp_proj.weight, b.mlp_proj.bias)
    x = F.layer_norm(x, (cfg.n_embd,), model.ln_f.weight, model.ln_f.bias, 1e-5)
    if cfg.use_mup:
        x = x / cfg.mup_width_mult
    return x @ model.wte.weight[:model.n_active].T


def tiny(mup=True, tokenizer=False):
    torch.manual_seed(42)
    cfg = GPTConfig(max_seq_len=64, vocab_size=50257 if tokenizer else 47,
                    padded_vocab_size=50304 if tokenizer else 64,
                    n_layer=1, n_head=3, n_embd=24, use_mup=mup,
                    mup_width_mult=4 if mup else 1, mup_base_attn_mult=4 if mup else 1)
    model = GPT(cfg)
    with torch.no_grad():
        model.wte.weight[cfg.vocab_size:].zero_()
    return model


class Checkpoints(unittest.TestCase):
    def test_forward_and_gradients(self):
        for mup in (False, True):
            with self.subTest(mup=mup):
                model = tiny(mup)
                ids = torch.tensor([[1, 5, 8, 3]])
                actual, expected = model(ids), reference(model, ids)
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
                params = tuple(model.parameters())
                a = torch.autograd.grad(actual.square().mean(), params)
                b = torch.autograd.grad(expected.square().mean(), params)
                for x, y in zip(a, b):
                    torch.testing.assert_close(x, y, atol=3e-6, rtol=2e-5)

    def test_roundtrip_both_precisions_and_parametrizations(self):
        with tempfile.TemporaryDirectory() as d:
            for mup in (False, True):
                for bf16 in (False, True):
                    model = tiny(mup)
                    p, q = Path(d)/"first.bin", Path(d)/"second.bin"
                    save_llmc(model, p, bf16=bf16)
                    restored = load_llmc(p)
                    save_llmc(restored, q, bf16=bf16)
                    self.assertEqual(digest(p), digest(q))
                    self.assertEqual(restored.config.use_mup, mup)
                    self.assertEqual(restored.config.mup_width_mult, 4 if mup else 1)
                    self.assertEqual(restored.config.mup_base_attn_mult, 4 if mup else 1)

    def test_unknown_or_invalid_metadata_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"model.bin"
            for index, value in ((8, 1234), (9, 2), (10, 3), (11, 0), (13, 0)):
                save_llmc(tiny(), p)
                with p.open("r+b") as f:
                    f.seek(4 * index)
                    f.write(np.array([value], dtype="<i4").tobytes())
                with self.assertRaises(ValueError):
                    load_llmc(p)

    def test_unknown_header_padding_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            p, q = Path(d)/"model.bin", Path(d)/"saved.bin"
            save_llmc(tiny(), p)
            with p.open("r+b") as f:
                f.seek(200 * 4)
                f.write(np.array([123456], dtype="<i4").tobytes())
            save_llmc(load_llmc(p), q)
            self.assertEqual(digest(p), digest(q))

    def test_hidden_optimizer_scaling(self):
        model = tiny()
        groups = adamw_groups(model, 0.001, 0.1)
        all_ids = [id(p) for g in groups for p in g["params"]]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual(set(all_ids), {id(p) for p in model.parameters()})
        for name, p in model.named_parameters():
            g = next(g for g in groups if any(p is other for other in g["params"]))
            hidden = name in {"h.0.c_attn.weight", "h.0.attn_proj.weight", "h.0.c_fc.weight", "h.0.mlp_proj.weight"}
            self.assertEqual(g["lr"], 0.00025 if hidden else 0.001)
            self.assertEqual(g["weight_decay"], 0.4 if hidden else 0.1 if p.ndim >= 2 else 0.0)

    def test_protected_seed(self):
        if not REAL_CHECKPOINT:
            self.skipTest("pass --checkpoint to audit the actual seed")
        before = digest(REAL_CHECKPOINT)
        model = load_llmc(REAL_CHECKPOINT)
        self.assertEqual((model.config.use_mup, model.config.mup_width_mult,
                          model.config.mup_base_attn_mult, model.config.mup_reference_width),
                         (True, 4.0, 4.0, 192.0))
        with torch.no_grad():
            ids = torch.tensor([[50256, 464, 2068, 7586]])
            logits, expected = model(ids), reference(model, ids)
            torch.testing.assert_close(logits, expected, atol=2e-5, rtol=2e-5)
            self.assertTrue(torch.isfinite(logits).all())
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/"seed-copy.bin"
            save_llmc(model, out)
            self.assertEqual(before, digest(out))
        self.assertEqual(before, digest(REAL_CHECKPOINT))


class Commands(unittest.TestCase):
    def run_script(self, script, *args):
        env = dict(os.environ, OMP_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false")
        return subprocess.run([sys.executable, str(HERE/script), *map(str, args)],
                              input="IGNORED INPUT\n/quit\n", text=True, capture_output=True,
                              timeout=90, env=env)

    def test_prompt_required_before_loading(self):
        result = self.run_script("generate.py", "--model", "/missing/model.bin")
        self.assertEqual(result.returncode, 2)
        self.assertIn("--prompt", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_single_generation_and_context_limit(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"model.bin"
            save_llmc(tiny(tokenizer=True), path)
            args = ("--model", path, "--prompt", "Hello", "--device", "cpu", "-g", "2", "--seed", "42")
            first = self.run_script("generate.py", *args)
            second = self.run_script("generate.py", *args)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(first.stdout, second.stdout)
            self.assertTrue(first.stdout.startswith("Hello"))
            self.assertNotIn("IGNORED INPUT", first.stdout)
            self.assertNotIn("User:", first.stdout)
            too_long = self.run_script("generate.py", *args, "-g", "100")
            self.assertEqual(too_long.returncode, 2)
            self.assertIn("context", too_long.stderr)

    def test_two_sft_updates_and_saved_generation(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            path = d/"model.bin"
            save_llmc(tiny(tokenizer=True), path)
            rows = [{"messages": [{"role": "user", "content": "Say hello."},
                                  {"role": "assistant", "content": "Hello there!"}]}] * 4
            train, val = d/"train.jsonl", d/"val.jsonl"
            for p in (train, val):
                p.write_text("".join(json.dumps(r) + "\n" for r in rows))
            out = d/"sft"
            result = self.run_script("chat_sft.py", "--model", path, "--out-dir", out,
                "--device", "cpu", "--compile", "0", "--train-file", train, "--val-file", val,
                "--data-workers", "1", "--max-seq-len", "32", "--device-batch-size", "1",
                "--total-batch-size", "32", "--num-iterations", "2", "--eval-every", "1",
                "--eval-rows", "1", "--sample-every", "1", "--sample-tokens", "1",
                "--sample-prompt", "Hello")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            saved = out/"model_sft_00002.bin"
            self.assertTrue(saved.is_file())
            model = load_llmc(saved)
            self.assertTrue(model.config.use_mup)
            self.assertEqual(model.config.mup_width_mult, 4)
            self.assertEqual(json.loads((out/"model_config.json").read_text())["mup_width_mult"], 4)
            # Initial, update 1, final: exactly one prompt/sequence at each sampling event.
            self.assertEqual((out/"samples.txt").read_text().count("USER:"), 3)
            self.assertFalse(torch.equal(model.wte.weight, load_llmc(path).wte.weight))
            self.assertNotIn("nan", (out/"main.log").read_text().lower())
            generation = self.run_script("generate.py", "--model", saved, "--prompt", "Hello",
                                         "--device", "cpu", "-g", "2")
            self.assertEqual(generation.returncode, 0, generation.stderr)
            self.assertNotIn("User:", generation.stdout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint")
    args, remaining = parser.parse_known_args()
    REAL_CHECKPOINT = args.checkpoint
    unittest.main(argv=[sys.argv[0], *remaining])
