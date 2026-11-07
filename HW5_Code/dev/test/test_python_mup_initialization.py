"""Check real GPT initialization without importing optional training dependencies."""
import ast
from dataclasses import dataclass
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
tree=ast.parse((ROOT/'train_gpt2.py').read_text())
tree.body=[n for n in tree.body if isinstance(n,ast.ClassDef)]
namespace=dict(torch=torch,nn=nn,F=F,math=math,dataclass=dataclass)
exec(compile(tree,str(ROOT/'train_gpt2.py'),'exec'),namespace)
GPT,GPTConfig=namespace['GPT'],namespace['GPTConfig']
torch.set_num_threads(2)
for width in [48,96,192,384,768]:
    multiplier=width/192
    sp=GPT(GPTConfig(block_size=64,vocab_size=128,n_layer=2,n_head=12,n_embd=width))
    mu=GPT(GPTConfig(block_size=64,vocab_size=128,n_layer=2,n_head=12,n_embd=width,
                     use_mup=1,mup_width_mult=multiplier,mup_base_attn_mult=4))
    expected=dict(sp.named_parameters())
    assert mu.transformer.wte.weight is mu.lm_head.weight
    assert torch.count_nonzero(mu.transformer.wte.weight)>0
    for name,value in mu.named_parameters():
        reference=expected[name].detach().clone()
        if name.startswith('transformer.h.') and value.ndim==2:
            reference/=math.sqrt(multiplier)
            if name.endswith('attn.c_attn.weight'): reference[:width]=0
            torch.testing.assert_close(value,reference,rtol=3e-6,atol=1e-8)
        else:
            torch.testing.assert_close(value,reference,rtol=0,atol=0)
    assert all(torch.count_nonzero(block.attn.c_attn.weight[:width])==0 for block in mu.transformer.h)
    print(f'PASS width={width}: tied nonzero embeddings, zero queries, all hidden matrix scales, SP controls')
