import math
from dataclasses import dataclass
import torch
import torch.nn as nn
ifrom torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embed = 384

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.n_embed % config.n_head == 0
        self.n_head = config.n_head
        self.c_attn =  nn.Linear(config.n_embed, config.n_embed * 3)

        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size,
                                                           config.block_size))).view(1, 1, config.block_size,
                                                                                     config.block_size)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.config.n_embed, dim=2)
        q = q.view(B, -1, self.h_embed, self.config.n_embed // self.n_head)
        k = k.view(B, -1, self.h_embed, self.config.n_embed // self.n_head)
        v = v.view(B, -1, self.h_embed, self.config.n_embed // self.n_head)


        dk = k.shape[-1]
        # calculate the scores
        scores = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(dk))
        scores = scores.masked_fill_(self.bias[:, :, :T, :T] == 0, float('-inf'))
        scores = F.softmax(scores, dim=-1)

        p_scores = scores @ v # B, T, n_h, hd
        p_scores = p_scores.tranpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(p_scores)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.c_fc = nn.Linear(config.n_embed, config.n_embed * 4)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(config.n_embed * 4, config.n_embed)

    def forward(self, x):
        x = self.gelu(self.c_fc(x))
        x = self.c_proj(x)
        return x
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.ln_2 = nn.LayerNorm(config.n_embed)

    def forward(self, x):
        x = self.attn(self.ln_1(x)) + x
        x = self.ffn(self.ln_2(x)) + x
        return x
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.block_size, config.n_embed),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embed)
        ))
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)