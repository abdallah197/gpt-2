import inspect
import os
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import transformers
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DPP


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of heads
    n_embd: int = 768  # embedding dimension


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)

        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size,
                                                           config.block_size)).view(1, 1, config.block_size,
                                                                                    config.block_size))
        self.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.config.n_embd, dim=2)
        q = q.view(B, -1, self.n_head, self.config.n_embd // self.n_head).transpose(1, 2)
        k = k.view(B, -1, self.n_head, self.config.n_embd // self.n_head).transpose(1, 2)
        v = v.view(B, -1, self.n_head, self.config.n_embd // self.n_head).transpose(1, 2)

        # dk = k.shape[-1]
        # # calculate the scores
        # scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(dk))
        # scores = scores.masked_fill_(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # scores = F.softmax(scores, dim=-1)
        #
        # p_scores = scores @ v  # B, T, n_h, hd
        p_scores = F.scaled_dot_product_attention(q, k, v)
        p_scores = p_scores.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(p_scores)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)
        self.NANOGPT_SCALE_INIT = 1

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
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = self.attn(self.ln_1(x)) + x
        x = self.mlp(self.ln_2(x)) + x
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type.split('/')[-1] in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type.split('/')[-1]]
        config_args['vocab_size'] = 50257  # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024  # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]  # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configue_optimizers(self, weight_decay, lr, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device

        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


class DataLoaderLite:
    def __init__(self, B, T, tokenizer, rank_process, num_processes):
        self.B = B
        self.T = T
        self.rank_process = rank_process
        self.num_processes = num_processes
        self.current_positon = self.B * self.T * self.rank_process
        with open('input.txt', 'r') as f:
            text = f.read()
        self.tokens = tokenizer(text, return_tensors='pt')['input_ids'].squeeze(0)

    def get_next_batch(self):
        B, T = self.B, self.T

        buffer = self.tokens[self.current_positon: self.current_positon + B * T + 1]

        x = buffer[:-1].view(self.B, self.T)
        y = buffer[1:].view(self.B, self.T)
        self.current_positon += B * T * self.num_processes

        if self.current_positon + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_positon = B * T * self.rank_process
        # sample a portion of context_length as X, Y
        return x, y


# print the generated text
# for i in range(num_sequences):
#     entry = x[i, :max_length]
#     decoded = tokenizer(entry)
#     print(decoded)

dpp = int(os.environ.get('RANK', -1)) != -1

if dpp:
    dist.init_process_group("nccl")
    dpp_world_size = int(os.environ['WORLD_SIZE'])
    dpp_local_rank = int(os.environ['LOCAL_RANK'])
    dpp_rank = int(os.environ['RANK'])
    device = f'cuda:{dpp_local_rank}'
    torch.cuda.set_device(device)
    master_process = dpp_rank == 0
else:
    device = 'cpu'
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")
    dpp_world_size = 1
    dpp_local_rank = 0
    master_process = True

total_batch_size = 524288  # num of tokens in one batch, 2**19, 0.5M in num of tokens
B = 16
T = 1024
assert total_batch_size % (B * T * dpp_world_size) == 0
gradient_accumm_steps = total_batch_size // (B * T * dpp_world_size)
tokenizer = transformers.AutoTokenizer.from_pretrained('openai-community/gpt2')

steps = 50
min_lr = 6e-5
max_lr = min_lr * 10
weight_decay = 0.01

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model)
model = DPP(model, device_ids=[dpp_local_rank])
raw_model = model.module

optimizer = raw_model.configue_optimizers(weight_decay=weight_decay, lr=max_lr, device=device)
tokens = tokenizer('Hello I am a language model, ', return_tensors='pt')['input_ids']
max_length = 30
num_sequences = 5
tokens = tokens.repeat(5, 1)
data_loader = DataLoaderLite(B=B, T=T, tokenizer=tokenizer, rank_process=dpp_rank, world_size=dpp_world_size)

# x = tokens.to(device)
# while x.size(1) < max_length:
#     logits, _ = model(x)  # B, T,C
#
#     logits = logits[:, -1, :]  # B, 1, C
#
#     probs = F.softmax(logits, dim=-1)
#
#     # get the top k tokens
#     top_probs, top_indices = torch.topk(probs, 50, dim=-1)
#
#     ix = torch.multinomial(top_probs, 1)
#
#     # gagther the corrospunding indicies
#     xcol = torch.gather(top_indices, -1, ix)  # B, 1
#
#     x = torch.cat((x, xcol), dim=1)


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=steps, eta_min=min_lr)
for i in range(steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(gradient_accumm_steps):
        x, y = data_loader.get_next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / gradient_accumm_steps
        loss_accum += loss.detach()
        if dpp and micro_step != gradient_accumm_steps - 1:
            with model.no_sync():
                loss.backward()
        else:
            loss.backward()
        if dpp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    dt = time.time() - t0
    current_lr = scheduler.get_last_lr()[0]
    # synchronize is called because GPU operations are asyn and Cuda can pass output to CPU before it finishes its tasks
    torch.cuda.synchronize()
    tokens_processed = data_loader.B * data_loader.T * dpp_world_size * gradient_accumm_steps
    tok_per_sec = tokens_processed / dt
    if master_process:
        print(
            f"Loss: {loss_accum.item():0.4f}, Step: {i}, Norm: {norm:.4f}, Lr: {current_lr:.6f}, time: {dt * 1000:.2f} ms, tok/sec: {tok_per_sec:.2f}")

if dpp:
    dist.destroy_process_group()