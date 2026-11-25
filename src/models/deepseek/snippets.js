// DeepSeek-V3 reference code (minimal PyTorch, same style and names as the LLaMA code where possible).
// Verified in PyTorch at a small size: shapes, causality, the MLA cache holds kv_lora + hs_rope numbers per token,
// each token uses exactly top_k routed experts, the balancing bias changes the choice but not the weights,
// and the training loop lowers the loss. At full size (meta device): 671.0B parameters, 37.6B active per token,
// plus 11.6B in the MTP module (it shares the embedding and the LM head).

export const deepseekTrainingCode = `from transformers import AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F


class DataLoader:
    def __init__(self, batch_size: int, block_size: int, val_fraction: float):
        self.batch_size = batch_size
        self.block_size = block_size

        # at init load tokens from local file
        with open("input.txt", "r") as f:
            text = f.read()                   # str
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")  # BPE, 129,280 tokens
        ids = tokenizer.encode(text) # return the token ids: list of N ints
        ids = torch.tensor(ids, dtype=torch.long)    # (N,)
        n = int((1.0 - val_fraction) * len(ids))     # int, 90% of N
        self.train_ids = ids[:n]                     # (n,)
        self.val_ids = ids[n:]                       # (N - n,)

    def get_batch(self, split: str):
        data = self.train_ids if split == "train" else self.val_ids   # (n,)
        # B = batch_size = 4, T = block_size = 32
        ix = torch.randint(len(data) - self.block_size - 1, (self.batch_size,))  # (B,) start positions
        x = torch.stack([data[i:i+self.block_size] for i in ix])      # (B, T)
        y = torch.stack([data[i+1:i+1+self.block_size] for i in ix])  # (B, T), x shifted by one
        return x, y


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))    # (dim,), no bias

    def forward(self, x):                              # x: (..., dim)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class MTP(nn.Module):
    # multi-token prediction: from the hidden state at position t and the embedding of token t+1, predict token t+2
    def __init__(self, cfg):
        super().__init__()
        C = cfg["n_embd"]
        self.norm_h = RMSNorm(C)
        self.norm_e = RMSNorm(C)
        self.proj = nn.Linear(2 * C, C, bias=False)   # weight (C, 2C): merge the two inputs
        self.block = TransformerBlock(cfg, moe=True)  # one more block of its own
        self.norm_out = RMSNorm(C)

    def forward(self, h, emb_next):                   # both (B, T-1, C)
        x = self.proj(torch.cat([self.norm_h(h), self.norm_e(emb_next)], dim=-1))   # (B, T-1, C)
        return self.norm_out(self.block(x))           # (B, T-1, C)


class DeepSeekV3(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # V = vocab_size = 129280, C = n_embd = 7168
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["n_embd"])   # weight (V, C); no position embedding
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg, moe=(i >= cfg["n_dense"]))   # first 3 blocks dense, the other 58 MoE
            for i in range(cfg["n_layer"])
        ])
        self.ln_f = RMSNorm(cfg["n_embd"])                                  # weight (C,)
        self.lm_head = nn.Linear(cfg["n_embd"], cfg["vocab_size"], bias=False)  # weight (V, C), not tied
        self.mtp = MTP(cfg)                                                 # training only

    def forward(self, idx, targets=None):    # idx: (B, T), targets: (B, T)
        x = self.tok_emb(idx)         # (B, T, C)
        for blk in self.blocks:
            x = blk(x)                # (B, T, C) -> (B, T, C)
        logits = self.lm_head(self.ln_f(x))   # (B, T, V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            # MTP: at position t, use x[t] and the embedding of token t+1 (= targets[t]) to predict token t+2
            h2 = self.mtp(x[:, :-1], self.tok_emb(targets[:, :-1]))           # (B, T-1, C)
            logits2 = self.lm_head(h2)                                         # (B, T-1, V), same LM head
            loss_mtp = F.cross_entropy(logits2.reshape(-1, logits2.size(-1)), targets[:, 1:].reshape(-1))
            loss = loss + 0.3 * loss_mtp      # MTP weight 0.3 (0.1 later in training)
        return logits, loss           # (B, T, V), () scalar

# ----- Simple training loop -----
device = "cuda" if torch.cuda.is_available() else "cpu"
# DeepSeek-V3: 671B parameters, 37B active per token
cfg = dict(vocab_size=129280, n_layer=61, n_dense=3, n_embd=7168, n_head=128,
           q_lora=1536, kv_lora=512, hs_nope=128, hs_rope=64, hs_v=128,      # MLA
           dense_hidden=18432, n_routed=256, n_shared=1, top_k=8, expert_hidden=2048,   # MoE
           max_toks=4096)
model = DeepSeekV3(cfg).to(device)
decay    = [p for p in model.parameters() if p.dim() >= 2]
no_decay = [p for p in model.parameters() if p.dim() < 2]
optimizer = torch.optim.AdamW([
    {"params": decay,    "weight_decay": 0.1},
    {"params": no_decay, "weight_decay": 0.0},
], lr=2.2e-4, betas=(0.9, 0.95))

batch_size = 4
block_size = 32
val_fraction = 0.1
dataloader = DataLoader(batch_size=batch_size, block_size=block_size, val_fraction=val_fraction)
max_steps = 1000
for step in range(max_steps):
    x, y = dataloader.get_batch("train")    # x: (4, 32), y: (4, 32)
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad(set_to_none=True)   # every p.grad -> None
    _, loss = model(x, y)                   # logits (4, 32, 129280), loss () scalar
    loss.backward()                         # each p.grad gets the same shape as p
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # rescale all grads so total norm <= 1.0
    optimizer.step()                        # each p updated in place, shape unchanged
    with torch.no_grad():                   # load balancing without an auxiliary loss:
        for blk in model.blocks:            # nudge each expert's bias toward the average load
            if isinstance(blk.mlp, MoE):
                blk.mlp.bias += 0.001 * torch.sign(blk.mlp.load.mean() - blk.mlp.load)
    if step % 100 == 0:
        print(f"step {step} loss {loss.item():.4f}")
`;

export const blockCode = `import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, cfg, moe):
        super().__init__()
        C = cfg["n_embd"]
        self.ln1 = RMSNorm(C)
        self.attn = CausalSelfAttention(C, cfg["n_head"], cfg["q_lora"], cfg["kv_lora"],
                                        cfg["hs_nope"], cfg["hs_rope"], cfg["hs_v"], cfg["max_toks"])
        self.ln2 = RMSNorm(C)
        if moe:
            self.mlp = MoE(C, cfg["n_routed"], cfg["n_shared"], cfg["top_k"], cfg["expert_hidden"])
        else:
            self.mlp = MLP(C, cfg["dense_hidden"])     # first 3 blocks: one dense SwiGLU MLP

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
`;

export const attnCode = `import torch
import torch.nn as nn
import torch.nn.functional as F


def rope_cache(hs, max_toks, theta=10000.0):
    inv_freq = 1.0 / theta ** (torch.arange(0, hs, 2).float() / hs)   # (hs/2,)
    angles = torch.outer(torch.arange(max_toks).float(), inv_freq)    # (max_toks, hs/2)
    return angles.cos(), angles.sin()                                 # each (max_toks, hs/2)


def apply_rope(x, cos, sin):                         # x: (..., T, hs), cos/sin: (T, hs/2)
    x1, x2 = x[..., 0::2], x[..., 1::2]              # each pair of dims (x1, x2)
    out1 = x1 * cos - x2 * sin                       # rotate each pair by its angle
    out2 = x1 * sin + x2 * cos
    return torch.stack([out1, out2], dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    # multi-head latent attention (MLA)
    def __init__(self, n_embd, n_head, q_lora, kv_lora, hs_nope, hs_rope, hs_v, max_toks):
        super().__init__()
        self.n_head, self.kv_lora = n_head, kv_lora   # 128 heads, latent of 512
        self.hs_nope, self.hs_rope, self.hs_v = hs_nope, hs_rope, hs_v   # 128, 64, 128 per head
        # q: down to a 1536 latent, then up to 128 heads
        self.q_down = nn.Linear(n_embd, q_lora, bias=False)                       # weight (1536, C)
        self.q_norm = RMSNorm(q_lora)
        self.q_up = nn.Linear(q_lora, n_head * (hs_nope + hs_rope), bias=False)  # weight (128*192, 1536)
        # k, v: down to a 512 latent (cached) plus one 64-d RoPE key shared by all heads (cached)
        self.kv_down = nn.Linear(n_embd, kv_lora + hs_rope, bias=False)           # weight (576, C)
        self.kv_norm = RMSNorm(kv_lora)
        self.kv_up = nn.Linear(kv_lora, n_head * (hs_nope + hs_v), bias=False)   # weight (128*256, 512)
        self.o_proj = nn.Linear(n_head * hs_v, n_embd, bias=False)               # weight (C, 128*128)
        cos, sin = rope_cache(hs_rope, max_toks)     # RoPE only on the 64-d part
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x):                            # x: (B, T, C)
        B, T, C = x.size()
        nh = self.n_head                             # 128

        q = self.q_up(self.q_norm(self.q_down(x)))                       # (B, T, nh*192)
        q = q.view(B, T, nh, self.hs_nope + self.hs_rope).transpose(1, 2)  # (B, nh, T, 192)
        q_nope, q_rope = q.split([self.hs_nope, self.hs_rope], dim=-1)    # (B, nh, T, 128), (B, nh, T, 64)

        # c_kv (after kv_norm) and the rotated k_rope: all that a KV cache keeps per token when generating (see generate.py)
        c_kv, k_rope = self.kv_down(x).split([self.kv_lora, self.hs_rope], dim=-1)  # (B, T, 512), (B, T, 64): the KV cache
        kv = self.kv_up(self.kv_norm(c_kv))                               # (B, T, nh*256)
        kv = kv.view(B, T, nh, self.hs_nope + self.hs_v).transpose(1, 2)  # (B, nh, T, 256)
        k_nope, v = kv.split([self.hs_nope, self.hs_v], dim=-1)          # (B, nh, T, 128) each

        q_rope = apply_rope(q_rope, self.cos[:T], self.sin[:T])                 # (B, nh, T, 64)
        k_rope = apply_rope(k_rope.unsqueeze(1), self.cos[:T], self.sin[:T])    # (B, 1, T, 64): one key for all heads
        q = torch.cat([q_nope, q_rope], dim=-1)                                 # (B, nh, T, 192)
        k = torch.cat([k_nope, k_rope.expand(-1, nh, -1, -1)], dim=-1)          # (B, nh, T, 192)

        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.hs_nope + self.hs_rope) ** 0.5)   # (B, nh, T, T)
        mask = torch.tril(torch.ones(T, T, device=x.device))                # (T, T)
        att = att.masked_fill(mask == 0, float('-inf'))                     # (B, nh, T, T)
        att = F.softmax(att, dim=-1)                                        # (B, nh, T, T)
        y = att @ v                                                         # (B, nh, T, 128)
        y = y.transpose(1, 2).contiguous().view(B, T, nh * self.hs_v)       # (B, T, nh*128)
        return self.o_proj(y)                                               # (B, T, C)
`;

export const mlpCode = `import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    # a SwiGLU MLP, as in LLaMA: the dense MLP of the first 3 blocks, and every expert
    def __init__(self, n_embd, hidden):
        super().__init__()
        self.gate_proj = nn.Linear(n_embd, hidden, bias=False)   # weight (H, C)
        self.up_proj   = nn.Linear(n_embd, hidden, bias=False)   # weight (H, C)
        self.down_proj = nn.Linear(hidden, n_embd, bias=False)   # weight (C, H)

    def forward(self, x):                              # x: (..., C)
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoE(nn.Module):
    # DeepSeekMoE: 1 shared expert + 256 routed experts (hidden 2048 each); each token uses the top 8
    def __init__(self, n_embd, n_routed, n_shared, top_k, hidden):
        super().__init__()
        self.top_k = top_k                                        # 8
        self.router = nn.Linear(n_embd, n_routed, bias=False)     # weight (256, C): one score per expert
        self.register_buffer("bias", torch.zeros(n_routed))       # (256,) balancing bias, not trained by gradients
        self.register_buffer("load", torch.zeros(n_routed))       # (256,) tokens per expert in the last batch
        self.experts = nn.ModuleList([MLP(n_embd, hidden) for _ in range(n_routed)])
        self.shared = MLP(n_embd, n_shared * hidden)              # used by every token

    def forward(self, x):                                         # x: (B, T, C)
        B, T, C = x.size()
        x = x.view(B * T, C)                                      # (N, C), N = B*T tokens
        scores = torch.sigmoid(self.router(x))                    # (N, 256) affinity of each token to each expert
        _, idx = (scores + self.bias).topk(self.top_k, dim=-1)    # (N, 8) chosen experts; bias only for choosing
        w = scores.gather(-1, idx)                                # (N, 8) their scores, without the bias
        w = w / w.sum(dim=-1, keepdim=True)                       # (N, 8) gate weights, sum to 1
        self.load = torch.bincount(idx.flatten(), minlength=len(self.experts)).float()   # (256,)

        y = self.shared(x)                                        # (N, C) shared expert, every token
        for e, expert in enumerate(self.experts):
            tok, slot = (idx == e).nonzero(as_tuple=True)         # which tokens chose expert e, and in which slot
            if tok.numel() > 0:
                y = y.index_add(0, tok, w[tok, slot, None] * expert(x[tok]))   # add weighted output
        return y.view(B, T, C)                                    # (B, T, C)
`;

// Inference: prefill + decode with an MLA KV cache and absorbed W_uk / W_uv. Verified in PyTorch at a small size:
// the cached logits match a full forward at every step (max difference 6e-7).
export const generateCode = `from transformers import AutoTokenizer
import torch
import torch.nn.functional as F

# Text generation with an MLA KV cache, for the DeepSeekV3 model in train_deepseek_v3.py.
# The training forward computes k and v from the latent for every token at once. Here:
#   prefill: the whole prompt in one pass, as in training, and its latents go into the cache
#   decode:  one new token per step, with W_uk and W_uv absorbed, so k and v are never computed


class MLACache:
    # one per layer: per token only 512 + 64 = 576 numbers
    def __init__(self):
        self.c_kv = None      # (B, T, 512) normalized latents
        self.k_rope = None    # (B, T, 64)  RoPE keys, already rotated to their positions

    def append(self, c_kv, k_rope):
        if self.c_kv is None:
            self.c_kv, self.k_rope = c_kv, k_rope
        else:
            self.c_kv = torch.cat([self.c_kv, c_kv], dim=1)
            self.k_rope = torch.cat([self.k_rope, k_rope], dim=1)
        return self.c_kv, self.k_rope


def mla_prefill(attn, x, cache):                      # x: (B, T, C), the whole prompt
    B, T, C = x.size()
    nh = attn.n_head                                  # 128
    q = attn.q_up(attn.q_norm(attn.q_down(x)))                              # (B, T, nh*192)
    q = q.view(B, T, nh, attn.hs_nope + attn.hs_rope).transpose(1, 2)       # (B, nh, T, 192)
    q_nope, q_rope = q.split([attn.hs_nope, attn.hs_rope], dim=-1)         # (B, nh, T, 128), (B, nh, T, 64)

    c_kv, k_rope = attn.kv_down(x).split([attn.kv_lora, attn.hs_rope], dim=-1)   # (B, T, 512), (B, T, 64)
    c_kv = attn.kv_norm(c_kv)
    k_rope = apply_rope(k_rope, attn.cos[:T], attn.sin[:T])               # (B, T, 64)
    cache.append(c_kv, k_rope)                        # store 576 numbers per token

    # T queries at once: computing k and v from the latent is cheaper here, as in training
    kv = attn.kv_up(c_kv).view(B, T, nh, attn.hs_nope + attn.hs_v).transpose(1, 2)   # (B, nh, T, 256)
    k_nope, v = kv.split([attn.hs_nope, attn.hs_v], dim=-1)               # (B, nh, T, 128) each
    q = torch.cat([q_nope, apply_rope(q_rope, attn.cos[:T], attn.sin[:T])], dim=-1)   # (B, nh, T, 192)
    k = torch.cat([k_nope, k_rope.unsqueeze(1).expand(-1, nh, -1, -1)], dim=-1)       # (B, nh, T, 192)
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)           # (B, nh, T, 128), causal as in attention.py
    y = y.transpose(1, 2).reshape(B, T, nh * attn.hs_v)                   # (B, T, nh*128)
    return attn.o_proj(y)                                                 # (B, T, C)


def mla_decode(attn, x, cache):                       # x: (B, 1, C), the newest token
    B = x.size(0)
    nh = attn.n_head
    pos = cache.c_kv.size(1)                          # its position = number of tokens already cached
    cos, sin = attn.cos[pos:pos+1], attn.sin[pos:pos+1]
    q = attn.q_up(attn.q_norm(attn.q_down(x)))                              # (B, 1, nh*192)
    q = q.view(B, 1, nh, attn.hs_nope + attn.hs_rope).transpose(1, 2)       # (B, nh, 1, 192)
    q_nope, q_rope = q.split([attn.hs_nope, attn.hs_rope], dim=-1)         # (B, nh, 1, 128), (B, nh, 1, 64)
    q_rope = apply_rope(q_rope, cos, sin)

    c_new, k_new = attn.kv_down(x).split([attn.kv_lora, attn.hs_rope], dim=-1)  # (B, 1, 512), (B, 1, 64)
    c_kv, k_rope = cache.append(attn.kv_norm(c_new), apply_rope(k_new, cos, sin))  # (B, T, 512), (B, T, 64)

    # absorb: split kv_up's weight into each head's rows for k (W_uk) and for v (W_uv)
    W = attn.kv_up.weight.view(nh, attn.hs_nope + attn.hs_v, attn.kv_lora)   # (nh, 256, 512)
    W_uk, W_uv = W[:, :attn.hs_nope], W[:, attn.hs_nope:]                     # (nh, 128, 512) each

    q_abs = q_nope @ W_uk                             # (B, nh, 1, 512): W_uk moved onto q
    scale = 1.0 / (attn.hs_nope + attn.hs_rope) ** 0.5
    att = (q_abs @ c_kv.unsqueeze(1).transpose(-2, -1)          # (B, nh, 1, T) content: dot with each cached latent
           + q_rope @ k_rope.unsqueeze(1).transpose(-2, -1)     # (B, nh, 1, T) position: dot with each RoPE key
           ) * scale
    att = F.softmax(att, dim=-1)                      # no mask needed: the newest token may see every cached token
    y = att @ c_kv.unsqueeze(1)                       # (B, nh, 1, 512): weighted sum of latents
    y = y @ W_uv.transpose(-2, -1)                    # (B, nh, 1, 128): W_uv applied once, after the sum
    y = y.transpose(1, 2).reshape(B, 1, nh * attn.hs_v)                   # (B, 1, nh*128)
    return attn.o_proj(y)                                                 # (B, 1, C)


def forward_cached(model, idx, caches):              # idx: (B, T) prompt, or (B, 1) newest token
    x = model.tok_emb(idx)                            # (B, T, C)
    for blk, cache in zip(model.blocks, caches):
        mla = mla_prefill if cache.c_kv is None else mla_decode
        x = x + mla(blk.attn, blk.ln1(x), cache)      # same block as TransformerBlock.forward
        x = x + blk.mlp(blk.ln2(x))
    return model.lm_head(model.ln_f(x[:, -1]))        # (B, V) logits for the last position only


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=50):   # idx: (B, T) prompt ids
    caches = [MLACache() for _ in model.blocks]       # one cache per layer (61)
    for i in range(max_new_tokens):
        new = idx if i == 0 else idx[:, -1:]          # prefill the prompt once, then only the newest token
        logits = forward_cached(model, new, caches) / temperature   # (B, V)
        top, _ = torch.topk(logits, top_k)            # (B, top_k)
        logits[logits < top[:, [-1]]] = float('-inf') # keep only the top_k most likely tokens
        next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)   # (B, 1)
        idx = torch.cat([idx, next_id], dim=1)        # (B, T+1)
    return idx


# ----- Generate -----
# model: the DeepSeekV3 trained in train_deepseek_v3.py (the MTP module is not used here)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")
prompt = torch.tensor([tokenizer.encode("The capital of France is")], device=device)   # (1, T)
out = generate(model, prompt, max_new_tokens=20)      # (1, T+20)
print(tokenizer.decode(out[0].tolist()))
`;

// One entry per diagram node: the code to show and the exact lines to highlight.
export const content = {
  gpt2: {
    title: "DeepSeek-V3 (model & training)",
    summary: "671B parameters, 37B active per token: MLA, DeepSeekMoE, multi-token prediction.",
    code: deepseekTrainingCode,
    highlight: ["class DeepSeekV3(nn.Module):"],
  },
  input_data: {
    title: "Input Data",
    summary: "Read raw text from a file.",
    code: deepseekTrainingCode,
    highlight: ['with open("input.txt", "r") as f:', "text = f.read()"],
  },
  tokenizer: {
    title: "Tokenizer",
    summary: "Byte-level BPE with 129,280 tokens (about 128K).",
    code: deepseekTrainingCode,
    highlight: ['tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")', "ids = tokenizer.encode(text)"],
  },
  token_embed: {
    title: "Token Embeddings",
    summary: "Token ids to 7168-d vectors; position enters through RoPE inside attention.",
    code: deepseekTrainingCode,
    highlight: ['self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["n_embd"])', "x = self.tok_emb(idx)"],
  },
  stack: {
    title: "Transformer Block",
    summary: "61 pre-norm blocks: MLA, then a dense MLP (first 3 blocks) or DeepSeekMoE (the other 58).",
    code: blockCode,
    highlight: ["class TransformerBlock(nn.Module):"],
  },
  block_ln: {
    title: "RMSNorm before Attention",
    summary: "As in LLaMA.",
    code: blockCode,
    highlight: ["self.ln1 = RMSNorm(C)", "self.ln1(x)"],
  },
  block_attn: {
    title: "Multi-head Latent Attention",
    summary: "128 heads whose k and v are rebuilt from one small cached latent per token.",
    code: blockCode,
    highlight: ["self.attn = CausalSelfAttention(", "x = x + self.attn(self.ln1(x))"],
  },
  block_attn_plus: {
    title: "Attention residual",
    summary: "Add the attention output back to x.",
    code: blockCode,
    highlight: ["x = x + self.attn(self.ln1(x))"],
  },
  block_ln2: {
    title: "RMSNorm before MLP",
    summary: "As in LLaMA.",
    code: blockCode,
    highlight: ["self.ln2 = RMSNorm(C)", "self.ln2(x)"],
  },
  block_mlp: {
    title: "DeepSeekMoE",
    summary: "Mixture of experts in 58 of the 61 blocks; the first 3 keep a dense SwiGLU MLP.",
    code: blockCode,
    highlight: ['self.mlp = MoE(C, cfg["n_routed"], cfg["n_shared"], cfg["top_k"], cfg["expert_hidden"])', 'self.mlp = MLP(C, cfg["dense_hidden"])', "x = x + self.mlp(self.ln2(x))"],
  },
  block_mlp_plus: {
    title: "MLP residual",
    summary: "Add the MoE output back to x.",
    code: blockCode,
    highlight: ["x = x + self.mlp(self.ln2(x))"],
  },
  attn_class: {
    title: "Multi-head Latent Attention",
    summary: "Low-rank compression of q and of k/v, with a separate RoPE part.",
    code: attnCode,
    highlight: ["class CausalSelfAttention(nn.Module):"],
  },
  attn_linear_q: {
    title: "Q: down, then up",
    summary: "q is compressed to a 1536-d latent and projected up to 128 heads of 192 (128 + 64 for RoPE).",
    code: attnCode,
    highlight: [
      "self.q_down = nn.Linear(n_embd, q_lora, bias=False)",
      "self.q_up = nn.Linear(q_lora, n_head * (hs_nope + hs_rope), bias=False)",
      "q = self.q_up(self.q_norm(self.q_down(x)))",
      "q_nope, q_rope = q.split([self.hs_nope, self.hs_rope], dim=-1)",
    ],
  },
  attn_linear_kv_down: {
    title: "K, V down to a latent (cached)",
    summary: "Each token is compressed to 512 numbers, plus a 64-d RoPE key: only these 576 numbers per layer are cached.",
    code: attnCode,
    highlight: [
      "self.kv_down = nn.Linear(n_embd, kv_lora + hs_rope, bias=False)",
      "c_kv, k_rope = self.kv_down(x).split([self.kv_lora, self.hs_rope], dim=-1)",
    ],
  },
  attn_linear_kv_up: {
    title: "Up-project K, V",
    summary: "k (non-RoPE part) and v for all 128 heads are rebuilt from the 512-d latent.",
    code: attnCode,
    highlight: [
      "self.kv_up = nn.Linear(kv_lora, n_head * (hs_nope + hs_v), bias=False)",
      "kv = self.kv_up(self.kv_norm(c_kv))",
      "k_nope, v = kv.split([self.hs_nope, self.hs_v], dim=-1)",
    ],
  },
  attn_rope: {
    title: "Decoupled RoPE",
    summary: "RoPE rotates only a separate 64-d part of q and k; one rotated key is shared by all heads.",
    code: attnCode,
    highlight: [
      "q_rope = apply_rope(q_rope, self.cos[:T], self.sin[:T])",
      "k_rope = apply_rope(k_rope.unsqueeze(1), self.cos[:T], self.sin[:T])",
      "q = torch.cat([q_nope, q_rope], dim=-1)",
      "k = torch.cat([k_nope, k_rope.expand(-1, nh, -1, -1)], dim=-1)",
    ],
  },
  attn_matmul_qk: {
    title: "Matmul (Q × Kᵀ)",
    summary: "Scaled dot products over 192 dimensions per head.",
    code: attnCode,
    highlight: ["att = (q @ k.transpose(-2, -1)) * (1.0 / (self.hs_nope + self.hs_rope) ** 0.5)"],
  },
  attn_masked: {
    title: "Masked Attention",
    summary: "Causal mask, as in GPT-2 and LLaMA.",
    code: attnCode,
    highlight: ["mask = torch.tril(torch.ones(T, T, device=x.device))", "att = att.masked_fill(mask == 0, float('-inf'))"],
  },
  attn_softmax: {
    title: "Softmax",
    summary: "Attention weights.",
    code: attnCode,
    highlight: ["att = F.softmax(att, dim=-1)"],
  },
  attn_matmul_v: {
    title: "Matmul (att × V)",
    summary: "Weighted sum of values.",
    code: attnCode,
    highlight: ["y = att @ v"],
  },
  attn_linear_out: {
    title: "Linear (output projection)",
    summary: "Merge 128 heads of 128 and project back to 7168.",
    code: attnCode,
    highlight: ["self.o_proj = nn.Linear(n_head * hs_v, n_embd, bias=False)", "return self.o_proj(y)"],
  },
  mlp_subdiagram: {
    title: "DeepSeekMoE",
    summary: "A router picks 8 of 256 routed experts per token; one shared expert is always used.",
    code: mlpCode,
    highlight: ["class MoE(nn.Module):"],
  },
  mlp_router: {
    title: "Router with balancing bias",
    summary: "Sigmoid scores per expert; a per-expert bias (adjusted outside backprop) only affects which 8 are chosen.",
    code: mlpCode,
    highlight: [
      "self.router = nn.Linear(n_embd, n_routed, bias=False)",
      'self.register_buffer("bias", torch.zeros(n_routed))',
      "scores = torch.sigmoid(self.router(x))",
      "_, idx = (scores + self.bias).topk(self.top_k, dim=-1)",
      "w = scores.gather(-1, idx)",
      "w = w / w.sum(dim=-1, keepdim=True)",
    ],
  },
  mlp_shared: {
    title: "Shared expert",
    summary: "One SwiGLU expert that every token uses.",
    code: mlpCode,
    highlight: ["self.shared = MLP(n_embd, n_shared * hidden)", "y = self.shared(x)"],
  },
  mlp_routed: {
    title: "Routed experts",
    summary: "256 small SwiGLU MLPs (hidden 2048); each runs only on the tokens that chose it.",
    code: mlpCode,
    highlight: [
      "self.experts = nn.ModuleList([MLP(n_embd, hidden) for _ in range(n_routed)])",
      "tok, slot = (idx == e).nonzero(as_tuple=True)",
      "expert(x[tok])",
    ],
  },
  mlp_moe_sum: {
    title: "Weighted sum",
    summary: "Shared output plus each chosen expert's output times its gate weight.",
    code: mlpCode,
    highlight: ["y = y.index_add(0, tok, w[tok, slot, None] * expert(x[tok]))"],
  },
  final_ln: {
    title: "Final RMSNorm",
    summary: "Normalize before the LM head.",
    code: deepseekTrainingCode,
    highlight: ["self.ln_f = RMSNorm(cfg[\"n_embd\"])", "logits = self.lm_head(self.ln_f(x))"],
  },
  lm_head: {
    title: "LM Head",
    summary: "Projection to 129,280 logits; its own weight, also used by the MTP module.",
    code: deepseekTrainingCode,
    highlight: ['self.lm_head = nn.Linear(cfg["n_embd"], cfg["vocab_size"], bias=False)', "logits = self.lm_head(self.ln_f(x))"],
  },
  loss_fn: {
    title: "Training Loss",
    summary: "Next-token cross-entropy, plus 0.3 × the MTP loss.",
    code: deepseekTrainingCode,
    highlight: ["loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))", "loss = loss + 0.3 * loss_mtp"],
  },
  mtp_loss: {
    title: "Multi-token prediction (MTP)",
    summary: "An extra block predicts token t+2 from the hidden state at t and the embedding of token t+1.",
    code: deepseekTrainingCode,
    highlight: [
      "class MTP(nn.Module):",
      "h2 = self.mtp(x[:, :-1], self.tok_emb(targets[:, :-1]))",
      "logits2 = self.lm_head(h2)",
      "loss_mtp = F.cross_entropy(logits2.reshape(-1, logits2.size(-1)), targets[:, 1:].reshape(-1))",
    ],
  },
  generate: {
    title: "Generate with an MLA KV cache",
    summary: "Inference: prefill the prompt, then one token per step against a cache of 576 numbers per token per layer; k and v are never computed.",
    code: generateCode,
    highlight: [
      "class MLACache:",
      "q_abs = q_nope @ W_uk",
      "y = y @ W_uv.transpose(-2, -1)",
      "def generate(model, idx, max_new_tokens, temperature=1.0, top_k=50):",
    ],
  },
};

for (const [id, entry] of Object.entries(content)) {
  for (const h of entry.highlight || []) {
    if (!entry.code.includes(h)) throw new Error(`DeepSeek content "${id}": highlight not found in its code: ${h}`);
  }
}
