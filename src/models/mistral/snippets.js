// Mistral 7B reference code. Mistral 7B has the same shape as LLaMA 3 8B (32 blocks, width 4096, 32 query / 8 key-value
// heads, SwiGLU hidden 14336, RMSNorm, no biases, untied LM head), so its code is derived from the LLaMA code by
// replacing exactly the parts that differ: a 32,000-token vocabulary, RoPE base 10000, and sliding-window attention.
// Every replaced string must exist, so a change to the LLaMA code that breaks a replacement fails at load time.
// Verified in PyTorch at a small size: the window mask; at full size the model has 7.24B parameters.
import {
  attnCode as llamaAttnCode,
  blockCode as llamaBlockCode,
  content as llamaContent,
  llamaTrainingCode,
  mlpCode,
} from "../llama/snippets";

function swap(code, pairs, where) {
  for (const [from, to] of pairs) {
    if (!code.includes(from)) throw new Error(`Mistral ${where}: text to replace not found: ${from}`);
    code = code.split(from).join(to);
  }
  return code;
}

const TRAINING_SWAPS = [
  [
    'tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")  # BPE, 128,256 tokens',
    'tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")  # BPE, 32,000 tokens',
  ],
  ["class LLaMA(nn.Module):", "class Mistral(nn.Module):"],
  [
    "def __init__(self, vocab_size, n_layer, n_embd, n_head, n_kv_head, hidden, max_toks):",
    "def __init__(self, vocab_size, n_layer, n_embd, n_head, n_kv_head, hidden, max_toks, window):",
  ],
  ["# V = vocab_size = 128256, C = n_embd = 4096", "# V = vocab_size = 32000, C = n_embd = 4096"],
  [
    "TransformerBlock(n_embd=n_embd, n_head=n_head, n_kv_head=n_kv_head, hidden=hidden, max_toks=max_toks)",
    "TransformerBlock(n_embd=n_embd, n_head=n_head, n_kv_head=n_kv_head, hidden=hidden, max_toks=max_toks, window=window)",
  ],
  ["# B = batch, T = sequence length, C = 4096, V = 128256", "# B = batch, T = sequence length, C = 4096, V = 32000"],
  [
    "# LLaMA 3 8B\nmodel = LLaMA(vocab_size=128256, n_layer=32, n_embd=4096, n_head=32, n_kv_head=8, hidden=14336, max_toks=8192).to(device)",
    "# Mistral 7B\nmodel = Mistral(vocab_size=32000, n_layer=32, n_embd=4096, n_head=32, n_kv_head=8, hidden=14336, max_toks=8192, window=4096).to(device)",
  ],
  ["LLaMA has no biases", "Mistral has no biases"],
  ["logits (4, 32, 128256)", "logits (4, 32, 32000)"],
];

const BLOCK_SWAPS = [
  ["def __init__(self, n_embd, n_head, n_kv_head, hidden, max_toks):", "def __init__(self, n_embd, n_head, n_kv_head, hidden, max_toks, window):"],
  [
    "CausalSelfAttention(n_embd=n_embd, n_head=n_head, n_kv_head=n_kv_head, max_toks=max_toks)",
    "CausalSelfAttention(n_embd=n_embd, n_head=n_head, n_kv_head=n_kv_head, max_toks=max_toks, window=window)",
  ],
];

const ATTN_SWAPS = [
  ["def rope_cache(hs, max_toks, theta=500000.0):", "def rope_cache(hs, max_toks, theta=10000.0):"],
  ["def __init__(self, n_embd, n_head, n_kv_head, max_toks):", "def __init__(self, n_embd, n_head, n_kv_head, max_toks, window):"],
  [
    "        self.n_kv_head = n_kv_head                   # 8 key/value heads (GQA)\n",
    "        self.n_kv_head = n_kv_head                   # 8 key/value heads (GQA)\n" +
      "        self.window = window                         # 4096: each token sees at most the last 4096 tokens\n",
  ],
  [
    "        mask = torch.tril(torch.ones(T, T, device=x.device))                # (T, T)\n",
    "        i = torch.arange(T, device=x.device)\n" +
      "        dist = i[:, None] - i[None, :]                                      # (T, T): query pos - key pos\n" +
      "        mask = (dist >= 0) & (dist < self.window)                           # causal, and within the window\n",
  ],
];

export const mistralTrainingCode = swap(llamaTrainingCode, TRAINING_SWAPS, "training code");
export const blockCode = swap(llamaBlockCode, BLOCK_SWAPS, "block code");
export const attnCode = swap(llamaAttnCode, ATTN_SWAPS, "attention code");
export { mlpCode };

const CODE_MAP = new Map([
  [llamaTrainingCode, mistralTrainingCode],
  [llamaBlockCode, blockCode],
  [llamaAttnCode, attnCode],
  [mlpCode, mlpCode],
]);
const ALL_SWAPS = [...TRAINING_SWAPS, ...BLOCK_SWAPS, ...ATTN_SWAPS];
// Highlight strings follow the same replacements (only where they apply).
const swapHighlight = (h) => ALL_SWAPS.reduce((s, [from, to]) => (s.includes(from) ? s.split(from).join(to) : s), h);

// Start from LLaMA's entries (same diagram ids), point them at the Mistral code, then override what differs.
const base = Object.fromEntries(
  Object.entries(llamaContent).map(([id, e]) => [
    id,
    { ...e, code: CODE_MAP.get(e.code), highlight: e.highlight?.map(swapHighlight) },
  ]),
);

export const content = {
  ...base,
  gpt2: {
    ...base.gpt2,
    title: "Mistral (model & training)",
    summary: "Mistral 7B: LLaMA's design, with sliding-window attention (W = 4096).",
    highlight: ["class Mistral(nn.Module):"],
  },
  tokenizer: {
    ...base.tokenizer,
    summary: "SentencePiece BPE with 32,000 tokens (LLaMA 3: 128,256).",
    highlight: ['tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")', "ids = tokenizer.encode(text)"],
  },
  stack: { ...base.stack, summary: "Pre-norm block with RMSNorm, sliding-window GQA and a SwiGLU MLP." },
  block_attn: {
    ...base.block_attn,
    title: "Sliding-Window GQA",
    summary: "Grouped-query attention where each token sees at most the last 4096 tokens.",
  },
  attn_class: {
    ...base.attn_class,
    title: "Sliding-Window GQA",
    summary: "Causal self-attention with RoPE, shared key/value heads and a sliding window.",
  },
  attn_rope: { ...base.attn_rope, summary: "RoPE as in LLaMA, with base 10000 (LLaMA 3: 500000)." },
  attn_masked: {
    title: "Sliding-Window Mask",
    summary: "Causal, and each token sees only the last 4096 tokens (itself included).",
    code: attnCode,
    highlight: [
      "self.window = window",
      "dist = i[:, None] - i[None, :]",
      "mask = (dist >= 0) & (dist < self.window)",
      "att = att.masked_fill(mask == 0, float('-inf'))",
    ],
  },
};

for (const [id, entry] of Object.entries(content)) {
  if (!entry.code) throw new Error(`Mistral content "${id}": no code`);
  for (const h of entry.highlight || []) {
    if (!entry.code.includes(h)) throw new Error(`Mistral content "${id}": highlight not found in its code: ${h}`);
  }
}
