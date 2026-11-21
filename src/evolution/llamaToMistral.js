import gpt2ToLlama from "./gpt2ToLlama";
import { relabel } from "./scene";

// LLaMA 3 8B → Mistral 7B. Starts from the last scene of the GPT-2 → LLaMA transition, so the two play as one story.
// Mistral 7B (Sep 2023) is older than LLaMA 3 (Apr 2024); the first step says so.

const LLAMA3 = gpt2ToLlama.scenes[gpt2ToLlama.scenes.length - 1];

const STEPS = [
  {
    title: { en: "LLaMA 3 8B (2024)", zh: "起点：LLaMA 3 8B（2024）" },
    body: {
      en: "Where the GPT-2 → LLaMA animation ended. Mistral 7B (Sep 2023) is actually older than LLaMA 3, but it has the same shape: 32 blocks, width 4096, 32 query and 8 key/value heads, SwiGLU with hidden size 14336. Only a few parts differ.",
      zh: "也就是 GPT-2 → LLaMA 动画的终点。Mistral 7B（2023 年 9 月）其实比 LLaMA 3 还早，但形状一样：32 个 block，宽度 4096，32 个 query head 和 8 个 key/value head，hidden size 14336 的 SwiGLU。不同的地方只有几处。",
    },
    page: "llama",
  },
  {
    title: { en: "Sliding-window attention", zh: "滑动窗口注意力" },
    body: {
      en: "Only the mask changes: on top of the causal mask, each token may attend to at most the last 4096 tokens, itself included. Past 4096 tokens, the attention cost per token stops growing.",
      zh: "只有 mask 变了：在 causal mask 的基础上，每个 token 最多只看最近的 4096 个 token（包括自己）。超过 4096 个 token 以后，每个 token 的 attention 计算量就不再增长。",
    },
    change: (s) => {
      relabel(s, { attn_masked: "Sliding-Window Mask (4096)", "attn:title": "Sliding-Window GQA", block_attn: "Sliding-Window GQA" });
    },
    highlight: ["attn_masked", "block_attn", "attn:title"],
    note: "sliding_window_notes",
  },
  {
    title: { en: "A rolling KV cache", zh: "循环使用的 KV cache" },
    body: {
      en: "When generating, no token looks more than 4095 positions back, so the KV cache keeps only the last 4096 tokens: a fixed buffer where token i goes into slot i mod 4096. At most 512 MB per sequence, however long the text.",
      zh: "生成时，没有哪个 token 会往前看超过 4095 个位置，所以 KV cache 只留最近 4096 个 token：一个固定大小的缓冲区，token i 写进第 i mod 4096 个槽。每条序列最多 512 MB，不管文本多长。",
    },
    change: (s) => {
      s.attn_kv_cache = { id: "attn_kv_cache", type: "node", x: 415, y: 620, w: 175, h: 40, label: "Rolling KV Cache · 4096" };
      // dashed connector from the attention frame, whose k and v it stores
      s.attn_kv_cache_link = { id: "attn_kv_cache_link", type: "line", x1: 600, y1: 505, x2: 560, y2: 616, variant: "detail" };
    },
    highlight: ["attn_kv_cache"],
    note: "sliding_window_notes",
  },
  {
    title: { en: "A smaller vocabulary", zh: "词表变小" },
    body: {
      en: "32,000 tokens, a SentencePiece BPE tokenizer like LLaMA 2's, instead of LLaMA 3's 128,256. The embedding table and the LM head shrink from 525M to 131M parameters each.",
      zh: "词表是 32,000 个 token，和 LLaMA 2 一样用 SentencePiece 的 BPE tokenizer，而 LLaMA 3 是 128,256 个。embedding 表和 LM head 各自从 5.25 亿参数缩到 1.31 亿。",
    },
    change: (s) => relabel(s, { tokenizer: "Tokenizer · 32K vocab" }),
    highlight: ["tokenizer", "token_embed", "lm_head"],
    note: "mistral_notes",
  },
  {
    title: { en: "RoPE base 10,000", zh: "RoPE base 是 10,000" },
    body: {
      en: "Mistral 7B keeps RoPE's original base of 10,000. LLaMA 3 raised it to 500,000 so that the slow dimension pairs turn less within long contexts.",
      zh: "Mistral 7B 沿用 RoPE 最初的 base 10,000。LLaMA 3 把它提高到 500,000，让转得慢的那些维度对在长文本里转得更少。",
    },
    change: (s) => relabel(s, { attn_rope: "RoPE (Q, K) · base 10K" }),
    highlight: ["attn_rope"],
    note: "rope_notes",
  },
  {
    title: { en: "Mistral 7B", zh: "Mistral 7B" },
    body: {
      en: "7.24B parameters, an 8K context, and a theoretical attention span of about 131K tokens through 32 stacked windows. The paper reports it beating LLaMA 2 13B on every benchmark tested.",
      zh: "72.4 亿参数，8K 上下文；32 层窗口叠起来，理论注意力范围约 131K 个 token。论文里它在所有测试项目上都超过了 LLaMA 2 13B。",
    },
    change: (s) => relabel(s, { model_title: "Mistral 7B" }),
    highlight: ["model_title"],
    note: "mistral_notes",
    page: "mistral",
  },
];

function buildScenes(base, steps) {
  const out = [base];
  for (const step of steps.slice(1)) {
    const next = { ...out[out.length - 1] };
    if (step.change) step.change(next);
    out.push(next);
  }
  return out;
}

export default {
  id: "llama-mistral",
  from: "LLaMA 3",
  to: "Mistral 7B",
  viewBox: gpt2ToLlama.viewBox,
  steps: STEPS,
  scenes: buildScenes(LLAMA3, STEPS),
};
