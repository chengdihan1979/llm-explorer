import gpt2ToLlama, { attention } from "./gpt2ToLlama";
import { relabel, replaceGroup } from "./scene";

// LLaMA 3 8B → DeepSeek-V3. Starts from the last scene of the GPT-2 → LLaMA transition.
// DeepSeek-V3 keeps LLaMA's skeleton (pre-norm RMSNorm, RoPE, SwiGLU experts, untied LM head) and changes
// the attention (MLA), the MLP (DeepSeekMoE), the load balancing and the training objective (MTP).
// Numbers from the DeepSeek-V3 technical report (arXiv:2412.19437); KV cache sizes computed for bf16.

const LLAMA3 = gpt2ToLlama.scenes[gpt2ToLlama.scenes.length - 1];

const TAIL = [
  ["attn_matmul_qk", "Matmul (Q × Kᵀ)"],
  ["attn_masked", "Masked Attention"],
  ["attn_softmax", "Softmax"],
  ["attn_matmul_v", "Matmul (att × V)"],
  ["attn_linear_out", "Linear"],
];

const MLA = [
  ["attn_linear_qkv", "Linear Q (128 heads)"],
  ["attn_kv_down", "K, V down → latent (cached)"],
  ["attn_kv_up", "Up-project K, V (128 heads)"],
  ["attn_rope", "RoPE (Q, K)"],
  ...TAIL,
];

// The MoE sub-diagram, in the MLP's place at (600, 530). Without the shared expert, the routed experts span the row.
function moe({ shared, router }) {
  const els = [
    { id: "mlp:frame", type: "frame", x: 600, y: 530, w: 220, h: 220, dashed: true },
    { id: "mlp:title", type: "title", x: 710, y: 554, label: "DeepSeekMoE" },
    { id: "mlp_router", type: "node", x: 610, y: 580, w: 200, h: 28, label: router, cat: "mlp" },
    { id: "mlp_moe_sum", type: "node", x: 620, y: 700, w: 180, h: 30, label: "Weighted sum", cat: "mlp" },
    { id: "mlp:link:routed>sum", type: "link", from: "mlp_routed", to: "mlp_moe_sum", x: shared ? 762 : 710 },
  ];
  if (shared) {
    els.push(
      { id: "mlp_shared", type: "node", x: 610, y: 640, w: 95, h: 30, label: "Shared × 1", cat: "mlp" },
      { id: "mlp_routed", type: "node", x: 715, y: 640, w: 95, h: 30, label: "Routed × 256", cat: "mlp" },
      { id: "mlp:link:router>routed", type: "link", from: "mlp_router", to: "mlp_routed", x: 762 },
      { id: "mlp:link:shared>sum", type: "link", from: "mlp_shared", to: "mlp_moe_sum", x: 657 },
    );
  } else {
    els.push(
      { id: "mlp_routed", type: "node", x: 610, y: 640, w: 200, h: 30, label: "Experts × 256, use top 8", cat: "mlp" },
      { id: "mlp:link:router>routed", type: "link", from: "mlp_router", to: "mlp_routed", x: 710 },
    );
  }
  return els;
}

const STEPS = [
  {
    title: { en: "LLaMA 3 8B (2024)", zh: "起点：LLaMA 3 8B（2024）" },
    body: {
      en: "Where the GPT-2 → LLaMA animation ended. DeepSeek-V3 (Dec 2024) keeps this skeleton: pre-norm RMSNorm blocks, RoPE, SwiGLU, no biases, an untied LM head. It changes the attention, the MLP and the training objective.",
      zh: "也就是 GPT-2 → LLaMA 动画的终点。DeepSeek-V3（2024 年 12 月）保留了这副骨架：pre-norm 的 RMSNorm block、RoPE、SwiGLU、没有 bias、LM head 不共享权重。它改的是 attention、MLP 和训练目标。",
    },
    page: "llama",
  },
  {
    title: { en: "MLA: cache a small latent", zh: "MLA：只缓存一个小的潜向量" },
    body: {
      en: "Multi-head latent attention. Each token's input is first compressed into one 512-number latent vector; the k and v of all 128 heads are projected back up from it. Only the latent goes into the KV cache, so there is no need to share k/v heads as in GQA.",
      zh: "多头潜在注意力（MLA）。每个 token 的输入先被压缩成一个 512 维的潜向量，128 个 head 的 k 和 v 都从它再投影回来。KV cache 里只存这个潜向量，所以不用像 GQA 那样让几个 head 共用 k、v。",
    },
    change: (s) => {
      replaceGroup(s, "attn:", attention(MLA, "Multi-head Latent Attention", 48));
      delete s.attn_repeat_kv;
      relabel(s, { block_attn: "Multi-head Latent Attention (MLA)" });
    },
    highlight: ["attn_linear_qkv", "attn_kv_down", "attn_kv_up", "block_attn", "attn:title"],
    note: "kv_cache_notes",
  },
  {
    title: { en: "Decoupled RoPE", zh: "解耦的 RoPE" },
    body: {
      en: "RoPE cannot be applied to keys rebuilt from the latent without breaking the trick that keeps inference cheap. So position goes into a separate 64-dimension part of q and k, with one rotated key shared by all heads and cached next to the latent: 512 + 64 numbers per token per layer.",
      zh: "如果对从潜向量还原出来的 k 直接做 RoPE，推理时省计算的那个技巧就用不了了。所以位置信息单独放在 q 和 k 的一个 64 维部分里，所有 head 共用一个转过的 key，和潜向量一起缓存：每个 token 每层存 512 + 64 个数。",
    },
    change: (s) => relabel(s, { attn_rope: "Decoupled RoPE (64-d part)" }),
    highlight: ["attn_rope"],
    note: "rope_notes",
  },
  {
    title: { en: "Mixture of experts", zh: "混合专家（MoE）" },
    body: {
      en: "The single MLP becomes 256 small SwiGLU experts. A router scores every expert for each token and sends the token to only the top 8. Total parameters grow a lot, but each token only pays for the 8 it uses. (The first 3 of the 61 layers keep a dense MLP.)",
      zh: "原来的一个 MLP 变成 256 个小的 SwiGLU 专家。router 给每个 token 的每个专家打分，只把 token 送到分数最高的 8 个。总参数大大增加，但每个 token 只用到这 8 个专家的计算量。（61 层里前 3 层仍然是普通的 MLP。）",
    },
    change: (s) => {
      replaceGroup(s, "mlp:", moe({ shared: false, router: "Router: top 8 of 256" }));
      for (const id of ["mlp_linear1", "mlp_linear_up", "mlp_gelu", "mlp_mul", "mlp_linear2"]) delete s[id];
      relabel(s, { block_mlp: "DeepSeekMoE" });
    },
    highlight: ["mlp_router", "mlp_routed", "mlp_moe_sum", "block_mlp", "mlp:title"],
  },
  {
    title: { en: "A shared expert", zh: "共享专家" },
    body: {
      en: "One more expert that every token always uses. It learns what all tokens need, so the 256 routed experts are free to specialize. Each routed expert is small (hidden size 2048), so the router can mix fine-grained pieces.",
      zh: "再加一个所有 token 都一定会用的专家。它负责学大家都需要的通用知识，256 个路由专家就可以各自专精。每个路由专家都很小（hidden size 2048），router 可以把很细的小块组合起来用。",
    },
    change: (s) => replaceGroup(s, "mlp:", moe({ shared: true, router: "Router: top 8 of 256" })),
    highlight: ["mlp_shared", "mlp_routed"],
  },
  {
    title: { en: "Load balancing without an auxiliary loss", zh: "不靠辅助 loss 的负载均衡" },
    body: {
      en: "If the router sends most tokens to a few experts, the others go untrained and the GPUs holding the busy ones become a bottleneck. Instead of relying on the usual extra loss term (only a very small one is kept), DeepSeek-V3 adds a bias to each expert's score when choosing the top 8, raising it for underused experts and lowering it for overloaded ones.",
      zh: "如果 router 把大部分 token 都送给少数几个专家，其他专家就练不到，放着热门专家的 GPU 也会成为瓶颈。DeepSeek-V3 不再主要靠常见的额外 loss 项（只保留了一个很小的），而是在挑前 8 个专家时，给每个专家的分数加一个偏置：用得少的调高，用得太多的调低。",
    },
    change: (s) => replaceGroup(s, "mlp:", moe({ shared: true, router: "Router + balancing bias" })),
    highlight: ["mlp_router"],
  },
  {
    title: { en: "Multi-token prediction", zh: "多 token 预测（MTP）" },
    body: {
      en: "In training, an extra module also predicts the token after next, from the main model's last hidden state and the next token's embedding. Its loss is added to the usual one. It gives a denser training signal; at inference it can be dropped, or used to draft tokens for speculative decoding.",
      zh: "训练时，多加一个模块，用主模型最后的隐藏状态和下一个 token 的 embedding，再预测下下个 token。它的 loss 加到原来的 loss 上，让训练信号更密。推理时可以直接去掉，也可以用来先猜 token，做投机解码（speculative decoding）。",
    },
    change: (s) => {
      s.mtp_plus = { id: "mtp_plus", type: "text", x: 407, y: 800, label: "+", size: 16, color: "#64748B" };
      s.mtp_loss = { id: "mtp_loss", type: "node", x: 415, y: 780, w: 175, h: 40, label: "MTP loss (token t+2)", cat: "loss" };
    },
    highlight: ["mtp_loss", "loss_fn"],
  },
  {
    title: { en: "DeepSeek-V3", zh: "DeepSeek-V3" },
    body: {
      en: "61 blocks, width 7168, 671B parameters of which 37B are used per token, a 128K context and a 128K vocabulary. The KV cache is about 69 KB per token, less than LLaMA 3 8B's 128 KB, for a model with 84 times as many parameters.",
      zh: "61 个 block，宽度 7168，6710 亿参数，每个 token 只用其中 370 亿；128K 上下文，128K 词表。KV cache 每个 token 约 69 KB，比 LLaMA 3 8B 的 128 KB 还小，而模型参数是它的 84 倍。",
    },
    change: (s) => relabel(s, { model_title: "DeepSeek-V3 · 671B (37B active)", block_title: "Transformer Block × 61" }),
    highlight: ["model_title", "block_title"],
    page: "deepseek",
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
  id: "llama-deepseek",
  from: "LLaMA 3",
  to: "DeepSeek-V3",
  viewBox: gpt2ToLlama.viewBox,
  steps: STEPS,
  scenes: buildScenes(LLAMA3, STEPS),
};
