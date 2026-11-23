import DeepSeekDiagram from "./diagram/DeepSeekDiagram";
import { attnCode, blockCode, content, deepseekTrainingCode, mlpCode } from "./snippets";

// DeepSeek-V3. Shared groups are relabeled through the diagram spec; 🆕 marks what changed compared with LLaMA.
export default {
  id: "deepseek",
  name: "DeepSeek",
  badge: "V3",
  diagram: {
    Component: DeepSeekDiagram,
    viewBox: "0 0 836 848",
    label: "DeepSeek-V3 architecture diagram",
    spec: {
      labels: {
        gpt2: "DeepSeek-V3 · 671B (37B active)",
        stack: "Transformer Block × 61",
        block_ln: "RMSNorm",
        block_ln2: "RMSNorm",
        block_attn: "Multi-head Latent Attention (MLA)",
        block_mlp: "DeepSeekMoE (first 3 blocks: dense MLP)",
        attn_class: "Multi-head Latent Attention",
        final_ln: "Final RMSNorm",
      },
      // only the nodes wide enough for the badge; the whole MoE frame and MLA frame are new anyway
      newIds: new Set(["block_attn", "block_mlp", "attn_linear_kv_down", "attn_linear_kv_up", "attn_rope", "mlp_router", "mtp_loss"]),
    },
  },
  defaultSelected: "attn_linear_kv_down",
  content,
  resolveCodeId: (id) => id,
  noteOverrides: {
    embedding_tok: "llama_embedding_notes",
    lm_head_notes: "llama_lm_head_notes",
    causal_analogy_notes: "kv_cache_notes",
    final_ln_notes: "llama_final_norm_notes",
  },
  fileNames: new Map([
    [deepseekTrainingCode, "train_deepseek_v3.py"],
    [blockCode, "block.py"],
    [attnCode, "attention.py"],
    [mlpCode, "moe.py"],
  ]),
};
