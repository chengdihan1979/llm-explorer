import llama from "../llama";
import { attnCode, blockCode, content, mistralTrainingCode, mlpCode } from "./snippets";

// Mistral 7B reuses the LLaMA diagram: same blocks, with the attention mask relabeled as a sliding window.
// 🆕 marks what changed compared with LLaMA.
export default {
  id: "mistral",
  name: "Mistral",
  badge: "M7",
  diagram: {
    ...llama.diagram,
    label: "Mistral architecture diagram",
    spec: {
      labels: {
        ...llama.diagram.spec.labels,
        gpt2: "Mistral 7B",
        block_attn: "Sliding-Window GQA",
        attn_class: "Sliding-Window GQA",
        attn_masked: "Sliding Window",
      },
      newIds: new Set(["block_attn", "attn_masked"]),
    },
  },
  defaultSelected: "attn_masked",
  content,
  resolveCodeId: (id) => id,
  noteOverrides: {
    ...llama.noteOverrides,
    causal_analogy_notes: "sliding_window_notes",
  },
  fileNames: new Map([
    [mistralTrainingCode, "train_mistral.py"],
    [blockCode, "block.py"],
    [attnCode, "attention.py"],
    [mlpCode, "mlp.py"],
  ]),
};
