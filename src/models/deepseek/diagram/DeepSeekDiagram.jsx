import React from "react";
import Node from "../../../components/diagram/Node";
import Arrow from "../../../components/diagram/Arrow";
import { LINE } from "../../../components/diagram/palette";
import GPT2OuterFrame from "../../gpt2/diagram/GPT2OuterFrame";
import TransformerBlockGroup from "../../gpt2/diagram/TransformerBlockGroup";
import AttnSubDiagram from "../../gpt2/diagram/AttnSubDiagram";
import MoESubDiagram from "./MoESubDiagram";

// Multi-head latent attention: q and k/v go through small latents; RoPE on a separate part.
const ATTN_STEPS = [
  ["attn_linear_q", "Q: down → up"],
  ["attn_linear_kv_down", "KV → latent (cached)"],
  ["attn_linear_kv_up", "Latent → K, V"],
  ["attn_rope", "Decoupled RoPE"],
  ["attn_matmul_qk", "Matmul (Q × Kᵀ)"],
  ["attn_masked", "Masked Attention"],
  ["attn_softmax", "Softmax"],
  ["attn_matmul_v", "Matmul (att × V)"],
  ["attn_linear_out", "Linear"],
];

// Same layout as the LLaMA diagram (viewBox 0 0 836 848), with MLA, DeepSeekMoE and the MTP loss.
export default function DeepSeekDiagram({ onSelect }) {
  return (
    <>
      <GPT2OuterFrame onClick={onSelect} />

      <Node id="input_data" label="Input Data (Text)" x={20} y={50} w={180} h={40} onClick={onSelect} />
      <Arrow x1={110} y1={90} x2={110} y2={110} />
      <Node id="tokenizer" label="Tokenizer" x={20} y={110} w={180} h={40} onClick={onSelect} />

      <Arrow x1={110} y1={150} x2={110} y2={170} />
      <Node id="token_embed" label="Token Embeddings (no position embedding)" x={20} y={170} w={380} h={40} onClick={onSelect} />

      <Arrow x1={210} y1={210} x2={210} y2={300} />
      <TransformerBlockGroup x={20} y={300} onClick={onSelect} />
      <polyline points="215,290 500,290 500,455 220,455" stroke={LINE} strokeWidth={1.25} fill="none" markerEnd="url(#arrowhead)" />

      <Arrow x1={210} y1={590} x2={210} y2={620} />
      <Node id="final_ln" label="Final RMSNorm" x={20} y={620} w={380} h={40} onClick={onSelect} />

      <Arrow x1={210} y1={660} x2={210} y2={700} />
      <Node id="lm_head" label="LM Head (Linear, not tied)" x={20} y={700} w={380} h={40} onClick={onSelect} />
      <Node id="generate" label="Generate (KV cache)" x={415} y={700} w={175} h={40} onClick={onSelect} />

      <Arrow x1={210} y1={740} x2={210} y2={780} />
      <Node id="loss_fn" label="Training Objective (CrossEntropyLoss)" x={20} y={780} w={380} h={40} onClick={onSelect} />
      <text x={407} y={800} fontSize={16} fill="#64748B" textAnchor="middle" dominantBaseline="central">+</text>
      <Node id="mtp_loss" label="MTP loss (token t+2)" x={415} y={780} w={175} h={40} onClick={onSelect} />

      <MoESubDiagram x={600} y={540} onClick={onSelect} />
      <AttnSubDiagram x={600} y={40} onClick={onSelect} steps={ATTN_STEPS} gap={48} height={486} />
      <Arrow x1={400} y1={410} x2={600} y2={120} variant="detail" />
    </>
  );
}
