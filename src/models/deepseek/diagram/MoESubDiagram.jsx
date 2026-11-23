import React from "react";
import Node from "../../../components/diagram/Node";
import Arrow from "../../../components/diagram/Arrow";
import FrameTitle from "../../../components/diagram/FrameTitle";

// DeepSeekMoE: a router picks 8 of 256 routed experts per token; one shared expert is always used;
// the outputs are added up with the router's weights.
export default function MoESubDiagram({ x, y, onClick }) {
  return (
    <g>
      <rect x={x} y={y} width={220} height={220} rx={12} fill="#F8FAFC" stroke="#C7D2FE" strokeDasharray="5 4" />
      <FrameTitle id="mlp_subdiagram" label="DeepSeekMoE" x={x + 110} y={y + 24} onClick={onClick} />

      <Node id="mlp_router" label="Router + balancing bias" x={x + 10} y={y + 50} w={200} h={28} onClick={onClick} />
      <Arrow x1={x + 162.5} y1={y + 78} x2={x + 162.5} y2={y + 110} />

      <Node id="mlp_shared" label="Shared × 1" x={x + 10} y={y + 110} w={95} h={30} onClick={onClick} />
      <Node id="mlp_routed" label="Routed × 256" x={x + 115} y={y + 110} w={95} h={30} onClick={onClick} />

      <Arrow x1={x + 57.5} y1={y + 140} x2={x + 57.5} y2={y + 170} />
      <Arrow x1={x + 162.5} y1={y + 140} x2={x + 162.5} y2={y + 170} />
      <Node id="mlp_moe_sum" label="Weighted sum" x={x + 20} y={y + 170} w={180} h={30} onClick={onClick} />
    </g>
  );
}
