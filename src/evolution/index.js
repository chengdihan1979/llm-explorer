import gpt2ToLlama from "./gpt2ToLlama";
import llamaToMistral from "./llamaToMistral";
import llamaToDeepseek from "./llamaToDeepseek";

// Model-to-model transitions shown on the home page, in order. Add one file per new pair (e.g. LLaMA → Mistral).
export const TRANSITIONS = [gpt2ToLlama, llamaToMistral, llamaToDeepseek];
