# rusty-graph

Transformer building blocks and the model builder of the LLM stack in
[Rusty](../README.md): `LoRAAdapter` and linear layers with optional LoRA,
attention with rotary embeddings, SwiGLU MLP, RMSNorm, KV caches, flash
attention wrappers (`FlashAttention`, `GroupedQueryAttention`,
`MultiQueryAttention`, `SlidingWindowAttention`) and `Lfm2Model`, a
Llama/LFM2-style model assembled from a config and safetensors weights.
**Experimental.**

You normally reach this crate through the CLI, which loads a model directory
and attaches LoRA adapters:

```bash
cargo run -p rusty-cli --release -- infer --model ./models/tinyllama --prompt "Hello"
```

Model loading lives in [`rusty-loader`](../rusty-loader); training in
[`rusty-trainer`](../rusty-trainer).
