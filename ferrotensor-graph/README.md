# ferrotensor-graph

Transformer building blocks and the model builder of the LLM stack in
[Ferrotensor](../README.md): `LoRAAdapter` and linear layers with optional LoRA,
attention with rotary embeddings, SwiGLU MLP, RMSNorm, KV caches, flash
attention wrappers (`FlashAttention`, `GroupedQueryAttention`,
`MultiQueryAttention`, `SlidingWindowAttention`) and `Lfm2Model`, a
Llama/LFM2-style model assembled from a config and safetensors weights.
**Experimental.**

You normally reach this crate through the CLI, which loads a model directory
and attaches LoRA adapters:

```bash
cargo run -p ferrotensor-cli --release -- infer --model ./models/tinyllama --prompt "Hello"
```

Model loading lives in [`ferrotensor-loader`](../ferrotensor-loader); training in
[`ferrotensor-trainer`](../ferrotensor-trainer).
