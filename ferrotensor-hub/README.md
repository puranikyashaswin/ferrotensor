# ferrotensor-hub

Hugging Face model-directory helpers for [Ferrotensor](../README.md): `FerrotensorHub`
resolves a local model directory, `ModelConfig::from_model_dir` reads its
`config.json`, and `ModelType` identifies the architecture.

```rust,ignore
use ferrotensor_hub::{ModelConfig, FerrotensorHub};

let hub = FerrotensorHub::new();
let model_path = hub.get_model_path("meta-llama/Llama-3.2-1B");
let config = ModelConfig::from_model_dir(&model_path)?;
```

The snippet needs a downloaded model directory (see
`scripts/download_model.sh` in the repository root), so it is not run in CI.
