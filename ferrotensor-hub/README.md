# ferrotensor-hub

Hugging Face model-cache helpers for [Ferrotensor](../README.md): `FerrotensorHub`
finds a model in the local Hugging Face cache (`HF_HUB_CACHE`, else
`$HF_HOME/hub`, else `$XDG_CACHE_HOME/huggingface/hub`, else
`~/.cache/huggingface/hub`) and downloads nothing, `ModelConfig::from_model_dir`
reads a model's `config.json`, and `ModelType` identifies the architecture. The
CLI uses it to turn `--model org/name` into the cached snapshot directory.

```rust,ignore
use ferrotensor_hub::{ModelConfig, FerrotensorHub};

let hub = FerrotensorHub::new();
let model_path = hub.get_model_path("HuggingFaceTB/SmolLM2-135M");
let config = ModelConfig::from_model_dir(&model_path)?;
```

The snippet needs a downloaded model directory (see
`scripts/download_model.sh` in the repository root), so it is not run in CI.
