//! Model builder: constructs an [`Lfm2Model`] from a [`WeightsContext`].
//!
//! Loads Hugging Face safetensors checkpoints of two families into the same
//! decoder structure, with optional LoRA adapters:
//!
//! * **Llama-style** (`model_type` `llama`/`mistral`, e.g. SmolLM2,
//!   TinyLlama): every block is RoPE grouped-query attention plus a SwiGLU
//!   MLP; weights named `self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj`,
//!   `input_layernorm`, `post_attention_layernorm` and `model.norm`.
//! * **LFM2** (`layer_types` present): blocks are short convolutions or
//!   attention, named `operator_norm`, `ffn_norm`, `feed_forward.w{1,2,3}`,
//!   with the final norm stored as `model.embedding_norm`.
//!
//! Base weights are frozen ([`Tensor::frozen`]) unless
//! [`Lfm2ModelBuilder::with_trainable_base`] is set, so LoRA training skips
//! their gradients.

use crate::lfm::{Lfm2Block, Lfm2Conv, Lfm2LayerOp, Lfm2Model};
use crate::{Attention, Linear, LoRAAdapter, RMSNorm, MLP};
use ferrotensor_autograd::Tensor;
use ferrotensor_backend::{ComputeEngine, UnifiedTensor, WgpuContext};
use ferrotensor_loader::{Lfm2Config, WeightsContext};
use std::sync::Arc;

/// Configuration for LoRA adapters
pub struct LoRaConfig {
    /// Adapter rank.
    pub rank: usize,
    /// Scale numerator (`alpha / rank`).
    pub alpha: f32,
    /// Projection names that get adapters, e.g. `["q_proj", "v_proj"]`.
    pub target_modules: Vec<String>, // e.g., ["q_proj", "v_proj"]
}

impl Default for LoRaConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 1.0,
            target_modules: vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
                "o_proj".to_string(),
                "in_proj".to_string(),
                "out_proj".to_string(),
            ],
        }
    }
}

/// Error returned when a checkpoint cannot be turned into a model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildError(pub String);

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for BuildError {}

type Result<T> = std::result::Result<T, BuildError>;

/// Weight-naming family of a checkpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Architecture {
    /// Llama / Mistral / SmolLM2 naming; every block is attention.
    Llama,
    /// Liquid LFM2 naming; conv and attention blocks per `layer_types`.
    Lfm2,
}

impl Architecture {
    /// Infer the family from a config: `layer_types` or `model_type: lfm2`
    /// mean LFM2, everything else is treated as Llama.
    pub fn detect(config: &Lfm2Config) -> Self {
        let lfm = config
            .model_type
            .as_deref()
            .is_some_and(|t| t.eq_ignore_ascii_case("lfm2"));
        if lfm || !config.layer_types.is_empty() {
            Self::Lfm2
        } else {
            Self::Llama
        }
    }
}

/// Builds an [`Lfm2Model`] from loaded weights.
pub struct Lfm2ModelBuilder<'a> {
    ctx: Arc<WgpuContext>,
    engine: Arc<ComputeEngine>,
    weights: &'a WeightsContext,
    config: Lfm2Config,
    lora_config: Option<LoRaConfig>,
    trainable_base: bool,
}

impl<'a> Lfm2ModelBuilder<'a> {
    fn transpose_2d(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut transposed = vec![0.0; data.len()];
        for row in 0..rows {
            for col in 0..cols {
                transposed[col * rows + row] = data[row * cols + col];
            }
        }
        transposed
    }

    /// A builder for `config` reading tensors from `weights`.
    pub fn new(
        ctx: Arc<WgpuContext>,
        engine: Arc<ComputeEngine>,
        weights: &'a WeightsContext,
        config: Lfm2Config,
    ) -> Self {
        Self {
            ctx,
            engine,
            weights,
            config,
            lora_config: None,
            trainable_base: false,
        }
    }

    /// Attach LoRA adapters to the projections named in `lora_config`.
    pub fn with_lora(mut self, lora_config: LoRaConfig) -> Self {
        self.lora_config = Some(lora_config);
        self
    }

    /// Let the base weights receive gradients (full or partial fine-tuning).
    /// By default they are frozen and only LoRA adapters train.
    pub fn with_trainable_base(mut self, trainable: bool) -> Self {
        self.trainable_base = trainable;
        self
    }

    fn wrap(&self, data: UnifiedTensor) -> Tensor {
        if self.trainable_base {
            Tensor::new(self.ctx.clone(), data)
        } else {
            Tensor::frozen(self.ctx.clone(), data)
        }
    }

    fn has(&self, name: &str) -> bool {
        self.weights.get_tensor_info(name).is_some()
    }

    fn read(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        self.weights
            .read_f32_tensor(name)
            .map_err(|e| BuildError(format!("failed to load weight {name}: {e}")))
    }

    /// Load a tensor from weights, dequantizing if needed
    fn load_tensor(&self, name: &str) -> Result<Tensor> {
        if self.has(&format!("{name}.scales")) {
            let ut = self
                .weights
                .upload_quantized(name, &self.ctx, &self.engine)
                .map_err(|e| BuildError(format!("failed to load quantized weight {name}: {e}")))?;
            Ok(self.wrap(ut))
        } else {
            self.load_raw_tensor(&format!("{name}.weight"))
        }
    }

    fn load_raw_tensor(&self, name: &str) -> Result<Tensor> {
        let (data, shape) = self.read(name)?;
        Ok(self.wrap(UnifiedTensor::new(&self.ctx, &data, &shape)))
    }

    /// Load a PyTorch `[out, in]` weight transposed to `[in, out]`.
    fn load_linear_weight_tensor(&self, prefix: &str) -> Result<Tensor> {
        let (data, shape) = self.read(&format!("{prefix}.weight"))?;
        let (data, shape) = if let [rows, cols] = shape[..] {
            (Self::transpose_2d(&data, rows, cols), vec![cols, rows])
        } else {
            (data, shape)
        };
        Ok(self.wrap(UnifiedTensor::new(&self.ctx, &data, &shape)))
    }

    fn maybe_add_lora(&self, linear: Linear, module_name: &str, seed: u64) -> Linear {
        if let Some(lora_cfg) = &self.lora_config {
            if lora_cfg
                .target_modules
                .iter()
                .any(|m| module_name.contains(m.as_str()))
            {
                let shape = &linear.weight.data.shape;
                let lora = LoRAAdapter::new_seeded(
                    self.ctx.clone(),
                    shape[0],
                    shape[1],
                    lora_cfg.rank,
                    lora_cfg.alpha,
                    seed,
                );
                return Linear::with_lora(linear.weight, lora);
            }
        }
        linear
    }

    /// `module_name` is matched against the LoRA target modules; `seed`
    /// keeps adapters of equal shape distinct.
    fn build_linear(&self, prefix: &str, module_name: &str, seed: u64) -> Result<Linear> {
        let weight = if self.has(&format!("{prefix}.scales")) {
            self.load_tensor(prefix)?
        } else {
            self.load_linear_weight_tensor(prefix)?
        };
        Ok(self.maybe_add_lora(Linear::new(weight), module_name, seed))
    }

    fn build_rms_norm(&self, prefix: &str) -> Result<RMSNorm> {
        let weight = self.load_raw_tensor(&format!("{prefix}.weight"))?;
        Ok(RMSNorm::new(weight, self.config.norm_eps))
    }

    fn build_mlp(&self, prefix: &str, arch: Architecture, seed: u64) -> Result<MLP> {
        // LFM2 names its SwiGLU projections w1 (gate), w3 (up), w2 (down).
        let [gate, up, down] = match arch {
            Architecture::Llama => ["gate_proj", "up_proj", "down_proj"],
            Architecture::Lfm2 => ["w1", "w3", "w2"],
        };
        Ok(MLP {
            gate_proj: self.build_linear(&format!("{prefix}.{gate}"), gate, seed)?,
            up_proj: self.build_linear(&format!("{prefix}.{up}"), up, seed)?,
            down_proj: self.build_linear(&format!("{prefix}.{down}"), down, seed)?,
        })
    }

    fn build_attention(&self, prefix: &str, arch: Architecture, seed: u64) -> Result<Attention> {
        let n_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads.unwrap_or(n_heads);
        let head_dim = self
            .config
            .head_dim
            .unwrap_or(self.config.hidden_size / n_heads.max(1));
        if n_heads == 0 || num_kv_heads == 0 || !n_heads.is_multiple_of(num_kv_heads) {
            return Err(BuildError(format!(
                "num_attention_heads ({n_heads}) must be a positive multiple of num_key_value_heads ({num_kv_heads})"
            )));
        }
        if head_dim > 128 {
            return Err(BuildError(format!(
                "head_dim {head_dim} is not supported (the attention kernel handles at most 128)"
            )));
        }
        if self.has(&format!("{prefix}.q_proj.bias")) {
            return Err(BuildError(
                "attention projections with bias (e.g. Qwen2) are not supported yet".into(),
            ));
        }
        if self.has(&format!("{prefix}.q_layernorm.weight")) {
            return Err(BuildError(
                "LFM2 per-head q/k layernorm is not supported yet".into(),
            ));
        }
        let o_name = match arch {
            Architecture::Llama => "o_proj",
            Architecture::Lfm2 => "out_proj",
        };

        Ok(Attention {
            q_proj: self.build_linear(&format!("{prefix}.q_proj"), "q_proj", seed)?,
            k_proj: self.build_linear(&format!("{prefix}.k_proj"), "k_proj", seed)?,
            v_proj: self.build_linear(&format!("{prefix}.v_proj"), "v_proj", seed)?,
            o_proj: self.build_linear(&format!("{prefix}.{o_name}"), o_name, seed)?,
            n_heads,
            head_dim,
            num_kv_heads,
            dropout_prob: self.config.attention_dropout.unwrap_or(0.0),
            rope_theta: self.config.rope_theta.unwrap_or(10000.0) as f32,
        })
    }

    fn build_conv(&self, prefix: &str, seed: u64) -> Result<Lfm2Conv> {
        let conv_weight = self.load_raw_tensor(&format!("{prefix}.conv.weight"))?;
        // LFM uses no conv bias typically, create zero tensor
        let conv_bias = self.wrap(UnifiedTensor::zeros(&self.ctx, &[self.config.hidden_size]));

        Ok(Lfm2Conv {
            in_proj: self.build_linear(&format!("{prefix}.in_proj"), "in_proj", seed)?,
            conv_weight,
            conv_bias,
            out_proj: self.build_linear(&format!("{prefix}.out_proj"), "out_proj", seed)?,
        })
    }

    fn build_layer(&self, layer_idx: usize, arch: Architecture) -> Result<Lfm2Block> {
        let prefix = format!("model.layers.{layer_idx}");
        let seed = layer_idx as u64 + 1;
        match arch {
            Architecture::Llama => Ok(Lfm2Block {
                operator_norm: self.build_rms_norm(&format!("{prefix}.input_layernorm"))?,
                op: Lfm2LayerOp::Attention(self.build_attention(
                    &format!("{prefix}.self_attn"),
                    arch,
                    seed,
                )?),
                ffn_norm: self.build_rms_norm(&format!("{prefix}.post_attention_layernorm"))?,
                mlp: self.build_mlp(&format!("{prefix}.mlp"), arch, seed)?,
            }),
            Architecture::Lfm2 => {
                let layer_type = self
                    .config
                    .layer_types
                    .get(layer_idx)
                    .map(String::as_str)
                    .unwrap_or("full_attention");
                let op = match layer_type {
                    "conv" => Lfm2LayerOp::Conv(self.build_conv(&format!("{prefix}.conv"), seed)?),
                    "full_attention" | "attention" => Lfm2LayerOp::Attention(
                        self.build_attention(&format!("{prefix}.self_attn"), arch, seed)?,
                    ),
                    other => {
                        return Err(BuildError(format!(
                            "unknown layer type {other:?} at layer {layer_idx}"
                        )))
                    }
                };
                Ok(Lfm2Block {
                    operator_norm: self.build_rms_norm(&format!("{prefix}.operator_norm"))?,
                    op,
                    ffn_norm: self.build_rms_norm(&format!("{prefix}.ffn_norm"))?,
                    mlp: self.build_mlp(&format!("{prefix}.feed_forward"), arch, seed)?,
                })
            }
        }
    }

    /// Build the complete model, or explain which weight or setting is
    /// missing or unsupported.
    pub fn build(self) -> Result<Lfm2Model> {
        let arch = Architecture::detect(&self.config);
        let n_layers = self.config.num_hidden_layers;
        log::info!("building {arch:?} model with {n_layers} layers");

        let embed_tokens = self.load_tensor("model.embed_tokens")?;
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            layers.push(self.build_layer(i, arch)?);
        }

        // Hugging Face LFM2 applies `embedding_norm` after the last block.
        let norm_f = if self.has("model.norm.weight") {
            self.build_rms_norm("model.norm")?
        } else if self.has("model.embedding_norm.weight") {
            self.build_rms_norm("model.embedding_norm")?
        } else {
            return Err(BuildError(
                "no final norm (model.norm.weight or model.embedding_norm.weight)".into(),
            ));
        };

        let lm_head = if self.has("lm_head.weight") {
            self.build_linear("lm_head", "lm_head", 0)?
        } else {
            // Tied embeddings: the head is embed_tokens^T, `[hidden, vocab]`.
            let (data, shape) = self.read("model.embed_tokens.weight")?;
            let [vocab, hidden] = shape[..] else {
                return Err(BuildError(format!(
                    "model.embed_tokens.weight must be 2-D, got {shape:?}"
                )));
            };
            let weight = Self::transpose_2d(&data, vocab, hidden);
            let weight = self.wrap(UnifiedTensor::new(&self.ctx, &weight, &[hidden, vocab]));
            self.maybe_add_lora(Linear::new(weight), "lm_head", 0)
        };

        Ok(Lfm2Model::new(embed_tokens, None, layers, norm_f, lm_head))
    }
}

/// Load the model in `model_path` (`config.json` plus safetensors).
pub fn load_lfm2_model(
    model_path: &str,
    ctx: Arc<WgpuContext>,
    engine: Arc<ComputeEngine>,
    lora_config: Option<LoRaConfig>,
) -> std::result::Result<Lfm2Model, String> {
    use std::path::Path;

    let model_dir = Path::new(model_path);
    let config = ferrotensor_loader::ModelLoader::load_config(model_dir.join("config.json"))
        .map_err(|e| format!("Failed to load config: {}", e))?;

    let mut weights = WeightsContext::new();
    weights
        .load_model_dir(model_dir)
        .map_err(|e| format!("Failed to load safetensors: {}", e))?;

    let mut builder = Lfm2ModelBuilder::new(ctx, engine, &weights, config);
    if let Some(lora) = lora_config {
        builder = builder.with_lora(lora);
    }

    builder.build().map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::Lfm2ModelBuilder;

    #[test]
    fn transpose_2d_swaps_rows_and_columns() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let transposed = Lfm2ModelBuilder::transpose_2d(&data, 2, 3);
        assert_eq!(transposed, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }
}
