//! Model builder: Constructs Lfm2Model from WeightsContext
//!
//! This module handles loading real weights from safetensors files
//! and constructing the full LFM2.5 architecture with optional LoRA adapters.

use crate::lfm::{Lfm2Block, Lfm2Conv, Lfm2LayerOp, Lfm2Model};
use crate::{Attention, Linear, LoRAAdapter, RMSNorm, MLP};
use rusty_autograd::Tensor;
use rusty_backend::{ComputeEngine, UnifiedTensor, WgpuContext};
use rusty_loader::{Lfm2Config, WeightsContext};
use std::sync::Arc;

/// Configuration for LoRA adapters
pub struct LoRaConfig {
    pub rank: usize,
    pub alpha: f32,
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

/// Builds Lfm2Model from weights
pub struct Lfm2ModelBuilder<'a> {
    ctx: Arc<WgpuContext>,
    engine: Arc<ComputeEngine>,
    weights: &'a WeightsContext,
    config: Lfm2Config,
    lora_config: Option<LoRaConfig>,
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
        }
    }

    pub fn with_lora(mut self, lora_config: LoRaConfig) -> Self {
        self.lora_config = Some(lora_config);
        self
    }

    /// Load a tensor from weights, dequantizing if needed
    fn load_tensor(&self, name: &str) -> Tensor {
        // Check if this weight has quantization (scales/biases)
        let scales_name = format!("{}.scales", name);
        if self.weights.get_tensor_info(&scales_name).is_some() {
            // Quantized weight - use dequantization
            let ut = self
                .weights
                .upload_quantized(name, &self.ctx, &self.engine)
                .expect(&format!("Failed to load quantized weight: {}", name));
            Tensor::new(self.ctx.clone(), ut)
        } else {
            // Regular FP32/FP16 weight - direct load
            self.load_raw_tensor(&format!("{}.weight", name))
        }
    }

    fn load_raw_tensor(&self, name: &str) -> Tensor {
        let (data, shape) = self
            .weights
            .read_f32_tensor(name)
            .expect(&format!("Failed to load weight: {}", name));

        let ut = UnifiedTensor::new(&self.ctx, &data, &shape);
        Tensor::new(self.ctx.clone(), ut)
    }

    fn load_linear_weight_tensor(&self, prefix: &str) -> Tensor {
        let weight_name = format!("{}.weight", prefix);
        let (data, shape) = self
            .weights
            .read_f32_tensor(&weight_name)
            .expect(&format!("Failed to load weight: {}", weight_name));

        let (data, shape) = if shape.len() == 2 {
            let rows = shape[0];
            let cols = shape[1];
            (Self::transpose_2d(&data, rows, cols), vec![cols, rows])
        } else {
            (data, shape)
        };

        Tensor::new(self.ctx.clone(), UnifiedTensor::new(&self.ctx, &data, &shape))
    }

    fn maybe_add_lora(&self, linear: Linear, module_name: &str) -> Linear {
        if let Some(lora_cfg) = &self.lora_config {
            if lora_cfg
                .target_modules
                .iter()
                .any(|m| module_name.contains(m))
            {
                let shape = &linear.weight.data.shape;
                let in_dim = shape[0];
                let out_dim = shape[1];
                let lora = LoRAAdapter::new(
                    self.ctx.clone(),
                    in_dim,
                    out_dim,
                    lora_cfg.rank,
                    lora_cfg.alpha,
                );
                return Linear::with_lora(linear.weight, lora);
            }
        }
        linear
    }

    fn build_linear(&self, prefix: &str, module_name: &str) -> Linear {
        let weight = if self
            .weights
            .get_tensor_info(&format!("{}.scales", prefix))
            .is_some()
        {
            self.load_tensor(prefix)
        } else {
            self.load_linear_weight_tensor(prefix)
        };
        let linear = Linear::new(weight);
        self.maybe_add_lora(linear, module_name)
    }

    fn build_rms_norm(&self, prefix: &str) -> RMSNorm {
        let weight = self.load_raw_tensor(&format!("{}.weight", prefix));
        RMSNorm::new(weight, self.config.norm_eps)
    }

    fn build_mlp(&self, prefix: &str) -> MLP {
        MLP {
            gate_proj: self.build_linear(&format!("{}.w1", prefix), "w1"),
            up_proj: self.build_linear(&format!("{}.w3", prefix), "w3"),
            down_proj: self.build_linear(&format!("{}.w2", prefix), "w2"),
        }
    }

    fn build_attention(&self, prefix: &str) -> Attention {
        let n_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads.unwrap_or(n_heads);
        let head_dim = self.config.hidden_size / n_heads;
        let dropout_prob = self.config.attention_dropout.unwrap_or(0.0);

        Attention {
            q_proj: self.build_linear(&format!("{}.q_proj", prefix), "q_proj"),
            k_proj: self.build_linear(&format!("{}.k_proj", prefix), "k_proj"),
            v_proj: self.build_linear(&format!("{}.v_proj", prefix), "v_proj"),
            o_proj: self.build_linear(&format!("{}.out_proj", prefix), "out_proj"),
            n_heads,
            head_dim,
            num_kv_heads,
            dropout_prob,
        }
    }

    fn build_conv(&self, prefix: &str) -> Lfm2Conv {
        let conv_weight = self.load_raw_tensor(&format!("{}.conv.weight", prefix));
        // LFM uses no conv bias typically, create zero tensor
        let conv_bias = Tensor::new(
            self.ctx.clone(),
            UnifiedTensor::new(
                &self.ctx,
                &vec![0.0; self.config.hidden_size],
                &[self.config.hidden_size],
            ),
        );

        Lfm2Conv {
            in_proj: self.build_linear(&format!("{}.in_proj", prefix), "in_proj"),
            conv_weight,
            conv_bias,
            out_proj: self.build_linear(&format!("{}.out_proj", prefix), "out_proj"),
        }
    }

    fn build_layer(&self, layer_idx: usize) -> Lfm2Block {
        let prefix = format!("model.layers.{}", layer_idx);
        let layer_type = &self.config.layer_types[layer_idx];

        let op = match layer_type.as_str() {
            "conv" => Lfm2LayerOp::Conv(self.build_conv(&format!("{}.conv", prefix))),
            "full_attention" | "attention" => {
                Lfm2LayerOp::Attention(self.build_attention(&format!("{}.self_attn", prefix)))
            }
            _ => panic!("Unknown layer type: {}", layer_type),
        };

        Lfm2Block {
            operator_norm: self.build_rms_norm(&format!("{}.operator_norm", prefix)),
            op,
            ffn_norm: self.build_rms_norm(&format!("{}.ffn_norm", prefix)),
            mlp: self.build_mlp(&format!("{}.feed_forward", prefix)),
        }
    }

    /// Build the complete Lfm2Model
    pub fn build(self) -> Lfm2Model {
        println!(
            "Building LFM2.5 model with {} layers...",
            self.config.num_hidden_layers
        );

        // Embedding
        let embed_tokens = self.load_tensor("model.embed_tokens");
        let embedding_norm = self.build_rms_norm("model.embedding_norm");

        // Layers
        let mut layers = Vec::with_capacity(self.config.num_hidden_layers);
        for i in 0..self.config.num_hidden_layers {
            print!(
                "  Loading layer {}/{}...\r",
                i + 1,
                self.config.num_hidden_layers
            );
            layers.push(self.build_layer(i));
        }
        println!("\n  All layers loaded.");

        // Final norm - LFM2.5 doesn't have a separate final norm, reuse embedding_norm
        let norm_f = if self.weights.get_tensor_info("model.norm.weight").is_some() {
            self.build_rms_norm("model.norm")
        } else {
            println!("  No separate final norm, reusing embedding_norm");
            RMSNorm::new(embedding_norm.weight.clone(), self.config.norm_eps)
        };

        // LM Head (tied with embeddings in LFM, but we load separately if exists)
        let lm_head = if self.weights.get_tensor_info("lm_head.weight").is_some() {
            self.build_linear("lm_head", "lm_head")
        } else {
            // Tied embeddings - reuse embed_tokens
            println!("  Using tied embeddings for lm_head");
            Linear::new(embed_tokens.clone())
        };

        Lfm2Model {
            embed_tokens,
            embedding_norm,
            layers,
            norm_f,
            lm_head,
        }
    }
}

/// Convenience function to load LFM2.5 model
pub fn load_lfm2_model(
    model_path: &str,
    ctx: Arc<WgpuContext>,
    engine: Arc<ComputeEngine>,
    lora_config: Option<LoRaConfig>,
) -> Result<Lfm2Model, String> {
    use std::path::Path;

    let model_dir = Path::new(model_path);

    // Load config
    let config = rusty_loader::ModelLoader::load_config(model_dir.join("config.json"))
        .map_err(|e| format!("Failed to load config: {}", e))?;

    println!(
        "Model config loaded: {} layers, hidden_size={}",
        config.num_hidden_layers, config.hidden_size
    );

    // Load weights
    let mut weights = WeightsContext::new();
    weights
        .load_file(model_dir.join("model.safetensors"))
        .map_err(|e| format!("Failed to load safetensors: {}", e))?;

    // Build model
    let mut builder = Lfm2ModelBuilder::new(ctx, engine, &weights, config);
    if let Some(lora) = lora_config {
        builder = builder.with_lora(lora);
    }

    Ok(builder.build())
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
