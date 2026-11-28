//! Model builder: constructs an [`Lfm2Model`] from a [`WeightsContext`].
//!
//! Loads Hugging Face safetensors checkpoints of two families into the same
//! decoder structure, with optional LoRA adapters:
//!
//! * **Llama-style** (any `model_type` but `lfm2`, e.g. SmolLM2, TinyLlama,
//!   Qwen2.5, Qwen3): every block is RoPE grouped-query attention plus a
//!   SwiGLU MLP; weights named `self_attn.{q,k,v,o}_proj`,
//!   `mlp.{gate,up,down}_proj`, `input_layernorm`, `post_attention_layernorm`
//!   and `model.norm`. A projection with a `<name>.bias` tensor (Qwen2) gets
//!   that bias. When a checkpoint has `self_attn.q_norm` / `self_attn.k_norm`
//!   (Qwen3), they are loaded as per-head query/key norms
//!   ([`Attention::q_norm`]). Config features the forward pass lacks are
//!   refused ([`family_spec`]).
//! * **LFM2** (see [`family_spec`]): blocks are short convolutions or
//!   attention, named `operator_norm`, `ffn_norm`, `feed_forward.w{1,2,3}`,
//!   with the final norm stored as `model.embedding_norm`.
//!
//! Base weights are frozen ([`Tensor::frozen`]) unless
//! [`Lfm2ModelBuilder::with_trainable_base`] is set, so LoRA training skips
//! their gradients. Frozen linear-layer weights can be stored as f16
//! ([`Lfm2ModelBuilder::with_f16_weights`]), and those of the decoder blocks,
//! or any [`Q4Selection`] of linear weights, at 4 bits
//! ([`Lfm2ModelBuilder::with_q4_weights`], which changes the model's
//! outputs). Pre-quantized checkpoints (`<name>.scales` tensors) are
//! rejected.

use crate::family::{family_spec, FamilySpec, LayerKind};
use crate::lfm::{Lfm2Block, Lfm2Conv, Lfm2LayerOp, Lfm2Model};
use crate::{Attention, FusedSlot, Linear, LoRAAdapter, RMSNorm, MLP};
use ferrotensor_autograd::Tensor;
use ferrotensor_backend::q4::Q4Group;
use ferrotensor_backend::{ComputeEngine, DType, RopeFreqs, UnifiedTensor, WgpuContext};
use ferrotensor_loader::{Lfm2Config, WeightsContext};
use std::ops::RangeInclusive;
use std::sync::Arc;

pub use crate::family::Architecture;

/// Whether the parameter `name` (a Hugging Face name such as
/// `model.layers.3.mlp.up_proj.weight`) belongs to the module group `key`.
///
/// `key` is case-insensitive: `all` is every weight but the token
/// embedding, `full` every weight, `embed` (or `embedding`) the token
/// embedding, `attn` (or `attention`) the attention or LFM2 attention
/// blocks, `mlp` the MLPs (`mlp` in Llama blocks, `feed_forward` in LFM2),
/// `head` the `lm_head`, `layers.N` the blocks of layer `N` and
/// `layers.A-B` those of layers `A` to `B` inclusive. Any other key matches
/// the names that contain it, such as `q_proj` or `norm`.
pub fn module_matches(key: &str, name: &str) -> bool {
    let key = key.to_ascii_lowercase();
    if let Some(layers) = layer_range(&key) {
        return layer_index(name).is_some_and(|i| layers.contains(&i));
    }
    let parts: &[&str] = match key.as_str() {
        "all" => return !name.contains("embed_tokens"),
        "full" => return true,
        "embed" | "embedding" => &["embed_tokens"],
        "attn" | "attention" => &["self_attn"],
        "mlp" => &[".mlp.", ".feed_forward."],
        "head" => &["lm_head"],
        other => return name.contains(other),
    };
    parts.iter().any(|p| name.contains(p))
}

/// The layers `layers.N` or `layers.A-B` names, or `None` for other keys.
fn layer_range(key: &str) -> Option<RangeInclusive<usize>> {
    let spec = key.strip_prefix("layers.")?;
    let (first, last) = spec.split_once('-').unwrap_or((spec, spec));
    Some(first.parse().ok()?..=last.parse().ok()?)
}

/// The block index in a `...layers.N....` parameter name.
fn layer_index(name: &str) -> Option<usize> {
    let mut parts = name.split('.');
    parts.find(|p| *p == "layers")?;
    parts.next()?.parse().ok()
}

/// Which frozen linear weights [`Lfm2ModelBuilder::with_q4_selection`]
/// stores at 4 bits, and how many weights share a scale.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Q4Selection {
    /// [`module_matches`] keys; a weight matching any of them is quantised.
    pub modules: Vec<String>,
    /// Keys that keep a weight out even when `modules` matches it.
    pub skip: Vec<String>,
    /// Weights per scale and minimum.
    pub group: Q4Group,
}

impl Default for Q4Selection {
    /// Every decoder-block projection in groups of 64: what
    /// [`Lfm2ModelBuilder::with_q4_weights`] stores at 4 bits.
    fn default() -> Self {
        Self {
            modules: vec!["layers".to_string()],
            skip: Vec::new(),
            group: Q4Group::DEFAULT,
        }
    }
}

impl Q4Selection {
    /// Whether the linear weight `name` (e.g. `lm_head.weight`) is selected.
    pub fn selects(&self, name: &str) -> bool {
        self.modules.iter().any(|k| module_matches(k, name))
            && !self.skip.iter().any(|k| module_matches(k, name))
    }
}

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

/// Builds an [`Lfm2Model`] from loaded weights.
pub struct Lfm2ModelBuilder<'a> {
    ctx: Arc<WgpuContext>,
    // Kept for API compatibility; unused since pre-quantized loading was removed.
    #[allow(dead_code)]
    engine: Arc<ComputeEngine>,
    weights: &'a WeightsContext,
    config: Lfm2Config,
    lora_config: Option<LoRaConfig>,
    trainable_base: bool,
    f16_weights: bool,
    q4: Option<Q4Selection>,
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
            f16_weights: false,
            q4: None,
        }
    }

    /// Attach LoRA adapters to the projections named in `lora_config`.
    pub fn with_lora(mut self, lora_config: LoRaConfig) -> Self {
        self.lora_config = Some(lora_config);
        self
    }

    /// Store frozen linear-layer weights (attention, MLP and `lm_head`) as
    /// f16, half the memory; activations, products and LoRA adapters stay
    /// f32. Ignored when the base weights are trainable. The frozen embedding
    /// table is stored as f16 too, since it is looked up from a host copy;
    /// norms stay f32.
    pub fn with_f16_weights(mut self, f16: bool) -> Self {
        self.f16_weights = f16;
        self
    }

    /// Store the frozen weights of every decoder-block projection
    /// (`q/k/v/o_proj`, `gate/up/down_proj`, LFM2 `in_proj`/`out_proj` and
    /// `w1`/`w2`/`w3`) at 4 bits ([`UnifiedTensor::from_f32_as_q4`]: affine
    /// codes in groups of 64 along the input dimension, with an f16 scale
    /// and minimum per group, about 4.5 bits per weight). The matmul decodes
    /// them and computes in f32.
    ///
    /// Unlike f16 storage, this changes the model: it computes with the
    /// decoded weights, which differ from the checkpoint's. `lm_head` is not
    /// quantised (it follows [`with_f16_weights`](Self::with_f16_weights), as
    /// does the frozen embedding table); norms and LoRA adapters stay f32.
    /// Quantisation runs on the host while the model is built.
    ///
    /// [`build`](Self::build) fails when this is combined with
    /// [`with_trainable_base`](Self::with_trainable_base) (4-bit weights
    /// cannot train) and when a weight cannot be quantised (NaN, infinite or
    /// beyond the f16 range). Off by default; the same as
    /// [`with_q4_selection`](Self::with_q4_selection) with
    /// [`Q4Selection::default`].
    pub fn with_q4_weights(mut self, q4: bool) -> Self {
        self.q4 = q4.then(Q4Selection::default);
        self
    }

    /// Store the frozen linear weights `selection` picks at 4 bits, in its
    /// group size, including `lm_head` (tied or not) when selected. As
    /// [`with_q4_weights`](Self::with_q4_weights) otherwise; the token
    /// embedding is a host lookup table and never quantised.
    pub fn with_q4_selection(mut self, selection: Q4Selection) -> Self {
        self.q4 = Some(selection);
        self
    }

    /// The 4-bit settings for the linear weight `name` of `shape`, if
    /// [`with_q4_selection`](Self::with_q4_selection) selects it.
    fn q4_for(&self, name: &str, shape: &[usize]) -> Option<&Q4Selection> {
        self.q4
            .as_ref()
            .filter(|q4| shape.len() == 2 && q4.selects(name))
    }

    /// Uploads the linear-layer weight `name`: f32 when the base trains,
    /// 4-bit when the 4-bit selection picks it, else f16 when
    /// [`with_f16_weights`](Self::with_f16_weights) applies, else f32.
    fn upload_linear(&self, name: &str, data: &[f32], shape: &[usize]) -> Result<UnifiedTensor> {
        Ok(if self.trainable_base {
            UnifiedTensor::new(&self.ctx, data, shape)
        } else if let Some(q4) = self.q4_for(name, shape) {
            UnifiedTensor::from_f32_as_q4(&self.ctx, data, shape, q4.group)
                .map_err(|e| BuildError(format!("{name}: {e}")))?
        } else if self.f16_weights {
            UnifiedTensor::from_f32_as_f16(&self.ctx, data, shape)
        } else {
            UnifiedTensor::new(&self.ctx, data, shape)
        })
    }

    /// The element type [`upload_linear`](Self::upload_linear) stores for a
    /// weight the 4-bit selection leaves out.
    fn linear_dtype(&self) -> DType {
        if self.f16_weights && !self.trainable_base {
            DType::FP16
        } else {
            DType::FP32
        }
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

    fn has(&self, name: &str) -> Result<bool> {
        self.weights
            .get_tensor_info(name)
            .map(|meta| meta.is_some())
            .map_err(|e| BuildError(format!("cannot read {name} from the checkpoint: {e}")))
    }

    fn read(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        self.weights
            .read_f32_tensor(name)
            .map_err(|e| BuildError(format!("failed to load weight {name}: {e}")))
    }

    /// Fails if `name` is stored pre-quantized (a `<name>.scales` tensor).
    fn reject_pre_quantized(&self, name: &str) -> Result<()> {
        if self.has(&format!("{name}.scales"))? {
            return Err(BuildError(format!(
                "{name} is pre-quantized ({name}.scales found); pre-quantized checkpoints are not supported. Load the f16/f32 checkpoint and use 4-bit base weights (with_q4_weights / --q4-weights) instead"
            )));
        }
        Ok(())
    }

    /// Load `<name>.weight` unchanged.
    fn load_tensor(&self, name: &str) -> Result<Tensor> {
        self.reject_pre_quantized(name)?;
        self.load_raw_tensor(&format!("{name}.weight"))
    }

    /// The token embeddings, and, unless they are stored quantized, the f32
    /// values and shape read for them: the model keeps those as its host
    /// lookup table, and a tied `lm_head` is made from them, so the
    /// checkpoint's table is read once and never read back from the GPU.
    ///
    /// A frozen table is looked up from that host copy only, so its GPU copy
    /// is stored at the linear weight type ([`linear_dtype`](Self::linear_dtype)):
    /// half the memory with f16 weights. A table that trains stays f32.
    #[allow(clippy::type_complexity)]
    fn load_embeddings(&self) -> Result<(Tensor, Option<(Vec<f32>, Vec<usize>)>)> {
        if self.has("model.embed_tokens.scales")? {
            return Ok((self.load_tensor("model.embed_tokens")?, None));
        }
        let (data, shape) = self.read("model.embed_tokens.weight")?;
        let table = match self.linear_dtype() {
            DType::FP16 => UnifiedTensor::from_f32_as_f16(&self.ctx, &data, &shape),
            _ => UnifiedTensor::new(&self.ctx, &data, &shape),
        };
        Ok((self.wrap(table), Some((data, shape))))
    }

    fn load_raw_tensor(&self, name: &str) -> Result<Tensor> {
        let (data, shape) = self.read(name)?;
        Ok(self.wrap(UnifiedTensor::new(&self.ctx, &data, &shape)))
    }

    /// Load a PyTorch `[out, in]` weight transposed to `[in, out]`.
    fn load_linear_weight_tensor(&self, prefix: &str) -> Result<Tensor> {
        let name = format!("{prefix}.weight");
        let (data, shape) = self.read(&name)?;
        let (data, shape) = if let [rows, cols] = shape[..] {
            (Self::transpose_2d(&data, rows, cols), vec![cols, rows])
        } else {
            (data, shape)
        };
        Ok(self.wrap(self.upload_linear(&name, &data, &shape)?))
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
                    lora_seed(seed, module_name),
                );
                return Linear::with_lora(linear.weight, lora);
            }
        }
        linear
    }

    /// `module_name` is matched against the LoRA target modules; `seed` is
    /// the layer's seed. The adapter's own seed also depends on
    /// `module_name`, so adapters of equal shape within a layer differ too.
    fn build_linear(&self, prefix: &str, module_name: &str, seed: u64) -> Result<Linear> {
        self.reject_pre_quantized(prefix)?;
        let weight = self.load_linear_weight_tensor(prefix)?;
        let bias_name = format!("{prefix}.bias");
        let bias = if self.has(&bias_name)? {
            let bias = self.load_raw_tensor(&bias_name)?;
            let out_dim = weight.data.shape[1];
            if bias.data.shape != [out_dim] {
                return Err(BuildError(format!(
                    "{bias_name} has shape {:?}, expected [{out_dim}]",
                    bias.data.shape
                )));
            }
            Some(bias)
        } else {
            None
        };
        let linear = self.maybe_add_lora(Linear::new(weight), module_name, seed);
        Ok(Linear { bias, ..linear })
    }

    fn build_rms_norm(&self, prefix: &str) -> Result<RMSNorm> {
        let weight = self.load_raw_tensor(&format!("{prefix}.weight"))?;
        Ok(RMSNorm::new(weight, self.config.norm_eps))
    }

    /// The per-head norm `{prefix}.weight` (`[head_dim]`) if the checkpoint
    /// has it.
    fn build_head_norm(&self, prefix: &str, head_dim: usize) -> Result<Option<RMSNorm>> {
        if !self.has(&format!("{prefix}.weight"))? {
            return Ok(None);
        }
        let norm = self.build_rms_norm(prefix)?;
        if norm.weight.data.shape != [head_dim] {
            return Err(BuildError(format!(
                "{prefix}.weight has shape {:?}, expected [{head_dim}] (one weight per head dimension)",
                norm.weight.data.shape
            )));
        }
        Ok(Some(norm))
    }

    fn build_mlp(&self, spec: &FamilySpec, prefix: &str, seed: u64) -> Result<MLP> {
        let [gate, up, down] = spec.names.mlp_proj;
        Ok(MLP {
            gate_proj: self.build_linear(&format!("{prefix}.{gate}"), gate, seed)?,
            up_proj: self.build_linear(&format!("{prefix}.{up}"), up, seed)?,
            down_proj: self.build_linear(&format!("{prefix}.{down}"), down, seed)?,
            fused: FusedSlot::default(),
        })
    }

    fn build_attention(&self, spec: &FamilySpec, prefix: &str, seed: u64) -> Result<Attention> {
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
        if head_dim > 128 || !head_dim.is_multiple_of(2) {
            return Err(BuildError(format!(
                "head_dim {head_dim} is not supported (RoPE needs an even size, and the attention kernel handles at most 128)"
            )));
        }
        if self.has(&format!("{prefix}.q_layernorm.weight"))? {
            return Err(BuildError(
                "LFM2 per-head q/k layernorm is not supported yet".into(),
            ));
        }
        let [q_name, k_name, v_name, o_name] = spec.names.attn_proj;
        let [q_norm, k_norm] = spec.names.head_norms;

        let rope_theta = spec.rope_theta;
        // The table is built once per layer; it is `head_dim / 2` floats.
        let rope_inv_freq = spec
            .rope
            .map(|scaling| {
                RopeFreqs::new(&self.ctx, &scaling.inv_freq(head_dim, rope_theta), head_dim)
            })
            .transpose()
            .map_err(|e| BuildError(e.to_string()))?;

        Ok(Attention {
            q_proj: self.build_linear(&format!("{prefix}.{q_name}"), q_name, seed)?,
            k_proj: self.build_linear(&format!("{prefix}.{k_name}"), k_name, seed)?,
            v_proj: self.build_linear(&format!("{prefix}.{v_name}"), v_name, seed)?,
            o_proj: self.build_linear(&format!("{prefix}.{o_name}"), o_name, seed)?,
            n_heads,
            head_dim,
            num_kv_heads,
            dropout_prob: self.config.attention_dropout.unwrap_or(0.0),
            rope_theta: rope_theta as f32,
            rope_inv_freq,
            q_norm: self.build_head_norm(&format!("{prefix}.{q_norm}"), head_dim)?,
            k_norm: self.build_head_norm(&format!("{prefix}.{k_norm}"), head_dim)?,
            fused: FusedSlot::default(),
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

    fn build_layer(&self, spec: &FamilySpec, layer_idx: usize) -> Result<Lfm2Block> {
        let names = &spec.names;
        let prefix = format!("{}.{layer_idx}", names.layers);
        let seed = layer_idx as u64 + 1;
        let op = match spec.layer(layer_idx)? {
            LayerKind::Conv => Lfm2LayerOp::Conv(self.build_conv(&format!("{prefix}.conv"), seed)?),
            LayerKind::Attention => Lfm2LayerOp::Attention(self.build_attention(
                spec,
                &format!("{prefix}.{}", names.attn),
                seed,
            )?),
        };
        Ok(Lfm2Block {
            operator_norm: self.build_rms_norm(&format!("{prefix}.{}", names.operator_norm))?,
            op,
            ffn_norm: self.build_rms_norm(&format!("{prefix}.{}", names.ffn_norm))?,
            mlp: self.build_mlp(spec, &format!("{prefix}.{}", names.mlp), seed)?,
        })
    }

    /// Build the complete model, or explain which weight or setting is
    /// missing or unsupported.
    pub fn build(self) -> Result<Lfm2Model> {
        if self.q4.is_some() && self.trainable_base {
            return Err(BuildError(
                "4-bit base weights (with_q4_weights / --q4-weights) cannot train: \
                 use them with LoRA adapters only, or load the base weights without 4-bit storage"
                    .into(),
            ));
        }
        let spec = family_spec(&self.config)?;
        let arch = spec.architecture;
        let n_layers = self.config.num_hidden_layers;
        log::info!("building {arch:?} model with {n_layers} layers");

        let (embed_tokens, embed_table) = self.load_embeddings()?;
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            layers.push(self.build_layer(&spec, i)?);
        }

        let mut final_norm = None;
        for name in spec.names.final_norms {
            if self.has(&format!("{name}.weight"))? {
                final_norm = Some(self.build_rms_norm(name)?);
                break;
            }
        }
        let norm_f = final_norm.ok_or_else(|| {
            let tried: Vec<String> = spec
                .names
                .final_norms
                .iter()
                .map(|name| format!("{name}.weight"))
                .collect();
            BuildError(format!("no final norm ({})", tried.join(" or ")))
        })?;

        let lm_head = if self.has("lm_head.weight")? {
            self.build_linear("lm_head", "lm_head", 0)?
        } else {
            // Tied embeddings: the head is embed_tokens^T, `[hidden, vocab]`,
            // made from the table already in memory.
            let read_again;
            let (data, shape) = match &embed_table {
                Some((data, shape)) => (data.as_slice(), shape.as_slice()),
                None => {
                    read_again = self.read("model.embed_tokens.weight")?;
                    (read_again.0.as_slice(), read_again.1.as_slice())
                }
            };
            let [vocab, hidden] = shape[..] else {
                return Err(BuildError(format!(
                    "model.embed_tokens.weight must be 2-D, got {shape:?}"
                )));
            };
            let head = "lm_head.weight";
            let weight = if self.q4_for(head, &[hidden, vocab]).is_some() {
                // Quantising needs the logical [hidden, vocab] order, so this
                // path holds one transposed copy of the table on the host.
                let transposed = Self::transpose_2d(data, vocab, hidden);
                self.upload_linear(head, &transposed, &[hidden, vocab])?
            } else {
                // Transposed and converted while it is written to the GPU
                // buffer: no transposed copy of the table on the host.
                UnifiedTensor::from_f32_transposed(
                    &self.ctx,
                    data,
                    vocab,
                    hidden,
                    self.linear_dtype(),
                )
            };
            self.maybe_add_lora(Linear::new(self.wrap(weight)), "lm_head", 0)
        };

        let model =
            Lfm2Model::new(embed_tokens, None, layers, norm_f, lm_head).with_architecture(arch);
        if let Some(q4) = &self.q4 {
            if !model
                .named_params()
                .values()
                .any(|t| t.data.dtype == DType::Q4)
            {
                return Err(BuildError(format!(
                    "the 4-bit selection (modules {:?}, skip {:?}) matches no linear weight",
                    q4.modules, q4.skip
                )));
            }
        }
        Ok(match embed_table {
            Some((table, _)) => model.with_host_embeddings(table),
            None => model,
        })
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

/// The generator seed of the adapter on `module_name` in the layer with seed
/// `layer_seed` (`layer_idx + 1`; `0` for `lm_head`).
///
/// FNV-1a over the name bytes and then the layer seed's little-endian bytes,
/// followed by the splitmix64 finalizer so that nearby inputs give unrelated
/// seeds. It is written out by hand because `DefaultHasher` is not stable
/// across Rust versions, and adapters must initialise identically on every
/// build.
fn lora_seed(layer_seed: u64, module_name: &str) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut h = FNV_OFFSET;
    for byte in module_name.bytes().chain(layer_seed.to_le_bytes()) {
        h ^= u64::from(byte);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94d0_49bb_1331_11eb);
    h ^ (h >> 31)
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::{lora_seed, module_matches, Lfm2ModelBuilder, Q4Group, Q4Selection};

    #[test]
    fn lora_seed_is_pinned() {
        // Fixed values: a change here changes every adapter's initial A.
        assert_eq!(lora_seed(1, "q_proj"), 0x5f28_d619_21d3_ddb3);
        assert_eq!(lora_seed(1, "o_proj"), 0xfabf_d797_7f14_3d7c);
        assert_eq!(lora_seed(2, "q_proj"), 0x4bae_9242_648e_2fc1);
        assert_eq!(lora_seed(0, "lm_head"), 0x6d1a_1dff_acc3_0def);
    }

    #[test]
    fn lora_seed_differs_per_layer_and_projection() {
        let names = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "in_proj",
            "out_proj",
            "w1",
            "w2",
            "w3",
        ];
        let mut seeds = std::collections::HashSet::new();
        let mut count = 0;
        for layer in 1..=128u64 {
            for name in names {
                seeds.insert(lora_seed(layer, name));
                count += 1;
            }
        }
        seeds.insert(lora_seed(0, "lm_head"));
        count += 1;
        assert_eq!(seeds.len(), count, "two adapters would share a seed");
    }

    #[test]
    fn transpose_2d_swaps_rows_and_columns() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let transposed = Lfm2ModelBuilder::transpose_2d(&data, 2, 3);
        assert_eq!(transposed, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    /// The linear weights of a three-layer Llama model and an LFM2 block.
    const LINEARS: [&str; 10] = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.1.self_attn.k_proj.weight",
        "model.layers.1.mlp.up_proj.weight",
        "model.layers.2.self_attn.o_proj.weight",
        "model.layers.12.self_attn.q_proj.weight",
        "model.layers.12.mlp.gate_proj.weight",
        "model.layers.3.feed_forward.w1.weight",
        "model.layers.3.conv.in_proj.weight",
        "lm_head.weight",
    ];

    fn selected(keys: &[&str], skip: &[&str]) -> Vec<&'static str> {
        let selection = Q4Selection {
            modules: keys.iter().map(|k| k.to_string()).collect(),
            skip: skip.iter().map(|k| k.to_string()).collect(),
            group: Q4Group::DEFAULT,
        };
        LINEARS
            .into_iter()
            .filter(|n| selection.selects(n))
            .collect()
    }

    #[test]
    fn module_keys_match_groups_names_and_layer_ranges() {
        let blocks = &LINEARS[..9];
        assert_eq!(selected(&["layers"], &[]), blocks);
        assert_eq!(Q4Selection::default().modules, ["layers"]);
        assert_eq!(selected(&["all"], &[]), LINEARS);
        assert_eq!(selected(&["head"], &[]), ["lm_head.weight"]);
        assert_eq!(selected(&["lm_head"], &[]), ["lm_head.weight"]);
        assert_eq!(
            selected(&["q_proj"], &[]),
            [LINEARS[0], LINEARS[5]],
            "a name part"
        );
        assert_eq!(
            selected(&["mlp"], &[]),
            [LINEARS[1], LINEARS[3], LINEARS[6], LINEARS[7]],
            "Llama mlp and LFM2 feed_forward"
        );
        assert_eq!(
            selected(&["ATTN"], &[]),
            [LINEARS[0], LINEARS[2], LINEARS[4], LINEARS[5]]
        );
        // `layers.1` is layer 1 alone, not 1x; a range is inclusive.
        assert_eq!(selected(&["layers.1"], &[]), [LINEARS[2], LINEARS[3]]);
        assert_eq!(selected(&["layers.0-1"], &[]), &LINEARS[..4]);
        assert_eq!(selected(&["layers.2-12"], &[]), &LINEARS[4..9]);
        assert_eq!(
            selected(&["q_proj", "head"], &[]),
            [LINEARS[0], LINEARS[5], LINEARS[9]],
            "keys add up"
        );
        assert!(selected(&["v_proj"], &[]).is_empty());
        assert!(selected(&[], &[]).is_empty());
    }

    #[test]
    fn skip_keys_remove_weights_from_the_selection() {
        assert_eq!(
            selected(&["all"], &["mlp", "layers.12"]),
            [LINEARS[0], LINEARS[2], LINEARS[4], LINEARS[8], LINEARS[9]]
        );
        assert_eq!(selected(&["layers"], &["layers.0-11"]), &LINEARS[5..7]);
        assert!(selected(&["head"], &["lm_head"]).is_empty());
    }

    #[test]
    fn train_module_keys_cover_every_parameter_kind() {
        let embed = "model.embed_tokens.weight";
        let norm = "model.layers.0.input_layernorm.weight";
        assert!(module_matches("embed", embed) && module_matches("Embedding", embed));
        assert!(!module_matches("all", embed) && module_matches("full", embed));
        assert!(module_matches("all", norm) && module_matches("norm", norm));
        assert!(!module_matches("attn", norm));
        assert!(module_matches("layers.0", norm) && !module_matches("layers.1-9", norm));
        // Not a layer range: matched as a name part.
        assert!(module_matches("layers.", norm) && !module_matches("layers.x", norm));
    }
}
