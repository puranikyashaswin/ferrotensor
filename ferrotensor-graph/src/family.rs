//! The family of a checkpoint as data: how its tensors are named, which
//! activation and RoPE table its blocks use, and what each layer is.
//!
//! [`family_spec`] reads a `config.json` once and either returns a
//! [`FamilySpec`] or refuses the config, so a model the forward pass cannot
//! compute fails to load instead of giving wrong logits. Every refusal sits
//! next to the field it guards. The model builder reads the layer, attention,
//! MLP and norm names and the settings from the spec, and
//! [`Architecture::names`] gives the same table to the code that names
//! checkpoint and PEFT tensors (`Lfm2Block::named_params_as`).
//!
//! Some names are the same for every family and stay in the builder: the
//! embedding table and head (`model.embed_tokens`, `lm_head`), the probe for
//! `q_layernorm` that refuses LFM2's per-head norms, and the conv names
//! (`conv`, `in_proj`, `out_proj`). The legacy Llama names in
//! `lfm::legacy_llama_name` are a second table, and a unit test there checks
//! that it agrees with `LLAMA_NAMES` and `LFM2_NAMES`.

use crate::model_builder::BuildError;
use crate::rope::{rope_scaling, Llama3Scaling};
use ferrotensor_loader::Lfm2Config;

/// Weight-naming family of a checkpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Architecture {
    /// Llama / Mistral / SmolLM2 / Qwen2 / Qwen3 naming; every block is attention.
    Llama,
    /// Liquid LFM2 naming; conv and attention blocks per `layer_types`.
    Lfm2,
}

impl Architecture {
    /// The tensor names of this family.
    #[must_use]
    pub fn names(self) -> &'static Names {
        match self {
            Self::Llama => &LLAMA_NAMES,
            Self::Lfm2 => &LFM2_NAMES,
        }
    }
}

/// The MLP activation of a family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Act {
    /// `x * sigmoid(x)`, the gate of a SwiGLU MLP.
    Silu,
}

/// What one decoder layer computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    /// Full attention with RoPE.
    Attention,
    /// An LFM2 short convolution.
    Conv,
}

/// The tensor names of a family, relative to the layer or the model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Names {
    /// Prefix of the decoder layers, followed by `.{index}`.
    pub layers: &'static str,
    /// Norm before the operator (attention or conv), inside a layer.
    pub operator_norm: &'static str,
    /// Norm before the MLP, inside a layer.
    pub ffn_norm: &'static str,
    /// The attention module, inside a layer.
    pub attn: &'static str,
    /// The query, key, value and output projections, inside the attention module.
    pub attn_proj: [&'static str; 4],
    /// The per-head query and key norms, inside the attention module; a
    /// checkpoint that lacks them has none.
    pub head_norms: [&'static str; 2],
    /// The MLP module, inside a layer.
    pub mlp: &'static str,
    /// The gate, up and down projections, inside the MLP module.
    pub mlp_proj: [&'static str; 3],
    /// Candidates for the final norm, tried in order.
    pub final_norms: &'static [&'static str],
}

/// Llama, Mistral, SmolLM2, Qwen2 and Qwen3 names.
pub(crate) const LLAMA_NAMES: Names = Names {
    layers: "model.layers",
    operator_norm: "input_layernorm",
    ffn_norm: "post_attention_layernorm",
    attn: "self_attn",
    attn_proj: ["q_proj", "k_proj", "v_proj", "o_proj"],
    head_norms: ["q_norm", "k_norm"],
    mlp: "mlp",
    mlp_proj: ["gate_proj", "up_proj", "down_proj"],
    // Hugging Face LFM2 applies `embedding_norm` after the last block, and
    // the builder tries both names for either family.
    final_norms: &["model.norm", "model.embedding_norm"],
};

/// LFM2 names: the SwiGLU projections are `w1` (gate), `w3` (up) and `w2`
/// (down).
pub(crate) const LFM2_NAMES: Names = Names {
    layers: "model.layers",
    operator_norm: "operator_norm",
    ffn_norm: "ffn_norm",
    attn: "self_attn",
    attn_proj: ["q_proj", "k_proj", "v_proj", "out_proj"],
    head_norms: ["q_norm", "k_norm"],
    mlp: "feed_forward",
    mlp_proj: ["w1", "w3", "w2"],
    final_norms: &["model.norm", "model.embedding_norm"],
};

/// Everything the model builder takes from a `config.json` besides sizes.
#[derive(Clone, Debug, PartialEq)]
pub struct FamilySpec {
    /// The naming family.
    pub architecture: Architecture,
    /// The tensor names.
    pub names: Names,
    /// The MLP activation.
    pub act: Act,
    /// RoPE scaling over the plain table, if the config asks for it.
    pub rope: Option<Llama3Scaling>,
    /// RoPE base frequency.
    pub rope_theta: f64,
    /// The `layer_types` of the config, one per layer, or empty.
    layer_types: Vec<String>,
}

impl FamilySpec {
    /// What layer `index` computes.
    ///
    /// # Errors
    /// A [`BuildError`] naming the layer when its type is not one this
    /// family can run: a Llama-style layer that is not full attention (a
    /// sliding-window layer would need a windowed mask the attention path
    /// lacks), or an LFM2 layer of an unknown type.
    pub fn layer(&self, index: usize) -> Result<LayerKind, BuildError> {
        let layer_type = self.layer_types.get(index).map(String::as_str);
        match self.architecture {
            Architecture::Llama => match layer_type.filter(|t| *t != "full_attention") {
                None => Ok(LayerKind::Attention),
                Some(t) => Err(BuildError(format!(
                    "layer {index} is {t:?}; only full attention is supported"
                ))),
            },
            Architecture::Lfm2 => match layer_type.unwrap_or("full_attention") {
                "conv" => Ok(LayerKind::Conv),
                "full_attention" | "attention" => Ok(LayerKind::Attention),
                other => Err(BuildError(format!(
                    "unknown layer type {other:?} at layer {index}"
                ))),
            },
        }
    }
}

/// Reads the family of `config` and refuses what the forward pass does not
/// compute: a non-SiLU activation, a RoPE scheme other than `default` and
/// `llama3`, sliding-window attention.
///
/// `model_type: lfm2`, a `conv` entry in `layer_types`, or `layer_types`
/// without a `model_type` mean LFM2; everything else is Llama-style.
/// `layer_types` alone does not mean LFM2: recent `transformers` versions
/// also write it (`full_attention` / `sliding_attention`) into Qwen3 and
/// other configs.
///
/// # Errors
/// A [`BuildError`] naming the config field and its value.
pub fn family_spec(config: &Lfm2Config) -> Result<FamilySpec, BuildError> {
    let model_type = config.model_type.as_deref();
    let lfm2 = model_type.is_some_and(|t| t.eq_ignore_ascii_case("lfm2"))
        || config.layer_types.iter().any(|t| t == "conv")
        || (model_type.is_none() && !config.layer_types.is_empty());
    let (architecture, names) = if lfm2 {
        (Architecture::Lfm2, LFM2_NAMES)
    } else {
        (Architecture::Llama, LLAMA_NAMES)
    };

    let act = match config.hidden_act.as_deref() {
        None | Some("silu" | "swish") => Act::Silu,
        Some(act) => {
            return Err(BuildError(format!(
                "hidden_act {act:?} is not supported: the MLP computes SiLU (silu) only"
            )))
        }
    };

    let rope = rope_scaling(config).map_err(BuildError)?;
    let rope_theta = config
        .rope_theta
        .or_else(|| config.rope_parameters.as_ref()?.rope_theta)
        .unwrap_or(10000.0);

    // Only `use_sliding_window: true` turns the window on: Hugging Face's
    // Mistral sets `sliding_window` alone and still attends over everything.
    if config.use_sliding_window == Some(true) {
        let window = config
            .sliding_window
            .map_or_else(|| "unset".to_owned(), |w| w.to_string());
        return Err(BuildError(format!(
            "sliding_window {window} is not supported: attention always covers the whole sequence"
        )));
    }

    Ok(FamilySpec {
        architecture,
        names,
        act,
        rope,
        rope_theta,
        layer_types: config.layer_types.clone(),
    })
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    fn config(json: serde_json::Value) -> Lfm2Config {
        serde_json::from_value(json).expect("config")
    }

    fn spec_of(json: serde_json::Value) -> FamilySpec {
        family_spec(&config(json)).expect("spec")
    }

    /// `config.json` of SmolLM2-135M on the Hugging Face hub.
    fn llama_config() -> serde_json::Value {
        serde_json::json!({
            "architectures": ["LlamaForCausalLM"],
            "attention_bias": false,
            "attention_dropout": 0.0,
            "bos_token_id": 0,
            "eos_token_id": 0,
            "hidden_act": "silu",
            "hidden_size": 576,
            "initializer_range": 0.041666666666666664,
            "intermediate_size": 1536,
            "is_llama_config": true,
            "max_position_embeddings": 8192,
            "model_type": "llama",
            "num_attention_heads": 9,
            "num_hidden_layers": 30,
            "num_key_value_heads": 3,
            "pretraining_tp": 1,
            "rms_norm_eps": 1e-05,
            "rope_interleaved": false,
            "rope_scaling": null,
            "rope_theta": 100000,
            "tie_word_embeddings": true,
            "torch_dtype": "bfloat16",
            "use_cache": true,
            "vocab_size": 49152
        })
    }

    /// The `config.json` of Qwen2.5-0.5B on the Hugging Face hub.
    fn qwen2_5_config() -> serde_json::Value {
        serde_json::json!({
            "architectures": ["Qwen2ForCausalLM"],
            "attention_dropout": 0.0,
            "bos_token_id": 151643,
            "eos_token_id": 151643,
            "hidden_act": "silu",
            "hidden_size": 896,
            "initializer_range": 0.02,
            "intermediate_size": 4864,
            "max_position_embeddings": 32768,
            "max_window_layers": 24,
            "model_type": "qwen2",
            "num_attention_heads": 14,
            "num_hidden_layers": 24,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-06,
            "rope_theta": 1000000.0,
            "sliding_window": 32768,
            "tie_word_embeddings": true,
            "torch_dtype": "bfloat16",
            "use_cache": true,
            "use_sliding_window": false,
            "vocab_size": 151936
        })
    }

    /// The `config.json` of Qwen3-0.6B (`transformers` 5 layout, with
    /// `layer_types` and `rope_parameters`), cut to 3 layers.
    fn qwen3_config() -> serde_json::Value {
        serde_json::json!({
            "architectures": ["Qwen3ForCausalLM"],
            "attention_bias": false,
            "attention_dropout": 0.0,
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 1024,
            "intermediate_size": 3072,
            "layer_types": ["full_attention", "full_attention", "full_attention"],
            "max_position_embeddings": 40960,
            "max_window_layers": 28,
            "model_type": "qwen3",
            "num_attention_heads": 16,
            "num_hidden_layers": 3,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-06,
            "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
            "sliding_window": null,
            "tie_word_embeddings": true,
            "use_cache": true,
            "use_sliding_window": false,
            "vocab_size": 151936
        })
    }

    /// The `config.json` of Mistral-7B-v0.1: a window that is set and not used.
    fn mistral_config() -> serde_json::Value {
        serde_json::json!({
            "architectures": ["MistralForCausalLM"],
            "bos_token_id": 1,
            "eos_token_id": 2,
            "hidden_act": "silu",
            "hidden_size": 4096,
            "initializer_range": 0.02,
            "intermediate_size": 14336,
            "max_position_embeddings": 32768,
            "model_type": "mistral",
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-05,
            "rope_theta": 10000.0,
            "sliding_window": 4096,
            "tie_word_embeddings": false,
            "torch_dtype": "bfloat16",
            "use_cache": true,
            "vocab_size": 32000
        })
    }

    /// The `config.json` of Llama-3.2-1B.
    fn llama3_config() -> serde_json::Value {
        serde_json::json!({
            "architectures": ["LlamaForCausalLM"],
            "attention_bias": false,
            "attention_dropout": 0.0,
            "bos_token_id": 128000,
            "eos_token_id": 128001,
            "head_dim": 64,
            "hidden_act": "silu",
            "hidden_size": 2048,
            "initializer_range": 0.02,
            "intermediate_size": 8192,
            "max_position_embeddings": 131072,
            "mlp_bias": false,
            "model_type": "llama",
            "num_attention_heads": 32,
            "num_hidden_layers": 16,
            "num_key_value_heads": 8,
            "pretraining_tp": 1,
            "rms_norm_eps": 1e-05,
            "rope_scaling": {
                "factor": 32.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3"
            },
            "rope_theta": 500000.0,
            "tie_word_embeddings": true,
            "torch_dtype": "bfloat16",
            "use_cache": true,
            "vocab_size": 128256
        })
    }

    /// The `config.json` of LFM2-350M, cut to 4 layers and the keys the
    /// loader reads.
    fn lfm2_config() -> serde_json::Value {
        serde_json::json!({
            "architectures": ["Lfm2ForCausalLM"],
            "block_dim": 1024,
            "block_ff_dim": 6656,
            "bos_token_id": 1,
            "eos_token_id": 7,
            "hidden_size": 1024,
            "intermediate_size": 6656,
            "layer_types": ["conv", "conv", "full_attention", "conv"],
            "max_position_embeddings": 128000,
            "model_type": "lfm2",
            "norm_eps": 1e-05,
            "num_attention_heads": 16,
            "num_hidden_layers": 4,
            "num_key_value_heads": 8,
            "rope_theta": 1000000.0,
            "vocab_size": 65536
        })
    }

    fn with(base: serde_json::Value, changes: serde_json::Value) -> serde_json::Value {
        let mut json = base;
        let fields = json.as_object_mut().expect("object");
        for (key, value) in changes.as_object().expect("object") {
            fields.insert(key.clone(), value.clone());
        }
        json
    }

    fn refused(changes: serde_json::Value) -> Result<FamilySpec, BuildError> {
        family_spec(&config(with(qwen2_5_config(), changes)))
    }

    #[test]
    fn llama_spec() {
        let spec = spec_of(llama_config());
        assert_eq!(spec.architecture, Architecture::Llama);
        assert_eq!(spec.names, LLAMA_NAMES);
        assert_eq!(spec.names.layers, "model.layers");
        assert_eq!(spec.names.operator_norm, "input_layernorm");
        assert_eq!(spec.names.ffn_norm, "post_attention_layernorm");
        assert_eq!(
            spec.names.attn_proj,
            ["q_proj", "k_proj", "v_proj", "o_proj"]
        );
        assert_eq!(spec.names.mlp, "mlp");
        assert_eq!(spec.names.mlp_proj, ["gate_proj", "up_proj", "down_proj"]);
        assert_eq!(spec.act, Act::Silu);
        assert_eq!(spec.rope, None);
        assert_eq!(spec.rope_theta, 100000.0);
        assert_eq!(spec.layer(0), Ok(LayerKind::Attention));
        assert_eq!(spec.layer(29), Ok(LayerKind::Attention));
    }

    #[test]
    fn qwen2_spec_has_the_llama_names_and_reads_the_base_frequency() {
        // Qwen2's q, k and v biases and Qwen3's head norms are tensors the
        // builder probes by these names; the spec has no flag for them.
        let spec = spec_of(qwen2_5_config());
        assert_eq!(spec.architecture, Architecture::Llama);
        assert_eq!(spec.names, LLAMA_NAMES);
        assert_eq!(spec.rope_theta, 1_000_000.0);
        assert_eq!(spec.layer(23), Ok(LayerKind::Attention));
        // transformers 5 writes the base inside rope_parameters.
        let spec = spec_of(with(
            qwen2_5_config(),
            serde_json::json!({
                "rope_theta": null,
                "rope_parameters": {"rope_type": "default", "rope_theta": 1000000.0}
            }),
        ));
        assert_eq!(spec.rope_theta, 1_000_000.0);
        assert_eq!(spec.rope, None);
    }

    #[test]
    fn qwen3_spec_is_llama_even_with_layer_types() {
        let spec = spec_of(qwen3_config());
        assert_eq!(spec.architecture, Architecture::Llama);
        assert_eq!(spec.names.head_norms, ["q_norm", "k_norm"]);
        assert_eq!(spec.rope_theta, 1_000_000.0);
        for layer in 0..3 {
            assert_eq!(spec.layer(layer), Ok(LayerKind::Attention));
        }
        // A sliding layer is refused when it is built, naming the layer.
        let sliding = family_spec(&config(with(
            qwen3_config(),
            serde_json::json!({
                "layer_types": ["full_attention", "sliding_attention", "full_attention"]
            }),
        )))
        .expect("the config loads; the layer does not");
        assert_eq!(sliding.layer(0), Ok(LayerKind::Attention));
        assert_eq!(
            sliding.layer(1),
            Err(BuildError(
                "layer 1 is \"sliding_attention\"; only full attention is supported".into()
            ))
        );
    }

    #[test]
    fn mistral_without_a_window_is_llama() {
        // Hugging Face applies no window here either.
        let spec = spec_of(mistral_config());
        assert_eq!(spec.architecture, Architecture::Llama);
        assert_eq!(spec.names, LLAMA_NAMES);
        assert_eq!(spec.rope_theta, 10000.0);
        assert_eq!(spec.layer(31), Ok(LayerKind::Attention));
        let null = spec_of(with(
            mistral_config(),
            serde_json::json!({"use_sliding_window": null}),
        ));
        assert_eq!(null, spec);
    }

    #[test]
    fn llama3_spec_carries_the_scaling() {
        let spec = spec_of(llama3_config());
        assert_eq!(spec.architecture, Architecture::Llama);
        assert_eq!(
            spec.rope,
            Some(Llama3Scaling {
                factor: 32.0,
                low_freq_factor: 1.0,
                high_freq_factor: 4.0,
                original_max_position_embeddings: 8192.0,
            })
        );
        assert_eq!(spec.rope_theta, 500_000.0);
    }

    #[test]
    fn lfm2_spec_names_and_layer_kinds() {
        let spec = spec_of(lfm2_config());
        assert_eq!(spec.architecture, Architecture::Lfm2);
        assert_eq!(spec.names, LFM2_NAMES);
        assert_eq!(spec.names.operator_norm, "operator_norm");
        assert_eq!(spec.names.ffn_norm, "ffn_norm");
        assert_eq!(
            spec.names.attn_proj,
            ["q_proj", "k_proj", "v_proj", "out_proj"]
        );
        assert_eq!(spec.names.mlp, "feed_forward");
        assert_eq!(spec.names.mlp_proj, ["w1", "w3", "w2"]);
        assert_eq!(spec.rope_theta, 1_000_000.0);
        assert_eq!(spec.layer(0), Ok(LayerKind::Conv));
        assert_eq!(spec.layer(2), Ok(LayerKind::Attention));
        assert_eq!(spec.layer(3), Ok(LayerKind::Conv));
        // Past the list, and the older "attention" name, are attention.
        assert_eq!(spec.layer(4), Ok(LayerKind::Attention));
        let named = spec_of(with(
            lfm2_config(),
            serde_json::json!({"layer_types": ["attention", "mamba"]}),
        ));
        assert_eq!(named.layer(0), Ok(LayerKind::Attention));
        assert_eq!(
            named.layer(1),
            Err(BuildError("unknown layer type \"mamba\" at layer 1".into()))
        );
    }

    #[test]
    fn the_family_is_read_from_model_type_and_layer_types() {
        let arch = |changes: serde_json::Value| spec_of(with(llama_config(), changes)).architecture;
        let types = serde_json::json!({"layer_types": ["full_attention"]});
        assert_eq!(
            arch(serde_json::json!({"model_type": "lfm2"})),
            Architecture::Lfm2
        );
        assert_eq!(
            arch(serde_json::json!({"model_type": "LFM2"})),
            Architecture::Lfm2
        );
        assert_eq!(
            arch(serde_json::json!({"layer_types": ["conv", "full_attention"]})),
            Architecture::Lfm2
        );
        // Without a model_type, layer_types alone mean LFM2; with one, they do not.
        assert_eq!(
            arch(with(serde_json::json!({"model_type": null}), types.clone())),
            Architecture::Lfm2
        );
        assert_eq!(arch(types), Architecture::Llama);
        assert_eq!(
            arch(serde_json::json!({"model_type": "qwen3"})),
            Architecture::Llama
        );
    }

    #[test]
    fn supported_configs_load() {
        // A RoPE object with neither a type nor a factor is plain RoPE.
        let plain = refused(serde_json::json!({"rope_scaling": {"rope_theta": 1000000.0}}))
            .expect("plain rope");
        assert_eq!(plain.rope, None);
        // "swish" is another name of SiLU, and no activation means SiLU.
        for act in [serde_json::json!("swish"), serde_json::Value::Null] {
            let spec = refused(serde_json::json!({"hidden_act": act})).expect("silu");
            assert_eq!(spec.act, Act::Silu);
        }
        // Mistral-7B-v0.1 sets a window without use_sliding_window.
        assert!(
            refused(serde_json::json!({"use_sliding_window": null, "sliding_window": 4096}))
                .is_ok()
        );
    }

    #[test]
    fn unsupported_config_features_are_named() {
        let cases = [
            (
                serde_json::json!({"hidden_act": "gelu_pytorch_tanh"}),
                "hidden_act \"gelu_pytorch_tanh\" is not supported: the MLP computes SiLU (silu) only",
            ),
            (
                serde_json::json!({"rope_scaling": {"rope_type": "llama3", "factor": 8.0}}),
                "rope_scaling type \"llama3\" needs a positive number for low_freq_factor",
            ),
            (
                serde_json::json!({"rope_scaling": {
                    "rope_type": "llama3", "factor": 8.0, "low_freq_factor": 4.0,
                    "high_freq_factor": 1.0, "original_max_position_embeddings": 8192
                }}),
                "rope_scaling type \"llama3\" needs high_freq_factor above low_freq_factor, got 1 and 4",
            ),
            (
                serde_json::json!({"rope_scaling": {"type": "linear", "factor": 2.0}}),
                "rope_scaling type \"linear\" is not supported: only \"default\" and \"llama3\" RoPE are",
            ),
            (
                serde_json::json!({"rope_scaling": {"rope_type": "dynamic", "factor": 2.0}}),
                "rope_scaling type \"dynamic\" is not supported: only \"default\" and \"llama3\" RoPE are",
            ),
            (
                serde_json::json!({"rope_parameters": {"rope_type": "yarn", "rope_theta": 1e6}}),
                "rope_parameters type \"yarn\" is not supported: only \"default\" and \"llama3\" RoPE are",
            ),
            (
                serde_json::json!({"rope_scaling": {"rope_type": "longrope", "factor": 2.0}}),
                "rope_scaling type \"longrope\" is not supported: only \"default\" and \"llama3\" RoPE are",
            ),
            (
                serde_json::json!({"use_sliding_window": true}),
                "sliding_window 32768 is not supported: attention always covers the whole sequence",
            ),
            (
                serde_json::json!({"use_sliding_window": true, "sliding_window": null}),
                "sliding_window unset is not supported: attention always covers the whole sequence",
            ),
            (
                serde_json::json!({"rope_scaling": {"factor": 4.0}}),
                "rope_scaling type \"unnamed\" is not supported: only \"default\" and \"llama3\" RoPE are",
            ),
        ];
        for (changes, message) in cases {
            assert_eq!(
                refused(changes.clone()),
                Err(BuildError(message.into())),
                "{changes}"
            );
        }
    }

    #[test]
    fn the_activation_is_refused_before_the_rope_and_the_window() {
        let both = refused(serde_json::json!({
            "hidden_act": "gelu",
            "rope_scaling": {"type": "linear", "factor": 2.0},
            "use_sliding_window": true
        }));
        assert_eq!(
            both,
            Err(BuildError(
                "hidden_act \"gelu\" is not supported: the MLP computes SiLU (silu) only".into()
            ))
        );
        let rope_and_window = refused(serde_json::json!({
            "rope_scaling": {"type": "linear", "factor": 2.0},
            "use_sliding_window": true
        }));
        assert_eq!(
            rope_and_window,
            Err(BuildError(
                "rope_scaling type \"linear\" is not supported: only \"default\" and \"llama3\" RoPE are".into()
            ))
        );
    }
}
