//! Training from scratch: a randomly initialised Llama-style checkpoint, and
//! trained weights written back as one.
//!
//! [`write_random_llama`] writes `config.json` and `model.safetensors` in the
//! Hugging Face Llama layout (`Linear.weight` as `[out, in]`, f32, a separate
//! `lm_head`), so the model loads like a downloaded checkpoint.
//! [`write_hf_weights`] turns the weights a training run saves (linears as
//! `[in, out]`, the layout the model computes with) back into that layout.

use crate::checkpoint::LoraCheckpoint;
use crate::classify::SplitMix64;
use crate::merge::{safetensors_header, write_through_temp_file, MERGED_WEIGHTS_FILE};
use crate::peft::transpose;
use safetensors::Dtype;
use std::io::Write;
use std::path::Path;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Standard deviation of the normal initialisation of the embedding and
/// every linear weight, as in GPT-2.
pub const INIT_STD: f32 = 0.02;

/// Sizes of a Llama-style model with as many key/value heads as query heads
/// and untied embeddings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LlamaSize {
    /// Decoder blocks.
    pub layers: usize,
    /// Width of the residual stream.
    pub hidden: usize,
    /// Attention heads; `hidden / heads` is the head size.
    pub heads: usize,
    /// Width of the SwiGLU hidden layer.
    pub mlp_hidden: usize,
    /// Longest sequence in tokens, written as `max_position_embeddings`.
    pub context: usize,
    /// Tokens in the vocabulary.
    pub vocab: usize,
}

impl LlamaSize {
    /// The Hugging Face name and shape of every tensor, linears as
    /// `[out, in]`, in the order [`write_random_llama`] draws them.
    pub fn tensor_shapes(&self) -> Vec<(String, Vec<usize>)> {
        let (h, m) = (self.hidden, self.mlp_hidden);
        let mut shapes = vec![("model.embed_tokens.weight".to_string(), vec![self.vocab, h])];
        for layer in 0..self.layers {
            for (name, shape) in [
                ("input_layernorm", vec![h]),
                ("self_attn.q_proj", vec![h, h]),
                ("self_attn.k_proj", vec![h, h]),
                ("self_attn.v_proj", vec![h, h]),
                ("self_attn.o_proj", vec![h, h]),
                ("post_attention_layernorm", vec![h]),
                ("mlp.gate_proj", vec![m, h]),
                ("mlp.up_proj", vec![m, h]),
                ("mlp.down_proj", vec![h, m]),
            ] {
                shapes.push((format!("model.layers.{layer}.{name}.weight"), shape));
            }
        }
        shapes.push(("model.norm.weight".to_string(), vec![h]));
        shapes.push(("lm_head.weight".to_string(), vec![self.vocab, h]));
        shapes
    }

    /// Number of weights in the model.
    pub fn parameters(&self) -> usize {
        self.tensor_shapes()
            .iter()
            .map(|(_, shape)| shape.iter().product::<usize>())
            .sum()
    }

    /// The `config.json` of the model, readable by Ferrotensor and by
    /// `transformers` (`LlamaForCausalLM`).
    pub fn config(&self) -> serde_json::Value {
        serde_json::json!({
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": self.hidden,
            "intermediate_size": self.mlp_hidden,
            "num_attention_heads": self.heads,
            "num_key_value_heads": self.heads,
            "num_hidden_layers": self.layers,
            "vocab_size": self.vocab,
            "max_position_embeddings": self.context,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "hidden_act": "silu",
            "attention_bias": false,
            "mlp_bias": false,
            "tie_word_embeddings": false,
            "initializer_range": INIT_STD,
        })
    }
}

/// Writes `config.json` and a randomly initialised `model.safetensors` of
/// `size` into `dir`: norm scales are ones, every other weight is drawn
/// from N(0, [`INIT_STD`]²) by a SplitMix64 generator seeded with `seed`, so
/// the same seed gives the same file on every platform.
///
/// # Errors
/// I/O errors while writing; the weights file is written through a
/// temporary file and never left half written.
pub fn write_random_llama(dir: &Path, size: &LlamaSize, seed: u64) -> Result<()> {
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_string_pretty(&size.config())?,
    )?;
    let mut rng = SplitMix64(seed);
    let tensors = size
        .tensor_shapes()
        .into_iter()
        .map(|(name, shape)| {
            let n = shape.iter().product();
            let values = if shape.len() == 1 {
                vec![1.0; n]
            } else {
                (0..n)
                    .map(|_| INIT_STD * standard_normal(&mut rng))
                    .collect()
            };
            (name, shape, values)
        })
        .collect();
    write_f32_safetensors(&dir.join(MERGED_WEIGHTS_FILE), tensors)
}

/// Writes the tensors of `trained`, a file of trained base weights of a
/// Llama-style model, to `path` as an F32 Hugging Face checkpoint: every
/// 2-D tensor but `model.embed_tokens.weight` is a linear weight and is
/// transposed from `[in, out]` to `[out, in]`; the embedding and the norm
/// scales are written as they are.
///
/// # Errors
/// I/O errors while writing; `path` keeps its old contents on an error.
pub fn write_hf_weights(trained: LoraCheckpoint, path: &Path) -> Result<()> {
    let tensors = trained
        .tensors
        .into_iter()
        .map(|(name, (shape, values))| {
            let linear = match shape[..] {
                [rows, cols] if name != "model.embed_tokens.weight" => Some((rows, cols)),
                _ => None,
            };
            match linear {
                Some((rows, cols)) => (name, vec![cols, rows], transpose(&values, rows, cols)),
                None => (name, shape, values),
            }
        })
        .collect();
    write_f32_safetensors(path, tensors)
}

/// Writes `tensors` (name, shape, values) as one F32 safetensors file,
/// sorted by name, through a temporary file.
fn write_f32_safetensors(
    path: &Path,
    mut tensors: Vec<(String, Vec<usize>, Vec<f32>)>,
) -> Result<()> {
    tensors.sort_by(|a, b| a.0.cmp(&b.0));
    let header = safetensors_header(
        tensors
            .iter()
            .map(|(name, shape, _)| (name.as_str(), Dtype::F32, shape.as_slice())),
    )?;
    write_through_temp_file(path, |out| {
        out.write_all(&(header.len() as u64).to_le_bytes())?;
        out.write_all(&header)?;
        for value in tensors.iter().flat_map(|(_, _, values)| values) {
            out.write_all(&value.to_le_bytes())?;
        }
        Ok(())
    })?;
    Ok(())
}

/// One draw from N(0, 1) by Box-Muller from two 53-bit uniforms; `1 - u`
/// keeps the logarithm's argument in (0, 1].
fn standard_normal(rng: &mut SplitMix64) -> f32 {
    let mut uniform = || (rng.next() >> 11) as f64 / (1u64 << 53) as f64;
    let u1 = 1.0 - uniform();
    let u2 = uniform();
    ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use ferrotensor_loader::{ModelLoader, WeightsContext};
    use std::collections::HashMap;
    use std::path::PathBuf;

    /// A fresh directory under the system temp dir.
    fn temp_dir(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("ferrotensor-pretrain-{tag}-{}", std::process::id()));
        if dir.exists() {
            std::fs::remove_dir_all(&dir).unwrap();
        }
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    const ONE_LAYER: LlamaSize = LlamaSize {
        layers: 1,
        hidden: 8,
        heads: 2,
        mlp_hidden: 16,
        context: 32,
        vocab: 259,
    };

    #[test]
    fn the_written_directory_loads_with_every_tensor_the_builder_reads() {
        let dir = temp_dir("loads");
        write_random_llama(&dir, &ONE_LAYER, 1).unwrap();
        let config = ModelLoader::load_config(dir.join("config.json")).unwrap();
        assert_eq!(
            (
                config.hidden_size,
                config.intermediate_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.num_hidden_layers,
                config.vocab_size,
                config.max_position_embeddings,
                config.tie_word_embeddings,
            ),
            (8, 16, 2, Some(2), 1, 259, Some(32), Some(false))
        );
        let mut weights = WeightsContext::new();
        weights.load_model_dir(&dir).unwrap();
        // The names Lfm2ModelBuilder reads for one Llama block, written out
        // by hand, with Hugging Face [out, in] shapes.
        let expected: HashMap<&str, Vec<usize>> = HashMap::from([
            ("model.embed_tokens.weight", vec![259, 8]),
            ("model.layers.0.input_layernorm.weight", vec![8]),
            ("model.layers.0.self_attn.q_proj.weight", vec![8, 8]),
            ("model.layers.0.self_attn.k_proj.weight", vec![8, 8]),
            ("model.layers.0.self_attn.v_proj.weight", vec![8, 8]),
            ("model.layers.0.self_attn.o_proj.weight", vec![8, 8]),
            ("model.layers.0.post_attention_layernorm.weight", vec![8]),
            ("model.layers.0.mlp.gate_proj.weight", vec![16, 8]),
            ("model.layers.0.mlp.up_proj.weight", vec![16, 8]),
            ("model.layers.0.mlp.down_proj.weight", vec![8, 16]),
            ("model.norm.weight", vec![8]),
            ("lm_head.weight", vec![259, 8]),
        ]);
        let mut names: Vec<&str> = weights.tensor_names().collect();
        names.sort_unstable();
        let mut expected_names: Vec<&str> = expected.keys().copied().collect();
        expected_names.sort_unstable();
        assert_eq!(names, expected_names);
        for (name, shape) in &expected {
            let (values, read_shape) = weights.read_f32_tensor(name).unwrap();
            assert_eq!(&read_shape, shape, "{name}");
            if shape.len() == 1 {
                assert!(values.iter().all(|&v| v == 1.0), "{name} is all ones");
            }
        }
        // 259 * 8 twice, four 8 x 8, three 16 x 8 and three norms of 8.
        assert_eq!(ONE_LAYER.parameters(), 2 * 2072 + 4 * 64 + 3 * 128 + 3 * 8);
        drop(weights);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn the_seed_fixes_the_weights() {
        let read = |seed: u64| {
            let dir = temp_dir(&format!("seed-{seed}"));
            write_random_llama(&dir, &ONE_LAYER, seed).unwrap();
            let bytes = std::fs::read(dir.join(MERGED_WEIGHTS_FILE)).unwrap();
            std::fs::remove_dir_all(dir).unwrap();
            bytes
        };
        assert_eq!(read(7), read(7));
        assert_ne!(read(7), read(8));
    }

    #[test]
    fn initial_weights_have_mean_zero_and_std_init_std() {
        let size = LlamaSize {
            hidden: 128,
            ..ONE_LAYER
        };
        let dir = temp_dir("moments");
        write_random_llama(&dir, &size, 3).unwrap();
        let mut weights = WeightsContext::new();
        weights.load_model_dir(&dir).unwrap();
        let (values, _) = weights
            .read_f32_tensor("model.embed_tokens.weight")
            .unwrap();
        let n = values.len() as f64;
        let mean = values.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
        let var = values
            .iter()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>()
            / n;
        // n = 259 * 128 = 33152 draws: the sample mean has standard error
        // 0.02 / sqrt(n) = 1.1e-4 and the sample std about 0.02 / sqrt(2n) =
        // 7.8e-5; the bounds are five standard errors.
        assert!(mean.abs() < 5.5e-4, "mean {mean}");
        assert!((var.sqrt() - 0.02).abs() < 3.9e-4, "std {}", var.sqrt());
        drop(weights);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn hf_weights_transpose_linears_and_keep_the_embedding_and_norms() {
        let tensors = HashMap::from([
            (
                "model.embed_tokens.weight".to_string(),
                (vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ),
            (
                "lm_head.weight".to_string(),
                (vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ),
            ("model.norm.weight".to_string(), (vec![2], vec![0.5, 2.0])),
        ]);
        let dir = temp_dir("hf-weights");
        let path = dir.join(MERGED_WEIGHTS_FILE);
        let trained = LoraCheckpoint {
            tensors,
            metadata: None,
        };
        write_hf_weights(trained, &path).unwrap();
        let mut weights = WeightsContext::new();
        weights.load_file(&path).unwrap();
        let read = |name: &str| weights.read_f32_tensor(name).unwrap();
        assert_eq!(
            read("model.embed_tokens.weight"),
            (vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])
        );
        // [in 2, out 3] becomes [out 3, in 2]: column j of the input is row j.
        assert_eq!(
            read("lm_head.weight"),
            (vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![3, 2])
        );
        assert_eq!(read("model.norm.weight"), (vec![0.5, 2.0], vec![2]));
        assert!(!path.with_extension("safetensors.tmp").exists());
        drop(weights);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
