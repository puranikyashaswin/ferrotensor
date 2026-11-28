//! Host-side RoPE frequencies: the scaling a config asks for, and the table of
//! inverse frequencies the RoPE kernel reads.

use ferrotensor_loader::{Lfm2Config, RopeConfig};

/// The `rope_type: "llama3"` scaling of Llama 3.1 and later.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Llama3Scaling {
    /// Divisor of the low frequencies.
    pub factor: f64,
    /// Wavelengths longer than `original_max_position_embeddings` over this
    /// are divided by `factor`.
    pub low_freq_factor: f64,
    /// Wavelengths shorter than `original_max_position_embeddings` over this
    /// are kept; those in between are blended.
    pub high_freq_factor: f64,
    /// Context length of the model before scaling.
    pub original_max_position_embeddings: f64,
}

impl Llama3Scaling {
    /// The inverse frequency of each of the `head_dim / 2` rotation pairs,
    /// as `transformers` computes it in `_compute_llama3_parameters`, from
    /// the base `theta`.
    ///
    /// The arithmetic runs in `f64` and the result is rounded once to `f32`,
    /// so it differs from the `float32` Hugging Face table by rounding only.
    pub fn inv_freq(&self, head_dim: usize, theta: f64) -> Vec<f32> {
        let low_wavelen = self.original_max_position_embeddings / self.low_freq_factor;
        let high_wavelen = self.original_max_position_embeddings / self.high_freq_factor;
        (0..head_dim / 2)
            .map(|pair| {
                let inv = 1.0 / theta.powf(2.0 * pair as f64 / head_dim as f64);
                let wavelen = 2.0 * std::f64::consts::PI / inv;
                let scaled = if wavelen > low_wavelen {
                    inv / self.factor
                } else if wavelen < high_wavelen {
                    inv
                } else {
                    let smooth = (self.original_max_position_embeddings / wavelen
                        - self.low_freq_factor)
                        / (self.high_freq_factor - self.low_freq_factor);
                    (1.0 - smooth) * inv / self.factor + smooth * inv
                };
                scaled as f32
            })
            .collect()
    }
}

/// The scaling that `config` asks for in `rope_scaling` or `rope_parameters`:
/// `None` for plain RoPE and the Llama 3 scheme for `rope_type: "llama3"`.
///
/// # Errors
/// A message naming the field when the scheme is any other one, or when a
/// Llama 3 field is missing or out of range. A Llama 3 object without
/// `original_max_position_embeddings` takes the config's own
/// `original_max_position_embeddings`, else its `max_position_embeddings`, as
/// `transformers` does.
pub fn rope_scaling(config: &Lfm2Config) -> Result<Option<Llama3Scaling>, String> {
    let mut found = None;
    for (field, rope) in [
        ("rope_scaling", &config.rope_scaling),
        ("rope_parameters", &config.rope_parameters),
    ] {
        let Some(rope) = rope else { continue };
        // An object without a type is plain RoPE unless it scales by a factor.
        let kind = match rope.kind() {
            None if rope.factor.is_none() => "default",
            kind => kind.unwrap_or("unnamed"),
        };
        match kind {
            "default" => {}
            "llama3" => {
                let scaling = llama3_of(field, rope, original_context(config))?;
                found = found.or(Some(scaling));
            }
            other => {
                return Err(format!(
                    "{field} type {other:?} is not supported: only \"default\" and \"llama3\" RoPE are"
                ));
            }
        }
    }
    Ok(found)
}

/// The context length a Llama 3 object falls back to when it names none.
fn original_context(config: &Lfm2Config) -> Option<f64> {
    config
        .original_max_position_embeddings
        .or(config.max_position_embeddings.map(|n| n as f64))
}

/// Reads the four Llama 3 fields of `rope` and checks them; `fallback` stands
/// in for a missing `original_max_position_embeddings`.
fn llama3_of(
    field: &str,
    rope: &RopeConfig,
    fallback: Option<f64>,
) -> Result<Llama3Scaling, String> {
    let need = |name: &str, value: Option<f64>| {
        value
            .filter(|v| v.is_finite() && *v > 0.0)
            .ok_or_else(|| format!("{field} type \"llama3\" needs a positive number for {name}"))
    };
    let scaling = Llama3Scaling {
        factor: need("factor", rope.factor)?,
        low_freq_factor: need("low_freq_factor", rope.low_freq_factor)?,
        high_freq_factor: need("high_freq_factor", rope.high_freq_factor)?,
        original_max_position_embeddings: need(
            "original_max_position_embeddings",
            rope.original_max_position_embeddings.or(fallback),
        )?,
    };
    if scaling.high_freq_factor <= scaling.low_freq_factor {
        return Err(format!(
            "{field} type \"llama3\" needs high_freq_factor above low_freq_factor, got {} and {}",
            scaling.high_freq_factor, scaling.low_freq_factor
        ));
    }
    Ok(scaling)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::excessive_precision)]
mod tests {
    use super::*;

    // Reference tables: `inv_freq` of `_compute_llama3_parameters` in
    // `transformers/modeling_rope_utils.py` (`ROPE_INIT_FUNCTIONS["llama3"]`),
    // run in float32 on the CPU as the library does, from torch 2.15.0 and
    // transformers 5.23.0, with base 500000, `low_freq_factor` 1 and
    // `high_freq_factor` 4. Each config was a `LlamaConfig` with
    // `rope_parameters` set to the llama3 object and the `head_dim` below. The
    // 1B and 8B values are those of the Llama-3.2-1B (checkpoint sha 9535bd9b of
    // the unsloth mirror) and Llama-3.1-8B configs; the tables depend on the
    // config only, not on the weights. The digits are `f32` round trips printed
    // with nine decimals, so clippy's precision lint is off for this module.
    // head size 64, base 500000.0, factor 32.0, original context 8192
    const LLAMA_3_2_1B: [f32; 32] = [
        1.000000000e+00,
        6.636012793e-01,
        4.403666258e-01,
        2.922278345e-01,
        1.939227581e-01,
        1.286873817e-01,
        8.539710194e-02,
        5.666961893e-02,
        3.760603070e-02,
        2.495540865e-02,
        1.656044088e-02,
        1.098952908e-02,
        7.292665076e-03,
        4.839421250e-03,
        3.211446106e-03,
        1.290548011e-03,
        4.295567051e-04,
        9.708286234e-05,
        1.946163866e-05,
        1.291476747e-05,
        8.570255886e-06,
        5.687232260e-06,
        3.774054449e-06,
        2.504467147e-06,
        1.661967417e-06,
        1.102883630e-06,
        7.318749340e-07,
        4.856731266e-07,
        3.222932889e-07,
        2.138742303e-07,
        1.419272024e-07,
        9.418306490e-08,
    ];
    // head size 128, base 500000.0, factor 8.0, original context 8192
    const LLAMA_3_1_8B: [f32; 64] = [
        1.000000000e+00,
        8.146172166e-01,
        6.636012793e-01,
        5.405809879e-01,
        4.403666258e-01,
        3.587302268e-01,
        2.922278345e-01,
        2.380538136e-01,
        1.939227581e-01,
        1.579728127e-01,
        1.286873817e-01,
        1.048309505e-01,
        8.539710194e-02,
        6.956595182e-02,
        5.666961893e-02,
        4.616405070e-02,
        3.760603070e-02,
        3.063452058e-02,
        2.495540865e-02,
        2.032910474e-02,
        1.656044088e-02,
        1.349041983e-02,
        1.098952908e-02,
        8.952259086e-03,
        7.292665076e-03,
        5.940730684e-03,
        4.839421250e-03,
        3.942275885e-03,
        3.211446106e-03,
        2.166570630e-03,
        1.371893683e-03,
        8.567514597e-04,
        5.248460220e-04,
        3.126936499e-04,
        1.785077911e-04,
        9.556212171e-05,
        7.784655463e-05,
        6.341514381e-05,
        5.165906987e-05,
        4.208236714e-05,
        3.428102355e-05,
        2.792590931e-05,
        2.274892904e-05,
        1.853166941e-05,
        1.509621779e-05,
        1.229763893e-05,
        1.001786859e-05,
        8.160727702e-06,
        6.647869668e-06,
        5.415469332e-06,
        4.411534519e-06,
        3.593711881e-06,
        2.927499736e-06,
        2.384791742e-06,
        1.942692506e-06,
        1.582550794e-06,
        1.289173156e-06,
        1.050182618e-06,
        8.554969213e-07,
        6.969025321e-07,
        5.677088097e-07,
        4.624653798e-07,
        3.767322596e-07,
        3.068925878e-07,
    ];
    // head size 16, base 500000.0, factor 8.0, original context 64
    const TINY: [f32; 8] = [
        1.000000000e+00,
        7.940301299e-02,
        4.700753838e-03,
        9.115831344e-04,
        1.767766807e-04,
        3.428102355e-05,
        6.647869668e-06,
        1.289173156e-06,
    ];

    fn config(extra: serde_json::Value) -> Lfm2Config {
        let mut config = serde_json::json!({
            "hidden_size": 8, "num_attention_heads": 2, "num_hidden_layers": 1,
            "intermediate_size": 16, "vocab_size": 8
        });
        for (key, value) in extra.as_object().into_iter().flatten() {
            config[key] = value.clone();
        }
        serde_json::from_value(config).expect("config")
    }

    fn scaling(factor: f64) -> Llama3Scaling {
        Llama3Scaling {
            factor,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position_embeddings: 100.0,
        }
    }

    #[test]
    fn each_regime_has_its_closed_form() {
        // Head size 6, base 100: the inverse frequencies are 1, 100^(-1/3) and
        // 100^(-2/3), with wavelengths 6.28, 29.16 and 135.37. With an original
        // context of 100 the wavelength limits are 25 and 100, so the first pair
        // is kept, the second is blended and the third is divided by the factor.
        // The blended value is worked out by hand: smooth = (100 / 29.164 - 1) / 3
        // = 0.80963, and (1 - smooth) * inv / 8 + smooth * inv.
        let table = scaling(8.0).inv_freq(6, 100.0);
        let expected = [1.0, 0.179_556_2, 0.005_801_986];
        assert_eq!(table.len(), 3);
        for (got, want) in table.iter().zip(expected) {
            assert!(
                (got - want).abs() <= 1e-6 * want,
                "{table:?} vs {expected:?}"
            );
        }
    }

    #[test]
    fn a_factor_of_one_changes_nothing() {
        let table = scaling(1.0).inv_freq(64, 500_000.0);
        assert_eq!(table.len(), 32);
        for (pair, got) in table.iter().enumerate() {
            let plain = 1.0 / 500_000f64.powf(2.0 * pair as f64 / 64.0);
            assert!(
                (f64::from(*got) - plain).abs() <= 1e-6 * plain,
                "pair {pair}: {got} vs {plain}"
            );
        }
    }

    #[test]
    fn both_config_layouts_give_the_same_scaling() {
        let fields = serde_json::json!({
            "factor": 32.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192, "rope_type": "llama3"
        });
        let expected = Llama3Scaling {
            factor: 32.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position_embeddings: 8192.0,
        };
        // transformers 4 and the hub configs: `rope_scaling`.
        let old = config(serde_json::json!({ "rope_scaling": fields }));
        assert_eq!(rope_scaling(&old), Ok(Some(expected)));
        // transformers 5: `rope_parameters`, with the base inside.
        let mut inner = fields;
        inner["rope_theta"] = serde_json::json!(500_000.0);
        let new = config(serde_json::json!({ "rope_parameters": inner }));
        assert_eq!(rope_scaling(&new), Ok(Some(expected)));
        // No scaling at all, and a plain `default` object.
        assert_eq!(rope_scaling(&config(serde_json::json!({}))), Ok(None));
        let plain = config(serde_json::json!({ "rope_scaling": {"rope_type": "default"} }));
        assert_eq!(rope_scaling(&plain), Ok(None));
    }

    #[test]
    fn other_schemes_and_incomplete_llama3_objects_are_refused() {
        for kind in ["linear", "dynamic", "yarn", "longrope"] {
            let c =
                config(serde_json::json!({ "rope_scaling": {"rope_type": kind, "factor": 2.0} }));
            assert_eq!(
                rope_scaling(&c),
                Err(format!(
                    "rope_scaling type {kind:?} is not supported: only \"default\" and \"llama3\" RoPE are"
                ))
            );
        }
        let missing = config(serde_json::json!({
            "rope_scaling": {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                             "high_freq_factor": 4.0}
        }));
        assert_eq!(
            rope_scaling(&missing),
            Err("rope_scaling type \"llama3\" needs a positive number for original_max_position_embeddings".to_owned())
        );
    }

    #[test]
    fn a_missing_original_context_falls_back_as_transformers_does() {
        let object = serde_json::json!({
            "rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
            "high_freq_factor": 4.0
        });
        let want = |original: f64| {
            Ok(Some(Llama3Scaling {
                factor: 8.0,
                low_freq_factor: 1.0,
                high_freq_factor: 4.0,
                original_max_position_embeddings: original,
            }))
        };
        // The config's own `original_max_position_embeddings` comes first.
        let top = config(serde_json::json!({
            "rope_scaling": object, "original_max_position_embeddings": 4096,
            "max_position_embeddings": 131_072
        }));
        assert_eq!(rope_scaling(&top), want(4096.0));
        // Then `max_position_embeddings`.
        let max = config(serde_json::json!({
            "rope_scaling": object, "max_position_embeddings": 131_072
        }));
        assert_eq!(rope_scaling(&max), want(131_072.0));
        // The object's own value beats both.
        let mut own = object;
        own["original_max_position_embeddings"] = serde_json::json!(8192);
        let both = config(serde_json::json!({
            "rope_scaling": own, "original_max_position_embeddings": 4096,
            "max_position_embeddings": 131_072
        }));
        assert_eq!(rope_scaling(&both), want(8192.0));
    }

    #[test]
    fn matches_transformers() {
        // Our table is `f64` arithmetic rounded once, theirs is `float32`
        // arithmetic: they agree to a few `f32` roundings, 1e-5 relative.
        let cases: [(&[f32], usize, f64, f64); 3] = [
            (&LLAMA_3_2_1B, 64, 32.0, 8192.0),
            (&LLAMA_3_1_8B, 128, 8.0, 8192.0),
            (&TINY, 16, 8.0, 64.0),
        ];
        for (expected, head_dim, factor, original) in cases {
            let scaling = Llama3Scaling {
                factor,
                low_freq_factor: 1.0,
                high_freq_factor: 4.0,
                original_max_position_embeddings: original,
            };
            let table = scaling.inv_freq(head_dim, 500_000.0);
            assert_eq!(table.len(), expected.len());
            for (pair, (got, want)) in table.iter().zip(expected).enumerate() {
                assert!(
                    (got - want).abs() <= 1e-5 * want,
                    "head size {head_dim}, pair {pair}: {got} vs {want}"
                );
            }
        }
    }
}
