//! The logits of a real Llama 3 checkpoint, such as Llama-3.2-1B, against
//! Hugging Face transformers' on 8 fixed prompts.
//!
//! The test reads `FERROTENSOR_LLAMA3_MODEL` (the checkpoint directory) and
//! `FERROTENSOR_LLAMA3_REAL` (the output directory of
//! `scripts/gen_family_fixtures.py --real MODEL_DIR OUT_DIR`), and skips
//! without them. The tiny fixture of the Llama 3 scaling is the `llama3` row
//! of `family_parity.rs`.

#![allow(clippy::expect_used)]

mod common;

use common::{gpu, host};
use ferrotensor_graph::model_builder::Lfm2ModelBuilder;
use ferrotensor_loader::{ModelLoader, WeightsContext};
use serde::Deserialize;
use std::path::PathBuf;

/// Logits agree with transformers within this absolute error, as in the
/// parity tests of `family_parity.rs`.
const REFERENCE_TOL: f32 = 1e-3;

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// A real checkpoint against transformers on 8 fixed prompts, with f32 base
/// weights. It prints the largest and the mean absolute difference of the
/// logits before it asserts, so a failing run still reports them.
#[test]
#[ignore = "needs a GPU, a Llama 3 checkpoint and its reference logits"]
fn real_checkpoint_logits_match_transformers() {
    let (Some(model_dir), Some(reference)) = (
        std::env::var_os("FERROTENSOR_LLAMA3_MODEL").map(PathBuf::from),
        std::env::var_os("FERROTENSOR_LLAMA3_REAL").map(PathBuf::from),
    ) else {
        eprintln!(
            "skipping the real-checkpoint check; set FERROTENSOR_LLAMA3_MODEL and FERROTENSOR_LLAMA3_REAL (the output of scripts/gen_family_fixtures.py --real)"
        );
        return;
    };
    #[derive(Deserialize)]
    struct Prompts {
        ids: Vec<Vec<u32>>,
    }
    let prompts: Prompts = serde_json::from_str(
        &std::fs::read_to_string(reference.join("cases.json")).expect("read cases.json"),
    )
    .expect("parse cases.json");
    let bytes = std::fs::read(reference.join("logits.safetensors")).expect("read logits");
    let file = safetensors::SafeTensors::deserialize(&bytes).expect("parse logits");

    let gpu = gpu();
    let config = ModelLoader::load_config(model_dir.join("config.json")).expect("config");
    let mut weights = WeightsContext::new();
    weights.load_model_dir(&model_dir).expect("weights");
    let model = Lfm2ModelBuilder::new(gpu.ctx.clone(), gpu.engine.clone(), &weights, config)
        .build()
        .expect("build");

    let (mut worst, mut sum, mut count) = (0f32, 0f64, 0usize);
    for (n, ids) in prompts.ids.iter().enumerate() {
        let view = file
            .tensor(&format!("logits.{n}"))
            .expect("reference logits");
        let expected: Vec<f32> = view
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        let actual = host(&gpu, &model.forward(&gpu.engine, ids, 0, None, None));
        assert_eq!(actual.len(), expected.len(), "prompt {n}: logits shape");
        assert!(
            actual.iter().all(|v| v.is_finite()),
            "prompt {n}: non-finite"
        );
        let diff = max_abs_diff(&actual, &expected);
        let mean = actual
            .iter()
            .zip(&expected)
            .map(|(a, e)| f64::from((a - e).abs()))
            .sum::<f64>()
            / actual.len() as f64;
        eprintln!(
            "prompt {n}: {} tokens, max abs diff {diff:e}, mean abs diff {mean:e}",
            ids.len()
        );
        worst = worst.max(diff);
        sum += mean;
        count += 1;
    }
    eprintln!(
        "LLAMA3_REAL_LOGITS max_abs_diff={worst:e} mean_abs_diff={:e} prompts={count}",
        sum / count as f64
    );
    assert!(
        worst <= REFERENCE_TOL,
        "max abs diff {worst} > {REFERENCE_TOL}"
    );
}
