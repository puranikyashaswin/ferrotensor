use rusty_autograd::Tensor;
use rusty_backend::{ComputeEngine, UnifiedTensor, WgpuContext};
use rusty_graph::{Linear, RMSNorm};
use serde::Deserialize;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Deserialize)]
struct Meta {
    batch: usize,
    seq: usize,
    hidden: usize,
    out_features: usize,
    eps: f32,
}

#[derive(Deserialize)]
struct Phase0Fixture {
    meta: Meta,
    input: Vec<Vec<Vec<f32>>>,
    linear_weight_hf_layout: Vec<Vec<f32>>,
    linear_output: Vec<Vec<Vec<f32>>>,
    rms_weight: Vec<f32>,
    rms_output: Vec<Vec<Vec<f32>>>,
}

fn gpu_tests_enabled() -> bool {
    std::env::var("RUSTY_RUN_GPU_TESTS").as_deref() == Ok("1")
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/reference/phase0_linear_rmsnorm.json")
}

fn ensure_fixture(path: &PathBuf) {
    if path.exists() {
        return;
    }
    let generator = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tests/reference/generate_phase0_pure.py");
    if generator.exists() {
        let status = std::process::Command::new("python3")
            .arg(&generator)
            .status();
        if let Ok(status) = status {
            if status.success() && path.exists() {
                return;
            }
        }
    }
    panic!(
        "missing phase0 fixture at {}; run `python3 tests/reference/generate_phase0_pure.py`",
        path.display()
    );
}

fn flatten_3d(data: &[Vec<Vec<f32>>]) -> Vec<f32> {
    let mut out = Vec::new();
    for batch in data {
        for row in batch {
            out.extend_from_slice(row);
        }
    }
    out
}

fn transpose_2d(data: &[Vec<f32>]) -> Vec<f32> {
    let rows = data.len();
    let cols = data.first().map(|row| row.len()).unwrap_or(0);
    let mut out = vec![0.0; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            out[col * rows + row] = data[row][col];
        }
    }
    out
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        assert!(
            diff <= tol,
            "mismatch at index {idx}: actual={a}, expected={e}, diff={diff}, tol={tol}"
        );
    }
}

#[test]
fn linear_and_rmsnorm_match_reference_fixture() {
    if !gpu_tests_enabled() {
        eprintln!("skipping phase0 reference test; set RUSTY_RUN_GPU_TESTS=1");
        return;
    }

    let path = fixture_path();
    ensure_fixture(&path);

    let fixture: Phase0Fixture =
        serde_json::from_str(&fs::read_to_string(path).expect("read fixture"))
            .expect("parse fixture");

    let ctx = Arc::new(pollster::block_on(async { WgpuContext::new().await }));
    let engine = ComputeEngine::new(&ctx);

    let input_flat = flatten_3d(&fixture.input);
    let input = Tensor::new(
        ctx.clone(),
        UnifiedTensor::new(
            &ctx,
            &input_flat,
            &[fixture.meta.batch, fixture.meta.seq, fixture.meta.hidden],
        ),
    );

    let linear_weight = Tensor::new(
        ctx.clone(),
        UnifiedTensor::new(
            &ctx,
            &transpose_2d(&fixture.linear_weight_hf_layout),
            &[fixture.meta.hidden, fixture.meta.out_features],
        ),
    );
    let linear = Linear::new(linear_weight);
    let linear_out = linear.forward(&engine, &input);
    let linear_actual = pollster::block_on(linear_out.data.to_vec(&ctx));
    let linear_expected = flatten_3d(&fixture.linear_output);
    assert_close(&linear_actual, &linear_expected, 1e-3);

    let rms_weight = Tensor::new(
        ctx.clone(),
        UnifiedTensor::new(&ctx, &fixture.rms_weight, &[fixture.meta.hidden]),
    );
    let rms = RMSNorm::new(rms_weight, fixture.meta.eps);
    let rms_out = rms.forward(&engine, &input);
    let rms_actual = pollster::block_on(rms_out.data.to_vec(&ctx));
    let rms_expected = flatten_3d(&fixture.rms_output);
    assert_close(&rms_actual, &rms_expected, 1e-3);
}
