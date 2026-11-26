//! Value and gradient checks for a LoRA-adapted [`Linear`] layer against a
//! plain CPU reference written from the math (it calls no Ferrotensor op).
//!
//! These need a GPU, so they are `#[ignore]`d by default. Run them with:
//!
//! ```text
//! cargo test --release -p ferrotensor-graph --test lora_linear -- --ignored --test-threads=1
//! ```

#![allow(clippy::expect_used)]

use ferrotensor_autograd::Tensor;
use ferrotensor_backend::{ComputeEngine, UnifiedTensor, WgpuContext};
use ferrotensor_graph::{Linear, LoRAAdapter};
use std::sync::Arc;

const ROWS: usize = 3;
const IN_DIM: usize = 4;
const OUT_DIM: usize = 5;
const RANK: usize = 2;
const ALPHA: f32 = 6.0;
/// `ALPHA / RANK`, written out so the reference does not share the formula.
const SCALE: f32 = 3.0;

/// Forward values agree within this absolute error (f32, summation order only).
const FORWARD_TOL: f32 = 1e-5;
/// Gradients agree within this absolute error.
const GRAD_TOL: f32 = 1e-4;

struct Setup {
    ctx: Arc<WgpuContext>,
    engine: ComputeEngine,
    x: Tensor,
    x_data: Vec<f32>,
    weight_data: Vec<f32>,
    lora: LoRAAdapter,
    a_data: Vec<f32>,
    b_data: Vec<f32>,
    layer: Linear,
}

fn setup() -> Setup {
    let ctx = Arc::new(pollster::block_on(WgpuContext::new()));
    let engine = ComputeEngine::new(&ctx);
    let x_data: Vec<f32> = (0..ROWS * IN_DIM).map(|i| 0.1 * (i + 1) as f32).collect();
    let weight_data: Vec<f32> = (0..IN_DIM * OUT_DIM)
        .map(|i| 0.05 * i as f32 - 0.4)
        .collect();
    let b_data: Vec<f32> = (0..RANK * OUT_DIM).map(|i| 0.2 - 0.03 * i as f32).collect();
    let x = Tensor::frozen(
        ctx.clone(),
        UnifiedTensor::new(&ctx, &x_data, &[ROWS, IN_DIM]),
    );
    let weight = Tensor::frozen(
        ctx.clone(),
        UnifiedTensor::new(&ctx, &weight_data, &[IN_DIM, OUT_DIM]),
    );
    let mut lora = LoRAAdapter::new_seeded(ctx.clone(), IN_DIM, OUT_DIM, RANK, ALPHA, 7);
    // `b` starts at zero, which would hide any error in the update path.
    lora.b = Tensor::new(
        ctx.clone(),
        UnifiedTensor::new(&ctx, &b_data, &[RANK, OUT_DIM]),
    );
    let a_data = pollster::block_on(lora.a.data.to_vec(&ctx)).expect("GPU readback");
    let layer = Linear::with_lora(weight, lora.clone());
    Setup {
        ctx,
        engine,
        x,
        x_data,
        weight_data,
        lora,
        a_data,
        b_data,
        layer,
    }
}

/// Row-major `[m, k] x [k, n]`.
fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            out[i * n + j] = (0..k).map(|p| a[i * k + p] * b[p * n + j]).sum();
        }
    }
    out
}

fn transpose(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0; a.len()];
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = a[i * cols + j];
        }
    }
    out
}

fn assert_close(name: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol,
            "{name}[{i}]: got {g}, want {w} (tol {tol})"
        );
    }
}

#[test]
#[ignore = "needs a GPU"]
fn lora_forward_matches_cpu_reference() {
    let s = setup();
    let out = s.layer.forward(&s.engine, &s.x);
    assert_eq!(out.data.shape, vec![ROWS, OUT_DIM]);
    let got = pollster::block_on(out.data.to_vec(&s.ctx)).expect("GPU readback");

    let base = matmul(&s.x_data, &s.weight_data, ROWS, IN_DIM, OUT_DIM);
    let xa = matmul(&s.x_data, &s.a_data, ROWS, IN_DIM, RANK);
    let update = matmul(&xa, &s.b_data, ROWS, RANK, OUT_DIM);
    let want: Vec<f32> = base
        .iter()
        .zip(&update)
        .map(|(w, u)| w + SCALE * u)
        .collect();
    assert_close("forward", &got, &want, FORWARD_TOL);
}

#[test]
#[ignore = "needs a GPU"]
fn lora_gradients_match_cpu_reference() {
    let s = setup();
    let out = s.layer.forward(&s.engine, &s.x);
    out.backward(&s.engine);

    // The root gradient defaults to ones.
    let g = vec![1.0_f32; ROWS * OUT_DIM];
    let xa = matmul(&s.x_data, &s.a_data, ROWS, IN_DIM, RANK);
    let want_b: Vec<f32> = matmul(&transpose(&xa, ROWS, RANK), &g, RANK, ROWS, OUT_DIM)
        .iter()
        .map(|v| SCALE * v)
        .collect();
    let g_bt = matmul(
        &g,
        &transpose(&s.b_data, RANK, OUT_DIM),
        ROWS,
        OUT_DIM,
        RANK,
    );
    let want_a: Vec<f32> = matmul(
        &transpose(&s.x_data, ROWS, IN_DIM),
        &g_bt,
        IN_DIM,
        ROWS,
        RANK,
    )
    .iter()
    .map(|v| SCALE * v)
    .collect();

    let got_b = pollster::block_on(
        s.lora
            .b
            .grad
            .borrow()
            .as_ref()
            .expect("b has a gradient")
            .to_vec(&s.ctx),
    )
    .expect("GPU readback");
    let got_a = pollster::block_on(
        s.lora
            .a
            .grad
            .borrow()
            .as_ref()
            .expect("a has a gradient")
            .to_vec(&s.ctx),
    )
    .expect("GPU readback");
    assert_close("grad b", &got_b, &want_b, GRAD_TOL);
    assert_close("grad a", &got_a, &want_a, GRAD_TOL);
    assert!(
        s.layer.weight.grad.borrow().is_none(),
        "the frozen weight must get no gradient"
    );
}
