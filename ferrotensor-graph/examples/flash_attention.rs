//! Causal flash attention on the GPU, checked against a plain CPU reference.
//!
//! Runs the `ferrotensor-backend` flash attention forward kernel through
//! `ferrotensor_graph::FlashAttention` on a `[1, 512, 12, 64]` Q/K/V, then
//! recomputes the same attention on the CPU with a materialised score matrix
//! and prints the largest difference.
//!
//! Run with: `cargo run --release -p ferrotensor-graph --example flash_attention`
//!
//! Output on an Apple M2 (timings and the device line vary):
//!
//! ```text
//! device: Apple M2 (Metal)
//! q, k, v: [1, 512, 12, 64] (heads 12, head_dim 64, causal)
//! flash attention forward: 27.06 ms (includes the read back)
//! first values: [0.0, 0.01, 0.02, 0.03, 0.04]
//! max |gpu - cpu reference| = 2.21e-6
//! ```

use ferrotensor_backend::{ComputeEngine, UnifiedTensor, WgpuContext};
use ferrotensor_graph::FlashAttention;
use std::time::Instant;

const SEQ: usize = 512;
const HEADS: usize = 12;
const HEAD_DIM: usize = 64;
const DIM: usize = HEADS * HEAD_DIM;

/// Causal softmax(QK^T / sqrt(d)) V for one batch, layout `[seq, heads, head_dim]`.
fn reference(q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let mut out = vec![0.0f32; SEQ * DIM];
    for h in 0..HEADS {
        for i in 0..SEQ {
            let qi = &q[i * DIM + h * HEAD_DIM..][..HEAD_DIM];
            let mut scores: Vec<f32> = (0..=i)
                .map(|j| {
                    let kj = &k[j * DIM + h * HEAD_DIM..][..HEAD_DIM];
                    scale * qi.iter().zip(kj).map(|(a, b)| a * b).sum::<f32>()
                })
                .collect();
            let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut total = 0.0;
            for s in &mut scores {
                *s = (*s - max).exp();
                total += *s;
            }
            let o = &mut out[i * DIM + h * HEAD_DIM..][..HEAD_DIM];
            for (j, s) in scores.iter().enumerate() {
                let vj = &v[j * DIM + h * HEAD_DIM..][..HEAD_DIM];
                for d in 0..HEAD_DIM {
                    o[d] += s / total * vj[d];
                }
            }
        }
    }
    out
}

fn main() {
    pollster::block_on(async {
        let ctx = WgpuContext::new().await;
        let engine = ComputeEngine::new(&ctx);
        let info = ctx.adapter.get_info();
        println!("device: {} ({:?})", info.name, info.backend);

        let n = SEQ * DIM;
        let q_data: Vec<f32> = (0..n).map(|i| ((i % 100) as f32 - 50.0) / 100.0).collect();
        let k_data = q_data.clone();
        let v_data: Vec<f32> = (0..n).map(|i| (i % 100) as f32 / 100.0).collect();

        let shape = [1, SEQ, HEADS, HEAD_DIM];
        let q = UnifiedTensor::new(&ctx, &q_data, &shape);
        let k = UnifiedTensor::new(&ctx, &k_data, &shape);
        let v = UnifiedTensor::new(&ctx, &v_data, &shape);
        println!("q, k, v: {shape:?} (heads {HEADS}, head_dim {HEAD_DIM}, causal)");

        let attention = FlashAttention::new(DIM, HEADS, true);
        let start = Instant::now();
        let out = attention.forward(&q, &k, &v, &ctx, &engine);
        let gpu = out.to_vec(&ctx).await;
        println!(
            "flash attention forward: {:.2} ms (includes the read back)",
            start.elapsed().as_secs_f64() * 1e3
        );
        println!("first values: {:?}", &gpu[..5]);

        let cpu = reference(&q_data, &k_data, &v_data);
        let max_diff = gpu
            .iter()
            .zip(&cpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("max |gpu - cpu reference| = {max_diff:.2e}");
        assert!(max_diff < 1e-3, "GPU output differs from the CPU reference");
    });
}
