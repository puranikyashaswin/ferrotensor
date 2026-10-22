//! Matmul, add, mul and SiLU on the GPU through `ComputeEngine`, each checked
//! against a CPU computation.
//!
//! Run with: `cargo run --release -p ferrotensor-backend --example basic_tensor`
//!
//! Output on an Apple M2 (the device line varies):
//!
//! ```text
//! device: Apple M2 (Metal)
//! matmul [32, 32] x [32, 32]: first values [0.1665039, 0.16603088, 0.16555786], max |gpu - cpu| = 0.00e0
//! add:  first values [1.0, 1.0, 1.0], max |gpu - cpu| = 0.00e0
//! mul:  first values [0.0, 0.0009756088, 0.0019493103], max |gpu - cpu| = 0.00e0
//! silu: first values [0.0, 0.00048851967, 0.0009775162], max |gpu - cpu| = 1.19e-7
//! ```

use ferrotensor_backend::{ComputeEngine, UnifiedTensor, WgpuContext};

const N: usize = 32;

fn max_diff(gpu: &[f32], cpu: &[f32]) -> f32 {
    gpu.iter()
        .zip(cpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

fn main() {
    pollster::block_on(async {
        let ctx = WgpuContext::new().await;
        let engine = ComputeEngine::new(&ctx);
        let info = ctx.adapter.get_info();
        println!("device: {} ({:?})", info.name, info.backend);

        let a_data: Vec<f32> = (0..N * N).map(|i| i as f32 / 1024.0).collect();
        let b_data: Vec<f32> = (0..N * N).map(|i| (1024 - i) as f32 / 1024.0).collect();
        let a = UnifiedTensor::new(&ctx, &a_data, &[N, N]);
        let b = UnifiedTensor::new(&ctx, &b_data, &[N, N]);

        let mut cpu_matmul = vec![0.0f32; N * N];
        for i in 0..N {
            for j in 0..N {
                cpu_matmul[i * N + j] = (0..N).map(|k| a_data[i * N + k] * b_data[k * N + j]).sum();
            }
        }
        let out = UnifiedTensor::empty(&ctx, &[N, N]);
        engine.matmul(&ctx, &a, &b, &out);
        let gpu = out.to_vec(&ctx).await;
        println!(
            "matmul [{N}, {N}] x [{N}, {N}]: first values {:?}, max |gpu - cpu| = {:.2e}",
            &gpu[..3],
            max_diff(&gpu, &cpu_matmul)
        );
        assert!(max_diff(&gpu, &cpu_matmul) < 1e-3);

        let cpu_add: Vec<f32> = a_data.iter().zip(&b_data).map(|(x, y)| x + y).collect();
        let out = UnifiedTensor::empty(&ctx, &[N, N]);
        engine.add(&ctx, &a, &b, &out);
        let gpu = out.to_vec(&ctx).await;
        println!(
            "add:  first values {:?}, max |gpu - cpu| = {:.2e}",
            &gpu[..3],
            max_diff(&gpu, &cpu_add)
        );
        assert!(max_diff(&gpu, &cpu_add) < 1e-6);

        let cpu_mul: Vec<f32> = a_data.iter().zip(&b_data).map(|(x, y)| x * y).collect();
        let out = UnifiedTensor::empty(&ctx, &[N, N]);
        engine.mul(&ctx, &a, &b, &out);
        let gpu = out.to_vec(&ctx).await;
        println!(
            "mul:  first values {:?}, max |gpu - cpu| = {:.2e}",
            &gpu[..3],
            max_diff(&gpu, &cpu_mul)
        );
        assert!(max_diff(&gpu, &cpu_mul) < 1e-6);

        let cpu_silu: Vec<f32> = a_data.iter().map(|x| x / (1.0 + (-x).exp())).collect();
        let out = UnifiedTensor::empty(&ctx, &[N, N]);
        engine.silu(&ctx, &a, &out);
        let gpu = out.to_vec(&ctx).await;
        println!(
            "silu: first values {:?}, max |gpu - cpu| = {:.2e}",
            &gpu[..3],
            max_diff(&gpu, &cpu_silu)
        );
        assert!(max_diff(&gpu, &cpu_silu) < 1e-5);
    });
}
