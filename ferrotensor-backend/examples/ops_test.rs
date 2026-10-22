//! Matmul, RMSNorm, SiLU and softmax through `ComputeEngine`, printed and
//! checked against hand-computed values.
//!
//! Run with: `cargo run -p ferrotensor-backend --example ops_test`
//!
//! Output:
//!
//! ```text
//! --- Testing MatMul ---
//! MatMul Result: [1.0, 2.0, 3.0, 4.0]
//! --- Testing RMSNorm ---
//! RMS Result: [0.36514813, 0.73029625, 1.0954444, 1.4605925]
//! --- Testing SiLU ---
//! SiLU Result: [0.7310586, -0.26894143]
//! --- Testing Softmax ---
//! Softmax Result: [0.2689414, 0.7310586]
//! ```

use ferrotensor_backend::{ComputeEngine, UnifiedTensor, WgpuContext};

fn assert_close(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(want) {
        assert!((g - w).abs() < 1e-5, "got {g}, expected {w}");
    }
}

fn main() {
    env_logger::init();
    pollster::block_on(async {
        let ctx = WgpuContext::new().await;
        let engine = ComputeEngine::new(&ctx);

        println!("--- Testing MatMul ---");
        // 2x2 * 2x2 = 2x2
        // [1 2]   [1 0]   [1 0]
        // [3 4] * [0 1] = [3 4]
        let a = UnifiedTensor::new(&ctx, &[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = UnifiedTensor::new(&ctx, &[1.0, 0.0, 0.0, 1.0], &[2, 2]); // Identity
        let out = UnifiedTensor::empty(&ctx, &[2, 2]);
        engine.matmul(&ctx, &a, &b, &out);
        let got = out.to_vec(&ctx).await;
        println!("MatMul Result: {:?}", got);
        assert_close(&got, &[1.0, 2.0, 3.0, 4.0]);

        println!("--- Testing RMSNorm ---");
        // [1, 2, 3, 4] -> Mean Sq = (1+4+9+16)/4 = 7.5. RMS = sqrt(7.5) ~= 2.738
        // Out ~= [1/2.738, 2/2.738, ...]
        let rms_in = UnifiedTensor::new(&ctx, &[1.0, 2.0, 3.0, 4.0], &[4]);
        let rms_w = UnifiedTensor::new(&ctx, &[1.0, 1.0, 1.0, 1.0], &[4]);
        let rms_out = UnifiedTensor::empty(&ctx, &[4]);
        engine.rms_norm(&ctx, &rms_in, &rms_w, &rms_out);
        let got = rms_out.to_vec(&ctx).await;
        println!("RMS Result: {:?}", got);
        assert_close(&got, &[0.36514813, 0.73029625, 1.0954444, 1.4605925]);

        println!("--- Testing SiLU ---");
        // SiLU(1) = 1 / (1 + e^-1) ~= 0.731
        let silu_in = UnifiedTensor::new(&ctx, &[1.0, -1.0], &[2]);
        let silu_out = UnifiedTensor::empty(&ctx, &[2]);
        engine.silu(&ctx, &silu_in, &silu_out);
        let got = silu_out.to_vec(&ctx).await;
        println!("SiLU Result: {:?}", got);
        assert_close(&got, &[0.7310586, -0.26894143]);

        println!("--- Testing Softmax ---");
        // Softmax([1, 2]) => [0.2689, 0.7310]
        let sm_in = UnifiedTensor::new(&ctx, &[1.0, 2.0], &[2]);
        let sm_out = UnifiedTensor::empty(&ctx, &[2]);
        engine.softmax(&ctx, &sm_in, &sm_out);
        let got = sm_out.to_vec(&ctx).await;
        println!("Softmax Result: {:?}", got);
        assert_close(&got, &[0.2689414, 0.7310586]);
    });
}
