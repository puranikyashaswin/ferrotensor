use rusty_backend::{ComputeEngine, UnifiedTensor, WgpuContext};

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
        println!("MatMul Result: {:?}", out.to_vec(&ctx).await);

        println!("--- Testing RMSNorm ---");
        // [1, 2, 3, 4] -> Mean Sq = (1+4+9+16)/4 = 7.5. RMS = sqrt(7.5) ~= 2.738
        // Out ~= [1/2.738, 2/2.738, ...]
        let rms_in = UnifiedTensor::new(&ctx, &[1.0, 2.0, 3.0, 4.0], &[4]);
        let rms_w = UnifiedTensor::new(&ctx, &[1.0, 1.0, 1.0, 1.0], &[4]);
        let rms_out = UnifiedTensor::empty(&ctx, &[4]);
        engine.rms_norm(&ctx, &rms_in, &rms_w, &rms_out);
        println!("RMS Result: {:?}", rms_out.to_vec(&ctx).await);

        println!("--- Testing SiLU ---");
        // SiLU(1) = 1 / (1 + e^-1) ~= 0.731
        let silu_in = UnifiedTensor::new(&ctx, &[1.0, -1.0], &[2]);
        let silu_out = UnifiedTensor::empty(&ctx, &[2]);
        engine.silu(&ctx, &silu_in, &silu_out);
        println!("SiLU Result: {:?}", silu_out.to_vec(&ctx).await);

        println!("--- Testing Softmax ---");
        // Softmax([1, 2]) => [0.2689, 0.7310]
        let sm_in = UnifiedTensor::new(&ctx, &[1.0, 2.0], &[2]);
        let sm_out = UnifiedTensor::empty(&ctx, &[2]);
        engine.softmax(&ctx, &sm_in, &sm_out);
        println!("Softmax Result: {:?}", sm_out.to_vec(&ctx).await);
    });
}
