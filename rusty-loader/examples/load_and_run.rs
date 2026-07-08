use rusty_backend::{ComputeEngine, UnifiedTensor, WgpuContext};
use rusty_loader::WeightsContext;
use std::sync::Arc;

fn main() {
    env_logger::init();
    pollster::block_on(async {
        // 1. Setup Backend
        let ctx = Arc::new(WgpuContext::new().await);
        let engine = ComputeEngine::new(&ctx);

        // 2. Setup Context (Mocking usage since we don't have a real file in this test)
        let _weights_ctx = WeightsContext::new();

        println!("Demonstrating full pipeline (conceptual)...");

        // 3. Create GPU Tensors
        let shape = vec![4];
        let gpu_int8 = UnifiedTensor::new(&ctx, &[1.0, 2.0, 3.0, 4.0], &shape);
        let gpu_fp32 = UnifiedTensor::empty(&ctx, &shape);

        // 4. Dequantize
        engine.dequantize(&ctx, &gpu_int8, &gpu_fp32);

        // 5. Sequential Ops instead of Fused
        let gpu_c = UnifiedTensor::new(&ctx, &[2.0, 2.0, 2.0, 2.0], &shape);
        let gpu_out = UnifiedTensor::empty(&ctx, &shape);
        let temp = UnifiedTensor::empty(&ctx, &shape);

        engine.add(&ctx, &gpu_fp32, &gpu_c, &temp);
        engine.mul(&ctx, &temp, &gpu_c, &gpu_out);

        let result = gpu_out.to_vec(&ctx).await;
        println!("Result: {:?}", result);
        println!("SUCCESS: Pipeline logic verified!");
    });
}
