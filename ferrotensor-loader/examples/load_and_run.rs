//! Load packed 8-bit weights from a safetensors file, dequantise them on the
//! GPU, then run two more kernels on the result: `(x + 2) * 2`.
//!
//! Run with: `cargo run -p ferrotensor-loader --example load_and_run [path]`
//!
//! The path defaults to `dummy.safetensors`, which the `gen_dummy` example
//! writes (`tensor_a` holds the bytes 1, 2, 3, 4). The example exits with an
//! error unless the result is [6.0, 8.0, 10.0, 12.0].

use ferrotensor_backend::{ComputeEngine, UnifiedTensor, WgpuContext};
use ferrotensor_loader::WeightsContext;
use std::sync::Arc;

fn main() {
    env_logger::init();
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "dummy.safetensors".to_string());

    // 1. Read the packed bytes from disk.
    let mut weights = WeightsContext::new();
    if let Err(e) = weights.load_file(&path) {
        eprintln!("cannot load {path}: {e}");
        eprintln!("run `cargo run -p ferrotensor-loader --example gen_dummy` first");
        std::process::exit(1);
    }
    let mut packed = Vec::new();
    weights
        .read_bytes("tensor_a", |bytes, _| packed = bytes.to_vec())
        .unwrap();
    println!("tensor_a bytes: {packed:?}");

    pollster::block_on(async {
        // 2. Setup Backend
        let ctx = Arc::new(WgpuContext::new().await);
        let engine = ComputeEngine::new(&ctx);

        // 3. Upload the bytes and dequantise them to f32 on the GPU.
        let shape = vec![packed.len()];
        let gpu_int8 = UnifiedTensor::from_bytes(&ctx, &packed, &shape);
        let gpu_fp32 = UnifiedTensor::empty(&ctx, &shape);
        engine.dequantize(&ctx, &gpu_int8, &gpu_fp32);

        // 4. Sequential Ops instead of Fused
        let gpu_c = UnifiedTensor::new(&ctx, &[2.0, 2.0, 2.0, 2.0], &shape);
        let gpu_out = UnifiedTensor::empty(&ctx, &shape);
        let temp = UnifiedTensor::empty(&ctx, &shape);

        engine.add(&ctx, &gpu_fp32, &gpu_c, &temp);
        engine.mul(&ctx, &temp, &gpu_c, &gpu_out);

        let result = gpu_out.to_vec(&ctx).await;
        println!("Result: {:?}", result);
        assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
        println!("SUCCESS: Pipeline logic verified!");
    });
}
