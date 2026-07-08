use rusty_backend::{ComputeEngine, UnifiedTensor, WgpuContext};

fn main() {
    env_logger::init();
    pollster::block_on(async {
        let ctx = WgpuContext::new().await;
        let engine = ComputeEngine::new(&ctx);

        let data_a = vec![1.0, 2.0, 3.0, 4.0];
        let data_b = vec![5.0, 6.0, 7.0, 8.0];
        let data_c = vec![2.0, 2.0, 2.0, 2.0];
        let shape = vec![4];

        let tensor_a = UnifiedTensor::new(&ctx, &data_a, &shape);
        let tensor_b = UnifiedTensor::new(&ctx, &data_b, &shape);
        let tensor_c = UnifiedTensor::new(&ctx, &data_c, &shape);
        let tensor_out = UnifiedTensor::new(&ctx, &[0.0; 4], &shape);

        println!("Executing (A + B) * C...");
        let temp = UnifiedTensor::empty(&ctx, &shape);
        engine.add(&ctx, &tensor_a, &tensor_b, &temp);
        engine.mul(&ctx, &temp, &tensor_c, &tensor_out);

        let result = tensor_out.to_vec(&ctx).await;
        println!("Result: {:?}", result);

        // Verification: (1+5)*2=12, (2+6)*2=16, (3+7)*2=20, (4+8)*2=24
        assert_eq!(result, vec![12.0, 16.0, 20.0, 24.0]);
        println!("Verification Successful!");
    });
}
