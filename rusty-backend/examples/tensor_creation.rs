use rusty_backend::{UnifiedTensor, WgpuContext};

fn main() {
    env_logger::init();
    pollster::block_on(async {
        let ctx = WgpuContext::new().await;

        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];

        println!("Creating tensor with data: {:?}", data);
        let tensor = UnifiedTensor::new(&ctx, &data, &shape);

        println!("Tensor created successfully: {:?}", tensor);
        println!("Buffer size: {} bytes", tensor.size_in_bytes());
    });
}
