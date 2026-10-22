//! Upload a tensor to the GPU and read it back.
//!
//! Run with: `cargo run -p ferrotensor-backend --example tensor_creation`
//!
//! Output:
//!
//! ```text
//! uploading [1.0, 2.0, 3.0, 4.0] as shape [2, 2]
//! buffer size: 16 bytes
//! read back: [1.0, 2.0, 3.0, 4.0]
//! ```

use ferrotensor_backend::{UnifiedTensor, WgpuContext};

fn main() {
    pollster::block_on(async {
        let ctx = WgpuContext::new().await;

        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];

        println!("uploading {data:?} as shape {shape:?}");
        let tensor = UnifiedTensor::new(&ctx, &data, &shape);
        println!("buffer size: {} bytes", tensor.size_in_bytes());

        let back = tensor.to_vec(&ctx).await;
        println!("read back: {back:?}");
        assert_eq!(back, data);
    });
}
