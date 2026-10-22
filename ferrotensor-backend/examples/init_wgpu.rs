//! Create the wgpu context (adapter, device and queue).
//!
//! Run with: `cargo run -p ferrotensor-backend --example init_wgpu`
//!
//! Output:
//!
//! ```text
//! Initializing WGPU Context...
//! WGPU Context initialized successfully!
//! ```
//!
//! Set `RUST_LOG=info` to see which adapter was picked.

use ferrotensor_backend::WgpuContext;

fn main() {
    env_logger::init();
    pollster::block_on(async {
        println!("Initializing WGPU Context...");
        let _ctx = WgpuContext::new().await;
        println!("WGPU Context initialized successfully!");
    });
}
