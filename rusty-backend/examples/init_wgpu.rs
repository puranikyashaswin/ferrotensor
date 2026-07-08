use rusty_backend::WgpuContext;

fn main() {
    env_logger::init();
    pollster::block_on(async {
        println!("Initializing WGPU Context...");
        let _ctx = WgpuContext::new().await;
        println!("WGPU Context initialized successfully!");
    });
}
