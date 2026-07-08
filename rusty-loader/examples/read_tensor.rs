use rusty_loader::WeightsContext;

fn main() {
    let _ctx = WeightsContext::new();
    // Assuming a file exists or just demonstrating the API
    println!("WeightsContext created.");

    // Note: In a real scenario, we'd call ctx.load_file("path/to/model.safetensors")
    println!("API Usage: ctx.load_file(path), then ctx.read_bytes(name, callback)");
}
