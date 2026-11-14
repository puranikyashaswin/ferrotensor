//! Read tensors from a safetensors file with `WeightsContext`.
//!
//! Run with: `cargo run -p ferrotensor-loader --example read_tensor [path]`
//!
//! The path defaults to `dummy.safetensors`, which the `gen_dummy` example
//! writes. The example exits with an error when the tensors are not the ones
//! `gen_dummy` wrote.

use ferrotensor_loader::WeightsContext;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "dummy.safetensors".to_string());

    let mut ctx = WeightsContext::new();
    if let Err(e) = ctx.load_file(&path) {
        eprintln!("cannot load {path}: {e}");
        eprintln!("run `cargo run -p ferrotensor-loader --example gen_dummy` first");
        std::process::exit(1);
    }

    for (name, want) in [
        ("tensor_a", [1u8, 2, 3, 4]),
        ("tensor_b", [10u8, 20, 30, 40]),
    ] {
        let (shape, dtype) = ctx
            .get_tensor_info(name)
            .unwrap_or_else(|| panic!("{path} has no tensor {name}"));
        ctx.read_bytes(name, |bytes, _| {
            println!("{name}: shape {shape:?}, dtype {dtype}, bytes {bytes:?}");
            assert_eq!(shape, [4]);
            assert_eq!(dtype, "U8");
            assert_eq!(bytes, want);
        })
        .unwrap();
    }
}
