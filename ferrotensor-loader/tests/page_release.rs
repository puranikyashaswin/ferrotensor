//! Tensors read after their pages were released come back unchanged.
//!
//! `WeightsContext::read_bytes` releases a tensor's whole pages from the
//! process once it has been read. The tensors here span several 64 KiB
//! blocks, so every read releases pages; reading them again, in any order,
//! must fault the same bytes back in from the file.

use ferrotensor_loader::WeightsContext;
use safetensors::Dtype;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!(
            "ferrotensor-loader-{tag}-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        Self(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Distinct, deterministic bytes for each tensor.
fn pattern(len: usize, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u32).wrapping_mul(2_654_435_761).to_le_bytes()[seed as usize % 4] ^ seed)
        .collect()
}

/// `(name, dtype, shape, bytes)`: sizes chosen so the tensors start and end
/// at unaligned offsets and each spans several 64 KiB blocks, plus a tiny
/// tensor between them that shares its pages with both neighbours.
fn tensors() -> Vec<(&'static str, Dtype, Vec<usize>, Vec<u8>)> {
    vec![
        ("a", Dtype::F32, vec![301, 257], pattern(301 * 257 * 4, 1)),
        ("b", Dtype::F32, vec![3], pattern(12, 2)),
        ("c", Dtype::BF16, vec![513, 255], pattern(513 * 255 * 2, 3)),
        ("d", Dtype::F16, vec![70_001], pattern(70_001 * 2, 4)),
    ]
}

fn write(path: &Path) {
    let data = tensors();
    let views: HashMap<String, safetensors::tensor::TensorView<'_>> = data
        .iter()
        .map(|(name, dtype, shape, bytes)| {
            let view = safetensors::tensor::TensorView::new(*dtype, shape.clone(), bytes)
                .expect("tensor view");
            (name.to_string(), view)
        })
        .collect();
    std::fs::write(
        path,
        safetensors::tensor::serialize(views, &None).expect("serialize"),
    )
    .expect("write");
}

fn read(weights: &WeightsContext, name: &str) -> Vec<u8> {
    let mut out = Vec::new();
    weights
        .read_bytes(name, |bytes, _| out = bytes.to_vec())
        .expect("read");
    out
}

#[test]
fn tensors_read_in_reverse_order_and_twice_are_unchanged() {
    let dir = TempDir::new("page-release");
    let path = dir.0.join("model.safetensors");
    write(&path);
    let mut weights = WeightsContext::new();
    weights.load_file(&path).expect("load");

    let expected: HashMap<&str, Vec<u8>> = tensors()
        .into_iter()
        .map(|(name, _, _, bytes)| (name, bytes))
        .collect();
    let order = ["a", "b", "c", "d", "d", "c", "b", "a", "c", "a", "d", "b"];
    for name in order {
        assert!(
            read(&weights, name) == expected[name],
            "tensor {name} changed after its pages were released"
        );
    }

    // The f32 conversion reads the same released ranges.
    let (values, shape) = weights.read_f32_tensor("c").expect("read c");
    assert_eq!(shape, vec![513, 255]);
    let want: Vec<f32> = expected["c"]
        .chunks_exact(2)
        .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect();
    let (values2, _) = weights.read_f32_tensor("c").expect("read c again");
    for got in [&values, &values2] {
        assert_eq!(got.len(), want.len());
        assert!(
            got.iter()
                .zip(&want)
                .all(|(x, y)| x.to_bits() == y.to_bits()),
            "bf16 values changed"
        );
    }
}
