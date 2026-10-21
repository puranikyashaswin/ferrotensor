use safetensors::serialize_to_file;
use safetensors::tensor::{Dtype, TensorView};
use std::collections::HashMap;

fn main() {
    let data_a: Vec<u8> = vec![1, 2, 3, 4]; // Int8 data
    let data_b: Vec<u8> = vec![10, 20, 30, 40]; // Int8 data

    let map: HashMap<String, TensorView> = [
        (
            "tensor_a".to_string(),
            TensorView::new(Dtype::U8, vec![4], &data_a).unwrap(),
        ),
        (
            "tensor_b".to_string(),
            TensorView::new(Dtype::U8, vec![4], &data_b).unwrap(),
        ),
    ]
    .into_iter()
    .collect();

    serialize_to_file(&map, &None, std::path::Path::new("dummy.safetensors")).unwrap();
    println!("Generated dummy.safetensors");
}
