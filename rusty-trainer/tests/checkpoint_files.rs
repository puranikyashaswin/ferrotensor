//! Adapter and training-state files, without a GPU.

mod common;

use common::TempDir;
use rusty_trainer::checkpoint::{load_lora_checkpoint, TrainingCheckpoint, TrainingState};
use std::collections::HashMap;

fn write(
    path: &std::path::Path,
    tensors: &[(&str, safetensors::Dtype, Vec<usize>, Vec<u8>)],
    meta: Option<HashMap<String, String>>,
) {
    let views: HashMap<String, safetensors::tensor::TensorView<'_>> = tensors
        .iter()
        .map(|(n, d, s, b)| {
            (
                n.to_string(),
                safetensors::tensor::TensorView::new(*d, s.clone(), b).expect("view"),
            )
        })
        .collect();
    std::fs::write(
        path,
        safetensors::tensor::serialize(views, &meta).expect("serialize"),
    )
    .expect("write");
}

#[test]
fn adapter_metadata_and_dtypes_are_read() {
    let dir = TempDir::new("adapter");
    let path = dir.path().join("a.safetensors");
    let a: Vec<u8> = [0.5f32, -1.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let b: Vec<u8> = [2.0f32, 0.25]
        .iter()
        .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
        .collect();
    let meta = HashMap::from([
        ("format".to_string(), "rusty-lora".to_string()),
        ("rank".to_string(), "2".to_string()),
        ("alpha".to_string(), "4".to_string()),
        ("target_modules".to_string(), "q_proj,v_proj".to_string()),
    ]);
    write(
        &path,
        &[
            ("x.lora_A.weight", safetensors::Dtype::F32, vec![1, 2], a),
            ("x.lora_B.weight", safetensors::Dtype::BF16, vec![2, 1], b),
        ],
        Some(meta),
    );

    let ckpt = load_lora_checkpoint(&path).expect("load");
    assert_eq!(ckpt.len(), 2);
    assert_eq!(
        ckpt.tensors["x.lora_A.weight"],
        (vec![1, 2], vec![0.5, -1.0])
    );
    assert_eq!(
        ckpt.tensors["x.lora_B.weight"],
        (vec![2, 1], vec![2.0, 0.25])
    );
    let m = ckpt.metadata.expect("metadata");
    assert_eq!(m.rank, 2);
    assert_eq!(m.alpha, 4.0);
    assert_eq!(m.target_modules, vec!["q_proj", "v_proj"]);
}

#[test]
fn bad_adapter_files_are_errors() {
    let dir = TempDir::new("bad-adapter");
    let missing = load_lora_checkpoint(dir.path().join("nope.safetensors")).expect_err("missing");
    assert!(missing.to_string().contains("cannot read"), "{missing}");

    let garbage = dir.path().join("garbage.safetensors");
    std::fs::write(&garbage, b"\x05\x00\x00\x00\x00\x00\x00\x00{oops").expect("write");
    let err = load_lora_checkpoint(&garbage).expect_err("garbage");
    assert!(err.to_string().contains("not a safetensors file"), "{err}");

    let ints = dir.path().join("ints.safetensors");
    write(
        &ints,
        &[("t", safetensors::Dtype::I32, vec![1], vec![1, 0, 0, 0])],
        None,
    );
    let err = load_lora_checkpoint(&ints).expect_err("dtype");
    assert!(err.to_string().contains("unsupported dtype"), "{err}");
}

#[test]
fn training_checkpoint_round_trips_and_old_files_still_load() {
    let dir = TempDir::new("state");
    let path = dir.path().join("s.json");
    let mut state = TrainingState::new(100);
    state.step = 42;
    state.seed = 7;
    TrainingCheckpoint {
        training_state: state,
        optimizer_state: None,
        grad_scaler_state: None,
    }
    .save(&path)
    .expect("save");
    let loaded = TrainingCheckpoint::load(&path).expect("load");
    assert_eq!(loaded.training_state.step, 42);
    assert_eq!(loaded.training_state.seed, 7);

    // A state written before `seed` existed defaults it to 0.
    let old = r#"{"step":3,"total_steps":10,"best_loss":null,"lr":0.001,"epoch":0,
        "batch_size":1,"grad_accum_steps":1,"sequence_length":128,"fp16":false,
        "train_modules":"lora"}"#;
    let old_path = dir.path().join("old.json");
    std::fs::write(&old_path, old).expect("write");
    let s = TrainingState::load(&old_path).expect("old state");
    assert_eq!((s.step, s.seed, s.task.as_str()), (3, 0, "sft"));
}
