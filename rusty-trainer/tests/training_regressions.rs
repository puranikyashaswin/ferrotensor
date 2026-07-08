//! Training-focused regression tests for masked loss and packing semantics.
//! These are CPU-only and do not require a GPU.

use rusty_trainer::{
    masked_cross_entropy_from_logits, DataLoader, Dataset, TextDataset, TrainingExample,
};

#[test]
fn sft_mask_ignores_prompt_tokens_in_host_loss() {
    let example = TrainingExample::from_sft_prompt_response(vec![1, 2, 3], vec![4, 5]);
    assert_eq!(example.loss_mask, vec![0, 0, 1, 1]);

    // Craft logits so the prompt targets would dominate unmasked loss.
    let vocab = 6usize;
    let seq = example.target_ids.len();
    let mut logits = vec![0.0f32; seq * vocab];
    for (pos, &tid) in example.target_ids.iter().enumerate() {
        // Make every true target strongly preferred.
        logits[pos * vocab + tid as usize] = 10.0;
    }
    // Poison a prompt position: if masking failed, flipping its target preference
    // would not matter because mask zeros it. Flip the preferred class away for pos0
    // and ensure masked loss still only counts response tokens.
    for v in 0..vocab {
        logits[v] = if v == 0 { 10.0 } else { 0.0 };
    }
    let (loss, count) = masked_cross_entropy_from_logits(
        &logits,
        vocab,
        &example.target_ids,
        &example.loss_mask,
    )
    .expect("masked loss");
    assert_eq!(count, 2);
    assert!(loss < 0.05, "response-only loss should be near zero, got {loss}");
}

#[test]
fn pretrain_packing_masks_all_targets() {
    let ds = TextDataset::from_packed_tokens((0..16).collect(), 8);
    assert!(!ds.is_empty());
    for i in 0..ds.len() {
        let mask = ds.get_loss_mask(i).unwrap();
        assert!(mask.iter().all(|&m| m == 1));
    }
}

#[test]
fn dataloader_pads_sft_masks_without_training_on_pad() {
    let mut ds = TextDataset::new();
    ds.add_example(TrainingExample::from_sft_prompt_response(vec![1, 2], vec![3, 4, 5]));
    ds.add_example(TrainingExample::from_sft_prompt_response(vec![9], vec![8]));
    let mut loader = DataLoader::new_with_pad_token(ds, 2, false, 0);
    let batch = loader.next().unwrap();
    for (attn, loss_mask) in batch.attention_mask.iter().zip(&batch.loss_masks) {
        for (a, m) in attn.iter().zip(loss_mask.iter()) {
            if *a == 0 {
                assert_eq!(*m, 0, "padded positions must not contribute to loss");
            }
        }
    }
}

#[test]
fn grad_accum_token_mean_matches_full_batch() {
    let logits = vec![
        4.0, 0.0, // t0 -> 0
        0.0, 5.0, // t1 -> 1
        3.0, 0.0, // t2 -> 0
        0.0, 2.0, // t3 -> 1
    ];
    let targets = [0u32, 1, 0, 1];
    let mask = [1u8, 1, 1, 1];
    let (full, _) = masked_cross_entropy_from_logits(&logits, 2, &targets, &mask).unwrap();

    let (a, _) = masked_cross_entropy_from_logits(&logits[..4], 2, &targets[..2], &mask[..2]).unwrap();
    let (b, _) = masked_cross_entropy_from_logits(&logits[4..], 2, &targets[2..], &mask[2..]).unwrap();
    assert!(((a + b) * 0.5 - full).abs() < 1e-5);
}
