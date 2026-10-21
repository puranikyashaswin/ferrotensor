use rusty_autograd::{Optimizer, Tensor};
use rusty_backend::{ComputeEngine, UnifiedTensor, WgpuContext};
use rusty_graph::lfm::Lfm2Model;
use std::sync::Arc;

pub mod checkpoint;
pub mod data;
pub mod demo;
pub mod distributed;
pub mod scaler;

pub use data::{Batch, DataLoader, Dataset, TextDataset, TrainingExample};
pub use demo::{print_banner, run_demo, DemoConfig};
pub use scaler::{GradScaler, GradScalerStats};

pub use distributed::FileGradSyncConfig;

pub struct Trainer {
    pub model: Lfm2Model,
    pub optimizer: Box<dyn Optimizer>,
    pub ctx: Arc<WgpuContext>,
    pub engine: Arc<ComputeEngine>,
    pub clip_grad_norm: Option<f32>,
    pub grad_accum_steps: usize,
    pub step_in_accum: usize,
    pub scaler: Option<GradScaler>,
    pub distributed_grad_sync: Option<FileGradSyncConfig>,
    optimizer_step_count: usize,
}

impl Trainer {
    pub fn new(
        model: Lfm2Model,
        optimizer: Box<dyn Optimizer>,
        ctx: Arc<WgpuContext>,
        engine: Arc<ComputeEngine>,
    ) -> Self {
        Self {
            model,
            optimizer,
            ctx,
            engine,
            clip_grad_norm: None,
            grad_accum_steps: 1,
            step_in_accum: 0,
            scaler: None,
            distributed_grad_sync: None,
            optimizer_step_count: 0,
        }
    }

    pub fn with_clip_grad_norm(mut self, max_norm: f32) -> Self {
        self.clip_grad_norm = Some(max_norm);
        self
    }

    pub fn with_grad_accum_steps(mut self, steps: usize) -> Self {
        self.grad_accum_steps = steps.max(1);
        self
    }

    pub fn with_grad_scaler(mut self, scaler: GradScaler) -> Self {
        self.scaler = Some(scaler);
        self
    }

    pub fn with_file_grad_sync(mut self, cfg: FileGradSyncConfig) -> Self {
        self.distributed_grad_sync = Some(cfg);
        self
    }

    pub fn train_step(
        &mut self,
        input_ids: &[u32],
        target_ids: &[u32],
        train_params: &[Tensor],
    ) -> Tensor {
        self.train_step_masked(input_ids, target_ids, None, train_params)
    }

    /// Train on one sequence, optionally ignoring target positions via `loss_mask`.
    /// `loss_mask` is aligned with `target_ids` (`1` = include, `0` = ignore).
    pub fn train_step_masked(
        &mut self,
        input_ids: &[u32],
        target_ids: &[u32],
        loss_mask: Option<&[u8]>,
        train_params: &[Tensor],
    ) -> Tensor {
        let logits = self.model.forward(&self.engine, input_ids, 0, None, None);

        let num_positions = valid_target_positions(input_ids.len(), target_ids.len());
        let mut included = 0usize;
        let mut loss: Option<Tensor> = None;

        for pos in 0..num_positions {
            let include = loss_mask
                .and_then(|mask| mask.get(pos).copied())
                .unwrap_or(1)
                != 0;
            if !include {
                continue;
            }
            let token_logits = Tensor::select_token(&logits, pos);
            let token_loss =
                Tensor::cross_entropy_loss(&token_logits, target_ids[pos], &self.engine);
            loss = Some(match loss {
                Some(acc) => Tensor::add(&acc, &token_loss, &self.engine),
                None => token_loss,
            });
            included += 1;
        }

        let mut loss = loss.unwrap_or_else(|| {
            Tensor::new(
                self.ctx.clone(),
                UnifiedTensor::new(&self.ctx, &[0.0], &[1]),
            )
        });

        if included > 1 {
            let scale = 1.0 / included as f32;
            let scale_tensor = Tensor::new(
                self.ctx.clone(),
                UnifiedTensor::new(&self.ctx, &[scale], &[1]),
            );
            loss = Tensor::mul(&loss, &scale_tensor, &self.engine);
        }

        let backward_loss = if let Some(scaler) = self.scaler.as_ref() {
            let scale_tensor = Tensor::new(
                self.ctx.clone(),
                UnifiedTensor::new(&self.ctx, &[scaler.scale_factor()], &[1]),
            );
            Tensor::mul(&loss, &scale_tensor, &self.engine)
        } else {
            loss.clone()
        };
        backward_loss.backward(&self.engine);

        self.step_in_accum += 1;
        if self.step_in_accum >= self.grad_accum_steps {
            let mut grads = Vec::new();
            for param in train_params {
                if let Some(grad) = param.grad.borrow().as_ref() {
                    grads.push(grad.clone());
                }
            }

            let grads_valid = if let Some(scaler) = self.scaler.as_mut() {
                let valid = scaler.unscale_and_check(&grads, &self.ctx, &self.engine);
                scaler.update();
                valid
            } else {
                true
            };

            if grads_valid {
                if let Some(cfg) = self.distributed_grad_sync.as_ref() {
                    // For distributed mode we average gradients across ranks
                    // before clipping and before optimizer.step.
                    //
                    // Note: step naming uses optimizer-step count (not batch count).
                    if cfg.world_size > 1 {
                        crate::distributed::average_grads_file_sync(
                            &mut grads,
                            &self.ctx,
                            cfg,
                            self.optimizer_step_count,
                        )
                        .expect("distributed grad sync failed");
                    }
                    self.optimizer_step_count += 1;
                }
                if let Some(max_norm) = self.clip_grad_norm {
                    for grad in &grads {
                        let grad_cpu = pollster::block_on(grad.to_vec(&self.ctx));
                        let grad_norm = grad_cpu.iter().map(|x| x * x).sum::<f32>().sqrt();
                        if grad_norm > max_norm && grad_norm > 0.0 {
                            self.engine
                                .clip_grad_by_norm(&self.ctx, grad, max_norm / grad_norm);
                        }
                    }
                }

                self.optimizer.step(train_params);
            }

            self.optimizer.zero_grad(train_params);
            self.step_in_accum = 0;
        }

        loss
    }

    pub fn train_batch(&mut self, batch: &Batch, train_params: &[Tensor]) -> Vec<Tensor> {
        let mut losses = Vec::with_capacity(batch.batch_size);
        for (((input_ids, target_ids), attn), loss_mask) in batch
            .padded_input_ids
            .iter()
            .zip(&batch.padded_target_ids)
            .zip(&batch.attention_mask)
            .zip(&batch.loss_masks)
        {
            let valid_len = attn.iter().take_while(|&&flag| flag == 1).count();
            let input_ids = &input_ids[..valid_len];
            let target_ids = &target_ids[..valid_len];
            let loss_mask = &loss_mask[..valid_len];
            if input_ids.len() < 2 || target_ids.is_empty() {
                continue;
            }
            if loss_mask.iter().all(|&m| m == 0) {
                continue;
            }
            losses.push(self.train_step_masked(
                input_ids,
                target_ids,
                Some(loss_mask),
                train_params,
            ));
        }
        losses
    }
}

fn valid_target_positions(input_len: usize, target_len: usize) -> usize {
    target_len.min(input_len)
}

/// Average cross-entropy over masked target positions using host logits.
pub fn masked_cross_entropy_from_logits(
    logits: &[f32],
    vocab_size: usize,
    target_ids: &[u32],
    loss_mask: &[u8],
) -> Option<(f32, usize)> {
    if vocab_size == 0 {
        return None;
    }
    let seq_len = logits.len() / vocab_size;
    let positions = seq_len.min(target_ids.len()).min(loss_mask.len());
    let mut loss_sum = 0.0f32;
    let mut count = 0usize;
    for pos in 0..positions {
        if loss_mask[pos] == 0 {
            continue;
        }
        let target = target_ids[pos] as usize;
        if target >= vocab_size {
            continue;
        }
        let start = pos * vocab_size;
        let end = start + vocab_size;
        if end > logits.len() {
            break;
        }
        let slice = &logits[start..end];
        let max_logit = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = slice.iter().map(|x| (x - max_logit).exp()).sum();
        let log_softmax = slice[target] - max_logit - exp_sum.ln();
        loss_sum += -log_softmax;
        count += 1;
    }
    if count == 0 {
        None
    } else {
        Some((loss_sum / count as f32, count))
    }
}

#[cfg(test)]
mod tests {
    use super::{masked_cross_entropy_from_logits, valid_target_positions};
    use crate::data::TrainingExample;

    #[test]
    fn valid_target_positions_uses_all_next_token_targets() {
        assert_eq!(valid_target_positions(8, 7), 7);
        assert_eq!(valid_target_positions(2, 1), 1);
    }

    #[test]
    fn valid_target_positions_caps_to_available_inputs() {
        assert_eq!(valid_target_positions(4, 9), 4);
        assert_eq!(valid_target_positions(0, 3), 0);
    }

    #[test]
    fn sft_mask_excludes_prompt_from_host_loss() {
        // tiny fake logits: two positions, vocab=2
        // pos0 targets token 1 with logit preference; will be masked out
        // pos1 targets token 0
        let logits = vec![
            0.0, 5.0, // pos0
            5.0, 0.0, // pos1
        ];
        let targets = [1u32, 0u32];
        let mask = [0u8, 1u8];
        let (loss, count) =
            masked_cross_entropy_from_logits(&logits, 2, &targets, &mask).expect("loss");
        assert_eq!(count, 1);
        assert!(loss < 0.1, "expected near-zero loss, got {loss}");
    }

    #[test]
    fn sft_training_example_mask_length_matches_targets() {
        let example = TrainingExample::from_sft_prompt_response(vec![1, 2, 3], vec![4, 5]);
        assert_eq!(example.loss_mask.len(), example.target_ids.len());
        assert!(example.loss_mask.iter().any(|&m| m == 0));
        assert!(example.loss_mask.iter().any(|&m| m == 1));
    }

    /// Averaging N micro-batches then summing should match one large batch
    /// mean when each micro-batch contributes the same token count.
    #[test]
    fn grad_accum_mean_matches_larger_batch_within_tolerance() {
        // Four positions, vocab=2. Token preference is always correct.
        let logits = vec![
            5.0, 0.0, // 0
            5.0, 0.0, // 1
            5.0, 0.0, // 2
            5.0, 0.0, // 3
        ];
        let targets = [0u32, 0, 0, 0];
        let full_mask = [1u8, 1, 1, 1];
        let (full_loss, full_count) =
            masked_cross_entropy_from_logits(&logits, 2, &targets, &full_mask).unwrap();
        assert_eq!(full_count, 4);

        // Simulate grad-accum of 2 microbatches of 2 tokens each.
        let micro_a =
            masked_cross_entropy_from_logits(&logits[..4], 2, &targets[..2], &full_mask[..2])
                .unwrap();
        let micro_b =
            masked_cross_entropy_from_logits(&logits[4..], 2, &targets[2..], &full_mask[2..])
                .unwrap();
        let accum_mean = (micro_a.0 + micro_b.0) / 2.0;
        assert!((accum_mean - full_loss).abs() < 1e-5);
    }

    #[test]
    fn batch_size_invariance_of_token_mean_loss() {
        // Same two token predictions evaluated as batch=1 twice vs batch=2 once.
        let logits_a = vec![3.0, 0.0];
        let logits_b = vec![0.0, 4.0];
        let (loss_a, _) = masked_cross_entropy_from_logits(&logits_a, 2, &[0], &[1]).unwrap();
        let (loss_b, _) = masked_cross_entropy_from_logits(&logits_b, 2, &[1], &[1]).unwrap();
        let mean_bs1 = (loss_a + loss_b) / 2.0;

        let logits_bs2 = [logits_a, logits_b].concat();
        let (mean_bs2, count) =
            masked_cross_entropy_from_logits(&logits_bs2, 2, &[0, 1], &[1, 1]).unwrap();
        assert_eq!(count, 2);
        assert!((mean_bs1 - mean_bs2).abs() < 1e-5);
    }

    #[test]
    fn pretrain_full_mask_equals_unmasked_host_loss() {
        let logits = vec![1.0, 0.0, 0.0, 2.0];
        let targets = [0u32, 1u32];
        let mask = [1u8, 1u8];
        let (masked, _) = masked_cross_entropy_from_logits(&logits, 2, &targets, &mask).unwrap();
        // Same as ignoring nothing.
        let (unmasked, _) =
            masked_cross_entropy_from_logits(&logits, 2, &targets, &[1, 1]).unwrap();
        assert!((masked - unmasked).abs() < 1e-6);
    }
}
