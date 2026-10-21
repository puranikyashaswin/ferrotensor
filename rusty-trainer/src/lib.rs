//! Training loop for Rusty language models.
//!
//! [`Trainer`] runs forward, a response-masked cross-entropy loss
//! ([`masked_cross_entropy`]), backward and an optimizer step on an
//! [`Lfm2Model`] (Llama-style or LFM2), with gradient accumulation, global
//! gradient-norm clipping, an optional learning-rate schedule, a loss scaler
//! and file-based multi-process gradient averaging. [`data`] builds SFT and
//! pretraining datasets, [`checkpoint`] saves and restores LoRA adapters and
//! optimizer state.

#![warn(missing_docs)]

use rusty_autograd::{AutogradNode, LRScheduler, Optimizer, Tensor};
use rusty_backend::{ComputeEngine, UnifiedTensor, WgpuContext};
use rusty_graph::lfm::Lfm2Model;
use std::sync::Arc;

pub mod checkpoint;
pub mod data;
pub mod demo;
pub mod distributed;
pub mod scaler;

pub use data::{build_sft_example, Batch, DataLoader, Dataset, TextDataset, TrainingExample};
pub use demo::{print_banner, run_demo, DemoConfig};
pub use scaler::{GradScaler, GradScalerStats};

pub use distributed::FileGradSyncConfig;

/// Result of one [`Trainer::train_step_masked`] call.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepStats {
    /// Mean cross-entropy over the supervised tokens of this sequence.
    pub loss: f32,
    /// Number of supervised (unmasked) target tokens.
    pub tokens: usize,
    /// Whether this call completed an accumulation window and updated the
    /// parameters (`false` while accumulating or after an fp16 overflow).
    pub optimizer_step: bool,
    /// Global L2 norm of the gradients before clipping, when the optimizer
    /// stepped.
    pub grad_norm: Option<f32>,
    /// Learning rate used for the optimizer step.
    pub lr: Option<f32>,
}

/// Error from a training step.
#[derive(Debug)]
pub struct TrainError(pub String);

impl std::fmt::Display for TrainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for TrainError {}

/// Runs optimization steps on a model.
pub struct Trainer {
    /// The model being trained.
    pub model: Lfm2Model,
    /// Optimizer applied to the trainable parameters.
    pub optimizer: Box<dyn Optimizer>,
    /// GPU context.
    pub ctx: Arc<WgpuContext>,
    /// Kernel engine.
    pub engine: Arc<ComputeEngine>,
    /// Clip the global gradient norm to this value before each step.
    pub clip_grad_norm: Option<f32>,
    /// Micro-batches (sequences) per optimizer step.
    pub grad_accum_steps: usize,
    /// Micro-batches accumulated so far in the current window.
    pub step_in_accum: usize,
    /// Dynamic loss scaler (fp16-style overflow protection).
    pub scaler: Option<GradScaler>,
    /// Multi-process gradient averaging through a shared directory.
    pub distributed_grad_sync: Option<FileGradSyncConfig>,
    /// Learning-rate schedule, advanced once per optimizer step (before it
    /// is read, so step `k` uses the schedule's rate at `k`).
    pub lr_scheduler: Option<Box<dyn LRScheduler>>,
    /// Sequences are right-padded to a multiple of this many tokens (with
    /// the padding excluded from the loss). Causal attention keeps real
    /// tokens from seeing the padding, so results are unchanged, but GPU
    /// buffer sizes repeat and can be reused. 1 disables padding.
    pub length_bucket: usize,
    optimizer_step_count: usize,
}

impl Trainer {
    /// A trainer with no clipping, accumulation, scaler or schedule.
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
            lr_scheduler: None,
            length_bucket: 32,
            optimizer_step_count: 0,
        }
    }

    /// Pad sequences to a multiple of `tokens` (see
    /// [`length_bucket`](Self::length_bucket)).
    pub fn with_length_bucket(mut self, tokens: usize) -> Self {
        self.length_bucket = tokens.max(1);
        self
    }

    /// Clip the global gradient L2 norm to `max_norm`.
    pub fn with_clip_grad_norm(mut self, max_norm: f32) -> Self {
        self.clip_grad_norm = Some(max_norm);
        self
    }

    /// Accumulate gradients over `steps` sequences per optimizer step. Each
    /// sequence's loss is divided by `steps`, so the update matches the mean
    /// over the window.
    pub fn with_grad_accum_steps(mut self, steps: usize) -> Self {
        self.grad_accum_steps = steps.max(1);
        self
    }

    /// Scale the loss before backward and skip steps whose gradients overflow.
    pub fn with_grad_scaler(mut self, scaler: GradScaler) -> Self {
        self.scaler = Some(scaler);
        self
    }

    /// Average gradients across processes before each optimizer step.
    pub fn with_file_grad_sync(mut self, cfg: FileGradSyncConfig) -> Self {
        self.distributed_grad_sync = Some(cfg);
        self
    }

    /// Set the learning rate from `scheduler` before each optimizer step.
    pub fn with_lr_scheduler(mut self, scheduler: Box<dyn LRScheduler>) -> Self {
        self.lr_scheduler = Some(scheduler);
        self
    }

    /// Optimizer steps taken so far.
    pub fn optimizer_steps(&self) -> usize {
        self.optimizer_step_count
    }

    /// Restore the optimizer-step counter (and advance the LR schedule to
    /// match) when resuming from a checkpoint.
    pub fn set_optimizer_steps(&mut self, steps: usize) {
        if let Some(s) = self.lr_scheduler.as_mut() {
            while s.current_step() < steps {
                s.step();
            }
        }
        self.optimizer_step_count = steps;
    }

    /// Train on one sequence with every target supervised.
    pub fn train_step(
        &mut self,
        input_ids: &[u32],
        target_ids: &[u32],
        train_params: &[Tensor],
    ) -> Result<StepStats, TrainError> {
        self.train_step_masked(input_ids, target_ids, None, train_params)
    }

    /// Train on one sequence, optionally ignoring target positions via `loss_mask`.
    /// `loss_mask` is aligned with `target_ids` (`1` = include, `0` = ignore).
    ///
    /// Gradients accumulate into `train_params`; every `grad_accum_steps`
    /// calls they are (optionally) synced, clipped and applied, then cleared.
    pub fn train_step_masked(
        &mut self,
        input_ids: &[u32],
        target_ids: &[u32],
        loss_mask: Option<&[u8]>,
        train_params: &[Tensor],
    ) -> Result<StepStats, TrainError> {
        let len = input_ids.len().min(target_ids.len());
        let padded_len = len.div_ceil(self.length_bucket) * self.length_bucket;
        let mut inputs = input_ids[..len].to_vec();
        let mut targets = target_ids[..len].to_vec();
        let mut mask = match loss_mask {
            Some(mask) => (0..len)
                .map(|i| mask.get(i).copied().unwrap_or(0))
                .collect(),
            None => vec![1u8; len],
        };
        let pad = inputs.last().copied().unwrap_or(0);
        inputs.resize(padded_len, pad);
        targets.resize(padded_len, pad);
        mask.resize(padded_len, 0);

        let logits = self.model.forward(&self.engine, &inputs, 0, None, None);

        let mut stats = StepStats {
            loss: 0.0,
            tokens: 0,
            optimizer_step: false,
            grad_norm: None,
            lr: None,
        };
        if let Some(loss) = masked_cross_entropy(&logits, &targets, &mask) {
            stats.loss = loss.value;
            stats.tokens = loss.tokens;
            // Seed backward with d(total)/d(loss): the accumulation average
            // times the loss scale.
            let mut seed = 1.0 / self.grad_accum_steps as f32;
            if let Some(scaler) = self.scaler.as_ref() {
                seed *= scaler.scale_factor();
            }
            *loss.tensor.grad.borrow_mut() = Some(UnifiedTensor::new(&self.ctx, &[seed], &[1]));
            loss.tensor.backward(&self.engine);
        }

        self.step_in_accum += 1;
        if self.step_in_accum >= self.grad_accum_steps {
            self.step_in_accum = 0;
            let applied = self.apply_gradients(train_params, &mut stats);
            self.optimizer.zero_grad(train_params);
            applied?;
        }
        Ok(stats)
    }

    /// Sync, unscale, clip and apply the accumulated gradients.
    fn apply_gradients(
        &mut self,
        train_params: &[Tensor],
        stats: &mut StepStats,
    ) -> Result<(), TrainError> {
        let mut grads: Vec<UnifiedTensor> = train_params
            .iter()
            .filter_map(|p| p.grad.borrow().clone())
            .collect();
        if grads.is_empty() {
            return Ok(());
        }

        if let Some(scaler) = self.scaler.as_mut() {
            let valid = scaler.unscale_and_check(&grads, &self.ctx, &self.engine);
            scaler.update();
            if !valid {
                log::warn!("non-finite gradients; skipping optimizer step");
                return Ok(());
            }
        }

        if let Some(cfg) = self.distributed_grad_sync.as_ref() {
            if cfg.world_size > 1 {
                crate::distributed::average_grads_file_sync(
                    &mut grads,
                    &self.ctx,
                    cfg,
                    self.optimizer_step_count,
                )
                .map_err(|e| TrainError(format!("distributed grad sync failed: {e}")))?;
            }
        }

        let norm = global_grad_norm(&grads, &self.ctx);
        stats.grad_norm = Some(norm);
        if !norm.is_finite() {
            log::warn!("gradient norm is {norm}; skipping optimizer step");
            return Ok(());
        }
        if let Some(max_norm) = self.clip_grad_norm {
            if norm > max_norm {
                let scale = max_norm / (norm + 1e-6);
                for grad in &grads {
                    self.engine.clip_grad_by_norm(&self.ctx, grad, scale);
                }
            }
        }

        // Step k (1-based) uses the schedule's value at k, so warmup starts
        // above zero and the last step lands on the final rate.
        if let Some(s) = self.lr_scheduler.as_mut() {
            s.step();
            let lr = s.get_lr();
            self.optimizer.set_lr(lr);
            stats.lr = Some(lr);
        }
        self.optimizer.step(train_params);
        self.optimizer_step_count += 1;
        stats.optimizer_step = true;
        Ok(())
    }

    /// Train on every sequence of `batch` (one micro-batch per sequence,
    /// padding stripped). Sequences without supervised tokens are skipped.
    pub fn train_batch(
        &mut self,
        batch: &Batch,
        train_params: &[Tensor],
    ) -> Result<Vec<StepStats>, TrainError> {
        let mut out = Vec::with_capacity(batch.batch_size);
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
            out.push(self.train_step_masked(
                input_ids,
                target_ids,
                Some(loss_mask),
                train_params,
            )?);
        }
        Ok(out)
    }
}

/// L2 norm over all gradient tensors (host reduction).
fn global_grad_norm(grads: &[UnifiedTensor], ctx: &WgpuContext) -> f32 {
    let sum_sq: f64 = grads
        .iter()
        .map(|g| {
            pollster::block_on(g.to_vec(ctx))
                .iter()
                .map(|&x| (x as f64) * (x as f64))
                .sum::<f64>()
        })
        .sum();
    sum_sq.sqrt() as f32
}

/// A differentiable masked cross-entropy loss, see [`masked_cross_entropy`].
pub struct MaskedLoss {
    /// Scalar loss tensor `[1]`; call `backward` on it.
    pub tensor: Tensor,
    /// The loss value (mean over supervised tokens).
    pub value: f32,
    /// Number of supervised tokens.
    pub tokens: usize,
}

/// Backward node of [`masked_cross_entropy`]: holds `d loss / d logits`,
/// computed together with the loss from one host read of the logits.
struct MaskedCrossEntropyNode {
    logits: Tensor,
    grad_logits: Vec<f32>,
}

impl AutogradNode for MaskedCrossEntropyNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.logits.clone()]
    }

    fn backward(&self, grad: &UnifiedTensor, _engine: &ComputeEngine) -> Vec<UnifiedTensor> {
        let ctx = &self.logits.ctx;
        let upstream = pollster::block_on(grad.to_vec(ctx))
            .first()
            .copied()
            .unwrap_or(1.0);
        let scaled: Vec<f32> = self.grad_logits.iter().map(|g| g * upstream).collect();
        vec![UnifiedTensor::new(ctx, &scaled, &self.logits.data.shape)]
    }
}

/// Mean cross-entropy of `logits` (`[.., seq, vocab]`) against `target_ids`
/// over the positions where `loss_mask` is 1, as a differentiable scalar.
///
/// The logits are read to the host once; the loss and its gradient
/// (`(softmax - onehot) / tokens` at supervised positions, zero elsewhere)
/// are computed there. Returns `None` when no position is supervised.
// Autograd nodes are shared through `Arc` but used on one thread.
#[allow(clippy::arc_with_non_send_sync)]
pub fn masked_cross_entropy(
    logits: &Tensor,
    target_ids: &[u32],
    loss_mask: &[u8],
) -> Option<MaskedLoss> {
    let vocab = *logits.data.shape.last()?;
    if vocab == 0 {
        return None;
    }
    let host = pollster::block_on(logits.data.to_vec(&logits.ctx));
    let seq = host.len() / vocab;
    let positions = seq.min(target_ids.len()).min(loss_mask.len());
    let supervised: Vec<usize> = (0..positions)
        .filter(|&p| loss_mask[p] != 0 && (target_ids[p] as usize) < vocab)
        .collect();
    if supervised.is_empty() {
        return None;
    }
    let inv = 1.0 / supervised.len() as f32;
    let mut grad = vec![0.0f32; host.len()];
    let mut loss_sum = 0.0f64;
    for &p in &supervised {
        let row = &host[p * vocab..(p + 1) * vocab];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = row.iter().map(|x| (x - max).exp()).sum();
        let log_z = max + sum_exp.ln();
        let target = target_ids[p] as usize;
        loss_sum += (log_z - row[target]) as f64;
        let g = &mut grad[p * vocab..(p + 1) * vocab];
        for (gi, &x) in g.iter_mut().zip(row) {
            *gi = (x - log_z).exp() * inv;
        }
        g[target] -= inv;
    }
    let value = (loss_sum / supervised.len() as f64) as f32;
    let ctx = logits.ctx.clone();
    let node = Arc::new(MaskedCrossEntropyNode {
        logits: logits.clone(),
        grad_logits: grad,
    });
    let tensor = Tensor::with_creator(ctx.clone(), UnifiedTensor::new(&ctx, &[value], &[1]), node);
    Some(MaskedLoss {
        tensor,
        value,
        tokens: supervised.len(),
    })
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
    use super::masked_cross_entropy_from_logits;
    use crate::data::TrainingExample;

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
        assert!(example.loss_mask.contains(&0));
        assert!(example.loss_mask.contains(&1));
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
            masked_cross_entropy_from_logits(&logits, 2, &targets, &full_mask).expect("loss");
        assert_eq!(full_count, 4);

        // Simulate grad-accum of 2 microbatches of 2 tokens each.
        let micro_a =
            masked_cross_entropy_from_logits(&logits[..4], 2, &targets[..2], &full_mask[..2])
                .expect("loss");
        let micro_b =
            masked_cross_entropy_from_logits(&logits[4..], 2, &targets[2..], &full_mask[2..])
                .expect("loss");
        let accum_mean = (micro_a.0 + micro_b.0) / 2.0;
        assert!((accum_mean - full_loss).abs() < 1e-5);
    }

    #[test]
    fn batch_size_invariance_of_token_mean_loss() {
        // Same two token predictions evaluated as batch=1 twice vs batch=2 once.
        let logits_a = vec![3.0, 0.0];
        let logits_b = vec![0.0, 4.0];
        let (loss_a, _) = masked_cross_entropy_from_logits(&logits_a, 2, &[0], &[1]).expect("loss");
        let (loss_b, _) = masked_cross_entropy_from_logits(&logits_b, 2, &[1], &[1]).expect("loss");
        let mean_bs1 = (loss_a + loss_b) / 2.0;

        let logits_bs2 = [logits_a, logits_b].concat();
        let (mean_bs2, count) =
            masked_cross_entropy_from_logits(&logits_bs2, 2, &[0, 1], &[1, 1]).expect("loss");
        assert_eq!(count, 2);
        assert!((mean_bs1 - mean_bs2).abs() < 1e-5);
    }

    #[test]
    fn pretrain_full_mask_equals_unmasked_host_loss() {
        let logits = vec![1.0, 0.0, 0.0, 2.0];
        let targets = [0u32, 1u32];
        let mask = [1u8, 1u8];
        let (masked, _) =
            masked_cross_entropy_from_logits(&logits, 2, &targets, &mask).expect("loss");
        // Same as ignoring nothing.
        let (unmasked, _) =
            masked_cross_entropy_from_logits(&logits, 2, &targets, &[1, 1]).expect("loss");
        assert!((masked - unmasked).abs() < 1e-6);
    }
}
