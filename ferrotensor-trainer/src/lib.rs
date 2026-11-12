//! Training loop for Ferrotensor language models.
//!
//! [`Trainer`] runs forward, a response-masked cross-entropy loss
//! ([`masked_cross_entropy`]), backward and an optimizer step on an
//! [`Lfm2Model`] (Llama-style or LFM2), with gradient accumulation, global
//! gradient-norm clipping, an optional learning-rate schedule, a loss scaler
//! and file-based multi-process gradient averaging. [`data`] builds SFT and
//! pretraining datasets, [`checkpoint`] saves and restores LoRA adapters and
//! optimizer state, [`peft`] converts adapters to and from the Hugging Face
//! PEFT format, and [`classify`] fine-tunes a decoder with LoRA as a sequence
//! classifier ([`ClassifierTrainer`]).

#![warn(missing_docs)]

use ferrotensor_autograd::{AutogradNode, LRScheduler, Optimizer, Tensor};
use ferrotensor_backend::{
    ComputeEngine, UnifiedTensor, WgpuContext, MASKED_CE_SKIP, SUM_SQUARES_CHUNK,
};
use ferrotensor_graph::lfm::Lfm2Model;
use std::sync::Arc;

pub mod checkpoint;
pub mod classify;
pub mod data;
pub mod demo;
pub mod distributed;
pub mod peft;
pub mod scaler;

pub use classify::{ClassifierStepStats, ClassifierTrainer};
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
    /// Run seed for attention dropout. Each forward pass draws its masks
    /// from this seed, the rank, the optimizer step and the position in the
    /// accumulation window (see
    /// [`attention_dropout_seed`](Self::attention_dropout_seed)), so a run is
    /// reproducible and a resumed run (with
    /// [`set_optimizer_steps`](Self::set_optimizer_steps)) draws the same
    /// masks as an uninterrupted one. Has no effect when the model's
    /// attention dropout is 0.
    pub dropout_seed: u64,
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
            dropout_seed: 0,
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

    /// Seed the attention-dropout masks with `seed` (see
    /// [`dropout_seed`](Self::dropout_seed)).
    pub fn with_dropout_seed(mut self, seed: u64) -> Self {
        self.dropout_seed = seed;
        self
    }

    /// The attention-dropout step seed for the next forward pass:
    /// [`derive_seed`](ferrotensor_graph::dropout::derive_seed) of the run
    /// seed, this process's rank, the optimizer step and the position in the
    /// accumulation window. Layers derive their own seeds from it.
    pub fn attention_dropout_seed(&self) -> u64 {
        let rank = self.distributed_grad_sync.as_ref().map_or(0, |c| c.rank);
        ferrotensor_graph::dropout::derive_seed(&[
            self.dropout_seed,
            rank as u64,
            self.optimizer_step_count as u64,
            self.step_in_accum as u64,
        ])
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

        let dropout_seed = Some(self.attention_dropout_seed());
        let logits =
            self.model
                .forward_with_dropout(&self.engine, &inputs, 0, None, None, dropout_seed);

        let mut stats = StepStats {
            loss: 0.0,
            tokens: 0,
            optimizer_step: false,
            grad_norm: None,
            lr: None,
        };
        if let Some(loss) = masked_cross_entropy(&logits, &targets, &mask, &self.engine) {
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

        let norm = global_grad_norm(&grads, &self.ctx, &self.engine);
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

    /// Train on every sequence of `batch`. Sequences without supervised
    /// tokens are skipped. Returns one [`StepStats`] per trained sequence.
    ///
    /// With an all-attention model ([`Lfm2Model::supports_batching`]) the
    /// sequences run through the model together, one forward and backward
    /// per accumulation chunk; otherwise one at a time. Both give the same
    /// result: each sequence contributes its own mean-loss gradient, and
    /// optimizer steps fall after the same sequences. With attention dropout
    /// on, the two paths draw different (identically distributed) masks.
    pub fn train_batch(
        &mut self,
        batch: &Batch,
        train_params: &[Tensor],
    ) -> Result<Vec<StepStats>, TrainError> {
        let mut seqs: Vec<(&[u32], &[u32], &[u8])> = Vec::with_capacity(batch.batch_size);
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
            seqs.push((input_ids, target_ids, loss_mask));
        }

        let mut out = Vec::with_capacity(seqs.len());
        if !self.model.supports_batching() || seqs.len() < 2 {
            for (input_ids, target_ids, loss_mask) in seqs {
                out.push(self.train_step_masked(
                    input_ids,
                    target_ids,
                    Some(loss_mask),
                    train_params,
                )?);
            }
            return Ok(out);
        }
        let mut rest = seqs.as_slice();
        while !rest.is_empty() {
            // Stop each chunk at the accumulation boundary, so the optimizer
            // steps after the same sequence as in the one-at-a-time path.
            let room = self
                .grad_accum_steps
                .saturating_sub(self.step_in_accum)
                .max(1);
            let (chunk, tail) = rest.split_at(room.min(rest.len()));
            out.extend(self.train_chunk(chunk, train_params)?);
            rest = tail;
        }
        Ok(out)
    }

    /// One forward and backward over `chunk`, padded on the right to a
    /// common length; then an optimizer step if accumulation is complete.
    fn train_chunk(
        &mut self,
        chunk: &[(&[u32], &[u32], &[u8])],
        train_params: &[Tensor],
    ) -> Result<Vec<StepStats>, TrainError> {
        let longest = chunk
            .iter()
            .map(|s| s.0.len().min(s.1.len()))
            .max()
            .unwrap_or(0);
        let padded_len = longest.div_ceil(self.length_bucket) * self.length_bucket;
        let mut inputs = Vec::with_capacity(chunk.len());
        let mut targets = Vec::with_capacity(chunk.len() * padded_len);
        let mut masks = Vec::with_capacity(chunk.len() * padded_len);
        for &(input_ids, target_ids, loss_mask) in chunk {
            let len = input_ids.len().min(target_ids.len());
            let pad = input_ids[..len].last().copied().unwrap_or(0);
            let mut row = input_ids[..len].to_vec();
            row.resize(padded_len, pad);
            inputs.push(row);
            targets.extend_from_slice(&target_ids[..len]);
            targets.resize(targets.len() + padded_len - len, pad);
            masks.extend((0..len).map(|i| loss_mask.get(i).copied().unwrap_or(0)));
            masks.resize(masks.len() + padded_len - len, 0);
        }
        let rows: Vec<&[u32]> = inputs.iter().map(Vec::as_slice).collect();
        let dropout_seed = Some(self.attention_dropout_seed());
        let logits = self
            .model
            .forward_batch_with_dropout(&self.engine, &rows, dropout_seed);

        let mut out: Vec<StepStats> = Vec::with_capacity(chunk.len());
        if let Some(loss) =
            masked_cross_entropy_batch(&logits, &targets, &masks, chunk.len(), &self.engine)
        {
            for &(value, tokens) in &loss.per_sequence {
                out.push(StepStats {
                    loss: value,
                    tokens,
                    optimizer_step: false,
                    grad_norm: None,
                    lr: None,
                });
            }
            let mut seed = 1.0 / self.grad_accum_steps as f32;
            if let Some(scaler) = self.scaler.as_ref() {
                seed *= scaler.scale_factor();
            }
            *loss.tensor.grad.borrow_mut() = Some(UnifiedTensor::new(&self.ctx, &[seed], &[1]));
            loss.tensor.backward(&self.engine);
        }

        self.step_in_accum += chunk.len();
        if self.step_in_accum >= self.grad_accum_steps {
            self.step_in_accum = 0;
            let mut stats = out.pop().unwrap_or(StepStats {
                loss: 0.0,
                tokens: 0,
                optimizer_step: false,
                grad_norm: None,
                lr: None,
            });
            let applied = self.apply_gradients(train_params, &mut stats);
            self.optimizer.zero_grad(train_params);
            applied?;
            out.push(stats);
        }
        Ok(out)
    }
}

/// L2 norm over all gradient tensors: partial sums of squares on the GPU,
/// read back once and added up on the host in f64.
pub(crate) fn global_grad_norm(
    grads: &[UnifiedTensor],
    ctx: &WgpuContext,
    engine: &ComputeEngine,
) -> f32 {
    let chunks: usize = grads
        .iter()
        .map(|g| {
            g.shape
                .iter()
                .product::<usize>()
                .div_ceil(SUM_SQUARES_CHUNK)
        })
        .sum();
    if chunks == 0 {
        return 0.0;
    }
    let partials = UnifiedTensor::empty(ctx, &[chunks]);
    let mut offset = 0;
    for grad in grads {
        offset += engine.sum_squares_partial(ctx, grad, &partials, offset);
    }
    let sum_sq: f64 = pollster::block_on(partials.to_vec(ctx))
        .iter()
        .map(|&x| x as f64)
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
/// computed on the GPU together with the loss.
struct MaskedCrossEntropyNode {
    logits: Tensor,
    grad_logits: UnifiedTensor,
}

impl AutogradNode for MaskedCrossEntropyNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.logits.clone()]
    }

    fn backward(&self, grad: &UnifiedTensor, engine: &ComputeEngine) -> Vec<UnifiedTensor> {
        let ctx = &self.logits.ctx;
        let upstream = pollster::block_on(grad.to_vec(ctx))
            .first()
            .copied()
            .unwrap_or(1.0);
        let scaled = UnifiedTensor::empty(ctx, &self.grad_logits.shape);
        engine.copy_tensor(ctx, &self.grad_logits, &scaled, 0);
        if (upstream - 1.0).abs() > f32::EPSILON {
            engine.gradient_scale(ctx, &scaled, upstream);
        }
        vec![scaled]
    }
}

/// Mean cross-entropy of `logits` (`[.., seq, vocab]`) against `target_ids`
/// over the positions where `loss_mask` is 1, as a differentiable scalar.
///
/// The loss and its gradient (`(softmax - onehot) / tokens` at supervised
/// positions, zero elsewhere) are computed on the GPU; only the per-position
/// losses are read back. Returns `None` when no position is supervised.
// Autograd nodes are shared through `Arc` but used on one thread.
#[allow(clippy::arc_with_non_send_sync)]
pub fn masked_cross_entropy(
    logits: &Tensor,
    target_ids: &[u32],
    loss_mask: &[u8],
    engine: &ComputeEngine,
) -> Option<MaskedLoss> {
    let vocab = *logits.data.shape.last()?;
    if vocab == 0 {
        return None;
    }
    let rows = logits.data.shape.iter().product::<usize>() / vocab;
    let positions = rows.min(target_ids.len()).min(loss_mask.len());
    let targets: Vec<u32> = (0..rows)
        .map(|p| {
            let supervised = p < positions && loss_mask[p] != 0 && (target_ids[p] as usize) < vocab;
            if supervised {
                target_ids[p]
            } else {
                MASKED_CE_SKIP
            }
        })
        .collect();
    let tokens = targets.iter().filter(|&&t| t != MASKED_CE_SKIP).count();
    if tokens == 0 {
        return None;
    }
    let ctx = logits.ctx.clone();
    let target_buf = UnifiedTensor::from_bytes(&ctx, bytemuck::cast_slice(&targets), &[rows]);
    let grad = UnifiedTensor::empty(&ctx, &logits.data.shape);
    let row_loss = UnifiedTensor::empty(&ctx, &[rows]);
    engine.masked_cross_entropy(
        &ctx,
        &logits.data,
        &target_buf,
        1.0 / tokens as f32,
        &grad,
        &row_loss,
    );
    let loss_sum: f64 = pollster::block_on(row_loss.to_vec(&ctx))
        .iter()
        .map(|&l| l as f64)
        .sum();
    let value = (loss_sum / tokens as f64) as f32;
    let node = Arc::new(MaskedCrossEntropyNode {
        logits: logits.clone(),
        grad_logits: grad,
    });
    let tensor = Tensor::with_creator(ctx.clone(), UnifiedTensor::new(&ctx, &[value], &[1]), node);
    Some(MaskedLoss {
        tensor,
        value,
        tokens,
    })
}

/// Mean cross-entropy of class `logits` (`[batch, classes]`, e.g. from
/// `ferrotensor_graph::ClassifierHead`) against one label per row, as a
/// differentiable scalar computed on the GPU.
///
/// Returns `None` when the number of labels differs from the batch size or
/// a label is not below `classes`, so a bad label cannot be skipped silently.
pub fn classification_loss(
    logits: &Tensor,
    labels: &[u32],
    engine: &ComputeEngine,
) -> Option<MaskedLoss> {
    let (batch, classes) = match logits.data.shape.as_slice() {
        [batch, classes] => (*batch, *classes),
        _ => return None,
    };
    if labels.len() != batch || labels.iter().any(|&l| l as usize >= classes) {
        return None;
    }
    masked_cross_entropy(logits, labels, &vec![1u8; batch], engine)
}

/// The loss of a batch of sequences, see [`masked_cross_entropy_batch`].
pub struct BatchLoss {
    /// Sum of the per-sequence mean losses `[1]`; call `backward` on it.
    pub tensor: Tensor,
    /// `(mean loss, supervised tokens)` of each sequence, in order.
    pub per_sequence: Vec<(f32, usize)>,
}

/// Cross-entropy of `logits` (`[sequences, seq, vocab]`, flattened rows)
/// against `target_ids` where `loss_mask` is 1, for a batch of sequences.
///
/// Each sequence is averaged over its own supervised tokens and the result
/// is the sum of those means, so its gradient equals the sum of
/// [`masked_cross_entropy`] gradients of the sequences taken one at a time.
/// `target_ids` and `loss_mask` hold one entry per row. Returns `None` when
/// no position is supervised.
// Autograd nodes are shared through `Arc` but used on one thread.
#[allow(clippy::arc_with_non_send_sync)]
pub fn masked_cross_entropy_batch(
    logits: &Tensor,
    target_ids: &[u32],
    loss_mask: &[u8],
    sequences: usize,
    engine: &ComputeEngine,
) -> Option<BatchLoss> {
    let vocab = *logits.data.shape.last()?;
    let rows = logits.data.shape.iter().product::<usize>() / vocab.max(1);
    if vocab == 0 || sequences == 0 || !rows.is_multiple_of(sequences) {
        return None;
    }
    let seq_len = rows / sequences;
    let supervised = |r: usize| {
        r < target_ids.len()
            && r < loss_mask.len()
            && loss_mask[r] != 0
            && (target_ids[r] as usize) < vocab
    };
    let tokens: Vec<usize> = (0..sequences)
        .map(|s| {
            (s * seq_len..(s + 1) * seq_len)
                .filter(|&r| supervised(r))
                .count()
        })
        .collect();
    if tokens.iter().all(|&t| t == 0) {
        return None;
    }
    let mut targets = Vec::with_capacity(rows);
    let mut weights = Vec::with_capacity(rows);
    for r in 0..rows {
        let t = tokens[r / seq_len];
        if supervised(r) {
            targets.push(target_ids[r]);
            weights.push(1.0 / t as f32);
        } else {
            targets.push(MASKED_CE_SKIP);
            weights.push(0.0);
        }
    }
    let ctx = logits.ctx.clone();
    let target_buf = UnifiedTensor::from_bytes(&ctx, bytemuck::cast_slice(&targets), &[rows]);
    let weight_buf = UnifiedTensor::new(&ctx, &weights, &[rows]);
    let grad = UnifiedTensor::empty(&ctx, &logits.data.shape);
    let row_loss = UnifiedTensor::empty(&ctx, &[rows]);
    engine.masked_cross_entropy_weighted(
        &ctx,
        &logits.data,
        &target_buf,
        &weight_buf,
        &grad,
        &row_loss,
    );
    let losses = pollster::block_on(row_loss.to_vec(&ctx));
    let per_sequence: Vec<(f32, usize)> = tokens
        .iter()
        .enumerate()
        .map(|(s, &t)| {
            let sum: f64 = losses[s * seq_len..(s + 1) * seq_len]
                .iter()
                .map(|&l| l as f64)
                .sum();
            ((sum / t.max(1) as f64) as f32, t)
        })
        .collect();
    let total: f32 = per_sequence.iter().map(|&(l, _)| l).sum();
    let node = Arc::new(MaskedCrossEntropyNode {
        logits: logits.clone(),
        grad_logits: grad,
    });
    let tensor = Tensor::with_creator(ctx.clone(), UnifiedTensor::new(&ctx, &[total], &[1]), node);
    Some(BatchLoss {
        tensor,
        per_sequence,
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
