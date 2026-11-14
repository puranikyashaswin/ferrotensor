//! Fine-tuning a decoder as a sequence classifier with LoRA.
//!
//! [`ClassifierTrainer`] trains the LoRA adapters of an [`Lfm2Model`] and a
//! [`ClassifierHead`] on batches of token rows with one label each, with
//! gradient accumulation, global gradient-norm clipping and a shared
//! learning-rate schedule. [`predict_rows`] turns rows into class
//! probabilities. The head is saved as safetensors next to the adapter
//! ([`save_classifier_head`], [`load_classifier_head`]) and the settings
//! needed to rebuild the classifier go in a small JSON file
//! ([`ClassifierMetadata`]).
//!
//! The rest of the module is host-side data handling used by
//! `ferrotensor classify-train` and `classify-predict`: JSONL records
//! ([`parse_classify_jsonl`]), label indexing ([`label_names`],
//! [`label_ids`]), deterministic length-bucketed batch plans
//! ([`plan_epoch`], [`parse_batch_plan`]) and scores ([`score`]).

use crate::checkpoint::{load_lora_checkpoint, save_lora_checkpoint};
use crate::{classification_loss, global_grad_norm, TrainError};
use ferrotensor_autograd::{AdamW, LRScheduler, Tensor};
use ferrotensor_backend::{ComputeEngine, UnifiedTensor, WgpuContext};
use ferrotensor_graph::lfm::Lfm2Model;
use ferrotensor_graph::ClassifierHead;
use std::path::Path;
use std::sync::Arc;

/// AdamW weight decay of both optimizers of a [`ClassifierTrainer`].
pub const WEIGHT_DECAY: f32 = 0.01;

/// File name of the LoRA adapter in a classifier directory.
pub const ADAPTER_FILE: &str = "adapter.safetensors";
/// File name of the classifier head in a classifier directory.
pub const HEAD_FILE: &str = "head.safetensors";
/// File name of the [`ClassifierMetadata`] in a classifier directory.
pub const METADATA_FILE: &str = "classifier.json";

/// What happened at the end of an accumulation window.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OptimizerUpdate {
    /// Whether the parameters were updated. `false` when the gradient norm
    /// was not finite: the window's gradients were dropped instead.
    pub applied: bool,
    /// Global L2 norm of the window's gradients over every trainable
    /// tensor, before clipping.
    pub grad_norm: f32,
    /// Learning rate of the LoRA optimizer (for a skipped update, the rate
    /// it already held).
    pub lr: f32,
    /// Learning rate of the head optimizer (as for `lr`).
    pub head_lr: f32,
}

/// Result of one [`ClassifierTrainer::train_batch`] call.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClassifierStepStats {
    /// Mean cross-entropy over the rows of this batch.
    pub loss: f32,
    /// Rows in this batch.
    pub rows: usize,
    /// Set when this batch completed an accumulation window.
    pub update: Option<OptimizerUpdate>,
}

/// Trains the LoRA adapters of a model and a classification head on top of
/// it, with one AdamW optimizer per group so they can use different
/// learning rates.
///
/// Gradients of `grad_accum` batches are averaged (each batch's mean loss
/// is scaled by `1 / grad_accum`, as in [`Trainer`](crate::Trainer)), then
/// the global norm over all trainable tensors is clipped and both
/// optimizers step. A window whose gradient norm is not finite is dropped
/// and counted in [`skipped_steps`](Self::skipped_steps).
pub struct ClassifierTrainer {
    /// The decoder; only its LoRA tensors are trained.
    pub model: Lfm2Model,
    /// The classification head.
    pub head: ClassifierHead,
    /// GPU context.
    pub ctx: Arc<WgpuContext>,
    /// Kernel engine.
    pub engine: Arc<ComputeEngine>,
    /// Peak learning rate of the LoRA tensors.
    pub lr: f32,
    /// Peak learning rate of the head tensors.
    pub head_lr: f32,
    /// Batches per optimizer step.
    pub grad_accum: usize,
    /// Clip the global gradient norm to this value before each step.
    pub clip_grad_norm: Option<f32>,
    /// Schedule giving a factor applied to both `lr` and `head_lr`, advanced
    /// once per applied update before it is read (step `k` uses the
    /// schedule's value at `k`). Without one both rates are constant.
    pub lr_scheduler: Option<Box<dyn LRScheduler>>,
    /// Token used to pad rows on the right. Attention is causal and padding
    /// is excluded from the mean, so its value does not change the result.
    pub pad_token: u32,
    lora_optimizer: AdamW,
    head_optimizer: AdamW,
    lora_params: Vec<Tensor>,
    head_params: Vec<Tensor>,
    batches_in_window: usize,
    optimizer_steps: usize,
    skipped_steps: usize,
}

impl ClassifierTrainer {
    /// A trainer for the LoRA tensors of `model` (at `lr`) and the tensors
    /// of `head` (at `head_lr`), both with AdamW and weight decay
    /// [`WEIGHT_DECAY`]. Defaults: no accumulation, clipping at 1.0, no
    /// schedule, pad token 0.
    pub fn new(
        model: Lfm2Model,
        head: ClassifierHead,
        ctx: Arc<WgpuContext>,
        engine: Arc<ComputeEngine>,
        lr: f32,
        head_lr: f32,
    ) -> Self {
        // A fixed order keeps the gradient-norm sum, and so the run,
        // reproducible.
        let named = model.named_lora_params();
        let mut names: Vec<&String> = named.keys().collect();
        names.sort();
        let lora_params = names.into_iter().map(|n| named[n].clone()).collect();
        let head_params = head.params();
        let mut lora_optimizer = AdamW::new(engine.clone(), lr);
        lora_optimizer.weight_decay = WEIGHT_DECAY;
        let mut head_optimizer = AdamW::new(engine.clone(), head_lr);
        head_optimizer.weight_decay = WEIGHT_DECAY;
        Self {
            model,
            head,
            ctx,
            engine,
            lr,
            head_lr,
            grad_accum: 1,
            clip_grad_norm: Some(1.0),
            lr_scheduler: None,
            pad_token: 0,
            lora_optimizer,
            head_optimizer,
            lora_params,
            head_params,
            batches_in_window: 0,
            optimizer_steps: 0,
            skipped_steps: 0,
        }
    }

    /// Average the gradients of `batches` batches per optimizer step.
    pub fn with_grad_accum(mut self, batches: usize) -> Self {
        self.grad_accum = batches.max(1);
        self
    }

    /// Clip the global gradient norm to `max_norm` (`None` disables it).
    pub fn with_clip_grad_norm(mut self, max_norm: Option<f32>) -> Self {
        self.clip_grad_norm = max_norm;
        self
    }

    /// Scale both learning rates by `scheduler` (see
    /// [`lr_scheduler`](Self::lr_scheduler)); build it with a peak of 1.0,
    /// e.g. `WarmupCosineScheduler::new(1.0, 0.1, warmup, total)`.
    pub fn with_lr_scheduler(mut self, scheduler: Box<dyn LRScheduler>) -> Self {
        self.lr_scheduler = Some(scheduler);
        self
    }

    /// Pad rows with `token`.
    pub fn with_pad_token(mut self, token: u32) -> Self {
        self.pad_token = token;
        self
    }

    /// The LoRA tensors being trained, sorted by name.
    pub fn lora_params(&self) -> &[Tensor] {
        &self.lora_params
    }

    /// The head tensors being trained.
    pub fn head_params(&self) -> &[Tensor] {
        &self.head_params
    }

    /// Updates applied so far.
    pub fn optimizer_steps(&self) -> usize {
        self.optimizer_steps
    }

    /// Windows dropped because their gradient norm was not finite.
    pub fn skipped_steps(&self) -> usize {
        self.skipped_steps
    }

    /// Batches accumulated in the current, unfinished window.
    pub fn pending_batches(&self) -> usize {
        self.batches_in_window
    }

    /// Forward, loss and backward on one batch of rows (any lengths; they
    /// are padded on the right to the longest), with one label per row.
    /// Every [`grad_accum`](Self::grad_accum) batches the accumulated
    /// gradients are clipped and applied.
    ///
    /// Fails, before running anything, on an empty batch or row, a label
    /// count that differs from the row count, or a label not below the
    /// number of classes.
    pub fn train_batch(
        &mut self,
        rows: &[&[u32]],
        labels: &[u32],
    ) -> Result<ClassifierStepStats, TrainError> {
        let classes = self.head.classes();
        if labels.len() != rows.len() {
            return Err(TrainError(format!(
                "{} labels for {} rows",
                labels.len(),
                rows.len()
            )));
        }
        if let Some(&bad) = labels.iter().find(|&&l| l as usize >= classes) {
            return Err(TrainError(format!(
                "label {bad} is not below the {classes} classes"
            )));
        }
        let (padded, lengths) = pad_right(rows, self.pad_token)?;
        let refs: Vec<&[u32]> = padded.iter().map(Vec::as_slice).collect();
        let logits = self
            .head
            .forward(&self.engine, &self.model, &refs, &lengths);
        let loss = classification_loss(&logits, labels, &self.engine)
            .ok_or_else(|| TrainError("labels do not match the logits".into()))?;
        // Seeding backward with 1 / grad_accum makes a window's gradient the
        // mean over its batches.
        *loss.tensor.grad.borrow_mut() = Some(UnifiedTensor::new(
            &self.ctx,
            &[1.0 / self.grad_accum as f32],
            &[1],
        ));
        loss.tensor.backward(&self.engine);

        self.batches_in_window += 1;
        let update = (self.batches_in_window >= self.grad_accum).then(|| self.finish_window());
        Ok(ClassifierStepStats {
            loss: loss.value,
            rows: rows.len(),
            update,
        })
    }

    /// Apply the gradients of an unfinished window (e.g. the last batches of
    /// a run), rescaled so they are the mean over the batches it holds.
    /// Returns `None` when no batch is pending.
    pub fn flush(&mut self) -> Option<OptimizerUpdate> {
        if self.batches_in_window == 0 {
            return None;
        }
        let rescale = self.grad_accum as f32 / self.batches_in_window as f32;
        if rescale != 1.0 {
            for param in self.lora_params.iter().chain(&self.head_params) {
                if let Some(grad) = param.grad.borrow().as_ref() {
                    self.engine.gradient_scale(&self.ctx, grad, rescale);
                }
            }
        }
        Some(self.finish_window())
    }

    /// Clip and apply (or drop) the accumulated gradients, then clear them.
    fn finish_window(&mut self) -> OptimizerUpdate {
        self.batches_in_window = 0;
        let grads: Vec<UnifiedTensor> = self
            .lora_params
            .iter()
            .chain(&self.head_params)
            .filter_map(|p| p.grad.borrow().clone())
            .collect();
        let grad_norm = global_grad_norm(&grads, &self.ctx, &self.engine);
        let applied = grad_norm.is_finite();
        if applied {
            if let Some(max_norm) = self.clip_grad_norm {
                if grad_norm > max_norm {
                    let scale = max_norm / (grad_norm + 1e-6);
                    for grad in &grads {
                        self.engine.clip_grad_by_norm(&self.ctx, grad, scale);
                    }
                }
            }
            let factor = match self.lr_scheduler.as_mut() {
                Some(s) => {
                    s.step();
                    s.get_lr()
                }
                None => 1.0,
            };
            self.lora_optimizer.lr = self.lr * factor;
            self.head_optimizer.lr = self.head_lr * factor;
            self.lora_optimizer.step(&self.lora_params);
            self.head_optimizer.step(&self.head_params);
            self.optimizer_steps += 1;
        } else {
            log::warn!("gradient norm is {grad_norm}; skipping optimizer step");
            self.skipped_steps += 1;
        }
        self.lora_optimizer.zero_grad(&self.lora_params);
        self.head_optimizer.zero_grad(&self.head_params);
        OptimizerUpdate {
            applied,
            grad_norm,
            lr: self.lora_optimizer.lr,
            head_lr: self.head_optimizer.lr,
        }
    }

    /// Class probabilities of one batch of rows, see [`predict_probs`].
    pub fn predict(&self, rows: &[&[u32]]) -> Result<Vec<Vec<f32>>, TrainError> {
        predict_probs(&self.engine, &self.model, &self.head, rows, self.pad_token)
    }
}

/// Right-pad `rows` with `pad` to the longest; returns the padded rows and
/// the original lengths.
fn pad_right(rows: &[&[u32]], pad: u32) -> Result<(Vec<Vec<u32>>, Vec<usize>), TrainError> {
    if rows.is_empty() {
        return Err(TrainError("a batch needs at least one row".into()));
    }
    if let Some(i) = rows.iter().position(|r| r.is_empty()) {
        return Err(TrainError(format!("row {i} of the batch has no tokens")));
    }
    let longest = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    let lengths = rows.iter().map(|r| r.len()).collect();
    let padded = rows
        .iter()
        .map(|r| {
            let mut row = r.to_vec();
            row.resize(longest, pad);
            row
        })
        .collect();
    Ok((padded, lengths))
}

/// Softmax probabilities `[classes]` per row of one batch of token rows,
/// padded on the right with `pad_token`. Only the forward pass runs; no
/// gradient is computed.
///
/// Fails on an empty row; an empty batch gives an empty result.
pub fn predict_probs(
    engine: &ComputeEngine,
    model: &Lfm2Model,
    head: &ClassifierHead,
    rows: &[&[u32]],
    pad_token: u32,
) -> Result<Vec<Vec<f32>>, TrainError> {
    if rows.is_empty() {
        return Ok(Vec::new());
    }
    let (padded, lengths) = pad_right(rows, pad_token)?;
    let refs: Vec<&[u32]> = padded.iter().map(Vec::as_slice).collect();
    let logits = head.forward(engine, model, &refs, &lengths);
    let host = pollster::block_on(logits.data.to_vec(&logits.ctx));
    Ok(softmax_rows(&host, head.classes()))
}

/// [`predict_probs`] over any number of rows, `batch_size` at a time.
/// Rows are grouped by length to keep padding small; the result is in the
/// order of `rows`.
pub fn predict_rows(
    engine: &ComputeEngine,
    model: &Lfm2Model,
    head: &ClassifierHead,
    rows: &[Vec<u32>],
    batch_size: usize,
    pad_token: u32,
) -> Result<Vec<Vec<f32>>, TrainError> {
    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_by_key(|&i| rows[i].len());
    let mut out = vec![Vec::new(); rows.len()];
    for chunk in order.chunks(batch_size.max(1)) {
        let batch: Vec<&[u32]> = chunk.iter().map(|&i| rows[i].as_slice()).collect();
        let probs = predict_probs(engine, model, head, &batch, pad_token)?;
        for (&i, p) in chunk.iter().zip(probs) {
            out[i] = p;
        }
    }
    Ok(out)
}

/// Row-wise softmax of `[rows, classes]` logits, computed in f64.
pub fn softmax_rows(logits: &[f32], classes: usize) -> Vec<Vec<f32>> {
    logits
        .chunks(classes.max(1))
        .map(|row| {
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let exps: Vec<f64> = row.iter().map(|&x| (x as f64 - max).exp()).collect();
            let sum: f64 = exps.iter().sum();
            exps.iter().map(|e| (e / sum) as f32).collect()
        })
        .collect()
}

/// Index of the largest probability (the first one on a tie).
pub fn predicted_class(probs: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &p) in probs.iter().enumerate() {
        if p > probs[best] {
            best = i;
        }
    }
    best as u32
}

// ---------------------------------------------------------------------------
// Files

/// Save the head's tensors (F32 safetensors, named as in
/// [`ClassifierHead::named_params`]) to `path`, through a temporary file.
pub async fn save_classifier_head<P: AsRef<Path>>(
    ctx: &Arc<WgpuContext>,
    path: P,
    head: &ClassifierHead,
) -> Result<(), Box<dyn std::error::Error>> {
    save_lora_checkpoint(ctx, path, &head.named_params(), None).await
}

/// Load a head saved by [`save_classifier_head`] for `taps`.
///
/// Fails when the file lacks a tensor for one of the taps, holds tensors for
/// other taps, or the shapes are not `[hidden]` and `[hidden, classes]`
/// with the same `hidden` and `classes` for every tap.
pub fn load_classifier_head<P: AsRef<Path>>(
    ctx: &Arc<WgpuContext>,
    path: P,
    taps: &[usize],
    skip_first: bool,
) -> Result<ClassifierHead, Box<dyn std::error::Error>> {
    let path = path.as_ref();
    if taps.is_empty() {
        return Err("a classifier head needs at least one tap".into());
    }
    let ckpt = load_lora_checkpoint(path)?;
    if ckpt.len() != 2 * taps.len() {
        return Err(format!(
            "{} holds {} tensors; a head over {} taps has {}",
            path.display(),
            ckpt.len(),
            taps.len(),
            2 * taps.len()
        )
        .into());
    }
    let mut dims: Option<(usize, usize)> = None;
    let mut norms = Vec::with_capacity(taps.len());
    let mut weights = Vec::with_capacity(taps.len());
    for &tap in taps {
        let norm_name = format!("classifier.norms.{tap}.weight");
        let weight_name = format!("classifier.heads.{tap}.weight");
        let missing = |name: &str| format!("{} has no tensor {name}", path.display());
        let (norm_shape, norm_values) = ckpt
            .tensors
            .get(&norm_name)
            .ok_or_else(|| missing(&norm_name))?;
        let (weight_shape, weight_values) = ckpt
            .tensors
            .get(&weight_name)
            .ok_or_else(|| missing(&weight_name))?;
        let (hidden, classes) = match weight_shape[..] {
            [hidden, classes] if hidden > 0 && classes > 0 => (hidden, classes),
            _ => {
                return Err(format!(
                    "{}: {weight_name} has shape {weight_shape:?}, expected [hidden, classes]",
                    path.display()
                )
                .into())
            }
        };
        if norm_shape[..] != [hidden] {
            return Err(format!(
                "{}: {norm_name} has shape {norm_shape:?}, expected [{hidden}]",
                path.display()
            )
            .into());
        }
        if dims.is_some_and(|d| d != (hidden, classes)) {
            return Err(format!("{}: the taps have different shapes", path.display()).into());
        }
        dims = Some((hidden, classes));
        norms.push(Tensor::new(
            ctx.clone(),
            UnifiedTensor::new(ctx, norm_values, norm_shape),
        ));
        weights.push(Tensor::new(
            ctx.clone(),
            UnifiedTensor::new(ctx, weight_values, weight_shape),
        ));
    }
    Ok(ClassifierHead::from_parts(taps.to_vec(), norms, weights).with_skip_first(skip_first))
}

/// Settings of a trained classifier, stored as [`METADATA_FILE`] next to
/// the adapter and the head.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ClassifierMetadata {
    /// Label names in class-index order.
    pub labels: Vec<String>,
    /// Tapped blocks, 1-based.
    pub taps: Vec<usize>,
    /// Rows were truncated to this many tokens.
    pub max_len: usize,
    /// Whether the first token of each row is left out of the mean.
    pub skip_first_token: bool,
    /// LoRA rank.
    pub rank: usize,
    /// LoRA alpha.
    pub alpha: f32,
    /// Projections that carry LoRA adapters.
    pub lora_targets: Vec<String>,
    /// Name of the base model directory.
    pub base_model: String,
    /// Seed of the head initialisation and the data order.
    pub seed: u64,
    /// Version of Ferrotensor that wrote the file.
    pub ferrotensor_version: String,
}

impl ClassifierMetadata {
    /// Write as pretty JSON through a temporary file.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let path = path.as_ref();
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_string_pretty(self)?)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Read a file written by [`save`](Self::save).
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let path = path.as_ref();
        let json = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        serde_json::from_str(&json).map_err(|e| format!("{}: {e}", path.display()).into())
    }

    /// Class index of `label`.
    pub fn label_index(&self, label: &str) -> Option<u32> {
        self.labels
            .iter()
            .position(|l| l == label)
            .map(|i| i as u32)
    }
}

// ---------------------------------------------------------------------------
// Data

/// One line of a classification JSONL file.
#[derive(Clone, Debug, PartialEq)]
pub struct ClassifyRecord {
    /// The `id` field (a string or a number), or the 0-based line number.
    pub id: serde_json::Value,
    /// The `text` field.
    pub text: String,
    /// The `label` field, if present.
    pub label: Option<String>,
}

/// Parse JSONL with one object per line: `text` (string, required), `label`
/// (string; required when `require_label`) and `id` (string or number;
/// defaults to the 0-based line number). Other fields are ignored.
///
/// Row `i` of the result is line `i`, so empty lines are an error, except
/// at the end of the file. Errors name the 1-based line.
pub fn parse_classify_jsonl(
    content: &str,
    require_label: bool,
) -> Result<Vec<ClassifyRecord>, String> {
    let mut lines: Vec<&str> = content.lines().collect();
    while lines.last().is_some_and(|l| l.trim().is_empty()) {
        lines.pop();
    }
    let mut records = Vec::with_capacity(lines.len());
    for (i, line) in lines.iter().enumerate() {
        let n = i + 1;
        if line.trim().is_empty() {
            return Err(format!("line {n}: empty line"));
        }
        let value: serde_json::Value =
            serde_json::from_str(line).map_err(|e| format!("line {n}: {e}"))?;
        let Some(object) = value.as_object() else {
            return Err(format!("line {n}: expected a JSON object"));
        };
        let text = match object.get("text") {
            Some(serde_json::Value::String(text)) => text.clone(),
            Some(_) => return Err(format!("line {n}: \"text\" must be a string")),
            None => return Err(format!("line {n}: missing \"text\"")),
        };
        let label = match object.get("label") {
            Some(serde_json::Value::String(label)) => Some(label.clone()),
            None | Some(serde_json::Value::Null) => None,
            Some(_) => return Err(format!("line {n}: \"label\" must be a string")),
        };
        if require_label && label.is_none() {
            return Err(format!("line {n}: missing \"label\""));
        }
        let id = match object.get("id") {
            Some(id @ (serde_json::Value::String(_) | serde_json::Value::Number(_))) => id.clone(),
            None | Some(serde_json::Value::Null) => serde_json::Value::from(i),
            Some(_) => return Err(format!("line {n}: \"id\" must be a string or a number")),
        };
        records.push(ClassifyRecord { id, text, label });
    }
    Ok(records)
}

/// The distinct labels of `records`, sorted in byte order: index `i` of the
/// result is class `i`.
pub fn label_names(records: &[ClassifyRecord]) -> Vec<String> {
    let mut names: Vec<String> = records.iter().filter_map(|r| r.label.clone()).collect();
    names.sort();
    names.dedup();
    names
}

/// The class index of every record's label in `labels`. Fails, naming the
/// row, when a record has no label or one that is not in `labels`.
pub fn label_ids(records: &[ClassifyRecord], labels: &[String]) -> Result<Vec<u32>, String> {
    records
        .iter()
        .enumerate()
        .map(|(row, r)| {
            let label = r
                .label
                .as_deref()
                .ok_or_else(|| format!("row {row} has no label"))?;
            labels
                .iter()
                .position(|l| l == label)
                .map(|i| i as u32)
                .ok_or_else(|| format!("row {row}: unknown label {label:?}"))
        })
        .collect()
}

/// Rows per chunk in [`plan_epoch`], in batches.
pub const BUCKET_BATCHES: usize = 50;

/// SplitMix64: small, fast, and the same on every platform.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Fisher-Yates shuffle.
    fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = (self.next() % (i as u64 + 1)) as usize;
            items.swap(i, j);
        }
    }
}

/// The batches of one epoch over rows with the given token `lengths`:
/// shuffle the rows, sort each chunk of [`BUCKET_BATCHES`] batches by
/// length (so a batch holds rows of similar length and little padding),
/// cut the chunks into batches of `batch_size` and shuffle the batch order.
///
/// The plan depends only on `lengths`, `batch_size`, `seed` and `epoch`.
/// Every row appears exactly once; only the last batch can be shorter.
///
/// # Panics
/// If `batch_size` is zero.
pub fn plan_epoch(lengths: &[usize], batch_size: usize, seed: u64, epoch: u64) -> Vec<Vec<usize>> {
    assert!(batch_size > 0, "plan_epoch: batch_size must be positive");
    let mut rng = SplitMix64(seed ^ epoch.wrapping_mul(0xD1B5_4A32_D192_ED03));
    let mut order: Vec<usize> = (0..lengths.len()).collect();
    rng.shuffle(&mut order);
    let mut batches = Vec::with_capacity(lengths.len().div_ceil(batch_size));
    for chunk in order.chunks_mut(batch_size * BUCKET_BATCHES) {
        // Stable, so rows of equal length keep their shuffled order.
        chunk.sort_by_key(|&i| lengths[i]);
        batches.extend(chunk.chunks(batch_size).map(<[usize]>::to_vec));
    }
    rng.shuffle(&mut batches);
    batches
}

/// Parse a batch plan: one JSON array of 0-based row indices per line, one
/// batch per line, in training order. Blank lines are ignored.
///
/// Fails, naming the 1-based line, on a line that is not an array of
/// non-negative integers, an empty batch, or an index not below `rows`;
/// and when the plan has no batch at all.
pub fn parse_batch_plan(content: &str, rows: usize) -> Result<Vec<Vec<usize>>, String> {
    let mut batches = Vec::new();
    for (i, line) in content.lines().enumerate() {
        let n = i + 1;
        if line.trim().is_empty() {
            continue;
        }
        let batch: Vec<usize> = serde_json::from_str(line)
            .map_err(|e| format!("line {n}: expected a JSON array of row indices: {e}"))?;
        if batch.is_empty() {
            return Err(format!("line {n}: empty batch"));
        }
        if let Some(&bad) = batch.iter().find(|&&r| r >= rows) {
            return Err(format!(
                "line {n}: row {bad} is out of range (the data has {rows} rows)"
            ));
        }
        batches.push(batch);
    }
    if batches.is_empty() {
        return Err("the batch plan has no batches".into());
    }
    Ok(batches)
}

/// Loss and accuracy of predictions against labels, see [`score`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClassifyScores {
    /// Rows scored.
    pub rows: usize,
    /// Mean of `-ln p(label)`, with `p` floored at `1e-30`.
    pub loss: f64,
    /// Fraction of rows whose most likely class is the label.
    pub accuracy: f64,
    /// Mean per-class recall over the classes that occur in the labels.
    pub balanced_accuracy: f64,
}

/// Fraction of `predicted` equal to `labels` (NaN when empty).
pub fn accuracy(predicted: &[u32], labels: &[u32]) -> f64 {
    let correct = predicted.iter().zip(labels).filter(|(p, l)| p == l).count();
    correct as f64 / labels.len() as f64
}

/// Mean recall over the classes that occur in `labels`: for each such
/// class, the fraction of its rows predicted as that class. Classes with
/// no row do not count. NaN when `labels` is empty.
pub fn balanced_accuracy(predicted: &[u32], labels: &[u32], classes: usize) -> f64 {
    let mut total = vec![0usize; classes];
    let mut correct = vec![0usize; classes];
    for (&p, &l) in predicted.iter().zip(labels) {
        let Some(t) = total.get_mut(l as usize) else {
            continue;
        };
        *t += 1;
        if p == l {
            correct[l as usize] += 1;
        }
    }
    let recalls: Vec<f64> = total
        .iter()
        .zip(&correct)
        .filter(|&(&t, _)| t > 0)
        .map(|(&t, &c)| c as f64 / t as f64)
        .collect();
    recalls.iter().sum::<f64>() / recalls.len() as f64
}

/// Loss, accuracy and balanced accuracy of per-row probabilities against
/// `labels` (one per row).
pub fn score(probs: &[Vec<f32>], labels: &[u32], classes: usize) -> ClassifyScores {
    let predicted: Vec<u32> = probs.iter().map(|p| predicted_class(p)).collect();
    let loss = probs
        .iter()
        .zip(labels)
        .map(|(p, &l)| {
            -(p.get(l as usize).copied().unwrap_or(0.0) as f64)
                .max(1e-30)
                .ln()
        })
        .sum::<f64>()
        / labels.len() as f64;
    ClassifyScores {
        rows: labels.len(),
        loss,
        accuracy: accuracy(&predicted, labels),
        balanced_accuracy: balanced_accuracy(&predicted, labels, classes),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jsonl_records_parse_with_default_ids() {
        let content = concat!(
            "{\"text\": \"a b\", \"label\": \"x\"}\n",
            "{\"text\": \"c\", \"label\": \"y\", \"id\": \"doc-7\"}\n",
            "{\"text\": \"d\", \"id\": 42, \"extra\": [1]}\n",
            "\n\n",
        );
        let records = parse_classify_jsonl(content, false).expect("parse");
        assert_eq!(records.len(), 3);
        assert_eq!(records[0].id, serde_json::json!(0));
        assert_eq!(records[0].text, "a b");
        assert_eq!(records[0].label.as_deref(), Some("x"));
        assert_eq!(records[1].id, serde_json::json!("doc-7"));
        assert_eq!(records[2].id, serde_json::json!(42));
        assert_eq!(records[2].label, None);
    }

    #[test]
    fn jsonl_errors_name_the_line() {
        let missing_label = "{\"text\": \"a\", \"label\": \"x\"}\n{\"text\": \"b\"}\n";
        let err = parse_classify_jsonl(missing_label, true).expect_err("label required");
        assert!(err.starts_with("line 2:"), "{err}");
        assert!(parse_classify_jsonl(missing_label, false).is_ok());

        let cases = [
            ("{\"label\": \"x\"}", "missing \"text\""),
            ("{\"text\": 3}", "\"text\" must be a string"),
            (
                "{\"text\": \"a\", \"label\": 1}",
                "\"label\" must be a string",
            ),
            ("{\"text\": \"a\", \"id\": [1]}", "\"id\" must be a string"),
            ("[1, 2]", "expected a JSON object"),
            ("not json", "line 1"),
        ];
        for (line, expected) in cases {
            let err = parse_classify_jsonl(line, false).expect_err(line);
            assert!(err.contains(expected), "{line}: {err}");
        }
        let gap = "{\"text\": \"a\"}\n\n{\"text\": \"b\"}\n";
        let err = parse_classify_jsonl(gap, false).expect_err("blank line inside");
        assert!(err.contains("line 2: empty line"), "{err}");
    }

    #[test]
    fn labels_are_sorted_in_byte_order_and_indexed() {
        let records: Vec<ClassifyRecord> = ["b", "a", "B", "b", "_"]
            .iter()
            .enumerate()
            .map(|(i, l)| ClassifyRecord {
                id: serde_json::json!(i),
                text: String::new(),
                label: Some(l.to_string()),
            })
            .collect();
        let names = label_names(&records);
        assert_eq!(names, vec!["B", "_", "a", "b"]);
        assert_eq!(
            label_ids(&records, &names).expect("ids"),
            vec![3, 2, 0, 3, 1]
        );

        let unknown = label_ids(&records, &names[..2]).expect_err("unknown label");
        assert!(unknown.contains("row 0"), "{unknown}");
    }

    #[test]
    fn epoch_plans_are_deterministic_and_cover_every_row() {
        let lengths: Vec<usize> = (0..1003).map(|i| (i * 37) % 200 + 1).collect();
        let plan = plan_epoch(&lengths, 8, 3, 0);
        assert_eq!(
            plan,
            plan_epoch(&lengths, 8, 3, 0),
            "same inputs, same plan"
        );
        assert_ne!(plan, plan_epoch(&lengths, 8, 4, 0), "the seed matters");
        assert_ne!(plan, plan_epoch(&lengths, 8, 3, 1), "the epoch matters");

        let mut seen: Vec<usize> = plan.iter().flatten().copied().collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..lengths.len()).collect::<Vec<_>>());
        assert_eq!(plan.len(), lengths.len().div_ceil(8));
        assert_eq!(plan.iter().filter(|b| b.len() != 8).count(), 1);

        // Length bucketing: within a batch the lengths are sorted and close.
        let spread: usize = plan
            .iter()
            .map(|b| {
                let l: Vec<usize> = b.iter().map(|&i| lengths[i]).collect();
                assert!(l.windows(2).all(|w| w[0] <= w[1]), "{l:?}");
                l[l.len() - 1] - l[0]
            })
            .sum();
        assert!(
            spread / plan.len() < 20,
            "mean spread {}",
            spread / plan.len()
        );
    }

    #[test]
    fn batch_plans_are_validated() {
        let plan = parse_batch_plan("[0, 2]\n\n[1]\n", 3).expect("plan");
        assert_eq!(plan, vec![vec![0, 2], vec![1]]);
        let out_of_range = parse_batch_plan("[0]\n[3]\n", 3).expect_err("index 3");
        assert!(
            out_of_range.contains("line 2") && out_of_range.contains("row 3"),
            "{out_of_range}"
        );
        let empty = parse_batch_plan("[0]\n[]\n", 3).expect_err("empty batch");
        assert!(empty.contains("line 2: empty batch"), "{empty}");
        assert!(parse_batch_plan("[-1]\n", 3).is_err());
        assert!(parse_batch_plan("{\"a\": 1}\n", 3).is_err());
        assert!(parse_batch_plan("\n", 3).is_err());
    }

    #[test]
    fn balanced_accuracy_is_mean_recall() {
        // Class 0: 3 of 4 right; class 1: 1 of 2; class 2 never occurs.
        let labels = [0, 0, 0, 0, 1, 1];
        let predicted = [0, 0, 0, 1, 1, 2];
        assert!((accuracy(&predicted, &labels) - 4.0 / 6.0).abs() < 1e-12);
        let balanced = balanced_accuracy(&predicted, &labels, 3);
        assert!((balanced - (0.75 + 0.5) / 2.0).abs() < 1e-12, "{balanced}");
        assert!(balanced_accuracy(&[], &[], 3).is_nan());
    }

    #[test]
    fn scores_follow_the_probabilities() {
        let probs = vec![
            vec![0.7, 0.2, 0.1],
            vec![0.1, 0.1, 0.8],
            vec![0.5, 0.5, 0.0],
        ];
        let s = score(&probs, &[0, 1, 0], 3);
        assert_eq!(s.rows, 3);
        assert!((s.accuracy - 2.0 / 3.0).abs() < 1e-12);
        let loss = -(0.7f64.ln() + 0.1f32.ln() as f64 + 0.5f64.ln()) / 3.0;
        assert!((s.loss - loss).abs() < 1e-6, "{} vs {loss}", s.loss);
        assert_eq!(predicted_class(&[0.5, 0.5, 0.0]), 0, "ties go to the first");
    }

    #[test]
    fn softmax_rows_sum_to_one() {
        let probs = softmax_rows(&[1.0, 2.0, 3.0, -50.0, 0.0, 50.0], 3);
        assert_eq!(probs.len(), 2);
        for row in &probs {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "{row:?}");
        }
        assert!(probs[0][2] > probs[0][1] && probs[0][1] > probs[0][0]);
    }
}
