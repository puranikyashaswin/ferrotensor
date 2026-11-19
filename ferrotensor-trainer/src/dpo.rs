//! Direct Preference Optimization (Rafailov et al., 2023) for LoRA adapters.
//!
//! The policy is the model with its adapters. The reference is the same
//! model with the adapters switched off ([`without_adapters`]) and run under
//! [`no_grad`], so no second copy of the weights is held. A sequence's
//! log-probability is the sum of its target log-probabilities over the
//! response positions (loss mask 1), and the loss of a pair is
//! `-log sigmoid(beta * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected)))`,
//! averaged over the pairs. [`Trainer::train_dpo`] runs it with the
//! trainer's accumulation, clipping, schedule and optimizer.

use crate::data::PreferencePair;
use crate::{memory, StepStats, TrainError, Trainer};
use ferrotensor_autograd::{no_grad, AutogradNode, Tensor};
use ferrotensor_backend::{
    ComputeEngine, MemoryCategory, UnifiedTensor, WgpuContext, MASKED_CE_SKIP,
};
use ferrotensor_graph::lfm::Lfm2Model;
use ferrotensor_graph::without_adapters;
use std::sync::Arc;

/// The DPO temperature used when none is given.
pub const DEFAULT_BETA: f32 = 0.1;

/// Policy logits of one forward pass, with the targets and loss mask of
/// their rows: the input of [`dpo_loss`].
#[derive(Clone, Copy)]
pub struct ScoredRows<'a> {
    /// Logits `[sequences, seq, vocab]` (or `[seq, vocab]` for one sequence).
    pub logits: &'a Tensor,
    /// One target id per row, sequences one after another.
    pub target_ids: &'a [u32],
    /// One entry per row: 1 for a response position, 0 to ignore.
    pub loss_mask: &'a [u8],
    /// Number of sequences in `logits`.
    pub sequences: usize,
}

/// The kernel's target per row ([`MASKED_CE_SKIP`] where a row is not
/// scored) and the scored rows of each sequence.
struct Rows {
    targets: Vec<u32>,
    tokens: Vec<usize>,
    seq_len: usize,
}

fn scored_rows(
    shape: &[usize],
    target_ids: &[u32],
    loss_mask: &[u8],
    sequences: usize,
) -> Option<Rows> {
    let vocab = *shape.last()?;
    let rows = shape.iter().product::<usize>() / vocab.max(1);
    if vocab == 0 || sequences == 0 || !rows.is_multiple_of(sequences) {
        return None;
    }
    let seq_len = rows / sequences;
    let mut targets = Vec::with_capacity(rows);
    let mut tokens = vec![0usize; sequences];
    for r in 0..rows {
        let scored = r < target_ids.len()
            && r < loss_mask.len()
            && loss_mask[r] != 0
            && (target_ids[r] as usize) < vocab;
        if scored {
            targets.push(target_ids[r]);
            tokens[r / seq_len] += 1;
        } else {
            targets.push(MASKED_CE_SKIP);
        }
    }
    Some(Rows {
        targets,
        tokens,
        seq_len,
    })
}

/// Runs the masked cross-entropy kernel on `logits` with one weight per row,
/// writing `(softmax - onehot) * weight` into `grad`, and returns each
/// sequence's log-probability: minus the sum of its rows' cross-entropy.
fn log_probs_and_grad(
    ctx: &WgpuContext,
    logits: &UnifiedTensor,
    rows: &Rows,
    target_buf: &UnifiedTensor,
    weights: &[f32],
    grad: &UnifiedTensor,
    engine: &ComputeEngine,
) -> Vec<f64> {
    let n = rows.targets.len();
    let weight_buf = UnifiedTensor::new(ctx, weights, &[n]);
    let row_loss = UnifiedTensor::empty(ctx, &[n]);
    engine.masked_cross_entropy_weighted(ctx, logits, target_buf, &weight_buf, grad, &row_loss);
    let losses = pollster::block_on(row_loss.to_vec(ctx));
    losses
        .chunks(rows.seq_len.max(1))
        .map(|seq| -seq.iter().map(|&l| l as f64).sum::<f64>())
        .collect()
}

/// Log-probability of each sequence of `logits` (`[sequences, seq, vocab]`,
/// rows flattened): the sum over the rows where `loss_mask` is 1 of the
/// log-softmax at the target id, with the number of such rows.
///
/// Nothing is recorded for backward. Returns `None` when the shapes do not
/// fit `sequences`.
pub fn sequence_log_probs(
    logits: &Tensor,
    target_ids: &[u32],
    loss_mask: &[u8],
    sequences: usize,
    engine: &ComputeEngine,
) -> Option<Vec<(f64, usize)>> {
    let rows = scored_rows(&logits.data.shape, target_ids, loss_mask, sequences)?;
    let ctx = &logits.ctx;
    let target_buf = UnifiedTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(&rows.targets),
        &[rows.targets.len()],
    );
    let grad = UnifiedTensor::empty(ctx, &logits.data.shape);
    let weights = vec![0.0f32; rows.targets.len()];
    let log_probs = log_probs_and_grad(
        ctx,
        &logits.data,
        &rows,
        &target_buf,
        &weights,
        &grad,
        engine,
    );
    Some(log_probs.into_iter().zip(rows.tokens).collect())
}

/// Log-probability of each of `rows` under the reference model: `model`
/// with its LoRA adapters switched off, run without recording a graph.
///
/// `rows` must be equally long; `target_ids` and `loss_mask` hold one entry
/// per row position, rows one after another. An all-attention model runs
/// them as one batch, other models one at a time. Returns `None` when the
/// lengths do not fit.
pub fn reference_log_probs(
    model: &Lfm2Model,
    engine: &ComputeEngine,
    rows: &[&[u32]],
    target_ids: &[u32],
    loss_mask: &[u8],
) -> Option<Vec<(f64, usize)>> {
    let seq_len = rows.first()?.len();
    if rows.iter().any(|r| r.len() != seq_len)
        || target_ids.len() != rows.len() * seq_len
        || loss_mask.len() != target_ids.len()
    {
        return None;
    }
    no_grad(|| {
        without_adapters(|| {
            if rows.len() > 1 && model.supports_batching() {
                let logits = model.forward_batch(engine, rows);
                sequence_log_probs(&logits, target_ids, loss_mask, rows.len(), engine)
            } else {
                let mut out = Vec::with_capacity(rows.len());
                for (i, row) in rows.iter().enumerate() {
                    let span = i * seq_len..(i + 1) * seq_len;
                    let logits = model.forward(engine, row, 0, None, None);
                    out.extend(sequence_log_probs(
                        &logits,
                        &target_ids[span.clone()],
                        &loss_mask[span],
                        1,
                        engine,
                    )?);
                }
                Some(out)
            }
        })
    })
}

/// The DPO loss of one pair from its four sequence log-probabilities.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairLoss {
    /// `-log sigmoid(margin)`.
    pub loss: f64,
    /// `beta * (pi_chosen - ref_chosen)`.
    pub chosen_reward: f64,
    /// `beta * (pi_rejected - ref_rejected)`.
    pub rejected_reward: f64,
    /// `chosen_reward - rejected_reward`.
    pub margin: f64,
    /// `d loss / d pi_chosen`, which is `-beta * sigmoid(-margin)`;
    /// `d loss / d pi_rejected` is its negative.
    pub grad_chosen: f64,
}

/// [`PairLoss`] for policy log-probabilities `policy` and reference
/// log-probabilities `reference`, each `(chosen, rejected)`.
pub fn pair_loss(policy: (f64, f64), reference: (f64, f64), beta: f64) -> PairLoss {
    let chosen_reward = beta * (policy.0 - reference.0);
    let rejected_reward = beta * (policy.1 - reference.1);
    let margin = chosen_reward - rejected_reward;
    // -log sigmoid(m) = log(1 + exp(-m)), computed without overflow.
    let loss = if margin >= 0.0 {
        (-margin).exp().ln_1p()
    } else {
        -margin + margin.exp().ln_1p()
    };
    // sigmoid(-m), computed without overflow.
    let sigmoid_neg = if margin >= 0.0 {
        let e = (-margin).exp();
        e / (1.0 + e)
    } else {
        1.0 / (1.0 + margin.exp())
    };
    PairLoss {
        loss,
        chosen_reward,
        rejected_reward,
        margin,
        grad_chosen: -beta * sigmoid_neg,
    }
}

/// What [`dpo_loss`] measured for one pair.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DpoPairStats {
    /// The pair's loss and rewards.
    pub loss: PairLoss,
    /// Policy log-probability of the chosen response.
    pub policy_chosen: f64,
    /// Policy log-probability of the rejected response.
    pub policy_rejected: f64,
    /// Scored (response) tokens of both responses.
    pub tokens: usize,
}

/// A differentiable DPO loss, see [`dpo_loss`].
pub struct DpoLoss {
    /// Mean loss over the pairs `[1]`; call `backward` on it.
    pub tensor: Tensor,
    /// The mean loss.
    pub value: f32,
    /// One entry per pair, in order.
    pub pairs: Vec<DpoPairStats>,
}

/// Backward node of [`dpo_loss`]: holds `d loss / d logits` of every
/// forward pass, computed on the GPU with the loss. Backward runs once.
struct DpoNode {
    logits: Vec<Tensor>,
    grads: Vec<UnifiedTensor>,
}

impl AutogradNode for DpoNode {
    fn inputs(&self) -> Vec<Tensor> {
        self.logits.clone()
    }

    fn backward(&self, grad: &UnifiedTensor, engine: &ComputeEngine) -> Vec<UnifiedTensor> {
        let Some(first) = self.logits.first() else {
            return Vec::new();
        };
        let ctx = &first.ctx;
        let upstream = pollster::block_on(grad.to_vec(ctx))
            .first()
            .copied()
            .unwrap_or(1.0);
        self.grads
            .iter()
            .map(|g| {
                let scaled = UnifiedTensor::empty(ctx, &g.shape);
                scaled.set_memory_category(MemoryCategory::Loss);
                engine.copy_tensor(ctx, g, &scaled, 0);
                if (upstream - 1.0).abs() > f32::EPSILON {
                    engine.gradient_scale(ctx, &scaled, upstream);
                }
                scaled
            })
            .collect()
    }
}

/// The DPO loss of a batch of preference pairs, as a differentiable scalar:
/// the mean over pairs of `-log sigmoid(beta * ((pi_c - ref_c) - (pi_r - ref_r)))`.
///
/// `policy` holds the policy logits of one or more forward passes; their
/// sequences, taken in order, alternate chosen and rejected (pair `i` is
/// sequences `2i` and `2i + 1`). `reference` holds the reference
/// log-probability of each sequence in the same order (see
/// [`reference_log_probs`]). The per-row cross-entropy is computed on the
/// GPU, read back and summed per sequence on the host; a second pass of the
/// same kernel writes `d loss / d logits`, which is
/// `-(d loss / d logp) * (softmax - onehot)` on each scored row.
///
/// Returns `None` when the shapes do not fit, the number of sequences is
/// odd or differs from `reference`, or a sequence has no scored row.
// Autograd nodes are shared through `Arc` but used on one thread.
#[allow(clippy::arc_with_non_send_sync)]
pub fn dpo_loss(
    policy: &[ScoredRows<'_>],
    reference: &[f64],
    beta: f32,
    engine: &ComputeEngine,
) -> Option<DpoLoss> {
    let ctx = policy.first()?.logits.ctx.clone();
    let mut passes = Vec::with_capacity(policy.len());
    let mut log_probs: Vec<f64> = Vec::new();
    let mut tokens: Vec<usize> = Vec::new();
    for p in policy {
        let rows = scored_rows(&p.logits.data.shape, p.target_ids, p.loss_mask, p.sequences)?;
        let target_buf = UnifiedTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&rows.targets),
            &[rows.targets.len()],
        );
        let grad = UnifiedTensor::empty(&ctx, &p.logits.data.shape);
        grad.set_memory_category(MemoryCategory::Loss);
        let zero = vec![0.0f32; rows.targets.len()];
        log_probs.extend(log_probs_and_grad(
            &ctx,
            &p.logits.data,
            &rows,
            &target_buf,
            &zero,
            &grad,
            engine,
        ));
        tokens.extend(&rows.tokens);
        passes.push((rows, target_buf, grad));
    }
    let sequences = log_probs.len();
    if sequences == 0
        || !sequences.is_multiple_of(2)
        || reference.len() != sequences
        || tokens.contains(&0)
    {
        return None;
    }

    let pairs = sequences / 2;
    let beta = beta as f64;
    let mut stats = Vec::with_capacity(pairs);
    // d loss / d logp of each sequence, for the mean over pairs.
    let mut d_log_prob = Vec::with_capacity(sequences);
    for i in 0..pairs {
        let (c, r) = (2 * i, 2 * i + 1);
        let loss = pair_loss(
            (log_probs[c], log_probs[r]),
            (reference[c], reference[r]),
            beta,
        );
        d_log_prob.push(loss.grad_chosen / pairs as f64);
        d_log_prob.push(-loss.grad_chosen / pairs as f64);
        stats.push(DpoPairStats {
            loss,
            policy_chosen: log_probs[c],
            policy_rejected: log_probs[r],
            tokens: tokens[c] + tokens[r],
        });
    }

    // Row r of a sequence with d loss / d logp = g gets weight -g: the
    // gradient of logp with respect to the row's logits is -(softmax - onehot).
    let mut first_seq = 0;
    let mut grads = Vec::with_capacity(passes.len());
    for (p, (rows, target_buf, grad)) in policy.iter().zip(passes) {
        let weights: Vec<f32> = rows
            .targets
            .iter()
            .enumerate()
            .map(|(r, &t)| {
                if t == MASKED_CE_SKIP {
                    0.0
                } else {
                    -d_log_prob[first_seq + r / rows.seq_len] as f32
                }
            })
            .collect();
        log_probs_and_grad(
            &ctx,
            &p.logits.data,
            &rows,
            &target_buf,
            &weights,
            &grad,
            engine,
        );
        first_seq += rows.tokens.len();
        grads.push(grad);
    }

    let value = (stats.iter().map(|s| s.loss.loss).sum::<f64>() / pairs as f64) as f32;
    let node = Arc::new(DpoNode {
        logits: policy.iter().map(|p| p.logits.clone()).collect(),
        grads,
    });
    let tensor = Tensor::with_creator(ctx.clone(), UnifiedTensor::new(&ctx, &[value], &[1]), node);
    Some(DpoLoss {
        tensor,
        value,
        pairs: stats,
    })
}

/// Result of one preference pair in [`Trainer::train_dpo`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DpoStats {
    /// `-log sigmoid(margin)` of this pair.
    pub loss: f32,
    /// `beta * (pi_chosen - ref_chosen)`.
    pub chosen_reward: f32,
    /// `beta * (pi_rejected - ref_rejected)`.
    pub rejected_reward: f32,
    /// `chosen_reward - rejected_reward`; the pair is ranked correctly when
    /// it is above 0.
    pub margin: f32,
    /// Scored (response) tokens of both responses.
    pub tokens: usize,
    /// Whether the optimizer stepped after this pair.
    pub optimizer_step: bool,
    /// Global gradient norm before clipping, when the optimizer stepped.
    pub grad_norm: Option<f32>,
    /// Learning rate of the optimizer step.
    pub lr: Option<f32>,
}

impl Trainer {
    /// Train on preference pairs with the DPO loss at temperature `beta`.
    /// Returns one [`DpoStats`] per pair.
    ///
    /// `grad_accum_steps` counts pairs here: each pair adds
    /// `1 / grad_accum_steps` of its loss gradient, so an optimizer step
    /// applies the gradient of the mean loss over its window. Chosen and
    /// rejected sequences of the pairs in one window (up to the accumulation
    /// boundary) run through the model together on an all-attention model,
    /// padded on the right to a common length; the reference pass runs
    /// first, with the adapters off and nothing recorded.
    ///
    /// # Errors
    ///
    /// If a pair has no response token to score, or the optimizer step
    /// fails (gradient sync).
    pub fn train_dpo(
        &mut self,
        pairs: &[PreferencePair],
        beta: f32,
        train_params: &[Tensor],
    ) -> Result<Vec<DpoStats>, TrainError> {
        let mut out = Vec::with_capacity(pairs.len());
        let mut rest = pairs;
        while !rest.is_empty() {
            let room = self
                .grad_accum_steps
                .saturating_sub(self.step_in_accum)
                .max(1);
            let (chunk, tail) = rest.split_at(room.min(rest.len()));
            out.extend(self.train_dpo_chunk(chunk, beta, train_params)?);
            rest = tail;
        }
        Ok(out)
    }

    fn train_dpo_chunk(
        &mut self,
        chunk: &[PreferencePair],
        beta: f32,
        train_params: &[Tensor],
    ) -> Result<Vec<DpoStats>, TrainError> {
        let seqs: Vec<_> = chunk
            .iter()
            .flat_map(|p| [&p.chosen, &p.rejected])
            .collect();
        let longest = seqs
            .iter()
            .map(|s| s.input_ids.len().min(s.target_ids.len()))
            .max()
            .unwrap_or(0);
        let padded_len = longest.div_ceil(self.length_bucket) * self.length_bucket;
        let mut inputs = Vec::with_capacity(seqs.len());
        let mut targets = Vec::with_capacity(seqs.len() * padded_len);
        let mut masks = Vec::with_capacity(seqs.len() * padded_len);
        for s in &seqs {
            let len = s.input_ids.len().min(s.target_ids.len());
            let pad = s.input_ids[..len].last().copied().unwrap_or(0);
            let mut row = s.input_ids[..len].to_vec();
            row.resize(padded_len, pad);
            inputs.push(row);
            targets.extend_from_slice(&s.target_ids[..len]);
            targets.resize(targets.len() + padded_len - len, pad);
            masks.extend((0..len).map(|i| s.loss_mask.get(i).copied().unwrap_or(0)));
            masks.resize(masks.len() + padded_len - len, 0);
        }
        let rows: Vec<&[u32]> = inputs.iter().map(Vec::as_slice).collect();
        let no_response = || TrainError("a preference pair has no response token to score".into());

        let reference: Vec<f64> = {
            let _tag = self.ctx.memory().scope(MemoryCategory::Activations);
            reference_log_probs(&self.model, &self.engine, &rows, &targets, &masks)
                .ok_or_else(no_response)?
                .into_iter()
                .map(|(log_prob, _)| log_prob)
                .collect()
        };

        let dropout_seed = Some(self.attention_dropout_seed());
        let batched = rows.len() > 1 && self.model.supports_batching();
        let logits: Vec<Tensor> = {
            let _tag = self.ctx.memory().scope(MemoryCategory::Activations);
            if batched {
                vec![self
                    .model
                    .forward_batch_with_dropout(&self.engine, &rows, dropout_seed)]
            } else {
                rows.iter()
                    .map(|row| {
                        self.model.forward_with_dropout(
                            &self.engine,
                            row,
                            0,
                            None,
                            None,
                            dropout_seed,
                        )
                    })
                    .collect()
            }
        };
        for l in &logits {
            memory::tag_loss(l);
        }
        let tag = self.ctx.memory().scope(MemoryCategory::Loss);
        let scored: Vec<ScoredRows<'_>> = if batched {
            vec![ScoredRows {
                logits: &logits[0],
                target_ids: &targets,
                loss_mask: &masks,
                sequences: rows.len(),
            }]
        } else {
            logits
                .iter()
                .enumerate()
                .map(|(i, l)| ScoredRows {
                    logits: l,
                    target_ids: &targets[i * padded_len..(i + 1) * padded_len],
                    loss_mask: &masks[i * padded_len..(i + 1) * padded_len],
                    sequences: 1,
                })
                .collect()
        };
        let loss = dpo_loss(&scored, &reference, beta, &self.engine).ok_or_else(no_response)?;

        // dpo_loss is the mean over this chunk; the window's update is the
        // mean over grad_accum_steps pairs.
        let mut seed = chunk.len() as f32 / self.grad_accum_steps as f32;
        if let Some(scaler) = self.scaler.as_ref() {
            seed *= scaler.scale_factor();
        }
        *loss.tensor.grad.borrow_mut() = Some(UnifiedTensor::new(&self.ctx, &[seed], &[1]));
        {
            let _tag = self.ctx.memory().scope(MemoryCategory::BackwardScratch);
            loss.tensor.backward(&self.engine);
            memory::tag_gradients(train_params);
            for l in &logits {
                memory::tag_loss(l);
            }
        }
        drop(tag);

        let mut out: Vec<DpoStats> = loss
            .pairs
            .iter()
            .map(|p| DpoStats {
                loss: p.loss.loss as f32,
                chosen_reward: p.loss.chosen_reward as f32,
                rejected_reward: p.loss.rejected_reward as f32,
                margin: p.loss.margin as f32,
                tokens: p.tokens,
                optimizer_step: false,
                grad_norm: None,
                lr: None,
            })
            .collect();

        self.step_in_accum += chunk.len();
        if self.step_in_accum >= self.grad_accum_steps {
            self.step_in_accum = 0;
            let mut step = StepStats {
                loss: 0.0,
                tokens: 0,
                optimizer_step: false,
                grad_norm: None,
                lr: None,
            };
            let applied = self.apply_gradients(train_params, &mut step);
            self.optimizer.zero_grad(train_params);
            applied?;
            if let Some(last) = out.last_mut() {
                last.optimizer_step = step.optimizer_step;
                last.grad_norm = step.grad_norm;
                last.lr = step.lr;
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::pair_loss;

    #[test]
    fn equal_policy_and_reference_give_log_2() {
        let l = pair_loss((-12.0, -9.0), (-12.0, -9.0), 0.1);
        assert_eq!(l.margin, 0.0);
        assert!((l.loss - std::f64::consts::LN_2).abs() < 1e-15, "{l:?}");
        // -beta * sigmoid(0)
        assert!((l.grad_chosen + 0.05).abs() < 1e-15, "{l:?}");
    }

    #[test]
    fn pair_loss_matches_the_formula() {
        // pi_c - ref_c = 2, pi_r - ref_r = -3, beta 0.5: margin 2.5.
        let l = pair_loss((-10.0, -15.0), (-12.0, -12.0), 0.5);
        assert!((l.chosen_reward - 1.0).abs() < 1e-15, "{l:?}");
        assert!((l.rejected_reward + 1.5).abs() < 1e-15, "{l:?}");
        assert!((l.margin - 2.5).abs() < 1e-15, "{l:?}");
        let sigmoid = 1.0 / (1.0 + (-2.5f64).exp());
        assert!((l.loss + sigmoid.ln()).abs() < 1e-14, "{l:?}");
        assert!(
            (l.grad_chosen + 0.5 * (1.0 - sigmoid)).abs() < 1e-14,
            "{l:?}"
        );
    }

    #[test]
    fn pair_loss_is_finite_for_large_margins() {
        let high = pair_loss((0.0, -1e5), (0.0, 0.0), 1.0);
        assert!(high.loss >= 0.0 && high.loss < 1e-30, "{high:?}");
        assert!(high.grad_chosen.abs() < 1e-30, "{high:?}");
        let low = pair_loss((-1e5, 0.0), (0.0, 0.0), 1.0);
        assert!((low.loss - 1e5).abs() < 1e-6, "{low:?}");
        assert!((low.grad_chosen + 1.0).abs() < 1e-15, "{low:?}");
    }
}
