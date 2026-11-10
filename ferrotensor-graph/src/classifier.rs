//! Sequence classification on top of a decoder: mean-pool the hidden states
//! of chosen blocks over each sequence's real tokens, normalise each pooled
//! vector, and map it to class logits with one linear layer per block.
//!
//! Summing one linear layer per tapped block is the same function as
//! concatenating the pooled vectors and applying a single linear layer, so
//! no concatenation op is needed. Everything is built from existing
//! differentiable ops (`matmul`, `rms_norm`, `add`), so gradients reach the
//! head and, through the hidden states, the LoRA adapters of the decoder.

use crate::lfm::Lfm2Model;
use ferrotensor_autograd::Tensor;
use ferrotensor_backend::{ComputeEngine, UnifiedTensor, WgpuContext};
use std::collections::HashMap;
use std::sync::Arc;

/// A frozen `[batch, batch * seq]` matrix that averages each row's first
/// `lengths[row]` positions: multiplied with hidden states flattened to
/// `[batch * seq, hidden]` it gives the masked mean `[batch, hidden]`.
///
/// Sequences are padded on the right, so the real tokens of row `r` are
/// positions `0..lengths[r]`. A row of length 0 pools to zeros.
///
/// # Panics
/// If `lengths.len() != batch` or a length exceeds `seq`.
pub fn mean_pool_matrix(
    ctx: &Arc<WgpuContext>,
    batch: usize,
    seq: usize,
    lengths: &[usize],
) -> Tensor {
    assert_eq!(lengths.len(), batch, "mean_pool_matrix: one length per row");
    let mut data = vec![0.0f32; batch * batch * seq];
    for (row, &len) in lengths.iter().enumerate() {
        assert!(len <= seq, "mean_pool_matrix: length {len} exceeds {seq}");
        if len == 0 {
            continue;
        }
        let start = row * batch * seq + row * seq;
        data[start..start + len].fill(1.0 / len as f32);
    }
    Tensor::frozen(
        ctx.clone(),
        UnifiedTensor::new(ctx, &data, &[batch, batch * seq]),
    )
}

/// Classification head over the hidden states of one or more decoder blocks.
///
/// For every tapped block: masked mean over the sequence, RMSNorm with a
/// learned scale, then a bias-free linear layer to `classes` logits. The
/// logits of all taps are summed.
pub struct ClassifierHead {
    /// Tapped blocks, 1-based: tap `l` is the output of block `l`, as in
    /// [`Lfm2Model::hidden_states_batch`].
    pub taps: Vec<usize>,
    /// RMSNorm scale `[hidden]` per tap.
    pub norms: Vec<Tensor>,
    /// Linear weight `[hidden, classes]` per tap.
    pub weights: Vec<Tensor>,
    /// Epsilon inside the RMSNorm of the pooled vectors.
    pub eps: f32,
}

impl ClassifierHead {
    /// A head for `classes` labels over `taps`. Norm scales start at one and
    /// the linear weights are drawn uniformly from `±1/sqrt(hidden)` with a
    /// deterministic generator, different per tap and per `seed`.
    ///
    /// # Panics
    /// If `taps` is empty or `hidden` or `classes` is zero.
    pub fn new_seeded(
        ctx: Arc<WgpuContext>,
        hidden: usize,
        classes: usize,
        taps: &[usize],
        seed: u64,
    ) -> Self {
        assert!(!taps.is_empty(), "ClassifierHead needs at least one tap");
        assert!(hidden > 0 && classes > 0, "ClassifierHead: empty shape");
        let bound = 1.0 / (hidden as f32).sqrt();
        let mut norms = Vec::with_capacity(taps.len());
        let mut weights = Vec::with_capacity(taps.len());
        for &tap in taps {
            let mut state = seed
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add((tap as u64) << 40 | (hidden as u64) << 16 | classes as u64)
                | 1;
            let w: Vec<f32> = (0..hidden * classes)
                .map(|_| {
                    // xorshift64*
                    state ^= state >> 12;
                    state ^= state << 25;
                    state ^= state >> 27;
                    let bits = state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40;
                    let unit = bits as f32 / (1u64 << 24) as f32;
                    (unit * 2.0 - 1.0) * bound
                })
                .collect();
            norms.push(Tensor::new(
                ctx.clone(),
                UnifiedTensor::new(&ctx, &vec![1.0; hidden], &[hidden]),
            ));
            weights.push(Tensor::new(
                ctx.clone(),
                UnifiedTensor::new(&ctx, &w, &[hidden, classes]),
            ));
        }
        Self {
            taps: taps.to_vec(),
            norms,
            weights,
            eps: 1e-5,
        }
    }

    /// A head from existing tensors, e.g. loaded from a checkpoint.
    ///
    /// # Panics
    /// If the three lists differ in length or are empty, a norm is not
    /// `[hidden]`, or a weight is not `[hidden, classes]` with the same
    /// `hidden` and `classes` for every tap.
    pub fn from_parts(taps: Vec<usize>, norms: Vec<Tensor>, weights: Vec<Tensor>) -> Self {
        assert!(!taps.is_empty(), "ClassifierHead needs at least one tap");
        assert!(
            taps.len() == norms.len() && taps.len() == weights.len(),
            "ClassifierHead: one norm and one weight per tap"
        );
        let shape = weights[0].data.shape.clone();
        assert_eq!(
            shape.len(),
            2,
            "ClassifierHead: weights are [hidden, classes]"
        );
        for (norm, weight) in norms.iter().zip(&weights) {
            assert_eq!(
                weight.data.shape, shape,
                "ClassifierHead: weight shapes differ"
            );
            assert_eq!(
                norm.data.shape,
                vec![shape[0]],
                "ClassifierHead: a norm scale is [hidden]"
            );
        }
        Self {
            taps,
            norms,
            weights,
            eps: 1e-5,
        }
    }

    /// Number of labels.
    pub fn classes(&self) -> usize {
        self.weights[0].data.shape[1]
    }

    /// The trainable tensors: every norm scale, then every linear weight.
    pub fn params(&self) -> Vec<Tensor> {
        self.norms.iter().chain(&self.weights).cloned().collect()
    }

    /// The trainable tensors by name: `classifier.norms.{tap}.weight` and
    /// `classifier.heads.{tap}.weight`.
    pub fn named_params(&self) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        for (i, tap) in self.taps.iter().enumerate() {
            map.insert(
                format!("classifier.norms.{tap}.weight"),
                self.norms[i].clone(),
            );
            map.insert(
                format!("classifier.heads.{tap}.weight"),
                self.weights[i].clone(),
            );
        }
        map
    }

    /// Class logits `[batch, classes]` from the tapped hidden states
    /// (`[batch, seq, hidden]` each, in the order of `taps`) and each row's
    /// number of real tokens.
    ///
    /// # Panics
    /// If `hidden_states` does not hold one tensor per tap, or
    /// `lengths` does not match the batch.
    pub fn logits(
        &self,
        engine: &ComputeEngine,
        hidden_states: &[Tensor],
        lengths: &[usize],
    ) -> Tensor {
        assert_eq!(
            hidden_states.len(),
            self.taps.len(),
            "ClassifierHead: one hidden state per tap"
        );
        let (batch, seq) = match hidden_states[0].data.shape.as_slice() {
            [batch, seq, _] => (*batch, *seq),
            shape => panic!("ClassifierHead expects [batch, seq, hidden], got {shape:?}"),
        };
        let pool = mean_pool_matrix(&hidden_states[0].ctx, batch, seq, lengths);
        let mut logits: Option<Tensor> = None;
        for (i, hidden) in hidden_states.iter().enumerate() {
            let flat = Tensor::reshape_2d(hidden);
            let pooled = Tensor::matmul(&pool, &flat, engine);
            let normed = Tensor::rms_norm(&pooled, &self.norms[i], self.eps, engine);
            let tap_logits = Tensor::matmul(&normed, &self.weights[i], engine);
            logits = Some(match logits {
                Some(sum) => Tensor::add(&sum, &tap_logits, engine),
                None => tap_logits,
            });
        }
        logits.expect("at least one tap")
    }

    /// Run `model` on equally long, right-padded `rows` up to the deepest
    /// tap and return the class logits `[batch, classes]`.
    ///
    /// # Panics
    /// As [`Lfm2Model::hidden_states_batch`] and [`logits`](Self::logits).
    pub fn forward(
        &self,
        engine: &ComputeEngine,
        model: &Lfm2Model,
        rows: &[&[u32]],
        lengths: &[usize],
    ) -> Tensor {
        let hidden_states = model.hidden_states_batch(engine, rows, &self.taps);
        self.logits(engine, &hidden_states, lengths)
    }
}
