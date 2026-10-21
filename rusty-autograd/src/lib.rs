//! Reverse-mode automatic differentiation over [`rusty_backend`] GPU tensors.
//!
//! A [`Tensor`] wraps a [`UnifiedTensor`] with a gradient slot and the
//! [`AutogradNode`] that produced it. Ops such as [`Tensor::matmul`] run a
//! GPU kernel and record a node; [`Tensor::backward`] walks the graph in
//! reverse topological order and accumulates gradients into every input.
//! The crate also provides [`AdamW`], learning-rate schedulers and
//! gradient [`checkpoint`]ing. This is the engine behind `rusty-graph` and
//! `rusty-trainer`.

#![warn(missing_docs)]
// `Tensor` shares `Arc<RefCell<..>>` gradient slots across clones; the graph
// is single-threaded, and the field types are public API used by
// rusty-graph and rusty-trainer.
#![allow(clippy::arc_with_non_send_sync)]

use rusty_backend::{AdamParams, AttentionOptions, ComputeEngine, UnifiedTensor, WgpuContext};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

/// A recorded op: maps the output gradient to one gradient per input.
pub trait AutogradNode {
    /// Gradients for each of [`inputs`](AutogradNode::inputs), in order.
    fn backward(&self, grad: &UnifiedTensor, engine: &ComputeEngine) -> Vec<UnifiedTensor>;
    /// The tensors this op consumed.
    fn inputs(&self) -> Vec<Tensor>;
}

/// A GPU tensor that records the ops applied to it for backpropagation.
///
/// Clones share data, gradient and creator.
#[derive(Clone)]
pub struct Tensor {
    /// The values (shared between clones).
    pub data: Arc<UnifiedTensor>,
    /// Accumulated gradient, if backward has reached this tensor.
    pub grad: Arc<RefCell<Option<UnifiedTensor>>>,
    /// GPU context the data lives in.
    pub ctx: Arc<WgpuContext>,
    /// The op that produced this tensor (`None` for leaves).
    pub creator: Option<Arc<dyn AutogradNode>>,
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.data.shape)
            .finish()
    }
}

impl Tensor {
    /// A leaf tensor holding `data`.
    pub fn new(ctx: Arc<WgpuContext>, data: UnifiedTensor) -> Self {
        Self {
            data: Arc::new(data),
            grad: Arc::new(RefCell::new(None)),
            ctx,
            creator: None,
        }
    }

    /// A tensor produced by the op `creator`.
    pub fn with_creator(
        ctx: Arc<WgpuContext>,
        data: UnifiedTensor,
        creator: Arc<dyn AutogradNode>,
    ) -> Tensor {
        Tensor {
            data: Arc::new(data),
            grad: Arc::new(RefCell::new(None)),
            ctx,
            creator: Some(creator),
        }
    }

    /// Backpropagate from this tensor (seeded with ones unless a gradient is
    /// already set) and accumulate gradients into every tensor in its graph.
    pub fn backward(&self, engine: &ComputeEngine) {
        let mut sorted_tensors = Vec::new();
        let mut visited = std::collections::HashSet::new();

        fn build_topo(
            tensor: &Tensor,
            sorted: &mut Vec<Tensor>,
            visited: &mut std::collections::HashSet<*const ()>,
        ) {
            let ptr = Arc::as_ptr(&tensor.data) as *const ();
            if visited.contains(&ptr) {
                return;
            }
            visited.insert(ptr);

            if let Some(creator) = &tensor.creator {
                for input in creator.inputs() {
                    build_topo(&input, sorted, visited);
                }
            }
            sorted.push(tensor.clone());
        }

        build_topo(self, &mut sorted_tensors, &mut visited);

        // Initial gradient = 1.0 if not already set
        if self.grad.borrow().is_none() {
            let shape = &self.data.shape;
            let size = shape.iter().product::<usize>();
            let ones = vec![1.0f32; size];
            *self.grad.borrow_mut() = Some(UnifiedTensor::new(&self.ctx, &ones, shape));
        }

        // Iterate backwards
        for tensor in sorted_tensors.into_iter().rev() {
            let grad_option = tensor.grad.borrow().clone();
            if let Some(grad) = grad_option {
                if let Some(creator) = &tensor.creator {
                    let input_grads = creator.backward(&grad, engine);
                    let inputs = creator.inputs();
                    for (input, input_grad) in inputs.iter().zip(input_grads) {
                        input.accumulate_grad(engine, &input_grad);
                    }
                }
            }
        }
    }

    /// Add `new_grad` to this tensor's gradient.
    pub fn accumulate_grad(&self, engine: &ComputeEngine, new_grad: &UnifiedTensor) {
        let mut grad_ref = self.grad.borrow_mut();
        if let Some(existing) = grad_ref.as_ref() {
            let res = UnifiedTensor::empty(&self.ctx, &self.data.shape);
            engine.add(&self.ctx, existing, new_grad, &res);
            *grad_ref = Some(res);
        } else {
            let res = UnifiedTensor::empty(&self.ctx, &self.data.shape);
            engine.copy_tensor(&self.ctx, new_grad, &res, 0);
            *grad_ref = Some(res);
        }
    }

    /// Clear this tensor's gradient.
    pub fn zero_grad(&self) {
        *self.grad.borrow_mut() = None;
    }

    // Recursive zero_grad for the whole graph
    /// Clear the gradients of every tensor in this tensor's graph.
    pub fn zero_grad_graph(&self) {
        let mut sorted_tensors = Vec::new();
        let mut visited = std::collections::HashSet::new();

        fn build_topo(
            tensor: &Tensor,
            sorted: &mut Vec<Tensor>,
            visited: &mut std::collections::HashSet<*const ()>,
        ) {
            let ptr = Arc::as_ptr(&tensor.data) as *const ();
            if visited.contains(&ptr) {
                return;
            }
            visited.insert(ptr);
            if let Some(creator) = &tensor.creator {
                for input in creator.inputs() {
                    build_topo(&input, sorted, visited);
                }
            }
            sorted.push(tensor.clone());
        }
        build_topo(self, &mut sorted_tensors, &mut visited);
        for t in sorted_tensors {
            t.zero_grad();
        }
    }
}

/// Backward node of [`Tensor::matmul`].
pub struct MatMulNode {
    /// Left operand `[M, K]`.
    pub a: Tensor,
    /// Right operand `[K, N]`.
    pub b: Tensor,
}
impl AutogradNode for MatMulNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.a.clone(), self.b.clone()]
    }
    /// Gradients for each of [`inputs`](AutogradNode::inputs), in order.
    fn backward(&self, grad: &UnifiedTensor, engine: &ComputeEngine) -> Vec<UnifiedTensor> {
        let grad_a = UnifiedTensor::empty(&self.a.ctx, &self.a.data.shape);
        engine.matmul_bt(&self.a.ctx, grad, &self.b.data, &grad_a);
        let grad_b = UnifiedTensor::empty(&self.b.ctx, &self.b.data.shape);
        engine.matmul_at(&self.a.ctx, &self.a.data, grad, &grad_b);
        vec![grad_a, grad_b]
    }
}

/// Backward node of [`Tensor::add`].
pub struct AddNode {
    /// Left operand.
    pub a: Tensor,
    /// Right operand.
    pub b: Tensor,
}
impl AutogradNode for AddNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.a.clone(), self.b.clone()]
    }
    /// Gradients for each of [`inputs`](AutogradNode::inputs), in order.
    fn backward(&self, grad: &UnifiedTensor, engine: &ComputeEngine) -> Vec<UnifiedTensor> {
        let grad_a = UnifiedTensor::empty(&self.a.ctx, &self.a.data.shape);
        let grad_b = UnifiedTensor::empty(&self.b.ctx, &self.b.data.shape);
        engine.add_backward(&self.a.ctx, grad, &grad_a, &grad_b);
        vec![grad_a, grad_b]
    }
}

/// Backward node of [`Tensor::mul`].
pub struct MulNode {
    /// Left operand.
    pub a: Tensor,
    /// Right operand.
    pub b: Tensor,
}
impl AutogradNode for MulNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.a.clone(), self.b.clone()]
    }
    /// Gradients for each of [`inputs`](AutogradNode::inputs), in order.
    fn backward(&self, grad: &UnifiedTensor, engine: &ComputeEngine) -> Vec<UnifiedTensor> {
        let grad_a = UnifiedTensor::empty(&self.a.ctx, &self.a.data.shape);
        let grad_b = UnifiedTensor::empty(&self.b.ctx, &self.b.data.shape);
        engine.mul_backward(
            &self.a.ctx,
            grad,
            &self.a.data,
            &self.b.data,
            &grad_a,
            &grad_b,
        );
        vec![grad_a, grad_b]
    }
}

/// Backward node of [`Tensor::sub`].
pub struct SubNode {
    /// Left operand.
    pub a: Tensor,
    /// Right operand.
    pub b: Tensor,
}
impl AutogradNode for SubNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.a.clone(), self.b.clone()]
    }
    /// Gradients for each of [`inputs`](AutogradNode::inputs), in order.
    fn backward(&self, grad: &UnifiedTensor, engine: &ComputeEngine) -> Vec<UnifiedTensor> {
        let grad_a = UnifiedTensor::empty(&self.a.ctx, &self.a.data.shape);
        let grad_b = UnifiedTensor::empty(&self.b.ctx, &self.b.data.shape);
        engine.sub_backward(&self.a.ctx, grad, &grad_a, &grad_b);
        vec![grad_a, grad_b]
    }
}

/// Backward node of [`Tensor::silu`].
pub struct SiLUNode {
    /// Input of the activation.
    pub input: Tensor,
}
impl AutogradNode for SiLUNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }
    /// Gradients for each of [`inputs`](AutogradNode::inputs), in order.
    fn backward(&self, grad: &UnifiedTensor, engine: &ComputeEngine) -> Vec<UnifiedTensor> {
        let grad_in = UnifiedTensor::empty(&self.input.ctx, &self.input.data.shape);
        engine.silu_backward(&self.input.ctx, grad, &self.input.data, &grad_in);
        vec![grad_in]
    }
}

/// Backward node of [`Tensor::relu`].
pub struct ReluNode {
    /// Input of the activation.
    pub input: Tensor,
}
impl AutogradNode for ReluNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }
    /// Gradients for each of [`inputs`](AutogradNode::inputs), in order.
    fn backward(&self, grad: &UnifiedTensor, engine: &ComputeEngine) -> Vec<UnifiedTensor> {
        let grad_in = UnifiedTensor::empty(&self.input.ctx, &self.input.data.shape);
        engine.relu_backward(&self.input.ctx, grad, &self.input.data, &grad_in);
        vec![grad_in]
    }
}

/// Backward node of [`Tensor::softmax`] (row-wise over the last dim).
pub struct SoftmaxNode {
    /// Input logits.
    pub input: Tensor,
    /// Softmax output (reused by the backward pass).
    pub output: Tensor,
}
impl AutogradNode for SoftmaxNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }
    /// Gradients for each of [`inputs`](AutogradNode::inputs), in order.
    fn backward(&self, grad: &UnifiedTensor, engine: &ComputeEngine) -> Vec<UnifiedTensor> {
        let grad_in = UnifiedTensor::empty(&self.input.ctx, &self.input.data.shape);
        engine.softmax_backward(&self.input.ctx, grad, &self.output.data, &grad_in);
        vec![grad_in]
    }
}

/// Backward node of [`Tensor::reshape`].
pub struct ReshapeNode {
    /// Tensor before the reshape.
    pub input: Tensor,
    /// Shape to restore for the gradient.
    pub original_shape: Vec<usize>,
}
impl AutogradNode for ReshapeNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }
    /// Gradients for each of [`inputs`](AutogradNode::inputs), in order.
    fn backward(&self, grad: &UnifiedTensor, _engine: &ComputeEngine) -> Vec<UnifiedTensor> {
        vec![grad.reshaped(&self.original_shape)]
    }
}

/// Rotate `data` (`[batch, seq, hidden]` or `[seq, hidden]`, heads of
/// `head_dim` packed in `hidden`) by the RoPE angles of positions
/// `pos..pos + seq`; `inverse` rotates back (the transpose, used for the
/// gradient).
pub fn apply_rope_in_place(
    data: &mut [f32],
    shape: &[usize],
    head_dim: usize,
    pos: usize,
    inverse: bool,
) {
    let (batch, seq_len, hidden) = match shape {
        [batch, seq_len, hidden] => (*batch, *seq_len, *hidden),
        [seq_len, hidden] => (1, *seq_len, *hidden),
        other => panic!("RoPE expects 2D or 3D tensor, got {other:?}"),
    };
    let half_dim = head_dim / 2;
    let num_heads = hidden / head_dim;
    let sign = if inverse { -1.0 } else { 1.0 };

    for batch_idx in 0..batch {
        for token_idx in 0..seq_len {
            let token_pos = (pos + token_idx) as f32;
            let token_base = (batch_idx * seq_len + token_idx) * hidden;
            for head_idx in 0..num_heads {
                let head_base = token_base + head_idx * head_dim;
                for dim_idx in 0..half_dim {
                    let freq_exp = -2.0 * dim_idx as f32 / head_dim as f32;
                    let angle = sign * token_pos * 10000.0_f32.powf(freq_exp);
                    let cos_a = angle.cos();
                    let sin_a = angle.sin();
                    let idx1 = head_base + dim_idx;
                    let idx2 = head_base + dim_idx + half_dim;
                    let v1 = data[idx1];
                    let v2 = data[idx2];
                    data[idx1] = v1 * cos_a - v2 * sin_a;
                    data[idx2] = v1 * sin_a + v2 * cos_a;
                }
            }
        }
    }
}

/// Backward node of [`Tensor::rope`].
pub struct RoPENode {
    /// Input before rotation.
    pub input: Tensor,
    /// Size of each attention head.
    pub head_dim: usize,
    /// Position of the first token.
    pub pos: usize,
}
impl AutogradNode for RoPENode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }
    /// Gradients for each of [`inputs`](AutogradNode::inputs), in order.
    fn backward(&self, grad: &UnifiedTensor, _engine: &ComputeEngine) -> Vec<UnifiedTensor> {
        let mut grad_cpu = pollster::block_on(grad.to_vec(&self.input.ctx));
        apply_rope_in_place(
            &mut grad_cpu,
            &self.input.data.shape,
            self.head_dim,
            self.pos,
            true,
        );
        vec![UnifiedTensor::new(
            &self.input.ctx,
            &grad_cpu,
            &self.input.data.shape,
        )]
    }
}

/// Gradients of `y = x * w / sqrt(mean(x^2) + eps)` (per row of `hidden`
/// elements) with respect to `x` and `w`, given the output gradient `g`.
pub fn rms_norm_backward_cpu(
    x: &[f32],
    g: &[f32],
    w: &[f32],
    eps: f32,
    hidden: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut grad_x = vec![0.0f32; x.len()];
    let mut grad_w = vec![0.0f32; w.len()];
    for ((x_row, g_row), gx_row) in x
        .chunks(hidden)
        .zip(g.chunks(hidden))
        .zip(grad_x.chunks_mut(hidden))
    {
        let mean_sq = x_row.iter().map(|v| v * v).sum::<f32>() / hidden as f32;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();
        let mut dot = 0.0f32;
        for idx in 0..hidden {
            dot += g_row[idx] * w[idx] * x_row[idx];
            grad_w[idx] += g_row[idx] * x_row[idx] * inv_rms;
        }
        for idx in 0..hidden {
            gx_row[idx] =
                g_row[idx] * w[idx] * inv_rms - x_row[idx] * dot * inv_rms.powi(3) / hidden as f32;
        }
    }
    (grad_x, grad_w)
}

/// Backward node of [`Tensor::rms_norm`].
pub struct RMSNormNode {
    /// Normalized input.
    pub input: Tensor,
    /// Per-feature scale.
    pub weight: Tensor,
    /// Epsilon inside the square root.
    pub eps: f32,
}
impl AutogradNode for RMSNormNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.input.clone(), self.weight.clone()]
    }
    /// Gradients for each of [`inputs`](AutogradNode::inputs), in order.
    fn backward(&self, grad: &UnifiedTensor, _engine: &ComputeEngine) -> Vec<UnifiedTensor> {
        let input_shape = self.input.data.shape.clone();
        let hidden = *input_shape.last().expect("RMSNorm input must be non-empty");

        let x = pollster::block_on(self.input.data.to_vec(&self.input.ctx));
        let g = pollster::block_on(grad.to_vec(&self.input.ctx));
        let w = pollster::block_on(self.weight.data.to_vec(&self.weight.ctx));

        let (grad_x, grad_w) = rms_norm_backward_cpu(&x, &g, &w, self.eps, hidden);

        vec![
            UnifiedTensor::new(&self.input.ctx, &grad_x, &input_shape),
            UnifiedTensor::new(&self.weight.ctx, &grad_w, &self.weight.data.shape),
        ]
    }
}

/// Backward node of [`Tensor::select_token`].
pub struct SelectTokenNode {
    /// Logits `[seq, vocab]`.
    pub input: Tensor,
    /// Selected row.
    pub token_index: usize,
    /// Row length.
    pub vocab_size: usize,
}
impl AutogradNode for SelectTokenNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }
    /// Gradients for each of [`inputs`](AutogradNode::inputs), in order.
    fn backward(&self, grad: &UnifiedTensor, _engine: &ComputeEngine) -> Vec<UnifiedTensor> {
        let mut full_grad = vec![0.0f32; self.input.data.shape.iter().product()];
        let token_grad = pollster::block_on(grad.to_vec(&self.input.ctx));
        let start = self.token_index * self.vocab_size;
        let end = start + self.vocab_size;
        full_grad[start..end].copy_from_slice(&token_grad);
        vec![UnifiedTensor::new(
            &self.input.ctx,
            &full_grad,
            &self.input.data.shape,
        )]
    }
}

/// Backward node of [`Tensor::mse_loss`].
pub struct MSELossNode {
    /// Predictions.
    pub pred: Tensor,
    /// Targets.
    pub target: Tensor,
}
impl AutogradNode for MSELossNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.pred.clone(), self.target.clone()]
    }
    /// Gradients for each of [`inputs`](AutogradNode::inputs), in order.
    fn backward(&self, grad: &UnifiedTensor, engine: &ComputeEngine) -> Vec<UnifiedTensor> {
        let grad_in = UnifiedTensor::empty(&self.pred.ctx, &self.pred.data.shape);
        engine.mse_loss_backward(&self.pred.ctx, &self.pred.data, &self.target.data, &grad_in);
        let upstream = pollster::block_on(grad.to_vec(&self.pred.ctx))
            .first()
            .copied()
            .unwrap_or(1.0);
        if (upstream - 1.0).abs() > f32::EPSILON {
            engine.gradient_scale(&self.pred.ctx, &grad_in, upstream);
        }
        // d/d(target) of 0.5 * (pred - target)^2 is the negated pred gradient.
        let grad_target = UnifiedTensor::empty(&self.target.ctx, &self.target.data.shape);
        engine.copy_tensor(&self.target.ctx, &grad_in, &grad_target, 0);
        engine.gradient_scale(&self.target.ctx, &grad_target, -1.0);
        vec![grad_in, grad_target]
    }
}

/// Cross-entropy loss node for LLM training
/// Forward: `-log(softmax(logits)[target])`
/// Backward: `softmax(logits) - one_hot(target)`
pub struct CrossEntropyLossNode {
    /// Logits `[vocab]`.
    pub logits: Tensor,
    /// Target class.
    pub target_idx: u32,
}
impl AutogradNode for CrossEntropyLossNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.logits.clone()]
    }
    /// Gradients for each of [`inputs`](AutogradNode::inputs), in order.
    fn backward(&self, grad: &UnifiedTensor, engine: &ComputeEngine) -> Vec<UnifiedTensor> {
        let grad_logits = UnifiedTensor::empty(&self.logits.ctx, &self.logits.data.shape);
        engine.cross_entropy_backward(
            &self.logits.ctx,
            &self.logits.data,
            self.target_idx,
            &grad_logits,
        );
        let upstream = pollster::block_on(grad.to_vec(&self.logits.ctx))
            .first()
            .copied()
            .unwrap_or(1.0);
        if (upstream - 1.0).abs() > f32::EPSILON {
            engine.gradient_scale(&self.logits.ctx, &grad_logits, upstream);
        }
        vec![grad_logits]
    }
}

/// Fused SiLU-Gate node for Llama MLP
/// Forward: out = SiLU(up) * gate
/// Reduces memory bandwidth by 2x compared to separate ops
pub struct SiLUGateNode {
    /// Input passed through SiLU.
    pub up: Tensor,
    /// Multiplicative gate.
    pub gate: Tensor,
}
impl AutogradNode for SiLUGateNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.up.clone(), self.gate.clone()]
    }
    /// Gradients for each of [`inputs`](AutogradNode::inputs), in order.
    fn backward(&self, grad: &UnifiedTensor, engine: &ComputeEngine) -> Vec<UnifiedTensor> {
        let grad_up = UnifiedTensor::empty(&self.up.ctx, &self.up.data.shape);
        let grad_gate = UnifiedTensor::empty(&self.gate.ctx, &self.gate.data.shape);
        engine.silu_gate_backward(
            &self.up.ctx,
            &self.up.data,
            &self.gate.data,
            grad,
            &grad_up,
            &grad_gate,
        );
        vec![grad_up, grad_gate]
    }
}

// --- Gradient Checkpointing ---
// Saves memory by recomputing activations during backward pass instead of storing them.
// Key for training large models on limited memory devices.

/// A forward computation that [`checkpoint`] can replay during backward.
pub trait CheckpointFn: Send + Sync {
    /// Run the forward computation on `inputs`.
    fn forward(&self, inputs: &[Tensor], engine: &ComputeEngine) -> Vec<Tensor>;
}

impl<F> CheckpointFn for F
where
    F: Fn(&[Tensor], &ComputeEngine) -> Vec<Tensor> + Send + Sync,
{
    fn forward(&self, inputs: &[Tensor], engine: &ComputeEngine) -> Vec<Tensor> {
        self(inputs, engine)
    }
}

/// Backward node of one output of a checkpointed computation: drops the
/// forward activations and recomputes them when gradients are needed.
pub struct CheckpointedNode {
    /// Inputs of the checkpointed computation.
    pub inputs: Vec<Tensor>,
    /// The computation to replay.
    pub checkpoint_fn: Arc<dyn CheckpointFn>,
    /// Which of the computation's outputs this node produced.
    pub output_index: usize,
}

impl AutogradNode for CheckpointedNode {
    fn inputs(&self) -> Vec<Tensor> {
        self.inputs.clone()
    }

    /// Gradients for each of [`inputs`](AutogradNode::inputs), in order.
    fn backward(&self, grad: &UnifiedTensor, engine: &ComputeEngine) -> Vec<UnifiedTensor> {
        // Replay the forward pass from fresh leaves that share the input
        // data, backpropagate through the recomputed graph, and read the
        // gradients that reach the leaves.
        let leaves: Vec<Tensor> = self
            .inputs
            .iter()
            .map(|t| Tensor::new(t.ctx.clone(), t.data.as_ref().clone()))
            .collect();
        let outputs = self.checkpoint_fn.forward(&leaves, engine);
        let output = &outputs[self.output_index];
        output.accumulate_grad(engine, grad);
        output.backward(engine);
        leaves
            .iter()
            .map(|leaf| {
                leaf.grad
                    .borrow()
                    .clone()
                    .unwrap_or_else(|| UnifiedTensor::zeros(&leaf.ctx, &leaf.data.shape))
            })
            .collect()
    }
}

/// Gradient checkpointing: run `forward_fn` now but keep only its outputs;
/// the intermediate activations are recomputed during backward, trading
/// compute for memory.
pub fn checkpoint<F>(inputs: Vec<Tensor>, forward_fn: F, engine: &ComputeEngine) -> Vec<Tensor>
where
    F: Fn(&[Tensor], &ComputeEngine) -> Vec<Tensor> + Send + Sync + 'static,
{
    let outputs = forward_fn(&inputs, engine);
    let checkpoint_fn: Arc<dyn CheckpointFn> = Arc::new(forward_fn);
    outputs
        .into_iter()
        .enumerate()
        .map(|(output_index, output)| {
            let node = Arc::new(CheckpointedNode {
                inputs: inputs.clone(),
                checkpoint_fn: checkpoint_fn.clone(),
                output_index,
            });
            // The output's own graph is dropped here; only its data is kept.
            Tensor::with_creator(output.ctx.clone(), output.data.as_ref().clone(), node)
        })
        .collect()
}

/// [`checkpoint`] for a single-input, single-output function.
///
/// Checkpointing each transformer block keeps one activation per block
/// instead of every intermediate:
/// ```rust,ignore
/// let mut x = input;
/// for layer in &layers {
///     x = checkpoint_single(x, move |h, eng| layer.forward(h, eng), engine);
/// }
/// ```
pub fn checkpoint_single<F>(input: Tensor, forward_fn: F, engine: &ComputeEngine) -> Tensor
where
    F: Fn(&Tensor, &ComputeEngine) -> Tensor + Send + Sync + 'static,
{
    let wrapper = move |inputs: &[Tensor], eng: &ComputeEngine| -> Vec<Tensor> {
        vec![forward_fn(&inputs[0], eng)]
    };
    checkpoint(vec![input], wrapper, engine)
        .pop()
        .expect("checkpointed function returned one output")
}

impl Tensor {
    /// Reshape (same element count); the gradient is reshaped back.
    pub fn reshape(input: &Tensor, new_shape: &[usize]) -> Tensor {
        let out_data = input.data.reshaped(new_shape);
        let node = Arc::new(ReshapeNode {
            input: input.clone(),
            original_shape: input.data.shape.clone(),
        });
        Tensor::with_creator(input.ctx.clone(), out_data, node)
    }

    /// Flatten `[batch, seq, hidden]` to `[batch * seq, hidden]` (2-D inputs pass through).
    pub fn reshape_2d(input: &Tensor) -> Tensor {
        match input.data.shape.as_slice() {
            [_, _] => input.clone(),
            [batch, seq, hidden] => Tensor::reshape(input, &[*batch * *seq, *hidden]),
            shape => panic!("reshape_2d expects 2D or 3D tensor, got {shape:?}"),
        }
    }

    /// Reshape `[batch * seq, hidden]` back to `[batch, seq, hidden]`.
    pub fn unflatten_last(input: &Tensor, batch: usize, seq: usize) -> Tensor {
        let hidden = *input
            .data
            .shape
            .last()
            .expect("unflatten_last requires a non-empty shape");
        Tensor::reshape(input, &[batch, seq, hidden])
    }

    /// Matrix product of `[M, K]` and `[K, N]`.
    pub fn matmul(a: &Tensor, b: &Tensor, engine: &ComputeEngine) -> Tensor {
        let out_shape = vec![a.data.shape[0], b.data.shape[1]];
        let out_data = UnifiedTensor::empty(&a.ctx, &out_shape);
        engine.matmul(&a.ctx, &a.data, &b.data, &out_data);
        let node = Arc::new(MatMulNode {
            a: a.clone(),
            b: b.clone(),
        });
        Tensor::with_creator(a.ctx.clone(), out_data, node)
    }
    /// Element-wise sum of equally shaped tensors.
    pub fn add(a: &Tensor, b: &Tensor, engine: &ComputeEngine) -> Tensor {
        let out_data = UnifiedTensor::empty(&a.ctx, &a.data.shape);
        engine.add(&a.ctx, &a.data, &b.data, &out_data);
        let node = Arc::new(AddNode {
            a: a.clone(),
            b: b.clone(),
        });
        Tensor::with_creator(a.ctx.clone(), out_data, node)
    }
    /// Element-wise product of equally shaped tensors.
    pub fn mul(a: &Tensor, b: &Tensor, engine: &ComputeEngine) -> Tensor {
        let out_data = UnifiedTensor::empty(&a.ctx, &a.data.shape);
        engine.mul(&a.ctx, &a.data, &b.data, &out_data);
        let node = Arc::new(MulNode {
            a: a.clone(),
            b: b.clone(),
        });
        Tensor::with_creator(a.ctx.clone(), out_data, node)
    }
    /// Element-wise difference of equally shaped tensors.
    pub fn sub(a: &Tensor, b: &Tensor, engine: &ComputeEngine) -> Tensor {
        let out_data = UnifiedTensor::empty(&a.ctx, &a.data.shape);
        engine.sub(&a.ctx, &a.data, &b.data, &out_data);
        let node = Arc::new(SubNode {
            a: a.clone(),
            b: b.clone(),
        });
        Tensor::with_creator(a.ctx.clone(), out_data, node)
    }
    /// SiLU activation `x * sigmoid(x)`.
    pub fn silu(input: &Tensor, engine: &ComputeEngine) -> Tensor {
        let out_data = UnifiedTensor::empty(&input.ctx, &input.data.shape);
        engine.silu(&input.ctx, &input.data, &out_data);
        let node = Arc::new(SiLUNode {
            input: input.clone(),
        });
        Tensor::with_creator(input.ctx.clone(), out_data, node)
    }
    /// Squared-error loss `0.5 * sum((pred - target)^2)`, shape `[1]`.
    ///
    /// Its gradient is `pred - target` (the backward kernel's output).
    pub fn mse_loss(pred: &Tensor, target: &Tensor, engine: &ComputeEngine) -> Tensor {
        // The kernel writes 0.5 * (p - t)^2 per element; sum on the host.
        let per_element = UnifiedTensor::empty(&pred.ctx, &pred.data.shape);
        engine.mse_loss(&pred.ctx, &pred.data, &target.data, &per_element);
        let total: f32 = pollster::block_on(per_element.to_vec(&pred.ctx))
            .iter()
            .sum();
        let out_data = UnifiedTensor::new(&pred.ctx, &[total], &[1]);
        let node = Arc::new(MSELossNode {
            pred: pred.clone(),
            target: target.clone(),
        });
        Tensor::with_creator(pred.ctx.clone(), out_data, node)
    }

    /// ReLU activation `max(0, x)`.
    pub fn relu(input: &Tensor, engine: &ComputeEngine) -> Tensor {
        let out_data = UnifiedTensor::empty(&input.ctx, &input.data.shape);
        engine.relu(&input.ctx, &input.data, &out_data);
        let node = Arc::new(ReluNode {
            input: input.clone(),
        });
        Tensor::with_creator(input.ctx.clone(), out_data, node)
    }

    /// Softmax over the last dimension.
    pub fn softmax(input: &Tensor, engine: &ComputeEngine) -> Tensor {
        let out_data = UnifiedTensor::empty(&input.ctx, &input.data.shape);
        engine.softmax(&input.ctx, &input.data, &out_data);
        let out_tensor = Tensor::new(input.ctx.clone(), out_data);
        let node = Arc::new(SoftmaxNode {
            input: input.clone(),
            output: out_tensor.clone(),
        });
        Tensor::with_creator(input.ctx.clone(), out_tensor.data.as_ref().clone(), node)
    }

    /// Rotary position embedding for tokens at positions `pos..pos + seq`
    /// (`[batch, seq, hidden]` or `[seq, hidden]`).
    pub fn rope(input: &Tensor, head_dim: usize, pos: usize, engine: &ComputeEngine) -> Tensor {
        let out_data = UnifiedTensor::empty(&input.ctx, &input.data.shape);
        engine.copy_tensor(&input.ctx, &input.data, &out_data, 0);
        engine.rope(&input.ctx, &out_data, head_dim, pos);
        let node = Arc::new(RoPENode {
            input: input.clone(),
            head_dim,
            pos,
        });
        Tensor::with_creator(input.ctx.clone(), out_data, node)
    }

    /// RMS normalization over the last dimension, scaled by `weight`.
    pub fn rms_norm(input: &Tensor, weight: &Tensor, eps: f32, engine: &ComputeEngine) -> Tensor {
        let out_data = UnifiedTensor::empty(&input.ctx, &input.data.shape);
        engine.rms_norm_eps(&input.ctx, &input.data, &weight.data, &out_data, eps);
        let node = Arc::new(RMSNormNode {
            input: input.clone(),
            weight: weight.clone(),
            eps,
        });
        Tensor::with_creator(input.ctx.clone(), out_data, node)
    }

    /// Row `token_index` of `[seq, vocab]` logits.
    pub fn select_token(input: &Tensor, token_index: usize) -> Tensor {
        let vocab_size = *input
            .data
            .shape
            .last()
            .expect("select_token requires a non-empty tensor");
        let full = pollster::block_on(input.data.to_vec(&input.ctx));
        let start = token_index * vocab_size;
        let end = start + vocab_size;
        let out_data = UnifiedTensor::new(&input.ctx, &full[start..end], &[vocab_size]);
        let node = Arc::new(SelectTokenNode {
            input: input.clone(),
            token_index,
            vocab_size,
        });
        Tensor::with_creator(input.ctx.clone(), out_data, node)
    }

    /// Cross-entropy loss for language model training
    /// Computes `-log(softmax(logits)[target])`
    pub fn cross_entropy_loss(logits: &Tensor, target: u32, engine: &ComputeEngine) -> Tensor {
        let out_data = UnifiedTensor::empty(&logits.ctx, &[1]);
        engine.cross_entropy_loss(&logits.ctx, &logits.data, target, &out_data);
        let node = Arc::new(CrossEntropyLossNode {
            logits: logits.clone(),
            target_idx: target,
        });
        Tensor::with_creator(logits.ctx.clone(), out_data, node)
    }

    /// Fused SiLU-Gate: out = SiLU(up) * gate
    /// Used in Llama MLP, reduces memory bandwidth by 2x
    pub fn silu_gate_fused(up: &Tensor, gate: &Tensor, engine: &ComputeEngine) -> Tensor {
        let out_data = UnifiedTensor::empty(&up.ctx, &up.data.shape);
        engine.silu_gate_fused(&up.ctx, &up.data, &gate.data, &out_data);
        let node = Arc::new(SiLUGateNode {
            up: up.clone(),
            gate: gate.clone(),
        });
        Tensor::with_creator(up.ctx.clone(), out_data, node)
    }

    /// Flash Attention V2 (fused) over `[batch, seq, heads, head_dim]`.
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attention(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        causal: bool,
        dropout_prob: f32,
        seed: u64,
        engine: &ComputeEngine,
    ) -> Tensor {
        // Output shape same as Query
        let out_data = UnifiedTensor::empty(&q.ctx, &q.data.shape);
        let mask_data = mask.map(|m| &*m.data);

        engine.flash_attention_v2(
            &q.ctx,
            &q.data,
            &k.data,
            &v.data,
            &out_data,
            causal,
            dropout_prob,
            seed,
            mask_data,
        );

        let node = Arc::new(FlashAttentionNode {
            q: q.clone(),
            k: k.clone(),
            v: v.clone(),
            mask: mask.cloned(),
            causal,
            dropout_prob,
            seed,
        });

        Tensor::with_creator(q.ctx.clone(), out_data, node)
    }
}

/// Backward node of [`Tensor::flash_attention`].
pub struct FlashAttentionNode {
    /// Queries.
    pub q: Tensor,
    /// Keys.
    pub k: Tensor,
    /// Values.
    pub v: Tensor,
    /// Optional additive mask.
    pub mask: Option<Tensor>,
    /// Whether future positions are masked.
    pub causal: bool,
    /// Attention dropout probability.
    pub dropout_prob: f32,
    /// Dropout RNG seed.
    pub seed: u64,
}

impl AutogradNode for FlashAttentionNode {
    fn inputs(&self) -> Vec<Tensor> {
        let mut inputs = vec![self.q.clone(), self.k.clone(), self.v.clone()];
        if let Some(m) = &self.mask {
            inputs.push(m.clone());
        }
        inputs
    }

    /// Gradients for each of [`inputs`](AutogradNode::inputs), in order.
    fn backward(&self, grad: &UnifiedTensor, engine: &ComputeEngine) -> Vec<UnifiedTensor> {
        let grad_q = UnifiedTensor::empty(&self.q.ctx, &self.q.data.shape);
        let grad_k = UnifiedTensor::empty(&self.k.ctx, &self.k.data.shape);
        let grad_v = UnifiedTensor::empty(&self.v.ctx, &self.v.data.shape);

        let out_recomputed = UnifiedTensor::empty(&self.q.ctx, &self.q.data.shape);
        let mask_data = self.mask.as_ref().map(|m| &*m.data);
        engine.flash_attention_v2(
            &self.q.ctx,
            &self.q.data,
            &self.k.data,
            &self.v.data,
            &out_recomputed,
            self.causal,
            self.dropout_prob,
            self.seed,
            mask_data,
        );

        let opts = AttentionOptions {
            causal: self.causal,
            dropout_prob: self.dropout_prob,
            seed: self.seed,
            ..Default::default()
        };
        engine.flash_attention_v2_backward_with(
            &self.q.ctx,
            &self.q.data,
            &self.k.data,
            &self.v.data,
            &out_recomputed,
            grad,
            &grad_q,
            &grad_k,
            &grad_v,
            &opts,
            mask_data,
        );

        let mut grads = vec![grad_q, grad_k, grad_v];
        if let Some(m) = &self.mask {
            // Mask gradient is usually not needed (if mask is constant/padding)
            // But if mask is learned (e.g. bias), we need it.
            // Current kernel doesn't support mask grad. Return zeros.
            grads.push(UnifiedTensor::zeros(&m.ctx, &m.data.shape));
        }
        grads
    }
}

/// An optimizer that updates parameters from their accumulated gradients.
pub trait Optimizer: Send + Sync {
    /// Update `params` from their gradients.
    fn step(&mut self, params: &[Tensor]);
    /// Clear the gradients of `params`.
    fn zero_grad(&mut self, params: &[Tensor]);
    /// Serialize optimizer state keyed by parameter name (`None` if unsupported).
    fn save_state(
        &self,
        _ctx: &Arc<WgpuContext>,
        _named_params: &HashMap<String, Tensor>,
    ) -> Option<OptimizerCheckpoint> {
        None
    }
    /// Restore state saved by [`save_state`](Optimizer::save_state);
    /// returns whether it was applied.
    fn load_state(
        &mut self,
        _ctx: &Arc<WgpuContext>,
        _named_params: &HashMap<String, Tensor>,
        _state: &OptimizerCheckpoint,
    ) -> bool {
        false
    }
}

/// One optimizer state tensor, serialized.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct OptimizerTensorState {
    /// Tensor shape.
    pub shape: Vec<usize>,
    /// Row-major values.
    pub data: Vec<f32>,
}

/// Serialized [`AdamW`] state for resuming training.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AdamWCheckpoint {
    /// Learning rate.
    pub lr: f32,
    /// First-moment decay.
    pub beta1: f32,
    /// Second-moment decay.
    pub beta2: f32,
    /// Denominator epsilon.
    pub eps: f32,
    /// Decoupled weight decay.
    pub weight_decay: f32,
    /// Steps taken.
    pub t: f32,
    /// First moments by parameter name.
    pub m_states: HashMap<String, OptimizerTensorState>,
    /// Second moments by parameter name.
    pub v_states: HashMap<String, OptimizerTensorState>,
}

/// Serialized optimizer state.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum OptimizerCheckpoint {
    /// State of an [`AdamW`] optimizer.
    AdamW(AdamWCheckpoint),
}

/// AdamW optimizer running its update as one GPU kernel per parameter.
pub struct AdamW {
    /// Engine that runs the update kernel.
    pub engine: Arc<ComputeEngine>,
    /// Learning rate.
    pub lr: f32,
    /// First-moment decay.
    pub beta1: f32,
    /// Second-moment decay.
    pub beta2: f32,
    /// Denominator epsilon.
    pub eps: f32,
    /// Decoupled weight decay.
    pub weight_decay: f32,
    /// Steps taken.
    pub t: f32,
    m_states: HashMap<usize, UnifiedTensor>,
    v_states: HashMap<usize, UnifiedTensor>,
}

impl AdamW {
    /// AdamW with betas (0.9, 0.999), eps 1e-8 and weight decay 0.01.
    pub fn new(engine: Arc<ComputeEngine>, lr: f32) -> Self {
        Self {
            engine,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            t: 0.0,
            m_states: HashMap::new(),
            v_states: HashMap::new(),
        }
    }

    /// Take one optimization step.
    pub fn step(&mut self, params: &[Tensor]) {
        self.apply_step(params);
    }

    fn apply_step(&mut self, params: &[Tensor]) {
        self.t += 1.0;
        let p = AdamParams {
            lr: self.lr,
            beta1: self.beta1,
            beta2: self.beta2,
            eps: self.eps,
            weight_decay: self.weight_decay,
            t: self.t,
            _pad: [0.0; 2],
        };

        for param in params {
            let grad_borrow = param.grad.borrow();
            if let Some(grad) = grad_borrow.as_ref() {
                let key = Arc::as_ptr(&param.data) as usize;

                let m = self
                    .m_states
                    .entry(key)
                    .or_insert_with(|| UnifiedTensor::zeros(&param.ctx, &param.data.shape));
                let v = self
                    .v_states
                    .entry(key)
                    .or_insert_with(|| UnifiedTensor::zeros(&param.ctx, &param.data.shape));

                self.engine
                    .adamw_update(&param.ctx, &param.data, grad, m, v, &p);
            }
        }
    }

    /// Clear the gradients of `params`.
    pub fn zero_grad(&mut self, params: &[Tensor]) {
        self.clear_grads(params);
    }

    fn clear_grads(&mut self, params: &[Tensor]) {
        for param in params {
            *param.grad.borrow_mut() = None;
        }
    }
}

impl Optimizer for AdamW {
    fn step(&mut self, params: &[Tensor]) {
        self.apply_step(params);
    }
    fn zero_grad(&mut self, params: &[Tensor]) {
        self.clear_grads(params);
    }
    /// Serialize optimizer state keyed by parameter name (`None` if unsupported).
    fn save_state(
        &self,
        ctx: &Arc<WgpuContext>,
        named_params: &HashMap<String, Tensor>,
    ) -> Option<OptimizerCheckpoint> {
        let names_by_ptr: HashMap<usize, String> = named_params
            .iter()
            .map(|(name, tensor)| (Arc::as_ptr(&tensor.data) as usize, name.clone()))
            .collect();

        let mut m_states = HashMap::new();
        let mut v_states = HashMap::new();
        for (ptr, state) in &self.m_states {
            if let Some(name) = names_by_ptr.get(ptr) {
                m_states.insert(
                    name.clone(),
                    OptimizerTensorState {
                        shape: state.shape.clone(),
                        data: pollster::block_on(state.to_vec(ctx)),
                    },
                );
            }
        }
        for (ptr, state) in &self.v_states {
            if let Some(name) = names_by_ptr.get(ptr) {
                v_states.insert(
                    name.clone(),
                    OptimizerTensorState {
                        shape: state.shape.clone(),
                        data: pollster::block_on(state.to_vec(ctx)),
                    },
                );
            }
        }

        Some(OptimizerCheckpoint::AdamW(AdamWCheckpoint {
            lr: self.lr,
            beta1: self.beta1,
            beta2: self.beta2,
            eps: self.eps,
            weight_decay: self.weight_decay,
            t: self.t,
            m_states,
            v_states,
        }))
    }
    /// Restore state saved by [`save_state`](Optimizer::save_state);
    /// returns whether it was applied.
    fn load_state(
        &mut self,
        ctx: &Arc<WgpuContext>,
        named_params: &HashMap<String, Tensor>,
        state: &OptimizerCheckpoint,
    ) -> bool {
        let OptimizerCheckpoint::AdamW(state) = state;
        self.lr = state.lr;
        self.beta1 = state.beta1;
        self.beta2 = state.beta2;
        self.eps = state.eps;
        self.weight_decay = state.weight_decay;
        self.t = state.t;
        self.m_states.clear();
        self.v_states.clear();

        for (name, tensor_state) in &state.m_states {
            if let Some(param) = named_params.get(name) {
                let key = Arc::as_ptr(&param.data) as usize;
                self.m_states.insert(
                    key,
                    UnifiedTensor::new(ctx, &tensor_state.data, &tensor_state.shape),
                );
            }
        }
        for (name, tensor_state) in &state.v_states {
            if let Some(param) = named_params.get(name) {
                let key = Arc::as_ptr(&param.data) as usize;
                self.v_states.insert(
                    key,
                    UnifiedTensor::new(ctx, &tensor_state.data, &tensor_state.shape),
                );
            }
        }
        true
    }
}

// --- Learning Rate Schedulers ---

/// Trait for learning rate schedulers
pub trait LRScheduler {
    /// Get current learning rate
    fn get_lr(&self) -> f32;
    /// Advance the scheduler by one step
    fn step(&mut self);
    /// Get current step number
    fn current_step(&self) -> usize;
}

/// Cosine Annealing Learning Rate Scheduler
/// LR decays from initial_lr to min_lr following a cosine curve
pub struct CosineAnnealingLR {
    /// Starting rate.
    pub initial_lr: f32,
    /// Final rate.
    pub min_lr: f32,
    /// Length of the decay.
    pub total_steps: usize,
    /// Steps taken.
    pub current: usize,
}

impl CosineAnnealingLR {
    /// Decay from `initial_lr` to `min_lr` over `total_steps`.
    pub fn new(initial_lr: f32, min_lr: f32, total_steps: usize) -> Self {
        Self {
            initial_lr,
            min_lr,
            total_steps,
            current: 0,
        }
    }
}

impl LRScheduler for CosineAnnealingLR {
    fn get_lr(&self) -> f32 {
        if self.current >= self.total_steps {
            return self.min_lr;
        }
        let progress = self.current as f32 / self.total_steps as f32;
        let cosine = (1.0 + (std::f32::consts::PI * progress).cos()) / 2.0;
        self.min_lr + (self.initial_lr - self.min_lr) * cosine
    }

    fn step(&mut self) {
        self.current += 1;
    }

    fn current_step(&self) -> usize {
        self.current
    }
}

/// Linear Warmup + Cosine Annealing Scheduler
/// Warmups linearly from 0 to peak_lr, then decays using cosine schedule
pub struct WarmupCosineScheduler {
    /// Rate at the end of warmup.
    pub peak_lr: f32,
    /// Final rate.
    pub min_lr: f32,
    /// Warmup length (linear from 0).
    pub warmup_steps: usize,
    /// Total length including warmup.
    pub total_steps: usize,
    /// Steps taken.
    pub current: usize,
}

impl WarmupCosineScheduler {
    /// Warm up to `peak_lr` over `warmup_steps`, then decay to `min_lr` at `total_steps`.
    pub fn new(peak_lr: f32, min_lr: f32, warmup_steps: usize, total_steps: usize) -> Self {
        Self {
            peak_lr,
            min_lr,
            warmup_steps,
            total_steps,
            current: 0,
        }
    }
}

impl LRScheduler for WarmupCosineScheduler {
    fn get_lr(&self) -> f32 {
        if self.current < self.warmup_steps {
            // Linear warmup
            let progress = self.current as f32 / self.warmup_steps as f32;
            self.peak_lr * progress
        } else if self.current >= self.total_steps {
            self.min_lr
        } else {
            // Cosine decay
            let decay_steps = self.total_steps - self.warmup_steps;
            let decay_current = self.current - self.warmup_steps;
            let progress = decay_current as f32 / decay_steps as f32;
            let cosine = (1.0 + (std::f32::consts::PI * progress).cos()) / 2.0;
            self.min_lr + (self.peak_lr - self.min_lr) * cosine
        }
    }

    fn step(&mut self) {
        self.current += 1;
    }

    fn current_step(&self) -> usize {
        self.current
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference RMSNorm forward on the CPU.
    fn rms_norm_cpu(x: &[f32], w: &[f32], eps: f32, hidden: usize) -> Vec<f32> {
        x.chunks(hidden)
            .flat_map(|row| {
                let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / hidden as f32;
                let inv = 1.0 / (mean_sq + eps).sqrt();
                row.iter().zip(w).map(move |(v, wi)| v * inv * wi)
            })
            .collect()
    }

    fn pseudo_random(n: usize, seed: u32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                (((i as u32 + 1).wrapping_mul(2_654_435_761) ^ seed) % 1000) as f32 / 500.0 - 1.0
            })
            .collect()
    }

    /// Central-difference gradient of `f` at `x`.
    fn numeric_grad(x: &[f32], f: impl Fn(&[f32]) -> f64) -> Vec<f64> {
        let h = 1e-3f32;
        (0..x.len())
            .map(|i| {
                let mut plus = x.to_vec();
                plus[i] += h;
                let mut minus = x.to_vec();
                minus[i] -= h;
                (f(&plus) - f(&minus)) / (2.0 * h as f64)
            })
            .collect()
    }

    fn assert_close(analytic: &[f32], numeric: &[f64], what: &str) {
        for (i, (a, n)) in analytic.iter().zip(numeric).enumerate() {
            let tol = 2e-3 + 1e-2 * n.abs();
            assert!(
                (*a as f64 - n).abs() <= tol,
                "{what}[{i}]: analytic {a} vs numeric {n}"
            );
        }
    }

    #[test]
    fn rms_norm_backward_matches_finite_differences() {
        let (hidden, eps) = (5, 1e-5);
        let x = pseudo_random(3 * hidden, 1);
        let w = pseudo_random(hidden, 2);
        let g = pseudo_random(3 * hidden, 3);
        let loss = |x: &[f32], w: &[f32]| -> f64 {
            rms_norm_cpu(x, w, eps, hidden)
                .iter()
                .zip(&g)
                .map(|(y, gi)| (*y * *gi) as f64)
                .sum()
        };
        let (grad_x, grad_w) = rms_norm_backward_cpu(&x, &g, &w, eps, hidden);
        assert_close(&grad_x, &numeric_grad(&x, |x| loss(x, &w)), "grad_x");
        assert_close(&grad_w, &numeric_grad(&w, |w| loss(&x, w)), "grad_w");
    }

    #[test]
    fn rope_backward_is_the_transposed_rotation() {
        // For loss = sum(g * rope(x)), dloss/dx = rope^T(g) = inverse rope of g.
        let shape = [2, 3, 8]; // batch 2, seq 3, two heads of 4
        let (head_dim, pos) = (4, 5);
        let x = pseudo_random(48, 4);
        let g = pseudo_random(48, 5);
        let loss = |x: &[f32]| -> f64 {
            let mut y = x.to_vec();
            apply_rope_in_place(&mut y, &shape, head_dim, pos, false);
            y.iter().zip(&g).map(|(a, b)| (*a * *b) as f64).sum()
        };
        let mut analytic = g.clone();
        apply_rope_in_place(&mut analytic, &shape, head_dim, pos, true);
        assert_close(&analytic, &numeric_grad(&x, loss), "rope grad");
    }

    #[test]
    fn rope_positions_depend_on_token_not_head() {
        // Two identical heads of one token must be rotated identically.
        let shape = [1, 2, 8];
        let mut data: Vec<f32> = [1.0, 2.0, 3.0, 4.0].repeat(4);
        apply_rope_in_place(&mut data, &shape, 4, 3, false);
        assert_eq!(data[0..4], data[4..8]);
        assert_eq!(data[8..12], data[12..16]);
        assert_ne!(data[0..4], data[8..12]);
        // Position 0 is the identity.
        let mut first = vec![1.0, 2.0, 3.0, 4.0];
        apply_rope_in_place(&mut first, &[1, 4], 4, 0, false);
        assert_eq!(first, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn schedulers_follow_their_curves() {
        let mut cosine = CosineAnnealingLR::new(1.0, 0.0, 10);
        assert_eq!(cosine.get_lr(), 1.0);
        for _ in 0..5 {
            cosine.step();
        }
        assert!((cosine.get_lr() - 0.5).abs() < 1e-6);
        for _ in 0..5 {
            cosine.step();
        }
        assert_eq!(cosine.get_lr(), 0.0);
        assert_eq!(cosine.current_step(), 10);

        let mut warm = WarmupCosineScheduler::new(1.0, 0.1, 4, 14);
        let lrs: Vec<f32> = (0..15)
            .map(|_| {
                let lr = warm.get_lr();
                warm.step();
                lr
            })
            .collect();
        assert_eq!(&lrs[..5], &[0.0, 0.25, 0.5, 0.75, 1.0]);
        assert!((lrs[9] - 0.55).abs() < 1e-6);
        assert_eq!(lrs[14], 0.1);
    }
}
