use rusty_backend::{AdamParams, ComputeEngine, UnifiedTensor, WgpuContext};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

pub trait AutogradNode {
    fn backward(&self, grad: &UnifiedTensor, engine: &ComputeEngine) -> Vec<UnifiedTensor>;
    fn inputs(&self) -> Vec<Tensor>;
}

#[derive(Clone)]
pub struct Tensor {
    pub data: Arc<UnifiedTensor>,
    pub grad: Arc<RefCell<Option<UnifiedTensor>>>,
    pub ctx: Arc<WgpuContext>,
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
    pub fn new(ctx: Arc<WgpuContext>, data: UnifiedTensor) -> Self {
        Self {
            data: Arc::new(data),
            grad: Arc::new(RefCell::new(None)),
            ctx,
            creator: None,
        }
    }

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
                    for (input, input_grad) in inputs.iter().zip(input_grads.into_iter()) {
                        input.accumulate_grad(engine, &input_grad);
                    }
                }
            }
        }
    }

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

    pub fn zero_grad(&self) {
        *self.grad.borrow_mut() = None;
    }

    // Recursive zero_grad for the whole graph
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

pub struct MatMulNode {
    pub a: Tensor,
    pub b: Tensor,
}
impl AutogradNode for MatMulNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.a.clone(), self.b.clone()]
    }
    fn backward(&self, grad: &UnifiedTensor, engine: &ComputeEngine) -> Vec<UnifiedTensor> {
        let grad_a = UnifiedTensor::empty(&self.a.ctx, &self.a.data.shape);
        engine.matmul_bt(&self.a.ctx, grad, &self.b.data, &grad_a);
        let grad_b = UnifiedTensor::empty(&self.b.ctx, &self.b.data.shape);
        engine.matmul_at(&self.a.ctx, &self.a.data, grad, &grad_b);
        vec![grad_a, grad_b]
    }
}

pub struct AddNode {
    pub a: Tensor,
    pub b: Tensor,
}
impl AutogradNode for AddNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.a.clone(), self.b.clone()]
    }
    fn backward(&self, grad: &UnifiedTensor, engine: &ComputeEngine) -> Vec<UnifiedTensor> {
        let grad_a = UnifiedTensor::empty(&self.a.ctx, &self.a.data.shape);
        let grad_b = UnifiedTensor::empty(&self.b.ctx, &self.b.data.shape);
        engine.add_backward(&self.a.ctx, grad, &grad_a, &grad_b);
        vec![grad_a, grad_b]
    }
}

pub struct MulNode {
    pub a: Tensor,
    pub b: Tensor,
}
impl AutogradNode for MulNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.a.clone(), self.b.clone()]
    }
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

pub struct SubNode {
    pub a: Tensor,
    pub b: Tensor,
}
impl AutogradNode for SubNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.a.clone(), self.b.clone()]
    }
    fn backward(&self, grad: &UnifiedTensor, engine: &ComputeEngine) -> Vec<UnifiedTensor> {
        let grad_a = UnifiedTensor::empty(&self.a.ctx, &self.a.data.shape);
        let grad_b = UnifiedTensor::empty(&self.b.ctx, &self.b.data.shape);
        engine.sub_backward(&self.a.ctx, grad, &grad_a, &grad_b);
        vec![grad_a, grad_b]
    }
}

pub struct SiLUNode {
    pub input: Tensor,
}
impl AutogradNode for SiLUNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }
    fn backward(&self, grad: &UnifiedTensor, engine: &ComputeEngine) -> Vec<UnifiedTensor> {
        let grad_in = UnifiedTensor::empty(&self.input.ctx, &self.input.data.shape);
        engine.silu_backward(&self.input.ctx, grad, &self.input.data, &grad_in);
        vec![grad_in]
    }
}

pub struct ReluNode {
    pub input: Tensor,
}
impl AutogradNode for ReluNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }
    fn backward(&self, grad: &UnifiedTensor, engine: &ComputeEngine) -> Vec<UnifiedTensor> {
        let grad_in = UnifiedTensor::empty(&self.input.ctx, &self.input.data.shape);
        engine.relu_backward(&self.input.ctx, grad, &self.input.data, &grad_in);
        vec![grad_in]
    }
}

pub struct SoftmaxNode {
    pub input: Tensor,
    pub output: Tensor,
}
impl AutogradNode for SoftmaxNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }
    fn backward(&self, grad: &UnifiedTensor, engine: &ComputeEngine) -> Vec<UnifiedTensor> {
        let grad_in = UnifiedTensor::empty(&self.input.ctx, &self.input.data.shape);
        engine.softmax_backward(&self.input.ctx, grad, &self.output.data, &grad_in);
        vec![grad_in]
    }
}

pub struct ReshapeNode {
    pub input: Tensor,
    pub original_shape: Vec<usize>,
}
impl AutogradNode for ReshapeNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }
    fn backward(&self, grad: &UnifiedTensor, _engine: &ComputeEngine) -> Vec<UnifiedTensor> {
        vec![grad.reshaped(&self.original_shape)]
    }
}

fn apply_rope_in_place(data: &mut [f32], shape: &[usize], head_dim: usize, pos: usize, inverse: bool) {
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

pub struct RoPENode {
    pub input: Tensor,
    pub head_dim: usize,
    pub pos: usize,
}
impl AutogradNode for RoPENode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }
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

pub struct RMSNormNode {
    pub input: Tensor,
    pub weight: Tensor,
    pub eps: f32,
}
impl AutogradNode for RMSNormNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.input.clone(), self.weight.clone()]
    }
    fn backward(&self, grad: &UnifiedTensor, _engine: &ComputeEngine) -> Vec<UnifiedTensor> {
        let input_shape = self.input.data.shape.clone();
        let hidden = *input_shape.last().expect("RMSNorm input must be non-empty");
        let rows = input_shape.iter().product::<usize>() / hidden;

        let x = pollster::block_on(self.input.data.to_vec(&self.input.ctx));
        let g = pollster::block_on(grad.to_vec(&self.input.ctx));
        let w = pollster::block_on(self.weight.data.to_vec(&self.weight.ctx));

        let mut grad_x = vec![0.0f32; x.len()];
        let mut grad_w = vec![0.0f32; w.len()];

        for row in 0..rows {
            let start = row * hidden;
            let end = start + hidden;
            let x_row = &x[start..end];
            let g_row = &g[start..end];

            let mean_sq = x_row.iter().map(|v| v * v).sum::<f32>() / hidden as f32;
            let inv_rms: f32 = 1.0f32 / (mean_sq + self.eps).sqrt();

            let mut dot = 0.0f32;
            for idx in 0..hidden {
                dot += g_row[idx] * w[idx] * x_row[idx];
                grad_w[idx] += g_row[idx] * x_row[idx] * inv_rms;
            }

            for idx in 0..hidden {
                grad_x[start + idx] = g_row[idx] * w[idx] * inv_rms
                    - x_row[idx] * dot * inv_rms.powi(3) / hidden as f32;
            }
        }

        vec![
            UnifiedTensor::new(&self.input.ctx, &grad_x, &input_shape),
            UnifiedTensor::new(&self.weight.ctx, &grad_w, &self.weight.data.shape),
        ]
    }
}

pub struct SelectTokenNode {
    pub input: Tensor,
    pub token_index: usize,
    pub vocab_size: usize,
}
impl AutogradNode for SelectTokenNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }
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

pub struct MSELossNode {
    pub pred: Tensor,
    pub target: Tensor,
}
impl AutogradNode for MSELossNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.pred.clone(), self.target.clone()]
    }
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
        // Target gradient is not usually needed, but technically it's -(pred-target)
        let grad_target = UnifiedTensor::empty(&self.target.ctx, &self.target.data.shape);
        vec![grad_in, grad_target]
    }
}

/// Cross-entropy loss node for LLM training
/// Forward: -log(softmax(logits)[target])
/// Backward: softmax(logits) - one_hot(target)
pub struct CrossEntropyLossNode {
    pub logits: Tensor,
    pub target_idx: u32,
}
impl AutogradNode for CrossEntropyLossNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.logits.clone()]
    }
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
    pub up: Tensor,
    pub gate: Tensor,
}
impl AutogradNode for SiLUGateNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.up.clone(), self.gate.clone()]
    }
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

/// Trait for functions that can be checkpointed
pub trait CheckpointFn: Send + Sync {
    fn forward(&self, inputs: &[Tensor], engine: &ComputeEngine) -> Vec<Tensor>;
    fn backward(
        &self,
        inputs: &[Tensor],
        outputs: &[Tensor],
        grad_outputs: &[UnifiedTensor],
        engine: &ComputeEngine,
    ) -> Vec<UnifiedTensor>;
}

/// Node that recomputes forward pass during backward to save memory
pub struct CheckpointedNode {
    pub inputs: Vec<Tensor>,
    pub checkpoint_fn: Arc<dyn CheckpointFn>,
    pub outputs: Vec<Tensor>, // Only shapes stored, data discarded after forward
}

impl AutogradNode for CheckpointedNode {
    fn inputs(&self) -> Vec<Tensor> {
        self.inputs.clone()
    }

    fn backward(&self, grad: &UnifiedTensor, engine: &ComputeEngine) -> Vec<UnifiedTensor> {
        // Recompute forward pass to get activations
        let recomputed_outputs = self.checkpoint_fn.forward(&self.inputs, engine);

        // Now compute backward with recomputed activations
        self.checkpoint_fn
            .backward(&self.inputs, &recomputed_outputs, &[grad.clone()], engine)
    }
}

/// Checkpoint a sequence of operations to save memory
/// During forward: computes normally but discards intermediate activations
/// During backward: recomputes activations from checkpoint
pub fn checkpoint<F>(inputs: Vec<Tensor>, forward_fn: F, engine: &ComputeEngine) -> Vec<Tensor>
where
    F: Fn(&[Tensor], &ComputeEngine) -> Vec<Tensor> + Send + Sync + 'static,
{
    // Create wrapper for the forward function
    struct FnWrapper<Func>(Func);

    impl<Func> CheckpointFn for FnWrapper<Func>
    where
        Func: Fn(&[Tensor], &ComputeEngine) -> Vec<Tensor> + Send + Sync,
    {
        fn forward(&self, inputs: &[Tensor], engine: &ComputeEngine) -> Vec<Tensor> {
            (self.0)(inputs, engine)
        }

        fn backward(
            &self,
            inputs: &[Tensor],
            outputs: &[Tensor],
            grad_outputs: &[UnifiedTensor],
            engine: &ComputeEngine,
        ) -> Vec<UnifiedTensor> {
            // Run backward on the recomputed outputs
            // Each output should have been created with autograd, so we can backprop through it
            let _input_grads: Vec<UnifiedTensor> = vec![];
            for (_i, (output, grad)) in outputs.iter().zip(grad_outputs.iter()).enumerate() {
                *output.grad.borrow_mut() = Some(grad.clone());
            }

            // Trigger backward on each output
            for output in outputs {
                if output.creator.is_some() {
                    // Get gradients for all inputs
                    let grad = output.grad.borrow().clone().unwrap();
                    if let Some(creator) = &output.creator {
                        return creator.backward(&grad, engine);
                    }
                }
            }

            // Return empty grads if no backward path
            inputs
                .iter()
                .map(|t| UnifiedTensor::zeros(&t.ctx, &t.data.shape))
                .collect()
        }
    }

    // Run forward pass
    let outputs = forward_fn(&inputs, engine);

    // Create checkpointed outputs that will recompute on backward
    let checkpoint_fn = Arc::new(FnWrapper(forward_fn));

    outputs
        .into_iter()
        .map(|output| {
            let node = Arc::new(CheckpointedNode {
                inputs: inputs.clone(),
                checkpoint_fn: checkpoint_fn.clone(),
                outputs: vec![output.clone()],
            });
            Tensor::with_creator(output.ctx.clone(), output.data.as_ref().clone(), node)
        })
        .collect()
}

/// Convenience function for checkpointing a single-output function
pub fn checkpoint_single<F>(input: Tensor, forward_fn: F, engine: &ComputeEngine) -> Tensor
where
    F: Fn(&Tensor, &ComputeEngine) -> Tensor + Send + Sync + 'static,
{
    let wrapper = move |inputs: &[Tensor], eng: &ComputeEngine| -> Vec<Tensor> {
        vec![forward_fn(&inputs[0], eng)]
    };
    checkpoint(vec![input], wrapper, engine).pop().unwrap()
}

/// Apply checkpointing to a sequence of functions (e.g., transformer blocks)
/// This is the main entry point for memory-efficient training of large models.
///
/// Usage:
/// ```rust
/// // Instead of:
/// let mut x = input;
/// for block in &blocks {
///     x = block.forward(&x, engine);
/// }
///
/// // Use checkpointed version:
/// let x = checkpoint_single(hidden, |h, eng| layer.forward(h, eng), engine);
/// ```
///
/// For sequential layers, apply checkpoint_single to each layer in a loop:
/// ```rust,ignore
/// let mut x = input;
/// for layer in &layers {
///     x = checkpoint_single(x, |h, eng| layer.forward(h, eng), engine);
/// }
/// ```

impl Tensor {
    pub fn reshape(input: &Tensor, new_shape: &[usize]) -> Tensor {
        let out_data = input.data.reshaped(new_shape);
        let node = Arc::new(ReshapeNode {
            input: input.clone(),
            original_shape: input.data.shape.clone(),
        });
        Tensor::with_creator(input.ctx.clone(), out_data, node)
    }

    pub fn reshape_2d(input: &Tensor) -> Tensor {
        match input.data.shape.as_slice() {
            [_, _] => input.clone(),
            [batch, seq, hidden] => Tensor::reshape(input, &[*batch * *seq, *hidden]),
            shape => panic!("reshape_2d expects 2D or 3D tensor, got {shape:?}"),
        }
    }

    pub fn unflatten_last(input: &Tensor, batch: usize, seq: usize) -> Tensor {
        let hidden = *input
            .data
            .shape
            .last()
            .expect("unflatten_last requires a non-empty shape");
        Tensor::reshape(input, &[batch, seq, hidden])
    }

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
    pub fn add(a: &Tensor, b: &Tensor, engine: &ComputeEngine) -> Tensor {
        let out_data = UnifiedTensor::empty(&a.ctx, &a.data.shape);
        engine.add(&a.ctx, &a.data, &b.data, &out_data);
        let node = Arc::new(AddNode {
            a: a.clone(),
            b: b.clone(),
        });
        Tensor::with_creator(a.ctx.clone(), out_data, node)
    }
    pub fn mul(a: &Tensor, b: &Tensor, engine: &ComputeEngine) -> Tensor {
        let out_data = UnifiedTensor::empty(&a.ctx, &a.data.shape);
        engine.mul(&a.ctx, &a.data, &b.data, &out_data);
        let node = Arc::new(MulNode {
            a: a.clone(),
            b: b.clone(),
        });
        Tensor::with_creator(a.ctx.clone(), out_data, node)
    }
    pub fn sub(a: &Tensor, b: &Tensor, engine: &ComputeEngine) -> Tensor {
        let out_data = UnifiedTensor::empty(&a.ctx, &a.data.shape);
        engine.sub(&a.ctx, &a.data, &b.data, &out_data);
        let node = Arc::new(SubNode {
            a: a.clone(),
            b: b.clone(),
        });
        Tensor::with_creator(a.ctx.clone(), out_data, node)
    }
    pub fn silu(input: &Tensor, engine: &ComputeEngine) -> Tensor {
        let out_data = UnifiedTensor::empty(&input.ctx, &input.data.shape);
        engine.silu(&input.ctx, &input.data, &out_data);
        let node = Arc::new(SiLUNode {
            input: input.clone(),
        });
        Tensor::with_creator(input.ctx.clone(), out_data, node)
    }
    pub fn mse_loss(pred: &Tensor, target: &Tensor, engine: &ComputeEngine) -> Tensor {
        let out_data = UnifiedTensor::empty(&pred.ctx, &[1]);
        engine.mse_loss(&pred.ctx, &pred.data, &target.data, &out_data);
        let node = Arc::new(MSELossNode {
            pred: pred.clone(),
            target: target.clone(),
        });
        Tensor::with_creator(pred.ctx.clone(), out_data, node)
    }

    pub fn relu(input: &Tensor, engine: &ComputeEngine) -> Tensor {
        let out_data = UnifiedTensor::empty(&input.ctx, &input.data.shape);
        engine.relu(&input.ctx, &input.data, &out_data);
        let node = Arc::new(ReluNode {
            input: input.clone(),
        });
        Tensor::with_creator(input.ctx.clone(), out_data, node)
    }

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

    pub fn rms_norm(input: &Tensor, weight: &Tensor, eps: f32, engine: &ComputeEngine) -> Tensor {
        let out_data = UnifiedTensor::empty(&input.ctx, &input.data.shape);
        engine.rms_norm(&input.ctx, &input.data, &weight.data, &out_data);
        let node = Arc::new(RMSNormNode {
            input: input.clone(),
            weight: weight.clone(),
            eps,
        });
        Tensor::with_creator(input.ctx.clone(), out_data, node)
    }

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
    /// Computes -log(softmax(logits)[target])
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

    /// Flash Attention V2 (Fused)
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

pub struct FlashAttentionNode {
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
    pub mask: Option<Tensor>,
    pub causal: bool,
    pub dropout_prob: f32,
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

        engine.flash_attention_v2_backward(
            &self.q.ctx,
            &self.q.data,
            &self.k.data,
            &self.v.data,
            &out_recomputed,
            grad,
            &grad_q,
            &grad_k,
            &grad_v,
            self.causal,
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

pub trait Optimizer: Send + Sync {
    fn step(&mut self, params: &[Tensor]);
    fn zero_grad(&mut self, params: &[Tensor]);
    fn save_state(
        &self,
        _ctx: &Arc<WgpuContext>,
        _named_params: &HashMap<String, Tensor>,
    ) -> Option<OptimizerCheckpoint> {
        None
    }
    fn load_state(
        &mut self,
        _ctx: &Arc<WgpuContext>,
        _named_params: &HashMap<String, Tensor>,
        _state: &OptimizerCheckpoint,
    ) -> bool {
        false
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct OptimizerTensorState {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AdamWCheckpoint {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub t: f32,
    pub m_states: HashMap<String, OptimizerTensorState>,
    pub v_states: HashMap<String, OptimizerTensorState>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum OptimizerCheckpoint {
    AdamW(AdamWCheckpoint),
}

pub struct AdamW {
    pub engine: Arc<ComputeEngine>,
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub t: f32,
    m_states: HashMap<usize, UnifiedTensor>,
    v_states: HashMap<usize, UnifiedTensor>,
}

impl AdamW {
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
    pub initial_lr: f32,
    pub min_lr: f32,
    pub total_steps: usize,
    pub current: usize,
}

impl CosineAnnealingLR {
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
    pub peak_lr: f32,
    pub min_lr: f32,
    pub warmup_steps: usize,
    pub total_steps: usize,
    pub current: usize,
}

impl WarmupCosineScheduler {
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
