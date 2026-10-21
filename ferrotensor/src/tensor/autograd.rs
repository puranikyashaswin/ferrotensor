//! Autograd - Automatic Differentiation Engine
//!
//! Reverse-mode autodiff over a graph recorded during the forward pass. Each
//! tracked op attaches a [`Node`] to its output holding the op's inputs and a
//! closure that maps the output gradient to input gradients. The closures are
//! written with ordinary tensor ops, run with gradient tracking disabled, so
//! the backward pass runs on the same device (GPU kernels or CPU) as the
//! forward pass.

use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::Tensor;

thread_local! {
    static GRAD_ENABLED: Cell<bool> = const { Cell::new(true) };
}

/// True unless inside [`no_grad`] (per thread).
pub fn is_grad_enabled() -> bool {
    GRAD_ENABLED.with(Cell::get)
}

/// Run `f` without recording ops for autograd (inference, optimizer
/// updates, metric computation).
///
/// # Example
/// ```rust
/// use ferrotensor::prelude::*;
///
/// let device = Device::cpu();
/// let w = Tensor::ones(&[2], &device).requires_grad_();
/// let y = ferrotensor::no_grad(|| w.mul_scalar(2.0));
/// assert!(!y.requires_grad);
/// ```
pub fn no_grad<R>(f: impl FnOnce() -> R) -> R {
    let _guard = NoGradGuard::new();
    f()
}

/// Disables gradient tracking on this thread until dropped.
pub struct NoGradGuard {
    previous: bool,
}

impl NoGradGuard {
    /// Disable tracking; the previous state is restored on drop.
    pub fn new() -> Self {
        let previous = GRAD_ENABLED.with(|g| g.replace(false));
        Self { previous }
    }
}

impl Default for NoGradGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        GRAD_ENABLED.with(|g| g.set(self.previous));
    }
}

/// Maps the gradient of an op's output to the gradients of its inputs
/// (`None` for an input that needs none).
pub(crate) type BackwardFn = Box<dyn Fn(&Tensor) -> Vec<Option<Tensor>> + Send + Sync>;

/// One recorded op in the autograd graph.
pub(crate) struct Node {
    pub(crate) name: &'static str,
    pub(crate) inputs: Vec<Tensor>,
    pub(crate) backward: BackwardFn,
}

impl Drop for Node {
    /// Unlink the graph iteratively: dropping a long chain of nodes
    /// recursively would overflow the stack. Backward closures capture only
    /// detached tensors, so `inputs` holds every link to earlier nodes.
    fn drop(&mut self) {
        let mut stack: Vec<Arc<Node>> = self
            .inputs
            .iter_mut()
            .filter_map(|t| t.node.take())
            .collect();
        while let Some(node) = stack.pop() {
            if let Ok(mut node) = Arc::try_unwrap(node) {
                stack.extend(node.inputs.iter_mut().filter_map(|t| t.node.take()));
            }
        }
    }
}

impl Tensor {
    /// Attach a backward node to a freshly computed `self` when gradient
    /// tracking is on and any input requires grad.
    ///
    /// Closures must capture [`Tensor::detach`]ed copies, never graph-carrying
    /// tensors: capturing the output would form a reference cycle, and
    /// capturing inputs would defeat the iterative `Drop` of [`Node`].
    pub(crate) fn with_node(
        mut self,
        name: &'static str,
        inputs: Vec<Tensor>,
        backward: impl Fn(&Tensor) -> Vec<Option<Tensor>> + Send + Sync + 'static,
    ) -> Tensor {
        if is_grad_enabled() && inputs.iter().any(|t| t.requires_grad) {
            self.requires_grad = true;
            self.node = Some(Arc::new(Node {
                name,
                inputs,
                backward: Box::new(backward),
            }));
        }
        self
    }

    /// Name of the op that produced this tensor (`None` for leaves).
    pub fn grad_fn_name(&self) -> Option<&'static str> {
        self.node.as_ref().map(|n| n.name)
    }

    /// Backpropagate from this tensor, seeding its gradient with ones.
    ///
    /// For a one-element loss this is the usual `loss.backward()`; for a
    /// larger tensor it differentiates the sum of its elements. Gradients
    /// accumulate into the `.grad()` of every leaf that requires grad, so
    /// call `zero_grad` between steps.
    pub fn backward(&self) {
        let seed = Tensor::ones(&self.shape, &self.device);
        self.backward_with(&seed);
    }

    /// Backpropagate from this tensor with an explicit output gradient.
    ///
    /// # Panics
    /// Panics when `grad` does not have this tensor's shape.
    pub fn backward_with(&self, grad: &Tensor) {
        assert_eq!(
            grad.shape, self.shape,
            "gradient shape must match the tensor"
        );
        if !self.requires_grad {
            return;
        }
        let _guard = NoGradGuard::new();

        let order = topological_order(self);
        let mut pending: HashMap<usize, Tensor> = HashMap::new();
        pending.insert(self.id(), grad.detach());

        for tensor in order.iter().rev() {
            let Some(g) = pending.remove(&tensor.id()) else {
                continue;
            };
            match &tensor.node {
                Some(node) => {
                    let input_grads = (node.backward)(&g);
                    debug_assert_eq!(input_grads.len(), node.inputs.len());
                    for (input, input_grad) in node.inputs.iter().zip(input_grads) {
                        let Some(input_grad) = input_grad else {
                            continue;
                        };
                        if !input.requires_grad {
                            continue;
                        }
                        assert_eq!(
                            input_grad.shape, input.shape,
                            "backward of '{}' produced a gradient of the wrong shape",
                            node.name
                        );
                        let merged = match pending.remove(&input.id()) {
                            Some(existing) => existing.add(&input_grad),
                            None => input_grad,
                        };
                        pending.insert(input.id(), merged);
                    }
                }
                None => {
                    let mut slot = tensor.grad.write().expect("grad lock poisoned");
                    // A fresh copy: backward closures may hand the same storage
                    // to several inputs, and optimizers update grads in place.
                    let total = match slot.take() {
                        Some(existing) => existing.add(&g),
                        None => g.deep_clone(),
                    };
                    *slot = Some(total);
                }
            }
        }
    }
}

/// Tensors reachable from `root` through grad-requiring inputs, inputs
/// before the ops that consume them. Iterative, so deep graphs cannot
/// overflow the stack.
fn topological_order(root: &Tensor) -> Vec<Tensor> {
    let mut order = Vec::new();
    let mut visited = HashSet::new();
    let mut stack = vec![(root.clone(), false)];
    while let Some((tensor, expanded)) = stack.pop() {
        if expanded {
            order.push(tensor);
            continue;
        }
        if !visited.insert(tensor.id()) {
            continue;
        }
        let inputs: Vec<Tensor> = tensor
            .node
            .as_ref()
            .map(|n| n.inputs.clone())
            .unwrap_or_default();
        stack.push((tensor, true));
        for input in inputs {
            if input.requires_grad && !visited.contains(&input.id()) {
                stack.push((input, false));
            }
        }
    }
    order
}
