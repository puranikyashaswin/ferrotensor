//! SGD Optimizer
//!
//! Stochastic Gradient Descent with optional momentum.

use super::Optimizer;
use crate::tensor::Tensor;

/// Stochastic Gradient Descent optimizer.
///
/// # Example
/// ```rust,no_run
/// use rusty::prelude::*;
///
/// let device = Device::new();
/// let model = nn::Linear::new(10, 5, &device);
/// let mut optimizer = optim::SGD::new(model.parameters(), 0.01);
///
/// // Training step
/// // ...
/// optimizer.step();
/// optimizer.zero_grad();
/// ```
pub struct SGD {
    params: Vec<Tensor>,
    lr: f32,
    momentum: f32,
    weight_decay: f32,
    velocities: Vec<Option<Vec<f32>>>,
}

impl SGD {
    /// Create a new SGD optimizer.
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        let num_params = params.len();
        Self {
            params,
            lr,
            momentum: 0.0,
            weight_decay: 0.0,
            velocities: vec![None; num_params],
        }
    }

    /// Create SGD with momentum.
    pub fn with_momentum(params: Vec<Tensor>, lr: f32, momentum: f32) -> Self {
        let num_params = params.len();
        Self {
            params,
            lr,
            momentum,
            weight_decay: 0.0,
            velocities: vec![None; num_params],
        }
    }

    /// Set weight decay.
    pub fn weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }
}

impl Optimizer for SGD {
    fn step(&mut self) {
        for (i, param) in self.params.iter().enumerate() {
            if let Some(grad_tensor) = param.grad() {
                let grad = grad_tensor.to_vec();
                let mut weights = param.to_vec();

                // Apply weight decay
                if self.weight_decay > 0.0 {
                    for (w, _g) in weights.iter_mut().zip(grad.iter()) {
                        *w -= self.weight_decay * self.lr * *w;
                    }
                }

                // Apply momentum if enabled
                if self.momentum > 0.0 {
                    let v = self.velocities[i].get_or_insert_with(|| vec![0.0; grad.len()]);
                    for ((w, g), vel) in weights.iter_mut().zip(grad.iter()).zip(v.iter_mut()) {
                        *vel = self.momentum * *vel + *g;
                        *w -= self.lr * *vel;
                    }
                } else {
                    for (w, g) in weights.iter_mut().zip(grad.iter()) {
                        *w -= self.lr * *g;
                    }
                }

                // Write back
                param.copy_from_slice(&weights);
            }
        }
    }

    fn zero_grad(&self) {
        for param in &self.params {
            param.zero_grad();
        }
    }

    fn lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

impl std::fmt::Debug for SGD {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SGD")
            .field("lr", &self.lr)
            .field("momentum", &self.momentum)
            .field("weight_decay", &self.weight_decay)
            .field("num_params", &self.params.len())
            .finish()
    }
}
