//! SGD Optimizer
//!
//! Stochastic Gradient Descent with optional momentum, Nesterov momentum and
//! weight decay. Updates run as tensor ops on the parameters' device.

use super::Optimizer;
use crate::tensor::{no_grad, Tensor};

/// Stochastic Gradient Descent optimizer.
///
/// With momentum `mu`, weight decay `wd` and learning rate `lr`:
/// `g = grad + wd * p`, `v = mu * v + g`, `p -= lr * v`
/// (Nesterov: `p -= lr * (g + mu * v)`), matching PyTorch.
///
/// # Example
/// ```rust
/// use rusty::prelude::*;
///
/// let device = Device::cpu();
/// let model = nn::Linear::new(10, 5, &device);
/// let mut optimizer = optim::SGD::with_momentum(model.parameters(), 0.01, 0.9);
///
/// let x = Tensor::randn(&[8, 10], &device);
/// let loss = model.forward(&x).square().mean();
/// optimizer.zero_grad();
/// loss.backward();
/// optimizer.step();
/// ```
pub struct SGD {
    params: Vec<Tensor>,
    lr: f32,
    momentum: f32,
    nesterov: bool,
    weight_decay: f32,
    velocities: Vec<Option<Tensor>>,
}

impl SGD {
    /// Create a new SGD optimizer.
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        Self::with_momentum(params, lr, 0.0)
    }

    /// Create SGD with momentum.
    pub fn with_momentum(params: Vec<Tensor>, lr: f32, momentum: f32) -> Self {
        let num_params = params.len();
        Self {
            params,
            lr,
            momentum,
            nesterov: false,
            weight_decay: 0.0,
            velocities: vec![None; num_params],
        }
    }

    /// Set weight decay (L2 penalty added to the gradient).
    pub fn weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Use Nesterov momentum.
    pub fn nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }

    /// The parameters being optimized.
    pub fn params(&self) -> &[Tensor] {
        &self.params
    }
}

impl Optimizer for SGD {
    fn step(&mut self) {
        no_grad(|| {
            for (param, velocity) in self.params.iter().zip(&mut self.velocities) {
                let Some(mut g) = param.grad() else { continue };
                if self.weight_decay != 0.0 {
                    g = g.add(&param.mul_scalar(self.weight_decay));
                }
                if self.momentum != 0.0 {
                    let v = match velocity.take() {
                        Some(v) => v.mul_scalar(self.momentum).add(&g),
                        None => g.deep_clone(),
                    };
                    g = if self.nesterov {
                        g.add(&v.mul_scalar(self.momentum))
                    } else {
                        v.clone()
                    };
                    *velocity = Some(v);
                }
                param.assign(&param.sub(&g.mul_scalar(self.lr)));
            }
        });
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
            .field("nesterov", &self.nesterov)
            .field("weight_decay", &self.weight_decay)
            .field("num_params", &self.params.len())
            .finish()
    }
}
