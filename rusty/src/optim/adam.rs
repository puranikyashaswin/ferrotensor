//! AdamW Optimizer
//!
//! Adam with decoupled weight decay. Updates run as tensor ops on the
//! parameters' device.

use super::{clip_grad_norm, Optimizer};
use crate::tensor::{no_grad, Tensor};

/// AdamW optimizer - Adam with decoupled weight decay.
///
/// Per step `t`: `m = b1 m + (1 - b1) g`, `v = b2 v + (1 - b2) g^2`,
/// `p = p (1 - lr wd) - lr m_hat / (sqrt(v_hat) + eps)` with bias-corrected
/// `m_hat`, `v_hat`. Set `weight_decay(0.0)` for plain Adam.
///
/// # Example
/// ```rust
/// use rusty::prelude::*;
///
/// let device = Device::cpu();
/// let model = nn::Linear::new(10, 5, &device);
/// let mut optimizer = optim::AdamW::new(model.parameters(), 1e-3).clip_grad_norm(1.0);
///
/// let x = Tensor::randn(&[8, 10], &device);
/// let loss = model.forward(&x).square().mean();
/// optimizer.zero_grad();
/// loss.backward();
/// optimizer.step();
/// ```
pub struct AdamW {
    params: Vec<Tensor>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    t: usize,
    max_grad_norm: Option<f32>,
    /// First and second moment estimates, created on first use.
    moments: Vec<Option<(Tensor, Tensor)>>,
}

impl AdamW {
    /// Create AdamW with betas (0.9, 0.999), eps 1e-8, weight decay 0.01.
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        Self::with_params(params, lr, 0.9, 0.999, 1e-8, 0.01)
    }

    /// Create AdamW with custom hyperparameters.
    pub fn with_params(
        params: Vec<Tensor>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    ) -> Self {
        let num_params = params.len();
        Self {
            params,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            t: 0,
            max_grad_norm: None,
            moments: vec![None; num_params],
        }
    }

    /// Set weight decay.
    pub fn weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Set beta1 (momentum).
    pub fn beta1(mut self, beta1: f32) -> Self {
        self.beta1 = beta1;
        self
    }

    /// Set beta2 (RMS).
    pub fn beta2(mut self, beta2: f32) -> Self {
        self.beta2 = beta2;
        self
    }

    /// Set epsilon.
    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// Clip the global gradient norm to `max_norm` before every step.
    pub fn clip_grad_norm(mut self, max_norm: f32) -> Self {
        self.max_grad_norm = Some(max_norm);
        self
    }

    /// Number of steps taken.
    pub fn steps(&self) -> usize {
        self.t
    }

    /// The parameters being optimized.
    pub fn params(&self) -> &[Tensor] {
        &self.params
    }
}

impl Optimizer for AdamW {
    fn step(&mut self) {
        if let Some(max_norm) = self.max_grad_norm {
            clip_grad_norm(&self.params, max_norm);
        }
        self.t += 1;
        let t = self.t as i32;
        let bias1 = 1.0 - self.beta1.powi(t);
        let bias2 = 1.0 - self.beta2.powi(t);
        no_grad(|| {
            for (param, moments) in self.params.iter().zip(&mut self.moments) {
                let Some(g) = param.grad() else { continue };
                let (m, v) = moments
                    .take()
                    .unwrap_or_else(|| (param.zeros_like(), param.zeros_like()));
                let m = m
                    .mul_scalar(self.beta1)
                    .add(&g.mul_scalar(1.0 - self.beta1));
                let v = v
                    .mul_scalar(self.beta2)
                    .add(&g.square().mul_scalar(1.0 - self.beta2));
                let denom = v.div_scalar(bias2).sqrt().add_scalar(self.eps);
                let update = m.div_scalar(bias1).div(&denom).mul_scalar(self.lr);
                let decayed = param.mul_scalar(1.0 - self.lr * self.weight_decay);
                param.assign(&decayed.sub(&update));
                *moments = Some((m, v));
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

impl std::fmt::Debug for AdamW {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdamW")
            .field("lr", &self.lr)
            .field("beta1", &self.beta1)
            .field("beta2", &self.beta2)
            .field("eps", &self.eps)
            .field("weight_decay", &self.weight_decay)
            .field("t", &self.t)
            .field("num_params", &self.params.len())
            .finish()
    }
}
