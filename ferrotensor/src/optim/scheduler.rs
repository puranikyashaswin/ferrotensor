//! Learning-rate schedules.
//!
//! A schedule is a pure function of the step number; call
//! [`LrScheduler::apply`] once per step to set the optimizer's rate.
//!
//! # Example
//! ```rust
//! use ferrotensor::prelude::*;
//! use ferrotensor::optim::{LrScheduler, WarmupCosineLr};
//!
//! let device = Device::cpu();
//! let model = nn::Linear::new(4, 2, &device);
//! let mut opt = optim::AdamW::new(model.parameters(), 1e-3);
//! let schedule = WarmupCosineLr::new(1e-3, 1e-5, 100, 1000);
//! for step in 0..3 {
//!     schedule.apply(&mut opt, step);
//!     // forward, backward, opt.step() ...
//! }
//! assert!((opt.lr() - 3e-5).abs() < 1e-9);
//! ```

use std::f32::consts::PI;

use super::Optimizer;

/// A learning rate as a function of the (0-based) step.
pub trait LrScheduler {
    /// Learning rate for `step`.
    fn lr_at(&self, step: usize) -> f32;

    /// Set `optimizer`'s learning rate for `step` and return it.
    fn apply(&self, optimizer: &mut dyn Optimizer, step: usize) -> f32 {
        let lr = self.lr_at(step);
        optimizer.set_lr(lr);
        lr
    }
}

/// A fixed learning rate.
#[derive(Debug, Clone, Copy)]
pub struct ConstantLr {
    /// The rate.
    pub lr: f32,
}

impl LrScheduler for ConstantLr {
    fn lr_at(&self, _step: usize) -> f32 {
        self.lr
    }
}

/// Multiply the rate by `gamma` every `step_size` steps.
#[derive(Debug, Clone, Copy)]
pub struct StepLr {
    /// Initial rate.
    pub base_lr: f32,
    /// Steps between decays.
    pub step_size: usize,
    /// Decay factor.
    pub gamma: f32,
}

impl LrScheduler for StepLr {
    fn lr_at(&self, step: usize) -> f32 {
        self.base_lr * self.gamma.powi((step / self.step_size.max(1)) as i32)
    }
}

/// Cosine decay from `base_lr` to `min_lr` over `total_steps`, then flat.
#[derive(Debug, Clone, Copy)]
pub struct CosineAnnealingLr {
    /// Initial rate.
    pub base_lr: f32,
    /// Final rate.
    pub min_lr: f32,
    /// Length of the decay.
    pub total_steps: usize,
}

impl LrScheduler for CosineAnnealingLr {
    fn lr_at(&self, step: usize) -> f32 {
        cosine(self.base_lr, self.min_lr, step, self.total_steps)
    }
}

/// Linear warmup from near 0 to `peak_lr` over `warmup_steps`, then cosine
/// decay to `min_lr` at `total_steps` (the usual LLM schedule).
#[derive(Debug, Clone, Copy)]
pub struct WarmupCosineLr {
    /// Rate at the end of warmup.
    pub peak_lr: f32,
    /// Final rate.
    pub min_lr: f32,
    /// Warmup length.
    pub warmup_steps: usize,
    /// Total schedule length, including warmup.
    pub total_steps: usize,
}

impl WarmupCosineLr {
    /// Create the schedule.
    pub fn new(peak_lr: f32, min_lr: f32, warmup_steps: usize, total_steps: usize) -> Self {
        Self {
            peak_lr,
            min_lr,
            warmup_steps,
            total_steps,
        }
    }
}

impl LrScheduler for WarmupCosineLr {
    fn lr_at(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            return self.peak_lr * (step + 1) as f32 / self.warmup_steps as f32;
        }
        let decay_steps = self.total_steps.saturating_sub(self.warmup_steps);
        cosine(
            self.peak_lr,
            self.min_lr,
            step - self.warmup_steps,
            decay_steps,
        )
    }
}

/// Linear warmup to `peak_lr`, then linear decay to `end_lr` at
/// `total_steps`.
#[derive(Debug, Clone, Copy)]
pub struct WarmupLinearLr {
    /// Rate at the end of warmup.
    pub peak_lr: f32,
    /// Final rate.
    pub end_lr: f32,
    /// Warmup length.
    pub warmup_steps: usize,
    /// Total schedule length, including warmup.
    pub total_steps: usize,
}

impl LrScheduler for WarmupLinearLr {
    fn lr_at(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            return self.peak_lr * (step + 1) as f32 / self.warmup_steps as f32;
        }
        let decay_steps = self.total_steps.saturating_sub(self.warmup_steps).max(1);
        let progress = ((step - self.warmup_steps) as f32 / decay_steps as f32).min(1.0);
        self.peak_lr + (self.end_lr - self.peak_lr) * progress
    }
}

fn cosine(start: f32, end: f32, step: usize, steps: usize) -> f32 {
    let progress = if steps == 0 {
        1.0
    } else {
        (step as f32 / steps as f32).min(1.0)
    };
    end + (start - end) * 0.5 * (1.0 + (PI * progress).cos())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-7
    }

    #[test]
    fn warmup_cosine_shape() {
        let s = WarmupCosineLr::new(1.0, 0.1, 10, 110);
        assert!(close(s.lr_at(0), 0.1));
        assert!(close(s.lr_at(9), 1.0));
        assert!(close(s.lr_at(10), 1.0));
        assert!(close(s.lr_at(60), 0.55));
        assert!(close(s.lr_at(110), 0.1));
        assert!(close(s.lr_at(500), 0.1));
        let decreasing = (10..110).all(|t| s.lr_at(t + 1) <= s.lr_at(t));
        assert!(decreasing);
    }

    #[test]
    fn step_and_linear_schedules() {
        let s = StepLr {
            base_lr: 1.0,
            step_size: 10,
            gamma: 0.5,
        };
        assert!(close(s.lr_at(9), 1.0) && close(s.lr_at(10), 0.5) && close(s.lr_at(25), 0.25));
        let l = WarmupLinearLr {
            peak_lr: 1.0,
            end_lr: 0.0,
            warmup_steps: 0,
            total_steps: 4,
        };
        assert!(close(l.lr_at(2), 0.5) && close(l.lr_at(9), 0.0));
        let c = CosineAnnealingLr {
            base_lr: 1.0,
            min_lr: 0.0,
            total_steps: 0,
        };
        assert!(close(c.lr_at(0), 0.0));
    }
}
