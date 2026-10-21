//! Dropout Layer

use super::Module;
use crate::tensor::Tensor;

/// Dropout: during training, zeroes each element with probability `p` and
/// scales the rest by `1 / (1 - p)`; the identity in evaluation mode.
///
/// # Example
/// ```rust
/// use ferrotensor::prelude::*;
///
/// let device = Device::cpu();
/// let mut drop = nn::Dropout::new(0.5);
/// drop.eval();
/// let x = Tensor::ones(&[4], &device);
/// assert_eq!(drop.forward(&x).to_vec(), x.to_vec());
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Dropout {
    /// Probability of zeroing an element.
    pub p: f32,
    /// Training mode (dropout active).
    pub training: bool,
}

impl Dropout {
    /// Create a Dropout layer in training mode.
    ///
    /// # Panics
    /// Panics unless `0 <= p < 1`.
    pub fn new(p: f32) -> Self {
        assert!(
            (0.0..1.0).contains(&p),
            "dropout probability must be in [0, 1), got {p}"
        );
        Self { p, training: true }
    }
}

impl Module for Dropout {
    fn forward(&self, x: &Tensor) -> Tensor {
        crate::functional::dropout(x, self.p, self.training)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}
