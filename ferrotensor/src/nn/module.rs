//! Module Trait
//!
//! Base trait for all neural network modules.

use crate::tensor::Tensor;
use crate::Device;

/// Base trait for all neural network modules.
///
/// Implement [`forward`](Module::forward) and
/// [`parameters`](Module::parameters); override
/// [`named_parameters`](Module::named_parameters) to give parameters stable
/// names for [`state_dict`](crate::serialize::state_dict), and
/// [`set_training`](Module::set_training) when the module (or a child)
/// behaves differently in training, like dropout.
///
/// # Example
/// ```rust
/// use ferrotensor::prelude::*;
///
/// struct MyModel {
///     fc1: nn::Linear,
///     fc2: nn::Linear,
/// }
///
/// impl Module for MyModel {
///     fn forward(&self, x: &Tensor) -> Tensor {
///         let x = self.fc1.forward(x).relu();
///         self.fc2.forward(&x)
///     }
///
///     fn parameters(&self) -> Vec<Tensor> {
///         let mut p = self.fc1.parameters();
///         p.extend(self.fc2.parameters());
///         p
///     }
/// }
///
/// let device = Device::cpu();
/// let model = MyModel {
///     fc1: nn::Linear::new(4, 8, &device),
///     fc2: nn::Linear::new(8, 2, &device),
/// };
/// let y = model.forward(&Tensor::randn(&[3, 4], &device));
/// assert_eq!(y.shape, vec![3, 2]);
/// assert_eq!(model.num_parameters(), 4 * 8 + 8 + 8 * 2 + 2);
/// ```
pub trait Module: Send + Sync {
    /// Forward pass.
    fn forward(&self, x: &Tensor) -> Tensor;

    /// All trainable parameters.
    fn parameters(&self) -> Vec<Tensor>;

    /// Parameters with names that are stable across runs (used as
    /// state-dict keys). Defaults to their positions: `"0"`, `"1"`, ...
    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        self.parameters()
            .into_iter()
            .enumerate()
            .map(|(i, p)| (i.to_string(), p))
            .collect()
    }

    /// Switch between training (`true`) and evaluation behaviour.
    fn set_training(&mut self, _training: bool) {}

    /// Set training mode.
    fn train(&mut self) {
        self.set_training(true);
    }

    /// Set evaluation mode.
    fn eval(&mut self) {
        self.set_training(false);
    }

    /// Zero gradients for all parameters.
    fn zero_grad(&self) {
        for param in self.parameters() {
            param.zero_grad();
        }
    }

    /// Count total number of parameters.
    fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.numel()).sum()
    }
}

/// Prefix each name with `prefix.` (for nesting modules).
pub fn prefixed(prefix: &str, named: Vec<(String, Tensor)>) -> Vec<(String, Tensor)> {
    named
        .into_iter()
        .map(|(name, p)| (format!("{prefix}.{name}"), p))
        .collect()
}

/// Trait for modules that can be initialized with a device.
pub trait ModuleInit {
    /// Create the module on `device`.
    fn new(device: &Device) -> Self;
}
