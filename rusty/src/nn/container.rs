//! Container Modules
//!
//! Sequential and ModuleList containers.

use super::{LayerNorm, Linear, Module, RMSNorm, ReLU, SiLU, Sigmoid, Tanh, GELU};
use crate::tensor::Tensor;
use crate::Device;

/// A layer that can be added to Sequential.
pub enum Layer {
    Linear(Linear),
    ReLU(ReLU),
    SiLU(SiLU),
    GELU(GELU),
    Sigmoid(Sigmoid),
    Tanh(Tanh),
    LayerNorm(LayerNorm),
    RMSNorm(RMSNorm),
    Custom(Box<dyn Module>),
}

impl Module for Layer {
    fn forward(&self, x: &Tensor) -> Tensor {
        match self {
            Layer::Linear(l) => l.forward(x),
            Layer::ReLU(l) => l.forward(x),
            Layer::SiLU(l) => l.forward(x),
            Layer::GELU(l) => l.forward(x),
            Layer::Sigmoid(l) => l.forward(x),
            Layer::Tanh(l) => l.forward(x),
            Layer::LayerNorm(l) => l.forward(x),
            Layer::RMSNorm(l) => l.forward(x),
            Layer::Custom(l) => l.forward(x),
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        match self {
            Layer::Linear(l) => l.parameters(),
            Layer::ReLU(l) => l.parameters(),
            Layer::SiLU(l) => l.parameters(),
            Layer::GELU(l) => l.parameters(),
            Layer::Sigmoid(l) => l.parameters(),
            Layer::Tanh(l) => l.parameters(),
            Layer::LayerNorm(l) => l.parameters(),
            Layer::RMSNorm(l) => l.parameters(),
            Layer::Custom(l) => l.parameters(),
        }
    }
}

/// Sequential container - runs layers in order.
///
/// # Example
/// ```rust,no_run
/// use rusty::prelude::*;
///
/// let device = Device::new();
/// let model = nn::Sequential::new(&device)
///     .add(nn::Linear::new(784, 256, &device))
///     .add(nn::ReLU)
///     .add(nn::Linear::new(256, 10, &device));
///
/// let x = Tensor::randn(&[32, 784], &device);
/// let y = model.forward(&x);
/// ```
pub struct Sequential {
    layers: Vec<Layer>,
    #[allow(dead_code)]
    device: Device,
}

impl Sequential {
    /// Create a new empty Sequential container.
    pub fn new(device: &Device) -> Self {
        Self {
            layers: Vec::new(),
            device: device.clone(),
        }
    }

    /// Add a Linear layer.
    pub fn add(mut self, layer: impl Into<Layer>) -> Self {
        self.layers.push(layer.into());
        self
    }

    /// Add a custom module.
    pub fn add_module(mut self, module: Box<dyn Module>) -> Self {
        self.layers.push(Layer::Custom(module));
        self
    }

    /// Get number of layers.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

impl Module for Sequential {
    fn forward(&self, x: &Tensor) -> Tensor {
        let mut out = x.clone();
        for layer in &self.layers {
            out = layer.forward(&out);
        }
        out
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
}

impl std::fmt::Debug for Sequential {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sequential")
            .field("num_layers", &self.layers.len())
            .finish()
    }
}

// Implement From for common types
impl From<Linear> for Layer {
    fn from(l: Linear) -> Self {
        Layer::Linear(l)
    }
}

impl From<ReLU> for Layer {
    fn from(l: ReLU) -> Self {
        Layer::ReLU(l)
    }
}

impl From<SiLU> for Layer {
    fn from(l: SiLU) -> Self {
        Layer::SiLU(l)
    }
}

impl From<GELU> for Layer {
    fn from(l: GELU) -> Self {
        Layer::GELU(l)
    }
}

impl From<Sigmoid> for Layer {
    fn from(l: Sigmoid) -> Self {
        Layer::Sigmoid(l)
    }
}

impl From<Tanh> for Layer {
    fn from(l: Tanh) -> Self {
        Layer::Tanh(l)
    }
}

impl From<LayerNorm> for Layer {
    fn from(l: LayerNorm) -> Self {
        Layer::LayerNorm(l)
    }
}

impl From<RMSNorm> for Layer {
    fn from(l: RMSNorm) -> Self {
        Layer::RMSNorm(l)
    }
}

/// ModuleList - stores modules in a list.
pub struct ModuleList {
    modules: Vec<Box<dyn Module>>,
}

impl ModuleList {
    /// Create an empty ModuleList.
    pub fn new() -> Self {
        Self {
            modules: Vec::new(),
        }
    }

    /// Add a module.
    pub fn push(&mut self, module: Box<dyn Module>) {
        self.modules.push(module);
    }

    /// Get number of modules.
    pub fn len(&self) -> usize {
        self.modules.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    /// Iterate over modules.
    pub fn iter(&self) -> impl Iterator<Item = &Box<dyn Module>> {
        self.modules.iter()
    }
}

impl Default for ModuleList {
    fn default() -> Self {
        Self::new()
    }
}

impl std::ops::Index<usize> for ModuleList {
    type Output = Box<dyn Module>;
    fn index(&self, index: usize) -> &Self::Output {
        &self.modules[index]
    }
}
