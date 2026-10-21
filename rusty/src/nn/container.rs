//! Container Modules
//!
//! Sequential and ModuleList containers.

use super::module::prefixed;
use super::{
    Dropout, Embedding, LayerNorm, LeakyReLU, Linear, Module, MultiHeadAttention, RMSNorm, ReLU,
    SiLU, Sigmoid, Softmax, Tanh, GELU,
};
use crate::tensor::Tensor;
use crate::Device;

macro_rules! layers {
    ($($variant:ident),* $(,)?) => {
        /// A layer that can be added to [`Sequential`].
        // Layers are built once per model, so variant size does not matter.
        #[allow(clippy::large_enum_variant)]
        pub enum Layer {
            $(
                #[doc = concat!("A [`", stringify!($variant), "`] layer.")]
                $variant($variant),
            )*
            /// Any other module.
            Custom(Box<dyn Module>),
        }

        impl Layer {
            fn module(&self) -> &dyn Module {
                match self {
                    $(Layer::$variant(l) => l,)*
                    Layer::Custom(l) => l.as_ref(),
                }
            }

            fn module_mut(&mut self) -> &mut dyn Module {
                match self {
                    $(Layer::$variant(l) => l,)*
                    Layer::Custom(l) => l.as_mut(),
                }
            }
        }

        $(
            impl From<$variant> for Layer {
                fn from(l: $variant) -> Self {
                    Layer::$variant(l)
                }
            }
        )*
    };
}

layers!(
    Linear,
    ReLU,
    LeakyReLU,
    SiLU,
    GELU,
    Sigmoid,
    Tanh,
    Softmax,
    Dropout,
    LayerNorm,
    RMSNorm,
    Embedding,
    MultiHeadAttention,
);

impl From<Box<dyn Module>> for Layer {
    fn from(m: Box<dyn Module>) -> Self {
        Layer::Custom(m)
    }
}

impl Module for Layer {
    fn forward(&self, x: &Tensor) -> Tensor {
        self.module().forward(x)
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.module().parameters()
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        self.module().named_parameters()
    }

    fn set_training(&mut self, training: bool) {
        self.module_mut().set_training(training);
    }
}

/// Sequential container - runs layers in order.
///
/// Parameters are named `"<index>.<name>"`, e.g. `"0.weight"`.
///
/// # Example
/// ```rust
/// use rusty::prelude::*;
///
/// let device = Device::cpu();
/// let model = nn::Sequential::new(&device)
///     .add(nn::Linear::new(784, 256, &device))
///     .add(nn::ReLU)
///     .add(nn::Dropout::new(0.1))
///     .add(nn::Linear::new(256, 10, &device));
///
/// let x = Tensor::randn(&[32, 784], &device);
/// let y = model.forward(&x);
/// assert_eq!(y.shape, vec![32, 10]);
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

    /// Append a layer.
    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, layer: impl Into<Layer>) -> Self {
        self.layers.push(layer.into());
        self
    }

    /// Append a custom module.
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

    /// The layer at `index`.
    pub fn get(&self, index: usize) -> Option<&Layer> {
        self.layers.get(index)
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

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        self.layers
            .iter()
            .enumerate()
            .flat_map(|(i, l)| prefixed(&i.to_string(), l.named_parameters()))
            .collect()
    }

    fn set_training(&mut self, training: bool) {
        for layer in &mut self.layers {
            layer.set_training(training);
        }
    }
}

impl std::fmt::Debug for Sequential {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sequential")
            .field("num_layers", &self.layers.len())
            .finish()
    }
}

/// ModuleList - an indexable list of modules (no forward of its own).
///
/// Parameters are named `"<index>.<name>"`.
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

    /// Parameters of every module, in order.
    pub fn parameters(&self) -> Vec<Tensor> {
        self.modules.iter().flat_map(|m| m.parameters()).collect()
    }

    /// Named parameters of every module, prefixed by its index.
    pub fn named_parameters(&self) -> Vec<(String, Tensor)> {
        self.modules
            .iter()
            .enumerate()
            .flat_map(|(i, m)| prefixed(&i.to_string(), m.named_parameters()))
            .collect()
    }

    /// Switch every module between training and evaluation.
    pub fn set_training(&mut self, training: bool) {
        for module in &mut self.modules {
            module.set_training(training);
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn sequential_names_and_mode_propagate() {
        let d = Device::cpu();
        let mut model = nn::Sequential::new(&d)
            .add(nn::Linear::new(3, 4, &d))
            .add(nn::Dropout::new(0.5))
            .add(nn::LayerNorm::new(&[4], &d));
        let names: Vec<String> = model
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        assert_eq!(names, ["0.weight", "0.bias", "2.weight", "2.bias"]);

        model.eval();
        let x = Tensor::randn(&[2, 3], &d);
        assert_eq!(model.forward(&x).to_vec(), model.forward(&x).to_vec());
    }

    #[test]
    fn attention_shapes_and_gradients() {
        let d = Device::cpu();
        let attn = nn::MultiHeadAttention::new(8, 2, &d).causal(true);
        let x = Tensor::randn(&[5, 8], &d).requires_grad_();
        let y = attn.forward(&x);
        assert_eq!(y.shape, vec![5, 8]);
        y.sum().backward();
        assert!(x.grad().is_some());
        assert!(attn.parameters().iter().all(|p| p.grad().is_some()));
        assert_eq!(attn.named_parameters()[0].0, "q_proj.weight");
    }

    #[test]
    fn causal_attention_output_ignores_later_tokens() {
        let d = Device::cpu();
        let attn = nn::MultiHeadAttention::new(8, 2, &d).causal(true);
        let x = Tensor::randn(&[1, 4, 8], &d);
        let full = attn.forward(&x).to_vec();
        let prefix = attn.forward(&x.narrow(1, 0, 2)).to_vec();
        for (a, b) in prefix.iter().zip(&full[..16]) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn embedding_forward_keeps_id_shape() {
        let d = Device::cpu();
        let emb = nn::Embedding::new(10, 3, &d);
        let ids = Tensor::from_data(&[1.0, 2.0, 3.0, 1.0], &[2, 2], &d);
        let out = emb.forward(&ids);
        assert_eq!(out.shape, vec![2, 2, 3]);
        out.sum().backward();
        let g = emb.weight.grad().unwrap().to_vec();
        assert_eq!(&g[3..6], &[2.0, 2.0, 2.0]); // id 1 used twice
        assert_eq!(&g[0..3], &[0.0, 0.0, 0.0]);
    }
}
