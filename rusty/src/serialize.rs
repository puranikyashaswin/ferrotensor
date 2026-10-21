//! Model Serialization
//!
//! Save and load tensors in the [safetensors] format, which PyTorch and the
//! Hugging Face ecosystem read and write, and round-trip a module's
//! parameters through a state dict keyed by
//! [`Module::named_parameters`].
//!
//! [safetensors]: https://github.com/huggingface/safetensors
//!
//! # Example
//! ```rust
//! use rusty::prelude::*;
//! use rusty::serialize::{load_module, save_module};
//!
//! let device = Device::cpu();
//! let model = nn::Sequential::new(&device)
//!     .add(nn::Linear::new(4, 8, &device))
//!     .add(nn::ReLU)
//!     .add(nn::Linear::new(8, 2, &device));
//!
//! let path = std::env::temp_dir().join("rusty_doc_model.safetensors");
//! save_module(&model, &path).unwrap();
//!
//! let restored = nn::Sequential::new(&device)
//!     .add(nn::Linear::new(4, 8, &device))
//!     .add(nn::ReLU)
//!     .add(nn::Linear::new(8, 2, &device));
//! load_module(&restored, &path).unwrap();
//! let x = Tensor::randn(&[3, 4], &device);
//! assert_eq!(model.forward(&x).to_vec(), restored.forward(&x).to_vec());
//! # std::fs::remove_file(&path).ok();
//! ```

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use safetensors::tensor::{Dtype, SafeTensors, TensorView};

use crate::nn::Module;
use crate::tensor::Tensor;
use crate::Device;

/// Error type for serialization operations.
#[derive(Debug, thiserror::Error)]
pub enum SerializeError {
    /// Reading or writing the file failed.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// The file is not valid safetensors.
    #[error("safetensors error: {0}")]
    Format(String),
    /// A tensor the module needs is missing from the file.
    #[error("tensor not found: {0}")]
    TensorNotFound(String),
    /// The file has tensors the module does not.
    #[error("unexpected tensors: {0:?}")]
    UnexpectedTensors(Vec<String>),
    /// A tensor's shape differs from the module's parameter.
    #[error("shape mismatch for {name}: expected {expected:?}, found {found:?}")]
    ShapeMismatch {
        /// Tensor name.
        name: String,
        /// Shape of the module's parameter.
        expected: Vec<usize>,
        /// Shape in the file.
        found: Vec<usize>,
    },
    /// A tensor's dtype cannot be converted to `f32`.
    #[error("unsupported dtype {dtype} for {name}")]
    UnsupportedDtype {
        /// Tensor name.
        name: String,
        /// The dtype found.
        dtype: String,
    },
}

impl From<safetensors::SafeTensorError> for SerializeError {
    fn from(e: safetensors::SafeTensorError) -> Self {
        Self::Format(format!("{e:?}"))
    }
}

/// Save named tensors to a safetensors file (as `F32`).
pub fn save_tensors<P: AsRef<Path>>(
    tensors: &HashMap<String, Tensor>,
    path: P,
) -> Result<(), SerializeError> {
    let bytes: BTreeMap<&str, (Vec<usize>, Vec<u8>)> = tensors
        .iter()
        .map(|(name, t)| {
            let data: Vec<u8> = t.to_vec().iter().flat_map(|v| v.to_le_bytes()).collect();
            (name.as_str(), (t.shape.clone(), data))
        })
        .collect();
    let views = bytes
        .iter()
        .map(|(name, (shape, data))| Ok((*name, TensorView::new(Dtype::F32, shape.clone(), data)?)))
        .collect::<Result<Vec<_>, SerializeError>>()?;
    safetensors::serialize_to_file(views, &None, path.as_ref())?;
    Ok(())
}

/// Load every tensor in a safetensors file onto `device`.
///
/// `F32`, `F16`, `BF16` and `F64` tensors are converted to `f32`.
pub fn load_tensors<P: AsRef<Path>>(
    path: P,
    device: &Device,
) -> Result<HashMap<String, Tensor>, SerializeError> {
    let bytes = std::fs::read(path)?;
    let file = SafeTensors::deserialize(&bytes)?;
    file.tensors()
        .into_iter()
        .map(|(name, view)| {
            let data = to_f32(&name, view.dtype(), view.data())?;
            let shape = view.shape().to_vec();
            Ok((name, Tensor::from_vec(data, &shape, device)))
        })
        .collect()
}

fn to_f32(name: &str, dtype: Dtype, bytes: &[u8]) -> Result<Vec<f32>, SerializeError> {
    let words = |n: usize| bytes.chunks_exact(n);
    Ok(match dtype {
        Dtype::F32 => words(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        Dtype::F64 => words(8)
            .map(|b| f64::from_le_bytes(b.try_into().expect("8-byte chunk")) as f32)
            .collect(),
        Dtype::BF16 => words(2)
            .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
            .collect(),
        Dtype::F16 => words(2)
            .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
            .collect(),
        other => {
            return Err(SerializeError::UnsupportedDtype {
                name: name.to_string(),
                dtype: format!("{other:?}"),
            })
        }
    })
}

/// IEEE 754 half to single precision (exact).
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = match (exp, frac) {
        (0, 0) => sign,
        (0, _) => {
            // Subnormal: value = frac * 2^-24.
            let value = frac as f32 * 2f32.powi(-24);
            return if sign != 0 { -value } else { value };
        }
        (0x1f, 0) => sign | 0x7f80_0000,
        (0x1f, _) => sign | 0x7fc0_0000,
        _ => sign | ((exp + 127 - 15) << 23) | (frac << 13),
    };
    f32::from_bits(bits)
}

/// A module's parameters keyed by [`Module::named_parameters`].
///
/// The tensors share storage with the module (no copy).
pub fn state_dict<M: Module + ?Sized>(module: &M) -> HashMap<String, Tensor> {
    module.named_parameters().into_iter().collect()
}

/// Copy `state` into the module's parameters in place.
///
/// Strict: every parameter must be present with the right shape, and no
/// extra tensors are allowed. Values are copied onto each parameter's own
/// device, so optimizers holding the parameters keep working.
pub fn load_state_dict<M: Module + ?Sized>(
    module: &M,
    state: &HashMap<String, Tensor>,
) -> Result<(), SerializeError> {
    let params = module.named_parameters();
    let mut unexpected: Vec<String> = state
        .keys()
        .filter(|k| !params.iter().any(|(name, _)| name == *k))
        .cloned()
        .collect();
    if !unexpected.is_empty() {
        unexpected.sort();
        return Err(SerializeError::UnexpectedTensors(unexpected));
    }
    for (name, param) in &params {
        let src = state
            .get(name)
            .ok_or_else(|| SerializeError::TensorNotFound(name.clone()))?;
        if src.shape != param.shape {
            return Err(SerializeError::ShapeMismatch {
                name: name.clone(),
                expected: param.shape.clone(),
                found: src.shape.clone(),
            });
        }
    }
    for (name, param) in &params {
        param.assign(&state[name]);
    }
    Ok(())
}

/// Save a module's parameters to a safetensors file.
pub fn save_module<M: Module + ?Sized, P: AsRef<Path>>(
    module: &M,
    path: P,
) -> Result<(), SerializeError> {
    save_tensors(&state_dict(module), path)
}

/// Load a module's parameters from a safetensors file (strict, in place).
pub fn load_module<M: Module + ?Sized, P: AsRef<Path>>(
    module: &M,
    path: P,
) -> Result<(), SerializeError> {
    let state = load_tensors(path, &Device::cpu())?;
    load_state_dict(module, &state)
}

/// Trait for types that can be saved/loaded as a state dict.
///
/// Every [`Module`] implements it through its named parameters.
pub trait Saveable {
    /// Get all parameters as a named dictionary.
    fn state_dict(&self) -> HashMap<String, Tensor>;

    /// Load parameters from a named dictionary.
    fn load_state_dict(
        &mut self,
        state_dict: HashMap<String, Tensor>,
    ) -> Result<(), SerializeError>;
}

impl<M: Module> Saveable for M {
    fn state_dict(&self) -> HashMap<String, Tensor> {
        state_dict(self)
    }

    fn load_state_dict(
        &mut self,
        state_dict: HashMap<String, Tensor>,
    ) -> Result<(), SerializeError> {
        load_state_dict(self, &state_dict)
    }
}

/// Save a [`Saveable`] to a safetensors file.
pub fn save_model<P: AsRef<Path>>(model: &impl Saveable, path: P) -> Result<(), SerializeError> {
    save_tensors(&model.state_dict(), path)
}

/// Load a [`Saveable`] from a safetensors file.
pub fn load_model<P: AsRef<Path>>(
    model: &mut impl Saveable,
    path: P,
    device: &Device,
) -> Result<(), SerializeError> {
    let state_dict = load_tensors(path, device)?;
    model.load_state_dict(state_dict)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn;

    fn temp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("rusty_{}_{name}.safetensors", std::process::id()))
    }

    #[test]
    fn tensors_round_trip() {
        let d = Device::cpu();
        let mut map = HashMap::new();
        map.insert("a".to_string(), Tensor::from_data(&[1.5, -2.0], &[2], &d));
        map.insert(
            "b.c".to_string(),
            Tensor::arange(0.0, 6.0, 1.0, &d).reshape(&[2, 3]),
        );
        map.insert("scalar".to_string(), Tensor::scalar(7.0, &d));
        let path = temp("tensors");
        save_tensors(&map, &path).unwrap();
        let loaded = load_tensors(&path, &d).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(loaded.len(), 3);
        for (k, v) in &map {
            assert_eq!(loaded[k].shape, v.shape);
            assert_eq!(loaded[k].to_vec(), v.to_vec());
        }
    }

    fn model(d: &Device) -> nn::Sequential {
        nn::Sequential::new(d)
            .add(nn::Embedding::new(10, 4, d))
            .add(nn::LayerNorm::new(&[4], d))
            .add(nn::Linear::new(4, 3, d))
    }

    #[test]
    fn module_round_trip_restores_outputs() {
        let d = Device::cpu();
        let (a, b) = (model(&d), model(&d));
        let path = temp("module");
        save_module(&a, &path).unwrap();
        load_module(&b, &path).unwrap();
        std::fs::remove_file(&path).ok();
        let ids = Tensor::from_data(&[1.0, 7.0], &[2], &d);
        assert_eq!(a.forward(&ids).to_vec(), b.forward(&ids).to_vec());
    }

    #[test]
    fn strict_loading_reports_problems() {
        let d = Device::cpu();
        let m = model(&d);
        let mut state = state_dict(&m);
        state.remove("2.bias");
        assert!(matches!(
            load_state_dict(&m, &state),
            Err(SerializeError::TensorNotFound(name)) if name == "2.bias"
        ));

        let mut state = state_dict(&m);
        state.insert("extra".to_string(), Tensor::zeros(&[1], &d));
        assert!(matches!(
            load_state_dict(&m, &state),
            Err(SerializeError::UnexpectedTensors(_))
        ));

        let mut state = state_dict(&m);
        state.insert("2.bias".to_string(), Tensor::zeros(&[5], &d));
        assert!(matches!(
            load_state_dict(&m, &state),
            Err(SerializeError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn half_precision_conversions() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert!(f16_to_f32(0x7c00).is_infinite());
        assert!(f16_to_f32(0x7e00).is_nan());
        let bf = to_f32("x", Dtype::BF16, &0x3fc0u16.to_le_bytes()).unwrap();
        assert_eq!(bf, vec![1.5]);
    }
}
