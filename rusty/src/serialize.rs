//! Model Serialization
//!
//! Save and load model weights using safetensors format.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use crate::tensor::Tensor;
use crate::Device;

/// Error type for serialization operations.
#[derive(Debug, thiserror::Error)]
pub enum SerializeError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Tensor not found: {0}")]
    TensorNotFound(String),
}

/// Save tensors to a custom binary format.
///
/// Format:
/// - 4 bytes: number of tensors (u32)
/// - For each tensor:
///   - 4 bytes: name length (u32)
///   - N bytes: name (utf8)
///   - 4 bytes: number of dimensions (u32)
///   - M * 4 bytes: shape (u32 each)
///   - 4 bytes: data length in floats (u32)
///   - K * 4 bytes: data (f32 each)
pub fn save_tensors<P: AsRef<Path>>(
    tensors: &HashMap<String, Tensor>,
    path: P,
) -> Result<(), SerializeError> {
    let mut file = File::create(path)?;

    // Write number of tensors
    let num_tensors = tensors.len() as u32;
    file.write_all(&num_tensors.to_le_bytes())?;

    for (name, tensor) in tensors {
        // Write name
        let name_bytes = name.as_bytes();
        let name_len = name_bytes.len() as u32;
        file.write_all(&name_len.to_le_bytes())?;
        file.write_all(name_bytes)?;

        // Write shape
        let ndim = tensor.shape.len() as u32;
        file.write_all(&ndim.to_le_bytes())?;
        for &dim in &tensor.shape {
            file.write_all(&(dim as u32).to_le_bytes())?;
        }

        // Write data
        let data = tensor.to_vec();
        let data_len = data.len() as u32;
        file.write_all(&data_len.to_le_bytes())?;
        for val in &data {
            file.write_all(&val.to_le_bytes())?;
        }
    }

    Ok(())
}

/// Load tensors from a custom binary format.
pub fn load_tensors<P: AsRef<Path>>(
    path: P,
    device: &Device,
) -> Result<HashMap<String, Tensor>, SerializeError> {
    let mut file = File::open(path)?;
    let mut tensors = HashMap::new();

    // Read number of tensors
    let mut buf4 = [0u8; 4];
    file.read_exact(&mut buf4)?;
    let num_tensors = u32::from_le_bytes(buf4);

    for _ in 0..num_tensors {
        // Read name
        file.read_exact(&mut buf4)?;
        let name_len = u32::from_le_bytes(buf4) as usize;
        let mut name_bytes = vec![0u8; name_len];
        file.read_exact(&mut name_bytes)?;
        let name = String::from_utf8_lossy(&name_bytes).to_string();

        // Read shape
        file.read_exact(&mut buf4)?;
        let ndim = u32::from_le_bytes(buf4) as usize;
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            file.read_exact(&mut buf4)?;
            shape.push(u32::from_le_bytes(buf4) as usize);
        }

        // Read data
        file.read_exact(&mut buf4)?;
        let data_len = u32::from_le_bytes(buf4) as usize;
        let mut data = Vec::with_capacity(data_len);
        for _ in 0..data_len {
            file.read_exact(&mut buf4)?;
            data.push(f32::from_le_bytes(buf4));
        }

        let tensor = Tensor::from_data(&data, &shape, device);
        tensors.insert(name, tensor);
    }

    Ok(tensors)
}

/// Trait for modules that can be saved/loaded.
pub trait Saveable {
    /// Get all parameters as a named dictionary.
    fn state_dict(&self) -> HashMap<String, Tensor>;

    /// Load parameters from a named dictionary.
    fn load_state_dict(
        &mut self,
        state_dict: HashMap<String, Tensor>,
    ) -> Result<(), SerializeError>;
}

/// Helper to save model parameters.
pub fn save_model<P: AsRef<Path>>(model: &impl Saveable, path: P) -> Result<(), SerializeError> {
    let state_dict = model.state_dict();
    save_tensors(&state_dict, path)
}

/// Helper to load model parameters.
pub fn load_model<P: AsRef<Path>>(
    model: &mut impl Saveable,
    path: P,
    device: &Device,
) -> Result<(), SerializeError> {
    let state_dict = load_tensors(path, device)?;
    model.load_state_dict(state_dict)
}
