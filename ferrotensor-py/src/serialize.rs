//! `ferrotensor.save` / `ferrotensor.load` (safetensors files).

use std::collections::HashMap;
use std::path::PathBuf;

use ferrotensor::serialize::{self, SerializeError};
use ferrotensor::Tensor;
use pyo3::prelude::*;

use crate::device::resolve_device;
use crate::error::{guard, value_error};
use crate::tensor::PyTensor;

fn to_py_err(e: SerializeError) -> PyErr {
    match e {
        SerializeError::Io(io) => io.into(),
        other => value_error(other.to_string()),
    }
}

/// Save a `dict[str, Tensor]` (e.g. `model.state_dict()`) as safetensors.
#[pyfunction]
pub(crate) fn save(tensors: HashMap<String, PyRef<'_, PyTensor>>, path: PathBuf) -> PyResult<()> {
    let tensors: HashMap<String, Tensor> = tensors
        .into_iter()
        .map(|(name, t)| (name, t.inner.clone()))
        .collect();
    guard(|| serialize::save_tensors(&tensors, &path))?.map_err(to_py_err)
}

/// Load every tensor in a safetensors file (F32, F16, BF16 or F64) as float32.
#[pyfunction]
#[pyo3(signature = (path, *, device = None))]
pub(crate) fn load(
    path: PathBuf,
    device: Option<&Bound<'_, PyAny>>,
) -> PyResult<HashMap<String, PyTensor>> {
    let device = resolve_device(device)?;
    let tensors = guard(|| serialize::load_tensors(&path, &device))?.map_err(to_py_err)?;
    Ok(tensors.into_iter().map(|(k, t)| (k, t.into())).collect())
}
