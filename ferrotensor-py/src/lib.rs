//! Python bindings for ferrotensor.

use ferrotensor::{Device, Tensor};
use pyo3::prelude::*;

/// A tensor.
#[pyclass(name = "Tensor", module = "ferrotensor")]
struct PyTensor {
    inner: Tensor,
}

#[pymethods]
impl PyTensor {
    #[new]
    fn new(data: Vec<f32>) -> Self {
        let n = data.len();
        Self {
            inner: Tensor::from_vec(data, &[n], &Device::cpu()),
        }
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape.clone()
    }

    fn tolist(&self) -> Vec<f32> {
        self.inner.to_vec()
    }
}

#[pymodule]
fn _ferrotensor(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    Ok(())
}
