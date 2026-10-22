//! `ferrotensor.Device`.

use std::sync::OnceLock;

use ferrotensor::Device;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::error::{guard, runtime_error, value_error};

/// The GPU is opened once per process and shared by every `Device("gpu")`.
static GPU: OnceLock<Option<Device>> = OnceLock::new();

fn shared_gpu() -> Option<Device> {
    GPU.get_or_init(|| guard(Device::gpu).ok().flatten())
        .clone()
}

fn gpu_or_err() -> PyResult<Device> {
    shared_gpu().ok_or_else(|| {
        runtime_error(
            "no GPU adapter available (wgpu found no Metal, Vulkan or DX12 device); \
             use Device('cpu')",
        )
    })
}

fn parse_kind(kind: &str) -> PyResult<Device> {
    match kind.to_ascii_lowercase().as_str() {
        "cpu" => Ok(Device::cpu()),
        "gpu" => gpu_or_err(),
        "best" => Ok(shared_gpu().unwrap_or_else(Device::cpu)),
        _ => Err(value_error(format!(
            "unknown device {kind:?}; expected 'cpu', 'gpu' or 'best'"
        ))),
    }
}

/// Where a tensor lives: the CPU, or a GPU through wgpu.
#[pyclass(name = "Device", module = "ferrotensor", frozen)]
#[derive(Clone)]
pub(crate) struct PyDevice {
    pub(crate) inner: Device,
}

#[pymethods]
impl PyDevice {
    #[new]
    #[pyo3(signature = (kind = "cpu"))]
    fn new(kind: &str) -> PyResult<Self> {
        Ok(Self {
            inner: parse_kind(kind)?,
        })
    }

    /// The CPU device.
    #[staticmethod]
    fn cpu() -> Self {
        Self {
            inner: Device::cpu(),
        }
    }

    /// The GPU device; raises `RuntimeError` when no adapter is available.
    #[staticmethod]
    fn gpu() -> PyResult<Self> {
        Ok(Self {
            inner: gpu_or_err()?,
        })
    }

    /// The GPU when one is available, otherwise the CPU.
    #[staticmethod]
    fn best() -> Self {
        Self {
            inner: shared_gpu().unwrap_or_else(Device::cpu),
        }
    }

    /// Whether a GPU adapter is available.
    #[staticmethod]
    fn gpu_available() -> bool {
        shared_gpu().is_some()
    }

    /// `"cpu"` or `"gpu"`.
    #[getter]
    fn r#type(&self) -> &'static str {
        if self.inner.is_cpu() {
            "cpu"
        } else {
            "gpu"
        }
    }

    #[getter]
    fn is_cpu(&self) -> bool {
        self.inner.is_cpu()
    }

    #[getter]
    fn is_gpu(&self) -> bool {
        self.inner.is_gpu()
    }

    /// `"CPU"` or the GPU adapter name.
    #[getter]
    fn name(&self) -> String {
        self.inner.name()
    }

    /// `"Cpu"` or the wgpu backend (`"Metal"`, `"Vulkan"`, `"Dx12"`, ...).
    #[getter]
    fn backend(&self) -> String {
        self.inner.backend()
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .cast::<PyDevice>()
            .is_ok_and(|o| o.get().inner.same_as(&self.inner))
    }

    fn __hash__(&self) -> u64 {
        u64::from(self.inner.is_gpu())
    }

    fn __repr__(&self) -> String {
        if self.inner.is_cpu() {
            "Device('cpu')".to_string()
        } else {
            format!(
                "Device('gpu', name='{}', backend='{}')",
                self.inner.name(),
                self.inner.backend()
            )
        }
    }
}

/// Resolve a `device=` argument: `None` (CPU), a `Device` or a string.
pub(crate) fn resolve_device(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Device> {
    let Some(obj) = obj else {
        return Ok(Device::cpu());
    };
    if obj.is_none() {
        return Ok(Device::cpu());
    }
    if let Ok(d) = obj.cast::<PyDevice>() {
        return Ok(d.get().inner.clone());
    }
    if let Ok(s) = obj.extract::<String>() {
        return parse_kind(&s);
    }
    Err(PyTypeError::new_err(
        "device must be a ferrotensor.Device, 'cpu', 'gpu' or None",
    ))
}
