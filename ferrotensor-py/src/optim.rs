//! Optimizers, gradient clipping and LR schedules behind `ferrotensor.optim`.

use ferrotensor::optim::{
    self, AdamW, ConstantLr, CosineAnnealingLr, LrScheduler, Optimizer, StepLr, WarmupCosineLr,
    WarmupLinearLr, SGD,
};
use ferrotensor::Tensor;
use pyo3::prelude::*;

use crate::error::{guard, value_error};
use crate::tensor::PyTensor;

fn tensors(params: &Bound<'_, PyAny>) -> PyResult<Vec<Tensor>> {
    params
        .try_iter()?
        .map(|p| Ok(p?.cast::<PyTensor>()?.borrow().inner.clone()))
        .collect()
}

/// Parameters for an optimizer: non-empty, all requiring grad.
fn trainable(params: &Bound<'_, PyAny>) -> PyResult<Vec<Tensor>> {
    let params = tensors(params)?;
    if params.is_empty() {
        return Err(value_error("optimizer got an empty parameter list"));
    }
    if params.iter().any(|p| !p.requires_grad) {
        return Err(value_error(
            "optimizer parameters must all have requires_grad=True",
        ));
    }
    Ok(params)
}

fn check_lr(lr: f32) -> PyResult<f32> {
    if lr.is_finite() && lr >= 0.0 {
        Ok(lr)
    } else {
        Err(value_error(format!("invalid learning rate {lr}")))
    }
}

/// Stochastic gradient descent with optional momentum and weight decay.
#[pyclass(name = "SGD", module = "ferrotensor.optim")]
pub(crate) struct PySGD {
    inner: SGD,
}

#[pymethods]
impl PySGD {
    #[new]
    #[pyo3(signature = (params, lr, momentum = 0.0, weight_decay = 0.0, nesterov = false))]
    fn new(
        params: &Bound<'_, PyAny>,
        lr: f32,
        momentum: f32,
        weight_decay: f32,
        nesterov: bool,
    ) -> PyResult<Self> {
        check_lr(lr)?;
        if momentum < 0.0 || weight_decay < 0.0 {
            return Err(value_error(
                "momentum and weight_decay must be non-negative",
            ));
        }
        if nesterov && momentum == 0.0 {
            return Err(value_error("nesterov momentum needs momentum > 0"));
        }
        let inner = SGD::with_momentum(trainable(params)?, lr, momentum)
            .weight_decay(weight_decay)
            .nesterov(nesterov);
        Ok(Self { inner })
    }

    /// Update every parameter from its `.grad` (parameters without one are skipped).
    fn step(&mut self) -> PyResult<()> {
        guard(|| self.inner.step())
    }

    /// Clear every parameter's `.grad`.
    fn zero_grad(&self) -> PyResult<()> {
        guard(|| self.inner.zero_grad())
    }

    #[getter]
    fn lr(&self) -> f32 {
        self.inner.lr()
    }

    #[setter]
    fn set_lr(&mut self, lr: f32) -> PyResult<()> {
        self.inner.set_lr(check_lr(lr)?);
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "SGD(lr={}, params={})",
            self.inner.lr(),
            self.inner.params().len()
        )
    }
}

/// Adam with decoupled weight decay.
#[pyclass(name = "AdamW", module = "ferrotensor.optim")]
pub(crate) struct PyAdamW {
    inner: AdamW,
}

#[pymethods]
impl PyAdamW {
    #[new]
    #[pyo3(signature = (
        params, lr = 1e-3, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 0.01,
        max_grad_norm = None
    ))]
    fn new(
        params: &Bound<'_, PyAny>,
        lr: f32,
        betas: (f32, f32),
        eps: f32,
        weight_decay: f32,
        max_grad_norm: Option<f32>,
    ) -> PyResult<Self> {
        check_lr(lr)?;
        let (b1, b2) = betas;
        if !(0.0..1.0).contains(&b1) || !(0.0..1.0).contains(&b2) {
            return Err(value_error(format!(
                "betas must be in [0, 1), got {betas:?}"
            )));
        }
        if eps <= 0.0 || weight_decay < 0.0 {
            return Err(value_error(
                "eps must be positive and weight_decay non-negative",
            ));
        }
        let mut inner = AdamW::with_params(trainable(params)?, lr, b1, b2, eps, weight_decay);
        if let Some(max_norm) = max_grad_norm {
            if max_norm <= 0.0 {
                return Err(value_error("max_grad_norm must be positive"));
            }
            inner = inner.clip_grad_norm(max_norm);
        }
        Ok(Self { inner })
    }

    /// Update every parameter from its `.grad` (parameters without one are skipped).
    fn step(&mut self) -> PyResult<()> {
        guard(|| self.inner.step())
    }

    /// Clear every parameter's `.grad`.
    fn zero_grad(&self) -> PyResult<()> {
        guard(|| self.inner.zero_grad())
    }

    #[getter]
    fn lr(&self) -> f32 {
        self.inner.lr()
    }

    #[setter]
    fn set_lr(&mut self, lr: f32) -> PyResult<()> {
        self.inner.set_lr(check_lr(lr)?);
        Ok(())
    }

    /// Number of steps taken.
    #[getter]
    fn steps(&self) -> usize {
        self.inner.steps()
    }

    fn __repr__(&self) -> String {
        format!(
            "AdamW(lr={}, params={})",
            self.inner.lr(),
            self.inner.params().len()
        )
    }
}

/// Scale gradients so their global L2 norm is at most `max_norm`; returns
/// the norm before clipping.
#[pyfunction]
pub(crate) fn clip_grad_norm(params: &Bound<'_, PyAny>, max_norm: f32) -> PyResult<f32> {
    if max_norm <= 0.0 {
        return Err(value_error("max_norm must be positive"));
    }
    let params = tensors(params)?;
    guard(|| optim::clip_grad_norm(&params, max_norm))
}

/// Clamp every gradient entry to `[-clip_value, clip_value]`.
#[pyfunction]
pub(crate) fn clip_grad_value(params: &Bound<'_, PyAny>, clip_value: f32) -> PyResult<()> {
    if clip_value <= 0.0 {
        return Err(value_error("clip_value must be positive"));
    }
    let params = tensors(params)?;
    guard(|| optim::clip_grad_value(&params, clip_value))
}

/// A learning-rate schedule (the Python classes in `ferrotensor.optim` wrap it).
#[pyclass(name = "_Schedule", module = "ferrotensor.optim", frozen)]
pub(crate) struct PySchedule {
    inner: Box<dyn LrScheduler + Send + Sync>,
}

fn schedule(s: impl LrScheduler + Send + Sync + 'static) -> PySchedule {
    PySchedule { inner: Box::new(s) }
}

#[pymethods]
impl PySchedule {
    #[staticmethod]
    fn constant(lr: f32) -> PyResult<Self> {
        Ok(schedule(ConstantLr { lr: check_lr(lr)? }))
    }

    #[staticmethod]
    fn step(base_lr: f32, step_size: usize, gamma: f32) -> PyResult<Self> {
        if step_size == 0 {
            return Err(value_error("step_size must be positive"));
        }
        Ok(schedule(StepLr {
            base_lr: check_lr(base_lr)?,
            step_size,
            gamma,
        }))
    }

    #[staticmethod]
    fn cosine(base_lr: f32, min_lr: f32, total_steps: usize) -> PyResult<Self> {
        Ok(schedule(CosineAnnealingLr {
            base_lr: check_lr(base_lr)?,
            min_lr: check_lr(min_lr)?,
            total_steps,
        }))
    }

    #[staticmethod]
    fn warmup_cosine(
        peak_lr: f32,
        min_lr: f32,
        warmup_steps: usize,
        total_steps: usize,
    ) -> PyResult<Self> {
        Ok(schedule(WarmupCosineLr::new(
            check_lr(peak_lr)?,
            check_lr(min_lr)?,
            warmup_steps,
            total_steps,
        )))
    }

    #[staticmethod]
    fn warmup_linear(
        peak_lr: f32,
        end_lr: f32,
        warmup_steps: usize,
        total_steps: usize,
    ) -> PyResult<Self> {
        Ok(schedule(WarmupLinearLr {
            peak_lr: check_lr(peak_lr)?,
            end_lr: check_lr(end_lr)?,
            warmup_steps,
            total_steps,
        }))
    }

    /// The learning rate at optimizer step `step` (0-based).
    fn lr_at(&self, step: usize) -> f32 {
        self.inner.lr_at(step)
    }
}
