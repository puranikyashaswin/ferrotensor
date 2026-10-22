//! Python bindings for ferrotensor, imported as `ferrotensor._ferrotensor`.
//!
//! The pure-Python package in `python/ferrotensor` re-exports these items and
//! adds `nn.Module`, `Sequential`, `no_grad` and the scheduler classes.

mod device;
mod error;
mod functional;
mod nn;
mod optim;
mod serialize;
mod tensor;

use pyo3::prelude::*;

#[pymodule]
fn _ferrotensor(m: &Bound<'_, PyModule>) -> PyResult<()> {
    error::install_panic_hook();
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    m.add_class::<device::PyDevice>()?;
    m.add_class::<tensor::PyTensor>()?;
    m.add_function(wrap_pyfunction!(tensor::tensor, m)?)?;
    m.add_function(wrap_pyfunction!(tensor::zeros, m)?)?;
    m.add_function(wrap_pyfunction!(tensor::ones, m)?)?;
    m.add_function(wrap_pyfunction!(tensor::full, m)?)?;
    m.add_function(wrap_pyfunction!(tensor::zeros_like, m)?)?;
    m.add_function(wrap_pyfunction!(tensor::ones_like, m)?)?;
    m.add_function(wrap_pyfunction!(tensor::full_like, m)?)?;
    m.add_function(wrap_pyfunction!(tensor::randn, m)?)?;
    m.add_function(wrap_pyfunction!(tensor::rand, m)?)?;
    m.add_function(wrap_pyfunction!(tensor::arange, m)?)?;
    m.add_function(wrap_pyfunction!(tensor::linspace, m)?)?;
    m.add_function(wrap_pyfunction!(tensor::eye, m)?)?;
    m.add_function(wrap_pyfunction!(tensor::cat, m)?)?;
    m.add_function(wrap_pyfunction!(tensor::stack, m)?)?;
    m.add_function(wrap_pyfunction!(tensor::where_, m)?)?;
    m.add_function(wrap_pyfunction!(tensor::grad_enabled, m)?)?;
    m.add_function(wrap_pyfunction!(tensor::_no_grad_enter, m)?)?;
    m.add_function(wrap_pyfunction!(tensor::_no_grad_exit, m)?)?;
    m.add_function(wrap_pyfunction!(serialize::save, m)?)?;
    m.add_function(wrap_pyfunction!(serialize::load, m)?)?;

    m.add_class::<nn::PyLayer>()?;
    m.add_function(wrap_pyfunction!(functional::mse_loss, m)?)?;
    m.add_function(wrap_pyfunction!(functional::l1_loss, m)?)?;
    m.add_function(wrap_pyfunction!(functional::smooth_l1_loss, m)?)?;
    m.add_function(wrap_pyfunction!(functional::cross_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(functional::nll_loss, m)?)?;
    m.add_function(wrap_pyfunction!(functional::binary_cross_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(
        functional::binary_cross_entropy_with_logits,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(functional::dropout, m)?)?;
    m.add_function(wrap_pyfunction!(functional::layer_norm, m)?)?;
    m.add_function(wrap_pyfunction!(functional::rms_norm, m)?)?;
    m.add_function(wrap_pyfunction!(functional::linear, m)?)?;
    m.add_function(wrap_pyfunction!(
        functional::scaled_dot_product_attention,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(functional::one_hot, m)?)?;
    m.add_function(wrap_pyfunction!(functional::accuracy, m)?)?;
    m.add_function(wrap_pyfunction!(functional::cosine_similarity, m)?)?;

    m.add_class::<optim::PySGD>()?;
    m.add_class::<optim::PyAdamW>()?;
    m.add_class::<optim::PySchedule>()?;
    m.add_function(wrap_pyfunction!(optim::clip_grad_norm, m)?)?;
    m.add_function(wrap_pyfunction!(optim::clip_grad_value, m)?)?;
    Ok(())
}
