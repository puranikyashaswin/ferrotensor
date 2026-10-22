//! Turning Rust panics into Python exceptions.
//!
//! The bindings check arguments up front and raise `ValueError`,
//! `IndexError` or `TypeError` for bad input. Every call into ferrotensor
//! also runs inside [`guard`], so an assertion the checks missed surfaces
//! as a `RuntimeError` instead of a `PanicException` (and without the panic
//! message being printed to stderr).

use std::any::Any;
use std::cell::Cell;
use std::panic::{self, AssertUnwindSafe};
use std::sync::Once;

use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyValueError};
use pyo3::{PyErr, PyResult};

thread_local! {
    static GUARDED: Cell<bool> = const { Cell::new(false) };
}

/// Silence the default panic message for panics caught by [`guard`].
pub(crate) fn install_panic_hook() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let previous = panic::take_hook();
        panic::set_hook(Box::new(move |info| {
            if !GUARDED.with(Cell::get) {
                previous(info);
            }
        }));
    });
}

/// Run `f`, converting a panic into a `RuntimeError`.
pub(crate) fn guard<T>(f: impl FnOnce() -> T) -> PyResult<T> {
    let outer = GUARDED.with(|g| g.replace(true));
    let result = panic::catch_unwind(AssertUnwindSafe(f));
    GUARDED.with(|g| g.set(outer));
    result.map_err(|payload| PyRuntimeError::new_err(panic_message(payload.as_ref())))
}

fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "ferrotensor raised an unknown error".to_string()
    }
}

/// A `ValueError` with `msg`.
pub(crate) fn value_error(msg: impl Into<String>) -> PyErr {
    PyValueError::new_err(msg.into())
}

/// An `IndexError` with `msg`.
pub(crate) fn index_error(msg: impl Into<String>) -> PyErr {
    PyIndexError::new_err(msg.into())
}

/// A `RuntimeError` with `msg`.
pub(crate) fn runtime_error(msg: impl Into<String>) -> PyErr {
    PyRuntimeError::new_err(msg.into())
}
