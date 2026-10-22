//! GPU Backend Module
//!
//! Low-level GPU compute via wgpu: the WGSL kernels and the engine that
//! compiles and dispatches them.

mod compute;
mod kernels;

pub use compute::ComputeEngine;
pub(crate) use compute::Dispatch;
pub(crate) use kernels::{binary_op, reduce_op, unary_op, KERNELS_WGSL};
