//! GPU Backend Module
//!
//! This module provides low-level GPU compute functionality via wgpu.
//! It includes the compute engine for kernel dispatch and GPU buffer management.

mod compute;
mod kernels;

pub use compute::ComputeEngine;
pub(crate) use kernels::KERNELS_WGSL;
