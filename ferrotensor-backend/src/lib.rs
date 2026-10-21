//! wgpu compute backend for Ferrotensor: GPU tensors and hand-written WGSL kernels.
//!
//! [`WgpuContext`] opens a GPU (Metal, Vulkan, DX12), [`UnifiedTensor`] holds
//! data on it and [`ComputeEngine`] compiles the kernels and dispatches them:
//! matmul (plain and transposed), element-wise ops, softmax, RMSNorm, RoPE,
//! fused flash attention (forward and backward, with GQA, masks, dropout,
//! KV-cache offsets and sliding windows), losses, optimisers and dequantisation.
//!
//! ```no_run
//! use ferrotensor_backend::{ComputeEngine, UnifiedTensor, WgpuContext};
//!
//! let ctx = pollster::block_on(WgpuContext::new());
//! let engine = ComputeEngine::new(&ctx);
//! let a = UnifiedTensor::new(&ctx, &[1.0, 2.0, 3.0, 4.0], &[2, 2]);
//! let b = UnifiedTensor::new(&ctx, &[1.0, 0.0, 0.0, 1.0], &[2, 2]);
//! let c = UnifiedTensor::empty(&ctx, &[2, 2]);
//! engine.matmul(&ctx, &a, &b, &c);
//! assert_eq!(pollster::block_on(c.to_vec(&ctx)), vec![1.0, 2.0, 3.0, 4.0]);
//! ```
//!
//! Kernels are checked against CPU references in `tests/gpu_kernels.rs`
//! (run with `cargo test -p ferrotensor-backend --test gpu_kernels -- --ignored
//! --test-threads=1`).

/// The `wgpu` version this crate is built on, re-exported for callers.
pub use wgpu; // Re-export wgpu for downstream crates

mod context;
mod engine;
/// Size-keyed reuse of GPU buffers.
pub mod pool;
mod tensor;

pub use context::WgpuContext;
pub use engine::{AdamParams, AttentionOptions, ComputeEngine};
pub use tensor::{DType, UnifiedTensor};
