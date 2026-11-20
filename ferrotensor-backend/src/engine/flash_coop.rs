//! Flash attention on cooperative matrices: the forward pass, dQ and dK/dV
//! for `head_dim` 64 (see `shaders/flash_coop.wgsl`).
//!
//! The kernels run when the device has cooperative matrices and
//! [`FLASH_COOP_ENV`] allows them, for calls without a mask, dropout or a
//! sliding window; every other call, and `seq_q == 1`, keeps the scalar
//! kernels. Q, K, V, dO, P and dS are rounded to f16 (an out-of-range value is
//! clamped, counted in the matmul's overflow counter, and the workgroup
//! writes NaN into its first output element).

use super::coop::immediate_source;
use super::{attention::AttentionOptions, Arg, ComputeEngine, Kernel, Params};
use crate::{UnifiedTensor, WgpuContext};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, OnceLock};

/// Environment variable that switches the cooperative-matrix attention
/// kernels: `0` keeps the scalar kernels, `1` turns them on when the device
/// has cooperative matrices, and unset follows
/// [`EXPERIMENTAL_ENV`](crate::adapters::EXPERIMENTAL_ENV).
pub const FLASH_COOP_ENV: &str = "FERROTENSOR_FLASH_COOP";

/// Query or key rows per workgroup of the cooperative kernels.
const ROWS: u32 = 64;
/// The only head size the kernels are written for.
const HEAD_DIM: u32 = 64;
/// Largest workgroup count per grid dimension.
const MAX_GROUPS: u32 = 65_535;

/// The WGSL of the three entry points.
pub(super) fn source() -> &'static str {
    include_str!("../shaders/flash_coop.wgsl")
}

/// Whether `value` of [`FLASH_COOP_ENV`] turns the kernels on, given whether
/// the experimental opt-in is set.
fn enabled_by(value: Option<&str>, experimental: bool) -> bool {
    match value {
        Some("0") => false,
        Some("1") => true,
        _ => experimental,
    }
}

/// The three pipelines of the module.
struct Kernels {
    fwd: Kernel,
    dq: Kernel,
    dkdv: Kernel,
}

/// The kernels, compiled the first time a call runs on them, and the switch
/// that routes calls to them.
pub(super) struct FlashCoop {
    kernels: OnceLock<Kernels>,
    enabled: AtomicBool,
    /// Calls routed to the forward kernel and to the backward pair so far.
    runs: (AtomicU64, AtomicU64),
}

impl FlashCoop {
    /// The switch for a device that has cooperative matrices.
    pub(super) fn new() -> Self {
        let value = std::env::var(FLASH_COOP_ENV).ok();
        Self {
            kernels: OnceLock::new(),
            enabled: AtomicBool::new(enabled_by(
                value.as_deref(),
                crate::adapters::experimental_opt_in(),
            )),
            runs: (AtomicU64::new(0), AtomicU64::new(0)),
        }
    }

    /// The pipelines, compiled for `ctx` on first use.
    fn kernels(&self, ctx: &WgpuContext) -> &Kernels {
        self.kernels.get_or_init(|| {
            let mut src = source().to_owned();
            if ctx.uses_immediates() {
                src = immediate_source(&src);
            }
            let module = ctx.shader_checks().module(&ctx.device, "flash_coop", src);
            let kernel = |entry_point: &'static str| {
                let pipeline = Arc::new(ctx.device.create_compute_pipeline(
                    &wgpu::ComputePipelineDescriptor {
                        label: Some(entry_point),
                        layout: None,
                        module: &module,
                        entry_point: Some(entry_point),
                        // Every workgroup value is written before it is read.
                        compilation_options: wgpu::PipelineCompilationOptions {
                            zero_initialize_workgroup_memory: false,
                            ..Default::default()
                        },
                        cache: None,
                    },
                ));
                let layout = pipeline.get_bind_group_layout(0);
                Kernel {
                    pipeline,
                    layout,
                    entry_point,
                }
            };
            Kernels {
                fwd: kernel("flash_coop_fwd"),
                dq: kernel("flash_coop_bwd_dq"),
                dkdv: kernel("flash_coop_bwd_dkdv"),
            }
        })
    }
}

/// What the router needs to know about a call.
#[derive(Clone, Copy)]
pub(super) struct Shape {
    pub batch: u32,
    pub heads: u32,
    pub seq_q: u32,
    pub seq_k: u32,
    pub head_dim: u32,
}

impl ComputeEngine {
    /// Whether this device can run attention on cooperative matrices,
    /// whether or not they are turned on.
    #[must_use]
    pub fn flash_coop_available(&self) -> bool {
        self.coop.as_ref().is_some_and(|c| c.flash.is_some())
    }

    /// Whether attention calls that fit the kernels run on cooperative
    /// matrices: the device has them and [`FLASH_COOP_ENV`] allows them.
    #[must_use]
    pub fn flash_coop_active(&self) -> bool {
        self.coop
            .as_ref()
            .and_then(|c| c.flash.as_ref())
            .is_some_and(|f| f.enabled.load(Relaxed))
    }

    /// Turns the cooperative-matrix attention kernels on or off, overriding
    /// [`FLASH_COOP_ENV`]; a no-op on a device without them.
    #[doc(hidden)]
    pub fn set_flash_coop(&self, on: bool) {
        if let Some(f) = self.coop.as_ref().and_then(|c| c.flash.as_ref()) {
            f.enabled.store(on, Relaxed);
        }
    }

    /// How many forward calls and how many backward calls ran on the
    /// cooperative-matrix kernels since the engine was created.
    #[doc(hidden)]
    #[must_use]
    pub fn flash_coop_runs(&self) -> (u64, u64) {
        self.coop
            .as_ref()
            .and_then(|c| c.flash.as_ref())
            .map_or((0, 0), |f| (f.runs.0.load(Relaxed), f.runs.1.load(Relaxed)))
    }

    /// Whether a call of this shape runs on the cooperative kernels.
    pub(super) fn flash_coop_routes(
        &self,
        shape: Shape,
        opts: &AttentionOptions,
        has_mask: bool,
    ) -> bool {
        self.flash_coop_active()
            && shape.head_dim == HEAD_DIM
            && !has_mask
            && opts.window == 0
            && opts.dropout_prob == 0.0
            && shape.seq_q > 1
            && shape.seq_q.div_ceil(ROWS) <= MAX_GROUPS
            && shape.seq_k.div_ceil(ROWS) <= MAX_GROUPS
            && shape.batch * shape.heads <= MAX_GROUPS
    }

    /// The forward kernel; `stats` is the buffer for the saved
    /// log-sum-exp, or a dummy when `params` says not to write it.
    pub(super) fn flash_coop_forward(
        &self,
        ctx: &WgpuContext,
        shape: Shape,
        [q, k, v, out]: [&UnifiedTensor; 4],
        stats: &dyn Arg,
        params: &Params,
    ) {
        let Some((coop, flash)) = self
            .coop
            .as_ref()
            .and_then(|c| Some((c, c.flash.as_ref()?)))
        else {
            return;
        };
        flash.runs.0.fetch_add(1, Relaxed);
        self.run(
            ctx,
            &flash.kernels(ctx).fwd,
            &[
                (0, q.buffer()),
                (1, k.buffer()),
                (2, v.buffer()),
                (3, out.buffer()),
                (4, params),
                (6, stats),
                (7, coop.overflows.buffer()),
            ],
            (1, shape.seq_q.div_ceil(ROWS), shape.batch * shape.heads),
        );
    }

    /// The dQ kernel; it also writes Delta into `stats`.
    pub(super) fn flash_coop_dq(
        &self,
        ctx: &WgpuContext,
        shape: Shape,
        [q, k, v, out, grad_out, grad_q]: [&UnifiedTensor; 6],
        (stats, params): (&dyn Arg, &Params),
    ) {
        let Some((coop, flash)) = self
            .coop
            .as_ref()
            .and_then(|c| Some((c, c.flash.as_ref()?)))
        else {
            return;
        };
        flash.runs.1.fetch_add(1, Relaxed);
        self.run(
            ctx,
            &flash.kernels(ctx).dq,
            &[
                (0, q.buffer()),
                (1, k.buffer()),
                (2, v.buffer()),
                (13, out.buffer()),
                (14, grad_out.buffer()),
                (15, grad_q.buffer()),
                (8, params),
                (10, stats),
                (11, coop.overflows.buffer()),
            ],
            (1, shape.seq_q.div_ceil(ROWS), shape.batch * shape.heads),
        );
    }

    /// The dK/dV kernel; reads Delta from `stats`, so it runs after dQ.
    /// `grads` are the dK and dV buffers it writes, direct or per query head
    /// as `params` says.
    pub(super) fn flash_coop_dkdv(
        &self,
        ctx: &WgpuContext,
        shape: Shape,
        [q, k, v, grad_out]: [&UnifiedTensor; 4],
        grads: (&dyn Arg, &dyn Arg),
        (stats, params): (&dyn Arg, &Params),
    ) {
        let Some((coop, flash)) = self
            .coop
            .as_ref()
            .and_then(|c| Some((c, c.flash.as_ref()?)))
        else {
            return;
        };
        self.run(
            ctx,
            &flash.kernels(ctx).dkdv,
            &[
                (0, q.buffer()),
                (1, k.buffer()),
                (2, v.buffer()),
                (14, grad_out.buffer()),
                (16, grads.0),
                (17, grads.1),
                (8, params),
                (10, stats),
                (11, coop.overflows.buffer()),
            ],
            (1, shape.seq_k.div_ceil(ROWS), shape.batch * shape.heads),
        );
    }
}

#[cfg(test)]
#[allow(clippy::panic)]
mod tests {
    use super::{enabled_by, immediate_source, source};

    fn capabilities() -> naga::valid::Capabilities {
        naga::valid::Capabilities::COOPERATIVE_MATRIX
            | naga::valid::Capabilities::SHADER_FLOAT16
            | naga::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32
    }

    #[test]
    fn the_switch_follows_the_env_and_the_opt_in() {
        assert!(!enabled_by(None, false));
        assert!(enabled_by(None, true));
        assert!(!enabled_by(Some("0"), true));
        assert!(enabled_by(Some("1"), false));
        assert!(enabled_by(Some("other"), true));
        assert!(!enabled_by(Some("other"), false));
    }

    #[test]
    fn the_kernels_validate_with_cooperative_matrices() {
        let src = source();
        let module = match naga::front::wgsl::parse_str(src) {
            Ok(m) => m,
            Err(e) => panic!("WGSL parse error:\n{}", e.emit_to_string(src)),
        };
        let mut validator =
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), capabilities());
        if let Err(e) = validator.validate(&module) {
            panic!("WGSL validation error:\n{}", e.emit_to_string(src));
        }
        let entries: Vec<&str> = module
            .entry_points
            .iter()
            .map(|e| e.name.as_str())
            .collect();
        assert_eq!(
            entries,
            ["flash_coop_fwd", "flash_coop_bwd_dq", "flash_coop_bwd_dkdv"]
        );
    }

    #[test]
    fn the_kernels_validate_with_immediates() {
        let src = immediate_source(source());
        assert!(!src.contains("var<uniform>"));
        let module = match naga::front::wgsl::parse_str(&src) {
            Ok(m) => m,
            Err(e) => panic!("WGSL parse error:\n{}", e.emit_to_string(&src)),
        };
        assert_eq!(
            naga::valid::ImmediateSlots::size_for_module(&module),
            crate::engine::IMMEDIATE_BYTES
        );
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            capabilities() | naga::valid::Capabilities::IMMEDIATES,
        );
        if let Err(e) = validator.validate(&module) {
            panic!("WGSL validation error:\n{}", e.emit_to_string(&src));
        }
    }

    /// The workgroup arrays of every entry point fit the smallest limit the
    /// kernels target (an A10G's 49,152 bytes).
    #[test]
    fn workgroup_memory_fits_an_a10g() {
        let src = source();
        let mut bytes = 0usize;
        for (name, elements, size) in [
            ("fc_ta", 4608, 2),
            ("fc_tb", 4608, 2),
            ("fc_tc", 4608, 2),
            ("fc_ss", 4352, 4),
            ("fc_pm", 128, 4),
            ("fc_zs", 256, 4),
            ("fc_sl", 64, 4),
            ("fc_sd", 64, 4),
        ] {
            assert!(
                src.contains(&format!("var<workgroup> {name}: array<")),
                "{name}"
            );
            assert_eq!(
                elements * size % 16,
                0,
                "{name} is not a multiple of 16 bytes"
            );
            bytes += elements * size;
        }
        assert!(bytes <= 49_152, "{bytes} bytes");
    }
}
