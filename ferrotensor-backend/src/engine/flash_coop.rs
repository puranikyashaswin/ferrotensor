//! Flash attention on cooperative matrices: the forward pass, dQ and dK/dV
//! for `head_dim` 64 (see `shaders/flash_coop.wgsl`).
//!
//! The kernels run when the device has cooperative matrices and
//! [`FLASH_COOP_ENV`] allows them, for calls without a mask, dropout or a
//! sliding window; every other call, and `seq_q == 1`, keeps the scalar
//! kernels. Q, K, V, dO, P and dS are rounded to f16 (an out-of-range value is
//! clamped, counted in the matmul's overflow counter, and the workgroup
//! writes NaN into its first output element).
//!
//! A workgroup covers 64 or 32 query (or key) rows. The 32-row kernels split
//! each tile's products over the four subgroups by column blocks, so a grid
//! with few tiles (a batch of one, a few heads) has twice the workgroups and
//! each of them a shorter serial chain; `FERROTENSOR_FLASH_ROWS` forces one
//! size. A call may also rotate Q and K inside the kernels (RoPE) and get the
//! gradients of the un-rotated tensors back.

use super::coop::immediate_source;
use super::{attention::AttentionOptions, Arg, ComputeEngine, Kernel, Params};
use crate::{UnifiedTensor, WgpuContext};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use wgpu::{util::DeviceExt, Buffer};

/// Environment variable that switches the cooperative-matrix attention
/// kernels: `0` keeps the scalar kernels, `1` turns them on when the device
/// has cooperative matrices, and unset follows
/// [`EXPERIMENTAL_ENV`](crate::adapters::EXPERIMENTAL_ENV).
pub const FLASH_COOP_ENV: &str = "FERROTENSOR_FLASH_COOP";

/// Environment variable that sets the rows per workgroup of the
/// cooperative kernels: `32` or `64`; anything else picks per call.
const FLASH_ROWS_ENV: &str = "FERROTENSOR_FLASH_ROWS";

/// Rows per workgroup of the wide kernels.
const WIDE: u32 = 64;
/// Rows per workgroup of the narrow kernels.
const NARROW: u32 = 32;
/// Query rows per tile of the grid that the wide kernels would launch.
const TILE: u32 = 64;
/// Workgroups below which a grid leaves SMs idle (two per SM of an A10G).
const FILL: u64 = 160;
/// The only head size the kernels are written for.
const HEAD_DIM: u32 = 64;
/// Largest workgroup count per grid dimension.
const MAX_GROUPS: u32 = 65_535;

/// The WGSL of the three entry points for `rows` (32 or 64) rows per
/// workgroup: the lines tagged for the other size are dropped and the size
/// tokens are filled in.
pub(super) fn source(rows: u32) -> String {
    let (keep, drop) = if rows == NARROW {
        ("//@32", "//@64")
    } else {
        ("//@64", "//@32")
    };
    let template = include_str!("../shaders/flash_coop.wgsl");
    let mut out = String::with_capacity(template.len());
    for line in template.lines() {
        let line = line.trim_end();
        if line.ends_with(drop) {
            continue;
        }
        out.push_str(line.strip_suffix(keep).map_or(line, str::trim_end));
        out.push('\n');
    }
    out.replace("@BR@", &rows.to_string())
        .replace("@TA@", &(rows * 72).to_string())
        .replace("@SS@", &(rows * 68).to_string())
        .replace("@NC@", &(rows / 2).to_string())
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

/// The rows `FERROTENSOR_FLASH_ROWS` asks for, or 0 for "pick per call".
fn rows_from_env(value: Option<&str>) -> u32 {
    match value {
        Some("32") => NARROW,
        Some("64") => WIDE,
        _ => 0,
    }
}

/// Rows per workgroup for a call: `forced` (32 or 64) when set, else 32 when
/// 64-row workgroups would leave SMs idle.
fn rows_for(forced: u32, shape: Shape) -> u32 {
    if forced == NARROW || forced == WIDE {
        return forced;
    }
    let tiles = u64::from(shape.seq_q.div_ceil(TILE));
    let groups = u64::from(shape.batch) * u64::from(shape.heads) * tiles;
    if groups < FILL {
        NARROW
    } else {
        WIDE
    }
}

/// RoPE inside the kernels: base frequency and the position of row 0 of Q
/// and of K (row `j` of either sits at position `pos + j`).
#[derive(Clone, Copy, Debug)]
pub(super) struct Rope {
    pub theta: f32,
    pub pos: u32,
}

/// The cos/sin table the kernels read: for each of `len` positions 32
/// cosines then 32 sines, in f32 with the formula of the scalar `rope`
/// kernel, `angle = position * theta^(-2 d / head_dim)`.
fn rope_table(rope: Rope, len: u32) -> Vec<f32> {
    let half = HEAD_DIM / 2;
    let mut out = Vec::with_capacity(len as usize * HEAD_DIM as usize);
    for t in 0..len {
        let position = rope.pos.saturating_add(t) as f32;
        let angles: Vec<f32> = (0..half)
            .map(|d| position * rope.theta.powf(-2.0 * d as f32 / HEAD_DIM as f32))
            .collect();
        out.extend(angles.iter().map(|a| a.cos()));
        out.extend(angles.iter().map(|a| a.sin()));
    }
    out
}

/// The pipelines of one row size.
struct Kernels {
    fwd: Kernel,
    dq: Kernel,
    dkdv: Kernel,
}

/// The table last built for RoPE calls, reused while the call's base
/// frequency and first position match and it is long enough.
struct RopeTable {
    theta_bits: u32,
    pos: u32,
    len: u32,
    buffer: Buffer,
}

/// The kernels, compiled the first time a call runs on them, and the switch
/// that routes calls to them.
pub(super) struct FlashCoop {
    /// Kernels for 64 rows, then for 32.
    kernels: [OnceLock<Kernels>; 2],
    enabled: AtomicBool,
    /// Rows per workgroup forced by the environment or a test; 0 picks per call.
    rows: AtomicU32,
    /// Calls routed to the forward kernel and to the backward pair so far.
    runs: (AtomicU64, AtomicU64),
    /// Calls that ran on 32-row kernels, forward then backward.
    narrow_runs: (AtomicU64, AtomicU64),
    table: Mutex<Option<RopeTable>>,
    /// The 16-byte table bound when a call does not rotate.
    blank: OnceLock<Buffer>,
}

impl FlashCoop {
    /// The switch for a device that has cooperative matrices.
    pub(super) fn new() -> Self {
        let value = std::env::var(FLASH_COOP_ENV).ok();
        let rows = std::env::var(FLASH_ROWS_ENV).ok();
        Self {
            kernels: [OnceLock::new(), OnceLock::new()],
            enabled: AtomicBool::new(enabled_by(
                value.as_deref(),
                crate::adapters::experimental_opt_in(),
            )),
            rows: AtomicU32::new(rows_from_env(rows.as_deref())),
            runs: (AtomicU64::new(0), AtomicU64::new(0)),
            narrow_runs: (AtomicU64::new(0), AtomicU64::new(0)),
            table: Mutex::new(None),
            blank: OnceLock::new(),
        }
    }

    /// The rows per workgroup this call runs with.
    fn rows(&self, shape: Shape) -> u32 {
        rows_for(self.rows.load(Relaxed), shape)
    }

    /// The pipelines for `rows` rows, compiled for `ctx` on first use.
    fn kernels(&self, ctx: &WgpuContext, rows: u32) -> &Kernels {
        let slot = &self.kernels[usize::from(rows == NARROW)];
        slot.get_or_init(|| {
            let mut src = source(rows);
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

    /// The buffer for binding 18: the cos/sin table of `rope` with at least
    /// `len` positions, or a dummy when the call does not rotate.
    fn rope_buffer(&self, ctx: &WgpuContext, rope: Option<Rope>, len: u32) -> Buffer {
        let Some(rope) = rope else {
            return self
                .blank
                .get_or_init(|| {
                    ctx.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Flash RoPE Blank"),
                            contents: &[0u8; 16],
                            usage: wgpu::BufferUsages::STORAGE,
                        })
                })
                .clone();
        };
        let mut slot = self.table.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(t) = slot.as_ref() {
            if t.theta_bits == rope.theta.to_bits() && t.pos == rope.pos && t.len >= len {
                return t.buffer.clone();
            }
        }
        let data = rope_table(rope, len);
        let buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Flash RoPE Table"),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::STORAGE,
            });
        *slot = Some(RopeTable {
            theta_bits: rope.theta.to_bits(),
            pos: rope.pos,
            len,
            buffer: buffer.clone(),
        });
        buffer
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

    /// Forces 32 or 64 rows per workgroup in the cooperative-matrix attention
    /// kernels, overriding the environment; any other value picks per call.
    /// A no-op on a device without them.
    #[doc(hidden)]
    pub fn set_flash_rows(&self, rows: u32) {
        if let Some(f) = self.coop.as_ref().and_then(|c| c.flash.as_ref()) {
            let forced = rows_from_env(Some(&rows.to_string()));
            f.rows.store(forced, Relaxed);
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

    /// How many of the [`flash_coop_runs`](Self::flash_coop_runs) forward and
    /// backward calls ran on the 32-row kernels.
    #[doc(hidden)]
    #[must_use]
    pub fn flash_coop_narrow_runs(&self) -> (u64, u64) {
        self.coop
            .as_ref()
            .and_then(|c| c.flash.as_ref())
            .map_or((0, 0), |f| {
                (f.narrow_runs.0.load(Relaxed), f.narrow_runs.1.load(Relaxed))
            })
    }

    /// Whether a call of this shape runs on the cooperative kernels.
    pub(super) fn flash_coop_routes(
        &self,
        shape: Shape,
        opts: &AttentionOptions,
        has_mask: bool,
    ) -> bool {
        let Some(flash) = self.coop.as_ref().and_then(|c| c.flash.as_ref()) else {
            return false;
        };
        let rows = flash.rows(shape);
        flash.enabled.load(Relaxed)
            && shape.head_dim == HEAD_DIM
            && !has_mask
            && opts.window == 0
            && opts.dropout_prob == 0.0
            && shape.seq_q > 1
            && shape.seq_q.div_ceil(rows) <= MAX_GROUPS
            && shape.seq_k.div_ceil(rows) <= MAX_GROUPS
            && shape.batch * shape.heads <= MAX_GROUPS
    }

    /// The forward kernel; `stats` is the buffer for the saved
    /// log-sum-exp, or a dummy when `params` says not to write it. With
    /// `rope` the kernel rotates Q and K as it stages them (`params` carries
    /// the matching bit).
    pub(super) fn flash_coop_forward(
        &self,
        ctx: &WgpuContext,
        shape: Shape,
        [q, k, v, out]: [&UnifiedTensor; 4],
        stats: &dyn Arg,
        params: (&Params, Option<Rope>),
    ) {
        let Some((coop, flash)) = self
            .coop
            .as_ref()
            .and_then(|c| Some((c, c.flash.as_ref()?)))
        else {
            return;
        };
        flash.runs.0.fetch_add(1, Relaxed);
        let rows = flash.rows(shape);
        if rows == NARROW {
            flash.narrow_runs.0.fetch_add(1, Relaxed);
        }
        let table = flash.rope_buffer(ctx, params.1, shape.seq_q.max(shape.seq_k));
        self.run(
            ctx,
            &flash.kernels(ctx, rows).fwd,
            &[
                (0, q.buffer()),
                (1, k.buffer()),
                (2, v.buffer()),
                (3, out.buffer()),
                (4, params.0),
                (6, stats),
                (7, coop.overflows.buffer()),
                (18, &table),
            ],
            (1, shape.seq_q.div_ceil(rows), shape.batch * shape.heads),
        );
    }

    /// The dQ kernel; it also writes Delta into `stats`. With `rope` it
    /// rotates Q and K as it stages them and writes the gradient of the
    /// un-rotated Q.
    pub(super) fn flash_coop_dq(
        &self,
        ctx: &WgpuContext,
        shape: Shape,
        [q, k, v, out, grad_out, grad_q]: [&UnifiedTensor; 6],
        (stats, params, rope): (&dyn Arg, &Params, Option<Rope>),
    ) {
        let Some((coop, flash)) = self
            .coop
            .as_ref()
            .and_then(|c| Some((c, c.flash.as_ref()?)))
        else {
            return;
        };
        flash.runs.1.fetch_add(1, Relaxed);
        let rows = flash.rows(shape);
        if rows == NARROW {
            flash.narrow_runs.1.fetch_add(1, Relaxed);
        }
        let table = flash.rope_buffer(ctx, rope, shape.seq_q.max(shape.seq_k));
        self.run(
            ctx,
            &flash.kernels(ctx, rows).dq,
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
                (18, &table),
            ],
            (1, shape.seq_q.div_ceil(rows), shape.batch * shape.heads),
        );
    }

    /// The dK/dV kernel; reads Delta from `stats`, so it runs after dQ.
    /// `grads` are the dK and dV buffers it writes, direct or per query head
    /// as `params` says. With `rope` it rotates Q and K as it stages them
    /// and writes the gradient of the un-rotated K.
    pub(super) fn flash_coop_dkdv(
        &self,
        ctx: &WgpuContext,
        shape: Shape,
        [q, k, v, grad_out]: [&UnifiedTensor; 4],
        grads: (&dyn Arg, &dyn Arg),
        (stats, params, rope): (&dyn Arg, &Params, Option<Rope>),
    ) {
        let Some((coop, flash)) = self
            .coop
            .as_ref()
            .and_then(|c| Some((c, c.flash.as_ref()?)))
        else {
            return;
        };
        let rows = flash.rows(shape);
        let table = flash.rope_buffer(ctx, rope, shape.seq_q.max(shape.seq_k));
        self.run(
            ctx,
            &flash.kernels(ctx, rows).dkdv,
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
                (18, &table),
            ],
            (1, shape.seq_k.div_ceil(rows), shape.batch * shape.heads),
        );
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::needless_range_loop)]
mod tests {
    use super::{
        enabled_by, immediate_source, rope_table, rows_for, rows_from_env, source, Rope, Shape,
    };

    fn capabilities() -> naga::valid::Capabilities {
        naga::valid::Capabilities::COOPERATIVE_MATRIX
            | naga::valid::Capabilities::SHADER_FLOAT16
            | naga::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32
    }

    fn shape(batch: u32, heads: u32, seq: u32) -> Shape {
        Shape {
            batch,
            heads,
            seq_q: seq,
            seq_k: seq,
            head_dim: 64,
        }
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
    fn the_rows_follow_the_env_and_the_grid() {
        assert_eq!(rows_from_env(None), 0);
        assert_eq!(rows_from_env(Some("32")), 32);
        assert_eq!(rows_from_env(Some("64")), 64);
        assert_eq!(rows_from_env(Some("16")), 0);
        // 135M at batch 1: 9 heads x 4 tiles = 36 workgroups; 360M: 60; 1.7B: 128.
        assert_eq!(rows_for(0, shape(1, 9, 256)), 32);
        assert_eq!(rows_for(0, shape(1, 15, 256)), 32);
        assert_eq!(rows_for(0, shape(1, 32, 256)), 32);
        // Batch 8 of 135M: 288 workgroups; exactly 160 is full.
        assert_eq!(rows_for(0, shape(8, 9, 256)), 64);
        assert_eq!(rows_for(0, shape(1, 40, 256)), 64);
        assert_eq!(rows_for(0, shape(1, 39, 256)), 32);
        // A forced size wins.
        assert_eq!(rows_for(64, shape(1, 9, 256)), 64);
        assert_eq!(rows_for(32, shape(8, 9, 256)), 32);
    }

    #[test]
    fn the_tagged_lines_follow_the_row_size() {
        let wide = source(64);
        let narrow = source(32);
        for src in [&wide, &narrow] {
            assert!(!src.contains("//@"), "a tag is left");
            assert!(!src.contains("@BR@") && !src.contains("@TA@"));
            assert!(!src.contains("@SS@") && !src.contains("@NC@"));
        }
        assert!(wide.contains("const FC_BR: u32 = 64u;"));
        assert!(narrow.contains("const FC_BR: u32 = 32u;"));
        // Only the wide kernels hold four column blocks of accumulators.
        assert!(
            wide.contains("var dq3 =") && wide.contains("var dk3 =") && wide.contains("var dv3 =")
        );
        assert!(!narrow.contains("dq2") && !narrow.contains("dk2") && !narrow.contains("dv2"));
        assert!(
            narrow.contains("var dq1 =")
                && narrow.contains("var dk1 =")
                && narrow.contains("var dv1 =")
        );
    }

    #[test]
    fn the_kernels_validate_with_cooperative_matrices() {
        for rows in [64, 32] {
            let src = source(rows);
            let module = match naga::front::wgsl::parse_str(&src) {
                Ok(m) => m,
                Err(e) => panic!(
                    "WGSL parse error ({rows} rows):\n{}",
                    e.emit_to_string(&src)
                ),
            };
            let mut validator =
                naga::valid::Validator::new(naga::valid::ValidationFlags::all(), capabilities());
            if let Err(e) = validator.validate(&module) {
                panic!(
                    "WGSL validation error ({rows} rows):\n{}",
                    e.emit_to_string(&src)
                );
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
    }

    #[test]
    fn the_kernels_validate_with_immediates() {
        for rows in [64, 32] {
            let src = immediate_source(&source(rows));
            assert!(!src.contains("var<uniform>"));
            let module = match naga::front::wgsl::parse_str(&src) {
                Ok(m) => m,
                Err(e) => panic!(
                    "WGSL parse error ({rows} rows):\n{}",
                    e.emit_to_string(&src)
                ),
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
                panic!(
                    "WGSL validation error ({rows} rows):\n{}",
                    e.emit_to_string(&src)
                );
            }
        }
    }

    /// The workgroup arrays of every entry point fit the smallest limit the
    /// kernels target (an A10G's 49,152 bytes), at both row sizes.
    #[test]
    fn workgroup_memory_fits_an_a10g() {
        for rows in [64usize, 32] {
            let src = source(rows as u32);
            let mut bytes = 0usize;
            for (name, kind, elements, size) in [
                ("fc_ta", "f16", rows * 72, 2),
                ("fc_tb", "f16", 4608, 2),
                ("fc_tc", "f16", 4608, 2),
                ("fc_ss", "f32", rows * 68, 4),
                ("fc_pm", "f32", 128, 4),
                ("fc_zs", "f32", 256, 4),
                ("fc_sl", "f32", 64, 4),
                ("fc_sd", "f32", 64, 4),
            ] {
                assert!(
                    src.contains(&format!(
                        "var<workgroup> {name}: array<{kind}, {elements}>;"
                    )),
                    "{name} at {rows} rows"
                );
                assert_eq!(
                    elements * size % 16,
                    0,
                    "{name} is not a multiple of 16 bytes"
                );
                bytes += elements * size;
            }
            assert!(bytes <= 49_152, "{bytes} bytes at {rows} rows");
        }
    }

    #[test]
    fn the_rope_table_holds_cosines_then_sines_per_position() {
        let theta = 10_000.0_f32;
        let table = rope_table(Rope { theta, pos: 3 }, 5);
        assert_eq!(table.len(), 5 * 64);
        for t in 0..5usize {
            let row = &table[t * 64..(t + 1) * 64];
            for d in 0..32usize {
                let angle = (3.0 + t as f64) * f64::from(theta).powf(-2.0 * d as f64 / 64.0);
                assert!(
                    (f64::from(row[d]) - angle.cos()).abs() < 2e-4,
                    "cos t={t} d={d}"
                );
                assert!(
                    (f64::from(row[32 + d]) - angle.sin()).abs() < 2e-4,
                    "sin t={t} d={d}"
                );
            }
        }
        // Position 0 is the identity.
        let first = rope_table(Rope { theta, pos: 0 }, 1);
        assert!(first[..32].iter().all(|&c| c == 1.0));
        assert!(first[32..].iter().all(|&s| s == 0.0));
    }
}
