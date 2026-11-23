//! Matmuls on cooperative matrices: `out = alpha * a @ op(b) + beta * out`
//! with f32 `a`, f16 `b`, 16x16x16 f16 fragments and f32 accumulation.
//!
//! The kernel stages `a` (converted to f16) and `b` in workgroup memory, one
//! K step of 32 or 64 at a time (with two staging buffers the next step loads
//! while this one multiplies), and each subgroup of 32 threads multiplies its
//! own block of 16x16 fragments. The accumulators go back through workgroup
//! memory, so the epilogue and the edges of M and N that are not multiples
//! of 16 are handled per element. A product with too few output tiles to
//! fill the GPU splits K over the grid's third dimension, and the split-K
//! reduction adds the slices in order.
//!
//! A value of `a` that f16 cannot hold (beyond +-65504, or NaN) is clamped
//! for the product, counted in [`ComputeEngine::coop_f16_overflows`], and the
//! first element of that workgroup's output tile is set to NaN, so a dynamic
//! loss scaler sees the overflow as a non-finite gradient.
//!
//! Every workgroup array is a multiple of 16 bytes long. The driver packs
//! them one after another, and a cooperative load or store from an address
//! that is not 16-byte aligned reads or writes the aligned address below it
//! (on the A10G a 4-byte atomic among them shifted the results by elements).
//!
//! The module is split by what a later change touches: `tiles` holds the tile
//! shapes, `plan` the router that picks a tile and a K split, `loader` the
//! staging of the operands, `mainloop` the K loop and `epilogue` the writes
//! back, and `autotune` (with its cache file in `autotune_cache`) times the
//! candidates where they run, when `FERROTENSOR_AUTOTUNE=1`. This file holds
//! the kernel template, the cache of compiled kernels and the engine methods.

use super::{grid, immediate_params, uniform, ComputeEngine, Epilogue, Kernel};
use crate::kernel_timing::KernelLabel;
use crate::shader_checks::ShaderChecks;
use crate::{DType, ReadbackError, UnifiedTensor, WgpuContext};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};
use std::time::Instant;

mod autotune;
mod autotune_cache;
mod epilogue;
mod fused;
mod loader;
mod mainloop;
mod plan;
mod tiles;

pub(crate) use autotune::autotune_requested;
use autotune::{candidates, TuneKey, Tuner};
pub use autotune::{AutotuneReport, AUTOTUNE_ENV};
use epilogue::{poison, store_fragment};
pub(super) use epilogue::{EpilogueKind, BIAS_BINDING, FIRST_BINDING, MAX_SEGMENTS};
use fused::params_vec4s;
pub(super) use fused::{params_vec4s as kind_params_vec4s, reference_source};
pub(super) use loader::{LoaderKind, MAX_SOURCES, SOURCE_BINDING};
use mainloop::main_loop;
use plan::{grid_fits, plan, split_k};
pub use tiles::{CoopConfig, CoopTile};
use tiles::{PAD, SUBGROUP};

/// Fewest output rows routed to the kernel: decode steps and thin products
/// stay on the f32 kernels. At 32 to 63 rows the kernel with the planner's
/// split beat the small-M kernel in 46 of 48 products of SmolLM2 135M, 360M
/// and 1.7B on an A10G (`matmul_bench --fused`, 20 November 2026), so those
/// rows run here too.
const MIN_ROWS: u32 = 32;

/// Fewest output rows of a product whose kernel is compiled without index
/// bounds clamps and without loop bounding (see [`Coop::kernel`]).
const UNCHECKED_MIN_ROWS: u32 = 256;

/// Row tiles that consecutive workgroups walk inside one column block before
/// the next column block starts (see the tile map in [`source`]).
const RASTER_GROUP: u32 = 8;

/// Column tiles from which a product walks the grid in groups. On an A10G
/// (notes/raster-2026-11-23.md) the grouped walk was 8 to 9% faster for the
/// gate products at 8192 columns (64 tiles of 128 columns) and 25% faster for
/// the head at 49152 columns, and no faster at 2048 columns (16 to 32 tiles).
/// With few column tiles a wave already spans all of them, so the order has
/// nothing to share. 48 lies between 32 and 64.
const GROUPED_MIN_TILES_N: u32 = 48;

/// Whether the kernel of a product walks the grid in groups of row tiles
/// (see [`RASTER_GROUP`]). Only a forward product against a wide weight stored
/// `[N, K]` and read by the plain loader gains from it. The `[K, N]` products
/// were no faster and the backward ones up to 13% slower (their `n` is the
/// model width, so the wave has little to share), and so was the fused qkv
/// backward, which reads its `a` through a fused loader. Those keep the
/// column-fastest order.
fn grouped_raster(loader: LoaderKind, transposed_b: bool, tile: CoopTile, n: u32) -> bool {
    loader == LoaderKind::Plain && transposed_b && n.div_ceil(tile.bn) >= GROUPED_MIN_TILES_N
}

/// [`RASTER_GROUP`], or the value of `FERROTENSOR_RASTER_GROUP` (read once, for
/// bench sweeps; 1 is the earlier column-fastest order for every product).
fn raster_group() -> u32 {
    static GROUP: OnceLock<u32> = OnceLock::new();
    *GROUP.get_or_init(|| {
        std::env::var("FERROTENSOR_RASTER_GROUP")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&g| g > 0)
            .unwrap_or(RASTER_GROUP)
    })
}

/// What one compiled kernel is specialised for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct CoopKey {
    tile: CoopTile,
    /// `b` stored `[N, K]` (true) or `[K, N]`.
    transposed_b: bool,
    /// `a` read as `vec4<f32>`, which needs K to be a multiple of 4.
    a_vec4: bool,
    /// `b` read 8 halves at a time, which needs its row length (N, or K when
    /// transposed) to be a multiple of 8.
    b_vec8: bool,
    /// How `a` and `b` reach the staging tiles.
    loader: LoaderKind,
    /// What happens to the accumulators.
    epilogue: EpilogueKind,
    /// Whether workgroups walk the grid in groups of row tiles (see
    /// [`grouped_raster`]) rather than along the columns first.
    grouped: bool,
}

/// The cooperative-matrix kernels of a device that has them.
pub(super) struct Coop {
    /// By key and whether the module was compiled without bounds checks.
    kernels: Mutex<HashMap<(CoopKey, bool), Arc<Kernel>>>,
    /// The reduction passes of split fused products, by epilogue.
    fused_reduce: Mutex<HashMap<EpilogueKind, Arc<Kernel>>>,
    /// One `u32` counter of staging threads that clamped an `a` value (or a
    /// value the attention kernels staged).
    pub(super) overflows: UnifiedTensor,
    /// The attention kernels, compiled with the matmul's.
    pub(super) flash: Option<super::flash_coop::FlashCoop>,
    /// Whether [`EXPERIMENTAL_ENV`](crate::adapters::EXPERIMENTAL_ENV) routes f16-weight matmuls here.
    routed: bool,
    /// Workgroups a grid aims for, as for the small-M kernel.
    target: u32,
    /// Whether the `TUNED` table of `plan` applies: it was measured on a discrete GPU.
    tuned: bool,
    max_bytes: u32,
    max_threads: u32,
    /// Whether the kernel reads its parameters from immediates.
    immediates: bool,
    /// The autotuner; `None` unless `FERROTENSOR_AUTOTUNE` asks for it on a
    /// device that routes matmuls here and times dispatches.
    tuner: Option<Mutex<Tuner>>,
}

/// The adapter line of the autotune cache file for `ctx`: what a cached
/// choice depends on besides the shape.
fn adapter_header(ctx: &WgpuContext) -> String {
    let info = ctx.adapter.get_info();
    autotune_cache::header_text(
        (&info.name, info.vendor, info.device),
        (&info.driver, &info.driver_info),
        &format!("{:?}", info.backend),
    )
}

/// The kernel `tile` runs a product with `b` stored as `transposed_b` says
/// and a K and N of `(k, n)`.
fn coop_key(tile: CoopTile, transposed_b: bool, (k, n): (u32, u32)) -> CoopKey {
    let b_row = if transposed_b { k } else { n };
    CoopKey {
        tile,
        transposed_b,
        a_vec4: k.is_multiple_of(4),
        b_vec8: b_row.is_multiple_of(8),
        loader: LoaderKind::Plain,
        epilogue: EpilogueKind::AlphaBeta,
        grouped: grouped_raster(LoaderKind::Plain, transposed_b, tile, n),
    }
}

/// `src` with its parameters moved to immediates. The `enable` directives
/// must stay first in the module, so only the text after them is rewritten.
pub(super) fn immediate_source(src: &str) -> String {
    const LAST_ENABLE: &str = "enable wgpu_cooperative_matrix;\n";
    let split = src.find(LAST_ENABLE).map_or(0, |at| at + LAST_ENABLE.len());
    let (directives, body) = src.split_at(split);
    format!("{directives}{}", immediate_params(body))
}

/// Whether `ctx` can run the kernel: f16 in shaders, 16x16x16 f16
/// fragments accumulated in f32, and subgroups of exactly 32 threads.
fn device_supports(ctx: &WgpuContext) -> bool {
    let wanted = wgpu::Features::SHADER_F16 | wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX;
    let info = ctx.adapter.get_info();
    ctx.device.features().contains(wanted)
        && info.subgroup_min_size == SUBGROUP
        && info.subgroup_max_size == SUBGROUP
        && ctx
            .cooperative_matrix()
            .iter()
            .any(|c| (c.m, c.n, c.k) == (16, 16, 16) && c.ab_type == "f16" && c.cr_type == "f32")
}

impl Coop {
    /// The kernels for `ctx`, or `None` when its device cannot run them.
    pub(super) fn new(ctx: &WgpuContext, target: u32) -> Option<Self> {
        if !device_supports(ctx) {
            return None;
        }
        let limits = ctx.device.limits();
        let routed = crate::adapters::experimental_opt_in();
        let tuner = Tuner::from_env(
            &|name| std::env::var(name).ok(),
            (routed, ctx.has_timestamps()),
            || adapter_header(ctx),
        );
        Some(Self {
            kernels: Mutex::new(HashMap::new()),
            fused_reduce: Mutex::new(HashMap::new()),
            overflows: UnifiedTensor::zeros(ctx, &[1]),
            flash: Some(super::flash_coop::FlashCoop::new()),
            routed,
            target,
            tuned: ctx.adapter.get_info().device_type == wgpu::DeviceType::DiscreteGpu,
            max_bytes: limits.max_compute_workgroup_storage_size,
            max_threads: limits
                .max_compute_invocations_per_workgroup
                .min(limits.max_compute_workgroup_size_x),
            immediates: ctx.uses_immediates(),
            tuner: tuner.map(Mutex::new),
        })
    }

    fn tuner(&self) -> Option<MutexGuard<'_, Tuner>> {
        self.tuner
            .as_ref()
            .map(|t| t.lock().unwrap_or_else(PoisonError::into_inner))
    }

    /// The configuration and sample id to run an `m x k x n` product of the
    /// kernel variant `kinds` (its loader and epilogue) with when the static
    /// rules pick `static_cfg`: the tuner's pick, and the sample to tag the
    /// run with when it is a timing. Without a tuner, the static pick and no
    /// sample.
    fn autotuned(
        &self,
        ctx: &WgpuContext,
        transposed_b: bool,
        kinds: (LoaderKind, EpilogueKind),
        dims: (u32, u32, u32),
        static_cfg: CoopConfig,
    ) -> (CoopConfig, Option<u64>) {
        let Some(mut tuner) = self.tuner() else {
            return (static_cfg, None);
        };
        if tuner.wants_samples() {
            for (sample, time) in ctx.take_tune_samples() {
                tuner.record(sample, time);
            }
        }
        let (m, _, n) = dims;
        let key = TuneKey::new(transposed_b, kinds, dims);
        let usable =
            |c: CoopConfig| self.fits(c.tile, transposed_b) && grid_fits(c.tile, m, n, c.split);
        let list = || {
            candidates(dims, key.m_bucket(), self.target, &|t| {
                self.fits(t, transposed_b)
            })
        };
        let pick = tuner.pick(key, m, static_cfg, &usable, &list);
        (pick.config, pick.sample)
    }

    fn fits(&self, tile: CoopTile, transposed_b: bool) -> bool {
        tile.threads() <= self.max_threads && tile.workgroup_bytes(transposed_b) <= self.max_bytes
    }

    /// The compiled kernel for `key`, for a product of `rows` output rows.
    fn kernel(&self, ctx: &WgpuContext, key: CoopKey, rows: u32) -> Arc<Kernel> {
        // From UNCHECKED_MIN_ROWS rows up, and while FERROTENSOR_SHADER_CHECKS is
        // unset, the module is built without bounds checks and without loop
        // bounding. That is safe because every global read and write of the
        // kernel has its own guard on `m`, `n` and `k` (`gr < m && gk < k` in
        // the `a` loads, `gn < n && gk < k` in the `b` loads, `r < m && col < n`
        // in every epilogue); the grid is `ceil(m / BM)` by `ceil(n / BN)`, so
        // `m0 < m` and `n0 < n` for the poison write (the grouped raster maps
        // the ids of that grid one to one onto its tiles); the host checks that each
        // buffer holds its shape; workgroup-memory indices are bounded by the
        // tile sizes; and every loop runs a constant number of times or
        // `steps` times. The access-by-access audit is in
        // notes/coop-unchecked-2026-11-22.md. Smaller products keep the
        // context's checks, and the two builds are cached apart.
        let unchecked = rows >= UNCHECKED_MIN_ROWS && !ShaderChecks::env_is_set();
        let mut cache = self.kernels.lock().unwrap_or_else(PoisonError::into_inner);
        Arc::clone(cache.entry((key, unchecked)).or_insert_with(|| {
            let mut src = source(key);
            if self.immediates {
                src = immediate_source(&src);
            }
            let checks = if unchecked {
                ShaderChecks {
                    loop_bounding: false,
                    bounds: false,
                    ..ctx.shader_checks()
                }
            } else {
                ctx.shader_checks()
            };
            let module = checks.module(&ctx.device, "matmul_coop", src);
            let pipeline = Arc::new(ctx.device.create_compute_pipeline(
                &wgpu::ComputePipelineDescriptor {
                    label: Some("matmul_coop"),
                    layout: None,
                    module: &module,
                    entry_point: Some("matmul_coop"),
                    // The kernel writes every workgroup value before reading it.
                    compilation_options: wgpu::PipelineCompilationOptions {
                        zero_initialize_workgroup_memory: false,
                        ..Default::default()
                    },
                    cache: None,
                },
            ));
            let layout = pipeline.get_bind_group_layout(0);
            Arc::new(Kernel {
                pipeline,
                layout,
                entry_point: "matmul_coop",
            })
        }))
    }
}

impl ComputeEngine {
    /// Whether this device can run matmuls on cooperative matrices (f16
    /// inputs, f32 accumulation), whether or not they are turned on.
    #[must_use]
    pub fn coop_matmul_available(&self) -> bool {
        self.coop.is_some()
    }

    /// Whether matmuls against f16 weights run on cooperative matrices: the
    /// device has them and [`EXPERIMENTAL_ENV`](crate::adapters::EXPERIMENTAL_ENV) is `1`.
    #[must_use]
    pub fn coop_matmul_active(&self) -> bool {
        self.coop.as_ref().is_some_and(|c| c.routed)
    }

    /// Staging threads of the cooperative-matrix matmul that met an input
    /// value beyond the f16 range (or NaN) since the engine was created; 0
    /// without the kernel. Reading it waits for the GPU.
    ///
    /// # Errors
    ///
    /// [`ReadbackError`] when wgpu cannot map the counter for reading.
    pub async fn coop_f16_overflows(&self, ctx: &WgpuContext) -> Result<u64, ReadbackError> {
        Ok(match self.coop.as_ref() {
            Some(coop) => coop
                .overflows
                .to_vec(ctx)
                .await?
                .first()
                .map_or(0, |v| u64::from(v.to_bits())),
            None => 0,
        })
    }

    /// Whether [`matmul_run`](Self::matmul_run) sends an `m x k x n` product
    /// against a `dtype` weight to the cooperative-matrix kernel.
    pub(super) fn coop_routes(&self, dtype: DType, (m, k, n): (u32, u32, u32)) -> bool {
        self.coop_matmul_active() && dtype == DType::FP16 && m >= MIN_ROWS && k >= 16 && n >= 16
    }

    /// Runs `out = alpha * a[M, K] @ op(b) + beta * out` (`ep`) on the
    /// cooperative-matrix kernel, with f16 `b` stored `[N, K]`
    /// (`transposed_b`) or `[K, N]`, whether or not the router would send
    /// the product there.
    ///
    /// # Errors
    ///
    /// Returns an error when the device has no cooperative matrices, `b` is
    /// not f16, the shapes do not match or no tile fits the device.
    #[doc(hidden)]
    pub fn matmul_coop_into(
        &self,
        ctx: &WgpuContext,
        transposed_b: bool,
        tensors: [&UnifiedTensor; 3],
        ep: Epilogue,
    ) -> Result<(), String> {
        self.matmul_coop_config_into(ctx, transposed_b, tensors, None, ep)
            .map(|_| ())
    }

    /// Runs [`matmul_coop_into`](Self::matmul_coop_into) with `config`, or
    /// with the configuration the router would pick when it is `None`, and
    /// returns the configuration that ran (a split can come out smaller, so
    /// that no slice of K is empty).
    ///
    /// # Errors
    ///
    /// As [`matmul_coop_into`](Self::matmul_coop_into), and when `config`'s
    /// tile does not fit this device or its grid is past the dispatch limit.
    #[doc(hidden)]
    pub fn matmul_coop_config_into(
        &self,
        ctx: &WgpuContext,
        transposed_b: bool,
        [a, b, out]: [&UnifiedTensor; 3],
        config: Option<CoopConfig>,
        ep: Epilogue,
    ) -> Result<CoopConfig, String> {
        let Some(coop) = self.coop.as_ref() else {
            return Err("matmul_coop_into: this device has no f16 cooperative matrices".into());
        };
        if b.dtype != DType::FP16 {
            return Err("matmul_coop_into: b must be f16".into());
        }
        let ([m, k], [b0, b1]) = (a.shape.as_slice(), b.shape.as_slice()) else {
            return Err(format!(
                "matmul_coop_into: a and b must be 2-D, got {:?} and {:?}",
                a.shape, b.shape
            ));
        };
        let (b_k, n) = if transposed_b { (b1, b0) } else { (b0, b1) };
        if b_k != k || out.shape.iter().product::<usize>() != m * n {
            return Err(format!(
                "matmul_coop_into: a {:?}, b {:?} (transposed_b {transposed_b}) and out {:?} do not match",
                a.shape, b.shape, out.shape
            ));
        }
        if *m == 0 || *k == 0 || *n == 0 {
            return Err("matmul_coop_into: empty product".into());
        }
        let dims = (*m as u32, *k as u32, *n as u32);
        let fits = |t: CoopTile| coop.fits(t, transposed_b);
        let config = match config {
            Some(c) if fits(c.tile) && grid_fits(c.tile, dims.0, dims.2, c.split) => Some(c),
            Some(c) => {
                return Err(format!(
                    "matmul_coop_into: {} with split {} does not fit {m} x {n} on this device",
                    c.tile.name(),
                    c.split
                ));
            }
            None => plan(dims, transposed_b, (coop.target, coop.tuned), fits),
        };
        let Some(config) = config else {
            return Err(format!(
                "matmul_coop_into: no tile fits {m} x {n} on this device"
            ));
        };
        self.matmul_coop_run(ctx, transposed_b, dims, [a, b, out], (config, None), ep)
            .ok_or_else(|| "matmul_coop_into: no cooperative-matrix kernel".into())
    }

    /// Records the cooperative-matrix matmul; false (recording nothing) when
    /// the device has no kernel or no tile fits.
    pub(super) fn matmul_coop(
        &self,
        ctx: &WgpuContext,
        transposed_b: bool,
        dims: (u32, u32, u32),
        tensors: [&UnifiedTensor; 3],
        ep: Epilogue,
    ) -> bool {
        let Some(coop) = self.coop.as_ref() else {
            return false;
        };
        let target = (coop.target, coop.tuned);
        let Some(config) = plan(dims, transposed_b, target, |t| coop.fits(t, transposed_b)) else {
            return false;
        };
        let (config, tune) = coop.autotuned(
            ctx,
            transposed_b,
            (LoaderKind::Plain, EpilogueKind::AlphaBeta),
            dims,
            config,
        );
        self.matmul_coop_run(ctx, transposed_b, dims, tensors, (config, tune), ep)
            .is_some()
    }

    /// Reads the autotuner's timings that have arrived, saves its cache file
    /// and reports what it did; `None` when autotuning is off (or the
    /// device has no cooperative matrices). The GPU work should have been
    /// waited for, or its last timings are missed.
    pub fn autotune_finish(&self, ctx: &WgpuContext) -> Option<AutotuneReport> {
        let mut tuner = self.coop.as_ref()?.tuner()?;
        for (sample, time) in ctx.take_tune_samples() {
            tuner.record(sample, time);
        }
        tuner.save();
        Some(tuner.report())
    }

    /// Records `config` for the product: one pass that applies `ep`, or with
    /// K split, one pass of partial sums and the reduction that adds them in
    /// slice order and applies `ep`. The dispatches carry the autotuner's
    /// sample id `tune` when the run is a timing. Returns the configuration
    /// that ran, or `None` (recording nothing) without the kernel.
    fn matmul_coop_run(
        &self,
        ctx: &WgpuContext,
        transposed_b: bool,
        (m, k, n): (u32, u32, u32),
        [a, b, out]: [&UnifiedTensor; 3],
        (config, tune): (CoopConfig, Option<u64>),
        ep: Epilogue,
    ) -> Option<CoopConfig> {
        let coop = self.coop.as_ref()?;
        let tile = config.tile;
        let (split, kps) = split_k(k, tile.bk, config.split);
        let key = coop_key(tile, transposed_b, (k, n));
        let compiling = Instant::now();
        let kernel = coop.kernel(ctx, key, m);
        if tune.is_some() {
            // A trial's compile time is part of what tuning costs.
            if let Some(mut tuner) = coop.tuner() {
                tuner.charge(compiling.elapsed());
            }
        }
        // With K split the kernel stores plain partial sums; the reduction
        // applies the epilogue once.
        let kernel_ep = if split == 1 { ep } else { Epilogue::STORE };
        let (alpha, beta) = (kernel_ep.alpha.to_bits(), kernel_ep.beta.to_bits());
        let words = [m, k, n, kps, alpha, beta, 0, 0];
        let params = uniform(ctx, bytemuck::cast_slice(&words));
        let label = KernelLabel {
            entry: "matmul_coop",
            layout: if transposed_b { "a@bT" } else { "a@b" },
            size_class: if m <= 256 { "m<=256" } else { "m>256" },
        };
        let (gx, gy) = tile.groups(m, n);
        if split == 1 {
            self.run_tagged(
                ctx,
                &kernel,
                label,
                &[
                    (0, a.buffer()),
                    (1, b.buffer()),
                    (2, out.buffer()),
                    (3, &params),
                    (4, coop.overflows.buffer()),
                ],
                (gx, gy, 1),
                tune,
            );
        } else {
            let parts = UnifiedTensor::empty(ctx, &[split as usize, m as usize, n as usize]);
            self.run_tagged(
                ctx,
                &kernel,
                label,
                &[
                    (0, a.buffer()),
                    (1, b.buffer()),
                    (2, parts.buffer()),
                    (3, &params),
                    (4, coop.overflows.buffer()),
                ],
                (gx, gy, split),
                tune,
            );
            let len = m * n;
            let params = uniform(
                ctx,
                bytemuck::cast_slice(&[len, split, ep.alpha.to_bits(), ep.beta.to_bits()]),
            );
            let reduce = KernelLabel {
                entry: self.split_reduce.entry_point,
                layout: "",
                size_class: "",
            };
            self.run_tagged(
                ctx,
                &self.split_reduce,
                reduce,
                &[(0, parts.buffer()), (1, out.buffer()), (2, &params)],
                grid(len.div_ceil(256)),
                tune,
            );
        }
        Some(CoopConfig { tile, split })
    }
}

/// The tile of workgroup `wg` when the grid is walked along the columns first.
const COLUMN_TILE_MAP: &str = "  let m0 = wg.y * BM;
  let n0 = wg.x * BN;
";

/// The tile of workgroup `wg` when the grid is walked in groups of row tiles.
const GROUPED_TILE_MAP: &str =
    "  // The grid is tiles_n by tiles_m, x fastest. Workgroup ids are dealt out in
  // groups of GROUP row tiles: a group walks down one column block, then moves
  // to the next, so the workgroups that run at the same time share the weight
  // columns they read. The last group has the row tiles that are left. This
  // is a one-to-one map of ids to tiles.
  let tiles_m = (m + BM - 1u) / BM;
  let tiles_n = (n + BN - 1u) / BN;
  let id = wg.y * tiles_n + wg.x;
  let first = (id / (GROUP * tiles_n)) * GROUP;
  let rows = min(tiles_m - first, GROUP);
  let within = id % (GROUP * tiles_n);
  let m0 = (first + within % rows) * BM;
  let n0 = (within / rows) * BN;
";

/// The WGSL of the kernel for `key`, entry point `matmul_coop`. Params are
/// `[M, K, N, K per slice]` and `[alpha bits, beta bits, 0, 0]`, then, for a
/// fused loader or epilogue, its column bounds; workgroup z is the slice of
/// K, whose sums go to `c[z * M * N ..]`.
pub(super) fn source(key: CoopKey) -> String {
    let t = key.tile;
    let (ti, tj) = (t.wm / 16, t.wn / 16);
    let warps = t.warps();
    let mut init = String::new();
    let mut store = String::new();
    for i in 0..ti {
        for j in 0..tj {
            init.push_str(&format!(
                "  var acc{i}_{j} = coopLoadT<coop_mat16x16<f32, C>>(&cs[0], 16u);\n"
            ));
            store.push_str(&store_fragment(key.epilogue, i, j));
        }
    }
    let (a_len, b_len) = (t.a_len(), t.b_len(key.transposed_b));
    let mut arrays = format!(
        "var<workgroup> sa0: array<f16, {a_len}>;\nvar<workgroup> sb0: array<f16, {b_len}>;\n"
    );
    if t.double {
        arrays.push_str(&format!(
            "var<workgroup> sa1: array<f16, {a_len}>;\nvar<workgroup> sb1: array<f16, {b_len}>;\n"
        ));
    }
    // A group of 1 is the column-fastest order, so that case keeps the plain
    // text: the kernel of every ungrouped product is the one from before the
    // raster, which the pinned hashes check.
    let group = raster_group();
    let (group_const, tile_map) = if key.grouped && group > 1 {
        (
            format!("\nconst GROUP: u32 = {group}u;"),
            GROUPED_TILE_MAP.to_string(),
        )
    } else {
        (String::new(), COLUMN_TILE_MAP.to_string())
    };
    let extra_bindings =
        loader::bindings(key.loader, key.a_vec4) + &epilogue::bindings(key.epilogue);
    let helpers = loader::helpers(key.loader, key.a_vec4) + &epilogue::helpers(key.epilogue);
    format!(
        "enable f16;
enable wgpu_cooperative_matrix;

@group(0) @binding(0) var<storage, read> a: array<{a_ty}>;
@group(0) @binding(1) var<storage, read> b: array<{b_ty}>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> params: array<vec4<u32>, {params_len}>;
@group(0) @binding(4) var<storage, read_write> overflows: array<atomic<u32>>;
{extra_bindings}
const BM: u32 = {bm}u;
const BN: u32 = {bn}u;
const BK: u32 = {bk}u;
const AP: u32 = {ap}u;
const BP: u32 = {bp}u;{group_const}
const F16_MAX: f32 = 65504.0;

{arrays}var<workgroup> cs: array<f32, {cs_len}>;
{helpers}
@compute @workgroup_size({nt})
fn matmul_coop(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
  let m = params[0].x;
  let k = params[0].y;
  let n = params[0].z;
  let alpha = bitcast<f32>(params[1].x);
  let beta = bitcast<f32>(params[1].y);
  let warp = li / 32u;
  let lane = li % 32u;
  let wr = (warp / {warps_n}u) * {wm}u;
  let wc = (warp % {warps_n}u) * {wn}u;
{tile_map}  // This workgroup's slice of K, [kb, ke), and where its sums go. Slices
  // are whole steps of BK, so a step never reads into the next slice.
  let kb = wg.z * params[0].w;
  let ke = min(k, kb + params[0].w);
  let steps = select(0u, (ke - kb + BK - 1u) / BK, kb < ke);
  let base = wg.z * m * n;
  for (var i = li; i < {cs_len}u; i += {nt}u) {{ cs[i] = 0.0; }}
  workgroupBarrier();
{init}  var overflow = false;
{main}{store}  // After the epilogue's writes, so the poison is the last write to that element.
  storageBarrier();
  if (overflow) {{
    atomicAdd(&overflows[0], 1u);
    {poison}
  }}
}}
",
        params_len = params_vec4s(key.loader, key.epilogue),
        poison = poison(key.epilogue),
        a_ty = if key.a_vec4 { "vec4<f32>" } else { "f32" },
        b_ty = if key.b_vec8 { "vec4<u32>" } else { "u32" },
        bm = t.bm,
        bn = t.bn,
        bk = t.bk,
        ap = t.bk + PAD,
        bp = t.bn + PAD,
        cs_len = warps * 256,
        nt = t.threads(),
        warps_n = t.bn / t.wn,
        wm = t.wm,
        wn = t.wn,
        main = main_loop(key),
    )
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests {
    use super::{
        grouped_raster, immediate_source, plan, source, CoopKey, CoopTile, EpilogueKind, LoaderKind,
    };

    /// Every kernel, ungrouped and grouped.
    fn keys() -> Vec<CoopKey> {
        let mut keys = keys_with(false);
        keys.extend(keys_with(true));
        keys
    }

    fn keys_with(grouped: bool) -> Vec<CoopKey> {
        let mut keys = Vec::new();
        for tile in CoopTile::ALL {
            for transposed_b in [false, true] {
                for a_vec4 in [false, true] {
                    for b_vec8 in [false, true] {
                        keys.push(CoopKey {
                            tile,
                            transposed_b,
                            a_vec4,
                            b_vec8,
                            loader: LoaderKind::Plain,
                            epilogue: EpilogueKind::AlphaBeta,
                            grouped,
                        });
                    }
                }
            }
        }
        keys
    }

    #[test]
    fn every_kernel_validates_with_cooperative_matrices() {
        for key in keys() {
            let src = source(key);
            let module = match naga::front::wgsl::parse_str(&src) {
                Ok(m) => m,
                Err(e) => panic!("{key:?}: WGSL parse error:\n{}", e.emit_to_string(&src)),
            };
            let mut validator = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::COOPERATIVE_MATRIX
                    | naga::valid::Capabilities::SHADER_FLOAT16
                    | naga::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32,
            );
            if let Err(e) = validator.validate(&module) {
                panic!(
                    "{key:?}: WGSL validation error:\n{}",
                    e.emit_to_string(&src)
                );
            }
        }
    }

    #[test]
    fn every_kernel_validates_with_immediates() {
        for key in keys() {
            let src = immediate_source(&source(key));
            assert!(!src.contains("var<uniform>"), "{key:?}");
            let module = match naga::front::wgsl::parse_str(&src) {
                Ok(m) => m,
                Err(e) => panic!("{key:?}: WGSL parse error:\n{}", e.emit_to_string(&src)),
            };
            assert_eq!(
                naga::valid::ImmediateSlots::size_for_module(&module),
                crate::engine::IMMEDIATE_BYTES,
                "{key:?}"
            );
            let mut validator = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::COOPERATIVE_MATRIX
                    | naga::valid::Capabilities::SHADER_FLOAT16
                    | naga::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32
                    | naga::valid::Capabilities::IMMEDIATES,
            );
            if let Err(e) = validator.validate(&module) {
                panic!(
                    "{key:?}: WGSL validation error:\n{}",
                    e.emit_to_string(&src)
                );
            }
        }
    }

    fn fnv1a(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
        })
    }

    /// FNV-1a 64 of the WGSL of every key of [`keys_with`]`(false)` in order,
    /// taken from the generator before it was split out of `matmul_coop.rs`.
    /// They are also the hashes of the kernels before the grouped raster, so an
    /// ungrouped product runs the same WGSL as before it.
    const SOURCE_HASHES: [u64; 112] = [
        11561312270760404045,
        16338313531991227733,
        9010546721910603871,
        7214329663846250259,
        9607423787901123493,
        8128846506299421923,
        7239040440319821139,
        3412286513356263805,
        15595660647109403214,
        579611057845456106,
        601902565835543782,
        4099647960230469486,
        12502018620123221006,
        5459227258467642672,
        12443708731755832342,
        7308542388084856196,
        4897971386158317493,
        3297169156890725017,
        1984468199060606420,
        13158515166100180318,
        7814938532924414675,
        11530719362987660629,
        1999278991572451510,
        13220500856541170890,
        6941566466834223097,
        8836585966891193904,
        1915243291982673644,
        13435980326272029181,
        15919405857711020237,
        4359916927262016070,
        17201816066131280700,
        3909783345644213739,
        8991267765447629719,
        12874584056793892770,
        10502676686481907884,
        14285653725314264009,
        3551184382933523739,
        17457262648324930164,
        17192545923066284116,
        6716064907120819239,
        5705213084781656430,
        11196061078061873843,
        13844279647717840859,
        5231307273082178342,
        14140342794670242888,
        11761345679114077775,
        11380179797312661621,
        4381321902460619430,
        5647962463574057465,
        4695158620340452280,
        9175515654119603634,
        1132382089057769739,
        15278350468023714265,
        906434759824789146,
        5627191109216927730,
        7449795567811122673,
        8847409965015831360,
        1628376108900066209,
        10134675602787364615,
        5822001216917074486,
        699834945472453194,
        2817226727796315169,
        2542265629306818229,
        2475930343174372170,
        17603989580741825145,
        14400635087337062584,
        3253416376929679960,
        12421142389766028381,
        16379276515244398249,
        13872524082229259710,
        12865437715842395396,
        10758315331840950747,
        14657062585320360824,
        13990108799418449171,
        16011988347940110159,
        17992581380810958460,
        8817136085765490806,
        18202174108224877779,
        6070401494196168761,
        2075198274904942004,
        519087706390674095,
        18422322472158980160,
        6322005717419358751,
        4796521073218356314,
        1995258627920552639,
        17294978783991052974,
        15576576722883854715,
        3979962472191413516,
        8509402793535655015,
        15985314220800305542,
        12372349888182113975,
        13210012642405315192,
        1787528791770232875,
        11607010899401908800,
        13160061647858470851,
        17437721834362212862,
        13981647278746972373,
        7140709782510070004,
        2564830905277548225,
        5833933849203377918,
        18314251420692429939,
        3023482341568362160,
        12407080566531949459,
        6184908848448135830,
        5197016299101613876,
        11412694723498967743,
        7844484645919387129,
        8658057791489555218,
        10501481690006856666,
        9418398957548768311,
        13147694747433239255,
        15820869452388351058,
    ];

    /// FNV-1a 64 of the WGSL of every key of [`keys_with`]`(true)` in order,
    /// with the default group of 8 (it is the text of the grouped raster).
    const GROUPED_SOURCE_HASHES: [u64; 112] = [
        6409254719821465726,
        12362183355494718850,
        15724217108186389712,
        11668617656355535208,
        9719651257319491322,
        8293494478080373484,
        16295030883298495024,
        12859698817071569934,
        540297674645252651,
        10658145239872514443,
        12050547188158091923,
        1297259160391276743,
        5986572268384075491,
        6478711574162007969,
        13841112332921757851,
        8020721159636024397,
        5103651184425438876,
        16381412007240750802,
        13045909008299483855,
        9146265108996819443,
        5196042121664466334,
        797257188504858390,
        17565182107796691049,
        9275142482074149811,
        1136056083983508660,
        1460774266848722553,
        9311408699763658465,
        4171795012858325140,
        10946245121702566964,
        8930931523793068651,
        500208016450841981,
        2791364511894921822,
        8930936733909741208,
        16980034154305236529,
        4903541771078071739,
        18077472802244353906,
        16539328802203004492,
        12817736338331090323,
        17390872808263024427,
        16991360231045983000,
        7688463622823301383,
        17780756982753378778,
        16464147515971255666,
        14222557830619853127,
        10636574255179610325,
        17056612978974465310,
        4182295171909621776,
        10703687835416452247,
        17521176236745891206,
        1354671279833546389,
        2882876554257660297,
        4892634333482524134,
        17391363820225031778,
        1669044325063327267,
        4611401005809321461,
        18006808931115494048,
        7371630166445411503,
        2040917113426781502,
        6553671754278406528,
        17175777862459508865,
        13855144656847690329,
        5924165076393021146,
        3049904525599951774,
        8346549215981208825,
        12294945230972176142,
        16967527435461176355,
        7015402264277623439,
        12940895244026491846,
        1590231311568550966,
        14903266123847589449,
        15450239571772660619,
        888239506653986460,
        16701096443851315029,
        10057982485102487408,
        4798908403422832918,
        212270811267863791,
        13175965949027332711,
        1095109731542368084,
        1484381595441487988,
        11692233741700498955,
        7414066871514876192,
        6656576184238384643,
        17470947115036275750,
        17265234031058390951,
        15884799721923416488,
        1513370883630330513,
        6633129632407665882,
        4049683377432685649,
        2091009567624558328,
        8131410943789649271,
        16725003153120438866,
        3877737658727968387,
        14587069163867877576,
        15455140122993461525,
        14405987295919378506,
        2858241301269138357,
        819456684246469768,
        15495532619810523913,
        7724375070566913926,
        9760290101852179353,
        9880167061212023838,
        11424101815086477733,
        9271220207619149096,
        4807978026365330609,
        5882131738330346735,
        13823116172736357624,
        1238853389870761070,
        1332224315573234625,
        16048981656201170125,
        1203357970495176840,
        16917445978796897052,
        9024970674147165913,
    ];

    #[test]
    fn generated_wgsl_is_what_it_was_before_the_split() {
        let hashes = |grouped| -> Vec<u64> {
            keys_with(grouped)
                .into_iter()
                .map(|key| fnv1a(source(key).as_bytes()))
                .collect()
        };
        assert_eq!(hashes(false), SOURCE_HASHES);
        assert_eq!(hashes(true), GROUPED_SOURCE_HASHES);
    }

    /// The `(k, n)` of the base-weight products of SmolLM2 135M, 360M and 1.7B,
    /// plus shapes with no table row and no multiple of 8 or 16.
    const KN: [(u32, u32); 24] = [
        (576, 576),
        (576, 192),
        (192, 576),
        (576, 1536),
        (1536, 576),
        (576, 49152),
        (49152, 576),
        (960, 960),
        (960, 320),
        (320, 960),
        (960, 2560),
        (2560, 960),
        (960, 49152),
        (49152, 960),
        (2048, 2048),
        (2048, 8192),
        (8192, 2048),
        (2048, 49152),
        (49152, 2048),
        (1000, 1000),
        (33, 17),
        (64, 4096),
        (4096, 64),
        (2050, 130),
    ];

    const ROWS: [u32; 16] = [
        16, 33, 64, 96, 106, 130, 200, 256, 257, 512, 1000, 1200, 1500, 2048, 2049, 4096,
    ];

    /// FNV-1a 64 of the configuration the selector picks for every product of
    /// [`KN`] x [`ROWS`] on four device descriptions, taken before the split
    /// and retaken (it was 3834093253146709999) when the 1024-row rows joined
    /// the tuned table, which can only move products of 257 to 1024 rows on a
    /// tuned device, and retaken again (it was 17442595157898385172) when the
    /// 64 to 256-row rows replaced the 256-row ones, which moves products of up
    /// to 256 rows on a tuned device, and retaken a third time (it was
    /// 1620736337270207953, with 1200 rows added to [`ROWS`]) when the 1280-row
    /// rows joined the table and the 2048-row rows of 360M and 1.7B were
    /// replaced, which moves products of 1025 to 2048 rows on a tuned device,
    /// and retaken a fourth time (it was 4766422471105062887) when the rows of
    /// 320, 384, 512 and 640 rows joined the table and the 1024, 1280 and 360M
    /// 2048-row rows of 360M and 1.7B were retuned, which moves products of
    /// 257 to 2048 rows on a tuned device.
    const PLAN_HASH: u64 = 9955952549834230422;

    #[test]
    fn selector_picks_what_it_picked_before_the_split() {
        type Fits = fn(CoopTile) -> bool;
        let devices: [(&str, u32, bool, Fits); 4] = [
            ("all", 160, true, |_| true),
            ("untuned", 160, false, |_| true),
            ("single", 160, true, |t| !t.double),
            ("small", 64, true, |t| t.bm * t.bn <= 64 * 64),
        ];
        let mut lines = String::new();
        for (name, target, use_tuned, fits) in devices {
            for (k, n) in KN {
                for transposed_b in [false, true] {
                    for m in ROWS {
                        let picked = plan((m, k, n), transposed_b, (target, use_tuned), fits)
                            .map_or_else(
                                || "none".to_string(),
                                |c| format!("{} s{}", c.tile.name(), c.split),
                            );
                        lines
                            .push_str(&format!("{name} m{m} k{k} n{n} t{transposed_b} {picked}\n"));
                    }
                }
            }
        }
        assert_eq!(fnv1a(lines.as_bytes()), PLAN_HASH);
    }

    /// The 1.7B products of a training step as `(name, k, n, transposed_b,
    /// grouped)`: `k` and `n` as the product sees them, with the layout of `b`.
    const PRODUCTS_1_7B: [(&str, u32, u32, bool, bool); 6] = [
        ("head fwd", 2048, 49152, true, true),
        ("gate fwd", 2048, 8192, true, true),
        ("head dX", 49152, 2048, false, false),
        ("gate dX", 8192, 2048, false, false),
        ("o dX", 2048, 2048, false, false),
        ("down dX", 2048, 8192, false, false),
    ];

    #[test]
    fn grouped_raster_is_for_the_wide_forward_products_of_1_7b() {
        // The tile the router picks at the three row counts of a 1.7B step on
        // a tuned device, and every tile of 64 or 128 columns.
        for (name, k, n, transposed_b, want) in PRODUCTS_1_7B {
            let mut tiles: Vec<CoopTile> = CoopTile::ALL
                .into_iter()
                .chain(CoopTile::PIPELINED)
                .filter(|t| t.bn >= 64)
                .collect();
            for m in [896, 1024, 1352] {
                let config = plan((m, k, n), transposed_b, (160, true), |_| true)
                    .expect("a tile fits the product");
                tiles.push(config.tile);
            }
            for tile in tiles {
                assert_eq!(
                    grouped_raster(LoaderKind::Plain, transposed_b, tile, n),
                    want,
                    "{name} with {}",
                    tile.name()
                );
            }
        }
    }

    #[test]
    fn grouped_raster_leaves_the_fused_loaders_and_the_qkv_backward_in_column_order() {
        // qkv dX is a@bT with K = 2048 + 2 * 2048 + 24 and N = 2048, read through
        // the Sources loader; the gate/up backward reads two sources too.
        let tile = CoopTile::T128X64_W64X32_K32;
        for loader in [
            LoaderKind::Sources { count: 4 },
            LoaderKind::Sources { count: 2 },
            LoaderKind::SwiGlu,
        ] {
            // Even a width that would qualify with the plain loader stays put.
            assert!(!grouped_raster(loader, true, tile, 2048), "{loader:?}");
            assert!(!grouped_raster(loader, true, tile, 49152), "{loader:?}");
        }
        // As a plain product (the step-0 bench) it is 32 column tiles of 64.
        assert!(!grouped_raster(LoaderKind::Plain, true, tile, 2048));
    }

    #[test]
    fn grouped_raster_threshold_sits_between_32_and_64_column_tiles() {
        let tile = CoopTile::T128X64_W64X32_K32;
        assert_eq!(tile.bn, 64);
        // 32 column tiles: N = 2048.
        assert!(!grouped_raster(LoaderKind::Plain, true, tile, 2048));
        // 47 and 48 column tiles (the last one partial).
        assert!(!grouped_raster(LoaderKind::Plain, true, tile, 47 * 64));
        assert!(grouped_raster(LoaderKind::Plain, true, tile, 47 * 64 + 1));
        // 135M and 360M: the gate is narrow, the head is as wide as at 1.7B.
        assert!(!grouped_raster(LoaderKind::Plain, true, tile, 1536));
        assert!(!grouped_raster(LoaderKind::Plain, true, tile, 2560));
        assert!(grouped_raster(LoaderKind::Plain, true, tile, 49152));
    }

    #[test]
    fn an_ungrouped_kernel_has_the_old_tile_map_and_a_grouped_one_has_the_new() {
        let mut key = keys_with(false)[0];
        let ungrouped = source(key);
        assert!(ungrouped.contains("let m0 = wg.y * BM;\n  let n0 = wg.x * BN;\n"));
        assert!(!ungrouped.contains("const GROUP"));
        key.grouped = true;
        let grouped = source(key);
        assert!(grouped.contains("const GROUP: u32 = 8u;"));
        assert!(grouped.contains("let n0 = (within / rows) * BN;"));
        assert!(!grouped.contains("let m0 = wg.y * BM;"));
    }
}
