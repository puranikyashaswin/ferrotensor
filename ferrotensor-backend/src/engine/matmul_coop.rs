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

use super::{grid, immediate_params, uniform, ComputeEngine, Epilogue, Kernel};
use crate::kernel_timing::KernelLabel;
use crate::{DType, ReadbackError, UnifiedTensor, WgpuContext};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

/// f16 elements of padding after each staged row, against bank conflicts.
const PAD: u32 = 8;
/// Threads per subgroup the kernel is written for.
const SUBGROUP: u32 = 32;
/// Fewest output rows routed to the kernel: decode steps and thin products
/// stay on the f32 kernels.
const MIN_ROWS: u32 = 64;
/// Largest workgroup count per grid dimension.
const MAX_GROUPS: u32 = 65_535;
/// Fewest K steps of `bk` per slice when the router splits K, so each
/// slice's loop still overlaps a load with a multiply.
const MIN_STEPS_PER_SPLIT: u32 = 2;

/// A workgroup's output tile `bm x bn`, split into one `wm x wn` block per
/// subgroup, stepping through K `bk` at a time with one staging buffer or
/// two (`double`: the next step's loads overlap this step's multiplies, and
/// each step needs one barrier instead of two).
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CoopTile {
    /// Output rows per workgroup.
    pub bm: u32,
    /// Output columns per workgroup.
    pub bn: u32,
    /// Output rows per subgroup.
    pub wm: u32,
    /// Output columns per subgroup.
    pub wn: u32,
    /// K per staged step, 32 or 64.
    pub bk: u32,
    /// Two staging buffers.
    pub double: bool,
}

impl CoopTile {
    const fn new(bm: u32, bn: u32, (wm, wn): (u32, u32), bk: u32, double: bool) -> Self {
        Self {
            bm,
            bn,
            wm,
            wn,
            bk,
            double,
        }
    }

    /// Every tile the generator is tested with, for tests and the tuning
    /// bench. The first three are the tiles of the first coop kernel.
    pub const ALL: [Self; 14] = [
        Self::new(128, 64, (64, 32), 32, false),
        Self::new(64, 64, (32, 32), 32, false),
        Self::new(32, 32, (16, 16), 32, false),
        Self::new(128, 64, (64, 32), 32, true),
        Self::new(64, 64, (32, 32), 32, true),
        Self::new(128, 128, (64, 64), 32, true),
        Self::new(128, 64, (32, 32), 32, true),
        Self::new(64, 128, (32, 64), 32, true),
        Self::new(64, 64, (32, 32), 64, true),
        Self::new(64, 32, (32, 16), 32, true),
        Self::new(32, 64, (16, 32), 32, true),
        Self::new(64, 64, (16, 32), 32, true),
        Self::new(32, 64, (16, 16), 64, true),
        Self::new(32, 32, (16, 16), 64, true),
    ];

    /// A short name such as `128x64/w64x32/k32/db`.
    pub fn name(self) -> String {
        format!(
            "{}x{}/w{}x{}/k{}{}",
            self.bm,
            self.bn,
            self.wm,
            self.wn,
            self.bk,
            if self.double { "/db" } else { "" }
        )
    }

    fn warps(self) -> u32 {
        (self.bm / self.wm) * (self.bn / self.wn)
    }

    fn threads(self) -> u32 {
        SUBGROUP * self.warps()
    }

    fn stages(self) -> u32 {
        if self.double {
            2
        } else {
            1
        }
    }

    /// f16 elements of one staged `a` tile.
    fn a_len(self) -> u32 {
        self.bm * (self.bk + PAD)
    }

    /// f16 elements of one staged `b` tile: `[n][k]` rows for `b[N, K]`,
    /// `[k][n]` rows for `b[K, N]`.
    fn b_len(self, transposed_b: bool) -> u32 {
        if transposed_b {
            self.bn * (self.bk + PAD)
        } else {
            self.bk * (self.bn + PAD)
        }
    }

    /// Workgroup memory in bytes: the staged `a` and `b` tiles in f16 (two
    /// of each when double-buffered) and one 16x16 f32 staging fragment per
    /// subgroup.
    pub fn workgroup_bytes(self, transposed_b: bool) -> u32 {
        self.stages() * (self.a_len() + self.b_len(transposed_b)) * 2 + self.warps() * 256 * 4
    }

    fn groups(self, m: u32, n: u32) -> (u32, u32) {
        (n.div_ceil(self.bn), m.div_ceil(self.bm))
    }
}

/// A tile and how many slices K is split into; with more than one slice the
/// kernel writes partial sums and a second pass adds them.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CoopConfig {
    /// The tile.
    pub tile: CoopTile,
    /// Slices of K (1: no reduction pass).
    pub split: u32,
}

/// Tiles the router tries for products of more than 256 rows, in order: the
/// first whose grid reaches the target runs. The 64x64 tile with one staging
/// buffer was within 6% of the fastest configuration on 21 of the 28 base
/// weight products at 2048 rows that `matmul-tune` timed on an A10G, and the
/// router's earlier picks (the 128x128 tile first) took 1.3 to 2.8 times the
/// fastest time on 18 of them.
const LARGE_M_TILES: [usize; 2] = [1, 2];
/// Tiles the router tries for products of up to 256 rows; when none reaches
/// the target, the last one runs with K split. These two were the fastest
/// tiles of most of the 256-row products `matmul-tune` timed on an A10G.
const SMALL_M_TILES: [usize; 2] = [12, 11];

/// A product `matmul-tune` timed on an A10G and the fastest configuration it
/// found.
struct Tuned {
    m: u32,
    transposed_b: bool,
    k: u32,
    n: u32,
    /// Index into [`CoopTile::ALL`].
    tile: usize,
    split: u32,
}

impl Tuned {
    const fn new(m: u32, transposed_b: bool, k: u32, n: u32, tile: usize, split: u32) -> Self {
        Self {
            m,
            transposed_b,
            k,
            n,
            tile,
            split,
        }
    }
}

/// The fastest configuration of every base-weight product of LoRA
/// fine-tuning SmolLM2-135M and 360M at 256 and 2048 rows, and of 1.7B at
/// 2048 rows, from one `matmul-tune` run on an NVIDIA A10G. A K split stands
/// only where it was at least 10% faster than the best unsplit configuration,
/// since it costs a second pass and a partial-sum buffer that the bench's
/// isolated calls charge only in part. The 1.7B products at 256 rows are not
/// here: their bench winners (the 128x64 and 64x64 tiles with one staging
/// buffer) took 1.3 to 2.4 times as long in a `--profile` of a 30-step 1.7B
/// run, with micro-batches of 64 to 256 rows next to the other kernels of the
/// step, as the double-buffered 64x64 tile with K split to fill the GPU. The
/// bench times one product alone, repeated with a warm cache, and the step
/// does not. Those products run the router's rule.
const TUNED: &[Tuned] = &[
    // SmolLM2-135M
    Tuned::new(256, true, 576, 576, 12, 1),
    Tuned::new(256, false, 576, 576, 12, 1),
    Tuned::new(256, true, 576, 192, 12, 1),
    Tuned::new(256, false, 192, 576, 12, 1),
    Tuned::new(256, true, 576, 1536, 2, 1),
    Tuned::new(256, false, 1536, 576, 11, 6),
    Tuned::new(256, true, 1536, 576, 11, 6),
    Tuned::new(256, false, 576, 1536, 2, 1),
    Tuned::new(256, true, 576, 49152, 6, 1),
    Tuned::new(256, false, 49152, 576, 11, 6),
    Tuned::new(2048, true, 576, 576, 6, 1),
    Tuned::new(2048, false, 576, 576, 6, 1),
    Tuned::new(2048, true, 576, 192, 2, 1),
    Tuned::new(2048, false, 192, 576, 6, 1),
    Tuned::new(2048, true, 576, 1536, 1, 1),
    Tuned::new(2048, false, 1536, 576, 6, 1),
    Tuned::new(2048, true, 1536, 576, 6, 1),
    Tuned::new(2048, false, 576, 1536, 1, 1),
    Tuned::new(2048, true, 576, 49152, 0, 1),
    Tuned::new(2048, false, 49152, 576, 6, 1),
    // SmolLM2-360M
    Tuned::new(256, true, 960, 960, 12, 1),
    Tuned::new(256, false, 960, 960, 12, 1),
    Tuned::new(256, true, 960, 320, 12, 4),
    Tuned::new(256, false, 320, 960, 12, 1),
    Tuned::new(256, true, 960, 2560, 11, 1),
    Tuned::new(256, false, 2560, 960, 11, 4),
    Tuned::new(256, true, 2560, 960, 11, 4),
    Tuned::new(256, false, 960, 2560, 11, 1),
    Tuned::new(256, true, 960, 49152, 0, 1),
    Tuned::new(256, false, 49152, 960, 1, 8),
    Tuned::new(2048, true, 960, 960, 1, 1),
    Tuned::new(2048, false, 960, 960, 1, 1),
    Tuned::new(2048, true, 960, 320, 11, 1),
    Tuned::new(2048, false, 320, 960, 1, 1),
    Tuned::new(2048, true, 960, 2560, 1, 1),
    Tuned::new(2048, false, 2560, 960, 1, 1),
    Tuned::new(2048, true, 2560, 960, 1, 1),
    Tuned::new(2048, false, 960, 2560, 1, 1),
    Tuned::new(2048, true, 960, 49152, 6, 1),
    Tuned::new(2048, false, 49152, 960, 1, 1),
    // SmolLM2-1.7B
    Tuned::new(2048, true, 2048, 2048, 11, 1),
    Tuned::new(2048, false, 2048, 2048, 11, 1),
    Tuned::new(2048, true, 2048, 8192, 1, 1),
    Tuned::new(2048, false, 8192, 2048, 11, 1),
    Tuned::new(2048, true, 8192, 2048, 11, 1),
    Tuned::new(2048, false, 2048, 8192, 1, 1),
    Tuned::new(2048, true, 2048, 49152, 0, 1),
    Tuned::new(2048, false, 49152, 2048, 11, 1),
];

/// Rows the [`TUNED`] rows stand for: the 256-row ones for up to 256 rows, the
/// 2048-row ones for more, up to 2048. Training batches are padded to their
/// longest sequence, so a step's row count is rarely exactly 256 or 2048, and
/// an exact match would leave most products to the fallback rule. `m` is
/// rounded up to a multiple of 16 first, the size of a fragment row.
fn tuned_rows(m: u32) -> Option<u32> {
    match m.div_ceil(16) * 16 {
        0..=256 => Some(256),
        257..=2048 => Some(2048),
        _ => None,
    }
}

/// The measured configuration for this product, when the router has one.
fn tuned(m: u32, transposed_b: bool, k: u32, n: u32) -> Option<CoopConfig> {
    let rows = tuned_rows(m)?;
    let row = TUNED
        .iter()
        .find(|t| (t.m, t.transposed_b, t.k, t.n) == (rows, transposed_b, k, n))?;
    Some(CoopConfig {
        tile: *CoopTile::ALL.get(row.tile)?,
        split: row.split,
    })
}

/// `(splits, K per split)` for `requested` slices of K = `k` in steps of
/// `bk`: every slice a whole number of steps, none empty.
fn split_k(k: u32, bk: u32, requested: u32) -> (u32, u32) {
    let per = k.div_ceil(bk).div_ceil(requested.max(1)).max(1) * bk;
    (k.div_ceil(per).max(1), per)
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
}

/// The cooperative-matrix kernels of a device that has them.
pub(super) struct Coop {
    kernels: Mutex<HashMap<CoopKey, Arc<Kernel>>>,
    /// One `u32` counter of staging threads that clamped an `a` value (or a
    /// value the attention kernels staged).
    pub(super) overflows: UnifiedTensor,
    /// The attention kernels, compiled with the matmul's.
    pub(super) flash: Option<super::flash_coop::FlashCoop>,
    /// Whether [`EXPERIMENTAL_ENV`](crate::adapters::EXPERIMENTAL_ENV) routes f16-weight matmuls here.
    routed: bool,
    /// Workgroups a grid aims for, as for the small-M kernel.
    target: u32,
    /// Whether [`TUNED`] applies: it was measured on a discrete GPU.
    tuned: bool,
    max_bytes: u32,
    max_threads: u32,
    /// Whether the kernel reads its parameters from immediates.
    immediates: bool,
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

/// Whether a grid of `tile` over `m x n` with `split` slices is within the
/// per-dimension limit.
fn grid_fits(tile: CoopTile, m: u32, n: u32, split: u32) -> bool {
    let (gx, gy) = tile.groups(m, n);
    gx <= MAX_GROUPS && gy <= MAX_GROUPS && split <= MAX_GROUPS
}

/// The configuration the router runs an `m x k x n` product with. On a
/// device the [`TUNED`] rows were measured on (`use_tuned`), a product with a
/// row there runs that row's configuration when its tile `fits`. Any other
/// product gets the first tile of the list for `m` that `fits` and whose grid
/// reaches `target` workgroups, else the last one that fits with K split
/// until the grid reaches the target (at most one slice per
/// [`MIN_STEPS_PER_SPLIT`] steps). The first kernel's tiles stand in on a
/// device that fits none of the list.
fn plan(
    (m, k, n): (u32, u32, u32),
    transposed_b: bool,
    (target, use_tuned): (u32, bool),
    fits: impl Fn(CoopTile) -> bool,
) -> Option<CoopConfig> {
    let measured = tuned(m, transposed_b, k, n)
        .filter(|c| use_tuned && fits(c.tile) && grid_fits(c.tile, m, n, c.split));
    if measured.is_some() {
        return measured;
    }
    let list: &[usize] = if m > 256 {
        &LARGE_M_TILES
    } else {
        &SMALL_M_TILES
    };
    let usable = |tile: &CoopTile| fits(*tile) && grid_fits(*tile, m, n, 1);
    let mut tiles: Vec<CoopTile> = list
        .iter()
        .map(|&i| CoopTile::ALL[i])
        .filter(usable)
        .collect();
    if tiles.is_empty() {
        tiles = CoopTile::ALL[..3].iter().copied().filter(usable).collect();
    }
    for &tile in &tiles {
        let (gx, gy) = tile.groups(m, n);
        if gx * gy >= target {
            return Some(CoopConfig { tile, split: 1 });
        }
    }
    let tile = *tiles.last()?;
    let (gx, gy) = tile.groups(m, n);
    let most = (k.div_ceil(tile.bk) / MIN_STEPS_PER_SPLIT).max(1);
    let wanted = target.div_ceil(gx * gy).clamp(1, most);
    Some(CoopConfig {
        tile,
        split: split_k(k, tile.bk, wanted).0,
    })
}

impl Coop {
    /// The kernels for `ctx`, or `None` when its device cannot run them.
    pub(super) fn new(ctx: &WgpuContext, target: u32) -> Option<Self> {
        if !device_supports(ctx) {
            return None;
        }
        let limits = ctx.device.limits();
        Some(Self {
            kernels: Mutex::new(HashMap::new()),
            overflows: UnifiedTensor::zeros(ctx, &[1]),
            flash: Some(super::flash_coop::FlashCoop::new()),
            routed: crate::adapters::experimental_opt_in(),
            target,
            tuned: ctx.adapter.get_info().device_type == wgpu::DeviceType::DiscreteGpu,
            max_bytes: limits.max_compute_workgroup_storage_size,
            max_threads: limits
                .max_compute_invocations_per_workgroup
                .min(limits.max_compute_workgroup_size_x),
            immediates: ctx.uses_immediates(),
        })
    }

    fn fits(&self, tile: CoopTile, transposed_b: bool) -> bool {
        tile.threads() <= self.max_threads && tile.workgroup_bytes(transposed_b) <= self.max_bytes
    }

    fn kernel(&self, ctx: &WgpuContext, key: CoopKey) -> Arc<Kernel> {
        let mut cache = self.kernels.lock().unwrap_or_else(PoisonError::into_inner);
        Arc::clone(cache.entry(key).or_insert_with(|| {
            let mut src = source(key);
            if self.immediates {
                src = immediate_source(&src);
            }
            let module = ctx.shader_checks().module(&ctx.device, "matmul_coop", src);
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
        self.matmul_coop_run(ctx, transposed_b, dims, [a, b, out], config, ep)
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
        self.matmul_coop_run(ctx, transposed_b, dims, tensors, config, ep)
            .is_some()
    }

    /// Records `config` for the product: one pass that applies `ep`, or with
    /// K split, one pass of partial sums and the reduction that adds them in
    /// slice order and applies `ep`. Returns the configuration that ran, or
    /// `None` (recording nothing) without the kernel.
    fn matmul_coop_run(
        &self,
        ctx: &WgpuContext,
        transposed_b: bool,
        (m, k, n): (u32, u32, u32),
        [a, b, out]: [&UnifiedTensor; 3],
        config: CoopConfig,
        ep: Epilogue,
    ) -> Option<CoopConfig> {
        let coop = self.coop.as_ref()?;
        let tile = config.tile;
        let (split, kps) = split_k(k, tile.bk, config.split);
        let b_row = if transposed_b { k } else { n };
        let key = CoopKey {
            tile,
            transposed_b,
            a_vec4: k.is_multiple_of(4),
            b_vec8: b_row.is_multiple_of(8),
        };
        let kernel = coop.kernel(ctx, key);
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
            self.run_labeled(
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
            );
        } else {
            let parts = UnifiedTensor::empty(ctx, &[split as usize, m as usize, n as usize]);
            self.run_labeled(
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
            );
            let len = m * n;
            let params = uniform(
                ctx,
                bytemuck::cast_slice(&[len, split, ep.alpha.to_bits(), ep.beta.to_bits()]),
            );
            self.run(
                ctx,
                &self.split_reduce,
                &[(0, parts.buffer()), (1, out.buffer()), (2, &params)],
                grid(len.div_ceil(256)),
            );
        }
        Some(CoopConfig { tile, split })
    }
}

/// The 16x16 fragments of one K substep `kk` from the staged tiles `sa` and
/// `sb`: loads of `a` and `b`, then every multiply-add of the subgroup's
/// block.
fn fragment_step(key: CoopKey, kk: u32, (sa, sb): (&str, &str)) -> String {
    let (ti, tj) = (key.tile.wm / 16, key.tile.wn / 16);
    let mut s = String::from("    {\n");
    for i in 0..ti {
        s.push_str(&format!(
            "      let fa{i} = coopLoadT<coop_mat16x16<f16, A>>(&{sa}[(wr + {}u) * AP + {}u], AP);\n",
            i * 16,
            kk * 16
        ));
    }
    for j in 0..tj {
        // b stored [N, K] is B column-major with stride AP; [K, N] is B
        // row-major with stride BP.
        let line = if key.transposed_b {
            format!(
                "      let fb{j} = coopLoad<coop_mat16x16<f16, B>>(&{sb}[(wc + {}u) * AP + {}u], AP);",
                j * 16,
                kk * 16
            )
        } else {
            format!(
                "      let fb{j} = coopLoadT<coop_mat16x16<f16, B>>(&{sb}[{}u * BP + wc + {}u], BP);",
                kk * 16,
                j * 16
            )
        };
        s.push_str(&line);
        s.push('\n');
    }
    for i in 0..ti {
        for j in 0..tj {
            s.push_str(&format!(
                "      acc{i}_{j} = coopMultiplyAdd(fa{i}, fb{j}, acc{i}_{j});\n"
            ));
        }
    }
    s.push_str("    }\n");
    s
}

/// Staging `a[m0.., k0..]` into `sa` as f16, flagging values f16 cannot hold.
fn load_a(key: CoopKey, sa: &str) -> String {
    let nt = key.tile.threads();
    let bk = key.tile.bk;
    if key.a_vec4 {
        let count = key.tile.bm * (bk / 4) / nt;
        format!(
            "    for (var i = 0u; i < {count}u; i++) {{
      let idx = li + i * {nt}u;
      let r = idx / {q}u;
      let q = idx % {q}u;
      let gr = m0 + r;
      let gk = k0 + q * 4u;
      var v = vec4<f32>(0.0);
      if (gr < m && gk < k) {{ v = a[(gr * k + gk) / 4u]; }}
      if (any(!(abs(v) <= vec4<f32>(F16_MAX)))) {{ overflow = true; }}
      let h = vec4<f16>(clamp(v, vec4<f32>(-F16_MAX), vec4<f32>(F16_MAX)));
      let o = r * AP + q * 4u;
      {sa}[o] = h.x;
      {sa}[o + 1u] = h.y;
      {sa}[o + 2u] = h.z;
      {sa}[o + 3u] = h.w;
    }}
",
            q = bk / 4
        )
    } else {
        let count = key.tile.bm * bk / nt;
        format!(
            "    for (var i = 0u; i < {count}u; i++) {{
      let idx = li + i * {nt}u;
      let r = idx / BK;
      let q = idx % BK;
      let gr = m0 + r;
      let gk = k0 + q;
      var v = 0.0;
      if (gr < m && gk < k) {{ v = a[gr * k + gk]; }}
      if (!(abs(v) <= F16_MAX)) {{ overflow = true; }}
      {sa}[r * AP + q] = f16(clamp(v, -F16_MAX, F16_MAX));
    }}
"
        )
    }
}

/// Staging the f16 `b` tile into `sb`: `[k][n]` rows of BP for `b[K, N]`,
/// `[n][k]` rows of AP for `b[N, K]`.
fn load_b(key: CoopKey, sb: &str) -> String {
    let nt = key.tile.threads();
    let (bn, bk) = (key.tile.bn, key.tile.bk);
    if key.b_vec8 {
        // (rows of the tile, 8-half chunks per row), then the global row,
        // column of the chunk, word index and workgroup offset.
        let (rows_chunks, body) = if key.transposed_b {
            (
                (bn, bk / 8),
                "      let gn = n0 + r;\n      let gk = k0 + q * 8u;\n      if (gn < n && gk < k) { w = b[(gn * k + gk) / 8u]; }\n      let o = r * AP + q * 8u;\n",
            )
        } else {
            (
                (bk, bn / 8),
                "      let gk = k0 + r;\n      let gn = n0 + q * 8u;\n      if (gk < k && gn < n) { w = b[(gk * n + gn) / 8u]; }\n      let o = r * BP + q * 8u;\n",
            )
        };
        let (rows, chunks) = rows_chunks;
        let count = rows * chunks / nt;
        format!(
            "    for (var i = 0u; i < {count}u; i++) {{
      let idx = li + i * {nt}u;
      let r = idx / {chunks}u;
      let q = idx % {chunks}u;
      var w = vec4<u32>(0u);
{body}      let h0 = vec2<f16>(unpack2x16float(w.x));
      let h1 = vec2<f16>(unpack2x16float(w.y));
      let h2 = vec2<f16>(unpack2x16float(w.z));
      let h3 = vec2<f16>(unpack2x16float(w.w));
      {sb}[o] = h0.x;
      {sb}[o + 1u] = h0.y;
      {sb}[o + 2u] = h1.x;
      {sb}[o + 3u] = h1.y;
      {sb}[o + 4u] = h2.x;
      {sb}[o + 5u] = h2.y;
      {sb}[o + 6u] = h3.x;
      {sb}[o + 7u] = h3.y;
    }}
"
        )
    } else {
        let count = bk * bn / nt;
        let body = if key.transposed_b {
            "      let r = idx / BK;\n      let q = idx % BK;\n      let gn = n0 + r;\n      let gk = k0 + q;\n      let e = gn * k + gk;\n      let o = r * AP + q;\n"
        } else {
            "      let r = idx / BN;\n      let q = idx % BN;\n      let gk = k0 + r;\n      let gn = n0 + q;\n      let e = gk * n + gn;\n      let o = r * BP + q;\n"
        };
        format!(
            "    for (var i = 0u; i < {count}u; i++) {{
      let idx = li + i * {nt}u;
{body}      var v = 0.0;
      if (gk < k && gn < n) {{ v = unpack2x16float(b[e >> 1u])[e & 1u]; }}
      {sb}[o] = f16(v);
    }}
"
        )
    }
}

/// The loads of the K step that starts at `k0` into the staging pair
/// `(sa, sb)`.
fn load_step(key: CoopKey, k0: &str, (sa, sb): (&str, &str)) -> String {
    format!(
        "    {{\n    let k0 = {k0};\n{}{}    }}\n",
        load_a(key, sa),
        load_b(key, sb)
    )
}

/// The multiplies of one K step from the staging pair `bufs`.
fn compute_step(key: CoopKey, bufs: (&str, &str)) -> String {
    (0..key.tile.bk / 16)
        .map(|kk| fragment_step(key, kk, bufs))
        .collect()
}

/// The K loop over this slice's `steps` steps. With one staging pair each
/// step loads, waits, multiplies and waits. With two, the loads of step
/// `t + 1` go to the other pair before step `t`'s multiplies, and one
/// barrier per step separates a pair's reads from its next writes.
fn main_loop(key: CoopKey) -> String {
    let p0 = ("sa0", "sb0");
    if !key.tile.double {
        return format!(
            "  for (var t = 0u; t < steps; t++) {{\n{}    workgroupBarrier();\n{}    workgroupBarrier();\n  }}\n",
            load_step(key, "kb + t * BK", p0),
            compute_step(key, p0)
        );
    }
    let p1 = ("sa1", "sb1");
    // naga takes a function-local variable such as the loop counter as
    // non-uniform, and cooperative ops and barriers must sit in uniform
    // control flow: so the loop runs whole pairs of steps, and only `let`s
    // of uniform values guard the multiplies. Step 2p is in buffer 0, 2p + 1
    // in buffer 1; an odd last step is in buffer 0.
    format!(
        "  let pairs = steps / 2u;
  let odd = steps % 2u;
  if (steps > 0u) {{\n{first}  }}
  workgroupBarrier();
  for (var p = 0u; p < pairs; p++) {{
{load1}{mul0}    workgroupBarrier();
    if (2u * p + 2u < steps) {{\n{load0}    }}
{mul1}    workgroupBarrier();
  }}
  if (odd == 1u) {{
{mul0}    workgroupBarrier();
  }}
",
        first = load_step(key, "kb", p0),
        load1 = load_step(key, "kb + (2u * p + 1u) * BK", p1),
        mul0 = compute_step(key, p0),
        load0 = load_step(key, "kb + (2u * p + 2u) * BK", p0),
        mul1 = compute_step(key, p1),
    )
}

/// The epilogue of fragment `(i, j)`: through the subgroup's staging
/// fragment to `c` (this slice's partial sums when K is split), guarded at
/// the edges, with alpha and beta.
fn store_fragment(i: u32, j: u32) -> String {
    format!(
        "  coopStoreT(acc{i}_{j}, &cs[warp * 256u], 16u);
  workgroupBarrier();
  for (var e = 0u; e < 8u; e++) {{
    let idx = lane + e * 32u;
    let r = m0 + wr + {ri}u + idx / 16u;
    let col = n0 + wc + {cj}u + idx % 16u;
    if (r < m && col < n) {{
      let o = base + r * n + col;
      var v = alpha * cs[warp * 256u + idx];
      if (beta != 0.0) {{ v += beta * c[o]; }}
      c[o] = v;
    }}
  }}
  workgroupBarrier();
",
        ri = i * 16,
        cj = j * 16
    )
}

/// The WGSL of the kernel for `key`, entry point `matmul_coop`. Params are
/// `[M, K, N, K per slice]` and `[alpha bits, beta bits, 0, 0]`; workgroup z
/// is the slice of K, whose sums go to `c[z * M * N ..]`.
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
            store.push_str(&store_fragment(i, j));
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
    format!(
        "enable f16;
enable wgpu_cooperative_matrix;

@group(0) @binding(0) var<storage, read> a: array<{a_ty}>;
@group(0) @binding(1) var<storage, read> b: array<{b_ty}>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> params: array<vec4<u32>, 2>;
@group(0) @binding(4) var<storage, read_write> overflows: array<atomic<u32>>;

const BM: u32 = {bm}u;
const BN: u32 = {bn}u;
const BK: u32 = {bk}u;
const AP: u32 = {ap}u;
const BP: u32 = {bp}u;
const F16_MAX: f32 = 65504.0;

{arrays}var<workgroup> cs: array<f32, {cs_len}>;

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
  let m0 = wg.y * BM;
  let n0 = wg.x * BN;
  // This workgroup's slice of K, [kb, ke), and where its sums go. Slices
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
    c[base + m0 * n + n0] = bitcast<f32>(0x7fc00000u);
  }}
}}
",
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
    use super::{immediate_source, plan, source, split_k, CoopKey, CoopTile, PAD, TUNED};

    fn keys() -> Vec<CoopKey> {
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

    #[test]
    fn tiles_split_evenly_over_their_threads() {
        for t in CoopTile::ALL {
            let nt = t.threads();
            let bk = t.bk;
            assert!(bk == 32 || bk == 64, "{t:?}");
            assert_eq!(t.bm % t.wm, 0, "{t:?}");
            assert_eq!(t.bn % t.wn, 0, "{t:?}");
            assert_eq!(t.wm % 16, 0, "{t:?}");
            assert_eq!(t.wn % 16, 0, "{t:?}");
            // Every staging loop gives each thread the same nonzero count.
            assert_eq!(t.bm * (bk / 4) % nt, 0, "{t:?} a vec4");
            assert_eq!(t.bm * bk % nt, 0, "{t:?} a");
            assert_eq!(bk * t.bn / 8 % nt, 0, "{t:?} b vec8");
            assert_eq!(bk * t.bn % nt, 0, "{t:?} b");
            assert!(t.bm * (bk / 4) >= nt, "{t:?} a vec4 count");
            assert!(bk * t.bn / 8 >= nt, "{t:?} b vec8 count");
            // NVIDIA's 48 KiB of workgroup memory holds every tile.
            assert!(t.workgroup_bytes(true) <= 48 * 1024, "{t:?}");
            assert!(t.workgroup_bytes(false) <= 48 * 1024, "{t:?}");
            // Each workgroup array (sa, sb either way, cs) is a multiple of
            // 16 bytes, so the packed arrays all start 16-byte aligned.
            assert_eq!(t.bm * (bk + PAD) * 2 % 16, 0, "{t:?} sa");
            assert_eq!(t.bn * (bk + PAD) * 2 % 16, 0, "{t:?} sb transposed");
            assert_eq!(bk * (t.bn + PAD) * 2 % 16, 0, "{t:?} sb");
            assert_eq!(t.warps() * 256 * 4 % 16, 0, "{t:?} cs");
            // Row strides in bytes keep every fragment 16-byte aligned.
            assert_eq!((bk + PAD) * 2 % 16, 0);
            assert_eq!((t.bn + PAD) * 2 % 16, 0, "{t:?}");
        }
        assert_eq!(CoopTile::ALL[0].threads(), 128);
        assert_eq!(CoopTile::ALL[6].threads(), 256);
        assert_eq!(CoopTile::ALL[0].name(), "128x64/w64x32/k32");
        assert_eq!(CoopTile::ALL[3].name(), "128x64/w64x32/k32/db");
    }

    #[test]
    fn split_k_keeps_whole_steps_and_no_empty_slice() {
        assert_eq!(split_k(576, 32, 1), (1, 576));
        assert_eq!(split_k(576, 32, 4), (4, 160));
        assert_eq!(split_k(576, 32, 9), (9, 64));
        // 18 steps in 7 slices: 3 steps each fills 6 slices.
        assert_eq!(split_k(576, 32, 7), (6, 96));
        assert_eq!(split_k(50, 32, 4), (2, 32));
        for k in [16, 33, 576, 1536, 8192] {
            for s in 1..20 {
                let (split, per) = split_k(k, 32, s);
                assert!(split <= s && per.is_multiple_of(32), "{k} {s}");
                assert!((split - 1) * per < k && split * per >= k, "{k} {s}");
            }
        }
    }

    #[test]
    fn plan_reaches_the_target_or_splits_k() {
        let all = |_: CoopTile| true;
        let untuned = (160, false);
        // More than 256 rows: the first listed tile whose grid reaches the target.
        let got = plan((2048, 2048, 8192), false, untuned, all).map(|c| (c.tile, c.split));
        assert_eq!(got, Some((CoopTile::ALL[1], 1)));
        // 512 x 576 x 576 has 8 x 9 = 72 tiles of 64x64: the next tile has 288.
        let got = plan((512, 576, 576), false, untuned, all).map(|c| (c.tile, c.split));
        assert_eq!(got, Some((CoopTile::ALL[2], 1)));
        // 256 rows: the last listed tile with K split until the grid reaches
        // the target, at least two steps per slice. Its 9 x 4 tiles want 5
        // slices of K = 576.
        let got = plan((256, 576, 576), false, untuned, all).map(|c| (c.tile, c.split));
        assert_eq!(got, Some((CoopTile::ALL[11], 5)));
        let got = plan((256, 64, 16), false, untuned, all).map(|c| (c.tile, c.split));
        assert_eq!(got, Some((CoopTile::ALL[11], 1)));
        // A device that fits none of the listed tiles runs the first kernel's.
        let single = |t: CoopTile| !t.double;
        let got = plan((256, 576, 576), false, untuned, single).map(|c| (c.tile, c.split));
        assert_eq!(got, Some((CoopTile::ALL[2], 2)));
        assert!(plan((256, 576, 576), false, untuned, |_: CoopTile| false).is_none());
    }

    /// The fastest configuration `matmul-tune` found on an NVIDIA A10G for
    /// each base-weight product of SmolLM2 training: `(m, a@bT, k, n,
    /// configuration)`, copied from that run's `matmul_tune.md`.
    const BENCH: &[(u32, bool, u32, u32, &str)] = &[
        // SmolLM2-135M
        (256, true, 576, 576, "32x64/w16x16/k64/db s1"),
        (256, false, 576, 576, "32x64/w16x16/k64/db s1"),
        (256, true, 576, 192, "32x64/w16x16/k64/db s5"),
        (256, false, 192, 576, "32x64/w16x16/k64/db s1"),
        (256, true, 576, 1536, "32x32/w16x16/k32 s1"),
        (256, false, 1536, 576, "64x64/w16x32/k32/db s6"),
        (256, true, 1536, 576, "64x64/w16x32/k32/db s6"),
        (256, false, 576, 1536, "32x32/w16x16/k32 s1"),
        (256, true, 576, 49152, "128x64/w32x32/k32/db s1"),
        (256, false, 49152, 576, "64x64/w16x32/k32/db s6"),
        (2048, true, 576, 576, "128x64/w32x32/k32/db s1"),
        (2048, false, 576, 576, "128x64/w32x32/k32/db s1"),
        (2048, true, 576, 192, "32x32/w16x16/k32 s1"),
        (2048, false, 192, 576, "128x64/w32x32/k32/db s1"),
        (2048, true, 576, 1536, "64x64/w32x32/k32 s1"),
        (2048, false, 1536, 576, "128x64/w32x32/k32/db s1"),
        (2048, true, 1536, 576, "128x64/w32x32/k32/db s1"),
        (2048, false, 576, 1536, "64x64/w32x32/k32 s1"),
        (2048, true, 576, 49152, "128x64/w64x32/k32 s1"),
        (2048, false, 49152, 576, "128x64/w64x32/k32 s2"),
        // SmolLM2-360M
        (256, true, 960, 960, "32x64/w16x16/k64/db s1"),
        (256, false, 960, 960, "32x64/w16x16/k64/db s1"),
        (256, true, 960, 320, "32x64/w16x16/k64/db s4"),
        (256, false, 320, 960, "32x64/w16x16/k64/db s1"),
        (256, true, 960, 2560, "64x64/w16x32/k32/db s1"),
        (256, false, 2560, 960, "64x64/w16x32/k32/db s4"),
        (256, true, 2560, 960, "64x64/w16x32/k32/db s4"),
        (256, false, 960, 2560, "64x64/w16x32/k32/db s1"),
        (256, true, 960, 49152, "128x64/w64x32/k32 s1"),
        (256, false, 49152, 960, "64x64/w32x32/k32 s8"),
        (2048, true, 960, 960, "64x64/w32x32/k32 s1"),
        (2048, false, 960, 960, "64x64/w32x32/k32 s1"),
        (2048, true, 960, 320, "64x64/w16x32/k32/db s1"),
        (2048, false, 320, 960, "64x64/w32x32/k32 s1"),
        (2048, true, 960, 2560, "64x64/w32x32/k32 s1"),
        (2048, false, 2560, 960, "64x64/w32x32/k32 s1"),
        (2048, true, 2560, 960, "64x64/w32x32/k32 s1"),
        (2048, false, 960, 2560, "64x64/w32x32/k32 s1"),
        (2048, true, 960, 49152, "128x64/w32x32/k32/db s1"),
        (2048, false, 49152, 960, "64x64/w32x32/k32 s1"),
        // SmolLM2-1.7B
        (2048, true, 2048, 2048, "64x64/w16x32/k32/db s1"),
        (2048, false, 2048, 2048, "64x64/w16x32/k32/db s1"),
        (2048, true, 2048, 8192, "64x64/w32x32/k32 s1"),
        (2048, false, 8192, 2048, "64x64/w16x32/k32/db s1"),
        (2048, true, 8192, 2048, "64x64/w16x32/k32/db s1"),
        (2048, false, 2048, 8192, "64x64/w32x32/k32 s1"),
        (2048, true, 2048, 49152, "128x64/w64x32/k32 s1"),
        (2048, false, 49152, 2048, "64x64/w16x32/k32/db s1"),
    ];

    /// The products whose selected configuration differs from the bench's:
    /// the bench winner split K and the best unsplit one was within 10% of it.
    const UNSPLIT: &[(u32, bool, u32, u32, &str)] = &[
        (256, true, 576, 192, "32x64/w16x16/k64/db s1"),
        (2048, false, 49152, 576, "128x64/w32x32/k32/db s1"),
    ];

    #[test]
    fn plan_selects_the_bench_winner_for_the_training_shapes() {
        let all = |_: CoopTile| true;
        assert_eq!(BENCH.len(), TUNED.len());
        for &(m, transposed_b, k, n, winner) in BENCH {
            let unsplit = UNSPLIT
                .iter()
                .find(|u| (u.0, u.1, u.2, u.3) == (m, transposed_b, k, n));
            let want = unsplit.map_or(winner, |u| u.4);
            let got = plan((m, k, n), transposed_b, (160, true), all).expect("a tile fits");
            let name = format!("{} s{}", got.tile.name(), got.split);
            assert_eq!(name, want, "m {m} k {k} n {n} transposed {transposed_b}");
        }
        assert_eq!(UNSPLIT.len(), 2);
        for u in UNSPLIT {
            assert!(u.4.ends_with(" s1"));
        }
    }

    #[test]
    fn splits_stay_where_the_bench_needed_them() {
        let all = |_: CoopTile| true;
        // At 2048 rows the grid fills the GPU on its own: no K split, so no
        // reduction pass, for any model.
        for &(m, transposed_b, k, n, _) in BENCH {
            if m == 2048 {
                let got = plan((m, k, n), transposed_b, (160, true), all).expect("a tile fits");
                assert_eq!(got.split, 1);
            }
        }
        // A tile that does not fit the device is not selected from the table.
        let none_but_small = |t: CoopTile| t.bm * t.bn <= 32 * 64;
        let got = plan((2048, 2048, 8192), true, (160, true), none_but_small);
        assert!(got.is_some_and(|c| c.tile.bm * c.tile.bn <= 32 * 64));
        // A padded batch has fewer rows than 256 or 2048: it runs the row
        // count's bucket, which is the 256-row winner of 576 x 576 for up to
        // 256 rows and the 2048-row winner above that, up to 2048.
        for (m, want) in [
            (64, 12),
            (106, 12),
            (130, 12),
            (256, 12),
            (257, 6),
            (1500, 6),
            (2048, 6),
        ] {
            let got = plan((m, 576, 576), true, (160, true), all).expect("a tile fits");
            assert_eq!((got.tile, got.split), (CoopTile::ALL[want], 1), "m {m}");
        }
        // 1.7B at up to 256 rows has no row, since the bench winners lost in
        // a training step: the rule's tile runs, split until the grid fills
        // the GPU.
        for (m, a_bt, k, n, want) in [
            (128, false, 2048, 8192, (12, 1)),
            (128, true, 2048, 2048, (11, 3)),
            (64, true, 8192, 2048, (11, 5)),
            (256, false, 2048, 8192, (12, 1)),
        ] {
            let got = plan((m, k, n), a_bt, (160, true), all).expect("a tile fits");
            assert_eq!(
                (got.tile, got.split),
                (CoopTile::ALL[want.0], want.1),
                "m {m} k {k} n {n}"
            );
        }
        // More rows than the table has run the router's own rule.
        let got = plan((4096, 576, 576), true, (160, true), all);
        assert_eq!(got.map(|c| c.tile), Some(CoopTile::ALL[1]));
        // Without the table (a device it was not measured on) the router's
        // own rule applies.
        let got = plan((2048, 2048, 8192), true, (160, false), all);
        assert_eq!(got.map(|c| c.tile), Some(CoopTile::ALL[1]));
    }

    #[test]
    fn padded_row_counts_run_the_256_row_winner() {
        let all = |_: CoopTile| true;
        // 106 and 200 rows are what a padded SmolLM2 micro-batch can have;
        // each runs what the 256-row product runs, split included.
        for &(m, transposed_b, k, n, _) in BENCH.iter().filter(|r| r.0 == 256) {
            let want = plan((256, k, n), transposed_b, (160, true), all).expect("a tile fits");
            for rows in [106, 200, 256] {
                let got = plan((rows, k, n), transposed_b, (160, true), all).expect("a tile fits");
                assert_eq!(
                    (got.tile, got.split),
                    (want.tile, want.split),
                    "m {m} rows {rows} k {k} n {n}"
                );
            }
        }
        // 135M down projection at 200 rows: tile 11 with K split into 6.
        let got = plan((200, 1536, 576), false, (160, true), all).expect("a tile fits");
        assert_eq!((got.tile, got.split), (CoopTile::ALL[11], 6));
        // The row count is rounded up to a multiple of 16 for the lookup, so
        // 2049 rows (2064) are past the table and 2048 are not.
        let past = plan((2049, 576, 576), true, (160, true), all).expect("a tile fits");
        assert_eq!(past.tile, CoopTile::ALL[1]);
    }
}
