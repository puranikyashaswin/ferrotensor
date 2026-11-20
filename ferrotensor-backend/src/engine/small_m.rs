//! The small-M matmul: a weight-stationary, split-K kernel for products with
//! few rows against a wide and deep weight (see matmul_small_m.wgsl).

use super::{grid, immediate_params, uniform, ComputeEngine, Epilogue, Kernel};
use crate::kernel_timing::KernelLabel;
use crate::{DType, UnifiedTensor, WgpuContext};
use std::borrow::Cow;
use std::sync::{Arc, PoisonError};

/// Fewest output rows the router sends to the small-M kernel. Below this a
/// row block of at least 16 rows computes mostly padding, and the thin
/// kernel stays. Not tuned yet.
pub(super) const SMALL_M_MIN_ROWS: u32 = 16;

/// Most output rows the router sends to the small-M kernel: two row blocks
/// of at most [`BLOCK_MAX_ROWS`].
pub(super) const SMALL_M_MAX_ROWS: u32 = 2 * BLOCK_MAX_ROWS;

/// Fewest output columns and fewest K the router sends to the small-M
/// kernel, so LoRA products of a small rank keep the thin kernel.
pub(super) const SMALL_M_MIN_WIDTH: u32 = 256;

/// Most rows of the default tile; a pass of more rows runs as two row blocks.
const BLOCK_MAX_ROWS: u32 = 128;

/// Smallest K step the router takes. Smaller steps fit a device with 16 KiB
/// of workgroup memory, but read `a` and `b` in short runs; there the
/// previous kernels stay until a measurement says otherwise.
pub(super) const ROUTE_MIN_BK: u32 = 16;

/// Fewest K slices per split, so a workgroup's loop is long enough to hide
/// its first load and the partial sums stay few.
const MIN_TILES_PER_SPLIT: u32 = 4;

/// The largest dimension of a dispatch grid.
const MAX_GROUPS: u32 = 65_535;

/// Tile shape of a small-M kernel: each of `tx * ty` threads keeps `tm` by
/// `tn` outputs in registers, so a workgroup covers `tm * ty` rows by
/// `tn * tx` columns, and it steps through K `bk` at a time.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SmallMTile {
    /// Output rows per thread.
    pub tm: u32,
    /// Output columns per thread, a multiple of 4.
    pub tn: u32,
    /// Threads along N.
    pub tx: u32,
    /// Threads along M.
    pub ty: u32,
    /// K per staged slice, a multiple of 4.
    pub bk: u32,
}

impl SmallMTile {
    /// Rows one workgroup covers.
    pub fn bm(self) -> u32 {
        self.tm * self.ty
    }

    /// Columns one workgroup covers.
    pub fn bn(self) -> u32 {
        self.tn * self.tx
    }

    /// Threads per workgroup.
    pub fn threads(self) -> u32 {
        self.tx * self.ty
    }

    /// Bytes of workgroup memory the kernel uses, with `b` stored `[N, K]`
    /// (`transposed_b`) or `[K, N]`.
    pub fn workgroup_bytes(self, transposed_b: bool) -> u32 {
        let (a_half, b_half) = self.halves(transposed_b);
        2 * (a_half + b_half) * 16
    }

    /// The vec4s in one of the two staging buffers of `a` and of `b`: rows
    /// of `bk / 4` vec4s plus one of padding, except a `[K, N]` `b`, which
    /// is staged as `bk` rows of `bn / 4` vec4s.
    fn halves(self, transposed_b: bool) -> (u32, u32) {
        let ks = self.bk / 4 + 1;
        let b = if transposed_b {
            self.bn() * ks
        } else {
            self.bk * self.bn() / 4
        };
        (self.bm() * ks, b)
    }

    /// Whether the kernel generator accepts the tile: every size positive,
    /// `tn` and `bk` multiples of 4.
    pub fn is_well_formed(self) -> bool {
        self.tm > 0
            && self.tn > 0
            && self.tx > 0
            && self.ty > 0
            && self.bk > 0
            && self.tn.is_multiple_of(4)
            && self.bk.is_multiple_of(4)
    }

    /// The default tile for a pass of `m` rows on a device with `max_bytes`
    /// of workgroup memory, or `None` when no tile fits.
    ///
    /// One row block covers up to 128 rows, more run as two
    /// blocks. 16 threads along N each keep 4 columns (64 per workgroup);
    /// 8 to 16 threads along M keep up to 8 rows each. K advances 32 at a
    /// time when both layouts fit in workgroup memory, else 16, 8 or 4 (a
    /// device with 16 KiB, such as DX12 on WARP, runs only those).
    pub fn for_rows(m: u32, max_bytes: u32) -> Option<Self> {
        if m == 0 {
            return None;
        }
        let rows = if m <= BLOCK_MAX_ROWS {
            m
        } else {
            m.div_ceil(2)
        };
        let ty = rows.div_ceil(8).clamp(8, 16);
        let tm = rows.div_ceil(ty);
        [32, 16, 8, 4]
            .into_iter()
            .map(|bk| Self {
                tm,
                tn: 4,
                tx: 16,
                ty,
                bk,
            })
            .find(|tile| tile.workgroup_bytes(true).max(tile.workgroup_bytes(false)) <= max_bytes)
    }
}

/// How a small-M matmul ran.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SmallMLaunch {
    /// The tile it ran with.
    pub tile: SmallMTile,
    /// How many slices K was split into (1: no reduction pass).
    pub split: u32,
}

/// A compiled small-M kernel: its tile, `b`'s layout and `b`'s storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct SmallMKey {
    tile: SmallMTile,
    transposed_b: bool,
    half_b: bool,
}

/// Whether the router sends `out[M, N] = a[M, K] @ op(b)` with a `dtype`
/// weight to the small-M kernel: an f32 or f16 weight, M from
/// [`SMALL_M_MIN_ROWS`] to [`SMALL_M_MAX_ROWS`], and N and K both at least
/// [`SMALL_M_MIN_WIDTH`].
pub(super) fn routes(dtype: DType, (m, k, n): (u32, u32, u32)) -> bool {
    matches!(dtype, DType::FP32 | DType::FP16)
        && (SMALL_M_MIN_ROWS..=SMALL_M_MAX_ROWS).contains(&m)
        && n >= SMALL_M_MIN_WIDTH
        && k >= SMALL_M_MIN_WIDTH
}

/// Workgroups the split heuristic aims for. A discrete GPU gets two waves of
/// one workgroup per compute unit on GPUs of up to 80 units (the A10G has
/// 72, the T4 40); an integrated GPU has fewer units; on a CPU adapter a
/// split only adds work.
pub(super) fn split_target(device_type: wgpu::DeviceType) -> u32 {
    match device_type {
        wgpu::DeviceType::DiscreteGpu => 160,
        wgpu::DeviceType::IntegratedGpu => 48,
        _ => 8,
    }
}

/// How many slices to split K = `k` into for a grid of `groups` workgroups:
/// enough to bring the grid to `target` workgroups, but no more than one per
/// [`MIN_TILES_PER_SPLIT`] slices of `bk`.
pub(super) fn split_count(groups: u32, k: u32, bk: u32, target: u32) -> u32 {
    let most = (k.div_ceil(bk) / MIN_TILES_PER_SPLIT).max(1);
    target.div_ceil(groups.max(1)).clamp(1, most)
}

/// `(splits, K per split)` for `requested` splits of K = `k`. K per split is
/// a whole number of `bk` slices, so a split never starts inside a slice; the
/// rounding can leave fewer splits than requested, and none empty.
pub(super) fn split_k(k: u32, bk: u32, requested: u32) -> (u32, u32) {
    let per = k.div_ceil(bk).div_ceil(requested.max(1)).max(1) * bk;
    (k.div_ceil(per).max(1), per)
}

const TEMPLATE: &str = include_str!("../shaders/matmul_small_m.wgsl");

const B4_F32: &str = "\
// Elements i..i+3 of b; those from `valid` on are 0.
fn skm_b4(i: u32, valid: u32) -> vec4<f32> {
    if (valid >= 4u) {
        return vec4<f32>(skm_b[i], skm_b[i + 1u], skm_b[i + 2u], skm_b[i + 3u]);
    }
    var v = vec4<f32>(0.0);
    if (valid > 0u) { v.x = skm_b[i]; }
    if (valid > 1u) { v.y = skm_b[i + 1u]; }
    if (valid > 2u) { v.z = skm_b[i + 2u]; }
    return v;
}";

const B4_F16: &str = "\
// Element i of b: the f16 in the low (even i) or high half of word i / 2.
fn skm_h(i: u32) -> f32 {
    return unpack2x16float(skm_b[i >> 1u])[i & 1u];
}

// Elements i..i+3 of b; those from `valid` on are 0. Four elements from an
// even index are two whole words.
fn skm_b4(i: u32, valid: u32) -> vec4<f32> {
    if (valid >= 4u && (i & 1u) == 0u) {
        let w = i >> 1u;
        return vec4<f32>(unpack2x16float(skm_b[w]), unpack2x16float(skm_b[w + 1u]));
    }
    var v = vec4<f32>(0.0);
    if (valid > 0u) { v.x = skm_h(i); }
    if (valid > 1u) { v.y = skm_h(i + 1u); }
    if (valid > 2u) { v.z = skm_h(i + 2u); }
    if (valid > 3u) { v.w = skm_h(i + 3u); }
    return v;
}";

const COMPONENTS: [char; 4] = ['x', 'y', 'z', 'w'];

/// One block per vec4 a thread stages: unit `u = lidx + t * threads` of
/// `count` with `body(t)` inside, guarded when the round has fewer units
/// than threads.
fn per_unit(count: u32, threads: u32, body: impl Fn(u32) -> String) -> String {
    let mut s = String::new();
    for t in 0..count.div_ceil(threads) {
        let first = t * threads;
        s += &format!("{{\nlet u = lidx + {first}u;\n");
        if first + threads > count {
            s += &format!("if (u < {count}u) {{\n{}}}\n", body(t));
        } else {
            s += &body(t);
        }
        s += "}\n";
    }
    s
}

/// The WGSL of the small-M kernel for `tile`, with `b` stored `[N, K]`
/// (`transposed_b`) or `[K, N]`, as f16 pairs in `u32` words (`half_b`) or
/// as f32. The register tile, staging and stores are unrolled here, since a
/// loop over a local array may not stay in registers.
pub(super) fn source(tile: SmallMTile, transposed_b: bool, half_b: bool) -> String {
    let SmallMTile { tm, tn, tx, ty, bk } = tile;
    let threads = tile.threads();
    let (k4, ks) = (bk / 4, bk / 4 + 1);
    let (a_half, b_half) = tile.halves(transposed_b);
    // Both operands are staged as units of 4 consecutive elements: BM (or
    // BN) rows of BK / 4 units, or BK rows of BN / 4.
    let a_units = tile.bm() * k4;
    let b_units = tile.bn() * k4;

    let mut stage = String::new();
    for t in 0..a_units.div_ceil(threads) {
        stage += &format!("    var ra{t}: vec4<f32>;\n");
    }
    for t in 0..b_units.div_ceil(threads) {
        stage += &format!("    var rb{t}: vec4<f32>;\n");
    }

    let mut load = per_unit(a_units, threads, |t| {
        format!(
            "let kk = kb + (u % SKM_K4) * 4u;\n\
             ra{t} = skm_a4(min(row0 + u / SKM_K4, m - 1u) * k + kk, skm_valid(k, kk));\n"
        )
    });
    let mut store = per_unit(a_units, threads, |t| {
        format!("skm_as[a_nxt + (u / SKM_K4) * SKM_KS + u % SKM_K4] = ra{t};\n")
    });
    if transposed_b {
        load += &per_unit(b_units, threads, |t| {
            format!(
                "let kk = kb + (u % SKM_K4) * 4u;\n\
                 rb{t} = skm_b4(min(col0 + u / SKM_K4, n - 1u) * k + kk, skm_valid(k, kk));\n"
            )
        });
        store += &per_unit(b_units, threads, |t| {
            format!("skm_bs[b_nxt + (u / SKM_K4) * SKM_KS + u % SKM_K4] = rb{t};\n")
        });
    } else {
        load += &per_unit(b_units, threads, |t| {
            format!(
                "let kk = kb + u / SKM_N4;\n\
                 let nn = col0 + (u % SKM_N4) * 4u;\n\
                 var valid = 0u;\n\
                 if (kk < k) {{ valid = skm_valid(n, nn); }}\n\
                 rb{t} = skm_b4(kk * n + nn, valid);\n"
            )
        });
        store += &per_unit(b_units, threads, |t| {
            format!("skm_bs[b_nxt + u] = rb{t};\n")
        });
    }

    // Accumulators: one f32 per output column for a@bT, one vec4 per 4
    // adjacent columns for a@b.
    let col_groups = if transposed_b { tn } else { tn / 4 };
    let mut acc = String::new();
    for i in 0..tm {
        for j in 0..col_groups {
            if transposed_b {
                acc += &format!("    var acc{i}_{j}: f32 = 0.0;\n");
            } else {
                acc += &format!("    var acc{i}_{j} = vec4<f32>(0.0);\n");
            }
        }
    }

    // One step of 4 along K: each row's 4 values of a are one vec4.
    let mut compute = String::new();
    for i in 0..tm {
        let off = i * ty * ks;
        compute += &format!("let a{i} = skm_as[a_cur + a_row + {off}u + k4];\n");
    }
    if transposed_b {
        // Each column's 4 values of b are one vec4 too: a dot product.
        for j in 0..tn {
            let off = j * tx * ks;
            compute += &format!("let b{j} = skm_bs[b_cur + b_col + {off}u + k4];\n");
        }
        for i in 0..tm {
            for j in 0..tn {
                compute += &format!(
                    "acc{i}_{j} = fma(a{i}.w, b{j}.w, fma(a{i}.z, b{j}.z, \
                     fma(a{i}.y, b{j}.y, fma(a{i}.x, b{j}.x, acc{i}_{j}))));\n"
                );
            }
        }
    } else {
        // Each K row of b gives 4 adjacent columns as one vec4: an outer
        // product per row.
        for (e, c) in COMPONENTS.into_iter().enumerate() {
            for j in 0..tn / 4 {
                let off = j * tx;
                compute += &format!(
                    "let b{c}{j} = skm_bs[b_cur + (k4 * 4u + {e}u) * SKM_N4 + tx + {off}u];\n"
                );
                for i in 0..tm {
                    compute +=
                        &format!("acc{i}_{j} = fma(vec4<f32>(a{i}.{c}), b{c}{j}, acc{i}_{j});\n");
                }
            }
        }
    }

    let mut write = String::new();
    for i in 0..tm {
        let r = i * ty;
        write += &format!(
            "    {{\n        let row = row0 + ty + {r}u;\n        if (row < m) {{\n            let o = base + row * n;\n"
        );
        for j in 0..col_groups {
            let c = j * tx;
            if transposed_b {
                write += &format!(
                    "            if (col0 + tx + {c}u < n) {{ skm_store(o + col0 + tx + {c}u, acc{i}_{j}); }}\n"
                );
            } else {
                write +=
                    &format!("            {{\n                let c = col0 + (tx + {c}u) * 4u;\n");
                for (e, comp) in COMPONENTS.into_iter().enumerate() {
                    write += &format!(
                        "                if (c + {e}u < n) {{ skm_store(o + c + {e}u, acc{i}_{j}.{comp}); }}\n"
                    );
                }
                write += "            }\n";
            }
        }
        write += "        }\n    }\n";
    }

    TEMPLATE
        .replace("\r\n", "\n")
        .replace("%BM%", &tile.bm().to_string())
        .replace("%BN%", &tile.bn().to_string())
        .replace("%BK%", &bk.to_string())
        .replace("%TX%", &tx.to_string())
        .replace("%K4%", &k4.to_string())
        .replace("%KS%", &ks.to_string())
        .replace("%N4%", &(tile.bn() / 4).to_string())
        .replace("%A_HALF%", &a_half.to_string())
        .replace("%B_HALF%", &b_half.to_string())
        .replace("%A_LEN%", &(2 * a_half).to_string())
        .replace("%B_LEN%", &(2 * b_half).to_string())
        .replace("%B_ELEM%", if half_b { "u32" } else { "f32" })
        .replace("%THREADS%", &threads.to_string())
        .replace("%B4%", if half_b { B4_F16 } else { B4_F32 })
        .replace("%ACC%", &acc)
        .replace("%STAGE%", &stage)
        .replace("%LOAD%", &load)
        .replace("%STORE%", &store)
        .replace("%COMPUTE%", &compute)
        .replace("%WRITE%", &write)
}

impl ComputeEngine {
    /// Whether this device can run `tile` with `b` stored `[N, K]`
    /// (`transposed_b`) or `[K, N]`: threads and workgroup memory within its
    /// limits.
    pub(super) fn small_m_fits(&self, tile: SmallMTile, transposed_b: bool) -> bool {
        tile.is_well_formed()
            && tile.threads() <= self.small_m_max_threads
            && tile.workgroup_bytes(transposed_b) <= self.small_m_max_bytes
    }

    /// The default small-M tile for `m` rows on this device, if one fits.
    pub(super) fn small_m_tile(&self, m: u32, transposed_b: bool) -> Option<SmallMTile> {
        SmallMTile::for_rows(m, self.small_m_max_bytes)
            .filter(|&tile| self.small_m_fits(tile, transposed_b))
    }

    /// The small-M kernel for `key`, compiled on first use. Each tile is its
    /// own shader module, so tiles that never run cost nothing.
    fn small_m_kernel(&self, ctx: &WgpuContext, key: SmallMKey) -> Arc<Kernel> {
        let mut cache = self.small_m.lock().unwrap_or_else(PoisonError::into_inner);
        Arc::clone(cache.entry(key).or_insert_with(|| {
            let mut src = source(key.tile, key.transposed_b, key.half_b);
            if self.immediates {
                src = immediate_params(&src);
            }
            let module = ctx
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("matmul_small_m"),
                    source: wgpu::ShaderSource::Wgsl(Cow::Owned(src)),
                });
            Arc::new(Kernel::new(&ctx.device, &module, "matmul_small_m"))
        }))
    }

    /// Records `out[M, N] = alpha * a[M, K] @ op(b) + beta * out` (`ep`) with
    /// the small-M kernel and `tile`, with `b` stored `[N, K]`
    /// (`transposed_b`) or `[K, N]`, as f32 or f16. K is split into `split`
    /// slices, or as [`split_count`] picks for this device; the slices'
    /// partial sums are added in a second pass. The caller checks the tile
    /// with [`small_m_fits`](Self::small_m_fits) and that the grid is within
    /// [`MAX_GROUPS`] per dimension.
    pub(super) fn matmul_small_m(
        &self,
        ctx: &WgpuContext,
        transposed_b: bool,
        (m, k, n): (u32, u32, u32),
        [a, b, out]: [&UnifiedTensor; 3],
        (tile, split): (SmallMTile, Option<u32>),
        ep: Epilogue,
    ) -> SmallMLaunch {
        let half_b = matches!(b.dtype, DType::FP16);
        let kernel = self.small_m_kernel(
            ctx,
            SmallMKey {
                tile,
                transposed_b,
                half_b,
            },
        );
        let (rows, cols) = (m.div_ceil(tile.bm()), n.div_ceil(tile.bn()));
        let wanted = split
            .unwrap_or_else(|| split_count(rows * cols, k, tile.bk, self.split_target))
            .min(MAX_GROUPS);
        let (split, k_per_split) = split_k(k, tile.bk, wanted);
        let label = KernelLabel {
            entry: if half_b {
                "matmul_small_m_h"
            } else {
                "matmul_small_m"
            },
            layout: if transposed_b { "a@bT" } else { "a@b" },
            size_class: "m<=256",
        };
        // With one slice the kernel applies the epilogue; with more it writes
        // plain partial sums and the reduction applies it.
        let kernel_ep = if split == 1 { ep } else { Epilogue::STORE };
        let params = uniform(
            ctx,
            bytemuck::cast_slice(&[
                m,
                k,
                n,
                k_per_split,
                kernel_ep.alpha.to_bits(),
                kernel_ep.beta.to_bits(),
                0,
                0,
            ]),
        );
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
                ],
                (rows, cols, 1),
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
                ],
                (rows, cols, split),
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
        SmallMLaunch { tile, split }
    }

    /// Whether a grid of `tile` over `(m, n)` with `split` slices fits the
    /// per-dimension limit.
    pub(super) fn small_m_grid_fits(tile: SmallMTile, m: u32, n: u32, split: Option<u32>) -> bool {
        m.div_ceil(tile.bm()) <= MAX_GROUPS
            && n.div_ceil(tile.bn()) <= MAX_GROUPS
            && split.is_none_or(|s| s <= MAX_GROUPS)
    }
}

#[cfg(test)]
#[allow(clippy::panic)]
mod tests {
    use super::{routes, source, split_count, split_k, split_target, SmallMTile};
    use crate::DType;
    use std::collections::HashSet;

    fn tile(tm: u32, tn: u32, tx: u32, ty: u32, bk: u32) -> SmallMTile {
        SmallMTile { tm, tn, tx, ty, bk }
    }

    #[test]
    fn default_tile_covers_the_pass_in_one_or_two_row_blocks() {
        // NVIDIA's 48 KiB of workgroup memory: K steps of 32 up to 104 rows.
        let nv = 48 * 1024;
        assert_eq!(SmallMTile::for_rows(64, nv), Some(tile(8, 4, 16, 8, 32)));
        assert_eq!(SmallMTile::for_rows(96, nv), Some(tile(8, 4, 16, 12, 32)));
        assert_eq!(SmallMTile::for_rows(128, nv), Some(tile(8, 4, 16, 16, 16)));
        // 160 rows run as two blocks of 80.
        assert_eq!(SmallMTile::for_rows(160, nv), Some(tile(8, 4, 16, 10, 32)));
        assert_eq!(SmallMTile::for_rows(256, nv), Some(tile(8, 4, 16, 16, 16)));
        // Few rows keep 8 threads along M with fewer rows each.
        assert_eq!(SmallMTile::for_rows(32, nv), Some(tile(4, 4, 16, 8, 32)));
        assert_eq!(SmallMTile::for_rows(17, nv), Some(tile(3, 4, 16, 8, 32)));
        // 32 KiB (Apple, DX12): K steps of 16 above 32 rows.
        let small = 32 * 1024;
        assert_eq!(SmallMTile::for_rows(32, small), Some(tile(4, 4, 16, 8, 32)));
        assert_eq!(SmallMTile::for_rows(64, small), Some(tile(8, 4, 16, 8, 16)));
        assert_eq!(
            SmallMTile::for_rows(128, small),
            Some(tile(8, 4, 16, 16, 16))
        );
        // 16 KiB (the WebGPU default, DX12 on WARP): K steps of 8 or 4.
        let least = 16 * 1024;
        assert_eq!(SmallMTile::for_rows(32, least), Some(tile(4, 4, 16, 8, 16)));
        assert_eq!(SmallMTile::for_rows(37, least), Some(tile(5, 4, 16, 8, 8)));
        assert_eq!(SmallMTile::for_rows(64, least), Some(tile(8, 4, 16, 8, 8)));
        assert_eq!(
            SmallMTile::for_rows(128, least),
            Some(tile(8, 4, 16, 16, 4))
        );
        assert_eq!(SmallMTile::for_rows(128, 8 * 1024), None);
        assert_eq!(SmallMTile::for_rows(0, nv), None);
    }

    #[test]
    fn router_takes_few_rows_against_a_wide_deep_weight() {
        // SmolLM2-1.7B projections at a micro-batch of 64 tokens.
        assert!(routes(DType::FP16, (64, 2048, 2048)));
        assert!(routes(DType::FP16, (64, 2048, 8192)));
        assert!(routes(DType::FP32, (160, 8192, 2048)));
        assert!(routes(DType::FP32, (16, 256, 256)));
        assert!(routes(DType::FP32, (256, 2048, 49152)));
        // Too few or too many rows.
        assert!(!routes(DType::FP16, (15, 2048, 2048)));
        assert!(!routes(DType::FP16, (257, 2048, 2048)));
        // LoRA products of rank 8 stay on the thin kernel.
        assert!(!routes(DType::FP32, (64, 2048, 8)));
        assert!(!routes(DType::FP32, (64, 8, 2048)));
        // 4-bit weights keep their own kernels.
        assert!(!routes(DType::Q4, (64, 2048, 2048)));
    }

    #[test]
    fn workgroup_bytes_count_both_buffers() {
        let t = tile(8, 4, 16, 8, 32);
        // a: 64 rows x 9 vec4s; [N, K] b: 64 rows x 9; [K, N] b: 32 rows x 16.
        assert_eq!(t.workgroup_bytes(true), 2 * (576 + 576) * 16);
        assert_eq!(t.workgroup_bytes(false), 2 * (576 + 512) * 16);
        assert_eq!((t.bm(), t.bn(), t.threads()), (64, 64, 128));
        assert!(t.is_well_formed());
        assert!(!tile(8, 6, 16, 8, 32).is_well_formed());
        assert!(!tile(8, 4, 16, 8, 30).is_well_formed());
        assert!(!tile(0, 4, 16, 8, 32).is_well_formed());
    }

    #[test]
    fn split_brings_the_grid_to_its_target() {
        // 64 x 2048 over K 8192 on a discrete GPU: 32 column blocks, 5 splits
        // of 1664 (52 slices of 32), 160 workgroups.
        let target = split_target(wgpu::DeviceType::DiscreteGpu);
        assert_eq!(split_count(32, 8192, 32, target), 5);
        assert_eq!(split_k(8192, 32, 5), (5, 1664));
        // 8192 columns: 128 blocks, 2 splits.
        assert_eq!(split_count(128, 2048, 32, target), 2);
        // The vocabulary projection already has 768 blocks.
        assert_eq!(split_count(768, 2048, 32, target), 1);
        // At most one split per 4 slices: K 256 is 8 slices of 32.
        assert_eq!(split_count(1, 256, 32, target), 2);
        assert_eq!(split_count(1, 100, 32, target), 1);
        // A CPU adapter gets few splits.
        assert_eq!(
            split_count(4, 8192, 32, split_target(wgpu::DeviceType::Cpu)),
            2
        );
    }

    #[test]
    fn split_k_rounds_to_whole_slices_and_leaves_none_empty() {
        // K 259 is 17 slices of 16.
        assert_eq!(split_k(259, 16, 1), (1, 272));
        assert_eq!(split_k(259, 16, 3), (3, 96));
        // 7 requested: 3 slices each, which 6 splits already cover.
        assert_eq!(split_k(259, 16, 7), (6, 48));
        // More splits than slices: one slice each.
        assert_eq!(split_k(40, 32, 9), (2, 32));
        assert_eq!(split_k(40, 32, 0), (1, 64));
    }

    fn validate(src: &str, what: &str) {
        let module = match naga::front::wgsl::parse_str(src) {
            Ok(m) => m,
            Err(e) => panic!("{what}: WGSL parse error:\n{}", e.emit_to_string(src)),
        };
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            // The f16 kernels unpack halves with `unpack2x16float`, which naga
            // 30 gates behind this capability; wgpu grants it from the adapter's
            // `SHADER_F16_IN_F32` downlevel flag.
            naga::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32,
        );
        if let Err(e) = validator.validate(&module) {
            panic!("{what}: WGSL validation error:\n{}", e.emit_to_string(src));
        }
    }

    #[test]
    fn generated_kernels_validate() {
        // The default tiles for every row count on both memory sizes, and the
        // shapes the benchmark sweeps: 8 columns per thread, more threads
        // along M or N, a thread count that leaves the last round of staged
        // units short.
        let mut tiles: HashSet<SmallMTile> = (1..=256)
            .flat_map(|m| [16 * 1024, 32 * 1024, 48 * 1024].map(|b| SmallMTile::for_rows(m, b)))
            .flatten()
            .collect();
        tiles.extend([
            tile(8, 8, 16, 8, 32),
            tile(4, 4, 8, 2, 32),
            tile(4, 4, 16, 16, 16),
            tile(8, 4, 32, 8, 16),
            tile(3, 8, 8, 5, 16),
            tile(1, 4, 1, 1, 4),
        ]);
        for t in tiles {
            for transposed_b in [true, false] {
                for half_b in [true, false] {
                    validate(
                        &source(t, transposed_b, half_b),
                        &format!("{t:?} transposed_b {transposed_b} half_b {half_b}"),
                    );
                }
            }
        }
    }
}
