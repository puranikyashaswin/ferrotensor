//! Which tile, and how many slices of K, the router runs a product with.

use std::collections::HashMap;
use std::sync::OnceLock;

use super::epilogue::EpilogueKind;
use super::loader::LoaderKind;
use super::{CoopConfig, CoopTile};

/// Largest workgroup count per grid dimension.
const MAX_GROUPS: u32 = 65_535;
/// Fewest K steps of `bk` per slice when the router splits K, so each
/// slice's loop still overlaps a load with a multiply.
const MIN_STEPS_PER_SPLIT: u32 = 2;

/// Tiles the router tries for products of more than 256 rows, in order: the
/// first whose grid reaches the target runs. The 64x64 tile with one staging
/// buffer was within 6% of the fastest configuration on 21 of the 28 base
/// weight products at 2048 rows that `matmul-tune` timed on an A10G, and the
/// router's earlier picks (the 128x128 tile first) took 1.3 to 2.8 times the
/// fastest time on 18 of them.
const LARGE_M_TILES: [CoopTile; 2] = [CoopTile::T64X64_W32X32_K32, CoopTile::T32X32_W16X16_K32];
/// Tiles the router tries for products of up to 256 rows; when none reaches
/// the target, the last one runs with K split. These two were the fastest
/// tiles of most of the 256-row products `matmul-tune` timed on an A10G.
const SMALL_M_TILES: [CoopTile; 2] = [
    CoopTile::T32X64_W16X16_K64_DB,
    CoopTile::T64X64_W16X32_K32_DB,
];

/// A product `matmul-tune` timed on an A10G and the fastest configuration it
/// found.
struct Tuned {
    m: u32,
    transposed_b: bool,
    k: u32,
    n: u32,
    /// The tile, by its parameters: adding a tile to [`CoopTile::ALL`] cannot
    /// shift a row.
    tile: CoopTile,
    split: u32,
}

impl Tuned {
    const fn new(m: u32, transposed_b: bool, k: u32, n: u32, tile: CoopTile, split: u32) -> Self {
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
/// fine-tuning SmolLM2-135M at 2048 rows, and the narrow 360M products there,
/// from one `matmul-tune` run on an NVIDIA A10G. A K split stands only where
/// it was at least 10% faster than the best unsplit configuration, since it
/// costs a second pass and a partial-sum buffer that the bench's isolated
/// calls charge only in part. The 64, 128, 192 and 256-row rows of all three
/// models (plain and fused products, and the head) are the configuration with
/// the lowest geometric mean time over the row counts of their bucket (32 and
/// 64, 96 and 128, 192, 256), from one `matmul_bench --tiles` run on an NVIDIA
/// A10G that timed every tile of `CoopTile::ALL` and `CoopTile::PIPELINED`
/// with K split 1 to 8 and to fill the GPU; four narrow 256-row products of
/// the older `matmul-tune` run, absent from that sweep, stay. The 135M
/// 1024-row rows are the fastest configuration of each product of a 1024-row
/// step (fused QKV and gate/up with their LoRA columns, the other products
/// plain, and the head), from one `matmul_bench --tiles --rows 1024` run on an
/// NVIDIA A10G, which times every tile of `CoopTile::ALL` and
/// `CoopTile::PIPELINED`. The rows of 360M and 1.7B from 257 rows (fused QKV
/// and the other products, and the head, and for 360M the 2048-row ones) and
/// the 135M rows at 384 and 512 are the configuration with the lowest
/// geometric mean time over the row counts the bucket was sampled at (see
/// `BENCH_MID` in the tests), from one `matmul_bench --tiles` run on an A10G
/// with the same tiles; the 1.7B 2048-row rows are those of 1536, 1792 and
/// 2048 rows. They replace rows tuned at exactly 1024, 1152 and 1280 rows,
/// whose picks took 1.07 to 1.36 times as long in sum as the best tile at
/// 320 to 768 rows and 1.16 to 1.18 times at 1352 and 1408 rows.
const TUNED: &[Tuned] = &[
    // SmolLM2-135M
    Tuned::new(256, true, 576, 192, CoopTile::T32X64_W16X16_K64_DB, 1),
    Tuned::new(256, false, 192, 576, CoopTile::T32X64_W16X16_K64_DB, 1),
    Tuned::new(2048, true, 576, 576, CoopTile::T128X64_W32X32_K32_DB, 1),
    Tuned::new(2048, false, 576, 576, CoopTile::T128X64_W32X32_K32_DB, 1),
    Tuned::new(2048, true, 576, 192, CoopTile::T32X32_W16X16_K32, 1),
    Tuned::new(2048, false, 192, 576, CoopTile::T128X64_W32X32_K32_DB, 1),
    Tuned::new(2048, true, 576, 1536, CoopTile::T64X64_W32X32_K32, 1),
    Tuned::new(2048, false, 1536, 576, CoopTile::T128X64_W32X32_K32_DB, 1),
    Tuned::new(2048, true, 1536, 576, CoopTile::T128X64_W32X32_K32_DB, 1),
    Tuned::new(2048, false, 576, 1536, CoopTile::T64X64_W32X32_K32, 1),
    Tuned::new(2048, true, 576, 49152, CoopTile::T128X64_W64X32_K32, 1),
    Tuned::new(2048, false, 49152, 576, CoopTile::T128X64_W32X32_K32_DB, 1),
    // SmolLM2-360M
    Tuned::new(256, true, 960, 320, CoopTile::T32X64_W16X16_K64_DB, 4),
    Tuned::new(256, false, 320, 960, CoopTile::T32X64_W16X16_K64_DB, 1),
    Tuned::new(2048, true, 960, 320, CoopTile::T64X64_W16X32_K32_DB, 1),
    Tuned::new(2048, false, 320, 960, CoopTile::T64X64_W32X32_K32, 1),
    // SmolLM2-135M at 64, 128, 192 and 256 rows
    Tuned::new(64, true, 576, 576, CoopTile::T32X32_W16X16_K64_DB, 5),
    Tuned::new(64, true, 576, 1536, CoopTile::T32X64_W16X32_K32_DB, 1),
    Tuned::new(64, true, 576, 49152, CoopTile::T64X64_W32X32_K32_PF, 1),
    Tuned::new(64, true, 984, 576, CoopTile::T32X32_W16X16_K64_DB, 4),
    Tuned::new(64, true, 1536, 576, CoopTile::T32X32_W16X16_K64_DB, 4),
    Tuned::new(64, true, 3088, 576, CoopTile::T32X64_W16X16_K64_DB, 7),
    Tuned::new(64, false, 576, 576, CoopTile::T32X32_W16X16_K64_DB, 5),
    Tuned::new(64, false, 576, 984, CoopTile::T32X32_W16X16_K64_DB, 1),
    Tuned::new(64, false, 576, 1536, CoopTile::T32X32_W16X16_K32, 1),
    Tuned::new(64, false, 576, 3088, CoopTile::T32X32_W16X16_K32, 1),
    Tuned::new(64, false, 1536, 576, CoopTile::T32X32_W16X16_K32, 4),
    Tuned::new(64, false, 49152, 576, CoopTile::T32X64_W16X16_K64_DB, 8),
    Tuned::new(128, true, 576, 576, CoopTile::T32X64_W16X16_K64_DB, 1),
    Tuned::new(128, true, 576, 1536, CoopTile::T32X32_W16X16_K64_DB, 1),
    Tuned::new(128, true, 576, 49152, CoopTile::T128X64_W32X32_K32_PF, 1),
    Tuned::new(128, true, 984, 576, CoopTile::T32X32_W16X16_K32, 8),
    Tuned::new(128, true, 1536, 576, CoopTile::T64X64_W16X32_K32_DB, 8),
    Tuned::new(128, true, 3088, 576, CoopTile::T32X32_W16X16_K32, 8),
    Tuned::new(128, false, 576, 576, CoopTile::T32X32_W16X16_K64_DB, 1),
    Tuned::new(128, false, 576, 984, CoopTile::T32X64_W16X16_K64_DB, 1),
    Tuned::new(128, false, 576, 1536, CoopTile::T32X32_W16X16_K64_DB, 1),
    Tuned::new(128, false, 576, 3088, CoopTile::T64X64_W32X32_K32_PF, 1),
    Tuned::new(128, false, 1536, 576, CoopTile::T64X64_W16X32_K32_DB, 6),
    Tuned::new(128, false, 49152, 576, CoopTile::T128X128_W64X64_K32_PF, 32),
    Tuned::new(192, true, 576, 576, CoopTile::T32X64_W16X16_K64_DB, 1),
    Tuned::new(192, true, 576, 1536, CoopTile::T128X64_W32X32_K32_PF, 1),
    Tuned::new(192, true, 576, 49152, CoopTile::T128X128_W64X64_K32_PF, 1),
    Tuned::new(192, true, 984, 576, CoopTile::T32X64_W16X32_K32_DB, 4),
    Tuned::new(192, true, 1536, 576, CoopTile::T32X32_W16X16_K64_DB, 1),
    Tuned::new(192, true, 3088, 576, CoopTile::T64X64_W16X32_K32_DB, 6),
    Tuned::new(192, false, 576, 576, CoopTile::T32X32_W16X16_K64_DB, 1),
    Tuned::new(192, false, 576, 984, CoopTile::T64X64_W32X32_K32_DB, 3),
    Tuned::new(192, false, 576, 1536, CoopTile::T32X32_W16X16_K64_DB, 1),
    Tuned::new(192, false, 576, 3088, CoopTile::T64X64_W16X32_K32_DB, 1),
    Tuned::new(192, false, 1536, 576, CoopTile::T64X64_W32X32_K32_DB, 6),
    Tuned::new(192, false, 49152, 576, CoopTile::T64X64_W32X32_K32_PF, 8),
    Tuned::new(256, true, 576, 576, CoopTile::T32X64_W16X32_K32_DB, 1),
    Tuned::new(256, true, 576, 1536, CoopTile::T64X64_W32X32_K32_PF, 1),
    Tuned::new(256, true, 576, 49152, CoopTile::T128X128_W64X64_K32_PF, 1),
    Tuned::new(256, true, 984, 576, CoopTile::T32X64_W16X16_K64_DB, 1),
    Tuned::new(256, true, 1536, 576, CoopTile::T32X64_W16X16_K64_DB, 1),
    Tuned::new(256, true, 3088, 576, CoopTile::T64X64_W16X32_K32_DB, 5),
    Tuned::new(256, false, 576, 576, CoopTile::T64X32_W32X16_K32_DB, 1),
    Tuned::new(256, false, 576, 984, CoopTile::T64X64_W32X32_K32, 1),
    Tuned::new(256, false, 576, 1536, CoopTile::T64X64_W32X32_K32_PF, 1),
    Tuned::new(256, false, 576, 3088, CoopTile::T64X64_W32X32_K32_PF, 1),
    Tuned::new(256, false, 1536, 576, CoopTile::T32X64_W16X32_K32_DB, 3),
    Tuned::new(256, false, 49152, 576, CoopTile::T128X128_W64X64_K32_PF, 16),
    // SmolLM2-360M at 64, 128, 192 and 256 rows
    Tuned::new(64, true, 960, 960, CoopTile::T32X32_W16X16_K64_DB, 1),
    Tuned::new(64, true, 960, 2560, CoopTile::T32X32_W16X16_K64_DB, 1),
    Tuned::new(64, true, 960, 49152, CoopTile::T64X64_W32X32_K32_DB, 1),
    Tuned::new(64, true, 1624, 960, CoopTile::T32X32_W16X16_K32, 5),
    Tuned::new(64, true, 2560, 960, CoopTile::T32X64_W16X16_K64_DB, 5),
    Tuned::new(64, false, 960, 960, CoopTile::T32X32_W16X16_K64_DB, 1),
    Tuned::new(64, false, 960, 1624, CoopTile::T32X32_W16X16_K64_DB, 1),
    Tuned::new(64, false, 960, 2560, CoopTile::T32X32_W16X16_K64_DB, 1),
    Tuned::new(64, false, 2560, 960, CoopTile::T32X64_W16X16_K64_DB, 8),
    Tuned::new(64, false, 49152, 960, CoopTile::T64X128_W32X64_K32_DB, 20),
    Tuned::new(128, true, 960, 960, CoopTile::T32X64_W16X32_K32_DB, 4),
    Tuned::new(128, true, 960, 2560, CoopTile::T32X64_W16X16_K64_DB, 1),
    Tuned::new(128, true, 960, 49152, CoopTile::T128X64_W32X32_K32_PF, 1),
    Tuned::new(128, true, 1624, 960, CoopTile::T64X64_W16X32_K32_DB, 6),
    Tuned::new(128, true, 2560, 960, CoopTile::T128X64_W32X32_K32_PF, 10),
    Tuned::new(128, false, 960, 960, CoopTile::T64X32_W32X16_K32_DB, 4),
    Tuned::new(128, false, 960, 1624, CoopTile::T32X64_W16X16_K64_DB, 1),
    Tuned::new(128, false, 960, 2560, CoopTile::T32X64_W16X16_K64_DB, 1),
    Tuned::new(128, false, 2560, 960, CoopTile::T64X64_W32X32_K32_PF, 8),
    Tuned::new(128, false, 49152, 960, CoopTile::T64X64_W32X32_K32_PF, 8),
    Tuned::new(192, true, 960, 960, CoopTile::T32X64_W16X16_K64_DB, 1),
    Tuned::new(192, true, 960, 2560, CoopTile::T64X64_W32X32_K32_PF, 2),
    Tuned::new(192, true, 960, 49152, CoopTile::T128X128_W64X64_K32_PF, 1),
    Tuned::new(192, true, 1624, 960, CoopTile::T64X64_W16X32_K32_DB, 4),
    Tuned::new(192, true, 2560, 960, CoopTile::T64X64_W32X32_K32_PF, 5),
    Tuned::new(192, false, 960, 960, CoopTile::T32X32_W16X16_K64_DB, 1),
    Tuned::new(192, false, 960, 1624, CoopTile::T64X64_W32X32_K32, 1),
    Tuned::new(192, false, 960, 2560, CoopTile::T32X64_W16X32_K32_DB, 1),
    Tuned::new(192, false, 2560, 960, CoopTile::T64X64_W32X32_K32_PF, 7),
    Tuned::new(192, false, 49152, 960, CoopTile::T64X64_W32X32_K32_DB, 7),
    Tuned::new(256, true, 960, 960, CoopTile::T32X32_W16X16_K64_DB, 1),
    Tuned::new(256, true, 960, 2560, CoopTile::T64X64_W16X32_K32_DB, 1),
    Tuned::new(256, true, 960, 49152, CoopTile::T128X128_W64X64_K32_PF, 1),
    Tuned::new(256, true, 1624, 960, CoopTile::T64X64_W16X32_K32_DB, 4),
    Tuned::new(256, true, 2560, 960, CoopTile::T64X64_W32X32_K32_DB, 4),
    Tuned::new(256, false, 960, 960, CoopTile::T32X64_W16X16_K64_DB, 1),
    Tuned::new(256, false, 960, 1624, CoopTile::T128X64_W32X32_K32_PF, 3),
    Tuned::new(256, false, 960, 2560, CoopTile::T32X64_W16X32_K32_DB, 1),
    Tuned::new(256, false, 2560, 960, CoopTile::T128X64_W32X32_K32_DB, 5),
    Tuned::new(256, false, 49152, 960, CoopTile::T128X128_W64X64_K32_PF, 10),
    // SmolLM2-1.7B at 64, 128, 192 and 256 rows
    Tuned::new(64, true, 2048, 2048, CoopTile::T32X32_W16X16_K64_DB, 1),
    Tuned::new(64, true, 2048, 8192, CoopTile::T32X32_W16X16_K64_DB, 1),
    Tuned::new(64, true, 2048, 49152, CoopTile::T64X64_W32X32_K32_DB, 1),
    Tuned::new(64, true, 6168, 2048, CoopTile::T32X64_W16X16_K64_DB, 5),
    Tuned::new(64, true, 8192, 2048, CoopTile::T32X32_W16X16_K64_DB, 5),
    Tuned::new(64, false, 2048, 2048, CoopTile::T64X64_W16X32_K32_DB, 5),
    Tuned::new(64, false, 2048, 6168, CoopTile::T64X64_W16X32_K32_DB, 1),
    Tuned::new(64, false, 2048, 8192, CoopTile::T64X64_W32X32_K32_PF, 1),
    Tuned::new(64, false, 8192, 2048, CoopTile::T64X64_W16X32_K32_DB, 6),
    Tuned::new(64, false, 49152, 2048, CoopTile::T64X128_W32X64_K32_DB, 10),
    Tuned::new(128, true, 2048, 2048, CoopTile::T128X64_W32X32_K32_DB, 5),
    Tuned::new(128, true, 2048, 8192, CoopTile::T128X64_W32X32_K32_PF, 1),
    Tuned::new(128, true, 2048, 49152, CoopTile::T128X64_W32X32_K32_PF, 1),
    Tuned::new(128, true, 6168, 2048, CoopTile::T128X128_W32X32_K32_PF1, 5),
    Tuned::new(128, true, 8192, 2048, CoopTile::T128X64_W32X32_K32_PF, 5),
    Tuned::new(128, false, 2048, 2048, CoopTile::T128X64_W32X32_K32_PF, 5),
    Tuned::new(128, false, 2048, 6168, CoopTile::T64X64_W32X32_K32_PF, 1),
    Tuned::new(128, false, 2048, 8192, CoopTile::T128X64_W32X32_K32_PF, 1),
    Tuned::new(128, false, 8192, 2048, CoopTile::T128X64_W32X32_K32_PF, 5),
    Tuned::new(
        128,
        false,
        49152,
        2048,
        CoopTile::T128X128_W64X64_K32_PF,
        10,
    ),
    Tuned::new(192, true, 2048, 2048, CoopTile::T64X64_W32X32_K32_PF, 3),
    Tuned::new(192, true, 2048, 8192, CoopTile::T64X64_W32X32_K32, 1),
    Tuned::new(192, true, 2048, 49152, CoopTile::T128X128_W64X64_K32_PF, 1),
    Tuned::new(192, true, 6168, 2048, CoopTile::T64X64_W16X32_K32_DB, 5),
    Tuned::new(192, true, 8192, 2048, CoopTile::T64X128_W32X64_K32_DB, 3),
    Tuned::new(192, false, 2048, 2048, CoopTile::T64X64_W32X32_K32_PF, 3),
    Tuned::new(192, false, 2048, 6168, CoopTile::T64X128_W32X64_K32_DB, 1),
    Tuned::new(192, false, 2048, 8192, CoopTile::T64X128_W32X64_K32_DB, 1),
    Tuned::new(192, false, 8192, 2048, CoopTile::T64X128_W32X64_K32_DB, 5),
    Tuned::new(192, false, 49152, 2048, CoopTile::T64X128_W32X64_K32_DB, 5),
    Tuned::new(256, true, 2048, 2048, CoopTile::T64X64_W32X32_K32_PF, 1),
    Tuned::new(256, true, 2048, 8192, CoopTile::T128X128_W64X64_K32_PF, 1),
    Tuned::new(256, true, 2048, 49152, CoopTile::T128X128_W64X64_K32_PF, 1),
    Tuned::new(256, true, 6168, 2048, CoopTile::T64X64_W32X32_K32, 4),
    Tuned::new(256, true, 8192, 2048, CoopTile::T128X128_W64X64_K32_PF, 5),
    Tuned::new(256, false, 2048, 2048, CoopTile::T64X64_W32X32_K32_PF, 1),
    Tuned::new(256, false, 2048, 6168, CoopTile::T64X128_W32X64_K32_DB, 1),
    Tuned::new(256, false, 2048, 8192, CoopTile::T128X128_W64X64_K32_PF, 1),
    Tuned::new(256, false, 8192, 2048, CoopTile::T128X128_W64X64_K32_PF, 5),
    Tuned::new(256, false, 49152, 2048, CoopTile::T128X128_W64X64_K32_PF, 5),
    // SmolLM2-135M at 1024 rows (fused QKV and gate/up with their LoRA columns)
    Tuned::new(1024, false, 576, 984, CoopTile::T64X64_W32X32_K32, 1),
    Tuned::new(1024, true, 984, 576, CoopTile::T64X64_W16X32_K32_DB, 1),
    Tuned::new(1024, false, 576, 3088, CoopTile::T64X64_W32X32_K32, 1),
    Tuned::new(1024, true, 3088, 576, CoopTile::T64X64_W16X32_K32_DB, 1),
    Tuned::new(1024, true, 576, 576, CoopTile::T128X64_W32X32_K32_PF, 1),
    Tuned::new(1024, false, 576, 576, CoopTile::T128X64_W32X32_K32_PF, 1),
    Tuned::new(1024, true, 576, 1536, CoopTile::T64X64_W32X32_K32_PF, 1),
    Tuned::new(1024, false, 1536, 576, CoopTile::T64X64_W16X32_K32_DB, 1),
    Tuned::new(1024, true, 1536, 576, CoopTile::T64X64_W32X32_K32_PF, 1),
    Tuned::new(1024, false, 576, 1536, CoopTile::T64X64_W32X32_K32_PF, 1),
    Tuned::new(1024, true, 576, 49152, CoopTile::T128X128_W64X64_K32_PF, 1),
    Tuned::new(1024, false, 49152, 576, CoopTile::T128X128_W64X64_K32_PF, 4),
    // SmolLM2-135M at 384 rows
    Tuned::new(384, false, 576, 984, CoopTile::T64X64_W32X32_K32, 1),
    Tuned::new(384, true, 984, 576, CoopTile::T64X64_W16X32_K32_DB, 4),
    Tuned::new(384, false, 576, 3088, CoopTile::T128X64_W32X32_K32_PF, 1),
    Tuned::new(384, true, 3088, 576, CoopTile::T64X64_W16X32_K32_DB, 4),
    Tuned::new(384, true, 576, 576, CoopTile::T32X32_W16X16_K64_DB, 1),
    Tuned::new(384, false, 576, 576, CoopTile::T32X32_W16X16_K64_DB, 1),
    Tuned::new(384, true, 576, 1536, CoopTile::T64X64_W32X32_K32_PF, 1),
    Tuned::new(384, false, 1536, 576, CoopTile::T64X64_W32X32_K32_PF, 4),
    Tuned::new(384, true, 1536, 576, CoopTile::T32X32_W16X16_K64_DB, 1),
    Tuned::new(384, false, 576, 1536, CoopTile::T128X64_W32X32_K32_PF, 1),
    Tuned::new(384, true, 576, 49152, CoopTile::T128X128_W64X64_K32_PF, 1),
    Tuned::new(384, false, 49152, 576, CoopTile::T64X128_W32X64_K32_DB, 8),
    // SmolLM2-135M at 512 rows
    Tuned::new(512, false, 576, 984, CoopTile::T64X64_W32X32_K32, 1),
    Tuned::new(512, true, 984, 576, CoopTile::T32X64_W16X16_K64_DB, 1),
    Tuned::new(512, false, 576, 3088, CoopTile::T64X64_W16X32_K32_DB, 1),
    Tuned::new(512, true, 3088, 576, CoopTile::T64X64_W16X32_K32_DB, 3),
    Tuned::new(512, true, 576, 576, CoopTile::T32X64_W16X16_K64_DB, 1),
    Tuned::new(512, false, 576, 576, CoopTile::T32X64_W16X16_K64_DB, 1),
    Tuned::new(512, true, 576, 1536, CoopTile::T64X64_W32X32_K32_PF, 1),
    Tuned::new(512, false, 1536, 576, CoopTile::T64X64_W32X32_K32_PF, 3),
    Tuned::new(512, true, 1536, 576, CoopTile::T128X64_W32X32_K32_PF, 4),
    Tuned::new(512, false, 576, 1536, CoopTile::T64X64_W16X32_K32_DB, 1),
    Tuned::new(512, true, 576, 49152, CoopTile::T128X128_W64X64_K32_PF, 1),
    Tuned::new(512, false, 49152, 576, CoopTile::T128X128_W64X64_K32_PF, 8),
    // SmolLM2-360M at 320 rows
    Tuned::new(320, false, 960, 1624, CoopTile::T64X64_W32X32_K32, 1),
    Tuned::new(320, true, 1624, 960, CoopTile::T64X64_W16X32_K32_DB, 3),
    Tuned::new(320, true, 960, 960, CoopTile::T32X64_W16X16_K64_DB, 1),
    Tuned::new(320, false, 960, 960, CoopTile::T32X64_W16X32_K32_DB, 1),
    Tuned::new(320, true, 960, 2560, CoopTile::T64X64_W32X32_K32_PF, 1),
    Tuned::new(320, false, 2560, 960, CoopTile::T64X64_W32X32_K32_PF, 3),
    Tuned::new(320, true, 2560, 960, CoopTile::T64X64_W16X32_K32_DB, 2),
    Tuned::new(320, false, 960, 2560, CoopTile::T64X64_W32X32_K32_DB, 1),
    Tuned::new(320, true, 960, 49152, CoopTile::T128X64_W32X32_K32_PF, 1),
    Tuned::new(320, false, 49152, 960, CoopTile::T64X128_W32X64_K32_DB, 6),
    // SmolLM2-360M at 384 rows
    Tuned::new(384, false, 960, 1624, CoopTile::T64X64_W32X32_K32, 1),
    Tuned::new(384, true, 1624, 960, CoopTile::T128X128_W32X32_K32_PF1, 3),
    Tuned::new(384, true, 960, 960, CoopTile::T64X64_W32X32_K32_DB, 1),
    Tuned::new(384, false, 960, 960, CoopTile::T64X64_W32X32_K32_DB, 1),
    Tuned::new(384, true, 960, 2560, CoopTile::T64X64_W32X32_K32_DB, 1),
    Tuned::new(384, false, 2560, 960, CoopTile::T64X64_W32X32_K32_DB, 3),
    Tuned::new(384, true, 2560, 960, CoopTile::T128X64_W32X32_K32_PF, 3),
    Tuned::new(384, false, 960, 2560, CoopTile::T64X64_W32X32_K32_PF, 1),
    Tuned::new(384, true, 960, 49152, CoopTile::T128X128_W64X64_K32_PF, 1),
    Tuned::new(384, false, 49152, 960, CoopTile::T128X64_W64X32_K32_PF, 7),
    // SmolLM2-360M at 512 rows
    Tuned::new(512, false, 960, 1624, CoopTile::T64X64_W32X32_K32, 1),
    Tuned::new(512, true, 1624, 960, CoopTile::T64X64_W16X32_K32_DB, 2),
    Tuned::new(512, true, 960, 960, CoopTile::T64X64_W32X32_K32_PF, 1),
    Tuned::new(512, false, 960, 960, CoopTile::T32X64_W16X32_K32_DB, 1),
    Tuned::new(512, true, 960, 2560, CoopTile::T128X64_W32X32_K32_PF, 1),
    Tuned::new(512, false, 2560, 960, CoopTile::T64X64_W32X32_K32_PF, 2),
    Tuned::new(512, true, 2560, 960, CoopTile::T64X64_W32X32_K32_DB, 2),
    Tuned::new(512, false, 960, 2560, CoopTile::T128X64_W32X32_K32_PF, 1),
    Tuned::new(512, true, 960, 49152, CoopTile::T128X128_W64X64_K32_PF, 1),
    Tuned::new(512, false, 49152, 960, CoopTile::T128X128_W64X64_K32_PF, 5),
    // SmolLM2-360M at 640 rows
    Tuned::new(640, false, 960, 1624, CoopTile::T64X128_W32X64_K32_DB, 1),
    Tuned::new(640, true, 1624, 960, CoopTile::T64X64_W16X32_K32_DB, 1),
    Tuned::new(640, true, 960, 960, CoopTile::T64X64_W32X32_K32_PF, 1),
    Tuned::new(640, false, 960, 960, CoopTile::T64X64_W32X32_K32_PF, 1),
    Tuned::new(640, true, 960, 2560, CoopTile::T64X64_W32X32_K32, 1),
    Tuned::new(640, false, 2560, 960, CoopTile::T64X64_W32X32_K32_PF, 1),
    Tuned::new(640, true, 2560, 960, CoopTile::T64X64_W32X32_K32_PF, 1),
    Tuned::new(640, false, 960, 2560, CoopTile::T64X128_W32X64_K32_DB, 1),
    Tuned::new(640, true, 960, 49152, CoopTile::T128X128_W64X64_K32_PF, 1),
    Tuned::new(640, false, 49152, 960, CoopTile::T128X128_W64X64_K32_PF, 4),
    // SmolLM2-360M at 1024 rows
    Tuned::new(1024, false, 960, 1624, CoopTile::T64X128_W32X64_K32_DB, 1),
    Tuned::new(1024, true, 1624, 960, CoopTile::T64X64_W16X32_K32_DB, 1),
    Tuned::new(1024, true, 960, 960, CoopTile::T64X64_W32X32_K32_PF, 1),
    Tuned::new(1024, false, 960, 960, CoopTile::T64X64_W32X32_K32_DB, 1),
    Tuned::new(1024, true, 960, 2560, CoopTile::T128X64_W32X32_K32_PF, 1),
    Tuned::new(1024, false, 2560, 960, CoopTile::T64X64_W32X32_K32_PF, 1),
    Tuned::new(1024, true, 2560, 960, CoopTile::T64X64_W32X32_K32_PF, 1),
    Tuned::new(1024, false, 960, 2560, CoopTile::T128X64_W32X32_K32_PF, 1),
    Tuned::new(1024, true, 960, 49152, CoopTile::T128X128_W64X64_K32_PF, 1),
    Tuned::new(1024, false, 49152, 960, CoopTile::T128X128_W64X64_K32_PF, 5),
    // SmolLM2-360M at 1280 rows
    Tuned::new(1280, false, 960, 1624, CoopTile::T64X64_W32X32_K32, 1),
    Tuned::new(1280, true, 1624, 960, CoopTile::T64X64_W16X32_K32_DB, 1),
    Tuned::new(1280, true, 960, 960, CoopTile::T64X64_W32X32_K32_PF, 1),
    Tuned::new(1280, false, 960, 960, CoopTile::T64X128_W32X64_K32_DB, 1),
    Tuned::new(1280, true, 960, 2560, CoopTile::T128X64_W32X32_K32_PF, 1),
    Tuned::new(1280, false, 2560, 960, CoopTile::T64X128_W32X64_K32_DB, 1),
    Tuned::new(1280, true, 2560, 960, CoopTile::T128X64_W32X32_K32_PF, 1),
    Tuned::new(1280, false, 960, 2560, CoopTile::T64X128_W32X64_K32_DB, 1),
    Tuned::new(1280, true, 960, 49152, CoopTile::T128X128_W64X64_K32_PF, 1),
    Tuned::new(1280, false, 49152, 960, CoopTile::T128X128_W64X64_K32_PF, 8),
    // SmolLM2-360M at 2048 rows
    Tuned::new(2048, false, 960, 1624, CoopTile::T64X64_W32X32_K32, 1),
    Tuned::new(2048, true, 1624, 960, CoopTile::T64X64_W16X32_K32_DB, 1),
    Tuned::new(2048, true, 960, 960, CoopTile::T64X64_W32X32_K32_PF, 1),
    Tuned::new(2048, false, 960, 960, CoopTile::T64X64_W32X32_K32_PF, 1),
    Tuned::new(2048, true, 960, 2560, CoopTile::T128X64_W32X32_K32_PF, 1),
    Tuned::new(2048, false, 2560, 960, CoopTile::T64X64_W32X32_K32_PF, 1),
    Tuned::new(2048, true, 2560, 960, CoopTile::T64X64_W32X32_K32_PF, 1),
    Tuned::new(2048, false, 960, 2560, CoopTile::T64X128_W32X64_K32_DB, 1),
    Tuned::new(2048, true, 960, 49152, CoopTile::T128X128_W64X64_K32_PF, 1),
    Tuned::new(2048, false, 49152, 960, CoopTile::T128X128_W64X64_K32_PF, 5),
    // SmolLM2-1.7B at 320 rows
    Tuned::new(320, false, 2048, 6168, CoopTile::T128X128_W64X64_K32_PF, 1),
    Tuned::new(320, true, 6168, 2048, CoopTile::T64X64_W16X32_K32_DB, 1),
    Tuned::new(320, true, 2048, 2048, CoopTile::T64X64_W32X32_K32_PF, 1),
    Tuned::new(320, false, 2048, 2048, CoopTile::T64X64_W16X32_K32_DB, 1),
    Tuned::new(320, true, 2048, 8192, CoopTile::T64X128_W32X64_K32_DB, 1),
    Tuned::new(320, false, 8192, 2048, CoopTile::T64X128_W32X64_K32_DB, 3),
    Tuned::new(320, true, 8192, 2048, CoopTile::T64X128_W32X64_K32_DB, 2),
    Tuned::new(320, false, 2048, 8192, CoopTile::T64X128_W32X64_K32_DB, 1),
    Tuned::new(320, true, 2048, 49152, CoopTile::T128X128_W64X64_K32_PF, 1),
    Tuned::new(320, false, 49152, 2048, CoopTile::T64X128_W32X64_K32_DB, 5),
    // SmolLM2-1.7B at 384 rows
    Tuned::new(384, false, 2048, 6168, CoopTile::T128X128_W64X64_K32_PF, 1),
    Tuned::new(384, true, 6168, 2048, CoopTile::T128X128_W64X32_K32_PF1, 3),
    Tuned::new(384, true, 2048, 2048, CoopTile::T64X64_W32X32_K32_PF, 1),
    Tuned::new(384, false, 2048, 2048, CoopTile::T64X64_W32X32_K32_PF, 1),
    Tuned::new(384, true, 2048, 8192, CoopTile::T128X64_W32X32_K32_PF, 1),
    Tuned::new(384, false, 8192, 2048, CoopTile::T128X128_W64X64_K32_PF, 3),
    Tuned::new(384, true, 8192, 2048, CoopTile::T128X128_W64X64_K32_PF, 3),
    Tuned::new(384, false, 2048, 8192, CoopTile::T128X64_W32X32_K32_PF, 1),
    Tuned::new(384, true, 2048, 49152, CoopTile::T128X128_W64X64_K32_PF, 1),
    Tuned::new(384, false, 49152, 2048, CoopTile::T128X128_W64X64_K32_PF, 8),
    // SmolLM2-1.7B at 512 rows
    Tuned::new(512, false, 2048, 6168, CoopTile::T64X128_W32X64_K32_DB, 1),
    Tuned::new(512, true, 6168, 2048, CoopTile::T64X64_W32X32_K32, 2),
    Tuned::new(512, true, 2048, 2048, CoopTile::T128X64_W32X32_K32_PF, 1),
    Tuned::new(512, false, 2048, 2048, CoopTile::T64X64_W32X32_K32_PF, 1),
    Tuned::new(512, true, 2048, 8192, CoopTile::T128X128_W64X64_K32_PF, 1),
    Tuned::new(512, false, 8192, 2048, CoopTile::T64X128_W32X64_K32_DB, 5),
    Tuned::new(512, true, 8192, 2048, CoopTile::T128X128_W64X64_K32_PF, 5),
    Tuned::new(512, false, 2048, 8192, CoopTile::T128X128_W64X64_K32_PF, 1),
    Tuned::new(512, true, 2048, 49152, CoopTile::T128X128_W64X64_K32_PF, 1),
    Tuned::new(512, false, 49152, 2048, CoopTile::T128X128_W64X64_K32_PF, 5),
    // SmolLM2-1.7B at 640 rows
    Tuned::new(640, false, 2048, 6168, CoopTile::T64X128_W32X64_K32_DB, 1),
    Tuned::new(640, true, 6168, 2048, CoopTile::T128X128_W32X32_K32_PF1, 1),
    Tuned::new(640, true, 2048, 2048, CoopTile::T128X64_W32X32_K32_PF, 1),
    Tuned::new(640, false, 2048, 2048, CoopTile::T128X64_W64X32_K32_PF, 1),
    Tuned::new(640, true, 2048, 8192, CoopTile::T128X128_W64X64_K32_PF, 1),
    Tuned::new(640, false, 8192, 2048, CoopTile::T64X128_W32X64_K32_DB, 1),
    Tuned::new(640, true, 8192, 2048, CoopTile::T128X128_W64X64_K32_PF, 2),
    Tuned::new(640, false, 2048, 8192, CoopTile::T128X128_W64X64_K32_PF, 1),
    Tuned::new(640, true, 2048, 49152, CoopTile::T128X128_W64X64_K32_PF, 1),
    Tuned::new(640, false, 49152, 2048, CoopTile::T128X128_W64X64_K32_PF, 2),
    // SmolLM2-1.7B at 1024 rows
    Tuned::new(1024, false, 2048, 6168, CoopTile::T128X128_W64X64_K32_PF, 1),
    Tuned::new(1024, true, 6168, 2048, CoopTile::T64X64_W32X32_K32, 1),
    Tuned::new(1024, true, 2048, 2048, CoopTile::T64X64_W32X32_K32_PF, 1),
    Tuned::new(1024, false, 2048, 2048, CoopTile::T64X64_W32X32_K32_PF, 1),
    Tuned::new(1024, true, 2048, 8192, CoopTile::T128X128_W64X64_K32_PF, 1),
    Tuned::new(1024, false, 8192, 2048, CoopTile::T128X128_W64X64_K32_PF, 3),
    Tuned::new(1024, true, 8192, 2048, CoopTile::T128X128_W64X64_K32_PF, 5),
    Tuned::new(1024, false, 2048, 8192, CoopTile::T128X128_W64X64_K32_PF, 1),
    Tuned::new(1024, true, 2048, 49152, CoopTile::T128X128_W64X64_K32_PF, 1),
    Tuned::new(
        1024,
        false,
        49152,
        2048,
        CoopTile::T128X128_W64X64_K32_PF,
        5,
    ),
    // SmolLM2-1.7B at 1280 rows
    Tuned::new(1280, false, 2048, 6168, CoopTile::T128X128_W64X64_K32_PF, 1),
    Tuned::new(1280, true, 6168, 2048, CoopTile::T64X64_W16X32_K32_DB, 1),
    Tuned::new(1280, true, 2048, 2048, CoopTile::T128X64_W32X32_K32_PF, 1),
    Tuned::new(1280, false, 2048, 2048, CoopTile::T64X128_W32X64_K32_DB, 1),
    Tuned::new(1280, true, 2048, 8192, CoopTile::T128X128_W64X64_K32_PF, 1),
    Tuned::new(1280, false, 8192, 2048, CoopTile::T128X128_W64X64_K32_PF, 1),
    Tuned::new(1280, true, 8192, 2048, CoopTile::T128X128_W64X64_K32_PF, 1),
    Tuned::new(1280, false, 2048, 8192, CoopTile::T128X128_W64X64_K32_PF, 1),
    Tuned::new(1280, true, 2048, 49152, CoopTile::T128X128_W64X64_K32_PF, 1),
    Tuned::new(
        1280,
        false,
        49152,
        2048,
        CoopTile::T128X128_W64X64_K32_PF,
        5,
    ),
    // SmolLM2-1.7B at 2048 rows
    Tuned::new(2048, false, 2048, 6168, CoopTile::T128X128_W64X64_K32_PF, 1),
    Tuned::new(2048, true, 6168, 2048, CoopTile::T64X64_W32X32_K32, 1),
    Tuned::new(2048, true, 2048, 2048, CoopTile::T128X64_W32X32_K32_PF, 1),
    Tuned::new(2048, false, 2048, 2048, CoopTile::T64X128_W32X64_K32_DB, 1),
    Tuned::new(2048, true, 2048, 8192, CoopTile::T128X128_W64X64_K32_PF, 1),
    Tuned::new(2048, false, 8192, 2048, CoopTile::T64X128_W32X64_K32_DB, 1),
    Tuned::new(2048, true, 8192, 2048, CoopTile::T128X128_W64X64_K32_PF, 1),
    Tuned::new(2048, false, 2048, 8192, CoopTile::T128X128_W64X64_K32_PF, 1),
    Tuned::new(2048, true, 2048, 49152, CoopTile::T128X128_W64X64_K32_PF, 1),
    Tuned::new(
        2048,
        false,
        49152,
        2048,
        CoopTile::T128X128_W64X64_K32_PF,
        5,
    ),
];

/// The size of a fragment row: the rows of a product are rounded up to a
/// multiple of it before the router looks them up.
pub(super) const ROW_STEP: u32 = 16;

/// The row buckets of the router, as `(last row, TUNED rows in the order they
/// are looked up)`. The first row of a bucket is one past the last row of the
/// bucket before. Up to 256 rows each bucket has its own row and then the
/// 256-row one. From 257 rows the bucket's own row comes first, then the
/// 1024-row one and the 2048-row one (135M and the narrow products have rows
/// only for some buckets, and run the next one that has theirs). The 1025 to
/// 1408 bucket is the 1280-row row, then the 2048-row one. Training batches
/// are padded to their longest sequence, so a step's row count is rarely
/// exactly the one a row was measured at, and an exact match would leave most
/// products to the fallback rule.
const ROW_BUCKETS: &[(u32, &[u32])] = &[
    (64, &[64, 256]),
    (128, &[128, 256]),
    (192, &[192, 256]),
    (256, &[256]),
    (320, &[320, 1024, 2048]),
    (384, &[384, 1024, 2048]),
    (512, &[512, 1024, 2048]),
    (640, &[640, 1024, 2048]),
    (1024, &[1024, 2048]),
    (1408, &[1280, 2048]),
    (2048, &[2048]),
];

/// The bucket of `m` rows (rounded up to a multiple of [`ROW_STEP`]), or
/// `None` past 2048 rows, where the router has no rows.
fn bucket_of(m: u32) -> Option<&'static (u32, &'static [u32])> {
    let rows = m.div_ceil(ROW_STEP) * ROW_STEP;
    ROW_BUCKETS.iter().find(|b| rows <= b.0)
}

/// The [`TUNED`] rows to look up for `m` rows, in order.
fn tuned_rows(m: u32) -> &'static [u32] {
    match bucket_of(m) {
        Some(bucket) => bucket.1,
        None => &[],
    }
}

/// [`TUNED`] by `(rows, transposed_b, k, n)`, to the position of its row. The
/// router looks a product up at every dispatch, and a scan of the table for
/// each of the up to three row keys of a bucket takes longer as the table
/// grows, which the 135M and 360M steps, that wait on the CPU, showed.
static TUNED_INDEX: OnceLock<HashMap<(u32, bool, u32, u32), usize>> = OnceLock::new();

/// The measured configuration for this product, when the router has one.
fn tuned(m: u32, transposed_b: bool, k: u32, n: u32) -> Option<CoopConfig> {
    let index = TUNED_INDEX.get_or_init(|| {
        let mut index = HashMap::with_capacity(TUNED.len());
        for (at, t) in TUNED.iter().enumerate() {
            // The first row of a key wins, as a scan of the table would.
            index.entry((t.m, t.transposed_b, t.k, t.n)).or_insert(at);
        }
        index
    });
    let row = tuned_rows(m).iter().find_map(|&rows| {
        index
            .get(&(rows, transposed_b, k, n))
            .and_then(|&at| TUNED.get(at))
    })?;
    Some(CoopConfig {
        tile: row.tile,
        split: row.split,
    })
}

/// The 1.7B base-weight products of up to 256 rows, as `(a@bT, k, n)`, whose
/// `matmul-tune` winners lost in a training step. Their rows are now the
/// small-row sweep's and this list applies only where a row's tile does not
/// fit the device. The bench winners (the
/// 128x64 and 64x64 tiles with one staging buffer, one K split of 3 or none)
/// took 1.3 to 2.4 times as long in a `--profile` of a 30-step 1.7B run with
/// micro-batches of 64 to 256 rows as the double-buffered 64x64 tile with K
/// split to fill the GPU, and the 60-step median was 0.464 s against 0.327 s.
/// The bench times one product alone, repeated with a warm cache, and the
/// step does not. These products run [`STEP_TILES`] instead; the other 1.7B
/// products and all of 135M and 360M keep their table rows or the rule above.
const STEP_RULE: &[(bool, u32, u32)] = &[
    (true, 2048, 2048),
    (false, 2048, 2048),
    (true, 2048, 8192),
    (false, 8192, 2048),
    (true, 8192, 2048),
    (false, 2048, 8192),
    (true, 2048, 49152),
    (false, 49152, 2048),
];
/// Tiles [`STEP_RULE`] products try, in order, as for [`SMALL_M_TILES`].
const STEP_TILES: [CoopTile; 2] = [
    CoopTile::T128X64_W64X32_K32_DB,
    CoopTile::T64X64_W32X32_K32_DB,
];

/// Environment variable that turns the [`M64_PINS`] off when it is `0`. It
/// exists for benchmarks that compare the router with and without the pins.
pub(super) const PINS_ENV: &str = "FERROTENSOR_M64_PINS";

/// Whether the [`M64_PINS`] stand, for a value of [`PINS_ENV`].
pub(super) fn pins_enabled(value: Option<&str>) -> bool {
    value.map(str::trim) != Some("0")
}

/// A product of up to [`PIN_ROWS`] rows with a configuration that won in a
/// training step.
struct Pin {
    transposed_b: bool,
    loader: LoaderKind,
    epilogue: EpilogueKind,
    k: u32,
    n: u32,
    tile: CoopTile,
    split: u32,
}

impl Pin {
    const fn new(
        (transposed_b, loader, epilogue): (bool, LoaderKind, EpilogueKind),
        (k, n): (u32, u32),
        tile: CoopTile,
        split: u32,
    ) -> Self {
        Self {
            transposed_b,
            loader,
            epilogue,
            k,
            n,
            tile,
            split,
        }
    }
}

/// Rows up to which the [`M64_PINS`] apply: the autotuner's first row bucket.
const PIN_ROWS: u32 = 64;

/// The eight SmolLM2-1.7B products of up to 64 rows (a batch of 1 at 64
/// tokens or fewer) whose picks made the training step faster. On one A10G,
/// a 1.7B run of batch 1 with 8 accumulation steps took 0.2536 s per step
/// when these eight picks replaced the eight of a cache that took 0.2624 s
/// (the other 22 picks of the cache stayed), and a `matmul_bench --tiles`
/// sweep of the products alone did not show the difference. The picks use
/// 2, 2, 1, 2, 2, 8, 2 and 2 slices of K where the cache had 5, 5, 1, 3, 4,
/// 5, 1 and 5, and on four of them another tile. The first four are
/// `a @ b` products and the rest `a @ bT`, each as `(transposed_b, loader,
/// epilogue)`.
const M64_PINS: &[Pin] = &[
    Pin::new(
        (false, LoaderKind::Plain, EpilogueKind::AlphaBeta),
        (2048, 2048),
        CoopTile::T32X64_W16X16_K64_DB,
        2,
    ),
    Pin::new(
        (false, LoaderKind::Plain, EpilogueKind::AlphaBeta),
        (8192, 2048),
        CoopTile::T32X64_W16X16_K64_DB,
        2,
    ),
    Pin::new(
        (false, LoaderKind::Plain, EpilogueKind::AlphaBeta),
        (2048, 8192),
        CoopTile::T64X64_W16X32_K32_DB,
        1,
    ),
    Pin::new(
        (
            false,
            LoaderKind::Plain,
            EpilogueKind::ColumnSplit {
                segments: 4,
                bias: false,
            },
        ),
        (2048, 6168),
        CoopTile::T64X64_W16X32_K32_DB,
        2,
    ),
    Pin::new(
        (true, LoaderKind::Plain, EpilogueKind::AlphaBeta),
        (2048, 2048),
        CoopTile::T32X64_W16X16_K64_DB,
        2,
    ),
    Pin::new(
        (true, LoaderKind::Plain, EpilogueKind::AlphaBeta),
        (8192, 2048),
        CoopTile::T64X64_W32X32_K32_PF,
        8,
    ),
    Pin::new(
        (true, LoaderKind::Plain, EpilogueKind::AlphaBeta),
        (2048, 8192),
        CoopTile::T64X64_W32X32_K32_PF,
        2,
    ),
    Pin::new(
        (
            true,
            LoaderKind::Sources { count: 4 },
            EpilogueKind::AlphaBeta,
        ),
        (6168, 2048),
        CoopTile::T32X64_W16X16_K64_DB,
        2,
    ),
];

/// The [`M64_PINS`] row for an `m x k x n` product of the kernel variant
/// `kinds`, when it has one that `fits` the device. Pins are measured on a
/// discrete GPU, so `use_tuned` is false elsewhere and there is none.
pub(super) fn pinned(
    (m, k, n): (u32, u32, u32),
    transposed_b: bool,
    kinds: (LoaderKind, EpilogueKind),
    use_tuned: bool,
    fits: impl Fn(CoopTile) -> bool,
) -> Option<CoopConfig> {
    if !use_tuned || m > PIN_ROWS {
        return None;
    }
    let pin = M64_PINS.iter().find(|p| {
        (p.transposed_b, p.loader, p.epilogue, p.k, p.n) == (transposed_b, kinds.0, kinds.1, k, n)
    })?;
    (fits(pin.tile) && grid_fits(pin.tile, m, n, pin.split)).then_some(CoopConfig {
        tile: pin.tile,
        split: pin.split,
    })
}

/// `config`, the router's pick for the product, or its [`pinned`] row.
#[cfg(test)]
pub(super) fn pin_or(
    config: CoopConfig,
    dims: (u32, u32, u32),
    transposed_b: bool,
    kinds: (LoaderKind, EpilogueKind),
    use_tuned: bool,
    fits: impl Fn(CoopTile) -> bool,
) -> CoopConfig {
    pinned(dims, transposed_b, kinds, use_tuned, fits).unwrap_or(config)
}

/// `(splits, K per split)` for `requested` slices of K = `k` in steps of
/// `bk`: every slice a whole number of steps, none empty.
pub(super) fn split_k(k: u32, bk: u32, requested: u32) -> (u32, u32) {
    let per = k.div_ceil(bk).div_ceil(requested.max(1)).max(1) * bk;
    (k.div_ceil(per).max(1), per)
}

/// Whether a grid of `tile` over `m x n` with `split` slices is within the
/// per-dimension limit.
pub(super) fn grid_fits(tile: CoopTile, m: u32, n: u32, split: u32) -> bool {
    let (gx, gy) = tile.groups(m, n);
    gx <= MAX_GROUPS && gy <= MAX_GROUPS && split <= MAX_GROUPS
}

/// The configuration the router runs an `m x k x n` product with. On a
/// device the [`TUNED`] rows were measured on (`use_tuned`), a product with a
/// row there runs that row's configuration when its tile `fits`. Any other
/// product (a [`STEP_RULE`] one with its own list) gets the first tile of the list for `m` that `fits` and whose grid
/// reaches `target` workgroups, else the last one that fits with K split
/// until the grid reaches the target (at most one slice per
/// [`MIN_STEPS_PER_SPLIT`] steps). The first kernel's tiles stand in on a
/// device that fits none of the list.
pub(super) fn plan(
    dims: (u32, u32, u32),
    transposed_b: bool,
    target: (u32, bool),
    fits: impl Fn(CoopTile) -> bool,
) -> Option<CoopConfig> {
    let (_, k, n) = dims;
    let in_step_rule = target.1 && STEP_RULE.contains(&(transposed_b, k, n));
    let small: &[CoopTile] = if in_step_rule {
        &STEP_TILES
    } else {
        &SMALL_M_TILES
    };
    plan_with(dims, transposed_b, target, small, fits)
}

/// Weight elements (`k * n`) from which a fused product of up to 256 rows
/// runs [`STEP_TILES`], the tiles of the 1.7B base products (2048 x 2048 and
/// up). Its fused N holds the LoRA `A` columns, so it is in no [`STEP_RULE`]
/// row.
const FUSED_STEP_WEIGHTS: u64 = 2048 * 2048;

/// The configuration a fused product runs with: as [`plan`], but a product
/// of up to 256 rows on a weight of [`FUSED_STEP_WEIGHTS`] elements or more
/// takes [`STEP_TILES`], which no [`STEP_RULE`] row names for it, and its K
/// split stands (the fused kernels run the split with the epilogue in the
/// reduction).
pub(super) fn plan_fused(
    dims: (u32, u32, u32),
    transposed_b: bool,
    target: (u32, bool),
    fits: impl Fn(CoopTile) -> bool,
) -> Option<CoopConfig> {
    let (_, k, n) = dims;
    let large_weight = u64::from(k) * u64::from(n) >= FUSED_STEP_WEIGHTS;
    let small: &[CoopTile] = if large_weight {
        &STEP_TILES
    } else {
        &SMALL_M_TILES
    };
    plan_with(dims, transposed_b, target, small, fits)
}

/// [`plan`] with `small` as the tile list of products of up to 256 rows.
fn plan_with(
    (m, k, n): (u32, u32, u32),
    transposed_b: bool,
    (target, use_tuned): (u32, bool),
    small: &[CoopTile],
    fits: impl Fn(CoopTile) -> bool,
) -> Option<CoopConfig> {
    let measured = tuned(m, transposed_b, k, n)
        .filter(|c| use_tuned && fits(c.tile) && grid_fits(c.tile, m, n, c.split));
    if measured.is_some() {
        return measured;
    }
    let list: &[CoopTile] = if m > 256 { &LARGE_M_TILES } else { small };
    let usable = |tile: &CoopTile| fits(*tile) && grid_fits(*tile, m, n, 1);
    let mut tiles: Vec<CoopTile> = list.iter().copied().filter(usable).collect();
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
    Some(CoopConfig {
        tile,
        split: split_to_fill(tile, (m, k, n), target),
    })
}

/// How many slices of K `tile` needs over `m x k x n` for its grid to reach
/// `target` workgroups: at most one slice per [`MIN_STEPS_PER_SPLIT`] steps
/// of K, and 1 when the grid already reaches it.
pub(super) fn split_to_fill(tile: CoopTile, (m, k, n): (u32, u32, u32), target: u32) -> u32 {
    let (gx, gy) = tile.groups(m, n);
    let most = (k.div_ceil(tile.bk) / MIN_STEPS_PER_SPLIT).max(1);
    let wanted = target.div_ceil(gx * gy).clamp(1, most);
    split_k(k, tile.bk, wanted).0
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::{
        bucket_of, pin_or, pinned, pins_enabled, plan, plan_fused, split_k, CoopConfig, CoopTile,
        EpilogueKind, LoaderKind, LARGE_M_TILES, M64_PINS, SMALL_M_TILES, STEP_TILES, TUNED,
    };

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

    #[test]
    fn fused_products_keep_the_split_and_take_step_tiles_on_large_weights() {
        let all = |_: CoopTile| true;
        let untuned = (160, false);
        let pick = |c: Option<super::CoopConfig>| c.map(|c| (c.tile, c.split));
        let rule = |dims, transposed_b| pick(plan(dims, transposed_b, untuned, all));
        let fused = |dims, transposed_b| pick(plan_fused(dims, transposed_b, untuned, all));
        // 1.7B QKV forward at 80 rows (N holds 3 x 16 LoRA columns): the
        // 64x64 step tile has 97 x 2 = 194 workgroups, past the target. The
        // plain rule would run the 32x64 tile there.
        assert_eq!(fused((80, 2048, 6192), true), Some((STEP_TILES[1], 1)));
        assert_eq!(rule((80, 2048, 6192), true), Some((SMALL_M_TILES[0], 1)));
        // 1.7B QKV dX: 32 x 2 = 64 workgroups, so K = 6192 splits in 3 slices
        // of 2,080.
        assert_eq!(fused((80, 6192, 2048), false), Some((STEP_TILES[1], 3)));
        // 135M QKV forward: the small weight keeps the small-M tiles, and
        // 16 x 2 = 32 workgroups split K = 576 in 5 slices of 128.
        assert_eq!(fused((78, 576, 1008), true), Some((SMALL_M_TILES[1], 5)));
        // Above 256 rows both rules agree.
        assert_eq!(
            fused((1024, 2048, 6192), true),
            rule((1024, 2048, 6192), true)
        );
    }

    /// The fastest configuration `matmul-tune` found on an NVIDIA A10G for
    /// each base-weight product of SmolLM2 training: `(m, a@bT, k, n,
    /// configuration)`, copied from that run's `matmul_tune.md`.
    const BENCH: &[(u32, bool, u32, u32, &str)] = &[
        // SmolLM2-135M
        (256, true, 576, 192, "32x64/w16x16/k64/db s5"),
        (256, false, 192, 576, "32x64/w16x16/k64/db s1"),
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
        (256, true, 960, 320, "32x64/w16x16/k64/db s4"),
        (256, false, 320, 960, "32x64/w16x16/k64/db s1"),
        (2048, true, 960, 320, "64x64/w16x32/k32/db s1"),
        (2048, false, 320, 960, "64x64/w32x32/k32 s1"),
    ];

    /// The fastest configuration of each SmolLM2-135M product at 1024 rows in
    /// one `matmul_bench --tiles --rows 1024` run on an NVIDIA A10G (the K
    /// split kept only where it was 10% faster than the best unsplit one):
    /// `(m, a@bT, k, n, configuration)`.
    const BENCH_1024: &[(u32, bool, u32, u32, &str)] = &[
        // SmolLM2-135M
        (1024, false, 576, 984, "64x64/w32x32/k32 s1"),
        (1024, true, 984, 576, "64x64/w16x32/k32/db s1"),
        (1024, false, 576, 3088, "64x64/w32x32/k32 s1"),
        (1024, true, 3088, 576, "64x64/w16x32/k32/db s1"),
        (1024, true, 576, 576, "128x64/w32x32/k32/db/pf s1"),
        (1024, false, 576, 576, "128x64/w32x32/k32/db/pf s1"),
        (1024, true, 576, 1536, "64x64/w32x32/k32/db/pf s1"),
        (1024, false, 1536, 576, "64x64/w16x32/k32/db s1"),
        (1024, true, 1536, 576, "64x64/w32x32/k32/db/pf s1"),
        (1024, false, 576, 1536, "64x64/w32x32/k32/db/pf s1"),
        (1024, true, 576, 49152, "128x128/w64x64/k32/db/pf s1"),
        (1024, false, 49152, 576, "128x128/w64x64/k32/db/pf s4"),
    ];

    /// The configuration with the lowest geometric mean time over 1536, 1792
    /// and 2048 rows that `matmul_bench --tiles` found on an NVIDIA A10G for
    /// the products of SmolLM2-1.7B training in the 2048-row bucket: `(bucket
    /// rows, a@bT, k, n, configuration)`.
    const BENCH_LARGE: &[(u32, bool, u32, u32, &str)] = &[
        // SmolLM2-1.7B at 2048 rows
        (2048, false, 2048, 6168, "128x128/w64x64/k32/db/pf s1"),
        (2048, true, 6168, 2048, "64x64/w32x32/k32 s1"),
        (2048, true, 2048, 2048, "128x64/w32x32/k32/db/pf s1"),
        (2048, false, 2048, 2048, "64x128/w32x64/k32/db s1"),
        (2048, true, 2048, 8192, "128x128/w64x64/k32/db/pf s1"),
        (2048, false, 8192, 2048, "64x128/w32x64/k32/db s1"),
        (2048, true, 8192, 2048, "128x128/w64x64/k32/db/pf s1"),
        (2048, false, 2048, 8192, "128x128/w64x64/k32/db/pf s1"),
        (2048, true, 2048, 49152, "128x128/w64x64/k32/db/pf s1"),
        (2048, false, 49152, 2048, "128x128/w64x64/k32/db/pf s5"),
    ];

    /// The configuration with the lowest geometric mean time over the row
    /// counts of each bucket that `matmul_bench --tiles` found on an NVIDIA
    /// A10G for the products of SmolLM2 training at up to 256 rows: `(bucket
    /// rows, a@bT, k, n, configuration)`.
    const BENCH_SMALL: &[(u32, bool, u32, u32, &str)] = &[
        // SmolLM2-135M
        (64, true, 576, 576, "32x32/w16x16/k64/db s5"),
        (64, true, 576, 1536, "32x64/w16x32/k32/db s1"),
        (64, true, 576, 49152, "64x64/w32x32/k32/db/pf s1"),
        (64, true, 984, 576, "32x32/w16x16/k64/db s4"),
        (64, true, 1536, 576, "32x32/w16x16/k64/db s4"),
        (64, true, 3088, 576, "32x64/w16x16/k64/db s7"),
        (64, false, 576, 576, "32x32/w16x16/k64/db s5"),
        (64, false, 576, 984, "32x32/w16x16/k64/db s1"),
        (64, false, 576, 1536, "32x32/w16x16/k32 s1"),
        (64, false, 576, 3088, "32x32/w16x16/k32 s1"),
        (64, false, 1536, 576, "32x32/w16x16/k32 s4"),
        (64, false, 49152, 576, "32x64/w16x16/k64/db s8"),
        (128, true, 576, 576, "32x64/w16x16/k64/db s1"),
        (128, true, 576, 1536, "32x32/w16x16/k64/db s1"),
        (128, true, 576, 49152, "128x64/w32x32/k32/db/pf s1"),
        (128, true, 984, 576, "32x32/w16x16/k32 s8"),
        (128, true, 1536, 576, "64x64/w16x32/k32/db s8"),
        (128, true, 3088, 576, "32x32/w16x16/k32 s8"),
        (128, false, 576, 576, "32x32/w16x16/k64/db s1"),
        (128, false, 576, 984, "32x64/w16x16/k64/db s1"),
        (128, false, 576, 1536, "32x32/w16x16/k64/db s1"),
        (128, false, 576, 3088, "64x64/w32x32/k32/db/pf s1"),
        (128, false, 1536, 576, "64x64/w16x32/k32/db s6"),
        (128, false, 49152, 576, "128x128/w64x64/k32/db/pf s32"),
        (192, true, 576, 576, "32x64/w16x16/k64/db s1"),
        (192, true, 576, 1536, "128x64/w32x32/k32/db/pf s1"),
        (192, true, 576, 49152, "128x128/w64x64/k32/db/pf s1"),
        (192, true, 984, 576, "32x64/w16x32/k32/db s4"),
        (192, true, 1536, 576, "32x32/w16x16/k64/db s1"),
        (192, true, 3088, 576, "64x64/w16x32/k32/db s6"),
        (192, false, 576, 576, "32x32/w16x16/k64/db s1"),
        (192, false, 576, 984, "64x64/w32x32/k32/db s3"),
        (192, false, 576, 1536, "32x32/w16x16/k64/db s1"),
        (192, false, 576, 3088, "64x64/w16x32/k32/db s1"),
        (192, false, 1536, 576, "64x64/w32x32/k32/db s6"),
        (192, false, 49152, 576, "64x64/w32x32/k32/db/pf s8"),
        (256, true, 576, 576, "32x64/w16x32/k32/db s1"),
        (256, true, 576, 1536, "64x64/w32x32/k32/db/pf s1"),
        (256, true, 576, 49152, "128x128/w64x64/k32/db/pf s1"),
        (256, true, 984, 576, "32x64/w16x16/k64/db s1"),
        (256, true, 1536, 576, "32x64/w16x16/k64/db s1"),
        (256, true, 3088, 576, "64x64/w16x32/k32/db s5"),
        (256, false, 576, 576, "64x32/w32x16/k32/db s1"),
        (256, false, 576, 984, "64x64/w32x32/k32 s1"),
        (256, false, 576, 1536, "64x64/w32x32/k32/db/pf s1"),
        (256, false, 576, 3088, "64x64/w32x32/k32/db/pf s1"),
        (256, false, 1536, 576, "32x64/w16x32/k32/db s3"),
        (256, false, 49152, 576, "128x128/w64x64/k32/db/pf s16"),
        // SmolLM2-360M
        (64, true, 960, 960, "32x32/w16x16/k64/db s1"),
        (64, true, 960, 2560, "32x32/w16x16/k64/db s1"),
        (64, true, 960, 49152, "64x64/w32x32/k32/db s1"),
        (64, true, 1624, 960, "32x32/w16x16/k32 s5"),
        (64, true, 2560, 960, "32x64/w16x16/k64/db s5"),
        (64, false, 960, 960, "32x32/w16x16/k64/db s1"),
        (64, false, 960, 1624, "32x32/w16x16/k64/db s1"),
        (64, false, 960, 2560, "32x32/w16x16/k64/db s1"),
        (64, false, 2560, 960, "32x64/w16x16/k64/db s8"),
        (64, false, 49152, 960, "64x128/w32x64/k32/db s20"),
        (128, true, 960, 960, "32x64/w16x32/k32/db s4"),
        (128, true, 960, 2560, "32x64/w16x16/k64/db s1"),
        (128, true, 960, 49152, "128x64/w32x32/k32/db/pf s1"),
        (128, true, 1624, 960, "64x64/w16x32/k32/db s6"),
        (128, true, 2560, 960, "128x64/w32x32/k32/db/pf s10"),
        (128, false, 960, 960, "64x32/w32x16/k32/db s4"),
        (128, false, 960, 1624, "32x64/w16x16/k64/db s1"),
        (128, false, 960, 2560, "32x64/w16x16/k64/db s1"),
        (128, false, 2560, 960, "64x64/w32x32/k32/db/pf s8"),
        (128, false, 49152, 960, "64x64/w32x32/k32/db/pf s8"),
        (192, true, 960, 960, "32x64/w16x16/k64/db s1"),
        (192, true, 960, 2560, "64x64/w32x32/k32/db/pf s2"),
        (192, true, 960, 49152, "128x128/w64x64/k32/db/pf s1"),
        (192, true, 1624, 960, "64x64/w16x32/k32/db s4"),
        (192, true, 2560, 960, "64x64/w32x32/k32/db/pf s5"),
        (192, false, 960, 960, "32x32/w16x16/k64/db s1"),
        (192, false, 960, 1624, "64x64/w32x32/k32 s1"),
        (192, false, 960, 2560, "32x64/w16x32/k32/db s1"),
        (192, false, 2560, 960, "64x64/w32x32/k32/db/pf s7"),
        (192, false, 49152, 960, "64x64/w32x32/k32/db s7"),
        (256, true, 960, 960, "32x32/w16x16/k64/db s1"),
        (256, true, 960, 2560, "64x64/w16x32/k32/db s1"),
        (256, true, 960, 49152, "128x128/w64x64/k32/db/pf s1"),
        (256, true, 1624, 960, "64x64/w16x32/k32/db s4"),
        (256, true, 2560, 960, "64x64/w32x32/k32/db s4"),
        (256, false, 960, 960, "32x64/w16x16/k64/db s1"),
        (256, false, 960, 1624, "128x64/w32x32/k32/db/pf s3"),
        (256, false, 960, 2560, "32x64/w16x32/k32/db s1"),
        (256, false, 2560, 960, "128x64/w32x32/k32/db s5"),
        (256, false, 49152, 960, "128x128/w64x64/k32/db/pf s10"),
        // SmolLM2-1.7B
        (64, true, 2048, 2048, "32x32/w16x16/k64/db s1"),
        (64, true, 2048, 8192, "32x32/w16x16/k64/db s1"),
        (64, true, 2048, 49152, "64x64/w32x32/k32/db s1"),
        (64, true, 6168, 2048, "32x64/w16x16/k64/db s5"),
        (64, true, 8192, 2048, "32x32/w16x16/k64/db s5"),
        (64, false, 2048, 2048, "64x64/w16x32/k32/db s5"),
        (64, false, 2048, 6168, "64x64/w16x32/k32/db s1"),
        (64, false, 2048, 8192, "64x64/w32x32/k32/db/pf s1"),
        (64, false, 8192, 2048, "64x64/w16x32/k32/db s6"),
        (64, false, 49152, 2048, "64x128/w32x64/k32/db s10"),
        (128, true, 2048, 2048, "128x64/w32x32/k32/db s5"),
        (128, true, 2048, 8192, "128x64/w32x32/k32/db/pf s1"),
        (128, true, 2048, 49152, "128x64/w32x32/k32/db/pf s1"),
        (128, true, 6168, 2048, "128x128/w32x32/k32/pf s5"),
        (128, true, 8192, 2048, "128x64/w32x32/k32/db/pf s5"),
        (128, false, 2048, 2048, "128x64/w32x32/k32/db/pf s5"),
        (128, false, 2048, 6168, "64x64/w32x32/k32/db/pf s1"),
        (128, false, 2048, 8192, "128x64/w32x32/k32/db/pf s1"),
        (128, false, 8192, 2048, "128x64/w32x32/k32/db/pf s5"),
        (128, false, 49152, 2048, "128x128/w64x64/k32/db/pf s10"),
        (192, true, 2048, 2048, "64x64/w32x32/k32/db/pf s3"),
        (192, true, 2048, 8192, "64x64/w32x32/k32 s1"),
        (192, true, 2048, 49152, "128x128/w64x64/k32/db/pf s1"),
        (192, true, 6168, 2048, "64x64/w16x32/k32/db s5"),
        (192, true, 8192, 2048, "64x128/w32x64/k32/db s3"),
        (192, false, 2048, 2048, "64x64/w32x32/k32/db/pf s3"),
        (192, false, 2048, 6168, "64x128/w32x64/k32/db s1"),
        (192, false, 2048, 8192, "64x128/w32x64/k32/db s1"),
        (192, false, 8192, 2048, "64x128/w32x64/k32/db s5"),
        (192, false, 49152, 2048, "64x128/w32x64/k32/db s5"),
        (256, true, 2048, 2048, "64x64/w32x32/k32/db/pf s1"),
        (256, true, 2048, 8192, "128x128/w64x64/k32/db/pf s1"),
        (256, true, 2048, 49152, "128x128/w64x64/k32/db/pf s1"),
        (256, true, 6168, 2048, "64x64/w32x32/k32 s4"),
        (256, true, 8192, 2048, "128x128/w64x64/k32/db/pf s5"),
        (256, false, 2048, 2048, "64x64/w32x32/k32/db/pf s1"),
        (256, false, 2048, 6168, "64x128/w32x64/k32/db s1"),
        (256, false, 2048, 8192, "128x128/w64x64/k32/db/pf s1"),
        (256, false, 8192, 2048, "128x128/w64x64/k32/db/pf s5"),
        (256, false, 49152, 2048, "128x128/w64x64/k32/db/pf s5"),
    ];

    /// The configuration with the lowest geometric mean time over the sampled
    /// row counts of each bucket (weighted 2, 1, 1 for 360M at 1536, 1792 and
    /// 2048 rows) that `matmul_bench --tiles` found on an NVIDIA A10G, with K
    /// split kept only where it was 10% faster than the best unsplit one: `(bucket
    /// rows, a@bT, k, n, configuration)`. The buckets and their sampled rows
    /// are 320 (320), 384 (384), 512 (448, 512), 640 (640), 1024 (768, 1024),
    /// 1280 (1152, 1280, 1352, 1408) and 2048 (1536, 1792, 2048).
    const BENCH_MID: &[(u32, bool, u32, u32, &str)] = &[
        // SmolLM2-135M at 384 rows
        (384, false, 576, 984, "64x64/w32x32/k32 s1"),
        (384, true, 984, 576, "64x64/w16x32/k32/db s4"),
        (384, false, 576, 3088, "128x64/w32x32/k32/db/pf s1"),
        (384, true, 3088, 576, "64x64/w16x32/k32/db s4"),
        (384, true, 576, 576, "32x32/w16x16/k64/db s1"),
        (384, false, 576, 576, "32x32/w16x16/k64/db s1"),
        (384, true, 576, 1536, "64x64/w32x32/k32/db/pf s1"),
        (384, false, 1536, 576, "64x64/w32x32/k32/db/pf s4"),
        (384, true, 1536, 576, "32x32/w16x16/k64/db s1"),
        (384, false, 576, 1536, "128x64/w32x32/k32/db/pf s1"),
        (384, true, 576, 49152, "128x128/w64x64/k32/db/pf s1"),
        (384, false, 49152, 576, "64x128/w32x64/k32/db s8"),
        // SmolLM2-135M at 512 rows
        (512, false, 576, 984, "64x64/w32x32/k32 s1"),
        (512, true, 984, 576, "32x64/w16x16/k64/db s1"),
        (512, false, 576, 3088, "64x64/w16x32/k32/db s1"),
        (512, true, 3088, 576, "64x64/w16x32/k32/db s3"),
        (512, true, 576, 576, "32x64/w16x16/k64/db s1"),
        (512, false, 576, 576, "32x64/w16x16/k64/db s1"),
        (512, true, 576, 1536, "64x64/w32x32/k32/db/pf s1"),
        (512, false, 1536, 576, "64x64/w32x32/k32/db/pf s3"),
        (512, true, 1536, 576, "128x64/w32x32/k32/db/pf s4"),
        (512, false, 576, 1536, "64x64/w16x32/k32/db s1"),
        (512, true, 576, 49152, "128x128/w64x64/k32/db/pf s1"),
        (512, false, 49152, 576, "128x128/w64x64/k32/db/pf s8"),
        // SmolLM2-360M at 320 rows
        (320, false, 960, 1624, "64x64/w32x32/k32 s1"),
        (320, true, 1624, 960, "64x64/w16x32/k32/db s3"),
        (320, true, 960, 960, "32x64/w16x16/k64/db s1"),
        (320, false, 960, 960, "32x64/w16x32/k32/db s1"),
        (320, true, 960, 2560, "64x64/w32x32/k32/db/pf s1"),
        (320, false, 2560, 960, "64x64/w32x32/k32/db/pf s3"),
        (320, true, 2560, 960, "64x64/w16x32/k32/db s2"),
        (320, false, 960, 2560, "64x64/w32x32/k32/db s1"),
        (320, true, 960, 49152, "128x64/w32x32/k32/db/pf s1"),
        (320, false, 49152, 960, "64x128/w32x64/k32/db s6"),
        // SmolLM2-360M at 384 rows
        (384, false, 960, 1624, "64x64/w32x32/k32 s1"),
        (384, true, 1624, 960, "128x128/w32x32/k32/pf s3"),
        (384, true, 960, 960, "64x64/w32x32/k32/db s1"),
        (384, false, 960, 960, "64x64/w32x32/k32/db s1"),
        (384, true, 960, 2560, "64x64/w32x32/k32/db s1"),
        (384, false, 2560, 960, "64x64/w32x32/k32/db s3"),
        (384, true, 2560, 960, "128x64/w32x32/k32/db/pf s3"),
        (384, false, 960, 2560, "64x64/w32x32/k32/db/pf s1"),
        (384, true, 960, 49152, "128x128/w64x64/k32/db/pf s1"),
        (384, false, 49152, 960, "128x64/w64x32/k32/db/pf s7"),
        // SmolLM2-360M at 512 rows
        (512, false, 960, 1624, "64x64/w32x32/k32 s1"),
        (512, true, 1624, 960, "64x64/w16x32/k32/db s2"),
        (512, true, 960, 960, "64x64/w32x32/k32/db/pf s1"),
        (512, false, 960, 960, "32x64/w16x32/k32/db s1"),
        (512, true, 960, 2560, "128x64/w32x32/k32/db/pf s1"),
        (512, false, 2560, 960, "64x64/w32x32/k32/db/pf s2"),
        (512, true, 2560, 960, "64x64/w32x32/k32/db s2"),
        (512, false, 960, 2560, "128x64/w32x32/k32/db/pf s1"),
        (512, true, 960, 49152, "128x128/w64x64/k32/db/pf s1"),
        (512, false, 49152, 960, "128x128/w64x64/k32/db/pf s5"),
        // SmolLM2-360M at 640 rows
        (640, false, 960, 1624, "64x128/w32x64/k32/db s1"),
        (640, true, 1624, 960, "64x64/w16x32/k32/db s1"),
        (640, true, 960, 960, "64x64/w32x32/k32/db/pf s1"),
        (640, false, 960, 960, "64x64/w32x32/k32/db/pf s1"),
        (640, true, 960, 2560, "64x64/w32x32/k32 s1"),
        (640, false, 2560, 960, "64x64/w32x32/k32/db/pf s1"),
        (640, true, 2560, 960, "64x64/w32x32/k32/db/pf s1"),
        (640, false, 960, 2560, "64x128/w32x64/k32/db s1"),
        (640, true, 960, 49152, "128x128/w64x64/k32/db/pf s1"),
        (640, false, 49152, 960, "128x128/w64x64/k32/db/pf s4"),
        // SmolLM2-360M at 1024 rows
        (1024, false, 960, 1624, "64x128/w32x64/k32/db s1"),
        (1024, true, 1624, 960, "64x64/w16x32/k32/db s1"),
        (1024, true, 960, 960, "64x64/w32x32/k32/db/pf s1"),
        (1024, false, 960, 960, "64x64/w32x32/k32/db s1"),
        (1024, true, 960, 2560, "128x64/w32x32/k32/db/pf s1"),
        (1024, false, 2560, 960, "64x64/w32x32/k32/db/pf s1"),
        (1024, true, 2560, 960, "64x64/w32x32/k32/db/pf s1"),
        (1024, false, 960, 2560, "128x64/w32x32/k32/db/pf s1"),
        (1024, true, 960, 49152, "128x128/w64x64/k32/db/pf s1"),
        (1024, false, 49152, 960, "128x128/w64x64/k32/db/pf s5"),
        // SmolLM2-360M at 1280 rows
        (1280, false, 960, 1624, "64x64/w32x32/k32 s1"),
        (1280, true, 1624, 960, "64x64/w16x32/k32/db s1"),
        (1280, true, 960, 960, "64x64/w32x32/k32/db/pf s1"),
        (1280, false, 960, 960, "64x128/w32x64/k32/db s1"),
        (1280, true, 960, 2560, "128x64/w32x32/k32/db/pf s1"),
        (1280, false, 2560, 960, "64x128/w32x64/k32/db s1"),
        (1280, true, 2560, 960, "128x64/w32x32/k32/db/pf s1"),
        (1280, false, 960, 2560, "64x128/w32x64/k32/db s1"),
        (1280, true, 960, 49152, "128x128/w64x64/k32/db/pf s1"),
        (1280, false, 49152, 960, "128x128/w64x64/k32/db/pf s8"),
        // SmolLM2-360M at 2048 rows
        (2048, false, 960, 1624, "64x64/w32x32/k32 s1"),
        (2048, true, 1624, 960, "64x64/w16x32/k32/db s1"),
        (2048, true, 960, 960, "64x64/w32x32/k32/db/pf s1"),
        (2048, false, 960, 960, "64x64/w32x32/k32/db/pf s1"),
        (2048, true, 960, 2560, "128x64/w32x32/k32/db/pf s1"),
        (2048, false, 2560, 960, "64x64/w32x32/k32/db/pf s1"),
        (2048, true, 2560, 960, "64x64/w32x32/k32/db/pf s1"),
        (2048, false, 960, 2560, "64x128/w32x64/k32/db s1"),
        (2048, true, 960, 49152, "128x128/w64x64/k32/db/pf s1"),
        (2048, false, 49152, 960, "128x128/w64x64/k32/db/pf s5"),
        // SmolLM2-1.7B at 320 rows
        (320, false, 2048, 6168, "128x128/w64x64/k32/db/pf s1"),
        (320, true, 6168, 2048, "64x64/w16x32/k32/db s1"),
        (320, true, 2048, 2048, "64x64/w32x32/k32/db/pf s1"),
        (320, false, 2048, 2048, "64x64/w16x32/k32/db s1"),
        (320, true, 2048, 8192, "64x128/w32x64/k32/db s1"),
        (320, false, 8192, 2048, "64x128/w32x64/k32/db s3"),
        (320, true, 8192, 2048, "64x128/w32x64/k32/db s2"),
        (320, false, 2048, 8192, "64x128/w32x64/k32/db s1"),
        (320, true, 2048, 49152, "128x128/w64x64/k32/db/pf s1"),
        (320, false, 49152, 2048, "64x128/w32x64/k32/db s5"),
        // SmolLM2-1.7B at 384 rows
        (384, false, 2048, 6168, "128x128/w64x64/k32/db/pf s1"),
        (384, true, 6168, 2048, "128x128/w64x32/k32/pf s3"),
        (384, true, 2048, 2048, "64x64/w32x32/k32/db/pf s1"),
        (384, false, 2048, 2048, "64x64/w32x32/k32/db/pf s1"),
        (384, true, 2048, 8192, "128x64/w32x32/k32/db/pf s1"),
        (384, false, 8192, 2048, "128x128/w64x64/k32/db/pf s3"),
        (384, true, 8192, 2048, "128x128/w64x64/k32/db/pf s3"),
        (384, false, 2048, 8192, "128x64/w32x32/k32/db/pf s1"),
        (384, true, 2048, 49152, "128x128/w64x64/k32/db/pf s1"),
        (384, false, 49152, 2048, "128x128/w64x64/k32/db/pf s8"),
        // SmolLM2-1.7B at 512 rows
        (512, false, 2048, 6168, "64x128/w32x64/k32/db s1"),
        (512, true, 6168, 2048, "64x64/w32x32/k32 s2"),
        (512, true, 2048, 2048, "128x64/w32x32/k32/db/pf s1"),
        (512, false, 2048, 2048, "64x64/w32x32/k32/db/pf s1"),
        (512, true, 2048, 8192, "128x128/w64x64/k32/db/pf s1"),
        (512, false, 8192, 2048, "64x128/w32x64/k32/db s5"),
        (512, true, 8192, 2048, "128x128/w64x64/k32/db/pf s5"),
        (512, false, 2048, 8192, "128x128/w64x64/k32/db/pf s1"),
        (512, true, 2048, 49152, "128x128/w64x64/k32/db/pf s1"),
        (512, false, 49152, 2048, "128x128/w64x64/k32/db/pf s5"),
        // SmolLM2-1.7B at 640 rows
        (640, false, 2048, 6168, "64x128/w32x64/k32/db s1"),
        (640, true, 6168, 2048, "128x128/w32x32/k32/pf s1"),
        (640, true, 2048, 2048, "128x64/w32x32/k32/db/pf s1"),
        (640, false, 2048, 2048, "128x64/w64x32/k32/db/pf s1"),
        (640, true, 2048, 8192, "128x128/w64x64/k32/db/pf s1"),
        (640, false, 8192, 2048, "64x128/w32x64/k32/db s1"),
        (640, true, 8192, 2048, "128x128/w64x64/k32/db/pf s2"),
        (640, false, 2048, 8192, "128x128/w64x64/k32/db/pf s1"),
        (640, true, 2048, 49152, "128x128/w64x64/k32/db/pf s1"),
        (640, false, 49152, 2048, "128x128/w64x64/k32/db/pf s2"),
        // SmolLM2-1.7B at 1024 rows
        (1024, false, 2048, 6168, "128x128/w64x64/k32/db/pf s1"),
        (1024, true, 6168, 2048, "64x64/w32x32/k32 s1"),
        (1024, true, 2048, 2048, "64x64/w32x32/k32/db/pf s1"),
        (1024, false, 2048, 2048, "64x64/w32x32/k32/db/pf s1"),
        (1024, true, 2048, 8192, "128x128/w64x64/k32/db/pf s1"),
        (1024, false, 8192, 2048, "128x128/w64x64/k32/db/pf s3"),
        (1024, true, 8192, 2048, "128x128/w64x64/k32/db/pf s5"),
        (1024, false, 2048, 8192, "128x128/w64x64/k32/db/pf s1"),
        (1024, true, 2048, 49152, "128x128/w64x64/k32/db/pf s1"),
        (1024, false, 49152, 2048, "128x128/w64x64/k32/db/pf s5"),
        // SmolLM2-1.7B at 1280 rows
        (1280, false, 2048, 6168, "128x128/w64x64/k32/db/pf s1"),
        (1280, true, 6168, 2048, "64x64/w16x32/k32/db s1"),
        (1280, true, 2048, 2048, "128x64/w32x32/k32/db/pf s1"),
        (1280, false, 2048, 2048, "64x128/w32x64/k32/db s1"),
        (1280, true, 2048, 8192, "128x128/w64x64/k32/db/pf s1"),
        (1280, false, 8192, 2048, "128x128/w64x64/k32/db/pf s1"),
        (1280, true, 8192, 2048, "128x128/w64x64/k32/db/pf s1"),
        (1280, false, 2048, 8192, "128x128/w64x64/k32/db/pf s1"),
        (1280, true, 2048, 49152, "128x128/w64x64/k32/db/pf s1"),
        (1280, false, 49152, 2048, "128x128/w64x64/k32/db/pf s5"),
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
        assert_eq!(
            BENCH.len()
                + BENCH_1024.len()
                + BENCH_MID.len()
                + BENCH_LARGE.len()
                + BENCH_SMALL.len(),
            TUNED.len()
        );
        let tables = BENCH_1024
            .iter()
            .chain(BENCH_MID)
            .chain(BENCH_LARGE)
            .chain(BENCH_SMALL);
        for &(m, transposed_b, k, n, winner) in tables {
            let got = plan((m, k, n), transposed_b, (160, true), all).expect("a tile fits");
            let name = format!("{} s{}", got.tile.name(), got.split);
            assert_eq!(name, winner, "m {m} k {k} n {n} transposed {transposed_b}");
        }
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
        // A padded batch has fewer rows than 256, 1024 or 2048: it runs the
        // row count's bucket, which is the winner of 576 x 576 at 64, 128,
        // 192 or 256 rows up to 256 rows, the 1024-row winner up to 1024 and
        // the 2048-row winner above that, up to 2048.
        for (m, want) in [
            (33, (CoopTile::T32X32_W16X16_K64_DB, 5)),
            (64, (CoopTile::T32X32_W16X16_K64_DB, 5)),
            (65, (CoopTile::T32X64_W16X16_K64_DB, 1)),
            (106, (CoopTile::T32X64_W16X16_K64_DB, 1)),
            (130, (CoopTile::T32X64_W16X16_K64_DB, 1)),
            (192, (CoopTile::T32X64_W16X16_K64_DB, 1)),
            (193, (CoopTile::T32X64_W16X32_K32_DB, 1)),
            (256, (CoopTile::T32X64_W16X32_K32_DB, 1)),
            (257, (CoopTile::T128X64_W32X32_K32_PF, 1)),
            (1000, (CoopTile::T128X64_W32X32_K32_PF, 1)),
            (1024, (CoopTile::T128X64_W32X32_K32_PF, 1)),
            (1025, (CoopTile::ALL[6], 1)),
            (1500, (CoopTile::ALL[6], 1)),
            (2048, (CoopTile::ALL[6], 1)),
        ] {
            let got = plan((m, 576, 576), true, (160, true), all).expect("a tile fits");
            assert_eq!((got.tile, got.split), want, "m {m}");
        }
        // A product with no 1024-row row runs its 2048-row one up to 1024
        // rows: 135M's LoRA-only 576 x 192 product.
        let got = plan((900, 576, 192), true, (160, true), all).expect("a tile fits");
        assert_eq!((got.tile, got.split), (CoopTile::ALL[2], 1));
        // 1.7B at up to 256 rows has a row per bucket from the small-row
        // sweep: the wide products take the larger tiles as the rows grow.
        for (m, a_bt, k, n, want) in [
            (64, true, 2048, 2048, (CoopTile::T32X32_W16X16_K64_DB, 1)),
            (128, true, 2048, 2048, (CoopTile::T128X64_W32X32_K32_DB, 5)),
            (64, true, 8192, 2048, (CoopTile::T32X32_W16X16_K64_DB, 5)),
            (
                256,
                false,
                2048,
                8192,
                (CoopTile::T128X128_W64X64_K32_PF, 1),
            ),
        ] {
            let got = plan((m, k, n), a_bt, (160, true), all).expect("a tile fits");
            assert_eq!((got.tile, got.split), want, "m {m} k {k} n {n}");
        }
        // A 1.7B product with no row, and a product on a device without the
        // table, run the router's own rule.
        let got = plan((128, 2048, 3000), false, (160, true), all).expect("a tile fits");
        assert_eq!((got.tile, got.split), (CoopTile::ALL[12], 1));
        let got = plan((128, 2048, 8192), false, (160, false), all).expect("a tile fits");
        assert_eq!((got.tile, got.split), (CoopTile::ALL[12], 1));
        // A row whose tile does not fit the device falls to the step rule.
        let small_only = |t: CoopTile| t.bm * t.bn <= 64 * 64;
        let got = plan((256, 2048, 8192), false, (160, true), small_only).expect("a tile fits");
        assert_eq!(got.tile, STEP_TILES[1]);
        // More rows than the table has run the router's own rule.
        let got = plan((4096, 576, 576), true, (160, true), all);
        assert_eq!(got.map(|c| c.tile), Some(CoopTile::ALL[1]));
        // Without the table (a device it was not measured on) the router's
        // own rule applies.
        let got = plan((2048, 2048, 8192), true, (160, false), all);
        assert_eq!(got.map(|c| c.tile), Some(CoopTile::ALL[1]));
    }

    #[test]
    fn rows_above_1024_split_at_1408_for_360m_and_1_7b() {
        let all = |_: CoopTile| true;
        let run = |m: u32, transposed_b: bool, k: u32, n: u32| {
            let got = plan((m, k, n), transposed_b, (160, true), all).expect("a tile fits");
            (got.tile, got.split)
        };
        // 1.7B down projection dX: the 1280-row row up to 1408 rows (the rows
        // are rounded up to a multiple of 16, so 1393 is 1408 and 1409 is
        // 1424), the 2048-row row above.
        let low = (CoopTile::T128X128_W64X64_K32_PF, 1);
        let high = (CoopTile::T64X128_W32X64_K32_DB, 1);
        for m in [1025, 1040, 1152, 1280, 1352, 1392, 1393, 1408] {
            assert_eq!(run(m, false, 8192, 2048), low, "m {m}");
        }
        for m in [1409, 1424, 1536, 1792, 2048] {
            assert_eq!(run(m, false, 8192, 2048), high, "m {m}");
        }
        // 1.7B head dX: the K split is 1 in the 1280-row row and 5 in the
        // 2048-row one.
        let head = |m| run(m, false, 49152, 2048);
        assert_eq!(head(1408), (CoopTile::T128X128_W64X64_K32_PF, 5));
        assert_eq!(head(1409), (CoopTile::T128X128_W64X64_K32_PF, 5));
        // 1.7B fused QKV dX and 360M QKV dX, which had no 2048-row row.
        assert_eq!(
            run(1300, true, 6168, 2048),
            (CoopTile::T64X64_W16X32_K32_DB, 1)
        );
        assert_eq!(
            run(1500, true, 6168, 2048),
            (CoopTile::T64X64_W32X32_K32, 1)
        );
        assert_eq!(
            run(1300, true, 1624, 960),
            (CoopTile::T64X64_W16X32_K32_DB, 1)
        );
        assert_eq!(
            run(1500, true, 1624, 960),
            (CoopTile::T64X64_W16X32_K32_DB, 1)
        );
        // 360M's narrow products have no 1280-row row and run the 2048-row one
        // from 1025 rows, as does all of 135M.
        let narrow = (CoopTile::T64X64_W16X32_K32_DB, 1);
        assert_eq!(run(1025, true, 960, 320), narrow);
        assert_eq!(run(1500, true, 960, 320), narrow);
        assert_eq!(run(1300, true, 576, 576), (CoopTile::ALL[6], 1));
        // Up to 1024 rows the 1024-row rows still run, and above 2048 rows
        // the router's own rule.
        assert_eq!(
            run(1024, true, 2048, 2048),
            (CoopTile::T64X64_W32X32_K32_PF, 1)
        );
        assert_eq!(run(2064, true, 2048, 2048), (CoopTile::ALL[1], 1));
    }

    #[test]
    fn rows_between_the_buckets_run_the_bucket_above() {
        let all = |_: CoopTile| true;
        for (m, bucket) in [(1, 64), (257, 320), (321, 384), (385, 512), (2048, 2048)] {
            assert_eq!(bucket_of(m).map(|b| b.0), Some(bucket), "m {m}");
        }
        assert_eq!(bucket_of(2049), None);
        for &(bucket, transposed_b, k, n, winner) in BENCH_MID {
            let rows: &[u32] = match bucket {
                320 => &[257, 300, 320],
                384 => &[321, 350, 384],
                512 => &[385, 449, 512],
                640 => &[513, 600, 640],
                1024 => &[641, 900, 1024],
                1280 => &[1025, 1100, 1352, 1408],
                _ => &[1409, 1500, 2048],
            };
            for &m in rows {
                let got = plan((m, k, n), transposed_b, (160, true), all).expect("a tile fits");
                let name = format!("{} s{}", got.tile.name(), got.split);
                assert_eq!(
                    name, winner,
                    "rows {m} k {k} n {n} transposed {transposed_b}"
                );
            }
        }
    }

    #[test]
    fn padded_row_counts_run_their_buckets_row() {
        let all = |_: CoopTile| true;
        // Each row count of a bucket (rounded up to a multiple of 16) runs
        // what the bucket's own row count runs, split included.
        for &(bucket, transposed_b, k, n, _) in BENCH_SMALL {
            let rows: &[u32] = match bucket {
                64 => &[17, 33, 50, 64],
                128 => &[65, 100, 128],
                192 => &[129, 160, 192],
                _ => &[193, 200, 256],
            };
            let want = plan((bucket, k, n), transposed_b, (160, true), all).expect("a tile fits");
            for &m in rows {
                let got = plan((m, k, n), transposed_b, (160, true), all).expect("a tile fits");
                assert_eq!(
                    (got.tile, got.split),
                    (want.tile, want.split),
                    "bucket {bucket} rows {m} k {k} n {n}"
                );
            }
        }
        // The four narrow 256-row rows of the older run serve every bucket:
        // 135M LoRA-only 576 x 192 at 100 rows.
        let got = plan((100, 576, 192), true, (160, true), all).expect("a tile fits");
        assert_eq!((got.tile, got.split), (CoopTile::ALL[12], 1));
        // 135M down projection at 200 rows takes the 256-row row.
        let got = plan((200, 1536, 576), false, (160, true), all).expect("a tile fits");
        assert_eq!((got.tile, got.split), (CoopTile::T32X64_W16X32_K32_DB, 3));
        // The row count is rounded up to a multiple of 16 for the lookup, so
        // 2049 rows (2064) are past the table and 2048 are not.
        let past = plan((2049, 576, 576), true, (160, true), all).expect("a tile fits");
        assert_eq!(past.tile, CoopTile::ALL[1]);
    }

    #[test]
    fn tile_lists_hold_the_tiles_their_old_indices_named() {
        let all = CoopTile::ALL;
        assert_eq!(LARGE_M_TILES, [all[1], all[2]]);
        assert_eq!(SMALL_M_TILES, [all[12], all[11]]);
        assert_eq!(STEP_TILES, [all[3], all[4]]);
        // Every row of the table names a tile the generator has.
        for row in TUNED {
            let known = all.contains(&row.tile) || CoopTile::PIPELINED.contains(&row.tile);
            assert!(known, "{:?}", row.tile);
        }
    }

    type CachePick = (bool, LoaderKind, EpilogueKind, u32, u32, &'static str, u32);

    /// The eight 1.7B products of 64 rows or fewer, as the cache of the run
    /// that won the step keeps them: `(a @ bT, loader, epilogue, k, n)` and
    /// the cache line's tile name and K slices.
    fn cache_picks() -> Vec<CachePick> {
        let plain = LoaderKind::Plain;
        let ab = EpilogueKind::AlphaBeta;
        let split4 = EpilogueKind::ColumnSplit {
            segments: 4,
            bias: false,
        };
        let sources = LoaderKind::Sources { count: 4 };
        vec![
            (false, plain, ab, 2048, 2048, "32x64/w16x16/k64/db", 2),
            (false, plain, ab, 8192, 2048, "32x64/w16x16/k64/db", 2),
            (false, plain, ab, 2048, 8192, "64x64/w16x32/k32/db", 1),
            (false, plain, split4, 2048, 6168, "64x64/w16x32/k32/db", 2),
            (true, plain, ab, 2048, 2048, "32x64/w16x16/k64/db", 2),
            (true, plain, ab, 8192, 2048, "64x64/w32x32/k32/db/pf", 8),
            (true, plain, ab, 2048, 8192, "64x64/w32x32/k32/db/pf", 2),
            (true, sources, ab, 6168, 2048, "32x64/w16x16/k64/db", 2),
        ]
    }

    /// What `Coop::autotuned` starts from without a pin: `plan` for a plain
    /// product and `plan_fused` for a fused one.
    fn routed(
        (m, k, n): (u32, u32, u32),
        transposed_b: bool,
        kinds: (LoaderKind, EpilogueKind),
    ) -> super::CoopConfig {
        let all = |_: CoopTile| true;
        let base = if kinds == (LoaderKind::Plain, EpilogueKind::AlphaBeta) {
            plan((m, k, n), transposed_b, (160, true), all)
        } else {
            plan_fused((m, k, n), transposed_b, (160, true), all)
        };
        base.expect("a tile fits")
    }

    #[test]
    fn the_pins_return_the_picks_of_the_cache_for_up_to_64_rows() {
        let all = |_: CoopTile| true;
        let picks = cache_picks();
        assert_eq!(picks.len(), M64_PINS.len());
        for (transposed_b, loader, epilogue, k, n, name, split) in picks {
            for m in [1, 16, 33, 64] {
                let kinds = (loader, epilogue);
                let base = routed((m, k, n), transposed_b, kinds);
                let got = pin_or(base, (m, k, n), transposed_b, kinds, true, all);
                assert_eq!(
                    (got.tile.name().as_str(), got.split),
                    (name, split),
                    "m {m} k {k} n {n} bT {transposed_b}"
                );
            }
        }
    }

    #[test]
    fn a_pin_stands_down_off_its_shape_off_a_discrete_gpu_and_off_a_tile_that_does_not_fit() {
        let all = |_: CoopTile| true;
        for (transposed_b, loader, epilogue, k, n, name, _) in cache_picks() {
            let kinds = (loader, epilogue);
            let at = |m: u32, use_tuned: bool, fits: &dyn Fn(CoopTile) -> bool| {
                let base = routed((m, k, n), transposed_b, kinds);
                let got = pin_or(base, (m, k, n), transposed_b, kinds, use_tuned, fits);
                (base, got)
            };
            // Past 64 rows: the router's pick.
            for m in [65, 128, 256, 1024] {
                let (base, got) = at(m, true, &all);
                assert_eq!(got, base, "m {m} k {k} n {n}");
            }
            // Where the tuned rows do not apply (Apple silicon, integrated GPUs).
            let (base, got) = at(64, false, &all);
            assert_eq!(got, base, "untuned k {k} n {n}");
            // A device the pinned tile does not fit.
            let not_pinned = |t: CoopTile| t.name() != name;
            let (base, got) = at(64, true, &not_pinned);
            assert_eq!(got, base, "no fit k {k} n {n}");
            // The other layout of the same weight shape is its own product: it
            // keeps the router's pick unless the pins name it too.
            let other = !transposed_b;
            let base = routed((64, k, n), other, kinds);
            let got = pin_or(base, (64, k, n), other, kinds, true, all);
            let named = M64_PINS.iter().any(|p| {
                (p.transposed_b, p.loader, p.epilogue, p.k, p.n) == (other, loader, epilogue, k, n)
            });
            if !named {
                assert_eq!(got, base, "other layout k {k} n {n}");
            }
        }
        // The same weight with a loader or epilogue the pins do not name.
        let kinds = (LoaderKind::Sources { count: 4 }, EpilogueKind::AlphaBeta);
        let base = routed((64, 2048, 2048), false, kinds);
        let got = pin_or(base, (64, 2048, 2048), false, kinds, true, all);
        assert_eq!(got, base);
    }

    #[test]
    fn pinned_names_the_row_of_a_pinned_product_and_nothing_else() {
        let all = |_: CoopTile| true;
        let plain = (LoaderKind::Plain, EpilogueKind::AlphaBeta);
        let want = CoopConfig {
            tile: CoopTile::T32X64_W16X16_K64_DB,
            split: 2,
        };
        assert_eq!(
            pinned((64, 2048, 2048), false, plain, true, all),
            Some(want)
        );
        assert_eq!(
            pinned((17, 2048, 2048), false, plain, true, all),
            Some(want)
        );
        assert_eq!(pinned((65, 2048, 2048), false, plain, true, all), None);
        assert_eq!(pinned((64, 2048, 2048), false, plain, false, all), None);
        assert_eq!(pinned((64, 576, 576), true, plain, true, all), None);
        let none = |_: CoopTile| false;
        assert_eq!(pinned((64, 2048, 2048), false, plain, true, none), None);
    }

    #[test]
    fn the_135m_and_360m_products_of_64_rows_keep_their_rows() {
        let all = |_: CoopTile| true;
        let plain = (LoaderKind::Plain, EpilogueKind::AlphaBeta);
        // 135M: the TUNED row of 64 rows.
        let base = routed((64, 576, 576), true, plain);
        assert_eq!((base.tile, base.split), (CoopTile::T32X32_W16X16_K64_DB, 5));
        let got = pin_or(base, (64, 576, 576), true, plain, true, all);
        assert_eq!(got, base);
        // 135M and 360M products of 64 rows and fewer, plain and fused.
        let shapes = [
            (576, 576),
            (576, 1536),
            (1536, 576),
            (576, 192),
            (960, 960),
            (960, 320),
            (960, 2560),
            (2560, 960),
            (960, 49152),
        ];
        let fused = (
            LoaderKind::Plain,
            EpilogueKind::ColumnSplit {
                segments: 4,
                bias: false,
            },
        );
        for (k, n) in shapes {
            for m in [16, 40, 64] {
                for transposed_b in [false, true] {
                    for kinds in [plain, fused] {
                        let base = routed((m, k, n), transposed_b, kinds);
                        let got = pin_or(base, (m, k, n), transposed_b, kinds, true, all);
                        assert_eq!(got, base, "m {m} k {k} n {n} bT {transposed_b}");
                    }
                }
            }
        }
    }

    #[test]
    fn the_pins_come_off_with_a_zero_and_stay_on_otherwise() {
        assert!(!pins_enabled(Some("0")));
        assert!(!pins_enabled(Some(" 0 ")));
        assert!(pins_enabled(None));
        assert!(pins_enabled(Some("1")));
        assert!(pins_enabled(Some(" 1 ")));
        assert!(pins_enabled(Some("")));
    }

    #[test]
    fn the_pins_are_distinct_products_with_known_tiles() {
        let known = |t: CoopTile| CoopTile::ALL.contains(&t) || CoopTile::PIPELINED.contains(&t);
        for (i, p) in M64_PINS.iter().enumerate() {
            assert!(known(p.tile), "{:?}", p.tile);
            assert!(p.split >= 1, "{i}");
            for q in &M64_PINS[i + 1..] {
                let same = (p.transposed_b, p.loader, p.epilogue, p.k, p.n)
                    == (q.transposed_b, q.loader, q.epilogue, q.k, q.n);
                assert!(!same, "pin {i} is repeated");
            }
        }
    }
}
