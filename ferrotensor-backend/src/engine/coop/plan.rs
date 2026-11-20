//! Which tile, and how many slices of K, the router runs a product with.

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
/// fine-tuning SmolLM2-135M and 360M at 256 and 2048 rows, and of 1.7B at
/// 2048 rows, from one `matmul-tune` run on an NVIDIA A10G. A K split stands
/// only where it was at least 10% faster than the best unsplit configuration,
/// since it costs a second pass and a partial-sum buffer that the bench's
/// isolated calls charge only in part. The 1.7B products at 256 rows are not
/// here: see [`STEP_RULE`].
const TUNED: &[Tuned] = &[
    // SmolLM2-135M
    Tuned::new(256, true, 576, 576, CoopTile::T32X64_W16X16_K64_DB, 1),
    Tuned::new(256, false, 576, 576, CoopTile::T32X64_W16X16_K64_DB, 1),
    Tuned::new(256, true, 576, 192, CoopTile::T32X64_W16X16_K64_DB, 1),
    Tuned::new(256, false, 192, 576, CoopTile::T32X64_W16X16_K64_DB, 1),
    Tuned::new(256, true, 576, 1536, CoopTile::T32X32_W16X16_K32, 1),
    Tuned::new(256, false, 1536, 576, CoopTile::T64X64_W16X32_K32_DB, 6),
    Tuned::new(256, true, 1536, 576, CoopTile::T64X64_W16X32_K32_DB, 6),
    Tuned::new(256, false, 576, 1536, CoopTile::T32X32_W16X16_K32, 1),
    Tuned::new(256, true, 576, 49152, CoopTile::T128X64_W32X32_K32_DB, 1),
    Tuned::new(256, false, 49152, 576, CoopTile::T64X64_W16X32_K32_DB, 6),
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
    Tuned::new(256, true, 960, 960, CoopTile::T32X64_W16X16_K64_DB, 1),
    Tuned::new(256, false, 960, 960, CoopTile::T32X64_W16X16_K64_DB, 1),
    Tuned::new(256, true, 960, 320, CoopTile::T32X64_W16X16_K64_DB, 4),
    Tuned::new(256, false, 320, 960, CoopTile::T32X64_W16X16_K64_DB, 1),
    Tuned::new(256, true, 960, 2560, CoopTile::T64X64_W16X32_K32_DB, 1),
    Tuned::new(256, false, 2560, 960, CoopTile::T64X64_W16X32_K32_DB, 4),
    Tuned::new(256, true, 2560, 960, CoopTile::T64X64_W16X32_K32_DB, 4),
    Tuned::new(256, false, 960, 2560, CoopTile::T64X64_W16X32_K32_DB, 1),
    Tuned::new(256, true, 960, 49152, CoopTile::T128X64_W64X32_K32, 1),
    Tuned::new(256, false, 49152, 960, CoopTile::T64X64_W32X32_K32, 8),
    Tuned::new(2048, true, 960, 960, CoopTile::T64X64_W32X32_K32, 1),
    Tuned::new(2048, false, 960, 960, CoopTile::T64X64_W32X32_K32, 1),
    Tuned::new(2048, true, 960, 320, CoopTile::T64X64_W16X32_K32_DB, 1),
    Tuned::new(2048, false, 320, 960, CoopTile::T64X64_W32X32_K32, 1),
    Tuned::new(2048, true, 960, 2560, CoopTile::T64X64_W32X32_K32, 1),
    Tuned::new(2048, false, 2560, 960, CoopTile::T64X64_W32X32_K32, 1),
    Tuned::new(2048, true, 2560, 960, CoopTile::T64X64_W32X32_K32, 1),
    Tuned::new(2048, false, 960, 2560, CoopTile::T64X64_W32X32_K32, 1),
    Tuned::new(2048, true, 960, 49152, CoopTile::T128X64_W32X32_K32_DB, 1),
    Tuned::new(2048, false, 49152, 960, CoopTile::T64X64_W32X32_K32, 1),
    // SmolLM2-1.7B
    Tuned::new(2048, true, 2048, 2048, CoopTile::T64X64_W16X32_K32_DB, 1),
    Tuned::new(2048, false, 2048, 2048, CoopTile::T64X64_W16X32_K32_DB, 1),
    Tuned::new(2048, true, 2048, 8192, CoopTile::T64X64_W32X32_K32, 1),
    Tuned::new(2048, false, 8192, 2048, CoopTile::T64X64_W16X32_K32_DB, 1),
    Tuned::new(2048, true, 8192, 2048, CoopTile::T64X64_W16X32_K32_DB, 1),
    Tuned::new(2048, false, 2048, 8192, CoopTile::T64X64_W32X32_K32, 1),
    Tuned::new(2048, true, 2048, 49152, CoopTile::T128X64_W64X32_K32, 1),
    Tuned::new(2048, false, 49152, 2048, CoopTile::T64X64_W16X32_K32_DB, 1),
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
        tile: row.tile,
        split: row.split,
    })
}

/// The 1.7B base-weight products of up to 256 rows, as `(a@bT, k, n)`, whose
/// `matmul-tune` winners lose in a training step. The bench winners (the
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
        plan, plan_fused, split_k, CoopTile, LARGE_M_TILES, SMALL_M_TILES, STEP_TILES, TUNED,
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
        // a training step: the double-buffered 64x64 tile runs, split until
        // the grid fills the GPU.
        for (m, a_bt, k, n, want) in [
            (128, false, 2048, 8192, (4, 1)),
            (128, true, 2048, 2048, (4, 3)),
            (64, true, 8192, 2048, (4, 5)),
            (256, false, 2048, 8192, (3, 1)),
        ] {
            let got = plan((m, k, n), a_bt, (160, true), all).expect("a tile fits");
            assert_eq!(
                (got.tile, got.split),
                (CoopTile::ALL[want.0], want.1),
                "m {m} k {k} n {n}"
            );
        }
        // Another 1.7B product, and the same product on a device without the
        // table, run the router's own rule.
        let got = plan((128, 2048, 49152), false, (160, true), all).expect("a tile fits");
        assert_eq!((got.tile, got.split), (CoopTile::ALL[12], 1));
        let got = plan((128, 2048, 8192), false, (160, false), all).expect("a tile fits");
        assert_eq!((got.tile, got.split), (CoopTile::ALL[12], 1));
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

    #[test]
    fn tile_lists_hold_the_tiles_their_old_indices_named() {
        let all = CoopTile::ALL;
        assert_eq!(LARGE_M_TILES, [all[1], all[2]]);
        assert_eq!(SMALL_M_TILES, [all[12], all[11]]);
        assert_eq!(STEP_TILES, [all[3], all[4]]);
        // Every row of the table names a tile the generator has.
        for row in TUNED {
            assert!(all.contains(&row.tile), "{:?}", row.tile);
        }
    }
}
