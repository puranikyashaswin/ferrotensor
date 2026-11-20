//! The tiles of the cooperative-matrix kernel and the configurations built
//! from them.

/// f16 elements of padding after each staged row, against bank conflicts.
pub(super) const PAD: u32 = 8;
/// Threads per subgroup the kernel is written for.
pub(super) const SUBGROUP: u32 = 32;

/// A workgroup's output tile `bm x bn`, split into one `wm x wn` block per
/// subgroup, stepping through K `bk` at a time with one staging buffer or
/// two (`double`: the next step's loads overlap this step's multiplies, and
/// each step needs one barrier instead of two). With `prefetch` the loads of
/// the next step go to registers before this step's multiplies and reach
/// workgroup memory after them, so the multiplies hide the global-memory
/// latency whatever the compiler does with the order of the code.
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
    /// Register prefetch of the next K step (see [`CoopTile`]).
    pub prefetch: bool,
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
            prefetch: false,
        }
    }

    /// As [`new`](Self::new), with register prefetch.
    const fn prefetching(bm: u32, bn: u32, (wm, wn): (u32, u32), bk: u32, double: bool) -> Self {
        Self {
            prefetch: true,
            ..Self::new(bm, bn, (wm, wn), bk, double)
        }
    }

    /// The 128x64 tile of 64x32 per subgroup, K 32, one staging buffer.
    pub(super) const T128X64_W64X32_K32: Self = Self::new(128, 64, (64, 32), 32, false);

    /// The 64x64 tile of 32x32 per subgroup, K 32, one staging buffer.
    pub(super) const T64X64_W32X32_K32: Self = Self::new(64, 64, (32, 32), 32, false);

    /// The 32x32 tile of 16x16 per subgroup, K 32, one staging buffer.
    pub(super) const T32X32_W16X16_K32: Self = Self::new(32, 32, (16, 16), 32, false);

    /// The 128x64 tile of 64x32 per subgroup, K 32, two staging buffers.
    pub(super) const T128X64_W64X32_K32_DB: Self = Self::new(128, 64, (64, 32), 32, true);

    /// The 64x64 tile of 32x32 per subgroup, K 32, two staging buffers.
    pub(super) const T64X64_W32X32_K32_DB: Self = Self::new(64, 64, (32, 32), 32, true);

    /// The 128x64 tile of 32x32 per subgroup, K 32, two staging buffers.
    pub(super) const T128X64_W32X32_K32_DB: Self = Self::new(128, 64, (32, 32), 32, true);

    /// The 64x64 tile of 16x32 per subgroup, K 32, two staging buffers.
    pub(super) const T64X64_W16X32_K32_DB: Self = Self::new(64, 64, (16, 32), 32, true);

    /// The 32x64 tile of 16x16 per subgroup, K 64, two staging buffers.
    pub(super) const T32X64_W16X16_K64_DB: Self = Self::new(32, 64, (16, 16), 64, true);

    /// The 64x64 tile of 32x32 per subgroup, K 32, two staging buffers, register prefetch.
    pub(super) const T64X64_W32X32_K32_PF: Self = Self::prefetching(64, 64, (32, 32), 32, true);

    /// The 128x64 tile of 32x32 per subgroup, K 32, two staging buffers, register prefetch.
    pub(super) const T128X64_W32X32_K32_PF: Self = Self::prefetching(128, 64, (32, 32), 32, true);

    /// The 128x64 tile of 64x32 per subgroup, K 32, two staging buffers, register prefetch.
    pub(super) const T128X64_W64X32_K32_PF: Self = Self::prefetching(128, 64, (64, 32), 32, true);

    /// The 128x128 tile of 64x32 per subgroup (8 subgroups), K 32, two staging
    /// buffers, register prefetch.
    pub(super) const T128X128_W64X32_K32_PF: Self = Self::prefetching(128, 128, (64, 32), 32, true);

    /// The 128x128 tile of 64x64 per subgroup (4 subgroups), K 32, two staging
    /// buffers, register prefetch.
    pub(super) const T128X128_W64X64_K32_PF: Self = Self::prefetching(128, 128, (64, 64), 32, true);

    /// The 128x256 tile of 64x64 per subgroup (8 subgroups), K 32, one staging
    /// buffer, register prefetch.
    pub(super) const T128X256_W64X64_K32_PF1: Self =
        Self::prefetching(128, 256, (64, 64), 32, false);

    /// The 256x128 tile of 64x64 per subgroup (8 subgroups), K 32, one staging
    /// buffer, register prefetch.
    pub(super) const T256X128_W64X64_K32_PF1: Self =
        Self::prefetching(256, 128, (64, 64), 32, false);

    /// The 128x128 tile of 32x32 per subgroup (16 subgroups), K 32, one
    /// staging buffer, register prefetch.
    pub(super) const T128X128_W32X32_K32_PF1: Self =
        Self::prefetching(128, 128, (32, 32), 32, false);

    /// The 128x128 tile of 64x32 per subgroup (8 subgroups), K 64, one staging
    /// buffer, register prefetch.
    pub(super) const T128X128_W64X32_K64_PF1: Self =
        Self::prefetching(128, 128, (64, 32), 64, false);

    /// The 128x128 tile of 64x32 per subgroup (8 subgroups), K 32, one staging
    /// buffer, register prefetch.
    pub(super) const T128X128_W64X32_K32_PF1: Self =
        Self::prefetching(128, 128, (64, 32), 32, false);

    /// The tiles with register prefetch, for the tests, the tuning bench and
    /// the router. They are kept apart from [`ALL`](Self::ALL), whose order
    /// the older tests and table rows count on.
    pub const PIPELINED: [Self; 10] = [
        Self::T64X64_W32X32_K32_PF,
        Self::T128X64_W32X32_K32_PF,
        Self::T128X64_W64X32_K32_PF,
        Self::T128X128_W64X32_K32_PF,
        Self::T128X128_W64X64_K32_PF,
        Self::T128X256_W64X64_K32_PF1,
        Self::T256X128_W64X64_K32_PF1,
        Self::T128X128_W32X32_K32_PF1,
        Self::T128X128_W64X32_K64_PF1,
        Self::T128X128_W64X32_K32_PF1,
    ];

    /// Every tile the generator is tested with, for tests and the tuning
    /// bench. The first three are the tiles of the first coop kernel.
    pub const ALL: [Self; 14] = [
        Self::T128X64_W64X32_K32,
        Self::T64X64_W32X32_K32,
        Self::T32X32_W16X16_K32,
        Self::T128X64_W64X32_K32_DB,
        Self::T64X64_W32X32_K32_DB,
        Self::new(128, 128, (64, 64), 32, true),
        Self::T128X64_W32X32_K32_DB,
        Self::new(64, 128, (32, 64), 32, true),
        Self::new(64, 64, (32, 32), 64, true),
        Self::new(64, 32, (32, 16), 32, true),
        Self::new(32, 64, (16, 32), 32, true),
        Self::T64X64_W16X32_K32_DB,
        Self::T32X64_W16X16_K64_DB,
        Self::new(32, 32, (16, 16), 64, true),
    ];

    /// A short name such as `128x64/w64x32/k32/db`.
    pub fn name(self) -> String {
        format!(
            "{}x{}/w{}x{}/k{}{}{}",
            self.bm,
            self.bn,
            self.wm,
            self.wn,
            self.bk,
            if self.double { "/db" } else { "" },
            if self.prefetch { "/pf" } else { "" }
        )
    }

    pub(super) fn warps(self) -> u32 {
        (self.bm / self.wm) * (self.bn / self.wn)
    }

    pub(super) fn threads(self) -> u32 {
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
    pub(super) fn a_len(self) -> u32 {
        self.bm * (self.bk + PAD)
    }

    /// f16 elements of one staged `b` tile: `[n][k]` rows for `b[N, K]`,
    /// `[k][n]` rows for `b[K, N]`.
    pub(super) fn b_len(self, transposed_b: bool) -> u32 {
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

    pub(super) fn groups(self, m: u32, n: u32) -> (u32, u32) {
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

#[cfg(test)]
mod tests {
    use super::{CoopTile, PAD};

    #[test]
    fn tiles_split_evenly_over_their_threads() {
        for t in CoopTile::ALL.into_iter().chain(CoopTile::PIPELINED) {
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
        assert_eq!(
            CoopTile::T128X128_W64X32_K32_PF.name(),
            "128x128/w64x32/k32/db/pf"
        );
        assert_eq!(
            CoopTile::T128X256_W64X64_K32_PF1.name(),
            "128x256/w64x64/k32/pf"
        );
    }

    #[test]
    fn pipelined_tiles_are_new_and_distinct() {
        for (i, t) in CoopTile::PIPELINED.iter().enumerate() {
            assert!(t.prefetch, "{t:?}");
            assert!(!CoopTile::ALL.contains(t), "{t:?}");
            assert!(CoopTile::ALL.iter().all(|a| !a.prefetch));
            for u in &CoopTile::PIPELINED[..i] {
                assert_ne!(t.name(), u.name());
            }
        }
    }
}
