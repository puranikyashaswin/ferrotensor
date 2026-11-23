//! Runtime autotuning of the cooperative-matrix matmul's tile and K split.
//!
//! The router's static rules ([`plan`](super::plan)) come from benchmarks of
//! one product at a time, and on one GPU. Inside a training step the same
//! products ran up to 2.4 times slower than the rule that won the bench, so
//! the tuner times the candidates where they run: the step itself.
//!
//! With `FERROTENSOR_AUTOTUNE=1`, each product (a [`TuneKey`]: layout, kernel
//! variant, row bucket, N and K) starts a trial on its first occurrence. The
//! candidates are the static pick and up to seven other tiles, each with K
//! whole or split to fill the GPU. Products of up to 256 rows (a batch of 1
//! at 256 tokens, or a padded micro-batch) have six tiles per row bucket and
//! K split into 1, 2, 3, 4, 6 or 8 slices or to fill the GPU, since a
//! `matmul_bench --tiles` sweep on an A10G found the static rule 1.4 to 3
//! times slower than the best of those at many of them. Every later occurrence runs the candidate
//! with the fewest timings so far as the real dispatch, since all candidates
//! compute the same product, and tags it so the context reads its GPU time
//! back from timestamp queries (see `kernel_timing`), at the wait the trainer
//! already does.
//!
//! The candidates of a key run on different occurrences, and those have
//! different row counts (a batch of 1 pads each micro-batch to its own
//! length), while the time of a product does not grow with its rows inside a
//! small bucket: a 32-row timing is 0.5 to 1.0 times a 64-row one, by tile.
//! So no scale by rows is right. Each candidate is timed as a ratio to the
//! static pick, which the tuner times again in the same row group (the rows
//! rounded up to the smallest tile height of the key's candidates, so every
//! candidate has the same grid there) and again after [`ROUND_MAX`]
//! candidate timings. After [`TIMINGS`] ratios of each candidate the key
//! settles on the one with the lowest median, unless the static pick is
//! within [`MARGIN`] of it. Candidates within [`TIE`] of the best take the one
//! with the fewest K slices, since a split costs a reduction dispatch and a
//! partial-sum buffer that the timing charges only in part. Settled keys
//! cost one hash lookup per product.
//!
//! The settled choices are saved to a cache file (see
//! [`autotune_cache`](super::autotune_cache)), so a later run on the same
//! adapter and driver starts settled. A trial's dispatches change the order a
//! K split sums its slices in, so the first steps of a tuning run differ from
//! an untuned run by rounding; with the cache they do not.
//!
//! The cost is bounded three ways: at most [`MAX_CANDIDATES`] configurations
//! (a K split shares its tile's kernel, so at most six kernels for a product
//! of up to 256 rows) are timed [`TIMINGS`] times each per key; once the
//! time spent on trials (the host time compiling their kernels plus the GPU
//! time of their dispatches, which also did the step's work) reaches the
//! budget, no new key starts a trial; and a key whose timings never arrive
//! keeps the static pick.

use super::autotune_cache::{self, Choice, Platform};
use super::epilogue::EpilogueKind;
use super::loader::LoaderKind;
use super::plan::{grid_fits, split_k, split_to_fill};
use super::{CoopConfig, CoopTile};
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::time::Duration;

/// Environment variable that turns autotuning on: `1` tunes products that
/// the cache file does not have, `retune` ignores the file, anything else
/// (the default) keeps the static rules exactly.
pub const AUTOTUNE_ENV: &str = "FERROTENSOR_AUTOTUNE";

/// Environment variable that sets the tuning budget in seconds (a positive
/// number).
const BUDGET_ENV: &str = "FERROTENSOR_AUTOTUNE_BUDGET";

/// Tuning time after which no new key starts a trial.
const DEFAULT_BUDGET: Duration = Duration::from_secs(30);

/// Timings of each candidate before a key settles.
const TIMINGS: usize = 3;

/// How much faster than the static pick a candidate must be to replace it.
const MARGIN: f64 = 0.03;

/// Candidate timings of a row group before the static pick is timed there
/// again, which bounds how far the clock can drift between a candidate and the
/// timing it is compared with.
const ROUND_MAX: usize = 8;

/// How close to the best median a candidate must be to count as tied with it;
/// of the tied candidates the one with the fewest K slices wins.
const TIE: f64 = 0.01;

/// A candidate's time as parts per million of the static pick's time, timed in
/// the same row group.
const UNIT: u64 = 1_000_000;

/// Most candidates per key above 256 rows, the static pick included.
const MAX_CANDIDATES: usize = 8;

/// Most candidates per key of up to 256 rows: six tiles with up to seven
/// splits each, and the static pick. A split reuses its tile's compiled
/// kernel and adds only a reduction dispatch.
const SMALL_MAX_CANDIDATES: usize = 48;

/// Slices of K, besides one and the split that fills the GPU, that a product
/// of up to 256 rows tries.
const SMALL_SPLITS: [u32; 5] = [2, 3, 4, 6, 8];

/// Most candidates the key's row bucket may time.
fn max_candidates(bucket: u32) -> usize {
    if bucket <= 256 {
        SMALL_MAX_CANDIDATES
    } else {
        MAX_CANDIDATES
    }
}

/// The tiles worth timing at a row bucket of up to 256 rows, chosen greedily
/// from the `matmul_bench --tiles` sweep of the three SmolLM2 sizes on an
/// A10G (rows 32 to 256): each next tile is the one that most lowers the
/// geometric mean of the per-product best time of the tiles chosen so far.
/// Empty above 256 rows.
fn tune_tiles(bucket: u32) -> &'static [CoopTile] {
    match bucket {
        0..=64 => &[
            CoopTile::T32X64_W16X16_K64_DB,
            CoopTile::T64X64_W16X32_K32_DB,
            CoopTile::T32X32_W16X16_K64_DB,
            CoopTile::T32X32_W16X16_K32,
            CoopTile::T64X64_W32X32_K32_PF,
            CoopTile::T64X128_W32X64_K32_DB,
        ],
        65..=128 => &[
            CoopTile::T128X64_W32X32_K32_PF,
            CoopTile::T32X64_W16X16_K64_DB,
            CoopTile::T64X64_W32X32_K32_PF,
            CoopTile::T64X64_W16X32_K32_DB,
            CoopTile::T128X128_W64X64_K32_PF,
            CoopTile::T32X32_W16X16_K64_DB,
        ],
        129..=192 => &[
            CoopTile::T64X64_W32X32_K32_PF,
            CoopTile::T64X64_W16X32_K32_DB,
            CoopTile::T128X128_W64X64_K32_PF,
            CoopTile::T32X64_W16X16_K64_DB,
            CoopTile::T64X128_W32X64_K32_DB,
            CoopTile::T64X64_W32X32_K32,
        ],
        193..=256 => &[
            CoopTile::T128X64_W32X32_K32_PF,
            CoopTile::T64X64_W16X32_K32_DB,
            CoopTile::T128X128_W64X64_K32_PF,
            CoopTile::T32X64_W16X16_K64_DB,
            CoopTile::T64X64_W32X32_K32,
            CoopTile::T64X64_W32X32_K32_PF,
        ],
        _ => &[],
    }
}

/// What `FERROTENSOR_AUTOTUNE` asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Mode {
    Off,
    On,
    Retune,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::On => "on",
            Self::Retune => "retune",
        }
    }
}

/// The mode a value of [`AUTOTUNE_ENV`] names, or `None` for a value that
/// names none.
fn parse_mode(value: Option<&str>) -> Option<Mode> {
    match value.map(str::trim) {
        None | Some("" | "0") => Some(Mode::Off),
        Some("1") => Some(Mode::On),
        Some("retune") => Some(Mode::Retune),
        Some(_) => None,
    }
}

/// The mode of [`AUTOTUNE_ENV`] in `get`'s environment; a value that names
/// none leaves autotuning off, with a warning.
fn mode_from(get: &dyn Fn(&str) -> Option<String>) -> Mode {
    let value = get(AUTOTUNE_ENV);
    parse_mode(value.as_deref()).unwrap_or_else(|| {
        log::warn!("{AUTOTUNE_ENV}={value:?} is not 0, 1 or retune; autotune stays off");
        Mode::Off
    })
}

/// Whether the environment asks for autotuning on a device that routes
/// matmuls to the cooperative-matrix kernel. A context uses it to ask for
/// timestamp queries.
pub(crate) fn autotune_requested() -> bool {
    let value = std::env::var(AUTOTUNE_ENV).ok();
    crate::adapters::experimental_opt_in()
        && matches!(parse_mode(value.as_deref()), Some(Mode::On | Mode::Retune))
}

/// The budget [`BUDGET_ENV`] sets, or the default when it is unset or not a
/// positive number.
fn budget_from(get: &dyn Fn(&str) -> Option<String>) -> Duration {
    get(BUDGET_ENV)
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|s| s.is_finite() && *s > 0.0)
        .map_or(DEFAULT_BUDGET, |s| Duration::from_secs_f64(s.min(3600.0)))
}

/// What the tuner did in this run, for the report of a training run.
#[derive(Clone, Debug, PartialEq)]
pub struct AutotuneReport {
    /// `"on"` or `"retune"`.
    pub mode: &'static str,
    /// Time spent on trials: the host time compiling their kernels plus the
    /// GPU time of their dispatches (which also did the step's work).
    pub seconds: f64,
    /// Products timed and settled in this run.
    pub shapes_tuned: usize,
    /// Products that took their choice from the cache file.
    pub shapes_cached: usize,
    /// Products whose trial had not settled when the report was made.
    pub shapes_open: usize,
    /// Whether the budget stopped a product from starting a trial.
    pub budget_exhausted: bool,
    /// The cache file; `None` when no location is known.
    pub cache_file: Option<PathBuf>,
}

/// A product the tuner settles one choice for: every row count in a bucket
/// shares it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct TuneKey {
    transposed_b: bool,
    loader: LoaderKind,
    epilogue: EpilogueKind,
    /// The smallest of 64, 128, 192, 256, 512, 1024, 2048 and 4096 that
    /// holds the rows.
    m_bucket: u32,
    n: u32,
    k: u32,
}

impl TuneKey {
    pub(super) fn new(
        transposed_b: bool,
        (loader, epilogue): (LoaderKind, EpilogueKind),
        (m, k, n): (u32, u32, u32),
    ) -> Self {
        Self {
            transposed_b,
            loader,
            epilogue,
            m_bucket: m_bucket(m),
            n,
            k,
        }
    }

    pub(super) fn m_bucket(&self) -> u32 {
        self.m_bucket
    }

    /// The key as the cache file writes it.
    fn line_key(&self) -> String {
        format!(
            "layout={};variant={:?}/{:?};m={};n={};k={}",
            if self.transposed_b { "a@bT" } else { "a@b" },
            self.loader,
            self.epilogue,
            self.m_bucket,
            self.n,
            self.k
        )
    }
}

/// The row bucket of `m` rows: products of up to 64 rows (the kernel's
/// smallest) share the first, and every product past 2048 rows the last. These
/// are coarser than the router's buckets: a key per router bucket gave 135M and
/// 360M steps, which visit each key a few times, 11 keys still timing
/// candidates at the end of a 120-step run instead of 7, and the timing of
/// slower candidates cost them about 1% of the step.
pub(super) fn m_bucket(m: u32) -> u32 {
    match m {
        0..=64 => 64,
        65..=128 => 128,
        129..=192 => 192,
        193..=256 => 256,
        257..=512 => 512,
        513..=1024 => 1024,
        1025..=2048 => 2048,
        _ => 4096,
    }
}

/// Workgroups over waves of `target`: the share of the last wave's slots
/// that hold a workgroup, so 1.0 when no slot idles.
fn fill(workgroups: u64, target: u32) -> f64 {
    let target = u64::from(target.max(1));
    workgroups as f64 / (workgroups.div_ceil(target).max(1) * target) as f64
}

/// The configurations worth timing for an `m x k x n` product, best first:
/// every tile that fits the device and has at most twice the bucket's rows
/// in a workgroup, each with K whole and, when its grid reaches under twice
/// `target` workgroups, with K split to fill the GPU. Up to 256 rows only the
/// bucket's [`tune_tiles`] are tried, each also with K split into
/// [`SMALL_SPLITS`] slices. They are ordered by how full the last wave of
/// workgroups is, then by tile size, then by their order in the tile list.
pub(super) fn candidates(
    (m, k, n): (u32, u32, u32),
    bucket: u32,
    target: u32,
    tile_fits: &dyn Fn(CoopTile) -> bool,
) -> Vec<CoopConfig> {
    let mut found: Vec<(CoopConfig, f64)> = Vec::new();
    let small = !tune_tiles(bucket).is_empty();
    let tiles: Vec<CoopTile> = if small {
        tune_tiles(bucket).to_vec()
    } else {
        CoopTile::ALL
            .into_iter()
            .chain(CoopTile::PIPELINED)
            .collect()
    };
    for tile in tiles {
        if !tile_fits(tile) || tile.bm > 2 * bucket || !grid_fits(tile, m, n, 1) {
            continue;
        }
        let (gx, gy) = tile.groups(m, n);
        let groups = u64::from(gx) * u64::from(gy);
        found.push((CoopConfig { tile, split: 1 }, fill(groups, target)));
        if groups < 2 * u64::from(target) {
            let mut splits = vec![split_to_fill(tile, (m, k, n), target)];
            if small {
                splits.extend(SMALL_SPLITS.iter().map(|&s| split_k(k, tile.bk, s).0));
            }
            let mut seen = vec![1];
            for split in splits {
                if !seen.contains(&split) {
                    seen.push(split);
                    let filled = fill(groups * u64::from(split), target);
                    found.push((CoopConfig { tile, split }, filled));
                }
            }
        }
    }
    let area = |c: &CoopConfig| c.tile.bm * c.tile.bn;
    found.sort_by(|a, b| {
        b.1.total_cmp(&a.1)
            .then_with(|| area(&b.0).cmp(&area(&a.0)))
    });
    found.into_iter().map(|(config, _)| config).collect()
}

/// The median of `times`; `u64::MAX` for none.
fn median(times: &[u64]) -> u64 {
    let mut sorted = times.to_vec();
    sorted.sort_unstable();
    sorted.get(sorted.len() / 2).copied().unwrap_or(u64::MAX)
}

/// A candidate's time as parts per million ([`UNIT`]) of its anchor's, the
/// static pick timed in the same row group; `u64::MAX` when either time is
/// unusable or the anchor took no time.
fn ratio(nanos: u64, anchor: Anchored) -> u64 {
    let Anchored::Nanos(anchor) = anchor else {
        return u64::MAX;
    };
    (u128::from(nanos) * u128::from(UNIT))
        .checked_div(u128::from(anchor))
        .and_then(|r| u64::try_from(r).ok())
        .unwrap_or(u64::MAX)
}

/// The index of the candidate to settle on and its median, once every
/// candidate has [`TIMINGS`] timings (`u64::MAX` for a timing that was
/// unusable). `slices[i]` is how many slices of K candidate `i` runs. Candidate
/// 0 is the static pick: another candidate wins only with a median at least
/// [`MARGIN`] below its. Of the candidates that clear the margin and are
/// within [`TIE`] of the best median, the one with the fewest slices wins, then
/// the one that comes first.
fn winner(times: &[Vec<u64>], slices: &[u32]) -> Option<(usize, u64)> {
    if times.is_empty() || times.iter().any(|t| t.len() < TIMINGS) {
        return None;
    }
    let medians: Vec<u64> = times.iter().map(|t| median(t)).collect();
    let static_median = *medians.first()?;
    let best_median = *medians.iter().min()?;
    let band = best_median as f64 * (1.0 + TIE);
    let clear = static_median as f64 * (1.0 - MARGIN);
    medians
        .iter()
        .enumerate()
        .filter(|&(_, &m)| m != u64::MAX && (m as f64) <= band && (m as f64) < clear)
        .min_by_key(|&(i, _)| (slices.get(i).copied().unwrap_or(u32::MAX), i))
        .map(|(i, &m)| (i, m))
        .or(Some((0, static_median)))
}

/// What the timing of an anchor came to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Anchored {
    Nanos(u64),
    Unusable,
}

/// The anchor a row group is compared against now.
struct GroupAnchor {
    sample: u64,
    /// Candidate timings paired with it so far.
    uses: usize,
}

/// A key's candidates and what has been timed so far.
struct Trial {
    candidates: Vec<CoopConfig>,
    /// Dispatches of each candidate so far. The static pick, candidate 0, has
    /// all of them from the start: it runs as an anchor, not as a candidate.
    dispatched: Vec<usize>,
    /// Timings of each candidate as parts per million of the static pick's
    /// time in the same row group. The static pick's own are all [`UNIT`].
    times: Vec<Vec<u64>>,
    /// Rows are grouped by this many: the greatest common divisor of the
    /// candidates' tile heights, so a group's rows give every candidate the
    /// same number of row tiles.
    group_size: u32,
    /// The anchor each row group compares against now.
    groups: HashMap<u32, GroupAnchor>,
    /// The anchors that have reported, by sample id.
    anchored: HashMap<u64, Anchored>,
    /// Candidate timings whose anchor has not reported yet: the candidate,
    /// the anchor's sample id and the nanoseconds.
    waiting: Vec<(usize, u64, u64)>,
    /// The anchors' raw times, for the median the cache file keeps.
    anchor_nanos: Vec<u64>,
}

impl Trial {
    fn new(candidates: Vec<CoopConfig>) -> Self {
        let count = candidates.len();
        let group_size = candidates
            .iter()
            .fold(0, |g, c| gcd(g, c.tile.bm))
            .max(ROW_GROUP_MIN);
        let mut dispatched = vec![0; count];
        let mut times = vec![Vec::new(); count];
        if let (Some(d), Some(t)) = (dispatched.first_mut(), times.first_mut()) {
            *d = TIMINGS;
            *t = vec![UNIT; TIMINGS];
        }
        Self {
            candidates,
            dispatched,
            times,
            group_size,
            groups: HashMap::new(),
            anchored: HashMap::new(),
            waiting: Vec::new(),
            anchor_nanos: Vec::new(),
        }
    }

    /// Takes the time of the anchor `sample`, `None` when unusable, and pairs
    /// the candidate timings that waited for it.
    fn report_anchor(&mut self, sample: u64, nanos: Option<u64>) {
        let anchored = nanos.map_or(Anchored::Unusable, Anchored::Nanos);
        self.anchored.insert(sample, anchored);
        self.anchor_nanos.extend(nanos);
        for (candidate, anchor, nanos) in std::mem::take(&mut self.waiting) {
            if anchor == sample {
                self.push(candidate, ratio(nanos, anchored));
            } else {
                self.waiting.push((candidate, anchor, nanos));
            }
        }
    }

    /// Takes the time of `candidate` paired with the anchor `anchor`, `None`
    /// when unusable. Timestamps are read back in dispatch order, so the
    /// anchor has nearly always reported; one that has not is waited for.
    fn report_candidate(&mut self, candidate: usize, anchor: u64, nanos: Option<u64>) {
        let Some(nanos) = nanos else {
            self.push(candidate, u64::MAX);
            return;
        };
        match self.anchored.get(&anchor).copied() {
            Some(anchored) => self.push(candidate, ratio(nanos, anchored)),
            None => self.waiting.push((candidate, anchor, nanos)),
        }
    }

    fn push(&mut self, candidate: usize, value: u64) {
        if let Some(times) = self.times.get_mut(candidate) {
            times.push(value);
        }
    }
}

/// The smallest rows a group holds: the fragment row, as in the router.
const ROW_GROUP_MIN: u32 = 16;

/// The greatest common divisor of `a` and `b`, `a` when `b` is 0.
fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// What a timed dispatch is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    /// The static pick, timed to be compared against.
    Anchor,
    /// A candidate, to be compared with the anchor of this sample id.
    Candidate { anchor: u64 },
}

/// A dispatch (or two, with K split) whose timestamps have not all arrived.
struct Open {
    key: TuneKey,
    candidate: usize,
    role: Role,
    /// Dispatches still to report: 2 with K split (the product and its
    /// reduction), else 1.
    parts_left: u32,
    nanos: u64,
    unusable: bool,
}

/// What the router runs for one product, and the sample to tag it with when
/// the run is a timing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Pick {
    pub(super) config: CoopConfig,
    /// The sample id the dispatches of this run carry, if it is a timing.
    pub(super) sample: Option<u64>,
}

/// The autotuner's state.
pub(super) struct Tuner {
    mode: Mode,
    file: Option<PathBuf>,
    header: String,
    /// The entries as the file holds them: loaded ones and those settled in
    /// this run.
    entries: BTreeMap<String, Choice>,
    /// What every settled or cached key runs.
    settled: HashMap<TuneKey, CoopConfig>,
    trials: HashMap<TuneKey, Trial>,
    open: HashMap<u64, Open>,
    next_sample: u64,
    dirty: bool,
    budget: Duration,
    spent: Duration,
    tuned: usize,
    cached: usize,
    budget_hit: bool,
}

impl Tuner {
    pub(super) fn new(
        mode: Mode,
        file: Option<PathBuf>,
        header: String,
        entries: BTreeMap<String, Choice>,
        budget: Duration,
    ) -> Self {
        Self {
            mode,
            file,
            header,
            entries,
            settled: HashMap::new(),
            trials: HashMap::new(),
            open: HashMap::new(),
            next_sample: 0,
            dirty: false,
            budget,
            spent: Duration::ZERO,
            tuned: 0,
            cached: 0,
            budget_hit: false,
        }
    }

    /// The tuner `get`'s environment asks for, or `None` for the static rules:
    /// autotuning is off, matmuls are not routed to the kernel, or the context
    /// has no timestamp queries (said on stderr). `header` is the adapter's
    /// line of the cache file, built only when it is needed.
    pub(super) fn from_env(
        get: &dyn Fn(&str) -> Option<String>,
        (routed, timestamps): (bool, bool),
        header: impl FnOnce() -> String,
    ) -> Option<Self> {
        let mode = mode_from(get);
        if mode == Mode::Off || !routed {
            return None;
        }
        if !timestamps {
            log::warn!(
                "{AUTOTUNE_ENV} is set but this adapter has no timestamp queries; \
                 the static rules stay"
            );
            return None;
        }
        let header = header();
        let file = autotune_cache::cache_path(get, Platform::current());
        let entries = match (&file, mode) {
            (Some(path), Mode::On) => autotune_cache::load(path, &header),
            _ => BTreeMap::new(),
        };
        Some(Self::new(mode, file, header, entries, budget_from(get)))
    }

    /// Whether timings are waiting to be read back.
    pub(super) fn wants_samples(&self) -> bool {
        !self.open.is_empty()
    }

    /// Adds to the time spent on trials.
    pub(super) fn charge(&mut self, time: Duration) {
        self.spent += time;
    }

    /// What to run for the `key` product of `rows` rows when the static rules
    /// pick `static_cfg`. `usable` says whether a configuration runs this
    /// product on this device, and `candidates` lists the configurations to
    /// try when the key has none yet.
    pub(super) fn pick(
        &mut self,
        key: TuneKey,
        rows: u32,
        static_cfg: CoopConfig,
        usable: &dyn Fn(CoopConfig) -> bool,
        candidates: &dyn Fn() -> Vec<CoopConfig>,
    ) -> Pick {
        let run = |config| Pick {
            config,
            sample: None,
        };
        let known = |config: CoopConfig| run(if usable(config) { config } else { static_cfg });
        if let Some(&config) = self.settled.get(&key) {
            return known(config);
        }
        if !self.trials.contains_key(&key) {
            if let Some(choice) = self.entries.get(&key.line_key()).copied() {
                self.settled.insert(key, choice.config);
                self.cached += 1;
                return known(choice.config);
            }
            if self.spent >= self.budget {
                self.budget_hit = true;
                return run(static_cfg);
            }
            let mut list = vec![static_cfg];
            for config in candidates() {
                let room = list.len() < max_candidates(key.m_bucket);
                if room && !list.contains(&config) && usable(config) {
                    list.push(config);
                }
            }
            if list.len() == 1 {
                self.settled.insert(key, static_cfg);
                return run(static_cfg);
            }
            self.trials.insert(key, Trial::new(list));
        }
        let Some(trial) = self.trials.get_mut(&key) else {
            return run(static_cfg);
        };
        let next = trial
            .dispatched
            .iter()
            .enumerate()
            .filter(|&(_, &d)| d < TIMINGS)
            .min_by_key(|&(i, &d)| (d, i))
            .map(|(i, _)| i);
        let Some(candidate) = next else {
            // Every candidate has its dispatches; the timings are on their way.
            return run(static_cfg);
        };
        // A candidate is timed against the static pick in its own row group:
        // the anchor runs first there, and again after ROUND_MAX timings.
        let group = rows.div_ceil(trial.group_size).max(1);
        let anchor = trial
            .groups
            .get(&group)
            .filter(|g| g.uses < ROUND_MAX)
            .map(|g| g.sample);
        let (index, role) = match anchor {
            Some(anchor) => (candidate, Role::Candidate { anchor }),
            None => (0, Role::Anchor),
        };
        let Some(&config) = trial.candidates.get(index) else {
            return run(static_cfg);
        };
        if !usable(config) {
            return run(static_cfg);
        }
        let sample = self.next_sample;
        self.next_sample += 1;
        match role {
            Role::Anchor => {
                trial.groups.insert(group, GroupAnchor { sample, uses: 0 });
            }
            Role::Candidate { .. } => {
                if let Some(count) = trial.dispatched.get_mut(candidate) {
                    *count += 1;
                }
                if let Some(g) = trial.groups.get_mut(&group) {
                    g.uses += 1;
                }
            }
        }
        let split = split_k(key.k, config.tile.bk, config.split).0;
        self.open.insert(
            sample,
            Open {
                key,
                candidate: index,
                role,
                parts_left: if split > 1 { 2 } else { 1 },
                nanos: 0,
                unusable: false,
            },
        );
        Pick {
            config,
            sample: Some(sample),
        }
    }

    /// Takes one dispatch's time of `sample`, `None` when its timestamps were
    /// unusable. Once the sample's dispatches have all reported, the key may
    /// settle.
    pub(super) fn record(&mut self, sample: u64, time: Option<Duration>) {
        let Some(open) = self.open.get_mut(&sample) else {
            return;
        };
        match time {
            Some(time) => {
                let nanos = u64::try_from(time.as_nanos()).unwrap_or(u64::MAX);
                open.nanos = open.nanos.saturating_add(nanos);
                self.spent += time;
            }
            None => open.unusable = true,
        }
        open.parts_left = open.parts_left.saturating_sub(1);
        if open.parts_left > 0 {
            return;
        }
        let Some(done) = self.open.remove(&sample) else {
            return;
        };
        let nanos = (!done.unusable).then_some(done.nanos);
        if let Some(trial) = self.trials.get_mut(&done.key) {
            match done.role {
                Role::Anchor => trial.report_anchor(sample, nanos),
                Role::Candidate { anchor } => trial.report_candidate(done.candidate, anchor, nanos),
            }
        }
        self.settle(done.key);
    }

    /// Settles `key` if every candidate has its timings.
    fn settle(&mut self, key: TuneKey) {
        let Some(trial) = self.trials.get(&key) else {
            return;
        };
        let slices: Vec<u32> = trial
            .candidates
            .iter()
            .map(|c| split_k(key.k, c.tile.bk, c.split).0)
            .collect();
        let Some((best, ratio)) = winner(&trial.times, &slices) else {
            return;
        };
        let Some(&config) = trial.candidates.get(best) else {
            return;
        };
        // The static pick's time in nanoseconds, as the anchors measured it.
        let anchor_nanos = median(&trial.anchor_nanos);
        if let Some(&first) = trial.candidates.first() {
            let name = |c: CoopConfig| format!("{} split {}", c.tile.name(), c.split);
            log::info!(
                "autotune: {} chose {} ({} ppm of the static {}, {} ns) from {} candidates",
                key.line_key(),
                name(config),
                ratio,
                name(first),
                anchor_nanos,
                trial.candidates.len()
            );
        }
        // A key whose static pick never timed usably runs the static pick
        // this run and is not saved.
        let usable = !trial.anchor_nanos.is_empty();
        let median_us =
            (u128::from(ratio) * u128::from(anchor_nanos) / u128::from(UNIT)) as f64 / 1000.0;
        self.trials.remove(&key);
        self.settled.insert(key, config);
        self.tuned += 1;
        if usable && ratio != u64::MAX {
            self.entries
                .insert(key.line_key(), Choice { config, median_us });
            self.dirty = true;
        }
    }

    /// Writes the cache file when a key settled since the last write; a
    /// failure is reported on stderr and the run goes on.
    pub(super) fn save(&mut self) {
        if !self.dirty {
            return;
        }
        let Some(path) = &self.file else {
            return;
        };
        let text = autotune_cache::render(&self.header, &self.entries);
        match autotune_cache::write_atomic(path, &text) {
            Ok(()) => self.dirty = false,
            Err(e) => log::warn!("cannot write the autotune cache {}: {e}", path.display()),
        }
    }

    pub(super) fn report(&self) -> AutotuneReport {
        AutotuneReport {
            mode: self.mode.name(),
            seconds: self.spent.as_secs_f64(),
            shapes_tuned: self.tuned,
            shapes_cached: self.cached,
            shapes_open: self.trials.len(),
            budget_exhausted: self.budget_hit,
            cache_file: self.file.clone(),
        }
    }
}

impl Drop for Tuner {
    fn drop(&mut self) {
        self.save();
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    const KEY: (LoaderKind, EpilogueKind) = (LoaderKind::Plain, EpilogueKind::AlphaBeta);
    const HEADER: &str = "adapter=fake";

    fn cfg(tile: CoopTile, split: u32) -> CoopConfig {
        CoopConfig { tile, split }
    }

    fn key() -> TuneKey {
        TuneKey::new(true, KEY, (256, 2048, 2048))
    }

    fn tuner(entries: BTreeMap<String, Choice>, budget: Duration) -> Tuner {
        Tuner::new(Mode::On, None, HEADER.to_owned(), entries, budget)
    }

    fn any(_: CoopConfig) -> bool {
        true
    }

    /// Runs `key` until the tuner stops asking for timings, answering each
    /// with `nanos(config)`; returns what the router runs afterwards and the
    /// number of timed runs. Every run has the rows of `key`'s bucket.
    fn drive(
        t: &mut Tuner,
        key: TuneKey,
        configs: (CoopConfig, &[CoopConfig]),
        nanos: &dyn Fn(CoopConfig) -> u64,
    ) -> (CoopConfig, usize) {
        drive_rows(t, key, configs, &|_| key.m_bucket(), nanos)
    }

    /// As [`drive`], with `rows(n)` rows on the `n`th run.
    fn drive_rows(
        t: &mut Tuner,
        key: TuneKey,
        (static_cfg, others): (CoopConfig, &[CoopConfig]),
        rows: &dyn Fn(u32) -> u32,
        nanos: &dyn Fn(CoopConfig) -> u64,
    ) -> (CoopConfig, usize) {
        let list = || others.to_vec();
        for timed in 0..1000_u32 {
            let pick = t.pick(key, rows(timed), static_cfg, &any, &list);
            let Some(sample) = pick.sample else {
                return (pick.config, timed as usize);
            };
            // A product with K split reports its two dispatches separately.
            let parts = split_k(key.k, pick.config.tile.bk, pick.config.split)
                .0
                .min(2);
            let each = nanos(pick.config) / u64::from(parts.max(1));
            for _ in 0..parts.max(1) {
                t.record(sample, Some(Duration::from_nanos(each)));
            }
        }
        panic!("the tuner kept asking for timings");
    }

    const A: CoopConfig = CoopConfig {
        tile: CoopTile::T64X64_W32X32_K32_DB,
        split: 1,
    };
    const B: CoopConfig = CoopConfig {
        tile: CoopTile::T128X64_W64X32_K32_DB,
        split: 1,
    };
    const C: CoopConfig = CoopConfig {
        tile: CoopTile::T64X64_W16X32_K32_DB,
        split: 1,
    };
    const D: CoopConfig = CoopConfig {
        tile: CoopTile::T32X64_W16X16_K64_DB,
        split: 1,
    };

    fn time_of(c: CoopConfig) -> u64 {
        match c {
            A => 1_000_000,
            B => 800_000,
            C => 500_000,
            D => 700_000,
            _ => 2_000_000,
        }
    }

    #[test]
    fn tuning_settles_on_the_fastest_of_the_timings() {
        let mut t = tuner(BTreeMap::new(), DEFAULT_BUDGET);
        let (ran, timed) = drive(&mut t, key(), (A, &[B, C, D]), &time_of);
        assert_eq!(ran, C);
        // 3 candidates besides the static pick, 3 timings each, and the
        // static pick timed twice as the anchor: first, and again after 8
        // candidate timings. All are real dispatches.
        assert_eq!(timed, 9 + 2);
        let report = t.report();
        assert_eq!((report.shapes_tuned, report.shapes_open), (1, 0));
        // The static pick twice at 1 ms, B 3 x 0.8, C 3 x 0.5, D 3 x 0.7.
        assert!((report.seconds - 8.0e-3).abs() < 1e-9, "{}", report.seconds);
        // The key stays settled, and the file image holds it.
        assert_eq!(drive(&mut t, key(), (A, &[B, C, D]), &time_of), (C, 0));
        let entry = t.entries[&key().line_key()];
        assert_eq!(entry.config, C);
        // C took 0.5 of the static pick's 1000 us.
        assert!((entry.median_us - 500.0).abs() < 1e-9);
        assert!(t.dirty);
    }

    #[test]
    fn a_split_product_adds_its_reduction_to_its_time() {
        // B split in 3 reports its product and its reduction: 300 + 200 us
        // each run, 500 in all, against A's 1000 and C's 600.
        let split = cfg(B.tile, 3);
        let time = |c: CoopConfig| match c {
            A => 1_000_000,
            C => 600_000,
            _ => 500_000,
        };
        let mut t = tuner(BTreeMap::new(), DEFAULT_BUDGET);
        let (ran, _) = drive(&mut t, key(), (A, &[C, split]), &time);
        assert_eq!(ran, split);
    }

    #[test]
    fn a_candidate_must_beat_the_static_pick_by_the_margin() {
        let near = |c: CoopConfig| if c == B { 975_000 } else { 1_000_000 };
        let mut t = tuner(BTreeMap::new(), DEFAULT_BUDGET);
        assert_eq!(drive(&mut t, key(), (A, &[B]), &near).0, A);
        let clear = |c: CoopConfig| if c == B { 960_000 } else { 1_000_000 };
        let mut t = tuner(BTreeMap::new(), DEFAULT_BUDGET);
        assert_eq!(drive(&mut t, key(), (A, &[B]), &clear).0, B);
        // The median ignores one slow timing in three.
        assert_eq!(
            winner(&[vec![100, 100, 100], vec![10, 90, 5_000]], &[1, 1]),
            Some((1, 90))
        );
        assert_eq!(
            winner(&[vec![100, 100, 100], vec![10, 99, 5_000]], &[1, 1]),
            Some((0, 100))
        );
        assert_eq!(winner(&[vec![100, 100]], &[1]), None);
        assert_eq!(winner(&[], &[]), None);
    }

    #[test]
    fn timings_from_steps_of_different_rows_are_compared_per_row() {
        // Both candidates take their time in proportion to the rows: B 20%
        // longer per row than A. The steps alternate between 1024 and 512
        // rows, so B is timed on the short ones and its raw times are lower.
        let key = TuneKey::new(true, KEY, (1024, 2048, 2048));
        let per_row = |c: CoopConfig| if c == B { 1_200 } else { 1_000 };
        let mut t = tuner(BTreeMap::new(), DEFAULT_BUDGET);
        let list = || vec![B];
        let mut ran = None;
        for step in 0..1000_u32 {
            let rows = if step % 2 == 0 { 1024 } else { 512 };
            let pick = t.pick(key, rows, A, &any, &list);
            let Some(sample) = pick.sample else {
                ran = Some(pick.config);
                break;
            };
            let nanos = per_row(pick.config) * u64::from(rows);
            t.record(sample, Some(Duration::from_nanos(nanos)));
        }
        assert_eq!(ran, Some(A));
    }

    #[test]
    fn paired_timings_cancel_the_row_count() {
        // The time of the product does not depend on its rows: the static
        // pick takes 100 us and D 90 us at 16 or at 64 rows. D is timed only
        // at 16 rows, a group of its own (the smallest tile height is 32),
        // after the static pick ran once at 64 rows. A scale by rows would
        // read D as 4 x 90 us against the static 100 us and drop it.
        let key = TuneKey::new(true, KEY, (64, 2048, 2048));
        let flat = |c: CoopConfig| if c == D { 90_000 } else { 100_000 };
        let rows = |n: u32| if n == 0 { 64 } else { 16 };
        let mut t = tuner(BTreeMap::new(), DEFAULT_BUDGET);
        let (ran, timed) = drive_rows(&mut t, key, (A, &[D]), &rows, &flat);
        // The anchor at 64 rows, the anchor at 16 rows, three timings of D.
        assert_eq!((ran, timed), (D, 5));
        let entry = t.entries[&key.line_key()];
        assert_eq!(entry.config, D);
        // 0.9 of the anchors' 100 us.
        assert!((entry.median_us - 90.0).abs() < 1e-9, "{}", entry.median_us);
    }

    #[test]
    fn a_key_seen_once_per_pass_still_settles() {
        // Nine candidates, one occurrence per pass, rows alternating between
        // two groups of 32 rows (64 is group 2, 200 is group 7). The time
        // does not depend on the rows; the tile D with 2 slices is fastest.
        let fastest = cfg(D.tile, 2);
        let others = [
            cfg(B.tile, 2),
            cfg(C.tile, 2),
            fastest,
            cfg(B.tile, 3),
            cfg(C.tile, 3),
            cfg(D.tile, 3),
            cfg(A.tile, 2),
            cfg(A.tile, 3),
        ];
        let time = |c: CoopConfig| if c == fastest { 600_000 } else { 900_000 };
        let key = TuneKey::new(true, KEY, (256, 2048, 2048));
        let rows = |n: u32| if n % 2 == 0 { 64 } else { 200 };
        let mut t = tuner(BTreeMap::new(), DEFAULT_BUDGET);
        let (ran, timed) = drive_rows(&mut t, key, (A, &others), &rows, &time);
        assert_eq!(ran, fastest);
        // 8 candidates x 3 timings, and the static pick as the anchor of each
        // group, first and again after 8 timings: 2 groups x 2.
        assert_eq!(timed, 24 + 4);
    }

    #[test]
    fn the_anchor_is_timed_again_after_eight_timings() {
        let others: Vec<CoopConfig> = (2..=10).map(|s| cfg(B.tile, s)).collect();
        let mut t = tuner(BTreeMap::new(), DEFAULT_BUDGET);
        let list = || others.clone();
        let mut ran = Vec::new();
        let mut samples = Vec::new();
        for _ in 0..12 {
            let pick = t.pick(key(), 256, A, &any, &list);
            ran.push(pick.config);
            samples.extend(pick.sample);
        }
        // The static pick, the first eight candidates, the static pick
        // again, then the ninth candidate and the first once more.
        let mut want = vec![A];
        want.extend(others.iter().take(8));
        want.push(A);
        want.push(others[8]);
        want.push(others[0]);
        assert_eq!(ran, want);
        assert_eq!(samples, (0..12).collect::<Vec<u64>>());
    }

    #[test]
    fn rows_in_one_group_share_an_anchor() {
        // A (64 rows) and D (32 rows) give groups of 32 rows: 33 and 64 are
        // group 2, 65 is group 3.
        let mut t = tuner(BTreeMap::new(), DEFAULT_BUDGET);
        let list = || vec![D];
        let role = |t: &Tuner, sample: Option<u64>| t.open[&sample.expect("a timing")].role;
        let first = t.pick(key(), 33, A, &any, &list);
        assert_eq!(role(&t, first.sample), Role::Anchor);
        let second = t.pick(key(), 64, A, &any, &list);
        let anchor = first.sample.expect("a timing");
        assert_eq!(role(&t, second.sample), Role::Candidate { anchor });
        assert_eq!(second.config, D);
        let third = t.pick(key(), 65, A, &any, &list);
        assert_eq!(role(&t, third.sample), Role::Anchor);
        assert_ne!(third.sample, first.sample);
        let trial = &t.trials[&key()];
        assert_eq!(trial.group_size, 32);
        assert_eq!(trial.groups.len(), 2);
    }

    #[test]
    fn a_candidate_reported_before_its_anchor_still_pairs() {
        let mut t = tuner(BTreeMap::new(), DEFAULT_BUDGET);
        let list = || vec![D];
        let anchor = t.pick(key(), 256, A, &any, &list).sample.expect("anchor");
        let pair = t.pick(key(), 256, A, &any, &list).sample.expect("pair");
        t.record(pair, Some(Duration::from_micros(90)));
        assert_eq!(t.trials[&key()].waiting.len(), 1);
        assert!(t.trials[&key()].times[1].is_empty());
        t.record(anchor, Some(Duration::from_micros(120)));
        let trial = &t.trials[&key()];
        assert!(trial.waiting.is_empty());
        // 90 us over 120 us.
        assert_eq!(trial.times[1], vec![750_000]);
    }

    #[test]
    fn an_unusable_anchor_makes_its_pairs_lose() {
        let mut t = tuner(BTreeMap::new(), DEFAULT_BUDGET);
        let list = || vec![D];
        let run = |t: &mut Tuner, usable: bool| {
            let sample = t.pick(key(), 256, A, &any, &list).sample.expect("a timing");
            t.record(sample, usable.then(|| Duration::from_micros(10)));
        };
        // The anchor's timestamps ran backwards, D's are fine.
        run(&mut t, false);
        run(&mut t, true);
        run(&mut t, true);
        assert_eq!(t.trials[&key()].times[1], vec![u64::MAX, u64::MAX]);
        run(&mut t, true);
        // D settles to the static pick, and an unmeasured key is not saved.
        let pick = t.pick(key(), 256, A, &any, &list);
        assert_eq!((pick.config, pick.sample), (A, None));
        assert_eq!(t.report().shapes_tuned, 1);
        assert!(t.entries.is_empty());
    }

    #[test]
    fn within_one_percent_the_fewer_slices_win() {
        let times = |ratios: [u64; 3]| -> Vec<Vec<u64>> {
            let mut all = vec![vec![1000; 3]];
            all.extend(ratios.map(|r| vec![r; 3]));
            all
        };
        // Candidate 1 has 4 slices, 2 has 1, 3 has 2. 900 and 905 are within
        // 1%: the unsplit one wins.
        assert_eq!(
            winner(&times([900, 905, 950]), &[1, 4, 1, 2]),
            Some((2, 905))
        );
        // 910 is outside the band of 900 (909).
        assert_eq!(
            winner(&times([900, 910, 950]), &[1, 4, 1, 2]),
            Some((1, 900))
        );
        // Among equal slices, the one first in the list.
        assert_eq!(
            winner(&times([900, 902, 950]), &[1, 2, 2, 2]),
            Some((1, 900))
        );
        // 969 is within the band of 965 and clears the margin (970).
        assert_eq!(
            winner(&times([965, 969, 990]), &[1, 4, 1, 2]),
            Some((2, 969))
        );
        // 972 is within the band of 965 (974.65) but does not clear the
        // margin: the one with 4 slices wins.
        assert_eq!(
            winner(&times([965, 972, 990]), &[1, 4, 1, 2]),
            Some((1, 965))
        );
        // Nothing clears the margin: the static pick.
        assert_eq!(
            winner(&times([975, 980, 990]), &[1, 4, 1, 2]),
            Some((0, 1000))
        );
    }

    #[test]
    fn a_ratio_is_parts_per_million_of_the_anchor() {
        assert_eq!(ratio(90_000, Anchored::Nanos(100_000)), 900_000);
        assert_eq!(ratio(100_000, Anchored::Nanos(100_000)), UNIT);
        assert_eq!(ratio(1, Anchored::Nanos(0)), u64::MAX);
        assert_eq!(ratio(1, Anchored::Unusable), u64::MAX);
        assert_eq!(ratio(u64::MAX, Anchored::Nanos(1)), u64::MAX);
        assert_eq!([48, 64, 96, 32].iter().fold(0, |g, &b| gcd(g, b)), 16);
    }

    #[test]
    fn unusable_timings_never_win() {
        let mut t = tuner(BTreeMap::new(), DEFAULT_BUDGET);
        let list = || vec![B];
        for _ in 0..1000 {
            let pick = t.pick(key(), 256, A, &any, &list);
            let Some(sample) = pick.sample else {
                break;
            };
            // B's timestamps ran backwards; A's are fine.
            let time = (pick.config == A).then(|| Duration::from_micros(10));
            t.record(sample, time);
        }
        let pick = t.pick(key(), 256, A, &any, &list);
        assert_eq!((pick.config, pick.sample), (A, None));
        assert_eq!(t.report().shapes_tuned, 1);
    }

    #[test]
    fn with_no_cache_and_no_time_left_the_static_pick_runs_every_product() {
        // A budget of nothing starts no trial: the router's rules, exactly.
        let mut t = tuner(BTreeMap::new(), Duration::ZERO);
        for (m, k, n) in [(64, 576, 576), (256, 2048, 8192), (2048, 49152, 576)] {
            let key = TuneKey::new(false, KEY, (m, k, n));
            let pick = t.pick(key, m, A, &any, &|| vec![B, C]);
            assert_eq!(
                pick,
                Pick {
                    config: A,
                    sample: None
                },
                "{m} {k} {n}"
            );
        }
        let report = t.report();
        assert!(report.budget_exhausted);
        assert_eq!((report.shapes_tuned, report.shapes_cached), (0, 0));
        assert!(!t.dirty);
    }

    #[test]
    fn a_product_with_no_other_candidate_keeps_the_static_pick_untimed() {
        let mut t = tuner(BTreeMap::new(), DEFAULT_BUDGET);
        assert_eq!(drive(&mut t, key(), (A, &[A]), &time_of), (A, 0));
        assert_eq!(t.report().shapes_open, 0);
        assert!(t.entries.is_empty());
    }

    #[test]
    fn the_cache_decides_before_any_trial() {
        let mut entries = BTreeMap::new();
        entries.insert(
            key().line_key(),
            Choice {
                config: D,
                median_us: 321.0,
            },
        );
        let mut t = tuner(entries, DEFAULT_BUDGET);
        assert_eq!(drive(&mut t, key(), (A, &[B, C]), &time_of), (D, 0));
        let report = t.report();
        assert_eq!((report.shapes_cached, report.shapes_tuned), (1, 0));
        // Another key of the same file is not affected: it tunes.
        let other = TuneKey::new(true, KEY, (256, 2048, 8192));
        assert_eq!(drive(&mut t, other, (A, &[B, C]), &time_of).0, C);
        // A cached configuration this product cannot run falls back.
        let mut entries = BTreeMap::new();
        entries.insert(
            key().line_key(),
            Choice {
                config: D,
                median_us: 1.0,
            },
        );
        let mut t = tuner(entries, DEFAULT_BUDGET);
        let pick = t.pick(key(), 256, A, &|c| c != D, &Vec::new);
        assert_eq!(pick.config, A);
    }

    #[test]
    fn rows_share_a_key_within_their_bucket() {
        let bucket = |m| TuneKey::new(false, KEY, (m, 576, 576)).m_bucket();
        assert_eq!(
            [1, 64, 65, 128, 129, 192, 193, 256, 257, 512, 513, 1024, 1025, 2048, 2049, 9000]
                .map(m_bucket),
            [64, 64, 128, 128, 192, 192, 256, 256, 512, 512, 1024, 1024, 2048, 2048, 4096, 4096]
        );
        assert_eq!(bucket(200), bucket(256));
        assert_ne!(bucket(256), bucket(257));
        assert_ne!(
            TuneKey::new(false, KEY, (256, 576, 576)),
            TuneKey::new(true, KEY, (256, 576, 576))
        );
        assert_eq!(
            key().line_key(),
            "layout=a@bT;variant=Plain/AlphaBeta;m=256;n=2048;k=2048"
        );
    }

    #[test]
    fn candidates_are_tiles_that_fit_the_bucket_and_fill_the_gpu() {
        let all = |_: CoopTile| true;
        let got = candidates((256, 2048, 2048), 256, 160, &all);
        assert!(!got.is_empty());
        for c in &got {
            assert!(c.tile.bm <= 512, "{c:?}");
            assert!(c.split >= 1);
        }
        // Each tile appears once unsplit; no configuration twice.
        for (i, c) in got.iter().enumerate() {
            assert!(!got[..i].contains(c), "{c:?}");
        }
        // The register-prefetch tiles are candidates too.
        assert!(got.iter().any(|c| c.tile.prefetch), "{got:?}");
        // 256 x 2048 under a 128x128 tile is 16 groups: split K to fill 160.
        assert!(got.iter().any(|c| c.split > 1));
        // At 64 rows no tile has more than 128 rows.
        for c in candidates((64, 2048, 2048), 64, 160, &all) {
            assert!(c.tile.bm <= 128, "{c:?}");
        }
        // A tile that does not fit the device is never a candidate.
        let small = |t: CoopTile| t.bm * t.bn <= 32 * 64;
        for c in candidates((256, 2048, 2048), 256, 160, &small) {
            assert!(c.tile.bm * c.tile.bn <= 32 * 64, "{c:?}");
        }
        assert!(candidates((256, 2048, 2048), 256, 160, &|_| false).is_empty());
        // The first candidate fills the last wave best.
        let best = got.first().expect("a candidate");
        let (gx, gy) = best.tile.groups(256, 2048);
        let groups = u64::from(gx) * u64::from(gy) * u64::from(best.split);
        for c in &got {
            let (gx, gy) = c.tile.groups(256, 2048);
            let g = u64::from(gx) * u64::from(gy) * u64::from(c.split);
            assert!(fill(groups, 160) >= fill(g, 160) - 1e-12, "{c:?}");
        }
    }

    #[test]
    fn small_buckets_try_their_six_tiles_with_the_extra_splits() {
        let all = |_: CoopTile| true;
        // 135M q/o projection at 128 rows: 576 x 576 under a 128x64 tile is
        // 9 groups, so every split of K reaches the candidates.
        let got = candidates((128, 576, 576), 128, 160, &all);
        let tiles = tune_tiles(128);
        assert_eq!(tiles.len(), 6);
        for c in &got {
            assert!(tiles.contains(&c.tile), "{c:?}");
        }
        for &tile in tiles {
            let splits: Vec<u32> = got
                .iter()
                .filter(|c| c.tile == tile)
                .map(|c| c.split)
                .collect();
            assert!(splits.contains(&1), "{tile:?} {splits:?}");
            for s in [2, 3, 4] {
                // 576 / bk is at least 9 steps, so 2, 3 and 4 slices are
                // all distinct and whole.
                assert!(splits.contains(&split_k(576, tile.bk, s).0), "{tile:?} {s}");
            }
        }
        // No configuration twice, and the list fits the cap with the static pick.
        for (i, c) in got.iter().enumerate() {
            assert!(!got[..i].contains(c), "{c:?}");
        }
        assert!(got.len() < SMALL_MAX_CANDIDATES);
        // A product whose grid already fills the GPU is timed unsplit only:
        // 49152 columns under any of the 64-row tiles is 384 groups or more.
        for c in candidates((64, 576, 49152), 64, 160, &all) {
            assert_eq!(c.split, 1, "{c:?}");
        }
        // Every bucket of up to 256 rows has six distinct tiles that fit.
        for bucket in [64, 128, 192, 256] {
            let list = tune_tiles(bucket);
            assert_eq!(list.len(), 6, "{bucket}");
            for (i, t) in list.iter().enumerate() {
                assert!(!list[..i].contains(t), "{bucket} {t:?}");
            }
        }
        assert!(tune_tiles(512).is_empty());
    }

    #[test]
    fn large_buckets_keep_the_old_candidates_and_cap() {
        let all = |_: CoopTile| true;
        assert_eq!(max_candidates(256), SMALL_MAX_CANDIDATES);
        assert_eq!(max_candidates(512), MAX_CANDIDATES);
        // At 1024 rows a tile has K whole or K split to fill: never 2, 3, 4
        // slices unless that is the fill split.
        for c in candidates((1024, 2048, 2048), 1024, 160, &all) {
            let (gx, gy) = c.tile.groups(1024, 2048);
            let groups = u64::from(gx) * u64::from(gy);
            let fill_split = split_to_fill(c.tile, (1024, 2048, 2048), 160);
            assert!(
                c.split == 1 || (groups < 320 && c.split == fill_split),
                "{c:?}"
            );
        }
    }

    #[test]
    fn a_small_bucket_times_more_than_eight_candidates() {
        let mut t = tuner(BTreeMap::new(), DEFAULT_BUDGET);
        let all = |_: CoopTile| true;
        let others = candidates((128, 576, 576), 128, 160, &all);
        let small_key = TuneKey::new(false, KEY, (128, 576, 576));
        let (_, timed) = drive(&mut t, small_key, (A, &others), &time_of);
        assert!(timed > 8 * TIMINGS, "{timed}");
        let mut t = tuner(BTreeMap::new(), DEFAULT_BUDGET);
        let large_key = TuneKey::new(false, KEY, (1024, 576, 576));
        let (_, timed) = drive(&mut t, large_key, (A, &others), &time_of);
        assert!(timed <= 8 * TIMINGS + 8, "{timed}");
    }

    #[test]
    fn the_mode_follows_the_environment_value() {
        assert_eq!(parse_mode(None), Some(Mode::Off));
        assert_eq!(parse_mode(Some("")), Some(Mode::Off));
        assert_eq!(parse_mode(Some("0")), Some(Mode::Off));
        assert_eq!(parse_mode(Some(" 1 ")), Some(Mode::On));
        assert_eq!(parse_mode(Some("retune")), Some(Mode::Retune));
        assert_eq!(parse_mode(Some("yes")), None);
    }

    fn env(pairs: Vec<(&'static str, String)>) -> impl Fn(&str) -> Option<String> {
        move |name: &str| {
            pairs
                .iter()
                .find(|(k, _)| *k == name)
                .map(|(_, v)| v.clone())
        }
    }

    #[test]
    fn the_tuner_exists_only_when_asked_for_routed_and_timed() {
        let header = || HEADER.to_owned();
        let on = || env(vec![(AUTOTUNE_ENV, "1".to_owned())]);
        assert!(Tuner::from_env(&env(vec![]), (true, true), header).is_none());
        let off = env(vec![(AUTOTUNE_ENV, "0".to_owned())]);
        assert!(Tuner::from_env(&off, (true, true), header).is_none());
        let junk = env(vec![(AUTOTUNE_ENV, "fast".to_owned())]);
        assert!(Tuner::from_env(&junk, (true, true), header).is_none());
        assert!(Tuner::from_env(&on(), (false, true), header).is_none());
        assert!(Tuner::from_env(&on(), (true, false), header).is_none());
        let t = Tuner::from_env(&on(), (true, true), header).expect("on");
        assert_eq!(t.mode, Mode::On);
        assert_eq!(t.budget, DEFAULT_BUDGET);
    }

    #[test]
    fn the_budget_comes_from_the_environment() {
        let with = |v: &str| env(vec![(BUDGET_ENV, v.to_owned())]);
        assert_eq!(budget_from(&with("2.5")), Duration::from_millis(2500));
        for bad in ["0", "-1", "soon", "inf", "NaN", ""] {
            assert_eq!(budget_from(&with(bad)), DEFAULT_BUDGET, "{bad:?}");
        }
        assert_eq!(budget_from(&env(vec![])), DEFAULT_BUDGET);
    }

    #[test]
    fn a_saved_run_is_the_next_runs_cache_and_a_bad_file_is_ignored() {
        let dir = std::env::temp_dir().join(format!("ferrotensor-tuner-{}", std::process::id()));
        let path = dir.join("autotune.txt");
        let file = |p: &std::path::Path| {
            env(vec![
                (AUTOTUNE_ENV, "1".to_owned()),
                (autotune_cache::CACHE_ENV, p.display().to_string()),
            ])
        };
        let header = || HEADER.to_owned();
        let mut first = Tuner::from_env(&file(&path), (true, true), header).expect("on");
        assert_eq!(drive(&mut first, key(), (A, &[B, C, D]), &time_of).0, C);
        first.save();
        assert!(!first.dirty);
        drop(first);

        let mut second = Tuner::from_env(&file(&path), (true, true), header).expect("on");
        assert_eq!(drive(&mut second, key(), (A, &[B, C, D]), &time_of), (C, 0));
        let report = second.report();
        assert_eq!((report.shapes_cached, report.shapes_tuned), (1, 0));
        assert_eq!(report.cache_file.as_deref(), Some(path.as_path()));
        assert!(report.seconds.abs() < f64::EPSILON);

        // Another adapter's header, then a corrupt file: both tune afresh.
        let other = || "adapter=other".to_owned();
        let mut third = Tuner::from_env(&file(&path), (true, true), other).expect("on");
        assert_eq!(drive(&mut third, key(), (A, &[B, C, D]), &time_of).0, C);
        assert_eq!(third.report().shapes_tuned, 1);
        drop(third);
        std::fs::write(&path, "adapter=fake\ngarbage\n").expect("write garbage");
        let mut fourth = Tuner::from_env(&file(&path), (true, true), header).expect("on");
        assert_eq!(drive(&mut fourth, key(), (A, &[B, C, D]), &time_of).0, C);
        assert_eq!(fourth.report().shapes_tuned, 1);
        drop(fourth);

        // `retune` ignores a good file.
        let retune = env(vec![
            (AUTOTUNE_ENV, "retune".to_owned()),
            (autotune_cache::CACHE_ENV, path.display().to_string()),
        ]);
        let mut fifth = Tuner::from_env(&retune, (true, true), header).expect("retune");
        assert_eq!(drive(&mut fifth, key(), (A, &[B, C, D]), &time_of).0, C);
        assert_eq!(fifth.report().shapes_cached, 0);
        std::fs::remove_dir_all(&dir).expect("clean up");
    }
}
