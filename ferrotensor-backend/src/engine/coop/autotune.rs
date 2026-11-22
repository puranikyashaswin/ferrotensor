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
//! already does. After [`TIMINGS`] timings of each candidate the key settles
//! on the one with the lowest median, unless the static pick is within
//! [`MARGIN`] of it. Settled keys cost one hash lookup per product.
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
use super::plan::{grid_fits, split_k, split_to_fill, ROW_STEP};
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

/// `nanos` that a dispatch of `rows` rows took, as it would take at the
/// `bucket`'s rows if the time grew with the rows. The candidates of a key are
/// timed on different steps, whose row counts differ by up to a factor of two
/// inside a bucket; comparing the raw times made a candidate that happened to
/// run on a short step look the fastest.
fn nanos_at_bucket_rows(nanos: u64, rows: u32, bucket: u32) -> u64 {
    let rows = u128::from(rows.div_ceil(ROW_STEP).max(1) * ROW_STEP);
    let reference = u128::from(bucket).max(rows);
    u64::try_from(u128::from(nanos) * reference / rows).unwrap_or(u64::MAX)
}

/// The index of the candidate to settle on and its median time in
/// nanoseconds, once every candidate has [`TIMINGS`] timings (`u64::MAX` for
/// a timing that was unusable). Candidate 0 is the static pick: another
/// candidate wins only with a median at least [`MARGIN`] below its.
fn winner(times: &[Vec<u64>]) -> Option<(usize, u64)> {
    if times.is_empty() || times.iter().any(|t| t.len() < TIMINGS) {
        return None;
    }
    let medians: Vec<u64> = times.iter().map(|t| median(t)).collect();
    let static_median = *medians.first()?;
    let (best, &best_median) = medians.iter().enumerate().min_by_key(|&(i, &m)| (m, i))?;
    let faster = (best_median as f64) < (static_median as f64) * (1.0 - MARGIN);
    Some(if faster {
        (best, best_median)
    } else {
        (0, static_median)
    })
}

/// A key's candidates and what has been timed so far.
struct Trial {
    candidates: Vec<CoopConfig>,
    /// Dispatches of each candidate so far.
    dispatched: Vec<usize>,
    /// Timings of each candidate in nanoseconds.
    times: Vec<Vec<u64>>,
}

/// A dispatch (or two, with K split) whose timestamps have not all arrived.
struct Open {
    key: TuneKey,
    candidate: usize,
    /// Rows of the product this dispatch ran.
    rows: u32,
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
            let count = list.len();
            self.trials.insert(
                key,
                Trial {
                    candidates: list,
                    dispatched: vec![0; count],
                    times: vec![Vec::new(); count],
                },
            );
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
        let Some(&config) = trial.candidates.get(candidate) else {
            return run(static_cfg);
        };
        if !usable(config) {
            return run(static_cfg);
        }
        if let Some(count) = trial.dispatched.get_mut(candidate) {
            *count += 1;
        }
        let sample = self.next_sample;
        self.next_sample += 1;
        let split = split_k(key.k, config.tile.bk, config.split).0;
        self.open.insert(
            sample,
            Open {
                key,
                candidate,
                rows,
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
        let value = if done.unusable {
            u64::MAX
        } else {
            nanos_at_bucket_rows(done.nanos, done.rows, done.key.m_bucket)
        };
        if let Some(trial) = self.trials.get_mut(&done.key) {
            if let Some(times) = trial.times.get_mut(done.candidate) {
                times.push(value);
            }
        }
        self.settle(done.key);
    }

    /// Settles `key` if every candidate has its timings.
    fn settle(&mut self, key: TuneKey) {
        let Some(trial) = self.trials.get(&key) else {
            return;
        };
        let Some((best, nanos)) = winner(&trial.times) else {
            return;
        };
        let Some(&config) = trial.candidates.get(best) else {
            return;
        };
        if let Some((&first, times)) = trial.candidates.first().zip(trial.times.first()) {
            let name = |c: CoopConfig| format!("{} split {}", c.tile.name(), c.split);
            log::info!(
                "autotune: {} chose {} ({} ns median) over the static {} ({} ns) from {} candidates",
                key.line_key(),
                name(config),
                nanos,
                name(first),
                median(times),
                trial.candidates.len()
            );
        }
        self.trials.remove(&key);
        self.settled.insert(key, config);
        self.tuned += 1;
        // A key whose timings were all unusable runs the static pick this run
        // and is not saved.
        if nanos != u64::MAX {
            let median_us = nanos as f64 / 1000.0;
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
    /// number of timed runs.
    fn drive(
        t: &mut Tuner,
        key: TuneKey,
        (static_cfg, others): (CoopConfig, &[CoopConfig]),
        nanos: &dyn Fn(CoopConfig) -> u64,
    ) -> (CoopConfig, usize) {
        let list = || others.to_vec();
        for timed in 0..1000 {
            let pick = t.pick(key, key.m_bucket(), static_cfg, &any, &list);
            let Some(sample) = pick.sample else {
                return (pick.config, timed);
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
        // 4 candidates, 3 timings each, all on real dispatches.
        assert_eq!(timed, 12);
        let report = t.report();
        assert_eq!((report.shapes_tuned, report.shapes_open), (1, 0));
        assert!(
            (report.seconds - 3.0 * 3.0e-3).abs() < 1e-9,
            "{}",
            report.seconds
        );
        // The key stays settled, and the file image holds it.
        assert_eq!(drive(&mut t, key(), (A, &[B, C, D]), &time_of), (C, 0));
        let entry = t.entries[&key().line_key()];
        assert_eq!(entry.config, C);
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
            winner(&[vec![100, 100, 100], vec![10, 90, 5_000]]),
            Some((1, 90))
        );
        assert_eq!(
            winner(&[vec![100, 100, 100], vec![10, 99, 5_000]]),
            Some((0, 100))
        );
        assert_eq!(winner(&[vec![100, 100]]), None);
        assert_eq!(winner(&[]), None);
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
    fn a_timing_is_scaled_to_the_rows_of_the_bucket() {
        assert_eq!(nanos_at_bucket_rows(100_000, 512, 1024), 200_000);
        assert_eq!(nanos_at_bucket_rows(100_000, 1024, 1024), 100_000);
        // Rows are rounded up like the router's, and a step past the
        // bucket's last row is not scaled down.
        assert_eq!(nanos_at_bucket_rows(100_000, 505, 512), 100_000);
        assert_eq!(nanos_at_bucket_rows(100_000, 4000, 4096), 102_400);
        assert_eq!(nanos_at_bucket_rows(100_000, 5000, 4096), 100_000);
    }

    #[test]
    fn unusable_timings_never_win() {
        let mut t = tuner(BTreeMap::new(), DEFAULT_BUDGET);
        let list = || vec![B];
        for _ in 0..TIMINGS {
            for _ in 0..2 {
                let pick = t.pick(key(), 256, A, &any, &list);
                let sample = pick.sample.expect("a timing");
                // B's timestamps ran backwards; A's are fine.
                let time = (pick.config == A).then(|| Duration::from_micros(10));
                t.record(sample, time);
            }
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
