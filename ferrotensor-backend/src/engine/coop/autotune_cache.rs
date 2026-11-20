//! The autotuner's cache file: where it lives, what it holds and how it is
//! read and written.
//!
//! The file is plain text. The first line names the adapter, driver, backend,
//! Ferrotensor version and file schema the entries were timed with, and a
//! file whose first line differs from this run's is not used. Every other
//! line is one entry, `<key>\t<tile>;split=<n>;us=<median>`, where the key
//! names the product (see [`TuneKey::line_key`](super::autotune::TuneKey)).

use super::{CoopConfig, CoopTile};
use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

/// Environment variable that names the cache file.
pub(super) const CACHE_ENV: &str = "FERROTENSOR_AUTOTUNE_CACHE";

/// Version of the entry format; a change makes older files unusable.
const SCHEMA: u32 = 1;

/// The configuration a key settled on and its median time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Choice {
    pub(super) config: CoopConfig,
    /// Median GPU time of the winning configuration in training context.
    pub(super) median_us: f64,
}

/// Why a cache file was not used.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum CacheError {
    /// The first line is not this run's header.
    HeaderMismatch { found: String },
    /// A line cannot be read back: its 1-based number and what is wrong.
    Corrupt { line: usize, why: &'static str },
}

impl fmt::Display for CacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HeaderMismatch { found } => write!(
                f,
                "it was written for another adapter, driver or version ({found})"
            ),
            Self::Corrupt { line, why } => write!(f, "line {line}: {why}"),
        }
    }
}

/// The platform the cache path follows the conventions of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Platform {
    Windows,
    MacOs,
    Other,
}

impl Platform {
    pub(super) fn current() -> Self {
        if cfg!(windows) {
            Self::Windows
        } else if cfg!(target_os = "macos") {
            Self::MacOs
        } else {
            Self::Other
        }
    }
}

/// The cache file's path: `$FERROTENSOR_AUTOTUNE_CACHE`; else the user cache
/// directory of `platform` (`$XDG_CACHE_HOME` or `~/.cache` on Linux,
/// `~/Library/Caches` on macOS, `%LOCALAPPDATA%` on Windows) with
/// `ferrotensor/autotune.txt` below it. `None` when no home directory is
/// known; the tuner then keeps its results in memory only.
pub(super) fn cache_path(
    get: &dyn Fn(&str) -> Option<String>,
    platform: Platform,
) -> Option<PathBuf> {
    let set = |name: &str| get(name).filter(|v| !v.trim().is_empty());
    if let Some(path) = set(CACHE_ENV) {
        return Some(PathBuf::from(path));
    }
    let base = match platform {
        Platform::Windows => PathBuf::from(set("LOCALAPPDATA")?),
        Platform::MacOs => Path::new(&set("HOME")?).join("Library/Caches"),
        Platform::Other => match set("XDG_CACHE_HOME") {
            Some(dir) => PathBuf::from(dir),
            None => Path::new(&set("HOME")?).join(".cache"),
        },
    };
    Some(base.join("ferrotensor").join("autotune.txt"))
}

/// The header line for an adapter and backend, this Ferrotensor version and
/// this schema. The free-text fields lose their separators, so a driver
/// string cannot split the line.
pub(super) fn header_text(
    (name, vendor, device): (&str, u32, u32),
    (driver, driver_info): (&str, &str),
    backend: &str,
) -> String {
    let clean = |text: &str| text.replace([';', '\n', '\r', '\t'], " ");
    format!(
        "adapter={};vendor={vendor};device={device};driver={} {};backend={};ferrotensor={};schema={SCHEMA}",
        clean(name),
        clean(driver),
        clean(driver_info),
        clean(backend),
        env!("CARGO_PKG_VERSION"),
    )
}

/// The file's text for `entries`, under `header`.
pub(super) fn render(header: &str, entries: &BTreeMap<String, Choice>) -> String {
    let mut text = format!("{header}\n");
    for (key, choice) in entries {
        text.push_str(&format!(
            "{key}\t{};split={};us={:.3}\n",
            choice.config.tile.name(),
            choice.config.split,
            choice.median_us
        ));
    }
    text
}

/// The entries of the file `text` when its first line is `header`.
pub(super) fn parse(text: &str, header: &str) -> Result<BTreeMap<String, Choice>, CacheError> {
    let mut lines = text.lines();
    let first = lines.next().unwrap_or_default();
    if first != header {
        return Err(CacheError::HeaderMismatch {
            found: first.chars().take(200).collect(),
        });
    }
    let mut entries = BTreeMap::new();
    for (i, line) in lines.enumerate() {
        let corrupt = |why| CacheError::Corrupt { line: i + 2, why };
        let (key, value) = line.split_once('\t').ok_or_else(|| corrupt("no tab"))?;
        let mut fields = value.split(';');
        let tile = fields.next().ok_or_else(|| corrupt("no tile"))?;
        let tile = CoopTile::ALL
            .iter()
            .copied()
            .find(|t| t.name() == tile)
            .ok_or_else(|| corrupt("unknown tile"))?;
        let number = |field: Option<&str>, name: &str| {
            field
                .and_then(|f| f.strip_prefix(name))
                .map(str::to_owned)
                .ok_or_else(|| corrupt("missing field"))
        };
        let split: u32 = number(fields.next(), "split=")?
            .parse()
            .map_err(|_| corrupt("bad split"))?;
        let median_us: f64 = number(fields.next(), "us=")?
            .parse()
            .map_err(|_| corrupt("bad time"))?;
        if split == 0 || !median_us.is_finite() || median_us < 0.0 || fields.next().is_some() {
            return Err(corrupt("value out of range"));
        }
        let choice = Choice {
            config: CoopConfig { tile, split },
            median_us,
        };
        if entries.insert(key.to_owned(), choice).is_some() {
            return Err(corrupt("duplicate key"));
        }
    }
    Ok(entries)
}

/// The entries in the file at `path`, or none when it is absent, unreadable,
/// corrupt or written for another adapter, driver or version; any of the
/// last three is reported on stderr, and the file is replaced when the tuner
/// next saves.
pub(super) fn load(path: &Path, header: &str) -> BTreeMap<String, Choice> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return BTreeMap::new(),
        Err(e) => {
            log::warn!("autotune cache {} cannot be read: {e}", path.display());
            return BTreeMap::new();
        }
    };
    match parse(&text, header) {
        Ok(entries) => entries,
        Err(e) => {
            log::warn!("ignoring the autotune cache {}: {e}", path.display());
            BTreeMap::new()
        }
    }
}

/// Writes `text` to `path` through a temporary file in the same directory
/// and a rename, so a reader never sees half a file.
///
/// # Errors
///
/// The I/O error of creating the directory, writing the file or renaming it.
pub(super) fn write_atomic(path: &Path, text: &str) -> std::io::Result<()> {
    let dir = path.parent().filter(|d| !d.as_os_str().is_empty());
    if let Some(dir) = dir {
        std::fs::create_dir_all(dir)?;
    }
    let name = path
        .file_name()
        .map_or_else(|| "autotune".into(), |n| n.to_string_lossy().into_owned());
    let temp = path.with_file_name(format!(".{name}.{}.tmp", std::process::id()));
    let written = std::fs::write(&temp, text).and_then(|()| std::fs::rename(&temp, path));
    if written.is_err() {
        // The write or the rename failed, so the temporary is a leftover.
        match std::fs::remove_file(&temp) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => log::warn!("cannot remove {}: {e}", temp.display()),
        }
    }
    written
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    fn header() -> String {
        header_text(
            ("NVIDIA A10G", 0x10de, 0x2237),
            ("NVIDIA", "535.104.05"),
            "Vulkan",
        )
    }

    fn entries() -> BTreeMap<String, Choice> {
        let choice = |tile, split, median_us| Choice {
            config: CoopConfig { tile, split },
            median_us,
        };
        [
            (
                "layout=a@bT;variant=Plain/AlphaBeta;m=256;n=2048;k=2048".to_owned(),
                choice(CoopTile::T64X64_W32X32_K32_DB, 3, 123.456),
            ),
            (
                "layout=a@b;variant=Plain/AlphaBeta;m=2048;n=576;k=576".to_owned(),
                choice(CoopTile::T128X64_W64X32_K32, 1, 17.5),
            ),
        ]
        .into_iter()
        .collect()
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ferrotensor-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create the test directory");
        dir
    }

    #[test]
    fn header_names_the_device_version_and_schema_without_separators() {
        let h = header();
        assert!(
            h.starts_with("adapter=NVIDIA A10G;vendor=4318;device=8759;driver=NVIDIA 535.104.05;")
        );
        assert!(h.ends_with(&format!(
            ";backend=Vulkan;ferrotensor={};schema=1",
            env!("CARGO_PKG_VERSION")
        )));
        let messy = header_text(("a;b\nc", 1, 2), ("d\te", "f\r"), "g;h");
        assert_eq!(messy.lines().count(), 1);
        assert!(messy.starts_with("adapter=a b c;vendor=1;device=2;driver=d e f ;backend=g h;"));
    }

    #[test]
    fn entries_round_trip_through_the_text() {
        let text = render(&header(), &entries());
        let back = parse(&text, &header()).expect("parse what render wrote");
        // The time is written to three decimals.
        assert_eq!(back.len(), 2);
        for (key, want) in entries() {
            let got = back[&key];
            assert_eq!(got.config, want.config, "{key}");
            assert!((got.median_us - want.median_us).abs() < 1e-3, "{key}");
        }
        assert_eq!(render(&header(), &back), text);
    }

    #[test]
    fn a_header_of_another_device_discards_the_file() {
        let text = render(&header(), &entries());
        for other in [
            header_text(
                ("NVIDIA L40S", 0x10de, 0x26b9),
                ("NVIDIA", "535.104.05"),
                "Vulkan",
            ),
            header_text(
                ("NVIDIA A10G", 0x10de, 0x2237),
                ("NVIDIA", "550.54.15"),
                "Vulkan",
            ),
            header_text(
                ("NVIDIA A10G", 0x10de, 0x2237),
                ("NVIDIA", "535.104.05"),
                "Dx12",
            ),
            header().replace("schema=1", "schema=2"),
            header().replace(env!("CARGO_PKG_VERSION"), "0.0.1"),
        ] {
            let err = parse(&text, &other).expect_err("a different header");
            assert!(matches!(err, CacheError::HeaderMismatch { .. }), "{other}");
        }
        assert!(matches!(
            parse("", &header()),
            Err(CacheError::HeaderMismatch { .. })
        ));
    }

    #[test]
    fn a_corrupt_line_discards_the_file() {
        let good = render(&header(), &entries());
        let first_key = "layout=a@b;variant=Plain/AlphaBeta;m=2048;n=576;k=576";
        let bad = [
            "no tab at all",
            "key\tnot a tile;split=1;us=2",
            "key\t64x64/w32x32/k32/db;split=0;us=2",
            "key\t64x64/w32x32/k32/db;split=x;us=2",
            "key\t64x64/w32x32/k32/db;split=1;us=NaN",
            "key\t64x64/w32x32/k32/db;split=1;us=-1",
            "key\t64x64/w32x32/k32/db;split=1",
            "key\t64x64/w32x32/k32/db;split=1;us=2;extra=1",
            &format!("{first_key}\t64x64/w32x32/k32/db;split=1;us=2"),
        ];
        for line in bad {
            let text = format!("{good}{line}\n");
            let err = parse(&text, &header()).expect_err(line);
            assert!(
                matches!(err, CacheError::Corrupt { line: 4, .. }),
                "{line}: {err:?}"
            );
        }
        // Binary garbage in place of the entries.
        let garbage = format!("{}\n\u{0}\u{1}\u{2}\n", header());
        assert!(matches!(
            parse(&garbage, &header()),
            Err(CacheError::Corrupt { line: 2, .. })
        ));
    }

    #[test]
    fn load_ignores_a_missing_corrupt_or_mismatched_file() {
        let dir = temp_dir("autotune-load");
        let path = dir.join("autotune.txt");
        assert!(load(&path, &header()).is_empty());
        write_atomic(&path, &render(&header(), &entries())).expect("write");
        assert_eq!(load(&path, &header()).len(), 2);
        std::fs::write(&path, b"adapter=x\n\xff\xfe").expect("write garbage");
        assert!(load(&path, &header()).is_empty());
        std::fs::write(&path, "adapter=other\n").expect("write another header");
        assert!(load(&path, &header()).is_empty());
        std::fs::remove_dir_all(&dir).expect("clean up");
    }

    #[test]
    fn write_replaces_the_file_whole_and_leaves_no_temporary() {
        let dir = temp_dir("autotune-write");
        let path = dir.join("nested").join("autotune.txt");
        write_atomic(&path, "first\n").expect("write into a new directory");
        write_atomic(&path, "second\n").expect("replace");
        assert_eq!(std::fs::read_to_string(&path).expect("read"), "second\n");
        let names: Vec<_> = std::fs::read_dir(path.parent().expect("parent"))
            .expect("list")
            .map(|e| e.expect("entry").file_name())
            .collect();
        assert_eq!(names, ["autotune.txt"]);
        std::fs::remove_dir_all(&dir).expect("clean up");
    }

    #[test]
    fn the_path_follows_the_platform_and_the_override() {
        let env = |pairs: &'static [(&str, &str)]| {
            move |name: &str| {
                pairs
                    .iter()
                    .find(|(k, _)| *k == name)
                    .map(|(_, v)| (*v).to_owned())
            }
        };
        let full = env(&[
            ("FERROTENSOR_AUTOTUNE_CACHE", "/tmp/mine.txt"),
            ("XDG_CACHE_HOME", "/x"),
            ("HOME", "/home/u"),
        ]);
        for platform in [Platform::Windows, Platform::MacOs, Platform::Other] {
            assert_eq!(
                cache_path(&full, platform),
                Some(PathBuf::from("/tmp/mine.txt"))
            );
        }
        let xdg = env(&[("XDG_CACHE_HOME", "/x"), ("HOME", "/home/u")]);
        assert_eq!(
            cache_path(&xdg, Platform::Other),
            Some(PathBuf::from("/x/ferrotensor/autotune.txt"))
        );
        let home = env(&[("HOME", "/home/u"), ("XDG_CACHE_HOME", " ")]);
        assert_eq!(
            cache_path(&home, Platform::Other),
            Some(PathBuf::from("/home/u/.cache/ferrotensor/autotune.txt"))
        );
        assert_eq!(
            cache_path(&home, Platform::MacOs),
            Some(PathBuf::from(
                "/home/u/Library/Caches/ferrotensor/autotune.txt"
            ))
        );
        let windows = env(&[("LOCALAPPDATA", "C:/Users/u/AppData/Local")]);
        assert_eq!(
            cache_path(&windows, Platform::Windows),
            Some(PathBuf::from(
                "C:/Users/u/AppData/Local/ferrotensor/autotune.txt"
            ))
        );
        assert_eq!(cache_path(&env(&[]), Platform::Other), None);
        assert_eq!(cache_path(&env(&[]), Platform::Windows), None);
    }
}
