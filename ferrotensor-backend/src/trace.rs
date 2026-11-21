//! TEMP diagnostic (not for merge): host events and GPU kernel timestamps to a file.
use std::io::Write;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

struct Trace {
    start: Instant,
    out: std::io::BufWriter<std::fs::File>,
}

static TRACE: OnceLock<Option<Mutex<Trace>>> = OnceLock::new();

fn trace() -> Option<&'static Mutex<Trace>> {
    TRACE
        .get_or_init(|| {
            let path = std::env::var("FERROTENSOR_TRACE").ok()?;
            let file = std::fs::File::create(path).ok()?;
            Some(Mutex::new(Trace {
                start: Instant::now(),
                out: std::io::BufWriter::with_capacity(1 << 20, file),
            }))
        })
        .as_ref()
}

/// A host event with the time since the trace started.
pub fn host(kind: &str, detail: u64) {
    if let Some(t) = trace() {
        if let Ok(mut t) = t.lock() {
            let ns = t.start.elapsed().as_nanos();
            let _ = writeln!(t.out, "H {ns} {kind} {detail}");
        }
    }
}

/// A host mark from the trainer.
pub fn mark(kind: &str) {
    host(kind, 0);
}

pub(crate) fn gpu(serial: u64, index: usize, name: &str, t0_ns: f64, t1_ns: f64) {
    if let Some(t) = trace() {
        if let Ok(mut t) = t.lock() {
            let _ = writeln!(t.out, "G {serial} {index} {name} {t0_ns:.0} {t1_ns:.0}");
        }
    }
}

pub(crate) fn flush() {
    if let Some(t) = trace() {
        if let Ok(mut t) = t.lock() {
            let _ = t.out.flush();
        }
    }
}
