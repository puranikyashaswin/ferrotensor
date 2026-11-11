//! GPU memory reports for training: bytes held at the peak of a phase
//! (model load, an optimizer step), split by [`MemoryCategory`].
//!
//! [`Trainer`](crate::Trainer) tags what it creates: the forward pass as
//! activations, the logits, the loss and the vocabulary-sized gradients
//! as loss, backward as scratch (with the parameters' gradients retagged
//! as gradients once backward returns) and the optimizer step as
//! optimizer state. Model weights are tagged by the caller: build the model inside a
//! [`MemoryCategory::BaseWeights`] scope and pass the trained parameters to
//! [`tag_trainable`].

use ferrotensor_autograd::Tensor;
use ferrotensor_backend::memory::{BufferRecord, MemorySnapshot};
use ferrotensor_backend::{DType, MemoryCategory, WgpuContext};
use serde_json::{json, Value};
use std::fmt::Write as _;
use std::sync::Arc;

/// What the report counts, printed with every table and stored in the JSON.
pub const MEMORY_REPORT_NOTE: &str = "Counts the buffers Ferrotensor allocated through wgpu \
(sizes after wgpu's 4-byte rounding). nvidia-smi and Activity Monitor show more: driver \
overhead, pipelines and shaders, uniform and upload staging buffers, and allocator slack \
are not counted.";

/// Count `params` (the tensors the optimizer updates) as trainable.
pub fn tag_trainable(params: &[Tensor]) {
    for p in params {
        p.data.set_memory_category(MemoryCategory::Trainable);
    }
}

/// Count the current gradients of `params` as parameter gradients.
pub fn tag_gradients(params: &[Tensor]) {
    for p in params {
        if let Some(g) = p.grad.borrow().as_ref() {
            g.set_memory_category(MemoryCategory::Gradients);
        }
    }
}

/// Count `logits` and, once backward has run, their gradient as loss
/// buffers.
pub fn tag_loss(logits: &Tensor) {
    logits.data.set_memory_category(MemoryCategory::Loss);
    if let Some(g) = logits.grad.borrow().as_ref() {
        g.set_memory_category(MemoryCategory::Loss);
    }
}

/// The bytes held during one phase.
#[derive(Debug, Clone)]
pub struct MemoryReport {
    /// Name of the phase, e.g. "after model load" or "optimizer step 1".
    pub label: String,
    /// At the highest tensor bytes of this phase.
    pub peak: MemorySnapshot,
    /// At the highest tensor bytes of all phases recorded so far.
    pub overall_peak: MemorySnapshot,
    /// When the phase was recorded.
    pub live: MemorySnapshot,
    /// Highest tensor plus idle pool bytes during this phase.
    pub peak_held_bytes: u64,
}

/// Records [`MemoryReport`]s of consecutive phases on one context.
#[derive(Debug)]
pub struct MemoryRecorder {
    ctx: Arc<WgpuContext>,
    overall: Option<MemorySnapshot>,
    /// Reports recorded so far, oldest first.
    pub reports: Vec<MemoryReport>,
}

impl MemoryRecorder {
    /// Turn on detailed tracking and start the first phase.
    pub fn start(ctx: Arc<WgpuContext>) -> Self {
        ctx.memory().set_detailed(true);
        Self {
            ctx,
            overall: None,
            reports: Vec::new(),
        }
    }

    /// Start a new phase without recording the current one.
    pub fn restart(&self) {
        self.ctx.memory().reset_peak();
    }

    /// End the current phase, record it as `label` and start the next.
    pub fn record(&mut self, label: impl Into<String>) -> &MemoryReport {
        let tracker = self.ctx.memory();
        let peak = tracker.peak().unwrap_or_else(|| tracker.current());
        if self
            .overall
            .as_ref()
            .is_none_or(|o| peak.live_bytes > o.live_bytes)
        {
            self.overall = Some(peak.clone());
        }
        let report = MemoryReport {
            label: label.into(),
            overall_peak: self.overall.clone().unwrap_or_default(),
            peak,
            live: tracker.current(),
            peak_held_bytes: tracker.peak_held_bytes(),
        };
        tracker.reset_peak();
        self.reports.push(report);
        self.reports.last().expect("just pushed")
    }

    /// Turn detailed tracking off again (the counters stay on).
    pub fn stop(&self) {
        self.ctx.memory().set_detailed(false);
    }

    /// Every report as JSON, with [`MEMORY_REPORT_NOTE`].
    pub fn to_json(&self) -> Value {
        json!({
            "note": MEMORY_REPORT_NOTE,
            "reports": self.reports.iter().map(MemoryReport::to_json).collect::<Vec<_>>(),
        })
    }
}

const MIB: f64 = 1024.0 * 1024.0;

fn mib(bytes: u64) -> f64 {
    bytes as f64 / MIB
}

fn dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::FP32 => "f32",
        DType::FP16 => "f16",
    }
}

fn snapshot_json(s: &MemorySnapshot) -> Value {
    let categories: serde_json::Map<String, Value> = MemoryCategory::ALL
        .iter()
        .map(|&c| {
            let b = s.category(c);
            (
                c.name().to_string(),
                json!({
                    "allocated_bytes": b.allocated_bytes,
                    "logical_bytes": b.logical_bytes,
                    "f16_bytes": b.f16_bytes,
                    "buffers": b.buffers,
                }),
            )
        })
        .collect();
    json!({
        "live_bytes": s.live_bytes,
        "logical_bytes": s.logical_bytes,
        "pool_idle_bytes": s.pool_idle_bytes,
        "categories": categories,
        "largest": s.largest.iter().map(record_json).collect::<Vec<_>>(),
    })
}

fn record_json(r: &BufferRecord) -> Value {
    json!({
        "category": r.category.name(),
        "dtype": dtype_name(r.dtype),
        "shape": r.shape,
        "allocated_bytes": r.allocated_bytes,
        "logical_bytes": r.logical_bytes,
    })
}

impl MemoryReport {
    /// The report as JSON (bytes, not MiB).
    pub fn to_json(&self) -> Value {
        json!({
            "label": self.label,
            "peak": snapshot_json(&self.peak),
            "overall_peak": snapshot_json(&self.overall_peak),
            "live": snapshot_json(&self.live),
            "peak_held_bytes": self.peak_held_bytes,
        })
    }

    /// The report as a text table in MiB.
    pub fn to_table(&self) -> String {
        let (p, o) = (&self.peak, &self.overall_peak);
        let mut t = String::new();
        let _ = writeln!(t, "GPU memory, {}", self.label);
        let _ = writeln!(t, "  {MEMORY_REPORT_NOTE}");
        let _ = writeln!(
            t,
            "  {:<34} {:>14} {:>14} {:>8}",
            "MiB", "phase peak", "overall peak", "buffers"
        );
        let mut row = |name: &str, a: u64, b: u64, n: Option<u64>| {
            let n = n.map_or(String::new(), |n| n.to_string());
            let _ = writeln!(t, "  {name:<34} {:>14.2} {:>14.2} {n:>8}", mib(a), mib(b));
        };
        for c in MemoryCategory::ALL {
            let (pc, oc) = (p.category(c), o.category(c));
            row(
                c.label(),
                pc.allocated_bytes,
                oc.allocated_bytes,
                Some(pc.buffers),
            );
            if c == MemoryCategory::BaseWeights {
                row("  of which f16", pc.f16_bytes, oc.f16_bytes, None);
                row(
                    "  of which f32",
                    pc.allocated_bytes - pc.f16_bytes,
                    oc.allocated_bytes - oc.f16_bytes,
                    None,
                );
            }
        }
        row("total (live tensors)", p.live_bytes, o.live_bytes, None);
        row(
            "pool idle (freed, kept for reuse)",
            p.pool_idle_bytes,
            o.pool_idle_bytes,
            None,
        );
        row(
            "total held",
            p.live_bytes + p.pool_idle_bytes,
            o.live_bytes + o.pool_idle_bytes,
            None,
        );
        let _ = writeln!(
            t,
            "  live tensors at the phase peak need {:.2} MiB; wgpu's rounding adds {:.2} MiB",
            mib(p.logical_bytes),
            mib(p.live_bytes - p.logical_bytes),
        );
        let _ = writeln!(
            t,
            "  highest live + pool idle in this phase: {:.2} MiB; now: {:.2} MiB live, {:.2} MiB idle",
            mib(self.peak_held_bytes),
            mib(self.live.live_bytes),
            mib(self.live.pool_idle_bytes),
        );
        if !p.largest.is_empty() {
            let _ = writeln!(t, "  largest live buffers at the phase peak:");
            for (i, r) in p.largest.iter().enumerate() {
                let _ = writeln!(
                    t,
                    "  {:>3}. {:>12.2} MiB  {:<26} {} {:?}",
                    i + 1,
                    mib(r.allocated_bytes),
                    r.category.label(),
                    dtype_name(r.dtype),
                    r.shape,
                );
            }
        }
        t
    }
}
