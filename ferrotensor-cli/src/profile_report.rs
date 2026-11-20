//! `--profile`. `train` counts the GPU work of each optimizer step and,
//! after the run, prints the step with the median wall time and keeps it as
//! JSON; `generate` reports the prefill and the mean decode step. Both add
//! the GPU time of each kernel when the adapter has timestamp queries.

use ferrotensor_backend::{
    GpuCounterSnapshot, HostSpan, KernelTimes, KernelTiming, KernelTimingMode, WgpuContext,
};
use serde_json::{json, Map, Value};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// File name of the JSON report in the run's output directory.
pub const FILE_NAME: &str = "profile.json";

/// The kernel timing a command with `--profile` opens its context with.
pub fn kernel_timing(profile: bool) -> KernelTiming {
    // Measurement build only: host spans without the per-dispatch timestamps.
    let kernels_off = std::env::var("FERROTENSOR_PROFILE_KERNELS").is_ok_and(|v| v == "0");
    if profile && !kernels_off {
        KernelTiming::On
    } else {
        KernelTiming::Off
    }
}

/// The next item of `items`, with the time it took counted as the host's
/// data preparation ([`HostSpan::Data`]).
pub fn timed_next<I: Iterator>(ctx: &WgpuContext, items: &mut I) -> Option<I::Item> {
    let _span = ctx.host_span(HostSpan::Data);
    items.next()
}

/// The host spans of one step as milliseconds, with what the spans leave of
/// the step (`other`), and the same inside readbacks, where `other` is what
/// the spans leave of the time in readbacks.
fn host_spans_ms(wall: Duration, work: &GpuCounterSnapshot) -> Value {
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    let mut spans = Map::new();
    let mut inside = Map::new();
    let (mut total, mut inside_total) = (Duration::ZERO, Duration::ZERO);
    for span in HostSpan::ALL {
        spans.insert(span.name().into(), ms(work.span(span)).into());
        inside.insert(span.name().into(), ms(work.readback_span(span)).into());
        total += work.span(span);
        inside_total += work.readback_span(span);
    }
    spans.insert("other".into(), ms(wall.saturating_sub(total)).into());
    inside.insert(
        "other".into(),
        ms(work.readback_time.saturating_sub(inside_total)).into(),
    );
    json!({
        "step": ms(wall),
        "readback": ms(work.readback_time),
        "spans": spans,
        "spans_in_readback": inside,
    })
}

/// A context's GPU work counters and kernel times at one moment, or the
/// work between two moments.
#[derive(Debug, Clone, Default)]
pub struct GpuMark {
    counters: GpuCounterSnapshot,
    kernels: KernelTimes,
    /// Buffers the pool created, and requests it served from an idle buffer.
    pool_created: u64,
    pool_reused: u64,
}

impl GpuMark {
    /// The totals of `ctx` now.
    pub fn now(ctx: &WgpuContext) -> Self {
        Self {
            counters: ctx.gpu_counters(),
            kernels: ctx.kernel_times(),
            pool_created: ctx.pool.buffers_created(),
            pool_reused: ctx.pool.buffers_reused(),
        }
    }

    fn since(&self, earlier: &Self) -> Self {
        Self {
            counters: self.counters.since(&earlier.counters),
            kernels: self.kernels.since(&earlier.kernels),
            pool_created: self.pool_created.saturating_sub(earlier.pool_created),
            pool_reused: self.pool_reused.saturating_sub(earlier.pool_reused),
        }
    }

    /// The pool counts with their report names.
    fn pool_counts(&self) -> [(&'static str, u64); 2] {
        [
            ("pool_buffers_created", self.pool_created),
            ("pool_buffers_reused", self.pool_reused),
        ]
    }
}

/// One optimizer step: its wall time and the GPU work done in it.
struct ProfiledStep {
    wall: Duration,
    work: GpuMark,
}

/// Per-step GPU work of one training run.
pub struct ProfileRun {
    ctx: Arc<WgpuContext>,
    step_start: Instant,
    at_step_start: GpuMark,
    steps: Vec<ProfiledStep>,
}

impl ProfileRun {
    /// Start the first step now.
    pub fn start(ctx: &Arc<WgpuContext>) -> Self {
        Self {
            at_step_start: GpuMark::now(ctx),
            ctx: ctx.clone(),
            step_start: Instant::now(),
            steps: Vec::new(),
        }
    }

    /// End the current optimizer step and start the next.
    pub fn after_step(&mut self) {
        let mark = GpuMark::now(&self.ctx);
        self.steps.push(ProfiledStep {
            wall: self.step_start.elapsed(),
            work: mark.since(&self.at_step_start),
        });
        self.step_start = Instant::now();
        self.at_step_start = mark;
    }

    /// Print the median step and write it to `json`.
    pub fn finish(mut self, json: Option<&Path>) -> std::io::Result<()> {
        self.steps.sort_by_key(|s| s.wall);
        let Some(median) = self.steps.get(self.steps.len() / 2) else {
            eprintln!("profile: no optimizer steps were taken");
            return Ok(());
        };
        let times = [
            ("step_seconds", "step wall time", median.wall),
            (
                "gpu_wait_seconds",
                "waiting on the GPU",
                median.work.counters.gpu_wait,
            ),
            (
                "readback_seconds",
                "in readbacks",
                median.work.counters.readback_time,
            ),
        ];
        let mut counts = median.work.counters.counts().to_vec();
        counts.extend(median.work.pool_counts());

        eprintln!(
            "profile: median of {} optimizer steps by wall time",
            self.steps.len()
        );
        for (_, label, time) in times {
            eprintln!("  {label:<26} {:>12.3} ms", time.as_secs_f64() * 1e3);
        }
        for &(name, count) in &counts {
            eprintln!("  {:<26} {count:>12} /step", name.replace('_', " "));
        }
        let host = host_spans_ms(median.wall, &median.work.counters);
        eprintln!("profile: host time by span in that step, ms (in readbacks / total)");
        for (name, ms) in host["spans"].as_object().into_iter().flatten() {
            let inside = host["spans_in_readback"][name].as_f64().unwrap_or(0.0);
            eprintln!(
                "  {name:<26} {inside:>12.3} / {:>8.3}",
                ms.as_f64().unwrap_or(0.0)
            );
        }
        let mode = self.ctx.kernel_timing_mode();
        if print_mode(mode) {
            eprintln!("profile: GPU time by kernel in that step");
            print_kernels(&median.work.kernels, 1.0, median.work.counters.gpu_wait);
        }

        if let Some(path) = json {
            let mut report = Map::new();
            report.insert("steps_profiled".into(), self.steps.len().into());
            for (key, _, time) in times {
                report.insert(key.into(), time.as_secs_f64().into());
            }
            for &(name, count) in &counts {
                report.insert(format!("{name}_per_step"), count.into());
            }
            report.insert("host_ms_per_step".into(), host);
            report.insert(
                "kernel_timing".into(),
                mode.map(KernelTimingMode::name).into(),
            );
            insert_kernels(&mut report, &median.work.kernels, 1.0, "_per_step");
            let text = serde_json::to_string_pretty(&report)?;
            std::fs::write(path, text + "\n")?;
            eprintln!("profile report: {}", path.display());
        }
        Ok(())
    }
}

/// `generate --profile`: prints the GPU work of the prefill (`start` to
/// `after_prefill`) and of the mean of `decode_steps` decode steps (to now),
/// and returns the same as JSON.
pub fn generate_report(
    ctx: &WgpuContext,
    start: &GpuMark,
    after_prefill: &GpuMark,
    decode_steps: usize,
) -> Value {
    let prefill = after_prefill.since(start);
    let decode = GpuMark::now(ctx).since(after_prefill);
    let tokens = decode_steps.max(1) as f64;
    let wait_per_token = Duration::from_secs_f64(decode.counters.gpu_wait.as_secs_f64() / tokens);
    let mode = ctx.kernel_timing_mode();
    if print_mode(mode) {
        eprintln!("profile: GPU time by kernel in the prefill");
        print_kernels(&prefill.kernels, 1.0, prefill.counters.gpu_wait);
        eprintln!("profile: GPU time by kernel per decode token, mean of {decode_steps}");
        print_kernels(&decode.kernels, tokens, wait_per_token);
    }

    let mut prefill_json = Map::new();
    prefill_json.insert(
        "gpu_wait_seconds".into(),
        prefill.counters.gpu_wait.as_secs_f64().into(),
    );
    insert_kernels(&mut prefill_json, &prefill.kernels, 1.0, "");
    let mut decode_json = Map::new();
    decode_json.insert("tokens".into(), decode_steps.into());
    decode_json.insert(
        "gpu_wait_seconds_per_token".into(),
        wait_per_token.as_secs_f64().into(),
    );
    insert_kernels(&mut decode_json, &decode.kernels, tokens, "_per_token");
    json!({
        "kernel_timing": mode.map(KernelTimingMode::name),
        "prefill": prefill_json,
        "decode": decode_json,
    })
}

/// Prints how kernels were timed, or that they were not; true when they were.
fn print_mode(mode: Option<KernelTimingMode>) -> bool {
    match mode {
        None => eprintln!("profile: the adapter has no timestamp queries; kernel times are not measured"),
        Some(KernelTimingMode::InsidePasses) => eprintln!(
            "profile: kernel timestamps before and after each dispatch, inside passes of up to 8 dispatches"
        ),
        Some(KernelTimingMode::PassBoundaries) => eprintln!(
            "profile: kernel timestamps at the start and end of passes of one dispatch each \
             (the adapter cannot write them inside a pass); passes and wall times differ from a run without --profile"
        ),
    }
    mode.is_some()
}

/// Prints `kernels` divided by `per`, longest GPU time first, and their sum
/// next to the measured wait for the GPU.
fn print_kernels(kernels: &KernelTimes, per: f64, gpu_wait: Duration) {
    let total = kernels.total().as_secs_f64();
    eprintln!(
        "  {:<44} {:>10} {:>12} {:>7}",
        "kernel", "dispatches", "GPU ms", "share"
    );
    for (name, time) in kernels.by_gpu_time() {
        let seconds = time.gpu_time.as_secs_f64();
        eprintln!(
            "  {name:<44} {:>10.1} {:>12.3} {:>6.1}%",
            time.dispatches as f64 / per,
            seconds / per * 1e3,
            if total > 0.0 {
                seconds / total * 100.0
            } else {
                0.0
            }
        );
    }
    eprintln!(
        "  sum of kernel times {:.3} ms; waiting on the GPU {:.3} ms",
        total / per * 1e3,
        gpu_wait.as_secs_f64() * 1e3
    );
}

/// Inserts `kernel_gpu_seconds{suffix}` and a `kernels` array (`name`,
/// `dispatches{suffix}`, `gpu_seconds{suffix}`) for `kernels` divided by
/// `per`, longest GPU time first.
fn insert_kernels(report: &mut Map<String, Value>, kernels: &KernelTimes, per: f64, suffix: &str) {
    report.insert(
        format!("kernel_gpu_seconds{suffix}"),
        (kernels.total().as_secs_f64() / per).into(),
    );
    let rows: Vec<Value> = kernels
        .by_gpu_time()
        .into_iter()
        .map(|(name, time)| {
            let mut row = Map::new();
            row.insert("name".into(), name.into());
            row.insert(
                format!("dispatches{suffix}"),
                (time.dispatches as f64 / per).into(),
            );
            row.insert(
                format!("gpu_seconds{suffix}"),
                (time.gpu_time.as_secs_f64() / per).into(),
            );
            Value::Object(row)
        })
        .collect();
    report.insert("kernels".into(), rows.into());
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotensor_backend::KernelTime;

    #[test]
    fn kernels_are_written_per_unit_longest_first() {
        let kernels: KernelTimes = [
            (
                "rope".to_owned(),
                KernelTime {
                    dispatches: 4,
                    gpu_time: Duration::from_millis(2),
                },
            ),
            (
                "matmul_blocked_h (a@bT, out>=1M)".to_owned(),
                KernelTime {
                    dispatches: 8,
                    gpu_time: Duration::from_millis(6),
                },
            ),
        ]
        .into_iter()
        .collect();
        let mut report = Map::new();
        insert_kernels(&mut report, &kernels, 2.0, "_per_token");
        assert_eq!(
            Value::Object(report),
            json!({
                "kernel_gpu_seconds_per_token": 0.004,
                "kernels": [
                    {
                        "name": "matmul_blocked_h (a@bT, out>=1M)",
                        "dispatches_per_token": 4.0,
                        "gpu_seconds_per_token": 0.003,
                    },
                    {
                        "name": "rope",
                        "dispatches_per_token": 2.0,
                        "gpu_seconds_per_token": 0.001,
                    },
                ],
            })
        );
    }

    #[test]
    fn host_spans_are_milliseconds_with_the_rest_as_other() {
        let mut work = GpuCounterSnapshot {
            readback_time: Duration::from_millis(30),
            ..GpuCounterSnapshot::default()
        };
        let index = |span: HostSpan| HostSpan::ALL.iter().position(|&s| s == span).unwrap_or(0);
        work.spans[index(HostSpan::Wait)] = Duration::from_millis(20);
        work.spans[index(HostSpan::Encode)] = Duration::from_millis(5);
        work.readback_spans[index(HostSpan::Wait)] = Duration::from_millis(20);
        work.readback_spans[index(HostSpan::Release)] = Duration::from_millis(4);
        work.spans[index(HostSpan::Release)] = Duration::from_millis(4);
        let report = host_spans_ms(Duration::from_millis(100), &work);
        let at = |value: &Value| value.as_f64().unwrap_or(f64::NAN);
        for (value, expected) in [
            (&report["step"], 100.0),
            (&report["readback"], 30.0),
            (&report["spans"]["wait"], 20.0),
            (&report["spans"]["encode"], 5.0),
            (&report["spans"]["release"], 4.0),
            (&report["spans"]["other"], 71.0),
            (&report["spans_in_readback"]["release"], 4.0),
            (&report["spans_in_readback"]["encode"], 0.0),
            (&report["spans_in_readback"]["other"], 6.0),
        ] {
            assert!((at(value) - expected).abs() < 1e-9, "{value} != {expected}");
        }
    }

    #[test]
    fn no_kernels_writes_an_empty_array() {
        let mut report = Map::new();
        insert_kernels(&mut report, &KernelTimes::default(), 1.0, "_per_step");
        assert_eq!(
            Value::Object(report),
            json!({"kernel_gpu_seconds_per_step": 0.0, "kernels": []})
        );
    }

    #[test]
    fn mark_since_subtracts_counters_and_kernels() {
        let earlier = GpuMark::default();
        let later = GpuMark {
            counters: GpuCounterSnapshot {
                dispatches: 3,
                ..GpuCounterSnapshot::default()
            },
            kernels: [(
                "rope".to_owned(),
                KernelTime {
                    dispatches: 3,
                    gpu_time: Duration::from_micros(30),
                },
            )]
            .into_iter()
            .collect(),
            pool_created: 5,
            pool_reused: 7,
        };
        let work = later.since(&earlier);
        assert_eq!(work.counters.dispatches, 3);
        assert_eq!(
            work.pool_counts(),
            [("pool_buffers_created", 5), ("pool_buffers_reused", 7)]
        );
        assert_eq!(work.kernels, later.kernels);
        assert_eq!(later.since(&later).kernels, KernelTimes::default());
    }
}
