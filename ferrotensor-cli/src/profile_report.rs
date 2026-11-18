//! `--profile`. `train` counts the GPU work of each optimizer step and,
//! after the run, prints the step with the median wall time and keeps it as
//! JSON; `generate` reports the prefill and the mean decode step. Both add
//! the GPU time of each kernel when the adapter has timestamp queries.

use ferrotensor_backend::{
    GpuCounterSnapshot, KernelTimes, KernelTiming, KernelTimingMode, MatmulShapes, WgpuContext,
};
use serde_json::{json, Map, Value};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// File name of the JSON report in the run's output directory.
pub const FILE_NAME: &str = "profile.json";

/// The kernel timing a command with `--profile` opens its context with.
pub fn kernel_timing(profile: bool) -> KernelTiming {
    if profile {
        KernelTiming::On
    } else {
        KernelTiming::Off
    }
}

/// A context's GPU work counters and kernel times at one moment, or the
/// work between two moments.
#[derive(Debug, Clone, Default)]
pub struct GpuMark {
    counters: GpuCounterSnapshot,
    kernels: KernelTimes,
    matmul_shapes: MatmulShapes,
}

impl GpuMark {
    /// The totals of `ctx` now.
    pub fn now(ctx: &WgpuContext) -> Self {
        Self {
            counters: ctx.gpu_counters(),
            kernels: ctx.kernel_times(),
            matmul_shapes: ctx.matmul_shapes(),
        }
    }

    fn since(&self, earlier: &Self) -> Self {
        Self {
            counters: self.counters.since(&earlier.counters),
            kernels: self.kernels.since(&earlier.kernels),
            matmul_shapes: self.matmul_shapes.since(&earlier.matmul_shapes),
        }
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
        let counts = median.work.counters.counts();

        eprintln!(
            "profile: median of {} optimizer steps by wall time",
            self.steps.len()
        );
        for (_, label, time) in times {
            eprintln!("  {label:<26} {:>12.3} ms", time.as_secs_f64() * 1e3);
        }
        for (name, count) in counts {
            eprintln!("  {:<26} {count:>12} /step", name.replace('_', " "));
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
            for (name, count) in counts {
                report.insert(format!("{name}_per_step"), count.into());
            }
            report.insert(
                "kernel_timing".into(),
                mode.map(KernelTimingMode::name).into(),
            );
            insert_kernels(&mut report, &median.work.kernels, 1.0, "_per_step");
            insert_matmul_shapes(&mut report, &median.work.matmul_shapes);
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

/// Inserts `matmul_shapes`: one object per distinct matmul (`entry`,
/// `layout`, `m`, `n`, `k`) with its `dispatches_per_step`,
/// `gpu_seconds_per_step` and `tflops` (`2 m n k` per dispatch over the GPU
/// time; null when that time is zero), longest GPU time first. `m`, `n` and
/// `k` are those of `out[m, n] = op(a) @ op(b)`; a batch is folded into `m`.
fn insert_matmul_shapes(report: &mut Map<String, Value>, shapes: &MatmulShapes) {
    let rows: Vec<Value> = shapes
        .by_gpu_time()
        .into_iter()
        .map(|(shape, time)| {
            let seconds = time.gpu_time.as_secs_f64();
            let flops = 2.0
                * f64::from(shape.m)
                * f64::from(shape.n)
                * f64::from(shape.k)
                * time.dispatches as f64;
            let tflops = (seconds > 0.0).then(|| flops / seconds / 1e12);
            json!({
                "entry": shape.entry,
                "layout": shape.layout,
                "m": shape.m,
                "n": shape.n,
                "k": shape.k,
                "dispatches_per_step": time.dispatches as f64,
                "gpu_seconds_per_step": seconds,
                "tflops": tflops,
            })
        })
        .collect();
    report.insert("matmul_shapes".into(), rows.into());
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use ferrotensor_backend::{KernelTime, MatmulShape};

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
    fn matmul_shapes_give_tflops_and_sort_longest_first() {
        let shape = |entry, m, n, k| MatmulShape {
            entry,
            layout: "a@b",
            m,
            n,
            k,
        };
        let shapes: MatmulShapes = [
            // Two dispatches of 2 * 66 * 2048 * 2048 flops each, 2 ms in all.
            (
                shape("matmul_thin", 66, 2048, 2048),
                KernelTime {
                    dispatches: 2,
                    gpu_time: Duration::from_millis(2),
                },
            ),
            // Longer in total, so first.
            (
                shape("matmul_blocked", 512, 256, 128),
                KernelTime {
                    dispatches: 1,
                    gpu_time: Duration::from_millis(4),
                },
            ),
            // A zero GPU time has no rate.
            (
                shape("matmul_blocked", 8, 8, 8),
                KernelTime {
                    dispatches: 3,
                    gpu_time: Duration::ZERO,
                },
            ),
        ]
        .into_iter()
        .collect();
        let mut report = Map::new();
        insert_matmul_shapes(&mut report, &shapes);
        let rows = report["matmul_shapes"].as_array().expect("an array");
        assert_eq!(rows.len(), 3);

        assert_eq!(rows[0]["entry"], "matmul_blocked");
        assert_eq!(rows[0]["m"], 512);
        assert_eq!(rows[0]["gpu_seconds_per_step"], 0.004);
        // 2 * 512 * 256 * 128 = 33_554_432 flops in 4 ms.
        let tflops = rows[0]["tflops"].as_f64().expect("a rate");
        assert!((tflops - 0.008_388_608).abs() < 1e-12, "{tflops}");

        let expected = json!({
            "entry": "matmul_thin",
            "layout": "a@b",
            "m": 66,
            "n": 2048,
            "k": 2048,
            "dispatches_per_step": 2.0,
            "gpu_seconds_per_step": 0.002,
        });
        for (key, value) in expected.as_object().expect("an object") {
            assert_eq!(&rows[1][key], value, "{key}");
        }
        // 2 dispatches * 553_648_128 flops in 2 ms.
        let tflops = rows[1]["tflops"].as_f64().expect("a rate");
        assert!((tflops - 0.553_648_128).abs() < 1e-12, "{tflops}");

        assert_eq!(rows[2]["dispatches_per_step"], 3.0);
        assert_eq!(rows[2]["tflops"], Value::Null);
    }

    #[test]
    fn no_matmul_shapes_writes_an_empty_array() {
        let mut report = Map::new();
        insert_matmul_shapes(&mut report, &MatmulShapes::default());
        assert_eq!(Value::Object(report), json!({"matmul_shapes": []}));
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
            matmul_shapes: MatmulShapes::default(),
        };
        let work = later.since(&earlier);
        assert_eq!(work.counters.dispatches, 3);
        assert_eq!(work.kernels, later.kernels);
        assert_eq!(later.since(&later).kernels, KernelTimes::default());
    }
}
