//! `train --profile`: count the GPU work of each optimizer step and, after
//! the run, print the step with the median wall time and keep it as JSON.

use ferrotensor_backend::{GpuCounterSnapshot, WgpuContext};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// File name of the JSON report in the run's output directory.
pub const FILE_NAME: &str = "profile.json";

/// One optimizer step: its wall time and the GPU work done in it.
struct ProfiledStep {
    wall: Duration,
    work: GpuCounterSnapshot,
}

/// Per-step GPU work of one training run.
pub struct ProfileRun {
    ctx: Arc<WgpuContext>,
    step_start: Instant,
    counters_at_start: GpuCounterSnapshot,
    steps: Vec<ProfiledStep>,
}

impl ProfileRun {
    /// Start the first step now.
    pub fn start(ctx: &Arc<WgpuContext>) -> Self {
        Self {
            counters_at_start: ctx.gpu_counters(),
            ctx: ctx.clone(),
            step_start: Instant::now(),
            steps: Vec::new(),
        }
    }

    /// End the current optimizer step and start the next.
    pub fn after_step(&mut self) {
        let counters = self.ctx.gpu_counters();
        self.steps.push(ProfiledStep {
            wall: self.step_start.elapsed(),
            work: counters.since(&self.counters_at_start),
        });
        self.step_start = Instant::now();
        self.counters_at_start = counters;
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
                median.work.gpu_wait,
            ),
            (
                "readback_seconds",
                "in readbacks",
                median.work.readback_time,
            ),
        ];
        let counts = median.work.counts();

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

        if let Some(path) = json {
            let mut report = serde_json::Map::new();
            report.insert("steps_profiled".into(), self.steps.len().into());
            for (key, _, time) in times {
                report.insert(key.into(), time.as_secs_f64().into());
            }
            for (name, count) in counts {
                report.insert(format!("{name}_per_step"), count.into());
            }
            let text = serde_json::to_string_pretty(&report)?;
            std::fs::write(path, text + "\n")?;
            eprintln!("profile report: {}", path.display());
        }
        Ok(())
    }
}
