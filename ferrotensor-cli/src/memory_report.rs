//! `train --memory-report`: print where GPU memory goes after the model
//! loads and over the first optimizer steps, and keep it as JSON.

use ferrotensor_backend::WgpuContext;
use ferrotensor_trainer::memory::MemoryRecorder;
use std::path::PathBuf;
use std::sync::Arc;

/// File name of the JSON report in the run's output directory.
pub const FILE_NAME: &str = "memory_report.json";

/// Reports of one training run.
pub struct MemoryReportRun {
    recorder: MemoryRecorder,
    steps: usize,
    json: Option<PathBuf>,
}

impl MemoryReportRun {
    /// Start counting on `ctx` (before the model is built), to report the
    /// first `steps` optimizer steps and write JSON to `json`.
    pub fn start(ctx: &Arc<WgpuContext>, steps: usize, json: Option<PathBuf>) -> Self {
        Self {
            recorder: MemoryRecorder::start(ctx.clone()),
            steps,
            json,
        }
    }

    /// Print and store the phase that just ended.
    pub fn record(&mut self, label: &str) -> std::io::Result<()> {
        eprint!("{}", self.recorder.record(label).to_table());
        if let Some(path) = &self.json {
            let text = serde_json::to_string_pretty(&self.recorder.to_json())?;
            std::fs::write(path, text + "\n")?;
        }
        Ok(())
    }

    /// Record the model load.
    pub fn after_load(&mut self) -> std::io::Result<()> {
        self.record("after model load")?;
        if self.steps == 0 {
            self.recorder.stop();
        }
        Ok(())
    }

    /// Start the first step's phase (after evaluation and resume, which
    /// are not part of a step).
    pub fn begin_steps(&self) {
        self.recorder.restart();
    }

    /// Call after optimizer step `step` of `done` steps taken in this run.
    pub fn after_step(&mut self, step: usize, done: usize) -> std::io::Result<()> {
        if done > self.steps {
            return Ok(());
        }
        self.record(&format!("optimizer step {step}"))?;
        if done == self.steps {
            self.recorder.stop();
            if let Some(path) = &self.json {
                eprintln!("memory report: {}", path.display());
            }
        }
        Ok(())
    }
}
