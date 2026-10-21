//! Multi-rank gradient averaging for Tier-2 training.
//!
//! Each process/rank:
//! 1. writes local gradients into a shared `sync_dir`
//! 2. rank 0 averages them and writes the result
//! 3. every rank loads the averaged gradients before `optimizer.step`
//!
//! This is slower than GPU collectives, but it is a working Python-free
//! multi-process path without pulling in MPI/NCCL.

use rusty_backend::{UnifiedTensor, WgpuContext};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct FileGradSyncConfig {
    pub world_size: usize,
    pub rank: usize,
    pub sync_dir: PathBuf,
    pub max_wait: Duration,
}

impl FileGradSyncConfig {
    pub fn new(
        world_size: usize,
        rank: usize,
        sync_dir: impl Into<PathBuf>,
    ) -> Result<Self, String> {
        if world_size == 0 {
            return Err("world_size must be >= 1".into());
        }
        if rank >= world_size {
            return Err(format!(
                "rank {rank} is out of range for world_size {world_size}"
            ));
        }
        Ok(Self {
            world_size,
            rank,
            sync_dir: sync_dir.into(),
            max_wait: Duration::from_secs(600),
        })
    }

    pub fn with_max_wait(mut self, max_wait: Duration) -> Self {
        self.max_wait = max_wait;
        self
    }

    pub fn is_root(&self) -> bool {
        self.rank == 0
    }

    pub fn enabled(&self) -> bool {
        self.world_size > 1
    }
}

fn atomic_write_bytes(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let tmp = parent.join(format!(
        ".{}.tmp.{}",
        path.file_name().and_then(|n| n.to_str()).unwrap_or("grad"),
        std::process::id()
    ));
    fs::write(&tmp, bytes)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

fn wait_for_path(path: &Path, max_wait: Duration) -> bool {
    let start = Instant::now();
    while start.elapsed() < max_wait {
        if path.exists() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    path.exists()
}

fn rank_grad_path(step_dir: &Path, rank: usize, param_idx: usize) -> PathBuf {
    step_dir.join(format!("rank{rank}_param{param_idx}.bin"))
}

fn avg_grad_path(step_dir: &Path, param_idx: usize) -> PathBuf {
    step_dir.join(format!("avg_param{param_idx}.bin"))
}

fn step_dir(cfg: &FileGradSyncConfig, step: usize) -> PathBuf {
    cfg.sync_dir.join(format!("step_{step}"))
}

/// Average `grads` across ranks and overwrite them in place.
pub fn average_grads_file_sync(
    grads: &mut [UnifiedTensor],
    ctx: &WgpuContext,
    cfg: &FileGradSyncConfig,
    step: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if !cfg.enabled() {
        return Ok(());
    }

    let step_dir = step_dir(cfg, step);
    if cfg.is_root() {
        fs::create_dir_all(&step_dir)?;
        // Marker so other ranks know the step directory is ready.
        atomic_write_bytes(&step_dir.join(".ready"), b"1")?;
    }
    let ready = step_dir.join(".ready");
    if !wait_for_path(&ready, cfg.max_wait) {
        return Err(format!("timed out waiting for {ready:?}").into());
    }

    let param_count = grads.len();
    for (param_idx, grad) in grads.iter().enumerate() {
        let values = pollster::block_on(grad.to_vec(ctx));
        let bytes: Vec<u8> = bytemuck::cast_slice(&values).to_vec();
        atomic_write_bytes(&rank_grad_path(&step_dir, cfg.rank, param_idx), &bytes)?;
    }

    if cfg.is_root() {
        for param_idx in 0..param_count {
            let mut sum: Option<Vec<f32>> = None;
            for rank in 0..cfg.world_size {
                let path = rank_grad_path(&step_dir, rank, param_idx);
                if !wait_for_path(&path, cfg.max_wait) {
                    return Err(format!("timed out waiting for {path:?}").into());
                }
                let bytes = fs::read(&path)?;
                if bytes.len() % 4 != 0 {
                    return Err(format!("corrupt gradient file {path:?}").into());
                }
                let values: &[f32] = bytemuck::cast_slice(&bytes);
                match sum.as_mut() {
                    None => sum = Some(values.to_vec()),
                    Some(acc) => {
                        if acc.len() != values.len() {
                            return Err(format!(
                                "gradient length mismatch for param {param_idx}: expected {}, got {}",
                                acc.len(),
                                values.len()
                            )
                            .into());
                        }
                        for (dst, src) in acc.iter_mut().zip(values.iter()) {
                            *dst += *src;
                        }
                    }
                }
            }
            let mut avg = sum.ok_or("missing gradients for average")?;
            let scale = 1.0 / cfg.world_size as f32;
            for v in &mut avg {
                *v *= scale;
            }
            let avg_bytes: Vec<u8> = bytemuck::cast_slice(&avg).to_vec();
            atomic_write_bytes(&avg_grad_path(&step_dir, param_idx), &avg_bytes)?;
        }
    }

    for param_idx in 0..param_count {
        let path = avg_grad_path(&step_dir, param_idx);
        if !wait_for_path(&path, cfg.max_wait) {
            return Err(format!("timed out waiting for {path:?}").into());
        }
        let bytes = fs::read(&path)?;
        grads[param_idx].write_data(ctx, &bytes);
    }

    Ok(())
}

/// Average host-side parameter vectors (unit-testable without GPU).
pub fn average_host_grads(
    per_rank: &[Vec<Vec<f32>>],
) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    if per_rank.is_empty() {
        return Err("no ranks provided".into());
    }
    let world = per_rank.len();
    let param_count = per_rank[0].len();
    let mut out = Vec::with_capacity(param_count);
    for param_idx in 0..param_count {
        let expected = per_rank[0][param_idx].len();
        let mut acc = vec![0.0f32; expected];
        for rank_grads in per_rank {
            if rank_grads.len() != param_count {
                return Err("rank param count mismatch".into());
            }
            let values = &rank_grads[param_idx];
            if values.len() != expected {
                return Err("rank gradient length mismatch".into());
            }
            for (dst, src) in acc.iter_mut().zip(values.iter()) {
                *dst += *src;
            }
        }
        let scale = 1.0 / world as f32;
        for v in &mut acc {
            *v *= scale;
        }
        out.push(acc);
    }
    Ok(out)
}

/// Select sample indices owned by `rank` in a world of `world_size`.
pub fn shard_indices(len: usize, rank: usize, world_size: usize) -> Vec<usize> {
    if world_size <= 1 {
        return (0..len).collect();
    }
    (0..len).filter(|i| i % world_size == rank).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn config_rejects_invalid_rank() {
        assert!(FileGradSyncConfig::new(2, 2, "/tmp/x").is_err());
        assert!(FileGradSyncConfig::new(0, 0, "/tmp/x").is_err());
        assert!(FileGradSyncConfig::new(2, 1, "/tmp/x").is_ok());
    }

    #[test]
    fn host_grad_average_is_mean() {
        let ranks = vec![
            vec![vec![1.0, 3.0], vec![10.0]],
            vec![vec![5.0, 7.0], vec![30.0]],
        ];
        let avg = average_host_grads(&ranks).unwrap();
        assert_eq!(avg[0], vec![3.0, 5.0]);
        assert_eq!(avg[1], vec![20.0]);
    }

    #[test]
    fn shard_indices_partition_samples() {
        let all: Vec<_> = (0..8).collect();
        let r0 = shard_indices(8, 0, 2);
        let r1 = shard_indices(8, 1, 2);
        assert_eq!(r0, vec![0, 2, 4, 6]);
        assert_eq!(r1, vec![1, 3, 5, 7]);
        let mut merged = r0;
        merged.extend(r1);
        merged.sort_unstable();
        assert_eq!(merged, all);
    }

    #[test]
    fn wait_for_path_times_out() {
        let missing = std::env::temp_dir().join("rusty_missing_marker_should_not_exist");
        let _ = fs::remove_file(&missing);
        assert!(!wait_for_path(&missing, Duration::from_millis(40)));
    }

    #[test]
    fn atomic_write_and_ready_marker_roundtrip() {
        let dir = std::env::temp_dir().join(format!("rusty_sync_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let marker = dir.join(".ready");
        atomic_write_bytes(&marker, b"1").unwrap();
        assert!(wait_for_path(&marker, Duration::from_millis(100)));
        assert_eq!(fs::read(&marker).unwrap(), b"1");
        let _ = fs::remove_dir_all(&dir);
    }
}
