//! What GPUs wgpu can see on this machine, and a quick correctness check of
//! the kernels on the one Ferrotensor uses.
//!
//! `ferrotensor info` prints [`list_adapters`] and, with `--check`, runs
//! [`self_check`]. Both are meant for bug reports and for trying Ferrotensor
//! on hardware it has not been tested on yet.

use crate::{ComputeEngine, UnifiedTensor, WgpuContext};

/// One adapter wgpu found, with the limits that matter for the kernels.
#[derive(Clone, Debug, PartialEq)]
pub struct AdapterReport {
    /// Adapter name reported by the driver, e.g. `"Apple M2"`.
    pub name: String,
    /// GPU vendor from the PCI vendor id, e.g. `"AMD"`, or the id in hex.
    pub vendor: String,
    /// `"DiscreteGpu"`, `"IntegratedGpu"`, `"Cpu"` (a software rasteriser), ...
    pub device_type: String,
    /// Graphics API: `"Metal"`, `"Vulkan"`, `"Dx12"` or `"Gl"`.
    pub backend: String,
    /// Driver name and version, when the backend reports them.
    pub driver: String,
    /// Largest single buffer, in bytes.
    pub max_buffer_size: u64,
    /// Largest storage buffer a kernel can bind, in bytes.
    pub max_storage_buffer_binding_size: u32,
    /// Most threads in one workgroup.
    pub max_compute_invocations_per_workgroup: u32,
    /// Workgroup shared memory, in bytes.
    pub max_compute_workgroup_storage_size: u32,
    /// Whether the adapter supports `f16` in shaders.
    pub shader_f16: bool,
    /// Whether [`WgpuContext::new`] picks this adapter.
    pub selected: bool,
}

/// Vendor name for a PCI vendor id.
pub fn vendor_name(id: u32) -> String {
    match id {
        0x1002 | 0x1022 => "AMD".into(),
        0x10de => "NVIDIA".into(),
        0x8086 => "Intel".into(),
        0x106b => "Apple".into(),
        0x5143 => "Qualcomm".into(),
        0x13b5 => "Arm".into(),
        0x1010 => "Imagination".into(),
        0x14e4 => "Broadcom".into(),
        0x10005 => "Mesa".into(),
        0 => "unknown".into(),
        other => format!("{other:#06x}"),
    }
}

fn report(adapter: &wgpu::Adapter, selected: Option<&wgpu::AdapterInfo>) -> AdapterReport {
    let info = adapter.get_info();
    let limits = adapter.limits();
    let driver = [info.driver.trim(), info.driver_info.trim()]
        .iter()
        .filter(|s| !s.is_empty())
        .copied()
        .collect::<Vec<_>>()
        .join(" ");
    AdapterReport {
        name: info.name.clone(),
        // Metal reports no PCI vendor id; every Metal GPU is Apple's.
        vendor: if info.vendor == 0 && info.backend == wgpu::Backend::Metal {
            "Apple".into()
        } else {
            vendor_name(info.vendor)
        },
        device_type: format!("{:?}", info.device_type),
        backend: format!("{:?}", info.backend),
        driver,
        max_buffer_size: limits.max_buffer_size,
        max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
        max_compute_invocations_per_workgroup: limits.max_compute_invocations_per_workgroup,
        max_compute_workgroup_storage_size: limits.max_compute_workgroup_storage_size,
        shader_f16: adapter.features().contains(wgpu::Features::SHADER_F16),
        selected: selected == Some(&info),
    }
}

/// Every adapter wgpu can open on this machine, across all backends, with the
/// one [`WgpuContext::new`] would pick marked `selected`.
pub fn list_adapters() -> Vec<AdapterReport> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let chosen = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .map(|a| a.get_info());
    instance
        .enumerate_adapters(wgpu::Backends::all())
        .iter()
        .map(|a| report(a, chosen.as_ref()))
        .collect()
}

/// Result of one kernel compared against a CPU reference.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckResult {
    /// What was run, e.g. `"matmul 37x53 @ 53x29"`.
    pub name: String,
    /// Largest absolute difference from the CPU result.
    pub max_abs_err: f32,
    /// Largest difference that still counts as a pass.
    pub tolerance: f32,
}

impl CheckResult {
    /// Whether the GPU result is within tolerance.
    pub fn passed(&self) -> bool {
        self.max_abs_err.is_finite() && self.max_abs_err <= self.tolerance
    }
}

/// Deterministic values in roughly [-1, 1], so the check needs no RNG.
fn values(n: usize, phase: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.37 + phase).sin() * ((i as f32) * 0.11).cos())
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::INFINITY;
    }
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// Run matmul (odd, non-square sizes), softmax and RMSNorm on `ctx`'s GPU
/// and compare each with a CPU reference computed in f64.
pub async fn self_check(ctx: &WgpuContext) -> Vec<CheckResult> {
    let engine = ComputeEngine::new(ctx);
    let mut results = Vec::new();

    // matmul with sizes that are not multiples of any tile size.
    let (m, k, n) = (37, 53, 29);
    let (a, b) = (values(m * k, 0.1), values(k * n, 0.7));
    let mut want = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let s: f64 = (0..k)
                .map(|p| a[i * k + p] as f64 * b[p * n + j] as f64)
                .sum();
            want[i * n + j] = s as f32;
        }
    }
    let (ga, gb) = (
        UnifiedTensor::new(ctx, &a, &[m, k]),
        UnifiedTensor::new(ctx, &b, &[k, n]),
    );
    let gc = UnifiedTensor::empty(ctx, &[m, n]);
    engine.matmul(ctx, &ga, &gb, &gc);
    results.push(CheckResult {
        name: format!("matmul {m}x{k} @ {k}x{n}"),
        max_abs_err: max_abs_diff(&gc.to_vec(ctx).await, &want),
        tolerance: 1e-4,
    });

    // softmax over rows longer than one workgroup.
    let (rows, cols) = (7, 1031);
    let x = values(rows * cols, 1.3);
    let mut want = vec![0f32; rows * cols];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let sum: f64 = row.iter().map(|&v| (v as f64 - max).exp()).sum();
        for c in 0..cols {
            want[r * cols + c] = ((row[c] as f64 - max).exp() / sum) as f32;
        }
    }
    let gx = UnifiedTensor::new(ctx, &x, &[rows, cols]);
    let gy = UnifiedTensor::empty(ctx, &[rows, cols]);
    engine.softmax(ctx, &gx, &gy);
    results.push(CheckResult {
        name: format!("softmax {rows}x{cols}"),
        max_abs_err: max_abs_diff(&gy.to_vec(ctx).await, &want),
        tolerance: 1e-6,
    });

    // RMSNorm with a non-trivial weight.
    let (rows, cols) = (5, 777);
    let (x, w) = (values(rows * cols, 2.1), values(cols, 0.4));
    let mut want = vec![0f32; rows * cols];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let ms: f64 = row.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / cols as f64;
        let inv = 1.0 / (ms + 1e-5).sqrt();
        for c in 0..cols {
            want[r * cols + c] = (row[c] as f64 * inv * w[c] as f64) as f32;
        }
    }
    let (gx, gw) = (
        UnifiedTensor::new(ctx, &x, &[rows, cols]),
        UnifiedTensor::new(ctx, &w, &[cols]),
    );
    let gy = UnifiedTensor::empty(ctx, &[rows, cols]);
    engine.rms_norm(ctx, &gx, &gw, &gy);
    results.push(CheckResult {
        name: format!("rms_norm {rows}x{cols}"),
        max_abs_err: max_abs_diff(&gy.to_vec(ctx).await, &want),
        tolerance: 1e-4,
    });

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vendor_ids_map_to_names() {
        assert_eq!(vendor_name(0x1002), "AMD");
        assert_eq!(vendor_name(0x10de), "NVIDIA");
        assert_eq!(vendor_name(0x8086), "Intel");
        assert_eq!(vendor_name(0x106b), "Apple");
        assert_eq!(vendor_name(0), "unknown");
        assert_eq!(vendor_name(0xabcd), "0xabcd");
    }

    #[test]
    fn check_result_pass_rule() {
        let r = |e| CheckResult {
            name: String::new(),
            max_abs_err: e,
            tolerance: 1e-4,
        };
        assert!(r(0.0).passed());
        assert!(r(1e-4).passed());
        assert!(!r(2e-4).passed());
        assert!(!r(f32::NAN).passed());
        assert!(!r(f32::INFINITY).passed());
    }

    #[test]
    fn diff_of_mismatched_lengths_fails() {
        assert_eq!(max_abs_diff(&[1.0], &[1.0, 2.0]), f32::INFINITY);
        assert_eq!(max_abs_diff(&[1.0, 2.0], &[1.5, 2.0]), 0.5);
    }
}
