//! What GPUs wgpu can see on this machine, and a quick correctness check of
//! the kernels on the one Ferrotensor uses.
//!
//! `ferrotensor info` prints [`list_adapters`] and, with `--check`, runs
//! [`self_check`]. Both are meant for bug reports and for trying Ferrotensor
//! on hardware it has not been tested on yet.

use crate::{ComputeEngine, KernelTimingMode, UnifiedTensor, WgpuContext};

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
    /// How `--profile` would time kernels here; `None` without timestamp
    /// queries.
    pub kernel_timing: Option<KernelTimingMode>,
    /// Cooperative-matrix shapes and types the adapter offers; empty without
    /// them.
    pub cooperative_matrix: Vec<CoopMatrixConfig>,
    /// Whether [`WgpuContext::new`] picks this adapter.
    pub selected: bool,
}

/// One cooperative-matrix configuration: an `M x K` times `K x N` product
/// with inputs of `ab_type` accumulated into `cr_type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoopMatrixConfig {
    /// Rows of A and of the result.
    pub m: u32,
    /// Columns of B and of the result.
    pub n: u32,
    /// Columns of A and rows of B.
    pub k: u32,
    /// Element type of A and B: `"f16"`, `"f32"`, `"i32"` or `"u32"`.
    pub ab_type: &'static str,
    /// Element type of the accumulator and the result.
    pub cr_type: &'static str,
    /// Whether the accumulation can saturate instead of overflowing.
    pub saturating_accumulation: bool,
}

fn scalar_name(ty: wgpu::CooperativeScalarType) -> &'static str {
    match ty {
        wgpu::CooperativeScalarType::F32 => "f32",
        wgpu::CooperativeScalarType::F16 => "f16",
        wgpu::CooperativeScalarType::I32 => "i32",
        wgpu::CooperativeScalarType::U32 => "u32",
    }
}

/// The cooperative-matrix configurations `adapter` offers, or none when it
/// lacks [`wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX`].
pub fn cooperative_matrix_configs(adapter: &wgpu::Adapter) -> Vec<CoopMatrixConfig> {
    if !adapter
        .features()
        .contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
    {
        return Vec::new();
    }
    adapter
        .cooperative_matrix_properties()
        .into_iter()
        .map(|p| CoopMatrixConfig {
            m: p.m_size,
            n: p.n_size,
            k: p.k_size,
            ab_type: scalar_name(p.ab_type),
            cr_type: scalar_name(p.cr_type),
            saturating_accumulation: p.saturating_accumulation,
        })
        .collect()
}

/// The optional features [`WgpuContext`] asks for when `adapter` has them:
/// `f16` in shaders and cooperative matrices. None of them is required.
pub fn optional_features(adapter: &wgpu::Adapter) -> wgpu::Features {
    adapter.features()
        & (wgpu::Features::SHADER_F16 | wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
}

/// The token wgpu needs before a device may enable an experimental feature
/// in `features`; disabled when there is none.
pub(crate) fn experimental_features(features: wgpu::Features) -> wgpu::ExperimentalFeatures {
    if features.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
        enable_experimental_features()
    } else {
        wgpu::ExperimentalFeatures::disabled()
    }
}

#[expect(
    unsafe_code,
    reason = "wgpu marks opting in to experimental features unsafe; see the SAFETY note"
)]
fn enable_experimental_features() -> wgpu::ExperimentalFeatures {
    // SAFETY: the opt-in only lets the device enable the experimental
    // cooperative-matrix feature. Ferrotensor dispatches no shader that uses
    // cooperative matrices, so none of that feature's code paths run.
    unsafe { wgpu::ExperimentalFeatures::enabled() }
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
        kernel_timing: KernelTimingMode::for_features(adapter.features()),
        cooperative_matrix: cooperative_matrix_configs(adapter),
        selected: selected == Some(&info),
    }
}

/// The backends to try: those named in `WGPU_BACKEND` (a comma list such as
/// `vulkan` or `metal,gl`), or all of them when it is unset.
pub fn backends() -> wgpu::Backends {
    wgpu::Backends::from_env().unwrap_or(wgpu::Backends::all())
}

/// A wgpu instance on `backends`, with the backend options (such as the DX12
/// shader compiler, `WGPU_DX12_COMPILER`) read from the environment.
pub fn instance(backends: wgpu::Backends) -> wgpu::Instance {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = backends;
    desc.backend_options = wgpu::BackendOptions::from_env_or_default();
    wgpu::Instance::new(desc)
}

/// Every adapter wgpu can open on this machine, across the [`backends`] in
/// use, with the one [`WgpuContext::new`] would pick marked `selected`.
pub fn list_adapters() -> Vec<AdapterReport> {
    let instance = instance(backends());
    let chosen = match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    })) {
        Ok(adapter) => Some(adapter.get_info()),
        // No adapter would be picked, so none is marked selected.
        Err(_) => None,
    };
    pollster::block_on(instance.enumerate_adapters(backends()))
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
