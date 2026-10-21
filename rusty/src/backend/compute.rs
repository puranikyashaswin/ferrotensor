//! Compute Engine - GPU Kernel Dispatch
//!
//! Compiles the WGSL kernels once per device and dispatches them.

use std::borrow::Cow;
use std::collections::HashMap;
use wgpu::{Buffer, ComputePipeline, Device, Queue};

use super::KERNELS_WGSL;

/// Entry points compiled from [`KERNELS_WGSL`].
pub(crate) const KERNEL_NAMES: [&str; 7] = [
    "unary",
    "binary",
    "reduce",
    "softmax",
    "matmul",
    "permute",
    "gather_rows",
];

/// Threads per workgroup of the 1-D kernels.
const WORKGROUP_1D: u32 = 256;
/// Largest workgroup count allowed in one grid dimension.
const MAX_GROUPS_PER_DIM: u32 = 65_535;

/// GPU compute engine: one compiled pipeline per kernel entry point.
pub struct ComputeEngine {
    pipelines: HashMap<&'static str, ComputePipeline>,
}

impl ComputeEngine {
    /// Compile all kernels for `device`.
    pub fn new(device: &Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Rusty Kernels"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(KERNELS_WGSL)),
        });

        let pipelines = KERNEL_NAMES
            .iter()
            .map(|&name| {
                let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(name),
                    layout: None,
                    module: &module,
                    entry_point: name,
                });
                (name, pipeline)
            })
            .collect();

        Self { pipelines }
    }

    /// Get a compiled pipeline by entry point name.
    pub fn get_pipeline(&self, name: &str) -> Option<&ComputePipeline> {
        self.pipelines.get(name)
    }

    /// Dispatch `kernel`, binding `buffers[i]` to `@binding(i)` of group 0.
    pub(crate) fn run(
        &self,
        device: &Device,
        queue: &Queue,
        kernel: &str,
        buffers: &[&Buffer],
        workgroups: (u32, u32, u32),
    ) {
        let pipeline = self
            .pipelines
            .get(kernel)
            .unwrap_or_else(|| panic!("kernel '{kernel}' is not compiled"));
        let layout = pipeline.get_bind_group_layout(0);
        let entries: Vec<wgpu::BindGroupEntry<'_>> = buffers
            .iter()
            .enumerate()
            .map(|(i, buffer)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(kernel),
            layout: &layout,
            entries: &entries,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(kernel),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(kernel),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
        }
        queue.submit(std::iter::once(encoder.finish()));
    }

    /// Workgroup grid for a 1-D kernel over `n` elements.
    ///
    /// Spills into the y dimension past 65535 workgroups; the kernels rebuild
    /// the flat index as `x + y * nx * 256`.
    pub fn workgroups_1d(n: usize) -> (u32, u32, u32) {
        let groups = (n as u32).div_ceil(WORKGROUP_1D).max(1);
        if groups <= MAX_GROUPS_PER_DIM {
            (groups, 1, 1)
        } else {
            let y = groups.div_ceil(MAX_GROUPS_PER_DIM);
            (MAX_GROUPS_PER_DIM, y, 1)
        }
    }

    /// Workgroup grid for the 16x16 tiled matmul kernel.
    pub fn workgroups_matmul(rows: usize, cols: usize, batch: usize) -> (u32, u32, u32) {
        (
            (cols as u32).div_ceil(16),
            (rows as u32).div_ceil(16),
            batch as u32,
        )
    }
}

impl std::fmt::Debug for ComputeEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComputeEngine")
            .field("num_pipelines", &self.pipelines.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parse and validate the WGSL with naga: catches shader errors on
    /// machines (and CI runners) without a GPU.
    #[test]
    fn kernels_pass_naga_validation() {
        let module = naga::front::wgsl::parse_str(KERNELS_WGSL)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(KERNELS_WGSL)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{:?}", e));

        for name in KERNEL_NAMES {
            assert!(
                module.entry_points.iter().any(|ep| ep.name == name),
                "missing entry point {name}"
            );
        }
    }

    #[test]
    fn workgroups_split_large_dispatches() {
        assert_eq!(ComputeEngine::workgroups_1d(0), (1, 1, 1));
        assert_eq!(ComputeEngine::workgroups_1d(256), (1, 1, 1));
        assert_eq!(ComputeEngine::workgroups_1d(257), (2, 1, 1));
        let (x, y, _) = ComputeEngine::workgroups_1d(256 * 70_000);
        assert_eq!(x, 65_535);
        assert!(x as usize * y as usize * 256 >= 256 * 70_000);
    }
}
