//! Compute Engine - GPU Kernel Dispatch
//!
//! Compiles the WGSL kernels once per device and dispatches them.

use std::borrow::Cow;
use std::collections::HashMap;
use wgpu::{BindGroup, BindGroupLayout, BindingResource, CommandEncoder, ComputePipeline, Device};

use super::KERNELS_WGSL;

/// Entry points compiled from [`KERNELS_WGSL`].
pub(crate) const KERNEL_NAMES: [&str; 9] = [
    "unary",
    "binary",
    "reduce",
    "reduce_all",
    "arg_reduce",
    "softmax",
    "matmul",
    "permute",
    "gather_rows",
];

/// Threads per workgroup of the 1-D kernels.
const WORKGROUP_1D: u32 = 256;
/// Largest workgroup count allowed in one grid dimension.
const MAX_GROUPS_PER_DIM: u32 = 65_535;

/// A kernel dispatch waiting to be encoded.
pub(crate) struct Dispatch {
    kernel: &'static str,
    bind_group: BindGroup,
    workgroups: (u32, u32, u32),
}

/// GPU compute engine: one compiled pipeline per kernel entry point, with its
/// bind group layout.
pub struct ComputeEngine {
    pipelines: HashMap<&'static str, (ComputePipeline, BindGroupLayout)>,
}

impl ComputeEngine {
    /// Compile all kernels for `device`.
    pub fn new(device: &Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Ferrotensor Kernels"),
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
                let layout = pipeline.get_bind_group_layout(0);
                (name, (pipeline, layout))
            })
            .collect();

        Self { pipelines }
    }

    /// Get a compiled pipeline by entry point name.
    pub fn get_pipeline(&self, name: &str) -> Option<&ComputePipeline> {
        self.pipelines.get(name).map(|(pipeline, _)| pipeline)
    }

    /// A dispatch of `kernel` over `workgroups`, with `resources[i]` bound
    /// to `@binding(i)` of group 0.
    pub(crate) fn dispatch(
        &self,
        device: &Device,
        kernel: &'static str,
        resources: Vec<BindingResource<'_>>,
        workgroups: (u32, u32, u32),
    ) -> Dispatch {
        let (_, layout) = self.kernel(kernel);
        let entries: Vec<wgpu::BindGroupEntry<'_>> = resources
            .into_iter()
            .enumerate()
            .map(|(i, resource)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource,
            })
            .collect();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(kernel),
            layout,
            entries: &entries,
        });
        Dispatch {
            kernel,
            bind_group,
            workgroups,
        }
    }

    /// Record `dispatches` in order into one compute pass of `encoder`.
    ///
    /// wgpu orders the storage reads and writes of consecutive dispatches in
    /// a pass, so each one sees the results of those before it.
    pub(crate) fn encode(&self, encoder: &mut CommandEncoder, dispatches: &[Dispatch]) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Ferrotensor ops"),
            timestamp_writes: None,
        });
        let mut current = None;
        for d in dispatches {
            if current != Some(d.kernel) {
                pass.set_pipeline(&self.kernel(d.kernel).0);
                current = Some(d.kernel);
            }
            pass.set_bind_group(0, &d.bind_group, &[]);
            let (x, y, z) = d.workgroups;
            pass.dispatch_workgroups(x, y, z);
        }
    }

    fn kernel(&self, name: &str) -> &(ComputePipeline, BindGroupLayout) {
        self.pipelines
            .get(name)
            .unwrap_or_else(|| panic!("kernel '{name}' is not compiled"))
    }

    /// Workgroup grid for a 1-D kernel over `n` elements.
    ///
    /// Spills into the y dimension past 65535 workgroups; the kernels rebuild
    /// the flat index as `x + y * nx * 256`.
    pub fn workgroups_1d(n: usize) -> (u32, u32, u32) {
        Self::workgroup_grid(n.div_ceil(WORKGROUP_1D as usize))
    }

    /// Grid of at least `groups` workgroups (at least one), spilling into the
    /// y dimension past 65535; the kernels rebuild the workgroup index as
    /// `x + y * nx`.
    pub fn workgroup_grid(groups: usize) -> (u32, u32, u32) {
        let groups = u32::try_from(groups.max(1)).expect("too many workgroups");
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
