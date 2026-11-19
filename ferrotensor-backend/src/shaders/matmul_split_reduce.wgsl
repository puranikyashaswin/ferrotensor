// ============================================================================
// SPLIT-K REDUCTION
// ============================================================================
// out[i] = parts[i] + parts[len + i] + ... + parts[(S - 1) * len + i], added
// in that order, so a split-K matmul gives the same bits on every run.

@group(0) @binding(0) var<storage, read> msr_parts: array<f32>;
@group(0) @binding(1) var<storage, read_write> msr_out: array<f32>;
@group(0) @binding(2) var<uniform> msr_params: vec4<u32>; // len, S

@compute @workgroup_size(256, 1, 1)
fn matmul_split_reduce(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    // Grids wider than 65535 workgroups spill into y.
    let i = gid.y * groups.x * 256u + gid.x;
    let len = msr_params.x;
    if (i >= len) { return; }
    var sum = msr_parts[i];
    for (var s = 1u; s < msr_params.y; s++) {
        sum += msr_parts[s * len + i];
    }
    msr_out[i] = sum;
}
