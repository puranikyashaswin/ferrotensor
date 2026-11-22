// ============================================================================
// SPLIT-K REDUCTION
// ============================================================================
// out[i] = parts[i] + parts[len + i] + ... + parts[(S - 1) * len + i], added
// in that order, so a split-K matmul gives the same bits on every run. The
// sum is then stored as alpha * sum + beta * out[i]; beta = 0 never reads out.

@group(0) @binding(0) var<storage, read> msr_parts: array<f32>;
@group(0) @binding(1) var<storage, read_write> msr_out: array<f32>;
@group(0) @binding(2) var<uniform> msr_params: vec4<u32>; // len, S, alpha bits, beta bits

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
    let beta = bitcast<f32>(msr_params.w);
    var o = bitcast<f32>(msr_params.z) * sum;
    if (beta != 0.0) { o = o + beta * msr_out[i]; }
    msr_out[i] = o;
}

// ============================================================================
// SPLIT-K REDUCTION OF A COLUMN RANGE
// ============================================================================
// The tail split of a coop matmul splits K only for the last column blocks of
// the grid; the other columns were stored whole by a first dispatch. This adds
// the slices of columns [c0, c0 + cols) of every row, in the same slice order
// as matmul_split_reduce, so the sums are the same for the same slices.
// msr_cols[0] = (m, n, c0, cols), msr_cols[1] = (S, alpha bits, beta bits, 0).

@group(0) @binding(0) var<storage, read> msrc_parts: array<f32>;
@group(0) @binding(1) var<storage, read_write> msrc_out: array<f32>;
@group(0) @binding(2) var<uniform> msr_cols: array<vec4<u32>, 2>;

@compute @workgroup_size(256, 1, 1)
fn matmul_split_reduce_cols(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let i = gid.y * groups.x * 256u + gid.x;
    let m = msr_cols[0].x;
    let n = msr_cols[0].y;
    let cols = msr_cols[0].w;
    if (i >= m * cols) { return; }
    let o = (i / cols) * n + msr_cols[0].z + i % cols;
    let len = m * n;
    var sum = msrc_parts[o];
    for (var s = 1u; s < msr_cols[1].x; s++) {
        sum += msrc_parts[s * len + o];
    }
    let beta = bitcast<f32>(msr_cols[1].z);
    var v = bitcast<f32>(msr_cols[1].y) * sum;
    if (beta != 0.0) { v = v + beta * msrc_out[o]; }
    msrc_out[o] = v;
}
