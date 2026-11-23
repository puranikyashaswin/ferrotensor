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

// Four outputs per thread, for len a multiple of 4: slice s of output 4 * i + j
// is lane j of msr4_parts[s * (len / 4) + i], so every slice start stays
// aligned to a vec4. Each lane is added in the same slice order as the scalar
// kernel above, and a vec4 add is four independent f32 adds, so every output
// has the same bits as the scalar kernel gives it.
@group(0) @binding(0) var<storage, read> msr4_parts: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> msr4_out: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> msr4_params: vec4<u32>; // len, S, alpha bits, beta bits

@compute @workgroup_size(256, 1, 1)
fn matmul_split_reduce4(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    // Grids wider than 65535 workgroups spill into y.
    let i = gid.y * groups.x * 256u + gid.x;
    let len4 = msr4_params.x / 4u;
    if (i >= len4) { return; }
    var sum = msr4_parts[i];
    for (var s = 1u; s < msr4_params.y; s++) {
        sum += msr4_parts[s * len4 + i];
    }
    let beta = bitcast<f32>(msr4_params.w);
    var o = bitcast<f32>(msr4_params.z) * sum;
    if (beta != 0.0) { o = o + beta * msr4_out[i]; }
    msr4_out[i] = o;
}
