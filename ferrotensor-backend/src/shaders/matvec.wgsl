// ============================================================================
// MATRIX-VECTOR PRODUCT (one row of activations)
// ============================================================================
// out[N] = x[K] @ w[K, N] for row-major w: the products of decoding one
// token. The 64x64-tile matmul would leave 15 of its 16 thread rows idle.
//
// A 16x16 workgroup covers 32 output columns. Thread (lane, slice) sums
// columns 2 * lane and 2 * lane + 1 over k = slice, slice + 16, ..., so the
// 16 lanes of one k read consecutive memory, and the 16 slices are added
// in workgroup memory.

const MV_LANES: u32 = 16u;
const MV_SLICES: u32 = 16u;

@group(0) @binding(0) var<storage, read> mv_x: array<f32>;
@group(0) @binding(1) var<storage, read> mv_w: array<f32>;
@group(0) @binding(2) var<storage, read_write> mv_out: array<f32>;
@group(0) @binding(3) var<uniform> mv_params: vec4<u32>; // K, N, unused, unused

var<workgroup> mv_partial: array<vec2<f32>, 256>; // [MV_SLICES][MV_LANES]

// Columns n and n + 1 (n even) of row k; 0 for a column past the end.
fn mv_load_pair(k: u32, n: u32, cols: u32) -> vec2<f32> {
    let i = k * cols + n;
    if (n + 1u < cols) { return vec2<f32>(mv_w[i], mv_w[i + 1u]); }
    return vec2<f32>(mv_w[i], 0.0);
}

@compute @workgroup_size(16, 16, 1)
fn matvec(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let k = mv_params.x;
    let n = mv_params.y;
    let col = ((wg.x + wg.y * nwg.x) * MV_LANES + lid.x) * 2u;
    var acc = vec2<f32>(0.0);
    if (col < n) {
        for (var kk = lid.y; kk < k; kk += MV_SLICES) {
            acc = fma(vec2<f32>(mv_x[kk]), mv_load_pair(kk, col, n), acc);
        }
    }
    mv_partial[lid.y * MV_LANES + lid.x] = acc;
    workgroupBarrier();
    if (lid.y == 0u && col < n) {
        var sum = vec2<f32>(0.0);
        for (var s = 0u; s < MV_SLICES; s++) {
            sum += mv_partial[s * MV_LANES + lid.x];
        }
        mv_out[col] = sum.x;
        if (col + 1u < n) { mv_out[col + 1u] = sum.y; }
    }
}
