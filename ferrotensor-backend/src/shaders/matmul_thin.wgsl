// ============================================================================
// THIN MATMUL (few output rows, many output columns)
// ============================================================================
// out = t @ w where the thin operand t has at most a few dozen rows: a short
// pass of activations against a weight, or a LoRA product with N = rank seen
// transposed. The 64x64-tile matmul launches one row of tiles for these and
// leaves most of the GPU idle; here a workgroup covers MT_ROWS rows and 32
// columns and its threads split K.
//
//   t(r, k)   = mt_t[r * ts.x + k * ts.y]
//   w(k, c)   = mt_w[k * cols + c] (mode 0) or mt_w[c * K + k] (mode 1)
//   out(r, c) = alpha * sum + beta * mt_out[r * os.x + c * os.y]
//
// alpha and beta are the bits of mt_params[2].xy; beta = 0 never reads out.
//
// Thread (lane, slice) owns columns 2 * lane and 2 * lane + 1 and steps K two
// at a time from 2 * slice, so it reads w as 2x2 blocks: two pairs that are
// adjacent in memory in either mode (one word each when w is f16). The lane
// is the fast thread index in mode 0 and the slice in mode 1, so neighbouring
// threads read neighbouring memory. The 16 slices are added in workgroup
// memory, one output per thread.

const MT_LANES: u32 = 16u;
const MT_SLICES: u32 = 16u;
const MT_ROWS: u32 = 8u;

@group(0) @binding(0) var<storage, read> mt_t: array<f32>;
@group(0) @binding(1) var<storage, read> mt_w: array<f32>;
@group(0) @binding(2) var<storage, read_write> mt_out: array<f32>;
// [rows, K, cols, mode], [t row stride, t k stride, out row stride, out col stride],
// [alpha, beta, 0, 0]
@group(0) @binding(3) var<uniform> mt_params: array<vec4<u32>, 3>;

var<workgroup> mt_partial: array<vec2<f32>, 2048>; // [MT_ROWS][256 threads]

// Elements i and i + 1 of w; the second is 0 unless `both`.
fn mt_pair(i: u32, both: bool) -> vec2<f32> {
    if (both) { return vec2<f32>(mt_w[i], mt_w[i + 1u]); }
    return vec2<f32>(mt_w[i], 0.0);
}

@compute @workgroup_size(16, 16, 1)
fn matmul_thin(
    @builtin(local_invocation_index) lidx: u32,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
) {
    let rows = mt_params[0].x;
    let k = mt_params[0].y;
    let cols = mt_params[0].z;
    let mode = mt_params[0].w;
    let ts = mt_params[1].xy;
    let os = mt_params[1].zw;
    let row0 = wg.x * MT_ROWS;
    let col0 = wg.y * MT_LANES * 2u;

    var lane = lid.x;
    var slice = lid.y;
    if (mode == 1u) {
        lane = lid.y;
        slice = lid.x;
    }
    let c = col0 + lane * 2u;

    var acc: array<vec2<f32>, 8>;
    if (c < cols) {
        for (var kk = slice * 2u; kk < k; kk += MT_SLICES * 2u) {
            let k1 = kk + 1u < k;
            // w0 = w(kk, c..c+1), w1 = w(kk+1, c..c+1)
            var w0: vec2<f32>;
            var w1 = vec2<f32>(0.0);
            if (mode == 0u) {
                w0 = mt_pair(kk * cols + c, c + 1u < cols);
                if (k1) { w1 = mt_pair((kk + 1u) * cols + c, c + 1u < cols); }
            } else {
                let p0 = mt_pair(c * k + kk, k1);
                var p1 = vec2<f32>(0.0);
                if (c + 1u < cols) { p1 = mt_pair((c + 1u) * k + kk, k1); }
                w0 = vec2<f32>(p0.x, p1.x);
                w1 = vec2<f32>(p0.y, p1.y);
            }
            for (var r = 0u; r < MT_ROWS; r++) {
                // Rows past the end repeat the last row; their sums are not stored.
                let i = min(row0 + r, rows - 1u) * ts.x + kk * ts.y;
                var t1 = 0.0;
                if (k1) { t1 = mt_t[i + ts.y]; }
                acc[r] = fma(vec2<f32>(mt_t[i]), w0, fma(vec2<f32>(t1), w1, acc[r]));
            }
        }
    }
    for (var r = 0u; r < MT_ROWS; r++) {
        mt_partial[r * 256u + lidx] = acc[r];
    }
    workgroupBarrier();

    let r = lidx / (MT_LANES * 2u);
    let out_lane = (lidx / 2u) % MT_LANES;
    var sum = 0.0;
    for (var s = 0u; s < MT_SLICES; s++) {
        var t = s * MT_LANES + out_lane;
        if (mode == 1u) { t = out_lane * MT_SLICES + s; }
        sum += mt_partial[r * 256u + t][lidx & 1u];
    }
    let row = row0 + r;
    let col = col0 + lidx % (MT_LANES * 2u);
    if (row < rows && col < cols) {
        let i = row * os.x + col * os.y;
        let beta = bitcast<f32>(mt_params[2].y);
        var o = bitcast<f32>(mt_params[2].x) * sum;
        if (beta != 0.0) { o = o + beta * mt_out[i]; }
        mt_out[i] = o;
    }
}
