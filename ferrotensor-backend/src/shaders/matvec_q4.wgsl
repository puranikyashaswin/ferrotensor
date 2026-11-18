// ============================================================================
// MATRIX-VECTOR PRODUCT, 4-BIT WEIGHTS
// ============================================================================
// out[N] = x[K] @ w[K, N] with w in the format of q4.rs: codes in [N, K]
// order, eight per word, then one (scale, min) f16 pair per group of G
// codes. A column's codes are contiguous, so each 32-thread row of the
// workgroup takes one column and its threads read consecutive code words.
// G is a multiple of 8, so a code word never straddles two groups and
// needs one scale read.

@group(0) @binding(0) var<storage, read> mvq_x: array<f32>;
@group(0) @binding(1) var<storage, read> mvq_w: array<u32>;
@group(0) @binding(2) var<storage, read_write> mvq_out: array<f32>;
@group(0) @binding(3) var<uniform> mvq_params: vec4<u32>; // K, N, log2 G, unused

var<workgroup> mvq_partial: array<f32, 256>; // [8 columns][32 threads]

@compute @workgroup_size(32, 8, 1)
fn matvec_q4(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let k = mvq_params.x;
    let n = mvq_params.y;
    let lg = mvq_params.z;
    let col = (wg.x + wg.y * nwg.x) * 8u + lid.y;
    var acc = 0.0;
    if (col < n) {
        let first = col * k;
        let end = first + k;
        let groups = (k * n + (1u << lg) - 1u) >> lg;
        let scales = groups << (lg - 3u);
        for (var word = (first >> 3u) + lid.x; word * 8u < end; word += 32u) {
            let codes = mvq_w[word];
            let sm = unpack2x16float(mvq_w[scales + ((word * 8u) >> lg)]);
            for (var e = 0u; e < 8u; e++) {
                let j = word * 8u + e;
                if (j >= first && j < end) {
                    let q = f32((codes >> (e * 4u)) & 15u);
                    acc = fma(q * sm.x + sm.y, mvq_x[j - first], acc);
                }
            }
        }
    }
    mvq_partial[lid.y * 32u + lid.x] = acc;
    workgroupBarrier();
    if (lid.x == 0u && col < n) {
        var sum = 0.0;
        for (var t = 0u; t < 32u; t++) {
            sum += mvq_partial[lid.y * 32u + t];
        }
        mvq_out[col] = sum;
    }
}
