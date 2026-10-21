// ============================================================================
// FLASH ATTENTION V2 - forward pass
// ============================================================================
// Layout: q/out [batch, seq_q, num_heads, head_dim],
//         k/v   [batch, seq_k, num_kv_heads, head_dim] (GQA when kv < heads).
// Query i sits at absolute position i + causal_offset (for decoding against
// a KV cache). Key j is visible when j < seq_k, j <= i + causal_offset if
// causal, and |i + causal_offset - j| < window if window > 0. An additive mask
// with broadcast strides and attention dropout are also supported.
// head_dim <= 128.
//
// One thread owns one query row; a workgroup covers FA_TILE_Q queries of one
// (batch, head) and streams K/V through workgroup memory FA_TILE_K keys at a
// time as vec4s, keeping a running max / sum (online softmax) so the full
// score matrix is never materialised.

const FA_TILE_Q: u32 = 128u;
const FA_TILE_K: u32 = 16u;
const FA_MAX_D: u32 = 128u;
// Scores at or below this are treated as masked out (-inf).
const FA_MASKED: f32 = -1.0e30;

@group(0) @binding(0) var<storage, read> flash_v2_q: array<f32>;
@group(0) @binding(1) var<storage, read> flash_v2_k: array<f32>;
@group(0) @binding(2) var<storage, read> flash_v2_v: array<f32>;
@group(0) @binding(3) var<storage, read_write> flash_v2_out: array<f32>;

struct FlashV2Params {
    batch_size: u32,
    num_heads: u32,
    seq_q: u32,
    seq_k: u32,
    head_dim: u32,
    causal: u32,
    scale: f32,
    num_kv_heads: u32,
    dropout_prob: f32,
    seed_val: u32,
    has_mask: u32,
    causal_offset: u32,
    // Mask strides; 0 broadcasts that dimension.
    mask_stride_b: u32,
    mask_stride_h: u32,
    mask_stride_q: u32,
    mask_stride_k: u32,
    window: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};
@group(0) @binding(4) var<uniform> flash_v2_params: FlashV2Params;
@group(0) @binding(5) var<storage, read> flash_v2_mask: array<f32>;

// Simple 3D hash for dropout RNG. Shared with the backward kernels, which
// must regenerate exactly the same keep/drop decisions.
fn hash3(x: u32, y: u32, z: u32) -> f32 {
    var a = x; var b = y; var c = z;
    a = a * 1664525u + 1013904223u;
    b = b * 1664525u + 1013904223u;
    c = c * 1664525u + 1013904223u;
    a = a + b * c;
    b = b + c * a;
    c = c + a * b;
    a = a ^ (a >> 16u);
    b = b ^ (b >> 16u);
    c = c ^ (c >> 16u);
    return f32(a + b + c) * 2.3283064365386963e-10; // * 1/2^32
}

// Dropout multiplier for attention weight (bh, q, k): 0 if dropped,
// 1 / (1 - p) if kept.
fn dropout_scale(prob: f32, seed: u32, bh: u32, q_idx: u32, k_idx: u32) -> f32 {
    if (prob <= 0.0) { return 1.0; }
    let r = hash3(bh, q_idx, k_idx + seed);
    return select(0.0, 1.0 / (1.0 - prob), r >= prob);
}

// Whether key `kp` is visible to query `q_idx` (causal, window, bounds).
fn fa_visible(p: FlashV2Params, q_idx: u32, kp: u32) -> bool {
    let qa = q_idx + p.causal_offset;
    if (kp >= p.seq_k) { return false; }
    if (p.causal == 1u && kp > qa) { return false; }
    if (p.window > 0u) {
        let dist = select(kp - qa, qa - kp, qa >= kp);
        if (dist >= p.window) { return false; }
    }
    return true;
}

// K/V tiles as vec4s: [FA_TILE_K][FA_MAX_D / 4].
var<workgroup> fa_k_tile: array<vec4<f32>, 512>;
var<workgroup> fa_v_tile: array<vec4<f32>, 512>;

// Element `c` of a head_dim-long row starting at `base`, or 0 past the end.
fn fa_ld(buf_sel: u32, base: u32, c: u32, d_len: u32) -> f32 {
    if (c >= d_len) { return 0.0; }
    if (buf_sel == 0u) { return flash_v2_q[base + c]; }
    if (buf_sel == 1u) { return flash_v2_k[base + c]; }
    return flash_v2_v[base + c];
}

fn fa_ld4(buf_sel: u32, base: u32, c4: u32, d_len: u32) -> vec4<f32> {
    let c = c4 * 4u;
    return vec4<f32>(
        fa_ld(buf_sel, base, c, d_len),
        fa_ld(buf_sel, base, c + 1u, d_len),
        fa_ld(buf_sel, base, c + 2u, d_len),
        fa_ld(buf_sel, base, c + 3u, d_len),
    );
}

@compute @workgroup_size(128, 1, 1)
fn flash_attention_v2_metal(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let p = flash_v2_params;
    let d_len = p.head_dim;
    let d4 = (d_len + 3u) / 4u;
    let batch_idx = wg_id.z / p.num_heads;
    let head_idx = wg_id.z % p.num_heads;
    let kv_head_idx = head_idx / (p.num_heads / p.num_kv_heads);

    let q_start = wg_id.y * FA_TILE_Q;
    let q_idx = q_start + local_id.x;
    let valid = q_idx < p.seq_q;

    var q_reg: array<vec4<f32>, 32>;
    var acc: array<vec4<f32>, 32>;
    let q_base = ((batch_idx * p.seq_q + q_idx) * p.num_heads + head_idx) * d_len;
    for (var c = 0u; c < d4; c++) {
        acc[c] = vec4<f32>(0.0);
        if (valid) { q_reg[c] = fa_ld4(0u, q_base, c, d_len) * p.scale; } else { q_reg[c] = vec4<f32>(0.0); }
    }

    var m_run = FA_MASKED; // running max
    var l_run = 0.0;       // running sum of exp(score - m_run)

    // Range of keys this workgroup can see (uniform across the workgroup).
    let q_first_abs = q_start + p.causal_offset;
    let q_last_abs = min(q_start + FA_TILE_Q, p.seq_q) - 1u + p.causal_offset;
    var k_end = p.seq_k;
    if (p.causal == 1u) { k_end = min(p.seq_k, q_last_abs + 1u); }
    var k_begin = 0u;
    if (p.window > 0u) {
        if (q_first_abs + 1u > p.window) { k_begin = q_first_abs + 1u - p.window; }
        if (p.causal == 0u) { k_end = min(k_end, q_last_abs + p.window); }
    }

    let bh = batch_idx * p.num_heads + head_idx;
    let mask_base = batch_idx * p.mask_stride_b + head_idx * p.mask_stride_h + q_idx * p.mask_stride_q;

    for (var k_start = k_begin; k_start < k_end; k_start += FA_TILE_K) {
        // Cooperative load of FA_TILE_K rows of K and V.
        for (var i = local_id.x; i < FA_TILE_K * d4; i += 128u) {
            let r = i / d4;
            let c = i % d4;
            let kp = k_start + r;
            var kv = vec4<f32>(0.0);
            var vv = vec4<f32>(0.0);
            if (kp < p.seq_k) {
                let base = ((batch_idx * p.seq_k + kp) * p.num_kv_heads + kv_head_idx) * d_len;
                kv = fa_ld4(1u, base, c, d_len);
                vv = fa_ld4(2u, base, c, d_len);
            }
            fa_k_tile[r * (FA_MAX_D / 4u) + c] = kv;
            fa_v_tile[r * (FA_MAX_D / 4u) + c] = vv;
        }
        workgroupBarrier();

        if (valid) {
            var s: array<f32, 16>;
            var tile_max = FA_MASKED;
            for (var j = 0u; j < FA_TILE_K; j++) {
                let kp = k_start + j;
                var sc = FA_MASKED;
                if (fa_visible(p, q_idx, kp)) {
                    var dot4 = vec4<f32>(0.0);
                    for (var c = 0u; c < d4; c++) {
                        dot4 = fma(q_reg[c], fa_k_tile[j * (FA_MAX_D / 4u) + c], dot4);
                    }
                    sc = dot4.x + dot4.y + dot4.z + dot4.w;
                    if (p.has_mask == 1u) {
                        sc += flash_v2_mask[mask_base + kp * p.mask_stride_k];
                    }
                }
                s[j] = sc;
                tile_max = max(tile_max, sc);
            }

            if (tile_max > FA_MASKED) {
                let m_new = max(m_run, tile_max);
                let corr = exp(m_run - m_new);
                l_run *= corr;
                for (var c = 0u; c < d4; c++) { acc[c] *= corr; }
                for (var j = 0u; j < FA_TILE_K; j++) {
                    if (s[j] > FA_MASKED) {
                        let pj = exp(s[j] - m_new);
                        l_run += pj;
                        let w = pj * dropout_scale(p.dropout_prob, p.seed_val, bh, q_idx, k_start + j);
                        for (var c = 0u; c < d4; c++) {
                            acc[c] = fma(vec4<f32>(w), fa_v_tile[j * (FA_MAX_D / 4u) + c], acc[c]);
                        }
                    }
                }
                m_run = m_new;
            }
        }
        workgroupBarrier();
    }

    if (valid) {
        var inv = 0.0;
        if (l_run > 0.0) { inv = 1.0 / l_run; }
        for (var c = 0u; c < d4; c++) {
            let o = acc[c] * inv;
            let d = c * 4u;
            flash_v2_out[q_base + d] = o.x;
            if (d + 1u < d_len) { flash_v2_out[q_base + d + 1u] = o.y; }
            if (d + 2u < d_len) { flash_v2_out[q_base + d + 2u] = o.z; }
            if (d + 3u < d_len) { flash_v2_out[q_base + d + 3u] = o.w; }
        }
    }
}
