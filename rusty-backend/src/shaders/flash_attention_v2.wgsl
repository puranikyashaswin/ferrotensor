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
// time, keeping a running max / sum (online softmax) so the full score matrix
// is never materialised.

const FA_TILE_Q: u32 = 64u;
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

var<workgroup> fa_k_tile: array<f32, 2048>; // FA_TILE_K * FA_MAX_D
var<workgroup> fa_v_tile: array<f32, 2048>;

@compute @workgroup_size(64, 1, 1)
fn flash_attention_v2_metal(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let p = flash_v2_params;
    let d_len = p.head_dim;
    let batch_idx = wg_id.z / p.num_heads;
    let head_idx = wg_id.z % p.num_heads;
    let kv_head_idx = head_idx / (p.num_heads / p.num_kv_heads);

    let q_start = wg_id.y * FA_TILE_Q;
    let q_idx = q_start + local_id.x;
    let valid = q_idx < p.seq_q;

    var q_reg: array<f32, 128>;
    var acc: array<f32, 128>;
    let q_base = ((batch_idx * p.seq_q + q_idx) * p.num_heads + head_idx) * d_len;
    for (var d = 0u; d < d_len; d++) {
        acc[d] = 0.0;
        if (valid) { q_reg[d] = flash_v2_q[q_base + d] * p.scale; } else { q_reg[d] = 0.0; }
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
        let n = FA_TILE_K * d_len;
        for (var i = local_id.x; i < n; i += 64u) {
            let r = i / d_len;
            let c = i % d_len;
            let kp = k_start + r;
            var kv = 0.0;
            var vv = 0.0;
            if (kp < p.seq_k) {
                let off = ((batch_idx * p.seq_k + kp) * p.num_kv_heads + kv_head_idx) * d_len + c;
                kv = flash_v2_k[off];
                vv = flash_v2_v[off];
            }
            fa_k_tile[r * FA_MAX_D + c] = kv;
            fa_v_tile[r * FA_MAX_D + c] = vv;
        }
        workgroupBarrier();

        if (valid) {
            var s: array<f32, 16>;
            var tile_max = FA_MASKED;
            for (var j = 0u; j < FA_TILE_K; j++) {
                let kp = k_start + j;
                var sc = FA_MASKED;
                if (fa_visible(p, q_idx, kp)) {
                    var dot = 0.0;
                    for (var d = 0u; d < d_len; d++) {
                        dot += q_reg[d] * fa_k_tile[j * FA_MAX_D + d];
                    }
                    sc = dot;
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
                for (var d = 0u; d < d_len; d++) { acc[d] *= corr; }
                for (var j = 0u; j < FA_TILE_K; j++) {
                    if (s[j] > FA_MASKED) {
                        let pj = exp(s[j] - m_new);
                        l_run += pj;
                        let w = pj * dropout_scale(p.dropout_prob, p.seed_val, bh, q_idx, k_start + j);
                        for (var d = 0u; d < d_len; d++) {
                            acc[d] += w * fa_v_tile[j * FA_MAX_D + d];
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
        for (var d = 0u; d < d_len; d++) {
            flash_v2_out[q_base + d] = acc[d] * inv;
        }
    }
}
