// ============================================================================
// FLASH ATTENTION V2 - backward pass
// ============================================================================
// Same layout, masking and dropout conventions as flash_attention_v2.wgsl
// (uses its hash3 / dropout_scale helpers). Two passes, no atomics:
//   1. flash_bwd_dq: per query row, Delta = dO . O, the log-sum-exp L of
//      the scores (first sweep over keys) and dQ = scale * sum_j dS_ij K_j
//      (second sweep). Delta and L are stored for pass 2.
//   2. flash_bwd_dk_dv: per key row, dV_j = sum_i P_ij M_ij dO_i and
//      dK_j = scale * sum_i dS_ij Q_i (summed over the query heads that
//      share this KV head).
// with P = softmax(S), M the dropout multiplier and
//      dS_ij = P_ij * (M_ij * (dO_i . V_j) - Delta_i).
// Like the forward kernel, each workgroup owns FAB_ROWS rows (queries or
// keys) of one head and streams the other side through workgroup memory as
// vec4 tiles of FAB_TILE rows. Each pass uses at most 8 storage buffers.

@group(0) @binding(0) var<storage, read> fab2_q: array<f32>;
@group(0) @binding(1) var<storage, read> fab2_k: array<f32>;
@group(0) @binding(2) var<storage, read> fab2_v: array<f32>;
@group(0) @binding(3) var<storage, read> fab2_out: array<f32>;
@group(0) @binding(4) var<storage, read> fab2_grad_out: array<f32>;
@group(0) @binding(5) var<storage, read_write> fab2_dq: array<f32>;
@group(0) @binding(6) var<storage, read_write> fab2_dk: array<f32>;
@group(0) @binding(7) var<storage, read_write> fab2_dv: array<f32>;

struct FlashBwdParams {
    batch_size: u32,
    num_heads: u32,
    seq_q: u32,
    seq_k: u32,
    head_dim: u32,
    num_kv_heads: u32,
    scale: f32,
    dropout_prob: f32,
    seed: u32,
    causal: u32,
    has_mask: u32,
    causal_offset: u32,
    mask_stride_b: u32,
    mask_stride_h: u32,
    mask_stride_q: u32,
    mask_stride_k: u32,
    window: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};
@group(0) @binding(8) var<uniform> fab2_params: FlashBwdParams;
@group(0) @binding(9) var<storage, read> fab2_mask: array<f32>;
// Per (batch, head, query) row: [Delta, LSE].
@group(0) @binding(10) var<storage, read_write> fab2_stats: array<vec2<f32>>;

const FAB_MASKED: f32 = -1.0e30;
const FAB_ROWS: u32 = 128u;
const FAB_TILE: u32 = 16u;
const FAB_D4: u32 = 32u; // max head_dim / 4

var<workgroup> fab2_t0: array<vec4<f32>, 512>; // FAB_TILE * FAB_D4
var<workgroup> fab2_t1: array<vec4<f32>, 512>;

// vec4 c4 of a head_dim-long row starting at `base` (zero past the end),
// one loader per buffer so each entry point binds only what it reads.
fn fab2_ld4_q(base: u32, c4: u32) -> vec4<f32> {
    let c = c4 * 4u;
    let d = fab2_params.head_dim;
    var r = vec4<f32>(0.0);
    if (c < d) { r.x = fab2_q[base + c]; }
    if (c + 1u < d) { r.y = fab2_q[base + c + 1u]; }
    if (c + 2u < d) { r.z = fab2_q[base + c + 2u]; }
    if (c + 3u < d) { r.w = fab2_q[base + c + 3u]; }
    return r;
}

fn fab2_ld4_k(base: u32, c4: u32) -> vec4<f32> {
    let c = c4 * 4u;
    let d = fab2_params.head_dim;
    var r = vec4<f32>(0.0);
    if (c < d) { r.x = fab2_k[base + c]; }
    if (c + 1u < d) { r.y = fab2_k[base + c + 1u]; }
    if (c + 2u < d) { r.z = fab2_k[base + c + 2u]; }
    if (c + 3u < d) { r.w = fab2_k[base + c + 3u]; }
    return r;
}

fn fab2_ld4_v(base: u32, c4: u32) -> vec4<f32> {
    let c = c4 * 4u;
    let d = fab2_params.head_dim;
    var r = vec4<f32>(0.0);
    if (c < d) { r.x = fab2_v[base + c]; }
    if (c + 1u < d) { r.y = fab2_v[base + c + 1u]; }
    if (c + 2u < d) { r.z = fab2_v[base + c + 2u]; }
    if (c + 3u < d) { r.w = fab2_v[base + c + 3u]; }
    return r;
}

fn fab2_ld4_o(base: u32, c4: u32) -> vec4<f32> {
    let c = c4 * 4u;
    let d = fab2_params.head_dim;
    var r = vec4<f32>(0.0);
    if (c < d) { r.x = fab2_out[base + c]; }
    if (c + 1u < d) { r.y = fab2_out[base + c + 1u]; }
    if (c + 2u < d) { r.z = fab2_out[base + c + 2u]; }
    if (c + 3u < d) { r.w = fab2_out[base + c + 3u]; }
    return r;
}

fn fab2_ld4_do(base: u32, c4: u32) -> vec4<f32> {
    let c = c4 * 4u;
    let d = fab2_params.head_dim;
    var r = vec4<f32>(0.0);
    if (c < d) { r.x = fab2_grad_out[base + c]; }
    if (c + 1u < d) { r.y = fab2_grad_out[base + c + 1u]; }
    if (c + 2u < d) { r.z = fab2_grad_out[base + c + 2u]; }
    if (c + 3u < d) { r.w = fab2_grad_out[base + c + 3u]; }
    return r;
}

// Whether key kj is visible to query qi (causal and window rules).
fn fab2_visible(qi: u32, kj: u32) -> bool {
    let p = fab2_params;
    let qa = qi + p.causal_offset;
    if (kj >= p.seq_k || qi >= p.seq_q) { return false; }
    if (p.causal == 1u && kj > qa) { return false; }
    if (p.window > 0u) {
        let dist = select(kj - qa, qa - kj, qa >= kj);
        if (dist >= p.window) { return false; }
    }
    return true;
}

fn fab2_mask_at(b: u32, h: u32, qi: u32, kj: u32) -> f32 {
    let p = fab2_params;
    if (p.has_mask == 0u) { return 0.0; }
    return fab2_mask[b * p.mask_stride_b + h * p.mask_stride_h + qi * p.mask_stride_q + kj * p.mask_stride_k];
}

fn fab2_q_off(b: u32, qi: u32, h: u32) -> u32 {
    let p = fab2_params;
    return ((b * p.seq_q + qi) * p.num_heads + h) * p.head_dim;
}

fn fab2_kv_off(b: u32, kj: u32, kvh: u32) -> u32 {
    let p = fab2_params;
    return ((b * p.seq_k + kj) * p.num_kv_heads + kvh) * p.head_dim;
}

fn fab2_row(b: u32, h: u32, qi: u32) -> u32 {
    let p = fab2_params;
    return (b * p.num_heads + h) * p.seq_q + qi;
}

// Key range [begin, end) visible to any query in [q0, q1] (uniform).
fn fab2_key_range(q0: u32, q1: u32) -> vec2<u32> {
    let p = fab2_params;
    var end = p.seq_k;
    if (p.causal == 1u) { end = min(end, q1 + p.causal_offset + 1u); }
    var begin = 0u;
    if (p.window > 0u) {
        let qa0 = q0 + p.causal_offset;
        if (qa0 + 1u > p.window) { begin = qa0 + 1u - p.window; }
        if (p.causal == 0u) { end = min(end, q1 + p.causal_offset + p.window); }
    }
    return vec2<u32>(begin, end);
}

// Loads keys/values [k0, k0 + FAB_TILE) of head kvh into fab2_t0 / fab2_t1.
fn fab2_load_kv(lid: u32, nthreads: u32, b: u32, kvh: u32, k0: u32, d4: u32) {
    for (var i = lid; i < FAB_TILE * d4; i += nthreads) {
        let r = i / d4;
        let c = i % d4;
        var kv = vec4<f32>(0.0);
        var vv = vec4<f32>(0.0);
        if (k0 + r < fab2_params.seq_k) {
            let base = fab2_kv_off(b, k0 + r, kvh);
            kv = fab2_ld4_k(base, c);
            vv = fab2_ld4_v(base, c);
        }
        fab2_t0[r * FAB_D4 + c] = kv;
        fab2_t1[r * FAB_D4 + c] = vv;
    }
}

// Dispatch: (1, ceil(seq_q / FAB_ROWS), batch * num_heads)
@compute @workgroup_size(128)
fn flash_bwd_dq(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let p = fab2_params;
    let d4 = (p.head_dim + 3u) / 4u;
    let b = wg.z / p.num_heads;
    let h = wg.z % p.num_heads;
    let kvh = h / (p.num_heads / p.num_kv_heads);
    let bh = b * p.num_heads + h;
    let q0 = wg.y * FAB_ROWS;
    let qi = q0 + lid;
    let valid = qi < p.seq_q;
    let q_off = fab2_q_off(b, qi, h);

    var q_reg: array<vec4<f32>, 32>;
    var do_reg: array<vec4<f32>, 32>;
    var dq_acc: array<vec4<f32>, 32>;
    var delta4 = vec4<f32>(0.0);
    for (var c = 0u; c < d4; c++) {
        dq_acc[c] = vec4<f32>(0.0);
        if (valid) {
            q_reg[c] = fab2_ld4_q(q_off, c) * p.scale;
            do_reg[c] = fab2_ld4_do(q_off, c);
            delta4 = fma(do_reg[c], fab2_ld4_o(q_off, c), delta4);
        } else {
            q_reg[c] = vec4<f32>(0.0);
            do_reg[c] = vec4<f32>(0.0);
        }
    }
    let delta = delta4.x + delta4.y + delta4.z + delta4.w;
    let range = fab2_key_range(q0, min(q0 + FAB_ROWS, p.seq_q) - 1u);

    // Sweep 1: log-sum-exp of this row's scores.
    var m = FAB_MASKED;
    var l = 0.0;
    for (var k0 = range.x; k0 < range.y; k0 += FAB_TILE) {
        fab2_load_kv(lid, 128u, b, kvh, k0, d4);
        workgroupBarrier();
        for (var j = 0u; j < FAB_TILE; j++) {
            let kj = k0 + j;
            if (!fab2_visible(qi, kj)) { continue; }
            var dot4 = vec4<f32>(0.0);
            for (var c = 0u; c < d4; c++) { dot4 = fma(q_reg[c], fab2_t0[j * FAB_D4 + c], dot4); }
            let s = dot4.x + dot4.y + dot4.z + dot4.w + fab2_mask_at(b, h, qi, kj);
            if (s > FAB_MASKED) {
                let m_new = max(m, s);
                l = l * exp(m - m_new) + exp(s - m_new);
                m = m_new;
            }
        }
        workgroupBarrier();
    }
    // A fully masked row has no probability mass; make every P_ij zero.
    var lse = 3.0e38;
    if (l > 0.0) { lse = m + log(l); }

    // Sweep 2: dQ.
    for (var k0 = range.x; k0 < range.y; k0 += FAB_TILE) {
        fab2_load_kv(lid, 128u, b, kvh, k0, d4);
        workgroupBarrier();
        for (var j = 0u; j < FAB_TILE; j++) {
            let kj = k0 + j;
            if (!fab2_visible(qi, kj)) { continue; }
            var dot4 = vec4<f32>(0.0);
            var dov4 = vec4<f32>(0.0);
            for (var c = 0u; c < d4; c++) {
                dot4 = fma(q_reg[c], fab2_t0[j * FAB_D4 + c], dot4);
                dov4 = fma(do_reg[c], fab2_t1[j * FAB_D4 + c], dov4);
            }
            let s = dot4.x + dot4.y + dot4.z + dot4.w + fab2_mask_at(b, h, qi, kj);
            if (s <= FAB_MASKED) { continue; }
            let pij = exp(s - lse);
            let mij = dropout_scale(p.dropout_prob, p.seed, bh, qi, kj);
            let ds = pij * (mij * (dov4.x + dov4.y + dov4.z + dov4.w) - delta) * p.scale;
            for (var c = 0u; c < d4; c++) {
                dq_acc[c] = fma(vec4<f32>(ds), fab2_t0[j * FAB_D4 + c], dq_acc[c]);
            }
        }
        workgroupBarrier();
    }

    if (valid) {
        fab2_stats[fab2_row(b, h, qi)] = vec2<f32>(delta, lse);
        for (var c = 0u; c < d4; c++) {
            let d = c * 4u;
            let v = dq_acc[c];
            fab2_dq[q_off + d] = v.x;
            if (d + 1u < p.head_dim) { fab2_dq[q_off + d + 1u] = v.y; }
            if (d + 2u < p.head_dim) { fab2_dq[q_off + d + 2u] = v.z; }
            if (d + 3u < p.head_dim) { fab2_dq[q_off + d + 3u] = v.w; }
        }
    }
}

// Dispatch: (1, ceil(seq_k / FAB_ROWS), batch * num_kv_heads)
@compute @workgroup_size(128)
fn flash_bwd_dk_dv(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let p = fab2_params;
    let d4 = (p.head_dim + 3u) / 4u;
    let b = wg.z / p.num_kv_heads;
    let kvh = wg.z % p.num_kv_heads;
    let group = p.num_heads / p.num_kv_heads;
    let k0 = wg.y * FAB_ROWS;
    let kj = k0 + lid;
    let valid = kj < p.seq_k;
    let kv_off = fab2_kv_off(b, kj, kvh);
    let k1 = min(k0 + FAB_ROWS, p.seq_k) - 1u;

    var k_reg: array<vec4<f32>, 32>;
    var v_reg: array<vec4<f32>, 32>;
    var dk_acc: array<vec4<f32>, 32>;
    var dv_acc: array<vec4<f32>, 32>;
    for (var c = 0u; c < d4; c++) {
        dk_acc[c] = vec4<f32>(0.0);
        dv_acc[c] = vec4<f32>(0.0);
        if (valid) {
            k_reg[c] = fab2_ld4_k(kv_off, c) * p.scale;
            v_reg[c] = fab2_ld4_v(kv_off, c);
        } else {
            k_reg[c] = vec4<f32>(0.0);
            v_reg[c] = vec4<f32>(0.0);
        }
    }

    // Queries that can see any key in [k0, k1] (uniform).
    var q_begin = 0u;
    var q_end = p.seq_q;
    if (p.causal == 1u && k0 > p.causal_offset) { q_begin = k0 - p.causal_offset; }
    if (p.window > 0u) {
        // |qa - kj| < window  =>  qa < k1 + window and qa > k0 - window
        if (k1 + p.window > p.causal_offset) {
            q_end = min(q_end, k1 + p.window - p.causal_offset);
        } else {
            q_end = 0u;
        }
        if (p.causal == 0u && k0 + 1u > p.window + p.causal_offset) {
            q_begin = max(q_begin, k0 + 1u - p.window - p.causal_offset);
        }
    }

    for (var g = 0u; g < group; g++) {
        let h = kvh * group + g;
        let bh = b * p.num_heads + h;
        for (var qt = q_begin; qt < q_end; qt += FAB_TILE) {
            // Stage Q (fab2_t0) and dO (fab2_t1) rows [qt, qt + FAB_TILE).
            for (var i = lid; i < FAB_TILE * d4; i += 128u) {
                let r = i / d4;
                let c = i % d4;
                var qv = vec4<f32>(0.0);
                var dv = vec4<f32>(0.0);
                if (qt + r < p.seq_q) {
                    let base = fab2_q_off(b, qt + r, h);
                    qv = fab2_ld4_q(base, c);
                    dv = fab2_ld4_do(base, c);
                }
                fab2_t0[r * FAB_D4 + c] = qv;
                fab2_t1[r * FAB_D4 + c] = dv;
            }
            workgroupBarrier();
            for (var i = 0u; i < FAB_TILE; i++) {
                let qi = qt + i;
                if (!(valid && fab2_visible(qi, kj))) { continue; }
                var dot4 = vec4<f32>(0.0);
                var dov4 = vec4<f32>(0.0);
                for (var c = 0u; c < d4; c++) {
                    dot4 = fma(k_reg[c], fab2_t0[i * FAB_D4 + c], dot4);
                    dov4 = fma(v_reg[c], fab2_t1[i * FAB_D4 + c], dov4);
                }
                let s = dot4.x + dot4.y + dot4.z + dot4.w + fab2_mask_at(b, h, qi, kj);
                if (s <= FAB_MASKED) { continue; }
                let stats = fab2_stats[fab2_row(b, h, qi)];
                let pij = exp(s - stats.y);
                let mij = dropout_scale(p.dropout_prob, p.seed, bh, qi, kj);
                let ds = pij * (mij * (dov4.x + dov4.y + dov4.z + dov4.w) - stats.x) * p.scale;
                let pm = pij * mij;
                for (var c = 0u; c < d4; c++) {
                    dv_acc[c] = fma(vec4<f32>(pm), fab2_t1[i * FAB_D4 + c], dv_acc[c]);
                    dk_acc[c] = fma(vec4<f32>(ds), fab2_t0[i * FAB_D4 + c], dk_acc[c]);
                }
            }
            workgroupBarrier();
        }
    }

    if (valid) {
        for (var c = 0u; c < d4; c++) {
            let d = c * 4u;
            let gk = dk_acc[c];
            let gv = dv_acc[c];
            fab2_dk[kv_off + d] = gk.x;
            fab2_dv[kv_off + d] = gv.x;
            if (d + 1u < p.head_dim) { fab2_dk[kv_off + d + 1u] = gk.y; fab2_dv[kv_off + d + 1u] = gv.y; }
            if (d + 2u < p.head_dim) { fab2_dk[kv_off + d + 2u] = gk.z; fab2_dv[kv_off + d + 2u] = gv.z; }
            if (d + 3u < p.head_dim) { fab2_dk[kv_off + d + 3u] = gk.w; fab2_dv[kv_off + d + 3u] = gv.w; }
        }
    }
}
