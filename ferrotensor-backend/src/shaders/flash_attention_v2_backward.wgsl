// ============================================================================
// FLASH ATTENTION V2 - backward pass
// ============================================================================
// Same layout, masking and dropout conventions as flash_attention_v2.wgsl
// (uses its hash3 / dropout_scale helpers). Two passes, no atomics:
//   1. flash_bwd_dq: per query row, Delta = dO . O, the log-sum-exp L of
//      the scores (first sweep over keys) and dQ = scale * sum_j dS_ij K_j
//      (second sweep). Delta and L are stored for pass 2. When lse_saved is
//      set, the forward pass already wrote L into the stats and the first
//      sweep is skipped.
//   2. flash_bwd_dk_dv: per key row and query head, dV_j = sum_i P_ij M_ij dO_i
//      and dK_j = scale * sum_i dS_ij Q_i. With grouped-query attention each
//      query head writes its own partial and flash_group_sum adds the heads
//      that share a KV head, so the workgroups of one KV head run side by
//      side rather than one after another.
// with P = softmax(S), M the dropout multiplier and
//      dS_ij = P_ij * (M_ij * (dO_i . V_j) - Delta_i).
// Like the forward kernel, each workgroup owns FAB_ROWS rows (queries or
// keys) of one head and streams the other side through workgroup memory as
// vec4 tiles of FAB_TILE rows. FAB_TPR threads share a row: each owns every
// FAB_TPR-th vec4 of head_dim, and a dot product is the sum of their partials,
// passed through workgroup memory. Small row blocks give short sequences
// enough workgroups to fill the GPU, and the tiles take 16 KB of workgroup
// memory, the floor every WebGPU adapter offers. Each pass uses at most 8
// storage buffers.

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
    seed_hi: u32,
    lse_saved: u32,
    // 1: flash_bwd_dk_dv writes dK and dV as [batch, seq_k, num_kv_heads,
    // head_dim] (one query head per key head); 0: per-query-head partials
    // [batch, seq_k, num_heads, head_dim] that flash_group_sum adds up.
    dkv_direct: u32,
};
@group(0) @binding(8) var<uniform> fab2_params: FlashBwdParams;
@group(0) @binding(9) var<storage, read> fab2_mask: array<f32>;
// Per (batch, head, query) row: [Delta, LSE].
@group(0) @binding(10) var<storage, read_write> fab2_stats: array<vec2<f32>>;
// The per-query-head dK and dV partials flash_group_sum adds up; only that entry binds them.
@group(0) @binding(11) var<storage, read> fab2_dkp: array<f32>;
@group(0) @binding(12) var<storage, read> fab2_dvp: array<f32>;

const FAB_MASKED: f32 = -1.0e30;
const FAB_ROWS: u32 = 16u;
const FAB_TILE: u32 = 8u;
const FAB_D4: u32 = 32u;     // max head_dim / 4
const FAB_TPR: u32 = 8u;     // threads per row
const FAB_THREADS: u32 = 128u; // FAB_ROWS * FAB_TPR
const FAB_SLICES: u32 = 4u;  // vec4s of head_dim per thread: FAB_D4 / FAB_TPR

var<workgroup> fab2_t0: array<vec4<f32>, 256>; // FAB_TILE * FAB_D4
var<workgroup> fab2_t1: array<vec4<f32>, 256>;
// Each thread's two partial dot products per tile row: [FAB_TILE][FAB_THREADS].
var<workgroup> fab2_part: array<vec2<f32>, 1024>;

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

// Number of head_dim vec4s thread `lane` of a row owns (lane, lane + FAB_TPR, ...).
fn fab2_slices(lane: u32, d4: u32) -> u32 {
    if (lane >= d4) { return 0u; }
    return (d4 - lane + FAB_TPR - 1u) / FAB_TPR;
}

// Sum of the FAB_TPR partials of tile row j for the row whose first thread
// is `row_first`. Every thread of the row adds them in the same order, so
// they agree on the result.
fn fab2_row_sum(j: u32, row_first: u32) -> vec2<f32> {
    var r = vec2<f32>(0.0);
    for (var t = 0u; t < FAB_TPR; t++) { r += fab2_part[j * FAB_THREADS + row_first + t]; }
    return r;
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
    let row = lid / FAB_TPR;
    let lane = lid % FAB_TPR;
    let row_first = row * FAB_TPR;
    let n_sl = fab2_slices(lane, d4);
    let q0 = wg.y * FAB_ROWS;
    let qi = q0 + row;
    let valid = qi < p.seq_q;
    let q_off = fab2_q_off(b, qi, h);

    var q_reg: array<vec4<f32>, 4>;
    var do_reg: array<vec4<f32>, 4>;
    var dq_acc: array<vec4<f32>, 4>;
    var delta4 = vec4<f32>(0.0);
    for (var i = 0u; i < FAB_SLICES; i++) {
        dq_acc[i] = vec4<f32>(0.0);
        q_reg[i] = vec4<f32>(0.0);
        do_reg[i] = vec4<f32>(0.0);
        if (valid && i < n_sl) {
            let c = lane + i * FAB_TPR;
            q_reg[i] = fab2_ld4_q(q_off, c) * p.scale;
            do_reg[i] = fab2_ld4_do(q_off, c);
            delta4 = fma(do_reg[i], fab2_ld4_o(q_off, c), delta4);
        }
    }
    // Delta = dO . O, summed over the row's threads.
    fab2_part[lid] = vec2<f32>(delta4.x + delta4.y + delta4.z + delta4.w, 0.0);
    workgroupBarrier();
    let delta = fab2_row_sum(0u, row_first).x;
    workgroupBarrier();
    let range = fab2_key_range(q0, min(q0 + FAB_ROWS, p.seq_q) - 1u);

    // A fully masked row has no probability mass; make every P_ij zero.
    var lse = 3.0e38;
    if (p.lse_saved == 1u) {
        if (valid) { lse = fab2_stats[fab2_row(b, h, qi)].y; }
    } else {
        // Sweep 1: log-sum-exp of this row's scores.
        var m = FAB_MASKED;
        var l = 0.0;
        for (var k0 = range.x; k0 < range.y; k0 += FAB_TILE) {
            fab2_load_kv(lid, FAB_THREADS, b, kvh, k0, d4);
            workgroupBarrier();
            if (valid) {
                for (var j = 0u; j < FAB_TILE; j++) {
                    var dot4 = vec4<f32>(0.0);
                    for (var i = 0u; i < n_sl; i++) {
                        dot4 = fma(q_reg[i], fab2_t0[j * FAB_D4 + lane + i * FAB_TPR], dot4);
                    }
                    fab2_part[j * FAB_THREADS + lid] = vec2<f32>(dot4.x + dot4.y + dot4.z + dot4.w, 0.0);
                }
            }
            workgroupBarrier();
            if (valid) {
                for (var j = 0u; j < FAB_TILE; j++) {
                    let kj = k0 + j;
                    if (!fab2_visible(qi, kj)) { continue; }
                    let s = fab2_row_sum(j, row_first).x + fab2_mask_at(b, h, qi, kj);
                    if (s > FAB_MASKED) {
                        let m_new = max(m, s);
                        l = l * exp(m - m_new) + exp(s - m_new);
                        m = m_new;
                    }
                }
            }
            workgroupBarrier();
        }
        if (l > 0.0) { lse = m + log(l); }
    }

    // Sweep 2: dQ.
    for (var k0 = range.x; k0 < range.y; k0 += FAB_TILE) {
        fab2_load_kv(lid, FAB_THREADS, b, kvh, k0, d4);
        workgroupBarrier();
        if (valid) {
            for (var j = 0u; j < FAB_TILE; j++) {
                var dot4 = vec4<f32>(0.0);
                var dov4 = vec4<f32>(0.0);
                for (var i = 0u; i < n_sl; i++) {
                    let c = j * FAB_D4 + lane + i * FAB_TPR;
                    dot4 = fma(q_reg[i], fab2_t0[c], dot4);
                    dov4 = fma(do_reg[i], fab2_t1[c], dov4);
                }
                fab2_part[j * FAB_THREADS + lid] = vec2<f32>(
                    dot4.x + dot4.y + dot4.z + dot4.w,
                    dov4.x + dov4.y + dov4.z + dov4.w,
                );
            }
        }
        workgroupBarrier();
        if (valid) {
            for (var j = 0u; j < FAB_TILE; j++) {
                let kj = k0 + j;
                if (!fab2_visible(qi, kj)) { continue; }
                let sums = fab2_row_sum(j, row_first);
                let s = sums.x + fab2_mask_at(b, h, qi, kj);
                if (s <= FAB_MASKED) { continue; }
                let pij = exp(s - lse);
                let mij = dropout_scale(p.dropout_prob, p.seed, p.seed_hi, bh, qi, kj);
                let ds = pij * (mij * sums.y - delta) * p.scale;
                for (var i = 0u; i < n_sl; i++) {
                    dq_acc[i] = fma(vec4<f32>(ds), fab2_t0[j * FAB_D4 + lane + i * FAB_TPR], dq_acc[i]);
                }
            }
        }
        workgroupBarrier();
    }

    if (valid) {
        if (lane == 0u) { fab2_stats[fab2_row(b, h, qi)] = vec2<f32>(delta, lse); }
        for (var i = 0u; i < n_sl; i++) {
            let d = (lane + i * FAB_TPR) * 4u;
            let v = dq_acc[i];
            fab2_dq[q_off + d] = v.x;
            if (d + 1u < p.head_dim) { fab2_dq[q_off + d + 1u] = v.y; }
            if (d + 2u < p.head_dim) { fab2_dq[q_off + d + 2u] = v.z; }
            if (d + 3u < p.head_dim) { fab2_dq[q_off + d + 3u] = v.w; }
        }
    }
}

// Dispatch: (1, ceil(seq_k / FAB_ROWS), batch * num_heads)
@compute @workgroup_size(128)
fn flash_bwd_dk_dv(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let p = fab2_params;
    let d4 = (p.head_dim + 3u) / 4u;
    let b = wg.z / p.num_heads;
    let h = wg.z % p.num_heads;
    let kvh = h / (p.num_heads / p.num_kv_heads);
    let bh = wg.z;
    let row = lid / FAB_TPR;
    let lane = lid % FAB_TPR;
    let row_first = row * FAB_TPR;
    let n_sl = fab2_slices(lane, d4);
    let k0 = wg.y * FAB_ROWS;
    let kj = k0 + row;
    let valid = kj < p.seq_k;
    let kv_off = fab2_kv_off(b, kj, kvh);
    // Where this head's dK and dV go: the KV head's own row, or a per-query-head partial.
    var out_off = kv_off;
    if (p.dkv_direct == 0u) { out_off = ((b * p.seq_k + kj) * p.num_heads + h) * p.head_dim; }
    let k1 = min(k0 + FAB_ROWS, p.seq_k) - 1u;

    var k_reg: array<vec4<f32>, 4>;
    var v_reg: array<vec4<f32>, 4>;
    var dk_acc: array<vec4<f32>, 4>;
    var dv_acc: array<vec4<f32>, 4>;
    for (var i = 0u; i < FAB_SLICES; i++) {
        dk_acc[i] = vec4<f32>(0.0);
        dv_acc[i] = vec4<f32>(0.0);
        k_reg[i] = vec4<f32>(0.0);
        v_reg[i] = vec4<f32>(0.0);
        if (valid && i < n_sl) {
            let c = lane + i * FAB_TPR;
            k_reg[i] = fab2_ld4_k(kv_off, c) * p.scale;
            v_reg[i] = fab2_ld4_v(kv_off, c);
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

    for (var qt = q_begin; qt < q_end; qt += FAB_TILE) {
        // Stage Q (fab2_t0) and dO (fab2_t1) rows [qt, qt + FAB_TILE).
        for (var i = lid; i < FAB_TILE * d4; i += FAB_THREADS) {
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
        if (valid) {
            for (var j = 0u; j < FAB_TILE; j++) {
                var dot4 = vec4<f32>(0.0);
                var dov4 = vec4<f32>(0.0);
                for (var i = 0u; i < n_sl; i++) {
                    let c = j * FAB_D4 + lane + i * FAB_TPR;
                    dot4 = fma(k_reg[i], fab2_t0[c], dot4);
                    dov4 = fma(v_reg[i], fab2_t1[c], dov4);
                }
                fab2_part[j * FAB_THREADS + lid] = vec2<f32>(
                    dot4.x + dot4.y + dot4.z + dot4.w,
                    dov4.x + dov4.y + dov4.z + dov4.w,
                );
            }
        }
        workgroupBarrier();
        if (valid) {
            for (var j = 0u; j < FAB_TILE; j++) {
                let qi = qt + j;
                if (!fab2_visible(qi, kj)) { continue; }
                let sums = fab2_row_sum(j, row_first);
                let s = sums.x + fab2_mask_at(b, h, qi, kj);
                if (s <= FAB_MASKED) { continue; }
                let stats = fab2_stats[fab2_row(b, h, qi)];
                let pij = exp(s - stats.y);
                let mij = dropout_scale(p.dropout_prob, p.seed, p.seed_hi, bh, qi, kj);
                let ds = pij * (mij * sums.y - stats.x) * p.scale;
                let pm = pij * mij;
                for (var i = 0u; i < n_sl; i++) {
                    let c = j * FAB_D4 + lane + i * FAB_TPR;
                    dv_acc[i] = fma(vec4<f32>(pm), fab2_t1[c], dv_acc[i]);
                    dk_acc[i] = fma(vec4<f32>(ds), fab2_t0[c], dk_acc[i]);
                }
            }
        }
        workgroupBarrier();
    }

    if (valid) {
        for (var i = 0u; i < n_sl; i++) {
            let d = (lane + i * FAB_TPR) * 4u;
            let gk = dk_acc[i];
            let gv = dv_acc[i];
            fab2_dk[out_off + d] = gk.x;
            fab2_dv[out_off + d] = gv.x;
            if (d + 1u < p.head_dim) { fab2_dk[out_off + d + 1u] = gk.y; fab2_dv[out_off + d + 1u] = gv.y; }
            if (d + 2u < p.head_dim) { fab2_dk[out_off + d + 2u] = gk.z; fab2_dv[out_off + d + 2u] = gv.z; }
            if (d + 3u < p.head_dim) { fab2_dk[out_off + d + 3u] = gk.w; fab2_dv[out_off + d + 3u] = gv.w; }
        }
    }
}

// dK and dV of each KV head as the sum of its query heads' partials, added in
// head order so a run gives the same bits every time.
// Dispatch: groups of 256 threads over batch * seq_k * num_kv_heads * head_dim.
@compute @workgroup_size(256)
fn flash_group_sum(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let p = fab2_params;
    // Grids wider than 65535 workgroups spill into y.
    let i = gid.y * groups.x * 256u + gid.x;
    if (i >= p.batch_size * p.seq_k * p.num_kv_heads * p.head_dim) { return; }
    let group = p.num_heads / p.num_kv_heads;
    let d = i % p.head_dim;
    // (batch * seq_k + key) * num_kv_heads + kv head: its query heads are the next `group` rows.
    let r = i / p.head_dim;
    var sk = 0.0;
    var sv = 0.0;
    for (var g = 0u; g < group; g++) {
        let o = (r * group + g) * p.head_dim + d;
        sk += fab2_dkp[o];
        sv += fab2_dvp[o];
    }
    fab2_dk[i] = sk;
    fab2_dv[i] = sv;
}
