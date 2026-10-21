// ============================================================================
// FLASH ATTENTION V2 - backward pass
// ============================================================================
// Same layout, masking and dropout conventions as flash_attention_v2.wgsl
// (uses its hash3 / dropout_scale helpers). Three passes, no atomics:
//   1. flash_bwd_preprocess: per query row, Delta = dO . O and the
//      log-sum-exp L of the scores.
//   2. flash_bwd_dq: per query row, dQ = scale * sum_j dS_ij K_j.
//   3. flash_bwd_dk_dv: per key row, dV_j = sum_i P_ij M_ij dO_i and
//      dK_j = scale * sum_i dS_ij Q_i (summed over the query heads that
//      share this KV head).
// with P = softmax(S), M the dropout multiplier and
//      dS_ij = P_ij * (M_ij * (dO_i . V_j) - Delta_i).
// Each pass uses at most 8 storage buffers (the WebGPU default limit).

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
    _pad: u32,
    mask_stride_b: u32,
    mask_stride_h: u32,
    mask_stride_q: u32,
    mask_stride_k: u32,
};
@group(0) @binding(8) var<uniform> fab2_params: FlashBwdParams;
@group(0) @binding(9) var<storage, read> fab2_mask: array<f32>;
// Per (batch, head, query) row: [Delta, LSE].
@group(0) @binding(10) var<storage, read_write> fab2_stats: array<vec2<f32>>;

const FAB_MASKED: f32 = -1.0e30;

// Scaled, masked score for (q row offset, k row offset); FAB_MASKED if the
// key is not visible.
fn fab2_score(q_off: u32, k_off: u32, b: u32, h: u32, qi: u32, kj: u32) -> f32 {
    let p = fab2_params;
    if (p.causal == 1u && kj > qi) { return FAB_MASKED; }
    var dot = 0.0;
    for (var d = 0u; d < p.head_dim; d++) {
        dot += fab2_q[q_off + d] * fab2_k[k_off + d];
    }
    var s = dot * p.scale;
    if (p.has_mask == 1u) {
        s += fab2_mask[b * p.mask_stride_b + h * p.mask_stride_h + qi * p.mask_stride_q + kj * p.mask_stride_k];
    }
    return s;
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

// Dispatch: (ceil(seq_q / 64), num_heads, batch)
@compute @workgroup_size(64)
fn flash_bwd_preprocess(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = fab2_params;
    let qi = gid.x;
    let h = gid.y;
    let b = gid.z;
    if (qi >= p.seq_q) { return; }
    let kvh = h / (p.num_heads / p.num_kv_heads);
    let q_off = fab2_q_off(b, qi, h);

    var delta = 0.0;
    for (var d = 0u; d < p.head_dim; d++) {
        delta += fab2_grad_out[q_off + d] * fab2_out[q_off + d];
    }

    var m = FAB_MASKED;
    var l = 0.0;
    for (var kj = 0u; kj < p.seq_k; kj++) {
        let s = fab2_score(q_off, fab2_kv_off(b, kj, kvh), b, h, qi, kj);
        if (s > FAB_MASKED) {
            let m_new = max(m, s);
            l = l * exp(m - m_new) + exp(s - m_new);
            m = m_new;
        }
    }
    // A fully masked row has no probability mass; make every P_ij zero.
    var lse = 3.0e38;
    if (l > 0.0) { lse = m + log(l); }
    fab2_stats[fab2_row(b, h, qi)] = vec2<f32>(delta, lse);
}

// Dispatch: (ceil(seq_q / 64), num_heads, batch)
@compute @workgroup_size(64)
fn flash_bwd_dq(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = fab2_params;
    let qi = gid.x;
    let h = gid.y;
    let b = gid.z;
    if (qi >= p.seq_q) { return; }
    let kvh = h / (p.num_heads / p.num_kv_heads);
    let q_off = fab2_q_off(b, qi, h);
    let stats = fab2_stats[fab2_row(b, h, qi)];
    let bh = b * p.num_heads + h;

    var dq_acc: array<f32, 128>;
    for (var d = 0u; d < p.head_dim; d++) { dq_acc[d] = 0.0; }

    for (var kj = 0u; kj < p.seq_k; kj++) {
        let kv_off = fab2_kv_off(b, kj, kvh);
        let s = fab2_score(q_off, kv_off, b, h, qi, kj);
        if (s <= FAB_MASKED) { continue; }
        let pij = exp(s - stats.y);
        var dov = 0.0;
        for (var d = 0u; d < p.head_dim; d++) {
            dov += fab2_grad_out[q_off + d] * fab2_v[kv_off + d];
        }
        let mij = dropout_scale(p.dropout_prob, p.seed, bh, qi, kj);
        let ds = pij * (mij * dov - stats.x) * p.scale;
        for (var d = 0u; d < p.head_dim; d++) {
            dq_acc[d] += ds * fab2_k[kv_off + d];
        }
    }

    for (var d = 0u; d < p.head_dim; d++) {
        fab2_dq[q_off + d] = dq_acc[d];
    }
}

// Dispatch: (ceil(seq_k / 64), num_kv_heads, batch)
@compute @workgroup_size(64)
fn flash_bwd_dk_dv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = fab2_params;
    let kj = gid.x;
    let kvh = gid.y;
    let b = gid.z;
    if (kj >= p.seq_k) { return; }
    let kv_off = fab2_kv_off(b, kj, kvh);
    let group = p.num_heads / p.num_kv_heads;

    var dk_acc: array<f32, 128>;
    var dv_acc: array<f32, 128>;
    for (var d = 0u; d < p.head_dim; d++) { dk_acc[d] = 0.0; dv_acc[d] = 0.0; }

    for (var g = 0u; g < group; g++) {
        let h = kvh * group + g;
        let bh = b * p.num_heads + h;
        // Causal: only queries at or after this key see it.
        var q_first = 0u;
        if (p.causal == 1u) { q_first = kj; }
        for (var qi = q_first; qi < p.seq_q; qi++) {
            let q_off = fab2_q_off(b, qi, h);
            let s = fab2_score(q_off, kv_off, b, h, qi, kj);
            if (s <= FAB_MASKED) { continue; }
            let stats = fab2_stats[fab2_row(b, h, qi)];
            let pij = exp(s - stats.y);
            let mij = dropout_scale(p.dropout_prob, p.seed, bh, qi, kj);
            var dov = 0.0;
            for (var d = 0u; d < p.head_dim; d++) {
                dov += fab2_grad_out[q_off + d] * fab2_v[kv_off + d];
            }
            let ds = pij * (mij * dov - stats.x) * p.scale;
            let pm = pij * mij;
            for (var d = 0u; d < p.head_dim; d++) {
                dv_acc[d] += pm * fab2_grad_out[q_off + d];
                dk_acc[d] += ds * fab2_q[q_off + d];
            }
        }
    }

    for (var d = 0u; d < p.head_dim; d++) {
        fab2_dk[kv_off + d] = dk_acc[d];
        fab2_dv[kv_off + d] = dv_acc[d];
    }
}
