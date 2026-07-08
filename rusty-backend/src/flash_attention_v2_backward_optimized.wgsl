// Optimized Flash Attention V2 Backward Kernel (Tiled 2-Pass)
// Eliminates Atomic Adds on Global Memory for dK/dV
// Handles [Batch, Seq, Heads, Dim]

// ========================================================================
// Bindings
// ========================================================================

@group(0) @binding(0) var<storage, read> grad_q: array<f32>; // Placeholder (Unused in read)
@group(0) @binding(1) var<storage, read> grad_k: array<f32>; // Placeholder
@group(0) @binding(2) var<storage, read> grad_v: array<f32>; // Placeholder

@group(0) @binding(3) var<storage, read> q: array<f32>;
@group(0) @binding(4) var<storage, read> k: array<f32>;
@group(0) @binding(5) var<storage, read> v: array<f32>;
@group(0) @binding(6) var<storage, read> out: array<f32>;
@group(0) @binding(7) var<storage, read> grad_out: array<f32>;

@group(0) @binding(8) var<storage, read_write> dq: array<f32>;
@group(0) @binding(9) var<storage, read_write> dk: array<f32>;
@group(0) @binding(10) var<storage, read_write> dv: array<f32>;

struct Params {
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
};
@group(0) @binding(11) var<uniform> params: Params;

@group(0) @binding(12) var<storage, read> mask: array<f32>;
struct MaskParams {
    has_mask: u32,
    s_b: u32,
    s_h: u32,
    s_q: u32,
    s_k: u32,
};
@group(0) @binding(13) var<uniform> mask_p: MaskParams;
@group(0) @binding(14) var<storage, read_write> delta: array<f32>;

// Output LSE (LogSumExp)
// Shape: [Batch, Heads, Seq_Q]
@group(0) @binding(15) var<storage, read_write> lse: array<f32>;

// ========================================================================
// Kernel 1: Preprocess (Compute LSE and Delta)
// ========================================================================
// Dispatch: (Seq_Q, Heads, Batch)

@compute @workgroup_size(64)
fn flash_bwd_preprocess(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let q_idx = global_id.x;
    let head_idx = global_id.y;
    let batch_idx = global_id.z;
    
    if (batch_idx >= params.batch_size || head_idx >= params.num_heads || q_idx >= params.seq_q) {
        return;
    }
    
    // Strides
    let s_d = 1u;
    let s_h = params.head_dim;
    let s_q = params.num_heads * params.head_dim;
    let s_b = params.seq_q * params.num_heads * params.head_dim;
    
    // KV Strides (GQA)
    let kv_head_idx = head_idx / (params.num_heads / params.num_kv_heads);
    let s_kv_h = params.head_dim;
    let s_kv_k = params.num_kv_heads * params.head_dim;
    let s_kv_b = params.seq_k * params.num_kv_heads * params.head_dim;
    
    let base_offset = batch_idx * s_b + q_idx * s_q + head_idx * s_h;
    
    // 1. Compute Delta = dO . Out
    var d_val = 0.0;
    for (var d = 0u; d < params.head_dim; d++) {
        d_val += grad_out[base_offset + d] * out[base_offset + d];
    }
    let write_idx = batch_idx * (params.num_heads * params.seq_q) + head_idx * params.seq_q + q_idx;
    delta[write_idx] = d_val;

    // 2. Compute LSE (Max, SumExp)
    // We must recompute forward pass logic.
    let kv_base = batch_idx * s_kv_b + kv_head_idx * s_kv_h; // + k * s_kv_k
    
    var max_score = -1e20;
    var sum_exp = 0.0;
    
    for (var k = 0u; k < params.seq_k; k++) {
        // Causal
        if (params.causal == 1u && k > q_idx) { continue; }
        
        let k_offset = kv_base + k * s_kv_k;
        
        var score = 0.0;
        for (var d = 0u; d < params.head_dim; d++) {
            score += q[base_offset + d] * k[k_offset + d];
        }
        score *= params.scale;
        
        if (mask_p.has_mask == 1u) {
             let m_idx = batch_idx * mask_p.s_b + head_idx * mask_p.s_h + q_idx * mask_p.s_q + k * mask_p.s_k;
             score += mask[m_idx];
        }
        
        if (score > max_score) {
             sum_exp = sum_exp * exp(max_score - score) + 1.0;
             max_score = score;
        } else {
             sum_exp += exp(score - max_score);
        }
    }
    
    let l_val = max_score + log(sum_exp);
    lse[write_idx] = l_val;
}


// ========================================================================
// Kernel 2: Compute dQ
// ========================================================================
// Dispatch: (Seq_Q, Heads, Batch)

@compute @workgroup_size(64)
fn flash_bwd_dq(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let q_idx = global_id.x;
    let head_idx = global_id.y;
    let batch_idx = global_id.z;
    
    if (batch_idx >= params.batch_size || head_idx >= params.num_heads || q_idx >= params.seq_q) {
        return;
    }
    
    // Strides
    let s_h = params.head_dim;
    let s_q = params.num_heads * params.head_dim;
    let s_b = params.seq_q * params.num_heads * params.head_dim;
    let q_offset = batch_idx * s_b + q_idx * s_q + head_idx * s_h;
    
    // KV Strides
    let kv_head_idx = head_idx / (params.num_heads / params.num_kv_heads);
    let s_kv_h = params.head_dim;
    let s_kv_k = params.num_kv_heads * params.head_dim;
    let s_kv_b = params.seq_k * params.num_kv_heads * params.head_dim;
    let kv_base = batch_idx * s_kv_b + kv_head_idx * s_kv_h;
    
    // Load LSE and Delta
    let lse_idx = batch_idx * (params.num_heads * params.seq_q) + head_idx * params.seq_q + q_idx;
    let l_val = lse[lse_idx];
    let d_val = delta[lse_idx];
    
    var dq_acc: array<f32, 128>;
    for(var i=0u; i<params.head_dim; i++) { dq_acc[i] = 0.0; }
    
    // Loop over K
    for (var k = 0u; k < params.seq_k; k++) {
        if (params.causal == 1u && k > q_idx) { continue; }
        
        let k_offset = kv_base + k * s_kv_k;
        let v_offset = kv_base + k * s_kv_k; // Same
        
        // Score
        var score = 0.0;
        for (var d = 0u; d < params.head_dim; d++) {
            score += q[q_offset + d] * k[k_offset + d];
        }
        score *= params.scale;
         if (mask_p.has_mask == 1u) {
             let m_idx = batch_idx * mask_p.s_b + head_idx * mask_p.s_h + q_idx * mask_p.s_q + k * mask_p.s_k;
             score += mask[m_idx];
        }
        
        // P_ij = exp(S_ij - L_i)
        var p_ij = exp(score - l_val);
        
        // Dropout
        let rand_val = hash3(batch_idx * params.num_heads + head_idx, q_idx, k + params.seed);
        let keep = step(params.dropout_prob, rand_val);
        let dropout_scale = 1.0 / (1.0 - params.dropout_prob + 0.00001);
        p_ij = p_ij * keep * dropout_scale;
        
        // Term
        var dv_dot = 0.0;
        for (var d = 0u; d < params.head_dim; d++) {
            dv_dot += grad_out[q_offset + d] * v[v_offset + d];
        }
        let term = p_ij * (dv_dot - d_val) * params.scale;
        
        // Accumulate dQ
        for (var d = 0u; d < params.head_dim; d++) {
            dq_acc[d] += term * k[k_offset + d];
        }
    }
    
    // Write dQ
    for (var d = 0u; d < params.head_dim; d++) {
        dq[q_offset + d] = dq_acc[d];
    }
}


// ========================================================================
// Kernel 3: Compute dK, dV
// ========================================================================
// Dispatch: (Seq_K, KV_Heads, Batch)

@compute @workgroup_size(64)
fn flash_bwd_dk_dv(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let k_idx = global_id.x;
    let kv_head_idx = global_id.y;
    let batch_idx = global_id.z;
    
    if (batch_idx >= params.batch_size || kv_head_idx >= params.num_kv_heads || k_idx >= params.seq_k) {
        return;
    }
    
    // KV Strides
    let s_kv_h = params.head_dim;
    let s_kv_k = params.num_kv_heads * params.head_dim;
    let s_kv_b = params.seq_k * params.num_kv_heads * params.head_dim;
    let kv_offset = batch_idx * s_kv_b + k_idx * s_kv_k + kv_head_idx * s_kv_h;
    
    // Q Strides
    let s_h = params.head_dim;
    let s_q = params.num_heads * params.head_dim;
    let s_b = params.seq_q * params.num_heads * params.head_dim;
    let q_headers_per_kv = params.num_heads / params.num_kv_heads;
    
    var dk_acc: array<f32, 128>;
    var dv_acc: array<f32, 128>;
    for(var i=0u; i<params.head_dim; i++) { dk_acc[i] = 0.0; dv_acc[i] = 0.0; }
    
    // Iterate over Q heads that share this KV head
    for (var h = 0u; h < q_headers_per_kv; h++) {
        let head_idx = kv_head_idx * q_headers_per_kv + h;
        
        // Iterate over Q tokens
        for (var q_idx = 0u; q_idx < params.seq_q; q_idx++) {
            if (params.causal == 1u && k_idx > q_idx) { continue; }
            
            let q_offset = batch_idx * s_b + q_idx * s_q + head_idx * s_h;
            
            // Recompute Score
            var score = 0.0;
            for (var d = 0u; d < params.head_dim; d++) {
                score += q[q_offset + d] * k[kv_offset + d];
            }
            score *= params.scale;
             if (mask_p.has_mask == 1u) {
                 let m_idx = batch_idx * mask_p.s_b + head_idx * mask_p.s_h + q_idx * mask_p.s_q + k_idx * mask_p.s_k;
                 score += mask[m_idx];
            }
            
            // Load LSE
            let lse_idx = batch_idx * (params.num_heads * params.seq_q) + head_idx * params.seq_q + q_idx;
            let l_val = lse[lse_idx];
            let d_val = delta[lse_idx];
            
            // P_ij
            var p_ij = exp(score - l_val);
            
            // Dropout
            let rand_val = hash3(batch_idx * params.num_heads + head_idx, q_idx, k_idx + params.seed);
            let keep = step(params.dropout_prob, rand_val);
            let dropout_scale = 1.0 / (1.0 - params.dropout_prob + 0.00001);
            p_ij = p_ij * keep * dropout_scale;
            
            // dV contribution: P_ij * dO_i
            // dK contribution: P_ij * (dO_i . V_j - D_i) * scale * Q_i
            // Note: V_j is k[kv_offset] (same memory for k/v in input binding implies same loop, but logic differs)
            
            // Wait: V is separate buffer 'v', K is 'k'.
            // kv_offset is same for both if layout matches.
            // Check bindings: 4=k, 5=v.
            
            for (var d = 0u; d < params.head_dim; d++) {
                dv_acc[d] += p_ij * grad_out[q_offset + d];
            }
            
            // Term for dK
             var dv_dot = 0.0;
             for (var d = 0u; d < params.head_dim; d++) {
                dv_dot += grad_out[q_offset + d] * v[kv_offset + d];
             }
             let term = p_ij * (dv_dot - d_val) * params.scale;
             
             for (var d = 0u; d < params.head_dim; d++) {
                 dk_acc[d] += term * q[q_offset + d];
             }
        }
    }
    
    // Write dK, dV
    for (var d = 0u; d < params.head_dim; d++) {
        dk[kv_offset + d] = dk_acc[d];
        dv[kv_offset + d] = dv_acc[d];
    }
}
