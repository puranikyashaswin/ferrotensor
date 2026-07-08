//! GPU Compute Kernels (WGSL)
//!
//! All GPU compute shaders are defined here in WebGPU Shading Language.

pub const KERNELS_WGSL: &str = r#"
// ============================================================================
// RUSTY ML LIBRARY - GPU COMPUTE KERNELS
// ============================================================================
// High-performance compute shaders for neural network operations.
// Optimized for Apple Metal via wgpu.
// ============================================================================

// --- Constants ---
const TILE_SIZE: u32 = 16u;
const WORKGROUP_SIZE: u32 = 64u;

// ============================================================================
// SECTION 1: TENSOR OPERATIONS
// ============================================================================

// --- 1.1 Element-wise Binary Operations ---
@group(0) @binding(0) var<storage, read> bin_a: array<f32>;
@group(0) @binding(1) var<storage, read> bin_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> bin_out: array<f32>;

@compute @workgroup_size(64)
fn add(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= arrayLength(&bin_out)) { return; }
    bin_out[i] = bin_a[i] + bin_b[i];
}

@compute @workgroup_size(64)
fn sub(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= arrayLength(&bin_out)) { return; }
    bin_out[i] = bin_a[i] - bin_b[i];
}

@compute @workgroup_size(64)
fn mul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= arrayLength(&bin_out)) { return; }
    bin_out[i] = bin_a[i] * bin_b[i];
}

@compute @workgroup_size(64)
fn div_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= arrayLength(&bin_out)) { return; }
    bin_out[i] = bin_a[i] / bin_b[i];
}

// --- 1.2 Scalar Operations ---
@group(0) @binding(0) var<storage, read> scalar_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> scalar_out: array<f32>;
@group(0) @binding(2) var<uniform> scalar_val: f32;

@compute @workgroup_size(64)
fn add_scalar(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= arrayLength(&scalar_out)) { return; }
    scalar_out[i] = scalar_in[i] + scalar_val;
}

@compute @workgroup_size(64)
fn mul_scalar(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= arrayLength(&scalar_out)) { return; }
    scalar_out[i] = scalar_in[i] * scalar_val;
}

// --- 1.3 Tiled Matrix Multiplication ---
@group(0) @binding(0) var<storage, read> mm_a: array<f32>;
@group(0) @binding(1) var<storage, read> mm_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> mm_out: array<f32>;
@group(0) @binding(3) var<uniform> mm_params: vec4<u32>; // M, K, N, padding

var<workgroup> tile_a: array<array<f32, 16>, 16>;
var<workgroup> tile_b: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16, 1)
fn matmul(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;
    let m = mm_params.x;
    let k = mm_params.y;
    let n = mm_params.z;

    var acc = 0.0;
    let num_tiles = (k + 15u) / 16u;

    for (var t = 0u; t < num_tiles; t = t + 1u) {
        let r = row;
        let c = t * 16u + local_id.x;
        if (r < m && c < k) {
            tile_a[local_id.y][local_id.x] = mm_a[r * k + c];
        } else {
            tile_a[local_id.y][local_id.x] = 0.0;
        }

        let r_b = t * 16u + local_id.y;
        let c_b = col;
        if (r_b < k && c_b < n) {
            tile_b[local_id.y][local_id.x] = mm_b[r_b * n + c_b];
        } else {
            tile_b[local_id.y][local_id.x] = 0.0;
        }

        workgroupBarrier();
        for (var i = 0u; i < 16u; i = i + 1u) {
            acc = acc + tile_a[local_id.y][i] * tile_b[i][local_id.x];
        }
        workgroupBarrier();
    }

    if (row < m && col < n) {
        mm_out[row * n + col] = acc;
    }
}

// ============================================================================
// SECTION 2: ACTIVATION FUNCTIONS
// ============================================================================

@group(0) @binding(0) var<storage, read> act_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> act_out: array<f32>;

@compute @workgroup_size(64)
fn relu(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= arrayLength(&act_out)) { return; }
    act_out[i] = max(0.0, act_in[i]);
}

@compute @workgroup_size(64)
fn silu(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= arrayLength(&act_out)) { return; }
    let x = act_in[i];
    act_out[i] = x / (1.0 + exp(-x));
}

@compute @workgroup_size(64)
fn gelu(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= arrayLength(&act_out)) { return; }
    let x = act_in[i];
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    let sqrt_2_over_pi = 0.7978845608;
    let coeff = 0.044715;
    let inner = sqrt_2_over_pi * (x + coeff * x * x * x);
    act_out[i] = 0.5 * x * (1.0 + tanh(inner));
}

@compute @workgroup_size(64)
fn sigmoid(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= arrayLength(&act_out)) { return; }
    act_out[i] = 1.0 / (1.0 + exp(-act_in[i]));
}

@compute @workgroup_size(64)
fn tanh_act(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= arrayLength(&act_out)) { return; }
    act_out[i] = tanh(act_in[i]);
}

// ============================================================================
// SECTION 3: NORMALIZATION
// ============================================================================

// --- 3.1 RMS Normalization ---
@group(0) @binding(0) var<storage, read> rms_input: array<f32>;
@group(0) @binding(1) var<storage, read> rms_weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> rms_out: array<f32>;
@group(0) @binding(3) var<uniform> rms_params: vec2<u32>; // size, eps_bits

@compute @workgroup_size(64)
fn rms_norm(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let size = rms_params.x;
    if (row >= 1u) { return; }
    
    var ss = 0.0;
    for (var i = 0u; i < size; i = i + 1u) {
        let val = rms_input[i];
        ss = ss + val * val;
    }
    ss = ss / f32(size);
    let inv_rms = 1.0 / sqrt(ss + 1e-5);
    
    for (var i = 0u; i < size; i = i + 1u) {
        rms_out[i] = rms_input[i] * inv_rms * rms_weight[i];
    }
}

// --- 3.2 Layer Normalization ---
@group(0) @binding(0) var<storage, read> ln_input: array<f32>;
@group(0) @binding(1) var<storage, read> ln_gamma: array<f32>;
@group(0) @binding(2) var<storage, read> ln_beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> ln_out: array<f32>;
@group(0) @binding(4) var<uniform> ln_params: vec2<u32>; // size, eps_bits

@compute @workgroup_size(1)
fn layer_norm(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let size = ln_params.x;
    let eps = 1e-5;
    
    // Compute mean
    var mean = 0.0;
    for (var i = 0u; i < size; i = i + 1u) {
        mean = mean + ln_input[i];
    }
    mean = mean / f32(size);
    
    // Compute variance
    var variance = 0.0;
    for (var i = 0u; i < size; i = i + 1u) {
        let diff = ln_input[i] - mean;
        variance = variance + diff * diff;
    }
    variance = variance / f32(size);
    
    // Normalize
    let inv_std = 1.0 / sqrt(variance + eps);
    for (var i = 0u; i < size; i = i + 1u) {
        let normalized = (ln_input[i] - mean) * inv_std;
        ln_out[i] = normalized * ln_gamma[i] + ln_beta[i];
    }
}

// ============================================================================
// SECTION 4: SOFTMAX & CROSS-ENTROPY
// ============================================================================

// --- 4.1 Softmax ---
@group(0) @binding(0) var<storage, read> sm_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> sm_out: array<f32>;
@group(0) @binding(2) var<uniform> sm_size: u32;

@compute @workgroup_size(1)
fn softmax(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let len = sm_size;
    
    // Find max for numerical stability
    var max_val = -1e9;
    for (var i = 0u; i < len; i = i + 1u) {
        max_val = max(max_val, sm_input[i]);
    }
    
    // Compute exp and sum
    var sum_exp = 0.0;
    for (var i = 0u; i < len; i = i + 1u) {
        let val = exp(sm_input[i] - max_val);
        sm_out[i] = val;
        sum_exp = sum_exp + val;
    }
    
    // Normalize
    for (var i = 0u; i < len; i = i + 1u) {
        sm_out[i] = sm_out[i] / sum_exp;
    }
}

// --- 4.2 Log Softmax ---
@compute @workgroup_size(1)
fn log_softmax(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let len = sm_size;
    
    // Find max for numerical stability
    var max_val = -1e9;
    for (var i = 0u; i < len; i = i + 1u) {
        max_val = max(max_val, sm_input[i]);
    }
    
    // Compute log(sum(exp(x - max)))
    var sum_exp = 0.0;
    for (var i = 0u; i < len; i = i + 1u) {
        sum_exp = sum_exp + exp(sm_input[i] - max_val);
    }
    let log_sum_exp = log(sum_exp) + max_val;
    
    // log_softmax = x - log_sum_exp
    for (var i = 0u; i < len; i = i + 1u) {
        sm_out[i] = sm_input[i] - log_sum_exp;
    }
}

// --- 4.3 Cross-Entropy Loss ---
@group(0) @binding(0) var<storage, read> ce_logits: array<f32>;  // [batch, vocab]
@group(0) @binding(1) var<storage, read> ce_targets: array<u32>; // [batch]
@group(0) @binding(2) var<storage, read_write> ce_loss: array<f32>; // [batch]
@group(0) @binding(3) var<uniform> ce_params: vec2<u32>; // batch_size, vocab_size

@compute @workgroup_size(64)
fn cross_entropy(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    let batch_size = ce_params.x;
    let vocab_size = ce_params.y;
    
    if (batch_idx >= batch_size) { return; }
    
    let offset = batch_idx * vocab_size;
    let target = ce_targets[batch_idx];
    
    // Find max for numerical stability
    var max_val = -1e9;
    for (var i = 0u; i < vocab_size; i = i + 1u) {
        max_val = max(max_val, ce_logits[offset + i]);
    }
    
    // Compute log_sum_exp
    var sum_exp = 0.0;
    for (var i = 0u; i < vocab_size; i = i + 1u) {
        sum_exp = sum_exp + exp(ce_logits[offset + i] - max_val);
    }
    let log_sum_exp = log(sum_exp) + max_val;
    
    // Cross-entropy = -log_softmax[target]
    ce_loss[batch_idx] = log_sum_exp - ce_logits[offset + target];
}

// ============================================================================
// SECTION 5: POSITIONAL ENCODINGS
// ============================================================================

// --- 5.1 Rotary Position Embeddings (RoPE) ---
@group(0) @binding(0) var<storage, read_write> rope_data: array<f32>;
@group(0) @binding(1) var<uniform> rope_params: vec4<u32>; // head_dim, n_heads, pos, padding

@compute @workgroup_size(64)
fn rope(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let head_dim = rope_params.x;
    let half_dim = head_dim / 2u;
    let token_idx = idx / half_dim;
    let dim_idx = idx % half_dim;
    let pos = f32(rope_params.z);
    
    let freq_exp = -2.0 * f32(dim_idx) / f32(head_dim);
    let angle = pos * pow(10000.0, freq_exp);
    let cos_a = cos(angle);
    let sin_a = sin(angle);
    
    let idx1 = token_idx * head_dim + dim_idx;
    let idx2 = idx1 + half_dim;
    
    if (idx1 >= arrayLength(&rope_data)) { return; }
    
    let v1 = rope_data[idx1];
    let v2 = rope_data[idx2];
    rope_data[idx1] = v1 * cos_a - v2 * sin_a;
    rope_data[idx2] = v1 * sin_a + v2 * cos_a;
}

// ============================================================================
// SECTION 6: EMBEDDING OPERATIONS
// ============================================================================

@group(0) @binding(0) var<storage, read> emb_table: array<f32>;  // [vocab, dim]
@group(0) @binding(1) var<storage, read> emb_indices: array<u32>; // [seq_len]
@group(0) @binding(2) var<storage, read_write> emb_out: array<f32>; // [seq_len, dim]
@group(0) @binding(3) var<uniform> emb_params: vec2<u32>; // seq_len, dim

@compute @workgroup_size(64)
fn embedding_lookup(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let seq_len = emb_params.x;
    let dim = emb_params.y;
    let total = seq_len * dim;
    
    if (idx >= total) { return; }
    
    let seq_idx = idx / dim;
    let dim_idx = idx % dim;
    let token_id = emb_indices[seq_idx];
    
    emb_out[idx] = emb_table[token_id * dim + dim_idx];
}

// ============================================================================
// SECTION 7: BACKWARD KERNELS (AUTOGRAD)
// ============================================================================

// --- 7.1 Binary Op Backwards ---
@group(0) @binding(3) var<storage, read> bk_grad_out: array<f32>;
@group(0) @binding(4) var<storage, read_write> bk_grad_a: array<f32>;
@group(0) @binding(5) var<storage, read_write> bk_grad_b: array<f32>;

@compute @workgroup_size(64)
fn add_backward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= arrayLength(&bk_grad_out)) { return; }
    let g = bk_grad_out[i];
    bk_grad_a[i] = g;
    bk_grad_b[i] = g;
}

@compute @workgroup_size(64)
fn sub_backward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= arrayLength(&bk_grad_out)) { return; }
    let g = bk_grad_out[i];
    bk_grad_a[i] = g;
    bk_grad_b[i] = -g;
}

@group(0) @binding(3) var<storage, read> mul_bk_grad: array<f32>;
@group(0) @binding(4) var<storage, read> mul_bk_a: array<f32>;
@group(0) @binding(5) var<storage, read> mul_bk_b: array<f32>;
@group(0) @binding(6) var<storage, read_write> mul_bk_grad_a: array<f32>;
@group(0) @binding(7) var<storage, read_write> mul_bk_grad_b: array<f32>;

@compute @workgroup_size(64)
fn mul_backward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= arrayLength(&mul_bk_grad)) { return; }
    let g = mul_bk_grad[i];
    mul_bk_grad_a[i] = g * mul_bk_b[i];
    mul_bk_grad_b[i] = g * mul_bk_a[i];
}

// --- 7.2 Activation Backwards ---
@group(0) @binding(3) var<storage, read> act_bk_grad: array<f32>;
@group(0) @binding(4) var<storage, read> act_bk_input: array<f32>;
@group(0) @binding(5) var<storage, read_write> act_bk_grad_in: array<f32>;

@compute @workgroup_size(64)
fn relu_backward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= arrayLength(&act_bk_grad)) { return; }
    if (act_bk_input[i] > 0.0) {
        act_bk_grad_in[i] = act_bk_grad[i];
    } else {
        act_bk_grad_in[i] = 0.0;
    }
}

@compute @workgroup_size(64)
fn silu_backward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= arrayLength(&act_bk_grad)) { return; }
    let x = act_bk_input[i];
    let s = 1.0 / (1.0 + exp(-x));
    act_bk_grad_in[i] = act_bk_grad[i] * s * (1.0 + x * (1.0 - s));
}

@compute @workgroup_size(64)
fn gelu_backward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= arrayLength(&act_bk_grad)) { return; }
    let x = act_bk_input[i];
    // GELU derivative approximation
    let sqrt_2_over_pi = 0.7978845608;
    let coeff = 0.044715;
    let inner = sqrt_2_over_pi * (x + coeff * x * x * x);
    let tanh_inner = tanh(inner);
    let sech2 = 1.0 - tanh_inner * tanh_inner;
    let inner_deriv = sqrt_2_over_pi * (1.0 + 3.0 * coeff * x * x);
    let gelu_deriv = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * inner_deriv;
    act_bk_grad_in[i] = act_bk_grad[i] * gelu_deriv;
}

// --- 7.3 MatMul Backwards ---
@group(0) @binding(0) var<storage, read> mm_bk_grad: array<f32>;
@group(0) @binding(1) var<storage, read> mm_bk_other: array<f32>;
@group(0) @binding(2) var<storage, read_write> mm_bk_out: array<f32>;
@group(0) @binding(3) var<uniform> mm_bk_params: vec4<u32>; // M, K, N, padding

@compute @workgroup_size(16, 16, 1)
fn matmul_bt(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;
    let m = mm_bk_params.x;
    let k = mm_bk_params.y;
    let n = mm_bk_params.z;
    if (row >= m || col >= n) { return; }
    var acc = 0.0;
    for (var i = 0u; i < k; i = i + 1u) {
        acc = acc + mm_bk_grad[row * k + i] * mm_bk_other[col * k + i];
    }
    mm_bk_out[row * n + col] = acc;
}

@compute @workgroup_size(16, 16, 1)
fn matmul_at(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;
    let m = mm_bk_params.x;
    let k = mm_bk_params.y;
    let n = mm_bk_params.z;
    if (row >= m || col >= n) { return; }
    var acc = 0.0;
    for (var i = 0u; i < k; i = i + 1u) {
        acc = acc + mm_bk_other[i * m + row] * mm_bk_grad[i * n + col];
    }
    mm_bk_out[row * n + col] = acc;
}

// --- 7.4 Softmax Backward ---
@group(0) @binding(3) var<storage, read> sm_bk_grad: array<f32>;
@group(0) @binding(4) var<storage, read> sm_bk_out: array<f32>;
@group(0) @binding(5) var<storage, read_write> sm_bk_grad_in: array<f32>;

@compute @workgroup_size(1)
fn softmax_backward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let len = sm_size;
    var sum_g_s = 0.0;
    for (var i = 0u; i < len; i = i + 1u) {
        sum_g_s = sum_g_s + sm_bk_grad[i] * sm_bk_out[i];
    }
    for (var i = 0u; i < len; i = i + 1u) {
        let s = sm_bk_out[i];
        sm_bk_grad_in[i] = s * (sm_bk_grad[i] - sum_g_s);
    }
}

// ============================================================================
// SECTION 8: OPTIMIZER KERNELS
// ============================================================================

@group(0) @binding(0) var<storage, read_write> opt_weight: array<f32>;
@group(0) @binding(1) var<storage, read> opt_grad: array<f32>;
@group(0) @binding(2) var<uniform> opt_lr: f32;

@compute @workgroup_size(64)
fn sgd_update(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= arrayLength(&opt_weight)) { return; }
    opt_weight[i] = opt_weight[i] - opt_lr * opt_grad[i];
}

struct AdamParams {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    t: f32,
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(3) var<storage, read_write> adam_m: array<f32>;
@group(0) @binding(4) var<storage, read_write> adam_v: array<f32>;
@group(0) @binding(5) var<uniform> adam_params: AdamParams;

@compute @workgroup_size(64)
fn adamw_update(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= arrayLength(&opt_weight)) { return; }
    
    let g = opt_grad[i];
    let m = adam_m[i] * adam_params.beta1 + (1.0 - adam_params.beta1) * g;
    let v = adam_v[i] * adam_params.beta2 + (1.0 - adam_params.beta2) * g * g;
    
    adam_m[i] = m;
    adam_v[i] = v;
    
    let m_hat = m / (1.0 - pow(adam_params.beta1, adam_params.t));
    let v_hat = v / (1.0 - pow(adam_params.beta2, adam_params.t));
    
    var w = opt_weight[i];
    w = w - adam_params.lr * (m_hat / (sqrt(v_hat) + adam_params.eps) + adam_params.weight_decay * w);
    opt_weight[i] = w;
}

// ============================================================================
// SECTION 9: LOSS FUNCTIONS
// ============================================================================

@group(0) @binding(0) var<storage, read> mse_pred: array<f32>;
@group(0) @binding(1) var<storage, read> mse_target: array<f32>;
@group(0) @binding(2) var<storage, read_write> mse_out: array<f32>;

@compute @workgroup_size(64)
fn mse_loss(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= arrayLength(&mse_out)) { return; }
    let diff = mse_pred[i] - mse_target[i];
    mse_out[i] = diff * diff;
}

@group(0) @binding(3) var<storage, read_write> mse_grad: array<f32>;

@compute @workgroup_size(64)
fn mse_loss_backward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= arrayLength(&mse_pred)) { return; }
    mse_grad[i] = 2.0 * (mse_pred[i] - mse_target[i]);
}

// ============================================================================
// SECTION 10: QUANTIZATION
// ============================================================================

@group(0) @binding(0) var<storage, read> q_weight: array<u32>;
@group(0) @binding(1) var<storage, read> q_scales: array<f32>;
@group(0) @binding(2) var<storage, read> q_biases: array<f32>;
@group(0) @binding(3) var<storage, read_write> q_out: array<f32>;

@compute @workgroup_size(64)
fn affine_dequantize(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_idx = global_id.x;
    if (out_idx >= arrayLength(&q_out)) { return; }
    
    let u32_idx = out_idx / 4u;
    let byte_inner = out_idx % 4u;
    let val_u32 = q_weight[u32_idx];
    let val_u8 = f32((val_u32 >> (byte_inner * 8u)) & 0xFFu);
    
    let group_idx = out_idx / 64u;
    let scale = q_scales[group_idx];
    let bias = q_biases[group_idx];
    
    q_out[out_idx] = val_u8 * scale + bias;
}

// ============================================================================
// SECTION 11: TENSOR UTILITIES
// ============================================================================

@group(0) @binding(0) var<storage, read> copy_src: array<f32>;
@group(0) @binding(1) var<storage, read_write> copy_dst: array<f32>;

@compute @workgroup_size(64)
fn copy(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= arrayLength(&copy_dst)) { return; }
    copy_dst[i] = copy_src[i];
}

@group(0) @binding(0) var<storage, read_write> fill_buf: array<f32>;
@group(0) @binding(1) var<uniform> fill_val: f32;

@compute @workgroup_size(64)
fn fill(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= arrayLength(&fill_buf)) { return; }
    fill_buf[i] = fill_val;
}
"#;
