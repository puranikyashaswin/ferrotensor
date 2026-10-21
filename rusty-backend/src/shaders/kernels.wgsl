// --- Constants ---
const TILE_SIZE: u32 = 16u; // Workgroup tile size for MatMul

// --- Dispatch ---
// 1-D kernels may be launched on a 2-D grid when they need more than 65535
// workgroups; they rebuild a linear index from gid.x + gid.y * grid width.

// --- Bindings ---
// We use generic bindings (0,1,2,3) to reuse layouts across pipelines where possible

// --- 1. Tiled Matrix Multiplication (Optimized) ---
@group(0) @binding(0) var<storage, read> mm_a: array<f32>;
@group(0) @binding(1) var<storage, read> mm_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> mm_out: array<f32>;
@group(0) @binding(3) var<uniform> mm_params: vec4<u32>; // M, K, N, Padding

var<workgroup> tile_a: array<array<f32, 16>, 16>;
var<workgroup> tile_b: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16, 1)
fn matmul_tiled(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
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

// --- 2. Rotary Positional Embeddings (RoPE) ---
@group(0) @binding(0) var<storage, read_write> rope_data: array<f32>;
@group(0) @binding(1) var<uniform> rope_params: vec4<u32>;

@compute @workgroup_size(64)
fn rope(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let global_id = vec3<u32>(gid.x + gid.y * nwg.x * 64u, 0u, 0u);
    let idx = global_id.x;
    let head_dim = rope_params.x;
    let half_dim = head_dim / 2u;
    let token_idx = idx / half_dim;
    let dim_idx = idx % half_dim;
    let seq_len = rope_params.y;
    let base_pos = rope_params.z;
    let heads = max(rope_params.w, 1u); // head vectors per token
    let pos = f32(base_pos + (token_idx / heads) % seq_len);
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

// --- 3. RMSNorm ---
@group(0) @binding(0) var<storage, read> rms_input: array<f32>;
@group(0) @binding(1) var<storage, read> rms_weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> rms_out: array<f32>;
@group(0) @binding(3) var<uniform> rms_params: vec2<u32>;

@compute @workgroup_size(64)
fn rms_norm(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let global_id = vec3<u32>(gid.x + gid.y * nwg.x * 64u, 0u, 0u);
    let row = global_id.x;
    let size = rms_params.x;
    let rows = rms_params.y;
    if (row >= rows) { return; }
    let row_offset = row * size;
    var ss = 0.0;
    for (var i = 0u; i < size; i = i + 1u) {
        let val = rms_input[row_offset + i];
        ss = ss + val * val;
    }
    ss = ss / f32(size);
    let inv_rms = 1.0 / sqrt(ss + 1e-5);
    for (var i = 0u; i < size; i = i + 1u) {
        rms_out[row_offset + i] = rms_input[row_offset + i] * inv_rms * rms_weight[i];
    }
}

// --- 4. SiLU ---
@group(0) @binding(0) var<storage, read> silu_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> silu_out: array<f32>;
@compute @workgroup_size(64)
fn silu(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let global_id = vec3<u32>(gid.x + gid.y * nwg.x * 64u, 0u, 0u);
    let i = global_id.x;
    if (i >= arrayLength(&silu_out)) { return; }
    let x = silu_input[i];
    silu_out[i] = x * (1.0 / (1.0 + exp(-x)));
}

// --- 5. Softmax ---
@group(0) @binding(0) var<storage, read> sm_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> sm_out: array<f32>;
@group(0) @binding(2) var<uniform> sm_params: vec2<u32>;

@compute @workgroup_size(1)
fn softmax(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let global_id = vec3<u32>(gid.x + gid.y * nwg.x * 1u, 0u, 0u);
    let row = global_id.x;
    let len = sm_params.x;
    let rows = sm_params.y;
    if (row >= rows) { return; }
    let row_offset = row * len;
    var max_val = -1e9;
    for (var i = 0u; i < len; i = i + 1u) { max_val = max(max_val, sm_input[row_offset + i]); }
    var sum_exp = 0.0;
    for (var i = 0u; i < len; i = i + 1u) {
        let val = exp(sm_input[row_offset + i] - max_val);
        sm_out[row_offset + i] = val;
        sum_exp = sum_exp + val;
    }
    for (var i = 0u; i < len; i = i + 1u) { sm_out[row_offset + i] = sm_out[row_offset + i] / sum_exp; }
}

// --- 6. Add & Mul ---
@group(0) @binding(0) var<storage, read> bin_a: array<f32>;
@group(0) @binding(1) var<storage, read> bin_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> bin_out: array<f32>;

@compute @workgroup_size(64)
fn add(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let global_id = vec3<u32>(gid.x + gid.y * nwg.x * 64u, 0u, 0u);
    let i = global_id.x;
    if (i >= arrayLength(&bin_out)) { return; }
    bin_out[i] = bin_a[i] + bin_b[i];
}

@compute @workgroup_size(64)
fn mul(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let global_id = vec3<u32>(gid.x + gid.y * nwg.x * 64u, 0u, 0u);
    let i = global_id.x;
    if (i >= arrayLength(&bin_out)) { return; }
    bin_out[i] = bin_a[i] * bin_b[i];
}

// --- 7. Dequantize ---
@group(0) @binding(0) var<storage, read> deq_in: array<u32>;
@group(0) @binding(1) var<storage, read_write> deq_out: array<f32>;
@compute @workgroup_size(64)
fn dequantize(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let global_id = vec3<u32>(gid.x + gid.y * nwg.x * 64u, 0u, 0u);
    let i = global_id.x;
    let base = i * 4u;
    if (i >= arrayLength(&deq_in)) { return; }
    let p = deq_in[i];
    if (base < arrayLength(&deq_out)) { deq_out[base] = f32(p & 0xFFu); }
    if (base+1u < arrayLength(&deq_out)) { deq_out[base+1u] = f32((p >> 8u) & 0xFFu); }
    if (base+2u < arrayLength(&deq_out)) { deq_out[base+2u] = f32((p >> 16u) & 0xFFu); }
    if (base+3u < arrayLength(&deq_out)) { deq_out[base+3u] = f32((p >> 24u) & 0xFFu); }
}

// --- 8. Top-K Sampling ---
@group(0) @binding(0) var<storage, read> tk_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> tk_out_idx: array<u32>;
@group(0) @binding(2) var<uniform> tk_params: u32;

@compute @workgroup_size(1)
fn top_k(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let k = tk_params;
    let len = arrayLength(&tk_input);
    for (var i = 0u; i < k; i = i + 1u) {
        var max_v = -1e9;
        var max_idx = 0u;
        for (var j = 0u; j < len; j = j + 1u) {
            let v = tk_input[j];
            var used = false;
            for (var p = 0u; p < i; p = p + 1u) { if (tk_out_idx[p] == j) { used = true; break; } }
            if (!used && v > max_v) { max_v = v; max_idx = j; }
        }
        tk_out_idx[i] = max_idx;
    }
}

// --- 9. Backward Kernels (Clean: No Accumulation) ---
@group(0) @binding(3) var<storage, read> add_grad_out: array<f32>;
@group(0) @binding(4) var<storage, read_write> add_grad_a: array<f32>;
@group(0) @binding(5) var<storage, read_write> add_grad_b: array<f32>;
@compute @workgroup_size(64)
fn add_backward(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let global_id = vec3<u32>(gid.x + gid.y * nwg.x * 64u, 0u, 0u);
    let i = global_id.x;
    if (i >= arrayLength(&add_grad_out)) { return; }
    let g = add_grad_out[i];
    add_grad_a[i] = g;
    add_grad_b[i] = g;
}

@group(0) @binding(3) var<storage, read> mul_grad_out: array<f32>;
@group(0) @binding(4) var<storage, read> mul_in_a: array<f32>;
@group(0) @binding(5) var<storage, read> mul_in_b: array<f32>;
@group(0) @binding(6) var<storage, read_write> mul_grad_a: array<f32>;
@group(0) @binding(7) var<storage, read_write> mul_grad_b: array<f32>;
@compute @workgroup_size(64)
fn mul_backward(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let global_id = vec3<u32>(gid.x + gid.y * nwg.x * 64u, 0u, 0u);
    let i = global_id.x;
    if (i >= arrayLength(&mul_grad_out)) { return; }
    let g = mul_grad_out[i];
    mul_grad_a[i] = g * mul_in_b[i];
    mul_grad_b[i] = g * mul_in_a[i];
}

@group(0) @binding(3) var<storage, read> silu_bk_grad_out: array<f32>;
@group(0) @binding(4) var<storage, read> silu_bk_input: array<f32>;
@group(0) @binding(5) var<storage, read_write> silu_bk_grad_in: array<f32>;
@compute @workgroup_size(64)
fn silu_backward_real(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let global_id = vec3<u32>(gid.x + gid.y * nwg.x * 64u, 0u, 0u);
    let i = global_id.x;
    if (i >= arrayLength(&silu_bk_grad_out)) { return; }
    let x = silu_bk_input[i];
    let s = 1.0 / (1.0 + exp(-x));
    silu_bk_grad_in[i] = silu_bk_grad_out[i] * s * (1.0 + x * (1.0 - s));
}

@group(0) @binding(0) var<storage, read> mm_bk_grad: array<f32>;
@group(0) @binding(1) var<storage, read> mm_bk_other: array<f32>; 
@group(0) @binding(2) var<storage, read_write> mm_bk_out: array<f32>;
@group(0) @binding(3) var<uniform> mm_bk_params: vec4<u32>; 

@compute @workgroup_size(16, 16, 1)
fn matmul_bt(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y; let col = global_id.x;
    let m = mm_bk_params.x; let k = mm_bk_params.y; let n = mm_bk_params.z;
    if (row >= m || col >= n) { return; }
    var acc = 0.0;
    for (var i = 0u; i < k; i = i + 1u) { acc = acc + mm_bk_grad[row * k + i] * mm_bk_other[col * k + i]; }
    mm_bk_out[row * n + col] = acc;
}

@compute @workgroup_size(16, 16, 1)
fn matmul_at(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y; let col = global_id.x;
    let m = mm_bk_params.x; let k = mm_bk_params.y; let n = mm_bk_params.z;
    if (row >= m || col >= n) { return; }
    var acc = 0.0;
    for (var i = 0u; i < k; i = i + 1u) { acc = acc + mm_bk_grad[i * m + row] * mm_bk_other[i * n + col]; }
    mm_bk_out[row * n + col] = acc;
}

// --- 10. Optimizer Kernels ---
@group(0) @binding(0) var<storage, read_write> opt_weight: array<f32>;
@group(0) @binding(1) var<storage, read> opt_grad: array<f32>;
@group(0) @binding(2) var<uniform> opt_lr: f32;

@compute @workgroup_size(64)
fn sgd_update(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let global_id = vec3<u32>(gid.x + gid.y * nwg.x * 64u, 0u, 0u);
    let i = global_id.x;
    if (i >= arrayLength(&opt_weight)) { return; }
    opt_weight[i] = opt_weight[i] - (opt_lr * opt_grad[i]);
}

@group(0) @binding(3) var<storage, read_write> adam_m: array<f32>;
@group(0) @binding(4) var<storage, read_write> adam_v: array<f32>;
struct AdamParams {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    t: f32,
};
@group(0) @binding(5) var<uniform> adam_params: AdamParams;

@compute @workgroup_size(64)
fn adamw_update(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let global_id = vec3<u32>(gid.x + gid.y * nwg.x * 64u, 0u, 0u);
    let i = global_id.x;
    if (i >= arrayLength(&opt_weight)) { return; }
    
    let g = opt_grad[i];
    let m = adam_m[i] * adam_params.beta1 + (1.0 - adam_params.beta1) * g;
    let v = adam_v[i] * adam_params.beta2 + (1.0 - adam_params.beta2) * g * g;
    
    adam_m[i] = m;
    adam_v[i] = v;
    
    let m_hat = m / (1.0 - pow(adam_params.beta1, adam_params.t));
    let v_hat = v / (1.0 - pow(adam_params.beta2, adam_params.t));
    
    // AdamW: weight decay applied directly to weight
    var w = opt_weight[i];
    w = w - adam_params.lr * (m_hat / (sqrt(v_hat) + adam_params.eps) + adam_params.weight_decay * w);
    opt_weight[i] = w;
}

@compute @workgroup_size(64)
fn sub(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let global_id = vec3<u32>(gid.x + gid.y * nwg.x * 64u, 0u, 0u);
    let i = global_id.x;
    if (i >= arrayLength(&bin_out)) { return; }
    bin_out[i] = bin_a[i] - bin_b[i];
}

@group(0) @binding(3) var<storage, read> sub_grad_out: array<f32>;
@group(0) @binding(4) var<storage, read_write> sub_grad_a: array<f32>;
@group(0) @binding(5) var<storage, read_write> sub_grad_b: array<f32>;
@compute @workgroup_size(64)
fn sub_backward(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let global_id = vec3<u32>(gid.x + gid.y * nwg.x * 64u, 0u, 0u);
    let i = global_id.x;
    if (i >= arrayLength(&sub_grad_out)) { return; }
    let g = sub_grad_out[i];
    sub_grad_a[i] = g;
    sub_grad_b[i] = -g;
}

// --- 11. Loss Kernels ---
@group(0) @binding(0) var<storage, read> mse_pred: array<f32>;
@group(0) @binding(1) var<storage, read> mse_target: array<f32>;
@group(0) @binding(2) var<storage, read_write> mse_out: array<f32>;
@compute @workgroup_size(64)
fn mse_loss(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let global_id = vec3<u32>(gid.x + gid.y * nwg.x * 64u, 0u, 0u);
    let i = global_id.x;
    if (i >= arrayLength(&mse_out)) { return; }
    let diff = mse_pred[i] - mse_target[i];
    mse_out[i] = 0.5 * diff * diff;
}

@group(0) @binding(3) var<storage, read_write> mse_grad_in: array<f32>;
@compute @workgroup_size(64)
fn mse_loss_backward(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let global_id = vec3<u32>(gid.x + gid.y * nwg.x * 64u, 0u, 0u);
    let i = global_id.x;
    if (i >= arrayLength(&mse_pred)) { return; }
    mse_grad_in[i] = mse_pred[i] - mse_target[i];
}

// --- 12. Relu ---
@group(0) @binding(0) var<storage, read> relu_input_f: array<f32>;
@group(0) @binding(1) var<storage, read_write> relu_out_f: array<f32>;
@compute @workgroup_size(64)
fn relu(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let global_id = vec3<u32>(gid.x + gid.y * nwg.x * 64u, 0u, 0u);
    let i = global_id.x;
    if (i >= arrayLength(&relu_out_f)) { return; }
    relu_out_f[i] = max(0.0, relu_input_f[i]);
}

@group(0) @binding(3) var<storage, read> relu_grad_out: array<f32>;
@group(0) @binding(4) var<storage, read> relu_in: array<f32>;
@group(0) @binding(5) var<storage, read_write> relu_grad_in: array<f32>;
@compute @workgroup_size(64)
fn relu_backward(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let global_id = vec3<u32>(gid.x + gid.y * nwg.x * 64u, 0u, 0u);
    let i = global_id.x;
    if (i >= arrayLength(&relu_grad_out)) { return; }
    if (relu_in[i] > 0.0) {
        relu_grad_in[i] = relu_grad_out[i];
    } else {
        relu_grad_in[i] = 0.0;
    }
}

@group(0) @binding(3) var<storage, read> sm_grad_out: array<f32>;
@group(0) @binding(4) var<storage, read> sm_final_out: array<f32>;
@group(0) @binding(5) var<storage, read_write> sm_grad_in: array<f32>;
// sm_size is at binding 2
@compute @workgroup_size(1)
fn softmax_backward(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let global_id = vec3<u32>(gid.x + gid.y * nwg.x * 1u, 0u, 0u);
    let row = global_id.x;
    let len = sm_params.x;
    let rows = sm_params.y;
    if (row >= rows) { return; }
    let row_offset = row * len;
    var sum_g_s = 0.0;
    for (var i = 0u; i < len; i = i + 1u) {
        sum_g_s = sum_g_s + sm_grad_out[row_offset + i] * sm_final_out[row_offset + i];
    }
    for (var i = 0u; i < len; i = i + 1u) {
        let s = sm_final_out[row_offset + i];
        sm_grad_in[row_offset + i] = s * (sm_grad_out[row_offset + i] - sum_g_s);
    }
}

// --- 13. Affine Dequantize (8-bit) ---
@group(0) @binding(0) var<storage, read> q_weight: array<u32>;
@group(0) @binding(1) var<storage, read> q_scales: array<f32>;
@group(0) @binding(2) var<storage, read> q_biases: array<f32>;
@group(0) @binding(3) var<storage, read_write> q_out: array<f32>;

@compute @workgroup_size(64)
fn affine_dequantize(@builtin(global_invocation_id) global_id: vec3<u32>,
                     @builtin(num_workgroups) num_wg: vec3<u32>) {
    let out_idx = global_id.y * num_wg.x * 64u + global_id.x;
    if (out_idx >= arrayLength(&q_out)) { return; }
    
    let u32_idx = out_idx / 4u;
    let byte_inner = out_idx % 4u;
    let val_u32 = q_weight[u32_idx];
    let val_u8 = f32((val_u32 >> (byte_inner * 8u)) & 0xFFu);
    
    let group_idx = out_idx / 64u; // MLX default group_size=64
    let scale = q_scales[group_idx];
    let bias = q_biases[group_idx];
    
    q_out[out_idx] = val_u8 * scale + bias;
}

// --- 14. Conv1D (Width 3) ---
@group(0) @binding(0) var<storage, read> conv_in: array<f32>;
@group(0) @binding(1) var<storage, read> conv_weight: array<f32>; // [dim, kernel_size]
@group(0) @binding(2) var<storage, read_write> conv_out: array<f32>;
@group(0) @binding(3) var<storage, read> conv_bias_v: array<f32>;

struct ConvParams {
    dim: u32,
    seq_len: u32,
};
@group(0) @binding(4) var<uniform> conv_p: ConvParams;

@compute @workgroup_size(64)
fn conv1d(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let global_id = vec3<u32>(gid.x + gid.y * nwg.x * 64u, 0u, 0u);
    let idx = global_id.x;
    let total = conv_p.dim * conv_p.seq_len;
    if (idx >= total) { return; }
    
    let d = idx / conv_p.seq_len;
    let s = idx % conv_p.seq_len;
    
    var res = conv_bias_v[d];
    
    // Depthwise conv1d (kernel_size=3)
    for (var k = 0u; k < 3u; k = k + 1u) {
        let i = s + k;
        if (i >= 1u && i - 1u < conv_p.seq_len) {
            let in_val = conv_in[d * conv_p.seq_len + (i - 1u)];
            res = res + in_val * conv_weight[d * 3u + k];
        }
    }
    
    conv_out[idx] = res;
}

// --- 15. Cross-Entropy Loss (for LLM Training) ---
// Forward: CE(logits, target) = -log(softmax(logits)[target])
// Inputs: logits[vocab_size], target (single u32 index)
// Output: scalar loss
@group(0) @binding(0) var<storage, read> ce_logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> ce_loss_out: array<f32>; // single element
@group(0) @binding(2) var<uniform> ce_params: vec4<u32>; // [target_idx, vocab_size, 0, 0]

@compute @workgroup_size(1)
fn cross_entropy_loss(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let target_idx = ce_params.x;
    let vocab_size = ce_params.y;
    
    // Stable softmax: subtract max
    var max_val = -1e9;
    for (var i = 0u; i < vocab_size; i = i + 1u) {
        max_val = max(max_val, ce_logits[i]);
    }
    
    var sum_exp = 0.0;
    for (var i = 0u; i < vocab_size; i = i + 1u) {
        sum_exp = sum_exp + exp(ce_logits[i] - max_val);
    }
    
    // log_softmax[target_idx] = logits[target_idx] - max - log(sum_exp)
    let log_softmax_target = ce_logits[target_idx] - max_val - log(sum_exp);
    
    // Loss = -log_softmax[target_idx]
    ce_loss_out[0] = -log_softmax_target;
}

// Backward: gradient = softmax(logits) - one_hot(target)
// This is the elegant combined gradient for softmax + cross-entropy
@group(0) @binding(0) var<storage, read> ce_bw_logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> ce_bw_grad: array<f32>;
@group(0) @binding(2) var<uniform> ce_bw_params: vec4<u32>; // [target_idx, vocab_size, 0, 0]

@compute @workgroup_size(64)
fn cross_entropy_backward(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let global_id = vec3<u32>(gid.x + gid.y * nwg.x * 64u, 0u, 0u);
    let idx = global_id.x;
    let target_idx = ce_bw_params.x;
    let vocab_size = ce_bw_params.y;
    
    if (idx >= vocab_size) { return; }
    
    // First compute softmax (need max for stability)
    var max_val = -1e9;
    for (var i = 0u; i < vocab_size; i = i + 1u) {
        max_val = max(max_val, ce_bw_logits[i]);
    }
    
    var sum_exp = 0.0;
    for (var i = 0u; i < vocab_size; i = i + 1u) {
        sum_exp = sum_exp + exp(ce_bw_logits[i] - max_val);
    }
    
    let softmax_val = exp(ce_bw_logits[idx] - max_val) / sum_exp;
    
    // gradient = softmax - one_hot(target_idx)
    if (idx == target_idx) {
        ce_bw_grad[idx] = softmax_val - 1.0;
    } else {
        ce_bw_grad[idx] = softmax_val;
    }
}

// --- 16. Gradient Norm (for gradient clipping) ---
// Computes sum of squares of all gradients (partial reduction)
@group(0) @binding(0) var<storage, read> gnorm_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> gnorm_out: array<f32>;

@compute @workgroup_size(64)
fn gradient_norm_squared(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let global_id = vec3<u32>(gid.x + gid.y * nwg.x * 64u, 0u, 0u);
    let idx = global_id.x;
    if (idx >= arrayLength(&gnorm_in)) { return; }
    let val = gnorm_in[idx];
    gnorm_out[idx] = val * val;
}

// --- 17. Gradient Clipping (scale if norm > max_norm) ---
@group(0) @binding(0) var<storage, read_write> gclip_grad: array<f32>;
@group(0) @binding(1) var<uniform> gclip_scale: f32; // min(1.0, max_norm / actual_norm)

@compute @workgroup_size(64)
fn clip_grad_by_norm(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let global_id = vec3<u32>(gid.x + gid.y * nwg.x * 64u, 0u, 0u);
    let idx = global_id.x;
    if (idx >= arrayLength(&gclip_grad)) { return; }
    gclip_grad[idx] = gclip_grad[idx] * gclip_scale;
}

// --- 18. Fused SiLU-Gate (for Llama MLP) ---
// out = SiLU(up) * gate = (up * sigmoid(up)) * gate
// Reduces memory bandwidth by 2x compared to separate ops
@group(0) @binding(0) var<storage, read> fused_up: array<f32>;
@group(0) @binding(1) var<storage, read> fused_gate: array<f32>;
@group(0) @binding(2) var<storage, read_write> fused_silu_gate_out: array<f32>;

@compute @workgroup_size(64)
fn silu_gate_fused(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let global_id = vec3<u32>(gid.x + gid.y * nwg.x * 64u, 0u, 0u);
    let i = global_id.x;
    if (i >= arrayLength(&fused_silu_gate_out)) { return; }
    let up = fused_up[i];
    let gate = fused_gate[i];
    let silu_up = up * (1.0 / (1.0 + exp(-up)));
    fused_silu_gate_out[i] = silu_up * gate;
}

// Backward for fused SiLU-Gate
@group(0) @binding(3) var<storage, read> fused_grad_out: array<f32>;
@group(0) @binding(4) var<storage, read_write> fused_grad_up: array<f32>;
@group(0) @binding(5) var<storage, read_write> fused_grad_gate: array<f32>;

@compute @workgroup_size(64)
fn silu_gate_backward(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let global_id = vec3<u32>(gid.x + gid.y * nwg.x * 64u, 0u, 0u);
    let i = global_id.x;
    if (i >= arrayLength(&fused_grad_out)) { return; }
    
    let up = fused_up[i];
    let gate = fused_gate[i];
    let g = fused_grad_out[i];
    
    let s = 1.0 / (1.0 + exp(-up));
    let silu_up = up * s;
    
    // d(silu_up * gate)/d_gate = silu_up
    fused_grad_gate[i] = g * silu_up;
    
    // d(silu_up * gate)/d_up = gate * d(silu)/d_up
    // d(silu)/d_up = s * (1 + up*(1-s))
    let dsilu = s * (1.0 + up * (1.0 - s));
    fused_grad_up[i] = g * gate * dsilu;
}

// --- 19. Fused RMSNorm + Residual Add ---
// out = RMSNorm(x + residual) * weight
// Common in Llama attention/MLP blocks
@group(0) @binding(0) var<storage, read> fused_rms_x: array<f32>;
@group(0) @binding(1) var<storage, read> fused_rms_residual: array<f32>;
@group(0) @binding(2) var<storage, read> fused_rms_weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> fused_rms_out: array<f32>;
@group(0) @binding(4) var<uniform> fused_rms_size: u32;

@compute @workgroup_size(1)
fn rms_norm_residual(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let size = fused_rms_size;
    
    // First add residual and compute RMS
    var ss = 0.0;
    for (var i = 0u; i < size; i = i + 1u) {
        let val = fused_rms_x[i] + fused_rms_residual[i];
        fused_rms_out[i] = val; // temp store sum
        ss = ss + val * val;
    }
    
    ss = ss / f32(size);
    let inv_rms = 1.0 / sqrt(ss + 1e-5);
    
    for (var i = 0u; i < size; i = i + 1u) {
        fused_rms_out[i] = fused_rms_out[i] * inv_rms * fused_rms_weight[i];
    }
}

// --- 20. Gradient Accumulation ---
// Accumulates gradients in-place: grad += new_grad
// Essential for micro-batching on memory-constrained devices
@group(0) @binding(0) var<storage, read_write> accum_grad: array<f32>;
@group(0) @binding(1) var<storage, read> accum_new_grad: array<f32>;

@compute @workgroup_size(64)
fn gradient_accumulate(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let global_id = vec3<u32>(gid.x + gid.y * nwg.x * 64u, 0u, 0u);
    let i = global_id.x;
    if (i >= arrayLength(&accum_grad)) { return; }
    accum_grad[i] = accum_grad[i] + accum_new_grad[i];
}

// --- 21. Scale Gradients (for gradient accumulation averaging) ---
@group(0) @binding(0) var<storage, read_write> scale_grad: array<f32>;
@group(0) @binding(1) var<uniform> scale_factor: f32;

@compute @workgroup_size(64)
fn gradient_scale(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let global_id = vec3<u32>(gid.x + gid.y * nwg.x * 64u, 0u, 0u);
    let i = global_id.x;
    if (i >= arrayLength(&scale_grad)) { return; }
    scale_grad[i] = scale_grad[i] * scale_factor;
}
// ============================================================================
// MIXED PRECISION (FP16) KERNELS
// ============================================================================

// --- 22. FP32 to FP16 Cast ---
@group(0) @binding(0) var<storage, read> cast_fp32_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> cast_fp16_out: array<u32>;

fn f32_to_f16_bits(val: f32) -> u32 {
    let bits = bitcast<u32>(val);
    let sign = (bits >> 16u) & 0x8000u;
    let exp = (bits >> 23u) & 0xFFu;
    let mant = bits & 0x7FFFFFu;
    
    if (exp == 0u) { return sign; }
    if (exp == 255u) {
        if (mant != 0u) { return sign | 0x7E00u; }
        return sign | 0x7C00u;
    }
    
    let new_exp = i32(exp) - 127 + 15;
    if (new_exp >= 31) { return sign | 0x7C00u; }
    if (new_exp <= 0) { return sign; }
    
    return sign | (u32(new_exp) << 10u) | (mant >> 13u);
}

@compute @workgroup_size(64)
fn fp32_to_fp16(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let global_id = vec3<u32>(gid.x + gid.y * nwg.x * 64u, 0u, 0u);
    let i = global_id.x;
    if (i >= arrayLength(&cast_fp16_out)) { return; }
    
    let pair_idx = i * 2u;
    let in_len = arrayLength(&cast_fp32_in);
    var lo: u32 = 0u;
    var hi: u32 = 0u;
    
    if (pair_idx < in_len) { lo = f32_to_f16_bits(cast_fp32_in[pair_idx]); }
    if (pair_idx + 1u < in_len) { hi = f32_to_f16_bits(cast_fp32_in[pair_idx + 1u]); }
    
    cast_fp16_out[i] = lo | (hi << 16u);
}

// --- 23. FP16 to FP32 Cast ---
@group(0) @binding(0) var<storage, read> cast_fp16_in: array<u32>;
@group(0) @binding(1) var<storage, read_write> cast_fp32_out: array<f32>;

fn f16_bits_to_f32(bits: u32) -> f32 {
    let sign = (bits & 0x8000u) << 16u;
    let exp = (bits >> 10u) & 0x1Fu;
    let mant = bits & 0x3FFu;
    
    if (exp == 0u) { return bitcast<f32>(sign); }
    if (exp == 31u) {
        if (mant != 0u) { return bitcast<f32>(sign | 0x7FC00000u); }
        return bitcast<f32>(sign | 0x7F800000u);
    }
    
    return bitcast<f32>(sign | ((exp + 127u - 15u) << 23u) | (mant << 13u));
}

@compute @workgroup_size(64)
fn fp16_to_fp32(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let global_id = vec3<u32>(gid.x + gid.y * nwg.x * 64u, 0u, 0u);
    let i = global_id.x;
    if (i >= arrayLength(&cast_fp32_out)) { return; }
    
    let pair_idx = i / 2u;
    let is_hi = (i % 2u) == 1u;
    let packed = cast_fp16_in[pair_idx];
    var bits: u32;
    if (is_hi) { bits = (packed >> 16u) & 0xFFFFu; }
    else { bits = packed & 0xFFFFu; }
    
    cast_fp32_out[i] = f16_bits_to_f32(bits);
}
