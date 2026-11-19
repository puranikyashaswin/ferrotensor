// ============================================================================
// SMALL-M MATMUL (weight-stationary, split-K)
// ============================================================================
// out[M, N] = a[M, K] @ op(b) for a product with few rows (a micro-batch of
// up to a few hundred tokens) against a wide and deep weight b, stored as
// either
//   a@bT: b is [N, K] (a linear weight in Hugging Face layout, forward), or
//   a@b:  b is [K, N] (the same weight, input gradient).
// b is f32, or f16 pairs packed into u32 words.
//
// Workgroup (x, y, z) covers rows x * BM.., columns y * BN.. and the z-th
// slice of K (params.w elements of K each). One row block holds every row of
// a pass of up to 128 rows, so each weight element is read once per pass;
// the K split gives the grid enough workgroups to fill the GPU. With more
// than one slice each workgroup writes its partial sums to slice z of an
// [S, M, N] buffer, which matmul_split_reduce adds in a fixed order.
//
// Each thread keeps TM x TN outputs in registers: rows ty + i * TY, and
// columns tx + j * TX (a@bT) or (tx + j * TX) * 4 + 0..3 (a@b). K advances
// BK at a time through two workgroup-memory buffers: while the threads read
// one, the next slice is loaded into registers and stored to the other, so
// one barrier per slice is enough. a and an [N, K] weight are staged as rows
// of BK / 4 vec4s plus one vec4 of padding, which puts the vec4 reads of
// neighbouring rows in different banks; a [K, N] weight is staged as rows of
// BN / 4 vec4s. Every global read takes 4 elements along the operand's
// contiguous dimension, so neighbouring threads read neighbouring memory.
// Rows of a (and of an [N, K] b) past the end repeat the last row; their
// sums are never stored.
//
// engine/small_m.rs fills in the tile sizes and the unrolled parts per tile;
// this file alone is not valid WGSL.

const SKM_BM: u32 = %BM%u;
const SKM_BN: u32 = %BN%u;
const SKM_BK: u32 = %BK%u;
const SKM_TX: u32 = %TX%u;
// vec4s per K slice of a staged row, and per staged row with its padding.
const SKM_K4: u32 = %K4%u;
const SKM_KS: u32 = %KS%u;
// vec4s per staged row of a [K, N] weight.
const SKM_N4: u32 = %N4%u;
// vec4s in one of the two buffers of a and of b.
const SKM_A_HALF: u32 = %A_HALF%u;
const SKM_B_HALF: u32 = %B_HALF%u;

@group(0) @binding(0) var<storage, read> skm_a: array<f32>;
@group(0) @binding(1) var<storage, read> skm_b: array<%B_ELEM%>;
@group(0) @binding(2) var<storage, read_write> skm_out: array<f32>;
@group(0) @binding(3) var<uniform> skm_params: vec4<u32>; // M, K, N, K per slice

var<workgroup> skm_as: array<vec4<f32>, %A_LEN%>;
var<workgroup> skm_bs: array<vec4<f32>, %B_LEN%>;

// How many of the 4 elements from `at` lie before `len`.
fn skm_valid(len: u32, at: u32) -> u32 {
    if (at >= len) { return 0u; }
    return min(4u, len - at);
}

// Elements i..i+3 of a; those from `valid` on are 0.
fn skm_a4(i: u32, valid: u32) -> vec4<f32> {
    if (valid >= 4u) {
        return vec4<f32>(skm_a[i], skm_a[i + 1u], skm_a[i + 2u], skm_a[i + 3u]);
    }
    var v = vec4<f32>(0.0);
    if (valid > 0u) { v.x = skm_a[i]; }
    if (valid > 1u) { v.y = skm_a[i + 1u]; }
    if (valid > 2u) { v.z = skm_a[i + 2u]; }
    return v;
}

%B4%

@compute @workgroup_size(%THREADS%, 1, 1)
fn matmul_small_m(
    @builtin(local_invocation_index) lidx: u32,
    @builtin(workgroup_id) wg: vec3<u32>,
) {
    let m = skm_params.x;
    let k = skm_params.y;
    let n = skm_params.z;
    let k_begin = wg.z * skm_params.w;
    let k_end = min(k, k_begin + skm_params.w);
    let tiles = (k_end - k_begin + SKM_BK - 1u) / SKM_BK;
    let row0 = wg.x * SKM_BM;
    let col0 = wg.y * SKM_BN;
    let tx = lidx % SKM_TX;
    let ty = lidx / SKM_TX;
    let a_row = ty * SKM_KS;
    let b_col = tx * SKM_KS;

%ACC%
%STAGE%
    {
        let kb = k_begin;
%LOAD%
    }
    {
        let a_nxt = 0u;
        let b_nxt = 0u;
%STORE%
    }
    workgroupBarrier();

    for (var t = 0u; t < tiles; t++) {
        let a_cur = (t & 1u) * SKM_A_HALF;
        let b_cur = (t & 1u) * SKM_B_HALF;
        let more = t + 1u < tiles;
        if (more) {
            let kb = k_begin + (t + 1u) * SKM_BK;
%LOAD%
        }
        for (var k4 = 0u; k4 < SKM_K4; k4++) {
%COMPUTE%
        }
        if (more) {
            let a_nxt = SKM_A_HALF - a_cur;
            let b_nxt = SKM_B_HALF - b_cur;
%STORE%
        }
        workgroupBarrier();
    }

    let base = wg.z * m * n;
%WRITE%
}
