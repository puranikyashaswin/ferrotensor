// ============================================================================
// REGISTER-BLOCKED MATMUL
// ============================================================================
// out[M, N] = op(a) @ op(b) for row-major f32 matrices, where mmb_params.w
// selects the operand layout:
//   0: a is [M, K], b is [K, N]
//   1: a is [M, K], b is [N, K]   (a @ b^T)
//   2: a is [K, M], b is [K, N]   (a^T @ b)
//
// Each 16x16 workgroup computes a 64x64 tile of `out`; each thread owns
// 4x4 outputs held in vec4 registers. K is consumed in slices of MMB_BK,
// staged through workgroup memory as As[k][m] and Bs[k][n] and read back as
// vec4s.
//
// When b is stored along k (mode 1, b[N, K]), the staging gives neighbouring
// threads neighbouring k of the same rows, so each warp-wide load reads
// contiguous memory; it then writes the slice transposed into Bs[k][n], so
// the inner loop is the same in every mode.

const MMB_BM: u32 = 64u;
const MMB_BN: u32 = 64u;
const MMB_BK: u32 = 16u;
// Row length of mmb_bs4 in vec4s, one more than MMB_BN / 4: the k-inner
// staging writes 16 rows at once, and the pad spreads those stores over the
// shared-memory banks instead of stacking them on one.
const MMB_BS_ROW: u32 = 17u;

@group(0) @binding(0) var<storage, read> mmb_a: array<f32>;
@group(0) @binding(1) var<storage, read> mmb_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> mmb_out: array<f32>;
@group(0) @binding(3) var<uniform> mmb_params: vec4<u32>; // M, K, N, mode

var<workgroup> mmb_as4: array<vec4<f32>, 256>; // [MMB_BK][MMB_BM / 4]
var<workgroup> mmb_bs4: array<vec4<f32>, 272>; // [MMB_BK][MMB_BS_ROW], last vec4 unused

fn mmb_load_a(gm: u32, gk: u32, m: u32, k: u32, mode: u32) -> f32 {
    if (gm >= m || gk >= k) { return 0.0; }
    if (mode == 2u) { return mmb_a[gk * m + gm]; }
    return mmb_a[gm * k + gk];
}

fn mmb_load_b(gk: u32, gn: u32, k: u32, n: u32, mode: u32) -> f32 {
    if (gn >= n || gk >= k) { return 0.0; }
    if (mode == 1u) { return mmb_b[gn * k + gk]; }
    return mmb_b[gk * n + gn];
}

// Whether consecutive k of one n are adjacent in mmb_b's storage.
fn mmb_b_k_inner(mode: u32) -> bool { return mode == 1u; }

// Whether the k-inner staging takes k in pairs. Off here, where a pair costs
// two loads anyway; the f16 kernel turns it on, since one word holds both.
const MMB_K_PAIRS: bool = false;

// b[gk][gn] and b[gk + 1][gn] (0 past k) for the paired k-inner staging.
fn mmb_load_b_pair(gk: u32, gn: u32, k: u32, n: u32, mode: u32) -> vec2<f32> {
    return vec2<f32>(mmb_load_b(gk, gn, k, n, mode), mmb_load_b(gk + 1u, gn, k, n, mode));
}

@compute @workgroup_size(16, 16, 1)
fn matmul_blocked(
    @builtin(local_invocation_index) lidx: u32,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
) {
    let m = mmb_params.x;
    let k = mmb_params.y;
    let n = mmb_params.z;
    let mode = mmb_params.w;
    let b_k_inner = mmb_b_k_inner(mode);
    let row0 = wg.y * MMB_BM;
    let col0 = wg.x * MMB_BN;

    var acc0_0 = vec4<f32>(0.0);
    var acc1_0 = vec4<f32>(0.0);
    var acc2_0 = vec4<f32>(0.0);
    var acc3_0 = vec4<f32>(0.0);

    for (var k0 = 0u; k0 < k; k0 += MMB_BK) {
        // Stage the A and B slices; each thread fills whole vec4s (4
        // consecutive m or n for one k).
        for (var idx = lidx; idx < MMB_BK * MMB_BM / 4u; idx += 256u) {
            let kk = idx / (MMB_BM / 4u);
            let m4 = (idx % (MMB_BM / 4u)) * 4u;
            let gk = k0 + kk;
            let gm = row0 + m4;
            mmb_as4[idx] = vec4<f32>(
                mmb_load_a(gm, gk, m, k, mode),
                mmb_load_a(gm + 1u, gk, m, k, mode),
                mmb_load_a(gm + 2u, gk, m, k, mode),
                mmb_load_a(gm + 3u, gk, m, k, mode),
            );
        }
        if (b_k_inner && MMB_K_PAIRS) {
            // Half the threads each fill the vec4s of four columns at k and
            // k + 1, so the f16 kernel takes each pair from one word.
            // Threads 4 to 7 of every 8 write the k + 1 row first, which
            // puts each 8 threads' stores in 8 different bank groups.
            for (var idx = lidx; idx < MMB_BK * MMB_BN / 8u; idx += 256u) {
                let kk = (idx % (MMB_BK / 2u)) * 2u;
                let n4 = (idx / (MMB_BK / 2u)) * 4u;
                let gk = k0 + kk;
                let gn = col0 + n4;
                let p0 = mmb_load_b_pair(gk, gn, k, n, mode);
                let p1 = mmb_load_b_pair(gk, gn + 1u, k, n, mode);
                let p2 = mmb_load_b_pair(gk, gn + 2u, k, n, mode);
                let p3 = mmb_load_b_pair(gk, gn + 3u, k, n, mode);
                let lo = vec4<f32>(p0.x, p1.x, p2.x, p3.x);
                let hi = vec4<f32>(p0.y, p1.y, p2.y, p3.y);
                let flip = (idx & 4u) != 0u;
                let c = n4 / 4u;
                mmb_bs4[(kk + select(0u, 1u, flip)) * MMB_BS_ROW + c] = select(lo, hi, flip);
                mmb_bs4[(kk + select(1u, 0u, flip)) * MMB_BS_ROW + c] = select(hi, lo, flip);
            }
        } else {
            for (var idx = lidx; idx < MMB_BK * MMB_BN / 4u; idx += 256u) {
                var kk = idx / (MMB_BN / 4u);
                var n4 = (idx % (MMB_BN / 4u)) * 4u;
                if (b_k_inner) {
                    kk = idx % MMB_BK;
                    n4 = (idx / MMB_BK) * 4u;
                }
                let gk = k0 + kk;
                let gn = col0 + n4;
                mmb_bs4[kk * MMB_BS_ROW + n4 / 4u] = vec4<f32>(
                    mmb_load_b(gk, gn, k, n, mode),
                    mmb_load_b(gk, gn + 1u, k, n, mode),
                    mmb_load_b(gk, gn + 2u, k, n, mode),
                    mmb_load_b(gk, gn + 3u, k, n, mode),
                );
            }
        }
        workgroupBarrier();

        for (var kk = 0u; kk < MMB_BK; kk++) {
            let a0 = mmb_as4[kk * (MMB_BM / 4u) + lid.y + 16u * 0u];
            let b0 = mmb_bs4[kk * MMB_BS_ROW + lid.x + 16u * 0u];
            acc0_0 = fma(vec4<f32>(a0.x), b0, acc0_0);
            acc1_0 = fma(vec4<f32>(a0.y), b0, acc1_0);
            acc2_0 = fma(vec4<f32>(a0.z), b0, acc2_0);
            acc3_0 = fma(vec4<f32>(a0.w), b0, acc3_0);
        }
        workgroupBarrier();
    }

    {
        let r = row0 + (lid.y + 16u * 0u) * 4u + 0u;
        if (r < m) {
            {
                let c = col0 + (lid.x + 16u * 0u) * 4u;
                let v = acc0_0;
                if (c < n) { mmb_out[r * n + c] = v.x; }
                if (c + 1u < n) { mmb_out[r * n + c + 1u] = v.y; }
                if (c + 2u < n) { mmb_out[r * n + c + 2u] = v.z; }
                if (c + 3u < n) { mmb_out[r * n + c + 3u] = v.w; }
            }
        }
    }
    {
        let r = row0 + (lid.y + 16u * 0u) * 4u + 1u;
        if (r < m) {
            {
                let c = col0 + (lid.x + 16u * 0u) * 4u;
                let v = acc1_0;
                if (c < n) { mmb_out[r * n + c] = v.x; }
                if (c + 1u < n) { mmb_out[r * n + c + 1u] = v.y; }
                if (c + 2u < n) { mmb_out[r * n + c + 2u] = v.z; }
                if (c + 3u < n) { mmb_out[r * n + c + 3u] = v.w; }
            }
        }
    }
    {
        let r = row0 + (lid.y + 16u * 0u) * 4u + 2u;
        if (r < m) {
            {
                let c = col0 + (lid.x + 16u * 0u) * 4u;
                let v = acc2_0;
                if (c < n) { mmb_out[r * n + c] = v.x; }
                if (c + 1u < n) { mmb_out[r * n + c + 1u] = v.y; }
                if (c + 2u < n) { mmb_out[r * n + c + 2u] = v.z; }
                if (c + 3u < n) { mmb_out[r * n + c + 3u] = v.w; }
            }
        }
    }
    {
        let r = row0 + (lid.y + 16u * 0u) * 4u + 3u;
        if (r < m) {
            {
                let c = col0 + (lid.x + 16u * 0u) * 4u;
                let v = acc3_0;
                if (c < n) { mmb_out[r * n + c] = v.x; }
                if (c + 1u < n) { mmb_out[r * n + c + 1u] = v.y; }
                if (c + 2u < n) { mmb_out[r * n + c + 2u] = v.z; }
                if (c + 3u < n) { mmb_out[r * n + c + 3u] = v.w; }
            }
        }
    }
}
