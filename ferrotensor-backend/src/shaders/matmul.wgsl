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

const MMB_BM: u32 = 64u;
const MMB_BN: u32 = 64u;
const MMB_BK: u32 = 16u;

@group(0) @binding(0) var<storage, read> mmb_a: array<f32>;
@group(0) @binding(1) var<storage, read> mmb_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> mmb_out: array<f32>;
@group(0) @binding(3) var<uniform> mmb_params: vec4<u32>; // M, K, N, mode

var<workgroup> mmb_as4: array<vec4<f32>, 256>; // [MMB_BK][MMB_BM / 4]
var<workgroup> mmb_bs4: array<vec4<f32>, 256>; // [MMB_BK][MMB_BN / 4]

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
        for (var idx = lidx; idx < MMB_BK * MMB_BN / 4u; idx += 256u) {
            let kk = idx / (MMB_BN / 4u);
            let n4 = (idx % (MMB_BN / 4u)) * 4u;
            let gk = k0 + kk;
            let gn = col0 + n4;
            mmb_bs4[idx] = vec4<f32>(
                mmb_load_b(gk, gn, k, n, mode),
                mmb_load_b(gk, gn + 1u, k, n, mode),
                mmb_load_b(gk, gn + 2u, k, n, mode),
                mmb_load_b(gk, gn + 3u, k, n, mode),
            );
        }
        workgroupBarrier();

        for (var kk = 0u; kk < MMB_BK; kk++) {
            let a0 = mmb_as4[kk * (MMB_BM / 4u) + lid.y + 16u * 0u];
            let b0 = mmb_bs4[kk * (MMB_BN / 4u) + lid.x + 16u * 0u];
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
