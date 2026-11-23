// ============================================================================
// FLASH ATTENTION ON COOPERATIVE MATRICES (forward, dQ, dK/dV)
// ============================================================================
// head_dim 64, f32 buffers, causal or full attention without mask, dropout
// or window; the host routes only those calls here (engine/flash_coop.rs).
//
// The host builds this source for 64 or for 32 rows per workgroup (FC_BR):
// lines tagged `64` or `32` after a double slash and an at sign exist in that
// variant only, and four size tokens in at signs are replaced by numbers. A
// workgroup is 4 subgroups of 32 threads. With 64 rows, subgroup w owns the 16 rows
// w*16..w*16+15 (query rows in fwd and dq, key rows in dk_dv) and all 64
// columns of every fragment product. With 32 rows the 4 subgroups form 2 row
// blocks of 2: both subgroups of a row block share its 16 rows and each
// computes 2 of the 4 column blocks of every product, so a tile takes about
// half the serial work and a grid of the same sequence has twice the
// workgroups (see FC_RB .. FC_NC below). Tiles are f16 in workgroup memory
// with a row stride of 72 halfs (144 bytes), accumulators are f32 16x16
// fragments, and every fragment goes back through workgroup memory (`fc_ss`,
// 68 floats per row) because the scalar phases (softmax, dS) need element
// access. Those phases give FC_LPR lanes to each row: lane q of a row owns
// the columns FC_LPR * e + q.
//
// With RoPE on (a bit in the params), Q and K are rotated while they are
// staged, from a cos/sin table the host builds per position (`fc_rot_tab`,
// one row of 32 cosines then 32 sines per position), and dQ and dK are
// rotated back (the transposed rotation) on their way out, so the caller
// passes and receives tensors of the un-rotated space.
//
// Q, K, V, dO, P and dS are rounded to f16 (clamped to the f16 range, and a
// workgroup that clamped anything counts itself in `fc_overflows` and writes
// NaN into its first output element, so a dynamic loss scaler backs off).
// S, dP, the softmax, Delta, O, dQ, dK and dV stay f32.
//
// Every barrier and cooperative-matrix op sits in uniform control flow: loop
// bounds are `let`s of workgroup ids and parameters or constants, and the
// lane-dependent `if`s guard only scalar code. Every workgroup array is a
// multiple of 16 bytes, which keeps all fragment loads 16-byte aligned.

enable f16;
enable wgpu_cooperative_matrix;

const FC_SH: u32 = 72u;          // f16 tile row stride in halfs
const FC_SF: u32 = 68u;          // f32 tile row stride in floats
const FC_F16_MAX: f32 = 65504.0;
const FC_NONE: f32 = -3.0e38;    // a score that is masked out
const FC_NONE_LIMIT: f32 = -1.0e37;

// Rows per workgroup and how the 4 subgroups cover them.
const FC_BR: u32 = @BR@u;
const FC_RB: u32 = FC_BR / 16u;    // blocks of 16 rows
const FC_WPR: u32 = 4u / FC_RB;    // subgroups per row block
const FC_NJ: u32 = 4u / FC_WPR;    // 16-column blocks of a 64-column product per subgroup
const FC_LPR: u32 = 128u / FC_BR;  // lanes per row in the scalar phases
const FC_NC: u32 = 64u / FC_LPR;   // columns per lane
const FC_DE: u32 = 16u / FC_LPR;   // vec4s of a row per lane (Delta)
const FC_QE: u32 = FC_BR / 16u;    // staging items per thread for a tile of FC_BR rows
const FC_TE: u32 = 4u;             // the same for a tile of 64 rows

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
    mask_stride_b: u32,
    mask_stride_h: u32,
    mask_stride_q: u32,
    mask_stride_k: u32,
    window: u32,
    seed_hi: u32,
    write_stats: u32,
    rope: u32,           // bit 0: rotate Q and K from fc_rot_tab
};

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
    // These four words hold the mask strides of the scalar kernels. The
    // cooperative kernels never run with a mask, so with bit 2 of lse_saved
    // set they carry where dQ, dK and dV go: the length of a row of the one
    // buffer that holds them side by side and the column where each starts.
    out_ld: u32,
    q_off: u32,
    k_off: u32,
    v_off: u32,
    window: u32,
    seed_hi: u32,
    // bit 0: the LSE is saved; bit 1: RoPE as in the forward; bit 2: dQ, dK
    // and dV are column blocks of one wider buffer (out_ld, q_off, k_off, v_off)
    lse_saved: u32,
    dkv_direct: u32,
};

// Bindings (each entry point binds only what it reads; numbers match the
// scalar kernels where they share a name).
@group(0) @binding(0) var<storage, read> fc_q: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> fc_k: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> fc_v: array<vec4<f32>>;
// fwd: the output (f32 scalars); dq: the saved forward output (vec4s).
@group(0) @binding(3) var<storage, read_write> fc_out: array<f32>;
@group(0) @binding(4) var<uniform> fc_fp: FlashV2Params;
@group(0) @binding(6) var<storage, read_write> fc_stats: array<f32>;
@group(0) @binding(7) var<storage, read_write> fc_overflows: array<atomic<u32>>;

@group(0) @binding(13) var<storage, read> fc_o4: array<vec4<f32>>;
@group(0) @binding(14) var<storage, read> fc_do: array<vec4<f32>>;
@group(0) @binding(15) var<storage, read_write> fc_dq: array<f32>;
@group(0) @binding(16) var<storage, read_write> fc_dk: array<f32>;
@group(0) @binding(17) var<storage, read_write> fc_dv: array<f32>;
@group(0) @binding(8) var<uniform> fc_bp: FlashBwdParams;
// Per (batch, head, query) row: [Delta, LSE].
@group(0) @binding(10) var<storage, read_write> fc_stats2: array<vec2<f32>>;
@group(0) @binding(11) var<storage, read_write> fc_overflows2: array<atomic<u32>>;
// RoPE: per position, 8 vec4s of cosines then 8 vec4s of sines for the 32
// frequencies (dims d and d + 32 form a pair). A 16-byte dummy when RoPE is off.
@group(0) @binding(18) var<storage, read> fc_rot_tab: array<vec4<f32>>;

// Three f16 tiles (K and V tiles have 64 rows, the Q-side tile FC_BR), one f32
// tile (FC_BR x 68), the lane exchange, the zero source for fresh
// accumulators, and per-row LSE and Delta.
var<workgroup> fc_ta: array<f16, @TA@>;
var<workgroup> fc_tb: array<f16, 4608>;
var<workgroup> fc_tc: array<f16, 4608>;
var<workgroup> fc_ss: array<f32, @SS@>;
var<workgroup> fc_pm: array<f32, 128>;
var<workgroup> fc_zs: array<f32, 256>;
var<workgroup> fc_sl: array<f32, 64>;
var<workgroup> fc_sd: array<f32, 64>;

fn fc_bad(v: vec4<f32>) -> bool {
    return any(!(abs(v) <= vec4<f32>(FC_F16_MAX)));
}

fn fc_half(v: vec4<f32>) -> vec4<f16> {
    return vec4<f16>(clamp(v, vec4<f32>(-FC_F16_MAX), vec4<f32>(FC_F16_MAX)));
}

struct FcPair {
    a: vec4<f32>,
    b: vec4<f32>,
}

// Rotates dims 4 * c4 .. +3 (`va`) and their partners 32 further on (`vb`) of
// the row at table position `pos`, as rotate-half RoPE does.
fn fc_rot(pos: u32, c4: u32, va: vec4<f32>, vb: vec4<f32>) -> FcPair {
    let cs = fc_rot_tab[pos * 16u + c4];
    let sn = fc_rot_tab[pos * 16u + 8u + c4];
    return FcPair(va * cs - vb * sn, va * sn + vb * cs);
}

// The transposed rotation of the pair (g1 at dim d, g2 at dim d + 32), d < 32.
fn fc_inv_rot(pos: u32, d: u32, g1: f32, g2: f32) -> vec2<f32> {
    let cs = fc_rot_tab[pos * 16u + d / 4u][d % 4u];
    let sn = fc_rot_tab[pos * 16u + 8u + d / 4u][d % 4u];
    return vec2<f32>(g1 * cs + g2 * sn, g2 * cs - g1 * sn);
}

// ---------------------------------------------------------------------------
// Forward. Dispatch: (1, ceil(seq_q / FC_BR), batch * num_heads).
// Per (query block, head) the K/V blocks stream through ta..tc as
//   ta: Q, then P    tb: K    tc: V
// with the online softmax in f32 registers and O kept in f32 registers (a
// fragment cannot be scaled per row, so O = O * corr + PV is done on lanes).
// ---------------------------------------------------------------------------
@compute @workgroup_size(128)
fn flash_coop_fwd(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let p = fc_fp;
    let n_qb = (p.seq_q + FC_BR - 1u) / FC_BR;
    let qb = n_qb - 1u - wg.y; // the longest causal blocks start first
    let b = wg.z / p.num_heads;
    let h = wg.z % p.num_heads;
    let kvh = h / (p.num_heads / p.num_kv_heads);
    let warp = li / 32u;
    let rb = warp / FC_WPR;          // the row block this subgroup's fragments cover
    let j0 = (warp % FC_WPR) * FC_NJ; // its first 16-column block
    let rl = li / FC_LPR;            // this lane's row of the block
    let q = li % FC_LPR;             // its lane within the row
    let gb = li - q;                 // first lane of the row
    let i = qb * FC_BR + rl;         // its query
    let scale = p.scale;
    let rope = (p.rope & 1u) != 0u;

    for (var z = li; z < 256u; z += 128u) { fc_zs[z] = 0.0; }
    var overflow = false;
    for (var e = 0u; e < FC_QE; e++) {
        let idx = li + e * 128u;
        let r = idx / 8u;
        let c4 = idx % 8u;
        let row = qb * FC_BR + r;
        var lo = vec4<f32>(0.0);
        var hi = vec4<f32>(0.0);
        if (row < p.seq_q) {
            let base = ((b * p.seq_q + row) * p.num_heads + h) * 16u + c4;
            lo = fc_q[base];
            hi = fc_q[base + 8u];
            if (rope) {
                let rp = fc_rot(row, c4, lo, hi);
                lo = rp.a;
                hi = rp.b;
            }
        }
        if (fc_bad(lo) || fc_bad(hi)) { overflow = true; }
        let hl = fc_half(lo);
        let hh = fc_half(hi);
        let o = r * FC_SH + c4 * 4u;
        fc_ta[o] = hl.x; fc_ta[o + 1u] = hl.y; fc_ta[o + 2u] = hl.z; fc_ta[o + 3u] = hl.w;
        fc_ta[o + 32u] = hh.x; fc_ta[o + 33u] = hh.y; fc_ta[o + 34u] = hh.z; fc_ta[o + 35u] = hh.w;
    }
    workgroupBarrier();
    let fq0 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(rb * 16u) * FC_SH + 0u], FC_SH);
    let fq1 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(rb * 16u) * FC_SH + 16u], FC_SH);
    let fq2 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(rb * 16u) * FC_SH + 32u], FC_SH);
    let fq3 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(rb * 16u) * FC_SH + 48u], FC_SH);
    workgroupBarrier();

    var m_run = FC_NONE;
    var l_run = 0.0;
    var o_acc: array<f32, @NC@>;

    let n_kv_all = (p.seq_k + 63u) / 64u;
    let n_kv_causal = min(n_kv_all, (qb * FC_BR + FC_BR - 1u + p.causal_offset) / 64u + 1u);
    let n_kv = select(n_kv_all, n_kv_causal, p.causal == 1u);
    for (var t = 0u; t < n_kv; t++) {
        let k0 = t * 64u;
        for (var e = 0u; e < FC_TE; e++) {
            let idx = li + e * 128u;
            let r = idx / 8u;
            let c4 = idx % 8u;
            let row = k0 + r;
            var k_lo = vec4<f32>(0.0);
            var k_hi = vec4<f32>(0.0);
            var v_lo = vec4<f32>(0.0);
            var v_hi = vec4<f32>(0.0);
            if (row < p.seq_k) {
                let base = ((b * p.seq_k + row) * p.num_kv_heads + kvh) * 16u + c4;
                k_lo = fc_k[base];
                k_hi = fc_k[base + 8u];
                v_lo = fc_v[base];
                v_hi = fc_v[base + 8u];
                if (rope) {
                    let rp = fc_rot(row, c4, k_lo, k_hi);
                    k_lo = rp.a;
                    k_hi = rp.b;
                }
            }
            if (fc_bad(k_lo) || fc_bad(k_hi) || fc_bad(v_lo) || fc_bad(v_hi)) { overflow = true; }
            let kl = fc_half(k_lo);
            let kh = fc_half(k_hi);
            let vl = fc_half(v_lo);
            let vh = fc_half(v_hi);
            let o = r * FC_SH + c4 * 4u;
            fc_tb[o] = kl.x; fc_tb[o + 1u] = kl.y; fc_tb[o + 2u] = kl.z; fc_tb[o + 3u] = kl.w;
            fc_tb[o + 32u] = kh.x; fc_tb[o + 33u] = kh.y; fc_tb[o + 34u] = kh.z; fc_tb[o + 35u] = kh.w;
            fc_tc[o] = vl.x; fc_tc[o + 1u] = vl.y; fc_tc[o + 2u] = vl.z; fc_tc[o + 3u] = vl.w;
            fc_tc[o + 32u] = vh.x; fc_tc[o + 33u] = vh.y; fc_tc[o + 34u] = vh.z; fc_tc[o + 35u] = vh.w;
        }
        workgroupBarrier();
        // S = Q K^T: this subgroup's FC_NJ column blocks of its row block.
        for (var jj = 0u; jj < FC_NJ; jj++) {
            let j = j0 + jj;
            var s = coopLoadT<coop_mat16x16<f32, C>>(&fc_zs[0], 16u);
            s = coopMultiplyAdd(fq0, coopLoad<coop_mat16x16<f16, B>>(&fc_tb[(j * 16u) * FC_SH + 0u], FC_SH), s);
            s = coopMultiplyAdd(fq1, coopLoad<coop_mat16x16<f16, B>>(&fc_tb[(j * 16u) * FC_SH + 16u], FC_SH), s);
            s = coopMultiplyAdd(fq2, coopLoad<coop_mat16x16<f16, B>>(&fc_tb[(j * 16u) * FC_SH + 32u], FC_SH), s);
            s = coopMultiplyAdd(fq3, coopLoad<coop_mat16x16<f16, B>>(&fc_tb[(j * 16u) * FC_SH + 48u], FC_SH), s);
            coopStoreT(s, &fc_ss[(rb * 16u) * FC_SF + j * 16u], FC_SF);
        }
        workgroupBarrier();
        // Online softmax over this lane's columns of its row.
        var x: array<f32, @NC@>;
        var mx = FC_NONE;
        for (var e = 0u; e < FC_NC; e++) {
            let c = FC_LPR * e + q;
            var xv = fc_ss[rl * FC_SF + c] * scale;
            let kc = k0 + c;
            if (kc >= p.seq_k || (p.causal == 1u && kc > i + p.causal_offset)) { xv = FC_NONE; }
            x[e] = xv;
            mx = max(mx, xv);
        }
        fc_pm[li] = mx;
        workgroupBarrier();
        for (var k = 0u; k < FC_LPR; k++) { mx = max(mx, fc_pm[gb + k]); }
        let m_new = max(m_run, mx);
        let live = m_new > FC_NONE_LIMIT;
        let corr = select(1.0, exp(m_run - m_new), live);
        var psum = 0.0;
        for (var e = 0u; e < FC_NC; e++) {
            let pe = select(0.0, exp(x[e] - m_new), live);
            psum += pe;
            fc_ta[rl * FC_SH + FC_LPR * e + q] = f16(pe);
        }
        l_run = l_run * corr + psum;
        m_run = m_new;
        workgroupBarrier();
        // PV
        let fp0 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(rb * 16u) * FC_SH + 0u], FC_SH);
        let fp1 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(rb * 16u) * FC_SH + 16u], FC_SH);
        let fp2 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(rb * 16u) * FC_SH + 32u], FC_SH);
        let fp3 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(rb * 16u) * FC_SH + 48u], FC_SH);
        for (var jj = 0u; jj < FC_NJ; jj++) {
            let n = j0 + jj;
            var pv = coopLoadT<coop_mat16x16<f32, C>>(&fc_zs[0], 16u);
            pv = coopMultiplyAdd(fp0, coopLoadT<coop_mat16x16<f16, B>>(&fc_tc[0u * FC_SH + n * 16u], FC_SH), pv);
            pv = coopMultiplyAdd(fp1, coopLoadT<coop_mat16x16<f16, B>>(&fc_tc[16u * FC_SH + n * 16u], FC_SH), pv);
            pv = coopMultiplyAdd(fp2, coopLoadT<coop_mat16x16<f16, B>>(&fc_tc[32u * FC_SH + n * 16u], FC_SH), pv);
            pv = coopMultiplyAdd(fp3, coopLoadT<coop_mat16x16<f16, B>>(&fc_tc[48u * FC_SH + n * 16u], FC_SH), pv);
            coopStoreT(pv, &fc_ss[(rb * 16u) * FC_SF + n * 16u], FC_SF);
        }
        workgroupBarrier();
        for (var e = 0u; e < FC_NC; e++) {
            o_acc[e] = o_acc[e] * corr + fc_ss[rl * FC_SF + FC_LPR * e + q];
        }
        workgroupBarrier();
    }

    // The lanes of a row each summed a share of its columns.
    fc_pm[li] = l_run;
    workgroupBarrier();
    var l = 0.0;
    for (var k = 0u; k < FC_LPR; k++) { l += fc_pm[gb + k]; }
    if (i < p.seq_q) {
        let ob = ((b * p.seq_q + i) * p.num_heads + h) * 64u;
        for (var e = 0u; e < FC_NC; e++) {
            fc_out[ob + FC_LPR * e + q] = select(0.0, o_acc[e] / l, l > 0.0);
        }
        if (p.write_stats == 1u && q == 0u) {
            // A row that saw no visible key keeps 3.0e38, which makes its P zero in backward.
            fc_stats[2u * ((b * p.num_heads + h) * p.seq_q + i) + 1u] = select(3.0e38, m_run + log(l), l > 0.0);
        }
    }
    // After the writes above, so the poison is the last write to that element.
    storageBarrier();
    if (overflow) {
        atomicAdd(&fc_overflows[0], 1u);
        fc_out[((b * p.seq_q + qb * FC_BR) * p.num_heads + h) * 64u] = bitcast<f32>(0x7fc00000u);
    }
}

// ---------------------------------------------------------------------------
// Backward, dQ. Dispatch: (1, ceil(seq_q / FC_BR), batch * num_heads).
//   ta: Q, then dS    tb: dO, then K    tc: V
// Writes Delta = dO . O (f32, from the saved O) into stats .x, reads the
// forward pass's LSE from stats .y. dS carries the softmax scale. With RoPE
// the dQ written is the gradient of the un-rotated Q.
// ---------------------------------------------------------------------------
@compute @workgroup_size(128)
fn flash_coop_bwd_dq(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let p = fc_bp;
    let n_qb = (p.seq_q + FC_BR - 1u) / FC_BR;
    let qb = n_qb - 1u - wg.y;
    let b = wg.z / p.num_heads;
    let h = wg.z % p.num_heads;
    let kvh = h / (p.num_heads / p.num_kv_heads);
    let warp = li / 32u;
    let rb = warp / FC_WPR;
    let j0 = (warp % FC_WPR) * FC_NJ;
    let rl = li / FC_LPR;
    let q = li % FC_LPR;
    let gb = li - q;
    let i = qb * FC_BR + rl;
    let scale = p.scale;
    let rope = (p.lse_saved & 2u) != 0u;
    let strided = (p.lse_saved & 4u) != 0u;
    let row_id = (b * p.num_heads + h) * p.seq_q + i;
    let cq0 = j0 * 16u;        // the columns of this subgroup's dQ accumulators
    let cq1 = cq0 + 16u;
    let cq2 = cq0 + 32u;       //@64
    let cq3 = cq0 + 48u;       //@64

    for (var z = li; z < 256u; z += 128u) { fc_zs[z] = 0.0; }
    var overflow = false;
    // Delta from the unrounded O and dO; the lanes of a row split its 16 vec4s.
    var dsum = 0.0;
    if (i < p.seq_q) {
        let ob4 = ((b * p.seq_q + i) * p.num_heads + h) * 16u;
        for (var e = 0u; e < FC_DE; e++) {
            let c4 = q * FC_DE + e;
            dsum += dot(fc_o4[ob4 + c4], fc_do[ob4 + c4]);
        }
    }
    fc_pm[li] = dsum;
    workgroupBarrier();
    var delta = 0.0;
    for (var k = 0u; k < FC_LPR; k++) { delta += fc_pm[gb + k]; }
    if (q == 0u) {
        var lse = 3.0e38;
        var del = 0.0;
        if (i < p.seq_q) {
            lse = fc_stats2[row_id].y;
            del = delta;
            fc_stats2[row_id].x = delta;
        }
        fc_sl[rl] = lse;
        fc_sd[rl] = del;
    }
    for (var e = 0u; e < FC_QE; e++) {
        let idx = li + e * 128u;
        let r = idx / 8u;
        let c4 = idx % 8u;
        let row = qb * FC_BR + r;
        var q_lo = vec4<f32>(0.0);
        var q_hi = vec4<f32>(0.0);
        var g_lo = vec4<f32>(0.0);
        var g_hi = vec4<f32>(0.0);
        if (row < p.seq_q) {
            let base = ((b * p.seq_q + row) * p.num_heads + h) * 16u + c4;
            q_lo = fc_q[base];
            q_hi = fc_q[base + 8u];
            g_lo = fc_do[base];
            g_hi = fc_do[base + 8u];
            if (rope) {
                let rp = fc_rot(row, c4, q_lo, q_hi);
                q_lo = rp.a;
                q_hi = rp.b;
            }
        }
        if (fc_bad(q_lo) || fc_bad(q_hi) || fc_bad(g_lo) || fc_bad(g_hi)) { overflow = true; }
        let ql = fc_half(q_lo);
        let qh = fc_half(q_hi);
        let gl = fc_half(g_lo);
        let gh = fc_half(g_hi);
        let o = r * FC_SH + c4 * 4u;
        fc_ta[o] = ql.x; fc_ta[o + 1u] = ql.y; fc_ta[o + 2u] = ql.z; fc_ta[o + 3u] = ql.w;
        fc_ta[o + 32u] = qh.x; fc_ta[o + 33u] = qh.y; fc_ta[o + 34u] = qh.z; fc_ta[o + 35u] = qh.w;
        fc_tb[o] = gl.x; fc_tb[o + 1u] = gl.y; fc_tb[o + 2u] = gl.z; fc_tb[o + 3u] = gl.w;
        fc_tb[o + 32u] = gh.x; fc_tb[o + 33u] = gh.y; fc_tb[o + 34u] = gh.z; fc_tb[o + 35u] = gh.w;
    }
    workgroupBarrier();
    let fq0 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(rb * 16u) * FC_SH + 0u], FC_SH);
    let fq1 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(rb * 16u) * FC_SH + 16u], FC_SH);
    let fq2 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(rb * 16u) * FC_SH + 32u], FC_SH);
    let fq3 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(rb * 16u) * FC_SH + 48u], FC_SH);
    let fg0 = coopLoadT<coop_mat16x16<f16, A>>(&fc_tb[(rb * 16u) * FC_SH + 0u], FC_SH);
    let fg1 = coopLoadT<coop_mat16x16<f16, A>>(&fc_tb[(rb * 16u) * FC_SH + 16u], FC_SH);
    let fg2 = coopLoadT<coop_mat16x16<f16, A>>(&fc_tb[(rb * 16u) * FC_SH + 32u], FC_SH);
    let fg3 = coopLoadT<coop_mat16x16<f16, A>>(&fc_tb[(rb * 16u) * FC_SH + 48u], FC_SH);
    workgroupBarrier();

    var dq0 = coopLoadT<coop_mat16x16<f32, C>>(&fc_zs[0], 16u);
    var dq1 = coopLoadT<coop_mat16x16<f32, C>>(&fc_zs[0], 16u);
    var dq2 = coopLoadT<coop_mat16x16<f32, C>>(&fc_zs[0], 16u); //@64
    var dq3 = coopLoadT<coop_mat16x16<f32, C>>(&fc_zs[0], 16u); //@64
    let lse_r = fc_sl[rl];
    let del_r = fc_sd[rl];

    let n_kv_all = (p.seq_k + 63u) / 64u;
    let n_kv_causal = min(n_kv_all, (qb * FC_BR + FC_BR - 1u + p.causal_offset) / 64u + 1u);
    let n_kv = select(n_kv_all, n_kv_causal, p.causal == 1u);
    for (var t = 0u; t < n_kv; t++) {
        let k0 = t * 64u;
        for (var e = 0u; e < FC_TE; e++) {
            let idx = li + e * 128u;
            let r = idx / 8u;
            let c4 = idx % 8u;
            let row = k0 + r;
            var k_lo = vec4<f32>(0.0);
            var k_hi = vec4<f32>(0.0);
            var v_lo = vec4<f32>(0.0);
            var v_hi = vec4<f32>(0.0);
            if (row < p.seq_k) {
                let base = ((b * p.seq_k + row) * p.num_kv_heads + kvh) * 16u + c4;
                k_lo = fc_k[base];
                k_hi = fc_k[base + 8u];
                v_lo = fc_v[base];
                v_hi = fc_v[base + 8u];
                if (rope) {
                    let rp = fc_rot(row, c4, k_lo, k_hi);
                    k_lo = rp.a;
                    k_hi = rp.b;
                }
            }
            if (fc_bad(k_lo) || fc_bad(k_hi) || fc_bad(v_lo) || fc_bad(v_hi)) { overflow = true; }
            let kl = fc_half(k_lo);
            let kh = fc_half(k_hi);
            let vl = fc_half(v_lo);
            let vh = fc_half(v_hi);
            let o = r * FC_SH + c4 * 4u;
            fc_tb[o] = kl.x; fc_tb[o + 1u] = kl.y; fc_tb[o + 2u] = kl.z; fc_tb[o + 3u] = kl.w;
            fc_tb[o + 32u] = kh.x; fc_tb[o + 33u] = kh.y; fc_tb[o + 34u] = kh.z; fc_tb[o + 35u] = kh.w;
            fc_tc[o] = vl.x; fc_tc[o + 1u] = vl.y; fc_tc[o + 2u] = vl.z; fc_tc[o + 3u] = vl.w;
            fc_tc[o + 32u] = vh.x; fc_tc[o + 33u] = vh.y; fc_tc[o + 34u] = vh.z; fc_tc[o + 35u] = vh.w;
        }
        workgroupBarrier();
        // S = Q K^T
        for (var jj = 0u; jj < FC_NJ; jj++) {
            let j = j0 + jj;
            var s = coopLoadT<coop_mat16x16<f32, C>>(&fc_zs[0], 16u);
            s = coopMultiplyAdd(fq0, coopLoad<coop_mat16x16<f16, B>>(&fc_tb[(j * 16u) * FC_SH + 0u], FC_SH), s);
            s = coopMultiplyAdd(fq1, coopLoad<coop_mat16x16<f16, B>>(&fc_tb[(j * 16u) * FC_SH + 16u], FC_SH), s);
            s = coopMultiplyAdd(fq2, coopLoad<coop_mat16x16<f16, B>>(&fc_tb[(j * 16u) * FC_SH + 32u], FC_SH), s);
            s = coopMultiplyAdd(fq3, coopLoad<coop_mat16x16<f16, B>>(&fc_tb[(j * 16u) * FC_SH + 48u], FC_SH), s);
            coopStoreT(s, &fc_ss[(rb * 16u) * FC_SF + j * 16u], FC_SF);
        }
        workgroupBarrier();
        var pr: array<f32, @NC@>;
        for (var e = 0u; e < FC_NC; e++) {
            let c = FC_LPR * e + q;
            let kc = k0 + c;
            var pe = exp(fc_ss[rl * FC_SF + c] * scale - lse_r);
            if (i >= p.seq_q || kc >= p.seq_k || (p.causal == 1u && kc > i + p.causal_offset)) { pe = 0.0; }
            pr[e] = pe;
        }
        workgroupBarrier();
        // dP = dO V^T
        for (var jj = 0u; jj < FC_NJ; jj++) {
            let j = j0 + jj;
            var d = coopLoadT<coop_mat16x16<f32, C>>(&fc_zs[0], 16u);
            d = coopMultiplyAdd(fg0, coopLoad<coop_mat16x16<f16, B>>(&fc_tc[(j * 16u) * FC_SH + 0u], FC_SH), d);
            d = coopMultiplyAdd(fg1, coopLoad<coop_mat16x16<f16, B>>(&fc_tc[(j * 16u) * FC_SH + 16u], FC_SH), d);
            d = coopMultiplyAdd(fg2, coopLoad<coop_mat16x16<f16, B>>(&fc_tc[(j * 16u) * FC_SH + 32u], FC_SH), d);
            d = coopMultiplyAdd(fg3, coopLoad<coop_mat16x16<f16, B>>(&fc_tc[(j * 16u) * FC_SH + 48u], FC_SH), d);
            coopStoreT(d, &fc_ss[(rb * 16u) * FC_SF + j * 16u], FC_SF);
        }
        workgroupBarrier();
        // dS = P (dP - Delta) scale, as f16 for the dQ product.
        for (var e = 0u; e < FC_NC; e++) {
            let c = FC_LPR * e + q;
            let ds = pr[e] * (fc_ss[rl * FC_SF + c] - del_r) * scale;
            if (!(abs(ds) <= FC_F16_MAX)) { overflow = true; }
            fc_ta[rl * FC_SH + c] = f16(clamp(ds, -FC_F16_MAX, FC_F16_MAX));
        }
        workgroupBarrier();
        // dQ += dS K
        let fd0 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(rb * 16u) * FC_SH + 0u], FC_SH);
        let fd1 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(rb * 16u) * FC_SH + 16u], FC_SH);
        let fd2 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(rb * 16u) * FC_SH + 32u], FC_SH);
        let fd3 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(rb * 16u) * FC_SH + 48u], FC_SH);
        dq0 = coopMultiplyAdd(fd0, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[0u * FC_SH + cq0], FC_SH), dq0);
        dq0 = coopMultiplyAdd(fd1, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[16u * FC_SH + cq0], FC_SH), dq0);
        dq0 = coopMultiplyAdd(fd2, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[32u * FC_SH + cq0], FC_SH), dq0);
        dq0 = coopMultiplyAdd(fd3, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[48u * FC_SH + cq0], FC_SH), dq0);
        dq1 = coopMultiplyAdd(fd0, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[0u * FC_SH + cq1], FC_SH), dq1);
        dq1 = coopMultiplyAdd(fd1, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[16u * FC_SH + cq1], FC_SH), dq1);
        dq1 = coopMultiplyAdd(fd2, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[32u * FC_SH + cq1], FC_SH), dq1);
        dq1 = coopMultiplyAdd(fd3, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[48u * FC_SH + cq1], FC_SH), dq1);
        dq2 = coopMultiplyAdd(fd0, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[0u * FC_SH + cq2], FC_SH), dq2); //@64
        dq2 = coopMultiplyAdd(fd1, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[16u * FC_SH + cq2], FC_SH), dq2); //@64
        dq2 = coopMultiplyAdd(fd2, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[32u * FC_SH + cq2], FC_SH), dq2); //@64
        dq2 = coopMultiplyAdd(fd3, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[48u * FC_SH + cq2], FC_SH), dq2); //@64
        dq3 = coopMultiplyAdd(fd0, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[0u * FC_SH + cq3], FC_SH), dq3); //@64
        dq3 = coopMultiplyAdd(fd1, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[16u * FC_SH + cq3], FC_SH), dq3); //@64
        dq3 = coopMultiplyAdd(fd2, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[32u * FC_SH + cq3], FC_SH), dq3); //@64
        dq3 = coopMultiplyAdd(fd3, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[48u * FC_SH + cq3], FC_SH), dq3); //@64
        workgroupBarrier();
    }

    coopStoreT(dq0, &fc_ss[(rb * 16u) * FC_SF + cq0], FC_SF);
    coopStoreT(dq1, &fc_ss[(rb * 16u) * FC_SF + cq1], FC_SF);
    coopStoreT(dq2, &fc_ss[(rb * 16u) * FC_SF + cq2], FC_SF); //@64
    coopStoreT(dq3, &fc_ss[(rb * 16u) * FC_SF + cq3], FC_SF); //@64
    workgroupBarrier();
    if (i < p.seq_q) {
        var ob = ((b * p.seq_q + i) * p.num_heads + h) * 64u;
        if (strided) { ob = (b * p.seq_q + i) * p.out_ld + p.q_off + h * 64u; }
        if (rope) {
            // The pair (c, c + 32) sits in one lane, since 32 is a multiple of FC_LPR.
            for (var e = 0u; e < FC_NC / 2u; e++) {
                let c = FC_LPR * e + q;
                let g = fc_inv_rot(i, c, fc_ss[rl * FC_SF + c], fc_ss[rl * FC_SF + c + 32u]);
                fc_dq[ob + c] = g.x;
                fc_dq[ob + c + 32u] = g.y;
            }
        } else {
            for (var e = 0u; e < FC_NC; e++) {
                fc_dq[ob + FC_LPR * e + q] = fc_ss[rl * FC_SF + FC_LPR * e + q];
            }
        }
    }
    storageBarrier();
    if (overflow) {
        atomicAdd(&fc_overflows2[0], 1u);
        var first = ((b * p.seq_q + qb * FC_BR) * p.num_heads + h) * 64u;
        if (strided) { first = (b * p.seq_q + qb * FC_BR) * p.out_ld + p.q_off + h * 64u; }
        fc_dq[first] = bitcast<f32>(0x7fc00000u);
    }
}

// ---------------------------------------------------------------------------
// Backward, dK and dV. Dispatch: (1, ceil(seq_k / FC_BR), batch * num_heads):
// one workgroup per (key block, query head), so a group of query heads that
// share a KV head runs in parallel. With dkv_direct (one query head per KV
// head) the result goes straight to dK/dV [batch, seq_k, kv heads, 64];
// otherwise to per-query-head partials [batch, seq_k, heads, 64] that
// flash_group_sum adds up. With RoPE the dK written is the gradient of the
// un-rotated K.
//   ta: K, then P^T, then dS^T    tb: V, then Q    tc: dO
// Row block rb holds key rows rb*16..rb*16+15 and the columns are queries.
// ---------------------------------------------------------------------------
@compute @workgroup_size(128)
fn flash_coop_bwd_dkdv(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let p = fc_bp;
    let kb = wg.y; // the blocks with the most query tiles start first
    let b = wg.z / p.num_heads;
    let h = wg.z % p.num_heads;
    let kvh = h / (p.num_heads / p.num_kv_heads);
    let warp = li / 32u;
    let rb = warp / FC_WPR;
    let j0 = (warp % FC_WPR) * FC_NJ;
    let rl = li / FC_LPR;
    let q = li % FC_LPR;
    let k0 = kb * FC_BR;
    let kr = k0 + rl; // this lane's key
    let scale = p.scale;
    let rope = (p.lse_saved & 2u) != 0u;
    let strided = (p.lse_saved & 4u) != 0u;
    let cq0 = j0 * 16u;
    let cq1 = cq0 + 16u;
    let cq2 = cq0 + 32u;       //@64
    let cq3 = cq0 + 48u;       //@64

    for (var z = li; z < 256u; z += 128u) { fc_zs[z] = 0.0; }
    var overflow = false;
    for (var e = 0u; e < FC_QE; e++) {
        let idx = li + e * 128u;
        let r = idx / 8u;
        let c4 = idx % 8u;
        let row = k0 + r;
        var k_lo = vec4<f32>(0.0);
        var k_hi = vec4<f32>(0.0);
        var v_lo = vec4<f32>(0.0);
        var v_hi = vec4<f32>(0.0);
        if (row < p.seq_k) {
            let base = ((b * p.seq_k + row) * p.num_kv_heads + kvh) * 16u + c4;
            k_lo = fc_k[base];
            k_hi = fc_k[base + 8u];
            v_lo = fc_v[base];
            v_hi = fc_v[base + 8u];
            if (rope) {
                let rp = fc_rot(row, c4, k_lo, k_hi);
                k_lo = rp.a;
                k_hi = rp.b;
            }
        }
        if (fc_bad(k_lo) || fc_bad(k_hi) || fc_bad(v_lo) || fc_bad(v_hi)) { overflow = true; }
        let kl = fc_half(k_lo);
        let kh = fc_half(k_hi);
        let vl = fc_half(v_lo);
        let vh = fc_half(v_hi);
        let o = r * FC_SH + c4 * 4u;
        fc_ta[o] = kl.x; fc_ta[o + 1u] = kl.y; fc_ta[o + 2u] = kl.z; fc_ta[o + 3u] = kl.w;
        fc_ta[o + 32u] = kh.x; fc_ta[o + 33u] = kh.y; fc_ta[o + 34u] = kh.z; fc_ta[o + 35u] = kh.w;
        fc_tb[o] = vl.x; fc_tb[o + 1u] = vl.y; fc_tb[o + 2u] = vl.z; fc_tb[o + 3u] = vl.w;
        fc_tb[o + 32u] = vh.x; fc_tb[o + 33u] = vh.y; fc_tb[o + 34u] = vh.z; fc_tb[o + 35u] = vh.w;
    }
    workgroupBarrier();
    let fk0 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(rb * 16u) * FC_SH + 0u], FC_SH);
    let fk1 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(rb * 16u) * FC_SH + 16u], FC_SH);
    let fk2 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(rb * 16u) * FC_SH + 32u], FC_SH);
    let fk3 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(rb * 16u) * FC_SH + 48u], FC_SH);
    let fv0 = coopLoadT<coop_mat16x16<f16, A>>(&fc_tb[(rb * 16u) * FC_SH + 0u], FC_SH);
    let fv1 = coopLoadT<coop_mat16x16<f16, A>>(&fc_tb[(rb * 16u) * FC_SH + 16u], FC_SH);
    let fv2 = coopLoadT<coop_mat16x16<f16, A>>(&fc_tb[(rb * 16u) * FC_SH + 32u], FC_SH);
    let fv3 = coopLoadT<coop_mat16x16<f16, A>>(&fc_tb[(rb * 16u) * FC_SH + 48u], FC_SH);
    workgroupBarrier();

    var dk0 = coopLoadT<coop_mat16x16<f32, C>>(&fc_zs[0], 16u);
    var dk1 = coopLoadT<coop_mat16x16<f32, C>>(&fc_zs[0], 16u);
    var dk2 = coopLoadT<coop_mat16x16<f32, C>>(&fc_zs[0], 16u); //@64
    var dk3 = coopLoadT<coop_mat16x16<f32, C>>(&fc_zs[0], 16u); //@64
    var dv0 = coopLoadT<coop_mat16x16<f32, C>>(&fc_zs[0], 16u);
    var dv1 = coopLoadT<coop_mat16x16<f32, C>>(&fc_zs[0], 16u);
    var dv2 = coopLoadT<coop_mat16x16<f32, C>>(&fc_zs[0], 16u); //@64
    var dv3 = coopLoadT<coop_mat16x16<f32, C>>(&fc_zs[0], 16u); //@64

    // Query tiles that can see this key block: from the one holding the first query that sees its first key when causal.
    let n_qt = (p.seq_q + 63u) / 64u;
    let qt0 = select(0u, (k0 - min(k0, p.causal_offset)) / 64u, p.causal == 1u);
    for (var qt = qt0; qt < n_qt; qt++) {
        let q0 = qt * 64u;
        for (var e = 0u; e < FC_TE; e++) {
            let idx = li + e * 128u;
            let r = idx / 8u;
            let c4 = idx % 8u;
            let row = q0 + r;
            var q_lo = vec4<f32>(0.0);
            var q_hi = vec4<f32>(0.0);
            var g_lo = vec4<f32>(0.0);
            var g_hi = vec4<f32>(0.0);
            if (row < p.seq_q) {
                let base = ((b * p.seq_q + row) * p.num_heads + h) * 16u + c4;
                q_lo = fc_q[base];
                q_hi = fc_q[base + 8u];
                g_lo = fc_do[base];
                g_hi = fc_do[base + 8u];
                if (rope) {
                    let rp = fc_rot(row, c4, q_lo, q_hi);
                    q_lo = rp.a;
                    q_hi = rp.b;
                }
            }
            if (fc_bad(q_lo) || fc_bad(q_hi) || fc_bad(g_lo) || fc_bad(g_hi)) { overflow = true; }
            let ql = fc_half(q_lo);
            let qh = fc_half(q_hi);
            let gl = fc_half(g_lo);
            let gh = fc_half(g_hi);
            let o = r * FC_SH + c4 * 4u;
            fc_tb[o] = ql.x; fc_tb[o + 1u] = ql.y; fc_tb[o + 2u] = ql.z; fc_tb[o + 3u] = ql.w;
            fc_tb[o + 32u] = qh.x; fc_tb[o + 33u] = qh.y; fc_tb[o + 34u] = qh.z; fc_tb[o + 35u] = qh.w;
            fc_tc[o] = gl.x; fc_tc[o + 1u] = gl.y; fc_tc[o + 2u] = gl.z; fc_tc[o + 3u] = gl.w;
            fc_tc[o + 32u] = gh.x; fc_tc[o + 33u] = gh.y; fc_tc[o + 34u] = gh.z; fc_tc[o + 35u] = gh.w;
        }
        if (li < 64u) {
            // A row past the sequence gets LSE 3.0e38, so its P is 0.
            var lse = 3.0e38;
            var del = 0.0;
            if (q0 + li < p.seq_q) {
                let st = fc_stats2[(b * p.num_heads + h) * p.seq_q + q0 + li];
                del = st.x;
                lse = st.y;
            }
            fc_sl[li] = lse;
            fc_sd[li] = del;
        }
        workgroupBarrier();
        // S^T = K Q^T (keys x queries)
        for (var jj = 0u; jj < FC_NJ; jj++) {
            let j = j0 + jj;
            var s = coopLoadT<coop_mat16x16<f32, C>>(&fc_zs[0], 16u);
            s = coopMultiplyAdd(fk0, coopLoad<coop_mat16x16<f16, B>>(&fc_tb[(j * 16u) * FC_SH + 0u], FC_SH), s);
            s = coopMultiplyAdd(fk1, coopLoad<coop_mat16x16<f16, B>>(&fc_tb[(j * 16u) * FC_SH + 16u], FC_SH), s);
            s = coopMultiplyAdd(fk2, coopLoad<coop_mat16x16<f16, B>>(&fc_tb[(j * 16u) * FC_SH + 32u], FC_SH), s);
            s = coopMultiplyAdd(fk3, coopLoad<coop_mat16x16<f16, B>>(&fc_tb[(j * 16u) * FC_SH + 48u], FC_SH), s);
            coopStoreT(s, &fc_ss[(rb * 16u) * FC_SF + j * 16u], FC_SF);
        }
        workgroupBarrier();
        var pr: array<f32, @NC@>;
        for (var e = 0u; e < FC_NC; e++) {
            let c = FC_LPR * e + q;
            let qc = q0 + c;
            var pe = exp(fc_ss[rl * FC_SF + c] * scale - fc_sl[c]);
            if (qc >= p.seq_q || kr >= p.seq_k || (p.causal == 1u && kr > qc + p.causal_offset)) { pe = 0.0; }
            pr[e] = pe;
            fc_ta[rl * FC_SH + c] = f16(pe);
        }
        workgroupBarrier();
        // dV += P^T dO
        let fp0 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(rb * 16u) * FC_SH + 0u], FC_SH);
        let fp1 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(rb * 16u) * FC_SH + 16u], FC_SH);
        let fp2 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(rb * 16u) * FC_SH + 32u], FC_SH);
        let fp3 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(rb * 16u) * FC_SH + 48u], FC_SH);
        dv0 = coopMultiplyAdd(fp0, coopLoadT<coop_mat16x16<f16, B>>(&fc_tc[0u * FC_SH + cq0], FC_SH), dv0);
        dv0 = coopMultiplyAdd(fp1, coopLoadT<coop_mat16x16<f16, B>>(&fc_tc[16u * FC_SH + cq0], FC_SH), dv0);
        dv0 = coopMultiplyAdd(fp2, coopLoadT<coop_mat16x16<f16, B>>(&fc_tc[32u * FC_SH + cq0], FC_SH), dv0);
        dv0 = coopMultiplyAdd(fp3, coopLoadT<coop_mat16x16<f16, B>>(&fc_tc[48u * FC_SH + cq0], FC_SH), dv0);
        dv1 = coopMultiplyAdd(fp0, coopLoadT<coop_mat16x16<f16, B>>(&fc_tc[0u * FC_SH + cq1], FC_SH), dv1);
        dv1 = coopMultiplyAdd(fp1, coopLoadT<coop_mat16x16<f16, B>>(&fc_tc[16u * FC_SH + cq1], FC_SH), dv1);
        dv1 = coopMultiplyAdd(fp2, coopLoadT<coop_mat16x16<f16, B>>(&fc_tc[32u * FC_SH + cq1], FC_SH), dv1);
        dv1 = coopMultiplyAdd(fp3, coopLoadT<coop_mat16x16<f16, B>>(&fc_tc[48u * FC_SH + cq1], FC_SH), dv1);
        dv2 = coopMultiplyAdd(fp0, coopLoadT<coop_mat16x16<f16, B>>(&fc_tc[0u * FC_SH + cq2], FC_SH), dv2); //@64
        dv2 = coopMultiplyAdd(fp1, coopLoadT<coop_mat16x16<f16, B>>(&fc_tc[16u * FC_SH + cq2], FC_SH), dv2); //@64
        dv2 = coopMultiplyAdd(fp2, coopLoadT<coop_mat16x16<f16, B>>(&fc_tc[32u * FC_SH + cq2], FC_SH), dv2); //@64
        dv2 = coopMultiplyAdd(fp3, coopLoadT<coop_mat16x16<f16, B>>(&fc_tc[48u * FC_SH + cq2], FC_SH), dv2); //@64
        dv3 = coopMultiplyAdd(fp0, coopLoadT<coop_mat16x16<f16, B>>(&fc_tc[0u * FC_SH + cq3], FC_SH), dv3); //@64
        dv3 = coopMultiplyAdd(fp1, coopLoadT<coop_mat16x16<f16, B>>(&fc_tc[16u * FC_SH + cq3], FC_SH), dv3); //@64
        dv3 = coopMultiplyAdd(fp2, coopLoadT<coop_mat16x16<f16, B>>(&fc_tc[32u * FC_SH + cq3], FC_SH), dv3); //@64
        dv3 = coopMultiplyAdd(fp3, coopLoadT<coop_mat16x16<f16, B>>(&fc_tc[48u * FC_SH + cq3], FC_SH), dv3); //@64
        // dP^T = V dO^T
        for (var jj = 0u; jj < FC_NJ; jj++) {
            let j = j0 + jj;
            var d = coopLoadT<coop_mat16x16<f32, C>>(&fc_zs[0], 16u);
            d = coopMultiplyAdd(fv0, coopLoad<coop_mat16x16<f16, B>>(&fc_tc[(j * 16u) * FC_SH + 0u], FC_SH), d);
            d = coopMultiplyAdd(fv1, coopLoad<coop_mat16x16<f16, B>>(&fc_tc[(j * 16u) * FC_SH + 16u], FC_SH), d);
            d = coopMultiplyAdd(fv2, coopLoad<coop_mat16x16<f16, B>>(&fc_tc[(j * 16u) * FC_SH + 32u], FC_SH), d);
            d = coopMultiplyAdd(fv3, coopLoad<coop_mat16x16<f16, B>>(&fc_tc[(j * 16u) * FC_SH + 48u], FC_SH), d);
            coopStoreT(d, &fc_ss[(rb * 16u) * FC_SF + j * 16u], FC_SF);
        }
        workgroupBarrier();
        // dS^T = P^T (dP^T - Delta) scale; the dV loads of ta above are done.
        for (var e = 0u; e < FC_NC; e++) {
            let c = FC_LPR * e + q;
            let ds = pr[e] * (fc_ss[rl * FC_SF + c] - fc_sd[c]) * scale;
            if (!(abs(ds) <= FC_F16_MAX)) { overflow = true; }
            fc_ta[rl * FC_SH + c] = f16(clamp(ds, -FC_F16_MAX, FC_F16_MAX));
        }
        workgroupBarrier();
        // dK += dS^T Q
        let fd0 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(rb * 16u) * FC_SH + 0u], FC_SH);
        let fd1 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(rb * 16u) * FC_SH + 16u], FC_SH);
        let fd2 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(rb * 16u) * FC_SH + 32u], FC_SH);
        let fd3 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(rb * 16u) * FC_SH + 48u], FC_SH);
        dk0 = coopMultiplyAdd(fd0, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[0u * FC_SH + cq0], FC_SH), dk0);
        dk0 = coopMultiplyAdd(fd1, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[16u * FC_SH + cq0], FC_SH), dk0);
        dk0 = coopMultiplyAdd(fd2, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[32u * FC_SH + cq0], FC_SH), dk0);
        dk0 = coopMultiplyAdd(fd3, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[48u * FC_SH + cq0], FC_SH), dk0);
        dk1 = coopMultiplyAdd(fd0, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[0u * FC_SH + cq1], FC_SH), dk1);
        dk1 = coopMultiplyAdd(fd1, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[16u * FC_SH + cq1], FC_SH), dk1);
        dk1 = coopMultiplyAdd(fd2, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[32u * FC_SH + cq1], FC_SH), dk1);
        dk1 = coopMultiplyAdd(fd3, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[48u * FC_SH + cq1], FC_SH), dk1);
        dk2 = coopMultiplyAdd(fd0, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[0u * FC_SH + cq2], FC_SH), dk2); //@64
        dk2 = coopMultiplyAdd(fd1, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[16u * FC_SH + cq2], FC_SH), dk2); //@64
        dk2 = coopMultiplyAdd(fd2, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[32u * FC_SH + cq2], FC_SH), dk2); //@64
        dk2 = coopMultiplyAdd(fd3, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[48u * FC_SH + cq2], FC_SH), dk2); //@64
        dk3 = coopMultiplyAdd(fd0, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[0u * FC_SH + cq3], FC_SH), dk3); //@64
        dk3 = coopMultiplyAdd(fd1, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[16u * FC_SH + cq3], FC_SH), dk3); //@64
        dk3 = coopMultiplyAdd(fd2, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[32u * FC_SH + cq3], FC_SH), dk3); //@64
        dk3 = coopMultiplyAdd(fd3, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[48u * FC_SH + cq3], FC_SH), dk3); //@64
        workgroupBarrier();
    }

    var dk_off = ((b * p.seq_k + kr) * p.num_heads + h) * 64u;
    var dv_off = dk_off;
    if (p.dkv_direct == 1u) {
        dk_off = ((b * p.seq_k + kr) * p.num_kv_heads + kvh) * 64u;
        dv_off = dk_off;
        if (strided) {
            dk_off = (b * p.seq_k + kr) * p.out_ld + p.k_off + kvh * 64u;
            dv_off = (b * p.seq_k + kr) * p.out_ld + p.v_off + kvh * 64u;
        }
    }
    coopStoreT(dk0, &fc_ss[(rb * 16u) * FC_SF + cq0], FC_SF);
    coopStoreT(dk1, &fc_ss[(rb * 16u) * FC_SF + cq1], FC_SF);
    coopStoreT(dk2, &fc_ss[(rb * 16u) * FC_SF + cq2], FC_SF); //@64
    coopStoreT(dk3, &fc_ss[(rb * 16u) * FC_SF + cq3], FC_SF); //@64
    workgroupBarrier();
    if (kr < p.seq_k) {
        if (rope) {
            for (var e = 0u; e < FC_NC / 2u; e++) {
                let c = FC_LPR * e + q;
                let g = fc_inv_rot(kr, c, fc_ss[rl * FC_SF + c], fc_ss[rl * FC_SF + c + 32u]);
                fc_dk[dk_off + c] = g.x;
                fc_dk[dk_off + c + 32u] = g.y;
            }
        } else {
            for (var e = 0u; e < FC_NC; e++) {
                fc_dk[dk_off + FC_LPR * e + q] = fc_ss[rl * FC_SF + FC_LPR * e + q];
            }
        }
    }
    workgroupBarrier();
    coopStoreT(dv0, &fc_ss[(rb * 16u) * FC_SF + cq0], FC_SF);
    coopStoreT(dv1, &fc_ss[(rb * 16u) * FC_SF + cq1], FC_SF);
    coopStoreT(dv2, &fc_ss[(rb * 16u) * FC_SF + cq2], FC_SF); //@64
    coopStoreT(dv3, &fc_ss[(rb * 16u) * FC_SF + cq3], FC_SF); //@64
    workgroupBarrier();
    if (kr < p.seq_k) {
        for (var e = 0u; e < FC_NC; e++) {
            fc_dv[dv_off + FC_LPR * e + q] = fc_ss[rl * FC_SF + FC_LPR * e + q];
        }
    }
    storageBarrier();
    if (overflow) {
        atomicAdd(&fc_overflows2[0], 1u);
        var first = ((b * p.seq_k + k0) * p.num_heads + h) * 64u;
        if (p.dkv_direct == 1u) {
            first = ((b * p.seq_k + k0) * p.num_kv_heads + kvh) * 64u;
            if (strided) { first = (b * p.seq_k + k0) * p.out_ld + p.k_off + kvh * 64u; }
        }
        fc_dk[first] = bitcast<f32>(0x7fc00000u);
    }
}
