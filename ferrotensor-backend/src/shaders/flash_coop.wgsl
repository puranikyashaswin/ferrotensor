// ============================================================================
// FLASH ATTENTION ON COOPERATIVE MATRICES (forward, dQ, dK/dV)
// ============================================================================
// head_dim 64, f32 buffers, causal or full attention without mask, dropout
// or window; the host routes only those calls here (engine/flash_coop.rs).
// A workgroup is 4 subgroups of 32 threads. Subgroup w owns 16 rows of the
// 64-row block (query rows in fwd and dq, key rows in dk_dv), so subgroups
// share only the staged K/V (or Q/dO) tile. Tiles are 64 x 64 f16 in
// workgroup memory with a row stride of 72 halfs (144 bytes), accumulators
// are f32 16x16 fragments, and every fragment goes back through workgroup
// memory (`fc_ss`, 68 floats per row) because the scalar phases (softmax,
// dS) need element access. Those phases give 2 lanes to each fragment row:
// lane pair (2r, 2r + 1) of a subgroup owns row r, and lane hf owns columns
// 2e + hf.
//
// Q, K, V, dO, P and dS are rounded to f16 (clamped to the f16 range, and a
// workgroup that clamped anything counts itself in `fc_overflows` and writes
// NaN into its first output element, so a dynamic loss scaler backs off).
// S, dP, the softmax, Delta, O, dQ, dK and dV stay f32.
//
// Every barrier and cooperative-matrix op sits in uniform control flow: loop
// bounds are `let`s of workgroup ids and parameters, and the lane-dependent
// `if`s guard only scalar code. Every workgroup array is a multiple of 16
// bytes, which keeps all fragment loads 16-byte aligned.

enable f16;
enable wgpu_cooperative_matrix;

const FC_SH: u32 = 72u;          // f16 tile row stride in halfs
const FC_SF: u32 = 68u;          // f32 tile row stride in floats
const FC_F16_MAX: f32 = 65504.0;
const FC_NONE: f32 = -3.0e38;    // a score that is masked out
const FC_NONE_LIMIT: f32 = -1.0e37;

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
    _pad2: u32,
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
    mask_stride_b: u32,
    mask_stride_h: u32,
    mask_stride_q: u32,
    mask_stride_k: u32,
    window: u32,
    seed_hi: u32,
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

// Three f16 tiles (64 rows x 72), one f32 tile (64 x 68), the lane-pair
// exchange, the zero source for fresh accumulators, and per-row LSE and Delta.
var<workgroup> fc_ta: array<f16, 4608>;
var<workgroup> fc_tb: array<f16, 4608>;
var<workgroup> fc_tc: array<f16, 4608>;
var<workgroup> fc_ss: array<f32, 4352>;
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

// ---------------------------------------------------------------------------
// Forward. Dispatch: (1, ceil(seq_q / 64), batch * num_heads).
// Per (query block, head) the K/V blocks stream through ta..tc as
//   ta: Q, then P    tb: K    tc: V
// with the online softmax in f32 registers and O kept in f32 registers (a
// fragment cannot be scaled per row, so O = O * corr + PV is done on lanes).
// ---------------------------------------------------------------------------
@compute @workgroup_size(128)
fn flash_coop_fwd(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let p = fc_fp;
    let n_qb = (p.seq_q + 63u) / 64u;
    let qb = n_qb - 1u - wg.y; // the longest causal blocks start first
    let b = wg.z / p.num_heads;
    let h = wg.z % p.num_heads;
    let kvh = h / (p.num_heads / p.num_kv_heads);
    let warp = li / 32u;
    let lane = li % 32u;
    let hf = lane % 2u;
    let rl = warp * 16u + lane / 2u; // this lane's row of the block
    let i = qb * 64u + rl;           // its query
    let scale = p.scale;

    for (var z = li; z < 256u; z += 128u) { fc_zs[z] = 0.0; }
    var overflow = false;
    for (var e = 0u; e < 8u; e++) {
        let idx = li + e * 128u;
        let r = idx / 16u;
        let c4 = idx % 16u;
        let row = qb * 64u + r;
        var v = vec4<f32>(0.0);
        if (row < p.seq_q) { v = fc_q[((b * p.seq_q + row) * p.num_heads + h) * 16u + c4]; }
        if (fc_bad(v)) { overflow = true; }
        let hv = fc_half(v);
        let o = r * FC_SH + c4 * 4u;
        fc_ta[o] = hv.x; fc_ta[o + 1u] = hv.y; fc_ta[o + 2u] = hv.z; fc_ta[o + 3u] = hv.w;
    }
    workgroupBarrier();
    let fq0 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(warp * 16u) * FC_SH + 0u], FC_SH);
    let fq1 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(warp * 16u) * FC_SH + 16u], FC_SH);
    let fq2 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(warp * 16u) * FC_SH + 32u], FC_SH);
    let fq3 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(warp * 16u) * FC_SH + 48u], FC_SH);
    workgroupBarrier();

    var m_run = FC_NONE;
    var l_run = 0.0;
    var o_acc: array<f32, 32>;

    let n_kv_all = (p.seq_k + 63u) / 64u;
    let n_kv_causal = min(n_kv_all, (qb * 64u + 63u + p.causal_offset) / 64u + 1u);
    let n_kv = select(n_kv_all, n_kv_causal, p.causal == 1u);
    for (var t = 0u; t < n_kv; t++) {
        let k0 = t * 64u;
        for (var e = 0u; e < 8u; e++) {
            let idx = li + e * 128u;
            let r = idx / 16u;
            let c4 = idx % 16u;
            let row = k0 + r;
            var kv = vec4<f32>(0.0);
            var vv = vec4<f32>(0.0);
            if (row < p.seq_k) {
                let base = ((b * p.seq_k + row) * p.num_kv_heads + kvh) * 16u + c4;
                kv = fc_k[base];
                vv = fc_v[base];
            }
            if (fc_bad(kv) || fc_bad(vv)) { overflow = true; }
            let kh = fc_half(kv);
            let vh = fc_half(vv);
            let o = r * FC_SH + c4 * 4u;
            fc_tb[o] = kh.x; fc_tb[o + 1u] = kh.y; fc_tb[o + 2u] = kh.z; fc_tb[o + 3u] = kh.w;
            fc_tc[o] = vh.x; fc_tc[o + 1u] = vh.y; fc_tc[o + 2u] = vh.z; fc_tc[o + 3u] = vh.w;
        }
        workgroupBarrier();
        // S = Q K^T
        for (var j = 0u; j < 4u; j++) {
            var s = coopLoadT<coop_mat16x16<f32, C>>(&fc_zs[0], 16u);
            s = coopMultiplyAdd(fq0, coopLoad<coop_mat16x16<f16, B>>(&fc_tb[(j * 16u) * FC_SH + 0u], FC_SH), s);
            s = coopMultiplyAdd(fq1, coopLoad<coop_mat16x16<f16, B>>(&fc_tb[(j * 16u) * FC_SH + 16u], FC_SH), s);
            s = coopMultiplyAdd(fq2, coopLoad<coop_mat16x16<f16, B>>(&fc_tb[(j * 16u) * FC_SH + 32u], FC_SH), s);
            s = coopMultiplyAdd(fq3, coopLoad<coop_mat16x16<f16, B>>(&fc_tb[(j * 16u) * FC_SH + 48u], FC_SH), s);
            coopStoreT(s, &fc_ss[(warp * 16u) * FC_SF + j * 16u], FC_SF);
        }
        workgroupBarrier();
        // Online softmax over this lane's 32 columns of its row.
        var x: array<f32, 32>;
        var mx = FC_NONE;
        for (var e = 0u; e < 32u; e++) {
            let c = 2u * e + hf;
            var xv = fc_ss[rl * FC_SF + c] * scale;
            let kc = k0 + c;
            if (kc >= p.seq_k || (p.causal == 1u && kc > i + p.causal_offset)) { xv = FC_NONE; }
            x[e] = xv;
            mx = max(mx, xv);
        }
        fc_pm[li] = mx;
        workgroupBarrier();
        mx = max(mx, fc_pm[warp * 32u + (lane ^ 1u)]);
        let m_new = max(m_run, mx);
        let live = m_new > FC_NONE_LIMIT;
        let corr = select(1.0, exp(m_run - m_new), live);
        var psum = 0.0;
        for (var e = 0u; e < 32u; e++) {
            let pe = select(0.0, exp(x[e] - m_new), live);
            psum += pe;
            fc_ta[rl * FC_SH + 2u * e + hf] = f16(pe);
        }
        l_run = l_run * corr + psum;
        m_run = m_new;
        workgroupBarrier();
        // PV
        let fp0 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(warp * 16u) * FC_SH + 0u], FC_SH);
        let fp1 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(warp * 16u) * FC_SH + 16u], FC_SH);
        let fp2 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(warp * 16u) * FC_SH + 32u], FC_SH);
        let fp3 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(warp * 16u) * FC_SH + 48u], FC_SH);
        for (var n = 0u; n < 4u; n++) {
            var pv = coopLoadT<coop_mat16x16<f32, C>>(&fc_zs[0], 16u);
            pv = coopMultiplyAdd(fp0, coopLoadT<coop_mat16x16<f16, B>>(&fc_tc[0u * FC_SH + n * 16u], FC_SH), pv);
            pv = coopMultiplyAdd(fp1, coopLoadT<coop_mat16x16<f16, B>>(&fc_tc[16u * FC_SH + n * 16u], FC_SH), pv);
            pv = coopMultiplyAdd(fp2, coopLoadT<coop_mat16x16<f16, B>>(&fc_tc[32u * FC_SH + n * 16u], FC_SH), pv);
            pv = coopMultiplyAdd(fp3, coopLoadT<coop_mat16x16<f16, B>>(&fc_tc[48u * FC_SH + n * 16u], FC_SH), pv);
            coopStoreT(pv, &fc_ss[(warp * 16u) * FC_SF + n * 16u], FC_SF);
        }
        workgroupBarrier();
        for (var e = 0u; e < 32u; e++) {
            o_acc[e] = o_acc[e] * corr + fc_ss[rl * FC_SF + 2u * e + hf];
        }
        workgroupBarrier();
    }

    // The two lanes of a row each summed half of its columns.
    fc_pm[li] = l_run;
    workgroupBarrier();
    let l = l_run + fc_pm[warp * 32u + (lane ^ 1u)];
    if (i < p.seq_q) {
        let ob = ((b * p.seq_q + i) * p.num_heads + h) * 64u;
        for (var e = 0u; e < 32u; e++) {
            fc_out[ob + 2u * e + hf] = select(0.0, o_acc[e] / l, l > 0.0);
        }
        if (p.write_stats == 1u && hf == 0u) {
            // A row that saw no visible key keeps 3.0e38, which makes its P zero in backward.
            fc_stats[2u * ((b * p.num_heads + h) * p.seq_q + i) + 1u] = select(3.0e38, m_run + log(l), l > 0.0);
        }
    }
    // After the writes above, so the poison is the last write to that element.
    storageBarrier();
    if (overflow) {
        atomicAdd(&fc_overflows[0], 1u);
        fc_out[((b * p.seq_q + qb * 64u) * p.num_heads + h) * 64u] = bitcast<f32>(0x7fc00000u);
    }
}

// ---------------------------------------------------------------------------
// Backward, dQ. Dispatch: (1, ceil(seq_q / 64), batch * num_heads).
//   ta: Q, then dS    tb: dO, then K    tc: V
// Writes Delta = dO . O (f32, from the saved O) into stats .x, reads the
// forward pass's LSE from stats .y. dS carries the softmax scale.
// ---------------------------------------------------------------------------
@compute @workgroup_size(128)
fn flash_coop_bwd_dq(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let p = fc_bp;
    let n_qb = (p.seq_q + 63u) / 64u;
    let qb = n_qb - 1u - wg.y;
    let b = wg.z / p.num_heads;
    let h = wg.z % p.num_heads;
    let kvh = h / (p.num_heads / p.num_kv_heads);
    let warp = li / 32u;
    let lane = li % 32u;
    let hf = lane % 2u;
    let rl = warp * 16u + lane / 2u;
    let i = qb * 64u + rl;
    let scale = p.scale;
    let row_id = (b * p.num_heads + h) * p.seq_q + i;

    for (var z = li; z < 256u; z += 128u) { fc_zs[z] = 0.0; }
    var overflow = false;
    // Delta from the unrounded O and dO; the lane pair splits the row's 16 vec4s.
    var dsum = 0.0;
    if (i < p.seq_q) {
        let ob4 = ((b * p.seq_q + i) * p.num_heads + h) * 16u;
        for (var e = 0u; e < 8u; e++) {
            let c4 = hf * 8u + e;
            dsum += dot(fc_o4[ob4 + c4], fc_do[ob4 + c4]);
        }
    }
    fc_pm[li] = dsum;
    workgroupBarrier();
    let delta = dsum + fc_pm[warp * 32u + (lane ^ 1u)];
    if (hf == 0u) {
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
    for (var e = 0u; e < 8u; e++) {
        let idx = li + e * 128u;
        let r = idx / 16u;
        let c4 = idx % 16u;
        let row = qb * 64u + r;
        var qv = vec4<f32>(0.0);
        var gv = vec4<f32>(0.0);
        if (row < p.seq_q) {
            let base = ((b * p.seq_q + row) * p.num_heads + h) * 16u + c4;
            qv = fc_q[base];
            gv = fc_do[base];
        }
        if (fc_bad(qv) || fc_bad(gv)) { overflow = true; }
        let qh = fc_half(qv);
        let gh = fc_half(gv);
        let o = r * FC_SH + c4 * 4u;
        fc_ta[o] = qh.x; fc_ta[o + 1u] = qh.y; fc_ta[o + 2u] = qh.z; fc_ta[o + 3u] = qh.w;
        fc_tb[o] = gh.x; fc_tb[o + 1u] = gh.y; fc_tb[o + 2u] = gh.z; fc_tb[o + 3u] = gh.w;
    }
    workgroupBarrier();
    let fq0 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(warp * 16u) * FC_SH + 0u], FC_SH);
    let fq1 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(warp * 16u) * FC_SH + 16u], FC_SH);
    let fq2 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(warp * 16u) * FC_SH + 32u], FC_SH);
    let fq3 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(warp * 16u) * FC_SH + 48u], FC_SH);
    let fg0 = coopLoadT<coop_mat16x16<f16, A>>(&fc_tb[(warp * 16u) * FC_SH + 0u], FC_SH);
    let fg1 = coopLoadT<coop_mat16x16<f16, A>>(&fc_tb[(warp * 16u) * FC_SH + 16u], FC_SH);
    let fg2 = coopLoadT<coop_mat16x16<f16, A>>(&fc_tb[(warp * 16u) * FC_SH + 32u], FC_SH);
    let fg3 = coopLoadT<coop_mat16x16<f16, A>>(&fc_tb[(warp * 16u) * FC_SH + 48u], FC_SH);
    workgroupBarrier();

    var dq0 = coopLoadT<coop_mat16x16<f32, C>>(&fc_zs[0], 16u);
    var dq1 = coopLoadT<coop_mat16x16<f32, C>>(&fc_zs[0], 16u);
    var dq2 = coopLoadT<coop_mat16x16<f32, C>>(&fc_zs[0], 16u);
    var dq3 = coopLoadT<coop_mat16x16<f32, C>>(&fc_zs[0], 16u);
    let lse_r = fc_sl[rl];
    let del_r = fc_sd[rl];

    let n_kv_all = (p.seq_k + 63u) / 64u;
    let n_kv_causal = min(n_kv_all, (qb * 64u + 63u + p.causal_offset) / 64u + 1u);
    let n_kv = select(n_kv_all, n_kv_causal, p.causal == 1u);
    for (var t = 0u; t < n_kv; t++) {
        let k0 = t * 64u;
        for (var e = 0u; e < 8u; e++) {
            let idx = li + e * 128u;
            let r = idx / 16u;
            let c4 = idx % 16u;
            let row = k0 + r;
            var kv = vec4<f32>(0.0);
            var vv = vec4<f32>(0.0);
            if (row < p.seq_k) {
                let base = ((b * p.seq_k + row) * p.num_kv_heads + kvh) * 16u + c4;
                kv = fc_k[base];
                vv = fc_v[base];
            }
            if (fc_bad(kv) || fc_bad(vv)) { overflow = true; }
            let kh = fc_half(kv);
            let vh = fc_half(vv);
            let o = r * FC_SH + c4 * 4u;
            fc_tb[o] = kh.x; fc_tb[o + 1u] = kh.y; fc_tb[o + 2u] = kh.z; fc_tb[o + 3u] = kh.w;
            fc_tc[o] = vh.x; fc_tc[o + 1u] = vh.y; fc_tc[o + 2u] = vh.z; fc_tc[o + 3u] = vh.w;
        }
        workgroupBarrier();
        // S = Q K^T
        for (var j = 0u; j < 4u; j++) {
            var s = coopLoadT<coop_mat16x16<f32, C>>(&fc_zs[0], 16u);
            s = coopMultiplyAdd(fq0, coopLoad<coop_mat16x16<f16, B>>(&fc_tb[(j * 16u) * FC_SH + 0u], FC_SH), s);
            s = coopMultiplyAdd(fq1, coopLoad<coop_mat16x16<f16, B>>(&fc_tb[(j * 16u) * FC_SH + 16u], FC_SH), s);
            s = coopMultiplyAdd(fq2, coopLoad<coop_mat16x16<f16, B>>(&fc_tb[(j * 16u) * FC_SH + 32u], FC_SH), s);
            s = coopMultiplyAdd(fq3, coopLoad<coop_mat16x16<f16, B>>(&fc_tb[(j * 16u) * FC_SH + 48u], FC_SH), s);
            coopStoreT(s, &fc_ss[(warp * 16u) * FC_SF + j * 16u], FC_SF);
        }
        workgroupBarrier();
        var pr: array<f32, 32>;
        for (var e = 0u; e < 32u; e++) {
            let c = 2u * e + hf;
            let kc = k0 + c;
            var pe = exp(fc_ss[rl * FC_SF + c] * scale - lse_r);
            if (i >= p.seq_q || kc >= p.seq_k || (p.causal == 1u && kc > i + p.causal_offset)) { pe = 0.0; }
            pr[e] = pe;
        }
        workgroupBarrier();
        // dP = dO V^T
        for (var j = 0u; j < 4u; j++) {
            var d = coopLoadT<coop_mat16x16<f32, C>>(&fc_zs[0], 16u);
            d = coopMultiplyAdd(fg0, coopLoad<coop_mat16x16<f16, B>>(&fc_tc[(j * 16u) * FC_SH + 0u], FC_SH), d);
            d = coopMultiplyAdd(fg1, coopLoad<coop_mat16x16<f16, B>>(&fc_tc[(j * 16u) * FC_SH + 16u], FC_SH), d);
            d = coopMultiplyAdd(fg2, coopLoad<coop_mat16x16<f16, B>>(&fc_tc[(j * 16u) * FC_SH + 32u], FC_SH), d);
            d = coopMultiplyAdd(fg3, coopLoad<coop_mat16x16<f16, B>>(&fc_tc[(j * 16u) * FC_SH + 48u], FC_SH), d);
            coopStoreT(d, &fc_ss[(warp * 16u) * FC_SF + j * 16u], FC_SF);
        }
        workgroupBarrier();
        // dS = P (dP - Delta) scale, as f16 for the dQ product.
        for (var e = 0u; e < 32u; e++) {
            let c = 2u * e + hf;
            let ds = pr[e] * (fc_ss[rl * FC_SF + c] - del_r) * scale;
            if (!(abs(ds) <= FC_F16_MAX)) { overflow = true; }
            fc_ta[rl * FC_SH + c] = f16(clamp(ds, -FC_F16_MAX, FC_F16_MAX));
        }
        workgroupBarrier();
        // dQ += dS K
        let fd0 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(warp * 16u) * FC_SH + 0u], FC_SH);
        let fd1 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(warp * 16u) * FC_SH + 16u], FC_SH);
        let fd2 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(warp * 16u) * FC_SH + 32u], FC_SH);
        let fd3 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(warp * 16u) * FC_SH + 48u], FC_SH);
        dq0 = coopMultiplyAdd(fd0, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[0u * FC_SH + 0u], FC_SH), dq0);
        dq0 = coopMultiplyAdd(fd1, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[16u * FC_SH + 0u], FC_SH), dq0);
        dq0 = coopMultiplyAdd(fd2, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[32u * FC_SH + 0u], FC_SH), dq0);
        dq0 = coopMultiplyAdd(fd3, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[48u * FC_SH + 0u], FC_SH), dq0);
        dq1 = coopMultiplyAdd(fd0, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[0u * FC_SH + 16u], FC_SH), dq1);
        dq1 = coopMultiplyAdd(fd1, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[16u * FC_SH + 16u], FC_SH), dq1);
        dq1 = coopMultiplyAdd(fd2, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[32u * FC_SH + 16u], FC_SH), dq1);
        dq1 = coopMultiplyAdd(fd3, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[48u * FC_SH + 16u], FC_SH), dq1);
        dq2 = coopMultiplyAdd(fd0, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[0u * FC_SH + 32u], FC_SH), dq2);
        dq2 = coopMultiplyAdd(fd1, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[16u * FC_SH + 32u], FC_SH), dq2);
        dq2 = coopMultiplyAdd(fd2, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[32u * FC_SH + 32u], FC_SH), dq2);
        dq2 = coopMultiplyAdd(fd3, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[48u * FC_SH + 32u], FC_SH), dq2);
        dq3 = coopMultiplyAdd(fd0, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[0u * FC_SH + 48u], FC_SH), dq3);
        dq3 = coopMultiplyAdd(fd1, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[16u * FC_SH + 48u], FC_SH), dq3);
        dq3 = coopMultiplyAdd(fd2, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[32u * FC_SH + 48u], FC_SH), dq3);
        dq3 = coopMultiplyAdd(fd3, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[48u * FC_SH + 48u], FC_SH), dq3);
        workgroupBarrier();
    }

    coopStoreT(dq0, &fc_ss[(warp * 16u) * FC_SF + 0u], FC_SF);
    coopStoreT(dq1, &fc_ss[(warp * 16u) * FC_SF + 16u], FC_SF);
    coopStoreT(dq2, &fc_ss[(warp * 16u) * FC_SF + 32u], FC_SF);
    coopStoreT(dq3, &fc_ss[(warp * 16u) * FC_SF + 48u], FC_SF);
    workgroupBarrier();
    if (i < p.seq_q) {
        let ob = ((b * p.seq_q + i) * p.num_heads + h) * 64u;
        for (var e = 0u; e < 32u; e++) {
            fc_dq[ob + 2u * e + hf] = fc_ss[rl * FC_SF + 2u * e + hf];
        }
    }
    storageBarrier();
    if (overflow) {
        atomicAdd(&fc_overflows2[0], 1u);
        fc_dq[((b * p.seq_q + qb * 64u) * p.num_heads + h) * 64u] = bitcast<f32>(0x7fc00000u);
    }
}

// ---------------------------------------------------------------------------
// Backward, dK and dV. Dispatch: (1, ceil(seq_k / 64), batch * num_heads):
// one workgroup per (key block, query head), so a group of query heads that
// share a KV head runs in parallel. With dkv_direct (one query head per KV
// head) the result goes straight to dK/dV [batch, seq_k, kv heads, 64];
// otherwise to per-query-head partials [batch, seq_k, heads, 64] that
// flash_group_sum adds up.
//   ta: K, then P^T, then dS^T    tb: V, then Q    tc: dO
// Warp w owns key rows w*16..w*16+15 and the columns are queries.
// ---------------------------------------------------------------------------
@compute @workgroup_size(128)
fn flash_coop_bwd_dkdv(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let p = fc_bp;
    let kb = wg.y; // the blocks with the most query tiles start first
    let b = wg.z / p.num_heads;
    let h = wg.z % p.num_heads;
    let kvh = h / (p.num_heads / p.num_kv_heads);
    let warp = li / 32u;
    let lane = li % 32u;
    let hf = lane % 2u;
    let rl = warp * 16u + lane / 2u;
    let k0 = kb * 64u;
    let kr = k0 + rl; // this lane's key
    let scale = p.scale;

    for (var z = li; z < 256u; z += 128u) { fc_zs[z] = 0.0; }
    var overflow = false;
    for (var e = 0u; e < 8u; e++) {
        let idx = li + e * 128u;
        let r = idx / 16u;
        let c4 = idx % 16u;
        let row = k0 + r;
        var kv = vec4<f32>(0.0);
        var vv = vec4<f32>(0.0);
        if (row < p.seq_k) {
            let base = ((b * p.seq_k + row) * p.num_kv_heads + kvh) * 16u + c4;
            kv = fc_k[base];
            vv = fc_v[base];
        }
        if (fc_bad(kv) || fc_bad(vv)) { overflow = true; }
        let kh = fc_half(kv);
        let vh = fc_half(vv);
        let o = r * FC_SH + c4 * 4u;
        fc_ta[o] = kh.x; fc_ta[o + 1u] = kh.y; fc_ta[o + 2u] = kh.z; fc_ta[o + 3u] = kh.w;
        fc_tb[o] = vh.x; fc_tb[o + 1u] = vh.y; fc_tb[o + 2u] = vh.z; fc_tb[o + 3u] = vh.w;
    }
    workgroupBarrier();
    let fk0 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(warp * 16u) * FC_SH + 0u], FC_SH);
    let fk1 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(warp * 16u) * FC_SH + 16u], FC_SH);
    let fk2 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(warp * 16u) * FC_SH + 32u], FC_SH);
    let fk3 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(warp * 16u) * FC_SH + 48u], FC_SH);
    let fv0 = coopLoadT<coop_mat16x16<f16, A>>(&fc_tb[(warp * 16u) * FC_SH + 0u], FC_SH);
    let fv1 = coopLoadT<coop_mat16x16<f16, A>>(&fc_tb[(warp * 16u) * FC_SH + 16u], FC_SH);
    let fv2 = coopLoadT<coop_mat16x16<f16, A>>(&fc_tb[(warp * 16u) * FC_SH + 32u], FC_SH);
    let fv3 = coopLoadT<coop_mat16x16<f16, A>>(&fc_tb[(warp * 16u) * FC_SH + 48u], FC_SH);
    workgroupBarrier();

    var dk0 = coopLoadT<coop_mat16x16<f32, C>>(&fc_zs[0], 16u);
    var dk1 = coopLoadT<coop_mat16x16<f32, C>>(&fc_zs[0], 16u);
    var dk2 = coopLoadT<coop_mat16x16<f32, C>>(&fc_zs[0], 16u);
    var dk3 = coopLoadT<coop_mat16x16<f32, C>>(&fc_zs[0], 16u);
    var dv0 = coopLoadT<coop_mat16x16<f32, C>>(&fc_zs[0], 16u);
    var dv1 = coopLoadT<coop_mat16x16<f32, C>>(&fc_zs[0], 16u);
    var dv2 = coopLoadT<coop_mat16x16<f32, C>>(&fc_zs[0], 16u);
    var dv3 = coopLoadT<coop_mat16x16<f32, C>>(&fc_zs[0], 16u);

    // Query tiles that can see this key block: from the one holding the first query that sees its first key when causal.
    let n_qt = (p.seq_q + 63u) / 64u;
    let qt0 = select(0u, (k0 - min(k0, p.causal_offset)) / 64u, p.causal == 1u);
    for (var qt = qt0; qt < n_qt; qt++) {
        let q0 = qt * 64u;
        for (var e = 0u; e < 8u; e++) {
            let idx = li + e * 128u;
            let r = idx / 16u;
            let c4 = idx % 16u;
            let row = q0 + r;
            var qv = vec4<f32>(0.0);
            var gv = vec4<f32>(0.0);
            if (row < p.seq_q) {
                let base = ((b * p.seq_q + row) * p.num_heads + h) * 16u + c4;
                qv = fc_q[base];
                gv = fc_do[base];
            }
            if (fc_bad(qv) || fc_bad(gv)) { overflow = true; }
            let qh = fc_half(qv);
            let gh = fc_half(gv);
            let o = r * FC_SH + c4 * 4u;
            fc_tb[o] = qh.x; fc_tb[o + 1u] = qh.y; fc_tb[o + 2u] = qh.z; fc_tb[o + 3u] = qh.w;
            fc_tc[o] = gh.x; fc_tc[o + 1u] = gh.y; fc_tc[o + 2u] = gh.z; fc_tc[o + 3u] = gh.w;
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
        for (var j = 0u; j < 4u; j++) {
            var s = coopLoadT<coop_mat16x16<f32, C>>(&fc_zs[0], 16u);
            s = coopMultiplyAdd(fk0, coopLoad<coop_mat16x16<f16, B>>(&fc_tb[(j * 16u) * FC_SH + 0u], FC_SH), s);
            s = coopMultiplyAdd(fk1, coopLoad<coop_mat16x16<f16, B>>(&fc_tb[(j * 16u) * FC_SH + 16u], FC_SH), s);
            s = coopMultiplyAdd(fk2, coopLoad<coop_mat16x16<f16, B>>(&fc_tb[(j * 16u) * FC_SH + 32u], FC_SH), s);
            s = coopMultiplyAdd(fk3, coopLoad<coop_mat16x16<f16, B>>(&fc_tb[(j * 16u) * FC_SH + 48u], FC_SH), s);
            coopStoreT(s, &fc_ss[(warp * 16u) * FC_SF + j * 16u], FC_SF);
        }
        workgroupBarrier();
        var pr: array<f32, 32>;
        for (var e = 0u; e < 32u; e++) {
            let c = 2u * e + hf;
            let qc = q0 + c;
            var pe = exp(fc_ss[rl * FC_SF + c] * scale - fc_sl[c]);
            if (qc >= p.seq_q || kr >= p.seq_k || (p.causal == 1u && kr > qc + p.causal_offset)) { pe = 0.0; }
            pr[e] = pe;
            fc_ta[rl * FC_SH + c] = f16(pe);
        }
        workgroupBarrier();
        // dV += P^T dO
        let fp0 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(warp * 16u) * FC_SH + 0u], FC_SH);
        let fp1 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(warp * 16u) * FC_SH + 16u], FC_SH);
        let fp2 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(warp * 16u) * FC_SH + 32u], FC_SH);
        let fp3 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(warp * 16u) * FC_SH + 48u], FC_SH);
        dv0 = coopMultiplyAdd(fp0, coopLoadT<coop_mat16x16<f16, B>>(&fc_tc[0u * FC_SH + 0u], FC_SH), dv0);
        dv0 = coopMultiplyAdd(fp1, coopLoadT<coop_mat16x16<f16, B>>(&fc_tc[16u * FC_SH + 0u], FC_SH), dv0);
        dv0 = coopMultiplyAdd(fp2, coopLoadT<coop_mat16x16<f16, B>>(&fc_tc[32u * FC_SH + 0u], FC_SH), dv0);
        dv0 = coopMultiplyAdd(fp3, coopLoadT<coop_mat16x16<f16, B>>(&fc_tc[48u * FC_SH + 0u], FC_SH), dv0);
        dv1 = coopMultiplyAdd(fp0, coopLoadT<coop_mat16x16<f16, B>>(&fc_tc[0u * FC_SH + 16u], FC_SH), dv1);
        dv1 = coopMultiplyAdd(fp1, coopLoadT<coop_mat16x16<f16, B>>(&fc_tc[16u * FC_SH + 16u], FC_SH), dv1);
        dv1 = coopMultiplyAdd(fp2, coopLoadT<coop_mat16x16<f16, B>>(&fc_tc[32u * FC_SH + 16u], FC_SH), dv1);
        dv1 = coopMultiplyAdd(fp3, coopLoadT<coop_mat16x16<f16, B>>(&fc_tc[48u * FC_SH + 16u], FC_SH), dv1);
        dv2 = coopMultiplyAdd(fp0, coopLoadT<coop_mat16x16<f16, B>>(&fc_tc[0u * FC_SH + 32u], FC_SH), dv2);
        dv2 = coopMultiplyAdd(fp1, coopLoadT<coop_mat16x16<f16, B>>(&fc_tc[16u * FC_SH + 32u], FC_SH), dv2);
        dv2 = coopMultiplyAdd(fp2, coopLoadT<coop_mat16x16<f16, B>>(&fc_tc[32u * FC_SH + 32u], FC_SH), dv2);
        dv2 = coopMultiplyAdd(fp3, coopLoadT<coop_mat16x16<f16, B>>(&fc_tc[48u * FC_SH + 32u], FC_SH), dv2);
        dv3 = coopMultiplyAdd(fp0, coopLoadT<coop_mat16x16<f16, B>>(&fc_tc[0u * FC_SH + 48u], FC_SH), dv3);
        dv3 = coopMultiplyAdd(fp1, coopLoadT<coop_mat16x16<f16, B>>(&fc_tc[16u * FC_SH + 48u], FC_SH), dv3);
        dv3 = coopMultiplyAdd(fp2, coopLoadT<coop_mat16x16<f16, B>>(&fc_tc[32u * FC_SH + 48u], FC_SH), dv3);
        dv3 = coopMultiplyAdd(fp3, coopLoadT<coop_mat16x16<f16, B>>(&fc_tc[48u * FC_SH + 48u], FC_SH), dv3);
        // dP^T = V dO^T
        for (var j = 0u; j < 4u; j++) {
            var d = coopLoadT<coop_mat16x16<f32, C>>(&fc_zs[0], 16u);
            d = coopMultiplyAdd(fv0, coopLoad<coop_mat16x16<f16, B>>(&fc_tc[(j * 16u) * FC_SH + 0u], FC_SH), d);
            d = coopMultiplyAdd(fv1, coopLoad<coop_mat16x16<f16, B>>(&fc_tc[(j * 16u) * FC_SH + 16u], FC_SH), d);
            d = coopMultiplyAdd(fv2, coopLoad<coop_mat16x16<f16, B>>(&fc_tc[(j * 16u) * FC_SH + 32u], FC_SH), d);
            d = coopMultiplyAdd(fv3, coopLoad<coop_mat16x16<f16, B>>(&fc_tc[(j * 16u) * FC_SH + 48u], FC_SH), d);
            coopStoreT(d, &fc_ss[(warp * 16u) * FC_SF + j * 16u], FC_SF);
        }
        workgroupBarrier();
        // dS^T = P^T (dP^T - Delta) scale; the dV loads of ta above are done.
        for (var e = 0u; e < 32u; e++) {
            let c = 2u * e + hf;
            let ds = pr[e] * (fc_ss[rl * FC_SF + c] - fc_sd[c]) * scale;
            if (!(abs(ds) <= FC_F16_MAX)) { overflow = true; }
            fc_ta[rl * FC_SH + c] = f16(clamp(ds, -FC_F16_MAX, FC_F16_MAX));
        }
        workgroupBarrier();
        // dK += dS^T Q
        let fd0 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(warp * 16u) * FC_SH + 0u], FC_SH);
        let fd1 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(warp * 16u) * FC_SH + 16u], FC_SH);
        let fd2 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(warp * 16u) * FC_SH + 32u], FC_SH);
        let fd3 = coopLoadT<coop_mat16x16<f16, A>>(&fc_ta[(warp * 16u) * FC_SH + 48u], FC_SH);
        dk0 = coopMultiplyAdd(fd0, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[0u * FC_SH + 0u], FC_SH), dk0);
        dk0 = coopMultiplyAdd(fd1, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[16u * FC_SH + 0u], FC_SH), dk0);
        dk0 = coopMultiplyAdd(fd2, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[32u * FC_SH + 0u], FC_SH), dk0);
        dk0 = coopMultiplyAdd(fd3, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[48u * FC_SH + 0u], FC_SH), dk0);
        dk1 = coopMultiplyAdd(fd0, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[0u * FC_SH + 16u], FC_SH), dk1);
        dk1 = coopMultiplyAdd(fd1, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[16u * FC_SH + 16u], FC_SH), dk1);
        dk1 = coopMultiplyAdd(fd2, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[32u * FC_SH + 16u], FC_SH), dk1);
        dk1 = coopMultiplyAdd(fd3, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[48u * FC_SH + 16u], FC_SH), dk1);
        dk2 = coopMultiplyAdd(fd0, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[0u * FC_SH + 32u], FC_SH), dk2);
        dk2 = coopMultiplyAdd(fd1, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[16u * FC_SH + 32u], FC_SH), dk2);
        dk2 = coopMultiplyAdd(fd2, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[32u * FC_SH + 32u], FC_SH), dk2);
        dk2 = coopMultiplyAdd(fd3, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[48u * FC_SH + 32u], FC_SH), dk2);
        dk3 = coopMultiplyAdd(fd0, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[0u * FC_SH + 48u], FC_SH), dk3);
        dk3 = coopMultiplyAdd(fd1, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[16u * FC_SH + 48u], FC_SH), dk3);
        dk3 = coopMultiplyAdd(fd2, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[32u * FC_SH + 48u], FC_SH), dk3);
        dk3 = coopMultiplyAdd(fd3, coopLoadT<coop_mat16x16<f16, B>>(&fc_tb[48u * FC_SH + 48u], FC_SH), dk3);
        workgroupBarrier();
    }

    var out_off = ((b * p.seq_k + kr) * p.num_heads + h) * 64u;
    if (p.dkv_direct == 1u) { out_off = ((b * p.seq_k + kr) * p.num_kv_heads + kvh) * 64u; }
    coopStoreT(dk0, &fc_ss[(warp * 16u) * FC_SF + 0u], FC_SF);
    coopStoreT(dk1, &fc_ss[(warp * 16u) * FC_SF + 16u], FC_SF);
    coopStoreT(dk2, &fc_ss[(warp * 16u) * FC_SF + 32u], FC_SF);
    coopStoreT(dk3, &fc_ss[(warp * 16u) * FC_SF + 48u], FC_SF);
    workgroupBarrier();
    if (kr < p.seq_k) {
        for (var e = 0u; e < 32u; e++) {
            fc_dk[out_off + 2u * e + hf] = fc_ss[rl * FC_SF + 2u * e + hf];
        }
    }
    workgroupBarrier();
    coopStoreT(dv0, &fc_ss[(warp * 16u) * FC_SF + 0u], FC_SF);
    coopStoreT(dv1, &fc_ss[(warp * 16u) * FC_SF + 16u], FC_SF);
    coopStoreT(dv2, &fc_ss[(warp * 16u) * FC_SF + 32u], FC_SF);
    coopStoreT(dv3, &fc_ss[(warp * 16u) * FC_SF + 48u], FC_SF);
    workgroupBarrier();
    if (kr < p.seq_k) {
        for (var e = 0u; e < 32u; e++) {
            fc_dv[out_off + 2u * e + hf] = fc_ss[rl * FC_SF + 2u * e + hf];
        }
    }
    storageBarrier();
    if (overflow) {
        atomicAdd(&fc_overflows2[0], 1u);
        var first = ((b * p.seq_k + k0) * p.num_heads + h) * 64u;
        if (p.dkv_direct == 1u) { first = ((b * p.seq_k + k0) * p.num_kv_heads + kvh) * 64u; }
        fc_dk[first] = bitcast<f32>(0x7fc00000u);
    }
}
