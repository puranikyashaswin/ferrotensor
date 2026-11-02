//! GPU Compute Kernels (WGSL)
//!
//! The `ferrotensor` tensor API dispatches a small set of generic kernels. Each one
//! selects its operation through an op code in a uniform, so adding an
//! element-wise op means adding one `case` here and one entry in
//! `tensor::cpu`, which holds the reference implementation the GPU result is
//! tested against.
//!
//! 1-D kernels are dispatched on a 2-D grid (`x + y * nx * 256`) so tensors
//! larger than 65535 workgroups still fit in one dispatch.

/// Op codes shared by the `unary` kernel and the CPU reference implementation.
pub(crate) mod unary_op {
    pub const RELU: u32 = 0;
    pub const SILU: u32 = 1;
    pub const GELU: u32 = 2;
    pub const SIGMOID: u32 = 3;
    pub const TANH: u32 = 4;
    pub const EXP: u32 = 5;
    pub const LOG: u32 = 6;
    pub const ABS: u32 = 7;
    pub const SQRT: u32 = 8;
    pub const NEG: u32 = 9;
    pub const ADD_SCALAR: u32 = 10;
    pub const MUL_SCALAR: u32 = 11;
    pub const POW_SCALAR: u32 = 12;
    pub const CLAMP: u32 = 13;
    pub const LEAKY_RELU: u32 = 14;
    pub const STEP: u32 = 15;
    pub const SIGN: u32 = 16;
    pub const RECIP: u32 = 17;
    pub const SILU_GRAD: u32 = 18;
    pub const GELU_GRAD: u32 = 19;
    pub const RSUB_SCALAR: u32 = 20;
    pub const SQUARE: u32 = 21;
    pub const IN_RANGE: u32 = 22;
    pub const SIN: u32 = 23;
    pub const COS: u32 = 24;
    pub const LEAKY_RELU_GRAD: u32 = 25;
}

/// Op codes for the broadcasting `binary` kernel.
pub(crate) mod binary_op {
    pub const ADD: u32 = 0;
    pub const SUB: u32 = 1;
    pub const MUL: u32 = 2;
    pub const DIV: u32 = 3;
    pub const MAX: u32 = 4;
    pub const MIN: u32 = 5;
    pub const GT: u32 = 6;
    pub const LT: u32 = 7;
    pub const EQ: u32 = 8;
    pub const GE: u32 = 9;
    pub const LE: u32 = 10;
}

/// Op codes for the axis `reduce` kernel.
pub(crate) mod reduce_op {
    pub const SUM: u32 = 0;
    pub const MAX: u32 = 1;
    pub const MIN: u32 = 2;
}

/// WGSL source for every kernel in [`ComputeEngine`](super::ComputeEngine).
pub(crate) const KERNELS_WGSL: &str = r#"
// ============================================================================
// FERROTENSOR TENSOR KERNELS
// ============================================================================

fn flat_index(gid: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return gid.x + gid.y * nwg.x * 256u;
}

fn gelu_tanh(x: f32) -> f32 {
    let c = 0.7978845608;
    return 0.5 * x * (1.0 + tanh(c * (x + 0.044715 * x * x * x)));
}

fn gelu_tanh_grad(x: f32) -> f32 {
    let c = 0.7978845608;
    let t = tanh(c * (x + 0.044715 * x * x * x));
    let dt = (1.0 - t * t) * c * (1.0 + 3.0 * 0.044715 * x * x);
    return 0.5 * (1.0 + t) + 0.5 * x * dt;
}

fn sigmoid_f(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

// --- Element-wise unary ops -------------------------------------------------

struct UnaryParams {
    op: u32,
    n: u32,
    a: f32,
    b: f32,
};

@group(0) @binding(0) var<storage, read> un_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> un_out: array<f32>;
@group(0) @binding(2) var<uniform> un_p: UnaryParams;

fn unary_apply(op: u32, x: f32, a: f32, b: f32) -> f32 {
    var y = x;
    switch op {
        case 0u: { y = max(x, 0.0); }
        case 1u: { y = x * sigmoid_f(x); }
        case 2u: { y = gelu_tanh(x); }
        case 3u: { y = sigmoid_f(x); }
        case 4u: { y = tanh(x); }
        case 5u: { y = exp(x); }
        case 6u: { y = log(x); }
        case 7u: { y = abs(x); }
        case 8u: { y = sqrt(x); }
        case 9u: { y = -x; }
        case 10u: { y = x + a; }
        case 11u: { y = x * a; }
        case 12u: {
            // pow is undefined for negative bases in WGSL; handle integer exponents.
            let ai = round(a);
            if (x < 0.0 && ai == a) {
                let r = pow(-x, a);
                y = select(r, -r, (i32(ai) & 1) == 1);
            } else if (x == 0.0 && a > 0.0) {
                y = 0.0;
            } else {
                y = pow(x, a);
            }
        }
        case 13u: { y = clamp(x, a, b); }
        case 14u: { y = select(a * x, x, x > 0.0); }
        case 15u: { y = select(0.0, 1.0, x > 0.0); }
        case 16u: { y = sign(x); }
        case 17u: { y = 1.0 / x; }
        case 18u: {
            let s = sigmoid_f(x);
            y = s * (1.0 + x * (1.0 - s));
        }
        case 19u: { y = gelu_tanh_grad(x); }
        case 20u: { y = a - x; }
        case 21u: { y = x * x; }
        case 22u: { y = select(0.0, 1.0, x >= a && x <= b); }
        case 23u: { y = sin(x); }
        case 24u: { y = cos(x); }
        case 25u: { y = select(a, 1.0, x > 0.0); }
        default: { y = x; }
    }
    return y;
}

@compute @workgroup_size(256)
fn unary(@builtin(global_invocation_id) gid: vec3<u32>,
         @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = flat_index(gid, nwg);
    if (i >= un_p.n) { return; }
    un_out[i] = unary_apply(un_p.op, un_in[i], un_p.a, un_p.b);
}

// --- Broadcasting binary ops ------------------------------------------------
// meta = [out_shape(ndim), a_strides(ndim), b_strides(ndim)]; a broadcast
// dimension has stride 0.

struct BinaryParams {
    op: u32,
    n: u32,
    ndim: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> bin_a: array<f32>;
@group(0) @binding(1) var<storage, read> bin_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> bin_out: array<f32>;
@group(0) @binding(3) var<uniform> bin_p: BinaryParams;
@group(0) @binding(4) var<storage, read> bin_meta: array<u32>;

fn binary_apply(op: u32, x: f32, y: f32) -> f32 {
    var r = 0.0;
    switch op {
        case 0u: { r = x + y; }
        case 1u: { r = x - y; }
        case 2u: { r = x * y; }
        case 3u: { r = x / y; }
        case 4u: { r = max(x, y); }
        case 5u: { r = min(x, y); }
        case 6u: { r = select(0.0, 1.0, x > y); }
        case 7u: { r = select(0.0, 1.0, x < y); }
        case 8u: { r = select(0.0, 1.0, x == y); }
        case 9u: { r = select(0.0, 1.0, x >= y); }
        case 10u: { r = select(0.0, 1.0, x <= y); }
        default: { r = x; }
    }
    return r;
}

@compute @workgroup_size(256)
fn binary(@builtin(global_invocation_id) gid: vec3<u32>,
          @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = flat_index(gid, nwg);
    if (i >= bin_p.n) { return; }
    let nd = bin_p.ndim;
    var rem = i;
    var ia = 0u;
    var ib = 0u;
    for (var d = 0u; d < nd; d = d + 1u) {
        let axis = nd - 1u - d;
        let size = bin_meta[axis];
        let idx = rem % size;
        rem = rem / size;
        ia = ia + idx * bin_meta[nd + axis];
        ib = ib + idx * bin_meta[2u * nd + axis];
    }
    bin_out[i] = binary_apply(bin_p.op, bin_a[ia], bin_b[ib]);
}

// --- Fused element-wise chains ---------------------------------------------
// A chain of same-size unary and non-broadcasting binary ops in one dispatch
// (FERROTENSOR_FUSION=1, see docs/design/op-fusion.md). Each step is 8 words:
// [kind, op, src0, src1, dst, a bits, b bits, 0]; kind 0 applies a unary op
// to src0, kind 1 a binary op to src0 and src1. Sources and destinations are
// binding slots 0..6. Every step stores its result and later steps load it
// back, as separate dispatches would.

struct FusedParams {
    n: u32,
    steps: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read_write> fb0: array<f32>;
@group(0) @binding(1) var<storage, read_write> fb1: array<f32>;
@group(0) @binding(2) var<storage, read_write> fb2: array<f32>;
@group(0) @binding(3) var<storage, read_write> fb3: array<f32>;
@group(0) @binding(4) var<storage, read_write> fb4: array<f32>;
@group(0) @binding(5) var<storage, read_write> fb5: array<f32>;
@group(0) @binding(6) var<storage, read_write> fb6: array<f32>;
@group(0) @binding(7) var<storage, read> fused_prog: array<u32>;
@group(0) @binding(8) var<uniform> fused_p: FusedParams;

fn fused_load(slot: u32, i: u32) -> f32 {
    var v = 0.0;
    switch slot {
        case 0u: { v = fb0[i]; }
        case 1u: { v = fb1[i]; }
        case 2u: { v = fb2[i]; }
        case 3u: { v = fb3[i]; }
        case 4u: { v = fb4[i]; }
        case 5u: { v = fb5[i]; }
        default: { v = fb6[i]; }
    }
    return v;
}

fn fused_store(slot: u32, i: u32, v: f32) {
    switch slot {
        case 0u: { fb0[i] = v; }
        case 1u: { fb1[i] = v; }
        case 2u: { fb2[i] = v; }
        case 3u: { fb3[i] = v; }
        case 4u: { fb4[i] = v; }
        case 5u: { fb5[i] = v; }
        default: { fb6[i] = v; }
    }
}

@compute @workgroup_size(256)
fn fused(@builtin(global_invocation_id) gid: vec3<u32>,
         @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = flat_index(gid, nwg);
    if (i >= fused_p.n) { return; }
    for (var k = 0u; k < fused_p.steps; k = k + 1u) {
        let w = k * 8u;
        let x = fused_load(fused_prog[w + 2u], i);
        var r = 0.0;
        if (fused_prog[w] == 0u) {
            let a = bitcast<f32>(fused_prog[w + 5u]);
            let b = bitcast<f32>(fused_prog[w + 6u]);
            r = unary_apply(fused_prog[w + 1u], x, a, b);
        } else {
            r = binary_apply(fused_prog[w + 1u], x, fused_load(fused_prog[w + 3u], i));
        }
        fused_store(fused_prog[w + 4u], i, r);
    }
}

// --- Axis reductions --------------------------------------------------------
// The input is viewed as [outer, dim, inner]; one thread per (outer, inner).

struct AxisParams {
    op: u32,
    outer: u32,
    dim: u32,
    inner: u32,
};

@group(0) @binding(0) var<storage, read> red_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> red_out: array<f32>;
@group(0) @binding(2) var<uniform> red_p: AxisParams;

@compute @workgroup_size(256)
fn reduce(@builtin(global_invocation_id) gid: vec3<u32>,
          @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = flat_index(gid, nwg);
    if (i >= red_p.outer * red_p.inner) { return; }
    let o = i / red_p.inner;
    let k = i % red_p.inner;
    let base = o * red_p.dim * red_p.inner + k;
    var acc = red_in[base];
    for (var j = 1u; j < red_p.dim; j = j + 1u) {
        let v = red_in[base + j * red_p.inner];
        switch red_p.op {
            case 1u: { acc = max(acc, v); }
            case 2u: { acc = min(acc, v); }
            default: { acc = acc + v; }
        }
    }
    red_out[i] = acc;
}

// --- Full reductions (two stages) ---------------------------------------------
// Workgroup w adds elements w*256 + lid, stepping by the whole grid, then
// combines its 256 partials in a tree and writes red_part_out[w]. The first
// dispatch leaves one partial per workgroup, a one-workgroup dispatch over
// those partials gives the result.

struct AllParams {
    op: u32,
    n: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> all_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> all_out: array<f32>;
@group(0) @binding(2) var<uniform> all_p: AllParams;

var<workgroup> all_part: array<f32, 256>;

fn reduce_combine(op: u32, a: f32, b: f32) -> f32 {
    switch op {
        case 1u: { return max(a, b); }
        case 2u: { return min(a, b); }
        default: { return a + b; }
    }
}

@compute @workgroup_size(256)
fn reduce_all(@builtin(local_invocation_index) lid: u32,
              @builtin(workgroup_id) wid: vec3<u32>,
              @builtin(num_workgroups) nwg: vec3<u32>) {
    let op = all_p.op;
    // Identity: 0 for sum, -inf for max, +inf for min.
    var acc = 0.0;
    if (op == 1u) { acc = bitcast<f32>(0xff800000u); }
    if (op == 2u) { acc = bitcast<f32>(0x7f800000u); }
    let step = nwg.x * 256u;
    for (var i = wid.x * 256u + lid; i < all_p.n; i = i + step) {
        acc = reduce_combine(op, acc, all_in[i]);
    }
    all_part[lid] = acc;
    workgroupBarrier();
    for (var half = 128u; half > 0u; half = half >> 1u) {
        if (lid < half) {
            all_part[lid] = reduce_combine(op, all_part[lid], all_part[lid + half]);
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        all_out[wid.x] = all_part[0];
    }
}

// --- Argmax / argmin along an axis ------------------------------------------
// The input is viewed as [outer, dim, inner] and each of the outer * inner
// outputs is split into `splits` workgroups, each taking every
// (splits * 256)-th element of its share of dim. Values travel as u32 bits so
// nothing is flushed or canonicalised on the way.
// meta = [flags, outputs, dim, inner, splits]; flags bit 0: argmin,
// bit 1: the input is (value bits, index) pairs from an earlier pass (inner
// must be 1), bit 2: write (value bits, index) pairs for workgroup w at
// [2w, 2w + 1] instead of the f32 index.
// Ties go to the lower index; for argmax NaN beats every number, for argmin
// every number beats NaN, as in the CPU path.

@group(0) @binding(0) var<storage, read> arg_in: array<u32>;
@group(0) @binding(1) var<storage, read_write> arg_out: array<u32>;
@group(0) @binding(2) var<storage, read> arg_meta: array<u32>;

var<workgroup> arg_val: array<u32, 256>;
var<workgroup> arg_idx: array<u32, 256>;

const ARG_NONE: u32 = 0xffffffffu;

fn is_nan_bits(v: u32) -> bool {
    return (v & 0x7fffffffu) > 0x7f800000u;
}

// Whether candidate (v, i) replaces the current best (bv, bi).
fn arg_better(smallest: bool, v: u32, i: u32, bv: u32, bi: u32) -> bool {
    if (i == ARG_NONE) { return false; }
    if (bi == ARG_NONE) { return true; }
    let vn = is_nan_bits(v);
    let bn = is_nan_bits(bv);
    if (vn && bn) { return i < bi; }
    if (vn || bn) { return vn != smallest; }
    let a = bitcast<f32>(v);
    let b = bitcast<f32>(bv);
    if (a == b) { return i < bi; }
    return select(a > b, a < b, smallest);
}

@compute @workgroup_size(256)
fn arg_reduce(@builtin(local_invocation_index) lid: u32,
              @builtin(workgroup_id) wid: vec3<u32>,
              @builtin(num_workgroups) nwg: vec3<u32>) {
    let flags = arg_meta[0];
    let dim = arg_meta[2];
    let inner = arg_meta[3];
    let splits = arg_meta[4];
    let smallest = (flags & 1u) != 0u;
    let pairs_in = (flags & 2u) != 0u;
    let w = wid.x + wid.y * nwg.x;
    let live = w < arg_meta[1] * splits;
    let o_flat = w / splits;
    let base = (o_flat / inner) * dim * inner + o_flat % inner;
    var bv = 0u;
    var bi = ARG_NONE;
    if (live) {
        for (var j = (w % splits) * 256u + lid; j < dim; j = j + splits * 256u) {
            var v = 0u;
            var i = j;
            if (pairs_in) {
                v = arg_in[2u * (o_flat * dim + j)];
                i = arg_in[2u * (o_flat * dim + j) + 1u];
            } else {
                v = arg_in[base + j * inner];
            }
            if (arg_better(smallest, v, i, bv, bi)) {
                bv = v;
                bi = i;
            }
        }
    }
    arg_val[lid] = bv;
    arg_idx[lid] = bi;
    workgroupBarrier();
    for (var half = 128u; half > 0u; half = half >> 1u) {
        if (lid < half && arg_better(smallest, arg_val[lid + half], arg_idx[lid + half],
                                     arg_val[lid], arg_idx[lid])) {
            arg_val[lid] = arg_val[lid + half];
            arg_idx[lid] = arg_idx[lid + half];
        }
        workgroupBarrier();
    }
    if (lid == 0u && live) {
        if ((flags & 4u) != 0u) {
            arg_out[2u * w] = arg_val[0];
            arg_out[2u * w + 1u] = arg_idx[0];
        } else {
            arg_out[o_flat] = bitcast<u32>(f32(arg_idx[0]));
        }
    }
}

// --- Softmax / log-softmax along an axis -------------------------------------
// op 0 = softmax, op 1 = log_softmax.

@group(0) @binding(0) var<storage, read> sm_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> sm_out: array<f32>;
@group(0) @binding(2) var<uniform> sm_p: AxisParams;

@compute @workgroup_size(256)
fn softmax(@builtin(global_invocation_id) gid: vec3<u32>,
           @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = flat_index(gid, nwg);
    if (i >= sm_p.outer * sm_p.inner) { return; }
    let o = i / sm_p.inner;
    let k = i % sm_p.inner;
    let base = o * sm_p.dim * sm_p.inner + k;
    var m = sm_in[base];
    for (var j = 1u; j < sm_p.dim; j = j + 1u) {
        m = max(m, sm_in[base + j * sm_p.inner]);
    }
    var s = 0.0;
    for (var j = 0u; j < sm_p.dim; j = j + 1u) {
        s = s + exp(sm_in[base + j * sm_p.inner] - m);
    }
    let log_s = log(s);
    for (var j = 0u; j < sm_p.dim; j = j + 1u) {
        let idx = base + j * sm_p.inner;
        let z = sm_in[idx] - m;
        if (sm_p.op == 1u) {
            sm_out[idx] = z - log_s;
        } else {
            sm_out[idx] = exp(z) / s;
        }
    }
}

// --- Batched tiled matrix multiplication -----------------------------------
// out[b] = a[b] @ b[b]; flags bit 0: a is batched, bit 1: b is batched.

struct MatmulParams {
    m: u32,
    k: u32,
    n: u32,
    flags: u32,
};

@group(0) @binding(0) var<storage, read> mm_a: array<f32>;
@group(0) @binding(1) var<storage, read> mm_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> mm_out: array<f32>;
@group(0) @binding(3) var<uniform> mm_p: MatmulParams;

var<workgroup> tile_a: array<array<f32, 16>, 16>;
var<workgroup> tile_b: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16, 1)
fn matmul(@builtin(global_invocation_id) gid: vec3<u32>,
          @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    let batch = gid.z;
    let m = mm_p.m;
    let k = mm_p.k;
    let n = mm_p.n;
    let a_off = select(0u, batch * m * k, (mm_p.flags & 1u) != 0u);
    let b_off = select(0u, batch * k * n, (mm_p.flags & 2u) != 0u);
    let o_off = batch * m * n;

    var acc = 0.0;
    let num_tiles = (k + 15u) / 16u;
    for (var t = 0u; t < num_tiles; t = t + 1u) {
        let ac = t * 16u + lid.x;
        if (row < m && ac < k) {
            tile_a[lid.y][lid.x] = mm_a[a_off + row * k + ac];
        } else {
            tile_a[lid.y][lid.x] = 0.0;
        }
        let br = t * 16u + lid.y;
        if (br < k && col < n) {
            tile_b[lid.y][lid.x] = mm_b[b_off + br * n + col];
        } else {
            tile_b[lid.y][lid.x] = 0.0;
        }
        workgroupBarrier();
        for (var i = 0u; i < 16u; i = i + 1u) {
            acc = acc + tile_a[lid.y][i] * tile_b[i][lid.x];
        }
        workgroupBarrier();
    }
    if (row < m && col < n) {
        mm_out[o_off + row * n + col] = acc;
    }
}

// --- Strided copy (permute, expand, narrow) ---------------------------------
// meta = [out_shape(ndim), source stride of each output dim(ndim)]; reads
// start at `offset`. Stride 0 broadcasts, permuted strides transpose.

struct PermuteParams {
    n: u32,
    ndim: u32,
    offset: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> perm_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> perm_out: array<f32>;
@group(0) @binding(2) var<uniform> perm_p: PermuteParams;
@group(0) @binding(3) var<storage, read> perm_meta: array<u32>;

@compute @workgroup_size(256)
fn permute(@builtin(global_invocation_id) gid: vec3<u32>,
           @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = flat_index(gid, nwg);
    if (i >= perm_p.n) { return; }
    let nd = perm_p.ndim;
    var rem = i;
    var src = perm_p.offset;
    for (var d = 0u; d < nd; d = d + 1u) {
        let axis = nd - 1u - d;
        let size = perm_meta[axis];
        src = src + (rem % size) * perm_meta[nd + axis];
        rem = rem / size;
    }
    perm_out[i] = perm_in[src];
}

// --- Row gather (embedding lookup / index_select on dim 0) -------------------
// With ids_f32 set, the indices are f32 values (an embedding's id tensor);
// an id outside [0, rows) gives a row of zeros.

struct GatherParams {
    n_idx: u32,
    row: u32,
    rows: u32,
    ids_f32: u32,
};

@group(0) @binding(0) var<storage, read> gat_table: array<f32>;
@group(0) @binding(1) var<storage, read> gat_idx: array<u32>;
@group(0) @binding(2) var<storage, read_write> gat_out: array<f32>;
@group(0) @binding(3) var<uniform> gat_p: GatherParams;

@compute @workgroup_size(256)
fn gather_rows(@builtin(global_invocation_id) gid: vec3<u32>,
               @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = flat_index(gid, nwg);
    if (i >= gat_p.n_idx * gat_p.row) { return; }
    let r = i / gat_p.row;
    let c = i % gat_p.row;
    var id = gat_idx[r];
    if (gat_p.ids_f32 != 0u) {
        let f = bitcast<f32>(id);
        id = select(gat_p.rows, u32(f), f >= 0.0 && f < f32(gat_p.rows));
    }
    if (id >= gat_p.rows) {
        gat_out[i] = 0.0;
        return;
    }
    gat_out[i] = gat_table[id * gat_p.row + c];
}

// --- Row scatter-add (index_select and embedding backward) -------------------
// WGSL has no float atomics, so the gradient rows are not added with atomics.
// Instead the (row, position) pairs of the ids are sorted by row, then by
// position; head[row] is the index of the row's first pair, or SORT_NONE when
// no id picks it. One thread per output element adds its row's gradients in
// position order, as the CPU loop does, so the result is deterministic and
// matches the CPU bit for bit.

const SORT_NONE: u32 = 0xffffffffu;

// sort_ids builds the pairs and the head table on the GPU from f32 ids.
// op 0: pair i = (row of id i, i) for i < n, padded with (SORT_NONE, i) up to
//       a = the power-of-two pair count; b = rows. An id outside [0, rows)
//       gets row SORT_NONE: it sorts last and adds nothing.
// op 1: one compare-and-swap step of a bitonic sort of n pairs (a = k, b = j).
// op 2: head[r] = SORT_NONE for r < n.
// op 3: head[row] = i for each pair i < n that is the first of its row.

struct SortParams {
    op: u32,
    n: u32,
    a: u32,
    b: u32,
};

@group(0) @binding(0) var<storage, read> srt_ids: array<f32>;
@group(0) @binding(1) var<storage, read_write> srt_keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> srt_head: array<u32>;
@group(0) @binding(3) var<uniform> srt_p: SortParams;

@compute @workgroup_size(256)
fn sort_ids(@builtin(global_invocation_id) gid: vec3<u32>,
            @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = flat_index(gid, nwg);
    switch srt_p.op {
        case 0u: {
            if (i >= srt_p.a) { return; }
            var row = SORT_NONE;
            if (i < srt_p.n) {
                let f = srt_ids[i];
                if (f >= 0.0 && f < f32(srt_p.b)) { row = u32(f); }
            }
            srt_keys[2u * i] = row;
            srt_keys[2u * i + 1u] = i;
        }
        case 1u: {
            let l = i ^ srt_p.b;
            if (i >= srt_p.n || l <= i) { return; }
            let ri = srt_keys[2u * i];
            let pi = srt_keys[2u * i + 1u];
            let rl = srt_keys[2u * l];
            let pl = srt_keys[2u * l + 1u];
            let greater = ri > rl || (ri == rl && pi > pl);
            if (greater == ((i & srt_p.a) == 0u)) {
                srt_keys[2u * i] = rl;
                srt_keys[2u * i + 1u] = pl;
                srt_keys[2u * l] = ri;
                srt_keys[2u * l + 1u] = pi;
            }
        }
        case 2u: {
            if (i >= srt_p.n) { return; }
            srt_head[i] = SORT_NONE;
        }
        default: {
            if (i >= srt_p.n) { return; }
            let row = srt_keys[2u * i];
            if (row == SORT_NONE) { return; }
            if (i == 0u || srt_keys[2u * i - 2u] != row) {
                srt_head[row] = i;
            }
        }
    }
}

// The gradient is viewed as [outer, n, row] and the output as
// [outer, rows, row].
struct ScatterParams {
    rows: u32,
    row: u32,
    n: u32,
    outer: u32,
};

@group(0) @binding(0) var<storage, read> sct_grad: array<f32>;
@group(0) @binding(1) var<storage, read> sct_keys: array<u32>;
@group(0) @binding(2) var<storage, read> sct_head: array<u32>;
@group(0) @binding(3) var<storage, read_write> sct_out: array<f32>;
@group(0) @binding(4) var<uniform> sct_p: ScatterParams;

@compute @workgroup_size(256)
fn scatter_rows(@builtin(global_invocation_id) gid: vec3<u32>,
                @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = flat_index(gid, nwg);
    if (i >= sct_p.outer * sct_p.rows * sct_p.row) { return; }
    let o = i / (sct_p.rows * sct_p.row);
    let r = (i / sct_p.row) % sct_p.rows;
    let c = i % sct_p.row;
    let base = o * sct_p.n * sct_p.row + c;
    var acc = 0.0;
    var t = sct_head[r];
    if (t != SORT_NONE) {
        loop {
            if (t >= sct_p.n || sct_keys[2u * t] != r) { break; }
            acc = acc + sct_grad[base + sct_keys[2u * t + 1u] * sct_p.row];
            t = t + 1u;
        }
    }
    sct_out[i] = acc;
}

// --- Block copy (cat) --------------------------------------------------------
// Copies an input viewed as [outer, block_in] into columns
// [offset, offset + block_in) of an output viewed as [outer, block_out].

struct CopyParams {
    n: u32,
    block_in: u32,
    block_out: u32,
    offset: u32,
};

@group(0) @binding(0) var<storage, read> cpy_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> cpy_out: array<f32>;
@group(0) @binding(2) var<uniform> cpy_p: CopyParams;

@compute @workgroup_size(256)
fn copy_block(@builtin(global_invocation_id) gid: vec3<u32>,
              @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = flat_index(gid, nwg);
    if (i >= cpy_p.n) { return; }
    let o = i / cpy_p.block_in;
    cpy_out[o * cpy_p.block_out + cpy_p.offset + i % cpy_p.block_in] = cpy_in[i];
}

// --- Cross-entropy with class-id targets ---------------------------------------
// Logits are viewed as [rows, classes]; targets hold one f32 class id per row.
// A target <= -1 (an id that is negative as an integer, such as -100) skips
// its row: loss 0, valid 0, gradient 0. Otherwise the id is truncated to an
// integer; one that is not below `classes` (or NaN) gives a NaN loss.
// cross_entropy_rows: one workgroup per row writes the row's loss
// (lse - logit[target]), its log-sum-exp and whether it counts.
// cross_entropy_grad: d loss[r] / d logit[r, c] = softmax[r, c] - [c == target],
// times the incoming gradient of the row's loss.

struct CeParams {
    rows: u32,
    classes: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> ce_logits: array<f32>;
@group(0) @binding(1) var<storage, read> ce_targets: array<f32>;
@group(0) @binding(2) var<storage, read_write> ce_loss: array<f32>;
@group(0) @binding(3) var<storage, read_write> ce_lse: array<f32>;
@group(0) @binding(4) var<storage, read_write> ce_valid: array<f32>;
@group(0) @binding(5) var<uniform> ce_p: CeParams;

var<workgroup> ce_part: array<f32, 256>;

@compute @workgroup_size(256)
fn cross_entropy_rows(@builtin(local_invocation_index) lid: u32,
                      @builtin(workgroup_id) wid: vec3<u32>,
                      @builtin(num_workgroups) nwg: vec3<u32>) {
    let r = wid.x + wid.y * nwg.x;
    // Uniform per workgroup, so every thread reaches the barriers or none.
    if (r >= ce_p.rows) { return; }
    let c = ce_p.classes;
    let base = r * c;
    var m = bitcast<f32>(0xff800000u);
    for (var j = lid; j < c; j = j + 256u) {
        m = max(m, ce_logits[base + j]);
    }
    ce_part[lid] = m;
    workgroupBarrier();
    for (var half = 128u; half > 0u; half = half >> 1u) {
        if (lid < half) {
            ce_part[lid] = max(ce_part[lid], ce_part[lid + half]);
        }
        workgroupBarrier();
    }
    m = ce_part[0];
    workgroupBarrier();
    var s = 0.0;
    for (var j = lid; j < c; j = j + 256u) {
        s = s + exp(ce_logits[base + j] - m);
    }
    ce_part[lid] = s;
    workgroupBarrier();
    for (var half = 128u; half > 0u; half = half >> 1u) {
        if (lid < half) {
            ce_part[lid] = ce_part[lid] + ce_part[lid + half];
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        let lse = m + log(ce_part[0]);
        let f = ce_targets[r];
        ce_lse[r] = lse;
        if (f <= -1.0) {
            ce_loss[r] = 0.0;
            ce_valid[r] = 0.0;
        } else {
            ce_valid[r] = 1.0;
            if (f < f32(c)) {
                ce_loss[r] = lse - ce_logits[base + u32(max(f, 0.0))];
            } else {
                ce_loss[r] = bitcast<f32>(0x7fc00000u);
            }
        }
    }
}

@group(0) @binding(0) var<storage, read> ceg_logits: array<f32>;
@group(0) @binding(1) var<storage, read> ceg_targets: array<f32>;
@group(0) @binding(2) var<storage, read> ceg_lse: array<f32>;
@group(0) @binding(3) var<storage, read> ceg_grad: array<f32>;
@group(0) @binding(4) var<storage, read_write> ceg_out: array<f32>;
@group(0) @binding(5) var<uniform> ceg_p: CeParams;

@compute @workgroup_size(256)
fn cross_entropy_grad(@builtin(global_invocation_id) gid: vec3<u32>,
                      @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = flat_index(gid, nwg);
    if (i >= ceg_p.rows * ceg_p.classes) { return; }
    let r = i / ceg_p.classes;
    let f = ceg_targets[r];
    if (f <= -1.0) {
        ceg_out[i] = 0.0;
        return;
    }
    let p = exp(ceg_logits[i] - ceg_lse[r]);
    let hit = f < f32(ceg_p.classes) && i % ceg_p.classes == u32(max(f, 0.0));
    ceg_out[i] = (p - select(0.0, 1.0, hit)) * ceg_grad[r];
}
"#;
