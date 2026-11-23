// Grouped LoRA kernels: one dispatch serves up to three projections that
// share an input (q, k and v). This file is a template read by
// engine/lora_group.rs, which, for one kernel and one group size, prepends
// the constants COUNT and RANK, the binding declarations and the parameter
// declaration, then expands every block between `//#repeat` and `//#end`
// once per projection g, replacing `$g` with g, `$t0` with 2 g and `$t1`
// with 2 g + 1. `$g` is always followed by `u` or a name suffix, so `$gu`
// becomes the literal `0u` and `b$g` becomes `b0`.
//
// Parameters, as array<vec4<u32>, 3> unless said otherwise:
//   [0] = (rows, ..), [1] = (cols of projection 0, 1, 2, 0),
//   [2] = (scale bits of projection 0, 1, 2, 0).
// The kernels `du_packed` and `grads_packed` take dY and dU as column blocks
// of one buffer and add, as array<vec4<u32>, 5>:
//   [3] = (row length of that buffer, first column of dU, 0, 0),
//   [4] = (first column of dY of projection 0, 1, 2, 0).

//#kernel ub
// Y_g += s_g * U[:, g * RANK ..][rows, RANK] @ B_g[RANK, n_g], for every row
// of Y_g. U has row length params[0].y. A workgroup is 4 rows by 64 columns
// of one projection (workgroup_id.z); a thread past the projection's width
// does nothing.
@compute @workgroup_size(256)
fn lora_ub(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let m = params[0].x;
  let ld = params[0].y;
  let g = wg.z;
  let row = wg.y * 4u + li / 64u;
  let col = wg.x * 64u + li % 64u;
  if (row >= m) { return; }
//#repeat
  if (g == $gu) {
    let n = params[1][$gu];
    if (col < n) {
      var acc = 0.0;
      for (var j = 0u; j < RANK; j += 1u) {
        acc = fma(u[row * ld + $gu * RANK + j], b$g[j * n + col], acc);
      }
      let i = row * n + col;
      y$g[i] = y$g[i] + bitcast<f32>(params[2][$gu]) * acc;
    }
  }
//#end
}

//#kernel du
// dU[:, g * RANK ..] = s_g * dY_g[rows, n_g] @ B_g^T, with B_g stored
// [RANK, n_g]. A workgroup is 4 rows of one projection; the 64 lanes of a row
// split n_g, read dY_g along n_g, and a tree in workgroup memory adds their
// sums in a fixed order, so every run gives the same bits.
var<workgroup> red: array<f32, RED_LEN>;

@compute @workgroup_size(256)
fn lora_du(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let m = params[0].x;
  let ld = params[0].y;
  let g = wg.z;
  let r = li / 64u;
  let lane = li % 64u;
  // Rows past the end read the last one; their sums are not stored.
  let row = min(wg.y * 4u + r, m - 1u);
  var acc: array<f32, RANK>;
  for (var j = 0u; j < RANK; j += 1u) { acc[j] = 0.0; }
//#repeat
  if (g == $gu) {
    let n = params[1][$gu];
    for (var c = lane; c < n; c += 64u) {
      let v = dy$g[row * n + c];
      for (var j = 0u; j < RANK; j += 1u) {
        acc[j] = fma(v, b$g[j * n + c], acc[j]);
      }
    }
  }
//#end
  for (var j = 0u; j < RANK; j += 1u) { red[j * 256u + li] = acc[j]; }
  workgroupBarrier();
  for (var s = 32u; s > 0u; s = s / 2u) {
    if (lane < s) {
      for (var j = 0u; j < RANK; j += 1u) {
        red[j * 256u + li] += red[j * 256u + li + s];
      }
    }
    workgroupBarrier();
  }
  let out_row = wg.y * 4u + r;
  if (lane < RANK && out_row < m) {
    du[out_row * ld + g * RANK + lane] = bitcast<f32>(params[2][g]) * red[lane * 256u + r * 64u];
  }
}

//#kernel grads
// Two products per projection, each a sum over the rows:
//   grad_A_g[k, RANK] = beta * grad_A_g + x^T @ dU[:, g * RANK ..]
//   grad_B_g[RANK, n_g] = beta * grad_B_g + s_g * U[:, g * RANK ..]^T @ dY_g
// (a beta of 0 never reads the gradient). x is [rows, k]; U and dU have row
// length params[0].z. Task t = workgroup_id.z is grad_A of projection t / 2
// when even and grad_B when odd. A workgroup is 32 output rows of the
// gradient (columns of x or dY_g) by 8 lanes that split the sum over rows,
// added by a tree in a fixed order. params[0] = (rows, k, ld of U and dU,
// beta bits).
var<workgroup> red: array<f32, RED_LEN>;

@compute @workgroup_size(256)
fn lora_grads(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let m = params[0].x;
  let k = params[0].y;
  let ld = params[0].z;
  let beta = bitcast<f32>(params[0].w);
  let task = wg.z;
  let cl = li % 32u;
  let lane = li / 32u;
  let c0 = wg.x * 32u + cl;
  var acc: array<f32, RANK>;
  for (var j = 0u; j < RANK; j += 1u) { acc[j] = 0.0; }
//#repeat
  if (task == $t0) {
    // Columns past the end read the last one; their sums are not stored.
    let cc = min(c0, k - 1u);
    for (var r = lane; r < m; r += 8u) {
      let p = x[r * k + cc];
      for (var j = 0u; j < RANK; j += 1u) {
        acc[j] = fma(p, du[r * ld + $gu * RANK + j], acc[j]);
      }
    }
  }
  if (task == $t1) {
    let n = params[1][$gu];
    let cc = min(c0, n - 1u);
    for (var r = lane; r < m; r += 8u) {
      let p = dy$g[r * n + cc];
      for (var j = 0u; j < RANK; j += 1u) {
        acc[j] = fma(p, u[r * ld + $gu * RANK + j], acc[j]);
      }
    }
  }
//#end
  for (var j = 0u; j < RANK; j += 1u) { red[j * 256u + li] = acc[j]; }
  workgroupBarrier();
  for (var s = 4u; s > 0u; s = s / 2u) {
    if (lane < s) {
      for (var j = 0u; j < RANK; j += 1u) {
        red[j * 256u + li] += red[j * 256u + li + s * 32u];
      }
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
//#repeat
    if (task == $t0 && c0 < k) {
      for (var j = 0u; j < RANK; j += 1u) {
        let i = c0 * RANK + j;
        var o = red[j * 256u + li];
        if (beta != 0.0) { o = o + beta * ga$g[i]; }
        ga$g[i] = o;
      }
    }
    if (task == $t1 && c0 < params[1][$gu]) {
      for (var j = 0u; j < RANK; j += 1u) {
        let i = j * params[1][$gu] + c0;
        var o = bitcast<f32>(params[2][$gu]) * red[j * 256u + li];
        if (beta != 0.0) { o = o + beta * gb$g[i]; }
        gb$g[i] = o;
      }
    }
//#end
  }
}

//#kernel du_packed
// As `du`, but dY_g and dU are blocks of columns of one buffer `du` (the
// bindings dy0, dy1 and dy2 are that buffer again: they are declared
// writable so that one dispatch does not use a buffer as both read-only and
// writable). dY_g[row, c] is du[row * params[3].x + params[4][g] + c] and
// dU[row, g * RANK + j] is du[row * params[3].x + params[3].y + g * RANK + j].
// The columns of dU are never columns of any dY, so nothing read is written.
var<workgroup> red: array<f32, RED_LEN>;

@compute @workgroup_size(256)
fn lora_du_packed(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let m = params[0].x;
  let ld = params[3].x;
  let g = wg.z;
  let r = li / 64u;
  let lane = li % 64u;
  // Rows past the end read the last one; their sums are not stored.
  let row = min(wg.y * 4u + r, m - 1u);
  var acc: array<f32, RANK>;
  for (var j = 0u; j < RANK; j += 1u) { acc[j] = 0.0; }
//#repeat
  if (g == $gu) {
    let n = params[1][$gu];
    let base = row * ld + params[4][$gu];
    for (var c = lane; c < n; c += 64u) {
      let v = dy$g[base + c];
      for (var j = 0u; j < RANK; j += 1u) {
        acc[j] = fma(v, b$g[j * n + c], acc[j]);
      }
    }
  }
//#end
  for (var j = 0u; j < RANK; j += 1u) { red[j * 256u + li] = acc[j]; }
  workgroupBarrier();
  for (var s = 32u; s > 0u; s = s / 2u) {
    if (lane < s) {
      for (var j = 0u; j < RANK; j += 1u) {
        red[j * 256u + li] += red[j * 256u + li + s];
      }
    }
    workgroupBarrier();
  }
  let out_row = wg.y * 4u + r;
  if (lane < RANK && out_row < m) {
    du[out_row * ld + params[3].y + g * RANK + lane] = bitcast<f32>(params[2][g]) * red[lane * 256u + r * 64u];
  }
}

//#kernel grads_packed
// As `grads`, but dY_g and dU are blocks of columns of the one buffer `du`
// (dy0, dy1 and dy2 are that buffer again, read only), laid out as for
// `du_packed`. U keeps its own row length params[0].z.
var<workgroup> red: array<f32, RED_LEN>;

@compute @workgroup_size(256)
fn lora_grads_packed(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let m = params[0].x;
  let k = params[0].y;
  let ld = params[0].z;
  let beta = bitcast<f32>(params[0].w);
  let pld = params[3].x;
  let task = wg.z;
  let cl = li % 32u;
  let lane = li / 32u;
  let c0 = wg.x * 32u + cl;
  var acc: array<f32, RANK>;
  for (var j = 0u; j < RANK; j += 1u) { acc[j] = 0.0; }
//#repeat
  if (task == $t0) {
    // Columns past the end read the last one; their sums are not stored.
    let cc = min(c0, k - 1u);
    for (var r = lane; r < m; r += 8u) {
      let p = x[r * k + cc];
      for (var j = 0u; j < RANK; j += 1u) {
        acc[j] = fma(p, du[r * pld + params[3].y + $gu * RANK + j], acc[j]);
      }
    }
  }
  if (task == $t1) {
    let n = params[1][$gu];
    let cc = min(c0, n - 1u);
    for (var r = lane; r < m; r += 8u) {
      let p = dy$g[r * pld + params[4][$gu] + cc];
      for (var j = 0u; j < RANK; j += 1u) {
        acc[j] = fma(p, u[r * ld + $gu * RANK + j], acc[j]);
      }
    }
  }
//#end
  for (var j = 0u; j < RANK; j += 1u) { red[j * 256u + li] = acc[j]; }
  workgroupBarrier();
  for (var s = 4u; s > 0u; s = s / 2u) {
    if (lane < s) {
      for (var j = 0u; j < RANK; j += 1u) {
        red[j * 256u + li] += red[j * 256u + li + s * 32u];
      }
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
//#repeat
    if (task == $t0 && c0 < k) {
      for (var j = 0u; j < RANK; j += 1u) {
        let i = c0 * RANK + j;
        var o = red[j * 256u + li];
        if (beta != 0.0) { o = o + beta * ga$g[i]; }
        ga$g[i] = o;
      }
    }
    if (task == $t1 && c0 < params[1][$gu]) {
      for (var j = 0u; j < RANK; j += 1u) {
        let i = j * params[1][$gu] + c0;
        var o = bitcast<f32>(params[2][$gu]) * red[j * 256u + li];
        if (beta != 0.0) { o = o + beta * gb$g[i]; }
        gb$g[i] = o;
      }
    }
//#end
  }
}

//#kernel mirror
// Copies f32 A_g[rows, RANK] into the f16 columns col0_g .. col0_g + RANK of
// the row-major f16 matrix `dst` of row length params[0].y, which is stored
// as pairs of halves packed in u32 words (the first in the low half). The row
// length, every col0 and RANK are even, so a thread owns one whole word and
// no two threads share one. Rounds to nearest even, like
// UnifiedTensor::from_f32_as_f16. Parameters, as array<vec4<u32>, 2>:
// [0] = (rows, row length of dst, RANK, 0), [1] = (col0 of projection 0, 1, 2, 0).
@compute @workgroup_size(64)
fn lora_mirror(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let rows = params[0].x;
  let ld = params[0].y;
  let half_rank = RANK / 2u;
  let idx = (wg.y * 65535u + wg.x) * 64u + li;
  let g = wg.z;
  if (idx >= rows * half_rank) { return; }
  let row = idx / half_rank;
  let w = idx % half_rank;
//#repeat
  if (g == $gu) {
    let s = row * RANK + 2u * w;
    let e = row * ld + params[1][$gu] + 2u * w;
    dst[e / 2u] = pack2x16float(vec2<f32>(a$g[s], a$g[s + 1u]));
  }
//#end
}
