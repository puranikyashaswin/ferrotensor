//! What the kernel does with each accumulator fragment on its way to `c`.
//!
//! Every epilogue but [`EpilogueKind::AlphaBeta`] is one WGSL function
//! `ep_store(r, col, acc)` that the fragment loop calls for each element
//! inside the matrix; the barriers around that loop stay in uniform control
//! flow, outside the guard, as in the first epilogue. None of them runs with
//! K split: a split product's slices are plain partial sums that the
//! reduction pass adds.

/// The epilogue a kernel ends with.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(in crate::engine) enum EpilogueKind {
    /// `c = alpha * acc + beta * c` per element, guarded at the edges of M
    /// and N.
    AlphaBeta,
    /// `c = alpha * acc + res` per element, with `res` its own `[M, N]`
    /// buffer: a residual add, or a LoRA product scaled and added to the base
    /// product, with `c` written once and never read.
    Residual,
    /// The N columns go to `segments` (1 to 4) outputs, segment `s` holding
    /// the columns from start `s` to start `s + 1` as its own `[M, width]`
    /// matrix (starts 0, `params[3].xyz` and N): the fused Q, K, V and LoRA
    /// `U` columns of one product to four buffers. Each element is
    /// `alpha * acc`, plus `bias[col]` with `bias`.
    ColumnSplit {
        /// How many outputs.
        segments: u8,
        /// Whether a bias row of N values is added.
        bias: bool,
    },
    /// The down projection's backward: `acc` is `dH[r, c]` and the epilogue
    /// reads the forward's gate `g` and up `u` at `gu[r, c]` and
    /// `gu[r, N + c]`, then writes `d_gu[r, c] = dH * u * silu'(g)` and
    /// `d_gu[r, N + c] = dH * silu(g)` to `c`, an `[M, 2N]` matrix.
    SwiGluBackward,
}

/// Most outputs a [`EpilogueKind::ColumnSplit`] writes.
pub(in crate::engine) const MAX_SEGMENTS: usize = 4;

/// First binding the epilogue adds: `res`, `gu`, or `c1` of the split.
pub(in crate::engine) const FIRST_BINDING: u32 = 8;

/// Binding of the bias row of a [`EpilogueKind::ColumnSplit`].
pub(in crate::engine) const BIAS_BINDING: u32 = FIRST_BINDING + MAX_SEGMENTS as u32 - 1;

/// Vec4s of `params` the epilogue reads: 2 for the sizes, alpha and beta, 4
/// when a split keeps its segment starts in `params[3]`.
pub(in crate::engine) fn params_vec4s(kind: EpilogueKind) -> u32 {
    match kind {
        EpilogueKind::ColumnSplit { segments, .. } if segments > 1 => 4,
        _ => 2,
    }
}

/// The binding declarations the epilogue adds to the kernel.
pub(in crate::engine) fn bindings(kind: EpilogueKind) -> String {
    match kind {
        EpilogueKind::AlphaBeta => String::new(),
        EpilogueKind::Residual => {
            format!("@group(0) @binding({FIRST_BINDING}) var<storage, read> res: array<f32>;\n")
        }
        EpilogueKind::ColumnSplit { segments, bias } => {
            let mut s: String = (1..u32::from(segments))
                .map(|i| {
                    format!(
                        "@group(0) @binding({}) var<storage, read_write> c{i}: array<f32>;\n",
                        FIRST_BINDING + i - 1
                    )
                })
                .collect();
            if bias {
                s.push_str(&format!(
                    "@group(0) @binding({BIAS_BINDING}) var<storage, read> bias: array<f32>;\n"
                ));
            }
            s
        }
        EpilogueKind::SwiGluBackward => {
            format!("@group(0) @binding({FIRST_BINDING}) var<storage, read> gu: array<f32>;\n")
        }
    }
}

/// The WGSL functions the kernel template defines for the epilogue, empty for
/// [`EpilogueKind::AlphaBeta`], whose element code is inline.
pub(in crate::engine) fn helpers(kind: EpilogueKind) -> String {
    match kind {
        EpilogueKind::AlphaBeta => String::new(),
        _ => element_fn(kind),
    }
}

/// `ep_store(r, col, acc)` for `kind`, [`EpilogueKind::AlphaBeta`] included:
/// the reference kernel finishes every element through it. It runs for
/// `r < M` and `col < N` only.
pub(in crate::engine) fn element_fn(kind: EpilogueKind) -> String {
    match kind {
        EpilogueKind::AlphaBeta => "fn ep_store(r: u32, col: u32, acc: f32) {
  let o = r * params[0].z + col;
  var v = bitcast<f32>(params[1].x) * acc;
  if (bitcast<f32>(params[1].y) != 0.0) { v += bitcast<f32>(params[1].y) * c[o]; }
  c[o] = v;
}
"
        .to_string(),
        EpilogueKind::Residual => "fn ep_store(r: u32, col: u32, acc: f32) {
  let o = r * params[0].z + col;
  c[o] = bitcast<f32>(params[1].x) * acc + res[o];
}
"
        .to_string(),
        EpilogueKind::ColumnSplit { segments, bias } => split_fn(segments, bias),
        EpilogueKind::SwiGluBackward => "fn ep_store(r: u32, col: u32, acc: f32) {
  let n = params[0].z;
  let o = r * 2u * n + col;
  let g = gu[o];
  let u = gu[o + n];
  let s = 1.0 / (1.0 + exp(-g));
  c[o] = acc * u * (s * (1.0 + g * (1.0 - s)));
  c[o + n] = acc * (g * s);
}
"
        .to_string(),
    }
}

/// `ep_put` (a value to the segment holding its column) and `ep_store` of a
/// [`EpilogueKind::ColumnSplit`].
fn split_fn(segments: u8, bias: bool) -> String {
    const NAMES: [&str; MAX_SEGMENTS] = ["c", "c1", "c2", "c3"];
    const START: [&str; MAX_SEGMENTS] = ["0u", "params[3].x", "params[3].y", "params[3].z"];
    let segments = usize::from(segments).clamp(1, MAX_SEGMENTS);
    let mut put = String::new();
    for (s, (name, start)) in NAMES.iter().zip(START).enumerate().take(segments) {
        let end = if s + 1 < segments {
            START[s + 1]
        } else {
            "params[0].z"
        };
        let write = format!("{name}[r * ({end} - {start}) + col - {start}] = v;");
        if s + 1 < segments {
            put.push_str(&format!("  if (col < {end}) {{ {write} return; }}\n"));
        } else {
            put.push_str(&format!("  {write}\n"));
        }
    }
    let add_bias = if bias { "  v += bias[col];\n" } else { "" };
    format!(
        "fn ep_put(r: u32, col: u32, v: f32) {{\n{put}}}
fn ep_store(r: u32, col: u32, acc: f32) {{
  var v = bitcast<f32>(params[1].x) * acc;
{add_bias}  ep_put(r, col, v);
}}
"
    )
}

/// The statement that poisons the first element of the workgroup's output
/// tile (`m0`, `n0`, `base`, `m` and `n` are in scope) when an `a` value
/// overflowed f16.
pub(in crate::engine) fn poison(kind: EpilogueKind) -> &'static str {
    match kind {
        EpilogueKind::AlphaBeta | EpilogueKind::Residual => {
            "c[base + m0 * n + n0] = bitcast<f32>(0x7fc00000u);"
        }
        EpilogueKind::ColumnSplit { .. } => "ep_put(m0, n0, bitcast<f32>(0x7fc00000u));",
        EpilogueKind::SwiGluBackward => "c[m0 * 2u * n + n0] = bitcast<f32>(0x7fc00000u);",
    }
}

/// The epilogue of fragment `(i, j)` for `kind`.
pub(super) fn store_fragment(kind: EpilogueKind, i: u32, j: u32) -> String {
    match kind {
        EpilogueKind::AlphaBeta => alpha_beta_fragment(i, j),
        _ => element_fragment(i, j),
    }
}

/// [`EpilogueKind::AlphaBeta`]'s epilogue of fragment `(i, j)`: through the
/// subgroup's staging fragment to `c` (this slice's partial sums when K is
/// split), guarded at the edges, with alpha and beta.
fn alpha_beta_fragment(i: u32, j: u32) -> String {
    format!(
        "  coopStoreT(acc{i}_{j}, &cs[warp * 256u], 16u);
  workgroupBarrier();
  for (var e = 0u; e < 8u; e++) {{
    let idx = lane + e * 32u;
    let r = m0 + wr + {ri}u + idx / 16u;
    let col = n0 + wc + {cj}u + idx % 16u;
    if (r < m && col < n) {{
      let o = base + r * n + col;
      var v = alpha * cs[warp * 256u + idx];
      if (beta != 0.0) {{ v += beta * c[o]; }}
      c[o] = v;
    }}
  }}
  workgroupBarrier();
",
        ri = i * 16,
        cj = j * 16
    )
}

/// The epilogue of fragment `(i, j)` for every kind with an `ep_store`: the
/// same staging through the subgroup's fragment, then `ep_store` per element
/// inside the matrix. Both barriers stay outside the guard.
fn element_fragment(i: u32, j: u32) -> String {
    format!(
        "  coopStoreT(acc{i}_{j}, &cs[warp * 256u], 16u);
  workgroupBarrier();
  for (var e = 0u; e < 8u; e++) {{
    let idx = lane + e * 32u;
    let r = m0 + wr + {ri}u + idx / 16u;
    let col = n0 + wc + {cj}u + idx % 16u;
    if (r < m && col < n) {{
      ep_store(r, col, cs[warp * 256u + idx]);
    }}
  }}
  workgroupBarrier();
",
        ri = i * 16,
        cj = j * 16
    )
}
