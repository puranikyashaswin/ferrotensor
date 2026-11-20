//! What the kernel does with each accumulator fragment on its way to `c`.

/// The epilogue a kernel ends with.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum EpilogueKind {
    /// `c = alpha * acc + beta * c` per element, guarded at the edges of M
    /// and N.
    AlphaBeta,
}

/// The epilogue of fragment `(i, j)` for `kind`.
pub(super) fn store_fragment(kind: EpilogueKind, i: u32, j: u32) -> String {
    match kind {
        EpilogueKind::AlphaBeta => alpha_beta_fragment(i, j),
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
