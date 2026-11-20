//! Staging `a` and `b` from global memory into workgroup memory.

use super::CoopKey;

/// How a kernel reads its operands from global memory into the staging tiles.
///
/// Only `a` has more than one way: `b` is always one f16 buffer, so a LoRA
/// `A` matrix folded into the weight is just more columns of `b`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(in crate::engine) enum LoaderKind {
    /// `a` as f32, converted to f16 with the overflow check, and `b` as f16,
    /// each read straight from its own buffer.
    Plain,
    /// `a` is `count` (2 to 4) f32 matrices side by side along K, each its
    /// own binding with its own row length: `[dQ | dK | dV | dU]` against the
    /// fused weight. Source `s` covers the columns from bound `s` to bound
    /// `s + 1`, the bounds being 0, `params[2].xyz` and K.
    Sources {
        /// How many matrices.
        count: u8,
    },
    /// `a` is `[M, 2K]` f32, a gate half then an up half, and the value
    /// staged for column `k` is `silu(a[r, k]) * a[r, K + k]`: the down
    /// projection reads the gate/up output with no product buffer between.
    SwiGlu,
}

/// Most matrices a [`LoaderKind::Sources`] loader reads.
pub(in crate::engine) const MAX_SOURCES: usize = 4;

/// First binding of the extra sources `a1`, `a2` and `a3`.
pub(in crate::engine) const SOURCE_BINDING: u32 = 5;

/// Vec4s of `params` the loader reads: 2 for the sizes, 3 with the source bounds.
pub(in crate::engine) fn params_vec4s(kind: LoaderKind) -> u32 {
    match kind {
        LoaderKind::Sources { .. } => 3,
        LoaderKind::Plain | LoaderKind::SwiGlu => 2,
    }
}

/// The binding declarations the loader adds to the kernel: the sources after
/// the first, typed as the first (`a_vec4`).
pub(in crate::engine) fn bindings(kind: LoaderKind, a_vec4: bool) -> String {
    let LoaderKind::Sources { count } = kind else {
        return String::new();
    };
    let ty = if a_vec4 { "vec4<f32>" } else { "f32" };
    (1..u32::from(count))
        .map(|s| {
            format!(
                "@group(0) @binding({}) var<storage, read> a{s}: array<{ty}>;\n",
                SOURCE_BINDING + s - 1
            )
        })
        .collect()
}

/// The binding name, first column and end column of each of `count` sources of
/// a [`LoaderKind::Sources`] loader, as WGSL expressions: source `s + 1` starts
/// where source `s` ends, and the last one ends at K.
pub(super) fn source_spans(count: u8) -> Vec<(&'static str, &'static str, &'static str)> {
    const NAMES: [&str; MAX_SOURCES] = ["a", "a1", "a2", "a3"];
    const LO: [&str; MAX_SOURCES] = ["0u", "params[2].x", "params[2].y", "params[2].z"];
    let count = usize::from(count).clamp(1, MAX_SOURCES);
    NAMES
        .iter()
        .zip(LO)
        .enumerate()
        .take(count)
        .map(|(s, (name, lo))| {
            let hi = if s + 1 < count {
                LO[s + 1]
            } else {
                "params[0].y"
            };
            (*name, lo, hi)
        })
        .collect()
}

/// The WGSL function `a_load` that reads `a` at row `gr`, column `gk` (four
/// elements as a `vec4<f32>` with `a_vec4`) for every loader but
/// [`LoaderKind::Plain`], which indexes `a` itself in the staging loop.
pub(in crate::engine) fn helpers(kind: LoaderKind, a_vec4: bool) -> String {
    match kind {
        LoaderKind::Plain => String::new(),
        _ => element_fn(kind, a_vec4),
    }
}

/// `a_load` for every loader, `Plain` included: the reference kernel reads
/// `a` through it.
pub(in crate::engine) fn element_fn(kind: LoaderKind, a_vec4: bool) -> String {
    let (ty, div) = if a_vec4 {
        ("vec4<f32>", " / 4u")
    } else {
        ("f32", "")
    };
    let mut body = String::new();
    match kind {
        LoaderKind::Plain => {
            body.push_str(&format!("  return a[(gr * params[0].y + gk){div}];\n"));
        }
        LoaderKind::Sources { count } => {
            let spans = source_spans(count);
            for (s, (name, lo, hi)) in spans.iter().enumerate() {
                let read = format!("{name}[(gr * ({hi} - {lo}) + gk - {lo}){div}]");
                if s + 1 < spans.len() {
                    body.push_str(&format!("  if (gk < {hi}) {{ return {read}; }}\n"));
                } else {
                    body.push_str(&format!("  return {read};\n"));
                }
            }
        }
        LoaderKind::SwiGlu => {
            let one = if a_vec4 { "vec4<f32>(1.0)" } else { "1.0" };
            body.push_str(&format!(
                "  let row = gr * 2u * params[0].y + gk;
  let g = a[row{div}];
  let u = a[(row + params[0].y){div}];
  return g / ({one} + exp(-g)) * u;
"
            ));
        }
    }
    format!("fn a_load(gr: u32, gk: u32) -> {ty} {{\n{body}}}\n")
}

/// The expression that reads `a` at `(gr, gk)` in the staging loop.
fn read_a(key: CoopKey) -> &'static str {
    match (key.loader, key.a_vec4) {
        (LoaderKind::Plain, true) => "a[(gr * k + gk) / 4u]",
        (LoaderKind::Plain, false) => "a[gr * k + gk]",
        _ => "a_load(gr, gk)",
    }
}

/// Staging `a[m0.., k0..]` into `sa` as f16, flagging values f16 cannot hold.
fn load_a(key: CoopKey, sa: &str) -> String {
    let nt = key.tile.threads();
    let bk = key.tile.bk;
    if key.a_vec4 {
        let count = key.tile.bm * (bk / 4) / nt;
        format!(
            "    for (var i = 0u; i < {count}u; i++) {{
      let idx = li + i * {nt}u;
      let r = idx / {q}u;
      let q = idx % {q}u;
      let gr = m0 + r;
      let gk = k0 + q * 4u;
      var v = vec4<f32>(0.0);
      if (gr < m && gk < k) {{ v = {read}; }}
      if (any(!(abs(v) <= vec4<f32>(F16_MAX)))) {{ overflow = true; }}
      let h = vec4<f16>(clamp(v, vec4<f32>(-F16_MAX), vec4<f32>(F16_MAX)));
      let o = r * AP + q * 4u;
      {sa}[o] = h.x;
      {sa}[o + 1u] = h.y;
      {sa}[o + 2u] = h.z;
      {sa}[o + 3u] = h.w;
    }}
",
            q = bk / 4,
            read = read_a(key)
        )
    } else {
        let count = key.tile.bm * bk / nt;
        format!(
            "    for (var i = 0u; i < {count}u; i++) {{
      let idx = li + i * {nt}u;
      let r = idx / BK;
      let q = idx % BK;
      let gr = m0 + r;
      let gk = k0 + q;
      var v = 0.0;
      if (gr < m && gk < k) {{ v = {read}; }}
      if (!(abs(v) <= F16_MAX)) {{ overflow = true; }}
      {sa}[r * AP + q] = f16(clamp(v, -F16_MAX, F16_MAX));
    }}
",
            read = read_a(key)
        )
    }
}

/// Staging the f16 `b` tile into `sb`: `[k][n]` rows of BP for `b[K, N]`,
/// `[n][k]` rows of AP for `b[N, K]`.
fn load_b(key: CoopKey, sb: &str) -> String {
    let nt = key.tile.threads();
    let (bn, bk) = (key.tile.bn, key.tile.bk);
    if key.b_vec8 {
        // (rows of the tile, 8-half chunks per row), then the global row,
        // column of the chunk, word index and workgroup offset.
        let (rows_chunks, body) = if key.transposed_b {
            (
                (bn, bk / 8),
                "      let gn = n0 + r;\n      let gk = k0 + q * 8u;\n      if (gn < n && gk < k) { w = b[(gn * k + gk) / 8u]; }\n      let o = r * AP + q * 8u;\n",
            )
        } else {
            (
                (bk, bn / 8),
                "      let gk = k0 + r;\n      let gn = n0 + q * 8u;\n      if (gk < k && gn < n) { w = b[(gk * n + gn) / 8u]; }\n      let o = r * BP + q * 8u;\n",
            )
        };
        let (rows, chunks) = rows_chunks;
        let count = rows * chunks / nt;
        format!(
            "    for (var i = 0u; i < {count}u; i++) {{
      let idx = li + i * {nt}u;
      let r = idx / {chunks}u;
      let q = idx % {chunks}u;
      var w = vec4<u32>(0u);
{body}      let h0 = vec2<f16>(unpack2x16float(w.x));
      let h1 = vec2<f16>(unpack2x16float(w.y));
      let h2 = vec2<f16>(unpack2x16float(w.z));
      let h3 = vec2<f16>(unpack2x16float(w.w));
      {sb}[o] = h0.x;
      {sb}[o + 1u] = h0.y;
      {sb}[o + 2u] = h1.x;
      {sb}[o + 3u] = h1.y;
      {sb}[o + 4u] = h2.x;
      {sb}[o + 5u] = h2.y;
      {sb}[o + 6u] = h3.x;
      {sb}[o + 7u] = h3.y;
    }}
"
        )
    } else {
        let count = bk * bn / nt;
        let body = if key.transposed_b {
            "      let r = idx / BK;\n      let q = idx % BK;\n      let gn = n0 + r;\n      let gk = k0 + q;\n      let e = gn * k + gk;\n      let o = r * AP + q;\n"
        } else {
            "      let r = idx / BN;\n      let q = idx % BN;\n      let gk = k0 + r;\n      let gn = n0 + q;\n      let e = gk * n + gn;\n      let o = r * BP + q;\n"
        };
        format!(
            "    for (var i = 0u; i < {count}u; i++) {{
      let idx = li + i * {nt}u;
{body}      var v = 0.0;
      if (gk < k && gn < n) {{ v = unpack2x16float(b[e >> 1u])[e & 1u]; }}
      {sb}[o] = f16(v);
    }}
"
        )
    }
}

/// The loads of the K step that starts at `k0` into the staging pair
/// `(sa, sb)`.
pub(super) fn load_step(key: CoopKey, k0: &str, bufs: (&str, &str)) -> String {
    match key.loader {
        // The loaders differ in how `a` is read (`read_a`), not in the shape
        // of a step.
        LoaderKind::Plain | LoaderKind::Sources { .. } | LoaderKind::SwiGlu => {
            plain_step(key, k0, bufs)
        }
    }
}

/// One K step's loads: `a` through the key's loader, then `b`.
fn plain_step(key: CoopKey, k0: &str, (sa, sb): (&str, &str)) -> String {
    format!(
        "    {{\n    let k0 = {k0};\n{}{}    }}\n",
        load_a(key, sa),
        load_b(key, sb)
    )
}
