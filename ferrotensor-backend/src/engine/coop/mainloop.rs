//! The K loop: the 16x16 fragment multiplies of each staged step, and the
//! order of loads, multiplies and barriers around them.

use super::loader::{load_step, source_spans, LoaderKind};
use super::CoopKey;

/// The 16x16 fragments of one K substep `kk` from the staged tiles `sa` and
/// `sb`: loads of `a` and `b`, then every multiply-add of the subgroup's
/// block.
fn fragment_step(key: CoopKey, kk: u32, (sa, sb): (&str, &str)) -> String {
    let (ti, tj) = (key.tile.wm / 16, key.tile.wn / 16);
    let mut s = String::from("    {\n");
    for i in 0..ti {
        s.push_str(&format!(
            "      let fa{i} = coopLoadT<coop_mat16x16<f16, A>>(&{sa}[(wr + {}u) * AP + {}u], AP);\n",
            i * 16,
            kk * 16
        ));
    }
    for j in 0..tj {
        // b stored [N, K] is B column-major with stride AP; [K, N] is B
        // row-major with stride BP.
        let line = if key.transposed_b {
            format!(
                "      let fb{j} = coopLoad<coop_mat16x16<f16, B>>(&{sb}[(wc + {}u) * AP + {}u], AP);",
                j * 16,
                kk * 16
            )
        } else {
            format!(
                "      let fb{j} = coopLoadT<coop_mat16x16<f16, B>>(&{sb}[{}u * BP + wc + {}u], BP);",
                kk * 16,
                j * 16
            )
        };
        s.push_str(&line);
        s.push('\n');
    }
    for i in 0..ti {
        for j in 0..tj {
            s.push_str(&format!(
                "      acc{i}_{j} = coopMultiplyAdd(fa{i}, fb{j}, acc{i}_{j});\n"
            ));
        }
    }
    s.push_str("    }\n");
    s
}

/// The multiplies of one K step from the staging pair `bufs`.
fn compute_step(key: CoopKey, bufs: (&str, &str)) -> String {
    (0..key.tile.bk / 16)
        .map(|kk| fragment_step(key, kk, bufs))
        .collect()
}

/// The K loop over this slice's `steps` steps. With one staging pair each
/// step loads, waits, multiplies and waits. With two, the loads of step
/// `t + 1` go to the other pair before step `t`'s multiplies, and one
/// barrier per step separates a pair's reads from its next writes. A tile
/// with `prefetch` runs [`pipelined_loop`] instead, with any loader.
pub(super) fn main_loop(key: CoopKey) -> String {
    if key.tile.prefetch {
        return pipelined_loop(key);
    }
    let p0 = ("sa0", "sb0");
    if !key.tile.double {
        return format!(
            "  for (var t = 0u; t < steps; t++) {{\n{}    workgroupBarrier();\n{}    workgroupBarrier();\n  }}\n",
            load_step(key, "kb + t * BK", p0),
            compute_step(key, p0)
        );
    }
    let p1 = ("sa1", "sb1");
    // naga takes a function-local variable such as the loop counter as
    // non-uniform, and cooperative ops and barriers must sit in uniform
    // control flow: so the loop runs whole pairs of steps, and only `let`s
    // of uniform values guard the multiplies. Step 2p is in buffer 0, 2p + 1
    // in buffer 1; an odd last step is in buffer 0.
    format!(
        "  let pairs = steps / 2u;
  let odd = steps % 2u;
  if (steps > 0u) {{\n{first}  }}
  workgroupBarrier();
  for (var p = 0u; p < pairs; p++) {{
{load1}{mul0}    workgroupBarrier();
    if (2u * p + 2u < steps) {{\n{load0}    }}
{mul1}    workgroupBarrier();
  }}
  if (odd == 1u) {{
{mul0}    workgroupBarrier();
  }}
",
        first = load_step(key, "kb", p0),
        load1 = load_step(key, "kb + (2u * p + 1u) * BK", p1),
        mul0 = compute_step(key, p0),
        load0 = load_step(key, "kb + (2u * p + 2u) * BK", p0),
        mul1 = compute_step(key, p1),
    )
}

/// Registers each thread holds for one prefetched K step: `(a, b)`. The `a`
/// registers are `vec4<f32>` (or `f32` when K is not a multiple of 4), the
/// `b` ones `vec4<u32>` (or `u32` for an N or K that is not a multiple of 8).
fn prefetch_counts(key: CoopKey) -> (u32, u32) {
    let nt = key.tile.threads();
    let (bm, bn, bk) = (key.tile.bm, key.tile.bn, key.tile.bk);
    let a = if key.a_vec4 {
        bm * (bk / 4) / nt
    } else {
        bm * bk / nt
    };
    let b = if key.b_vec8 {
        bn * bk / 8 / nt
    } else {
        bn * bk / nt
    };
    (a, b)
}

/// The prefetch registers, zeroed, declared in the kernel's function scope.
fn prefetch_decls(key: CoopKey) -> String {
    let (na, nb) = prefetch_counts(key);
    let mut s = String::new();
    for i in 0..na {
        s.push_str(&if key.a_vec4 {
            format!("  var pa{i} = vec4<f32>(0.0);\n")
        } else {
            format!("  var pa{i} = 0.0;\n")
        });
        // The SwiGlu loader reads two values of `a` per staged element.
        if key.loader == LoaderKind::SwiGlu {
            s.push_str(&if key.a_vec4 {
                format!("  var pu{i} = vec4<f32>(0.0);\n")
            } else {
                format!("  var pu{i} = 0.0;\n")
            });
        }
    }
    for i in 0..nb {
        s.push_str(&if key.b_vec8 {
            format!("  var pb{i} = vec4<u32>(0u);\n")
        } else {
            format!("  var pb{i} = 0u;\n")
        });
    }
    s
}

/// One thread's loads of `a` for the step at `k0` (in scope) into its
/// registers, `assign(i)` being the statements that set register `i` from
/// the row `gr` and column `gk` in scope.
fn a_register_loads(key: CoopKey, assign: &dyn Fn(u32) -> String) -> String {
    let nt = key.tile.threads();
    let (na, _) = prefetch_counts(key);
    let mut s = String::new();
    for i in 0..na {
        let off = i * nt;
        let body = assign(i);
        s.push_str(&if key.a_vec4 {
            let q = key.tile.bk / 4;
            format!(
                "    {{
      let idx = li + {off}u;
      let gr = m0 + idx / {q}u;
      let gk = k0 + (idx % {q}u) * 4u;
{body}    }}
"
            )
        } else {
            format!(
                "    {{
      let idx = li + {off}u;
      let gr = m0 + idx / BK;
      let gk = k0 + idx % BK;
{body}    }}
"
            )
        });
    }
    s
}

/// The statements that set register `pa{i}` to `read` when the element is in
/// range and to zero when it is not, as in the plain loader.
fn guarded_a(key: CoopKey, i: u32, read: &str) -> String {
    let zero = if key.a_vec4 { "vec4<f32>(0.0)" } else { "0.0" };
    format!("      pa{i} = {zero};\n      if (gr < m && gk < k) {{ pa{i} = {read}; }}\n")
}

/// The global loads of `a` for the step at `k0` into `pa0..` (and `pu0..`
/// with the SwiGlu loader), in the form of the key's loader.
///
/// Sources picks the matrix once per K step: the step lies inside one source,
/// so every register reads the same binding with the plain vec4 load. A step
/// that straddles a source boundary (a boundary that is no multiple of `BK`)
/// reads through `a_load` per element, as the unpipelined loop does.
fn prefetch_load_a(key: CoopKey) -> String {
    let (div, index) = if key.a_vec4 {
        (" / 4u", "(gr * k + gk) / 4u")
    } else {
        ("", "gr * k + gk")
    };
    match key.loader {
        LoaderKind::Plain => a_register_loads(key, &|i| guarded_a(key, i, &format!("a[{index}]"))),
        LoaderKind::SwiGlu => a_register_loads(key, &|i| {
            let zero = if key.a_vec4 { "vec4<f32>(0.0)" } else { "0.0" };
            format!(
                "      pa{i} = {zero};
      pu{i} = {zero};
      if (gr < m && gk < k) {{
        let row = gr * 2u * k + gk;
        pa{i} = a[row{div}];
        pu{i} = a[(row + k){div}];
      }}
"
            )
        }),
        LoaderKind::Sources { count } => {
            let spans = source_spans(count);
            let last = spans.len().saturating_sub(1);
            let mut chain = String::new();
            for (s, (name, lo, hi)) in spans.iter().enumerate() {
                let mut conds = Vec::new();
                if s > 0 {
                    conds.push(format!("k0 >= {lo}"));
                }
                if s < last {
                    conds.push(format!("k0 + BK <= {hi}"));
                }
                let cond = if conds.is_empty() {
                    "true".to_string()
                } else {
                    conds.join(" && ")
                };
                let read = format!("{name}[(gr * ({hi} - {lo}) + gk - {lo}){div}]");
                chain.push_str(&format!(
                    "{}if ({cond}) {{\n{}    }}",
                    if s == 0 { "    " } else { " else " },
                    a_register_loads(key, &|i| guarded_a(key, i, &read))
                ));
            }
            chain.push_str(&format!(
                " else {{\n{}    }}\n",
                a_register_loads(key, &|i| guarded_a(key, i, "a_load(gr, gk)"))
            ));
            chain
        }
    }
}

/// The global loads of the K step that starts at `k0` into the prefetch
/// registers. Nothing here waits for them: the first use is the store of
/// [`prefetch_store`], after the multiplies. Out-of-range elements load as
/// zero, as in the plain loader.
fn prefetch_load(key: CoopKey, k0: &str) -> String {
    let nt = key.tile.threads();
    let (bn, bk) = (key.tile.bn, key.tile.bk);
    let (_, nb) = prefetch_counts(key);
    let mut s = format!("    {{\n    let k0 = {k0};\n");
    s.push_str(&prefetch_load_a(key));
    for i in 0..nb {
        let off = i * nt;
        s.push_str(&match (key.b_vec8, key.transposed_b) {
            (true, true) => {
                let chunks = bk / 8;
                format!(
                    "    {{
      let idx = li + {off}u;
      let gn = n0 + idx / {chunks}u;
      let gk = k0 + (idx % {chunks}u) * 8u;
      pb{i} = vec4<u32>(0u);
      if (gn < n && gk < k) {{ pb{i} = b[(gn * k + gk) / 8u]; }}
    }}
"
                )
            }
            (true, false) => {
                let chunks = bn / 8;
                format!(
                    "    {{
      let idx = li + {off}u;
      let gk = k0 + idx / {chunks}u;
      let gn = n0 + (idx % {chunks}u) * 8u;
      pb{i} = vec4<u32>(0u);
      if (gk < k && gn < n) {{ pb{i} = b[(gk * n + gn) / 8u]; }}
    }}
"
                )
            }
            (false, true) => format!(
                "    {{
      let idx = li + {off}u;
      let gn = n0 + idx / BK;
      let gk = k0 + idx % BK;
      pb{i} = 0u;
      if (gn < n && gk < k) {{ pb{i} = b[(gn * k + gk) >> 1u]; }}
    }}
"
            ),
            (false, false) => format!(
                "    {{
      let idx = li + {off}u;
      let gk = k0 + idx / BN;
      let gn = n0 + idx % BN;
      pb{i} = 0u;
      if (gk < k && gn < n) {{ pb{i} = b[(gk * n + gn) >> 1u]; }}
    }}
"
            ),
        });
    }
    s.push_str("    }\n");
    s
}

/// Writes the prefetch registers of the K step that starts at `k0` to the
/// staging pair `(sa, sb)`: `a` converted to f16 with the overflow check, `b`
/// unpacked, each in address order so the driver can merge the f16 stores.
fn prefetch_store(key: CoopKey, k0: &str, (sa, sb): (&str, &str)) -> String {
    let nt = key.tile.threads();
    let (bn, bk) = (key.tile.bn, key.tile.bk);
    let (na, nb) = prefetch_counts(key);
    let mut s = format!("    {{\n    let k0 = {k0};\n");
    for i in 0..na {
        let off = i * nt;
        // The value staged: the register, or for SwiGlu silu(gate) * up.
        let value = match (key.loader, key.a_vec4) {
            (LoaderKind::SwiGlu, true) => {
                format!("pa{i} / (vec4<f32>(1.0) + exp(-pa{i})) * pu{i}")
            }
            (LoaderKind::SwiGlu, false) => format!("pa{i} / (1.0 + exp(-pa{i})) * pu{i}"),
            _ => format!("pa{i}"),
        };
        s.push_str(&if key.a_vec4 {
            let q = bk / 4;
            format!(
                "    {{
      let v = {value};
      let idx = li + {off}u;
      let o = (idx / {q}u) * AP + (idx % {q}u) * 4u;
      if (any(!(abs(v) <= vec4<f32>(F16_MAX)))) {{ overflow = true; }}
      let h = vec4<f16>(clamp(v, vec4<f32>(-F16_MAX), vec4<f32>(F16_MAX)));
      {sa}[o] = h.x;
      {sa}[o + 1u] = h.y;
      {sa}[o + 2u] = h.z;
      {sa}[o + 3u] = h.w;
    }}
"
            )
        } else {
            format!(
                "    {{
      let v = {value};
      let idx = li + {off}u;
      if (!(abs(v) <= F16_MAX)) {{ overflow = true; }}
      {sa}[(idx / BK) * AP + idx % BK] = f16(clamp(v, -F16_MAX, F16_MAX));
    }}
"
            )
        });
    }
    for i in 0..nb {
        let off = i * nt;
        s.push_str(&if key.b_vec8 {
            // Chunks per row and row stride of the staged tile, as in the loader.
            let (chunks, stride) = if key.transposed_b {
                (bk / 8, "AP")
            } else {
                (bn / 8, "BP")
            };
            format!(
                "    {{
      let w = pb{i};
      let idx = li + {off}u;
      let o = (idx / {chunks}u) * {stride} + (idx % {chunks}u) * 8u;
      let h0 = vec2<f16>(unpack2x16float(w.x));
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
            // The word came from `e >> 1`: the element is its half `e & 1`,
            // and a zero word (out of range) gives zero for either half.
            let (elem, at) = if key.transposed_b {
                (
                    "(n0 + idx / BK) * k + k0 + idx % BK",
                    "(idx / BK) * AP + idx % BK",
                )
            } else {
                (
                    "(k0 + idx / BN) * n + n0 + idx % BN",
                    "(idx / BN) * BP + idx % BN",
                )
            };
            format!(
                "    {{
      let idx = li + {off}u;
      let e = {elem};
      {sb}[{at}] = f16(unpack2x16float(pb{i})[e & 1u]);
    }}
"
            )
        });
    }
    s.push_str("    }\n");
    s
}

/// The K loop with register prefetch. Each step issues the global loads of
/// the next into registers, runs its own multiplies out of workgroup memory,
/// and only then writes the registers to workgroup memory, so the multiplies
/// stand between a load and its first use.
///
/// With two staging pairs the next step goes to the other pair and one barrier
/// per step is enough: it makes this step's stores visible and ends the reads
/// of the pair the next step's stores overwrite. With one pair a barrier
/// before the stores keeps them from overwriting what slower subgroups still
/// read, and a second one publishes them.
///
/// Cooperative ops and barriers must be in uniform control flow, so as in the
/// plain double-buffered loop the buffers alternate over whole pairs of steps
/// and an odd last step follows the loop; only loads and stores sit under the
/// per-step `if`s, and they do not wait for anything.
fn pipelined_loop(key: CoopKey) -> String {
    let p0 = ("sa0", "sb0");
    let decls = prefetch_decls(key);
    let first = format!(
        "  if (steps > 0u) {{\n{}{}  }}\n  workgroupBarrier();\n",
        prefetch_load(key, "kb"),
        prefetch_store(key, "kb", p0)
    );
    if !key.tile.double {
        let next = "kb + (t + 1u) * BK";
        return format!(
            "{decls}{first}  for (var t = 0u; t < steps; t++) {{
    if (t + 1u < steps) {{\n{load}    }}
{mul}    workgroupBarrier();
    if (t + 1u < steps) {{\n{store}    }}
    workgroupBarrier();
  }}
",
            load = prefetch_load(key, next),
            mul = compute_step(key, p0),
            store = prefetch_store(key, next, p0),
        );
    }
    let p1 = ("sa1", "sb1");
    let (odd_next, even_next) = ("kb + (2u * p + 1u) * BK", "kb + (2u * p + 2u) * BK");
    format!(
        "{decls}  let pairs = steps / 2u;
  let odd = steps % 2u;
{first}  for (var p = 0u; p < pairs; p++) {{
{load1}{mul0}{store1}    workgroupBarrier();
    if (2u * p + 2u < steps) {{\n{load0}    }}
{mul1}    if (2u * p + 2u < steps) {{\n{store0}    }}
    workgroupBarrier();
  }}
  if (odd == 1u) {{
{mul0}    workgroupBarrier();
  }}
",
        load1 = prefetch_load(key, odd_next),
        mul0 = compute_step(key, p0),
        store1 = prefetch_store(key, odd_next, p1),
        load0 = prefetch_load(key, even_next),
        mul1 = compute_step(key, p1),
        store0 = prefetch_store(key, even_next, p0),
    )
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests {
    use super::super::{
        epilogue::EpilogueKind, immediate_source, loader::LoaderKind, source, CoopKey, CoopTile,
    };

    /// Every pipelined tile in every layout and staging-load variant.
    fn keys() -> Vec<CoopKey> {
        let mut keys = Vec::new();
        for tile in CoopTile::PIPELINED {
            for transposed_b in [false, true] {
                for a_vec4 in [false, true] {
                    for b_vec8 in [false, true] {
                        keys.push(CoopKey {
                            tile,
                            transposed_b,
                            a_vec4,
                            b_vec8,
                            loader: LoaderKind::Plain,
                            epilogue: EpilogueKind::AlphaBeta,
                        });
                    }
                }
            }
        }
        keys
    }

    fn validate(key: CoopKey, src: &str, capabilities: naga::valid::Capabilities) {
        let module = match naga::front::wgsl::parse_str(src) {
            Ok(m) => m,
            Err(e) => panic!("{key:?}: WGSL parse error:\n{}", e.emit_to_string(src)),
        };
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            capabilities
                | naga::valid::Capabilities::COOPERATIVE_MATRIX
                | naga::valid::Capabilities::SHADER_FLOAT16
                | naga::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32,
        );
        if let Err(e) = validator.validate(&module) {
            panic!("{key:?}: WGSL validation error:\n{}", e.emit_to_string(src));
        }
    }

    #[test]
    fn every_pipelined_kernel_validates() {
        for key in keys() {
            validate(key, &source(key), naga::valid::Capabilities::empty());
        }
    }

    #[test]
    fn every_pipelined_kernel_validates_with_immediates() {
        for key in keys() {
            let src = immediate_source(&source(key));
            assert!(!src.contains("var<uniform>"), "{key:?}");
            validate(key, &src, naga::valid::Capabilities::IMMEDIATES);
        }
    }

    #[test]
    fn prefetch_registers_cover_the_staged_tile() {
        for key in keys() {
            let (na, nb) = super::prefetch_counts(key);
            let t = key.tile;
            let lanes = t.threads();
            // Every f32 of `a` and every f16 of `b` has exactly one owner.
            let a_per = if key.a_vec4 { 4 } else { 1 };
            assert_eq!(na * a_per * lanes, t.bm * t.bk, "{key:?}");
            // A vec4<u32> register holds 8 halves; a u32 register stands
            // for one element (the other half of its word is a neighbour's).
            let b_per = if key.b_vec8 { 8 } else { 1 };
            assert_eq!(nb * b_per * lanes, t.bn * t.bk, "{key:?}");
            let src = source(key);
            assert!(src.contains(&format!("var pa{} ", na - 1)), "{key:?}");
            assert!(!src.contains(&format!("var pa{na} ")), "{key:?}");
            assert!(src.contains(&format!("var pb{} ", nb - 1)), "{key:?}");
            assert!(!src.contains(&format!("var pb{nb} ")), "{key:?}");
        }
    }

    #[test]
    fn multiplies_stand_between_a_prefetch_and_its_store() {
        for key in keys() {
            let src = source(key);
            let start = src
                .find("for (var p")
                .or_else(|| src.find("for (var t"))
                .expect("a K loop");
            let body = &src[start..];
            let load = body.find("pa0 = ").expect("a load into pa0");
            let mul = body.find("coopMultiplyAdd").expect("a multiply");
            let store = body.find("let v = pa0;").expect("a store of pa0");
            assert!(load < mul && mul < store, "{key:?}");
        }
    }

    #[test]
    fn pipelined_double_buffering_needs_one_barrier_per_step() {
        let key = CoopKey {
            tile: CoopTile::T128X128_W64X32_K32_PF,
            transposed_b: false,
            a_vec4: true,
            b_vec8: true,
            loader: LoaderKind::Plain,
            epilogue: EpilogueKind::AlphaBeta,
        };
        let src = source(key);
        let body = &src[src.find("for (var p").expect("the pair loop")..];
        let loop_end = body.find("if (odd == 1u)").expect("the odd step");
        // Two steps per iteration, one barrier each.
        assert_eq!(body[..loop_end].matches("workgroupBarrier()").count(), 2);
    }

    /// Every pipelined tile with every fused loader, in each layout and
    /// staging-load variant, under the plain epilogue.
    fn fused_keys() -> Vec<CoopKey> {
        let loaders = [
            LoaderKind::Sources { count: 2 },
            LoaderKind::Sources { count: 3 },
            LoaderKind::Sources { count: 4 },
            LoaderKind::SwiGlu,
        ];
        let mut keys = Vec::new();
        for loader in loaders {
            for tile in CoopTile::PIPELINED {
                for transposed_b in [false, true] {
                    for a_vec4 in [false, true] {
                        for b_vec8 in [false, true] {
                            keys.push(CoopKey {
                                tile,
                                transposed_b,
                                a_vec4,
                                b_vec8,
                                loader,
                                epilogue: EpilogueKind::AlphaBeta,
                            });
                        }
                    }
                }
            }
        }
        keys
    }

    #[test]
    fn every_fused_pipelined_kernel_validates() {
        for key in fused_keys() {
            let src = source(key);
            validate(key, &src, naga::valid::Capabilities::empty());
            validate(
                key,
                &immediate_source(&src),
                naga::valid::Capabilities::IMMEDIATES,
            );
        }
    }

    #[test]
    fn fused_loaders_run_the_register_prefetch_loop() {
        for key in fused_keys() {
            let src = source(key);
            let start = src
                .find("for (var p")
                .or_else(|| src.find("for (var t"))
                .expect("a K loop");
            let body = &src[start..];
            let load = body.find("pa0 = ").expect("a load into pa0");
            let mul = body.find("coopMultiplyAdd").expect("a multiply");
            let store = body.find("let v = pa0").expect("a store of pa0");
            assert!(load < mul && mul < store, "{key:?}");
            let (na, _) = super::prefetch_counts(key);
            let swiglu = key.loader == LoaderKind::SwiGlu;
            assert_eq!(src.contains("var pu0 "), swiglu, "{key:?}");
            assert_eq!(
                src.contains(&format!("var pu{} ", na - 1)),
                swiglu,
                "{key:?}"
            );
            assert!(!src.contains(&format!("var pu{na} ")), "{key:?}");
        }
    }

    #[test]
    fn a_sources_step_reads_one_binding_and_falls_back_across_a_boundary() {
        let key = |count| CoopKey {
            tile: CoopTile::T64X64_W32X32_K32_PF,
            transposed_b: true,
            a_vec4: true,
            b_vec8: true,
            loader: LoaderKind::Sources { count },
            epilogue: EpilogueKind::AlphaBeta,
        };
        let src = source(key(3));
        let body = &src[src.find("for (var p").expect("the pair loop")..];
        // One chain over the three sources, then the per-element fallback.
        assert!(body.contains("if (k0 + BK <= params[2].x) {"));
        assert!(body.contains("else if (k0 >= params[2].x && k0 + BK <= params[2].y) {"));
        assert!(body.contains("else if (k0 >= params[2].y) {"));
        assert!(body.contains("a_load(gr, gk)"));
        // The loop reads each binding with the plain vec4 index form.
        for name in ["a", "a1", "a2"] {
            assert!(body.contains(&format!("= {name}[(gr * (")), "{name}");
        }
        // Plain and SwiGlu kernels carry no source chain.
        let plain = CoopKey {
            loader: LoaderKind::Plain,
            ..key(3)
        };
        assert!(!source(plain).contains("params[2]"));
    }
}
