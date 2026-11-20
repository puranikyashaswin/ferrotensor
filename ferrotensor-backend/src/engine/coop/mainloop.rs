//! The K loop: the 16x16 fragment multiplies of each staged step, and the
//! order of loads, multiplies and barriers around them.

use super::loader::load_step;
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
/// barrier per step separates a pair's reads from its next writes.
pub(super) fn main_loop(key: CoopKey) -> String {
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
