//! The fused loaders and epilogues of the matmul against CPU references.
//!
//! Every case runs twice when the device allows it: on the plain reference
//! kernel, which needs only a GPU, and on the cooperative-matrix kernel, which
//! needs a GPU with f16 cooperative matrices (16x16x16, f32 accumulation),
//! subgroups of 32 threads and `FERROTENSOR_EXPERIMENTAL=1`. Without those the
//! second half is skipped with a message. Run them with:
//!
//! ```text
//! FERROTENSOR_EXPERIMENTAL=1 cargo test -p ferrotensor-backend --test gpu_coop_fused -- --ignored --test-threads=1
//! ```
//!
//! The `a` inputs are rounded to f16 before the CPU sums them, so the only
//! differences left are the order of the f32 sums. The SwiGLU loader rounds
//! `silu(g) * u` to f16 on its way in, which the CPU does not, so those
//! cases take a looser bound.

#![allow(clippy::unwrap_used, clippy::panic, clippy::too_many_arguments)]

use ferrotensor_backend::{
    ComputeEngine, Epilogue, FusedA, FusedOut, FusedSpec, UnifiedTensor, WgpuContext,
};
use std::sync::OnceLock;

struct Gpu {
    ctx: WgpuContext,
    engine: ComputeEngine,
}

fn gpu() -> &'static Gpu {
    static GPU: OnceLock<Gpu> = OnceLock::new();
    GPU.get_or_init(|| {
        let ctx = pollster::block_on(WgpuContext::new());
        let engine = ComputeEngine::new(&ctx);
        Gpu { ctx, engine }
    })
}

#[derive(Clone, Copy, Debug)]
enum Mode {
    Reference,
    Coop,
}

/// The kernels to run: the reference always, the cooperative one when the
/// device has it.
fn modes(g: &Gpu, test: &str) -> Vec<Mode> {
    let mut modes = vec![Mode::Reference];
    if g.engine.coop_matmul_available() {
        modes.push(Mode::Coop);
    } else {
        eprintln!(
            "{test}: the cooperative kernel is skipped, {} has no f16 cooperative matrices \
             with f32 accumulation (or FERROTENSOR_EXPERIMENTAL is not 1)",
            g.ctx.adapter_info()
        );
    }
    modes
}

fn exec(g: &Gpu, mode: Mode, spec: &FusedSpec<'_>) {
    let ran = match mode {
        Mode::Reference => g.engine.matmul_fused_reference_into(&g.ctx, spec),
        Mode::Coop => g.engine.matmul_fused_coop_into(&g.ctx, spec),
    };
    ran.unwrap_or_else(|e| panic!("{mode:?}: {e}"));
}

/// Deterministic pseudo-random values in [-1, 1).
fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn f16_round(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect()
}

/// `a[M, K] @ b[K, N]` in f64.
fn cpu_matmul(a: &[f32], b: &[f32], (m, k, n): (usize, usize, usize)) -> Vec<f64> {
    let mut out = vec![0.0f64; m * n];
    for i in 0..m {
        for p in 0..k {
            let av = f64::from(a[i * k + p]);
            for j in 0..n {
                out[i * n + j] += av * f64::from(b[p * n + j]);
            }
        }
    }
    out
}

fn transpose(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0.0; x.len()];
    for r in 0..rows {
        for c in 0..cols {
            t[c * rows + r] = x[r * cols + c];
        }
    }
    t
}

fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

/// Largest |got - want| over the largest |want|.
fn max_rel_error(got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    let mut max_err = 0.0f64;
    let mut max_ref = 0.0f64;
    for (&g, &w) in got.iter().zip(want) {
        assert!(g.is_finite(), "non-finite output {g}");
        max_err = max_err.max((f64::from(g) - w).abs());
        max_ref = max_ref.max(w.abs());
    }
    max_err / max_ref.max(f64::MIN_POSITIVE)
}

/// Bound for f16 inputs summed in f32 against f64.
const EXACT: f64 = 2e-4;

/// Bound when the kernel rounds `silu(g) * u` to f16 and the CPU does not.
const SWIGLU: f64 = 3e-3;

fn check(label: &str, got: &[f32], want: &[f64], tol: f64) {
    let err = max_rel_error(got, want);
    assert!(err < tol, "{label}: relative error {err} (bound {tol})");
}

fn tensor(g: &Gpu, data: &[f32], shape: &[usize]) -> UnifiedTensor {
    UnifiedTensor::new(&g.ctx, data, shape)
}

/// A finite value that a kernel that only writes must replace.
fn garbage(g: &Gpu, shape: &[usize]) -> UnifiedTensor {
    tensor(g, &vec![7.0; shape.iter().product()], shape)
}

fn read(g: &Gpu, t: &UnifiedTensor) -> Vec<f32> {
    pollster::block_on(t.to_vec(&g.ctx)).unwrap()
}

/// `b[K, N]` as the f16 weight, stored `[N, K]` when `transposed_b`.
fn weight(g: &Gpu, b: &[f32], k: usize, n: usize, transposed_b: bool) -> UnifiedTensor {
    if transposed_b {
        UnifiedTensor::from_f32_as_f16(&g.ctx, &transpose(b, k, n), &[n, k])
    } else {
        UnifiedTensor::from_f32_as_f16(&g.ctx, b, &[k, n])
    }
}

/// The 135M training shapes at batch 1 x 256 tokens, then shapes whose M, N
/// and K are not multiples of 16 (and K not of 4, N not of 8, which take the
/// scalar staging loads).
const SHAPES: &[(usize, usize, usize)] = &[
    (256, 576, 576),
    (256, 576, 1536),
    (64, 16, 16),
    (77, 50, 33),
    (100, 37, 70),
    (130, 576, 200),
    (65, 1000, 17),
];

/// What the product ends with, over the `[M, N]` matrix `base`.
#[derive(Clone, Copy, Debug)]
enum Tail {
    /// `out = alpha * product + beta * out`.
    AlphaBeta(f32, f32),
    /// `out = alpha * product + residual`.
    Residual(f32),
}

impl Tail {
    fn alpha_beta(self) -> (f64, f64) {
        match self {
            Tail::AlphaBeta(alpha, beta) => (f64::from(alpha), f64::from(beta)),
            Tail::Residual(alpha) => (f64::from(alpha), 1.0),
        }
    }
}

/// Runs `a @ op(b)` with `tail` over a random `[M, N]` base and checks it
/// against `alpha * prod + beta * base`, where `prod` is the CPU product.
fn check_tail(
    label: &str,
    g: &Gpu,
    mode: Mode,
    (a, prod): (FusedA<'_>, &[f64]),
    (b, transposed_b): (&UnifiedTensor, bool),
    (m, n): (usize, usize),
    tail: Tail,
    tol: f64,
) {
    let base = rand_vec(m * n, 99);
    let (alpha, beta) = tail.alpha_beta();
    let want: Vec<f64> = prod
        .iter()
        .zip(&base)
        .map(|(&p, &o)| alpha * p + beta * f64::from(o))
        .collect();
    let base_t = tensor(g, &base, &[m, n]);
    let got = match tail {
        Tail::AlphaBeta(alpha, beta) => {
            let spec = FusedSpec {
                a,
                b,
                transposed_b,
                out: FusedOut::AlphaBeta {
                    out: &base_t,
                    epilogue: Epilogue { alpha, beta },
                },
            };
            exec(g, mode, &spec);
            read(g, &base_t)
        }
        Tail::Residual(alpha) => {
            let out = garbage(g, &[m, n]);
            let spec = FusedSpec {
                a,
                b,
                transposed_b,
                out: FusedOut::Residual {
                    out: &out,
                    residual: &base_t,
                    alpha,
                },
            };
            exec(g, mode, &spec);
            assert_eq!(read(g, &base_t), base, "the residual is only read");
            read(g, &out)
        }
    };
    check(&format!("{label} {mode:?} {tail:?}"), &got, &want, tol);
}

const TAILS: [Tail; 4] = [
    Tail::AlphaBeta(1.0, 0.0),
    Tail::AlphaBeta(0.5, 1.0),
    Tail::Residual(1.0),
    Tail::Residual(2.0),
];

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn plain_loader_matches_cpu_with_every_tail() {
    let g = gpu();
    for mode in modes(g, "plain_loader_matches_cpu_with_every_tail") {
        for &(m, k, n) in SHAPES {
            let a = f16_round(&rand_vec(m * k, 1));
            let b = f16_round(&rand_vec(k * n, 2));
            let prod = cpu_matmul(&a, &b, (m, k, n));
            let a_t = tensor(g, &a, &[m, k]);
            for transposed_b in [false, true] {
                let b_t = weight(g, &b, k, n, transposed_b);
                for tail in TAILS {
                    check_tail(
                        &format!("{m}x{k}x{n} bT={transposed_b}"),
                        g,
                        mode,
                        (FusedA::Plain(&a_t), &prod),
                        (&b_t, transposed_b),
                        (m, n),
                        tail,
                        EXACT,
                    );
                }
            }
        }
    }
}

/// Column widths of the four sources of the backward `[dQ | dK | dV | dU]`
/// at the 135M sizes, then ones that no vec4 load can read.
const SOURCE_WIDTHS: &[&[usize]] = &[
    &[576, 192, 192, 48],
    &[40, 24],
    &[16, 8, 4],
    &[13, 7, 5, 3],
    &[33, 17],
    &[1, 1, 1, 1],
];

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn sources_loader_reads_the_matrices_side_by_side() {
    let g = gpu();
    for mode in modes(g, "sources_loader_reads_the_matrices_side_by_side") {
        for (&widths, (m, n)) in SOURCE_WIDTHS.iter().zip([
            (256, 576),
            (77, 33),
            (130, 200),
            (100, 70),
            (65, 17),
            (31, 9),
        ]) {
            let k: usize = widths.iter().sum();
            let parts: Vec<Vec<f32>> = widths
                .iter()
                .enumerate()
                .map(|(i, &w)| f16_round(&rand_vec(m * w, 10 + i as u64)))
                .collect();
            let mut a = vec![0.0f32; m * k];
            let mut start = 0;
            for (part, &w) in parts.iter().zip(widths) {
                for r in 0..m {
                    a[r * k + start..r * k + start + w].copy_from_slice(&part[r * w..(r + 1) * w]);
                }
                start += w;
            }
            let b = f16_round(&rand_vec(k * n, 3));
            let prod = cpu_matmul(&a, &b, (m, k, n));
            let tensors: Vec<UnifiedTensor> = parts
                .iter()
                .zip(widths)
                .map(|(p, &w)| tensor(g, p, &[m, w]))
                .collect();
            let sources: Vec<&UnifiedTensor> = tensors.iter().collect();
            for transposed_b in [false, true] {
                let b_t = weight(g, &b, k, n, transposed_b);
                for tail in TAILS {
                    check_tail(
                        &format!("widths {widths:?} bT={transposed_b}"),
                        g,
                        mode,
                        (FusedA::Sources(&sources), &prod),
                        (&b_t, transposed_b),
                        (m, n),
                        tail,
                        EXACT,
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn swiglu_loader_reads_silu_gate_times_up() {
    let g = gpu();
    for mode in modes(g, "swiglu_loader_reads_silu_gate_times_up") {
        for &(m, k, n) in SHAPES {
            let gu: Vec<f32> = rand_vec(m * 2 * k, 5).iter().map(|v| v * 3.0).collect();
            let h: Vec<f32> = (0..m * k)
                .map(|i| {
                    let (r, c) = (i / k, i % k);
                    let gate = f64::from(gu[r * 2 * k + c]);
                    let up = f64::from(gu[r * 2 * k + k + c]);
                    (silu(gate) * up) as f32
                })
                .collect();
            let b = f16_round(&rand_vec(k * n, 6));
            let prod = cpu_matmul(&h, &b, (m, k, n));
            let gu_t = tensor(g, &gu, &[m, 2 * k]);
            for transposed_b in [false, true] {
                let b_t = weight(g, &b, k, n, transposed_b);
                // The down projection and the residual add that follows it.
                for tail in [Tail::AlphaBeta(1.0, 0.0), Tail::Residual(1.0)] {
                    check_tail(
                        &format!("{m}x{k}x{n} bT={transposed_b}"),
                        g,
                        mode,
                        (FusedA::SwiGlu(&gu_t), &prod),
                        (&b_t, transposed_b),
                        (m, n),
                        tail,
                        SWIGLU,
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn swiglu_backward_epilogue_matches_cpu() {
    let g = gpu();
    for mode in modes(g, "swiglu_backward_epilogue_matches_cpu") {
        for &(m, k, n) in SHAPES {
            // `dy[M, K] @ w[K, N]` is dH over the intermediate width N.
            let dy = f16_round(&rand_vec(m * k, 7));
            let w = f16_round(&rand_vec(k * n, 8));
            let dh = cpu_matmul(&dy, &w, (m, k, n));
            let gu: Vec<f32> = rand_vec(m * 2 * n, 9).iter().map(|v| v * 3.0).collect();
            let mut want = vec![0.0f64; m * 2 * n];
            for r in 0..m {
                for c in 0..n {
                    let gate = f64::from(gu[r * 2 * n + c]);
                    let up = f64::from(gu[r * 2 * n + n + c]);
                    let s = 1.0 / (1.0 + (-gate).exp());
                    let d = dh[r * n + c];
                    want[r * 2 * n + c] = d * up * (s * (1.0 + gate * (1.0 - s)));
                    want[r * 2 * n + n + c] = d * (gate * s);
                }
            }
            let dy_t = tensor(g, &dy, &[m, k]);
            let gu_t = tensor(g, &gu, &[m, 2 * n]);
            for transposed_b in [false, true] {
                let w_t = weight(g, &w, k, n, transposed_b);
                let d_gu = garbage(g, &[m, 2 * n]);
                let spec = FusedSpec {
                    a: FusedA::Plain(&dy_t),
                    b: &w_t,
                    transposed_b,
                    out: FusedOut::SwiGluBackward {
                        d_gu: &d_gu,
                        gu: &gu_t,
                    },
                };
                exec(g, mode, &spec);
                assert_eq!(read(g, &gu_t), gu, "gu is only read");
                check(
                    &format!("{m}x{k}x{n} bT={transposed_b} {mode:?}"),
                    &read(g, &d_gu),
                    &want,
                    1e-3,
                );
            }
        }
    }
}

/// Widths of the Q, K, V and LoRA `U` columns of the fused QKV product at the
/// 135M sizes (9 heads, 3 KV heads of 64, three rank-16 `A` matrices), then
/// odd ones.
const SPLIT_WIDTHS: &[(usize, usize, &[usize])] = &[
    (256, 576, &[576, 192, 192, 48]),
    (77, 50, &[33, 17, 17, 9]),
    (130, 37, &[40, 13, 8, 1]),
];

/// The fused QKV and gate/up products: the columns of one product go to up to
/// four buffers, each as the separate product would hold them, with the bias
/// added. `U` is the last segment: the LoRA `A` fold is columns of `b`.
#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn column_split_equals_separate_products() {
    let g = gpu();
    for mode in modes(g, "column_split_equals_separate_products") {
        for &(m, k, all_widths) in SPLIT_WIDTHS {
            for segments in 1..=4usize {
                let widths = &all_widths[..segments];
                let n: usize = widths.iter().sum();
                let x = f16_round(&rand_vec(m * k, 21));
                let w = f16_round(&rand_vec(k * n, 22));
                let bias = rand_vec(n, 23);
                let prod = cpu_matmul(&x, &w, (m, k, n));
                let x_t = tensor(g, &x, &[m, k]);
                let bias_t = tensor(g, &bias, &[n]);
                for (transposed_b, with_bias, alpha) in [
                    (false, false, 1.0f32),
                    (false, true, 1.0),
                    (true, true, 1.0),
                    (true, false, 0.5),
                ] {
                    let w_t = weight(g, &w, k, n, transposed_b);
                    let outs: Vec<UnifiedTensor> =
                        widths.iter().map(|&wd| garbage(g, &[m, wd])).collect();
                    let refs: Vec<&UnifiedTensor> = outs.iter().collect();
                    let spec = FusedSpec {
                        a: FusedA::Plain(&x_t),
                        b: &w_t,
                        transposed_b,
                        out: FusedOut::Split {
                            outs: &refs,
                            bias: with_bias.then_some(&bias_t),
                            alpha,
                        },
                    };
                    exec(g, mode, &spec);
                    let mut start = 0;
                    for (i, (&wd, out)) in widths.iter().zip(&outs).enumerate() {
                        let mut want = Vec::with_capacity(m * wd);
                        for r in 0..m {
                            for c in 0..wd {
                                let col = start + c;
                                let mut v = f64::from(alpha) * prod[r * n + col];
                                if with_bias {
                                    v += f64::from(bias[col]);
                                }
                                want.push(v);
                            }
                        }
                        check(
                            &format!(
                                "{m}x{k} segments {segments} #{i} bT={transposed_b} \
                                 bias={with_bias} {mode:?}"
                            ),
                            &read(g, out),
                            &want,
                            EXACT,
                        );
                        start += wd;
                    }
                }
            }
        }
    }
}

/// A fused product and the separate ones it replaces agree: QKV from one
/// `x @ [Wq | Wk | Wv]` against three products on the same kernel.
#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn fused_qkv_equals_three_products_on_the_same_kernel() {
    let g = gpu();
    for mode in modes(g, "fused_qkv_equals_three_products_on_the_same_kernel") {
        let (m, k) = (256usize, 576usize);
        let widths = [576usize, 192, 192];
        let n: usize = widths.iter().sum();
        let x = f16_round(&rand_vec(m * k, 31));
        let ws: Vec<Vec<f32>> = widths
            .iter()
            .enumerate()
            .map(|(i, &w)| f16_round(&rand_vec(k * w, 32 + i as u64)))
            .collect();
        let mut joined = vec![0.0f32; k * n];
        for r in 0..k {
            let mut start = 0;
            for (w, &wd) in ws.iter().zip(&widths) {
                joined[r * n + start..r * n + start + wd].copy_from_slice(&w[r * wd..(r + 1) * wd]);
                start += wd;
            }
        }
        let x_t = tensor(g, &x, &[m, k]);
        let joined_t = weight(g, &joined, k, n, false);
        let outs: Vec<UnifiedTensor> = widths.iter().map(|&wd| garbage(g, &[m, wd])).collect();
        let refs: Vec<&UnifiedTensor> = outs.iter().collect();
        let spec = FusedSpec {
            a: FusedA::Plain(&x_t),
            b: &joined_t,
            transposed_b: false,
            out: FusedOut::Split {
                outs: &refs,
                bias: None,
                alpha: 1.0,
            },
        };
        exec(g, mode, &spec);
        for (i, (w, &wd)) in ws.iter().zip(&widths).enumerate() {
            let w_t = weight(g, w, k, wd, false);
            let single = garbage(g, &[m, wd]);
            let spec = FusedSpec {
                a: FusedA::Plain(&x_t),
                b: &w_t,
                transposed_b: false,
                out: FusedOut::Split {
                    outs: &[&single],
                    bias: None,
                    alpha: 1.0,
                },
            };
            exec(g, mode, &spec);
            let want: Vec<f64> = read(g, &single).iter().map(|&v| f64::from(v)).collect();
            check(
                &format!("projection {i} {mode:?}"),
                &read(g, &outs[i]),
                &want,
                EXACT,
            );
        }
    }
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn invalid_specs_are_refused() {
    let g = gpu();
    let (m, k, n) = (32usize, 16usize, 16usize);
    let a = tensor(g, &f16_round(&rand_vec(m * k, 41)), &[m, k]);
    let b = weight(g, &f16_round(&rand_vec(k * n, 42)), k, n, false);
    let out = tensor(g, &vec![0.0; m * n], &[m, n]);
    let run = |spec: FusedSpec<'_>| g.engine.matmul_fused_reference_into(&g.ctx, &spec);
    fn plain<'a>(
        (a, b): (&'a UnifiedTensor, &'a UnifiedTensor),
        out: FusedOut<'a>,
    ) -> FusedSpec<'a> {
        FusedSpec {
            a: FusedA::Plain(a),
            b,
            transposed_b: false,
            out,
        }
    }
    let store = Epilogue {
        alpha: 1.0,
        beta: 0.0,
    };
    run(plain(
        (&a, &b),
        FusedOut::AlphaBeta {
            out: &out,
            epilogue: store,
        },
    ))
    .unwrap();
    // A residual that is the output, an output that is the input.
    assert!(run(plain(
        (&a, &b),
        FusedOut::Residual {
            out: &out,
            residual: &out,
            alpha: 1.0
        }
    ))
    .is_err());
    assert!(run(plain(
        (&a, &b),
        FusedOut::AlphaBeta {
            out: &a,
            epilogue: store
        }
    ))
    .is_err());
    // A split whose widths do not add up to N, and two outputs on one buffer.
    let half = tensor(g, &vec![0.0; m * 4], &[m, 4]);
    assert!(run(plain(
        (&a, &b),
        FusedOut::Split {
            outs: &[&half],
            bias: None,
            alpha: 1.0
        }
    ))
    .is_err());
    let rest = tensor(g, &vec![0.0; m * 12], &[m, 12]);
    assert!(run(plain(
        (&a, &b),
        FusedOut::Split {
            outs: &[&half, &rest],
            bias: None,
            alpha: 1.0
        }
    ))
    .is_ok());
    let eighth = tensor(g, &vec![0.0; m * 8], &[m, 8]);
    assert!(run(plain(
        (&a, &b),
        FusedOut::Split {
            outs: &[&eighth, &eighth],
            bias: None,
            alpha: 1.0
        }
    ))
    .is_err());
    // A bias of the wrong length.
    assert!(run(plain(
        (&a, &b),
        FusedOut::Split {
            outs: &[&out],
            bias: Some(&half),
            alpha: 1.0
        }
    ))
    .is_err());
    // K that does not match, one source, an odd gate/up width, and f32 weights.
    let long_b = weight(g, &f16_round(&rand_vec(8 * n, 43)), 8, n, false);
    let spec = FusedSpec {
        a: FusedA::Plain(&a),
        b: &long_b,
        transposed_b: false,
        out: FusedOut::AlphaBeta {
            out: &out,
            epilogue: store,
        },
    };
    assert!(run(spec).is_err());
    let spec = FusedSpec {
        a: FusedA::Sources(&[&a]),
        b: &b,
        transposed_b: false,
        out: FusedOut::AlphaBeta {
            out: &out,
            epilogue: store,
        },
    };
    assert!(run(spec).is_err());
    let odd = tensor(g, &vec![0.0; m * 15], &[m, 15]);
    let spec = FusedSpec {
        a: FusedA::SwiGlu(&odd),
        b: &b,
        transposed_b: false,
        out: FusedOut::AlphaBeta {
            out: &out,
            epilogue: store,
        },
    };
    assert!(run(spec).is_err());
    let b32 = tensor(g, &rand_vec(k * n, 44), &[k, n]);
    let spec = FusedSpec {
        a: FusedA::Plain(&a),
        b: &b32,
        transposed_b: false,
        out: FusedOut::AlphaBeta {
            out: &out,
            epilogue: store,
        },
    };
    assert!(run(spec).is_err());
}

/// An `a` value past the f16 range is clamped and counted by the loader that
/// staged it, whichever source it is in, and poisons the output tile.
#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn fused_coop_counts_f16_overflow_and_poisons_its_tile() {
    let g = gpu();
    if !g.engine.coop_matmul_available() {
        eprintln!(
            "fused_coop_counts_f16_overflow_and_poisons_its_tile: skipped, no cooperative matrices"
        );
        return;
    }
    // N = 16 is one column of workgroups whatever the tile, so exactly one
    // staging thread loads the value set below.
    let (m, n) = (256usize, 16usize);
    let widths = [32usize, 32];
    let k: usize = widths.iter().sum();
    let b = weight(g, &f16_round(&rand_vec(k * n, 51)), k, n, false);
    let overflows = || pollster::block_on(g.engine.coop_f16_overflows(&g.ctx)).unwrap();
    for big_source in [0usize, 1] {
        let mut parts: Vec<Vec<f32>> = widths
            .iter()
            .enumerate()
            .map(|(i, &w)| f16_round(&rand_vec(m * w, 52 + i as u64)))
            .collect();
        let tensors: Vec<UnifiedTensor> = parts
            .iter()
            .zip(&widths)
            .map(|(p, &w)| tensor(g, p, &[m, w]))
            .collect();
        let sources: Vec<&UnifiedTensor> = tensors.iter().collect();
        let out = garbage(g, &[m, n]);
        let spec = FusedSpec {
            a: FusedA::Sources(&sources),
            b: &b,
            transposed_b: false,
            out: FusedOut::AlphaBeta {
                out: &out,
                epilogue: Epilogue {
                    alpha: 1.0,
                    beta: 0.0,
                },
            },
        };
        let before = overflows();
        g.engine.matmul_fused_coop_into(&g.ctx, &spec).unwrap();
        assert!(read(g, &out).iter().all(|v| v.is_finite()));
        assert_eq!(overflows(), before);
        parts[big_source][3 * widths[big_source] + 5] = 70_000.0;
        let big: Vec<UnifiedTensor> = parts
            .iter()
            .zip(&widths)
            .map(|(p, &w)| tensor(g, p, &[m, w]))
            .collect();
        let big_refs: Vec<&UnifiedTensor> = big.iter().collect();
        let spec = FusedSpec {
            a: FusedA::Sources(&big_refs),
            ..spec
        };
        g.engine.matmul_fused_coop_into(&g.ctx, &spec).unwrap();
        assert_eq!(overflows(), before + 1, "source {big_source}");
        let got = read(g, &out);
        assert!(got[0].is_nan(), "the tile's first element is poisoned");
    }
}
