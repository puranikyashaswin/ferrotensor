//! Gradient checks and GPU/CPU parity for every differentiable op.
//!
//! Each [`Case`] is checked on the CPU device against central finite
//! differences. With `FERROTENSOR_RUN_GPU_TESTS=1` the same cases also run on the
//! GPU and must match the CPU forward values and gradients.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::{no_grad, Device, Tensor};

/// Input generator for one argument of a case.
#[derive(Clone, Copy)]
pub(crate) enum Init {
    /// Uniform in `[lo, hi)`.
    Uniform(f32, f32),
    /// Uniform in `[lo, hi)`, pushed at least 0.2 away from zero (keeps
    /// finite differences off kinks such as relu and abs at 0).
    AwayFromZero(f32, f32),
    /// Shuffled values 0.25 apart (no near-ties for max/min).
    Distinct,
}

pub(crate) struct Case {
    pub name: &'static str,
    pub inputs: Vec<(Vec<usize>, Init)>,
    pub f: fn(&[Tensor]) -> Tensor,
}

fn case(name: &'static str, inputs: &[(&[usize], Init)], f: fn(&[Tensor]) -> Tensor) -> Case {
    Case {
        name,
        inputs: inputs.iter().map(|(s, i)| (s.to_vec(), *i)).collect(),
        f,
    }
}

const U: Init = Init::Uniform(-2.0, 2.0);
const POS: Init = Init::Uniform(0.5, 2.0);
const NZ: Init = Init::AwayFromZero(-2.0, 2.0);
const D: Init = Init::Distinct;

pub(crate) fn cases() -> Vec<Case> {
    vec![
        case("add", &[(&[2, 3], U), (&[2, 3], U)], |x| x[0].add(&x[1])),
        case("add_broadcast_row", &[(&[2, 3], U), (&[3], U)], |x| {
            x[0].add(&x[1])
        }),
        case("add_broadcast_scalar", &[(&[], U), (&[2, 3], U)], |x| {
            x[0].add(&x[1])
        }),
        case("sub_broadcast", &[(&[2, 1], U), (&[1, 3], U)], |x| {
            x[0].sub(&x[1])
        }),
        case("mul_broadcast", &[(&[4, 1, 3], U), (&[2, 1], U)], |x| {
            x[0].mul(&x[1])
        }),
        case("mul_self", &[(&[5], U)], |x| x[0].mul(&x[0])),
        case("div", &[(&[2, 3], U), (&[2, 3], POS)], |x| x[0].div(&x[1])),
        case("div_broadcast", &[(&[3], U), (&[2, 3], POS)], |x| {
            x[0].div(&x[1])
        }),
        case("maximum", &[(&[6], D), (&[6], U)], |x| x[0].maximum(&x[1])),
        case("minimum", &[(&[2, 3], D), (&[3], U)], |x| {
            x[0].minimum(&x[1])
        }),
        case("add_scalar", &[(&[4], U)], |x| x[0].add_scalar(1.5)),
        case("sub_scalar", &[(&[4], U)], |x| x[0].sub_scalar(0.5)),
        case("rsub_scalar", &[(&[4], U)], |x| x[0].rsub_scalar(2.0)),
        case("mul_scalar", &[(&[4], U)], |x| x[0].mul_scalar(-3.0)),
        case("div_scalar", &[(&[4], U)], |x| x[0].div_scalar(4.0)),
        case("pow_cube_negative_base", &[(&[5], NZ)], |x| {
            x[0].pow_scalar(3.0)
        }),
        case("pow_half", &[(&[5], POS)], |x| x[0].pow_scalar(0.5)),
        case("neg", &[(&[4], U)], |x| x[0].neg()),
        case("relu", &[(&[8], NZ)], |x| x[0].relu()),
        case("leaky_relu", &[(&[8], NZ)], |x| x[0].leaky_relu(0.1)),
        case("silu", &[(&[8], U)], |x| x[0].silu()),
        case("gelu", &[(&[8], U)], |x| x[0].gelu()),
        case("sigmoid", &[(&[8], U)], |x| x[0].sigmoid()),
        case("tanh", &[(&[8], U)], |x| x[0].tanh()),
        case("exp", &[(&[6], U)], |x| x[0].exp()),
        case("log", &[(&[6], POS)], |x| x[0].log()),
        case("abs", &[(&[6], NZ)], |x| x[0].abs()),
        case("sqrt", &[(&[6], POS)], |x| x[0].sqrt()),
        case("rsqrt", &[(&[6], POS)], |x| x[0].rsqrt()),
        case("recip", &[(&[6], POS)], |x| x[0].recip()),
        case("square", &[(&[6], U)], |x| x[0].square()),
        case("sin", &[(&[6], U)], |x| x[0].sin()),
        case("cos", &[(&[6], U)], |x| x[0].cos()),
        case("clamp", &[(&[3, 4], U)], |x| x[0].clamp(-1.0, 1.0)),
        case("sum", &[(&[2, 3], U)], |x| x[0].sum()),
        case("mean", &[(&[2, 3], U)], |x| x[0].mean()),
        case("sum_dim0", &[(&[2, 3, 4], U)], |x| x[0].sum_dim(0, false)),
        case("sum_dim1_keep", &[(&[2, 3, 4], U)], |x| {
            x[0].sum_dim(1, true)
        }),
        case("sum_dim_last", &[(&[2, 3, 4], U)], |x| {
            x[0].sum_dim(-1, false)
        }),
        case("mean_dim", &[(&[3, 4], U)], |x| x[0].mean_dim(-1, true)),
        case("var_dim", &[(&[3, 4], U)], |x| x[0].var_dim(1, false)),
        case("max", &[(&[7], D)], |x| x[0].max()),
        case("min", &[(&[2, 3], D)], |x| x[0].min()),
        case("max_dim", &[(&[3, 4], D)], |x| x[0].max_dim(1, false)),
        case("min_dim_keep", &[(&[3, 4], D)], |x| x[0].min_dim(0, true)),
        case("reshape", &[(&[2, 6], U)], |x| {
            x[0].reshape(&[3, 4]).square()
        }),
        case("unsqueeze_squeeze", &[(&[3], U)], |x| {
            x[0].unsqueeze(0).mul(&x[0].unsqueeze(1)).squeeze()
        }),
        case("permute", &[(&[2, 3, 4], U)], |x| {
            x[0].permute(&[2, 0, 1])
                .mul(&x[0].permute(&[2, 0, 1]).sin())
        }),
        case("transpose", &[(&[3, 4], U)], |x| x[0].transpose().exp()),
        case("expand", &[(&[3, 1], U)], |x| {
            x[0].expand(&[2, 3, 4]).square()
        }),
        case("narrow", &[(&[4, 5], U)], |x| x[0].narrow(1, 1, 3).square()),
        case("cat", &[(&[2, 3], U), (&[2, 2], U)], |x| {
            Tensor::cat(&[x[0].clone(), x[1].clone()], 1).square()
        }),
        case("stack", &[(&[2, 3], U), (&[2, 3], U)], |x| {
            Tensor::stack(&[x[0].clone(), x[1].sin()], 1)
        }),
        case("split", &[(&[5, 2], U)], |x| {
            let parts = x[0].split(2, 0);
            parts[0].sum().add(&parts[2].square().sum())
        }),
        case("index_select_rows", &[(&[5, 3], U)], |x| {
            x[0].index_select(0, &[4, 0, 4, 2]).square()
        }),
        case("index_select_cols", &[(&[2, 4, 3], U)], |x| {
            x[0].index_select(1, &[1, 1, 3]).exp()
        }),
        case("matmul", &[(&[3, 4], U), (&[4, 2], U)], |x| {
            x[0].matmul(&x[1])
        }),
        case("matmul_nd_by_2d", &[(&[2, 3, 4], U), (&[4, 5], U)], |x| {
            x[0].matmul(&x[1])
        }),
        case("matmul_batched", &[(&[2, 3, 4], U), (&[2, 4, 2], U)], |x| {
            x[0].matmul(&x[1])
        }),
        case(
            "matmul_2d_by_batched",
            &[(&[3, 4], U), (&[2, 4, 2], U)],
            |x| x[0].matmul(&x[1]),
        ),
        case("softmax_last", &[(&[3, 5], U)], |x| x[0].softmax(-1)),
        case("softmax_first", &[(&[3, 5], U)], |x| x[0].softmax(0)),
        case("log_softmax", &[(&[2, 3, 4], U)], |x| x[0].log_softmax(1)),
        case("where_cond", &[(&[2, 3], U), (&[2, 3], U)], |x| {
            let cond = x[0].gt(&x[1]);
            Tensor::where_cond(&cond, &x[0].square(), &x[1].exp())
        }),
        case("masked_fill", &[(&[3, 3], U)], |x| {
            let mask = Tensor::from_data(
                &[0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                &[3, 3],
                &x[0].device,
            );
            x[0].masked_fill(&mask, -1e9).softmax(-1)
        }),
        case("tril", &[(&[2, 3, 4], U)], |x| x[0].tril(0).exp()),
        case("tril_negative", &[(&[4, 3], U)], |x| x[0].tril(-1).square()),
        case("triu", &[(&[3, 5], U)], |x| x[0].triu(1).exp()),
        case("triu_negative", &[(&[2, 4, 4], U)], |x| x[0].triu(-2).sin()),
        case("shared_subexpression", &[(&[4], U), (&[4], U)], |x| {
            let h = x[0].mul(&x[1]).tanh();
            h.mul(&h).add(&h.mul(&x[0]))
        }),
        case(
            "mlp",
            &[(&[4, 3], U), (&[3, 5], U), (&[5], U), (&[5, 2], U)],
            |x| {
                x[0].matmul(&x[1])
                    .add(&x[2])
                    .gelu()
                    .matmul(&x[3])
                    .log_softmax(-1)
            },
        ),
    ]
}

pub(crate) fn make_inputs(case: &Case, device: &Device) -> Vec<Tensor> {
    let mut rng = StdRng::seed_from_u64(0x5eed ^ case.name.len() as u64);
    case.inputs
        .iter()
        .map(|(shape, init)| {
            let n: usize = shape.iter().product();
            if let Init::Distinct = init {
                let mut data: Vec<f32> = (0..n).map(|i| i as f32 * 0.25 - 1.0).collect();
                for i in (1..n).rev() {
                    data.swap(i, rng.gen_range(0..=i));
                }
                return Tensor::from_data(&data, shape, device);
            }
            let data: Vec<f32> = (0..n)
                .map(|_| match *init {
                    Init::Uniform(lo, hi) => rng.gen_range(lo..hi),
                    Init::Distinct => unreachable!(),
                    Init::AwayFromZero(lo, hi) => {
                        let v: f32 = rng.gen_range(lo..hi);
                        if v.abs() < 0.2 {
                            v + 0.4 * v.signum()
                        } else {
                            v
                        }
                    }
                })
                .collect();
            Tensor::from_data(&data, shape, device)
        })
        .collect()
}

/// Fixed random weights so the checked scalar `sum(w * f(x))` exercises
/// every output element with a different coefficient.
fn projection(shape: &[usize], device: &Device) -> Tensor {
    let mut rng = StdRng::seed_from_u64(7);
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|_| rng.gen_range(0.5..1.5)).collect();
    Tensor::from_data(&data, shape, device)
}

/// Analytic gradients of `sum(w * f(inputs))` for each input.
fn analytic_grads(case: &Case, inputs: &[Tensor]) -> (Vec<f32>, Vec<Vec<f32>>) {
    let leaves: Vec<Tensor> = inputs.iter().map(|t| t.detach().requires_grad_()).collect();
    let out = (case.f)(&leaves);
    let w = projection(&out.shape, &out.device);
    out.mul(&w).sum().backward();
    let grads = leaves
        .iter()
        .map(|leaf| {
            leaf.grad()
                .unwrap_or_else(|| panic!("{}: no gradient reached an input", case.name))
                .to_vec()
        })
        .collect();
    (out.to_vec(), grads)
}

/// Check every case's gradients against central finite differences.
pub(crate) fn check_gradients(case: &Case) {
    let device = Device::cpu();
    let inputs = make_inputs(case, &device);
    let (_, analytic) = analytic_grads(case, &inputs);
    let w_shape = no_grad(|| (case.f)(&inputs)).shape;
    let w = projection(&w_shape, &device);
    let loss = |xs: &[Tensor]| no_grad(|| (case.f)(xs).mul(&w).sum().item()) as f64;

    let h = 1e-2f32;
    for (i, input) in inputs.iter().enumerate() {
        let base = input.to_vec();
        for j in 0..base.len() {
            let eval = |delta: f32| {
                let mut data = base.clone();
                data[j] += delta;
                let mut xs = inputs.clone();
                xs[i] = Tensor::from_data(&data, &input.shape, &device);
                loss(&xs)
            };
            let numeric = (eval(h) - eval(-h)) / (2.0 * h as f64);
            let got = analytic[i][j] as f64;
            let tol = 2e-3 + 2e-2 * numeric.abs();
            assert!(
                (got - numeric).abs() <= tol,
                "{}: d/d(input {i})[{j}] analytic {got:.6} vs numeric {numeric:.6}",
                case.name
            );
        }
    }
}

#[test]
fn gradients_match_finite_differences() {
    for case in cases() {
        check_gradients(&case);
    }
}

/// With `FERROTENSOR_RUN_GPU_TESTS=1`, check that each case's forward values and
/// gradients on the GPU match the CPU.
pub(crate) fn check_gpu_parity(cases: &[Case]) {
    if std::env::var("FERROTENSOR_RUN_GPU_TESTS").as_deref() != Ok("1") {
        eprintln!("skipping GPU parity; set FERROTENSOR_RUN_GPU_TESTS=1 to enable");
        return;
    }
    let gpu = Device::gpu().expect("FERROTENSOR_RUN_GPU_TESTS=1 but no GPU is available");
    let cpu = Device::cpu();
    let close = |a: &[f32], b: &[f32], what: &str| {
        assert_eq!(a.len(), b.len(), "{what}: length");
        for (k, (x, y)) in a.iter().zip(b).enumerate() {
            assert!(
                (x - y).abs() <= 1e-4 + 1e-3 * y.abs(),
                "{what}[{k}]: gpu {x} vs cpu {y}"
            );
        }
    };
    for case in cases {
        let (cpu_out, cpu_grads) = analytic_grads(case, &make_inputs(case, &cpu));
        let (gpu_out, gpu_grads) = analytic_grads(case, &make_inputs(case, &gpu));
        close(&gpu_out, &cpu_out, &format!("{} forward", case.name));
        for (i, (g, c)) in gpu_grads.iter().zip(&cpu_grads).enumerate() {
            close(g, c, &format!("{} grad {i}", case.name));
        }
    }
}

#[test]
fn gpu_matches_cpu() {
    check_gpu_parity(&cases());
}

// ============================================================================
// AUTOGRAD ENGINE BEHAVIOUR
// ============================================================================

fn cpu_leaf(data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_data(data, shape, &Device::cpu()).requires_grad_()
}

#[test]
fn gradients_accumulate_across_backward_calls() {
    let w = cpu_leaf(&[1.0, 2.0], &[2]);
    w.mul_scalar(3.0).sum().backward();
    w.mul_scalar(3.0).sum().backward();
    assert_eq!(w.grad().unwrap().to_vec(), vec![6.0, 6.0]);
    w.zero_grad();
    assert!(w.grad().is_none());
}

#[test]
fn leaf_grads_do_not_alias() {
    // add hands the same gradient to both inputs; each leaf needs its own copy.
    let a = cpu_leaf(&[1.0], &[1]);
    let b = cpu_leaf(&[2.0], &[1]);
    a.add(&b).sum().backward();
    a.grad().unwrap().copy_from_slice(&[10.0]);
    assert_eq!(b.grad().unwrap().to_vec(), vec![1.0]);
}

#[test]
fn no_grad_records_nothing() {
    let w = cpu_leaf(&[1.0], &[1]);
    let y = no_grad(|| w.mul_scalar(2.0));
    assert!(!y.requires_grad);
    assert!(y.is_leaf());
    assert!(crate::tensor::is_grad_enabled());
}

#[test]
fn detach_stops_gradient() {
    let w = cpu_leaf(&[3.0], &[1]);
    let y = w.mul(&w.detach());
    y.sum().backward();
    assert_eq!(w.grad().unwrap().to_vec(), vec![3.0]);
}

#[test]
fn backward_with_explicit_seed() {
    let w = cpu_leaf(&[1.0, 2.0], &[2]);
    let seed = Tensor::from_data(&[0.5, -1.0], &[2], &Device::cpu());
    w.square().backward_with(&seed);
    assert_eq!(w.grad().unwrap().to_vec(), vec![1.0, -4.0]);
}

#[test]
fn deep_graphs_do_not_overflow_the_stack() {
    let w = cpu_leaf(&[1.0], &[1]);
    let mut y = w.clone();
    for _ in 0..20_000 {
        y = y.add_scalar(0.0);
    }
    y.sum().backward();
    assert_eq!(w.grad().unwrap().to_vec(), vec![1.0]);
}

#[test]
fn ops_without_grad_inputs_build_no_graph() {
    let x = Tensor::ones(&[2], &Device::cpu());
    let y = x.exp().add(&x);
    assert!(y.is_leaf());
    assert_eq!(y.grad_fn_name(), None);
}

#[test]
fn reductions_return_scalars() {
    let x = Tensor::from_data(&[1.0, 5.0, 3.0, 2.0], &[2, 2], &Device::cpu());
    assert_eq!(x.sum().shape, Vec::<usize>::new());
    assert_eq!(x.sum().item(), 11.0);
    assert_eq!(x.max().item(), 5.0);
    assert_eq!(x.argmax(), 1);
    assert_eq!(x.argmin(), 0);
    assert_eq!(x.argmax_dim(1).to_vec(), vec![1.0, 0.0]);
    assert_eq!(x.mean_dim(0, false).to_vec(), vec![2.0, 3.5]);
}

#[test]
fn chunk_and_cat_round_trip() {
    let x = Tensor::arange(0.0, 10.0, 1.0, &Device::cpu()).reshape(&[5, 2]);
    let parts = x.chunk(2, 0);
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0].shape, vec![3, 2]);
    assert_eq!(parts[1].shape, vec![2, 2]);
    assert_eq!(Tensor::cat(&parts, 0).to_vec(), x.to_vec());
}

#[test]
fn operators_and_constructors() {
    let d = Device::cpu();
    let a = Tensor::from_data(&[1.0, 2.0], &[2], &d);
    let b = Tensor::full(&[2], 4.0, &d);
    assert_eq!((&a + &b).to_vec(), vec![5.0, 6.0]);
    assert_eq!((&b / 2.0).to_vec(), vec![2.0, 2.0]);
    assert_eq!((-&a).to_vec(), vec![-1.0, -2.0]);
    assert_eq!(Tensor::arange(0.0, 1.0, 0.25, &d).numel(), 4);
    assert_eq!(
        Tensor::linspace(0.0, 1.0, 3, &d).to_vec(),
        vec![0.0, 0.5, 1.0]
    );
    assert_eq!(Tensor::eye(2, &d).to_vec(), vec![1.0, 0.0, 0.0, 1.0]);
    assert_eq!(format!("{}", Tensor::scalar(1.5, &d)), "tensor(1.5000)");
}

#[test]
fn assign_updates_every_clone() {
    let d = Device::cpu();
    let p = Tensor::zeros(&[2], &d);
    let alias = p.clone();
    p.assign(&Tensor::ones(&[2], &d));
    assert_eq!(alias.to_vec(), vec![1.0, 1.0]);
}

#[test]
fn tril_and_triu_pick_triangles() {
    let x = Tensor::arange(1.0, 10.0, 1.0, &Device::cpu()).reshape(&[3, 3]);
    assert_eq!(
        x.tril(0).to_vec(),
        vec![1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0]
    );
    assert_eq!(
        x.triu(1).to_vec(),
        vec![0.0, 2.0, 3.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0]
    );
    assert_eq!(
        x.tril(-1).to_vec(),
        vec![0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 7.0, 8.0, 0.0]
    );
    // tril(k) + triu(k + 1) splits the matrix, also for non-square shapes.
    let y = Tensor::randn(&[2, 3, 5], &Device::cpu());
    let parts = y.tril(1).add(&y.triu(2));
    assert_eq!(parts.to_vec(), y.to_vec());
}

#[test]
#[should_panic(expected = "at least 2 dims")]
fn tril_needs_a_matrix() {
    Tensor::ones(&[3], &Device::cpu()).tril(0);
}

#[test]
#[should_panic(expected = "cannot be broadcast")]
fn incompatible_shapes_panic() {
    let d = Device::cpu();
    Tensor::zeros(&[2, 3], &d).add(&Tensor::zeros(&[2], &d));
}

// ============================================================================
// GPU REDUCTIONS
// ============================================================================

/// The GPU when `FERROTENSOR_RUN_GPU_TESTS=1`.
fn test_gpu() -> Option<Device> {
    if std::env::var("FERROTENSOR_RUN_GPU_TESTS").as_deref() != Ok("1") {
        eprintln!("skipping GPU reductions; set FERROTENSOR_RUN_GPU_TESTS=1 to enable");
        return None;
    }
    Some(Device::gpu().expect("FERROTENSOR_RUN_GPU_TESTS=1 but no GPU is available"))
}

#[test]
fn full_reductions_match_cpu_on_gpu() {
    let Some(gpu) = test_gpu() else { return };
    let cpu = Device::cpu();
    let mut rng = StdRng::seed_from_u64(11);
    for n in [1, 255, 256, 257, 65_537, 1_000_003, 4096 * 4096] {
        let data: Vec<f32> = (0..n).map(|_| rng.gen_range(-2.0..2.0)).collect();
        let (c, g) = (
            Tensor::from_data(&data, &[n], &cpu),
            Tensor::from_data(&data, &[n], &gpu),
        );
        // max and min do not depend on the order of the elements.
        assert_eq!(g.max().item(), c.max().item(), "max of {n}");
        assert_eq!(g.min().item(), c.min().item(), "min of {n}");
        // The GPU adds in a different order than the CPU's left-to-right loop;
        // each order's rounding error is a few f32 epsilons per element of
        // sum(|x|) at most, so allow 1e-6 of sum(|x|) between them.
        let abs: f64 = data.iter().map(|v| v.abs() as f64).sum();
        let (gs, cs) = (g.sum().item(), c.sum().item());
        assert!(
            ((gs - cs) as f64).abs() <= 1e-6 * abs,
            "sum of {n}: gpu {gs} vs cpu {cs}"
        );
    }
}

/// Values with many ties, and NaN at `nans`.
fn tied(n: usize, nans: &[usize]) -> Vec<f32> {
    let mut data: Vec<f32> = (0..n).map(|i| ((i * 7919) % 13) as f32 - 6.0).collect();
    for &i in nans {
        data[i] = f32::NAN;
    }
    data
}

#[test]
fn arg_reductions_match_cpu_on_gpu() {
    let Some(gpu) = test_gpu() else { return };
    let cpu = Device::cpu();
    let cases: [(&[usize], &[usize]); 6] = [
        (&[1_000_003], &[]),
        (&[1_000_003], &[500_000, 900_000]),
        (&[3, 5, 4], &[]),
        (&[3, 5, 4], &[7, 9, 30, 31, 59]),
        (&[2, 100_000], &[150_000]),
        (&[70_000, 3], &[4, 5, 100]),
    ];
    for (shape, nans) in cases {
        let data = tied(shape.iter().product(), nans);
        let (c, g) = (
            Tensor::from_data(&data, shape, &cpu),
            Tensor::from_data(&data, shape, &gpu),
        );
        assert_eq!(g.argmax(), c.argmax(), "argmax {shape:?}");
        assert_eq!(g.argmin(), c.argmin(), "argmin {shape:?}");
        for d in 0..shape.len() {
            for smallest in [false, true] {
                assert_eq!(
                    g.arg_reduce_raw(d, smallest).to_vec(),
                    c.arg_reduce_raw(d, smallest).to_vec(),
                    "arg_reduce {shape:?} dim {d} smallest {smallest}"
                );
            }
        }
    }
}

#[test]
fn gpu_argmax_index_is_exact_past_f32_integers() {
    let Some(gpu) = test_gpu() else { return };
    // 2^24 + 1 is the first integer f32 cannot hold.
    let n = (1 << 24) + 2;
    let mut data = vec![0.0; n];
    data[n - 1] = 1.0;
    data[n - 2] = -1.0;
    let x = Tensor::from_vec(data, &[n], &gpu);
    assert_eq!(x.argmax(), n - 1);
    assert_eq!(x.argmin(), n - 2);
}
