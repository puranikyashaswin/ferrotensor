//! Finite-difference gradient checks for every rusty-autograd op on the GPU.
//!
//! Needs a GPU: runs only with `RUSTY_RUN_GPU_TESTS=1` (use
//! `--test-threads=1`; creating several wgpu devices at once can fail).

use std::sync::Arc;

use rusty_autograd::{checkpoint_single, Tensor};
use rusty_backend::{ComputeEngine, UnifiedTensor, WgpuContext};

type Op = Box<dyn Fn(&[Tensor], &ComputeEngine) -> Tensor>;

struct Case {
    name: &'static str,
    inputs: Vec<(Vec<f32>, Vec<usize>)>,
    f: Op,
}

fn values(n: usize, seed: u32, away_from_zero: bool) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let v =
                (((i as u32 + 1).wrapping_mul(2_654_435_761) ^ seed) % 1000) as f32 / 500.0 - 1.0;
            if away_from_zero && v.abs() < 0.2 {
                v + 0.4 * v.signum()
            } else {
                v
            }
        })
        .collect()
}

fn input(shape: &[usize], seed: u32) -> (Vec<f32>, Vec<usize>) {
    (values(shape.iter().product(), seed, true), shape.to_vec())
}

fn case(
    name: &'static str,
    inputs: Vec<(Vec<f32>, Vec<usize>)>,
    f: impl Fn(&[Tensor], &ComputeEngine) -> Tensor + 'static,
) -> Case {
    Case {
        name,
        inputs,
        f: Box::new(f),
    }
}

fn cases() -> Vec<Case> {
    vec![
        case(
            "matmul",
            vec![input(&[3, 4], 1), input(&[4, 2], 2)],
            |x, e| Tensor::matmul(&x[0], &x[1], e),
        ),
        case("add", vec![input(&[6], 1), input(&[6], 2)], |x, e| {
            Tensor::add(&x[0], &x[1], e)
        }),
        case("sub", vec![input(&[6], 1), input(&[6], 2)], |x, e| {
            Tensor::sub(&x[0], &x[1], e)
        }),
        case("mul", vec![input(&[6], 1), input(&[6], 2)], |x, e| {
            Tensor::mul(&x[0], &x[1], e)
        }),
        case("silu", vec![input(&[6], 3)], |x, e| Tensor::silu(&x[0], e)),
        case("relu", vec![input(&[6], 3)], |x, e| Tensor::relu(&x[0], e)),
        case("softmax_rows", vec![input(&[2, 5], 4)], |x, e| {
            Tensor::softmax(&x[0], e)
        }),
        // The rms_norm kernel uses eps = 1e-5.
        case(
            "rms_norm",
            vec![input(&[2, 6], 5), input(&[6], 6)],
            |x, e| Tensor::rms_norm(&x[0], &x[1], 0.1, e),
        ),
        case("rope_one_head", vec![input(&[1, 3, 4], 7)], |x, e| {
            Tensor::rope(&x[0], 4, 2, e)
        }),
        case("rope_two_heads", vec![input(&[1, 3, 8], 7)], |x, e| {
            Tensor::rope(&x[0], 4, 2, e)
        }),
        case("mse_loss", vec![input(&[5], 8), input(&[5], 9)], |x, e| {
            Tensor::mse_loss(&x[0], &x[1], e)
        }),
        case("cross_entropy", vec![input(&[7], 10)], |x, e| {
            Tensor::cross_entropy_loss(&x[0], 3, e)
        }),
        case(
            "silu_gate",
            vec![input(&[6], 11), input(&[6], 12)],
            |x, e| Tensor::silu_gate_fused(&x[0], &x[1], e),
        ),
        case("select_token", vec![input(&[3, 5], 13)], |x, _| {
            Tensor::select_token(&x[0], 1)
        }),
        case("reshape", vec![input(&[2, 3], 14)], |x, e| {
            let r = Tensor::reshape(&x[0], &[3, 2]);
            Tensor::mul(&r, &r, e)
        }),
        case(
            "flash_attention_causal",
            vec![
                input(&[1, 4, 2, 8], 16),
                input(&[1, 4, 2, 8], 17),
                input(&[1, 4, 2, 8], 18),
            ],
            |x, e| Tensor::flash_attention(&x[0], &x[1], &x[2], None, true, 0.0, 0, e),
        ),
        case(
            "flash_attention_mask",
            vec![
                input(&[1, 4, 2, 8], 19),
                input(&[1, 4, 2, 8], 20),
                input(&[1, 4, 2, 8], 21),
            ],
            |x, e| {
                // Additive [1, 1, q, k] mask, a constant (not a checked input).
                let mask_values: Vec<f32> = (0..16)
                    .map(|i| if (i / 4 + i % 4) % 3 == 0 { -1e4 } else { 0.0 })
                    .collect();
                let ctx = x[0].ctx.clone();
                let mask = Tensor::new(
                    ctx.clone(),
                    UnifiedTensor::new(&ctx, &mask_values, &[1, 1, 4, 4]),
                );
                Tensor::flash_attention(&x[0], &x[1], &x[2], Some(&mask), false, 0.0, 0, e)
            },
        ),
        case(
            "flash_attention_dropout",
            vec![
                input(&[1, 4, 2, 8], 22),
                input(&[1, 4, 2, 8], 23),
                input(&[1, 4, 2, 8], 24),
            ],
            |x, e| Tensor::flash_attention(&x[0], &x[1], &x[2], None, false, 0.3, 7, e),
        ),
        case("checkpoint", vec![input(&[6], 15)], |x, e| {
            checkpoint_single(
                x[0].clone(),
                |h, eng| Tensor::mul(&Tensor::silu(h, eng), h, eng),
                e,
            )
        }),
    ]
}

fn read(ctx: &WgpuContext, t: &UnifiedTensor) -> Vec<f32> {
    pollster::block_on(t.to_vec(ctx))
}

/// Analytic gradients of `sum(w * f(inputs))`, and the forward output.
fn analytic(
    ctx: &Arc<WgpuContext>,
    engine: &ComputeEngine,
    case: &Case,
    w: &[f32],
) -> Vec<Vec<f32>> {
    let leaves: Vec<Tensor> = case
        .inputs
        .iter()
        .map(|(data, shape)| Tensor::new(ctx.clone(), UnifiedTensor::new(ctx, data, shape)))
        .collect();
    let out = (case.f)(&leaves, engine);
    let seed = UnifiedTensor::new(ctx, w, &out.data.shape);
    out.accumulate_grad(engine, &seed);
    out.backward(engine);
    leaves
        .iter()
        .map(|leaf| match leaf.grad.borrow().as_ref() {
            Some(g) => read(ctx, g),
            None => vec![0.0; leaf.data.shape.iter().product()],
        })
        .collect()
}

fn loss(
    ctx: &Arc<WgpuContext>,
    engine: &ComputeEngine,
    case: &Case,
    inputs: &[(Vec<f32>, Vec<usize>)],
) -> (f64, usize) {
    let tensors: Vec<Tensor> = inputs
        .iter()
        .map(|(data, shape)| Tensor::new(ctx.clone(), UnifiedTensor::new(ctx, data, shape)))
        .collect();
    let out = read(ctx, &(case.f)(&tensors, engine).data);
    let w = values(out.len(), 99, false);
    let total = out.iter().zip(&w).map(|(y, wi)| (*y * *wi) as f64).sum();
    (total, out.len())
}

#[test]
fn gpu_gradients_match_finite_differences() {
    if std::env::var("RUSTY_RUN_GPU_TESTS").as_deref() != Ok("1") {
        eprintln!("skipping GPU grad checks; set RUSTY_RUN_GPU_TESTS=1 to enable");
        return;
    }
    let ctx = Arc::new(pollster::block_on(WgpuContext::new()));
    let engine = ComputeEngine::new(&ctx);
    let h = 1e-2f32;
    let mut failures = Vec::new();
    for case in cases() {
        let (_, n_out) = loss(&ctx, &engine, &case, &case.inputs);
        let w = values(n_out, 99, false);
        let grads = analytic(&ctx, &engine, &case, &w);
        for (i, (data, shape)) in case.inputs.iter().enumerate() {
            for (j, &got) in grads[i].iter().enumerate().take(data.len()) {
                let eval = |delta: f32| {
                    let mut inputs = case.inputs.clone();
                    inputs[i].0[j] += delta;
                    loss(&ctx, &engine, &case, &inputs).0
                };
                let numeric = (eval(h) - eval(-h)) / (2.0 * h as f64);
                let got = got as f64;
                if (got - numeric).abs() > 5e-3 + 2e-2 * numeric.abs() {
                    failures.push(format!(
                        "{} input {i} {shape:?}[{j}]: analytic {got:.5} vs numeric {numeric:.5}",
                        case.name
                    ));
                    break;
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "gradient mismatches:\n{}",
        failures.join("\n")
    );
}
