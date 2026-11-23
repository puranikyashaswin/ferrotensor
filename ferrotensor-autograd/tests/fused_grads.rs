//! Gradients that matmuls add straight into a gradient slot, and the fused
//! LoRA linear op: the same bits as the separate ops, and the values of a
//! CPU reference.
//!
//! These need a real GPU, so they are `#[ignore]`d by default. Run them with:
//!
//! ```text
//! cargo test -p ferrotensor-autograd --test fused_grads -- --ignored --test-threads=1
//! ```

#![allow(clippy::expect_used, clippy::panic, clippy::arc_with_non_send_sync)]

use std::cell::Cell;
use std::sync::Arc;

use ferrotensor_autograd::{AttentionGeometry, AutogradNode, FusedLora, FusedProjections, Tensor};
use ferrotensor_backend::{ComputeEngine, UnifiedTensor, WgpuContext};

fn setup() -> (Arc<WgpuContext>, ComputeEngine) {
    let ctx = Arc::new(pollster::block_on(WgpuContext::new()));
    let engine = ComputeEngine::new(&ctx);
    (ctx, engine)
}

fn host(ctx: &WgpuContext, t: &UnifiedTensor) -> Vec<f32> {
    pollster::block_on(t.to_vec(ctx)).expect("GPU readback")
}

fn grad_of(ctx: &WgpuContext, t: &Tensor) -> Vec<f32> {
    host(ctx, t.grad.borrow().as_ref().expect("a gradient"))
}

/// The address of the GPU buffer behind `t`: clones of a tensor share it, so
/// equal addresses mean the same buffer.
fn buffer_of(t: &UnifiedTensor) -> *const () {
    std::ptr::from_ref(t.buffer()).cast()
}

fn grad_buffer(t: &Tensor) -> *const () {
    buffer_of(t.grad.borrow().as_ref().expect("a gradient"))
}

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

/// Deterministic pseudo-random values in [-1, 1).
fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// `op(a) @ op(b)` in f64 for `a` `[m, k]` (or `[k, m]` when `ta`) and `b`
/// `[k, n]` (or `[n, k]` when `tb`).
fn mm(a: &[f64], b: &[f64], (m, k, n): (usize, usize, usize), ta: bool, tb: bool) -> Vec<f64> {
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            out[i * n + j] = (0..k)
                .map(|p| {
                    let av = if ta { a[p * m + i] } else { a[i * k + p] };
                    let bv = if tb { b[j * k + p] } else { b[p * n + j] };
                    av * bv
                })
                .sum();
        }
    }
    out
}

fn twice(x: &[f64]) -> Vec<f64> {
    x.iter().map(|v| 2.0 * v).collect()
}

fn f64s(x: &[f32]) -> Vec<f64> {
    x.iter().map(|&v| f64::from(v)).collect()
}

#[track_caller]
fn assert_close(name: &str, got: &[f32], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let tol = 1e-4 + 1e-4 * w.abs();
        assert!(
            (f64::from(*g) - w).abs() <= tol,
            "{name}[{i}]: got {g}, want {w}"
        );
    }
}

/// Two LoRA linears that read the same input, summed, with the input's
/// gradient flowing on to a leaf.
struct Layers {
    x: Tensor,
    w: [Tensor; 2],
    a: [Tensor; 2],
    b: [Tensor; 2],
}

const ROWS: usize = 24;
const IN: usize = 40;
const OUT: usize = 72;
const RANK: usize = 8;

impl Layers {
    fn new(ctx: &Arc<WgpuContext>, half_w: bool, train_w: bool) -> Self {
        let leaf = |data: &[f32], shape: &[usize]| {
            Tensor::new(ctx.clone(), UnifiedTensor::new(ctx, data, shape))
        };
        let weight = |seed| {
            // Multiples of 1/64 are exact in f16.
            let w: Vec<f32> = rand_vec(IN * OUT, seed)
                .iter()
                .map(|v| (v * 64.0).round() / 64.0)
                .collect();
            if half_w {
                let data = UnifiedTensor::from_f32_as_f16(ctx, &w, &[IN, OUT]);
                Tensor::frozen(ctx.clone(), data)
            } else if train_w {
                leaf(&w, &[IN, OUT])
            } else {
                Tensor::frozen(ctx.clone(), UnifiedTensor::new(ctx, &w, &[IN, OUT]))
            }
        };
        Self {
            x: leaf(&rand_vec(ROWS * IN, 1), &[ROWS, IN]),
            w: [weight(2), weight(3)],
            a: [
                leaf(&rand_vec(IN * RANK, 4), &[IN, RANK]),
                leaf(&rand_vec(IN * RANK, 5), &[IN, RANK]),
            ],
            b: [
                leaf(&rand_vec(RANK * OUT, 6), &[RANK, OUT]),
                leaf(&rand_vec(RANK * OUT, 7), &[RANK, OUT]),
            ],
        }
    }

    /// `sum_i x @ w_i + scale * (x @ a_i) @ b_i`, fused or with the separate
    /// matmul, scale and add ops.
    fn forward(&self, engine: &ComputeEngine, scale: f32, fused: bool) -> Tensor {
        // The input reaches both layers through an op, so its gradient is
        // summed in a slot before it flows on to the leaf.
        let x = Tensor::scale(&self.x, 1.0, engine);
        let layer = |i: usize| {
            let adapter = (&self.a[i], &self.b[i]);
            if fused {
                return Tensor::lora_matmul(&x, &self.w[i], adapter, scale, engine);
            }
            let base = Tensor::matmul(&x, &self.w[i], engine);
            let mut t = Tensor::matmul(&x, &self.a[i], engine);
            if scale != 1.0 {
                t = Tensor::scale(&t, scale, engine);
            }
            Tensor::add(&base, &Tensor::matmul(&t, &self.b[i], engine), engine)
        };
        Tensor::add(&layer(0), &layer(1), engine)
    }

    fn leaves(&self) -> Vec<(&'static str, &Tensor)> {
        let mut leaves = vec![
            ("x", &self.x),
            ("a0", &self.a[0]),
            ("a1", &self.a[1]),
            ("b0", &self.b[0]),
            ("b1", &self.b[1]),
        ];
        if self.w[0].requires_grad {
            leaves.extend([("w0", &self.w[0]), ("w1", &self.w[1])]);
        }
        leaves
    }
}

/// Runs two forward and backward passes (two micro-batches) and returns the
/// output of the last and every leaf's gradient.
fn two_passes(
    ctx: &Arc<WgpuContext>,
    engine: &ComputeEngine,
    layers: &Layers,
    scale: f32,
    fused: bool,
) -> (Vec<f32>, Vec<Vec<f32>>) {
    let seed = UnifiedTensor::new(ctx, &rand_vec(ROWS * OUT, 8), &[ROWS, OUT]);
    let mut out = Vec::new();
    for _ in 0..2 {
        let y = layers.forward(engine, scale, fused);
        *y.grad.borrow_mut() = Some(seed.clone());
        y.backward(engine);
        out = host(ctx, &y.data);
    }
    let grads = layers
        .leaves()
        .into_iter()
        .map(|(_, t)| grad_of(ctx, t))
        .collect();
    for (_, t) in layers.leaves() {
        t.zero_grad();
    }
    (out, grads)
}

#[test]
#[ignore = "needs a GPU"]
fn lora_matmul_matches_the_separate_ops_bit_for_bit_and_the_cpu() {
    let (ctx, engine) = setup();
    // 16 / 12 is not a power of two, so its rounding shows; 2 and 1 are exact.
    for (half_w, train_w, scale) in [
        (false, false, 16.0 / 12.0),
        (false, true, 2.0),
        (true, false, 16.0 / 12.0),
        (false, false, 1.0),
    ] {
        let layers = Layers::new(&ctx, half_w, train_w);
        let label = format!("f16 w {half_w}, trained w {train_w}, scale {scale}");
        let (fused_out, fused_grads) = two_passes(&ctx, &engine, &layers, scale, true);
        let (sep_out, sep_grads) = two_passes(&ctx, &engine, &layers, scale, false);
        assert_eq!(bits(&fused_out), bits(&sep_out), "{label}: output");
        let names = layers.leaves().into_iter().map(|(name, _)| name);
        for (name, (f, s)) in names.zip(fused_grads.iter().zip(&sep_grads)) {
            assert_eq!(bits(f), bits(s), "{label}: gradient of {name}");
        }

        // CPU reference, f64: y = sum_i x w_i + s (x a_i) b_i, and with
        // g the seed, two passes double every gradient.
        let x = f64s(&host(&ctx, &layers.x.data));
        let g = f64s(&rand_vec(ROWS * OUT, 8));
        let s = f64::from(scale);
        let mut y = vec![0.0; ROWS * OUT];
        let mut gx = vec![0.0; ROWS * IN];
        let mut want = vec![("x", Vec::new())];
        let mut gab = Vec::new();
        let mut gw = Vec::new();
        for (i, seed) in [(0, 2), (1, 3)] {
            let w: Vec<f64> = rand_vec(IN * OUT, seed)
                .iter()
                .map(|v| f64::from((v * 64.0).round() / 64.0))
                .collect();
            let a = f64s(&host(&ctx, &layers.a[i].data));
            let b = f64s(&host(&ctx, &layers.b[i].data));
            let xa: Vec<f64> = mm(&x, &a, (ROWS, IN, RANK), false, false)
                .iter()
                .map(|v| s * v)
                .collect();
            let base = mm(&x, &w, (ROWS, IN, OUT), false, false);
            let lora = mm(&xa, &b, (ROWS, RANK, OUT), false, false);
            for (j, v) in y.iter_mut().enumerate() {
                *v += base[j] + lora[j];
            }
            // d(xa) = s g b^T; dx = g w^T + d(xa) a^T; da = x^T d(xa); db = xa^T g.
            let gxa: Vec<f64> = mm(&g, &b, (ROWS, OUT, RANK), false, true)
                .iter()
                .map(|v| s * v)
                .collect();
            let dx_base = mm(&g, &w, (ROWS, OUT, IN), false, true);
            let dx_lora = mm(&gxa, &a, (ROWS, RANK, IN), false, true);
            for (j, v) in gx.iter_mut().enumerate() {
                *v += 2.0 * (dx_base[j] + dx_lora[j]);
            }
            let da = mm(&x, &gxa, (IN, ROWS, RANK), true, false);
            let db = mm(&xa, &g, (RANK, ROWS, OUT), true, false);
            let dw = mm(&x, &g, (IN, ROWS, OUT), true, false);
            gab.push((["a0", "a1"][i], twice(&da)));
            gab.push((["b0", "b1"][i], twice(&db)));
            gw.push((["w0", "w1"][i], twice(&dw)));
        }
        want[0].1 = gx;
        // The order of `Layers::leaves`.
        for j in [0, 2, 1, 3] {
            want.push(gab[j].clone());
        }
        if train_w {
            want.extend(gw);
        }
        assert_close(&format!("{label}: output"), &fused_out, &y);
        let names = layers.leaves().into_iter().map(|(name, _)| name);
        for ((name, got), (want_name, want)) in names.zip(&fused_grads).zip(&want) {
            assert_eq!(name, *want_name);
            assert_close(&format!("{label}: gradient of {name}"), got, want);
        }
    }
}

#[test]
#[ignore = "needs a GPU"]
fn matmul_adds_in_place_only_into_a_gradient_nothing_shares() {
    let (ctx, engine) = setup();
    let (m, k, n) = (6, 20, 12);
    let x = Tensor::frozen(
        ctx.clone(),
        UnifiedTensor::new(&ctx, &rand_vec(m * k, 20), &[m, k]),
    );
    let p = Tensor::new(
        ctx.clone(),
        UnifiedTensor::new(&ctx, &rand_vec(k * n, 21), &[k, n]),
    );
    let step = || {
        let y = Tensor::matmul(&x, &p, &engine);
        y.backward(&engine);
    };
    // dp = x^T @ ones.
    let xs = f64s(&host(&ctx, &x.data));
    let once = mm(&xs, &vec![1.0; m * n], (k, m, n), true, false);

    step();
    let first = grad_of(&ctx, &p);
    assert_close("first pass", &first, &once);
    // A second pass with the gradient unshared adds into its buffer.
    let before = grad_buffer(&p);
    step();
    assert_eq!(grad_buffer(&p), before, "added in place");
    let twice: Vec<f64> = once.iter().map(|v| 2.0 * v).collect();
    assert_close("second pass", &grad_of(&ctx, &p), &twice);

    // A third pass while something else holds the gradient leaves that
    // holder's values alone and writes the sum to a new buffer.
    let held = p.grad.borrow().clone().expect("a gradient");
    step();
    assert_close("held copy", &host(&ctx, &held), &twice);
    let thrice: Vec<f64> = once.iter().map(|v| 3.0 * v).collect();
    assert_close("third pass", &grad_of(&ctx, &p), &thrice);
    assert_ne!(grad_buffer(&p), buffer_of(&held));
}

/// Passes its gradient on unchanged and records which buffer it was.
struct Probe {
    input: Tensor,
    seen: Cell<Option<*const ()>>,
}

impl AutogradNode for Probe {
    fn backward(&self, grad: &UnifiedTensor, _engine: &ComputeEngine) -> Vec<UnifiedTensor> {
        self.seen.set(Some(buffer_of(grad)));
        vec![grad.clone()]
    }

    fn inputs(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }
}

#[test]
#[ignore = "needs a GPU"]
fn add_hands_on_its_gradient_and_a_shared_slot_is_copied_before_a_write() {
    // h = y + probe(y @ w) with y = x * 1: the add gives both sides the
    // seed's own buffer, so when the matmul adds its share of y's gradient,
    // y's slot holds a buffer the matmul is still reading. The sum must go
    // to a new buffer, leaving the seed as it was.
    let (ctx, engine) = setup();
    let (m, k) = (6, 20);
    let leaf = |n: usize, shape: &[usize], seed: u64| {
        Tensor::new(
            ctx.clone(),
            UnifiedTensor::new(&ctx, &rand_vec(n, seed), shape),
        )
    };
    let x = leaf(m * k, &[m, k], 30);
    let w = leaf(k * k, &[k, k], 31);
    let y = Tensor::scale(&x, 1.0, &engine);
    let z = Tensor::matmul(&y, &w, &engine);
    let probe = Arc::new(Probe {
        input: z.clone(),
        seen: Cell::new(None),
    });
    let p = Tensor::with_creator(ctx.clone(), z.data.as_ref().clone(), probe.clone());
    let h = Tensor::add(&y, &p, &engine);
    let g = rand_vec(m * k, 32);
    let seed = UnifiedTensor::new(&ctx, &g, &[m, k]);
    *h.grad.borrow_mut() = Some(seed.clone());
    h.backward(&engine);

    assert_eq!(probe.seen.get(), Some(buffer_of(&seed)), "add copied");
    assert_eq!(bits(&host(&ctx, &seed)), bits(&g), "seed written");
    // dx = g + g @ w^T, dw = x^T @ g.
    let xs = f64s(&host(&ctx, &x.data));
    let ws = f64s(&host(&ctx, &w.data));
    let gs = f64s(&g);
    let gw = mm(&gs, &ws, (m, k, k), false, true);
    let dx: Vec<f64> = gs.iter().zip(&gw).map(|(a, b)| a + b).collect();
    assert_close("dx", &grad_of(&ctx, &x), &dx);
    let dw = mm(&xs, &gs, (k, m, k), true, false);
    assert_close("dw", &grad_of(&ctx, &w), &dw);

    // Two parameters summed get a buffer each, so scaling one in place (as
    // gradient clipping does) leaves the other and the seed alone.
    let a = leaf(m * k, &[m, k], 33);
    let b = leaf(m * k, &[m, k], 34);
    let s = Tensor::add(&a, &b, &engine);
    *s.grad.borrow_mut() = Some(seed.clone());
    s.backward(&engine);
    assert_ne!(grad_buffer(&a), grad_buffer(&b));
    assert_ne!(grad_buffer(&a), buffer_of(&seed));
    assert_eq!(bits(&grad_of(&ctx, &a)), bits(&g));
    let a_grad = a.grad.borrow().clone().expect("a gradient");
    engine.gradient_scale(&ctx, &a_grad, 2.0);
    assert_eq!(bits(&grad_of(&ctx, &b)), bits(&g));
    assert_eq!(bits(&host(&ctx, &seed)), bits(&g));
    let doubled: Vec<f64> = g.iter().map(|&v| 2.0 * f64::from(v)).collect();
    assert_close("scaled a", &grad_of(&ctx, &a), &doubled);
}

/// The fused QKV attention with its gradients packed into one buffer (one
/// product for the input gradient) against the same op with three buffers:
/// the input gradient and every adapter gradient agree to f32 rounding.
/// Shapes as 135M (9 heads over 3), 360M (15 over 5) and 1.7B (32 over 32),
/// with two batches and a length that leaves a partial tile.
#[test]
#[ignore = "needs a GPU"]
fn packed_attention_gradients_equal_the_three_buffer_ones() {
    let (ctx, engine) = setup();
    if !engine.flash_coop_available() {
        eprintln!("skipped: no cooperative matrices (or FERROTENSOR_EXPERIMENTAL is not 1)");
        return;
    }
    engine.set_flash_coop(true);
    const HEAD: usize = 64;
    for (n, &(heads, kv, batch, seq, rank)) in [
        (9usize, 3usize, 2usize, 100usize, 8usize),
        (15, 5, 1, 130, 8),
        (4, 4, 2, 70, 16),
    ]
    .iter()
    .enumerate()
    {
        let hidden = heads * HEAD;
        let (qw, kw) = (heads * HEAD, kv * HEAD);
        let seed = 100 * n as u64;
        let leaf = |data: &[f32], shape: &[usize]| {
            Tensor::new(ctx.clone(), UnifiedTensor::new(&ctx, data, shape))
        };
        let weight = |w: usize, salt: u64| {
            let data: Vec<f32> = rand_vec(hidden * w, seed + salt)
                .iter()
                .map(|v| v * 0.1)
                .collect();
            Tensor::frozen(
                ctx.clone(),
                UnifiedTensor::from_f32_as_f16(&ctx, &data, &[hidden, w]),
            )
        };
        let weights = [weight(qw, 1), weight(kw, 2), weight(kw, 3)];
        let widths = [qw, kw, kw];
        let run = |packed: bool| {
            engine.set_qkv_packed(packed);
            let lora = FusedLora {
                a: (0..3)
                    .map(|g| leaf(&rand_vec(hidden * rank, seed + 10 + g), &[hidden, rank]))
                    .collect(),
                b: (0..3)
                    .map(|g| {
                        let b: Vec<f32> = rand_vec(rank * widths[g as usize], seed + 20 + g)
                            .iter()
                            .map(|v| v * 0.1)
                            .collect();
                        leaf(&b, &[rank, widths[g as usize]])
                    })
                    .collect(),
                scales: vec![2.0, 0.5, 1.25],
                rank,
            };
            let proj = std::sync::Arc::new(
                FusedProjections::new(
                    &ctx,
                    &engine,
                    &[&weights[0], &weights[1], &weights[2]],
                    Some(lora.clone()),
                )
                .expect("projections"),
            );
            let x = leaf(
                &rand_vec(batch * seq * hidden, seed + 30),
                &[batch, seq, hidden],
            );
            let geom = AttentionGeometry {
                n_heads: heads,
                n_kv_heads: kv,
                head_dim: HEAD,
                theta: 10_000.0,
                pos: 3,
                dropout_prob: 0.0,
                seed: 0,
            };
            let y = Tensor::fused_attention(&x, &proj, &geom, &engine);
            let dy = UnifiedTensor::new(
                &ctx,
                &rand_vec(batch * seq * qw, seed + 40),
                &[batch, seq, qw],
            );
            *y.grad.borrow_mut() = Some(dy);
            y.backward(&engine);
            let mut grads = vec![("x".to_string(), grad_of(&ctx, &x))];
            for g in 0..3 {
                grads.push((format!("a{g}"), grad_of(&ctx, &lora.a[g])));
                grads.push((format!("b{g}"), grad_of(&ctx, &lora.b[g])));
            }
            (host(&ctx, &y.data), grads)
        };
        let (out_split, split) = run(false);
        let (out_packed, packed) = run(true);
        engine.set_qkv_packed(true);
        assert_eq!(bits(&out_split), bits(&out_packed), "forward {heads}/{kv}");
        for ((name, got), (_, want)) in packed.iter().zip(&split) {
            let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            assert!(scale > 0.0, "{name}: the reference gradient is zero");
            let worst = got
                .iter()
                .zip(want)
                .map(|(g, w)| (g - w).abs())
                .fold(0.0f32, f32::max);
            assert!(
                worst <= 1e-3 * scale,
                "{heads}/{kv} {name}: largest difference {worst}, gradient scale {scale}"
            );
        }
    }
}
