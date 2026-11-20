//! The flat optimizer step: the norm, the clip and the AdamW update over
//! gradients packed into one buffer give the same bits as one dispatch per
//! tensor, and the values of a CPU reference.
//!
//! These need a real GPU, so they are `#[ignore]`d by default. Run them with:
//!
//! ```text
//! cargo test -p ferrotensor-autograd --test flat_optimizer -- --ignored --test-threads=1
//! ```

#![allow(
    clippy::expect_used,
    clippy::panic,
    clippy::arc_with_non_send_sync,
    clippy::needless_range_loop
)]

use std::collections::HashMap;
use std::sync::Arc;

use ferrotensor_autograd::{AdamW, FlatGrads, Optimizer, OptimizerCheckpoint, Tensor};
use ferrotensor_backend::{ComputeEngine, UnifiedTensor, WgpuContext, SUM_SQUARES_CHUNK};

/// A rate big enough that weight decay and the update both stand well above
/// the rounding of f32 and of the GPU's `pow`.
const LR: f32 = 0.1;
const LR_REF: f64 = 0.1;

struct Rig {
    ctx: Arc<WgpuContext>,
    engine: Arc<ComputeEngine>,
}

fn rig() -> Rig {
    let ctx = Arc::new(pollster::block_on(WgpuContext::new()));
    let engine = Arc::new(ComputeEngine::new(&ctx));
    Rig { ctx, engine }
}

/// Deterministic pseudo-random values in [-scale, scale).
fn rand_vec(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * scale
        })
        .collect()
}

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

fn params(rig: &Rig, weights: &[Vec<f32>]) -> Vec<Tensor> {
    weights
        .iter()
        .map(|w| {
            let data = UnifiedTensor::new(&rig.ctx, w, &[w.len()]);
            Tensor::new(Arc::clone(&rig.ctx), data)
        })
        .collect()
}

fn weights_of(rig: &Rig, params: &[Tensor]) -> Vec<Vec<f32>> {
    params
        .iter()
        .map(|p| pollster::block_on(p.data.to_vec(&rig.ctx)).expect("GPU readback"))
        .collect()
}

/// Gives every parameter the gradient in `grads` and returns the gradients
/// as the trainer collects them.
fn set_grads(rig: &Rig, params: &[Tensor], grads: &[Vec<f32>]) -> Vec<UnifiedTensor> {
    for (p, g) in params.iter().zip(grads) {
        *p.grad.borrow_mut() = Some(UnifiedTensor::new(&rig.ctx, g, &[g.len()]));
    }
    params
        .iter()
        .filter_map(|p| p.grad.borrow().clone())
        .collect()
}

fn host_norm(rig: &Rig, partials: &UnifiedTensor) -> f32 {
    let sum_sq: f64 = pollster::block_on(partials.to_vec(&rig.ctx))
        .expect("GPU readback")
        .iter()
        .map(|&x| f64::from(x))
        .sum();
    sum_sq.sqrt() as f32
}

/// One optimizer step the old way: one sum-of-squares, one clip and one
/// AdamW dispatch per tensor. Returns the gradient norm.
fn step_per_tensor(
    rig: &Rig,
    opt: &mut AdamW,
    params: &[Tensor],
    grads: &[Vec<f32>],
    max_norm: f32,
) -> f32 {
    let (ctx, engine) = (&rig.ctx, &rig.engine);
    let gs = set_grads(rig, params, grads);
    let sizes: Vec<usize> = grads.iter().map(Vec::len).collect();
    let chunks: usize = sizes.iter().map(|n| n.div_ceil(SUM_SQUARES_CHUNK)).sum();
    let partials = UnifiedTensor::empty(ctx, &[chunks]);
    let mut offset = 0;
    for g in &gs {
        offset += engine.sum_squares_partial(ctx, g, &partials, offset);
    }
    let norm = host_norm(rig, &partials);
    if norm > max_norm {
        for g in &gs {
            engine.clip_grad_by_norm(ctx, g, max_norm / (norm + 1e-6));
        }
    }
    opt.step(params);
    norm
}

/// One optimizer step over one buffer: one sum-of-squares, one clip and one
/// AdamW dispatch. Returns the gradient norm.
fn step_flat(
    rig: &Rig,
    opt: &mut AdamW,
    params: &[Tensor],
    grads: &[Vec<f32>],
    max_norm: f32,
) -> f32 {
    let (ctx, engine) = (&rig.ctx, &rig.engine);
    let gs = set_grads(rig, params, grads);
    let sizes: Vec<usize> = grads.iter().map(Vec::len).collect();
    let flat = FlatGrads::new(ctx, &sizes).expect("the gradients fit one buffer");
    flat.pack(ctx, engine, &gs).expect("pack");
    let chunks = flat.layout.total() / SUM_SQUARES_CHUNK;
    let partials = UnifiedTensor::empty(ctx, &[chunks]);
    engine.sum_squares_partial(ctx, &flat.data, &partials, 0);
    let norm = host_norm(rig, &partials);
    if norm > max_norm {
        engine.clip_grad_by_norm(ctx, &flat.data, max_norm / (norm + 1e-6));
    }
    opt.step_flat(params, &flat).expect("flat step");
    norm
}

/// One AdamW step in f64 (the optimizer's defaults), updating `w`, `m`, `v`.
fn adamw_ref(w: &mut [f64], g: &[f64], m: &mut [f64], v: &mut [f64], lr: f64, t: i32) {
    let (b1, b2, eps, wd) = (0.9f64, 0.999f64, 1e-8f64, 0.01f64);
    for i in 0..w.len() {
        m[i] = m[i] * b1 + (1.0 - b1) * g[i];
        v[i] = v[i] * b2 + (1.0 - b2) * g[i] * g[i];
        let m_hat = m[i] / (1.0 - b1.powi(t));
        let v_hat = v[i] / (1.0 - b2.powi(t));
        w[i] -= lr * (m_hat / (v_hat.sqrt() + eps) + wd * w[i]);
    }
}

/// Checks `|got - want| <= rel * max(|want|, floor)` for every element.
fn assert_close(got: &[f32], want: &[f64], (rel, floor): (f64, f64), what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (&a, &b)) in got.iter().zip(want).enumerate() {
        let tol = rel * b.abs().max(floor);
        assert!(
            (f64::from(a) - b).abs() <= tol,
            "{what}[{i}]: got {a}, want {b}"
        );
    }
}

#[test]
#[ignore = "needs a GPU"]
fn flat_step_matches_per_tensor_steps_bit_for_bit() {
    let rig = rig();
    // Sizes that are not multiples of a chunk, one tiny, one just past two.
    let sizes = [5000usize, 7, 2 * SUM_SQUARES_CHUNK + 3];
    let init: Vec<Vec<f32>> = sizes
        .iter()
        .enumerate()
        .map(|(i, &n)| rand_vec(n, 10 + i as u64, 1.0))
        .collect();
    let per_tensor = params(&rig, &init);
    let flat = params(&rig, &init);
    let mixed = params(&rig, &init);
    let mut opt_per = AdamW::new(Arc::clone(&rig.engine), LR);
    let mut opt_flat = AdamW::new(Arc::clone(&rig.engine), LR);
    let mut opt_mixed = AdamW::new(Arc::clone(&rig.engine), LR);

    let mut w_ref: Vec<Vec<f64>> = init
        .iter()
        .map(|w| w.iter().map(|&x| f64::from(x)).collect())
        .collect();
    let mut m_ref: Vec<Vec<f64>> = sizes.iter().map(|&n| vec![0.0; n]).collect();
    let mut v_ref = m_ref.clone();

    let max_norm = 1.0f32;
    for step in 1..=3i32 {
        // About 30 in norm, so every step is clipped.
        let grads: Vec<Vec<f32>> = sizes
            .iter()
            .enumerate()
            .map(|(i, &n)| rand_vec(n, 100 * step as u64 + i as u64, 0.5))
            .collect();

        let norm_per = step_per_tensor(&rig, &mut opt_per, &per_tensor, &grads, max_norm);
        let norm_flat = step_flat(&rig, &mut opt_flat, &flat, &grads, max_norm);
        // Flat, then per tensor (the moments move out of the flat buffers),
        // then flat again (they move back in).
        let norm_mixed = if step == 2 {
            step_per_tensor(&rig, &mut opt_mixed, &mixed, &grads, max_norm)
        } else {
            step_flat(&rig, &mut opt_mixed, &mixed, &grads, max_norm)
        };
        assert!(norm_per > max_norm, "step {step} should clip: {norm_per}");
        assert_eq!(norm_per.to_bits(), norm_flat.to_bits(), "norm, step {step}");
        assert_eq!(
            norm_per.to_bits(),
            norm_mixed.to_bits(),
            "norm, step {step}"
        );

        let scale = f64::from(max_norm / (norm_per + 1e-6));
        for i in 0..sizes.len() {
            let g: Vec<f64> = grads[i].iter().map(|&x| f64::from(x) * scale).collect();
            adamw_ref(
                &mut w_ref[i],
                &g,
                &mut m_ref[i],
                &mut v_ref[i],
                LR_REF,
                step,
            );
        }

        let w_per = weights_of(&rig, &per_tensor);
        let w_flat = weights_of(&rig, &flat);
        let w_mixed = weights_of(&rig, &mixed);
        for i in 0..sizes.len() {
            assert_eq!(
                bits(&w_per[i]),
                bits(&w_flat[i]),
                "flat, tensor {i}, step {step}"
            );
            assert_eq!(
                bits(&w_per[i]),
                bits(&w_mixed[i]),
                "mixed, tensor {i}, step {step}"
            );
            assert_close(
                &w_flat[i],
                &w_ref[i],
                (1e-4, 1.0),
                "weights against the f64 reference",
            );
        }
    }

    // The saved moments are the same whether they live in flat buffers or in
    // one tensor per parameter.
    let names = ["a", "b", "c"];
    let saved = |p: &[Tensor], opt: &AdamW| {
        let named: HashMap<String, Tensor> = names
            .iter()
            .zip(p)
            .map(|(n, t)| ((*n).to_string(), t.clone()))
            .collect();
        match opt.save_state(&rig.ctx, &named).expect("state") {
            OptimizerCheckpoint::AdamW(state) => state,
        }
    };
    let s_per = saved(&per_tensor, &opt_per);
    let s_flat = saved(&flat, &opt_flat);
    let s_mixed = saved(&mixed, &opt_mixed);
    assert_eq!(s_per.t, 3.0);
    assert_eq!(s_flat.t, 3.0);
    assert_eq!(s_mixed.t, 3.0);
    for (i, name) in names.iter().enumerate() {
        let key = (*name).to_string();
        for (what, per, other, want, floor) in [
            ("m", &s_per.m_states, &s_flat.m_states, &m_ref[i], 1e-3),
            ("v", &s_per.v_states, &s_flat.v_states, &v_ref[i], 1e-7),
            ("m", &s_per.m_states, &s_mixed.m_states, &m_ref[i], 1e-3),
            ("v", &s_per.v_states, &s_mixed.v_states, &v_ref[i], 1e-7),
        ] {
            assert_eq!(per[&key].shape, other[&key].shape, "{what} shape of {name}");
            assert_eq!(
                bits(&per[&key].data),
                bits(&other[&key].data),
                "{what} of {name}"
            );
            assert_close(
                &other[&key].data,
                want,
                (1e-4, floor),
                "moments, f64 reference",
            );
        }
    }
}

#[test]
#[ignore = "needs a GPU"]
fn flat_step_first_update_has_the_closed_form_values() {
    let rig = rig();
    // With a fresh optimizer the first AdamW step is
    // w - lr * (sign(g) + weight_decay * w), up to eps.
    let init = vec![vec![1.0f32], vec![2.0, -1.0]];
    let p = params(&rig, &init);
    let mut opt = AdamW::new(Arc::clone(&rig.engine), LR);
    let grads = vec![vec![0.5f32], vec![0.25, -0.5]];
    let norm = step_flat(&rig, &mut opt, &p, &grads, 100.0);
    let want_norm = (0.25f64 + 0.0625 + 0.25).sqrt();
    assert!((f64::from(norm) - want_norm).abs() < 1e-6, "norm {norm}");
    let w = weights_of(&rig, &p);
    assert_close(&w[0], &[0.899], (1e-4, 1.0), "first weight");
    assert_close(&w[1], &[1.898, -0.899], (1e-4, 1.0), "second weight");

    let named: HashMap<String, Tensor> = HashMap::from([
        ("a".to_string(), p[0].clone()),
        ("b".to_string(), p[1].clone()),
    ]);
    let OptimizerCheckpoint::AdamW(state) = opt.save_state(&rig.ctx, &named).expect("state");
    let tight = (1e-5, 1e-6);
    // beta2 = 0.999 is stored as an f32, so 1 - beta2 is off by about 1e-6
    // in relative terms and the second moment inherits that.
    let loose = (1e-4, 1e-6);
    assert_close(
        &state.m_states["a"].data,
        &[0.05],
        tight,
        "first moment of a",
    );
    assert_close(
        &state.m_states["b"].data,
        &[0.025, -0.05],
        tight,
        "first moment of b",
    );
    assert_close(
        &state.v_states["a"].data,
        &[0.00025],
        loose,
        "second moment of a",
    );
    assert_close(
        &state.v_states["b"].data,
        &[0.0000625, 0.00025],
        loose,
        "second moment of b",
    );
}

#[test]
#[ignore = "needs a GPU"]
fn flat_step_uploads_nothing_from_the_host() {
    let rig = rig();
    let (ctx, engine) = (&rig.ctx, &rig.engine);
    // Padded to chunks these are 8192 + 4096 + 12288 = 24,576 elements, so
    // each arena is 98,304 bytes. Zero-filling the gradient and weight
    // arenas from the host would upload 2 * 98,304 = 196,608 bytes per step,
    // and the first step's two moment buffers another 196,608.
    let sizes = [5000usize, 7, 2 * SUM_SQUARES_CHUNK + 3];
    let init: Vec<Vec<f32>> = sizes
        .iter()
        .enumerate()
        .map(|(i, &n)| rand_vec(n, 30 + i as u64, 1.0))
        .collect();
    let p = params(&rig, &init);
    let mut opt = AdamW::new(Arc::clone(engine), LR);

    for step in 0..3u64 {
        // Setting the gradients is the backward pass's job, not the step's.
        let grads: Vec<Vec<f32>> = sizes
            .iter()
            .enumerate()
            .map(|(i, &n)| rand_vec(n, 40 + 10 * step + i as u64, 0.5))
            .collect();
        let gs = set_grads(&rig, &p, &grads);
        ctx.synchronize();

        let before = ctx.gpu_counters();
        let flat = FlatGrads::new(ctx, &sizes).expect("the gradients fit one buffer");
        flat.pack(ctx, engine, &gs).expect("pack");
        opt.step_flat(&p, &flat).expect("flat step");
        let used = ctx.gpu_counters().since(&before);
        assert_eq!(
            used.upload_bytes, 0,
            "step {step}: the arenas are cleared on the GPU, {used:?}"
        );
        drop(flat);
        ctx.synchronize();
    }

    // The next arena reuses a pooled buffer that held gradients; its gaps and
    // padding must read as zero all the same.
    let again = FlatGrads::new(ctx, &sizes).expect("fits");
    let all = pollster::block_on(again.data.to_vec(ctx)).expect("GPU readback");
    assert_eq!(all.len(), again.layout.total());
    assert!(all.iter().all(|&x| x.to_bits() == 0), "not zeroed");
}
