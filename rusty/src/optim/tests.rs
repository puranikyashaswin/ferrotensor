//! Optimizer behaviour and end-to-end training on the CPU device.

use crate::prelude::*;

fn leaf(data: &[f32]) -> Tensor {
    Tensor::from_data(data, &[data.len()], &Device::cpu()).requires_grad_()
}

fn close(a: &[f32], b: &[f32]) {
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < 1e-5, "{a:?} vs {b:?}");
    }
}

#[test]
fn sgd_momentum_matches_hand_computation() {
    let w = leaf(&[1.0]);
    let mut opt = optim::SGD::with_momentum(vec![w.clone()], 0.1, 0.9);
    // loss = w^2 -> grad 2w
    for _ in 0..2 {
        opt.zero_grad();
        w.square().sum().backward();
        opt.step();
    }
    // step 1: v = 2, w = 0.8; step 2: g = 1.6, v = 0.9 * 2 + 1.6 = 3.4, w = 0.46
    close(&w.to_vec(), &[0.46]);
}

#[test]
fn sgd_weight_decay_and_nesterov() {
    let w = leaf(&[1.0]);
    let mut opt = optim::SGD::with_momentum(vec![w.clone()], 0.1, 0.5)
        .nesterov(true)
        .weight_decay(1.0);
    w.mul_scalar(0.0).sum().backward(); // grad 0, so only decay acts
    opt.step();
    // g = 0 + 1 * w = 1, v = 1, nesterov g + mu v = 1.5 -> w = 0.85
    close(&w.to_vec(), &[0.85]);
}

#[test]
fn adamw_first_step_is_sign_times_lr() {
    let w = leaf(&[1.0, -2.0]);
    let mut opt = optim::AdamW::new(vec![w.clone()], 0.1).weight_decay(0.0);
    w.mul(&Tensor::from_data(&[3.0, 5.0], &[2], &Device::cpu()))
        .sum()
        .backward();
    opt.step();
    close(&w.to_vec(), &[0.9, -2.1]);
    assert_eq!(opt.steps(), 1);
}

#[test]
fn adamw_weight_decay_is_decoupled() {
    let w = leaf(&[2.0]);
    let mut opt = optim::AdamW::new(vec![w.clone()], 0.1).weight_decay(0.5);
    w.mul_scalar(1.0).sum().backward();
    opt.step();
    // decay: 2 * (1 - 0.05) = 1.9, then the Adam update of 0.1.
    close(&w.to_vec(), &[1.8]);
}

#[test]
fn optimizers_skip_params_without_grad() {
    let a = leaf(&[1.0]);
    let b = leaf(&[1.0]);
    let mut opt = optim::AdamW::new(vec![a.clone(), b.clone()], 0.1);
    a.sum().backward();
    opt.step();
    assert_eq!(b.to_vec(), vec![1.0]);
    assert!(a.to_vec()[0] < 1.0);
}

#[test]
fn clip_grad_value_clamps() {
    let w = leaf(&[1.0, 2.0]);
    w.mul_scalar(10.0).sum().backward();
    optim::clip_grad_value(&[w.clone()], 3.0);
    assert_eq!(w.grad().unwrap().to_vec(), vec![3.0, 3.0]);
}

#[test]
fn clip_grad_norm_is_global() {
    let a = leaf(&[3.0]);
    let b = leaf(&[4.0]);
    a.add(&b).sum().backward(); // both grads 1 -> norm sqrt(2)
    a.mul_scalar(0.0).sum().backward(); // keeps grads
    let norm = optim::clip_grad_norm(&[a.clone(), b.clone()], 1.0);
    assert!((norm - 2f32.sqrt()).abs() < 1e-6);
    let ga = a.grad().unwrap().item();
    let gb = b.grad().unwrap().item();
    assert!(((ga * ga + gb * gb).sqrt() - 1.0).abs() < 1e-4);
}

#[test]
fn mlp_learns_xor() {
    let d = Device::cpu();
    let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(1);
    let model = nn::Sequential::new(&d)
        .add(nn::Linear::from_weights(
            Tensor::randn_with(&[2, 16], &mut rng, &d).requires_grad_(),
            Some(Tensor::zeros(&[16], &d).requires_grad_()),
        ))
        .add(nn::Tanh)
        .add(nn::Linear::from_weights(
            Tensor::randn_with(&[16, 2], &mut rng, &d)
                .mul_scalar(0.3)
                .requires_grad_(),
            Some(Tensor::zeros(&[2], &d).requires_grad_()),
        ));
    let x = Tensor::from_data(&[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], &[4, 2], &d);
    let y = Tensor::from_data(&[0.0, 1.0, 1.0, 0.0], &[4], &d);
    let mut opt = optim::AdamW::new(model.parameters(), 0.05).weight_decay(0.0);
    let mut loss = f32::INFINITY;
    for _ in 0..300 {
        opt.zero_grad();
        let l = F::cross_entropy(&model.forward(&x), &y);
        l.backward();
        opt.step();
        loss = l.item();
    }
    assert!(loss < 0.05, "XOR loss stayed at {loss}");
    assert_eq!(F::accuracy(&model.forward(&x), &[0, 1, 1, 0]), 1.0);
}

#[test]
fn linear_regression_converges_with_sgd() {
    let d = Device::cpu();
    let x = Tensor::randn(&[64, 3], &d);
    let true_w = Tensor::from_data(&[2.0, -1.0, 0.5], &[3, 1], &d);
    let y = x.matmul(&true_w).add_scalar(0.3);
    let model = nn::Linear::new(3, 1, &d);
    let mut opt = optim::SGD::with_momentum(model.parameters(), 0.1, 0.9);
    for _ in 0..200 {
        opt.zero_grad();
        F::mse_loss(&model.forward(&x), &y).backward();
        opt.step();
    }
    let w = model.weight.to_vec();
    for (got, want) in w.iter().zip([2.0, -1.0, 0.5]) {
        assert!((got - want).abs() < 1e-3, "weights {w:?}");
    }
    let b = model.bias.as_ref().unwrap().item();
    assert!((b - 0.3).abs() < 1e-3, "bias {b}");
}

/// With `RUSTY_RUN_GPU_TESTS=1`: ten AdamW steps on an MLP give the same
/// parameters on the GPU as on the CPU.
#[test]
fn gpu_training_matches_cpu() {
    if std::env::var("RUSTY_RUN_GPU_TESTS").as_deref() != Ok("1") {
        eprintln!("skipping GPU training parity; set RUSTY_RUN_GPU_TESTS=1 to enable");
        return;
    }
    let gpu = Device::gpu().expect("RUSTY_RUN_GPU_TESTS=1 but no GPU is available");
    let train = |d: &Device| {
        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(3);
        let w1 = Tensor::randn_with(&[8, 32], &mut rng, d).requires_grad_();
        let w2 = Tensor::randn_with(&[32, 4], &mut rng, d).requires_grad_();
        let x = Tensor::randn_with(&[16, 8], &mut rng, d);
        let y = Tensor::from_vec((0..16).map(|i| (i % 4) as f32).collect(), &[16], d);
        let mut opt = optim::AdamW::new(vec![w1.clone(), w2.clone()], 1e-2).clip_grad_norm(1.0);
        for _ in 0..10 {
            opt.zero_grad();
            F::cross_entropy(&x.matmul(&w1).gelu().matmul(&w2), &y).backward();
            opt.step();
        }
        [w1.to_vec(), w2.to_vec()].concat()
    };
    let (on_cpu, on_gpu) = (train(&Device::cpu()), train(&gpu));
    for (a, b) in on_gpu.iter().zip(&on_cpu) {
        assert!((a - b).abs() < 1e-3, "gpu {a} vs cpu {b}");
    }
}
