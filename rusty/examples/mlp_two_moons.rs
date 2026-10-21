//! Train a small MLP on a synthetic two-moons dataset with `DataLoader`.
//!
//! Run with: cargo run --release -p rusty --example mlp_two_moons
//! Set RUSTY_DEVICE=gpu to train on the GPU instead of the CPU.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rusty::data::{DataLoader, Sample, TensorDataset};
use rusty::optim::{LrScheduler, WarmupCosineLr};
use rusty::prelude::*;

/// `n` points per moon, with Gaussian-ish noise. Label 0 = upper moon, 1 = lower.
fn two_moons(n: usize, noise: f32, rng: &mut StdRng) -> (Vec<[f32; 2]>, Vec<usize>) {
    let mut points = Vec::new();
    let mut labels = Vec::new();
    for i in 0..n {
        let t = std::f32::consts::PI * i as f32 / n as f32;
        let mut jitter = || (rng.gen::<f32>() + rng.gen::<f32>() + rng.gen::<f32>() - 1.5) * noise;
        points.push([t.cos() + jitter(), t.sin() + jitter()]);
        labels.push(0);
        points.push([1.0 - t.cos() + jitter(), 0.5 - t.sin() + jitter()]);
        labels.push(1);
    }
    (points, labels)
}

fn dataset(points: &[[f32; 2]], labels: &[usize]) -> TensorDataset {
    let inputs = points
        .iter()
        .map(|p| Sample::new(p.to_vec(), vec![2]))
        .collect();
    let targets = labels
        .iter()
        .map(|&l| Sample::new(vec![l as f32], vec![1]))
        .collect();
    TensorDataset::new(inputs, targets)
}

fn main() {
    let device = match std::env::var("RUSTY_DEVICE").as_deref() {
        Ok("gpu") => Device::new(),
        _ => Device::cpu(),
    };
    println!("device: {} ({})", device.name(), device.backend());

    let mut rng = StdRng::seed_from_u64(7);
    let (train_x, train_y) = two_moons(400, 0.25, &mut rng);
    let (test_x, test_y) = two_moons(200, 0.25, &mut rng);
    println!(
        "train: {} points, test: {} points",
        train_x.len(),
        test_x.len()
    );

    let model = nn::Sequential::new(&device)
        .add(nn::Linear::new(2, 32, &device))
        .add(nn::Tanh)
        .add(nn::Linear::new(32, 32, &device))
        .add(nn::Tanh)
        .add(nn::Linear::new(32, 2, &device));
    println!(
        "model: 2-32-32-2 MLP, {} parameters",
        model.num_parameters()
    );

    let loader = DataLoader::new(dataset(&train_x, &train_y), 64, true, device.clone()).seed(1);
    let epochs = 60;
    let schedule = WarmupCosineLr::new(0.02, 0.001, 20, epochs * loader.num_batches());
    let mut opt = optim::AdamW::new(model.parameters(), 0.02).weight_decay(0.0);

    // Held-out data as a single batch.
    let flat: Vec<f32> = test_x.iter().flatten().copied().collect();
    let tx = Tensor::from_data(&flat, &[test_x.len(), 2], &device);

    let mut step = 0;
    for epoch in 0..epochs {
        let mut total = 0.0;
        for (x, y) in loader.epoch(epoch as u64) {
            schedule.apply(&mut opt, step);
            step += 1;
            opt.zero_grad();
            let loss = F::cross_entropy(&model.forward(&x), &y.reshape(&[y.shape[0]]));
            loss.backward();
            opt.step();
            total += loss.item();
        }
        if epoch % 10 == 0 || epoch == epochs - 1 {
            let acc = F::accuracy(&model.forward(&tx), &test_y);
            println!(
                "epoch {epoch:>2}  train loss {:.4}  test accuracy {:.1}%",
                total / loader.num_batches() as f32,
                acc * 100.0
            );
        }
    }
}
