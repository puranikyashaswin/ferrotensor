//! End-to-end timings of the `ferrotensor` Tensor API on the CPU and GPU
//! devices: `cargo run -p ferrotensor-benchmarks --release --bin tensor-api`.
//!
//! Unlike the kernel suite, every case includes the host-side cost of each op
//! (autograd bookkeeping, buffer and bind group creation, submission) and ends
//! with a read back to the host, as user code does. Each sample is one
//! iteration; the table shows the median.
//!
//! The checksum column hashes the bits of a seeded run's results, so two
//! builds can be checked for bit-identical output.

use ferrotensor::prelude::*;
use rand::{rngs::StdRng, SeedableRng};
use std::time::{Duration, Instant};

/// Median wall time of `iter` after three warm-up calls.
fn time(mut iter: impl FnMut()) -> Duration {
    for _ in 0..3 {
        iter();
    }
    let mut samples = Vec::new();
    let budget = Instant::now();
    while samples.len() < 10 || (samples.len() < 200 && budget.elapsed() < Duration::from_secs(2)) {
        let t = Instant::now();
        iter();
        samples.push(t.elapsed());
    }
    samples.sort();
    samples[samples.len() / 2]
}

/// FNV-1a over the bits of `values`.
fn checksum(values: &[f32]) -> u64 {
    values.iter().fold(0xcbf2_9ce4_8422_2325, |h, v| {
        (h ^ v.to_bits() as u64).wrapping_mul(0x0100_0000_01b3)
    })
}

/// 100 element-wise ops on a `[n]` tensor, then a read back.
fn op_chain(x: &Tensor) -> Vec<f32> {
    let mut y = x.clone();
    for i in 0..25 {
        y = y
            .mul_scalar(1.001)
            .add_scalar(0.01 * i as f32)
            .tanh()
            .add(x);
    }
    y.to_vec()
}

/// Parameters and data of a 784-256-10 MLP with batch 64.
struct Mlp {
    params: Vec<Tensor>,
    x: Tensor,
    y: Tensor,
    opt: optim::AdamW,
}

impl Mlp {
    fn new(device: &Device) -> Self {
        let mut rng = StdRng::seed_from_u64(7);
        let w1 = Tensor::randn_with(&[784, 256], &mut rng, device).mul_scalar(0.05);
        let b1 = Tensor::zeros(&[256], device);
        let w2 = Tensor::randn_with(&[256, 10], &mut rng, device).mul_scalar(0.05);
        let b2 = Tensor::zeros(&[10], device);
        let params: Vec<Tensor> = [w1, b1, w2, b2]
            .into_iter()
            .map(Tensor::requires_grad_)
            .collect();
        let x = Tensor::randn_with(&[64, 784], &mut rng, device);
        let y = Tensor::from_vec((0..64).map(|i| (i % 10) as f32).collect(), &[64], device);
        let opt = optim::AdamW::new(params.clone(), 1e-3);
        Self { params, x, y, opt }
    }

    /// One training step; returns the loss.
    fn step(&mut self) -> f32 {
        let [w1, b1, w2, b2] = &self.params[..] else {
            unreachable!()
        };
        self.opt.zero_grad();
        let h = self.x.matmul(w1).add(b1).relu();
        let loss = F::cross_entropy(&h.matmul(w2).add(b2), &self.y);
        loss.backward();
        self.opt.step();
        loss.item()
    }

    fn values(&self) -> Vec<f32> {
        self.params.iter().flat_map(Tensor::to_vec).collect()
    }
}

/// Forward and backward of an `nn::Embedding` lookup; returns the weight
/// gradient.
fn embedding_step(emb: &nn::Embedding, ids: &Tensor) -> Vec<f32> {
    emb.weight.zero_grad();
    emb.forward(ids).sum().backward();
    emb.weight.grad().expect("embedding gradient").to_vec()
}

/// Backward of `index_select` along `dim`; returns the gradient.
fn index_select_step(x: &Tensor, dim: isize, indices: &[usize]) -> Vec<f32> {
    x.zero_grad();
    x.index_select(dim, indices).sum().backward();
    x.grad().expect("index_select gradient").to_vec()
}

fn main() {
    let filter = std::env::args().skip_while(|a| a != "--filter").nth(1);
    let mut devices = vec![Device::cpu()];
    match Device::gpu() {
        Some(gpu) => devices.push(gpu),
        None => eprintln!("no GPU found: CPU only"),
    }
    for d in &devices {
        println!(
            "{}: {} ({})",
            if d.is_cpu() { "CPU" } else { "GPU" },
            d.name(),
            d.backend()
        );
    }
    println!();
    println!("| case | device | median | checksum |");
    println!("|---|---|---:|---|");

    let row = |name: &str, d: &Device, t: Duration, sum: u64| {
        let dev = if d.is_cpu() { "CPU" } else { "GPU" };
        println!(
            "| {name} | {dev} | {:.3} ms | {sum:016x} |",
            t.as_secs_f64() * 1e3
        );
    };
    let wants = |name: &str| filter.as_deref().is_none_or(|f| name.contains(f));

    for n in [64, 4096] {
        let name = format!("100 element-wise ops, [{n}]");
        if !wants(&name) {
            continue;
        }
        for d in &devices {
            let mut rng = StdRng::seed_from_u64(1);
            let x = Tensor::randn_with(&[n], &mut rng, d);
            let sum = checksum(&op_chain(&x));
            row(&name, d, time(|| drop(op_chain(&x))), sum);
        }
    }

    let name = "MLP 784-256-10 AdamW step, batch 64";
    if wants(name) {
        for d in &devices {
            let mut check = Mlp::new(d);
            for _ in 0..5 {
                check.step();
            }
            let sum = checksum(&check.values());
            let mut mlp = Mlp::new(d);
            row(
                name,
                d,
                time(|| {
                    mlp.step();
                }),
                sum,
            );
        }
    }

    let name = "matmul 512x512x512 + read back";
    if wants(name) {
        for d in &devices {
            let mut rng = StdRng::seed_from_u64(2);
            let a = Tensor::randn_with(&[512, 512], &mut rng, d);
            let b = Tensor::randn_with(&[512, 512], &mut rng, d);
            let sum = checksum(&a.matmul(&b).to_vec());
            row(name, d, time(|| drop(a.matmul(&b).to_vec())), sum);
        }
    }

    for shape in [&[4096, 4096][..], &[1_000_000]] {
        let name = format!("sum() {shape:?} + read back");
        if !wants(&name) {
            continue;
        }
        for d in &devices {
            let mut rng = StdRng::seed_from_u64(3);
            let x = Tensor::randn_with(shape, &mut rng, d);
            let sum = checksum(&[x.sum().item()]);
            row(
                &name,
                d,
                time(|| {
                    x.sum().item();
                }),
                sum,
            );
        }
    }

    let name = "argmax_dim(-1) [1024, 4096] + read back";
    if wants(name) {
        for d in &devices {
            let mut rng = StdRng::seed_from_u64(4);
            let x = Tensor::randn_with(&[1024, 4096], &mut rng, d);
            let sum = checksum(&x.argmax_dim(-1).to_vec());
            row(name, d, time(|| drop(x.argmax_dim(-1).to_vec())), sum);
        }
    }

    let name = "argmax() [4096, 4096]";
    if wants(name) {
        for d in &devices {
            let mut rng = StdRng::seed_from_u64(5);
            let x = Tensor::randn_with(&[4096, 4096], &mut rng, d);
            let sum = checksum(&[x.argmax() as f32]);
            row(
                name,
                d,
                time(|| {
                    x.argmax();
                }),
                sum,
            );
        }
    }

    let name = "Embedding 32000x256 fwd+bwd, 4096 ids";
    if wants(name) {
        for d in &devices {
            let mut rng = StdRng::seed_from_u64(6);
            let weight = Tensor::randn_with(&[32_000, 256], &mut rng, d).requires_grad_();
            let emb = nn::Embedding::from_weights(weight);
            let ids: Vec<f32> = (0..4096u64)
                .map(|i| ((i * 7919 + i * i * 31) % 32_000) as f32)
                .collect();
            let ids = Tensor::from_vec(ids, &[4096], d);
            let sum = checksum(&embedding_step(&emb, &ids));
            row(name, d, time(|| drop(embedding_step(&emb, &ids))), sum);
        }
    }

    for dim in [0isize, 1] {
        let name = format!("index_select({dim}) bwd [2048, 2048], 1024 ids");
        if !wants(&name) {
            continue;
        }
        for d in &devices {
            let mut rng = StdRng::seed_from_u64(7);
            let x = Tensor::randn_with(&[2048, 2048], &mut rng, d).requires_grad_();
            let indices: Vec<usize> = (0..1024).map(|i| (i * 37 + i * i) % 2048).collect();
            let sum = checksum(&index_select_step(&x, dim, &indices));
            row(
                &name,
                d,
                time(|| drop(index_select_step(&x, dim, &indices))),
                sum,
            );
        }
    }

    for dim in [0isize, 1] {
        let name = format!("cat 8 x [512, 512] dim {dim} + read back");
        if !wants(&name) {
            continue;
        }
        for d in &devices {
            let mut rng = StdRng::seed_from_u64(8);
            let parts: Vec<Tensor> = (0..8)
                .map(|_| Tensor::randn_with(&[512, 512], &mut rng, d))
                .collect();
            let sum = checksum(&Tensor::cat(&parts, dim).to_vec());
            row(
                &name,
                d,
                time(|| drop(Tensor::cat(&parts, dim).to_vec())),
                sum,
            );
        }
    }

    let name = "cross_entropy [4096, 32000] class ids fwd+bwd";
    if wants(name) {
        for d in &devices {
            let mut rng = StdRng::seed_from_u64(9);
            let logits = Tensor::randn_with(&[4096, 32_000], &mut rng, d).requires_grad_();
            let targets: Vec<f32> = (0..4096u64)
                .map(|i| ((i * 104_729) % 32_000) as f32)
                .collect();
            let targets = Tensor::from_vec(targets, &[4096], d);
            let step = || {
                logits.zero_grad();
                let loss = F::cross_entropy(&logits, &targets);
                loss.backward();
                let grad = logits.grad().expect("logits gradient");
                // Read back the loss and one value per row, not 500 MB.
                let mut out = vec![loss.item()];
                out.extend(grad.narrow(1, 0, 1).to_vec());
                out
            };
            let sum = checksum(&step());
            row(name, d, time(|| drop(step())), sum);
        }
    }
}
