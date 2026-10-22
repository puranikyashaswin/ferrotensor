//! With `FERROTENSOR_RUN_GPU_TESTS=1`: whole layers give the same outputs and
//! gradients on the GPU as on the CPU, and GPU modules survive a
//! safetensors round trip.
//!
//! Each test builds a layer twice, once per device, copies the CPU layer's
//! (randomised) weights into the GPU one, and compares the forward values,
//! the input gradient and every parameter gradient.

use rand::rngs::StdRng;
use rand::SeedableRng;

use super::*;
use crate::serialize;
use crate::{Device, Tensor};

fn gpu() -> Option<Device> {
    if std::env::var("FERROTENSOR_RUN_GPU_TESTS").as_deref() != Ok("1") {
        eprintln!("skipping GPU layer parity; set FERROTENSOR_RUN_GPU_TESTS=1 to enable");
        return None;
    }
    Some(Device::gpu().expect("FERROTENSOR_RUN_GPU_TESTS=1 but no GPU is available"))
}

fn close(gpu: &[f32], cpu: &[f32], what: &str) {
    assert_eq!(gpu.len(), cpu.len(), "{what}: length");
    for (k, (x, y)) in gpu.iter().zip(cpu).enumerate() {
        assert!(
            (x - y).abs() <= 1e-4 + 1e-3 * y.abs(),
            "{what}[{k}]: gpu {x} vs cpu {y}"
        );
    }
}

/// Give every parameter random values (so LayerNorm's ones and zeros do not
/// hide a wrong gradient) and copy them to the matching GPU parameters.
fn sync_weights(cpu: &dyn Module, gpu: &dyn Module) {
    let mut rng = StdRng::seed_from_u64(11);
    let (cpu_params, gpu_params) = (cpu.named_parameters(), gpu.named_parameters());
    assert_eq!(cpu_params.len(), gpu_params.len());
    for ((name, c), (gpu_name, g)) in cpu_params.iter().zip(&gpu_params) {
        assert_eq!(name, gpu_name);
        c.assign(&Tensor::randn_with(&c.shape, &mut rng, c.device()).mul_scalar(0.5));
        g.assign(c);
    }
}

/// Forward values, input gradient (when `x` is differentiable) and parameter
/// gradients of `sum(w * module(x))` for fixed weights `w`.
fn run(module: &dyn Module, x: &Tensor, input_grad: bool) -> Vec<(String, Vec<f32>)> {
    let x = if input_grad {
        x.detach().requires_grad_()
    } else {
        x.clone()
    };
    module.zero_grad();
    let out = module.forward(&x);
    let mut rng = StdRng::seed_from_u64(3);
    let w = Tensor::uniform_with(&out.shape, 0.5, 1.5, &mut rng, out.device());
    out.mul(&w).sum().backward();
    let mut results = vec![("forward".to_string(), out.to_vec())];
    if input_grad {
        results.push((
            "input grad".to_string(),
            x.grad().expect("input grad").to_vec(),
        ));
    }
    for (name, p) in module.named_parameters() {
        let grad = p.grad().unwrap_or_else(|| panic!("no gradient for {name}"));
        results.push((format!("grad {name}"), grad.to_vec()));
    }
    results
}

/// Build the layer on both devices with `build`, then compare on input `x`.
fn check_layer<M: Module>(name: &str, build: impl Fn(&Device) -> M, x: &Tensor, input_grad: bool) {
    let Some(gpu) = gpu() else { return };
    let cpu_layer = build(&Device::cpu());
    let gpu_layer = build(&gpu);
    sync_weights(&cpu_layer, &gpu_layer);
    let on_cpu = run(&cpu_layer, x, input_grad);
    let on_gpu = run(&gpu_layer, &x.to_device(&gpu), input_grad);
    assert_eq!(on_cpu.len(), on_gpu.len());
    for ((what, c), (_, g)) in on_cpu.iter().zip(&on_gpu) {
        close(g, c, &format!("{name} {what}"));
    }
}

fn input(shape: &[usize]) -> Tensor {
    let mut rng = StdRng::seed_from_u64(29);
    Tensor::randn_with(shape, &mut rng, &Device::cpu())
}

#[test]
fn gpu_linear_matches_cpu() {
    check_layer("linear", |d| Linear::new(6, 5, d), &input(&[4, 6]), true);
}

#[test]
fn gpu_layer_norm_matches_cpu() {
    check_layer(
        "layer_norm",
        |d| LayerNorm::new(&[3, 4], d),
        &input(&[2, 5, 3, 4]),
        true,
    );
}

#[test]
fn gpu_rms_norm_matches_cpu() {
    check_layer("rms_norm", |d| RMSNorm::new(8, d), &input(&[3, 8]), true);
}

#[test]
fn gpu_embedding_matches_cpu() {
    let ids = Tensor::from_data(&[3.0, 0.0, 9.0, 3.0, 5.0, 3.0], &[2, 3], &Device::cpu());
    check_layer("embedding", |d| Embedding::new(10, 4, d), &ids, false);
}

#[test]
fn gpu_attention_matches_cpu() {
    for causal in [false, true] {
        check_layer(
            &format!("attention causal={causal}"),
            |d| MultiHeadAttention::new(16, 4, d).causal(causal),
            &input(&[2, 6, 16]),
            true,
        );
    }
    check_layer(
        "attention unbatched",
        |d| MultiHeadAttention::new(8, 2, d).causal(true),
        &input(&[5, 8]),
        true,
    );
}

fn mlp(d: &Device) -> Sequential {
    Sequential::new(d)
        .add(Linear::new(6, 16, d))
        .add(GELU)
        .add(LayerNorm::new(&[16], d))
        .add(Linear::new(16, 16, d))
        .add(Tanh)
        .add(MultiHeadAttention::new(16, 2, d).causal(true))
        .add(RMSNorm::new(16, d))
        .add(Linear::new(16, 3, d))
}

#[test]
fn gpu_sequential_matches_cpu() {
    check_layer("sequential", mlp, &input(&[2, 4, 6]), true);
}

fn temp(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "ferrotensor_nn_{}_{name}.safetensors",
        std::process::id()
    ))
}

#[test]
fn gpu_module_round_trips_through_safetensors() {
    let Some(gpu) = gpu() else { return };
    let cpu = Device::cpu();
    let trained = mlp(&gpu);
    sync_weights(&mlp(&cpu), &trained);
    let path = temp("gpu_module");
    serialize::save_module(&trained, &path).unwrap();
    // Load into a fresh module on each device.
    let (on_gpu, on_cpu) = (mlp(&gpu), mlp(&cpu));
    serialize::load_module(&on_gpu, &path).unwrap();
    serialize::load_module(&on_cpu, &path).unwrap();
    std::fs::remove_file(&path).ok();

    for ((name, a), ((_, b), (_, c))) in trained.named_parameters().iter().zip(
        on_gpu
            .named_parameters()
            .iter()
            .zip(&on_cpu.named_parameters()),
    ) {
        assert!(
            b.device().same_as(&gpu) && c.device().same_as(&cpu),
            "{name}"
        );
        assert_eq!(a.to_vec(), b.to_vec(), "{name} on the GPU");
        assert_eq!(a.to_vec(), c.to_vec(), "{name} on the CPU");
    }
    let x = input(&[2, 4, 6]);
    let want = trained.forward(&x.to_device(&gpu)).to_vec();
    assert_eq!(on_gpu.forward(&x.to_device(&gpu)).to_vec(), want);
    close(&on_cpu.forward(&x).to_vec(), &want, "reloaded forward");
}

#[test]
fn gpu_tensors_round_trip_through_safetensors() {
    let Some(gpu) = gpu() else { return };
    let mut map = std::collections::HashMap::new();
    map.insert("w".to_string(), input(&[3, 5]).to_device(&gpu));
    map.insert("s".to_string(), Tensor::scalar(2.5, &gpu));
    let path = temp("gpu_tensors");
    serialize::save_tensors(&map, &path).unwrap();
    let loaded = serialize::load_tensors(&path, &gpu).unwrap();
    std::fs::remove_file(&path).ok();
    for (k, v) in &map {
        assert!(loaded[k].device().same_as(&gpu));
        assert_eq!(loaded[k].shape, v.shape);
        assert_eq!(loaded[k].to_vec(), v.to_vec());
    }
}
