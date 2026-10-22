//! Embedding Layer
//!
//! Token embedding lookup table.

use super::Module;
use crate::tensor::Tensor;
use crate::Device;

/// Embedding layer: maps integer token ids to learned dense vectors.
///
/// # Example
/// ```rust
/// use ferrotensor::prelude::*;
///
/// let device = Device::cpu();
/// let emb = nn::Embedding::new(100, 16, &device);
/// let out = emb.forward_ids(&[3, 7, 3]);
/// assert_eq!(out.shape, vec![3, 16]);
/// ```
pub struct Embedding {
    /// Embedding weights [vocab_size, embedding_dim]
    pub weight: Tensor,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Embedding dimension
    pub embedding_dim: usize,
}

impl Embedding {
    /// Create an Embedding with N(0, 0.02^2) weights.
    pub fn new(vocab_size: usize, embedding_dim: usize, device: &Device) -> Self {
        let weight = Tensor::randn(&[vocab_size, embedding_dim], device)
            .mul_scalar(0.02)
            .requires_grad_();
        Self {
            weight,
            vocab_size,
            embedding_dim,
        }
    }

    /// Create an Embedding from existing weights (`[vocab, dim]`).
    pub fn from_weights(weight: Tensor) -> Self {
        let vocab_size = weight.shape[0];
        let embedding_dim = weight.shape[1];
        Self {
            weight,
            vocab_size,
            embedding_dim,
        }
    }

    /// Look up `ids`; returns `[ids.len(), embedding_dim]` (differentiable
    /// in the weights).
    pub fn forward_ids(&self, ids: &[usize]) -> Tensor {
        crate::functional::embedding(&self.weight, ids)
    }

    /// Look up `u32` token ids.
    pub fn forward_indices(&self, token_ids: &[u32]) -> Tensor {
        let ids: Vec<usize> = token_ids.iter().map(|&id| id as usize).collect();
        self.forward_ids(&ids)
    }
}

impl Module for Embedding {
    /// `x` holds token ids stored as `f32`; the output has shape
    /// `x.shape + [embedding_dim]`.
    ///
    /// On a GPU the lookup reads the ids on the device, so the forward pass
    /// needs no copy to the host. Ids outside `[0, vocab_size)` give rows of
    /// zeros and no gradient.
    fn forward(&self, x: &Tensor) -> Tensor {
        self.weight.embedding(x)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![("weight".to_string(), self.weight.clone())]
    }
}

impl std::fmt::Debug for Embedding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Embedding")
            .field("vocab_size", &self.vocab_size)
            .field("embedding_dim", &self.embedding_dim)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    /// `forward` on an id tensor and `forward_ids` on the same ids: values
    /// and weight gradients, on `device`.
    fn both_paths(device: &Device) -> [(Vec<f32>, Vec<f32>); 2] {
        let mut rng = rand::rngs::StdRng::seed_from_u64(5);
        let weight = Tensor::randn_with(&[11, 6], &mut rng, device);
        let ids = [3usize, 0, 10, 3, 7, 3];
        let run = |by_tensor: bool| {
            let emb = Embedding::from_weights(weight.deep_clone().requires_grad_());
            let out = if by_tensor {
                let x = Tensor::from_vec(ids.iter().map(|&i| i as f32).collect(), &[2, 3], device);
                emb.forward(&x)
            } else {
                emb.forward_ids(&ids).reshape(&[2, 3, 6])
            };
            assert_eq!(out.shape, vec![2, 3, 6]);
            let seed = Tensor::from_vec(
                (0..36).map(|i| i as f32 * 0.1).collect(),
                &[2, 3, 6],
                device,
            );
            out.mul(&seed).sum().backward();
            (out.to_vec(), emb.weight.grad().unwrap().to_vec())
        };
        [run(true), run(false)]
    }

    #[test]
    fn forward_matches_forward_ids() {
        let [by_tensor, by_ids] = both_paths(&Device::cpu());
        assert_eq!(by_tensor, by_ids);
    }

    #[test]
    fn out_of_range_ids_give_zero_rows_and_no_gradient() {
        let device = Device::cpu();
        let emb = Embedding::from_weights(Tensor::ones(&[4, 2], &device).requires_grad_());
        let x = Tensor::from_data(&[1.0, -1.0, 4.0], &[3], &device);
        let out = emb.forward(&x);
        assert_eq!(out.to_vec(), vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        out.sum().backward();
        assert_eq!(
            emb.weight.grad().unwrap().to_vec(),
            vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        );
    }

    /// With `FERROTENSOR_RUN_GPU_TESTS=1`: on the GPU, `forward` gathers with
    /// the ids still on the device and matches `forward_ids` and the CPU
    /// bit for bit, including out-of-range ids.
    #[test]
    fn gpu_forward_matches_cpu() {
        if std::env::var("FERROTENSOR_RUN_GPU_TESTS").as_deref() != Ok("1") {
            eprintln!("skipping GPU embedding test; set FERROTENSOR_RUN_GPU_TESTS=1 to enable");
            return;
        }
        let gpu = Device::gpu().expect("FERROTENSOR_RUN_GPU_TESTS=1 but no GPU is available");
        let on_gpu = both_paths(&gpu);
        assert_eq!(on_gpu[0], on_gpu[1]);
        assert_eq!(on_gpu, both_paths(&Device::cpu()));

        let weight = Tensor::ones(&[4, 2], &gpu);
        let x = Tensor::from_data(&[1.0, -1.0, 4.0, 3.9], &[4], &gpu);
        assert_eq!(
            Embedding::from_weights(weight).forward(&x).to_vec(),
            vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
        );
    }
}
