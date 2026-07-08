//! Normalization Layers
//!
//! Layer normalization and RMS normalization.

use super::Module;
use crate::tensor::Tensor;
use crate::Device;

/// Layer Normalization.
///
/// Normalizes over the last dimension with learnable scale (gamma) and shift (beta).
pub struct LayerNorm {
    /// Gamma (scale) parameter
    pub gamma: Tensor,
    /// Beta (bias) parameter  
    pub beta: Tensor,
    /// Epsilon for numerical stability
    pub eps: f32,
    /// Normalized shape
    pub normalized_shape: Vec<usize>,
}

impl LayerNorm {
    /// Create a new LayerNorm layer.
    pub fn new(normalized_shape: &[usize], device: &Device) -> Self {
        let size: usize = normalized_shape.iter().product();
        let gamma = Tensor::ones(&[size], device).requires_grad_();
        let beta = Tensor::zeros(&[size], device).requires_grad_();

        Self {
            gamma,
            beta,
            eps: 1e-5,
            normalized_shape: normalized_shape.to_vec(),
        }
    }

    /// Create LayerNorm with custom epsilon.
    pub fn with_eps(normalized_shape: &[usize], eps: f32, device: &Device) -> Self {
        let mut ln = Self::new(normalized_shape, device);
        ln.eps = eps;
        ln
    }
}

impl Module for LayerNorm {
    fn forward(&self, x: &Tensor) -> Tensor {
        // For now, use CPU fallback for LayerNorm
        // TODO: Optimize with GPU kernel
        let data = x.to_vec();
        let size = self.normalized_shape.iter().product::<usize>();
        
        // Assume the last dimension is the normalization dimension
        let total_elements = data.len();
        let num_rows = total_elements / size;

        // Compute mean and variance per row
        let gamma_data = self.gamma.to_vec();
        let beta_data = self.beta.to_vec();

        let mut result = Vec::with_capacity(total_elements);
        
        for row in 0..num_rows {
            let row_start = row * size;
            let row_end = row_start + size;
            let row_data = &data[row_start..row_end];
            
            // Compute mean for this row
            let mean: f32 = row_data.iter().sum::<f32>() / size as f32;
            
            // Compute variance for this row
            let variance: f32 = row_data.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / size as f32;
            let inv_std = 1.0 / (variance + self.eps).sqrt();

            // Normalize and apply affine for this row
            for (i, &v) in row_data.iter().enumerate() {
                let idx = i % size;
                let normalized = (v - mean) * inv_std;
                result.push(normalized * gamma_data[idx] + beta_data[idx]);
            }
        }

        Tensor::from_data(&result, &x.shape, &x.device)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
    }
}

impl std::fmt::Debug for LayerNorm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayerNorm")
            .field("normalized_shape", &self.normalized_shape)
            .field("eps", &self.eps)
            .finish()
    }
}

/// RMS Normalization (used in Llama, etc.)
///
/// Normalizes by root mean square without centering.
pub struct RMSNorm {
    /// Weight (scale) parameter
    pub weight: Tensor,
    /// Epsilon for numerical stability
    pub eps: f32,
    /// Hidden dimension
    pub hidden_size: usize,
}

impl RMSNorm {
    /// Create a new RMSNorm layer.
    pub fn new(hidden_size: usize, device: &Device) -> Self {
        let weight = Tensor::ones(&[hidden_size], device).requires_grad_();

        Self {
            weight,
            eps: 1e-5,
            hidden_size,
        }
    }

    /// Create RMSNorm with custom epsilon.
    pub fn with_eps(hidden_size: usize, eps: f32, device: &Device) -> Self {
        let mut rms = Self::new(hidden_size, device);
        rms.eps = eps;
        rms
    }

    /// Create from existing weights.
    pub fn from_weights(weight: Tensor, eps: f32) -> Self {
        let hidden_size = weight.numel();
        Self {
            weight,
            eps,
            hidden_size,
        }
    }
}

impl Module for RMSNorm {
    fn forward(&self, x: &Tensor) -> Tensor {
        // CPU fallback for now
        let data = x.to_vec();
        let size = self.hidden_size;
        
        let total_elements = data.len();
        let num_rows = total_elements / size;

        // Normalize and scale
        let weight_data = self.weight.to_vec();
        let mut result = Vec::with_capacity(total_elements);
        
        for row in 0..num_rows {
            let row_start = row * size;
            let row_end = row_start + size;
            let row_data = &data[row_start..row_end];
            
            // Compute RMS for this row
            let sq_sum: f32 = row_data.iter().map(|v| v * v).sum::<f32>();
            let rms = (sq_sum / size as f32 + self.eps).sqrt();
            let inv_rms = 1.0 / rms;

            // Normalize and scale for this row
            for (i, &v) in row_data.iter().enumerate() {
                let idx = i % size;
                result.push(v * inv_rms * weight_data[idx]);
            }
        }

        Tensor::from_data(&result, &x.shape, &x.device)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }
}

impl std::fmt::Debug for RMSNorm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RMSNorm")
            .field("hidden_size", &self.hidden_size)
            .field("eps", &self.eps)
            .finish()
    }
}
