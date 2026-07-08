//! Supported model types and auto-detection

use crate::config::ModelConfig;

/// Supported model architectures
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    /// Llama family (Llama 2, Llama 3, etc.)
    Llama,
    /// Mistral models
    Mistral,
    /// Microsoft Phi models
    Phi,
    /// Qwen models (Alibaba)
    Qwen,
    /// Google Gemma models
    Gemma,
    /// Unknown/unsupported
    Unknown,
}

impl ModelType {
    /// Detect model type from config.
    pub fn from_config(config: &ModelConfig) -> Self {
        // Check architectures field
        for arch in &config.architectures {
            let arch_lower = arch.to_lowercase();
            if arch_lower.contains("llama") {
                return Self::Llama;
            }
            if arch_lower.contains("mistral") {
                return Self::Mistral;
            }
            if arch_lower.contains("phi") {
                return Self::Phi;
            }
            if arch_lower.contains("qwen") {
                return Self::Qwen;
            }
            if arch_lower.contains("gemma") {
                return Self::Gemma;
            }
        }

        // Check model_type field
        let model_type = config.model_type.to_lowercase();
        if model_type.contains("llama") {
            return Self::Llama;
        }
        if model_type.contains("mistral") {
            return Self::Mistral;
        }
        if model_type.contains("phi") {
            return Self::Phi;
        }
        if model_type.contains("qwen") {
            return Self::Qwen;
        }
        if model_type.contains("gemma") {
            return Self::Gemma;
        }

        Self::Unknown
    }

    /// Check if the model type is supported by Rusty.
    pub fn is_supported(&self) -> bool {
        matches!(
            self,
            Self::Llama | Self::Mistral | Self::Phi | Self::Qwen | Self::Gemma
        )
    }

    /// Get the display name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Llama => "Llama",
            Self::Mistral => "Mistral",
            Self::Phi => "Phi",
            Self::Qwen => "Qwen",
            Self::Gemma => "Gemma",
            Self::Unknown => "Unknown",
        }
    }
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Supported quantization formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantFormat {
    /// Full precision FP32
    FP32,
    /// Half precision FP16
    FP16,
    /// Brain float 16
    BF16,
    /// 8-bit integer (GPTQ, etc.)
    Int8,
    /// 4-bit integer
    Int4,
}

impl QuantFormat {
    /// Bytes per parameter.
    pub fn bytes_per_param(&self) -> f32 {
        match self {
            Self::FP32 => 4.0,
            Self::FP16 | Self::BF16 => 2.0,
            Self::Int8 => 1.0,
            Self::Int4 => 0.5,
        }
    }

    /// Estimate model size in GB.
    pub fn estimate_size_gb(&self, num_params: usize) -> f32 {
        (num_params as f32 * self.bytes_per_param()) / 1_000_000_000.0
    }
}
