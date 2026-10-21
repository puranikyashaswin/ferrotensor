//! Flash Attention Correctness Tests
//!
//! Comprehensive test suite to verify numerical accuracy and correctness
//! of the Flash Attention implementation.

#[cfg(test)]
mod tests {
    use rusty_backend::{ComputeEngine, UnifiedTensor, WgpuContext};
    use rusty_graph::FlashAttention;
    use std::sync::Arc;

    fn gpu_tests_enabled() -> bool {
        std::env::var("RUSTY_RUN_GPU_TESTS").as_deref() == Ok("1")
    }

    fn setup() -> Option<(Arc<WgpuContext>, ComputeEngine)> {
        if !gpu_tests_enabled() {
            eprintln!("skipping flash attention correctness tests; set RUSTY_RUN_GPU_TESTS=1");
            return None;
        }
        let ctx = Arc::new(pollster::block_on(async { WgpuContext::new().await }));
        let engine = ComputeEngine::new(&ctx);
        Some((ctx, engine))
    }

    #[test]
    fn test_flash_attention_shape() {
        let Some((ctx, engine)) = setup() else { return };
        let flash_attn = FlashAttention::new(768, 12, true);

        let seq_len = 128;
        let hidden_dim = 768;

        let q_data: Vec<f32> = (0..seq_len * hidden_dim)
            .map(|i| (i as f32 % 100.0) / 100.0)
            .collect();

        let q = UnifiedTensor::new(&ctx, &q_data, &[1, seq_len, hidden_dim]);
        let k = UnifiedTensor::new(&ctx, &q_data, &[1, seq_len, hidden_dim]);
        let v = UnifiedTensor::new(&ctx, &q_data, &[1, seq_len, hidden_dim]);

        let output = flash_attn.forward(&q, &k, &v, &ctx, &engine);

        assert_eq!(output.shape, vec![1, seq_len, hidden_dim]);
    }

    #[test]
    fn test_causal_masking() {
        let Some((ctx, engine)) = setup() else { return };
        let flash_attn = FlashAttention::new(64, 1, true); // Causal

        let seq_len = 8;
        let hidden_dim = 64;

        // Create simple input where we can verify causality
        let q_data: Vec<f32> = vec![1.0; seq_len * hidden_dim];
        let k_data: Vec<f32> = (0..seq_len * hidden_dim)
            .map(|i| (i / hidden_dim) as f32)
            .collect();
        let v_data = k_data.clone();

        let q = UnifiedTensor::new(&ctx, &q_data, &[seq_len, hidden_dim]);
        let k = UnifiedTensor::new(&ctx, &k_data, &[seq_len, hidden_dim]);
        let v = UnifiedTensor::new(&ctx, &v_data, &[seq_len, hidden_dim]);

        let output = flash_attn.forward(&q, &k, &v, &ctx, &engine);
        let result = pollster::block_on(output.to_vec(&ctx));

        // Verify output is not all zeros
        assert!(result.iter().any(|&x| x.abs() > 1e-6));

        // Query 0 sees only key 0, so its output is v[0] (all zeros here);
        // query i averages v[0..=i] with weights favouring larger keys, so
        // every output row stays within [0, i].
        assert!(result[..hidden_dim].iter().all(|&x| x.abs() < 1e-6));
        for i in 0..seq_len {
            for &x in &result[i * hidden_dim..(i + 1) * hidden_dim] {
                assert!((-1e-4..=i as f32 + 1e-4).contains(&x), "row {i}: {x}");
            }
        }
    }

    #[test]
    fn test_attention_normalization() {
        let Some((ctx, engine)) = setup() else { return };
        let flash_attn = FlashAttention::new(64, 1, false);

        let seq_len = 16;
        let hidden_dim = 64;

        let q_data: Vec<f32> = (0..seq_len * hidden_dim)
            .map(|i| ((i % 100) as f32 - 50.0) / 100.0)
            .collect();

        let q = UnifiedTensor::new(&ctx, &q_data, &[seq_len, hidden_dim]);
        let k = UnifiedTensor::new(&ctx, &q_data, &[seq_len, hidden_dim]);
        let v = UnifiedTensor::new(&ctx, &q_data, &[seq_len, hidden_dim]);

        let output = flash_attn.forward(&q, &k, &v, &ctx, &engine);
        let result = pollster::block_on(output.to_vec(&ctx));

        // Check that output is reasonable (not NaN, not Inf)
        for &val in &result {
            assert!(val.is_finite(), "Output contains non-finite values");
        }

        // Check that output is not all zeros
        let sum: f32 = result.iter().sum();
        assert!(sum.abs() > 1e-4, "Output is all zeros");
    }

    #[test]
    fn test_backward_pass_shapes() {
        let Some((ctx, engine)) = setup() else { return };
        let flash_attn = FlashAttention::new(64, 1, true);

        let seq_len = 64;
        let head_dim = 64;

        let q_data: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| (i as f32 % 100.0) / 100.0)
            .collect();

        let q = UnifiedTensor::new(&ctx, &q_data, &[seq_len, head_dim]);
        let k = UnifiedTensor::new(&ctx, &q_data, &[seq_len, head_dim]);
        let v = UnifiedTensor::new(&ctx, &q_data, &[seq_len, head_dim]);

        let output = flash_attn.forward(&q, &k, &v, &ctx, &engine);
        let grad_output = UnifiedTensor::new(&ctx, &q_data, &[seq_len, head_dim]);

        let grads = flash_attn.backward(&q, &k, &v, &output, &grad_output, &ctx, &engine);

        assert_eq!(grads.grad_q.shape, q.shape);
        assert_eq!(grads.grad_k.shape, k.shape);
        assert_eq!(grads.grad_v.shape, v.shape);
    }

    #[test]
    fn test_memory_savings() {
        let flash_attn = FlashAttention::new(768, 12, true);
        let savings = flash_attn.memory_savings(2048);

        assert!(savings.contains("x less memory"));
        assert!(savings.contains("2048"));
    }

    #[test]
    fn test_multi_head_creation() {
        let flash_attn = FlashAttention::new(768, 12, true);
        assert!(!flash_attn.memory_savings(1024).is_empty());
    }

    #[test]
    fn test_different_sequence_lengths() {
        let Some((ctx, engine)) = setup() else { return };
        let flash_attn = FlashAttention::new(256, 4, true);

        let test_lengths = vec![16, 32, 64, 128, 256];

        for seq_len in test_lengths {
            let hidden_dim = 256;
            let q_data: Vec<f32> = (0..seq_len * hidden_dim)
                .map(|i| (i as f32 % 100.0) / 100.0)
                .collect();

            let q = UnifiedTensor::new(&ctx, &q_data, &[1, seq_len, hidden_dim]);
            let k = UnifiedTensor::new(&ctx, &q_data, &[1, seq_len, hidden_dim]);
            let v = UnifiedTensor::new(&ctx, &q_data, &[1, seq_len, hidden_dim]);

            let output = flash_attn.forward(&q, &k, &v, &ctx, &engine);
            assert_eq!(output.shape, vec![1, seq_len, hidden_dim]);

            let result = pollster::block_on(output.to_vec(&ctx));
            assert!(result.iter().all(|&x| x.is_finite()));
        }
    }
}
