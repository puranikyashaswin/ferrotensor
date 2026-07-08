
#[cfg(test)]
mod tests {
    use rusty_autograd::Tensor;
    use rusty_backend::{ComputeEngine, UnifiedTensor, WgpuContext};
    use std::sync::Arc;

    fn gpu_tests_enabled() -> bool {
        std::env::var("RUSTY_RUN_GPU_TESTS").as_deref() == Ok("1")
    }

    async fn run_grad_check() {
        if !gpu_tests_enabled() {
            eprintln!("skipping flash grad check; set RUSTY_RUN_GPU_TESTS=1 to enable");
            return;
        }

        let ctx = Arc::new(WgpuContext::new().await);
        let engine = Arc::new(ComputeEngine::new(&ctx));

        // 1. Setup Input: [1, 4, 1, 16] (Batch=1, Seq=4, Heads=1, Dim=16)
        // Small size for numerical check
        let batch = 1;
        let seq = 4;
        let heads = 1;
        let dim = 16;
        let shape = vec![batch, seq, heads, dim];
        
        let size = batch * seq * heads * dim;
        let mut data = vec![0.0f32; size];
        for i in 0..size {
            data[i] = (i as f32 * 0.1).sin(); // Random-ish data
        }
        
        let q_raw = UnifiedTensor::new(&ctx, &data, &shape);
        let k_raw = UnifiedTensor::new(&ctx, &data, &shape); // Self-attention
        let v_raw = UnifiedTensor::new(&ctx, &data, &shape);
        
        let q = Tensor::new(ctx.clone(), q_raw);
        let k = Tensor::new(ctx.clone(), k_raw);
        let v = Tensor::new(ctx.clone(), v_raw);
        
        // Enable gradients
        *q.grad.borrow_mut() = Some(UnifiedTensor::zeros(&ctx, &shape));
        *k.grad.borrow_mut() = Some(UnifiedTensor::zeros(&ctx, &shape));
        *v.grad.borrow_mut() = Some(UnifiedTensor::zeros(&ctx, &shape));
        
        // 2. Forward Pass
        let out = Tensor::flash_attention(&q, &k, &v, None, false, 0.0, 0, &engine);
        
        // 3. Backward Pass (Analytical)
        // Loss = Sum(Out)
        let loss_grad_raw = UnifiedTensor::new(&ctx, &vec![1.0f32; size], &shape);
        out.accumulate_grad(&engine, &loss_grad_raw);
        out.backward(&engine);
        
        let ana_grad_q = q.grad.borrow().as_ref().unwrap().to_vec(&ctx).await;
        let ana_grad_k = k.grad.borrow().as_ref().unwrap().to_vec(&ctx).await;
        let ana_grad_v = v.grad.borrow().as_ref().unwrap().to_vec(&ctx).await;

        let eps = 1e-3;
        let idx_to_check = 5;

        async fn numerical_grad(
            ctx: &Arc<WgpuContext>,
            engine: &Arc<ComputeEngine>,
            base: &[f32],
            shape: &[usize],
            idx: usize,
            target: char,
            q: &Tensor,
            k: &Tensor,
            v: &Tensor,
            eps: f32,
        ) -> f32 {
            let mut data_plus = base.to_vec();
            data_plus[idx] += eps;
            let mut data_minus = base.to_vec();
            data_minus[idx] -= eps;

            let mk = |buf: &[f32]| Tensor::new(ctx.clone(), UnifiedTensor::new(ctx, buf, shape));
            let (q_plus, k_plus, v_plus) = match target {
                'q' => (mk(&data_plus), k.clone(), v.clone()),
                'k' => (q.clone(), mk(&data_plus), v.clone()),
                'v' => (q.clone(), k.clone(), mk(&data_plus)),
                _ => unreachable!(),
            };
            let (q_minus, k_minus, v_minus) = match target {
                'q' => (mk(&data_minus), k.clone(), v.clone()),
                'k' => (q.clone(), mk(&data_minus), v.clone()),
                'v' => (q.clone(), k.clone(), mk(&data_minus)),
                _ => unreachable!(),
            };

            let out_plus =
                Tensor::flash_attention(&q_plus, &k_plus, &v_plus, None, false, 0.0, 0, engine);
            let out_minus = Tensor::flash_attention(
                &q_minus, &k_minus, &v_minus, None, false, 0.0, 0, engine,
            );
            let sum_plus: f32 = out_plus.data.to_vec(ctx).await.iter().sum();
            let sum_minus: f32 = out_minus.data.to_vec(ctx).await.iter().sum();
            (sum_plus - sum_minus) / (2.0 * eps)
        }

        let num_grad_q =
            numerical_grad(&ctx, &engine, &data, &shape, idx_to_check, 'q', &q, &k, &v, eps)
                .await;
        let num_grad_k =
            numerical_grad(&ctx, &engine, &data, &shape, idx_to_check, 'k', &q, &k, &v, eps)
                .await;
        let num_grad_v =
            numerical_grad(&ctx, &engine, &data, &shape, idx_to_check, 'v', &q, &k, &v, eps)
                .await;

        assert!(
            (ana_grad_q[idx_to_check] - num_grad_q).abs() < 1e-2,
            "q gradient mismatch: ana={}, num={}",
            ana_grad_q[idx_to_check],
            num_grad_q
        );
        assert!(
            (ana_grad_k[idx_to_check] - num_grad_k).abs() < 1e-2,
            "k gradient mismatch: ana={}, num={}",
            ana_grad_k[idx_to_check],
            num_grad_k
        );
        assert!(
            (ana_grad_v[idx_to_check] - num_grad_v).abs() < 1e-2,
            "v gradient mismatch: ana={}, num={}",
            ana_grad_v[idx_to_check],
            num_grad_v
        );
    }

    #[test]
    fn test_flash_grad_check() {
        pollster::block_on(run_grad_check());
    }
}
