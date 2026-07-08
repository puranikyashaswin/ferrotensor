use rusty_autograd::Tensor;
use rusty_backend::{ComputeEngine, UnifiedTensor, WgpuContext};
use rusty_graph::{Attention, Linear, LlamaBlock, RMSNorm, MLP};
use std::sync::Arc;

fn main() {
    env_logger::init();
    pollster::block_on(async {
        let ctx = Arc::new(WgpuContext::new().await);
        let engine = ComputeEngine::new(&ctx);

        // Dimensions
        let batch = 1;
        let dim = 4;
        let hidden = 16; // MLP expansion

        let mk_tensor = |data: Vec<f32>, shape: &[usize]| {
            Tensor::new(ctx.clone(), UnifiedTensor::new(&ctx, &data, shape))
        };

        // --- Create Weights (Ones/Identity for simple debug) ---
        let w_gate = mk_tensor(vec![1.0; dim * hidden], &[dim, hidden]);
        let w_up = mk_tensor(vec![1.0; dim * hidden], &[dim, hidden]);
        let w_down = mk_tensor(vec![1.0; hidden * dim], &[hidden, dim]);
        let w_norm = mk_tensor(vec![1.0; dim], &[dim]);

        // --- Build Layer ---
        let mlp = MLP {
            gate_proj: Linear::new(w_gate),
            up_proj: Linear::new(w_up),
            down_proj: Linear::new(w_down),
        };

        let mk_lin = || Linear::new(mk_tensor(vec![0.1; dim * dim], &[dim, dim]));

        let block = LlamaBlock {
            norm1: RMSNorm::new(w_norm.clone(), 1e-5),
            attn: Attention {
                q_proj: mk_lin(),
                k_proj: mk_lin(),
                v_proj: mk_lin(),
                o_proj: mk_lin(),
                n_heads: 1,
                head_dim: 4,
                num_kv_heads: 1,
                dropout_prob: 0.0,
            },
            norm2: RMSNorm::new(w_norm.clone(), 1e-5),
            mlp,
        };

        // --- Input ---
        let input = mk_tensor(vec![1.0, 1.0, 1.0, 1.0], &[batch, 1, dim]);

        println!("Running Llama Block...");
        let out = block.forward(&engine, &input, 0, None, None);

        let result = out.data.to_vec(&ctx).await;
        println!("Block Output: {:?}", result);
    });
}
