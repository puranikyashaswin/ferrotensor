//! Example: Flash Attention
//!
//! This example demonstrates the memory-efficient Flash Attention implementation.
//!
//! Run with: cargo run --example flash_attention --release

use rusty_backend::{ComputeEngine, UnifiedTensor, WgpuContext};
use rusty_graph::FlashAttention;

fn main() {
    println!("Rusty ML - Flash Attention Example");
    println!();

    // Initialize GPU
    let ctx = std::sync::Arc::new(pollster::block_on(async { WgpuContext::new().await }));
    let engine = ComputeEngine::new(&ctx);

    // Print GPU info
    let info = ctx.adapter.get_info();
    println!("[GPU] {} ({:?})", info.name, info.backend);

    // Configuration
    let batch_size = 1;
    let seq_len = 512;
    let hidden_dim = 768;
    let num_heads = 12;
    let head_dim = hidden_dim / num_heads;

    println!();
    println!("[CONFIG] Configuration:");
    println!("         Batch size:  {}", batch_size);
    println!("         Sequence:    {}", seq_len);
    println!("         Hidden dim:  {}", hidden_dim);
    println!("         Num heads:   {}", num_heads);
    println!("         Head dim:    {}", head_dim);

    // Create Flash Attention layer
    let flash_attn = FlashAttention::new(hidden_dim, num_heads, true);

    // Memory comparison
    println!();
    println!("[MEMORY] Memory Analysis:");
    println!("         {}", flash_attn.memory_savings(seq_len));

    // Standard attention memory
    let standard_mem = seq_len * seq_len * 4; // O(N^2)
    let flash_mem = seq_len * head_dim * 4; // O(N)

    println!(
        "         Standard attention: {} MB",
        standard_mem / 1_000_000
    );
    println!("         Flash attention:    {} KB", flash_mem / 1_000);
    println!(
        "         Savings:            {:.0}x",
        standard_mem as f32 / flash_mem as f32
    );

    // Create Q, K, V tensors
    println!();
    println!("[INIT] Creating Q, K, V tensors...");

    let qkv_size = batch_size * seq_len * hidden_dim;
    let q_data: Vec<f32> = (0..qkv_size)
        .map(|i| ((i % 100) as f32 - 50.0) / 100.0)
        .collect();
    let k_data: Vec<f32> = (0..qkv_size)
        .map(|i| ((i % 100) as f32 - 50.0) / 100.0)
        .collect();
    let v_data: Vec<f32> = (0..qkv_size).map(|i| (i % 100) as f32 / 100.0).collect();

    let q = UnifiedTensor::new(&ctx, &q_data, &[batch_size, seq_len, hidden_dim]);
    let k = UnifiedTensor::new(&ctx, &k_data, &[batch_size, seq_len, hidden_dim]);
    let v = UnifiedTensor::new(&ctx, &v_data, &[batch_size, seq_len, hidden_dim]);

    println!("       Q: {:?}", q.shape);
    println!("       K: {:?}", k.shape);
    println!("       V: {:?}", v.shape);

    // Forward pass
    println!();
    println!("[COMPUTE] Running Flash Attention forward pass...");
    let output = flash_attn.forward(&q, &k, &v, &ctx, &engine);
    println!("          Output: {:?}", output.shape);

    // Read result
    let result = pollster::block_on(output.to_vec(&ctx));
    println!("          First values: {:?}", &result[..5]);
    println!(
        "          Mean: {:.6}",
        result.iter().sum::<f32>() / result.len() as f32
    );

    // Causal mask info
    println!();
    println!("[INFO] Causal Masking: Enabled");
    println!("       Each position can only attend to previous positions.");
    println!("       This is standard for autoregressive language models.");

    println!();
    println!("[DONE] Flash Attention completed successfully!");
}
