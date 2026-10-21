//! Nearest-neighbour search over hashed character-trigram embeddings.
//!
//! Each document becomes a unit vector by hashing its character trigrams
//! into `DIM` buckets. This is a *lexical* embedding (typos and shared words
//! match; synonyms do not), chosen so the example needs no model weights.
//! The search itself is the useful part: one `[queries, DIM] @ [DIM, docs]`
//! matmul on the device, then a top-k on the host. Swap the embedding for a
//! learned one and the search code stays the same.
//!
//! Run with: cargo run --release -p rusty --example similarity_search

use rusty::prelude::*;

const DIM: usize = 512;

fn embed(text: &str) -> Vec<f32> {
    let chars: Vec<char> = format!("  {} ", text.to_lowercase()).chars().collect();
    let mut v = vec![0.0f32; DIM];
    for w in chars.windows(3) {
        // FNV-1a over the trigram's chars.
        let mut h: u32 = 2_166_136_261;
        for c in w {
            h = (h ^ *c as u32).wrapping_mul(16_777_619);
        }
        v[h as usize % DIM] += 1.0;
    }
    v
}

fn embed_all(texts: &[&str], device: &Device) -> Tensor {
    let flat: Vec<f32> = texts.iter().flat_map(|t| embed(t)).collect();
    let m = Tensor::from_data(&flat, &[texts.len(), DIM], device);
    // L2-normalise rows so a dot product is cosine similarity.
    let norm = m.square().sum_dim(1, true).sqrt();
    m.div(&norm)
}

fn main() {
    let device = Device::best();
    println!("device: {} ({})\n", device.name(), device.backend());

    let docs = [
        "How do I reset my password?",
        "Steps to change your account password",
        "Quarterly revenue grew by twelve percent",
        "Revenue increased 12% quarter over quarter",
        "The cat sat on the warm windowsill",
        "A kitten is sleeping on the window ledge",
        "Install the package with cargo add",
        "Run cargo install to get the command line tool",
        "Best hiking trails near the mountain lake",
        "Where to find scenic walking routes by the lake",
    ];
    let queries = [
        "forgot my password",
        "revenue growth this quarter",
        "kittens on a ledge",
        "how to install with cargo",
        "lake hiking trails",
    ];

    let doc_emb = embed_all(&docs, &device); // [docs, DIM]
    let query_emb = embed_all(&queries, &device); // [queries, DIM]
    let scores = query_emb.matmul(&doc_emb.transpose()).to_vec(); // [queries, docs]

    for (q, query) in queries.iter().enumerate() {
        let row = &scores[q * docs.len()..(q + 1) * docs.len()];
        let mut ranked: Vec<usize> = (0..docs.len()).collect();
        ranked.sort_by(|&a, &b| row[b].total_cmp(&row[a]));
        println!("query: {query:?}");
        for &d in ranked.iter().take(2) {
            println!("  {:.3}  {}", row[d], docs[d]);
        }
    }
}
