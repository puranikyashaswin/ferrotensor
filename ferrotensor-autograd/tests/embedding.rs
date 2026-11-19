//! `Tensor::embedding` and `Tensor::embedding_gpu`: the gathered rows, the
//! scatter-add gradient against values worked out by hand or against each
//! other, and when the lookup records a node.
//!
//! These need a real GPU, so they are `#[ignore]`d by default. Run them with:
//!
//! ```text
//! cargo test -p ferrotensor-autograd --test embedding -- --ignored --test-threads=1
//! ```

#![allow(clippy::expect_used)]

use std::sync::Arc;

use ferrotensor_autograd::{no_grad, Tensor};
use ferrotensor_backend::{ComputeEngine, UnifiedTensor, WgpuContext};

fn setup() -> (Arc<WgpuContext>, ComputeEngine) {
    let ctx = Arc::new(pollster::block_on(WgpuContext::new()));
    let engine = ComputeEngine::new(&ctx);
    (ctx, engine)
}

fn host(ctx: &WgpuContext, t: &UnifiedTensor) -> Vec<f32> {
    pollster::block_on(t.to_vec(ctx))
}

const TABLE: [f32; 6] = [0.5, -1.0, 2.0, 0.25, -3.0, 4.0];

/// Loss `sum(w * embedding(table, [[2, 0, 2]]))` with `w` rows (1, 2),
/// (3, 4), (5, 6). Its gradient is `w`'s row summed into the row of each
/// position's id: row 0 gets (3, 4), row 1 nothing, row 2 (1 + 5, 2 + 6).
/// Every value is a small integer, so the comparison is exact.
#[test]
#[ignore = "needs a GPU"]
fn embedding_gradient_is_the_scatter_add_of_the_output_gradient() {
    let (ctx, engine) = setup();
    let table = Tensor::new(ctx.clone(), UnifiedTensor::new(&ctx, &TABLE, &[3, 2]));
    let out = Tensor::embedding(&table, &TABLE, &[&[2, 0, 2]]);
    assert_eq!(out.data.shape, [1, 3, 2]);
    assert_eq!(host(&ctx, &out.data), [-3.0, 4.0, 0.5, -1.0, -3.0, 4.0]);
    assert!(out.requires_grad && out.creator.is_some());

    let w = UnifiedTensor::new(&ctx, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 3, 2]);
    out.accumulate_grad(&engine, &w);
    out.backward(&engine);
    let grad = table
        .grad
        .borrow()
        .clone()
        .expect("the table has a gradient");
    assert_eq!(grad.shape, [3, 2]);
    assert_eq!(host(&ctx, &grad), [3.0, 4.0, 0.0, 0.0, 6.0, 8.0]);
}

/// An id outside the table looks up zeros and its gradient goes nowhere.
#[test]
#[ignore = "needs a GPU"]
fn an_id_outside_the_table_gives_zeros_and_no_gradient() {
    let (ctx, engine) = setup();
    let table = Tensor::new(ctx.clone(), UnifiedTensor::new(&ctx, &TABLE, &[3, 2]));
    let out = Tensor::embedding(&table, &TABLE, &[&[7, 1]]);
    assert_eq!(host(&ctx, &out.data), [0.0, 0.0, 2.0, 0.25]);
    out.backward(&engine);
    let grad = table
        .grad
        .borrow()
        .clone()
        .expect("the table has a gradient");
    assert_eq!(host(&ctx, &grad), [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]);
}

/// A frozen table, or any table inside `no_grad`, gives a constant: the
/// same rows with no node behind them.
#[test]
#[ignore = "needs a GPU"]
fn a_frozen_table_or_no_grad_records_nothing() {
    let (ctx, _engine) = setup();
    let rows: &[&[u32]] = &[&[1], &[0]];
    let frozen = Tensor::frozen(ctx.clone(), UnifiedTensor::new(&ctx, &TABLE, &[3, 2]));
    let out = Tensor::embedding(&frozen, &TABLE, rows);
    assert!(out.creator.is_none() && !out.requires_grad);
    assert_eq!(out.data.shape, [2, 1, 2]);
    assert_eq!(host(&ctx, &out.data), [2.0, 0.25, 0.5, -1.0]);

    let trainable = Tensor::new(ctx.clone(), UnifiedTensor::new(&ctx, &TABLE, &[3, 2]));
    let out = no_grad(|| Tensor::embedding(&trainable, &TABLE, rows));
    assert!(out.creator.is_none() && !out.requires_grad);
    assert_eq!(host(&ctx, &out.data), [2.0, 0.25, 0.5, -1.0]);
}

/// `embedding_gpu` against `embedding` on the same table, ids and output
/// gradient: the rows, the zeros for an id past the table and the summed
/// table gradient are equal bit for bit, and a frozen table records nothing.
#[test]
#[ignore = "needs a GPU"]
fn the_gpu_lookup_matches_the_host_lookup_bit_for_bit() {
    let (ctx, engine) = setup();
    let (vocab, hidden) = (7usize, 5usize);
    // Values with many mantissa bits, so a different summation order shows.
    let values: Vec<f32> = (0..vocab * hidden)
        .map(|i| ((i * 7919 % 1013) as f32 / 1013.0 - 0.5) * 3.1)
        .collect();
    let rows: &[&[u32]] = &[&[3, 0, 3, 6, 9], &[1, 3, 3, 0, 2]];
    let w: Vec<f32> = (0..rows.len() * 5 * hidden)
        .map(|i| ((i * 4583 % 997) as f32 / 997.0 - 0.5) * 1.7)
        .collect();

    let mut got = Vec::new();
    for on_gpu in [false, true] {
        let table = Tensor::new(
            ctx.clone(),
            UnifiedTensor::new(&ctx, &values, &[vocab, hidden]),
        );
        let out = if on_gpu {
            Tensor::embedding_gpu(&table, rows, &engine)
        } else {
            Tensor::embedding(&table, &values, rows)
        };
        assert_eq!(out.data.shape, [2, 5, hidden]);
        assert!(out.requires_grad && out.creator.is_some());
        out.accumulate_grad(&engine, &UnifiedTensor::new(&ctx, &w, &[2, 5, hidden]));
        out.backward(&engine);
        let grad = table
            .grad
            .borrow()
            .clone()
            .expect("the table has a gradient");
        let bits = |t: &UnifiedTensor| -> Vec<u32> {
            host(&ctx, t).iter().map(|v| v.to_bits()).collect()
        };
        got.push((bits(&out.data), bits(&grad)));
    }
    assert_eq!(got[0].0, got[1].0, "rows");
    assert_eq!(got[0].1, got[1].1, "table gradient");
    // Row 3 is looked up four times, row 4 never.
    assert!(got[1].1[4 * hidden..5 * hidden].iter().all(|&b| b == 0));

    let frozen = Tensor::frozen(
        ctx.clone(),
        UnifiedTensor::new(&ctx, &values, &[vocab, hidden]),
    );
    let out = Tensor::embedding_gpu(&frozen, rows, &engine);
    assert!(out.creator.is_none() && !out.requires_grad);
}
