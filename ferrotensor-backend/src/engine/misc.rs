//! Top-k and argmax selection, depthwise 1-D convolution, and the embedding
//! lookup with its gradient.

use super::{grid, groups64, uniform, ComputeEngine};
use crate::{UnifiedTensor, WgpuContext};

impl ComputeEngine {
    /// Writes the indices of the `k` largest values of `input`, largest first,
    /// as `u32`s into `out_idx`. Runs on a single GPU thread; meant for sampling
    /// from one logits vector.
    pub fn top_k(&self, ctx: &WgpuContext, input: &UnifiedTensor, out_idx: &UnifiedTensor, k: u32) {
        let params = uniform(ctx, bytemuck::cast_slice(&[k]));
        self.run(
            ctx,
            &self.top_k,
            &[(0, input.buffer()), (1, out_idx.buffer()), (2, &params)],
            (1, 1, 1),
        );
    }

    /// Writes, as a `u32` per row into `out_idx`, the index of the largest
    /// value in each row (last dimension) of `input`, in the order of
    /// `f32::total_cmp` with the last index winning ties, as
    /// `ferrotensor_loader::argmax` picks a greedy token.
    pub fn argmax_rows(&self, ctx: &WgpuContext, input: &UnifiedTensor, out_idx: &UnifiedTensor) {
        let len = input.shape.last().map_or(1, |&len| len.max(1));
        let rows = input.shape.iter().product::<usize>() / len;
        let params = uniform(ctx, bytemuck::cast_slice(&[len as u32]));
        self.run(
            ctx,
            &self.argmax_rows,
            &[(0, input.buffer()), (1, out_idx.buffer()), (2, &params)],
            grid(rows as u32),
        );
    }

    /// Depthwise 1-D convolution with kernel size 3 and padding 1:
    /// `out[d, s] = bias[d] + sum_k weight[d, k] * input[d, s + k - 1]`, with
    /// `input`/`out` `[dim, seq_len]`, `weight` `[dim, 3]` and `bias` `[dim]`.
    pub fn conv1d(
        &self,
        ctx: &WgpuContext,
        input: &UnifiedTensor,
        weight: &UnifiedTensor,
        bias: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        let dim = weight.shape[0];
        let seq_len = input.shape[input.shape.len() - 1];
        let params = uniform(ctx, bytemuck::cast_slice(&[dim as u32, seq_len as u32]));
        self.run(
            ctx,
            &self.conv1d,
            &[
                (0, input.buffer()),
                (1, weight.buffer()),
                (2, out.buffer()),
                (3, bias.buffer()),
                (4, &params),
            ],
            groups64(dim * seq_len),
        );
    }

    /// Gradients of [`conv1d`](Self::conv1d) with respect to `input`,
    /// `weight` and `bias`, given the output gradient `grad_out`. Each
    /// gradient has the shape of the tensor it belongs to. Writes only the
    /// first `dim * seq_len` elements of `grad_input`.
    #[allow(clippy::too_many_arguments)]
    pub fn conv1d_backward(
        &self,
        ctx: &WgpuContext,
        input: &UnifiedTensor,
        weight: &UnifiedTensor,
        grad_out: &UnifiedTensor,
        grad_input: &UnifiedTensor,
        grad_weight: &UnifiedTensor,
        grad_bias: &UnifiedTensor,
    ) {
        let dim = weight.shape[0];
        let seq_len = input.shape[input.shape.len() - 1];
        let params = uniform(ctx, bytemuck::cast_slice(&[dim as u32, seq_len as u32]));
        self.run(
            ctx,
            &self.conv1d_bw,
            &[
                (0, input.buffer()),
                (1, weight.buffer()),
                (2, grad_out.buffer()),
                (3, grad_input.buffer()),
                (4, grad_weight.buffer()),
                (5, grad_bias.buffer()),
                (6, &params),
            ],
            groups64(dim),
        );
    }

    /// Writes into `out` (`[n, hidden]`) the rows of `table` (`[vocab,
    /// hidden]`) named by `ids`, `n` ids stored as `u32`s; an id outside the
    /// table gives a row of zeros.
    pub fn embedding_gather(
        &self,
        ctx: &WgpuContext,
        table: &UnifiedTensor,
        ids: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        let n = out.shape.iter().product::<usize>();
        let (vocab, hidden) = (table.shape[0], table.shape[1]);
        if n == 0 || hidden == 0 {
            return;
        }
        let params = uniform(ctx, bytemuck::cast_slice(&[hidden as u32, vocab as u32, 0, 0]));
        self.run(
            ctx,
            &self.emb_gather,
            &[
                (0, table.buffer()),
                (1, ids.buffer()),
                (2, out.buffer()),
                (3, &params),
            ],
            groups64(n),
        );
    }

    /// Writes into `table_grad` (`[vocab, hidden]`) the gradient of
    /// [`embedding_gather`](Self::embedding_gather): each row is the sum, in
    /// position order, of the rows of `grad` (`[n, hidden]`) whose id it is,
    /// and zero for a row no position looked up. `index` is the `u32` layout
    /// [`embedding_scatter_index`](Self::embedding_scatter_index) builds from
    /// the ids, with `slots` unique ids.
    pub fn embedding_scatter(
        &self,
        ctx: &WgpuContext,
        grad: &UnifiedTensor,
        index: &UnifiedTensor,
        slots: usize,
        table_grad: &UnifiedTensor,
    ) {
        let (vocab, hidden) = (table_grad.shape[0], table_grad.shape[1]);
        if vocab == 0 || hidden == 0 {
            return;
        }
        let params = uniform(
            ctx,
            bytemuck::cast_slice(&[hidden as u32, vocab as u32, slots as u32, 0]),
        );
        self.run(
            ctx,
            &self.emb_scatter,
            &[
                (0, grad.buffer()),
                (1, index.buffer()),
                (2, table_grad.buffer()),
                (3, &params),
            ],
            groups64(vocab * hidden),
        );
    }

    /// The index [`embedding_scatter`](Self::embedding_scatter) reads for the
    /// looked-up `ids` of a `vocab`-row table, and its number of unique
    /// in-table ids: per table row the slot of its id or `u32::MAX`, the
    /// start of each slot's positions (one more entry than slots), then the
    /// positions grouped by slot in ascending order. Ids outside the table
    /// get no slot.
    pub fn embedding_scatter_index(ids: &[u32], vocab: usize) -> (Vec<u32>, usize) {
        let mut slot_of_row = vec![u32::MAX; vocab];
        let mut counts: Vec<u32> = Vec::new();
        for &id in ids {
            if let Some(slot) = slot_of_row.get_mut(id as usize) {
                if *slot == u32::MAX {
                    *slot = counts.len() as u32;
                    counts.push(0);
                }
                counts[*slot as usize] += 1;
            }
        }
        let slots = counts.len();
        let mut index = slot_of_row;
        index.reserve(slots + 1 + ids.len());
        let mut start = 0u32;
        for &count in &counts {
            index.push(start);
            start += count;
        }
        index.push(start);
        let mut next: Vec<u32> = index[vocab..vocab + slots].to_vec();
        index.resize(vocab + slots + 1 + start as usize, 0);
        let positions = vocab + slots + 1;
        // Walking the ids in position order keeps each slot's positions
        // ascending, the order the host scatter-add sums them in.
        for (position, &id) in ids.iter().enumerate() {
            let row = id as usize;
            if row < vocab {
                let at = &mut next[index[row] as usize];
                index[positions + *at as usize] = position as u32;
                *at += 1;
            }
        }
        (index, slots)
    }
}
