//! Fused training projections for [`Attention`] and [`MLP`].
//!
//! [`Lfm2Model::enable_fused_projections`](crate::Lfm2Model::enable_fused_projections)
//! gives each eligible layer a [`FusedProjections`] group: the q, k and v
//! projections (with RoPE inside flash attention) of an attention layer, and
//! the gate and up projections (with the SwiGLU product) of an MLP. Their
//! forward and backward passes then run as the fused kernels of
//! `ferrotensor-backend` instead of one op per projection.
//!
//! The group holds an f16 copy of the `A` matrices of the layer's LoRA
//! adapters. [`Lfm2Model::refresh_lora_mirrors`](crate::Lfm2Model::refresh_lora_mirrors)
//! rewrites it, and must run after every optimizer step, adapter load and
//! resume. A layer that is not eligible, or a model that was never enabled,
//! runs the unfused code of before.

use crate::{Attention, Linear, MLP};
use ferrotensor_autograd::{AttentionGeometry, FusedLora, FusedProjections, Tensor};
use ferrotensor_backend::{ComputeEngine, DType, WgpuContext, LORA_GROUP_MAX_RANK};
use std::cell::RefCell;
use std::sync::Arc;

/// When the fused projections are used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FusedMode {
    /// Only when the engine routes matmuls to the cooperative-matrix kernels
    /// (`FERROTENSOR_EXPERIMENTAL=1` on a device that has them).
    Auto,
    /// Whether or not the cooperative kernels run: the fused ops then use
    /// their reference kernels. For tests.
    Force,
}

impl FusedMode {
    fn allows(self, engine: &ComputeEngine) -> bool {
        self == Self::Force || engine.coop_matmul_active()
    }
}

/// The fused group of one layer, empty until enabled.
#[derive(Clone, Default)]
pub struct FusedSlot(RefCell<Option<Arc<FusedProjections>>>);

impl FusedSlot {
    /// The group, when the layer has one.
    pub fn get(&self) -> Option<Arc<FusedProjections>> {
        self.0.borrow().clone()
    }

    fn set(&self, group: Option<Arc<FusedProjections>>) {
        *self.0.borrow_mut() = group;
    }
}

/// The adapters of `lins` as one group, `Ok(None)` without adapters, and
/// `Err(())` when the layer cannot be fused: the adapters are not on every
/// projection, differ in rank, or have a rank the grouped kernels refuse.
fn adapters(lins: &[&Linear]) -> Result<Option<FusedLora>, ()> {
    if lins.iter().all(|l| l.lora.is_none()) {
        return Ok(None);
    }
    let all: Option<Vec<_>> = lins.iter().map(|l| l.lora.as_ref()).collect();
    let Some(all) = all else {
        return Err(());
    };
    let rank = all[0].rank;
    let ok = rank > 0
        && rank.is_multiple_of(2)
        && rank <= LORA_GROUP_MAX_RANK
        && all.iter().all(|l| l.rank == rank);
    if !ok {
        return Err(());
    }
    Ok(Some(FusedLora {
        a: all.iter().map(|l| l.a.clone()).collect(),
        b: all.iter().map(|l| l.b.clone()).collect(),
        // The scale of `Linear::project_2d`, with its snap to exactly 1.
        scales: all
            .iter()
            .map(|l| {
                let s = l.alpha / l.rank as f32;
                if (s - 1.0).abs() <= f32::EPSILON {
                    1.0
                } else {
                    s
                }
            })
            .collect(),
        rank,
    }))
}

/// Whether `l` is a bias-free projection with a frozen f16 weight.
fn plain_frozen(l: &Linear) -> bool {
    l.bias.is_none() && l.weight.data.dtype == DType::FP16 && !l.weight.requires_grad
}

/// The group of `lins`, `Ok(None)` when they cannot be fused.
fn build(
    ctx: &WgpuContext,
    engine: &ComputeEngine,
    lins: &[&Linear],
) -> Result<Option<Arc<FusedProjections>>, String> {
    if !lins.iter().all(|l| plain_frozen(l)) {
        return Ok(None);
    }
    let Ok(lora) = adapters(lins) else {
        return Ok(None);
    };
    let base: usize = lins.iter().map(|l| l.weight.data.shape[1]).sum();
    let total = base + lora.as_ref().map_or(0, |l| l.rank * lins.len());
    if !total.is_multiple_of(2) {
        return Ok(None);
    }
    let weights: Vec<&Tensor> = lins.iter().map(|l| &l.weight).collect();
    FusedProjections::new(ctx, engine, &weights, lora).map(|g| Some(Arc::new(g)))
}

impl Attention {
    /// Gives the layer a fused q, k and v group, or none when `mode` or the
    /// layer's shape does not allow one (a bias, a per-head norm, a weight
    /// that is not frozen f16, adapters that differ). Returns whether the
    /// layer now runs fused.
    ///
    /// # Errors
    ///
    /// When building the group fails, such as a failed GPU readback.
    pub fn enable_fused(
        &self,
        ctx: &WgpuContext,
        engine: &ComputeEngine,
        mode: FusedMode,
    ) -> Result<bool, String> {
        self.fused.set(None);
        if !mode.allows(engine) || self.q_norm.is_some() || self.k_norm.is_some() {
            return Ok(false);
        }
        let kv = self.num_kv_heads * self.head_dim;
        let q = self.n_heads * self.head_dim;
        let shapes_fit = [(&self.q_proj, q), (&self.k_proj, kv), (&self.v_proj, kv)]
            .iter()
            .all(|(l, n)| l.weight.data.shape.get(1) == Some(n));
        if !shapes_fit {
            return Ok(false);
        }
        let group = build(ctx, engine, &[&self.q_proj, &self.k_proj, &self.v_proj])?;
        let on = group.is_some();
        self.fused.set(group);
        Ok(on)
    }

    /// Rewrites the f16 mirror of the layer's adapters, if it has a group.
    ///
    /// # Errors
    ///
    /// When the kernel refuses the call.
    pub fn refresh_fused(&self, ctx: &WgpuContext, engine: &ComputeEngine) -> Result<(), String> {
        self.fused
            .get()
            .map_or(Ok(()), |g| g.refresh_mirror(engine, ctx))
    }

    /// The group, when it can serve a pass that has no KV cache, mask or
    /// per-head norm: the weights are still frozen and the adapters still
    /// train.
    pub(crate) fn fused_for(&self, input: &Tensor) -> Option<Arc<FusedProjections>> {
        let group = self.fused.get()?;
        let frozen = [&self.q_proj, &self.k_proj, &self.v_proj]
            .iter()
            .all(|l| !l.weight.requires_grad);
        let fits = input.data.shape.len() == 3 && input.data.shape[2] == group.input_size();
        (frozen && fits && group.adapters_train()).then_some(group)
    }

    /// The attention of `x` through the fused group.
    pub(crate) fn forward_fused(
        &self,
        engine: &ComputeEngine,
        group: &Arc<FusedProjections>,
        x: &Tensor,
        pos: usize,
        dropout_seed: Option<u64>,
    ) -> Tensor {
        let (dropout_prob, seed) = match dropout_seed {
            Some(seed) => (self.dropout_prob, seed),
            None => (0.0, 0),
        };
        let geom = AttentionGeometry {
            n_heads: self.n_heads,
            n_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            theta: self.rope_theta,
            pos: pos as u32,
            dropout_prob,
            seed,
        };
        let out = Tensor::fused_attention(x, group, &geom, engine);
        self.o_proj.forward(engine, &out)
    }
}

impl MLP {
    /// Gives the layer a fused gate and up group, or none when `mode` or the
    /// layer's shape does not allow one. Returns whether the layer now runs
    /// fused.
    ///
    /// # Errors
    ///
    /// When building the group fails, such as a failed GPU readback.
    pub fn enable_fused(
        &self,
        ctx: &WgpuContext,
        engine: &ComputeEngine,
        mode: FusedMode,
    ) -> Result<bool, String> {
        self.fused.set(None);
        if !mode.allows(engine) {
            return Ok(false);
        }
        let same_width = self.gate_proj.weight.data.shape == self.up_proj.weight.data.shape;
        if !same_width {
            return Ok(false);
        }
        let group = build(ctx, engine, &[&self.gate_proj, &self.up_proj])?;
        let on = group.is_some();
        self.fused.set(group);
        Ok(on)
    }

    /// Rewrites the f16 mirror of the layer's adapters, if it has a group.
    ///
    /// # Errors
    ///
    /// When the kernel refuses the call.
    pub fn refresh_fused(&self, ctx: &WgpuContext, engine: &ComputeEngine) -> Result<(), String> {
        self.fused
            .get()
            .map_or(Ok(()), |g| g.refresh_mirror(engine, ctx))
    }

    /// The group, when the weights are still frozen, the adapters still
    /// train and `input` has the group's input size.
    pub(crate) fn fused_for(&self, input: &Tensor) -> Option<Arc<FusedProjections>> {
        let group = self.fused.get()?;
        let frozen = !self.gate_proj.weight.requires_grad && !self.up_proj.weight.requires_grad;
        let shape = &input.data.shape;
        let fits = (2..=3).contains(&shape.len()) && shape.last() == Some(&group.input_size());
        (frozen && fits && group.adapters_train()).then_some(group)
    }
}
