//! Fused training projections for [`Attention`] and [`MLP`].
//!
//! [`Lfm2Model::enable_fused_projections`](crate::Lfm2Model::enable_fused_projections)
//! gives each eligible layer a [`FusedProjections`] group: the q, k and v
//! projections (with RoPE inside flash attention) of an attention layer, and
//! the gate and up projections (with the SwiGLU product) of an MLP. Their
//! forward and backward passes then run as the fused kernels of
//! `ferrotensor-backend` instead of one op per projection.
//!
//! The group owns the concatenated f16 weight of its projections, with room for
//! an f16 copy of the `A` matrices of the layer's LoRA adapters. It is the
//! only copy of those weights: the layer's own `Linear`s keep a placeholder,
//! and a pass the group cannot serve (a KV cache, a mask) copies the weights
//! back out of it once and keeps them.
//!
//! `FERROTENSOR_FUSE_QKV` and `FERROTENSOR_FUSE_GATEUP` (`on`, `off` or
//! `auto`, read once) choose per kind of group whether a layer gets one. [`Lfm2Model::refresh_lora_mirrors`](crate::Lfm2Model::refresh_lora_mirrors)
//! rewrites it, and must run after every optimizer step, adapter load and
//! resume. A layer that is not eligible, or a model that was never enabled,
//! runs the unfused code of before.

use crate::{Attention, Linear, MLP};
use ferrotensor_autograd::{AttentionGeometry, FusedLora, FusedProjections, Tensor};
use ferrotensor_backend::{ComputeEngine, DType, UnifiedTensor, WgpuContext, LORA_GROUP_MAX_RANK};
use std::cell::RefCell;
use std::sync::{Arc, OnceLock};

/// The environment variable that switches the fused q, k and v groups.
pub const FUSE_QKV_ENV: &str = "FERROTENSOR_FUSE_QKV";
/// The environment variable that switches the fused gate and up groups.
pub const FUSE_GATEUP_ENV: &str = "FERROTENSOR_FUSE_GATEUP";

/// Whether one kind of fused group is used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FuseSwitch {
    /// Every eligible layer gets the group.
    On,
    /// No layer does: the unfused code of before runs.
    Off,
    /// A layer gets it where [`FuseKind::auto_fuses`] says it is faster.
    Auto,
}

impl FuseSwitch {
    /// The switch a variable value names; `None` (unset) is `auto`, and a
    /// value that names none is `auto` with a warning.
    fn parse(name: &str, value: Option<&str>) -> Self {
        match value.map(|v| v.trim().to_ascii_lowercase()).as_deref() {
            None | Some("auto") => Self::Auto,
            Some("on") => Self::On,
            Some("off") => Self::Off,
            Some(other) => {
                log::warn!("{name}={other:?} is not on, off or auto; using auto");
                Self::Auto
            }
        }
    }
}

/// The two kinds of fused group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FuseKind {
    /// The q, k and v projections of an attention layer.
    Qkv,
    /// The gate and up projections of a feed-forward.
    GateUp,
}

/// The shape of one group, for the `auto` rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FuseShape {
    /// Rows of one training pass (batch size times sequence length).
    pub rows: usize,
    /// The input width of the projections.
    pub hidden: usize,
    /// The summed output widths of the projections.
    pub width: usize,
}

impl FuseKind {
    /// Whether `auto` fuses a group of this kind and `shape`.
    ///
    /// The q, k and v group was at least as fast as the three projections in
    /// every SmolLM2 size and batch shape measured on one A10G. The gate and
    /// up group was within 3% of them at 135M. Its backward product `[dG|dU]
    /// @ W^T` has a K of the summed widths; before that product ran with a K
    /// split it was 17% slower at 360M and 12 to 37% slower at 1.7B. With
    /// the split it is 3% to 6% faster at 360M (256 rows, batch 1 by 8 and 8
    /// by 1) and still 5% slower at 1.7B. Other GPUs are not measured, so
    /// `on` and `off` stay available.
    pub fn auto_fuses(self, shape: FuseShape) -> bool {
        match self {
            Self::Qkv => true,
            Self::GateUp => shape.width <= GATE_UP_AUTO_MAX_WIDTH,
        }
    }
}

/// The widest gate and up group (summed output widths) that `auto` fuses:
/// SmolLM2-135M's 3,072 and SmolLM2-360M's 5,120 do, SmolLM2-1.7B's 16,384
/// does not. Widths between 5,120 and 16,384 are not measured; the limit sits
/// below the middle of that range.
const GATE_UP_AUTO_MAX_WIDTH: usize = 8192;

/// Which fused groups [`Lfm2Model::enable_fused_projections_with`](crate::Lfm2Model::enable_fused_projections_with) builds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FuseChoice {
    /// The q, k and v groups.
    pub qkv: FuseSwitch,
    /// The gate and up groups.
    pub gate_up: FuseSwitch,
}

impl FuseChoice {
    /// Both kinds on.
    pub const ON: Self = Self {
        qkv: FuseSwitch::On,
        gate_up: FuseSwitch::On,
    };
    /// Both kinds off.
    pub const OFF: Self = Self {
        qkv: FuseSwitch::Off,
        gate_up: FuseSwitch::Off,
    };
    /// Both kinds decided by the `auto` rule.
    pub const AUTO: Self = Self {
        qkv: FuseSwitch::Auto,
        gate_up: FuseSwitch::Auto,
    };

    /// The choice [`FUSE_QKV_ENV`] and [`FUSE_GATEUP_ENV`] make, read from the
    /// environment the first time and kept for the life of the process.
    pub fn from_env() -> Self {
        static CHOICE: OnceLock<FuseChoice> = OnceLock::new();
        *CHOICE.get_or_init(|| Self::from_vars(&|name| std::env::var(name).ok()))
    }

    /// The choice `get`'s environment makes.
    pub fn from_vars(get: &dyn Fn(&str) -> Option<String>) -> Self {
        Self {
            qkv: FuseSwitch::parse(FUSE_QKV_ENV, get(FUSE_QKV_ENV).as_deref()),
            gate_up: FuseSwitch::parse(FUSE_GATEUP_ENV, get(FUSE_GATEUP_ENV).as_deref()),
        }
    }

    /// Whether a group of `kind` and `shape` is built.
    pub fn wants(self, kind: FuseKind, shape: FuseShape) -> bool {
        let switch = match kind {
            FuseKind::Qkv => self.qkv,
            FuseKind::GateUp => self.gate_up,
        };
        match switch {
            FuseSwitch::On => true,
            FuseSwitch::Off => false,
            FuseSwitch::Auto => kind.auto_fuses(shape),
        }
    }
}

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
pub struct FusedSlot {
    group: RefCell<Option<Arc<FusedProjections>>>,
    /// The projections copied back out of the group, made the first time a
    /// pass needs them.
    restored: RefCell<Option<Vec<Tensor>>>,
}

impl FusedSlot {
    /// The group, when the layer has one.
    pub fn get(&self) -> Option<Arc<FusedProjections>> {
        self.group.borrow().clone()
    }

    fn set(&self, group: Option<Arc<FusedProjections>>) {
        *self.group.borrow_mut() = group;
        *self.restored.borrow_mut() = None;
    }

    /// The weights of the group's projections as separate frozen tensors, or
    /// `None` when the layer has no group (its own weights are then the
    /// real ones). They are copied out of the group the first time.
    #[expect(
        clippy::panic,
        reason = "the unfused forward pass has no error path, and the group cannot be rebuilt without these weights"
    )]
    fn restored(&self, ctx: &Arc<WgpuContext>) -> Option<Vec<Tensor>> {
        let group = self.get()?;
        let mut cached = self.restored.borrow_mut();
        if cached.is_none() {
            let weights: Result<Vec<Tensor>, String> = (0..group.widths().len())
                .map(|i| group.projection(ctx, i))
                .collect();
            match weights {
                Ok(w) => *cached = Some(w),
                Err(e) => panic!("copying the projections out of the fused group: {e}"),
            }
        }
        cached.clone()
    }
}

/// `lin` with `weight` in place of its own.
fn with_weight(lin: &Linear, weight: &Tensor) -> Linear {
    Linear {
        weight: weight.clone(),
        bias: lin.bias.clone(),
        lora: lin.lora.clone(),
    }
}

/// What stands in for a weight whose real copy lives in a fused group.
fn placeholder(ctx: &Arc<WgpuContext>) -> Tensor {
    Tensor::frozen(ctx.clone(), UnifiedTensor::new(ctx, &[0.0], &[1, 1]))
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
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "tensors are single-threaded handles (the autograd crate allows the same), the group never crosses a thread"
)]
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
    /// The input width and summed output width of the q, k and v projections.
    fn qkv_shape(&self, rows: usize) -> FuseShape {
        let widths = [&self.q_proj, &self.k_proj, &self.v_proj]
            .map(|l| l.weight.data.shape.get(1).copied().unwrap_or(0));
        FuseShape {
            rows,
            hidden: self.q_proj.weight.data.shape.first().copied().unwrap_or(0),
            width: widths.iter().sum(),
        }
    }

    /// Gives the layer a fused q, k and v group, or none when `mode`,
    /// `choice` or the layer's shape does not allow one (a bias, a per-head
    /// norm, a weight that is not frozen f16, adapters that differ). `rows`
    /// is the row count of one pass, for the `auto` rule. Returns whether
    /// the layer now runs fused.
    ///
    /// The group keeps the only copy of the three weights: the layer's own
    /// are replaced by a placeholder, so a layer that has a group keeps it,
    /// and calling this again returns `true` without changing anything.
    ///
    /// # Errors
    ///
    /// When building the group fails, such as a failed GPU readback.
    pub fn enable_fused(
        &mut self,
        ctx: &Arc<WgpuContext>,
        engine: &ComputeEngine,
        mode: FusedMode,
        choice: FuseChoice,
        rows: usize,
    ) -> Result<bool, String> {
        if self.fused.get().is_some() {
            return Ok(true);
        }
        if !mode.allows(engine)
            || self.q_norm.is_some()
            || self.k_norm.is_some()
            || !choice.wants(FuseKind::Qkv, self.qkv_shape(rows))
        {
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
        if on {
            let gone = placeholder(ctx);
            self.q_proj.weight = gone.clone();
            self.k_proj.weight = gone.clone();
            self.v_proj.weight = gone;
        }
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

    /// The q, k and v layers for a pass without the group: the layer's own,
    /// or, when their weights live in the group, copies with the weights
    /// taken back out of it.
    pub(crate) fn unfused_qkv(&self, ctx: &Arc<WgpuContext>) -> Option<[Linear; 3]> {
        let weights = self.fused.restored(ctx)?;
        let [q, k, v] = weights.as_slice() else {
            return None;
        };
        Some([
            with_weight(&self.q_proj, q),
            with_weight(&self.k_proj, k),
            with_weight(&self.v_proj, v),
        ])
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
            inv_freq: self.rope_inv_freq.clone(),
            pos: pos as u32,
            dropout_prob,
            seed,
        };
        let out = Tensor::fused_attention(x, group, &geom, engine);
        self.o_proj.forward(engine, &out)
    }
}

impl MLP {
    /// The input width and summed output width of the gate and up
    /// projections.
    fn gate_up_shape(&self, rows: usize) -> FuseShape {
        let widths = [&self.gate_proj, &self.up_proj]
            .map(|l| l.weight.data.shape.get(1).copied().unwrap_or(0));
        FuseShape {
            rows,
            hidden: self
                .gate_proj
                .weight
                .data
                .shape
                .first()
                .copied()
                .unwrap_or(0),
            width: widths.iter().sum(),
        }
    }

    /// Gives the layer a fused gate and up group, or none when `mode`,
    /// `choice` or the layer's shape does not allow one. `rows` is the row
    /// count of one pass, for the `auto` rule. Returns whether the layer now
    /// runs fused. As for [`Attention::enable_fused`], the group keeps the
    /// only copy of the two weights.
    ///
    /// # Errors
    ///
    /// When building the group fails, such as a failed GPU readback.
    pub fn enable_fused(
        &mut self,
        ctx: &Arc<WgpuContext>,
        engine: &ComputeEngine,
        mode: FusedMode,
        choice: FuseChoice,
        rows: usize,
    ) -> Result<bool, String> {
        if self.fused.get().is_some() {
            return Ok(true);
        }
        if !mode.allows(engine) || !choice.wants(FuseKind::GateUp, self.gate_up_shape(rows)) {
            return Ok(false);
        }
        let same_width = self.gate_proj.weight.data.shape == self.up_proj.weight.data.shape;
        if !same_width {
            return Ok(false);
        }
        let group = build(ctx, engine, &[&self.gate_proj, &self.up_proj])?;
        let on = group.is_some();
        if on {
            let gone = placeholder(ctx);
            self.gate_proj.weight = gone.clone();
            self.up_proj.weight = gone;
        }
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

    /// The gate and up layers for a pass without the group: the layer's own,
    /// or, when their weights live in the group, copies with the weights
    /// taken back out of it.
    pub(crate) fn unfused_gate_up(&self, ctx: &Arc<WgpuContext>) -> Option<[Linear; 2]> {
        let weights = self.fused.restored(ctx)?;
        let [gate, up] = weights.as_slice() else {
            return None;
        };
        Some([
            with_weight(&self.gate_proj, gate),
            with_weight(&self.up_proj, up),
        ])
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

#[cfg(test)]
mod tests {
    use super::*;

    fn choice(qkv: Option<&str>, gate_up: Option<&str>) -> FuseChoice {
        FuseChoice::from_vars(&|name| match name {
            FUSE_QKV_ENV => qkv.map(String::from),
            FUSE_GATEUP_ENV => gate_up.map(String::from),
            _ => None,
        })
    }

    const SHAPE: FuseShape = FuseShape {
        rows: 8,
        hidden: 2048,
        width: 4096,
    };

    #[test]
    fn unset_variables_are_auto() {
        assert_eq!(choice(None, None), FuseChoice::AUTO);
    }

    #[test]
    fn each_variable_sets_its_own_switch() {
        let c = choice(Some("on"), Some("off"));
        assert_eq!(c.qkv, FuseSwitch::On);
        assert_eq!(c.gate_up, FuseSwitch::Off);
        let c = choice(Some(" OFF "), Some("Auto"));
        assert_eq!(c.qkv, FuseSwitch::Off);
        assert_eq!(c.gate_up, FuseSwitch::Auto);
    }

    #[test]
    fn a_value_that_names_no_switch_is_auto() {
        assert_eq!(choice(Some("yes"), Some("")), FuseChoice::AUTO);
    }

    #[test]
    fn on_and_off_ignore_the_shape() {
        for kind in [FuseKind::Qkv, FuseKind::GateUp] {
            assert!(FuseChoice::ON.wants(kind, SHAPE));
            assert!(!FuseChoice::OFF.wants(kind, SHAPE));
        }
        let only_qkv = choice(Some("on"), Some("off"));
        assert!(only_qkv.wants(FuseKind::Qkv, SHAPE));
        assert!(!only_qkv.wants(FuseKind::GateUp, SHAPE));
    }

    #[test]
    fn auto_fuses_gate_up_only_in_the_sizes_it_won_in() {
        let gate_up = |width| FuseShape {
            rows: 256,
            hidden: 576,
            width,
        };
        // SmolLM2-135M, 360M and 1.7B: twice the feed-forward width.
        assert!(FuseChoice::AUTO.wants(FuseKind::GateUp, gate_up(3072)));
        assert!(FuseChoice::AUTO.wants(FuseKind::GateUp, gate_up(5120)));
        assert!(FuseChoice::AUTO.wants(FuseKind::GateUp, gate_up(8192)));
        assert!(!FuseChoice::AUTO.wants(FuseKind::GateUp, gate_up(8193)));
        assert!(!FuseChoice::AUTO.wants(FuseKind::GateUp, gate_up(16384)));
        for width in [3072, 5120, 16384] {
            assert!(FuseChoice::AUTO.wants(FuseKind::Qkv, gate_up(width)));
        }
    }
}
