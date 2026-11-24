//! Dynamic loss scaling.
//!
//! Scales the loss before the backward pass and unscales the gradients before
//! the optimizer step, adjusting the scale when gradients overflow. The CLI
//! turns it on when matmuls run on cooperative matrices, which round their
//! f32 inputs (activations and gradients) to f16.

use ferrotensor_backend::{ComputeEngine, UnifiedTensor, WgpuContext};

/// Dynamic gradient scaler.
///
/// Scales the loss, then unscales gradients before the optimizer step, and
/// adjusts the scale based on whether gradients contain inf/nan. This keeps
/// gradients that a matmul rounds to f16 from underflowing.
///
/// # Example
/// ```no_run
/// # use ferrotensor_backend::{ComputeEngine, UnifiedTensor, WgpuContext};
/// # use ferrotensor_trainer::GradScaler;
/// # let ctx = pollster::block_on(WgpuContext::new());
/// # let engine = ComputeEngine::new(&ctx);
/// # let loss = UnifiedTensor::zeros(&ctx, &[1]);
/// # let gradients: Vec<UnifiedTensor> = Vec::new();
/// let mut scaler = GradScaler::new();
///
/// // Scale the loss in place, then run the backward pass on it.
/// scaler.scale_loss(&loss, &ctx, &engine);
/// // ... backward pass fills `gradients` ...
///
/// // Unscale and skip the optimizer step if any gradient overflowed.
/// if scaler.unscale_and_check(&gradients, &ctx, &engine) {
///     // optimizer.step(...)
/// }
/// scaler.update();
/// ```
pub struct GradScaler {
    /// Current loss scale factor (starts high, e.g., 65536)
    scale: f32,
    /// Factor to grow scale when no overflow occurs
    growth_factor: f32,
    /// Factor to shrink scale when overflow occurs  
    backoff_factor: f32,
    /// Number of consecutive successful steps before growing
    growth_interval: u32,
    /// Counter of consecutive successful steps
    consecutive_good_steps: u32,
    /// Whether overflow was detected in the current step
    found_overflow: bool,
    /// Whether the scaler is enabled
    enabled: bool,
    /// Optimizer steps skipped because the gradients overflowed.
    skipped_steps: u64,
}

/// Serialized [`GradScaler`] state.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GradScalerCheckpoint {
    /// Current loss scale.
    pub scale: f32,
    /// Factor applied after `growth_interval` good steps.
    pub growth_factor: f32,
    /// Factor applied after an overflow.
    pub backoff_factor: f32,
    /// Good steps between scale increases.
    pub growth_interval: u32,
    /// Good steps since the last change.
    pub consecutive_good_steps: u32,
    /// Whether the last step overflowed.
    pub found_overflow: bool,
    /// Whether scaling is on.
    pub enabled: bool,
    /// Optimizer steps skipped so far; 0 in checkpoints written before it
    /// was recorded.
    #[serde(default)]
    pub skipped_steps: u64,
}

impl GradScaler {
    /// Create a new GradScaler with default settings.
    ///
    /// Default: scale=65536, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000
    pub fn new() -> Self {
        Self {
            scale: 65536.0, // 2^16, common starting point
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            consecutive_good_steps: 0,
            found_overflow: false,
            enabled: true,
            skipped_steps: 0,
        }
    }

    /// Create a GradScaler with custom parameters.
    pub fn with_params(
        init_scale: f32,
        growth_factor: f32,
        backoff_factor: f32,
        growth_interval: u32,
    ) -> Self {
        Self {
            scale: init_scale,
            growth_factor,
            backoff_factor,
            growth_interval,
            consecutive_good_steps: 0,
            found_overflow: false,
            enabled: true,
            skipped_steps: 0,
        }
    }

    /// Get current scale factor.
    pub fn scale_factor(&self) -> f32 {
        if self.enabled {
            self.scale
        } else {
            1.0
        }
    }

    /// Scale the loss value before backward pass.
    ///
    /// Scales `loss` in place (no new tensor is allocated).
    pub fn scale_loss(&self, loss: &UnifiedTensor, ctx: &WgpuContext, engine: &ComputeEngine) {
        if !self.enabled {
            return;
        }

        // Scale loss in-place using gradient_scale kernel
        engine.gradient_scale(ctx, loss, self.scale);
    }

    /// Unscale gradients and check for inf/nan.
    ///
    /// Returns true if gradients are valid (no overflow), false otherwise.
    /// If false, you should skip the optimizer step.
    #[expect(
        clippy::expect_used,
        reason = "a readback fails only on device loss, and the caller has no error path"
    )]
    pub fn unscale_and_check(
        &mut self,
        gradients: &[UnifiedTensor],
        ctx: &WgpuContext,
        engine: &ComputeEngine,
    ) -> bool {
        if !self.enabled {
            return true;
        }

        let inv_scale = 1.0 / self.scale;
        let mut has_overflow = false;

        for grad in gradients {
            // Unscale the gradient using gradient_scale kernel
            engine.gradient_scale(ctx, grad, inv_scale);

            // Check for inf/nan (CPU readback - sync point)
            let data =
                pollster::block_on(grad.to_vec(ctx)).expect("GPU device lost during readback");
            if data.iter().any(|&x| x.is_infinite() || x.is_nan()) {
                has_overflow = true;
                break;
            }
        }

        self.found_overflow = has_overflow;
        !has_overflow
    }

    /// The factor that unscales gradients whose global L2 norm at the current
    /// scale is `scaled_norm`, or `None` when that norm is not finite.
    ///
    /// A non-finite norm means a gradient overflowed: the caller skips the
    /// optimizer step, and the scale backs off (halves by default). Either
    /// way this calls [`update`](Self::update). With the scaler disabled the
    /// factor is 1 and nothing changes.
    pub fn unscale_factor(&mut self, scaled_norm: f32) -> Option<f32> {
        if !self.enabled {
            return scaled_norm.is_finite().then_some(1.0);
        }
        let overflowed = !scaled_norm.is_finite();
        let factor = 1.0 / self.scale;
        if overflowed {
            self.skipped_steps += 1;
        }
        self.found_overflow = overflowed;
        self.update();
        (!overflowed).then_some(factor)
    }

    /// Optimizer steps skipped because the gradients overflowed.
    pub fn skipped_steps(&self) -> u64 {
        self.skipped_steps
    }

    /// Update the scale factor based on whether overflow occurred.
    ///
    /// Call this at the end of each training step.
    pub fn update(&mut self) {
        if !self.enabled {
            return;
        }

        if self.found_overflow {
            // Reduce scale on overflow
            self.scale *= self.backoff_factor;
            self.consecutive_good_steps = 0;

            // Clamp to minimum scale
            if self.scale < 1.0 {
                self.scale = 1.0;
            }
        } else {
            // Increment good step counter
            self.consecutive_good_steps += 1;

            // Grow scale after enough good steps
            if self.consecutive_good_steps >= self.growth_interval {
                self.scale *= self.growth_factor;
                self.consecutive_good_steps = 0;

                // Clamp to maximum scale
                if self.scale > 65536.0 * 256.0 {
                    self.scale = 65536.0 * 256.0;
                }
            }
        }

        // Reset overflow flag for next step
        self.found_overflow = false;
    }

    /// Check if overflow was detected in the last step.
    pub fn had_overflow(&self) -> bool {
        self.found_overflow
    }

    /// Enable or disable the scaler.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if scaler is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get statistics for logging.
    pub fn stats(&self) -> GradScalerStats {
        GradScalerStats {
            scale: self.scale,
            consecutive_good_steps: self.consecutive_good_steps,
            enabled: self.enabled,
        }
    }

    /// Snapshot the scaler for a checkpoint.
    pub fn checkpoint(&self) -> GradScalerCheckpoint {
        GradScalerCheckpoint {
            scale: self.scale,
            growth_factor: self.growth_factor,
            backoff_factor: self.backoff_factor,
            growth_interval: self.growth_interval,
            consecutive_good_steps: self.consecutive_good_steps,
            found_overflow: self.found_overflow,
            enabled: self.enabled,
            skipped_steps: self.skipped_steps,
        }
    }

    /// Restore a snapshot taken by [`checkpoint`](Self::checkpoint).
    pub fn load_checkpoint(&mut self, checkpoint: &GradScalerCheckpoint) {
        self.scale = checkpoint.scale;
        self.growth_factor = checkpoint.growth_factor;
        self.backoff_factor = checkpoint.backoff_factor;
        self.growth_interval = checkpoint.growth_interval;
        self.consecutive_good_steps = checkpoint.consecutive_good_steps;
        self.found_overflow = checkpoint.found_overflow;
        self.enabled = checkpoint.enabled;
        self.skipped_steps = checkpoint.skipped_steps;
    }
}

impl Default for GradScaler {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics from the GradScaler for logging/monitoring.
#[derive(Debug, Clone)]
pub struct GradScalerStats {
    /// Current loss scale.
    pub scale: f32,
    /// Good steps since the last change.
    pub consecutive_good_steps: u32,
    /// Whether scaling is on.
    pub enabled: bool,
}

impl std::fmt::Display for GradScalerStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GradScaler(scale={:.0}, good_steps={}, enabled={})",
            self.scale, self.consecutive_good_steps, self.enabled
        )
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_scaler_creation() {
        let scaler = GradScaler::new();
        assert_eq!(scaler.scale_factor(), 65536.0);
        assert!(scaler.is_enabled());
    }

    #[test]
    fn test_scaler_update_no_overflow() {
        let mut scaler = GradScaler::with_params(1024.0, 2.0, 0.5, 5);

        // Simulate 5 good steps
        for _ in 0..5 {
            scaler.found_overflow = false;
            scaler.update();
        }

        // Scale should have doubled
        assert_eq!(scaler.scale, 2048.0);
    }

    #[test]
    fn test_scaler_update_with_overflow() {
        let mut scaler = GradScaler::with_params(1024.0, 2.0, 0.5, 5);

        // Simulate overflow
        scaler.found_overflow = true;
        scaler.update();

        // Scale should have halved
        assert_eq!(scaler.scale, 512.0);
        assert_eq!(scaler.consecutive_good_steps, 0);
    }

    #[test]
    fn overflow_skips_the_step_halves_the_scale_and_counts_it() {
        let mut scaler = GradScaler::new();
        assert_eq!(scaler.unscale_factor(3.0 * 65536.0), Some(1.0 / 65536.0));
        for bad in [f32::INFINITY, f32::NAN, f32::NEG_INFINITY] {
            assert_eq!(scaler.unscale_factor(bad), None, "{bad}");
        }
        assert_eq!(scaler.scale_factor(), 65536.0 / 8.0);
        assert_eq!(scaler.skipped_steps(), 3);
        assert_eq!(scaler.stats().consecutive_good_steps, 0);
        // The next good step unscales by the halved scale.
        assert_eq!(scaler.unscale_factor(1.0), Some(8.0 / 65536.0));
        assert_eq!(scaler.stats().consecutive_good_steps, 1);
    }

    #[test]
    fn scale_doubles_after_the_growth_interval_of_good_steps() {
        // PyTorch GradScaler defaults: 2^16, x2 every 2000 good steps, x0.5.
        let mut scaler = GradScaler::new();
        for _ in 0..1999 {
            assert!(scaler.unscale_factor(1.0).is_some());
        }
        assert_eq!(scaler.scale_factor(), 65536.0);
        assert_eq!(scaler.unscale_factor(1.0), Some(1.0 / 65536.0));
        assert_eq!(scaler.scale_factor(), 131072.0);
        // An overflow restarts the count.
        assert_eq!(scaler.unscale_factor(f32::INFINITY), None);
        assert_eq!(scaler.scale_factor(), 65536.0);
        assert_eq!(scaler.skipped_steps(), 1);
    }

    #[test]
    fn disabled_scaler_unscales_by_one_and_still_refuses_overflow() {
        let mut scaler = GradScaler::new();
        scaler.set_enabled(false);
        assert_eq!(scaler.unscale_factor(5.0), Some(1.0));
        assert_eq!(scaler.unscale_factor(f32::NAN), None);
        assert_eq!(scaler.skipped_steps(), 0);
        assert_eq!(scaler.stats().scale, 65536.0);
    }

    #[test]
    fn checkpoint_keeps_the_skipped_steps_and_reads_old_files() {
        let mut scaler = GradScaler::new();
        assert_eq!(scaler.unscale_factor(f32::NAN), None);
        let mut restored = GradScaler::new();
        restored.load_checkpoint(&scaler.checkpoint());
        assert_eq!(restored.skipped_steps(), 1);
        assert_eq!(restored.scale_factor(), 32768.0);
        let old: GradScalerCheckpoint = serde_json::from_str(
            r#"{"scale":1024.0,"growth_factor":2.0,"backoff_factor":0.5,"growth_interval":2000,"consecutive_good_steps":3,"found_overflow":false,"enabled":true}"#,
        )
        .unwrap();
        assert_eq!(old.skipped_steps, 0);
    }

    #[test]
    fn test_scaler_disabled() {
        let mut scaler = GradScaler::new();
        scaler.set_enabled(false);

        assert_eq!(scaler.scale_factor(), 1.0);
        assert!(!scaler.is_enabled());
    }
}
