//! Dynamic Loss Scaling for Mixed Precision Training
//!
//! Prevents gradient underflow/overflow during FP16 training by dynamically
//! adjusting the loss scale factor.

use rusty_backend::{ComputeEngine, UnifiedTensor, WgpuContext};

/// Dynamic gradient scaler for FP16 mixed precision training.
///
/// Scales the loss to prevent gradient underflow in FP16, then unscales
/// gradients before the optimizer step. Automatically adjusts the scale
/// based on whether gradients contain inf/nan.
///
/// # Example
/// ```rust,ignore
/// let mut scaler = GradScaler::new();
///
/// // Training loop
/// let scaled_loss = scaler.scale(&loss, &engine);
/// scaled_loss.backward(&engine);
///
/// // Check for overflow and update scale
/// if scaler.unscale_and_check(&gradients, &ctx, &engine) {
///     optimizer.step(&params);
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
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GradScalerCheckpoint {
    pub scale: f32,
    pub growth_factor: f32,
    pub backoff_factor: f32,
    pub growth_interval: u32,
    pub consecutive_good_steps: u32,
    pub found_overflow: bool,
    pub enabled: bool,
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
    /// Returns the scaled loss tensor (modifies in place for efficiency).
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
            let data = pollster::block_on(grad.to_vec(ctx));
            if data.iter().any(|&x| x.is_infinite() || x.is_nan()) {
                has_overflow = true;
                break;
            }
        }

        self.found_overflow = has_overflow;
        !has_overflow
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

    pub fn checkpoint(&self) -> GradScalerCheckpoint {
        GradScalerCheckpoint {
            scale: self.scale,
            growth_factor: self.growth_factor,
            backoff_factor: self.backoff_factor,
            growth_interval: self.growth_interval,
            consecutive_good_steps: self.consecutive_good_steps,
            found_overflow: self.found_overflow,
            enabled: self.enabled,
        }
    }

    pub fn load_checkpoint(&mut self, checkpoint: &GradScalerCheckpoint) {
        self.scale = checkpoint.scale;
        self.growth_factor = checkpoint.growth_factor;
        self.backoff_factor = checkpoint.backoff_factor;
        self.growth_interval = checkpoint.growth_interval;
        self.consecutive_good_steps = checkpoint.consecutive_good_steps;
        self.found_overflow = checkpoint.found_overflow;
        self.enabled = checkpoint.enabled;
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
    pub scale: f32,
    pub consecutive_good_steps: u32,
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
    fn test_scaler_disabled() {
        let mut scaler = GradScaler::new();
        scaler.set_enabled(false);

        assert_eq!(scaler.scale_factor(), 1.0);
        assert!(!scaler.is_enabled());
    }
}
