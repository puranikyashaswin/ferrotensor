//! The runtime checks wgpu compiles into Ferrotensor's built-in kernels.

use std::borrow::Cow;
use std::fmt;

/// Environment variable that picks the runtime checks wgpu compiles into the
/// built-in kernels, read once when a [`WgpuContext`](crate::WgpuContext) is
/// created: `default` (all of wgpu's checks), or a comma-separated list of
/// the checks to leave out: `noloop`, `nobounds`, `nodiv`, `nozeroinit`, or
/// `none` for all of them. Unset, it is [`ShaderChecks::DEFAULT`], which is
/// `noloop`.
pub const SHADER_CHECKS_ENV: &str = "FERROTENSOR_SHADER_CHECKS";

/// Which of wgpu's runtime checks the built-in kernels are compiled with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShaderChecks {
    /// naga's counter that ends every loop after 2^64 iterations (`noloop`
    /// leaves it out).
    pub loop_bounding: bool,
    /// Clamping of array and buffer indices into bounds (`nobounds`).
    pub bounds: bool,
    /// Guards on integer `/` and `%` against a zero divisor and signed
    /// overflow (`nodiv`).
    pub int_div: bool,
    /// Zeroing of workgroup memory when a workgroup starts (`nozeroinit`).
    pub zero_workgroup_memory: bool,
}

impl ShaderChecks {
    /// Every check wgpu compiles in by default (`default`).
    pub const CHECKED: Self = Self {
        loop_bounding: true,
        bounds: true,
        int_div: true,
        zero_workgroup_memory: true,
    };

    /// No runtime checks (`none`).
    pub const NONE: Self = Self {
        loop_bounding: false,
        bounds: false,
        int_div: false,
        zero_workgroup_memory: false,
    };

    /// The checks used when [`SHADER_CHECKS_ENV`] is unset or empty: all but
    /// the forced loop bounding, which adds a counter and a branch to every
    /// iteration of the kernels' inner loops, all of them already bounded.
    pub const DEFAULT: Self = Self {
        loop_bounding: false,
        ..Self::CHECKED
    };

    /// Parses a [`SHADER_CHECKS_ENV`] value; an empty value is [`Self::DEFAULT`].
    ///
    /// # Errors
    /// Returns the first word that is not one of the documented values.
    pub fn parse(value: &str) -> Result<Self, String> {
        if value.trim().is_empty() {
            return Ok(Self::DEFAULT);
        }
        let mut checks = Self::CHECKED;
        for word in value.split(',').map(str::trim) {
            match word.to_ascii_lowercase().as_str() {
                "default" => {}
                "noloop" => checks.loop_bounding = false,
                "nobounds" => checks.bounds = false,
                "nodiv" => checks.int_div = false,
                "nozeroinit" => checks.zero_workgroup_memory = false,
                "none" => checks = Self::NONE,
                _ => {
                    return Err(format!(
                        "{SHADER_CHECKS_ENV}={value}: unknown value {word:?}; use default, \
                         or a comma-separated list of noloop, nobounds, nodiv, nozeroinit, none"
                    ))
                }
            }
        }
        Ok(checks)
    }

    /// The checks [`SHADER_CHECKS_ENV`] asks for; an unknown value is
    /// logged and [`Self::DEFAULT`] used.
    pub(crate) fn from_env() -> Self {
        let Ok(value) = std::env::var(SHADER_CHECKS_ENV) else {
            return Self::DEFAULT;
        };
        match Self::parse(&value) {
            Ok(checks) => {
                if checks != Self::DEFAULT {
                    log::warn!("built-in kernels compiled with shader checks: {checks}");
                }
                checks
            }
            Err(e) => {
                log::warn!("{e}; using {}", Self::DEFAULT);
                Self::DEFAULT
            }
        }
    }

    /// The pipeline options for a kernel compiled with these checks.
    pub(crate) fn compilation_options(self) -> wgpu::PipelineCompilationOptions<'static> {
        wgpu::PipelineCompilationOptions {
            zero_initialize_workgroup_memory: self.zero_workgroup_memory,
            ..Default::default()
        }
    }

    /// Compiles `source`, one of the WGSL kernels built into this crate, with
    /// these checks.
    #[expect(
        unsafe_code,
        reason = "wgpu marks building a shader module without some runtime checks unsafe; \
                  see the SAFETY note"
    )]
    pub(crate) fn module(
        self,
        device: &wgpu::Device,
        label: &str,
        source: String,
    ) -> wgpu::ShaderModule {
        let desc = wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(source)),
        };
        if self.loop_bounding && self.bounds && self.int_div {
            return device.create_shader_module(desc);
        }
        let runtime_checks = wgpu::ShaderRuntimeChecks {
            bounds_checks: self.bounds,
            force_loop_bounding: self.loop_bounding,
            int_div_checks: self.int_div,
            ..wgpu::ShaderRuntimeChecks::checked()
        };
        // SAFETY: `source` is always one of this crate's own WGSL kernels,
        // compiled into the binary (callers pass `include_str!` text or text
        // generated from it), never user input. Every loop in them is bounded
        // by a size read from a uniform or an immediate, or by a constant, so
        // none runs forever without the loop counter. Their indices are kept
        // in bounds by those sizes, and their integer divisors are constants
        // or the dimensions and head counts of a tensor shape, which are never
        // zero for a shape that is dispatched. Only the loop bounding is left
        // out unless `FERROTENSOR_SHADER_CHECKS` asks for more.
        unsafe { device.create_shader_module_trusted(desc, runtime_checks) }
    }
}

impl Default for ShaderChecks {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl fmt::Display for ShaderChecks {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if *self == Self::CHECKED {
            return f.write_str("default");
        }
        if *self == Self::NONE {
            return f.write_str("none");
        }
        let off: Vec<&str> = [
            (self.loop_bounding, "noloop"),
            (self.bounds, "nobounds"),
            (self.int_div, "nodiv"),
            (self.zero_workgroup_memory, "nozeroinit"),
        ]
        .into_iter()
        .filter(|&(on, _)| !on)
        .map(|(_, word)| word)
        .collect();
        f.write_str(&off.join(","))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn parses_each_value() {
        assert_eq!(ShaderChecks::parse("").unwrap(), ShaderChecks::DEFAULT);
        assert_eq!(ShaderChecks::parse(" ").unwrap(), ShaderChecks::DEFAULT);
        assert_eq!(
            ShaderChecks::parse("default").unwrap(),
            ShaderChecks::CHECKED
        );
        assert_eq!(
            ShaderChecks::parse("noloop").unwrap(),
            ShaderChecks {
                loop_bounding: false,
                bounds: true,
                int_div: true,
                zero_workgroup_memory: true,
            }
        );
        assert_eq!(
            ShaderChecks::parse("nobounds").unwrap(),
            ShaderChecks {
                loop_bounding: true,
                bounds: false,
                int_div: true,
                zero_workgroup_memory: true,
            }
        );
        assert_eq!(
            ShaderChecks::parse("nodiv").unwrap(),
            ShaderChecks {
                loop_bounding: true,
                bounds: true,
                int_div: false,
                zero_workgroup_memory: true,
            }
        );
        assert_eq!(
            ShaderChecks::parse("nozeroinit").unwrap(),
            ShaderChecks {
                loop_bounding: true,
                bounds: true,
                int_div: true,
                zero_workgroup_memory: false,
            }
        );
        assert_eq!(ShaderChecks::parse("NONE").unwrap(), ShaderChecks::NONE);
    }

    #[test]
    fn the_default_leaves_out_only_loop_bounding() {
        assert_eq!(
            ShaderChecks::DEFAULT,
            ShaderChecks {
                loop_bounding: false,
                bounds: true,
                int_div: true,
                zero_workgroup_memory: true,
            }
        );
        assert_eq!(
            ShaderChecks::parse("noloop").unwrap(),
            ShaderChecks::DEFAULT
        );
        assert_eq!(ShaderChecks::default().to_string(), "noloop");
        assert_ne!(
            ShaderChecks::parse("default").unwrap(),
            ShaderChecks::DEFAULT
        );
    }

    #[test]
    fn combines_a_list() {
        let checks = ShaderChecks::parse("noloop, nozeroinit").unwrap();
        assert_eq!(
            checks,
            ShaderChecks {
                loop_bounding: false,
                bounds: true,
                int_div: true,
                zero_workgroup_memory: false,
            }
        );
        assert_eq!(checks.to_string(), "noloop,nozeroinit");
        assert_eq!(ShaderChecks::CHECKED.to_string(), "default");
        assert_eq!(ShaderChecks::NONE.to_string(), "none");
        assert_eq!(
            ShaderChecks::parse("noloop,nobounds,nodiv,nozeroinit").unwrap(),
            ShaderChecks::NONE
        );
    }

    #[test]
    fn rejects_an_unknown_word() {
        let err = ShaderChecks::parse("noloop,fast").unwrap_err();
        assert!(err.contains("\"fast\""), "{err}");
    }

    #[test]
    fn zero_init_reaches_the_pipeline_options() {
        assert!(
            ShaderChecks::CHECKED
                .compilation_options()
                .zero_initialize_workgroup_memory
        );
        assert!(
            !ShaderChecks::parse("nozeroinit")
                .unwrap()
                .compilation_options()
                .zero_initialize_workgroup_memory
        );
    }
}
