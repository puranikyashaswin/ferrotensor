//! Error handling for rusty-backend
//!
//! Provides comprehensive error types for GPU operations, memory management,
//! and compute pipeline failures.

use std::fmt;

/// Result type alias for backend operations
pub type Result<T> = std::result::Result<T, BackendError>;

/// Main error type for backend operations
#[derive(Debug, Clone, PartialEq)]
pub enum BackendError {
    /// GPU initialization failed
    GpuInitialization(String),
    /// Device creation failed
    DeviceCreation(String),
    /// Memory allocation failed
    MemoryAllocation(String),
    /// Invalid tensor dimensions
    InvalidDimensions {
        /// Expected dimensions
        expected: Vec<usize>,
        /// Actual dimensions
        got: Vec<usize>,
        /// Context of the operation
        context: String,
    },
    /// Invalid data type
    InvalidDataType {
        /// Expected type
        expected: String,
        /// Actual type
        got: String,
    },
    /// Compute pipeline error
    ComputePipeline(String),
    /// Shader compilation failed
    ShaderCompilation(String),
    /// Buffer operation failed
    BufferOperation(String),
    /// Kernel execution failed
    KernelExecution(String),
    /// Unsupported operation
    UnsupportedOperation(String),
    /// Out of memory
    OutOfMemory {
        /// Requested bytes
        requested: u64,
        /// Available bytes
        available: u64,
    },
    /// Data transfer failed
    DataTransfer(String),
    /// Invalid operation
    InvalidOperation(String),
    /// Timeout
    Timeout(String),
    /// Other error
    Other(String),
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GpuInitialization(msg) => {
                write!(f, "GPU initialization failed: {}", msg)
            }
            Self::DeviceCreation(msg) => {
                write!(f, "Device creation failed: {}", msg)
            }
            Self::MemoryAllocation(msg) => {
                write!(f, "Memory allocation failed: {}", msg)
            }
            Self::InvalidDimensions {
                expected,
                got,
                context,
            } => {
                write!(
                    f,
                    "Invalid dimensions in {}: expected {:?}, got {:?}",
                    context, expected, got
                )
            }
            Self::InvalidDataType { expected, got } => {
                write!(f, "Invalid data type: expected {}, got {}", expected, got)
            }
            Self::ComputePipeline(msg) => {
                write!(f, "Compute pipeline error: {}", msg)
            }
            Self::ShaderCompilation(msg) => {
                write!(f, "Shader compilation failed: {}", msg)
            }
            Self::BufferOperation(msg) => {
                write!(f, "Buffer operation failed: {}", msg)
            }
            Self::KernelExecution(msg) => {
                write!(f, "Kernel execution failed: {}", msg)
            }
            Self::UnsupportedOperation(msg) => {
                write!(f, "Unsupported operation: {}", msg)
            }
            Self::OutOfMemory {
                requested,
                available,
            } => {
                write!(
                    f,
                    "Out of memory: requested {} bytes, available {} bytes",
                    requested, available
                )
            }
            Self::DataTransfer(msg) => {
                write!(f, "Data transfer failed: {}", msg)
            }
            Self::InvalidOperation(msg) => {
                write!(f, "Invalid operation: {}", msg)
            }
            Self::Timeout(msg) => {
                write!(f, "Operation timed out: {}", msg)
            }
            Self::Other(msg) => {
                write!(f, "{}", msg)
            }
        }
    }
}

impl std::error::Error for BackendError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

impl BackendError {
    /// Create a GPU initialization error
    pub fn gpu_init<S: Into<String>>(msg: S) -> Self {
        Self::GpuInitialization(msg.into())
    }

    /// Create an invalid dimensions error
    pub fn invalid_dims<S: Into<String>>(
        expected: Vec<usize>,
        got: Vec<usize>,
        context: S,
    ) -> Self {
        Self::InvalidDimensions {
            expected,
            got,
            context: context.into(),
        }
    }

    /// Create an out of memory error
    pub fn oom(requested: u64, available: u64) -> Self {
        Self::OutOfMemory {
            requested,
            available,
        }
    }

    /// Check if this is a memory-related error
    pub fn is_memory_error(&self) -> bool {
        matches!(self, Self::MemoryAllocation(_) | Self::OutOfMemory { .. })
    }

    /// Check if this is a recoverable error
    pub fn is_recoverable(&self) -> bool {
        !matches!(
            self,
            Self::GpuInitialization(_) | Self::DeviceCreation(_) | Self::ShaderCompilation(_)
        )
    }
}

/// Error context for better error messages
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Operation being performed
    pub operation: String,
    /// Additional context
    pub context: Vec<(String, String)>,
}

impl ErrorContext {
    /// Create a new error context
    pub fn new<S: Into<String>>(operation: S) -> Self {
        Self {
            operation: operation.into(),
            context: Vec::new(),
        }
    }

    /// Add context information
    pub fn with<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.context.push((key.into(), value.into()));
        self
    }

    /// Format as string
    pub fn format(&self) -> String {
        let mut result = format!("Operation: {}", self.operation);
        for (key, value) in &self.context {
            result.push_str(&format!("\n  {}: {}", key, value));
        }
        result
    }
}

/// Extension trait for Results to add context
pub trait ResultExt<T, E> {
    /// Add context to an error
    fn with_context<F, S>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> S,
        S: Into<String>;
}

impl<T, E: Into<BackendError>> ResultExt<T, E> for std::result::Result<T, E> {
    fn with_context<F, S>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> S,
        S: Into<String>,
    {
        self.map_err(|e| {
            let err = e.into();
            BackendError::Other(format!("{} - {}", f().into(), err))
        })
    }
}

/// Ensure dimensions match for binary operations
pub fn ensure_same_dimensions(a_shape: &[usize], b_shape: &[usize], operation: &str) -> Result<()> {
    if a_shape != b_shape {
        return Err(BackendError::invalid_dims(
            a_shape.to_vec(),
            b_shape.to_vec(),
            operation,
        ));
    }
    Ok(())
}

/// Ensure dimensions are compatible for matrix multiplication
pub fn ensure_matmul_compatible(
    a_shape: &[usize],
    b_shape: &[usize],
) -> Result<(usize, usize, usize)> {
    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(BackendError::invalid_dims(
            vec![a_shape.len()],
            vec![2],
            "matmul requires 2D tensors",
        ));
    }

    let m = a_shape[0];
    let k_a = a_shape[1];
    let k_b = b_shape[0];
    let n = b_shape[1];

    if k_a != k_b {
        return Err(BackendError::invalid_dims(
            vec![m, k_a],
            vec![k_b, n],
            format!("matmul dimension mismatch: {} != {}", k_a, k_b),
        ));
    }

    Ok((m, k_a, n))
}

/// Calculate total size from shape
pub fn calculate_size(shape: &[usize]) -> usize {
    shape.iter().product()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = BackendError::gpu_init("test");
        assert_eq!(format!("{}", err), "GPU initialization failed: test");
    }

    #[test]
    fn test_invalid_dims() {
        let err = BackendError::invalid_dims(vec![2, 3], vec![4, 5], "test_op");
        assert!(err.to_string().contains("Invalid dimensions"));
    }

    #[test]
    fn test_ensure_same_dimensions() {
        assert!(ensure_same_dimensions(&[2, 3], &[2, 3], "add").is_ok());
        assert!(ensure_same_dimensions(&[2, 3], &[4, 5], "add").is_err());
    }

    #[test]
    fn test_ensure_matmul_compatible() {
        assert_eq!(
            ensure_matmul_compatible(&[2, 3], &[3, 4]).unwrap(),
            (2, 3, 4)
        );
        assert!(ensure_matmul_compatible(&[2, 3], &[4, 5]).is_err());
        assert!(ensure_matmul_compatible(&[1, 2, 3], &[3, 4]).is_err());
    }
}
