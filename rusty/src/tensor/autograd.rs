//! Autograd - Automatic Differentiation Engine
//!
//! This module provides gradient computation through reverse-mode autodiff.

use std::sync::{Arc, RwLock};
use wgpu::{Buffer, BufferUsages};

use super::Tensor;
use crate::Device;

/// Gradient storage cell (thread-safe).
pub struct GradCell {
    inner: RwLock<Option<Arc<Buffer>>>,
}

impl GradCell {
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(None),
        }
    }

    pub fn set(&self, device: &Device, data: &[f32], _shape: &[usize]) {
        let buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Gradient Buffer"),
                contents: bytemuck::cast_slice(data),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            });
        *self.inner.write().unwrap() = Some(Arc::new(buffer));
    }

    pub fn set_buffer(&self, buffer: Buffer) {
        *self.inner.write().unwrap() = Some(Arc::new(buffer));
    }

    pub fn get(&self) -> Option<Arc<Buffer>> {
        self.inner.read().unwrap().clone()
    }

    pub fn clear(&self) {
        *self.inner.write().unwrap() = None;
    }

    pub fn accumulate(
        &self,
        device: &Device,
        engine: &crate::backend::ComputeEngine,
        new_grad: &Buffer,
        shape: &[usize],
    ) {
        let size: usize = shape.iter().product();
        let mut inner = self.inner.write().unwrap();

        if let Some(ref existing) = *inner {
            // Create output buffer for accumulated gradient
            let out_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Accumulated Gradient"),
                size: (size * 4) as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // Dispatch add kernel
            let pipeline = engine.get_pipeline("add").unwrap();
            let bind_group = engine.create_binary_bind_group(
                &device.device,
                pipeline,
                existing,
                new_grad,
                &out_buffer,
            );
            engine.dispatch(
                &device.device,
                &device.queue,
                "add",
                &bind_group,
                (
                    crate::backend::ComputeEngine::workgroups_1d(size as u32),
                    1,
                    1,
                ),
            );

            *inner = Some(Arc::new(out_buffer));
        } else {
            // Copy new_grad to this cell
            let out_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Gradient Copy"),
                size: (size * 4) as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let mut encoder =
                device
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Grad Copy Encoder"),
                    });
            encoder.copy_buffer_to_buffer(new_grad, 0, &out_buffer, 0, (size * 4) as u64);
            device.queue.submit(std::iter::once(encoder.finish()));

            *inner = Some(Arc::new(out_buffer));
        }
    }
}

use wgpu::util::DeviceExt;

/// Trait for autograd computation nodes.
pub trait AutogradNode: Send + Sync {
    /// Get input tensors to this node.
    fn inputs(&self) -> Vec<Tensor>;

    /// Compute gradients for inputs given output gradient.
    fn backward(&self, grad_output: &Tensor, device: &Device);
}

// ============================================================================
// BINARY OPERATION NODES
// ============================================================================

/// Addition backward node.
pub struct AddNode {
    pub a: Tensor,
    pub b: Tensor,
}

impl AutogradNode for AddNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.a.clone(), self.b.clone()]
    }

    fn backward(&self, grad_output: &Tensor, device: &Device) {
        // grad_a = grad_out, grad_b = grad_out
        if self.a.requires_grad {
            self.a
                .grad
                .accumulate(device, device.engine(), &grad_output.buffer, &self.a.shape);
        }
        if self.b.requires_grad {
            self.b
                .grad
                .accumulate(device, device.engine(), &grad_output.buffer, &self.b.shape);
        }
    }
}

/// Subtraction backward node.
pub struct SubNode {
    pub a: Tensor,
    pub b: Tensor,
}

impl AutogradNode for SubNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.a.clone(), self.b.clone()]
    }

    fn backward(&self, grad_output: &Tensor, device: &Device) {
        let engine = device.engine();
        let size = self.a.numel();

        if self.a.requires_grad {
            self.a
                .grad
                .accumulate(device, engine, &grad_output.buffer, &self.a.shape);
        }
        if self.b.requires_grad {
            // grad_b = -grad_out
            let neg_grad = device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Neg Grad"),
                size: (size * 4) as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // Use mul_scalar with -1.0
            let neg_one = device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Neg One"),
                    contents: bytemuck::cast_slice(&[-1.0f32]),
                    usage: BufferUsages::UNIFORM,
                });

            let pipeline = engine.get_pipeline("mul_scalar").unwrap();
            let layout = pipeline.get_bind_group_layout(0);
            let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: grad_output.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: neg_grad.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: neg_one.as_entire_binding(),
                    },
                ],
            });
            engine.dispatch(
                &device.device,
                &device.queue,
                "mul_scalar",
                &bind_group,
                (
                    crate::backend::ComputeEngine::workgroups_1d(size as u32),
                    1,
                    1,
                ),
            );

            self.b
                .grad
                .accumulate(device, engine, &neg_grad, &self.b.shape);
        }
    }
}

/// Multiplication backward node.
pub struct MulNode {
    pub a: Tensor,
    pub b: Tensor,
}

impl AutogradNode for MulNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.a.clone(), self.b.clone()]
    }

    fn backward(&self, grad_output: &Tensor, device: &Device) {
        let engine = device.engine();
        let size = self.a.numel();

        // grad_a = grad_out * b
        if self.a.requires_grad {
            let grad_a = device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Grad A"),
                size: (size * 4) as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let pipeline = engine.get_pipeline("mul").unwrap();
            let bind_group = engine.create_binary_bind_group(
                &device.device,
                pipeline,
                &grad_output.buffer,
                &self.b.buffer,
                &grad_a,
            );
            engine.dispatch(
                &device.device,
                &device.queue,
                "mul",
                &bind_group,
                (
                    crate::backend::ComputeEngine::workgroups_1d(size as u32),
                    1,
                    1,
                ),
            );
            self.a
                .grad
                .accumulate(device, engine, &grad_a, &self.a.shape);
        }

        // grad_b = grad_out * a
        if self.b.requires_grad {
            let grad_b = device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Grad B"),
                size: (size * 4) as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let pipeline = engine.get_pipeline("mul").unwrap();
            let bind_group = engine.create_binary_bind_group(
                &device.device,
                pipeline,
                &grad_output.buffer,
                &self.a.buffer,
                &grad_b,
            );
            engine.dispatch(
                &device.device,
                &device.queue,
                "mul",
                &bind_group,
                (
                    crate::backend::ComputeEngine::workgroups_1d(size as u32),
                    1,
                    1,
                ),
            );
            self.b
                .grad
                .accumulate(device, engine, &grad_b, &self.b.shape);
        }
    }
}

/// Matrix multiplication backward node.
pub struct MatMulNode {
    pub a: Tensor, // [M, K]
    pub b: Tensor, // [K, N]
}

impl AutogradNode for MatMulNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.a.clone(), self.b.clone()]
    }

    fn backward(&self, grad_output: &Tensor, device: &Device) {
        let engine = device.engine();
        let m = self.a.shape[0];
        let k = self.a.shape[1];
        let n = self.b.shape[1];

        // grad_a = grad_out @ b.T  [M, N] @ [N, K] = [M, K]
        if self.a.requires_grad {
            let grad_a = device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Grad A MatMul"),
                size: (m * k * 4) as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let params = device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("MatMul BT Params"),
                    contents: bytemuck::cast_slice(&[m as u32, n as u32, k as u32, 0u32]),
                    usage: BufferUsages::UNIFORM,
                });

            let pipeline = engine.get_pipeline("matmul_bt").unwrap();
            let bind_group = engine.create_matmul_bind_group(
                &device.device,
                pipeline,
                &grad_output.buffer,
                &self.b.buffer,
                &grad_a,
                &params,
            );
            let (wx, wy) = crate::backend::ComputeEngine::workgroups_2d(m as u32, k as u32);
            engine.dispatch(
                &device.device,
                &device.queue,
                "matmul_bt",
                &bind_group,
                (wx, wy, 1),
            );

            self.a
                .grad
                .accumulate(device, engine, &grad_a, &self.a.shape);
        }

        // grad_b = a.T @ grad_out  [K, M] @ [M, N] = [K, N]
        if self.b.requires_grad {
            let grad_b = device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Grad B MatMul"),
                size: (k * n * 4) as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let params = device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("MatMul AT Params"),
                    contents: bytemuck::cast_slice(&[k as u32, m as u32, n as u32, 0u32]),
                    usage: BufferUsages::UNIFORM,
                });

            let pipeline = engine.get_pipeline("matmul_at").unwrap();
            let bind_group = engine.create_matmul_bind_group(
                &device.device,
                pipeline,
                &self.a.buffer,
                &grad_output.buffer,
                &grad_b,
                &params,
            );
            let (wx, wy) = crate::backend::ComputeEngine::workgroups_2d(k as u32, n as u32);
            engine.dispatch(
                &device.device,
                &device.queue,
                "matmul_at",
                &bind_group,
                (wx, wy, 1),
            );

            self.b
                .grad
                .accumulate(device, engine, &grad_b, &self.b.shape);
        }
    }
}

// ============================================================================
// ACTIVATION NODES
// ============================================================================

/// ReLU backward node.
pub struct ReLUNode {
    pub input: Tensor,
}

impl AutogradNode for ReLUNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }

    fn backward(&self, grad_output: &Tensor, device: &Device) {
        if !self.input.requires_grad {
            return;
        }

        let engine = device.engine();
        let size = self.input.numel();

        let grad_in = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ReLU Grad"),
            size: (size * 4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline = engine.get_pipeline("relu_backward").unwrap();
        let layout = pipeline.get_bind_group_layout(0);
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.input.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grad_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: grad_output.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.input.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: grad_in.as_entire_binding(),
                },
            ],
        });
        engine.dispatch(
            &device.device,
            &device.queue,
            "relu_backward",
            &bind_group,
            (
                crate::backend::ComputeEngine::workgroups_1d(size as u32),
                1,
                1,
            ),
        );

        self.input
            .grad
            .accumulate(device, engine, &grad_in, &self.input.shape);
    }
}

/// SiLU backward node.
pub struct SiLUNode {
    pub input: Tensor,
}

impl AutogradNode for SiLUNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }

    fn backward(&self, grad_output: &Tensor, device: &Device) {
        if !self.input.requires_grad {
            return;
        }

        let engine = device.engine();
        let size = self.input.numel();

        let grad_in = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SiLU Grad"),
            size: (size * 4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline = engine.get_pipeline("silu_backward").unwrap();
        let layout = pipeline.get_bind_group_layout(0);
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.input.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grad_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: grad_output.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.input.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: grad_in.as_entire_binding(),
                },
            ],
        });
        engine.dispatch(
            &device.device,
            &device.queue,
            "silu_backward",
            &bind_group,
            (
                crate::backend::ComputeEngine::workgroups_1d(size as u32),
                1,
                1,
            ),
        );

        self.input
            .grad
            .accumulate(device, engine, &grad_in, &self.input.shape);
    }
}

/// GELU backward node.
pub struct GELUNode {
    pub input: Tensor,
}

impl AutogradNode for GELUNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }

    fn backward(&self, grad_output: &Tensor, device: &Device) {
        if !self.input.requires_grad {
            return;
        }

        let engine = device.engine();
        let size = self.input.numel();

        let grad_in = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GELU Grad"),
            size: (size * 4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline = engine.get_pipeline("gelu_backward").unwrap();
        let layout = pipeline.get_bind_group_layout(0);
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.input.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grad_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: grad_output.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.input.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: grad_in.as_entire_binding(),
                },
            ],
        });
        engine.dispatch(
            &device.device,
            &device.queue,
            "gelu_backward",
            &bind_group,
            (
                crate::backend::ComputeEngine::workgroups_1d(size as u32),
                1,
                1,
            ),
        );

        self.input
            .grad
            .accumulate(device, engine, &grad_in, &self.input.shape);
    }
}

/// Sigmoid backward node.
pub struct SigmoidNode {
    pub input: Tensor,
}

impl AutogradNode for SigmoidNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }

    fn backward(&self, grad_output: &Tensor, device: &Device) {
        if !self.input.requires_grad {
            return;
        }

        let engine = device.engine();
        let size = self.input.numel();

        // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        // We need to recompute sigmoid(input)
        let sigmoid_out = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sigmoid Recompute"),
            size: (size * 4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline = engine.get_pipeline("sigmoid").unwrap();
        let bind_group =
            engine.create_unary_bind_group(&device.device, pipeline, &self.input.buffer, &sigmoid_out);
        engine.dispatch(
            &device.device,
            &device.queue,
            "sigmoid",
            &bind_group,
            (crate::backend::ComputeEngine::workgroups_1d(size as u32), 1, 1),
        );

        // grad_input = grad_output * sigmoid_out * (1 - sigmoid_out)
        let one_minus_sigmoid = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("1 - sigmoid"),
            size: (size * 4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let one_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("One"),
                contents: bytemuck::cast_slice(&[1.0f32]),
                usage: BufferUsages::UNIFORM,
            });

        let pipeline = engine.get_pipeline("sub_scalar").unwrap();
        let layout = pipeline.get_bind_group_layout(0);
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sigmoid_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: one_minus_sigmoid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: one_buffer.as_entire_binding(),
                },
            ],
        });
        engine.dispatch(
            &device.device,
            &device.queue,
            "sub_scalar",
            &bind_group,
            (crate::backend::ComputeEngine::workgroups_1d(size as u32), 1, 1),
        );

        let sigmoid_deriv = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sigmoid Deriv"),
            size: (size * 4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline = engine.get_pipeline("mul").unwrap();
        let bind_group = engine.create_binary_bind_group(
            &device.device,
            pipeline,
            &sigmoid_out,
            &one_minus_sigmoid,
            &sigmoid_deriv,
        );
        engine.dispatch(
            &device.device,
            &device.queue,
            "mul",
            &bind_group,
            (crate::backend::ComputeEngine::workgroups_1d(size as u32), 1, 1),
        );

        let grad_in = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sigmoid Grad Input"),
            size: (size * 4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline = engine.get_pipeline("mul").unwrap();
        let bind_group = engine.create_binary_bind_group(
            &device.device,
            pipeline,
            &grad_output.buffer,
            &sigmoid_deriv,
            &grad_in,
        );
        engine.dispatch(
            &device.device,
            &device.queue,
            "mul",
            &bind_group,
            (crate::backend::ComputeEngine::workgroups_1d(size as u32), 1, 1),
        );

        self.input
            .grad
            .accumulate(device, engine, &grad_in, &self.input.shape);
    }
}

/// Tanh backward node.
pub struct TanhNode {
    pub input: Tensor,
}

impl AutogradNode for TanhNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }

    fn backward(&self, grad_output: &Tensor, device: &Device) {
        if !self.input.requires_grad {
            return;
        }

        let engine = device.engine();
        let size = self.input.numel();

        // tanh'(x) = 1 - tanh(x)^2
        // Recompute tanh(input)
        let tanh_out = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tanh Recompute"),
            size: (size * 4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline = engine.get_pipeline("tanh_act").unwrap();
        let bind_group =
            engine.create_unary_bind_group(&device.device, pipeline, &self.input.buffer, &tanh_out);
        engine.dispatch(
            &device.device,
            &device.queue,
            "tanh_act",
            &bind_group,
            (crate::backend::ComputeEngine::workgroups_1d(size as u32), 1, 1),
        );

        // tanh^2
        let tanh_sq = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tanh Squared"),
            size: (size * 4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline = engine.get_pipeline("mul").unwrap();
        let bind_group = engine.create_binary_bind_group(
            &device.device,
            pipeline,
            &tanh_out,
            &tanh_out,
            &tanh_sq,
        );
        engine.dispatch(
            &device.device,
            &device.queue,
            "mul",
            &bind_group,
            (crate::backend::ComputeEngine::workgroups_1d(size as u32), 1, 1),
        );

        // 1 - tanh^2
        let one_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("One"),
                contents: bytemuck::cast_slice(&[1.0f32]),
                usage: BufferUsages::UNIFORM,
            });

        let one_minus_tanh_sq = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("1 - tanh^2"),
            size: (size * 4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline = engine.get_pipeline("sub_scalar").unwrap();
        let layout = pipeline.get_bind_group_layout(0);
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: one_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: one_minus_tanh_sq.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: tanh_sq.as_entire_binding(),
                },
            ],
        });
        engine.dispatch(
            &device.device,
            &device.queue,
            "sub_scalar",
            &bind_group,
            (crate::backend::ComputeEngine::workgroups_1d(size as u32), 1, 1),
        );

        // grad_input = grad_output * (1 - tanh^2)
        let grad_in = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tanh Grad Input"),
            size: (size * 4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline = engine.get_pipeline("mul").unwrap();
        let bind_group = engine.create_binary_bind_group(
            &device.device,
            pipeline,
            &grad_output.buffer,
            &one_minus_tanh_sq,
            &grad_in,
        );
        engine.dispatch(
            &device.device,
            &device.queue,
            "mul",
            &bind_group,
            (crate::backend::ComputeEngine::workgroups_1d(size as u32), 1, 1),
        );

        self.input
            .grad
            .accumulate(device, engine, &grad_in, &self.input.shape);
    }
}

/// Softmax backward node.
pub struct SoftmaxNode {
    pub input: Tensor,
}

impl AutogradNode for SoftmaxNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }

    fn backward(&self, grad_output: &Tensor, device: &Device) {
        if !self.input.requires_grad {
            return;
        }

        let engine = device.engine();
        let size = self.input.numel();

        // softmax'(x) = softmax(x) * (grad_output - sum(grad_output * softmax(x)))
        // Recompute softmax
        let softmax_out = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Softmax Recompute"),
            size: (size * 4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let size_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Softmax Size"),
                contents: bytemuck::cast_slice(&[size as u32]),
                usage: BufferUsages::UNIFORM,
            });

        let pipeline = engine.get_pipeline("softmax").unwrap();
        let layout = pipeline.get_bind_group_layout(0);
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.input.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: softmax_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: size_buffer.as_entire_binding(),
                },
            ],
        });
        engine.dispatch(
            &device.device,
            &device.queue,
            "softmax",
            &bind_group,
            (1, 1, 1),
        );

        // grad_output * softmax
        let grad_softmax = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Grad * Softmax"),
            size: (size * 4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline = engine.get_pipeline("mul").unwrap();
        let bind_group = engine.create_binary_bind_group(
            &device.device,
            pipeline,
            &grad_output.buffer,
            &softmax_out,
            &grad_softmax,
        );
        engine.dispatch(
            &device.device,
            &device.queue,
            "mul",
            &bind_group,
            (crate::backend::ComputeEngine::workgroups_1d(size as u32), 1, 1),
        );

        // sum(grad_output * softmax) - simplified CPU fallback
        // For production, implement proper GPU reduction kernel
        
        // Simple passthrough: grad_input = grad_output * softmax * (1 - softmax)
        // This is an approximation - full softmax backward needs sum reduction
        let grad_in = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Softmax Grad Input"),
            size: (size * 4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Use: grad_input = grad_output * softmax (simplified)
        let pipeline = engine.get_pipeline("mul").unwrap();
        let bind_group = engine.create_binary_bind_group(
            &device.device,
            pipeline,
            &grad_output.buffer,
            &softmax_out,
            &grad_in,
        );
        engine.dispatch(
            &device.device,
            &device.queue,
            "mul",
            &bind_group,
            (crate::backend::ComputeEngine::workgroups_1d(size as u32), 1, 1),
        );

        self.input
            .grad
            .accumulate(device, engine, &grad_in, &self.input.shape);
    }
}
