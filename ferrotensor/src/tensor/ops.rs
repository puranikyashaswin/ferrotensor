//! Tensor Operations
//!
//! Differentiable math on tensors. Binary ops broadcast NumPy-style; every
//! op records a backward closure when an input requires grad (see
//! `autograd.rs`). Gradients of every op here are checked against finite
//! differences in `tensor/tests.rs`.

use super::shape::{broadcast_shapes, contiguous_strides, normalize_dim};
use super::Tensor;
use crate::backend::{binary_op, reduce_op, unary_op};

/// Sum `grad` down to `shape`, undoing a broadcast.
pub(crate) fn sum_to_shape(grad: &Tensor, shape: &[usize]) -> Tensor {
    if grad.shape == shape {
        return grad.clone();
    }
    let mut g = grad.clone();
    while g.ndim() > shape.len() {
        g = g.reduce_raw(0, reduce_op::SUM);
    }
    for (d, &size) in shape.iter().enumerate() {
        if size == 1 && g.shape[d] != 1 {
            let mut keep = g.shape.clone();
            keep[d] = 1;
            g = g.reduce_raw(d, reduce_op::SUM).view_raw(&keep);
        }
    }
    g.view_raw(shape)
}

impl Tensor {
    // ========================================================================
    // BINARY OPERATIONS (broadcasting)
    // ========================================================================

    /// Element-wise `self + other`.
    pub fn add(&self, other: &Tensor) -> Tensor {
        let out = self.binary_raw(other, binary_op::ADD);
        let (sa, sb) = (self.shape.clone(), other.shape.clone());
        out.with_node("add", vec![self.clone(), other.clone()], move |g| {
            vec![Some(sum_to_shape(g, &sa)), Some(sum_to_shape(g, &sb))]
        })
    }

    /// Element-wise `self - other`.
    pub fn sub(&self, other: &Tensor) -> Tensor {
        let out = self.binary_raw(other, binary_op::SUB);
        let (sa, sb) = (self.shape.clone(), other.shape.clone());
        out.with_node("sub", vec![self.clone(), other.clone()], move |g| {
            vec![
                Some(sum_to_shape(g, &sa)),
                Some(sum_to_shape(&g.neg(), &sb)),
            ]
        })
    }

    /// Element-wise `self * other`.
    pub fn mul(&self, other: &Tensor) -> Tensor {
        let out = self.binary_raw(other, binary_op::MUL);
        let (a, b) = (self.detach(), other.detach());
        let need = (self.requires_grad, other.requires_grad);
        out.with_node("mul", vec![self.clone(), other.clone()], move |g| {
            vec![
                need.0.then(|| sum_to_shape(&g.mul(&b), &a.shape)),
                need.1.then(|| sum_to_shape(&g.mul(&a), &b.shape)),
            ]
        })
    }

    /// Element-wise `self / other`.
    pub fn div(&self, other: &Tensor) -> Tensor {
        let out = self.binary_raw(other, binary_op::DIV);
        let (a, b) = (self.detach(), other.detach());
        let need = (self.requires_grad, other.requires_grad);
        out.with_node("div", vec![self.clone(), other.clone()], move |g| {
            vec![
                need.0.then(|| sum_to_shape(&g.div(&b), &a.shape)),
                need.1.then(|| {
                    let gb = g.mul(&a).div(&b.square()).neg();
                    sum_to_shape(&gb, &b.shape)
                }),
            ]
        })
    }

    /// Element-wise maximum. Where equal, the gradient goes to `self`.
    pub fn maximum(&self, other: &Tensor) -> Tensor {
        let out = self.binary_raw(other, binary_op::MAX);
        let (a, b) = (self.detach(), other.detach());
        out.with_node("maximum", vec![self.clone(), other.clone()], move |g| {
            let mask = a.binary_raw(&b, binary_op::GE);
            vec![
                Some(sum_to_shape(&g.mul(&mask), &a.shape)),
                Some(sum_to_shape(&g.mul(&mask.rsub_scalar(1.0)), &b.shape)),
            ]
        })
    }

    /// Element-wise minimum. Where equal, the gradient goes to `self`.
    pub fn minimum(&self, other: &Tensor) -> Tensor {
        let out = self.binary_raw(other, binary_op::MIN);
        let (a, b) = (self.detach(), other.detach());
        out.with_node("minimum", vec![self.clone(), other.clone()], move |g| {
            let mask = a.binary_raw(&b, binary_op::LE);
            vec![
                Some(sum_to_shape(&g.mul(&mask), &a.shape)),
                Some(sum_to_shape(&g.mul(&mask.rsub_scalar(1.0)), &b.shape)),
            ]
        })
    }

    /// `1.0` where `self > other`, else `0.0` (not differentiable).
    pub fn gt(&self, other: &Tensor) -> Tensor {
        self.binary_raw(other, binary_op::GT)
    }

    /// `1.0` where `self < other`, else `0.0` (not differentiable).
    pub fn lt(&self, other: &Tensor) -> Tensor {
        self.binary_raw(other, binary_op::LT)
    }

    /// `1.0` where `self >= other`, else `0.0` (not differentiable).
    pub fn ge(&self, other: &Tensor) -> Tensor {
        self.binary_raw(other, binary_op::GE)
    }

    /// `1.0` where `self <= other`, else `0.0` (not differentiable).
    pub fn le(&self, other: &Tensor) -> Tensor {
        self.binary_raw(other, binary_op::LE)
    }

    /// `1.0` where `self == other`, else `0.0` (not differentiable).
    pub fn eq_elem(&self, other: &Tensor) -> Tensor {
        self.binary_raw(other, binary_op::EQ)
    }

    /// Alias of [`Tensor::lt`].
    pub fn less_than(&self, other: &Tensor) -> Tensor {
        self.lt(other)
    }

    /// Pick `a` where `cond` is non-zero, else `b` (broadcasting).
    ///
    /// `cond` is a 0/1 mask, e.g. from [`Tensor::gt`]. Differentiable in `a`
    /// and `b`.
    pub fn where_cond(cond: &Tensor, a: &Tensor, b: &Tensor) -> Tensor {
        let zero = Tensor::zeros(&[], &cond.device);
        let keep = cond
            .unary_raw(unary_op::ABS, 0.0, 0.0)
            .binary_raw(&zero, binary_op::GT);
        a.mul(&keep).add(&b.mul(&keep.rsub_scalar(1.0)))
    }

    /// Replace elements where `mask` is non-zero with `value` (broadcasting).
    ///
    /// # Panics
    /// Panics when `value` is not finite; use a large finite value such as
    /// `-1e9` for attention masks.
    pub fn masked_fill(&self, mask: &Tensor, value: f32) -> Tensor {
        assert!(value.is_finite(), "masked_fill needs a finite value");
        let fill = Tensor::full(&[], value, &self.device);
        Tensor::where_cond(mask, &fill, self)
    }

    /// Lower triangle of the last two dims: keep elements `[.., i, j]` with
    /// `j - i <= diagonal` and zero the rest. `diagonal = 0` keeps the main
    /// diagonal, positive values keep that many diagonals above it, negative
    /// values drop diagonals below it. Leading dims are batch dims.
    ///
    /// # Panics
    /// Panics when the tensor has fewer than two dims.
    pub fn tril(&self, diagonal: isize) -> Tensor {
        self.mul(&self.triangle_mask(diagonal, true))
    }

    /// Upper triangle of the last two dims: keep elements `[.., i, j]` with
    /// `j - i >= diagonal` and zero the rest (see [`Tensor::tril`]).
    ///
    /// `Tensor::ones(&[l, l], &device).triu(1)` is the causal attention mask
    /// for [`Tensor::masked_fill`]: ones above the diagonal mark the future.
    ///
    /// # Panics
    /// Panics when the tensor has fewer than two dims.
    pub fn triu(&self, diagonal: isize) -> Tensor {
        self.mul(&self.triangle_mask(diagonal, false))
    }

    /// 0/1 mask `[rows, cols]` of the last two dims for `tril` (`lower`) or
    /// `triu`.
    fn triangle_mask(&self, diagonal: isize, lower: bool) -> Tensor {
        assert!(
            self.ndim() >= 2,
            "tril/triu need at least 2 dims, got shape {:?}",
            self.shape
        );
        let (rows, cols) = (self.shape[self.ndim() - 2], self.shape[self.ndim() - 1]);
        let mask: Vec<f32> = (0..rows * cols)
            .map(|k| {
                let offset = (k % cols) as isize - (k / cols) as isize;
                let keep = if lower {
                    offset <= diagonal
                } else {
                    offset >= diagonal
                };
                if keep {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        Tensor::from_vec(mask, &[rows, cols], &self.device)
    }

    // ========================================================================
    // SCALAR OPERATIONS
    // ========================================================================

    fn unary_with_grad(
        &self,
        name: &'static str,
        op: u32,
        a: f32,
        b: f32,
        backward: impl Fn(&Tensor, &Tensor, &Tensor) -> Tensor + Send + Sync + 'static,
    ) -> Tensor {
        let out = self.unary_raw(op, a, b);
        let x = self.detach();
        let y = out.detach();
        out.with_node(name, vec![self.clone()], move |g| {
            vec![Some(backward(g, &x, &y))]
        })
    }

    /// `self + scalar`.
    pub fn add_scalar(&self, scalar: f32) -> Tensor {
        self.unary_with_grad(
            "add_scalar",
            unary_op::ADD_SCALAR,
            scalar,
            0.0,
            |g, _, _| g.clone(),
        )
    }

    /// `self - scalar`.
    pub fn sub_scalar(&self, scalar: f32) -> Tensor {
        self.add_scalar(-scalar)
    }

    /// `scalar - self`.
    pub fn rsub_scalar(&self, scalar: f32) -> Tensor {
        self.unary_with_grad(
            "rsub_scalar",
            unary_op::RSUB_SCALAR,
            scalar,
            0.0,
            |g, _, _| g.neg(),
        )
    }

    /// `self * scalar`.
    pub fn mul_scalar(&self, scalar: f32) -> Tensor {
        self.unary_with_grad(
            "mul_scalar",
            unary_op::MUL_SCALAR,
            scalar,
            0.0,
            move |g, _, _| g.mul_scalar(scalar),
        )
    }

    /// `self / scalar`.
    pub fn div_scalar(&self, scalar: f32) -> Tensor {
        self.mul_scalar(1.0 / scalar)
    }

    /// `self ^ exponent` element-wise.
    pub fn pow_scalar(&self, exponent: f32) -> Tensor {
        self.unary_with_grad(
            "pow_scalar",
            unary_op::POW_SCALAR,
            exponent,
            0.0,
            move |g, x, _| g.mul(&x.pow_scalar(exponent - 1.0).mul_scalar(exponent)),
        )
    }

    /// `-self`.
    pub fn neg(&self) -> Tensor {
        self.unary_with_grad("neg", unary_op::NEG, 0.0, 0.0, |g, _, _| g.neg())
    }

    // ========================================================================
    // ELEMENT-WISE FUNCTIONS
    // ========================================================================

    /// ReLU: `max(0, x)`.
    pub fn relu(&self) -> Tensor {
        self.unary_with_grad("relu", unary_op::RELU, 0.0, 0.0, |g, x, _| {
            g.mul(&x.unary_raw(unary_op::STEP, 0.0, 0.0))
        })
    }

    /// Leaky ReLU: `x` for `x > 0`, else `slope * x`.
    pub fn leaky_relu(&self, slope: f32) -> Tensor {
        self.unary_with_grad(
            "leaky_relu",
            unary_op::LEAKY_RELU,
            slope,
            0.0,
            move |g, x, _| g.mul(&x.unary_raw(unary_op::LEAKY_RELU_GRAD, slope, 0.0)),
        )
    }

    /// SiLU (swish): `x * sigmoid(x)`.
    pub fn silu(&self) -> Tensor {
        self.unary_with_grad("silu", unary_op::SILU, 0.0, 0.0, |g, x, _| {
            g.mul(&x.unary_raw(unary_op::SILU_GRAD, 0.0, 0.0))
        })
    }

    /// GELU, tanh approximation (as in GPT-2 and most LLMs).
    pub fn gelu(&self) -> Tensor {
        self.unary_with_grad("gelu", unary_op::GELU, 0.0, 0.0, |g, x, _| {
            g.mul(&x.unary_raw(unary_op::GELU_GRAD, 0.0, 0.0))
        })
    }

    /// Logistic sigmoid.
    pub fn sigmoid(&self) -> Tensor {
        self.unary_with_grad("sigmoid", unary_op::SIGMOID, 0.0, 0.0, |g, _, y| {
            g.mul(&y.mul(&y.rsub_scalar(1.0)))
        })
    }

    /// Hyperbolic tangent.
    pub fn tanh(&self) -> Tensor {
        self.unary_with_grad("tanh", unary_op::TANH, 0.0, 0.0, |g, _, y| {
            g.mul(&y.square().rsub_scalar(1.0))
        })
    }

    /// `e^x`.
    pub fn exp(&self) -> Tensor {
        self.unary_with_grad("exp", unary_op::EXP, 0.0, 0.0, |g, _, y| g.mul(y))
    }

    /// Natural logarithm.
    pub fn log(&self) -> Tensor {
        self.unary_with_grad("log", unary_op::LOG, 0.0, 0.0, |g, x, _| g.div(x))
    }

    /// Absolute value (subgradient 0 at 0).
    pub fn abs(&self) -> Tensor {
        self.unary_with_grad("abs", unary_op::ABS, 0.0, 0.0, |g, x, _| {
            g.mul(&x.unary_raw(unary_op::SIGN, 0.0, 0.0))
        })
    }

    /// Square root.
    pub fn sqrt(&self) -> Tensor {
        self.unary_with_grad("sqrt", unary_op::SQRT, 0.0, 0.0, |g, _, y| {
            g.div(y).mul_scalar(0.5)
        })
    }

    /// `1 / sqrt(x)`.
    pub fn rsqrt(&self) -> Tensor {
        self.sqrt().recip()
    }

    /// `1 / x`.
    pub fn recip(&self) -> Tensor {
        self.unary_with_grad("recip", unary_op::RECIP, 0.0, 0.0, |g, _, y| {
            g.mul(&y.square()).neg()
        })
    }

    /// `x * x`.
    pub fn square(&self) -> Tensor {
        self.unary_with_grad("square", unary_op::SQUARE, 0.0, 0.0, |g, x, _| {
            g.mul(x).mul_scalar(2.0)
        })
    }

    /// Sine.
    pub fn sin(&self) -> Tensor {
        self.unary_with_grad("sin", unary_op::SIN, 0.0, 0.0, |g, x, _| g.mul(&x.cos()))
    }

    /// Cosine.
    pub fn cos(&self) -> Tensor {
        self.unary_with_grad("cos", unary_op::COS, 0.0, 0.0, |g, x, _| {
            g.mul(&x.sin()).neg()
        })
    }

    /// Clamp to `[min, max]`; the gradient passes only inside the range.
    pub fn clamp(&self, min: f32, max: f32) -> Tensor {
        self.unary_with_grad("clamp", unary_op::CLAMP, min, max, move |g, x, _| {
            g.mul(&x.unary_raw(unary_op::IN_RANGE, min, max))
        })
    }

    /// Sign: -1, 0 or 1 (not differentiable).
    pub fn sign(&self) -> Tensor {
        self.unary_raw(unary_op::SIGN, 0.0, 0.0)
    }

    // ========================================================================
    // REDUCTIONS
    // ========================================================================

    /// Sum of all elements, as a 0-dimensional tensor.
    pub fn sum(&self) -> Tensor {
        let out = self.view_raw(&[self.numel()]).reduce_raw(0, reduce_op::SUM);
        let shape = self.shape.clone();
        out.with_node("sum", vec![self.clone()], move |g| {
            vec![Some(g.expand_raw(&shape))]
        })
    }

    /// Mean of all elements, as a 0-dimensional tensor.
    pub fn mean(&self) -> Tensor {
        self.sum().div_scalar(self.numel() as f32)
    }

    /// Sum over `dim` (negative counts from the end).
    pub fn sum_dim(&self, dim: isize, keepdim: bool) -> Tensor {
        let d = normalize_dim(dim, self.ndim());
        let reduced = self.reduce_raw(d, reduce_op::SUM);
        let mut keep_shape = self.shape.clone();
        keep_shape[d] = 1;
        let out = if keepdim {
            reduced.view_raw(&keep_shape)
        } else {
            reduced
        };
        let shape = self.shape.clone();
        out.with_node("sum_dim", vec![self.clone()], move |g| {
            vec![Some(g.view_raw(&keep_shape).expand_raw(&shape))]
        })
    }

    /// Mean over `dim`.
    pub fn mean_dim(&self, dim: isize, keepdim: bool) -> Tensor {
        let d = normalize_dim(dim, self.ndim());
        self.sum_dim(dim, keepdim).div_scalar(self.shape[d] as f32)
    }

    /// Population variance over `dim`.
    pub fn var_dim(&self, dim: isize, keepdim: bool) -> Tensor {
        let centered = self.sub(&self.mean_dim(dim, true));
        centered.square().mean_dim(dim, keepdim)
    }

    fn extreme(&self, op: u32, name: &'static str) -> Tensor {
        assert!(self.numel() > 0, "{name} of an empty tensor");
        let out = self.view_raw(&[self.numel()]).reduce_raw(0, op);
        let (x, y) = (self.detach(), out.detach());
        out.with_node(name, vec![self.clone()], move |g| {
            // Split the gradient evenly among tied extreme elements.
            let mask = x.binary_raw(&y, binary_op::EQ);
            let count = mask.view_raw(&[mask.numel()]).reduce_raw(0, reduce_op::SUM);
            vec![Some(mask.mul(&g.div(&count)))]
        })
    }

    fn extreme_dim(&self, dim: isize, keepdim: bool, op: u32, name: &'static str) -> Tensor {
        let d = normalize_dim(dim, self.ndim());
        let mut keep_shape = self.shape.clone();
        keep_shape[d] = 1;
        let reduced = self.reduce_raw(d, op).view_raw(&keep_shape);
        let (x, y) = (self.detach(), reduced.detach());
        let out = if keepdim {
            reduced
        } else {
            reduced.view_raw(&[&self.shape[..d], &self.shape[d + 1..]].concat())
        };
        out.with_node(name, vec![self.clone()], move |g| {
            let mask = x.binary_raw(&y, binary_op::EQ);
            let count = mask.reduce_raw(d, reduce_op::SUM).view_raw(&keep_shape);
            vec![Some(mask.mul(&g.view_raw(&keep_shape).div(&count)))]
        })
    }

    /// Largest element, as a 0-dimensional tensor.
    pub fn max(&self) -> Tensor {
        self.extreme(reduce_op::MAX, "max")
    }

    /// Smallest element, as a 0-dimensional tensor.
    pub fn min(&self) -> Tensor {
        self.extreme(reduce_op::MIN, "min")
    }

    /// Maximum over `dim`.
    pub fn max_dim(&self, dim: isize, keepdim: bool) -> Tensor {
        self.extreme_dim(dim, keepdim, reduce_op::MAX, "max_dim")
    }

    /// Minimum over `dim`.
    pub fn min_dim(&self, dim: isize, keepdim: bool) -> Tensor {
        self.extreme_dim(dim, keepdim, reduce_op::MIN, "min_dim")
    }

    /// Index of the first largest element (NaN counts as largest).
    pub fn argmax(&self) -> usize {
        self.arg_index_raw(false)
    }

    /// Index of the first smallest element (NaN counts as largest).
    pub fn argmin(&self) -> usize {
        self.arg_index_raw(true)
    }

    /// Indices of the largest elements along `dim`, as an `f32` tensor with
    /// `dim` removed.
    pub fn argmax_dim(&self, dim: isize) -> Tensor {
        self.arg_reduce_raw(normalize_dim(dim, self.ndim()), false)
    }

    // ========================================================================
    // SHAPE MANIPULATION
    // ========================================================================

    /// Reshape (element count must match). Shares storage with `self`.
    pub fn reshape(&self, new_shape: &[usize]) -> Tensor {
        let new_size: usize = new_shape.iter().product();
        assert_eq!(
            new_size,
            self.numel(),
            "cannot reshape tensor of size {} to shape {:?}",
            self.numel(),
            new_shape
        );
        let out = self.view_raw(new_shape);
        let shape = self.shape.clone();
        out.with_node("reshape", vec![self.clone()], move |g| {
            vec![Some(g.view_raw(&shape))]
        })
    }

    /// Flatten to 1-D.
    pub fn flatten(&self) -> Tensor {
        self.reshape(&[self.numel()])
    }

    /// Remove every dimension of size 1.
    pub fn squeeze(&self) -> Tensor {
        let new_shape: Vec<usize> = self.shape.iter().copied().filter(|&d| d != 1).collect();
        self.reshape(&new_shape)
    }

    /// Remove dimension `dim`, which must have size 1.
    pub fn squeeze_dim(&self, dim: isize) -> Tensor {
        let d = normalize_dim(dim, self.ndim());
        assert_eq!(
            self.shape[d], 1,
            "squeeze_dim on a dimension of size {}",
            self.shape[d]
        );
        let mut new_shape = self.shape.clone();
        new_shape.remove(d);
        self.reshape(&new_shape)
    }

    /// Insert a dimension of size 1 at `dim`.
    pub fn unsqueeze(&self, dim: usize) -> Tensor {
        assert!(
            dim <= self.ndim(),
            "dimension {} out of range for a {}-D tensor",
            dim,
            self.ndim()
        );
        let mut new_shape = self.shape.clone();
        new_shape.insert(dim, 1);
        self.reshape(&new_shape)
    }

    /// Reorder dimensions: output dim `d` is input dim `perm[d]`.
    pub fn permute(&self, perm: &[usize]) -> Tensor {
        let mut seen = vec![false; self.ndim()];
        assert_eq!(
            perm.len(),
            self.ndim(),
            "permute needs one entry per dimension"
        );
        for &p in perm {
            assert!(p < self.ndim() && !seen[p], "invalid permutation {perm:?}");
            seen[p] = true;
        }
        let out = self.permute_raw(perm);
        let mut inverse = vec![0; perm.len()];
        for (i, &p) in perm.iter().enumerate() {
            inverse[p] = i;
        }
        out.with_node("permute", vec![self.clone()], move |g| {
            vec![Some(g.permute_raw(&inverse))]
        })
    }

    /// Swap dimensions `dim0` and `dim1`.
    pub fn transpose_dims(&self, dim0: isize, dim1: isize) -> Tensor {
        let (a, b) = (
            normalize_dim(dim0, self.ndim()),
            normalize_dim(dim1, self.ndim()),
        );
        let mut perm: Vec<usize> = (0..self.ndim()).collect();
        perm.swap(a, b);
        self.permute(&perm)
    }

    /// Swap the last two dimensions (matrix transpose).
    pub fn transpose(&self) -> Tensor {
        assert!(self.ndim() >= 2, "transpose needs at least 2 dims");
        self.transpose_dims(-2, -1)
    }

    /// Broadcast to `shape` (materialised copy).
    pub fn expand(&self, shape: &[usize]) -> Tensor {
        assert_eq!(
            broadcast_shapes(&self.shape, shape).as_deref(),
            Some(shape),
            "cannot expand {:?} to {:?}",
            self.shape,
            shape
        );
        let out = self.expand_raw(shape);
        let orig = self.shape.clone();
        out.with_node("expand", vec![self.clone()], move |g| {
            vec![Some(sum_to_shape(g, &orig))]
        })
    }

    /// The slice `start..start + len` of dimension `dim`.
    pub fn narrow(&self, dim: isize, start: usize, len: usize) -> Tensor {
        let d = normalize_dim(dim, self.ndim());
        assert!(
            start + len <= self.shape[d],
            "narrow {}..{} out of range for dimension of size {}",
            start,
            start + len,
            self.shape[d]
        );
        let strides = contiguous_strides(&self.shape);
        let mut out_shape = self.shape.clone();
        out_shape[d] = len;
        let out = self.strided_copy_raw(&out_shape, &strides, start * strides[d]);
        let shape = self.shape.clone();
        out.with_node("narrow", vec![self.clone()], move |g| {
            let mut parts = Vec::new();
            let mut before = shape.clone();
            before[d] = start;
            let mut after = shape.clone();
            after[d] = shape[d] - start - len;
            if start > 0 {
                parts.push(Tensor::zeros(&before, &g.device));
            }
            parts.push(g.clone());
            if after[d] > 0 {
                parts.push(Tensor::zeros(&after, &g.device));
            }
            vec![Some(Tensor::cat(&parts, d as isize))]
        })
    }

    /// Split into `chunks` pieces along `dim` (the last may be smaller).
    pub fn chunk(&self, chunks: usize, dim: isize) -> Vec<Tensor> {
        assert!(chunks > 0, "chunk count must be positive");
        let d = normalize_dim(dim, self.ndim());
        let size = self.shape[d].div_ceil(chunks).max(1);
        self.split(size, dim)
    }

    /// Split into pieces of `size` along `dim` (the last may be smaller).
    pub fn split(&self, size: usize, dim: isize) -> Vec<Tensor> {
        assert!(size > 0, "split size must be positive");
        let d = normalize_dim(dim, self.ndim());
        (0..self.shape[d])
            .step_by(size)
            .map(|start| self.narrow(dim, start, size.min(self.shape[d] - start)))
            .collect()
    }

    /// Concatenate tensors along `dim`; all other dims must match.
    pub fn cat(tensors: &[Tensor], dim: isize) -> Tensor {
        assert!(!tensors.is_empty(), "cat needs at least one tensor");
        let first = &tensors[0];
        let d = normalize_dim(dim, first.ndim());
        for t in tensors {
            first.assert_same_device(t);
            assert_eq!(t.ndim(), first.ndim(), "cat needs tensors of equal rank");
            for (i, (&a, &b)) in t.shape.iter().zip(&first.shape).enumerate() {
                assert!(
                    i == d || a == b,
                    "cat shapes differ: {:?} vs {:?}",
                    t.shape,
                    first.shape
                );
            }
        }
        let mut out_shape = first.shape.clone();
        out_shape[d] = tensors.iter().map(|t| t.shape[d]).sum();
        let outer: usize = first.shape[..d].iter().product();
        let inner: usize = first.shape[d + 1..].iter().product();
        let datas: Vec<Vec<f32>> = tensors.iter().map(Tensor::to_vec).collect();
        let mut data = Vec::with_capacity(out_shape.iter().product());
        for o in 0..outer {
            for (t, values) in tensors.iter().zip(&datas) {
                let block = t.shape[d] * inner;
                data.extend_from_slice(&values[o * block..(o + 1) * block]);
            }
        }
        let out = first.host_output(data, &out_shape);
        let sizes: Vec<usize> = tensors.iter().map(|t| t.shape[d]).collect();
        out.with_node("cat", tensors.to_vec(), move |g| {
            let mut start = 0;
            sizes
                .iter()
                .map(|&len| {
                    let piece = g.narrow(d as isize, start, len);
                    start += len;
                    Some(piece)
                })
                .collect()
        })
    }

    /// Stack equally shaped tensors along a new dimension `dim`.
    pub fn stack(tensors: &[Tensor], dim: usize) -> Tensor {
        let expanded: Vec<Tensor> = tensors.iter().map(|t| t.unsqueeze(dim)).collect();
        Tensor::cat(&expanded, dim as isize)
    }

    /// Select entries `indices` along `dim`.
    ///
    /// With `dim = 0` on a `[vocab, dim]` table this is an embedding lookup.
    pub fn index_select(&self, dim: isize, indices: &[usize]) -> Tensor {
        let d = normalize_dim(dim, self.ndim());
        let out = if d == 0 {
            self.gather_rows_raw(indices)
        } else {
            let mut perm: Vec<usize> = (0..self.ndim()).collect();
            perm.swap(0, d);
            self.permute_raw(&perm)
                .gather_rows_raw(indices)
                .permute_raw(&perm)
        };
        let shape = self.shape.clone();
        let indices = indices.to_vec();
        out.with_node("index_select", vec![self.clone()], move |g| {
            vec![Some(index_add(g, d, &indices, &shape))]
        })
    }

    /// Embedding lookup: rows of this `[vocab, ..]` table picked by the ids
    /// stored as `f32` in `ids`, with shape `ids.shape + self.shape[1..]`.
    ///
    /// Ids outside `[0, vocab)` give rows of zeros and no gradient. On a GPU
    /// the forward pass reads the ids on the device; the backward pass reads
    /// them back to scatter the gradient, like [`Tensor::index_select`].
    pub(crate) fn embedding(&self, ids: &Tensor) -> Tensor {
        let out = self.gather_rows_by_ids_raw(ids);
        let ids = ids.detach();
        let shape = self.shape.clone();
        out.with_node("embedding", vec![self.clone()], move |g| {
            let row: usize = shape[1..].iter().product();
            let grad = g.to_vec();
            let mut out = vec![0.0f32; shape.iter().product()];
            for (r, id) in super::raw::valid_ids(&ids.to_vec(), shape[0]) {
                for (o, &v) in out[id * row..][..row]
                    .iter_mut()
                    .zip(&grad[r * row..][..row])
                {
                    *o += v;
                }
            }
            vec![Some(g.host_output(out, &shape))]
        })
    }

    // ========================================================================
    // MATRIX PRODUCT AND SOFTMAX
    // ========================================================================

    /// Matrix product over the last two dims.
    ///
    /// Supports `[.., M, K] @ [K, N]`, `[M, K] @ [.., K, N]` and batched
    /// `[B.., M, K] @ [B.., K, N]` with equal batch dims.
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let out = self.matmul_raw(other);
        let (a, b) = (self.detach(), other.detach());
        let need = (self.requires_grad, other.requires_grad);
        out.with_node("matmul", vec![self.clone(), other.clone()], move |g| {
            vec![
                need.0
                    .then(|| sum_to_shape(&g.matmul_raw(&b.transpose()), &a.shape)),
                need.1
                    .then(|| sum_to_shape(&a.transpose().matmul_raw(g), &b.shape)),
            ]
        })
    }

    /// Softmax over `dim` (negative counts from the end).
    pub fn softmax(&self, dim: isize) -> Tensor {
        let d = normalize_dim(dim, self.ndim());
        let out = self.softmax_raw(d, false);
        let y = out.detach();
        out.with_node("softmax", vec![self.clone()], move |g| {
            let dot = g.mul(&y).sum_dim(d as isize, true);
            vec![Some(y.mul(&g.sub(&dot)))]
        })
    }

    /// Log-softmax over `dim` (numerically stable).
    pub fn log_softmax(&self, dim: isize) -> Tensor {
        let d = normalize_dim(dim, self.ndim());
        let out = self.softmax_raw(d, true);
        let y = out.detach();
        out.with_node("log_softmax", vec![self.clone()], move |g| {
            let total = g.sum_dim(d as isize, true);
            vec![Some(g.sub(&y.exp().mul(&total)))]
        })
    }
}

/// Gradient of `index_select`: scatter-add `grad` back into a zero tensor of
/// `shape` along `dim`.
fn index_add(grad: &Tensor, dim: usize, indices: &[usize], shape: &[usize]) -> Tensor {
    let (outer, size, inner) = super::shape::split_at_dim(shape, dim);
    let g = grad.to_vec();
    let mut out = vec![0.0f32; shape.iter().product()];
    let n_idx = indices.len();
    for o in 0..outer {
        for (j, &idx) in indices.iter().enumerate() {
            let src = (o * n_idx + j) * inner;
            let dst = (o * size + idx) * inner;
            for k in 0..inner {
                out[dst + k] += g[src + k];
            }
        }
    }
    grad.host_output(out, shape)
}

// ============================================================================
// OPERATOR OVERLOADS
// ============================================================================

macro_rules! binary_operator {
    ($trait:ident, $method:ident, $scalar_fn:expr) => {
        impl std::ops::$trait<&Tensor> for &Tensor {
            type Output = Tensor;
            fn $method(self, other: &Tensor) -> Tensor {
                Tensor::$method(self, other)
            }
        }

        impl std::ops::$trait<Tensor> for Tensor {
            type Output = Tensor;
            fn $method(self, other: Tensor) -> Tensor {
                Tensor::$method(&self, &other)
            }
        }

        impl std::ops::$trait<f32> for &Tensor {
            type Output = Tensor;
            fn $method(self, scalar: f32) -> Tensor {
                let f: fn(&Tensor, f32) -> Tensor = $scalar_fn;
                f(self, scalar)
            }
        }
    };
}

binary_operator!(Add, add, Tensor::add_scalar);
binary_operator!(Sub, sub, Tensor::sub_scalar);
binary_operator!(Mul, mul, Tensor::mul_scalar);
binary_operator!(Div, div, Tensor::div_scalar);

impl std::ops::Neg for &Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        Tensor::neg(self)
    }
}

impl std::ops::Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        Tensor::neg(&self)
    }
}
