//! `ferrotensor.Tensor` and the functions that create tensors.

use std::cell::RefCell;

use ferrotensor::tensor::{is_grad_enabled, NoGradGuard};
use ferrotensor::{Device, Tensor};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyFloat, PyList, PySlice, PyString, PyTuple};

use crate::device::{resolve_device, PyDevice};
use crate::error::{guard, index_error, runtime_error, value_error};

/// Tensors with more elements than this print their shape instead of values.
const REPR_MAX_ELEMENTS: usize = 1000;

/// A float32 tensor on the CPU or a GPU, with autograd.
#[pyclass(name = "Tensor", module = "ferrotensor")]
#[derive(Clone)]
pub(crate) struct PyTensor {
    pub(crate) inner: Tensor,
}

impl From<Tensor> for PyTensor {
    fn from(inner: Tensor) -> Self {
        Self { inner }
    }
}

/// The right-hand side of an arithmetic op: a tensor or a Python number.
#[derive(FromPyObject)]
pub(crate) enum Operand<'py> {
    Tensor(PyRef<'py, PyTensor>),
    Scalar(f64),
}

// ============================================================================
// ARGUMENT CHECKS
// ============================================================================

pub(crate) fn check_same_device(a: &Tensor, b: &Tensor) -> PyResult<()> {
    if a.device().same_as(b.device()) {
        Ok(())
    } else {
        Err(value_error(format!(
            "expected tensors on the same device, got {} and {}",
            a.device().name(),
            b.device().name()
        )))
    }
}

/// NumPy broadcast of two shapes, or a `ValueError` naming `op`.
pub(crate) fn broadcast(op: &str, a: &[usize], b: &[usize]) -> PyResult<Vec<usize>> {
    let ndim = a.len().max(b.len());
    let dim = |s: &[usize], i: usize| {
        if i < s.len() {
            s[s.len() - 1 - i]
        } else {
            1
        }
    };
    let mut out = vec![0; ndim];
    for i in 0..ndim {
        out[ndim - 1 - i] = match (dim(a, i), dim(b, i)) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            _ => {
                return Err(value_error(format!(
                    "{op}: shapes {a:?} and {b:?} cannot be broadcast together"
                )))
            }
        };
    }
    Ok(out)
}

/// Resolve a possibly negative `dim` for a tensor with `ndim` dimensions.
pub(crate) fn normalize_dim(dim: isize, ndim: usize) -> PyResult<usize> {
    let n = ndim as isize;
    let d = if dim < 0 { dim + n } else { dim };
    if (0..n).contains(&d) {
        Ok(d as usize)
    } else {
        Err(index_error(format!(
            "dimension {dim} is out of range for a {ndim}-D tensor"
        )))
    }
}

/// Check that `t` has a single element (for `item()`, implicit `backward()`).
fn check_one_element(t: &Tensor, what: &str) -> PyResult<()> {
    if t.numel() == 1 {
        Ok(())
    } else {
        Err(value_error(format!(
            "{what} needs a one-element tensor, got shape {:?}",
            t.shape
        )))
    }
}

/// Shape from `*size` arguments: `f(2, 3)`, `f((2, 3))` or `f([2, 3])`.
pub(crate) fn parse_shape(args: &Bound<'_, PyTuple>) -> PyResult<Vec<usize>> {
    let items = shape_items(args)?;
    items
        .iter()
        .map(|x| {
            let v: i64 = x.extract()?;
            usize::try_from(v).map_err(|_| value_error(format!("negative dimension {v} in shape")))
        })
        .collect()
}

fn shape_items<'py>(args: &Bound<'py, PyTuple>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    if args.len() == 1 {
        let first = args.get_item(0)?;
        if first.is_instance_of::<PyTuple>() || first.is_instance_of::<PyList>() {
            return first.try_iter()?.collect();
        }
    }
    Ok(args.iter().collect())
}

/// Shape for `reshape`/`view`, where one entry may be `-1`.
fn parse_view_shape(args: &Bound<'_, PyTuple>, numel: usize) -> PyResult<Vec<usize>> {
    let dims: Vec<i64> = shape_items(args)?
        .iter()
        .map(|x| x.extract())
        .collect::<PyResult<_>>()?;
    let mut infer = None;
    let mut known = 1usize;
    for (i, &d) in dims.iter().enumerate() {
        match d {
            -1 if infer.is_none() => infer = Some(i),
            -1 => return Err(value_error("only one dimension can be -1")),
            d if d < 0 => return Err(value_error(format!("invalid dimension {d} in shape"))),
            d => known *= d as usize,
        }
    }
    let mut shape: Vec<usize> = dims.iter().map(|&d| d.max(0) as usize).collect();
    if let Some(i) = infer {
        if known == 0 || !numel.is_multiple_of(known) {
            return Err(value_error(format!(
                "shape {dims:?} is invalid for a tensor of {numel} elements"
            )));
        }
        shape[i] = numel / known;
    } else if known != numel {
        return Err(value_error(format!(
            "shape {dims:?} is invalid for a tensor of {numel} elements"
        )));
    }
    Ok(shape)
}

/// Dimensions from an `int`, a sequence of ints, or `None` (all).
fn parse_dims(dim: Option<&Bound<'_, PyAny>>, ndim: usize) -> PyResult<Option<Vec<usize>>> {
    let Some(dim) = dim.filter(|d| !d.is_none()) else {
        return Ok(None);
    };
    let raw: Vec<isize> = if let Ok(d) = dim.extract::<isize>() {
        vec![d]
    } else {
        dim.extract()?
    };
    let mut dims = raw
        .into_iter()
        .map(|d| normalize_dim(d, ndim))
        .collect::<PyResult<Vec<_>>>()?;
    dims.sort_unstable();
    dims.dedup();
    Ok(Some(dims))
}

// ============================================================================
// PYTHON DATA <-> TENSORS
// ============================================================================

/// Flatten a number or nested list/tuple (or anything with `.tolist()`,
/// such as `array.array` or a NumPy array) into row-major values and a shape.
pub(crate) fn parse_data(obj: &Bound<'_, PyAny>) -> PyResult<(Vec<f32>, Vec<usize>)> {
    if let Ok(t) = obj.cast::<PyTensor>() {
        let t = &t.borrow().inner;
        return Ok((guard(|| t.to_vec())?, t.shape.clone()));
    }
    let is_seq = obj.is_instance_of::<PyList>() || obj.is_instance_of::<PyTuple>();
    let obj = if !is_seq && obj.hasattr("tolist")? {
        obj.call_method0("tolist")?
    } else {
        obj.clone()
    };
    let mut shape = Vec::new();
    let mut leaf_depth = None;
    let mut values = Vec::new();
    walk(&obj, 0, &mut shape, &mut leaf_depth, &mut values)?;
    Ok((values, shape))
}

fn ragged() -> PyErr {
    value_error("expected a rectangular nested sequence (all rows the same length)")
}

fn walk(
    obj: &Bound<'_, PyAny>,
    depth: usize,
    shape: &mut Vec<usize>,
    leaf_depth: &mut Option<usize>,
    values: &mut Vec<f32>,
) -> PyResult<()> {
    if obj.is_instance_of::<PyList>() || obj.is_instance_of::<PyTuple>() {
        let n = obj.len()?;
        if leaf_depth.is_some_and(|l| l <= depth) {
            return Err(ragged());
        }
        if shape.len() == depth {
            shape.push(n);
        } else if shape[depth] != n {
            return Err(ragged());
        }
        for item in obj.try_iter()? {
            walk(&item?, depth + 1, shape, leaf_depth, values)?;
        }
        return Ok(());
    }
    let not_a_number = || {
        let ty = obj
            .get_type()
            .name()
            .map(|n| n.to_string())
            .unwrap_or_default();
        PyTypeError::new_err(format!(
            "expected a number or a nested sequence of numbers, got {ty}"
        ))
    };
    if obj.is_instance_of::<PyString>() {
        return Err(not_a_number());
    }
    let v: f64 = obj.extract().map_err(|_| not_a_number())?;
    match *leaf_depth {
        None if shape.len() == depth => *leaf_depth = Some(depth),
        Some(l) if l == depth => {}
        _ => return Err(ragged()),
    }
    values.push(v as f32);
    Ok(())
}

/// Nested Python lists (or a float for a 0-D tensor) from row-major values.
fn nested<'py>(py: Python<'py>, data: &[f32], shape: &[usize]) -> PyResult<Bound<'py, PyAny>> {
    match shape {
        [] => Ok(PyFloat::new(py, data.first().copied().map_or(0.0, f64::from)).into_any()),
        [_] => Ok(PyList::new(py, data.iter().map(|&v| f64::from(v)))?.into_any()),
        [rows, rest @ ..] => {
            let stride: usize = rest.iter().product();
            let items = (0..*rows)
                .map(|i| nested(py, &data[i * stride..(i + 1) * stride], rest))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyList::new(py, items)?.into_any())
        }
    }
}

/// Values as nested brackets, each float in its shortest f32 form.
fn format_nested(data: &[f32], shape: &[usize]) -> String {
    match shape {
        [] => format!("{:?}", data.first().copied().unwrap_or(0.0)),
        [rows, rest @ ..] => {
            let stride: usize = rest.iter().product();
            let items: Vec<String> = (0..*rows)
                .map(|i| format_nested(&data[i * stride..(i + 1) * stride], rest))
                .collect();
            format!("[{}]", items.join(", "))
        }
    }
}

fn with_grad(t: Tensor, requires_grad: bool) -> PyTensor {
    if requires_grad {
        t.requires_grad_().into()
    } else {
        t.into()
    }
}

/// Build a tensor from Python data on `device`.
pub(crate) fn tensor_from_py(obj: &Bound<'_, PyAny>, device: &Device) -> PyResult<Tensor> {
    let (values, shape) = parse_data(obj)?;
    guard(|| Tensor::from_vec(values, &shape, device))
}

// ============================================================================
// TENSOR
// ============================================================================

type BinaryFn = fn(&Tensor, &Tensor) -> Tensor;

impl PyTensor {
    /// `other` as a tensor on this tensor's device, checked for broadcasting.
    fn operand(&self, op: &str, other: &Operand<'_>) -> PyResult<Tensor> {
        match other {
            Operand::Tensor(o) => {
                check_same_device(&self.inner, &o.inner)?;
                broadcast(op, &self.inner.shape, &o.inner.shape)?;
                Ok(o.inner.clone())
            }
            Operand::Scalar(s) => guard(|| Tensor::scalar(*s as f32, self.inner.device())),
        }
    }

    fn binary(&self, op: &str, other: &Operand<'_>, f: BinaryFn) -> PyResult<Self> {
        let o = self.operand(op, other)?;
        guard(|| f(&self.inner, &o)).map(Into::into)
    }

    fn map(&self, f: impl FnOnce(&Tensor) -> Tensor) -> PyResult<Self> {
        guard(|| f(&self.inner)).map(Into::into)
    }

    /// Apply a per-dimension reduction over `dims` (all when `None`).
    fn reduce(
        &self,
        dim: Option<&Bound<'_, PyAny>>,
        keepdim: bool,
        all: fn(&Tensor) -> Tensor,
        along: fn(&Tensor, isize, bool) -> Tensor,
    ) -> PyResult<Self> {
        let t = &self.inner;
        match parse_dims(dim, t.ndim())? {
            None => guard(|| {
                let r = all(t);
                if keepdim {
                    r.reshape(&vec![1; t.ndim()])
                } else {
                    r
                }
            }),
            // Highest dim first, so lower dims keep their index.
            Some(dims) => guard(|| {
                dims.iter()
                    .rev()
                    .fold(t.clone(), |acc, &d| along(&acc, d as isize, keepdim))
            }),
        }
        .map(Into::into)
    }

    fn matmul_impl(&self, other: &PyTensor) -> PyResult<Self> {
        let (a, b) = (&self.inner, &other.inner);
        check_same_device(a, b)?;
        if a.ndim() == 0 || b.ndim() == 0 {
            return Err(value_error(
                "matmul needs tensors with at least 1 dimension",
            ));
        }
        let k_a = a.shape[a.ndim() - 1];
        let k_b = if b.ndim() == 1 {
            b.shape[0]
        } else {
            b.shape[b.ndim() - 2]
        };
        if k_a != k_b {
            return Err(value_error(format!(
                "matmul: inner dimensions differ for shapes {:?} and {:?}",
                a.shape, b.shape
            )));
        }
        let (ba, bb) = (
            &a.shape[..a.ndim().saturating_sub(2)],
            &b.shape[..b.ndim().saturating_sub(2)],
        );
        if !(ba == bb || ba.is_empty() || bb.is_empty()) {
            return Err(value_error(format!(
                "matmul: batch dimensions of {:?} and {:?} must match or be absent on one side",
                a.shape, b.shape
            )));
        }
        guard(|| {
            // 1-D operands follow NumPy: promote, multiply, then drop the dim.
            let a2 = if a.ndim() == 1 {
                a.unsqueeze(0)
            } else {
                a.clone()
            };
            let b2 = if b.ndim() == 1 {
                b.unsqueeze(1)
            } else {
                b.clone()
            };
            let mut out = a2.matmul(&b2);
            if b.ndim() == 1 {
                out = out.squeeze_dim(-1);
            }
            if a.ndim() == 1 {
                out = out.squeeze_dim(if b.ndim() == 1 { 0 } else { -2 });
            }
            out
        })
        .map(Into::into)
    }
}

#[pymethods]
impl PyTensor {
    #[new]
    #[pyo3(signature = (data, *, device = None, requires_grad = false))]
    fn new(
        data: &Bound<'_, PyAny>,
        device: Option<&Bound<'_, PyAny>>,
        requires_grad: bool,
    ) -> PyResult<Self> {
        let device = resolve_device(device)?;
        Ok(with_grad(tensor_from_py(data, &device)?, requires_grad))
    }

    // ---- metadata ----------------------------------------------------------

    /// The shape as a tuple of ints.
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.inner.shape)
    }

    /// The shape (no argument) or the size of dimension `dim`.
    #[pyo3(signature = (dim = None))]
    fn size<'py>(&self, py: Python<'py>, dim: Option<isize>) -> PyResult<Bound<'py, PyAny>> {
        match dim {
            None => Ok(self.shape(py)?.into_any()),
            Some(d) => {
                let d = normalize_dim(d, self.inner.ndim())?;
                Ok(self.inner.shape[d].into_pyobject(py)?.into_any())
            }
        }
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    fn dim(&self) -> usize {
        self.inner.ndim()
    }

    fn numel(&self) -> usize {
        self.inner.numel()
    }

    /// Always `"float32"`.
    #[getter]
    fn dtype(&self) -> &'static str {
        "float32"
    }

    #[getter]
    fn device(&self) -> PyDevice {
        PyDevice {
            inner: self.inner.device().clone(),
        }
    }

    fn __len__(&self) -> PyResult<usize> {
        self.inner
            .shape
            .first()
            .copied()
            .ok_or_else(|| PyTypeError::new_err("len() of a 0-d tensor"))
    }

    // ---- autograd ----------------------------------------------------------

    #[getter]
    fn requires_grad(&self) -> bool {
        self.inner.requires_grad
    }

    #[setter]
    fn set_requires_grad(&mut self, requires_grad: bool) -> PyResult<()> {
        if !self.inner.is_leaf() {
            return Err(runtime_error(
                "requires_grad can only be changed on leaf tensors; use detach() first",
            ));
        }
        self.inner.set_requires_grad(requires_grad);
        Ok(())
    }

    /// Set `requires_grad` in place and return `self`.
    #[pyo3(signature = (requires_grad = true))]
    fn requires_grad_(
        mut slf: PyRefMut<'_, Self>,
        requires_grad: bool,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.set_requires_grad(requires_grad)?;
        Ok(slf)
    }

    /// True for tensors not produced by a tracked op.
    #[getter]
    fn is_leaf(&self) -> bool {
        self.inner.is_leaf()
    }

    /// Name of the op that produced this tensor, or `None` for leaves.
    #[getter]
    fn grad_fn(&self) -> Option<&'static str> {
        self.inner.grad_fn_name()
    }

    /// The accumulated gradient, or `None`.
    #[getter]
    fn grad(&self) -> PyResult<Option<PyTensor>> {
        guard(|| self.inner.grad().map(Into::into))
    }

    #[setter(grad)]
    fn set_grad(&self, grad: Option<PyRef<'_, PyTensor>>) -> PyResult<()> {
        if let Some(g) = &grad {
            if g.inner.shape != self.inner.shape {
                return Err(value_error(format!(
                    "gradient shape {:?} does not match tensor shape {:?}",
                    g.inner.shape, self.inner.shape
                )));
            }
            check_same_device(&self.inner, &g.inner)?;
        }
        guard(|| self.inner.set_grad(grad.map(|g| g.inner.clone())))
    }

    /// Backpropagate from this tensor into the `.grad` of every leaf.
    #[pyo3(signature = (gradient = None))]
    fn backward(&self, gradient: Option<PyRef<'_, PyTensor>>) -> PyResult<()> {
        if !self.inner.requires_grad {
            return Err(runtime_error(
                "backward() on a tensor that does not require grad",
            ));
        }
        match gradient {
            None => {
                check_one_element(&self.inner, "backward() without a gradient")?;
                guard(|| self.inner.backward())
            }
            Some(g) => {
                if g.inner.shape != self.inner.shape {
                    return Err(value_error(format!(
                        "gradient shape {:?} does not match tensor shape {:?}",
                        g.inner.shape, self.inner.shape
                    )));
                }
                check_same_device(&self.inner, &g.inner)?;
                guard(|| self.inner.backward_with(&g.inner))
            }
        }
    }

    /// The same values, cut from the autograd graph (shares storage).
    fn detach(&self) -> Self {
        self.inner.detach().into()
    }

    /// A copy with its own storage (differentiable).
    fn clone(&self) -> PyResult<Self> {
        self.map(|t| t.add_scalar(0.0))
    }

    /// Overwrite the values in place with `src` (not tracked by autograd).
    fn copy_<'py>(slf: PyRef<'py, Self>, src: PyRef<'_, PyTensor>) -> PyResult<PyRef<'py, Self>> {
        if src.inner.shape != slf.inner.shape {
            return Err(value_error(format!(
                "copy_: source shape {:?} does not match {:?}",
                src.inner.shape, slf.inner.shape
            )));
        }
        guard(|| slf.inner.assign(&src.inner))?;
        Ok(slf)
    }

    // ---- data out ----------------------------------------------------------

    /// The values as (nested) Python lists; a float for a 0-D tensor.
    fn tolist<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let data = guard(|| self.inner.to_vec())?;
        nested(py, &data, &self.inner.shape)
    }

    /// The value of a one-element tensor as a float.
    fn item(&self) -> PyResult<f32> {
        check_one_element(&self.inner, "item()")?;
        guard(|| self.inner.item())
    }

    fn __float__(&self) -> PyResult<f32> {
        self.item()
    }

    /// Copy to `device` (differentiable).
    fn to(&self, device: &Bound<'_, PyAny>) -> PyResult<Self> {
        let device = resolve_device(Some(device))?;
        self.map(|t| t.to_device(&device))
    }

    fn cpu(&self) -> PyResult<Self> {
        self.map(|t| t.to_device(&Device::cpu()))
    }

    fn __repr__(&self) -> PyResult<String> {
        let t = &self.inner;
        let mut extra = String::new();
        if t.device().is_gpu() {
            extra.push_str(", device='gpu'");
        }
        match t.grad_fn_name() {
            Some(name) => extra.push_str(&format!(", grad_fn=<{name}>")),
            None if t.requires_grad => extra.push_str(", requires_grad=True"),
            None => {}
        }
        if t.numel() > REPR_MAX_ELEMENTS {
            return Ok(format!("tensor(shape={:?}{extra})", t.shape));
        }
        let data = guard(|| t.to_vec())?;
        Ok(format!("tensor({}{extra})", format_nested(&data, &t.shape)))
    }

    // ---- operators ---------------------------------------------------------

    fn __add__(&self, other: Operand<'_>) -> PyResult<Self> {
        match other {
            Operand::Scalar(s) => self.map(|t| t.add_scalar(s as f32)),
            o => self.binary("add", &o, Tensor::add),
        }
    }

    fn __radd__(&self, other: Operand<'_>) -> PyResult<Self> {
        self.__add__(other)
    }

    fn __sub__(&self, other: Operand<'_>) -> PyResult<Self> {
        match other {
            Operand::Scalar(s) => self.map(|t| t.sub_scalar(s as f32)),
            o => self.binary("sub", &o, Tensor::sub),
        }
    }

    fn __rsub__(&self, other: f64) -> PyResult<Self> {
        self.map(|t| t.rsub_scalar(other as f32))
    }

    fn __mul__(&self, other: Operand<'_>) -> PyResult<Self> {
        match other {
            Operand::Scalar(s) => self.map(|t| t.mul_scalar(s as f32)),
            o => self.binary("mul", &o, Tensor::mul),
        }
    }

    fn __rmul__(&self, other: Operand<'_>) -> PyResult<Self> {
        self.__mul__(other)
    }

    fn __truediv__(&self, other: Operand<'_>) -> PyResult<Self> {
        match other {
            Operand::Scalar(s) => self.map(|t| t.div_scalar(s as f32)),
            o => self.binary("div", &o, Tensor::div),
        }
    }

    fn __rtruediv__(&self, other: f64) -> PyResult<Self> {
        self.map(|t| t.recip().mul_scalar(other as f32))
    }

    fn __pow__(&self, exponent: f64, _modulo: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        self.map(|t| t.pow_scalar(exponent as f32))
    }

    fn __matmul__(&self, other: PyRef<'_, PyTensor>) -> PyResult<Self> {
        self.matmul_impl(&other)
    }

    fn __neg__(&self) -> PyResult<Self> {
        self.map(Tensor::neg)
    }

    fn __abs__(&self) -> PyResult<Self> {
        self.map(Tensor::abs)
    }

    /// Hash by identity (`==` is identity too; use `eq()` for values).
    fn __hash__(slf: &Bound<'_, Self>) -> u64 {
        slf.as_ptr() as usize as u64
    }

    fn __lt__(&self, other: Operand<'_>) -> PyResult<Self> {
        self.binary("lt", &other, Tensor::lt)
    }

    fn __le__(&self, other: Operand<'_>) -> PyResult<Self> {
        self.binary("le", &other, Tensor::le)
    }

    fn __gt__(&self, other: Operand<'_>) -> PyResult<Self> {
        self.binary("gt", &other, Tensor::gt)
    }

    fn __ge__(&self, other: Operand<'_>) -> PyResult<Self> {
        self.binary("ge", &other, Tensor::ge)
    }

    /// Index with ints and slices, e.g. `t[0]`, `t[:, 1:3]`, `t[::2]`.
    fn __getitem__(&self, index: &Bound<'_, PyAny>) -> PyResult<Self> {
        let items: Vec<Bound<'_, PyAny>> = match index.cast::<PyTuple>() {
            Ok(t) => t.iter().collect(),
            Err(_) => vec![index.clone()],
        };
        let mut t = self.inner.clone();
        let mut dropped = Vec::new();
        for (dim, item) in items.iter().enumerate() {
            if dim >= t.ndim() {
                return Err(index_error(format!(
                    "too many indices for a {}-D tensor",
                    self.inner.ndim()
                )));
            }
            let size = t.shape[dim];
            if let Ok(slice) = item.cast::<PySlice>() {
                let ind = slice.indices(size as isize)?;
                t = if ind.step == 1 {
                    let start = ind.start.clamp(0, size as isize) as usize;
                    guard(|| t.narrow(dim as isize, start, ind.slicelength))?
                } else {
                    let picks: Vec<usize> = (0..ind.slicelength)
                        .map(|i| (ind.start + i as isize * ind.step) as usize)
                        .collect();
                    guard(|| t.index_select(dim as isize, &picks))?
                };
            } else if let Ok(i) = item.extract::<isize>() {
                let n = size as isize;
                let j = if i < 0 { i + n } else { i };
                if !(0..n).contains(&j) {
                    return Err(index_error(format!(
                        "index {i} is out of bounds for dimension {dim} with size {size}"
                    )));
                }
                t = guard(|| t.narrow(dim as isize, j as usize, 1))?;
                dropped.push(dim);
            } else {
                return Err(PyTypeError::new_err(
                    "tensor indices must be ints, slices or a tuple of them",
                ));
            }
        }
        if !dropped.is_empty() {
            let shape: Vec<usize> = t
                .shape
                .iter()
                .enumerate()
                .filter(|(d, _)| !dropped.contains(d))
                .map(|(_, &s)| s)
                .collect();
            t = guard(|| t.reshape(&shape))?;
        }
        Ok(t.into())
    }

    // ---- element-wise ------------------------------------------------------

    fn add(&self, other: Operand<'_>) -> PyResult<Self> {
        self.__add__(other)
    }

    fn sub(&self, other: Operand<'_>) -> PyResult<Self> {
        self.__sub__(other)
    }

    fn mul(&self, other: Operand<'_>) -> PyResult<Self> {
        self.__mul__(other)
    }

    fn div(&self, other: Operand<'_>) -> PyResult<Self> {
        self.__truediv__(other)
    }

    fn pow(&self, exponent: f64) -> PyResult<Self> {
        self.__pow__(exponent, None)
    }

    fn matmul(&self, other: PyRef<'_, PyTensor>) -> PyResult<Self> {
        self.matmul_impl(&other)
    }

    fn maximum(&self, other: Operand<'_>) -> PyResult<Self> {
        self.binary("maximum", &other, Tensor::maximum)
    }

    fn minimum(&self, other: Operand<'_>) -> PyResult<Self> {
        self.binary("minimum", &other, Tensor::minimum)
    }

    /// 1.0 where `self > other`, else 0.0.
    fn gt(&self, other: Operand<'_>) -> PyResult<Self> {
        self.__gt__(other)
    }

    fn lt(&self, other: Operand<'_>) -> PyResult<Self> {
        self.__lt__(other)
    }

    fn ge(&self, other: Operand<'_>) -> PyResult<Self> {
        self.__ge__(other)
    }

    fn le(&self, other: Operand<'_>) -> PyResult<Self> {
        self.__le__(other)
    }

    fn eq(&self, other: Operand<'_>) -> PyResult<Self> {
        self.binary("eq", &other, Tensor::eq_elem)
    }

    fn neg(&self) -> PyResult<Self> {
        self.map(Tensor::neg)
    }

    fn abs(&self) -> PyResult<Self> {
        self.map(Tensor::abs)
    }

    fn relu(&self) -> PyResult<Self> {
        self.map(Tensor::relu)
    }

    #[pyo3(signature = (negative_slope = 0.01))]
    fn leaky_relu(&self, negative_slope: f32) -> PyResult<Self> {
        self.map(|t| t.leaky_relu(negative_slope))
    }

    fn gelu(&self) -> PyResult<Self> {
        self.map(Tensor::gelu)
    }

    fn silu(&self) -> PyResult<Self> {
        self.map(Tensor::silu)
    }

    fn sigmoid(&self) -> PyResult<Self> {
        self.map(Tensor::sigmoid)
    }

    fn tanh(&self) -> PyResult<Self> {
        self.map(Tensor::tanh)
    }

    fn exp(&self) -> PyResult<Self> {
        self.map(Tensor::exp)
    }

    fn log(&self) -> PyResult<Self> {
        self.map(Tensor::log)
    }

    fn sqrt(&self) -> PyResult<Self> {
        self.map(Tensor::sqrt)
    }

    fn rsqrt(&self) -> PyResult<Self> {
        self.map(Tensor::rsqrt)
    }

    fn square(&self) -> PyResult<Self> {
        self.map(Tensor::square)
    }

    fn reciprocal(&self) -> PyResult<Self> {
        self.map(Tensor::recip)
    }

    fn sin(&self) -> PyResult<Self> {
        self.map(Tensor::sin)
    }

    fn cos(&self) -> PyResult<Self> {
        self.map(Tensor::cos)
    }

    fn sign(&self) -> PyResult<Self> {
        self.map(Tensor::sign)
    }

    #[pyo3(signature = (min = None, max = None))]
    fn clamp(&self, min: Option<f32>, max: Option<f32>) -> PyResult<Self> {
        let (lo, hi) = (
            min.unwrap_or(f32::NEG_INFINITY),
            max.unwrap_or(f32::INFINITY),
        );
        if lo > hi {
            return Err(value_error(format!(
                "clamp: min {lo} is greater than max {hi}"
            )));
        }
        self.map(|t| t.clamp(lo, hi))
    }

    #[pyo3(signature = (dim = -1))]
    fn softmax(&self, dim: isize) -> PyResult<Self> {
        normalize_dim(dim, self.inner.ndim())?;
        self.map(|t| t.softmax(dim))
    }

    #[pyo3(signature = (dim = -1))]
    fn log_softmax(&self, dim: isize) -> PyResult<Self> {
        normalize_dim(dim, self.inner.ndim())?;
        self.map(|t| t.log_softmax(dim))
    }

    /// Replace elements where `mask` is non-zero with the finite `value`.
    fn masked_fill(&self, mask: PyRef<'_, PyTensor>, value: f32) -> PyResult<Self> {
        if !value.is_finite() {
            return Err(value_error(
                "masked_fill needs a finite value; use a large one such as -1e9",
            ));
        }
        check_same_device(&self.inner, &mask.inner)?;
        broadcast("masked_fill", &self.inner.shape, &mask.inner.shape)?;
        self.map(|t| t.masked_fill(&mask.inner, value))
    }

    // ---- reductions --------------------------------------------------------

    #[pyo3(signature = (dim = None, keepdim = false))]
    fn sum(&self, dim: Option<&Bound<'_, PyAny>>, keepdim: bool) -> PyResult<Self> {
        self.reduce(dim, keepdim, Tensor::sum, Tensor::sum_dim)
    }

    #[pyo3(signature = (dim = None, keepdim = false))]
    fn mean(&self, dim: Option<&Bound<'_, PyAny>>, keepdim: bool) -> PyResult<Self> {
        self.reduce(dim, keepdim, Tensor::mean, Tensor::mean_dim)
    }

    /// Largest value overall or along `dim` (values only, no indices).
    #[pyo3(signature = (dim = None, keepdim = false))]
    fn max(&self, dim: Option<&Bound<'_, PyAny>>, keepdim: bool) -> PyResult<Self> {
        if self.inner.numel() == 0 {
            return Err(value_error("max() of an empty tensor"));
        }
        self.reduce(dim, keepdim, Tensor::max, Tensor::max_dim)
    }

    /// Smallest value overall or along `dim` (values only, no indices).
    #[pyo3(signature = (dim = None, keepdim = false))]
    fn min(&self, dim: Option<&Bound<'_, PyAny>>, keepdim: bool) -> PyResult<Self> {
        if self.inner.numel() == 0 {
            return Err(value_error("min() of an empty tensor"));
        }
        self.reduce(dim, keepdim, Tensor::min, Tensor::min_dim)
    }

    /// Population variance (divides by n) overall or along one `dim`.
    #[pyo3(signature = (dim = None, keepdim = false))]
    fn var(&self, dim: Option<isize>, keepdim: bool) -> PyResult<Self> {
        let t = &self.inner;
        match dim {
            Some(d) => {
                normalize_dim(d, t.ndim())?;
                self.map(|t| t.var_dim(d, keepdim))
            }
            None => self.map(|t| {
                let v = t.reshape(&[t.numel()]).var_dim(0, false);
                if keepdim {
                    v.reshape(&vec![1; t.ndim()])
                } else {
                    v
                }
            }),
        }
    }

    /// Index of the largest value (flattened), or indices along `dim`.
    #[pyo3(signature = (dim = None, keepdim = false))]
    fn argmax(&self, dim: Option<isize>, keepdim: bool) -> PyResult<Self> {
        self.arg_extreme(dim, keepdim, false)
    }

    /// Index of the smallest value (flattened), or indices along `dim`.
    #[pyo3(signature = (dim = None, keepdim = false))]
    fn argmin(&self, dim: Option<isize>, keepdim: bool) -> PyResult<Self> {
        self.arg_extreme(dim, keepdim, true)
    }

    // ---- shape -------------------------------------------------------------

    #[pyo3(signature = (*shape))]
    fn reshape(&self, shape: &Bound<'_, PyTuple>) -> PyResult<Self> {
        let shape = parse_view_shape(shape, self.inner.numel())?;
        self.map(|t| t.reshape(&shape))
    }

    #[pyo3(signature = (*shape))]
    fn view(&self, shape: &Bound<'_, PyTuple>) -> PyResult<Self> {
        self.reshape(shape)
    }

    #[pyo3(signature = (start_dim = 0, end_dim = -1))]
    fn flatten(&self, start_dim: isize, end_dim: isize) -> PyResult<Self> {
        let t = &self.inner;
        if t.ndim() == 0 {
            return self.map(|t| t.reshape(&[1]));
        }
        let (s, e) = (
            normalize_dim(start_dim, t.ndim())?,
            normalize_dim(end_dim, t.ndim())?,
        );
        if s > e {
            return Err(value_error("flatten: start_dim comes after end_dim"));
        }
        let shape = [
            &t.shape[..s],
            &[t.shape[s..=e].iter().product()],
            &t.shape[e + 1..],
        ]
        .concat();
        self.map(|t| t.reshape(&shape))
    }

    /// Remove size-1 dimensions (all, or just `dim` when it has size 1).
    #[pyo3(signature = (dim = None))]
    fn squeeze(&self, dim: Option<isize>) -> PyResult<Self> {
        match dim {
            None => self.map(Tensor::squeeze),
            Some(d) => {
                let n = normalize_dim(d, self.inner.ndim())?;
                if self.inner.shape[n] == 1 {
                    self.map(|t| t.squeeze_dim(d))
                } else {
                    Ok(self.clone_handle())
                }
            }
        }
    }

    fn unsqueeze(&self, dim: isize) -> PyResult<Self> {
        let d = normalize_dim(dim, self.inner.ndim() + 1)?;
        self.map(|t| t.unsqueeze(d))
    }

    #[pyo3(signature = (*dims))]
    fn permute(&self, dims: &Bound<'_, PyTuple>) -> PyResult<Self> {
        let ndim = self.inner.ndim();
        let raw: Vec<isize> = shape_items(dims)?
            .iter()
            .map(|x| x.extract())
            .collect::<PyResult<_>>()?;
        let perm = raw
            .iter()
            .map(|&d| normalize_dim(d, ndim))
            .collect::<PyResult<Vec<_>>>()?;
        let mut sorted = perm.clone();
        sorted.sort_unstable();
        if sorted != (0..ndim).collect::<Vec<_>>() {
            return Err(value_error(format!(
                "permute: {raw:?} is not a permutation of the {ndim} dimensions"
            )));
        }
        self.map(|t| t.permute(&perm))
    }

    fn transpose(&self, dim0: isize, dim1: isize) -> PyResult<Self> {
        let ndim = self.inner.ndim();
        normalize_dim(dim0, ndim)?;
        normalize_dim(dim1, ndim)?;
        self.map(|t| t.transpose_dims(dim0, dim1))
    }

    /// The tensor with its dimensions reversed.
    #[getter(T)]
    fn t_reversed(&self) -> PyResult<Self> {
        let perm: Vec<usize> = (0..self.inner.ndim()).rev().collect();
        self.map(|t| t.permute(&perm))
    }

    /// Broadcast to a larger shape; `-1` keeps a dimension's size.
    #[pyo3(signature = (*sizes))]
    fn expand(&self, sizes: &Bound<'_, PyTuple>) -> PyResult<Self> {
        let raw: Vec<i64> = shape_items(sizes)?
            .iter()
            .map(|x| x.extract())
            .collect::<PyResult<_>>()?;
        let t = &self.inner;
        if raw.len() < t.ndim() {
            return Err(value_error(format!(
                "expand: {raw:?} has fewer dimensions than shape {:?}",
                t.shape
            )));
        }
        let offset = raw.len() - t.ndim();
        let mut shape = Vec::with_capacity(raw.len());
        for (i, &r) in raw.iter().enumerate() {
            let current = i.checked_sub(offset).map(|j| t.shape[j]);
            let size = match (r, current) {
                (-1, Some(c)) => c,
                (r, _) if r < 0 => {
                    return Err(value_error(format!("expand: invalid size {r}")));
                }
                (r, Some(c)) if c != 1 && c != r as usize => {
                    return Err(value_error(format!(
                        "expand: cannot expand shape {:?} to {raw:?}",
                        t.shape
                    )));
                }
                (r, _) => r as usize,
            };
            shape.push(size);
        }
        self.map(|t| t.expand(&shape))
    }

    /// `length` entries of dimension `dim` starting at `start`.
    fn narrow(&self, dim: isize, start: usize, length: usize) -> PyResult<Self> {
        let d = normalize_dim(dim, self.inner.ndim())?;
        if start + length > self.inner.shape[d] {
            return Err(index_error(format!(
                "narrow: {start}..{} is out of range for dimension {d} of size {}",
                start + length,
                self.inner.shape[d]
            )));
        }
        self.map(|t| t.narrow(dim, start, length))
    }

    #[pyo3(signature = (chunks, dim = 0))]
    fn chunk(&self, chunks: usize, dim: isize) -> PyResult<Vec<Self>> {
        if chunks == 0 {
            return Err(value_error("chunk: chunks must be positive"));
        }
        normalize_dim(dim, self.inner.ndim())?;
        guard(|| self.inner.chunk(chunks, dim))
            .map(|parts| parts.into_iter().map(Into::into).collect())
    }

    #[pyo3(signature = (split_size, dim = 0))]
    fn split(&self, split_size: usize, dim: isize) -> PyResult<Vec<Self>> {
        if split_size == 0 {
            return Err(value_error("split: split_size must be positive"));
        }
        normalize_dim(dim, self.inner.ndim())?;
        guard(|| self.inner.split(split_size, dim))
            .map(|parts| parts.into_iter().map(Into::into).collect())
    }

    /// Entries of dimension `dim` at `index` (a list of ints or a tensor).
    fn index_select(&self, dim: isize, index: &Bound<'_, PyAny>) -> PyResult<Self> {
        let d = normalize_dim(dim, self.inner.ndim())?;
        let indices = parse_indices(index, self.inner.shape[d])?;
        self.map(|t| t.index_select(dim, &indices))
    }
}

impl PyTensor {
    fn clone_handle(&self) -> Self {
        self.inner.clone().into()
    }

    fn arg_extreme(&self, dim: Option<isize>, keepdim: bool, smallest: bool) -> PyResult<Self> {
        let t = &self.inner;
        if t.numel() == 0 {
            return Err(value_error("argmax/argmin of an empty tensor"));
        }
        match dim {
            None => guard(|| {
                let i = if smallest { t.argmin() } else { t.argmax() };
                Tensor::scalar(i as f32, t.device())
            }),
            Some(d) => {
                let n = normalize_dim(d, t.ndim())?;
                guard(|| {
                    let src = if smallest { t.neg() } else { t.clone() };
                    let out = src.argmax_dim(d);
                    if keepdim {
                        out.unsqueeze(n)
                    } else {
                        out
                    }
                })
            }
        }
        .map(Into::into)
    }
}

/// Integer indices below `size` from a list of ints or a tensor.
pub(crate) fn parse_indices(index: &Bound<'_, PyAny>, size: usize) -> PyResult<Vec<usize>> {
    let (values, _) = parse_data(index)?;
    values
        .iter()
        .map(|&v| {
            let i = v as i64;
            if i as f32 != v {
                return Err(value_error(format!("index {v} is not an integer")));
            }
            let j = if i < 0 { i + size as i64 } else { i };
            if (0..size as i64).contains(&j) {
                Ok(j as usize)
            } else {
                Err(index_error(format!(
                    "index {i} is out of bounds for size {size}"
                )))
            }
        })
        .collect()
}

// ============================================================================
// CREATION FUNCTIONS
// ============================================================================

/// Create a tensor from a number or (nested) list.
#[pyfunction]
#[pyo3(signature = (data, *, device = None, requires_grad = false))]
pub(crate) fn tensor(
    data: &Bound<'_, PyAny>,
    device: Option<&Bound<'_, PyAny>>,
    requires_grad: bool,
) -> PyResult<PyTensor> {
    PyTensor::new(data, device, requires_grad)
}

fn create(
    size: &Bound<'_, PyTuple>,
    device: Option<&Bound<'_, PyAny>>,
    requires_grad: bool,
    f: impl FnOnce(&[usize], &Device) -> Tensor,
) -> PyResult<PyTensor> {
    let shape = parse_shape(size)?;
    let device = resolve_device(device)?;
    Ok(with_grad(guard(|| f(&shape, &device))?, requires_grad))
}

#[pyfunction]
#[pyo3(signature = (*size, device = None, requires_grad = false))]
pub(crate) fn zeros(
    size: &Bound<'_, PyTuple>,
    device: Option<&Bound<'_, PyAny>>,
    requires_grad: bool,
) -> PyResult<PyTensor> {
    create(size, device, requires_grad, Tensor::zeros)
}

#[pyfunction]
#[pyo3(signature = (*size, device = None, requires_grad = false))]
pub(crate) fn ones(
    size: &Bound<'_, PyTuple>,
    device: Option<&Bound<'_, PyAny>>,
    requires_grad: bool,
) -> PyResult<PyTensor> {
    create(size, device, requires_grad, Tensor::ones)
}

/// Standard-normal samples.
#[pyfunction]
#[pyo3(signature = (*size, device = None, requires_grad = false))]
pub(crate) fn randn(
    size: &Bound<'_, PyTuple>,
    device: Option<&Bound<'_, PyAny>>,
    requires_grad: bool,
) -> PyResult<PyTensor> {
    create(size, device, requires_grad, Tensor::randn)
}

/// Uniform samples in `[0, 1)`.
#[pyfunction]
#[pyo3(signature = (*size, device = None, requires_grad = false))]
pub(crate) fn rand(
    size: &Bound<'_, PyTuple>,
    device: Option<&Bound<'_, PyAny>>,
    requires_grad: bool,
) -> PyResult<PyTensor> {
    create(size, device, requires_grad, Tensor::rand)
}

#[pyfunction]
#[pyo3(signature = (size, fill_value, *, device = None, requires_grad = false))]
pub(crate) fn full(
    size: &Bound<'_, PyAny>,
    fill_value: f32,
    device: Option<&Bound<'_, PyAny>>,
    requires_grad: bool,
) -> PyResult<PyTensor> {
    let size = PyTuple::new(size.py(), [size])?;
    create(&size, device, requires_grad, |s, d| {
        Tensor::full(s, fill_value, d)
    })
}

#[pyfunction]
#[pyo3(signature = (input, *, requires_grad = false))]
pub(crate) fn zeros_like(input: PyRef<'_, PyTensor>, requires_grad: bool) -> PyResult<PyTensor> {
    Ok(with_grad(
        guard(|| input.inner.zeros_like())?,
        requires_grad,
    ))
}

#[pyfunction]
#[pyo3(signature = (input, *, requires_grad = false))]
pub(crate) fn ones_like(input: PyRef<'_, PyTensor>, requires_grad: bool) -> PyResult<PyTensor> {
    Ok(with_grad(guard(|| input.inner.ones_like())?, requires_grad))
}

#[pyfunction]
#[pyo3(signature = (input, fill_value, *, requires_grad = false))]
pub(crate) fn full_like(
    input: PyRef<'_, PyTensor>,
    fill_value: f32,
    requires_grad: bool,
) -> PyResult<PyTensor> {
    Ok(with_grad(
        guard(|| input.inner.full_like(fill_value))?,
        requires_grad,
    ))
}

/// `[start, end)` in steps of `step`; `arange(n)` is `0, 1, ..., n - 1`.
#[pyfunction]
#[pyo3(signature = (start, end = None, step = 1.0, *, device = None, requires_grad = false))]
pub(crate) fn arange(
    start: f64,
    end: Option<f64>,
    step: f64,
    device: Option<&Bound<'_, PyAny>>,
    requires_grad: bool,
) -> PyResult<PyTensor> {
    let (start, end) = match end {
        Some(end) => (start, end),
        None => (0.0, start),
    };
    if step == 0.0 || !step.is_finite() {
        return Err(value_error("arange: step must be finite and non-zero"));
    }
    let n = ((end - start) / step).ceil().max(0.0) as usize;
    let values: Vec<f32> = (0..n).map(|i| (start + i as f64 * step) as f32).collect();
    let device = resolve_device(device)?;
    Ok(with_grad(
        guard(|| Tensor::from_vec(values, &[n], &device))?,
        requires_grad,
    ))
}

/// `steps` evenly spaced values from `start` to `end` inclusive.
#[pyfunction]
#[pyo3(signature = (start, end, steps, *, device = None, requires_grad = false))]
pub(crate) fn linspace(
    start: f64,
    end: f64,
    steps: usize,
    device: Option<&Bound<'_, PyAny>>,
    requires_grad: bool,
) -> PyResult<PyTensor> {
    let values: Vec<f32> = match steps {
        0 => vec![],
        1 => vec![start as f32],
        n => (0..n)
            .map(|i| (start + (end - start) * i as f64 / (n - 1) as f64) as f32)
            .collect(),
    };
    let device = resolve_device(device)?;
    Ok(with_grad(
        guard(|| Tensor::from_vec(values, &[steps], &device))?,
        requires_grad,
    ))
}

/// The `n x n` identity matrix.
#[pyfunction]
#[pyo3(signature = (n, *, device = None, requires_grad = false))]
pub(crate) fn eye(
    n: usize,
    device: Option<&Bound<'_, PyAny>>,
    requires_grad: bool,
) -> PyResult<PyTensor> {
    let device = resolve_device(device)?;
    Ok(with_grad(guard(|| Tensor::eye(n, &device))?, requires_grad))
}

fn collect_tensors(tensors: &Bound<'_, PyAny>) -> PyResult<Vec<Tensor>> {
    let list: Vec<Tensor> = tensors
        .try_iter()?
        .map(|t| Ok(t?.cast::<PyTensor>()?.borrow().inner.clone()))
        .collect::<PyResult<_>>()?;
    let Some(first) = list.first() else {
        return Err(value_error("expected a non-empty sequence of tensors"));
    };
    for t in &list {
        check_same_device(first, t)?;
    }
    Ok(list)
}

/// Join tensors along an existing dimension.
#[pyfunction]
#[pyo3(signature = (tensors, dim = 0))]
pub(crate) fn cat(tensors: &Bound<'_, PyAny>, dim: isize) -> PyResult<PyTensor> {
    let list = collect_tensors(tensors)?;
    let first = &list[0];
    let d = normalize_dim(dim, first.ndim())?;
    for t in &list {
        let compatible = t.ndim() == first.ndim()
            && t.shape
                .iter()
                .zip(&first.shape)
                .enumerate()
                .all(|(i, (a, b))| i == d || a == b);
        if !compatible {
            return Err(value_error(format!(
                "cat: shapes {:?} and {:?} differ outside dimension {d}",
                first.shape, t.shape
            )));
        }
    }
    guard(|| Tensor::cat(&list, dim)).map(Into::into)
}

/// Join same-shape tensors along a new dimension.
#[pyfunction]
#[pyo3(signature = (tensors, dim = 0))]
pub(crate) fn stack(tensors: &Bound<'_, PyAny>, dim: isize) -> PyResult<PyTensor> {
    let list = collect_tensors(tensors)?;
    let first = &list[0];
    if let Some(t) = list.iter().find(|t| t.shape != first.shape) {
        return Err(value_error(format!(
            "stack: all tensors need the same shape, got {:?} and {:?}",
            first.shape, t.shape
        )));
    }
    let d = normalize_dim(dim, first.ndim() + 1)?;
    guard(|| Tensor::stack(&list, d)).map(Into::into)
}

/// `input` where `condition` is non-zero, else `other` (broadcasting).
#[pyfunction]
#[pyo3(name = "where")]
pub(crate) fn where_(
    condition: PyRef<'_, PyTensor>,
    input: Operand<'_>,
    other: Operand<'_>,
) -> PyResult<PyTensor> {
    let cond = &condition.inner;
    let a = condition.operand("where", &input)?;
    let b = condition.operand("where", &other)?;
    broadcast("where", &a.shape, &b.shape)?;
    guard(|| Tensor::where_cond(cond, &a, &b)).map(Into::into)
}

// ============================================================================
// GRADIENT MODE
// ============================================================================

thread_local! {
    static NO_GRAD_GUARDS: RefCell<Vec<NoGradGuard>> = const { RefCell::new(Vec::new()) };
}

/// Whether ops are being recorded for autograd on this thread.
#[pyfunction(name = "is_grad_enabled")]
pub(crate) fn grad_enabled() -> bool {
    is_grad_enabled()
}

/// Start a `no_grad` block on this thread (used by `ferrotensor.no_grad`).
#[pyfunction]
pub(crate) fn _no_grad_enter() {
    NO_GRAD_GUARDS.with(|g| g.borrow_mut().push(NoGradGuard::new()));
}

/// End the innermost `no_grad` block on this thread.
#[pyfunction]
pub(crate) fn _no_grad_exit() {
    NO_GRAD_GUARDS.with(|g| drop(g.borrow_mut().pop()));
}
