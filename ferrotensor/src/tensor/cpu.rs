//! CPU reference kernels.
//!
//! Each function mirrors one WGSL entry point in `backend::kernels` and is the
//! ground truth the GPU results are tested against.

use crate::backend::{binary_op, reduce_op, unary_op};

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn gelu_tanh(x: f32) -> f32 {
    const C: f32 = 0.797_884_6;
    0.5 * x * (1.0 + (C * (x + 0.044715 * x * x * x)).tanh())
}

fn gelu_tanh_grad(x: f32) -> f32 {
    const C: f32 = 0.797_884_6;
    let t = (C * (x + 0.044715 * x * x * x)).tanh();
    let dt = (1.0 - t * t) * C * (1.0 + 3.0 * 0.044715 * x * x);
    0.5 * (1.0 + t) + 0.5 * x * dt
}

/// One element of the `unary` kernel.
pub(crate) fn unary_scalar(op: u32, x: f32, a: f32, b: f32) -> f32 {
    match op {
        unary_op::RELU => x.max(0.0),
        unary_op::SILU => x * sigmoid(x),
        unary_op::GELU => gelu_tanh(x),
        unary_op::SIGMOID => sigmoid(x),
        unary_op::TANH => x.tanh(),
        unary_op::EXP => x.exp(),
        unary_op::LOG => x.ln(),
        unary_op::ABS => x.abs(),
        unary_op::SQRT => x.sqrt(),
        unary_op::NEG => -x,
        unary_op::ADD_SCALAR => x + a,
        unary_op::MUL_SCALAR => x * a,
        unary_op::POW_SCALAR => x.powf(a),
        unary_op::CLAMP => x.clamp(a, b),
        unary_op::LEAKY_RELU => {
            if x > 0.0 {
                x
            } else {
                a * x
            }
        }
        unary_op::STEP => {
            if x > 0.0 {
                1.0
            } else {
                0.0
            }
        }
        unary_op::SIGN => {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        }
        unary_op::RECIP => 1.0 / x,
        unary_op::SILU_GRAD => {
            let s = sigmoid(x);
            s * (1.0 + x * (1.0 - s))
        }
        unary_op::GELU_GRAD => gelu_tanh_grad(x),
        unary_op::RSUB_SCALAR => a - x,
        unary_op::SQUARE => x * x,
        unary_op::IN_RANGE => {
            if x >= a && x <= b {
                1.0
            } else {
                0.0
            }
        }
        unary_op::SIN => x.sin(),
        unary_op::COS => x.cos(),
        unary_op::LEAKY_RELU_GRAD => {
            if x > 0.0 {
                1.0
            } else {
                a
            }
        }
        _ => x,
    }
}

/// The `unary` kernel.
pub(crate) fn unary(op: u32, x: &[f32], a: f32, b: f32) -> Vec<f32> {
    x.iter().map(|&v| unary_scalar(op, v, a, b)).collect()
}

/// One element of the `binary` kernel.
pub(crate) fn binary_scalar(op: u32, x: f32, y: f32) -> f32 {
    let flag = |c: bool| if c { 1.0 } else { 0.0 };
    match op {
        binary_op::ADD => x + y,
        binary_op::SUB => x - y,
        binary_op::MUL => x * y,
        binary_op::DIV => x / y,
        binary_op::MAX => x.max(y),
        binary_op::MIN => x.min(y),
        binary_op::GT => flag(x > y),
        binary_op::LT => flag(x < y),
        binary_op::EQ => flag(x == y),
        binary_op::GE => flag(x >= y),
        binary_op::LE => flag(x <= y),
        _ => x,
    }
}

/// The broadcasting `binary` kernel.
pub(crate) fn binary(
    op: u32,
    a: &[f32],
    a_strides: &[usize],
    b: &[f32],
    b_strides: &[usize],
    out_shape: &[usize],
) -> Vec<f32> {
    let n: usize = out_shape.iter().product();
    let ndim = out_shape.len();
    let mut out = Vec::with_capacity(n);
    let mut index = vec![0usize; ndim];
    let (mut ia, mut ib) = (0usize, 0usize);
    for _ in 0..n {
        out.push(binary_scalar(op, a[ia], b[ib]));
        // Odometer increment of the multi-index, updating both offsets.
        for d in (0..ndim).rev() {
            index[d] += 1;
            ia += a_strides[d];
            ib += b_strides[d];
            if index[d] < out_shape[d] {
                break;
            }
            ia -= a_strides[d] * out_shape[d];
            ib -= b_strides[d] * out_shape[d];
            index[d] = 0;
        }
    }
    out
}

/// The axis `reduce` kernel over a `[outer, dim, inner]` view.
pub(crate) fn reduce(op: u32, x: &[f32], outer: usize, dim: usize, inner: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(outer * inner);
    for o in 0..outer {
        for k in 0..inner {
            let base = o * dim * inner + k;
            let mut acc = x[base];
            for j in 1..dim {
                let v = x[base + j * inner];
                acc = match op {
                    reduce_op::MAX => acc.max(v),
                    reduce_op::MIN => acc.min(v),
                    _ => acc + v,
                };
            }
            out.push(acc);
        }
    }
    out
}

/// The `softmax` kernel (or log-softmax) over a `[outer, dim, inner]` view.
pub(crate) fn softmax(x: &[f32], outer: usize, dim: usize, inner: usize, log: bool) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    for o in 0..outer {
        for k in 0..inner {
            let base = o * dim * inner + k;
            let idx = |j: usize| base + j * inner;
            let m = (0..dim)
                .map(|j| x[idx(j)])
                .fold(f32::NEG_INFINITY, f32::max);
            let s: f32 = (0..dim).map(|j| (x[idx(j)] - m).exp()).sum();
            let log_s = s.ln();
            for j in 0..dim {
                let z = x[idx(j)] - m;
                out[idx(j)] = if log { z - log_s } else { z.exp() / s };
            }
        }
    }
    out
}

/// The batched `matmul` kernel: `batch` products of `[m, k] @ [k, n]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn matmul(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    batch: usize,
    a_batched: bool,
    b_batched: bool,
) -> Vec<f32> {
    let mut out = vec![0.0; batch * m * n];
    for bi in 0..batch {
        let a_off = if a_batched { bi * m * k } else { 0 };
        let b_off = if b_batched { bi * k * n } else { 0 };
        let o_off = bi * m * n;
        for i in 0..m {
            let row = &mut out[o_off + i * n..o_off + (i + 1) * n];
            for p in 0..k {
                let av = a[a_off + i * k + p];
                let b_row = &b[b_off + p * n..b_off + (p + 1) * n];
                for (o, &bv) in row.iter_mut().zip(b_row) {
                    *o += av * bv;
                }
            }
        }
    }
    out
}

/// The `permute` (strided copy) kernel: `src_strides[d]` is the input stride
/// of output dim `d`, and reading starts at `offset`.
pub(crate) fn permute(
    x: &[f32],
    out_shape: &[usize],
    src_strides: &[usize],
    offset: usize,
) -> Vec<f32> {
    let n: usize = out_shape.iter().product();
    let ndim = out_shape.len();
    let mut out = Vec::with_capacity(n);
    let mut index = vec![0usize; ndim];
    let mut src = offset;
    for _ in 0..n {
        out.push(x[src]);
        for d in (0..ndim).rev() {
            index[d] += 1;
            src += src_strides[d];
            if index[d] < out_shape[d] {
                break;
            }
            src -= src_strides[d] * out_shape[d];
            index[d] = 0;
        }
    }
    out
}

/// The `gather_rows` kernel: rows of a `[_, row]` table picked by `idx`.
pub(crate) fn gather_rows(table: &[f32], idx: &[u32], row: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(idx.len() * row);
    for &i in idx {
        let start = i as usize * row;
        out.extend_from_slice(&table[start..start + row]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_broadcasts_rows() {
        // [2, 3] + [3]
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [10.0, 20.0, 30.0];
        let out = binary(binary_op::ADD, &a, &[3, 1], &b, &[0, 1], &[2, 3]);
        assert_eq!(out, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn matmul_small() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        assert_eq!(
            matmul(&a, &b, 2, 2, 2, 1, true, true),
            vec![19.0, 22.0, 43.0, 50.0]
        );
    }

    #[test]
    fn permute_transposes() {
        // [2, 3] -> [3, 2]
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(
            permute(&x, &[3, 2], &[1, 3], 0),
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
    }

    #[test]
    fn softmax_rows_sum_to_one() {
        let x = [1.0, 2.0, 3.0, -1.0, 0.0, 1.0];
        let s = softmax(&x, 2, 3, 1, false);
        assert!((s[..3].iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!((s[3..].iter().sum::<f32>() - 1.0).abs() < 1e-6);
        let ls = softmax(&x, 2, 3, 1, true);
        assert!((ls[0].exp() - s[0]).abs() < 1e-6);
    }

    #[test]
    fn reduce_over_middle_axis() {
        // [2, 2, 2] summed over dim 1.
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert_eq!(
            reduce(reduce_op::SUM, &x, 2, 2, 2),
            vec![4.0, 6.0, 12.0, 14.0]
        );
        assert_eq!(
            reduce(reduce_op::MAX, &x, 2, 2, 2),
            vec![3.0, 4.0, 7.0, 8.0]
        );
    }
}
