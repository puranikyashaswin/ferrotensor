//! Shape arithmetic: broadcasting, strides and dimension indices.

/// NumPy-style broadcast of two shapes, or `None` when they are incompatible.
pub(crate) fn broadcast_shapes(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let ndim = a.len().max(b.len());
    let mut out = vec![0; ndim];
    for i in 0..ndim {
        let da = dim_from_right(a, ndim - 1 - i);
        let db = dim_from_right(b, ndim - 1 - i);
        out[i] = match (da, db) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            _ => return None,
        };
    }
    Some(out)
}

/// Size of the `i`-th dimension counted from the right (1 when absent).
fn dim_from_right(shape: &[usize], i: usize) -> usize {
    if i < shape.len() {
        shape[shape.len() - 1 - i]
    } else {
        1
    }
}

/// Row-major strides of a contiguous tensor.
pub(crate) fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Strides that read a contiguous `shape` as if it had `out_shape`:
/// broadcast (size-1 or missing leading) dimensions get stride 0.
pub(crate) fn broadcast_strides(shape: &[usize], out_shape: &[usize]) -> Vec<usize> {
    let own = contiguous_strides(shape);
    let offset = out_shape.len() - shape.len();
    (0..out_shape.len())
        .map(|i| {
            if i < offset || shape[i - offset] == 1 {
                0
            } else {
                own[i - offset]
            }
        })
        .collect()
}

/// Resolve a possibly negative dimension index against `ndim` dimensions.
///
/// # Panics
/// Panics when `dim` is out of range.
pub(crate) fn normalize_dim(dim: isize, ndim: usize) -> usize {
    let resolved = if dim < 0 { dim + ndim as isize } else { dim };
    assert!(
        resolved >= 0 && (resolved as usize) < ndim,
        "dimension {dim} out of range for a {ndim}-D tensor"
    );
    resolved as usize
}

/// Split `shape` around `dim` into `(outer, dim_size, inner)` element counts.
pub(crate) fn split_at_dim(shape: &[usize], dim: usize) -> (usize, usize, usize) {
    let outer = shape[..dim].iter().product();
    let inner = shape[dim + 1..].iter().product();
    (outer, shape[dim], inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcasts_like_numpy() {
        assert_eq!(broadcast_shapes(&[2, 3], &[3]), Some(vec![2, 3]));
        assert_eq!(broadcast_shapes(&[4, 1, 3], &[2, 1]), Some(vec![4, 2, 3]));
        assert_eq!(broadcast_shapes(&[], &[2, 2]), Some(vec![2, 2]));
        assert_eq!(broadcast_shapes(&[2, 3], &[2]), None);
    }

    #[test]
    fn strides() {
        assert_eq!(contiguous_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(contiguous_strides(&[]), Vec::<usize>::new());
        assert_eq!(broadcast_strides(&[3], &[2, 3]), vec![0, 1]);
        assert_eq!(broadcast_strides(&[2, 1], &[2, 5]), vec![1, 0]);
    }

    #[test]
    fn negative_dims() {
        assert_eq!(normalize_dim(-1, 3), 2);
        assert_eq!(normalize_dim(0, 3), 0);
        assert_eq!(split_at_dim(&[2, 3, 4], 1), (2, 3, 4));
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn rejects_bad_dim() {
        normalize_dim(3, 3);
    }
}
