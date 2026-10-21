//! Datasets and batched loading for the `rusty` tensor API.
//!
//! A [`Dataset`] yields one `(input, target)` sample as flat `f32` data plus its
//! shape. A [`DataLoader`] groups samples into batches and stacks them into
//! [`Tensor`]s on a [`Device`], optionally shuffling the order each epoch.

use crate::tensor::Tensor;
use crate::Device;

/// One sample: flat values and the shape they describe.
#[derive(Debug, Clone, PartialEq)]
pub struct Sample {
    /// Row-major values.
    pub data: Vec<f32>,
    /// Shape of `data` (product must equal `data.len()`).
    pub shape: Vec<usize>,
}

impl Sample {
    /// Build a sample, checking that `shape` matches `data`.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        assert_eq!(
            shape.iter().product::<usize>(),
            data.len(),
            "sample shape {shape:?} does not match {} values",
            data.len()
        );
        Self { data, shape }
    }
}

/// A finite, indexable collection of `(input, target)` samples.
pub trait Dataset {
    /// Number of samples.
    fn len(&self) -> usize;

    /// The sample at `index` (`index < len()`).
    fn get(&self, index: usize) -> (Sample, Sample);

    /// True when the dataset has no samples.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// In-memory dataset backed by two vectors of samples.
#[derive(Debug, Clone, Default)]
pub struct TensorDataset {
    inputs: Vec<Sample>,
    targets: Vec<Sample>,
}

impl TensorDataset {
    /// Pair inputs with targets; both must have the same length.
    pub fn new(inputs: Vec<Sample>, targets: Vec<Sample>) -> Self {
        assert_eq!(
            inputs.len(),
            targets.len(),
            "inputs and targets differ in length"
        );
        Self { inputs, targets }
    }
}

impl Dataset for TensorDataset {
    fn len(&self) -> usize {
        self.inputs.len()
    }

    fn get(&self, index: usize) -> (Sample, Sample) {
        (self.inputs[index].clone(), self.targets[index].clone())
    }
}

/// Iterates a [`Dataset`] in batches of stacked tensors.
pub struct DataLoader<D: Dataset> {
    dataset: D,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
    seed: u64,
    device: Device,
}

impl<D: Dataset> DataLoader<D> {
    /// Create a loader. `batch_size` must be at least 1.
    pub fn new(dataset: D, batch_size: usize, shuffle: bool, device: Device) -> Self {
        assert!(batch_size > 0, "batch_size must be at least 1");
        Self {
            dataset,
            batch_size,
            shuffle,
            drop_last: false,
            seed: 0x5eed,
            device,
        }
    }

    /// Skip the final batch when it is smaller than `batch_size`.
    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    /// Seed for the shuffle order (deterministic across runs).
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Number of batches one epoch yields.
    pub fn num_batches(&self) -> usize {
        batch_count(self.dataset.len(), self.batch_size, self.drop_last)
    }

    /// Sample order for `epoch`: identity, or a seeded Fisher-Yates shuffle.
    pub fn indices(&self, epoch: u64) -> Vec<usize> {
        sample_order(self.dataset.len(), self.shuffle, self.seed, epoch)
    }

    /// Batches for `epoch` as `(inputs, targets)` with a leading batch dimension.
    pub fn epoch(&self, epoch: u64) -> impl Iterator<Item = (Tensor, Tensor)> + '_ {
        let order = self.indices(epoch);
        let batches = self.num_batches();
        (0..batches).map(move |b| {
            let start = b * self.batch_size;
            let end = (start + self.batch_size).min(order.len());
            self.stack(&order[start..end])
        })
    }

    fn stack(&self, idx: &[usize]) -> (Tensor, Tensor) {
        let (mut xs, mut ys) = (Vec::new(), Vec::new());
        let (mut x_shape, mut y_shape) = (Vec::new(), Vec::new());
        for &i in idx {
            let (x, y) = self.dataset.get(i);
            if x_shape.is_empty() {
                x_shape = x.shape.clone();
                y_shape = y.shape.clone();
            }
            assert_eq!(x.shape, x_shape, "inputs in a batch must share a shape");
            assert_eq!(y.shape, y_shape, "targets in a batch must share a shape");
            xs.extend(x.data);
            ys.extend(y.data);
        }
        let with_batch = |s: &[usize]| [&[idx.len()][..], s].concat();
        (
            Tensor::from_data(&xs, &with_batch(&x_shape), &self.device),
            Tensor::from_data(&ys, &with_batch(&y_shape), &self.device),
        )
    }
}

fn batch_count(n: usize, batch_size: usize, drop_last: bool) -> usize {
    if drop_last {
        n / batch_size
    } else {
        n.div_ceil(batch_size)
    }
}

fn sample_order(n: usize, shuffle: bool, seed: u64, epoch: u64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    if shuffle {
        let mut state = (seed ^ epoch.wrapping_mul(0x9e37_79b9_7f4a_7c15)) | 1;
        for i in (1..n).rev() {
            // xorshift64*
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let r = state.wrapping_mul(0x2545_f491_4f6c_dd1d);
            order.swap(i, (r % (i as u64 + 1)) as usize);
        }
    }
    order
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_count_respects_drop_last() {
        assert_eq!(batch_count(10, 3, false), 4);
        assert_eq!(batch_count(10, 3, true), 3);
        assert_eq!(batch_count(0, 3, false), 0);
    }

    #[test]
    fn shuffle_is_a_seeded_permutation() {
        let mut a = sample_order(50, true, 7, 0);
        assert_ne!(a, (0..50).collect::<Vec<_>>());
        assert_eq!(a, sample_order(50, true, 7, 0));
        assert_ne!(a, sample_order(50, true, 7, 1));
        a.sort_unstable();
        assert_eq!(a, (0..50).collect::<Vec<_>>());
    }

    #[test]
    fn unshuffled_order_is_identity() {
        assert_eq!(sample_order(5, false, 7, 3), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn tensor_dataset_returns_pairs() {
        let ds = TensorDataset::new(
            vec![Sample::new(vec![1.0, 2.0], vec![2])],
            vec![Sample::new(vec![3.0], vec![1])],
        );
        assert_eq!(ds.len(), 1);
        assert_eq!(ds.get(0).1.data, vec![3.0]);
    }
}
