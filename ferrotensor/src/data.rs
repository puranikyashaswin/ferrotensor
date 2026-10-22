//! Datasets and batched loading for the `ferrotensor` tensor API.
//!
//! A [`Dataset`] yields one `(input, target)` sample as flat `f32` data plus its
//! shape. A [`DataLoader`] groups samples into batches and stacks them into
//! [`Tensor`]s on a [`Device`], optionally shuffling the order each epoch, and
//! can prepare the next batches on a background thread
//! ([`DataLoader::epoch_prefetched`]).

use std::sync::mpsc::{sync_channel, Receiver};
use std::sync::Arc;
use std::thread::JoinHandle;

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

    /// Call `f` with the input and target at `index` (`index < len()`).
    ///
    /// [`DataLoader`] reads samples through this method. The default calls
    /// [`get`](Dataset::get); datasets that hold their samples override it
    /// to lend them without a copy, as [`TensorDataset`] does.
    fn with_sample(&self, index: usize, f: &mut dyn FnMut(&Sample, &Sample)) {
        let (input, target) = self.get(index);
        f(&input, &target);
    }

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

    fn with_sample(&self, index: usize, f: &mut dyn FnMut(&Sample, &Sample)) {
        f(&self.inputs[index], &self.targets[index]);
    }
}

/// Iterates a [`Dataset`] in batches of stacked tensors.
pub struct DataLoader<D: Dataset> {
    dataset: Arc<D>,
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
            dataset: Arc::new(dataset),
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

    /// The dataset the loader reads.
    pub fn dataset(&self) -> &D {
        &self.dataset
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
        let batch_size = self.batch_size;
        (0..batches).map(move |b| {
            let start = b * batch_size;
            let end = (start + batch_size).min(order.len());
            HostBatch::gather(&*self.dataset, &order[start..end]).upload(&self.device)
        })
    }
}

impl<D: Dataset + Send + Sync + 'static> DataLoader<D> {
    /// Like [`epoch`](Self::epoch), but a background thread gathers up to
    /// `depth` batches ahead while the caller works on the current one.
    ///
    /// The batches and their order are the same as `epoch(epoch)` gives.
    /// The thread only reads the dataset into host memory; the tensors are
    /// created (and uploaded to a GPU) on the calling thread. Dropping the
    /// iterator early stops the thread. A panic in the dataset is raised
    /// again on the calling thread.
    ///
    /// # Panics
    /// Panics unless `depth >= 1`.
    pub fn epoch_prefetched(&self, epoch: u64, depth: usize) -> Prefetch {
        assert!(depth > 0, "prefetch depth must be at least 1");
        let order = self.indices(epoch);
        let batches = self.num_batches();
        let batch_size = self.batch_size;
        let dataset = Arc::clone(&self.dataset);
        // `sync_channel(depth - 1)` plus the batch the worker holds while
        // blocked on `send` keeps `depth` batches ready.
        let (tx, rx) = sync_channel(depth - 1);
        let worker = std::thread::spawn(move || {
            for b in 0..batches {
                let start = b * batch_size;
                let end = (start + batch_size).min(order.len());
                let batch = HostBatch::gather(&*dataset, &order[start..end]);
                if tx.send(batch).is_err() {
                    return; // the iterator was dropped
                }
            }
        });
        Prefetch {
            rx: Some(rx),
            worker: Some(worker),
            device: self.device.clone(),
        }
    }
}

/// Batches gathered on a background thread; see
/// [`DataLoader::epoch_prefetched`].
pub struct Prefetch {
    rx: Option<Receiver<HostBatch>>,
    worker: Option<JoinHandle<()>>,
    device: Device,
}

impl Prefetch {
    /// Wait for the worker; re-raise its panic, if any.
    fn finish(&mut self) {
        self.rx = None;
        if let Some(worker) = self.worker.take() {
            if let Err(panic) = worker.join() {
                if !std::thread::panicking() {
                    std::panic::resume_unwind(panic);
                }
            }
        }
    }
}

impl Iterator for Prefetch {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        match self.rx.as_ref()?.recv() {
            Ok(batch) => Some(batch.upload(&self.device)),
            Err(_) => {
                self.finish();
                None
            }
        }
    }
}

impl Drop for Prefetch {
    fn drop(&mut self) {
        // Dropping the receiver makes the worker's next `send` fail.
        self.finish();
    }
}

impl std::fmt::Debug for Prefetch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Prefetch")
            .field("running", &self.worker.is_some())
            .finish()
    }
}

/// One batch stacked in host memory: inputs and targets with their shapes,
/// including the leading batch dim.
struct HostBatch {
    inputs: (Vec<f32>, Vec<usize>),
    targets: (Vec<f32>, Vec<usize>),
}

impl HostBatch {
    /// Stack the samples at `idx`, copying each sample's values once.
    fn gather<D: Dataset + ?Sized>(dataset: &D, idx: &[usize]) -> Self {
        let (mut xs, mut ys) = (Vec::new(), Vec::new());
        let (mut x_shape, mut y_shape) = (None, None);
        for &i in idx {
            dataset.with_sample(i, &mut |x, y| {
                if x_shape.is_none() {
                    xs.reserve_exact(idx.len() * x.data.len());
                    ys.reserve_exact(idx.len() * y.data.len());
                    x_shape = Some(x.shape.clone());
                    y_shape = Some(y.shape.clone());
                }
                assert_eq!(
                    x_shape.as_ref(),
                    Some(&x.shape),
                    "inputs in a batch must share a shape"
                );
                assert_eq!(
                    y_shape.as_ref(),
                    Some(&y.shape),
                    "targets in a batch must share a shape"
                );
                xs.extend_from_slice(&x.data);
                ys.extend_from_slice(&y.data);
            });
        }
        let with_batch = |s: Option<Vec<usize>>| [vec![idx.len()], s.unwrap_or_default()].concat();
        Self {
            inputs: (xs, with_batch(x_shape)),
            targets: (ys, with_batch(y_shape)),
        }
    }

    fn upload(self, device: &Device) -> (Tensor, Tensor) {
        let (xs, x_shape) = self.inputs;
        let (ys, y_shape) = self.targets;
        (
            Tensor::from_vec(xs, &x_shape, device),
            Tensor::from_vec(ys, &y_shape, device),
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

    /// The default seed's order is part of the public behaviour: runs must
    /// stay reproducible across releases.
    #[test]
    fn default_seed_order_is_pinned() {
        let loader = DataLoader::new(dataset(10), 3, true, Device::cpu());
        assert_eq!(loader.indices(0), vec![4, 7, 6, 5, 3, 9, 1, 8, 2, 0]);
        assert_eq!(loader.indices(1), vec![8, 2, 1, 6, 4, 7, 0, 5, 9, 3]);
    }

    #[test]
    fn unshuffled_order_is_identity() {
        assert_eq!(sample_order(5, false, 7, 3), vec![0, 1, 2, 3, 4]);
    }

    fn dataset(n: usize) -> TensorDataset {
        TensorDataset::new(
            (0..n)
                .map(|i| Sample::new(vec![i as f32, -(i as f32)], vec![2]))
                .collect(),
            (0..n)
                .map(|i| Sample::new(vec![i as f32], vec![]))
                .collect(),
        )
    }

    fn batches(
        it: impl Iterator<Item = (Tensor, Tensor)>,
    ) -> Vec<(Vec<usize>, Vec<f32>, Vec<f32>)> {
        it.map(|(x, y)| (x.shape.clone(), x.to_vec(), y.to_vec()))
            .collect()
    }

    #[test]
    fn batches_follow_the_sample_order() {
        let loader = DataLoader::new(dataset(10), 4, true, Device::cpu()).seed(3);
        let order = loader.indices(2);
        let got = batches(loader.epoch(2));
        assert_eq!(got.len(), 3);
        assert_eq!(got[2].0, vec![2, 2]); // the last, smaller batch
        let targets: Vec<f32> = got.iter().flat_map(|b| b.2.clone()).collect();
        assert_eq!(targets, order.iter().map(|&i| i as f32).collect::<Vec<_>>());
        for (shape, x, y) in &got {
            assert_eq!(shape[1], 2);
            for (row, t) in x.chunks(2).zip(y) {
                assert_eq!(row, [*t, -t]);
            }
        }
        // drop_last keeps the same order and skips the partial batch.
        let loader = loader.drop_last(true);
        assert_eq!(batches(loader.epoch(2)), got[..2]);
    }

    /// A dataset that can only be read through `get`, to check the default
    /// `with_sample`.
    struct Owned(TensorDataset);

    impl Dataset for Owned {
        fn len(&self) -> usize {
            self.0.len()
        }

        fn get(&self, index: usize) -> (Sample, Sample) {
            self.0.get(index)
        }
    }

    /// A dataset whose `get` must not be called.
    struct BorrowOnly(TensorDataset);

    impl Dataset for BorrowOnly {
        fn len(&self) -> usize {
            self.0.len()
        }

        fn get(&self, _: usize) -> (Sample, Sample) {
            panic!("DataLoader should read samples through with_sample");
        }

        fn with_sample(&self, index: usize, f: &mut dyn FnMut(&Sample, &Sample)) {
            self.0.with_sample(index, f);
        }
    }

    #[test]
    fn loader_borrows_samples_and_default_with_sample_matches() {
        let reference = batches(DataLoader::new(dataset(9), 2, true, Device::cpu()).epoch(1));
        let owned = DataLoader::new(Owned(dataset(9)), 2, true, Device::cpu());
        assert_eq!(batches(owned.epoch(1)), reference);
        let borrowed = DataLoader::new(BorrowOnly(dataset(9)), 2, true, Device::cpu());
        assert_eq!(batches(borrowed.epoch(1)), reference);
        assert_eq!(batches(borrowed.epoch_prefetched(1, 2)), reference);
    }

    #[test]
    fn prefetched_epochs_match_plain_epochs() {
        for (shuffle, drop_last) in [(false, false), (true, false), (true, true)] {
            let loader = DataLoader::new(dataset(23), 5, shuffle, Device::cpu())
                .seed(9)
                .drop_last(drop_last);
            for epoch in 0..3 {
                for depth in [1, 2, 8] {
                    assert_eq!(
                        batches(loader.epoch_prefetched(epoch, depth)),
                        batches(loader.epoch(epoch)),
                        "shuffle {shuffle} drop_last {drop_last} epoch {epoch} depth {depth}"
                    );
                }
            }
        }
    }

    #[test]
    fn dropping_a_prefetch_early_stops_the_worker() {
        let loader = DataLoader::new(dataset(100), 1, false, Device::cpu());
        let mut it = loader.epoch_prefetched(0, 2);
        assert_eq!(it.next().unwrap().1.to_vec(), vec![0.0]);
        drop(it); // joins the worker, which would otherwise block on send
        let empty = DataLoader::new(dataset(0), 4, false, Device::cpu());
        assert_eq!(empty.epoch_prefetched(0, 1).count(), 0);
    }

    #[test]
    #[should_panic(expected = "inputs in a batch must share a shape")]
    fn prefetch_reraises_dataset_panics() {
        let ds = TensorDataset::new(
            vec![
                Sample::new(vec![1.0], vec![1]),
                Sample::new(vec![1.0, 2.0], vec![2]),
            ],
            vec![Sample::new(vec![0.0], vec![]); 2],
        );
        DataLoader::new(ds, 2, false, Device::cpu())
            .epoch_prefetched(0, 1)
            .for_each(drop);
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
