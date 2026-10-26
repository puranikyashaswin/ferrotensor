//! The process-wide random number generator, seedable for reproducible runs.
//!
//! [`Tensor::randn`](crate::Tensor::randn), [`Tensor::rand`](crate::Tensor::rand),
//! [`Tensor::uniform`](crate::Tensor::uniform), weight initialisation in [`nn`](crate::nn)
//! and [`functional::dropout`](crate::functional::dropout) all draw from it. Until
//! [`manual_seed`] is called it is the thread-local OS-seeded generator; after it,
//! every draw comes from one seeded generator, so the same program produces the
//! same numbers on every run.

use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use std::sync::{Mutex, MutexGuard};

static SEEDED: Mutex<Option<StdRng>> = Mutex::new(None);

fn seeded() -> MutexGuard<'static, Option<StdRng>> {
    // A panic while holding the lock leaves the generator usable.
    SEEDED
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Seed the generator behind every random tensor, initialiser and dropout mask.
///
/// Like `torch.manual_seed`: call it once at the start of a program. Calling it
/// again with the same seed restarts the same sequence.
///
/// ```
/// use ferrotensor::{manual_seed, Device, Tensor};
///
/// let device = Device::cpu();
/// manual_seed(42);
/// let a = Tensor::randn(&[4], &device).to_vec();
/// manual_seed(42);
/// let b = Tensor::randn(&[4], &device).to_vec();
/// assert_eq!(a, b);
/// ```
pub fn manual_seed(seed: u64) {
    *seeded() = Some(StdRng::seed_from_u64(seed));
}

/// Run `f` with the seeded generator if [`manual_seed`] was called, else with
/// the thread-local OS-seeded one.
pub(crate) fn with_rng<T>(f: impl FnOnce(&mut dyn RngCore) -> T) -> T {
    let mut guard = seeded();
    match guard.as_mut() {
        Some(rng) => f(rng),
        None => {
            drop(guard);
            f(&mut rand::thread_rng())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Device, Tensor};

    // One test, so parallel tests can't interleave draws from the shared generator.
    #[test]
    fn manual_seed_makes_every_source_reproducible() {
        let d = Device::cpu();
        let draw = || {
            let r = Tensor::randn(&[8], &d).to_vec();
            let u = Tensor::uniform(&[8], -1.0, 1.0, &d).to_vec();
            let m = crate::functional::dropout(&Tensor::ones(&[64], &d), 0.5, true).to_vec();
            (r, u, m)
        };
        manual_seed(7);
        let first = draw();
        manual_seed(7);
        assert_eq!(first, draw());
        manual_seed(8);
        assert_ne!(first.0, draw().0);
    }
}
