//! `manual_seed` reproducibility.
//!
//! The seeded generator is process-wide, so this lives in its own test binary:
//! inside the library's unit tests, other tests running in parallel draw from
//! it between the seed and the draws.

use ferrotensor::{functional, manual_seed, Device, Tensor};

#[test]
fn manual_seed_makes_every_source_reproducible() {
    let d = Device::cpu();
    let draw = || {
        let r = Tensor::randn(&[8], &d).to_vec();
        let u = Tensor::uniform(&[8], -1.0, 1.0, &d).to_vec();
        let m = functional::dropout(&Tensor::ones(&[64], &d), 0.5, true).to_vec();
        (r, u, m)
    };
    manual_seed(7);
    let first = draw();
    manual_seed(7);
    assert_eq!(first, draw());
    manual_seed(8);
    assert_ne!(first.0, draw().0);
}
