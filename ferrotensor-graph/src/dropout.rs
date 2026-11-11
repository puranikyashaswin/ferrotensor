//! Seeds for attention dropout.
//!
//! A dropout mask is a pure function of its seed: the attention kernels hash
//! `(batch * heads + head, query, key + seed)`, and the backward pass (and any
//! recomputation of a forward, such as activation checkpointing) regenerates
//! the same mask from the same seed. Training therefore derives one seed per
//! step from the run seed and the step counter
//! ([`derive_seed`]`(&[run_seed, ..step])`), and each attention layer derives
//! its own from that ([`layer_seed`]). Nothing here keeps state, so a resumed
//! run that restores its step counter draws the same masks as an
//! uninterrupted one.
//!
//! The kernels add the seed to the key index, so two seeds that differ by a
//! small amount give masks that are shifted copies of each other. Every seed
//! is therefore passed through a full 64-bit mix before use.

/// SplitMix64 finaliser: a bijective mix of all 64 bits.
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A well-mixed seed from an ordered list of integers, e.g.
/// `[run_seed, rank, optimizer_step, micro_batch]`. Equal lists give equal
/// seeds; changing any element changes all bits of the result.
pub fn derive_seed(parts: &[u64]) -> u64 {
    parts
        .iter()
        .fold(mix(parts.len() as u64), |acc, &p| mix(acc ^ mix(p)))
}

/// The dropout seed of attention layer `layer` (its index in the model's
/// layer list) for one step whose seed is `step_seed`.
pub fn layer_seed(step_seed: u64, layer: usize) -> u64 {
    derive_seed(&[step_seed, layer as u64])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeds_are_deterministic() {
        assert_eq!(derive_seed(&[42, 0, 7, 1]), derive_seed(&[42, 0, 7, 1]));
        assert_eq!(layer_seed(9, 3), layer_seed(9, 3));
    }

    #[test]
    fn seeds_differ_by_step_layer_and_run() {
        let base = derive_seed(&[42, 0, 7, 0]);
        let others = [
            derive_seed(&[43, 0, 7, 0]),
            derive_seed(&[42, 1, 7, 0]),
            derive_seed(&[42, 0, 8, 0]),
            derive_seed(&[42, 0, 7, 1]),
            derive_seed(&[42, 0, 7]),
        ];
        for o in others {
            assert_ne!(base, o);
        }
        assert_ne!(layer_seed(base, 0), layer_seed(base, 1));
    }

    #[test]
    fn consecutive_steps_are_not_small_offsets() {
        // The kernels use the low 32 bits as a (wrapping) offset on the key
        // index; a small difference would make two masks shifted copies.
        let far = |a: u32, b: u32| a.wrapping_sub(b).min(b.wrapping_sub(a)) > 1 << 12;
        for step in 0..64u64 {
            let a = derive_seed(&[0, 0, step, 0]) as u32;
            let b = derive_seed(&[0, 0, step + 1, 0]) as u32;
            assert!(far(a, b), "step {step}: {a} vs {b}");
            for layer in 0..8 {
                let x = layer_seed(a as u64, layer) as u32;
                let y = layer_seed(a as u64, layer + 1) as u32;
                assert!(far(x, y), "layer {layer}: {x} vs {y}");
            }
        }
    }
}
