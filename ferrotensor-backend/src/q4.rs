//! 4-bit affine weight format: codes `0..=15` in groups of [`GROUP`]
//! consecutive storage elements, each group with an f16 scale and minimum
//! (`w = q * scale + min`).
//!
//! # Storage format
//!
//! A logical `[k, n]` weight (`k` inputs, `n` outputs, row-major as
//! `Tensor::matmul` expects it) is stored in `[n, k]` order: element
//! `[i][j]` has storage index `s = j * k + i`, so a group runs along the
//! input dimension (PyTorch's `[out, in]` order, as in QLoRA, GPTQ and MLX).
//!
//! With `numel = k * n`, `groups = numel.div_ceil(GROUP)` and
//! `nib_words = groups * GROUP / 8`, the buffer is [`packed_words`]`(k, n)`
//! `u32` words:
//!
//! - words `0 .. nib_words`: the codes, 8 per word, element `s` in bits
//!   `4 * (s % 8) .. 4 * (s % 8) + 4` of word `s / 8` (lowest nibble first);
//!   the padding after the last real element has code 0;
//! - words `nib_words .. nib_words + groups`: one word per group, the f16
//!   bit pattern of its scale in the low half and of its minimum in the
//!   high half (so WGSL's `unpack2x16float` gives `(scale, min)`).
//!
//! Group `g` covers storage indices `64 g .. 64 g + 63`. When `k` is a
//! multiple of 64 a group never crosses an output column; otherwise it may,
//! which is valid but slightly less accurate.
//!
//! Quantising group `g`: with `mn`, `mx` the minimum and maximum of its real
//! elements, `scale = f16((mx - mn) / 15)`, `min = f16(mn)`, and each value
//! `v` gets `q = round((v - min) / scale)` clamped to `0..=15` (`q = 0` when
//! the scale is 0). Decoding is `w = q as f32 * scale + min`.
//!
//! # Round-trip error
//!
//! For every element `v` of a group with real minimum `mn` and maximum `mx`,
//! [`max_round_trip_error`]`(mn, mx)` bounds `|v - w|`. Rounding to the
//! nearest code costs at most half a step, `scale / 2 ≈ (mx - mn) / 30`; the
//! f16 rounding of the scale and the minimum (relative error at most
//! 2⁻¹¹ for normal f16 values, absolute at most 2⁻²⁵ for subnormal ones)
//! can push the largest values past the last code, which adds at most
//! `(mx - mn) · 2⁻¹¹ + |mn| · 2⁻¹¹ + 16 · 2⁻²⁵`; the f32 arithmetic adds a
//! few units in the last place of `max(|mn|, |mx|)`.

use std::fmt;

/// Weights that share one scale and minimum.
pub const GROUP: usize = 64;

/// Largest finite f16 value; group scales and minimums must fit in f16.
const F16_MAX: f32 = 65504.0;

/// Why [`quantize`] refused a weight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Q4Error {
    /// A value is NaN or infinite.
    NonFinite {
        /// Row of the logical `[k, n]` weight (input index).
        row: usize,
        /// Column of the logical `[k, n]` weight (output index).
        col: usize,
        /// The value.
        value: f32,
    },
    /// A group's minimum or scale does not fit in f16 (values beyond
    /// ±65504, or a range wider than 15 × 65504).
    OutOfF16Range {
        /// Index of the group.
        group: usize,
        /// The group's smallest value.
        min: f32,
        /// The group's largest value.
        max: f32,
    },
}

impl fmt::Display for Q4Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite { row, col, value } => write!(
                f,
                "cannot quantise to 4 bits: element [{row}, {col}] is {value}"
            ),
            Self::OutOfF16Range { group, min, max } => write!(
                f,
                "cannot quantise to 4 bits: group {group} spans {min} ..= {max}, \
                 whose minimum or scale does not fit in f16"
            ),
        }
    }
}

impl std::error::Error for Q4Error {}

/// Number of groups of a `[k, n]` weight.
fn groups(k: usize, n: usize) -> usize {
    (k * n).div_ceil(GROUP)
}

/// Number of code words of a `[k, n]` weight (the scale words follow them).
fn nib_words(k: usize, n: usize) -> usize {
    groups(k, n) * GROUP / 8
}

/// Number of `u32` words [`quantize`] returns for a `[k, n]` weight (at least 1).
pub fn packed_words(k: usize, n: usize) -> usize {
    (nib_words(k, n) + groups(k, n)).max(1)
}

/// Bound on `|v - dequantize(quantize(v))|` for any element of a group whose
/// real values span `mn ..= mx` (see the [module docs](self)).
pub fn max_round_trip_error(mn: f32, mx: f32) -> f32 {
    let range = mx - mn;
    range / 30.0 + range / 1024.0 + mn.abs().max(mx.abs()) / 1024.0 + 1e-6
}

/// Checks the shape and returns `k * n`.
fn numel(k: usize, n: usize) -> usize {
    let numel = k.checked_mul(n).expect("q4: k * n overflows");
    assert!(
        numel < 1 << 31,
        "q4: {k} x {n} weights are too many for the shader's u32 indices"
    );
    numel
}

/// Quantise row-major `data` of logical shape `[k, n]` to the format in the
/// [module docs](self). Fails on NaN or infinite values and on groups whose
/// minimum or scale does not fit in f16.
pub fn quantize(data: &[f32], k: usize, n: usize) -> Result<Vec<u32>, Q4Error> {
    let numel = numel(k, n);
    assert_eq!(data.len(), numel, "q4::quantize: data is not {k} x {n}");
    if let Some(i) = data.iter().position(|v| !v.is_finite()) {
        return Err(Q4Error::NonFinite {
            row: i / n,
            col: i % n,
            value: data[i],
        });
    }
    let mut words = vec![0u32; packed_words(k, n)];
    if numel == 0 {
        return Ok(words);
    }
    let nib = nib_words(k, n);
    // Storage index `s` holds logical element [s % k][s / k].
    let at = |s: usize| data[(s % k) * n + s / k];
    for g in 0..groups(k, n) {
        let range = g * GROUP..((g + 1) * GROUP).min(numel);
        let (mut mn, mut mx) = (f32::INFINITY, f32::NEG_INFINITY);
        for s in range.clone() {
            mn = mn.min(at(s));
            mx = mx.max(at(s));
        }
        let s16 = half::f16::from_f32((mx - mn) / 15.0);
        let m16 = half::f16::from_f32(mn);
        if !s16.is_finite() || !m16.is_finite() || mn.abs().max(mx.abs()) > F16_MAX {
            return Err(Q4Error::OutOfF16Range {
                group: g,
                min: mn,
                max: mx,
            });
        }
        let (scale, min) = (s16.to_f32(), m16.to_f32());
        for s in range {
            let q = if scale > 0.0 {
                ((at(s) - min) / scale).round().clamp(0.0, 15.0) as u32
            } else {
                0
            };
            words[s / 8] |= q << (4 * (s % 8));
        }
        words[nib + g] = u32::from(s16.to_bits()) | (u32::from(m16.to_bits()) << 16);
    }
    Ok(words)
}

/// Decode the output of [`quantize`] back to row-major `[k, n]` f32.
pub fn dequantize(words: &[u32], k: usize, n: usize) -> Vec<f32> {
    let numel = numel(k, n);
    assert_eq!(
        words.len(),
        packed_words(k, n),
        "q4::dequantize: wrong number of words for {k} x {n}"
    );
    let nib = nib_words(k, n);
    let mut out = vec![0.0f32; numel];
    for s in 0..numel {
        let q = (words[s / 8] >> (4 * (s % 8))) & 15;
        let sm = words[nib + s / GROUP];
        let scale = half::f16::from_bits(sm as u16).to_f32();
        let min = half::f16::from_bits((sm >> 16) as u16).to_f32();
        out[(s % k) * n + s / k] = (q as f32) * scale + min;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    /// Deterministic pseudo-random values in [-1, 1) (as `tests/gpu_kernels.rs`).
    fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// Printed past the test harness's capture, so CI logs show it.
    fn report(line: &str) {
        let _ = writeln!(std::io::stderr(), "{line}");
    }

    fn round_trip(data: &[f32], k: usize, n: usize) -> Vec<f32> {
        dequantize(&quantize(data, k, n).unwrap(), k, n)
    }

    /// Measured error of one round trip.
    #[derive(Default)]
    struct Errors {
        /// Largest `|v - w| / (step / 2)` over all elements.
        max_over_half_step: f32,
        /// Sum over elements of `((v - w) / step)²`, for groups with a step.
        sum_sq_over_step: f64,
        stepped: usize,
        max_abs: f32,
        sum_sq: f64,
        sum_sq_w: f64,
    }

    /// Checks every element of `data` against [`max_round_trip_error`] of its
    /// own group (recomputed from the data and the storage order) and
    /// returns the measured errors.
    #[track_caller]
    fn check_bound(name: &str, data: &[f32], k: usize, n: usize) -> Errors {
        let words = quantize(data, k, n).unwrap();
        let got = dequantize(&words, k, n);
        let numel = k * n;
        let nib = nib_words(k, n);
        let mut e = Errors::default();
        for g in 0..groups(k, n) {
            let idx: Vec<usize> = (g * GROUP..((g + 1) * GROUP).min(numel))
                .map(|s| (s % k) * n + s / k)
                .collect();
            let mn = idx.iter().map(|&i| data[i]).fold(f32::INFINITY, f32::min);
            let mx = idx
                .iter()
                .map(|&i| data[i])
                .fold(f32::NEG_INFINITY, f32::max);
            let bound = max_round_trip_error(mn, mx);
            let step = half::f16::from_bits(words[nib + g] as u16).to_f32();
            for &i in &idx {
                let err = (data[i] - got[i]).abs();
                assert!(
                    err <= bound,
                    "{name}: element {i} = {} decoded as {} (error {err}, bound {bound}, group {g} spans {mn} ..= {mx})",
                    data[i],
                    got[i]
                );
                e.max_abs = e.max_abs.max(err);
                e.sum_sq += f64::from(err).powi(2);
                e.sum_sq_w += f64::from(data[i]).powi(2);
                if step > 0.0 {
                    e.max_over_half_step = e.max_over_half_step.max(err / (step / 2.0));
                    e.sum_sq_over_step += f64::from(err / step).powi(2);
                    e.stepped += 1;
                }
            }
        }
        e
    }

    #[test]
    fn layout_is_exact() {
        let data: Vec<f32> = (0..16).map(|v| v as f32).collect();
        assert_eq!(packed_words(16, 1), 9);
        let words = quantize(&data, 16, 1).unwrap();
        assert_eq!(words.len(), 9);
        assert_eq!(words[0], 0x7654_3210);
        assert_eq!(words[1], 0xFEDC_BA98);
        assert!(words[2..8].iter().all(|&w| w == 0), "padding codes are 0");
        // Scale 1.0 (f16 0x3C00) in the low half, minimum 0.0 in the high half.
        assert_eq!(words[8], 0x0000_3C00);
        assert_eq!(dequantize(&words, 16, 1), data);
    }

    #[test]
    fn storage_order_groups_along_k() {
        // Column 0 is all 0 and column 1 all 100: grouped along k, each
        // group is one constant column and decodes exactly; grouped along n
        // by mistake, every group would mix 0 and 100.
        let (k, n) = (64, 2);
        let data: Vec<f32> = (0..k * n)
            .map(|i| if i % n == 0 { 0.0 } else { 100.0 })
            .collect();
        assert_eq!(round_trip(&data, k, n), data);
    }

    #[test]
    fn round_trip_error_is_bounded() {
        let mut total = Errors::default();
        for (seed, &(k, n)) in [(1, 1), (3, 5), (64, 1), (33, 65), (128, 96)]
            .iter()
            .enumerate()
        {
            let data: Vec<f32> = rand_vec(k * n, 700 + seed as u64)
                .iter()
                .map(|v| v * 0.1)
                .collect();
            let e = check_bound(&format!("random {k}x{n}"), &data, k, n);
            total.max_over_half_step = total.max_over_half_step.max(e.max_over_half_step);
            total.sum_sq_over_step += e.sum_sq_over_step;
            total.stepped += e.stepped;
            total.max_abs = total.max_abs.max(e.max_abs);
            total.sum_sq += e.sum_sq;
            total.sum_sq_w += e.sum_sq_w;
        }
        let rms_over_step = (total.sum_sq_over_step / total.stepped as f64).sqrt();
        let rms_over_weight_rms = (total.sum_sq / total.sum_sq_w).sqrt();
        report(&format!(
            "q4 round trip, uniform weights in [-0.1, 0.1): max |error| {:.3e} = {:.4} x half a step; \
             RMS error = {:.4} x step (uniform rounding: 1/sqrt(12) = 0.2887); \
             RMS error / RMS weight = {:.4}",
            total.max_abs, total.max_over_half_step, rms_over_step, rms_over_weight_rms
        ));
        // Rounding to the nearest code: at most half a step (plus f16
        // rounding of the scale and minimum), and for spread-out values an
        // RMS error near step / sqrt(12).
        assert!(
            total.max_over_half_step <= 1.01,
            "{}",
            total.max_over_half_step
        );
        assert!(
            (0.2..0.35).contains(&rms_over_step),
            "RMS error is {rms_over_step} steps"
        );
    }

    #[test]
    fn edge_groups_stay_within_the_bound() {
        // One group each: constant, all zero, a single outlier, negative
        // only, and (k = 70) a group that crosses an output column.
        let constant = vec![0.37f32; 64];
        let rt = round_trip(&constant, 64, 1);
        check_bound("constant", &constant, 64, 1);
        assert!(rt.iter().all(|&w| w == half::f16::from_f32(0.37).to_f32()));

        let zeros = vec![0.0f32; 64];
        assert_eq!(round_trip(&zeros, 64, 1), zeros);
        assert_eq!(quantize(&zeros, 64, 1).unwrap(), vec![0; 9]);

        let mut outlier: Vec<f32> = rand_vec(64, 900).iter().map(|v| v * 0.01).collect();
        outlier[17] = 2.0;
        let e = check_bound("outlier", &outlier, 64, 1);
        report(&format!(
            "q4 round trip, 63 values in [-0.01, 0.01) and one 2.0: max |error| {:.3e} \
             (the outlier stretches the step to {:.3e})",
            e.max_abs,
            (2.0 - outlier.iter().cloned().fold(f32::INFINITY, f32::min)) / 15.0
        ));
        assert!((round_trip(&outlier, 64, 1)[17] - 2.0).abs() <= max_round_trip_error(-0.01, 2.0));

        let negative: Vec<f32> = rand_vec(64, 901).iter().map(|v| v - 2.0).collect();
        check_bound("negative", &negative, 64, 1);

        let crossing: Vec<f32> = rand_vec(70 * 3, 902).iter().map(|v| v * 0.1).collect();
        check_bound("70x3", &crossing, 70, 3);
    }

    #[test]
    fn group_minimum_and_maximum_decode_exactly() {
        // min -1 and max 0.875 give scale 0.125: both are exact in f16, so
        // the ends of the range decode exactly and so do values on the grid.
        let mut data: Vec<f32> = (0..64).map(|i| -1.0 + 0.125 * (i % 16) as f32).collect();
        data[5] = -1.0;
        data[40] = 0.875;
        assert_eq!(round_trip(&data, 64, 1), data);
        let words = quantize(&data, 64, 1).unwrap();
        let sm = words[8];
        assert_eq!(half::f16::from_bits(sm as u16).to_f32(), 0.125);
        assert_eq!(half::f16::from_bits((sm >> 16) as u16).to_f32(), -1.0);
        assert_eq!((words[0] >> 20) & 15, 0, "the minimum has code 0");
        assert_eq!(words[5] & 15, 15, "the maximum has code 15");
    }

    #[test]
    fn sizes_off_the_group_are_padded_with_zero_codes() {
        for &(k, n) in &[(1, 1), (3, 5), (33, 65), (5, 13)] {
            let numel = k * n;
            let data: Vec<f32> = rand_vec(numel, 903).iter().map(|v| v + 0.5).collect();
            let words = quantize(&data, k, n).unwrap();
            assert_eq!(words.len(), packed_words(k, n));
            assert_eq!(packed_words(k, n), numel.div_ceil(64) * 9);
            for s in numel..nib_words(k, n) * 8 {
                assert_eq!(
                    (words[s / 8] >> (4 * (s % 8))) & 15,
                    0,
                    "{k}x{n}: padding code {s}"
                );
            }
            check_bound(&format!("{k}x{n}"), &data, k, n);
        }
        assert_eq!(packed_words(0, 7), 1);
        assert_eq!(quantize(&[], 0, 7).unwrap(), vec![0]);
        assert!(dequantize(&[0], 0, 7).is_empty());
    }

    #[test]
    fn non_finite_and_out_of_range_values_are_rejected() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut data = vec![0.5f32; 6];
            data[4] = bad;
            let err = quantize(&data, 2, 3).unwrap_err();
            assert!(
                matches!(err, Q4Error::NonFinite { row: 1, col: 1, .. }),
                "{err:?}"
            );
            assert!(err.to_string().contains("[1, 1]"), "{err}");
        }
        let mut data = vec![0.0f32; 128];
        data[100] = 1.0e6;
        assert!(matches!(
            quantize(&data, 128, 1),
            Err(Q4Error::OutOfF16Range { group: 1, .. })
        ));
        data[100] = -7.0e4;
        assert!(matches!(
            quantize(&data, 128, 1),
            Err(Q4Error::OutOfF16Range { group: 1, .. })
        ));
    }
}
