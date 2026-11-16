//! 4-bit affine weight format: codes `0..=15` in groups of `G`
//! ([`Q4Group`]: 16, 32, 64 or 128; 64 unless chosen) consecutive storage
//! elements, each group with an f16 scale and minimum (`w = q * scale + min`).
//!
//! # Storage format
//!
//! A logical `[k, n]` weight (`k` inputs, `n` outputs, row-major as
//! `Tensor::matmul` expects it) is stored in `[n, k]` order: element
//! `[i][j]` has storage index `s = j * k + i`, so a group runs along the
//! input dimension (PyTorch's `[out, in]` order, as in QLoRA, GPTQ and MLX).
//!
//! With `numel = k * n`, `groups = numel.div_ceil(G)` and
//! `nib_words = groups * G / 8`, the buffer is [`packed_words`]`(k, n, G)`
//! `u32` words:
//!
//! - words `0 .. nib_words`: the codes, 8 per word, element `s` in bits
//!   `4 * (s % 8) .. 4 * (s % 8) + 4` of word `s / 8` (lowest nibble first);
//!   the padding after the last real element has code 0;
//! - words `nib_words .. nib_words + groups`: one word per group, the f16
//!   bit pattern of its scale in the low half and of its minimum in the
//!   high half (so WGSL's `unpack2x16float` gives `(scale, min)`).
//!
//! Group `g` covers storage indices `G g .. G g + G - 1`. When `k` is a
//! multiple of `G` a group never crosses an output column; otherwise it may,
//! which is valid but slightly less accurate. The buffer does not record
//! `G`: the tensor carries it, and the matmul passes it to the kernel.
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
//! few units in the last place of `max(|mn|, |mx|)`. A smaller group holds
//! a subset of the values of the larger group around it, so its bound is
//! never larger.

use std::fmt;

/// Number of consecutive storage elements that share one scale and minimum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Q4Group(usize);

impl Q4Group {
    /// The sizes the matmul kernel decodes: powers of two, so it divides by
    /// shifting, and multiples of 8, so a group fills whole code words.
    pub const SIZES: [usize; 4] = [16, 32, 64, 128];

    /// 64 weights per group, the size used unless another is chosen.
    pub const DEFAULT: Self = Self(64);

    /// # Errors
    ///
    /// [`Q4Error::UnsupportedGroup`] unless `size` is one of [`Self::SIZES`].
    pub fn new(size: usize) -> Result<Self, Q4Error> {
        if Self::SIZES.contains(&size) {
            Ok(Self(size))
        } else {
            Err(Q4Error::UnsupportedGroup { size })
        }
    }

    /// Weights per group.
    pub fn size(self) -> usize {
        self.0
    }
}

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
    /// The shape is not a 2-D `[k, n]` matrix.
    NotMatrix {
        /// Number of dimensions of the shape.
        rank: usize,
    },
    /// A group size the kernel does not decode (see [`Q4Group::SIZES`]).
    UnsupportedGroup {
        /// The requested size.
        size: usize,
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
            Self::NotMatrix { rank } => write!(
                f,
                "cannot quantise to 4 bits: expected a 2-D shape, got {rank} dimensions"
            ),
            Self::UnsupportedGroup { size } => write!(
                f,
                "4-bit group size {size} is not supported; use one of {:?}",
                Q4Group::SIZES
            ),
        }
    }
}

impl std::error::Error for Q4Error {}

/// Number of groups of a `[k, n]` weight.
fn groups(k: usize, n: usize, group: Q4Group) -> usize {
    (k * n).div_ceil(group.0)
}

/// Number of code words of a `[k, n]` weight (the scale words follow them).
fn nib_words(k: usize, n: usize, group: Q4Group) -> usize {
    groups(k, n, group) * group.0 / 8
}

/// Number of `u32` words [`quantize`] returns for a `[k, n]` weight (at least 1).
pub fn packed_words(k: usize, n: usize, group: Q4Group) -> usize {
    (nib_words(k, n, group) + groups(k, n, group)).max(1)
}

/// Bound on `|v - dequantize(quantize(v))|` for any element of a group whose
/// real values span `mn ..= mx` (see the [module docs](self)).
pub fn max_round_trip_error(mn: f32, mx: f32) -> f32 {
    let range = mx - mn;
    range / 30.0 + range / 1024.0 + mn.abs().max(mx.abs()) / 1024.0 + 1e-6
}

/// Checks the shape and returns `k * n`.
#[expect(
    clippy::expect_used,
    reason = "shape precondition: k * n fits in usize"
)]
fn numel(k: usize, n: usize) -> usize {
    let numel = k.checked_mul(n).expect("q4: k * n overflows");
    assert!(
        numel < 1 << 31,
        "q4: {k} x {n} weights are too many for the shader's u32 indices"
    );
    numel
}

/// Smallest and largest of `values`.
fn min_max(values: impl Iterator<Item = f32>) -> (f32, f32) {
    values.fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), v| {
        (mn.min(v), mx.max(v))
    })
}

/// The f16 scale and minimum of group `g`, whose real values span `mn ..= mx`.
fn group_scale(g: usize, mn: f32, mx: f32) -> Result<(half::f16, half::f16), Q4Error> {
    let s16 = half::f16::from_f32((mx - mn) / 15.0);
    let m16 = half::f16::from_f32(mn);
    if !s16.is_finite() || !m16.is_finite() || mn.abs().max(mx.abs()) > F16_MAX {
        return Err(Q4Error::OutOfF16Range {
            group: g,
            min: mn,
            max: mx,
        });
    }
    Ok((s16, m16))
}

/// The code of `v` in a group whose decoded scale and minimum are `scale`
/// and `min`.
fn code(v: f32, scale: f32, min: f32) -> u32 {
    if scale > 0.0 {
        ((v - min) / scale).round().clamp(0.0, 15.0) as u32
    } else {
        0
    }
}

/// Quantise row-major `data` of logical shape `[k, n]` to the format in the
/// [module docs](self), `group` weights per scale. Fails on NaN or infinite
/// values and on groups whose minimum or scale does not fit in f16.
pub fn quantize(data: &[f32], k: usize, n: usize, group: Q4Group) -> Result<Vec<u32>, Q4Error> {
    let numel = numel(k, n);
    assert_eq!(data.len(), numel, "q4::quantize: data is not {k} x {n}");
    if let Some(i) = data.iter().position(|v| !v.is_finite()) {
        return Err(Q4Error::NonFinite {
            row: i / n,
            col: i % n,
            value: data[i],
        });
    }
    let mut words = vec![0u32; packed_words(k, n, group)];
    if numel == 0 {
        return Ok(words);
    }
    let nib = nib_words(k, n, group);
    // Storage index `s` holds logical element [s % k][s / k].
    let at = |s: usize| data[(s % k) * n + s / k];
    for g in 0..groups(k, n, group) {
        let range = g * group.0..((g + 1) * group.0).min(numel);
        let (mn, mx) = min_max(range.clone().map(at));
        let (s16, m16) = group_scale(g, mn, mx)?;
        let (scale, min) = (s16.to_f32(), m16.to_f32());
        for s in range {
            words[s / 8] |= code(at(s), scale, min) << (4 * (s % 8));
        }
        words[nib + g] = u32::from(s16.to_bits()) | (u32::from(m16.to_bits()) << 16);
    }
    Ok(words)
}

/// Decode the output of [`quantize`] back to row-major `[k, n]` f32.
pub fn dequantize(words: &[u32], k: usize, n: usize, group: Q4Group) -> Vec<f32> {
    let numel = numel(k, n);
    assert_eq!(
        words.len(),
        packed_words(k, n, group),
        "q4::dequantize: wrong number of words for {k} x {n}"
    );
    let nib = nib_words(k, n, group);
    let mut out = vec![0.0f32; numel];
    for s in 0..numel {
        let q = (words[s / 8] >> (4 * (s % 8))) & 15;
        let sm = words[nib + s / group.0];
        let scale = half::f16::from_bits(sm as u16).to_f32();
        let min = half::f16::from_bits((sm >> 16) as u16).to_f32();
        out[(s % k) * n + s / k] = (q as f32) * scale + min;
    }
    out
}

/// How far the 4-bit round trip moves a weight ([`round_trip_error`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoundTripError {
    /// `‖w − ŵ‖ / ‖w‖` over the whole weight (Frobenius norms; 0 when `w`
    /// is all zero), with `ŵ = dequantize(quantize(w))`.
    pub relative: f64,
    /// Largest `|w − ŵ|`.
    pub max_abs: f32,
    /// Number of groups.
    pub groups: usize,
    /// Groups whose range at least halves when the one value farthest from
    /// the group mean is left out: a single outlier sets most of their step.
    pub outlier_groups: usize,
}

/// Measures [`quantize`] followed by [`dequantize`] on a weight given in
/// storage order (a PyTorch `[out, in]` weight read row by row, `k` inputs
/// per row), one group at a time and without building the packed buffer.
/// Fails where [`quantize`] fails.
pub fn round_trip_error(
    storage: &[f32],
    k: usize,
    group: Q4Group,
) -> Result<RoundTripError, Q4Error> {
    if let Some(s) = storage.iter().position(|v| !v.is_finite()) {
        return Err(Q4Error::NonFinite {
            row: s % k,
            col: s / k,
            value: storage[s],
        });
    }
    let (mut sum_sq_err, mut sum_sq) = (0.0f64, 0.0f64);
    let mut out = RoundTripError {
        relative: 0.0,
        max_abs: 0.0,
        groups: 0,
        outlier_groups: 0,
    };
    for (g, values) in storage.chunks(group.0).enumerate() {
        let (mn, mx) = min_max(values.iter().copied());
        let (s16, m16) = group_scale(g, mn, mx)?;
        let (scale, min) = (s16.to_f32(), m16.to_f32());
        for &v in values {
            let err = v - (code(v, scale, min) as f32 * scale + min);
            out.max_abs = out.max_abs.max(err.abs());
            sum_sq_err += f64::from(err).powi(2);
            sum_sq += f64::from(v).powi(2);
        }
        let mean = values.iter().map(|&v| f64::from(v)).sum::<f64>() / values.len() as f64;
        let distance = |v: f32| (f64::from(v) - mean).abs();
        let farthest = (0..values.len())
            .max_by(|&a, &b| distance(values[a]).total_cmp(&distance(values[b])))
            .unwrap_or(0);
        let (rest_mn, rest_mx) = min_max(
            values
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != farthest)
                .map(|(_, &v)| v),
        );
        if mx > mn && (rest_mx - rest_mn).max(0.0) * 2.0 <= mx - mn {
            out.outlier_groups += 1;
        }
        out.groups += 1;
    }
    if sum_sq > 0.0 {
        out.relative = (sum_sq_err / sum_sq).sqrt();
    }
    Ok(out)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::io::Write as _;

    const D: Q4Group = Q4Group::DEFAULT;

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

    /// Deterministic standard normal values (Box-Muller on [`rand_vec`]).
    fn gaussian_vec(n: usize, seed: u64) -> Vec<f32> {
        let u = rand_vec(2 * n, seed);
        u.chunks_exact(2)
            .map(|p| {
                let r = (-2.0 * (1.0 - (p[0] + 1.0) / 2.0).ln()).sqrt();
                r * (std::f32::consts::PI * p[1]).cos()
            })
            .collect()
    }

    /// Printed past the test harness's capture, so CI logs show it.
    fn report(line: &str) {
        let _ = writeln!(std::io::stderr(), "{line}");
    }

    fn round_trip(data: &[f32], k: usize, n: usize, group: Q4Group) -> Vec<f32> {
        dequantize(&quantize(data, k, n, group).unwrap(), k, n, group)
    }

    /// Row-major `[k, n]` `data` in storage order (`[n, k]`).
    fn storage_order(data: &[f32], k: usize, n: usize) -> Vec<f32> {
        (0..k * n).map(|s| data[(s % k) * n + s / k]).collect()
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

    /// The real minimum and maximum of every group of a `[k, n]` weight,
    /// recomputed from the data and the storage order.
    fn group_ranges(data: &[f32], k: usize, n: usize, group: Q4Group) -> Vec<(f32, f32)> {
        storage_order(data, k, n)
            .chunks(group.0)
            .map(|c| min_max(c.iter().copied()))
            .collect()
    }

    /// Checks every element of `data` against [`max_round_trip_error`] of its
    /// own group and returns the measured errors.
    #[track_caller]
    fn check_bound(name: &str, data: &[f32], k: usize, n: usize, group: Q4Group) -> Errors {
        let words = quantize(data, k, n, group).unwrap();
        let got = dequantize(&words, k, n, group);
        let nib = nib_words(k, n, group);
        let mut e = Errors::default();
        for (g, (mn, mx)) in group_ranges(data, k, n, group).into_iter().enumerate() {
            let bound = max_round_trip_error(mn, mx);
            let step = half::f16::from_bits(words[nib + g] as u16).to_f32();
            for s in g * group.0..((g + 1) * group.0).min(k * n) {
                let i = (s % k) * n + s / k;
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
        assert_eq!(packed_words(16, 1, D), 9);
        let words = quantize(&data, 16, 1, D).unwrap();
        assert_eq!(words.len(), 9);
        assert_eq!(words[0], 0x7654_3210);
        assert_eq!(words[1], 0xFEDC_BA98);
        assert!(words[2..8].iter().all(|&w| w == 0), "padding codes are 0");
        // Scale 1.0 (f16 0x3C00) in the low half, minimum 0.0 in the high half.
        assert_eq!(words[8], 0x0000_3C00);
        assert_eq!(dequantize(&words, 16, 1, D), data);
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
        assert_eq!(round_trip(&data, k, n, D), data);
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
            let e = check_bound(&format!("random {k}x{n}"), &data, k, n, D);
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
        let rt = round_trip(&constant, 64, 1, D);
        check_bound("constant", &constant, 64, 1, D);
        assert!(rt.iter().all(|&w| w == half::f16::from_f32(0.37).to_f32()));

        let zeros = vec![0.0f32; 64];
        assert_eq!(round_trip(&zeros, 64, 1, D), zeros);
        assert_eq!(quantize(&zeros, 64, 1, D).unwrap(), vec![0; 9]);

        let mut outlier: Vec<f32> = rand_vec(64, 900).iter().map(|v| v * 0.01).collect();
        outlier[17] = 2.0;
        let e = check_bound("outlier", &outlier, 64, 1, D);
        report(&format!(
            "q4 round trip, 63 values in [-0.01, 0.01) and one 2.0: max |error| {:.3e} \
             (the outlier stretches the step to {:.3e})",
            e.max_abs,
            (2.0 - outlier.iter().cloned().fold(f32::INFINITY, f32::min)) / 15.0
        ));
        assert!(
            (round_trip(&outlier, 64, 1, D)[17] - 2.0).abs() <= max_round_trip_error(-0.01, 2.0)
        );

        let negative: Vec<f32> = rand_vec(64, 901).iter().map(|v| v - 2.0).collect();
        check_bound("negative", &negative, 64, 1, D);

        let crossing: Vec<f32> = rand_vec(70 * 3, 902).iter().map(|v| v * 0.1).collect();
        check_bound("70x3", &crossing, 70, 3, D);
    }

    #[test]
    fn group_minimum_and_maximum_decode_exactly() {
        // min -1 and max 0.875 give scale 0.125: both are exact in f16, so
        // the ends of the range decode exactly and so do values on the grid.
        let mut data: Vec<f32> = (0..64).map(|i| -1.0 + 0.125 * (i % 16) as f32).collect();
        data[5] = -1.0;
        data[40] = 0.875;
        assert_eq!(round_trip(&data, 64, 1, D), data);
        let words = quantize(&data, 64, 1, D).unwrap();
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
            let words = quantize(&data, k, n, D).unwrap();
            assert_eq!(words.len(), packed_words(k, n, D));
            assert_eq!(packed_words(k, n, D), numel.div_ceil(64) * 9);
            for s in numel..nib_words(k, n, D) * 8 {
                assert_eq!(
                    (words[s / 8] >> (4 * (s % 8))) & 15,
                    0,
                    "{k}x{n}: padding code {s}"
                );
            }
            check_bound(&format!("{k}x{n}"), &data, k, n, D);
        }
        assert_eq!(packed_words(0, 7, D), 1);
        assert_eq!(quantize(&[], 0, 7, D).unwrap(), vec![0]);
        assert!(dequantize(&[0], 0, 7, D).is_empty());
    }

    #[test]
    fn non_finite_and_out_of_range_values_are_rejected() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut data = vec![0.5f32; 6];
            data[4] = bad;
            let err = quantize(&data, 2, 3, D).unwrap_err();
            assert!(
                matches!(err, Q4Error::NonFinite { row: 1, col: 1, .. }),
                "{err:?}"
            );
            assert!(err.to_string().contains("[1, 1]"), "{err}");
        }
        let mut data = vec![0.0f32; 128];
        data[100] = 1.0e6;
        assert!(matches!(
            quantize(&data, 128, 1, D),
            Err(Q4Error::OutOfF16Range { group: 1, .. })
        ));
        data[100] = -7.0e4;
        assert!(matches!(
            quantize(&data, 128, 1, D),
            Err(Q4Error::OutOfF16Range { group: 1, .. })
        ));
    }

    #[test]
    fn only_the_kernel_group_sizes_are_accepted() {
        assert_eq!(Q4Group::new(64).unwrap(), Q4Group::DEFAULT);
        for size in Q4Group::SIZES {
            assert_eq!(Q4Group::new(size).unwrap().size(), size);
        }
        for size in [0, 8, 48, 96, 256] {
            let err = Q4Group::new(size).unwrap_err();
            assert_eq!(err, Q4Error::UnsupportedGroup { size });
            assert_eq!(
                err.to_string(),
                format!("4-bit group size {size} is not supported; use one of [16, 32, 64, 128]")
            );
        }
    }

    #[test]
    fn every_group_size_round_trips_within_its_bound() {
        // Codes then one scale word per group: 4 + 32 / G bits per weight.
        for size in Q4Group::SIZES {
            let group = Q4Group::new(size).unwrap();
            assert_eq!(packed_words(256, 256, group), 65536 / 8 + 65536 / size);
            for (seed, &(k, n)) in [(1, 1), (3, 5), (16, 2), (33, 65), (128, 96)]
                .iter()
                .enumerate()
            {
                let data: Vec<f32> = rand_vec(k * n, 1100 + seed as u64)
                    .iter()
                    .map(|v| v * 0.1)
                    .collect();
                let words = quantize(&data, k, n, group).unwrap();
                assert_eq!(words.len(), (k * n).div_ceil(size) * (size / 8 + 1));
                check_bound(&format!("G={size} {k}x{n}"), &data, k, n, group);
            }
        }
        // 16 values 0..=15 fill one group of 16 exactly; with G = 32 the
        // same codes leave 16 padding codes before the scale word.
        let data: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let g16 = quantize(&data, 16, 1, Q4Group::new(16).unwrap()).unwrap();
        assert_eq!(g16, vec![0x7654_3210, 0xFEDC_BA98, 0x0000_3C00]);
        let g32 = quantize(&data, 16, 1, Q4Group::new(32).unwrap()).unwrap();
        assert_eq!(g32, vec![0x7654_3210, 0xFEDC_BA98, 0, 0, 0x0000_3C00]);
    }

    #[test]
    fn smaller_groups_never_give_a_larger_error() {
        // Gaussian weights with a few large values, as trained weights have.
        let (k, n) = (256, 256);
        let mut data: Vec<f32> = gaussian_vec(k * n, 1200).iter().map(|v| v * 0.02).collect();
        for v in data.iter_mut().step_by(997) {
            *v *= 8.0;
        }
        let mut relative = Vec::new();
        for (i, &size) in Q4Group::SIZES.iter().enumerate() {
            let group = Q4Group::new(size).unwrap();
            let e = check_bound(&format!("G={size}"), &data, k, n, group);
            relative.push((e.sum_sq / e.sum_sq_w).sqrt());
            // Each element's bound in this group size is at most its bound
            // in the next larger size, whose group contains this one.
            if let Some(&larger) = Q4Group::SIZES.get(i + 1) {
                let small = group_ranges(&data, k, n, group);
                let large = group_ranges(&data, k, n, Q4Group::new(larger).unwrap());
                for (g, &(mn, mx)) in small.iter().enumerate() {
                    let (lmn, lmx) = large[g * size / larger];
                    assert!(lmn <= mn && mx <= lmx, "G={size} group {g}");
                    assert!(max_round_trip_error(mn, mx) <= max_round_trip_error(lmn, lmx));
                }
            }
        }
        report(&format!(
            "q4 round trip, Gaussian weights (sd 0.02, every 997th x8): relative error \
             G=16 {:.4}, G=32 {:.4}, G=64 {:.4}, G=128 {:.4}",
            relative[0], relative[1], relative[2], relative[3]
        ));
        for pair in relative.windows(2) {
            assert!(pair[0] < pair[1], "{relative:?}");
        }
    }

    #[test]
    fn host_error_report_matches_the_round_trip() {
        for size in Q4Group::SIZES {
            let group = Q4Group::new(size).unwrap();
            for (seed, &(k, n)) in [(1, 1), (33, 65), (64, 2), (128, 96)].iter().enumerate() {
                let data: Vec<f32> = rand_vec(k * n, 1300 + seed as u64)
                    .iter()
                    .map(|v| v * 0.1)
                    .collect();
                let got = round_trip(&data, k, n, group);
                let (mut max_abs, mut sum_sq_err, mut sum_sq) = (0f32, 0f64, 0f64);
                for (v, w) in storage_order(&data, k, n)
                    .iter()
                    .zip(storage_order(&got, k, n))
                {
                    max_abs = max_abs.max((v - w).abs());
                    sum_sq_err += f64::from(v - w).powi(2);
                    sum_sq += f64::from(*v).powi(2);
                }
                let e = round_trip_error(&storage_order(&data, k, n), k, group).unwrap();
                assert_eq!(e.max_abs, max_abs, "G={size} {k}x{n}");
                assert_eq!(e.relative, (sum_sq_err / sum_sq).sqrt(), "G={size} {k}x{n}");
                assert_eq!(e.groups, (k * n).div_ceil(size));
            }
        }
        let zeros = round_trip_error(&[0.0; 64], 64, D).unwrap();
        assert_eq!((zeros.relative, zeros.max_abs, zeros.groups), (0.0, 0.0, 1));
        let mut data = vec![0.5f32; 6];
        data[4] = f32::NAN;
        // Storage index 4 of a weight with k = 3 inputs is input 1 of output 1.
        assert!(matches!(
            round_trip_error(&data, 3, D),
            Err(Q4Error::NonFinite { row: 1, col: 1, .. })
        ));
    }

    #[test]
    fn groups_set_by_one_outlier_are_counted() {
        let spread: Vec<f32> = rand_vec(64, 1400).iter().map(|v| v * 0.01).collect();
        let mut outlier = spread.clone();
        outlier[17] = 2.0;
        let count = |values: &[f32], size| {
            round_trip_error(values, 64, Q4Group::new(size).unwrap())
                .unwrap()
                .outlier_groups
        };
        assert_eq!(count(&spread, 64), 0);
        assert_eq!(count(&outlier, 64), 1);
        // With groups of 16 only the group holding element 17 has it.
        assert_eq!(count(&outlier, 16), 1);
        assert_eq!(
            round_trip_error(&outlier, 64, Q4Group::new(16).unwrap())
                .unwrap()
                .groups,
            4
        );
        assert_eq!(count(&[0.3; 64], 64), 0, "a constant group has no range");
        // Two values at each end: leaving one out keeps the full range.
        let mut two_ends = spread;
        two_ends[3] = 2.0;
        two_ends[9] = 2.0;
        assert_eq!(count(&two_ends, 64), 0);
    }
}
