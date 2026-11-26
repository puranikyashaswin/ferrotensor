//! The grouped LoRA kernels and the f16 mirror of `A` against CPU references.
//!
//! These need a real GPU, so they are `#[ignore]`d by default. Run them with:
//!
//! ```text
//! cargo test -p ferrotensor-backend --test gpu_lora_group -- --ignored --test-threads=1
//! ```

#![allow(clippy::unwrap_used)]

use ferrotensor_backend::{
    AdamParams, ComputeEngine, LoraGrad, LoraMirror, LoraProj, UnifiedTensor, WgpuContext,
    LORA_GROUP_MAX_RANK,
};
use std::sync::OnceLock;

struct Gpu {
    ctx: WgpuContext,
    engine: ComputeEngine,
}

fn gpu() -> &'static Gpu {
    static GPU: OnceLock<Gpu> = OnceLock::new();
    GPU.get_or_init(|| {
        let ctx = pollster::block_on(WgpuContext::new());
        let engine = ComputeEngine::new(&ctx);
        Gpu { ctx, engine }
    })
}

/// Deterministic pseudo-random values in [-1, 1).
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

fn upload(data: &[f32], shape: &[usize]) -> UnifiedTensor {
    UnifiedTensor::new(&gpu().ctx, data, shape)
}

fn read(t: &UnifiedTensor) -> Vec<f32> {
    pollster::block_on(t.to_vec(&gpu().ctx)).unwrap()
}

#[track_caller]
fn assert_close(name: &str, got: &[f32], want: &[f32], atol: f32, rtol: f32) {
    assert_eq!(got.len(), want.len(), "{name}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let err = (g - w).abs();
        let tol = atol + rtol * w.abs();
        assert!(
            err <= tol && g.is_finite() == w.is_finite(),
            "{name}: index {i}: got {g}, want {w} (err {err}, tol {tol})"
        );
    }
}

/// The scale of projection `g`.
fn scale(g: usize) -> f32 {
    [2.0, 0.5, -1.25][g]
}

/// One group of projections: `m` rows, `ns` output columns per projection,
/// `k` input columns, `ld - ns.len() * rank` unused columns in `U` and `dU`.
struct Case {
    m: usize,
    ns: &'static [usize],
    rank: usize,
    spare: usize,
    k: usize,
}

impl Case {
    fn ld(&self) -> usize {
        self.ns.len() * self.rank + self.spare
    }

    fn name(&self) -> String {
        format!(
            "m {} ns {:?} rank {} spare {} k {}",
            self.m, self.ns, self.rank, self.spare, self.k
        )
    }
}

const CASES: &[Case] = &[
    // The 135M q, k, v group at one and eight sequences.
    Case {
        m: 256,
        ns: &[576, 192, 192],
        rank: 8,
        spare: 0,
        k: 576,
    },
    Case {
        m: 2048,
        ns: &[576, 192, 192],
        rank: 8,
        spare: 0,
        k: 576,
    },
    // Rows, columns and k off every tile multiple.
    Case {
        m: 17,
        ns: &[65, 7, 130],
        rank: 8,
        spare: 0,
        k: 33,
    },
    // One row, two projections, U wider than the group.
    Case {
        m: 1,
        ns: &[100, 3],
        rank: 4,
        spare: 3,
        k: 5,
    },
    // A group of one at the widest rank.
    Case {
        m: 130,
        ns: &[1000],
        rank: LORA_GROUP_MAX_RANK,
        spare: 0,
        k: 40,
    },
    // An odd rank and a row count that is not a multiple of 4.
    Case {
        m: 2051,
        ns: &[64, 128, 192],
        rank: 3,
        spare: 2,
        k: 65,
    },
    // Rank 1 with wide projections.
    Case {
        m: 300,
        ns: &[1536, 512, 512],
        rank: 1,
        spare: 0,
        k: 1000,
    },
];

/// The inputs of one case, as host vectors.
struct Data {
    u: Vec<f32>,
    du: Vec<f32>,
    x: Vec<f32>,
    /// `B_g`, `[rank, n_g]`.
    b: Vec<Vec<f32>>,
    /// `dY_g` and `Y_g`, `[m, n_g]`.
    dy: Vec<Vec<f32>>,
    y: Vec<Vec<f32>>,
}

fn data(c: &Case, seed: u64) -> Data {
    let (m, ld) = (c.m, c.ld());
    let per = |len: fn(&Case, usize) -> usize, salt: u64, factor: f32| -> Vec<Vec<f32>> {
        (0..c.ns.len())
            .map(|g| {
                let v = rand_vec(len(c, g), seed + salt + g as u64);
                v.iter().map(|x| x * factor).collect()
            })
            .collect()
    };
    Data {
        u: rand_vec(m * ld, seed + 1),
        du: rand_vec(m * ld, seed + 2),
        x: rand_vec(m * c.k, seed + 3),
        b: per(|c, g| c.rank * c.ns[g], 10, 0.1),
        dy: per(|c, g| c.m * c.ns[g], 20, 1.0),
        y: per(|c, g| c.m * c.ns[g], 30, 1.0),
    }
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn ub_grouped_matches_cpu() {
    let g = gpu();
    for (seed, c) in CASES.iter().enumerate() {
        let d = data(c, 100 * seed as u64);
        let (m, rank, ld) = (c.m, c.rank, c.ld());
        let u = upload(&d.u, &[m, ld]);
        let b: Vec<_> = (0..c.ns.len())
            .map(|p| upload(&d.b[p], &[rank, c.ns[p]]))
            .collect();
        let y: Vec<_> = (0..c.ns.len())
            .map(|p| upload(&d.y[p], &[m, c.ns[p]]))
            .collect();
        let projs: Vec<_> = (0..c.ns.len())
            .map(|p| LoraProj {
                b: &b[p],
                act: &y[p],
                scale: scale(p),
            })
            .collect();
        g.engine.lora_ub_grouped(&g.ctx, &u, &projs).unwrap();
        for p in 0..c.ns.len() {
            let n = c.ns[p];
            let mut want = d.y[p].clone();
            for i in 0..m {
                for col in 0..n {
                    let mut acc = 0.0f64;
                    for j in 0..rank {
                        acc +=
                            f64::from(d.u[i * ld + p * rank + j]) * f64::from(d.b[p][j * n + col]);
                    }
                    want[i * n + col] += (f64::from(scale(p)) * acc) as f32;
                }
            }
            assert_close(
                &format!("ub {} projection {p}", c.name()),
                &read(&y[p]),
                &want,
                1e-5,
                1e-5,
            );
        }
    }
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn du_grouped_matches_cpu_and_leaves_other_columns() {
    let g = gpu();
    for (seed, c) in CASES.iter().enumerate() {
        let d = data(c, 100 * seed as u64 + 50);
        let (m, rank, ld) = (c.m, c.rank, c.ld());
        let du = upload(&d.du, &[m, ld]);
        let b: Vec<_> = (0..c.ns.len())
            .map(|p| upload(&d.b[p], &[rank, c.ns[p]]))
            .collect();
        let dy: Vec<_> = (0..c.ns.len())
            .map(|p| upload(&d.dy[p], &[m, c.ns[p]]))
            .collect();
        let projs: Vec<_> = (0..c.ns.len())
            .map(|p| LoraProj {
                b: &b[p],
                act: &dy[p],
                scale: scale(p),
            })
            .collect();
        g.engine.lora_du_grouped(&g.ctx, &du, &projs).unwrap();
        // The columns past the group keep their old values.
        let mut want = d.du.clone();
        for p in 0..c.ns.len() {
            let n = c.ns[p];
            for i in 0..m {
                for j in 0..rank {
                    let mut acc = 0.0f64;
                    for col in 0..n {
                        acc += f64::from(d.dy[p][i * n + col]) * f64::from(d.b[p][j * n + col]);
                    }
                    want[i * ld + p * rank + j] = (f64::from(scale(p)) * acc) as f32;
                }
            }
        }
        assert_close(&format!("du {}", c.name()), &read(&du), &want, 1e-4, 1e-4);
    }
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn grads_grouped_match_cpu_when_accumulating_and_storing() {
    let g = gpu();
    for (seed, c) in CASES.iter().enumerate() {
        let d = data(c, 100 * seed as u64 + 70);
        let (m, rank, ld, k) = (c.m, c.rank, c.ld(), c.k);
        for beta in [1.0f32, 0.5, 0.0] {
            let x = upload(&d.x, &[m, k]);
            let u = upload(&d.u, &[m, ld]);
            let du = upload(&d.du, &[m, ld]);
            let dy: Vec<_> = (0..c.ns.len())
                .map(|p| upload(&d.dy[p], &[m, c.ns[p]]))
                .collect();
            // Old gradients: NaN when beta is 0, which must never be read.
            let old_a: Vec<Vec<f32>> = (0..c.ns.len())
                .map(|p| {
                    if beta == 0.0 {
                        vec![f32::NAN; k * rank]
                    } else {
                        rand_vec(k * rank, 900 + p as u64)
                    }
                })
                .collect();
            let old_b: Vec<Vec<f32>> = (0..c.ns.len())
                .map(|p| {
                    if beta == 0.0 {
                        vec![f32::NAN; rank * c.ns[p]]
                    } else {
                        rand_vec(rank * c.ns[p], 950 + p as u64)
                    }
                })
                .collect();
            let ga: Vec<_> = old_a.iter().map(|v| upload(v, &[k, rank])).collect();
            let gb: Vec<_> = (0..c.ns.len())
                .map(|p| upload(&old_b[p], &[rank, c.ns[p]]))
                .collect();
            let projs: Vec<_> = (0..c.ns.len())
                .map(|p| LoraGrad {
                    dy: &dy[p],
                    grad_a: &ga[p],
                    grad_b: &gb[p],
                    scale: scale(p),
                })
                .collect();
            g.engine
                .lora_grads_grouped(&g.ctx, [&x, &u, &du], &projs, beta)
                .unwrap();
            let old = |v: &[f32], i: usize| if beta == 0.0 { 0.0 } else { beta * v[i] };
            for p in 0..c.ns.len() {
                let n = c.ns[p];
                let mut want_a = vec![0.0f32; k * rank];
                for col in 0..k {
                    for j in 0..rank {
                        let mut acc = 0.0f64;
                        for i in 0..m {
                            acc += f64::from(d.x[i * k + col])
                                * f64::from(d.du[i * ld + p * rank + j]);
                        }
                        want_a[col * rank + j] = acc as f32 + old(&old_a[p], col * rank + j);
                    }
                }
                let mut want_b = vec![0.0f32; rank * n];
                for j in 0..rank {
                    for col in 0..n {
                        let mut acc = 0.0f64;
                        for i in 0..m {
                            acc += f64::from(d.u[i * ld + p * rank + j])
                                * f64::from(d.dy[p][i * n + col]);
                        }
                        want_b[j * n + col] =
                            (f64::from(scale(p)) * acc) as f32 + old(&old_b[p], j * n + col);
                    }
                }
                let name = format!("{} beta {beta} projection {p}", c.name());
                assert_close(
                    &format!("grad_A {name}"),
                    &read(&ga[p]),
                    &want_a,
                    1e-3,
                    1e-4,
                );
                assert_close(
                    &format!("grad_B {name}"),
                    &read(&gb[p]),
                    &want_b,
                    1e-3,
                    1e-4,
                );
            }
        }
    }
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn reducing_kernels_give_the_same_bits_every_run() {
    let g = gpu();
    let c = &CASES[1];
    let d = data(c, 4242);
    let (m, rank, ld, k) = (c.m, c.rank, c.ld(), c.k);
    let b: Vec<_> = (0..c.ns.len())
        .map(|p| upload(&d.b[p], &[rank, c.ns[p]]))
        .collect();
    let dy: Vec<_> = (0..c.ns.len())
        .map(|p| upload(&d.dy[p], &[m, c.ns[p]]))
        .collect();
    let (x, u) = (upload(&d.x, &[m, k]), upload(&d.u, &[m, ld]));
    let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<_>>();
    let mut runs = Vec::new();
    for _ in 0..2 {
        let du = upload(&vec![0.0; m * ld], &[m, ld]);
        let projs: Vec<_> = (0..c.ns.len())
            .map(|p| LoraProj {
                b: &b[p],
                act: &dy[p],
                scale: scale(p),
            })
            .collect();
        g.engine.lora_du_grouped(&g.ctx, &du, &projs).unwrap();
        let ga: Vec<_> = (0..c.ns.len())
            .map(|_| upload(&vec![0.0; k * rank], &[k, rank]))
            .collect();
        let gb: Vec<_> = (0..c.ns.len())
            .map(|p| upload(&vec![0.0; rank * c.ns[p]], &[rank, c.ns[p]]))
            .collect();
        let gp: Vec<_> = (0..c.ns.len())
            .map(|p| LoraGrad {
                dy: &dy[p],
                grad_a: &ga[p],
                grad_b: &gb[p],
                scale: scale(p),
            })
            .collect();
        g.engine
            .lora_grads_grouped(&g.ctx, [&x, &u, &du], &gp, 0.0)
            .unwrap();
        let mut all = bits(read(&du));
        for t in ga.iter().chain(&gb) {
            all.extend(bits(read(t)));
        }
        runs.push(all);
    }
    assert!(runs[0] == runs[1], "two runs differ");
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn grouped_kernels_reject_what_they_cannot_run() {
    let g = gpu();
    let (m, n, rank) = (8usize, 16usize, 4usize);
    let u = upload(&rand_vec(m * 3 * rank, 1), &[m, 3 * rank]);
    let b: Vec<_> = (0..4)
        .map(|i| upload(&rand_vec(rank * n, 2 + i), &[rank, n]))
        .collect();
    let y: Vec<_> = (0..4)
        .map(|i| upload(&rand_vec(m * n, 20 + i), &[m, n]))
        .collect();
    let projs: Vec<_> = (0..4)
        .map(|i| LoraProj {
            b: &b[i],
            act: &y[i],
            scale: 1.0,
        })
        .collect();
    assert!(g.engine.lora_ub_grouped(&g.ctx, &u, &[]).is_err());
    assert!(g.engine.lora_ub_grouped(&g.ctx, &u, &projs).is_err());
    // U holds 3 ranks, so 3 projections fit and run.
    assert!(g.engine.lora_ub_grouped(&g.ctx, &u, &projs[..3]).is_ok());
    // U too narrow for 3 ranks.
    let narrow = upload(&rand_vec(m * (3 * rank - 1), 4), &[m, 3 * rank - 1]);
    assert!(g
        .engine
        .lora_ub_grouped(&g.ctx, &narrow, &projs[..3])
        .is_err());
    // Two ranks in one group.
    let b2 = upload(&rand_vec(2 * n, 5), &[2, n]);
    let mixed = [
        projs[0],
        LoraProj {
            b: &b2,
            act: &y[1],
            scale: 1.0,
        },
    ];
    assert!(g.engine.lora_ub_grouped(&g.ctx, &u, &mixed).is_err());
    // Above the largest rank.
    let r = LORA_GROUP_MAX_RANK + 1;
    let wide_u = upload(&rand_vec(m * r, 6), &[m, r]);
    let wide_b = upload(&rand_vec(r * n, 7), &[r, n]);
    let wide = [LoraProj {
        b: &wide_b,
        act: &y[0],
        scale: 1.0,
    }];
    assert!(g.engine.lora_ub_grouped(&g.ctx, &wide_u, &wide).is_err());
    // An output of the wrong size.
    let short = upload(&rand_vec(m * n - 1, 8), &[m * n - 1]);
    let bad = [LoraProj {
        b: &b[0],
        act: &short,
        scale: 1.0,
    }];
    assert!(g.engine.lora_ub_grouped(&g.ctx, &u, &bad).is_err());
}

/// The f16 bits of a tensor from `UnifiedTensor::from_f32_as_f16`, element by
/// element: the buffer read as f32 words keeps its bytes.
fn f16_bits(t: &UnifiedTensor) -> Vec<u16> {
    read(t)
        .into_iter()
        .flat_map(|w| {
            let bits = w.to_bits();
            [(bits & 0xffff) as u16, (bits >> 16) as u16]
        })
        .collect()
}

/// One mirror layout: `rows` x `ld` f16 weight, the `col0` of each `A`.
struct Layout {
    rows: usize,
    ld: usize,
    rank: usize,
    col0s: &'static [usize],
}

const LAYOUTS: &[Layout] = &[
    // The 135M fused q, k, v weight: 576 + 192 + 192 columns, then three A.
    Layout {
        rows: 576,
        ld: 984,
        rank: 8,
        col0s: &[960, 968, 976],
    },
    // The 135M o_proj weight: 576 columns, then A.
    Layout {
        rows: 576,
        ld: 584,
        rank: 8,
        col0s: &[576],
    },
    // Small, with A copies in the middle and a gap between them.
    Layout {
        rows: 5,
        ld: 20,
        rank: 4,
        col0s: &[2, 12],
    },
    // The widest rank, the A copy filling the row.
    Layout {
        rows: 33,
        ld: 16,
        rank: LORA_GROUP_MAX_RANK,
        col0s: &[0],
    },
];

/// The mirror equals `f16(A)` bit for bit after the initial copy and after
/// each of several real optimizer steps, and a step without a refresh leaves
/// it stale, so the check has teeth.
#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn mirror_equals_f16_of_a_bit_for_bit_after_each_optimizer_step() {
    let g = gpu();
    for (case, l) in LAYOUTS.iter().enumerate() {
        let seed = 10_000 * case as u64;
        let (rows, ld, rank) = (l.rows, l.ld, l.rank);
        // The rest of the weight is random values the mirror must not touch.
        let weight = rand_vec(rows * ld, seed + 1);
        let dst = UnifiedTensor::from_f32_as_f16(&g.ctx, &weight, &[rows, ld]);
        let untouched = f16_bits(&dst);
        let a: Vec<UnifiedTensor> = (0..l.col0s.len())
            .map(|p| {
                let v: Vec<f32> = rand_vec(rows * rank, seed + 10 + p as u64)
                    .iter()
                    .map(|x| x * 0.3)
                    .collect();
                upload(&v, &[rows, rank])
            })
            .collect();
        let srcs: Vec<_> = a
            .iter()
            .zip(l.col0s)
            .map(|(a, &col0)| LoraMirror { a, col0 })
            .collect();
        let owned = |col: usize| {
            l.col0s
                .iter()
                .position(|&c0| col >= c0 && col < c0 + rank)
                .map(|p| (p, col - l.col0s[p]))
        };
        let check = |label: &str| {
            let bits = f16_bits(&dst);
            let a_now: Vec<Vec<f32>> = a.iter().map(read).collect();
            let mut mismatches = 0;
            for r in 0..rows {
                for col in 0..ld {
                    let got = bits[r * ld + col];
                    let want = match owned(col) {
                        Some((p, j)) => half::f16::from_f32(a_now[p][r * rank + j]).to_bits(),
                        None => untouched[r * ld + col],
                    };
                    mismatches += usize::from(got != want);
                }
            }
            if mismatches == 0 {
                Ok(())
            } else {
                Err(format!(
                    "layout {case}, {label}: {mismatches} elements differ"
                ))
            }
        };
        // Before the first copy the A columns hold the old weights.
        assert!(check("before the first copy").is_err());
        g.engine.lora_mirror_f16(&g.ctx, &dst, &srcs).unwrap();
        check("after the initial copy").unwrap();

        let zeros = vec![0.0f32; rows * rank];
        let m: Vec<_> = a.iter().map(|_| upload(&zeros, &[rows, rank])).collect();
        let v: Vec<_> = a.iter().map(|_| upload(&zeros, &[rows, rank])).collect();
        for step in 1..=4u32 {
            for (p, t) in a.iter().enumerate() {
                let grad = upload(
                    &rand_vec(rows * rank, seed + 100 * u64::from(step) + p as u64),
                    &[rows, rank],
                );
                let params = AdamParams {
                    lr: 1e-2,
                    beta1: 0.9,
                    beta2: 0.999,
                    eps: 1e-8,
                    weight_decay: 0.01,
                    t: step as f32,
                    _pad: [0.0; 2],
                };
                g.engine
                    .adamw_update(&g.ctx, t, &grad, &m[p], &v[p], &params);
            }
            // The optimizer step moved A: the copy is stale until refreshed.
            assert!(
                check("after a step, before the refresh").is_err(),
                "layout {case} step {step}: the step changed no f16 value"
            );
            g.engine.lora_mirror_f16(&g.ctx, &dst, &srcs).unwrap();
            check(&format!("after step {step}")).unwrap();
        }
    }
}

/// Values that sit exactly between two f16 numbers round to the even one, as
/// `half::f16::from_f32` does, and f16 subnormals and zeros of both signs
/// carry over.
#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn mirror_rounds_ties_to_even_and_keeps_subnormals_and_signed_zero() {
    let g = gpu();
    let (rows, ld, rank) = (4usize, 8usize, 8usize);
    let ulp = 2.0f32.powi(-10);
    let half_ulp = ulp / 2.0;
    #[rustfmt::skip]
    let mut values = vec![
        // Ties around 1.0: below an even mantissa and below an odd one.
        1.0 + half_ulp, 1.0 + ulp + half_ulp, -(1.0 + half_ulp), -(1.0 + ulp + half_ulp),
        // Just off a tie, either side.
        1.0 + half_ulp * 1.001, 1.0 + half_ulp * 0.999, 0.1, -0.3,
        // Zeros, an f16 subnormal, the smallest subnormal, the largest f16.
        0.0, -0.0, 2.0f32.powi(-20), 2.0f32.powi(-24),
        65504.0, -65504.0, 6.1035156e-5, 0.0,
    ];
    values.resize(rows * rank, 0.25);
    let a = upload(&values, &[rows, rank]);
    let dst = UnifiedTensor::from_f32_as_f16(&g.ctx, &vec![0.0; rows * ld], &[rows, ld]);
    g.engine
        .lora_mirror_f16(&g.ctx, &dst, &[LoraMirror { a: &a, col0: 0 }])
        .unwrap();
    let bits = f16_bits(&dst);
    for r in 0..rows {
        for j in 0..rank {
            let v = values[r * rank + j];
            assert_eq!(
                bits[r * ld + j],
                half::f16::from_f32(v).to_bits(),
                "row {r} column {j}: {v}"
            );
        }
    }
}

/// The mirror gives the f16 bits of `half::f16::from_f32` for every kind of
/// f32: ties both ways, either side of a tie, the largest finite value and
/// the first one that overflows, subnormals down to the smallest one, zeros,
/// infinities and NaNs. WGSL leaves the rounding of f32 to f16 to the device
/// (WARP rounds 1.0014648 down), so the kernel rounds the bits itself.
#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn mirror_matches_the_host_conversion_for_every_kind_of_f32() {
    let g = gpu();
    // (f32 bits, f16 bits): the expected value is written out, and the host
    // conversion must agree with it too.
    #[rustfmt::skip]
    let fixed: &[(u32, u16)] = &[
        // 1 + 2^-11 is a tie, the even mantissa 0 wins; 1 + 3 * 2^-11 is a
        // tie, the even mantissa 2 wins; each is a tie in the other sign too.
        (0x3F80_1000, 0x3C00), (0x3F80_3000, 0x3C02),
        (0xBF80_1000, 0xBC00), (0xBF80_3000, 0xBC02),
        // One f32 step above and below the first tie.
        (0x3F80_0FFF, 0x3C00), (0x3F80_1001, 0x3C01),
        // 1.0014648 = 1 + 3 * 2^-11 (0x3F803000 above) is the value WARP
        // rounded down to 0x3C01.
        // The largest finite f16, the tie above it (even: infinity), just
        // below that tie, and a large value.
        (0x477F_E000, 0x7BFF), (0x477F_F000, 0x7C00), (0x477F_EFFF, 0x7BFF),
        (0xC77F_E000, 0xFBFF), (0xC77F_F000, 0xFC00), (0x4780_0000, 0x7C00),
        (0x7F7F_FFFF, 0x7C00), (0xFF7F_FFFF, 0xFC00),
        // Smallest normal f16 (2^-14), the largest subnormal and the value
        // just below the smallest normal that rounds up to it.
        (0x3880_0000, 0x0400), (0x387F_C000, 0x03FF), (0x387F_E000, 0x0400),
        // Smallest subnormal 2^-24 and its negative.
        (0x3380_0000, 0x0001), (0xB380_0000, 0x8001),
        // 2^-25 is a tie between 0 and the smallest subnormal: zero wins.
        // Just above it rounds to the smallest subnormal, and so does 0.75
        // of it. 3 * 2^-25 is a tie between 1 and 2 subnormals: 2 wins.
        // 5 * 2^-25 is a tie between 2 and 3: 2 wins.
        (0x3300_0000, 0x0000), (0xB300_0000, 0x8000), (0x3300_0001, 0x0001),
        (0x3340_0000, 0x0001), (0x33C0_0000, 0x0002), (0x3420_0000, 0x0002),
        // Far below the f16 range, and f32 subnormals: zeros keep the sign.
        (0x3000_0000, 0x0000), (0xB000_0000, 0x8000),
        (0x0000_0001, 0x0000), (0x8000_0001, 0x8000), (0x007F_FFFF, 0x0000),
        // Zeros.
        (0x0000_0000, 0x0000), (0x8000_0000, 0x8000),
        // Infinities.
        (0x7F80_0000, 0x7C00), (0xFF80_0000, 0xFC00),
        // NaNs: quiet bit set, sign and the top ten payload bits kept.
        (0x7FC0_0000, 0x7E00), (0xFFC0_0000, 0xFE00), (0x7F80_0001, 0x7E00),
        (0x7FA0_0000, 0x7F00), (0xFFFF_FFFF, 0xFFFF),
    ];
    let mut bits: Vec<u32> = fixed.iter().map(|&(b, _)| b).collect();
    // Every finite f16 value, the exact middle between it and the next one,
    // and the f32 values one step either side of that middle, in both signs.
    for h in 0..0x7BFFu16 {
        let lo = half::f16::from_bits(h).to_f32();
        let hi = half::f16::from_bits(h + 1).to_f32();
        let mid = ((lo + hi) / 2.0).to_bits();
        for b in [lo.to_bits(), mid - 1, mid, mid + 1] {
            bits.push(b);
            bits.push(b | 0x8000_0000);
        }
    }
    // Random patterns over all of f32, and over the f16 exponent range.
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 32) as u32
    };
    for _ in 0..50_000 {
        let x = next();
        bits.push(x);
        let exponent = 100 + next() % 45;
        bits.push((x & 0x807F_FFFF) | (exponent << 23));
    }
    let (rank, ld) = (8usize, 8usize);
    let rows = bits.len().div_ceil(rank);
    bits.resize(rows * rank, 0);
    let values: Vec<f32> = bits.iter().map(|&b| f32::from_bits(b)).collect();
    let a = upload(&values, &[rows, rank]);
    let dst = UnifiedTensor::from_f32_as_f16(&g.ctx, &vec![0.0; rows * ld], &[rows, ld]);
    g.engine
        .lora_mirror_f16(&g.ctx, &dst, &[LoraMirror { a: &a, col0: 0 }])
        .unwrap();
    let got = f16_bits(&dst);
    for &(b, want) in fixed {
        assert_eq!(
            half::f16::from_f32(f32::from_bits(b)).to_bits(),
            want,
            "host conversion of {b:#010x}"
        );
    }
    let mut wrong = Vec::new();
    for (i, &b) in bits.iter().enumerate() {
        let want = half::f16::from_f32(f32::from_bits(b)).to_bits();
        if got[i] != want {
            wrong.push(format!("{b:#010x}: got {:#06x}, want {want:#06x}", got[i]));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of {} values differ from the host conversion, first: {:?}",
        wrong.len(),
        bits.len(),
        &wrong[..wrong.len().min(8)]
    );
    for (i, &(b, want)) in fixed.iter().enumerate() {
        assert_eq!(got[i], want, "fixed case {b:#010x}");
    }
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn mirror_rejects_layouts_it_cannot_copy_whole_words_for() {
    let g = gpu();
    let (rows, ld, rank) = (4usize, 24usize, 8usize);
    let dst = UnifiedTensor::from_f32_as_f16(&g.ctx, &vec![0.0; rows * ld], &[rows, ld]);
    let a = upload(&rand_vec(rows * rank, 1), &[rows, rank]);
    let at = |col0| LoraMirror { a: &a, col0 };
    // Odd start, past the end, overlapping and empty.
    assert!(g.engine.lora_mirror_f16(&g.ctx, &dst, &[at(1)]).is_err());
    assert!(g.engine.lora_mirror_f16(&g.ctx, &dst, &[at(18)]).is_err());
    assert!(g
        .engine
        .lora_mirror_f16(&g.ctx, &dst, &[at(0), at(6)])
        .is_err());
    assert!(g.engine.lora_mirror_f16(&g.ctx, &dst, &[]).is_err());
    // An odd row length, an odd rank and a destination that is not f16.
    let odd = UnifiedTensor::from_f32_as_f16(&g.ctx, &vec![0.0; rows * 23], &[rows, 23]);
    assert!(g.engine.lora_mirror_f16(&g.ctx, &odd, &[at(0)]).is_err());
    let a3 = upload(&rand_vec(rows * 3, 2), &[rows, 3]);
    let three = [LoraMirror { a: &a3, col0: 0 }];
    assert!(g.engine.lora_mirror_f16(&g.ctx, &dst, &three).is_err());
    let f32_dst = upload(&vec![0.0; rows * ld], &[rows, ld]);
    assert!(g
        .engine
        .lora_mirror_f16(&g.ctx, &f32_dst, &[at(0)])
        .is_err());
    // Aligned layouts work.
    assert!(g
        .engine
        .lora_mirror_f16(&g.ctx, &dst, &[at(0), at(8)])
        .is_ok());
}
