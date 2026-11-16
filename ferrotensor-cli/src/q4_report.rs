//! `ferrotensor q4-report`: how much 4-bit storage changes each weight of a
//! checkpoint, measured on the host, one tensor in memory at a time.

use crate::{print_json, resolve_model, unmet, Result};
use clap::Args;
use ferrotensor_backend::q4::{self, Q4Group, RoundTripError};
use ferrotensor_loader::WeightsContext;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

/// Options for `ferrotensor q4-report`.
#[derive(Args, Debug)]
pub struct Q4ReportArgs {
    /// Model directory or cached Hugging Face model id.
    #[arg(long, value_name = "DIR_OR_ID")]
    model: PathBuf,
    /// Weights per scale and minimum, as for --q4-weights: 16, 32, 64 or 128.
    #[arg(long, value_name = "N", default_value = "64", value_parser = crate::parse_q4_group)]
    q4_group: Q4Group,
    /// Print one JSON object with a `tensors` list.
    #[arg(long)]
    pub json: bool,
}

/// The 4-bit round trip of one checkpoint tensor.
struct TensorError {
    name: String,
    /// As stored: PyTorch `[out, in]`.
    shape: Vec<usize>,
    error: RoundTripError,
}

/// Measures every 2-D tensor of the checkpoint in `dir`, sorted by name.
fn measure(dir: &Path, group: Q4Group) -> Result<Vec<TensorError>> {
    let mut weights = WeightsContext::new();
    weights
        .load_model_dir(dir)
        .map_err(|e| unmet(format!("cannot load weights from {}: {e}", dir.display())))?;
    let mut names: Vec<String> = weights.tensor_names().map(str::to_string).collect();
    names.sort();
    let mut rows = Vec::new();
    for name in names {
        let shape = weights
            .get_tensor_info(&name)?
            .map_or_else(Vec::new, |meta| meta.shape);
        let [_, inputs] = shape[..] else {
            continue;
        };
        // Read, measured and dropped before the next tensor is read.
        let (values, _) = weights.read_f32_tensor(&name)?;
        // A PyTorch [out, in] weight read row by row is the 4-bit storage
        // order, so it is measured without a transpose.
        let error = q4::round_trip_error(&values, inputs, group)
            .map_err(|e| unmet(format!("{name}: {e}")))?;
        rows.push(TensorError { name, shape, error });
    }
    Ok(rows)
}

fn outlier_fraction(e: &RoundTripError) -> f64 {
    e.outlier_groups as f64 / e.groups.max(1) as f64
}

fn report_json(dir: &Path, group: Q4Group, rows: &[TensorError]) -> Value {
    json!({
        "model": dir,
        "group": group.size(),
        "tensors": rows.iter().map(|r| json!({
            "name": r.name,
            "shape": r.shape,
            "relative_error": r.error.relative,
            "max_abs_error": r.error.max_abs,
            "groups": r.error.groups,
            "outlier_groups": r.error.outlier_groups,
            "outlier_fraction": outlier_fraction(&r.error),
        })).collect::<Vec<_>>(),
    })
}

/// Runs the report on the host; no GPU is opened.
pub fn run(args: Q4ReportArgs, json: bool) -> Result<()> {
    let dir = resolve_model(&args.model)?;
    let rows = measure(&dir, args.q4_group)?;
    if json {
        return print_json(&report_json(&dir, args.q4_group, &rows));
    }
    println!(
        "4-bit round trip of {}, groups of {}: relative error |w - q(w)| / |w| \
         (Frobenius), largest |w - q(w)|, and groups where one value sets at \
         least half of the range",
        dir.display(),
        args.q4_group.size()
    );
    println!(
        "{:<52} {:>13} {:>10} {:>11} {:>9}",
        "tensor", "shape", "rel error", "max error", "outliers"
    );
    for r in &rows {
        let shape = format!("{:?}", r.shape);
        println!(
            "{:<52} {shape:>13} {:>10.5} {:>11.3e} {:>8.3}%",
            r.name,
            r.error.relative,
            r.error.max_abs,
            100.0 * outlier_fraction(&r.error)
        );
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// A sine of period 9: any 16 consecutive values span almost the whole
    /// range `[-scale, scale]`, so no group has an outlier.
    fn values(n: usize, phase: f32, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (i as f32 * 0.7 + phase).sin() * scale)
            .collect()
    }

    fn write_checkpoint(dir: &Path, tensors: &[(&str, Vec<usize>, Vec<f32>)]) {
        let bytes: Vec<Vec<u8>> = tensors
            .iter()
            .map(|(_, _, v)| v.iter().flat_map(|x| x.to_le_bytes()).collect())
            .collect();
        let views: HashMap<String, safetensors::tensor::TensorView<'_>> = tensors
            .iter()
            .zip(&bytes)
            .map(|((name, shape, _), b)| {
                let view =
                    safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape.clone(), b)
                        .unwrap();
                (name.to_string(), view)
            })
            .collect();
        let data = safetensors::tensor::serialize(views, &None).unwrap();
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("model.safetensors"), data).unwrap();
    }

    /// The relative error of the builder's path: transposed to the logical
    /// `[in, out]`, quantised, decoded.
    fn builder_relative_error(w: &[f32], out: usize, inp: usize, group: Q4Group) -> f64 {
        let logical: Vec<f32> = (0..inp * out)
            .map(|i| w[(i % out) * inp + i / out])
            .collect();
        let decoded = q4::dequantize(
            &q4::quantize(&logical, inp, out, group).unwrap(),
            inp,
            out,
            group,
        );
        let (err, norm) = logical
            .iter()
            .zip(&decoded)
            .fold((0f64, 0f64), |(e, n), (v, d)| {
                (e + f64::from(v - d).powi(2), n + f64::from(*v).powi(2))
            });
        (err / norm).sqrt()
    }

    #[test]
    fn reports_every_matrix_in_name_order() {
        let dir =
            std::env::temp_dir().join(format!("ferrotensor-q4-report-{}", std::process::id()));
        let q = values(48 * 96, 0.1, 0.05);
        let mut down = values(96 * 48, 0.2, 0.05);
        down[100] = 3.0;
        write_checkpoint(
            &dir,
            &[
                (
                    "model.layers.0.self_attn.q_proj.weight",
                    vec![48, 96],
                    q.clone(),
                ),
                (
                    "model.layers.0.mlp.down_proj.weight",
                    vec![96, 48],
                    down.clone(),
                ),
                ("model.norm.weight", vec![96], values(96, 0.3, 1.0)),
            ],
        );
        for size in [16, 64] {
            let group = Q4Group::new(size).unwrap();
            let rows = measure(&dir, group).unwrap();
            let names: Vec<&str> = rows.iter().map(|r| r.name.as_str()).collect();
            assert_eq!(
                names,
                [
                    "model.layers.0.mlp.down_proj.weight",
                    "model.layers.0.self_attn.q_proj.weight"
                ],
                "1-D tensors are left out"
            );
            let (d, q_row) = (&rows[0], &rows[1]);
            assert_eq!(d.shape, [96, 48]);
            assert_eq!(d.error.groups, 96 * 48 / size);
            assert_eq!(d.error.outlier_groups, 1, "the 3.0 sets one group's range");
            assert_eq!(q_row.error.outlier_groups, 0);
            let want = builder_relative_error(&q, 48, 96, group);
            assert!(
                (q_row.error.relative - want).abs() <= 1e-9 * want,
                "G={size}: {} vs {want}",
                q_row.error.relative
            );
            let want = builder_relative_error(&down, 96, 48, group);
            assert!((d.error.relative - want).abs() <= 1e-9 * want);

            let json = report_json(&dir, group, &rows);
            assert_eq!(json["group"], size);
            assert_eq!(
                json["tensors"][0]["name"],
                "model.layers.0.mlp.down_proj.weight"
            );
            assert_eq!(json["tensors"][0]["shape"], json!([96, 48]));
            assert_eq!(json["tensors"][0]["outlier_groups"], 1);
            assert_eq!(json["tensors"][0]["groups"], 96 * 48 / size);
            assert_eq!(
                json["tensors"][1]["relative_error"].as_f64().unwrap(),
                q_row.error.relative
            );
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_weight_that_cannot_be_quantised_names_the_tensor() {
        let dir =
            std::env::temp_dir().join(format!("ferrotensor-q4-report-nan-{}", std::process::id()));
        let mut w = values(16 * 8, 0.4, 0.1);
        w[9] = f32::NAN;
        write_checkpoint(&dir, &[("lm_head.weight", vec![16, 8], w)]);
        let err = measure(&dir, Q4Group::DEFAULT).err().unwrap().to_string();
        // Storage index 9 of a weight with 8 inputs: input 1 of output 1.
        assert_eq!(
            err,
            "lm_head.weight: cannot quantise to 4 bits: element [1, 1] is NaN"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
