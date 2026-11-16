//! `ferrotensor export-gguf`: write a model as one GGUF file for llama.cpp
//! and Ollama.

use crate::{print_json, resolve_model, unmet, Result};
use clap::Args;
use ferrotensor_trainer::gguf::{export_gguf, GgufPrecision};
use std::path::PathBuf;

/// Options for `ferrotensor export-gguf`.
#[derive(Args, Debug)]
pub struct ExportGgufArgs {
    /// Llama-style model directory (for example a `merge` output) or cached
    /// Hugging Face model id.
    #[arg(long, value_name = "DIR_OR_ID")]
    model: PathBuf,
    /// Output GGUF file; must not exist.
    #[arg(long, alias = "output", value_name = "FILE")]
    out: PathBuf,
    /// LoRA adapter to add to the weights while writing: a Ferrotensor file
    /// or a PEFT adapter directory.
    #[arg(long, alias = "adapter", value_name = "FILE")]
    lora: Option<PathBuf>,
    /// Store every tensor in f32 instead of f16.
    #[arg(long)]
    f32: bool,
    /// Print one JSON object with the output file and tensor counts.
    #[arg(long)]
    pub json: bool,
}

/// Runs the export on the host; no GPU is opened.
pub fn run(args: ExportGgufArgs, json: bool) -> Result<()> {
    let model_dir = resolve_model(&args.model)?;
    let adapter = args
        .lora
        .as_deref()
        .map(crate::read_adapter)
        .transpose()
        .map_err(|e| unmet(e.to_string()))?;
    let precision = if args.f32 {
        GgufPrecision::F32
    } else {
        GgufPrecision::F16
    };
    let name = args
        .model
        .file_name()
        .unwrap_or(args.model.as_os_str())
        .to_string_lossy();
    // I/O failures while writing stay runtime errors; everything else
    // `export_gguf` reports is about its input.
    let report =
        export_gguf(&model_dir, &name, adapter.as_ref(), &args.out, precision).map_err(|e| {
            if e.is::<std::io::Error>() {
                e
            } else {
                unmet(e.to_string())
            }
        })?;
    if json {
        print_json(&serde_json::json!({
            "out_file": args.out,
            "tensors": report.tensors,
            "merged": report.merged,
        }))
    } else {
        println!(
            "wrote {} ({} tensors, {} with the adapter added)",
            args.out.display(),
            report.tensors,
            report.merged
        );
        Ok(())
    }
}
