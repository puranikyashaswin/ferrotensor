//! `ferrotensor merge`: write a model directory with a LoRA adapter folded
//! into the base weights.

use crate::{print_json, unmet, Result};
use clap::Args;
use ferrotensor_trainer::merge::merge_adapter;
use std::path::PathBuf;

/// Options for `ferrotensor merge`.
#[derive(Args, Debug)]
pub struct MergeArgs {
    /// Base model directory the adapter was trained on.
    #[arg(long, value_name = "DIR")]
    model: PathBuf,
    /// LoRA adapter: a Ferrotensor file or a PEFT adapter directory.
    #[arg(long, alias = "adapter", value_name = "FILE")]
    lora: PathBuf,
    /// Output model directory; must be new or empty.
    #[arg(long, alias = "output", value_name = "DIR")]
    out: PathBuf,
    /// Print one JSON object with the output directory and tensor counts.
    #[arg(long)]
    pub json: bool,
}

/// Runs the merge on the host; no GPU is opened.
pub fn run(args: MergeArgs, json: bool) -> Result<()> {
    let adapter = crate::read_adapter(&args.lora).map_err(|e| unmet(e.to_string()))?;
    // I/O failures while writing stay runtime errors; everything else
    // `merge_adapter` reports is about its input.
    let report = merge_adapter(&args.model, &adapter, &args.out).map_err(|e| {
        if e.is::<std::io::Error>() {
            e
        } else {
            unmet(e.to_string())
        }
    })?;
    if json {
        print_json(&serde_json::json!({
            "out_dir": args.out,
            "merged": report.merged,
            "copied": report.copied,
        }))
    } else {
        println!(
            "wrote {} ({} tensors merged, {} copied)",
            args.out.display(),
            report.merged,
            report.copied
        );
        Ok(())
    }
}
