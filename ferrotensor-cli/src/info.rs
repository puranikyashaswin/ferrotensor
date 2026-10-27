//! `ferrotensor info`: the GPUs wgpu sees, which one Ferrotensor uses, and
//! (with `--check`) whether its kernels give correct results on it.

use clap::Args;
use ferrotensor_backend::adapters::{self, AdapterReport, CheckResult};
use ferrotensor_backend::WgpuContext;
use serde_json::{json, Value};

/// Options for `ferrotensor info`.
#[derive(Args, Debug)]
pub struct InfoArgs {
    /// Also run matmul, softmax and RMSNorm on the selected GPU and compare
    /// them with a CPU reference.
    #[arg(long)]
    pub check: bool,
    /// Print JSON (paste it into a bug report).
    #[arg(long)]
    pub json: bool,
}

fn mib(bytes: u64) -> String {
    format!("{} MiB", bytes / (1024 * 1024))
}

fn adapter_json(a: &AdapterReport) -> Value {
    json!({
        "name": a.name,
        "vendor": a.vendor,
        "device_type": a.device_type,
        "backend": a.backend,
        "driver": a.driver,
        "max_buffer_size": a.max_buffer_size,
        "max_storage_buffer_binding_size": a.max_storage_buffer_binding_size,
        "max_compute_invocations_per_workgroup": a.max_compute_invocations_per_workgroup,
        "max_compute_workgroup_storage_size": a.max_compute_workgroup_storage_size,
        "shader_f16": a.shader_f16,
        "selected": a.selected,
    })
}

fn check_json(r: &CheckResult) -> Value {
    json!({
        "name": r.name,
        "max_abs_err": r.max_abs_err,
        "tolerance": r.tolerance,
        "passed": r.passed(),
    })
}

/// Runs `ferrotensor info`. Fails (exit code 1) if no adapter is found or a
/// `--check` kernel is out of tolerance.
pub fn run(args: InfoArgs) -> Result<(), Box<dyn std::error::Error>> {
    let adapters = adapters::list_adapters();
    let checks = if args.check && adapters.iter().any(|a| a.selected) {
        let ctx = pollster::block_on(WgpuContext::new());
        Some(pollster::block_on(adapters::self_check(&ctx)))
    } else {
        None
    };

    if args.json {
        let out = json!({
            "ferrotensor": env!("CARGO_PKG_VERSION"),
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "adapters": adapters.iter().map(adapter_json).collect::<Vec<_>>(),
            "checks": checks.as_ref().map(|c| c.iter().map(check_json).collect::<Vec<_>>()),
        });
        println!("{}", serde_json::to_string_pretty(&out)?);
    } else {
        println!(
            "ferrotensor {} on {} {}",
            env!("CARGO_PKG_VERSION"),
            std::env::consts::OS,
            std::env::consts::ARCH
        );
        if adapters.is_empty() {
            println!("\nNo GPU adapter found. Ferrotensor's CPU device still works.");
        }
        for a in &adapters {
            let mark = if a.selected { "* " } else { "  " };
            println!(
                "\n{mark}{} ({}, {}, {})",
                a.name, a.vendor, a.device_type, a.backend
            );
            if !a.driver.is_empty() {
                println!("    driver            {}", a.driver);
            }
            println!("    max buffer        {}", mib(a.max_buffer_size));
            println!(
                "    max storage bind  {}",
                mib(u64::from(a.max_storage_buffer_binding_size))
            );
            println!(
                "    workgroup         {} threads, {} KiB shared",
                a.max_compute_invocations_per_workgroup,
                a.max_compute_workgroup_storage_size / 1024
            );
            println!(
                "    shader f16        {}",
                if a.shader_f16 { "yes" } else { "no" }
            );
        }
        if adapters.iter().any(|a| a.selected) {
            println!("\n* = the adapter Ferrotensor uses");
        }
        if let Some(checks) = &checks {
            println!("\nKernel check against a CPU reference:");
            for r in checks {
                println!(
                    "  {:<6} {:<24} max error {:.1e} (tolerance {:.0e})",
                    if r.passed() { "ok" } else { "FAIL" },
                    r.name,
                    r.max_abs_err,
                    r.tolerance
                );
            }
        } else if args.check {
            println!("\nNo GPU to check.");
        }
    }

    if adapters.is_empty() {
        return Err("no GPU adapter found".into());
    }
    if let Some(failed) = checks.iter().flatten().find(|r| !r.passed()) {
        return Err(format!(
            "{} is out of tolerance on this GPU; please open an issue with `ferrotensor info --check --json`",
            failed.name
        )
        .into());
    }
    Ok(())
}
