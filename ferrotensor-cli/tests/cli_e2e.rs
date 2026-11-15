//! End-to-end tests: run the `ferrotensor` binary on a tiny random model and
//! check exit codes and the files it writes.

mod common;

use std::path::Path;
use std::process::{Command, Output};

fn ferrotensor(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_ferrotensor"))
        .args(args)
        .output()
        .expect("run the ferrotensor binary")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

fn s(p: &Path) -> &str {
    p.to_str().expect("utf-8 path")
}

/// A temp directory holding a tiny model: config, weights and tokenizer.
fn model_dir(tag: &str) -> common::TempDir {
    let dir = common::TempDir::new(tag);
    common::write_tiny_llama(dir.path());
    common::write_tiny_tokenizer(dir.path());
    dir
}

fn assert_success(out: &Output) {
    assert!(
        out.status.success(),
        "stdout: {}\nstderr: {}",
        text(&out.stdout),
        text(&out.stderr)
    );
}

/// Run `train --task pretrain` on `data` with `--max-steps steps`, writing
/// into `out`; `extra` is appended to the arguments.
fn train(model: &Path, data: &Path, out: &Path, steps: usize, extra: &[&str]) -> Output {
    let steps = steps.to_string();
    let mut args = vec![
        "train",
        "--model",
        s(model),
        "--data",
        s(data),
        "--task",
        "pretrain",
        "--seq-len",
        "16",
        "--out",
        s(out),
        "--max-steps",
        steps.as_str(),
        "--grad-accum",
        "1",
        "--log-every",
        "1",
        "--lora-rank",
        "4",
    ];
    args.extend_from_slice(extra);
    ferrotensor(&args)
}

fn read_json(path: &Path) -> serde_json::Value {
    let content = std::fs::read_to_string(path).expect("read json file");
    serde_json::from_str(&content).expect("parse json file")
}

#[test]
fn missing_model_directory_is_an_error_not_a_panic() {
    let tmp = common::TempDir::new("missing");
    let missing = tmp.path().join("does-not-exist");
    let out = ferrotensor(&["train", "--model", s(&missing), "--data", s(&missing)]);
    let stderr = text(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "stdout: {}\nstderr: {stderr}",
        text(&out.stdout)
    );
    assert!(stderr.contains("error:"), "stderr: {stderr}");
    assert!(stderr.contains("does not exist"), "stderr: {stderr}");
    assert!(!stderr.contains("panicked"), "stderr: {stderr}");
}

#[test]
#[ignore = "needs a GPU"]
fn train_writes_adapter_state_and_metrics() {
    let model = model_dir("train-model");
    let out = common::TempDir::new("train-out");
    let data = out.path().join("tokens.bin");
    common::write_token_bin(&data, 400);

    let run = train(model.path(), &data, out.path(), 4, &["--profile"]);
    assert_success(&run);

    let profile = read_json(&out.path().join("profile.json"));
    assert_eq!(profile["steps_profiled"], 4);
    let dispatches = profile["dispatches_per_step"]
        .as_u64()
        .expect("dispatches_per_step is a count");
    assert!(dispatches > 0, "profile: {profile}");

    for name in [
        "lora_adapters.safetensors",
        "lora_adapters.safetensors.state.json",
        "training_state.json",
        "metrics.jsonl",
    ] {
        let path = out.path().join(name);
        let meta = std::fs::metadata(&path)
            .unwrap_or_else(|e| panic!("{} is missing: {e}", path.display()));
        assert!(meta.len() > 0, "{} is empty", path.display());
    }

    let metrics = std::fs::read_to_string(out.path().join("metrics.jsonl")).expect("metrics");
    for line in metrics.lines().filter(|l| !l.trim().is_empty()) {
        serde_json::from_str::<serde_json::Value>(line)
            .unwrap_or_else(|e| panic!("metrics line {line:?} is not JSON: {e}"));
    }
}

#[test]
#[ignore = "needs a GPU"]
fn eval_reports_a_finite_loss_with_the_adapter() {
    let model = model_dir("eval-model");
    let out = common::TempDir::new("eval-out");
    let data = out.path().join("tokens.bin");
    common::write_token_bin(&data, 400);

    assert_success(&train(model.path(), &data, out.path(), 4, &[]));

    let adapter = out.path().join("lora_adapters.safetensors");
    let run = ferrotensor(&[
        "eval",
        "--model",
        s(model.path()),
        "--data",
        s(&data),
        "--task",
        "pretrain",
        "--seq-len",
        "16",
        "--lora",
        s(&adapter),
    ]);
    assert_success(&run);

    let stdout = text(&run.stdout);
    let line = stdout
        .lines()
        .find(|l| l.starts_with("eval: "))
        .unwrap_or_else(|| panic!("no `eval: ` line in stdout: {stdout}"));
    let loss_text = line
        .split_once("loss ")
        .and_then(|(_, rest)| rest.split_once(','))
        .map(|(loss, _)| loss.trim())
        .unwrap_or_else(|| panic!("no loss in line: {line}"));
    let loss: f32 = loss_text
        .parse()
        .unwrap_or_else(|e| panic!("loss {loss_text:?} is not a number: {e}"));
    assert!(
        loss.is_finite() && loss > 0.0,
        "loss {loss} in line: {line}"
    );
}

#[test]
#[ignore = "needs a GPU"]
fn resume_continues_from_a_saved_adapter() {
    let model = model_dir("resume-model");
    let out = common::TempDir::new("resume-out");
    let data = out.path().join("tokens.bin");
    common::write_token_bin(&data, 400);
    let state = out.path().join("training_state.json");

    assert_success(&train(model.path(), &data, out.path(), 2, &[]));
    assert_eq!(read_json(&state)["step"], serde_json::json!(2));

    let adapter = out.path().join("lora_adapters.safetensors");
    assert_success(&train(
        model.path(),
        &data,
        out.path(),
        4,
        &["--resume", s(&adapter)],
    ));
    assert_eq!(read_json(&state)["step"], serde_json::json!(4));
}

#[test]
#[ignore = "needs a GPU"]
fn generate_emits_tokens() {
    let model = model_dir("generate-model");
    let run = ferrotensor(&[
        "generate",
        "--model",
        s(model.path()),
        "--prompt",
        "w5 w6 w7",
        "--raw",
        "--max-new-tokens",
        "4",
        "--seed",
        "1",
    ]);
    assert_success(&run);
    let stderr = text(&run.stderr);
    assert!(stderr.contains("[3 prompt + "), "stderr: {stderr}");
    assert!(stderr.contains("new tokens"), "stderr: {stderr}");
}

/// With 4-bit base weights the checkpoint is unmapped once the model is
/// built; training (with block recomputation) and generation run on what
/// the model holds.
#[test]
#[ignore = "needs a GPU"]
fn q4_weights_train_and_generate_without_the_checkpoint() {
    let model = model_dir("q4-model");
    let out = common::TempDir::new("q4-out");
    let data = out.path().join("tokens.bin");
    common::write_token_bin(&data, 400);

    let run = train(
        model.path(),
        &data,
        out.path(),
        2,
        &["--q4-weights", "--checkpoint-blocks"],
    );
    assert_success(&run);
    let metrics = std::fs::read_to_string(out.path().join("metrics.jsonl")).expect("metrics");
    let losses: Vec<f64> = metrics
        .lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .filter_map(|v| v["loss"].as_f64())
        .collect();
    assert!(!losses.is_empty(), "no loss in metrics: {metrics}");
    assert!(
        losses.iter().all(|l| l.is_finite() && *l > 0.0),
        "losses {losses:?}"
    );

    let adapter = out.path().join("lora_adapters.safetensors");
    let run = ferrotensor(&[
        "generate",
        "--model",
        s(model.path()),
        "--lora",
        s(&adapter),
        "--q4-weights",
        "--prompt",
        "w5 w6 w7",
        "--raw",
        "--max-new-tokens",
        "4",
        "--seed",
        "1",
    ]);
    assert_success(&run);
    let stderr = text(&run.stderr);
    assert!(stderr.contains("new tokens"), "stderr: {stderr}");
}

/// Run `plan` and parse its JSON; returns the exit code too.
fn plan_json(args: &[&str]) -> (Option<i32>, serde_json::Value) {
    let out = ferrotensor(args);
    let stdout = text(&out.stdout);
    let json = serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("{e}\nstdout: {stdout}\nstderr: {}", text(&out.stderr)));
    (out.status.code(), json)
}

#[test]
fn plan_picks_the_largest_batch_that_fits_a_given_budget() {
    let model = model_dir("plan-fits");
    let (code, plan) = plan_json(&[
        "plan",
        "--model",
        s(model.path()),
        "--json",
        "--memory-mib",
        "100000",
    ]);
    assert_eq!(code, Some(0), "{plan}");
    assert_eq!(plan["budget_mib"], 100000);
    assert_eq!(plan["budget_source"], "user");
    // Tiny model: 2 layers, hidden 32, FFN 64, 4 heads (2 KV), vocab 64, tied.
    assert_eq!(plan["parameters"], 20640);
    assert_eq!(plan["fits"], true);
    assert_eq!(plan["margin"], 0.15);
    assert_eq!(
        plan["chosen"],
        serde_json::json!({
            "batch_size": 8,
            "seq_len": 256,
            "rank": 8,
            "precision": "f16",
            "checkpoint_blocks": false,
        })
    );
    // weights 1 + adapters 1 + activations 62 + logits 1 + overhead 1350.
    assert_eq!(plan["estimated_peak_mib"], 1415);
    let command = plan["train_command"].as_str().expect("train_command");
    assert!(
        command.ends_with("--batch-size 8 --grad-accum 1 --seq-len 256 --lora-rank 8"),
        "{command}"
    );
}

#[test]
fn plan_exits_with_2_when_nothing_fits() {
    let model = model_dir("plan-too-small");
    let (code, plan) = plan_json(&[
        "plan",
        "--model",
        s(model.path()),
        "--json",
        "--memory-mib",
        "1000",
    ]);
    assert_eq!(code, Some(2), "{plan}");
    assert_eq!(plan["fits"], false);
    assert_eq!(plan["chosen"]["checkpoint_blocks"], true);
    assert_eq!(plan["chosen"]["batch_size"], 1);
    assert_eq!(plan["chosen"]["seq_len"], 128);
    // weights 1 + adapters 1 + activations 2 + logits 1 + overhead 1350.
    assert_eq!(plan["estimated_peak_mib"], 1355);
}

#[test]
fn plan_without_a_model_lists_the_documented_sizes() {
    let (code, plan) = plan_json(&["plan", "--json", "--memory-mib", "8000"]);
    assert_eq!(code, Some(0), "{plan}");
    let models = plan["models"].as_array().expect("models array");
    let names: Vec<&str> = models.iter().filter_map(|m| m["model"].as_str()).collect();
    assert_eq!(names, ["SmolLM2-135M", "SmolLM2-360M", "SmolLM2-1.7B"]);
    assert_eq!(models[0]["estimated_peak_mib"], 4679);
    assert_eq!(models[2]["chosen"]["batch_size"], 1);
    assert_eq!(models[2]["chosen"]["seq_len"], 128);
    assert_eq!(models[2]["fits"], true);
}
