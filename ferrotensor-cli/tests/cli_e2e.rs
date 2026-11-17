//! End-to-end tests: run the `ferrotensor` binary on a tiny random model and
//! check exit codes and the files it writes.

#![allow(clippy::expect_used, clippy::panic)]

mod common;

use std::io::Write as _;
use std::path::Path;
use std::process::{Command, Output, Stdio};

fn ferrotensor(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_ferrotensor"))
        .args(args)
        .output()
        .expect("run the ferrotensor binary")
}

/// Run the binary with `input` piped to its stdin.
fn ferrotensor_with_input(args: &[&str], input: &str) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_ferrotensor"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("run the ferrotensor binary");
    child
        .stdin
        .take()
        .expect("piped stdin")
        .write_all(input.as_bytes())
        .expect("write stdin");
    child
        .wait_with_output()
        .expect("wait for the ferrotensor binary")
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

/// Files a LoRA training run writes into its output directory.
const RUN_FILES: [&str; 4] = [
    "lora_adapters.safetensors",
    "lora_adapters.safetensors.state.json",
    "training_state.json",
    "metrics.jsonl",
];

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
    // 2: invalid input, which the caller fixes by changing the request.
    assert_eq!(
        out.status.code(),
        Some(2),
        "stdout: {}\nstderr: {stderr}",
        text(&out.stdout)
    );
    assert!(stderr.contains("error:"), "stderr: {stderr}");
    assert!(stderr.contains("does not exist"), "stderr: {stderr}");
    assert!(!stderr.contains("panicked"), "stderr: {stderr}");
}

#[test]
fn json_errors_are_one_object_on_stdout() {
    let tmp = common::TempDir::new("missing-json");
    let missing = tmp.path().join("does-not-exist");
    let missing = s(&missing);
    let model_and_data = ["--model", missing, "--data", missing, "--json"];
    for (command, extra) in [
        ("train", &[][..]),
        ("eval", &[][..]),
        ("finetune", &["--memory-mib", "4096"][..]),
    ] {
        let mut args = vec![command];
        args.extend_from_slice(&model_and_data);
        args.extend_from_slice(extra);
        let out = ferrotensor(&args);
        let stdout = text(&out.stdout);
        let json: serde_json::Value = serde_json::from_str(&stdout)
            .unwrap_or_else(|e| panic!("{command}: {e}\nstdout: {stdout}"));
        assert_eq!(out.status.code(), Some(2), "{command}: {json}");
        let error = json["error"].as_str().expect("error string");
        assert!(error.contains(missing), "{command}: {error}");
    }
}

/// A config feature the forward pass lacks is refused before the GPU is
/// touched, with exit code 2 and the field named, for every model command.
#[test]
fn unsupported_config_feature_exits_with_2() {
    let model = model_dir("unsupported-act");
    let config_path = model.path().join("config.json");
    let mut config = read_json(&config_path);
    config["hidden_act"] = serde_json::json!("gelu");
    std::fs::write(&config_path, config.to_string()).expect("write config");
    let dir = s(model.path());
    for args in [
        &[
            "generate", "--model", dir, "--raw", "--prompt", "w5", "--json",
        ][..],
        &["eval", "--model", dir, "--data", dir, "--json"][..],
        &["train", "--model", dir, "--data", dir, "--json"][..],
    ] {
        let out = ferrotensor(args);
        let stdout = text(&out.stdout);
        assert_eq!(out.status.code(), Some(2), "{args:?}: {stdout}");
        let json: serde_json::Value = serde_json::from_str(&stdout).expect("error json");
        let error = json["error"].as_str().expect("error string");
        assert!(
            error.contains("hidden_act \"gelu\" is not supported"),
            "{args:?}: {error}"
        );
    }
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
    let passes = profile["compute_passes_per_step"]
        .as_u64()
        .expect("compute_passes_per_step is a count");
    if profile["kernel_timing"] == "pass_boundaries" {
        assert_eq!(passes, dispatches, "one dispatch per timed pass: {profile}");
    } else {
        assert!(
            passes > 0 && passes * 4 <= dispatches,
            "up to 8 dispatches share a compute pass: {profile}"
        );
    }
    assert!(
        profile["device_polls_per_step"].as_u64() > Some(0),
        "each step waits for the GPU at least once: {profile}"
    );
    assert_kernels(&profile, "_per_step");

    for name in RUN_FILES {
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
        "--json",
    ]);
    assert_success(&run);
    let json: serde_json::Value = serde_json::from_str(&text(&run.stdout)).expect("eval json");
    let json_loss = json["loss"].as_f64().expect("loss");
    assert!(
        (json_loss - f64::from(loss)).abs() < 1e-4,
        "{json} vs {line}"
    );
    let perplexity = json["perplexity"].as_f64().expect("perplexity");
    assert!((perplexity - json_loss.exp()).abs() < 1e-3, "{json}");
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

/// `--train-modules embed` trains the token embedding: `eval --lora` on the
/// saved weights gives the held-out loss the trained model gave at the end
/// of `train`, and `--resume` continues from the file.
#[test]
#[ignore = "needs a GPU"]
fn a_trained_embedding_saves_loads_and_resumes() {
    let model = model_dir("embed-model");
    let out = common::TempDir::new("embed-out");
    let data = out.path().join("tokens.bin");
    common::write_token_bin(&data, 400);
    let embed = [
        "--train-modules",
        "embed",
        "--lr",
        "1e-2",
        "--warmup-steps",
        "0",
    ];

    let mut extra = embed.to_vec();
    extra.extend_from_slice(&["--eval-data", s(&data), "--json"]);
    let run = train(model.path(), &data, out.path(), 3, &extra);
    assert_success(&run);
    let summary: serde_json::Value = serde_json::from_str(&text(&run.stdout)).expect("train json");
    let before = summary["eval_loss_before"]
        .as_f64()
        .expect("eval_loss_before");
    let after = summary["eval_loss_after"]
        .as_f64()
        .expect("eval_loss_after");
    assert!(
        (before - after).abs() > 1e-3,
        "training the embedding did not change the loss: {summary}"
    );

    let weights = out.path().join("trained_weights.safetensors");
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
        s(&weights),
        "--json",
    ]);
    assert_success(&run);
    let json: serde_json::Value = serde_json::from_str(&text(&run.stdout)).expect("eval json");
    let loss = json["loss"].as_f64().expect("loss");
    // Both runs evaluate the same f32 weights through the same code; a stale
    // host copy of the table would give `before` instead.
    assert!(
        (loss - after).abs() < 1e-4,
        "eval {loss} vs end of train {after} (before training {before})"
    );

    let mut extra = embed.to_vec();
    extra.extend_from_slice(&["--resume", s(&weights)]);
    assert_success(&train(model.path(), &data, out.path(), 5, &extra));
    let state = out.path().join("training_state.json");
    assert_eq!(read_json(&state)["step"], serde_json::json!(5));
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

    // With --json the text streams to stderr and stdout holds only the object.
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
        "--json",
    ]);
    assert_success(&run);
    let json: serde_json::Value = serde_json::from_str(&text(&run.stdout)).expect("generate json");
    assert_eq!(json["prompt_tokens"], 3, "{json}");
    assert_eq!(json["kv_cache"], true, "{json}");
    assert!(
        json["new_tokens"].as_u64().is_some_and(|n| n <= 4),
        "{json}"
    );
    let generated = json["text"].as_str().expect("text");
    let stderr = text(&run.stderr);
    assert!(stderr.contains(generated), "{json}\nstderr: {stderr}");
}

/// Checks the kernel table under `report`: none when the adapter has no
/// timestamp queries (`kernel_timing` null), else named rows, longest first,
/// that sum to `kernel_gpu_seconds{suffix}`.
fn assert_kernels(report: &serde_json::Value, suffix: &str) {
    let kernels = report["kernels"].as_array().expect("kernels is an array");
    let Some(mode) = report["kernel_timing"].as_str() else {
        assert!(kernels.is_empty(), "{report}");
        return;
    };
    assert!(
        ["inside_passes", "pass_boundaries"].contains(&mode),
        "{report}"
    );
    assert!(!kernels.is_empty(), "{report}");
    let seconds: Vec<f64> = kernels
        .iter()
        .map(|k| {
            assert!(k["name"].as_str().is_some_and(|n| !n.is_empty()), "{k}");
            assert!(k[format!("dispatches{suffix}")].as_f64() > Some(0.0), "{k}");
            k[format!("gpu_seconds{suffix}")]
                .as_f64()
                .expect("gpu_seconds is a number")
        })
        .collect();
    assert!(seconds.windows(2).all(|w| w[0] >= w[1]), "{report}");
    let total = report[format!("kernel_gpu_seconds{suffix}")]
        .as_f64()
        .expect("kernel_gpu_seconds is a number");
    assert!(
        (seconds.iter().sum::<f64>() - total).abs() < 1e-9,
        "{report}"
    );
}

#[test]
#[ignore = "needs a GPU"]
fn generate_profile_times_the_prefill_and_each_decode_token() {
    let model = model_dir("generate-profile");
    let json = greedy_generate(model.path(), "w5 w6 w7", "4", &["--raw", "--profile"]);
    let profile = &json["profile"];
    let with_mode = |section: &serde_json::Value| {
        let mut section = section.clone();
        section["kernel_timing"] = profile["kernel_timing"].clone();
        section
    };
    let prefill = with_mode(&profile["prefill"]);
    let decode = with_mode(&profile["decode"]);
    assert_kernels(&prefill, "");
    // A forward pass follows each new token but the last one before
    // --max-new-tokens.
    let new_tokens = json["new_tokens"].as_u64().expect("new_tokens");
    let decode_tokens = decode["tokens"].as_u64().expect("decode tokens");
    assert!(
        decode_tokens + 1 == new_tokens || decode_tokens == new_tokens,
        "{json}"
    );
    if decode_tokens > 0 {
        assert_kernels(&decode, "_per_token");
    }
    assert!(prefill["gpu_wait_seconds"].as_f64() >= Some(0.0), "{json}");
    assert!(
        decode["gpu_wait_seconds_per_token"].as_f64() >= Some(0.0),
        "{json}"
    );
    let plain = greedy_generate(model.path(), "w5 w6 w7", "4", &["--raw"]);
    assert_eq!(plain.get("profile"), None, "{plain}");
    assert_eq!(
        plain["text"], json["text"],
        "timestamps do not change tokens"
    );
}

/// Greedy `generate --json` of up to `max_new` tokens after `prompt`.
fn greedy_generate(model: &Path, prompt: &str, max_new: &str, extra: &[&str]) -> serde_json::Value {
    let mut args = vec![
        "generate",
        "--model",
        s(model),
        "--prompt",
        prompt,
        "--max-new-tokens",
        max_new,
        "--temperature",
        "0",
        "--json",
    ];
    args.extend_from_slice(extra);
    let run = ferrotensor(&args);
    assert_success(&run);
    serde_json::from_str(&text(&run.stdout)).expect("generate json")
}

fn words(json: &serde_json::Value) -> Vec<String> {
    let text = json["text"].as_str().expect("text");
    text.split_whitespace().map(str::to_owned).collect()
}

#[test]
#[ignore = "needs a GPU"]
fn greedy_tokens_do_not_depend_on_where_generation_stops() {
    let model = model_dir("generate-prefix");
    let short = greedy_generate(model.path(), "w5 w6 w7", "3", &["--raw"]);
    let long = greedy_generate(model.path(), "w5 w6 w7", "6", &["--raw"]);
    let uncached = greedy_generate(model.path(), "w5 w6 w7", "6", &["--raw", "--no-kv-cache"]);
    // Each id of the word-level vocabulary decodes to its own word.
    assert_eq!(words(&long), words(&uncached), "{long}\n{uncached}");
    assert!(words(&long).starts_with(&words(&short)), "{short}\n{long}");
    assert_eq!(
        as_count(&short["new_tokens"]),
        as_count(&long["new_tokens"]).min(3),
        "{short}\n{long}"
    );
    for json in [&short, &long, &uncached] {
        let speed = &json["tokens_per_second"];
        assert!(speed.is_number() || speed.is_null(), "{json}");
        assert!(
            json["prefill_seconds"].as_f64().is_some_and(|s| s >= 0.0),
            "{json}"
        );
    }
}

/// Zeroes the final norm weight of the tiny model in `dir`, so every logit
/// is 0 and greedy decoding picks the last id, 63.
fn zero_final_norm(dir: &Path) {
    let path = dir.join("model.safetensors");
    let bytes = std::fs::read(&path).expect("read weights");
    let weights = safetensors::SafeTensors::deserialize(&bytes).expect("parse weights");
    let name = "model.norm.weight";
    let len = weights.tensor(name).expect("final norm").data().len();
    let zeros = vec![0u8; len];
    let tensors: Vec<_> = weights
        .tensors()
        .into_iter()
        .map(|(n, view)| {
            if n == name {
                let shape = vec![len / 4];
                let zero =
                    safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape, &zeros)
                        .expect("zero view");
                (n, zero)
            } else {
                (n, view)
            }
        })
        .collect();
    let out = safetensors::tensor::serialize(tensors, &None).expect("serialize");
    std::fs::write(&path, out).expect("write weights");
}

#[test]
#[ignore = "needs a GPU"]
fn a_chatml_prompt_stops_at_im_end_when_the_config_names_another_eos() {
    // Without the ChatML tokens id 63 is the word w63, and nothing stops it.
    let plain = model_dir("stop-plain");
    zero_final_norm(plain.path());
    let run = greedy_generate(plain.path(), "w5 w6", "4", &[]);
    assert_eq!(words(&run), ["w63"; 4], "{run}");

    let model = chatml_model_dir("stop-chatml");
    zero_final_norm(model.path());
    let config_path = model.path().join("config.json");
    let mut config = read_json(&config_path);
    config["eos_token_id"] = serde_json::json!([2, 0]);
    std::fs::write(&config_path, config.to_string()).expect("write config");

    let templated = greedy_generate(model.path(), "w5 w6", "4", &[]);
    assert_eq!(templated["new_tokens"], 0, "{templated}");
    assert_eq!(templated["text"], "", "{templated}");
    // The only sampled token came from the prefill: no decode step ran.
    assert!(templated["tokens_per_second"].is_null(), "{templated}");
    assert!(
        templated["prefill_seconds"]
            .as_f64()
            .is_some_and(|s| s >= 0.0),
        "{templated}"
    );

    let replies = chat_replies(&chat(
        model.path(),
        &["--json", "--temperature", "0"],
        "w5 w6\nw7\n",
    ));
    assert_eq!(replies.len(), 2);
    for reply in &replies {
        assert_eq!(reply["new_tokens"], 0, "{reply}");
        assert_eq!(reply["reply"], "", "{reply}");
    }

    // --raw keeps its stop set, whose tokenizer EOS the loader takes to be
    // <|im_end|> whenever the vocabulary has it.
    let raw = greedy_generate(model.path(), "w5 w6", "4", &["--raw"]);
    assert_eq!(raw["new_tokens"], 0, "{raw}");
}

#[test]
#[ignore = "needs a GPU"]
fn a_tokenizer_that_cannot_encode_the_prompt_exits_with_1_and_names_the_error() {
    let model = model_dir("generate-encoder-fails");
    // An `unk_token` missing from the vocabulary makes the word-level model
    // fail on any word it does not know.
    let path = model.path().join("tokenizer.json");
    let mut json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("read tokenizer"))
            .expect("parse tokenizer");
    json["model"]["unk_token"] = serde_json::json!("<missing>");
    std::fs::write(&path, json.to_string()).expect("write tokenizer");

    let run = ferrotensor(&[
        "generate",
        "--model",
        s(model.path()),
        "--prompt",
        "not-a-word",
        "--raw",
        "--json",
    ]);
    let stdout = text(&run.stdout);
    assert_eq!(run.status.code(), Some(1), "stdout: {stdout}");
    let error: serde_json::Value = serde_json::from_str(&stdout).expect("one JSON object");
    let message = error["error"].as_str().expect("error message");
    assert!(
        message.contains("tokenizer cannot encode the text") && message.contains("[UNK]"),
        "{message}"
    );
}

#[test]
#[ignore = "needs a GPU"]
fn generate_refuses_a_prompt_that_encodes_to_no_tokens() {
    let model = model_dir("generate-empty-prompt");
    let run = ferrotensor(&[
        "generate",
        "--model",
        s(model.path()),
        "--prompt",
        "",
        "--raw",
    ]);
    let stderr = text(&run.stderr);
    assert_eq!(run.status.code(), Some(2), "stderr: {stderr}");
    assert!(
        stderr.contains("the prompt encodes to no tokens"),
        "stderr: {stderr}"
    );
}

#[test]
#[ignore = "needs a GPU"]
fn generate_clips_new_tokens_to_the_context_and_refuses_a_longer_prompt() {
    let model = model_dir("generate-context");
    let config_path = model.path().join("config.json");
    let mut config = read_json(&config_path);
    config["max_position_embeddings"] = serde_json::json!(10);
    std::fs::write(&config_path, config.to_string()).expect("write config");

    // 3 prompt tokens leave room for 7 of the 20 requested.
    let run = ferrotensor(&[
        "generate",
        "--model",
        s(model.path()),
        "--prompt",
        "w5 w6 w7",
        "--raw",
        "--max-new-tokens",
        "20",
        "--seed",
        "1",
        "--json",
    ]);
    assert_success(&run);
    let stderr = text(&run.stderr);
    assert!(
        stderr.contains("lowered --max-new-tokens from 20 to 7 to fit the model's 10 positions"),
        "stderr: {stderr}"
    );
    let json: serde_json::Value = serde_json::from_str(&text(&run.stdout)).expect("generate json");
    assert!(
        json["new_tokens"].as_u64().is_some_and(|n| n <= 7),
        "{json}"
    );

    let long = ["w9"; 11].join(" ");
    let run = ferrotensor(&[
        "generate",
        "--model",
        s(model.path()),
        "--prompt",
        &long,
        "--raw",
    ]);
    let stderr = text(&run.stderr);
    assert_eq!(run.status.code(), Some(2), "stderr: {stderr}");
    assert!(
        stderr.contains("the prompt is 11 tokens, longer than the model's 10 positions"),
        "stderr: {stderr}"
    );
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
    // weights 1 + adapters 1 + activations 15 + logits 1 + overhead 1600.
    assert_eq!(plan["estimated_peak_mib"], 1618);
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
    // weights 1 + adapters 1 + activations 1 + logits 1 + overhead 1600.
    assert_eq!(plan["estimated_peak_mib"], 1604);
    let notes = plan["notes"].as_array().expect("notes array");
    assert!(
        notes
            .iter()
            .filter_map(|n| n.as_str())
            .any(|n| n
                .contains("seq 128 cuts every example longer than 128 tokens; pass --seq-len N")),
        "{notes:?}"
    );
}

#[test]
fn plan_without_a_model_lists_the_documented_sizes() {
    let (code, plan) = plan_json(&["plan", "--json", "--memory-mib", "8000"]);
    assert_eq!(code, Some(0), "{plan}");
    let models = plan["models"].as_array().expect("models array");
    let names: Vec<&str> = models.iter().filter_map(|m| m["model"].as_str()).collect();
    assert_eq!(names, ["SmolLM2-135M", "SmolLM2-360M", "SmolLM2-1.7B"]);
    assert_eq!(models[0]["chosen"]["batch_size"], 8);
    assert_eq!(models[0]["chosen"]["seq_len"], 256);
    assert_eq!(models[0]["estimated_peak_mib"], 6516);
    assert_eq!(models[2]["chosen"]["batch_size"], 4);
    assert_eq!(models[2]["chosen"]["seq_len"], 256);
    assert_eq!(models[2]["chosen"]["checkpoint_blocks"], true);
    assert_eq!(models[2]["estimated_peak_mib"], 6325);
    assert_eq!(models[2]["fits"], true);
}

/// Run the binary with `HF_HUB_CACHE` set to `cache` and `dir` as the working
/// directory.
fn ferrotensor_with_cache(args: &[&str], cache: &Path, dir: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_ferrotensor"))
        .args(args)
        .env("HF_HUB_CACHE", cache)
        .current_dir(dir)
        .output()
        .expect("run the ferrotensor binary")
}

/// Run `plan --model model` against the cache `cache`.
fn plan_model_in(model: &str, cache: &Path) -> Output {
    let args = ["plan", "--model", model, "--memory-mib", "100000", "--json"];
    ferrotensor_with_cache(&args, cache, cache)
}

/// A snapshot of `org/name` in the cache `cache`, pinned by `refs/main`, with a
/// model in it unless `complete` is false.
fn cached_snapshot(cache: &Path, complete: bool) -> std::path::PathBuf {
    let folder = cache.join("models--org--name");
    let snapshot = folder.join("snapshots/0123abcd");
    std::fs::create_dir_all(&snapshot).expect("create snapshot");
    std::fs::create_dir_all(folder.join("refs")).expect("create refs");
    std::fs::write(folder.join("refs/main"), "0123abcd\n").expect("write refs/main");
    if complete {
        common::write_tiny_llama(&snapshot);
        common::write_tiny_tokenizer(&snapshot);
    }
    snapshot
}

#[test]
fn a_cached_model_id_resolves_to_its_snapshot() {
    let cache = common::TempDir::new("hf-cache");
    let snapshot = cached_snapshot(cache.path(), true);
    let run = plan_model_in("org/name", cache.path());
    assert_success(&run);
    let stderr = text(&run.stderr);
    assert!(
        stderr.contains(&format!("model: org/name ({})\n", snapshot.display())),
        "stderr: {stderr}"
    );
    let plan: serde_json::Value = serde_json::from_slice(&run.stdout).expect("plan json");
    assert_eq!(plan["model"], "org/name");
    assert_eq!(plan["parameters"], 20640);
}

#[test]
fn an_existing_directory_wins_over_the_cache() {
    let cache = common::TempDir::new("hf-cache-dir-wins");
    cached_snapshot(cache.path(), true);
    let local = cache.path().join("org/name");
    std::fs::create_dir_all(&local).expect("create model dir");
    common::write_tiny_llama(&local);
    let run = plan_model_in("org/name", cache.path());
    assert_success(&run);
    assert!(!text(&run.stderr).contains("model: org/name"));
}

#[test]
fn an_uncached_model_id_exits_with_2_and_says_how_to_fetch_it() {
    let cache = common::TempDir::new("hf-cache-empty");
    let run = plan_model_in("org/name", cache.path());
    let stdout = text(&run.stdout);
    assert_eq!(run.status.code(), Some(2), "stdout: {stdout}");
    let json: serde_json::Value = serde_json::from_str(&stdout).expect("error json");
    let error = json["error"].as_str().expect("error string");
    assert!(error.contains("hf download org/name"), "{error}");
    assert!(error.contains(s(cache.path())), "{error}");
}

#[test]
fn a_cached_snapshot_without_config_json_exits_with_2() {
    let cache = common::TempDir::new("hf-cache-incomplete");
    let snapshot = cached_snapshot(cache.path(), false);
    let run = plan_model_in("org/name", cache.path());
    let stdout = text(&run.stdout);
    assert_eq!(run.status.code(), Some(2), "stdout: {stdout}");
    let json: serde_json::Value = serde_json::from_str(&stdout).expect("error json");
    let error = json["error"].as_str().expect("error string");
    assert!(error.contains(s(&snapshot)), "{error}");
    assert!(error.contains("download looks incomplete"), "{error}");
}

#[test]
#[ignore = "needs a GPU"]
fn finetune_on_a_model_id_announces_it_once_and_prints_the_next_commands() {
    let cache = common::TempDir::new("hf-cache-finetune");
    cached_snapshot(cache.path(), true);
    let out = common::TempDir::new("hf-finetune-out");
    let data = out.path().join("tokens.bin");
    common::write_token_bin(&data, 400);
    let run = ferrotensor_with_cache(
        &[
            "finetune",
            "--model",
            "org/name",
            "--data",
            s(&data),
            "--task",
            "pretrain",
            "--seq-len",
            "16",
            "--out",
            s(out.path()),
            "--max-steps",
            "2",
            "--lora-rank",
            "4",
            "--memory-mib",
            "100000",
        ],
        cache.path(),
        out.path(),
    );
    assert_success(&run);
    let stderr = text(&run.stderr);
    assert_eq!(stderr.matches("model: org/name (").count(), 1, "{stderr}");
    let adapter = out.path().join("lora_adapters.safetensors");
    let next = format!(
        "next:\n  ferrotensor chat --model org/name --lora {adapter}\n  \
         ferrotensor merge --model org/name --lora {adapter} --out {out}-merged\n",
        adapter = adapter.display(),
        out = out.path().display()
    );
    assert!(stderr.contains(&next), "stderr: {stderr}");
}

/// Run `finetune --task pretrain --seq-len 16 --max-steps 2` on `data`,
/// writing into `out`; `extra` is appended to the arguments.
fn finetune(model: &Path, data: &Path, out: &Path, extra: &[&str]) -> Output {
    let mut args = vec![
        "finetune",
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
        "2",
        "--log-every",
        "1",
        "--lora-rank",
        "4",
    ];
    args.extend_from_slice(extra);
    ferrotensor(&args)
}

#[test]
#[ignore = "needs a GPU"]
fn finetune_trains_with_the_plan_and_writes_the_train_files() {
    let model = model_dir("finetune-model");
    let out = common::TempDir::new("finetune-out");
    let data = out.path().join("tokens.bin");
    common::write_token_bin(&data, 400);

    let run = finetune(
        model.path(),
        &data,
        out.path(),
        &["--memory-mib", "100000", "--json"],
    );
    assert_success(&run);
    let json: serde_json::Value = serde_json::from_str(&text(&run.stdout)).expect("finetune json");
    assert_eq!(json["steps"], 2, "{json}");
    assert_eq!(json["fits"], true, "{json}");
    assert_eq!(json["budget_mib"], 100000, "{json}");
    assert_eq!(json["grad_accum"], 1, "{json}");
    assert_eq!(json["chosen"]["batch_size"], 8, "{json}");
    assert_eq!(json["chosen"]["seq_len"], 16, "{json}");
    assert_eq!(json["out_dir"], s(out.path()), "{json}");
    assert!(
        json["final_loss"].as_f64().is_some_and(f64::is_finite),
        "{json}"
    );
    assert!(
        json["measured_peak_mib"]
            .as_u64()
            .is_some_and(|mib| mib > 0),
        "{json}"
    );
    let stderr = text(&run.stderr);
    assert!(
        stderr.contains(
            "finetune: --batch-size 8 (plan), --grad-accum 1 (plan), --seq-len 16 (user), \
             --checkpoint-blocks off (plan), f16 weights (default)"
        ),
        "stderr: {stderr}"
    );
    assert!(
        stderr.contains("memory: estimated peak "),
        "stderr: {stderr}"
    );
    for name in RUN_FILES {
        let path = out.path().join(name);
        let meta = std::fs::metadata(&path)
            .unwrap_or_else(|e| panic!("{} is missing: {e}", path.display()));
        assert!(meta.len() > 0, "{} is empty", path.display());
    }
    let state = read_json(&out.path().join("training_state.json"));
    assert_eq!(
        (
            &state["step"],
            &state["batch_size"],
            &state["grad_accum_steps"],
            &state["sequence_length"]
        ),
        (
            &serde_json::json!(2),
            &serde_json::json!(8),
            &serde_json::json!(1),
            &serde_json::json!(16)
        )
    );
}

#[test]
fn finetune_exits_with_2_and_does_not_train_when_nothing_fits() {
    let model = model_dir("finetune-too-small");
    let out = common::TempDir::new("finetune-too-small-out");
    let data = out.path().join("tokens.bin");
    common::write_token_bin(&data, 400);

    let run = finetune(model.path(), &data, out.path(), &["--memory-mib", "1"]);
    let stderr = text(&run.stderr);
    assert_eq!(run.status.code(), Some(2), "stderr: {stderr}");
    // weights 1 + adapters 1 + activations 1 + logits 1 + overhead 1600;
    // 1604 / 0.85 rounds up to 1888.
    assert!(
        stderr.contains(
            "error: no configuration fits the 1 MiB budget; the smallest needs 1604 MiB, \
             which fits a budget of 1888 MiB or more; pass --force to train anyway"
        ),
        "stderr: {stderr}"
    );
    assert!(
        stderr.contains("plan      batch 1, seq 16"),
        "stderr: {stderr}"
    );
    assert!(
        !out.path().join("lora_adapters.safetensors").exists(),
        "an adapter was written"
    );

    let run = finetune(
        model.path(),
        &data,
        out.path(),
        &["--memory-mib", "1", "--json"],
    );
    let stdout = text(&run.stdout);
    let json: serde_json::Value =
        serde_json::from_str(&stdout).unwrap_or_else(|e| panic!("{e}\nstdout: {stdout}"));
    assert_eq!(run.status.code(), Some(2), "{json}");
    assert!(
        json["error"]
            .as_str()
            .is_some_and(|e| e.starts_with("no configuration fits the 1 MiB budget")),
        "{json}"
    );
}

#[test]
#[ignore = "needs a GPU"]
fn finetune_force_trains_the_smallest_configuration() {
    let model = model_dir("finetune-force");
    let out = common::TempDir::new("finetune-force-out");
    let data = out.path().join("tokens.bin");
    common::write_token_bin(&data, 400);

    let run = finetune(
        model.path(),
        &data,
        out.path(),
        &["--memory-mib", "1", "--force"],
    );
    assert_success(&run);
    let stderr = text(&run.stderr);
    assert!(
        stderr.contains("training anyway (--force)"),
        "stderr: {stderr}"
    );
    assert!(
        stderr.contains("--batch-size 1 (plan), --grad-accum 8 (plan)"),
        "stderr: {stderr}"
    );
    assert!(
        stderr.contains("block checkpointing: on"),
        "stderr: {stderr}"
    );
    let state = read_json(&out.path().join("training_state.json"));
    assert_eq!(
        (&state["step"], &state["batch_size"]),
        (&serde_json::json!(2), &serde_json::json!(1))
    );
}

#[test]
fn merge_writes_a_model_directory_and_refuses_a_non_empty_output() {
    let model = model_dir("merge-model");
    let tmp = common::TempDir::new("merge-out");
    // A rank-2 adapter on q_proj of both layers, with the metadata `train`
    // writes.
    let a = vec![0u8; common::HIDDEN * 2 * 4];
    let b = vec![0u8; 2 * common::HIDDEN * 4];
    let mut views = std::collections::HashMap::new();
    for l in 0..common::LAYERS {
        let p = format!("model.layers.{l}.self_attn.q_proj");
        for (suffix, shape, bytes) in [
            ("lora_A", vec![common::HIDDEN, 2], &a),
            ("lora_B", vec![2, common::HIDDEN], &b),
        ] {
            let view = safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape, bytes)
                .expect("tensor view");
            views.insert(format!("{p}.{suffix}.weight"), view);
        }
    }
    let metadata = std::collections::HashMap::from([
        ("format".to_string(), "ferrotensor-lora".to_string()),
        ("rank".to_string(), "2".to_string()),
        ("alpha".to_string(), "4".to_string()),
        ("target_modules".to_string(), "q_proj".to_string()),
    ]);
    let adapter = tmp.path().join("adapter.safetensors");
    std::fs::write(
        &adapter,
        safetensors::tensor::serialize(views, &Some(metadata)).expect("serialize"),
    )
    .expect("write adapter");

    let merged = tmp.path().join("merged");
    let merge = |out: &Path| {
        ferrotensor(&[
            "merge",
            "--model",
            s(model.path()),
            "--lora",
            s(&adapter),
            "--out",
            s(out),
        ])
    };
    let run = merge(&merged);
    assert_success(&run);
    assert!(
        text(&run.stdout).contains("(2 tensors merged, 18 copied)"),
        "stdout: {}",
        text(&run.stdout)
    );
    for name in ["model.safetensors", "config.json", "tokenizer.json"] {
        assert!(merged.join(name).is_file(), "{name} is missing");
    }

    for out in [merged.as_path(), model.path()] {
        let run = merge(out);
        assert_eq!(run.status.code(), Some(2), "invalid input exits with 2");
        let stderr = text(&run.stderr);
        assert!(stderr.contains("is not empty"), "stderr: {stderr}");
    }

    let json_out = tmp.path().join("merged-json");
    let run = ferrotensor(&[
        "merge",
        "--model",
        s(model.path()),
        "--lora",
        s(&adapter),
        "--out",
        s(&json_out),
        "--json",
    ]);
    assert_success(&run);
    let report: serde_json::Value = serde_json::from_slice(&run.stdout).expect("one JSON object");
    assert_eq!(
        report,
        serde_json::json!({ "out_dir": s(&json_out), "merged": 2, "copied": 18 })
    );
}

#[test]
fn export_gguf_refuses_a_word_level_tokenizer_and_an_existing_file_with_2() {
    let model = model_dir("gguf-model");
    let config_path = model.path().join("config.json");
    let mut config = read_json(&config_path);
    config["max_position_embeddings"] = 64.into();
    std::fs::write(&config_path, config.to_string()).expect("write config");
    let out = model.path().join("model.gguf");
    let run = ferrotensor(&[
        "export-gguf",
        "--model",
        s(model.path()),
        "--out",
        s(&out),
        "--json",
    ]);
    assert_eq!(run.status.code(), Some(2), "stderr: {}", text(&run.stderr));
    let report: serde_json::Value = serde_json::from_slice(&run.stdout).expect("one JSON object");
    assert_eq!(
        report["error"],
        "tokenizer.json has a \"WordLevel\" model; export-gguf needs a byte-level BPE tokenizer"
    );
    assert!(!out.exists());

    std::fs::write(&out, b"keep").expect("write file");
    let run = ferrotensor(&["export-gguf", "--model", s(model.path()), "--out", s(&out)]);
    assert_eq!(run.status.code(), Some(2));
    let stderr = text(&run.stderr);
    assert!(
        stderr.contains("model.gguf exists; export-gguf writes a new file"),
        "stderr: {stderr}"
    );
    assert_eq!(std::fs::read(&out).expect("read file"), b"keep");
}

#[test]
fn chat_on_a_missing_model_exits_with_2() {
    let tmp = common::TempDir::new("chat-missing");
    let missing = tmp.path().join("does-not-exist");
    let out = ferrotensor(&["chat", "--model", s(&missing)]);
    let stderr = text(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "stderr: {stderr}");
    assert!(stderr.contains("does not exist"), "stderr: {stderr}");

    // One compact line, like the reply objects.
    let out = ferrotensor(&["chat", "--model", s(&missing), "--json"]);
    assert_eq!(out.status.code(), Some(2));
    let stdout = text(&out.stdout);
    assert_eq!(stdout.lines().count(), 1, "stdout: {stdout}");
    let json: serde_json::Value = serde_json::from_str(&stdout).expect("error json");
    let error = json["error"].as_str().expect("error string");
    assert!(error.contains(s(&missing)), "{error}");
}

/// A temp directory holding a tiny model whose tokenizer has the ChatML
/// tokens.
fn chatml_model_dir(tag: &str) -> common::TempDir {
    let dir = common::TempDir::new(tag);
    common::write_tiny_llama(dir.path());
    common::write_tiny_chatml_tokenizer(dir.path());
    dir
}

/// Run `chat` with up to 4 new tokens per reply on the lines of `input`.
fn chat(model: &Path, extra: &[&str], input: &str) -> Output {
    let mut args = vec![
        "chat",
        "--model",
        s(model),
        "--max-new-tokens",
        "4",
        "--seed",
        "1",
    ];
    args.extend_from_slice(extra);
    ferrotensor_with_input(&args, input)
}

#[test]
#[ignore = "needs a GPU"]
fn chat_answers_each_line_until_exit() {
    let model = chatml_model_dir("chat-model");
    // The line after /exit is never read.
    let run = chat(model.path(), &[], "w5 w6\nw7\n/exit\nw8\n");
    assert_success(&run);
    let stderr = text(&run.stderr);
    // <|im_start|> user w5 w6 <|im_end|> <|im_start|> assistant, where the
    // tiny vocabulary reads "user" and "assistant" as <unk>.
    assert!(stderr.contains("[7 prompt + "), "stderr: {stderr}");
    assert_eq!(
        stderr.matches(" new tokens, ").count(),
        2,
        "stderr: {stderr}"
    );

    // After /reset the same line gives the same prompt again.
    let run = chat(model.path(), &["--json"], "w5 w6\n/reset\nw5 w6\n");
    assert_success(&run);
    let stdout = text(&run.stdout);
    let replies: Vec<serde_json::Value> = stdout
        .lines()
        .map(|line| serde_json::from_str(line).expect("one JSON object per line"))
        .collect();
    assert_eq!(replies.len(), 2, "stdout: {stdout}");
    let stderr = text(&run.stderr);
    for reply in &replies {
        let mut keys: Vec<&str> = reply
            .as_object()
            .expect("object")
            .keys()
            .map(String::as_str)
            .collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            [
                "cached_tokens",
                "new_tokens",
                "prefill_seconds",
                "prompt_tokens",
                "reply",
                "tokens_per_second"
            ]
        );
        assert_eq!(reply["prompt_tokens"], 7, "{reply}");
        assert!(
            reply["new_tokens"].as_u64().is_some_and(|n| n <= 4),
            "{reply}"
        );
        let streamed = reply["reply"].as_str().expect("reply text");
        assert!(stderr.contains(streamed), "{reply}\nstderr: {stderr}");
    }
    // The cache still holds the first prompt; only its last token runs again.
    assert_eq!(replies[0]["cached_tokens"], 0);
    assert_eq!(replies[1]["cached_tokens"], 6);
}

/// The reply objects a `chat --json` run printed, one per stdout line.
fn chat_replies(run: &Output) -> Vec<serde_json::Value> {
    assert_success(run);
    text(&run.stdout)
        .lines()
        .map(|line| serde_json::from_str(line).expect("one JSON object per line"))
        .collect()
}

/// Greedy replies of `chat` on `input` with the KV cache kept across turns,
/// after checking them against a run of the whole conversation every turn.
fn kept_cache_replies(model: &Path, input: &str) -> Vec<serde_json::Value> {
    let greedy = ["--json", "--temperature", "0"];
    let kept = chat_replies(&chat(model, &greedy, input));
    let rerun_flags = [&greedy[..], &["--rerun-conversation"][..]].concat();
    let rerun = chat_replies(&chat(model, &rerun_flags, input));
    assert_eq!(kept.len(), rerun.len());
    // Each id of the word-level vocabulary decodes to its own word, so equal
    // replies are equal token ids.
    for (turn, (k, r)) in kept.iter().zip(&rerun).enumerate() {
        for key in ["reply", "new_tokens", "prompt_tokens"] {
            assert_eq!(k[key], r[key], "turn {turn}: {k} vs {r}");
        }
        assert_eq!(r["cached_tokens"], 0, "turn {turn}: {r}");
    }
    kept
}

/// A token count from a reply object.
fn as_count(value: &serde_json::Value) -> u64 {
    value.as_u64().expect("a token count")
}

#[test]
#[ignore = "needs a GPU"]
fn chat_with_the_kept_kv_cache_replies_like_a_full_rerun() {
    let model = chatml_model_dir("chat-cache");
    let kept = kept_cache_replies(model.path(), "w5 w6\nw7\nw8 w9 w10\nw11\n");
    assert_eq!(kept.len(), 4);
    assert!(
        kept.iter().any(|r| as_count(&r["new_tokens"]) > 0),
        "every reply is empty: {kept:?}"
    );
    assert_eq!(kept[0]["cached_tokens"], 0);
    // The cache holds the previous prompt and every reply token run after
    // it; a reply that reached --max-new-tokens 4 never ran its last token.
    for pair in kept.windows(2) {
        let prompt = as_count(&pair[0]["prompt_tokens"]);
        let new = as_count(&pair[0]["new_tokens"]);
        let held = prompt + new - u64::from(new == 4);
        let cached = as_count(&pair[1]["cached_tokens"]);
        assert!(
            (prompt..=held).contains(&cached),
            "{cached} cached tokens after {pair:?}"
        );
    }
}

#[test]
#[ignore = "needs a GPU"]
fn chat_rebuilds_the_kv_cache_after_dropping_turns() {
    let model = chatml_model_dir("chat-cache-trim");
    let config_path = model.path().join("config.json");
    let mut config = read_json(&config_path);
    config["max_position_embeddings"] = serde_json::json!(18);
    std::fs::write(&config_path, config.to_string()).expect("write config");
    // 14 prompt positions: turns 2 and 3 each drop the exchange before them.
    let kept = kept_cache_replies(model.path(), "w5 w6\nw7 w8 w9\nw10\n");
    let counts = |key: &str| kept.iter().map(|r| as_count(&r[key])).collect::<Vec<_>>();
    assert_eq!(counts("prompt_tokens"), [7, 8, 6], "{kept:?}");
    // After a drop only `<|im_start|>user` still starts the conversation.
    assert_eq!(counts("cached_tokens"), [0, 2, 2], "{kept:?}");
}

#[test]
#[ignore = "needs a GPU"]
fn chat_drops_the_oldest_turns_and_refuses_a_message_that_cannot_fit() {
    let model = chatml_model_dir("chat-trim");
    let config_path = model.path().join("config.json");
    let mut config = read_json(&config_path);
    config["max_position_embeddings"] = serde_json::json!(18);
    std::fs::write(&config_path, config.to_string()).expect("write config");
    // 18 positions less 4 new tokens leave 14 for the prompt. The second
    // message alone is 8 tokens; the third alone is 20.
    let long = vec!["w9"; 15].join(" ");
    let run = chat(model.path(), &[], &format!("w5 w6\nw7 w8 w9\n{long}\n"));
    let stderr = text(&run.stderr);
    assert_eq!(run.status.code(), Some(2), "stderr: {stderr}");
    assert!(stderr.contains("[7 prompt + "), "stderr: {stderr}");
    assert!(stderr.contains("[8 prompt + "), "stderr: {stderr}");
    assert_eq!(
        stderr.matches("dropped the oldest turns").count(),
        1,
        "stderr: {stderr}"
    );
    assert!(
        stderr.contains("does not fit the model's 18 positions"),
        "stderr: {stderr}"
    );
}

#[test]
#[ignore = "needs a GPU"]
fn chat_refuses_a_tokenizer_without_chatml_tokens() {
    let model = model_dir("chat-no-chatml");
    let run = ferrotensor_with_input(&["chat", "--model", s(model.path())], "");
    let stderr = text(&run.stderr);
    assert_eq!(run.status.code(), Some(2), "stderr: {stderr}");
    assert!(stderr.contains("ChatML models only"), "stderr: {stderr}");
    assert!(stderr.contains("generate --raw"), "stderr: {stderr}");
}
