//! `ferrotensor pretrain`: flag checks on the host, and on the GPU a tiny
//! model that learns a repeated sentence, continues it in `generate`, and
//! reloads from its directory with the logits it trained to.

#![allow(clippy::expect_used, clippy::panic)]

mod common;

use serde_json::Value;
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

/// The one JSON object a successful `--json` run prints.
fn json_of(out: &Output) -> Value {
    assert!(
        out.status.success(),
        "stdout: {}\nstderr: {}",
        text(&out.stdout),
        text(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).expect("one JSON object on stdout")
}

const SENTENCE: &str = "the quick brown fox jumps over the lazy dog.\n";

/// `pretrain` of a 2-block, hidden-64 model with a 64-byte context on
/// `corpus`, writing into `out`; `extra` is appended.
fn pretrain(corpus: &Path, out: &Path, extra: &[&str]) -> Output {
    let mut args = vec![
        "pretrain",
        "--data",
        s(corpus),
        "--out",
        s(out),
        "--layers",
        "2",
        "--hidden",
        "64",
        "--heads",
        "4",
        "--context",
        "64",
        "--json",
    ];
    args.extend_from_slice(extra);
    ferrotensor(&args)
}

#[test]
fn invalid_sizes_and_a_missing_corpus_exit_with_2() {
    let dir = common::TempDir::new("pretrain-flags");
    let corpus = dir.path().join("corpus.txt");
    std::fs::write(&corpus, SENTENCE.repeat(4)).expect("write corpus");
    let out = dir.path().join("model");
    for (extra, message) in [
        (&["--hidden", "30"][..], "is not a multiple of --heads"),
        (&["--hidden", "12"][..], "head size --hidden / --heads is 3"),
        (&["--hidden", "1032"][..], "is 258"),
        (&["--layers", "0"][..], "must be positive"),
        (
            &["--seq-len", "257"][..],
            "--seq-len must be between 2 and --context 256",
        ),
    ] {
        // The default sizes (hidden 128, 4 heads, context 256) plus `extra`.
        let mut args = vec!["pretrain", "--data", s(&corpus), "--out", s(&out), "--json"];
        args.extend_from_slice(extra);
        let run = ferrotensor(&args);
        assert_eq!(
            run.status.code(),
            Some(2),
            "{extra:?}: {}",
            text(&run.stderr)
        );
        let error = json_error(&run);
        assert!(error.contains(message), "{extra:?}: {error}");
    }
    let missing = pretrain(&dir.path().join("absent.txt"), &out, &[]);
    assert_eq!(missing.status.code(), Some(2));
    assert!(json_error(&missing).contains("absent.txt does not exist"));
    assert!(!out.exists(), "nothing is written for invalid input");

    std::fs::create_dir_all(&out).expect("create out");
    std::fs::write(out.join("model.safetensors"), b"").expect("write model");
    let taken = pretrain(&corpus, &out, &[]);
    assert_eq!(taken.status.code(), Some(2));
    assert!(json_error(&taken).contains("already holds a model; pass --resume"));
    let no_state = pretrain(&corpus, &out, &["--resume"]);
    assert_eq!(no_state.status.code(), Some(2));
    assert!(json_error(&no_state).contains("trained_weights.safetensors does not exist"));
}

/// A model directory without `model.safetensors` names `pretrain --resume`
/// when it holds the `init` directory of an unfinished run, and only then.
#[test]
fn an_unfinished_run_points_to_resume() {
    let dir = common::TempDir::new("pretrain-unfinished");
    common::write_tiny_llama(dir.path());
    common::write_tiny_tokenizer(dir.path());
    std::fs::remove_file(dir.path().join("model.safetensors")).expect("remove weights");
    let generate = || {
        let run = ferrotensor(&[
            "generate",
            "--model",
            s(dir.path()),
            "--raw",
            "--prompt",
            "the",
            "--json",
        ]);
        assert_eq!(run.status.code(), Some(1), "{}", text(&run.stderr));
        json_error(&run)
    };
    let plain = generate();
    assert!(plain.contains("no model.safetensors"), "{plain}");
    assert!(!plain.contains("--resume"), "{plain}");
    std::fs::create_dir(dir.path().join("init")).expect("create init");
    let unfinished = generate();
    assert!(
        unfinished.contains("unfinished `pretrain` run: `pretrain --resume`"),
        "{unfinished}"
    );
}

/// The `error` of the JSON object a failed `--json` run prints.
fn json_error(out: &Output) -> String {
    let value: Value = serde_json::from_slice(&out.stdout)
        .unwrap_or_else(|e| panic!("no JSON on stdout ({e}); stderr: {}", text(&out.stderr)));
    value["error"].as_str().expect("error string").to_string()
}

#[test]
#[ignore = "needs a GPU"]
fn pretrain_learns_a_repeated_sentence_and_the_directory_reloads() {
    let dir = common::TempDir::new("pretrain-overfit");
    let corpus = dir.path().join("corpus.txt");
    std::fs::write(&corpus, SENTENCE.repeat(100)).expect("write corpus");
    let model = dir.path().join("model");
    let summary = json_of(&pretrain(
        &corpus,
        &model,
        &[
            "--max-steps",
            "150",
            "--warmup-steps",
            "20",
            "--lr",
            "3e-3",
            "--log-every",
            "25",
        ],
    ));
    // 259 x 64 twice; per block four 64 x 64, three 256 x 64 and two norms
    // of 64; the final norm.
    assert_eq!(
        summary["parameters"],
        2 * 259 * 64 + 2 * (4 * 64 * 64 + 3 * 256 * 64 + 2 * 64) + 64
    );
    assert_eq!(summary["vocab_size"], 259);
    assert_eq!(summary["steps"], 150);
    let first = summary["first_loss"].as_f64().expect("first loss");
    let last = summary["final_loss"].as_f64().expect("final loss");
    eprintln!("pretrain loss: first logged {first:.4}, final {last:.4}");
    // Uniform guessing over 259 tokens costs ln 259 = 5.56. This test on
    // software Vulkan logged 3.91 at step 25, 0.041 at step 75 and 0.0114 at
    // step 150; the bound, 2% of ln 259 (0.111), leaves about a tenfold
    // margin and still fails a model that has not memorised the sentence.
    assert!(first > 1.0, "first logged loss {first}");
    assert!(last < 0.02 * 259f64.ln(), "final loss {last}");
    for file in [
        "config.json",
        "tokenizer.json",
        "model.safetensors",
        "tokens.bin",
        "trained_weights.safetensors",
    ] {
        assert!(model.join(file).is_file(), "{file}");
    }
    assert_eq!(
        std::fs::metadata(model.join("tokens.bin"))
            .expect("tokens.bin")
            .len(),
        4 * SENTENCE.len() as u64 * 100
    );

    let generated = json_of(&ferrotensor(&[
        "generate",
        "--model",
        s(&model),
        "--raw",
        "--prompt",
        "the quick brown",
        "--temperature",
        "0",
        "--max-new-tokens",
        "10",
        "--f32-weights",
        "--json",
    ]));
    let continuation = generated["text"].as_str().expect("text");
    eprintln!("greedy continuation: {continuation:?}");
    assert!(
        continuation.starts_with(" fox jum"),
        "continuation {continuation:?}"
    );

    // The trained weights as `train` saved them, applied on top of the
    // exported directory, and the exported directory alone, give one loss.
    let eval = |extra: &[&str]| {
        let tokens = model.join("tokens.bin");
        let mut args = vec![
            "eval",
            "--model",
            s(&model),
            "--data",
            s(&tokens),
            "--task",
            "pretrain",
            "--seq-len",
            "64",
            "--max-samples",
            "8",
            "--f32-weights",
            "--json",
        ];
        args.extend_from_slice(extra);
        json_of(&ferrotensor(&args))["loss"]
            .as_f64()
            .expect("eval loss")
    };
    let exported = eval(&[]);
    let trained = model.join("trained_weights.safetensors");
    let in_training_layout = eval(&["--lora", s(&trained)]);
    eprintln!("eval loss: exported {exported:.6}, trained weights {in_training_layout:.6}");
    assert!(
        (exported - in_training_layout).abs() < 1e-5,
        "exported {exported}, trained {in_training_layout}"
    );
    assert!(exported < 0.02 * 259f64.ln(), "exported loss {exported}");
}

#[test]
#[ignore = "needs a GPU"]
fn resume_continues_the_run_in_the_same_directory() {
    let dir = common::TempDir::new("pretrain-resume");
    let corpus = dir.path().join("corpus.txt");
    std::fs::write(&corpus, SENTENCE.repeat(20)).expect("write corpus");
    let model = dir.path().join("model");
    let first = json_of(&pretrain(&corpus, &model, &["--max-steps", "2"]));
    assert_eq!(first["steps"], 2);
    let resumed = pretrain(&corpus, &model, &["--max-steps", "4", "--resume"]);
    let stderr = text(&resumed.stderr);
    assert_eq!(json_of(&resumed)["steps"], 4);
    assert!(stderr.contains("continuing at step 2"), "{stderr}");
    assert!(
        !stderr.contains("init:"),
        "a resume keeps the model: {stderr}"
    );
}

/// A run stopped after its last checkpoint and before the export, as an
/// interrupt leaves it: `generate` refuses the directory instead of loading
/// random weights, and a resume at the saved step trains nothing and
/// exports the saved weights.
#[test]
#[ignore = "needs a GPU"]
fn resume_at_the_saved_step_exports_an_interrupted_run() {
    let dir = common::TempDir::new("pretrain-interrupted");
    let corpus = dir.path().join("corpus.txt");
    std::fs::write(&corpus, SENTENCE.repeat(20)).expect("write corpus");
    let model = dir.path().join("model");
    assert_eq!(
        json_of(&pretrain(&corpus, &model, &["--max-steps", "2"]))["steps"],
        2
    );
    let weights = model.join("model.safetensors");
    std::fs::remove_file(&weights).expect("remove the export");
    let metrics = std::fs::read(model.join("metrics.jsonl")).expect("metrics");

    let refused = ferrotensor(&[
        "generate",
        "--model",
        s(&model),
        "--raw",
        "--prompt",
        "the",
        "--json",
    ]);
    assert!(json_error(&refused).contains("pretrain --resume"));

    let recovered = pretrain(&corpus, &model, &["--max-steps", "2", "--resume"]);
    let stderr = text(&recovered.stderr);
    // Before the fix the loop took one more step (3) after skipping the
    // batches it had already seen.
    assert_eq!(json_of(&recovered)["steps"], 2, "{stderr}");
    assert!(stderr.contains("continuing at step 2"), "{stderr}");
    assert!(weights.is_file(), "{stderr}");
    assert_eq!(
        std::fs::read(model.join("metrics.jsonl")).expect("metrics"),
        metrics,
        "a resume at the saved step logs no step"
    );

    let eval = |extra: &[&str]| {
        let tokens = model.join("tokens.bin");
        let mut args = vec![
            "eval",
            "--model",
            s(&model),
            "--data",
            s(&tokens),
            "--task",
            "pretrain",
            "--seq-len",
            "64",
            "--max-samples",
            "4",
            "--f32-weights",
            "--json",
        ];
        args.extend_from_slice(extra);
        json_of(&ferrotensor(&args))["loss"]
            .as_f64()
            .expect("eval loss")
    };
    let exported = eval(&[]);
    let trained = model.join("trained_weights.safetensors");
    let in_training_layout = eval(&["--lora", s(&trained)]);
    assert!(
        (exported - in_training_layout).abs() < 1e-5,
        "exported {exported}, trained {in_training_layout}"
    );
}
