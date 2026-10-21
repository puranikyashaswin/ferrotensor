//! `ferrotensor-cli`: LoRA fine-tuning, evaluation and text generation for small
//! Llama-style (and LFM2) models on consumer GPUs.
//!
//! ```text
//! ferrotensor-cli train    --model DIR --data train.jsonl --out runs/sft
//! ferrotensor-cli eval     --model DIR --data val.jsonl --lora runs/sft/lora_adapters.safetensors
//! ferrotensor-cli generate --model DIR --lora runs/sft/lora_adapters.safetensors --prompt "Hi"
//! ```

use clap::{Args, Parser, Subcommand, ValueEnum};
use ferrotensor_autograd::{AdamW, WarmupCosineScheduler};
use ferrotensor_backend::{ComputeEngine, WgpuContext};
use ferrotensor_graph::lfm::Lfm2Model;
use ferrotensor_graph::model_builder::{Lfm2ModelBuilder, LoRaConfig};
use ferrotensor_loader::{Lfm2Config, ModelLoader, Sampler, Tokenizer, WeightsContext};
use ferrotensor_trainer::checkpoint::{
    apply_checkpoint, load_lora_checkpoint, save_lora_checkpoint, LoraMetadata, TrainingCheckpoint,
    TrainingState,
};
use ferrotensor_trainer::{
    build_sft_example, masked_cross_entropy_from_logits, DataLoader, Dataset, FileGradSyncConfig,
    TextDataset, Trainer,
};
use std::collections::HashMap;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Fine-tune, evaluate and sample from small language models on the GPU
/// (Metal, Vulkan or DX12 through wgpu).
///
/// Models are Hugging Face checkpoint directories with config.json,
/// tokenizer.json and safetensors weights (Llama-style models such as
/// SmolLM2 or TinyLlama).
#[derive(Parser, Debug)]
#[command(name = "ferrotensor-cli", version, propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// LoRA fine-tuning (SFT with response-only loss) or LM pretraining.
    Train(Box<TrainArgs>),
    /// Mean loss and perplexity of a model (plus optional adapter) on a dataset.
    Eval(EvalArgs),
    /// Generate text from a prompt (top-k / top-p / temperature sampling).
    #[command(visible_alias = "infer")]
    Generate(GenerateArgs),
    /// Simulated training UI (does not train anything).
    #[command(long_flag = "demo")]
    Demo,
    /// Terminal dashboard that follows a run's training_state.json.
    #[command(long_flag = "monitor")]
    Monitor,
    /// Print how to run the kernel benchmarks.
    #[command(long_flag = "benchmark")]
    Benchmark,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum Task {
    /// Instruction tuning: JSON/JSONL `{prompt, response[, system]}`; loss
    /// on response tokens only.
    Sft,
    /// Next-token prediction on .txt lines, .json text/token arrays or
    /// packed little-endian u32 .bin tokens.
    Pretrain,
}

impl Task {
    fn as_str(self) -> &'static str {
        match self {
            Self::Sft => "sft",
            Self::Pretrain => "pretrain",
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum Device {
    /// High-performance GPU.
    Auto,
    /// Same as auto.
    Gpu,
    /// wgpu's software fallback adapter (slow; for machines without a GPU).
    Cpu,
}

#[derive(Args, Debug, Clone)]
struct ModelArgs {
    /// Model directory (config.json, tokenizer.json, *.safetensors).
    #[arg(long, value_name = "DIR")]
    model: PathBuf,
    /// Compute device.
    #[arg(long, value_enum, default_value_t = Device::Auto)]
    device: Device,
}

#[derive(Args, Debug, Clone)]
struct DataArgs {
    /// Dataset file (see --task for formats).
    #[arg(long, alias = "dataset", value_name = "FILE")]
    data: PathBuf,
    /// Training objective.
    #[arg(long, value_enum, default_value_t = Task::Sft)]
    task: Task,
    /// Maximum tokens per example. SFT examples are cut at the end; examples
    /// whose prompt does not fit are skipped.
    #[arg(long, default_value_t = 256)]
    seq_len: usize,
    /// Use only the first N examples.
    #[arg(long)]
    max_samples: Option<usize>,
    /// Do not wrap SFT prompts in the ChatML template.
    #[arg(long)]
    no_chat_template: bool,
}

#[derive(Args, Debug)]
struct TrainArgs {
    #[command(flatten)]
    model: ModelArgs,
    #[command(flatten)]
    data: DataArgs,
    /// Output directory for the adapter, checkpoints and metrics.jsonl.
    #[arg(
        long,
        alias = "output-dir",
        value_name = "DIR",
        default_value = "outputs/run"
    )]
    out: PathBuf,
    /// Where to write the adapter [default: <out>/lora_adapters.safetensors].
    #[arg(long, value_name = "FILE")]
    save_lora: Option<PathBuf>,
    /// Resume from an adapter written by a previous run (its .state.json
    /// next to it restores optimizer state and the step counter).
    #[arg(long, value_name = "FILE")]
    resume: Option<PathBuf>,
    /// Peak learning rate.
    #[arg(long, default_value_t = 2e-4)]
    lr: f32,
    /// Linear warmup steps before cosine decay to lr/10.
    #[arg(long, default_value_t = 10)]
    warmup_steps: usize,
    /// Passes over the data (ignored when --max-steps is reached first).
    #[arg(long, default_value_t = 1)]
    epochs: usize,
    /// Stop after this many optimizer steps.
    #[arg(long)]
    max_steps: Option<usize>,
    /// Sequences per batch.
    #[arg(long, default_value_t = 1)]
    batch_size: usize,
    /// Batches per optimizer step.
    #[arg(long, default_value_t = 4)]
    grad_accum: usize,
    /// Clip the global gradient norm.
    #[arg(long, default_value_t = 1.0)]
    grad_clip: f32,
    /// AdamW weight decay.
    #[arg(long, default_value_t = 0.0)]
    weight_decay: f32,
    /// LoRA rank.
    #[arg(long, default_value_t = 8)]
    lora_rank: usize,
    /// LoRA alpha (updates are scaled by alpha / rank).
    #[arg(long, default_value_t = 16.0)]
    lora_alpha: f32,
    /// Comma-separated projections that get LoRA adapters.
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "q_proj,k_proj,v_proj,o_proj"
    )]
    lora_targets: Vec<String>,
    /// What to train: lora, or base weights by group (all, attn, mlp, norm, lm_head).
    #[arg(long, default_value = "lora")]
    train_modules: String,
    /// Log every N optimizer steps.
    #[arg(long, default_value_t = 10)]
    log_every: usize,
    /// Save adapter and checkpoint every N optimizer steps (0 = only at the end).
    #[arg(long, default_value_t = 0)]
    save_every: usize,
    /// Held-out dataset evaluated before and after training (and every
    /// --eval-every steps).
    #[arg(long, value_name = "FILE")]
    eval_data: Option<PathBuf>,
    /// Evaluate on --eval-data every N optimizer steps (0 = start and end only).
    #[arg(long, default_value_t = 0)]
    eval_every: usize,
    /// Cap on evaluation examples.
    #[arg(long, default_value_t = 64)]
    eval_samples: usize,
    /// Data-order seed.
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Generate from this prompt after training (a quick sanity check).
    #[arg(long)]
    sample_prompt: Option<String>,
    /// Consumer-hardware preset: seq-len 128, rank 4, grad-accum 8 unless set.
    #[arg(long)]
    lite: bool,
    /// Number of processes in file-based data-parallel training (prototype).
    #[arg(long, default_value_t = 1)]
    world_size: usize,
    /// This process's rank in 0..world-size.
    #[arg(long, default_value_t = 0)]
    rank: usize,
    /// Shared directory for gradient exchange [default: <out>/grad_sync].
    #[arg(long, value_name = "DIR")]
    sync_dir: Option<PathBuf>,
}

#[derive(Args, Debug)]
struct EvalArgs {
    #[command(flatten)]
    model: ModelArgs,
    #[command(flatten)]
    data: DataArgs,
    /// LoRA adapter to apply.
    #[arg(long, value_name = "FILE")]
    lora: Option<PathBuf>,
}

#[derive(Args, Debug)]
struct GenerateArgs {
    #[command(flatten)]
    model: ModelArgs,
    /// LoRA adapter to apply (rank, alpha and targets are read from the file).
    #[arg(long, value_name = "FILE")]
    lora: Option<PathBuf>,
    /// The user message (wrapped in the ChatML template unless --raw).
    #[arg(long, default_value = "Hello!")]
    prompt: String,
    /// Optional system message.
    #[arg(long)]
    system: Option<String>,
    /// Feed the prompt to the model verbatim.
    #[arg(long)]
    raw: bool,
    /// Maximum tokens to generate.
    #[arg(long, default_value_t = 128)]
    max_new_tokens: usize,
    /// Sampling temperature (0 = greedy).
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,
    /// Sample from the K most likely tokens (0 = no limit).
    #[arg(long, default_value_t = 50)]
    top_k: usize,
    /// Nucleus sampling threshold (1 = off).
    #[arg(long, default_value_t = 0.9)]
    top_p: f32,
    /// Seed for reproducible sampling.
    #[arg(long)]
    seed: Option<u64>,
    /// Recompute the whole sequence every token instead of using the KV cache.
    #[arg(long)]
    no_kv_cache: bool,
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Train(args) => pollster::block_on(train(*args)),
        Command::Eval(args) => pollster::block_on(eval(args)),
        Command::Generate(args) => pollster::block_on(generate(args)),
        Command::Demo => {
            ferrotensor_trainer::run_demo(ferrotensor_trainer::DemoConfig::default());
            Ok(())
        }
        Command::Monitor => ferrotensor_monitor::run().map_err(|e| format!("monitor: {e}").into()),
        Command::Benchmark => {
            println!(
                "Run the kernel benchmarks with: cargo run -p ferrotensor-benchmarks --release"
            );
            Ok(())
        }
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

// ---------------------------------------------------------------------------
// Model loading

struct Loaded {
    ctx: Arc<WgpuContext>,
    engine: Arc<ComputeEngine>,
    tokenizer: Tokenizer,
    config: Lfm2Config,
    weights: WeightsContext,
}

async fn load_base(args: &ModelArgs) -> Result<Loaded> {
    let dir = &args.model;
    if !dir.is_dir() {
        return Err(format!("model directory {} does not exist", dir.display()).into());
    }
    let config_path = dir.join("config.json");
    let config = ModelLoader::load_config(&config_path)
        .map_err(|e| format!("cannot read {}: {e}", config_path.display()))?;
    let tok_path = dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tok_path)
        .map_err(|e| format!("cannot load {}: {e}", tok_path.display()))?;
    let mut weights = WeightsContext::new();
    weights
        .load_model_dir(dir)
        .map_err(|e| format!("cannot load weights from {}: {e}", dir.display()))?;

    let (power, fallback) = match args.device {
        Device::Auto | Device::Gpu => (
            ferrotensor_backend::wgpu::PowerPreference::HighPerformance,
            false,
        ),
        Device::Cpu => (ferrotensor_backend::wgpu::PowerPreference::LowPower, true),
    };
    let ctx = Arc::new(WgpuContext::new_with_options(power, fallback).await);
    let engine = Arc::new(ComputeEngine::new(&ctx));
    eprintln!("device: {}", ctx.adapter_info());
    eprintln!(
        "model:  {} ({} layers, hidden {}, vocab {})",
        dir.display(),
        config.num_hidden_layers,
        config.hidden_size,
        config.vocab_size
    );
    Ok(Loaded {
        ctx,
        engine,
        tokenizer,
        config,
        weights,
    })
}

fn build_model(
    base: &Loaded,
    lora: Option<&LoraMetadata>,
    trainable_base: bool,
) -> Result<Lfm2Model> {
    let mut builder = Lfm2ModelBuilder::new(
        base.ctx.clone(),
        base.engine.clone(),
        &base.weights,
        base.config.clone(),
    )
    .with_trainable_base(trainable_base);
    if let Some(meta) = lora {
        builder = builder.with_lora(LoRaConfig {
            rank: meta.rank,
            alpha: meta.alpha,
            target_modules: meta.target_modules.clone(),
        });
    }
    Ok(builder.build()?)
}

/// Load `path` and copy it into the model's LoRA tensors.
fn restore_adapter(base: &Loaded, model: &Lfm2Model, path: &Path) -> Result<usize> {
    let ckpt = load_lora_checkpoint(path)?;
    let n = apply_checkpoint(&base.ctx, &ckpt, &model.named_lora_params())
        .map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(n)
}

/// The LoRA settings stored in an adapter file.
fn adapter_metadata(path: &Path) -> Result<LoraMetadata> {
    load_lora_checkpoint(path)?.metadata.ok_or_else(|| {
        format!(
            "{} has no LoRA metadata (rank/alpha/targets); it was not written by ferrotensor-cli",
            path.display()
        )
        .into()
    })
}

// ---------------------------------------------------------------------------
// Data

fn load_dataset(args: &DataArgs, tokenizer: &Tokenizer) -> Result<TextDataset> {
    let path = &args.data;
    if !path.is_file() {
        return Err(format!("dataset file {} does not exist", path.display()).into());
    }
    if args.seq_len < 2 {
        return Err("--seq-len must be at least 2".into());
    }
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let chat = !args.no_chat_template;
    let seq = args.seq_len;
    let ds = match (args.task, ext.as_str()) {
        (Task::Sft, "json") => load_json_sft(path, tokenizer, chat, seq)?,
        (Task::Sft, "txt" | "bin") => {
            return Err(
                "SFT expects .json/.jsonl with {\"prompt\", \"response\"}; use --task pretrain for .txt/.bin"
                    .into(),
            )
        }
        (Task::Sft, _) => TextDataset::from_jsonl_sft(path, tokenizer, chat, seq)
            .map_err(|e| format!("{}: {e}", path.display()))?,
        (Task::Pretrain, "bin") => TextDataset::from_token_bin(path, seq)?,
        (Task::Pretrain, "json") => {
            let content = fs::read_to_string(path)?;
            if let Ok(tokens) = serde_json::from_str::<Vec<u32>>(&content) {
                TextDataset::from_packed_tokens(tokens, seq)
            } else if let Ok(lines) = serde_json::from_str::<Vec<String>>(&content) {
                let seqs = lines
                    .iter()
                    .filter(|l| !l.trim().is_empty())
                    .map(|l| tokenizer.encode(l))
                    .collect();
                TextDataset::pack_sequences(seqs, seq)
            } else {
                return Err("pretrain .json must be [token ids] or [\"text\", ...]".into());
            }
        }
        (Task::Pretrain, _) => {
            let text = fs::read_to_string(path)?;
            let seqs = text
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(|l| tokenizer.encode(l))
                .collect();
            TextDataset::pack_sequences(seqs, seq)
        }
    };
    let ds = match args.max_samples {
        Some(n) => take_first(ds, n),
        None => ds,
    };
    if ds.is_empty() {
        return Err(format!(
            "{} produced no usable examples (empty, or every prompt is longer than --seq-len {seq})",
            path.display()
        )
        .into());
    }
    Ok(ds)
}

fn take_first(ds: TextDataset, n: usize) -> TextDataset {
    let mut out = TextDataset::new();
    for ex in ds.examples().iter().take(n) {
        out.add_example(ex.clone());
    }
    out
}

fn load_json_sft(
    path: &Path,
    tokenizer: &Tokenizer,
    chat: bool,
    seq: usize,
) -> Result<TextDataset> {
    #[derive(serde::Deserialize)]
    struct Sample {
        prompt: String,
        response: String,
        #[serde(default)]
        system: Option<String>,
    }
    let content = fs::read_to_string(path)?;
    let samples: Vec<Sample> = serde_json::from_str(&content).map_err(|e| {
        format!(
            "{}: expected a JSON array of {{\"prompt\", \"response\"}} objects: {e}",
            path.display()
        )
    })?;
    let mut ds = TextDataset::new();
    for s in samples {
        if let Some(ex) = build_sft_example(
            tokenizer,
            &s.prompt,
            &s.response,
            s.system.as_deref(),
            chat,
            seq,
        ) {
            ds.add_example(ex);
        }
    }
    Ok(ds)
}

fn supervised_tokens(ds: &TextDataset) -> usize {
    ds.examples()
        .iter()
        .map(|e| e.loss_mask.iter().filter(|&&m| m != 0).count())
        .sum()
}

// ---------------------------------------------------------------------------
// eval

/// Token-weighted mean loss over (at most `limit`) examples of `ds`.
async fn evaluate(
    base: &Loaded,
    model: &Lfm2Model,
    ds: &TextDataset,
    limit: usize,
) -> (f32, usize) {
    let mut loss_sum = 0.0f64;
    let mut tokens = 0usize;
    for ex in ds.examples().iter().take(limit) {
        let logits = model.forward(&base.engine, &ex.input_ids, 0, None, None);
        let host = logits.data.to_vec(&base.ctx).await;
        if let Some((loss, n)) = masked_cross_entropy_from_logits(
            &host,
            base.config.vocab_size,
            &ex.target_ids,
            &ex.loss_mask,
        ) {
            loss_sum += loss as f64 * n as f64;
            tokens += n;
        }
    }
    if tokens == 0 {
        (f32::NAN, 0)
    } else {
        ((loss_sum / tokens as f64) as f32, tokens)
    }
}

async fn eval(args: EvalArgs) -> Result<()> {
    let base = load_base(&args.model).await?;
    let ds = load_dataset(&args.data, &base.tokenizer)?;
    let meta = args.lora.as_deref().map(adapter_metadata).transpose()?;
    let model = build_model(&base, meta.as_ref(), false)?;
    if let Some(path) = &args.lora {
        let n = restore_adapter(&base, &model, path)?;
        eprintln!("lora:   {} ({n} tensors)", path.display());
    }
    let t = Instant::now();
    let (loss, tokens) = evaluate(&base, &model, &ds, usize::MAX).await;
    println!(
        "eval: {} examples, {tokens} tokens, loss {loss:.4}, perplexity {:.3} ({:.1}s)",
        ds.len(),
        loss.exp(),
        t.elapsed().as_secs_f32()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// generate

async fn generate(args: GenerateArgs) -> Result<()> {
    let base = load_base(&args.model).await?;
    let meta = args.lora.as_deref().map(adapter_metadata).transpose()?;
    let model = build_model(&base, meta.as_ref(), false)?;
    if let Some(path) = &args.lora {
        let n = restore_adapter(&base, &model, path)?;
        eprintln!("lora:   {} ({n} tensors)", path.display());
    }
    if !(0.0..=1.0).contains(&args.top_p) || args.temperature < 0.0 {
        return Err("--top-p must be in [0, 1] and --temperature >= 0".into());
    }
    let mut sampler = Sampler::new(args.temperature, args.top_k, args.top_p, args.seed);
    let prompt_ids = if args.raw {
        base.tokenizer.encode(&args.prompt)
    } else {
        base.tokenizer
            .encode_chat(args.system.as_deref(), &args.prompt)
    };
    let use_cache = !args.no_kv_cache;
    let stats = generate_tokens(
        &base,
        &model,
        &prompt_ids,
        args.max_new_tokens,
        &mut sampler,
        use_cache,
        |text| {
            print!("{text}");
            let _ = std::io::stdout().flush();
        },
    )
    .await;
    println!();
    eprintln!(
        "[{} prompt + {} new tokens, {:.1} tok/s, kv cache {}]",
        prompt_ids.len(),
        stats.0,
        stats.0 as f32 / stats.1.as_secs_f32().max(1e-6),
        if stats.2 { "on" } else { "off" }
    );
    Ok(())
}

/// Sample up to `max_new` tokens after `prompt`, streaming decoded text to
/// `emit`. Returns (tokens generated, decode time, whether the KV cache was
/// used).
async fn generate_tokens(
    base: &Loaded,
    model: &Lfm2Model,
    prompt: &[u32],
    max_new: usize,
    sampler: &mut Sampler,
    use_cache: bool,
    mut emit: impl FnMut(&str),
) -> (usize, Duration, bool) {
    let vocab = base.config.vocab_size;
    let stop: Vec<u32> = [
        Some(base.tokenizer.eos_token_id),
        base.config.eos_token_id,
        base.tokenizer
            .hf_tokenizer
            .as_ref()
            .and_then(|t| t.token_to_id("<|endoftext|>")),
    ]
    .into_iter()
    .flatten()
    .collect();
    let mut cache = if use_cache {
        model.new_kv_cache(prompt.len() + max_new)
    } else {
        None
    };
    let cached = cache.is_some();

    let mut tokens = prompt.to_vec();
    let mut generated: Vec<u32> = Vec::new();
    let mut printed = String::new();
    let start = Instant::now();
    let mut logits = model.forward(&base.engine, &tokens, 0, cache.as_mut(), None);
    for _ in 0..max_new {
        let host = logits.data.to_vec(&base.ctx).await;
        let rows = host.len() / vocab;
        let last = &host[(rows - 1) * vocab..rows * vocab];
        let next = sampler.sample(last);
        if stop.contains(&next) {
            break;
        }
        generated.push(next);
        tokens.push(next);
        let text = base.tokenizer.decode(&generated);
        if let Some(delta) = text.strip_prefix(printed.as_str()) {
            emit(delta);
            printed = text;
        }
        logits = if cached {
            model.forward(
                &base.engine,
                &[next],
                tokens.len() - 1,
                cache.as_mut(),
                None,
            )
        } else {
            model.forward(&base.engine, &tokens, 0, None, None)
        };
    }
    (generated.len(), start.elapsed(), cached)
}

// ---------------------------------------------------------------------------
// train

fn select_train_params(
    model: &Lfm2Model,
    modules: &str,
) -> Result<HashMap<String, ferrotensor_autograd::Tensor>> {
    let mode = modules.to_ascii_lowercase();
    let params: HashMap<_, _> = match mode.as_str() {
        "lora" => model.named_lora_params(),
        "all" => model
            .named_params()
            .into_iter()
            .filter(|(n, _)| !n.contains("embed_tokens"))
            .collect(),
        "embed" | "embedding" => {
            return Err(
                "training the token embedding is not supported (the lookup runs on the CPU)".into(),
            )
        }
        m => {
            let key = match m {
                "attn" | "attention" => "self_attn",
                "mlp" => "feed_forward",
                "head" => "lm_head",
                other => other,
            };
            model
                .named_params()
                .into_iter()
                .filter(|(n, _)| n.contains(key))
                .collect()
        }
    };
    if params.is_empty() {
        return Err(format!("--train-modules {modules} selects no parameters").into());
    }
    Ok(params)
}

fn param_count(params: &HashMap<String, ferrotensor_autograd::Tensor>) -> usize {
    params
        .values()
        .map(|t| t.data.shape.iter().product::<usize>())
        .sum()
}

fn state_path_for(adapter: &Path) -> PathBuf {
    let name = adapter
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("lora_adapters.safetensors");
    adapter.with_file_name(format!("{name}.state.json"))
}

async fn train(mut args: TrainArgs) -> Result<()> {
    if args.lite {
        // Explicit flags win; only defaults are replaced.
        if args.data.seq_len == 256 {
            args.data.seq_len = 128;
        }
        if args.lora_rank == 8 {
            args.lora_rank = 4;
            args.lora_alpha = 8.0;
        }
        if args.grad_accum == 4 {
            args.grad_accum = 8;
        }
    }
    if args.batch_size == 0 || args.grad_accum == 0 || args.lora_rank == 0 {
        return Err("--batch-size, --grad-accum and --lora-rank must be positive".into());
    }
    if args.world_size == 0 || args.rank >= args.world_size {
        return Err(format!(
            "--rank {} must be below --world-size {}",
            args.rank, args.world_size
        )
        .into());
    }
    let distributed = args.world_size > 1;
    if distributed && args.resume.is_some() {
        return Err("--resume is not supported with --world-size > 1 yet".into());
    }
    let is_root = args.rank == 0;

    let base = load_base(&args.model).await?;
    let mut dataset = load_dataset(&args.data, &base.tokenizer)?;
    let eval_ds = match &args.eval_data {
        Some(path) => {
            let mut eval_args = args.data.clone();
            eval_args.data = path.clone();
            eval_args.max_samples = Some(args.eval_samples);
            Some(load_dataset(&eval_args, &base.tokenizer)?)
        }
        None => None,
    };

    let lora_on = args.train_modules.eq_ignore_ascii_case("lora");
    let meta = LoraMetadata {
        rank: args.lora_rank,
        alpha: args.lora_alpha,
        target_modules: args.lora_targets.clone(),
    };
    let model = build_model(&base, lora_on.then_some(&meta), !lora_on)?;
    let named_params = select_train_params(&model, &args.train_modules)?;
    let train_params: Vec<_> = {
        let mut names: Vec<&String> = named_params.keys().collect();
        names.sort();
        names.into_iter().map(|n| named_params[n].clone()).collect()
    };

    fs::create_dir_all(&args.out)
        .map_err(|e| format!("cannot create {}: {e}", args.out.display()))?;
    let adapter_path = args
        .save_lora
        .clone()
        .unwrap_or_else(|| args.out.join("lora_adapters.safetensors"));
    let state_path = state_path_for(&adapter_path);

    // Data-parallel sharding: each rank trains on a disjoint slice.
    if distributed {
        let idx = ferrotensor_trainer::distributed::shard_indices(
            dataset.len(),
            args.rank,
            args.world_size,
        );
        let mut shard = TextDataset::new();
        for i in idx {
            shard.add_example(dataset.examples()[i].clone());
        }
        dataset = shard;
    }

    let micro_per_step = args.batch_size * args.grad_accum;
    let steps_per_epoch = dataset.len().div_ceil(micro_per_step).max(1);
    let total_steps = args
        .max_steps
        .unwrap_or(steps_per_epoch * args.epochs.max(1))
        .max(1);
    let warmup = args.warmup_steps.min(total_steps.saturating_sub(1));

    let mut optimizer = AdamW::new(base.engine.clone(), args.lr);
    optimizer.weight_decay = args.weight_decay;
    let scheduler = WarmupCosineScheduler::new(args.lr, args.lr * 0.1, warmup, total_steps);
    let mut trainer = Trainer::new(
        model,
        Box::new(optimizer),
        base.ctx.clone(),
        base.engine.clone(),
    )
    .with_grad_accum_steps(micro_per_step)
    .with_clip_grad_norm(args.grad_clip)
    .with_lr_scheduler(Box::new(scheduler));
    if distributed {
        let sync_dir = args
            .sync_dir
            .clone()
            .unwrap_or_else(|| args.out.join("grad_sync"));
        let cfg = FileGradSyncConfig::new(args.world_size, args.rank, sync_dir)?
            .with_max_wait(Duration::from_secs(600));
        trainer = trainer.with_file_grad_sync(cfg);
    }

    // Resume: adapter weights, optimizer moments, step counter.
    let mut start_step = 0usize;
    if let Some(resume) = &args.resume {
        let n = restore_adapter(&base, &trainer.model, resume)?;
        let resume_state = state_path_for(resume);
        if resume_state.is_file() {
            let ckpt = TrainingCheckpoint::load(&resume_state)?;
            ckpt.training_state.validate_resume(
                &args.train_modules,
                args.data.task.as_str(),
                args.data.seq_len,
                &args.model.model.display().to_string(),
            )?;
            if let Some(opt) = &ckpt.optimizer_state {
                if !trainer.optimizer.load_state(&base.ctx, &named_params, opt) {
                    log::warn!(
                        "optimizer state in {} did not apply",
                        resume_state.display()
                    );
                }
            }
            start_step = ckpt.training_state.step;
            args.seed = ckpt.training_state.seed;
            trainer.set_optimizer_steps(start_step);
        }
        eprintln!(
            "resume: {} ({n} tensors), continuing at step {start_step}",
            resume.display()
        );
    }

    let sup = supervised_tokens(&dataset);
    eprintln!(
        "data:   {} examples ({} task), {} supervised tokens, seq-len {}",
        dataset.len(),
        args.data.task.as_str(),
        sup,
        args.data.seq_len
    );
    eprintln!(
        "train:  {} ({} trainable params), lr {} warmup {} cosine, {} steps x {} sequences",
        args.train_modules,
        param_count(&named_params),
        args.lr,
        warmup,
        total_steps,
        micro_per_step
    );

    let metrics_path = args.out.join("metrics.jsonl");
    let mut metrics = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&metrics_path)?;

    if let Some(ds) = &eval_ds {
        let (loss, tokens) = evaluate(&base, &trainer.model, ds, args.eval_samples).await;
        eprintln!(
            "eval before: loss {loss:.4} ppl {:.3} ({tokens} tokens)",
            loss.exp()
        );
        writeln!(
            metrics,
            "{}",
            serde_json::json!({"step": start_step, "eval_loss": loss})
        )?;
    }

    let save = |trainer: &Trainer, step: usize, best: Option<f32>| -> Result<()> {
        if !is_root {
            return Ok(());
        }
        pollster::block_on(save_lora_checkpoint(
            &base.ctx,
            &adapter_path,
            &trainer.model.named_lora_params(),
            Some(&meta),
        ))?;
        let state = TrainingState {
            step,
            total_steps,
            best_loss: best,
            lr: args.lr,
            epoch: step / steps_per_epoch,
            batch_size: args.batch_size,
            grad_accum_steps: args.grad_accum,
            sequence_length: args.data.seq_len,
            fp16: false,
            train_modules: args.train_modules.clone(),
            task: args.data.task.as_str().to_string(),
            model_path: args.model.model.display().to_string(),
            dataset_path: args.data.data.display().to_string(),
            seed: args.seed,
        };
        state.save(args.out.join("training_state.json"))?;
        TrainingCheckpoint {
            training_state: state,
            optimizer_state: trainer.optimizer.save_state(&base.ctx, &named_params),
            grad_scaler_state: None,
        }
        .save(&state_path)?;
        Ok(())
    };

    eprintln!("  step |   loss | grad norm |       lr | tok/s | s/step");
    let mut loader = DataLoader::new_with_pad_token(
        dataset.clone(),
        args.batch_size,
        !distributed,
        base.tokenizer.pad_token_id,
    )
    .with_seed(args.seed);
    let mut step = start_step;
    let skip_micro = start_step * micro_per_step;
    let mut micro_seen = 0usize;
    let (mut win_loss, mut win_tok, mut win_seq) = (0.0f64, 0usize, 0usize);
    let mut win_start = Instant::now();
    let run_start = Instant::now();
    let mut best: Option<f32> = None;
    let mut first_loss: Option<f32> = None;
    let mut last_loss = f32::NAN;
    let mut last_norm = 0.0f32;
    let mut last_lr = 0.0f32;

    'outer: for epoch in 0.. {
        loader.set_epoch(epoch as u64);
        loader.reset();
        let mut any = false;
        for batch in &mut loader {
            any = true;
            if micro_seen < skip_micro {
                micro_seen += batch.batch_size;
                continue;
            }
            micro_seen += batch.batch_size;
            for s in trainer.train_batch(&batch, &train_params)? {
                win_loss += s.loss as f64 * s.tokens as f64;
                win_tok += s.tokens;
                win_seq += 1;
                if let Some(n) = s.grad_norm {
                    last_norm = n;
                }
                if let Some(lr) = s.lr {
                    last_lr = lr;
                }
                if !s.optimizer_step {
                    continue;
                }
                step += 1;
                if step % args.log_every.max(1) == 0 || step == total_steps {
                    let loss = (win_loss / win_tok.max(1) as f64) as f32;
                    let secs = win_start.elapsed().as_secs_f32();
                    let steps_in_win = win_seq as f32 / micro_per_step as f32;
                    eprintln!(
                        "{step:6} | {loss:6.4} | {last_norm:9.3} | {last_lr:8.2e} | {:5.0} | {:6.2}",
                        win_tok as f32 / secs.max(1e-6),
                        secs / steps_in_win.max(1e-6)
                    );
                    writeln!(
                        metrics,
                        "{}",
                        serde_json::json!({
                            "step": step, "loss": loss, "grad_norm": last_norm, "lr": last_lr,
                            "tokens": win_tok, "seconds": secs,
                            "elapsed": run_start.elapsed().as_secs_f32(),
                        })
                    )?;
                    first_loss.get_or_insert(loss);
                    last_loss = loss;
                    best = Some(best.map_or(loss, |b: f32| b.min(loss)));
                    (win_loss, win_tok, win_seq) = (0.0, 0, 0);
                    win_start = Instant::now();
                }
                if args.save_every > 0 && step % args.save_every == 0 && step < total_steps {
                    save(&trainer, step, best)?;
                }
                if let Some(ds) = &eval_ds {
                    if args.eval_every > 0 && step % args.eval_every == 0 && step < total_steps {
                        let (loss, _) =
                            evaluate(&base, &trainer.model, ds, args.eval_samples).await;
                        eprintln!("eval @ {step}: loss {loss:.4} ppl {:.3}", loss.exp());
                        writeln!(
                            metrics,
                            "{}",
                            serde_json::json!({"step": step, "eval_loss": loss})
                        )?;
                    }
                }
                if step >= total_steps {
                    break 'outer;
                }
            }
        }
        if !any || epoch + 1 >= args.epochs.max(1) && args.max_steps.is_none() {
            break;
        }
    }

    let elapsed = run_start.elapsed();
    save(&trainer, step, best)?;
    eprintln!(
        "done:   {step} steps in {:.1}s; train loss {} -> {last_loss:.4}",
        elapsed.as_secs_f32(),
        first_loss.map_or("n/a".to_string(), |l| format!("{l:.4}")),
    );
    if is_root {
        eprintln!(
            "saved:  {} (+ {})",
            adapter_path.display(),
            state_path.display()
        );
        eprintln!("metrics: {}", metrics_path.display());
    }

    if let Some(ds) = &eval_ds {
        let (loss, tokens) = evaluate(&base, &trainer.model, ds, args.eval_samples).await;
        eprintln!(
            "eval after: loss {loss:.4} ppl {:.3} ({tokens} tokens)",
            loss.exp()
        );
        writeln!(
            metrics,
            "{}",
            serde_json::json!({"step": step, "eval_loss": loss})
        )?;
    }

    if let Some(prompt) = &args.sample_prompt {
        let ids = base.tokenizer.encode_chat(None, prompt);
        let mut sampler = Sampler::new(0.0, 0, 1.0, None);
        eprintln!("sample ({prompt:?}):");
        generate_tokens(&base, &trainer.model, &ids, 96, &mut sampler, true, |t| {
            print!("{t}");
            let _ = std::io::stdout().flush();
        })
        .await;
        println!();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> std::result::Result<Cli, clap::Error> {
        Cli::try_parse_from(std::iter::once("ferrotensor-cli").chain(args.iter().copied()))
    }

    #[test]
    fn cli_definition_is_consistent() {
        use clap::CommandFactory;
        Cli::command().debug_assert();
    }

    #[test]
    fn train_parses_flags() {
        let cli = parse(&[
            "train",
            "--model",
            "m",
            "--data",
            "d.jsonl",
            "--task",
            "pretrain",
            "--lr",
            "1e-3",
            "--lora-targets",
            "q_proj,v_proj",
            "--max-steps",
            "5",
        ])
        .expect("parse");
        let Command::Train(t) = cli.command else {
            panic!("expected train");
        };
        assert_eq!(t.data.task, Task::Pretrain);
        assert_eq!(t.lora_targets, vec!["q_proj", "v_proj"]);
        assert_eq!(t.max_steps, Some(5));
        assert!((t.lr - 1e-3).abs() < 1e-9);
    }

    #[test]
    fn train_requires_model_and_data() {
        assert!(parse(&["train", "--data", "d.jsonl"]).is_err());
        assert!(parse(&["train", "--model", "m"]).is_err());
    }

    #[test]
    fn bad_numbers_are_errors_not_panics() {
        let err = parse(&["train", "--model", "m", "--data", "d", "--seq-len", "abc"])
            .expect_err("should fail");
        assert!(err.to_string().contains("seq-len"));
    }

    #[test]
    fn infer_is_an_alias_of_generate() {
        let cli =
            parse(&["infer", "--model", "m", "--prompt", "hi", "--top-k", "5"]).expect("parse");
        let Command::Generate(g) = cli.command else {
            panic!("expected generate");
        };
        assert_eq!(g.top_k, 5);
        assert_eq!(g.prompt, "hi");
    }

    #[test]
    fn legacy_demo_flag_still_works() {
        assert!(matches!(
            parse(&["--demo"]).expect("parse").command,
            Command::Demo
        ));
    }

    #[test]
    fn state_path_sits_next_to_adapter() {
        assert_eq!(
            state_path_for(Path::new("out/a.safetensors")),
            PathBuf::from("out/a.safetensors.state.json")
        );
    }
}
