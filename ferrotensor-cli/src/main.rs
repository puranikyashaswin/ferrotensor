//! `ferrotensor-cli`: LoRA fine-tuning, evaluation and text generation for small
//! Llama-style (and LFM2) models on consumer GPUs.
//!
//! ```text
//! ferrotensor-cli train    --model DIR --data train.jsonl --out runs/sft
//! ferrotensor-cli eval     --model DIR --data val.jsonl --lora runs/sft/lora_adapters.safetensors
//! ferrotensor-cli generate --model DIR --lora runs/sft/lora_adapters.safetensors --prompt "Hi"
//! ferrotensor-cli classify-train   --model DIR --data train.jsonl --output runs/cls
//! ferrotensor-cli classify-predict --model DIR --classifier runs/cls --data test.jsonl --output probs.jsonl
//! ferrotensor-cli info --check
//! ferrotensor-cli export-adapter --lora runs/sft/lora_adapters.safetensors --model DIR --output runs/sft/peft
//! ```

mod classify;
mod info;
mod memory_report;

use clap::{Args, Parser, Subcommand, ValueEnum};
use ferrotensor_autograd::{AdamW, OptimizerLoad, WarmupCosineScheduler};
use ferrotensor_backend::{ComputeEngine, MemoryCategory, WgpuContext};
use ferrotensor_graph::lfm::{legacy_llama_name, Lfm2Model};
use ferrotensor_graph::model_builder::{Lfm2ModelBuilder, LoRaConfig};
use ferrotensor_loader::{Lfm2Config, ModelLoader, Sampler, Tokenizer, WeightsContext};
use ferrotensor_trainer::checkpoint::{
    apply_checkpoint, load_lora_checkpoint, save_lora_checkpoint, upgrade_optimizer_names,
    LoraCheckpoint, LoraMetadata, TrainingCheckpoint, TrainingState,
};
use ferrotensor_trainer::peft::{export_peft_adapter, is_peft_adapter_dir, load_peft_adapter};
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
    /// LoRA fine-tuning of a decoder as a text classifier, from JSONL
    /// {"text", "label"} lines.
    ClassifyTrain(Box<classify::ClassifyTrainArgs>),
    /// Class probabilities for every JSONL line, from a classify-train output.
    ClassifyPredict(classify::ClassifyPredictArgs),
    /// Simulated training UI (does not train anything).
    #[command(long_flag = "demo")]
    Demo,
    /// Terminal dashboard that follows a run's training_state.json.
    #[command(long_flag = "monitor")]
    Monitor,
    /// Show the GPUs wgpu can use and, with --check, test the kernels on one.
    Info(info::InfoArgs),
    /// Convert a LoRA adapter to another format (PEFT, for PyTorch).
    ExportAdapter(ExportAdapterArgs),
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
    /// Store frozen base weights as f32 instead of the default f16. f16 halves
    /// their memory and computation stays f32; on SmolLM2-1.7B the held-out
    /// loss after LoRA differs by under 0.002. Weights that train are always f32.
    #[arg(long)]
    f32_weights: bool,
    /// Frozen base weights are already f16 by default; kept for compatibility.
    #[arg(long, hide = true, conflicts_with = "f32_weights")]
    f16_weights: bool,
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
    /// Where to write the adapter, or the trained weights when --train-modules
    /// is not lora [default: OUT/lora_adapters.safetensors, or
    /// OUT/trained_weights.safetensors].
    #[arg(long, value_name = "FILE")]
    save_lora: Option<PathBuf>,
    /// Resume from the adapter (or trained weights) written by a previous run
    /// with the same --train-modules (its .state.json and .state.safetensors
    /// next to it restore the step counter and optimizer state).
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
    /// Sequences per batch. Llama-style models run a batch through the GPU
    /// together (padded to the longest); the result matches training them
    /// one at a time.
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
    /// What to train: lora, or base weights by group (all, attn, mlp, norm,
    /// lm_head). Trained base weights are saved in f32, only the tensors that
    /// trained, and load with --lora in eval and generate or with --resume.
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
    /// Shared directory for gradient exchange [default: OUT/grad_sync].
    #[arg(long, value_name = "DIR")]
    sync_dir: Option<PathBuf>,
    /// Print GPU memory by category after the model loads and after each of
    /// the first STEPS optimizer steps, and write OUT/memory_report.json.
    #[arg(
        long,
        value_name = "STEPS",
        num_args = 0..=1,
        require_equals = true,
        default_missing_value = "3"
    )]
    memory_report: Option<usize>,
}

#[derive(Args, Debug)]
struct EvalArgs {
    #[command(flatten)]
    model: ModelArgs,
    #[command(flatten)]
    data: DataArgs,
    /// LoRA adapter to apply: a Ferrotensor file or a PEFT adapter directory.
    /// A file of trained base weights (from --train-modules other than lora)
    /// also works; the model then loads in f32.
    #[arg(long, value_name = "FILE")]
    lora: Option<PathBuf>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum AdapterFormat {
    /// Hugging Face PEFT: adapter_model.safetensors + adapter_config.json,
    /// loadable with `peft.PeftModel.from_pretrained`.
    Peft,
}

#[derive(Args, Debug)]
struct ExportAdapterArgs {
    /// Adapter written by `train`.
    #[arg(long, alias = "adapter", value_name = "FILE")]
    lora: PathBuf,
    /// Base model directory the adapter was trained on (only config.json is read).
    #[arg(long, value_name = "DIR")]
    model: PathBuf,
    /// Output directory.
    #[arg(long, alias = "out", value_name = "DIR")]
    output: PathBuf,
    /// Output format.
    #[arg(long, value_enum, default_value_t = AdapterFormat::Peft)]
    format: AdapterFormat,
    /// Value of base_model_name_or_path in adapter_config.json (a Hub id
    /// such as HuggingFaceTB/SmolLM2-135M) [default: the --model path].
    #[arg(long, value_name = "NAME")]
    base_model: Option<String>,
}

#[derive(Args, Debug)]
struct GenerateArgs {
    #[command(flatten)]
    model: ModelArgs,
    /// LoRA adapter to apply: a Ferrotensor file (rank, alpha and targets are
    /// read from it) or a PEFT adapter directory. A file of trained base
    /// weights (from --train-modules other than lora) also works; the model
    /// then loads in f32.
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

/// Hint printed after a panic that looks like the GPU running out of memory.
/// wgpu reports an out-of-memory GPU as a lost device on the next submission,
/// which panics inside wgpu with no mention of memory.
const GPU_MEMORY_HINT: &str =
    "\nhint: the GPU device was lost, most often because it ran out of memory. \
Try a smaller --batch-size (with a larger --grad-accum for the same sequences per step), \
a shorter --seq-len, or --lite; keep the default f16 base weights (no --f32-weights).";

/// Whether a panic message means the GPU device was lost or ran out of memory.
fn looks_like_gpu_memory_loss(message: &str) -> bool {
    let m = message.to_ascii_lowercase();
    m.contains("device is lost") || m.contains("out of memory") || m.contains("outofmemory")
}

fn install_gpu_memory_hint() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        default_hook(info);
        let payload = info.payload();
        let message = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied())
            .unwrap_or("");
        if looks_like_gpu_memory_loss(message) {
            eprintln!("{GPU_MEMORY_HINT}");
        }
    }));
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    install_gpu_memory_hint();
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Train(args) => pollster::block_on(train(*args)),
        Command::Eval(args) => pollster::block_on(eval(args)),
        Command::Generate(args) => pollster::block_on(generate(args)),
        Command::ClassifyTrain(args) => pollster::block_on(classify::classify_train(*args)),
        Command::ClassifyPredict(args) => pollster::block_on(classify::classify_predict(args)),
        Command::Demo => {
            ferrotensor_trainer::run_demo(ferrotensor_trainer::DemoConfig::default());
            Ok(())
        }
        Command::Info(args) => info::run(args),
        Command::ExportAdapter(args) => export_adapter(args),
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
    f16_weights: bool,
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
        f16_weights: !args.f32_weights,
    })
}

/// Build the model. `saved` is the metadata of the file the model will load,
/// if any: LoRA settings give the model adapters of that shape; trained base
/// weights give it none and store every weight as f32 (what the file holds,
/// and what training used), whatever `--f32-weights` says.
fn build_model(
    base: &Loaded,
    saved: Option<&LoraMetadata>,
    trainable_base: bool,
) -> Result<Lfm2Model> {
    let weights_file = saved.is_some_and(|m| m.trained_modules.is_some());
    let mut builder = Lfm2ModelBuilder::new(
        base.ctx.clone(),
        base.engine.clone(),
        &base.weights,
        base.config.clone(),
    )
    .with_trainable_base(trainable_base)
    .with_f16_weights(base.f16_weights && !weights_file);
    if let Some(meta) = saved.filter(|m| m.trained_modules.is_none()) {
        builder = builder.with_lora(LoRaConfig {
            rank: meta.rank,
            alpha: meta.alpha,
            target_modules: meta.target_modules.clone(),
        });
    }
    Ok(builder.build()?)
}

/// Read an adapter: a Ferrotensor safetensors file or a PEFT directory.
fn read_adapter(path: &Path) -> Result<LoraCheckpoint> {
    if is_peft_adapter_dir(path) {
        load_peft_adapter(path)
    } else {
        load_lora_checkpoint(path)
    }
}

/// The tensors of `model` that a file with this metadata holds: the base
/// weights selected by `trained_modules` for a file of trained weights, the
/// LoRA adapters otherwise.
fn restorable_params(
    model: &Lfm2Model,
    meta: Option<&LoraMetadata>,
) -> Result<HashMap<String, ferrotensor_autograd::Tensor>> {
    match meta.and_then(|m| m.trained_modules.as_deref()) {
        Some(modules) => select_train_params(model, modules),
        None => Ok(model.named_lora_params()),
    }
}

/// Load `path` and copy it into the model: its LoRA tensors, or the trained
/// base weights the file holds.
fn restore_adapter(base: &Loaded, model: &Lfm2Model, path: &Path) -> Result<usize> {
    restore_checkpoint(base, model, &read_adapter(path)?, path)
}

/// Copy the tensors of `ckpt` (read from `path`) into the model.
fn restore_checkpoint(
    base: &Loaded,
    model: &Lfm2Model,
    ckpt: &LoraCheckpoint,
    path: &Path,
) -> Result<usize> {
    ensure_not_empty(ckpt, path)?;
    let params = restorable_params(model, ckpt.metadata.as_ref())?;
    apply_checkpoint(&base.ctx, ckpt, &params)
        .map_err(|e| format!("{}: {e}", path.display()).into())
}

/// Reject a file without tensors. Versions before trained base weights were
/// saved wrote such a file when training with `--train-modules` other than
/// `lora`; nothing in it can be loaded.
fn ensure_not_empty(ckpt: &LoraCheckpoint, path: &Path) -> Result<()> {
    if ckpt.is_empty() {
        return Err(format!(
            "{} holds no tensors (older versions wrote an empty file when training base weights with --train-modules; that run's weights were not saved)",
            path.display()
        )
        .into());
    }
    Ok(())
}

/// The line `eval` and `generate` print after loading `path`.
fn loaded_line(path: &Path, tensors: usize, meta: Option<&LoraMetadata>) -> String {
    match meta.and_then(|m| m.trained_modules.as_deref()) {
        Some(modules) => format!(
            "weights: {} ({tensors} tensors trained with --train-modules {modules})",
            path.display()
        ),
        None => format!("lora:   {} ({tensors} tensors)", path.display()),
    }
}

/// Whether `--resume` can continue a run of `train_modules` from a file with
/// this metadata: a LoRA run needs an adapter file, a run that trains base
/// weights needs the weights of the same modules.
fn check_resume_file(meta: Option<&LoraMetadata>, train_modules: &str) -> Result<()> {
    let lora_run = train_modules.eq_ignore_ascii_case("lora");
    match (meta.and_then(|m| m.trained_modules.as_deref()), lora_run) {
        (None, true) => Ok(()),
        (Some(saved), false) if saved.eq_ignore_ascii_case(train_modules) => Ok(()),
        (Some(saved), true) => Err(format!(
            "it holds base weights trained with --train-modules {saved}, not a LoRA adapter; resume with --train-modules {saved}"
        )
        .into()),
        (Some(saved), false) => Err(format!(
            "it holds base weights trained with --train-modules {saved}, but this run trains {train_modules}"
        )
        .into()),
        (None, false) => Err(format!(
            "it is a LoRA adapter, but --train-modules {train_modules} trains base weights; resume with --train-modules lora"
        )
        .into()),
    }
}

/// What `--resume` makes of the optimizer state it loaded at `step`: an error
/// when no saved moment fits a parameter, a warning when only some do.
fn check_optimizer_load(load: OptimizerLoad, step: usize) -> Result<Option<String>> {
    if load.matched == 0 {
        if step == 0 && load.unused == 0 {
            // Saved before the first optimizer step: there is nothing to restore.
            return Ok(None);
        }
        return Err(format!(
            "none of the {} saved optimizer entries belongs to one of the {} trainable parameters; the state was saved for a different model or --train-modules",
            load.unused, load.missing
        )
        .into());
    }
    if load.is_complete() {
        return Ok(None);
    }
    Ok(Some(format!(
        "optimizer state matched {} parameters; {} start with empty moments and {} saved entries were not used",
        load.matched, load.missing, load.unused
    )))
}

/// The metadata `train` writes next to the tensors it saves: the LoRA
/// settings, or, when base weights train, the modules that trained.
fn saved_metadata(lora: &LoraMetadata, train_modules: &str) -> LoraMetadata {
    if train_modules.eq_ignore_ascii_case("lora") {
        lora.clone()
    } else {
        LoraMetadata::trained_weights(&train_modules.to_ascii_lowercase())
    }
}

/// The settings stored in an adapter or trained-weights file.
fn adapter_metadata(path: &Path) -> Result<LoraMetadata> {
    read_adapter(path)?.metadata.ok_or_else(|| {
        format!(
            "{} has no LoRA metadata (rank/alpha/targets); it was not written by ferrotensor-cli",
            path.display()
        )
        .into()
    })
}

// ---------------------------------------------------------------------------
// export-adapter

/// Convert an adapter file without loading the model's weights or a GPU.
fn export_adapter(args: ExportAdapterArgs) -> Result<()> {
    let config_path = args.model.join("config.json");
    let config = ModelLoader::load_config(&config_path)
        .map_err(|e| format!("cannot read {}: {e}", config_path.display()))?;
    let ckpt = load_lora_checkpoint(&args.lora)?;
    if let Some(modules) = ckpt
        .metadata
        .as_ref()
        .and_then(|m| m.trained_modules.as_deref())
    {
        return Err(format!(
            "{} holds base weights trained with --train-modules {modules}, not a LoRA adapter; there is nothing to export to PEFT",
            args.lora.display()
        )
        .into());
    }
    let base_model = args
        .base_model
        .unwrap_or_else(|| args.model.display().to_string());
    match args.format {
        AdapterFormat::Peft => {
            let export = export_peft_adapter(&ckpt, &config, &base_model, &args.output)
                .map_err(|e| format!("{}: {e}", args.lora.display()))?;
            println!(
                "wrote {} ({} tensors, r {}, lora_alpha {}, targets {})",
                args.output.display(),
                export.tensors,
                export.rank,
                export.alpha,
                export.target_modules.join(",")
            );
        }
    }
    Ok(())
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
        eprintln!("{}", loaded_line(path, n, meta.as_ref()));
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
        eprintln!("{}", loaded_line(path, n, meta.as_ref()));
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
            // Llama blocks name the MLP `mlp`, LFM2 blocks `feed_forward`.
            let keys = match m {
                "attn" | "attention" => vec!["self_attn"],
                "mlp" => vec![".mlp.", ".feed_forward."],
                "head" => vec!["lm_head"],
                other => vec![other],
            };
            model
                .named_params()
                .into_iter()
                .filter(|(n, _)| keys.iter().any(|k| n.contains(k)))
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
        trained_modules: None,
    };
    let mut memory = args.memory_report.map(|steps| {
        let json = is_root.then(|| args.out.join(memory_report::FILE_NAME));
        memory_report::MemoryReportRun::start(&base.ctx, steps, json)
    });
    let model = {
        let _tag = base.ctx.memory().scope(MemoryCategory::BaseWeights);
        build_model(&base, lora_on.then_some(&meta), !lora_on)?
    };
    let named_params = select_train_params(&model, &args.train_modules)?;
    let train_params: Vec<_> = {
        let mut names: Vec<&String> = named_params.keys().collect();
        // Sorted by the names Llama blocks had before they took Hugging Face
        // names, so the parameter order (and the gradient-norm sum) is the
        // same as in earlier releases.
        names.sort_by_cached_key(|n| legacy_llama_name(n).unwrap_or_else(|| n.to_string()));
        names.into_iter().map(|n| named_params[n].clone()).collect()
    };
    ferrotensor_trainer::memory::tag_trainable(&train_params);

    fs::create_dir_all(&args.out)
        .map_err(|e| format!("cannot create {}: {e}", args.out.display()))?;
    if let Some(m) = memory.as_mut() {
        m.after_load()?;
    }
    let adapter_path = args.save_lora.clone().unwrap_or_else(|| {
        args.out.join(if lora_on {
            "lora_adapters.safetensors"
        } else {
            "trained_weights.safetensors"
        })
    });
    let saved_meta = saved_metadata(&meta, &args.train_modules);
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

    // Resume: adapter or trained base weights, optimizer moments, step counter.
    let mut start_step = 0usize;
    if let Some(resume) = &args.resume {
        let file = read_adapter(resume)?;
        ensure_not_empty(&file, resume)?;
        check_resume_file(file.metadata.as_ref(), &args.train_modules)
            .map_err(|e| format!("{}: {e}", resume.display()))?;
        let n = restore_checkpoint(&base, &trainer.model, &file, resume)?;
        let resume_state = state_path_for(resume);
        if resume_state.is_file() {
            let ckpt = TrainingCheckpoint::load(&resume_state)?;
            ckpt.training_state.validate_resume(
                &args.train_modules,
                args.data.task.as_str(),
                args.data.seq_len,
                &args.model.model.display().to_string(),
            )?;
            if let Some(mut opt) = ckpt.optimizer_state.clone() {
                upgrade_optimizer_names(&mut opt, &named_params);
                let _tag = base.ctx.memory().scope(MemoryCategory::OptimizerState);
                let load = trainer
                    .optimizer
                    .load_state(&base.ctx, &named_params, &opt)
                    .map_err(|e| format!("optimizer state in {}: {e}", resume_state.display()))?;
                let verdict = check_optimizer_load(load, ckpt.training_state.step)
                    .map_err(|e| format!("optimizer state in {}: {e}", resume_state.display()))?;
                if let Some(warning) = verdict {
                    log::warn!("{}: {warning}", resume_state.display());
                }
            }
            start_step = ckpt.training_state.step;
            args.seed = ckpt.training_state.seed;
            trainer.set_optimizer_steps(start_step);
        } else if trainer.model.layers.iter().any(|l| {
            matches!(&l.op, ferrotensor_graph::lfm::Lfm2LayerOp::Attention(a) if a.dropout_prob > 0.0)
        }) {
            eprintln!(
                "warning: no training state at {}, so the step counter is not restored: \
                 attention-dropout masks restart from step 0 (seeded by --seed {}) \
                 instead of continuing the original run's sequence",
                resume_state.display(),
                args.seed
            );
        }
        eprintln!(
            "resume: {} ({n} tensors), continuing at step {start_step}",
            resume.display()
        );
    }
    // Attention-dropout masks follow the run seed (restored on resume).
    trainer.dropout_seed = args.seed;

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
        // LoRA runs save the adapters; runs that train base weights save
        // exactly the tensors that train.
        let saved_params = if lora_on {
            trainer.model.named_lora_params()
        } else {
            named_params.clone()
        };
        pollster::block_on(save_lora_checkpoint(
            &base.ctx,
            &adapter_path,
            &saved_params,
            Some(&saved_meta),
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
    if let Some(m) = &memory {
        m.begin_steps();
    }

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
                if let Some(m) = memory.as_mut() {
                    m.after_step(step, step - start_step)?;
                }
                if step.is_multiple_of(args.log_every.max(1)) || step == total_steps {
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
                if args.save_every > 0 && step.is_multiple_of(args.save_every) && step < total_steps
                {
                    save(&trainer, step, best)?;
                }
                if let Some(ds) = &eval_ds {
                    if args.eval_every > 0
                        && step.is_multiple_of(args.eval_every)
                        && step < total_steps
                    {
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
    fn gpu_memory_loss_is_recognised() {
        assert!(looks_like_gpu_memory_loss(
            "Error in Queue::submit: Validation Error\n\nCaused by:\n    Parent device is lost"
        ));
        assert!(looks_like_gpu_memory_loss(
            "Not enough memory left: OutOfMemory"
        ));
        assert!(!looks_like_gpu_memory_loss("expected train"));
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
        assert_eq!(t.memory_report, None);
    }

    #[test]
    fn memory_report_takes_an_optional_step_count() {
        let steps = |extra: &[&str]| {
            let mut args = vec!["train", "--model", "m", "--data", "d.jsonl"];
            args.extend_from_slice(extra);
            match parse(&args).expect("parse").command {
                Command::Train(t) => t.memory_report,
                _ => panic!("expected train"),
            }
        };
        assert_eq!(steps(&["--memory-report"]), Some(3));
        assert_eq!(steps(&["--memory-report=5"]), Some(5));
        assert_eq!(steps(&["--memory-report", "--lite"]), Some(3));
        assert_eq!(steps(&[]), None);
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
    fn info_parses_its_flags() {
        let Command::Info(i) = parse(&["info", "--check", "--json"])
            .expect("parse")
            .command
        else {
            panic!("expected info");
        };
        assert!(i.check && i.json);
        let Command::Info(i) = parse(&["info"]).expect("parse").command else {
            panic!("expected info");
        };
        assert!(!i.check && !i.json);
    }

    #[test]
    fn export_adapter_parses_its_flags() {
        let cli = parse(&[
            "export-adapter",
            "--adapter",
            "a.safetensors",
            "--model",
            "m",
            "--output",
            "out",
            "--format",
            "peft",
        ])
        .expect("parse");
        let Command::ExportAdapter(e) = cli.command else {
            panic!("expected export-adapter");
        };
        assert_eq!(e.lora, PathBuf::from("a.safetensors"));
        assert_eq!(e.output, PathBuf::from("out"));
        assert_eq!(e.format, AdapterFormat::Peft);
        assert!(e.base_model.is_none());
        assert!(parse(&["export-adapter", "--lora", "a", "--model", "m"]).is_err());
    }

    #[test]
    fn legacy_demo_flag_still_works() {
        assert!(matches!(
            parse(&["--demo"]).expect("parse").command,
            Command::Demo
        ));
    }

    #[test]
    fn classify_train_has_the_documented_defaults() {
        let cli = parse(&[
            "classify-train",
            "--model",
            "m",
            "--data",
            "d.jsonl",
            "--output",
            "out",
        ])
        .expect("parse");
        let Command::ClassifyTrain(t) = cli.command else {
            panic!("expected classify-train");
        };
        assert_eq!(t.taps, None);
        assert_eq!(t.max_len, 384);
        assert_eq!((t.rank, t.alpha), (16, 32.0));
        assert_eq!(
            t.lora_targets,
            vec![
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj"
            ]
        );
        assert!((t.lr - 1e-4).abs() < 1e-12 && (t.head_lr - 1e-3).abs() < 1e-12);
        assert_eq!((t.epochs, t.batch_size, t.grad_accum), (2, 8, 2));
        assert_eq!((t.clip_grad_norm, t.warmup_steps, t.seed), (1.0, 0, 0));
        assert_eq!(t.log_every, 10);
        assert!(
            !t.no_skip_first_token,
            "the first token is skipped by default"
        );
        assert!(t.batches.is_none() && t.eval.is_none());
        assert!(!t.model.f32_weights, "f16 base weights by default");
    }

    #[test]
    fn classify_train_parses_flags() {
        let cli = parse(&[
            "classify-train",
            "--model",
            "m",
            "--data",
            "d.jsonl",
            "--output",
            "out",
            "--taps",
            "4,8",
            "--lora-targets",
            "q_proj,v_proj",
            "--no-skip-first-token",
            "--batches",
            "plan.jsonl",
            "--eval",
            "val.jsonl",
            "--f32-weights",
        ])
        .expect("parse");
        let Command::ClassifyTrain(t) = cli.command else {
            panic!("expected classify-train");
        };
        assert_eq!(t.taps, Some(vec![4, 8]));
        assert_eq!(t.lora_targets, vec!["q_proj", "v_proj"]);
        assert!(t.no_skip_first_token);
        assert_eq!(t.batches, Some(PathBuf::from("plan.jsonl")));
        assert_eq!(t.eval, Some(PathBuf::from("val.jsonl")));
        assert!(t.model.f32_weights);

        // The last of the two skip flags wins.
        let base = [
            "classify-train",
            "--model",
            "m",
            "--data",
            "d",
            "--output",
            "o",
        ];
        let last = |flags: &[&str]| {
            let args: Vec<&str> = base.iter().chain(flags).copied().collect();
            match parse(&args).expect("parse").command {
                Command::ClassifyTrain(t) => !t.no_skip_first_token,
                _ => panic!("expected classify-train"),
            }
        };
        assert!(last(&["--no-skip-first-token", "--skip-first-token"]));
        assert!(!last(&["--skip-first-token", "--no-skip-first-token"]));

        assert!(parse(&["classify-train", "--model", "m", "--data", "d"]).is_err());
        assert!(parse(&[
            "classify-train",
            "--model",
            "m",
            "--data",
            "d",
            "--output",
            "o",
            "--taps",
            "x"
        ])
        .is_err());
    }

    #[test]
    fn classify_predict_parses_flags() {
        let cli = parse(&[
            "classify-predict",
            "--model",
            "m",
            "--classifier",
            "runs/cls",
            "--data",
            "test.jsonl",
            "--output",
            "probs.jsonl",
        ])
        .expect("parse");
        let Command::ClassifyPredict(p) = cli.command else {
            panic!("expected classify-predict");
        };
        assert_eq!(p.classifier, PathBuf::from("runs/cls"));
        assert_eq!(p.output, PathBuf::from("probs.jsonl"));
        assert_eq!(p.batch_size, 8);
        assert!(parse(&[
            "classify-predict",
            "--model",
            "m",
            "--data",
            "d",
            "--output",
            "o"
        ])
        .is_err());
    }

    #[test]
    fn taps_must_be_distinct_blocks_of_the_model() {
        assert!(classify::check_taps(&[2, 4], 4).is_ok());
        assert!(classify::check_taps(&[], 4).is_err());
        assert!(classify::check_taps(&[0], 4).is_err());
        assert!(classify::check_taps(&[5], 4).is_err());
        assert!(classify::check_taps(&[3, 3], 4)
            .expect_err("duplicate")
            .contains("twice"));
    }

    #[test]
    fn state_path_sits_next_to_adapter() {
        assert_eq!(
            state_path_for(Path::new("out/a.safetensors")),
            PathBuf::from("out/a.safetensors.state.json")
        );
    }

    fn lora_meta() -> LoraMetadata {
        LoraMetadata {
            rank: 8,
            alpha: 16.0,
            target_modules: vec!["q_proj".into()],
            trained_modules: None,
        }
    }

    #[test]
    fn lora_runs_save_their_lora_settings_and_base_runs_the_trained_modules() {
        assert_eq!(saved_metadata(&lora_meta(), "lora"), lora_meta());
        assert_eq!(saved_metadata(&lora_meta(), "LoRA"), lora_meta());
        let weights = saved_metadata(&lora_meta(), "Attn");
        assert_eq!(weights, LoraMetadata::trained_weights("attn"));
        assert_eq!(weights.trained_modules.as_deref(), Some("attn"));
        assert!(weights.target_modules.is_empty());
    }

    #[test]
    fn resume_needs_a_file_of_the_same_kind() {
        let weights = LoraMetadata::trained_weights("attn");
        let ok = |meta: Option<&LoraMetadata>, modules: &str| check_resume_file(meta, modules);
        assert!(ok(Some(&lora_meta()), "lora").is_ok());
        assert!(ok(None, "lora").is_ok());
        assert!(ok(Some(&weights), "attn").is_ok());
        assert!(ok(Some(&weights), "ATTN").is_ok());

        let err = ok(Some(&weights), "lora").unwrap_err().to_string();
        assert!(
            err.contains("not a LoRA adapter") && err.contains("attn"),
            "{err}"
        );
        let err = ok(Some(&weights), "mlp").unwrap_err().to_string();
        assert!(err.contains("attn") && err.contains("mlp"), "{err}");
        let err = ok(Some(&lora_meta()), "all").unwrap_err().to_string();
        assert!(
            err.contains("LoRA adapter") && err.contains("--train-modules lora"),
            "{err}"
        );
        assert!(ok(None, "all").is_err());
    }

    #[test]
    fn optimizer_state_that_fits_nothing_is_an_error() {
        let load = |matched, missing, unused| OptimizerLoad {
            matched,
            missing,
            unused,
        };
        assert_eq!(check_optimizer_load(load(4, 0, 0), 10).unwrap(), None);

        let err = check_optimizer_load(load(0, 4, 8), 10)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("none") && err.contains('8') && err.contains('4'),
            "{err}"
        );

        let warning = check_optimizer_load(load(3, 1, 2), 10)
            .unwrap()
            .expect("warning");
        assert!(
            warning.contains("matched 3")
                && warning.contains("1 start")
                && warning.contains("2 saved"),
            "{warning}"
        );

        // Saved before the first optimizer step: no moments to restore.
        assert_eq!(check_optimizer_load(load(0, 4, 0), 0).unwrap(), None);
        assert!(check_optimizer_load(load(0, 4, 0), 5).is_err());
        assert!(check_optimizer_load(load(0, 4, 2), 0).is_err());
    }

    #[test]
    fn loaded_line_names_what_was_loaded() {
        let p = Path::new("out/x.safetensors");
        let line = loaded_line(p, 8, Some(&LoraMetadata::trained_weights("mlp")));
        assert!(
            line.starts_with("weights:") && line.contains("--train-modules mlp"),
            "{line}"
        );
        assert_eq!(
            loaded_line(p, 8, Some(&lora_meta())),
            "lora:   out/x.safetensors (8 tensors)"
        );
    }
}
