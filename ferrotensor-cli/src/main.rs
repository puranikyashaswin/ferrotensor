//! `ferrotensor-cli`: LoRA fine-tuning, evaluation and text generation for small
//! Llama-style (and LFM2) models on consumer GPUs.
//!
//! ```text
//! ferrotensor-cli finetune --model DIR --data train.jsonl --out runs/sft
//! ferrotensor-cli train    --model DIR --data train.jsonl --out runs/sft
//! ferrotensor-cli eval     --model DIR --data val.jsonl --lora runs/sft/lora_adapters.safetensors
//! ferrotensor-cli generate --model DIR --lora runs/sft/lora_adapters.safetensors --prompt "Hi"
//! ferrotensor-cli chat     --model DIR --lora runs/sft/lora_adapters.safetensors
//! ferrotensor-cli classify-train   --model DIR --data train.jsonl --output runs/cls
//! ferrotensor-cli classify-predict --model DIR --classifier runs/cls --data test.jsonl --output probs.jsonl
//! ferrotensor-cli info --check
//! ferrotensor-cli plan --model DIR --json
//! ferrotensor-cli pretrain --data corpus.txt --out runs/tiny
//! ferrotensor-cli export-adapter --lora runs/sft/lora_adapters.safetensors --model DIR --output runs/sft/peft
//! ferrotensor-cli export-gguf --model DIR --out model.gguf
//! ```

mod chat;
mod classify;
mod export_gguf;
mod finetune;
mod info;
mod memory_report;
mod merge;
mod model_id;
mod plan;
mod pretrain;
mod profile_report;

use clap::{Args, CommandFactory, FromArgMatches, Parser, Subcommand, ValueEnum};
use ferrotensor_autograd::{no_grad, AdamW, OptimizerLoad, WarmupCosineScheduler};
use ferrotensor_backend::{
    ComputeEngine, ContextError, MemoryCategory, UnifiedTensor, WgpuContext,
};
use ferrotensor_graph::lfm::{legacy_llama_name, Lfm2Model};
use ferrotensor_graph::model_builder::{check_supported, Lfm2ModelBuilder, LoRaConfig};
use ferrotensor_graph::{KVCache, KVCacheLayer};
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
use model_id::resolve_model;
use std::collections::HashMap;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const MIB: u64 = 1024 * 1024;

/// Process exit codes, the same for every command. clap's usage errors also
/// exit with 2.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Exit {
    /// The command did what was asked.
    Success = 0,
    /// It failed while running: I/O, the GPU, a kernel out of tolerance.
    RuntimeError = 1,
    /// The input is invalid, or this machine cannot meet the request.
    Unmet = 2,
}

impl Exit {
    /// The code for a command that failed with `error`.
    fn for_error(error: &(dyn std::error::Error + 'static)) -> Self {
        // No adapter is this machine, as in `info` and `plan`; a device the
        // adapter refused is a runtime failure.
        let no_adapter = matches!(
            error.downcast_ref::<ContextError>(),
            Some(ContextError::NoAdapter { .. })
        );
        if no_adapter || error.is::<Unmet>() {
            Self::Unmet
        } else {
            Self::RuntimeError
        }
    }
}

/// An error the caller fixes by changing the request; exits with
/// [`Exit::Unmet`].
#[derive(Debug)]
struct Unmet(String);

impl std::fmt::Display for Unmet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Unmet {}

fn unmet(message: impl Into<String>) -> Box<dyn std::error::Error> {
    Box::new(Unmet(message.into()))
}

/// Prints `value` as the one JSON object a `--json` run writes to stdout.
fn print_json(value: &impl serde::Serialize) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

/// Writes generated text as it arrives: to stdout, or to stderr when stdout
/// carries `--json` output.
fn stream_text(text: &str, json: bool) {
    if json {
        eprint!("{text}");
        return;
    }
    let mut stdout = std::io::stdout();
    if let Err(e) = write!(stdout, "{text}").and_then(|()| stdout.flush()) {
        log::warn!("cannot write generated text: {e}");
    }
}

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
    /// `train` with the batch size, sequence length, gradient accumulation
    /// and block checkpointing that `plan` picks for this machine, except
    /// those given on the command line.
    Finetune(Box<finetune::FinetuneArgs>),
    /// LoRA fine-tuning (SFT with response-only loss) or LM pretraining.
    Train(Box<TrainArgs>),
    /// Mean loss and perplexity of a model (plus optional adapter) on a dataset.
    Eval(EvalArgs),
    /// Generate text from a prompt (top-k / top-p / temperature sampling).
    #[command(visible_alias = "infer")]
    Generate(GenerateArgs),
    /// A multi-turn conversation with a ChatML model, one user message per
    /// line of stdin.
    Chat(chat::ChatArgs),
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
    /// Pick the largest LoRA batch size, sequence length and checkpointing
    /// setting expected to fit this machine's GPU memory.
    Plan(plan::PlanArgs),
    /// Convert a LoRA adapter to another format (PEFT, for PyTorch).
    ExportAdapter(ExportAdapterArgs),
    /// Write a model directory with a LoRA adapter folded into the weights.
    Merge(merge::MergeArgs),
    /// Write a Llama-style model as one GGUF file for llama.cpp and Ollama.
    ExportGguf(export_gguf::ExportGgufArgs),
    /// Train a small byte-level Llama-style model from random weights on a
    /// text file, into a model directory that generate loads.
    Pretrain(Box<pretrain::PretrainArgs>),
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
    /// Model directory (config.json, tokenizer.json, *.safetensors), or a
    /// Hugging Face model id already in the local cache.
    #[arg(long, value_name = "DIR_OR_ID")]
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
    /// Store the frozen weights of the decoder blocks at 4 bits (affine,
    /// groups of 64, about 4.5 bits per weight; computation stays f32).
    /// This changes the model's outputs: it computes with the decoded
    /// weights. lm_head stays f16; embeddings, norms and LoRA adapters stay
    /// f32. Not allowed when base weights train. Use the same setting to
    /// evaluate or generate with an adapter trained this way.
    #[arg(long, conflicts_with = "f32_weights")]
    q4_weights: bool,
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
    /// lm_head, embed). `all` is every base weight but the token embedding,
    /// `embed` only the embedding, `full` both. With tied embeddings lm_head
    /// is a separate copy: training one does not change the other. Trained
    /// base weights are saved in f32, only the tensors that trained, and load
    /// with --lora in eval and generate or with --resume.
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
    /// Recompute each decoder block during backward instead of keeping its
    /// activations: less GPU memory for a second forward pass through every
    /// block per step. Printed losses and gradient norms are unchanged in the
    /// measured runs.
    #[arg(long)]
    checkpoint_blocks: bool,
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
    /// Count GPU work per optimizer step; after the run, print the step
    /// with the median wall time and write OUT/profile.json.
    #[arg(long)]
    profile: bool,
    /// Print one JSON summary to stdout when the run ends; everything else
    /// goes to stderr.
    #[arg(long)]
    json: bool,
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
    /// Print the result as one JSON object.
    #[arg(long)]
    json: bool,
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
    /// Base model directory or cached Hugging Face model id the adapter was
    /// trained on (only config.json is read).
    #[arg(long, value_name = "DIR_OR_ID")]
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
    #[command(flatten)]
    sampling: SamplingArgs,
    /// Recompute the whole sequence every token instead of using the KV cache.
    #[arg(long)]
    no_kv_cache: bool,
    /// Stream the text to stderr and print one JSON object with it to stdout.
    #[arg(long)]
    json: bool,
}

/// Sampling flags shared by `generate` and `chat`.
#[derive(Args, Debug)]
struct SamplingArgs {
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
}

impl SamplingArgs {
    fn sampler(&self) -> Result<Sampler> {
        if !(0.0..=1.0).contains(&self.top_p) || self.temperature < 0.0 {
            return Err(unmet("--top-p must be in [0, 1] and --temperature >= 0"));
        }
        Ok(Sampler::new(
            self.temperature,
            self.top_k,
            self.top_p,
            self.seed,
        ))
    }
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
    // Parsed through ArgMatches so `finetune` can tell flags given on the
    // command line from defaults.
    let matches = Cli::command().get_matches();
    let cli = Cli::from_arg_matches(&matches).unwrap_or_else(|e| e.exit());
    let json = match &cli.command {
        Command::Finetune(args) => args.train.json,
        Command::Train(args) => args.json,
        Command::Eval(args) => args.json,
        Command::Generate(args) => args.json,
        Command::Chat(args) => args.json,
        Command::Info(args) => args.json,
        Command::Plan(args) => args.json,
        Command::Merge(args) => args.json,
        Command::ExportGguf(args) => args.json,
        Command::Pretrain(args) => args.json,
        _ => false,
    };
    // `chat --json` writes one object per line, so its error is one line too.
    let one_line = matches!(cli.command, Command::Chat(_));
    let code = match run(cli.command, &matches, json) {
        Ok(code) => code,
        Err(e) => {
            if json {
                let error = serde_json::json!({ "error": e.to_string() });
                if one_line {
                    println!("{error}");
                } else {
                    println!("{error:#}");
                }
            } else {
                eprintln!("error: {e}");
            }
            Exit::for_error(e.as_ref())
        }
    };
    std::process::exit(code as i32);
}

fn run(command: Command, matches: &clap::ArgMatches, json: bool) -> Result<Exit> {
    match command {
        Command::Finetune(args) => {
            let user_set = matches
                .subcommand_matches("finetune")
                .map_or_else(finetune::UserSet::default, finetune::UserSet::from_matches);
            finetune::run(*args, user_set)?;
        }
        Command::Train(args) => {
            let summary = pollster::block_on(train(*args))?;
            if json {
                print_json(&summary)?;
            }
        }
        Command::Eval(args) => pollster::block_on(eval(args))?,
        Command::Generate(args) => pollster::block_on(generate(args))?,
        Command::Chat(args) => pollster::block_on(chat::run(args))?,
        Command::ClassifyTrain(args) => pollster::block_on(classify::classify_train(*args))?,
        Command::ClassifyPredict(args) => pollster::block_on(classify::classify_predict(args))?,
        Command::Demo => ferrotensor_trainer::run_demo(ferrotensor_trainer::DemoConfig::default()),
        Command::Info(args) => return info::run(args),
        Command::Plan(args) => return plan::run(args),
        Command::ExportAdapter(args) => export_adapter(args)?,
        Command::Merge(args) => merge::run(args, json)?,
        Command::ExportGguf(args) => export_gguf::run(args, json)?,
        Command::Pretrain(args) => pollster::block_on(pretrain::run(*args))?,
        Command::Monitor => ferrotensor_monitor::run().map_err(|e| format!("monitor: {e}"))?,
        Command::Benchmark => {
            println!(
                "Run the kernel benchmarks with: cargo run -p ferrotensor-benchmarks --release"
            );
        }
    }
    Ok(Exit::Success)
}

// ---------------------------------------------------------------------------
// Model loading

struct Loaded {
    ctx: Arc<WgpuContext>,
    engine: Arc<ComputeEngine>,
    tokenizer: Tokenizer,
    config: Lfm2Config,
    /// The memory-mapped checkpoint, until `build_model` takes it.
    weights: std::cell::Cell<Option<WeightsContext>>,
    f16_weights: bool,
    q4_weights: bool,
}

async fn load_base(args: &ModelArgs) -> Result<Loaded> {
    let dir = &resolve_model(&args.model)?;
    let config_path = dir.join("config.json");
    let config = ModelLoader::load_config(&config_path)
        .map_err(|e| unmet(format!("cannot read {}: {e}", config_path.display())))?;
    check_supported(&config).map_err(|e| unmet(format!("{}: {e}", config_path.display())))?;
    let tok_path = dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tok_path)
        .map_err(|e| format!("cannot load {}: {e}", tok_path.display()))?;
    let mut weights = WeightsContext::new();
    weights.load_model_dir(dir).map_err(|e| {
        let unfinished = if dir.join(pretrain::INIT_DIR).is_dir() {
            "; it is an unfinished `pretrain` run: `pretrain --resume` with the run's flags continues it and writes model.safetensors, and with --max-steps at or below the step it saved (training_state.json) it writes the saved weights without training further"
        } else {
            ""
        };
        format!("cannot load weights from {}: {e}{unfinished}", dir.display())
    })?;

    let (power, fallback) = match args.device {
        Device::Auto | Device::Gpu => (
            ferrotensor_backend::wgpu::PowerPreference::HighPerformance,
            false,
        ),
        Device::Cpu => (ferrotensor_backend::wgpu::PowerPreference::LowPower, true),
    };
    let ctx = Arc::new(WgpuContext::try_new_with_options(power, fallback).await?);
    let engine = Arc::new(ComputeEngine::new(&ctx));
    eprintln!("device: {}", ctx.adapter_info());
    eprintln!(
        "model:  {} ({} layers, hidden {}, vocab {})",
        dir.display(),
        config.num_hidden_layers,
        config.hidden_size,
        config.vocab_size
    );
    if args.q4_weights {
        eprintln!("weights: 4-bit decoder blocks, f16 lm_head");
    }
    Ok(Loaded {
        ctx,
        engine,
        tokenizer,
        config,
        weights: std::cell::Cell::new(Some(weights)),
        f16_weights: !args.f32_weights,
        q4_weights: args.q4_weights,
    })
}

/// Build the model. `saved` is the metadata of the file the model will load,
/// if any: LoRA settings give the model adapters of that shape; trained base
/// weights give it none and store every weight as f32 (what the file holds,
/// and what training used), whatever `--f32-weights` or `--q4-weights` says.
fn build_model(
    base: &Loaded,
    saved: Option<&LoraMetadata>,
    trainable_base: bool,
) -> Result<Lfm2Model> {
    // Taken, so the checkpoint is unmapped when the model is built: the
    // model holds its own copies of the weights, and nothing reads the file
    // afterwards.
    let weights = base
        .weights
        .take()
        .ok_or("the checkpoint was already used to build a model")?;
    let weights_file = saved.is_some_and(|m| m.trained_modules.is_some());
    let mut builder = Lfm2ModelBuilder::new(
        base.ctx.clone(),
        base.engine.clone(),
        &weights,
        base.config.clone(),
    )
    .with_trainable_base(trainable_base)
    .with_f16_weights(base.f16_weights && !weights_file)
    .with_q4_weights(base.q4_weights && !weights_file);
    if let Some(meta) = saved.filter(|m| m.trained_modules.is_none()) {
        builder = builder.with_lora(LoRaConfig {
            rank: meta.rank,
            alpha: meta.alpha,
            target_modules: meta.target_modules.clone(),
        });
    }
    // Every build error is a checkpoint this code cannot load: a missing or
    // misshapen weight, or a feature it does not implement.
    builder.build().map_err(|e| unmet(e.to_string()))
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
    let restored = apply_checkpoint(&base.ctx, ckpt, &params)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    if params.contains_key(EMBED_TOKENS) {
        // The lookup of a frozen table reads a host copy made at build time.
        model.clear_host_embeddings();
    }
    Ok(restored)
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
    let config_path = resolve_model(&args.model)?.join("config.json");
    let config = ModelLoader::load_config(&config_path)
        .map_err(|e| unmet(format!("cannot read {}: {e}", config_path.display())))?;
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
        return Err(unmet(format!(
            "dataset file {} does not exist",
            path.display()
        )));
    }
    if args.seq_len < 2 {
        return Err(unmet("--seq-len must be at least 2"));
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
            return Err(unmet(
                "SFT expects .json/.jsonl with {\"prompt\", \"response\"}; use --task pretrain for .txt/.bin",
            ))
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
                    .collect::<std::result::Result<_, _>>()?;
                TextDataset::pack_sequences(seqs, seq)
            } else {
                return Err(unmet(
                    "pretrain .json must be [token ids] or [\"text\", ...]",
                ));
            }
        }
        (Task::Pretrain, _) => {
            let text = fs::read_to_string(path)?;
            let seqs = text
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(|l| tokenizer.encode(l))
                .collect::<std::result::Result<_, _>>()?;
            TextDataset::pack_sequences(seqs, seq)
        }
    };
    let ds = match args.max_samples {
        Some(n) => take_first(ds, n),
        None => ds,
    };
    if ds.is_empty() {
        return Err(unmet(format!(
            "{} produced no usable examples (empty, or every prompt is longer than --seq-len {seq})",
            path.display()
        )));
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
        )? {
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
        let logits = no_grad(|| model.forward(&base.engine, &ex.input_ids, 0, None, None));
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
    let summary = EvalSummary {
        examples: ds.len(),
        tokens,
        loss,
        perplexity: loss.exp(),
        seconds: t.elapsed().as_secs_f32(),
    };
    if args.json {
        return print_json(&summary);
    }
    println!(
        "eval: {} examples, {tokens} tokens, loss {loss:.4}, perplexity {:.3} ({:.1}s)",
        summary.examples, summary.perplexity, summary.seconds
    );
    Ok(())
}

/// What `eval --json` prints. With no supervised tokens the loss is NaN,
/// which serializes as null.
#[derive(serde::Serialize, Debug)]
struct EvalSummary {
    examples: usize,
    tokens: usize,
    loss: f32,
    perplexity: f32,
    seconds: f32,
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
    let mut sampler = args.sampling.sampler()?;
    let prompt = if args.raw {
        let ids = base.tokenizer.encode(&args.prompt)?;
        Prompt::raw(&base.tokenizer, &base.config, ids)
    } else {
        let ids = base
            .tokenizer
            .encode_chat(args.system.as_deref(), &args.prompt)?;
        Prompt::chatml(&base.tokenizer, &base.config, ids)
    };
    let mut max_new = args.sampling.max_new_tokens;
    if let Some(limit) = base.config.max_position_embeddings {
        let fits = new_tokens_within(limit, prompt.ids.len(), max_new).ok_or_else(|| {
            unmet(format!(
                "the prompt is {} tokens, longer than the model's {limit} positions",
                prompt.ids.len()
            ))
        })?;
        if fits < max_new {
            eprintln!(
                "lowered --max-new-tokens from {max_new} to {fits} to fit the model's {limit} positions"
            );
            max_new = fits;
        }
    }
    let mut cache = if args.no_kv_cache {
        None
    } else {
        DecodeCache::new(&model, prompt.ids.len() + max_new)
    };
    let kv_cache = cache.is_some();
    let mut text = String::new();
    let decoded = generate_tokens(
        &base,
        &model,
        &prompt,
        max_new,
        &mut sampler,
        cache.as_mut(),
        |delta| {
            text.push_str(delta);
            stream_text(delta, args.json);
        },
    )
    .await?;
    stream_text("\n", args.json);
    let summary = GenerateSummary::new(text, prompt.ids.len(), &decoded, kv_cache);
    eprintln!("{}", summary.stats_line());
    if args.json {
        print_json(&summary)?;
    }
    Ok(())
}

/// What `generate --json` prints.
#[derive(serde::Serialize, Debug)]
struct GenerateSummary {
    text: String,
    prompt_tokens: usize,
    new_tokens: usize,
    /// Decode speed after the prompt; NaN (null) when no decode step ran.
    tokens_per_second: f32,
    prefill_seconds: f32,
    kv_cache: bool,
}

impl GenerateSummary {
    fn new(text: String, prompt_tokens: usize, decoded: &Decoded, kv_cache: bool) -> Self {
        let tokens_per_second = if decoded.decode_steps == 0 {
            f32::NAN
        } else {
            decoded.decode_steps as f32 / decoded.decode.as_secs_f32().max(1e-6)
        };
        Self {
            text,
            prompt_tokens,
            new_tokens: decoded.new_tokens,
            tokens_per_second,
            prefill_seconds: decoded.prefill.as_secs_f32(),
            kv_cache,
        }
    }

    /// The line `generate` and `chat` print to stderr after a reply.
    fn stats_line(&self) -> String {
        let speed = if self.tokens_per_second.is_nan() {
            "no decode step".to_string()
        } else {
            format!("{:.1} tok/s", self.tokens_per_second)
        };
        format!(
            "[{} prompt + {} new tokens, prefill {:.2}s, {speed}, kv cache {}]",
            self.prompt_tokens,
            self.new_tokens,
            self.prefill_seconds,
            if self.kv_cache { "on" } else { "off" }
        )
    }
}

/// How many of `max_new` tokens fit after a prompt of `prompt_len` tokens in
/// a context of `limit` positions; `None` when the prompt alone is longer
/// than `limit`.
fn new_tokens_within(limit: usize, prompt_len: usize, max_new: usize) -> Option<usize> {
    limit.checked_sub(prompt_len).map(|room| room.min(max_new))
}

/// A KV cache and the token ids whose keys and values it holds, at positions
/// `0..ids.len()`.
struct DecodeCache {
    kv: KVCache,
    capacity: usize,
    ids: Vec<u32>,
}

impl DecodeCache {
    /// An empty cache for `capacity` positions; `None` when the model has
    /// conv blocks.
    fn new(model: &Lfm2Model, capacity: usize) -> Option<Self> {
        model.new_kv_cache(capacity).map(|kv| Self {
            kv,
            capacity,
            ids: Vec::new(),
        })
    }

    /// Keeps the cached ids that `prompt` starts with, leaving at least the
    /// last prompt token to run for its logits, and makes room for `len`
    /// positions. Returns how many prompt tokens the cache holds.
    fn reuse(&mut self, base: &Loaded, prompt: &[u32], len: usize) -> usize {
        // Keys and values at a position depend only on the tokens up to it,
        // so they stay valid for any prompt with the same prefix.
        let common = self.ids.iter().zip(prompt).take_while(|(a, b)| a == b);
        let keep = common.count().min(prompt.len().saturating_sub(1));
        self.ids.truncate(keep);
        if len > self.capacity {
            let doubled = 2 * self.capacity;
            let limit = base.config.max_position_embeddings;
            let capacity = limit.map_or(doubled, |l| doubled.min(l)).max(len);
            let grow = |old: &UnifiedTensor| {
                let new = UnifiedTensor::zeros(&base.ctx, &[1, capacity, old.shape[2]]);
                base.engine.copy_tensor(&base.ctx, old, &new, 0);
                new
            };
            for layer in &mut self.kv.layers {
                *layer = KVCacheLayer {
                    key: grow(&layer.key),
                    value: grow(&layer.value),
                };
            }
            self.capacity = capacity;
        }
        keep
    }

    /// Logits for `ids`, which follow the cached ids; their keys and values
    /// join the cache.
    fn feed(
        &mut self,
        base: &Loaded,
        model: &Lfm2Model,
        ids: &[u32],
    ) -> ferrotensor_autograd::Tensor {
        let pos = self.ids.len();
        let logits = model.forward(&base.engine, ids, pos, Some(&mut self.kv), None);
        self.ids.extend_from_slice(ids);
        logits
    }
}

/// Prompt token ids and the token ids that end the reply to them.
struct Prompt {
    ids: Vec<u32>,
    stop: Vec<u32>,
}

impl Prompt {
    /// A prompt used as is: the reply ends at the tokenizer's EOS, an
    /// `eos_token_id` of the config or `<|endoftext|>`.
    fn raw(tokenizer: &Tokenizer, config: &Lfm2Config, ids: Vec<u32>) -> Self {
        let mut stop = vec![tokenizer.eos_token_id];
        stop.extend(&config.eos_token_ids);
        stop.extend(
            tokenizer
                .hf_tokenizer
                .as_ref()
                .and_then(|t| t.token_to_id("<|endoftext|>")),
        );
        Self { ids, stop }
    }

    /// A prompt in the ChatML template: the reply also ends at `<|im_end|>`,
    /// which closes the assistant turn even when the config names another
    /// EOS.
    fn chatml(tokenizer: &Tokenizer, config: &Lfm2Config, ids: Vec<u32>) -> Self {
        let mut prompt = Self::raw(tokenizer, config, ids);
        prompt.stop.extend(tokenizer.vocab.get("<|im_end|>"));
        prompt
    }
}

/// What [`generate_tokens`] did.
struct Decoded {
    new_tokens: usize,
    /// Prompt tokens taken from the cache rather than run.
    cached_tokens: usize,
    /// Time to run the prompt and read back its logits.
    prefill: Duration,
    /// One-token forward passes after the prompt.
    decode_steps: usize,
    /// Time from the end of the prefill to the last new token.
    decode: Duration,
}

/// Sample up to `max_new` tokens after `prompt`, streaming decoded text to
/// `emit`. With `cache`, only the prompt tokens it does not already hold are
/// run.
///
/// # Errors
/// [`Unmet`] when `prompt` is empty.
async fn generate_tokens(
    base: &Loaded,
    model: &Lfm2Model,
    prompt: &Prompt,
    max_new: usize,
    sampler: &mut Sampler,
    mut cache: Option<&mut DecodeCache>,
    mut emit: impl FnMut(&str),
) -> Result<Decoded> {
    if prompt.ids.is_empty() {
        return Err(unmet("the prompt encodes to no tokens"));
    }
    let vocab = base.config.vocab_size;
    let mut tokens = prompt.ids.clone();
    let mut generated: Vec<u32> = Vec::new();
    let mut printed = String::new();
    let start = Instant::now();
    let cached_tokens = cache.as_deref_mut().map_or(0, |c| {
        c.reuse(base, &prompt.ids, prompt.ids.len() + max_new)
    });
    let logits = no_grad(|| match cache.as_deref_mut() {
        Some(c) => c.feed(base, model, &prompt.ids[cached_tokens..]),
        None => model.forward(&base.engine, &tokens, 0, None, None),
    });
    let mut host = logits.data.to_vec(&base.ctx).await;
    let prefill = start.elapsed();
    let decode_start = Instant::now();
    let mut decode_steps = 0;
    while generated.len() < max_new {
        let rows = host.len() / vocab;
        let next = sampler.sample(&host[(rows - 1) * vocab..rows * vocab]);
        if prompt.stop.contains(&next) {
            break;
        }
        generated.push(next);
        tokens.push(next);
        let text = base.tokenizer.decode(&generated)?;
        if let Some(delta) = text.strip_prefix(printed.as_str()) {
            emit(delta);
            printed = text;
        }
        // Nothing would sample from the logits after the last token.
        if generated.len() == max_new {
            break;
        }
        let logits = no_grad(|| match cache.as_deref_mut() {
            Some(c) => c.feed(base, model, &[next]),
            None => model.forward(&base.engine, &tokens, 0, None, None),
        });
        host = logits.data.to_vec(&base.ctx).await;
        decode_steps += 1;
    }
    Ok(Decoded {
        new_tokens: generated.len(),
        cached_tokens,
        prefill,
        decode_steps,
        decode: decode_start.elapsed(),
    })
}

// ---------------------------------------------------------------------------
// train

/// Parameter name of the token embedding table.
const EMBED_TOKENS: &str = "model.embed_tokens.weight";

fn select_train_params(
    model: &Lfm2Model,
    modules: &str,
) -> Result<HashMap<String, ferrotensor_autograd::Tensor>> {
    let mode = modules.to_ascii_lowercase();
    let params: HashMap<_, _> = if mode == "lora" {
        model.named_lora_params()
    } else {
        model
            .named_params()
            .into_iter()
            .filter(|(n, _)| trains_base_weight(&mode, n))
            .collect()
    };
    if params.is_empty() {
        return Err(format!("--train-modules {modules} selects no parameters").into());
    }
    Ok(params)
}

/// Whether `--train-modules` value `mode` (lowercase, not `lora`) trains the
/// base weight `name`.
fn trains_base_weight(mode: &str, name: &str) -> bool {
    // Llama blocks name the MLP `mlp`, LFM2 blocks `feed_forward`.
    let keys: &[&str] = match mode {
        "all" => return !name.contains("embed_tokens"),
        "full" => return true,
        "embed" | "embedding" => &["embed_tokens"],
        "attn" | "attention" => &["self_attn"],
        "mlp" => &[".mlp.", ".feed_forward."],
        "head" => &["lm_head"],
        other => return name.contains(other),
    };
    keys.iter().any(|k| name.contains(k))
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

/// What `train --json` prints at the end of a run. A loss that was never
/// measured is NaN, which serializes as null.
#[derive(serde::Serialize, Debug)]
struct TrainSummary {
    steps: usize,
    first_loss: Option<f32>,
    final_loss: f32,
    eval_loss_before: Option<f32>,
    eval_loss_after: Option<f32>,
    /// Peak of the GPU buffers Ferrotensor tracks, without the driver context.
    measured_peak_mib: u64,
    seconds: f32,
    out_dir: PathBuf,
    /// `None` on ranks other than 0, which save nothing.
    adapter: Option<PathBuf>,
}

/// Runs `train` and returns its summary.
async fn train(mut args: TrainArgs) -> Result<TrainSummary> {
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
        return Err(unmet(
            "--batch-size, --grad-accum and --lora-rank must be positive",
        ));
    }
    if args.world_size == 0 || args.rank >= args.world_size {
        return Err(unmet(format!(
            "--rank {} must be below --world-size {}",
            args.rank, args.world_size
        )));
    }
    let distributed = args.world_size > 1;
    if distributed && args.resume.is_some() {
        return Err(unmet("--resume is not supported with --world-size > 1 yet"));
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
    // Base weights outside the selection are frozen, so backward neither
    // computes nor keeps a gradient the optimizer would never use.
    let model = if lora_on {
        model
    } else {
        model.with_trainable_base_weights(|name| named_params.contains_key(name))
    };
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

    let model = model.with_block_checkpointing(args.checkpoint_blocks);
    if args.checkpoint_blocks {
        eprintln!("block checkpointing: on");
    }
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

    let eval_loss_before = match &eval_ds {
        Some(ds) => {
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
            Some(loss)
        }
        None => None,
    };

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
    let mut profile = args
        .profile
        .then(|| profile_report::ProfileRun::start(&base.ctx));

    'outer: for epoch in 0.. {
        // A resume at or past --max-steps trains nothing and only saves.
        if step >= total_steps {
            break;
        }
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
                if let Some(p) = profile.as_mut() {
                    p.after_step();
                }
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
    // The tracker's counters are always on; --memory-report resets its peak
    // at every phase it records, so take the larger of the two.
    let peak_bytes = base.ctx.memory().peak_total_bytes().max(
        memory
            .as_ref()
            .map_or(0, memory_report::MemoryReportRun::overall_peak_total_bytes),
    );
    if let Some(p) = profile {
        let json = is_root.then(|| args.out.join(profile_report::FILE_NAME));
        p.finish(json.as_deref())?;
    }
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

    let eval_loss_after = match &eval_ds {
        Some(ds) => {
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
            Some(loss)
        }
        None => None,
    };

    if let Some(prompt) = &args.sample_prompt {
        let ids = base.tokenizer.encode_chat(None, prompt)?;
        let sample_prompt = Prompt::chatml(&base.tokenizer, &base.config, ids);
        let mut sampler = Sampler::new(0.0, 0, 1.0, None);
        eprintln!("sample ({prompt:?}):");
        let mut cache = DecodeCache::new(&trainer.model, sample_prompt.ids.len() + 96);
        generate_tokens(
            &base,
            &trainer.model,
            &sample_prompt,
            96,
            &mut sampler,
            cache.as_mut(),
            |t| stream_text(t, args.json),
        )
        .await?;
        stream_text("\n", args.json);
    }
    if is_root && lora_on && !args.json {
        let model = args.model.model.display();
        let adapter = adapter_path.display();
        eprintln!("next:");
        eprintln!("  ferrotensor chat --model {model} --lora {adapter}");
        eprintln!(
            "  ferrotensor merge --model {model} --lora {adapter} --out {}-merged",
            args.out.display()
        );
    }
    Ok(TrainSummary {
        steps: step,
        first_loss,
        final_loss: last_loss,
        eval_loss_before,
        eval_loss_after,
        measured_peak_mib: peak_bytes.div_ceil(MIB),
        seconds: elapsed.as_secs_f32(),
        out_dir: args.out,
        adapter: is_root.then_some(adapter_path),
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
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
    fn train_parses_checkpoint_blocks() {
        let checkpoint_blocks = |extra: &[&str]| {
            let mut args = vec!["train", "--model", "m", "--data", "d.jsonl"];
            args.extend_from_slice(extra);
            match parse(&args).expect("parse").command {
                Command::Train(t) => t.checkpoint_blocks,
                _ => panic!("expected train"),
            }
        };
        assert!(checkpoint_blocks(&["--checkpoint-blocks"]));
        assert!(!checkpoint_blocks(&[]));
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
        assert_eq!(g.sampling.top_k, 5);
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
    fn q4_weights_parse_on_every_command() {
        let model = |args: &[&str]| match parse(args).expect("parse").command {
            Command::Train(t) => t.model,
            Command::Eval(e) => e.model,
            Command::Generate(g) => g.model,
            _ => panic!("expected train, eval or generate"),
        };
        let train = ["train", "--model", "m", "--data", "d.jsonl"];
        let eval = ["eval", "--model", "m", "--data", "d.jsonl"];
        let generate = ["generate", "--model", "m"];
        for cmd in [&train[..], &eval[..], &generate[..]] {
            assert!(!model(cmd).q4_weights, "{cmd:?}: off by default");
            let mut with = cmd.to_vec();
            with.push("--q4-weights");
            let m = model(&with);
            assert!(m.q4_weights, "{with:?}");
            assert!(!m.f32_weights, "{with:?}: lm_head stays f16");
        }
        assert!(
            parse(&[
                "train",
                "--model",
                "m",
                "--data",
                "d",
                "--q4-weights",
                "--f32-weights"
            ])
            .is_err(),
            "--q4-weights keeps lm_head f16, so it conflicts with --f32-weights"
        );
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
    fn train_modules_select_the_embedding_only_when_asked() {
        let names = [
            EMBED_TOKENS,
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.input_layernorm.weight",
            "model.norm.weight",
            "lm_head.weight",
        ];
        let selected = |modules: &str| -> Vec<&str> {
            let cli = parse(&[
                "train",
                "--model",
                "m",
                "--data",
                "d",
                "--train-modules",
                modules,
            ])
            .unwrap();
            let Command::Train(args) = cli.command else {
                panic!("not a train command");
            };
            let mode = args.train_modules.to_ascii_lowercase();
            names
                .into_iter()
                .filter(|n| trains_base_weight(&mode, n))
                .collect()
        };
        assert_eq!(selected("all"), &names[1..]);
        assert_eq!(selected("embed"), [EMBED_TOKENS]);
        assert_eq!(selected("Embedding"), [EMBED_TOKENS]);
        assert_eq!(selected("full"), names);
        assert_eq!(selected("FULL"), names);
        assert_eq!(selected("attn"), [names[1]]);
        assert_eq!(selected("head"), ["lm_head.weight"]);
        assert_eq!(
            saved_metadata(&lora_meta(), "Full"),
            LoraMetadata::trained_weights("full")
        );
        let weights = LoraMetadata::trained_weights("embed");
        assert!(check_resume_file(Some(&weights), "embed").is_ok());
        assert!(check_resume_file(Some(&weights), "full").is_err());
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

    #[test]
    fn exit_codes_are_0_1_2_and_unmet_errors_exit_with_2() {
        assert_eq!(
            [
                Exit::Success as i32,
                Exit::RuntimeError as i32,
                Exit::Unmet as i32
            ],
            [0, 1, 2]
        );
        let invalid = unmet("--seq-len must be at least 2");
        assert_eq!(Exit::for_error(invalid.as_ref()), Exit::Unmet);
        assert_eq!(invalid.to_string(), "--seq-len must be at least 2");
        let runtime: Box<dyn std::error::Error> = "the GPU device was lost".into();
        assert_eq!(Exit::for_error(runtime.as_ref()), Exit::RuntimeError);
        let no_adapter: Box<dyn std::error::Error> = Box::new(ContextError::NoAdapter {
            backends: ferrotensor_backend::wgpu::Backends::VULKAN,
            force_fallback_adapter: false,
        });
        assert_eq!(Exit::for_error(no_adapter.as_ref()), Exit::Unmet);
    }

    #[test]
    fn train_summary_json_has_every_field_and_nan_as_null() {
        let summary = TrainSummary {
            steps: 2,
            first_loss: Some(4.5),
            final_loss: f32::NAN,
            eval_loss_before: None,
            eval_loss_after: Some(3.25),
            measured_peak_mib: 1400,
            seconds: 1.5,
            out_dir: PathBuf::from("runs/a"),
            adapter: Some(PathBuf::from("runs/a/lora_adapters.safetensors")),
        };
        assert_eq!(
            serde_json::to_value(&summary).unwrap(),
            serde_json::json!({
                "steps": 2,
                "first_loss": 4.5,
                "final_loss": null,
                "eval_loss_before": null,
                "eval_loss_after": 3.25,
                "measured_peak_mib": 1400,
                "seconds": 1.5,
                "out_dir": "runs/a",
                "adapter": "runs/a/lora_adapters.safetensors",
            })
        );
    }

    #[test]
    fn eval_and_generate_summaries_serialize_their_fields() {
        let eval = EvalSummary {
            examples: 3,
            tokens: 40,
            loss: 2.0,
            perplexity: 7.5,
            seconds: 0.25,
        };
        assert_eq!(
            serde_json::to_value(&eval).unwrap(),
            serde_json::json!({
                "examples": 3, "tokens": 40, "loss": 2.0, "perplexity": 7.5, "seconds": 0.25,
            })
        );
        let generate = GenerateSummary {
            text: "w5 w6".to_string(),
            prompt_tokens: 3,
            new_tokens: 2,
            tokens_per_second: 8.0,
            prefill_seconds: 0.5,
            kv_cache: true,
        };
        assert_eq!(
            serde_json::to_value(&generate).unwrap(),
            serde_json::json!({
                "text": "w5 w6", "prompt_tokens": 3, "new_tokens": 2,
                "tokens_per_second": 8.0, "prefill_seconds": 0.5, "kv_cache": true,
            })
        );
    }

    #[test]
    fn a_chatml_prompt_also_stops_at_im_end() {
        let dir = std::env::temp_dir().join(format!("ferrotensor-stop-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tokenizer.json");
        let vocab = serde_json::json!({
            "<unk>": 0, "</s>": 1, "<|im_start|>": 2, "<|im_end|>": 3, "<|endoftext|>": 4,
        });
        let tokenizer_json = serde_json::json!({
            "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
            "normalizer": null, "pre_tokenizer": { "type": "Whitespace" },
            "post_processor": null, "decoder": null,
            "model": { "type": "WordLevel", "vocab": vocab, "unk_token": "<unk>" },
        });
        std::fs::write(&path, tokenizer_json.to_string()).unwrap();
        let mut tokenizer = Tokenizer::from_file(&path).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        // A base model's tokenizer can name `</s>` as its EOS.
        tokenizer.eos_token_id = 1;
        let config: Lfm2Config = serde_json::from_value(serde_json::json!({
            "hidden_size": 8, "intermediate_size": 16, "num_attention_heads": 2,
            "num_hidden_layers": 1, "vocab_size": 5, "eos_token_id": [1, 4],
        }))
        .unwrap();
        assert_eq!(Prompt::raw(&tokenizer, &config, vec![2]).stop, [1, 1, 4, 4]);
        assert_eq!(
            Prompt::chatml(&tokenizer, &config, vec![2]).stop,
            [1, 1, 4, 4, 3]
        );
    }

    fn decoded(new_tokens: usize, decode_steps: usize, decode_ms: u64) -> Decoded {
        Decoded {
            new_tokens,
            cached_tokens: 0,
            prefill: Duration::from_millis(250),
            decode_steps,
            decode: Duration::from_millis(decode_ms),
        }
    }

    #[test]
    fn decode_speed_leaves_out_the_prefill() {
        // 4 tokens with max_new 4: the first comes from the prefill logits and
        // 3 one-token passes give the rest.
        let summary = GenerateSummary::new(String::new(), 7, &decoded(4, 3, 1500), true);
        assert_eq!(summary.tokens_per_second, 2.0);
        assert_eq!(summary.prefill_seconds, 0.25);
        assert_eq!(
            summary.stats_line(),
            "[7 prompt + 4 new tokens, prefill 0.25s, 2.0 tok/s, kv cache on]"
        );
    }

    #[test]
    fn decode_speed_is_null_without_a_decode_step() {
        let summary = GenerateSummary::new(String::new(), 7, &decoded(1, 0, 3), false);
        assert!(summary.tokens_per_second.is_nan());
        let json = serde_json::to_value(&summary).unwrap();
        assert_eq!(json["tokens_per_second"], serde_json::Value::Null);
        assert_eq!(json["new_tokens"], 1);
        assert_eq!(
            summary.stats_line(),
            "[7 prompt + 1 new tokens, prefill 0.25s, no decode step, kv cache off]"
        );
    }

    #[test]
    fn json_flag_parses_on_every_command_that_takes_it() {
        for command in ["train", "finetune", "eval"] {
            let cli = parse(&[command, "--model", "m", "--data", "d.jsonl", "--json"]).unwrap();
            let json = match cli.command {
                Command::Train(args) => args.json,
                Command::Finetune(args) => args.train.json,
                Command::Eval(args) => args.json,
                _ => false,
            };
            assert!(json, "{command}");
        }
        let cli = parse(&["generate", "--model", "m", "--json"]).unwrap();
        assert!(matches!(cli.command, Command::Generate(args) if args.json));
    }

    #[test]
    fn new_tokens_are_clipped_to_the_context() {
        assert_eq!(new_tokens_within(2048, 100, 128), Some(128));
        assert_eq!(new_tokens_within(2048, 1920, 128), Some(128));
        assert_eq!(new_tokens_within(2048, 1921, 128), Some(127));
        assert_eq!(new_tokens_within(10, 3, 20), Some(7));
        assert_eq!(new_tokens_within(10, 10, 20), Some(0));
    }

    #[test]
    fn a_prompt_longer_than_the_context_has_no_room() {
        assert_eq!(new_tokens_within(10, 11, 20), None);
        assert_eq!(new_tokens_within(0, 1, 0), None);
    }

    const INPUT: [u32; 7] = [5, 9, 5, 12, 9, 5, 20];
    const TARGET: [u32; 7] = [9, 5, 12, 9, 5, 20, 3];

    /// A random one-block byte-level model in `dir`, loaded as `train` loads
    /// a base whose weights train.
    fn tiny_base(dir: &Path) -> Loaded {
        let size = ferrotensor_trainer::pretrain::LlamaSize {
            layers: 1,
            hidden: 16,
            heads: 2,
            mlp_hidden: 32,
            // Room for the trainer's padding to 32 tokens.
            context: 64,
            vocab: ferrotensor_loader::tokenizer::BYTE_LEVEL_VOCAB_SIZE,
        };
        fs::create_dir_all(dir).unwrap();
        ferrotensor_trainer::pretrain::write_random_llama(dir, &size, 0).unwrap();
        let tokenizer = ferrotensor_loader::tokenizer::byte_level_tokenizer_json();
        fs::write(dir.join("tokenizer.json"), tokenizer.to_string()).unwrap();
        pollster::block_on(load_base(&ModelArgs {
            model: dir.to_path_buf(),
            device: Device::Auto,
            f32_weights: true,
            f16_weights: false,
            q4_weights: false,
        }))
        .unwrap()
    }

    /// The model `train --train-modules mode` builds, and the parameters it
    /// hands the optimizer.
    fn partial_model(
        base: &Loaded,
        dir: &Path,
        mode: &str,
    ) -> (Lfm2Model, HashMap<String, ferrotensor_autograd::Tensor>) {
        let mut weights = WeightsContext::new();
        weights.load_model_dir(dir).unwrap();
        base.weights.set(Some(weights));
        let model = build_model(base, None, true).unwrap();
        let selected = select_train_params(&model, mode).unwrap();
        let model = model.with_trainable_base_weights(|name| selected.contains_key(name));
        (model, selected)
    }

    fn backward(base: &Loaded, model: &Lfm2Model) {
        let logits = model.forward(&base.engine, &INPUT, 0, None, None);
        ferrotensor_trainer::masked_cross_entropy(&logits, &TARGET, &[1; 7], &base.engine)
            .unwrap()
            .tensor
            .backward(&base.engine);
    }

    #[test]
    #[ignore = "needs a GPU"]
    fn only_the_selected_base_weights_get_a_gradient() {
        let dir = std::env::temp_dir().join(format!("ferrotensor-frozen-{}", std::process::id()));
        let base = tiny_base(&dir);
        let names = [
            EMBED_TOKENS,
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.norm.weight",
            "lm_head.weight",
        ];
        for (mode, expected) in [
            ("attn", &names[2..6]),
            ("mlp", &names[7..10]),
            ("head", &names[11..]),
            ("embed", &names[..1]),
            ("all", &names[1..]),
            ("full", &names[..]),
        ] {
            let (model, selected) = partial_model(&base, &dir, mode);
            backward(&base, &model);
            let mut with_grad: Vec<String> = model
                .named_params()
                .into_iter()
                .filter(|(_, tensor)| tensor.grad.borrow().is_some())
                .map(|(name, _)| name)
                .collect();
            with_grad.sort();
            let mut expected = expected.to_vec();
            expected.sort_unstable();
            assert_eq!(with_grad, expected, "--train-modules {mode}");
            let mut optimized: Vec<String> = selected.into_keys().collect();
            optimized.sort();
            assert_eq!(optimized, expected, "--train-modules {mode}");
        }
        fs::remove_dir_all(&dir).unwrap();
    }

    /// One step with `--train-modules attn`: the attention weights move,
    /// every other weight keeps its bits, and the clipped norm is the norm of
    /// the attention gradients alone, summed by hand.
    #[test]
    #[ignore = "needs a GPU"]
    fn a_partial_step_moves_and_measures_only_the_selected_weights() {
        let dir = std::env::temp_dir().join(format!("ferrotensor-step-{}", std::process::id()));
        let base = tiny_base(&dir);
        let host =
            |data: &ferrotensor_backend::UnifiedTensor| pollster::block_on(data.to_vec(&base.ctx));
        let (model, selected) = partial_model(&base, &dir, "attn");
        let params: Vec<_> = selected.values().cloned().collect();
        backward(&base, &model);
        let by_hand = params
            .iter()
            .flat_map(|p| host(p.grad.borrow().as_ref().unwrap()))
            .map(|g| f64::from(g).powi(2))
            .sum::<f64>()
            .sqrt();
        let before: HashMap<String, Vec<f32>> = model
            .named_params()
            .into_iter()
            .map(|(name, tensor)| {
                tensor.zero_grad();
                (name, host(tensor.data.as_ref()))
            })
            .collect();

        let mut trainer = Trainer::new(
            model,
            Box::new(AdamW::new(base.engine.clone(), 1e-2)),
            base.ctx.clone(),
            base.engine.clone(),
        )
        .with_clip_grad_norm(1.0);
        let stats = trainer.train_step(&INPUT, &TARGET, &params).unwrap();
        let norm = f64::from(stats.grad_norm.unwrap());
        assert!(
            (norm - by_hand).abs() <= 1e-4 * by_hand,
            "step norm {norm}, by hand {by_hand}"
        );
        for (name, tensor) in trainer.model.named_params() {
            let after = host(tensor.data.as_ref());
            if selected.contains_key(&name) {
                assert_ne!(after, before[&name], "{name} did not move");
            } else {
                assert_eq!(after, before[&name], "{name} moved");
            }
        }
        fs::remove_dir_all(&dir).unwrap();
    }
}
