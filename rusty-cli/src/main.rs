//! Rusty CLI - Lite-first SFT + Pretrain
//!
//! Primary commands:
//! - `rusty-cli train --task sft|pretrain --model ... --data ...`
//! - `rusty-cli eval  --task sft|pretrain --model ... --data ...`
//! - `rusty-cli infer --model ...`
//!
//! Legacy positional flags remain supported for compatibility.

use rusty_autograd::AdamW;
use rusty_backend::{ComputeEngine, WgpuContext};
use rusty_graph::model_builder::{Lfm2ModelBuilder, LoRaConfig};
use rusty_loader::{ModelLoader, Sampler, Tokenizer, WeightsContext};
use rusty_trainer::checkpoint::{
    apply_checkpoint, load_lora_checkpoint, save_lora_checkpoint, TrainingCheckpoint, TrainingState,
};
use rusty_trainer::{
    masked_cross_entropy_from_logits, DataLoader, Dataset, FileGradSyncConfig, GradScaler,
    TextDataset, Trainer, TrainingExample,
};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

#[derive(serde::Deserialize)]
struct Sample {
    prompt: String,
    response: String,
    #[serde(default)]
    system: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DevicePreference {
    Auto,
    Gpu,
    Cpu,
}

impl DevicePreference {
    fn parse(value: &str) -> Self {
        match value.to_ascii_lowercase().as_str() {
            "gpu" => Self::Gpu,
            "cpu" => Self::Cpu,
            _ => Self::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TaskKind {
    Sft,
    Pretrain,
}

impl TaskKind {
    fn parse(value: &str) -> Self {
        match value.to_ascii_lowercase().as_str() {
            "pretrain" | "lm" | "next-token" => Self::Pretrain,
            _ => Self::Sft,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Sft => "sft",
            Self::Pretrain => "pretrain",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CommandMode {
    Train,
    Eval,
    Infer,
}

#[derive(Clone, Debug)]
struct RunConfig {
    mode: CommandMode,
    task: TaskKind,
    model_path: String,
    dataset_path: Option<String>,
    save_lora: Option<String>,
    out_dir: Option<String>,
    resume_path: Option<String>,
    train_modules: String,
    grad_clip: Option<f32>,
    grad_accum: usize,
    fp16: bool,
    sequence_length: usize,
    batch_size: usize,
    epochs: usize,
    max_steps: Option<usize>,
    max_samples: Option<usize>,
    no_infer: bool,
    lite_mode: bool,
    auto_mode: bool,
    allow_fallback: bool,
    use_chat_template: bool,
    device: DevicePreference,
    max_vram_gb: Option<f32>,
    prompt: Option<String>,

    // Tier-2 scaling: multi-process data-parallel gradient averaging.
    world_size: usize,
    rank: usize,
    sync_dir: Option<String>,
}

fn main() {
    env_logger::init();
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_usage();
        return;
    }

    match args[1].as_str() {
        "--help" | "-h" | "help" => print_usage(),
        "--demo" | "demo" => run_demo(),
        "--benchmark" | "benchmark" => run_benchmark(),
        "--monitor" | "monitor" => run_monitor(),
        "train" | "eval" | "infer" => {
            let config = parse_subcommand(&args[1], &args[2..]);
            pollster::block_on(run_pipeline(&config));
        }
        "--eval" => {
            if args.len() < 4 {
                print_usage();
                return;
            }
            let mut config = default_config(CommandMode::Eval);
            config.model_path = args[2].clone();
            config.dataset_path = Some(args[3].clone());
            pollster::block_on(run_pipeline(&config));
        }
        _ => {
            let config = parse_legacy_args(&args[1..]);
            pollster::block_on(run_pipeline(&config));
        }
    }
}

fn default_config(mode: CommandMode) -> RunConfig {
    RunConfig {
        mode,
        task: TaskKind::Sft,
        model_path: String::new(),
        dataset_path: None,
        save_lora: None,
        out_dir: None,
        resume_path: None,
        train_modules: "lora".to_string(),
        grad_clip: None,
        grad_accum: 1,
        fp16: false,
        sequence_length: 128,
        batch_size: 1,
        epochs: 10,
        max_steps: None,
        max_samples: None,
        no_infer: mode != CommandMode::Infer,
        lite_mode: false,
        auto_mode: false,
        allow_fallback: false,
        use_chat_template: true,
        device: DevicePreference::Auto,
        max_vram_gb: None,
        prompt: None,
        world_size: 1,
        rank: 0,
        sync_dir: None,
    }
}

fn print_usage() {
    println!("======================================================================");
    println!("                    RUSTY WORLD TRAINER                               ");
    println!("         Lite-first SFT + Pretrain for consumer hardware              ");
    println!("======================================================================");
    println!();
    println!("USAGE:");
    println!(
        "  rusty-cli train --model <PATH> --data <DATA> [--task sft|pretrain] [--lite] [--out DIR]"
    );
    println!("  rusty-cli eval  --model <PATH> --data <DATA> [--task sft|pretrain]");
    println!("  rusty-cli infer --model <PATH> [--prompt TEXT] [--resume LORA]");
    println!("  rusty-cli --demo | --monitor | --benchmark | --help");
    println!();
    println!("COMMON FLAGS:");
    println!("  --task sft|pretrain     Training objective (default: sft)");
    println!("  --lite / --auto         Safe 8GB consumer defaults");
    println!("  --seq-len N --batch-size N --grad-accum N --epochs N --max-steps N");
    println!("  --train-modules lora|all|attn|mlp|norm|embed|lm_head");
    println!("  --fp16 --grad-clip N --device auto|gpu|cpu --max-vram-gb N");
    println!("  --no-infer --allow-fallback --no-chat-template");
    println!("  --world-size N --rank R --sync-dir DIR  (file grad-sync; Tier-2)");
    println!("  --save-lora PATH --resume CHECKPOINT --out DIR");
    println!();
    println!("8GB MACBOOK RECIPES:");
    println!("  # SFT (instruction tuning, response-only loss + chat template)");
    println!("  rusty-cli train --lite --task sft --model ./models/tinyllama --data ./data/train.jsonl --out ./outputs/run1");
    println!();
    println!("  # Pretrain / continued LM on packed tokens");
    println!("  rusty-cli train --lite --task pretrain --model ./models/tinyllama --data ./data/train.bin --out ./outputs/lm1");
    println!();
    println!("  # Eval + resume");
    println!("  rusty-cli eval --task sft --model ./models/tinyllama --data ./data/val.jsonl");
    println!("  rusty-cli train --lite --task sft --model ./models/tinyllama --data ./data/train.jsonl --resume ./outputs/run1/lora_adapters.safetensors");
    println!();
    println!("DATASET FORMATS:");
    println!("  sft:      .json / .jsonl with {{prompt,response}} (optional system)");
    println!("  pretrain: .txt (lines) or .bin (packed little-endian u32 tokens)");
}

fn run_demo() {
    rusty_trainer::run_demo(rusty_trainer::DemoConfig::default());
}

fn run_benchmark() {
    println!("Run benchmarks with: cargo run -p rusty-benchmarks --release");
}

fn run_monitor() {
    if let Err(e) = rusty_monitor::run() {
        eprintln!("TUI error: {}", e);
    }
}

#[derive(Default, Clone, Copy)]
struct FlagOverrides {
    train_modules: bool,
    grad_accum: bool,
    seq_len: bool,
    batch_size: bool,
    epochs: bool,
}

fn parse_subcommand(cmd: &str, args: &[String]) -> RunConfig {
    let mode = match cmd {
        "train" => CommandMode::Train,
        "eval" => CommandMode::Eval,
        _ => CommandMode::Infer,
    };
    let mut config = default_config(mode);
    let overrides = apply_flags(&mut config, args);
    if config.lite_mode || config.auto_mode {
        apply_lite_defaults(&mut config, overrides);
    }
    if mode == CommandMode::Train {
        config.mode = CommandMode::Train;
    }
    config
}

fn parse_legacy_args(args: &[String]) -> RunConfig {
    let mut config = default_config(CommandMode::Eval);
    let overrides = apply_flags(&mut config, args);
    if config.lite_mode || args.iter().any(|a| a == "--train") {
        config.mode = CommandMode::Train;
    }
    if config.lite_mode || config.auto_mode {
        apply_lite_defaults(&mut config, overrides);
    }
    config
}

fn apply_lite_defaults(config: &mut RunConfig, overrides: FlagOverrides) {
    // Apply consumer-hardware defaults without forcing a command mode.
    // (`train --lite` already sets Train; `eval --lite` should stay Eval.)
    if !overrides.train_modules {
        config.train_modules = "lora".to_string();
    }
    if !overrides.grad_accum {
        config.grad_accum = 8;
    }
    if !overrides.seq_len {
        config.sequence_length = 128;
    }
    if !overrides.batch_size {
        config.batch_size = 1;
    }
    if !overrides.epochs {
        config.epochs = 3;
    }
    config.fp16 = true;
    config.no_infer = true;
    if config.max_vram_gb.is_none() {
        config.max_vram_gb = Some(8.0);
    }
}

fn apply_flags(config: &mut RunConfig, args: &[String]) -> FlagOverrides {
    let mut overrides = FlagOverrides::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--train" => {
                config.mode = CommandMode::Train;
                i += 1;
            }
            "--lite" => {
                config.lite_mode = true;
                i += 1;
            }
            "--auto" => {
                config.auto_mode = true;
                i += 1;
            }
            "--fp16" => {
                config.fp16 = true;
                i += 1;
            }
            "--no-infer" => {
                config.no_infer = true;
                i += 1;
            }
            "--allow-fallback" => {
                config.allow_fallback = true;
                i += 1;
            }
            "--no-chat-template" => {
                config.use_chat_template = false;
                i += 1;
            }
            "--task" => {
                if let Some(v) = args.get(i + 1) {
                    config.task = TaskKind::parse(v);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--model" => {
                if let Some(v) = args.get(i + 1) {
                    config.model_path = v.clone();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--data" | "--dataset" => {
                if let Some(v) = args.get(i + 1) {
                    config.dataset_path = Some(v.clone());
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--out" | "--output-dir" => {
                if let Some(v) = args.get(i + 1) {
                    config.out_dir = Some(v.clone());
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--save-lora" => {
                if let Some(v) = args.get(i + 1) {
                    config.save_lora = Some(v.clone());
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--resume" => {
                if let Some(v) = args.get(i + 1) {
                    config.resume_path = Some(v.clone());
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--train-modules" => {
                if let Some(v) = args.get(i + 1) {
                    config.train_modules = v.clone();
                    overrides.train_modules = true;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--grad-clip" => {
                if let Some(v) = args.get(i + 1) {
                    config.grad_clip = v.parse().ok();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--grad-accum" => {
                if let Some(v) = args.get(i + 1) {
                    config.grad_accum = v.parse().unwrap_or(1);
                    overrides.grad_accum = true;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--seq-len" => {
                if let Some(v) = args.get(i + 1) {
                    config.sequence_length = v.parse().unwrap_or(128);
                    overrides.seq_len = true;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--batch-size" => {
                if let Some(v) = args.get(i + 1) {
                    config.batch_size = v.parse().unwrap_or(1);
                    overrides.batch_size = true;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--epochs" => {
                if let Some(v) = args.get(i + 1) {
                    config.epochs = v.parse().unwrap_or(10);
                    overrides.epochs = true;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--max-steps" => {
                if let Some(v) = args.get(i + 1) {
                    config.max_steps = v.parse().ok();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--max-samples" => {
                if let Some(v) = args.get(i + 1) {
                    config.max_samples = v.parse().ok();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--device" => {
                if let Some(v) = args.get(i + 1) {
                    config.device = DevicePreference::parse(v);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--max-vram-gb" => {
                if let Some(v) = args.get(i + 1) {
                    config.max_vram_gb = v.parse().ok();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--world-size" => {
                if let Some(v) = args.get(i + 1) {
                    config.world_size = v.parse().unwrap_or(1).max(1);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--rank" => {
                if let Some(v) = args.get(i + 1) {
                    config.rank = v.parse().unwrap_or(0).max(0);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--sync-dir" => {
                if let Some(v) = args.get(i + 1) {
                    config.sync_dir = Some(v.clone());
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--prompt" => {
                if let Some(v) = args.get(i + 1) {
                    config.prompt = Some(v.clone());
                    i += 2;
                } else {
                    i += 1;
                }
            }
            flag if flag.starts_with('-') => {
                i += 1;
            }
            _ => {
                if config.model_path.is_empty() {
                    config.model_path = args[i].clone();
                } else if config.dataset_path.is_none() {
                    config.dataset_path = Some(args[i].clone());
                }
                i += 1;
            }
        }
    }

    overrides
}

async fn run_pipeline(config: &RunConfig) {
    if config.model_path.is_empty() || !Path::new(&config.model_path).exists() {
        println!("[ERROR] Model not found. Pass --model <PATH>.");
        print_usage();
        return;
    }

    if matches!(config.mode, CommandMode::Train | CommandMode::Eval)
        && config.dataset_path.is_none()
        && !config.allow_fallback
    {
        println!(
            "[ERROR] Dataset required for train/eval. Pass --data <PATH> or --allow-fallback."
        );
        return;
    }

    let model_path = config.model_path.as_str();
    println!("======================================================================");
    println!(
        "   RUSTY WORLD TRAINER  [{} / {}]",
        match config.mode {
            CommandMode::Train => "TRAIN",
            CommandMode::Eval => "EVAL",
            CommandMode::Infer => "INFER",
        },
        config.task.as_str()
    );
    println!("======================================================================");

    println!("[INIT] Initializing compute backend...");
    let (power_preference, force_fallback_adapter) = match config.device {
        DevicePreference::Auto | DevicePreference::Gpu => {
            (rusty_backend::wgpu::PowerPreference::HighPerformance, false)
        }
        DevicePreference::Cpu => (rusty_backend::wgpu::PowerPreference::LowPower, true),
    };
    let ctx =
        Arc::new(WgpuContext::new_with_options(power_preference, force_fallback_adapter).await);
    let engine = Arc::new(ComputeEngine::new(&ctx));
    println!("       Ready: {}", ctx.adapter_info());

    println!("[LOAD] Loading tokenizer...");
    let tokenizer_path = format!("{}/tokenizer.json", model_path);
    let tokenizer = match Tokenizer::from_file(&tokenizer_path) {
        Ok(t) => {
            println!("       Loaded {} tokens", t.vocab.len());
            t
        }
        Err(e) => {
            println!("       [WARN] Tokenizer load failed: {e}; using byte fallback");
            create_byte_tokenizer()
        }
    };

    let dataset = if matches!(config.mode, CommandMode::Train | CommandMode::Eval) {
        match load_task_dataset(config, &tokenizer) {
            Ok(ds) => {
                println!(
                    "[DATA] Loaded {} samples ({})",
                    ds.len(),
                    config.task.as_str()
                );
                ds
            }
            Err(e) => {
                if config.allow_fallback {
                    println!("[WARN] Dataset load failed ({e}); using fallback samples");
                    create_default_dataset(&tokenizer, config.sequence_length, config.task)
                } else {
                    println!("[ERROR] Failed to load dataset: {e}");
                    return;
                }
            }
        }
    } else {
        TextDataset::new()
    };

    println!("[LOAD] Loading model configuration...");
    let model_config = match ModelLoader::load_config(format!("{}/config.json", model_path)) {
        Ok(c) => {
            println!(
                "       hidden={} layers={} vocab={}",
                c.hidden_size, c.num_hidden_layers, c.vocab_size
            );
            c
        }
        Err(e) => {
            println!("[ERROR] Failed to load config: {e}");
            return;
        }
    };

    print_capacity_guidance(
        model_path,
        model_config.hidden_size,
        model_config.num_hidden_layers,
        config.sequence_length,
        config.batch_size.max(1),
        config.max_vram_gb,
    );

    println!("[LOAD] Loading model weights...");
    let mut weights_ctx = WeightsContext::new();
    if let Err(e) = weights_ctx.load_model_dir(model_path) {
        println!("[ERROR] Failed to load weights: {e}");
        return;
    }

    let out_dir = config
        .out_dir
        .clone()
        .unwrap_or_else(|| "outputs".to_string());
    let _ = fs::create_dir_all(&out_dir);
    let save_path = config
        .save_lora
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(&out_dir).join("lora_adapters.safetensors"));
    let checkpoint_path = checkpoint_state_path(&save_path);

    let lora_rank = if config.lite_mode || config.auto_mode {
        4
    } else {
        8
    };
    let lora_config = LoRaConfig {
        rank: lora_rank,
        alpha: (lora_rank * 2) as f32,
        target_modules: vec![
            "q_proj".to_string(),
            "v_proj".to_string(),
            "o_proj".to_string(),
        ],
    };

    println!("[BUILD] Building model with LoRA rank={lora_rank}...");
    let model = Lfm2ModelBuilder::new(
        ctx.clone(),
        engine.clone(),
        &weights_ctx,
        model_config.clone(),
    )
    .with_lora(lora_config)
    .build();
    let named_lora_params = model.named_lora_params();
    println!("       LoRA tensors: {}", named_lora_params.len());

    if let Some(resume_path) = config.resume_path.as_deref() {
        println!("[LOAD] Restoring LoRA tensors from {resume_path}...");
        match load_lora_checkpoint(resume_path) {
            Ok(data) => {
                apply_checkpoint(&ctx, &data, &named_lora_params);
                println!("       Restored {} tensors", data.len());
            }
            Err(e) => {
                println!("[ERROR] Failed to load LoRA checkpoint: {e}");
                return;
            }
        }
    }

    if config.mode == CommandMode::Infer {
        run_infer(
            &ctx,
            &engine,
            &model,
            &tokenizer,
            model_config.vocab_size,
            config.prompt.as_deref().unwrap_or("Hello!"),
        )
        .await;
        return;
    }

    let train_mode = config.mode == CommandMode::Train;
    let learning_rate = 1e-4f32;
    let epochs = config.epochs.max(1);
    let batch_size = config.batch_size.max(1);
    let grad_accum = config.grad_accum.max(1);
    let is_distributed = train_mode && config.world_size > 1;
    let is_root = !is_distributed || config.rank == 0;
    // In distributed mode, shard data by rank and avoid early stopping on
    // max_samples so step counts stay aligned across ranks.
    let max_samples_effective = if is_distributed {
        None
    } else {
        config.max_samples
    };

    println!("[CONFIG] task={} lr={learning_rate} epochs={epochs} batch={batch_size} accum={grad_accum} seq={}",
        config.task.as_str(), config.sequence_length);

    let mut resumed_checkpoint = None;
    if train_mode {
        if checkpoint_path.exists() {
            if let Ok(state) = TrainingCheckpoint::load(&checkpoint_path) {
                resumed_checkpoint = Some(state);
            }
        } else if Path::new("outputs/training_state.json").exists() {
            if let Ok(state) = TrainingState::load("outputs/training_state.json") {
                resumed_checkpoint = Some(TrainingCheckpoint {
                    training_state: state,
                    optimizer_state: None,
                    grad_scaler_state: None,
                });
            }
        }
    }

    let mut eval_model = Some(model);
    let mut trainer = if train_mode {
        let optimizer = Box::new(AdamW::new(engine.clone(), learning_rate));
        let trainer = Trainer::new(
            eval_model.take().expect("model moved"),
            optimizer,
            ctx.clone(),
            engine.clone(),
        )
        .with_grad_accum_steps(grad_accum);
        let trainer = if let Some(max_norm) = config.grad_clip {
            trainer.with_clip_grad_norm(max_norm)
        } else {
            trainer
        };
        Some(if config.fp16 {
            trainer.with_grad_scaler(GradScaler::new())
        } else {
            trainer
        })
    } else {
        None
    };

    // Enable Tier-2 multi-process grad averaging if requested.
    if train_mode && config.world_size > 1 {
        if config.resume_path.is_some() {
            println!("[ERROR] Distributed mode with --resume is not supported yet.");
            return;
        }
        let sync_dir = config
            .sync_dir
            .as_ref()
            .map(PathBuf::from)
            .unwrap_or_else(|| Path::new(&out_dir).join("grad_sync").to_path_buf());

        if config.rank >= config.world_size {
            println!(
                "[ERROR] Invalid distributed config: rank={} >= world_size={}",
                config.rank, config.world_size
            );
            return;
        }

        let cfg = match FileGradSyncConfig::new(config.world_size, config.rank, sync_dir) {
            Ok(cfg) => cfg.with_max_wait(Duration::from_secs(600)),
            Err(e) => {
                println!("[ERROR] Invalid distributed config: {e}");
                return;
            }
        };

        trainer = trainer.map(|t| t.with_file_grad_sync(cfg));
        println!(
            "[DIST] grad-sync file mode: world_size={} rank={} sync_dir={}",
            config.world_size,
            config.rank,
            config.sync_dir.as_deref().unwrap_or("outputs/grad_sync")
        );
    }

    let named_train_params = trainer
        .as_ref()
        .map(|t| select_named_train_params(&t.model, &config.train_modules));
    let train_params = named_train_params
        .as_ref()
        .map(|params| params.values().cloned().collect::<Vec<_>>());

    if let Some(checkpoint) = resumed_checkpoint.as_ref() {
        if let Err(e) = checkpoint.training_state.validate_resume(
            &config.train_modules,
            config.task.as_str(),
            config.sequence_length,
            model_path,
        ) {
            println!("[ERROR] Incompatible checkpoint: {e}");
            return;
        }
        if let (Some(trainer), Some(named), Some(opt)) = (
            trainer.as_mut(),
            named_train_params.as_ref(),
            checkpoint.optimizer_state.as_ref(),
        ) {
            let _ = trainer.optimizer.load_state(&ctx, named, opt);
        }
        if let (Some(trainer), Some(scaler_state)) =
            (trainer.as_mut(), checkpoint.grad_scaler_state.as_ref())
        {
            if let Some(scaler) = trainer.scaler.as_mut() {
                scaler.load_checkpoint(scaler_state);
            }
        }
    }

    let start_epoch = resumed_checkpoint
        .as_ref()
        .map(|c| c.training_state.epoch)
        .unwrap_or(0);
    let mut best_loss = resumed_checkpoint
        .as_ref()
        .and_then(|c| c.training_state.best_loss);

    let mut train_indices: Option<Vec<usize>> = None;
    if is_distributed {
        let mut idxs = rusty_trainer::distributed::shard_indices(
            dataset.len(),
            config.rank,
            config.world_size,
        );

        // Ensure equal number of *full* batches per rank.
        let usable_per_rank = (dataset.len() / config.world_size) / batch_size * batch_size;
        idxs.truncate(usable_per_rank);
        if idxs.is_empty() {
            println!(
                "[ERROR] Distributed sharding produced no samples (world_size={} rank={} dataset_len={}).",
                config.world_size,
                config.rank,
                dataset.len()
            );
            return;
        }
        train_indices = Some(idxs);
    }

    let shard_len = train_indices
        .as_ref()
        .map(|v| v.len())
        .unwrap_or_else(|| dataset.len());

    let effective_dataset_len = max_samples_effective
        .map(|limit| limit.min(shard_len))
        .unwrap_or(shard_len)
        .max(1);
    let steps_per_epoch = effective_dataset_len.div_ceil(batch_size);
    let total_steps = config
        .max_steps
        .unwrap_or(epochs.saturating_mul(steps_per_epoch))
        .max(1);

    println!("   Epoch |   Avg Loss  |   Progress");
    println!("   ------+-------------+------------");

    let mut train_loader = if train_mode {
        let mut loader = if let Some(indices) = train_indices.take() {
            DataLoader::new_with_indices(
                dataset.clone(),
                batch_size,
                false, // deterministic in distributed mode
                tokenizer.pad_token_id,
                indices,
            )
        } else {
            DataLoader::new_with_pad_token(
                dataset.clone(),
                batch_size,
                true,
                tokenizer.pad_token_id,
            )
        };
        loader.reset();
        Some(loader)
    } else {
        None
    };

    for epoch in start_epoch..epochs {
        let mut total_loss = 0.0f32;
        let mut total_tokens = 0usize;
        let mut total_samples = 0usize;
        let mut epoch_steps = 0usize;

        if let Some(loader) = train_loader.as_mut() {
            loader.reset();
            for batch in loader {
                if let Some(max_steps) = config.max_steps {
                    if epoch_steps >= max_steps {
                        break;
                    }
                }
                if let Some(trainer) = trainer.as_mut() {
                    let losses = trainer.train_batch(
                        &batch,
                        train_params.as_ref().map(Vec::as_slice).unwrap_or(&[]),
                    );
                    for (idx, loss) in losses.iter().enumerate() {
                        let loss_val = loss.data.to_vec(&ctx).await.first().copied().unwrap_or(0.0);
                        total_loss += loss_val;
                        let counted = batch
                            .loss_masks
                            .get(idx)
                            .map(|m| m.iter().filter(|&&v| v == 1).count())
                            .unwrap_or(0);
                        total_tokens += counted;
                        total_samples += 1;
                    }
                }
                epoch_steps += 1;
                if let Some(max_samples) = max_samples_effective {
                    if total_samples >= max_samples {
                        break;
                    }
                }
            }
        } else {
            for sample_idx in 0..dataset.len() {
                if let Some(max_steps) = config.max_steps {
                    if epoch_steps >= max_steps {
                        break;
                    }
                }
                if let Some(max_samples) = max_samples_effective {
                    if total_samples >= max_samples {
                        break;
                    }
                }
                let (input_ids, target_vec) = dataset.get(sample_idx);
                let loss_mask = dataset
                    .get_loss_mask(sample_idx)
                    .unwrap_or_else(|| vec![1u8; target_vec.len()]);
                if input_ids.len() < 2 {
                    continue;
                }
                let logits = eval_model
                    .as_ref()
                    .expect("eval model")
                    .forward(&engine, &input_ids, 0, None, None);
                let logits_data = logits.data.to_vec(&ctx).await;
                if let Some((loss_val, count)) = masked_cross_entropy_from_logits(
                    &logits_data,
                    model_config.vocab_size,
                    &target_vec,
                    &loss_mask,
                ) {
                    total_loss += loss_val;
                    total_tokens += count;
                    total_samples += 1;
                }
                epoch_steps += 1;
            }
        }

        let avg_loss = total_loss / total_samples.max(1) as f32;
        let perplexity = if total_tokens > 0 {
            avg_loss.exp()
        } else {
            f32::INFINITY
        };
        best_loss = Some(best_loss.map_or(avg_loss, |best| best.min(avg_loss)));
        let progress = "=".repeat(((epoch + 1) * 20 / epochs).max(1));
        println!(
            "   {:5} |   {:9.4} | {}  ppl={:.3}",
            epoch, avg_loss, progress, perplexity
        );

        if train_mode && is_root {
            let state = TrainingState {
                step: ((epoch + 1) * steps_per_epoch).min(total_steps),
                total_steps,
                best_loss,
                lr: learning_rate,
                epoch: epoch + 1,
                batch_size,
                grad_accum_steps: grad_accum,
                sequence_length: config.sequence_length,
                fp16: config.fp16,
                train_modules: config.train_modules.clone(),
                task: config.task.as_str().to_string(),
                model_path: model_path.to_string(),
                dataset_path: config.dataset_path.clone().unwrap_or_default(),
            };
            let monitor_path = Path::new(&out_dir).join("training_state.json");
            let _ = state.save(&monitor_path);
            let _ = state.save("outputs/training_state.json");
            if let (Some(trainer), Some(named)) = (trainer.as_ref(), named_train_params.as_ref()) {
                let checkpoint = TrainingCheckpoint {
                    training_state: state,
                    optimizer_state: trainer.optimizer.save_state(&ctx, named),
                    grad_scaler_state: trainer.scaler.as_ref().map(|s| s.checkpoint()),
                };
                let _ = checkpoint.save(&checkpoint_path);
            }
        }

        if let Some(max_steps) = config.max_steps {
            if epoch_steps >= max_steps {
                break;
            }
        }
    }

    if !named_lora_params.is_empty() && train_mode && is_root {
        println!("[SAVE] Writing LoRA adapters to {}...", save_path.display());
        match save_lora_checkpoint(&ctx, &save_path, &named_lora_params).await {
            Ok(()) => println!("       Saved {} tensors", named_lora_params.len()),
            Err(e) => println!("       [ERROR] Failed to save LoRA: {e}"),
        }
    }

    if !config.no_infer {
        let inference_model = if let Some(trainer) = trainer.as_ref() {
            &trainer.model
        } else {
            eval_model.as_ref().expect("eval model")
        };
        run_infer(
            &ctx,
            &engine,
            inference_model,
            &tokenizer,
            model_config.vocab_size,
            "Hello!",
        )
        .await;
    } else {
        println!("[TEST] Skipping inference (--no-infer).");
    }

    println!("[DONE]");
}

async fn run_infer(
    ctx: &Arc<WgpuContext>,
    engine: &Arc<ComputeEngine>,
    model: &rusty_graph::Lfm2Model,
    tokenizer: &Tokenizer,
    vocab_size: usize,
    prompt: &str,
) {
    println!("[INFER] prompt={prompt:?}");
    let sampler = Sampler::default();
    let mut input_tokens = tokenizer.encode_chat(None, prompt);
    let mut generated = Vec::new();
    for _ in 0..32 {
        let logits = model.forward(engine, &input_tokens, 0, None, None);
        let logits_data = logits.data.to_vec(ctx).await;
        let seq_len = logits_data.len() / vocab_size;
        if seq_len == 0 {
            break;
        }
        let start = (seq_len - 1) * vocab_size;
        let end = (start + vocab_size).min(logits_data.len());
        let token = sampler.sample(&logits_data[start..end]);
        if token == tokenizer.eos_token_id {
            break;
        }
        generated.push(token);
        input_tokens.push(token);
    }
    println!("       {}", tokenizer.decode(&generated));
}

fn load_task_dataset(
    config: &RunConfig,
    tokenizer: &Tokenizer,
) -> Result<TextDataset, Box<dyn std::error::Error>> {
    let path = config.dataset_path.as_ref().ok_or("dataset path missing")?;
    let dataset_path = Path::new(path);
    if !dataset_path.exists() {
        return Err(format!("dataset file not found: {path}").into());
    }

    let extension = dataset_path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();

    match config.task {
        TaskKind::Sft => match extension.as_str() {
            "jsonl" => TextDataset::from_jsonl_sft(
                dataset_path,
                tokenizer,
                config.use_chat_template,
                config.sequence_length,
            )
            .map_err(|e| e.into()),
            "json" => load_json_sft_dataset(
                dataset_path,
                tokenizer,
                config.use_chat_template,
                config.sequence_length,
            ),
            "txt" | "bin" => Err(
                "SFT task expects .json/.jsonl with {prompt,response}. Use --task pretrain for txt/bin."
                    .into(),
            ),
            _ => TextDataset::from_jsonl_sft(
                dataset_path,
                tokenizer,
                config.use_chat_template,
                config.sequence_length,
            )
            .map_err(|e| e.into()),
        },
        TaskKind::Pretrain => match extension.as_str() {
            "bin" => TextDataset::from_token_bin(dataset_path, config.sequence_length)
                .map_err(|e| e.into()),
            "txt" => {
                let ds = TextDataset::from_text_file(dataset_path, tokenizer)?;
                // Optionally pack short lines into windows.
                let seqs: Vec<Vec<u32>> = (0..ds.len()).map(|i| {
                    let (inp, tgt) = ds.get(i);
                    let mut full = inp;
                    if let Some(last) = tgt.last() {
                        full.push(*last);
                    }
                    full
                }).collect();
                Ok(TextDataset::pack_sequences(seqs, config.sequence_length))
            }
            "json" => {
                // Accept token-id arrays or raw strings for pretrain.
                let content = fs::read_to_string(dataset_path)?;
                if let Ok(tokens) = serde_json::from_str::<Vec<u32>>(&content) {
                    Ok(TextDataset::from_packed_tokens(tokens, config.sequence_length))
                } else if let Ok(lines) = serde_json::from_str::<Vec<String>>(&content) {
                    let sequences = lines
                        .into_iter()
                        .filter(|l| !l.trim().is_empty())
                        .map(|l| tokenizer.encode(&l))
                        .collect();
                    Ok(TextDataset::pack_sequences(sequences, config.sequence_length))
                } else {
                    Err("pretrain json must be [token_ids] or [\"text\", ...]".into())
                }
            }
            _ => TextDataset::from_text_file(dataset_path, tokenizer).map_err(|e| e.into()),
        },
    }
}

fn load_json_sft_dataset(
    path: &Path,
    tokenizer: &Tokenizer,
    use_chat_template: bool,
    max_len: usize,
) -> Result<TextDataset, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;
    if let Ok(samples) = serde_json::from_str::<Vec<Sample>>(&content) {
        let mut dataset = TextDataset::new();
        for sample in samples {
            let prompt_tokens = if use_chat_template {
                tokenizer.encode_chat(sample.system.as_deref(), &sample.prompt)
            } else {
                tokenizer.encode(&sample.prompt)
            };
            let response_tokens = tokenizer.encode(&sample.response);
            if let Some(example) = truncate_example(
                TrainingExample::from_sft_prompt_response(prompt_tokens, response_tokens),
                max_len,
            ) {
                dataset.add_example(example);
            }
        }
        return Ok(dataset);
    }
    Err("unsupported sft json; expected [{prompt,response}]".into())
}

fn truncate_example(mut example: TrainingExample, max_len: usize) -> Option<TrainingExample> {
    if example.input_ids.len() < 2 {
        return None;
    }
    if example.input_ids.len() > max_len && max_len >= 2 {
        let drop = example.input_ids.len().saturating_sub(max_len);
        example.input_ids = example.input_ids[drop..].to_vec();
        example.target_ids = example.target_ids[drop..].to_vec();
        example.loss_mask = example.loss_mask[drop..].to_vec();
    }
    if example.input_ids.len() < 2 {
        None
    } else {
        Some(example)
    }
}

fn create_default_dataset(
    tokenizer: &Tokenizer,
    sequence_length: usize,
    task: TaskKind,
) -> TextDataset {
    let samples = vec![
        Sample {
            prompt: "Hello".into(),
            response: "Hi! How can I help you today?".into(),
            system: None,
        },
        Sample {
            prompt: "What is Rust?".into(),
            response: "Rust is a systems programming language focused on safety and performance."
                .into(),
            system: None,
        },
    ];
    match task {
        TaskKind::Sft => {
            let mut dataset = TextDataset::new();
            for sample in samples {
                let prompt_tokens = tokenizer.encode_chat(None, &sample.prompt);
                let response_tokens = tokenizer.encode(&sample.response);
                if let Some(example) = truncate_example(
                    TrainingExample::from_sft_prompt_response(prompt_tokens, response_tokens),
                    sequence_length,
                ) {
                    dataset.add_example(example);
                }
            }
            dataset
        }
        TaskKind::Pretrain => {
            let sequences = samples
                .into_iter()
                .map(|s| tokenizer.encode(&format!("{} {}", s.prompt, s.response)))
                .collect();
            TextDataset::pack_sequences(sequences, sequence_length)
        }
    }
}

fn create_byte_tokenizer() -> Tokenizer {
    let mut vocab = std::collections::HashMap::new();
    for i in 0u32..256 {
        vocab.insert(format!("{}", i as u8 as char), i);
    }
    vocab.insert("<s>".to_string(), 1);
    vocab.insert("</s>".to_string(), 2);
    vocab.insert("<pad>".to_string(), 0);
    let id_to_token: std::collections::HashMap<u32, String> =
        vocab.iter().map(|(k, v)| (*v, k.clone())).collect();
    Tokenizer {
        vocab,
        id_to_token,
        bos_token_id: 1,
        eos_token_id: 2,
        pad_token_id: 0,
        hf_tokenizer: None,
    }
}

fn estimate_model_weight_gb(model_path: &str) -> f32 {
    fn dir_size(path: &Path) -> u64 {
        let Ok(entries) = fs::read_dir(path) else {
            return 0;
        };
        entries
            .filter_map(Result::ok)
            .map(|entry| {
                let path = entry.path();
                if path.is_dir() {
                    dir_size(&path)
                } else {
                    entry.metadata().map(|meta| meta.len()).unwrap_or(0)
                }
            })
            .sum()
    }
    dir_size(Path::new(model_path)) as f32 / (1024.0 * 1024.0 * 1024.0)
}

fn print_capacity_guidance(
    model_path: &str,
    hidden: usize,
    layers: usize,
    seq_len: usize,
    batch_size: usize,
    max_vram_gb: Option<f32>,
) {
    let weight_gb = estimate_model_weight_gb(model_path);
    let activation_gb = batch_size as f32 * seq_len as f32 * hidden as f32 * layers as f32 * 4.0
        / (1024.0 * 1024.0 * 1024.0);
    let estimated = weight_gb + activation_gb * 2.0;
    println!("[CHECK] estimated working set ~{estimated:.2} GB (weights {weight_gb:.2} GB)");
    let budget = max_vram_gb.unwrap_or(8.0);
    if estimated > budget {
        println!("        [WARN] exceeds {budget:.1} GB budget; prefer --lite / lower --seq-len / LoRA-only");
    }
}

fn checkpoint_state_path(save_path: &Path) -> PathBuf {
    let filename = save_path
        .file_name()
        .and_then(|name| name.to_str())
        .map(|name| format!("{name}.state.json"))
        .unwrap_or_else(|| "training_checkpoint.state.json".to_string());
    save_path.with_file_name(filename)
}

fn select_named_train_params(
    model: &rusty_graph::Lfm2Model,
    train_modules: &str,
) -> std::collections::HashMap<String, rusty_autograd::Tensor> {
    match train_modules.to_ascii_lowercase().as_str() {
        "all" => model.named_params(),
        "lora" => model.named_lora_params(),
        mode => model
            .named_params()
            .into_iter()
            .filter(|(name, _)| match mode {
                "attn" | "attention" => name.contains("self_attn"),
                "mlp" => name.contains("feed_forward"),
                "norm" => name.contains("norm"),
                "embed" | "embedding" => name.contains("embed_tokens"),
                "lm_head" | "head" => name.contains("lm_head"),
                _ => name.contains(mode),
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn train_subcommand_sets_mode_and_task() {
        let args = vec![
            "--model".into(),
            "m".into(),
            "--data".into(),
            "d.jsonl".into(),
            "--task".into(),
            "pretrain".into(),
            "--lite".into(),
        ];
        let config = parse_subcommand("train", &args);
        assert_eq!(config.mode, CommandMode::Train);
        assert_eq!(config.task, TaskKind::Pretrain);
        assert!(config.lite_mode);
        assert!(config.no_infer);
        assert_eq!(config.grad_accum, 8);
    }

    #[test]
    fn sft_default_uses_chat_template() {
        let config = default_config(CommandMode::Train);
        assert!(config.use_chat_template);
        assert_eq!(config.task, TaskKind::Sft);
    }

    #[test]
    fn lite_preserves_explicit_grad_accum_override() {
        let args = vec![
            "--model".into(),
            "m".into(),
            "--data".into(),
            "d.jsonl".into(),
            "--lite".into(),
            "--grad-accum".into(),
            "4".into(),
            "--seq-len".into(),
            "64".into(),
            "--epochs".into(),
            "2".into(),
        ];
        let config = parse_subcommand("train", &args);
        assert_eq!(config.grad_accum, 4);
        assert_eq!(config.sequence_length, 64);
        assert_eq!(config.epochs, 2);
        assert_eq!(config.train_modules, "lora");
        assert!(config.fp16);
    }

    #[test]
    fn allow_fallback_defaults_off() {
        let config = default_config(CommandMode::Train);
        assert!(!config.allow_fallback);
    }

    #[test]
    fn eval_lite_keeps_eval_mode() {
        let args = vec![
            "--model".into(),
            "m".into(),
            "--data".into(),
            "d.jsonl".into(),
            "--lite".into(),
            "--task".into(),
            "sft".into(),
        ];
        let config = parse_subcommand("eval", &args);
        assert_eq!(config.mode, CommandMode::Eval);
        assert!(config.lite_mode);
        assert_eq!(config.grad_accum, 8);
    }

    #[test]
    fn distributed_flags_parse() {
        let args = vec![
            "--model".into(),
            "m".into(),
            "--data".into(),
            "d.jsonl".into(),
            "--world-size".into(),
            "2".into(),
            "--rank".into(),
            "1".into(),
            "--sync-dir".into(),
            "./tmp_sync".into(),
            "--task".into(),
            "sft".into(),
        ];
        let config = parse_subcommand("train", &args);
        assert_eq!(config.mode, CommandMode::Train);
        assert_eq!(config.world_size, 2);
        assert_eq!(config.rank, 1);
        assert_eq!(config.sync_dir.as_deref(), Some("./tmp_sync"));
    }
}
