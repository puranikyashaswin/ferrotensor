//! `ferrotensor pretrain`: train a small Llama-style model from random
//! initialisation on a text file, with a byte-level tokenizer or a BPE one
//! learned from the same file, leaving a model directory that `generate`
//! loads.

use crate::{
    print_json, train, unmet, DataArgs, Device, ModelArgs, Result, Task, TrainArgs, TrainSummary,
};
use clap::{Args, ValueEnum};
use ferrotensor_loader::tokenizer::{
    byte_level_tokenizer_json, train_byte_level_bpe, Tokenizer, BYTE_LEVEL_FIRST_BYTE_ID,
    BYTE_LEVEL_VOCAB_SIZE,
};
use ferrotensor_trainer::checkpoint::load_lora_checkpoint;
use ferrotensor_trainer::merge::MERGED_WEIGHTS_FILE;
use ferrotensor_trainer::pretrain::{write_hf_weights, write_random_llama, LlamaSize};
use std::fs::{self, File};
use std::io::{BufWriter, Write as _};
use std::path::{Path, PathBuf};

/// Subdirectory of `--out` that holds the random initial model training
/// starts from, so `--out` holds `model.safetensors` only once trained
/// weights are exported into it.
pub(crate) const INIT_DIR: &str = "init";

/// How `pretrain` turns the corpus into tokens.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum, serde::Serialize)]
#[serde(rename_all = "lowercase")]
enum TokenizerKind {
    /// One token per byte, 259 tokens in all.
    Byte,
    /// Byte-level BPE learned from --data, of --vocab-size tokens.
    Bpe,
}

/// Options for `ferrotensor pretrain`. The training defaults are
/// conservative and not tuned.
#[derive(Args, Debug)]
pub struct PretrainArgs {
    /// Text file to train on; newlines and other whitespace are kept.
    /// Training holds about 20 bytes of host memory per token.
    #[arg(long, value_name = "FILE")]
    data: PathBuf,
    /// Tokenizer written to --out. `bpe` needs UTF-8 text and is learned
    /// before training; --resume reuses the saved one.
    #[arg(long, value_enum, default_value_t = TokenizerKind::Byte)]
    tokenizer: TokenizerKind,
    /// Tokens of the `bpe` tokenizer: the 259 byte-level ones plus one per
    /// merge learned [default: 4096].
    #[arg(long)]
    vocab_size: Option<usize>,
    /// Model directory to write: config.json, tokenizer.json, tokens.bin,
    /// the initial model in init/ and the training state, and
    /// model.safetensors when training ends.
    #[arg(long, value_name = "DIR")]
    out: PathBuf,
    /// Continue the run in --out from its trained_weights.safetensors and
    /// training state. Pass the flags of the first run; --max-steps counts
    /// the steps before the resume too, and at or below the saved step
    /// exports the saved weights without training.
    #[arg(long)]
    resume: bool,
    /// Decoder blocks.
    #[arg(long, default_value_t = 4)]
    layers: usize,
    /// Width of the residual stream; a multiple of --heads.
    #[arg(long, default_value_t = 128)]
    hidden: usize,
    /// Attention heads. The head size --hidden / --heads must be even (RoPE)
    /// and at most 128 (the attention kernel).
    #[arg(long, default_value_t = 4)]
    heads: usize,
    /// Width of the MLP hidden layer [default: 4 x --hidden].
    #[arg(long)]
    mlp_hidden: Option<usize>,
    /// Longest sequence in tokens, written as max_position_embeddings.
    #[arg(long, default_value_t = 256)]
    context: usize,
    /// Tokens per training window, at most --context [default: --context].
    #[arg(long)]
    seq_len: Option<usize>,
    /// Peak learning rate (`train` defaults to 2e-4 for fine-tuning).
    #[arg(long, default_value_t = 1e-3)]
    lr: f32,
    /// Linear warmup steps before cosine decay to lr/10.
    #[arg(long, default_value_t = 100)]
    warmup_steps: usize,
    /// Optimizer steps of the whole run.
    #[arg(long, default_value_t = 2000)]
    max_steps: usize,
    /// Windows per batch.
    #[arg(long, default_value_t = 8)]
    batch_size: usize,
    /// Batches per optimizer step.
    #[arg(long, default_value_t = 1)]
    grad_accum: usize,
    /// Clip the global gradient norm.
    #[arg(long, default_value_t = 1.0)]
    grad_clip: f32,
    /// AdamW weight decay (also applied to norm scales and the embedding).
    #[arg(long, default_value_t = 0.0)]
    weight_decay: f32,
    /// Log every N optimizer steps.
    #[arg(long, default_value_t = 50)]
    log_every: usize,
    /// Save the trained weights and training state every N optimizer steps,
    /// for --resume (0 = only at the end).
    #[arg(long, default_value_t = 500)]
    save_every: usize,
    /// Seed of the initial weights and of the data order.
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Compute device.
    #[arg(long, value_enum, default_value_t = Device::Auto)]
    device: Device,
    /// Recompute each decoder block during backward instead of keeping its
    /// activations.
    #[arg(long)]
    checkpoint_blocks: bool,
    /// Print one JSON summary to stdout when the run ends; everything else
    /// goes to stderr.
    #[arg(long)]
    pub json: bool,
}

/// What `pretrain --json` prints: every `train` key, plus the model's
/// parameter count, vocabulary size and tokenizer.
#[derive(serde::Serialize, Debug)]
struct PretrainSummary {
    #[serde(flatten)]
    train: TrainSummary,
    parameters: usize,
    vocab_size: usize,
    tokenizer: TokenizerKind,
}

/// The vocabulary size `args` asks for, or the flag that is out of range.
fn vocab_size(args: &PretrainArgs) -> Result<usize> {
    match (args.tokenizer, args.vocab_size) {
        (TokenizerKind::Byte, None) => Ok(BYTE_LEVEL_VOCAB_SIZE),
        (TokenizerKind::Byte, Some(_)) => Err(unmet("--vocab-size needs --tokenizer bpe")),
        (TokenizerKind::Bpe, size) => match size.unwrap_or(4096) {
            size if size > BYTE_LEVEL_VOCAB_SIZE => Ok(size),
            size => Err(unmet(format!(
                "--vocab-size {size} leaves no token to learn; it must be above the {BYTE_LEVEL_VOCAB_SIZE} byte-level tokens"
            ))),
        },
    }
}

/// The model `args` asks for, or the flag that is out of range.
fn model_size(args: &PretrainArgs) -> Result<LlamaSize> {
    let mlp_hidden = args.mlp_hidden.unwrap_or(4 * args.hidden);
    if [
        args.layers,
        args.hidden,
        args.heads,
        mlp_hidden,
        args.context,
    ]
    .contains(&0)
    {
        return Err(unmet(
            "--layers, --hidden, --heads, --mlp-hidden and --context must be positive",
        ));
    }
    if !args.hidden.is_multiple_of(args.heads) {
        return Err(unmet(format!(
            "--hidden {} is not a multiple of --heads {}",
            args.hidden, args.heads
        )));
    }
    let head_size = args.hidden / args.heads;
    if !head_size.is_multiple_of(2) || head_size > 128 {
        return Err(unmet(format!(
            "the head size --hidden / --heads is {head_size}; RoPE needs it even and the attention kernel at most 128"
        )));
    }
    Ok(LlamaSize {
        layers: args.layers,
        hidden: args.hidden,
        heads: args.heads,
        mlp_hidden,
        context: args.context,
        vocab: vocab_size(args)?,
    })
}

/// The `train` run on `--out` from the model in [`INIT_DIR`]: every base
/// weight and the embedding train in f32 on the packed tokens. Every field
/// is named, so a new `train` flag must be decided here.
fn train_args(args: &PretrainArgs, seq_len: usize) -> TrainArgs {
    TrainArgs {
        model: ModelArgs {
            model: args.out.join(INIT_DIR),
            device: args.device,
            f32_weights: true,
            f16_weights: false,
            q4_weights: false,
        },
        data: DataArgs {
            data: args.out.join("tokens.bin"),
            task: Task::Pretrain,
            seq_len,
            max_samples: None,
            no_chat_template: true,
        },
        out: args.out.clone(),
        save_lora: None,
        resume: args
            .resume
            .then(|| args.out.join("trained_weights.safetensors")),
        lr: args.lr,
        warmup_steps: args.warmup_steps,
        epochs: 1,
        max_steps: Some(args.max_steps),
        batch_size: args.batch_size,
        grad_accum: args.grad_accum,
        grad_clip: args.grad_clip,
        weight_decay: args.weight_decay,
        lora_rank: 8,
        lora_alpha: 16.0,
        lora_targets: Vec::new(),
        train_modules: "full".to_string(),
        log_every: args.log_every,
        save_every: args.save_every,
        eval_data: None,
        eval_every: 0,
        eval_samples: 64,
        seed: args.seed,
        sample_prompt: None,
        lite: false,
        checkpoint_blocks: args.checkpoint_blocks,
        world_size: 1,
        rank: 0,
        sync_dir: None,
        memory_report: None,
        profile: false,
        json: args.json,
    }
}

/// Runs `ferrotensor pretrain`: writes the tokenizer, `config.json` and a
/// random model in [`INIT_DIR`] (unless resuming), encodes the corpus,
/// trains, and exports the trained weights as `model.safetensors`.
pub async fn run(args: PretrainArgs) -> Result<()> {
    let size = model_size(&args)?;
    let seq_len = args.seq_len.unwrap_or(args.context);
    if !(2..=args.context).contains(&seq_len) {
        return Err(unmet(format!(
            "--seq-len must be between 2 and --context {}",
            args.context
        )));
    }
    if !args.data.is_file() {
        return Err(unmet(format!(
            "dataset file {} does not exist",
            args.data.display()
        )));
    }
    let weights_path = args.out.join(MERGED_WEIGHTS_FILE);
    let trained_path = args.out.join("trained_weights.safetensors");
    let init_dir = args.out.join(INIT_DIR);
    if args.resume && !trained_path.is_file() {
        return Err(unmet(format!(
            "{} does not exist; --resume continues a run that saved it",
            trained_path.display()
        )));
    }
    if !args.resume && (weights_path.exists() || init_dir.exists()) {
        return Err(unmet(format!(
            "{} already holds a model; pass --resume to continue it, or choose a new --out",
            args.out.display()
        )));
    }
    tokenize_corpus(&args, &size)?;

    let summary = train(train_args(&args, seq_len)).await?;
    let trained = load_lora_checkpoint(&trained_path)?;
    let parameters = trained
        .tensors
        .values()
        .map(|(_, values)| values.len())
        .sum();
    write_hf_weights(trained, &weights_path)?;
    eprintln!(
        "export: {} ({parameters} parameters, Hugging Face Llama layout)",
        weights_path.display()
    );
    if args.json {
        return print_json(&PretrainSummary {
            train: summary,
            parameters,
            vocab_size: size.vocab,
            tokenizer: args.tokenizer,
        });
    }
    eprintln!("next:");
    eprintln!(
        "  ferrotensor generate --model {} --raw --prompt \"...\"",
        args.out.display()
    );
    Ok(())
}

/// Reads the corpus, writes the tokenizer and the initial model unless
/// resuming (a resume checks the saved tokenizer against the flags instead),
/// and encodes the corpus into `tokens.bin`.
fn tokenize_corpus(args: &PretrainArgs, size: &LlamaSize) -> Result<()> {
    let corpus =
        fs::read(&args.data).map_err(|e| format!("cannot read {}: {e}", args.data.display()))?;
    let bpe_text = match args.tokenizer {
        TokenizerKind::Byte => None,
        TokenizerKind::Bpe => Some(std::str::from_utf8(&corpus).map_err(|e| {
            unmet(format!(
                "{} is not UTF-8 text (byte {}); --tokenizer bpe needs UTF-8",
                args.data.display(),
                e.valid_up_to()
            ))
        })?),
    };
    if !args.resume {
        let tokenizer = match bpe_text {
            None => byte_level_tokenizer_json(),
            Some(text) => {
                let learned = train_byte_level_bpe(text, size.vocab)
                    .map_err(|e| unmet(format!("--vocab-size {}: {e}", size.vocab)))?;
                eprintln!(
                    "bpe:    {} tokens learned from {}",
                    size.vocab,
                    args.data.display()
                );
                learned
            }
        };
        write_initial_model(args, size, &tokenizer.to_string())?;
    }
    let tokenizer_path = args.out.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)?;
    if tokenizer.vocab.len() != size.vocab {
        return Err(unmet(format!(
            "{} has {} tokens and the flags ask for {}; pass the --tokenizer and --vocab-size of the first run",
            tokenizer_path.display(),
            tokenizer.vocab.len(),
            size.vocab
        )));
    }

    let tokens_path = args.out.join("tokens.bin");
    // Lines are encoded one at a time, as the BPE tokenizer learned them.
    let ids: Box<dyn Iterator<Item = u32> + '_> = match bpe_text {
        None => Box::new(
            corpus
                .iter()
                .map(|&byte| u32::from(byte) + BYTE_LEVEL_FIRST_BYTE_ID),
        ),
        Some(text) => Box::new(
            text.split_inclusive('\n')
                .flat_map(|line| tokenizer.encode_continuation(line)),
        ),
    };
    let token_count = write_tokens(ids, &tokens_path)?;
    eprintln!(
        "tokens: {token_count} from {} bytes of {} -> {}",
        corpus.len(),
        args.data.display(),
        tokens_path.display()
    );
    Ok(())
}

/// Writes `tokenizer` into `--out` and [`INIT_DIR`], and a random model of
/// `size` with its `config.json` into [`INIT_DIR`], copying the config to
/// `--out`.
fn write_initial_model(args: &PretrainArgs, size: &LlamaSize, tokenizer: &str) -> Result<()> {
    let init_dir = args.out.join(INIT_DIR);
    fs::create_dir_all(&init_dir)
        .map_err(|e| format!("cannot create {}: {e}", init_dir.display()))?;
    for dir in [&args.out, &init_dir] {
        fs::write(dir.join("tokenizer.json"), tokenizer)?;
    }
    write_random_llama(&init_dir, size, args.seed)?;
    fs::copy(init_dir.join("config.json"), args.out.join("config.json"))?;
    eprintln!(
        "init:   {} parameters ({} layers, hidden {}, {} heads, mlp {}, context {}, vocab {}), seed {}",
        size.parameters(),
        size.layers,
        size.hidden,
        size.heads,
        size.mlp_hidden,
        size.context,
        size.vocab,
        args.seed
    );
    Ok(())
}

/// Writes `ids` to `path` as packed little-endian `u32`, returning how many.
fn write_tokens(ids: impl Iterator<Item = u32>, path: &Path) -> Result<usize> {
    let mut tokens = BufWriter::new(File::create(path)?);
    let mut count = 0;
    for id in ids {
        tokens.write_all(&id.to_le_bytes())?;
        count += 1;
    }
    tokens
        .into_inner()
        .map_err(|e| e.into_error())?
        .sync_all()?;
    Ok(count)
}
