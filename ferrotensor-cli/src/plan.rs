//! `ferrotensor plan`: the largest LoRA training configuration that is
//! expected to fit this machine's GPU memory, from a first-principles
//! estimate calibrated on measured peaks.

use clap::{Args, ValueEnum};
use ferrotensor_backend::adapters::{self, AdapterReport};
use ferrotensor_loader::{Lfm2Config, ModelLoader};
use serde_json::{json, Map, Value};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

const MIB: f64 = 1024.0 * 1024.0;

/// Fixed GPU memory every run holds: the driver context and the buffer
/// pool's idle buffers (cap 512 MiB). Fitted to the measured peaks.
const OVERHEAD_MIB: u64 = 1350;
/// f32 values kept per token per decoder block for backward, per unit of
/// hidden size, FFN width and attention row (n_heads x seq_len). Fitted to
/// the measured peaks; the ratio follows the tensors a Llama block keeps.
const ACTIVATION_FLOATS_PER_HIDDEN: f64 = 16.0;
const ACTIVATION_FLOATS_PER_FFN: f64 = 6.0;
const ACTIVATION_FLOATS_PER_ATTENTION_ROW: f64 = 3.0;
/// The logits and their gradient, both f32 and vocab-sized per token.
const LOGIT_FLOATS_PER_VOCAB: f64 = 2.0;
/// Fraction of the budget a plan leaves free; above the estimator's worst
/// error on the measured peaks (13.3%).
const SAFETY_MARGIN: f64 = 0.15;
/// Share of system RAM budgeted for a GPU that shares it (the training
/// process itself needs the rest).
const UNIFIED_RAM_PERCENT: u64 = 60;
/// Budget when nothing better is known.
const ASSUMED_BUDGET_MIB: u64 = 4096;
/// Sequences per optimizer step, as in the measured runs.
const SEQUENCES_PER_STEP: usize = 8;
const BATCH_SIZES: [usize; 4] = [8, 4, 2, 1];
const SEQ_LENS: [usize; 2] = [256, 128];

/// Options for `ferrotensor plan`.
#[derive(Args, Debug)]
pub struct PlanArgs {
    /// Model directory (only config.json is read). Without it, the
    /// documented SmolLM2 sizes are listed.
    #[arg(long, value_name = "DIR")]
    pub model: Option<PathBuf>,
    /// GPU memory budget in MiB; overrides everything else.
    #[arg(long, value_name = "MIB")]
    pub memory_mib: Option<u64>,
    /// Memory of a discrete GPU in MiB (wgpu cannot read it).
    #[arg(long, value_name = "MIB")]
    pub vram_mib: Option<u64>,
    /// Frozen base weight precision to plan for.
    #[arg(long, value_enum, default_value_t = Precision::F16)]
    pub weights: Precision,
    /// LoRA rank.
    #[arg(long, default_value_t = 8)]
    pub lora_rank: usize,
    /// Plan for this sequence length only [default: try 256, then 128].
    #[arg(long)]
    pub seq_len: Option<NonZeroUsize>,
    /// Print one JSON object.
    #[arg(long)]
    pub json: bool,
}

/// Storage of the frozen base weights.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum Precision {
    /// f16 decoder and lm_head weights (the `train` default).
    F16,
    /// Every weight in f32 (`--f32-weights`).
    F32,
    /// 4-bit decoder weights, f16 lm_head (`--q4-weights`, experimental).
    Q4,
}

impl Precision {
    fn as_str(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::Q4 => "q4",
        }
    }

    fn decoder_bytes_per_weight(self) -> f64 {
        match self {
            Self::F16 => 2.0,
            Self::F32 => 4.0,
            // 4 bits plus an f16 scale and bias per group of 64.
            Self::Q4 => 4.5 / 8.0,
        }
    }

    fn lm_head_bytes_per_weight(self) -> f64 {
        match self {
            Self::F16 | Self::Q4 => 2.0,
            Self::F32 => 4.0,
        }
    }

    fn train_flag(self) -> &'static str {
        match self {
            Self::F16 => "",
            Self::F32 => " --f32-weights",
            Self::Q4 => " --q4-weights",
        }
    }
}

/// Whether decoder blocks are recomputed during backward.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Checkpointing {
    /// Keep every block's activations.
    Off,
    /// `--checkpoint-blocks`: keep only block inputs.
    Blocks,
}

/// Sizes of a Llama-style decoder that decide its memory.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelShape {
    /// Name shown in the output.
    pub name: String,
    /// Width of the residual stream.
    pub hidden: usize,
    /// Width of the MLP hidden layer.
    pub ffn: usize,
    /// Decoder blocks.
    pub n_layers: usize,
    /// Query heads.
    pub n_heads: usize,
    /// Key/value heads.
    pub n_kv_heads: usize,
    /// Width of one head.
    pub head_dim: usize,
    /// Vocabulary size.
    pub vocab: usize,
    /// Whether lm_head shares the embedding in the checkpoint.
    pub tied_embeddings: bool,
}

impl ModelShape {
    fn from_config(name: String, config: &Lfm2Config) -> Self {
        Self {
            name,
            hidden: config.hidden_size,
            ffn: config.intermediate_size,
            n_layers: config.num_hidden_layers,
            n_heads: config.num_attention_heads,
            n_kv_heads: config
                .num_key_value_heads
                .unwrap_or(config.num_attention_heads),
            head_dim: config
                .head_dim
                .unwrap_or(config.hidden_size / config.num_attention_heads),
            vocab: config.vocab_size,
            tied_embeddings: config.tie_word_embeddings.unwrap_or(false),
        }
    }

    fn smollm2(name: &str, hidden: usize, ffn: usize, layers: usize, heads: [usize; 2]) -> Self {
        Self {
            name: name.into(),
            hidden,
            ffn,
            n_layers: layers,
            n_heads: heads[0],
            n_kv_heads: heads[1],
            head_dim: 64,
            vocab: 49152,
            tied_embeddings: true,
        }
    }

    /// The SmolLM2 sizes whose peaks are measured in the docs.
    pub fn documented() -> [Self; 3] {
        [
            Self::smollm2("SmolLM2-135M", 576, 1536, 30, [9, 3]),
            Self::smollm2("SmolLM2-360M", 960, 2560, 32, [15, 5]),
            Self::smollm2("SmolLM2-1.7B", 2048, 8192, 24, [32, 32]),
        ]
    }

    fn decoder_weights(&self) -> u64 {
        let q_width = (self.n_heads * self.head_dim) as u64;
        let kv_width = (self.n_kv_heads * self.head_dim) as u64;
        let h = self.hidden as u64;
        let per_layer = 2 * h * q_width + 2 * h * kv_width + 3 * h * self.ffn as u64;
        self.n_layers as u64 * per_layer
    }

    /// Parameter count, including norms, counting lm_head once when tied.
    pub fn parameters(&self) -> u64 {
        let h = self.hidden as u64;
        let embedding = self.vocab as u64 * h;
        let norms = (2 * self.n_layers as u64 + 1) * h;
        let lm_head = if self.tied_embeddings { 0 } else { embedding };
        self.decoder_weights() + embedding + norms + lm_head
    }
}

/// One training configuration.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TrainConfig {
    /// Sequences per forward pass.
    pub batch_size: usize,
    /// Tokens per sequence.
    pub seq_len: usize,
    /// LoRA rank (adapters on q, k, v and o projections).
    pub lora_rank: usize,
    /// Frozen base weight storage.
    pub precision: Precision,
    /// Block checkpointing.
    pub checkpointing: Checkpointing,
}

/// Estimated GPU memory of a training run, by part, in MiB.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Estimate {
    /// Frozen base weights (embedding in f32, lm_head and decoder blocks at
    /// the chosen precision).
    pub weights_mib: u64,
    /// LoRA adapters, their gradients and AdamW moments.
    pub adapters_mib: u64,
    /// Activations kept for backward.
    pub activations_mib: u64,
    /// Logits and their gradient.
    pub logits_mib: u64,
    /// Driver context and buffer pool.
    pub overhead_mib: u64,
    /// Sum of the parts.
    pub peak_mib: u64,
}

fn ceil_mib(bytes: f64) -> u64 {
    (bytes / MIB).ceil() as u64
}

/// Estimated peak GPU memory of LoRA training `shape` with `config`.
pub fn estimate_peak_mib(shape: &ModelShape, config: &TrainConfig) -> Estimate {
    let h = shape.hidden as f64;
    let vocab = shape.vocab as f64;
    let layers = shape.n_layers as f64;
    let tokens = (config.batch_size * config.seq_len) as f64;
    let precision = config.precision;

    // The embedding stays f32 and lm_head gets its own copy at the chosen
    // precision even when the checkpoint ties them; this gives the 3,648 MiB
    // of base weights `--memory-report` shows for SmolLM2-1.7B. Norms are
    // under 1 MiB and left out.
    let weights = shape.decoder_weights() as f64 * precision.decoder_bytes_per_weight()
        + vocab * h * (4.0 + precision.lm_head_bytes_per_weight());

    let q_width = (shape.n_heads * shape.head_dim) as f64;
    let kv_width = (shape.n_kv_heads * shape.head_dim) as f64;
    let adapter_weights =
        layers * config.lora_rank as f64 * (2.0 * (h + q_width) + 2.0 * (h + kv_width));
    // f32 weight, gradient and the two AdamW moments.
    let adapters = adapter_weights * 4.0 * 4.0;

    let floats_per_block = ACTIVATION_FLOATS_PER_HIDDEN * h
        + ACTIVATION_FLOATS_PER_FFN * shape.ffn as f64
        + ACTIVATION_FLOATS_PER_ATTENTION_ROW * (shape.n_heads * config.seq_len) as f64;
    let activations = match config.checkpointing {
        Checkpointing::Off => tokens * layers * floats_per_block * 4.0,
        // Every block's input, plus one block's activations while it is
        // recomputed.
        Checkpointing::Blocks => tokens * (layers * h + floats_per_block) * 4.0,
    };
    let logits = LOGIT_FLOATS_PER_VOCAB * tokens * vocab * 4.0;

    let parts = [
        ceil_mib(weights),
        ceil_mib(adapters),
        ceil_mib(activations),
        ceil_mib(logits),
        OVERHEAD_MIB,
    ];
    Estimate {
        weights_mib: parts[0],
        adapters_mib: parts[1],
        activations_mib: parts[2],
        logits_mib: parts[3],
        overhead_mib: parts[4],
        peak_mib: parts.iter().sum(),
    }
}

/// Sequence lengths the chooser tries.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SeqLens {
    /// 256, then 128.
    Default,
    /// Only this one.
    Fixed(NonZeroUsize),
}

/// A chosen configuration and whether it fits the budget.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    /// The configuration.
    pub config: TrainConfig,
    /// Its estimated memory.
    pub estimate: Estimate,
    /// Whether the estimate plus the safety margin fits the budget.
    pub fits: bool,
}

/// The first configuration that fits `budget_mib` with the safety margin:
/// no checkpointing before checkpointing, then larger batch, then longer
/// sequence. When none fits, the smallest one, with `fits` false.
pub fn choose(
    shape: &ModelShape,
    budget_mib: u64,
    precision: Precision,
    lora_rank: usize,
    seq_lens: SeqLens,
) -> Plan {
    let fixed;
    let seq_lens: &[usize] = match seq_lens {
        SeqLens::Default => &SEQ_LENS,
        SeqLens::Fixed(n) => {
            fixed = [n.get()];
            &fixed
        }
    };
    let usable_mib = budget_mib as f64 * (1.0 - SAFETY_MARGIN);
    let plan_for = |checkpointing, batch_size, seq_len| {
        let config = TrainConfig {
            batch_size,
            seq_len,
            lora_rank,
            precision,
            checkpointing,
        };
        let estimate = estimate_peak_mib(shape, &config);
        Plan {
            config,
            estimate,
            fits: estimate.peak_mib as f64 <= usable_mib,
        }
    };
    for checkpointing in [Checkpointing::Off, Checkpointing::Blocks] {
        for batch_size in BATCH_SIZES {
            for &seq_len in seq_lens {
                let plan = plan_for(checkpointing, batch_size, seq_len);
                if plan.fits {
                    return plan;
                }
            }
        }
    }
    let shortest = seq_lens.iter().copied().fold(usize::MAX, usize::min);
    plan_for(Checkpointing::Blocks, 1, shortest)
}

/// Where the memory budget came from.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BudgetSource {
    /// Derived from what the machine reports (system RAM of a unified GPU).
    Measured,
    /// A conservative default, not measured.
    Assumed,
    /// Given on the command line.
    User,
}

impl BudgetSource {
    fn as_str(self) -> &'static str {
        match self {
            Self::Measured => "measured",
            Self::Assumed => "assumed",
            Self::User => "user",
        }
    }
}

/// GPU memory a plan may use.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Budget {
    /// Size in MiB.
    pub mib: u64,
    /// Where it came from.
    pub source: BudgetSource,
    /// How it was worked out, for the output.
    pub reason: String,
}

/// The budget from `--memory-mib`, `--vram-mib`, the system RAM of a GPU
/// that shares it, or else an assumed default, in that order.
pub fn budget(
    args: &PlanArgs,
    adapter: Option<&AdapterReport>,
    total_ram_mib: Result<u64, String>,
) -> Budget {
    let user = |mib, flag: &str| Budget {
        mib,
        source: BudgetSource::User,
        reason: format!("set with {flag}"),
    };
    if let Some(mib) = args.memory_mib {
        return user(mib, "--memory-mib");
    }
    if let Some(mib) = args.vram_mib {
        return user(mib, "--vram-mib");
    }
    let assumed = |why: String| Budget {
        mib: ASSUMED_BUDGET_MIB,
        source: BudgetSource::Assumed,
        reason: format!("assumed, not measured: {why}; pass --vram-mib or --memory-mib"),
    };
    match adapter {
        // Apple silicon GPUs share system RAM whatever device type Metal reports.
        Some(a) if a.device_type == "IntegratedGpu" || a.vendor == "Apple" => match total_ram_mib {
            Ok(ram) => Budget {
                mib: ram * UNIFIED_RAM_PERCENT / 100,
                source: BudgetSource::Measured,
                reason: format!(
                    "{UNIFIED_RAM_PERCENT}% of {ram} MiB system RAM, which this GPU shares"
                ),
            },
            Err(e) => assumed(format!("cannot read system RAM ({e})")),
        },
        Some(_) => assumed("wgpu does not report the memory of a discrete GPU".into()),
        None => assumed("no GPU adapter found".into()),
    }
}

/// `MemTotal` from the text of /proc/meminfo, in MiB.
///
/// # Errors
///
/// When there is no `MemTotal` line or its value is not a number.
pub fn parse_meminfo_total_mib(meminfo: &str) -> Result<u64, String> {
    let line = meminfo
        .lines()
        .find(|l| l.starts_with("MemTotal:"))
        .ok_or("no MemTotal line in /proc/meminfo")?;
    let kib: u64 = line
        .split_whitespace()
        .nth(1)
        .unwrap_or("")
        .parse()
        .map_err(|e| format!("/proc/meminfo {line:?}: {e}"))?;
    Ok(kib / 1024)
}

fn total_ram_mib() -> Result<u64, String> {
    if cfg!(target_os = "linux") {
        let meminfo =
            std::fs::read_to_string("/proc/meminfo").map_err(|e| format!("/proc/meminfo: {e}"))?;
        parse_meminfo_total_mib(&meminfo)
    } else if cfg!(target_os = "macos") {
        let out = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .map_err(|e| format!("sysctl: {e}"))?;
        let text = String::from_utf8_lossy(&out.stdout);
        let bytes: u64 = text
            .trim()
            .parse()
            .map_err(|e| format!("sysctl hw.memsize {:?}: {e}", text.trim()))?;
        Ok(bytes / (1024 * 1024))
    } else {
        Err(format!("not implemented on {}", std::env::consts::OS))
    }
}

/// The `ferrotensor train` line for a plan, keeping 8 sequences per step.
pub fn train_command(model: &str, config: &TrainConfig) -> String {
    let grad_accum = (SEQUENCES_PER_STEP / config.batch_size).max(1);
    let checkpoint = match config.checkpointing {
        Checkpointing::Off => "",
        Checkpointing::Blocks => " --checkpoint-blocks",
    };
    format!(
        "ferrotensor train --model {model} --data train.jsonl --batch-size {} \
         --grad-accum {grad_accum} --seq-len {} --lora-rank {}{checkpoint}{}",
        config.batch_size,
        config.seq_len,
        config.lora_rank,
        config.precision.train_flag()
    )
}

fn plan_json(shape: &ModelShape, model_arg: &str, plan: &Plan) -> Map<String, Value> {
    let c = &plan.config;
    let mut fields = Map::new();
    fields.insert("model".into(), json!(shape.name));
    fields.insert("parameters".into(), json!(shape.parameters()));
    fields.insert(
        "chosen".into(),
        json!({
            "batch_size": c.batch_size,
            "seq_len": c.seq_len,
            "rank": c.lora_rank,
            "precision": c.precision.as_str(),
            "checkpoint_blocks": c.checkpointing == Checkpointing::Blocks,
        }),
    );
    fields.insert("estimated_peak_mib".into(), json!(plan.estimate.peak_mib));
    fields.insert("margin".into(), json!(SAFETY_MARGIN));
    fields.insert("fits".into(), json!(plan.fits));
    fields.insert("train_command".into(), json!(train_command(model_arg, c)));
    fields
}

fn describe(config: &TrainConfig) -> String {
    let checkpoint = match config.checkpointing {
        Checkpointing::Off => "no checkpointing",
        Checkpointing::Blocks => "--checkpoint-blocks",
    };
    format!(
        "batch {}, seq {}, rank {}, {} weights, {checkpoint}",
        config.batch_size,
        config.seq_len,
        config.lora_rank,
        config.precision.as_str()
    )
}

fn millions(parameters: u64) -> String {
    format!("{:.1}M", parameters as f64 / 1e6)
}

/// Whether `plan` found a configuration that fits.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// A plan fits (exit code 0).
    Fits,
    /// Nothing fits (exit code 2).
    DoesNotFit,
}

/// What one run of `plan` prints besides the adapter and budget.
struct Report {
    outcome: Outcome,
    fields: Map<String, Value>,
    lines: Vec<String>,
    notes: Vec<String>,
}

/// Runs `ferrotensor plan`.
///
/// # Errors
///
/// When no GPU adapter is found and no budget is given, or `--model` has
/// no readable config.json.
pub fn run(args: PlanArgs) -> Result<Outcome, Box<dyn std::error::Error>> {
    let adapters = adapters::list_adapters();
    let adapter = adapters.iter().find(|a| a.selected);
    if adapter.is_none() && args.memory_mib.is_none() && args.vram_mib.is_none() {
        return Err("no GPU adapter found; pass --memory-mib to plan for another machine".into());
    }
    let budget = budget(&args, adapter, total_ram_mib());
    let seq_lens = args.seq_len.map_or(SeqLens::Default, SeqLens::Fixed);
    let mut report = match &args.model {
        Some(dir) => plan_model(&args, dir, budget.mib, seq_lens)?,
        None => plan_documented(&args, budget.mib, seq_lens),
    };
    report.notes.push(
        "the estimate is calibrated on 8 measured peaks (SmolLM2 135M, 360M and 1.7B, \
         f16 weights, seq 256, one A10G); its worst error there is 13.3%"
            .into(),
    );
    report
        .notes
        .push("replace train.jsonl in the train command with your dataset".into());
    if budget.source == BudgetSource::Assumed {
        report
            .notes
            .push(format!("budget {} MiB is {}", budget.mib, budget.reason));
    }
    let (adapter_name, backend, device_type) = adapter.map_or(("none", "none", "none"), |a| {
        (a.name.as_str(), a.backend.as_str(), a.device_type.as_str())
    });

    if args.json {
        let mut out = Map::new();
        out.insert("adapter".into(), json!(adapter_name));
        out.insert("backend".into(), json!(backend));
        out.insert("device_type".into(), json!(device_type));
        out.insert("budget_mib".into(), json!(budget.mib));
        out.insert("budget_source".into(), json!(budget.source.as_str()));
        out.extend(report.fields);
        out.insert("notes".into(), json!(report.notes));
        println!("{}", serde_json::to_string_pretty(&Value::Object(out))?);
    } else {
        println!("adapter   {adapter_name} ({device_type}, {backend})");
        println!(
            "budget    {} MiB ({}: {})",
            budget.mib,
            budget.source.as_str(),
            budget.reason
        );
        for line in &report.lines {
            println!("{line}");
        }
        for note in &report.notes {
            println!("note: {note}");
        }
    }
    Ok(report.outcome)
}

fn plan_model(
    args: &PlanArgs,
    dir: &Path,
    budget_mib: u64,
    seq_lens: SeqLens,
) -> Result<Report, Box<dyn std::error::Error>> {
    let config_path = dir.join("config.json");
    let config = ModelLoader::load_config(&config_path)
        .map_err(|e| format!("cannot read {}: {e}", config_path.display()))?;
    let model_arg = dir.display().to_string();
    let shape = ModelShape::from_config(model_arg.clone(), &config);
    let plan = choose(&shape, budget_mib, args.weights, args.lora_rank, seq_lens);
    let mut notes = Vec::new();
    if !config.layer_types.is_empty() || config.model_type.as_deref() == Some("lfm2") {
        notes.push("LFM2 blocks are estimated as Llama blocks; not calibrated".into());
    }
    if plan.config.checkpointing == Checkpointing::Blocks {
        notes.push(
            "--checkpoint-blocks took 19% to 39% more time per step in the measured runs".into(),
        );
    }
    if !plan.fits && args.weights == Precision::F16 {
        let q4 = choose(&shape, budget_mib, Precision::Q4, args.lora_rank, seq_lens);
        if q4.fits {
            notes.push(format!(
                "with --weights q4 (experimental, worse held-out loss) this fits: {}, {} MiB",
                describe(&q4.config),
                q4.estimate.peak_mib
            ));
        }
    }
    let e = &plan.estimate;
    let verdict = if plan.fits { "fits" } else { "does not fit" };
    let lines = vec![
        format!(
            "model     {} ({} parameters)",
            shape.name,
            millions(shape.parameters())
        ),
        format!("plan      {} ({verdict})", describe(&plan.config)),
        format!(
            "estimate  {} MiB peak: weights {}, adapters {}, activations {}, logits {}, \
             overhead {}; a plan keeps {:.0}% of the budget free",
            e.peak_mib,
            e.weights_mib,
            e.adapters_mib,
            e.activations_mib,
            e.logits_mib,
            e.overhead_mib,
            SAFETY_MARGIN * 100.0
        ),
        format!("command   {}", train_command(&model_arg, &plan.config)),
    ];
    Ok(Report {
        outcome: if plan.fits {
            Outcome::Fits
        } else {
            Outcome::DoesNotFit
        },
        fields: plan_json(&shape, &model_arg, &plan),
        lines,
        notes,
    })
}

fn plan_documented(args: &PlanArgs, budget_mib: u64, seq_lens: SeqLens) -> Report {
    let mut models = Vec::new();
    let mut lines = vec![format!(
        "\n{:<14} {:>10}  {:<58} {:>10}  fits",
        "model", "parameters", "plan", "estimate"
    )];
    let mut any_fits = false;
    for shape in ModelShape::documented() {
        let plan = choose(&shape, budget_mib, args.weights, args.lora_rank, seq_lens);
        any_fits |= plan.fits;
        let model_arg = format!("path/to/{}", shape.name);
        models.push(Value::Object(plan_json(&shape, &model_arg, &plan)));
        lines.push(format!(
            "{:<14} {:>10}  {:<58} {:>6} MiB  {}",
            shape.name,
            millions(shape.parameters()),
            describe(&plan.config),
            plan.estimate.peak_mib,
            if plan.fits { "yes" } else { "no" }
        ));
    }
    let mut fields = Map::new();
    fields.insert("models".into(), Value::Array(models));
    Report {
        outcome: if any_fits {
            Outcome::Fits
        } else {
            Outcome::DoesNotFit
        },
        fields,
        lines,
        notes: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn smollm2(name: &str) -> ModelShape {
        ModelShape::documented()
            .into_iter()
            .find(|s| s.name == name)
            .expect("documented shape")
    }

    fn config(batch_size: usize, checkpointing: Checkpointing) -> TrainConfig {
        TrainConfig {
            batch_size,
            seq_len: 256,
            lora_rank: 8,
            precision: Precision::F16,
            checkpointing,
        }
    }

    /// Peak GPU memory (MiB) measured with `nvidia-smi` on an A10G, f16
    /// weights, seq 256, rank 8: docs/examples/lora-sft.md, "Confirmation
    /// run of merged main". The last column is the estimate, worked out by
    /// hand from the same formula.
    const MEASURED: [(&str, usize, Checkpointing, u64, u64); 8] = [
        ("SmolLM2-135M", 1, Checkpointing::Off, 2306, 2569),
        ("SmolLM2-135M", 1, Checkpointing::Blocks, 1890, 1868),
        ("SmolLM2-360M", 1, Checkpointing::Off, 3579, 3661),
        ("SmolLM2-360M", 1, Checkpointing::Blocks, 2783, 2413),
        ("SmolLM2-1.7B", 1, Checkpointing::Off, 8142, 7638),
        ("SmolLM2-1.7B", 1, Checkpointing::Blocks, 5313, 5294),
        ("SmolLM2-135M", 8, Checkpointing::Off, 9441, 8438),
        ("SmolLM2-135M", 8, Checkpointing::Blocks, 2767, 2831),
    ];

    #[test]
    fn estimate_is_within_15_percent_of_every_measured_peak() {
        let mut worst = 0.0_f64;
        for (name, batch, checkpointing, measured, expected) in MEASURED {
            let estimate = estimate_peak_mib(&smollm2(name), &config(batch, checkpointing));
            assert_eq!(
                estimate.peak_mib, expected,
                "{name} batch {batch} {checkpointing:?}"
            );
            let error = (estimate.peak_mib as f64 - measured as f64).abs() / measured as f64;
            assert!(
                error <= 0.15,
                "{name} batch {batch} {checkpointing:?}: {error}"
            );
            worst = worst.max(error);
        }
        // 360M with --checkpoint-blocks: 2,413 against 2,783 MiB.
        assert!((worst - 0.1330).abs() < 1e-3, "worst error {worst}");
        assert!(worst < SAFETY_MARGIN);
    }

    #[test]
    fn base_weights_match_the_memory_report_for_smollm2_1_7b() {
        // `train --memory-report` shows 3,648 MiB of frozen base weights and
        // under 50 MiB for adapters, gradients and AdamW state.
        let estimate = estimate_peak_mib(&smollm2("SmolLM2-1.7B"), &config(1, Checkpointing::Off));
        assert_eq!(estimate.weights_mib, 3648);
        assert_eq!(estimate.adapters_mib, 48);
    }

    #[test]
    fn weight_precision_changes_only_the_weights() {
        let shape = smollm2("SmolLM2-1.7B");
        let mut c = config(1, Checkpointing::Off);
        c.precision = Precision::F32;
        assert_eq!(estimate_peak_mib(&shape, &c).weights_mib, 6912);
        c.precision = Precision::Q4;
        assert_eq!(estimate_peak_mib(&shape, &c).weights_mib, 1440);
        assert_eq!(estimate_peak_mib(&shape, &c).activations_mib, 2496);
    }

    #[test]
    fn parameter_counts_match_the_published_models() {
        assert_eq!(smollm2("SmolLM2-135M").parameters(), 134_515_008);
        assert_eq!(smollm2("SmolLM2-360M").parameters(), 361_821_120);
        assert_eq!(smollm2("SmolLM2-1.7B").parameters(), 1_711_376_384);
    }

    #[test]
    fn chooser_prefers_no_checkpointing_then_batch_then_sequence() {
        let p = choose(
            &smollm2("SmolLM2-135M"),
            8000,
            Precision::F16,
            8,
            SeqLens::Default,
        );
        assert_eq!(
            (
                p.config.checkpointing,
                p.config.batch_size,
                p.config.seq_len,
                p.estimate.peak_mib,
                p.fits
            ),
            (Checkpointing::Off, 8, 128, 4679, true)
        );
    }

    #[test]
    fn chooser_falls_back_to_checkpointing_on_a_small_budget() {
        let p = choose(
            &smollm2("SmolLM2-360M"),
            3000,
            Precision::F16,
            8,
            SeqLens::Default,
        );
        assert_eq!(
            (
                p.config.checkpointing,
                p.config.batch_size,
                p.config.seq_len,
                p.estimate.peak_mib,
                p.fits
            ),
            (Checkpointing::Blocks, 2, 128, 2407, true)
        );
    }

    #[test]
    fn chooser_keeps_a_fixed_sequence_length() {
        let seq = SeqLens::Fixed(NonZeroUsize::new(256).expect("non-zero"));
        let p = choose(&smollm2("SmolLM2-1.7B"), 8000, Precision::F16, 8, seq);
        assert_eq!(
            (
                p.config.checkpointing,
                p.config.batch_size,
                p.config.seq_len,
                p.estimate.peak_mib
            ),
            (Checkpointing::Blocks, 4, 256, 6038)
        );
    }

    #[test]
    fn chooser_reports_the_smallest_plan_when_nothing_fits() {
        let p = choose(
            &smollm2("SmolLM2-135M"),
            500,
            Precision::F16,
            8,
            SeqLens::Default,
        );
        assert_eq!(
            (
                p.config.checkpointing,
                p.config.batch_size,
                p.config.seq_len,
                p.estimate.peak_mib,
                p.fits
            ),
            (Checkpointing::Blocks, 1, 128, 1798, false)
        );
        // 1.7B on 6,000 MiB: f16 needs 5,164 MiB plus the margin; 4-bit fits.
        let shape = smollm2("SmolLM2-1.7B");
        assert!(!choose(&shape, 6000, Precision::F16, 8, SeqLens::Default).fits);
        let q4 = choose(&shape, 6000, Precision::Q4, 8, SeqLens::Default);
        assert_eq!(
            (q4.config.batch_size, q4.estimate.peak_mib, q4.fits),
            (1, 3990, true)
        );
    }

    fn args(memory_mib: Option<u64>, vram_mib: Option<u64>) -> PlanArgs {
        PlanArgs {
            model: None,
            memory_mib,
            vram_mib,
            weights: Precision::F16,
            lora_rank: 8,
            seq_len: None,
            json: false,
        }
    }

    fn adapter(vendor: &str, device_type: &str) -> AdapterReport {
        AdapterReport {
            name: "test".into(),
            vendor: vendor.into(),
            device_type: device_type.into(),
            backend: "Vulkan".into(),
            driver: String::new(),
            max_buffer_size: 1 << 30,
            max_storage_buffer_binding_size: 1 << 27,
            max_compute_invocations_per_workgroup: 256,
            max_compute_workgroup_storage_size: 16384,
            shader_f16: false,
            selected: true,
        }
    }

    #[test]
    fn budget_flags_win_over_the_adapter() {
        let apple = adapter("Apple", "DiscreteGpu");
        let b = budget(&args(Some(3000), Some(9000)), Some(&apple), Ok(8192));
        assert_eq!((b.mib, b.source), (3000, BudgetSource::User));
        let b = budget(&args(None, Some(9000)), Some(&apple), Ok(8192));
        assert_eq!((b.mib, b.source), (9000, BudgetSource::User));
    }

    #[test]
    fn unified_memory_budgets_a_share_of_system_ram() {
        let b = budget(
            &args(None, None),
            Some(&adapter("Apple", "DiscreteGpu")),
            Ok(8192),
        );
        assert_eq!((b.mib, b.source), (4915, BudgetSource::Measured));
        let b = budget(
            &args(None, None),
            Some(&adapter("Intel", "IntegratedGpu")),
            Ok(16384),
        );
        assert_eq!((b.mib, b.source), (9830, BudgetSource::Measured));
        let b = budget(
            &args(None, None),
            Some(&adapter("Intel", "IntegratedGpu")),
            Err("no /proc".into()),
        );
        assert_eq!((b.mib, b.source), (4096, BudgetSource::Assumed));
    }

    #[test]
    fn discrete_gpu_budget_is_assumed() {
        let b = budget(
            &args(None, None),
            Some(&adapter("NVIDIA", "DiscreteGpu")),
            Ok(65536),
        );
        assert_eq!((b.mib, b.source), (4096, BudgetSource::Assumed));
        assert!(
            b.reason.starts_with("assumed, not measured"),
            "{}",
            b.reason
        );
    }

    #[test]
    fn meminfo_total_is_read_in_mib() {
        let meminfo = "MemTotal:       16303624 kB\nMemFree:         1234 kB\n";
        assert_eq!(parse_meminfo_total_mib(meminfo), Ok(15921));
        assert_eq!(
            parse_meminfo_total_mib("MemFree: 1 kB\n"),
            Err("no MemTotal line in /proc/meminfo".to_string())
        );
    }

    #[test]
    fn train_command_uses_the_train_flags() {
        let mut c = config(2, Checkpointing::Blocks);
        c.precision = Precision::Q4;
        assert_eq!(
            train_command("models/m", &c),
            "ferrotensor train --model models/m --data train.jsonl --batch-size 2 \
             --grad-accum 4 --seq-len 256 --lora-rank 8 --checkpoint-blocks --q4-weights"
        );
    }
}
