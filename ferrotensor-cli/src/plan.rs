//! `ferrotensor plan`: the largest LoRA training configuration that is
//! expected to fit this machine's GPU memory, from a first-principles
//! estimate calibrated on measured peaks.

use crate::Exit;
use clap::{Args, ValueEnum};
use ferrotensor_backend::adapters::{self, AdapterReport};
use ferrotensor_loader::{Lfm2Config, ModelLoader};
use serde_json::{json, Map, Value};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

const MIB: f64 = 1024.0 * 1024.0;

/// Fixed GPU memory every run holds: the driver context, the buffer pool's
/// idle buffers (cap 512 MiB) and allocator slack. Fitted to the measured
/// peaks.
const OVERHEAD_MIB: u64 = 1600;
/// Measured activation bytes over 4 bytes per value that
/// [`estimate_peak_mib`] counts. Fitted to the measured peaks.
const ACTIVATION_SCALE: f64 = 0.92;
/// Values per token and block, per unit of key/value width, kept beyond
/// the counted ones. Ferrotensor's own count of its buffers shows
/// SmolLM2-1.7B, the one measured model whose keys and values are as wide
/// as its queries, holding about 1.25 times the counted activations and
/// the other models 0.8 to 0.9 times; which buffers these are is not known.
const UNCOUNTED_PER_KV_WIDTH: f64 = 13.0;
/// Extra activations once a run that accumulates several forward passes
/// per optimizer step reaches its steady peak. Each pass is padded to its
/// longest sequence, so consecutive passes ask for buffers of different
/// sizes, and nothing waits for the GPU between them; the peak keeps
/// growing for more than 30 steps. Fitted to the 250-step peaks.
const ACCUMULATION_OVERLAP: f64 = 0.20;
/// The logits and their gradient, both f32 and vocab-sized per token.
const LOGIT_FLOATS_PER_VOCAB: f64 = 2.0;
/// Fraction of the budget a plan leaves free; above the estimator's largest
/// underestimate of the measured peaks (4.6%).
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
const CHECKPOINTINGS: [Checkpointing; 2] = [Checkpointing::Off, Checkpointing::Blocks];

/// Options for `ferrotensor plan`.
#[derive(Args, Debug)]
pub struct PlanArgs {
    /// Model directory or cached Hugging Face model id (only config.json is
    /// read). Without it, the documented SmolLM2 sizes are listed.
    #[arg(long, value_name = "DIR_OR_ID")]
    pub model: Option<PathBuf>,
    #[command(flatten)]
    pub budget: BudgetArgs,
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

/// Flags that set the GPU memory budget.
#[derive(Args, Debug)]
pub struct BudgetArgs {
    /// GPU memory budget in MiB; overrides everything else.
    #[arg(long, value_name = "MIB")]
    pub memory_mib: Option<u64>,
    /// Memory of a discrete GPU in MiB (wgpu cannot read it).
    #[arg(long, value_name = "MIB")]
    pub vram_mib: Option<u64>,
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
    /// The name `plan` prints.
    pub fn as_str(self) -> &'static str {
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

    /// The frozen embedding table is stored at the same type as `lm_head`,
    /// since a LoRA run looks it up from a host copy only.
    fn embedding_bytes_per_weight(self) -> f64 {
        self.lm_head_bytes_per_weight()
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
    /// Forward passes per optimizer step.
    pub grad_accum: usize,
    /// Tokens per sequence, the cut `--seq-len` makes.
    pub seq_len: usize,
    /// Rows per sequence the memory estimate plans for: `seq_len` when the
    /// data is not known, else what [`LengthProfile::planning_len`] gives.
    pub padded_len: usize,
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
    /// Frozen base weights (embedding and lm_head at the chosen precision,
    /// f32 only with `--f32-weights`; decoder blocks at the chosen precision).
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
    estimate_with_embedding(shape, config, config.precision.embedding_bytes_per_weight())
}

/// [`estimate_peak_mib`] with the frozen embedding table at
/// `embedding_bytes` per weight.
fn estimate_with_embedding(
    shape: &ModelShape,
    config: &TrainConfig,
    embedding_bytes: f64,
) -> Estimate {
    let h = shape.hidden as f64;
    let vocab = shape.vocab as f64;
    let layers = shape.n_layers as f64;
    let tokens = (config.batch_size * config.padded_len) as f64;
    let precision = config.precision;

    // The embedding is stored at the lm_head precision and lm_head gets its
    // own copy even when the checkpoint ties them. Norms are under 1 MiB and
    // left out. The frozen table was f32 before it was stored at the weight
    // type, which gave the 3,648 MiB of base weights `--memory-report` showed
    // for SmolLM2-1.7B; the f16 table takes 192 MiB less.
    let weights = shape.decoder_weights() as f64 * precision.decoder_bytes_per_weight()
        + vocab * h * (embedding_bytes + precision.lm_head_bytes_per_weight());

    let q_width = (shape.n_heads * shape.head_dim) as f64;
    let kv_width = (shape.n_kv_heads * shape.head_dim) as f64;
    let adapter_weights =
        layers * config.lora_rank as f64 * (2.0 * (h + q_width) + 2.0 * (h + kv_width));
    // f32 weight, gradient and the two AdamW moments.
    let adapters = adapter_weights * 4.0 * 4.0;

    // Each op output stays alive until backward, held by the node that
    // reads it. Per token: the two norms, the two residual sums and the down
    // projection; q, k, v and o each as base output, LoRA output, scaled
    // LoRA output and their sum; RoPE of q and k; the attention output
    // (flash attention keeps no scores); gate, up and their product.
    let values_per_block = 9.0 * h
        + 6.0 * q_width
        + (9.0 + UNCOUNTED_PER_KV_WIDTH) * kv_width
        + 3.0 * shape.ffn as f64;
    let values_per_token = match config.checkpointing {
        Checkpointing::Off => layers * values_per_block,
        // Every block's input, plus one block's activations while it is
        // recomputed.
        Checkpointing::Blocks => layers * h + values_per_block,
    };
    let overlap = if config.grad_accum > 1 {
        1.0 + ACCUMULATION_OVERLAP
    } else {
        1.0
    };
    let activations = tokens * values_per_token * 4.0 * ACTIVATION_SCALE * overlap;
    // The trainer runs lm_head and the loss over row chunks when the logits
    // and their gradient would pass a size it names, so only one chunk of
    // them exists at a time. The planner counts every padded token as a row
    // of the head, which is at least the rows the trainer gathers.
    let logit_rows = ferrotensor_trainer::head_chunk_rows(tokens as usize, shape.vocab)
        .map_or(tokens, |chunk| chunk as f64);
    let logits = LOGIT_FLOATS_PER_VOCAB * logit_rows * vocab * 4.0;

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

/// Share of a dataset's sequences that fit the rows the estimate plans for.
/// A micro-batch is padded to its longest sequence, so the longest ones set
/// the peak. The margin is judged at this length; the longest sequence of
/// the data must also fit the budget without the margin (see [`Plan::fits`]).
const PLANNING_PERCENT: usize = 99;

/// Token counts of a dataset's examples, as the planner needs them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LengthProfile {
    /// Ascending.
    lengths: Vec<usize>,
}

impl LengthProfile {
    /// The profile of `lengths`; `None` when there are none.
    pub fn new(mut lengths: Vec<usize>) -> Option<Self> {
        if lengths.is_empty() {
            return None;
        }
        lengths.sort_unstable();
        Some(Self { lengths })
    }

    /// Number of examples.
    pub fn examples(&self) -> usize {
        self.lengths.len()
    }

    /// Rows per sequence to plan memory for when examples are cut at `cap`
    /// tokens: the [`PLANNING_PERCENT`]th percentile of the cut lengths, nearest rank,
    /// and at least 1.
    pub fn planning_len(&self, cap: usize) -> usize {
        let rank = (PLANNING_PERCENT * self.lengths.len()).div_ceil(100);
        let index = rank.clamp(1, self.lengths.len()) - 1;
        self.lengths[index].min(cap).max(1)
    }

    /// Rows per sequence of the longest batch the data can make when examples
    /// are cut at `cap` tokens, and at least 1.
    pub fn longest_len(&self, cap: usize) -> usize {
        self.lengths.last().copied().unwrap_or(1).min(cap).max(1)
    }
}

/// Values the chooser keeps instead of trying its own; `None` tries them.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Pinned {
    /// Tokens per sequence [tried: 256, then 128].
    pub seq_len: Option<NonZeroUsize>,
    /// Sequences per forward pass [tried: 8, 4, 2, 1].
    pub batch_size: Option<NonZeroUsize>,
    /// Block checkpointing [tried: off, then on].
    pub checkpointing: Option<Checkpointing>,
}

/// A chosen configuration and whether it fits the budget.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    /// The configuration.
    pub config: TrainConfig,
    /// Its estimated memory.
    pub estimate: Estimate,
    /// Whether the estimate plus the safety margin fits the budget, and the
    /// estimate for a batch of the data's longest sequences fits it without
    /// the margin.
    pub fits: bool,
    /// The estimate for a batch whose sequences are all as long as the longest
    /// one of the data (the same as `estimate` without data).
    pub longest_peak_mib: u64,
}

/// The first configuration that fits `budget_mib` with the safety margin:
/// longer sequence first (cutting examples changes what is trained, batch
/// size only changes speed), then no checkpointing before checkpointing,
/// then larger batch, trying only the `pinned` value where there is one.
/// When none fits, the smallest one, with `fits` false.
pub fn choose(
    shape: &ModelShape,
    budget_mib: u64,
    precision: Precision,
    lora_rank: usize,
    pinned: Pinned,
) -> Plan {
    choose_for(shape, budget_mib, precision, lora_rank, pinned, None)
}

/// [`choose`] with the memory estimated for the rows `lengths` says the
/// data pads to, instead of `--seq-len` rows for every sequence.
pub fn choose_for(
    shape: &ModelShape,
    budget_mib: u64,
    precision: Precision,
    lora_rank: usize,
    pinned: Pinned,
    lengths: Option<&LengthProfile>,
) -> Plan {
    let seq_pin = pinned.seq_len.map(NonZeroUsize::get);
    let seq_lens: &[usize] = match &seq_pin {
        Some(n) => std::slice::from_ref(n),
        None => &SEQ_LENS,
    };
    let batch_pin = pinned.batch_size.map(NonZeroUsize::get);
    let batch_sizes: &[usize] = match &batch_pin {
        Some(n) => std::slice::from_ref(n),
        None => &BATCH_SIZES,
    };
    let checkpointings: &[Checkpointing] = match &pinned.checkpointing {
        Some(c) => std::slice::from_ref(c),
        None => &CHECKPOINTINGS,
    };
    let usable_mib = budget_mib as f64 * (1.0 - SAFETY_MARGIN);
    let plan_for = |checkpointing, batch_size, seq_len| {
        let config = TrainConfig {
            batch_size,
            grad_accum: grad_accum_for(batch_size),
            seq_len,
            padded_len: lengths.map_or(seq_len, |profile| profile.planning_len(seq_len)),
            lora_rank,
            precision,
            checkpointing,
        };
        let estimate = estimate_peak_mib(shape, &config);
        // The margin covers the rare batch that is longer than the planning
        // length, but not a data set whose longest batch would overflow the
        // GPU: that one has to fit the budget itself.
        let longest = TrainConfig {
            padded_len: lengths.map_or(seq_len, |profile| profile.longest_len(seq_len)),
            ..config
        };
        let longest_peak_mib = estimate_peak_mib(shape, &longest)
            .peak_mib
            .max(estimate.peak_mib);
        Plan {
            config,
            estimate,
            fits: estimate.peak_mib as f64 <= usable_mib && longest_peak_mib <= budget_mib,
            longest_peak_mib,
        }
    };
    for &seq_len in seq_lens {
        for &checkpointing in checkpointings {
            for &batch_size in batch_sizes {
                let plan = plan_for(checkpointing, batch_size, seq_len);
                if plan.fits {
                    return plan;
                }
            }
        }
    }
    let shortest = seq_lens.iter().copied().fold(usize::MAX, usize::min);
    let smallest_batch = batch_sizes.iter().copied().fold(usize::MAX, usize::min);
    plan_for(
        pinned.checkpointing.unwrap_or(Checkpointing::Blocks),
        smallest_batch,
        shortest,
    )
}

/// Where the memory budget came from.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BudgetSource {
    /// Derived from what the machine reports (system RAM of a unified GPU,
    /// or nvidia-smi).
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
/// that shares it, `nvidia_smi_mib` for an NVIDIA GPU, or else an assumed
/// default, in that order.
pub fn budget(
    args: &BudgetArgs,
    adapter: Option<&AdapterReport>,
    total_ram_mib: Result<u64, String>,
    nvidia_smi_mib: impl FnOnce() -> Result<u64, String>,
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
        Some(a) if a.vendor == "NVIDIA" => match nvidia_smi_mib() {
            Ok(mib) => Budget {
                mib,
                source: BudgetSource::Measured,
                reason: "memory.total from nvidia-smi (the smallest GPU listed)".into(),
            },
            Err(e) => assumed(format!(
                "wgpu does not report the memory of a discrete GPU and nvidia-smi failed ({e})"
            )),
        },
        Some(_) => assumed("wgpu does not report the memory of a discrete GPU".into()),
        None => assumed("no GPU adapter found".into()),
    }
}

/// The smallest `memory.total` in the output of `nvidia-smi
/// --query-gpu=memory.total --format=csv,noheader,nounits`, in MiB.
///
/// # Errors
///
/// When the output lists no GPU or a line is not a number.
pub fn parse_nvidia_smi_total_mib(output: &str) -> Result<u64, String> {
    let mut smallest: Option<u64> = None;
    for line in output.lines().map(str::trim).filter(|l| !l.is_empty()) {
        let mib: u64 = line
            .parse()
            .map_err(|e| format!("nvidia-smi line {line:?}: {e}"))?;
        smallest = Some(smallest.map_or(mib, |s| s.min(mib)));
    }
    smallest.ok_or_else(|| "nvidia-smi listed no GPU".into())
}

fn nvidia_smi_total_mib() -> Result<u64, String> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output()
        .map_err(|e| format!("cannot run nvidia-smi: {e}"))?;
    if !out.status.success() {
        return Err(format!("nvidia-smi exited with {}", out.status));
    }
    parse_nvidia_smi_total_mib(&String::from_utf8_lossy(&out.stdout))
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

/// Batches per optimizer step that keep 8 sequences per step.
pub fn grad_accum_for(batch_size: usize) -> usize {
    (SEQUENCES_PER_STEP / batch_size).max(1)
}

/// The `ferrotensor train` line for a plan.
pub fn train_command(model: &str, config: &TrainConfig) -> String {
    let checkpoint = match config.checkpointing {
        Checkpointing::Off => "",
        Checkpointing::Blocks => " --checkpoint-blocks",
    };
    format!(
        "ferrotensor train --model {model} --data train.jsonl --batch-size {} \
         --grad-accum {} --seq-len {} --lora-rank {}{checkpoint}{}",
        config.batch_size,
        config.grad_accum,
        config.seq_len,
        config.lora_rank,
        config.precision.train_flag()
    )
}

/// The `chosen` object of `plan --json`, which `finetune --json` repeats.
pub fn chosen_json(config: &TrainConfig) -> Value {
    json!({
        "batch_size": config.batch_size,
        "seq_len": config.seq_len,
        "rank": config.lora_rank,
        "precision": config.precision.as_str(),
        "checkpoint_blocks": config.checkpointing == Checkpointing::Blocks,
    })
}

fn plan_json(shape: &ModelShape, model_arg: &str, plan: &Plan) -> Map<String, Value> {
    let mut fields = Map::new();
    fields.insert("model".into(), json!(shape.name));
    fields.insert("parameters".into(), json!(shape.parameters()));
    fields.insert("chosen".into(), chosen_json(&plan.config));
    fields.insert("estimated_peak_mib".into(), json!(plan.estimate.peak_mib));
    fields.insert("margin".into(), json!(SAFETY_MARGIN));
    fields.insert("fits".into(), json!(plan.fits));
    fields.insert(
        "train_command".into(),
        json!(train_command(model_arg, &plan.config)),
    );
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

/// The note for a plan below the preferred sequence length, which cuts data.
fn short_seq_note(model: &str, seq_len: usize) -> Option<String> {
    (seq_len < SEQ_LENS[0]).then(|| {
        format!(
            "{model}: seq {seq_len} cuts every example longer than {seq_len} tokens; \
             pass --seq-len N to plan and train to keep a length"
        )
    })
}

fn millions(parameters: u64) -> String {
    format!("{:.1}M", parameters as f64 / 1e6)
}

/// What one run of `plan` prints besides the adapter and budget.
struct Report {
    /// [`Exit::Success`] when a plan fits, [`Exit::Unmet`] when none does.
    exit: Exit,
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
pub fn run(args: PlanArgs) -> Result<Exit, Box<dyn std::error::Error>> {
    let (adapter, budget) = machine_budget(&args.budget)?;
    let pinned = Pinned {
        seq_len: args.seq_len,
        ..Pinned::default()
    };
    let mut report = match &args.model {
        Some(dir) => plan_model(dir, budget.mib, args.weights, args.lora_rank, pinned, None)?.1,
        None => plan_documented(&args, budget.mib, pinned),
    };
    push_common_notes(&mut report.notes, &budget);

    if args.json {
        let (adapter_name, backend, device_type) = adapter_fields(adapter.as_ref());
        let mut out = Map::new();
        out.insert("adapter".into(), json!(adapter_name));
        out.insert("backend".into(), json!(backend));
        out.insert("device_type".into(), json!(device_type));
        out.insert("budget_mib".into(), json!(budget.mib));
        out.insert("budget_source".into(), json!(budget.source.as_str()));
        out.extend(report.fields);
        out.insert("notes".into(), json!(report.notes));
        crate::print_json(&out)?;
    } else {
        for line in human_lines(adapter.as_ref(), &budget, &report) {
            println!("{line}");
        }
    }
    Ok(report.exit)
}

/// A plan for training one model, as `ferrotensor finetune` uses it.
pub struct TrainingPlan {
    /// The chosen configuration.
    pub plan: Plan,
    /// The budget it was chosen for, in MiB.
    pub budget_mib: u64,
    /// The smallest budget the configuration fits with the safety margin.
    pub needed_budget_mib: u64,
    /// What `plan --model` prints for it, line by line.
    pub lines: Vec<String>,
}

/// Plans training the model in `dir` at `precision` and `lora_rank`,
/// keeping the `pinned` values, with memory estimated for the rows `lengths`
/// says the data pads to (`--seq-len` rows when it is `None`).
///
/// # Errors
///
/// When no GPU adapter is found and no budget is given, or `dir` has no
/// readable config.json.
pub fn plan_training(
    flags: &BudgetArgs,
    dir: &Path,
    precision: Precision,
    lora_rank: usize,
    pinned: Pinned,
    lengths: Option<&LengthProfile>,
) -> Result<TrainingPlan, Box<dyn std::error::Error>> {
    let (adapter, budget) = machine_budget(flags)?;
    let (plan, mut report) = plan_model(dir, budget.mib, precision, lora_rank, pinned, lengths)?;
    push_common_notes(&mut report.notes, &budget);
    Ok(TrainingPlan {
        plan,
        budget_mib: budget.mib,
        needed_budget_mib: ((plan.estimate.peak_mib as f64 / (1.0 - SAFETY_MARGIN)).ceil() as u64)
            .max(plan.longest_peak_mib),
        lines: human_lines(adapter.as_ref(), &budget, &report),
    })
}

/// The selected GPU adapter and the memory budget for it.
fn machine_budget(
    flags: &BudgetArgs,
) -> Result<(Option<AdapterReport>, Budget), Box<dyn std::error::Error>> {
    let adapter = adapters::list_adapters().into_iter().find(|a| a.selected);
    if adapter.is_none() && flags.memory_mib.is_none() && flags.vram_mib.is_none() {
        return Err(crate::unmet(
            "no GPU adapter found; pass --memory-mib to plan for another machine",
        ));
    }
    let budget = budget(
        flags,
        adapter.as_ref(),
        total_ram_mib(),
        nvidia_smi_total_mib,
    );
    Ok((adapter, budget))
}

fn push_common_notes(notes: &mut Vec<String>, budget: &Budget) {
    notes.push(
        "the estimate is fitted to 24 measured peaks (SmolLM2 135M, 360M and 1.7B, \
         Qwen2.5-0.5B, f16 weights, 8 sequences per step, an A10G and a T4); it is at \
         most 4.6% below them, at most 12% above the 250-step and single-pass peaks, \
         and up to 35% above the shorter runs that accumulate passes, which have not \
         reached their peak; a head with more than 1 GiB of logits runs in chunks of \
         512 rows and is counted at that size, which no run has measured"
            .into(),
    );
    notes.push("replace train.jsonl in the train command with your dataset".into());
    if budget.source == BudgetSource::Assumed {
        notes.push(format!("budget {} MiB is {}", budget.mib, budget.reason));
    }
}

/// Name, backend and device type of the adapter, or "none".
fn adapter_fields(adapter: Option<&AdapterReport>) -> (&str, &str, &str) {
    adapter.map_or(("none", "none", "none"), |a| {
        (a.name.as_str(), a.backend.as_str(), a.device_type.as_str())
    })
}

/// The lines `plan` prints without `--json`.
fn human_lines(adapter: Option<&AdapterReport>, budget: &Budget, report: &Report) -> Vec<String> {
    let (adapter_name, backend, device_type) = adapter_fields(adapter);
    let mut lines = vec![
        format!("adapter   {adapter_name} ({device_type}, {backend})"),
        format!(
            "budget    {} MiB ({}: {})",
            budget.mib,
            budget.source.as_str(),
            budget.reason
        ),
    ];
    lines.extend(report.lines.iter().cloned());
    lines.extend(report.notes.iter().map(|note| format!("note: {note}")));
    lines
}

fn plan_model(
    dir: &Path,
    budget_mib: u64,
    precision: Precision,
    lora_rank: usize,
    pinned: Pinned,
    lengths: Option<&LengthProfile>,
) -> Result<(Plan, Report), Box<dyn std::error::Error>> {
    let config_path = crate::resolve_model(dir)?.join("config.json");
    let config = ModelLoader::load_config(&config_path)
        .map_err(|e| crate::unmet(format!("cannot read {}: {e}", config_path.display())))?;
    let model_arg = dir.display().to_string();
    let shape = ModelShape::from_config(model_arg.clone(), &config);
    let plan = choose_for(&shape, budget_mib, precision, lora_rank, pinned, lengths);
    let mut notes: Vec<String> = short_seq_note(&shape.name, plan.config.seq_len)
        .into_iter()
        .collect();
    if !config.layer_types.is_empty() || config.model_type.as_deref() == Some("lfm2") {
        notes.push("LFM2 blocks are estimated as Llama blocks; not calibrated".into());
    }
    if let Some(profile) = lengths.filter(|_| plan.config.padded_len < plan.config.seq_len) {
        notes.push(format!(
            "memory is estimated for {} rows per sequence, the {}th percentile of the \
             {} tokenized examples, not for --seq-len {}; a batch of the longest sequences \
             ({} rows each) is estimated at {} MiB and has to fit the budget without the margin",
            plan.config.padded_len,
            PLANNING_PERCENT,
            profile.examples(),
            plan.config.seq_len,
            profile.longest_len(plan.config.seq_len),
            plan.longest_peak_mib
        ));
    }
    if plan.config.checkpointing == Checkpointing::Blocks {
        notes.push(
            "--checkpoint-blocks took 19% to 39% more time per step in the measured runs".into(),
        );
    }
    if !plan.fits && precision == Precision::F16 {
        let q4 = choose_for(
            &shape,
            budget_mib,
            Precision::Q4,
            lora_rank,
            pinned,
            lengths,
        );
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
    let report = Report {
        exit: if plan.fits {
            Exit::Success
        } else {
            Exit::Unmet
        },
        fields: plan_json(&shape, &model_arg, &plan),
        lines,
        notes,
    };
    Ok((plan, report))
}

fn plan_documented(args: &PlanArgs, budget_mib: u64, pinned: Pinned) -> Report {
    let mut models = Vec::new();
    let mut lines = vec![format!(
        "\n{:<14} {:>10}  {:<58} {:>10}  fits",
        "model", "parameters", "plan", "estimate"
    )];
    let mut any_fits = false;
    let mut notes = Vec::new();
    for shape in ModelShape::documented() {
        let plan = choose(&shape, budget_mib, args.weights, args.lora_rank, pinned);
        any_fits |= plan.fits;
        notes.extend(short_seq_note(&shape.name, plan.config.seq_len));
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
        exit: if any_fits { Exit::Success } else { Exit::Unmet },
        fields,
        lines,
        notes,
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use self::Peak::{AtLeast, Steady};
    use super::Checkpointing::{Blocks, Off};
    use super::*;

    fn smollm2(name: &str) -> ModelShape {
        ModelShape::documented()
            .into_iter()
            .find(|s| s.name == name)
            .expect("documented shape")
    }

    /// A documented SmolLM2 size, or a measured model given here.
    fn shape(name: &str) -> ModelShape {
        match name {
            "Qwen2.5-0.5B-Instruct" => ModelShape {
                name: name.into(),
                hidden: 896,
                ffn: 4864,
                n_layers: 24,
                n_heads: 14,
                n_kv_heads: 2,
                head_dim: 64,
                vocab: 151_936,
                tied_embeddings: true,
            },
            _ => smollm2(name),
        }
    }

    fn config(batch_size: usize, checkpointing: Checkpointing) -> TrainConfig {
        TrainConfig {
            batch_size,
            grad_accum: grad_accum_for(batch_size),
            seq_len: 256,
            padded_len: 256,
            lora_rank: 8,
            precision: Precision::F16,
            checkpointing,
        }
    }

    /// A measured peak, or one below the steady peak of its configuration:
    /// a run of 30 steps that accumulates passes has not reached it (see
    /// [`ACCUMULATION_OVERLAP`]), and a peak at the card's whole memory
    /// hides the rest.
    #[derive(Copy, Clone, Debug)]
    enum Peak {
        Steady(u64),
        AtLeast(u64),
    }

    /// Peak GPU memory (MiB, largest `nvidia-smi` sample at 200 ms) of
    /// LoRA training, f16 weights, rank 8: card, model, batch size,
    /// gradient accumulation, sequence length, checkpointing, peak. A new
    /// run adds a row; a new model also goes into [`shape`].
    const MEASURED: [(&str, &str, usize, usize, usize, Checkpointing, Peak); 26] = [
        // 30 steps of `train` on main at af41068, 15 Nov:
        // docs/examples/lora-sft.md, "Confirmation run of merged main".
        ("A10G", "SmolLM2-135M", 1, 8, 256, Off, AtLeast(2306)),
        ("A10G", "SmolLM2-135M", 1, 8, 256, Blocks, AtLeast(1890)),
        ("A10G", "SmolLM2-360M", 1, 8, 256, Off, AtLeast(3579)),
        ("A10G", "SmolLM2-360M", 1, 8, 256, Blocks, AtLeast(2783)),
        ("A10G", "SmolLM2-1.7B", 1, 8, 256, Off, AtLeast(8142)),
        ("A10G", "SmolLM2-1.7B", 1, 8, 256, Blocks, AtLeast(5313)),
        ("A10G", "SmolLM2-135M", 8, 8, 256, Off, AtLeast(9441)),
        ("A10G", "SmolLM2-135M", 8, 8, 256, Blocks, AtLeast(2767)),
        // Main at ca9630a, 16 Nov: 30 steps of `finetune`, then 250 steps,
        // then 30 steps of `train --profile`.
        ("A10G", "SmolLM2-135M", 8, 1, 256, Off, Steady(6331)),
        ("A10G", "SmolLM2-360M", 8, 1, 256, Off, Steady(9104)),
        ("A10G", "SmolLM2-360M", 4, 2, 256, Off, AtLeast(7220)),
        ("A10G", "SmolLM2-1.7B", 4, 2, 256, Off, AtLeast(15064)),
        ("A10G", "SmolLM2-1.7B", 8, 1, 128, Off, Steady(14196)),
        (
            "A10G",
            "Qwen2.5-0.5B-Instruct",
            8,
            1,
            256,
            Off,
            Steady(11197),
        ),
        ("A10G", "SmolLM2-1.7B", 4, 2, 256, Off, Steady(15128)),
        ("A10G", "SmolLM2-135M", 1, 8, 256, Off, AtLeast(2146)),
        ("A10G", "SmolLM2-1.7B", 1, 8, 256, Off, AtLeast(7710)),
        // T4 (15,360 MiB), main at c7a0131, 16 Nov: the same runs. 1.7B at
        // seq 128 filled the card; Ferrotensor counted 16,354 MiB of
        // buffers.
        ("T4", "SmolLM2-135M", 8, 1, 256, Off, Steady(6146)),
        ("T4", "SmolLM2-360M", 8, 1, 256, Off, Steady(8920)),
        ("T4", "SmolLM2-360M", 4, 2, 256, Off, AtLeast(7002)),
        ("T4", "SmolLM2-1.7B", 2, 4, 256, Off, AtLeast(9541)),
        ("T4", "SmolLM2-1.7B", 2, 4, 256, Off, Steady(11138)),
        ("T4", "SmolLM2-1.7B", 8, 1, 128, Off, AtLeast(14913)),
        ("T4", "Qwen2.5-0.5B-Instruct", 8, 1, 256, Off, Steady(11012)),
        ("T4", "SmolLM2-135M", 1, 8, 256, Off, AtLeast(1953)),
        ("T4", "SmolLM2-1.7B", 1, 8, 256, Off, AtLeast(7520)),
    ];

    /// Every peak of a configuration `plan` can choose (8 sequences per
    /// step) is estimated at most 5% low, and every steady peak at most
    /// 12% high. The two 64-sequence rows are data only: 135M at batch
    /// 8 x 8 measured 9,441 MiB on af41068, against 6,331 MiB at batch
    /// 8 x 1 on ca9630a. Every measured run kept a frozen f32 embedding
    /// table on the GPU, so it is estimated with the f32 term.
    #[test]
    fn estimate_error_on_every_measured_peak() {
        let (mut lowest, mut highest_steady, mut highest) = (0.0_f64, 0.0_f64, 0.0_f64);
        for (card, name, batch_size, grad_accum, seq_len, checkpointing, peak) in MEASURED {
            if batch_size * grad_accum != SEQUENCES_PER_STEP {
                continue;
            }
            // The two Qwen2.5-0.5B peaks (2,048 rows over 151,936 classes)
            // were measured with the whole logits; the trainer now chunks
            // that head and the estimate counts the chunk.
            if ferrotensor_trainer::head_chunk_rows(batch_size * seq_len, shape(name).vocab)
                .is_some()
            {
                continue;
            }
            let config = TrainConfig {
                batch_size,
                grad_accum,
                seq_len,
                padded_len: seq_len,
                lora_rank: 8,
                precision: Precision::F16,
                checkpointing,
            };
            let estimate = estimate_with_embedding(&shape(name), &config, 4.0).peak_mib;
            let (Steady(measured) | AtLeast(measured)) = peak;
            let error = (estimate as f64 - measured as f64) / measured as f64;
            let context = format!(
                "{card} {name} {batch_size} x {grad_accum} seq {seq_len} {checkpointing:?}: \
                 estimate {estimate} MiB, measured {measured} MiB"
            );
            assert!(error >= -0.05, "{context}");
            if let Steady(_) = peak {
                assert!(error <= 0.12, "{context}");
                highest_steady = highest_steady.max(error);
            }
            lowest = lowest.min(error);
            highest = highest.max(error);
        }
        // A10G 360M with --checkpoint-blocks: 2,656 against 2,783 MiB.
        assert!((lowest + 0.0456).abs() < 1e-3, "lowest error {lowest}");
        assert!(-lowest < SAFETY_MARGIN);
        // T4 360M at batch 8: 9,961 against 8,920 MiB.
        assert!(
            (highest_steady - 0.1167).abs() < 1e-3,
            "highest steady error {highest_steady}"
        );
        // T4 135M at batch 1 x 8 with --profile: 2,642 against 1,953 MiB.
        assert!((highest - 0.3528).abs() < 1e-3, "highest error {highest}");
    }

    #[test]
    fn estimate_parts_for_smollm2_360m_at_batch_4() {
        // Activations by hand: 24,960 counted values per token and block,
        // plus 13 x 320 for the key/value width, x 32 blocks x 1,024 tokens
        // x 4 bytes = 3,640 MiB, x 0.92, x 1.20 for the two accumulated
        // passes. Logits: 2 x 1,024 x 49,152 x 4 bytes.
        let estimate = estimate_peak_mib(&smollm2("SmolLM2-360M"), &config(4, Checkpointing::Off));
        assert_eq!(
            (
                estimate.weights_mib,
                estimate.adapters_mib,
                estimate.activations_mib,
                estimate.logits_mib,
                estimate.overhead_mib,
                estimate.peak_mib
            ),
            (780, 25, 4019, 384, 1600, 6808)
        );
    }

    #[test]
    fn the_frozen_embedding_is_stored_at_the_weight_type() {
        // SmolLM2-360M: 49,152 x 960 weights. The embedding took 4 bytes
        // before (180 MiB) and takes 2 bytes at f16 and 4-bit (90 MiB); with
        // f32 weights it is still 4 bytes.
        let shape = smollm2("SmolLM2-360M");
        let table_weights = 49_152.0 * 960.0;
        let mut c = config(1, Checkpointing::Off);
        for (precision, decoder_bytes, table_bytes) in [
            (Precision::F16, 2.0, 2.0),
            (Precision::Q4, 4.5 / 8.0, 2.0),
            (Precision::F32, 4.0, 4.0),
        ] {
            c.precision = precision;
            // lm_head is a second table at the same type.
            let by_hand = (shape.decoder_weights() as f64 * decoder_bytes
                + 2.0 * table_weights * table_bytes)
                / MIB;
            assert_eq!(
                estimate_peak_mib(&shape, &c).weights_mib,
                by_hand.ceil() as u64,
                "{precision:?}"
            );
        }
        c.precision = Precision::F16;
        assert_eq!(
            estimate_with_embedding(&shape, &c, 4.0).weights_mib
                - estimate_peak_mib(&shape, &c).weights_mib,
            90
        );
    }

    #[test]
    fn logits_use_the_model_vocabulary() {
        // Up to 512 rows the whole logits count, whatever the vocabulary.
        let qwen = shape("Qwen2.5-0.5B-Instruct");
        let mut c = config(4, Checkpointing::Off);
        c.padded_len = 128;
        // 4 x 128 tokens: 2 x 512 x 151,936 x 4 bytes.
        assert_eq!(estimate_peak_mib(&qwen, &c).logits_mib, 594);
    }

    #[test]
    fn a_chunked_head_counts_one_chunk_of_logits() {
        // 2,048 tokens over 151,936 classes need 2.32 GiB whole. The head
        // runs in chunks of 512 rows: 2 x 512 x 151,936 x 4 bytes.
        let qwen = shape("Qwen2.5-0.5B-Instruct");
        assert_eq!(
            estimate_peak_mib(&qwen, &config(8, Checkpointing::Off)).logits_mib,
            594
        );
        // One sequence of 4,096 tokens counts the same 512 rows.
        let mut long = config(1, Checkpointing::Off);
        long.padded_len = 4096;
        assert_eq!(estimate_peak_mib(&qwen, &long).logits_mib, 594);
        // Gemma's 262,208 classes at 512 rows: 2 x 512 x 262,208 x 4 bytes.
        let mut gemma = qwen.clone();
        gemma.vocab = 262_208;
        assert_eq!(
            estimate_peak_mib(&gemma, &config(8, Checkpointing::Off)).logits_mib,
            1025
        );
    }

    #[test]
    fn smollm2_at_the_speed_gate_keeps_its_whole_logits() {
        // 2,048 tokens over 49,152 classes are 0.75 GiB of logits and
        // gradient, under the 1 GiB that turns chunking on: 2 x 2,048 x
        // 49,152 x 4 bytes. A pass of 256 tokens needs 96 MiB.
        for name in ["SmolLM2-135M", "SmolLM2-360M", "SmolLM2-1.7B"] {
            for batch in [1, 8] {
                assert_eq!(
                    estimate_peak_mib(&smollm2(name), &config(batch, Checkpointing::Off))
                        .logits_mib,
                    if batch == 8 { 768 } else { 96 },
                    "{name} batch {batch}"
                );
            }
        }
    }

    #[test]
    fn base_weights_match_the_memory_report_for_smollm2_1_7b() {
        // `train --memory-report` showed 3,648 MiB of frozen base weights
        // and under 50 MiB for adapters, gradients and AdamW state, with an
        // f32 embedding table. The f16 table is 49,152 x 2,048 x 2 bytes =
        // 192 MiB smaller.
        let estimate = estimate_peak_mib(&smollm2("SmolLM2-1.7B"), &config(1, Checkpointing::Off));
        assert_eq!(estimate.weights_mib, 3648 - 192);
        assert_eq!(estimate.adapters_mib, 48);
    }

    #[test]
    fn weight_precision_changes_only_the_weights() {
        let shape = smollm2("SmolLM2-1.7B");
        let mut c = config(1, Checkpointing::Off);
        c.precision = Precision::F32;
        assert_eq!(estimate_peak_mib(&shape, &c).weights_mib, 6912);
        c.precision = Precision::Q4;
        assert_eq!(estimate_peak_mib(&shape, &c).weights_mib, 1440 - 192);
        assert_eq!(estimate_peak_mib(&shape, &c).activations_mib, 2597);
    }

    #[test]
    fn parameter_counts_match_the_published_models() {
        assert_eq!(smollm2("SmolLM2-135M").parameters(), 134_515_008);
        assert_eq!(smollm2("SmolLM2-360M").parameters(), 361_821_120);
        assert_eq!(smollm2("SmolLM2-1.7B").parameters(), 1_711_376_384);
    }

    #[test]
    fn chooser_prefers_the_longer_sequence_then_no_checkpointing_then_batch() {
        // Batch 8 at seq 128 (4,194 MiB) also fits, but seq 256 wins.
        let p = choose(
            &smollm2("SmolLM2-135M"),
            6000,
            Precision::F16,
            8,
            Pinned::default(),
        );
        assert_eq!(
            (
                p.config.checkpointing,
                p.config.batch_size,
                p.config.seq_len,
                p.estimate.peak_mib,
                p.fits
            ),
            (Checkpointing::Off, 4, 256, 4571, true)
        );
        assert_eq!(short_seq_note("m", p.config.seq_len), None);
    }

    #[test]
    fn chooser_drops_to_seq_128_only_when_256_fails_with_checkpointing() {
        // Seq 256 needs 5,362 MiB even with checkpointing at batch 1; 6,200
        // MiB leaves 5,270 usable.
        let p = choose(
            &smollm2("SmolLM2-1.7B"),
            6200,
            Precision::F16,
            8,
            Pinned::default(),
        );
        assert_eq!(
            (
                p.config.checkpointing,
                p.config.batch_size,
                p.config.seq_len,
                p.estimate.peak_mib,
                p.fits
            ),
            (Checkpointing::Blocks, 1, 128, 5233, true)
        );
        assert_eq!(
            short_seq_note("SmolLM2-1.7B", p.config.seq_len).as_deref(),
            Some(
                "SmolLM2-1.7B: seq 128 cuts every example longer than 128 tokens; \
                 pass --seq-len N to plan and train to keep a length"
            )
        );
    }

    /// An instruction-tuning set like the one the A10G runs train on: 990
    /// examples of 20 to 160 tokens and 10 longer ones up to 224.
    fn alpaca_like_lengths() -> LengthProfile {
        let mut lengths: Vec<usize> = (0..990).map(|i| 20 + i * 140 / 989).collect();
        lengths.extend((0..10).map(|i| 161 + i * 7));
        LengthProfile::new(lengths).expect("lengths")
    }

    #[test]
    fn planning_len_is_the_nearest_rank_percentile_cut_at_the_cap() {
        assert_eq!(LengthProfile::new(Vec::new()), None);
        let three = LengthProfile::new(vec![5, 1, 3]).expect("lengths");
        assert_eq!((three.examples(), three.planning_len(256)), (3, 5));
        assert_eq!(three.planning_len(4), 4);
        let hundred = LengthProfile::new((1..=100).rev().collect()).expect("lengths");
        // The 99th of 100 sorted lengths, not the longest.
        assert_eq!(hundred.planning_len(256), 99);
        assert_eq!(hundred.planning_len(50), 50);
        assert_eq!(hundred.planning_len(0), 1);
        let alpaca = alpaca_like_lengths();
        assert_eq!((alpaca.examples(), alpaca.planning_len(256)), (1000, 160));
        assert_eq!(alpaca.planning_len(128), 128);
    }

    #[test]
    fn short_data_plans_batch_8_without_checkpointing_for_1_7b_on_24_gb() {
        let shape = smollm2("SmolLM2-1.7B");
        let lengths = alpaca_like_lengths();
        let batch_8 = Pinned {
            batch_size: NonZeroUsize::new(8),
            ..Pinned::default()
        };
        // 8 x 160 rows: 16,404 MiB, under the 19,573 MiB an A10G's 23,028
        // MiB leaves after the margin.
        let with_data = choose_for(&shape, 23028, Precision::F16, 8, batch_8, Some(&lengths));
        assert_eq!(
            (
                with_data.config.checkpointing,
                with_data.config.batch_size,
                with_data.config.seq_len,
                with_data.config.padded_len,
                with_data.estimate.peak_mib,
                with_data.fits
            ),
            (Checkpointing::Off, 8, 256, 160, 16404, true)
        );
        // A batch of the longest sequences, 8 x 224 rows, is 20,923 MiB and
        // fits the 23,028 MiB budget itself.
        assert_eq!(with_data.longest_peak_mib, 20923);
        // On a 20,500 MiB budget 16,404 MiB is under the margin (17,425 MiB),
        // but the longest batch does not fit, so the planner checkpoints.
        let tight = choose_for(&shape, 20500, Precision::F16, 8, batch_8, Some(&lengths));
        assert_eq!(
            (
                tight.config.checkpointing,
                tight.longest_peak_mib < 20500,
                tight.fits
            ),
            (Checkpointing::Blocks, true, true)
        );
        // 8 x 256 rows (23,183 MiB) do not fit, so without the data the
        // planner checkpoints.
        let blind = choose(&shape, 23028, Precision::F16, 8, batch_8);
        assert_eq!(
            (
                blind.config.checkpointing,
                blind.config.padded_len,
                blind.estimate.peak_mib
            ),
            (Checkpointing::Blocks, 256, 6947)
        );
        // Unpinned, the data lets batch 8 win over batch 4 x 2.
        let unpinned = choose_for(
            &shape,
            23028,
            Precision::F16,
            8,
            Pinned::default(),
            Some(&lengths),
        );
        assert_eq!(
            (unpinned.config.checkpointing, unpinned.config.batch_size),
            (Checkpointing::Off, 8)
        );
        let unpinned_blind = choose(&shape, 23028, Precision::F16, 8, Pinned::default());
        assert_eq!(
            (
                unpinned_blind.config.checkpointing,
                unpinned_blind.config.batch_size
            ),
            (Checkpointing::Off, 4)
        );
    }

    #[test]
    fn the_smaller_models_plan_the_same_with_or_without_the_data() {
        let lengths = alpaca_like_lengths();
        for name in ["SmolLM2-135M", "SmolLM2-360M"] {
            let shape = smollm2(name);
            let pick = |profile| {
                let plan = choose_for(&shape, 23028, Precision::F16, 8, Pinned::default(), profile);
                (plan.config.checkpointing, plan.config.batch_size)
            };
            assert_eq!(pick(None), (Checkpointing::Off, 8), "{name}");
            assert_eq!(pick(Some(&lengths)), (Checkpointing::Off, 8), "{name}");
        }
    }

    #[test]
    fn chooser_falls_back_to_checkpointing_on_a_small_budget() {
        let p = choose(
            &smollm2("SmolLM2-360M"),
            3200,
            Precision::F16,
            8,
            Pinned::default(),
        );
        assert_eq!(
            (
                p.config.checkpointing,
                p.config.batch_size,
                p.config.seq_len,
                p.estimate.peak_mib,
                p.fits
            ),
            (Checkpointing::Blocks, 1, 256, 2566, true)
        );
    }

    #[test]
    fn chooser_keeps_a_fixed_sequence_length() {
        let shape = smollm2("SmolLM2-1.7B");
        let p = choose(&shape, 8000, Precision::F16, 8, Pinned::default());
        assert_eq!(
            (
                p.config.checkpointing,
                p.config.batch_size,
                p.config.seq_len,
                p.estimate.peak_mib
            ),
            (Checkpointing::Blocks, 4, 256, 6133)
        );
        let seq = Pinned {
            seq_len: NonZeroUsize::new(128),
            ..Pinned::default()
        };
        let p = choose(&shape, 8000, Precision::F16, 8, seq);
        assert_eq!(
            (
                p.config.checkpointing,
                p.config.batch_size,
                p.config.seq_len,
                p.estimate.peak_mib
            ),
            (Checkpointing::Off, 1, 128, 6451)
        );
    }

    #[test]
    fn chooser_keeps_a_pinned_batch_size_and_checkpointing() {
        let shape = smollm2("SmolLM2-135M");
        let summary = |p: Plan| {
            (
                p.config.checkpointing,
                p.config.batch_size,
                p.config.seq_len,
                p.estimate.peak_mib,
                p.fits,
            )
        };
        // Unpinned this is batch 8 at 6,462 MiB. Batch 2 by hand: weights
        // 311 + adapters 15 + activations 1,131 + logits 192 + overhead 1,600.
        let batch_2 = Pinned {
            batch_size: NonZeroUsize::new(2),
            ..Pinned::default()
        };
        assert_eq!(
            summary(choose(&shape, 8000, Precision::F16, 8, batch_2)),
            (Checkpointing::Off, 2, 256, 3249, true)
        );
        let blocks = Pinned {
            checkpointing: Some(Checkpointing::Blocks),
            ..Pinned::default()
        };
        assert_eq!(
            summary(choose(&shape, 8000, Precision::F16, 8, blocks)),
            (Checkpointing::Blocks, 8, 256, 2944, true)
        );
        // Nothing fits 500 MiB: the smallest plan keeps the pinned batch.
        // Activations 75 and logits 192 at 512 tokens of seq 128.
        let batch_4 = Pinned {
            batch_size: NonZeroUsize::new(4),
            ..Pinned::default()
        };
        assert_eq!(
            summary(choose(&shape, 500, Precision::F16, 8, batch_4)),
            (Checkpointing::Blocks, 4, 128, 2193, false)
        );
    }

    #[test]
    fn chooser_reports_the_smallest_plan_when_nothing_fits() {
        let p = choose(
            &smollm2("SmolLM2-135M"),
            500,
            Precision::F16,
            8,
            Pinned::default(),
        );
        assert_eq!(
            (
                p.config.checkpointing,
                p.config.batch_size,
                p.config.seq_len,
                p.estimate.peak_mib,
                p.fits
            ),
            (Checkpointing::Blocks, 1, 128, 1993, false)
        );
        // 1.7B on 6,000 MiB: f16 needs 5,233 MiB plus the margin; 4-bit fits.
        let shape = smollm2("SmolLM2-1.7B");
        assert!(!choose(&shape, 6000, Precision::F16, 8, Pinned::default()).fits);
        let q4 = choose(&shape, 6000, Precision::Q4, 8, Pinned::default());
        assert_eq!(
            (
                q4.config.checkpointing,
                q4.config.batch_size,
                q4.config.seq_len,
                q4.estimate.peak_mib,
                q4.fits
            ),
            (Checkpointing::Blocks, 8, 256, 4739, true)
        );
    }

    fn args(memory_mib: Option<u64>, vram_mib: Option<u64>) -> BudgetArgs {
        BudgetArgs {
            memory_mib,
            vram_mib,
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
            kernel_timing: None,
            cooperative_matrix: Vec::new(),
            selected: true,
        }
    }

    fn no_nvidia_smi() -> Result<u64, String> {
        Err("cannot run nvidia-smi: not found".into())
    }

    #[test]
    fn budget_flags_win_over_the_adapter() {
        let nvidia = adapter("NVIDIA", "DiscreteGpu");
        let b = budget(
            &args(Some(3000), Some(9000)),
            Some(&nvidia),
            Ok(8192),
            || Ok(24000),
        );
        assert_eq!((b.mib, b.source), (3000, BudgetSource::User));
        let b = budget(&args(None, Some(9000)), Some(&nvidia), Ok(8192), || {
            Ok(24000)
        });
        assert_eq!((b.mib, b.source), (9000, BudgetSource::User));
    }

    #[test]
    fn unified_memory_budgets_a_share_of_system_ram() {
        let b = budget(
            &args(None, None),
            Some(&adapter("Apple", "DiscreteGpu")),
            Ok(8192),
            no_nvidia_smi,
        );
        assert_eq!((b.mib, b.source), (4915, BudgetSource::Measured));
        let b = budget(
            &args(None, None),
            Some(&adapter("Intel", "IntegratedGpu")),
            Ok(16384),
            no_nvidia_smi,
        );
        assert_eq!((b.mib, b.source), (9830, BudgetSource::Measured));
        let b = budget(
            &args(None, None),
            Some(&adapter("Intel", "IntegratedGpu")),
            Err("no /proc".into()),
            no_nvidia_smi,
        );
        assert_eq!((b.mib, b.source), (4096, BudgetSource::Assumed));
    }

    #[test]
    fn nvidia_budget_comes_from_nvidia_smi() {
        let b = budget(
            &args(None, None),
            Some(&adapter("NVIDIA", "DiscreteGpu")),
            Ok(65536),
            || Ok(23028),
        );
        assert_eq!((b.mib, b.source), (23028, BudgetSource::Measured));
        assert!(b.reason.contains("nvidia-smi"), "{}", b.reason);
    }

    #[test]
    fn discrete_gpu_budget_is_assumed_without_nvidia_smi() {
        let b = budget(
            &args(None, None),
            Some(&adapter("NVIDIA", "DiscreteGpu")),
            Ok(65536),
            no_nvidia_smi,
        );
        assert_eq!((b.mib, b.source), (4096, BudgetSource::Assumed));
        assert!(
            b.reason.starts_with("assumed, not measured"),
            "{}",
            b.reason
        );
        // nvidia-smi is only asked about NVIDIA adapters.
        let b = budget(
            &args(None, None),
            Some(&adapter("AMD", "DiscreteGpu")),
            Ok(65536),
            || Ok(23028),
        );
        assert_eq!((b.mib, b.source), (4096, BudgetSource::Assumed));
    }

    #[test]
    fn nvidia_smi_output_gives_the_smallest_total() {
        assert_eq!(parse_nvidia_smi_total_mib("23028\n"), Ok(23028));
        assert_eq!(parse_nvidia_smi_total_mib("24576\n 8192 \n"), Ok(8192));
        assert_eq!(
            parse_nvidia_smi_total_mib(""),
            Err("nvidia-smi listed no GPU".to_string())
        );
        assert_eq!(
            parse_nvidia_smi_total_mib("NVIDIA-SMI has failed\n"),
            Err(
                "nvidia-smi line \"NVIDIA-SMI has failed\": invalid digit found in string"
                    .to_string()
            )
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
