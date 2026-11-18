//! `ferrotensor dpo`: LoRA training on preference pairs with Direct
//! Preference Optimization.
//!
//! The policy is the base model with LoRA adapters; the reference is the
//! same model with the adapters switched off, so no second copy of the
//! weights is loaded. The optimizer, schedule, clipping, checkpoints,
//! `--resume`, `metrics.jsonl` and `--json` work as in `train`, and the
//! adapter it writes exports and merges like any other.

use crate::{
    build_model, check_optimizer_load, check_resume_file, ensure_not_empty, load_base,
    profile_report, read_adapter, restore_checkpoint, state_path_for, unmet, ModelArgs, Result,
    MIB,
};
use clap::Args;
use ferrotensor_autograd::{AdamW, WarmupCosineScheduler};
use ferrotensor_backend::MemoryCategory;
use ferrotensor_graph::lfm::legacy_llama_name;
use ferrotensor_trainer::checkpoint::{
    save_lora_checkpoint, upgrade_optimizer_names, LoraMetadata, TrainingCheckpoint,
    TrainingState,
};
use ferrotensor_trainer::data::shuffled_indices;
use ferrotensor_trainer::dpo::{DpoStats, DEFAULT_BETA};
use ferrotensor_trainer::{load_preference_jsonl, PreferencePair, Trainer};
use std::fs;
use std::io::Write as _;
use std::path::PathBuf;
use std::time::Instant;

/// The objective name in `training_state.json`; `--resume` checks it.
const TASK: &str = "dpo";

/// Options for `ferrotensor dpo`.
#[derive(Args, Debug)]
pub struct DpoArgs {
    #[command(flatten)]
    model: ModelArgs,
    /// JSONL preference pairs, one {"prompt", "chosen", "rejected"} object
    /// per line ("system" optional).
    #[arg(long, alias = "dataset", value_name = "FILE")]
    data: PathBuf,
    /// Maximum tokens of prompt plus response, for each side of a pair.
    /// Responses are cut at the end; pairs where either side keeps no
    /// response token are skipped.
    #[arg(long, default_value_t = 256)]
    seq_len: usize,
    /// Use only the first N pairs.
    #[arg(long)]
    max_samples: Option<usize>,
    /// Do not wrap prompts in the ChatML template.
    #[arg(long)]
    no_chat_template: bool,
    /// DPO temperature: how far the policy may move from the reference.
    /// The loss is -log sigmoid(beta * margin of the log-probability ratios).
    #[arg(long, default_value_t = DEFAULT_BETA)]
    beta: f32,
    /// Output directory for the adapter, checkpoints and metrics.jsonl.
    #[arg(
        long,
        alias = "output-dir",
        value_name = "DIR",
        default_value = "outputs/dpo"
    )]
    out: PathBuf,
    /// Where to write the adapter [default: OUT/lora_adapters.safetensors].
    #[arg(long, value_name = "FILE")]
    save_lora: Option<PathBuf>,
    /// Resume from the adapter written by a previous `dpo` run (its
    /// .state.json and .state.safetensors restore the step and optimizer).
    #[arg(long, value_name = "FILE")]
    resume: Option<PathBuf>,
    /// Peak learning rate.
    #[arg(long, default_value_t = 5e-5)]
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
    /// Pairs per batch. On Llama-style models the chosen and rejected
    /// sequences of a batch run through the GPU together.
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
    /// Log every N optimizer steps.
    #[arg(long, default_value_t = 10)]
    log_every: usize,
    /// Save adapter and checkpoint every N optimizer steps (0 = only at the end).
    #[arg(long, default_value_t = 0)]
    save_every: usize,
    /// Data-order seed.
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Recompute each decoder block during backward instead of keeping its
    /// activations.
    #[arg(long)]
    checkpoint_blocks: bool,
    /// Print one JSON summary to stdout when the run ends; everything else
    /// goes to stderr.
    #[arg(long)]
    pub(crate) json: bool,
}

/// What `dpo --json` prints at the end of a run. Values never measured
/// (no step logged) are null.
#[derive(serde::Serialize, Debug)]
pub struct DpoSummary {
    steps: usize,
    first_loss: Option<f32>,
    final_loss: Option<f32>,
    /// Mean of `beta * ((pi_c - ref_c) - (pi_r - ref_r))` over the last
    /// logged window.
    reward_margin: Option<f32>,
    /// Share of the last logged window's pairs with a positive margin.
    accuracy: Option<f32>,
    /// Peak of the GPU buffers Ferrotensor tracks, without the driver context.
    measured_peak_mib: u64,
    seconds: f32,
    out_dir: PathBuf,
    adapter: PathBuf,
}

/// Running sums over the pairs since the last log line.
#[derive(Default)]
struct Window {
    pairs: usize,
    loss: f64,
    margin: f64,
    chosen: f64,
    rejected: f64,
    correct: usize,
    tokens: usize,
}

/// Means of a [`Window`].
#[derive(Clone, Copy, Debug, PartialEq)]
struct WindowMeans {
    loss: f32,
    margin: f32,
    chosen: f32,
    rejected: f32,
    accuracy: f32,
}

impl Window {
    fn add(&mut self, s: &DpoStats) {
        self.pairs += 1;
        self.loss += s.loss as f64;
        self.margin += s.margin as f64;
        self.chosen += s.chosen_reward as f64;
        self.rejected += s.rejected_reward as f64;
        self.correct += usize::from(s.margin > 0.0);
        self.tokens += s.tokens;
    }

    fn means(&self) -> WindowMeans {
        let n = self.pairs.max(1) as f64;
        WindowMeans {
            loss: (self.loss / n) as f32,
            margin: (self.margin / n) as f32,
            chosen: (self.chosen / n) as f32,
            rejected: (self.rejected / n) as f32,
            accuracy: (self.correct as f64 / n) as f32,
        }
    }
}

/// Reads and tokenizes the pairs, keeping the first `max_samples`.
fn load_pairs(args: &DpoArgs, base: &crate::Loaded) -> Result<Vec<PreferencePair>> {
    let mut pairs = load_preference_jsonl(
        &args.data,
        &base.tokenizer,
        !args.no_chat_template,
        args.seq_len,
    )
    .map_err(|e| unmet(format!("{}: {e}", args.data.display())))?;
    if let Some(n) = args.max_samples {
        pairs.truncate(n);
    }
    if pairs.is_empty() {
        return Err(unmet(format!(
            "{} has no usable preference pair (each needs a response token on both sides within --seq-len {})",
            args.data.display(),
            args.seq_len
        )));
    }
    Ok(pairs)
}

/// Runs `ferrotensor dpo` and returns its summary.
///
/// # Errors
///
/// Exit 2 for invalid flags, a missing data file or no usable pair; 1 when
/// training or saving fails.
pub async fn run(mut args: DpoArgs) -> Result<DpoSummary> {
    if args.batch_size == 0 || args.grad_accum == 0 || args.lora_rank == 0 {
        return Err(unmet(
            "--batch-size, --grad-accum and --lora-rank must be positive",
        ));
    }
    if !(args.beta.is_finite() && args.beta > 0.0) {
        return Err(unmet("--beta must be a positive number"));
    }
    if args.seq_len < 2 {
        return Err(unmet("--seq-len must be at least 2"));
    }
    if !args.data.is_file() {
        return Err(unmet(format!(
            "dataset file {} does not exist",
            args.data.display()
        )));
    }

    let base = load_base(&args.model, profile_report::kernel_timing(false)).await?;
    let pairs = load_pairs(&args, &base)?;
    let meta = LoraMetadata {
        rank: args.lora_rank,
        alpha: args.lora_alpha,
        target_modules: args.lora_targets.clone(),
        trained_modules: None,
    };
    let model = {
        let _tag = base.ctx.memory().scope(MemoryCategory::BaseWeights);
        build_model(&base, Some(&meta), false)?
    };
    let named_params = model.named_lora_params();
    if named_params.is_empty() {
        return Err(unmet(format!(
            "--lora-targets {} matches no projection of this model",
            args.lora_targets.join(",")
        )));
    }
    let train_params: Vec<_> = {
        let mut names: Vec<&String> = named_params.keys().collect();
        names.sort_by_cached_key(|n| legacy_llama_name(n).unwrap_or_else(|| n.to_string()));
        names.into_iter().map(|n| named_params[n].clone()).collect()
    };
    ferrotensor_trainer::memory::tag_trainable(&train_params);

    fs::create_dir_all(&args.out)
        .map_err(|e| format!("cannot create {}: {e}", args.out.display()))?;
    let adapter_path = args
        .save_lora
        .clone()
        .unwrap_or_else(|| args.out.join("lora_adapters.safetensors"));
    let state_path = state_path_for(&adapter_path);

    let pairs_per_step = args.batch_size * args.grad_accum;
    let steps_per_epoch = pairs.len().div_ceil(pairs_per_step).max(1);
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
    .with_grad_accum_steps(pairs_per_step)
    .with_clip_grad_norm(args.grad_clip)
    .with_lr_scheduler(Box::new(scheduler));

    let mut start_step = 0usize;
    if let Some(resume) = &args.resume {
        let file = read_adapter(resume)?;
        ensure_not_empty(&file, resume)?;
        check_resume_file(file.metadata.as_ref(), "lora")
            .map_err(|e| format!("{}: {e}", resume.display()))?;
        let n = restore_checkpoint(&base, &trainer.model, &file, resume)?;
        let resume_state = state_path_for(resume);
        if resume_state.is_file() {
            let ckpt = TrainingCheckpoint::load(&resume_state)?;
            ckpt.training_state.validate_resume(
                "lora",
                TASK,
                args.seq_len,
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
        }
        eprintln!(
            "resume: {} ({n} tensors), continuing at step {start_step}",
            resume.display()
        );
    }
    trainer.dropout_seed = args.seed;

    let supervised: usize = pairs.iter().map(PreferencePair::supervised_tokens).sum();
    eprintln!(
        "data:   {} preference pairs, {supervised} response tokens, seq-len {}",
        pairs.len(),
        args.seq_len
    );
    eprintln!(
        "dpo:    beta {}, lora rank {} ({} tensors), lr {} warmup {} cosine, {} steps x {} pairs",
        args.beta,
        args.lora_rank,
        train_params.len(),
        args.lr,
        warmup,
        total_steps,
        pairs_per_step
    );

    let metrics_path = args.out.join("metrics.jsonl");
    let mut metrics = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&metrics_path)?;

    let save = |trainer: &Trainer, step: usize, best: Option<f32>| -> Result<()> {
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
            sequence_length: args.seq_len,
            fp16: false,
            train_modules: "lora".to_string(),
            task: TASK.to_string(),
            model_path: args.model.model.display().to_string(),
            dataset_path: args.data.display().to_string(),
            seed: args.seed,
            q4_weights: None,
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

    eprintln!("  step |   loss | margin | accuracy | grad norm |       lr | s/step");
    let mut step = start_step;
    let skip_pairs = start_step * pairs_per_step;
    let mut seen = 0usize;
    let mut window = Window::default();
    let mut win_steps = 0usize;
    let mut win_start = Instant::now();
    let run_start = Instant::now();
    let mut best: Option<f32> = None;
    let mut first_loss: Option<f32> = None;
    let mut last: Option<WindowMeans> = None;
    let (mut last_norm, mut last_lr) = (0.0f32, 0.0f32);

    'outer: for epoch in 0.. {
        if step >= total_steps {
            break;
        }
        let order = shuffled_indices(pairs.len(), args.seed, epoch);
        for chunk in order.chunks(args.batch_size) {
            if seen < skip_pairs {
                seen += chunk.len();
                continue;
            }
            seen += chunk.len();
            let batch: Vec<PreferencePair> = chunk.iter().map(|&i| pairs[i].clone()).collect();
            for s in trainer.train_dpo(&batch, args.beta, &train_params)? {
                window.add(&s);
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
                win_steps += 1;
                if step.is_multiple_of(args.log_every.max(1)) || step == total_steps {
                    let m = window.means();
                    let secs = win_start.elapsed().as_secs_f32();
                    eprintln!(
                        "{step:6} | {:6.4} | {:6.3} | {:8.3} | {last_norm:9.3} | {last_lr:8.2e} | {:6.2}",
                        m.loss,
                        m.margin,
                        m.accuracy,
                        secs / win_steps.max(1) as f32
                    );
                    writeln!(
                        metrics,
                        "{}",
                        serde_json::json!({
                            "step": step, "loss": m.loss, "reward_margin": m.margin,
                            "accuracy": m.accuracy, "chosen_reward": m.chosen,
                            "rejected_reward": m.rejected, "grad_norm": last_norm,
                            "lr": last_lr, "pairs": window.pairs, "tokens": window.tokens,
                            "seconds": secs, "elapsed": run_start.elapsed().as_secs_f32(),
                        })
                    )?;
                    first_loss.get_or_insert(m.loss);
                    best = Some(best.map_or(m.loss, |b: f32| b.min(m.loss)));
                    last = Some(m);
                    window = Window::default();
                    win_steps = 0;
                    win_start = Instant::now();
                }
                if args.save_every > 0 && step.is_multiple_of(args.save_every) && step < total_steps
                {
                    save(&trainer, step, best)?;
                }
                if step >= total_steps {
                    break 'outer;
                }
            }
        }
        if epoch + 1 >= args.epochs.max(1) as u64 && args.max_steps.is_none() {
            break;
        }
    }

    let elapsed = run_start.elapsed();
    let peak_bytes = base.ctx.memory().peak_total_bytes();
    save(&trainer, step, best)?;
    eprintln!(
        "done:   {step} steps in {:.1}s; loss {} -> {}",
        elapsed.as_secs_f32(),
        first_loss.map_or("n/a".to_string(), |l| format!("{l:.4}")),
        last.map_or("n/a".to_string(), |m| format!("{:.4}", m.loss)),
    );
    eprintln!(
        "saved:  {} (+ {})",
        adapter_path.display(),
        state_path.display()
    );
    eprintln!("metrics: {}", metrics_path.display());
    if !args.json {
        let model = args.model.model.display();
        let adapter = adapter_path.display();
        eprintln!("next:");
        eprintln!("  ferrotensor chat --model {model} --lora {adapter}");
        eprintln!(
            "  ferrotensor export-adapter --model {model} --lora {adapter} --output {}-peft",
            args.out.display()
        );
    }
    Ok(DpoSummary {
        steps: step,
        first_loss,
        final_loss: last.map(|m| m.loss),
        reward_margin: last.map(|m| m.margin),
        accuracy: last.map(|m| m.accuracy),
        measured_peak_mib: peak_bytes.div_ceil(MIB),
        seconds: elapsed.as_secs_f32(),
        out_dir: args.out,
        adapter: adapter_path,
    })
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::{Cli, Command};
    use clap::Parser;

    fn parse(extra: &[&str]) -> DpoArgs {
        let mut argv = vec![
            "ferrotensor",
            "dpo",
            "--model",
            "m",
            "--data",
            "pairs.jsonl",
        ];
        argv.extend_from_slice(extra);
        match Cli::try_parse_from(argv).expect("parse dpo").command {
            Command::Dpo(args) => *args,
            _ => panic!("expected dpo"),
        }
    }

    #[test]
    fn dpo_flags_and_defaults() {
        let args = parse(&[]);
        assert_eq!(args.beta, DEFAULT_BETA);
        assert_eq!(args.data, PathBuf::from("pairs.jsonl"));
        assert_eq!(args.out, PathBuf::from("outputs/dpo"));
        assert_eq!((args.batch_size, args.grad_accum, args.seq_len), (1, 4, 256));
        assert!(!args.json && !args.no_chat_template);
        let args = parse(&["--beta", "0.25", "--batch-size", "2", "--json", "--seq-len", "64"]);
        assert_eq!(args.beta, 0.25);
        assert_eq!((args.batch_size, args.seq_len), (2, 64));
        assert!(args.json);
    }

    fn stats(loss: f32, margin: f32) -> DpoStats {
        DpoStats {
            loss,
            chosen_reward: margin,
            rejected_reward: 0.0,
            margin,
            tokens: 3,
            optimizer_step: false,
            grad_norm: None,
            lr: None,
        }
    }

    #[test]
    fn window_means_and_accuracy() {
        let mut w = Window::default();
        for s in [stats(0.5, 0.25), stats(0.75, -0.5), stats(0.25, 1.0), stats(0.5, 0.0)] {
            w.add(&s);
        }
        let m = w.means();
        assert_eq!(m.loss, 0.5);
        assert_eq!(m.margin, 0.1875);
        // A zero margin does not count as ranked correctly.
        assert_eq!(m.accuracy, 0.5);
        assert_eq!(w.tokens, 12);
    }

    #[test]
    fn dpo_summary_keys() {
        let summary = DpoSummary {
            steps: 2,
            first_loss: Some(0.75),
            final_loss: Some(0.5),
            reward_margin: Some(0.25),
            accuracy: Some(1.0),
            measured_peak_mib: 10,
            seconds: 1.5,
            out_dir: "runs/d".into(),
            adapter: "runs/d/lora_adapters.safetensors".into(),
        };
        assert_eq!(
            serde_json::to_value(&summary).expect("serialize"),
            serde_json::json!({
                "steps": 2,
                "first_loss": 0.75,
                "final_loss": 0.5,
                "reward_margin": 0.25,
                "accuracy": 1.0,
                "measured_peak_mib": 10,
                "seconds": 1.5,
                "out_dir": "runs/d",
                "adapter": "runs/d/lora_adapters.safetensors",
            })
        );
    }
}
