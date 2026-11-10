//! `classify-train` and `classify-predict`: LoRA fine-tuning of a decoder as
//! a text classifier, and class probabilities from the result.
//!
//! Both read JSONL with one `{"text", "label", "id"}` object per line. Texts
//! are tokenized without special tokens or a chat template and cut to
//! `max_len` tokens. The model is built only up to the deepest tapped block.

use super::{adapter_metadata, build_model, load_base, restore_adapter, Loaded, ModelArgs, Result};
use clap::Args;
use ferrotensor_autograd::WarmupCosineScheduler;
use ferrotensor_graph::lfm::Lfm2Model;
use ferrotensor_graph::ClassifierHead;
use ferrotensor_trainer::checkpoint::{save_lora_checkpoint, LoraMetadata};
use ferrotensor_trainer::classify::{
    label_ids, label_names, load_classifier_head, parse_batch_plan, parse_classify_jsonl,
    plan_epoch, predict_rows, predicted_class, save_classifier_head, score, ClassifierMetadata,
    ClassifyRecord, ClassifyScores, OptimizerUpdate, ADAPTER_FILE, HEAD_FILE, METADATA_FILE,
};
use ferrotensor_trainer::ClassifierTrainer;
use std::fs;
use std::io::{BufWriter, Write as _};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Default LoRA targets: every attention and MLP projection.
const ALL_PROJECTIONS: &str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj";

#[derive(Args, Debug)]
pub(crate) struct ClassifyTrainArgs {
    #[command(flatten)]
    pub(crate) model: ModelArgs,
    /// Labelled JSONL: one {"text", "label"[, "id"]} object per line.
    #[arg(long, value_name = "FILE")]
    pub(crate) data: PathBuf,
    /// Output directory for adapter.safetensors, head.safetensors,
    /// classifier.json and train_log.jsonl.
    #[arg(long, value_name = "DIR")]
    pub(crate) output: PathBuf,
    /// Comma-separated blocks (1-based) whose hidden states feed the head
    /// [default: the last block]. Blocks after the deepest tap are not built.
    #[arg(long, value_delimiter = ',', value_name = "N,..")]
    pub(crate) taps: Option<Vec<usize>>,
    /// Maximum tokens per text; longer texts are cut at the end.
    #[arg(long, default_value_t = 384)]
    pub(crate) max_len: usize,
    /// LoRA rank.
    #[arg(long, default_value_t = 16)]
    pub(crate) rank: usize,
    /// LoRA alpha (updates are scaled by alpha / rank).
    #[arg(long, default_value_t = 32.0)]
    pub(crate) alpha: f32,
    /// Comma-separated projections that get LoRA adapters.
    #[arg(long, value_delimiter = ',', default_value = ALL_PROJECTIONS)]
    pub(crate) lora_targets: Vec<String>,
    /// Peak learning rate of the LoRA adapters.
    #[arg(long, default_value_t = 1e-4)]
    pub(crate) lr: f32,
    /// Peak learning rate of the classifier head.
    #[arg(long, default_value_t = 1e-3)]
    pub(crate) head_lr: f32,
    /// Passes over the data (ignored with --batches).
    #[arg(long, default_value_t = 2)]
    pub(crate) epochs: usize,
    /// Texts per batch (ignored with --batches).
    #[arg(long, default_value_t = 8)]
    pub(crate) batch_size: usize,
    /// Batches per optimizer step; their gradients are averaged.
    #[arg(long, default_value_t = 2)]
    pub(crate) grad_accum: usize,
    /// Clip the global gradient norm over the adapters and the head
    /// (0 = no clipping).
    #[arg(long, default_value_t = 1.0)]
    pub(crate) clip_grad_norm: f32,
    /// Linear warmup steps, then cosine decay of both learning rates to a
    /// tenth of their peak.
    #[arg(long, default_value_t = 0)]
    pub(crate) warmup_steps: usize,
    /// Seed of the head initialisation and the data order.
    #[arg(long, default_value_t = 0)]
    pub(crate) seed: u64,
    /// Leave each text's first token out of the mean pooling (the default;
    /// the first position often acts as an attention sink).
    #[arg(long, overrides_with = "no_skip_first_token")]
    pub(crate) skip_first_token: bool,
    /// Pool over every token, the first one included.
    #[arg(long, overrides_with = "skip_first_token")]
    pub(crate) no_skip_first_token: bool,
    /// Batch plan: JSONL with one JSON array of 0-based row indices into
    /// --data per line, one batch per line, in training order for the whole
    /// run. Replaces --epochs, --batch-size and shuffling.
    #[arg(long, value_name = "FILE")]
    pub(crate) batches: Option<PathBuf>,
    /// Labelled JSONL scored at the end of every epoch (loss, accuracy and
    /// balanced accuracy).
    #[arg(long, value_name = "FILE")]
    pub(crate) eval: Option<PathBuf>,
    /// Print every N optimizer steps.
    #[arg(long, default_value_t = 10)]
    pub(crate) log_every: usize,
}

impl ClassifyTrainArgs {
    fn skip_first(&self) -> bool {
        !self.no_skip_first_token
    }
}

#[derive(Args, Debug)]
pub(crate) struct ClassifyPredictArgs {
    #[command(flatten)]
    pub(crate) model: ModelArgs,
    /// Output directory of classify-train.
    #[arg(long, value_name = "DIR")]
    pub(crate) classifier: PathBuf,
    /// JSONL with one {"text"[, "id", "label"]} object per line. When
    /// labels are present, accuracy and balanced accuracy are printed.
    #[arg(long, value_name = "FILE")]
    pub(crate) data: PathBuf,
    /// Output JSONL: one {"id", "probs", "label"} object per input line,
    /// probabilities in label-index order.
    #[arg(long, value_name = "FILE")]
    pub(crate) output: PathBuf,
    /// Texts per forward pass.
    #[arg(long, default_value_t = 8)]
    pub(crate) batch_size: usize,
}

/// Check that `taps` are distinct block numbers in `1..=blocks`.
pub(crate) fn check_taps(taps: &[usize], blocks: usize) -> std::result::Result<(), String> {
    if taps.is_empty() {
        return Err("--taps needs at least one block".into());
    }
    for (i, &tap) in taps.iter().enumerate() {
        if tap == 0 || tap > blocks {
            return Err(format!(
                "tap {tap} is not a block of this model (blocks are 1 to {blocks})"
            ));
        }
        if taps[..i].contains(&tap) {
            return Err(format!("tap {tap} is listed twice"));
        }
    }
    Ok(())
}

fn read_records(path: &Path, require_label: bool) -> Result<Vec<ClassifyRecord>> {
    let content =
        fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let records = parse_classify_jsonl(&content, require_label)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    if records.is_empty() {
        return Err(format!("{} has no rows", path.display()).into());
    }
    Ok(records)
}

/// Token ids of every text (no special tokens, no template), cut to
/// `max_len`; also the number of texts that were cut.
fn tokenize(
    base: &Loaded,
    records: &[ClassifyRecord],
    max_len: usize,
    path: &Path,
) -> Result<(Vec<Vec<u32>>, usize)> {
    let mut rows = Vec::with_capacity(records.len());
    let mut cut = 0;
    for (row, record) in records.iter().enumerate() {
        let mut ids = base.tokenizer.encode_continuation(&record.text);
        if ids.is_empty() {
            return Err(format!(
                "{}: row {row} (id {}) has no tokens",
                path.display(),
                record.id
            )
            .into());
        }
        if ids.len() > max_len {
            ids.truncate(max_len);
            cut += 1;
        }
        rows.push(ids);
    }
    Ok((rows, cut))
}

/// Build the model with `lora` adapters, only up to the deepest tap: the
/// blocks after it never contribute to the logits.
fn build_up_to_deepest_tap(
    base: &mut Loaded,
    taps: &[usize],
    lora: &LoraMetadata,
) -> Result<Lfm2Model> {
    let blocks = base.config.num_hidden_layers;
    check_taps(taps, blocks)?;
    let deepest = taps.iter().copied().max().unwrap_or(blocks);
    base.config.num_hidden_layers = deepest;
    let model = build_model(base, Some(lora), false)?;
    if !model.supports_batching() {
        return Err(
            "classification needs a model whose blocks are all attention (Llama-style)".into(),
        );
    }
    if model.named_lora_params().is_empty() {
        return Err(format!(
            "LoRA targets {} match no projection of this model",
            lora.target_modules.join(",")
        )
        .into());
    }
    eprintln!("blocks: {deepest} of {blocks} built (up to the deepest tap)");
    Ok(model)
}

fn print_scores(tag: &str, s: &ClassifyScores) {
    eprintln!(
        "{tag}: {} rows, loss {:.4}, accuracy {:.4}, balanced accuracy {:.4}",
        s.rows, s.loss, s.accuracy, s.balanced_accuracy
    );
}

fn model_name(dir: &Path) -> String {
    fs::canonicalize(dir)
        .unwrap_or_else(|_| dir.to_path_buf())
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| dir.display().to_string())
}

/// One train_log.jsonl line per optimizer step, and the console table.
struct StepLog {
    file: BufWriter<fs::File>,
    log_every: usize,
    total_steps: usize,
    steps: usize,
    loss_sum: f64,
    batches: usize,
    window_start: Instant,
    first_loss: Option<f64>,
    last_loss: f64,
}

impl StepLog {
    fn batch(&mut self, loss: f32) {
        self.loss_sum += loss as f64;
        self.batches += 1;
    }

    fn step(&mut self, epoch: usize, update: &OptimizerUpdate) -> Result<()> {
        self.steps += 1;
        let step = self.steps;
        let loss = self.loss_sum / self.batches.max(1) as f64;
        let seconds = self.window_start.elapsed().as_secs_f64();
        writeln!(
            self.file,
            "{}",
            serde_json::json!({
                "step": step, "epoch": epoch + 1, "loss": loss,
                "grad_norm": update.grad_norm, "lr": update.lr, "head_lr": update.head_lr,
                "seconds": seconds, "skipped": !update.applied,
            })
        )?;
        if step.is_multiple_of(self.log_every.max(1)) || step == self.total_steps || !update.applied
        {
            eprintln!(
                "{step:6} | {loss:6.4} | {:9.3} | {:8.2e} | {:8.2e} | {seconds:6.2}{}",
                update.grad_norm,
                update.lr,
                update.head_lr,
                if update.applied { "" } else { " (skipped)" }
            );
        }
        self.first_loss.get_or_insert(loss);
        self.last_loss = loss;
        self.loss_sum = 0.0;
        self.batches = 0;
        self.window_start = Instant::now();
        Ok(())
    }
}

async fn save_classifier(
    trainer: &ClassifierTrainer,
    lora: &LoraMetadata,
    metadata: &ClassifierMetadata,
    dir: &Path,
) -> Result<()> {
    save_lora_checkpoint(
        &trainer.ctx,
        dir.join(ADAPTER_FILE),
        &trainer.model.named_lora_params(),
        Some(lora),
    )
    .await?;
    save_classifier_head(&trainer.ctx, dir.join(HEAD_FILE), &trainer.head).await?;
    metadata.save(dir.join(METADATA_FILE))?;
    Ok(())
}

pub(crate) async fn classify_train(args: ClassifyTrainArgs) -> Result<()> {
    if args.batch_size == 0 || args.grad_accum == 0 || args.rank == 0 || args.max_len == 0 {
        return Err("--batch-size, --grad-accum, --rank and --max-len must be positive".into());
    }
    if args.epochs == 0 && args.batches.is_none() {
        return Err("--epochs must be positive".into());
    }
    if args.clip_grad_norm < 0.0 {
        return Err("--clip-grad-norm must be 0 (off) or positive".into());
    }

    // Read and check every input file before loading the model.
    let records = read_records(&args.data, true)?;
    let labels = label_names(&records);
    if labels.len() < 2 {
        return Err(format!(
            "{} has {} distinct label(s); a classifier needs at least two",
            args.data.display(),
            labels.len()
        )
        .into());
    }
    let row_labels =
        label_ids(&records, &labels).map_err(|e| format!("{}: {e}", args.data.display()))?;
    let fixed_plan = match &args.batches {
        Some(path) => {
            let content = fs::read_to_string(path)
                .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
            Some(
                parse_batch_plan(&content, records.len())
                    .map_err(|e| format!("{}: {e}", path.display()))?,
            )
        }
        None => None,
    };
    let eval = match &args.eval {
        Some(path) => {
            let records = read_records(path, true)?;
            let ids = label_ids(&records, &labels)
                .map_err(|e| format!("{}: {e} (labels must occur in --data)", path.display()))?;
            Some((path.clone(), records, ids))
        }
        None => None,
    };

    let mut base = load_base(&args.model).await?;
    let taps = args
        .taps
        .clone()
        .unwrap_or_else(|| vec![base.config.num_hidden_layers]);
    let lora = LoraMetadata {
        rank: args.rank,
        alpha: args.alpha,
        target_modules: args.lora_targets.clone(),
    };
    let model = build_up_to_deepest_tap(&mut base, &taps, &lora)?;
    let (rows, cut) = tokenize(&base, &records, args.max_len, &args.data)?;
    let eval = match eval {
        Some((path, records, ids)) => {
            Some((tokenize(&base, &records, args.max_len, &path)?.0, ids))
        }
        None => None,
    };
    let lengths: Vec<usize> = rows.iter().map(Vec::len).collect();

    let epochs = if fixed_plan.is_some() { 1 } else { args.epochs };
    let total_batches = match &fixed_plan {
        Some(plan) => plan.len(),
        None => epochs * rows.len().div_ceil(args.batch_size),
    };
    let total_steps = total_batches.div_ceil(args.grad_accum).max(1);
    let warmup = args.warmup_steps.min(total_steps.saturating_sub(1));

    let head = ClassifierHead::new_seeded(
        base.ctx.clone(),
        base.config.hidden_size,
        labels.len(),
        &taps,
        args.seed,
    )
    .with_skip_first(args.skip_first());
    let head_params: usize = head
        .params()
        .iter()
        .map(|t| t.data.shape.iter().product::<usize>())
        .sum();
    let mut trainer = ClassifierTrainer::new(
        model,
        head,
        base.ctx.clone(),
        base.engine.clone(),
        args.lr,
        args.head_lr,
    )
    .with_grad_accum(args.grad_accum)
    .with_clip_grad_norm((args.clip_grad_norm > 0.0).then_some(args.clip_grad_norm))
    .with_lr_scheduler(Box::new(WarmupCosineScheduler::new(
        1.0,
        0.1,
        warmup,
        total_steps,
    )));
    let lora_params: usize = trainer
        .lora_params()
        .iter()
        .map(|t| t.data.shape.iter().product::<usize>())
        .sum();

    let metadata = ClassifierMetadata {
        labels: labels.clone(),
        taps: taps.clone(),
        max_len: args.max_len,
        skip_first_token: args.skip_first(),
        rank: args.rank,
        alpha: args.alpha,
        lora_targets: args.lora_targets.clone(),
        base_model: model_name(&args.model.model),
        seed: args.seed,
        ferrotensor_version: env!("CARGO_PKG_VERSION").to_string(),
    };
    fs::create_dir_all(&args.output)
        .map_err(|e| format!("cannot create {}: {e}", args.output.display()))?;
    let log_path = args.output.join("train_log.jsonl");
    let mut log = StepLog {
        file: BufWriter::new(fs::File::create(&log_path)?),
        log_every: args.log_every,
        total_steps,
        steps: 0,
        loss_sum: 0.0,
        batches: 0,
        window_start: Instant::now(),
        first_loss: None,
        last_loss: f64::NAN,
    };

    eprintln!(
        "data:   {} rows, {} labels, {} tokens ({} cut to --max-len {})",
        rows.len(),
        labels.len(),
        lengths.iter().sum::<usize>(),
        cut,
        args.max_len
    );
    eprintln!(
        "train:  taps {taps:?}, {} LoRA params (rank {}, {}) + {head_params} head params, \
         first token {}",
        lora_params,
        args.rank,
        args.lora_targets.join(","),
        if args.skip_first() {
            "skipped"
        } else {
            "pooled"
        }
    );
    eprintln!(
        "        lr {} head-lr {}, warmup {warmup} then cosine, {total_steps} steps x {} batches{}",
        args.lr,
        args.head_lr,
        args.grad_accum,
        if fixed_plan.is_some() {
            format!(" (plan of {total_batches} batches)")
        } else {
            format!(", {epochs} epochs of batch size {}", args.batch_size)
        }
    );
    eprintln!("  step |   loss | grad norm |       lr |  head lr | s/step");

    let run_start = Instant::now();
    for epoch in 0..epochs {
        let shuffled;
        let plan: &[Vec<usize>] = match &fixed_plan {
            Some(plan) => plan.as_slice(),
            None => {
                shuffled = plan_epoch(&lengths, args.batch_size, args.seed, epoch as u64);
                shuffled.as_slice()
            }
        };
        for batch in plan {
            let batch_rows: Vec<&[u32]> = batch.iter().map(|&i| rows[i].as_slice()).collect();
            let batch_labels: Vec<u32> = batch.iter().map(|&i| row_labels[i]).collect();
            let stats = trainer.train_batch(&batch_rows, &batch_labels)?;
            log.batch(stats.loss);
            if let Some(update) = stats.update {
                log.step(epoch, &update)?;
            }
        }
        if epoch + 1 == epochs {
            // The last batches of the run may not fill a window.
            if let Some(update) = trainer.flush() {
                log.step(epoch, &update)?;
            }
        }
        log.file.flush()?;
        if let Some((eval_rows, eval_labels)) = &eval {
            let probs = predict_rows(
                &trainer.engine,
                &trainer.model,
                &trainer.head,
                eval_rows,
                args.batch_size,
                trainer.pad_token,
            )?;
            print_scores(
                &format!("eval epoch {}", epoch + 1),
                &score(&probs, eval_labels, labels.len()),
            );
        }
        save_classifier(&trainer, &lora, &metadata, &args.output).await?;
    }

    eprintln!(
        "done:   {} steps ({} skipped) in {:.1}s; train loss {} -> {:.4}",
        log.steps,
        trainer.skipped_steps(),
        run_start.elapsed().as_secs_f32(),
        log.first_loss
            .map_or("n/a".to_string(), |l| format!("{l:.4}")),
        log.last_loss
    );
    eprintln!(
        "saved:  {} ({ADAPTER_FILE}, {HEAD_FILE}, {METADATA_FILE}, train_log.jsonl)",
        args.output.display()
    );
    Ok(())
}

pub(crate) async fn classify_predict(args: ClassifyPredictArgs) -> Result<()> {
    if args.batch_size == 0 {
        return Err("--batch-size must be positive".into());
    }
    let meta = ClassifierMetadata::load(args.classifier.join(METADATA_FILE))?;
    let adapter_path = args.classifier.join(ADAPTER_FILE);
    let lora = adapter_metadata(&adapter_path)?;
    let records = read_records(&args.data, false)?;
    let truth = records
        .iter()
        .enumerate()
        .map(|(row, r)| match &r.label {
            None => Ok(None),
            Some(label) => meta.label_index(label).map(Some).ok_or_else(|| {
                format!(
                    "{}: row {row}: label {label:?} is not one of the classifier's labels",
                    args.data.display()
                )
            }),
        })
        .collect::<std::result::Result<Vec<Option<u32>>, String>>()?;

    let mut base = load_base(&args.model).await?;
    let name = model_name(&args.model.model);
    if name != meta.base_model {
        eprintln!(
            "warning: the classifier was trained on {:?}, this model is {name:?}",
            meta.base_model
        );
    }
    let model = build_up_to_deepest_tap(&mut base, &meta.taps, &lora)?;
    let n = restore_adapter(&base, &model, &adapter_path)?;
    let head = load_classifier_head(
        &base.ctx,
        args.classifier.join(HEAD_FILE),
        &meta.taps,
        meta.skip_first_token,
    )?;
    if head.classes() != meta.labels.len() || head.hidden() != base.config.hidden_size {
        return Err(format!(
            "{}: the head maps {} to {} classes; expected {} to {}",
            args.classifier.join(HEAD_FILE).display(),
            head.hidden(),
            head.classes(),
            base.config.hidden_size,
            meta.labels.len()
        )
        .into());
    }
    eprintln!(
        "classifier: {} ({n} adapter tensors, {} labels, taps {:?})",
        args.classifier.display(),
        meta.labels.len(),
        meta.taps
    );

    let (rows, cut) = tokenize(&base, &records, meta.max_len, &args.data)?;
    let t = Instant::now();
    let probs = predict_rows(&base.engine, &model, &head, &rows, args.batch_size, 0)?;
    let secs = t.elapsed().as_secs_f32();

    if let Some(parent) = args.output.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
    }
    let mut out = BufWriter::new(
        fs::File::create(&args.output)
            .map_err(|e| format!("cannot create {}: {e}", args.output.display()))?,
    );
    for (record, p) in records.iter().zip(&probs) {
        let label = &meta.labels[predicted_class(p) as usize];
        writeln!(
            out,
            "{}",
            serde_json::json!({"id": record.id, "probs": p, "label": label})
        )?;
    }
    out.flush()?;
    eprintln!(
        "predict: {} rows ({cut} cut to {} tokens) in {secs:.1}s -> {}",
        rows.len(),
        meta.max_len,
        args.output.display()
    );

    let (scored_probs, scored_labels): (Vec<Vec<f32>>, Vec<u32>) = probs
        .iter()
        .zip(&truth)
        .filter_map(|(p, t)| t.map(|t| (p.clone(), t)))
        .unzip();
    if !scored_labels.is_empty() {
        print_scores(
            "labelled",
            &score(&scored_probs, &scored_labels, meta.labels.len()),
        );
    }
    Ok(())
}
