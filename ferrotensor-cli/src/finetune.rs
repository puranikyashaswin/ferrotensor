//! `ferrotensor finetune`: `train` with the memory settings `plan` picks for
//! this machine, for each setting not given on the command line.

use crate::plan::{self, BudgetArgs, Checkpointing, Pinned, Precision, TrainConfig};
use crate::{unmet, Result, TrainArgs, TrainSummary};
use clap::parser::ValueSource;
use clap::{ArgMatches, Args};
use serde::Serialize;
use serde_json::Value;
use std::num::NonZeroUsize;

/// Options for `ferrotensor finetune`: those of `train` plus the budget.
#[derive(Args, Debug)]
pub struct FinetuneArgs {
    #[command(flatten)]
    pub(crate) train: TrainArgs,
    #[command(flatten)]
    budget: BudgetArgs,
    /// Train even when no configuration is expected to fit, with the
    /// smallest one.
    #[arg(long)]
    force: bool,
}

/// Which planner-owned `train` flags were given on the command line.
/// `--checkpoint-blocks` is not listed: it is set exactly when given.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct UserSet {
    batch_size: bool,
    grad_accum: bool,
    seq_len: bool,
}

impl UserSet {
    /// Reads the flags from the matches of the `finetune` subcommand.
    pub fn from_matches(matches: &ArgMatches) -> Self {
        let given = |id: &str| matches.value_source(id) == Some(ValueSource::CommandLine);
        Self {
            batch_size: given("batch_size"),
            grad_accum: given("grad_accum"),
            seq_len: given("seq_len"),
        }
    }
}

/// The values the chooser must keep: the planner-owned flags the user set.
fn pinned(args: &TrainArgs, user_set: UserSet) -> Result<Pinned> {
    let positive = |value: usize, flag: &str| {
        NonZeroUsize::new(value).ok_or_else(|| unmet(format!("{flag} must be positive")))
    };
    Ok(Pinned {
        seq_len: user_set
            .seq_len
            .then(|| positive(args.data.seq_len, "--seq-len"))
            .transpose()?,
        batch_size: user_set
            .batch_size
            .then(|| positive(args.batch_size, "--batch-size"))
            .transpose()?,
        checkpointing: args.checkpoint_blocks.then_some(Checkpointing::Blocks),
    })
}

/// Sets each planner-owned flag the user did not set to `config`'s value,
/// and returns each flag with its value and where it came from.
fn apply_plan(args: &mut TrainArgs, user_set: UserSet, config: &TrainConfig) -> Vec<String> {
    let source = |given: bool| if given { "user" } else { "plan" };
    let checkpoint_given = args.checkpoint_blocks;
    if !user_set.batch_size {
        args.batch_size = config.batch_size;
    }
    if !user_set.grad_accum {
        args.grad_accum = plan::grad_accum_for(args.batch_size);
    }
    if !user_set.seq_len {
        args.data.seq_len = config.seq_len;
    }
    if !checkpoint_given {
        args.checkpoint_blocks = config.checkpointing == Checkpointing::Blocks;
    }
    vec![
        format!(
            "--batch-size {} ({})",
            args.batch_size,
            source(user_set.batch_size)
        ),
        format!(
            "--grad-accum {} ({})",
            args.grad_accum,
            source(user_set.grad_accum)
        ),
        format!(
            "--seq-len {} ({})",
            args.data.seq_len,
            source(user_set.seq_len)
        ),
        format!(
            "--checkpoint-blocks {} ({})",
            if args.checkpoint_blocks { "on" } else { "off" },
            source(checkpoint_given)
        ),
    ]
}

/// What `finetune --json` prints: the plan, then the `train` summary.
#[derive(Serialize, Debug)]
struct FinetuneSummary {
    chosen: Value,
    grad_accum: usize,
    budget_mib: u64,
    estimated_peak_mib: u64,
    /// False only when `--force` trained a configuration that does not fit.
    fits: bool,
    #[serde(flatten)]
    train: TrainSummary,
}

/// Runs `ferrotensor finetune`: prints the plan to stderr, then trains.
///
/// # Errors
///
/// When `--lite` is given, a pinned value is zero, planning fails, nothing
/// fits and `--force` was not given, or training fails.
pub fn run(args: FinetuneArgs, user_set: UserSet) -> Result<()> {
    let FinetuneArgs {
        train: mut train_args,
        budget,
        force,
    } = args;
    if train_args.lite {
        return Err(unmet(
            "finetune picks the memory settings itself; drop --lite or use train --lite",
        ));
    }
    let (precision, precision_source) =
        match (train_args.model.q4_weights, train_args.model.f32_weights) {
            (true, _) => (Precision::Q4, "user"),
            (_, true) => (Precision::F32, "user"),
            _ => (Precision::F16, "default"),
        };
    let planned = plan::plan_training(
        &budget,
        &train_args.model.model,
        precision,
        train_args.lora_rank,
        pinned(&train_args, user_set)?,
    )?;
    for line in &planned.lines {
        eprintln!("{line}");
    }
    let mut settings = apply_plan(&mut train_args, user_set, &planned.plan.config);
    settings.push(format!(
        "{} weights ({precision_source})",
        precision.as_str()
    ));
    eprintln!("finetune: {}", settings.join(", "));

    let estimated_mib = planned.plan.estimate.peak_mib;
    if !planned.plan.fits {
        let shortfall = format!(
            "no configuration fits the {} MiB budget; the smallest needs {estimated_mib} MiB, \
             which fits a budget of {} MiB or more",
            planned.budget_mib, planned.needed_budget_mib
        );
        if !force {
            return Err(unmet(format!("{shortfall}; pass --force to train anyway")));
        }
        eprintln!("finetune: {shortfall}; training anyway (--force)");
    }

    let json = train_args.json;
    let grad_accum = train_args.grad_accum;
    let train = pollster::block_on(crate::train(train_args))?;
    let measured_mib = train.measured_peak_mib;
    eprintln!(
        "memory: estimated peak {estimated_mib} MiB, measured {measured_mib} MiB ({:+.1}%); \
         measured counts the GPU buffers Ferrotensor tracks, not the driver context",
        (measured_mib as f64 - estimated_mib as f64) / estimated_mib as f64 * 100.0
    );
    if json {
        crate::print_json(&FinetuneSummary {
            chosen: plan::chosen_json(&planned.plan.config),
            grad_accum,
            budget_mib: planned.budget_mib,
            estimated_peak_mib: estimated_mib,
            fits: planned.plan.fits,
            train,
        })?;
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::{Cli, Command};
    use clap::{CommandFactory, FromArgMatches};

    /// `finetune --model m --data d.jsonl` plus `extra`, parsed as main does.
    fn parse(extra: &[&str]) -> (TrainArgs, UserSet) {
        let mut argv = vec![
            "ferrotensor",
            "finetune",
            "--model",
            "m",
            "--data",
            "d.jsonl",
        ];
        argv.extend_from_slice(extra);
        let matches = Cli::command()
            .try_get_matches_from(argv)
            .expect("parse finetune");
        let user_set = UserSet::from_matches(
            matches
                .subcommand_matches("finetune")
                .expect("finetune matches"),
        );
        match Cli::from_arg_matches(&matches).expect("cli").command {
            Command::Finetune(args) => (args.train, user_set),
            _ => panic!("expected finetune"),
        }
    }

    fn config(batch_size: usize, seq_len: usize, checkpointing: Checkpointing) -> TrainConfig {
        TrainConfig {
            batch_size,
            grad_accum: plan::grad_accum_for(batch_size),
            seq_len,
            lora_rank: 8,
            precision: Precision::F16,
            checkpointing,
        }
    }

    #[test]
    fn flags_left_at_their_defaults_are_not_user_set() {
        let (_, user_set) = parse(&[]);
        assert_eq!(user_set, UserSet::default());
        // Given at the default value still counts as set.
        let (_, user_set) = parse(&["--batch-size", "1", "--seq-len", "256"]);
        assert_eq!(
            user_set,
            UserSet {
                batch_size: true,
                grad_accum: false,
                seq_len: true,
            }
        );
    }

    #[test]
    fn unset_fields_take_the_plan_values() {
        let (mut args, user_set) = parse(&[]);
        let settings = apply_plan(&mut args, user_set, &config(2, 128, Checkpointing::Blocks));
        assert_eq!(
            (
                args.batch_size,
                args.grad_accum,
                args.data.seq_len,
                args.checkpoint_blocks
            ),
            (2, 4, 128, true)
        );
        assert_eq!(
            settings,
            [
                "--batch-size 2 (plan)",
                "--grad-accum 4 (plan)",
                "--seq-len 128 (plan)",
                "--checkpoint-blocks on (plan)",
            ]
        );
    }

    #[test]
    fn user_set_fields_survive_the_plan() {
        let (mut args, user_set) = parse(&[
            "--batch-size",
            "3",
            "--grad-accum",
            "5",
            "--checkpoint-blocks",
        ]);
        let settings = apply_plan(&mut args, user_set, &config(8, 256, Checkpointing::Off));
        assert_eq!(
            (
                args.batch_size,
                args.grad_accum,
                args.data.seq_len,
                args.checkpoint_blocks
            ),
            (3, 5, 256, true)
        );
        assert_eq!(
            settings,
            [
                "--batch-size 3 (user)",
                "--grad-accum 5 (user)",
                "--seq-len 256 (plan)",
                "--checkpoint-blocks on (user)",
            ]
        );
        // A user batch size without --grad-accum keeps 8 sequences per step.
        let (mut args, user_set) = parse(&["--batch-size", "2"]);
        apply_plan(&mut args, user_set, &config(2, 256, Checkpointing::Off));
        assert_eq!((args.batch_size, args.grad_accum), (2, 4));
    }

    #[test]
    fn user_set_values_are_pinned_for_the_chooser() {
        let (args, user_set) = parse(&["--seq-len", "128", "--batch-size", "2"]);
        let pins = pinned(&args, user_set).expect("pinned");
        assert_eq!(
            pins,
            Pinned {
                seq_len: NonZeroUsize::new(128),
                batch_size: NonZeroUsize::new(2),
                checkpointing: None,
            }
        );
        // SmolLM2-135M on 100,000 MiB would get seq 256 and batch 8 unpinned.
        let shape = plan::ModelShape::documented()[0].clone();
        let chosen = plan::choose(&shape, 100_000, Precision::F16, 8, pins).config;
        assert_eq!(
            (chosen.seq_len, chosen.batch_size, chosen.checkpointing),
            (128, 2, Checkpointing::Off)
        );
        let (args, _) = parse(&["--checkpoint-blocks"]);
        assert_eq!(
            pinned(&args, UserSet::default()).expect("pinned"),
            Pinned {
                checkpointing: Some(Checkpointing::Blocks),
                ..Pinned::default()
            }
        );
        let (args, user_set) = parse(&["--seq-len", "0"]);
        assert_eq!(
            pinned(&args, user_set).map_err(|e| e.to_string()),
            Err("--seq-len must be positive".to_string())
        );
    }

    #[test]
    fn finetune_summary_puts_the_plan_next_to_the_train_fields() {
        let summary = FinetuneSummary {
            chosen: plan::chosen_json(&config(2, 128, Checkpointing::Blocks)),
            grad_accum: 4,
            budget_mib: 4096,
            estimated_peak_mib: 3000,
            fits: true,
            train: TrainSummary {
                steps: 2,
                first_loss: Some(4.5),
                final_loss: 4.25,
                eval_loss_before: None,
                eval_loss_after: None,
                measured_peak_mib: 2900,
                seconds: 1.5,
                mean_input_tokens_per_step: None,
                out_dir: "runs/a".into(),
                adapter: Some("runs/a/lora_adapters.safetensors".into()),
                q4_weights: None,
            },
        };
        assert_eq!(
            serde_json::to_value(&summary).expect("serialize"),
            serde_json::json!({
                "chosen": {
                    "batch_size": 2,
                    "seq_len": 128,
                    "rank": 8,
                    "precision": "f16",
                    "checkpoint_blocks": true,
                },
                "grad_accum": 4,
                "budget_mib": 4096,
                "estimated_peak_mib": 3000,
                "fits": true,
                "steps": 2,
                "first_loss": 4.5,
                "final_loss": 4.25,
                "eval_loss_before": null,
                "eval_loss_after": null,
                "measured_peak_mib": 2900,
                "seconds": 1.5,
                "mean_input_tokens_per_step": null,
                "out_dir": "runs/a",
                "adapter": "runs/a/lora_adapters.safetensors",
            })
        );
    }
}
