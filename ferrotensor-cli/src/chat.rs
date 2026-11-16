//! `ferrotensor chat`: a multi-turn ChatML conversation with a model, one
//! user message per line of stdin.

use crate::{
    adapter_metadata, build_model, generate_tokens, load_base, loaded_line, restore_adapter,
    stream_text, unmet, DecodeCache, GenerateSummary, ModelArgs, Prompt, Result, SamplingArgs,
};
use clap::Args;
use ferrotensor_loader::Tokenizer;
use std::io::{BufRead as _, IsTerminal as _};
use std::path::PathBuf;

/// Options for `ferrotensor chat`.
#[derive(Args, Debug)]
pub struct ChatArgs {
    #[command(flatten)]
    model: ModelArgs,
    /// LoRA adapter to apply: a Ferrotensor file or a PEFT adapter directory.
    #[arg(long, value_name = "FILE")]
    lora: Option<PathBuf>,
    /// Optional system message; it is kept when old turns are dropped.
    #[arg(long)]
    system: Option<String>,
    #[command(flatten)]
    sampling: SamplingArgs,
    /// Stream each reply to stderr and print one JSON object per reply to
    /// stdout.
    #[arg(long)]
    pub json: bool,
    /// Run the whole conversation through the model every turn, with a new
    /// KV cache; tests compare it with the kept cache.
    #[arg(long, hide = true)]
    rerun_conversation: bool,
}

/// Runs the conversation until end of input, `/exit` or `/quit`.
pub async fn run(args: ChatArgs) -> Result<()> {
    let base = load_base(&args.model).await?;
    let tokenizer = &base.tokenizer;
    if !["<|im_start|>", "<|im_end|>"]
        .iter()
        .all(|token| tokenizer.vocab.contains_key(*token))
    {
        return Err(unmet(
            "chat supports ChatML models only for now: the tokenizer has no <|im_start|> and <|im_end|> tokens; use generate --raw",
        ));
    }
    let meta = args.lora.as_deref().map(adapter_metadata).transpose()?;
    let model = build_model(&base, meta.as_ref(), false)?;
    if let Some(path) = &args.lora {
        let n = restore_adapter(&base, &model, path)?;
        eprintln!("{}", loaded_line(path, n, meta.as_ref()));
    }
    let mut sampler = args.sampling.sampler()?;
    let max_new = args.sampling.max_new_tokens;
    let system = args.system.as_deref();
    let interactive = std::io::stdin().is_terminal();
    if interactive {
        eprintln!("/reset clears the conversation, /exit ends it");
    }
    let mut turns: Vec<String> = Vec::new();
    let mut cache: Option<DecodeCache> = None;
    let mut said_dropped = false;
    let mut lines = std::io::stdin().lock().lines();
    loop {
        if interactive {
            eprint!("> ");
        }
        let Some(line) = lines.next().transpose()? else {
            break;
        };
        match line.trim() {
            "" => continue,
            "/exit" | "/quit" => break,
            "/reset" => {
                turns.clear();
                continue;
            }
            _ => {}
        }
        turns.push(line);
        let mut prompt_ids = tokenizer.encode_conversation(system, &turns)?;
        if let Some(limit) = base.config.max_position_embeddings {
            let budget = limit.saturating_sub(max_new);
            if prompt_ids.len() > budget {
                drop_oldest_exchanges(tokenizer, system, &mut turns, budget)?.ok_or_else(|| {
                    unmet(format!(
                        "the message does not fit the model's {limit} positions with --max-new-tokens {max_new}"
                    ))
                })?;
                prompt_ids = tokenizer.encode_conversation(system, &turns)?;
                if !said_dropped {
                    eprintln!("dropped the oldest turns to fit the model's {limit} positions");
                    said_dropped = true;
                }
            }
        }
        let prompt = Prompt::chatml(&base.tokenizer, &base.config, prompt_ids);
        if args.rerun_conversation || cache.is_none() {
            cache = DecodeCache::new(&model, prompt.ids.len() + max_new);
        }
        let mut reply = String::new();
        let decoded = generate_tokens(
            &base,
            &model,
            &prompt,
            max_new,
            &mut sampler,
            cache.as_mut(),
            |delta| {
                reply.push_str(delta);
                stream_text(delta, args.json);
            },
        )
        .await?;
        stream_text("\n", args.json);
        let kv_cache = cache.is_some();
        let summary = GenerateSummary::new(reply, prompt.ids.len(), &decoded, kv_cache);
        eprintln!("{}", summary.stats_line());
        if args.json {
            let line = serde_json::json!({
                "reply": summary.text,
                "prompt_tokens": summary.prompt_tokens,
                "cached_tokens": decoded.cached_tokens,
                "new_tokens": summary.new_tokens,
                "tokens_per_second": summary.tokens_per_second,
                "prefill_seconds": summary.prefill_seconds,
            });
            println!("{line}");
        }
        turns.push(summary.text);
    }
    Ok(())
}

/// Drops the oldest exchanges (a user turn and its reply) from `turns`, which
/// ends with the new user message, until the conversation fits `budget`
/// tokens. `None`, with `turns` unchanged, when the new message does not fit
/// even alone.
fn drop_oldest_exchanges(
    tokenizer: &Tokenizer,
    system: Option<&str>,
    turns: &mut Vec<String>,
    budget: usize,
) -> Result<Option<()>> {
    // Encodings of growing prefixes give the token count of the system turn
    // and of each exchange; special tokens split ChatML turns, so the counts
    // add up.
    let ends = (0..turns.len()).step_by(2).chain([turns.len()]);
    let lens: Vec<usize> = ends
        .map(|end| {
            tokenizer
                .encode_conversation(system, &turns[..end])
                .map(|ids| ids.len())
        })
        .collect::<std::result::Result<_, _>>()?;
    let exchanges: Vec<usize> = lens.windows(2).map(|w| w[1].saturating_sub(w[0])).collect();
    let Some(drop) = exchanges_to_drop(lens[0], &exchanges, budget) else {
        return Ok(None);
    };
    turns.drain(..2 * drop);
    Ok(Some(()))
}

/// How many of the oldest exchanges to drop so that `fixed` tokens plus the
/// remaining `exchanges` fit in `budget`. The last exchange is never
/// dropped; `None` when it does not fit even alone.
fn exchanges_to_drop(fixed: usize, exchanges: &[usize], budget: usize) -> Option<usize> {
    let mut total = fixed + exchanges.iter().sum::<usize>();
    for (dropped, &count) in exchanges.iter().enumerate() {
        if total <= budget {
            return Some(dropped);
        }
        total -= count;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Cli, Command};
    use clap::Parser;

    #[test]
    fn keeps_every_exchange_that_fits() {
        assert_eq!(exchanges_to_drop(10, &[30, 30, 20], 100), Some(0));
        assert_eq!(exchanges_to_drop(10, &[30, 30, 20], 90), Some(0));
    }

    #[test]
    fn drops_the_oldest_exchanges_first() {
        assert_eq!(exchanges_to_drop(10, &[30, 30, 20], 89), Some(1));
        assert_eq!(exchanges_to_drop(10, &[30, 30, 20], 60), Some(1));
        assert_eq!(exchanges_to_drop(10, &[30, 30, 20], 59), Some(2));
        assert_eq!(exchanges_to_drop(10, &[30, 30, 20], 30), Some(2));
    }

    #[test]
    fn a_message_that_does_not_fit_alone_is_refused() {
        assert_eq!(exchanges_to_drop(10, &[30, 30, 20], 29), None);
        assert_eq!(exchanges_to_drop(10, &[25], 34), None);
    }

    #[test]
    fn generate_and_chat_share_the_sampling_flags() -> Result<()> {
        let set = [
            "--max-new-tokens",
            "9",
            "--temperature",
            "0.5",
            "--top-k",
            "7",
            "--top-p",
            "0.8",
            "--seed",
            "3",
        ];
        let defaults = (128, 0.7, 50, 0.9, None);
        for command in ["generate", "chat"] {
            for (flags, want) in [(&set[..], (9, 0.5, 7, 0.8, Some(3))), (&[][..], defaults)] {
                let args = ["ferrotensor", command, "--model", "m"].into_iter();
                let cli = Cli::try_parse_from(args.chain(flags.iter().copied()))?;
                let s = match cli.command {
                    Command::Generate(g) => g.sampling,
                    Command::Chat(c) => c.sampling,
                    _ => return Err(format!("{command} parsed as another command").into()),
                };
                let got = (s.max_new_tokens, s.temperature, s.top_k, s.top_p, s.seed);
                assert_eq!(got, want, "{command} {flags:?}");
            }
        }
        Ok(())
    }
}
