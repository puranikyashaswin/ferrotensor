//! Turns a `--model` argument into a model directory: a path, or a Hugging
//! Face model id that is already in the local Hugging Face cache.

use crate::{unmet, Result};
use ferrotensor_hub::{FerrotensorHub, HubError};
use std::path::{Path, PathBuf};
use std::sync::Once;

/// The directory `--model arg` names. An existing directory wins; otherwise
/// an id such as `HuggingFaceTB/SmolLM2-135M` resolves to its cached
/// snapshot, announced once on stderr. Nothing is downloaded.
///
/// # Errors
///
/// When `arg` is no directory and either not an id, an id that is not
/// cached, or an id whose snapshot has no `config.json`.
pub fn resolve_model(arg: &Path) -> Result<PathBuf> {
    resolve_in(arg, &FerrotensorHub::new())
}

fn resolve_in(arg: &Path, hub: &FerrotensorHub) -> Result<PathBuf> {
    if arg.is_dir() {
        return Ok(arg.to_path_buf());
    }
    let Some(id) = arg.to_str().filter(|text| is_model_id(text)) else {
        return Err(unmet(format!(
            "model directory {} does not exist",
            arg.display()
        )));
    };
    match hub.find_latest_snapshot(id) {
        Ok(snapshot) if !snapshot.join("config.json").is_file() => Err(unmet(format!(
            "snapshot {} of {id} has no config.json, so the download looks incomplete; \
             run `hf download {id}` again",
            snapshot.display()
        ))),
        Ok(snapshot) => {
            // `finetune` resolves the model to plan and again to load it.
            static ANNOUNCED: Once = Once::new();
            ANNOUNCED.call_once(|| eprintln!("model: {id} ({})", snapshot.display()));
            Ok(snapshot)
        }
        Err(HubError::ModelNotFound(_)) => Err(unmet(format!(
            "model {id} is not in the Hugging Face cache {}; download it with \
             `hf download {id}` (older installs: `huggingface-cli download {id}`)",
            hub.cache_dir().display()
        ))),
        Err(e) => Err(e.into()),
    }
}

/// Whether `arg` is `org/name`, each part made of the characters Hugging Face
/// allows in a repository name.
fn is_model_id(arg: &str) -> bool {
    let allowed = |c: char| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.');
    let part_ok = |part: &str| !part.is_empty() && part.chars().all(allowed);
    !arg.starts_with('.')
        && arg
            .split_once('/')
            .is_some_and(|(org, name)| part_ok(org) && part_ok(name))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Unmet;

    #[test]
    fn ids_are_two_plain_segments() {
        for id in ["HuggingFaceTB/SmolLM2-135M", "a/b", "org_1/name.v2-x"] {
            assert!(is_model_id(id), "{id}");
        }
        for not_id in [
            "./a/b", "/a/b", "a/b/c", "a", "a/", "/a", "~/x", "a\\b", "a/b c", "../a", "", "/",
        ] {
            assert!(!is_model_id(not_id), "{not_id}");
        }
    }

    #[test]
    fn an_existing_directory_is_returned_unchanged() {
        let hub = FerrotensorHub::with_cache_dir(PathBuf::from("no-such-cache"));
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        assert_eq!(resolve_in(dir, &hub).ok().as_deref(), Some(dir));
    }

    #[test]
    fn an_uncached_id_names_the_cache_and_the_download_command() {
        let hub = FerrotensorHub::with_cache_dir(PathBuf::from("no-such-cache"));
        let err = resolve_in(Path::new("org/name"), &hub).err();
        let err = err.as_deref().map(|e| (e.is::<Unmet>(), e.to_string()));
        assert_eq!(
            err,
            Some((
                true,
                "model org/name is not in the Hugging Face cache no-such-cache; download it \
                 with `hf download org/name` (older installs: `huggingface-cli download org/name`)"
                    .to_string()
            ))
        );
    }

    #[test]
    fn a_missing_path_keeps_the_directory_error() {
        let hub = FerrotensorHub::with_cache_dir(PathBuf::from("no-such-cache"));
        let err = resolve_in(Path::new("./a/b"), &hub).err();
        assert_eq!(
            err.map(|e| e.to_string()).as_deref(),
            Some("model directory ./a/b does not exist")
        );
    }
}
