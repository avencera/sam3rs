//! Downloads the SAM 3 checkpoint from the avencera/sam3-safetensors Hugging Face repo
//!
//! The repo mirrors facebook/sam3's `model.safetensors`, `vocab.json` and `merges.txt`
//! byte for byte under Meta's SAM License, without facebook/sam3's access gate, so
//! downloads need no account or token. A token from `HF_TOKEN` or the file
//! `hf auth login` writes is still sent when present. Files go into the standard hub cache

use std::path::{Path, PathBuf};

use eyre::{Result, eyre};
use hf_fetch::{Hub, HubError};

/// Hugging Face repo the weights download from, an ungated copy of facebook/sam3
pub const MODEL_REPO: &str = "avencera/sam3-safetensors";

/// Files [`crate::Sam3::load`] reads from the model directory
pub const MODEL_FILES: [&str; 3] = ["vocab.json", "merges.txt", "model.safetensors"];

/// Downloads [`MODEL_FILES`] from [`MODEL_REPO`] into the Hugging Face cache, or finds
/// them there, and returns the directory holding them
///
/// `model.safetensors` is 3.4 GB. No Hugging Face token is needed
pub fn download_model() -> Result<PathBuf> {
    let hub = Hub::from_env().map_err(|e| eyre!("set up Hugging Face downloads: {e}"))?;
    let repo = hub.model(MODEL_REPO, "main");
    let mut dir = None;
    for file in MODEL_FILES {
        // a cached file loads without asking the Hub whether `main` moved, so a model that
        // is already downloaded works offline
        let path = repo.get(file).map_err(|e| download_error(file, e))?;
        dir = path.parent().map(Path::to_path_buf);
    }

    dir.ok_or_else(|| eyre!("Hugging Face returned no path for {MODEL_REPO}"))
}

fn download_error(file: &str, error: HubError) -> eyre::Report {
    // the repo is public, so 401 or 403 means the Hub rejected a token found in the
    // environment rather than a missing one
    match error.status() {
        Some(status @ (401 | 403)) => eyre!(
            "Hugging Face rejected the token in HF_TOKEN or `hf auth login`'s token file \
             (HTTP {status} for {MODEL_REPO}/{file}); no token is needed, so unset HF_TOKEN \
             or run `hf auth logout`, or replace it with a valid one"
        ),
        _ => eyre!(
            "download {MODEL_REPO}/{file}: {error} (check the network connection, or pass a \
             local model directory instead)"
        ),
    }
}
