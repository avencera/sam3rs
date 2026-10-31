//! Downloads the SAM 3 checkpoint from the gated facebook/sam3 Hugging Face repo
//!
//! Meta gates the repo: each user accepts the SAM License on the model page, then
//! downloads with their own access token, which comes from `HF_TOKEN` or the file
//! `hf auth login` writes. Files go into the standard hub cache

use std::path::{Path, PathBuf};

use eyre::{Result, eyre};
use hf_fetch::{Hub, HubError};

/// Hugging Face repo of the official checkpoint
pub const MODEL_REPO: &str = "facebook/sam3";

/// Files [`crate::Sam3::load`] reads from the model directory
pub const MODEL_FILES: [&str; 3] = ["vocab.json", "merges.txt", "model.safetensors"];

/// Downloads [`MODEL_FILES`] from [`MODEL_REPO`] into the Hugging Face cache, or finds
/// them there, and returns the directory holding them
///
/// `model.safetensors` is 3.4 GB. Fails with instructions when the token is missing or
/// its account hasn't accepted Meta's terms for the repo
pub fn download_model() -> Result<PathBuf> {
    let hub = Hub::from_env().map_err(|e| eyre!("set up Hugging Face downloads: {e}"))?;
    let repo = hub.model(MODEL_REPO, "main");
    let mut dir = None;
    for file in MODEL_FILES {
        // a cached file loads without asking the Hub whether `main` moved, so a model that
        // is already downloaded works offline and without a token
        let path = repo.get(file).map_err(|e| download_error(file, e))?;
        dir = path.parent().map(Path::to_path_buf);
    }
    dir.ok_or_else(|| eyre!("Hugging Face returned no path for {MODEL_REPO}"))
}

fn download_error(file: &str, error: HubError) -> eyre::Report {
    // the Hub answers 401 without a valid token and 403 when the token's account hasn't
    // accepted the gate
    match error.status() {
        Some(status @ (401 | 403)) => eyre!(
            "{MODEL_REPO} is gated: accept Meta's SAM License at https://huggingface.co/{MODEL_REPO} while \
             logged in, then run `hf auth login` or set HF_TOKEN to a token of that account \
             (Hugging Face answered HTTP {status} for {file})"
        ),
        _ => eyre!("download {MODEL_REPO}/{file}: {error}"),
    }
}
