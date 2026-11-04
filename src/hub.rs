//! Downloads the SAM 3 checkpoint from the avencera/sam3-safetensors Hugging Face repo
//!
//! The repo mirrors facebook/sam3's `model.safetensors`, `vocab.json` and `merges.txt`
//! byte for byte under Meta's SAM License, without facebook/sam3's access gate, so
//! downloads need no account or token. A token from `HF_TOKEN` or the file
//! `hf auth login` writes is still sent when present. Downloads are pinned to
//! [`MODEL_REVISION`] and go into the standard hub cache, or a directory given to
//! [`ModelManager::with_cache_dir`]

use std::path::{Path, PathBuf};

use eyre::{Result, eyre};
use hf_fetch::{Hub, HubError, HubRepo};

/// Hugging Face repo the weights download from, an ungated copy of facebook/sam3
pub const MODEL_REPO: &str = "avencera/sam3-safetensors";

/// Commit of [`MODEL_REPO`] that this release loads
pub const MODEL_REVISION: &str = "deb497b5bfe05a41845368e8d94e711da78c5b65";

/// Files [`crate::Sam3::load`] reads from the model directory
pub const MODEL_FILES: [&str; 3] = ["vocab.json", "merges.txt", "model.safetensors"];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ModelRepositoryId(&'static str);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ModelRevision(&'static str);

/// A repo id bound to a commit, so the pair can't drift apart at a call site
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PinnedModelRepository {
    id: ModelRepositoryId,
    revision: ModelRevision,
}

impl PinnedModelRepository {
    const fn sam3() -> Self {
        Self { id: ModelRepositoryId(MODEL_REPO), revision: ModelRevision(MODEL_REVISION) }
    }

    fn bind(self, hub: &Hub) -> HubRepo {
        hub.model(self.id.0, self.revision.0)
    }
}

const PINNED_MODEL_REPOSITORY: PinnedModelRepository = PinnedModelRepository::sam3();

/// Downloads and caches the SAM 3 weights from [`MODEL_REPO`] at [`MODEL_REVISION`]
#[derive(Clone)]
pub struct ModelManager {
    repo: HubRepo,
}

// hand-written so the repo's HTTP client, which holds any Hugging Face token in plain
// text, never reaches logs
impl std::fmt::Debug for ModelManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelManager")
            .field("repo", &MODEL_REPO)
            .field("revision", &MODEL_REVISION)
            .finish_non_exhaustive()
    }
}

impl ModelManager {
    /// Manager for the default Hugging Face cache
    ///
    /// The cache is [`default_hub_cache_dir`](crate::default_hub_cache_dir). Downloads read
    /// `HF_ENDPOINT`, and send the token in `HF_TOKEN` or saved by `hf auth login` when
    /// there is one; none is needed
    pub fn new() -> Result<Self, HubError> {
        Ok(Self::from_hub(&Hub::from_env()?))
    }

    /// Manager for the hub cache at `cache_dir`, otherwise configured like
    /// [`ModelManager::new`]
    pub fn with_cache_dir(cache_dir: PathBuf) -> Result<Self, HubError> {
        Ok(Self::from_hub(&Hub::with_cache_dir(cache_dir)?))
    }

    fn from_hub(hub: &Hub) -> Self {
        Self { repo: PINNED_MODEL_REPOSITORY.bind(hub) }
    }

    /// Downloads one file of the repo, or finds it in the cache, and returns its path
    pub fn get(&self, filename: impl AsRef<str>) -> Result<PathBuf, HubError> {
        // a commit-hash revision makes a cached snapshot file resolve without a request,
        // so a downloaded model works offline
        self.repo.get(filename.as_ref())
    }

    /// Downloads [`MODEL_FILES`], or finds them in the cache, and returns the directory
    /// holding them, ready for [`crate::Sam3::load`]
    ///
    /// `model.safetensors` is 3.4 GB
    pub fn ensure(&self) -> Result<PathBuf, HubError> {
        let [first, rest @ ..] = MODEL_FILES;
        let first = self.get(first)?;
        for file in rest {
            self.get(file)?;
        }

        // every file lands in the same snapshot directory
        Ok(first.parent().map_or_else(|| first.clone(), Path::to_path_buf))
    }
}

/// Downloads [`MODEL_FILES`] into the default Hugging Face cache, or finds them there,
/// and returns the directory holding them
///
/// Shorthand for [`ModelManager::new`] then [`ModelManager::ensure`], with errors that
/// say what to try. `model.safetensors` is 3.4 GB. No Hugging Face token is needed
pub fn download_model() -> Result<PathBuf> {
    let manager = ModelManager::new().map_err(|e| eyre!("set up Hugging Face downloads: {e}"))?;
    manager.ensure().map_err(download_error)
}

fn download_error(error: HubError) -> eyre::Report {
    // the repo is public, so 401 means the Hub rejected a token found in the environment
    // rather than a missing one. 403 can also come from the storage server for an expired
    // download link, so it gets no token advice of its own
    match error.status() {
        Some(401) => eyre!(
            "{error}: Hugging Face rejected the token in HF_TOKEN or `hf auth login`'s token \
             file; no token is needed, so unset HF_TOKEN or run `hf auth logout`, or replace \
             it with a valid one"
        ),
        Some(403) => eyre!(
            "{error}: access to {MODEL_REPO} was refused; retry the download, and if it keeps \
             failing, unset HF_TOKEN or run `hf auth logout`, since no token is needed"
        ),
        _ => eyre!(
            "download {MODEL_REPO}: {error} (check the network connection, or pass a local \
             model directory instead)"
        ),
    }
}
