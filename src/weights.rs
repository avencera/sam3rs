//! Loads the Hugging Face format `model.safetensors` (all F32) straight onto the GPU
//!
//! `weights_schema.txt` lists the name and shape of every tensor sam3rs reads. Opening a
//! checkpoint checks all of them up front, so a wrong or truncated file fails with the
//! tensors it lacks instead of a panic halfway through building a layer

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::LazyLock;

use eyre::{Result, WrapErr, bail, eyre};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};

use crate::gpu::{Gpu, Tensor};

/// One `name dim,dim,...` line per tensor, from the facebook/sam3 checkpoint
static SCHEMA: LazyLock<HashMap<&'static str, Vec<usize>>> = LazyLock::new(|| {
    include_str!("weights_schema.txt")
        .lines()
        .map(|line| {
            let (name, dims) = line.split_once(' ').expect("schema line is `name dims`");
            (name, dims.split(',').map(|d| d.parse().expect("schema dim is a number")).collect())
        })
        .collect()
});

/// Mismatches listed in full before the rest are only counted
const LISTED: usize = 10;

pub struct Weights {
    mmap: Mmap,
}

impl Weights {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).wrap_err_with(|| format!("open {}", path.display()))?;
        // the file is read-only and never truncated while the process runs
        let mmap = unsafe { Mmap::map(&file)? };
        let st = SafeTensors::deserialize(&mmap).wrap_err_with(|| format!("parse {}", path.display()))?;
        let mut problems: Vec<String> = SCHEMA
            .iter()
            .filter_map(|(name, shape)| match st.tensor(name) {
                Err(_) => Some(format!("{name} is missing")),
                Ok(view) if view.dtype() != Dtype::F32 => Some(format!("{name} is {:?}, expected F32", view.dtype())),
                Ok(view) if view.shape() != shape.as_slice() => {
                    Some(format!("{name} has shape {:?}, expected {shape:?}", view.shape()))
                }
                Ok(_) => None,
            })
            .collect();
        if !problems.is_empty() {
            problems.sort();
            let more = problems.len().saturating_sub(LISTED);
            problems.truncate(LISTED);
            if more > 0 {
                problems.push(format!("and {more} more"));
            }
            bail!("{} is not a SAM 3 checkpoint:\n  {}", path.display(), problems.join("\n  "));
        }
        Ok(Self { mmap })
    }

    fn tensors(&self) -> SafeTensors<'_> {
        SafeTensors::deserialize(&self.mmap).expect("validated in open")
    }

    /// Host copy of one tensor and its shape
    pub fn host(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        if !SCHEMA.contains_key(name) {
            bail!("weight {name} is not in weights_schema.txt");
        }
        let st = self.tensors();
        let view = st.tensor(name).map_err(|e| eyre!("weight {name}: {e}"))?;
        let data = bytemuck::pod_collect_to_vec(view.data());
        Ok((data, view.shape().to_vec()))
    }

    /// Whether the model has this tensor, such as an optional bias
    pub fn contains(&self, name: &str) -> bool {
        SCHEMA.contains_key(name)
    }

    pub fn get(&self, gpu: &Gpu, name: &str) -> Result<Tensor> {
        let (data, shape) = self.host(name)?;
        gpu.upload(data, &shape)
    }

    /// Several tensors concatenated along dim 0, such as separate q/k/v projections
    pub fn concat(&self, gpu: &Gpu, names: &[String]) -> Result<Tensor> {
        let mut data = Vec::new();
        let mut shape: Option<Vec<usize>> = None;
        for name in names {
            let (d, s) = self.host(name)?;
            data.extend_from_slice(&d);
            match &mut shape {
                None => shape = Some(s),
                Some(shape) => {
                    if shape[1..] != s[1..] {
                        bail!("cannot concat {name} {s:?} onto {shape:?}");
                    }
                    shape[0] += s[0];
                }
            }
        }
        let shape = shape.ok_or_else(|| eyre!("concat of nothing"))?;
        gpu.upload(data, &shape)
    }
}
