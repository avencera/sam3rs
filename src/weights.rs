//! Loads the Hugging Face format `model.safetensors` (all F32) straight onto the GPU

use std::fs::File;
use std::path::Path;

use eyre::{Result, WrapErr, bail, eyre};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};

use crate::gpu::{Gpu, Tensor};

pub struct Weights {
    mmap: Mmap,
}

impl Weights {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).wrap_err_with(|| format!("open {}", path.display()))?;
        // the file is read-only and never truncated while the process runs
        let mmap = unsafe { Mmap::map(&file)? };
        SafeTensors::deserialize(&mmap).wrap_err("parse safetensors header")?;
        Ok(Self { mmap })
    }

    fn tensors(&self) -> SafeTensors<'_> {
        SafeTensors::deserialize(&self.mmap).expect("validated in open")
    }

    /// Host copy of one tensor and its shape
    pub fn host(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let st = self.tensors();
        let view = st.tensor(name).map_err(|e| eyre!("weight {name}: {e}"))?;
        if view.dtype() != Dtype::F32 {
            bail!("weight {name} is {:?}, expected F32", view.dtype());
        }
        let data = bytemuck::pod_collect_to_vec(view.data());
        Ok((data, view.shape().to_vec()))
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tensors().tensor(name).is_ok()
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
