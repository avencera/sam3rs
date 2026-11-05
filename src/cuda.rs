//! The CUDA backend: cuBLAS and cuBLASLt GEMMs plus cuda-oxide kernels shipped as PTX, on
//! one stream per model

mod blaslt;
mod cublas;
pub(crate) mod detector;
pub(crate) mod gpu;
pub(crate) mod image;
pub(crate) mod ops;
pub(crate) mod pipeline;
pub(crate) mod point;
pub(crate) mod text;
pub(crate) mod video;
pub(crate) mod vision;

#[cfg(test)]
mod kernel_tests;
#[cfg(test)]
mod test_hooks;

use eyre::Result;

use crate::weights::Weights;
use gpu::{Gpu, Tensor};

pub(crate) use pipeline::Cuda;

impl Weights {
    /// One tensor uploaded to the GPU
    pub(crate) fn get(&self, gpu: &Gpu, name: &str) -> Result<Tensor> {
        let (data, shape) = self.host(name)?;
        gpu.upload(data, &shape)
    }

    /// Several tensors concatenated along dim 0 and uploaded, such as separate q/k/v
    /// projections
    pub(crate) fn concat(&self, gpu: &Gpu, names: &[String]) -> Result<Tensor> {
        let (data, shape) = self.host_concat(names)?;
        gpu.upload(data, &shape)
    }
}
