//! The Metal backend for Apple silicon: Metal 4 tensor-op (`matmul2d`) GEMMs and MSL
//! kernels compiled when a model loads, on one command queue per model

pub(crate) mod detector;
pub(crate) mod gpu;
pub(crate) mod image;
pub(crate) mod masks;
pub(crate) mod ops;
pub(crate) mod pipeline;
pub(crate) mod point;
pub(crate) mod text;
pub(crate) mod video;
pub(crate) mod vision;

#[cfg(test)]
mod kernel_tests;
#[cfg(test)]
mod stage_tests;
#[cfg(test)]
mod video_tests;

use eyre::Result;

use crate::weights::Weights;
use gpu::{DType, Gpu, Tensor};

pub use pipeline::{Embedding, Sam3};
pub use video::VideoTracker;

impl Weights {
    /// One FP32 tensor uploaded to the GPU
    pub(crate) fn get(&self, gpu: &Gpu, name: &str) -> Result<Tensor> {
        let (data, shape) = self.host(name)?;
        gpu.upload_f32(&data, &shape)
    }

    /// One tensor uploaded as FP32 or rounded to BF16 on the host
    pub(crate) fn get_as(&self, gpu: &Gpu, name: &str, dtype: DType) -> Result<Tensor> {
        let (data, shape) = self.host(name)?;
        gpu.upload_as(&data, &shape, dtype)
    }

    /// Several tensors concatenated along dim 0, such as separate q/k/v projections,
    /// uploaded as FP32 or BF16
    pub(crate) fn concat_as(&self, gpu: &Gpu, names: &[String], dtype: DType) -> Result<Tensor> {
        let (data, shape) = self.host_concat(names)?;
        gpu.upload_as(&data, &shape, dtype)
    }
}
