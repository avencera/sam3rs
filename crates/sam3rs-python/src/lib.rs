//! The `sam3rs._native` extension module behind the `sam3rs` Python package
//!
//! Kept thin: the Python layer validates and converts numpy arrays, and this module
//! moves bytes and plain lists across, releasing the GIL for every GPU call

use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard, PoisonError};

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use sam3rs::{BoxPrompt, PointPrompt, Precision, RgbImage, Sam3};

pyo3::import_exception!(sam3rs._errors, Sam3rsError);

fn error(e: eyre::Report) -> PyErr {
    Sam3rsError::new_err(format!("{e:#}"))
}

fn precision(name: &str) -> PyResult<Precision> {
    name.parse().map_err(error)
}

fn precision_name(p: Precision) -> &'static str {
    match p {
        Precision::Fp32 => "fp32",
        Precision::Bf16 => "bf16",
    }
}

/// The loaded model on one GPU
///
/// `Sam3` is `Send` but not `Sync`, so calls from several Python threads take turns
/// on the mutex, which matches the model's single CUDA stream anyway
#[pyclass(name = "Model", module = "sam3rs._native", frozen)]
struct Model(Mutex<Sam3>);

impl Model {
    fn lock(&self) -> MutexGuard<'_, Sam3> {
        // a panic mid-call leaves no Rust state half-written that later calls rely on
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Features of one encoded image, usable only with the model that encoded it
#[pyclass(name = "Embedding", module = "sam3rs._native", frozen)]
struct Embedding(sam3rs::Embedding);

#[pymethods]
impl Embedding {
    #[getter]
    fn width(&self) -> usize {
        self.0.width()
    }

    #[getter]
    fn height(&self) -> usize {
        self.0.height()
    }
}

/// Scores, `[x0, y0, x1, y1]` boxes flattened, masks as `n * height * width` bytes and
/// DETR query indices
type Detections<'py> = (Vec<f32>, Vec<f32>, Bound<'py, PyBytes>, Vec<usize>);

#[pymethods]
impl Model {
    #[staticmethod]
    fn load(py: Python<'_>, path: PathBuf, device: usize, precision_name: &str) -> PyResult<Self> {
        let p = precision(precision_name)?;
        let model = py.detach(|| Sam3::load(&path, device, p)).map_err(error)?;
        Ok(Self(Mutex::new(model)))
    }

    #[staticmethod]
    fn from_pretrained(py: Python<'_>, device: usize, precision_name: &str) -> PyResult<Self> {
        let p = precision(precision_name)?;
        let model = py.detach(|| Sam3::from_pretrained(device, p)).map_err(error)?;
        Ok(Self(Mutex::new(model)))
    }

    #[getter]
    fn precision(&self) -> &'static str {
        precision_name(self.lock().precision())
    }

    fn encode(&self, py: Python<'_>, width: usize, height: usize, rgb: &[u8]) -> PyResult<Embedding> {
        let img = RgbImage::new(width, height, rgb.to_vec()).map_err(error)?;
        let emb = py.detach(|| self.lock().encode(&img)).map_err(error)?;
        Ok(Embedding(emb))
    }

    fn encode_path(&self, py: Python<'_>, path: PathBuf) -> PyResult<Embedding> {
        let emb = py.detach(|| RgbImage::open(&path).and_then(|img| self.lock().encode(&img))).map_err(error)?;
        Ok(Embedding(emb))
    }

    /// `boxes` holds `(x0, y0, x1, y1, positive)` exemplars in pixels
    fn detect<'py>(
        &self,
        py: Python<'py>,
        emb: &Embedding,
        text: &str,
        boxes: Vec<(f32, f32, f32, f32, bool)>,
    ) -> PyResult<Detections<'py>> {
        let (w, h) = (emb.0.width(), emb.0.height());
        let boxes: Vec<BoxPrompt> = boxes
            .into_iter()
            .map(|(x0, y0, x1, y1, positive)| BoxPrompt::from_xyxy([x0, y0, x1, y1], w, h, positive))
            .collect::<eyre::Result<_>>()
            .map_err(error)?;
        let dets = py.detach(|| self.lock().detect(&emb.0, text, &boxes)).map_err(error)?;

        let scores = dets.iter().map(|d| d.score).collect();
        let bboxes = dets.iter().flat_map(|d| d.bbox).collect();
        let queries = dets.iter().map(|d| d.query).collect();
        let masks = PyBytes::new_with(py, dets.len() * w * h, |buf| {
            for (dst, d) in buf.chunks_exact_mut(w * h).zip(&dets) {
                dst.copy_from_slice(&d.mask);
            }
            Ok(())
        })?;
        Ok((scores, bboxes, masks, queries))
    }

    /// `points` holds `(x, y, positive)` clicks in pixels; returns the predicted IoU and
    /// a `height * width` byte mask
    fn segment_points<'py>(
        &self,
        py: Python<'py>,
        emb: &Embedding,
        points: Vec<(f32, f32, bool)>,
    ) -> PyResult<(f32, Bound<'py, PyBytes>)> {
        let points: Vec<PointPrompt> = points
            .into_iter()
            .map(|(x, y, positive)| PointPrompt::new(x, y, positive))
            .collect::<eyre::Result<_>>()
            .map_err(error)?;
        let m = py.detach(|| self.lock().segment_points(&emb.0, &points)).map_err(error)?;
        Ok((m.score, PyBytes::new(py, &m.mask)))
    }
}

/// Tints each `height * width` mask over the RGB image, as the CLI's `--out` does
#[pyfunction]
fn overlay<'py>(
    py: Python<'py>,
    width: usize,
    height: usize,
    rgb: &[u8],
    masks: &[u8],
) -> PyResult<Bound<'py, PyBytes>> {
    let img = RgbImage::new(width, height, rgb.to_vec()).map_err(error)?;
    let masks: Vec<&[u8]> = masks.chunks(width * height).collect();
    let out = img.overlay(&masks).map_err(error)?;
    Ok(PyBytes::new(py, out.data()))
}

/// Writes RGB bytes as PNG or JPEG, chosen by the extension
#[pyfunction]
fn save_image(py: Python<'_>, width: usize, height: usize, rgb: &[u8], path: PathBuf) -> PyResult<()> {
    let img = RgbImage::new(width, height, rgb.to_vec()).map_err(error)?;
    py.detach(|| img.save(&path)).map_err(error)
}

/// Decodes a PNG or JPEG into `(width, height, rgb bytes)`
#[pyfunction]
fn open_image<'py>(py: Python<'py>, path: PathBuf) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
    let img = py.detach(|| RgbImage::open(&path)).map_err(error)?;
    Ok((img.width(), img.height(), PyBytes::new(py, img.data())))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Model>()?;
    m.add_class::<Embedding>()?;
    m.add_function(wrap_pyfunction!(overlay, m)?)?;
    m.add_function(wrap_pyfunction!(save_image, m)?)?;
    m.add_function(wrap_pyfunction!(open_image, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("SCORE_THRESHOLD", sam3rs::SCORE_THRESHOLD)?;
    m.add("VISUAL_PROMPT", sam3rs::VISUAL_PROMPT)?;
    m.add("MODEL_REPO", sam3rs::MODEL_REPO)?;
    Ok(())
}
