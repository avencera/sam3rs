//! Box and point prompts, validated on construction

use eyre::{Result, WrapErr, ensure};

/// A box prompt in normalized cxcywh with its label (true = positive)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoxPrompt {
    cxcywh: [f32; 4],
    positive: bool,
}

impl BoxPrompt {
    /// A box from pixel corners `[x0, y0, x1, y1]` in a `width` x `height` image
    ///
    /// The box may extend past the image by up to one image size on each side
    pub fn from_xyxy(xyxy: [f32; 4], width: usize, height: usize, positive: bool) -> Result<Self> {
        ensure!(width > 0 && height > 0, "image size {width}x{height} is empty");
        ensure!(xyxy.iter().all(|v| v.is_finite()), "box {xyxy:?} has a non-finite coordinate");
        let (w, h) = (width as f32, height as f32);
        let [x0, y0, x1, y1] = xyxy;
        Self::from_cxcywh([(x0 + x1) / 2.0 / w, (y0 + y1) / 2.0 / h, (x1 - x0) / w, (y1 - y0) / h], positive)
            .wrap_err_with(|| format!("box {xyxy:?} in a {width}x{height} image"))
    }

    /// A box from its center and size as fractions of the image width and height
    ///
    /// Width and height must be positive, and the corners within `[-1, 2]`: no more than
    /// one image size past each edge. That bounds the ROI pooling work per box
    pub fn from_cxcywh(cxcywh: [f32; 4], positive: bool) -> Result<Self> {
        let [cx, cy, w, h] = cxcywh;
        ensure!(cxcywh.iter().all(|v| v.is_finite()), "box {cxcywh:?} has a non-finite value");
        ensure!(w > 0.0 && h > 0.0, "box {cxcywh:?} has no area");
        let corners = [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0];
        ensure!(
            corners.iter().all(|c| (-1.0..=2.0).contains(c)),
            "box {cxcywh:?} reaches more than one image size past the image"
        );
        Ok(Self { cxcywh, positive })
    }

    /// Center x, center y, width and height as fractions of the image size
    pub fn cxcywh(&self) -> [f32; 4] {
        self.cxcywh
    }

    /// Whether the box is a positive exemplar rather than a negative one
    pub fn positive(&self) -> bool {
        self.positive
    }
}

/// A click in original image pixels; positive = foreground
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointPrompt {
    x: f32,
    y: f32,
    positive: bool,
}

impl PointPrompt {
    /// A click at `(x, y)` in original image pixels; fails on a non-finite coordinate
    pub fn new(x: f32, y: f32, positive: bool) -> Result<Self> {
        ensure!(x.is_finite() && y.is_finite(), "point ({x}, {y}) has a non-finite coordinate");
        Ok(Self { x, y, positive })
    }

    /// Horizontal position in pixels
    pub fn x(&self) -> f32 {
        self.x
    }

    /// Vertical position in pixels
    pub fn y(&self) -> f32 {
        self.y
    }

    /// Whether the click marks foreground rather than background
    pub fn positive(&self) -> bool {
        self.positive
    }
}
