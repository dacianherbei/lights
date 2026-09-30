use crate::error::{LightsError, Result};
use crate::image::FloatImage;
use crate::numerics::vector::Rgb;

pub const DEFAULT_GAP: u32 = 3;
pub const DEFAULT_ZOOM: u32 = 4;

pub fn upscale(image: &FloatImage<f64>, factor: u32) -> Result<FloatImage<f64>> {
    if factor == 0 {
        return Err(LightsError::ViewportZero {
            width: factor,
            height: factor,
        });
    }
    if factor == 1 {
        return Ok(image.clone());
    }
    let width = image.width * factor;
    let height = image.height * factor;
    let mut out = FloatImage::<f64>::new(width, height);
    for y in 0..height {
        for x in 0..width {
            out.set(x, y, image.get(x / factor, y / factor));
        }
    }
    Ok(out)
}

pub fn horizontal_strip(
    panels: &[FloatImage<f64>],
    gap: u32,
    separator: Rgb<f64>,
) -> Result<FloatImage<f64>> {
    let first = match panels.first() {
        Some(p) => p,
        None => {
            return Err(LightsError::ViewportZero {
                width: 0,
                height: 0,
            });
        }
    };
    let height = first.height;
    let mut width = 0u32;
    for panel in panels.iter() {
        if panel.height != height {
            return Err(LightsError::SizeMismatch {
                a: (panel.width, panel.height),
                b: (first.width, height),
            });
        }
        width += panel.width;
    }
    let separators = (panels.len() as u32).saturating_sub(1) * gap;
    width += separators;
    if width == 0 || height == 0 {
        return Err(LightsError::ViewportZero { width, height });
    }
    let mut out = FloatImage::<f64>::filled(width, height, separator);
    let mut cursor = 0u32;
    for panel in panels.iter() {
        for y in 0..panel.height {
            for x in 0..panel.width {
                out.set(cursor + x, y, panel.get(x, y));
            }
        }
        cursor += panel.width + gap;
    }
    Ok(out)
}
