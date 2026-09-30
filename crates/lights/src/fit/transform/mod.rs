pub mod cdf97;
pub mod haar;
pub mod weight;

use crate::error::{LightsError, Result};
use crate::guard::unreachable;
use crate::image::FloatImage;
use crate::numerics::vector::Rgb;
use crate::fit::transform::weight::SubbandWeights;

pub const CHANNELS: usize = 3;
pub const MIN_COARSE_SIDE: u32 = 4;
pub const MAX_LEVELS: u32 = 24;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Transform {
    Haar,
    Cdf97,
}

impl Transform {
    pub fn name(self) -> &'static str {
        match self {
            Transform::Haar => "haar",
            Transform::Cdf97 => "cdf97",
        }
    }

    pub fn orthonormal(self) -> bool {
        match self {
            Transform::Haar => true,
            Transform::Cdf97 => false,
        }
    }

    pub fn forward_1d(self, buf: &mut [f64]) {
        match self {
            Transform::Haar => haar::forward(buf),
            Transform::Cdf97 => cdf97::forward(buf),
        }
    }

    pub fn inverse_1d(self, buf: &mut [f64]) {
        match self {
            Transform::Haar => haar::inverse(buf),
            Transform::Cdf97 => cdf97::inverse(buf),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Subband {
    Approximation,
    DetailX,
    DetailY,
    DetailXY,
}

impl Subband {
    pub fn name(self) -> &'static str {
        match self {
            Subband::Approximation => "approximation",
            Subband::DetailX => "detail-x",
            Subband::DetailY => "detail-y",
            Subband::DetailXY => "detail-xy",
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Band {
    pub level: u32,
    pub subband: Subband,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Ranking {
    Weighted,
    RawMagnitude,
}

impl Ranking {
    pub fn name(self) -> &'static str {
        match self {
            Ranking::Weighted => "subband-weighted",
            Ranking::RawMagnitude => "raw-magnitude",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Coefficients {
    pub transform: Transform,
    pub width: u32,
    pub height: u32,
    pub levels: u32,
    pub planes: Vec<Vec<f64>>,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Retained {
    pub channel: usize,
    pub x: u32,
    pub y: u32,
    pub band: Band,
    pub value: f64,
    pub key: f64,
}

pub fn max_levels(width: u32, height: u32) -> u32 {
    let mut best = 0u32;
    for level in 1..=MAX_LEVELS {
        let step = 1u32 << level;
        if width % step != 0 || height % step != 0 {
            break;
        }
        if width / step < MIN_COARSE_SIDE || height / step < MIN_COARSE_SIDE {
            break;
        }
        best = level;
    }
    best
}

pub fn expressible(width: u32, height: u32) -> bool {
    max_levels(width, height) > 0
}

pub fn level_sizes(width: u32, height: u32, levels: u32) -> Vec<(u32, u32)> {
    let mut out = Vec::with_capacity(levels as usize);
    for level in 0..levels {
        out.push((width >> level, height >> level));
    }
    out
}

pub fn validate(width: u32, height: u32, levels: u32) -> Result<()> {
    if width == 0 || height == 0 {
        return Err(LightsError::ViewportZero { width, height });
    }
    if levels == 0 || levels > max_levels(width, height) {
        let smallest = if levels >= MAX_LEVELS {
            u32::MAX
        } else {
            MIN_COARSE_SIDE << levels
        };
        return Err(LightsError::SizeMismatch {
            a: (width, height),
            b: (smallest, smallest),
        });
    }
    Ok(())
}

pub fn band_of(width: u32, height: u32, levels: u32, x: u32, y: u32) -> Band {
    if x >= width || y >= height {
        unreachable(
            "fit::transform::band_of",
            &format!("coefficient ({x},{y}) outside {width}x{height}"),
        );
    }
    for level in 1..=levels {
        let half_width = width >> level;
        let half_height = height >> level;
        if x >= half_width || y >= half_height {
            let subband = if x >= half_width && y < half_height {
                Subband::DetailX
            } else if x < half_width && y >= half_height {
                Subband::DetailY
            } else {
                Subband::DetailXY
            };
            return Band { level, subband };
        }
    }
    Band {
        level: levels,
        subband: Subband::Approximation,
    }
}

pub fn bands(levels: u32) -> Vec<Band> {
    let mut out = Vec::with_capacity(3 * (levels as usize) + 1);
    for level in 1..=levels {
        out.push(Band { level, subband: Subband::DetailX });
        out.push(Band { level, subband: Subband::DetailY });
        out.push(Band { level, subband: Subband::DetailXY });
    }
    out.push(Band { level: levels, subband: Subband::Approximation });
    out.sort();
    out
}

pub fn band_rect(width: u32, height: u32, levels: u32, band: Band) -> (u32, u32, u32, u32) {
    if band.level == 0 || band.level > levels {
        unreachable(
            "fit::transform::band_rect",
            &format!("level {} is not one of the {levels} decomposed levels", band.level),
        );
    }
    let half_width = width >> band.level;
    let half_height = height >> band.level;
    match band.subband {
        Subband::Approximation => {
            if band.level != levels {
                unreachable(
                    "fit::transform::band_rect",
                    &format!(
                        "an approximation band exists only at the coarsest level {levels}, not {}",
                        band.level
                    ),
                );
            }
            (0, 0, half_width, half_height)
        }
        Subband::DetailX => (half_width, 0, half_width * 2, half_height),
        Subband::DetailY => (0, half_height, half_width, half_height * 2),
        Subband::DetailXY => (half_width, half_height, half_width * 2, half_height * 2),
    }
}

pub fn band_representative(width: u32, height: u32, levels: u32, band: Band) -> (u32, u32) {
    let (x0, y0, x1, y1) = band_rect(width, height, levels, band);
    (x0 + (x1 - x0) / 2, y0 + (y1 - y0) / 2)
}

pub fn zeroed(transform: Transform, width: u32, height: u32, levels: u32) -> Coefficients {
    let cells = (width as usize) * (height as usize);
    Coefficients {
        transform,
        width,
        height,
        levels,
        planes: vec![vec![0.0f64; cells]; CHANNELS],
    }
}

impl Coefficients {
    pub fn cells(&self) -> usize {
        (self.width as usize) * (self.height as usize)
    }

    pub fn terms(&self) -> usize {
        CHANNELS * self.cells()
    }

    fn offset(&self, x: u32, y: u32) -> usize {
        if x >= self.width || y >= self.height {
            unreachable(
                "fit::transform::Coefficients::offset",
                &format!("coefficient ({x},{y}) outside {}x{}", self.width, self.height),
            );
        }
        (y as usize) * (self.width as usize) + (x as usize)
    }

    pub fn get(&self, channel: usize, x: u32, y: u32) -> f64 {
        if channel >= CHANNELS {
            unreachable(
                "fit::transform::Coefficients::get",
                &format!("channel {channel} outside the {CHANNELS} colour planes"),
            );
        }
        self.planes[channel][self.offset(x, y)]
    }

    pub fn set(&mut self, channel: usize, x: u32, y: u32, value: f64) {
        if channel >= CHANNELS {
            unreachable(
                "fit::transform::Coefficients::set",
                &format!("channel {channel} outside the {CHANNELS} colour planes"),
            );
        }
        let o = self.offset(x, y);
        self.planes[channel][o] = value;
    }

    pub fn energy(&self) -> f64 {
        let mut acc = 0.0f64;
        for plane in self.planes.iter() {
            for v in plane.iter() {
                acc += v * v;
            }
        }
        acc
    }

    pub fn band_energy(&self, band: Band) -> f64 {
        let mut acc = 0.0f64;
        for plane in self.planes.iter() {
            for y in 0..self.height {
                for x in 0..self.width {
                    if band_of(self.width, self.height, self.levels, x, y) == band {
                        let v = plane[(y as usize) * (self.width as usize) + (x as usize)];
                        acc += v * v;
                    }
                }
            }
        }
        acc
    }

    pub fn nonzero(&self) -> usize {
        let mut count = 0usize;
        for plane in self.planes.iter() {
            for v in plane.iter() {
                if *v != 0.0 {
                    count += 1;
                }
            }
        }
        count
    }
}

pub fn forward(transform: Transform, image: &FloatImage<f64>, levels: u32) -> Result<Coefficients> {
    validate(image.width, image.height, levels)?;
    for p in image.pixels.iter() {
        if !p.x.is_finite() || !p.y.is_finite() || !p.z.is_finite() {
            return Err(LightsError::NonFinite {
                what: "fit::transform::forward input",
            });
        }
    }
    let width = image.width;
    let height = image.height;
    let stride = width as usize;
    let mut planes = vec![vec![0.0f64; stride * (height as usize)]; CHANNELS];
    for (i, p) in image.pixels.iter().enumerate() {
        planes[0][i] = p.x;
        planes[1][i] = p.y;
        planes[2][i] = p.z;
    }
    for (active_width, active_height) in level_sizes(width, height, levels) {
        for plane in planes.iter_mut() {
            let mut row = vec![0.0f64; active_width as usize];
            for y in 0..active_height {
                let base = (y as usize) * stride;
                row.copy_from_slice(&plane[base..base + (active_width as usize)]);
                transform.forward_1d(&mut row);
                plane[base..base + (active_width as usize)].copy_from_slice(&row);
            }
            let mut column = vec![0.0f64; active_height as usize];
            for x in 0..active_width {
                for y in 0..active_height {
                    column[y as usize] = plane[(y as usize) * stride + (x as usize)];
                }
                transform.forward_1d(&mut column);
                for y in 0..active_height {
                    plane[(y as usize) * stride + (x as usize)] = column[y as usize];
                }
            }
        }
    }
    Ok(Coefficients {
        transform,
        width,
        height,
        levels,
        planes,
    })
}

pub fn inverse(coefficients: &Coefficients) -> Result<FloatImage<f64>> {
    validate(coefficients.width, coefficients.height, coefficients.levels)?;
    if coefficients.planes.len() != CHANNELS {
        unreachable(
            "fit::transform::inverse",
            &format!("{} colour planes, expected {CHANNELS}", coefficients.planes.len()),
        );
    }
    let width = coefficients.width;
    let height = coefficients.height;
    let stride = width as usize;
    let mut planes = coefficients.planes.clone();
    let sizes = level_sizes(width, height, coefficients.levels);
    for (active_width, active_height) in sizes.iter().rev() {
        let active_width = *active_width;
        let active_height = *active_height;
        for plane in planes.iter_mut() {
            let mut column = vec![0.0f64; active_height as usize];
            for x in 0..active_width {
                for y in 0..active_height {
                    column[y as usize] = plane[(y as usize) * stride + (x as usize)];
                }
                coefficients.transform.inverse_1d(&mut column);
                for y in 0..active_height {
                    plane[(y as usize) * stride + (x as usize)] = column[y as usize];
                }
            }
            let mut row = vec![0.0f64; active_width as usize];
            for y in 0..active_height {
                let base = (y as usize) * stride;
                row.copy_from_slice(&plane[base..base + (active_width as usize)]);
                coefficients.transform.inverse_1d(&mut row);
                plane[base..base + (active_width as usize)].copy_from_slice(&row);
            }
        }
    }
    let mut out = FloatImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let o = (y as usize) * stride + (x as usize);
            out.set(x, y, Rgb::new(planes[0][o], planes[1][o], planes[2][o]));
        }
    }
    Ok(out)
}

pub fn rank_coefficients(
    coefficients: &Coefficients,
    weights: &SubbandWeights,
    ranking: Ranking,
) -> Result<Vec<Retained>> {
    if weights.transform != coefficients.transform
        || weights.width != coefficients.width
        || weights.height != coefficients.height
        || weights.levels != coefficients.levels
    {
        return Err(LightsError::SizeMismatch {
            a: (coefficients.width, coefficients.height),
            b: (weights.width, weights.height),
        });
    }
    let width = coefficients.width;
    let height = coefficients.height;
    let levels = coefficients.levels;
    let mut out = Vec::with_capacity(coefficients.terms());
    for channel in 0..CHANNELS {
        for y in 0..height {
            for x in 0..width {
                let value = coefficients.planes[channel][(y as usize) * (width as usize) + (x as usize)];
                let band = band_of(width, height, levels, x, y);
                let scale = match ranking {
                    Ranking::Weighted => weights.of(band),
                    Ranking::RawMagnitude => 1.0,
                };
                out.push(Retained {
                    channel,
                    x,
                    y,
                    band,
                    value,
                    key: value.abs() * scale,
                });
            }
        }
    }
    out.sort_unstable_by(|a, b| {
        b.key
            .total_cmp(&a.key)
            .then(a.band.cmp(&b.band))
            .then(a.y.cmp(&b.y))
            .then(a.x.cmp(&b.x))
            .then(a.channel.cmp(&b.channel))
    });
    Ok(out)
}

pub fn keep_largest(
    coefficients: &Coefficients,
    weights: &SubbandWeights,
    ranking: Ranking,
    terms: usize,
) -> Result<Coefficients> {
    let ranked = rank_coefficients(coefficients, weights, ranking)?;
    let mut kept = zeroed(
        coefficients.transform,
        coefficients.width,
        coefficients.height,
        coefficients.levels,
    );
    for entry in ranked.iter().take(terms) {
        kept.set(entry.channel, entry.x, entry.y, entry.value);
    }
    Ok(kept)
}

pub fn truncate_and_reconstruct(
    transform: Transform,
    image: &FloatImage<f64>,
    levels: u32,
    terms: usize,
    ranking: Ranking,
) -> Result<FloatImage<f64>> {
    let coefficients = forward(transform, image, levels)?;
    let weights = weight::measure(transform, image.width, image.height, levels)?;
    let kept = keep_largest(&coefficients, &weights, ranking, terms)?;
    inverse(&kept)
}
