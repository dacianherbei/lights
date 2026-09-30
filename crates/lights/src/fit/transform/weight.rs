use crate::error::Result;
use crate::fit::transform::{Band, Transform, band_representative, bands, inverse, validate, zeroed};
use crate::guard::unreachable;

#[derive(Clone, Debug, PartialEq)]
pub struct SubbandWeights {
    pub transform: Transform,
    pub width: u32,
    pub height: u32,
    pub levels: u32,
    pub entries: Vec<(Band, f64)>,
}

impl SubbandWeights {
    pub fn of(&self, band: Band) -> f64 {
        for (candidate, norm) in self.entries.iter() {
            if *candidate == band {
                return *norm;
            }
        }
        unreachable(
            "fit::transform::weight::SubbandWeights::of",
            &format!(
                "band {:?} at level {} is not one of the {} measured subbands",
                band.subband,
                band.level,
                self.entries.len()
            ),
        );
    }

    pub fn largest(&self) -> f64 {
        let mut best = 0.0f64;
        for (_, norm) in self.entries.iter() {
            best = best.max(*norm);
        }
        best
    }

    pub fn smallest(&self) -> f64 {
        let mut best = f64::INFINITY;
        for (_, norm) in self.entries.iter() {
            best = best.min(*norm);
        }
        best
    }
}

pub fn measure(transform: Transform, width: u32, height: u32, levels: u32) -> Result<SubbandWeights> {
    validate(width, height, levels)?;
    let mut entries = Vec::new();
    for band in bands(levels) {
        let (x, y) = band_representative(width, height, levels, band);
        let mut impulse = zeroed(transform, width, height, levels);
        impulse.set(0, x, y, 1.0);
        let synthesised = inverse(&impulse)?;
        let mut acc = 0.0f64;
        for p in synthesised.pixels.iter() {
            acc += p.x * p.x;
        }
        entries.push((band, acc.sqrt()));
    }
    Ok(SubbandWeights {
        transform,
        width,
        height,
        levels,
        entries,
    })
}
