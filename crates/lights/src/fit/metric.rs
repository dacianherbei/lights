use crate::error::{LightsError, Result};
use crate::fit::coverage::{CoverageError, coverage_image_to_colour, exact_coverage};
use crate::fit::target::Target;
use crate::guard::Budget;
use crate::image::{FloatImage, Image};
use crate::numerics::vector::Rgb;
use crate::numerics::float::FloatingPoint;

pub const EXACTNESS_LIMIT: f64 = 1.0e-4;
pub const PEAK: f64 = 1.0;
pub const LN_10: f64 = 2.302_585_092_994_046;
pub const DECIBEL_GAIN: f64 = 10.0;
pub const RASTER_LEVELS: u32 = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Provenance {
    Quadrature,
    Raster { levels: u32 },
}

#[derive(Clone, Debug)]
pub struct ExactReference<T> {
    image: FloatImage<T>,
    residual: CoverageError<T>,
    undecided_fraction: T,
    provenance: Provenance,
}

impl<T: FloatingPoint> ExactReference<T> {
    pub fn from_target(target: &Target<T>, max_depth: u32, budget: &mut Budget) -> Result<Self> {
        let (coverage, residual) = exact_coverage(target, max_depth, budget)?;
        let image = coverage_image_to_colour(target, &coverage);
        let area = T::from_f64((image.width as f64) * (image.height as f64));
        if !(area > T::ZERO) {
            return Err(LightsError::ViewportZero {
                width: image.width,
                height: image.height,
            });
        }
        let undecided_fraction = residual.undecided_area / area;
        let limit = T::from_f64(EXACTNESS_LIMIT);
        if !(undecided_fraction <= limit) {
            return Err(LightsError::ReferenceNotExact {
                fraction: undecided_fraction.to_f64(),
                limit: EXACTNESS_LIMIT,
            });
        }
        Ok(Self {
            image,
            residual,
            undecided_fraction,
            provenance: Provenance::Quadrature,
        })
    }

    pub fn from_raster(raster: &Image) -> Result<Self> {
        if raster.width == 0 || raster.height == 0 {
            return Err(LightsError::ViewportZero {
                width: raster.width,
                height: raster.height,
            });
        }
        let expected = (raster.width as usize) * (raster.height as usize);
        if raster.pixels.len() != expected {
            return Err(LightsError::MetricUndefined {
                what: "a raster whose pixel count is not width times height is not an image",
            });
        }
        let top = f64::from(RASTER_LEVELS - 1);
        let decode = |c: u8| T::from_f64(f64::from(c) / top);
        let mut image = FloatImage::<T>::new(raster.width, raster.height);
        for (out, p) in image.pixels.iter_mut().zip(raster.pixels.iter()) {
            *out = Rgb::new(decode(p[0]), decode(p[1]), decode(p[2]));
        }
        Ok(Self {
            image,
            residual: CoverageError {
                undecided_area: T::ZERO,
                max_depth_reached: 0,
                cells_visited: 0,
            },
            undecided_fraction: T::ZERO,
            provenance: Provenance::Raster {
                levels: RASTER_LEVELS,
            },
        })
    }

    pub fn provenance(&self) -> Provenance {
        self.provenance
    }

    pub fn quantisation_floor(&self) -> T {
        match self.provenance {
            Provenance::Quadrature => T::ZERO,
            Provenance::Raster { levels } => {
                let step = 1.0 / f64::from(levels - 1);
                T::from_f64(step * step / 12.0)
            }
        }
    }

    pub fn image(&self) -> &FloatImage<T> {
        &self.image
    }

    pub fn residual(&self) -> CoverageError<T> {
        self.residual
    }

    pub fn undecided_fraction(&self) -> T {
        self.undecided_fraction
    }
}

pub fn mse<T: FloatingPoint>(candidate: &FloatImage<T>, reference: &ExactReference<T>) -> Result<T> {
    let difference = candidate.difference(reference.image())?;
    let samples = difference.pixels.len() * 3;
    if samples == 0 {
        return Err(LightsError::MetricUndefined {
            what: "an empty image has no mean squared error",
        });
    }
    let mut acc = T::ZERO;
    for p in difference.pixels.iter() {
        acc += p.x * p.x + p.y * p.y + p.z * p.z;
    }
    Ok(acc / T::from_f64(samples as f64))
}

pub fn psnr<T: FloatingPoint>(
    candidate: &FloatImage<T>,
    reference: &ExactReference<T>,
) -> Result<T> {
    let error = mse(candidate, reference)?;
    if !(error > T::ZERO) {
        return Err(LightsError::MetricUndefined {
            what: "a candidate identical to the reference has no finite ratio in decibels",
        });
    }
    let floor = reference.quantisation_floor();
    if error < floor {
        return Err(LightsError::BeyondQuantisation {
            mean_square: error.to_f64(),
            floor: floor.to_f64(),
        });
    }
    let peak = T::from_f64(PEAK);
    let ratio = peak * peak / error;
    Ok(T::from_f64(DECIBEL_GAIN / LN_10) * ratio.ln())
}
