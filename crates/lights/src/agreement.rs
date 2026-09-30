#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ContractViolation {
    DilationEnabled,
    EnergyCompensationMismatch,
    AlphaCutoffMismatch,
    TransmittanceCutoffMismatch,
    AlphaCapMismatch,
    EarlyOutMismatch,
    OrderingMismatch,
    SupportModeMismatch,
    SupportNotCutoffSolved,
    NearPlaneMismatch,
    ClampedAtomsPresent { count: u64 },
    DeterminantCulledAtomsPresent { count: u64 },
    ScaleBelowFloor { index: usize },
    BlendClassNotOver { kernel: &'static str },
    KernelCurveNotGaussian { kernel: &'static str },
    LightPresent,
    UnnamedSettingsMismatch,
}

pub use crate::bound::{
    BoundTerms, FIRST_ORDER_COEFFICIENT, HEADROOM, SECOND_ORDER_COEFFICIENT, atom_bound,
    atom_bound_terms, atom_bound_terms_with_lambda, atom_bound_with_lambda,
};
use crate::dispatch::{Collect, RenderOutcome, RenderRequest, render};
use crate::error::LightsError;
use crate::image::FloatImage;
use crate::numerics::float::FloatingPoint;
use crate::numerics::vector::Rgb;
use crate::method::RenderMethod;
use crate::scene::Scene;
use crate::settings::{SharedSettings, SupportMode};

pub fn convention_divergence(opacity: f64, gaussian_response: f64) -> f64 {
    let beer = 1.0 - (1.0 - opacity).powf(gaussian_response);
    beer - opacity * gaussian_response
}

pub fn assert_contract<T: FloatingPoint>(
    a: &SharedSettings<T>,
    b: &SharedSettings<T>,
    clamped: u64,
    determinant_culled: u64,
    scene: &Scene<T>,
) -> crate::error::Result<()> {
    let fail = |v: ContractViolation| Err(LightsError::ContractViolation(v));

    if a.screen_dilation != T::ZERO || b.screen_dilation != T::ZERO {
        return fail(ContractViolation::DilationEnabled);
    }
    if a.energy_compensation || b.energy_compensation {
        return fail(ContractViolation::EnergyCompensationMismatch);
    }
    if a.blend.alpha_cutoff != b.blend.alpha_cutoff {
        return fail(ContractViolation::AlphaCutoffMismatch);
    }
    if a.blend.transmittance_cutoff != b.blend.transmittance_cutoff {
        return fail(ContractViolation::TransmittanceCutoffMismatch);
    }
    if a.blend.alpha_cap != b.blend.alpha_cap {
        return fail(ContractViolation::AlphaCapMismatch);
    }
    if a.blend.early_out != b.blend.early_out {
        return fail(ContractViolation::EarlyOutMismatch);
    }
    if a.ordering != b.ordering {
        return fail(ContractViolation::OrderingMismatch);
    }
    if a.support != b.support {
        return fail(ContractViolation::SupportModeMismatch);
    }
    if a.support != SupportMode::CutoffSolved || b.support != SupportMode::CutoffSolved {
        return fail(ContractViolation::SupportNotCutoffSolved);
    }
    if a.near_plane != b.near_plane {
        return fail(ContractViolation::NearPlaneMismatch);
    }
    if clamped > 0 {
        return fail(ContractViolation::ClampedAtomsPresent { count: clamped });
    }
    if determinant_culled > 0 {
        return fail(ContractViolation::DeterminantCulledAtomsPresent {
            count: determinant_culled,
        });
    }
    let floor = a.scale_floor.max(b.scale_floor);
    for (index, atom) in scene.atoms().iter().enumerate() {
        if atom.scale.min_component() < floor {
            return fail(ContractViolation::ScaleBelowFloor { index });
        }
    }
    if a.light.is_some() || b.light.is_some() {
        return fail(ContractViolation::LightPresent);
    }
    if a.fingerprint() != b.fingerprint() {
        return fail(ContractViolation::UnnamedSettingsMismatch);
    }
    Ok(())
}

pub fn compose_tolerance<T: FloatingPoint>(
    accumulated: &FloatImage<T>,
    headroom: T,
    colour_max: T,
    transmittance_cutoff: T,
) -> FloatImage<T> {
    let gain = headroom * T::TWO * colour_max;
    FloatImage {
        width: accumulated.width,
        height: accumulated.height,
        pixels: accumulated
            .pixels
            .iter()
            .map(|p| *p * gain + Rgb::splat(transmittance_cutoff))
            .collect(),
    }
}

pub fn compose_tolerance_with_truncation<T: FloatingPoint>(
    accumulated: &FloatImage<T>,
    truncation: &FloatImage<T>,
    headroom: T,
    colour_max: T,
    alpha_cutoff: T,
    transmittance_cutoff: T,
) -> crate::error::Result<FloatImage<T>> {
    if accumulated.width != truncation.width || accumulated.height != truncation.height {
        return Err(LightsError::SizeMismatch {
            a: (accumulated.width, accumulated.height),
            b: (truncation.width, truncation.height),
        });
    }
    let bound_gain = headroom * T::TWO * colour_max;
    let step_gain = T::TWO * colour_max * alpha_cutoff;
    Ok(FloatImage {
        width: accumulated.width,
        height: accumulated.height,
        pixels: accumulated
            .pixels
            .iter()
            .zip(truncation.pixels.iter())
            .map(|(b, s)| *b * bound_gain + *s * step_gain + Rgb::splat(transmittance_cutoff))
            .collect(),
    })
}

pub fn envelope<T: FloatingPoint>(
    left: &FloatImage<T>,
    right: &FloatImage<T>,
) -> crate::error::Result<FloatImage<T>> {
    if left.width != right.width || left.height != right.height {
        return Err(LightsError::SizeMismatch {
            a: (left.width, left.height),
            b: (right.width, right.height),
        });
    }
    Ok(FloatImage {
        width: left.width,
        height: left.height,
        pixels: left
            .pixels
            .iter()
            .zip(right.pixels.iter())
            .map(|(a, b)| Rgb::new(a.x.max(b.x), a.y.max(b.y), a.z.max(b.z)))
            .collect(),
    })
}

#[derive(Clone, Debug, PartialEq)]
pub struct AgreementReport {
    pub failures: usize,
    pub pixels: usize,
    pub worst_pixel: (u32, u32),
    pub worst_difference: f64,
    pub worst_tolerance: f64,
    pub worst_ratio: f64,
    pub tolerance_span: f64,
    pub tolerance_min: f64,
    pub tolerance_max: f64,
    pub rms_difference: f64,
    pub clamped_atoms: u64,
    pub determinant_culled_atoms: u64,
}

impl AgreementReport {
    pub fn is_clean(&self) -> bool {
        self.failures == 0
    }
}

pub fn compare(
    a: &FloatImage<f64>,
    b: &FloatImage<f64>,
    tolerance: &FloatImage<f64>,
) -> crate::error::Result<AgreementReport> {
    if a.width != b.width || a.height != b.height {
        return Err(LightsError::SizeMismatch {
            a: (a.width, a.height),
            b: (b.width, b.height),
        });
    }
    if a.width != tolerance.width || a.height != tolerance.height {
        return Err(LightsError::SizeMismatch {
            a: (a.width, a.height),
            b: (tolerance.width, tolerance.height),
        });
    }

    let mut failures = 0usize;
    let mut worst_ratio = 0.0f64;
    let mut worst_pixel = (0u32, 0u32);
    let mut worst_difference = 0.0f64;
    let mut worst_tolerance = 0.0f64;
    let mut tolerance_min = f64::INFINITY;
    let mut tolerance_max = 0.0f64;
    let mut square_sum = 0.0f64;

    for y in 0..a.height {
        for x in 0..a.width {
            let left = a.get(x, y);
            let right = b.get(x, y);
            let delta = left - right;
            let difference = delta.x.abs().max(delta.y.abs()).max(delta.z.abs());
            square_sum += delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
            let tol = tolerance.get(x, y);
            let tol = tol.x.max(tol.y).max(tol.z);
            tolerance_min = tolerance_min.min(tol);
            tolerance_max = tolerance_max.max(tol);
            if difference > tol {
                failures += 1;
            }
            let ratio = if tol > 0.0 {
                difference / tol
            } else if difference > 0.0 {
                f64::INFINITY
            } else {
                0.0
            };
            if ratio > worst_ratio {
                worst_ratio = ratio;
                worst_pixel = (x, y);
                worst_difference = difference;
                worst_tolerance = tol;
            }
        }
    }

    let pixels = (a.width as usize) * (a.height as usize);
    let rms_difference = if pixels == 0 {
        0.0
    } else {
        (square_sum / ((pixels * 3) as f64)).sqrt()
    };
    if !tolerance_min.is_finite() {
        tolerance_min = 0.0;
    }
    let tolerance_span = if tolerance_min > 0.0 {
        tolerance_max / tolerance_min
    } else {
        f64::INFINITY
    };

    Ok(AgreementReport {
        failures,
        pixels,
        worst_pixel,
        worst_difference,
        worst_tolerance,
        worst_ratio,
        tolerance_span,
        tolerance_min,
        tolerance_max,
        rms_difference,
        clamped_atoms: 0,
        determinant_culled_atoms: 0,
    })
}

pub fn run_pair(
    request: &RenderRequest,
    left: &RenderMethod,
    right: &RenderMethod,
) -> crate::error::Result<(AgreementReport, RenderOutcome, RenderOutcome, FloatImage<f64>)> {
    if !request.kernel.blend_is_over() {
        return Err(LightsError::ContractViolation(
            ContractViolation::BlendClassNotOver {
                kernel: request.kernel.name(),
            },
        ));
    }
    if !request.kernel.curve_is_gaussian() {
        return Err(LightsError::ContractViolation(
            ContractViolation::KernelCurveNotGaussian {
                kernel: request.kernel.name(),
            },
        ));
    }

    let settings = request.settings;
    assert_contract(&settings, &settings, 0, 0, &request.scene)?;

    let mut a = request.clone();
    a.method = left.clone();
    a.collect = Collect::PixelsAndTolerance;
    let mut b = request.clone();
    b.method = right.clone();
    b.collect = Collect::PixelsAndTolerance;

    let outcome_a = render(&a)?;
    let outcome_b = render(&b)?;

    let clamped = outcome_a.report.clamped_atoms + outcome_b.report.clamped_atoms;
    let determinant_culled = outcome_a.report.determinant_culled_atoms
        + outcome_b.report.determinant_culled_atoms;
    assert_contract(
        &settings,
        &settings,
        clamped,
        determinant_culled,
        &request.scene,
    )?;

    let tolerance = envelope(
        &tolerance_of(&outcome_a, &settings)?,
        &tolerance_of(&outcome_b, &settings)?,
    )?;

    let mut report = compare(&outcome_a.image, &outcome_b.image, &tolerance)?;
    report.clamped_atoms = clamped;
    report.determinant_culled_atoms = determinant_culled;
    Ok((report, outcome_a, outcome_b, tolerance))
}

fn tolerance_of(
    outcome: &RenderOutcome,
    settings: &SharedSettings<f64>,
) -> crate::error::Result<FloatImage<f64>> {
    let accumulated = match outcome.tolerance.as_ref() {
        Some(t) => t,
        None => {
            return Err(LightsError::NonFinite {
                what: "agreement: the render produced no tolerance image",
            });
        }
    };
    let truncation = match outcome.truncation.as_ref() {
        Some(t) => t,
        None => {
            return Err(LightsError::NonFinite {
                what: "agreement: the render produced no truncation image",
            });
        }
    };
    compose_tolerance_with_truncation(
        accumulated,
        truncation,
        HEADROOM,
        1.0,
        settings.blend.alpha_cutoff,
        settings.blend.transmittance_cutoff,
    )
}
