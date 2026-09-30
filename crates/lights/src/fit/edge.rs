use crate::error::{LightsError, Result};
use crate::fit::atom2d::pixel_centre;
use crate::fit::target::{EdgeProbe, FeatureId, ProbeKind, Target};
use crate::image::FloatImage;
use crate::numerics::vector::{Rgb, Vec2};

pub const BIN_WIDTH: f64 = 0.05;
pub const PLATEAU_NEAR: f64 = 1.5;
pub const PLATEAU_FAR: f64 = 3.0;
pub const THIN_BIN_SAMPLES: u32 = 4;
pub const INTERIOR_DEPTH: f64 = 3.0;
pub const LOW_LEVEL: f64 = 0.1;
pub const HIGH_LEVEL: f64 = 0.9;
pub const HALF_LEVEL: f64 = 0.5;
pub const SDF_ARM_TRUE_TRANSITION: f64 = HIGH_LEVEL - LOW_LEVEL;
pub const SDF_ARM_TRUE_FWHM: f64 = 1.0;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Criterion {
    pub max_transition: f64,
    pub min_contrast_frac: f64,
    pub min_peak_frac: f64,
    pub target_fwhm: f64,
    pub max_fwhm_error: f64,
    pub max_interior_ripple: f64,
}

impl Criterion {
    pub fn registered() -> Self {
        Self {
            max_transition: 1.0,
            min_contrast_frac: 0.90,
            min_peak_frac: 0.90,
            target_fwhm: 1.0,
            max_fwhm_error: 0.5,
            max_interior_ripple: 1.0 / 255.0,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ProfileBin {
    pub sdf: f64,
    pub mean: f64,
    pub samples: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EdgeMetric {
    pub feature: FeatureId,
    pub kind: ProbeKind,
    pub transition_10_90: f64,
    pub sdf_90: f64,
    pub sdf_10: f64,
    pub contrast: f64,
    pub target_contrast: f64,
    pub peak: f64,
    pub fwhm: f64,
    pub plateau_low: f64,
    pub plateau_high: f64,
    pub apex_displacement: f64,
    pub interior_ripple: f64,
    pub bins: Vec<ProfileBin>,
    pub samples: u32,
    pub thin_bins: u32,
    pub judged: bool,
    pub meets: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TargetMetric {
    pub probes: Vec<EdgeMetric>,
    pub transition_10_90: f64,
    pub contrast_frac: f64,
    pub peak_frac: f64,
    pub fwhm: f64,
    pub fwhm_error: f64,
    pub apex_displacement: f64,
    pub interior_ripple: f64,
    pub thin_bins: u32,
    pub judged: u32,
    pub meets: bool,
}

fn luminance(c: Rgb<f64>) -> f64 {
    (c.x + c.y + c.z) / 3.0
}

fn deviation(c: Rgb<f64>, reference: Rgb<f64>) -> f64 {
    (c.x - reference.x)
        .abs()
        .max((c.y - reference.y).abs())
        .max((c.z - reference.z).abs())
}

#[derive(Copy, Clone, Debug, PartialEq)]
struct Sample {
    sdf: f64,
    ordinate: f64,
    ripple: f64,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Mapping {
    pub scale: f64,
    pub offset: Vec2<f64>,
}

impl Mapping {
    pub fn identity() -> Self {
        Self {
            scale: 1.0,
            offset: Vec2::new(0.0, 0.0),
        }
    }

    pub fn shifted(offset: Vec2<f64>) -> Self {
        Self { scale: 1.0, offset }
    }

    pub fn at_scale(scale: f64) -> Self {
        Self {
            scale,
            offset: Vec2::new(0.0, 0.0),
        }
    }

    pub fn scene_of_pixel(&self, px: u32, py: u32) -> Vec2<f64> {
        let device = pixel_centre(px, py) + self.offset;
        Vec2::new(device.x / self.scale, device.y / self.scale)
    }

    pub fn device_viewport(&self, base: (u32, u32)) -> (u32, u32) {
        (
            (base.0 as f64 * self.scale).round() as u32,
            (base.1 as f64 * self.scale).round() as u32,
        )
    }
}

fn gather(
    image: &FloatImage<f64>,
    target: &Target<f64>,
    probe: &EdgeProbe<f64>,
    polarity: f64,
    mapping: Mapping,
) -> Vec<Sample> {
    let mut out = Vec::new();
    for py in 0..image.height {
        for px in 0..image.width {
            let centre = mapping.scene_of_pixel(px, py);
            if (centre - probe.anchor).length() > probe.window {
                continue;
            }
            if target.nearest_feature(centre) != probe.feature {
                continue;
            }
            let colour = image.get(px, py);
            out.push(Sample {
                sdf: target.sdf(centre),
                ordinate: polarity * luminance(colour),
                ripple: deviation(colour, target.set.ink),
            });
        }
    }
    out
}

fn bin(samples: &[Sample]) -> Vec<ProfileBin> {
    let mut keyed: Vec<(i64, f64, f64)> = samples
        .iter()
        .map(|s| ((s.sdf / BIN_WIDTH).floor() as i64, s.sdf, s.ordinate))
        .collect();
    keyed.sort_by(|a, b| match a.0.cmp(&b.0) {
        core::cmp::Ordering::Equal => a
            .1
            .partial_cmp(&b.1)
            .unwrap_or(core::cmp::Ordering::Equal),
        other => other,
    });
    let mut out: Vec<ProfileBin> = Vec::new();
    let mut key = i64::MIN;
    let mut sdf_sum = 0.0f64;
    let mut ordinate_sum = 0.0f64;
    let mut count = 0u32;
    for (k, sdf, ordinate) in keyed {
        if k != key && count > 0 {
            out.push(ProfileBin {
                sdf: sdf_sum / count as f64,
                mean: ordinate_sum / count as f64,
                samples: count,
            });
            sdf_sum = 0.0;
            ordinate_sum = 0.0;
            count = 0;
        }
        key = k;
        sdf_sum += sdf;
        ordinate_sum += ordinate;
        count += 1;
    }
    if count > 0 {
        out.push(ProfileBin {
            sdf: sdf_sum / count as f64,
            mean: ordinate_sum / count as f64,
            samples: count,
        });
    }
    out
}

fn band_mean(samples: &[Sample], lo: f64, hi: f64) -> Option<f64> {
    let mut sum = 0.0f64;
    let mut count = 0u32;
    for s in samples.iter() {
        if s.sdf >= lo && s.sdf <= hi {
            sum += s.ordinate;
            count += 1;
        }
    }
    if count == 0 {
        return None;
    }
    Some(sum / count as f64)
}

fn descending_crossing(bins: &[ProfileBin], from: usize, level: f64) -> f64 {
    let mut previous: Option<usize> = None;
    for index in from..bins.len() {
        let here = bins[index];
        if here.mean <= level {
            let above = match previous {
                Some(p) => bins[p],
                None => return f64::NAN,
            };
            let span = above.mean - here.mean;
            if !(span > 0.0) {
                return here.sdf;
            }
            let t = (above.mean - level) / span;
            return above.sdf + t * (here.sdf - above.sdf);
        }
        previous = Some(index);
    }
    f64::NAN
}

fn peak_bin(bins: &[ProfileBin]) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (index, b) in bins.iter().enumerate() {
        let better = match best {
            Some(p) => b.mean > bins[p].mean,
            None => true,
        };
        if better {
            best = Some(index);
        }
    }
    best
}

pub fn measure(
    image: &FloatImage<f64>,
    target: &Target<f64>,
    probe: &EdgeProbe<f64>,
    criterion: &Criterion,
) -> Result<EdgeMetric> {
    measure_shifted(image, target, probe, criterion, Vec2::new(0.0, 0.0))
}

pub fn measure_shifted(
    image: &FloatImage<f64>,
    target: &Target<f64>,
    probe: &EdgeProbe<f64>,
    criterion: &Criterion,
    offset: Vec2<f64>,
) -> Result<EdgeMetric> {
    measure_mapped(image, target, probe, criterion, Mapping::shifted(offset))
}

pub fn measure_mapped(
    image: &FloatImage<f64>,
    target: &Target<f64>,
    probe: &EdgeProbe<f64>,
    criterion: &Criterion,
    mapping: Mapping,
) -> Result<EdgeMetric> {
    let expected = mapping.device_viewport(target.viewport);
    if image.width != expected.0 || image.height != expected.1 {
        return Err(LightsError::SizeMismatch {
            a: (image.width, image.height),
            b: expected,
        });
    }
    let ink = luminance(target.set.ink);
    let paper = luminance(target.paper);
    let polarity = if ink >= paper { 1.0 } else { -1.0 };
    let target_contrast = (ink - paper).abs();
    let samples = gather(image, target, probe, polarity, mapping);
    let bins = bin(&samples);
    let thin_bins = bins.iter().filter(|b| b.samples < THIN_BIN_SAMPLES).count() as u32;

    let plateau_low = match band_mean(&samples, PLATEAU_NEAR, PLATEAU_FAR) {
        Some(v) => v,
        None => match bins.last() {
            Some(b) => b.mean,
            None => f64::NAN,
        },
    };
    let plateau_high = match band_mean(&samples, -PLATEAU_FAR, -PLATEAU_NEAR) {
        Some(v) => v,
        None => match peak_bin(&bins) {
            Some(i) => bins[i].mean,
            None => f64::NAN,
        },
    };
    let contrast = plateau_high - plateau_low;

    let (sdf_90, sdf_10) = if contrast > 0.0 {
        (
            descending_crossing(&bins, 0, plateau_low + HIGH_LEVEL * contrast),
            descending_crossing(&bins, 0, plateau_low + LOW_LEVEL * contrast),
        )
    } else {
        (f64::NAN, f64::NAN)
    };
    let transition_10_90 = sdf_10 - sdf_90;

    let (peak, fwhm) = match peak_bin(&bins) {
        Some(i) => {
            let peak = bins[i].mean - plateau_low;
            let half = descending_crossing(&bins, i, plateau_low + HALF_LEVEL * peak);
            (peak, 2.0 * (half - bins[i].sdf))
        }
        None => (f64::NAN, f64::NAN),
    };

    let half_level = plateau_low + HALF_LEVEL * contrast;
    let mut apex_displacement = 0.0f64;
    for s in samples.iter() {
        if s.sdf < 0.0 && s.ordinate < half_level && -s.sdf > apex_displacement {
            apex_displacement = -s.sdf;
        }
    }
    let mut interior_ripple = f64::NAN;
    for s in samples.iter() {
        if s.sdf <= -INTERIOR_DEPTH && !(s.ripple <= interior_ripple) {
            interior_ripple = s.ripple;
        }
    }

    let (judged, meets) = match probe.kind {
        ProbeKind::Edge => (
            true,
            transition_10_90 < criterion.max_transition
                && contrast >= criterion.min_contrast_frac * target_contrast,
        ),
        ProbeKind::Stroke => (
            true,
            peak >= criterion.min_peak_frac * target_contrast
                && (fwhm - criterion.target_fwhm).abs() <= criterion.max_fwhm_error,
        ),
        ProbeKind::Interior => (true, interior_ripple <= criterion.max_interior_ripple),
        ProbeKind::Apex => (false, false),
    };

    Ok(EdgeMetric {
        feature: probe.feature,
        kind: probe.kind,
        transition_10_90,
        sdf_90,
        sdf_10,
        contrast,
        target_contrast,
        peak,
        fwhm,
        plateau_low,
        plateau_high,
        apex_displacement,
        interior_ripple,
        bins,
        samples: samples.len() as u32,
        thin_bins,
        judged,
        meets,
    })
}

fn worse(current: f64, candidate: f64) -> f64 {
    if !candidate.is_finite() {
        return f64::NAN;
    }
    if current.is_nan() {
        return candidate;
    }
    if candidate > current { candidate } else { current }
}

fn weaker(current: f64, candidate: f64) -> f64 {
    if !candidate.is_finite() {
        return f64::NAN;
    }
    if current.is_nan() {
        return candidate;
    }
    if candidate < current { candidate } else { current }
}

pub fn measure_target(
    image: &FloatImage<f64>,
    target: &Target<f64>,
    criterion: &Criterion,
) -> Result<TargetMetric> {
    measure_target_shifted(image, target, criterion, Vec2::new(0.0, 0.0))
}

pub fn measure_target_shifted(
    image: &FloatImage<f64>,
    target: &Target<f64>,
    criterion: &Criterion,
    offset: Vec2<f64>,
) -> Result<TargetMetric> {
    measure_target_mapped(image, target, criterion, Mapping::shifted(offset))
}

pub fn measure_target_mapped(
    image: &FloatImage<f64>,
    target: &Target<f64>,
    criterion: &Criterion,
    mapping: Mapping,
) -> Result<TargetMetric> {
    let mut probes = Vec::new();
    for probe in target.probes.iter() {
        probes.push(measure_mapped(image, target, probe, criterion, mapping)?);
    }
    let mut transition_10_90 = f64::NAN;
    let mut contrast_frac = f64::NAN;
    let mut peak_frac = f64::NAN;
    let mut fwhm = f64::NAN;
    let mut fwhm_error = f64::NAN;
    let mut apex_displacement = f64::NAN;
    let mut interior_ripple = f64::NAN;
    let mut thin_bins = 0u32;
    let mut judged = 0u32;
    let mut meets = true;
    for m in probes.iter() {
        thin_bins += m.thin_bins;
        if m.judged {
            judged += 1;
            meets = meets && m.meets;
        }
        match m.kind {
            ProbeKind::Edge => {
                transition_10_90 = worse(transition_10_90, m.transition_10_90);
                contrast_frac = weaker(contrast_frac, m.contrast / m.target_contrast);
            }
            ProbeKind::Stroke => {
                peak_frac = weaker(peak_frac, m.peak / m.target_contrast);
                let error = (m.fwhm - criterion.target_fwhm).abs();
                if !(error <= fwhm_error) {
                    fwhm_error = error;
                    fwhm = m.fwhm;
                }
            }
            ProbeKind::Interior => {
                interior_ripple = worse(interior_ripple, m.interior_ripple);
            }
            ProbeKind::Apex => {
                apex_displacement = worse(apex_displacement, m.apex_displacement);
            }
        }
    }
    Ok(TargetMetric {
        probes,
        transition_10_90,
        contrast_frac,
        peak_frac,
        fwhm,
        fwhm_error,
        apex_displacement,
        interior_ripple,
        thin_bins,
        judged,
        meets: meets && judged > 0,
    })
}
