use crate::error::{LightsError, Result};
use crate::fit::atom2d::Atom2d;
use crate::fit::content::{
    N1_NAME, N2_NAME, N3_AMPLITUDES, N3_FLOOR, N3_LATTICES, N3_NAME, N3_SEED, N3_SPAN, N4_AMPLITUDES,
    N4_FLOOR, N4_LATTICES, N4_NAME, N4_SEED_FAR, N4_SEED_NEAR, N4_SPAN, RASTER_HEIGHT,
    RASTER_WIDTH, ValueLattice, colour_noise, n1_sample, n2_soft_shadow_sample, n4_region,
    sample_colour_noise,
};
use crate::fit::coverage::exact_coverage;
use crate::fit::coverage::coverage_image_to_colour;
use crate::fit::edge::{Criterion, Mapping, TargetMetric, measure_target_mapped};
use crate::fit::target::{EdgeProbe, Part, Primitive, PrimitiveSet, Target};
use crate::guard::Budget;
use crate::image::FloatImage;
pub use crate::numerics::similarity::Similarity2;
use crate::numerics::vector::{Rgb, Vec2};

pub const SUPERSAMPLES_PER_SCENE_UNIT: u32 = 4;
pub const MAX_SUPERSAMPLE: u32 = 64;
pub const MAX_SCALE_NUMERATOR: u32 = 1024;

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Scale {
    numerator: u32,
    denominator: u32,
}

fn gcd(a: u32, b: u32) -> u32 {
    let mut x = a;
    let mut y = b;
    let mut budget = Budget::new("fit::scale::gcd", 64);
    for _ in 0..64 {
        budget.tick_or_die();
        if y == 0 {
            return x;
        }
        let r = x % y;
        x = y;
        y = r;
    }
    x
}

impl Scale {
    pub fn new(numerator: u32, denominator: u32) -> Result<Self> {
        if numerator == 0 || denominator == 0 || numerator > MAX_SCALE_NUMERATOR {
            return Err(LightsError::SizeMismatch {
                a: (numerator, denominator),
                b: (1, MAX_SCALE_NUMERATOR),
            });
        }
        let d = gcd(numerator, denominator);
        Ok(Self {
            numerator: numerator / d,
            denominator: denominator / d,
        })
    }

    pub fn numerator(self) -> u32 {
        self.numerator
    }

    pub fn denominator(self) -> u32 {
        self.denominator
    }

    pub fn factor(self) -> f64 {
        self.numerator as f64 / self.denominator as f64
    }

    pub fn apply(self, value: u32) -> Result<u32> {
        let scaled = (value as u64) * (self.numerator as u64);
        if scaled % (self.denominator as u64) != 0 {
            return Err(LightsError::SizeMismatch {
                a: (value, self.denominator),
                b: (self.numerator, self.denominator),
            });
        }
        let out = scaled / (self.denominator as u64);
        if out == 0 || out > u32::MAX as u64 {
            return Err(LightsError::ViewportZero {
                width: value,
                height: self.numerator,
            });
        }
        Ok(out as u32)
    }

    pub fn viewport(self, base: (u32, u32)) -> Result<(u32, u32)> {
        Ok((self.apply(base.0)?, self.apply(base.1)?))
    }

    pub fn ratio_to(self, other: Scale) -> Result<Scale> {
        Scale::new(
            self.numerator * other.denominator,
            self.denominator * other.numerator,
        )
    }

    pub fn label(self) -> String {
        if self.denominator == 1 {
            return format!("{}x", self.numerator);
        }
        format!("{}x", self.factor())
    }

    pub fn slug(self) -> String {
        if self.denominator == 1 {
            return format!("s{}", self.numerator);
        }
        format!("s{}over{}", self.numerator, self.denominator)
    }

    pub fn supersample(self) -> u32 {
        let wanted =
            ((SUPERSAMPLES_PER_SCENE_UNIT as u64) * (self.denominator as u64)).div_ceil(self.numerator as u64);
        let clamped = wanted.max(1).min(MAX_SUPERSAMPLE as u64);
        clamped as u32
    }
}

pub fn scaled_primitive(shape: &Primitive<f64>, scale: f64) -> Primitive<f64> {
    match shape {
        Primitive::HalfPlane { point, normal } => Primitive::HalfPlane {
            point: *point * scale,
            normal: *normal,
        },
        Primitive::Segment { a, b, half_width } => Primitive::Segment {
            a: *a * scale,
            b: *b * scale,
            half_width: *half_width * scale,
        },
        Primitive::Arc {
            centre,
            radius,
            from,
            to,
            half_width,
        } => Primitive::Arc {
            centre: *centre * scale,
            radius: *radius * scale,
            from: *from,
            to: *to,
            half_width: *half_width * scale,
        },
        Primitive::Disc { centre, radius } => Primitive::Disc {
            centre: *centre * scale,
            radius: *radius * scale,
        },
    }
}

fn scaled_parts(parts: &[Part<f64>], scale: f64) -> Vec<Part<f64>> {
    parts
        .iter()
        .map(|p| Part {
            id: p.id,
            shape: scaled_primitive(&p.shape, scale),
        })
        .collect()
}

fn scaled_probes(probes: &[EdgeProbe<f64>], scale: f64) -> Vec<EdgeProbe<f64>> {
    probes
        .iter()
        .map(|p| EdgeProbe {
            feature: p.feature,
            anchor: p.anchor * scale,
            window: p.window * scale,
            kind: p.kind,
        })
        .collect()
}

pub fn scaled_target(target: &Target<f64>, scale: Scale) -> Result<Target<f64>> {
    let factor = scale.factor();
    Ok(Target {
        name: target.name,
        viewport: scale.viewport(target.viewport)?,
        set: PrimitiveSet {
            ink: target.set.ink,
            parts: scaled_parts(&target.set.parts, factor),
            holes: scaled_parts(&target.set.holes, factor),
            regions: target.set.regions.iter().map(|r| r.mapped(factor, Vec2::zero())).collect(),
        },
        paper: target.paper,
        probes: scaled_probes(&target.probes, factor),
    })
}

pub fn scaled_atoms(atoms: &[Atom2d], scale: Scale) -> Vec<Atom2d> {
    let factor = scale.factor();
    let shift = factor.ln();
    atoms
        .iter()
        .map(|a| Atom2d {
            centre: a.centre * factor,
            log_scale: Vec2::new(a.log_scale.x + shift, a.log_scale.y + shift),
            angle: a.angle,
            opacity: a.opacity,
            colour: a.colour,
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq)]
pub enum Sampler {
    ColourField,
    SoftShadow,
    BandLimitedNoise(Vec<Vec<ValueLattice>>),
    BroadbandPatch(Vec<Vec<ValueLattice>>, Vec<Vec<ValueLattice>>),
}

impl Sampler {
    pub fn band_limited_noise() -> Self {
        Sampler::BandLimitedNoise(colour_noise(
            N3_SEED,
            &N3_LATTICES,
            RASTER_WIDTH,
            RASTER_HEIGHT,
        ))
    }

    pub fn broadband_patch() -> Self {
        Sampler::BroadbandPatch(
            colour_noise(N4_SEED_NEAR, &N4_LATTICES, RASTER_WIDTH, RASTER_HEIGHT),
            colour_noise(N4_SEED_FAR, &N4_LATTICES, RASTER_WIDTH, RASTER_HEIGHT),
        )
    }

    pub fn sample(&self, p: Vec2<f64>) -> Rgb<f64> {
        match self {
            Sampler::ColourField => n1_sample(p),
            Sampler::SoftShadow => n2_soft_shadow_sample(p),
            Sampler::BandLimitedNoise(channels) => sample_colour_noise(
                channels,
                &N3_AMPLITUDES,
                N3_FLOOR,
                N3_SPAN,
                p,
            ),
            Sampler::BroadbandPatch(near, far) => {
                let source = if n4_region(p) { near } else { far };
                sample_colour_noise(source, &N4_AMPLITUDES, N4_FLOOR, N4_SPAN, p)
            }
        }
    }
}

pub fn box_average(
    sampler: &Sampler,
    viewport: (u32, u32),
    scale: Scale,
    supersample: u32,
) -> FloatImage<f64> {
    let factor = scale.factor();
    let k = supersample.max(1);
    let inverse = 1.0 / (k as f64);
    let weight = 1.0 / ((k as f64) * (k as f64));
    let mut image = FloatImage::<f64>::new(viewport.0, viewport.1);
    for py in 0..viewport.1 {
        for px in 0..viewport.0 {
            let mut acc = Rgb::zero();
            for sy in 0..k {
                for sx in 0..k {
                    let device = Vec2::new(
                        px as f64 + (sx as f64 + 0.5) * inverse,
                        py as f64 + (sy as f64 + 0.5) * inverse,
                    );
                    acc = acc + sampler.sample(Vec2::new(device.x / factor, device.y / factor));
                }
            }
            image.set(px, py, acc * weight);
        }
    }
    image
}

#[derive(Clone, Debug, PartialEq)]
pub enum ScaleClass {
    Analytic(Target<f64>),
    Continuous {
        name: &'static str,
        viewport: (u32, u32),
        sampler: Sampler,
    },
}

impl ScaleClass {
    pub fn colour_field() -> Self {
        ScaleClass::Continuous {
            name: N1_NAME,
            viewport: (RASTER_WIDTH, RASTER_HEIGHT),
            sampler: Sampler::ColourField,
        }
    }

    pub fn soft_shadow() -> Self {
        ScaleClass::Continuous {
            name: N2_NAME,
            viewport: (RASTER_WIDTH, RASTER_HEIGHT),
            sampler: Sampler::SoftShadow,
        }
    }

    pub fn band_limited_noise() -> Self {
        ScaleClass::Continuous {
            name: N3_NAME,
            viewport: (RASTER_WIDTH, RASTER_HEIGHT),
            sampler: Sampler::band_limited_noise(),
        }
    }

    pub fn broadband_patch() -> Self {
        ScaleClass::Continuous {
            name: N4_NAME,
            viewport: (RASTER_WIDTH, RASTER_HEIGHT),
            sampler: Sampler::broadband_patch(),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            ScaleClass::Analytic(t) => t.name,
            ScaleClass::Continuous { name, .. } => name,
        }
    }

    pub fn base_viewport(&self) -> (u32, u32) {
        match self {
            ScaleClass::Analytic(t) => t.viewport,
            ScaleClass::Continuous { viewport, .. } => *viewport,
        }
    }

    pub fn analytic(&self) -> Option<&Target<f64>> {
        match self {
            ScaleClass::Analytic(t) => Some(t),
            ScaleClass::Continuous { .. } => None,
        }
    }

    pub fn viewport_at(&self, scale: Scale) -> Result<(u32, u32)> {
        scale.viewport(self.base_viewport())
    }

    pub fn scaffold(&self, scale: Scale) -> Result<Target<f64>> {
        match self {
            ScaleClass::Analytic(t) => scaled_target(t, scale),
            ScaleClass::Continuous { name, .. } => Ok(Target::ink_on_paper(
                name,
                self.viewport_at(scale)?,
                Vec::new(),
                Vec::new(),
                Vec::new(),
            )),
        }
    }

    pub fn truth(
        &self,
        scale: Scale,
        max_depth: u32,
        budget: &mut Budget,
    ) -> Result<FloatImage<f64>> {
        match self {
            ScaleClass::Analytic(t) => {
                let scaled = scaled_target(t, scale)?;
                let (coverage, _) = exact_coverage(&scaled, max_depth, budget)?;
                Ok(coverage_image_to_colour(&scaled, &coverage))
            }
            ScaleClass::Continuous { sampler, .. } => Ok(box_average(
                sampler,
                self.viewport_at(scale)?,
                scale,
                scale.supersample(),
            )),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Convention {
    SceneUnits,
    DevicePixels,
}

impl Convention {
    pub const ALL: [Convention; 2] = [Convention::SceneUnits, Convention::DevicePixels];

    pub fn name(self) -> &'static str {
        match self {
            Convention::SceneUnits => "scene-units",
            Convention::DevicePixels => "device-pixels",
        }
    }

    pub fn legend(self) -> &'static str {
        match self {
            Convention::SceneUnits => {
                "the criterion is expressed in SCENE units, so the same question is asked at \
                 every scale and a change along the ladder is attributable to the description"
            }
            Convention::DevicePixels => {
                "the criterion is expressed in DEVICE pixels and therefore gets STRICTER with \
                 zoom: at 4x one pixel is a quarter of the scene distance it was at 1x, so a \
                 fixed pixel threshold is four times tighter"
            }
        }
    }

    pub fn convert(self, scene_value: f64, scale: Scale) -> f64 {
        match self {
            Convention::SceneUnits => scene_value,
            Convention::DevicePixels => scene_value * scale.factor(),
        }
    }
}

pub fn measure_scene(
    image: &FloatImage<f64>,
    base_target: &Target<f64>,
    scale: Scale,
    criterion: &Criterion,
) -> Result<TargetMetric> {
    measure_target_mapped(
        image,
        base_target,
        criterion,
        Mapping::at_scale(scale.factor()),
    )
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Judgement {
    pub scene: f64,
    pub device: f64,
    pub meets_scene: bool,
    pub meets_device: bool,
}

pub fn judge_transition(metric: &TargetMetric, scale: Scale, criterion: &Criterion) -> Judgement {
    let scene = metric.transition_10_90;
    let device = Convention::DevicePixels.convert(scene, scale);
    Judgement {
        scene,
        device,
        meets_scene: scene < criterion.max_transition,
        meets_device: device < criterion.max_transition,
    }
}

pub fn judge_fwhm(metric: &TargetMetric, scale: Scale, criterion: &Criterion) -> Judgement {
    let scene = metric.fwhm;
    let device = Convention::DevicePixels.convert(scene, scale);
    Judgement {
        scene,
        device,
        meets_scene: (scene - criterion.target_fwhm).abs() <= criterion.max_fwhm_error,
        meets_device: (device - criterion.target_fwhm).abs() <= criterion.max_fwhm_error,
    }
}
