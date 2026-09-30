use crate::blend::{Additive, Over};
use crate::kernel::frame::{RayFrame, ScreenFrame};
use crate::kernel::gaussian::radius_for_cutoff;
use crate::kernel::{CurveShape, Kernel, KernelId, moment_prefilter};
use crate::numerics::float::FloatingPoint;
use crate::numerics::symmetric::Sym2;
use crate::numerics::vector::{Vec2, Vec3};

pub const BOX_VARIANCE_TO_WIDTH_SQUARED: f64 = 12.0;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct GaborParams<T> {
    pub frequency: T,
    pub orientation: T,
    pub elevation: T,
    pub phase: T,
    pub amplitude: T,
}

impl<T: FloatingPoint> Default for GaborParams<T> {
    fn default() -> Self {
        Self {
            frequency: T::ZERO,
            orientation: T::ZERO,
            elevation: T::ZERO,
            phase: T::ZERO,
            amplitude: T::ONE,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct GaborScreen<T> {
    pub carrier: Vec2<T>,
    pub phase: T,
    pub amplitude: T,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct GaborRay<T> {
    pub carrier: Vec3<T>,
    pub phase: T,
    pub amplitude: T,
}

#[inline(always)]
pub fn gabor_envelope<T: FloatingPoint>(quadratic: T) -> T {
    (-T::HALF * quadratic).exp()
}

#[inline(always)]
pub fn gabor_carrier<T: FloatingPoint>(argument: T) -> T {
    let (_, cosine) = argument.sin_cos();
    cosine
}

#[inline(always)]
pub fn gabor_carrier_screen<T: FloatingPoint>(s: &GaborScreen<T>, u: Vec2<T>) -> T {
    s.amplitude * gabor_carrier(s.carrier.dot(u) + s.phase)
}

#[inline(always)]
pub fn gabor_carrier_ray<T: FloatingPoint>(r: &GaborRay<T>, u: Vec3<T>) -> T {
    r.amplitude * gabor_carrier(r.carrier.dot(u) + r.phase)
}

pub fn gabor_params_from<T: FloatingPoint>(shape: &CurveShape) -> GaborParams<T> {
    GaborParams {
        frequency: T::from_f64(shape.frequency),
        orientation: T::from_f64(shape.orientation),
        elevation: T::from_f64(shape.elevation),
        phase: T::from_f64(shape.phase),
        amplitude: T::ONE,
    }
}

pub fn gabor_prepare_screen<T: FloatingPoint>(
    p: &GaborParams<T>,
    f: &ScreenFrame<T>,
) -> GaborScreen<T> {
    let (sine, cosine) = p.orientation.sin_cos();
    let direction = Vec2::new(cosine, sine);
    let lifted = f.l.apply_transpose(direction);
    GaborScreen {
        carrier: lifted * (T::TAU * p.frequency),
        phase: p.phase,
        amplitude: p.amplitude,
    }
}

pub fn gabor_prepare_ray<T: FloatingPoint>(p: &GaborParams<T>, f: &RayFrame<T>) -> GaborRay<T> {
    let (sine, cosine) = p.orientation.sin_cos();
    let (sine_e, cosine_e) = p.elevation.sin_cos();
    let d = Vec3::new(cosine_e * cosine, cosine_e * sine, sine_e);
    let l = f.whiten.inverse();
    let lifted = Vec3::new(
        l.m[0] * d.x + l.m[1] * d.y + l.m[3] * d.z,
        l.m[2] * d.y + l.m[4] * d.z,
        l.m[5] * d.z,
    );
    GaborRay {
        carrier: lifted * (T::TAU * p.frequency),
        phase: p.phase,
        amplitude: p.amplitude,
    }
}

pub fn normalised_sinc<T: FloatingPoint>(x: T) -> T {
    if x == T::ZERO {
        return T::ONE;
    }
    let a = T::PI * x;
    let (sine, _) = a.sin_cos();
    sine / a
}

pub fn carrier_box_attenuation<T: FloatingPoint>(
    frequency: T,
    orientation: T,
    pixel_variance: T,
) -> T {
    if !(pixel_variance > T::ZERO) {
        return T::ONE;
    }
    let width = (T::from_f64(BOX_VARIANCE_TO_WIDTH_SQUARED) * pixel_variance).sqrt();
    let (sine, cosine) = orientation.sin_cos();
    normalised_sinc(frequency * cosine * width) * normalised_sinc(frequency * sine * width)
}

pub fn gabor_prefilter<T: FloatingPoint>(
    p: &GaborParams<T>,
    cov: Sym2<T>,
    pixel_variance: T,
) -> (GaborParams<T>, Sym2<T>, T) {
    let (dilated, gain) = moment_prefilter(cov, pixel_variance);
    let mut filtered = *p;
    filtered.amplitude =
        p.amplitude * carrier_box_attenuation(p.frequency, p.orientation, pixel_variance);
    (filtered, dilated, gain)
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct GaborColour;

impl<T: FloatingPoint> Kernel<T> for GaborColour {
    type Params = GaborParams<T>;
    type Screen = GaborScreen<T>;
    type Ray = GaborRay<T>;
    type Blend = Over;

    const ID: KernelId = KernelId(4);
    const NAME: &'static str = "gabor-colour";
    const RADIAL: bool = false;
    const HAS_SCREEN_MARGINAL: bool = true;
    const MODULATED: bool = true;

    #[inline(always)]
    fn params_from(shape: &CurveShape) -> Self::Params {
        gabor_params_from(shape)
    }

    #[inline(always)]
    fn prepare_screen(p: &Self::Params, f: &ScreenFrame<T>) -> Self::Screen {
        gabor_prepare_screen(p, f)
    }

    #[inline(always)]
    fn prepare_ray(p: &Self::Params, f: &RayFrame<T>) -> Self::Ray {
        gabor_prepare_ray(p, f)
    }

    #[inline(always)]
    fn eval_screen(_s: &Self::Screen, u: Vec2<T>) -> T {
        gabor_envelope(u.length_squared())
    }

    #[inline(always)]
    fn eval_ray(_r: &Self::Ray, u: Vec3<T>) -> T {
        gabor_envelope(u.length_squared())
    }

    #[inline(always)]
    fn modulate_screen(s: &Self::Screen, u: Vec2<T>) -> T {
        gabor_carrier_screen(s, u)
    }

    #[inline(always)]
    fn modulate_ray(r: &Self::Ray, u: Vec3<T>) -> T {
        gabor_carrier_ray(r, u)
    }

    #[inline(always)]
    fn support_screen(_s: &Self::Screen, peak: T, cutoff: T) -> T {
        radius_for_cutoff(peak, cutoff)
    }

    #[inline(always)]
    fn support_ray(_r: &Self::Ray, peak: T, cutoff: T) -> T {
        radius_for_cutoff(peak, cutoff)
    }

    fn prefilter(p: &Self::Params, cov: Sym2<T>, pixel_variance: T) -> (Self::Params, Sym2<T>, T) {
        gabor_prefilter(p, cov, pixel_variance)
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct GaborDensity;

impl<T: FloatingPoint> Kernel<T> for GaborDensity {
    type Params = GaborParams<T>;
    type Screen = GaborScreen<T>;
    type Ray = GaborRay<T>;
    type Blend = Additive;

    const ID: KernelId = KernelId(5);
    const NAME: &'static str = "gabor-density";
    const RADIAL: bool = false;
    const HAS_SCREEN_MARGINAL: bool = true;

    #[inline(always)]
    fn params_from(shape: &CurveShape) -> Self::Params {
        gabor_params_from(shape)
    }

    #[inline(always)]
    fn prepare_screen(p: &Self::Params, f: &ScreenFrame<T>) -> Self::Screen {
        gabor_prepare_screen(p, f)
    }

    #[inline(always)]
    fn prepare_ray(p: &Self::Params, f: &RayFrame<T>) -> Self::Ray {
        gabor_prepare_ray(p, f)
    }

    #[inline(always)]
    fn eval_screen(s: &Self::Screen, u: Vec2<T>) -> T {
        gabor_envelope(u.length_squared()) * gabor_carrier_screen(s, u)
    }

    #[inline(always)]
    fn eval_ray(r: &Self::Ray, u: Vec3<T>) -> T {
        gabor_envelope(u.length_squared()) * gabor_carrier_ray(r, u)
    }

    #[inline(always)]
    fn support_screen(_s: &Self::Screen, peak: T, cutoff: T) -> T {
        radius_for_cutoff(peak, cutoff)
    }

    #[inline(always)]
    fn support_ray(_r: &Self::Ray, peak: T, cutoff: T) -> T {
        radius_for_cutoff(peak, cutoff)
    }

    fn prefilter(p: &Self::Params, cov: Sym2<T>, pixel_variance: T) -> (Self::Params, Sym2<T>, T) {
        gabor_prefilter(p, cov, pixel_variance)
    }
}
