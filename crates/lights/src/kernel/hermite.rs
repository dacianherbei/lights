use crate::blend::{Additive, Over};
use crate::kernel::frame::{RayFrame, ScreenFrame};
use crate::kernel::{CurveShape, FIRST_HERMITE_ORDER, Kernel, KernelId, SECOND_HERMITE_ORDER,
    moment_prefilter};
use crate::numerics::float::FloatingPoint;
use crate::numerics::symmetric::Sym2;
use crate::numerics::vector::{Vec2, Vec3};

pub const ORDER_ONE_PEAK_NORMALISER: f64 = 1.6487212707001282;
pub const ORDER_TWO_PEAK_NORMALISER: f64 = 1.0;
pub const SUPPORT_MODE_RADIUS: f64 = 1.0;
pub const SUPPORT_GROWTH_STEPS: usize = 48;
pub const SUPPORT_BISECTION_STEPS: usize = 96;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct HermiteParams<T> {
    pub order: u8,
    pub orientation: T,
    pub elevation: T,
}

impl<T: FloatingPoint> Default for HermiteParams<T> {
    fn default() -> Self {
        Self {
            order: FIRST_HERMITE_ORDER,
            orientation: T::ZERO,
            elevation: T::ZERO,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct HermiteScreen<T> {
    pub direction: Vec2<T>,
    pub normaliser: T,
    pub order: u8,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct HermiteRay<T> {
    pub direction: Vec3<T>,
    pub normaliser: T,
    pub order: u8,
}

pub fn clamped_order(order: u8) -> u8 {
    if order <= FIRST_HERMITE_ORDER {
        FIRST_HERMITE_ORDER
    } else {
        SECOND_HERMITE_ORDER
    }
}

pub fn peak_normaliser<T: FloatingPoint>(order: u8) -> T {
    if clamped_order(order) == FIRST_HERMITE_ORDER {
        T::from_f64(ORDER_ONE_PEAK_NORMALISER)
    } else {
        T::from_f64(ORDER_TWO_PEAK_NORMALISER)
    }
}

#[inline(always)]
pub fn probabilists_hermite<T: FloatingPoint>(order: u8, t: T) -> T {
    if order == FIRST_HERMITE_ORDER {
        t
    } else {
        t * t - T::ONE
    }
}

#[inline(always)]
pub fn hermite_envelope<T: FloatingPoint>(quadratic: T) -> T {
    (-T::HALF * quadratic).exp()
}

#[inline(always)]
pub fn hermite_factor_screen<T: FloatingPoint>(s: &HermiteScreen<T>, u: Vec2<T>) -> T {
    s.normaliser * probabilists_hermite(s.order, s.direction.dot(u))
}

#[inline(always)]
pub fn hermite_factor_ray<T: FloatingPoint>(r: &HermiteRay<T>, u: Vec3<T>) -> T {
    r.normaliser * probabilists_hermite(r.order, r.direction.dot(u))
}

#[inline(always)]
pub fn hermite_majorant<T: FloatingPoint>(order: u8, normaliser: T, r: T) -> T {
    let polynomial = if order == FIRST_HERMITE_ORDER {
        r
    } else {
        r * r + T::ONE
    };
    normaliser * polynomial * (-T::HALF * r * r).exp()
}

pub fn hermite_radius<T: FloatingPoint>(order: u8, normaliser: T, peak: T, cutoff: T) -> T {
    if !(peak > T::ZERO) || !(cutoff > T::ZERO) {
        return T::ZERO;
    }
    let mode = T::from_f64(SUPPORT_MODE_RADIUS);
    if !(hermite_majorant(order, normaliser, mode) * peak > cutoff) {
        return T::ZERO;
    }
    let mut hi = mode;
    for _ in 0..SUPPORT_GROWTH_STEPS {
        hi = hi + hi;
        if hermite_majorant(order, normaliser, hi) * peak <= cutoff {
            break;
        }
    }
    let mut lo = mode;
    for _ in 0..SUPPORT_BISECTION_STEPS {
        let mid = (lo + hi) * T::HALF;
        if hermite_majorant(order, normaliser, mid) * peak > cutoff {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    hi
}

pub fn hermite_params_from<T: FloatingPoint>(shape: &CurveShape) -> HermiteParams<T> {
    HermiteParams {
        order: clamped_order(shape.order),
        orientation: T::from_f64(shape.orientation),
        elevation: T::from_f64(shape.elevation),
    }
}

pub fn hermite_prepare_screen<T: FloatingPoint>(
    p: &HermiteParams<T>,
    _f: &ScreenFrame<T>,
) -> HermiteScreen<T> {
    let (sine, cosine) = p.orientation.sin_cos();
    HermiteScreen {
        direction: Vec2::new(cosine, sine),
        normaliser: peak_normaliser(p.order),
        order: clamped_order(p.order),
    }
}

pub fn hermite_prepare_ray<T: FloatingPoint>(
    p: &HermiteParams<T>,
    _f: &RayFrame<T>,
) -> HermiteRay<T> {
    let (sine, cosine) = p.orientation.sin_cos();
    let (sine_e, cosine_e) = p.elevation.sin_cos();
    HermiteRay {
        direction: Vec3::new(cosine_e * cosine, cosine_e * sine, sine_e),
        normaliser: peak_normaliser(p.order),
        order: clamped_order(p.order),
    }
}

pub fn hermite_prefilter<T: FloatingPoint>(
    p: &HermiteParams<T>,
    cov: Sym2<T>,
    pixel_variance: T,
) -> (HermiteParams<T>, Sym2<T>, T) {
    let (dilated, gain) = moment_prefilter(cov, pixel_variance);
    (*p, dilated, gain)
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct GaussHermiteColour;

impl<T: FloatingPoint> Kernel<T> for GaussHermiteColour {
    type Params = HermiteParams<T>;
    type Screen = HermiteScreen<T>;
    type Ray = HermiteRay<T>;
    type Blend = Over;

    const ID: KernelId = KernelId(6);
    const NAME: &'static str = "gauss-hermite-colour";
    const RADIAL: bool = false;
    const HAS_SCREEN_MARGINAL: bool = true;
    const MODULATED: bool = true;

    #[inline(always)]
    fn params_from(shape: &CurveShape) -> Self::Params {
        hermite_params_from(shape)
    }

    #[inline(always)]
    fn prepare_screen(p: &Self::Params, f: &ScreenFrame<T>) -> Self::Screen {
        hermite_prepare_screen(p, f)
    }

    #[inline(always)]
    fn prepare_ray(p: &Self::Params, f: &RayFrame<T>) -> Self::Ray {
        hermite_prepare_ray(p, f)
    }

    #[inline(always)]
    fn eval_screen(_s: &Self::Screen, u: Vec2<T>) -> T {
        hermite_envelope(u.length_squared())
    }

    #[inline(always)]
    fn eval_ray(_r: &Self::Ray, u: Vec3<T>) -> T {
        hermite_envelope(u.length_squared())
    }

    #[inline(always)]
    fn modulate_screen(s: &Self::Screen, u: Vec2<T>) -> T {
        hermite_factor_screen(s, u)
    }

    #[inline(always)]
    fn modulate_ray(r: &Self::Ray, u: Vec3<T>) -> T {
        hermite_factor_ray(r, u)
    }

    #[inline(always)]
    fn support_screen(s: &Self::Screen, peak: T, cutoff: T) -> T {
        hermite_radius(s.order, s.normaliser, peak, cutoff)
    }

    #[inline(always)]
    fn support_ray(r: &Self::Ray, peak: T, cutoff: T) -> T {
        hermite_radius(r.order, r.normaliser, peak, cutoff)
    }

    fn prefilter(p: &Self::Params, cov: Sym2<T>, pixel_variance: T) -> (Self::Params, Sym2<T>, T) {
        hermite_prefilter(p, cov, pixel_variance)
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct GaussHermiteDensity;

impl<T: FloatingPoint> Kernel<T> for GaussHermiteDensity {
    type Params = HermiteParams<T>;
    type Screen = HermiteScreen<T>;
    type Ray = HermiteRay<T>;
    type Blend = Additive;

    const ID: KernelId = KernelId(7);
    const NAME: &'static str = "gauss-hermite-density";
    const RADIAL: bool = false;
    const HAS_SCREEN_MARGINAL: bool = true;

    #[inline(always)]
    fn params_from(shape: &CurveShape) -> Self::Params {
        hermite_params_from(shape)
    }

    #[inline(always)]
    fn prepare_screen(p: &Self::Params, f: &ScreenFrame<T>) -> Self::Screen {
        hermite_prepare_screen(p, f)
    }

    #[inline(always)]
    fn prepare_ray(p: &Self::Params, f: &RayFrame<T>) -> Self::Ray {
        hermite_prepare_ray(p, f)
    }

    #[inline(always)]
    fn eval_screen(s: &Self::Screen, u: Vec2<T>) -> T {
        hermite_envelope(u.length_squared()) * hermite_factor_screen(s, u)
    }

    #[inline(always)]
    fn eval_ray(r: &Self::Ray, u: Vec3<T>) -> T {
        hermite_envelope(u.length_squared()) * hermite_factor_ray(r, u)
    }

    #[inline(always)]
    fn support_screen(s: &Self::Screen, peak: T, cutoff: T) -> T {
        hermite_radius(s.order, s.normaliser, peak, cutoff)
    }

    #[inline(always)]
    fn support_ray(r: &Self::Ray, peak: T, cutoff: T) -> T {
        hermite_radius(r.order, r.normaliser, peak, cutoff)
    }

    fn prefilter(p: &Self::Params, cov: Sym2<T>, pixel_variance: T) -> (Self::Params, Sym2<T>, T) {
        hermite_prefilter(p, cov, pixel_variance)
    }
}
