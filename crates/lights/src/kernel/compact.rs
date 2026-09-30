use crate::blend::Over;
use crate::kernel::frame::{RayFrame, ScreenFrame};
use crate::kernel::{CurveShape, Kernel, KernelId, moment_prefilter};
use crate::numerics::float::FloatingPoint;
use crate::numerics::symmetric::Sym2;
use crate::numerics::vector::{Vec2, Vec3};

pub const ZERO_RADIUS_SQUARED: f64 = 10.0;
pub const ZERO_RADIUS: f64 = 3.1622776601683795;
pub const ONE_THIRD: f64 = 1.0 / 3.0;

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct CompactParams;

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct CompactBump;

#[inline(always)]
pub fn compact_response<T: FloatingPoint>(quadratic: T) -> T {
    let s = quadratic / T::from_f64(ZERO_RADIUS_SQUARED);
    if !(s < T::ONE) {
        return T::ZERO;
    }
    let base = T::ONE - s;
    base * base * base
}

pub fn compact_radius<T: FloatingPoint>(peak: T, cutoff: T) -> T {
    let full = T::from_f64(ZERO_RADIUS);
    if !(peak > T::ZERO) {
        return T::ZERO;
    }
    if !(cutoff > T::ZERO) {
        return full;
    }
    if !(peak > cutoff) {
        return T::ZERO;
    }
    let level = (cutoff / peak).powf(T::from_f64(ONE_THIRD));
    let fraction = (T::ONE - level).max(T::ZERO).sqrt();
    full * fraction.min(T::ONE)
}

impl<T: FloatingPoint> Kernel<T> for CompactBump {
    type Params = CompactParams;
    type Screen = ();
    type Ray = ();
    type Blend = Over;

    const ID: KernelId = KernelId(3);
    const NAME: &'static str = "compact-bump";
    const RADIAL: bool = true;
    const HAS_SCREEN_MARGINAL: bool = true;

    #[inline(always)]
    fn params_from(_shape: &CurveShape) -> Self::Params {
        CompactParams
    }

    #[inline(always)]
    fn prepare_screen(_p: &Self::Params, _f: &ScreenFrame<T>) -> Self::Screen {}

    #[inline(always)]
    fn prepare_ray(_p: &Self::Params, _f: &RayFrame<T>) -> Self::Ray {}

    #[inline(always)]
    fn eval_screen(_s: &Self::Screen, u: Vec2<T>) -> T {
        compact_response(u.length_squared())
    }

    #[inline(always)]
    fn eval_ray(_r: &Self::Ray, u: Vec3<T>) -> T {
        compact_response(u.length_squared())
    }

    #[inline(always)]
    fn support_screen(_s: &Self::Screen, peak: T, cutoff: T) -> T {
        compact_radius(peak, cutoff)
    }

    #[inline(always)]
    fn support_ray(_r: &Self::Ray, peak: T, cutoff: T) -> T {
        compact_radius(peak, cutoff)
    }

    fn prefilter(
        p: &Self::Params,
        cov: Sym2<T>,
        pixel_variance: T,
    ) -> (Self::Params, Sym2<T>, T) {
        let (dilated, gain) = moment_prefilter(cov, pixel_variance);
        (*p, dilated, gain)
    }
}
