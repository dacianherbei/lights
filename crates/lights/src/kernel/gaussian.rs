use crate::blend::{Normalised, Over};
use crate::kernel::frame::{RayFrame, ScreenFrame};
use crate::kernel::{CurveShape, Kernel, KernelId, moment_prefilter};
use crate::numerics::float::FloatingPoint;
use crate::numerics::symmetric::Sym2;
use crate::numerics::vector::{Vec2, Vec3};

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct GaussianParams;

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Gaussian;

pub const REFERENCE_THREE_SIGMA_CUTOFF: f64 = 0.011108996538242306;

pub fn radius_for_cutoff<T: FloatingPoint>(peak: T, cutoff: T) -> T {
    if !(peak > T::ZERO) || !(cutoff > T::ZERO) || !(peak > cutoff) {
        return T::ZERO;
    }
    let ratio = peak / cutoff;
    (T::TWO * ratio.ln()).max(T::ZERO).sqrt()
}

impl<T: FloatingPoint> Kernel<T> for Gaussian {
    type Params = GaussianParams;
    type Screen = ();
    type Ray = ();
    type Blend = Over;

    const ID: KernelId = KernelId(0);
    const NAME: &'static str = "gaussian";
    const RADIAL: bool = true;
    const HAS_SCREEN_MARGINAL: bool = true;

    #[inline(always)]
    fn params_from(_shape: &CurveShape) -> Self::Params {
        GaussianParams
    }

    #[inline(always)]
    fn prepare_screen(_p: &Self::Params, _f: &ScreenFrame<T>) -> Self::Screen {}

    #[inline(always)]
    fn prepare_ray(_p: &Self::Params, _f: &RayFrame<T>) -> Self::Ray {}

    #[inline(always)]
    fn eval_screen(_s: &Self::Screen, u: Vec2<T>) -> T {
        (-T::HALF * u.length_squared()).exp()
    }

    #[inline(always)]
    fn eval_ray(_r: &Self::Ray, u: Vec3<T>) -> T {
        (-T::HALF * u.length_squared()).exp()
    }

    #[inline(always)]
    fn d_eval_screen(s: &Self::Screen, u: Vec2<T>) -> Option<Vec2<T>> {
        Some(u * -<Gaussian as Kernel<T>>::eval_screen(s, u))
    }

    #[inline(always)]
    fn support_screen(_s: &Self::Screen, peak: T, cutoff: T) -> T {
        radius_for_cutoff(peak, cutoff)
    }

    #[inline(always)]
    fn support_ray(_r: &Self::Ray, peak: T, cutoff: T) -> T {
        radius_for_cutoff(peak, cutoff)
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

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct GaussianNormalised;

impl<T: FloatingPoint> Kernel<T> for GaussianNormalised {
    type Params = GaussianParams;
    type Screen = ();
    type Ray = ();
    type Blend = Normalised;

    const ID: KernelId = KernelId(1);
    const NAME: &'static str = "gaussian-normalised";
    const RADIAL: bool = true;
    const HAS_SCREEN_MARGINAL: bool = true;

    #[inline(always)]
    fn params_from(shape: &CurveShape) -> Self::Params {
        <Gaussian as Kernel<T>>::params_from(shape)
    }

    #[inline(always)]
    fn prepare_screen(p: &Self::Params, f: &ScreenFrame<T>) -> Self::Screen {
        <Gaussian as Kernel<T>>::prepare_screen(p, f)
    }

    #[inline(always)]
    fn prepare_ray(p: &Self::Params, f: &RayFrame<T>) -> Self::Ray {
        <Gaussian as Kernel<T>>::prepare_ray(p, f)
    }

    #[inline(always)]
    fn eval_screen(s: &Self::Screen, u: Vec2<T>) -> T {
        <Gaussian as Kernel<T>>::eval_screen(s, u)
    }

    #[inline(always)]
    fn eval_ray(r: &Self::Ray, u: Vec3<T>) -> T {
        <Gaussian as Kernel<T>>::eval_ray(r, u)
    }

    #[inline(always)]
    fn d_eval_screen(s: &Self::Screen, u: Vec2<T>) -> Option<Vec2<T>> {
        <Gaussian as Kernel<T>>::d_eval_screen(s, u)
    }

    #[inline(always)]
    fn support_screen(s: &Self::Screen, peak: T, cutoff: T) -> T {
        <Gaussian as Kernel<T>>::support_screen(s, peak, cutoff)
    }

    #[inline(always)]
    fn support_ray(r: &Self::Ray, peak: T, cutoff: T) -> T {
        <Gaussian as Kernel<T>>::support_ray(r, peak, cutoff)
    }

    fn prefilter(
        p: &Self::Params,
        cov: Sym2<T>,
        pixel_variance: T,
    ) -> (Self::Params, Sym2<T>, T) {
        <Gaussian as Kernel<T>>::prefilter(p, cov, pixel_variance)
    }
}
