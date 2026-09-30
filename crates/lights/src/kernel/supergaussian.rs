use crate::blend::Over;
use crate::kernel::frame::{RayFrame, ScreenFrame};
use crate::kernel::{CurveShape, Kernel, KernelId, moment_prefilter};
use crate::numerics::float::FloatingPoint;
use crate::numerics::symmetric::Sym2;
use crate::numerics::vector::{Vec2, Vec3};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SuperGaussianParams<T> {
    pub beta: T,
}

impl<T: FloatingPoint> Default for SuperGaussianParams<T> {
    fn default() -> Self {
        Self { beta: T::ONE }
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct SuperGaussian;

#[inline(always)]
pub fn super_gaussian_response<T: FloatingPoint>(beta: T, quadratic: T) -> T {
    (-(T::HALF * quadratic).powf(beta)).exp()
}

pub fn super_gaussian_radius<T: FloatingPoint>(beta: T, peak: T, cutoff: T) -> T {
    if !(peak > T::ZERO) || !(cutoff > T::ZERO) || !(peak > cutoff) || !(beta > T::ZERO) {
        return T::ZERO;
    }
    let ratio = peak / cutoff;
    let solved = ratio.ln().max(T::ZERO).powf(beta.recip());
    (T::TWO * solved).max(T::ZERO).sqrt()
}

impl<T: FloatingPoint> Kernel<T> for SuperGaussian {
    type Params = SuperGaussianParams<T>;
    type Screen = SuperGaussianParams<T>;
    type Ray = SuperGaussianParams<T>;
    type Blend = Over;

    const ID: KernelId = KernelId(2);
    const NAME: &'static str = "super-gaussian";
    const RADIAL: bool = true;
    const HAS_SCREEN_MARGINAL: bool = true;

    #[inline(always)]
    fn params_from(shape: &CurveShape) -> Self::Params {
        SuperGaussianParams {
            beta: T::from_f64(shape.beta),
        }
    }

    #[inline(always)]
    fn prepare_screen(p: &Self::Params, _f: &ScreenFrame<T>) -> Self::Screen {
        *p
    }

    #[inline(always)]
    fn prepare_ray(p: &Self::Params, _f: &RayFrame<T>) -> Self::Ray {
        *p
    }

    #[inline(always)]
    fn eval_screen(s: &Self::Screen, u: Vec2<T>) -> T {
        super_gaussian_response(s.beta, u.length_squared())
    }

    #[inline(always)]
    fn eval_ray(r: &Self::Ray, u: Vec3<T>) -> T {
        super_gaussian_response(r.beta, u.length_squared())
    }

    #[inline(always)]
    fn support_screen(s: &Self::Screen, peak: T, cutoff: T) -> T {
        super_gaussian_radius(s.beta, peak, cutoff)
    }

    #[inline(always)]
    fn support_ray(r: &Self::Ray, peak: T, cutoff: T) -> T {
        super_gaussian_radius(r.beta, peak, cutoff)
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
