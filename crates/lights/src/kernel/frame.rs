use crate::numerics::float::FloatingPoint;
use crate::numerics::symmetric::{LowerTri2, LowerTri3, Sym2, Sym3};
use crate::numerics::vector::{Vec2, Vec3};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ScreenFrame<T> {
    pub cov: Sym2<T>,
    pub conic: Sym2<T>,
    pub l: LowerTri2<T>,
    pub l_inv: LowerTri2<T>,
}

impl<T: FloatingPoint> ScreenFrame<T> {
    pub fn from_covariance(cov: Sym2<T>) -> Option<Self> {
        let conic = cov.inverse()?;
        let l = cov.cholesky()?;
        let l_inv = l.inverse();
        if !(conic.is_finite() && l.is_finite() && l_inv.is_finite()) {
            return None;
        }
        Some(Self {
            cov,
            conic,
            l,
            l_inv,
        })
    }

    #[inline(always)]
    pub fn whiten(&self, offset: Vec2<T>) -> Vec2<T> {
        self.l_inv.apply(offset)
    }

    #[inline(always)]
    pub fn whiten_step_x(&self) -> Vec2<T> {
        Vec2::new(self.l_inv.a, self.l_inv.b)
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RayFrame<T> {
    pub lambda: Sym3<T>,
    pub whiten: LowerTri3<T>,
    pub direction: Vec3<T>,
}

impl<T: FloatingPoint> RayFrame<T> {
    pub fn new(lambda: Sym3<T>, whiten: LowerTri3<T>, direction: Vec3<T>) -> Self {
        Self {
            lambda,
            whiten,
            direction,
        }
    }
}
