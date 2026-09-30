use crate::error::{LightsError, Result};
use crate::numerics::float::FloatingPoint;
use crate::numerics::vector::Vec2;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Similarity2<T> {
    scale: T,
    rotation: T,
    translation: Vec2<T>,
}

impl<T: FloatingPoint> Similarity2<T> {
    pub fn new(scale: T, rotation: T, translation: Vec2<T>) -> Result<Self> {
        let finite = scale.is_finite()
            && rotation.is_finite()
            && translation.x.is_finite()
            && translation.y.is_finite();
        if !finite {
            return Err(LightsError::NonFinite { what: "Similarity2" });
        }
        if scale <= T::ZERO {
            return Err(LightsError::InvalidCamera("Similarity2 scale must be positive"));
        }
        Ok(Self {
            scale,
            rotation,
            translation,
        })
    }

    pub fn identity() -> Self {
        Self {
            scale: T::ONE,
            rotation: T::ZERO,
            translation: Vec2::new(T::ZERO, T::ZERO),
        }
    }

    pub fn uniform(factor: T) -> Self {
        Self {
            scale: factor,
            rotation: T::ZERO,
            translation: Vec2::new(T::ZERO, T::ZERO),
        }
    }

    pub fn scale(self) -> T {
        self.scale
    }

    pub fn rotation(self) -> T {
        self.rotation
    }

    pub fn translation(self) -> Vec2<T> {
        self.translation
    }

    pub fn apply(self, p: Vec2<T>) -> Vec2<T> {
        let (sin, cos) = self.rotation.sin_cos();
        Vec2::new(
            self.scale * (cos * p.x - sin * p.y) + self.translation.x,
            self.scale * (sin * p.x + cos * p.y) + self.translation.y,
        )
    }

    pub fn invert(self, q: Vec2<T>) -> Vec2<T> {
        let (sin, cos) = self.rotation.sin_cos();
        let d = q - self.translation;
        Vec2::new(
            (cos * d.x + sin * d.y) / self.scale,
            (cos * d.y - sin * d.x) / self.scale,
        )
    }

    pub fn image_of_box(self, lo: Vec2<T>, hi: Vec2<T>) -> (Vec2<T>, Vec2<T>) {
        let corners = [
            self.apply(lo),
            self.apply(Vec2::new(hi.x, lo.y)),
            self.apply(Vec2::new(lo.x, hi.y)),
            self.apply(hi),
        ];
        let mut min = corners[0];
        let mut max = corners[0];
        for c in corners.iter() {
            min = Vec2::new(min.x.min(c.x), min.y.min(c.y));
            max = Vec2::new(max.x.max(c.x), max.y.max(c.y));
        }
        (min, max)
    }

    pub fn then(self, outer: Similarity2<T>) -> Self {
        Self {
            scale: self.scale * outer.scale,
            rotation: self.rotation + outer.rotation,
            translation: outer.apply(self.translation),
        }
    }
}
