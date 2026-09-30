use crate::error::{LightsError, Result};
use crate::image::{FloatImage, RasterLayer};
use crate::numerics::float::FloatingPoint;
use crate::numerics::similarity::Similarity2;
use crate::numerics::vector::{Rgb, Vec2};
use crate::outline::Filled;

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FeatureId(pub u16);

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Primitive<T> {
    HalfPlane {
        point: Vec2<T>,
        normal: Vec2<T>,
    },
    Segment {
        a: Vec2<T>,
        b: Vec2<T>,
        half_width: T,
    },
    Arc {
        centre: Vec2<T>,
        radius: T,
        from: T,
        to: T,
        half_width: T,
    },
    Disc {
        centre: Vec2<T>,
        radius: T,
    },
}

impl<T: FloatingPoint> Primitive<T> {
    pub fn half_plane(point: Vec2<T>, normal: Vec2<T>) -> Self {
        let unit = match normal.normalize() {
            Some(n) => n,
            None => Vec2::new(T::ONE, T::ZERO),
        };
        Primitive::HalfPlane {
            point,
            normal: unit,
        }
    }

    pub fn sdf(&self, p: Vec2<T>) -> T {
        match self {
            Primitive::HalfPlane { point, normal } => (p - *point).dot(*normal),
            Primitive::Segment { a, b, half_width } => {
                (p - segment_closest(*a, *b, p)).length() - *half_width
            }
            Primitive::Arc {
                centre,
                radius,
                from,
                to,
                half_width,
            } => {
                let v = p - *centre;
                let r = v.length();
                if arc_in_sweep(v, r, *from, *to) {
                    (r - *radius).abs() - *half_width
                } else {
                    let (lo, hi) = arc_endpoints(*centre, *radius, *from, *to);
                    (p - lo).length().min((p - hi).length()) - *half_width
                }
            }
            Primitive::Disc { centre, radius } => (p - *centre).length() - *radius,
        }
    }

    pub fn gradient(&self, p: Vec2<T>) -> Vec2<T> {
        let fallback = Vec2::new(T::ONE, T::ZERO);
        match self {
            Primitive::HalfPlane { normal, .. } => *normal,
            Primitive::Segment { a, b, .. } => {
                match (p - segment_closest(*a, *b, p)).normalize() {
                    Some(n) => n,
                    None => fallback,
                }
            }
            Primitive::Arc {
                centre,
                radius,
                from,
                to,
                ..
            } => {
                let v = p - *centre;
                let r = v.length();
                if arc_in_sweep(v, r, *from, *to) {
                    let outward = match v.normalize() {
                        Some(n) => n,
                        None => fallback,
                    };
                    if r >= *radius { outward } else { -outward }
                } else {
                    let (lo, hi) = arc_endpoints(*centre, *radius, *from, *to);
                    let near = if (p - lo).length() <= (p - hi).length() {
                        lo
                    } else {
                        hi
                    };
                    match (p - near).normalize() {
                        Some(n) => n,
                        None => fallback,
                    }
                }
            }
            Primitive::Disc { centre, .. } => match (p - *centre).normalize() {
                Some(n) => n,
                None => fallback,
            },
        }
    }

    pub fn scalar_count(&self) -> usize {
        match self {
            Primitive::HalfPlane { .. } => 4,
            Primitive::Segment { .. } => 5,
            Primitive::Arc { .. } => 6,
            Primitive::Disc { .. } => 3,
        }
    }
}

fn segment_closest<T: FloatingPoint>(a: Vec2<T>, b: Vec2<T>, p: Vec2<T>) -> Vec2<T> {
    let ab = b - a;
    let denominator = ab.length_squared();
    if !(denominator > T::ZERO) {
        return a;
    }
    let t = ((p - a).dot(ab) / denominator).clamp(T::ZERO, T::ONE);
    a + ab * t
}

fn arc_in_sweep<T: FloatingPoint>(v: Vec2<T>, r: T, from: T, to: T) -> bool {
    if !(r > T::ZERO) {
        return false;
    }
    let mid = (from + to) * T::HALF;
    let half = (to - from) * T::HALF;
    let (mid_sin, mid_cos) = mid.sin_cos();
    let (_, half_cos) = half.sin_cos();
    v.dot(Vec2::new(mid_cos, mid_sin)) >= half_cos * r
}

fn arc_endpoints<T: FloatingPoint>(
    centre: Vec2<T>,
    radius: T,
    from: T,
    to: T,
) -> (Vec2<T>, Vec2<T>) {
    let (from_sin, from_cos) = from.sin_cos();
    let (to_sin, to_cos) = to.sin_cos();
    (
        centre + Vec2::new(from_cos, from_sin) * radius,
        centre + Vec2::new(to_cos, to_sin) * radius,
    )
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Part<T> {
    pub id: FeatureId,
    pub shape: Primitive<T>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Region {
    pub id: FeatureId,
    pub shape: Filled,
}

impl Region {
    pub fn mapped(&self, scale: f64, offset: Vec2<f64>) -> Self {
        Region {
            id: self.id,
            shape: self.shape.mapped(scale, offset),
        }
    }

    pub fn sdf<T: FloatingPoint>(&self, p: Vec2<T>) -> T {
        T::from_f64(self.shape.signed_distance(Vec2::new(p.x.to_f64(), p.y.to_f64())))
    }

    pub fn gradient<T: FloatingPoint>(&self, p: Vec2<T>) -> Vec2<T> {
        let q = Vec2::new(p.x.to_f64(), p.y.to_f64());
        let h = REGION_GRADIENT_STEP * (1.0 + q.x.abs().max(q.y.abs()));
        let f = |d: Vec2<f64>| self.shape.signed_distance(q + d);
        let g = Vec2::new(
            f(Vec2::new(h, 0.0)) - f(Vec2::new(-h, 0.0)),
            f(Vec2::new(0.0, h)) - f(Vec2::new(0.0, -h)),
        );
        match g.normalize() {
            Some(n) => Vec2::new(T::from_f64(n.x), T::from_f64(n.y)),
            None => Vec2::new(T::ONE, T::ZERO),
        }
    }

    pub fn scalar_count(&self) -> usize {
        self.shape
            .contours
            .iter()
            .flat_map(|c| c.edges.iter())
            .map(|e| match e {
                crate::outline::Edge::Line { .. } => 4,
                crate::outline::Edge::Arc { .. } => 5,
            })
            .sum()
    }
}

pub const REGION_GRADIENT_STEP: f64 = 1e-7;

#[derive(Clone, Debug, PartialEq)]
pub struct PrimitiveSet<T> {
    pub ink: Rgb<T>,
    pub parts: Vec<Part<T>>,
    pub regions: Vec<Region>,
    pub holes: Vec<Part<T>>,
}

impl<T: FloatingPoint> PrimitiveSet<T> {
    pub fn combined(&self, p: Vec2<T>) -> T {
        let mut solid = T::INFINITY;
        for part in self.parts.iter() {
            solid = solid.min(part.shape.sdf(p));
        }
        for region in self.regions.iter() {
            solid = solid.min(region.sdf(p));
        }
        let mut carved = T::INFINITY;
        for hole in self.holes.iter() {
            carved = carved.min(hole.shape.sdf(p));
        }
        solid.max(-carved)
    }

    pub fn coverage_through(&self, view: Similarity2<T>, viewport: (u32, u32)) -> FloatImage<T> {
        let (width, height) = viewport;
        let mut image = FloatImage::<T>::new(width, height);
        for py in 0..height {
            for px in 0..width {
                let centre = Vec2::new(
                    T::from_f64(px as f64) + T::HALF,
                    T::from_f64(py as f64) + T::HALF,
                );
                let distance = view.scale() * self.combined(view.invert(centre));
                let c = (T::HALF - distance).clamp(T::ZERO, T::ONE);
                image.set(px, py, Rgb::splat(c));
            }
        }
        image
    }

    pub fn render(&self, view: Similarity2<T>, viewport: (u32, u32)) -> Result<RasterLayer<T>> {
        let (width, height) = viewport;
        if width == 0 || height == 0 {
            return Err(LightsError::ViewportZero { width, height });
        }
        let coverage = self.coverage_through(view, viewport);
        let premultiplied = FloatImage {
            width,
            height,
            pixels: coverage.pixels.iter().map(|c| self.ink * c.x).collect(),
        };
        Ok(RasterLayer {
            premultiplied,
            coverage,
        })
    }
}
