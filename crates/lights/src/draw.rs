use crate::error::{LightsError, Result};
use crate::image::{FloatImage, RasterLayer};
use crate::numerics::float::FloatingPoint;
use crate::numerics::similarity::Similarity2;
use crate::numerics::vector::{Rgb, Vec2};
use rayon::prelude::*;
use std::num::NonZeroU32;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Ink<T> {
    pub premultiplied: Rgb<T>,
    pub alpha: T,
}

impl<T: FloatingPoint> Ink<T> {
    pub fn opaque(colour: Rgb<T>) -> Self {
        Self {
            premultiplied: colour,
            alpha: T::ONE,
        }
    }

    pub fn straight(colour: Rgb<T>, alpha: T) -> Self {
        Self {
            premultiplied: colour * alpha,
            alpha,
        }
    }

    pub fn premultiplied(premultiplied: Rgb<T>, alpha: T) -> Self {
        Self {
            premultiplied,
            alpha,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Clip<T> {
    pub min: Vec2<T>,
    pub max: Vec2<T>,
}

impl<T: FloatingPoint> Clip<T> {
    pub fn new(min: Vec2<T>, max: Vec2<T>) -> Self {
        Self { min, max }
    }

    pub fn pixel_fraction(&self, px: u32, py: u32) -> T {
        self.pixel_fraction_through(Similarity2::identity(), px, py)
    }

    pub fn pixel_fraction_through(&self, view: Similarity2<T>, px: u32, py: u32) -> T {
        let x0 = T::from_f64(f64::from(px));
        let y0 = T::from_f64(f64::from(py));
        if view.rotation() != T::ZERO {
            let centre = Vec2::new(x0 + T::HALF, y0 + T::HALF);
            let d = view.scale() * box_sdf(self.min, self.max, T::ZERO, view.invert(centre));
            return (T::HALF - d).clamp(T::ZERO, T::ONE);
        }
        let s = view.scale();
        let t = view.translation();
        let side = s.recip();
        let wx = (self.max.x.min((x0 + T::ONE - t.x) / s) - self.min.x.max((x0 - t.x) / s))
            .clamp(T::ZERO, side);
        let wy = (self.max.y.min((y0 + T::ONE - t.y) / s) - self.min.y.max((y0 - t.y) / s))
            .clamp(T::ZERO, side);
        (wx * s) * (wy * s)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum DrawShape<T> {
    Rect {
        min: Vec2<T>,
        max: Vec2<T>,
        corner_radius: T,
    },
    RectOutline {
        min: Vec2<T>,
        max: Vec2<T>,
        corner_radius: T,
        half_width: T,
    },
    Segment {
        a: Vec2<T>,
        b: Vec2<T>,
        half_width: T,
    },
    Disc {
        centre: Vec2<T>,
        radius: T,
    },
    Ring {
        centre: Vec2<T>,
        radius: T,
        half_width: T,
    },
    Polygon {
        points: Vec<Vec2<T>>,
    },
    Polyline {
        points: Vec<Vec2<T>>,
        closed: bool,
        half_width: T,
    },
}

fn segment_distance<T: FloatingPoint>(a: Vec2<T>, b: Vec2<T>, p: Vec2<T>) -> T {
    let ab = b - a;
    let len2 = ab.length_squared();
    let t = if len2 > T::ZERO {
        ((p - a).dot(ab) / len2).clamp(T::ZERO, T::ONE)
    } else {
        T::ZERO
    };
    (p - (a + ab * t)).length()
}

fn box_sdf<T: FloatingPoint>(min: Vec2<T>, max: Vec2<T>, corner_radius: T, p: Vec2<T>) -> T {
    let centre = (min + max) * T::HALF;
    let half = (max - min) * T::HALF;
    let r = corner_radius.min(half.x).min(half.y);
    let qx = (p.x - centre.x).abs() - (half.x - r);
    let qy = (p.y - centre.y).abs() - (half.y - r);
    let outside = Vec2::new(qx.max(T::ZERO), qy.max(T::ZERO)).length();
    let inside = qx.max(qy).min(T::ZERO);
    outside + inside - r
}

fn polygon_sdf<T: FloatingPoint>(points: &[Vec2<T>], p: Vec2<T>) -> T {
    let n = points.len();
    let mut inside = false;
    let mut nearest = T::INFINITY;
    for i in 0..n {
        let a = points[i];
        let b = points[(i + 1) % n];
        if (a.y > p.y) != (b.y > p.y) {
            let crossing = a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y);
            if p.x < crossing {
                inside = !inside;
            }
        }
        nearest = nearest.min(segment_distance(a, b, p));
    }
    if inside { -nearest } else { nearest }
}

fn polyline_distance<T: FloatingPoint>(points: &[Vec2<T>], closed: bool, p: Vec2<T>) -> T {
    let n = points.len();
    let edges = if closed { n } else { n - 1 };
    let mut nearest = T::INFINITY;
    for i in 0..edges {
        nearest = nearest.min(segment_distance(points[i], points[(i + 1) % n], p));
    }
    nearest
}

impl<T: FloatingPoint> DrawShape<T> {
    pub fn sdf(&self, p: Vec2<T>) -> T {
        match self {
            DrawShape::Rect {
                min,
                max,
                corner_radius,
            } => box_sdf(*min, *max, *corner_radius, p),
            DrawShape::RectOutline {
                min,
                max,
                corner_radius,
                half_width,
            } => box_sdf(*min, *max, *corner_radius, p).abs() - *half_width,
            DrawShape::Segment { a, b, half_width } => segment_distance(*a, *b, p) - *half_width,
            DrawShape::Disc { centre, radius } => (p - *centre).length() - *radius,
            DrawShape::Ring {
                centre,
                radius,
                half_width,
            } => ((p - *centre).length() - *radius).abs() - *half_width,
            DrawShape::Polygon { points } => polygon_sdf(points, p),
            DrawShape::Polyline {
                points,
                closed,
                half_width,
            } => polyline_distance(points, *closed, p) - *half_width,
        }
    }

    pub fn bounds(&self) -> (Vec2<T>, Vec2<T>) {
        let (lo, hi, pad) = match self {
            DrawShape::Rect { min, max, .. } => (*min, *max, T::ZERO),
            DrawShape::RectOutline {
                min,
                max,
                half_width,
                ..
            } => (*min, *max, *half_width),
            DrawShape::Segment { a, b, half_width } => (
                Vec2::new(a.x.min(b.x), a.y.min(b.y)),
                Vec2::new(a.x.max(b.x), a.y.max(b.y)),
                *half_width,
            ),
            DrawShape::Disc { centre, radius } => (*centre, *centre, *radius),
            DrawShape::Ring {
                centre,
                radius,
                half_width,
            } => (*centre, *centre, *radius + *half_width),
            DrawShape::Polygon { points } => {
                let (lo, hi) = point_bounds(points);
                (lo, hi, T::ZERO)
            }
            DrawShape::Polyline {
                points, half_width, ..
            } => {
                let (lo, hi) = point_bounds(points);
                (lo, hi, *half_width)
            }
        };
        let reach = pad + T::ONE;
        (lo - Vec2::splat(reach), hi + Vec2::splat(reach))
    }

    fn points(&self) -> Vec<Vec2<T>> {
        match self {
            DrawShape::Rect { min, max, .. } | DrawShape::RectOutline { min, max, .. } => {
                vec![*min, *max]
            }
            DrawShape::Segment { a, b, .. } => vec![*a, *b],
            DrawShape::Disc { centre, .. } | DrawShape::Ring { centre, .. } => vec![*centre],
            DrawShape::Polygon { points } | DrawShape::Polyline { points, .. } => {
                points.clone()
            }
        }
    }

    fn lengths(&self) -> Vec<T> {
        match self {
            DrawShape::Polygon { .. } => Vec::new(),
            DrawShape::Rect { corner_radius, .. } => vec![*corner_radius],
            DrawShape::RectOutline {
                corner_radius,
                half_width,
                ..
            } => vec![*corner_radius, *half_width],
            DrawShape::Segment { half_width, .. }
            | DrawShape::Polyline { half_width, .. } => vec![*half_width],
            DrawShape::Disc { radius, .. } => vec![*radius],
            DrawShape::Ring {
                radius, half_width, ..
            } => vec![*radius, *half_width],
        }
    }

    fn refusal(&self) -> Option<&'static str> {
        if self.points().iter().any(|p| !p.is_finite()) {
            return Some("a coordinate is not finite");
        }
        if self.lengths().iter().any(|l| !l.is_finite() || *l < T::ZERO) {
            return Some("a radius or half width is negative or not finite");
        }
        match self {
            DrawShape::Rect { min, max, .. } | DrawShape::RectOutline { min, max, .. } => {
                if min.x > max.x || min.y > max.y {
                    return Some("a rect's min exceeds its max");
                }
                None
            }
            DrawShape::Polygon { points } => {
                if points.len() < 3 {
                    return Some("a polygon needs at least three points");
                }
                None
            }
            DrawShape::Polyline { points, .. } => {
                if points.len() < 2 {
                    return Some("a polyline needs at least two points");
                }
                None
            }
            DrawShape::Segment { .. } | DrawShape::Disc { .. } | DrawShape::Ring { .. } => None,
        }
    }
}

fn point_bounds<T: FloatingPoint>(points: &[Vec2<T>]) -> (Vec2<T>, Vec2<T>) {
    let mut lo = Vec2::splat(T::INFINITY);
    let mut hi = Vec2::splat(T::NEG_INFINITY);
    for p in points {
        lo = Vec2::new(lo.x.min(p.x), lo.y.min(p.y));
        hi = Vec2::new(hi.x.max(p.x), hi.y.max(p.y));
    }
    (lo, hi)
}

#[derive(Clone, Debug, PartialEq)]
pub struct DrawOp<T> {
    pub shape: DrawShape<T>,
    pub ink: Ink<T>,
    pub clip: Option<Clip<T>>,
}

impl<T: FloatingPoint> DrawOp<T> {
    pub fn new(shape: DrawShape<T>, ink: Ink<T>) -> Self {
        Self {
            shape,
            ink,
            clip: None,
        }
    }

    pub fn clipped(mut self, clip: Clip<T>) -> Self {
        self.clip = Some(clip);
        self
    }

    pub fn coverage_at(&self, px: u32, py: u32) -> T {
        self.coverage_through(Similarity2::identity(), px, py)
    }

    pub fn coverage_through(&self, view: Similarity2<T>, px: u32, py: u32) -> T {
        let centre = Vec2::new(
            T::from_f64(f64::from(px)) + T::HALF,
            T::from_f64(f64::from(py)) + T::HALF,
        );
        let distance = view.scale() * self.shape.sdf(view.invert(centre));
        let shape = (T::HALF - distance).clamp(T::ZERO, T::ONE);
        match &self.clip {
            Some(clip) => shape * clip.pixel_fraction_through(view, px, py),
            None => shape,
        }
    }

    pub fn footprint(&self, view: Similarity2<T>, width: u32, height: u32) -> Option<[u32; 4]> {
        let (mut lo, mut hi) = self.shape.bounds();
        if let Some(clip) = &self.clip {
            lo = Vec2::new(lo.x.max(clip.min.x), lo.y.max(clip.min.y));
            hi = Vec2::new(hi.x.min(clip.max.x), hi.y.min(clip.max.y));
        }
        if lo.x > hi.x || lo.y > hi.y {
            return None;
        }
        let (lo, hi) = view.image_of_box(lo, hi);
        let (x0, x1) = pixel_span(lo.x, hi.x, width);
        let (y0, y1) = pixel_span(lo.y, hi.y, height);
        if x0 >= x1 || y0 >= y1 {
            return None;
        }
        Some([x0, y0, x1, y1])
    }

    fn refusal(&self) -> Option<&'static str> {
        if let Some(reason) = self.shape.refusal() {
            return Some(reason);
        }
        if !self.ink.premultiplied.is_finite() || !self.ink.alpha.is_finite() {
            return Some("an ink channel is not finite");
        }
        if self.ink.alpha < T::ZERO || self.ink.alpha > T::ONE {
            return Some("an ink alpha lies outside 0..=1");
        }
        if let Some(clip) = &self.clip {
            if !clip.min.is_finite() || !clip.max.is_finite() {
                return Some("a clip coordinate is not finite");
            }
        }
        None
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct DrawList<T> {
    ops: Vec<DrawOp<T>>,
}

fn pixel_span<T: FloatingPoint>(lo: T, hi: T, extent: u32) -> (u32, u32) {
    let top = f64::from(extent);
    let first = lo.floor().to_f64().clamp(0.0, top) as u32;
    let last = hi.ceil().to_f64().clamp(0.0, top) as u32;
    (first, last)
}

impl<T: FloatingPoint> DrawList<T> {
    pub fn new() -> Self {
        Self { ops: Vec::new() }
    }

    pub fn push(&mut self, op: DrawOp<T>) -> Result<()> {
        if let Some(reason) = op.refusal() {
            return Err(LightsError::InvalidDrawOp {
                index: self.ops.len(),
                reason,
            });
        }
        self.ops.push(op);
        Ok(())
    }

    pub fn ops(&self) -> &[DrawOp<T>] {
        &self.ops
    }

    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    pub fn render(&self, width: u32, height: u32) -> Result<RasterLayer<T>> {
        self.render_through(Similarity2::identity(), width, height)
    }

    pub fn render_through(
        &self,
        view: Similarity2<T>,
        width: u32,
        height: u32,
    ) -> Result<RasterLayer<T>> {
        self.render_banded(view, width, height, None)
    }

    pub fn render_through_parallel(
        &self,
        view: Similarity2<T>,
        width: u32,
        height: u32,
        band_rows: NonZeroU32,
    ) -> Result<RasterLayer<T>> {
        self.render_banded(view, width, height, Some(band_rows))
    }

    fn render_banded(
        &self,
        view: Similarity2<T>,
        width: u32,
        height: u32,
        band_rows: Option<NonZeroU32>,
    ) -> Result<RasterLayer<T>> {
        if width == 0 || height == 0 {
            return Err(LightsError::ViewportZero { width, height });
        }
        let mut premultiplied = FloatImage::<T>::new(width, height);
        let mut coverage = FloatImage::<T>::new(width, height);
        let spans: Vec<Option<[u32; 4]>> = self
            .ops
            .iter()
            .map(|op| op.footprint(view, width, height))
            .collect();
        match band_rows {
            None => self.composite(view, &spans, width, 0, &mut premultiplied.pixels, &mut coverage.pixels),
            Some(rows) => {
                let band = rows.get() as usize * width as usize;
                premultiplied
                    .pixels
                    .par_chunks_mut(band)
                    .zip(coverage.pixels.par_chunks_mut(band))
                    .enumerate()
                    .for_each(|(i, (p, c))| self.composite(view, &spans, width, i as u32 * rows.get(), p, c));
            }
        }
        Ok(RasterLayer {
            premultiplied,
            coverage,
        })
    }

    fn composite(
        &self,
        view: Similarity2<T>,
        spans: &[Option<[u32; 4]>],
        width: u32,
        top: u32,
        premultiplied: &mut [Rgb<T>],
        coverage: &mut [Rgb<T>],
    ) {
        let stride = width as usize;
        let bottom = top + (premultiplied.len() / stride) as u32;
        for (op, span) in self.ops.iter().zip(spans.iter()) {
            let Some([x0, y0, x1, y1]) = *span else {
                continue;
            };
            for py in y0.max(top)..y1.min(bottom) {
                let row = (py - top) as usize * stride;
                for px in x0..x1 {
                    let c = op.coverage_through(view, px, py);
                    if c <= T::ZERO {
                        continue;
                    }
                    let a = op.ink.alpha * c;
                    let keep = T::ONE - a;
                    let at = row + px as usize;
                    premultiplied[at] = op.ink.premultiplied * c + premultiplied[at] * keep;
                    coverage[at] = Rgb::splat(a + coverage[at].x * keep);
                }
            }
        }
    }
}
