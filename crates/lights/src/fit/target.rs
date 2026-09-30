pub use crate::primitive::{FeatureId, Part, Primitive, PrimitiveSet};
use crate::numerics::float::FloatingPoint;
use crate::numerics::vector::{Rgb, Vec2};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Side {
    Inside,
    Outside,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ProbeKind {
    Edge,
    Stroke,
    Interior,
    Apex,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct EdgeProbe<T> {
    pub feature: FeatureId,
    pub anchor: Vec2<T>,
    pub window: T,
    pub kind: ProbeKind,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Target<T> {
    pub name: &'static str,
    pub viewport: (u32, u32),
    pub set: PrimitiveSet<T>,
    pub paper: Rgb<T>,
    pub probes: Vec<EdgeProbe<T>>,
}

impl<T: FloatingPoint> Target<T> {
    pub fn combined(&self, p: Vec2<T>) -> T {
        self.set.combined(p)
    }

    pub fn classify(&self, p: Vec2<T>) -> Side {
        if self.combined(p) <= T::ZERO {
            Side::Inside
        } else {
            Side::Outside
        }
    }

    pub fn clearance(&self, p: Vec2<T>) -> T {
        let mut best = T::INFINITY;
        for part in self.set.parts.iter().chain(self.set.holes.iter()) {
            best = best.min(part.shape.sdf(p).abs());
        }
        for region in self.set.regions.iter() {
            best = best.min(region.sdf(p).abs());
        }
        best
    }

    pub fn sdf(&self, p: Vec2<T>) -> T {
        let d = self.clearance(p);
        match self.classify(p) {
            Side::Inside => -d,
            Side::Outside => d,
        }
    }

    pub fn nearest_feature(&self, p: Vec2<T>) -> FeatureId {
        let mut best = T::INFINITY;
        let mut id = FeatureId(u16::MAX);
        for part in self.set.parts.iter().chain(self.set.holes.iter()) {
            let d = part.shape.sdf(p).abs();
            if d < best {
                best = d;
                id = part.id;
            }
        }
        for region in self.set.regions.iter() {
            let d = region.sdf(p).abs();
            if d < best {
                best = d;
                id = region.id;
            }
        }
        id
    }

    pub fn nearest_surface(&self, p: Vec2<T>) -> (T, Vec2<T>) {
        let mut best = T::INFINITY;
        let mut signed = T::INFINITY;
        let mut direction = Vec2::new(T::ONE, T::ZERO);
        for part in self.set.parts.iter() {
            let value = part.shape.sdf(p);
            if value.abs() < best {
                best = value.abs();
                signed = value;
                direction = part.shape.gradient(p);
            }
        }
        for region in self.set.regions.iter() {
            let value = region.sdf(p);
            if value.abs() < best {
                best = value.abs();
                signed = value;
                direction = region.gradient(p);
            }
        }
        for hole in self.set.holes.iter() {
            let value = hole.shape.sdf(p);
            if value.abs() < best {
                best = value.abs();
                signed = -value;
                direction = -hole.shape.gradient(p);
            }
        }
        (signed, direction)
    }

    pub fn gradient(&self, p: Vec2<T>) -> Vec2<T> {
        self.nearest_surface(p).1
    }

    pub fn primitive_count(&self) -> usize {
        self.set.parts.len() + self.set.regions.len() + self.set.holes.len()
    }

    pub fn description_bytes(&self) -> usize {
        self.description_bytes_at(core::mem::size_of::<T>())
    }

    pub fn description_bytes_at(&self, scalar: usize) -> usize {
        let mut total = 8 + 6 * scalar;
        for part in self.set.parts.iter().chain(self.set.holes.iter()) {
            total += 3 + part.shape.scalar_count() * scalar;
        }
        for region in self.set.regions.iter() {
            total += 3 + region.scalar_count() * scalar;
        }
        total
    }

    pub fn feature_ids(&self) -> Vec<FeatureId> {
        self.set.parts
            .iter()
            .chain(self.set.holes.iter())
            .map(|p| p.id)
            .chain(self.set.regions.iter().map(|r| r.id))
            .collect()
    }
}

fn v<T: FloatingPoint>(x: f64, y: f64) -> Vec2<T> {
    Vec2::new(T::from_f64(x), T::from_f64(y))
}

fn direction<T: FloatingPoint>(degrees: f64) -> Vec2<T> {
    let (s, c) = degrees.to_radians().sin_cos();
    v(c, s)
}

fn turn<T: FloatingPoint>(p: Vec2<T>, centre: Vec2<T>, radians: T) -> Vec2<T> {
    let (s, c) = radians.sin_cos();
    let d = p - centre;
    centre + Vec2::new(d.x * c - d.y * s, d.x * s + d.y * c)
}

fn part<T: FloatingPoint>(id: u16, shape: Primitive<T>) -> Part<T> {
    Part {
        id: FeatureId(id),
        shape,
    }
}

fn probe<T: FloatingPoint>(
    feature: u16,
    anchor: Vec2<T>,
    window: f64,
    kind: ProbeKind,
) -> EdgeProbe<T> {
    EdgeProbe {
        feature: FeatureId(feature),
        anchor,
        window: T::from_f64(window),
        kind,
    }
}

pub const EDGE_DEGREES: f64 = 30.0;
pub const CORNER_DEGREES: f64 = 120.0;
pub const STROKE_HALF_WIDTH: f64 = 0.5;
pub const GLYPH_EM: f64 = 12.0;
pub const SUBPIXEL_X: f64 = 0.5773502691896258;
pub const SUBPIXEL_Y: f64 = 0.3779644730092272;
pub const CURVE_RADIUS: f64 = 3.0;
pub const GLYPH_TURN_DEGREES: f64 = EDGE_DEGREES;
pub const GLYPH_NOMINAL_PIVOT: (f64, f64) = (16.0, 19.0);

pub fn curve_centre<T: FloatingPoint>() -> Vec2<T> {
    v(16.0 + SUBPIXEL_X, 16.0 + SUBPIXEL_Y)
}

pub fn glyph_pivot<T: FloatingPoint>() -> Vec2<T> {
    v(
        GLYPH_NOMINAL_PIVOT.0 + SUBPIXEL_X,
        GLYPH_NOMINAL_PIVOT.1 + SUBPIXEL_Y,
    )
}

pub fn glyph_point<T: FloatingPoint>(x: f64, y: f64) -> Vec2<T> {
    let nominal = v::<T>(GLYPH_NOMINAL_PIVOT.0, GLYPH_NOMINAL_PIVOT.1);
    let pivot = glyph_pivot::<T>();
    turn(v::<T>(x, y) - nominal + pivot, pivot, T::from_f64(GLYPH_TURN_DEGREES.to_radians()))
}

impl<T: FloatingPoint> Target<T> {
    pub fn ink_on_paper(
        name: &'static str,
        viewport: (u32, u32),
        parts: Vec<Part<T>>,
        holes: Vec<Part<T>>,
        probes: Vec<EdgeProbe<T>>,
    ) -> Self {
        Self {
            name,
            viewport,
            set: PrimitiveSet {
                ink: Rgb::new(T::ONE, T::ONE, T::ONE),
                parts,
                regions: Vec::new(),
                holes,
            },
            paper: Rgb::new(T::ZERO, T::ZERO, T::ZERO),
            probes,
        }
    }

    pub fn t1_flat_fill() -> Self {
        Self::ink_on_paper(
            "t1-flat-fill",
            (64, 64),
            vec![part(
                0,
                Primitive::Disc {
                    centre: v(32.0, 32.0),
                    radius: T::from_f64(200.0),
                },
            )],
            Vec::new(),
            vec![probe(0, v(32.0, 32.0), 24.0, ProbeKind::Interior)],
        )
    }

    pub fn t2_straight_edge() -> Self {
        Self::ink_on_paper(
            "t2-straight-edge",
            (64, 64),
            vec![part(
                0,
                Primitive::half_plane(v(32.0, 32.0), direction(EDGE_DEGREES)),
            )],
            Vec::new(),
            vec![probe(0, v(32.0, 32.0), 12.0, ProbeKind::Edge)],
        )
    }

    pub fn t3_right_angle_corner() -> Self {
        let apex = v::<T>(32.0, 32.0);
        let first = direction::<T>(EDGE_DEGREES);
        let second = direction::<T>(CORNER_DEGREES);
        let along_first = Vec2::new(-first.y, first.x);
        let along_second = Vec2::new(-second.y, second.x);
        let on_first = apex - along_first * T::from_f64(8.0);
        let on_second = apex + along_second * T::from_f64(8.0);
        Self::ink_on_paper(
            "t3-right-angle-corner",
            (64, 64),
            vec![part(0, Primitive::half_plane(apex, first))],
            vec![part(1, Primitive::half_plane(apex, -second))],
            vec![
                probe(0, on_first, 6.0, ProbeKind::Edge),
                probe(1, on_second, 6.0, ProbeKind::Edge),
                probe(0, apex, 5.0, ProbeKind::Apex),
            ],
        )
    }

    pub fn t4_curve() -> Self {
        Self::ink_on_paper(
            "t4-curve",
            (32, 32),
            vec![part(
                0,
                Primitive::Disc {
                    centre: curve_centre(),
                    radius: T::from_f64(CURVE_RADIUS),
                },
            )],
            Vec::new(),
            vec![probe(
                0,
                curve_centre::<T>() + v(CURVE_RADIUS, 0.0),
                8.0,
                ProbeKind::Edge,
            )],
        )
    }

    pub fn t5_thin_stroke() -> Self {
        let centre = v::<T>(32.0, 32.0);
        let along = direction::<T>(EDGE_DEGREES) * T::from_f64(12.0);
        Self::ink_on_paper(
            "t5-thin-stroke",
            (64, 64),
            vec![part(
                0,
                Primitive::Segment {
                    a: centre - along,
                    b: centre + along,
                    half_width: T::from_f64(STROKE_HALF_WIDTH),
                },
            )],
            Vec::new(),
            vec![probe(0, centre, 10.0, ProbeKind::Stroke)],
        )
    }

    pub fn t6_glyph_like() -> Self {
        let half = T::PI * T::HALF;
        let theta = T::from_f64(GLYPH_TURN_DEGREES.to_radians());
        let g = glyph_point::<T>;
        Self::ink_on_paper(
            "t6-glyph-like",
            (32, 32),
            vec![
                part(
                    0,
                    Primitive::Segment {
                        a: g(12.0, 10.0),
                        b: g(12.0, 22.0),
                        half_width: T::from_f64(STROKE_HALF_WIDTH),
                    },
                ),
                part(
                    1,
                    Primitive::Segment {
                        a: g(16.0, 16.0),
                        b: g(16.0, 22.0),
                        half_width: T::from_f64(STROKE_HALF_WIDTH),
                    },
                ),
                part(
                    2,
                    Primitive::Arc {
                        centre: glyph_pivot(),
                        radius: T::from_f64(CURVE_RADIUS),
                        from: -half + theta,
                        to: half + theta,
                        half_width: T::from_f64(STROKE_HALF_WIDTH),
                    },
                ),
                part(
                    3,
                    Primitive::Disc {
                        centre: g(12.0, 10.0),
                        radius: T::from_f64(STROKE_HALF_WIDTH),
                    },
                ),
                part(
                    4,
                    Primitive::Disc {
                        centre: g(12.0, 22.0),
                        radius: T::from_f64(STROKE_HALF_WIDTH),
                    },
                ),
            ],
            vec![part(
                5,
                Primitive::Disc {
                    centre: glyph_pivot(),
                    radius: T::from_f64(1.6),
                },
            )],
            vec![
                probe(0, g(12.0, 14.0), 5.0, ProbeKind::Stroke),
                probe(2, g(19.0, 19.0), 3.0, ProbeKind::Stroke),
                probe(0, g(12.0, 9.5), 1.5, ProbeKind::Apex),
            ],
        )
    }

    pub fn all() -> Vec<Self> {
        vec![
            Self::t1_flat_fill(),
            Self::t2_straight_edge(),
            Self::t3_right_angle_corner(),
            Self::t4_curve(),
            Self::t5_thin_stroke(),
            Self::t6_glyph_like(),
        ]
    }
}
