use crate::error::{LightsError, Result};
use crate::numerics::vector::Vec2;

pub type EmPoint = Vec2<f64>;

pub const FIT_SAMPLES: usize = 32;
pub const MAX_SPLIT_DEPTH: u32 = 24;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Edge {
    Line {
        a: EmPoint,
        b: EmPoint,
    },
    Arc {
        centre: EmPoint,
        radius: f64,
        from: f64,
        to: f64,
    },
}

impl Edge {
    pub fn start(&self) -> EmPoint {
        self.point_at(0.0)
    }

    pub fn end(&self) -> EmPoint {
        self.point_at(1.0)
    }

    pub fn point_at(&self, s: f64) -> EmPoint {
        match *self {
            Edge::Line { a, b } => a + (b - a) * s,
            Edge::Arc {
                centre,
                radius,
                from,
                to,
            } => {
                let angle = from + (to - from) * s;
                centre + Vec2::new(angle.cos(), angle.sin()) * radius
            }
        }
    }

    pub fn distance(&self, p: EmPoint) -> f64 {
        match *self {
            Edge::Line { a, b } => {
                let ab = b - a;
                let denominator = ab.length_squared();
                if !(denominator > 0.0) {
                    return (p - a).length();
                }
                let t = ((p - a).dot(ab) / denominator).clamp(0.0, 1.0);
                (p - (a + ab * t)).length()
            }
            Edge::Arc {
                centre,
                radius,
                from,
                to,
            } => {
                let v = p - centre;
                let r = v.length();
                let mid = 0.5 * (from + to);
                let half = 0.5 * (to - from).abs();
                if r > 0.0 && v.dot(Vec2::new(mid.cos(), mid.sin())) >= half.cos() * r {
                    (r - radius).abs()
                } else {
                    (p - self.start()).length().min((p - self.end()).length())
                }
            }
        }
    }

    pub fn crossings(&self, p: EmPoint) -> i32 {
        self.crossings_between(self.start(), self.end(), p)
    }

    fn crossings_between(&self, start: EmPoint, end: EmPoint, p: EmPoint) -> i32 {
        match *self {
            Edge::Line { .. } => chord_crossing(start, end, p),
            Edge::Arc {
                centre,
                radius,
                from,
                to,
            } => {
                let pi = core::f64::consts::PI;
                let quarter = core::f64::consts::FRAC_PI_2;
                let (lo, hi) = (from.min(to), from.max(to));
                let first = ((lo - quarter) / pi).ceil() as i64;
                let last = ((hi - quarter) / pi).floor() as i64;
                let mut angles = vec![from];
                let mut turns: Vec<f64> = (first..=last)
                    .map(|k| quarter + k as f64 * pi)
                    .filter(|&t| t > lo && t < hi)
                    .collect();
                if to < from {
                    turns.reverse();
                }
                angles.extend(turns);
                angles.push(to);
                let at = |t: f64| centre + Vec2::new(t.cos(), t.sin()) * radius;
                let last_piece = angles.len() - 2;
                let mut w = 0;
                for (i, pair) in angles.windows(2).enumerate() {
                    let a = if i == 0 { start } else { at(pair[0]) };
                    let b = if i == last_piece { end } else { at(pair[1]) };
                    let direction = if a.y <= p.y && b.y > p.y {
                        1
                    } else if a.y > p.y && b.y <= p.y {
                        -1
                    } else {
                        0
                    };
                    if direction == 0 {
                        continue;
                    }
                    let side = if (0.5 * (pair[0] + pair[1])).cos() >= 0.0 { 1.0 } else { -1.0 };
                    let dy = p.y - centre.y;
                    let x = centre.x + side * (radius * radius - dy * dy).max(0.0).sqrt();
                    if x > p.x {
                        w += direction;
                    }
                }
                w
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Contour {
    pub edges: Vec<Edge>,
}

impl Contour {
    pub fn winding(&self, p: EmPoint) -> i32 {
        let n = self.edges.len();
        let mut w = 0;
        for (i, edge) in self.edges.iter().enumerate() {
            let end = self.edges[(i + 1) % n].start();
            w += edge.crossings_between(edge.start(), end, p);
        }
        w
    }

    pub fn distance(&self, p: EmPoint) -> f64 {
        self.edges
            .iter()
            .fold(f64::INFINITY, |d, e| d.min(e.distance(p)))
    }

    pub fn arc_count(&self) -> usize {
        self.edges
            .iter()
            .filter(|e| matches!(e, Edge::Arc { .. }))
            .count()
    }
}

pub fn nonzero_winding(contours: &[Contour], p: EmPoint) -> i32 {
    contours.iter().map(|c| c.winding(p)).sum()
}

pub fn nonzero_signed_distance(contours: &[Contour], p: EmPoint) -> f64 {
    Filled::new(contours.to_vec()).signed_distance(p)
}

pub const ON_EDGE_EM: f64 = 1e-9;
pub const SIDE_PROBE_EM: f64 = 1e-7;

#[derive(Clone, Debug, PartialEq)]
pub struct Filled {
    pub contours: Vec<Contour>,
    pub boundary: Vec<Edge>,
}

impl Filled {
    pub fn new(contours: Vec<Contour>) -> Self {
        let edges: Vec<Edge> = contours.iter().flat_map(|c| c.edges.iter().copied()).collect();
        let mut boundary = Vec::new();
        for (i, edge) in edges.iter().enumerate() {
            let mut cuts = vec![0.0, 1.0];
            for (j, other) in edges.iter().enumerate() {
                if i == j {
                    continue;
                }
                for q in curve_intersections(edge, other) {
                    if let (Some(s), Some(_)) = (edge.param_of(q), other.param_of(q)) {
                        if s > ON_EDGE_EM && s < 1.0 - ON_EDGE_EM {
                            cuts.push(s);
                        }
                    }
                }
            }
            cuts.sort_by(f64::total_cmp);
            cuts.dedup_by(|b, a| *b - *a <= ON_EDGE_EM);
            for pair in cuts.windows(2) {
                let piece = edge.sub(pair[0], pair[1]);
                let m = piece.point_at(0.5);
                let n = piece.normal_at(0.5);
                let left = nonzero_winding(&contours, m + n * SIDE_PROBE_EM) != 0;
                let right = nonzero_winding(&contours, m - n * SIDE_PROBE_EM) != 0;
                if left != right {
                    boundary.push(piece);
                }
            }
        }
        Self { contours, boundary }
    }

    pub fn mapped(&self, scale: f64, offset: EmPoint) -> Self {
        let map = |edges: &[Edge]| -> Vec<Edge> { edges.iter().map(|e| e.mapped(scale, offset)).collect() };
        Self {
            contours: self
                .contours
                .iter()
                .map(|c| Contour { edges: map(&c.edges) })
                .collect(),
            boundary: map(&self.boundary),
        }
    }

    pub fn winding(&self, p: EmPoint) -> i32 {
        nonzero_winding(&self.contours, p)
    }

    pub fn boundary_distance(&self, p: EmPoint) -> f64 {
        self.boundary
            .iter()
            .fold(f64::INFINITY, |d, e| d.min(e.distance(p)))
    }

    pub fn signed_distance(&self, p: EmPoint) -> f64 {
        let d = self.boundary_distance(p);
        if self.winding(p) != 0 { -d } else { d }
    }
}

impl Edge {
    pub fn mapped(&self, scale: f64, offset: EmPoint) -> Edge {
        match *self {
            Edge::Line { a, b } => Edge::Line {
                a: a * scale + offset,
                b: b * scale + offset,
            },
            Edge::Arc {
                centre,
                radius,
                from,
                to,
            } => {
                let turn = if scale < 0.0 { core::f64::consts::PI } else { 0.0 };
                Edge::Arc {
                    centre: centre * scale + offset,
                    radius: radius * scale.abs(),
                    from: from + turn,
                    to: to + turn,
                }
            }
        }
    }

    pub fn sub(&self, s0: f64, s1: f64) -> Edge {
        match *self {
            Edge::Line { .. } => Edge::Line {
                a: self.point_at(s0),
                b: self.point_at(s1),
            },
            Edge::Arc {
                centre,
                radius,
                from,
                to,
            } => Edge::Arc {
                centre,
                radius,
                from: from + (to - from) * s0,
                to: from + (to - from) * s1,
            },
        }
    }

    pub fn normal_at(&self, s: f64) -> EmPoint {
        let tangent = match *self {
            Edge::Line { a, b } => b - a,
            Edge::Arc { from, to, .. } => {
                let angle = from + (to - from) * s;
                Vec2::new(-angle.sin(), angle.cos()) * (to - from).signum()
            }
        };
        match Vec2::new(-tangent.y, tangent.x).normalize() {
            Some(n) => n,
            None => Vec2::new(1.0, 0.0),
        }
    }

    pub fn param_of(&self, q: EmPoint) -> Option<f64> {
        match *self {
            Edge::Line { a, b } => {
                let d = b - a;
                let length = d.length();
                if !(length > 0.0) {
                    return None;
                }
                let t = (q - a).dot(d) / (length * length);
                let off = cross(d, q - a).abs() / length;
                let slack = ON_EDGE_EM / length;
                if off <= ON_EDGE_EM && t >= -slack && t <= 1.0 + slack {
                    Some(t.clamp(0.0, 1.0))
                } else {
                    None
                }
            }
            Edge::Arc {
                centre,
                radius,
                from,
                to,
            } => {
                let v = q - centre;
                let span = to - from;
                if (v.length() - radius).abs() > ON_EDGE_EM || !(span.abs() > 0.0) {
                    return None;
                }
                let tau = 2.0 * core::f64::consts::PI;
                let theta = v.y.atan2(v.x);
                let slack = ON_EDGE_EM / (radius * span.abs());
                (-2..=2)
                    .map(|k| (theta + k as f64 * tau - from) / span)
                    .find(|s| *s >= -slack && *s <= 1.0 + slack)
                    .map(|s| s.clamp(0.0, 1.0))
            }
        }
    }
}

fn curve_intersections(e: &Edge, f: &Edge) -> Vec<EmPoint> {
    match (*e, *f) {
        (Edge::Line { a: a0, b: b0 }, Edge::Line { a: a1, b: b1 }) => {
            let (d0, d1) = (b0 - a0, b1 - a1);
            let denominator = cross(d0, d1);
            if !(denominator.abs() > 0.0) {
                return Vec::new();
            }
            vec![a0 + d0 * (cross(a1 - a0, d1) / denominator)]
        }
        (Edge::Line { a, b }, Edge::Arc { centre, radius, .. })
        | (Edge::Arc { centre, radius, .. }, Edge::Line { a, b }) => {
            line_circle(a, b, centre, radius)
        }
        (
            Edge::Arc {
                centre: c0,
                radius: r0,
                ..
            },
            Edge::Arc {
                centre: c1,
                radius: r1,
                ..
            },
        ) => circle_circle(c0, r0, c1, r1),
    }
}

fn line_circle(a: EmPoint, b: EmPoint, centre: EmPoint, radius: f64) -> Vec<EmPoint> {
    let d = b - a;
    let f = a - centre;
    let qa = d.dot(d);
    if !(qa > 0.0) {
        return Vec::new();
    }
    let qb = 2.0 * f.dot(d);
    let qc = f.dot(f) - radius * radius;
    let discriminant = qb * qb - 4.0 * qa * qc;
    if discriminant < 0.0 {
        return Vec::new();
    }
    let root = discriminant.sqrt();
    [(-qb - root) / (2.0 * qa), (-qb + root) / (2.0 * qa)]
        .iter()
        .map(|t| a + d * *t)
        .collect()
}

fn circle_circle(c0: EmPoint, r0: f64, c1: EmPoint, r1: f64) -> Vec<EmPoint> {
    let d = c1 - c0;
    let distance = d.length();
    if !(distance > 0.0) || distance > r0 + r1 || distance < (r0 - r1).abs() {
        return Vec::new();
    }
    let along = (r0 * r0 - r1 * r1 + distance * distance) / (2.0 * distance);
    let h = (r0 * r0 - along * along).max(0.0).sqrt();
    let mid = c0 + d * (along / distance);
    let perp = Vec2::new(-d.y, d.x) * (h / distance);
    vec![mid + perp, mid - perp]
}

fn cross(u: EmPoint, v: EmPoint) -> f64 {
    u.x * v.y - u.y * v.x
}

fn chord_crossing(a: EmPoint, b: EmPoint, p: EmPoint) -> i32 {
    let side = cross(b - a, p - a);
    if a.y <= p.y {
        if b.y > p.y && side > 0.0 { 1 } else { 0 }
    } else if b.y <= p.y && side < 0.0 {
        -1
    } else {
        0
    }
}

pub fn quad_point(p0: EmPoint, p1: EmPoint, p2: EmPoint, t: f64) -> EmPoint {
    let u = 1.0 - t;
    p0 * (u * u) + p1 * (2.0 * u * t) + p2 * (t * t)
}

fn circle_through(a: EmPoint, m: EmPoint, b: EmPoint) -> Option<(EmPoint, f64)> {
    let d = 2.0 * cross(m - a, b - a);
    if !(d.abs() > 0.0) {
        return None;
    }
    let am = m - a;
    let ab = b - a;
    let lm = am.length_squared();
    let lb = ab.length_squared();
    let offset = Vec2::new(ab.y * lm - am.y * lb, am.x * lb - ab.x * lm) * (1.0 / d);
    let centre = a + offset;
    let radius = offset.length();
    if centre.is_finite() && radius.is_finite() {
        Some((centre, radius))
    } else {
        None
    }
}

fn arc_through(a: EmPoint, m: EmPoint, b: EmPoint) -> Option<Edge> {
    let (centre, radius) = circle_through(a, m, b)?;
    let angle = |q: EmPoint| (q.y - centre.y).atan2(q.x - centre.x);
    let from = angle(a);
    let tau = core::f64::consts::TAU;
    let to = if cross(m - a, b - m) > 0.0 {
        from + (angle(b) - from).rem_euclid(tau)
    } else {
        from - (from - angle(b)).rem_euclid(tau)
    };
    Some(Edge::Arc {
        centre,
        radius,
        from,
        to,
    })
}

fn deviation<F: Fn(f64) -> EmPoint>(curve: &F, t0: f64, t1: f64, sag: f64, edge: &Edge) -> f64 {
    let mut forward: f64 = 0.0;
    for k in 1..FIT_SAMPLES {
        let t = t0 + (t1 - t0) * (k as f64 / FIT_SAMPLES as f64);
        forward = forward.max(edge.distance(curve(t)));
    }
    let pieces = 4 * FIT_SAMPLES;
    let chords: Vec<Edge> = (0..pieces)
        .map(|k| Edge::Line {
            a: curve(t0 + (t1 - t0) * (k as f64 / pieces as f64)),
            b: curve(t0 + (t1 - t0) * ((k + 1) as f64 / pieces as f64)),
        })
        .collect();
    let step = (t1 - t0) / pieces as f64;
    let mut backward: f64 = 0.0;
    for k in 1..FIT_SAMPLES {
        let q = edge.point_at(k as f64 / FIT_SAMPLES as f64);
        let near = chords
            .iter()
            .fold(f64::INFINITY, |d, c| d.min(c.distance(q)));
        backward = backward.max(near + sag * step * step);
    }
    forward.max(backward)
}

fn fit_piece<F: Fn(f64) -> EmPoint>(
    curve: &F,
    t0: f64,
    t1: f64,
    sag: f64,
    tolerance: f64,
    depth: u32,
    out: &mut Vec<Edge>,
) {
    let a = curve(t0);
    let b = curve(t1);
    let line = Edge::Line { a, b };
    let line_error = deviation(curve, t0, t1, sag, &line);
    if line_error <= tolerance {
        out.push(line);
        return;
    }
    let arc = arc_through(a, curve(0.5 * (t0 + t1)), b);
    let arc_error = arc.map(|e| deviation(curve, t0, t1, sag, &e));
    if let (Some(edge), Some(error)) = (arc, arc_error) {
        if error <= tolerance {
            out.push(edge);
            return;
        }
    }
    if depth >= MAX_SPLIT_DEPTH {
        match (arc, arc_error) {
            (Some(edge), Some(error)) if error < line_error => out.push(edge),
            _ => out.push(line),
        }
        return;
    }
    let mid = 0.5 * (t0 + t1);
    fit_piece(curve, t0, mid, sag, tolerance, depth + 1, out);
    fit_piece(curve, mid, t1, sag, tolerance, depth + 1, out);
}

pub fn quad_to_edges(p0: EmPoint, p1: EmPoint, p2: EmPoint, tolerance: f64) -> Vec<Edge> {
    let mut out = Vec::new();
    let curve = |t: f64| quad_point(p0, p1, p2, t);
    let sag = 0.25 * (p0 - p1 * 2.0 + p2).length();
    fit_piece(&curve, 0.0, 1.0, sag, tolerance, 0, &mut out);
    out
}

pub fn cubic_point(p0: EmPoint, p1: EmPoint, p2: EmPoint, p3: EmPoint, t: f64) -> EmPoint {
    let u = 1.0 - t;
    p0 * (u * u * u) + p1 * (3.0 * u * u * t) + p2 * (3.0 * u * t * t) + p3 * (t * t * t)
}

pub fn cubic_to_edges(p0: EmPoint, p1: EmPoint, p2: EmPoint, p3: EmPoint, tolerance: f64) -> Vec<Edge> {
    let mut out = Vec::new();
    let curve = |t: f64| cubic_point(p0, p1, p2, p3, t);
    let sag = 0.75 * (p0 - p1 * 2.0 + p2).length().max((p1 - p2 * 2.0 + p3).length());
    fit_piece(&curve, 0.0, 1.0, sag, tolerance, 0, &mut out);
    out
}

#[derive(Clone, Debug)]
pub struct ContourBuilder {
    tolerance: f64,
    contours: Vec<Contour>,
    edges: Vec<Edge>,
    start: EmPoint,
    pen: EmPoint,
    open: bool,
    refused: Option<&'static str>,
}

impl ContourBuilder {
    pub fn new(tolerance_em: f64) -> Result<Self> {
        if !(tolerance_em > 0.0 && tolerance_em.is_finite()) {
            return Err(LightsError::InvalidOutline(
                "the arc tolerance must be a positive, finite length in em",
            ));
        }
        Ok(Self {
            tolerance: tolerance_em,
            contours: Vec::new(),
            edges: Vec::new(),
            start: Vec2::zero(),
            pen: Vec2::zero(),
            open: false,
            refused: None,
        })
    }

    pub fn tolerance(&self) -> f64 {
        self.tolerance
    }

    fn accept(&mut self, points: &[EmPoint]) -> bool {
        if self.refused.is_some() {
            return false;
        }
        if points.iter().any(|p| !p.is_finite()) {
            self.refused = Some("a coordinate is not finite");
            return false;
        }
        if !self.open {
            self.refused = Some("a drawing command came before any move_to");
            return false;
        }
        true
    }

    pub fn move_to(&mut self, p: EmPoint) {
        if self.open {
            self.close();
        }
        self.open = true;
        if self.accept(&[p]) {
            self.start = p;
            self.pen = p;
        }
    }

    pub fn line_to(&mut self, p: EmPoint) {
        if !self.accept(&[p]) {
            return;
        }
        if p != self.pen {
            self.edges.push(Edge::Line { a: self.pen, b: p });
        }
        self.pen = p;
    }

    pub fn quad_to(&mut self, control: EmPoint, p: EmPoint) {
        if !self.accept(&[control, p]) {
            return;
        }
        if p != self.pen || control != self.pen {
            let edges = quad_to_edges(self.pen, control, p, self.tolerance);
            self.edges.extend(edges);
        }
        self.pen = p;
    }

    pub fn cubic_to(&mut self, c1: EmPoint, c2: EmPoint, p: EmPoint) {
        if !self.accept(&[c1, c2, p]) {
            return;
        }
        if p != self.pen || c1 != self.pen || c2 != self.pen {
            let edges = cubic_to_edges(self.pen, c1, c2, p, self.tolerance);
            self.edges.extend(edges);
        }
        self.pen = p;
    }

    pub fn close(&mut self) {
        if !self.open {
            return;
        }
        if self.pen != self.start {
            self.edges.push(Edge::Line {
                a: self.pen,
                b: self.start,
            });
        }
        self.open = false;
        self.pen = self.start;
        let edges = core::mem::take(&mut self.edges);
        if !edges.is_empty() {
            self.contours.push(Contour { edges });
        }
    }

    pub fn finish(mut self) -> Result<Vec<Contour>> {
        self.close();
        match self.refused {
            Some(reason) => Err(LightsError::InvalidOutline(reason)),
            None => Ok(self.contours),
        }
    }
}
