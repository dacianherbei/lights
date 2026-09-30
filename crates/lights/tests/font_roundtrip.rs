use lights::numerics::vector::Vec2;
use lights::outline::{Contour, ContourBuilder, EmPoint, nonzero_winding, quad_point};
use ttf_parser::{Face, OutlineBuilder};

const UBUNTU_LIGHT: &[u8] = include_bytes!("fonts/Ubuntu-Light.ttf");
const GLYPHS: [char; 9] = ['o', 'g', '&', '8', 'B', '@', '%', '\u{e9}', '\u{e4}'];
const TOLERANCE_EM: f64 = 1.0 / 4000.0;
const FLATTEN: usize = 256;
const GRID: usize = 90;

enum Segment {
    Line(EmPoint, EmPoint),
    Quad(EmPoint, EmPoint, EmPoint),
    Cubic,
}

struct Adapter {
    upem: f64,
    builder: ContourBuilder,
    raw: Vec<Vec<Segment>>,
    start: EmPoint,
    pen: EmPoint,
}

impl Adapter {
    fn new(upem: f64, tolerance: f64) -> Self {
        Self {
            upem,
            builder: ContourBuilder::new(tolerance).unwrap(),
            raw: Vec::new(),
            start: Vec2::zero(),
            pen: Vec2::zero(),
        }
    }

    fn em(&self, x: f32, y: f32) -> EmPoint {
        Vec2::new(x as f64 / self.upem, y as f64 / self.upem)
    }

    fn push(&mut self, s: Segment) {
        self.raw.last_mut().expect("a segment before move_to").push(s);
    }
}

impl OutlineBuilder for Adapter {
    fn move_to(&mut self, x: f32, y: f32) {
        let p = self.em(x, y);
        self.builder.move_to(p);
        self.raw.push(Vec::new());
        self.start = p;
        self.pen = p;
    }

    fn line_to(&mut self, x: f32, y: f32) {
        let p = self.em(x, y);
        self.builder.line_to(p);
        self.push(Segment::Line(self.pen, p));
        self.pen = p;
    }

    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        let (c, p) = (self.em(x1, y1), self.em(x, y));
        self.builder.quad_to(c, p);
        self.push(Segment::Quad(self.pen, c, p));
        self.pen = p;
    }

    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        let (c1, c2, p) = (self.em(x1, y1), self.em(x2, y2), self.em(x, y));
        self.builder.cubic_to(c1, c2, p);
        self.push(Segment::Cubic);
        self.pen = p;
    }

    fn close(&mut self) {
        self.builder.close();
        if self.pen != self.start {
            self.push(Segment::Line(self.pen, self.start));
        }
        self.pen = self.start;
    }
}

struct Glyph {
    ch: char,
    contours: Vec<Contour>,
    source: Vec<Vec<EmPoint>>,
    min: EmPoint,
    max: EmPoint,
    quads: usize,
}

fn load(ch: char, tolerance: f64) -> Glyph {
    let face = Face::parse(UBUNTU_LIGHT, 0).unwrap();
    let id = face.glyph_index(ch).unwrap_or_else(|| panic!("the face has no {ch:?}"));
    let upem = face.units_per_em() as f64;
    let mut adapter = Adapter::new(upem, tolerance);
    let rect = face.outline_glyph(id, &mut adapter).unwrap_or_else(|| panic!("{ch:?} has no outline"));
    let mut quads = 0;
    let source = adapter
        .raw
        .iter()
        .map(|contour| {
            let mut line = Vec::new();
            for s in contour {
                match *s {
                    Segment::Line(a, _) => line.push(a),
                    Segment::Quad(a, c, b) => {
                        quads += 1;
                        line.extend((0..FLATTEN).map(|k| quad_point(a, c, b, k as f64 / FLATTEN as f64)));
                    }
                    Segment::Cubic => panic!("{ch:?}: a TrueType face emitted a cubic"),
                }
            }
            line
        })
        .collect();
    Glyph {
        ch,
        contours: adapter.builder.finish().unwrap(),
        source,
        min: Vec2::new(rect.x_min as f64 / upem, rect.y_min as f64 / upem),
        max: Vec2::new(rect.x_max as f64 / upem, rect.y_max as f64 / upem),
        quads,
    }
}

fn source_winding(source: &[Vec<EmPoint>], p: EmPoint) -> i32 {
    let mut w = 0;
    for line in source {
        for i in 0..line.len() {
            let (a, b) = (line[i], line[(i + 1) % line.len()]);
            let side = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
            if a.y <= p.y {
                if b.y > p.y && side > 0.0 {
                    w += 1;
                }
            } else if b.y <= p.y && side < 0.0 {
                w -= 1;
            }
        }
    }
    w
}

#[test]
fn a_vendored_truetype_face_converts_and_keeps_its_nonzero_region() {
    let mut overlapped = 0;
    for ch in GLYPHS {
        let g = load(ch, TOLERANCE_EM);
        let edges: usize = g.contours.iter().map(|c| c.edges.len()).sum();
        let arcs: usize = g.contours.iter().map(|c| c.arc_count()).sum();
        let (mut compared, mut inside, mut deep) = (0, 0, 0);
        for j in 0..GRID {
            for i in 0..GRID {
                let t = Vec2::new((i as f64 + 0.5) / GRID as f64, (j as f64 + 0.5) / GRID as f64);
                let p = Vec2::new(g.min.x + t.x * (g.max.x - g.min.x), g.min.y + t.y * (g.max.y - g.min.y));
                if g.contours.iter().map(|c| c.distance(p)).fold(f64::INFINITY, f64::min) < 2.0 * TOLERANCE_EM {
                    continue;
                }
                let expected = source_winding(&g.source, p);
                assert_eq!(nonzero_winding(&g.contours, p), expected, "{ch:?} at {p:?}");
                compared += 1;
                inside += (expected != 0) as usize;
                deep += (expected.abs() >= 2) as usize;
            }
        }
        overlapped += (deep > 0) as usize;
        println!(
            "{:?}: {} contours, {} quadratics -> {} edges ({} arcs); {} grid points compared, {} inside, {} at |winding| >= 2",
            g.ch, g.contours.len(), g.quads, edges, arcs, compared, inside, deep
        );
        assert!(inside > 200 && compared - inside > 200, "{ch:?}: the grid did not straddle the glyph");
        assert!(g.quads > 0 && arcs > 0, "{ch:?}: no curve reached the converter");
    }
    println!("{overlapped} of {} glyphs carry winding >= 2 somewhere", GLYPHS.len());
}

#[cfg(feature = "fit")]
use lights::fit::coverage::{coverage_budget, exact_coverage, DEFAULT_CELL_LIMIT};
#[cfg(feature = "fit")]
use lights::fit::target::Target;
#[cfg(feature = "fit")]
use lights::image::FloatImage;
#[cfg(feature = "fit")]
use lights::outline::Filled;
#[cfg(feature = "fit")]
use lights::primitive::{FeatureId, PrimitiveSet, Region};
#[cfg(feature = "fit")]
use lights::Rgb;

#[cfg(feature = "fit")]
const PX_PER_EM: f64 = 32.0;
#[cfg(feature = "fit")]
const PAD_PX: f64 = 2.0;
#[cfg(feature = "fit")]
const MAX_DEPTH: u32 = 6;
#[cfg(feature = "fit")]
const SUBROWS: usize = 64;
#[cfg(feature = "fit")]
const BOUND_BOUNDARY_PX: f64 = 4.0;

#[cfg(feature = "fit")]
fn to_px(g: &Glyph, p: EmPoint) -> EmPoint {
    Vec2::new((p.x - g.min.x) * PX_PER_EM + PAD_PX, (p.y - g.min.y) * PX_PER_EM + PAD_PX)
}

#[cfg(feature = "fit")]
fn viewport(g: &Glyph) -> (u32, u32) {
    (
        ((g.max.x - g.min.x) * PX_PER_EM + 2.0 * PAD_PX).ceil() as u32,
        ((g.max.y - g.min.y) * PX_PER_EM + 2.0 * PAD_PX).ceil() as u32,
    )
}

#[cfg(feature = "fit")]
fn scanline_reference(g: &Glyph) -> Vec<f64> {
    let (w, h) = viewport(g);
    let lines: Vec<Vec<EmPoint>> = g.source.iter().map(|l| l.iter().map(|&p| to_px(g, p)).collect()).collect();
    let mut out = vec![0.0; (w * h) as usize];
    for row in 0..h as usize * SUBROWS {
        let y = (row as f64 + 0.5) / SUBROWS as f64;
        let mut crossings: Vec<(f64, i32)> = Vec::new();
        for line in &lines {
            for i in 0..line.len() {
                let (a, b) = (line[i], line[(i + 1) % line.len()]);
                let dir = if a.y <= y && b.y > y {
                    1
                } else if a.y > y && b.y <= y {
                    -1
                } else {
                    continue;
                };
                crossings.push((a.x + (y - a.y) / (b.y - a.y) * (b.x - a.x), dir));
            }
        }
        crossings.sort_by(|p, q| p.0.total_cmp(&q.0));
        let py = row / SUBROWS;
        let mut winding = 0;
        for k in 0..crossings.len() {
            winding += crossings[k].1;
            if winding == 0 || k + 1 == crossings.len() {
                continue;
            }
            let (x0, x1) = (crossings[k].0, crossings[k + 1].0);
            for px in (x0.floor().max(0.0) as usize)..(x1.ceil().min(w as f64) as usize) {
                let overlap = (x1.min(px as f64 + 1.0) - x0.max(px as f64)).max(0.0);
                out[py * w as usize + px] += overlap / SUBROWS as f64;
            }
        }
    }
    out
}

#[cfg(feature = "fit")]
fn target(g: &Glyph) -> Target<f64> {
    let offset = Vec2::new(PAD_PX - g.min.x * PX_PER_EM, PAD_PX - g.min.y * PX_PER_EM);
    Target {
        name: "ubuntu-light-glyph",
        viewport: viewport(g),
        set: PrimitiveSet {
            ink: Rgb::new(1.0, 1.0, 1.0),
            parts: Vec::new(),
            regions: vec![Region {
                id: FeatureId(0),
                shape: Filled::new(g.contours.clone()).mapped(PX_PER_EM, offset),
            }],
            holes: Vec::new(),
        },
        paper: Rgb::new(0.0, 0.0, 0.0),
        probes: Vec::new(),
    }
}

#[cfg(feature = "fit")]
#[test]
fn exact_coverage_of_a_real_glyph_matches_a_scanline_integral_of_its_quadratics() {
    let bound = BOUND_BOUNDARY_PX * TOLERANCE_EM * PX_PER_EM;
    for ch in GLYPHS {
        let g = load(ch, TOLERANCE_EM);
        let t = target(&g);
        let mut budget = coverage_budget(DEFAULT_CELL_LIMIT);
        let (image, error): (FloatImage<f64>, _) = exact_coverage(&t, MAX_DEPTH, &mut budget).unwrap();
        let reference = scanline_reference(&g);
        let (mut worst, mut ink, mut ink_ref) = (0.0f64, 0.0, 0.0);
        for (k, r) in reference.iter().enumerate() {
            let c = image.pixels[k].x;
            worst = worst.max((c - r).abs());
            ink += c;
            ink_ref += r;
        }
        println!(
            "{ch:?}: {}x{} px, worst |exact - scanline| {:.5} (bound {:.5}), ink ratio {:.6}, undecided area {:.4} px^2, {} cells",
            t.viewport.0, t.viewport.1, worst, bound, ink / ink_ref, error.undecided_area, error.cells_visited
        );
        assert!(ink_ref > 20.0, "{ch:?}: the reference drew almost nothing");
        assert!(worst <= bound, "{ch:?}: worst pixel {worst} exceeds {bound}");
    }
}

#[cfg(feature = "fit")]
#[test]
fn a_region_targets_nearest_surface_steps_back_onto_its_boundary() {
    let g = load('o', TOLERANCE_EM);
    let t = target(&g);
    let (w, h) = t.viewport;
    let (mut checked, mut worst) = (0, 0.0f64);
    for j in 0..4 * h {
        for i in 0..4 * w {
            let p = Vec2::new((i as f64 + 0.5) / 4.0, (j as f64 + 0.5) / 4.0);
            let (signed, normal) = t.nearest_surface(p);
            if !(signed.abs() > 0.05 && signed.abs() < 1.0) {
                continue;
            }
            let foot = p - normal * signed;
            worst = worst.max(t.combined(foot).abs());
            checked += 1;
        }
    }
    println!("'o': {checked} points 0.05-1 px from the boundary, worst |field at the foot| {worst:.2e} px");
    assert!(checked > 500, "too few points near the boundary: {checked}");
    assert!(worst < 1e-3, "stepping back along the normal missed the boundary by {worst} px");
}
