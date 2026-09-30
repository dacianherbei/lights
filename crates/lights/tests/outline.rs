use lights::numerics::vector::Vec2;
use lights::outline::{
    Contour, ContourBuilder, EmPoint, Edge, Filled, cubic_point, cubic_to_edges, nonzero_signed_distance,
    nonzero_winding, quad_point, quad_to_edges,
};
use lights::primitive::{FeatureId, PrimitiveSet, Region};
use lights::Rgb;

const TOLERANCES: [f64; 3] = [1.0 / 100.0, 1.0 / 1000.0, 1.0 / 4000.0];
const BATTERY: usize = 48;
const POLYLINE: usize = 1024;
const EDGE_SAMPLES: usize = 16;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    fn point(&mut self) -> EmPoint {
        Vec2::new(self.next(), self.next())
    }
}

fn battery() -> Vec<[EmPoint; 3]> {
    let mut rng = Lcg(0x6c69_6768_7473);
    (0..BATTERY)
        .map(|_| [rng.point(), rng.point() * 1.4 - Vec2::splat(0.2), rng.point()])
        .collect()
}

fn polyline(q: &[EmPoint; 3]) -> Vec<EmPoint> {
    (0..=POLYLINE)
        .map(|k| quad_point(q[0], q[1], q[2], k as f64 / POLYLINE as f64))
        .collect()
}

fn to_polyline(line: &[EmPoint], p: EmPoint) -> f64 {
    line.windows(2)
        .map(|w| Edge::Line { a: w[0], b: w[1] }.distance(p))
        .fold(f64::INFINITY, f64::min)
}

fn two_sided(q: &[EmPoint; 3], edges: &[Edge]) -> f64 {
    let line = polyline(q);
    let forward = line
        .iter()
        .map(|&p| edges.iter().map(|e| e.distance(p)).fold(f64::INFINITY, f64::min))
        .fold(0.0, f64::max);
    let backward = edges
        .iter()
        .flat_map(|e| (0..=EDGE_SAMPLES).map(move |k| e.point_at(k as f64 / EDGE_SAMPLES as f64)))
        .map(|p| to_polyline(&line, p))
        .fold(0.0, f64::max);
    forward.max(backward)
}

#[test]
fn quadratic_edges_stay_within_the_em_tolerance_on_both_sides() {
    for tolerance in TOLERANCES {
        let mut worst_ratio: f64 = 0.0;
        let mut edges_total = 0;
        let mut arcs_total = 0;
        for q in battery() {
            let edges = quad_to_edges(q[0], q[1], q[2], tolerance);
            assert!(!edges.is_empty());
            assert!((edges[0].start() - q[0]).length() < 1e-12);
            assert!((edges[edges.len() - 1].end() - q[2]).length() < 1e-12);
            for pair in edges.windows(2) {
                assert!((pair[0].end() - pair[1].start()).length() < 1e-12);
            }
            worst_ratio = worst_ratio.max(two_sided(&q, &edges) / tolerance);
            edges_total += edges.len();
            arcs_total += edges.iter().filter(|e| matches!(e, Edge::Arc { .. })).count();
        }
        println!(
            "tolerance 1/{:.0} em: worst two-sided distance {:.4} x tolerance, {} edges ({} arcs) over {} quadratics",
            1.0 / tolerance,
            worst_ratio,
            edges_total,
            arcs_total,
            BATTERY
        );
        assert!(
            worst_ratio <= 1.0,
            "at 1/{:.0} em the converted edges stray {worst_ratio:.4} x the stated tolerance from the curve",
            1.0 / tolerance
        );
    }
}

fn cubic_battery() -> Vec<[EmPoint; 4]> {
    let mut rng = Lcg(0x6375_6269_6373);
    (0..BATTERY)
        .map(|_| {
            [
                rng.point(),
                rng.point() * 1.4 - Vec2::splat(0.2),
                rng.point() * 1.4 - Vec2::splat(0.2),
                rng.point(),
            ]
        })
        .collect()
}

fn cubic_two_sided(c: &[EmPoint; 4], edges: &[Edge]) -> f64 {
    let line: Vec<EmPoint> = (0..=POLYLINE)
        .map(|k| cubic_point(c[0], c[1], c[2], c[3], k as f64 / POLYLINE as f64))
        .collect();
    let forward = line
        .iter()
        .map(|&p| edges.iter().map(|e| e.distance(p)).fold(f64::INFINITY, f64::min))
        .fold(0.0, f64::max);
    let backward = edges
        .iter()
        .flat_map(|e| (0..=EDGE_SAMPLES).map(move |k| e.point_at(k as f64 / EDGE_SAMPLES as f64)))
        .map(|p| to_polyline(&line, p))
        .fold(0.0, f64::max);
    forward.max(backward)
}

#[test]
fn cubic_edges_stay_within_the_em_tolerance_on_both_sides() {
    for tolerance in TOLERANCES {
        let mut worst_ratio: f64 = 0.0;
        let mut edges_total = 0;
        let mut arcs_total = 0;
        for c in cubic_battery() {
            let edges = cubic_to_edges(c[0], c[1], c[2], c[3], tolerance);
            assert!((edges[0].start() - c[0]).length() < 1e-12);
            assert!((edges[edges.len() - 1].end() - c[3]).length() < 1e-12);
            for pair in edges.windows(2) {
                assert!((pair[0].end() - pair[1].start()).length() < 1e-12);
            }
            worst_ratio = worst_ratio.max(cubic_two_sided(&c, &edges) / tolerance);
            edges_total += edges.len();
            arcs_total += edges.iter().filter(|e| matches!(e, Edge::Arc { .. })).count();
        }
        println!(
            "cubic, tolerance 1/{:.0} em: worst two-sided distance {:.4} x tolerance, {} edges ({} arcs) over {} cubics",
            1.0 / tolerance,
            worst_ratio,
            edges_total,
            arcs_total,
            BATTERY
        );
        assert!(
            worst_ratio <= 1.0,
            "at 1/{:.0} em the converted cubic strays {worst_ratio:.4} x the stated tolerance",
            1.0 / tolerance
        );
    }
}

#[test]
fn a_cubic_through_the_builder_is_one_chained_contour() {
    let mut builder = ContourBuilder::new(1.0 / 1000.0).unwrap();
    builder.move_to(Vec2::new(0.1, 0.1));
    builder.cubic_to(Vec2::new(0.1, 0.9), Vec2::new(0.9, 0.9), Vec2::new(0.9, 0.1));
    let contours = builder.finish().unwrap();
    assert_eq!(contours.len(), 1);
    let edges = &contours[0].edges;
    for i in 0..edges.len() {
        assert!((edges[i].end() - edges[(i + 1) % edges.len()].start()).length() < 1e-12);
    }
    assert_eq!(nonzero_winding(&contours, Vec2::new(0.5, 0.5)), -1);
    assert_eq!(nonzero_winding(&contours, Vec2::new(0.5, 0.95)), 0);
    let mut refused = ContourBuilder::new(1.0 / 1000.0).unwrap();
    refused.cubic_to(Vec2::new(0.0, 1.0), Vec2::new(1.0, 1.0), Vec2::new(1.0, 0.0));
    assert!(refused.finish().is_err());
}

#[test]
fn edge_count_grows_as_the_cube_root_of_the_tolerance_not_the_square_root() {
    let count = |tolerance: f64| -> usize {
        battery()
            .iter()
            .map(|q| quad_to_edges(q[0], q[1], q[2], tolerance).len())
            .sum()
    };
    let coarse = count(TOLERANCES[0]);
    let fine = count(TOLERANCES[2]);
    let exponent = (fine as f64 / coarse as f64).ln() / (TOLERANCES[0] / TOLERANCES[2]).ln();
    println!(
        "edges {coarse} at 1/100 em, {fine} at 1/4000 em: {:.2}x, exponent {exponent:.3}",
        fine as f64 / coarse as f64
    );
    assert!(
        (0.25..0.40).contains(&exponent),
        "three-point arcs should scale as tolerance^-1/3; measured exponent {exponent:.3}"
    );
}

fn circle_pair(centre: EmPoint, radius: f64, counter_clockwise: bool) -> Contour {
    let pi = core::f64::consts::PI;
    let sign = if counter_clockwise { 1.0 } else { -1.0 };
    Contour {
        edges: vec![
            Edge::Arc {
                centre,
                radius,
                from: 0.0,
                to: sign * pi,
            },
            Edge::Arc {
                centre,
                radius,
                from: sign * pi,
                to: sign * 2.0 * pi,
            },
        ],
    }
}

#[test]
fn a_disc_of_two_half_arcs_winds_once_although_its_chords_enclose_nothing() {
    let centre = Vec2::new(0.5, 0.5);
    let ccw = circle_pair(centre, 0.25, true);
    let cw = circle_pair(centre, 0.25, false);
    for p in [centre, Vec2::new(0.5, 0.7), Vec2::new(0.5, 0.3), Vec2::new(0.3, 0.52)] {
        assert_eq!(ccw.winding(p), 1, "counter-clockwise disc at {p:?}");
        assert_eq!(cw.winding(p), -1, "clockwise disc at {p:?}");
    }
    for p in [Vec2::new(0.0, 0.5), Vec2::new(0.5, 0.8), Vec2::new(0.9, 0.9)] {
        assert_eq!(ccw.winding(p), 0, "outside at {p:?}");
    }
    assert!(nonzero_signed_distance(&[ccw.clone()], centre) < -0.24);
    assert!((nonzero_signed_distance(&[ccw], Vec2::new(0.5, 0.0)) - 0.25).abs() < 1e-12);
}

type Source = Vec<Vec<[EmPoint; 3]>>;

fn blob(cx: f64, cy: f64, r: f64, counter_clockwise: bool) -> Vec<[EmPoint; 3]> {
    let k = 1.3;
    let pts: Vec<EmPoint> = (0..4)
        .map(|i| {
            let a = i as f64 * core::f64::consts::FRAC_PI_2;
            Vec2::new(cx + r * a.cos(), cy + r * a.sin())
        })
        .collect();
    let ctrl: Vec<EmPoint> = (0..4)
        .map(|i| {
            let a = (i as f64 + 0.5) * core::f64::consts::FRAC_PI_2;
            Vec2::new(cx + k * r * a.cos(), cy + k * r * a.sin())
        })
        .collect();
    let mut out: Vec<[EmPoint; 3]> = (0..4).map(|i| [pts[i], ctrl[i], pts[(i + 1) % 4]]).collect();
    if !counter_clockwise {
        out = out.into_iter().rev().map(|q| [q[2], q[1], q[0]]).collect();
    }
    out
}

fn named_sources() -> Vec<(&'static str, Source)> {
    vec![
        ("ring: outer +1, counter -1", vec![blob(0.5, 0.5, 0.4, true), blob(0.5, 0.5, 0.18, false)]),
        ("two overlapping +1 contours", vec![blob(0.4, 0.5, 0.25, true), blob(0.6, 0.5, 0.25, true)]),
        (
            "winding +1 +1 -1: a hole that only partly cancels",
            vec![blob(0.4, 0.5, 0.3, true), blob(0.6, 0.5, 0.3, true), blob(0.5, 0.5, 0.12, false)],
        ),
    ]
}

fn source_winding(source: &Source, p: EmPoint) -> i32 {
    let mut w = 0;
    for contour in source {
        let line: Vec<EmPoint> = contour
            .iter()
            .flat_map(|q| (0..POLYLINE).map(move |k| quad_point(q[0], q[1], q[2], k as f64 / POLYLINE as f64)))
            .collect();
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

fn convert(source: &Source, tolerance: f64) -> Vec<Contour> {
    let mut builder = ContourBuilder::new(tolerance).unwrap();
    for contour in source {
        builder.move_to(contour[0][0]);
        for q in contour {
            builder.quad_to(q[1], q[2]);
        }
        builder.close();
    }
    builder.finish().unwrap()
}

#[test]
fn converted_contours_keep_the_source_nonzero_region_away_from_the_boundary() {
    let tolerance = 1.0 / 1000.0;
    let grid = 61;
    for (name, source) in named_sources() {
        let contours = convert(&source, tolerance);
        let mut compared = 0;
        let mut inside = 0;
        for j in 0..grid {
            for i in 0..grid {
                let p = Vec2::new((i as f64 + 0.5) / grid as f64, (j as f64 + 0.5) / grid as f64);
                if contours.iter().map(|c| c.distance(p)).fold(f64::INFINITY, f64::min) < 2.0 * tolerance {
                    continue;
                }
                let expected = source_winding(&source, p);
                assert_eq!(nonzero_winding(&contours, p), expected, "{name} at {p:?}");
                compared += 1;
                inside += (expected != 0) as usize;
            }
        }
        println!("{name}: {compared} grid points compared, {inside} inside");
        assert!(inside > 100 && compared - inside > 100, "{name}: the grid did not straddle the region");
    }
}

#[test]
fn the_plus_one_plus_one_minus_one_case_is_filled_where_a_parts_minus_holes_reading_would_carve() {
    let (_, source) = named_sources().remove(2);
    let contours = convert(&source, 1.0 / 1000.0);
    let centre = Vec2::new(0.5, 0.5);
    assert_eq!(nonzero_winding(&contours, centre), 1);
    assert!(nonzero_signed_distance(&contours, centre) < 0.0);
}

#[test]
fn the_builder_refuses_what_it_cannot_honour() {
    assert!(ContourBuilder::new(0.0).is_err());
    assert!(ContourBuilder::new(f64::NAN).is_err());
    let mut early = ContourBuilder::new(0.001).unwrap();
    early.line_to(Vec2::new(1.0, 0.0));
    assert!(early.finish().is_err());
    let mut nan = ContourBuilder::new(0.001).unwrap();
    nan.move_to(Vec2::new(0.0, 0.0));
    nan.quad_to(Vec2::new(f64::NAN, 0.0), Vec2::new(1.0, 0.0));
    assert!(nan.finish().is_err());
    let mut closes = ContourBuilder::new(0.001).unwrap();
    closes.move_to(Vec2::new(0.0, 0.0));
    closes.line_to(Vec2::new(1.0, 0.0));
    closes.line_to(Vec2::new(1.0, 1.0));
    let contours = closes.finish().unwrap();
    assert_eq!(contours.len(), 1);
    assert_eq!(contours[0].edges.len(), 3);
    assert_eq!(contours[0].edges[2].end(), Vec2::new(0.0, 0.0));
}

const BOUNDARY_SAMPLES: usize = 4000;

fn pentagram() -> Contour {
    let at = |k: usize| {
        let a = core::f64::consts::FRAC_PI_2 + k as f64 * 4.0 * core::f64::consts::PI / 5.0;
        Vec2::new(0.5 + 0.42 * a.cos(), 0.5 + 0.42 * a.sin())
    };
    Contour {
        edges: (0..5).map(|k| Edge::Line { a: at(k), b: at(k + 1) }).collect(),
    }
}

fn region_inputs() -> Vec<(&'static str, Vec<Contour>)> {
    let mut out: Vec<(&'static str, Vec<Contour>)> = named_sources()
        .into_iter()
        .map(|(name, source)| (name, convert(&source, 1.0 / 1000.0)))
        .collect();
    out.push(("self-intersecting pentagram, centre winds 2", vec![pentagram()]));
    let bullseye = vec![blob(0.5, 0.5, 0.42, true), blob(0.5, 0.5, 0.28, false), blob(0.5, 0.5, 0.12, true)];
    out.push(("bullseye: ring and a dot inside its counter", convert(&bullseye, 1.0 / 1000.0)));
    out
}

fn sampled_boundary(contours: &[Contour]) -> (Vec<EmPoint>, f64) {
    let mut kept = Vec::new();
    let mut spacing: f64 = 0.0;
    for edge in contours.iter().flat_map(|c| c.edges.iter()) {
        let step = (edge.end() - edge.start()).length().max(match *edge {
            Edge::Arc { radius, from, to, .. } => radius * (to - from).abs(),
            Edge::Line { .. } => 0.0,
        }) / BOUNDARY_SAMPLES as f64;
        spacing = spacing.max(step);
        for k in 0..=BOUNDARY_SAMPLES {
            let s = k as f64 / BOUNDARY_SAMPLES as f64;
            let q = edge.point_at(s);
            let n = edge.normal_at(s);
            let left = nonzero_winding(contours, q + n * 1e-6) != 0;
            let right = nonzero_winding(contours, q - n * 1e-6) != 0;
            if left != right {
                kept.push(q);
            }
        }
    }
    (kept, spacing)
}

#[test]
fn the_nonzero_field_measures_distance_to_the_region_boundary_and_not_to_edges_inside_the_ink() {
    let grid = 41;
    for (name, contours) in region_inputs() {
        let filled = Filled::new(contours.clone());
        let (kept, spacing) = sampled_boundary(&contours);
        assert!(kept.len() > BOUNDARY_SAMPLES, "{name}: the reference kept no boundary");
        let mut worst: f64 = 0.0;
        let mut inside = 0;
        for j in 0..grid {
            for i in 0..grid {
                let p = Vec2::new((i as f64 + 0.5) / grid as f64, (j as f64 + 0.5) / grid as f64);
                let reference = kept.iter().fold(f64::INFINITY, |d, q| d.min((p - *q).length()));
                let filled_in = nonzero_winding(&contours, p) != 0;
                inside += filled_in as usize;
                for (field, value) in [
                    ("Filled::signed_distance", filled.signed_distance(p)),
                    ("nonzero_signed_distance", nonzero_signed_distance(&contours, p)),
                ] {
                    assert_eq!(value < 0.0, filled_in, "{name}: {field} sign at {p:?}");
                    let error = (value.abs() - reference).abs();
                    worst = worst.max(error);
                    assert!(
                        error <= 2.0 * spacing,
                        "{name}: {field} is {} at {p:?} where the region boundary is {reference} away",
                        value.abs()
                    );
                }
            }
        }
        println!(
            "{name}: {} boundary pieces of {} edges, {inside} of {} grid points inside, worst |d - reference| {worst:.2e} against spacing {spacing:.2e}",
            filled.boundary.len(),
            contours.iter().map(|c| c.edges.len()).sum::<usize>(),
            grid * grid
        );
        assert!(inside > 50, "{name}: the grid did not reach the ink");
    }
}

#[test]
fn a_region_is_a_solid_part_of_a_primitive_set_under_the_nonzero_rule() {
    for (name, contours) in region_inputs() {
        let filled = Filled::new(contours.clone());
        let set = PrimitiveSet::<f64> {
            ink: Rgb::new(1.0, 1.0, 1.0),
            parts: Vec::new(),
            regions: vec![Region { id: FeatureId(0), shape: filled.clone() }],
            holes: Vec::new(),
        };
        for j in 0..29 {
            for i in 0..29 {
                let p = Vec2::new((i as f64 + 0.5) / 29.0, (j as f64 + 0.5) / 29.0);
                assert_eq!(set.combined(p), filled.signed_distance(p), "{name} at {p:?}");
            }
        }
        let scaled = Region { id: FeatureId(0), shape: filled.clone() }.mapped(64.0, Vec2::new(3.0, -5.0));
        let q = Vec2::new(0.37, 0.61);
        let moved = scaled.sdf(q * 64.0 + Vec2::new(3.0, -5.0));
        assert!((moved - 64.0 * filled.signed_distance(q)).abs() < 1e-9, "{name}: mapped region");
    }
}

#[test]
fn reading_every_clockwise_contour_as_a_hole_carves_what_the_nonzero_rule_fills() {
    let inputs = region_inputs();
    let centre = Vec2::new(0.5, 0.5);
    let mut carved = Vec::new();
    for (name, contours) in inputs.iter() {
        let clockwise = |c: &Contour| c.winding(centre) < 0;
        let parts: Vec<Filled> = contours.iter().filter(|c| !clockwise(c)).map(|c| Filled::new(vec![c.clone()])).collect();
        let holes: Vec<Filled> = contours.iter().filter(|c| clockwise(c)).map(|c| Filled::new(vec![c.clone()])).collect();
        let naive = parts.iter().any(|f| f.winding(centre) != 0) && !holes.iter().any(|f| f.winding(centre) != 0);
        let nonzero = Filled::new(contours.clone()).winding(centre) != 0;
        if naive != nonzero {
            carved.push(*name);
        }
    }
    println!("the clockwise-is-hole reading disagrees with non-zero at the centre of: {carved:?}");
    assert_eq!(
        carved,
        vec!["winding +1 +1 -1: a hole that only partly cancels", "bullseye: ring and a dot inside its counter"]
    );
}
