#![cfg(feature = "fit")]
use lights::error::LightsError;
use lights::fit::analytic::{AnalyticArm, render_analytic, sdf_coverage};
use lights::fit::coverage::{coverage_budget, exact_coverage};
use lights::fit::target::{
    FeatureId, Part, Primitive, STROKE_HALF_WIDTH, Side, Target, glyph_pivot, glyph_point,
};
use lights::image::FloatImage;
use lights::numerics::vector::Vec2;

const CELL_LIMIT: u64 = 64_000_000;

struct Deterministic {
    state: u64,
}

impl Deterministic {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.state >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

fn clip_area_below(square: (f64, f64), point: Vec2<f64>, normal: Vec2<f64>) -> f64 {
    let (x0, y0) = square;
    let mut polygon = vec![
        Vec2::new(x0, y0),
        Vec2::new(x0 + 1.0, y0),
        Vec2::new(x0 + 1.0, y0 + 1.0),
        Vec2::new(x0, y0 + 1.0),
    ];
    let value = |p: Vec2<f64>| (p - point).dot(normal);
    let mut clipped: Vec<Vec2<f64>> = Vec::new();
    for i in 0..polygon.len() {
        let current = polygon[i];
        let next = polygon[(i + 1) % polygon.len()];
        let a = value(current);
        let b = value(next);
        if a <= 0.0 {
            clipped.push(current);
        }
        if (a <= 0.0) != (b <= 0.0) {
            let t = a / (a - b);
            clipped.push(current + (next - current) * t);
        }
    }
    polygon = clipped;
    if polygon.len() < 3 {
        return 0.0;
    }
    let mut twice = 0.0;
    for i in 0..polygon.len() {
        let current = polygon[i];
        let next = polygon[(i + 1) % polygon.len()];
        twice += current.x * next.y - next.x * current.y;
    }
    (twice * 0.5).abs()
}

fn half_plane_target(angle: f64, size: u32) -> Target<f64> {
    let centre = Vec2::new((size as f64) * 0.5, (size as f64) * 0.5);
    let normal = Vec2::new(angle.cos(), angle.sin());
    Target::ink_on_paper(
        "probe-half-plane",
        (size, size),
        vec![Part {
            id: FeatureId(0),
            shape: Primitive::half_plane(centre, normal),
        }],
        Vec::new(),
        Vec::new(),
    )
}

fn coverage_of(target: &Target<f64>, depth: u32) -> FloatImage<f64> {
    let mut budget = coverage_budget(CELL_LIMIT);
    match exact_coverage(target, depth, &mut budget) {
        Ok((image, _)) => image,
        Err(e) => panic!("the quadrature must not run out of budget here: {e:?}"),
    }
}

fn total_coverage(image: &FloatImage<f64>) -> f64 {
    image.pixels.iter().map(|p| p.x).sum()
}

fn boundary_samples(target: &Target<f64>, spacing: f64) -> Vec<Vec2<f64>> {
    fn ring(
        out: &mut Vec<Vec2<f64>>,
        spacing: f64,
        centre: Vec2<f64>,
        radius: f64,
        from: f64,
        to: f64,
    ) {
        if radius <= 0.0 {
            return;
        }
        let steps = (((to - from).abs() * radius / spacing).ceil() as usize).max(2);
        for i in 0..=steps {
            let a = from + (to - from) * (i as f64) / (steps as f64);
            out.push(centre + Vec2::new(a.cos(), a.sin()) * radius);
        }
    }
    let mut candidates: Vec<Vec2<f64>> = Vec::new();
    let tau = std::f64::consts::TAU;
    for part in target.set.parts.iter().chain(target.set.holes.iter()) {
        match part.shape {
            Primitive::HalfPlane { point, normal } => {
                let along = Vec2::new(-normal.y, normal.x);
                let steps = ((128.0 / spacing).ceil() as usize).max(2);
                for i in 0..=steps {
                    let t = -64.0 + 128.0 * (i as f64) / (steps as f64);
                    candidates.push(point + along * t);
                }
            }
            Primitive::Segment { a, b, half_width } => {
                let ab = b - a;
                let length = ab.length();
                let normal = match Vec2::new(-ab.y, ab.x).normalize() {
                    Some(n) => n,
                    None => Vec2::new(1.0, 0.0),
                };
                let steps = ((length / spacing).ceil() as usize).max(2);
                for i in 0..=steps {
                    let t = (i as f64) / (steps as f64);
                    let base = a + ab * t;
                    candidates.push(base + normal * half_width);
                    candidates.push(base - normal * half_width);
                }
                ring(&mut candidates, spacing, a, half_width, 0.0, tau);
                ring(&mut candidates, spacing, b, half_width, 0.0, tau);
            }
            Primitive::Arc {
                centre,
                radius,
                from,
                to,
                half_width,
            } => {
                ring(&mut candidates, spacing, centre, radius + half_width, from, to);
                ring(&mut candidates, spacing, centre, radius - half_width, from, to);
                let lo = centre + Vec2::new(from.cos(), from.sin()) * radius;
                let hi = centre + Vec2::new(to.cos(), to.sin()) * radius;
                ring(&mut candidates, spacing, lo, half_width, 0.0, tau);
                ring(&mut candidates, spacing, hi, half_width, 0.0, tau);
            }
            Primitive::Disc { centre, radius } => {
                ring(&mut candidates, spacing, centre, radius, 0.0, tau)
            }
        }
    }
    candidates
        .into_iter()
        .filter(|p| target.combined(*p).abs() < 1.0e-9)
        .collect()
}

fn brute_force_distance(samples: &[Vec2<f64>], p: Vec2<f64>) -> f64 {
    let mut best = f64::INFINITY;
    for s in samples.iter() {
        let d = (p - *s).length_squared();
        if d < best {
            best = d;
        }
    }
    best.sqrt()
}

fn independently_inside(target: &Target<f64>, p: Vec2<f64>) -> bool {
    let covered = |part: &Part<f64>| match part.shape {
        Primitive::HalfPlane { point, normal } => (p - point).dot(normal) <= 0.0,
        Primitive::Segment { a, b, half_width } => {
            let ab = b - a;
            let denominator = ab.length_squared();
            let t = if denominator > 0.0 {
                ((p - a).dot(ab) / denominator).clamp(0.0, 1.0)
            } else {
                0.0
            };
            (p - (a + ab * t)).length() <= half_width
        }
        Primitive::Arc {
            centre,
            radius,
            from,
            to,
            half_width,
        } => {
            let v = p - centre;
            let r = v.length();
            let mut angle = v.y.atan2(v.x);
            let mid = (from + to) * 0.5;
            let tau = std::f64::consts::TAU;
            let half_sweep = (to - from) * 0.5;
            let mut delta = angle - mid;
            let mut guard = 0;
            while delta > std::f64::consts::PI && guard < 8 {
                delta -= tau;
                guard += 1;
            }
            while delta < -std::f64::consts::PI && guard < 16 {
                delta += tau;
                guard += 1;
            }
            angle = mid + delta;
            let _ = angle;
            if delta.abs() <= half_sweep {
                (r - radius).abs() <= half_width
            } else {
                let lo = centre + Vec2::new(from.cos(), from.sin()) * radius;
                let hi = centre + Vec2::new(to.cos(), to.sin()) * radius;
                (p - lo).length().min((p - hi).length()) <= half_width
            }
        }
        Primitive::Disc { centre, radius } => (p - centre).length() <= radius,
    };
    let solid = target.set.parts.iter().any(covered);
    let carved = target.set.holes.iter().any(covered);
    solid && !carved
}

#[test]
fn exact_coverage_matches_the_closed_form_half_plane_area() {
    let size = 8u32;
    let mut worst = 0.0f64;
    let mut worst_angle = 0.0f64;
    for step in 0..64 {
        let angle = std::f64::consts::TAU * (step as f64) / 64.0 + 0.013;
        let target = half_plane_target(angle, size);
        let point = Vec2::new((size as f64) * 0.5, (size as f64) * 0.5);
        let normal = Vec2::new(angle.cos(), angle.sin());
        let image = coverage_of(&target, 4);
        for py in 0..size {
            for px in 0..size {
                let closed = clip_area_below((px as f64, py as f64), point, normal);
                let error = (image.get(px, py).x - closed).abs();
                if error > worst {
                    worst = error;
                    worst_angle = angle;
                }
            }
        }
    }
    assert!(
        worst < 1.0e-9,
        "the exact-coverage quadrature disagrees with an independent polygon \
         clip of the same half-plane by {worst} at angle {worst_angle} rad. \
         Every number downstream — the ground truth the fitter descends, the \
         control arm it is compared against, and every transition width \
         measured on either — is this quantity, so a wrong coverage would be \
         wrong in both arms at once and the difference between them would \
         still look clean"
    );
}

#[test]
fn exact_coverage_matches_the_closed_form_disc_area() {
    let target = Target::<f64>::t4_curve();
    let radius = 3.0f64;
    let image = coverage_of(&target, 12);
    let measured = total_coverage(&image);
    let closed = std::f64::consts::PI * radius * radius;
    let error = (measured - closed).abs();
    assert!(
        error < 1.0e-7,
        "the summed coverage of a disc of radius {radius} is {measured} but \
         pi r^2 is {closed}, an error of {error}. A curved boundary is closed \
         by the tangent line at the depth floor, so the residual is second \
         order in the cell size and one-signed: it does not cancel, and it is \
         the reason this tolerance is 1e-7 where the half-plane's is 1e-9"
    );
}

#[test]
fn clearance_is_a_certified_lower_bound() {
    let target = Target::<f64>::t6_glyph_like();
    let samples = boundary_samples(&target, 0.02);
    assert!(
        samples.len() > 2000,
        "the brute-force boundary carries only {} points, which is not dense \
         enough for the comparison below to mean anything",
        samples.len()
    );

    let mut rng = Deterministic::new(0x5eed_1234_abcd_0001);
    let mut worst_violation = 0.0f64;
    let mut tight = 0usize;
    let total = 10000usize;
    for _ in 0..total {
        let p = Vec2::new(rng.next() * 32.0, rng.next() * 32.0);
        let bound = target.clearance(p);
        let truth = brute_force_distance(&samples, p);
        let violation = bound - truth;
        if violation > worst_violation {
            worst_violation = violation;
        }
        if truth - bound < 0.05 {
            tight += 1;
        }
    }
    assert!(
        worst_violation <= 1.0e-9,
        "clearance exceeded the true distance to the combined boundary by \
         {worst_violation}. The subdivision decides a cell whenever \
         clearance(centre) > half-diagonal, so this inequality is the entire \
         soundness argument: an upper bound would decide cells that straddle \
         the boundary, and every ground-truth image would be quietly wrong"
    );
    assert!(
        tight * 2 > total,
        "only {tight} of {total} sampled points had the bound within 0.05 of \
         the true distance. The bound is allowed to be loose at an interior \
         seam where two parts overlap, but if it were loose almost everywhere \
         the inequality above would hold vacuously"
    );
}

#[test]
fn the_undecided_area_falls_with_the_budget_and_is_reported() {
    let target = Target::<f64>::t4_curve();
    let mut previous_area = f64::INFINITY;
    let mut previous_cells = 0u64;
    for depth in [4u32, 6, 8] {
        let mut budget = coverage_budget(CELL_LIMIT);
        let (_, residual) = match exact_coverage(&target, depth, &mut budget) {
            Ok(v) => v,
            Err(e) => panic!("depth {depth} exhausted the budget: {e:?}"),
        };
        assert!(
            residual.undecided_area > 0.0,
            "depth {depth} reported no undecided area at all on a curved \
             boundary, which cannot be right and would mean the residual is \
             assumed rather than measured"
        );
        assert!(
            residual.undecided_area < previous_area,
            "the undecided area did not fall from {previous_area} at the \
             previous depth to {} at depth {depth}",
            residual.undecided_area
        );
        assert!(
            residual.cells_visited > previous_cells,
            "depth {depth} visited {} cells, no more than the {previous_cells} \
             of the previous depth",
            residual.cells_visited
        );
        assert!(
            residual.max_depth_reached == depth,
            "depth {depth} reported a deepest cell at level {}",
            residual.max_depth_reached
        );
        previous_area = residual.undecided_area;
        previous_cells = residual.cells_visited;
    }
}

#[test]
fn the_quadrature_fails_hard_rather_than_truncating_the_image() {
    let target = Target::<f64>::t4_curve();
    let mut budget = coverage_budget(64);
    match exact_coverage(&target, 12, &mut budget) {
        Err(LightsError::BudgetExhausted { .. }) => (),
        other => panic!(
            "a quadrature that runs out of cells must fail through the shared \
             budget rather than return a silently truncated image, got \
             {:?}",
            other.map(|(image, residual)| (image.width, image.height, residual))
        ),
    }
}

#[test]
fn the_sign_agrees_with_a_dense_point_in_shape_test_including_the_counter() {
    let target = Target::<f64>::t6_glyph_like();
    let step = 0.05f64;
    let steps = (32.0 / step) as u32;
    let mut compared = 0usize;
    let mut ink_seen = 0usize;
    for j in 0..steps {
        for i in 0..steps {
            let p = Vec2::new((i as f64) * step, (j as f64) * step);
            if target.combined(p).abs() < 1.0e-12 {
                continue;
            }
            compared += 1;
            let expected = independently_inside(&target, p);
            let actual = target.classify(p) == Side::Inside;
            if expected {
                ink_seen += 1;
            }
            assert!(
                expected == actual,
                "at {p:?} the union-minus-holes classifier says {actual} while \
                 an independent per-primitive membership test says {expected}"
            );
        }
    }
    assert!(
        compared > 100_000 && ink_seen > 1000,
        "the dense sign scan compared {compared} points of which {ink_seen} \
         were ink, which is not enough of either to have tested anything"
    );

    let counter_centre = glyph_pivot::<f64>();
    let mut carved = 0usize;
    for k in 0..400 {
        let t = (k as f64) / 400.0;
        let angle = std::f64::consts::TAU * t;
        for radius in [0.1f64, 0.4, 0.8, 1.2, 1.5] {
            let p = counter_centre + Vec2::new(angle.cos(), angle.sin()) * radius;
            assert!(
                target.classify(p) == Side::Outside,
                "the counter at {p:?} reads as ink. A hole that removes no ink \
                 is a hole whose code path is never exercised, which is why the \
                 bowl's closing stem runs through the counter centre"
            );
            let stem = Primitive::Segment {
                a: glyph_point::<f64>(16.0, 16.0),
                b: glyph_point::<f64>(16.0, 22.0),
                half_width: STROKE_HALF_WIDTH,
            };
            if stem.sdf(p) <= 0.0 {
                carved += 1;
            }
        }
    }
    assert!(
        carved > 100,
        "only {carved} probe points inside the counter were also inside the \
         closing stem, so the hole is removing no ink and this test would pass \
         with the hole deleted"
    );
}

#[test]
fn the_two_control_arms_differ_so_the_cheap_one_is_not_the_ceiling() {
    for target in Target::<f64>::all() {
        let mut budget = coverage_budget(CELL_LIMIT);
        let exact = match render_analytic(&target, AnalyticArm::Exact, 8, &mut budget) {
            Ok((image, cost)) => (image, cost),
            Err(e) => panic!("the exact arm failed on {}: {e:?}", target.name),
        };
        let cheap = match render_analytic(&target, AnalyticArm::Sdf, 8, &mut budget) {
            Ok((image, cost)) => (image, cost),
            Err(e) => panic!("the cheap arm failed on {}: {e:?}", target.name),
        };

        assert!(
            exact.1.primitives == cheap.1.primitives
                && exact.1.description_bytes == cheap.1.description_bytes,
            "the two arms must describe the same target from the same \
             definition, but {} reports {:?} against {:?}",
            target.name,
            exact.1,
            cheap.1
        );
        let difference = match exact.0.difference(&cheap.0) {
            Ok(d) => d,
            Err(e) => panic!("{e:?}"),
        };
        let (peak, x, y) = difference.max_abs();
        if target.name == "t1-flat-fill" {
            assert!(
                peak < 1.0e-12,
                "the two arms disagree by {peak} on a target with no boundary \
                 in the frame at all, at pixel ({x},{y})"
            );
            assert!(
                exact.1.sdf_evaluations == cheap.1.sdf_evaluations,
                "with no boundary in the frame every cell decides at depth \
                 zero, so the exact arm must cost exactly one cell per pixel \
                 like the cheap arm, got {} against {}",
                exact.1.sdf_evaluations,
                cheap.1.sdf_evaluations
            );
            continue;
        }
        assert!(
            exact.1.sdf_evaluations > cheap.1.sdf_evaluations,
            "on {} the exact arm cost {} evaluations and the cheap arm {}: if \
             the quality ceiling were not the expensive one there would be no \
             cost question to ask",
            target.name,
            exact.1.sdf_evaluations,
            cheap.1.sdf_evaluations
        );
        assert!(
            peak > 0.02,
            "the two control arms differ by only {peak} on {}: a cost \
             comparison against an arm that is indistinguishable from the \
             quality ceiling is a comparison against a straw man, and the \
             report would be claiming the cheap arm costs nothing in quality",
            target.name
        );
    }
}

#[test]
fn the_six_targets_are_six_separate_content_classes() {
    let targets = Target::<f64>::all();
    assert!(targets.len() == 6, "P2 measures six content classes");
    let expected = [
        ("t1-flat-fill", (64u32, 64u32)),
        ("t2-straight-edge", (64, 64)),
        ("t3-right-angle-corner", (64, 64)),
        ("t4-curve", (32, 32)),
        ("t5-thin-stroke", (64, 64)),
        ("t6-glyph-like", (32, 32)),
    ];
    for (target, (name, viewport)) in targets.iter().zip(expected.iter()) {
        assert!(
            target.name == *name && target.viewport == *viewport,
            "expected {name} at {viewport:?}, found {} at {:?}",
            target.name,
            target.viewport
        );
        assert!(
            !target.probes.is_empty(),
            "{} declares no probe, so nothing about it can be measured",
            target.name
        );
        let ids = target.feature_ids();
        for probe in target.probes.iter() {
            assert!(
                ids.contains(&probe.feature),
                "{} anchors a probe on feature {:?}, which no part or hole \
                 declares",
                target.name,
                probe.feature
            );
        }
        for probe in target.probes.iter() {
            let nearest = target.nearest_feature(probe.anchor);
            assert!(
                nearest == probe.feature,
                "{}: the probe anchored at {:?} for feature {:?} is nearest to \
                 feature {:?} instead, so every sample it collects would be \
                 rejected by the nearest-feature filter",
                target.name,
                probe.anchor,
                probe.feature,
                nearest
            );
        }
    }
}

#[test]
fn an_interior_seam_does_not_read_as_half_covered() {
    let target = Target::ink_on_paper(
        "probe-crossed-strokes",
        (32, 24),
        vec![
            Part {
                id: FeatureId(0),
                shape: Primitive::Segment {
                    a: Vec2::new(4.0, 10.0),
                    b: Vec2::new(28.0, 10.0),
                    half_width: 3.5,
                },
            },
            Part {
                id: FeatureId(1),
                shape: Primitive::Segment {
                    a: Vec2::new(16.0, 2.0),
                    b: Vec2::new(16.0, 20.0),
                    half_width: 3.5,
                },
            },
        ],
        Vec::new(),
        Vec::new(),
    );

    let seam = Vec2::new(16.5, 6.5);
    assert!(
        target.clearance(seam) < 1.0e-12,
        "the probe point {seam:?} was chosen because it lies on one stroke's \
         surface inside the other, where the clearance bound collapses to zero \
         although the combined boundary is 3px away; it reports {} instead",
        target.clearance(seam)
    );
    assert!(
        target.classify(seam) == Side::Inside,
        "the seam point must be interior ink"
    );

    let image = coverage_of(&target, 8);
    for (px, py) in [(16u32, 6u32), (15, 6), (16, 13), (12, 10), (19, 10)] {
        let value = image.get(px, py).x;
        assert!(
            (value - 1.0).abs() < 1.0e-12,
            "pixel ({px},{py}) is wholly inside the union of two overlapping \
             strokes but reports coverage {value}. Its cells never satisfy the \
             decided test, because min-of-|sdf| is zero along the interior \
             seam, so they reach the depth floor and are handed to the tangent \
             closure — which would report about half coverage for a pixel that \
             is entirely ink. The closure therefore checks whether the class \
             actually changes across that surface before believing it is a \
             boundary"
        );
    }

    let mut budget = coverage_budget(CELL_LIMIT);
    let (_, residual) = match exact_coverage(&target, 8, &mut budget) {
        Ok(v) => v,
        Err(e) => panic!("{e:?}"),
    };
    assert!(
        residual.undecided_area > 0.0,
        "the seam must show up in the reported residual rather than vanish"
    );
}

#[test]
fn the_cheap_arm_paints_nothing_a_pixel_away_from_the_shape() {
    let spacing = 0.05f64;
    for target in [
        Target::<f64>::t3_right_angle_corner(),
        Target::<f64>::t6_glyph_like(),
    ] {
        let samples = boundary_samples(&target, spacing);
        assert!(
            samples.len() > 1000,
            "the brute-force boundary of {} carries only {} points, so the \
             distances below mean nothing",
            target.name,
            samples.len()
        );
        let image = sdf_coverage(&target);
        let (width, height) = target.viewport;
        let mut worst = 0.0f64;
        let mut worst_at = (0u32, 0u32, 0.0f64);
        let mut checked = 0usize;
        for py in 0..height {
            for px in 0..width {
                let centre = Vec2::new(px as f64 + 0.5, py as f64 + 0.5);
                if target.classify(centre) == Side::Inside {
                    continue;
                }
                let certain = brute_force_distance(&samples, centre) - spacing;
                if !(certain >= 0.6) {
                    continue;
                }
                checked += 1;
                let painted = image.get(px, py).x;
                if painted > worst {
                    worst = painted;
                    worst_at = (px, py, certain);
                }
            }
        }
        assert!(
            checked > 200,
            "only {checked} pixels of {} were far enough outside to test",
            target.name
        );
        assert!(
            worst <= 1.0e-12,
            "on {} the cheap analytic arm painted {worst} coverage at pixel \
             ({},{}), whose centre is at least {} px outside the shape. The arm \
             is clamp(0.5 - d, 0, 1), so any pixel more than half a pixel out \
             must be paper. It is not, because the d it reads is \
             Target::sdf = sign * clearance, and clearance is min |sdf_i| over \
             the PRIMITIVES - a certified LOWER bound on the distance to the \
             combined boundary, which is exactly what target.rs's own prose says \
             must never be conflated with a distance. On a corner the bound \
             collapses to zero all along the extension of each edge line past \
             the apex, and the arm paints a half-lit ray across open paper",
            target.name,
            worst_at.0,
            worst_at.1,
            worst_at.2
        );
    }
}

#[test]
fn the_cheap_arms_error_against_exact_coverage_is_bounded_on_the_corner() {
    for target in Target::<f64>::all() {
        let mut budget = coverage_budget(CELL_LIMIT);
        let (exact, _) = match exact_coverage(&target, 10, &mut budget) {
            Ok(v) => v,
            Err(e) => panic!("the exact coverage of {} failed: {e:?}", target.name),
        };
        let cheap = sdf_coverage(&target);
        let (width, height) = target.viewport;
        let mut peak = 0.0f64;
        let mut sum = 0.0f64;
        for py in 0..height {
            for px in 0..width {
                let d = (cheap.get(px, py).x - exact.get(px, py).x).abs();
                sum += d;
                if d > peak {
                    peak = d;
                }
            }
        }
        let mean = sum / ((width as f64) * (height as f64));
        println!("{} peak {peak:.6} mean {mean:.8}", target.name);
        assert!(
            peak <= 0.20 && mean <= 0.0020,
            "the cheap arm is {peak} off exact coverage at its worst pixel on \
             {} and {mean} off on average. The arm is a linear-in-distance \
             approximation of box coverage, so the measured ceiling is 0.1775 \
             peak and 0.00182 mean, both on the glyph. Reading the primitive \
             clearance instead of the shape's own combined distance put those \
             at 0.4940 and 0.00461, and put the corner at 0.4869 and 0.00494 \
             against the 0.0335 and 0.000438 it reaches from the combined \
             distance",
            target.name
        );
    }
}
