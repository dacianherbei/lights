use lights::{
    Clip, DrawList, DrawOp, DrawShape, FloatImage, Ink, LightsError, Rgb, Similarity2, Vec2,
};
use std::num::NonZeroU32;

fn v(x: f64, y: f64) -> Vec2<f64> {
    Vec2::new(x, y)
}

fn cov(layer: &lights::RasterLayer<f64>, x: u32, y: u32) -> f64 {
    layer.coverage.get(x, y).x
}

fn total_coverage(layer: &lights::RasterLayer<f64>) -> f64 {
    layer.coverage.pixels.iter().map(|p| p.x).sum()
}

fn one(shape: DrawShape<f64>, ink: Ink<f64>) -> DrawList<f64> {
    let mut list = DrawList::new();
    list.push(DrawOp::new(shape, ink)).unwrap();
    list
}

fn white() -> Ink<f64> {
    Ink::opaque(Rgb::splat(1.0))
}

#[test]
fn a_pixel_aligned_rect_covers_exactly_its_area() {
    let list = one(
        DrawShape::Rect {
            min: v(2.0, 3.0),
            max: v(7.0, 9.0),
            corner_radius: 0.0,
        },
        white(),
    );
    let layer = list.render(12, 12).unwrap();
    for y in 0..12 {
        for x in 0..12 {
            let inside = (2..7).contains(&x) && (3..9).contains(&y);
            let expected = if inside { 1.0 } else { 0.0 };
            assert_eq!(cov(&layer, x, y), expected, "pixel ({x},{y})");
        }
    }
    assert_eq!(total_coverage(&layer), 30.0);
}

#[test]
fn a_fractional_rect_edge_covers_its_area_fraction() {
    let list = one(
        DrawShape::Rect {
            min: v(2.25, 0.0),
            max: v(6.0, 4.0),
            corner_radius: 0.0,
        },
        white(),
    );
    let layer = list.render(8, 4).unwrap();
    for y in 0..4 {
        assert!((cov(&layer, 2, y) - 0.75).abs() < 1e-12);
        assert_eq!(cov(&layer, 1, y), 0.0);
        assert_eq!(cov(&layer, 3, y), 1.0);
    }
    assert!((total_coverage(&layer) - 3.75 * 4.0).abs() < 1e-12);
}

#[test]
fn the_later_op_paints_over_the_earlier_and_order_is_the_list_order() {
    let red = Ink::opaque(Rgb::new(1.0, 0.0, 0.0));
    let blue = Ink::opaque(Rgb::new(0.0, 0.0, 1.0));
    let left = DrawShape::Rect {
        min: v(0.0, 0.0),
        max: v(6.0, 4.0),
        corner_radius: 0.0,
    };
    let right = DrawShape::Rect {
        min: v(3.0, 0.0),
        max: v(9.0, 4.0),
        corner_radius: 0.0,
    };
    let mut forward = DrawList::new();
    forward.push(DrawOp::new(left.clone(), red)).unwrap();
    forward.push(DrawOp::new(right.clone(), blue)).unwrap();
    let mut backward = DrawList::new();
    backward.push(DrawOp::new(right, blue)).unwrap();
    backward.push(DrawOp::new(left, red)).unwrap();
    let f = forward.render(10, 4).unwrap();
    let b = backward.render(10, 4).unwrap();
    assert_eq!(f.premultiplied.get(4, 1), Rgb::new(0.0, 0.0, 1.0));
    assert_eq!(b.premultiplied.get(4, 1), Rgb::new(1.0, 0.0, 0.0));
    assert_eq!(f.premultiplied.get(1, 1), Rgb::new(1.0, 0.0, 0.0));
    assert_eq!(f.premultiplied.get(8, 1), Rgb::new(0.0, 0.0, 1.0));
}

#[test]
fn translucent_ink_composes_as_premultiplied_over() {
    let square = DrawShape::Rect {
        min: v(0.0, 0.0),
        max: v(4.0, 4.0),
        corner_radius: 0.0,
    };
    let first = Ink::straight(Rgb::new(1.0, 0.0, 0.0), 0.5);
    let second = Ink::straight(Rgb::new(0.0, 1.0, 0.0), 0.5);
    let mut list = DrawList::new();
    list.push(DrawOp::new(square.clone(), first)).unwrap();
    list.push(DrawOp::new(square, second)).unwrap();
    let layer = list.render(4, 4).unwrap();
    assert!((cov(&layer, 1, 1) - 0.75).abs() < 1e-12);
    let c = layer.premultiplied.get(1, 1);
    assert!((c.x - 0.25).abs() < 1e-12, "red {}", c.x);
    assert!((c.y - 0.5).abs() < 1e-12, "green {}", c.y);
    let paper = FloatImage::filled(4, 4, Rgb::new(0.0, 0.0, 1.0));
    let out = layer.over(&paper).unwrap().get(1, 1);
    assert!((out.x - 0.25).abs() < 1e-12);
    assert!((out.y - 0.5).abs() < 1e-12);
    assert!((out.z - 0.25).abs() < 1e-12);
}

#[test]
fn a_clip_removes_everything_outside_it_and_box_filters_its_edges() {
    let disc = DrawShape::Disc {
        centre: v(8.0, 8.0),
        radius: 6.0,
    };
    let mut list = DrawList::new();
    list.push(DrawOp::new(disc, white()).clipped(Clip::new(v(0.0, 7.5), v(8.5, 16.0))))
        .unwrap();
    let layer = list.render(16, 16).unwrap();
    for y in 0..16 {
        for x in 9..16 {
            assert_eq!(cov(&layer, x, y), 0.0, "right of the clip at ({x},{y})");
        }
    }
    for x in 0..16 {
        for y in 0..7 {
            assert_eq!(cov(&layer, x, y), 0.0, "above the clip at ({x},{y})");
        }
    }
    assert!((cov(&layer, 8, 10) - 0.5).abs() < 1e-12);
    assert!((cov(&layer, 5, 7) - 0.5).abs() < 1e-12);
    assert!((cov(&layer, 5, 7) * cov(&layer, 8, 7) - 0.125).abs() < 1e-12);
    assert_eq!(cov(&layer, 6, 10), 1.0);
}

fn square_outline(closed: bool, half_width: f64) -> DrawShape<f64> {
    DrawShape::Polyline {
        points: vec![v(3.0, 3.0), v(13.0, 3.0), v(13.0, 13.0), v(3.0, 13.0)],
        closed,
        half_width,
    }
}

#[test]
fn a_closed_polyline_draws_its_closing_edge_and_an_open_one_does_not() {
    let closed = one(square_outline(true, 0.5), white()).render(16, 16).unwrap();
    let open = one(square_outline(false, 0.5), white()).render(16, 16).unwrap();
    assert!(cov(&closed, 2, 8) + cov(&closed, 3, 8) > 0.99);
    assert_eq!(cov(&open, 2, 8) + cov(&open, 3, 8), 0.0);
    assert!(cov(&open, 12, 8) + cov(&open, 13, 8) > 0.99);
}

#[test]
fn a_polyline_is_one_shape_so_a_translucent_joint_is_not_painted_twice() {
    let ink = Ink::straight(Rgb::splat(1.0), 0.5);
    let joined = one(square_outline(true, 1.0), ink).render(16, 16).unwrap();
    let peak = joined
        .coverage
        .pixels
        .iter()
        .map(|p| p.x)
        .fold(0.0, f64::max);
    assert!((peak - 0.5).abs() < 1e-12, "joined peak {peak}");
    let corners = [v(3.0, 3.0), v(13.0, 3.0), v(13.0, 13.0), v(3.0, 13.0)];
    let mut separate = DrawList::new();
    for i in 0..4 {
        let shape = DrawShape::Segment {
            a: corners[i],
            b: corners[(i + 1) % 4],
            half_width: 1.0,
        };
        separate.push(DrawOp::new(shape, ink)).unwrap();
    }
    let seams = separate.render(16, 16).unwrap();
    assert!((cov(&seams, 3, 3) - 0.75).abs() < 1e-12, "a corner painted by two segments");
}

fn star(centre: Vec2<f64>, r: f64) -> Vec<Vec2<f64>> {
    (0..10)
        .map(|i| {
            let theta = std::f64::consts::PI * (i as f64) / 5.0 - std::f64::consts::FRAC_PI_2;
            let radius = if i % 2 == 0 { r } else { r * 0.45 };
            v(centre.x + radius * theta.cos(), centre.y + radius * theta.sin())
        })
        .collect()
}

fn brute_boundary_distance(points: &[Vec2<f64>], p: Vec2<f64>) -> f64 {
    let n = points.len();
    let mut best = f64::INFINITY;
    for i in 0..n {
        let a = points[i];
        let b = points[(i + 1) % n];
        for k in 0..=4000 {
            let t = k as f64 / 4000.0;
            let q = a + (b - a) * t;
            best = best.min((p - q).length());
        }
    }
    best
}

#[test]
fn a_non_convex_polygon_has_an_exact_signed_distance() {
    let points = star(v(0.0, 0.0), 10.0);
    let shape = DrawShape::Polygon {
        points: points.clone(),
    };
    assert!(shape.sdf(v(0.0, 0.0)) < -3.0, "the centre is inside");
    let notch = (points[0] + points[2]) * 0.5;
    assert!(shape.sdf(notch) > 0.5, "the notch between two tips is outside");
    let mut checked = 0;
    for gy in -12..=12 {
        for gx in -12..=12 {
            let p = v(gx as f64 + 0.3, gy as f64 + 0.1);
            let got = shape.sdf(p);
            let want = brute_boundary_distance(&points, p);
            assert!(
                (got.abs() - want).abs() < 5e-3,
                "|sdf| at {p:?}: {got} against a sampled {want}"
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 625);
}

#[test]
fn a_filled_star_covers_its_area_to_within_the_sampling_of_the_edge() {
    let points = star(v(16.0, 16.0), 12.0);
    let layer = one(DrawShape::Polygon { points: points.clone() }, white())
        .render(32, 32)
        .unwrap();
    let n = points.len();
    let mut twice = 0.0;
    for i in 0..n {
        let a = points[i];
        let b = points[(i + 1) % n];
        twice += a.x * b.y - b.x * a.y;
    }
    let area = twice.abs() * 0.5;
    let drawn = total_coverage(&layer);
    assert!(
        (drawn - area).abs() / area < 0.01,
        "drawn {drawn} against an exact area {area}"
    );
}

#[test]
fn malformed_ops_are_refused_with_their_index() {
    let mut list = DrawList::new();
    list.push(DrawOp::new(
        DrawShape::Disc {
            centre: v(1.0, 1.0),
            radius: 1.0,
        },
        white(),
    ))
    .unwrap();
    let cases: Vec<DrawOp<f64>> = vec![
        DrawOp::new(
            DrawShape::Polygon {
                points: vec![v(0.0, 0.0), v(1.0, 0.0)],
            },
            white(),
        ),
        DrawOp::new(
            DrawShape::Disc {
                centre: v(f64::NAN, 0.0),
                radius: 1.0,
            },
            white(),
        ),
        DrawOp::new(
            DrawShape::Segment {
                a: v(0.0, 0.0),
                b: v(1.0, 0.0),
                half_width: -0.5,
            },
            white(),
        ),
        DrawOp::new(
            DrawShape::Rect {
                min: v(4.0, 0.0),
                max: v(1.0, 2.0),
                corner_radius: 0.0,
            },
            white(),
        ),
        DrawOp::new(
            DrawShape::RectOutline {
                min: v(0.0, 0.0),
                max: v(4.0, 2.0),
                corner_radius: -1.0,
                half_width: 0.5,
            },
            white(),
        ),
        DrawOp::new(
            DrawShape::Disc {
                centre: v(1.0, 1.0),
                radius: 1.0,
            },
            Ink::premultiplied(Rgb::splat(0.5), 1.5),
        ),    ];
    for op in cases {
        match list.push(op) {
            Err(LightsError::InvalidDrawOp { index, .. }) => assert_eq!(index, 1),
            other => panic!("expected a refusal, got {other:?}"),
        }
    }
    assert_eq!(list.len(), 1);
    assert!(matches!(
        list.render(0, 4),
        Err(LightsError::ViewportZero { .. })
    ));
}

fn rounded_outline(min: Vec2<f64>, max: Vec2<f64>, r: f64) -> Vec<Vec2<f64>> {
    let r = r.min((max.x - min.x) * 0.5).min((max.y - min.y) * 0.5);
    let corners = [
        (v(max.x - r, max.y - r), 0.0),
        (v(min.x + r, max.y - r), 0.5),
        (v(min.x + r, min.y + r), 1.0),
        (v(max.x - r, min.y + r), 1.5),
    ];
    let mut points = Vec::new();
    for (centre, start) in corners {
        for k in 0..=256 {
            let theta = std::f64::consts::PI * (start + 0.5 * k as f64 / 256.0);
            points.push(v(centre.x + r * theta.cos(), centre.y + r * theta.sin()));
        }
    }
    points
}

fn inside_outline(points: &[Vec2<f64>], p: Vec2<f64>) -> bool {
    let n = points.len();
    let mut inside = false;
    for i in 0..n {
        let a = points[i];
        let b = points[(i + 1) % n];
        if (a.y > p.y) != (b.y > p.y) && p.x < a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y) {
            inside = !inside;
        }
    }
    inside
}

fn outline_distance(points: &[Vec2<f64>], p: Vec2<f64>) -> f64 {
    let n = points.len();
    let mut best = f64::INFINITY;
    for i in 0..n {
        let a = points[i];
        let b = points[(i + 1) % n];
        let ab = b - a;
        let t = ((p - a).dot(ab) / ab.length_squared().max(1e-300)).clamp(0.0, 1.0);
        best = best.min((p - (a + ab * t)).length());
    }
    best
}

#[test]
fn a_rounded_rect_has_the_signed_distance_of_its_sampled_outline() {
    let (min, max, r, half_width) = (v(2.0, 3.0), v(22.0, 15.0), 4.0, 0.75);
    let outline = rounded_outline(min, max, r);
    let fill = DrawShape::Rect { min, max, corner_radius: r };
    let stroke = DrawShape::RectOutline { min, max, corner_radius: r, half_width };
    let mut checked = 0;
    let mut in_a_corner = 0;
    for gy in 0..49 {
        for gx in 0..65 {
            let p = v(gx as f64 * 0.37 + 0.05, gy as f64 * 0.37 + 0.02);
            let magnitude = outline_distance(&outline, p);
            let want = if inside_outline(&outline, p) { -magnitude } else { magnitude };
            let got = fill.sdf(p);
            assert!((got - want).abs() < 1e-3, "fill sdf at {p:?}: {got} against a sampled {want}");
            let got = stroke.sdf(p);
            let want = magnitude - half_width;
            assert!((got - want).abs() < 1e-3, "outline sdf at {p:?}: {got} against a sampled {want}");
            let cx = (p.x - 12.0).abs() > 6.0;
            let cy = (p.y - 9.0).abs() > 2.0;
            if cx && cy {
                in_a_corner += 1;
            }
            checked += 1;
        }
    }
    assert_eq!(checked, 65 * 49);
    assert!(in_a_corner > 400, "only {in_a_corner} samples fell where the corners differ");
}

#[test]
fn a_rounded_rect_loses_exactly_its_four_corner_areas() {
    let (w, h, r) = (20.0, 12.0, 4.0);
    let layer = one(
        DrawShape::Rect { min: v(2.0, 3.0), max: v(2.0 + w, 3.0 + h), corner_radius: r },
        white(),
    )
    .render(26, 18)
    .unwrap();
    let area = w * h - (4.0 - std::f64::consts::PI) * r * r;
    let drawn = total_coverage(&layer);
    let removed = w * h - area;
    assert!(
        (drawn - area).abs() < 0.05 * removed,
        "drawn {drawn} against an exact area {area}; the corners remove {removed}"
    );
    assert_eq!(cov(&layer, 2, 3), 0.0, "the corner pixel lies outside the arc");
    assert_eq!(cov(&layer, 12, 9), 1.0);
}

#[test]
fn a_corner_radius_beyond_half_the_short_side_is_clamped_to_a_pill() {
    let (w, h) = (30.0, 10.0);
    let min = v(1.0, 1.0);
    let max = v(1.0 + w, 1.0 + h);
    let pill = DrawShape::Rect { min, max, corner_radius: 100.0 };
    let exact = DrawShape::Rect { min, max, corner_radius: h * 0.5 };
    for gy in 0..24 {
        for gx in 0..64 {
            let p = v(gx as f64 * 0.5 + 0.1, gy as f64 * 0.5 + 0.2);
            assert_eq!(pill.sdf(p), exact.sdf(p), "at {p:?}");
        }
    }
    let layer = one(pill, white()).render(32, 12).unwrap();
    let area = (w - h) * h + std::f64::consts::PI * (h * 0.5) * (h * 0.5);
    let drawn = total_coverage(&layer);
    let removed = w * h - area;
    assert!(
        (drawn - area).abs() < 0.05 * removed,
        "drawn {drawn} against a stadium of area {area}; the ends remove {removed}"
    );
}

struct RatingWalk {
    bound_min: Vec2<f64>,
    bound_max: Vec2<f64>,
    step: f64,
    r: f64,
}

fn rating_walk(value: f64, fill: Ink<f64>, stroke: Ink<f64>) -> (DrawList<f64>, RatingWalk) {
    let max = 5u32;
    let size = 20.0;
    let gap = 2.0;
    let bound_min = v(2.0, 2.0);
    let width = max as f64 * size + (max - 1) as f64 * gap;
    let bound_max = v(bound_min.x + width, bound_min.y + size);
    let bound = Clip::new(bound_min, bound_max);
    let step = size + gap;
    let r = size * 0.5;
    let mut list = DrawList::new();
    for i in 0..max {
        let centre = v(bound_min.x + i as f64 * step + r, (bound_min.y + bound_max.y) * 0.5);
        let points = star(centre, r);
        let outline = DrawShape::Polyline {
            points: points.clone(),
            closed: true,
            half_width: 0.5,
        };
        list.push(DrawOp::new(outline, stroke).clipped(bound)).unwrap();
        let fraction = (value - i as f64).clamp(0.0, 1.0);
        if fraction > 0.0 {
            let clip = if fraction >= 0.999 {
                bound
            } else {
                let right = centre.x - r + 2.0 * r * fraction;
                Clip::new(bound_min, v(right.min(bound_max.x), bound_max.y))
            };
            list.push(DrawOp::new(DrawShape::Polygon { points }, fill).clipped(clip))
                .unwrap();
        }
    }
    (
        list,
        RatingWalk {
            bound_min,
            bound_max,
            step,
            r,
        },
    )
}

#[test]
fn the_rating_widget_is_drawn_by_the_display_list() {
    let accent = Rgb::new(0.2, 0.5, 1.0);
    let border = Rgb::splat(0.4);
    let (list, walk) = rating_walk(2.5, Ink::opaque(accent), Ink::opaque(border));
    assert_eq!(list.len(), 5 + 3);
    let layer = list.render(112, 24).unwrap();
    let cy = ((walk.bound_min.y + walk.bound_max.y) * 0.5) as u32;
    let cx = |i: u32| (walk.bound_min.x + i as f64 * walk.step + walk.r) as u32;
    for i in 0..2 {
        assert_eq!(cov(&layer, cx(i), cy), 1.0, "star {i} is full");
        assert_eq!(layer.premultiplied.get(cx(i), cy), accent);
    }
    assert_eq!(cov(&layer, cx(2) - 3, cy), 1.0, "the left half of star 2");
    assert_eq!(cov(&layer, cx(2) + 2, cy), 0.0, "the right half of star 2");
    for i in 3..5 {
        assert_eq!(cov(&layer, cx(i), cy), 0.0, "star {i} is empty inside");
        let tip = (walk.bound_min.y + 1.0) as u32;
        assert!(cov(&layer, cx(i), tip) > 0.0, "star {i} still has its outline");
        assert_eq!(layer.premultiplied.get(cx(i), tip).x, border.x * cov(&layer, cx(i), tip));
    }
    for x in 0..112 {
        for y in [0u32, 1, 22, 23] {
            assert_eq!(cov(&layer, x, y), 0.0, "outside the bound rect at ({x},{y})");
        }
    }
}

#[test]
fn the_disabled_rating_colour_is_expressible_because_ink_is_premultiplied() {
    let alpha = 90.0 / 255.0;
    let accent = Rgb::new(0.2, 0.5, 1.0);
    let muted = Ink::premultiplied(accent, alpha);
    let (list, walk) = rating_walk(1.0, muted, Ink::opaque(Rgb::splat(0.4)));
    let layer = list.render(112, 24).unwrap();
    let x = (walk.bound_min.x + walk.r) as u32;
    let y = ((walk.bound_min.y + walk.bound_max.y) * 0.5) as u32;
    assert!((cov(&layer, x, y) - alpha).abs() < 1e-12);
    assert_eq!(layer.premultiplied.get(x, y), accent);
    assert!(accent.z > alpha, "a premultiplied channel above its alpha is kept, as egui keeps it");
}

fn every_shape_over_the_rating(value: f64) -> DrawList<f64> {
    let (mut list, _) = rating_walk(value, Ink::opaque(Rgb::new(0.2, 0.5, 1.0)), white());
    let ink = Ink::straight(Rgb::new(0.9, 0.3, 0.1), 0.6);
    let shapes = [
        DrawShape::Rect { min: v(4.25, 30.5), max: v(20.75, 41.0), corner_radius: 3.0 },
        DrawShape::RectOutline { min: v(24.0, 30.0), max: v(44.5, 42.25), corner_radius: 2.5, half_width: 0.75 },
        DrawShape::Segment { a: v(50.0, 31.0), b: v(70.5, 43.25), half_width: 1.25 },
        DrawShape::Disc { centre: v(80.5, 37.0), radius: 5.5 },
        DrawShape::Ring { centre: v(96.0, 37.5), radius: 6.0, half_width: 0.5 },
    ];
    for shape in shapes {
        list.push(DrawOp::new(shape, ink).clipped(Clip::new(v(3.5, 29.75), v(104.25, 44.0))))
            .unwrap();
    }
    list
}

fn drawn_at_device_coordinates(list: &DrawList<f64>, view: Similarity2<f64>) -> DrawList<f64> {
    assert_eq!(view.rotation(), 0.0, "this oracle maps boxes to boxes, so it cannot rotate");
    let s = view.scale();
    let at = |p: &Vec2<f64>| view.apply(*p);
    let mut out = DrawList::new();
    for op in list.ops() {
        let shape = match &op.shape {
            DrawShape::Rect { min, max, corner_radius } => DrawShape::Rect {
                min: at(min),
                max: at(max),
                corner_radius: corner_radius * s,
            },
            DrawShape::RectOutline { min, max, corner_radius, half_width } => DrawShape::RectOutline {
                min: at(min),
                max: at(max),
                corner_radius: corner_radius * s,
                half_width: half_width * s,
            },
            DrawShape::Segment { a, b, half_width } => DrawShape::Segment {
                a: at(a),
                b: at(b),
                half_width: half_width * s,
            },
            DrawShape::Disc { centre, radius } => DrawShape::Disc {
                centre: at(centre),
                radius: radius * s,
            },
            DrawShape::Ring { centre, radius, half_width } => DrawShape::Ring {
                centre: at(centre),
                radius: radius * s,
                half_width: half_width * s,
            },
            DrawShape::Polygon { points } => DrawShape::Polygon {
                points: points.iter().map(at).collect(),
            },
            DrawShape::Polyline { points, closed, half_width } => DrawShape::Polyline {
                points: points.iter().map(at).collect(),
                closed: *closed,
                half_width: half_width * s,
            },
        };
        let mut mapped = DrawOp::new(shape, op.ink);
        if let Some(clip) = op.clip {
            mapped = mapped.clipped(Clip::new(at(&clip.min), at(&clip.max)));
        }
        out.push(mapped).unwrap();
    }
    out
}

fn largest_layer_gap(a: &lights::RasterLayer<f64>, b: &lights::RasterLayer<f64>) -> f64 {
    let mut worst = 0.0f64;
    for (p, q) in a.coverage.pixels.iter().zip(b.coverage.pixels.iter()) {
        worst = worst.max((p.x - q.x).abs());
    }
    for (p, q) in a.premultiplied.pixels.iter().zip(b.premultiplied.pixels.iter()) {
        worst = worst.max((p.x - q.x).abs()).max((p.y - q.y).abs()).max((p.z - q.z).abs());
    }
    worst
}

#[test]
fn a_zoomed_and_panned_list_is_the_list_drawn_at_device_coordinates() {
    let list = every_shape_over_the_rating(2.5);
    let (w, h) = (300u32, 140u32);
    for view in [
        Similarity2::new(2.5, 0.0, v(-7.25, 3.5)).unwrap(),
        Similarity2::new(0.625, 0.0, v(11.0, 0.75)).unwrap(),
        Similarity2::new(1.0, 0.0, v(0.5, -0.25)).unwrap(),
    ] {
        let viewed = list.render_through(view, w, h).unwrap();
        let oracle = drawn_at_device_coordinates(&list, view).render(w, h).unwrap();
        let gap = largest_layer_gap(&viewed, &oracle);
        assert!(gap <= 1.0e-9, "{view:?}: the view differs from redrawing the list by {gap}");
        let unviewed = list.render(w, h).unwrap();
        assert!(
            largest_layer_gap(&viewed, &unviewed) > 0.5,
            "{view:?} left the picture where it was, so the comparison saw nothing"
        );
    }
}

#[test]
fn a_quarter_turned_list_moves_every_pixel_to_its_rotated_mate() {
    let (list, _) = rating_walk(2.5, Ink::opaque(Rgb::new(0.2, 0.5, 1.0)), white());
    let side = 112u32;
    let half = side as f64 / 2.0;
    let turn = Similarity2::new(1.0, std::f64::consts::FRAC_PI_2, v(2.0 * half, 0.0)).unwrap();
    let upright = list.render(side, side).unwrap();
    let turned = list.render_through(turn, side, side).unwrap();
    assert!(total_coverage(&upright) > 100.0, "the rating drew almost nothing upright");
    for py in 0..side {
        for px in 0..side {
            let want = upright.coverage.get(py, side - 1 - px).x;
            let got = turned.coverage.get(px, py).x;
            assert!(
                (want - got).abs() <= 1.0e-9,
                "pixel ({px},{py}) covers {got}, its rotated mate covers {want}"
            );
        }
    }
}

fn busy_list() -> DrawList<f64> {
    let mut list = every_shape_over_the_rating(3.5);
    let ink = Ink::straight(Rgb::new(0.1, 0.7, 0.4), 0.55);
    list.push(DrawOp::new(DrawShape::Polygon { points: star(v(60.0, 12.0), 9.0) }, ink)).unwrap();
    list.push(DrawOp::new(square_outline(false, 0.8), ink).clipped(Clip::new(v(0.0, 0.0), v(30.0, 30.0))))
        .unwrap();
    list.push(DrawOp::new(DrawShape::Rect { min: v(-5.0, 20.0), max: v(200.0, 26.5), corner_radius: 0.0 }, ink))
        .unwrap();
    list
}

fn views() -> [Similarity2<f64>; 3] {
    [
        Similarity2::identity(),
        Similarity2::new(2.5, 0.0, v(-7.25, 3.5)).unwrap(),
        Similarity2::new(1.3, 0.4, v(20.0, -5.0)).unwrap(),
    ]
}

fn first_differing_bit(a: &lights::RasterLayer<f64>, b: &lights::RasterLayer<f64>) -> Option<String> {
    let bits = |c: &Rgb<f64>| [c.x.to_bits(), c.y.to_bits(), c.z.to_bits()];
    for (name, x, y) in [("premultiplied", &a.premultiplied, &b.premultiplied), ("coverage", &a.coverage, &b.coverage)] {
        if x.pixels.len() != y.pixels.len() {
            return Some(format!("{name}: {} pixels against {}", x.pixels.len(), y.pixels.len()));
        }
        for (i, (p, q)) in x.pixels.iter().zip(y.pixels.iter()).enumerate() {
            if bits(p) != bits(q) {
                return Some(format!("{name} pixel {i}: {p:?} against {q:?}"));
            }
        }
    }
    None
}

#[test]
fn the_parallel_render_is_the_serial_render_bit_for_bit_at_every_band_height() {
    let list = busy_list();
    let (w, h) = (113, 47);
    let mut compared = 0;
    for view in views() {
        let serial = list.render_through(view, w, h).unwrap();
        assert!(total_coverage(&serial) > 100.0, "{view:?}: the list drew almost nothing, so equality would mean nothing");
        for band in [1, 2, 7, 16, 46, 47, 48, 500] {
            let parallel = list.render_through_parallel(view, w, h, NonZeroU32::new(band).unwrap()).unwrap();
            if let Some(gap) = first_differing_bit(&serial, &parallel) {
                panic!("{band}-row bands through {view:?} differ from the serial render at {gap}");
            }
            compared += 1;
        }
    }
    assert_eq!(compared, 24);
    let empty = list.render_through_parallel(Similarity2::identity(), 0, 5, NonZeroU32::new(4).unwrap());
    assert!(matches!(empty, Err(LightsError::ViewportZero { width: 0, height: 5 })), "{empty:?}");
}

#[test]
fn the_footprint_holds_every_pixel_an_op_covers_and_the_frame_is_empty_outside_every_footprint() {
    let list = busy_list();
    let (w, h) = (113u32, 47u32);
    for view in views() {
        let layer = list.render_through(view, w, h).unwrap();
        let mut claimed = vec![false; (w * h) as usize];
        let mut covered = 0usize;
        for (k, op) in list.ops().iter().enumerate() {
            let footprint = op.footprint(view, w, h);
            if let Some([x0, y0, x1, y1]) = footprint {
                assert!(x0 < x1 && y0 < y1 && x1 <= w && y1 <= h, "op {k}: footprint {footprint:?} in a {w}x{h} frame");
            }
            for py in 0..h {
                for px in 0..w {
                    let inside = footprint.is_some_and(|[x0, y0, x1, y1]| px >= x0 && px < x1 && py >= y0 && py < y1);
                    let c = op.coverage_through(view, px, py);
                    if c > 0.0 {
                        assert!(inside, "op {k} covers ({px},{py}) by {c} outside its footprint {footprint:?} through {view:?}");
                        covered += 1;
                    }
                    if inside {
                        claimed[(py * w + px) as usize] = true;
                    }
                }
            }
        }
        assert!(covered > 500, "{view:?}: only {covered} covered pixels, so the containment check saw little");
        for py in 0..h {
            for px in 0..w {
                if !claimed[(py * w + px) as usize] {
                    assert_eq!(layer.coverage.get(px, py).x, 0.0, "({px},{py}) is in no footprint and was painted");
                }
            }
        }
    }
    let away = Similarity2::new(1.0, 0.0, v(-1.0e4, -1.0e4)).unwrap();
    assert!(
        list.ops().iter().all(|op| op.footprint(away, w, h).is_none()),
        "an op moved off the frame still claims pixels"
    );
}
