use lights::image::{FloatImage, RasterLayer};
use lights::primitive::{FeatureId, Part, Primitive, PrimitiveSet};
use lights::{Rgb, Similarity2, Vec2};
use std::path::PathBuf;

fn disc_set(ink: Rgb<f64>) -> PrimitiveSet<f64> {
    PrimitiveSet {
        ink,
        parts: vec![Part {
            id: FeatureId(0),
            shape: Primitive::Disc {
                centre: Vec2::new(8.0, 8.0),
                radius: 5.0,
            },
        }],
        regions: Vec::new(),
        holes: vec![Part {
            id: FeatureId(1),
            shape: Primitive::Disc {
                centre: Vec2::new(8.0, 8.0),
                radius: 2.0,
            },
        }],
    }
}

fn reference_coverage(set: &PrimitiveSet<f64>, view: Similarity2<f64>, w: u32, h: u32) -> Vec<f64> {
    let mut out = Vec::new();
    for py in 0..h {
        for px in 0..w {
            let centre = Vec2::new(px as f64 + 0.5, py as f64 + 0.5);
            let d = view.scale() * set.combined(view.invert(centre));
            out.push((0.5 - d).clamp(0.0, 1.0));
        }
    }
    out
}

#[test]
fn the_core_pass_is_half_minus_the_scaled_combined_distance_per_pixel() {
    let set = disc_set(Rgb::new(1.0, 1.0, 1.0));
    let view = Similarity2::identity();
    let layer: RasterLayer<f64> = set.render(view, (16, 16)).unwrap();
    let want = reference_coverage(&set, view, 16, 16);
    let got: Vec<f64> = layer.coverage.pixels.iter().map(|p| p.x).collect();
    assert_eq!(got, want);
    let partial = got.iter().filter(|c| **c > 0.0 && **c < 1.0).count();
    let full = got.iter().filter(|c| **c == 1.0).count();
    let empty = got.iter().filter(|c| **c == 0.0).count();
    assert!(partial > 0 && full > 0 && empty > 0, "partial {partial} full {full} empty {empty}: the case exercises no edge");
    let hole_centre = layer.coverage.get(8, 8).x;
    assert_eq!(hole_centre, 0.0, "the hole at the centre is not carved");
}

#[test]
fn the_premultiplied_colour_is_ink_times_coverage() {
    let ink = Rgb::new(0.9, 0.4, 0.1);
    let set = disc_set(ink);
    let layer = set.render(Similarity2::identity(), (16, 16)).unwrap();
    for (c, p) in layer.coverage.pixels.iter().zip(layer.premultiplied.pixels.iter()) {
        assert_eq!(c.x, c.y);
        assert_eq!(c.x, c.z);
        assert_eq!(*p, ink * c.x);
    }
}

#[test]
fn a_zero_viewport_is_refused_not_rendered_empty() {
    let set = disc_set(Rgb::new(1.0, 1.0, 1.0));
    assert!(set.render(Similarity2::identity(), (0, 16)).is_err());
    assert!(set.render(Similarity2::identity(), (16, 0)).is_err());
}

#[test]
fn the_pass_composites_over_a_canvas_it_never_saw() {
    let set = disc_set(Rgb::new(1.0, 0.0, 0.0));
    let layer = set.render(Similarity2::identity(), (16, 16)).unwrap();
    let bg = Rgb::new(0.0, 0.0, 1.0);
    let canvas = FloatImage::filled(16, 16, bg);
    let canvas = layer.premultiplied.composite_over(&layer.coverage, &canvas).unwrap();
    let outside = canvas.get(0, 0);
    let inside = canvas.get(8, 4);
    assert_eq!(outside, bg);
    assert_eq!(inside, Rgb::new(1.0, 0.0, 0.0));
}

fn crate_file(rel: &str) -> String {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push(rel);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()))
}

#[test]
fn the_core_module_never_names_fit_and_is_not_feature_gated() {
    let module = crate_file("src/primitive.rs");
    assert!(!module.is_empty());
    for (i, line) in module.lines().enumerate() {
        assert!(!line.contains("fit::") && !line.contains("feature = \"fit\""), "src/primitive.rs:{} names fit: {line}", i + 1);
    }
    let lib = crate_file("src/lib.rs");
    let lines: Vec<&str> = lib.lines().collect();
    let at = lines.iter().position(|l| l.trim() == "pub mod primitive;").expect("lib.rs declares no `pub mod primitive;`");
    assert!(at > 0 && !lines[at - 1].contains("cfg("), "`pub mod primitive;` sits under a cfg gate: {}", lines[at - 1]);
}

#[cfg(feature = "fit")]
#[test]
fn the_core_pass_equals_render_analytic_bit_for_bit_on_t1_to_t6() {
    use lights::fit::analytic::{AnalyticArm, render_analytic};
    use lights::fit::target::Target;
    let targets = Target::<f64>::all();
    assert_eq!(targets.len(), 6, "Target::all() is not t1..t6");
    for target in targets.iter() {
        let mut budget = lights::Budget::new("tests::primitive_pass", u64::MAX);
        let (analytic, _) = render_analytic(target, AnalyticArm::Sdf, 0, &mut budget).unwrap();
        let layer = target.set.render(Similarity2::identity(), target.viewport).unwrap();
        let want = reference_coverage(&target.set, Similarity2::identity(), target.viewport.0, target.viewport.1);
        let got: Vec<f64> = layer.coverage.pixels.iter().map(|p| p.x).collect();
        assert_eq!(got, want, "{}: the core pass is not the per-pixel formula", target.name);
        assert_eq!(analytic.pixels, layer.premultiplied.pixels, "{}: render_analytic's sdf arm differs from the core pass", target.name);
    }
}
