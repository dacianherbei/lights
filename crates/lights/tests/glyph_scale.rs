#![cfg(feature = "fit")]
use lights::fit::glyph_scale::{
    BROKEN_FACTOR, EDGE_FLOOR, HAIRLINE_INK_SWING, MATCH_TOLERANCE, GlyphScaleRow, floor_row, horizontal_hairline, verdict, aligned_edge, didone_n, fine_atoms, glyph_curve,
    inflation_for, matches_down_to, predicted_at, reference_at, residual, viewport_at,
};
use lights::fit::target::Target;
use lights::numerics::vector::Vec2;

const CALIBRATION_FINE_PER_UNIT: f64 = 8.0;
const CALIBRATION_MARGIN_UNITS: f64 = 4.0;

fn count_above(a: &lights::image::FloatImage<f64>, b: &lights::image::FloatImage<f64>, level: f64) -> usize {
    a.pixels
        .iter()
        .zip(b.pixels.iter())
        .filter(|(x, y)| (x.x - y.x).abs() > level)
        .count()
}

fn aligned_prediction(shift: f64) -> (lights::image::FloatImage<f64>, lights::image::FloatImage<f64>) {
    let target = aligned_edge();
    let atoms = match fine_atoms(&target, CALIBRATION_FINE_PER_UNIT, CALIBRATION_MARGIN_UNITS) {
        Ok(a) => a,
        Err(e) => panic!("fine atoms: {e}"),
    };
    let inflation = match inflation_for(&atoms, 1.0) {
        Ok(v) => v,
        Err(e) => panic!("inflation: {e}"),
    };
    let reference = match reference_at(&target, 1.0, Vec2::new(0.0, 0.0)) {
        Ok(r) => r,
        Err(e) => panic!("reference: {e}"),
    };
    let predicted = predicted_at(&atoms, 1.0, viewport_at(&target, 1.0), inflation, Vec2::new(shift, 0.0));
    (reference, predicted)
}

#[test]
fn a_half_pixel_shift_of_the_reference_costs_exactly_half_a_pixel_in_every_row() {
    let target = aligned_edge();
    let (a, b) = match (
        reference_at(&target, 1.0, Vec2::new(0.0, 0.0)),
        reference_at(&target, 1.0, Vec2::new(0.5, 0.0)),
    ) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(e), _) | (_, Err(e)) => panic!("reference: {e}"),
    };
    let r = match residual(&a, &b) {
        Ok(r) => r,
        Err(e) => panic!("residual: {e}"),
    };
    assert!((r.max_abs - 0.5).abs() < 1e-12, "max_abs {}", r.max_abs);
    assert_eq!(count_above(&a, &b, 1e-12), 16, "one pixel per row, 16 rows");
    assert_eq!(count_above(&a, &b, 0.5 - 1e-12), 16);
}

#[test]
fn inflation_meets_an_aligned_edge_at_the_box_against_gaussian_constant() {
    let (reference, predicted) = aligned_prediction(0.0);
    let r = match residual(&reference, &predicted) {
        Ok(r) => r,
        Err(e) => panic!("residual: {e}"),
    };
    println!("aligned edge: max_abs {:.6} predicted {:.6} ink_ratio {:.6}", r.max_abs, EDGE_FLOOR, r.ink_ratio);
    assert!((r.max_abs - EDGE_FLOOR).abs() < 1e-3, "max_abs {} against {}", r.max_abs, EDGE_FLOOR);
    assert!((r.ink_ratio - 1.0).abs() < 1e-3, "ink_ratio {}", r.ink_ratio);
}

#[test]
fn a_half_pixel_convention_error_in_the_prediction_is_seen() {
    let (reference, predicted) = aligned_prediction(0.5);
    let r = match residual(&reference, &predicted) {
        Ok(r) => r,
        Err(e) => panic!("residual: {e}"),
    };
    println!("shifted prediction: max_abs {:.6}", r.max_abs);
    assert!(r.max_abs > 0.45, "a half-pixel origin error must read as about 0.5, got {}", r.max_abs);
}

fn print_curve(target: &Target<f64>, rows: &[GlyphScaleRow]) {
    println!("outline\tem_px\tthinnest_px\tinflation_px2\tinflated_max_abs\tinflated_mean_abs\tink_low\tink_high\tscaled_only_max_abs\tscaled_only_mean_abs\tsupport");
    for r in rows {
        println!(
            "{}\t{}\t{:.3}\t{:.5}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{}",
            target.name,
            r.em_px,
            r.thinnest_px,
            r.inflation,
            r.inflated.max_abs,
            r.inflated.mean_abs,
            r.ink_low,
            r.ink_high,
            r.scaled_only.max_abs,
            r.scaled_only.mean_abs,
            r.inflated.support
        );
    }
    let floor = match floor_row(rows) {
        Some(f) => f,
        None => panic!("no row is resolved by the fine grid"),
    };
    for r in rows.iter().filter(|r| r.em_px <= floor.em_px) {
        println!("{}\tverdict\t{}\t{}\texcess {:.4}", target.name, r.em_px, verdict(r, floor).name(), r.inflated.max_abs - floor.inflated.max_abs);
    }
    println!(
        "{}\tmatches_down_to\t{:?}\tfloor {}px max_abs {:.4}, holds within {}, broken beyond {}",
        target.name,
        matches_down_to(rows),
        floor.em_px,
        floor.inflated.max_abs,
        MATCH_TOLERANCE,
        BROKEN_FACTOR * MATCH_TOLERANCE
    );
}

fn run(target: Target<f64>) -> Vec<GlyphScaleRow> {
    let rows = match glyph_curve(&target) {
        Ok(r) => r,
        Err(e) => panic!("curve for {}: {e}", target.name),
    };
    print_curve(&target, &rows);
    rows
}

#[test]
fn the_glyph_over_scale_curve() {
    let monoline = run(Target::t6_glyph_like());
    let didone = run(didone_n());
    assert_eq!(matches_down_to(&monoline), Some(10));
    assert_eq!(matches_down_to(&didone), Some(48));
}

#[test]
fn a_horizontal_hairline_swings_its_ink_with_subpixel_position() {
    let mut low = f64::INFINITY;
    let mut high = f64::NEG_INFINITY;
    for step in 0..8 {
        let phase = step as f64 / 8.0;
        let target = horizontal_hairline(phase, 0.1);
        let atoms = match fine_atoms(&target, 48.0, 0.0) {
            Ok(a) => a,
            Err(e) => panic!("{e}"),
        };
        let inflation = match inflation_for(&atoms, 1.0) {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        let reference = match reference_at(&target, 1.0, Vec2::new(0.0, 0.0)) {
            Ok(r) => r,
            Err(e) => panic!("{e}"),
        };
        let predicted = predicted_at(&atoms, 1.0, viewport_at(&target, 1.0), inflation, Vec2::new(0.0, 0.0));
        let r = match residual(&reference, &predicted) {
            Ok(r) => r,
            Err(e) => panic!("{e}"),
        };
        println!("hairline 0.2px phase {phase:.3}: ink_ratio {:.4} max_abs {:.4}", r.ink_ratio, r.max_abs);
        low = low.min(r.ink_ratio);
        high = high.max(r.ink_ratio);
    }
    println!("hairline ink swing: low {low:.4} high {high:.4}, predicted 1 -/+ {HAIRLINE_INK_SWING}");
    assert!((high - 1.0 - HAIRLINE_INK_SWING).abs() < 0.02, "high {high}");
    assert!((1.0 - low - HAIRLINE_INK_SWING).abs() < 0.02, "low {low}");
}

#[test]
#[ignore]
fn the_finest_row_does_not_move_when_the_fine_grid_doubles() {
    for (target, fine) in [(Target::t6_glyph_like(), 48.0), (Target::t6_glyph_like(), 96.0), (didone_n(), 48.0), (didone_n(), 96.0)] {
        let rows = match lights::fit::glyph_scale::curve(&target, lights::fit::target::GLYPH_EM, &[64, 96, 128, 200], fine) {
            Ok(r) => r,
            Err(e) => panic!("{e}"),
        };
        for r in rows {
            println!("{} fine {fine} em {} max {:.5} mean {:.5} ink {:.5}", target.name, r.em_px, r.inflated.max_abs, r.inflated.mean_abs, r.ink_high);
        }
    }
}
