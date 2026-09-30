#![cfg(feature = "fit")]
use lights::error::LightsError;
use lights::fit::coverage::{DEFAULT_CELL_LIMIT, DEFAULT_MAX_DEPTH, coverage_budget,
    coverage_image_to_colour, exact_coverage};
use lights::fit::target::Target;
use lights::fit::transform::weight::measure;
use lights::fit::transform::{
    Band, CHANNELS, Ranking, Subband, Transform, band_of, bands, forward, inverse, keep_largest,
    level_sizes, max_levels, rank_coefficients, truncate_and_reconstruct,
};
use lights::image::FloatImage;
use lights::numerics::vector::Rgb;
use lights::scene::demo::Lcg;

const SIDE: u32 = 64;
const SMALL: u32 = 16;
const SMALL_LEVELS: u32 = 2;

fn constant_field(width: u32, height: u32, value: f64) -> FloatImage<f64> {
    FloatImage::filled(width, height, Rgb::splat(value))
}

fn ramp_field(width: u32, height: u32) -> FloatImage<f64> {
    let mut out = FloatImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let fx = x as f64;
            let fy = y as f64;
            out.set(
                x,
                y,
                Rgb::new(
                    0.05 + 0.003 * fx + 0.0017 * fy,
                    0.11 + 0.0021 * fx - 0.0009 * fy,
                    0.30 - 0.0013 * fx + 0.0025 * fy,
                ),
            );
        }
    }
    out
}

fn smooth_field(width: u32, height: u32) -> FloatImage<f64> {
    let mut out = FloatImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let fx = x as f64;
            let fy = y as f64;
            out.set(
                x,
                y,
                Rgb::new(
                    0.5 + 0.4 * (fx * 0.19 + 0.3).sin() * (fy * 0.11 - 0.7).cos(),
                    0.45 + 0.3 * (fx * 0.07 - 1.1).cos() * (fy * 0.23 + 0.4).sin(),
                    0.6 - 0.35 * (fx * 0.13 + fy * 0.05).sin(),
                ),
            );
        }
    }
    out
}

fn noise_field(width: u32, height: u32, seed: u64) -> FloatImage<f64> {
    let mut rng = Lcg::new(seed);
    let mut out = FloatImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            out.set(x, y, Rgb::new(rng.unit(), rng.unit(), rng.unit()));
        }
    }
    out
}

fn step_field(width: u32, height: u32) -> FloatImage<f64> {
    let angle = 30.0f64.to_radians();
    let (sin, cos) = angle.sin_cos();
    let cut = 0.58 * (width as f64);
    let mut out = FloatImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let projected = (x as f64 + 0.5) * cos + (y as f64 + 0.5) * sin;
            let v = if projected > cut {
                Rgb::new(0.92, 0.88, 0.81)
            } else {
                Rgb::new(0.07, 0.11, 0.19)
            };
            out.set(x, y, v);
        }
    }
    out
}

fn rendered(target: &Target<f64>) -> FloatImage<f64> {
    let mut budget = coverage_budget(DEFAULT_CELL_LIMIT);
    let (coverage, _) = exact_coverage(target, DEFAULT_MAX_DEPTH, &mut budget).expect("coverage");
    coverage_image_to_colour(target, &coverage)
}

fn procedural_fields(width: u32, height: u32) -> Vec<(&'static str, FloatImage<f64>)> {
    vec![
        ("constant", constant_field(width, height, 0.37)),
        ("ramp", ramp_field(width, height)),
        ("smooth", smooth_field(width, height)),
        ("noise", noise_field(width, height, 20260908)),
        ("step", step_field(width, height)),
    ]
}

fn max_abs_difference(a: &FloatImage<f64>, b: &FloatImage<f64>) -> f64 {
    a.difference(b).expect("same size").max_abs().0
}

fn energy(image: &FloatImage<f64>) -> f64 {
    let mut acc = 0.0f64;
    for p in image.pixels.iter() {
        acc += p.x * p.x + p.y * p.y + p.z * p.z;
    }
    acc
}

fn relative_l2(truth: &FloatImage<f64>, approximation: &FloatImage<f64>) -> f64 {
    let residual = truth.difference(approximation).expect("same size");
    (energy(&residual) / energy(truth)).sqrt()
}

#[test]
fn forward_then_inverse_returns_every_field_it_was_given() {
    let levels = max_levels(SIDE, SIDE);
    assert_eq!(levels, 4, "a 64x64 field decomposes four times before the coarse side hits 4");
    let mut checked = 0usize;
    for (name, field) in procedural_fields(SIDE, SIDE) {
        let coefficients = forward(Transform::Haar, &field, levels).expect("forward");
        let back = inverse(&coefficients).expect("inverse");
        let error = max_abs_difference(&field, &back);
        assert!(
            error < 1.0e-12,
            "haar round trip on {name} lost {error}, which is not a round trip"
        );
        checked += 1;
    }
    for target in [Target::t2_straight_edge(), Target::t5_thin_stroke(), Target::t6_glyph_like()] {
        let field = rendered(&target);
        let side_levels = max_levels(field.width, field.height);
        let coefficients = forward(Transform::Haar, &field, side_levels).expect("forward");
        let back = inverse(&coefficients).expect("inverse");
        let error = max_abs_difference(&field, &back);
        assert!(
            error < 1.0e-12,
            "haar round trip on the rendered target {} lost {error}",
            target.name
        );
        checked += 1;
    }
    assert_eq!(checked, 8, "the round-trip scan must not pass by looking at nothing");
}

#[test]
fn a_constant_field_puts_every_unit_of_energy_in_the_coarsest_approximation_band() {
    let levels = max_levels(SIDE, SIDE);
    let field = constant_field(SIDE, SIDE, 0.37);
    let coefficients = forward(Transform::Haar, &field, levels).expect("forward");
    let approximation = Band { level: levels, subband: Subband::Approximation };
    let mut detail_energy = 0.0f64;
    let mut detail_cells = 0usize;
    for band in bands(levels) {
        if band == approximation {
            continue;
        }
        detail_energy += coefficients.band_energy(band);
    }
    for y in 0..SIDE {
        for x in 0..SIDE {
            if band_of(SIDE, SIDE, levels, x, y) != approximation {
                detail_cells += 1;
                for channel in 0..CHANNELS {
                    assert_eq!(
                        coefficients.get(channel, x, y),
                        0.0,
                        "haar differences a constant field to exactly zero, not nearly zero, \
                         but ({x},{y}) channel {channel} is not"
                    );
                }
            }
        }
    }
    assert_eq!(detail_energy, 0.0);
    assert_eq!(detail_cells, (SIDE * SIDE - (SIDE >> levels) * (SIDE >> levels)) as usize);
    let approximation_energy = coefficients.band_energy(approximation);
    assert!(
        (approximation_energy - energy(&field)).abs() < 1.0e-9,
        "the coarse band holds {approximation_energy} of the field's {}",
        energy(&field)
    );
    let coarse = coefficients.get(0, 0, 0);
    let expected = 0.37 * ((1u32 << levels) as f64);
    assert!(
        (coarse - expected).abs() < 1.0e-12,
        "an orthonormal haar approximation gains 2 per level: expected {expected}, got {coarse}"
    );
}

#[test]
fn the_haar_transform_preserves_energy_exactly_enough_to_make_keep_n_optimal() {
    let levels = max_levels(SIDE, SIDE);
    let mut checked = 0usize;
    for (name, field) in procedural_fields(SIDE, SIDE) {
        let coefficients = forward(Transform::Haar, &field, levels).expect("forward");
        let ratio = coefficients.energy() / energy(&field);
        assert!(
            (ratio - 1.0).abs() < 1.0e-12,
            "parseval fails on {name}: coefficients carry {ratio} of the field's energy, and \
             without parseval keeping the N largest is not the optimal N-term approximation"
        );
        checked += 1;
    }
    assert_eq!(checked, 5);
}

#[test]
fn every_haar_subband_weight_is_exactly_one_because_the_basis_is_orthonormal() {
    let levels = max_levels(SIDE, SIDE);
    let weights = measure(Transform::Haar, SIDE, SIDE, levels).expect("weights");
    assert_eq!(weights.entries.len(), 3 * (levels as usize) + 1);
    for (band, norm) in weights.entries.iter() {
        assert!(
            (norm - 1.0).abs() < 1.0e-12,
            "the synthesis basis function of {:?} at level {} has L2 norm {norm}, so this \
             transform is not the orthonormal one the keep-N argument assumes",
            band.subband,
            band.level
        );
    }
    let deviation = (weights.largest() - 1.0).abs().max((weights.smallest() - 1.0).abs());
    println!("haar subband weight deviation from unity: {deviation:e}");
    assert!(deviation < 1.0e-14, "measured haar weight deviation {deviation}");
}

#[test]
fn the_weighted_and_raw_rankings_are_the_same_permutation_for_haar() {
    let levels = max_levels(SMALL, SMALL);
    let field = noise_field(SMALL, SMALL, 4242);
    let coefficients = forward(Transform::Haar, &field, levels).expect("forward");
    let weights = measure(Transform::Haar, SMALL, SMALL, levels).expect("weights");
    let weighted = rank_coefficients(&coefficients, &weights, Ranking::Weighted).expect("rank");
    let raw = rank_coefficients(&coefficients, &weights, Ranking::RawMagnitude).expect("rank");
    assert_eq!(weighted.len(), CHANNELS * (SMALL as usize) * (SMALL as usize));
    let mut key_gap = 0.0f64;
    for (a, b) in weighted.iter().zip(raw.iter()) {
        assert_eq!(
            (a.channel, a.x, a.y),
            (b.channel, b.x, b.y),
            "the two rankings put different coefficients in the same rank, so the haar \
             weights are not the identity the orthonormality argument says they are"
        );
        key_gap = key_gap.max((a.key - b.key).abs());
    }
    assert!(
        key_gap < 1.0e-15,
        "the weighted and raw haar keys differ by {key_gap}, which is more than the rounding \
         of a measured unit norm"
    );
    assert!(key_gap > 0.0, "the measured haar weights are exactly one, which the recorded \
         deviation below says they are not; re-measure before trusting either");
}

#[test]
fn keeping_every_coefficient_is_bit_for_bit_the_untruncated_inverse() {
    let levels = max_levels(SIDE, SIDE);
    let field = step_field(SIDE, SIDE);
    let coefficients = forward(Transform::Haar, &field, levels).expect("forward");
    let weights = measure(Transform::Haar, SIDE, SIDE, levels).expect("weights");
    let terms = coefficients.terms();
    let kept = keep_largest(&coefficients, &weights, Ranking::Weighted, terms).expect("keep");
    assert_eq!(kept, coefficients, "keeping every term must zero nothing at all");
    let full = inverse(&coefficients).expect("inverse");
    let truncated = inverse(&kept).expect("inverse");
    for (a, b) in full.pixels.iter().zip(truncated.pixels.iter()) {
        assert_eq!(a.x.to_bits(), b.x.to_bits());
        assert_eq!(a.y.to_bits(), b.y.to_bits());
        assert_eq!(a.z.to_bits(), b.z.to_bits());
    }
    let error = max_abs_difference(&field, &truncated);
    assert!(
        error < 1.0e-12,
        "perfect reconstruction at the full coefficient count is the control for this whole \
         arm, and it lost {error}"
    );
}

#[test]
fn keeping_n_terms_keeps_the_n_largest_and_nothing_else() {
    let levels = max_levels(SMALL, SMALL);
    let field = smooth_field(SMALL, SMALL);
    let coefficients = forward(Transform::Haar, &field, levels).expect("forward");
    let weights = measure(Transform::Haar, SMALL, SMALL, levels).expect("weights");
    let terms = 37usize;
    let kept = keep_largest(&coefficients, &weights, Ranking::Weighted, terms).expect("keep");
    let mut smallest_kept = f64::INFINITY;
    let mut largest_dropped = 0.0f64;
    let mut kept_count = 0usize;
    let mut dropped_count = 0usize;
    for channel in 0..CHANNELS {
        for y in 0..SMALL {
            for x in 0..SMALL {
                let band = band_of(SMALL, SMALL, levels, x, y);
                let value = coefficients.get(channel, x, y);
                let key = value.abs() * weights.of(band);
                if kept.get(channel, x, y) != 0.0 {
                    assert_eq!(
                        kept.get(channel, x, y).to_bits(),
                        value.to_bits(),
                        "a retained coefficient was altered on the way through keep_largest"
                    );
                    kept_count += 1;
                    smallest_kept = smallest_kept.min(key);
                } else {
                    dropped_count += 1;
                    largest_dropped = largest_dropped.max(key);
                }
            }
        }
    }
    assert_eq!(kept_count, terms, "keep_largest retained {kept_count} of the {terms} asked for");
    assert_eq!(dropped_count, coefficients.terms() - terms);
    assert!(
        largest_dropped > 0.0,
        "every discarded coefficient is exactly zero, so this test cannot tell the largest \
         from the smallest and proves nothing"
    );
    assert!(
        smallest_kept >= largest_dropped,
        "the smallest retained coefficient has key {smallest_kept} and the largest discarded \
         one {largest_dropped}: keep-N is keeping the wrong end of the ranking, which is the \
         whole N-term claim"
    );
}

#[test]
fn the_n_term_curve_never_rises_as_a_term_is_added() {
    let field = noise_field(SMALL, SMALL, 99);
    let coefficients = forward(Transform::Haar, &field, SMALL_LEVELS).expect("forward");
    let weights = measure(Transform::Haar, SMALL, SMALL, SMALL_LEVELS).expect("weights");
    let terms = coefficients.terms();
    let mut previous = f64::INFINITY;
    let mut fell = 0usize;
    for n in 0..=terms {
        let kept = keep_largest(&coefficients, &weights, Ranking::Weighted, n).expect("keep");
        let reconstruction = inverse(&kept).expect("inverse");
        let error = relative_l2(&field, &reconstruction);
        assert!(
            error <= previous + 1.0e-12,
            "the haar curve rose from {previous} to {error} at N={n}: in an orthonormal basis \
             keeping one more of the largest coefficients cannot make the approximation worse"
        );
        if error < previous - 1.0e-12 {
            fell += 1;
        }
        previous = error;
    }
    assert!(
        fell > terms / 2,
        "only {fell} of {terms} steps improved the reconstruction, so this scan is measuring \
         a flat curve rather than an approximation"
    );
    assert!(previous < 1.0e-12, "the full term count must reconstruct exactly, got {previous}");
}

#[test]
fn a_second_level_only_touches_the_coarse_quadrant() {
    let field = smooth_field(SIDE, SIDE);
    let one = forward(Transform::Haar, &field, 1).expect("forward");
    let two = forward(Transform::Haar, &field, 2).expect("forward");
    let half = SIDE >> 1;
    let mut outside = 0usize;
    let mut inside_changed = 0usize;
    for channel in 0..CHANNELS {
        for y in 0..SIDE {
            for x in 0..SIDE {
                let a = one.get(channel, x, y);
                let b = two.get(channel, x, y);
                if x < half && y < half {
                    if a.to_bits() != b.to_bits() {
                        inside_changed += 1;
                    }
                } else {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "the second level rewrote the level-1 detail coefficient at \
                         ({x},{y}) channel {channel}, so it is not recursing into the coarse \
                         quadrant alone"
                    );
                    outside += 1;
                }
            }
        }
    }
    assert_eq!(outside, CHANNELS * ((SIDE * SIDE - half * half) as usize));
    assert!(
        inside_changed > (CHANNELS * ((half * half) as usize)) / 2,
        "only {inside_changed} coefficients inside the coarse quadrant moved, so the second \
         level did nothing"
    );
}

#[test]
fn the_transform_refuses_a_size_it_cannot_halve() {
    let odd = noise_field(SIDE - 1, SIDE, 7);
    match forward(Transform::Haar, &odd, 1) {
        Err(LightsError::SizeMismatch { a, b: _ }) => assert_eq!(a, (SIDE - 1, SIDE)),
        other => panic!("a 63-wide field cannot be halved, got {other:?}"),
    }
    let square = noise_field(SIDE, SIDE, 8);
    match forward(Transform::Haar, &square, max_levels(SIDE, SIDE) + 1) {
        Err(LightsError::SizeMismatch { .. }) => (),
        other => panic!("one level past the coarse-side floor must be refused, got {other:?}"),
    }
    match forward(Transform::Haar, &square, 0) {
        Err(LightsError::SizeMismatch { .. }) => (),
        other => panic!("a zero-level transform is not a transform, got {other:?}"),
    }
    assert_eq!(max_levels(SIDE, SIDE), 4);
    assert_eq!(max_levels(32, 32), 3);
    assert_eq!(max_levels(SMALL, SMALL), 2);
    assert_eq!(max_levels(4, 4), 0);
    assert_eq!(max_levels(48, 64), 3);
    assert_eq!(max_levels(24, 64), 2);
    assert_eq!(level_sizes(SIDE, SIDE, 3), vec![(64, 64), (32, 32), (16, 16)]);
}

#[test]
fn the_bands_partition_every_coefficient_exactly_once() {
    let levels = max_levels(SIDE, SIDE);
    let listed = bands(levels);
    assert_eq!(listed.len(), 3 * (levels as usize) + 1);
    let mut counted = vec![0usize; listed.len()];
    for y in 0..SIDE {
        for x in 0..SIDE {
            let band = band_of(SIDE, SIDE, levels, x, y);
            let index = listed.iter().position(|b| *b == band).expect("band is listed");
            counted[index] += 1;
        }
    }
    let mut total = 0usize;
    for (band, count) in listed.iter().zip(counted.iter()) {
        let side = (SIDE >> band.level) as usize;
        assert_eq!(*count, side * side, "band {:?} at level {}", band.subband, band.level);
        total += count;
    }
    assert_eq!(total, (SIDE * SIDE) as usize);
}

#[test]
fn the_convenience_path_agrees_with_the_three_steps_it_replaces() {
    let levels = max_levels(SMALL, SMALL);
    let field = step_field(SMALL, SMALL);
    let coefficients = forward(Transform::Haar, &field, levels).expect("forward");
    let weights = measure(Transform::Haar, SMALL, SMALL, levels).expect("weights");
    let kept = keep_largest(&coefficients, &weights, Ranking::Weighted, 40).expect("keep");
    let stepwise = inverse(&kept).expect("inverse");
    let direct =
        truncate_and_reconstruct(Transform::Haar, &field, levels, 40, Ranking::Weighted)
            .expect("truncate");
    for (a, b) in stepwise.pixels.iter().zip(direct.pixels.iter()) {
        assert_eq!(a.x.to_bits(), b.x.to_bits());
    }
    assert_eq!(kept.nonzero(), 40);
    assert!(relative_l2(&field, &direct) < 1.0);
}
