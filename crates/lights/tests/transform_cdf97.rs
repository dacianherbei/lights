#![cfg(feature = "fit")]
use lights::fit::coverage::{DEFAULT_CELL_LIMIT, DEFAULT_MAX_DEPTH, coverage_budget,
    coverage_image_to_colour, exact_coverage};
use lights::fit::target::Target;
use lights::fit::transform::cdf97::{ALPHA, BETA, DELTA, GAMMA, K, reflect};
use lights::fit::transform::weight::measure;
use lights::fit::transform::{
    Band, CHANNELS, Ranking, Subband, Transform, band_of, bands, cdf97, forward, inverse,
    keep_largest, max_levels, truncate_and_reconstruct,
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

fn steep_ramp_field(width: u32, height: u32) -> FloatImage<f64> {
    let mut out = FloatImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let fx = x as f64;
            let fy = y as f64;
            out.set(
                x,
                y,
                Rgb::new(
                    0.05 + 0.030 * fx + 0.017 * fy,
                    0.11 + 0.021 * fx - 0.009 * fy,
                    0.30 - 0.013 * fx + 0.025 * fy,
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

fn ladder(terms: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut n = 1usize;
    for _ in 0..20 {
        if n > terms {
            break;
        }
        out.push(n);
        n *= 2;
    }
    out.push(terms);
    out
}

fn curve(
    field: &FloatImage<f64>,
    levels: u32,
    ranking: Ranking,
    steps: &[usize],
) -> Vec<f64> {
    let coefficients = forward(Transform::Cdf97, field, levels).expect("forward");
    let weights = measure(Transform::Cdf97, field.width, field.height, levels).expect("weights");
    let mut out = Vec::with_capacity(steps.len());
    for n in steps.iter() {
        let kept = keep_largest(&coefficients, &weights, ranking, *n).expect("keep");
        let reconstruction = inverse(&kept).expect("inverse");
        out.push(relative_l2(field, &reconstruction));
    }
    out
}

#[test]
fn the_biorthogonal_forward_and_inverse_return_every_field_they_were_given() {
    let levels = max_levels(SIDE, SIDE);
    let mut checked = 0usize;
    for (name, field) in procedural_fields(SIDE, SIDE) {
        let coefficients = forward(Transform::Cdf97, &field, levels).expect("forward");
        let back = inverse(&coefficients).expect("inverse");
        let error = max_abs_difference(&field, &back);
        assert!(
            error < 1.0e-12,
            "cdf 9/7 round trip on {name} lost {error}, which is not a round trip"
        );
        checked += 1;
    }
    for target in [Target::t2_straight_edge(), Target::t5_thin_stroke(), Target::t6_glyph_like()] {
        let field = rendered(&target);
        let side_levels = max_levels(field.width, field.height);
        let coefficients = forward(Transform::Cdf97, &field, side_levels).expect("forward");
        let back = inverse(&coefficients).expect("inverse");
        let error = max_abs_difference(&field, &back);
        assert!(
            error < 1.0e-12,
            "cdf 9/7 round trip on the rendered target {} lost {error}",
            target.name
        );
        checked += 1;
    }
    assert_eq!(checked, 8, "the round-trip scan must not pass by looking at nothing");
}

#[test]
fn keeping_every_biorthogonal_coefficient_is_bit_for_bit_the_untruncated_inverse() {
    let levels = max_levels(SIDE, SIDE);
    let field = step_field(SIDE, SIDE);
    let coefficients = forward(Transform::Cdf97, &field, levels).expect("forward");
    let weights = measure(Transform::Cdf97, SIDE, SIDE, levels).expect("weights");
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
fn a_constant_field_leaves_the_constant_in_the_approximation_and_nothing_anywhere_else() {
    let levels = max_levels(SIDE, SIDE);
    let field = constant_field(SIDE, SIDE, 0.37);
    let coefficients = forward(Transform::Cdf97, &field, levels).expect("forward");
    let approximation = Band { level: levels, subband: Subband::Approximation };
    let mut worst_detail = 0.0f64;
    for y in 0..SIDE {
        for x in 0..SIDE {
            if band_of(SIDE, SIDE, levels, x, y) != approximation {
                for channel in 0..CHANNELS {
                    worst_detail = worst_detail.max(coefficients.get(channel, x, y).abs());
                }
            }
        }
    }
    println!("cdf97 constant-field worst detail coefficient: {worst_detail:e}");
    assert!(
        worst_detail < 1.0e-12,
        "the second lifting step is what cancels the detail of a constant field, and it left \
         {worst_detail} behind"
    );
    let coarse = coefficients.get(0, 0, 0);
    assert!(
        (coarse - 0.37).abs() < 1.0e-12,
        "this normalisation gives the analysis low-pass a DC gain of exactly one, so the \
         coarse coefficient of a constant field is the constant: expected 0.37, got {coarse}"
    );
    let mut detail_energy = 0.0f64;
    for band in bands(levels) {
        if band != approximation {
            detail_energy += coefficients.band_energy(band);
        }
    }
    assert!(detail_energy / energy(&field) < 1.0e-24);
}

#[test]
fn the_biorthogonal_transform_does_not_preserve_energy_and_the_deviation_is_recorded() {
    let levels = max_levels(SIDE, SIDE);
    let mut worst = 0.0f64;
    for (name, field) in procedural_fields(SIDE, SIDE) {
        let biorthogonal = forward(Transform::Cdf97, &field, levels).expect("forward");
        let orthonormal = forward(Transform::Haar, &field, levels).expect("forward");
        let field_energy = energy(&field);
        let biorthogonal_ratio = biorthogonal.energy() / field_energy;
        let orthonormal_ratio = orthonormal.energy() / field_energy;
        println!(
            "{name}: cdf97 carries {biorthogonal_ratio:.6} of the field energy, haar carries \
             {orthonormal_ratio:.15}"
        );
        assert!(
            (orthonormal_ratio - 1.0).abs() < 1.0e-12,
            "the orthonormal control must hold parseval, got {orthonormal_ratio}"
        );
        worst = worst.max((biorthogonal_ratio - 1.0).abs());
    }
    assert!(
        worst > 0.1,
        "cdf 9/7 is biorthogonal, so parseval must fail measurably; the largest measured \
         deviation was only {worst}, which means either the normalisation or this instrument \
         is wrong"
    );
    println!("largest measured cdf97 parseval deviation: {worst}");
}

#[test]
fn the_subband_weights_are_not_all_one_and_the_coarse_band_is_the_heaviest() {
    let levels = max_levels(SIDE, SIDE);
    let weights = measure(Transform::Cdf97, SIDE, SIDE, levels).expect("weights");
    for (band, norm) in weights.entries.iter() {
        println!("cdf97 weight level {} {:<14} {norm:.6}", band.level, band.subband.name());
    }
    let approximation = weights.of(Band { level: levels, subband: Subband::Approximation });
    let finest = weights.of(Band { level: 1, subband: Subband::DetailXY });
    assert!(
        approximation / finest > 10.0,
        "the coarse synthesis functions are {approximation} and the finest {finest}: a spread \
         under a factor of ten would make the weighting decorative"
    );
    assert!(
        (weights.largest() - approximation).abs() < 1.0e-12,
        "the widest synthesis function must be the coarsest one"
    );
    assert!(
        finest < 1.0,
        "the finest diagonal synthesis function has norm {finest}, so raw magnitudes would \
         over-rank it rather than under-rank it and the sign of the correction is wrong"
    );
}

#[test]
fn the_subband_weighting_is_load_bearing_and_the_gap_is_measured() {
    let field = noise_field(SMALL, SMALL, 1305);
    let terms = CHANNELS * (SMALL as usize) * (SMALL as usize);
    let steps = ladder(terms);
    let weighted = curve(&field, SMALL_LEVELS, Ranking::Weighted, &steps);
    let raw = curve(&field, SMALL_LEVELS, Ranking::RawMagnitude, &steps);
    let mut best_ratio = 1.0f64;
    let mut best_n = 0usize;
    for (i, n) in steps.iter().enumerate() {
        println!("N={n:<5} weighted {:.6}  raw {:.6}", weighted[i], raw[i]);
        assert!(
            weighted[i] <= raw[i] + 1.0e-12,
            "at N={n} the raw ranking beat the weighted one ({} against {}); the ladder \
             dominance this arm relies on does not hold",
            raw[i],
            weighted[i]
        );
        if raw[i] > 0.0 && weighted[i] / raw[i] < best_ratio {
            best_ratio = weighted[i] / raw[i];
            best_n = *n;
        }
    }
    println!("largest gap: weighted/raw = {best_ratio:.4} at N={best_n}");
    assert!(
        best_ratio < 0.7,
        "the weighted ranking never beat the raw one by more than a factor of {best_ratio}, \
         so this content cannot show whether the weighting matters"
    );
    assert!(
        weighted[weighted.len() - 1] < 1.0e-12 && raw[raw.len() - 1] < 1.0e-12,
        "both rankings must still reconstruct exactly at the full term count"
    );
}

#[test]
fn neither_cdf97_ranking_gives_a_monotone_curve_and_both_counts_are_recorded() {
    let field = noise_field(SMALL, SMALL, 99);
    let coefficients = forward(Transform::Cdf97, &field, SMALL_LEVELS).expect("forward");
    let weights = measure(Transform::Cdf97, SMALL, SMALL, SMALL_LEVELS).expect("weights");
    let terms = coefficients.terms();
    let mut counts = [0usize; 2];
    let mut worst = [0.0f64; 2];
    for (slot, ranking) in [Ranking::Weighted, Ranking::RawMagnitude].iter().enumerate() {
        let mut previous = f64::INFINITY;
        for n in 0..=terms {
            let kept = keep_largest(&coefficients, &weights, *ranking, n).expect("keep");
            let reconstruction = inverse(&kept).expect("inverse");
            let error = relative_l2(&field, &reconstruction);
            if error > previous + 1.0e-14 {
                counts[slot] += 1;
                worst[slot] = worst[slot].max(error - previous);
            }
            previous = error;
        }
        println!(
            "cdf97 {}: {} rises over {} steps, worst rise {:e}",
            ranking.name(),
            counts[slot],
            terms,
            worst[slot]
        );
    }
    assert!(
        counts[0] > 0,
        "the weighted cdf 9/7 curve came out monotone. The specification predicted that, this \
         run measured the opposite on four fields, and a change of answer must be investigated \
         rather than absorbed: no per-coefficient ranking is exactly optimal in a basis whose \
         synthesis functions are not mutually orthogonal"
    );
    assert!(
        counts[1] > counts[0],
        "the raw ranking rose {} times against the weighted ranking's {}: the weighting is \
         supposed to reduce, not eliminate, the non-monotonicity",
        counts[1],
        counts[0]
    );
}

#[test]
fn a_linear_ramp_leaves_zero_detail_away_from_the_mirrored_boundary() {
    let levels = 1;
    let field = steep_ramp_field(SMALL, SMALL);
    let coefficients = forward(Transform::Cdf97, &field, levels).expect("forward");
    let approximation = Band { level: levels, subband: Subband::Approximation };
    let mut worst = 0.0f64;
    let mut vanished = 0usize;
    let mut total = 0usize;
    for y in 0..SMALL {
        for x in 0..SMALL {
            if band_of(SMALL, SMALL, levels, x, y) == approximation {
                continue;
            }
            for channel in 0..CHANNELS {
                let v = coefficients.get(channel, x, y).abs();
                worst = worst.max(v);
                total += 1;
                if v < 1.0e-12 {
                    vanished += 1;
                }
            }
        }
    }
    println!("cdf97 ramp: worst detail {worst:e}, {vanished} of {total} details vanish");
    assert!(
        worst < 0.08,
        "the 9/7 predictor annihilates a linear signal, and a whole-point symmetric extension \
         leaves only the mirror kink at the border; a detail of {worst} is the signature of a \
         periodic wrap-around instead"
    );
    assert!(
        vanished * 2 > total,
        "only {vanished} of {total} detail coefficients vanished on a linear ramp, so the two \
         vanishing moments are not there"
    );
}

#[test]
fn a_second_biorthogonal_level_only_touches_the_coarse_quadrant() {
    let field = smooth_field(SIDE, SIDE);
    let one = forward(Transform::Cdf97, &field, 1).expect("forward");
    let two = forward(Transform::Cdf97, &field, 2).expect("forward");
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
    assert!(inside_changed > (CHANNELS * ((half * half) as usize)) / 2);
}

#[test]
fn the_lifting_steps_invert_in_the_reverse_order() {
    let mut rng = Lcg::new(5150);
    let mut buffer = Vec::new();
    for _ in 0..32 {
        buffer.push(rng.signed());
    }
    let original = buffer.clone();
    cdf97::forward(&mut buffer);
    let transformed = buffer.clone();
    cdf97::inverse(&mut buffer);
    let mut worst = 0.0f64;
    for (a, b) in original.iter().zip(buffer.iter()) {
        worst = worst.max((a - b).abs());
    }
    assert!(worst < 1.0e-12, "the 1D lifting round trip lost {worst}");
    assert!(
        transformed != original,
        "the transform left the buffer untouched, so this round trip proves nothing"
    );

    let mut flat = vec![0.44f64; 16];
    cdf97::forward(&mut flat);
    for (i, v) in flat.iter().enumerate() {
        if i < 8 {
            assert!((v - 0.44).abs() < 1.0e-12, "approximation {i} is {v}");
        } else {
            assert!(v.abs() < 1.0e-12, "detail {i} is {v}");
        }
    }
}

#[test]
fn the_boundary_is_a_whole_point_mirror() {
    assert_eq!(reflect(0, 8), 0);
    assert_eq!(reflect(7, 8), 7);
    assert_eq!(reflect(-1, 8), 1);
    assert_eq!(reflect(-3, 8), 3);
    assert_eq!(reflect(8, 8), 6);
    assert_eq!(reflect(9, 8), 5);
    assert_eq!(reflect(13, 8), 1);
    assert_eq!(reflect(14, 8), 0);
    for i in 0..8isize {
        assert_eq!(reflect(i, 8), i as usize);
    }
    assert_ne!(reflect(8, 8), 0);
    assert_ne!(reflect(-1, 8), 7);
    let coefficients = [ALPHA, BETA, GAMMA, DELTA, K];
    assert!((1.0 + 2.0 * ALPHA + 2.0 * GAMMA * (1.0 + 2.0 * BETA * (1.0 + 2.0 * ALPHA))).abs() < 1.0e-9,
        "the two predict steps together must annihilate a constant, which is what makes the \
         detail of a flat field zero");
    assert!((K - 1.230174104914001).abs() < 1.0e-15);
    for c in coefficients.iter() {
        assert!(c.is_finite());
    }
}

#[test]
fn the_biorthogonal_convenience_path_agrees_with_the_three_steps_it_replaces() {
    let levels = max_levels(SMALL, SMALL);
    let field = step_field(SMALL, SMALL);
    let coefficients = forward(Transform::Cdf97, &field, levels).expect("forward");
    let weights = measure(Transform::Cdf97, SMALL, SMALL, levels).expect("weights");
    let kept = keep_largest(&coefficients, &weights, Ranking::Weighted, 40).expect("keep");
    let stepwise = inverse(&kept).expect("inverse");
    let direct =
        truncate_and_reconstruct(Transform::Cdf97, &field, levels, 40, Ranking::Weighted)
            .expect("truncate");
    for (a, b) in stepwise.pixels.iter().zip(direct.pixels.iter()) {
        assert_eq!(a.x.to_bits(), b.x.to_bits());
    }
    assert_eq!(kept.nonzero(), 40);
}
