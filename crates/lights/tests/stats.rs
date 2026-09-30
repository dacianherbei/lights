use lights::stats::{CI95_Z, Stat, intervals_overlap, stat};

fn hand_computed_ci95(samples: &[f64]) -> f64 {
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let mut sum = 0.0f64;
    for s in samples.iter() {
        sum += (s - mean) * (s - mean);
    }
    let variance = sum / (n - 1.0);
    1.96 * variance.sqrt() / n.sqrt()
}

#[test]
fn a_constant_sample_has_no_interval_at_all() {
    let s = stat(&[1.0, 1.0, 1.0, 1.0, 1.0]);
    assert_eq!(s.mean, 1.0);
    assert_eq!(s.min, 1.0);
    assert_eq!(
        s.ci95, 0.0,
        "five identical measurements carry no uncertainty, and a non-zero \
         interval here would widen every verdict the benchmark prints"
    );
}

#[test]
fn the_interval_is_the_normal_quantile_over_the_sample_standard_deviation() {
    let samples = [102.0f64, 97.0, 110.0, 99.0, 101.0];
    let s = stat(&samples);
    let expected_mean = (102.0 + 97.0 + 110.0 + 99.0 + 101.0) / 5.0;
    assert!((s.mean - expected_mean).abs() < 1e-12, "{} vs {expected_mean}", s.mean);
    let expected = hand_computed_ci95(&samples);
    assert!(
        (s.ci95 - expected).abs() < 1e-12,
        "ci95 is {} and the hand computation with the n-1 denominator gives \
         {expected}. Dividing by n instead understates the interval by a factor \
         sqrt(4/5) at five samples — 10.6% narrower — which is the same order as \
         the effects this decides",
        s.ci95
    );
    assert!(
        (CI95_Z - 1.96).abs() < 1e-12,
        "CI95_Z is the normal quantile the published verdicts were computed with"
    );
}

#[test]
fn min_is_the_minimum_and_not_the_first_sample() {
    let s = stat(&[9.0, 3.0, 7.0]);
    assert_eq!(s.min, 3.0, "the min column is what a reader reaches for when the mean looks noisy");
    let s = stat(&[3.0, 9.0, 7.0]);
    assert_eq!(s.min, 3.0);
}

#[test]
fn one_sample_and_no_samples_are_answered_rather_than_poisoned() {
    let one = stat(&[4.5]);
    assert_eq!(one.mean, 4.5);
    assert_eq!(one.min, 4.5);
    assert_eq!(one.ci95, 0.0, "a single measurement has no interval, and nan would poison every comparison it enters");
    let none = stat(&[]);
    assert_eq!(none, Stat::default());
    assert!(
        intervals_overlap(none, none),
        "a degenerate stat must compare equal to itself rather than name a winner"
    );
}

#[test]
fn intervals_that_merely_touch_are_too_close_to_call() {
    let a = Stat { mean: 10.0, ci95: 1.0, min: 9.0 };
    let touching = Stat { mean: 12.0, ci95: 1.0, min: 11.0 };
    assert!(
        intervals_overlap(a, touching),
        "10+1 and 12-1 meet exactly at 11. A strict < would name a winner on the \
         boundary, which is the direction that makes the instrument flatter its \
         own conclusions"
    );
    assert!(intervals_overlap(touching, a), "the predicate must be symmetric");

    let apart = Stat {
        mean: f64::from_bits((12.0f64).to_bits() + 1),
        ci95: 1.0,
        min: 11.0,
    };
    assert!(
        !intervals_overlap(a, apart),
        "one ULP past touching is a separation, and the benchmark may name a \
         winner there"
    );
    assert!(!intervals_overlap(apart, a), "and it must be symmetric there too");
}

#[test]
fn a_wide_interval_swallows_a_narrow_one_in_both_directions() {
    let wide = Stat { mean: 50.0, ci95: 40.0, min: 10.0 };
    let narrow = Stat { mean: 55.0, ci95: 0.1, min: 54.9 };
    assert!(intervals_overlap(wide, narrow));
    assert!(intervals_overlap(narrow, wide));
    let outside = Stat { mean: 200.0, ci95: 0.1, min: 199.9 };
    assert!(!intervals_overlap(wide, outside));
}
