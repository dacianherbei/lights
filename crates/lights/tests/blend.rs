use lights::blend::{Additive, BlendMode, BlendSettings, EarlyOut, Over, Step};
use lights::numerics::vector::Rgb;

fn no_early_out() -> BlendSettings<f64> {
    BlendSettings {
        transmittance_cutoff: 0.0,
        ..BlendSettings::<f64>::contract()
    }
}

fn composite_over(
    order: &[(f64, Rgb<f64>)],
    background: Rgb<f64>,
    settings: &BlendSettings<f64>,
) -> (Rgb<f64>, f64, usize) {
    let mut acc = <Over as BlendMode<f64>>::init();
    let mut consumed = 0usize;
    for (alpha, colour) in order.iter() {
        consumed += 1;
        if let Step::Stop = <Over as BlendMode<f64>>::step(&mut acc, 1.0, *colour, *alpha, settings)
        {
            break;
        }
    }
    (
        <Over as BlendMode<f64>>::resolve(&acc, background),
        acc.transmittance,
        consumed,
    )
}

#[test]
fn tv_s15_two_atom_order_and_background_weighting() {
    let red = Rgb::new(1.0, 0.0, 0.0);
    let blue = Rgb::new(0.0, 0.0, 1.0);
    let s = BlendSettings::<f64>::reference();

    let (front_red, t, _) = composite_over(&[(0.5, red), (0.5, blue)], Rgb::zero(), &s);
    assert!((front_red.x - 0.5).abs() < 1e-15, "{front_red:?}");
    assert!(front_red.y.abs() < 1e-15);
    assert!((front_red.z - 0.25).abs() < 1e-15);
    assert!((t - 0.25).abs() < 1e-15);

    let (front_blue, t2, _) = composite_over(&[(0.5, blue), (0.5, red)], Rgb::zero(), &s);
    assert!((front_blue.x - 0.25).abs() < 1e-15);
    assert!((front_blue.z - 0.5).abs() < 1e-15);
    assert!((t2 - 0.25).abs() < 1e-15);

    let white = Rgb::new(1.0, 1.0, 1.0);
    let (wa, _, _) = composite_over(&[(0.5, red), (0.5, blue)], white, &s);
    assert!((wa.x - 0.75).abs() < 1e-15 && (wa.y - 0.25).abs() < 1e-15 && (wa.z - 0.5).abs() < 1e-15);
    let (wb, _, _) = composite_over(&[(0.5, blue), (0.5, red)], white, &s);
    assert!((wb.x - 0.5).abs() < 1e-15 && (wb.y - 0.25).abs() < 1e-15 && (wb.z - 0.75).abs() < 1e-15);
    assert!(
        (wa.y - 0.25).abs() < 1e-15 && (wb.y - 0.25).abs() < 1e-15,
        "the green channel equals T_final in BOTH orders: the background is \
         weighted by leftover transmittance, not added unconditionally"
    );
}

#[test]
fn tv_s16_order_spread_with_identical_transmittance() {
    let grey = |v: f64| Rgb::new(v, v, v);
    let atoms = [(0.6, grey(0.1)), (0.4, grey(0.5)), (0.9, grey(0.9))];
    let s = no_early_out();

    let orders: [[usize; 3]; 3] = [[0, 1, 2], [2, 1, 0], [1, 0, 2]];
    let expected = [0.3344, 0.8336, 0.4304];
    let mut transmittances = Vec::new();
    for (order, want) in orders.iter().zip(expected.iter()) {
        let list: Vec<(f64, Rgb<f64>)> = order.iter().map(|i| atoms[*i]).collect();
        let (c, t, _) = composite_over(&list, Rgb::zero(), &s);
        assert!((c.x - want).abs() < 1e-12, "order {order:?}: {} vs {want}", c.x);
        transmittances.push(t);
    }
    for t in transmittances.iter() {
        assert!(
            (t - 0.024).abs() < 1e-15,
            "T is a PRODUCT and therefore ORDER-INVARIANT: {t}"
        );
    }
    assert!(
        expected[1] / expected[0] > 2.4,
        "a 2.5x colour spread with a bit-identical transmittance means the \
         alpha/coverage channel CANNOT detect a sort bug"
    );
}

#[test]
fn tv_s16b_exact_swap_identity() {
    let s = no_early_out();
    let cases = [
        (0.3, 0.7, Rgb::new(0.9, 0.1, 0.2), Rgb::new(0.1, 0.8, 0.4)),
        (0.55, 0.21, Rgb::new(0.2, 0.2, 0.9), Rgb::new(0.7, 0.0, 0.1)),
        (0.9, 0.45, Rgb::new(0.05, 0.6, 0.3), Rgb::new(0.95, 0.4, 0.7)),
    ];
    for (ai, aj, ci, cj) in cases {
        let (correct, _, _) = composite_over(&[(ai, ci), (aj, cj)], Rgb::zero(), &s);
        let (swapped, _, _) = composite_over(&[(aj, cj), (ai, ci)], Rgb::zero(), &s);
        let difference = correct - swapped;
        let predicted = (ci - cj) * (1.0 * ai * aj);
        assert!((difference.x - predicted.x).abs() < 1e-14);
        assert!((difference.y - predicted.y).abs() < 1e-14);
        assert!((difference.z - predicted.z).abs() < 1e-14);
    }

    let uniform = Rgb::new(0.4, 0.4, 0.4);
    let (a, _, _) = composite_over(&[(0.3, uniform), (0.7, uniform)], Rgb::zero(), &s);
    let (b, _, _) = composite_over(&[(0.7, uniform), (0.3, uniform)], Rgb::zero(), &s);
    assert!(
        (a - b).length() < 1e-15,
        "the swap error is LINEAR in the colour difference, so it vanishes on a \
         uniformly-coloured object and is maximal at silhouettes"
    );
}

#[test]
fn tv_s17_early_out_exact_stops_at_fourteen() {
    let list: Vec<(f64, Rgb<f64>)> = (0..40).map(|_| (0.5, Rgb::new(1.0, 1.0, 1.0))).collect();
    let exact = BlendSettings::<f64>::contract();
    assert_eq!(exact.early_out, EarlyOut::Exact);
    let (truncated, t, consumed) = composite_over(&list, Rgb::zero(), &exact);
    assert_eq!(consumed, 14, "exact variant must stop at exactly n = 14");
    assert!(
        (t - 6.103515625e-05).abs() < 1e-18,
        "T at termination {t}"
    );

    let (full, _, consumed_full) = composite_over(&list, Rgb::zero(), &no_early_out());
    assert_eq!(consumed_full, 40);
    let tail = full - truncated;
    let per_channel = tail.x.abs().max(tail.y.abs()).max(tail.z.abs());
    assert!(
        per_channel <= 6.103515625e-05,
        "the neglected tail is bounded PER CHANNEL by T_n * c_max, not by the \
         3-channel vector length: {per_channel}"
    );
}

#[test]
fn tv_s17b_reference_versus_exact_bounds() {
    let white = Rgb::new(1.0, 1.0, 1.0);
    let mut list = vec![(0.9, white), (0.901, white)];
    for _ in 0..8 {
        list.push((0.99, white));
    }

    let (truth, _, _) = composite_over(&list, Rgb::zero(), &no_early_out());
    let reference = BlendSettings::<f64>::reference();
    let exact = BlendSettings::<f64>::contract();
    let (r, _, _) = composite_over(&list, Rgb::zero(), &reference);
    let (e, _, _) = composite_over(&list, Rgb::zero(), &exact);

    let reference_error = (truth - r).max_component();
    let exact_error = (truth - e).max_component();

    assert!((reference.error_ceiling() - 1.0e-2).abs() < 1e-12);
    assert!((exact.error_ceiling() - 1.0e-4).abs() < 1e-18);
    assert!(
        reference_error <= reference.error_ceiling(),
        "reference error {reference_error} exceeded its own bound"
    );
    assert!(
        reference_error > 9.0e-3,
        "this list is built to approach the reference bound: {reference_error}"
    );
    assert!(
        exact_error <= exact.error_ceiling(),
        "exact error {exact_error} exceeded eps_T"
    );
    assert!(
        reference_error > 90.0 * exact_error,
        "the exact variant is provably about 100x tighter: {reference_error} vs \
         {exact_error}"
    );
}

#[test]
fn alpha_cap_applies_to_alpha_not_opacity() {
    let s = BlendSettings::<f64>::reference();
    let mut acc = <Over as BlendMode<f64>>::init();
    let colour = Rgb::new(1.0, 0.5, 0.25);
    let step = <Over as BlendMode<f64>>::step(&mut acc, 1.0, colour, 1.0, &s);
    assert_eq!(step, Step::Continue);
    assert!(
        (acc.rgb.x - 0.99).abs() < 1e-15,
        "the composited alpha is capped at 0.99 EXACTLY, not 1.0: {}",
        acc.rgb.x
    );
    assert!((acc.rgb.y - 0.495).abs() < 1e-15);
    assert!((acc.transmittance - 0.01).abs() < 1e-15);

    let mut below = <Over as BlendMode<f64>>::init();
    let tiny = <Over as BlendMode<f64>>::step(&mut below, 0.001, colour, 1.0, &s);
    assert_eq!(tiny, Step::Continue);
    assert_eq!(below.rgb, Rgb::zero(), "alpha below 1/255 skips the pixel");
    assert_eq!(below.transmittance, 1.0);

    let mut capped_then_cut = <Over as BlendMode<f64>>::init();
    <Over as BlendMode<f64>>::step(&mut capped_then_cut, 1.0, colour, 0.003, &s);
    assert_eq!(
        capped_then_cut.rgb,
        Rgb::zero(),
        "the cap fires BEFORE the alpha_cutoff test"
    );

    let mut partial = <Over as BlendMode<f64>>::init();
    <Over as BlendMode<f64>>::step(&mut partial, 0.5, Rgb::new(1.0, 1.0, 1.0), 1.0, &s);
    assert!(
        (partial.rgb.x - 0.5).abs() < 1e-15,
        "alpha = min(cap, o*weight) = 0.5 here; CAPPING THE OPACITY FIRST would \
         give min(0.99, 1.0)*0.5 = 0.495 and change every pixel of a \
         high-opacity atom rather than only its core. got {}",
        partial.rgb.x
    );
    assert!((partial.transmittance - 0.5).abs() < 1e-15);

    let mut deep_core = <Over as BlendMode<f64>>::init();
    <Over as BlendMode<f64>>::step(&mut deep_core, 0.999, Rgb::new(1.0, 1.0, 1.0), 1.0, &s);
    assert!(
        (deep_core.rgb.x - 0.99).abs() < 1e-15,
        "the cap bites only in the core: {}",
        deep_core.rgb.x
    );
}

#[test]
fn additive_is_permutation_invariant() {
    assert_eq!(<Additive as BlendMode<f64>>::ORDER_DEPENDENT, false);
    assert_eq!(<Additive as BlendMode<f64>>::SIGNED, true);
    assert_eq!(<Over as BlendMode<f64>>::ORDER_DEPENDENT, true);
    assert_eq!(<Over as BlendMode<f64>>::SIGNED, false);

    let mut atoms: Vec<(f64, Rgb<f64>)> = Vec::new();
    let mut state: u64 = 7;
    let mut next_bits = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 40) as i64
    };
    let dyadic = |n: i64| (n % 512 - 256) as f64 / 256.0;
    for _ in 0..32 {
        let o = dyadic(next_bits());
        atoms.push((
            o,
            Rgb::new(
                dyadic(next_bits()),
                dyadic(next_bits()),
                dyadic(next_bits()),
            ),
        ));
    }
    let s = BlendSettings::<f64>::contract();

    let accumulate = |order: &[usize]| {
        let mut acc = <Additive as BlendMode<f64>>::init();
        for i in order.iter() {
            let (o, c) = atoms[*i];
            let step = <Additive as BlendMode<f64>>::step(&mut acc, 1.0, c, o, &s);
            assert_eq!(step, Step::Continue, "additive never early-outs");
        }
        acc
    };

    let identity: Vec<usize> = (0..atoms.len()).collect();
    let baseline = accumulate(&identity);
    let mut permutation = identity.clone();
    let mut shuffle_state: u64 = 12345;
    for _ in 0..1000 {
        for i in (1..permutation.len()).rev() {
            shuffle_state = shuffle_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            let j = (shuffle_state >> 33) as usize % (i + 1);
            permutation.swap(i, j);
        }
        let acc = accumulate(&permutation);
        assert_eq!(
            acc.x.to_bits(),
            baseline.x.to_bits(),
            "additive accumulation of exactly-representable values must be \
             bit-identical under permutation"
        );
        assert_eq!(acc.y.to_bits(), baseline.y.to_bits());
        assert_eq!(acc.z.to_bits(), baseline.z.to_bits());
    }

    let mut rough: Vec<(f64, Rgb<f64>)> = Vec::new();
    let mut rough_state: u64 = 99;
    let mut unit = || {
        rough_state = rough_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((rough_state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    for _ in 0..32 {
        let o = unit() * 2.0 - 1.0;
        rough.push((o, Rgb::new(unit(), unit(), unit())));
    }
    let accumulate_rough = |order: &[usize]| {
        let mut acc = <Additive as BlendMode<f64>>::init();
        for i in order.iter() {
            let (o, c) = rough[*i];
            <Additive as BlendMode<f64>>::step(&mut acc, 1.0, c, o, &s);
        }
        acc
    };
    let rough_baseline = accumulate_rough(&identity);
    let mut rough_permutation = identity.clone();
    let mut worst_ulp_drift: f64 = 0.0;
    for _ in 0..1000 {
        for i in (1..rough_permutation.len()).rev() {
            shuffle_state = shuffle_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            let j = (shuffle_state >> 33) as usize % (i + 1);
            rough_permutation.swap(i, j);
        }
        let acc = accumulate_rough(&rough_permutation);
        let drift = (acc - rough_baseline).max_component().abs();
        worst_ulp_drift = worst_ulp_drift.max(drift);
    }
    assert!(
        worst_ulp_drift < 1e-14,
        "additive accumulation is mathematically order-independent: {worst_ulp_drift}"
    );
    assert!(
        worst_ulp_drift > 0.0,
        "FLOAT ADDITION IS NOT ASSOCIATIVE. Order independence buys freedom from \
         the SORT, not bit-identity under an arbitrary visit order; determinism \
         still comes from visiting atoms in a fixed order. Measured drift over \
         1000 permutations of 32 non-dyadic atoms: one ULP."
    );

    assert!(
        !<Additive as BlendMode<f64>>::saturated(&baseline, &s),
        "an additive accumulator has no transmittance and can never saturate"
    );
    assert_eq!(
        <Additive as BlendMode<f64>>::resolve(&baseline, Rgb::zero()),
        baseline
    );
}

#[test]
fn over_accumulator_carries_transmittance_and_coverage() {
    let s = no_early_out();
    let mut acc = <Over as BlendMode<f64>>::init();
    assert_eq!(acc.transmittance, 1.0);
    assert_eq!(<Over as BlendMode<f64>>::coverage(&acc), 0.0);
    <Over as BlendMode<f64>>::step(&mut acc, 1.0, Rgb::new(1.0, 1.0, 1.0), 0.75, &s);
    assert!((<Over as BlendMode<f64>>::coverage(&acc) - 0.75).abs() < 1e-15);
    let saturating = BlendSettings::<f64>::contract();
    assert!(!<Over as BlendMode<f64>>::saturated(&acc, &saturating));
    let deep = lights::blend::OverAccum {
        rgb: Rgb::zero(),
        transmittance: 1.0e-9,
    };
    assert!(<Over as BlendMode<f64>>::saturated(&deep, &saturating));
}

#[test]
fn blend_settings_convert_between_precisions() {
    let wide = BlendSettings::<f64>::reference();
    let narrow = BlendSettings::<f32>::from_f64(&wide);
    assert_eq!(narrow.early_out, EarlyOut::Reference);
    assert!((narrow.alpha_cap as f64 - 0.99).abs() < 1e-7);
    let back = narrow.to_f64();
    assert!((back.transmittance_cutoff - 1.0e-4).abs() < 1e-10);
}

#[test]
fn tv_s12_the_cap_is_applied_before_the_cutoff_test() {
    let pathological = BlendSettings::<f64> {
        alpha_cap: 0.002,
        alpha_cutoff: 0.004,
        transmittance_cutoff: 0.0,
        early_out: EarlyOut::Exact,
    };
    let mut acc = <Over as BlendMode<f64>>::init();
    let step = <Over as BlendMode<f64>>::step(
        &mut acc,
        1.0,
        Rgb::new(1.0, 1.0, 1.0),
        1.0,
        &pathological,
    );
    assert_eq!(step, Step::Continue);
    assert_eq!(
        acc.rgb,
        Rgb::zero(),
        "alpha = min(cap, o*w) = 0.002 is BELOW the 0.004 cutoff, so the pixel is \
         skipped. Testing the RAW o*w = 1.0 against the cutoff first would \
         composite it. At the shipped settings (cap 0.99 > cutoff 1/255) the two \
         orderings are indistinguishable, which is why this test has to reach for \
         a configuration where cap < cutoff to pin the ordering at all."
    );
    assert_eq!(acc.transmittance, 1.0);

    let shipped = BlendSettings::<f64>::reference();
    assert!(
        shipped.alpha_cap > shipped.alpha_cutoff,
        "recording WHY the ordering is unobservable in production"
    );
}
