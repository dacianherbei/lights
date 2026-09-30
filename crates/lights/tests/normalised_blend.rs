use lights::blend::{BlendMode, BlendSettings, Normalised, NormalisedAccum, Over, Step};
use lights::dispatch::{Collect, Precision, RenderRequest, render};
use lights::kernel::KernelFamily;
use lights::method::RenderMethod;
use lights::method::splat::SplatConfig;
use lights::numerics::vector::{Rgb, Vec3};
use lights::scene::demo::orthographic_camera;
use lights::scene::{Atom, Scene};
use lights::settings::{MIN_SCALE, SharedSettings};

const LN_NINE: f64 = 2.1972245773362196;

fn no_floor() -> BlendSettings<f64> {
    BlendSettings {
        alpha_cutoff: 0.0,
        ..BlendSettings::<f64>::contract()
    }
}

fn with_floor(alpha_cutoff: f64) -> BlendSettings<f64> {
    BlendSettings {
        alpha_cutoff,
        ..BlendSettings::<f64>::contract()
    }
}

fn accumulate(
    contributions: &[(f64, Rgb<f64>)],
    opacity: f64,
    s: &BlendSettings<f64>,
) -> NormalisedAccum<f64> {
    let mut acc = <Normalised as BlendMode<f64>>::init();
    for (weight, colour) in contributions.iter() {
        let step = <Normalised as BlendMode<f64>>::step(&mut acc, *weight, *colour, opacity, s);
        assert_eq!(
            step,
            Step::Continue,
            "a normalised accumulator has no transmittance and therefore no \
             argument for stopping: every candidate must be consumed"
        );
    }
    acc
}

fn resolve(acc: &NormalisedAccum<f64>, background: Rgb<f64>) -> Rgb<f64> {
    <Normalised as BlendMode<f64>>::resolve(acc, background)
}

fn held(weight: f64, colour: Rgb<f64>) -> NormalisedAccum<f64> {
    NormalisedAccum {
        weighted: colour * weight,
        weight,
    }
}

fn gaussian_weight(x: f64, centre: f64, sigma: f64) -> f64 {
    let u = (x - centre) / sigma;
    (-0.5 * u * u).exp()
}

fn edge_pair(x: f64, sigma: f64, separation: f64) -> [(f64, Rgb<f64>); 2] {
    let paper = Rgb::new(0.0, 0.0, 0.0);
    let ink = Rgb::new(1.0, 1.0, 1.0);
    [
        (gaussian_weight(x, -0.5 * separation, sigma), paper),
        (gaussian_weight(x, 0.5 * separation, sigma), ink),
    ]
}

fn logistic(x: f64, sigma: f64, separation: f64) -> f64 {
    1.0 / (1.0 + (-x * separation / (sigma * sigma)).exp())
}

fn ink_fraction(x: f64, sigma: f64, separation: f64, opacity: f64, s: &BlendSettings<f64>) -> Option<f64> {
    let acc = accumulate(&edge_pair(x, sigma, separation), opacity, s);
    if acc.weight > 0.0 {
        Some((acc.weighted * acc.weight.recip()).x)
    } else {
        None
    }
}

fn transition_width(sigma: f64, separation: f64, s: &BlendSettings<f64>) -> f64 {
    let span = 8.0 * sigma + 4.0 * separation;
    let samples = 400_001usize;
    let step = 2.0 * span / ((samples - 1) as f64);
    let mut previous: Option<(f64, f64)> = None;
    let mut low_crossing: Option<f64> = None;
    let mut high_crossing: Option<f64> = None;
    for i in 0..samples {
        let x = -span + (i as f64) * step;
        let t = match ink_fraction(x, sigma, separation, 1.0, s) {
            Some(v) => v,
            None => continue,
        };
        if let Some((px, pt)) = previous {
            if low_crossing.is_none() && pt < 0.1 && t >= 0.1 {
                low_crossing = Some(px + (x - px) * (0.1 - pt) / (t - pt));
            }
            if high_crossing.is_none() && pt < 0.9 && t >= 0.9 {
                high_crossing = Some(px + (x - px) * (0.9 - pt) / (t - pt));
            }
        }
        previous = Some((x, t));
    }
    match (low_crossing, high_crossing) {
        (Some(lo), Some(hi)) => hi - lo,
        _ => panic!(
            "the profile never crossed both 0.1 and 0.9 for sigma {sigma} and \
             separation {separation}: the instrument measured nothing"
        ),
    }
}

fn stack_scene(count: usize, opacity: f64) -> Scene<f64> {
    let mut scene = Scene::<f64>::new();
    for i in 0..count {
        let atom = Atom::isotropic(
            Vec3::new(0.0, 0.0, 4.0 + (i as f64) * 0.1),
            0.2,
            opacity,
            Rgb::new(1.0, 1.0, 1.0),
        );
        match scene.push(atom, MIN_SCALE) {
            Ok(()) => (),
            Err(e) => panic!("{e:?}"),
        }
    }
    scene
}

#[cfg(feature = "method-splat")]
fn render_stack(kernel: KernelFamily, scene: Scene<f64>) -> lights::dispatch::RenderOutcome {
    let mut settings = SharedSettings::<f64>::contract();
    settings.threads = 1;
    let request = RenderRequest {
        precision: Precision::F64,
        kernel,
        method: RenderMethod::SplatRaster(SplatConfig::contract()),
        settings,
        camera: orthographic_camera(32, 32, 1.0),
        scene,
        collect: Collect::PixelsAndCounters,
        shapes: None,
    };
    match render(&request) {
        Ok(o) => o,
        Err(e) => panic!("render failed: {e:?}"),
    }
}

#[test]
fn partition_of_unity_reproduces_a_colour_every_atom_agrees_on() {
    let s = no_floor();
    let colour = Rgb::new(0.5, 0.25, 0.75);
    let agreed: Vec<(f64, Rgb<f64>)> = (0..4).map(|_| (0.5, colour)).collect();
    let acc = accumulate(&agreed, 1.0, &s);
    assert_eq!(acc.weight, 2.0);

    for background in [Rgb::zero(), Rgb::new(1.0, 1.0, 1.0), Rgb::new(0.25, 0.5, 0.125)] {
        let out = resolve(&acc, background);
        assert_eq!(
            out.x.to_bits(),
            colour.x.to_bits(),
            "a partition of unity reproduces a colour every covering atom agrees \
             on EXACTLY, and does so independently of the background: got {out:?} \
             against background {background:?}"
        );
        assert_eq!(out.y.to_bits(), colour.y.to_bits());
        assert_eq!(out.z.to_bits(), colour.z.to_bits());
    }

    let uneven = [
        (0.125, colour),
        (1.0, colour),
        (0.375, colour),
        (0.5, colour),
    ];
    let uneven_acc = accumulate(&uneven, 1.0, &s);
    assert_eq!(uneven_acc.weight, 2.0);
    let uneven_out = resolve(&uneven_acc, Rgb::zero());
    assert!(
        (uneven_out - colour).max_component().abs() < 1e-15,
        "the reproduction does not depend on HOW the weight is distributed \
         between the agreeing atoms: {uneven_out:?}"
    );

    let mut over = <Over as BlendMode<f64>>::init();
    <Over as BlendMode<f64>>::step(&mut over, 1.0, colour, 0.5, &s);
    <Over as BlendMode<f64>>::step(&mut over, 1.0, colour, 0.5, &s);
    let over_out = <Over as BlendMode<f64>>::resolve(&over, Rgb::zero());
    assert!(
        (over_out.x - 0.75 * colour.x).abs() < 1e-15,
        "two half-opaque atoms of one colour over black give 0.75 of it under \
         alpha compositing: {over_out:?}"
    );
    assert!(
        (over_out.x - colour.x).abs() > 0.1,
        "ALPHA COMPOSITING PROVABLY CANNOT DO THIS. Over reaches the atom colour \
         only as total alpha reaches one, so the reconstruction of a flat region \
         is a function of how much opacity was deposited there. Normalisation \
         makes it a convex combination instead, and a convex combination of \
         identical colours is that colour. This is the whole reason the blend \
         rule is the first variable of the measurement and not a detail of it."
    );
}

#[test]
fn the_resolve_is_continuous_through_the_zero_over_zero_point() {
    let ink = Rgb::new(1.0, 1.0, 1.0);
    let background = Rgb::new(0.0, 0.0, 0.0);

    let empty = <Normalised as BlendMode<f64>>::init();
    assert_eq!(empty.weight, 0.0);
    let uncovered = resolve(&empty, background);
    assert_eq!(
        uncovered, background,
        "a pixel covered by no atom is 0/0 and must resolve to the background \
         rather than to a nonsense number"
    );

    let samples = 1001usize;
    let mut previous = uncovered;
    let mut worst_step: f64 = 0.0;
    for k in 0..samples {
        let weight = (k as f64) * 1.0e-15;
        let acc = held(weight, ink);
        let out = resolve(&acc, background);
        let step = (out - previous).max_component().abs();
        worst_step = worst_step.max(step);
        previous = out;
    }
    assert!(
        worst_step <= 1e-12,
        "the resolve must be continuous THROUGH the 0/0 point, not merely \
         defined at it. Worst neighbouring step across the first thousand \
         multiples of 1e-15 of weight: {worst_step}"
    );

    let mut ladder_weight = 1.0e-15;
    let contrast = (ink - background).max_component().abs();
    for _ in 0..64 {
        let acc = held(ladder_weight, ink);
        let out = resolve(&acc, background);
        let excursion = (out - background).max_component().abs();
        assert!(
            excursion <= contrast * ladder_weight.min(1.0) + 1e-15,
            "below unit weight the resolve is the coverage mix and its distance \
             from the background is bounded by coverage times contrast: weight \
             {ladder_weight} gave {excursion}"
        );
        ladder_weight *= 1.7;
    }

    let faint = held(1.0e-15, ink);
    let hard_switch = faint.weighted * faint.weight.recip();
    let amplitude = (hard_switch - uncovered).max_component().abs();
    assert!(
        amplitude > 0.99,
        "RECORDING WHAT THE REJECTED RULE WOULD HAVE COST. A hard switch \
         (weight > 0 ? mean : background) returns the FULL atom colour at a \
         pixel carrying 1e-15 of weight, so a step of amplitude {amplitude} \
         appears wherever the support ends. The measured 10-90 transition width \
         would then be set by the truncation radius rather than by the atoms, \
         which is precisely the confident wrong number this crate exists to \
         avoid. The shipped coverage mix keeps that step at {worst_step}."
    );
}

#[test]
fn the_coverage_mix_reduces_exactly_to_equation_five_where_weight_reaches_one() {
    let s = no_floor();
    let cases = [
        (vec![(0.5, Rgb::new(1.0, 0.0, 0.0)), (0.75, Rgb::new(0.0, 0.0, 1.0))], 1.0),
        (vec![(1.0, Rgb::new(0.25, 0.5, 0.75)), (2.0, Rgb::new(0.5, 0.25, 0.125))], 1.0),
        (vec![(0.3, Rgb::new(0.9, 0.1, 0.2)); 7], 1.0),
    ];
    for (contributions, opacity) in cases.iter() {
        let acc = accumulate(contributions, *opacity, &s);
        assert!(acc.weight >= 1.0, "this case must saturate coverage");
        assert_eq!(<Normalised as BlendMode<f64>>::coverage(&acc), 1.0);
        let equation_five = acc.weighted * acc.weight.recip();
        for background in [Rgb::zero(), Rgb::new(1.0, 1.0, 1.0), Rgb::new(0.3, 0.7, 0.2)] {
            let out = resolve(&acc, background);
            assert_eq!(
                out.x.to_bits(),
                equation_five.x.to_bits(),
                "where total weight reaches one the coverage mix must reduce \
                 BIT-EXACTLY to Image-GS Eq. 5, and the background must drop out \
                 entirely: {out:?} against {equation_five:?} on {background:?}"
            );
            assert_eq!(out.y.to_bits(), equation_five.y.to_bits());
            assert_eq!(out.z.to_bits(), equation_five.z.to_bits());
        }
    }

    let below = accumulate(&[(0.4, Rgb::new(1.0, 1.0, 1.0))], 1.0, &s);
    assert!(below.weight < 1.0);
    let below_mean = below.weighted * below.weight.recip();
    let below_out = resolve(&below, Rgb::zero());
    assert!(
        (below_out - below_mean).max_component().abs() > 0.5,
        "THE INSTRUMENT IS NOT VACUOUS: below unit weight the resolve is NOT \
         Eq. 5, so the identity above is a statement about the saturated case \
         and not a tautology about the resolve. {below_out:?} vs {below_mean:?}"
    );
}

#[test]
fn normalised_is_permutation_invariant_to_one_ulp() {
    assert_eq!(<Normalised as BlendMode<f64>>::ORDER_DEPENDENT, false);
    assert_eq!(<Normalised as BlendMode<f64>>::SIGNED, false);
    assert_eq!(<Over as BlendMode<f64>>::ORDER_DEPENDENT, true);

    let s = no_floor();
    let mut state: u64 = 7;
    let mut next_bits = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 40) as i64
    };
    let dyadic = |n: i64| ((n % 512).abs() as f64) / 512.0;

    let mut atoms: Vec<(f64, Rgb<f64>)> = Vec::new();
    for _ in 0..32 {
        let w = dyadic(next_bits());
        atoms.push((
            w,
            Rgb::new(
                dyadic(next_bits()),
                dyadic(next_bits()),
                dyadic(next_bits()),
            ),
        ));
    }

    let identity: Vec<usize> = (0..atoms.len()).collect();
    let run = |order: &[usize], list: &[(f64, Rgb<f64>)]| {
        let permuted: Vec<(f64, Rgb<f64>)> = order.iter().map(|i| list[*i]).collect();
        accumulate(&permuted, 1.0, &s)
    };

    let baseline = run(&identity, &atoms);
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
        let acc = run(&permutation, &atoms);
        assert_eq!(
            acc.weight.to_bits(),
            baseline.weight.to_bits(),
            "both sums are plain sums, so exactly-representable inputs must \
             accumulate bit-identically under permutation"
        );
        assert_eq!(acc.weighted.x.to_bits(), baseline.weighted.x.to_bits());
        assert_eq!(acc.weighted.y.to_bits(), baseline.weighted.y.to_bits());
        assert_eq!(acc.weighted.z.to_bits(), baseline.weighted.z.to_bits());
    }

    let mut rough_state: u64 = 99;
    let mut unit = || {
        rough_state = rough_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((rough_state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut rough: Vec<(f64, Rgb<f64>)> = Vec::new();
    for _ in 0..32 {
        let w = unit();
        rough.push((w, Rgb::new(unit(), unit(), unit())));
    }
    let rough_baseline = run(&identity, &rough);
    let rough_resolved = resolve(&rough_baseline, Rgb::zero());
    let mut rough_permutation = identity.clone();
    let mut worst_drift: f64 = 0.0;
    for _ in 0..1000 {
        for i in (1..rough_permutation.len()).rev() {
            shuffle_state = shuffle_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            let j = (shuffle_state >> 33) as usize % (i + 1);
            rough_permutation.swap(i, j);
        }
        let acc = run(&rough_permutation, &rough);
        let drift = (resolve(&acc, Rgb::zero()) - rough_resolved)
            .max_component()
            .abs();
        worst_drift = worst_drift.max(drift);
    }
    assert!(
        worst_drift < 1e-14,
        "normalisation is mathematically order-independent: {worst_drift}"
    );
    assert!(
        worst_drift > 0.0,
        "FLOAT ADDITION IS NOT ASSOCIATIVE, and the quotient of two \
         order-perturbed sums inherits that. Order independence buys freedom \
         from the DEPTH SORT, not bit-identity under an arbitrary visit order; \
         determinism still comes from visiting atoms in a fixed (tile, slot) \
         order. Measured drift over 1000 permutations of 32 non-dyadic atoms: \
         one ULP."
    );
}

#[test]
fn the_normalised_path_never_early_outs() {
    let s = no_floor();
    let ink = Rgb::new(1.0, 1.0, 1.0);
    let mut ladder = <Normalised as BlendMode<f64>>::init();
    assert!(!<Normalised as BlendMode<f64>>::saturated(&ladder, &s));
    for i in 0..64 {
        let step = <Normalised as BlendMode<f64>>::step(&mut ladder, 1.0, ink, 1.0, &s);
        assert_eq!(
            step,
            Step::Continue,
            "step {i} returned Stop at accumulated weight {}",
            ladder.weight
        );
        assert!(
            !<Normalised as BlendMode<f64>>::saturated(&ladder, &s),
            "A NORMALISED ACCUMULATOR CAN NEVER SATURATE, at any weight. \
             `saturated` currently has NO CONSUMER in any rendering path in \
             src/ — only tests/blend.rs calls it, for Over and Additive — so a \
             wrong answer here changes no pixel and no counter and can be caught \
             ONLY by asserting the law directly, exactly as tests/blend.rs does \
             for the other two rules. Weight {} after {i} unit contributions.",
            ladder.weight
        );
    }
    assert!(ladder.weight >= 64.0, "the ladder must reach well past unit weight");
}

#[test]
#[cfg(feature = "method-splat")]
fn the_normalised_path_never_early_outs_through_the_rasteriser() {
    let over = render_stack(KernelFamily::Gaussian, stack_scene(40, 0.5));
    let normalised = render_stack(KernelFamily::GaussianNormalised, stack_scene(40, 0.5));

    assert!(
        over.counters.early_outs > 0,
        "the scene must actually saturate under alpha compositing or the \
         comparison is empty: {} early-outs",
        over.counters.early_outs
    );
    assert_eq!(
        normalised.counters.early_outs, 0,
        "there is no transmittance in a normalised accumulator, so there is no \
         `the rest is hidden` argument to make and Step::Stop can never be \
         returned. A nonzero count here means the accumulator acquired a \
         saturation rule it has no basis for."
    );
    assert_eq!(
        normalised.counters.pixels_visited, over.counters.pixels_visited,
        "both arms must have walked the same pixels for the counts to be \
         comparable at all"
    );
    assert!(
        normalised.counters.composited_hits > over.counters.composited_hits,
        "THE COST ASYMMETRY IS A MEASURED NUMBER, NOT A CLAIM: alpha \
         compositing stops on a saturated interior and normalisation does not. \
         normalised {} composited hits against over {} over the same {} pixels.",
        normalised.counters.composited_hits,
        over.counters.composited_hits,
        over.counters.pixels_visited
    );
    assert_eq!(
        normalised.counters.composited_hits, normalised.counters.kernel_evaluations,
        "every binned candidate whose support covers the pixel is evaluated AND \
         composited on the normalised path: the two counts must coincide"
    );
    assert!(
        over.counters.composited_hits < over.counters.kernel_evaluations,
        "on the Over path they must not, or the saturated interior never fired"
    );
}

#[test]
fn two_atoms_render_the_closed_form_logistic() {
    let s = no_floor();
    let configurations = [(1.0, 1.0), (1.0, 2.0), (0.5, 0.5), (2.0, 3.0), (1.5, 0.75)];

    for (sigma, separation) in configurations {
        let half_width = LN_NINE * sigma * sigma / separation;
        for i in 0..41 {
            let x = -2.0 * half_width + (i as f64) * 0.1 * half_width;
            let measured = match ink_fraction(x, sigma, separation, 1.0, &s) {
                Some(v) => v,
                None => panic!("no weight at {x} for sigma {sigma}"),
            };
            let expected = logistic(x, sigma, separation);
            assert!(
                (measured - expected).abs() < 1e-12,
                "two isotropic atoms at +-d/2 carrying ink and paper must \
                 reproduce t(x) = 1/(1+exp(-x d / sigma^2)) exactly, because \
                 w_ink/w_paper = exp(x d / sigma^2) and normalisation turns that \
                 ratio into the colour: at x {x}, sigma {sigma}, d {separation} \
                 got {measured} against {expected}"
            );
        }

        let scaled = match ink_fraction(0.37 * half_width, sigma, separation, 250.0, &s) {
            Some(v) => v,
            None => panic!("no weight under an opacity scale"),
        };
        let unscaled = match ink_fraction(0.37 * half_width, sigma, separation, 1.0, &s) {
            Some(v) => v,
            None => panic!("no weight without an opacity scale"),
        };
        assert!(
            (scaled - unscaled).abs() < 1e-14,
            "the profile is a RATIO, so a common opacity scale cancels out of it \
             while moving coverage: {scaled} against {unscaled}"
        );

        let upper = match ink_fraction(half_width, sigma, separation, 1.0, &s) {
            Some(v) => v,
            None => panic!("no weight at the upper crossing"),
        };
        let lower = match ink_fraction(-half_width, sigma, separation, 1.0, &s) {
            Some(v) => v,
            None => panic!("no weight at the lower crossing"),
        };
        assert!(
            (upper - 0.9).abs() < 1e-12 && (lower - 0.1).abs() < 1e-12,
            "the 0.9 and 0.1 crossings sit at +-ln(9) sigma^2 / d: got {upper} \
             and {lower}"
        );

        let measured_width = transition_width(sigma, separation, &s);
        let closed_form = 2.0 * LN_NINE * sigma * sigma / separation;
        assert!(
            (measured_width - closed_form).abs() < 1e-6,
            "THE IDENTITY THE ENTIRE P2 COMPARISON TURNS ON: the 10-90 \
             transition width of a normalised two-atom edge is 4.394 sigma^2 / \
             d, and it does not contain the along-edge extent at all, which is \
             why a straight edge of ANY length is two atoms under \
             normalisation. sigma {sigma}, d {separation}: scanned {measured_width} \
             against closed form {closed_form}"
        );

        let saturated = accumulate(&edge_pair(0.0, sigma, separation), 250.0, &s);
        assert!(saturated.weight >= 1.0);
        let mixed = resolve(&saturated, Rgb::zero());
        let mean = saturated.weighted * saturated.weight.recip();
        assert_eq!(
            mixed.x.to_bits(),
            mean.x.to_bits(),
            "where the edge carries unit weight the rendered colour IS the \
             logistic, with no background term left in it"
        );
    }

    assert!(
        (2.0 * LN_NINE - 4.394).abs() < 5.0e-4,
        "4.394 is 2 ln 9 rounded: {}",
        2.0 * LN_NINE
    );
}

#[test]
fn the_measured_width_moves_with_the_weight_floor() {
    let floors = [1.0 / 255.0, 1.0 / 1024.0, 1.0 / 65536.0];
    let sigma = 1.0;

    let close = 1.0;
    let close_widths: Vec<f64> = floors
        .iter()
        .map(|f| transition_width(sigma, close, &with_floor(*f)))
        .collect();
    let close_closed_form = 2.0 * LN_NINE * sigma * sigma / close;
    for (floor, width) in floors.iter().zip(close_widths.iter()) {
        assert!(
            (width - close_closed_form).abs() < 1e-6,
            "at d {close} the whole 10-90 band sits well above every floor in \
             the sweep, so the width must be the closed form at floor {floor}: \
             {width} against {close_closed_form}"
        );
    }

    let far = 6.0;
    let far_widths: Vec<f64> = floors
        .iter()
        .map(|f| transition_width(sigma, far, &with_floor(*f)))
        .collect();
    let far_closed_form = 2.0 * LN_NINE * sigma * sigma / far;

    for i in 1..floors.len() {
        assert!(
            far_widths[i] >= far_widths[i - 1] - 1e-9,
            "a HIGHER floor truncates more of the far atom's tail and can only \
             NARROW the transition: floor {} gave {} while floor {} gave {}",
            floors[i - 1],
            far_widths[i - 1],
            floors[i],
            far_widths[i]
        );
    }

    let widest = far_widths[far_widths.len() - 1];
    let narrowest = far_widths[0];
    let sensitivity = (widest - narrowest) / widest;
    assert!(
        sensitivity > 0.05,
        "THE MECHANISM MUST BE SHOWN TO BE REAL, NOT ASSUMED. At d {far} sigma \
         the far atom's weight at the 0.9 crossing falls near 1/255, so the \
         floor truncates it and the profile jumps to pure ink early. Measured \
         widths across floors {floors:?}: {far_widths:?} against a closed form \
         of {far_closed_form}. Relative sensitivity {sensitivity}."
    );
    assert!(
        (far_widths[far_widths.len() - 1] - far_closed_form).abs() < 1e-6,
        "at the finest floor the truncation must be pushed out of the band and \
         the closed form recovered: {} against {far_closed_form}",
        far_widths[far_widths.len() - 1]
    );
    assert!(
        sensitivity < 0.5,
        "ALPHA_CUTOFF DOES DOUBLE DUTY as the normalised weight floor, and the \
         direction of the bias FAVOURS THE ANSWER BEING TESTED: too high a floor \
         truncates the gaussian tail, NARROWS the measured transition, and so \
         flatters normalised blending exactly where the P2 comparison is \
         decided. Widths across floors {floors:?} at sigma {sigma}, d {far}: \
         {far_widths:?}, a relative spread of {sensitivity}. Any headline \
         transition width must therefore be reported WITH the floor it was \
         measured at, and the P2 harness must sweep this axis rather than pick \
         a value."
    );
}
