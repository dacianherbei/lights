#![cfg(feature = "fit")]
use lights::fit::content::Content;
use lights::fit::coverage::{DEFAULT_CELL_LIMIT, DEFAULT_MAX_DEPTH, coverage_budget};
use lights::fit::atom2d::camera;
use lights::fit::aspect2::{TermChoice, class_panels};
use lights::fit::greedy::{FitRequest, Reading, fit, fit_budget, render_or_background};
use lights::fit::harness::{ProbeRequest, ProfileChoice, run};
use lights::fit::nterm::{
    ATOM_SCALARS_CARRIED, ATOM_SCALARS_NATIVE, Arm, ByteCrossing, ByteRule, ByteVerdict, LADDER,
    NTermCurve, NTermPoint, NTermRequest, Optimality, byte_crossings,
    REACHABLE_ARMS, VALUE_BYTES, atom_bytes, bytes_combinatorial, bytes_explicit, image_energy, index_bits, ladder_upto,
    log2_choose, max_abs_error, nterm_curve, positions, relative_l2,
};
use lights::fit::transform::{CHANNELS, Ranking, Transform};
use lights::image::FloatImage;
use lights::kernel::{Beta, KernelFamily};
use lights::numerics::vector::Rgb;

fn truth_of(content: &Content) -> FloatImage<f64> {
    let mut budget = coverage_budget(DEFAULT_CELL_LIMIT);
    match content.truth(DEFAULT_MAX_DEPTH, &mut budget) {
        Ok(t) => t,
        Err(e) => panic!("{} produced no truth image: {e}", content.name()),
    }
}

#[test]
fn relative_l2_is_zero_on_an_exact_match_and_one_when_the_reconstruction_is_dropped() {
    let mut truth = FloatImage::<f64>::new(8, 8);
    for y in 0..8 {
        for x in 0..8 {
            truth.set(x, y, Rgb::new(0.25 + 0.01 * x as f64, 0.5, 0.75 - 0.01 * y as f64));
        }
    }
    let exact = match relative_l2(&truth, &truth) {
        Ok(v) => v,
        Err(e) => panic!("{e}"),
    };
    assert_eq!(exact, 0.0, "an exact match has no relative error at all");

    let empty = FloatImage::<f64>::new(8, 8);
    let dropped = match relative_l2(&truth, &empty) {
        Ok(v) => v,
        Err(e) => panic!("{e}"),
    };
    assert!(
        (dropped - 1.0).abs() < 1.0e-15,
        "a reconstruction that returns nothing carries exactly the truth's own energy, so \
         the relative error is one and not one squared: got {dropped}. Dropping the square \
         root would report {} here",
        image_energy(&truth) / image_energy(&truth)
    );

    let mut half = truth.clone();
    for p in half.pixels.iter_mut() {
        *p = *p * 0.5;
    }
    let scaled = match relative_l2(&truth, &half) {
        Ok(v) => v,
        Err(e) => panic!("{e}"),
    };
    assert!(
        (scaled - 0.5).abs() < 1.0e-15,
        "halving every pixel must halve the relative L2, got {scaled}: this is the value \
         that separates a norm from an energy"
    );

    let zero = FloatImage::<f64>::new(4, 4);
    assert!(
        relative_l2(&zero, &zero).is_ok(),
        "a zero truth matched exactly is not an error"
    );
    assert!(
        relative_l2(&zero, &truth).is_err(),
        "a zero-energy truth has no scale to be relative to, and inventing one would hide a \
         broken content class behind a plausible number"
    );
}

#[test]
fn the_two_metrics_rank_the_same_pair_in_opposite_orders() {
    let truth = FloatImage::<f64>::filled(16, 16, Rgb::new(0.5, 0.5, 0.5));

    let mut broad = truth.clone();
    for p in broad.pixels.iter_mut() {
        *p = *p + Rgb::new(0.03, 0.03, 0.03);
    }
    let mut spike = truth.clone();
    spike.set(7, 9, Rgb::new(0.5, 0.5, 0.8));

    let broad_l2 = relative_l2(&truth, &broad).unwrap_or(f64::NAN);
    let spike_l2 = relative_l2(&truth, &spike).unwrap_or(f64::NAN);
    let broad_max = max_abs_error(&truth, &broad).unwrap_or(f64::NAN);
    let spike_max = max_abs_error(&truth, &spike).unwrap_or(f64::NAN);

    assert!(
        broad_l2 > spike_l2,
        "relative L2 must prefer the single wrong pixel: {broad_l2} against {spike_l2}"
    );
    assert!(
        spike_max > broad_max,
        "max-absolute error must prefer the broad small drift: {spike_max} against \
         {broad_max}"
    );
    assert!(
        broad_l2 / spike_l2 > 2.0 && spike_max / broad_max > 2.0,
        "the disagreement must be wide enough to matter, not a rounding accident: L2 ratio \
         {:.3}, max-abs ratio {:.3}. Reporting either metric alone would decide which family \
         wins on a one-pixel stroke before any measurement was taken",
        broad_l2 / spike_l2,
        spike_max / broad_max
    );
}

#[test]
fn the_combinatorial_index_cost_is_the_exact_binomial() {
    let cases: [(usize, usize, u128); 6] = [
        (48, 0, 1),
        (48, 1, 48),
        (48, 3, 17_296),
        (48, 45, 17_296),
        (48, 48, 1),
        (30, 15, 155_117_520),
    ];
    for (total, chosen, expected) in cases {
        let mut exact: u128 = 1;
        for i in 0..chosen as u128 {
            exact = exact * ((total as u128) - i) / (i + 1);
        }
        assert_eq!(
            exact, expected,
            "the integer reference for C({total},{chosen}) is wrong, so this test measures \
             nothing"
        );
        let got = log2_choose(total, chosen);
        let want = (expected as f64).log2();
        assert!(
            (got - want).abs() < 1.0e-9,
            "log2 C({total},{chosen}) is {want} and the closed form returned {got}"
        );
    }

    let (width, height) = (64u32, 64u32);
    assert_eq!(
        positions(width, height),
        (width as usize) * (height as usize) * CHANNELS,
        "a retained coefficient is named among the positions of ALL THREE colour planes, not \
         of one of them: paying for a lattice index alone understates the cost by log2(3) \
         bits a term and would put the lower bracket ABOVE the upper one"
    );
    assert_eq!(index_bits(64, 64), 14.0);
    assert_eq!(index_bits(32, 32), 12.0);
}

#[test]
fn the_two_byte_columns_bracket_the_unknown_number_and_never_cross() {
    for (width, height) in [(64u32, 64u32), (32u32, 32u32)] {
        let available = positions(width, height);
        let mut previous_explicit = 0.0f64;
        let mut previous_combinatorial = 0.0f64;
        for terms in ladder_upto(available) {
            let explicit = bytes_explicit(terms, width, height);
            let combinatorial = bytes_combinatorial(terms, width, height);
            assert!(
                explicit >= combinatorial - 1.0e-9,
                "at {terms} terms in {width}x{height} the explicit upper bracket is \
                 {explicit} and the combinatorial lower bracket is {combinatorial}: a lower \
                 bound above its own upper bound would mean the byte accounting is not an \
                 accounting at all"
            );
            assert!(
                combinatorial > (terms as f64) * 4.0 - 1.0e-9,
                "naming which positions were kept is never free"
            );
            assert!(
                explicit > previous_explicit && combinatorial > previous_combinatorial,
                "both byte columns must be strictly monotone in the term count"
            );
            previous_explicit = explicit;
            previous_combinatorial = combinatorial;
        }
        assert!(
            bytes_combinatorial(available, width, height)
                < bytes_explicit(available, width, height),
            "keeping everything costs no index at all under the combinatorial rule and a \
             full index under the explicit one, so the gap must be widest here"
        );
    }
    assert_eq!(ATOM_SCALARS_CARRIED, 14);
    assert_eq!(ATOM_SCALARS_NATIVE, 9);
}

#[test]
fn an_atom_scalar_and_a_coefficient_value_are_billed_in_one_currency() {
    assert_eq!(VALUE_BYTES, 4.0);
    assert_eq!(VALUE_BYTES, lights::fit::aspect3::SYMMETRIC_VALUE_BYTES);
    assert_eq!(ATOM_SCALARS_NATIVE, lights::fit::aspect3::ATOM_SCALARS_NATIVE);
    assert_eq!(
        atom_bytes(1, 1),
        VALUE_BYTES,
        "one atom scalar must cost exactly what one retained coefficient value costs, or the \
         byte column compares two currencies"
    );
    assert_eq!(atom_bytes(1, ATOM_SCALARS_NATIVE), 36.0);
    assert_eq!(atom_bytes(1, ATOM_SCALARS_CARRIED), 56.0);
    assert_eq!(
        atom_bytes(8, ATOM_SCALARS_NATIVE),
        lights::fit::aspect3::atom_bytes_symmetric(8),
        "aspect 2 and aspect 3 must bill the same atom identically"
    );
    let value_part = bytes_combinatorial(1, 64, 64) - log2_choose(positions(64, 64), 1) / 8.0;
    assert!(
        (value_part - atom_bytes(1, 1)).abs() < 1.0e-12,
        "a coefficient's value part is {value_part} B and an atom scalar is {} B",
        atom_bytes(1, 1)
    );
}

#[test]
fn the_ladder_never_exceeds_the_available_term_count() {
    for available in [0usize, 1, 2, 5, 8, 9, 1024, 4096, 12288] {
        let ladder = ladder_upto(available);
        for n in ladder.iter() {
            assert!(
                *n > 0 && *n <= available,
                "the ladder offered {n} terms out of {available} available"
            );
        }
        let mut sorted = ladder.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted, ladder, "the ladder must be strictly increasing");
        if available > 0 {
            assert_eq!(
                ladder.last(),
                Some(&available),
                "the last rung must be everything the arm actually had, or the curve stops \
                 short of its own ceiling without saying so"
            );
            assert_eq!(ladder.first(), Some(&1));
        } else {
            assert!(ladder.is_empty());
        }
    }
    assert_eq!(ladder_upto(5), vec![1, 2, 3, 4, 5]);
    assert_eq!(ladder_upto(8), LADDER[..6].to_vec());
}

#[test]
fn the_atom_curve_reproduces_the_carried_forward_harness_residual_energy() {
    let mut request = ProbeRequest::all();
    request.targets = vec![1, 2];
    request.blends = vec![KernelFamily::Gaussian];
    request.profiles = vec![ProfileChoice::Prefiltered];
    request.max_atoms = 2;
    request.repeats = 1;
    request.sensitivity = false;
    let report = match run(&request) {
        Ok(r) => r,
        Err(e) => panic!("the carried-forward harness failed: {e}"),
    };

    let mut compared = 0usize;
    for content in Content::carried_forward().into_iter().take(2) {
        let mut nrequest = NTermRequest::new(Arm::Atoms(KernelFamily::Gaussian));
        nrequest.max_atoms = 2;
        let curve = match nterm_curve(&content, &nrequest) {
            Ok(c) => c,
            Err(e) => panic!("{}: {e}", content.name()),
        };
        for row in report.sweep.iter().filter(|r| r.target == content.name()) {
            let mine = match curve.residual_energy_at(row.atoms) {
                Some(v) => v,
                None => panic!(
                    "{} has no N-term point at {} atoms",
                    content.name(),
                    row.atoms
                ),
            };
            let gap = (mine - row.residual_energy).abs() / row.residual_energy.max(1.0e-300);
            assert!(
                gap < 1.0e-9,
                "{} at {} atoms: the earlier run recorded residual energy {:.12e} and this \
                 harness computes {:.12e}, a relative gap of {gap:.3e}. The N-term axis is \
                 only readable if the atom side is the SAME fit, the SAME prefix slices and \
                 the SAME render as the run whose numbers are already published; if this \
                 fails, nothing downstream of it can be believed",
                content.name(),
                row.atoms,
                row.residual_energy,
                mine
            );
            compared += 1;
        }
    }
    assert!(
        compared >= 4,
        "only {compared} atom counts were compared against the earlier run, which is not \
         enough for this gate to mean anything"
    );
}

#[test]
fn every_arm_produces_a_complete_curve_on_every_content_class() {
    let mut seen_optimality = std::collections::BTreeSet::new();
    let mut rises: Vec<String> = Vec::new();
    let mut atom_cells = 0usize;
    let mut transform_rises = 0usize;
    let mut atom_improved = 0usize;
    let mut rows = 0usize;
    for content in Content::all() {
        let truth_energy = image_energy(&truth_of(&content));
        for arm in REACHABLE_ARMS {
            let mut request = NTermRequest::new(arm);
            request.max_atoms = 2;
            request.refine_iterations = 1;
            request.refine_tail = 1;
            let curve = match nterm_curve(&content, &request) {
                Ok(c) => c,
                Err(e) => panic!("{} on {}: {e}", arm.name(), content.name(), ),
            };
            rows += curve.points.len();
            seen_optimality.insert(curve.optimality.name());

            assert_eq!(curve.content, content.name());
            assert_eq!(curve.arm, arm.name());
            assert_eq!(curve.viewport, content.viewport());
            assert_eq!(curve.control, content.control_label());
            assert_eq!(curve.byte_rule, arm.byte_rule());
            assert_eq!(curve.optimality, arm.optimality());
            assert!(
                (curve.truth_energy - truth_energy).abs() <= truth_energy * 1.0e-12,
                "{} on {}: the curve is normalised against a different truth image than the \
                 class hands out",
                arm.name(),
                content.name()
            );
            assert!(
                !curve.points.is_empty(),
                "{} produced no point at all on {}",
                arm.name(),
                content.name()
            );
            assert_eq!(
                curve.points.iter().map(|p| p.terms).collect::<Vec<_>>(),
                ladder_upto(curve.available_terms),
                "{} on {} skipped a rung of its own ladder",
                arm.name(),
                content.name()
            );

            for point in curve.points.iter() {
                assert_eq!(point.optimality, curve.optimality);
                assert!(
                    point.relative_l2.is_finite() && point.relative_l2 >= 0.0,
                    "{} on {} reported relative L2 {}",
                    arm.name(),
                    content.name(),
                    point.relative_l2
                );
                assert!(
                    point.max_abs_error.is_finite() && point.max_abs_error >= 0.0,
                    "{} on {} reported max-abs {}",
                    arm.name(),
                    content.name(),
                    point.max_abs_error
                );
                assert!(point.bytes_explicit > 0.0 && point.bytes_combinatorial > 0.0);
                match curve.byte_rule {
                    ByteRule::AtomParameterVector => assert!(
                        point.bytes_explicit > point.bytes_combinatorial,
                        "the fourteen-scalar convention must cost more than the nine-scalar \
                         one"
                    ),
                    ByteRule::RetainedCoefficient => assert!(
                        point.bytes_explicit >= point.bytes_combinatorial - 1.0e-9,
                        "the coefficient brackets crossed"
                    ),
                }
            }

            let first = curve.points[0].relative_l2;
            let last = match curve.points.last() {
                Some(p) => p.relative_l2,
                None => f64::NAN,
            };
            for pair in curve.points.windows(2) {
                assert!(
                    pair[0].relative_l2 != pair[1].relative_l2
                        || pair[0].relative_l2 < 1.0e-12,
                    "{} on {} reports the identical error {} at {} and {} terms while still \
                     short of the truth. Two rungs that land on the same number above machine \
                     zero are what an arm scored on its whole answer at every rung looks \
                     like; an arm that has already reproduced the field exactly is allowed to \
                     repeat itself and pay for the extra terms",
                    arm.name(),
                    content.name(),
                    pair[0].relative_l2,
                    pair[0].terms,
                    pair[1].terms
                );
                if pair[1].relative_l2 > pair[0].relative_l2 {
                    if !arm.has_fitter() {
                        transform_rises += 1;
                    }
                    rises.push(format!(
                        "{} on {}: {} terms {:.6} then {} terms {:.6}",
                        arm.name(),
                        content.name(),
                        pair[0].terms,
                        pair[0].relative_l2,
                        pair[1].terms,
                        pair[1].relative_l2
                    ));
                }
            }
            let best = curve
                .points
                .iter()
                .map(|p| p.relative_l2)
                .fold(f64::INFINITY, f64::min);
            if arm.has_fitter() {
                atom_cells += 1;
                if best < first * (1.0 - 1.0e-9) {
                    atom_improved += 1;
                }
                println!(
                    "ATOM {:24} {:24} available {} first {:.6} last {:.6} best {:.6}",
                    content.name(),
                    arm.name(),
                    curve.available_terms,
                    first,
                    last,
                    best
                );
            }

            if arm.has_fitter() {
                assert!(
                    curve.available_terms <= request.max_atoms,
                    "the atom arm may not place more atoms than it was allowed"
                );
            } else {
                assert_eq!(
                    curve.available_terms,
                    positions(content.viewport().0, content.viewport().1),
                    "a transform arm's ceiling is every coefficient of every plane"
                );
                assert!(
                    last < 1.0e-12,
                    "{} on {} keeps every coefficient at its last rung and must land back on \
                     the truth, and it stopped at {last}",
                    arm.name(),
                    content.name()
                );
                assert!(
                    !curve.improvement_exhausted,
                    "a transform arm has no fitter that could exhaust"
                );
            }
        }
    }
    assert_eq!(rows, 8 * (2 * 2 + 2 * 21), "the matrix lost or gained rows");
    assert_eq!(
        transform_rises, 0,
        "a transform arm got worse as a coefficient was added. On this ladder the kept sets \
         are nested and the earlier run measured the biorthogonal rises as fractions of a \
         thousandth at single-term spacing, so a rise here at ladder spacing is a different \
         fault: {rises:?}"
    );
    assert!(
        atom_improved * 4 >= atom_cells * 3,
        "only {atom_improved} of {atom_cells} atom cells improved between the first and the \
         best rung of their own curve"
    );
    assert!(
        atom_improved < atom_cells,
        "every atom cell improved, so the recorded finding that a greedy prefix can get \
         WORSE as an atom is added is no longer visible in this run and the caveat under the \
         tables would be unsupported. The three cells that carried it were the super-gaussian \
         on the three new raster classes"
    );
    assert!(
        rises.len() < rows / 4,
        "the curves rise at {} of {rows} rungs, which is too many for the N-term axis to be \
         readable at all: {rises:?}. Prefix slices of a greedy fit are NOT a nested sequence \
         and neither ranking of a biorthogonal basis is exactly optimal, so a rise is legal \
         and is recorded rather than asserted away; a run full of them is not",
        rises.len()
    );
    assert_eq!(
        seen_optimality.len(),
        3,
        "all three optimality claims must appear in one run, or the column is decorative: \
         saw {seen_optimality:?}"
    );
}

#[test]
fn the_biorthogonal_arm_never_claims_the_exact_n_term_optimum() {
    for ranking in [Ranking::Weighted, Ranking::RawMagnitude] {
        assert_eq!(
            Arm::Wavelet(Transform::Haar, ranking).optimality(),
            Optimality::ExactNTerm,
            "Haar is orthonormal, which is the whole reason keep-N-largest is optimal for it"
        );
        assert_eq!(
            Arm::Wavelet(Transform::Cdf97, ranking).optimality(),
            Optimality::BasisRanked,
            "CDF 9/7 is biorthogonal: its synthesis functions are not mutually orthogonal, so \
             no per-coefficient ranking is exactly optimal and the column must not say it is"
        );
    }
    for kernel in [
        KernelFamily::Gaussian,
        KernelFamily::SuperGaussian(Beta::Four),
    ] {
        assert_eq!(
            Arm::Atoms(kernel).optimality(),
            Optimality::GreedyLowerBound
        );
        assert!(Arm::Atoms(kernel).has_fitter());
    }
    let mut legends = std::collections::BTreeSet::new();
    for optimality in Optimality::ALL {
        assert!(
            legends.insert(optimality.legend()),
            "two optimality values share a legend line"
        );
        assert!(optimality.legend().len() > 40);
    }
    let mut names = std::collections::BTreeSet::new();
    for arm in REACHABLE_ARMS {
        assert!(names.insert(arm.name()), "two arms share a name");
    }
    assert_eq!(names.len(), 4);
}

#[test]
fn the_atom_curve_scores_prefix_slices_of_one_fit_and_not_the_whole_set() {
    let content = Content::n2_soft_shadow();
    let truth = truth_of(&content);
    let kernel = KernelFamily::SuperGaussian(Beta::Four);
    let mut request = NTermRequest::new(Arm::Atoms(kernel));
    request.max_atoms = 3;
    request.refine_iterations = 1;
    request.refine_tail = 1;

    let curve = match nterm_curve(&content, &request) {
        Ok(c) => c,
        Err(e) => panic!("{e}"),
    };
    assert!(
        curve.points.len() >= 2,
        "this test needs a fit that placed more than one atom, it placed {}",
        curve.available_terms
    );

    let mut fit_request = FitRequest::new(kernel, request.settings);
    fit_request.max_atoms = request.max_atoms;
    fit_request.tile = request.tile;
    fit_request.refine_iterations = request.refine_iterations;
    fit_request.refine_tail = request.refine_tail;
    fit_request.initialisation = request.initialisation;
    let mut budget = fit_budget(request.render_limit);
    let outcome = match fit(
        &content.fit_scaffold(),
        &truth,
        &fit_request,
        &mut budget,
    ) {
        Ok(o) => o,
        Err(e) => panic!("the independent fit failed: {e}"),
    };
    assert_eq!(
        outcome.atoms.len(),
        curve.available_terms,
        "the same request must place the same number of atoms twice"
    );

    let (width, height) = content.viewport();
    let spec = camera(width, height);
    for point in curve.points.iter() {
        let image = match render_or_background(
            &outcome.atoms[..point.terms],
            &spec,
            kernel,
            &request.settings,
            request.tile,
        ) {
            Ok(i) => i,
            Err(e) => panic!("rendering the prefix of {}: {e}", point.terms),
        };
        let mine = match relative_l2(&truth, &image) {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        assert_eq!(
            mine.to_bits(),
            point.relative_l2.to_bits(),
            "at {} terms the curve reports {} and an independent render of the first {} \
             atoms of the same fit gives {}",
            point.terms,
            point.relative_l2,
            point.terms,
            mine
        );
    }

    let whole = match render_or_background(
        &outcome.atoms,
        &spec,
        kernel,
        &request.settings,
        request.tile,
    ) {
        Ok(i) => i,
        Err(e) => panic!("{e}"),
    };
    let whole_error = match relative_l2(&truth, &whole) {
        Ok(v) => v,
        Err(e) => panic!("{e}"),
    };
    let differing = curve
        .points
        .iter()
        .filter(|p| p.relative_l2.to_bits() != whole_error.to_bits())
        .count();
    assert!(
        differing >= curve.points.len() - 1,
        "only {differing} of {} rungs differ from the error of the whole atom set. A curve \
         that renders every atom at every rung reports the same number all the way down and \
         still looks like a curve in a table",
        curve.points.len()
    );

    let mut alone_request = fit_request.clone();
    alone_request.max_atoms = 1;
    let mut alone_budget = fit_budget(request.render_limit);
    let alone = match fit(
        &content.fit_scaffold(),
        &truth,
        &alone_request,
        &mut alone_budget,
    ) {
        Ok(o) => o,
        Err(e) => panic!("the single-atom fit failed: {e}"),
    };
    assert_eq!(alone.atoms.len(), 1, "the single-atom fit placed nothing");
    assert_ne!(
        alone.atoms[0].bits(),
        outcome.atoms[0].bits(),
        "the first atom of the three-atom fit is bit-identical to the atom a one-atom fit \
         places. If that is ever true the tail refinement did nothing, and the distinction \
         this test exists to pin would be invisible"
    );
    let alone_image = match render_or_background(
        &alone.atoms,
        &spec,
        kernel,
        &request.settings,
        request.tile,
    ) {
        Ok(i) => i,
        Err(e) => panic!("{e}"),
    };
    let alone_error = match relative_l2(&truth, &alone_image) {
        Ok(v) => v,
        Err(e) => panic!("{e}"),
    };
    assert_ne!(
        alone_error.to_bits(),
        curve.points[0].relative_l2.to_bits(),
        "the one-term rung of this curve is the first atom of a THREE-atom fit, retuned by \
         the tail refinement for the set it ends up in, and it is not the atom a one-atom \
         fit would place. The two must be allowed to differ and must be reported as \
         different things: prefix-of-N at {} against refit-at-one at {alone_error}",
        curve.points[0].relative_l2
    );
}

#[test]
fn the_prefix_never_rises_at_the_published_search_effort() {
    let mut rises: Vec<String> = Vec::new();
    let mut rungs = 0usize;
    let mut cells = 0usize;
    let mut improved = 0usize;
    let mut exhausted = 0usize;
    for content in Content::all() {
        for arm in REACHABLE_ARMS {
            if !arm.has_fitter() {
                continue;
            }
            let request = NTermRequest::new(arm);
            assert_eq!(
                (request.max_atoms, request.refine_iterations, request.refine_tail),
                (8, 6, 3),
                "this test exists to run at the setting lights-p4 published, and \
                 NTermRequest::new no longer carries it"
            );
            let curve = match nterm_curve(&content, &request) {
                Ok(c) => c,
                Err(e) => panic!("{} on {}: {e}", arm.name(), content.name()),
            };
            cells += 1;
            rungs += curve.points.len();
            if curve.improvement_exhausted {
                exhausted += 1;
            }
            let first = curve.points[0].relative_l2;
            let best = curve
                .points
                .iter()
                .map(|p| p.relative_l2)
                .fold(f64::INFINITY, f64::min);
            if best < first * (1.0 - 1.0e-9) {
                improved += 1;
            }
            for pair in curve.points.windows(2) {
                if pair[1].relative_l2 > pair[0].relative_l2 {
                    rises.push(format!(
                        "{} on {}: {} terms {:.6} then {} terms {:.6}",
                        arm.name(),
                        content.name(),
                        pair[0].terms,
                        pair[0].relative_l2,
                        pair[1].terms,
                        pair[1].relative_l2
                    ));
                }
            }
            println!(
                "PUBLISHED {:24} {:24} atoms {} rungs {} first {:.6} best {:.6} exhausted {}",
                content.name(),
                arm.name(),
                curve.available_terms,
                curve.points.len(),
                first,
                best,
                curve.improvement_exhausted
            );
        }
    }
    println!(
        "PUBLISHED TOTAL: {cells} atom cells, {rungs} rungs, {} rises, {improved} improved, {exhausted} exhausted",
        rises.len()
    );
    for r in rises.iter() {
        println!("  RISE {r}");
    }
    assert!(
        cells >= 8,
        "only {cells} atom cells ran, so this sweep is not the matrix it names"
    );
    assert!(
        rungs > cells,
        "{rungs} rungs across {cells} cells means at least one curve is a single \
         point and no prefix comparison happened at all"
    );
    assert_eq!(
        improved, cells,
        "only {improved} of {cells} atom cells improved between their first and \
         their best rung. At the published effort every one of them did, so a \
         cell that no longer improves means the prefix slices stopped being \
         distinct answers"
    );
    assert!(
        rises.is_empty(),
        "a prefix rose at the published search effort: {rises:?}. MEASURED \
         2026-09-15: at max_atoms 8, 6 refinement passes and a tail of 3, the \
         curves rise at ZERO of {rungs} rungs across {cells} cells, where the \
         reduced setting in the sibling test (2 atoms, one pass, tail one) does \
         rise. That is why this assertion is the way round it is. A rise \
         appearing here is NEWS and not a regression to patch: it would mean the \
         aspect-2 caveat holds at the published setting after all, and the aspect-2 caveat is what needs \
         correcting"
    );
    assert_eq!(
        exhausted, 0,
        "{exhausted} of {cells} cells reported improvement_exhausted at the \
         published effort. Measured 2026-09-15 it was zero everywhere: the ATOM \
         CAP stops this sweep, not the fitter, which is the opposite of what P2 \
         reported at its own setting and is the reason the caveat cannot be \
         carried across settings unexamined"
    );
}

fn read_at(n: usize, placed: usize) -> Reading {
    if n < placed { Reading::Prefix } else { Reading::OwnFit }
}

#[test]
fn every_row_read_off_a_shorter_slice_of_one_fit_is_labelled_prefix_and_only_the_full_count_is_not() {
    let mut seen = [0usize; 2];
    let mut tally = |reading: Reading| match reading {
        Reading::Prefix => seen[0] += 1,
        Reading::OwnFit => seen[1] += 1,
    };

    let mut probe = ProbeRequest::all();
    probe.targets = vec![1, 2];
    probe.blends = vec![KernelFamily::Gaussian];
    probe.profiles = vec![ProfileChoice::Prefiltered];
    probe.max_atoms = 3;
    probe.repeats = 1;
    probe.sensitivity = false;
    probe.tiled_field = true;
    let report = match run(&probe) {
        Ok(r) => r,
        Err(e) => panic!("the harness failed: {e}"),
    };
    for row in report.rows.iter().filter(|r| r.arm == "atoms" && r.field == "natural") {
        let placed = row.atoms_available;
        let cells: Vec<_> = report
            .sweep
            .iter()
            .filter(|s| s.target == row.target && s.blend == row.blend && s.profile == row.profile)
            .collect();
        assert_eq!(cells.len(), placed, "{}: one sweep cell per placed atom", row.target);
        for cell in cells {
            assert_eq!(
                cell.reading,
                read_at(cell.atoms, placed),
                "{} sweep n={} of {placed}",
                cell.target,
                cell.atoms
            );
            assert_eq!(cell.reading == Reading::Prefix, cell.atoms < placed);
            tally(cell.reading);
        }
        assert_eq!(
            row.reading,
            read_at(row.atoms, placed),
            "{}: the report row read n={} of {placed}",
            row.target,
            row.atoms
        );
    }
    for tiled in report.rows.iter().filter(|r| r.arm == "atoms" && r.field == "tiled") {
        let natural = match report.rows.iter().find(|r| {
            r.arm == "atoms" && r.field == "natural" && r.target == tiled.target && r.blend == tiled.blend
        }) {
            Some(r) => r,
            None => panic!("{}: a tiled row with no natural row", tiled.target),
        };
        assert_eq!(tiled.reading, natural.reading, "{}: the field tiles the row it was read from", tiled.target);
    }

    let content = Content::n2_soft_shadow();
    let mut request = NTermRequest::new(Arm::Atoms(KernelFamily::Gaussian));
    request.max_atoms = 3;
    let curve = match nterm_curve(&content, &request) {
        Ok(c) => c,
        Err(e) => panic!("{e}"),
    };
    for point in curve.points.iter() {
        assert_eq!(
            point.reading,
            read_at(point.terms, curve.available_terms),
            "atom curve n={} of {}",
            point.terms,
            curve.available_terms
        );
        tally(point.reading);
    }
    let mut wavelet = NTermRequest::new(REACHABLE_ARMS[2]);
    wavelet.max_atoms = 3;
    let curve = match nterm_curve(&content, &wavelet) {
        Ok(c) => c,
        Err(e) => panic!("{e}"),
    };
    assert!(
        curve.points.iter().all(|p| p.reading == Reading::OwnFit),
        "a truncated transform is that arm's own N-term answer, not a slice of a refined fit"
    );

    let arms = REACHABLE_ARMS.to_vec();
    let class = match class_panels(&content, &arms, TermChoice::Explicit(1), &request) {
        Ok(c) => c,
        Err(e) => panic!("{e}"),
    };
    for panel in class.panels.iter() {
        let expected = match (panel.terms, panel.has_fitter) {
            (Some(t), true) => read_at(t.rendered, t.available),
            _ => Reading::OwnFit,
        };
        assert_eq!(panel.reading, expected, "{}", panel.label);
        assert_eq!(panel.caption().contains("PREFIX"), expected == Reading::Prefix, "{}", panel.caption());
        if panel.has_fitter {
            tally(panel.reading);
        }
    }

    assert!(
        seen[0] > 0 && seen[1] > 0,
        "the test must read at least one prefix and one full count, read {} and {}",
        seen[0],
        seen[1]
    );
}

fn synthetic_curve(rule: ByteRule, points: &[(usize, f64)]) -> NTermCurve {
    let viewport = (32u32, 32u32);
    let rows = points
        .iter()
        .map(|(terms, error)| {
            let (explicit, combinatorial) = match rule {
                ByteRule::AtomParameterVector => (
                    atom_bytes(*terms, ATOM_SCALARS_CARRIED),
                    atom_bytes(*terms, ATOM_SCALARS_NATIVE),
                ),
                ByteRule::RetainedCoefficient => (
                    bytes_explicit(*terms, viewport.0, viewport.1),
                    bytes_combinatorial(*terms, viewport.0, viewport.1),
                ),
            };
            NTermPoint {
                terms: *terms,
                relative_l2: *error,
                max_abs_error: *error,
                bytes_explicit: explicit,
                bytes_combinatorial: combinatorial,
                optimality: Optimality::ExactNTerm,
                reading: Reading::OwnFit,
                wall_nanos: 0,
            }
        })
        .collect();
    NTermCurve {
        content: "synthetic".to_string(),
        arm: "synthetic".to_string(),
        viewport,
        optimality: Optimality::ExactNTerm,
        byte_rule: rule,
        control: "none".to_string(),
        available_terms: points.len(),
        improvement_exhausted: false,
        truth_energy: 1.0,
        points: rows,
    }
}

fn only_crossing(atoms: &NTermCurve, transform: &NTermCurve) -> ByteCrossing {
    let crossings = match byte_crossings(atoms, transform) {
        Ok(c) => c,
        Err(e) => panic!("byte_crossings refused two well-formed curves: {e}"),
    };
    assert_eq!(crossings.len(), atoms.points.len());
    crossings[0]
}

#[test]
fn a_transform_is_cheaper_only_when_its_upper_bracket_beats_the_native_atom_bytes() {
    let ladder = [1usize, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64];
    let four_atoms = synthetic_curve(ByteRule::AtomParameterVector, &[(4, 0.1)]);
    assert_eq!(atom_bytes(4, ATOM_SCALARS_NATIVE), 144.0);

    let early: Vec<(usize, f64)> = ladder.iter().map(|n| (*n, 0.05)).collect();
    let won = only_crossing(&four_atoms, &synthetic_curve(ByteRule::RetainedCoefficient, &early));
    assert_eq!(won.verdict, ByteVerdict::TransformCheaper);
    assert_eq!(won.transform_terms, Some(1));
    assert_eq!(won.transform_terms_above, 0);

    let late: Vec<(usize, f64)> = ladder
        .iter()
        .map(|n| (*n, if *n >= 64 { 0.05 } else { 0.2 }))
        .collect();
    let lost = only_crossing(&four_atoms, &synthetic_curve(ByteRule::RetainedCoefficient, &late));
    assert_eq!(lost.verdict, ByteVerdict::AtomsCheaper);
    assert_eq!(lost.transform_bytes_low, Some(bytes_combinatorial(49, 32, 32)));

    let between: Vec<(usize, f64)> = ladder
        .iter()
        .map(|n| (*n, if *n >= 32 { 0.05 } else { 0.2 }))
        .collect();
    let open = only_crossing(&four_atoms, &synthetic_curve(ByteRule::RetainedCoefficient, &between));
    assert_eq!(open.transform_terms_above, 24);
    assert_eq!(open.transform_terms, Some(32));
    assert_eq!(open.transform_bytes_low, Some(bytes_combinatorial(25, 32, 32)));
    assert_eq!(open.transform_bytes_high, Some(bytes_explicit(32, 32, 32)));
    assert!(bytes_combinatorial(25, 32, 32) < 144.0 && 144.0 < bytes_explicit(32, 32, 32));
    assert!(bytes_combinatorial(32, 32, 32) > 144.0);
    assert_eq!(open.verdict, ByteVerdict::Bracketed);

    let never: Vec<(usize, f64)> = ladder.iter().map(|n| (*n, 0.2)).collect();
    let missed = only_crossing(&four_atoms, &synthetic_curve(ByteRule::RetainedCoefficient, &never));
    assert_eq!(missed.verdict, ByteVerdict::NeverReached);
    assert_eq!(missed.transform_bytes_low, None);

    let coefficients = synthetic_curve(ByteRule::RetainedCoefficient, &early);
    assert!(byte_crossings(&coefficients, &coefficients).is_err());
    assert!(byte_crossings(&four_atoms, &four_atoms).is_err());
    assert_eq!(ByteVerdict::ALL.len(), 4);
}
