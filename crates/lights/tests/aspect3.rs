#![cfg(feature = "fit")]
use lights::fit::aspect3::{
    ATOM_SCALARS_NATIVE, Arm3, Aspect3Request, Direction, Independence, MAGNIFY_LADDER,
    MINIFY_LADDER, Resample, SYMMETRIC_VALUE_BYTES, ScaleRow, analytic_bytes_symmetric, arms,
    atom_bytes_symmetric, boundary_localised_reference, class_ladder,
    coefficient_bytes_symmetric, fit_at, fit_request_for,
    pixelwise_identical, prepare_wavelet, render_atoms_at, resample, run, wavelet_at,
};
use lights::fit::greedy::Initialisation;
use lights::fit::harness::ProfileChoice;
use lights::fit::scale::{Sampler, Scale, ScaleClass};
use lights::fit::target::Target;
use lights::fit::transform::{Ranking, Transform, max_levels};
use lights::image::FloatImage;
use lights::numerics::vector::Rgb;

fn scale_of(n: u32, d: u32) -> Scale {
    match Scale::new(n, d) {
        Ok(s) => s,
        Err(e) => panic!("{n}/{d}: {e}"),
    }
}

fn checkerboard(width: u32, height: u32) -> FloatImage<f64> {
    let mut image = FloatImage::<f64>::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let t = ((x * 7 + y * 13) % 17) as f64 / 17.0;
            image.set(x, y, Rgb::new(t, 1.0 - t, (t * t).min(1.0)));
        }
    }
    image
}

fn cheap(direction: Direction, rungs: &[(u32, u32)]) -> Aspect3Request {
    let mut request = match Aspect3Request::new(direction) {
        Ok(r) => r,
        Err(e) => panic!("{e}"),
    };
    request.ladder = rungs.iter().map(|(n, d)| scale_of(*n, *d)).collect();
    request.max_atoms = 2;
    request.refine_iterations = 1;
    request.refine_tail = 0;
    request.initialisation = Initialisation::Isotropic;
    request.wavelet_terms = 32;
    request.profiles = vec![ProfileChoice::Contract];
    request
}

fn ladder_of(class: &ScaleClass, request: &Aspect3Request) -> Vec<ScaleRow> {
    match class_ladder(class, request) {
        Ok(l) => l.rows,
        Err(e) => panic!("{} did not produce a ladder: {e}", class.name()),
    }
}

#[test]
fn a_resampler_asked_for_the_lattice_it_already_has_returns_it_bit_for_bit() {
    let image = checkerboard(16, 12);
    for rule in [Resample::Bilinear, Resample::PixelBoxArea] {
        let same = match resample(&image, (16, 12), rule) {
            Ok(i) => i,
            Err(e) => panic!("{}: {e}", rule.name()),
        };
        for (a, b) in image.pixels.iter().zip(same.pixels.iter()) {
            assert_eq!(
                (a.x.to_bits(), a.y.to_bits(), a.z.to_bits()),
                (b.x.to_bits(), b.y.to_bits(), b.z.to_bits()),
                "{} changed an image it was asked to leave alone. If the identity is not \
                 exact, the lattice arm's row AT ITS OWN BASE SCALE carries a resampling \
                 error, and every decay ratio measured against that base is wrong",
                rule.name()
            );
        }
    }
}

#[test]
fn area_resampling_conserves_the_mean_and_bilinear_never_invents_range() {
    let image = checkerboard(16, 16);
    let mut source_mean = Rgb::zero();
    for p in image.pixels.iter() {
        source_mean = source_mean + *p;
    }
    source_mean = source_mean * (1.0 / (image.pixels.len() as f64));

    let small = match resample(&image, (4, 4), Resample::PixelBoxArea) {
        Ok(i) => i,
        Err(e) => panic!("{e}"),
    };
    let mut small_mean = Rgb::zero();
    for p in small.pixels.iter() {
        small_mean = small_mean + *p;
    }
    small_mean = small_mean * (1.0 / (small.pixels.len() as f64));
    assert!(
        (small_mean.x - source_mean.x).abs() < 1.0e-12
            && (small_mean.y - source_mean.y).abs() < 1.0e-12
            && (small_mean.z - source_mean.z).abs() < 1.0e-12,
        "area averaging must conserve the mean exactly, or the lattice arm loses energy on \
         minification for a reason that has nothing to do with the lattice: {small_mean:?} \
         against {source_mean:?}"
    );

    let (lo, hi) = image.min_max_channel();
    let big = match resample(&image, (64, 64), Resample::Bilinear) {
        Ok(i) => i,
        Err(e) => panic!("{e}"),
    };
    let (blo, bhi) = big.min_max_channel();
    assert!(
        blo >= lo - 1.0e-12 && bhi <= hi + 1.0e-12,
        "bilinear magnification is a convex combination and cannot leave the source range; \
         got [{blo}, {bhi}] from [{lo}, {hi}]"
    );
}

#[test]
fn magnifying_a_lattice_does_not_shift_the_picture_by_half_a_source_pixel() {
    let mut ramp = FloatImage::<f64>::new(8, 8);
    for y in 0..8 {
        for x in 0..8 {
            ramp.set(x, y, Rgb::splat(x as f64));
        }
    }
    let big = match resample(&ramp, (16, 16), Resample::Bilinear) {
        Ok(i) => i,
        Err(e) => panic!("{e}"),
    };
    for i in 1..7u32 {
        let left = big.get(2 * i, 4).x;
        let right = big.get(2 * i + 1, 4).x;
        let straddled = 0.5 * (left + right);
        assert!(
            (straddled - (i as f64)).abs() < 1.0e-12,
            "source pixel {i} carries the value {i}, and after doubling its two children must \
             straddle it symmetrically, averaging back to {i}. They average to {straddled}, so \
             the magnified picture is offset by half a source pixel. An offset like that adds \
             a shift to every magnified row of the lattice arm, and the run would report it as \
             the lattice failing to carry detail when it is really the resampler being \
             misaligned"
        );
        assert!(
            (right - left - 0.5).abs() < 1.0e-12,
            "doubling a unit ramp must halve its per-pixel step"
        );
    }
}

#[test]
fn the_lattice_arm_is_handed_the_correct_resampler_in_each_direction() {
    assert_eq!(Resample::for_ratio(4.0), Resample::Bilinear);
    assert_eq!(Resample::for_ratio(1.0), Resample::Bilinear);
    assert_eq!(Resample::for_ratio(0.5), Resample::PixelBoxArea);
    assert_eq!(Resample::for_ratio(0.25), Resample::PixelBoxArea);
    let base = checkerboard(16, 16);
    let prepared = match prepare_wavelet(&base, Transform::Haar, Ranking::Weighted, 32) {
        Ok(Some(w)) => w,
        Ok(None) => panic!("a 16x16 base is dyadic and the lattice arm must be present on it"),
        Err(e) => panic!("{e}"),
    };
    let up = match wavelet_at(&prepared, (32, 32), 2.0) {
        Ok(i) => i,
        Err(e) => panic!("{e}"),
    };
    assert_eq!((up.width, up.height), (32, 32));
    let down = match wavelet_at(&prepared, (8, 8), 0.5) {
        Ok(i) => i,
        Err(e) => panic!("{e}"),
    };
    assert_eq!((down.width, down.height), (8, 8));
    assert!(
        prepared.available > prepared.terms,
        "the lattice arm must be truncated, or it is the truth wearing a hat"
    );
}

#[test]
fn the_arms_are_labelled_asymmetric_because_they_are() {
    let all = arms(lights::kernel::KernelFamily::Gaussian);
    assert_eq!(all.len(), 5);
    assert_eq!(
        Arm3::Frozen(lights::kernel::KernelFamily::Gaussian).independence(),
        Independence::ContinuousFrozen
    );
    assert_eq!(
        Arm3::Refit(lights::kernel::KernelFamily::Gaussian).independence(),
        Independence::ContinuousRefit
    );
    assert_eq!(Arm3::AnalyticSdf.independence(), Independence::ByConstruction);
    assert_eq!(
        Arm3::Wavelet(Transform::Haar).independence(),
        Independence::LatticeBound
    );
    assert!(
        Arm3::Wavelet(Transform::Haar)
            .independence()
            .legend()
            .contains("STRUCTURAL"),
        "the lattice arm's loss must be declared structural in the arm's own legend, not \
         buried in a footnote: it is a fixed dyadic lattice and it cannot resample for free"
    );
    assert!(!Arm3::AnalyticSdf.honours_profile());
    assert!(!Arm3::Wavelet(Transform::Cdf97).honours_profile());
    assert!(Arm3::Frozen(lights::kernel::KernelFamily::Gaussian).honours_profile());
}

#[test]
fn the_two_ladders_start_at_the_scale_they_fit_at_and_move_in_opposite_directions() {
    assert_eq!(MAGNIFY_LADDER[0], (1, 1));
    assert_eq!(MINIFY_LADDER[0], (8, 1));
    for direction in Direction::ALL {
        let ladder = match direction.ladder() {
            Ok(l) => l,
            Err(e) => panic!("{e}"),
        };
        let base = match direction.base() {
            Ok(b) => b,
            Err(e) => panic!("{e}"),
        };
        assert_eq!(ladder.first(), Some(&base));
        match direction {
            Direction::Magnify => assert!(
                ladder.last().map(|s| s.factor()) == Some(8.0),
                "magnification must reach 8x"
            ),
            Direction::Minify => assert!(
                ladder.last().map(|s| s.factor()) == Some(0.25),
                "minification must reach a quarter"
            ),
        }
    }
}

#[test]
fn the_frozen_and_refitted_arms_are_the_same_picture_at_the_scale_they_were_fitted_at() {
    let request = cheap(Direction::Magnify, &[(1, 1), (2, 1)]);
    let class = ScaleClass::Analytic(Target::<f64>::t4_curve());
    let ladder = match class_ladder(&class, &request) {
        Ok(l) => l,
        Err(e) => panic!("{e}"),
    };
    let checked: Vec<&ScaleRow> = ladder
        .rows
        .iter()
        .filter(|r| r.agrees_with_frozen_at_base.is_some())
        .collect();
    assert!(
        !checked.is_empty(),
        "the base-scale agreement check must actually run, or this test is measuring nothing"
    );
    let frozen_name = Arm3::Frozen(lights::kernel::KernelFamily::Gaussian).name();
    let refit_name = Arm3::Refit(lights::kernel::KernelFamily::Gaussian).name();
    let two = scale_of(2, 1);
    let frozen_up = ladder.render_of(&frozen_name, two, "contract");
    let refit_up = ladder.render_of(&refit_name, two, "contract");
    match (frozen_up, refit_up) {
        (Some(a), Some(b)) => assert!(
            !pixelwise_identical(a, b),
            "away from the base scale the refitted arm sees a different truth and must \
             produce a different picture, or the two arms are not actually distinct and the \
             cost of resolution independence would read as zero for a trivial reason"
        ),
        (_, _) => panic!("both arms must have rendered at 2x"),
    }
    for row in checked {
        assert_eq!(
            row.agrees_with_frozen_at_base,
            Some(true),
            "at the base scale the refitted arm re-runs the SAME fit on the SAME truth, so \
             its picture must be the frozen arm's picture bit for bit. A difference means \
             the ladder is not deterministic and no decay ratio on it can be believed"
        );
    }
}

#[test]
fn the_agreement_instrument_can_tell_two_images_apart() {
    let a = checkerboard(8, 8);
    let mut b = a.clone();
    assert!(
        pixelwise_identical(&a, &b),
        "the agreement check must see two copies of one image as the same"
    );
    let one = b.get(3, 4);
    b.set(3, 4, Rgb::new(one.x + f64::EPSILON, one.y, one.z));
    assert!(
        !pixelwise_identical(&a, &b),
        "one ulp in one channel of one pixel must break the agreement check. An instrument \
         that cannot fail is not measuring the base-scale identity, it is asserting it"
    );
    let wrong_size = checkerboard(8, 7);
    assert!(!pixelwise_identical(&a, &wrong_size));
}

#[test]
fn the_refit_search_is_put_in_scene_units_and_the_run_says_which() {
    let mut request = cheap(Direction::Minify, &[(8, 1), (1, 1)]);
    let settings = ProfileChoice::Contract.settings();
    let eight = scale_of(8, 1);
    assert!(request.scene_invariant_search);
    assert_eq!(request.search_convention(), "scene-invariant");
    assert_eq!(
        fit_request_for(eight, settings, &request).step_scale,
        8.0,
        "with the scene-invariant search the fitter's pixel-dimensioned constants must move \
         with the scale, or the refitted arm is a different searcher at every rung and the \
         gap to the frozen arm is measuring the fitter rather than the description"
    );
    request.scene_invariant_search = false;
    assert_eq!(request.search_convention(), "device-pixel");
    assert_eq!(fit_request_for(eight, settings, &request).step_scale, 1.0);
}

#[test]
fn every_arm_decays_from_its_own_base_and_the_base_row_says_one() {
    let request = cheap(Direction::Magnify, &[(1, 1), (2, 1)]);
    let class = ScaleClass::Analytic(Target::<f64>::t4_curve());
    let rows = ladder_of(&class, &request);
    let mut base_rows = 0usize;
    for row in rows.iter() {
        if !row.at_base || row.absent {
            continue;
        }
        base_rows += 1;
        assert!(
            (row.decay - 1.0).abs() < 1.0e-12,
            "{} at its own base must decay by exactly one, got {}. The headline of this \
             aspect is each arm against ITSELF, because the absolute cross-arm comparison \
             was left ambiguous by aspect 2 and the byte convention is still unpinned",
            row.arm,
            row.decay
        );
    }
    assert!(base_rows >= 4, "only {base_rows} base rows were checked");
}

#[test]
fn the_analytic_column_stays_absent_on_content_no_signed_distance_describes() {
    let request = cheap(Direction::Magnify, &[(1, 1), (2, 1)]);
    let rows = ladder_of(&ScaleClass::colour_field(), &request);
    let analytic: Vec<&ScaleRow> = rows
        .iter()
        .filter(|r| r.arm == Arm3::AnalyticSdf.name())
        .collect();
    assert!(!analytic.is_empty(), "the analytic arm must still appear as a row");
    for row in analytic {
        assert!(
            row.absent,
            "no signed distance describes a Shepard colour field, so the bar must declare \
             itself ABSENT rather than be replaced by a weak invented arm"
        );
        assert!(row.relative_l2.is_nan(), "an absent arm has no error to report");
    }
    let present = ladder_of(&ScaleClass::Analytic(Target::<f64>::t4_curve()), &request);
    assert!(
        present
            .iter()
            .any(|r| r.arm == Arm3::AnalyticSdf.name() && !r.absent),
        "on a class the analytic arm CAN express, it must be present"
    );
}

#[test]
fn the_scene_column_and_the_pixel_column_differ_by_exactly_the_scale_factor() {
    let request = cheap(Direction::Magnify, &[(1, 1), (2, 1), (4, 1)]);
    let class = ScaleClass::Analytic(Target::<f64>::t2_straight_edge());
    let rows = ladder_of(&class, &request);
    let mut judged = 0usize;
    let mut device_at: Vec<(f64, f64)> = Vec::new();
    for row in rows.iter() {
        if !row.quality_measured || !row.transition.scene.is_finite() {
            continue;
        }
        judged += 1;
        assert!(
            (row.transition.scene * row.scale.factor() - row.transition.device).abs() < 1.0e-9,
            "the two conventions must be one multiplication apart"
        );
        if row.arm == Arm3::AnalyticSdf.name() {
            device_at.push((row.scale.factor(), row.transition.device));
        }
    }
    assert!(judged >= 6, "only {judged} rows were judged");
    assert!(device_at.len() >= 3, "the analytic bar was judged at {} scales", device_at.len());
    for (factor, device) in device_at.iter() {
        assert!(
            (device - 1.0).abs() < 0.25,
            "the analytic bar measures {device} device pixels at {factor}x. In pixels it is \
             flat by construction; in scene units it is 1/s. Reporting one without naming \
             the convention makes the whole ladder uninterpretable"
        );
    }
}

#[test]
fn both_byte_columns_bill_a_value_at_the_same_width() {
    assert_eq!(SYMMETRIC_VALUE_BYTES, 4.0);
    assert_eq!(
        atom_bytes_symmetric(1),
        (ATOM_SCALARS_NATIVE as f64) * SYMMETRIC_VALUE_BYTES
    );
    let one = coefficient_bytes_symmetric(1, 64, 64);
    assert!(
        one > SYMMETRIC_VALUE_BYTES,
        "a retained coefficient is a value PLUS a position and must cost more than the value"
    );
    assert!(
        one < 2.0 * SYMMETRIC_VALUE_BYTES,
        "the index on a 64x64 three-channel image is under four bytes"
    );
    let target = Target::<f64>::t2_straight_edge();
    assert!(
        analytic_bytes_symmetric(&target) < (target.description_bytes() as f64),
        "the analytic description billed in f32 must be smaller than the f64 count the \
         earlier aspects used; C2 says the byte comparison is NOT established and this run \
         does not settle it, it only stops the two arms being billed in different currencies"
    );
}

#[test]
fn the_analytic_description_is_billed_by_one_formula_at_either_width() {
    let wide = Target::<f64>::all();
    let narrow = Target::<f32>::all();
    assert!(!wide.is_empty());
    assert_eq!(wide.len(), narrow.len());
    for (w, n) in wide.iter().zip(narrow.iter()) {
        assert_eq!(
            analytic_bytes_symmetric(w),
            n.description_bytes() as f64,
            "{}: the f32 analytic column and an f32 target's own description must be one \
             formula, not two copies held together by an inequality",
            w.name
        );
    }
    let t6 = Target::<f64>::t6_glyph_like();
    assert_eq!(t6.description_bytes(), 274);
    assert_eq!(analytic_bytes_symmetric(&t6), 150.0);
}

#[test]
fn a_refitted_arm_that_lands_worse_than_the_frozen_one_is_reported_as_a_broken_ceiling() {
    let request = cheap(Direction::Magnify, &[(1, 1), (2, 1)]);
    let report = match run(&[ScaleClass::Analytic(Target::<f64>::t6_glyph_like())], &request) {
        Ok(r) => r,
        Err(e) => panic!("{e}"),
    };
    let breached = report.ceiling_violations();
    assert!(
        !breached.is_empty(),
        "on this class and this budget the fresh fit at 2x is KNOWN to land worse than the \
         atom set carried up from 1x. If nothing is reported the detector is blind, and the \
         run would silently present a broken ceiling as the atom quality ceiling"
    );
    for (class, profile, scale, frozen, refit) in breached.iter() {
        assert!(
            refit > frozen,
            "{class} {profile} at {}: a ceiling violation means the REFITTED error exceeds \
             the FROZEN error, got refitted {refit} against frozen {frozen}",
            scale.label()
        );
    }
}

#[test]
fn a_run_carries_the_convention_it_ruled_on_in_its_own_header() {
    let request = cheap(Direction::Minify, &[(2, 1), (1, 1)]);
    let report = match run(&[ScaleClass::Analytic(Target::<f64>::t4_curve())], &request) {
        Ok(r) => r,
        Err(e) => panic!("{e}"),
    };
    let header = lights::fit::aspect3::header_lines(&report).join("\n");
    assert!(header.contains("primary_criterion_convention\tscene-units"));
    assert!(
        header.contains("STRICTER"),
        "the header must say out loud that a pixel criterion tightens with zoom: {header}"
    );
    assert!(header.contains("symmetric-f32-provisional-see-C2"));
    assert!(header.contains("fitter_search_convention\tscene-invariant"));
    assert!(report.base_agreement_checked() > 0);
    assert!(report.base_agreement_failures().is_empty());
}

fn two_profiles(direction: Direction, rungs: &[(u32, u32)]) -> Aspect3Request {
    let mut request = cheap(direction, rungs);
    request.profiles = vec![ProfileChoice::Contract, ProfileChoice::Prefiltered];
    request
}

#[test]
fn the_profile_axis_runs_for_the_atom_arms_and_exactly_once_for_the_arms_that_carry_their_own() {
    let request = two_profiles(Direction::Minify, &[(2, 1), (1, 1)]);
    let rows = ladder_of(&ScaleClass::Analytic(Target::<f64>::t4_curve()), &request);
    let rungs = [scale_of(2, 1), scale_of(1, 1)];
    let mut billed_to_the_profile: Vec<String> = Vec::new();
    let mut carrying_their_own = 0usize;
    for arm in arms(lights::kernel::KernelFamily::Gaussian) {
        let name = arm.name();
        for scale in rungs.iter() {
            let here: Vec<&ScaleRow> = rows
                .iter()
                .filter(|r| r.arm == name && r.scale == *scale)
                .collect();
            if arm.honours_profile() {
                assert_eq!(
                    here.len(),
                    2,
                    "{name} at {} took part in the atom prefilter, so it must be run under \
                     BOTH profiles: C3 says a minification loss under contract() alone could \
                     be the missing prefilter rather than the representation, and only the \
                     pair of rows can tell those apart. Got {} row(s)",
                    scale.label(),
                    here.len()
                );
                for row in here {
                    billed_to_the_profile.push(row.profile.clone());
                }
            } else {
                assert_eq!(
                    here.len(),
                    1,
                    "{name} at {} takes no part in the atom prefilter and carries its own, so \
                     it must appear ONCE however many profiles are run. A second identical \
                     row would double its weight in every summary and would invite the reader \
                     to believe the prefilter moved an arm it cannot touch. Got {} row(s)",
                    scale.label(),
                    here.len()
                );
                carrying_their_own += 1;
                for row in here {
                    assert_eq!(
                        row.profile, "profile-free",
                        "an arm that does not honour the profile must say so in its own \
                         profile column rather than borrow the name of a settings profile it \
                         never used"
                    );
                }
            }
        }
    }
    assert_eq!(
        carrying_their_own, 6,
        "the analytic bar and the two lattice arms across two rungs are six profile-free rows"
    );
    assert!(billed_to_the_profile.iter().any(|p| p == "contract"));
    assert!(billed_to_the_profile.iter().any(|p| p == "prefiltered"));
}

#[test]
fn the_prefilter_audit_pairs_every_contract_row_with_its_own_arm_under_the_other_profile() {
    let request = two_profiles(Direction::Minify, &[(2, 1), (1, 1)]);
    let report = match run(&[ScaleClass::Analytic(Target::<f64>::t4_curve())], &request) {
        Ok(r) => r,
        Err(e) => panic!("{e}"),
    };
    let rows = report.rows();
    let under_contract: Vec<&ScaleRow> = rows
        .iter()
        .copied()
        .filter(|r| r.profile == "contract")
        .collect();
    assert!(
        !under_contract.is_empty(),
        "the audit has nothing to pair if the unprefiltered profile never ran"
    );
    let gaps = report.prefilter_gap();
    assert_eq!(
        gaps.len(),
        under_contract.len(),
        "C3 is sized by pairing EVERY unprefiltered row with its prefiltered mate. {} rows \
         ran under contract() and {} pairs came back, so some row's mate is missing and the \
         confound is only partly sized",
        under_contract.len(),
        gaps.len()
    );
    let mut the_prefilter_moved_something = 0usize;
    for (class, arm, scale, contract_error, prefiltered_error) in gaps.iter() {
        assert_eq!(class, "t4-curve");
        let mine = rows
            .iter()
            .copied()
            .find(|r| r.profile == "contract" && &r.arm == arm && r.scale == *scale);
        let mate = rows
            .iter()
            .copied()
            .find(|r| r.profile == "prefiltered" && &r.arm == arm && r.scale == *scale);
        match (mine, mate) {
            (Some(a), Some(b)) => {
                assert_eq!(
                    *contract_error, a.relative_l2,
                    "{arm} at {}: the first number of a prefilter pair must be the error the \
                     SAME arm posted under contract(). If it is any other row's number the \
                     audit is reporting a difference between two things that were never the \
                     same measurement",
                    scale.label()
                );
                assert_eq!(
                    *prefiltered_error, b.relative_l2,
                    "{arm} at {}: the second number must be the error the SAME arm at the \
                     SAME rung posted under prefiltered(). Pairing across arms would credit \
                     the prefilter with a difference that is really the difference between \
                     two arms",
                    scale.label()
                );
                if a.relative_l2 != b.relative_l2 {
                    the_prefilter_moved_something += 1;
                }
            }
            (_, _) => panic!(
                "{arm} at {} was reported as a prefilter pair but one half of it is not in \
                 the table",
                scale.label()
            ),
        }
    }
    assert!(
        the_prefilter_moved_something > 0,
        "every pair came back identical, which would mean the prefilter changed nothing and \
         the audit is measuring nothing. C3 says the two profiles differ; an audit that \
         cannot see the difference cannot size the confound"
    );
}

#[test]
fn the_lattice_arm_picks_its_resampler_from_its_distance_from_its_own_base_not_the_device_scale() {
    let class = ScaleClass::Analytic(Target::<f64>::t4_curve());
    let request = cheap(Direction::Minify, &[(2, 1), (1, 1)]);
    let ladder = match class_ladder(&class, &request) {
        Ok(l) => l,
        Err(e) => panic!("{e}"),
    };
    let base = ladder.base_scale;
    let base_truth = match ladder.truth_of(base) {
        Some(t) => t.clone(),
        None => panic!("the ladder kept no truth at its own base"),
    };
    let mut checked = 0usize;
    let mut minified = 0usize;
    for transform in [Transform::Haar, Transform::Cdf97] {
        let arm = Arm3::Wavelet(transform).name();
        let prepared =
            match prepare_wavelet(&base_truth, transform, request.ranking, request.wavelet_terms) {
                Ok(Some(w)) => w,
                Ok(None) => panic!("this class's base is dyadic and must carry a lattice arm"),
                Err(e) => panic!("{e}"),
            };
        for scale in request.ladder.iter() {
            let row = match ladder
                .rows
                .iter()
                .find(|r| r.arm == arm && r.scale == *scale)
            {
                Some(r) => r,
                None => panic!("{arm} has no row at {}", scale.label()),
            };
            let ratio = match scale.ratio_to(base) {
                Ok(r) => r,
                Err(e) => panic!("{e}"),
            };
            let wanted = Resample::for_ratio(ratio.factor());
            if wanted == Resample::PixelBoxArea {
                minified += 1;
            }
            assert_eq!(
                row.resample,
                wanted.name(),
                "{arm} at {} is {}x of its OWN base, so it must be resampled by {}. The DEVICE \
                 scale there is {}x, which names the magnifying filter, so a run that chose \
                 its filter from the device scale would minify a lattice with bilinear and \
                 alias, then report the aliasing as the lattice failing to carry detail",
                scale.label(),
                ratio.factor(),
                wanted.name(),
                scale.factor()
            );
            let independently = match wavelet_at(&prepared, row.viewport, ratio.factor()) {
                Ok(i) => i,
                Err(e) => panic!("{e}"),
            };
            let recorded = match ladder.render_of(&arm, *scale, "profile-free") {
                Some(i) => i,
                None => panic!("{arm} did not render at {}", scale.label()),
            };
            assert!(
                pixelwise_identical(&independently, recorded),
                "{arm} at {}: the PICTURE must be the base lattice resampled by its distance \
                 from that base, not merely a row LABELLED with that rule. The label and the \
                 picture are two separate facts here and a run can carry a correct label over \
                 a picture that was resampled some other way",
                scale.label()
            );
            checked += 1;
        }
    }
    assert_eq!(
        checked, 4,
        "two lattice arms across two rungs are four rows; only {checked} were checked"
    );
    assert!(
        minified > 0,
        "this ladder must contain a rung INSIDE the base, or the minifying filter is never \
         asked for and the check cannot tell the two rules apart"
    );
    let outwards = ladder_of(&class, &cheap(Direction::Magnify, &[(1, 1), (2, 1)]));
    let haar = Arm3::Wavelet(Transform::Haar).name();
    let cdf = Arm3::Wavelet(Transform::Cdf97).name();
    let mut outward_rows = 0usize;
    for row in outwards.iter().filter(|r| r.arm == haar || r.arm == cdf) {
        outward_rows += 1;
        assert_eq!(
            row.resample,
            Resample::Bilinear.name(),
            "magnifying away from the base lattice must use the magnifying filter"
        );
    }
    assert_eq!(outward_rows, 4);
}
#[test]
fn a_lattice_bound_description_keeps_its_size_when_it_is_rendered_at_another_scale() {
    let request = cheap(Direction::Minify, &[(2, 1), (1, 1)]);
    let ladder = match class_ladder(&ScaleClass::Analytic(Target::<f64>::t4_curve()), &request) {
        Ok(l) => l,
        Err(e) => panic!("{e}"),
    };
    let base = ladder.base_viewport;
    let mut lattice_rows = 0usize;
    let mut away_from_the_base = 0usize;
    for row in ladder
        .rows
        .iter()
        .filter(|r| r.independence == Independence::LatticeBound.name())
    {
        lattice_rows += 1;
        assert_eq!(
            row.description_bytes,
            coefficient_bytes_symmetric(row.terms, base.0, base.1),
            "{} at {}: a coefficient set LIVES on the base lattice and is merely resampled \
             onto the rung. Billing its index against the rung's lattice would make the \
             description appear to shrink when it is minified and grow when it is magnified, \
             and the byte column would then be measuring the camera rather than the \
             description",
            row.arm,
            row.scale.label()
        );
        if row.viewport != base {
            away_from_the_base += 1;
            assert_ne!(
                coefficient_bytes_symmetric(row.terms, base.0, base.1),
                coefficient_bytes_symmetric(row.terms, row.viewport.0, row.viewport.1),
                "this rung must bill differently from the base, or the check above cannot \
                 tell the two lattices apart and asserts nothing"
            );
        }
    }
    assert_eq!(lattice_rows, 4);
    assert!(away_from_the_base > 0);
    for row in ladder
        .rows
        .iter()
        .filter(|r| r.independence == Independence::ContinuousFrozen.name())
    {
        assert_eq!(
            row.description_bytes,
            atom_bytes_symmetric(row.terms),
            "an atom set is billed by its atoms in the same currency"
        );
    }
}

#[test]
fn the_control_arm_is_a_fresh_fit_at_every_rung_and_not_the_frozen_set_wearing_a_hat() {
    let request = cheap(Direction::Minify, &[(2, 1), (1, 1)]);
    let class = ScaleClass::Analytic(Target::<f64>::t4_curve());
    let ladder = match class_ladder(&class, &request) {
        Ok(l) => l,
        Err(e) => panic!("{e}"),
    };
    let settings = ProfileChoice::Contract.settings();
    let refit = Arm3::Refit(lights::kernel::KernelFamily::Gaussian).name();
    let mut checked = 0usize;
    for scale in request.ladder.iter() {
        let truth = match ladder.truth_of(*scale) {
            Some(t) => t,
            None => panic!("the ladder kept no truth at {}", scale.label()),
        };
        let fresh = match fit_at(&class, truth, *scale, settings, &request) {
            Ok(f) => f,
            Err(e) => panic!("{e}"),
        };
        let viewport = match class.viewport_at(*scale) {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        let independently = match render_atoms_at(
            &fresh.atoms,
            viewport,
            request.kernel,
            &settings,
            request.tile,
        ) {
            Ok(i) => i,
            Err(e) => panic!("{e}"),
        };
        let recorded = match ladder.render_of(&refit, *scale, "contract") {
            Some(i) => i,
            None => panic!("the control arm did not render at {}", scale.label()),
        };
        checked += 1;
        assert!(
            pixelwise_identical(&independently, recorded),
            "at {} the control arm's picture must be a fit performed AT THAT RUNG, against \
             THAT RUNG'S truth, bit for bit. If it is the frozen set rendered at the rung's \
             viewport instead, the ceiling is the arm on trial wearing a hat, every \
             frozen-to-refitted gap collapses towards zero for a reason that has nothing to \
             do with resolution independence, and C1's broken-ceiling audit goes blind",
            scale.label()
        );
    }
    assert_eq!(checked, 2);
}

fn ladder_binary() -> std::process::Command {
    std::process::Command::new(env!("CARGO_BIN_EXE_lights-p5"))
}

fn output(mut command: std::process::Command) -> (i32, String) {
    let out = match command.output() {
        Ok(o) => o,
        Err(e) => panic!("running lights-p5: {e}"),
    };
    (
        out.status.code().unwrap_or(-1),
        format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ),
    )
}

#[test]
fn the_ladder_binary_refuses_to_write_to_the_repository_root() {
    let (status, text) = output(ladder_binary());
    assert_eq!(status, 2, "{text}");
    assert!(text.contains("--out-dir is REQUIRED"), "{text}");
}

#[test]
fn the_ladder_binary_rules_on_the_convention_and_names_every_artefact_before_any_number() {
    let held = match tempfile::Builder::new().prefix("lights-p5-banner-").tempdir() {
        Ok(d) => d,
        Err(e) => panic!("tempdir: {e}"),
    };
    let dir = held.path().to_path_buf();
    let mut command = ladder_binary();
    command.args([
        "--classes",
        "t6",
        "--direction",
        "magnify",
        "--profile",
        "contract",
        "--max-atoms",
        "1",
        "--terms",
        "8",
        "--no-search-probe",
        "--no-strips",
        "--out-dir",
    ]);
    command.arg(&dir);
    let (status, text) = output(command);
    assert_eq!(status, 0, "{text}");
    let ruling = text.find("PRIMARY: SCENE UNITS");
    let first_number = text.find("relL2");
    match (ruling, first_number) {
        (Some(r), Some(n)) => assert!(
            r < n,
            "the criterion convention must be ruled on BEFORE the first number, or every \
             table above it is uninterpretable"
        ),
        (_, _) => panic!("the banner or the table is missing entirely:\n{text}"),
    }
    for needed in [
        "SECONDARY: DEVICE PIXELS",
        "EVERY WAY A NUMBER BELOW COULD BE AN ARTEFACT",
        "CRITERION SCALING",
        "THE MISSING PREFILTER UNDER MINIFICATION",
        "THE FITTER FLOOR DIFFERS BY SCALE",
        "THE BASE SCALE PRIVILEGES THREE OF THE FOUR ARMS",
        "THE TRUTH ITSELF",
        "A FLAT DECAY FROM A BAD BASE IS NOT RESOLUTION INDEPENDENCE",
        "CARRIED CAVEATS",
        "THE FOUR ARMS ARE NOT SYMMETRIC",
    ] {
        assert!(
            text.contains(needed),
            "the run must name `{needed}` in its own banner: a zoom number whose confounds \
             are not listed beside it is a number nobody can act on\n{text}"
        );
    }
    assert!(
        dir.join("ladder-magnify.tsv").is_file(),
        "the ladder table must be written"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_boundary_reference_is_the_share_a_fixed_width_edge_takes_of_a_growing_image() {
    assert!(
        (boundary_localised_reference((64, 64), (64, 64)) - 1.0).abs() < 1.0e-12,
        "at its own base the reference must be exactly one, or the base row of every \
         ladder would be judged against a moving target"
    );
    for (base, side, want) in [
        ((64u32, 64u32), (256u32, 256u32), 0.5),
        ((64, 64), (512, 512), 0.35355339059327373),
        ((512, 512), (16, 16), 5.656854249492381),
        ((32, 32), (128, 128), 0.5),
    ] {
        let got = boundary_localised_reference(base, side);
        assert!(
            (got - want).abs() < 1.0e-12,
            "a boundary of fixed device width covers sqrt(base_side/side) of the image: \
             base {base:?} at {side:?} wanted {want} got {got}"
        );
    }
    let rectangular = boundary_localised_reference((64, 16), (256, 64));
    assert!(
        (rectangular - 0.5).abs() < 1.0e-12,
        "the reference must use the geometric mean side so a non-square viewport is not \
         judged on its width alone, got {rectangular}"
    );
}

#[test]
fn the_analytic_bar_decays_at_the_boundary_reference_and_its_error_never_moves() {
    let class = ScaleClass::Analytic(Target::<f64>::t2_straight_edge());
    let mut request = cheap(Direction::Magnify, &[(1, 1), (2, 1), (4, 1)]);
    request.wavelet_terms = 16;
    let rows = ladder_of(&class, &request);
    let analytic: Vec<&ScaleRow> = rows
        .iter()
        .filter(|r| r.arm == Arm3::AnalyticSdf.name() && !r.absent)
        .collect();
    assert_eq!(
        analytic.len(),
        3,
        "the analytic bar must be present at all three rungs or this test measures nothing"
    );
    let mut off_base = 0usize;
    let mut worst_error = f64::NEG_INFINITY;
    let mut best_error = f64::INFINITY;
    for row in analytic.iter() {
        worst_error = worst_error.max(row.max_abs);
        best_error = best_error.min(row.max_abs);
        if row.at_base {
            continue;
        }
        off_base += 1;
        assert!(
            (row.boundary_reference - 1.0).abs() > 0.1,
            "at {} the boundary reference is {}, indistinguishable from the full-field \
             reference of one, so this rung cannot tell the two apart",
            row.scale.label(),
            row.boundary_reference
        );
        let ratio = row.decay / row.boundary_reference;
        assert!(
            (ratio - 1.0).abs() < 0.10,
            "the analytic bar is a one-device-pixel boundary ramp at every scale, so its \
             decay must sit ON the boundary reference: at {} decay {} against reference {} \
             is a ratio of {ratio}",
            row.scale.label(),
            row.decay,
            row.boundary_reference
        );
    }
    assert_eq!(off_base, 2, "two rungs off the base or the check above never ran");
    assert!(
        worst_error / best_error < 1.5,
        "the analytic bar's max|e| must stay put across the ladder: that is what makes its \
         decay the DENOMINATOR moving rather than the description changing. Saw {best_error} \
         to {worst_error}"
    );
}

#[test]
fn an_error_spread_over_the_whole_field_decays_at_one_and_not_at_the_boundary_reference() {
    let class = ScaleClass::colour_field();
    let request = cheap(Direction::Magnify, &[(1, 1), (4, 1)]);
    let rows = ladder_of(&class, &request);
    let far: Vec<&ScaleRow> = rows
        .iter()
        .filter(|r| r.arm.starts_with("frozen-atoms") && !r.at_base && !r.absent)
        .collect();
    assert_eq!(far.len(), 1, "one off-base frozen row is what this test needs");
    for row in far.iter() {
        assert!(
            (row.boundary_reference - 0.5).abs() < 1.0e-12,
            "the 4x rung of a 1x base must carry a boundary reference of one half, got {}",
            row.boundary_reference
        );
        assert!(
            (row.decay - 1.0).abs() < 0.15,
            "a frozen atom set is wrong over the WHOLE colour field, not along a boundary, \
             so its decay must track the full-field reference of one and NOT the boundary \
             reference of {}: decay was {}",
            row.boundary_reference,
            row.decay
        );
        assert!(
            (row.decay / row.boundary_reference - 1.0).abs() > 0.5,
            "if a whole-field error also matched the boundary reference the two references \
             would not discriminate and the audit would be measuring nothing"
        );
    }
}

#[test]
fn the_metric_audit_names_the_rows_whose_decay_is_the_denominator_and_spares_the_rest() {
    let classes = vec![
        ScaleClass::Analytic(Target::<f64>::t2_straight_edge()),
        ScaleClass::colour_field(),
    ];
    let request = cheap(Direction::Magnify, &[(1, 1), (4, 1)]);
    let report = match run(&classes, &request) {
        Ok(r) => r,
        Err(e) => panic!("{e}"),
    };
    let flagged = report.decay_that_is_only_the_metric(0.10);
    assert!(
        flagged.iter().any(|(_, arm, _, _, _, _)| arm == &Arm3::AnalyticSdf.name()),
        "the analytic bar is the one arm whose decay IS the denominator; an audit that does \
         not name it is blind: {flagged:?}"
    );
    for (class, arm, _, scale, decay, reference) in flagged.iter() {
        assert!(
            (decay / reference - 1.0).abs() <= 0.10,
            "{class} {arm} at {} was flagged with decay {decay} against reference {reference}, \
             which is outside the tolerance it was flagged under",
            scale.label()
        );
    }
    let frozen_on_the_field = flagged.iter().any(|(class, arm, _, _, _, _)| {
        class.contains("colour-field") && arm.starts_with("frozen-atoms")
    });
    assert!(
        !frozen_on_the_field,
        "the frozen arm on the colour field is wrong over the whole field, so flagging it as \
         a denominator artefact would excuse a real measurement: {flagged:?}"
    );
    let base_rows = flagged.iter().any(|(_, _, _, scale, _, _)| *scale == request.base().unwrap());
    assert!(
        !base_rows,
        "the base rung has a reference of exactly one and a decay of exactly one, so flagging \
         it would fill the audit with rows that cannot say anything"
    );
}

#[test]
fn every_row_carries_a_boundary_reference_and_the_table_prints_it_beside_the_decay() {
    let class = ScaleClass::Analytic(Target::<f64>::t2_straight_edge());
    let request = cheap(Direction::Minify, &[(4, 1), (1, 1)]);
    let rows = ladder_of(&class, &request);
    assert!(!rows.is_empty(), "an empty ladder proves nothing");
    let columns = lights::fit::aspect3::SCALE_COLUMNS;
    let decay_at = columns.iter().position(|c| *c == "decay_from_own_base");
    let reference_at = columns
        .iter()
        .position(|c| *c == "decay_if_error_is_boundary_localised");
    match (decay_at, reference_at) {
        (Some(d), Some(r)) => assert_eq!(
            r,
            d + 1,
            "the reference must sit immediately beside the decay it qualifies, or a reader \
             takes the decay column alone and the artefact is back"
        ),
        (_, _) => panic!("the table lost one of the two columns: {columns:?}"),
    }
    for row in rows.iter() {
        let cells = lights::fit::aspect3::row_cells(row);
        assert_eq!(
            cells.len(),
            columns.len(),
            "every row must fill every column or the TSV is misaligned"
        );
        assert!(
            row.boundary_reference.is_finite() && row.boundary_reference > 0.0,
            "{} at {} carries no boundary reference",
            row.arm,
            row.scale.label()
        );
        if row.at_base {
            assert!(
                (row.boundary_reference - 1.0).abs() < 1.0e-12,
                "the base row's reference must be exactly one"
            );
        }
    }
}

#[test]
fn the_ladder_binary_stamps_only_the_table_it_wrote_and_leaves_a_foreign_one_alone() {
    let held = match tempfile::Builder::new().prefix("lights-p5-stamp-").tempdir() {
        Ok(d) => d,
        Err(e) => panic!("tempdir: {e}"),
    };
    let dir = held.path().to_path_buf();

    let foreign = dir.join("ladder-minify.tsv");
    let foreign_body = "#direction\tminify\nclass\tarm\nt2-straight-edge\tanalytic-sdf\n";
    if let Err(e) = std::fs::write(&foreign, foreign_body) {
        panic!("seeding the foreign table: {e}");
    }
    let unrelated = dir.join("someone-elses.tsv");
    if let Err(e) = std::fs::write(&unrelated, "a\tb\n1\t2\n") {
        panic!("seeding the unrelated table: {e}");
    }

    let mut command = ladder_binary();
    command.args([
        "--classes",
        "t6",
        "--direction",
        "magnify",
        "--profile",
        "contract",
        "--max-atoms",
        "1",
        "--terms",
        "8",
        "--no-search-probe",
        "--no-strips",
        "--out-dir",
    ]);
    command.arg(&dir);
    let (status, text) = output(command);
    assert_eq!(status, 0, "{text}");

    let written = dir.join("ladder-magnify.tsv");
    let written_text = match std::fs::read_to_string(&written) {
        Ok(t) => t,
        Err(e) => panic!("reading the table the run wrote: {e}"),
    };
    let first = written_text.lines().next().unwrap_or("");
    assert!(
        first.starts_with("#invocation\t"),
        "the table the run DID write must open with the command that wrote it, \
         and it opens with {first:?}"
    );
    assert!(
        first.contains("--direction magnify"),
        "the stamped line must be this run's argv verbatim: {first:?}"
    );

    for (path, expected) in [(&foreign, foreign_body), (&unrelated, "a\tb\n1\t2\n")] {
        let after = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) => panic!("reading {}: {e}", path.display()),
        };
        assert_eq!(
            after,
            expected,
            "{} was not written by this run, so this run must not put its \
             command on top of it. A stale table wearing a fresh #invocation is \
             a header that lies about its own body.",
            path.display()
        );
    }
}

#[test]
fn the_ladder_binary_writes_every_image_it_lists_and_lists_every_image_it_writes() {
    let held = match tempfile::Builder::new().prefix("lights-p5-strips-").tempdir() {
        Ok(d) => d,
        Err(e) => panic!("tempdir: {e}"),
    };
    let dir = held.path().to_path_buf();
    let mut command = ladder_binary();
    command.args([
        "--classes",
        "t6",
        "--direction",
        "magnify",
        "--profile",
        "contract",
        "--max-atoms",
        "1",
        "--terms",
        "8",
        "--no-search-probe",
        "--out-dir",
    ]);
    command.arg(&dir);
    let (status, text) = output(command);
    assert_eq!(status, 0, "{text}");

    let listed: Vec<&str> = text
        .lines()
        .filter(|l| l.trim_start().starts_with('/') && l.trim_end().ends_with(".png"))
        .map(|l| l.trim())
        .collect();
    assert!(
        !listed.is_empty(),
        "a run with strips on must list at least one image; the manifest is the only way a \
         reader finds the pictures:\n{text}"
    );
    for path in listed.iter() {
        let p = std::path::Path::new(path);
        assert!(p.is_absolute(), "the manifest must print absolute paths, got {path}");
        assert!(p.is_file(), "the manifest lists {path} but nothing was written there");
        let bytes = match std::fs::metadata(p) {
            Ok(m) => m.len(),
            Err(e) => panic!("{path}: {e}"),
        };
        assert!(bytes > 0, "{path} was written empty");
    }

    let canonical = |p: &std::path::Path| -> String {
        match std::fs::canonicalize(p) {
            Ok(c) => c.to_string_lossy().to_string(),
            Err(e) => panic!("{}: {e}", p.display()),
        }
    };
    let listed_real: Vec<String> =
        listed.iter().map(|l| canonical(std::path::Path::new(l))).collect();

    let mut on_disk: Vec<String> = Vec::new();
    let entries = match std::fs::read_dir(&dir) {
        Ok(e) => e,
        Err(e) => panic!("{}: {e}", dir.display()),
    };
    for entry in entries {
        let path = match entry {
            Ok(e) => e.path(),
            Err(e) => panic!("{e}"),
        };
        if path.extension().map(|x| x == "png").unwrap_or(false) {
            on_disk.push(canonical(&path));
        }
    }
    assert_eq!(
        on_disk.len(),
        listed_real.len(),
        "every PNG on disk must appear in the manifest and every manifest line must exist: \
         disk {on_disk:?} against manifest {listed_real:?}"
    );
    for path in on_disk.iter() {
        assert!(
            listed_real.iter().any(|l| l == path),
            "{path} was written but never listed, so a reader would never find it"
        );
    }
    assert!(
        text.contains(&format!("{} image(s) written.", listed.len())),
        "the run must count the images it wrote and the count must match the manifest:\n{text}"
    );
    assert!(
        text.contains("ladder-t6-glyph-like-magnify-contract-frozen.png"),
        "the frozen-atom ladder strip is the headline picture of this aspect and must be \
         emitted for every class:\n{text}"
    );

    let checkpoints: Vec<&str> = text
        .lines()
        .filter(|l| l.trim_start().starts_with('/') && l.trim_end().ends_with(".fit"))
        .map(|l| l.trim())
        .collect();
    assert_eq!(
        checkpoints.len(),
        1,
        "one class under one profile must produce exactly one frozen checkpoint; the manifest \
         lists {}:\n{text}",
        checkpoints.len()
    );
    assert!(
        text.contains(&format!("{} checkpoint(s) written.", checkpoints.len())),
        "the run must count the checkpoints it wrote and the count must match its own \
         manifest:\n{text}"
    );
    let mut fits_on_disk: Vec<String> = Vec::new();
    let entries = match std::fs::read_dir(&dir) {
        Ok(e) => e,
        Err(e) => panic!("{}: {e}", dir.display()),
    };
    for entry in entries {
        let path = match entry {
            Ok(e) => e.path(),
            Err(e) => panic!("{e}"),
        };
        if path.extension().map(|x| x == "fit").unwrap_or(false) {
            fits_on_disk.push(canonical(&path));
        }
    }
    assert_eq!(
        fits_on_disk.len(),
        checkpoints.len(),
        "the checkpoints on disk and the ones listed must be the same set: disk \
         {fits_on_disk:?} against manifest {checkpoints:?}"
    );
    for path in checkpoints.iter() {
        let real = canonical(std::path::Path::new(path));
        assert!(
            fits_on_disk.iter().any(|f| f == &real),
            "{path} was listed as a checkpoint and is not on disk"
        );
        match lights::fit::atoms_io::read_atoms(std::path::Path::new(path)) {
            Ok((atoms, viewport)) => {
                assert!(
                    !atoms.is_empty(),
                    "{path} holds no atoms, so the strips beside it cannot be redrawn from it"
                );
                assert!(
                    viewport.0 > 0 && viewport.1 > 0,
                    "{path} declares a zero viewport"
                );
            }
            Err(e) => panic!("{path} is listed as a checkpoint and does not read: {e}"),
        }
    }
    assert!(
        !listed.iter().any(|l| l.ends_with(".fit")),
        "a checkpoint appeared in the IMAGE manifest. It is not an image, and the image \
         manifest is what a reader counts pictures from:\n{text}"
    );
    assert!(
        text.contains("nearest-neighbour upscaled"),
        "a strip whose caption does not say the panels are nearest-neighbour upscaled invites \
         a reader to blame the resampler for the representation:\n{text}"
    );
    assert!(
        text.contains("ABSENT") || text.contains("magenta hatch"),
        "the captions must explain the absent panel, or a magenta block reads as a bug:\n{text}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_two_self_audits_name_the_project_caveat_they_size_and_never_the_wrong_one() {
    let held = match tempfile::Builder::new().prefix("lights-p5-audit-labels-").tempdir() {
        Ok(d) => d,
        Err(e) => panic!("tempdir: {e}"),
    };
    let dir = held.path().to_path_buf();
    let mut command = ladder_binary();
    command.args([
        "--classes",
        "t2",
        "--direction",
        "minify",
        "--profile",
        "both",
        "--max-atoms",
        "1",
        "--terms",
        "8",
        "--no-strips",
        "--out-dir",
    ]);
    command.arg(&dir);
    let (status, text) = output(command);
    assert_eq!(status, 0, "{text}");

    let prefilter = match text.find("THE MISSING PREFILTER UNDER MINIFICATION (ARTEFACT") {
        Some(i) => i,
        None => panic!("the prefilter audit lost its heading:\n{text}"),
    };
    let heading = match text[prefilter..].find("===\n") {
        Some(e) => &text[prefilter..prefilter + e],
        None => panic!("the prefilter heading never closes:\n{text}"),
    };
    assert!(
        heading.contains("C3"),
        "the prefilter is project caveat C3; a heading that does not say so leaves a reader \
         to guess: {heading}"
    );
    assert!(
        !heading.contains("C2"),
        "C2 is the BYTE comparison, which this run explicitly refuses to settle. Naming it \
         over the prefilter table claims a result the run does not have: {heading}"
    );
    assert!(
        !text.contains("CONFOUND 2,") && !text.contains("CONFOUND 3,"),
        "a bare `CONFOUND <n>` heading collides with the project's C1/C2/C3 caveats, where \
         C2 is bytes and C3 is the prefilter, and reads as the opposite of what it sizes:\n\
         {text}"
    );
    assert!(
        text.contains("symmetric-f32-provisional-see-C2"),
        "C2 must still appear exactly where it belongs, on the byte convention:\n{text}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_denominator_audit_declines_to_claim_where_the_two_references_are_indistinguishable() {
    let classes = vec![ScaleClass::Analytic(Target::<f64>::t2_straight_edge())];
    let request = cheap(Direction::Magnify, &[(1, 1), (2, 1)]);
    let report = match run(&classes, &request) {
        Ok(r) => r,
        Err(e) => panic!("{e}"),
    };
    let tight = report.decay_that_is_only_the_metric(0.10);
    let analytic_at_2x = tight
        .iter()
        .find(|(_, arm, _, scale, _, _)| arm == &Arm3::AnalyticSdf.name() && scale.factor() == 2.0);
    let reference = match analytic_at_2x {
        Some((_, _, _, _, _, r)) => *r,
        None => panic!(
            "at a tolerance of 0.10 the 2x rung must still be audited and the analytic bar \
             flagged, or the loose-tolerance half of this test proves nothing: {tight:?}"
        ),
    };
    assert!(
        (reference - 0.7071067811865476).abs() < 1.0e-9,
        "the 2x rung of a 1x base carries a boundary reference of 1/sqrt(2), got {reference}"
    );

    let slack = 0.35;
    assert!(
        (reference - 1.0).abs() <= slack,
        "this test needs a tolerance under which the boundary reference and the full-field \
         reference of one are NOT distinguishable at this rung; {reference} against 1.0 is \
         {} apart and the slack is {slack}",
        (reference - 1.0).abs()
    );
    let loose = report.decay_that_is_only_the_metric(slack);
    assert!(
        loose.is_empty(),
        "where the boundary reference is within the tolerance of the full-field reference the \
         audit cannot tell the two apart, so saying `this decay is only the denominator` is a \
         claim it has no standing to make. It must decline instead of filling itself with \
         rows that mean nothing: {loose:?}"
    );
}

#[test]
fn the_tsv_names_which_probe_each_verdict_belongs_to_and_says_nothing_where_a_probe_is_absent() {
    let columns = lights::fit::aspect3::SCALE_COLUMNS;
    let at = |name: &str| match columns.iter().position(|c| *c == name) {
        Some(i) => i,
        None => panic!("the table has no {name} column: {columns:?}"),
    };
    let transition_value = at("transition_10_90_scene");
    let transition_verdict = at("meets_transition_scene");
    let fwhm_value = at("fwhm_scene");
    let fwhm_verdict = at("meets_fwhm_scene");
    let request = cheap(Direction::Magnify, &[(1, 1), (2, 1)]);
    let mut seen_absent_transition = false;
    let mut seen_present_transition = false;
    for class in [
        ScaleClass::Analytic(Target::<f64>::t6_glyph_like()),
        ScaleClass::Analytic(Target::<f64>::t2_straight_edge()),
    ] {
        for row in ladder_of(&class, &request) {
            let cells = lights::fit::aspect3::row_cells(&row);
            assert_eq!(cells.len(), columns.len(), "the TSV is misaligned");
            for (value, verdict, probe) in [
                (transition_value, transition_verdict, "transition"),
                (fwhm_value, fwhm_verdict, "fwhm"),
            ] {
                let measured = cells[value].parse::<f64>().map(|v| v.is_finite()) == Ok(true);
                if measured {
                    assert!(
                        cells[verdict] == "yes" || cells[verdict] == "no",
                        "{} carries a {probe} of {} and must be judged, not dashed",
                        row.arm,
                        cells[value]
                    );
                } else {
                    assert_eq!(
                        cells[verdict], "-",
                        "{} carries NO {probe} probe ({}), so the verdict must say so; \
                         `no` reads as a failure the target was never asked to pass",
                        row.arm, cells[value]
                    );
                }
            }
            if cells[transition_value] == "nan" {
                seen_absent_transition = true;
            } else {
                seen_present_transition = true;
            }
        }
    }
    assert!(
        seen_absent_transition && seen_present_transition,
        "the case proves nothing unless it saw both a target with a transition probe and one without"
    );
}

#[test]
fn no_arm_legend_calls_the_refitted_fit_a_ceiling() {
    let every = [
        Independence::ContinuousFrozen,
        Independence::ContinuousRefit,
        Independence::ByConstruction,
        Independence::LatticeBound,
    ];
    for independence in every {
        let legend = independence.legend().to_ascii_lowercase();
        assert!(
            !legend.contains("ceiling"),
            "{} calls itself a ceiling: {legend}. A greedy fitter returns ONE \
             achievable description, not the best one, so a refit at a rung is a \
             FLOOR on atom quality exactly as C1 says every atom row is. It is \
             measured, not argued: in docs/lights/p5/ladder-magnify.tsv the \
             FROZEN set beats the fresh fit in 13 of the 24 magnification cells \
             of t2, t3 and t6, worst by 0.0963 relative L2 on t3 prefiltered at \
             4x, and in ladder-minify.tsv in 17 of 48 more. A column the frozen \
             arm beats in 30 of 80 published cells is not a ceiling, and \
             ceiling_violations() is named for the CLAIM it audits, not for a \
             property the arm has",
            independence.name()
        );
    }
    let refit = Independence::ContinuousRefit.legend().to_ascii_lowercase();
    assert!(
        refit.contains("floor"),
        "the refitted arm's legend must say what it IS - a greedy floor, the \
         honest control for the frozen arm - and not only what it is not: {refit}"
    );
}

#[test]
fn a_viewport_the_lattice_cannot_decompose_empties_its_column_instead_of_killing_the_run() {
    assert_eq!(
        max_levels(6, 6),
        0,
        "6x6 must be the case this test says it is: one level leaves a 3x3 \
         approximation, under the 4x4 floor, so the lattice has no dyadic \
         decomposition of it at all"
    );
    let class = ScaleClass::Continuous {
        name: "l9-six-square",
        viewport: (6, 6),
        sampler: Sampler::ColourField,
    };
    let request = cheap(Direction::Magnify, &[(1, 1), (2, 1)]);
    let ladder = match class_ladder(&class, &request) {
        Ok(l) => l,
        Err(e) => panic!(
            "a base viewport the lattice cannot decompose killed the entire run: {e}. \
             Every other arm reports 'this arm cannot express this target' as a RESULT - \
             the analytic arm returns None on a class with no signed distance and the \
             column comes out ABSENT - and the lattice arm must do the same. A hard \
             error here loses the atom arms, the analytic arm and both ladders for a \
             fact about ONE column"
        ),
    };
    let mut lattice_rows = 0usize;
    for row in ladder.rows.iter() {
        if !row.arm.starts_with("wavelet-") {
            continue;
        }
        lattice_rows += 1;
        assert!(
            row.absent,
            "{} at {} is present on a 6x6 base the transform cannot decompose",
            row.arm,
            row.scale.label()
        );
        assert_eq!(
            row.terms, 0,
            "an absent {} must bill no terms at {}",
            row.arm,
            row.scale.label()
        );
        assert!(
            !row.relative_l2.is_finite(),
            "an absent {} must carry no error at {}, got {}",
            row.arm,
            row.scale.label(),
            row.relative_l2
        );
    }
    assert_eq!(
        lattice_rows, 4,
        "two transforms over two rungs is four lattice rows; the column has to be \
         EMPTIED, not dropped, or the table stops saying the arm was asked"
    );
    assert!(
        ladder
            .rows
            .iter()
            .any(|r| r.arm.starts_with("frozen-atoms") && !r.absent && r.relative_l2.is_finite()),
        "the atom arms must still have run: emptying one column must not empty the run"
    );
    let dyadic = ScaleClass::Continuous {
        name: "l9-sixteen-square",
        viewport: (16, 16),
        sampler: Sampler::ColourField,
    };
    let fine = match class_ladder(&dyadic, &request) {
        Ok(l) => l,
        Err(e) => panic!("{e}"),
    };
    assert!(
        fine
            .rows
            .iter()
            .any(|r| r.arm.starts_with("wavelet-") && !r.absent && r.terms > 0),
        "on a 16x16 base the lattice arm must still be PRESENT, or the absent path \
         has swallowed the arm rather than reporting one viewport it cannot express"
    );
}
