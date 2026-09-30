#![cfg(feature = "fit")]
use lights::dispatch::Collect;
use lights::error::{DegenerateReason, LightsError};
use lights::fit::atom2d::{
    Atom2d, DZ, Z0, camera, depth_of, image_to_world, pixel_centre, shifted_camera, to_atom,
    to_scene, world_to_image,
};
use lights::fit::coverage::{coverage_budget, coverage_image_to_colour, exact_coverage};
use lights::fit::greedy::{
    DEFAULT_REFINE_TAIL, ELONGATION_GAIN, FitRequest, Initialisation, MAX_OPACITY, SEED_SIGMA, fit,
    fit_budget, render_atoms, render_or_background, residual_energy, seed_atom, seed_pixel,
    structure_tensor, tensor_axis,
};
use lights::fit::edge::{Criterion, TargetMetric, measure_target};
use lights::fit::nterm::{MAX_ATOMS as NTERM_MAX_ATOMS, ladder_upto};
use lights::fit::target::{EDGE_DEGREES, Target};
use lights::image::FloatImage;
use lights::kernel::{Beta, KernelFamily};
use lights::numerics::vector::{Rgb, Vec2};
use lights::settings::{PIXEL_BOX_VARIANCE, SharedSettings};

const TILE: u32 = 16;
const CELL_LIMIT: u64 = 64_000_000;
const RENDER_LIMIT: u64 = 40_000_000;
const QUADRATURE_DEPTH: u32 = 10;

fn truth_of(target: &Target<f64>) -> FloatImage<f64> {
    let mut budget = coverage_budget(CELL_LIMIT);
    let (coverage, _) = match exact_coverage(target, QUADRATURE_DEPTH, &mut budget) {
        Ok(v) => v,
        Err(e) => panic!("the ground truth for {} did not compute: {e}", target.name),
    };
    coverage_image_to_colour(target, &coverage)
}

fn fitted(
    target: &Target<f64>,
    truth: &FloatImage<f64>,
    kernel: KernelFamily,
    max_atoms: usize,
) -> lights::fit::greedy::FitOutcome {
    let mut request = FitRequest::new(kernel, SharedSettings::<f64>::prefiltered());
    request.max_atoms = max_atoms;
    request.tile = TILE;
    let mut budget = fit_budget(RENDER_LIMIT);
    match fit(target, truth, &request, &mut budget) {
        Ok(o) => o,
        Err(e) => panic!("the fit of {} under {} failed: {e}", target.name, kernel.name()),
    }
}

fn rendered(
    atoms: &[Atom2d],
    viewport: (u32, u32),
    kernel: KernelFamily,
    settings: &SharedSettings<f64>,
) -> FloatImage<f64> {
    match render_or_background(atoms, &camera(viewport.0, viewport.1), kernel, settings, TILE) {
        Ok(i) => i,
        Err(e) => panic!("rendering {} atoms failed: {e}", atoms.len()),
    }
}

fn energy(truth: &FloatImage<f64>, image: &FloatImage<f64>) -> f64 {
    match residual_energy(truth, image) {
        Ok(v) => v,
        Err(e) => panic!("the residual energy did not compute: {e}"),
    }
}

fn empty_energy(truth: &FloatImage<f64>, settings: &SharedSettings<f64>) -> f64 {
    energy(
        truth,
        &FloatImage::filled(truth.width, truth.height, settings.background),
    )
}

fn angle_between(a: f64, b: f64) -> f64 {
    let raw = (a - b).rem_euclid(core::f64::consts::PI);
    raw.min(core::f64::consts::PI - raw)
}

fn worst_channel_gap(a: &FloatImage<f64>, b: &FloatImage<f64>) -> f64 {
    let mut worst = 0.0f64;
    for (p, q) in a.pixels.iter().zip(b.pixels.iter()) {
        worst = worst
            .max((p.x - q.x).abs())
            .max((p.y - q.y).abs())
            .max((p.z - q.z).abs());
    }
    worst
}

#[test]
fn the_fitter_is_bit_deterministic() {
    let target = Target::<f64>::t4_curve();
    let truth = truth_of(&target);
    let first = fitted(&target, &truth, KernelFamily::Gaussian, 3);
    let second = fitted(&target, &truth, KernelFamily::Gaussian, 3);
    assert_eq!(
        first.atoms.len(),
        second.atoms.len(),
        "two identical requests produced different atom counts, so nothing else \
         in this suite can be trusted to reproduce"
    );
    for (i, (a, b)) in first.atoms.iter().zip(second.atoms.iter()).enumerate() {
        assert_eq!(
            a.bits(),
            b.bits(),
            "atom {i} differs between two identical fits: {a:?} versus {b:?}. \
             The fitter is RNG-free by construction, so any difference is a \
             read of uninitialised order somewhere in the render path"
        );
    }
    assert_eq!(
        first.renders, second.renders,
        "the two fits consumed different amounts of budget ({} versus {}), so \
         the search path itself is not reproducible even where the answer is",
        first.renders, second.renders
    );

    let single = SharedSettings::<f64>::prefiltered();
    let mut pooled = single;
    pooled.threads = 0;
    let one = rendered(&first.atoms, target.viewport, KernelFamily::Gaussian, &single);
    let many = rendered(&first.atoms, target.viewport, KernelFamily::Gaussian, &pooled);
    for (i, (p, q)) in one.pixels.iter().zip(many.pixels.iter()).enumerate() {
        assert_eq!(
            (p.x.to_bits(), p.y.to_bits(), p.z.to_bits()),
            (q.x.to_bits(), q.y.to_bits(), q.z.to_bits()),
            "pixel {i} of the fitted scene differs between threads=1 and \
             threads=0. The fitter searches at threads=1; if the pooled render \
             is not bit-identical then every number the harness reports depends \
             on how many cores happened to be free"
        );
    }
}

#[test]
fn the_residual_never_increases_when_an_atom_is_added() {
    let settings = SharedSettings::<f64>::prefiltered();
    for target in Target::<f64>::all() {
        let truth = truth_of(&target);
        for kernel in KernelFamily::FIT_SWEEP {
            let outcome = fitted(&target, &truth, kernel, 5);
            let mut previous = empty_energy(&truth, &settings);
            for step in outcome.steps.iter() {
                assert!(
                    step.residual_energy < previous,
                    "{} under {}: adding atom {} raised the residual energy from \
                     {previous} to {}. A greedy fitter that can go backwards is \
                     measuring its own instability, and the atom-count column \
                     would then be a property of the search rather than of the \
                     representation",
                    target.name,
                    kernel.name(),
                    step.atoms,
                    step.residual_energy
                );
                assert!(
                    step.residual_energy.is_finite() && step.residual_energy >= 0.0,
                    "{} under {}: step {} reported a residual energy of {}",
                    target.name,
                    kernel.name(),
                    step.atoms,
                    step.residual_energy
                );
                previous = step.residual_energy;
            }
            let image = rendered(&outcome.atoms, target.viewport, kernel, &settings);
            match outcome.steps.last() {
                Some(last) => assert!(
                    (energy(&truth, &image) - last.residual_energy).abs() <= 1.0e-9,
                    "{} under {}: the recorded final energy {} does not match a \
                     fresh render of the emitted atoms ({}). The step list would \
                     then be describing a scene the caller never receives",
                    target.name,
                    kernel.name(),
                    last.residual_energy,
                    energy(&truth, &image)
                ),
                None => assert!(
                    outcome.improvement_exhausted,
                    "{} under {} placed no atom at all and did not say why",
                    target.name,
                    kernel.name()
                ),
            }
        }
    }
}

#[test]
fn the_budget_fails_hard_rather_than_returning_a_short_answer() {
    let target = Target::<f64>::t4_curve();
    let truth = truth_of(&target);
    let mut request = FitRequest::new(KernelFamily::Gaussian, SharedSettings::<f64>::prefiltered());
    request.max_atoms = 4;
    request.tile = TILE;
    for limit in [0u64, 1, 5, 50, 500] {
        let mut budget = fit_budget(limit);
        match fit(&target, &truth, &request, &mut budget) {
            Ok(o) => panic!(
                "a budget of {limit} renders was enough to return {} atoms. A \
                 fitter that silently truncates its own search reports an atom \
                 count that is a budget artefact",
                o.atoms.len()
            ),
            Err(LightsError::BudgetExhausted { site, limit: seen }) => {
                assert_eq!(site, "fit::greedy::fit");
                assert_eq!(seen, limit);
            }
            Err(e) => panic!("a budget of {limit} produced the wrong failure: {e}"),
        }
    }
    let mut generous = fit_budget(RENDER_LIMIT);
    match fit(&target, &truth, &request, &mut generous) {
        Ok(o) => assert!(
            !o.atoms.is_empty() && generous.used() > 500,
            "the generous run used {} renders for {} atoms, so the tiny-budget \
             refusals above may not have been measuring the budget at all",
            generous.used(),
            o.atoms.len()
        ),
        Err(e) => panic!("the generous budget failed: {e}"),
    }
}

#[test]
fn every_emitted_atom_survives_scene_push() {
    let settings = SharedSettings::<f64>::prefiltered();
    for target in Target::<f64>::all() {
        let truth = truth_of(&target);
        for kernel in KernelFamily::FIT_SWEEP {
            let outcome = fitted(&target, &truth, kernel, 4);
            for (index, a) in outcome.atoms.iter().enumerate() {
                assert!(
                    a.is_finite(),
                    "{} under {}: atom {index} carries a non-finite parameter: {a:?}",
                    target.name,
                    kernel.name()
                );
                assert!(
                    a.opacity > settings.blend.alpha_cutoff && a.opacity <= MAX_OPACITY,
                    "{} under {}: atom {index} has opacity {}, outside the box \
                     the fitter is allowed to search",
                    target.name,
                    kernel.name(),
                    a.opacity
                );
                let atom = to_atom(a, index, target.viewport);
                match atom.validate(settings.scale_floor) {
                    Ok(()) => (),
                    Err(e) => panic!(
                        "{} under {}: atom {index} would be refused by the \
                         renderer it was fitted with: {e}",
                        target.name,
                        kernel.name()
                    ),
                }
            }
            match to_scene(&outcome.atoms, target.viewport, settings.scale_floor) {
                Ok(s) => assert_eq!(s.len(), outcome.atoms.len()),
                Err(e) => panic!(
                    "{} under {}: the emitted atoms do not form a scene: {e}",
                    target.name,
                    kernel.name()
                ),
            }
        }
    }
}

#[test]
fn a_flat_fill_is_reproduced_exactly_under_normalisation() {
    let settings = SharedSettings::<f64>::prefiltered();
    let target = Target::<f64>::t1_flat_fill();
    let truth = truth_of(&target);
    let outcome = fitted(&target, &truth, KernelFamily::GaussianNormalised, 6);
    let last = match outcome.steps.last() {
        Some(s) => *s,
        None => panic!("the flat fill was fitted with no atoms at all"),
    };
    assert!(
        last.atoms <= 2,
        "the flat fill took {} atoms to reach energy {}. Under a partition of \
         unity two overlapping atoms of the ink colour already resolve to that \
         colour exactly, so more than two is a fitter defect and not a cost of \
         the representation",
        last.atoms,
        last.residual_energy
    );
    assert!(
        last.residual_energy <= 1.0e-20,
        "the flat fill settled at residual energy {} on {} atoms. Alpha \
         compositing provably cannot reach zero here, normalisation can, and \
         this is the end-to-end form of that difference",
        last.residual_energy,
        last.atoms
    );
    let image = rendered(
        &outcome.atoms,
        target.viewport,
        KernelFamily::GaussianNormalised,
        &settings,
    );
    assert!(
        worst_channel_gap(&truth, &image) <= 1.0e-12,
        "the flat fill's worst channel gap is {}, so the zero energy above is \
         an accident of summation rather than an exact reproduction",
        worst_channel_gap(&truth, &image)
    );
}

#[test]
fn the_closed_form_pair_is_measured_beside_the_fitters_two_atoms() {
    let settings = SharedSettings::<f64>::prefiltered();
    let target = Target::<f64>::t2_straight_edge();
    let truth = truth_of(&target);
    let empty = empty_energy(&truth, &settings);
    let (sine, cosine) = EDGE_DEGREES.to_radians().sin_cos();
    let normal = Vec2::new(cosine, sine);
    let along = Vec2::new(-normal.y, normal.x);
    let angle = along.y.atan2(along.x);
    let centre = Vec2::new(32.0, 32.0);
    let needle = 148.0f64;

    let pair_at = |sigma: f64, separation: f64| -> [Atom2d; 2] {
        [
            Atom2d::from_sigma(
                centre - normal * (separation * 0.5),
                Vec2::new(needle, sigma),
                angle,
                MAX_OPACITY,
                Rgb::new(1.0, 1.0, 1.0),
            ),
            Atom2d::from_sigma(
                centre + normal * (separation * 0.5),
                Vec2::new(needle, sigma),
                angle,
                MAX_OPACITY,
                Rgb::new(0.0, 0.0, 0.0),
            ),
        ]
    };

    let mut best = (f64::INFINITY, 0.0f64, 0.0f64);
    for si in 1..=10 {
        for di in 1..=10 {
            let sigma = 0.5 * si as f64;
            let separation = 0.5 * di as f64;
            let image = rendered(
                &pair_at(sigma, separation),
                target.viewport,
                KernelFamily::GaussianNormalised,
                &settings,
            );
            let e = energy(&truth, &image);
            if e < best.0 {
                best = (e, sigma, separation);
            }
        }
    }
    let best_width = 2.0 * 9.0f64.ln() * best.1 * best.1 / best.2;

    let criterion_sigma = 0.5f64;
    let criterion_separation = 1.5f64;
    let criterion_width =
        2.0 * 9.0f64.ln() * criterion_sigma * criterion_sigma / criterion_separation;
    let criterion_image = rendered(
        &pair_at(criterion_sigma, criterion_separation),
        target.viewport,
        KernelFamily::GaussianNormalised,
        &settings,
    );
    let criterion_energy = energy(&truth, &criterion_image);
    assert!(
        criterion_width < 1.0,
        "the pair chosen to meet the one-pixel criterion has closed-form width \
         {criterion_width}, so this test is not probing what it claims to"
    );
    assert!(
        criterion_energy >= 0.9 * empty,
        "the pair whose closed-form 10-90 width is {criterion_width} px reaches \
         residual energy {criterion_energy} against an empty-scene energy of \
         {empty}. It was expected to be nearly as bad as painting nothing: \
         under this crate's Normalised the resolve is a coverage mix, so where \
         the total weight falls below one the pixel fades to the background \
         instead of following the Image-GS ratio, and a pair sharp enough to \
         meet the criterion carries almost no weight anywhere. If this \
         assertion now fails the blend rule changed and the finding below must \
         be re-derived"
    );
    assert!(
        best_width > 4.0,
        "the best two-atom pair over the sweep has closed-form width \
         {best_width} px at sigma {} and separation {}. It was expected to be \
         far above the one-pixel criterion, because energy is minimised by \
         carrying weight, not by sharpening",
        best.1,
        best.2
    );

    let outcome = fitted(&target, &truth, KernelFamily::GaussianNormalised, 2);
    let fitter_two = match outcome.steps.last() {
        Some(s) => s.residual_energy,
        None => panic!("the fitter placed no atom on the straight edge"),
    };
    assert_eq!(
        outcome.atoms.len(),
        2,
        "the fitter stopped at {} atoms on the straight edge, so there is no \
         two-atom number to set beside the closed form",
        outcome.atoms.len()
    );
    assert!(
        fitter_two < best.0 / 8.0,
        "the fitter's two atoms reach residual energy {fitter_two} while the \
         best closed-form pair over the sweep reaches {} (empty scene {empty}). \
         The closed form beating the fitter would make every atom count in the \
         report a fitter artefact rather than a cost of the representation, so \
         this margin is the guard on that reading",
        best.0
    );
}

#[test]
fn the_same_edge_under_over_is_measured_and_not_assumed() {
    let settings = SharedSettings::<f64>::prefiltered();
    let target = Target::<f64>::t2_straight_edge();
    let truth = truth_of(&target);
    let empty = empty_energy(&truth, &settings);
    let over = fitted(&target, &truth, KernelFamily::Gaussian, 4);
    let normalised = fitted(&target, &truth, KernelFamily::GaussianNormalised, 4);
    assert_eq!(over.steps.len(), 4);
    assert_eq!(normalised.steps.len(), 4);
    for (a, b) in over.steps.iter().zip(normalised.steps.iter()) {
        assert!(
            a.residual_energy.is_finite() && b.residual_energy.is_finite(),
            "one of the two blend arms produced a non-finite energy at {} atoms",
            a.atoms
        );
        assert!(
            a.residual_energy < empty && b.residual_energy < empty,
            "at {} atoms one arm is no better than an empty scene ({empty}): \
             over {} normalised {}",
            a.atoms,
            a.residual_energy,
            b.residual_energy
        );
    }
    let last_over = over.steps[3].residual_energy;
    let last_normalised = normalised.steps[3].residual_energy;
    assert!(
        (last_over - last_normalised).abs() > 1.0e-9,
        "the two blend rules reached identical residual energy {last_over} on \
         the straight edge. No atom count in this build is meaningful if the \
         blend axis does not move the number"
    );
}

#[test]
fn the_structure_tensor_seed_lies_along_the_thin_stroke_and_the_isotropic_seed_does_not() {
    let settings = SharedSettings::<f64>::prefiltered();
    let target = Target::<f64>::t5_thin_stroke();
    let truth = truth_of(&target);
    let background = FloatImage::filled(target.viewport.0, target.viewport.1, settings.background);
    let residual = match truth.difference(&background) {
        Ok(v) => v,
        Err(e) => panic!("the opening residual did not compute: {e}"),
    };
    let seed = match seed_pixel(&residual) {
        Some(s) => s,
        None => panic!("the opening residual has no maximum, so there is nothing to seed on"),
    };
    let (xx, xy, yy) = structure_tensor(&residual, seed);
    let (along, coherence) = tensor_axis(xx, xy, yy);
    let stroke = EDGE_DEGREES.to_radians();
    let gap = angle_between(along, stroke).to_degrees();
    let across_gap = angle_between(along + core::f64::consts::FRAC_PI_2, stroke).to_degrees();
    assert!(
        gap < 25.0,
        "the structure tensor put the seed's long axis {gap} degrees from the \
         stroke it was seeded on (tensor {xx}, {xy}, {yy}). An atom that does \
         not stretch ALONG a boundary is why cost would scale with length \
         instead of with perimeter and curvature"
    );
    assert!(
        across_gap - gap > 30.0,
        "the along-edge axis is {gap} degrees from the stroke and the \
         perpendicular is {across_gap}: the two are too close for this test to \
         tell a minor eigenvector from a major one"
    );
    assert!(
        coherence > 0.5,
        "the residual over the seed window has coherence {coherence}, so the \
         tensor is nearly isotropic there and the elongation it produces is noise"
    );
    let elongation = 1.0 + ELONGATION_GAIN * coherence;
    assert!(
        elongation * elongation > 4.0,
        "the seed's anisotropy is {}, which is not a needle by any reading",
        elongation * elongation
    );
    assert!(
        (SEED_SIGMA * elongation) * (SEED_SIGMA / elongation) - SEED_SIGMA * SEED_SIGMA
            < 1.0e-12,
        "the elongation changed the seed's area as well as its shape, which \
         confounds the initialisation with the scale"
    );

    let tensor_seed = seed_atom(
        &truth,
        &residual,
        seed,
        Initialisation::StructureTensor,
        &settings,
    );
    let round_seed = seed_atom(&truth, &residual, seed, Initialisation::Isotropic, &settings);
    assert!(
        tensor_seed.anisotropy() > 4.0,
        "the atom the fitter actually seeds has anisotropy {} on a one-pixel \
         stroke. tensor_axis may report a direction, but if seed_atom does not \
         BUILD the needle then nothing downstream stretches along anything",
        tensor_seed.anisotropy()
    );
    assert_eq!(
        round_seed.anisotropy(),
        1.0,
        "the isotropic seed is not isotropic, so the two initialisations are \
         not the contrast this test claims to draw"
    );
    let long = tensor_seed.long_axis();
    assert!(
        angle_between(long.y.atan2(long.x), stroke).to_degrees() < 25.0,
        "the seeded atom's long axis is {} degrees from the stroke",
        angle_between(long.y.atan2(long.x), stroke).to_degrees()
    );
    assert_eq!(
        (tensor_seed.centre.x, tensor_seed.centre.y),
        (round_seed.centre.x, round_seed.centre.y),
        "the two initialisations disagree about WHERE to seed as well as about \
         what shape to seed, so the shape comparison is confounded"
    );
}

#[test]
fn the_initialisation_is_absorbed_by_the_refinement_and_the_gap_is_reported() {
    let target = Target::<f64>::t5_thin_stroke();
    let truth = truth_of(&target);
    let measure = |initialisation: Initialisation, iterations: u32| -> f64 {
        let mut request =
            FitRequest::new(KernelFamily::GaussianNormalised, SharedSettings::<f64>::prefiltered());
        request.max_atoms = 4;
        request.tile = TILE;
        request.initialisation = initialisation;
        request.refine_iterations = iterations;
        request.refine_tail = 0;
        let mut budget = fit_budget(RENDER_LIMIT);
        let outcome = match fit(&target, &truth, &request, &mut budget) {
            Ok(o) => o,
            Err(e) => panic!("the {} fit failed: {e}", initialisation.name()),
        };
        match outcome.steps.last() {
            Some(s) => s.residual_energy,
            None => panic!("the {} fit placed no atom", initialisation.name()),
        }
    };
    let tensor_cold = measure(Initialisation::StructureTensor, 1);
    let isotropic_cold = measure(Initialisation::Isotropic, 1);
    let tensor_warm = measure(Initialisation::StructureTensor, 6);
    let isotropic_warm = measure(Initialisation::Isotropic, 6);
    assert!(
        tensor_cold < isotropic_cold / 1.5,
        "with one refinement pass the structure tensor reached {tensor_cold} and \
         an isotropic seed reached {isotropic_cold} on a one-pixel stroke. The \
         tensor was expected to be at least half again better, because that is \
         the whole mechanism by which an atom stretches along a thin feature"
    );
    assert!(
        (tensor_warm - isotropic_warm).abs() < 0.5 * tensor_warm.max(isotropic_warm),
        "with six refinement passes the structure tensor reached {tensor_warm} \
         and an isotropic seed reached {isotropic_warm}. This assertion records \
         the measured finding that a strong coordinate descent ABSORBS the \
         initialisation: the tensor buys a better first guess, not a better \
         answer. Any report that credits the tensor for the final atom count is \
         crediting the wrong part of the fitter"
    );
    assert!(
        tensor_warm.min(isotropic_warm) < tensor_cold.max(isotropic_cold),
        "six refinement passes did not beat one on either arm (tensor \
         {tensor_cold} to {tensor_warm}, isotropic {isotropic_cold} to \
         {isotropic_warm}), so the comparison above is not measuring \
         refinement at all"
    );
    assert!(
        tensor_cold.is_finite()
            && isotropic_cold.is_finite()
            && tensor_warm.is_finite()
            && isotropic_warm.is_finite(),
        "one of the four measured energies is not a number: tensor \
         {tensor_cold}/{tensor_warm}, isotropic {isotropic_cold}/{isotropic_warm}"
    );
    assert!(
        tensor_warm > tensor_cold * 0.1,
        "the four measured energies are tensor {tensor_cold} at one pass and \
         {tensor_warm} at six, isotropic {isotropic_cold} at one pass and \
         {isotropic_warm} at six, with the tail refinement switched off so that \
         only the initialisation and the per-atom effort vary. Refining the \
         CURRENT atom harder is not monotone in the FINAL result, because a \
         better fit to today's residual can leave a worse configuration for the \
         atoms that follow. That is greedy myopia, it is a property of matching \
         pursuit rather than of the representation, and it is the reason the \
         report carries the whole quality-versus-atom-count curve instead of a \
         single first-pass count"
    );
}

pub const RECOVERY_CEILING_PROFILE: &str = "prefiltered";
pub const RECOVERY_CEILING_HEADROOM: f64 = 1.15;
pub const RECOVERY_QUANTISATION_STEP: f64 = 1.0 / 255.0;
pub const SEARCH_LIMITED_RECOVERY_B_HALF: f64 = 5.30 / 255.0;
pub const SEARCH_LIMITED_RECOVERY_B_TWO: f64 = 1.52 / 255.0;
pub const SEARCH_LIMITED_RECOVERY_B_FOUR: f64 = 1.49 / 255.0;
pub const SEARCH_LIMITED_RECOVERY_B_EIGHT: f64 = 4.49 / 255.0;

fn recovery_ceiling(kernel: KernelFamily) -> f64 {
    match kernel {
        KernelFamily::Gaussian => RECOVERY_QUANTISATION_STEP,
        KernelFamily::GaussianNormalised => RECOVERY_QUANTISATION_STEP,
        KernelFamily::SuperGaussian(Beta::One) => RECOVERY_QUANTISATION_STEP,
        KernelFamily::SuperGaussian(Beta::Half) => SEARCH_LIMITED_RECOVERY_B_HALF,
        KernelFamily::SuperGaussian(Beta::Two) => SEARCH_LIMITED_RECOVERY_B_TWO,
        KernelFamily::SuperGaussian(Beta::Four) => SEARCH_LIMITED_RECOVERY_B_FOUR,
        KernelFamily::SuperGaussian(Beta::Eight) => SEARCH_LIMITED_RECOVERY_B_EIGHT,
        KernelFamily::CompactBump => RECOVERY_QUANTISATION_STEP,
        KernelFamily::GaborColour(_)
        | KernelFamily::GaborDensity(_)
        | KernelFamily::GaussHermiteColour(_)
        | KernelFamily::GaussHermiteDensity(_) => panic!(
            "{} has no measured recovery ceiling. Every constant in this table \
             was measured under one profile against a gaussian-tuned search, \
             and the modulated families have never been through it — the \
             fitter has no knob for a carrier or a hermite order (L-135). They \
             are excluded from KernelFamily::FIT_SWEEP for that reason, so \
             reaching this arm means a fit sweep was widened without measuring \
             the ceilings it now needs",
            kernel.name()
        ),
    }
}

#[test]
fn the_recovery_ceiling_is_a_measured_property_of_a_gaussian_tuned_search() {
    let settings = SharedSettings::<f64>::prefiltered();
    assert_eq!(
        RECOVERY_CEILING_PROFILE, "prefiltered",
        "EVERY NUMBER IN THE CEILING TABLE WAS MEASURED UNDER ONE PROFILE and the \
         table has to say which, because the prefilter changes the very thing the \
         ceiling measures. If this case is ever moved to contract(), the constants \
         must be re-measured, not carried over"
    );
    let target = Target::<f64>::t4_curve();
    let known = Atom2d::from_sigma(
        pixel_centre(16, 15),
        Vec2::new(3.0, 1.5),
        0.5,
        0.8,
        Rgb::new(0.9, 0.4, 0.2),
    );
    let mut tight = 0usize;
    let mut loose = 0usize;
    for kernel in KernelFamily::FIT_SWEEP {
        let truth = rendered(&[known], target.viewport, kernel, &settings);
        let mut request = FitRequest::new(kernel, settings);
        request.max_atoms = 1;
        request.tile = TILE;
        let mut budget = fit_budget(RENDER_LIMIT);
        let outcome = match fit(&target, &truth, &request, &mut budget) {
            Ok(o) => o,
            Err(e) => panic!("{}: {e}", kernel.name()),
        };
        let image = rendered(&outcome.atoms, target.viewport, kernel, &settings);
        let gap = worst_channel_gap(&truth, &image);
        if recovery_ceiling(kernel) > 1.0 / 255.0 {
            loose += 1;
            let ceiling = recovery_ceiling(kernel);
            assert!(
                gap <= ceiling * RECOVERY_CEILING_HEADROOM,
                "{}: the loose ceiling is a MEASUREMENT, {ceiling} worst-channel, and \
                 this curve now recovers to {gap} — past the {RECOVERY_CEILING_HEADROOM}x \
                 headroom. A ceiling with no upper bound is an exemption, not a \
                 measurement: it would let a curve regress by any amount and still \
                 pass. Re-measure the constant and say what moved, or fix the search",
                kernel.name()
            );
            assert!(
                gap >= 1.0 / 255.0,
                "{} is listed as needing the loose ceiling but now recovers its \
                 own known atom to within 1/255 ({gap}). The loose ceiling is a \
                 MEASUREMENT of a gaussian-tuned coordinate ladder failing on a \
                 curve it was not tuned for, not a licence to be sloppy: if the \
                 search has improved, move this curve back to the tight ceiling \
                 rather than leaving a dead exemption behind",
                kernel.name()
            );
        } else {
            tight += 1;
            assert!(gap < 1.0 / 255.0, "{}: {gap}", kernel.name());
        }
    }
    assert_eq!(
        (tight, loose),
        (4, 4),
        "the split between curves the greedy search reaches a known optimum on \
         and curves it does not IS the confound this exemption exists to make \
         visible, so its size is asserted rather than left to drift"
    );
}

#[test]
fn a_single_gaussian_target_is_recovered_by_one_atom() {
    let settings = SharedSettings::<f64>::prefiltered();
    let target = Target::<f64>::t4_curve();
    let known = Atom2d::from_sigma(
        pixel_centre(16, 15),
        Vec2::new(3.0, 1.5),
        0.5,
        0.8,
        Rgb::new(0.9, 0.4, 0.2),
    );
    for kernel in KernelFamily::FIT_SWEEP {
        let truth = rendered(&[known], target.viewport, kernel, &settings);
        let opening = empty_energy(&truth, &settings);
        let mut request = FitRequest::new(kernel, settings);
        request.max_atoms = 1;
        request.tile = TILE;
        let mut budget = fit_budget(RENDER_LIMIT);
        let outcome = match fit(&target, &truth, &request, &mut budget) {
            Ok(o) => o,
            Err(e) => panic!("the single-atom recovery under {} failed: {e}", kernel.name()),
        };
        assert_eq!(
            outcome.atoms.len(),
            1,
            "the recovery under {} emitted {} atoms for a target that is one atom",
            kernel.name(),
            outcome.atoms.len()
        );
        let image = rendered(&outcome.atoms, target.viewport, kernel, &settings);
        let gap = worst_channel_gap(&truth, &image);
        assert!(
            gap < recovery_ceiling(kernel),
            "under {} the fitter's single atom differs from the single atom it \
             was handed by {gap} at its worst pixel ({} of 255), past this \
             curve's ceiling of {} of 255. A fitter that cannot reach a known \
             optimum where one exists measures itself rather than the \
             representation, and every atom count in the report would then be a \
             fitter number",
            kernel.name(),
            gap * 255.0,
            recovery_ceiling(kernel) * 255.0
        );
        assert!(
            outcome.steps[0].residual_energy < opening / 1000.0,
            "under {} the recovery only reduced the residual energy from \
             {opening} to {}",
            kernel.name(),
            outcome.steps[0].residual_energy
        );
    }
}

#[test]
fn the_coordinate_bridge_puts_an_atom_at_the_pixel_the_target_names() {
    let settings = SharedSettings::<f64>::prefiltered();
    let viewport = (32u32, 32u32);
    let sigma = 0.35f64;
    let expected_peak =
        sigma * sigma / (sigma * sigma + PIXEL_BOX_VARIANCE) * MAX_OPACITY;
    for (px, py) in [(0u32, 0u32), (20, 8), (8, 20), (31, 31), (16, 16)] {
        let spot = Atom2d::from_sigma(
            pixel_centre(px, py),
            Vec2::new(sigma, sigma),
            0.0,
            MAX_OPACITY,
            Rgb::new(1.0, 1.0, 1.0),
        );
        let image = rendered(&[spot], viewport, KernelFamily::Gaussian, &settings);
        let (value, bx, by) = image.max_abs();
        assert_eq!(
            (bx, by),
            (px, py),
            "an atom declared at image coordinate {:?} peaked at pixel ({bx},{by}). \
             Targets, probes and atoms all live in image pixel coordinates and \
             the single shift to world coordinates happens in to_atom; a \
             half-pixel error here would land entirely in the transition-width \
             column and would look like a property of the representation",
            pixel_centre(px, py)
        );
        assert!(
            (value - expected_peak).abs() < 1.0e-9,
            "the peak at ({px},{py}) is {value} where the EWA prefactor \
             predicts {expected_peak}. The bridge places the atom but the \
             prefilter is not the one the prefiltered profile declares"
        );
    }

    let spot = Atom2d::from_sigma(
        pixel_centre(20, 8),
        Vec2::new(sigma, sigma),
        0.0,
        MAX_OPACITY,
        Rgb::new(1.0, 1.0, 1.0),
    );
    let shifted = match render_or_background(
        &[spot],
        &shifted_camera(viewport.0, viewport.1, Vec2::new(1.0, 1.0)),
        KernelFamily::Gaussian,
        &settings,
        TILE,
    ) {
        Ok(i) => i,
        Err(e) => panic!("the shifted-grid render failed: {e}"),
    };
    let (shifted_value, sx, sy) = shifted.max_abs();
    assert!(
        (shifted_value - expected_peak).abs() < 1.0e-9,
        "after a one-pixel principal-point shift the peak reads {shifted_value} \
         where the unshifted peak reads {expected_peak}. A whole-pixel shift \
         must land the sample on the atom centre again; a value below the peak \
         means the sample grid moved by a different amount than the principal \
         point did, which is exactly the half-pixel confusion the overfit-gap \
         column would then be reporting as representation quality"
    );
    assert_eq!(
        (sx, sy),
        (21, 9),
        "shifting the principal point by one pixel moved the peak to ({sx},{sy}) \
         instead of (21,9). Stage 4's overfit-gap column is the difference \
         between the fitted grid and a half-pixel-shifted one, and it is \
         meaningless if the shift does not move the sample grid"
    );

    let world = image_to_world(pixel_centre(20, 8), viewport);
    assert_eq!(
        (world.x, world.y),
        (20.5 - 16.0, 8.5 - 16.0),
        "the image-to-world shift is not half the viewport"
    );
    let back = world_to_image(world, viewport);
    assert_eq!(
        (back.x, back.y),
        (pixel_centre(20, 8).x, pixel_centre(20, 8).y),
        "the coordinate bridge does not round-trip"
    );
}

#[test]
fn the_normalised_image_is_order_invariant_and_the_over_image_is_not() {
    let settings = SharedSettings::<f64>::prefiltered();
    let viewport = (32u32, 32u32);
    let first = Atom2d::from_sigma(
        Vec2::new(14.0, 16.0),
        Vec2::new(2.0, 2.0),
        0.0,
        0.7,
        Rgb::new(1.0, 0.2, 0.2),
    );
    let second = Atom2d::from_sigma(
        Vec2::new(18.0, 16.0),
        Vec2::new(3.0, 1.0),
        0.4,
        0.6,
        Rgb::new(0.1, 0.9, 0.3),
    );
    let forward = rendered(&[first, second], viewport, KernelFamily::Gaussian, &settings);
    let reverse = rendered(&[second, first], viewport, KernelFamily::Gaussian, &settings);
    let over_gap = worst_channel_gap(&forward, &reverse);
    assert!(
        over_gap > 0.01,
        "swapping two overlapping atoms changed the alpha-composited image by \
         only {over_gap}. Over sorts on the depth key the paint-order \
         convention assigns, so if order does not matter here the convention is \
         not reaching the renderer and the Over arm is being handed a freedom \
         this build claims to have frozen"
    );
    let forward = rendered(
        &[first, second],
        viewport,
        KernelFamily::GaussianNormalised,
        &settings,
    );
    let reverse = rendered(
        &[second, first],
        viewport,
        KernelFamily::GaussianNormalised,
        &settings,
    );
    let normalised_gap = worst_channel_gap(&forward, &reverse);
    assert!(
        normalised_gap <= 4.0 * f64::EPSILON,
        "swapping the same two atoms changed the normalised image by \
         {normalised_gap}, which is more than a rounding difference. Both of \
         its sums are order-free, so a real difference means the depth key is \
         being used where it must not be, and an atom-count win for Over could \
         then come from a freedom the normalised arm does not have"
    );
}

#[test]
fn the_paint_order_is_frozen_to_the_insertion_index() {
    let viewport = (16u32, 16u32);
    let one = Atom2d::from_sigma(
        pixel_centre(8, 8),
        Vec2::new(1.0, 1.0),
        0.0,
        1.0,
        Rgb::new(1.0, 1.0, 1.0),
    );
    let atoms = [one, one, one, one];
    let scene = match to_scene(&atoms, viewport, 1.0e-8) {
        Ok(s) => s,
        Err(e) => panic!("the paint-order scene did not build: {e}"),
    };
    assert_eq!(scene.len(), atoms.len());
    let mut previous = f64::INFINITY;
    for index in 0..atoms.len() {
        let z = depth_of(index);
        assert_eq!(
            z,
            Z0 - index as f64 * DZ,
            "the depth of atom {index} is not the frozen convention"
        );
        assert!(
            z < previous,
            "atom {index} is not nearer than atom {}: a greedy correction has \
             to paint over what it corrects, and under Over the later atom only \
             does that if it sorts in front",
            index - 1
        );
        previous = z;
    }
    assert!(
        Z0 - (atoms.len() as f64) * DZ > 0.2,
        "the depth ladder runs past the near plane before it runs out of atoms"
    );
}

#[test]
fn the_tolerance_collect_is_refused_on_the_normalised_arm() {
    let settings = SharedSettings::<f64>::prefiltered();
    let spot = Atom2d::from_sigma(
        pixel_centre(8, 8),
        Vec2::new(1.0, 1.0),
        0.0,
        1.0,
        Rgb::new(1.0, 1.0, 1.0),
    );
    let spec = camera(16, 16);
    match render_atoms(
        &[spot],
        &spec,
        KernelFamily::GaussianNormalised,
        &settings,
        TILE,
        Collect::PixelsAndTolerance,
    ) {
        Ok(_) => panic!(
            "the normalised arm accepted the tolerance collect. Its probe \
             channel is one minus a coverage read as a transmittance, and there \
             is no transmittance in a normalised accumulator: the image would \
             look plausible and mean nothing"
        ),
        Err(LightsError::ContractViolation(v)) => {
            assert_eq!(format!("{v:?}").contains("BlendClassNotOver"), true)
        }
        Err(e) => panic!("the refusal came back as the wrong error: {e}"),
    }
    for collect in [Collect::Pixels, Collect::PixelsAndCounters] {
        match render_atoms(
            &[spot],
            &spec,
            KernelFamily::GaussianNormalised,
            &settings,
            TILE,
            collect,
        ) {
            Ok(_) => (),
            Err(e) => panic!("the normalised arm refused {collect:?} as well: {e}"),
        }
    }
    match render_atoms(
        &[spot],
        &spec,
        KernelFamily::Gaussian,
        &settings,
        TILE,
        Collect::PixelsAndTolerance,
    ) {
        Ok(_) => (),
        Err(e) => panic!(
            "the tolerance collect was refused on the Over arm too, so the \
             refusal above is not about the blend rule: {e}"
        ),
    }
}

#[test]
fn the_angle_convention_matches_the_image_axes() {
    let settings = SharedSettings::<f64>::prefiltered();
    let viewport = (48u32, 48u32);
    for declared in [0.0f64, 30.0, 60.0, 120.0] {
        let needle = Atom2d::from_sigma(
            pixel_centre(24, 24),
            Vec2::new(5.0, 0.6),
            declared.to_radians(),
            MAX_OPACITY,
            Rgb::new(1.0, 1.0, 1.0),
        );
        let image = rendered(&[needle], viewport, KernelFamily::Gaussian, &settings);
        let mut mass = 0.0f64;
        let mut xx = 0.0f64;
        let mut xy = 0.0f64;
        let mut yy = 0.0f64;
        for py in 0..viewport.1 {
            for px in 0..viewport.0 {
                let w = image.get(px, py).x;
                let dx = px as f64 + 0.5 - 24.5;
                let dy = py as f64 + 0.5 - 24.5;
                mass += w;
                xx += w * dx * dx;
                xy += w * dx * dy;
                yy += w * dy * dy;
            }
        }
        assert!(
            mass > 1.0,
            "the needle declared at {declared} degrees put almost no light on \
             the image ({mass}), so its shape cannot be measured"
        );
        let measured = 0.5 * (2.0 * xy).atan2(xx - yy);
        let gap = angle_between(measured, declared.to_radians()).to_degrees();
        assert!(
            gap < 3.0,
            "an atom declared at {declared} degrees rendered with its long axis \
             {gap} degrees away (second moments {xx}, {xy}, {yy}). Atom2d::angle \
             is a rotation about world z, which under this camera is a rotation \
             in image coordinates with x to the right and y DOWNWARD, the same \
             axes the targets and the structure tensor use. If the axis or the \
             sign disagreed, every needle the fitter places along a boundary \
             would be placed across it and the atom counts would be a different \
             experiment"
        );
        let ratio = (xx.max(yy)) / (xx.min(yy)).max(1.0e-12);
        assert!(
            ratio > 1.5 || (declared - 45.0).abs() < 1.0e-12,
            "the rendered needle at {declared} degrees is nearly round \
             (moment ratio {ratio}), so the angle recovered above is noise"
        );
    }
}

#[test]
fn the_exhaustion_flag_tells_a_stopped_fitter_from_a_spent_budget() {
    let flat = Target::<f64>::t1_flat_fill();
    let flat_truth = truth_of(&flat);
    let stopped = fitted(&flat, &flat_truth, KernelFamily::GaussianNormalised, 6);
    assert!(
        stopped.improvement_exhausted && stopped.atoms.len() < 6,
        "the flat fill under normalisation settled on {} of the 6 atoms it was \
         allowed and reported improvement_exhausted {}. It must report TRUE \
         here: two atoms already resolve the interior exactly, so the fitter \
         stops because nothing it can add lowers the residual",
        stopped.atoms.len(),
        stopped.improvement_exhausted
    );
    let corner = Target::<f64>::t3_right_angle_corner();
    let corner_truth = truth_of(&corner);
    let bounded = fitted(&corner, &corner_truth, KernelFamily::GaussianNormalised, 2);
    assert!(
        !bounded.improvement_exhausted && bounded.atoms.len() == 2,
        "the corner under normalisation placed {} atoms of the 2 it was \
         allowed and reported improvement_exhausted {}. It must report FALSE \
         here: it stopped because the budget ran out, with improvement still \
         available. The whole reading of the P2 report turns on this one bit - \
         a count that ends with the flag TRUE is a claim about the FITTER and \
         a count that ends with it FALSE is a claim about the BUDGET - and \
         until this test the flag could be nailed to either value with the \
         suite still green",
        bounded.atoms.len(),
        bounded.improvement_exhausted
    );
}

#[test]
fn a_cutoff_above_a_half_is_refused_instead_of_panicking() {
    let target = Target::<f64>::t1_flat_fill();
    let mut truth =
        FloatImage::<f64>::filled(target.viewport.0, target.viewport.1, Rgb::new(0.0, 0.0, 0.0));
    truth.set(20, 20, Rgb::new(1.0, 1.0, 1.0));

    let mut illegal = SharedSettings::<f64>::prefiltered();
    illegal.blend.alpha_cutoff = 0.6;
    let mut request = FitRequest::new(KernelFamily::Gaussian, illegal);
    request.max_atoms = 1;
    request.tile = TILE;
    let mut budget = fit_budget(RENDER_LIMIT);
    match fit(&target, &truth, &request, &mut budget) {
        Err(LightsError::DegenerateAtom { index, reason }) => {
            assert_eq!(
                index, 0,
                "the refusal must name the atom the fitter was about to place"
            );
            assert_eq!(
                reason,
                DegenerateReason::OpacityBelowCutoff,
                "an alpha_cutoff of 0.6 puts the opacity floor 2*cutoff = 1.2 above \
                 MAX_OPACITY = 1, so no atom the fitter can build clears the cutoff"
            );
        }
        Err(other) => panic!(
            "an alpha_cutoff of 0.6 was refused, but with the wrong error: {other}"
        ),
        Ok(outcome) => panic!(
            "an alpha_cutoff of 0.6 fitted {} atoms. The opacity floor is 2*cutoff = 1.2 \
             and the ceiling is 1, so the admissible opacity range is empty and the fit \
             cannot be meaningful",
            outcome.atoms.len()
        ),
    }

    let mut boundary = SharedSettings::<f64>::prefiltered();
    boundary.blend.alpha_cutoff = 0.5;
    let mut request = FitRequest::new(KernelFamily::Gaussian, boundary);
    request.max_atoms = 1;
    request.tile = TILE;
    let mut budget = fit_budget(RENDER_LIMIT);
    match fit(&target, &truth, &request, &mut budget) {
        Ok(_) => (),
        Err(e) => panic!(
            "an alpha_cutoff of exactly 0.5 was refused: {e}. At 0.5 the floor 2*cutoff \
             equals MAX_OPACITY exactly, which is a one-point range, not an empty one, \
             and the refusal must not eat it"
        ),
    }
}

pub const RECOVERY_HEADROOM: f64 = 1.15;

#[test]
fn every_curves_recovery_ceiling_is_that_curves_own_measured_number() {
    let settings = SharedSettings::<f64>::prefiltered();
    let target = Target::<f64>::t4_curve();
    let known = Atom2d::from_sigma(
        pixel_centre(16, 15),
        Vec2::new(3.0, 1.5),
        0.5,
        0.8,
        Rgb::new(0.9, 0.4, 0.2),
    );
    for kernel in KernelFamily::FIT_SWEEP {
        let truth = rendered(&[known], target.viewport, kernel, &settings);
        let mut request = FitRequest::new(kernel, settings);
        request.max_atoms = 1;
        request.tile = TILE;
        let mut budget = fit_budget(RENDER_LIMIT);
        let outcome = match fit(&target, &truth, &request, &mut budget) {
            Ok(o) => o,
            Err(e) => panic!("{}: {e}", kernel.name()),
        };
        let image = rendered(&outcome.atoms, target.viewport, kernel, &settings);
        let gap = worst_channel_gap(&truth, &image);
        let ceiling = recovery_ceiling(kernel);
        println!(
            "{}\tgap {:.4}/255\tceiling {:.4}/255",
            kernel.name(),
            gap * 255.0,
            ceiling * 255.0
        );
        assert!(
            gap < ceiling,
            "{} recovered its own known atom to {} of 255, past its declared \
             ceiling of {} of 255",
            kernel.name(),
            gap * 255.0,
            ceiling * 255.0
        );
        assert!(
            gap * RECOVERY_HEADROOM >= ceiling,
            "{} recovers its own known atom to {:.4} of 255 but is allowed \
             {:.4} of 255, which is {:.2}x its measurement. A ceiling that \
             loose is not this curve's number: it is a shared exemption the \
             curve INHERITED, and under it this curve could regress by {:.2}x \
             with the suite still green. Every curve's ceiling must be that \
             curve's own measured recovery plus at most {}% headroom, so that \
             a new kernel cannot pick up a number measured on a different shape",
            kernel.name(),
            gap * 255.0,
            ceiling * 255.0,
            ceiling / gap,
            ceiling / gap,
            (RECOVERY_HEADROOM - 1.0) * 100.0
        );
    }
}

#[test]
fn the_fitters_one_thread_pin_is_reported_rather_than_inferred_from_the_atom_bits() {
    let target = Target::<f64>::t1_flat_fill();
    let truth = truth_of(&target);
    let mut seen = 0usize;
    for caller in [0usize, 1, 2, 8] {
        let mut settings = SharedSettings::<f64>::prefiltered();
        settings.threads = caller;
        let mut request = FitRequest::new(KernelFamily::Gaussian, settings);
        request.max_atoms = 1;
        request.tile = TILE;
        let mut budget = fit_budget(RENDER_LIMIT);
        let outcome = match fit(&target, &truth, &request, &mut budget) {
            Ok(o) => o,
            Err(e) => panic!("caller threads {caller}: {e}"),
        };
        seen += 1;
        assert_eq!(
            outcome.render_threads, 1,
            "the caller asked for {caller} and the fit must still have rendered on one \
             thread. THE OLD GUARD INFERRED THIS FROM THE ATOM BITS BEING EQUAL, WHICH \
             DOES NOT FOLLOW: a threaded render is bit-identical to a serial one on this \
             content, so the atom bits stay equal whether or not the pin is there. This \
             asserts the pin itself."
        );
    }
    assert_eq!(seen, 4, "the loop must have run every caller value");
}

#[test]
fn explore_the_needle() {
    let settings = SharedSettings::<f64>::prefiltered();
    let target = Target::<f64>::t5_thin_stroke();
    let truth = truth_of(&target);
    let criterion = Criterion::registered();
    let angle = EDGE_DEGREES.to_radians();
    let centre = Vec2::new(32.0, 32.0);
    let ink = Rgb::new(1.0, 1.0, 1.0);

    let judged = |atoms: &[Atom2d]| -> TargetMetric {
        let image = rendered(atoms, target.viewport, KernelFamily::Gaussian, &settings);
        match measure_target(&image, &target, &criterion) {
            Ok(m) => m,
            Err(e) => panic!("the metrology did not compute: {e}"),
        }
    };
    for li in [4.0f64, 6.0, 8.0, 10.0, 12.0] {
        for ai in 1..=20 {
            let across = 0.05 * ai as f64;
            let atoms = [Atom2d::from_sigma(
                centre,
                Vec2::new(li, across),
                angle,
                MAX_OPACITY,
                ink,
            )];
            let m = judged(&atoms);
            let image = rendered(&atoms, target.viewport, KernelFamily::Gaussian, &settings);
            let e = energy(&truth, &image);
            if m.meets || (m.fwhm - 1.0).abs() < 0.6 {
                println!(
                    "len {li} across {across:.2} meets {} trans {:.3} contrast {:.3} peak {:.3} fwhm {:.3} ripple {:.5} energy {e:.3}",
                    m.meets, m.transition_10_90, m.contrast_frac, m.peak_frac, m.fwhm, m.interior_ripple
                );
            }
        }
    }
    let one = fitted(&target, &truth, KernelFamily::Gaussian, 1);
    let m = judged(&one.atoms);
    let image = rendered(&one.atoms, target.viewport, KernelFamily::Gaussian, &settings);
    println!(
        "GREEDY 1: meets {} trans {:.3} contrast {:.3} peak {:.3} fwhm {:.3} ripple {:.5} energy {:.3}",
        m.meets, m.transition_10_90, m.contrast_frac, m.peak_frac, m.fwhm, m.interior_ripple,
        energy(&truth, &image)
    );
    let twelve = fitted(&target, &truth, KernelFamily::Gaussian, 12);
    let m12 = judged(&twelve.atoms);
    let image12 = rendered(&twelve.atoms, target.viewport, KernelFamily::Gaussian, &settings);
    println!(
        "GREEDY {}: meets {} trans {:.3} contrast {:.3} peak {:.3} fwhm {:.3} ripple {:.5} energy {:.3} exhausted {}",
        twelve.atoms.len(), m12.meets, m12.transition_10_90, m12.contrast_frac, m12.peak_frac,
        m12.fwhm, m12.interior_ripple, energy(&truth, &image12), twelve.improvement_exhausted
    );
    println!("empty energy {:.3}", empty_energy(&truth, &settings));
}

fn needle(a: Vec2<f64>, b: Vec2<f64>, across: f64, trim: f64) -> Atom2d {
    let d = b - a;
    let length = (d.x * d.x + d.y * d.y).sqrt();
    Atom2d::from_sigma(
        (a + b) * 0.5,
        Vec2::new((length * 0.5 * trim).max(across), across),
        d.y.atan2(d.x),
        MAX_OPACITY,
        Rgb::new(1.0, 1.0, 1.0),
    )
}

fn judged_by(
    atoms: &[Atom2d],
    target: &Target<f64>,
    kernel: KernelFamily,
    settings: &SharedSettings<f64>,
) -> (TargetMetric, f64) {
    let image = rendered(atoms, target.viewport, kernel, settings);
    let truth = truth_of(target);
    let metric = match measure_target(&image, target, &Criterion::registered()) {
        Ok(m) => m,
        Err(e) => panic!("the metrology did not compute on {}: {e}", target.name),
    };
    (metric, energy(&truth, &image))
}

const NEEDLE_LONG: f64 = 10.0;
const NEEDLE_ACROSS: f64 = 0.75;

#[test]
fn one_hand_placed_needle_meets_the_stroke_criterion_twelve_fitted_atoms_miss() {
    let settings = SharedSettings::<f64>::prefiltered();
    let target = Target::<f64>::t5_thin_stroke();
    let truth = truth_of(&target);
    let kernel = KernelFamily::Gaussian;

    let (exact, _) = judged_by(&[], &target, kernel, &settings);
    let exact_metric = match measure_target(&truth, &target, &Criterion::registered()) {
        Ok(m) => m,
        Err(e) => panic!("{e}"),
    };
    let _ = exact;
    assert_eq!(
        exact_metric.judged, 1,
        "t5 carries exactly one judged probe; {} means this test is measuring a \
         different target than it names",
        exact_metric.judged
    );
    assert!(
        exact_metric.meets,
        "the EXACT raster fails its own criterion (peak {:.3}, fwhm {:.3}), so \
         nothing below is a statement about the representation",
        exact_metric.peak_frac, exact_metric.fwhm
    );

    let needle_atoms = [Atom2d::from_sigma(
        Vec2::new(32.0, 32.0),
        Vec2::new(NEEDLE_LONG, NEEDLE_ACROSS),
        EDGE_DEGREES.to_radians(),
        MAX_OPACITY,
        Rgb::new(1.0, 1.0, 1.0),
    )];
    let (hand, hand_energy) = judged_by(&needle_atoms, &target, kernel, &settings);
    assert_eq!(hand.judged, 1);
    assert!(
        hand.meets,
        "the hand-placed needle at sigma ({NEEDLE_LONG}, {NEEDLE_ACROSS}) does \
         not meet: peak {:.3} (bar 0.900), fwhm {:.3} (bar 1.0 +/- 0.5). This \
         construction is the whole point of the test; if the render path moved, \
         re-run the sweep in the sibling prose folder rather than relaxing the \
         criterion",
        hand.peak_frac, hand.fwhm
    );

    let twelve = fitted(&target, &truth, kernel, 12);
    let (fitted_metric, fitted_energy) = judged_by(&twelve.atoms, &target, kernel, &settings);
    assert_eq!(
        twelve.atoms.len(),
        12,
        "the fitter placed {} atoms, so the comparison below is not the one this \
         test names",
        twelve.atoms.len()
    );
    assert!(
        !fitted_metric.meets,
        "twelve fitted atoms now MEET the stroke criterion (peak {:.3}, fwhm \
         {:.3}). That would be good news and it breaks the finding this test \
         records, so it must be read and not patched",
        fitted_metric.peak_frac, fitted_metric.fwhm
    );
    assert!(
        hand_energy > 10.0 * fitted_energy,
        "the needle sits at residual energy {hand_energy} and the fitter's \
         twelve atoms at {fitted_energy}; the gap is the finding -- one atom \
         meets the criterion at more than ten times the L2 cost of twelve atoms \
         that do not. A shrinking gap means the loss and the criterion have \
         stopped diverging and P2's caveat needs re-reading"
    );
    println!(
        "needle: meets {} energy {hand_energy:.3} | fitted 12: meets {} energy {fitted_energy:.3} | empty {:.3}",
        hand.meets,
        fitted_metric.meets,
        empty_energy(&truth, &settings)
    );
}

const GLYPH_STEM_ACROSS: f64 = 0.95;
const GLYPH_STEM_TRIM: f64 = 0.8;
const GLYPH_ARC_ACROSS: f64 = 0.55;
const GLYPH_ARC_LONG: f64 = 2.0;

fn glyph_construction(target: &Target<f64>) -> Vec<Atom2d> {
    let mut out = Vec::new();
    for part in target.set.parts.iter() {
        match part.shape {
            lights::fit::target::Primitive::Segment { a, b, .. } => {
                if part.id == lights::fit::target::FeatureId(0) {
                    out.push(needle(a, b, GLYPH_STEM_ACROSS, GLYPH_STEM_TRIM));
                }
            }
            lights::fit::target::Primitive::Arc {
                centre,
                radius,
                from,
                to,
                ..
            } => {
                for k in 0..3 {
                    let t = from + (to - from) * ((k as f64) + 0.5) / 3.0;
                    out.push(Atom2d::from_sigma(
                        centre + Vec2::new(t.cos(), t.sin()) * radius,
                        Vec2::new(GLYPH_ARC_LONG, GLYPH_ARC_ACROSS),
                        t + core::f64::consts::FRAC_PI_2,
                        MAX_OPACITY,
                        Rgb::new(1.0, 1.0, 1.0),
                    ));
                }
            }
            lights::fit::target::Primitive::HalfPlane { .. } => (),
            lights::fit::target::Primitive::Disc { .. } => (),
        }
    }
    out
}

#[test]
fn a_four_atom_glyph_construction_meets_the_criterion_the_fitters_four_atoms_miss() {
    let settings = SharedSettings::<f64>::prefiltered();
    let target = Target::<f64>::t6_glyph_like();
    let truth = truth_of(&target);
    let kernel = KernelFamily::GaussianNormalised;
    let empty = empty_energy(&truth, &settings);

    let exact_metric = match measure_target(&truth, &target, &Criterion::registered()) {
        Ok(m) => m,
        Err(e) => panic!("{e}"),
    };
    assert_eq!(
        exact_metric.judged, 2,
        "t6 carries two judged probes and one Apex probe that is not judged; \
         {} means the target moved",
        exact_metric.judged
    );
    assert!(
        exact_metric.meets,
        "the EXACT raster fails its own glyph criterion (peak {:.3}, fwhm \
         {:.3}), so nothing below is a statement about the representation",
        exact_metric.peak_frac, exact_metric.fwhm
    );

    let atoms = glyph_construction(&target);
    assert_eq!(
        atoms.len(),
        4,
        "the construction is {} atoms, not four, so it is not the one this test \
         names",
        atoms.len()
    );
    let (hand, hand_energy) = judged_by(&atoms, &target, kernel, &settings);
    assert!(
        hand.meets,
        "the four-atom construction does not meet: peak {:.3} (bar 0.900), fwhm \
         {:.3} (bar 1.0 +/- 0.5). The arc probe is the tight one -- see the \
         sibling prose folder for the measured caps",
        hand.peak_frac, hand.fwhm
    );

    let four = fitted(&target, &truth, kernel, 4);
    let (fitted_metric, fitted_energy) = judged_by(&four.atoms, &target, kernel, &settings);
    assert!(
        !fitted_metric.meets,
        "the fitter's four atoms now MEET the glyph criterion (peak {:.3}, fwhm \
         {:.3}), which breaks the finding this test records",
        fitted_metric.peak_frac, fitted_metric.fwhm
    );
    assert!(
        hand_energy > 2.0 * fitted_energy,
        "the construction sits at residual energy {hand_energy} and the \
         fitter's four atoms at {fitted_energy}. The construction passing while \
         costing more than twice the L2 is the divergence this test exists for"
    );
    assert!(
        hand_energy < empty,
        "the construction reaches residual energy {hand_energy} against an \
         empty scene's {empty}. P2_RESULTS.md reports a four-atom construction \
         passing at 68.19 against this same 50.74, i.e. WORSE than painting \
         nothing; no construction in the needle family searched here is worse \
         than empty, so that stronger claim is recorded as unreproduced rather \
         than as reproduced. If this assertion starts failing, the stronger \
         claim has been reached and the note in the prose folder is stale"
    );
    println!(
        "glyph construction: meets {} energy {hand_energy:.3} | fitted 4: meets {} energy {fitted_energy:.3} | empty {empty:.3}",
        hand.meets, fitted_metric.meets
    );
}

const PREFIX_DEPTH: usize = 3;
const PREFIX_GAP_FLOOR: f64 = 1e-2;

fn prefix_energy(
    atoms: &[Atom2d],
    target: &Target<f64>,
    truth: &FloatImage<f64>,
    settings: &SharedSettings<f64>,
) -> f64 {
    let image = rendered(atoms, target.viewport, KernelFamily::Gaussian, settings);
    energy(truth, &image)
}

#[test]
fn the_first_n_atoms_of_a_longer_fit_are_not_the_n_atom_answer() {
    let settings = SharedSettings::<f64>::prefiltered();
    let mut shortfalls: Vec<f64> = Vec::new();
    for target in [Target::<f64>::t5_thin_stroke(), Target::<f64>::t4_curve()] {
        let truth = truth_of(&target);
        let full = fitted(&target, &truth, KernelFamily::Gaussian, PREFIX_DEPTH);
        assert_eq!(
            full.atoms.len(),
            PREFIX_DEPTH,
            "{} did not place {PREFIX_DEPTH} atoms, so there is no prefix to read",
            target.name
        );
        println!(
            "prefix: {} full fit has {} atoms, exhausted {}",
            target.name,
            full.atoms.len(),
            full.improvement_exhausted
        );
        for n in 1..=full.atoms.len() {
            let alone = fitted(&target, &truth, KernelFamily::Gaussian, n);
            assert_eq!(
                alone.atoms.len(),
                n,
                "{} at max_atoms {n} placed {} atoms",
                target.name,
                alone.atoms.len()
            );
            let slice = &full.atoms[..n];
            let identical = slice
                .iter()
                .zip(alone.atoms.iter())
                .all(|(a, b)| a.bits() == b.bits());
            let e_slice = prefix_energy(slice, &target, &truth, &settings);
            let e_alone = prefix_energy(&alone.atoms, &target, &truth, &settings);
            let relative = (e_slice - e_alone) / e_alone;
            println!(
                "prefix:   n={n} identical={identical} slice_energy={e_slice:.6} alone_energy={e_alone:.6} gap={:.6e} relative={relative:.3e}",
                e_slice - e_alone
            );
            for (i, (a, b)) in slice.iter().zip(alone.atoms.iter()).enumerate() {
                if a.bits() != b.bits() {
                    println!(
                        "prefix:     atom {i} moved: centre ({:.6},{:.6}) -> ({:.6},{:.6}) angle {:.6} -> {:.6}",
                        b.centre.x, b.centre.y, a.centre.x, a.centre.y, b.angle, a.angle
                    );
                }
            }
            if n == full.atoms.len() {
                assert!(
                    identical,
                    "{} at n={n}: the whole atom list must equal the n-atom fit, because it IS \
                     that fit; if this fails the fitter is not deterministic and every other \
                     number here is unreadable",
                    target.name
                );
                assert_eq!(
                    relative, 0.0,
                    "{} at n={n}: identical atom lists rendered to different energies",
                    target.name
                );
                continue;
            }
            assert!(
                !identical,
                "{} at n={n}: the prefix of the {PREFIX_DEPTH}-atom fit IS bit-identical to the \
                 {n}-atom fit. The prefix contract holds here, which contradicts the measurement \
                 BURNDOWN L-142 was raised on; refine_tail ({}) may have been changed to zero, or \
                 the tail refinement no longer moves an earlier atom. Re-measure before trusting \
                 either reading.",
                target.name,
                DEFAULT_REFINE_TAIL
            );
            assert!(
                relative > PREFIX_GAP_FLOOR,
                "{} at n={n}: the prefix renders only {relative:.3e} worse than the {n}-atom fit \
                 (floor {PREFIX_GAP_FLOOR:e}). The gap is the whole finding: a consumer reading \
                 atoms[..n] is reading atoms tuned to cooperate with atoms it is discarding.",
                target.name
            );
            shortfalls.push(relative);
        }
    }
    let worst = shortfalls.iter().fold(0.0f64, |a, b| a.max(*b));
    println!(
        "prefix: {} sub-full readings, every one of them worse than its own fit; worst {worst:.3e}",
        shortfalls.len()
    );
    assert_eq!(
        shortfalls.len(),
        2 * (PREFIX_DEPTH - 1),
        "every target must contribute PREFIX_DEPTH-1 sub-full readings, or this test passed by \
         measuring fewer cells than it claims"
    );
}

const LADDER_CELLS_FLOOR: usize = 12;

fn every_class() -> [Target<f64>; 6] {
    [
        Target::<f64>::t1_flat_fill(),
        Target::<f64>::t2_straight_edge(),
        Target::<f64>::t3_right_angle_corner(),
        Target::<f64>::t4_curve(),
        Target::<f64>::t5_thin_stroke(),
        Target::<f64>::t6_glyph_like(),
    ]
}

#[test]
fn the_published_ladder_reads_a_prefix_at_every_rung_and_the_shortfall_is_tabulated() {
    let settings = SharedSettings::<f64>::prefiltered();
    let mut cells: Vec<(String, usize, f64)> = Vec::new();
    println!("ladder: class            n   prefix_energy     rung_fit_energy   shortfall");
    for target in every_class() {
        let truth = truth_of(&target);
        let full = fitted(&target, &truth, KernelFamily::Gaussian, NTERM_MAX_ATOMS);
        let placed = full.atoms.len();
        for n in ladder_upto(placed) {
            if n == placed {
                continue;
            }
            let alone = fitted(&target, &truth, KernelFamily::Gaussian, n);
            if alone.atoms.len() != n {
                println!(
                    "ladder: {:<16} {n:>3}   REFUSED: the {n}-atom fit stopped at {} atoms",
                    target.name,
                    alone.atoms.len()
                );
                continue;
            }
            let e_prefix = prefix_energy(&full.atoms[..n], &target, &truth, &settings);
            let e_rung = prefix_energy(&alone.atoms, &target, &truth, &settings);
            let shortfall = (e_prefix - e_rung) / e_rung;
            println!(
                "ladder: {:<16} {n:>3}   {e_prefix:>14.6}   {e_rung:>15.6}   {shortfall:>+9.3e}",
                target.name
            );
            cells.push((target.name.to_string(), n, shortfall));
        }
        println!(
            "ladder: {} placed {placed} of a {NTERM_MAX_ATOMS} cap, exhausted {}",
            target.name, full.improvement_exhausted
        );
    }
    assert!(
        cells.len() >= LADDER_CELLS_FLOOR,
        "only {} (class, rung) cells were measured (floor {LADDER_CELLS_FLOOR}); a sweep that \
         reads almost nothing can report almost anything",
        cells.len()
    );
    let worse: Vec<&(String, usize, f64)> = cells.iter().filter(|c| c.2 > 0.0).collect();
    let better: Vec<&(String, usize, f64)> = cells.iter().filter(|c| c.2 < 0.0).collect();
    let worst = cells.iter().fold(0.0f64, |a, c| a.max(c.2));
    let mean = cells.iter().map(|c| c.2).sum::<f64>() / cells.len() as f64;
    println!(
        "ladder: {} cells; {} read worse than their own fit, {} read better, {} exactly equal; \
         worst {worst:+.3e}, mean {mean:+.3e}",
        cells.len(),
        worse.len(),
        better.len(),
        cells.len() - worse.len() - better.len()
    );
    for c in better.iter() {
        println!("ladder:   BETTER: {} n={} {:+.3e}", c.0, c.1, c.2);
    }
    assert!(
        mean > 0.0,
        "the prefix reads BETTER on average ({mean:+.3e}) than the rung's own fit. That reverses \
         the sign L-142 was raised on and the entry must be rewritten, not this assertion"
    );
    assert!(
        worst > 1e-2,
        "the worst rung reads only {worst:+.3e} worse than its own fit; at that size the prefix \
         is a rounding matter and L-142 is a footnote"
    );
}

#[test]
fn the_fit_tree_declares_none_of_the_seven_dead_public_items() {
    let sources = [
        ("src/fit/greedy.rs", include_str!("../src/fit/greedy.rs"), "pub struct FitRequest"),
        ("src/fit/harness.rs", include_str!("../src/fit/harness.rs"), "pub fn run("),
        ("src/fit/scale.rs", include_str!("../src/fit/scale.rs"), "pub fn factor(self)"),
        ("src/fit/aspect3.rs", include_str!("../src/fit/aspect3.rs"), "fn class_ladder"),
        ("src/fit/transform/mod.rs", include_str!("../src/fit/transform/mod.rs"), "pub enum Transform"),
    ];
    let dead = [
        "target_index",
        "fn background_of(",
        "fn quadrature_budget(",
        "ALL_TRANSFORMS",
        "fn whole(",
        "fn unit()",
        "fn is_unit(",
        "fn scene_point(",
    ];
    for (path, text, anchor) in sources {
        assert!(text.contains(anchor), "{path} no longer contains {anchor:?}; the scan is pointed at the wrong file");
        for name in dead {
            assert!(!text.contains(name), "{path} declares {name:?} again, which nothing in the tree reads");
        }
    }
}
