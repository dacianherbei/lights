#![cfg(feature = "fit")]
use lights::fit::analytic::{AnalyticArm, render_analytic};
use lights::fit::atom2d::pixel_centre;
use lights::fit::coverage::{DEFAULT_MAX_DEPTH, coverage_budget};
use lights::fit::edge::{
    BIN_WIDTH, Criterion, EdgeMetric, HIGH_LEVEL, LOW_LEVEL, SDF_ARM_TRUE_FWHM,
    SDF_ARM_TRUE_TRANSITION, THIN_BIN_SAMPLES, measure, measure_target,
};
use lights::fit::target::{EdgeProbe, FeatureId, ProbeKind, Target};
use lights::image::FloatImage;
use lights::numerics::vector::{Rgb, Vec2};

const CELL_LIMIT: u64 = 64_000_000;

fn painted<F: Fn(f64) -> f64>(target: &Target<f64>, f: F) -> FloatImage<f64> {
    let (width, height) = target.viewport;
    let mut image = FloatImage::<f64>::new(width, height);
    for py in 0..height {
        for px in 0..width {
            let value = f(target.sdf(pixel_centre(px, py)));
            image.set(px, py, Rgb::splat(value));
        }
    }
    image
}

fn probe_of(target: &Target<f64>, index: usize) -> EdgeProbe<f64> {
    match target.probes.get(index) {
        Some(p) => *p,
        None => panic!("{} has no probe {index}", target.name),
    }
}

fn measured(target: &Target<f64>, image: &FloatImage<f64>, index: usize) -> EdgeMetric {
    let probe = probe_of(target, index);
    match measure(image, target, &probe, &Criterion::registered()) {
        Ok(m) => m,
        Err(e) => panic!("measuring {} probe {index} failed: {e}", target.name),
    }
}

fn control(target: &Target<f64>, arm: AnalyticArm) -> FloatImage<f64> {
    let mut budget = coverage_budget(CELL_LIMIT);
    match render_analytic(target, arm, DEFAULT_MAX_DEPTH, &mut budget) {
        Ok((image, _)) => image,
        Err(e) => panic!("the {} arm on {} failed: {e}", arm.name(), target.name),
    }
}

#[test]
fn the_metrology_recovers_the_linear_ramp_width_exactly() {
    let target = Target::<f64>::t2_straight_edge();
    let image = painted(&target, |sdf| (0.5 - sdf).clamp(0.0, 1.0));
    let metric = measured(&target, &image, 0);
    assert!(
        (metric.transition_10_90 - SDF_ARM_TRUE_TRANSITION).abs() < 1.0e-12,
        "the one profile whose 10-90 width has an exact closed form is the unit \
         linear ramp: coverage falls from 1 to 0 over exactly one pixel of \
         signed distance, so the 0.9 and 0.1 crossings sit at sdf -0.4 and \
         +0.4 and the width is exactly {SDF_ARM_TRUE_TRANSITION}. The \
         instrument returned {}, off by {:.3e}. An instrument that cannot \
         return a width it was handed cannot measure one it was not, and this \
         is the ONLY case where the answer is exact rather than quantised, so \
         it is the calibration the other cases are read against.",
        metric.transition_10_90,
        metric.transition_10_90 - SDF_ARM_TRUE_TRANSITION
    );
    assert!(
        (metric.sdf_90 + 0.4).abs() < 1.0e-12 && (metric.sdf_10 - 0.4).abs() < 1.0e-12,
        "the two crossings of the unit ramp sit at sdf -0.4 and +0.4 exactly; \
         the instrument places them at {} and {}. The WIDTH cannot see this on \
         its own - an interpolation that reads the wrong end of every bracket \
         shifts both crossings by the same bin gap and their difference is \
         unchanged - so the positions are asserted as well as the span",
        metric.sdf_90,
        metric.sdf_10
    );
    assert!(
        (metric.contrast - 1.0).abs() < 1.0e-12 && metric.plateau_low.abs() < 1.0e-12,
        "the ramp saturates well before the plateau band, so contrast must be \
         exactly one, got {} with plateaus {} and {}",
        metric.contrast,
        metric.plateau_low,
        metric.plateau_high
    );
}

#[test]
fn the_metrology_recovers_a_known_logistic_width() {
    let target = Target::<f64>::t2_straight_edge();
    for k in [4.394449154672439f64, 8.0, 16.0] {
        let image = painted(&target, |sdf| 1.0 / (1.0 + (k * sdf).exp()));
        let metric = measured(&target, &image, 0);
        let ideal = 2.0 * 9.0f64.ln() / k;
        let error = metric.transition_10_90 - ideal;
        assert!(
            error.abs() < 6.0e-3,
            "a logistic of steepness {k} has 10-90 width exactly 2 ln 9 / k = \
             {ideal}; the instrument returned {} (error {error:.6}). Two named \
             terms make up that error and neither is noise: the plateaus are \
             means over |sdf| in [1.5, 3.0] rather than limits at infinity, \
             which for a logistic biases the levels by about 1/(1+exp(1.5k)); \
             and the crossings are linear interpolations across bins {BIN_WIDTH} \
             px wide, whose chord error on a curve of this curvature is of \
             order k h^2 / 8.",
            metric.transition_10_90
        );
        assert!(
            metric.transition_10_90 > 0.0,
            "the width came back {} at steepness {k}. A NEGATIVE width is what \
             swapping the {LOW_LEVEL} and {HIGH_LEVEL} crossings produces, and \
             it is the shape of the whole instrument being read backwards",
            metric.transition_10_90
        );
    }
}

#[test]
fn the_metrology_recovers_a_known_fwhm() {
    let target = Target::<f64>::t5_thin_stroke();
    let probe = probe_of(&target, 0);
    let half_width = 0.5f64;
    let mut closest = f64::INFINITY;
    for py in 0..target.viewport.1 {
        for px in 0..target.viewport.0 {
            let centre = pixel_centre(px, py);
            if (centre - probe.anchor).length() > probe.window {
                continue;
            }
            if target.nearest_feature(centre) != probe.feature {
                continue;
            }
            let sdf = target.sdf(centre);
            if sdf < closest {
                closest = sdf;
            }
        }
    }
    let gap = closest + half_width;
    assert!(
        gap >= 0.0 && gap < 0.5,
        "the closest pixel centre to the stroke centreline sits {gap} px off \
         it, which is outside the half pixel that is geometrically possible"
    );

    for sigma in [0.4f64, 0.6, 0.9] {
        let image = painted(&target, |sdf| {
            let x = sdf + half_width;
            (-(x * x) / (2.0 * sigma * sigma)).exp()
        });
        let metric = measured(&target, &image, 0);
        let ideal = 2.0 * (2.0 * 2.0f64.ln()).sqrt() * sigma;
        let error = metric.fwhm - ideal;
        assert!(
            error <= 1.0e-6 && error >= -2.0 * (gap + BIN_WIDTH),
            "a gaussian stroke of sigma {sigma} has FWHM exactly 2 sqrt(2 ln 2) \
             sigma = {ideal}; the instrument returned {} (error {error:.6}). \
             The instrument can only UNDERSTATE it, and by a bounded amount: \
             the peak is located at the lowest-sdf BIN, and no pixel centre \
             lands closer than {gap} px to the centreline, so the measured \
             half-width is short by at most that plus one bin. It must never \
             OVERSTATE, because that would mean the half-maximum crossing was \
             found beyond the profile rather than on it.",
            metric.fwhm
        );
    }
}

#[test]
fn the_metrology_never_interpolates() {
    let target = Target::<f64>::t2_straight_edge();
    let levels = 17.0f64;
    let (width, height) = target.viewport;
    let mut image = FloatImage::<f64>::new(width, height);
    for py in 0..height {
        for px in 0..width {
            let step = ((px * 7 + py * 13) % 17) as f64;
            image.set(px, py, Rgb::splat(step / levels));
        }
    }
    let metric = measured(&target, &image, 0);
    assert!(
        !metric.bins.is_empty(),
        "the interpolation check binned nothing, so it checked nothing"
    );
    let mut total = 0u32;
    for b in metric.bins.iter() {
        let scaled = b.mean * levels * b.samples as f64;
        assert!(
            (scaled - scaled.round()).abs() < 1.0e-9,
            "bin at sdf {} carries mean {} over {} samples. Every value in this \
             image is k/{levels} for an integer k that is a function of the \
             PIXEL INDEX alone, so any bin mean must be an integer multiple of \
             1/({levels} * samples). {scaled} is not an integer, which means a \
             value entered the profile that was never at a pixel centre - and \
             bilinear resampling would add close to a pixel of blur to a \
             quantity whose acceptance threshold is one pixel.",
            b.sdf,
            b.mean,
            b.samples
        );
        total += b.samples;
    }
    assert_eq!(
        total, metric.samples,
        "the bins hold {total} samples but {} were gathered: a sample that \
         reaches no bin is a sample the profile silently dropped",
        metric.samples
    );
}

#[test]
fn the_corner_probe_rejects_the_other_edges_samples() {
    let target = Target::<f64>::t3_right_angle_corner();
    let image = control(&target, AnalyticArm::Sdf);
    for index in [0usize, 1] {
        let probe = probe_of(&target, index);
        let mut inside_window = 0u32;
        let mut same_feature = 0u32;
        for py in 0..target.viewport.1 {
            for px in 0..target.viewport.0 {
                let centre = pixel_centre(px, py);
                if (centre - probe.anchor).length() > probe.window {
                    continue;
                }
                inside_window += 1;
                if target.nearest_feature(centre) == probe.feature {
                    same_feature += 1;
                }
            }
        }
        let metric = measured(&target, &image, index);
        assert_eq!(
            metric.samples, same_feature,
            "probe {index} used {} samples where {same_feature} pixel centres \
             in its window have feature {:?} as their nearest",
            metric.samples, probe.feature
        );
        assert!(
            same_feature < inside_window,
            "probe {index}: all {inside_window} pixel centres in the window \
             already belong to feature {:?}, so the nearest-feature filter \
             removed nothing and this test would pass with the filter deleted. \
             A probe window that reaches the apex must contain the OTHER \
             edge's samples, or the mixed-profile widening this filter exists \
             to prevent could not occur",
            probe.feature
        );
        assert!(
            metric.transition_10_90 < 1.0,
            "with the other edge's samples excluded, probe {index} on the \
             analytic-sdf corner measures {} px. A profile that mixed two edges \
             meeting at 90 degrees would report a fake widening at exactly the \
             feature the corner target exists to ask about",
            metric.transition_10_90
        );
    }
}

#[test]
fn thin_bins_are_counted_and_reported() {
    let target = Target::<f64>::t2_straight_edge();
    let image = control(&target, AnalyticArm::Sdf);
    let wide = measured(&target, &image, 0);
    let starved = EdgeProbe {
        feature: FeatureId(0),
        anchor: Vec2::new(32.0, 32.0),
        window: 1.5,
        kind: ProbeKind::Edge,
    };
    let thin = match measure(&image, &target, &starved, &Criterion::registered()) {
        Ok(m) => m,
        Err(e) => panic!("the starved probe failed to measure: {e}"),
    };
    assert!(
        thin.samples < wide.samples / 8,
        "the starved probe gathered {} samples against the wide probe's {}, \
         which is not starved enough to be a test",
        thin.samples,
        wide.samples
    );
    let thin_fraction = thin.thin_bins as f64 / thin.bins.len().max(1) as f64;
    assert!(
        thin_fraction > 0.9,
        "the starved probe left {} of its {} bins under {THIN_BIN_SAMPLES} \
         samples. thin_bins exists so that a starved scatter is VISIBLE in the \
         report rather than reported as a confident width, and a counter that \
         does not rise when the scatter is deliberately starved is measuring \
         nothing",
        thin.thin_bins,
        thin.bins.len()
    );
    assert!(
        wide.thin_bins > 0,
        "the wide probe reports zero thin bins, which would mean a 64x64 image \
         gives four samples in every 0.05 px bin of signed distance. It does \
         not, and a report that claimed so would be hiding its own sparsity"
    );
}

#[test]
fn the_control_arm_meets_the_criterion_on_every_target() {
    let criterion = Criterion::registered();
    for target in Target::<f64>::all() {
        for arm in AnalyticArm::ALL {
            let image = control(&target, arm);
            let metric = match measure_target(&image, &target, &criterion) {
                Ok(m) => m,
                Err(e) => panic!("measuring {} on {} failed: {e}", arm.name(), target.name),
            };
            assert!(
                metric.judged > 0,
                "{} on {} judged no probe at all, so its verdict is vacuous",
                arm.name(),
                target.name
            );
            assert!(
                metric.meets,
                "THE CRITERION IS INVALID, NOT THE ATOMS. {} is the quality \
                 ceiling for {}: nothing can beat exact box coverage of the \
                 same analytic shape, so if it cannot meet the criterion then \
                 no atom count measured against that criterion means anything. \
                 Measured 10-90 {} px, contrast {}, peak {}, fwhm {} px, \
                 interior ripple {}, against max width {}, min contrast {}, min \
                 peak {}, fwhm {} +- {}, max ripple {}. The binary exits 3 in \
                 exactly this case and refuses to print atom numbers.",
                arm.name(),
                target.name,
                metric.transition_10_90,
                metric.contrast_frac,
                metric.peak_frac,
                metric.fwhm,
                metric.interior_ripple,
                criterion.max_transition,
                criterion.min_contrast_frac,
                criterion.min_peak_frac,
                criterion.target_fwhm,
                criterion.max_fwhm_error,
                criterion.max_interior_ripple
            );
        }
    }
}

#[test]
fn the_sdf_arm_carries_a_closed_form_and_its_miss_is_the_instruments_own_error() {
    let dense = Target::<f64>::t2_straight_edge();
    let dense_metric = match measure_target(
        &control(&dense, AnalyticArm::Sdf),
        &dense,
        &Criterion::registered(),
    ) {
        Ok(m) => m,
        Err(e) => panic!("{e}"),
    };
    assert!(
        (dense_metric.transition_10_90 - SDF_ARM_TRUE_TRANSITION).abs() < 1.0e-12,
        "the analytic-sdf arm paints clamp(0.5 - sdf), a unit linear ramp, so \
         its 10-90 width is exactly {SDF_ARM_TRUE_TRANSITION} px on EVERY \
         target. On the 30 degree edge, where the pixel offsets equidistribute \
         and the scatter is dense, the instrument returns {} - so any departure \
         from {SDF_ARM_TRUE_TRANSITION} on another target is that target's own \
         scatter quantisation and not a property of the shape",
        dense_metric.transition_10_90
    );

    let sparse = Target::<f64>::t4_curve();
    let sparse_metric = match measure_target(
        &control(&sparse, AnalyticArm::Sdf),
        &sparse,
        &Criterion::registered(),
    ) {
        Ok(m) => m,
        Err(e) => panic!("{e}"),
    };
    let offset = sparse_metric.transition_10_90 - SDF_ARM_TRUE_TRANSITION;
    assert!(
        offset > 0.005 && offset < 0.2,
        "the same closed form on the 3 px radius curve is off by {offset} px. A \
         circle of radius 3 has about 19 pixel centres per unit of signed \
         distance, so its 10-90 band holds about 15 samples spread over 16 \
         bins and the crossings are bracketed across gaps. This number is \
         REPORTED as instrument_offset_px on every row of that target, because \
         a width quoted for glyph-scale curvature without it would be \
         confidently wrong by about this much"
    );

    let stroke = Target::<f64>::t5_thin_stroke();
    let stroke_metric = match measure_target(
        &control(&stroke, AnalyticArm::Sdf),
        &stroke,
        &Criterion::registered(),
    ) {
        Ok(m) => m,
        Err(e) => panic!("{e}"),
    };
    assert!(
        stroke_metric.fwhm < SDF_ARM_TRUE_FWHM
            && SDF_ARM_TRUE_FWHM - stroke_metric.fwhm < 0.15,
        "the linear ramp over a stroke of half width 0.5 has FWHM exactly \
         {SDF_ARM_TRUE_FWHM} px; the instrument returns {}. It is SHORT, always, \
         because the peak is placed at the lowest-sdf bin and no pixel centre \
         lands on the centreline. That bias is reported as \
         instrument_offset_fwhm_px and it is why the stroke rows are read as a \
         delta against the control arm rather than as absolutes",
        stroke_metric.fwhm
    );
}

#[test]
fn a_shift_of_the_principal_point_translates_the_sampling_grid_and_nothing_else() {
    let target = Target::<f64>::t2_straight_edge();
    let (width, height) = target.viewport;
    let settings = lights::settings::SharedSettings::<f64>::prefiltered();
    let atoms = [
        lights::fit::atom2d::Atom2d::from_sigma(
            Vec2::new(28.0, 30.0),
            Vec2::new(9.0, 0.9),
            30.0f64.to_radians(),
            1.0,
            Rgb::splat(1.0),
        ),
        lights::fit::atom2d::Atom2d::from_sigma(
            Vec2::new(36.0, 34.0),
            Vec2::new(6.0, 1.4),
            30.0f64.to_radians(),
            0.7,
            Rgb::splat(0.4),
        ),
    ];
    let render = |spec: &lights::camera::CameraSpec| -> FloatImage<f64> {
        match lights::fit::greedy::render_or_background(
            &atoms,
            spec,
            lights::kernel::KernelFamily::GaussianNormalised,
            &settings,
            16,
        ) {
            Ok(i) => i,
            Err(e) => panic!("rendering the shift probe failed: {e}"),
        }
    };
    let plain = render(&lights::fit::atom2d::camera(width, height));
    let moved = render(&lights::fit::atom2d::shifted_camera(
        width,
        height,
        Vec2::new(1.0, 1.0),
    ));
    let mut compared = 0usize;
    for py in 1..height {
        for px in 1..width {
            let a = plain.get(px - 1, py - 1);
            let b = moved.get(px, py);
            assert!(
                a.x.to_bits() == b.x.to_bits()
                    && a.y.to_bits() == b.y.to_bits()
                    && a.z.to_bits() == b.z.to_bits(),
                "shifting the principal point by one whole pixel must reproduce \
                 the same image translated by one whole pixel, bit for bit. At \
                 ({px}, {py}) the shifted render is {b:?} where the plain \
                 render at ({}, {}) is {a:?}. This is what fixes the SIGN and \
                 the SCALE of the half-pixel overfit column: pixel px under a \
                 shift of s samples image coordinate px + 0.5 - s, so the \
                 metrology must move its abscissa by -s",
                px - 1,
                py - 1
            );
            compared += 1;
        }
    }
    assert!(compared > 3000, "only {compared} pixels were compared");
}

#[test]
fn the_shifted_grid_must_be_measured_at_the_positions_it_sampled() {
    let target = Target::<f64>::t2_straight_edge();
    let (width, height) = target.viewport;
    let plain = control(&target, AnalyticArm::Sdf);
    let mut moved = FloatImage::<f64>::new(width, height);
    for py in 0..height {
        for px in 0..width {
            let sx = px.saturating_sub(1);
            let sy = py.saturating_sub(1);
            moved.set(px, py, plain.get(sx, sy));
        }
    }
    let probe = probe_of(&target, 0);
    let criterion = Criterion::registered();
    let correct = match lights::fit::edge::measure_shifted(
        &moved,
        &target,
        &probe,
        &criterion,
        Vec2::new(-1.0, -1.0),
    ) {
        Ok(m) => m,
        Err(e) => panic!("{e}"),
    };
    let reference = measured(&target, &plain, 0);
    assert_eq!(
        correct.bins.len(),
        reference.bins.len(),
        "the abscissae are a lattice, so a whole-pixel shift cannot change \
         WHICH bins exist, only what lands in them"
    );
    for (a, b) in correct.bins.iter().zip(reference.bins.iter()) {
        assert!(
            a.sdf.to_bits() == b.sdf.to_bits()
                && a.mean.to_bits() == b.mean.to_bits()
                && a.samples == b.samples,
            "bin {a:?} against {b:?}: measured with the offset the shift \
             actually introduced, the translated render must reproduce the \
             SAME (value, abscissa) pairs as the plain one, bit for bit"
        );
    }

    let ignored = measured(&target, &moved, 0);
    let mut worst = 0.0f64;
    for (a, b) in ignored.bins.iter().zip(reference.bins.iter()) {
        let gap = (a.mean - b.mean).abs();
        if gap > worst {
            worst = gap;
        }
    }
    assert!(
        worst > 0.9,
        "measuring the translated render at the UNSHIFTED pixel centres moved \
         no bin mean by more than {worst}. Neither the bin positions nor the \
         10-90 width can see this error: the abscissa is a lattice quantity so \
         the SET of signed distances is unchanged, and a profile that is a \
         function of signed distance has a translation-invariant width. What \
         changes is the PAIRING - each signed distance is handed the luminance \
         of a pixel one step away, which along a 30 degree edge is cos 30 + \
         sin 30 = 1.366 px of signed distance, enough to carry a unit ramp from \
         one plateau to the other. This assertion is the only thing in the \
         suite that can see the sign of the overfit column's offset"
    );
}

#[test]
fn an_apex_probe_is_measured_and_never_judged() {
    let criterion = Criterion::registered();
    let mut apex_probes = 0usize;
    for target in Target::<f64>::all() {
        let image = control(&target, AnalyticArm::Exact);
        let metric = match measure_target(&image, &target, &criterion) {
            Ok(m) => m,
            Err(e) => panic!("{e}"),
        };
        let mut judged = 0u32;
        for m in metric.probes.iter() {
            match m.kind {
                ProbeKind::Apex => {
                    apex_probes += 1;
                    assert!(
                        !m.judged,
                        "{} carries an apex probe on feature {:?} that is \
                         JUDGED. The plan pre-registered a criterion for edges, \
                         strokes and interiors and none for apex rounding, so \
                         judging one means a threshold was invented after \
                         seeing the data. It is measured - apex_displacement is \
                         a column - and that is all",
                        target.name,
                        m.feature
                    );
                    assert!(
                        m.apex_displacement.is_finite(),
                        "{}: an unjudged probe still has to produce a NUMBER, \
                         or leaving it unjudged is just dropping it",
                        target.name
                    );
                }
                ProbeKind::Edge => judged += 1,
                ProbeKind::Stroke => judged += 1,
                ProbeKind::Interior => judged += 1,
            }
        }
        assert_eq!(
            metric.judged, judged,
            "{} judged {} probes where {judged} are of a judged kind",
            target.name, metric.judged
        );
    }
    assert!(
        apex_probes >= 2,
        "only {apex_probes} apex probes exist across the six targets, so this \
         law is asserted on almost nothing"
    );
}

#[test]
fn the_targets_scatter_densely_enough_to_carry_a_profile() {
    let criterion = Criterion::registered();
    let mut checked = 0usize;
    for target in Target::<f64>::all() {
        let image = control(&target, AnalyticArm::Sdf);
        let metric = match measure_target(&image, &target, &criterion) {
            Ok(m) => m,
            Err(e) => panic!("{e}"),
        };
        for m in metric.probes.iter() {
            match m.kind {
                ProbeKind::Interior => continue,
                ProbeKind::Apex => continue,
                ProbeKind::Edge => (),
                ProbeKind::Stroke => (),
            }
            checked += 1;
            assert!(
                m.bins.len() * 4 >= m.samples as usize,
                "{} probe on feature {:?} spread {} samples over only {} \
                 distinct bins of signed distance. The whole metrology rests on \
                 the pixel offsets EQUIDISTRIBUTING against the geometry - the \
                 same Weyl argument that puts the edge at 30 degrees. A disc \
                 centred on a pixel corner, or a stem parallel to the pixel \
                 rows, collapses the abscissa to a handful of values and there \
                 is no profile left to measure, however many samples are \
                 gathered. This is why T4's centre is offset by irrational \
                 fractions of a pixel and T6 is rotated",
                target.name,
                m.feature,
                m.samples,
                m.bins.len()
            );
        }
    }
    assert!(checked >= 6, "only {checked} profiles were checked for density");
}

#[test]
fn one_failing_probe_fails_the_target_however_many_others_pass() {
    let target = Target::<f64>::t3_right_angle_corner();
    let criterion = Criterion::registered();
    let (width, height) = target.viewport;
    let spoiled = FeatureId(1);
    let mut image = FloatImage::<f64>::new(width, height);
    for py in 0..height {
        for px in 0..width {
            let centre = pixel_centre(px, py);
            let sdf = target.sdf(centre);
            let value = if target.nearest_feature(centre) == spoiled {
                (0.5 - sdf / 6.0).clamp(0.0, 1.0)
            } else {
                (0.5 - sdf).clamp(0.0, 1.0)
            };
            image.set(px, py, Rgb::splat(value));
        }
    }
    let metric = match measure_target(&image, &target, &criterion) {
        Ok(m) => m,
        Err(e) => panic!("{e}"),
    };
    let mut passed = 0usize;
    let mut failed = 0usize;
    for m in metric.probes.iter() {
        if !m.judged {
            continue;
        }
        if m.meets { passed += 1 } else { failed += 1 }
    }
    assert!(
        passed >= 1 && failed >= 1,
        "the spoiled image was built so that one judged probe passes and one \\
         fails; it produced {passed} passing and {failed} failing, so the \\
         aggregation rule below is asserted on nothing"
    );
    assert!(
        !metric.meets,
        "one edge of the corner transitions over six pixels and the other over \\
         one, and the target came back as MEETING its criterion. The aggregate \\
         is worst-probe-wins: a target passes only when EVERY judged probe \\
         passes. Taking the best probe instead would let a glyph whose bowl is \\
         a smear pass on the strength of its stem"
    );
    assert!(
        metric.transition_10_90 > 1.0,
        "the reported width {} is the BEST of the two edges, not the worst. \\
         The column is the worst judged probe or it is not a bound on anything",
        metric.transition_10_90
    );
}

#[test]
fn a_profile_with_no_descent_reports_no_width_rather_than_inventing_one() {
    let target = Target::<f64>::t2_straight_edge();
    let image = painted(&target, |sdf| {
        let x = sdf + 1.0;
        (-(x * x) / (2.0 * 0.25)).exp()
    });
    let metric = measured(&target, &image, 0);
    assert!(
        metric.contrast > 0.0,
        "the bump image must still present a positive contrast, or the width \\
         is refused by the contrast guard rather than by the bracket and this \\
         test proves nothing about the bracket. Contrast came back {}",
        metric.contrast
    );
    assert!(
        !metric.transition_10_90.is_finite() && !metric.sdf_90.is_finite(),
        "the profile rises from zero to a bump and falls again, so the 0.9 \\
         level is never crossed from ABOVE inside the window and there is no \\
         10-90 transition to report. The instrument returned a width of {} \\
         with the 0.9 crossing at {}. A crossing that was never bracketed must \\
         come back as no measurement: falling back to the first bin's abscissa \\
         invents a width out of a profile that has none, and at low atom counts \\
         - where the fit is still blobby and this is exactly what the profile \\
         looks like - that fabricated number would be read as a result",
        metric.transition_10_90,
        metric.sdf_90
    );
}

#[test]
fn the_registered_criterion_is_the_one_the_plan_pre_registered() {
    let c = Criterion::registered();
    let registered: [(&str, f64, f64); 6] = [
        ("max_transition_px", c.max_transition, 1.0),
        ("min_contrast_frac", c.min_contrast_frac, 0.90),
        ("min_peak_frac", c.min_peak_frac, 0.90),
        ("target_fwhm_px", c.target_fwhm, 1.0),
        ("max_fwhm_error_px", c.max_fwhm_error, 0.5),
        ("max_interior_ripple", c.max_interior_ripple, 1.0 / 255.0),
    ];
    for (name, measured_value, pre_registered) in registered {
        assert!(
            (measured_value - pre_registered).abs() < 1.0e-15,
            "criterion.{name} is {measured_value} and the plan pre-registered \
             {pre_registered}. Every atom count in the P2 report is the first \
             count at which a cell crosses THESE six numbers, so loosening one \
             of them moves a headline without touching a single measurement. \
             The suite carried no pin on any of the six until this test: a \
             1.0 to 2.0 change in max_transition_px alone turns t2 from never \
             meeting inside eight atoms into meeting at six, and every other \
             test still passes. If a threshold is to change, change it here \
             and re-run the report, so the change is visible in the diff \
             rather than in a number nobody can source",
            measured_value = measured_value,
            pre_registered = pre_registered
        );
    }
    let target = Target::<f64>::t2_straight_edge();
    let dim = painted(&target, |sdf| 0.6 * (0.5 - sdf).clamp(0.0, 1.0));
    let metric = measured(&target, &dim, 0);
    assert!(
        (metric.contrast - 0.6).abs() < 1.0e-12
            && (metric.transition_10_90 - SDF_ARM_TRUE_TRANSITION).abs() < 1.0e-12,
        "the dim ramp was painted to carry contrast 0.6 over the exact \
         {SDF_ARM_TRUE_TRANSITION} px transition of the unit ramp, and the \
         instrument reads contrast {} over {} px. This assertion exists so the \
         one below cannot pass for the wrong reason",
        metric.contrast,
        metric.transition_10_90
    );
    assert!(
        !metric.meets,
        "an edge whose transition is a clean {} px but whose contrast reaches \
         only {} of the target's own {} FAILS the registered criterion: a \
         sharp grey edge is not a crisp edge. The gate is contrast >= \
         min_contrast_frac * TARGET contrast, and comparing contrast against \
         itself instead makes that half of the edge criterion vacuous while \
         every other test in this file still passes",
        metric.transition_10_90,
        metric.contrast,
        metric.target_contrast
    );
}
