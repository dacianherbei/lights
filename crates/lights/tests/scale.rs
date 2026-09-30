#![cfg(feature = "fit")]
use lights::fit::analytic::{sdf_coverage, sdf_coverage_through};
use lights::fit::atom2d::{Atom2d, camera};
use lights::fit::content::{N2_CENTRE, N2_ERF_MAX_ERROR, RASTER_HEIGHT, RASTER_WIDTH, erf,
    n1_field, n1_sample, n2_soft_shadow_sample, pixel_sample};
use lights::fit::coverage::{DEFAULT_CELL_LIMIT, DEFAULT_MAX_DEPTH, coverage_budget, exact_coverage};
use lights::fit::coverage::coverage_image_to_colour;
use lights::fit::edge::{Criterion, EdgeMetric, Mapping, TargetMetric, measure_target};
use lights::fit::greedy::{
    Initialisation, UNIT_STEP_SCALE, centre_step, log_scale_bounds, render_or_background,
    seed_atom, seed_atom_scaled, seed_sigma,
};
use lights::fit::scale::{
    Convention, SUPERSAMPLES_PER_SCENE_UNIT, Sampler, Scale, ScaleClass, Similarity2, box_average,
    judge_transition,
    measure_scene, scaled_atoms, scaled_target,
};
use lights::fit::target::{EdgeProbe, Part, Primitive, Target};
use lights::image::FloatImage;
use lights::kernel::KernelFamily;
use lights::numerics::vector::{Rgb, Vec2};
use lights::settings::SharedSettings;

fn ladder() -> Vec<Scale> {
    let mut out = Vec::new();
    for (n, d) in [(1u32, 4u32), (1, 2), (1, 1), (2, 1), (4, 1), (8, 1)] {
        match Scale::new(n, d) {
            Ok(s) => out.push(s),
            Err(e) => panic!("{n}/{d} is not a scale: {e}"),
        }
    }
    out
}

fn scale_of(n: u32, d: u32) -> Scale {
    match Scale::new(n, d) {
        Ok(s) => s,
        Err(e) => panic!("{n}/{d} is not a scale: {e}"),
    }
}

fn scaled(target: &Target<f64>, s: Scale) -> Target<f64> {
    match scaled_target(target, s) {
        Ok(t) => t,
        Err(e) => panic!("{} does not scale to {}: {e}", target.name, s.label()),
    }
}

#[test]
fn a_scaled_description_carries_a_signed_distance_that_scales_with_it() {
    let criterion = 1.0e-9;
    let mut checked = 0usize;
    for target in Target::<f64>::all() {
        for s in ladder() {
            let big = scaled(&target, s);
            let factor = s.factor();
            for y in 0..target.viewport.1 {
                for x in 0..target.viewport.0 {
                    let p = Vec2::new(x as f64 + 0.5, y as f64 + 0.5);
                    let want = factor * target.sdf(p);
                    let got = big.sdf(Vec2::new(p.x * factor, p.y * factor));
                    let slack = criterion * (1.0 + want.abs());
                    assert!(
                        (got - want).abs() <= slack,
                        "{} at {}: sdf({:?}) is {got}, but a signed distance that scales with \
                         the description must be {want}. If this drifts, every scene-unit \
                         number in the aspect-3 tables is measured against the wrong ruler",
                        target.name,
                        s.label(),
                        p
                    );
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 100_000, "the sdf scan covered only {checked} points");

    let mut probes_seen = 0usize;
    for target in Target::<f64>::all() {
        for s in ladder() {
            let big = scaled(&target, s);
            let factor = s.factor();
            for (a, b) in target.probes.iter().zip(big.probes.iter()) {
                assert!((b.anchor.x - a.anchor.x * factor).abs() < 1.0e-9);
                assert!((b.anchor.y - a.anchor.y * factor).abs() < 1.0e-9);
                assert!(
                    (b.window - a.window * factor).abs() < 1.0e-9,
                    "{} at {}: a probe window is a distance in the scene and must scale with \
                     it. A window left in base units gathers a different set of pixels at \
                     every scale, so the profile would be measured over a shrinking patch and \
                     the decay curve would be an artefact of the instrument",
                    target.name,
                    s.label()
                );
                assert_eq!(a.feature, b.feature);
                probes_seen += 1;
            }
        }
    }
    assert!(probes_seen >= 30, "only {probes_seen} probes were checked");
}

#[test]
fn a_scale_that_does_not_divide_the_viewport_is_refused_instead_of_rounded() {
    let odd = scale_of(1, 3);
    let refused = odd.viewport((64, 64));
    assert!(
        refused.is_err(),
        "64 is not divisible by 3, so a third-scale viewport would have to be rounded. A \
         rounded viewport silently changes the sampling rate the whole ladder is measuring, \
         so it must be an error and not a nearest integer"
    );
    let fine = scale_of(1, 4);
    match fine.viewport((64, 64)) {
        Ok(v) => assert_eq!(v, (16, 16), "a scale that divides exactly is exact"),
        Err(e) => panic!("a quarter of 64 is an integer: {e}"),
    }
    match scale_of(8, 1).viewport((32, 32)) {
        Ok(v) => assert_eq!(v, (256, 256)),
        Err(e) => panic!("{e}"),
    }
}

#[test]
fn the_supersample_rate_holds_the_scene_sample_spacing_fixed_across_the_ladder() {
    for s in ladder() {
        let k = s.supersample();
        let spacing = 1.0 / ((k as f64) * s.factor());
        assert!(
            spacing <= 1.0 / (SUPERSAMPLES_PER_SCENE_UNIT as f64) + 1.0e-12,
            "at {} the truth is sampled every {spacing} scene units, coarser than the \
             1/{SUPERSAMPLES_PER_SCENE_UNIT} the whole ladder is held to. A truth that is \
             better resolved at one scale than another turns a property of the instrument \
             into a result about the representation",
            s.label()
        );
    }
    assert_eq!(scale_of(8, 1).supersample(), 1);
    assert_eq!(scale_of(1, 4).supersample(), 16);
}

#[test]
fn the_continuous_truth_is_the_same_sampler_the_earlier_aspect_used() {
    let one = scale_of(1, 1);
    let point_sampled = box_average(&Sampler::ColourField, (RASTER_WIDTH, RASTER_HEIGHT), one, 1);
    let earlier = n1_field();
    for y in 0..RASTER_HEIGHT {
        for x in 0..RASTER_WIDTH {
            let a = point_sampled.get(x, y);
            let b = earlier.get(x, y);
            assert_eq!(
                (a.x.to_bits(), a.y.to_bits(), a.z.to_bits()),
                (b.x.to_bits(), b.y.to_bits(), b.z.to_bits()),
                "at one sample per pixel the scale-aware truth must be the aspect-2 field \
                 BIT FOR BIT: any difference means the ladder is measuring a different \
                 content class and the two runs cannot be compared at all"
            );
        }
    }
    let integrated = box_average(
        &Sampler::ColourField,
        (RASTER_WIDTH, RASTER_HEIGHT),
        one,
        one.supersample(),
    );
    let mut worst = 0.0f64;
    for (p, q) in integrated.pixels.iter().zip(earlier.pixels.iter()) {
        worst = worst
            .max((p.x - q.x).abs())
            .max((p.y - q.y).abs())
            .max((p.z - q.z).abs());
    }
    assert!(
        worst > 0.0,
        "the pixel-box integral must NOT coincide with the point sample, or the ladder is \
         not integrating at all"
    );
    assert!(
        worst < 0.2,
        "the pixel-box integral of the colour field differs from the aspect-2 point sample by \
         {worst}, which is too large to be a quadrature difference"
    );
}

#[test]
fn the_continuous_truth_reads_one_field_at_every_scale() {
    let sampler = Sampler::ColourField;
    for s in ladder() {
        let viewport = match s.viewport((RASTER_WIDTH, RASTER_HEIGHT)) {
            Ok(v) => v,
            Err(e) => panic!("{}: {e}", s.label()),
        };
        let image = box_average(&sampler, viewport, s, s.supersample());
        for (px, py) in [(0u32, 0u32), (viewport.0 / 2, viewport.1 / 2)] {
            let centre = Vec2::new(
                (px as f64 + 0.5) / s.factor(),
                (py as f64 + 0.5) / s.factor(),
            );
            let direct = n1_sample(centre);
            let got = image.get(px, py);
            let gap = (got.x - direct.x)
                .abs()
                .max((got.y - direct.y).abs())
                .max((got.z - direct.z).abs());
            assert!(
                gap < 0.35,
                "at {} pixel ({px},{py}) integrates to {got:?} but the field at its centre is \
                 {direct:?}: the box average has drifted off the field it is meant to average",
                s.label()
            );
        }
    }
    let _ = pixel_sample(0, 0);
}

#[test]
fn freezing_an_atom_set_scales_the_footprint_and_leaves_the_shape_alone() {
    let base = vec![
        Atom2d::from_sigma(
            Vec2::new(12.5, 7.25),
            Vec2::new(3.0, 0.5),
            0.7,
            0.9,
            Rgb::new(0.2, 0.4, 0.6),
        ),
        Atom2d::from_sigma(
            Vec2::new(1.0, 2.0),
            Vec2::new(0.75, 0.75),
            -1.1,
            0.4,
            Rgb::new(1.0, 0.0, 0.5),
        ),
    ];
    let unit = scaled_atoms(&base, scale_of(1, 1));
    for (a, b) in base.iter().zip(unit.iter()) {
        assert_eq!(a.bits(), b.bits(), "the unit scale must be the identity BIT FOR BIT");
    }
    for s in ladder() {
        let moved = scaled_atoms(&base, s);
        let factor = s.factor();
        for (a, b) in base.iter().zip(moved.iter()) {
            assert!((b.centre.x - a.centre.x * factor).abs() < 1.0e-12);
            assert!((b.centre.y - a.centre.y * factor).abs() < 1.0e-12);
            assert!((b.sigma().x - a.sigma().x * factor).abs() < 1.0e-12);
            assert!((b.sigma().y - a.sigma().y * factor).abs() < 1.0e-12);
            assert_eq!(
                a.angle.to_bits(),
                b.angle.to_bits(),
                "a similarity does not rotate an atom"
            );
            assert_eq!(a.opacity.to_bits(), b.opacity.to_bits());
            assert!(
                (a.anisotropy() - b.anisotropy()).abs() < 1.0e-9,
                "scaling must not change the anisotropy: an atom that changed shape under \
                 zoom would be a different description, and the frozen arm would be measuring \
                 nothing"
            );
        }
    }
}

#[test]
fn a_frozen_atom_field_is_the_same_continuous_field_at_every_scale() {
    let settings = SharedSettings::<f64>::contract();
    let base_viewport = (24u32, 24u32);
    let atoms = vec![
        Atom2d::from_sigma(
            Vec2::new(9.5, 11.5),
            Vec2::new(2.5, 1.25),
            0.6,
            0.85,
            Rgb::new(0.9, 0.3, 0.1),
        ),
        Atom2d::from_sigma(
            Vec2::new(15.0, 8.0),
            Vec2::new(1.5, 1.5),
            0.0,
            0.5,
            Rgb::new(0.1, 0.7, 0.9),
        ),
    ];
    let one = scale_of(1, 1);
    let three = scale_of(3, 1);
    let small = match render_or_background(
        &atoms,
        &camera(base_viewport.0, base_viewport.1),
        KernelFamily::Gaussian,
        &settings,
        16,
    ) {
        Ok(i) => i,
        Err(e) => panic!("the unit render failed: {e}"),
    };
    let big_viewport = match three.viewport(base_viewport) {
        Ok(v) => v,
        Err(e) => panic!("{e}"),
    };
    let big = match render_or_background(
        &scaled_atoms(&atoms, three),
        &camera(big_viewport.0, big_viewport.1),
        KernelFamily::Gaussian,
        &settings,
        16,
    ) {
        Ok(i) => i,
        Err(e) => panic!("the tripled render failed: {e}"),
    };
    let mut worst = 0.0f64;
    for y in 0..base_viewport.1 {
        for x in 0..base_viewport.0 {
            let a = small.get(x, y);
            let b = big.get(3 * x + 1, 3 * y + 1);
            worst = worst
                .max((a.x - b.x).abs())
                .max((a.y - b.y).abs())
                .max((a.z - b.z).abs());
        }
    }
    assert!(
        worst < 1.0e-12,
        "the centre subpixel of every tripled pixel samples EXACTLY the scene point the unit \
         pixel sampled, so the two renders must agree to floating point there; the worst gap \
         is {worst}. This is the whole claim of arm 1: an atom set is a continuous field and \
         rendering it at another rate re-samples it rather than re-approximating it",
    );
    let _ = FloatImage::<f64>::new(1, 1);
    let _ = scaled_atoms(&atoms, one);
}

#[test]
fn the_analytic_arm_holds_its_transition_in_pixels_and_sharpens_in_scene_units() {
    let criterion = Criterion::registered();
    let base = Target::<f64>::t2_straight_edge();
    for s in [scale_of(1, 1), scale_of(2, 1), scale_of(4, 1)] {
        let big = scaled(&base, s);
        let image = coverage_image_to_colour(&big, &sdf_coverage(&big));
        let metric = match measure_scene(&image, &base, s, &criterion) {
            Ok(m) => m,
            Err(e) => panic!("{}: {e}", s.label()),
        };
        let judged = judge_transition(&metric, s, &criterion);
        assert!(
            (judged.device - 1.0).abs() < 0.25,
            "at {} the analytic arm measures {} device pixels of 10-90 transition. It is a \
             one-pixel ramp BY CONSTRUCTION at every scale, so the DEVICE column must be flat",
            s.label(),
            judged.device
        );
        assert!(
            (judged.scene * s.factor() - judged.device).abs() < 1.0e-9,
            "the two conventions must differ by exactly the scale factor and nothing else"
        );
        assert!(
            judged.meets_device,
            "the analytic control must meet the registered criterion under the pixel \
             convention at every scale, or the criterion is invalid rather than the arm"
        );
    }
    let four = scale_of(4, 1);
    let big = scaled(&base, four);
    let image = coverage_image_to_colour(&big, &sdf_coverage(&big));
    let metric = match measure_scene(&image, &base, four, &criterion) {
        Ok(m) => m,
        Err(e) => panic!("{e}"),
    };
    let judged = judge_transition(&metric, four, &criterion);
    assert!(
        judged.scene < 0.4,
        "in SCENE units the analytic arm at 4x must be about four times sharper than at 1x \
         ({} measured). A criterion held in pixels would call this unchanged and a criterion \
         held in scene units calls it four times better: that is why the convention has to be \
         stated before any number is read",
        judged.scene
    );
    assert_eq!(
        Convention::SceneUnits.convert(0.25, four),
        0.25,
        "the scene convention is the identity"
    );
    assert_eq!(Convention::DevicePixels.convert(0.25, four), 1.0);
}

#[test]
fn the_measurement_mapping_at_the_unit_scale_is_the_instrument_the_earlier_runs_used() {
    let identity = Mapping::identity();
    let unit = Mapping::at_scale(1.0);
    assert_eq!(identity, unit);
    assert_eq!(identity.scene_of_pixel(3, 5), Vec2::new(3.5, 5.5));
    assert_eq!(Mapping::at_scale(4.0).scene_of_pixel(3, 5), Vec2::new(0.875, 1.375));
    assert_eq!(Mapping::at_scale(4.0).device_viewport((64, 32)), (256, 128));
}

#[test]
fn the_fitter_search_can_be_put_in_scene_units_and_the_unit_setting_is_the_old_one() {
    assert_eq!(centre_step(UNIT_STEP_SCALE), lights::fit::greedy::CENTRE_STEP);
    assert_eq!(seed_sigma(UNIT_STEP_SCALE), lights::fit::greedy::SEED_SIGMA);
    assert_eq!(
        log_scale_bounds(UNIT_STEP_SCALE),
        (
            lights::fit::greedy::MIN_LOG_SCALE,
            lights::fit::greedy::MAX_LOG_SCALE
        )
    );
    let eight = 8.0f64;
    assert!((centre_step(eight) - 8.0 * lights::fit::greedy::CENTRE_STEP).abs() < 1.0e-12);
    assert!((seed_sigma(eight) - 8.0 * lights::fit::greedy::SEED_SIGMA).abs() < 1.0e-12);
    let (lo, hi) = log_scale_bounds(eight);
    assert!(
        (hi.exp() - eight * lights::fit::greedy::MAX_LOG_SCALE.exp()).abs() < 1.0e-6,
        "the largest atom the fitter may build is a PIXEL constant; at 8x it must be allowed \
         to be eight times larger in pixels or the refitted arm is capped by an accident of \
         units rather than by the representation"
    );
    assert!((lo.exp() - eight * lights::fit::greedy::MIN_LOG_SCALE.exp()).abs() < 1.0e-6);
}

#[test]
fn every_scale_class_produces_a_truth_of_exactly_the_viewport_it_claims() {
    let mut budget = coverage_budget(DEFAULT_CELL_LIMIT);
    let classes = vec![
        ScaleClass::Analytic(Target::<f64>::t2_straight_edge()),
        ScaleClass::Analytic(Target::<f64>::t6_glyph_like()),
        ScaleClass::colour_field(),
        ScaleClass::band_limited_noise(),
        ScaleClass::broadband_patch(),
    ];
    for class in classes.iter() {
        for s in [scale_of(1, 2), scale_of(1, 1), scale_of(2, 1)] {
            let viewport = match class.viewport_at(s) {
                Ok(v) => v,
                Err(e) => panic!("{} at {}: {e}", class.name(), s.label()),
            };
            let truth = match class.truth(s, DEFAULT_MAX_DEPTH, &mut budget) {
                Ok(t) => t,
                Err(e) => panic!("{} at {}: {e}", class.name(), s.label()),
            };
            assert_eq!(
                (truth.width, truth.height),
                viewport,
                "{} at {} produced a truth of the wrong size",
                class.name(),
                s.label()
            );
            let scaffold = match class.scaffold(s) {
                Ok(t) => t,
                Err(e) => panic!("{e}"),
            };
            assert_eq!(scaffold.viewport, viewport);
        }
    }
}

fn primitive_bits(shape: &Primitive<f64>, out: &mut Vec<u64>) {
    match shape {
        Primitive::HalfPlane { point, normal } => {
            out.push(0);
            out.push(point.x.to_bits());
            out.push(point.y.to_bits());
            out.push(normal.x.to_bits());
            out.push(normal.y.to_bits());
        }
        Primitive::Segment { a, b, half_width } => {
            out.push(1);
            out.push(a.x.to_bits());
            out.push(a.y.to_bits());
            out.push(b.x.to_bits());
            out.push(b.y.to_bits());
            out.push(half_width.to_bits());
        }
        Primitive::Arc {
            centre,
            radius,
            from,
            to,
            half_width,
        } => {
            out.push(2);
            out.push(centre.x.to_bits());
            out.push(centre.y.to_bits());
            out.push(radius.to_bits());
            out.push(from.to_bits());
            out.push(to.to_bits());
            out.push(half_width.to_bits());
        }
        Primitive::Disc { centre, radius } => {
            out.push(3);
            out.push(centre.x.to_bits());
            out.push(centre.y.to_bits());
            out.push(radius.to_bits());
        }
    }
}

fn parts_bits(parts: &[Part<f64>], out: &mut Vec<u64>) {
    out.push(parts.len() as u64);
    for part in parts.iter() {
        out.push(part.id.0 as u64);
        primitive_bits(&part.shape, out);
    }
}

fn probes_bits(probes: &[EdgeProbe<f64>], out: &mut Vec<u64>) {
    out.push(probes.len() as u64);
    for probe in probes.iter() {
        out.push(probe.feature.0 as u64);
        out.push(probe.anchor.x.to_bits());
        out.push(probe.anchor.y.to_bits());
        out.push(probe.window.to_bits());
        out.push(probe.kind as u64);
    }
}

fn target_bits(target: &Target<f64>) -> Vec<u64> {
    let mut out = Vec::new();
    out.push(target.viewport.0 as u64);
    out.push(target.viewport.1 as u64);
    for c in [target.set.ink, target.paper] {
        out.push(c.x.to_bits());
        out.push(c.y.to_bits());
        out.push(c.z.to_bits());
    }
    parts_bits(&target.set.parts, &mut out);
    parts_bits(&target.set.holes, &mut out);
    probes_bits(&target.probes, &mut out);
    out
}

fn edge_metric_bits(metric: &EdgeMetric, out: &mut Vec<u64>) {
    out.push(metric.feature.0 as u64);
    out.push(metric.kind as u64);
    for v in [
        metric.transition_10_90,
        metric.sdf_90,
        metric.sdf_10,
        metric.contrast,
        metric.target_contrast,
        metric.peak,
        metric.fwhm,
        metric.plateau_low,
        metric.plateau_high,
        metric.apex_displacement,
        metric.interior_ripple,
    ] {
        out.push(v.to_bits());
    }
    out.push(metric.bins.len() as u64);
    for b in metric.bins.iter() {
        out.push(b.sdf.to_bits());
        out.push(b.mean.to_bits());
        out.push(b.samples as u64);
    }
    out.push(metric.samples as u64);
    out.push(metric.thin_bins as u64);
    out.push(metric.judged as u64);
    out.push(metric.meets as u64);
}

fn target_metric_bits(metric: &TargetMetric) -> Vec<u64> {
    let mut out = Vec::new();
    out.push(metric.probes.len() as u64);
    for probe in metric.probes.iter() {
        edge_metric_bits(probe, &mut out);
    }
    for v in [
        metric.transition_10_90,
        metric.contrast_frac,
        metric.peak_frac,
        metric.fwhm,
        metric.fwhm_error,
        metric.apex_displacement,
        metric.interior_ripple,
    ] {
        out.push(v.to_bits());
    }
    out.push(metric.thin_bins as u64);
    out.push(metric.judged as u64);
    out.push(metric.meets as u64);
    out
}

fn image_bits(image: &FloatImage<f64>) -> Vec<u64> {
    let mut out = Vec::new();
    out.push(image.width as u64);
    out.push(image.height as u64);
    for y in 0..image.height {
        for x in 0..image.width {
            let c = image.get(x, y);
            out.push(c.x.to_bits());
            out.push(c.y.to_bits());
            out.push(c.z.to_bits());
        }
    }
    out
}

#[test]
fn the_unit_rung_of_the_ladder_is_the_single_scale_run_bit_for_bit() {
    let one = scale_of(1, 1);
    assert_eq!(
        Mapping::at_scale(one.factor()).scene_of_pixel(0, 0),
        Vec2::new(0.5, 0.5),
        "the unit rung must sample the CENTRE of pixel (0,0). Agreement between the old and \
         the new entry points is not enough on its own: they share one mapping, so a shift \
         common to both would move the two sides together and leave the comparison silent. \
         This literal is the anchor the comparison hangs from"
    );
    assert_eq!(
        Mapping::at_scale(one.factor()).scene_of_pixel(7, 3),
        Vec2::new(7.5, 3.5)
    );
    let criterion = Criterion::registered();
    let mut budget = coverage_budget(DEFAULT_CELL_LIMIT);
    let targets = vec![
        Target::<f64>::t2_straight_edge(),
        Target::<f64>::t3_right_angle_corner(),
        Target::<f64>::t5_thin_stroke(),
        Target::<f64>::t6_glyph_like(),
    ];
    for base in targets.iter() {
        assert_eq!(
            target_bits(&scaled(base, one)),
            target_bits(base),
            "{}: scaling by exactly 1.0 must return the SAME description, bit for bit; a \
             description that drifts at the unit rung makes every ratio along the ladder a \
             comparison between two different experiments",
            base.name
        );

        let old = match exact_coverage(base, DEFAULT_MAX_DEPTH, &mut budget) {
            Ok((c, _)) => coverage_image_to_colour(base, &c),
            Err(e) => panic!("{}: {e}", base.name),
        };
        let ladder = match ScaleClass::Analytic(base.clone()).truth(one, DEFAULT_MAX_DEPTH, &mut budget) {
            Ok(t) => t,
            Err(e) => panic!("{}: {e}", base.name),
        };
        assert_eq!(
            image_bits(&ladder),
            image_bits(&old),
            "{}: the ladder's truth at the unit rung must be the truth the single-scale runs \
             used, bit for bit",
            base.name
        );

        let single = match measure_target(&old, base, &criterion) {
            Ok(m) => m,
            Err(e) => panic!("{}: {e}", base.name),
        };
        let mapped = match measure_scene(&old, base, one, &criterion) {
            Ok(m) => m,
            Err(e) => panic!("{}: {e}", base.name),
        };
        assert_eq!(
            target_metric_bits(&mapped),
            target_metric_bits(&single),
            "{}: at a scale of exactly 1.0 the scale-aware metrology must reproduce the \
             existing single-scale numbers BIT FOR BIT. This is the control: without it a \
             change anywhere along the ladder could be the new instrument rather than the \
             description under test",
            base.name
        );
        assert_eq!(
            mapped.meets, single.meets,
            "{}: the verdict must be the old verdict too",
            base.name
        );
    }

    let settings = SharedSettings::<f64>::contract();
    let mut truth = FloatImage::<f64>::new(9, 9);
    let mut residual = FloatImage::<f64>::new(9, 9);
    for y in 0..9u32 {
        for x in 0..9u32 {
            let a = (x as f64) * 0.07 + (y as f64) * 0.011;
            truth.set(x, y, Rgb::new(a, 1.0 - a, 0.5 * a));
            residual.set(x, y, Rgb::new(a * 0.5, a * 0.25, a));
        }
    }
    for init in [Initialisation::Isotropic, Initialisation::StructureTensor] {
        let old = seed_atom(&truth, &residual, (4, 5), init, &settings);
        let unit = seed_atom_scaled(&truth, &residual, (4, 5), init, &settings, UNIT_STEP_SCALE);
        assert_eq!(
            old.bits(),
            unit.bits(),
            "the unit step scale must seed the atom the old fitter seeded, bit for bit"
        );
    }
}

#[test]
fn the_soft_shadow_blur_is_a_scene_length_and_widens_with_the_zoom() {
    let class = ScaleClass::soft_shadow();
    assert_eq!(class.name(), "n2-soft-shadow");

    let width_at = |scale: Scale| -> f64 {
        let mut budget = coverage_budget(DEFAULT_CELL_LIMIT);
        let truth = match class.truth(scale, DEFAULT_MAX_DEPTH, &mut budget) {
            Ok(t) => t,
            Err(e) => panic!("truth at {scale:?} failed: {e:?}"),
        };
        let ink_free = |row: u32| -> bool {
            (0..truth.width).all(|x| {
                let p = truth.get(x, row);
                p.z - p.x < 0.2
            })
        };
        let mut row = u32::MAX;
        let mut best = 0.0f64;
        for candidate in 0..truth.height {
            if !ink_free(candidate) {
                continue;
            }
            let mut bright = f64::NEG_INFINITY;
            let mut dark = f64::INFINITY;
            for x in 0..truth.width {
                let v = truth.get(x, candidate).y;
                bright = bright.max(v);
                dark = dark.min(v);
            }
            if bright - dark > best {
                best = bright - dark;
                row = candidate;
            }
        }
        assert!(
            row != u32::MAX,
            "every row of the truth contains ink, so no row measures the PENUMBRA alone. \
             The card's edge is hard and would report a one-pixel transition at every scale"
        );
        let mut lo = f64::NAN;
        let mut hi = f64::NAN;
        let mut bright = f64::NEG_INFINITY;
        let mut dark = f64::INFINITY;
        for x in 0..truth.width {
            let v = truth.get(x, row).y;
            bright = bright.max(v);
            dark = dark.min(v);
        }
        let span = bright - dark;
        assert!(
            span > 0.05,
            "the chosen scan line must cross the shadow: row {row}, brightest {bright}, \
             darkest {dark}"
        );
        let ten = dark + 0.1 * span;
        let ninety = dark + 0.9 * span;
        for x in 0..truth.width {
            let v = truth.get(x, row).y;
            if lo.is_nan() {
                if v < ninety {
                    lo = x as f64;
                }
                continue;
            }
            if v <= ten {
                hi = x as f64;
                break;
            }
        }
        assert!(
            lo.is_finite() && hi.is_finite() && hi > lo,
            "the LEADING 90-to-10 descent must be found on the scan line, before the card \
             itself is reached: {lo} to {hi}. Measuring darkest-to-brightest across the whole \
             row would span the shadow core AND the card, and would scale with the zoom \
             whatever the blur did - an instrument that cannot fail the test it is written for"
        );
        hi - lo
    };

    let one = width_at(Scale::new(1, 1).expect("scale"));
    let two = width_at(Scale::new(2, 1).expect("scale"));
    let four = width_at(Scale::new(4, 1).expect("scale"));

    println!("N2S widths in device pixels: 1x {one:.3}  2x {two:.3}  4x {four:.3}");
    assert!(
        (two / one - 2.0).abs() < 0.25,
        "a blur expressed in SCENE units must cover twice as many DEVICE pixels at 2x: \
         {two:.3} against {one:.3} is a ratio of {:.3}. A shadow whose sigma is fixed in \
         pixels would hold this ratio at 1.0 and the class would be measuring the blur \
         radius shrinking rather than the representation",
        two / one
    );
    assert!(
        (four / one - 4.0).abs() < 0.5,
        "and four times as many at 4x: {four:.3} against {one:.3}, ratio {:.3}",
        four / one
    );

    let mut budget = coverage_budget(DEFAULT_CELL_LIMIT);
    let base = match class.truth(Scale::new(1, 1).expect("scale"), DEFAULT_MAX_DEPTH, &mut budget) {
        Ok(t) => t,
        Err(e) => panic!("truth failed: {e:?}"),
    };
    let card = base.get(N2_CENTRE.0 as u32, N2_CENTRE.1 as u32);
    assert!(
        (card.x - 0.22).abs() < 1.0e-9
            && (card.y - 0.35).abs() < 1.0e-9
            && (card.z - 0.62).abs() < 1.0e-9,
        "the CLASS's own truth must carry n2's ink at the card centre, got {card:?}. Measuring \
         the sampler function directly leaves the Sampler::SoftShadow dispatch arm untested, \
         and a mutation that routed this class to n1's colour field ESCAPED until this \
         assertion existed"
    );

    let paper = n2_soft_shadow_sample(Vec2::new(1.0, 1.0));
    let ink = n2_soft_shadow_sample(Vec2::new(N2_CENTRE.0, N2_CENTRE.1));
    assert!(
        (paper.x - 0.93).abs() < 0.02,
        "far from the card the sampler must return the paper colour, got {paper:?}"
    );
    assert!(
        (ink.z - 0.62).abs() < 1.0e-12,
        "inside the card the sampler must return the ink colour, got {ink:?}"
    );
    assert!(
        erf(0.0).abs() < N2_ERF_MAX_ERROR && (erf(3.0) - 0.9999779095).abs() < N2_ERF_MAX_ERROR,
        "the error function this class is built on must be the Abramowitz-Stegun form, \
         accurate to {N2_ERF_MAX_ERROR:e}: erf(3) = {}",
        erf(3.0)
    );
}

fn similarity(scale: f64, rotation: f64, translation: Vec2<f64>) -> Similarity2<f64> {
    match Similarity2::new(scale, rotation, translation) {
        Ok(v) => v,
        Err(e) => panic!("{scale} {rotation} {translation:?} is not a similarity: {e}"),
    }
}

fn largest_gap(a: &FloatImage<f64>, b: &FloatImage<f64>) -> f64 {
    assert_eq!((a.width, a.height), (b.width, b.height));
    let mut worst = 0.0f64;
    for y in 0..a.height {
        for x in 0..a.width {
            worst = worst.max((a.get(x, y).x - b.get(x, y).x).abs());
        }
    }
    worst
}

#[test]
fn a_view_through_a_similarity_agrees_with_rewriting_the_target() {
    let tolerance = 1.0e-9;
    let mut compared = 0usize;
    for target in Target::<f64>::all() {
        for s in ladder() {
            let Ok(viewport) = s.viewport(target.viewport) else {
                continue;
            };
            let oracle = sdf_coverage(&scaled(&target, s));
            let viewed = sdf_coverage_through(&target, Similarity2::uniform(s.factor()), viewport);
            let gap = largest_gap(&oracle, &viewed);
            assert!(
                gap <= tolerance,
                "{} at {}: viewing through the similarity differs from rewriting the target by {gap}",
                target.name,
                s.label()
            );
            compared += 1;
        }
    }
    assert!(compared >= Target::<f64>::all().len() * 4, "only {compared} cells compared");
}

#[test]
fn a_quarter_turn_about_the_viewport_centre_moves_every_pixel_to_its_rotated_mate() {
    let side = 64u32;
    let half = side as f64 / 2.0;
    let turn = similarity(1.0, std::f64::consts::FRAC_PI_2, Vec2::new(2.0 * half, 0.0));
    let pivot = turn.apply(Vec2::new(half, half));
    assert!((pivot.x - half).abs() < 1.0e-12 && (pivot.y - half).abs() < 1.0e-12);
    let mut visible = 0usize;
    for target in Target::<f64>::all() {
        let upright = sdf_coverage_through(&target, Similarity2::identity(), (side, side));
        let turned = sdf_coverage_through(&target, turn, (side, side));
        if largest_gap(&upright, &turned) > 1.0e-3 {
            visible += 1;
        }
        for py in 0..side {
            for px in 0..side {
                let want = upright.get(py, side - 1 - px).x;
                let got = turned.get(px, py).x;
                assert!(
                    (want - got).abs() <= 1.0e-9,
                    "{} pixel ({px},{py}) shows {got}, its rotated mate shows {want}",
                    target.name
                );
            }
        }
    }
    assert!(visible >= 1, "no target looks different after a quarter turn, so the check saw nothing");
}

#[test]
fn a_zoom_then_a_pan_composes_into_one_similarity_and_inverts_to_the_start() {
    let zoom = similarity(2.5, 0.3, Vec2::new(0.0, 0.0));
    let pan = similarity(1.0, 0.0, Vec2::new(-11.0, 7.25));
    let tilt = similarity(0.75, -1.1, Vec2::new(3.0, -4.0));
    let both = zoom.then(pan).then(tilt);
    for p in [Vec2::new(0.0, 0.0), Vec2::new(13.5, -2.25), Vec2::new(-40.0, 31.0)] {
        let stepwise = tilt.apply(pan.apply(zoom.apply(p)));
        let composed = both.apply(p);
        assert!(
            (stepwise.x - composed.x).abs() < 1.0e-9 && (stepwise.y - composed.y).abs() < 1.0e-9,
            "{p:?}: stepwise {stepwise:?} but composed {composed:?}"
        );
        let back = both.invert(composed);
        assert!(
            (back.x - p.x).abs() < 1.0e-9 && (back.y - p.y).abs() < 1.0e-9,
            "{p:?} came back as {back:?}"
        );
    }
}

#[test]
fn a_similarity_refuses_a_scale_that_is_not_positive_and_finite() {
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(Similarity2::new(bad, 0.0, Vec2::new(0.0, 0.0)).is_err(), "scale {bad} was accepted");
    }
    assert!(Similarity2::new(1.0, f64::NAN, Vec2::new(0.0, 0.0)).is_err());
    assert!(Similarity2::new(1.0, 0.0, Vec2::new(f64::NAN, 0.0)).is_err());
}
