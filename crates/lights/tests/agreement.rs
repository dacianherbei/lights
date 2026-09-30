#![cfg(all(feature = "method-splat", feature = "method-raycast"))]
use lights::agreement::{
    AgreementReport, ContractViolation, FIRST_ORDER_COEFFICIENT, HEADROOM,
    SECOND_ORDER_COEFFICIENT, assert_contract, atom_bound, atom_bound_terms,
    compare, compose_tolerance, compose_tolerance_with_truncation, convention_divergence, envelope,
    run_pair,
};
use lights::blend::EarlyOut;
use lights::camera::{Camera, CameraSpec, Projection};
use lights::dispatch::{Collect, Precision, RenderOutcome, RenderRequest, render};
use lights::error::LightsError;
use lights::image::FloatImage;
use lights::kernel::KernelFamily;
use lights::method::RenderMethod;
use lights::numerics::matrix::Mat3;
use lights::numerics::quaternion::Quat;
use lights::numerics::symmetric::Sym3;
use lights::numerics::vector::{Rgb, Vec3};
use lights::scene::demo::{
    Lcg, sphere_camera, spheres_anisotropic, spheres_anisotropic_no_ground, single_atom_scene,
};
use lights::scene::{Atom, Scene};
use lights::settings::{
    DirectionalLight, MIN_SCALE, OrderingModel, REFERENCE_NEAR_PLANE, SharedSettings,
    SupportMode,
};

fn pinhole(focal: f64, principal: [f64; 2], width: u32, height: u32) -> CameraSpec {
    CameraSpec::look_at(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
        45.0,
        width,
        height,
    )
    .with_intrinsics([focal, focal], principal)
}

fn camera_of(spec: &CameraSpec) -> Camera<f64> {
    match Camera::<f64>::from_spec(spec, REFERENCE_NEAR_PLANE) {
        Ok(c) => c,
        Err(e) => panic!("camera: {e:?}"),
    }
}

fn scene_of(atoms: &[Atom<f64>]) -> Scene<f64> {
    let mut scene = Scene::<f64>::new();
    for atom in atoms {
        match scene.push(*atom, MIN_SCALE) {
            Ok(()) => (),
            Err(e) => panic!("push: {e:?}"),
        }
    }
    scene
}

fn request(
    scene: Scene<f64>,
    camera: CameraSpec,
    settings: SharedSettings<f64>,
    method: RenderMethod,
) -> RenderRequest {
    RenderRequest {
        precision: Precision::F64,
        kernel: KernelFamily::Gaussian,
        method,
        settings,
        camera,
        scene,
        collect: Collect::Pixels,
        shapes: None,
    }
}

fn render_ok(r: &RenderRequest) -> RenderOutcome {
    match render(r) {
        Ok(o) => o,
        Err(e) => panic!("render: {e:?}"),
    }
}

fn try_pair(
    scene: Scene<f64>,
    camera: CameraSpec,
    settings: SharedSettings<f64>,
) -> lights::error::Result<(AgreementReport, RenderOutcome, RenderOutcome, FloatImage<f64>)> {
    let base = request(scene, camera, settings, RenderMethod::splat_contract());
    run_pair(
        &base,
        &RenderMethod::splat_contract(),
        &RenderMethod::raycast_contract(),
    )
}

fn pair(
    scene: Scene<f64>,
    camera: CameraSpec,
    settings: SharedSettings<f64>,
) -> (AgreementReport, RenderOutcome, RenderOutcome, FloatImage<f64>) {
    match try_pair(scene, camera, settings) {
        Ok(v) => v,
        Err(e) => panic!("run_pair: {e:?}"),
    }
}

fn worst_channel(a: &FloatImage<f64>, b: &FloatImage<f64>) -> f64 {
    let mut worst = 0.0f64;
    for (l, r) in a.pixels.iter().zip(b.pixels.iter()) {
        worst = worst
            .max((l.x - r.x).abs())
            .max((l.y - r.y).abs())
            .max((l.z - r.z).abs());
    }
    worst
}

fn isotropic(sigma: f64) -> Sym3<f64> {
    Sym3::new(sigma * sigma, 0.0, 0.0, sigma * sigma, 0.0, sigma * sigma)
}

#[test]
fn tv_x1_centre_pixel_exact() {
    let settings = SharedSettings::<f64>::contract();
    let mut rng = Lcg::new(0x5eed_0001);
    let target = 3u32;
    let mut worst = 0.0f64;
    let mut worst_case = String::new();
    let count = 400usize;

    for index in 0..count {
        let z = 2.0 + rng.unit() * 8.0;
        let (x, y) = if index == 0 {
            (0.0, 0.0)
        } else if index == 1 {
            (3.4 * z, 0.0)
        } else {
            (rng.signed() * 0.7 * z, rng.signed() * 0.7 * z)
        };
        let sigma = 0.01 + rng.unit() * 0.25;
        let opacity = 0.05 + rng.unit() * 0.9;
        let focal = 400.0 + rng.unit() * 900.0;
        let half = rng.unit() * std::f64::consts::PI;
        let axis = match Vec3::new(rng.signed(), rng.signed(), rng.signed()).normalize() {
            Some(v) => v,
            None => Vec3::new(0.0, 0.0, 1.0),
        };
        let sin = half.sin();
        let atom = Atom::new(
            Vec3::new(x, y, z),
            Quat::new_wxyz(half.cos(), axis.x * sin, axis.y * sin, axis.z * sin),
            Vec3::new(
                sigma * (0.4 + rng.unit()),
                sigma * (0.4 + rng.unit()),
                sigma * (0.4 + rng.unit()),
            ),
            opacity,
            Rgb::splat(1.0),
        );
        let principal = [
            target as f64 - focal * x / z,
            target as f64 - focal * y / z,
        ];
        let spec = CameraSpec::look_at(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
            45.0,
            7,
            7,
        )
        .with_intrinsics([focal, focal], principal);

        let scene = match single_atom_scene(atom) {
            Ok(s) => s,
            Err(e) => panic!("scene: {e:?}"),
        };
        let splat = render_ok(&request(
            scene.clone(),
            spec.clone(),
            settings,
            RenderMethod::splat_contract(),
        ));
        let ray = render_ok(&request(
            scene,
            spec,
            settings,
            RenderMethod::raycast_contract(),
        ));
        if splat.report.clamped_atoms > 0 {
            continue;
        }
        let a = splat.image.get(target, target).x;
        let b = ray.image.get(target, target).x;
        let expected = opacity.min(settings.blend.alpha_cap);
        let e = (a - expected).abs().max((b - expected).abs());
        let d = (a - b).abs();
        if e.max(d) > worst {
            worst = e.max(d);
            worst_case = format!(
                "atom {index}: mu ({x}, {y}, {z}) sigma {sigma} o {opacity} f {focal} \
                 splat {a} ray {b} expected {expected}"
            );
        }
    }
    assert!(
        worst <= 1.0e-12,
        "TV-X1: at the pixel where an atom's centre projects both methods must \
         return exactly o, because the ray-direction deviation is ZERO there \
         regardless of depth, footprint or field angle. worst {worst} at {worst_case}"
    );
}

#[test]
fn tv_x2_orthographic_every_pixel() {
    let settings = SharedSettings::<f64>::contract();
    let mut rng = Lcg::new(0x5eed_0002);
    let mut atoms = Vec::new();
    for _ in 0..14 {
        let half = rng.unit() * std::f64::consts::PI;
        let axis = match Vec3::new(rng.signed(), rng.signed(), rng.signed()).normalize() {
            Some(v) => v,
            None => Vec3::new(0.0, 0.0, 1.0),
        };
        let sin = half.sin();
        atoms.push(Atom::new(
            Vec3::new(rng.signed() * 0.7, rng.signed() * 0.5, 4.0 + rng.unit() * 4.0),
            Quat::new_wxyz(half.cos(), axis.x * sin, axis.y * sin, axis.z * sin),
            Vec3::new(
                0.03 + rng.unit() * 0.09,
                0.03 + rng.unit() * 0.09,
                0.03 + rng.unit() * 0.16,
            ),
            0.2 + rng.unit() * 0.6,
            Rgb::new(rng.unit(), rng.unit(), rng.unit()),
        ));
    }
    let spec = CameraSpec::look_at(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
        45.0,
        128,
        96,
    )
    .orthographic(0.8);
    assert_eq!(spec.projection, Projection::Orthographic);

    let scene = scene_of(&atoms);
    let splat = render_ok(&request(
        scene.clone(),
        spec.clone(),
        settings,
        RenderMethod::splat_contract(),
    ));
    let ray = render_ok(&request(
        scene,
        spec,
        settings,
        RenderMethod::raycast_contract(),
    ));
    assert_eq!(splat.report.clamped_atoms, 0);
    assert_eq!(ray.report.clamped_atoms, 0);
    let (peak, _, _) = splat.image.max_abs();
    assert!(peak > 0.4, "the scene must actually render: {peak}");

    let worst = worst_channel(&splat.image, &ray.image);
    assert!(
        worst <= 1.0e-14,
        "TV-X2: under an ORTHOGRAPHIC projection the Jacobian is exact and \
         constant, so the two methods are mathematically IDENTICAL at every \
         pixel. A failure here is unambiguously a bug with no legitimate \
         explanation and nothing may be tuned to make it pass. worst {worst}"
    );
}

#[test]
fn tv_x11_alpha_convention_divergence_is_documented() {
    assert_eq!(convention_divergence(0.5, 1.0), 0.0);
    assert_eq!(convention_divergence(0.9, 1.0), 0.0);
    assert!(convention_divergence(0.5, 0.0).abs() < 1e-15);

    for (opacity, expected_max, expected_at) in [
        (0.10f64, 0.001317f64, 0.4956f64),
        (0.25, 0.008980, 0.48),
        (0.50, 0.043036, 0.4712),
        (0.90, 0.241959, 0.4080),
    ] {
        let mut best = 0.0f64;
        let mut best_g = 0.0f64;
        for step in 0..=200000u32 {
            let g = step as f64 / 200000.0;
            let d = convention_divergence(opacity, g);
            if d > best {
                best = d;
                best_g = g;
            }
        }
        assert!(
            (best - expected_max).abs() < 1e-5,
            "o {opacity}: peak-versus-Beer-Lambert divergence {best} against the \
             math sheet's {expected_max}"
        );
        assert!(
            (best_g - expected_at).abs() < 0.02,
            "o {opacity}: the maximum sits at g {best_g}, sheet says {expected_at}"
        );
    }
    assert!(
        convention_divergence(0.9, 0.408) * 255.0 > 60.0,
        "at o = 0.9 the convention difference is 62/255 - obviously visible. \
         This function ships so that when someone later compares against a \
         volumetric renderer and sees a 4-24% gap, the crate already says it is \
         a convention and not a bug"
    );
}

#[test]
fn run_pair_refuses_every_kernel_whose_blend_or_curve_the_tolerance_was_not_derived_for() {
    let camera = sphere_camera(320, 180);
    let settings = SharedSettings::<f64>::contract();
    let scene = spheres_anisotropic_no_ground::<f64>();

    let mut ok = request(
        scene.clone(),
        camera.clone(),
        settings,
        RenderMethod::splat_contract(),
    );
    ok.kernel = KernelFamily::Gaussian;
    match run_pair(
        &ok,
        &RenderMethod::splat_contract(),
        &RenderMethod::raycast_contract(),
    ) {
        Ok(_) => (),
        Err(e) => panic!("the gaussian is the kernel the bound was derived for: {e:?}"),
    }

    let cases: [(KernelFamily, ContractViolation); 2] = [
        (
            KernelFamily::GaussianNormalised,
            ContractViolation::BlendClassNotOver {
                kernel: KernelFamily::GaussianNormalised.name(),
            },
        ),
        (
            KernelFamily::CompactBump,
            ContractViolation::KernelCurveNotGaussian {
                kernel: KernelFamily::CompactBump.name(),
            },
        ),
    ];
    for (family, want) in cases {
        let mut bad = request(
            spheres_anisotropic_no_ground::<f64>(),
            camera.clone(),
            settings,
            RenderMethod::splat_contract(),
        );
        bad.kernel = family;
        match run_pair(
            &bad,
            &RenderMethod::splat_contract(),
            &RenderMethod::raycast_contract(),
        ) {
            Err(LightsError::ContractViolation(got)) => assert_eq!(
                got, want,
                "{} must be refused by name, or the harness composes a \
                 transmittance-weighted tolerance for a blend that has no transmittance",
                family.name()
            ),
            Err(e) => panic!("{}: wrong error {e:?}", family.name()),
            Ok(_) => panic!(
                "{} was COMPARED. The tolerance in bound.rs was derived for an \
                 over-composited gaussian and is a confident wrong number for anything else",
                family.name()
            ),
        }
    }
}

#[test]
fn tv_x12_precondition_gate() {
    let base = SharedSettings::<f64>::contract();
    let scene = scene_of(&[Atom::isotropic(
        Vec3::new(0.0, 0.0, 5.0),
        0.05,
        0.8,
        Rgb::splat(1.0),
    )]);
    match assert_contract(&base, &base, 0, 0, &scene) {
        Ok(()) => (),
        Err(e) => panic!("the contract settings must satisfy their own gate: {e:?}"),
    }

    let mut cases: Vec<(&'static str, SharedSettings<f64>, ContractViolation)> = Vec::new();
    let mut s = base;
    s.screen_dilation = 0.3;
    cases.push(("dilation", s, ContractViolation::DilationEnabled));
    let mut s = base;
    s.energy_compensation = true;
    cases.push((
        "energy compensation",
        s,
        ContractViolation::EnergyCompensationMismatch,
    ));
    let mut s = base;
    s.blend.alpha_cutoff = 0.01;
    cases.push(("alpha cutoff", s, ContractViolation::AlphaCutoffMismatch));
    let mut s = base;
    s.blend.transmittance_cutoff = 1e-3;
    cases.push((
        "transmittance cutoff",
        s,
        ContractViolation::TransmittanceCutoffMismatch,
    ));
    let mut s = base;
    s.blend.alpha_cap = 0.999;
    cases.push(("alpha cap", s, ContractViolation::AlphaCapMismatch));
    let mut s = base;
    s.blend.early_out = EarlyOut::Reference;
    cases.push(("early out", s, ContractViolation::EarlyOutMismatch));
    let mut s = base;
    s.ordering = OrderingModel::HitDepth;
    cases.push(("ordering", s, ContractViolation::OrderingMismatch));
    let mut s = base;
    s.support = SupportMode::FixedSigmas(3.0);
    cases.push(("support mode", s, ContractViolation::SupportModeMismatch));
    let mut s = base;
    s.near_plane = 0.5;
    cases.push(("near plane", s, ContractViolation::NearPlaneMismatch));

    let mut seen: Vec<ContractViolation> = Vec::new();
    for (name, other, expected) in cases {
        match assert_contract(&base, &other, 0, 0, &scene) {
            Err(LightsError::ContractViolation(v)) => {
                assert_eq!(v, expected, "{name}: wrong violation reported");
                assert!(
                    !seen.contains(&v),
                    "{name}: every violation must be individually distinguishable, \
                     because each has a characteristic fake-bug magnitude"
                );
                seen.push(v);
            }
            other => panic!("{name} must be refused, got {other:?}"),
        }
    }
    assert_eq!(
        seen.len(),
        9,
        "NINE, not ten: ResponseMismatch was removed on 2026-09-28 (L-117) with \
         ResponseModel itself, because pruning SegmentIntegral left the enum one value \
         and a comparison of one value with itself can never fire. TEN, not eleven: \
         PixelConventionMismatch was removed on 2026-09-14 because it \
         guarded SharedSettings::convention, a field NO renderer read — the camera's own \
         convention decides the sample point — and both call sites of assert_contract pass \
         the same settings struct twice, so it could never fire under any input. This count \
         is asserted rather than left to drift precisely so that removing a guard has to be \
         a decision someone writes down"
    );

    let mut fixed = SharedSettings::<f64>::contract();
    fixed.support = SupportMode::FixedSigmas(3.0);
    match assert_contract(&fixed, &fixed, 0, 0, &scene) {
        Err(LightsError::ContractViolation(ContractViolation::SupportNotCutoffSolved)) => (),
        other => panic!(
            "a cross-method run must refuse a fixed-sigma support outright. The \
             truncation channel of compose_tolerance_with_truncation is sized at \
             2 * c_max * eps_alpha, which is the step ONLY when the truncation \
             radius is the cutoff radius. Under FixedSigmas(n) the step is \
             2 * c_max * o * exp(-n^2/2). Got {other:?}"
        ),
    }

    match assert_contract(&base, &base, 0, 4, &scene) {
        Err(LightsError::ContractViolation(
            ContractViolation::DeterminantCulledAtomsPresent { count },
        )) => assert_eq!(
            count, 4,
            "the determinant cull has no ray-casting counterpart - the ray path \
             never forms Sigma_2D - so it is a ONE-SIDED cull of the same class \
             as the Jacobian clamp, and must be reported as a count rather than \
             folded silently into AtomsCulledDet"
        ),
        other => panic!("a determinant-culled atom must be refused, got {other:?}"),
    }

    match assert_contract(&base, &base, 7, 0, &scene) {
        Err(LightsError::ContractViolation(ContractViolation::ClampedAtomsPresent {
            count,
        })) => assert_eq!(
            count, 7,
            "assert_contract must report the COUNT so an exclusion is visible in \
             the test output rather than silent"
        ),
        other => panic!("a clamped atom must be refused, got {other:?}"),
    }

    let mut tiny = SharedSettings::<f64>::contract();
    tiny.scale_floor = 0.1;
    match assert_contract(&tiny, &tiny, 0, 0, &scene) {
        Err(LightsError::ContractViolation(ContractViolation::ScaleBelowFloor { index })) => {
            assert_eq!(index, 0)
        }
        other => panic!("a scale below the floor must be refused, got {other:?}"),
    }
}

#[test]
fn compose_tolerance_is_the_math_sheet_formula_plus_a_truncation_term() {
    let accumulated = FloatImage {
        width: 2,
        height: 1,
        pixels: vec![Rgb::splat(0.5), Rgb::splat(0.0)],
    };
    let plain = compose_tolerance(&accumulated, HEADROOM, 1.0, 1.0e-4);
    assert!((plain.get(0, 0).x - (1.05 * 2.0 * 0.5 + 1.0e-4)).abs() < 1e-18);
    assert_eq!(
        plain.get(1, 0).x,
        1.0e-4,
        "a pixel no atom reaches still carries the early-out floor eps_T"
    );

    let truncation = FloatImage {
        width: 2,
        height: 1,
        pixels: vec![Rgb::splat(3.0), Rgb::splat(0.0)],
    };
    let eps_alpha = 1.0 / 255.0;
    let with_step = match compose_tolerance_with_truncation(
        &accumulated,
        &truncation,
        HEADROOM,
        1.0,
        eps_alpha,
        1.0e-4,
    ) {
        Ok(v) => v,
        Err(e) => panic!("{e:?}"),
    };
    let expected = 1.05 * 2.0 * 0.5 + 2.0 * eps_alpha * 3.0 + 1.0e-4;
    assert!((with_step.get(0, 0).x - expected).abs() < 1e-18);
    assert!(
        with_step.get(0, 0).x > plain.get(0, 0).x,
        "the math sheet's Tol = 1.05*2*sum T_i*B_i + eps_T is INCOMPLETE: both \
         methods cut their kernel at their own r_cut, so on the annulus where \
         the two radii straddle it one composites an alpha of exactly eps_alpha \
         and the other composites nothing. That step is bounded by \
         2*c_max*T_i*eps_alpha per evaluated atom and it is not in the sheet"
    );

    let mismatched = FloatImage::<f64>::new(3, 1);
    assert!(matches!(
        compose_tolerance_with_truncation(
            &accumulated,
            &mismatched,
            HEADROOM,
            1.0,
            eps_alpha,
            1.0e-4
        ),
        Err(LightsError::SizeMismatch { .. })
    ));

    let a = FloatImage {
        width: 2,
        height: 1,
        pixels: vec![Rgb::new(1.0, 5.0, 2.0), Rgb::splat(0.0)],
    };
    let b = FloatImage {
        width: 2,
        height: 1,
        pixels: vec![Rgb::new(4.0, 3.0, 2.0), Rgb::splat(1.0)],
    };
    let e = match envelope(&a, &b) {
        Ok(v) => v,
        Err(err) => panic!("{err:?}"),
    };
    assert_eq!(e.get(0, 0), Rgb::new(4.0, 5.0, 2.0));
    assert_eq!(e.get(1, 0), Rgb::splat(1.0));
}

#[test]
fn tv_x13_tolerance_image_end_to_end() {
    let settings = SharedSettings::<f64>::contract();
    let spec = sphere_camera(320, 180);
    let scene = spheres_anisotropic_no_ground::<f64>();
    let (report, splat, ray, tolerance) = pair(scene, spec, settings);

    println!("TV-X13 {report:?}");
    let (peak, _, _) = splat.image.max_abs();
    println!("splat peak {peak}");
    assert!(peak > 0.5, "the scene must actually render: {peak}");
    assert_eq!(splat.method, lights::method::MethodId::SplatRaster);
    assert_eq!(ray.method, lights::method::MethodId::RayCast);
    assert_eq!(
        report.clamped_atoms, 0,
        "the ground atom is excluded by construction, so nothing may be clamped"
    );
    assert_eq!(report.pixels, 320 * 180);
    assert!(
        report.is_clean(),
        "TV-X13: {} of {} pixels exceed the per-pixel tolerance. Worst pixel \
         {:?}: measured {} against tolerance {} (ratio {}). rms {}",
        report.failures,
        report.pixels,
        report.worst_pixel,
        report.worst_difference,
        report.worst_tolerance,
        report.worst_ratio,
        report.rms_difference
    );
    assert!(
        report.worst_ratio > 0.0,
        "a zero worst ratio means the two renders are bit-identical, which for a \
         PERSPECTIVE camera would mean one of the renderers is not running"
    );
    let (tol_peak, _, _) = tolerance.max_abs();
    assert!(tol_peak > 0.0);

    assert!(
        report.tolerance_max > 0.9,
        "the demo camera is 121 degrees horizontal, where the EWA \
         local-affine approximation genuinely breaks down, so the JUSTIFIED \
         tolerance reaches {} on a [0,1] image. That is honest, and it is also \
         why is_clean() alone is nearly unfalsifiable here: at that scale the \
         assertion absorbs a 40 percent kernel-amplitude error. The two-sided \
         pins below are what make this test bite. If this number has moved, the \
         bound itself has changed and every pin here must be re-derived rather \
         than widened",
        report.tolerance_max
    );
    pin(
        "TV-X13 worst ratio",
        report.worst_ratio,
        0.394_527_791_840_987_1,
    );
    pin(
        "TV-X13 rms difference",
        report.rms_difference,
        0.024_332_530_492_729_834,
    );
}

fn pin(what: &str, measured: f64, expected: f64) {
    let relative = (measured - expected).abs() / expected;
    assert!(
        relative < 0.01,
        "{what} is PINNED, two-sided, at {expected}; measured {measured}, a \
         relative move of {relative}. A one-sided is_clean() cannot see a change \
         that shifts the whole frame without moving the worst pixel past its \
         tolerance: unnormalising the ray direction in Camera::ray moves the rms \
         by 2.4 percent and the worst pixel by 3e-16, and every agreement test \
         that asserts only is_clean() passes through it. Re-derive this number \
         when the bound or the renderers change on purpose; never widen it to \
         make a failure go away"
    );
}

#[test]
fn tv_x13b_tolerance_is_genuinely_per_pixel() {
    let settings = SharedSettings::<f64>::contract();
    let spec = sphere_camera(320, 180);
    let scene = spheres_anisotropic_no_ground::<f64>();
    let (report, _, _, _) = pair(scene, spec, settings);
    println!(
        "TV-X13b span {} min {} max {}",
        report.tolerance_span, report.tolerance_min, report.tolerance_max
    );
    assert!(
        report.tolerance_span > 50.0,
        "B_atom varies by roughly 800x across a single frame, so ONE GLOBAL \
         TOLERANCE IS NOT A TEST BUT A FUDGE FACTOR. Measured span {}",
        report.tolerance_span
    );
}

const EPS_ALPHA: f64 = 1.0 / 255.0;

struct BoundMeasurement {
    bulk: f64,
    fringe: f64,
    predicted: f64,
    argmax_r: f64,
}

fn measure_single_atom(
    scale: Vec3<f64>,
    rotation: Quat<f64>,
    offset: f64,
    z: f64,
) -> BoundMeasurement {
    measure_on(scale, rotation, offset, z, 200.0, 400, 120)
}

fn measure_finely(
    scale: Vec3<f64>,
    rotation: Quat<f64>,
    offset: f64,
    z: f64,
) -> BoundMeasurement {
    measure_on(scale, rotation, offset, z, 400.0, 700, 200)
}

fn measure_on(
    scale: Vec3<f64>,
    rotation: Quat<f64>,
    offset: f64,
    z: f64,
    focal: f64,
    width: u32,
    height: u32,
) -> BoundMeasurement {
    let settings = SharedSettings::<f64>::contract();
    let principal = [
        width as f64 * 0.5 - focal * offset / z,
        height as f64 * 0.5,
    ];
    let spec = pinhole(focal, principal, width, height);
    let camera = camera_of(&spec);
    let limits = camera.jacobian_limits();
    assert!(
        limits.x > offset / z,
        "the off-axis clamp has NO ray-casting counterpart, so every test atom \
         must sit inside it: limit {} against tan {}",
        limits.x,
        offset / z
    );

    let mean = Vec3::new(offset, 0.0, z);
    let atom = Atom::new(mean, rotation, scale, 1.0, Rgb::splat(1.0));
    let scene = scene_of(&[atom]);
    let splat = render_ok(&request(
        scene.clone(),
        spec.clone(),
        settings,
        RenderMethod::splat_contract(),
    ));
    let ray = render_ok(&request(
        scene,
        spec,
        settings,
        RenderMethod::raycast_contract(),
    ));
    assert_eq!(splat.report.clamped_atoms, 0);

    let sigma_cam = atom.covariance();
    let predicted = atom_bound(mean, &sigma_cam, 1.0);
    let lambda = match sigma_cam.inverse_spd() {
        Some(l) => l,
        None => panic!("the test atom must be SPD"),
    };
    let centre = match camera.project(mean) {
        Some(p) => p,
        None => panic!("the test atom must project"),
    };

    let mut bulk = 0.0f64;
    let mut fringe = 0.0f64;
    let mut argmax_r = 0.0f64;
    for y in 0..height {
        for x in 0..width {
            let a = splat.image.get(x, y).x;
            let b = ray.image.get(x, y).x;
            let difference = (a - b).abs();
            if a >= 2.0 * EPS_ALPHA && b >= 2.0 * EPS_ALPHA {
                if difference > bulk {
                    bulk = difference;
                    let sample = camera.sample_point(x, y);
                    let dx = (sample.x - centre.x) / focal * z;
                    let dy = (sample.y - centre.y) / focal * z;
                    let offset3 = Vec3::new(dx, dy, 0.0);
                    argmax_r = lambda.quad_form(offset3).max(0.0).sqrt();
                }
            } else {
                fringe = fringe.max(difference);
            }
        }
    }
    BoundMeasurement {
        bulk,
        fringe,
        predicted,
        argmax_r,
    }
}

#[test]
fn tv_x8_bound_tightness_sweep() {
    let shapes = [
        ("isotropic", Vec3::new(0.05, 0.05, 0.05)),
        ("needle in depth", Vec3::new(0.02, 0.02, 0.30)),
        ("disc facing the eye", Vec3::new(0.30, 0.30, 0.02)),
        ("general", Vec3::new(0.12, 0.04, 0.25)),
    ];
    let half: f64 = 0.61;
    let axis = Vec3::new(0.4, 0.5, 0.7681145747868608);
    let sin = half.sin();
    let rotations = [
        ("identity", Quat::<f64>::identity()),
        (
            "fixed random",
            Quat::new_wxyz(half.cos(), axis.x * sin, axis.y * sin, axis.z * sin),
        ),
    ];

    let mut rows = 0usize;
    let mut smallest = f64::INFINITY;
    let mut largest = 0.0f64;
    let mut disc_on_axis_identity = 0.0f64;
    let mut disc_on_axis_rotated = 0.0f64;

    for tan in [0.0f64, 0.5, 1.0] {
        for (shape_name, scale) in shapes {
            for (rotation_name, rotation) in rotations {
                let m = measure_single_atom(scale, rotation, tan * 5.0, 5.0);
                rows += 1;
                smallest = smallest.min(m.bulk);
                largest = largest.max(m.bulk);
                let ratio = m.bulk / m.predicted;
                println!(
                    "TV-X8 tan {tan} {shape_name} / {rotation_name}: measured \
                     {} predicted {} ratio {ratio} argmax r {} fringe {}",
                    m.bulk, m.predicted, m.argmax_r, m.fringe
                );
                assert!(
                    (0.90..=1.02).contains(&ratio),
                    "tan {tan} {shape_name} / {rotation_name}: measured {} against \
                     the closed form {} is a ratio of {ratio}. Above 1.02 the \
                     closed form is wrong; far below it the bound is loose and \
                     hides real bugs",
                    m.bulk,
                    m.predicted
                );
                if tan == 0.0 && shape_name == "disc facing the eye" {
                    if rotation_name == "identity" {
                        disc_on_axis_identity = m.bulk;
                    } else {
                        disc_on_axis_rotated = m.bulk;
                    }
                }
            }
        }
    }
    assert_eq!(rows, 24);
    println!(
        "TV-X8 span {} to {} ({}x)",
        smallest,
        largest,
        largest / smallest
    );
    assert!(
        largest / smallest > 100.0,
        "the measured maxima must span orders of magnitude, which is exactly why \
         a global tolerance is a fudge factor: {smallest} to {largest}"
    );
    assert!(
        disc_on_axis_rotated > 100.0 * disc_on_axis_identity,
        "a disc PERPENDICULAR to the view axis on-axis is the smallest error \
         ({disc_on_axis_identity}) because the DEPTH extent, not the footprint, \
         drives the divergence; the same disc rotated is {disc_on_axis_rotated}"
    );
}

#[test]
fn tv_x9_on_axis_quadratic_law() {
    let mut previous: Option<(f64, f64)> = None;
    for sigma_z in [0.2f64, 0.1, 0.05, 0.025] {
        let m = measure_single_atom(
            Vec3::new(0.05, 0.05, sigma_z),
            Quat::identity(),
            0.0,
            5.0,
        );
        println!(
            "TV-X9 sigma_z {sigma_z}: measured {} predicted {}",
            m.bulk, m.predicted
        );
        let expected = 1.0826822658929016 * (sigma_z / 5.0) * (sigma_z / 5.0);
        assert!(
            (m.predicted / expected - 1.0).abs() < 0.02,
            "on-axis the bound is exactly 1.082682*(sigma_z/z)^2, QUADRATIC in \
             the depth extent and INDEPENDENT of the transverse footprint: \
             predicted {} against {expected}",
            m.predicted
        );
        if let Some((prev_sigma, prev)) = previous {
            let exponent = (prev / m.bulk).ln() / (prev_sigma / sigma_z).ln();
            assert!(
                (exponent - 2.0).abs() < 0.10,
                "the fitted exponent is {exponent}, not 2"
            );
        }
        previous = Some((sigma_z, m.bulk));
    }

    let wide = measure_single_atom(Vec3::new(0.2, 0.2, 0.05), Quat::identity(), 0.0, 5.0);
    let thin = measure_single_atom(Vec3::new(0.02, 0.02, 0.05), Quat::identity(), 0.0, 5.0);
    assert!(
        (wide.bulk / thin.bulk - 1.0).abs() < 0.06,
        "a 10x change in the TRANSVERSE footprint must barely move the on-axis \
         divergence: {} against {}",
        wide.bulk,
        thin.bulk
    );
}

#[test]
fn tv_x10_off_axis_linear_law_and_sin_versus_tan() {
    let sigma = 0.05f64;
    let z = 5.0f64;
    let mut argmaxes = Vec::new();
    for tan in [0.0f64, 0.25, 0.5, 1.0] {
        let m = measure_finely(Vec3::splat(sigma), Quat::identity(), tan * z, z);
        println!(
            "TV-X10 tan {tan}: measured {} predicted {} ratio {} argmax r {}",
            m.bulk,
            m.predicted,
            m.bulk / m.predicted,
            m.argmax_r
        );
        assert!(
            (0.90..=1.02).contains(&(m.bulk / m.predicted)),
            "tan {tan}: measured {} against the closed form {}",
            m.bulk,
            m.predicted
        );
        argmaxes.push((tan, m.argmax_r));
    }

    assert!(
        (argmaxes[0].1 - 2.0).abs() < 0.13,
        "ON AXIS the divergence is the pure second-order r^4 e^(-r^2/2) term and \
         its maximum sits at r = 2 exactly. Measured {}. This row is what proves \
         the whitened radius is reconstructed correctly, which is why the \
         off-axis rows below are a finding and not an instrument error",
        argmaxes[0].1
    );
    assert!(
        (argmaxes[1].1 - 1.75).abs() < 0.13,
        "at tan theta = 0.25 the first-order r^3 e^(-r^2/2) law does switch on \
         and the maximum moves IN to r = 1.75, exactly as the math sheet says: {}",
        argmaxes[1].1
    );
    assert!(
        argmaxes[3].1 > argmaxes[1].1 + 0.4,
        "MATH SHEET DISCREPANCY, MEASURED IN THIS SUITE. The sheet asserts the \
         off-axis argmax is r = 1.750 +- 0.03 for EVERY off-axis row. It is not. \
         Measured on a 0.25-wide whitened grid: tan 0 -> 2.00, tan 0.25 -> 1.75, \
         tan 0.5 -> 2.00, tan 1.0 -> 2.50. The maximum moves in as the \
         first-order term switches on and then back OUT as the field angle \
         grows, because the sheet takes the argmax of the first-order term \
         ALONE - legitimate for building the BOUND, but not the argmax of the \
         sum. The magnitudes reproduce to 0.3% at every row, so the bound itself \
         stands; only its argmax annotation does not. Measured {argmaxes:?}"
    );
    for (tan, r) in argmaxes.iter() {
        assert!(
            (1.6..=2.7).contains(r),
            "tan {tan}: argmax r {r} is outside the band where the divergence \
             law can put it at all"
        );
    }

    let a = measure_finely(Vec3::splat(0.05), Quat::identity(), 0.5 * z, z);
    let b = measure_finely(Vec3::splat(0.10), Quat::identity(), 0.5 * z, z);
    let growth = b.bulk / a.bulk;
    assert!(
        (growth - 2.0).abs() < 0.25,
        "off axis the divergence is LINEAR in sigma/z: doubling sigma multiplied \
         it by {growth}"
    );

    for tan in [0.5f64, 1.0] {
        let sin_theta = tan / (1.0 + tan * tan).sqrt();
        let terms = atom_bound_terms(Vec3::new(tan * z, 0.0, z), &isotropic(sigma), 1.0);
        assert!(
            (terms.sin_theta - sin_theta).abs() < 1e-15,
            "the instrument must REPORT the field angle it uses, not leave the \
             test to reimplement it: a future edit that substitutes tan for sin \
             inside the bound would otherwise be caught only by this test's own \
             arithmetic. Reported {} against {sin_theta}",
            terms.sin_theta
        );
        let first_with_sin = FIRST_ORDER_COEFFICIENT * (sigma / z) * terms.sin_theta;
        assert!(
            (terms.first_order / first_with_sin - 1.0).abs() < 1e-9,
            "the off-axis factor is sin(theta), not tan(theta): {} against {}",
            terms.first_order,
            first_with_sin
        );
        let with_tan = FIRST_ORDER_COEFFICIENT * (sigma / z) * tan;
        let inverse_cos = (1.0 + tan * tan).sqrt();
        assert!(
            (with_tan / first_with_sin - inverse_cos).abs() < 1e-12,
            "substituting tan for sin overshoots by exactly 1/cos(theta) = \
             {inverse_cos}, which is 41% at 45 degrees and turns a rigorous \
             bound into one that lets real bugs through at wide field angles"
        );
    }
}

#[test]
fn tv_x3_the_divergence_changes_sign_across_the_atom_centre() {
    let settings = SharedSettings::<f64>::contract();
    let z = 5.0f64;
    let focal = 400.0f64;
    let (width, height) = (700u32, 200u32);
    let offset = 0.5 * z;
    let principal = [width as f64 * 0.5 - focal * offset / z, height as f64 * 0.5];
    let spec = pinhole(focal, principal, width, height);
    let camera = camera_of(&spec);
    let mean = Vec3::new(offset, 0.0, z);
    let atom = Atom::new(mean, Quat::identity(), Vec3::splat(0.05), 1.0, Rgb::splat(1.0));
    let scene = scene_of(&[atom]);
    let splat = render_ok(&request(
        scene.clone(),
        spec.clone(),
        settings,
        RenderMethod::splat_contract(),
    ));
    let ray = render_ok(&request(
        scene,
        spec,
        settings,
        RenderMethod::raycast_contract(),
    ));
    let centre = match camera.project(mean) {
        Some(p) => p,
        None => panic!("the test atom must project"),
    };
    let row = centre.y.round() as u32;
    let column = centre.x.round() as i32;
    let signed = |d: i32| -> f64 {
        let x = (column + d) as u32;
        splat.image.get(x, row).x - ray.image.get(x, row).x
    };
    let inner = signed(-8);
    let outer = signed(8);
    println!("TV-X3 inner {inner} outer {outer}");
    assert!(
        inner > 0.0 && outer < 0.0,
        "the EWA-versus-ray ratio is BELOW one on the side toward the optical \
         axis and ABOVE it on the far side, so the sign of the discrepancy \
         FLIPS. An abs() assertion at a single offset would hide a one-sided \
         sign error: inner {inner} outer {outer}"
    );
    assert!(
        (inner.abs() / outer.abs() - 1.0).abs() < 0.05,
        "the two lobes are near mirror images at this field angle: {inner} {outer}"
    );
    assert!(signed(0).abs() < 1e-15, "the centre pixel is exact");
}

fn tilted_needles() -> Scene<f64> {
    let palette = [
        Rgb::new(1.0, 0.0, 0.0),
        Rgb::new(0.0, 0.0, 1.0),
        Rgb::new(0.0, 1.0, 0.0),
        Rgb::new(1.0, 1.0, 0.0),
    ];
    let mut atoms = Vec::new();
    for i in 0..8u32 {
        let t = i as f64;
        let half: f64 = 0.35 + 0.29 * t;
        atoms.push(Atom::new(
            Vec3::new(0.05 * (1.7 * t).sin(), 0.05 * (2.3 * t).cos(), 4.0),
            Quat::new_wxyz(half.cos(), 0.0, half.sin(), 0.0),
            Vec3::new(0.30, 0.010, 0.010),
            0.7,
            palette[(i % 4) as usize],
        ));
    }
    scene_of(&atoms)
}

fn narrow_random_scene() -> Scene<f64> {
    let mut atoms = Vec::new();
    let mut rng = Lcg::new(0x5eed_0013);
    for _ in 0..40 {
        let half = rng.unit() * std::f64::consts::PI;
        let axis = match Vec3::new(rng.signed(), rng.signed(), rng.signed()).normalize() {
            Some(v) => v,
            None => Vec3::new(0.0, 0.0, 1.0),
        };
        let sin = half.sin();
        atoms.push(Atom::new(
            Vec3::new(rng.signed() * 0.35, rng.signed() * 0.20, 5.0 + rng.unit() * 3.0),
            Quat::new_wxyz(half.cos(), axis.x * sin, axis.y * sin, axis.z * sin),
            Vec3::new(
                0.02 + rng.unit() * 0.05,
                0.02 + rng.unit() * 0.05,
                0.02 + rng.unit() * 0.05,
            ),
            0.25 + rng.unit() * 0.5,
            Rgb::new(rng.unit(), rng.unit(), rng.unit()),
        ));
    }
    scene_of(&atoms)
}

#[test]
fn tv_x13c_loose_mode_is_a_raycast_only_switch() {
    let mut loose = SharedSettings::<f64>::contract();
    loose.ordering = OrderingModel::HitDepth;
    let spec = pinhole(500.0, [80.0, 60.0], 160, 120);

    match render(&request(
        tilted_needles(),
        spec.clone(),
        loose,
        RenderMethod::splat_contract(),
    )) {
        Err(LightsError::OrderingUnsupported { ordering, method }) => {
            assert_eq!(ordering, OrderingModel::HitDepth);
            assert_eq!(method, lights::method::MethodId::SplatRaster);
        }
        other => panic!(
            "per-pixel sorting is StopThePop and is not built: SplatRaster must \
             refuse HitDepth, got {other:?}"
        ),
    }

    let strict = SharedSettings::<f64>::contract();
    let splat = render_ok(&request(
        tilted_needles(),
        spec.clone(),
        strict,
        RenderMethod::splat_contract(),
    ));
    let ray_strict = render_ok(&request(
        tilted_needles(),
        spec.clone(),
        strict,
        RenderMethod::raycast_contract(),
    ));
    let ray_loose = render_ok(&request(
        tilted_needles(),
        spec.clone(),
        loose,
        RenderMethod::raycast_contract(),
    ));

    let geometry_only = worst_channel(&splat.image, &ray_strict.image);
    let with_ordering = worst_channel(&splat.image, &ray_loose.image);
    println!(
        "TV-X13c geometry only {geometry_only}, geometry plus ordering \
         {with_ordering}"
    );
    assert!(
        with_ordering > 5.0 * geometry_only,
        "t* ordering and centre-depth ordering are genuinely different orders. \
         Making that a switch is what lets the EWA error and the ordering error \
         be measured ONE AT A TIME instead of superimposed: {geometry_only} \
         against {with_ordering}"
    );

    let (report, _, _, tolerance) = pair(tilted_needles(), spec, strict);
    assert!(
        report.is_clean(),
        "strict mode must be clean: {} failures, worst ratio {}",
        report.failures,
        report.worst_ratio
    );
    let loose_report = match compare(&splat.image, &ray_loose.image, &tolerance) {
        Ok(r) => r,
        Err(e) => panic!("{e:?}"),
    };
    assert!(
        loose_report.failures > 0,
        "the strict-mode tolerance covers the EWA geometry ONLY. It must NOT \
         absorb the ordering difference, or the loose mode would be \
         indistinguishable from a bug"
    );
    println!(
        "TV-X13c loose failures {} of {}",
        loose_report.failures, loose_report.pixels
    );
}

#[test]
fn tv_x13_narrow_camera_keeps_the_tolerance_tight() {
    let settings = SharedSettings::<f64>::contract();
    let spec = pinhole(700.0, [160.0, 90.0], 320, 180);
    let (report, splat, _, _) = pair(narrow_random_scene(), spec, settings);
    let (peak, _, _) = splat.image.max_abs();
    println!("TV-X13-narrow {report:?} peak {peak}");
    assert!(peak > 0.5);
    assert_eq!(report.clamped_atoms, 0);
    assert!(
        report.is_clean(),
        "{} of {} pixels exceed the tolerance; worst {:?} measured {} tolerance {}",
        report.failures,
        report.pixels,
        report.worst_pixel,
        report.worst_difference,
        report.worst_tolerance
    );
    assert!(
        report.worst_tolerance < 0.05,
        "at a sane field of view the justified tolerance must stay SMALL enough \
         to catch a real bug. Measured worst tolerance {} - compare the same \
         scene through the demo 121 degree camera, where it reaches 0.31 \
         because the EWA local-affine approximation genuinely breaks down there",
        report.worst_tolerance
    );
    assert!(
        report.worst_difference > 1.0e-6,
        "a perspective camera MUST produce a non-zero divergence; {} means one \
         of the two renderers is not running",
        report.worst_difference
    );
    pin(
        "TV-X13-narrow worst ratio",
        report.worst_ratio,
        0.410_371_931_795_914_47,
    );
    pin(
        "TV-X13-narrow rms difference",
        report.rms_difference,
        0.000_290_198_450_801_597_47,
    );
}

#[test]
fn precision_axis_f32_versus_f64() {
    let settings = SharedSettings::<f64>::contract();
    let spec = pinhole(600.0, [80.0, 60.0], 160, 120);
    let scene = tilted_needles();
    for method in [
        RenderMethod::splat_contract(),
        RenderMethod::raycast_contract(),
    ] {
        let mut wide = request(scene.clone(), spec.clone(), settings, method.clone());
        wide.precision = Precision::F64;
        let mut narrow = wide.clone();
        narrow.precision = Precision::F32;
        let a = render_ok(&wide);
        let b = render_ok(&narrow);
        let worst = worst_channel(&a.image, &b.image);
        let budget = 4.0 * 2.0f64.powi(-24) * 1.0;
        println!(
            "precision axis {:?}: worst f32-versus-f64 difference {worst}, \
             rounding budget {budget}",
            method.id()
        );
        assert!(
            worst < 40.0 * budget,
            "{:?}: {worst} is far outside the f32 rounding budget {budget}",
            method.id()
        );
        assert!(
            worst > 0.0,
            "{:?}: an EXACTLY zero difference means the precision axis is \
             nominal rather than real",
            method.id()
        );
    }
}

#[test]
fn the_agreement_harness_reports_a_deliberately_misprojected_render() {
    let settings = SharedSettings::<f64>::contract();
    let width = 320u32;
    let height = 180u32;
    let focal = 700.0f64;
    let principal = [160.0f64, 90.0];
    let spec = pinhole(focal, principal, width, height);
    let scene = narrow_random_scene();

    let (honest, splat, _, tolerance) = pair(scene.clone(), spec.clone(), settings);
    assert!(
        honest.is_clean(),
        "the honest pair must be clean before the instrument is broken: {honest:?}"
    );

    let broken: [(&'static str, CameraSpec); 4] = [
        (
            "half-pixel principal shift",
            pinhole(focal, [principal[0] + 0.5, principal[1] + 0.5], width, height),
        ),
        (
            "one part in a thousand focal error",
            pinhole(focal * 1.001, principal, width, height),
        ),
        (
            "non-square focal",
            CameraSpec::look_at(
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0],
                45.0,
                width,
                height,
            )
            .with_intrinsics([focal, focal * 1.01], principal),
        ),
        (
            "one-pixel principal shift",
            pinhole(focal, [principal[0] + 1.0, principal[1]], width, height),
        ),
    ];

    let mut caught = 0usize;
    for (name, wrong) in broken {
        let misprojected = render_ok(&request(
            scene.clone(),
            wrong,
            settings,
            RenderMethod::raycast_contract(),
        ));
        let report = match compare(&splat.image, &misprojected.image, &tolerance) {
            Ok(r) => r,
            Err(e) => panic!("{e:?}"),
        };
        println!(
            "BROKEN {name}: {} of {} pixels fail, worst ratio {}",
            report.failures, report.pixels, report.worst_ratio
        );
        assert!(
            !report.is_clean(),
            "A HARNESS THAT PASSES A WRONG IMAGE IS WORTHLESS. {name} produced a \
             clean report: {report:?}"
        );
        assert!(
            report.worst_ratio > 1.0,
            "{name}: worst ratio {} must exceed one",
            report.worst_ratio
        );
        caught += 1;
    }
    assert_eq!(caught, 4);

    let doubled = render_ok(&request(
        scene,
        spec,
        settings,
        RenderMethod::splat_contract(),
    ));
    let self_report = match compare(&splat.image, &doubled.image, &tolerance) {
        Ok(r) => r,
        Err(e) => panic!("{e:?}"),
    };
    assert!(self_report.is_clean());
    assert_eq!(
        self_report.worst_ratio, 0.0,
        "comparing a renderer against ITSELF must give exactly zero, which is \
         the signal that catches a harness returning the same image twice - the \
         reason every must-differ assertion in this file is TWO-SIDED"
    );
    assert_eq!(self_report.worst_difference, 0.0);
}

#[test]
fn the_truncation_channel_is_weighted_by_the_transmittance_in_front_of_each_atom() {
    let settings = SharedSettings::<f64>::contract();
    let focal = 200.0f64;
    let (width, height) = (400u32, 200u32);
    let spec = pinhole(focal, [100.0, 100.0], width, height);
    let camera = camera_of(&spec);
    let mean = Vec3::new(5.0, 0.0, 5.0);
    assert!(
        camera.jacobian_limits().x > 1.0,
        "this atom sits at tan theta = 1 and must stay inside the off-axis clamp"
    );
    let mut atoms = Vec::new();
    for _ in 0..3u32 {
        atoms.push(Atom::isotropic(mean, 0.5, 1.0, Rgb::splat(1.0)));
    }
    let bound = atom_bound(mean, &isotropic(0.5), 1.0);
    assert!(
        bound > 0.05,
        "the straddle annulus is alpha in [eps, eps + B_atom], so this test \
         needs a FAT off-axis atom for the annulus to be wider than a pixel. \
         B_atom {bound}"
    );

    let mut r = request(
        scene_of(&atoms),
        spec,
        settings,
        RenderMethod::splat_contract(),
    );
    r.collect = Collect::PixelsAndTolerance;

    for method in [
        RenderMethod::splat_contract(),
        RenderMethod::raycast_contract(),
    ] {
        let mut one = r.clone();
        one.method = method.clone();
        let outcome = render_ok(&one);
        let truncation = match outcome.truncation.as_ref() {
            Some(t) => t,
            None => panic!(
                "{:?}: PixelsAndTolerance must produce a truncation image",
                method.id()
            ),
        };
        let centre = match camera.project(mean) {
            Some(c) => c,
            None => panic!("the test atom must project"),
        };
        let cx = centre.x.round() as u32;
        let cy = centre.y.round() as u32;
        assert_eq!(
            truncation.get(cx, cy).x,
            0.0,
            "{:?}: at the atom centre alpha is 1.0, which no cutoff of 1/255 can \
             straddle however far the two radii differ, so charging the \
             eps_alpha step there is pure slack",
            method.id()
        );
        assert_eq!(
            truncation.get(0, 0).x,
            0.0,
            "{:?}: a pixel no atom reaches carries no truncation term",
            method.id()
        );

        let mut charged = 0usize;
        let mut worst = 0.0f64;
        for y in 0..height {
            for x in 0..width {
                let t = truncation.get(x, y).x;
                if t <= 0.0 {
                    continue;
                }
                charged += 1;
                let composited = outcome.image.get(x, y).x;
                let alpha = 1.0 - (1.0 - composited).cbrt();
                let expected = 1.0 + (1.0 - alpha) + (1.0 - alpha) * (1.0 - alpha);
                worst = worst.max((t - expected).abs());
                assert!(
                    (t - expected).abs() < 1e-9,
                    "{:?} at ({x},{y}): the truncation term is \
                     2*c_max*eps_alpha*sum_i T_i, so the channel must accumulate \
                     the TRANSMITTANCE IN FRONT of each straddling atom, not a \
                     bare count. Three identical atoms composite to \
                     1-(1-a)^3, so the transmittances are 1, 1-a and (1-a)^2. \
                     Measured {t}, expected {expected}; an unweighted count \
                     would be exactly 3",
                    method.id()
                );
                assert!(
                    alpha <= settings.blend.alpha_cutoff + bound + 1e-12,
                    "{:?} at ({x},{y}): alpha {alpha} is above eps_alpha + \
                     B_atom, so the other method - whose alpha differs by at \
                     most B_atom - cannot have dropped this atom. Charging the \
                     step here is unjustified",
                    method.id()
                );
            }
        }
        println!(
            "{:?}: {charged} pixels charged, worst deviation from the \
             transmittance sum {worst}",
            method.id()
        );
        assert!(
            charged > 20,
            "{:?}: only {charged} pixels were charged; the annulus where the two \
             cuts can straddle must be a real region of this image or the test \
             is measuring nothing",
            method.id()
        );
    }
}

fn compose_both(
    outcome: &RenderOutcome,
    settings: &SharedSettings<f64>,
) -> (FloatImage<f64>, FloatImage<f64>) {
    let accumulated = match outcome.tolerance.as_ref() {
        Some(t) => t,
        None => panic!("the render must produce a tolerance image"),
    };
    let truncation = match outcome.truncation.as_ref() {
        Some(t) => t,
        None => panic!("the render must produce a truncation image"),
    };
    let bound_only = compose_tolerance(
        accumulated,
        HEADROOM,
        1.0,
        settings.blend.transmittance_cutoff,
    );
    let full = match compose_tolerance_with_truncation(
        accumulated,
        truncation,
        HEADROOM,
        1.0,
        settings.blend.alpha_cutoff,
        settings.blend.transmittance_cutoff,
    ) {
        Ok(v) => v,
        Err(e) => panic!("{e:?}"),
    };
    (bound_only, full)
}

fn frame_sum(image: &FloatImage<f64>) -> f64 {
    image.pixels.iter().map(|p| p.x).sum()
}

#[test]
fn the_truncation_channel_is_charged_only_where_the_two_cuts_can_straddle() {
    let settings = SharedSettings::<f64>::contract();
    let spec = pinhole(700.0, [160.0, 90.0], 320, 180);
    let scene = narrow_random_scene();

    let mut base = request(
        scene,
        spec,
        settings,
        RenderMethod::splat_contract(),
    );
    base.collect = Collect::PixelsAndTolerance;
    let mut splat = base.clone();
    splat.method = RenderMethod::splat_contract();
    let mut ray = base.clone();
    ray.method = RenderMethod::raycast_contract();
    let a = render_ok(&splat);
    let b = render_ok(&ray);

    let (bound_a, full_a) = compose_both(&a, &settings);
    let (bound_b, full_b) = compose_both(&b, &settings);
    let bound = match envelope(&bound_a, &bound_b) {
        Ok(v) => v,
        Err(e) => panic!("{e:?}"),
    };
    let full = match envelope(&full_a, &full_b) {
        Ok(v) => v,
        Err(e) => panic!("{e:?}"),
    };
    let bound_report = match compare(&a.image, &b.image, &bound) {
        Ok(r) => r,
        Err(e) => panic!("{e:?}"),
    };
    let full_report = match compare(&a.image, &b.image, &full) {
        Ok(r) => r,
        Err(e) => panic!("{e:?}"),
    };
    let ratio = frame_sum(&full) / frame_sum(&bound);
    println!(
        "truncation gate: bound-only failures {} worst ratio {}; composed \
         failures {} worst ratio {}; frame-summed tolerance ratio {ratio}",
        bound_report.failures,
        bound_report.worst_ratio,
        full_report.failures,
        full_report.worst_ratio
    );

    assert!(
        bound_report.failures > 0,
        "the truncation channel must be LOAD-BEARING: with the S22 bound term \
         alone this scene reports failing pixels, because the eps_alpha step \
         where the two supports straddle is not covered by any multiple of \
         B_atom. Measured {} failures at worst ratio {}",
        bound_report.failures,
        bound_report.worst_ratio
    );
    assert!(
        full_report.failures == 0,
        "and once the step is priced the report must be clean: {} failures, \
         worst ratio {}",
        full_report.failures,
        full_report.worst_ratio
    );
    assert!(
        ratio < 1.5,
        "the step must be charged only for atoms that can actually straddle the \
         cut. Charging every evaluated atom instead measured 3.12x the bound \
         term summed over this frame - which is the fudge factor the whole \
         per-pixel construction exists to avoid. Measured {ratio}"
    );
    assert!(
        ratio > 1.0,
        "a ratio of exactly 1 means the truncation channel is dead: {ratio}"
    );
}

#[test]
fn a_rotated_and_translated_camera_is_certified_by_the_same_per_pixel_tolerance() {
    let settings = SharedSettings::<f64>::contract();
    let spec = CameraSpec::look_at(
        [1.5, -0.7, 2.0],
        [0.2, 0.4, 5.5],
        [0.0, -1.0, 0.0],
        50.0,
        160,
        120,
    );
    let camera = camera_of(&spec);
    let identity = Mat3::<f64>::identity();
    let mut off_diagonal = 0.0f64;
    for i in 0..3 {
        for j in 0..3 {
            off_diagonal = off_diagonal.max((camera.rotation.m[i][j] - identity.m[i][j]).abs());
        }
    }
    assert!(
        off_diagonal > 0.1,
        "every other camera in this file is look_at([0,0,0],[0,0,1],[0,-1,0]), \
         which is rotation = identity and translation = 0. A transposed \
         congruence in Camera::covariance_to_camera - R^T Sigma R instead of \
         R Sigma R^T - is INVISIBLE to all of them. Measured departure from \
         the identity {off_diagonal}"
    );
    assert!(
        camera.translation.length() > 0.1,
        "the translation must also be non-zero: {:?}",
        camera.translation
    );

    let (report, splat, ray, _) = match try_pair(rotated_view_scene(), spec, settings) {
        Ok(v) => v,
        Err(e) => panic!(
            "under a correct implementation this camera and scene are \
             contractually clean, so the pair MUST render and be comparable. \
             An error HERE means the two view transforms disagree, not that \
             the scene is out of contract: SplatRaster carries atoms \
             world->camera through Camera::to_camera and \
             Camera::covariance_to_camera, while RayCast carries rays \
             camera->world through to_world_ray. Transposing either one \
             desynchronises them and the symptom arrives disguised as {e:?}"
        ),
    };
    println!("rotated view {report:?}");
    let (peak, _, _) = splat.image.max_abs();
    assert!(peak > 0.5, "the scene must actually render: {peak}");
    assert_eq!(splat.method, lights::method::MethodId::SplatRaster);
    assert_eq!(ray.method, lights::method::MethodId::RayCast);
    assert_eq!(report.clamped_atoms, 0);
    assert_eq!(report.determinant_culled_atoms, 0);
    assert!(
        report.is_clean(),
        "{} of {} pixels exceed the per-pixel tolerance under a general view \
         transform. Worst pixel {:?}: measured {} against tolerance {} (ratio \
         {})",
        report.failures,
        report.pixels,
        report.worst_pixel,
        report.worst_difference,
        report.worst_tolerance,
        report.worst_ratio
    );
    assert!(
        report.worst_tolerance < 0.05,
        "at 50 degrees the justified tolerance must stay small enough to catch \
         a real bug: {}",
        report.worst_tolerance
    );
    assert!(
        report.worst_difference > 1.0e-6,
        "a perspective camera MUST produce a non-zero divergence; {} means one \
         of the two renderers is not running",
        report.worst_difference
    );
    assert!(
        report.rms_difference > 1.0e-7,
        "and the bulk of the frame must differ too: {}",
        report.rms_difference
    );
}

fn rotated_view_scene() -> Scene<f64> {
    let mut atoms = Vec::new();
    let mut rng = Lcg::new(0x5eed_0021);
    for _ in 0..24 {
        let half = rng.unit() * std::f64::consts::PI;
        let axis = match Vec3::new(rng.signed(), rng.signed(), rng.signed()).normalize() {
            Some(v) => v,
            None => Vec3::new(0.0, 0.0, 1.0),
        };
        let sin = half.sin();
        atoms.push(Atom::new(
            Vec3::new(
                0.2 + rng.signed() * 0.6,
                0.4 + rng.signed() * 0.5,
                5.5 + rng.signed() * 1.2,
            ),
            Quat::new_wxyz(half.cos(), axis.x * sin, axis.y * sin, axis.z * sin),
            Vec3::new(
                0.03 + rng.unit() * 0.10,
                0.03 + rng.unit() * 0.10,
                0.03 + rng.unit() * 0.10,
            ),
            0.30 + rng.unit() * 0.5,
            Rgb::new(rng.unit(), rng.unit(), rng.unit()),
        ));
    }
    scene_of(&atoms)
}

#[test]
fn the_bound_coefficients_equal_their_closed_forms() {
    let first = 3.0f64.powf(1.5) * (-1.5f64).exp();
    let second = 16.0f64 * (-2.0f64).exp() / 2.0;
    assert_eq!(
        FIRST_ORDER_COEFFICIENT, first,
        "the first-order coefficient is max_r r^3 e^(-r^2/2) = 3^1.5 e^-1.5. \
         Rust cannot fold powf or exp in a const, so the digits are a literal - \
         but they must be pinned against the FORMULA, never against a second \
         copy of the same digits, or a mistyped digit silently widens every \
         tolerance in the crate. Sheet-printed {FIRST_ORDER_COEFFICIENT}, \
         closed form {first}"
    );
    assert_eq!(
        SECOND_ORDER_COEFFICIENT, second,
        "the second-order coefficient is max_r r^4 e^(-r^2/2) / 2 = 16 e^-2 / 2. \
         Sheet-printed {SECOND_ORDER_COEFFICIENT}, closed form {second}"
    );
    let terms = atom_bound_terms(Vec3::new(0.0, 0.0, 5.0), &isotropic(0.05), 1.0);
    assert!(
        (terms.second_order - SECOND_ORDER_COEFFICIENT * 1.0e-4).abs() < 1e-18,
        "and the constants must be the ones the bound actually uses: {:?}",
        terms
    );
}

#[test]
fn the_normalised_blend_is_refused_by_the_certified_path_and_compared_without_it() {
    assert!(
        !KernelFamily::GaussianNormalised.blend_is_over(),
        "the P2 flat-fill win runs on this kernel and the certified path is gated \
         on blend_is_over"
    );
    assert!(
        KernelFamily::GaussianNormalised.curve_is_gaussian(),
        "the CURVE half of the contract passes; it is the BLEND that has no \
         derived bound, and saying which half fails is the difference between a \
         limit and a mystery"
    );

    let settings = SharedSettings::<f64>::contract();
    let spec = sphere_camera(320, 180);
    let mut base = request(
        spheres_anisotropic_no_ground::<f64>(),
        spec,
        settings,
        RenderMethod::splat_contract(),
    );
    base.kernel = KernelFamily::GaussianNormalised;
    match run_pair(
        &base,
        &RenderMethod::splat_contract(),
        &RenderMethod::raycast_contract(),
    ) {
        Err(LightsError::ContractViolation(ContractViolation::BlendClassNotOver { kernel })) => {
            assert_eq!(kernel, KernelFamily::GaussianNormalised.name())
        }
        Ok(_) => panic!(
            "the certified path ACCEPTED the normalised blend. Its tolerance is \
             derived for transmittance-weighted compositing and Normalised has no \
             transmittance, so composing one here would certify a bound nobody \
             derived"
        ),
        Err(e) => panic!(
            "the refusal must name the BLEND half of the contract, not some other \
             failure: {e:?}"
        ),
    }

    let mut splat_request = base.clone();
    splat_request.method = RenderMethod::splat_contract();
    let mut ray_request = base.clone();
    ray_request.method = RenderMethod::raycast_contract();
    let splat = render_ok(&splat_request);
    let ray = render_ok(&ray_request);
    let (peak, _, _) = splat.image.max_abs();
    assert!(peak > 0.2, "the scene must actually render: {peak}");
    let normalised_gap = worst_channel(&splat.image, &ray.image);

    let mut over_splat = base.clone();
    over_splat.kernel = KernelFamily::Gaussian;
    over_splat.method = RenderMethod::splat_contract();
    let mut over_ray = over_splat.clone();
    over_ray.method = RenderMethod::raycast_contract();
    let over_gap = worst_channel(&render_ok(&over_splat).image, &render_ok(&over_ray).image);

    println!(
        "UNCERTIFIED SECOND OPINION: worst channel splat-vs-raycast is \
         {normalised_gap} under Normalised and {over_gap} under Over, on \
         spheres-anisotropic-no-ground at 320x180 f64"
    );
    assert!(
        normalised_gap > 0.0,
        "two independently implemented methods agreeing to the last bit would \
         mean one of them is not being exercised"
    );
    assert!(
        normalised_gap < NORMALISED_UNCERTIFIED_GAP,
        "the normalised blend has no CERTIFIED bound, but it is not unmeasured: \
         the two methods differ by {normalised_gap} here, against \
         {NORMALISED_UNCERTIFIED_GAP} recorded on 2026-09-15. A rise means the \
         flat-fill result lost the only second opinion it has"
    );
    assert!(
        normalised_gap < over_gap * 1.5,
        "the number that matters is the RATIO, because it is the one that does \
         not move when the scene does: the normalised gap {normalised_gap} must \
         stay the same order as the Over gap {over_gap}, which IS certified. \
         Measured 2026-09-15: 0.1411988743075459 against 0.12389689254766625, a \
         ratio of 1.1396. If this rises, 'no certified bound' has become 'the two \
         methods do not agree', which is a different and much worse sentence"
    );
}

pub const NORMALISED_UNCERTIFIED_GAP: f64 = 0.145;

#[test]
fn every_field_that_changes_the_image_is_caught_by_the_gate_or_by_the_fingerprint() {
    let base = SharedSettings::<f64>::contract();
    let scene = scene_of(&[Atom::isotropic(
        Vec3::new(0.0, 0.0, 5.0),
        0.05,
        0.8,
        Rgb::splat(1.0),
    )]);
    assert!(
        assert_contract(&base, &base, 0, 0, &scene).is_ok(),
        "the contract profile must pass against itself, or every row below is \
         a pass for the wrong reason"
    );

    let mut moved: Vec<SharedSettings<f64>> = Vec::new();
    let mut s = base;
    s.blend.alpha_cap = base.blend.alpha_cap * 0.5;
    moved.push(s);
    let mut s = base;
    s.blend.alpha_cutoff = base.blend.alpha_cutoff * 2.0;
    moved.push(s);
    let mut s = base;
    s.blend.transmittance_cutoff = base.blend.transmittance_cutoff * 10.0;
    moved.push(s);
    let mut s = base;
    s.blend.early_out = match base.blend.early_out {
        EarlyOut::Reference => EarlyOut::Exact,
        EarlyOut::Exact => EarlyOut::Reference,
    };
    moved.push(s);
    let mut s = base;
    s.ordering = lights::settings::OrderingModel::HitDepth;
    moved.push(s);
    let mut s = base;
    s.support = lights::settings::SupportMode::FixedSigmas(3.0);
    moved.push(s);
    let mut s = base;
    s.screen_dilation = 0.3;
    moved.push(s);
    let mut s = base;
    s.energy_compensation = !base.energy_compensation;
    moved.push(s);
    let mut s = base;
    s.near_plane = base.near_plane * 2.0;
    moved.push(s);
    let mut s = base;
    s.background = Rgb::splat(0.5);
    moved.push(s);
    let mut s = base;
    s.scale_floor = base.scale_floor * 10.0;
    moved.push(s);
    let mut s = base;
    s.light = lights::settings::DirectionalLight::new(Vec3::new(1.0, 0.0, 0.0), 0.2, 0.8);
    moved.push(s);

    let mut by_fingerprint_only = 0usize;
    for (index, other) in moved.iter().enumerate() {
        assert_ne!(
            base.fingerprint(),
            other.fingerprint(),
            "row {index} changed a field of SharedSettings and the fingerprint \
             did not move. The fingerprint is a hand-written list over a \
             hand-written struct, so a field added to one and not the other \
             drifts silently — that is what this row exists to catch"
        );
        match assert_contract(&base, other, 0, 0, &scene) {
            Ok(()) => panic!(
                "row {index}: two settings that produce different images passed \
                 the gate. Eight of assert_contract's arms compare named fields \
                 and no arm covered this one"
            ),
            Err(LightsError::ContractViolation(ContractViolation::UnnamedSettingsMismatch)) => {
                by_fingerprint_only += 1;
            }
            Err(LightsError::ContractViolation(_)) => (),
            Err(e) => panic!("row {index}: refused for the wrong reason: {e:?}"),
        }
    }
    assert_eq!(moved.len(), 12, "the sweep must cover every fingerprinted field");
    assert_eq!(
        by_fingerprint_only, 2,
        "EXACTLY TWO fields reach the gate only through the fingerprint, and \
         naming the number is what makes a third one visible: `background`, \
         which nothing compared at all before 2026-09-15, and `scale_floor`, \
         which assert_contract reads as max(a, b) for the scene check and never \
         compares between the two. Both change the image"
    );

    let mut threaded = base;
    threaded.threads = base.threads + 7;
    assert_eq!(
        base.fingerprint(),
        threaded.fingerprint(),
        "`threads` is the one field deliberately outside the fingerprint: a \
         render is a pure function of the scene and the thread count must not \
         appear in the image. tests/fit_parallel.rs asserts that property \
         directly; if it ever entered the fingerprint, two runs at different \
         thread counts would stop being comparable"
    );
    assert!(assert_contract(&base, &threaded, 0, 0, &scene).is_ok());
}

fn variant_name(v: &ContractViolation) -> &'static str {
    match v {
        ContractViolation::DilationEnabled => "DilationEnabled",
        ContractViolation::EnergyCompensationMismatch => "EnergyCompensationMismatch",
        ContractViolation::AlphaCutoffMismatch => "AlphaCutoffMismatch",
        ContractViolation::TransmittanceCutoffMismatch => "TransmittanceCutoffMismatch",
        ContractViolation::AlphaCapMismatch => "AlphaCapMismatch",
        ContractViolation::EarlyOutMismatch => "EarlyOutMismatch",
        ContractViolation::OrderingMismatch => "OrderingMismatch",
        ContractViolation::SupportModeMismatch => "SupportModeMismatch",
        ContractViolation::SupportNotCutoffSolved => "SupportNotCutoffSolved",
        ContractViolation::NearPlaneMismatch => "NearPlaneMismatch",
        ContractViolation::ClampedAtomsPresent { .. } => "ClampedAtomsPresent",
        ContractViolation::DeterminantCulledAtomsPresent { .. } => "DeterminantCulledAtomsPresent",
        ContractViolation::ScaleBelowFloor { .. } => "ScaleBelowFloor",
        ContractViolation::BlendClassNotOver { .. } => "BlendClassNotOver",
        ContractViolation::KernelCurveNotGaussian { .. } => "KernelCurveNotGaussian",
        ContractViolation::LightPresent => "LightPresent",
        ContractViolation::UnnamedSettingsMismatch => "UnnamedSettingsMismatch",
    }
}

fn every_variant() -> Vec<ContractViolation> {
    vec![
        ContractViolation::DilationEnabled,
        ContractViolation::EnergyCompensationMismatch,
        ContractViolation::AlphaCutoffMismatch,
        ContractViolation::TransmittanceCutoffMismatch,
        ContractViolation::AlphaCapMismatch,
        ContractViolation::EarlyOutMismatch,
        ContractViolation::OrderingMismatch,
        ContractViolation::SupportModeMismatch,
        ContractViolation::SupportNotCutoffSolved,
        ContractViolation::NearPlaneMismatch,
        ContractViolation::ClampedAtomsPresent { count: 0 },
        ContractViolation::DeterminantCulledAtomsPresent { count: 0 },
        ContractViolation::ScaleBelowFloor { index: 0 },
        ContractViolation::BlendClassNotOver { kernel: "" },
        ContractViolation::KernelCurveNotGaussian { kernel: "" },
        ContractViolation::LightPresent,
        ContractViolation::UnnamedSettingsMismatch,
    ]
}

fn doc_section_variants(doc: &str, heading: &str, names: &[&'static str]) -> Vec<&'static str> {
    let start = match doc.find(heading) {
        Some(i) => i + heading.len(),
        None => panic!("preconditions.md has no section `{heading}`"),
    };
    let rest = &doc[start..];
    let body = match rest.find("\n## ") {
        Some(end) => &rest[..end],
        None => rest,
    };
    let mut found: Vec<&'static str> = Vec::new();
    for (i, piece) in body.split('`').enumerate() {
        if i % 2 == 1 {
            for name in names {
                if piece == *name && !found.contains(name) {
                    found.push(name);
                }
            }
        }
    }
    found.sort_unstable();
    found
}

fn run_pair_probe(
    kernel: KernelFamily,
    scene: Scene<f64>,
    settings: SharedSettings<f64>,
) -> lights::error::Result<(AgreementReport, RenderOutcome, RenderOutcome, FloatImage<f64>)> {
    let mut base = request(
        scene,
        sphere_camera(64, 36),
        settings,
        RenderMethod::splat_contract(),
    );
    base.kernel = kernel;
    run_pair(
        &base,
        &RenderMethod::splat_contract(),
        &RenderMethod::raycast_contract(),
    )
}

#[test]
fn the_arms_run_pair_can_raise_are_the_arms_the_preconditions_doc_names() {
    let all = every_variant();
    let names: Vec<&'static str> = all.iter().map(variant_name).collect();
    let mut distinct = names.clone();
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(
        distinct.len(),
        17,
        "every_variant must list each ContractViolation once; seventeen since L-117 removed ResponseMismatch"
    );

    let one_atom = |sigma: f64| {
        scene_of(&[Atom::isotropic(
            Vec3::new(0.0, 0.0, 5.0),
            sigma,
            0.8,
            Rgb::splat(1.0),
        )])
    };
    let contract = SharedSettings::<f64>::contract();
    let gaussian = KernelFamily::Gaussian;

    let mut dilated = contract;
    dilated.screen_dilation = 0.3;
    let mut compensated = contract;
    compensated.energy_compensation = true;
    let mut fixed = contract;
    fixed.support = SupportMode::FixedSigmas(3.0);
    let mut floored = contract;
    floored.scale_floor = 0.1;
    let mut lit = contract;
    lit.light = DirectionalLight::new(Vec3::new(1.0, 0.0, 0.0), 0.2, 0.8);
    assert!(lit.light.is_some(), "the probe light must exist");

    let live: Vec<(&'static str, KernelFamily, Scene<f64>, SharedSettings<f64>)> = vec![
        ("BlendClassNotOver", KernelFamily::GaussianNormalised, one_atom(0.05), contract),
        ("KernelCurveNotGaussian", KernelFamily::CompactBump, one_atom(0.05), contract),
        ("DilationEnabled", gaussian, one_atom(0.05), dilated),
        ("EnergyCompensationMismatch", gaussian, one_atom(0.05), compensated),
        ("SupportNotCutoffSolved", gaussian, one_atom(0.05), fixed),
        ("ScaleBelowFloor", gaussian, one_atom(0.05), floored),
        ("LightPresent", gaussian, one_atom(0.05), lit),
        ("ClampedAtomsPresent", gaussian, spheres_anisotropic::<f64>(), contract),
        ("DeterminantCulledAtomsPresent", gaussian, one_atom(1.0e-5), contract),
    ];
    let mut raised: Vec<&'static str> = Vec::new();
    for (want, kernel, scene, settings) in live {
        match run_pair_probe(kernel, scene, settings) {
            Err(LightsError::ContractViolation(v)) => {
                assert_eq!(variant_name(&v), want, "the probe for {want} raised another arm");
                raised.push(variant_name(&v));
            }
            Err(e) => panic!("{want}: refused for a reason outside the contract: {e:?}"),
            Ok(_) => panic!("{want}: run_pair COMPARED instead of refusing"),
        }
    }
    raised.sort_unstable();

    let mut one_sided: Vec<SharedSettings<f64>> = Vec::new();
    let mut s = contract;
    s.blend.alpha_cutoff = 0.01;
    one_sided.push(s);
    let mut s = contract;
    s.blend.transmittance_cutoff = 1e-3;
    one_sided.push(s);
    let mut s = contract;
    s.blend.alpha_cap = 0.999;
    one_sided.push(s);
    let mut s = contract;
    s.blend.early_out = EarlyOut::Reference;
    one_sided.push(s);
    let mut s = contract;
    s.ordering = OrderingModel::HitDepth;
    one_sided.push(s);
    let mut s = contract;
    s.near_plane = 0.5;
    one_sided.push(s);
    for settings in one_sided {
        if let Err(LightsError::ContractViolation(v)) =
            run_pair_probe(gaussian, one_atom(0.05), settings)
        {
            assert!(
                raised.contains(&variant_name(&v)),
                "run_pair raised {} from ONE settings value; the doc calls that arm \
                 structurally unreachable",
                variant_name(&v)
            );
        }
    }

}
