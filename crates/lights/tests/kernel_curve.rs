use lights::agreement::ContractViolation;
use lights::camera::CameraSpec;
use lights::dispatch::{Collect, Precision, RenderRequest, render};
use lights::error::LightsError;
use lights::image::FloatImage;
use lights::kernel::compact::{CompactBump, ONE_THIRD, ZERO_RADIUS, ZERO_RADIUS_SQUARED};
use lights::kernel::frame::{RayFrame, ScreenFrame};
use lights::kernel::gabor::{GaborColour, GaborDensity};
use lights::kernel::gaussian::{Gaussian, GaussianNormalised};
use lights::kernel::hermite::{GaussHermiteColour, GaussHermiteDensity};
use lights::kernel::supergaussian::SuperGaussian;
use lights::kernel::{Beta, CurveShape, GAUSSIAN_BETA, Kernel, KernelFamily};
use lights::method::RenderMethod;
use lights::numerics::quaternion::Quat;
use lights::numerics::symmetric::{Sym2, Sym3};
use lights::numerics::vector::{Rgb, Vec2, Vec3};
use lights::scene::prepare::PreparedScene;
use lights::scene::{Atom, Scene};
use lights::settings::{MIN_SCALE, SharedSettings};

const EPS_ALPHA: f64 = 1.0 / 255.0;

fn screen_frame() -> ScreenFrame<f64> {
    match ScreenFrame::from_covariance(Sym2::new(4.0, 0.7, 2.5)) {
        Some(f) => f,
        None => panic!("the test frame must be SPD"),
    }
}

fn ray_frame() -> RayFrame<f64> {
    let sigma = Sym3::<f64>::new(0.04, 0.0, 0.0, 0.01, 0.0, 0.0025);
    let lambda = match sigma.inverse_spd() {
        Some(l) => l,
        None => panic!("the test covariance must be SPD"),
    };
    let chol = match sigma.cholesky() {
        Some(c) => c,
        None => panic!("the test covariance must be SPD"),
    };
    RayFrame::new(lambda, chol.inverse(), Vec3::new(0.0, 0.0, 1.0))
}

fn screen_grid() -> Vec<Vec2<f64>> {
    let mut out = Vec::new();
    for i in 0..64 {
        for j in 0..64 {
            out.push(Vec2::new(
                -7.0 + 14.0 * (i as f64) / 63.0,
                -7.0 + 14.0 * (j as f64) / 63.0,
            ));
        }
    }
    out
}

fn ray_grid() -> Vec<Vec3<f64>> {
    let mut out = Vec::new();
    for i in 0..14 {
        for j in 0..14 {
            for k in 0..14 {
                out.push(Vec3::new(
                    -5.0 + 10.0 * (i as f64) / 13.0,
                    -5.0 + 10.0 * (j as f64) / 13.0,
                    -5.0 + 10.0 * (k as f64) / 13.0,
                ));
            }
        }
    }
    out
}

fn curve_scene() -> Scene<f64> {
    let mut scene = Scene::<f64>::new();
    let atoms = [
        (Vec3::new(-0.35, 0.10, 5.0), Vec3::new(0.30, 0.09, 0.09), 0.85, Rgb::new(0.9, 0.3, 0.2)),
        (Vec3::new(0.20, -0.15, 4.4), Vec3::new(0.12, 0.24, 0.12), 0.65, Rgb::new(0.2, 0.8, 0.4)),
        (Vec3::new(0.05, 0.25, 6.1), Vec3::new(0.40, 0.13, 0.20), 0.45, Rgb::new(0.3, 0.4, 0.95)),
    ];
    let rotation = match Quat::from_axis_angle(Vec3::new(0.3, 0.8, 0.5), 0.7) {
        Some(q) => q,
        None => panic!("the test rotation axis must be normalisable"),
    };
    for (mean, scale, opacity, colour) in atoms {
        let atom = Atom::new(mean, rotation, scale, opacity, colour);
        match scene.push(atom, MIN_SCALE) {
            Ok(()) => (),
            Err(e) => panic!("the curve test scene must be valid: {e}"),
        }
    }
    scene
}

fn curve_request(kernel: KernelFamily, method: RenderMethod) -> RenderRequest {
    RenderRequest {
        precision: Precision::F64,
        kernel,
        method,
        settings: SharedSettings::<f64>::contract(),
        camera: CameraSpec::look_at(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
            45.0,
            48,
            36,
        ),
        scene: curve_scene(),
        collect: Collect::Pixels,
        shapes: None,
    }
}

fn rendered(kernel: KernelFamily, method: RenderMethod) -> FloatImage<f64> {
    match render(&curve_request(kernel, method)) {
        Ok(o) => o.image,
        Err(e) => panic!("{} failed to render: {e}", kernel.name()),
    }
}

fn identical_bits(a: &FloatImage<f64>, b: &FloatImage<f64>) -> Option<(u32, u32)> {
    for y in 0..a.height {
        for x in 0..a.width {
            let (p, q) = (a.get(x, y), b.get(x, y));
            let same = p.x.to_bits() == q.x.to_bits()
                && p.y.to_bits() == q.y.to_bits()
                && p.z.to_bits() == q.z.to_bits();
            if !same {
                return Some((x, y));
            }
        }
    }
    None
}

fn worst_gap(a: &FloatImage<f64>, b: &FloatImage<f64>) -> f64 {
    let mut worst = 0.0f64;
    for y in 0..a.height {
        for x in 0..a.width {
            let (p, q) = (a.get(x, y), b.get(x, y));
            worst = worst
                .max((p.x - q.x).abs())
                .max((p.y - q.y).abs())
                .max((p.z - q.z).abs());
        }
    }
    worst
}

#[test]
fn beta_one_reproduces_the_gaussian_bit_for_bit() {
    let shape = CurveShape::with_beta(GAUSSIAN_BETA);
    let frame = screen_frame();
    let plain_p = <Gaussian as Kernel<f64>>::params_from(&shape);
    let super_p = <SuperGaussian as Kernel<f64>>::params_from(&shape);
    let plain = <Gaussian as Kernel<f64>>::prepare_screen(&plain_p, &frame);
    let sup = <SuperGaussian as Kernel<f64>>::prepare_screen(&super_p, &frame);

    for u in screen_grid() {
        let a = <Gaussian as Kernel<f64>>::eval_screen(&plain, u);
        let b = <SuperGaussian as Kernel<f64>>::eval_screen(&sup, u);
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "eval_screen at {u:?}: beta = 1 IS the gaussian, and it is the \
             control every other beta in the sweep is read against. A one-ulp \
             drift here turns the whole atom-curve comparison into a comparison \
             of two different gaussians"
        );
    }

    let plain_ray = <Gaussian as Kernel<f64>>::prepare_ray(&plain_p, &ray_frame());
    let super_ray = <SuperGaussian as Kernel<f64>>::prepare_ray(&super_p, &ray_frame());
    for u in ray_grid() {
        let a = <Gaussian as Kernel<f64>>::eval_ray(&plain_ray, u);
        let b = <SuperGaussian as Kernel<f64>>::eval_ray(&super_ray, u);
        assert_eq!(a.to_bits(), b.to_bits(), "eval_ray at {u:?}");
    }

    for peak in [1.0f64, 0.7, 0.25, 0.05, 0.004, 1.0 / 255.0, 0.0009] {
        let a = <Gaussian as Kernel<f64>>::support_screen(&plain, peak, EPS_ALPHA);
        let b = <SuperGaussian as Kernel<f64>>::support_screen(&sup, peak, EPS_ALPHA);
        assert_eq!(a.to_bits(), b.to_bits(), "support_screen at peak {peak}");
        let c = <Gaussian as Kernel<f64>>::support_ray(&plain_ray, peak, EPS_ALPHA);
        let d = <SuperGaussian as Kernel<f64>>::support_ray(&super_ray, peak, EPS_ALPHA);
        assert_eq!(c.to_bits(), d.to_bits(), "support_ray at peak {peak}");
    }

    for variance in [0.0f64, 1.0 / 12.0, 0.3, 1.7] {
        let cov = Sym2::<f64>::new(4.0, 0.7, 2.5);
        let (_, plain_cov, plain_gain) =
            <Gaussian as Kernel<f64>>::prefilter(&plain_p, cov, variance);
        let (super_params, super_cov, super_gain) =
            <SuperGaussian as Kernel<f64>>::prefilter(&super_p, cov, variance);
        assert_eq!(plain_cov, super_cov, "prefilter covariance at {variance}");
        assert_eq!(plain_gain.to_bits(), super_gain.to_bits());
        assert_eq!(
            super_params.beta.to_bits(),
            GAUSSIAN_BETA.to_bits(),
            "prefilter must carry the shape through untouched"
        );
    }
}

#[test]
fn beta_one_reproduces_the_gaussian_bit_for_bit_in_f32() {
    let shape = CurveShape::with_beta(GAUSSIAN_BETA);
    let frame = match ScreenFrame::from_covariance(Sym2::<f32>::new(4.0, 0.7, 2.5)) {
        Some(f) => f,
        None => panic!("the f32 test frame must be SPD"),
    };
    let plain_p = <Gaussian as Kernel<f32>>::params_from(&shape);
    let super_p = <SuperGaussian as Kernel<f32>>::params_from(&shape);
    let plain = <Gaussian as Kernel<f32>>::prepare_screen(&plain_p, &frame);
    let sup = <SuperGaussian as Kernel<f32>>::prepare_screen(&super_p, &frame);
    for u in screen_grid() {
        let v = Vec2::new(u.x as f32, u.y as f32);
        let a = <Gaussian as Kernel<f32>>::eval_screen(&plain, v);
        let b = <SuperGaussian as Kernel<f32>>::eval_screen(&sup, v);
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "f32 eval_screen at {v:?}: the control must hold at BOTH precisions, \
             because precision is a later aspect and this run must not decide it"
        );
    }
}

#[test]
fn a_whole_frame_at_beta_one_is_bit_identical_to_the_gaussian_frame() {
    let mut covered = 0usize;
    for method in RenderMethod::implemented() {
        let plain = rendered(KernelFamily::Gaussian, method.clone());
        let sup = rendered(KernelFamily::SuperGaussian(Beta::One), method.clone());
        match identical_bits(&plain, &sup) {
            None => (),
            Some((x, y)) => panic!(
                "{:?}: the beta = 1 frame differs from the gaussian frame at \
                 pixel ({x},{y}). The control arm must come out of the WHOLE \
                 pipeline identical, not merely out of eval_screen, or the \
                 sweep is comparing pipelines rather than curves",
                method.id()
            ),
        }
        covered += 1;
    }
    assert!(
        covered >= 1,
        "the control ran on {covered} methods; with every method feature off this          file has nothing to compare and must be skipped, not pass vacuously"
    );
}

#[test]
fn the_beta_parameter_reaches_the_kernel_through_dispatch() {
    let mut covered = 0usize;
    for method in RenderMethod::implemented() {
        let control = rendered(KernelFamily::SuperGaussian(Beta::One), method.clone());
        for beta in Beta::SWEEP {
            if beta == Beta::One {
                continue;
            }
            let swept = rendered(KernelFamily::SuperGaussian(beta), method.clone());
            let gap = worst_gap(&control, &swept);
            assert!(
                gap > 0.01,
                "{:?}: beta = {} rendered within {gap} of beta = 1 everywhere. \
                 Before this aspect, dispatch handed PreparedScene a PERMANENTLY \
                 EMPTY params slice, so every parameterised kernel silently fell \
                 back to Params::default() and a beta sweep measured nothing at \
                 all. This assertion is the only thing standing between that \
                 defect and a report full of identical rows",
                method.id(),
                beta.label()
            );
        }
        let flat = rendered(KernelFamily::SuperGaussian(Beta::Eight), method.clone());
        let heavy = rendered(KernelFamily::SuperGaussian(Beta::Half), method.clone());
        assert!(
            worst_gap(&flat, &heavy) > 0.01,
            "{:?}: beta must reach the ray path as well as the screen path",
            method.id()
        );
        covered += 1;
    }
    assert!(
        covered >= 1,
        "the beta sweep ran on {covered} methods, so it measured nothing"
    );
}

#[test]
fn the_uniform_params_slice_is_the_length_of_the_scene() {
    let settings = SharedSettings::<f64>::contract();
    let scene = curve_scene();
    let shape = CurveShape::with_beta(4.0);
    let prepared = match PreparedScene::<f64, SuperGaussian>::build_uniform(&scene, &shape, &settings)
    {
        Ok(p) => p,
        Err(e) => panic!("build_uniform failed: {e}"),
    };
    assert_eq!(prepared.params.len(), scene.len());
    for index in 0..prepared.len() {
        assert_eq!(
            prepared.atom(index).params.beta,
            4.0,
            "atom {index} carries Params::default() rather than the requested \
             shape: a params slice shorter than the scene silently defaults the \
             tail, which is exactly how the empty slice hid this axis"
        );
    }
}

#[test]
fn per_atom_params_remain_reachable_under_the_uniform_helper() {
    let settings = SharedSettings::<f64>::contract();
    let scene = curve_scene();
    let mixed: Vec<_> = [0.5f64, 2.0, 8.0]
        .iter()
        .map(|b| <SuperGaussian as Kernel<f64>>::params_from(&CurveShape::with_beta(*b)))
        .collect();
    let prepared = match PreparedScene::<f64, SuperGaussian>::build(&scene, &mixed, &settings) {
        Ok(p) => p,
        Err(e) => panic!("build failed: {e}"),
    };
    let seen: Vec<f64> = (0..prepared.len()).map(|i| prepared.atom(i).params.beta).collect();
    assert_eq!(
        seen,
        vec![0.5, 2.0, 8.0],
        "the uniform helper must not have closed the per-atom door it was built \
         beside"
    );
}

#[test]
fn the_super_gaussian_solves_its_own_cutoff() {
    let frame = screen_frame();
    for beta in Beta::SWEEP {
        let shape = CurveShape::with_beta(beta.value());
        let p = <SuperGaussian as Kernel<f64>>::params_from(&shape);
        let prepared = <SuperGaussian as Kernel<f64>>::prepare_screen(&p, &frame);
        for peak in [1.0f64, 0.6, 0.2, 0.03] {
            let r = <SuperGaussian as Kernel<f64>>::support_screen(&prepared, peak, EPS_ALPHA);
            let closed = (2.0 * (peak / EPS_ALPHA).ln().powf(1.0 / beta.value())).sqrt();
            assert!(
                (r - closed).abs() < 1e-12,
                "beta {} peak {peak}: support {r} against the closed form {closed}",
                beta.label()
            );
            let at_edge =
                <SuperGaussian as Kernel<f64>>::eval_screen(&prepared, Vec2::new(r, 0.0)) * peak;
            assert!(
                (at_edge - EPS_ALPHA).abs() < 1e-12,
                "beta {} peak {peak}: the atom is worth {at_edge} at its own \
                 support radius, and the cutoff is {EPS_ALPHA}",
                beta.label()
            );
        }
    }
    assert!(
        super_gaussian_shrinks_with_beta(),
        "a higher beta must give a SMALLER support radius: that is the entire \
         binning-cost argument for the flat-topped end of the sweep"
    );
}

fn super_gaussian_shrinks_with_beta() -> bool {
    let frame = screen_frame();
    let mut previous = f64::INFINITY;
    for beta in Beta::SWEEP {
        let p = <SuperGaussian as Kernel<f64>>::params_from(&CurveShape::with_beta(beta.value()));
        let prepared = <SuperGaussian as Kernel<f64>>::prepare_screen(&p, &frame);
        let r = <SuperGaussian as Kernel<f64>>::support_screen(&prepared, 1.0, EPS_ALPHA);
        if !(r < previous) {
            return false;
        }
        previous = r;
    }
    true
}

#[test]
fn the_compact_bump_is_exactly_zero_at_and_beyond_its_radius() {
    let p = <CompactBump as Kernel<f64>>::params_from(&CurveShape::gaussian());
    let screen = <CompactBump as Kernel<f64>>::prepare_screen(&p, &screen_frame());
    assert_eq!(
        ZERO_RADIUS,
        ZERO_RADIUS_SQUARED.sqrt(),
        "the radius constant and the squared radius the evaluator divides by \
         must be the same number"
    );
    for spoke in 0..64 {
        let theta = core::f64::consts::TAU * (spoke as f64) / 64.0;
        let (s, c) = theta.sin_cos();
        for scale in [1.0f64, 1.0 + 1e-12, 1.5, 4.0, 100.0] {
            let r = ZERO_RADIUS * scale;
            let v = <CompactBump as Kernel<f64>>::eval_screen(&screen, Vec2::new(r * c, r * s));
            assert_eq!(
                v, 0.0,
                "at radius {r} the compact kernel returned {v}. `exactly zero at \
                 a finite radius` is the property being tested against the \
                 gaussian's `small but never zero`, and `small` would pass a \
                 tolerance-based check while failing the claim"
            );
        }
        let inside = ZERO_RADIUS * (1.0 - 1e-9);
        let v = <CompactBump as Kernel<f64>>::eval_screen(
            &screen,
            Vec2::new(inside * c, inside * s),
        );
        assert!(v > 0.0 && v < 1e-20, "just inside the radius the value is {v}");
    }
    let ray = <CompactBump as Kernel<f64>>::prepare_ray(&p, &ray_frame());
    assert_eq!(
        <CompactBump as Kernel<f64>>::eval_ray(&ray, Vec3::new(ZERO_RADIUS, 0.0, 0.0)),
        0.0
    );
    assert_eq!(
        <CompactBump as Kernel<f64>>::eval_screen(&screen, Vec2::zero()),
        1.0,
        "peak-normalised, like every other kernel here"
    );
}

#[test]
fn the_compact_support_solves_the_cutoff_and_never_exceeds_the_zero_radius() {
    let p = <CompactBump as Kernel<f64>>::params_from(&CurveShape::gaussian());
    let screen = <CompactBump as Kernel<f64>>::prepare_screen(&p, &screen_frame());
    for peak in [1.0f64, 0.6, 0.2, 0.03, 2.0 / 255.0] {
        let r = <CompactBump as Kernel<f64>>::support_screen(&screen, peak, EPS_ALPHA);
        assert!(
            r <= ZERO_RADIUS,
            "peak {peak}: support {r} exceeds the radius at which the kernel is \
             already exactly zero"
        );
        let closed = ZERO_RADIUS * (1.0 - (EPS_ALPHA / peak).powf(ONE_THIRD)).sqrt();
        assert!(
            (r - closed).abs() < 1e-12,
            "peak {peak}: support {r} against the closed form {closed}"
        );
        let at_edge = <CompactBump as Kernel<f64>>::eval_screen(&screen, Vec2::new(r, 0.0)) * peak;
        assert!(
            (at_edge - EPS_ALPHA).abs() < 1e-12,
            "peak {peak}: worth {at_edge} at its own support radius"
        );
    }
    let full = <CompactBump as Kernel<f64>>::support_screen(&screen, 1.0, 0.0);
    assert_eq!(
        full, ZERO_RADIUS,
        "with no cutoff the support must be the true radius"
    );
    let gaussian_radius = lights::kernel::gaussian::radius_for_cutoff(1.0, EPS_ALPHA);
    let compact_radius = <CompactBump as Kernel<f64>>::support_screen(&screen, 1.0, EPS_ALPHA);
    assert!(
        compact_radius < gaussian_radius,
        "at unit variance and the same cutoff the compact kernel must bin \
         smaller than the gaussian ({compact_radius} against {gaussian_radius}), \
         which is the only reason to carry a second curve at all"
    );
}

#[test]
fn the_compact_bump_carries_the_same_second_moment_as_the_gaussian() {
    let steps = 400_000usize;
    let mut mass = 0.0f64;
    let mut moment = 0.0f64;
    for i in 0..steps {
        let r = ZERO_RADIUS * ((i as f64) + 0.5) / (steps as f64);
        let s = r * r / ZERO_RADIUS_SQUARED;
        let f = (1.0 - s) * (1.0 - s) * (1.0 - s);
        mass += f * r;
        moment += f * r * r * r;
    }
    let variance_per_axis = moment / (2.0 * mass);
    assert!(
        (variance_per_axis - 1.0).abs() < 1e-6,
        "the compact kernel's per-axis second moment is {variance_per_axis}, not \
         1. The covariance an atom carries would then MEAN something different \
         under this kernel than under the gaussian, the prefilter dilation would \
         be adding a pixel box to the wrong quantity, and the sweep would be \
         comparing widths rather than shapes"
    );
}

#[test]
fn the_family_table_names_exactly_one_kernel_each() {
    let mut names: Vec<&'static str> = KernelFamily::ALL.iter().map(|k| k.name()).collect();
    let count = names.len();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), count, "two families share a report label");

    for family in KernelFamily::ALL {
        let (kernel_name, id, blend_over) = match family {
            KernelFamily::Gaussian => (
                <Gaussian as Kernel<f64>>::NAME,
                <Gaussian as Kernel<f64>>::ID,
                true,
            ),
            KernelFamily::GaussianNormalised => (
                <GaussianNormalised as Kernel<f64>>::NAME,
                <GaussianNormalised as Kernel<f64>>::ID,
                false,
            ),
            KernelFamily::SuperGaussian(_) => (
                <SuperGaussian as Kernel<f64>>::NAME,
                <SuperGaussian as Kernel<f64>>::ID,
                true,
            ),
            KernelFamily::CompactBump => (
                <CompactBump as Kernel<f64>>::NAME,
                <CompactBump as Kernel<f64>>::ID,
                true,
            ),
            KernelFamily::GaborColour(_) => (
                <GaborColour as Kernel<f64>>::NAME,
                <GaborColour as Kernel<f64>>::ID,
                true,
            ),
            KernelFamily::GaborDensity(_) => (
                <GaborDensity as Kernel<f64>>::NAME,
                <GaborDensity as Kernel<f64>>::ID,
                false,
            ),
            KernelFamily::GaussHermiteColour(_) => (
                <GaussHermiteColour as Kernel<f64>>::NAME,
                <GaussHermiteColour as Kernel<f64>>::ID,
                true,
            ),
            KernelFamily::GaussHermiteDensity(_) => (
                <GaussHermiteDensity as Kernel<f64>>::NAME,
                <GaussHermiteDensity as Kernel<f64>>::ID,
                false,
            ),
        };
        assert_eq!(family.kernel_name(), kernel_name, "{family:?}");
        assert_eq!(family.id(), id, "{family:?}");
        assert_eq!(family.blend_is_over(), blend_over, "{family:?}");
        assert_eq!(
            family.is_signed(),
            matches!(
                family,
                KernelFamily::GaborDensity(_) | KernelFamily::GaussHermiteDensity(_)
            ),
            "{family:?}: the signed families are exactly the two Additive arms. \
             GaussianNormalised is the standing proof that signed cannot be \
             read off blend_is_over: it composites Normalised, so blend_is_over \
             is false, and it never goes negative"
        );
    }

    assert!(KernelFamily::Gaussian.curve_is_gaussian());
    assert!(KernelFamily::GaussianNormalised.curve_is_gaussian());
    assert!(KernelFamily::SuperGaussian(Beta::One).curve_is_gaussian());
    assert!(!KernelFamily::SuperGaussian(Beta::Two).curve_is_gaussian());
    assert!(!KernelFamily::CompactBump.curve_is_gaussian());

    assert_eq!(KernelFamily::SuperGaussian(Beta::One).shape().beta, 1.0);
    assert_eq!(KernelFamily::SuperGaussian(Beta::Half).shape().beta, 0.5);
    assert_eq!(KernelFamily::SuperGaussian(Beta::Eight).shape().beta, 8.0);
    assert!(CurveShape::default().is_gaussian());

    let labels: Vec<&'static str> = Beta::SWEEP.iter().map(|b| b.label()).collect();
    assert_eq!(labels, vec!["0.5", "1", "2", "4", "8"]);
    let values: Vec<f64> = Beta::SWEEP.iter().map(|b| b.value()).collect();
    assert_eq!(values, vec![0.5, 1.0, 2.0, 4.0, 8.0]);

    for family in KernelFamily::CURVE_SWEEP {
        assert!(
            family.blend_is_over(),
            "{family:?} is on the atom-curve sweep with a blend other than Over: \
             the blend axis is HELD for this run and a second axis moving inside \
             it is the exact mistake this run exists to stop repeating"
        );
    }
}

#[test]
#[cfg(all(feature = "method-splat", feature = "method-raycast"))]
fn the_tolerance_path_refuses_a_curve_its_bound_was_not_derived_for() {
    let mut request = curve_request(KernelFamily::CompactBump, RenderMethod::splat_contract());
    request.collect = Collect::PixelsAndTolerance;
    match render(&request) {
        Ok(_) => (),
        Err(e) => panic!("dispatch itself must still render: {e}"),
    }

    let outcome = lights::agreement::run_pair(
        &request,
        &RenderMethod::splat_contract(),
        &RenderMethod::raycast_contract(),
    );
    match outcome {
        Err(LightsError::ContractViolation(ContractViolation::KernelCurveNotGaussian {
            kernel,
        })) => assert_eq!(kernel, "compact-bump"),
        Err(e) => panic!("the wrong refusal: {e}"),
        Ok(_) => panic!(
            "the cross-method agreement harness certified a non-gaussian curve. \
             Its per-atom divergence bound is a derivative bound on exp(-r^2/2) \
             and on nothing else, so a pass here would be a certificate that was \
             never computed for the kernel it was applied to"
        ),
    }

    let mut beta_one = curve_request(
        KernelFamily::SuperGaussian(Beta::One),
        RenderMethod::splat_contract(),
    );
    beta_one.collect = Collect::PixelsAndTolerance;
    match lights::agreement::run_pair(
        &beta_one,
        &RenderMethod::splat_contract(),
        &RenderMethod::raycast_contract(),
    ) {
        Ok((report, _, _, _)) => assert_eq!(
            report.failures, 0,
            "beta = 1 IS the gaussian, so it must pass the same cross-method \
             certificate the gaussian passes"
        ),
        Err(e) => panic!("beta = 1 must be admitted by the agreement harness: {e}"),
    }

    let mut beta_two = curve_request(
        KernelFamily::SuperGaussian(Beta::Two),
        RenderMethod::splat_contract(),
    );
    beta_two.collect = Collect::PixelsAndTolerance;
    match lights::agreement::run_pair(
        &beta_two,
        &RenderMethod::splat_contract(),
        &RenderMethod::raycast_contract(),
    ) {
        Err(LightsError::ContractViolation(ContractViolation::KernelCurveNotGaussian {
            kernel,
        })) => assert_eq!(kernel, "super-gaussian-b2"),
        Err(e) => panic!("the wrong refusal for beta = 2: {e}"),
        Ok(_) => panic!("beta = 2 is not a gaussian and must not be certified"),
    }
}

#[test]
fn the_curve_instrument_can_actually_fail() {
    let frame = screen_frame();
    let p = <SuperGaussian as Kernel<f64>>::params_from(&CurveShape::with_beta(2.0));
    let prepared = <SuperGaussian as Kernel<f64>>::prepare_screen(&p, &frame);
    let one = <SuperGaussian as Kernel<f64>>::params_from(&CurveShape::with_beta(1.0));
    let control = <SuperGaussian as Kernel<f64>>::prepare_screen(&one, &frame);
    let u = Vec2::new(1.3, -0.4);
    assert_ne!(
        <SuperGaussian as Kernel<f64>>::eval_screen(&prepared, u).to_bits(),
        <SuperGaussian as Kernel<f64>>::eval_screen(&control, u).to_bits(),
        "the bit-for-bit comparison used by the control test must be able to see \
         a difference when one exists, or it certifies nothing"
    );

    let compact = <CompactBump as Kernel<f64>>::prepare_screen(
        &<CompactBump as Kernel<f64>>::params_from(&CurveShape::gaussian()),
        &frame,
    );
    assert_ne!(
        <CompactBump as Kernel<f64>>::eval_screen(&compact, u),
        <Gaussian as Kernel<f64>>::eval_screen(
            &<Gaussian as Kernel<f64>>::prepare_screen(
                &<Gaussian as Kernel<f64>>::params_from(&CurveShape::gaussian()),
                &frame
            ),
            u
        ),
        "the compact curve and the gaussian must not agree at an interior point"
    );

    let mut covered = 0usize;
    for method in RenderMethod::implemented() {
        let a = rendered(KernelFamily::Gaussian, method.clone());
        let b = rendered(KernelFamily::CompactBump, method.clone());
        assert!(
            identical_bits(&a, &b).is_some(),
            "{:?}: the whole-frame bit comparison must be able to report a \
             difference",
            method.id()
        );
        covered += 1;
    }
    assert!(
        covered >= 1,
        "the whole-frame instrument ran on {covered} methods"
    );
}

#[test]
fn a_uniform_shapes_list_is_bit_identical_to_handing_dispatch_none() {
    let mut covered = 0usize;
    for method in RenderMethod::implemented() {
        for beta in Beta::SWEEP {
            let kernel = KernelFamily::SuperGaussian(beta);
            let base = curve_request(kernel, method.clone());
            let atoms = base.scene.len();
            let mut spelled = base.clone();
            spelled.shapes = Some(vec![kernel.shape(); atoms]);
            let a = match render(&base) {
                Ok(o) => o.image,
                Err(e) => panic!("{e}"),
            };
            let b = match render(&spelled) {
                Ok(o) => o.image,
                Err(e) => panic!("{e}"),
            };
            match identical_bits(&a, &b) {
                None => (),
                Some((x, y)) => panic!(
                    "{:?} beta {}: spelling out the SAME shape per atom differs \
                     from the uniform path at pixel ({x},{y}). None must mean \
                     exactly what build_uniform meant, or opening this seam has \
                     moved every number the crate ever published",
                    method.id(),
                    beta.label()
                ),
            }
        }
        covered += 1;
    }
    assert_eq!(
        covered,
        RenderMethod::implemented().len(),
        "every built method must be covered, and with none built this file is \
         skipped rather than passing vacuously"
    );
    assert!(covered >= 1);
}

#[test]
fn two_atoms_can_now_carry_different_curves_in_one_render() {
    let mut covered = 0usize;
    for method in RenderMethod::implemented() {
        let kernel = KernelFamily::SuperGaussian(Beta::One);
        let base = curve_request(kernel, method.clone());
        let atoms = base.scene.len();
        assert!(atoms >= 2, "the mixed case needs at least two atoms, got {atoms}");

        let flat = KernelFamily::SuperGaussian(Beta::Eight).shape();
        let heavy = KernelFamily::SuperGaussian(Beta::Half).shape();

        let mut mixed = base.clone();
        let mut list = vec![heavy; atoms];
        list[0] = flat;
        mixed.shapes = Some(list);

        let mut all_flat = base.clone();
        all_flat.shapes = Some(vec![flat; atoms]);
        let mut all_heavy = base.clone();
        all_heavy.shapes = Some(vec![heavy; atoms]);

        let m = match render(&mixed) {
            Ok(o) => o.image,
            Err(e) => panic!("{e}"),
        };
        let f = match render(&all_flat) {
            Ok(o) => o.image,
            Err(e) => panic!("{e}"),
        };
        let h = match render(&all_heavy) {
            Ok(o) => o.image,
            Err(e) => panic!("{e}"),
        };
        assert!(
            worst_gap(&m, &f) > 0.01,
            "{:?}: the mixed render matched the all-flat one, so the per-atom \
             shape never reached the atoms that were not atom 0",
            method.id()
        );
        assert!(
            worst_gap(&m, &h) > 0.01,
            "{:?}: the mixed render matched the all-heavy one, so atom 0's own \
             shape was discarded. A scene where every atom carries the SAME \
             curve is reachable through KernelFamily alone; this test exists \
             because a scene where they differ was not",
            method.id()
        );
        covered += 1;
    }
    assert_eq!(covered, RenderMethod::implemented().len());
    assert!(covered >= 1);
}

#[test]
fn a_shapes_list_that_does_not_match_the_scene_is_refused_rather_than_padded() {
    let mut covered = 0usize;
    for method in RenderMethod::implemented() {
        let base = curve_request(KernelFamily::SuperGaussian(Beta::Two), method.clone());
        let atoms = base.scene.len();
        for wrong in [atoms - 1, atoms + 1, 0] {
            let mut bad = base.clone();
            bad.shapes = Some(vec![CurveShape::gaussian(); wrong]);
            match render(&bad) {
                Err(LightsError::ShapeCountMismatch { shapes, atoms: n }) => {
                    assert_eq!(shapes, wrong);
                    assert_eq!(n, atoms);
                }
                Err(e) => panic!("the refusal must name the mismatch: {e}"),
                Ok(_) => panic!(
                    "{:?}: {wrong} shapes for {atoms} atoms rendered anyway. \
                     PreparedScene::build falls back to Params::default() per \
                     index, so a short list would silently give the tail atoms a \
                     curve nobody asked for",
                    method.id()
                ),
            }
        }
        covered += 1;
    }
    assert!(covered >= 1);
}
