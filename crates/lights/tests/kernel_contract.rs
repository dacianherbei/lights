use lights::blend::{BlendMode, CompositeClass};
use lights::kernel::frame::{RayFrame, ScreenFrame};
use lights::kernel::compact::CompactBump;
use lights::kernel::gabor::{GaborColour, GaborDensity};
use lights::kernel::gaussian::{
    Gaussian, GaussianNormalised, GaussianParams, REFERENCE_THREE_SIGMA_CUTOFF,
};
use lights::kernel::hermite::{GaussHermiteColour, GaussHermiteDensity};
use lights::kernel::supergaussian::SuperGaussian;
use lights::kernel::{CurveShape, Kernel, KernelFamily, KernelId, describe};
use lights::numerics::symmetric::{LowerTri3, Sym2, Sym3};
use lights::numerics::vector::{Vec2, Vec3};

const EPS_ALPHA: f64 = 1.0 / 255.0;

fn screen_frame() -> ScreenFrame<f64> {
    ScreenFrame::from_covariance(Sym2::new(4.0, 0.7, 2.5)).expect("SPD")
}

fn ray_frame() -> RayFrame<f64> {
    let sigma = Sym3::<f64>::new(0.04, 0.0, 0.0, 0.01, 0.0, 0.0025);
    RayFrame::new(
        sigma.inverse_spd().expect("SPD"),
        sigma.cholesky().expect("SPD").inverse(),
        Vec3::new(0.0, 0.0, 1.0),
    )
}

fn bounded<K: Kernel<f64>>(shape: &CurveShape) {
    let p = K::params_from(shape);
    let screen = K::prepare_screen(&p, &screen_frame());
    let signed = <K::Blend as BlendMode<f64>>::SIGNED;
    for i in 0..64 {
        for j in 0..64 {
            let u = Vec2::new(
                -6.0 + 12.0 * (i as f64) / 63.0,
                -6.0 + 12.0 * (j as f64) / 63.0,
            );
            let v = K::eval_screen(&screen, u);
            assert!(v.is_finite(), "eval_screen must be finite at {u:?}");
            assert!(v.abs() <= 1.0 + 1e-5, "eval_screen exceeded 1: {v}");
            if !signed {
                assert!(v >= 0.0, "an unsigned kernel must never go negative: {v}");
            }
            let effective = v * K::modulate_screen(&screen, u);
            assert!(
                effective.is_finite() && effective.abs() <= 1.0 + 1e-5,
                "eval_screen * modulate_screen is the value the renderer \
                 actually uses and it exceeded 1 at {u:?}: {effective}"
            );
        }
    }
    let ray = K::prepare_ray(&p, &ray_frame());
    for i in 0..16 {
        for j in 0..16 {
            for k in 0..16 {
                let u = Vec3::new(
                    -5.0 + 10.0 * (i as f64) / 15.0,
                    -5.0 + 10.0 * (j as f64) / 15.0,
                    -5.0 + 10.0 * (k as f64) / 15.0,
                );
                let v = K::eval_ray(&ray, u);
                assert!(v.is_finite() && v.abs() <= 1.0 + 1e-5);
                if !signed {
                    assert!(v >= 0.0);
                }
                let effective = v * K::modulate_ray(&ray, u);
                assert!(effective.is_finite() && effective.abs() <= 1.0 + 1e-5);
            }
        }
    }
}

fn peak_normalised<K: Kernel<f64>>(shape: &CurveShape) {
    let p = K::params_from(shape);
    let screen = K::prepare_screen(&p, &screen_frame());
    let ray = K::prepare_ray(&p, &ray_frame());
    assert!(
        (K::eval_screen(&screen, Vec2::zero()) - 1.0).abs() < 1e-15,
        "the kernel is peak-normalised, not a pdf"
    );
    assert!((K::eval_ray(&ray, Vec3::zero()) - 1.0).abs() < 1e-15);
}

fn not_peak_normalised<K: Kernel<f64>>(shape: &CurveShape) {
    let p = K::params_from(shape);
    let screen = K::prepare_screen(&p, &screen_frame());
    let v = K::eval_screen(&screen, Vec2::zero());
    assert!(
        (v - 1.0).abs() >= 1e-15,
        "this family is EXEMPTED from peak-normalisation because it folds a \
         signed factor into eval_screen: gauss-hermite-density is 0 at its own \
         centre at order 1 and -1 at order 2, and gabor-density is cos(phase). \
         It just returned {v}, which is 1.0 — either the exemption is now \
         wrong, or the fold was removed and the modulation is silently dead"
    );
    assert!(v.abs() <= 1.0 + 1e-5, "the exemption is from the NORMALISATION, not from the bound: {v}");
}

fn modulation_is_live<K: Kernel<f64>>(shape: &CurveShape) {
    let p = K::params_from(shape);
    let screen = K::prepare_screen(&p, &screen_frame());
    let ray = K::prepare_ray(&p, &ray_frame());
    let mut screen_moved = false;
    let mut ray_moved = false;
    for i in 0..64 {
        let t = -4.0 + 8.0 * (i as f64) / 63.0;
        if (K::modulate_screen(&screen, Vec2::new(t, 0.3 * t)) - 1.0).abs() > 1e-9 {
            screen_moved = true;
        }
        if (K::modulate_ray(&ray, Vec3::new(t, 0.3 * t, 0.1 * t)) - 1.0).abs() > 1e-9 {
            ray_moved = true;
        }
    }
    assert_eq!(
        screen_moved, K::MODULATED,
        "MODULATED declares whether the renderers take the \
         K::modulate_screen branch at all (src/method/splat/execute.rs, \
         src/method/raycast/execute.rs). A kernel that declares it and returns \
         the trait default T::ONE everywhere has a dead branch; one that does \
         not declare it and modulates anyway is silently ignored"
    );
    assert_eq!(ray_moved, K::MODULATED, "the same, on the ray path");
}

fn radial_symmetry_matches_its_declaration<K: Kernel<f64>>(shape: &CurveShape) {
    let p = K::params_from(shape);
    let screen = K::prepare_screen(&p, &screen_frame());
    let ray = K::prepare_ray(&p, &ray_frame());
    let mut screen_spread = 0.0f64;
    let mut ray_spread = 0.0f64;
    for ring in [0.4f64, 0.9, 1.3, 2.1] {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for spoke in 0..24 {
            let theta = core::f64::consts::TAU * (spoke as f64) / 24.0;
            let u = Vec2::new(ring * theta.cos(), ring * theta.sin());
            let v = K::eval_screen(&screen, u) * K::modulate_screen(&screen, u);
            lo = lo.min(v);
            hi = hi.max(v);
        }
        screen_spread = screen_spread.max(hi - lo);
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for spoke in 0..12 {
            let theta = core::f64::consts::TAU * (spoke as f64) / 12.0;
            for tilt in [-0.8f64, -0.3, 0.0, 0.5, 0.9] {
                let flat = (1.0 - tilt * tilt).sqrt();
                let u = Vec3::new(
                    ring * flat * theta.cos(),
                    ring * flat * theta.sin(),
                    ring * tilt,
                );
                let v = K::eval_ray(&ray, u) * K::modulate_ray(&ray, u);
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
        ray_spread = ray_spread.max(hi - lo);
    }
    assert_eq!(
        screen_spread < 1e-12,
        K::RADIAL,
        "{}: RADIAL declares that the value the renderer uses depends on the \
         whitened radius alone. Largest spread around one screen ring: {screen_spread}",
        K::NAME
    );
    assert_eq!(
        ray_spread < 1e-12,
        K::RADIAL,
        "{}: the same, around one sphere on the ray path: {ray_spread}",
        K::NAME
    );
}

fn truncation_beyond_support<K: Kernel<f64>>(shape: &CurveShape) {
    let p = K::params_from(shape);
    let screen = K::prepare_screen(&p, &screen_frame());
    for peak in [1.0, 0.7, 0.25, 0.05] {
        let r = K::support_screen(&screen, peak, EPS_ALPHA);
        if r <= 0.0 {
            continue;
        }
        for ring in 1..=6 {
            let radius = r * (1.0 + 0.05 * ring as f64);
            for spoke in 0..64 {
                let theta = core::f64::consts::TAU * (spoke as f64) / 64.0;
                let u = Vec2::new(radius * theta.cos(), radius * theta.sin());
                let contribution = K::eval_screen(&screen, u).abs() * peak;
                assert!(
                    contribution <= EPS_ALPHA * (1.0 + 1e-4),
                    "peak {peak} radius {radius}: {contribution} exceeds the cutoff"
                );
            }
        }
    }
    let ray = K::prepare_ray(&p, &ray_frame());
    for peak in [1.0, 0.4] {
        let r = K::support_ray(&ray, peak, EPS_ALPHA);
        for ring in 1..=6 {
            let radius = r * (1.0 + 0.05 * ring as f64);
            for spoke in 0..32 {
                let theta = core::f64::consts::TAU * (spoke as f64) / 32.0;
                for tilt in [-0.6f64, 0.0, 0.6] {
                    let flat = (1.0 - tilt * tilt).max(0.0).sqrt();
                    let u = Vec3::new(
                        radius * flat * theta.cos(),
                        radius * flat * theta.sin(),
                        radius * tilt,
                    );
                    assert!(K::eval_ray(&ray, u).abs() * peak <= EPS_ALPHA * (1.0 + 1e-4));
                }
            }
        }
    }
}

macro_rules! for_kernel {
    ($family:expr, $check:ident, $shape:expr) => {
        match $family {
            KernelFamily::Gaussian => $check::<Gaussian>($shape),
            KernelFamily::GaussianNormalised => $check::<GaussianNormalised>($shape),
            KernelFamily::SuperGaussian(_) => $check::<SuperGaussian>($shape),
            KernelFamily::CompactBump => $check::<CompactBump>($shape),
            KernelFamily::GaborColour(_) => $check::<GaborColour>($shape),
            KernelFamily::GaborDensity(_) => $check::<GaborDensity>($shape),
            KernelFamily::GaussHermiteColour(_) => $check::<GaussHermiteColour>($shape),
            KernelFamily::GaussHermiteDensity(_) => $check::<GaussHermiteDensity>($shape),
        }
    };
}

#[test]
fn kernel_bounded_and_sign_consistent() {
    let mut seen = 0usize;
    for family in KernelFamily::ALL {
        let shape = family.shape();
        for_kernel!(family, bounded, &shape);
        seen += 1;
    }
    assert_eq!(
        seen,
        KernelFamily::ALL.len(),
        "the bound is the only half of the old contract that is true for all \
         eight kernels, so it is the only half that runs over ALL"
    );
}

#[test]
fn kernel_peak_normalisation_holds_where_it_is_claimed() {
    let mut normalised = 0usize;
    let mut exempt = 0usize;
    for family in KernelFamily::ALL {
        let shape = family.shape();
        if family.eval_is_peak_normalised() {
            for_kernel!(family, peak_normalised, &shape);
            normalised += 1;
        } else {
            for_kernel!(family, not_peak_normalised, &shape);
            exempt += 1;
        }
    }
    assert_eq!(
        exempt, 3,
        "THREE families are exempted, not four: gabor-density at zero phase \
         returns cos(0) = 1 at the origin and IS peak-normalised, so only the \
         quadrature carrier and the two hermite orders miss. An exemption that \
         grows silently is how a harness comes to test nothing"
    );
    assert_eq!(normalised + exempt, KernelFamily::ALL.len());
    assert!(normalised > 0);
}

#[test]
fn kernel_modulation_matches_its_declaration() {
    let mut modulated = 0usize;
    for family in KernelFamily::ALL {
        let shape = family.shape();
        for_kernel!(family, modulation_is_live, &shape);
        if family.is_modulated() {
            modulated += 1;
        }
    }
    assert_eq!(
        modulated, 5,
        "three gabor-colour carriers and two hermite-colour orders declare \
         MODULATED. Until L-56 no family declared it at all, so the \
         K::MODULATED branch in both renderers was unreachable from render()"
    );
}

#[test]
fn kernel_radial_declaration_is_measured_for_every_family() {
    let mut radial = 0usize;
    let mut seen = 0usize;
    for family in KernelFamily::ALL {
        let shape = family.shape();
        for_kernel!(family, radial_symmetry_matches_its_declaration, &shape);
        if matches!(
            family,
            KernelFamily::Gaussian
                | KernelFamily::GaussianNormalised
                | KernelFamily::SuperGaussian(_)
                | KernelFamily::CompactBump
        ) {
            radial += 1;
        }
        seen += 1;
    }
    assert_eq!(seen, KernelFamily::ALL.len());
    assert!(
        radial > 0 && radial < seen,
        "both answers must be exercised, or the check can pass by testing one side only"
    );
}

#[test]
fn kernel_truncation_beyond_support() {
    let mut seen = 0usize;
    for family in KernelFamily::ALL {
        let shape = family.shape();
        for_kernel!(family, truncation_beyond_support, &shape);
        seen += 1;
    }
    assert_eq!(seen, KernelFamily::ALL.len());
}

#[test]
fn kernel_support_solves_the_equation() {
    let screen = screen_frame();
    let p = GaussianParams::default();
    let prepared = <Gaussian as Kernel<f64>>::prepare_screen(&p, &screen);

    for peak in [1.00f64, 0.50, 0.25, 0.10] {
        let expected = (2.0 * (peak / EPS_ALPHA).ln()).sqrt();
        let got = <Gaussian as Kernel<f64>>::support_screen(&prepared, peak, EPS_ALPHA);
        assert!(
            (got - expected).abs() < 1e-12,
            "peak {peak}: support {got} does not SOLVE r = sqrt(2 ln(peak/cutoff)) = {expected}"
        );
        assert!(
            (got - 3.0).abs() > 1e-6 || (expected - 3.0).abs() < 1e-6,
            "a constant 3.0 must not pass"
        );
    }
    for cutoff in [1.0e-2f64, 1.0e-3, 1.0e-4] {
        let expected = (2.0 * (1.0f64 / cutoff).ln()).sqrt();
        let got = <Gaussian as Kernel<f64>>::support_screen(&prepared, 1.0, cutoff);
        assert!((got - expected).abs() < 1e-12, "cutoff {cutoff}: {got} vs {expected}");
    }

    let at_one = <Gaussian as Kernel<f64>>::support_screen(&prepared, 1.0, EPS_ALPHA);
    assert!(
        (at_one - 3.3290429691304455).abs() < 1e-12,
        "recomputed from the formula in this session: {at_one}"
    );
    let quarter = <Gaussian as Kernel<f64>>::support_screen(&prepared, 0.25, EPS_ALPHA);
    assert!(
        (quarter - 2.8826963711215012).abs() < 1e-12,
        "the math sheet prints 2.8825 for this row and is WRONG: {quarter}"
    );
    let tenth = <Gaussian as Kernel<f64>>::support_screen(&prepared, 0.10, EPS_ALPHA);
    assert!(
        (tenth - 2.5450652063019445).abs() < 1e-12,
        "the math sheet prints 2.5427 for this row and is WRONG: {tenth}"
    );
}

#[test]
fn kernel_support_is_amplitude_aware() {
    let prepared =
        <Gaussian as Kernel<f64>>::prepare_screen(&GaussianParams::default(), &screen_frame());
    let support = |peak: f64| <Gaussian as Kernel<f64>>::support_screen(&prepared, peak, EPS_ALPHA);

    assert_eq!(
        support(EPS_ALPHA),
        0.0,
        "a peak exactly at the cutoff contributes nowhere"
    );
    assert_eq!(support(0.003), 0.0, "a peak below the cutoff is rejected");
    assert_eq!(support(0.0), 0.0);
    assert_eq!(support(-1.0), 0.0);

    let just_above = support(0.004);
    assert!(
        (just_above - 0.19901068964344468).abs() < 1e-12,
        "peak 0.004 is ABOVE 1/255 = 0.0039216, so the support is a small \
         positive number, not zero as the build spec claimed: {just_above}"
    );
    let modest = support(0.06);
    assert!(
        (modest - 2.3357452037405064).abs() < 1e-12,
        "recomputed from the formula; the build spec's 2.331 is wrong: {modest}"
    );

    let mut previous = 0.0;
    for peak in [0.01f64, 0.1, 0.3, 0.6, 1.0] {
        let r = support(peak);
        assert!(
            r > previous,
            "support must be strictly increasing in the peak: a support function \
             that IGNORES its peak argument fails here and nowhere else"
        );
        previous = r;
    }
}

#[test]
fn gaussian_reference_three_sigma_is_a_cutoff_choice() {
    assert!(
        (REFERENCE_THREE_SIGMA_CUTOFF - (-4.5f64).exp()).abs() < 1e-18,
        "3 sigma is the solution at exactly one epsilon"
    );
    assert!((REFERENCE_THREE_SIGMA_CUTOFF - 0.011108996538242306).abs() < 1e-18);
    let prepared =
        <Gaussian as Kernel<f64>>::prepare_screen(&GaussianParams::default(), &screen_frame());
    let r = <Gaussian as Kernel<f64>>::support_screen(&prepared, 1.0, REFERENCE_THREE_SIGMA_CUTOFF);
    assert!((r - 3.0).abs() < 1e-12, "got {r}");
    assert!(
        REFERENCE_THREE_SIGMA_CUTOFF > EPS_ALPHA,
        "3 sigma leaves 0.0111 of peak, ABOVE the 1/255 cutoff for opacity above \
         0.353: the reference 3-sigma box is a HARD TRUNCATION"
    );
}

#[test]
fn gaussian_prefilter_dilates_and_reports_energy_gain() {
    let p = GaussianParams::default();
    let raw = Sym2::<f64>::new(100.0, 0.0, 100.0);
    let (params, dilated, gain) = <Gaussian as Kernel<f64>>::prefilter(&p, raw, 0.3);
    assert_eq!(params, p);
    assert!((dilated.xx - 100.3).abs() < 1e-12 && (dilated.yy - 100.3).abs() < 1e-12);
    assert!(
        (dilated.xy - raw.xy).abs() < 1e-15,
        "the low-pass adds to the DIAGONAL only"
    );
    let expected_gain = (raw.determinant() / dilated.determinant()).sqrt();
    assert!((gain - expected_gain).abs() < 1e-14);
    assert!(gain < 1.0 && gain > 0.99);

    let (_, untouched, unit_gain) = <Gaussian as Kernel<f64>>::prefilter(&p, raw, 0.0);
    assert_eq!(untouched, raw, "zero dilation is a no-op");
    assert_eq!(unit_gain, 1.0);

    let subpixel = Sym2::<f64>::new(0.04, 0.0, 0.04);
    let (_, sub_dilated, sub_gain) = <Gaussian as Kernel<f64>>::prefilter(&p, subpixel, 0.3);
    assert!((sub_dilated.xx - 0.34).abs() < 1e-12);
    assert!(
        sub_gain < 0.13,
        "for a 0.2x0.2 px atom the compensation factor is about 2/17 - an 8.5x \
         brightness difference if the flag is set differently on the two \
         renderers. got {sub_gain}"
    );
}

#[test]
fn gaussian_evaluates_the_whitened_mahalanobis_form() {
    let cov = Sym2::<f64>::new(4.0, 0.7, 2.5);
    let frame = ScreenFrame::from_covariance(cov).expect("SPD");
    let prepared = <Gaussian as Kernel<f64>>::prepare_screen(&GaussianParams::default(), &frame);
    for offset in [
        Vec2::new(0.0, 0.0),
        Vec2::new(1.0, 0.0),
        Vec2::new(0.0, 1.0),
        Vec2::new(3.0, -2.0),
        Vec2::new(-1.5, 0.9),
    ] {
        let u = frame.whiten(offset);
        let via_kernel = <Gaussian as Kernel<f64>>::eval_screen(&prepared, u);
        let via_conic = (-0.5 * frame.conic.quad_form(offset)).exp();
        assert!(
            (via_kernel - via_conic).abs() < 1e-14,
            "whitening must reproduce the conic quadratic form at {offset:?}"
        );
    }
    let step = frame.whiten_step_x();
    let base = frame.whiten(Vec2::new(0.0, 0.0));
    let one = frame.whiten(Vec2::new(1.0, 0.0));
    assert!(
        (one - (base + step)).length() < 1e-15,
        "the whitened offset must advance by two adds per pixel along a scanline"
    );
    assert!(ScreenFrame::from_covariance(Sym2::new(1.0, 1.0, 1.0)).is_none());
    assert!(ScreenFrame::from_covariance(Sym2::new(0.0, 0.0, 0.0)).is_none());
    assert!(ScreenFrame::from_covariance(Sym2::new(f64::NAN, 0.0, 1.0)).is_none());
    assert!(ScreenFrame::from_covariance(Sym2::new(-1.0, 0.0, 1.0)).is_none());

    let indefinite = Sym2::<f64>::new(1.0, 2.0, 1.0);
    assert!(
        indefinite.determinant() < 0.0,
        "det = 1 - 4 = -3, so this matrix is invertible as a matrix but is not a \
         covariance"
    );
    assert!(
        indefinite.inverse().is_none(),
        "S10 spells the guard as the one-sided !(det > eps) so that a sign slip \
         cannot survive it. An |det| > 0 guard returns a mirrored, indefinite \
         quadratic form here and calls it a conic; the shipped guard refuses it, \
         which puts Sym2::inverse in agreement with Sym2::cholesky, which \
         already tests !(xx > 0) and !(cc > 0) one-sidedly"
    );
    assert!(
        indefinite.cholesky().is_none(),
        "and it is not positive definite, so it is not a covariance"
    );
    assert!(
        ScreenFrame::from_covariance(indefinite).is_none(),
        "from_covariance is now refused TWICE over - by the inverse and by the \
         Cholesky. The Cholesky remains the stronger of the two, so this \
         particular input cannot distinguish a mutation to the inverse guard; \
         sym2_inverse_sign_and_swap_pattern in tests/numerics.rs tests the \
         inverse on its own, which is where it is reachable as a public function"
    );

    let negative_definite = Sym2::<f64>::new(-1.0, 0.0, -1.0);
    assert!((negative_definite.determinant() - 1.0).abs() < 1e-15);
    assert!(negative_definite.cholesky().is_none());
    assert!(ScreenFrame::from_covariance(negative_definite).is_none());

    for (xx, xy, yy) in [(1.0f64, 0.0, 1.0), (4.0, 0.7, 2.5), (100.0, 30.0, 9.1)] {
        let cov = Sym2::<f64>::new(xx, xy, yy);
        assert!(cov.cholesky().is_some() && cov.inverse().is_some());
        assert!(ScreenFrame::from_covariance(cov).is_some());
    }
}

#[test]
fn kernel_description_reports_the_compositing_class() {
    let d = describe::<f64, Gaussian>();
    assert_eq!(d.name, "gaussian");
    assert_eq!(d.class, CompositeClass::Over);
    assert!(!d.signed);
    assert!(d.order_dependent);
    assert!(d.radial);
    assert!(
        d.projectable,
        "SplatRaster checks HAS_SCREEN_MARGINAL once per render rather than \
         silently marginalising something with no marginal"
    );
    assert_eq!(d.id, <Gaussian as Kernel<f64>>::ID);
    assert_eq!(d.id, KernelFamily::Gaussian.id());
    assert_eq!(<Gaussian as Kernel<f64>>::NAME, KernelFamily::Gaussian.name());
    assert!(!KernelFamily::Gaussian.is_signed());
    assert_eq!(describe::<f32, Gaussian>().id, d.id);
}

#[test]
fn ray_frame_carries_lambda_and_whitening() {
    let f = ray_frame();
    assert_eq!(f.direction, Vec3::new(0.0, 0.0, 1.0));
    let v = Vec3::new(0.3, -0.2, 0.5);
    assert!(
        (f.whiten.apply(v).length_squared() - f.lambda.quad_form(v)).abs() < 1e-10,
        "the whitening factor must reproduce the Mahalanobis form"
    );
    let identity = RayFrame::new(
        Sym3::<f64>::new(1.0, 0.0, 0.0, 1.0, 0.0, 1.0),
        LowerTri3::<f64>::identity(),
        Vec3::new(0.0, 0.0, 1.0),
    );
    assert_eq!(identity.whiten.apply(v), v);
}

#[test]
fn the_normalised_kernel_differs_from_gaussian_only_in_its_blend() {
    let p = GaussianParams::default();
    let screen = screen_frame();
    let plain_screen = <Gaussian as Kernel<f64>>::prepare_screen(&p, &screen);
    let normalised_screen = <GaussianNormalised as Kernel<f64>>::prepare_screen(&p, &screen);
    let ray = ray_frame();
    let plain_ray = <Gaussian as Kernel<f64>>::prepare_ray(&p, &ray);
    let normalised_ray = <GaussianNormalised as Kernel<f64>>::prepare_ray(&p, &ray);

    for i in 0..48 {
        for j in 0..48 {
            let u = Vec2::new(
                -5.0 + 10.0 * (i as f64) / 47.0,
                -5.0 + 10.0 * (j as f64) / 47.0,
            );
            let plain = <Gaussian as Kernel<f64>>::eval_screen(&plain_screen, u);
            let normalised =
                <GaussianNormalised as Kernel<f64>>::eval_screen(&normalised_screen, u);
            assert_eq!(
                plain.to_bits(),
                normalised.to_bits(),
                "eval_screen must be BIT-IDENTICAL at {u:?}: the entire P2 \
                 measurement rests on the two arms differing in the blend rule \
                 and in nothing else"
            );
        }
    }
    for i in 0..12 {
        for j in 0..12 {
            for k in 0..12 {
                let u = Vec3::new(
                    -4.0 + 8.0 * (i as f64) / 11.0,
                    -4.0 + 8.0 * (j as f64) / 11.0,
                    -4.0 + 8.0 * (k as f64) / 11.0,
                );
                let plain = <Gaussian as Kernel<f64>>::eval_ray(&plain_ray, u);
                let normalised =
                    <GaussianNormalised as Kernel<f64>>::eval_ray(&normalised_ray, u);
                assert_eq!(plain.to_bits(), normalised.to_bits(), "eval_ray at {u:?}");
            }
        }
    }
    for peak in [1.0f64, 0.7, 0.25, 0.05, 0.004, 0.001] {
        let plain = <Gaussian as Kernel<f64>>::support_screen(&plain_screen, peak, EPS_ALPHA);
        let normalised = <GaussianNormalised as Kernel<f64>>::support_screen(
            &normalised_screen,
            peak,
            EPS_ALPHA,
        );
        assert_eq!(plain.to_bits(), normalised.to_bits(), "support_screen at {peak}");
        let plain_ray_r = <Gaussian as Kernel<f64>>::support_ray(&plain_ray, peak, EPS_ALPHA);
        let normalised_ray_r =
            <GaussianNormalised as Kernel<f64>>::support_ray(&normalised_ray, peak, EPS_ALPHA);
        assert_eq!(plain_ray_r.to_bits(), normalised_ray_r.to_bits());
    }
    for variance in [0.0f64, 0.3, 1.7] {
        let cov = Sym2::<f64>::new(4.0, 0.7, 2.5);
        let (plain_p, plain_cov, plain_gain) =
            <Gaussian as Kernel<f64>>::prefilter(&p, cov, variance);
        let (normalised_p, normalised_cov, normalised_gain) =
            <GaussianNormalised as Kernel<f64>>::prefilter(&p, cov, variance);
        assert_eq!(plain_p, normalised_p);
        assert_eq!(plain_cov, normalised_cov);
        assert_eq!(plain_gain.to_bits(), normalised_gain.to_bits());
    }

    let plain = describe::<f64, Gaussian>();
    let normalised = describe::<f64, GaussianNormalised>();
    assert_eq!(normalised.name, "gaussian-normalised");
    assert_eq!(normalised.class, CompositeClass::Normalised);
    assert_ne!(normalised.class, plain.class);
    assert!(!normalised.order_dependent);
    assert!(
        plain.order_dependent,
        "the two descriptions must differ on ORDER_DEPENDENT: that declaration \
         is what lets bin_and_sort skip the depth sort, and a flipped constant \
         is invisible in the pixels"
    );
    assert!(!normalised.signed);
    assert!(normalised.radial && normalised.projectable);
    assert_eq!(normalised.id, KernelId(1));
    assert_ne!(normalised.id, plain.id, "two kernels, two ids");
    assert_eq!(normalised.id, KernelFamily::GaussianNormalised.id());
    assert_eq!(
        <GaussianNormalised as Kernel<f64>>::NAME,
        KernelFamily::GaussianNormalised.name()
    );
    assert!(!KernelFamily::GaussianNormalised.is_signed());
    assert_eq!(describe::<f32, GaussianNormalised>().id, normalised.id);
    assert_eq!(
        KernelFamily::BLEND_PAIR.len(),
        2,
        "the blend axis is a PAIR of kernels that differ only in their blend \
         rule; the atom-curve families live on a different axis and must never \
         be swept as blends"
    );
    assert_eq!(
        KernelFamily::ALL.len(),
        18,
        "eight unmodulated entries over four kernels, plus ten modulated \
         entries over four more. L-56 widened this from 8"
    );
}


const D_EPS: f64 = 1.0e-5;

fn screen_derivative_probe<K: Kernel<f64>>(shape: &CurveShape) -> (bool, usize, f64, f64) {
    let p = K::params_from(shape);
    let screen = K::prepare_screen(&p, &screen_frame());
    let mut probed = 0usize;
    let mut worst = 0.0f64;
    let mut largest = 0.0f64;
    let mut declared = false;
    for i in 0..21 {
        for j in 0..21 {
            let u = Vec2::new(
                -3.0 + 6.0 * (i as f64) / 20.0,
                -3.0 + 6.0 * (j as f64) / 20.0,
            );
            let analytic = match K::d_eval_screen(&screen, u) {
                Some(g) => g,
                None => return (false, 0, 0.0, 0.0),
            };
            declared = true;
            let dx = (K::eval_screen(&screen, Vec2::new(u.x + D_EPS, u.y))
                - K::eval_screen(&screen, Vec2::new(u.x - D_EPS, u.y)))
                / (2.0 * D_EPS);
            let dy = (K::eval_screen(&screen, Vec2::new(u.x, u.y + D_EPS))
                - K::eval_screen(&screen, Vec2::new(u.x, u.y - D_EPS)))
                / (2.0 * D_EPS);
            worst = worst
                .max((analytic.x - dx).abs())
                .max((analytic.y - dy).abs());
            largest = largest.max(analytic.x.abs()).max(analytic.y.abs());
            probed += 1;
        }
    }
    (declared, probed, worst, largest)
}

#[test]
fn a_declared_screen_derivative_agrees_with_a_central_difference_of_its_own_evaluation() {
    let mut checked = 0usize;
    for family in KernelFamily::ALL {
        let shape = family.shape();
        let (declared, probed, worst, largest) =
            for_kernel!(family, screen_derivative_probe, &shape);
        if !declared {
            continue;
        }
        assert!(
            probed >= 400,
            "{}: a derivative claimed over {probed} samples is not measured",
            family.name()
        );
        assert!(
            largest > 0.1,
            "{}: the derivative peaks at {largest} on the probe grid, so \
             agreeing with a central difference of it proves nothing",
            family.name()
        );
        assert!(
            worst < 1.0e-8,
            "{}: the closed-form screen derivative disagrees with a central \
             difference of its own eval_screen by {worst} at eps {D_EPS}. One \
             of the two is wrong, and the finite difference is the one with no \
             chain rule in it.",
            family.name()
        );
        checked += 1;
    }
    assert!(
        checked > 0,
        "no kernel declared a screen derivative, so this check measured nothing"
    );
}

#[test]
fn exactly_the_gaussian_pair_answers_with_a_closed_form_and_every_other_kernel_says_it_has_none() {
    let mut with: Vec<&'static str> = Vec::new();
    let mut without: Vec<&'static str> = Vec::new();
    for family in KernelFamily::ALL {
        let shape = family.shape();
        let (declared, _, _, _) = for_kernel!(family, screen_derivative_probe, &shape);
        if declared {
            with.push(family.name());
        } else {
            without.push(family.name());
        }
    }
    with.sort();
    with.dedup();
    without.sort();
    without.dedup();
    assert_eq!(
        with,
        vec!["gaussian", "gaussian-normalised"],
        "L-146 adds the closed form for the gaussian pair ONLY. A kernel that \
         starts answering must be added here in the same commit, because the \
         point of the Option is that absence stays VISIBLE: a default returning \
         a plausible vector would make every other kernel silently wrong."
    );
    assert!(
        !without.is_empty(),
        "every kernel answered, so the refusing default is untested and a \
         future kernel could inherit a wrong answer unnoticed"
    );
    assert!(!with.is_empty(), "no kernel answered at all");
}
