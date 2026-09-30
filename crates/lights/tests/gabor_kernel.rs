use lights::blend::{BlendMode, CompositeClass};
use lights::camera::{Camera, CameraSpec};
use lights::image::{FloatImage, Framebuffer};
use lights::kernel::compact::CompactBump;
use lights::kernel::frame::{RayFrame, ScreenFrame};
use lights::kernel::gabor::{GaborColour, GaborDensity, GaborParams};
use lights::kernel::gaussian::{Gaussian, GaussianNormalised};
use lights::kernel::hermite::{GaussHermiteColour, GaussHermiteDensity};
use lights::kernel::supergaussian::SuperGaussian;
use lights::kernel::{CurveShape, Kernel, KernelId, describe};
use lights::method::{ImageFormation, RenderMethod, create_strategy};
use lights::numerics::quaternion::Quat;
use lights::numerics::symmetric::{LowerTri3, Sym2, Sym3};
use lights::numerics::vector::{Rgb, Vec2, Vec3};
use lights::probe::{NoProbe, ProbeSink};
use lights::scene::prepare::PreparedScene;
use lights::scene::{Atom, Scene};
use lights::settings::{PixelConvention, SharedSettings};

const EPS_ALPHA: f64 = 1.0 / 255.0;
const VIEWPORT: (u32, u32) = (48, 40);
const Z0: f64 = 64.0;
const DZ: f64 = 1.0 / 64.0;

fn screen_frame() -> ScreenFrame<f64> {
    ScreenFrame::from_covariance(Sym2::new(4.0, 0.7, 2.5)).expect("SPD")
}

fn isotropic_frame() -> ScreenFrame<f64> {
    ScreenFrame::from_covariance(Sym2::new(1.0, 0.0, 1.0)).expect("SPD")
}

fn ray_frame() -> RayFrame<f64> {
    let sigma = Sym3::<f64>::new(0.04, 0.0, 0.0, 0.01, 0.0, 0.0025);
    RayFrame::new(
        sigma.inverse_spd().expect("SPD"),
        sigma.cholesky().expect("SPD").inverse(),
        Vec3::new(0.0, 0.0, 1.0),
    )
}

fn identity_ray_frame() -> RayFrame<f64> {
    RayFrame::new(
        Sym3::<f64>::new(1.0, 0.0, 0.0, 1.0, 0.0, 1.0),
        LowerTri3::<f64>::identity(),
        Vec3::new(0.0, 0.0, 1.0),
    )
}

fn camera_spec() -> CameraSpec {
    CameraSpec::look_at(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
        45.0,
        VIEWPORT.0,
        VIEWPORT.1,
    )
    .orthographic(VIEWPORT.1 as f64 * 0.5)
    .with_convention(PixelConvention::CentreSample)
}

fn flat_atom(
    index: usize,
    centre: Vec2<f64>,
    sigma: Vec2<f64>,
    angle: f64,
    opacity: f64,
    colour: Rgb<f64>,
) -> Atom<f64> {
    let rotation = match Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), angle) {
        Some(q) => q,
        None => Quat::identity(),
    };
    Atom::new(
        Vec3::new(
            centre.x - VIEWPORT.0 as f64 * 0.5,
            centre.y - VIEWPORT.1 as f64 * 0.5,
            Z0 - index as f64 * DZ,
        ),
        rotation,
        Vec3::new(sigma.x, sigma.y, sigma.x.min(sigma.y)),
        opacity,
        colour,
    )
}

fn scene_of(atoms: &[Atom<f64>], settings: &SharedSettings<f64>) -> Scene<f64> {
    let mut scene = Scene::<f64>::new();
    for a in atoms {
        scene.push(*a, settings.scale_floor).expect("push");
    }
    scene
}

fn render_with<K: Kernel<f64>>(
    scene: &Scene<f64>,
    params: &[K::Params],
    settings: &SharedSettings<f64>,
    method: &RenderMethod,
) -> FloatImage<f64> {
    let camera = Camera::<f64>::from_spec(&camera_spec(), settings.near_plane).expect("camera");
    let prepared = PreparedScene::<f64, K>::build(scene, params, settings).expect("prepare");
    let mut strategy = create_strategy::<f64, K>(method, settings).expect("strategy");
    strategy.build(&prepared).expect("build");
    strategy.bind(&prepared, &camera).expect("bind");
    let mut framebuffer =
        Framebuffer::<f64, K::Blend>::new(camera.width, camera.height, method.tile())
            .expect("framebuffer");
    let mut sink = ProbeSink::<f64>::counters_only(camera.width, camera.height);
    strategy
        .render::<NoProbe>(&prepared, &mut framebuffer, &mut sink)
        .expect("render");
    framebuffer.resolve_float(settings.background)
}

fn methods() -> Vec<RenderMethod> {
    RenderMethod::implemented()
}

fn stack() -> Vec<Atom<f64>> {
    vec![
        flat_atom(
            0,
            Vec2::new(18.0, 17.0),
            Vec2::new(5.0, 2.5),
            0.4,
            0.55,
            Rgb::new(0.9, 0.35, 0.2),
        ),
        flat_atom(
            1,
            Vec2::new(26.0, 22.0),
            Vec2::new(3.0, 3.0),
            0.0,
            0.7,
            Rgb::new(0.2, 0.8, 0.45),
        ),
        flat_atom(
            2,
            Vec2::new(22.0, 25.0),
            Vec2::new(6.0, 1.8),
            -0.9,
            0.4,
            Rgb::new(0.3, 0.4, 0.95),
        ),
    ]
}

fn max_channel_gap(a: &FloatImage<f64>, b: &FloatImage<f64>) -> f64 {
    assert_eq!((a.width, a.height), (b.width, b.height));
    let mut worst = 0.0f64;
    for (p, q) in a.pixels.iter().zip(b.pixels.iter()) {
        worst = worst
            .max((p.x - q.x).abs())
            .max((p.y - q.y).abs())
            .max((p.z - q.z).abs());
    }
    worst
}

fn min_channel(image: &FloatImage<f64>) -> f64 {
    let mut lowest = f64::INFINITY;
    for p in image.pixels.iter() {
        lowest = lowest.min(p.x).min(p.y).min(p.z);
    }
    lowest
}

fn assert_bit_identical(a: &FloatImage<f64>, b: &FloatImage<f64>, what: &str) {
    assert_eq!((a.width, a.height), (b.width, b.height), "{what}");
    for (i, (p, q)) in a.pixels.iter().zip(b.pixels.iter()).enumerate() {
        assert_eq!(
            (p.x.to_bits(), p.y.to_bits(), p.z.to_bits()),
            (q.x.to_bits(), q.y.to_bits(), q.z.to_bits()),
            "{what}: pixel {i} differs. A zero-frequency Gabor whose carrier is \
             cos(0) = 1 must reduce to the gaussian BIT FOR BIT, not to within a \
             tolerance: that reduction is the control the whole oscillatory arm \
             is read against, and a tolerance would hide exactly the drift it \
             exists to catch"
        );
    }
}

#[test]
fn a_zero_frequency_gabor_is_the_gaussian_bit_for_bit() {
    let flat = CurveShape::gabor(0.0, 0.0, 0.0);
    assert!(flat.is_zero_frequency());
    assert!(CurveShape::gaussian().is_zero_frequency());

    let gaussian_params = <Gaussian as Kernel<f64>>::params_from(&flat);
    let gabor_params = <GaborColour as Kernel<f64>>::params_from(&flat);
    assert_eq!(gabor_params, GaborParams::<f64>::default());

    for frame in [screen_frame(), isotropic_frame()] {
        let plain = <Gaussian as Kernel<f64>>::prepare_screen(&gaussian_params, &frame);
        let colour = <GaborColour as Kernel<f64>>::prepare_screen(&gabor_params, &frame);
        let density = <GaborDensity as Kernel<f64>>::prepare_screen(&gabor_params, &frame);
        assert_eq!(colour.carrier, Vec2::zero());
        assert_eq!(colour.phase, 0.0);

        for i in 0..64 {
            for j in 0..64 {
                let u = Vec2::new(
                    -6.0 + 12.0 * (i as f64) / 63.0,
                    -6.0 + 12.0 * (j as f64) / 63.0,
                );
                let reference = <Gaussian as Kernel<f64>>::eval_screen(&plain, u);
                assert_eq!(
                    <GaborColour as Kernel<f64>>::eval_screen(&colour, u).to_bits(),
                    reference.to_bits(),
                    "gabor-colour eval_screen at {u:?}"
                );
                assert_eq!(
                    <GaborDensity as Kernel<f64>>::eval_screen(&density, u).to_bits(),
                    reference.to_bits(),
                    "gabor-density eval_screen at {u:?}: the carrier is cos(0) and \
                     multiplying by exactly 1.0 is the identity in IEEE 754"
                );
                assert_eq!(
                    <GaborColour as Kernel<f64>>::modulate_screen(&colour, u).to_bits(),
                    1.0f64.to_bits(),
                    "the colour placement's modulation must be exactly one at zero \
                     frequency and zero phase"
                );
            }
        }
        for peak in [1.0f64, 0.7, 0.25, 0.05, 0.004, 0.001] {
            let reference =
                <Gaussian as Kernel<f64>>::support_screen(&plain, peak, EPS_ALPHA);
            assert_eq!(
                <GaborColour as Kernel<f64>>::support_screen(&colour, peak, EPS_ALPHA)
                    .to_bits(),
                reference.to_bits(),
                "gabor-colour support_screen at {peak}"
            );
            assert_eq!(
                <GaborDensity as Kernel<f64>>::support_screen(&density, peak, EPS_ALPHA)
                    .to_bits(),
                reference.to_bits(),
                "gabor-density support_screen at {peak}"
            );
        }
    }

    for frame in [ray_frame(), identity_ray_frame()] {
        let plain = <Gaussian as Kernel<f64>>::prepare_ray(&gaussian_params, &frame);
        let colour = <GaborColour as Kernel<f64>>::prepare_ray(&gabor_params, &frame);
        let density = <GaborDensity as Kernel<f64>>::prepare_ray(&gabor_params, &frame);
        assert_eq!(colour.carrier, Vec3::zero());
        for i in 0..16 {
            for j in 0..16 {
                for k in 0..16 {
                    let u = Vec3::new(
                        -5.0 + 10.0 * (i as f64) / 15.0,
                        -5.0 + 10.0 * (j as f64) / 15.0,
                        -5.0 + 10.0 * (k as f64) / 15.0,
                    );
                    let reference = <Gaussian as Kernel<f64>>::eval_ray(&plain, u);
                    assert_eq!(
                        <GaborColour as Kernel<f64>>::eval_ray(&colour, u).to_bits(),
                        reference.to_bits(),
                        "gabor-colour eval_ray at {u:?}"
                    );
                    assert_eq!(
                        <GaborDensity as Kernel<f64>>::eval_ray(&density, u).to_bits(),
                        reference.to_bits(),
                        "gabor-density eval_ray at {u:?}"
                    );
                    assert_eq!(
                        <GaborColour as Kernel<f64>>::modulate_ray(&colour, u).to_bits(),
                        1.0f64.to_bits()
                    );
                }
            }
        }
    }
}

#[test]
fn a_zero_frequency_gabor_renders_the_gaussian_image_bit_for_bit() {
    let settings = SharedSettings::<f64>::contract();
    let atoms = stack();
    let scene = scene_of(&atoms, &settings);
    let flat = CurveShape::gabor(0.0, 0.0, 0.0);
    let gaussian_params = vec![<Gaussian as Kernel<f64>>::params_from(&flat); atoms.len()];
    let gabor_params = vec![<GaborColour as Kernel<f64>>::params_from(&flat); atoms.len()];

    let mut covered = 0usize;
    for method in methods() {
        let reference = render_with::<Gaussian>(&scene, &gaussian_params, &settings, &method);
        let arm = render_with::<GaborColour>(&scene, &gabor_params, &settings, &method);
        assert_bit_identical(
            &reference,
            &arm,
            &format!("{:?} zero-frequency gabor-colour", method.id()),
        );
        let (peak, _, _) = reference.max_abs();
        assert!(
            peak > 0.1,
            "{:?}: the control image is nearly empty ({peak}), so a bit-for-bit \
             agreement on it would prove nothing",
            method.id()
        );
        covered += 1;
    }
    assert_eq!(
        covered,
        RenderMethod::implemented().len(),
        "the control must run on every method built into this binary"
    );
    assert!(covered >= 1, "no method was built in, so this case asserted nothing");
}

#[test]
fn the_carrier_is_a_plane_wave_in_screen_space() {
    let frequency = 0.13f64;
    let orientation = 0.7f64;
    let phase = 0.4f64;
    let shape = CurveShape::gabor(frequency, orientation, phase);
    let params = <GaborColour as Kernel<f64>>::params_from(&shape);
    let (sine, cosine) = orientation.sin_cos();
    let direction = Vec2::new(cosine, sine);

    let mut moved = 0usize;
    for frame in [screen_frame(), isotropic_frame()] {
        let prepared = <GaborColour as Kernel<f64>>::prepare_screen(&params, &frame);
        for i in 0..24 {
            for j in 0..24 {
                let delta = Vec2::new(
                    -6.0 + 12.0 * (i as f64) / 23.0,
                    -6.0 + 12.0 * (j as f64) / 23.0,
                );
                let u = frame.whiten(delta);
                let got = <GaborColour as Kernel<f64>>::modulate_screen(&prepared, u);
                let (_, want) =
                    (core::f64::consts::TAU * frequency * delta.dot(direction) + phase).sin_cos();
                assert!(
                    (got - want).abs() < 1.0e-12,
                    "the carrier must be a plane wave in SCREEN space at {delta:?}: \
                     got {got}, want {want}. prepare_screen lifts the direction with \
                     L^T because delta = L u; lifting it with L^-1 instead makes the \
                     wavelength scale with the inverse of the atom's anisotropy, which \
                     is invisible on an isotropic frame and wrong on every elongated one"
                );
                if (got - 1.0).abs() > 1.0e-3 {
                    moved += 1;
                }
            }
        }
    }
    assert!(
        moved > 400,
        "the carrier only left 1.0 at {moved} of 1152 samples, so this test may be \
         comparing two constants"
    );
}

#[test]
fn the_phase_shifts_the_carrier_and_the_zero_phase_case_is_the_identity() {
    let frame = screen_frame();
    let cases = [
        (0.0f64, 1.0f64),
        (core::f64::consts::FRAC_PI_2, 0.0),
        (core::f64::consts::PI, -1.0),
    ];
    for (phase, want) in cases {
        let shape = CurveShape::gabor(0.11, 0.35, phase);
        let params = <GaborColour as Kernel<f64>>::params_from(&shape);
        let prepared = <GaborColour as Kernel<f64>>::prepare_screen(&params, &frame);
        let got = <GaborColour as Kernel<f64>>::modulate_screen(&prepared, Vec2::zero());
        assert!(
            (got - want).abs() < 1.0e-12,
            "at the atom centre the carrier is cos(phase); with phase {phase} it must \
             be {want} and it is {got}. A kernel that drops the phase term reads 1.0 \
             here for every phase"
        );
    }

    let carried = CurveShape::gabor(0.11, 0.35, core::f64::consts::PI);
    let dropped = CurveShape::gabor(0.11, 0.35, 0.0);
    let a = <GaborDensity as Kernel<f64>>::prepare_screen(
        &<GaborDensity as Kernel<f64>>::params_from(&carried),
        &frame,
    );
    let b = <GaborDensity as Kernel<f64>>::prepare_screen(
        &<GaborDensity as Kernel<f64>>::params_from(&dropped),
        &frame,
    );
    let u = Vec2::new(0.3, -0.2);
    assert!(
        (<GaborDensity as Kernel<f64>>::eval_screen(&a, u)
            + <GaborDensity as Kernel<f64>>::eval_screen(&b, u))
        .abs()
            < 1.0e-12,
        "a half-turn of phase must negate the density placement's response"
    );
}

#[test]
fn both_sign_placements_drive_a_composited_channel_negative() {
    let settings = SharedSettings::<f64>::contract();
    let atoms = vec![flat_atom(
        0,
        Vec2::new(24.0, 20.0),
        Vec2::new(5.0, 5.0),
        0.0,
        0.75,
        Rgb::new(0.9, 0.6, 0.3),
    )];
    let scene = scene_of(&atoms, &settings);
    let shape = CurveShape::gabor(0.09, 0.3, 0.0);
    let params = vec![<GaborColour as Kernel<f64>>::params_from(&shape); 1];

    let mut checked = 0usize;
    for method in methods() {
        let colour = render_with::<GaborColour>(&scene, &params, &settings, &method);
        let density = render_with::<GaborDensity>(&scene, &params, &settings, &method);
        assert!(
            min_channel(&colour) < -1.0e-3,
            "{:?}: the colour placement never drove a channel negative (lowest \
             {}). Either the modulation never reached shade_pixel or the carrier \
             never changed sign, and in both cases nothing signed was measured",
            method.id(),
            min_channel(&colour)
        );
        assert!(
            min_channel(&density) < -1.0e-3,
            "{:?}: the density placement never drove a channel negative (lowest {})",
            method.id(),
            min_channel(&density)
        );
        let plain = render_with::<Gaussian>(
            &scene,
            &vec![<Gaussian as Kernel<f64>>::params_from(&shape); 1],
            &settings,
            &method,
        );
        assert!(
            min_channel(&plain) >= 0.0,
            "{:?}: the unsigned control went negative, so the negativity above is \
             not evidence of anything",
            method.id()
        );
        checked += 1;
    }
    assert_eq!(
        checked,
        RenderMethod::implemented().len(),
        "the control must run on every method built into this binary"
    );
    assert!(checked >= 1, "no method was built in, so this case asserted nothing");
}

fn settings_with_cutoff(cutoff: f64) -> SharedSettings<f64> {
    let mut s = SharedSettings::<f64>::contract();
    s.blend.alpha_cutoff = cutoff;
    s
}

#[test]
fn both_sign_placements_represent_the_same_single_atom() {
    let atoms = vec![flat_atom(
        0,
        Vec2::new(22.0, 21.0),
        Vec2::new(4.5, 3.0),
        0.5,
        0.6,
        Rgb::new(0.8, 0.5, 0.35),
    )];
    let shape = CurveShape::gabor(0.08, 0.9, 0.25);
    let params = vec![<GaborColour as Kernel<f64>>::params_from(&shape); 1];

    for method in methods() {
        let production = settings_with_cutoff(EPS_ALPHA);
        let scene = scene_of(&atoms, &production);
        let colour = render_with::<GaborColour>(&scene, &params, &production, &method);
        let density = render_with::<GaborDensity>(&scene, &params, &production, &method);
        let gap = max_channel_gap(&colour, &density);
        assert!(
            gap <= 2.0 * EPS_ALPHA,
            "{:?}: the two sign placements disagree by {gap} on a single atom over \
             a black ground, where Over resolves to colour*carrier*alpha and \
             Additive to colour*(opacity*envelope*carrier) and the two are the same \
             number. They may differ only in the thin ring where Over drops a sample \
             under alpha_cutoff. A larger gap means the two arms are representing \
             different functions and the head-to-head comparison is meaningless",
            method.id()
        );
        let (peak, _, _) = colour.max_abs();
        assert!(peak > 0.05, "{:?}: both arms are nearly empty", method.id());

        let open = settings_with_cutoff(1.0e-9);
        let open_scene = scene_of(&atoms, &open);
        let a = render_with::<GaborColour>(&open_scene, &params, &open, &method);
        let b = render_with::<GaborDensity>(&open_scene, &params, &open, &method);
        let converged = max_channel_gap(&a, &b);
        assert!(
            converged < 1.0e-8,
            "{:?}: with the alpha cutoff opened to 1e-9 the two placements still \
             differ by {converged}. For a Gabor - unlike Gauss-Hermite - they must \
             converge, because the carrier is bounded by 1 and the support is the \
             envelope's own radius, so there is no annulus in which the modulated \
             colour survives a dropped alpha. A residue here is not the cutoff, it \
             is a real difference between the two arms",
            method.id()
        );
    }
}

#[test]
#[cfg(feature = "method-splat")]
fn permuting_a_colour_signed_set_changes_the_image_and_a_density_signed_set_does_not() {
    let settings = SharedSettings::<f64>::contract();
    let shape = CurveShape::gabor(0.07, 0.6, 0.15);
    let method = RenderMethod::splat_contract();

    let forward: Vec<Atom<f64>> = stack();
    let mut reversed_source = stack();
    reversed_source.reverse();
    let reversed: Vec<Atom<f64>> = reversed_source
        .iter()
        .enumerate()
        .map(|(i, a)| {
            let mut b = *a;
            b.mean.z = Z0 - i as f64 * DZ;
            b
        })
        .collect();

    let params_colour =
        vec![<GaborColour as Kernel<f64>>::params_from(&shape); forward.len()];
    let a = render_with::<GaborColour>(
        &scene_of(&forward, &settings),
        &params_colour,
        &settings,
        &method,
    );
    let b = render_with::<GaborColour>(
        &scene_of(&reversed, &settings),
        &params_colour,
        &settings,
        &method,
    );
    assert!(
        max_channel_gap(&a, &b) > EPS_ALPHA,
        "the colour placement composites through Over, which DECLARES itself \
         order-dependent; a permutation that leaves the image alone means the \
         declaration is not describing this path"
    );

    let params_density =
        vec![<GaborDensity as Kernel<f64>>::params_from(&shape); forward.len()];
    let c = render_with::<GaborDensity>(
        &scene_of(&forward, &settings),
        &params_density,
        &settings,
        &method,
    );
    let d = render_with::<GaborDensity>(
        &scene_of(&reversed, &settings),
        &params_density,
        &settings,
        &method,
    );
    let drift = max_channel_gap(&c, &d);
    assert!(
        drift < 1.0e-12,
        "the density placement composites through Additive, which is \
         order-independent up to the rounding of a reassociated sum; a permutation \
         moved it by {drift}"
    );
    let (peak, _, _) = c.max_abs();
    assert!(peak > 0.05, "the density arm is nearly empty");
}

#[test]
fn the_gabor_support_bounds_the_true_signed_response() {
    let frames = [screen_frame(), isotropic_frame()];
    for frequency in [0.0f64, 0.05, 0.2, 0.6] {
        let shape = CurveShape::gabor(frequency, 0.8, 0.3);
        let params = <GaborColour as Kernel<f64>>::params_from(&shape);
        for frame in frames {
            let colour = <GaborColour as Kernel<f64>>::prepare_screen(&params, &frame);
            let density = <GaborDensity as Kernel<f64>>::prepare_screen(&params, &frame);
            for peak in [1.0f64, 0.7, 0.25, 0.05] {
                let r = <GaborColour as Kernel<f64>>::support_screen(&colour, peak, EPS_ALPHA);
                if r <= 0.0 {
                    continue;
                }
                for ring in 1..=6 {
                    let radius = r * (1.0 + 0.05 * ring as f64);
                    for spoke in 0..64 {
                        let theta = core::f64::consts::TAU * (spoke as f64) / 64.0;
                        let u = Vec2::new(radius * theta.cos(), radius * theta.sin());
                        let signed = <GaborColour as Kernel<f64>>::eval_screen(&colour, u)
                            * <GaborColour as Kernel<f64>>::modulate_screen(&colour, u);
                        assert!(
                            signed.abs() * peak <= EPS_ALPHA * (1.0 + 1.0e-4),
                            "gabor-colour peak {peak} frequency {frequency} radius \
                             {radius}: the modulated response {signed} still exceeds \
                             the cutoff. The support must bound the CONTRIBUTION, \
                             which is envelope times carrier, not the alpha alone"
                        );
                        let full = <GaborDensity as Kernel<f64>>::eval_screen(&density, u);
                        assert!(full.abs() * peak <= EPS_ALPHA * (1.0 + 1.0e-4));
                    }
                }
            }
        }
    }
}

#[test]
fn the_unmodulated_kernels_declare_themselves_unmodulated() {
    assert!(!<Gaussian as Kernel<f64>>::MODULATED);
    assert!(!<Gaussian as Kernel<f32>>::MODULATED);
    assert!(!<GaussianNormalised as Kernel<f64>>::MODULATED);
    assert!(!<SuperGaussian as Kernel<f64>>::MODULATED);
    assert!(!<CompactBump as Kernel<f64>>::MODULATED);
    assert!(!<GaborDensity as Kernel<f64>>::MODULATED);
    assert!(!<GaussHermiteDensity as Kernel<f64>>::MODULATED);

    assert!(<GaborColour as Kernel<f64>>::MODULATED);
    assert!(<GaborColour as Kernel<f32>>::MODULATED);
    assert!(<GaussHermiteColour as Kernel<f64>>::MODULATED);

    for d in [
        describe::<f64, Gaussian>(),
        describe::<f64, GaussianNormalised>(),
        describe::<f64, SuperGaussian>(),
        describe::<f64, CompactBump>(),
    ] {
        assert!(
            !d.modulated,
            "{} declares itself modulated. The const gate is what lets the six \
             kernels that came before keep a hot loop with no branch and no call \
             in it, and a const nothing checks is a const nobody can trust",
            d.name
        );
    }

    let colour = describe::<f64, GaborColour>();
    assert_eq!(colour.name, "gabor-colour");
    assert_eq!(colour.id, KernelId(4));
    assert_eq!(colour.class, CompositeClass::Over);
    assert!(!colour.signed, "the BLEND is unsigned: the sign is in the colour");
    assert!(colour.order_dependent);
    assert!(colour.modulated);
    assert!(!colour.radial);

    let density = describe::<f64, GaborDensity>();
    assert_eq!(density.name, "gabor-density");
    assert_eq!(density.id, KernelId(5));
    assert_eq!(density.class, CompositeClass::Additive);
    assert!(density.signed, "the density placement puts the sign in the weight");
    assert!(!density.order_dependent);
    assert!(!density.modulated);

    assert!(<<GaborColour as Kernel<f64>>::Blend as BlendMode<f64>>::ORDER_DEPENDENT);
    assert!(!<<GaborDensity as Kernel<f64>>::Blend as BlendMode<f64>>::ORDER_DEPENDENT);
    assert!(<<GaborDensity as Kernel<f64>>::Blend as BlendMode<f64>>::SIGNED);
}

#[test]
fn a_negative_weight_never_reaches_the_over_path() {
    let shape = CurveShape::gabor(0.09, 0.3, 0.0);
    let params = <GaborColour as Kernel<f64>>::params_from(&shape);
    let frame = screen_frame();
    let prepared = <GaborColour as Kernel<f64>>::prepare_screen(&params, &frame);
    let mut negative_modulations = 0usize;
    for i in 0..64 {
        for j in 0..64 {
            let u = Vec2::new(
                -4.0 + 8.0 * (i as f64) / 63.0,
                -4.0 + 8.0 * (j as f64) / 63.0,
            );
            let weight = <GaborColour as Kernel<f64>>::eval_screen(&prepared, u);
            assert!(
                weight >= 0.0,
                "the colour placement's eval_screen became {weight} at {u:?}. \
                 Over::step computes alpha = (opacity*weight).min(cap) and then \
                 DROPS anything below alpha_cutoff, so a negative weight is not \
                 negated there, it disappears, and transmittance *= (1 - alpha) \
                 would grow. The whole point of the colour placement is that this \
                 cannot happen"
            );
            if <GaborColour as Kernel<f64>>::modulate_screen(&prepared, u) < 0.0 {
                negative_modulations += 1;
            }
        }
    }
    assert!(
        negative_modulations > 100,
        "only {negative_modulations} of 4096 samples carried a negative \
         modulation, so the non-negativity above was not tested against anything"
    );
}

pub const BOX_QUADRATURE_INTERVALS: usize = 400;

fn sinc_pi(x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    let a = core::f64::consts::PI * x;
    a.sin() / a
}

fn box_average_of_the_sample(
    frame: &ScreenFrame<f64>,
    prepared: &lights::kernel::gabor::GaborScreen<f64>,
    centre: Vec2<f64>,
    intervals: usize,
) -> f64 {
    let n = intervals;
    let step = 1.0 / n as f64;
    let mut total = 0.0f64;
    for i in 0..n {
        let dx = -0.5 + (i as f64 + 0.5) * step;
        for j in 0..n {
            let dy = -0.5 + (j as f64 + 0.5) * step;
            let delta = Vec2::new(centre.x + dx, centre.y + dy);
            let u = frame.whiten(delta);
            total += <GaborColour as Kernel<f64>>::eval_screen(prepared, u)
                * <GaborColour as Kernel<f64>>::modulate_screen(prepared, u);
        }
    }
    total * step * step
}

pub const MEASURED_OVER_BRIGHTNESS_BEFORE: [(f64, f64); 6] = [
    (0.00, 1.000025),
    (0.05, 1.004134),
    (0.10, 1.016604),
    (0.20, 1.068716),
    (0.30, 1.164298),
    (0.40, 1.319823),
];

#[test]
fn the_prefiltered_carrier_now_matches_the_box_average_it_used_to_exceed() {
    let raw = Sym2::<f64>::new(9.0, 0.0, 9.0);
    let dilation = lights::settings::PIXEL_BOX_VARIANCE;
    let orientation = 0.0f64;

    let mut rows = 0usize;
    let mut worst = 0.0f64;
    for (frequency, before) in MEASURED_OVER_BRIGHTNESS_BEFORE {
        let shape = CurveShape::gabor(frequency, orientation, 0.0);
        let params = <GaborColour as Kernel<f64>>::params_from(&shape);
        assert_eq!(params.amplitude, 1.0, "an unfiltered carrier is unattenuated");

        let (filtered, cov, gain) =
            <GaborColour as Kernel<f64>>::prefilter(&params, raw, dilation);
        assert_eq!(filtered.frequency, params.frequency);
        assert_eq!(filtered.orientation, params.orientation);
        assert_eq!(filtered.phase, params.phase);
        assert!(
            filtered.amplitude <= 1.0 && filtered.amplitude > 0.0,
            "the box average of a carrier can only shrink its amplitude: {}",
            filtered.amplitude
        );

        let dilated = ScreenFrame::from_covariance(cov).expect("SPD");
        let prepared = <GaborColour as Kernel<f64>>::prepare_screen(&filtered, &dilated);
        let raw_frame = ScreenFrame::from_covariance(raw).expect("SPD");
        let raw_prepared = <GaborColour as Kernel<f64>>::prepare_screen(&params, &raw_frame);

        let centre = Vec2::new(0.0, 0.0);
        let u = dilated.whiten(centre);
        let value = gain
            * <GaborColour as Kernel<f64>>::eval_screen(&prepared, u)
            * <GaborColour as Kernel<f64>>::modulate_screen(&prepared, u);
        let truth =
            box_average_of_the_sample(&raw_frame, &raw_prepared, centre, BOX_QUADRATURE_INTERVALS);
        let ratio = value / truth;
        println!(
            "f {frequency:.2} cyc/px: ratio now {ratio:.6}, was {before:.6}, \
             amplitude {:.6}",
            filtered.amplitude
        );
        assert!(
            (ratio - 1.0).abs() < 2.5e-3,
            "at {frequency} cycles per pixel the prefiltered sample must agree \
             with the 400x400 box quadrature of the unfiltered one. It is off \
             by {}, and before the carrier attenuation landed it was off by {}",
            ratio - 1.0,
            before - 1.0
        );
        assert!(
            (ratio - 1.0).abs() <= (before - 1.0).abs() + 1.0e-6,
            "the correction must not make any row worse: {ratio} against \
             {before}. The tolerance is 1e-6 because the recorded BEFORE \
             column is rounded to six decimals, not because the rows are \
             allowed to drift"
        );
        worst = worst.max((ratio - 1.0).abs());
        rows += 1;
    }
    assert_eq!(rows, 6, "the sweep asserted nothing if it ran no rows");
    assert!(
        worst > 0.0,
        "a ratio of exactly 1.0 on every row would mean the quadrature and the \
         kernel are the same code, not two measurements of one quantity"
    );
}

#[test]
fn the_carrier_attenuation_is_the_box_sinc_and_not_the_gaussian_moment_match() {
    let dilation = lights::settings::PIXEL_BOX_VARIANCE;
    let mut rows = 0usize;
    for frequency in [0.10f64, 0.20, 0.30, 0.40] {
        let box_factor = lights::kernel::gabor::carrier_box_attenuation(frequency, 0.0, dilation);
        let gaussian_factor =
            (-2.0 * core::f64::consts::PI.powi(2) * frequency * frequency * dilation).exp();
        let expected = sinc_pi(frequency);
        assert!(
            (box_factor - expected).abs() < 1.0e-14,
            "at f = {frequency} the shipped factor is {box_factor} and the \
             closed form for a unit box is sinc(pi f) = {expected}"
        );
        if frequency >= 0.40 {
            assert!(
                (box_factor - gaussian_factor).abs() > 0.010,
                "at the Nyquist end the two models must differ by more than a \
                 percent, which is why the choice had to be measured rather \
                 than assumed: box {box_factor} against gaussian \
                 {gaussian_factor}. the carrier-frame reasoning \
                 named the GAUSSIAN factor; the crate's pixel is a BOX, and \
                 PIXEL_BOX_VARIANCE = 1/12 is that box's second moment"
            );
        }
        rows += 1;
    }
    assert_eq!(rows, 4);
    assert_eq!(
        lights::kernel::gabor::carrier_box_attenuation(0.4f64, 0.0, 0.0).to_bits(),
        1.0f64.to_bits(),
        "with no pixel dilation at all - the contract profile - the carrier is \
         untouched, exactly, so the contract arm is unchanged by this fix"
    );
    for variance in [f64::NAN, -1.0f64, -0.0f64, f64::NEG_INFINITY] {
        assert_eq!(
            lights::kernel::gabor::carrier_box_attenuation(0.4f64, 0.3, variance).to_bits(),
            1.0f64.to_bits(),
            "pixel_variance = {variance} must leave the carrier untouched. The \
             guard is written !(v > 0) rather than v < 0 for exactly this: at \
             v = 0 the two spellings agree, because sqrt(0) makes both sinc \
             arguments zero and the product is 1.0 either way, so a mutation \
             between them is invisible at zero and ONLY this row can see it. \
             NaN is where they part - NaN < 0 is false, so the v < 0 spelling \
             falls through and returns NaN"
        );
    }
}

#[test]
fn a_zero_frequency_attenuation_factor_is_exactly_one_and_multiplying_by_it_is_bit_identical() {
    let dilation = lights::settings::PIXEL_BOX_VARIANCE;
    let gaussian_factor = (-2.0 * core::f64::consts::PI.powi(2) * 0.0 * 0.0 * dilation).exp();
    assert_eq!(
        gaussian_factor.to_bits(),
        1.0f64.to_bits(),
        "exp(-2 pi^2 f^2 s^2) at f = 0 is exactly 1.0, not approximately 1.0"
    );
    assert_eq!(sinc_pi(0.0).to_bits(), 1.0f64.to_bits());
    let mut checked = 0usize;
    for x in [1.0f64, 0.5, 0.1, 1.0 / 3.0, 1.0e-300, 0.9999999999999999, 255.0 / 256.0] {
        assert_eq!(
            (x * gaussian_factor).to_bits(),
            x.to_bits(),
            "multiplying by exactly 1.0 is the identity on every finite double, \
             so applying the attenuation CANNOT break the zero-frequency \
             bit-for-bit control. That was the second of the two reasons \
             src/kernel/gabor_rs_comment/carrier_frame.md gave for omitting it, \
             and it is withdrawn: the real reason is the first one, that the \
             correction had never been measured"
        );
        checked += 1;
    }
    assert_eq!(checked, 7);
}
