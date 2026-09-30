use lights::blend::{BlendMode, CompositeClass};
use lights::camera::{Camera, CameraSpec};
use lights::image::{FloatImage, Framebuffer};
use lights::kernel::frame::{RayFrame, ScreenFrame};
use lights::kernel::gaussian::radius_for_cutoff;
use lights::kernel::hermite::{
    GaussHermiteColour, GaussHermiteDensity, HermiteParams, ORDER_ONE_PEAK_NORMALISER,
    ORDER_TWO_PEAK_NORMALISER, clamped_order, hermite_majorant, hermite_radius, peak_normaliser,
    probabilists_hermite,
};
use lights::kernel::{CurveShape, FIRST_HERMITE_ORDER, Kernel, KernelId, SECOND_HERMITE_ORDER,
    ZERO_ELEVATION, describe};
use lights::method::{ImageFormation, RenderMethod, create_strategy};
use lights::numerics::quaternion::Quat;
use lights::numerics::symmetric::{Sym2, Sym3};
use lights::numerics::vector::{Rgb, Vec2, Vec3};
use lights::probe::{NoProbe, ProbeSink};
use lights::scene::prepare::PreparedScene;
use lights::scene::{Atom, Scene};
use lights::settings::{PixelConvention, SharedSettings};

const EPS_ALPHA: f64 = 1.0 / 255.0;
const VIEWPORT: (u32, u32) = (48, 40);
const Z0: f64 = 64.0;
const DZ: f64 = 1.0 / 64.0;
const QUADRATURE_HALF_WIDTH: f64 = 12.0;
const QUADRATURE_STEPS: usize = 240_000;
const ORDERS: [u8; 2] = [FIRST_HERMITE_ORDER, SECOND_HERMITE_ORDER];

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

fn max_channel_gap(a: &FloatImage<f64>, b: &FloatImage<f64>) -> f64 {
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

fn midpoint_integral<F: Fn(f64) -> f64>(f: F) -> f64 {
    let step = 2.0 * QUADRATURE_HALF_WIDTH / QUADRATURE_STEPS as f64;
    let mut total = 0.0f64;
    for i in 0..QUADRATURE_STEPS {
        let t = -QUADRATURE_HALF_WIDTH + step * (i as f64 + 0.5);
        total += f(t);
    }
    total * step
}

fn order_polynomial(order: u8, t: f64) -> f64 {
    if order == 0 { 1.0 } else { probabilists_hermite(order, t) }
}

#[test]
fn the_probabilists_hermite_polynomials_are_orthogonal_under_the_gaussian_weight() {
    let root_two_pi = core::f64::consts::TAU.sqrt();
    let expected_norm = [root_two_pi, root_two_pi, 2.0 * root_two_pi];
    for m in 0u8..3 {
        for n in 0u8..3 {
            let value = midpoint_integral(|t| {
                order_polynomial(m, t) * order_polynomial(n, t) * (-0.5 * t * t).exp()
            });
            if m == n {
                assert!(
                    (value - expected_norm[m as usize]).abs() < 1.0e-6,
                    "the He_{m} norm under exp(-t^2/2) must be sqrt(2 pi) * {m}! = {}, \
                     and quadrature gives {value}",
                    expected_norm[m as usize]
                );
            } else {
                assert!(
                    value.abs() < 1.0e-6,
                    "He_{m} and He_{n} must be orthogonal under the gaussian weight; \
                     quadrature gives {value}. This is the whole reason an order-2 atom \
                     adds a correction the gaussian cannot supply rather than respending \
                     budget on the shape the gaussian already has"
                );
            }
        }
    }
    assert_eq!(probabilists_hermite(FIRST_HERMITE_ORDER, 2.5), 2.5);
    assert_eq!(probabilists_hermite(SECOND_HERMITE_ORDER, 2.0), 3.0);
    assert_eq!(probabilists_hermite(SECOND_HERMITE_ORDER, 0.0), -1.0);
}

#[test]
fn the_atom_envelope_makes_the_orders_non_orthogonal_and_the_number_is_recorded() {
    let paired = midpoint_integral(|t| order_polynomial(2, t) * (-t * t).exp());
    let want = -core::f64::consts::PI.sqrt() / 2.0;
    assert!(
        (paired - want).abs() < 1.0e-6,
        "the atoms this crate builds use the envelope exp(-|u|^2/2), not the \
         exp(-|u|^2/4) that makes Gauss-Hermite FUNCTIONS orthogonal in L2. The \
         order-0 and order-2 atoms therefore overlap by {want}, not by zero, and \
         quadrature gives {paired}. The envelope was held at the crate's gaussian \
         on purpose so that the atom-curve axis moves one thing at a time; this \
         number is the price, and it is asserted here so that no later report can \
         claim an orthogonality the run never had"
    );
    let odd_even = midpoint_integral(|t| {
        order_polynomial(1, t) * order_polynomial(2, t) * (-t * t).exp()
    });
    assert!(
        odd_even.abs() < 1.0e-9,
        "orders 1 and 2 remain orthogonal to EACH OTHER under this envelope by \
         parity, and that much does survive: {odd_even}"
    );
}

#[test]
fn each_order_has_the_expected_number_of_sign_changes() {
    let frame = screen_frame();
    for (order, expected) in [(FIRST_HERMITE_ORDER, 1usize), (SECOND_HERMITE_ORDER, 2)] {
        let shape = CurveShape::hermite(order, 0.0);
        let params = <GaussHermiteDensity as Kernel<f64>>::params_from(&shape);
        let prepared = <GaussHermiteDensity as Kernel<f64>>::prepare_screen(&params, &frame);
        let mut changes = 0usize;
        let mut previous = 0i8;
        let mut sampled = 0usize;
        for i in 0..4001 {
            let t = -8.0 + 16.0 * (i as f64) / 4000.0;
            let v = <GaussHermiteDensity as Kernel<f64>>::eval_screen(
                &prepared,
                Vec2::new(t, 0.0),
            );
            if v == 0.0 {
                continue;
            }
            sampled += 1;
            let sign = if v < 0.0 { -1i8 } else { 1i8 };
            if previous != 0 && sign != previous {
                changes += 1;
            }
            previous = sign;
        }
        assert!(
            sampled > 3900,
            "order {order}: only {sampled} of 4001 samples were non-zero, so the \
             sign-change count is describing an underflow rather than a lobe"
        );
        assert_eq!(
            changes, expected,
            "He_{order} must change sign {expected} times along the atom's own \
             direction and it changed {changes} times"
        );
    }
}

#[test]
fn the_normalisation_puts_the_peak_magnitude_at_exactly_one() {
    assert_eq!(peak_normaliser::<f64>(FIRST_HERMITE_ORDER), ORDER_ONE_PEAK_NORMALISER);
    assert_eq!(peak_normaliser::<f64>(SECOND_HERMITE_ORDER), ORDER_TWO_PEAK_NORMALISER);
    assert!(
        (ORDER_ONE_PEAK_NORMALISER - 0.5f64.exp()).abs() < 1.0e-15,
        "the order-1 normaliser is sqrt(e): the axial maximum of t exp(-t^2/2) sits \
         at t = 1 with value exp(-1/2)"
    );

    for frame in [screen_frame(), isotropic_frame()] {
        for order in ORDERS {
            let shape = CurveShape::hermite(order, 0.0);
            let params = <GaussHermiteDensity as Kernel<f64>>::params_from(&shape);
            let prepared =
                <GaussHermiteDensity as Kernel<f64>>::prepare_screen(&params, &frame);
            let axial = if order == FIRST_HERMITE_ORDER {
                <GaussHermiteDensity as Kernel<f64>>::eval_screen(
                    &prepared,
                    Vec2::new(1.0, 0.0),
                )
            } else {
                <GaussHermiteDensity as Kernel<f64>>::eval_screen(&prepared, Vec2::zero())
            };
            assert!(
                (axial.abs() - 1.0).abs() < 1.0e-15,
                "order {order}: the peak magnitude must be exactly 1 and it is \
                 {axial}. Dropping the normaliser leaves order 1 peaking at \
                 exp(-1/2) = 0.6065 and the kernel contract's |eval| <= 1 would \
                 still pass, which is why the peak is asserted rather than the bound"
            );
            let mut worst = 0.0f64;
            for i in 0..201 {
                for j in 0..201 {
                    let u = Vec2::new(
                        -7.0 + 14.0 * (i as f64) / 200.0,
                        -7.0 + 14.0 * (j as f64) / 200.0,
                    );
                    let v = <GaussHermiteDensity as Kernel<f64>>::eval_screen(&prepared, u);
                    assert!(v.is_finite());
                    worst = worst.max(v.abs());
                }
            }
            assert!(
                worst <= 1.0 + 1.0e-12,
                "order {order} reached {worst}, past the peak-normalised bound the \
                 kernel contract holds every kernel to"
            );
        }
    }
}

#[test]
fn an_order_one_atom_is_zero_at_its_own_centre_and_an_order_two_atom_is_minus_one() {
    let frame = isotropic_frame();
    for (order, want) in [(FIRST_HERMITE_ORDER, 0.0f64), (SECOND_HERMITE_ORDER, -1.0)] {
        let params =
            <GaussHermiteDensity as Kernel<f64>>::params_from(&CurveShape::hermite(order, 0.0));
        let prepared = <GaussHermiteDensity as Kernel<f64>>::prepare_screen(&params, &frame);
        let centre = <GaussHermiteDensity as Kernel<f64>>::eval_screen(&prepared, Vec2::zero());
        assert!(
            (centre - want).abs() < 1.0e-15,
            "order {order} evaluates to {centre} at its own centre, not {want}. The \
             shared kernel contract asserts eval(0) == 1 as 'peak-normalised, not a \
             pdf'; that is a property of the unsigned unimodal kernels and it does \
             NOT transfer to a signed one, which is why these two are pinned here \
             instead of the contract being loosened to let them through"
        );
    }
}

#[test]
fn the_support_is_solved_for_the_polynomial_rather_than_borrowed_from_the_gaussian() {
    for order in ORDERS {
        let normaliser = peak_normaliser::<f64>(order);
        for peak in [1.0f64, 0.7, 0.25, 0.05] {
            let solved = hermite_radius(order, normaliser, peak, EPS_ALPHA);
            let borrowed = radius_for_cutoff(peak, EPS_ALPHA);
            assert!(
                solved > borrowed,
                "order {order} peak {peak}: the solved radius {solved} is not larger \
                 than the gaussian's {borrowed}. The polynomial GROWS, so the \
                 gaussian's radius truncates the atom while it is still above the \
                 cutoff, and that truncation would read as a property of the \
                 representation"
            );
            assert!(
                hermite_majorant(order, normaliser, solved) * peak
                    <= EPS_ALPHA * (1.0 + 1.0e-9),
                "order {order} peak {peak}: the returned radius does not solve its \
                 own equation"
            );
        }
        assert_eq!(
            hermite_radius(order, normaliser, EPS_ALPHA * 0.5, EPS_ALPHA),
            0.0,
            "an atom everywhere under the cutoff contributes nowhere"
        );
        assert_eq!(hermite_radius(order, normaliser, 0.0, EPS_ALPHA), 0.0);
        assert_eq!(hermite_radius(order, normaliser, -1.0, EPS_ALPHA), 0.0);
    }
    let mut previous = 0.0f64;
    for peak in [0.01f64, 0.1, 0.3, 0.6, 1.0] {
        let r = hermite_radius(
            SECOND_HERMITE_ORDER,
            peak_normaliser::<f64>(SECOND_HERMITE_ORDER),
            peak,
            EPS_ALPHA,
        );
        assert!(
            r > previous,
            "the support must be strictly increasing in the peak: a support that \
             IGNORES its peak argument fails here and nowhere else"
        );
        previous = r;
    }
}

#[test]
fn the_hermite_support_bounds_the_true_signed_response() {
    for order in ORDERS {
        let shape = CurveShape::hermite(order, 0.8);
        let params = <GaussHermiteColour as Kernel<f64>>::params_from(&shape);
        for frame in [screen_frame(), isotropic_frame()] {
            let colour = <GaussHermiteColour as Kernel<f64>>::prepare_screen(&params, &frame);
            let density = <GaussHermiteDensity as Kernel<f64>>::prepare_screen(&params, &frame);
            for peak in [1.0f64, 0.7, 0.25, 0.05] {
                let r =
                    <GaussHermiteColour as Kernel<f64>>::support_screen(&colour, peak, EPS_ALPHA);
                if r <= 0.0 {
                    continue;
                }
                for ring in 1..=6 {
                    let radius = r * (1.0 + 0.05 * ring as f64);
                    for spoke in 0..64 {
                        let theta = core::f64::consts::TAU * (spoke as f64) / 64.0;
                        let u = Vec2::new(radius * theta.cos(), radius * theta.sin());
                        let modulated =
                            <GaussHermiteColour as Kernel<f64>>::eval_screen(&colour, u)
                                * <GaussHermiteColour as Kernel<f64>>::modulate_screen(
                                    &colour, u,
                                );
                        assert!(
                            modulated.abs() * peak <= EPS_ALPHA * (1.0 + 1.0e-4),
                            "order {order} peak {peak} radius {radius}: the colour \
                             placement's contribution {modulated} still exceeds the \
                             cutoff. What the support has to bound is the \
                             CONTRIBUTION, opacity * envelope * N He_n, not the alpha"
                        );
                        let full =
                            <GaussHermiteDensity as Kernel<f64>>::eval_screen(&density, u);
                        assert!(full.abs() * peak <= EPS_ALPHA * (1.0 + 1.0e-4));
                    }
                }
            }
        }
        let ray_params = <GaussHermiteColour as Kernel<f64>>::params_from(&shape);
        let ray = <GaussHermiteDensity as Kernel<f64>>::prepare_ray(&ray_params, &ray_frame());
        for peak in [1.0f64, 0.4] {
            let r = <GaussHermiteDensity as Kernel<f64>>::support_ray(&ray, peak, EPS_ALPHA);
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
                        let v = <GaussHermiteDensity as Kernel<f64>>::eval_ray(&ray, u);
                        assert!(v.abs() * peak <= EPS_ALPHA * (1.0 + 1.0e-4));
                    }
                }
            }
        }
    }
}

#[test]
fn the_order_is_clamped_to_the_two_that_are_built() {
    assert_eq!(clamped_order(0), FIRST_HERMITE_ORDER);
    assert_eq!(clamped_order(1), FIRST_HERMITE_ORDER);
    assert_eq!(clamped_order(2), SECOND_HERMITE_ORDER);
    assert_eq!(clamped_order(7), SECOND_HERMITE_ORDER);
    assert_eq!(
        <GaussHermiteColour as Kernel<f64>>::params_from(&CurveShape::hermite(9, 0.0)),
        HermiteParams::<f64> {
            order: SECOND_HERMITE_ORDER,
            orientation: 0.0,
            elevation: ZERO_ELEVATION
        }
    );
    assert_eq!(
        <GaussHermiteColour as Kernel<f64>>::params_from(&CurveShape::gaussian()).order,
        FIRST_HERMITE_ORDER,
        "the default shape names order 1, so a caller that never sets the knob gets \
         the first order rather than an undefined one"
    );
}

fn settings_with_cutoff(cutoff: f64) -> SharedSettings<f64> {
    let mut s = SharedSettings::<f64>::contract();
    s.blend.alpha_cutoff = cutoff;
    s
}

#[test]
fn the_only_thing_separating_the_two_sign_placements_is_the_alpha_cutoff_drop() {
    let opacity = 0.6f64;
    let brightest = 0.8f64;
    let atoms = vec![flat_atom(
        0,
        Vec2::new(22.0, 21.0),
        Vec2::new(4.5, 3.0),
        0.5,
        opacity,
        Rgb::new(brightest, 0.5, 0.35),
    )];

    let ceiling = |order: u8, cutoff: f64| {
        let radius = hermite_radius(order, peak_normaliser::<f64>(order), opacity, cutoff);
        let polynomial = if order == FIRST_HERMITE_ORDER {
            peak_normaliser::<f64>(order) * radius
        } else {
            radius * radius + 1.0
        };
        cutoff * polynomial * brightest
    };

    for order in ORDERS {
        let shape = CurveShape::hermite(order, 0.9);
        let params = vec![<GaussHermiteColour as Kernel<f64>>::params_from(&shape); 1];
        for method in RenderMethod::implemented() {
            let mut measured = Vec::new();
            for cutoff in [EPS_ALPHA, 1.0e-9] {
                let settings = settings_with_cutoff(cutoff);
                let scene = scene_of(&atoms, &settings);
                let colour =
                    render_with::<GaussHermiteColour>(&scene, &params, &settings, &method);
                let density =
                    render_with::<GaussHermiteDensity>(&scene, &params, &settings, &method);
                let gap = max_channel_gap(&colour, &density);
                let (peak, _, _) = colour.max_abs();
                assert!(
                    peak > 0.02,
                    "{:?} order {order} cutoff {cutoff}: both arms are nearly empty",
                    method.id()
                );
                assert!(
                    gap <= ceiling(order, cutoff),
                    "{:?} order {order} cutoff {cutoff}: the two sign placements \
                     disagree by {gap}, past the {} that alpha_cutoff * N_He_n(R) * \
                     colour allows. Over resolves a single atom over a black ground \
                     to colour*N_He_n*alpha and Additive to \
                     colour*(opacity*envelope*N_He_n): the same number. The only \
                     licence for a difference is the outer annulus where Over drops \
                     a sample whose alpha = opacity*envelope fell under the cutoff, \
                     and a larger gap means the two arms are representing different \
                     functions",
                    method.id(),
                    ceiling(order, cutoff)
                );
                measured.push(gap);
            }
            assert!(
                measured[0] > 1.0e-6,
                "{:?} order {order}: the placements agreed to {} at the production \
                 cutoff. They are NOT expected to. Gauss-Hermite's support is solved \
                 for the polynomial and is therefore WIDER than the envelope alone \
                 would justify, so there is always an annulus where alpha falls under \
                 the cutoff while the modulated colour, multiplied by an N_He_n that \
                 GROWS, does not. If this ever reaches zero the ceiling above is \
                 measuring nothing",
                method.id(),
                measured[0]
            );
            assert!(
                measured[1] * 1000.0 < measured[0],
                "{:?} order {order}: lowering the cutoff from {EPS_ALPHA} to 1e-9 \
                 moved the disagreement only from {} to {}. If the gap does not \
                 shrink with the cutoff then it is not the cutoff drop, and the two \
                 placements differ for some reason this test has not identified",
                method.id(),
                measured[0],
                measured[1]
            );
        }
    }
}

#[test]
fn both_hermite_sign_placements_drive_a_composited_channel_negative() {
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
    for order in ORDERS {
        let shape = CurveShape::hermite(order, 0.25);
        let params = vec![<GaussHermiteColour as Kernel<f64>>::params_from(&shape); 1];
        for method in RenderMethod::implemented() {
            let colour = render_with::<GaussHermiteColour>(&scene, &params, &settings, &method);
            let density = render_with::<GaussHermiteDensity>(&scene, &params, &settings, &method);
            assert!(
                min_channel(&colour) < -1.0e-3,
                "{:?} order {order}: the colour placement never drove a channel \
                 negative (lowest {}), so either the modulation never reached the \
                 compositing loop or the negative lobe was never sampled",
                method.id(),
                min_channel(&colour)
            );
            assert!(
                min_channel(&density) < -1.0e-3,
                "{:?} order {order}: the density placement never drove a channel \
                 negative (lowest {})",
                method.id(),
                min_channel(&density)
            );
        }
    }
}

#[test]
fn a_negative_hermite_weight_never_reaches_the_over_path() {
    let frame = screen_frame();
    for order in ORDERS {
        let params =
            <GaussHermiteColour as Kernel<f64>>::params_from(&CurveShape::hermite(order, 0.3));
        let prepared = <GaussHermiteColour as Kernel<f64>>::prepare_screen(&params, &frame);
        let mut negative_modulations = 0usize;
        for i in 0..64 {
            for j in 0..64 {
                let u = Vec2::new(
                    -4.0 + 8.0 * (i as f64) / 63.0,
                    -4.0 + 8.0 * (j as f64) / 63.0,
                );
                let weight = <GaussHermiteColour as Kernel<f64>>::eval_screen(&prepared, u);
                assert!(
                    weight >= 0.0,
                    "order {order}: the colour placement's eval_screen became \
                     {weight} at {u:?}. Over::step DROPS a sample whose alpha falls \
                     under the cutoff rather than negating it, and transmittance \
                     *= (1 - alpha) would grow, so the colour placement's whole \
                     claim is that its weight stays non-negative"
                );
                if <GaussHermiteColour as Kernel<f64>>::modulate_screen(&prepared, u) < 0.0 {
                    negative_modulations += 1;
                }
            }
        }
        assert!(
            negative_modulations > 100,
            "order {order}: only {negative_modulations} of 4096 samples carried a \
             negative modulation"
        );
    }
}

#[test]
fn the_hermite_kernels_declare_their_sign_placement() {
    let colour = describe::<f64, GaussHermiteColour>();
    assert_eq!(colour.name, "gauss-hermite-colour");
    assert_eq!(colour.id, KernelId(6));
    assert_eq!(colour.class, CompositeClass::Over);
    assert!(!colour.signed, "the BLEND is unsigned: the sign is in the colour");
    assert!(colour.order_dependent);
    assert!(colour.modulated);
    assert!(!colour.radial);
    assert!(colour.projectable);

    let density = describe::<f64, GaussHermiteDensity>();
    assert_eq!(density.name, "gauss-hermite-density");
    assert_eq!(density.id, KernelId(7));
    assert_eq!(density.class, CompositeClass::Additive);
    assert!(density.signed);
    assert!(!density.order_dependent);
    assert!(!density.modulated);

    assert_ne!(colour.id, density.id, "two kernels, two ids");
    assert!(<<GaussHermiteDensity as Kernel<f64>>::Blend as BlendMode<f64>>::SIGNED);
    assert!(!<<GaussHermiteColour as Kernel<f64>>::Blend as BlendMode<f64>>::SIGNED);
    assert_eq!(describe::<f32, GaussHermiteColour>().id, colour.id);
}

#[test]
fn the_lobes_follow_the_declared_orientation() {
    let frame = isotropic_frame();
    let mut turned = 0usize;
    for orientation in [0.0f64, 0.6, 1.9, -1.1] {
        let (sine, cosine) = orientation.sin_cos();
        let along = Vec2::new(cosine, sine);
        let across = Vec2::new(-sine, cosine);
        for order in ORDERS {
            let params = <GaussHermiteDensity as Kernel<f64>>::params_from(
                &CurveShape::hermite(order, orientation),
            );
            let prepared =
                <GaussHermiteDensity as Kernel<f64>>::prepare_screen(&params, &frame);
            let at_zero = if order == FIRST_HERMITE_ORDER { 0.0 } else { -1.0 };
            for r in [0.5f64, 1.0, 2.0, 3.0] {
                let sideways =
                    <GaussHermiteDensity as Kernel<f64>>::eval_screen(&prepared, across * r);
                let want = at_zero * (-0.5 * r * r).exp();
                assert!(
                    (sideways - want).abs() < 1.0e-14,
                    "order {order} at orientation {orientation}: {r} px ACROSS the \
                     declared direction the polynomial argument u.d is zero, so the \
                     value must be He_{order}(0) times the envelope = {want}, and it \
                     is {sideways}. A prepared direction that ignores the orientation \
                     and always points along x passes this at orientation 0 and \
                     nowhere else"
                );
            }
            let axial =
                <GaussHermiteDensity as Kernel<f64>>::eval_screen(&prepared, along * 1.0);
            let want_axial = if order == FIRST_HERMITE_ORDER {
                1.0
            } else {
                0.0
            };
            assert!(
                (axial - want_axial).abs() < 1.0e-14,
                "order {order} at orientation {orientation}: one unit ALONG the \
                 declared direction must read {want_axial} and it reads {axial}"
            );
            if orientation != 0.0 {
                turned += 1;
            }
        }
    }
    assert!(turned >= 6, "only {turned} of the cases used a non-zero orientation");
}
