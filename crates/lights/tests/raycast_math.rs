#![cfg(feature = "method-raycast")]
#[cfg(feature = "method-splat")]
use lights::agreement::atom_bound_terms;
use lights::agreement::atom_bound;
use lights::blend::{Additive, BlendMode, Over};
use lights::camera::{Camera, CameraSpec, Ray};
#[cfg(feature = "method-splat")]
use lights::camera::Projection;
use lights::dispatch::{Collect, Precision, RenderOutcome, RenderRequest, render};
use lights::error::LightsError;
use lights::image::{FloatImage, Framebuffer};
use lights::kernel::frame::{RayFrame, ScreenFrame};
use lights::kernel::gaussian::{Gaussian, radius_for_cutoff};
use lights::kernel::{CurveShape, Kernel, KernelFamily, KernelId};
use lights::method::raycast::execute::{
    Hit, collect_hits, composite_hits, intersect, order_key, sort_hits,
};
use lights::method::raycast::prepare::{
    RayAtom, TileBinIndex, axial_sigma, beer_lambert_alpha, cheap_accept, cheap_reject,
    extinction_for_opacity, line_integral, peak_parameter, prepare_atoms, quadratic, residual_r2,
    to_world_ray, whitened_residual,
};
use lights::method::raycast::{AccelKind, RayCast, RayCastConfig};
#[cfg(feature = "method-splat")]
use lights::method::splat::{SplatConfig, SplatRaster};
use lights::method::{ImageFormation, MethodId, RenderMethod};
use lights::numerics::float::FloatingPoint;
use lights::numerics::matrix::Mat3;
use lights::numerics::quaternion::Quat;
use lights::numerics::symmetric::{Sym2, Sym3};
use lights::numerics::vector::{Rgb, Vec2, Vec3};
use lights::probe::{NoProbe, ProbeSink};
use lights::scene::prepare::PreparedScene;
use lights::scene::{Atom, Scene};
use lights::settings::{
    MIN_SCALE, OrderingModel, PixelConvention, SharedSettings, SupportMode,
};
use lights::tiling::TileGrid;

const FLOOR: f64 = MIN_SCALE;
const EPS_ALPHA: f64 = 1.0 / 255.0;

fn diagonal(xx: f64, yy: f64, zz: f64) -> Sym3<f64> {
    Sym3::new(xx, 0.0, 0.0, yy, 0.0, zz)
}

fn lambda_of(sigma: &Sym3<f64>) -> Sym3<f64> {
    match sigma.inverse_spd() {
        Some(l) => l,
        None => panic!("the test covariance must be SPD"),
    }
}

fn whiten_of(sigma: &Sym3<f64>) -> lights::numerics::symmetric::LowerTri3<f64> {
    match sigma.cholesky() {
        Some(c) => c.inverse(),
        None => panic!("the test covariance must be SPD"),
    }
}

fn unit(v: Vec3<f64>) -> Vec3<f64> {
    match v.normalize() {
        Some(u) => u,
        None => panic!("cannot normalise a zero direction"),
    }
}

fn scene_of(atoms: &[Atom<f64>]) -> Scene<f64> {
    let mut scene = Scene::<f64>::new();
    for atom in atoms {
        match scene.push(*atom, FLOOR) {
            Ok(()) => (),
            Err(e) => panic!("scene rejected an atom: {e:?}"),
        }
    }
    scene
}

fn camera_of(spec: &CameraSpec, settings: &SharedSettings<f64>) -> Camera<f64> {
    match Camera::<f64>::from_spec(spec, settings.near_plane) {
        Ok(c) => c,
        Err(e) => panic!("camera failed: {e:?}"),
    }
}

fn prepare<K: Kernel<f64>>(
    scene: &Scene<f64>,
    settings: &SharedSettings<f64>,
) -> PreparedScene<f64, K> {
    match PreparedScene::<f64, K>::build(scene, &[], settings) {
        Ok(p) => p,
        Err(e) => panic!("prepare failed: {e:?}"),
    }
}

fn bound_raycast<K: Kernel<f64>>(
    scene: &Scene<f64>,
    spec: &CameraSpec,
    settings: &SharedSettings<f64>,
    config: RayCastConfig,
) -> (RayCast<f64, K>, PreparedScene<f64, K>, Camera<f64>) {
    let prepared = prepare::<K>(scene, settings);
    let camera = camera_of(spec, settings);
    let mut caster = match RayCast::<f64, K>::new(config, settings) {
        Ok(r) => r,
        Err(e) => panic!("RayCast::new failed: {e:?}"),
    };
    if let Err(e) = caster.build(&prepared) {
        panic!("build failed: {e:?}");
    }
    if let Err(e) = caster.bind(&prepared, &camera) {
        panic!("bind failed: {e:?}");
    }
    (caster, prepared, camera)
}

#[cfg(feature = "method-splat")]
fn bound_splat(
    scene: &Scene<f64>,
    spec: &CameraSpec,
    settings: &SharedSettings<f64>,
) -> (
    SplatRaster<f64, Gaussian>,
    PreparedScene<f64, Gaussian>,
    Camera<f64>,
) {
    let prepared = prepare::<Gaussian>(scene, settings);
    let camera = camera_of(spec, settings);
    let mut raster = match SplatRaster::<f64, Gaussian>::new(SplatConfig::contract(), settings) {
        Ok(r) => r,
        Err(e) => panic!("SplatRaster::new failed: {e:?}"),
    };
    if let Err(e) = raster.build(&prepared) {
        panic!("build failed: {e:?}");
    }
    if let Err(e) = raster.bind(&prepared, &camera) {
        panic!("bind failed: {e:?}");
    }
    (raster, prepared, camera)
}

fn render_through(
    scene: Scene<f64>,
    spec: CameraSpec,
    settings: SharedSettings<f64>,
    method: RenderMethod,
    collect: Collect,
) -> RenderOutcome {
    let request = RenderRequest {
        precision: Precision::F64,
        kernel: KernelFamily::Gaussian,
        method,
        settings,
        camera: spec,
        scene,
        collect,
        shapes: None,
    };
    match render(&request) {
        Ok(o) => o,
        Err(e) => panic!("render failed: {e:?}"),
    }
}

fn pinhole(focal: f64, principal: [f64; 2], w: u32, h: u32) -> CameraSpec {
    CameraSpec::look_at([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0], 45.0, w, h)
        .with_intrinsics([focal, focal], principal)
        .with_convention(PixelConvention::CornerSample)
}

#[derive(Copy, Clone, Debug, Default)]
struct Unprojectable;

#[derive(Copy, Clone, Debug, Default)]
struct SignedTest;

macro_rules! plain_kernel {
    ($name:ident, $blend:ty, $id:expr, $label:expr, $marginal:expr) => {
        impl<T: FloatingPoint> Kernel<T> for $name {
            type Params = ();
            type Screen = ();
            type Ray = ();
            type Blend = $blend;
            const ID: KernelId = KernelId($id);
            const NAME: &'static str = $label;
            const RADIAL: bool = true;
            const HAS_SCREEN_MARGINAL: bool = $marginal;

            fn params_from(_shape: &CurveShape) {}
            fn prepare_screen(_p: &(), _f: &ScreenFrame<T>) {}
            fn prepare_ray(_p: &(), _f: &RayFrame<T>) {}
            fn eval_screen(_s: &(), u: Vec2<T>) -> T {
                (-T::HALF * u.length_squared()).exp()
            }
            fn eval_ray(_r: &(), u: Vec3<T>) -> T {
                (-T::HALF * u.length_squared()).exp()
            }
            fn support_screen(_s: &(), peak: T, cutoff: T) -> T {
                radius_for_cutoff(peak, cutoff)
            }
            fn support_ray(_r: &(), peak: T, cutoff: T) -> T {
                radius_for_cutoff(peak, cutoff)
            }
            fn prefilter(p: &(), cov: Sym2<T>, pixel_variance: T) -> ((), Sym2<T>, T) {
                (*p, cov.add_diagonal(pixel_variance.max(T::ZERO)), T::ONE)
            }
        }
    };
}

plain_kernel!(Unprojectable, Over, 910, "raycast-unprojectable", false);
plain_kernel!(SignedTest, Additive, 911, "raycast-signed", true);

#[test]
fn tv_r1_axial_line_integral() {
    let sigma = diagonal(0.01, 0.0025, 0.04);
    let lambda = lambda_of(&sigma);
    let whiten = whiten_of(&sigma);
    let mu = Vec3::new(0.0, 0.0, 5.0);
    let delta = -mu;
    let d = Vec3::new(0.0, 0.0, 1.0);

    let q = quadratic(&lambda, delta, d);
    assert!((q.a - 25.0).abs() < 1e-13, "a {}", q.a);
    assert!((q.b + 125.0).abs() < 1e-12, "b {}", q.b);
    assert!((q.c - 625.0).abs() < 1e-10, "c {}", q.c);

    let t_star = peak_parameter(&q);
    assert!((t_star - 5.0).abs() < 1e-14, "t* {t_star}");
    assert!((axial_sigma(&q) - 0.2).abs() < 1e-15);

    let r2 = residual_r2(&lambda, delta, d, t_star);
    assert!(r2 < 1e-24, "r2 must vanish on axis, got {r2}");
    let u = whitened_residual(&whiten, delta, d, t_star);
    assert!(u.length_squared() < 1e-24, "whitened residual {u:?}");
    assert!((Gaussian::eval_ray(&(), u) - 1.0).abs() < 1e-15);

    let expected = (2.0 * std::f64::consts::PI).sqrt() * 0.2;
    assert!((expected - 0.5013256549262001).abs() < 1e-15);
    assert!(
        (line_integral(&q, r2) - expected).abs() < 1e-15,
        "I {}",
        line_integral(&q, r2)
    );
}

#[test]
fn tv_r2_direction_scale_invariance() {
    let sigma = diagonal(0.01, 0.0025, 0.04);
    let lambda = lambda_of(&sigma);
    let whiten = whiten_of(&sigma);
    let mu = Vec3::new(0.0, 0.0, 5.0);
    let delta = -mu;

    let unit_d = Vec3::new(0.0, 0.0, 1.0);
    let scaled = Vec3::new(0.0, 0.0, 3.0);

    let q1 = quadratic(&lambda, delta, unit_d);
    let q3 = quadratic(&lambda, delta, scaled);
    assert!((q3.a - 225.0).abs() < 1e-11, "a {}", q3.a);
    assert!((q3.b + 375.0).abs() < 1e-11, "b {}", q3.b);

    let t1 = peak_parameter(&q1);
    let t3 = peak_parameter(&q3);
    assert!((t3 - 5.0 / 3.0).abs() < 1e-15, "t* {t3}");
    assert!((axial_sigma(&q3) - 1.0 / 15.0).abs() < 1e-16);

    let r1 = residual_r2(&lambda, delta, unit_d, t1);
    let r3 = residual_r2(&lambda, delta, scaled, t3);
    assert!(
        (r1 - r3).abs() < 1e-15,
        "r2 is invariant to the direction length: {r1} vs {r3}"
    );
    let u1 = whitened_residual(&whiten, delta, unit_d, t1);
    let u3 = whitened_residual(&whiten, delta, scaled, t3);
    assert!((u1.length_squared() - u3.length_squared()).abs() < 1e-15);

    let i1 = line_integral(&q1, r1);
    let i3 = line_integral(&q3, r3);
    assert!((i3 - i1 / 3.0).abs() < 1e-16, "I scales as 1/|d|: {i3}");
    assert!(
        (3.0 * i3 - i1).abs() < 1e-15,
        "the ARC-LENGTH integral |d| * I is the invariant, {} vs {}",
        3.0 * i3,
        i1
    );
    assert!((i1 - 0.5013256549262001).abs() < 1e-15);
}

#[test]
fn tv_r3_oblique_both_normalisations() {
    let sigma = diagonal(0.01, 0.0025, 0.04);
    let lambda = lambda_of(&sigma);
    let whiten = whiten_of(&sigma);
    let mu = Vec3::new(0.0, 0.0, 5.0);
    let delta = -mu;

    let raw = Vec3::new(0.02, 0.01, 1.0);
    let norm = raw.length();
    assert!((norm - 1.0002499687578101).abs() < 1e-15, "|d_raw| {norm}");
    let normalised = unit(raw);

    let exact = 1250.0 / 627.0;
    assert!(
        (exact - 1.9936204146730462).abs() < 1e-15,
        "the closed form is 1250/627"
    );

    let mut radii = Vec::new();
    for d in [raw, normalised] {
        let q = quadratic(&lambda, delta, d);
        let t = peak_parameter(&q);
        let r2 = residual_r2(&lambda, delta, d, t);
        let u = whitened_residual(&whiten, delta, d, t);
        assert!(
            (u.length_squared() - r2).abs() < 1e-14,
            "the whitened and the quadratic-form spellings must agree"
        );
        assert!(
            (r2 - exact).abs() < 1e-12,
            "r2 {r2} against the exact 1250/627 = {exact}"
        );
        radii.push(r2);
    }
    assert!(
        (radii[0] - radii[1]).abs() < 1e-15,
        "normalising the direction must not move r2: {} vs {}",
        radii[0],
        radii[1]
    );

    let q_raw = quadratic(&lambda, delta, raw);
    let q_norm = quadratic(&lambda, delta, normalised);
    assert!((q_raw.a - 25.08).abs() < 1e-13, "a_raw {}", q_raw.a);
    assert!(
        (peak_parameter(&q_raw) - 4.984051036682616).abs() < 1e-13,
        "t*_raw {}",
        peak_parameter(&q_raw)
    );
    assert!(
        (peak_parameter(&q_norm) - 4.985296893729118).abs() < 1e-13,
        "t*_norm {}",
        peak_parameter(&q_norm)
    );
    let peak = (-0.5 * exact).exp();
    assert!(
        (peak - 0.36905477385115215).abs() < 1e-15,
        "peak response {peak}"
    );

    let i_raw = line_integral(&q_raw, radii[0]);
    let i_norm = line_integral(&q_norm, radii[1]);
    assert!(
        (norm * i_raw - i_norm).abs() < 1e-15,
        "|d_raw| * I_raw must recover I_norm: {} vs {i_norm}",
        norm * i_raw
    );
}

#[test]
fn tv_r5_f32_cancellation_regression() {
    let sigma32 = diagonal(0.0025, 0.0025, 0.0025);
    let lambda64 = lambda_of(&sigma32);
    let whiten64 = whiten_of(&sigma32);
    let mu = Vec3::new(0.0, 0.0, 50.0);
    let delta = -mu;
    let d = unit(Vec3::new(2.0e-5, 0.0, 1.0));

    let q = quadratic(&lambda64, delta, d);
    let t = peak_parameter(&q);
    let stable64 = residual_r2(&lambda64, delta, d, t);
    let cancelling64 = q.c - q.b * q.b / q.a;
    assert!(
        (stable64 - 4.0e-4).abs() < 1e-12,
        "f64 residual {stable64} must land on the true 4.0e-4"
    );
    assert!(
        (cancelling64 - 4.0e-4).abs() > 1.0e-11,
        "the cancelling form already loses seven digits in f64: {cancelling64}"
    );

    let lambda32 = Sym3::<f32>::from_f64(&lambda64);
    let whiten32 = lights::numerics::symmetric::LowerTri3::<f32> {
        m: [
            whiten64.m[0] as f32,
            whiten64.m[1] as f32,
            whiten64.m[2] as f32,
            whiten64.m[3] as f32,
            whiten64.m[4] as f32,
            whiten64.m[5] as f32,
        ],
    };
    let delta32 = Vec3::<f32>::from_f64(delta);
    let d32 = Vec3::<f32>::from_f64(d);
    let q32 = quadratic(&lambda32, delta32, d32);
    let t32 = peak_parameter(&q32);
    let stable32 = residual_r2(&lambda32, delta32, d32, t32);
    let whitened32 = whitened_residual(&whiten32, delta32, d32, t32).length_squared();
    let cancelling32 = q32.c - q32.b * q32.b / q32.a;

    assert!(
        stable32 >= 3.9e-4,
        "THE SHIPPED residual form must survive f32: got {stable32}"
    );
    assert!(
        whitened32 >= 3.9e-4,
        "the whitened spelling must survive f32 too: got {whitened32}"
    );
    assert_eq!(
        cancelling32, 0.0f32,
        "the cancelling form returns EXACTLY zero in f32, which renders the atom \
         as a perfectly opaque disc; this is the whole reason residual_r2 exists"
    );
}

#[test]
fn tv_r6_peak_parameter_versus_centre_depth() {
    let half = (std::f64::consts::PI / 8.0).sin();
    let rotation = Quat::new_wxyz((std::f64::consts::PI / 8.0).cos(), 0.0, half, 0.0);
    let atom = Atom::new(
        Vec3::new(0.0, 0.0, 5.0),
        rotation,
        Vec3::new(0.6, 0.05, 0.05),
        1.0,
        Rgb::new(1.0, 1.0, 1.0),
    );
    let sigma = atom.covariance();
    assert!(
        (sigma.zz - 0.18125).abs() < 1e-14,
        "Sigma_zz {} must be 0.18125",
        sigma.zz
    );
    let lambda = lambda_of(&sigma);
    let mu = atom.mean;
    let delta = -mu;

    let table = [
        (0.0f64, 0.0f64),
        (0.05, -0.22307137007497513),
        (0.10, -0.4023633676607501),
        (0.15, -0.5423500781199735),
    ];
    for (px, expected) in table {
        let d = unit(Vec3::new(px, 0.0, 1.0));
        let q = quadratic(&lambda, delta, d);
        let t_star = peak_parameter(&q);
        let t_euclidean = d.dot(mu);
        let shift = t_star - t_euclidean;
        assert!(
            (shift - expected).abs() < 1e-6,
            "px {px}: t* - t_e = {shift}, expected {expected}"
        );
    }
    let last = {
        let d = unit(Vec3::new(0.15, 0.0, 1.0));
        let q = quadratic(&lambda, delta, d);
        peak_parameter(&q) - d.dot(mu)
    };
    assert!(
        last.abs() > 0.5,
        "a renderer that sorts by centre depth reports a shift of ZERO here and \
         is wrong by half a world unit: {last}"
    );
}

#[test]
fn tv_r7_cull_threshold_table() {
    for (opacity, expected) in [
        (1.00f64, 3.3290429691304455f64),
        (0.50, 3.1138774428671665),
        (0.25, 2.8826963711215012),
        (0.10, 2.5450652063019445),
    ] {
        let from_formula = (2.0 * (opacity / EPS_ALPHA).ln()).sqrt();
        assert!(
            (from_formula - expected).abs() < 1e-13,
            "the reference value must come from the formula, not a printed digit"
        );
        let shipped = Gaussian::support_ray(&(), opacity, EPS_ALPHA);
        assert!(
            (shipped - from_formula).abs() < 1e-14,
            "support_ray({opacity}) = {shipped}, formula {from_formula}"
        );
    }
    for (cutoff, expected) in [
        (1.0e-2f64, 3.034854258770293f64),
        (1.0e-3, 3.7169221888498383),
        (1.0e-4, 4.291932052578694),
    ] {
        let from_formula = (2.0 * (1.0 / cutoff).ln()).sqrt();
        assert!((from_formula - expected).abs() < 1e-13);
        assert!((Gaussian::support_ray(&(), 1.0, cutoff) - from_formula).abs() < 1e-14);
    }
    assert_eq!(
        Gaussian::support_ray(&(), EPS_ALPHA, EPS_ALPHA),
        0.0,
        "an atom at or below the cutoff has no support at all"
    );
    assert_eq!(Gaussian::support_ray(&(), 0.5 * EPS_ALPHA, EPS_ALPHA), 0.0);
}

#[test]
fn tv_r8_schur_identity_for_any_perpendicular_basis() {
    let rz = 37.0f64.to_radians();
    let ry = 19.0f64.to_radians();
    let qz = Quat::new_wxyz((rz * 0.5).cos(), 0.0, 0.0, (rz * 0.5).sin());
    let qy = Quat::new_wxyz((ry * 0.5).cos(), 0.0, (ry * 0.5).sin(), 0.0);
    let rotation = qz.to_mat3().mul_mat(&qy.to_mat3());
    let factor = rotation.scale_columns(Vec3::new(0.13, 0.041, 0.27));
    let sigma = Sym3::from_factor(&factor);
    let lambda = lambda_of(&sigma);
    let mu = Vec3::new(0.3, -0.2, 6.0);
    let d = unit(Vec3::new(0.11, -0.07, 1.0));

    let helper = Vec3::new(0.0, 0.0, 1.0);
    let p0 = unit(d.cross(helper));
    let p1 = d.cross(p0);
    let projected = Sym2::new(
        sigma.quad_form(p0),
        p0.dot(sigma.apply(p1)),
        sigma.quad_form(p1),
    );
    let conic = match projected.inverse() {
        Some(c) => c,
        None => panic!("the perpendicular block must be invertible"),
    };

    let mut worst = 0.0f64;
    for k in 0..6 {
        let eye = Vec3::new(
            0.05 * (k as f64) - 0.1,
            0.03 * (k as f64),
            -0.02 * (k as f64),
        );
        let delta = eye - mu;
        let q = quadratic(&lambda, delta, d);
        let r2 = residual_r2(&lambda, delta, d, peak_parameter(&q));
        let perp = Vec2::new(delta.dot(p0), delta.dot(p1));
        let schur = conic.quad_form(perp);
        let relative = (r2 - schur).abs() / schur.max(1.0e-12);
        worst = worst.max(relative);
    }
    assert!(
        worst <= 1e-14,
        "the ray r2 IS the perpendicular Mahalanobis form; worst relative {worst}"
    );
}

#[test]
fn tv_r9_marginal_not_conditional_numeric() {
    let sigma = Sym3::new(2.0, 0.0, 1.0, 1.0, 0.0, 1.0);
    let lambda = lambda_of(&sigma);
    assert!((lambda.xx - 1.0).abs() < 1e-14);
    assert!((lambda.xz + 1.0).abs() < 1e-14);
    assert!((lambda.zz - 2.0).abs() < 1e-14);

    let origin = Vec3::new(0.3, -0.2, 0.0);
    let d = Vec3::new(0.0, 0.0, 1.0);
    let q = quadratic(&lambda, origin, d);
    assert!((q.a - 2.0).abs() < 1e-14, "a {}", q.a);
    let t_star = peak_parameter(&q);
    assert!((t_star - 0.15).abs() < 1e-14, "t* {t_star}");
    let r2 = residual_r2(&lambda, origin, d, t_star);
    assert!((r2 - 0.085).abs() < 1e-14, "r2 {r2}");

    let amplitude = (2.0 * std::f64::consts::PI / 2.0).sqrt();
    assert!((amplitude - 1.7724538509055159).abs() < 1e-15);
    let shipped = line_integral(&q, r2);
    assert!(
        (shipped - 1.6987028712837324).abs() < 1e-12,
        "the MARGINAL answer is 1.6987028712837324, got {shipped}"
    );

    let conditional_block = match Sym2::new(lambda.xx, lambda.xy, lambda.yy).inverse() {
        Some(c) => c,
        None => panic!("the conditional block must be invertible"),
    };
    let offset = Vec2::new(0.3, -0.2);
    let conditional_conic = match conditional_block.inverse() {
        Some(c) => c,
        None => panic!("the conditional conic must exist"),
    };
    let conditional = amplitude * (-0.5 * conditional_conic.quad_form(offset)).exp();
    assert!(
        (conditional - 1.660908834021542).abs() < 1e-12,
        "the CONDITIONAL answer is 1.6609088340, got {conditional}"
    );
    assert!(
        (shipped - conditional).abs() > 0.03,
        "the two differ by 2.2 percent; a test with zero xz and yz covariance \
         could not tell them apart"
    );

    let samples = 400001usize;
    let span = 40.0f64;
    let step = 2.0 * span / ((samples - 1) as f64);
    let mut integral = 0.0f64;
    for i in 0..samples {
        let z = -span + step * (i as f64);
        let p = Vec3::new(0.3, -0.2, z);
        let weight = if i == 0 || i == samples - 1 { 0.5 } else { 1.0 };
        integral += weight * (-0.5 * lambda.quad_form(p)).exp();
    }
    integral *= step;
    assert!(
        (integral - shipped).abs() < 1e-9,
        "numerical line integration gives {integral}, the closed form {shipped}"
    );
}

#[test]
fn raycast_r2_is_clamped_at_zero_and_the_weight_never_exceeds_one() {
    let sigma = diagonal(0.0025, 0.0025, 0.0025);
    let lambda = lambda_of(&sigma);
    let whiten = whiten_of(&sigma);
    let mu = Vec3::new(0.0, 0.0, 5.0);
    let delta = -mu;
    let d = Vec3::new(0.0, 0.0, 1.0);
    let q = quadratic(&lambda, delta, d);
    let t = peak_parameter(&q);
    let r2 = residual_r2(&lambda, delta, d, t);
    assert_eq!(r2, 0.0, "the on-axis residual must clamp to exactly zero");

    let indefinite = Sym3::new(1.0, 0.0, 0.0, 1.0, 0.0, -1.0);
    let contrived = residual_r2(&indefinite, Vec3::new(0.0, 0.0, 1.0), d, 0.0);
    assert_eq!(
        contrived, 0.0,
        "a negative quadratic form must clamp, or exp(-r2/2) exceeds one and \
         alpha exceeds the atom's opacity"
    );

    let u = whitened_residual(&whiten, delta, d, t);
    let weight = Gaussian::eval_ray(&(), u);
    assert!(weight <= 1.0, "weight {weight}");
    assert!((weight - 1.0).abs() < 1e-15);
}

#[test]
fn the_alpha_convention_is_peak_along_ray_not_beer_lambert() {
    let sigma = diagonal(0.01, 0.0025, 0.04);
    let lambda = lambda_of(&sigma);
    let mu = Vec3::new(0.0, 0.0, 5.0);
    let delta = -mu;
    let d = Vec3::new(0.0, 0.0, 1.0);
    let q = quadratic(&lambda, delta, d);
    let axial = axial_sigma(&q);
    assert!((axial - 0.2).abs() < 1e-15);

    for (opacity, expected_kappa) in [(0.5f64, 1.382628584332f64), (0.9, 4.592992739087)] {
        let kappa = match extinction_for_opacity(opacity, axial) {
            Some(k) => k,
            None => panic!("the peak-matched conversion must exist for o < 1"),
        };
        assert!(
            (kappa - expected_kappa).abs() < 1e-9,
            "kappa {kappa} for o {opacity}"
        );

        let at_peak = beer_lambert_alpha(kappa, line_integral(&q, 0.0));
        assert!(
            (at_peak - opacity).abs() < 1e-12,
            "the two conventions agree EXACTLY at r = 0, and nowhere else: {at_peak}"
        );

        let mut worst = 0.0f64;
        let mut worst_g = 0.0f64;
        for i in 1..=200000 {
            let g = (i as f64) / 200000.0;
            let r2 = -2.0 * g.ln();
            let peak_alpha = opacity * g;
            let beer = beer_lambert_alpha(kappa, line_integral(&q, r2));
            let difference = beer - peak_alpha;
            if difference > worst {
                worst = difference;
                worst_g = g;
            }
        }
        let closed_form = if opacity == 0.5 {
            (0.043035666027964325f64, 0.4712335f64)
        } else {
            (0.24195910495291578, 0.407973)
        };
        assert!(
            (worst - closed_form.0).abs() < 1e-6,
            "o {opacity}: measured divergence {worst}, closed form {}",
            closed_form.0
        );
        assert!(
            (worst_g - closed_form.1).abs() < 1e-3,
            "o {opacity}: the maximum sits at g = {worst_g}"
        );
    }

    assert!(
        extinction_for_opacity(1.0f64, 0.2).is_none(),
        "a fully opaque atom has no finite extinction coefficient"
    );
    assert!(extinction_for_opacity(0.5f64, 0.0).is_none());
}

#[test]
fn the_shipped_weight_is_the_kernel_response_and_the_blend_makes_the_alpha() {
    let settings = SharedSettings::<f64>::contract();
    let spec = pinhole(600.0, [32.0, 24.0], 64, 48);
    let atom = Atom::isotropic(
        Vec3::new(0.0, 0.0, 5.0),
        0.05,
        0.6,
        Rgb::new(1.0, 0.5, 0.25),
    );
    let scene = scene_of(&[atom]);
    let (caster, prepared, _) =
        bound_raycast::<Gaussian>(&scene, &spec, &settings, RayCastConfig::contract());
    let cell = match caster.sample_pixel(&prepared, 32, 24) {
        Ok(c) => c,
        Err(e) => panic!("sample_pixel failed: {e:?}"),
    };
    assert!(
        (cell.transmittance - 0.4).abs() < 1e-12,
        "at the centre the ray alpha must be exactly the atom opacity 0.6, so \
         the leftover transmittance is 0.4, got {}",
        cell.transmittance
    );
    assert!((cell.rgb.x - 0.6).abs() < 1e-12, "rgb {:?}", cell.rgb);
    assert!((cell.rgb.y - 0.3).abs() < 1e-12);
    assert!((cell.rgb.z - 0.15).abs() < 1e-12);
}

#[test]
fn the_alpha_cap_comes_from_the_shared_blend_not_from_the_ray_path() {
    let settings = SharedSettings::<f64>::contract();
    let spec = pinhole(600.0, [32.0, 24.0], 64, 48);
    let atom = Atom::isotropic(Vec3::new(0.0, 0.0, 5.0), 0.05, 1.0, Rgb::new(1.0, 1.0, 1.0));
    let scene = scene_of(&[atom]);
    let (caster, prepared, _) =
        bound_raycast::<Gaussian>(&scene, &spec, &settings, RayCastConfig::contract());
    let cell = match caster.sample_pixel(&prepared, 32, 24) {
        Ok(c) => c,
        Err(e) => panic!("sample_pixel failed: {e:?}"),
    };
    assert!(
        (cell.transmittance - 0.01).abs() < 1e-14,
        "the 0.99 cap is the blend's, applied identically on both methods: {}",
        cell.transmittance
    );
}

#[test]
#[cfg(feature = "method-splat")]
fn centre_pixel_agreement_with_the_splat_path() {
    let settings = SharedSettings::<f64>::contract();
    let spec = pinhole(600.0, [32.0, 24.0], 96, 72);
    let cases = [
        (Vec3::new(0.0, 0.0, 5.0), 32u32, 24u32),
        (Vec3::new(0.1, 0.05, 5.0), 44, 30),
        (Vec3::new(-0.2, 0.15, 6.0), 12, 39),
    ];
    for (mean, px, py) in cases {
        for scale in [
            Vec3::new(0.05, 0.05, 0.05),
            Vec3::new(0.12, 0.04, 0.25),
            Vec3::new(0.3, 0.3, 0.02),
        ] {
            let rotation = Quat::new_wxyz(0.6, 0.5, -0.4, 0.47);
            let atom = Atom::new(mean, rotation, scale, 0.7, Rgb::new(0.9, 0.4, 0.2));
            let scene = scene_of(&[atom]);
            let (caster, ray_scene, _) =
                bound_raycast::<Gaussian>(&scene, &spec, &settings, RayCastConfig::contract());
            let (raster, splat_scene, _) = bound_splat(&scene, &spec, &settings);
            let ray_cell = match caster.sample_pixel(&ray_scene, px, py) {
                Ok(c) => c,
                Err(e) => panic!("raycast sample failed: {e:?}"),
            };
            let splat_cell = match raster.sample_pixel(&splat_scene, px, py) {
                Ok(c) => c,
                Err(e) => panic!("splat sample failed: {e:?}"),
            };
            let difference = (ray_cell.rgb - splat_cell.rgb).max_component().abs();
            assert!(
                difference <= 1e-12,
                "at the pixel where the centre projects the two methods must \
                 return exactly o: {mean:?} {scale:?} gave {ray_cell:?} vs \
                 {splat_cell:?}"
            );
            assert!(
                (ray_cell.transmittance - 0.3).abs() < 1e-12,
                "and that alpha is the atom's opacity: {}",
                ray_cell.transmittance
            );
        }
    }
}

#[test]
#[cfg(feature = "method-splat")]
fn orthographic_agreement_with_the_splat_path_at_every_pixel() {
    let mut settings = SharedSettings::<f64>::contract();
    settings.background = Rgb::new(0.05, 0.09, 0.13);
    let spec = CameraSpec::look_at(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
        45.0,
        64,
        48,
    )
    .orthographic(0.5)
    .with_convention(PixelConvention::CornerSample);
    assert_eq!(spec.projection, Projection::Orthographic);

    let atoms = [
        Atom::new(
            Vec3::new(0.1, -0.05, 5.0),
            Quat::new_wxyz(0.92387953251128674, 0.0, 0.0, 0.38268343236508978),
            Vec3::new(0.16, 0.07, 0.2),
            0.55,
            Rgb::new(0.9, 0.2, 0.1),
        ),
        Atom::new(
            Vec3::new(-0.12, 0.08, 4.0),
            Quat::new_wxyz(0.6, 0.5, -0.4, 0.47),
            Vec3::new(0.09, 0.19, 0.11),
            0.65,
            Rgb::new(0.1, 0.8, 0.3),
        ),
        Atom::isotropic(
            Vec3::new(0.02, 0.02, 6.0),
            0.13,
            0.45,
            Rgb::new(0.2, 0.3, 0.95),
        ),
    ];
    let scene = scene_of(&atoms);

    let splat = render_through(
        scene.clone(),
        spec.clone(),
        settings,
        RenderMethod::splat_contract(),
        Collect::Pixels,
    );
    let ray = render_through(
        scene,
        spec,
        settings,
        RenderMethod::raycast_contract(),
        Collect::Pixels,
    );

    let (peak, _, _) = splat.image.max_abs();
    assert!(peak > 0.3, "the orthographic scene must actually render");
    let difference = match ray.image.difference(&splat.image) {
        Ok(d) => d,
        Err(e) => panic!("size mismatch: {e:?}"),
    };
    let (worst, wx, wy) = difference.max_abs();
    assert!(
        worst <= 1e-12,
        "under an orthographic projection the Jacobian is EXACT and constant, so \
         the two methods are mathematically identical at every pixel; a failure \
         here is a bug with no legitimate explanation. Worst {worst} at ({wx},{wy})"
    );
}

#[test]
#[cfg(feature = "method-splat")]
fn perspective_disagreement_stays_inside_the_atom_bound() {
    let settings = SharedSettings::<f64>::contract();
    let focal = 800.0f64;
    let width = 2400u32;
    let height = 96u32;
    for (offset, ceiling, floor) in [
        (0.0f64, 2.0e-4f64, 1.0e-4f64),
        (2.5, 8.0e-3, 5.0e-3),
        (5.0, 1.2e-2, 8.0e-3),
    ] {
        let mean = Vec3::new(offset, 0.0, 5.0);
        let principal = [1200.0 - focal * offset / 5.0, 48.0];
        let spec = pinhole(focal, principal, width, height);
        let camera = camera_of(&spec, &settings);
        let limits = camera.jacobian_limits();
        assert!(
            limits.x > offset / 5.0,
            "the off-axis clamp has NO ray-casting counterpart, so every test \
             atom must sit inside it: limit {} against tan {}",
            limits.x,
            offset / 5.0
        );

        let atom = Atom::isotropic(mean, 0.05, 1.0, Rgb::splat(1.0));
        let scene = scene_of(&[atom]);
        let splat = render_through(
            scene.clone(),
            spec.clone(),
            settings,
            RenderMethod::splat_contract(),
            Collect::Pixels,
        );
        let ray = render_through(
            scene,
            spec,
            settings,
            RenderMethod::raycast_contract(),
            Collect::Pixels,
        );
        assert_eq!(splat.report.clamped_atoms, 0, "no atom may be clamped");

        let (peak, _, _) = splat.image.max_abs();
        assert!(peak > 0.9, "the atom must actually render: {peak}");

        let mut worst = 0.0f64;
        let mut fringe_worst = 0.0f64;
        let mut fringe = 0usize;
        for (left, right) in ray.image.pixels.iter().zip(splat.image.pixels.iter()) {
            let difference = (left.x - right.x).abs();
            if right.x >= 2.0 * EPS_ALPHA {
                worst = worst.max(difference);
            } else {
                fringe_worst = fringe_worst.max(difference);
                if difference > ceiling {
                    fringe += 1;
                }
            }
        }
        assert!(
            worst <= ceiling,
            "tan theta {}: measured {worst} exceeds the math sheet's ceiling \
             {ceiling}",
            offset / 5.0
        );
        assert!(
            worst >= floor,
            "tan theta {}: measured {worst} is BELOW {floor}. The lower bound is \
             the important half - it proves the harness is measuring the \
             perspective term and has not silently disabled one of the two \
             renderers",
            offset / 5.0
        );
        assert!(
            fringe_worst <= EPS_ALPHA + ceiling,
            "tan theta {}: the truncation fringe reached {fringe_worst}. Both \
             methods cut their kernel at their OWN r_cut, so on the annulus \
             where the two radii straddle it one composites an alpha of exactly \
             eps_alpha and the other composites nothing. That step is bounded by \
             eps_alpha + B_atom and it is NOT covered by the math sheet's \
             composited tolerance formula",
            offset / 5.0
        );
        assert!(
            fringe < 600,
            "tan theta {}: {fringe} fringe pixels exceed the bound, which is more \
             than the straddle annulus can hold",
            offset / 5.0
        );

        let sigma_cam = diagonal(0.0025, 0.0025, 0.0025);
        let predicted = atom_bound(mean, &sigma_cam, 1.0);
        let terms = atom_bound_terms(mean, &sigma_cam, 1.0);
        assert!(
            (predicted - terms.total).abs() < 1e-18,
            "the bound and its decomposition must agree"
        );
        let ratio = worst / predicted;
        assert!(
            (0.95..=1.02).contains(&ratio),
            "tan theta {}: measured {worst} against the closed form {predicted} is \
             a ratio of {ratio}. A loose bound hides real bugs and a violated one \
             means the closed form is wrong",
            offset / 5.0
        );
        if offset > 0.0 {
            assert!(
                terms.first_order > 10.0 * terms.second_order,
                "off axis the divergence is dominated by the FIRST-order sin(theta) \
                 term: {:?}",
                terms
            );
        }
    }
}

#[test]
fn tile_bins_and_brute_force_produce_the_same_image() {
    let mut settings = SharedSettings::<f64>::contract();
    settings.background = Rgb::new(0.02, 0.03, 0.04);
    let spec = pinhole(500.0, [48.0, 36.0], 96, 72);
    let mut atoms = Vec::new();
    for i in 0..20u32 {
        let t = i as f64;
        let half = 0.11 * t;
        atoms.push(Atom::new(
            Vec3::new(0.4 * (0.8 * t).sin(), 0.3 * (1.1 * t).cos(), 3.0 + 0.21 * t),
            Quat::new_wxyz(half.cos(), 0.0, half.sin(), 0.0),
            Vec3::new(0.03 + 0.002 * t, 0.05, 0.04),
            0.4,
            Rgb::new((0.17 * t).fract(), (0.31 * t).fract(), (0.53 * t).fract()),
        ));
    }
    atoms.push(Atom::new(
        Vec3::new(0.0, 0.05, 4.5),
        Quat::new_wxyz(0.9238795325112867, 0.0, 0.0, 0.3826834323650898),
        Vec3::new(0.22, 0.004, 0.004),
        0.6,
        Rgb::new(0.9, 0.9, 0.1),
    ));
    let scene = scene_of(&atoms);

    let brute = render_through(
        scene.clone(),
        spec.clone(),
        settings,
        RenderMethod::RayCast(RayCastConfig {
            accel: AccelKind::BruteForce,
            tile: 16,
            max_hits_per_pixel: 4096,
        }),
        Collect::PixelsAndCounters,
    );
    let binned = render_through(
        scene,
        spec,
        settings,
        RenderMethod::RayCast(RayCastConfig {
            accel: AccelKind::TileBins,
            tile: 16,
            max_hits_per_pixel: 4096,
        }),
        Collect::PixelsAndCounters,
    );

    let (peak, _, _) = brute.image.max_abs();
    assert!(peak > 0.2, "the binning scene must actually render");
    for (i, (a, b)) in brute
        .image
        .pixels
        .iter()
        .zip(binned.image.pixels.iter())
        .enumerate()
    {
        assert_eq!(
            (a.x.to_bits(), a.y.to_bits(), a.z.to_bits()),
            (b.x.to_bits(), b.y.to_bits(), b.z.to_bits()),
            "pixel {i}: the conservative sphere bound must never drop a \
             contributor, so the two acceleration kinds are bit-identical"
        );
    }
    assert!(
        binned.counters.candidates_tested < brute.counters.candidates_tested,
        "and the index must actually prune: {} tested against {}",
        binned.counters.candidates_tested,
        brute.counters.candidates_tested
    );
    assert_eq!(
        binned.counters.kernel_evaluations, brute.counters.kernel_evaluations,
        "pruning must remove only candidates that could not have contributed"
    );
}

#[test]
fn the_candidate_bound_is_two_sided_and_never_drops_a_contributor() {
    let sigma = diagonal(0.04, 0.0025, 0.01);
    let lambda = lambda_of(&sigma);
    let scale_max = 0.2f64;
    let scale_min = 0.05f64;
    let support = 3.0f64;
    let mu = Vec3::new(0.0, 0.0, 5.0);

    let mut rejected = 0usize;
    let mut accepted = 0usize;
    for i in 0..60 {
        for j in 0..60 {
            let d = unit(Vec3::new(
                -0.3 + 0.01 * (i as f64),
                -0.3 + 0.01 * (j as f64),
                1.0,
            ));
            let relative = mu;
            let q = quadratic(&lambda, -mu, d);
            let r2 = residual_r2(&lambda, -mu, d, peak_parameter(&q));
            let h = relative.cross(d).length();

            assert!(
                r2 >= h * h / (scale_max * scale_max) - 1e-12,
                "the reject bracket r2 >= h^2 / s_max^2 must hold"
            );
            assert!(
                r2 <= h * h / (scale_min * scale_min) + 1e-9,
                "the accept bracket r2 <= h^2 / s_min^2 must hold"
            );

            if cheap_reject(relative, d, support, scale_max) {
                rejected += 1;
                assert!(
                    r2 > support * support,
                    "cheap_reject dropped a ray whose exact r2 is {r2}, inside the \
                     support of {}",
                    support * support
                );
            }
            if cheap_accept(relative, d, support, scale_min) {
                accepted += 1;
                assert!(
                    r2 <= support * support,
                    "cheap_accept promised a contributor whose exact r2 is {r2}"
                );
            }
        }
    }
    assert!(rejected > 100, "the reject test must actually fire: {rejected}");
    assert!(accepted > 20, "the accept test must actually fire: {accepted}");
}

#[test]
fn cheap_reject_only_removes_rays_the_exact_test_would_reject() {
    let settings = SharedSettings::<f64>::contract();
    let spec = pinhole(600.0, [256.0, 256.0], 512, 512);
    let camera = camera_of(&spec, &settings);
    let needle = Atom::new(
        Vec3::new(0.0, 0.0, 5.0),
        Quat::new_wxyz(0.9238795325112867, 0.0, 0.0, 0.3826834323650898),
        Vec3::new(0.25, 0.008, 0.05),
        0.8,
        Rgb::splat(1.0),
    );
    let scene = scene_of(&[needle]);
    let prepared = prepare::<Gaussian>(&scene, &settings);
    let mut atoms: Vec<RayAtom<f64, Gaussian>> = Vec::new();
    prepare_atoms::<f64, Gaussian>(&prepared, &camera, &settings, &mut atoms);
    let atom = atoms[0];
    let to_world = camera.rotation.transpose();

    let mut rejected = 0usize;
    let mut hit = 0usize;
    let mut surviving_misses = 0usize;
    for py in (0..512u32).step_by(6) {
        for px in (0..512u32).step_by(6) {
            let ray = to_world_ray(&to_world, camera.eye, &camera.ray(px, py));
            let exact = intersect(&atom, &ray, settings.ordering, settings.near_plane);
            if cheap_reject(
                atom.mean - ray.origin,
                ray.direction,
                atom.support,
                atom.scale_max,
            ) {
                rejected += 1;
                assert!(
                    exact.is_none(),
                    "cheap_reject dropped pixel ({px},{py}) that the EXACT test \
                     accepts. The sphere bound must be conservative, or removing \
                     it would change the image instead of only the run time"
                );
            } else if exact.is_none() {
                surviving_misses += 1;
            } else {
                hit += 1;
            }
        }
    }
    assert!(rejected > 200, "the cheap reject must actually fire: {rejected}");
    assert!(hit > 20, "and the atom must actually be hit: {hit}");
    assert!(
        surviving_misses > 100,
        "the sphere is much looser than the exact test for a needle, which is \
         why cheap_reject cannot replace it: {surviving_misses}"
    );
}

#[test]
fn the_tile_index_is_a_lossless_prune_at_the_pixel_level() {
    let settings = SharedSettings::<f64>::contract();
    let spec = pinhole(500.0, [48.0, 36.0], 96, 72);
    let mut atoms = Vec::new();
    for i in 0..12u32 {
        let t = i as f64;
        let half = 0.23 * t;
        atoms.push(Atom::new(
            Vec3::new(0.35 * (0.9 * t).sin(), 0.3 * (1.3 * t).cos(), 3.0 + 0.19 * t),
            Quat::new_wxyz(half.cos(), 0.0, half.sin(), 0.0),
            Vec3::new(0.16, 0.005, 0.03),
            0.5,
            Rgb::new((0.19 * t).fract(), (0.37 * t).fract(), (0.59 * t).fract()),
        ));
    }
    let scene = scene_of(&atoms);
    let (binned, binned_scene, _) = bound_raycast::<Gaussian>(
        &scene,
        &spec,
        &settings,
        RayCastConfig {
            accel: AccelKind::TileBins,
            tile: 16,
            max_hits_per_pixel: 4096,
        },
    );
    let (brute, brute_scene, _) = bound_raycast::<Gaussian>(
        &scene,
        &spec,
        &settings,
        RayCastConfig {
            accel: AccelKind::BruteForce,
            tile: 16,
            max_hits_per_pixel: 4096,
        },
    );
    assert!(binned.bin_index().is_some());
    assert!(brute.bin_index().is_none());

    let mut touched = 0usize;
    for py in 0..72u32 {
        for px in 0..96u32 {
            let a = match binned.sample_pixel(&binned_scene, px, py) {
                Ok(c) => c,
                Err(e) => panic!("binned sample failed: {e:?}"),
            };
            let b = match brute.sample_pixel(&brute_scene, px, py) {
                Ok(c) => c,
                Err(e) => panic!("brute sample failed: {e:?}"),
            };
            assert_eq!(
                a.transmittance.to_bits(),
                b.transmittance.to_bits(),
                "({px},{py}): walking the tile candidate list and walking every \
                 atom must give the SAME accumulator, bit for bit"
            );
            assert_eq!(a.rgb.x.to_bits(), b.rgb.x.to_bits(), "({px},{py}) red");
            assert_eq!(a.rgb.y.to_bits(), b.rgb.y.to_bits(), "({px},{py}) green");
            assert_eq!(a.rgb.z.to_bits(), b.rgb.z.to_bits(), "({px},{py}) blue");
            if a.transmittance < 1.0 {
                touched += 1;
            }
        }
    }
    assert!(touched > 300, "the scene must cover enough pixels: {touched}");
}

#[test]
fn raycast_hit_budget_is_an_error_not_a_truncation() {
    let settings = SharedSettings::<f64>::contract();
    let spec = pinhole(600.0, [32.0, 24.0], 64, 48);
    let mut atoms = Vec::new();
    for i in 0..4u32 {
        atoms.push(Atom::isotropic(
            Vec3::new(0.0, 0.0, 4.0 + 0.5 * (i as f64)),
            0.05,
            0.3,
            Rgb::new(0.4, 0.4, 0.4),
        ));
    }
    let scene = scene_of(&atoms);

    let at_limit = RayCastConfig {
        accel: AccelKind::BruteForce,
        tile: 16,
        max_hits_per_pixel: 4,
    };
    let (caster, prepared, _) = bound_raycast::<Gaussian>(&scene, &spec, &settings, at_limit);
    match caster.sample_pixel(&prepared, 32, 24) {
        Ok(cell) => assert!(
            cell.transmittance < 1.0,
            "exactly the limit must render, not fail"
        ),
        Err(e) => panic!("a pixel with exactly the limit must succeed: {e:?}"),
    }

    let below = RayCastConfig {
        max_hits_per_pixel: 3,
        ..at_limit
    };
    let (tight, tight_scene, _) = bound_raycast::<Gaussian>(&scene, &spec, &settings, below);
    match tight.sample_pixel(&tight_scene, 32, 24) {
        Err(LightsError::BudgetExhausted { site, limit }) => {
            assert_eq!(limit, 3, "the error must report the limit it hit");
            assert!(
                site.contains("raycast"),
                "the error must name the site, got {site}"
            );
        }
        Err(other) => panic!("wrong error: {other:?}"),
        Ok(cell) => panic!(
            "the budget SILENTLY TRUNCATED instead of failing: {cell:?}"
        ),
    }

    let outcome = render(&RenderRequest {
        precision: Precision::F64,
        kernel: KernelFamily::Gaussian,
        method: RenderMethod::RayCast(below),
        settings,
        camera: spec,
        scene,
        collect: Collect::Pixels,
        shapes: None,
    });
    match outcome {
        Err(LightsError::BudgetExhausted { limit, .. }) => assert_eq!(limit, 3),
        Err(other) => panic!("the whole render must fail hard too: {other:?}"),
        Ok(_) => panic!("the render produced an image past its hit budget"),
    }
}

#[test]
fn raycast_accepts_both_orderings_and_refuses_a_zero_budget_or_tile() {
    let mut hit_depth = SharedSettings::<f64>::contract();
    hit_depth.ordering = OrderingModel::HitDepth;
    assert!(
        RayCast::<f64, Gaussian>::new(RayCastConfig::contract(), &hit_depth).is_ok(),
        "RayCast honours BOTH ordering models"
    );

    let mut zero_budget = RayCastConfig::contract();
    zero_budget.max_hits_per_pixel = 0;
    assert!(RayCast::<f64, Gaussian>::new(zero_budget, &SharedSettings::contract()).is_err());
    let mut zero_tile = RayCastConfig::contract();
    zero_tile.tile = 0;
    assert!(RayCast::<f64, Gaussian>::new(zero_tile, &SharedSettings::contract()).is_err());
}

#[test]
fn raycast_accepts_a_kernel_that_has_no_screen_marginal() {
    let settings = SharedSettings::<f64>::contract();
    assert!(
        RayCast::<f64, Unprojectable>::new(RayCastConfig::contract(), &settings).is_ok(),
        "ray casting integrates along the ray and needs no screen marginal"
    );
}

#[test]
#[cfg(feature = "method-splat")]
fn the_capability_matrix_is_asymmetric_and_both_halves_are_live() {
    let settings = SharedSettings::<f64>::contract();
    match SplatRaster::<f64, Unprojectable>::new(SplatConfig::contract(), &settings) {
        Err(LightsError::KernelNotProjectable { kernel, method }) => {
            assert_eq!(kernel, "raycast-unprojectable");
            assert_eq!(method, MethodId::SplatRaster);
        }
        Err(other) => panic!("wrong error: {other:?}"),
        Ok(_) => panic!("EWA projection IS marginalisation and this kernel has none"),
    }

    let mut hit_depth = SharedSettings::<f64>::contract();
    hit_depth.ordering = OrderingModel::HitDepth;
    match SplatRaster::<f64, Gaussian>::new(SplatConfig::contract(), &hit_depth) {
        Err(LightsError::OrderingUnsupported { ordering, method }) => {
            assert_eq!(ordering, OrderingModel::HitDepth);
            assert_eq!(method, MethodId::SplatRaster);
        }
        Err(other) => panic!("wrong error: {other:?}"),
        Ok(_) => panic!("per-pixel sorting is not built into the rasteriser"),
    }
    assert!(
        RayCast::<f64, Gaussian>::new(RayCastConfig::contract(), &hit_depth).is_ok(),
        "the switch is asymmetric BY DESIGN: only the image-order method can \
         order per ray"
    );
}

#[test]
fn hit_depth_ordering_is_a_different_order_from_centre_depth() {
    let mut centre = SharedSettings::<f64>::contract();
    centre.ordering = OrderingModel::CentreDepth;
    let mut hit = SharedSettings::<f64>::contract();
    hit.ordering = OrderingModel::HitDepth;

    let spec = pinhole(600.0, [32.0, 24.0], 64, 48);
    let tilt = std::f64::consts::PI / 8.0;
    let atoms = [
        Atom::new(
            Vec3::new(0.0, 0.0, 5.0),
            Quat::new_wxyz(tilt.cos(), 0.0, tilt.sin(), 0.0),
            Vec3::new(0.6, 0.05, 0.05),
            0.9,
            Rgb::new(1.0, 0.0, 0.0),
        ),
        Atom::isotropic(
            Vec3::new(0.06, 0.0, 4.85),
            0.06,
            0.9,
            Rgb::new(0.0, 0.0, 1.0),
        ),
    ];
    let scene = scene_of(&atoms);

    let by_centre = render_through(
        scene.clone(),
        spec.clone(),
        centre,
        RenderMethod::raycast_contract(),
        Collect::Pixels,
    );
    let by_hit = render_through(
        scene,
        spec,
        hit,
        RenderMethod::raycast_contract(),
        Collect::Pixels,
    );
    let difference = match by_hit.image.difference(&by_centre.image) {
        Ok(d) => d,
        Err(e) => panic!("size mismatch: {e:?}"),
    };
    let (worst, _, _) = difference.max_abs();
    assert!(
        worst > 0.01,
        "t* ordering and centre-depth ordering are genuinely different orders; \
         if they agree the switch is dead configuration. Worst {worst}"
    );
}

#[test]
fn the_sort_key_is_a_strict_total_order_that_survives_an_unstable_sort() {
    let mut previous = order_key(f64::NEG_INFINITY);
    for i in 0..2000 {
        let v = -10.0 + 0.01 * (i as f64);
        let bits = order_key(v);
        assert!(
            bits > previous,
            "order_key must be strictly increasing across zero: {v}"
        );
        previous = bits;
    }
    assert!(order_key(-1.0e-300f64) < order_key(0.0f64));
    assert!(order_key(0.0f64) < order_key(1.0e-300f64));
    assert!(order_key(-1.0f32) < order_key(1.0f32));

    let mut hits: Vec<Hit<f64>> = Vec::new();
    for i in 0..96u32 {
        hits.push(Hit {
            key: 3.0 + ((i % 6) as f64),
            weight: 1.0,
            opacity: 0.1,
            colour: Rgb::splat(0.5),
            bound: 0.0,
            index: 95 - i,
        });
    }
    sort_hits(&mut hits);
    for pair in hits.windows(2) {
        let left = (order_key(pair[0].key), pair[0].index);
        let right = (order_key(pair[1].key), pair[1].index);
        assert!(
            left < right,
            "the emitted order must be the STRICT total order on the whole key, \
             not whatever an unstable sort happened to leave: {left:?} {right:?}"
        );
    }
}

#[test]
fn sample_pixel_reproduces_the_tile_render_bit_for_bit() {
    let mut settings = SharedSettings::<f64>::contract();
    settings.background = Rgb::new(0.03, 0.05, 0.07);
    settings.threads = 1;
    let spec = pinhole(420.0, [33.0, 25.0], 67, 51);
    let mut atoms = Vec::new();
    for i in 0..9u32 {
        let t = i as f64;
        let half = 0.19 * t;
        atoms.push(Atom::new(
            Vec3::new(
                0.6 * (1.3 * t).sin(),
                0.5 * (0.7 * t).cos(),
                3.0 + 0.29 * t,
            ),
            Quat::new_wxyz(half.cos(), half.sin(), 0.0, 0.0),
            Vec3::new(0.05, 0.08, 0.06),
            0.5,
            Rgb::new((0.23 * t).fract(), (0.41 * t).fract(), (0.61 * t).fract()),
        ));
    }
    let scene = scene_of(&atoms);
    let (caster, prepared, camera) =
        bound_raycast::<Gaussian>(&scene, &spec, &settings, RayCastConfig::production());

    let mut fb = match Framebuffer::<f64, Over>::new(camera.width, camera.height, 16) {
        Ok(f) => f,
        Err(e) => panic!("framebuffer failed: {e:?}"),
    };
    let mut sink = ProbeSink::<f64>::counters_only(camera.width, camera.height);
    if let Err(e) = caster.render::<NoProbe>(&prepared, &mut fb, &mut sink) {
        panic!("render failed: {e:?}");
    }

    let mut touched = 0usize;
    for py in 0..camera.height {
        for px in 0..camera.width {
            let tiled = fb.cell(px, py);
            let single = match caster.sample_pixel(&prepared, px, py) {
                Ok(c) => c,
                Err(e) => panic!("sample_pixel({px},{py}) failed: {e:?}"),
            };
            assert_eq!(
                tiled.transmittance.to_bits(),
                single.transmittance.to_bits(),
                "pixel ({px},{py}) transmittance"
            );
            assert_eq!(tiled.rgb.x.to_bits(), single.rgb.x.to_bits(), "({px},{py}) red");
            assert_eq!(tiled.rgb.y.to_bits(), single.rgb.y.to_bits(), "({px},{py}) green");
            assert_eq!(tiled.rgb.z.to_bits(), single.rgb.z.to_bits(), "({px},{py}) blue");
            if tiled.transmittance < 1.0 {
                touched += 1;
            }
        }
    }
    assert!(
        touched > 400,
        "the scene must cover enough of a viewport whose right and bottom tiles \
         are CLIPPED, or the stride is untested: {touched}"
    );
    assert_ne!(camera.width % 16, 0, "the viewport must clip its edge tiles");
    assert_ne!(camera.height % 16, 0);
}

#[test]
fn counters_and_tolerance_come_out_of_the_render_pass() {
    let mut settings = SharedSettings::<f64>::contract();
    settings.threads = 1;
    let spec = pinhole(600.0, [32.0, 24.0], 64, 48);
    let atoms = [
        Atom::isotropic(Vec3::new(0.0, 0.0, 4.0), 0.05, 0.8, Rgb::new(1.0, 0.0, 0.0)),
        Atom::isotropic(Vec3::new(0.02, 0.0, 5.0), 0.05, 0.8, Rgb::new(0.0, 1.0, 0.0)),
        Atom::isotropic(
            Vec3::new(0.0, 0.02, 6.0),
            0.05,
            0.8,
            Rgb::new(0.0, 0.0, 1.0),
        ),
    ];
    let scene = scene_of(&atoms);
    let outcome = render_through(
        scene,
        spec,
        settings,
        RenderMethod::raycast_contract(),
        Collect::PixelsAndTolerance,
    );

    let counters = outcome.counters;
    assert_eq!(counters.atoms_prepared, 3);
    assert_eq!(counters.atoms_projected, 3);
    assert_eq!(counters.atoms_culled_near, 0);
    assert_eq!(
        counters.pixels_visited,
        64 * 48,
        "brute force visits every pixel"
    );
    assert_eq!(
        counters.candidates_tested,
        3 * 64 * 48,
        "and tests every atom at every pixel"
    );
    assert!(
        counters.kernel_evaluations < counters.candidates_tested,
        "the cheap sphere reject and the support test must remove work: {} of {}",
        counters.kernel_evaluations,
        counters.candidates_tested
    );
    assert!(counters.kernel_evaluations > 0);
    assert!(counters.composited_hits > 0);
    assert!(
        counters.early_outs <= counters.pixels_visited,
        "a pixel can stop at most once; {} stops over {} pixels means Step::Stop \
         was observed and ignored",
        counters.early_outs,
        counters.pixels_visited
    );

    let tolerance = match outcome.tolerance {
        Some(t) => t,
        None => panic!("PixelsAndTolerance must produce a tolerance image"),
    };
    assert_eq!((tolerance.width, tolerance.height), (64, 48));
    let (peak, _, _) = tolerance.max_abs();
    assert!(
        peak > 0.0,
        "the tolerance must be accumulated IN the render pass, weighted by the \
         transmittance in front of each atom"
    );
    let (lo, hi) = tolerance.min_max_channel();
    assert_eq!(lo, 0.0, "pixels no atom reaches carry no tolerance");
    assert!(hi > 0.0);

    let camera = camera_of(&pinhole(600.0, [32.0, 24.0], 64, 48), &settings);
    let prepared = prepare::<Gaussian>(&scene_of(&atoms), &settings);
    let mut ray_atoms: Vec<RayAtom<f64, Gaussian>> = Vec::new();
    prepare_atoms::<f64, Gaussian>(&prepared, &camera, &settings, &mut ray_atoms);
    let unweighted: f64 = ray_atoms.iter().map(|a| a.bound).sum();
    let centre = tolerance.get(32, 24).x;
    assert!(
        centre > ray_atoms[0].bound * 0.99,
        "the nearest atom sees a transmittance of one, so the accumulated \
         tolerance is at least its own bound: {centre} against {}",
        ray_atoms[0].bound
    );
    assert!(
        centre < unweighted * 0.95,
        "and every atom behind it is weighted DOWN by the transmittance in front \
         of it: {centre} against the unweighted sum {unweighted}"
    );
}

#[test]
fn the_near_cull_is_the_same_test_the_splat_path_uses() {
    let settings = SharedSettings::<f64>::contract();
    let spec = pinhole(600.0, [32.0, 24.0], 64, 48);
    let camera = camera_of(&spec, &settings);
    assert!((settings.near_plane - 0.2).abs() < 1e-15);

    for (z, survives) in [(0.1f64, false), (0.2, false), (0.2 + 1e-9, true)] {
        let atom = Atom::isotropic(Vec3::new(0.0, 0.0, z), 0.01, 0.9, Rgb::splat(1.0));
        let scene = scene_of(&[atom]);
        let prepared = prepare::<Gaussian>(&scene, &settings);
        let mut out: Vec<RayAtom<f64, Gaussian>> = Vec::new();
        let (counters, clamped) =
            prepare_atoms::<f64, Gaussian>(&prepared, &camera, &settings, &mut out);
        assert_eq!(clamped, 0, "the ray path has no Jacobian and never clamps");
        assert_eq!(
            out.len(),
            usize::from(survives),
            "z = {z}: the near test is <=, not <, exactly as the rasteriser's is"
        );
        assert_eq!(counters.atoms_culled_near, u64::from(!survives));
    }

    let below_cutoff = Atom::isotropic(
        Vec3::new(0.0, 0.0, 5.0),
        0.05,
        0.5 * EPS_ALPHA,
        Rgb::splat(1.0),
    );
    let scene = scene_of(&[below_cutoff]);
    let prepared = prepare::<Gaussian>(&scene, &settings);
    let mut out: Vec<RayAtom<f64, Gaussian>> = Vec::new();
    let (counters, _) = prepare_atoms::<f64, Gaussian>(&prepared, &camera, &settings, &mut out);
    assert_eq!(out.len(), 0, "an atom below the alpha cutoff cannot contribute");
    assert_eq!(counters.atoms_culled_support, 1);

    let mut fixed = SharedSettings::<f64>::contract();
    fixed.support = SupportMode::FixedSigmas(3.0);
    let mut fixed_out: Vec<RayAtom<f64, Gaussian>> = Vec::new();
    let (fixed_counters, _) =
        prepare_atoms::<f64, Gaussian>(&prepared, &camera, &fixed, &mut fixed_out);
    assert_eq!(
        fixed_out.len(),
        0,
        "under FixedSigmas the support radius is 3.0 whatever the opacity, so the \
         cutoff cull is the ONLY thing standing between a transparent atom and \
         the pixel loop"
    );
    assert_eq!(fixed_counters.atoms_culled_support, 1);
}

#[test]
fn the_shipped_hot_path_survives_f32_cancellation() {
    let settings = SharedSettings::<f32>::contract();
    let spec = CameraSpec::look_at([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0], 45.0, 64, 48)
        .with_intrinsics([50000.0, 50000.0], [32.0, 24.0])
        .with_convention(PixelConvention::CornerSample);
    let camera = match Camera::<f32>::from_spec(&spec, settings.near_plane) {
        Ok(c) => c,
        Err(e) => panic!("camera failed: {e:?}"),
    };
    let mut scene = Scene::<f32>::new();
    let atom = Atom::isotropic(
        Vec3::new(0.0f32, 0.0, 50.0),
        0.05,
        0.5,
        Rgb::new(1.0f32, 1.0, 1.0),
    );
    match scene.push(atom, MIN_SCALE as f32) {
        Ok(()) => (),
        Err(e) => panic!("scene rejected the atom: {e:?}"),
    }
    let prepared = match PreparedScene::<f32, Gaussian>::build(&scene, &[], &settings) {
        Ok(p) => p,
        Err(e) => panic!("prepare failed: {e:?}"),
    };
    let mut caster = match RayCast::<f32, Gaussian>::new(RayCastConfig::contract(), &settings) {
        Ok(r) => r,
        Err(e) => panic!("RayCast::new failed: {e:?}"),
    };
    if let Err(e) = caster.build(&prepared) {
        panic!("build failed: {e:?}");
    }
    if let Err(e) = caster.bind(&prepared, &camera) {
        panic!("bind failed: {e:?}");
    }

    let cell = match caster.sample_pixel(&prepared, 33, 24) {
        Ok(c) => c,
        Err(e) => panic!("sample_pixel failed: {e:?}"),
    };
    let expected = 0.5f32 * (-0.5f32 * 4.0e-4f32).exp();
    assert!(
        (cell.rgb.x - expected).abs() < 1.0e-6,
        "the pixel one step off axis must see r2 = 4e-4 and therefore {expected}, \
         got {}. The cancelling form c - b*b/a returns EXACTLY zero here in f32 \
         and paints the atom as a perfectly opaque disc",
        cell.rgb.x
    );
    assert!(
        cell.rgb.x < 0.49995,
        "and it must be strictly below the on-axis value: {}",
        cell.rgb.x
    );
    let centre = match caster.sample_pixel(&prepared, 32, 24) {
        Ok(c) => c,
        Err(e) => panic!("sample_pixel failed: {e:?}"),
    };
    assert!(
        (centre.rgb.x - 0.5).abs() < 1.0e-7,
        "while the on-axis pixel does see the full opacity: {}",
        centre.rgb.x
    );
}

#[test]
fn the_exact_test_prunes_far_more_than_the_conservative_sphere() {
    let mut settings = SharedSettings::<f64>::contract();
    settings.threads = 1;
    let width = 512u32;
    let height = 512u32;
    let spec = pinhole(600.0, [256.0, 256.0], width, height);
    let needle = Atom::new(
        Vec3::new(0.0, 0.0, 5.0),
        Quat::identity(),
        Vec3::new(0.30, 0.006, 0.006),
        0.9,
        Rgb::new(1.0, 1.0, 1.0),
    );
    let scene = scene_of(&[needle]);
    let outcome = render_through(
        scene,
        spec.clone(),
        settings,
        RenderMethod::RayCast(RayCastConfig {
            accel: AccelKind::TileBins,
            tile: 16,
            max_hits_per_pixel: 4096,
        }),
        Collect::PixelsAndCounters,
    );
    let counters = outcome.counters;
    let frame = u64::from(width) * u64::from(height);

    assert!(
        counters.pixels_visited < frame,
        "the tile index must leave whole tiles unvisited: {} of {frame}",
        counters.pixels_visited
    );
    assert_eq!(
        counters.candidates_tested, counters.pixels_visited,
        "one atom, one candidate per visited pixel"
    );
    assert!(counters.kernel_evaluations > 100, "and it must shade something");
    let (peak, _, _) = outcome.image.max_abs();
    assert!(peak > 0.8, "the needle must render: {peak}");

    let camera = camera_of(&spec, &settings);
    let prepared = prepare::<Gaussian>(&scene_of(&[needle]), &settings);
    let mut atoms: Vec<RayAtom<f64, Gaussian>> = Vec::new();
    prepare_atoms::<f64, Gaussian>(&prepared, &camera, &settings, &mut atoms);
    assert_eq!(atoms.len(), 1);
    let to_world = camera.rotation.transpose();
    let mut sphere_survivors = 0u64;
    let mut exact_survivors = 0u64;
    for py in 0..height {
        for px in 0..width {
            let ray = to_world_ray(&to_world, camera.eye, &camera.ray(px, py));
            if !cheap_reject(
                atoms[0].mean - ray.origin,
                ray.direction,
                atoms[0].support,
                atoms[0].scale_max,
            ) {
                sphere_survivors += 1;
            }
            if intersect(&atoms[0], &ray, settings.ordering, settings.near_plane).is_some() {
                exact_survivors += 1;
            }
        }
    }
    assert_eq!(
        exact_survivors, counters.kernel_evaluations,
        "the per-pixel walk must reproduce the render's own count"
    );
    assert!(
        exact_survivors * 10 < sphere_survivors,
        "the atom is a 50:1 needle, so the conservative SPHERE the cheap reject \
         uses covers a disc of {sphere_survivors} pixels while the exact per-ray \
         test admits only the thin ellipse of {exact_survivors}. Measured against \
         the RAYS and not against the tile index, because a tile index that bins \
         the ellipsoid would otherwise flatter this ratio"
    );
}

#[test]
fn the_early_out_stops_the_walk_and_is_counted_once_per_pixel() {
    let mut settings = SharedSettings::<f64>::contract();
    settings.threads = 1;
    let spec = pinhole(600.0, [32.0, 24.0], 64, 48);
    let mut atoms = Vec::new();
    for i in 0..8u32 {
        atoms.push(Atom::isotropic(
            Vec3::new(0.0, 0.0, 4.0 + 0.2 * (i as f64)),
            0.05,
            0.95,
            Rgb::new(1.0, 1.0, 1.0),
        ));
    }
    let scene = scene_of(&atoms);
    let outcome = render_through(
        scene,
        spec,
        settings,
        RenderMethod::raycast_contract(),
        Collect::PixelsAndCounters,
    );
    let counters = outcome.counters;
    assert!(
        counters.early_outs > 20,
        "eight atoms at alpha 0.95 drive the transmittance under 1e-4 by the \
         fourth, so the early out must fire: {}",
        counters.early_outs
    );
    assert!(
        counters.early_outs <= counters.pixels_visited,
        "a pixel can stop at most ONCE; {} stops over {} pixels means Step::Stop \
         was observed and then ignored",
        counters.early_outs,
        counters.pixels_visited
    );
    assert!(
        counters.composited_hits + counters.early_outs < counters.kernel_evaluations,
        "every collected hit either composites or stops, so the two counters sum \
         to the number of STEPS taken. {} + {} against {} collected: if they are \
         equal the walk never stopped and Step::Stop was ignored",
        counters.composited_hits,
        counters.early_outs,
        counters.kernel_evaluations
    );
}

#[test]
fn the_support_radius_comes_from_the_kernel_not_from_a_literal() {
    let spec = pinhole(600.0, [32.0, 24.0], 64, 48);
    let atom = Atom::isotropic(Vec3::new(0.0, 0.0, 5.0), 0.05, 0.25, Rgb::splat(1.0));
    let scene = scene_of(&[atom]);

    let mut solved = SharedSettings::<f64>::contract();
    solved.support = SupportMode::CutoffSolved;
    let camera = camera_of(&spec, &solved);
    let prepared = prepare::<Gaussian>(&scene, &solved);
    let mut out: Vec<RayAtom<f64, Gaussian>> = Vec::new();
    prepare_atoms::<f64, Gaussian>(&prepared, &camera, &solved, &mut out);
    let expected = (2.0 * (0.25 / EPS_ALPHA).ln()).sqrt();
    assert!(
        (out[0].support - expected).abs() < 1e-13,
        "support {} must SOLVE r = sqrt(2 ln(peak/cutoff)) = {expected}, not \
         return a hardcoded 3.0",
        out[0].support
    );
    assert!(
        (out[0].support - 3.0).abs() > 0.1,
        "and this atom's answer is deliberately far from 3"
    );

    let mut fixed = SharedSettings::<f64>::contract();
    fixed.support = SupportMode::FixedSigmas(3.0);
    let mut fixed_out: Vec<RayAtom<f64, Gaussian>> = Vec::new();
    prepare_atoms::<f64, Gaussian>(&prepared, &camera, &fixed, &mut fixed_out);
    assert_eq!(fixed_out[0].support, 3.0);
    assert_eq!(fixed_out[0].support_sq, 9.0);
}

fn general_camera(w: u32, h: u32) -> CameraSpec {
    CameraSpec::look_at(
        [1.5, -0.7, 2.0],
        [0.2, 0.4, 5.5],
        [0.0, -1.0, 0.0],
        50.0,
        w,
        h,
    )
    .with_convention(PixelConvention::CornerSample)
}

fn tilted_atoms() -> Vec<Atom<f64>> {
    let mut out = Vec::new();
    for i in 0..7u32 {
        let t = i as f64;
        let half = 0.37 * t;
        out.push(Atom::new(
            Vec3::new(0.3 + 0.21 * t, 0.5 - 0.13 * t, 5.0 + 0.23 * t),
            Quat::new_wxyz(half.cos(), 0.31 * half.sin(), half.sin(), 0.17 * half.sin()),
            Vec3::new(0.07, 0.11, 0.05),
            0.55,
            Rgb::new((0.29 * t).fract(), (0.43 * t).fract(), (0.67 * t).fract()),
        ));
    }
    out
}

#[test]
fn the_world_ray_carries_both_the_eye_and_the_camera_rotation() {
    let settings = SharedSettings::<f64>::contract();
    let spec = general_camera(64, 48);
    let camera = camera_of(&spec, &settings);
    assert_ne!(
        camera.rotation,
        Mat3::<f64>::identity(),
        "this camera must actually rotate, or the test proves nothing"
    );
    assert!(
        camera.eye.length() > 1.0,
        "and it must actually sit away from the world origin"
    );

    let to_world = camera.rotation.transpose();
    for (px, py) in [(0u32, 0u32), (17, 5), (31, 23), (63, 47), (40, 11)] {
        let ray = to_world_ray(&to_world, camera.eye, &camera.ray(px, py));
        assert!(
            (ray.origin - camera.eye).length() < 1e-14,
            "a pinhole world ray starts at the eye, not at the world origin: \
             {:?} against {:?}",
            ray.origin,
            camera.eye
        );
        assert!(
            (ray.direction.length() - 1.0).abs() < 1e-15,
            "the rotation is orthonormal so the direction stays unit"
        );
        let far = ray.origin + ray.direction * 7.0;
        let back = match camera.project(camera.to_camera(far)) {
            Some(v) => v,
            None => panic!("a point down the ray must reproject"),
        };
        let sample = camera.sample_point(px, py);
        assert!(
            (back - sample).length() < 1e-10,
            "the world ray must reproject onto its own pixel: {back:?} against \
             {sample:?}"
        );
    }
}

#[test]
#[cfg(feature = "method-splat")]
fn a_rotated_and_translated_camera_agrees_with_the_splat_path() {
    let mut settings = SharedSettings::<f64>::contract();
    settings.background = Rgb::new(0.04, 0.06, 0.08);
    let spec = general_camera(128, 96);
    let scene = scene_of(&tilted_atoms());

    let splat = render_through(
        scene.clone(),
        spec.clone(),
        settings,
        RenderMethod::splat_contract(),
        Collect::Pixels,
    );
    let ray = render_through(
        scene,
        spec,
        settings,
        RenderMethod::raycast_contract(),
        Collect::Pixels,
    );
    let (peak, _, _) = splat.image.max_abs();
    assert!(peak > 0.3, "the tilted scene must actually render: {peak}");
    assert_eq!(splat.report.clamped_atoms, 0);

    let difference = match ray.image.difference(&splat.image) {
        Ok(d) => d,
        Err(e) => panic!("size mismatch: {e:?}"),
    };
    let (worst, wx, wy) = difference.max_abs();
    assert!(
        worst <= 1.5e-2,
        "under a general view transform the ray path runs in WORLD space and the \
         splat path in CAMERA space; a congruence by an orthogonal matrix \
         preserves every quadratic form, so the two must still agree. Worst \
         {worst} at ({wx},{wy})"
    );
    assert!(
        difference.rms() < 1.0e-3,
        "and the bulk of the frame must agree far more closely than the fringe: \
         rms {}",
        difference.rms()
    );
}

#[test]
fn the_agreement_bound_travels_on_every_ray_atom() {
    let settings = SharedSettings::<f64>::contract();
    let spec = pinhole(1000.0, [512.0, 512.0], 1024, 1024);
    let camera = camera_of(&spec, &settings);
    for (mean, expected) in [
        (Vec3::new(0.0, 0.0, 5.0), 1.0826822658929016e-4f64),
        (Vec3::new(2.5, 0.0, 5.0), 5.271690947020207e-3),
        (Vec3::new(5.0, 0.0, 5.0), 8.252459692131845e-3),
    ] {
        let atom = Atom::isotropic(mean, 0.05, 1.0, Rgb::splat(1.0));
        let scene = scene_of(&[atom]);
        let prepared = prepare::<Gaussian>(&scene, &settings);
        let mut out: Vec<RayAtom<f64, Gaussian>> = Vec::new();
        prepare_atoms::<f64, Gaussian>(&prepared, &camera, &settings, &mut out);
        assert_eq!(out.len(), 1);
        assert!(
            (out[0].bound - expected).abs() <= 1e-5 * expected,
            "B_atom for {mean:?} is {}, expected {expected}. On axis Omega is a \
             difference of two equal quantities, Sigma_zz/z^2 and 1/rho_e^2, so \
             the first-order term is a cancellation artefact of order 1e-10 \
             rather than the exact zero the algebra promises",
            out[0].bound
        );
        assert!(
            (out[0].rho_e_sq.sqrt() - (out[0].mean_cam.length() / 0.05)).abs() < 1e-9,
            "for an isotropic atom rho_e is |mu_cam| / sigma"
        );
        assert!((out[0].sigma_zz - 0.0025).abs() < 1e-15);
    }

    let spec = general_camera(128, 96);
    let camera = camera_of(&spec, &settings);
    let scene = scene_of(&tilted_atoms());
    let prepared = prepare::<Gaussian>(&scene, &settings);
    let mut out: Vec<RayAtom<f64, Gaussian>> = Vec::new();
    prepare_atoms::<f64, Gaussian>(&prepared, &camera, &settings, &mut out);
    assert_eq!(out.len(), scene.atoms().len());
    for (slot, atom) in scene.atoms().iter().enumerate() {
        let cam = camera.to_camera(atom.mean);
        let sigma_cam = camera.covariance_to_camera(&atom.covariance());
        assert!(
            (sigma_cam.zz - atom.covariance().zz).abs() > 1e-6,
            "this camera must actually rotate the covariance, or Sigma_zz is not \
             under test"
        );
        assert!(
            (out[slot].sigma_zz - sigma_cam.zz).abs() < 1e-14,
            "Sigma_zz is the VIEW-axis entry: {} against {}",
            out[slot].sigma_zz,
            sigma_cam.zz
        );
        let expected = atom_bound(cam, &sigma_cam, atom.opacity);
        assert!(
            expected > 0.0,
            "the tilted scene must produce a non-zero bound"
        );
        assert!(
            (out[slot].bound - expected).abs() <= 1e-9 * expected,
            "atom {slot}: bound {} against the closed form {expected}. Lambda must \
             be the CAMERA-frame inverse covariance W Lambda W^T",
            out[slot].bound
        );
        let rho_e_sq = match sigma_cam.inverse_spd() {
            Some(l) => l.quad_form(cam),
            None => panic!("the camera-frame covariance must stay SPD"),
        };
        assert!(
            (out[slot].rho_e_sq - rho_e_sq).abs() <= 1e-9 * rho_e_sq,
            "rho_e is a congruence invariant: {} against {rho_e_sq}",
            out[slot].rho_e_sq
        );
    }
}

#[test]
fn a_sphere_bound_that_straddles_the_eye_is_a_candidate_everywhere() {
    let settings = SharedSettings::<f64>::contract();
    let spec = pinhole(600.0, [32.0, 24.0], 64, 48);
    let camera = camera_of(&spec, &settings);
    let grid = match TileGrid::new(64, 48, 16) {
        Ok(g) => g,
        Err(e) => panic!("grid failed: {e:?}"),
    };

    let huge = Atom::isotropic(Vec3::new(0.0, 0.0, 0.5), 0.4, 0.9, Rgb::splat(1.0));
    let small = Atom::isotropic(Vec3::new(0.0, 0.0, 5.0), 0.05, 0.9, Rgb::splat(1.0));
    let scene = scene_of(&[huge, small]);
    let prepared = prepare::<Gaussian>(&scene, &settings);
    let mut atoms: Vec<RayAtom<f64, Gaussian>> = Vec::new();
    prepare_atoms::<f64, Gaussian>(&prepared, &camera, &settings, &mut atoms);
    assert_eq!(atoms.len(), 2);
    assert!(
        camera
            .screen_disc_bound(atoms[0].mean_cam, atoms[0].support * atoms[0].scale_max)
            .is_none(),
        "this atom's support sphere reaches back past the eye, so its screen \
         footprint has no finite bound"
    );

    let index = match TileBinIndex::build::<f64, Gaussian>(&atoms, &camera, &grid) {
        Ok(i) => i,
        Err(e) => panic!("index build failed: {e:?}"),
    };
    assert_eq!(index.unbounded_atoms(), &[0u32]);
    assert_eq!(
        index.replicated_pairs(),
        grid.tile_count() as u64,
        "an atom with no finite disc is replicated into EVERY tile, so the \
         index is O(unbounded x tiles) and the cost must be reported rather \
         than hidden: TileBinIndex::build returns Result and refuses past \
         {}",
        lights::method::raycast::prepare::UNBOUNDED_REPLICATION_LIMIT
    );
    assert!(index.pair_count() >= index.replicated_pairs());
    for ty in 0..grid.grid_y {
        for tx in 0..grid.grid_x {
            let tile = grid.tile_id(tx, ty) as usize;
            assert!(
                index.tile_candidates(tile).contains(&0u32),
                "an atom with no finite disc must be a candidate in EVERY tile, \
                 or it silently vanishes from the corners"
            );
        }
    }
    assert!(index.candidates(32, 24).contains(&1u32));
    assert!(index.resident_bytes() > 0);
}

#[test]
fn the_signed_additive_path_skips_the_sort() {
    assert!(!<Additive as BlendMode<f64>>::ORDER_DEPENDENT);
    assert!(<Additive as BlendMode<f64>>::SIGNED);

    let settings = SharedSettings::<f64>::contract();
    let spec = pinhole(500.0, [32.0, 24.0], 64, 48);
    let mut atoms = Vec::new();
    for i in 0..12u32 {
        let t = i as f64;
        atoms.push(Atom::isotropic(
            Vec3::new(0.013 * t - 0.07, 0.011 * t - 0.06, 8.0 - 0.31 * t),
            0.07,
            0.31 + 0.017 * t,
            Rgb::new(0.317 * t + 0.113, 0.211 * t + 0.077, 0.173 * t + 0.191),
        ));
    }
    let forward = scene_of(&atoms);
    let mut backward_list = atoms.clone();
    backward_list.reverse();
    let backward = scene_of(&backward_list);

    let render_signed = |scene: &Scene<f64>| -> FloatImage<f64> {
        let (caster, prepared, camera) =
            bound_raycast::<SignedTest>(scene, &spec, &settings, RayCastConfig::contract());
        let mut fb = match Framebuffer::<f64, Additive>::new(camera.width, camera.height, 16) {
            Ok(f) => f,
            Err(e) => panic!("framebuffer failed: {e:?}"),
        };
        let mut sink = ProbeSink::<f64>::counters_only(camera.width, camera.height);
        if let Err(e) = caster.render::<NoProbe>(&prepared, &mut fb, &mut sink) {
            panic!("signed render failed: {e:?}");
        }
        fb.resolve_float(Rgb::zero())
    };

    let a = render_signed(&forward);
    let b = render_signed(&backward);
    let (peak, _, _) = a.max_abs();
    assert!(peak > 0.5, "the signed scene must actually render: {peak}");

    let mut identical = 0usize;
    let mut differing = 0usize;
    let mut worst = 0.0f64;
    for (left, right) in a.pixels.iter().zip(b.pixels.iter()) {
        worst = worst.max((*left - *right).max_component().abs());
        if left.x.to_bits() == right.x.to_bits()
            && left.y.to_bits() == right.y.to_bits()
            && left.z.to_bits() == right.z.to_bits()
        {
            identical += 1;
        } else {
            differing += 1;
        }
    }
    assert!(
        worst < 1e-13,
        "the additive accumulator is order independent to within rounding: {worst}"
    );
    assert!(
        differing > 0,
        "float addition is NOT associative, so reversing the visit order must \
         move the last bits. Bit-identical images mean a depth sort ran even \
         though ORDER_DEPENDENT is false: {identical} identical of {}",
        a.pixels.len()
    );
}

#[test]
fn intersect_and_collect_agree_with_the_hand_written_chain() {
    let settings = SharedSettings::<f64>::contract();
    let spec = pinhole(600.0, [32.0, 24.0], 64, 48);
    let camera = camera_of(&spec, &settings);
    let atom = Atom::new(
        Vec3::new(0.1, -0.05, 5.0),
        Quat::new_wxyz(0.6, 0.5, -0.4, 0.47),
        Vec3::new(0.12, 0.04, 0.25),
        0.7,
        Rgb::new(0.3, 0.6, 0.9),
    );
    let scene = scene_of(&[atom]);
    let prepared = prepare::<Gaussian>(&scene, &settings);
    let mut atoms: Vec<RayAtom<f64, Gaussian>> = Vec::new();
    prepare_atoms::<f64, Gaussian>(&prepared, &camera, &settings, &mut atoms);
    assert_eq!(atoms.len(), 1);
    let ray_atom = atoms[0];

    let to_world = camera.rotation.transpose();
    assert_eq!(to_world, Mat3::<f64>::identity());
    let ray = to_world_ray(&to_world, camera.eye, &camera.ray(40, 20));
    assert!(
        (ray.direction.length() - 1.0).abs() < 1e-15,
        "the ray direction must be unit"
    );

    let hit = match intersect(
        &ray_atom,
        &ray,
        OrderingModel::CentreDepth,
        settings.near_plane,
    ) {
        Some(h) => h,
        None => panic!("the atom must be hit at this pixel"),
    };
    let delta = ray.origin - ray_atom.mean;
    let q = quadratic(&ray_atom.lambda, delta, ray.direction);
    let t_star = peak_parameter(&q);
    let r2 = residual_r2(&ray_atom.lambda, delta, ray.direction, t_star);
    let expected = (-0.5 * r2).exp();
    assert!(
        (hit.weight - expected).abs() < 1e-14,
        "weight {} against the hand-written exp(-r2/2) {expected}",
        hit.weight
    );
    assert_eq!(hit.key, ray_atom.depth, "CentreDepth keys on mu_cam.z");
    assert_eq!(hit.index, 0);
    assert!((hit.opacity - 0.7).abs() < 1e-15);

    let hit_depth = match intersect(&ray_atom, &ray, OrderingModel::HitDepth, settings.near_plane) {
        Some(h) => h,
        None => panic!("the atom must be hit at this pixel"),
    };
    assert!(
        (hit_depth.key - t_star).abs() < 1e-14,
        "HitDepth keys on t*: {} vs {t_star}",
        hit_depth.key
    );
    assert!(
        (hit_depth.key - hit.key).abs() > 1e-6,
        "and the two keys are genuinely different numbers"
    );

    let mut probe = NoProbe;
    let mut hits: Vec<Hit<f64>> = Vec::new();
    match collect_hits::<f64, Gaussian, NoProbe>(
        &atoms,
        None,
        &ray,
        &settings,
        16,
        &mut probe,
        &mut hits,
    ) {
        Ok(()) => (),
        Err(e) => panic!("collect failed: {e:?}"),
    }
    assert_eq!(hits.len(), 1);
    let mut cell = <Over as BlendMode<f64>>::init();
    composite_hits::<f64, Gaussian, NoProbe>(&hits, 0, &settings, &mut cell, &mut probe);
    let alpha = (0.7 * expected).min(0.99);
    assert!((cell.transmittance - (1.0 - alpha)).abs() < 1e-15);
}

#[test]
fn a_ray_pointing_away_from_the_atom_yields_no_hit() {
    let settings = SharedSettings::<f64>::contract();
    let spec = pinhole(600.0, [32.0, 24.0], 64, 48);
    let camera = camera_of(&spec, &settings);
    let atom = Atom::isotropic(Vec3::new(0.0, 0.0, 5.0), 0.05, 0.9, Rgb::splat(1.0));
    let scene = scene_of(&[atom]);
    let prepared = prepare::<Gaussian>(&scene, &settings);
    let mut atoms: Vec<RayAtom<f64, Gaussian>> = Vec::new();
    prepare_atoms::<f64, Gaussian>(&prepared, &camera, &settings, &mut atoms);

    let backwards = Ray {
        origin: Vec3::zero(),
        direction: Vec3::new(0.0, 0.0, -1.0),
        t_min: settings.near_plane,
        t_max: f64::INFINITY,
    };
    assert!(
        intersect(
            &atoms[0],
            &backwards,
            OrderingModel::CentreDepth,
            settings.near_plane
        )
        .is_none(),
        "the peak sits at t* = -5, far behind the near plane, and must not shade"
    );

    let sideways = Ray {
        origin: Vec3::zero(),
        direction: Vec3::new(1.0, 0.0, 0.0),
        t_min: settings.near_plane,
        t_max: f64::INFINITY,
    };
    assert!(
        intersect(
            &atoms[0],
            &sideways,
            OrderingModel::CentreDepth,
            settings.near_plane
        )
        .is_none(),
        "a ray five sigma-decades off the atom must fail the support test"
    );
}

#[test]
fn the_near_cull_uses_the_kernels_own_support_and_not_a_three_sigma_literal() {
    let mut settings = SharedSettings::<f64>::contract();
    settings.blend.alpha_cutoff = 1.0e-6;
    let expected_support = radius_for_cutoff(1.0, settings.blend.alpha_cutoff);
    assert!(
        expected_support > 5.0,
        "this test needs a kernel whose solved support is comfortably wider \
         than three sigma, or it cannot tell the two apart: {expected_support}"
    );

    let spec = pinhole(600.0, [32.0, 24.0], 64, 48);
    let camera = camera_of(&spec, &settings);
    let half = std::f64::consts::FRAC_PI_8;
    let atom = Atom::new(
        Vec3::new(0.0, 0.0, 0.25),
        Quat::new_wxyz(half.cos(), 0.0, half.sin(), 0.0),
        Vec3::new(0.03, 0.0025, 0.0025),
        1.0,
        Rgb::splat(1.0),
    );
    let scene = scene_of(&[atom]);
    let prepared = prepare::<Gaussian>(&scene, &settings);
    let mut atoms: Vec<RayAtom<f64, Gaussian>> = Vec::new();
    prepare_atoms::<f64, Gaussian>(&prepared, &camera, &settings, &mut atoms);
    assert_eq!(atoms.len(), 1, "the atom centre is beyond the near plane");
    let prepared_atom = atoms[0];
    assert!(
        (prepared_atom.support - expected_support).abs() < 1e-12,
        "the atom must carry the kernel-solved support: {} against {}",
        prepared_atom.support,
        expected_support
    );

    let mut found: Option<(Vec3<f64>, f64, f64, f64)> = None;
    for i in 0..201i32 {
        for j in 0..7i32 {
            let ax = -1.0 + 0.01 * i as f64;
            let ay = 0.02 * (j - 3) as f64;
            let direction = match Vec3::new(ax, ay, 1.0).normalize() {
                Some(d) => d,
                None => continue,
            };
            let delta = -prepared_atom.mean;
            let q = quadratic(&prepared_atom.lambda, delta, direction);
            if !(q.a > 0.0) {
                continue;
            }
            let t_star = peak_parameter(&q);
            let sigma_t = axial_sigma(&q);
            let r2 = residual_r2(&prepared_atom.lambda, delta, direction, t_star);
            if r2 > prepared_atom.support_sq {
                continue;
            }
            let three = t_star + 3.0 * sigma_t;
            let solved = t_star + prepared_atom.support * sigma_t;
            if three < settings.near_plane && solved >= settings.near_plane {
                found = Some((direction, t_star, sigma_t, r2));
            }
        }
    }

    let (direction, t_star, sigma_t, r2) = match found {
        Some(v) => v,
        None => panic!(
            "no ray in the swept fan straddles the two cull radii, so this test \
             measures nothing. The near cull is `t* + extent * sigma_t < \
             near_plane`; the whole point is that `extent` must be the atom's \
             own r_cut ({expected_support}) and not a hardcoded 3.0"
        ),
    };
    println!(
        "near cull straddle: direction {direction:?} t* {t_star} sigma_t \
         {sigma_t} r2 {r2}, three sigma reaches {} and the solved support \
         reaches {}",
        t_star + 3.0 * sigma_t,
        t_star + prepared_atom.support * sigma_t
    );

    let ray = Ray {
        origin: Vec3::zero(),
        direction,
        t_min: settings.near_plane,
        t_max: f64::INFINITY,
    };
    let hit = intersect(
        &prepared_atom,
        &ray,
        settings.ordering,
        settings.near_plane,
    );
    assert!(
        hit.is_some(),
        "the ray passes within the atom's own alpha cutoff (r = {}, r_cut = \
         {}), and the atom's support reaches past the near plane, so the \
         contribution is real. A hardcoded three-sigma extent rejects it, and \
         the splat path - which culls only on mu_cam.z - keeps it, which is a \
         cross-method divergence with no B_atom term covering it",
        r2.sqrt(),
        prepared_atom.support
    );
    let weight = match hit {
        Some(h) => h.weight,
        None => 0.0,
    };
    assert!(
        weight > settings.blend.alpha_cutoff,
        "and it must carry a weight above the cutoff it was admitted by: {weight}"
    );
}

#[test]
fn the_tile_loop_rejects_the_corners_the_disc_never_reaches() {
    let mut settings = SharedSettings::<f64>::contract();
    settings.threads = 1;
    let request = |config: RayCastConfig| lights::dispatch::RenderRequest {
        precision: lights::dispatch::Precision::F64,
        kernel: KernelFamily::Gaussian,
        method: RenderMethod::RayCast(config),
        settings,
        camera: lights::scene::demo::sphere_camera(256, 256),
        scene: lights::scene::demo::spheres_anisotropic_no_ground::<f64>(),
        collect: lights::dispatch::Collect::PixelsAndCounters,
        shapes: None,
    };
    let tiled = match lights::dispatch::render(&request(RayCastConfig::production())) {
        Ok(o) => o,
        Err(e) => panic!("tiled: {e:?}"),
    };
    let plain = match lights::dispatch::render(&request(RayCastConfig::contract())) {
        Ok(o) => o,
        Err(e) => panic!("plain: {e:?}"),
    };
    assert!(
        tiled.counters.tile_atom_pairs > 0,
        "the tiled index binned nothing, so this case measured nothing"
    );
    assert_eq!(
        tiled.image.pixels, plain.image.pixels,
        "REJECTING A TILE THE DISC NEVER REACHES MUST NOT MOVE A PIXEL. A tile is \
         only relevant if some point of it lies inside the projected disc, so the \
         clamped-point test can only remove candidates that could never have \
         contributed. A difference here means the test is not conservative and it \
         is dropping real work"
    );
}

#[test]
fn the_tile_binner_bounds_the_ellipsoid_and_not_its_sphere() {
    let mut settings = SharedSettings::<f64>::contract();
    settings.threads = 1;
    let width = 512u32;
    let height = 512u32;
    let spec = pinhole(600.0, [256.0, 256.0], width, height);
    let needle = Atom::new(
        Vec3::new(0.0, 0.0, 5.0),
        Quat::identity(),
        Vec3::new(0.30, 0.006, 0.006),
        0.9,
        Rgb::new(1.0, 1.0, 1.0),
    );
    let tiled = render_through(
        scene_of(&[needle]),
        spec.clone(),
        settings,
        RenderMethod::RayCast(RayCastConfig {
            accel: AccelKind::TileBins,
            tile: 16,
            max_hits_per_pixel: 4096,
        }),
        Collect::PixelsAndCounters,
    );
    let plain = render_through(
        scene_of(&[needle]),
        spec,
        settings,
        RenderMethod::RayCast(RayCastConfig {
            accel: AccelKind::BruteForce,
            tile: 16,
            max_hits_per_pixel: 4096,
        }),
        Collect::PixelsAndCounters,
    );
    println!(
        "NEEDLE pairs={} pixels_visited={} candidates={} kernel_evals={}",
        tiled.counters.tile_atom_pairs,
        tiled.counters.pixels_visited,
        tiled.counters.candidates_tested,
        tiled.counters.kernel_evaluations
    );
    assert_eq!(
        tiled.image.pixels, plain.image.pixels,
        "BINNING BY THE ELLIPSOID MUST NOT MOVE A PIXEL: the tile index is a \
         prune, and a prune that changes the image is dropping real work"
    );
    assert!(
        tiled.counters.tile_atom_pairs > 0,
        "the tiled index binned nothing, so this case measured nothing"
    );
    assert!(
        tiled.counters.tile_atom_pairs <= 64,
        "this atom is a 50:1 needle lying along x: its support ELLIPSOID projects \
         to a band of 32 tiles, while the support SPHERE that contains it \
         projects to a disc of 600 (measured on the unfixed tree, 2026-09-15). \
         {} tile-atom pairs means the binner is still bounding the sphere",
        tiled.counters.tile_atom_pairs
    );
}
