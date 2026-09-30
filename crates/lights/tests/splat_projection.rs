#![cfg(feature = "method-splat")]
use lights::agreement::{atom_bound, atom_bound_terms, atom_bound_with_lambda};
use lights::blend::{Additive, BlendMode, Over};
use lights::camera::{Camera, CameraSpec};
use lights::dispatch::{Collect, Precision, RenderOutcome, RenderRequest, render};
use lights::error::LightsError;
use lights::image::Framebuffer;
use lights::kernel::frame::{RayFrame, ScreenFrame};
use lights::kernel::gaussian::{Gaussian, radius_for_cutoff};
use lights::kernel::{CurveShape, Kernel, KernelFamily, KernelId};
use lights::method::splat::bin::{bin_and_sort, depth_bits, scan_tile_ranges};
use lights::method::splat::execute::{BatchEntry, entry_of, gather_batch, shade_pixel};
use lights::method::splat::project::{
    CullReason, ProjectOutcome, ScreenAtom, determinant_guard, project_atom, screen_covariance,
    tight_half_extent,
};
use lights::method::splat::{SortKind, SplatConfig, SplatRaster};
use lights::method::{ImageFormation, MethodId, RenderMethod};
use lights::numerics::float::FloatingPoint;
use lights::numerics::matrix::Mat3;
use lights::numerics::quaternion::Quat;
use lights::numerics::symmetric::{Sym2, Sym3};
use lights::numerics::vector::{Rgb, Vec2, Vec3};
use lights::probe::{Counter, NoProbe, ProbeSink};
use lights::scene::prepare::PreparedScene;
use lights::scene::{Atom, Scene};
use lights::settings::{
    MIN_SCALE, OrderingModel, PixelConvention, SharedSettings, SupportMode,
};
use lights::tiling::TileGrid;

const FLOOR: f64 = MIN_SCALE;

fn intrinsic_camera(focal: [f64; 2], principal: [f64; 2], w: u32, h: u32) -> CameraSpec {
    CameraSpec::look_at([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0], 45.0, w, h)
        .with_intrinsics(focal, principal)
        .with_convention(PixelConvention::CornerSample)
}

fn one_atom_scene(atom: Atom<f64>) -> Scene<f64> {
    let mut scene = Scene::<f64>::new();
    match scene.push(atom, FLOOR) {
        Ok(()) => (),
        Err(e) => panic!("scene rejected the test atom: {e:?}"),
    }
    scene
}

fn prepare(scene: &Scene<f64>, settings: &SharedSettings<f64>) -> PreparedScene<f64, Gaussian> {
    match PreparedScene::<f64, Gaussian>::build(scene, &[], settings) {
        Ok(p) => p,
        Err(e) => panic!("prepare failed: {e:?}"),
    }
}

fn camera_of(spec: &CameraSpec, settings: &SharedSettings<f64>) -> Camera<f64> {
    match Camera::<f64>::from_spec(spec, settings.near_plane) {
        Ok(c) => c,
        Err(e) => panic!("camera failed: {e:?}"),
    }
}

fn project_one(
    atom: Atom<f64>,
    spec: &CameraSpec,
    settings: &SharedSettings<f64>,
) -> (ProjectOutcome<f64, Gaussian>, bool) {
    let scene = one_atom_scene(atom);
    let prepared = prepare(&scene, settings);
    let camera = camera_of(spec, settings);
    project_atom::<f64, Gaussian>(prepared.atom(0), &camera, settings)
}

fn visible(outcome: &ProjectOutcome<f64, Gaussian>) -> &ScreenAtom<f64, Gaussian> {
    match outcome.visible() {
        Some(a) => a,
        None => panic!("expected a visible atom, got {:?}", outcome.cull_reason()),
    }
}

fn bound_raster(
    scene: &Scene<f64>,
    spec: &CameraSpec,
    settings: &SharedSettings<f64>,
    config: SplatConfig,
) -> (
    SplatRaster<f64, Gaussian>,
    PreparedScene<f64, Gaussian>,
    Camera<f64>,
) {
    let prepared = prepare(scene, settings);
    let camera = camera_of(spec, settings);
    let mut raster = match SplatRaster::<f64, Gaussian>::new(config, settings) {
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

fn render_scene(
    scene: Scene<f64>,
    spec: CameraSpec,
    settings: SharedSettings<f64>,
    collect: Collect,
) -> RenderOutcome {
    let request = RenderRequest {
        precision: Precision::F64,
        kernel: KernelFamily::Gaussian,
        method: RenderMethod::SplatRaster(SplatConfig::contract()),
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

fn synthetic_atom(
    centre: Vec2<f64>,
    half: Vec2<f64>,
    depth: f64,
    index: u32,
    colour: Rgb<f64>,
    opacity: f64,
) -> ScreenAtom<f64, Gaussian> {
    let cov = Sym2::new(half.x * half.x / 9.0, 0.0, half.y * half.y / 9.0);
    let frame = match ScreenFrame::from_covariance(cov) {
        Some(f) => f,
        None => panic!("synthetic covariance is not SPD"),
    };
    ScreenAtom {
        centre,
        depth,
        frame,
        prepared: (),
        half_extent: half,
        support_sq: 9.0,
        opacity,
        colour,
        bound: 0.0,
        index,
    }
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

plain_kernel!(Unprojectable, Over, 900, "unprojectable", false);
plain_kernel!(SignedTest, Additive, 901, "signed-test", true);

#[test]
fn tv_s4_perspective_shear() {
    let settings = SharedSettings::<f64>::contract();
    let spec = intrinsic_camera([500.0, 500.0], [320.0, 240.0], 640, 480);
    let camera = camera_of(&spec, &settings);
    let cam_point = Vec3::new(1.0, 2.0, 10.0);
    let sigma_cam = Sym3::new(0.04, 0.0, 0.0, 0.01, 0.0, 0.0025);

    let (raw, fired) = screen_covariance(&sigma_cam, &camera, cam_point);
    assert!(!fired, "an atom inside the frustum must not clamp");
    assert!((raw.xx - 100.0625).abs() < 1e-12, "raw.xx {}", raw.xx);
    assert!((raw.xy - 0.125).abs() < 1e-12, "raw.xy {}", raw.xy);
    assert!((raw.yy - 25.25).abs() < 1e-12, "raw.yy {}", raw.yy);

    let (clamped, _) = camera.clamp_for_jacobian(cam_point);
    let mut flat = camera.jacobian(clamped);
    flat.m[0][2] = 0.0;
    flat.m[1][2] = 0.0;
    let without_shear = sigma_cam.project_2x3(&flat);
    assert_eq!(
        without_shear.xy, 0.0,
        "Sigma01 comes ENTIRELY from the third column of J; dropping the \
         perspective shear zeroes it, and the term vanishes on the optical axis \
         so a centred scene cannot detect the bug"
    );
    assert!((without_shear.xx - 100.0).abs() < 1e-12);
}

#[test]
fn tv_s5_dilation_conic_and_three_probes() {
    let settings = SharedSettings::<f64>::reference();
    assert_eq!(settings.support, SupportMode::FixedSigmas(3.0));
    assert!((settings.screen_dilation - 0.3).abs() < 1e-15);

    let spec = intrinsic_camera([500.0, 500.0], [320.0, 240.0], 640, 480);
    let atom = Atom::new(
        Vec3::new(1.0, 2.0, 10.0),
        Quat::identity(),
        Vec3::new(0.2, 0.1, 0.05),
        1.0,
        Rgb::new(1.0, 1.0, 1.0),
    );
    let (outcome, _) = project_one(atom, &spec, &settings);
    let screen = visible(&outcome);

    assert!((screen.centre.x - 370.0).abs() < 1e-12, "{}", screen.centre.x);
    assert!((screen.centre.y - 340.0).abs() < 1e-12, "{}", screen.centre.y);
    assert!((screen.frame.cov.xx - 100.3625).abs() < 1e-12);
    assert!((screen.frame.cov.xy - 0.125).abs() < 1e-12);
    assert!((screen.frame.cov.yy - 25.55).abs() < 1e-12);
    assert!(
        (screen.frame.cov.determinant() - 2564.24625).abs() < 1e-9,
        "det {}",
        screen.frame.cov.determinant()
    );
    assert!((screen.frame.conic.xx - 0.0099639416456).abs() < 1e-12);
    assert!((screen.frame.conic.xy + 0.0000487472683).abs() < 1e-13);
    assert!((screen.frame.conic.yy - 0.0391391817381).abs() < 1e-12);

    let probes = [
        (Vec2::new(10.0, 0.0), 0.6076251709465168),
        (Vec2::new(0.0, 5.0), 0.6130923071681672),
        (Vec2::new(3.0, 2.0), 0.8844196451542083),
    ];
    for (offset, expected) in probes {
        let whitened = screen.frame.whiten(offset);
        let weight = <Gaussian as Kernel<f64>>::eval_screen(&screen.prepared, whitened);
        assert!(
            (weight - expected).abs() < 1e-12,
            "probe {offset:?}: got {weight}, want {expected}"
        );
        let quadratic = screen.frame.conic.quad_form(offset);
        assert!(
            (whitened.length_squared() - quadratic).abs() < 1e-9,
            "the whitened radius and the conic quadratic form must agree, \
             which is where the factor of 2 on the cross term lives"
        );
    }

    let dropped_two = -0.5
        * (screen.frame.conic.xx * 9.0
            + screen.frame.conic.xy * 3.0 * 2.0
            + screen.frame.conic.yy * 4.0);
    assert!(
        (dropped_two.exp() - 0.8844196451542083).abs() > 1e-9,
        "the (3,2) probe is the one that exercises the factor of 2"
    );

    assert!((screen.half_extent.x - 30.0543258118).abs() < 1e-9);
    assert!((screen.half_extent.y - 15.1641023473).abs() < 1e-9);

    let centred = Vec2::new(320.0, 240.0);
    let x0 = (centred.x - screen.half_extent.x).floor();
    let x1 = (centred.x + screen.half_extent.x).ceil();
    let y0 = (centred.y - screen.half_extent.y).floor();
    let y1 = (centred.y + screen.half_extent.y).ceil();
    assert_eq!((x0, x1, y0, y1), (289.0, 351.0, 224.0, 256.0));
}

#[test]
fn tv_s8_anisotropic_rotated_assert_conic_not_radius() {
    let settings = SharedSettings::<f64>::reference();
    let spec = intrinsic_camera([1000.0, 1000.0], [400.0, 300.0], 800, 600);
    let half = std::f64::consts::FRAC_1_SQRT_2;
    let atom = Atom::new(
        Vec3::new(0.0, 0.0, 5.0),
        Quat::new_wxyz(half, 0.0, 0.0, half),
        Vec3::new(0.1, 0.02, 0.02),
        1.0,
        Rgb::new(1.0, 1.0, 1.0),
    );
    let scene = one_atom_scene(atom);
    let sigma = scene.atoms()[0].covariance();
    assert!((sigma.xx - 0.0004).abs() < 1e-15, "{}", sigma.xx);
    assert!((sigma.yy - 0.01).abs() < 1e-15, "{}", sigma.yy);
    assert!((sigma.zz - 0.0004).abs() < 1e-15);
    assert!(sigma.xy.abs() < 1e-15);

    let (outcome, _) = project_one(atom, &spec, &settings);
    let screen = visible(&outcome);
    assert!((screen.frame.cov.xx - 16.3).abs() < 1e-10);
    assert!((screen.frame.cov.yy - 400.3).abs() < 1e-10);
    assert!((screen.frame.conic.xx - 0.06134969325153374).abs() < 1e-14);
    assert!((screen.frame.conic.yy - 0.002498126405196103).abs() < 1e-15);

    let along_y = <Gaussian as Kernel<f64>>::eval_screen(
        &screen.prepared,
        screen.frame.whiten(Vec2::new(0.0, 20.0)),
    );
    let along_x = <Gaussian as Kernel<f64>>::eval_screen(
        &screen.prepared,
        screen.frame.whiten(Vec2::new(20.0, 0.0)),
    );
    assert!((along_y - 0.60675798083923).abs() < 1e-12, "{along_y}");
    assert!((along_x - 4.690653480707e-06).abs() < 1e-16, "{along_x}");
    assert!(
        along_x < settings.blend.alpha_cutoff,
        "the across-axis probe must fall below the alpha cutoff"
    );

    let s_times_r = Sym2::new(400.0 + 0.3, 0.0, 16.0 + 0.3);
    let (correct_hi, _) = screen.frame.cov.eigenvalues();
    let (wrong_hi, _) = s_times_r.eigenvalues();
    assert!(
        (correct_hi - wrong_hi).abs() < 1e-9,
        "the S*R covariance has the SAME largest eigenvalue, so a radius \
         assertion passes against the bug: {correct_hi} vs {wrong_hi}"
    );
    let wrong_conic = match s_times_r.inverse() {
        Some(c) => c,
        None => panic!("invertible"),
    };
    assert!(
        (wrong_conic.xx - screen.frame.conic.xx).abs() > 1e-3,
        "the conic DOES discriminate the covariance order"
    );
}

#[test]
fn tv_s9_third_jacobian_row_is_irrelevant() {
    let settings = SharedSettings::<f64>::contract();
    let angle = (30.0f64).to_radians();
    let spec = CameraSpec::look_at(
        [0.0, 0.0, 0.0],
        [-angle.sin(), 0.0, angle.cos()],
        [0.0, -1.0, 0.0],
        45.0,
        800,
        600,
    )
    .with_intrinsics([600.0, 600.0], [400.0, 300.0]);
    let camera = camera_of(&spec, &settings);
    let sigma_world = Sym3::<f64>::new(0.025, 0.015, 0.0, 0.025, 0.0, 0.0025);
    let sigma_cam = camera.covariance_to_camera(&sigma_world);
    let mu = camera.to_camera(Vec3::new(0.5, 0.3, 8.0));
    let (clamped, _) = camera.clamp_for_jacobian(mu);
    let j = camera.jacobian(clamped);
    let two_by_three = sigma_cam.project_2x3(&j);

    let planar = Mat3::from_rows(j.row(0), j.row(1), Vec3::new(0.0, 0.0, 1.0));
    let radial_dir = match mu.normalize() {
        Some(d) => d,
        None => panic!("non-zero mu"),
    };
    let radial = Mat3::from_rows(j.row(0), j.row(1), radial_dir);
    let a = sigma_cam.congruence(&planar);
    let b = sigma_cam.congruence(&radial);

    assert!((a.xx - b.xx).abs() < 1e-15);
    assert!((a.xy - b.xy).abs() < 1e-15);
    assert!((a.yy - b.yy).abs() < 1e-15);
    assert!((a.xx - two_by_three.xx).abs() < 1e-15);
    assert!((a.xy - two_by_three.xy).abs() < 1e-15);
    assert!((a.yy - two_by_three.yy).abs() < 1e-15);
    assert!(
        (two_by_three.xx - 289.7033860835).abs() < 1e-7,
        "{}",
        two_by_three.xx
    );
    assert!(
        (a.zz - b.zz).abs() > 1e-6,
        "the third row DOES change the third entry; it just never reaches the screen"
    );
}

#[test]
fn tv_s10_end_to_end_projection() {
    let settings = SharedSettings::<f64>::reference();
    let angle = (30.0f64).to_radians();
    let spec = CameraSpec::look_at(
        [0.0, 0.0, 0.0],
        [-angle.sin(), 0.0, angle.cos()],
        [0.0, -1.0, 0.0],
        45.0,
        800,
        600,
    )
    .with_intrinsics([600.0, 600.0], [400.0, 300.0]);
    let quarter = (22.5f64).to_radians();
    let atom = Atom::new(
        Vec3::new(0.5, 0.3, 8.0),
        Quat::new_wxyz(quarter.cos(), 0.0, 0.0, quarter.sin()),
        Vec3::new(0.2, 0.1, 0.05),
        1.0,
        Rgb::new(1.0, 1.0, 1.0),
    );
    let (outcome, fired) = project_one(atom, &spec, &settings);
    assert!(!fired);
    let screen = visible(&outcome);

    assert!((screen.depth - 6.6782032303).abs() < 1e-9);
    assert!((screen.centre.x - 798.2819224604).abs() < 1e-7);
    assert!((screen.centre.y - 326.9533576313).abs() < 1e-7);
    assert!(
        (screen.frame.cov.determinant() - 37565.1641167934).abs() < 1e-4,
        "det {}",
        screen.frame.cov.determinant()
    );
    assert!((screen.frame.cov.xy.abs() - 150.5342951464).abs() < 1e-7);
    assert!((screen.frame.conic.xx - 0.0055283276).abs() < 1e-9);
    assert!((screen.frame.conic.xy + 0.0040072844).abs() < 1e-9);
    assert!((screen.frame.conic.yy - 0.0077200085).abs() < 1e-9);
    assert!((screen.half_extent.x - 51.08845735).abs() < 1e-7);
    assert!((screen.half_extent.y - 43.23254339).abs() < 1e-7);

    let x0 = (screen.centre.x - screen.half_extent.x).floor();
    let x1 = (screen.centre.x + screen.half_extent.x).ceil();
    let y0 = (screen.centre.y - screen.half_extent.y).floor();
    let y1 = (screen.centre.y + screen.half_extent.y).ceil();
    assert_eq!((x0, x1, y0, y1), (747.0, 850.0, 283.0, 371.0));
}

#[test]
fn tv_s13_bbox_tightness() {
    let rotated = Sym2::new(50.5, 49.5, 50.5);
    let (hi, lo) = rotated.eigenvalues();
    assert!((hi - 100.0).abs() < 1e-12, "{hi}");
    assert!((lo - 1.0).abs() < 1e-12, "{lo}");
    let tight = tight_half_extent(rotated, 3.0);
    assert!((tight.x - 21.319005605327842).abs() < 1e-12);
    let eigen_radius = 3.0 * hi.sqrt();
    assert!((eigen_radius - 30.0).abs() < 1e-12);
    let ratio = (2.0 * eigen_radius).powi(2) / (2.0 * tight.x * 2.0 * tight.y);
    assert!((ratio - 1.9801980198019804).abs() < 1e-12, "{ratio}");

    let axis_aligned = Sym2::new(100.0, 0.0, 1.0);
    let tight2 = tight_half_extent(axis_aligned, 3.0);
    let (hi2, lo2) = axis_aligned.eigenvalues();
    let ratio2 = (2.0 * 3.0 * hi2.sqrt()).powi(2) / (2.0 * tight2.x * 2.0 * tight2.y);
    assert!(
        (ratio2 - 10.0).abs() < 1e-12,
        "the worst case is the axis-aligned elongated atom, at exactly \
         sqrt(kappa) = {}: {ratio2}",
        (hi2 / lo2).sqrt()
    );
}

#[test]
fn tv_s14a_f32_cancellation_is_culled() {
    let pathological = Sym2::<f64>::new(1.0e8, 1.0e8, 1.0e8);
    assert_eq!(pathological.determinant(), 0.0);
    assert!(
        !determinant_guard(pathological),
        "an exactly cancelling determinant must cull"
    );
    let f32_version = Sym2::<f32>::new(1.0e8, 1.0e8, 1.0e8);
    assert!(!determinant_guard(f32_version));
    assert!(ScreenFrame::from_covariance(pathological).is_none());
}

#[test]
fn tv_s14b_nan_is_culled_by_the_negated_guard() {
    let poisoned = Sym2::<f64>::new(f64::NAN, 0.0, 1.0);
    assert!(
        !determinant_guard(poisoned),
        "only the negated form !(det > eps) is false on NaN"
    );
    let det = poisoned.determinant();
    assert!(det.is_nan());
    assert!(
        !(det == 0.0),
        "a bare equality guard falls through on NaN and the atom renders as NaN"
    );
    assert!(
        !(det <= 1.0e-9),
        "`det <= eps` also falls through on NaN"
    );
    assert!(ScreenFrame::from_covariance(poisoned).is_none());
}

#[test]
fn tv_s14c_zero_scale_atom_is_not_culled() {
    let settings = SharedSettings::<f64>::reference();
    let spec = intrinsic_camera([1000.0, 1000.0], [400.0, 300.0], 800, 600);
    let atom = Atom::new(
        Vec3::new(0.0, 0.0, 5.0),
        Quat::identity(),
        Vec3::splat(0.0),
        1.0,
        Rgb::new(1.0, 1.0, 1.0),
    );
    let (outcome, _) = project_one(atom, &spec, &settings);
    let screen = visible(&outcome);
    assert!(
        (screen.frame.cov.determinant() - 0.09).abs() < 1e-9,
        "after dilation the determinant is 0.09, far above any sane epsilon, \
         so the determinant guard CANNOT be relied on to remove degenerate atoms"
    );
    assert!((screen.half_extent.x - 1.6431676725154982).abs() < 1e-9);
    assert!((screen.half_extent.y - 1.6431676725154982).abs() < 1e-9);
}

#[test]
fn tv_s14d_near_plane_is_inclusive() {
    let settings = SharedSettings::<f64>::reference();
    let spec = intrinsic_camera([1000.0, 1000.0], [400.0, 300.0], 800, 600);
    let at_plane = Atom::isotropic(
        Vec3::new(0.0, 0.0, settings.near_plane),
        0.05,
        1.0,
        Rgb::new(1.0, 1.0, 1.0),
    );
    let (outcome, _) = project_one(at_plane, &spec, &settings);
    assert_eq!(outcome.cull_reason(), Some(CullReason::Near));

    let just_past = Atom::isotropic(
        Vec3::new(0.0, 0.0, settings.near_plane + 1.0e-9),
        0.05,
        1.0,
        Rgb::new(1.0, 1.0, 1.0),
    );
    let (outcome, _) = project_one(just_past, &spec, &settings);
    assert_eq!(outcome.cull_reason(), None);

    let behind = Atom::isotropic(Vec3::new(0.0, 0.0, -3.0), 0.05, 1.0, Rgb::new(1.0, 1.0, 1.0));
    let (outcome, _) = project_one(behind, &spec, &settings);
    assert_eq!(outcome.cull_reason(), Some(CullReason::Near));
    assert!(
        depth_bits(-3.0f64) > depth_bits(1.0e30f64),
        "a negative depth sorts after every positive one, which is why the \
         near cull must precede the sort"
    );
}

#[test]
fn tv_s18_tile_rectangle_edge_cases() {
    let grid = match TileGrid::new(800, 600, 16) {
        Ok(g) => g,
        Err(e) => panic!("{e:?}"),
    };
    let mut keys = Vec::new();
    let mut ranges = Vec::new();
    let white = Rgb::new(1.0, 1.0, 1.0);

    let straddling = [synthetic_atom(
        Vec2::new(5.0, 5.0),
        Vec2::new(10.0, 10.0),
        5.0,
        0,
        white,
        1.0,
    )];
    bin_and_sort::<f64, Gaussian>(
        &straddling,
        &grid,
        true,
        SortKind::GlobalSort,
        &mut keys,
        &mut ranges,
    );
    assert_eq!(keys.len(), 1, "a splat spanning x in [-5,15] covers one tile");
    assert_eq!(keys[0].0, grid.tile_id(0, 0));

    let rightmost = [synthetic_atom(
        Vec2::new(799.0, 300.0),
        Vec2::new(2.0, 2.0),
        5.0,
        0,
        white,
        1.0,
    )];
    bin_and_sort::<f64, Gaussian>(
        &rightmost,
        &grid,
        true,
        SortKind::GlobalSort,
        &mut keys,
        &mut ranges,
    );
    let columns: Vec<i32> = keys
        .iter()
        .map(|k| (k.0 as i32) % grid.grid_x)
        .collect();
    assert!(
        columns.contains(&(grid.grid_x - 1)),
        "the rightmost tile column must be INCLUDED; a plain floor for the \
         exclusive maximum drops the last column of every atom"
    );

    let offscreen = [synthetic_atom(
        Vec2::new(-100.0, 300.0),
        Vec2::new(10.0, 10.0),
        5.0,
        0,
        white,
        1.0,
    )];
    let pairs = bin_and_sort::<f64, Gaussian>(
        &offscreen,
        &grid,
        true,
        SortKind::GlobalSort,
        &mut keys,
        &mut ranges,
    );
    assert_eq!(pairs, 0);
    assert!(keys.is_empty(), "an off-screen atom emits no keys at all");
    assert_eq!(ranges.len(), grid.tile_count());
    assert!(ranges.iter().all(|r| *r == (0, 0)));
}

#[test]
fn tile_ranges_partition_the_sorted_keys() {
    let grid = match TileGrid::new(64, 64, 16) {
        Ok(g) => g,
        Err(e) => panic!("{e:?}"),
    };
    let keys = vec![(0u32, 1u64, 0u32), (0, 2, 1), (3, 0, 2), (7, 5, 3), (7, 6, 4)];
    let mut ranges = Vec::new();
    scan_tile_ranges(&keys, grid.tile_count(), &mut ranges);
    assert_eq!(ranges[0], (0, 2));
    assert_eq!(ranges[1], (0, 0));
    assert_eq!(ranges[3], (2, 3));
    assert_eq!(ranges[7], (3, 5));
    let covered: u32 = ranges.iter().map(|r| r.1 - r.0).sum();
    assert_eq!(covered as usize, keys.len());
}

#[test]
fn depth_sort_orders_front_to_back_and_the_two_sort_kinds_agree() {
    let grid = match TileGrid::new(32, 32, 16) {
        Ok(g) => g,
        Err(e) => panic!("{e:?}"),
    };
    let white = Rgb::new(1.0, 1.0, 1.0);
    let screen = [
        synthetic_atom(Vec2::new(8.0, 8.0), Vec2::splat(3.0), 9.0, 0, white, 1.0),
        synthetic_atom(Vec2::new(8.0, 8.0), Vec2::splat(3.0), 2.0, 1, white, 1.0),
        synthetic_atom(Vec2::new(8.0, 8.0), Vec2::splat(3.0), 5.0, 2, white, 1.0),
    ];
    let mut global_keys = Vec::new();
    let mut global_ranges = Vec::new();
    bin_and_sort::<f64, Gaussian>(
        &screen,
        &grid,
        true,
        SortKind::GlobalSort,
        &mut global_keys,
        &mut global_ranges,
    );
    let order: Vec<u32> = global_keys.iter().map(|k| k.2).collect();
    assert_eq!(order, vec![1, 2, 0], "front to back is ascending camera z");

    let mut tiled_keys = Vec::new();
    let mut tiled_ranges = Vec::new();
    bin_and_sort::<f64, Gaussian>(
        &screen,
        &grid,
        true,
        SortKind::PerTileSort,
        &mut tiled_keys,
        &mut tiled_ranges,
    );
    assert_eq!(tiled_keys, global_keys, "the two sort kinds must agree exactly");
    assert_eq!(tiled_ranges, global_ranges);

    let mut unsorted_keys = Vec::new();
    let mut unsorted_ranges = Vec::new();
    bin_and_sort::<f64, Gaussian>(
        &screen,
        &grid,
        false,
        SortKind::GlobalSort,
        &mut unsorted_keys,
        &mut unsorted_ranges,
    );
    let scene_order: Vec<u32> = unsorted_keys.iter().map(|k| k.2).collect();
    assert_eq!(
        scene_order,
        vec![0, 1, 2],
        "an order-independent blend skips the depth sort entirely"
    );
}

#[test]
fn tv_s15_two_atom_composite_through_the_renderer() {
    let settings = SharedSettings::<f64>::reference();
    let spec = intrinsic_camera([1000.0, 1000.0], [8.0, 8.0], 17, 17);
    let mut scene = Scene::<f64>::new();
    let red = Atom::isotropic(Vec3::new(0.0, 0.0, 5.0), 0.05, 0.5, Rgb::new(1.0, 0.0, 0.0));
    let blue = Atom::isotropic(Vec3::new(0.0, 0.0, 10.0), 0.10, 0.5, Rgb::new(0.0, 0.0, 1.0));
    match (scene.push(red, FLOOR), scene.push(blue, FLOOR)) {
        (Ok(()), Ok(())) => (),
        _ => panic!("scene rejected an atom"),
    }
    let outcome = render_scene(scene.clone(), spec.clone(), settings, Collect::Pixels);
    let pixel = outcome.image.get(8, 8);
    assert!((pixel.x - 0.5).abs() < 1e-12, "red {}", pixel.x);
    assert!(pixel.y.abs() < 1e-15);
    assert!((pixel.z - 0.25).abs() < 1e-12, "blue {}", pixel.z);

    let mut swapped = Scene::<f64>::new();
    let far_red = Atom::isotropic(Vec3::new(0.0, 0.0, 10.0), 0.10, 0.5, Rgb::new(1.0, 0.0, 0.0));
    let near_blue = Atom::isotropic(Vec3::new(0.0, 0.0, 5.0), 0.05, 0.5, Rgb::new(0.0, 0.0, 1.0));
    match (swapped.push(far_red, FLOOR), swapped.push(near_blue, FLOOR)) {
        (Ok(()), Ok(())) => (),
        _ => panic!("scene rejected an atom"),
    }
    let swapped_outcome = render_scene(swapped, spec.clone(), settings, Collect::Pixels);
    let swapped_pixel = swapped_outcome.image.get(8, 8);
    assert!((swapped_pixel.x - 0.25).abs() < 1e-12, "{}", swapped_pixel.x);
    assert!((swapped_pixel.z - 0.5).abs() < 1e-12, "{}", swapped_pixel.z);

    let mut lit = settings;
    lit.background = Rgb::new(1.0, 1.0, 1.0);
    let on_white = render_scene(scene, spec, lit, Collect::Pixels);
    let white_pixel = on_white.image.get(8, 8);
    assert!((white_pixel.x - 0.75).abs() < 1e-12, "{}", white_pixel.x);
    assert!(
        (white_pixel.y - 0.25).abs() < 1e-12,
        "the green channel is exactly the leftover transmittance, which proves \
         the background is weighted and not added unconditionally: {}",
        white_pixel.y
    );
    assert!((white_pixel.z - 0.5).abs() < 1e-12, "{}", white_pixel.z);
}

#[test]
fn background_reaches_the_resolved_image_at_uncovered_pixels() {
    let settings = SharedSettings::<f64>::reference();
    let mut lit = settings;
    lit.background = Rgb::new(0.25, 0.5, 0.75);
    let spec = intrinsic_camera([1000.0, 1000.0], [8.0, 8.0], 33, 33);
    let scene = one_atom_scene(Atom::isotropic(
        Vec3::new(0.0, 0.0, 5.0),
        0.02,
        1.0,
        Rgb::new(1.0, 0.0, 0.0),
    ));
    let outcome = render_scene(scene, spec, lit, Collect::Pixels);
    let far = outcome.image.get(32, 32);
    assert!((far.x - 0.25).abs() < 1e-15, "{}", far.x);
    assert!((far.y - 0.5).abs() < 1e-15, "{}", far.y);
    assert!((far.z - 0.75).abs() < 1e-15, "{}", far.z);
    let centre = outcome.image.get(8, 8);
    assert!(
        centre.x > 0.9 && centre.y < 0.1,
        "the covered pixel must be dominated by the atom: {centre:?}"
    );
}

#[test]
fn early_termination_stops_the_walk_and_costs_under_the_cutoff() {
    let mut settings = SharedSettings::<f64>::contract();
    settings.threads = 1;
    let spec = intrinsic_camera([1000.0, 1000.0], [8.0, 8.0], 17, 17);
    let mut scene = Scene::<f64>::new();
    for i in 0..40 {
        let atom = Atom::isotropic(
            Vec3::new(0.0, 0.0, 5.0 + (i as f64) * 0.1),
            0.05,
            0.5,
            Rgb::new(1.0, 1.0, 1.0),
        );
        match scene.push(atom, FLOOR) {
            Ok(()) => (),
            Err(e) => panic!("{e:?}"),
        }
    }
    let with_cutoff = render_scene(
        scene.clone(),
        spec.clone(),
        settings,
        Collect::PixelsAndCounters,
    );
    assert!(
        with_cutoff.counters.early_outs > 0,
        "the early-out must actually fire"
    );
    assert!(
        with_cutoff.counters.early_outs <= with_cutoff.counters.pixels_visited,
        "a pixel can stop at most once: {} stops over {} pixels means Step::Stop \
         was observed and then ignored",
        with_cutoff.counters.early_outs,
        with_cutoff.counters.pixels_visited
    );

    let mut exhaustive = settings;
    exhaustive.blend.transmittance_cutoff = 0.0;
    let without = render_scene(scene, spec, exhaustive, Collect::PixelsAndCounters);
    assert_eq!(without.counters.early_outs, 0);
    let difference = match with_cutoff.image.difference(&without.image) {
        Ok(d) => d,
        Err(e) => panic!("{e:?}"),
    };
    let (worst, _, _) = difference.max_abs();
    assert!(
        worst < 1.0e-4,
        "the exact early-out's error is bounded by eps_T: {worst}"
    );
    assert!(
        with_cutoff.counters.composited_hits < without.counters.composited_hits,
        "the early-out must save work, not merely be present"
    );
}

#[test]
fn the_support_radius_comes_from_the_kernel_not_from_a_literal() {
    let mut settings = SharedSettings::<f64>::contract();
    assert_eq!(settings.support, SupportMode::CutoffSolved);
    let spec = intrinsic_camera([1000.0, 1000.0], [400.0, 300.0], 800, 600);
    let make = |opacity: f64| {
        Atom::isotropic(
            Vec3::new(0.0, 0.0, 5.0),
            0.05,
            opacity,
            Rgb::new(1.0, 1.0, 1.0),
        )
    };

    for opacity in [1.0, 0.5, 0.25, 0.10] {
        let (outcome, _) = project_one(make(opacity), &spec, &settings);
        let screen = visible(&outcome);
        let expected = (2.0 * (opacity / settings.blend.alpha_cutoff).ln()).sqrt();
        let radius = screen.support_sq.sqrt();
        assert!(
            (radius - expected).abs() < 1e-12,
            "opacity {opacity}: support {radius} must solve the cutoff equation \
             ({expected}), never be a hardcoded 3.0"
        );
        assert!(
            (radius - 3.0).abs() > 1e-3 || (opacity - 1.0).abs() < 1e-9,
            "a hardcoded 3.0 must be visibly wrong for most opacities"
        );
        let half = screen.half_extent.x / screen.frame.cov.xx.sqrt();
        assert!((half - radius).abs() < 1e-9);
    }

    settings.support = SupportMode::FixedSigmas(3.0);
    let (outcome, _) = project_one(make(1.0), &spec, &settings);
    let fixed = visible(&outcome).support_sq.sqrt();
    assert!((fixed - 3.0).abs() < 1e-15);

    settings.support = SupportMode::CutoffSolved;
    let faint = Atom::isotropic(
        Vec3::new(0.0, 0.0, 5.0),
        0.05,
        settings.blend.alpha_cutoff * 0.5,
        Rgb::new(1.0, 1.0, 1.0),
    );
    let (outcome, _) = project_one(faint, &spec, &settings);
    assert_eq!(outcome.cull_reason(), Some(CullReason::Opacity));
}

#[test]
fn splat_rejects_unprojectable_kernel() {
    let settings = SharedSettings::<f64>::contract();
    match SplatRaster::<f64, Unprojectable>::new(SplatConfig::contract(), &settings) {
        Err(LightsError::KernelNotProjectable { kernel, method }) => {
            assert_eq!(kernel, "unprojectable");
            assert_eq!(method, MethodId::SplatRaster);
        }
        Err(other) => panic!("wrong error: {other:?}"),
        Ok(_) => panic!("a kernel with no screen marginal must not be projectable"),
    }
    assert!(SplatRaster::<f64, Gaussian>::new(SplatConfig::contract(), &settings).is_ok());
}

#[test]
fn splat_rejects_hit_depth_ordering() {
    let mut ordering = SharedSettings::<f64>::contract();
    ordering.ordering = OrderingModel::HitDepth;
    match SplatRaster::<f64, Gaussian>::new(SplatConfig::contract(), &ordering) {
        Err(LightsError::OrderingUnsupported { ordering, method }) => {
            assert_eq!(ordering, OrderingModel::HitDepth);
            assert_eq!(method, MethodId::SplatRaster);
        }
        other => panic!("per-pixel sorting is not built: {other:?}", other = other.err()),
    }
}

#[test]
fn the_signed_additive_path_renders_through_the_same_seam() {
    let settings = SharedSettings::<f64>::contract();
    let spec = intrinsic_camera([1000.0, 1000.0], [8.0, 8.0], 17, 17);
    let mut scene = Scene::<f64>::new();
    for i in 0..4 {
        let atom = Atom::isotropic(
            Vec3::new(0.0, 0.0, 8.0 - (i as f64)),
            0.05,
            0.5,
            Rgb::new(0.25, 0.0, 0.0),
        );
        match scene.push(atom, FLOOR) {
            Ok(()) => (),
            Err(e) => panic!("{e:?}"),
        }
    }
    let prepared = match PreparedScene::<f64, SignedTest>::build(&scene, &[], &settings) {
        Ok(p) => p,
        Err(e) => panic!("{e:?}"),
    };
    let camera = camera_of(&spec, &settings);
    let mut raster = match SplatRaster::<f64, SignedTest>::new(SplatConfig::contract(), &settings) {
        Ok(r) => r,
        Err(e) => panic!("{e:?}"),
    };
    match (raster.build(&prepared), raster.bind(&prepared, &camera)) {
        (Ok(_), Ok(_)) => (),
        _ => panic!("the additive path must bind"),
    }
    assert!(!<Additive as BlendMode<f64>>::ORDER_DEPENDENT);
    for range in raster.ranges() {
        let slots: Vec<u32> = raster.keys()[range.0 as usize..range.1 as usize]
            .iter()
            .map(|k| k.2)
            .collect();
        let mut expected = slots.clone();
        expected.sort_unstable();
        assert_eq!(
            slots, expected,
            "an unsorted blend visits a tile's atoms in scene order"
        );
    }
    assert!(raster.ranges().iter().any(|r| r.1 > r.0));

    let mut fb = match Framebuffer::<f64, Additive>::new(17, 17, SplatConfig::contract().tile) {
        Ok(f) => f,
        Err(e) => panic!("{e:?}"),
    };
    let mut sink = ProbeSink::<f64>::counters_only(17, 17);
    match raster.render::<NoProbe>(&prepared, &mut fb, &mut sink) {
        Ok(_) => (),
        Err(e) => panic!("{e:?}"),
    }
    let image = fb.resolve_float(Rgb::zero());
    let centre = image.get(8, 8);
    assert!(
        (centre.x - 4.0 * 0.5 * 0.25).abs() < 1e-12,
        "four atoms of weight 1 at the centre accumulate additively: {}",
        centre.x
    );
    assert_eq!(<Additive as BlendMode<f64>>::coverage(&Rgb::splat(1.0)), 0.0);
}

#[test]
fn sample_pixel_reproduces_the_tile_render_bit_for_bit() {
    let mut settings = SharedSettings::<f64>::contract();
    settings.threads = 1;
    let spec = intrinsic_camera([600.0, 600.0], [33.5, 25.5], 67, 51);
    let mut scene = Scene::<f64>::new();
    for i in 0..14 {
        let t = i as f64;
        let atom = Atom::new(
            Vec3::new(0.03 * t - 0.12, 0.022 * t - 0.09, 4.0 + 0.3 * t),
            Quat::identity(),
            Vec3::new(0.03 + 0.002 * t, 0.02, 0.05),
            0.4,
            Rgb::new(0.1 * t, 1.0 - 0.05 * t, 0.5),
        );
        match scene.push(atom, FLOOR) {
            Ok(()) => (),
            Err(e) => panic!("{e:?}"),
        }
    }
    for corner in [
        Vec3::new(0.21, 0.157, 4.0),
        Vec3::new(0.22, 0.0, 4.0),
        Vec3::new(0.0, 0.16, 4.0),
    ] {
        let atom = Atom::isotropic(corner, 0.02, 0.7, Rgb::new(0.2, 0.9, 0.3));
        match scene.push(atom, FLOOR) {
            Ok(()) => (),
            Err(e) => panic!("{e:?}"),
        }
    }
    let (raster, prepared, _) = bound_raster(&scene, &spec, &settings, SplatConfig::contract());
    let grid = match raster.grid() {
        Some(g) => g,
        None => panic!("bound"),
    };
    assert_eq!((grid.grid_x, grid.grid_y), (5, 4));
    assert_eq!(grid.rect(4, 3), lights::tiling::TileRect { x0: 64, y0: 48, w: 3, h: 3 });
    let mut fb = match Framebuffer::<f64, Over>::new(67, 51, SplatConfig::contract().tile) {
        Ok(f) => f,
        Err(e) => panic!("{e:?}"),
    };
    let mut sink = ProbeSink::<f64>::counters_only(67, 51);
    match raster.render::<NoProbe>(&prepared, &mut fb, &mut sink) {
        Ok(_) => (),
        Err(e) => panic!("{e:?}"),
    }
    let mut lit = 0usize;
    let mut checked = 0usize;
    for y in 0..51 {
        for x in 0..67 {
            let direct = match raster.sample_pixel(&prepared, x, y) {
                Ok(a) => a,
                Err(e) => panic!("{e:?}"),
            };
            let tiled = fb.cell(x, y);
            assert_eq!(
                direct, tiled,
                "sample_pixel and the tile render must agree bit for bit at ({x},{y})"
            );
            checked += 1;
            if tiled.transmittance < 1.0 {
                lit += 1;
            }
        }
    }
    assert_eq!(checked, 67 * 51);
    assert!(
        lit > 300,
        "the comparison must cover pixels that actually received atoms: {lit}"
    );
    let mut edge = 0usize;
    for y in 48..51 {
        for x in 64..67 {
            if fb.cell(x, y).transmittance < 1.0 {
                edge += 1;
            }
        }
    }
    assert!(
        edge > 0,
        "the clipped corner tile must receive contributions, so a cell index \
         built from the clipped width instead of the tile stride is visible"
    );
}

#[test]
fn screen_atoms_stay_in_scene_order_so_the_slot_is_the_atom_index() {
    let settings = SharedSettings::<f64>::contract();
    let spec = intrinsic_camera([600.0, 600.0], [32.0, 24.0], 64, 48);
    let mut scene = Scene::<f64>::new();
    let behind = Atom::isotropic(Vec3::new(0.0, 0.0, -1.0), 0.05, 1.0, Rgb::splat(1.0));
    for i in 0..6 {
        let atom = if i == 2 {
            behind
        } else {
            Atom::isotropic(Vec3::new(0.0, 0.0, 4.0 + i as f64), 0.05, 1.0, Rgb::splat(1.0))
        };
        match scene.push(atom, FLOOR) {
            Ok(()) => (),
            Err(e) => panic!("{e:?}"),
        }
    }
    let (raster, _, _) = bound_raster(&scene, &spec, &settings, SplatConfig::contract());
    let indices: Vec<u32> = raster.screen_atoms().iter().map(|a| a.index).collect();
    assert_eq!(indices, vec![0, 1, 3, 4, 5], "the near-culled atom is dropped");
    let mut ascending = indices.clone();
    ascending.sort_unstable();
    assert_eq!(
        indices, ascending,
        "the compacted slot is order-isomorphic to the scene index, which is \
         what makes the third field of the sort key a valid tiebreak"
    );
    assert_eq!(raster.bind_counters().atoms_culled_near, 1);
    assert_eq!(raster.bind_counters().atoms_projected, 5);
    assert_eq!(raster.bind_counters().atoms_prepared, 6);
}

#[test]
fn counters_and_tolerance_come_out_of_the_render_pass() {
    let mut settings = SharedSettings::<f64>::contract();
    settings.threads = 1;
    let spec = intrinsic_camera([1000.0, 1000.0], [64.0, 64.0], 129, 129);
    let mut scene = Scene::<f64>::new();
    for i in 0..3 {
        let atom = Atom::isotropic(
            Vec3::new(0.0, 0.0, 5.0 + i as f64),
            0.05,
            0.6,
            Rgb::splat(1.0),
        );
        match scene.push(atom, FLOOR) {
            Ok(()) => (),
            Err(e) => panic!("{e:?}"),
        }
    }
    let outcome = render_scene(scene, spec, settings, Collect::PixelsAndTolerance);
    assert_eq!(outcome.counters.atoms_prepared, 3);
    assert_eq!(outcome.counters.atoms_projected, 3);
    assert!(outcome.counters.tile_atom_pairs >= 3);
    assert_eq!(outcome.counters.sort_keys, outcome.counters.tile_atom_pairs);
    assert_eq!(
        outcome.counters.pixels_visited, 4096,
        "only the sixteen tiles the footprint box touches are visited"
    );
    assert!(outcome.counters.kernel_evaluations > outcome.counters.pixels_visited);
    assert!(
        outcome.counters.kernel_evaluations < outcome.counters.pixels_visited * 3,
        "all three atoms cover all sixteen visited tiles, so without the support \
         truncation this would be exactly {} evaluations; it is {}",
        outcome.counters.pixels_visited * 3,
        outcome.counters.kernel_evaluations
    );
    assert!(
        outcome.counters.candidates_tested > outcome.counters.kernel_evaluations,
        "CandidatesTested is charged once per (pixel, binned atom) pair BEFORE \
         the support test and KernelEvaluations only after it, so their \
         difference IS the over-binning: the atoms a tile carries whose \
         support does not reach the pixel. Until 2026-09-15 the splat path \
         charged this counter nowhere, so every splat row of lights-bench and \
         of the lights-p2 report printed candidates = 0 and the waste was \
         invisible in the portable counters. candidates {} against evaluations \
         {}",
        outcome.counters.candidates_tested,
        outcome.counters.kernel_evaluations
    );
    assert_eq!(
        outcome.counters.candidates_tested,
        outcome.counters.pixels_visited * 3,
        "every visited pixel tests every one of the three atoms its tile \
         carries, so the candidate count is exactly pixels x binned atoms. \
         That equality is what makes candidates - evaluations readable as a \
         waste figure rather than as an unnamed difference"
    );

    let tolerance = match outcome.tolerance {
        Some(t) => t,
        None => panic!("PixelsAndTolerance must produce a tolerance image"),
    };
    let (peak, _, _) = tolerance.max_abs();
    assert!(
        peak > 0.0,
        "the tolerance must be accumulated in the render loop, where T_i is known"
    );
    let outside = tolerance.get(0, 0);
    let centre = tolerance.get(64, 64);
    let rim = tolerance.get(64, 84);
    assert_eq!(
        outside.x, 0.0,
        "a pixel no atom reaches must carry no accumulated tolerance"
    );
    assert!(centre.x > 0.0 && rim.x > 0.0);
    assert!(
        (rim.x / centre.x - 1.0).abs() > 0.1,
        "the tolerance varies across the frame, so a single global number would \
         be a fudge factor: centre {} rim {}",
        centre.x,
        rim.x
    );
}

#[test]
fn atom_bound_matches_the_closed_form() {
    let iso = |sigma: f64| Sym3::new(sigma * sigma, 0.0, 0.0, sigma * sigma, 0.0, sigma * sigma);
    let centre = atom_bound_terms(Vec3::new(0.0, 0.0, 5.0), &iso(0.05), 1.0);
    assert!((centre.rho_e - 100.0).abs() < 1e-9);
    assert!(
        centre.omega.abs() < 1e-8,
        "on-axis Omega vanishes up to the rounding of the Cholesky: {}",
        centre.omega
    );
    assert!(
        (centre.total / 1.0826822658929016e-4 - 1.0).abs() < 1e-5,
        "TV-X4: {}",
        centre.total
    );

    let exact = iso(0.0625);
    let dyadic = atom_bound_terms(Vec3::new(0.0, 0.0, 5.0), &exact, 1.0);
    assert!((dyadic.rho_e - 80.0).abs() < 1e-9);
    assert_eq!(
        dyadic.total,
        1.0826822658929016 / 6400.0,
        "on dyadic inputs the second-order coefficient is reproduced exactly"
    );

    let mid = atom_bound_terms(Vec3::new(2.5, 0.0, 5.0), &iso(0.05), 1.0);
    assert!((mid.rho_e - 111.80339887498948).abs() < 1e-9);
    assert!((mid.omega - 0.004472135954999579).abs() < 1e-9);
    assert!(
        (mid.total / 0.005271690947020207 - 1.0).abs() < 1e-5,
        "TV-X5: {}",
        mid.total
    );

    let corner = atom_bound(Vec3::new(5.0, 0.0, 5.0), &iso(0.05), 1.0);
    assert!(
        (corner / 0.008252459692131845 - 1.0).abs() < 1e-5,
        "TV-X6: {corner}"
    );

    let sigma = iso(0.05);
    let lambda = match sigma.inverse_spd() {
        Some(l) => l,
        None => panic!("spd"),
    };
    let via_lambda = atom_bound_with_lambda(Vec3::new(2.5, 0.0, 5.0), sigma.zz, &lambda, 1.0);
    assert!((via_lambda - mid.total).abs() < 1e-18);

    let degenerate = atom_bound(Vec3::new(0.0, 0.0, 0.0), &iso(0.05), 1.0);
    assert_eq!(degenerate, 0.0);
}

#[test]
fn the_whitened_offset_advances_by_one_step_per_column() {
    let cov = Sym2::new(100.3625, 0.125, 25.55);
    let frame = match ScreenFrame::from_covariance(cov) {
        Some(f) => f,
        None => panic!("spd"),
    };
    let step = frame.whiten_step_x();
    for column in 0..16 {
        let here = frame.whiten(Vec2::new(10.0 - column as f64, 4.0));
        let next = frame.whiten(Vec2::new(10.0 - (column + 1) as f64, 4.0));
        assert!((here.x - next.x - step.x).abs() < 1e-12);
        assert!((here.y - next.y - step.y).abs() < 1e-12);
    }
}

#[test]
fn gathering_a_tile_batch_follows_the_sorted_key_order() {
    let grid = match TileGrid::new(32, 32, 16) {
        Ok(g) => g,
        Err(e) => panic!("{e:?}"),
    };
    let screen = [
        synthetic_atom(Vec2::new(8.0, 8.0), Vec2::splat(3.0), 9.0, 0, Rgb::new(1.0, 0.0, 0.0), 0.5),
        synthetic_atom(Vec2::new(8.0, 8.0), Vec2::splat(3.0), 2.0, 1, Rgb::new(0.0, 1.0, 0.0), 0.5),
    ];
    let mut keys = Vec::new();
    let mut ranges = Vec::new();
    bin_and_sort::<f64, Gaussian>(
        &screen,
        &grid,
        true,
        SortKind::GlobalSort,
        &mut keys,
        &mut ranges,
    );
    let mut batch = Vec::new();
    gather_batch(&screen, &keys, ranges[0], &mut batch);
    assert_eq!(batch.len(), 2);
    assert_eq!(batch[0].index, 1, "the nearer atom is gathered first");
    assert_eq!(batch[1].index, 0);

    let settings = SharedSettings::<f64>::reference();
    let mut cell = <Over as BlendMode<f64>>::init();
    let mut probe = NoProbe;
    shade_pixel::<f64, Gaussian, NoProbe>(
        &batch,
        Vec2::new(8.0, 8.0),
        0,
        &settings,
        &mut cell,
        &mut probe,
    );
    assert!(
        cell.rgb.y > cell.rgb.x,
        "the front atom (green) must dominate: {:?}",
        cell.rgb
    );
    assert!((cell.transmittance - 0.25).abs() < 1e-12);
}

#[test]
fn a_probe_reports_the_same_counters_at_every_thread_count() {
    let spec = intrinsic_camera([600.0, 600.0], [32.0, 24.0], 64, 48);
    let mut scene = Scene::<f64>::new();
    for i in 0..9 {
        let atom = Atom::isotropic(
            Vec3::new(0.03 * i as f64, -0.02 * i as f64, 4.0 + 0.4 * i as f64),
            0.06,
            0.5,
            Rgb::splat(0.8),
        );
        match scene.push(atom, FLOOR) {
            Ok(()) => (),
            Err(e) => panic!("{e:?}"),
        }
    }
    let mut reference = None;
    for threads in [1usize, 4, 18] {
        let mut settings = SharedSettings::<f64>::contract();
        settings.threads = threads;
        let outcome = render_scene(
            scene.clone(),
            spec.clone(),
            settings,
            Collect::PixelsAndCounters,
        );
        let seen = (
            outcome.counters.get(Counter::KernelEvaluations),
            outcome.counters.get(Counter::PixelsVisited),
            outcome.counters.get(Counter::CompositedHits),
        );
        match reference {
            None => reference = Some(seen),
            Some(expected) => assert_eq!(
                seen, expected,
                "counters are a deterministic function of the scene, not of the \
                 thread count ({threads} threads)"
            ),
        }
    }
}

#[test]
fn the_projected_centre_uses_the_unclamped_position_and_the_clamp_is_counted() {
    let settings = SharedSettings::<f64>::contract();
    let spec = intrinsic_camera([600.0, 600.0], [400.0, 300.0], 800, 600);
    let camera = camera_of(&spec, &settings);
    let limits = camera.jacobian_limits();
    assert!((limits.x - 1.3 * 400.0 / 600.0).abs() < 1e-12);

    let far_off_axis = Atom::isotropic(Vec3::new(5.0, 0.0, 1.0), 0.05, 1.0, Rgb::splat(1.0));
    let (outcome, fired) = project_one(far_off_axis, &spec, &settings);
    assert!(fired, "x/z = 5 is far outside the 1.3*tan(fov/2) limit");
    let screen = visible(&outcome);
    assert!(
        (screen.centre.x - 3400.0).abs() < 1e-9,
        "the centre must come from the UNCLAMPED position: {}",
        screen.centre.x
    );
    let (clamped_point, _) = camera.clamp_for_jacobian(Vec3::new(5.0, 0.0, 1.0));
    let wrong = match camera.project(clamped_point) {
        Some(p) => p,
        None => panic!("in front"),
    };
    assert!(
        (wrong.x - 920.0).abs() < 1e-9,
        "projecting the clamped position drags the atom to {}",
        wrong.x
    );

    let inside = Atom::isotropic(Vec3::new(0.5, 0.0, 1.0), 0.05, 1.0, Rgb::splat(1.0));
    let (_, quiet) = project_one(inside, &spec, &settings);
    assert!(!quiet, "an atom inside the frustum must not clamp");

    let mut scene = Scene::<f64>::new();
    match (
        scene.push(inside, FLOOR),
        scene.push(far_off_axis, FLOOR),
        scene.push(
            Atom::isotropic(Vec3::new(0.0, 0.0, 5.0), 0.05, 1.0, Rgb::splat(1.0)),
            FLOOR,
        ),
    ) {
        (Ok(()), Ok(()), Ok(())) => (),
        _ => panic!("scene rejected an atom"),
    }
    let (raster, _, _) = bound_raster(&scene, &spec, &settings, SplatConfig::contract());
    assert_eq!(
        raster.clamped_atoms(),
        1,
        "a clamped atom has no ray-casting counterpart, so the count must be \
         reported rather than silently dropped"
    );
    assert_eq!(raster.bind_counters().atoms_clamped, 1);

    let outcome = render_scene(scene, spec, settings, Collect::PixelsAndCounters);
    assert_eq!(outcome.report.clamped_atoms, 1);
    assert_eq!(outcome.counters.atoms_clamped, 1);
}

#[test]
fn the_sort_key_tiebreak_survives_an_unstable_sort() {
    let grid = match TileGrid::new(32, 32, 16) {
        Ok(g) => g,
        Err(e) => panic!("{e:?}"),
    };
    let mut screen = Vec::new();
    for i in 0..96u32 {
        let depth = 2.0 + ((i % 6) as f64);
        screen.push(synthetic_atom(
            Vec2::new(8.0, 8.0),
            Vec2::splat(3.0),
            depth,
            i,
            Rgb::splat(1.0),
            0.5,
        ));
    }
    let mut keys = Vec::new();
    let mut ranges = Vec::new();
    bin_and_sort::<f64, Gaussian>(
        &screen,
        &grid,
        true,
        SortKind::GlobalSort,
        &mut keys,
        &mut ranges,
    );
    assert_eq!(keys.len(), 96);
    for pair in keys.windows(2) {
        let (left, right) = (pair[0], pair[1]);
        assert!(
            (left.0, left.1, left.2) < (right.0, right.1, right.2),
            "the emitted order must be the strict total order on the whole key, \
             tiebreak included: {left:?} then {right:?}"
        );
    }
    let mut per_tile_keys = Vec::new();
    let mut per_tile_ranges = Vec::new();
    bin_and_sort::<f64, Gaussian>(
        &screen,
        &grid,
        true,
        SortKind::PerTileSort,
        &mut per_tile_keys,
        &mut per_tile_ranges,
    );
    assert_eq!(per_tile_keys, keys);
}

#[test]
fn a_rendered_frame_round_trips_through_the_png_writer() {
    let mut settings = SharedSettings::<f64>::reference();
    settings.background = Rgb::new(0.0, 0.05, 0.2);
    let spec = intrinsic_camera([600.0, 600.0], [24.0, 18.0], 48, 36);
    let mut scene = Scene::<f64>::new();
    for i in 0..6 {
        let t = i as f64;
        let atom = Atom::new(
            Vec3::new(0.05 * t - 0.12, 0.03 * t - 0.06, 3.5 + 0.4 * t),
            Quat::identity(),
            Vec3::new(0.04, 0.025, 0.05),
            0.6,
            Rgb::new(1.0 - 0.15 * t, 0.2 * t, 0.4),
        );
        match scene.push(atom, FLOOR) {
            Ok(()) => (),
            Err(e) => panic!("{e:?}"),
        }
    }
    let outcome = render_scene(scene, spec, settings, Collect::Pixels);
    let quantised = outcome.image.to_clamped_u8();
    assert_eq!((quantised.width, quantised.height), (48, 36));

    let directory = match tempfile::tempdir() {
        Ok(d) => d,
        Err(e) => panic!("tempdir: {e:?}"),
    };
    let path = directory.path().join("frame.png");
    if let Err(e) = lights::image::write_png(&path, &quantised) {
        panic!("write_png: {e:?}");
    }
    let decoded = match lights::image::read_png(&path) {
        Ok(i) => i,
        Err(e) => panic!("read_png: {e:?}"),
    };
    assert_eq!(decoded, quantised, "the PNG codec must round trip exactly");

    let distinct: std::collections::BTreeSet<[u8; 3]> =
        decoded.pixels.iter().copied().collect();
    assert!(
        distinct.len() > 20,
        "the written frame must carry real structure, not one flat colour: {} \
         distinct colours",
        distinct.len()
    );
    assert!(
        decoded.pixels.iter().any(|p| p[0] > 128),
        "the near red atom must survive quantisation"
    );
}

fn rotated_needle(kappa: f64, sigma_major: f64) -> Sym2<f64> {
    let major = sigma_major * sigma_major;
    let minor = major / kappa;
    Sym2::new((major + minor) * 0.5, (major - minor) * 0.5, (major + minor) * 0.5)
}

fn needle_atom(cov: Sym2<f64>, centre: Vec2<f64>, radius: f64) -> ScreenAtom<f64, Gaussian> {
    let frame = match ScreenFrame::from_covariance(cov) {
        Some(f) => f,
        None => panic!("the needle covariance is not SPD"),
    };
    ScreenAtom {
        centre,
        depth: 5.0,
        frame,
        prepared: (),
        half_extent: tight_half_extent(cov, radius),
        support_sq: radius * radius,
        opacity: 0.8,
        colour: Rgb::splat(1.0),
        bound: 0.0,
        index: 0,
    }
}

fn aabb_span_pairs(grid: &TileGrid, atom: &ScreenAtom<f64, Gaussian>) -> u64 {
    let min = Vec2::new(
        atom.centre.x - atom.half_extent.x,
        atom.centre.y - atom.half_extent.y,
    );
    let max = Vec2::new(
        atom.centre.x + atom.half_extent.x,
        atom.centre.y + atom.half_extent.y,
    );
    match grid.span_of_box(min, max) {
        Some(s) => ((s.tx1 - s.tx0) as u64) * ((s.ty1 - s.ty0) as u64),
        None => 0,
    }
}

#[test]
fn tv_s13c_the_tile_loop_rejects_tiles_the_ellipse_misses() {
    let grid = match TileGrid::new(512, 512, 16) {
        Ok(g) => g,
        Err(e) => panic!("grid: {e:?}"),
    };
    let radius = 3.0;
    let mut keys = Vec::new();
    let mut ranges = Vec::new();
    let mut reported = Vec::new();
    for kappa in [1.0f64, 4.0, 100.0, 400.0] {
        let cov = rotated_needle(kappa, 50.0);
        let atom = needle_atom(cov, Vec2::new(256.0, 256.0), radius);
        let box_pairs = aabb_span_pairs(&grid, &atom);
        let emitted = bin_and_sort::<f64, Gaussian>(
            &[atom],
            &grid,
            true,
            SortKind::GlobalSort,
            &mut keys,
            &mut ranges,
        );
        assert_eq!(emitted, keys.len() as u64);
        reported.push((kappa, box_pairs, emitted));
        assert!(
            emitted <= box_pairs,
            "kappa {kappa}: the tile loop emitted {emitted} keys against an \
             axis-aligned span of {box_pairs}; the span is the conservative first \
             pass and nothing may be emitted outside it"
        );
        let inside = grid.tile_id(16, 16);
        assert!(
            keys.iter().any(|k| k.0 == inside),
            "kappa {kappa}: the tile holding the atom centre was rejected, so the \
             rejection is not conservative and the image has changed"
        );
    }
    let dense = reported[0];
    assert!(
        (dense.2 as f64) < (dense.1 as f64) * 0.85,
        "even the ISOTROPIC footprint (kappa 1) should lose its four box corners: \
         {} keys against a {}-tile span",
        dense.2,
        dense.1
    );
    let worst = reported[3];
    assert!(
        (worst.2 as f64) < (worst.1 as f64) * 0.45,
        "THE BOX IS THE COST, NOT THE KERNEL. At kappa 400 rotated 45 degrees the \
         axis-aligned box around the support ellipse has 6.4x the ellipse's area \
         (tv_s13_bbox_tightness measures the OTHER ratio, box against bounding \
         circle, and that one the tight box wins). span_of_box stays the \
         conservative first pass; the per-tile Mahalanobis rejection is what turns \
         the span into the footprint. Got {} keys against a {}-tile span",
        worst.2,
        worst.1
    );
    for (kappa, box_pairs, emitted) in reported {
        println!("kappa {kappa}: span {box_pairs} emitted {emitted}");
    }
}

#[test]
fn tv_s13d_the_tile_rejection_leaves_the_rendered_image_bit_identical() {
    let mut settings = SharedSettings::<f64>::contract();
    settings.threads = 1;
    let spec = intrinsic_camera([900.0, 900.0], [128.0, 128.0], 256, 256);
    let mut scene = Scene::<f64>::new();
    for i in 0..7 {
        let t = i as f64;
        let axis = match Vec3::new(0.0, 0.0, 1.0).normalize() {
            Some(a) => a,
            None => panic!("the view axis must normalise"),
        };
        let spin = match Quat::from_axis_angle(axis, 0.7853981633974483 + 0.21 * t) {
            Some(q) => q,
            None => panic!("the needle rotation must normalise"),
        };
        let atom = Atom::new(
            Vec3::new(0.02 * t - 0.06, 0.015 * t - 0.05, 4.0 + 0.25 * t),
            spin,
            Vec3::new(0.2, 0.004, 0.008),
            0.45,
            Rgb::new(0.1 * t, 1.0 - 0.06 * t, 0.4),
        );
        match scene.push(atom, FLOOR) {
            Ok(()) => (),
            Err(e) => panic!("{e:?}"),
        }
    }
    let config = SplatConfig {
        tile: 16,
        sort: SortKind::GlobalSort,
    };
    let (raster, prepared, camera) = bound_raster(&scene, &spec, &settings, config);
    let screen = raster.screen_atoms();
    assert!(
        screen.len() >= 6,
        "the needle scene projected only {} atoms, so the comparison below is thin",
        screen.len()
    );
    let pairs = raster.bind_counters().tile_atom_pairs;
    let mut span_total = 0u64;
    let grid = match raster.grid() {
        Some(g) => g,
        None => panic!("bound"),
    };
    for atom in screen {
        span_total += aabb_span_pairs(&grid, atom);
    }
    println!("needle field: pairs {pairs} span {span_total}");
    assert!(
        pairs * 2 < span_total,
        "the rejection must more than halve the span on a field of 45-degree \
         needles: {pairs} pairs against a {span_total}-tile span"
    );

    let mut fb = match Framebuffer::<f64, Over>::new(256, 256, config.tile) {
        Ok(f) => f,
        Err(e) => panic!("{e:?}"),
    };
    let mut sink = ProbeSink::<f64>::counters_only(256, 256);
    match raster.render::<NoProbe>(&prepared, &mut fb, &mut sink) {
        Ok(_) => (),
        Err(e) => panic!("{e:?}"),
    }

    let mut order: Vec<usize> = (0..screen.len()).collect();
    order.sort_by_key(|&i| (depth_bits(screen[i].depth), i as u32));
    let batch: Vec<BatchEntry<f64, Gaussian>> =
        order.iter().map(|&i| entry_of(&screen[i])).collect();

    let mut lit = 0usize;
    for y in 0..256u32 {
        for x in 0..256u32 {
            let mut cell = <Over as BlendMode<f64>>::init();
            let mut probe = NoProbe;
            shade_pixel::<f64, Gaussian, NoProbe>(
                &batch,
                camera.sample_point(x, y),
                0,
                &settings,
                &mut cell,
                &mut probe,
            );
            assert_eq!(
                cell,
                fb.cell(x, y),
                "the tile-binned render and the brute-force walk over EVERY \
                 projected atom must agree bit for bit at ({x},{y}). The per-tile \
                 Mahalanobis rejection may only drop (tile, atom) pairs whose every \
                 sample point already failed `r2 <= support_sq` inside shade_pixel; \
                 a single pixel that moves means the rejection is tighter than the \
                 shading test and the binning has become lossy"
            );
            if fb.cell(x, y).transmittance < 1.0 {
                lit += 1;
            }
        }
    }
    assert!(
        lit > 512,
        "only {lit} pixels carry signal, so the bit-identity check above is \
         comparing two nearly empty images"
    );
}

#[test]
fn the_binning_waste_of_a_rotated_needle_is_now_visible_in_the_portable_counters() {
    let mut settings = SharedSettings::<f64>::contract();
    settings.threads = 1;
    let spec = intrinsic_camera([1000.0, 1000.0], [128.0, 128.0], 257, 257);

    let mut measured: Vec<(f64, u64, u64, f64)> = Vec::new();
    for angle in [0.0f64, core::f64::consts::FRAC_PI_4] {
        let rotation = match Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), angle) {
            Some(q) => q,
            None => Quat::identity(),
        };
        let mut scene = Scene::<f64>::new();
        let atom = Atom::new(
            Vec3::new(0.0, 0.0, 5.0),
            rotation,
            Vec3::new(0.45, 0.01, 0.01),
            0.9,
            Rgb::splat(1.0),
        );
        match scene.push(atom, FLOOR) {
            Ok(()) => (),
            Err(e) => panic!("{e:?}"),
        }
        let outcome = render_scene(scene, spec.clone(), settings, Collect::PixelsAndCounters);
        let candidates = outcome.counters.candidates_tested;
        let evaluations = outcome.counters.kernel_evaluations;
        assert!(
            candidates > 0,
            "the splat path charged CandidatesTested nowhere before 2026-09-15, \
             and a zero here means the increment was removed again"
        );
        assert!(
            candidates >= evaluations,
            "a kernel evaluation happens only after the support test a \
             candidate passes, so candidates can never be the smaller number: \
             {candidates} against {evaluations}"
        );
        let waste = candidates as f64 / evaluations.max(1) as f64;
        println!(
            "angle {angle:.4} rad: {candidates} candidates, {evaluations} \
             evaluations, {waste:.3}x"
        );
        measured.push((angle, candidates, evaluations, waste));
    }
    assert_eq!(measured.len(), 2, "the sweep asserted nothing if it ran no rows");

    let axis_aligned = measured[0];
    let diagonal = measured[1];
    assert!(
        diagonal.3 > axis_aligned.3,
        "a 45:1 needle turned 45 degrees fills a far larger fraction of the \
         axis-aligned tiles it is binned into than the same needle lying along \
         an axis, so its candidate-to-evaluation ratio must be the larger. \
         axis-aligned {:.3}x against diagonal {:.3}x. This comparison is the \
         whole reason the counter had to exist: it is not derivable from \
         kernel_evaluations alone, and the fit-only box_over_ellipse_area \
         column needs a fitted 2D scene rather than an ordinary render",
        axis_aligned.3,
        diagonal.3
    );
    assert!(
        diagonal.3 > 1.5,
        "the diagonal needle's measured waste was {:.3}x, which is small \
         enough that the test is no longer pointed at an over-binned case",
        diagonal.3
    );
}
