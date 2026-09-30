use lights::camera::{Camera, CameraSpec};
use lights::image::{FloatImage, Framebuffer};
use lights::kernel::gabor::{GaborColour, gabor_params_from};
use lights::kernel::hermite::{GaussHermiteColour, hermite_params_from};
use lights::kernel::{CurveShape, Kernel};
use lights::method::{ImageFormation, RenderMethod, create_strategy};
use lights::numerics::quaternion::Quat;
use lights::numerics::vector::{Rgb, Vec3};
use lights::probe::{NoProbe, ProbeSink};
use lights::scene::prepare::PreparedScene;
use lights::scene::{Atom, Scene};
use lights::settings::{PixelConvention, SharedSettings};

const VIEWPORT: (u32, u32) = (40, 32);
const Z0: f64 = 40.0;

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

fn deep_atom(index: usize, colour: Rgb<f64>) -> Atom<f64> {
    let tilt = Quat::from_axis_angle(Vec3::new(1.0, 0.4, 0.2), 0.6).unwrap_or(Quat::identity());
    Atom::new(
        Vec3::new(
            index as f64 * 5.0 - 5.0,
            index as f64 * -3.0 + 3.0,
            Z0 - index as f64 * 0.25,
        ),
        tilt,
        Vec3::new(7.0, 4.5, 6.0),
        0.62,
        colour,
    )
}

fn deep_scene(settings: &SharedSettings<f64>) -> Scene<f64> {
    let mut scene = Scene::<f64>::new();
    for (i, colour) in [
        Rgb::new(0.90, 0.35, 0.20),
        Rgb::new(0.20, 0.80, 0.45),
        Rgb::new(0.30, 0.45, 0.95),
    ]
    .into_iter()
    .enumerate()
    {
        scene
            .push(deep_atom(i, colour), settings.scale_floor)
            .expect("push");
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

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn digest(image: &FloatImage<f64>) -> u64 {
    let mut hash = FNV_OFFSET;
    for pixel in &image.pixels {
        for channel in [pixel.x, pixel.y, pixel.z] {
            for byte in channel.to_bits().to_be_bytes() {
                hash ^= byte as u64;
                hash = hash.wrapping_mul(FNV_PRIME);
            }
        }
    }
    hash
}

fn vitality(image: &FloatImage<f64>) -> String {
    let background = image.pixels[0];
    let moved = image
        .pixels
        .iter()
        .filter(|p| {
            (p.x - background.x).abs() > 1e-9
                || (p.y - background.y).abs() > 1e-9
                || (p.z - background.z).abs() > 1e-9
        })
        .count();
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for p in &image.pixels {
        for c in [p.x, p.y, p.z] {
            lo = lo.min(c);
            hi = hi.max(c);
        }
    }
    format!(
        "{} of {} pixels differ from corner; channel range {:.6}..{:.6}",
        moved,
        image.pixels.len(),
        lo,
        hi
    )
}

fn settings() -> SharedSettings<f64> {
    SharedSettings::<f64>::contract()
}

fn gabor_shape() -> CurveShape {
    CurveShape::gabor(0.55, 0.7, 0.25)
}

fn hermite_shape() -> CurveShape {
    CurveShape::hermite(3, 0.7)
}

const FROZEN_IN_PLANE: &[(&str, u64, u64)] = &[
    ("splat-raster", 0x46f8_e6b6_035d_6925, 0x6b7c_909b_2160_4b3b),
    ("ray-cast", 0x323f_4c83_aeba_1e7c, 0x6b7c_909b_2160_4b3b),
];

fn frozen(name: &str) -> (u64, u64) {
    for (method, gabor, hermite) in FROZEN_IN_PLANE {
        if *method == name {
            return (*gabor, *hermite);
        }
    }
    panic!(
        "no frozen in-plane digest for method {name}; a new method must have its \
         zero-elevation digests captured on a tree where the carrier is still \
         pinned to the xy plane, because that is the only thing that can prove \
         the elevation parameter did not move the in-plane path"
    );
}

const ELEVATION: f64 = 0.9;

#[test]
fn the_zero_elevation_path_is_bit_identical_to_the_tree_that_had_no_elevation() {
    let settings = settings();
    let scene = deep_scene(&settings);
    let n = scene.len();
    let mut covered = 0usize;
    for method in RenderMethod::implemented() {
        let name = method.id().name();
        let (want_gabor, want_hermite) = frozen(name);
        let gabor = vec![gabor_params_from::<f64>(&gabor_shape()); n];
        let hermite = vec![hermite_params_from::<f64>(&hermite_shape()); n];
        let g = render_with::<GaborColour>(&scene, &gabor, &settings, &method);
        let h = render_with::<GaussHermiteColour>(&scene, &hermite, &settings, &method);
        assert_eq!(
            digest(&g),
            want_gabor,
            "{name}: the gabor carrier at zero elevation must reproduce the tree \
             that had no elevation parameter BIT FOR BIT. Every published figure \
             in this crate is a screen-path figure and this is what says they \
             still stand. {}",
            vitality(&g)
        );
        assert_eq!(
            digest(&h),
            want_hermite,
            "{name}: the hermite carrier at zero elevation must reproduce the \
             tree that had no elevation parameter BIT FOR BIT. {}",
            vitality(&h)
        );
        covered += 1;
    }
    assert_eq!(covered, RenderMethod::implemented().len());
    assert!(covered >= 1, "no method was built, so nothing was checked");
}

#[test]
fn elevation_reaches_the_ray_path_and_deliberately_never_reaches_the_screen_path() {
    let settings = settings();
    let scene = deep_scene(&settings);
    let n = scene.len();
    let mut covered = 0usize;
    for method in RenderMethod::implemented() {
        let name = method.id().name();
        let up_g = vec![gabor_params_from::<f64>(&gabor_shape().with_elevation(ELEVATION)); n];
        let down_g = vec![gabor_params_from::<f64>(&gabor_shape().with_elevation(-ELEVATION)); n];
        let up_h = vec![hermite_params_from::<f64>(&hermite_shape().with_elevation(ELEVATION)); n];
        let down_h =
            vec![hermite_params_from::<f64>(&hermite_shape().with_elevation(-ELEVATION)); n];

        let g_up = digest(&render_with::<GaborColour>(&scene, &up_g, &settings, &method));
        let g_down = digest(&render_with::<GaborColour>(
            &scene, &down_g, &settings, &method,
        ));
        let h_up = digest(&render_with::<GaussHermiteColour>(
            &scene, &up_h, &settings, &method,
        ));
        let h_down = digest(&render_with::<GaussHermiteColour>(
            &scene, &down_h, &settings, &method,
        ));

        match name {
            "splat-raster" => {
                let (flat_g, flat_h) = frozen(name);
                assert_eq!(
                    g_up, flat_g,
                    "splat must ignore elevation ENTIRELY. gabor_prepare_screen \
                     lifts by L^T over a two-dimensional frame and \
                     carrier_frame.md rules that it stays that way; a screen path \
                     that moved would mean the elevation had leaked into every \
                     published figure in this crate"
                );
                assert_eq!(g_down, flat_g, "splat must ignore a negative elevation too");
                assert_eq!(h_up, flat_h, "splat must ignore elevation for hermite");
                assert_eq!(h_down, flat_h, "splat must ignore a negative elevation");
            }
            "ray-cast" => {
                assert_ne!(
                    g_up, g_down,
                    "+e and -e share cos(e) EXACTLY, so the in-plane components of \
                     the carrier are identical and only the z term differs in \
                     sign. If these two images agree, the z term is dead -- which \
                     is what the tree did before this entry, and what a test \
                     comparing zero against non-zero elevation CANNOT see, because \
                     the cos(e) scaling of the in-plane terms moves that image on \
                     its own"
                );
                assert_ne!(
                    h_up, h_down,
                    "the hermite carrier's z term must be live and signed on the \
                     ray path, by the same argument"
                );
            }
            other => panic!(
                "method {other} is built but this test does not say whether its \
                 carrier is a screen-space or a ray-space one, and guessing is \
                 how a method gets silently exempted from the check"
            ),
        }
        covered += 1;
    }
    assert_eq!(covered, RenderMethod::implemented().len());
    assert!(covered >= 1, "no method was built, so nothing was checked");
}

#[test]
fn the_elevation_is_a_renderer_parameter_that_the_fitter_does_not_reach() {
    assert!(CurveShape::gaussian().is_in_plane());
    assert!(CurveShape::gabor(0.5, 0.3, 0.1).is_in_plane());
    assert!(CurveShape::hermite(2, 0.3).is_in_plane());
    assert!(!CurveShape::gabor(0.5, 0.3, 0.1).with_elevation(0.4).is_in_plane());
}
