use lights::camera::{Camera, CameraSpec};
use lights::dispatch::{Collect, Precision, RenderRequest, render};
use lights::image::FloatImage;
use lights::kernel::KernelFamily;
use lights::kernel::gaussian::Gaussian;
use lights::method::RenderMethod;
use lights::numerics::vector::{Rgb, Vec3};
use lights::renderer::Renderer;
use lights::scene::{Atom, Scene};
use lights::settings::{MIN_SCALE, PixelConvention, SharedSettings};

const WIDTH: u32 = 96;
const HEIGHT: u32 = 72;

fn spec() -> CameraSpec {
    CameraSpec::look_at(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
        45.0,
        WIDTH,
        HEIGHT,
    )
    .with_convention(PixelConvention::CornerSample)
}

fn scene_of(shift: f64) -> Scene<f64> {
    let mut scene = Scene::<f64>::new();
    for index in 0..5u32 {
        let t = index as f64;
        let atom = Atom::isotropic(
            Vec3::new(-0.4 + 0.2 * t + shift, 0.12 * t - 0.2, 3.2 + 0.1 * t),
            0.14,
            0.8,
            Rgb::new((0.17 * t).fract(), (0.41 * t).fract(), (0.63 * t).fract()),
        );
        match scene.push(atom, MIN_SCALE) {
            Ok(()) => (),
            Err(e) => panic!("scene rejected an atom: {e:?}"),
        }
    }
    scene
}

fn settings() -> SharedSettings<f64> {
    let mut out = SharedSettings::<f64>::contract();
    out.threads = 1;
    out.background = Rgb::new(0.04, 0.05, 0.08);
    out
}

fn cold(method: &RenderMethod, scene: Scene<f64>) -> FloatImage<f64> {
    let request = RenderRequest {
        precision: Precision::F64,
        kernel: KernelFamily::Gaussian,
        method: method.clone(),
        settings: settings(),
        camera: spec(),
        scene,
        collect: Collect::Pixels,
        shapes: None,
    };
    match render(&request) {
        Ok(o) => o.image,
        Err(e) => panic!("cold render failed: {e:?}"),
    }
}

fn bit_identical(a: &FloatImage<f64>, b: &FloatImage<f64>, what: &str) {
    assert_eq!((a.width, a.height), (b.width, b.height), "{what}: size");
    let mut differing = 0usize;
    let mut worst = 0.0f64;
    for (left, right) in a.pixels.iter().zip(b.pixels.iter()) {
        if left.x.to_bits() != right.x.to_bits()
            || left.y.to_bits() != right.y.to_bits()
            || left.z.to_bits() != right.z.to_bits()
        {
            differing += 1;
            worst = worst
                .max((left.x - right.x).abs())
                .max((left.y - right.y).abs())
                .max((left.z - right.z).abs());
        }
    }
    assert_eq!(
        differing, 0,
        "{what}: {differing} of {} pixels differ, worst by {worst:e}",
        a.pixels.len()
    );
}

#[test]
fn the_warm_renderer_and_the_one_shot_agree_bit_for_bit_on_every_built_method() {
    let methods = RenderMethod::implemented();
    assert!(
        !methods.is_empty(),
        "no method is built into this test binary, so this case would pass by measuring nothing"
    );

    let shape = KernelFamily::Gaussian.shape();
    let settings = settings();
    let camera = match Camera::<f64>::from_spec(&spec(), settings.near_plane) {
        Ok(c) => c,
        Err(e) => panic!("camera failed: {e:?}"),
    };

    let mut checked = 0usize;
    for method in methods.iter() {
        let first_scene = scene_of(0.0);
        let mut warm = match Renderer::<f64, Gaussian>::new(
            method,
            &settings,
            &shape,
            WIDTH,
            HEIGHT,
            &first_scene,
        ) {
            Ok(r) => r,
            Err(e) => panic!("Renderer::new failed: {e:?}"),
        };
        assert_eq!(warm.frames(), 0, "a renderer that has drawn nothing must say so");

        if let Err(e) = warm.render_frame(&camera) {
            panic!("render_frame failed: {e:?}");
        }
        let one = warm.resolve(settings.background);
        bit_identical(
            &one,
            &cold(method, scene_of(0.0)),
            &format!(
                "{}: the warm path must produce the SAME PIXELS as dispatch::render(). A held \
                 framebuffer and a held strategy are an allocation decision, never a rendering \
                 one",
                method.id().name()
            ),
        );

        if let Err(e) = warm.render_frame(&camera) {
            panic!("second render_frame failed: {e:?}");
        }
        let two = warm.resolve(settings.background);
        bit_identical(
            &two,
            &one,
            &format!(
                "{}: the SECOND frame through the same renderer must equal the first. This is \
                 the whole risk of holding a framebuffer: dispatch::render() gets a fresh zeroed \
                 one every call and the warm path must clear it itself, or frame N composites \
                 on top of frame N-1",
                method.id().name()
            ),
        );
        assert_eq!(warm.frames(), 2, "the frame counter must count frames");

        let moved = scene_of(0.35);
        if let Err(e) = warm.set_scene(&moved) {
            panic!("set_scene failed: {e:?}");
        }
        if let Err(e) = warm.render_frame(&camera) {
            panic!("render_frame after set_scene failed: {e:?}");
        }
        let after = warm.resolve(settings.background);
        bit_identical(
            &after,
            &cold(method, scene_of(0.35)),
            &format!(
                "{}: a scene replaced through set_scene must render exactly as a cold render of \
                 that scene — otherwise the held strategy is carrying state from the scene it \
                 was built on",
                method.id().name()
            ),
        );
        let (peak, _, _) = after.max_abs();
        assert!(
            peak > 0.2,
            "{}: the warm scene must actually render, peak {peak}",
            method.id().name()
        );
        assert!(
            after.pixels != one.pixels,
            "{}: moving every atom by 0.35 must MOVE something, or set_scene did nothing and \
             the two comparisons above are the same comparison twice",
            method.id().name()
        );

        assert!(
            warm.resident_bytes() > warm.framebuffer().resident_bytes(),
            "{}: resident_bytes must count the scene and the strategy beside the framebuffer",
            method.id().name()
        );
        assert_eq!(
            warm.report().framebuffer_bytes,
            warm.framebuffer().resident_bytes(),
            "{}: the report must carry the framebuffer it actually holds",
            method.id().name()
        );
        checked += 1;
    }
    assert_eq!(
        checked,
        methods.len(),
        "every built method must be crossed with the warm path"
    );
}

#[test]
fn set_scene_refuses_an_empty_scene_because_only_the_strategy_build_checks_for_one() {
    let methods = RenderMethod::implemented();
    assert!(!methods.is_empty(), "no method is built");
    let shape = KernelFamily::Gaussian.shape();
    let settings = settings();
    for method in methods.iter() {
        let mut warm = match Renderer::<f64, Gaussian>::new(
            method,
            &settings,
            &shape,
            WIDTH,
            HEIGHT,
            &scene_of(0.0),
        ) {
            Ok(r) => r,
            Err(e) => panic!("Renderer::new failed: {e:?}"),
        };
        let empty = Scene::<f64>::new();
        match warm.set_scene(&empty) {
            Err(lights::error::LightsError::EmptyScene) => (),
            other => panic!(
                "{}: set_scene must refuse an empty scene with EmptyScene, got {other:?}. \
                 ImageFormation::build is the ONLY place in either method that rejects one \
                 (splat/implementation.rs:118, raycast/implementation.rs:114), so a set_scene \
                 that replaces the prepared scene without rebuilding accepts a scene the \
                 one-shot path refuses",
                method.id().name()
            ),
        }
        let refused = Renderer::<f64, Gaussian>::new(
            method, &settings, &shape, WIDTH, HEIGHT, &empty,
        )
        .err();
        match refused {
            Some(lights::error::LightsError::EmptyScene) => (),
            other => panic!(
                "{}: Renderer::new must refuse an empty scene too, got {other:?}",
                method.id().name()
            ),
        }
    }
}
