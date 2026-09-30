use lights::camera::{Camera, CameraSpec};
use lights::numerics::vector::{Vec2, Vec3};
use lights::scene::demo::{
    GROUND_INDEX, DEMO_HEIGHT, SPHERE_SCENE, DEMO_WIDTH, SIGMA_PER_RADIUS,
    flip_row, sphere_camera, sphere_exact_intrinsics_camera, sphere_intrinsics,
    spheres, spheres_no_ground, to_lights_frame,
};
use lights::scene::Scene;
use lights::scene::demo::stress_scene;
use lights::settings::{PixelConvention, REFERENCE_NEAR_PLANE};

const GOLDEN: [([f64; 3], f64, f64); 5] = [
    ([0.0, 0.0, -3.0], 899.500, 505.500),
    ([0.2, 0.5, -1.0], 1000.644, 758.250),
    ([0.5, 0.7, -1.0], 1152.359, 859.350),
    ([0.7, 0.8, -1.0], 1253.503, 909.900),
    ([0.9, 0.2, -1.0], 1354.647, 606.600),
];

fn camera_of(spec: &CameraSpec) -> Camera<f64> {
    match Camera::<f64>::from_spec(spec, REFERENCE_NEAR_PLANE) {
        Ok(c) => c,
        Err(e) => panic!("camera: {e:?}"),
    }
}

fn project(camera: &Camera<f64>, world: Vec3<f64>) -> Vec2<f64> {
    let cam = camera.to_camera(world);
    match camera.project(cam) {
        Some(p) => p,
        None => panic!("{world:?} projected to nothing"),
    }
}

#[test]
fn sphere_intrinsics_reproduced() {
    let (focal, principal) = sphere_intrinsics(DEMO_WIDTH, DEMO_HEIGHT);
    assert!(
        (focal[0] - 505.71888888888885).abs() < 1e-9,
        "f_x is (W-1)/viewport_width with viewport_width = 2*W/H, because BOTH \
         image_width and image_height carry clap default_values, so the source \
         aspect_ratio is recomputed as 1800/1012 and is NOT the 16/9 constant: {}",
        focal[0]
    );
    assert!((focal[1] - 505.5).abs() < 1e-12, "f_y is (H-1)/2: {}", focal[1]);
    assert!(
        (focal[0] / focal[1] - 1.0).abs() > 1.0e-4,
        "f_x and f_y differ by 0.0433% and must be asserted independently; \
         deriving one from the other through the aspect ratio silently \
         reintroduces the 16/9 constant"
    );
    assert_eq!(principal, [899.5, 505.5]);

    let spec = sphere_exact_intrinsics_camera(DEMO_WIDTH, DEMO_HEIGHT);
    assert_eq!(spec.convention, PixelConvention::CornerSample);
    let camera = camera_of(&spec);
    assert_eq!(camera.focal.x, focal[0]);
    assert_eq!(camera.focal.y, focal[1]);
    assert_eq!(camera.principal.x, 899.5);
    assert_eq!(camera.principal.y, 505.5);
}

#[test]
fn sphere_centre_reprojection() {
    let camera = camera_of(&sphere_exact_intrinsics_camera(DEMO_WIDTH, DEMO_HEIGHT));
    for (centre, golden_u, golden_buffer_row) in GOLDEN {
        let pixel = project(&camera, to_lights_frame(centre));
        let buffer_row = flip_row(pixel.y, DEMO_HEIGHT);
        assert!(
            (pixel.x - golden_u).abs() < 0.05,
            "{centre:?}: column {} against the surveyed {golden_u}",
            pixel.x
        );
        assert!(
            (buffer_row - golden_buffer_row).abs() < 0.05,
            "{centre:?}: source buffer row {buffer_row} against the surveyed \
             {golden_buffer_row}. the source's v = j/(H-1) makes buffer row 0 the \
             BOTTOM of the viewport, so lights_row = (H-1) - buffer_row",
            
        );
    }
}

#[test]
fn the_reprojection_gate_fails_on_a_wrong_camera() {
    let good = camera_of(&sphere_exact_intrinsics_camera(DEMO_WIDTH, DEMO_HEIGHT));

    let square = camera_of(
        &sphere_camera(DEMO_WIDTH, DEMO_HEIGHT).with_intrinsics(
            [505.5, 505.5],
            [899.5, 505.5],
        ),
    );
    let swapped = camera_of(
        &sphere_camera(DEMO_WIDTH, DEMO_HEIGHT).with_intrinsics(
            [505.5, 505.71888888888885],
            [899.5, 505.5],
        ),
    );
    let half_pixel = camera_of(
        &sphere_camera(DEMO_WIDTH, DEMO_HEIGHT).with_intrinsics(
            [505.71888888888885, 505.5],
            [900.0, 506.0],
        ),
    );
    let y_up = camera_of(&CameraSpec::look_at(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        90.0,
        DEMO_WIDTH,
        DEMO_HEIGHT,
    )
    .with_intrinsics([505.71888888888885, 505.5], [899.5, 505.5]));

    let mut broken = 0usize;
    for (name, camera) in [
        ("square focal", square),
        ("swapped f_x and f_y", swapped),
        ("half-pixel principal shift", half_pixel),
        ("y-up world frame", y_up),
    ] {
        let mut worst = 0.0f64;
        for (centre, _, _) in GOLDEN {
            let world = to_lights_frame(centre);
            let a = project(&good, world);
            let b = project(&camera, world);
            worst = worst
                .max((a.x - b.x).abs())
                .max((a.y - b.y).abs());
        }
        assert!(
            worst > 0.05,
            "the reprojection gate must REPORT a wrong camera: {name} moved the \
             worst sphere centre by only {worst} px"
        );
        broken += 1;
    }
    assert_eq!(broken, 4, "every deliberately wrong camera must be caught");
}

#[test]
fn the_demo_scene_is_the_sphere_geometry() {
    assert_eq!(SPHERE_SCENE.len(), 6);
    let scene = spheres::<f64>();
    assert_eq!(scene.len(), 6);
    for (atom, (centre, radius, colour)) in scene.atoms().iter().zip(SPHERE_SCENE.iter()) {
        assert_eq!(atom.mean, to_lights_frame(*centre));
        let sigma = radius * SIGMA_PER_RADIUS;
        assert!((atom.scale.x - sigma).abs() < 1e-12);
        assert_eq!(atom.scale.x, atom.scale.y);
        assert_eq!(atom.scale.y, atom.scale.z);
        assert_eq!(atom.colour.x, colour[0]);
        let three_sigma = 3.0 * sigma;
        assert!(
            (three_sigma - radius).abs() < 1e-12,
            "sigma = R/3 puts the 3-sigma isosurface exactly on the sphere surface, \
             which is what makes the silhouette diagnostic meaningful"
        );
    }

    let no_ground = spheres_no_ground::<f64>();
    assert_eq!(no_ground.len(), 5);
    let ground = to_lights_frame(SPHERE_SCENE[GROUND_INDEX].0);
    assert!(
        no_ground.atoms().iter().all(|a| a.mean != ground),
        "the ground atom must be gone"
    );
}

#[test]
fn the_ground_sphere_is_outside_the_agreement_contract_by_construction() {
    let camera = camera_of(&sphere_exact_intrinsics_camera(DEMO_WIDTH, DEMO_HEIGHT));
    let limits = camera.jacobian_limits();
    let ground = to_lights_frame(SPHERE_SCENE[GROUND_INDEX].0);
    let cam = camera.to_camera(ground);
    let ratio = (cam.y / cam.z).abs();
    assert!(
        ratio > 50.0 * limits.y,
        "the ground sphere's off-axis ratio is {ratio} against a clamp limit of \
         {}; it is outside the S22-P6 contract BY CONSTRUCTION, not by \
         convenience",
        limits.y
    );
    let (_, fired) = camera.clamp_for_jacobian(cam);
    assert!(fired, "the Jacobian clamp must actually fire for the ground atom");

    for (index, (centre, _, _)) in SPHERE_SCENE.iter().enumerate() {
        if index == GROUND_INDEX {
            continue;
        }
        let cam = camera.to_camera(to_lights_frame(*centre));
        let (_, fired) = camera.clamp_for_jacobian(cam);
        assert!(
            !fired,
            "sphere {index} at {centre:?} must sit inside the clamp so it stays \
             inside the contract"
        );
    }
}

#[test]
fn the_stress_scene_is_deterministic_and_its_clustering_knob_is_live() {
    let a = stress_scene::<f64>(600, 0xabcd, 0.0);
    let again = stress_scene::<f64>(600, 0xabcd, 0.0);
    assert_eq!(a.len(), 600);
    assert_eq!(
        a.atoms(),
        again.atoms(),
        "the LCG must make the stress scene reproducible, or no benchmark number \
         taken on it is comparable to any other"
    );
    let other_seed = stress_scene::<f64>(600, 0x1234, 0.0);
    assert_ne!(a.atoms(), other_seed.atoms(), "the seed must be live");

    let spread = |scene: &Scene<f64>| -> f64 {
        let n = scene.len() as f64;
        let mut mean = Vec3::new(0.0, 0.0, 0.0);
        for atom in scene.atoms() {
            mean = mean + atom.mean;
        }
        let mean = mean * (1.0 / n);
        let mut acc = 0.0;
        for atom in scene.atoms() {
            acc += (atom.mean - mean).length_squared();
        }
        (acc / n).sqrt()
    };

    let uniform = spread(&a);
    let clustered = spread(&stress_scene::<f64>(600, 0xabcd, 1.0));
    println!("stress spread: uniform {uniform}, clustered {clustered}");
    assert!(
        clustered < 0.85 * uniform,
        "the clustering argument must actually concentrate the scene, or the \
         spatial-clustering axis of the benchmark is dead configuration: \
         uniform {uniform}, clustered {clustered}"
    );

    let mut nearest_uniform = 0.0f64;
    let mut nearest_clustered = 0.0f64;
    for (scene, out) in [
        (stress_scene::<f64>(200, 7, 0.0), &mut nearest_uniform),
        (stress_scene::<f64>(200, 7, 1.0), &mut nearest_clustered),
    ] {
        let atoms = scene.atoms();
        let mut total = 0.0;
        for (i, a) in atoms.iter().enumerate() {
            let mut best = f64::INFINITY;
            for (j, b) in atoms.iter().enumerate() {
                if i != j {
                    best = best.min((a.mean - b.mean).length());
                }
            }
            total += best;
        }
        *out = total / atoms.len() as f64;
    }
    println!(
        "mean nearest-neighbour distance: uniform {nearest_uniform}, clustered \
         {nearest_clustered}"
    );
    assert!(
        nearest_clustered < nearest_uniform,
        "clustered atoms must sit closer to one another: {nearest_clustered} \
         against {nearest_uniform}"
    );
}
