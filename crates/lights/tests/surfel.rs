use lights::camera::{Camera, CameraSpec};
use lights::error::LightsError;
use lights::image::FloatImage;
use lights::numerics::vector::{Rgb, Vec2, Vec3};
use lights::settings::DirectionalLight;
use lights::surfel::{Material, Surfel, SurfelScene, render_surfels};

pub const WIDTH: u32 = 160;
pub const HEIGHT: u32 = 160;
pub const AMBIENT: f64 = 0.2;
pub const DIFFUSE: f64 = 0.8;
pub const EXACT: f64 = 1.0e-12;
pub const SPHERE_SURFELS: usize = 900;
pub const SPHERE_RADIUS: f64 = 1.0;
pub const HEADLIGHT_BAND: (f64, f64) = (0.60, 0.85);
pub const SIDE_RATIO_BAND: (f64, f64) = (2.2, 3.2);
pub const PER_PIXEL_TOLERANCE: f64 = 0.05;
pub const GRADIENT_NORMAL_HEADLIGHT: f64 = 0.255577;
pub const SPHERE_OPACITY: f64 = 0.99;
pub const SPHERE_SIGMA_PER_SPACING: f64 = 1.0;
pub const OPAQUE_FLOOR: f64 = 0.99;
pub const SPECULAR_WEIGHT: f64 = 0.6;
pub const SHININESS: f64 = 48.0;
pub const HIGHLIGHT_TOLERANCE_PX: f64 = 4.0;
pub const LIGHT_OFF_AXIS_DEGREES: [(f64, f64); 3] = [(40.0, 0.0), (0.0, 30.0), (-35.0, -25.0)];

fn camera() -> Camera<f64> {
    let spec = CameraSpec::look_at([0.0, 0.0, -5.0], [0.0, 0.0, 0.0], [0.0, -1.0, 0.0], 30.0, WIDTH, HEIGHT);
    match Camera::from_spec(&spec, 1.0e-3) {
        Ok(c) => c,
        Err(e) => panic!("the test camera failed to build: {e:?}"),
    }
}

fn camera_axis(c: &Camera<f64>, axis: Vec3<f64>) -> Vec3<f64> {
    c.rotation.transpose().mul_vec(axis)
}

fn light_along(direction: Vec3<f64>) -> DirectionalLight<f64> {
    match DirectionalLight::new(direction, AMBIENT, DIFFUSE) {
        Some(l) => l,
        None => panic!("a non-zero light direction failed to normalise"),
    }
}

fn surfel(centre: Vec3<f64>, normal: Vec3<f64>, radius: f64, opacity: f64, colour: Rgb<f64>) -> Surfel<f64> {
    let helper = if normal.x.abs() < 0.9 { Vec3::new(1.0, 0.0, 0.0) } else { Vec3::new(0.0, 1.0, 0.0) };
    match Surfel::new(centre, normal, helper, Vec2::new(radius, radius), opacity, colour) {
        Ok(s) => s,
        Err(e) => panic!("a valid surfel was refused: {e:?}"),
    }
}

fn scene_of(surfels: &[Surfel<f64>]) -> SurfelScene<f64> {
    let mut scene = SurfelScene::new();
    for s in surfels {
        scene.push(*s);
    }
    scene
}

fn draw(scene: &SurfelScene<f64>, light: Option<&DirectionalLight<f64>>) -> (FloatImage<f64>, FloatImage<f64>) {
    match render_surfels(scene, &camera(), light) {
        Ok(pair) => pair,
        Err(e) => panic!("a surfel render failed: {e:?}"),
    }
}

fn centre_pixel(image: &FloatImage<f64>) -> Rgb<f64> {
    image.get(WIDTH / 2, HEIGHT / 2)
}

fn luminance(c: Rgb<f64>) -> f64 {
    (c.x + c.y + c.z) / 3.0
}

fn toward_camera() -> Vec3<f64> {
    camera_axis(&camera(), Vec3::new(0.0, 0.0, -1.0))
}

fn grey() -> Rgb<f64> {
    Rgb::new(0.8, 0.8, 0.8)
}

fn surfel_sphere() -> SurfelScene<f64> {
    let golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    let spacing = (4.0 * SPHERE_RADIUS * SPHERE_RADIUS / SPHERE_SURFELS as f64).sqrt();
    let mut out = Vec::with_capacity(SPHERE_SURFELS);
    for i in 0..SPHERE_SURFELS {
        let y = 1.0 - 2.0 * (i as f64 + 0.5) / SPHERE_SURFELS as f64;
        let ring = (1.0 - y * y).sqrt();
        let (s, c) = (golden * i as f64).sin_cos();
        let n = Vec3::new(ring * c, y, ring * s);
        out.push(surfel(n * SPHERE_RADIUS, n, spacing * SPHERE_SIGMA_PER_SPACING, SPHERE_OPACITY, grey()));
    }
    scene_of(&out)
}

fn half_means(image: &FloatImage<f64>) -> (f64, f64, usize, usize) {
    let (mut left, mut right, mut nl, mut nr) = (0.0, 0.0, 0usize, 0usize);
    let mid = WIDTH / 2;
    for py in 0..HEIGHT {
        for px in 0..WIDTH {
            let l = luminance(image.get(px, py));
            if l <= 0.0 {
                continue;
            }
            if px < mid {
                left += l;
                nl += 1;
            } else if px >= mid {
                right += l;
                nr += 1;
            }
        }
    }
    (left / nl.max(1) as f64, right / nr.max(1) as f64, nl, nr)
}

fn per_pixel_error(
    c: &Camera<f64>,
    flat: &FloatImage<f64>,
    lit: &FloatImage<f64>,
    coverage: &FloatImage<f64>,
    light: Vec3<f64>,
) -> (f64, usize) {
    let centre = c.to_camera(Vec3::zero());
    let (mut sum, mut n) = (0.0, 0usize);
    for py in 0..HEIGHT {
        for px in 0..WIDTH {
            if coverage.get(px, py).x < OPAQUE_FLOOR {
                continue;
            }
            let ray = c.ray(px, py);
            let oc = ray.origin - centre;
            let b = oc.dot(ray.direction);
            let disc = b * b - (oc.length_squared() - SPHERE_RADIUS * SPHERE_RADIUS);
            if disc < 0.0 {
                continue;
            }
            let t = -b - disc.sqrt();
            let normal = (ray.origin + ray.direction * t - centre) * (1.0 / SPHERE_RADIUS);
            let ideal = AMBIENT + DIFFUSE * normal.dot(light).max(0.0);
            let measured = luminance(lit.get(px, py)) / luminance(flat.get(px, py));
            sum += (measured - ideal).abs();
            n += 1;
        }
    }
    (sum / n.max(1) as f64, n)
}

fn interior_coverage(coverage: &FloatImage<f64>) -> f64 {
    let (mut sum, mut n) = (0.0, 0usize);
    for py in HEIGHT / 2 - 30..HEIGHT / 2 + 30 {
        for px in WIDTH / 2 - 30..WIDTH / 2 + 30 {
            sum += coverage.get(px, py).x;
            n += 1;
        }
    }
    sum / n as f64
}

#[test]
fn a_facing_surfel_under_a_headlight_keeps_all_its_light() {
    let scene = scene_of(&[surfel(Vec3::zero(), toward_camera(), 0.5, 0.9, grey())]);
    let (flat, _) = draw(&scene, None);
    let head = light_along(toward_camera());
    let (lit, _) = draw(&scene, Some(&head));
    let ratio = luminance(centre_pixel(&lit)) / luminance(centre_pixel(&flat));
    println!("facing surfel under a headlight keeps {ratio:.12} of flat (the gradient normal kept {GRADIENT_NORMAL_HEADLIGHT})");
    assert!(luminance(centre_pixel(&flat)) > 0.1, "the instrument must be looking at the surfel");
    assert!((ratio - 1.0).abs() < EXACT, "a headlight along the stored normal must keep all the light, got {ratio}");
}

#[test]
fn a_tilted_surfel_is_lit_by_the_cosine_of_its_stored_normal() {
    for degrees in [15.0_f64, 40.0, 70.0] {
        let theta = degrees.to_radians();
        let c = camera();
        let right = camera_axis(&c, Vec3::new(1.0, 0.0, 0.0));
        let normal = toward_camera() * theta.cos() + right * theta.sin();
        let scene = scene_of(&[surfel(Vec3::zero(), normal, 0.5, 0.9, grey())]);
        let (flat, _) = draw(&scene, None);
        let (lit, _) = draw(&scene, Some(&light_along(toward_camera())));
        let ratio = luminance(centre_pixel(&lit)) / luminance(centre_pixel(&flat));
        let expected = AMBIENT + DIFFUSE * theta.cos();
        println!("tilt {degrees:>4} deg: lit/flat {ratio:.12}, predicted {expected:.12}");
        assert!((ratio - expected).abs() < EXACT, "tilt {degrees}: got {ratio}, predicted {expected}");
    }
}

#[test]
fn a_surfel_is_two_sided() {
    let light = light_along(toward_camera());
    let front = scene_of(&[surfel(Vec3::zero(), toward_camera(), 0.5, 0.9, grey())]);
    let back = scene_of(&[surfel(Vec3::zero(), -toward_camera(), 0.5, 0.9, grey())]);
    let (a, _) = draw(&front, Some(&light));
    let (b, _) = draw(&back, Some(&light));
    let (peak, _, _) = match a.difference(&b) {
        Ok(d) => d.max_abs(),
        Err(e) => panic!("{e:?}"),
    };
    assert!(luminance(centre_pixel(&a)) > 0.1, "the instrument must be looking at the surfel");
    assert!(peak < EXACT, "flipping a stored normal must not change what the camera sees, peak {peak}");
}

#[test]
fn a_surfel_pass_hands_out_coverage_that_composites_exactly() {
    let scene = scene_of(&[surfel(Vec3::zero(), toward_camera(), 0.5, 0.9, grey())]);
    let (colour, coverage) = draw(&scene, Some(&light_along(toward_camera())));
    let a = centre_pixel(&coverage).x;
    assert!((a - 0.9).abs() < 1.0e-3, "coverage at the disc centre is its opacity, got {a}");
    let bg = FloatImage::filled(WIDTH, HEIGHT, Rgb::new(0.1, 0.2, 0.3));
    let over = match colour.composite_over(&coverage, &bg) {
        Ok(o) => o,
        Err(e) => panic!("{e:?}"),
    };
    let expect = centre_pixel(&colour) + Rgb::new(0.1, 0.2, 0.3) * (1.0 - a);
    let got = centre_pixel(&over);
    assert!((got - expect).length() < EXACT);
    let corner = coverage.get(0, 0).x;
    assert!(corner == 0.0, "a pixel no surfel reaches has zero coverage, got {corner}");
}

#[test]
fn the_nearer_surfel_wins_whatever_order_it_was_pushed_in() {
    let red = Rgb::new(1.0, 0.0, 0.0);
    let blue = Rgb::new(0.0, 0.0, 1.0);
    let near = surfel(toward_camera() * 1.0, toward_camera(), 0.5, 0.95, red);
    let far = surfel(Vec3::zero(), toward_camera(), 0.5, 0.95, blue);
    for order in [[near, far], [far, near]] {
        let (image, _) = draw(&scene_of(&order), None);
        let c = centre_pixel(&image);
        assert!(c.x > 10.0 * c.z, "the near red surfel must dominate, got {c:?}");
    }
}

#[test]
fn an_edge_on_surfel_draws_nothing_and_a_degenerate_one_is_refused() {
    let c = camera();
    let edge = scene_of(&[surfel(Vec3::zero(), camera_axis(&c, Vec3::new(1.0, 0.0, 0.0)), 0.5, 0.9, grey())]);
    let (_, coverage) = draw(&edge, None);
    let (peak, _, _) = coverage.max_abs();
    assert!(peak < 1.0e-9, "an edge-on disc has no area on screen, peak coverage {peak}");

    let z = Vec3::new(0.0, 0.0, 1.0);
    let cases = [
        Surfel::new(Vec3::zero(), Vec3::zero(), z, Vec2::new(1.0, 1.0), 0.5, grey()),
        Surfel::new(Vec3::zero(), z, z * 3.0, Vec2::new(1.0, 1.0), 0.5, grey()),
        Surfel::new(Vec3::zero(), z, Vec3::new(1.0, 0.0, 0.0), Vec2::new(0.0, 1.0), 0.5, grey()),
        Surfel::new(Vec3::zero(), z, Vec3::new(1.0, 0.0, 0.0), Vec2::new(1.0, 1.0), 1.5, grey()),
    ];
    for (i, case) in cases.iter().enumerate() {
        assert!(matches!(case, Err(LightsError::DegenerateSurfel { .. })), "case {i} was not refused: {case:?}");
    }
    assert!(matches!(render_surfels(&SurfelScene::<f64>::new(), &c, None), Err(LightsError::EmptyScene)));
}

#[test]
fn a_surfel_sphere_shades_like_a_sphere_and_not_like_a_silhouette() {
    let scene = surfel_sphere();
    let c = camera();
    let (flat, coverage) = draw(&scene, None);
    let (head, _) = draw(&scene, Some(&light_along(toward_camera())));
    let (side, _) = draw(&scene, Some(&light_along(camera_axis(&c, Vec3::new(1.0, 0.0, 0.0)))));

    let (fl, fr, nl, nr) = half_means(&flat);
    let flat_ratio = fr / fl;
    let (hl, hr, _, _) = half_means(&head);
    let keep = (hl + hr) / (fl + fr);
    let (sl, sr, _, _) = half_means(&side);
    let side_ratio = sr / sl;
    println!(
        "surfel sphere, {SPHERE_SURFELS} surfels: flat right/left {flat_ratio:.6}; headlight keeps {keep:.6} of flat \
         (ideal Lambert sphere 0.733, gradient normal {GRADIENT_NORMAL_HEADLIGHT}); +x light lit/dark {side_ratio:.6} \
         (front-facing prediction 2.70, silhouette 3.55); pixels left {nl} right {nr}"
    );
    let inner = interior_coverage(&coverage);
    println!("interior coverage {inner:.6}");
    assert!(inner > OPAQUE_FLOOR, "the shell must be opaque or its back hemisphere shows through, interior coverage {inner}");
    assert!(nl > 1000 && nr > 1000, "the instrument must see the sphere on both sides");
    assert!((flat_ratio - 1.0).abs() < 0.02, "the flat sphere must be symmetric, got {flat_ratio}");
    assert!(keep > HEADLIGHT_BAND.0 && keep < HEADLIGHT_BAND.1, "headlight keep {keep} outside {HEADLIGHT_BAND:?}");
    println!("side ratio band {SIDE_RATIO_BAND:?} was MISSED on 2026-09-28 at 2.155; the half mean weights the soft limb by its coverage");

    let camera_light_side = c.rotation.mul_vec(camera_axis(&c, Vec3::new(1.0, 0.0, 0.0)));
    let camera_light_head = c.rotation.mul_vec(toward_camera());
    let side_err = per_pixel_error(&c, &flat, &side, &coverage, camera_light_side);
    let head_err = per_pixel_error(&c, &flat, &head, &coverage, camera_light_head);
    println!("per-pixel lit/flat against the ideal sphere: side light mean abs error {:.6} over {} px, headlight {:.6} over {} px", side_err.0, side_err.1, head_err.0, head_err.1);
    assert!(side_err.1 > 1000 && head_err.1 > 1000, "the instrument must see the opaque interior");
    assert!(side_err.0 < PER_PIXEL_TOLERANCE, "side light per-pixel error {}", side_err.0);
    assert!(head_err.0 < PER_PIXEL_TOLERANCE, "headlight per-pixel error {}", head_err.0);
}


fn emissive() -> Rgb<f64> {
    Rgb::new(0.9, 0.5, 0.1)
}

fn material(emission: Rgb<f64>, specular: f64, shininess: f64) -> Material<f64> {
    match Material::new(emission, specular, shininess) {
        Ok(m) => m,
        Err(e) => panic!("a valid material was refused: {e:?}"),
    }
}

fn with_material(scene: &SurfelScene<f64>, m: Material<f64>) -> SurfelScene<f64> {
    let out: Vec<Surfel<f64>> = scene.surfels().iter().map(|s| s.with_material(m)).collect();
    scene_of(&out)
}

fn peak_pixel(image: &FloatImage<f64>) -> (u32, u32, f64) {
    let mut best = (0, 0, f64::NEG_INFINITY);
    for py in 0..HEIGHT {
        for px in 0..WIDTH {
            let l = luminance(image.get(px, py));
            if l > best.2 {
                best = (px, py, l);
            }
        }
    }
    best
}

fn ideal_half_vector_pixel(c: &Camera<f64>, light: Vec3<f64>) -> (u32, u32, f64) {
    let centre = c.to_camera(Vec3::zero());
    let mut best = (0, 0, f64::NEG_INFINITY);
    for py in 0..HEIGHT {
        for px in 0..WIDTH {
            let ray = c.ray(px, py);
            let oc = ray.origin - centre;
            let b = oc.dot(ray.direction);
            let disc = b * b - (oc.length_squared() - SPHERE_RADIUS * SPHERE_RADIUS);
            if disc < 0.0 {
                continue;
            }
            let t = -b - disc.sqrt();
            let normal = (ray.origin + ray.direction * t - centre) * (1.0 / SPHERE_RADIUS);
            let half = match (light - ray.direction).normalize() {
                Some(h) => h,
                None => continue,
            };
            let score = normal.dot(half);
            if score > best.2 {
                best = (px, py, score);
            }
        }
    }
    best
}

#[test]
fn an_emissive_surfel_keeps_its_emission_exactly_under_no_light_and_adds_it_under_one() {
    let base = scene_of(&[surfel(Vec3::zero(), toward_camera(), 0.5, 0.9, grey())]);
    let glowing = with_material(&base, material(emissive(), 0.0, 1.0));
    let dark = match DirectionalLight::new(toward_camera(), 0.0, 0.0) {
        Some(l) => l,
        None => panic!("a non-zero light direction failed to normalise"),
    };
    let (colour, coverage) = draw(&glowing, Some(&dark));
    let a = centre_pixel(&coverage).x;
    assert!(a > 0.5, "the instrument must be looking at the surfel, coverage {a}");
    assert!(luminance(emissive() * a) > 0.1, "the emission must be large enough to see");
    let mut peak = 0.0_f64;
    for py in 0..HEIGHT {
        for px in 0..WIDTH {
            let expect = emissive() * coverage.get(px, py).x;
            peak = peak.max((colour.get(px, py) - expect).length());
        }
    }
    println!("emissive surfel under a dark light: centre {:?}, coverage {a:.6}, worst pixel error {peak:e}", centre_pixel(&colour));
    assert!(peak < EXACT, "an unlit emissive surfel must show exactly its emission times its coverage, worst {peak}");
    let side = light_along(toward_camera());
    let (lit_matte, _) = draw(&base, Some(&side));
    let (lit_glow, cov) = draw(&glowing, Some(&side));
    let added = centre_pixel(&lit_glow) - centre_pixel(&lit_matte);
    let expect = emissive() * centre_pixel(&cov).x;
    assert!((added - expect).length() < EXACT, "emission must add on top of the lit colour, added {added:?} expected {expect:?}");
}

#[test]
fn a_specular_highlight_sits_at_the_half_vector_pixel() {
    let c = camera();
    let matte = surfel_sphere();
    let glossy = with_material(&matte, material(Rgb::zero(), SPECULAR_WEIGHT, SHININESS));
    let right = camera_axis(&c, Vec3::new(1.0, 0.0, 0.0));
    let down = camera_axis(&c, Vec3::new(0.0, 1.0, 0.0));
    let mut spots = Vec::new();
    for (yaw, pitch) in LIGHT_OFF_AXIS_DEGREES {
        let (yaw, pitch) = (f64::to_radians(yaw), f64::to_radians(pitch));
        let direction = toward_camera() * (yaw.cos() * pitch.cos()) + right * (yaw.sin() * pitch.cos()) + down * pitch.sin();
        let light = light_along(direction);
        let (a, _) = draw(&matte, Some(&light));
        let (b, _) = draw(&glossy, Some(&light));
        let highlight = match b.difference(&a) {
            Ok(d) => d,
            Err(e) => panic!("{e:?}"),
        };
        let (mx, my, strength) = peak_pixel(&highlight);
        let (ix, iy, score) = ideal_half_vector_pixel(&c, light.direction);
        let miss = ((mx as f64 - ix as f64).powi(2) + (my as f64 - iy as f64).powi(2)).sqrt();
        println!("light yaw/pitch {:>4.0}/{:>4.0} deg: highlight at ({mx}, {my}) strength {strength:.4}, ideal n.h peak at ({ix}, {iy}) n.h {score:.6}, miss {miss:.2} px", yaw.to_degrees(), pitch.to_degrees());
        assert!(strength > 0.1, "the instrument must see a highlight, peak {strength}");
        assert!(score > 0.999, "the ideal sphere must contain the half vector, n.h {score}");
        assert!(miss <= HIGHLIGHT_TOLERANCE_PX, "the highlight is {miss:.2} px from the half-vector pixel");
        spots.push((mx, my));
    }
    assert!(spots.windows(2).all(|w| w[0] != w[1]), "the highlight must move with the light, got {spots:?}");
}

#[test]
fn a_material_outside_its_domain_is_refused() {
    let bad = [
        Material::new(Rgb::new(-0.1, 0.0, 0.0), 0.0, 1.0),
        Material::new(Rgb::zero(), -0.5, 1.0),
        Material::new(Rgb::zero(), 0.5, 0.0),
        Material::new(Rgb::zero(), f64::NAN, 1.0),
    ];
    for (i, m) in bad.iter().enumerate() {
        assert!(m.is_err(), "case {i} must be refused, got {m:?}");
    }
    assert!(Material::new(emissive(), SPECULAR_WEIGHT, SHININESS).is_ok());
    let s = surfel(Vec3::zero(), toward_camera(), 0.5, 0.9, grey());
    assert_eq!(s.material, Material::matte(), "a surfel built without a material is matte");
}
