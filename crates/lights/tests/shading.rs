#![cfg(feature = "method-raycast")]
use lights::agreement::{ContractViolation, assert_contract};
use lights::dispatch::{Collect, Precision, RenderRequest, render};
use lights::error::LightsError;
use lights::image::FloatImage;
use lights::kernel::KernelFamily;
use lights::method::{MethodId, RenderMethod};
use lights::numerics::vector::{Rgb, Vec3};
use lights::scene::demo::{
    DEMO_OPACITY, SIGMA_PER_RADIUS, SPHERE_SCENE, single_atom_scene, sphere_camera,
    to_lights_frame,
};
use lights::scene::{Atom, Scene};
use lights::settings::{DirectionalLight, SharedSettings};

pub const WIDTH: u32 = 640;
pub const HEIGHT: u32 = 360;
pub const AMBIENT: f64 = 0.2;
pub const DIFFUSE: f64 = 0.8;
pub const LIT_RATIO_FLOOR: f64 = 2.0;
pub const FLAT_RATIO_TOLERANCE: f64 = 1.0e-9;
pub const MIRROR_TOLERANCE: f64 = 1.0e-9;

fn scene_of(atom: Atom<f64>) -> Scene<f64> {
    match single_atom_scene(atom) {
        Ok(s) => s,
        Err(e) => panic!("a one-atom scene failed validation: {e:?}"),
    }
}

fn red_sphere() -> Scene<f64> {
    let (centre, radius, colour) = SPHERE_SCENE[0];
    scene_of(Atom::isotropic(
        to_lights_frame(centre),
        radius * SIGMA_PER_RADIUS,
        DEMO_OPACITY,
        Rgb::new(colour[0], colour[1], colour[2]),
    ))
}

fn light_along_x(x: f64) -> DirectionalLight<f64> {
    match DirectionalLight::new(Vec3::new(x, 0.0, 0.0), AMBIENT, DIFFUSE) {
        Some(l) => l,
        None => panic!("a light along x with a zero direction"),
    }
}

fn settings(light: Option<DirectionalLight<f64>>) -> SharedSettings<f64> {
    let mut s = SharedSettings::<f64>::contract();
    s.background = Rgb::zero();
    s.light = light;
    s
}

fn request(
    method: RenderMethod,
    scene: Scene<f64>,
    light: Option<DirectionalLight<f64>>,
) -> RenderRequest {
    RenderRequest {
        precision: Precision::F64,
        kernel: KernelFamily::Gaussian,
        method,
        settings: settings(light),
        camera: sphere_camera(WIDTH, HEIGHT),
        scene,
        collect: Collect::PixelsAndCounters,
        shapes: None,
    }
}

fn raycast() -> RenderMethod {
    match RenderMethod::implemented()
        .into_iter()
        .find(|m| m.id() == MethodId::RayCast)
    {
        Some(m) => m,
        None => panic!(
            "this file is compiled only under method-raycast, and implemented() \
             does not list the ray caster"
        ),
    }
}

fn rendered(scene: Scene<f64>, light: Option<DirectionalLight<f64>>) -> FloatImage<f64> {
    match render(&request(raycast(), scene, light)) {
        Ok(outcome) => outcome.image,
        Err(e) => panic!("the ray caster refused the render: {e:?}"),
    }
}

fn luminance(p: Rgb<f64>) -> f64 {
    0.2126 * p.x + 0.7152 * p.y + 0.0722 * p.z
}

fn covered(p: Rgb<f64>) -> bool {
    p.x > 0.0 || p.y > 0.0 || p.z > 0.0
}

struct Halves {
    left: f64,
    right: f64,
    left_pixels: usize,
    right_pixels: usize,
}

fn halves(image: &FloatImage<f64>, mask: &FloatImage<f64>) -> Halves {
    let centre = (image.width as f64 - 1.0) * 0.5;
    let mut sum = [0.0f64; 2];
    let mut count = [0usize; 2];
    for y in 0..image.height {
        for x in 0..image.width {
            if !covered(mask.get(x, y)) {
                continue;
            }
            let side = if (x as f64) < centre { 0 } else { 1 };
            sum[side] += luminance(image.get(x, y));
            count[side] += 1;
        }
    }
    assert!(
        count[0] > 0 && count[1] > 0,
        "the mask covers {} pixels left of centre and {} right of it; a half \
         with no pixels makes the ratio a division by nothing",
        count[0],
        count[1]
    );
    Halves {
        left: sum[0] / count[0] as f64,
        right: sum[1] / count[1] as f64,
        left_pixels: count[0],
        right_pixels: count[1],
    }
}

fn side_holding(image: &FloatImage<f64>) -> usize {
    let centre = (image.width as f64 - 1.0) * 0.5;
    let mut count = [0usize; 2];
    for y in 0..image.height {
        for x in 0..image.width {
            if covered(image.get(x, y)) {
                count[if (x as f64) < centre { 0 } else { 1 }] += 1;
            }
        }
    }
    assert!(
        (count[0] == 0) != (count[1] == 0),
        "the probe atom must land entirely on one side of centre; it covers \
         {} pixels left and {} right",
        count[0],
        count[1]
    );
    if count[0] > 0 { 0 } else { 1 }
}

fn plus_x_side() -> usize {
    let probe = rendered(
        scene_of(Atom::isotropic(
            Vec3::new(0.6, 0.0, 3.0),
            0.5 * SIGMA_PER_RADIUS,
            DEMO_OPACITY,
            Rgb::splat(1.0),
        )),
        None,
    );
    side_holding(&probe)
}

fn lit_over_dark(h: &Halves, plus_x: usize) -> (f64, f64, f64) {
    let (lit, dark) = if plus_x == 1 { (h.right, h.left) } else { (h.left, h.right) };
    (lit / dark, lit, dark)
}

#[test]
fn a_light_from_plus_x_lights_the_plus_x_half_of_the_red_sphere_and_the_flat_render_is_symmetric() {
    let flat = rendered(red_sphere(), None);
    let f = halves(&flat, &flat);
    let flat_ratio = f.right / f.left;
    assert!(
        (flat_ratio - 1.0).abs() <= FLAT_RATIO_TOLERANCE,
        "the flat render of one isotropic atom on the optical axis must be \
         left-right symmetric; right/left luminance is {flat_ratio:.15} over \
         {} and {} pixels. If this moves, the mask is measuring something other \
         than the sphere",
        f.left_pixels,
        f.right_pixels
    );

    let plus_x = plus_x_side();
    let lit = rendered(red_sphere(), Some(light_along_x(1.0)));
    let (ratio, lit_mean, dark_mean) = lit_over_dark(&halves(&lit, &flat), plus_x);
    println!(
        "shading: flat right/left {flat_ratio:.12}; lit +x: lit-half mean {lit_mean:.6} \
         over dark-half mean {dark_mean:.6} = {ratio:.6}; pixels left {} right {}; \
         plus-x is screen side {plus_x}",
        f.left_pixels, f.right_pixels
    );
    assert!(
        ratio > LIT_RATIO_FLOOR,
        "under ambient {AMBIENT} and diffuse {DIFFUSE} an ideal Lambert \
         hemisphere pair gives (0.2 + 0.8 * 0.5) / 0.2 = 3.0; the gradient \
         normal gave {ratio:.6} ({lit_mean:.6} over {dark_mean:.6}), below the \
         floor {LIT_RATIO_FLOOR}. Below the floor the gradient normal is not a \
         usable normal"
    );
}

pub const SILHOUETTE_PREDICTION: f64 = (AMBIENT + DIFFUSE * 2.0 / std::f64::consts::PI) / AMBIENT;
pub const SILHOUETTE_TOLERANCE: f64 = 0.02;
pub const HEADLIGHT_CEILING: f64 = 0.30;

#[test]
fn the_gradient_normal_is_the_silhouette_normal_so_a_side_light_matches_two_over_pi_and_a_headlight_barely_shows() {
    let flat = rendered(red_sphere(), None);
    let plus_x = plus_x_side();
    let side = rendered(red_sphere(), Some(light_along_x(1.0)));
    let (ratio, _, _) = lit_over_dark(&halves(&side, &flat), plus_x);
    assert!(
        ((ratio - SILHOUETTE_PREDICTION) / SILHOUETTE_PREDICTION).abs() <= SILHOUETTE_TOLERANCE,
        "the closest-approach vector is perpendicular to the ray, so the normal it \
         gives is the SILHOUETTE normal, radial in the image plane: over a half \
         disc max(0, cos theta) averages 2/pi and the lit/dark ratio should be \
         {SILHOUETTE_PREDICTION:.4}; a front-facing sphere normal would average \
         4/(3 pi) and give 2.70. Measured {ratio:.4}"
    );

    let headlight = match DirectionalLight::new(Vec3::new(0.0, 0.0, -1.0), AMBIENT, DIFFUSE) {
        Some(l) => l,
        None => panic!("a headlight with a zero direction"),
    };
    let head = rendered(red_sphere(), Some(headlight));
    let mut lit_sum = 0.0;
    let mut flat_sum = 0.0;
    for y in 0..flat.height {
        for x in 0..flat.width {
            if covered(flat.get(x, y)) {
                lit_sum += luminance(head.get(x, y));
                flat_sum += luminance(flat.get(x, y));
            }
        }
    }
    let fraction = lit_sum / flat_sum;
    println!("shading: headlight keeps {fraction:.6} of the flat luminance (ambient is {AMBIENT})");
    assert!(
        fraction >= AMBIENT && fraction <= HEADLIGHT_CEILING,
        "a light AT the camera cannot reach a normal that is perpendicular to \
         every ray: the sphere should sit near ambient ({AMBIENT}) with only a \
         faint rim, never above {HEADLIGHT_CEILING} of flat. A true sphere under \
         a headlight would keep about 0.73. Measured {fraction:.6}. This is the \
         limitation L-141 records, not a defect"
    );
}

#[test]
fn flipping_the_light_to_minus_x_mirrors_the_render_exactly() {
    let flat = rendered(red_sphere(), None);
    let plus = halves(&rendered(red_sphere(), Some(light_along_x(1.0))), &flat);
    let minus = halves(&rendered(red_sphere(), Some(light_along_x(-1.0))), &flat);
    let forward = plus.right / plus.left;
    let mirrored = minus.left / minus.right;
    assert!(
        ((forward - mirrored) / forward).abs() <= MIRROR_TOLERANCE,
        "a light from -x must produce the mirror image of a light from +x: \
         right/left under +x is {forward:.15}, left/right under -x is \
         {mirrored:.15}"
    );
    assert!(
        (forward - 1.0).abs() > 0.5,
        "both renders are symmetric ({forward:.6}), so the light did nothing \
         and the mirror test passed by measuring nothing"
    );
}

#[test]
fn a_lit_pixel_is_never_negative_and_never_brighter_than_its_flat_self() {
    let flat = rendered(red_sphere(), None);
    let lit = rendered(red_sphere(), Some(light_along_x(1.0)));
    let mut darkest = f64::INFINITY;
    let mut checked = 0usize;
    for y in 0..flat.height {
        for x in 0..flat.width {
            let (a, b) = (flat.get(x, y), lit.get(x, y));
            for (f, l) in [(a.x, b.x), (a.y, b.y), (a.z, b.z)] {
                darkest = darkest.min(l);
                assert!(
                    l >= 0.0,
                    "pixel ({x},{y}) has a negative lit component {l}: the \
                     back hemisphere must be clamped to ambient, not driven \
                     below zero"
                );
                assert!(
                    l <= f + 1.0e-12,
                    "pixel ({x},{y}) is brighter lit ({l}) than flat ({f}); \
                     ambient {AMBIENT} plus diffuse {DIFFUSE} never exceeds one"
                );
                checked += 1;
            }
        }
    }
    assert!(checked == (3 * WIDTH * HEIGHT) as usize && darkest.is_finite());
}

#[test]
fn every_built_method_either_lights_the_scene_or_refuses_the_light_by_name() {
    let built = RenderMethod::implemented();
    let mut covered = 0usize;
    for method in built.iter() {
        let outcome = render(&request(method.clone(), red_sphere(), Some(light_along_x(1.0))));
        match (method.id(), outcome) {
            (MethodId::RayCast, Ok(_)) => covered += 1,
            (MethodId::RayCast, Err(e)) => panic!("the ray caster refused a light: {e:?}"),
            (other, Err(LightsError::LightingUnsupported { method: named })) => {
                assert_eq!(named, other, "the refusal names the wrong method");
                covered += 1;
            }
            (other, Ok(_)) => panic!(
                "{other:?} rendered under a light it has no normal to honour; it \
                 must refuse, not render flat"
            ),
            (other, Err(e)) => panic!("{other:?} refused for the wrong reason: {e:?}"),
        }
    }
    assert_eq!(
        covered,
        built.len(),
        "every built method must be judged; {covered} of {} were",
        built.len()
    );
    assert!(covered >= 1, "no method is built, so this case measured nothing");
}

#[test]
fn the_agreement_gate_refuses_a_light_rather_than_comparing_under_it() {
    let lit = settings(Some(light_along_x(1.0)));
    let flat = settings(None);
    assert_ne!(lit.fingerprint(), flat.fingerprint(), "the light is outside the fingerprint");
    match assert_contract(&lit, &flat, 0, 0, &red_sphere()) {
        Err(LightsError::ContractViolation(ContractViolation::LightPresent)) => (),
        Err(e) => panic!("refused for the wrong reason: {e:?}"),
        Ok(()) => panic!("the gate compared two methods under a light only one can render"),
    }
    match assert_contract(&lit, &lit, 0, 0, &red_sphere()) {
        Err(LightsError::ContractViolation(ContractViolation::LightPresent)) => (),
        Err(e) => panic!("refused for the wrong reason: {e:?}"),
        Ok(()) => panic!("two equal lit settings passed the gate; splat cannot render either"),
    }
}
