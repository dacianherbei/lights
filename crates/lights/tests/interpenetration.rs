#![cfg(feature = "method-raycast")]

use lights::blend::{BlendMode, Over, OverAccum, Step};
use lights::camera::{Camera, CameraSpec};
use lights::dispatch::{Collect, Precision, RenderRequest, render, render_depth_limited};
use lights::error::LightsError;
use lights::image::{DepthImage, FloatImage};
use lights::kernel::KernelFamily;
use lights::kernel::gaussian::Gaussian;
use lights::method::RenderMethod;
use lights::method::raycast::execute::{Hit, collect_hits};
use lights::method::raycast::prepare::{RayAtom, prepare_atoms, to_world_ray};
use lights::numerics::vector::{Rgb, Vec2, Vec3};
use lights::probe::NoProbe;
use lights::renderer::Renderer;
use lights::scene::prepare::PreparedScene;
use lights::scene::{Atom, Scene};
use lights::settings::{DirectionalLight, OrderingModel, SharedSettings};
use lights::surfel::{
    OPAQUE_TRANSMITTANCE, Surfel, SurfelHit, SurfelLayer, SurfelScene, intersect_surfel, render_surfel_layer, render_surfels,
    to_camera_frame,
};

pub const WIDTH: u32 = 160;
pub const HEIGHT: u32 = 120;
pub const VISIBLE: f64 = 1.0 / 255.0;
pub const GROSS: f64 = 0.1;
pub const SEEN: f64 = 1.0e-3;
pub const EXACT: f64 = 1.0e-12;
pub const ATOM_SIGMA: f64 = 0.35;
pub const ATOM_OPACITY: f64 = 0.9;
pub const HEIGHT_IN_SIGMAS: [f64; 5] = [1.0, 0.5, 0.0, -0.5, -1.0];
pub const SWEEP_IN_SIGMAS: [f64; 9] = [-1.0, 0.0, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0];
pub const LIFT_IN_SIGMAS: f64 = 6.0;
pub const FLOOR_SPACING: f64 = 0.25;
pub const FLOOR_HALF_X: i32 = 14;
pub const FLOOR_Z: (i32, i32) = (-10, 18);
pub const FLOOR_OPACITY: f64 = 0.99;
pub const MAX_HITS: u64 = 4096;

fn spec() -> CameraSpec {
    CameraSpec::look_at([0.0, -2.5, -5.0], [0.0, 0.0, 0.5], [0.0, -1.0, 0.0], 45.0, WIDTH, HEIGHT)
}

fn light() -> DirectionalLight<f64> {
    match DirectionalLight::new(Vec3::new(0.3, -1.0, -0.4), 0.25, 0.75) {
        Some(l) => l,
        None => panic!("a non-zero light direction failed to normalise"),
    }
}

fn settings() -> SharedSettings<f64> {
    SharedSettings {
        ordering: OrderingModel::HitDepth,
        light: Some(light()),
        threads: 1,
        ..SharedSettings::contract()
    }
}

fn atoms(lift: f64) -> Scene<f64> {
    let heights: Vec<f64> = HEIGHT_IN_SIGMAS.iter().map(|h| h + lift).collect();
    atoms_at(&heights)
}

fn atoms_at(heights: &[f64]) -> Scene<f64> {
    let colours = [
        Rgb::new(0.9, 0.2, 0.2),
        Rgb::new(0.2, 0.8, 0.3),
        Rgb::new(0.2, 0.3, 0.9),
        Rgb::new(0.9, 0.8, 0.2),
        Rgb::new(0.8, 0.3, 0.8),
    ];
    let mut scene = Scene::new();
    for (i, (height, colour)) in heights.iter().zip(colours.iter()).enumerate() {
        let x = -1.6 + 0.8 * i as f64;
        let y = -height * ATOM_SIGMA;
        let atom = Atom::isotropic(Vec3::new(x, y, 0.5), ATOM_SIGMA, ATOM_OPACITY, *colour);
        if let Err(e) = scene.push(atom, 1.0e-6) {
            panic!("a valid atom was refused: {e:?}");
        }
    }
    scene
}

fn floor() -> SurfelScene<f64> {
    let mut scene = SurfelScene::new();
    for iz in FLOOR_Z.0..=FLOOR_Z.1 {
        for ix in -FLOOR_HALF_X..=FLOOR_HALF_X {
            let checker = if (ix + iz).rem_euclid(2) == 0 { 0.55 } else { 0.45 };
            let centre = Vec3::new(ix as f64 * FLOOR_SPACING, 0.0, iz as f64 * FLOOR_SPACING);
            match Surfel::new(
                centre,
                Vec3::new(0.0, -1.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec2::new(FLOOR_SPACING, FLOOR_SPACING),
                FLOOR_OPACITY,
                Rgb::splat(checker),
            ) {
                Ok(s) => scene.push(s),
                Err(e) => panic!("a valid surfel was refused: {e:?}"),
            }
        }
    }
    scene
}

fn camera() -> Camera<f64> {
    match Camera::from_spec(&spec(), settings().near_plane) {
        Ok(c) => c,
        Err(e) => panic!("the camera failed to build: {e:?}"),
    }
}

fn atom_request(scene: &Scene<f64>, method: RenderMethod) -> RenderRequest {
    RenderRequest {
        precision: Precision::F64,
        kernel: KernelFamily::Gaussian,
        method,
        settings: settings(),
        camera: spec(),
        scene: scene.clone(),
        collect: Collect::PixelsAndCoverage,
        shapes: None,
    }
}

fn limited_atom_pass(scene: &Scene<f64>, depth: &DepthImage<f64>) -> (FloatImage<f64>, FloatImage<f64>) {
    let outcome = match render_depth_limited(&atom_request(scene, RenderMethod::raycast_contract()), depth) {
        Ok(o) => o,
        Err(e) => panic!("the depth-limited atom pass failed: {e:?}"),
    };
    match (outcome.premultiplied, outcome.coverage) {
        (Some(c), Some(a)) => (c, a),
        (None, _) => panic!("the depth-limited atom pass handed out no premultiplied colour"),
        (Some(_), None) => panic!("the depth-limited atom pass handed out no coverage"),
    }
}

fn floor_layer(scene: &SurfelScene<f64>) -> SurfelLayer<f64> {
    let l = light();
    match render_surfel_layer(scene, &camera(), Some(&l)) {
        Ok(layer) => layer,
        Err(e) => panic!("the floor layer failed: {e:?}"),
    }
}

fn atom_pass(scene: &Scene<f64>) -> (FloatImage<f64>, FloatImage<f64>) {
    let request = RenderRequest {
        precision: Precision::F64,
        kernel: KernelFamily::Gaussian,
        method: RenderMethod::raycast_contract(),
        settings: settings(),
        camera: spec(),
        scene: scene.clone(),
        collect: Collect::PixelsAndCoverage,
        shapes: None,
    };
    let outcome = match render(&request) {
        Ok(o) => o,
        Err(e) => panic!("the atom pass failed: {e:?}"),
    };
    match (outcome.premultiplied, outcome.coverage) {
        (Some(c), Some(a)) => (c, a),
        (None, _) => panic!("the atom pass handed out no premultiplied colour"),
        (Some(_), None) => panic!("the atom pass handed out no coverage"),
    }
}

fn floor_pass(scene: &SurfelScene<f64>) -> (FloatImage<f64>, FloatImage<f64>) {
    let l = light();
    match render_surfels(scene, &camera(), Some(&l)) {
        Ok(pair) => pair,
        Err(e) => panic!("the floor pass failed: {e:?}"),
    }
}

enum Sample {
    Atom(Hit<f64>),
    Floor(SurfelHit<f64>),
}

fn interleaved(atoms: Option<&Scene<f64>>, floor: Option<&SurfelScene<f64>>) -> (FloatImage<f64>, FloatImage<f64>) {
    let s = settings();
    let c = camera();
    let l = light();
    let mut ray_atoms: Vec<RayAtom<f64, Gaussian>> = Vec::new();
    if let Some(scene) = atoms {
        let prepared = match PreparedScene::<f64, Gaussian>::build_uniform(scene, &KernelFamily::Gaussian.shape(), &s) {
            Ok(p) => p,
            Err(e) => panic!("the atom scene failed to prepare: {e:?}"),
        };
        prepare_atoms(&prepared, &c, &s, &mut ray_atoms);
    }
    let local: Vec<Surfel<f64>> = match floor {
        Some(f) => f.surfels().iter().map(|x| to_camera_frame(&c, x)).collect(),
        None => Vec::new(),
    };
    let to_world = c.rotation.transpose();
    let mut colour = FloatImage::new(WIDTH, HEIGHT);
    let mut coverage = FloatImage::new(WIDTH, HEIGHT);
    let mut hits: Vec<Hit<f64>> = Vec::new();
    let mut samples: Vec<(f64, u32, Sample)> = Vec::new();
    let mut probe = NoProbe;
    for py in 0..HEIGHT {
        for px in 0..WIDTH {
            let camera_ray = c.ray(px, py);
            let world_ray = to_world_ray(&to_world, c.eye, &camera_ray);
            if let Err(e) = collect_hits::<f64, Gaussian, NoProbe>(&ray_atoms, None, &world_ray, &s, MAX_HITS, &mut probe, &mut hits) {
                panic!("the atom hit list overflowed: {e:?}");
            }
            samples.clear();
            for h in hits.iter() {
                samples.push((h.key, h.index, Sample::Atom(*h)));
            }
            for (i, surfel) in local.iter().enumerate() {
                if let Some(h) = intersect_surfel(surfel, &camera_ray, Some(&l), i as u32) {
                    samples.push((h.t, i as u32, Sample::Floor(h)));
                }
            }
            samples.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            let mut acc: OverAccum<f64> = <Over as BlendMode<f64>>::init();
            for (_, _, sample) in samples.iter() {
                match sample {
                    Sample::Atom(h) => {
                        match <Over as BlendMode<f64>>::step(&mut acc, h.weight, h.colour, h.opacity, &s.blend) {
                            Step::Continue => {}
                            Step::Stop => break,
                        }
                    }
                    Sample::Floor(h) => {
                        let surfel = &local[h.index as usize];
                        let alpha = surfel.opacity * h.weight;
                        let radiance = surfel.colour * h.shade + Rgb::splat(h.specular) + surfel.material.emission;
                        acc.rgb = acc.rgb + radiance * (acc.transmittance * alpha);
                        acc.transmittance = acc.transmittance * (1.0 - alpha);
                    }
                }
            }
            colour.set(px, py, acc.rgb);
            coverage.set(px, py, Rgb::splat(1.0 - acc.transmittance));
        }
    }
    (colour, coverage)
}

fn over(front: &(FloatImage<f64>, FloatImage<f64>), back: &(FloatImage<f64>, FloatImage<f64>)) -> FloatImage<f64> {
    match front.0.composite_over(&front.1, &back.0) {
        Ok(img) => img,
        Err(e) => panic!("composite failed: {e:?}"),
    }
}

fn channel_gap(a: Rgb<f64>, b: Rgb<f64>) -> f64 {
    (a.x - b.x).abs().max((a.y - b.y).abs()).max((a.z - b.z).abs())
}

fn largest_gap(a: &FloatImage<f64>, b: &FloatImage<f64>) -> f64 {
    let mut worst = 0.0_f64;
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            worst = worst.max(channel_gap(a.get(x, y), b.get(x, y)));
        }
    }
    worst
}

#[derive(Debug, Default, Clone, Copy)]
struct Tally {
    visible: usize,
    gross: usize,
    worst: f64,
    overlap: usize,
    overlap_visible: usize,
    single_visible: usize,
    single_worst: f64,
}

fn tally(
    composited: &FloatImage<f64>,
    truth: &FloatImage<f64>,
    atom_cov: &FloatImage<f64>,
    floor_cov: &FloatImage<f64>,
) -> Tally {
    let mut t = Tally::default();
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let gap = channel_gap(composited.get(x, y), truth.get(x, y));
            let both = atom_cov.get(x, y).x > SEEN && floor_cov.get(x, y).x > SEEN;
            t.worst = t.worst.max(gap);
            if gap > VISIBLE {
                t.visible += 1;
            }
            if gap > GROSS {
                t.gross += 1;
            }
            if both {
                t.overlap += 1;
                if gap > VISIBLE {
                    t.overlap_visible += 1;
                }
            } else {
                t.single_worst = t.single_worst.max(gap);
                if gap > VISIBLE {
                    t.single_visible += 1;
                }
            }
        }
    }
    t
}

struct Measured {
    atoms_over_floor: Tally,
    floor_over_atoms: Tally,
    atom_pixels: usize,
}

fn measure(lift: f64) -> Measured {
    measure_scene(&atoms(lift))
}

fn measure_scene(a: &Scene<f64>) -> Measured {
    let f = floor();
    let atom = atom_pass(a);
    let flr = floor_pass(&f);
    let (truth, _) = interleaved(Some(a), Some(&f));
    let aof = over(&atom, &flr);
    let foa = over(&flr, &atom);
    let mut atom_pixels = 0usize;
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            if atom.1.get(x, y).x > SEEN {
                atom_pixels += 1;
            }
        }
    }
    Measured {
        atoms_over_floor: tally(&aof, &truth, &atom.1, &flr.1),
        floor_over_atoms: tally(&foa, &truth, &atom.1, &flr.1),
        atom_pixels,
    }
}

fn report(name: &str, m: &Measured) {
    let total = (WIDTH * HEIGHT) as f64;
    for (order, t) in [("atoms over floor", m.atoms_over_floor), ("floor over atoms", m.floor_over_atoms)] {
        println!(
            "{name:<14} {order}: disagreeing {} px ({:.3}% of frame), gross {} px, worst {:.6}; overlap {} px of which {} disagree; single-kind disagreeing {} px, single-kind worst {:.3e}; atom footprint {} px",
            t.visible,
            100.0 * t.visible as f64 / total,
            t.gross,
            t.worst,
            t.overlap,
            t.overlap_visible,
            t.single_visible,
            t.single_worst,
            m.atom_pixels,
        );
    }
}

#[test]
fn the_interleaved_instrument_reproduces_each_pass_alone() {
    let a = atoms(0.0);
    let f = floor();
    let (atoms_alone, atoms_alone_cov) = interleaved(Some(&a), None);
    let (floor_alone, floor_alone_cov) = interleaved(None, Some(&f));
    let atom = atom_pass(&a);
    let flr = floor_pass(&f);
    let atom_gap = largest_gap(&atoms_alone, &atom.0).max(largest_gap(&atoms_alone_cov, &atom.1));
    let floor_gap = largest_gap(&floor_alone, &flr.0).max(largest_gap(&floor_alone_cov, &flr.1));
    println!("interleaved vs shipped pass: atoms {atom_gap:.3e}, floor {floor_gap:.3e}");
    assert!(atom.1.max_abs().0 > 0.5, "the atom pass must put something on screen");
    assert!(flr.1.max_abs().0 > 0.9, "the floor pass must be opaque somewhere");
    assert!(atom_gap < EXACT, "the interleaved arm must reproduce the shipped atom pass, gap {atom_gap}");
    assert!(floor_gap < EXACT, "the interleaved arm must reproduce the shipped floor pass, gap {floor_gap}");
}

#[test]
fn compositing_is_exact_wherever_only_one_kind_is_seen() {
    let m = measure(0.0);
    report("interpenetrate", &m);
    for t in [m.atoms_over_floor, m.floor_over_atoms] {
        assert!(t.overlap > 0, "the scene must put atoms over the floor, or this measures nothing");
        assert!(
            t.single_worst < 1.0e-9,
            "a pixel only one pass reaches must composite exactly, worst {}",
            t.single_worst
        );
    }
}

#[test]
fn lifted_clear_of_the_floor_atoms_over_floor_agrees() {
    let m = measure(LIFT_IN_SIGMAS);
    report("lifted", &m);
    assert!(m.atoms_over_floor.overlap > 0, "the control must still overlap the floor on screen");
    assert_eq!(
        m.atoms_over_floor.visible, 0,
        "with no interpenetration, atoms over floor must agree with depth interleaving"
    );
}

#[test]
fn an_interpenetrating_scene_defeats_both_composite_orders() {
    let m = measure(0.0);
    report("interpenetrate", &m);
    assert!(m.atoms_over_floor.visible > 0, "atoms over floor must show the buried halves");
    assert!(m.floor_over_atoms.visible > 0, "floor over atoms must hide the risen halves");
}

#[test]
fn one_centred_atom_needs_clearance_before_atoms_over_floor_is_exact() {
    let mut first_clear = None;
    for h in SWEEP_IN_SIGMAS {
        let m = measure_scene(&atoms_at(&[h]));
        let t = m.atoms_over_floor;
        println!(
            "centre {h:>4} sigma above the floor: atoms over floor disagreeing {} px of {} overlap, gross {}, worst {:.6}; floor over atoms disagreeing {}",
            t.visible, t.overlap, t.gross, t.worst, m.floor_over_atoms.visible
        );
        assert!(t.overlap > 0, "the swept atom must overlap the floor on screen at {h} sigma");
        if t.visible == 0 && first_clear.is_none() {
            first_clear = Some(h);
        }
    }
    println!("first clearance at which atoms over floor agrees: {first_clear:?}");
    assert!(first_clear.is_some(), "some clearance inside the sweep must composite exactly");
    assert!(first_clear > Some(0.0), "an atom centred on the floor cannot composite exactly");
}

fn measure_depth_limited(a: &Scene<f64>) -> Tally {
    let f = floor();
    let layer = floor_layer(&f);
    let atom = limited_atom_pass(a, &layer.depth);
    let unlimited = atom_pass(a);
    let (truth, _) = interleaved(Some(a), Some(&f));
    let composited = over(&atom, &(layer.colour.clone(), layer.coverage.clone()));
    tally(&composited, &truth, &unlimited.1, &layer.coverage)
}

#[test]
fn the_floor_layer_is_the_floor_pass_plus_a_depth_that_is_finite_only_where_it_is_opaque() {
    let f = floor();
    let layer = floor_layer(&f);
    let flr = floor_pass(&f);
    assert!(largest_gap(&layer.colour, &flr.0) == 0.0, "the layer's colour must be the floor pass's");
    assert!(largest_gap(&layer.coverage, &flr.1) == 0.0, "the layer's coverage must be the floor pass's");
    let mut finite = 0usize;
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let transmittance = 1.0 - layer.coverage.get(x, y).x;
            let d = layer.depth.get(x, y);
            assert_eq!(
                d.is_finite(),
                transmittance <= OPAQUE_TRANSMITTANCE,
                "depth at ({x}, {y}) is {d} under a floor transmittance of {transmittance}"
            );
            if d.is_finite() {
                finite += 1;
                assert!(d > 0.0, "an opaque floor pixel must carry a depth in front of the eye, got {d}");
            }
        }
    }
    println!("opaque floor pixels carrying a depth: {finite} of {}", WIDTH * HEIGHT);
    assert_eq!(finite, layer.depth.finite_count());
    assert!(finite > (WIDTH * HEIGHT / 4) as usize, "the floor must be opaque over much of the frame");
}

#[test]
fn a_depth_limited_atom_pass_composites_a_resting_scene_exactly() {
    let t = measure_depth_limited(&atoms(0.0));
    println!(
        "depth-limited atoms over floor: disagreeing {} px, gross {}, worst {:.3e}; overlap {} px of which {} disagree",
        t.visible, t.gross, t.worst, t.overlap, t.overlap_visible
    );
    assert!(t.overlap > 0, "the scene must put atoms over the floor, or this measures nothing");
    assert_eq!(t.visible, 0, "with the floor's depth carried, atoms resting on it must composite exactly");
}

#[test]
fn the_depth_limit_holds_at_every_clearance_of_the_sweep() {
    for h in SWEEP_IN_SIGMAS {
        let t = measure_depth_limited(&atoms_at(&[h]));
        println!(
            "centre {h:>4} sigma above the floor, depth-limited: disagreeing {} px of {} overlap, worst {:.3e}",
            t.visible, t.overlap, t.worst
        );
        assert!(t.overlap > 0, "the swept atom must overlap the floor on screen at {h} sigma");
        assert_eq!(t.visible, 0, "the depth-limited composite must be exact at {h} sigma");
    }
}

#[test]
fn a_far_depth_limit_changes_nothing() {
    let a = atoms(0.0);
    let plain = atom_pass(&a);
    let far = limited_atom_pass(&a, &DepthImage::far(WIDTH, HEIGHT));
    assert!(largest_gap(&plain.0, &far.0) == 0.0, "an infinite limit must leave the colour bit-identical");
    assert!(largest_gap(&plain.1, &far.1) == 0.0, "an infinite limit must leave the coverage bit-identical");
}

#[test]
fn the_warm_loop_honours_the_depth_limit_as_dispatch_does() {
    let a = atoms(0.0);
    let layer = floor_layer(&floor());
    let dispatched = limited_atom_pass(&a, &layer.depth);
    let mut renderer = match Renderer::<f64, Gaussian>::new(
        &RenderMethod::raycast_contract(),
        &settings(),
        &KernelFamily::Gaussian.shape(),
        WIDTH,
        HEIGHT,
        &a,
    ) {
        Ok(r) => r,
        Err(e) => panic!("the renderer failed to build: {e:?}"),
    };
    if let Err(e) = renderer.limit_depth(Some(&layer.depth)) {
        panic!("the ray caster refused a depth limit: {e:?}");
    }
    if let Err(e) = renderer.render_frame(&camera()) {
        panic!("the frame failed: {e:?}");
    }
    let (colour, coverage) = renderer.resolve_premultiplied();
    assert!(largest_gap(&colour, &dispatched.0) == 0.0, "the warm loop must match dispatch's colour");
    assert!(largest_gap(&coverage, &dispatched.1) == 0.0, "the warm loop must match dispatch's coverage");
    let unlimited = atom_pass(&a);
    assert!(largest_gap(&coverage, &unlimited.1) > VISIBLE, "the limit must remove the buried halves");
}

#[test]
fn a_depth_limit_of_the_wrong_size_is_refused() {
    let a = atoms(0.0);
    let wrong = DepthImage::far(WIDTH + 1, HEIGHT);
    match render_depth_limited(&atom_request(&a, RenderMethod::raycast_contract()), &wrong) {
        Err(LightsError::SizeMismatch { .. }) => {}
        Err(e) => panic!("expected a size mismatch, got {e:?}"),
        Ok(_) => panic!("a depth limit of the wrong size was accepted"),
    }
}

#[cfg(feature = "method-splat")]
#[test]
fn the_splat_raster_refuses_a_depth_limit_rather_than_ignoring_it() {
    let a = atoms(0.0);
    let layer = floor_layer(&floor());
    let mut request = atom_request(&a, RenderMethod::splat_contract());
    request.settings = SharedSettings {
        threads: 1,
        ..SharedSettings::contract()
    };
    if let Err(e) = render(&request) {
        panic!("the splat raster must render this request unlimited, or the refusal below proves nothing: {e:?}");
    }
    match render_depth_limited(&request, &layer.depth) {
        Err(LightsError::DepthLimitUnsupported { .. }) => {}
        Err(e) => panic!("expected the splat raster to refuse the depth limit, got {e:?}"),
        Ok(_) => panic!("the splat raster accepted a depth limit it cannot honour"),
    }
}
