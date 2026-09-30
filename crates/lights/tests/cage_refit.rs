#![cfg(feature = "fit")]
use std::time::Instant;

use lights::camera::Camera;
use lights::fit::atom2d::{Atom2d, camera, image_to_world, to_atom, world_to_image};
use lights::fit::coverage::{coverage_budget, coverage_image_to_colour, exact_coverage};
use lights::fit::greedy::{FitRequest, fit, fit_budget, render_or_background, residual_energy};
use lights::fit::target::{Side, Target};
use lights::image::FloatImage;
use lights::kernel::KernelFamily;
use lights::numerics::matrix::Mat3;
use lights::numerics::vector::{Rgb, Vec2, Vec3};
use lights::scene::Atom;
use lights::scene::cage::Cage;
use lights::settings::SharedSettings;

const TILE: u32 = 16;
const RENDER_LIMIT: u64 = 40_000_000;
const CELL_LIMIT: u64 = 64_000_000;
const QUADRATURE_DEPTH: u32 = 10;
const SUPERSAMPLE: u32 = 8;
const JACOBIAN_STEP: f64 = 1e-4;
const CAGE_LEVELS: [usize; 5] = [1, 2, 4, 8, 16];
const REPS: usize = 5;
const CLOUD_ATOMS: usize = 100_000;
const SAMPLER_AGREEMENT: f64 = 1e-9;

#[derive(Copy, Clone)]
enum Warp {
    Bend { radius: f64 },
    Twist { turn: f64, reach: f64 },
}

impl Warp {
    fn name(self) -> &'static str {
        match self {
            Warp::Bend { .. } => "bend",
            Warp::Twist { .. } => "twist",
        }
    }

    fn forward(self, p: Vec2<f64>) -> Vec2<f64> {
        match self {
            Warp::Bend { radius } => {
                let (s, c) = (p.x / radius).sin_cos();
                let r = radius - p.y;
                Vec2::new(r * s, radius - r * c)
            }
            Warp::Twist { turn, reach } => {
                let angle = turn * (-(p.x * p.x + p.y * p.y) / (2.0 * reach * reach)).exp();
                let (s, c) = angle.sin_cos();
                Vec2::new(c * p.x - s * p.y, s * p.x + c * p.y)
            }
        }
    }

    fn inverse(self, q: Vec2<f64>) -> Vec2<f64> {
        match self {
            Warp::Bend { radius } => {
                let theta = q.x.atan2(radius - q.y);
                let r = q.x.hypot(radius - q.y);
                Vec2::new(radius * theta, radius - r)
            }
            Warp::Twist { turn, reach } => {
                let angle = -turn * (-(q.x * q.x + q.y * q.y) / (2.0 * reach * reach)).exp();
                let (s, c) = angle.sin_cos();
                Vec2::new(c * q.x - s * q.y, s * q.x + c * q.y)
            }
        }
    }

    fn forward3(self, p: Vec3<f64>) -> Vec3<f64> {
        let q = self.forward(Vec2::new(p.x, p.y));
        Vec3::new(q.x, q.y, p.z)
    }

    fn jacobian(self, p: Vec2<f64>) -> Mat3<f64> {
        let h = JACOBIAN_STEP;
        let dx = (self.forward(Vec2::new(p.x + h, p.y)) - self.forward(Vec2::new(p.x - h, p.y))) * (0.5 / h);
        let dy = (self.forward(Vec2::new(p.x, p.y + h)) - self.forward(Vec2::new(p.x, p.y - h))) * (0.5 / h);
        Mat3 {
            m: [[dx.x, dy.x, 0.0], [dx.y, dy.y, 0.0], [0.0, 0.0, 1.0]],
        }
    }
}

fn warps_for(viewport: (u32, u32)) -> [Warp; 2] {
    let w = viewport.0.max(viewport.1) as f64;
    [
        Warp::Bend { radius: 1.5 * w },
        Warp::Twist { turn: 0.6, reach: 0.3 * w },
    ]
}

fn supersampled_truth(target: &Target<f64>, warp: Option<Warp>) -> FloatImage<f64> {
    let (w, h) = target.viewport;
    let mut coverage = FloatImage::<f64>::new(w, h);
    let n = SUPERSAMPLE as f64;
    for py in 0..h {
        for px in 0..w {
            let mut inside = 0u32;
            for sy in 0..SUPERSAMPLE {
                for sx in 0..SUPERSAMPLE {
                    let image = Vec2::new(
                        px as f64 + (sx as f64 + 0.5) / n,
                        py as f64 + (sy as f64 + 0.5) / n,
                    );
                    let source = match warp {
                        Some(warp) => world_to_image(warp.inverse(image_to_world(image, target.viewport)), target.viewport),
                        None => image,
                    };
                    if target.classify(source) == Side::Inside {
                        inside += 1;
                    }
                }
            }
            let c = inside as f64 / (n * n);
            coverage.pixels[(py * w + px) as usize] = Rgb::new(c, c, c);
        }
    }
    coverage_image_to_colour(target, &coverage)
}

fn exact_truth(target: &Target<f64>) -> FloatImage<f64> {
    let mut budget = coverage_budget(CELL_LIMIT);
    let (coverage, _) = exact_coverage(target, QUADRATURE_DEPTH, &mut budget).expect("exact coverage");
    coverage_image_to_colour(target, &coverage)
}

fn request(max_atoms: usize) -> FitRequest {
    let mut r = FitRequest::new(KernelFamily::Gaussian, SharedSettings::<f64>::prefiltered());
    r.max_atoms = max_atoms;
    r.tile = TILE;
    r
}

fn render(atoms: &[Atom2d], viewport: (u32, u32), request: &FitRequest) -> FloatImage<f64> {
    render_or_background(atoms, &camera(viewport.0, viewport.1), request.kernel, &request.settings, TILE)
        .expect("render")
}

fn sample_at(atoms: &[Atom2d], world: Vec2<f64>, viewport: (u32, u32), request: &FitRequest) -> Rgb<f64> {
    let full = Camera::<f64>::from_spec(&camera(viewport.0, viewport.1), 0.0).expect("camera");
    let at = world_to_image(world, viewport);
    let spec = camera(1, 1).with_intrinsics(
        [full.focal.x, full.focal.y],
        [full.principal.x - (at.x - 0.5), full.principal.y - (at.y - 0.5)],
    );
    let shift = Vec2::new(0.5 - viewport.0 as f64 * 0.5, 0.5 - viewport.1 as f64 * 0.5);
    let placed: Vec<Atom2d> = atoms
        .iter()
        .map(|a| Atom2d {
            centre: a.centre + shift,
            ..*a
        })
        .collect();
    render_or_background(&placed, &spec, request.kernel, &request.settings, TILE)
        .expect("point render")
        .pixels[0]
}

fn pulled_back(atoms: &[Atom2d], viewport: (u32, u32), warp: Option<Warp>, request: &FitRequest) -> FloatImage<f64> {
    let mut out = FloatImage::<f64>::new(viewport.0, viewport.1);
    for py in 0..viewport.1 {
        for px in 0..viewport.0 {
            let world = image_to_world(Vec2::new(px as f64 + 0.5, py as f64 + 0.5), viewport);
            let source = match warp {
                Some(warp) => warp.inverse(world),
                None => world,
            };
            out.pixels[(py * viewport.0 + px) as usize] = sample_at(atoms, source, viewport, request);
        }
    }
    out
}

fn to_2d(atom: &Atom<f64>, viewport: (u32, u32)) -> Atom2d {
    let s = atom.covariance().marginal_xy();
    let angle = 0.5 * (2.0 * s.xy).atan2(s.xx - s.yy);
    let mid = 0.5 * (s.xx + s.yy);
    let half = (0.25 * (s.xx - s.yy) * (s.xx - s.yy) + s.xy * s.xy).sqrt();
    Atom2d::from_sigma(
        world_to_image(Vec2::new(atom.mean.x, atom.mean.y), viewport),
        Vec2::new((mid + half).sqrt(), (mid - half).max(0.0).sqrt()),
        angle,
        atom.opacity,
        atom.colour,
    )
}

fn energy(truth: &FloatImage<f64>, image: &FloatImage<f64>) -> f64 {
    residual_energy(truth, image).expect("energy")
}

fn spread(samples: &mut [f64]) -> (f64, f64, f64) {
    samples.sort_by(f64::total_cmp);
    (samples[0], samples[samples.len() / 2], samples[samples.len() - 1])
}

fn jacobian_deform(atoms: &[Atom<f64>], warp: Warp) -> Vec<Atom<f64>> {
    atoms
        .iter()
        .map(|a| {
            let linear = warp.jacobian(Vec2::new(a.mean.x, a.mean.y));
            let moved = warp.forward3(a.mean);
            a.mapped_affine(&linear, moved - linear.mul_vec(a.mean)).expect("jacobian map")
        })
        .collect()
}

fn measure(target: Target<f64>, max_atoms: usize) {
    let viewport = target.viewport;
    let request = request(max_atoms);
    let truth0 = supersampled_truth(&target, None);
    let quadrature_gap = energy(&exact_truth(&target), &truth0);
    let mut budget = fit_budget(RENDER_LIMIT);
    let started = Instant::now();
    let fitted = fit(&target, &truth0, &request, &mut budget).expect("the original fit");
    let first_fit_s = started.elapsed().as_secs_f64();
    let atoms2d = fitted.atoms.clone();
    let atoms3d: Vec<Atom<f64>> = atoms2d.iter().enumerate().map(|(i, a)| to_atom(a, i, viewport)).collect();
    let round_trip: Vec<Atom2d> = atoms3d.iter().map(|a| to_2d(a, viewport)).collect();
    let r0 = render(&atoms2d, viewport, &request);
    let round_gap = energy(&r0, &render(&round_trip, viewport, &request));
    let sampler_gap = energy(&r0, &pulled_back(&atoms2d, viewport, None, &request));
    println!(
        "CAGEFIT {} viewport {}x{} atoms {} fit0 energy {:.6e} renders {} time {:.3}s | instrument: \
         supersample-vs-exact truth {:.3e}, 3D->2D round trip {:.3e}, point sampler {:.3e}",
        target.name,
        viewport.0,
        viewport.1,
        atoms2d.len(),
        energy(&truth0, &r0),
        fitted.renders,
        first_fit_s,
        quadrature_gap,
        round_gap,
        sampler_gap
    );
    assert!(round_gap <= SAMPLER_AGREEMENT, "the 3D->2D round trip changes the render by {round_gap:e}");
    assert!(sampler_gap <= SAMPLER_AGREEMENT, "the point sampler misses the grid render by {sampler_gap:e}");

    for warp in warps_for(viewport) {
        let truth1 = supersampled_truth(&target, Some(warp));
        let reference = pulled_back(&atoms2d, viewport, Some(warp), &request);
        let image_of = |deformed: &[Atom<f64>]| {
            let flat: Vec<Atom2d> = deformed.iter().map(|a| to_2d(a, viewport)).collect();
            render(&flat, viewport, &request)
        };
        let jacobian_image = image_of(&jacobian_deform(&atoms3d, warp));
        println!(
            "CAGEFIT {} {} none      truth {:.6e} pulled-back {:.6e}",
            target.name,
            warp.name(),
            energy(&truth1, &r0),
            energy(&reference, &r0)
        );
        let (jt, jr) = (energy(&truth1, &jacobian_image), energy(&reference, &jacobian_image));
        println!("CAGEFIT {} {} jacobian  truth {jt:.6e} pulled-back {jr:.6e}", target.name, warp.name());
        let mut by_level = Vec::new();
        for k in CAGE_LEVELS {
            let cage = Cage::around(&atoms3d, [k, k, 1], |p| warp.forward3(p)).expect("cage");
            let deformed = cage.deform(&atoms3d).expect("cage deform");
            let image = image_of(&deformed);
            let (ct, cr, cj) = (
                energy(&truth1, &image),
                energy(&reference, &image),
                energy(&jacobian_image, &image),
            );
            by_level.push(cj);
            let mut times: Vec<f64> = (0..REPS)
                .map(|_| {
                    let t = Instant::now();
                    let cage = Cage::around(&atoms3d, [k, k, 1], |p| warp.forward3(p)).expect("cage");
                    let d = cage.deform(&atoms3d).expect("deform");
                    let e = t.elapsed().as_secs_f64();
                    assert_eq!(d.len(), atoms3d.len());
                    e
                })
                .collect();
            let (lo, mid, hi) = spread(&mut times);
            println!(
                "CAGEFIT {} {} cage k={k:<2} truth {ct:.6e} pulled-back {cr:.6e} vs-jacobian {cj:.6e} build+deform {:.1}/{:.1}/{:.1} us (min/median/max of {REPS})",
                target.name,
                warp.name(),
                lo * 1e6,
                mid * 1e6,
                hi * 1e6
            );
        }
        let mut budget = fit_budget(RENDER_LIMIT);
        let started = Instant::now();
        let refit = fit(&target, &truth1, &request, &mut budget).expect("the re-fit");
        let refit_s = started.elapsed().as_secs_f64();
        let refit_image = render(&refit.atoms, viewport, &request);
        let mut render_times: Vec<f64> = (0..REPS)
            .map(|_| {
                let t = Instant::now();
                let _ = render(&atoms2d, viewport, &request);
                t.elapsed().as_secs_f64()
            })
            .collect();
        let (rlo, rmid, rhi) = spread(&mut render_times);
        println!(
            "CAGEFIT {} {} re-fit    truth {:.6e} pulled-back {:.6e} atoms {} renders {} time {refit_s:.3}s | one render {:.1}/{:.1}/{:.1} us",
            target.name,
            warp.name(),
            energy(&truth1, &refit_image),
            energy(&reference, &refit_image),
            refit.atoms.len(),
            refit.renders,
            rlo * 1e6,
            rmid * 1e6,
            rhi * 1e6
        );
        assert!(refit.renders > 0, "a re-fit that rendered nothing measured nothing");
        assert!(
            by_level[by_level.len() - 1] < by_level[0],
            "{} {}: the finest cage ({:e}) is no closer to the per-atom Jacobian deform than one cell ({:e})",
            target.name,
            warp.name(),
            by_level[by_level.len() - 1],
            by_level[0]
        );
    }
}

#[test]
fn a_cage_deform_against_a_re_fit_on_a_bent_and_a_twisted_target() {
    measure(Target::t4_curve(), 8);
    measure(Target::t6_glyph_like(), 16);
}

#[test]
fn a_cage_deform_scales_per_atom_on_a_large_cloud() {
    let warp = Warp::Twist { turn: 0.6, reach: 20.0 };
    let side = (CLOUD_ATOMS as f64).cbrt().ceil() as usize;
    let atoms: Vec<Atom<f64>> = (0..CLOUD_ATOMS)
        .map(|i| {
            let (x, y, z) = (i % side, (i / side) % side, i / (side * side));
            Atom::isotropic(
                Vec3::new(x as f64 - side as f64 * 0.5, y as f64 - side as f64 * 0.5, 64.0 + z as f64),
                0.4,
                0.8,
                Rgb::new(0.5, 0.5, 0.5),
            )
        })
        .collect();
    let mut times: Vec<f64> = (0..REPS)
        .map(|_| {
            let t = Instant::now();
            let cage = Cage::around(&atoms, [16, 16, 1], |p| warp.forward3(p)).expect("cage");
            let d = cage.deform(&atoms).expect("deform");
            let e = t.elapsed().as_secs_f64();
            assert_eq!(d.len(), atoms.len());
            e
        })
        .collect();
    let (lo, mid, hi) = spread(&mut times);
    println!(
        "CAGEFIT cloud {CLOUD_ATOMS} atoms k=16: build+deform {:.2}/{:.2}/{:.2} ms, {:.1} ns per atom at the median",
        lo * 1e3,
        mid * 1e3,
        hi * 1e3,
        mid * 1e9 / CLOUD_ATOMS as f64
    );
}
