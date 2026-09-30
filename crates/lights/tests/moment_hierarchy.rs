use std::time::Instant;

use lights::camera::{Camera, CameraSpec};
use lights::dispatch::{Collect, Precision, RenderOutcome, RenderRequest, render};
use lights::kernel::KernelFamily;
use lights::method::RenderMethod;
use lights::numerics::quaternion::Quat;
use lights::numerics::vector::{Rgb, Vec3};
use lights::scene::hierarchy::{MomentTree, atom_mass, moment_merge};
use lights::scene::{Atom, Scene};
use lights::settings::{MIN_SCALE, PixelConvention, SharedSettings};

const WIDTH: u32 = 192;
const HEIGHT: u32 = 144;
const THRESHOLDS: [f64; 8] = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 16.0];
const REPS: usize = 5;
const CODE_VALUE: f64 = 1.0 / 255.0;

struct Stream(u64);

impl Stream {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next()
    }

    fn log_range(&mut self, lo: f64, hi: f64) -> f64 {
        (lo.ln() + (hi.ln() - lo.ln()) * self.next()).exp()
    }

    fn rotation(&mut self) -> Quat<f64> {
        let q = Quat::new_wxyz(
            self.range(-1.0, 1.0),
            self.range(-1.0, 1.0),
            self.range(-1.0, 1.0),
            self.range(-1.0, 1.0),
        );
        q.normalize().unwrap_or(Quat::identity())
    }
}

fn covariance_entries(atom: &Atom<f64>) -> [[f64; 3]; 3] {
    let r = atom.rotation.to_mat3();
    let s = [atom.scale.x, atom.scale.y, atom.scale.z];
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            for (k, sk) in s.iter().enumerate() {
                *cell += r.row(i)[k] * sk * sk * r.row(j)[k];
            }
        }
    }
    out
}

fn merged_entries(atom: &Atom<f64>) -> [[f64; 3]; 3] {
    let c = atom.covariance();
    [[c.xx, c.xy, c.xz], [c.xy, c.yy, c.yz], [c.xz, c.yz, c.zz]]
}

#[test]
fn two_equal_atoms_merge_to_the_midpoint_with_the_separation_added_to_the_variance() {
    let sigma: f64 = 0.3;
    let d: f64 = 2.0;
    let a = Atom::isotropic(Vec3::new(-d / 2.0, 1.0, 5.0), sigma, 0.5, Rgb::new(1.0, 0.0, 0.0));
    let b = Atom::isotropic(Vec3::new(d / 2.0, 1.0, 5.0), sigma, 0.5, Rgb::new(0.0, 0.0, 1.0));
    let Some(m) = moment_merge::<f64>(&[a, b]) else {
        panic!("two finite atoms must merge");
    };
    let c = m.covariance();
    let want_xx = sigma * sigma + d * d / 4.0;
    assert!((c.xx - want_xx).abs() < 1e-12, "xx {} want {want_xx}", c.xx);
    assert!((c.yy - sigma * sigma).abs() < 1e-12, "yy {}", c.yy);
    assert!((c.zz - sigma * sigma).abs() < 1e-12, "zz {}", c.zz);
    assert!(c.xy.abs() < 1e-12 && c.xz.abs() < 1e-12 && c.yz.abs() < 1e-12);
    assert!((m.mean - Vec3::new(0.0, 1.0, 5.0)).length() < 1e-12);
    assert!((m.colour - Rgb::new(0.5, 0.0, 0.5)).length() < 1e-12);
    let mass = atom_mass(&a) + atom_mass(&b);
    assert!((atom_mass(&m) - mass).abs() < 1e-12 * mass);
}

#[test]
fn a_random_mixture_merges_to_its_own_first_and_second_moments() {
    let mut rng = Stream(0x9e37_79b9_7f4a_7c15);
    let mut worst_cov = 0.0f64;
    let mut worst_mass = 0.0f64;
    let mut worst_single = 0.0f64;
    for case in 0..500usize {
        let n = 1 + case % 9;
        let atoms: Vec<Atom<f64>> = (0..n)
            .map(|_| Atom {
                mean: Vec3::new(rng.range(-2.0, 2.0), rng.range(-2.0, 2.0), rng.range(3.0, 9.0)),
                rotation: rng.rotation(),
                scale: Vec3::new(
                    rng.log_range(0.01, 1.0),
                    rng.log_range(0.01, 1.0),
                    rng.log_range(0.01, 1.0),
                ),
                opacity: rng.range(0.05, 1.0),
                colour: Rgb::new(rng.next(), rng.next(), rng.next()),
            })
            .collect();
        let Some(m) = moment_merge(&atoms) else {
            panic!("case {case}: merge refused a finite mixture");
        };
        let masses: Vec<f64> = atoms
            .iter()
            .map(|a| a.opacity * a.scale.x * a.scale.y * a.scale.z)
            .collect();
        let total: f64 = masses.iter().sum();
        let mut mean = [0.0; 3];
        for (a, w) in atoms.iter().zip(&masses) {
            for (k, slot) in mean.iter_mut().enumerate() {
                *slot += a.mean[k] * w / total;
            }
        }
        let mut reference = [[0.0; 3]; 3];
        for (a, w) in atoms.iter().zip(&masses) {
            let c = covariance_entries(a);
            for i in 0..3 {
                for j in 0..3 {
                    let di = a.mean[i] - mean[i];
                    let dj = a.mean[j] - mean[j];
                    reference[i][j] += w / total * (c[i][j] + di * dj);
                }
            }
        }
        let got = merged_entries(&m);
        let scale = reference.iter().flatten().fold(0.0f64, |acc, v| acc.max(v.abs()));
        for i in 0..3 {
            for j in 0..3 {
                worst_cov = worst_cov.max((got[i][j] - reference[i][j]).abs() / scale);
            }
            assert!((m.mean[i] - mean[i]).abs() < 1e-12, "case {case}: mean[{i}]");
        }
        worst_mass = worst_mass.max((atom_mass(&m) - total).abs() / total);
        if n == 1 {
            let own = covariance_entries(&atoms[0]);
            for i in 0..3 {
                for j in 0..3 {
                    worst_single = worst_single.max((got[i][j] - own[i][j]).abs() / scale);
                }
            }
        }
    }
    println!(
        "MERGE f64, 500 mixtures of 1-9 atoms: covariance {worst_cov:.3e} relative, \
         mass {worst_mass:.3e} relative, single-atom round trip {worst_single:.3e}"
    );
    assert!(worst_cov < 1e-12, "covariance {worst_cov:e}");
    assert!(worst_mass < 1e-12, "mass {worst_mass:e}");
    assert!(worst_single < 1e-12, "single {worst_single:e}");
}

fn spec() -> CameraSpec {
    CameraSpec::look_at([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0], 45.0, WIDTH, HEIGHT)
        .with_convention(PixelConvention::CornerSample)
}

fn settings() -> SharedSettings<f64> {
    let mut out = SharedSettings::<f64>::contract();
    out.threads = 1;
    out.background = Rgb::new(0.0, 0.0, 0.0);
    out
}

fn camera() -> Camera<f64> {
    match Camera::from_spec(&spec(), settings().near_plane) {
        Ok(c) => c,
        Err(e) => panic!("camera: {e:?}"),
    }
}

fn cloud() -> Vec<Atom<f64>> {
    let mut rng = Stream(0x2545_f491_4f6c_dd1d);
    (0..1200)
        .map(|_| Atom {
            mean: Vec3::new(rng.range(-3.0, 3.0), rng.range(-2.2, 2.2), rng.range(8.0, 12.0)),
            rotation: rng.rotation(),
            scale: Vec3::new(
                rng.log_range(0.03, 0.1),
                rng.log_range(0.03, 0.1),
                rng.log_range(0.03, 0.1),
            ),
            opacity: rng.range(0.4, 0.9),
            colour: Rgb::new(rng.next(), rng.next(), rng.next()),
        })
        .collect()
}

fn flat_cloud() -> Vec<Atom<f64>> {
    let mut rng = Stream(0x2545_f491_4f6c_dd1d);
    let axis = Vec3::new(0.0, 0.0, 1.0);
    (0..1200)
        .map(|_| Atom {
            mean: Vec3::new(rng.range(-3.0, 3.0), rng.range(-2.2, 2.2), 10.0),
            rotation: Quat::from_axis_angle(axis, rng.range(0.0, 3.2)).unwrap_or(Quat::identity()),
            scale: Vec3::new(rng.log_range(0.03, 0.1), rng.log_range(0.03, 0.1), 0.03),
            opacity: rng.range(0.4, 0.9),
            colour: Rgb::new(rng.next(), rng.next(), rng.next()),
        })
        .collect()
}

fn lateral_pairs() -> Vec<Atom<f64>> {
    let mut out = Vec::new();
    for gy in 0..7 {
        for gx in 0..9 {
            let cx = -2.8 + 0.7 * gx as f64;
            let cy = -2.1 + 0.7 * gy as f64;
            for side in [-1.0, 1.0] {
                out.push(Atom::isotropic(
                    Vec3::new(cx + side * 0.15, cy, 10.0),
                    0.05,
                    0.9,
                    Rgb::new(0.9, 0.9, 0.9),
                ));
            }
        }
    }
    out
}

fn depth_pairs() -> Vec<Atom<f64>> {
    let mut out = Vec::new();
    for gy in 0..7 {
        for gx in 0..9 {
            let cx = -2.8 + 0.7 * gx as f64;
            let cy = -2.1 + 0.7 * gy as f64;
            out.push(Atom::isotropic(Vec3::new(cx, cy, 9.9), 0.05, 0.9, Rgb::new(1.0, 0.0, 0.0)));
            out.push(Atom::isotropic(Vec3::new(cx, cy, 10.1), 0.05, 0.9, Rgb::new(0.0, 0.0, 1.0)));
        }
    }
    out
}

fn scene_of(atoms: &[Atom<f64>]) -> Scene<f64> {
    let mut scene = Scene::new();
    for atom in atoms {
        if let Err(e) = scene.push(*atom, MIN_SCALE) {
            panic!("scene rejected an atom: {e:?}");
        }
    }
    scene
}

fn run(method: &RenderMethod, atoms: &[Atom<f64>], collect: Collect) -> RenderOutcome {
    let request = RenderRequest {
        precision: Precision::F64,
        kernel: KernelFamily::Gaussian,
        method: method.clone(),
        settings: settings(),
        camera: spec(),
        scene: scene_of(atoms),
        collect,
        shapes: None,
    };
    match render(&request) {
        Ok(o) => o,
        Err(e) => panic!("render failed: {e:?}"),
    }
}

fn timed(method: &RenderMethod, atoms: &[Atom<f64>]) -> (f64, f64) {
    let mut ms: Vec<f64> = (0..REPS)
        .map(|_| {
            let t = Instant::now();
            let _ = run(method, atoms, Collect::Pixels);
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    ms.sort_by(f64::total_cmp);
    (ms[REPS / 2], ms[REPS - 1] / ms[0])
}

struct Error {
    max: f64,
    rms: f64,
    coverage_bias: f64,
}

fn error_of(full: &RenderOutcome, cut: &RenderOutcome) -> Error {
    let mut max = 0.0f64;
    let mut sq = 0.0f64;
    for (a, b) in full.image.pixels.iter().zip(cut.image.pixels.iter()) {
        let d = *b - *a;
        max = max.max(d.x.abs()).max(d.y.abs()).max(d.z.abs());
        sq += d.x * d.x + d.y * d.y + d.z * d.z;
    }
    let rms = (sq / (3 * full.image.pixels.len()) as f64).sqrt();
    let sum = |o: &RenderOutcome| -> f64 {
        match &o.coverage {
            Some(c) => c.pixels.iter().map(|p| p.x).sum(),
            None => panic!("coverage was requested and not returned"),
        }
    };
    let (f, c) = (sum(full), sum(cut));
    Error {
        max,
        rms,
        coverage_bias: (c - f) / f,
    }
}

#[test]
fn every_cut_conserves_mass_and_a_zero_threshold_cut_is_the_scene_itself() {
    let camera = camera();
    for atoms in [cloud(), flat_cloud(), lateral_pairs(), depth_pairs()] {
        let Some(tree) = MomentTree::build(&atoms) else {
            panic!("tree refused");
        };
        let total: f64 = atoms.iter().map(atom_mass).sum();
        let root = tree.root_summary();
        assert!((atom_mass(&root) - total).abs() < 1e-9 * total, "root mass");
        let mut last = usize::MAX;
        for threshold in THRESHOLDS {
            let (cut, stats) = tree.cut(&camera, threshold);
            let mass: f64 = cut.iter().map(atom_mass).sum();
            assert!((mass - total).abs() < 1e-9 * total, "tau {threshold}: mass {mass} of {total}");
            assert_eq!(stats.leaves + stats.summarised_atoms, atoms.len(), "tau {threshold}");
            assert_eq!(cut.len(), stats.leaves + stats.summaries, "tau {threshold}");
            assert!(cut.len() <= last, "tau {threshold}: the cut grew");
            last = cut.len();
            if threshold == 0.0 {
                assert_eq!(cut, tree.atoms().to_vec(), "tau 0 must emit every atom in tree order");
            }
        }
        assert!(last < atoms.len(), "the largest threshold summarised nothing");
    }
}

#[test]
fn a_moment_cut_trades_image_error_for_kernel_evaluations_and_the_bimodal_scenes_pay_for_it() {
    let camera = camera();
    let scenes: [(&str, Vec<Atom<f64>>); 4] = [
        ("cloud", cloud()),
        ("flat-cloud", flat_cloud()),
        ("lateral-pairs", lateral_pairs()),
        ("depth-pairs", depth_pairs()),
    ];
    let methods = RenderMethod::implemented();
    assert!(!methods.is_empty(), "no render method is built");
    let mut covered = 0usize;
    let mut bimodal_worst = [0.0f64; 2];
    for method in &methods {
        covered += 1;
        for (label, atoms) in &scenes {
            let Some(tree) = MomentTree::build(atoms) else {
                panic!("{label}: tree refused");
            };
            let total_mass: f64 = atoms.iter().map(atom_mass).sum();
            let full = run(method, tree.atoms(), Collect::PixelsAndCoverage);
            let full_counts = run(method, tree.atoms(), Collect::PixelsAndCounters).counters;
            let (full_ms, full_spread) = timed(method, tree.atoms());
            println!(
                "CURVE {} {label}: {} atoms, {} nodes, full {:.2} ms (spread {:.2}x), {} kernel evaluations",
                method.id().name(),
                atoms.len(),
                tree.node_count(),
                full_ms,
                full_spread,
                full_counts.kernel_evaluations
            );
            let mut last_count = usize::MAX;
            for threshold in THRESHOLDS {
                let t = Instant::now();
                let (cut, stats) = tree.cut(&camera, threshold);
                let cut_ms = t.elapsed().as_secs_f64() * 1e3;
                let cut_mass: f64 = cut.iter().map(atom_mass).sum();
                assert!(
                    (cut_mass - total_mass).abs() < 1e-9 * total_mass,
                    "{label} tau {threshold}: the cut lost mass"
                );
                assert!(cut.len() <= last_count, "{label}: cut grew as the threshold rose");
                last_count = cut.len();
                let image = run(method, &cut, Collect::PixelsAndCoverage);
                let counts = run(method, &cut, Collect::PixelsAndCounters).counters;
                let (ms, spread) = timed(method, &cut);
                let err = error_of(&full, &image);
                if threshold == 0.0 {
                    assert_eq!(stats.summaries, 0, "{label}: tau 0 summarised a node");
                    assert_eq!(err.max, 0.0, "{label}: tau 0 is not the full render");
                }
                match *label {
                    "lateral-pairs" => bimodal_worst[0] = bimodal_worst[0].max(err.max),
                    "depth-pairs" => bimodal_worst[1] = bimodal_worst[1].max(err.max),
                    other => assert!(other.ends_with("cloud")),
                }
                println!(
                    "  tau {threshold:>4.1} px: {:>5} atoms ({} summaries of {} atoms), \
                     evals {:.3}x, time {:.2} ms {:.3}x (spread {:.2}x, cut {:.2} ms), \
                     max {:.4} ({:.1} code values) rms {:.5} coverage {:+.4} {}",
                    cut.len(),
                    stats.summaries,
                    stats.summarised_atoms,
                    counts.kernel_evaluations as f64 / full_counts.kernel_evaluations as f64,
                    ms,
                    ms / full_ms,
                    spread,
                    cut_ms,
                    err.max,
                    err.max / CODE_VALUE,
                    err.rms,
                    err.coverage_bias,
                    if err.max <= CODE_VALUE { "INSIDE" } else { "outside" }
                );
            }
        }
    }
    assert_eq!(covered, methods.len());
    assert!(
        bimodal_worst[0] > 10.0 * CODE_VALUE && bimodal_worst[1] > 10.0 * CODE_VALUE,
        "the bimodal scenes must expose the third-moment failure: {bimodal_worst:?}"
    );
}
