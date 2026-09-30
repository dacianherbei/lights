#![cfg(feature = "fit")]
use lights::dispatch::Collect;
use lights::fit::atom2d::{Atom2d, ShapedAtom2d, camera};
use lights::fit::coverage::{coverage_budget, coverage_image_to_colour, exact_coverage};
use lights::fit::greedy::{
    FitRequest, MAX_BETA, MIN_BETA, fit, fit_budget, fit_shaped, nudge_beta, render_shaped_atoms,
};
use lights::fit::target::Target;
use lights::image::FloatImage;
use lights::kernel::{Beta, CurveShape, KernelFamily};
use lights::numerics::vector::{Rgb, Vec2};
use lights::settings::SharedSettings;

const TILE: u32 = 16;
const CELL_LIMIT: u64 = 64_000_000;
const RENDER_LIMIT: u64 = 40_000_000;
const QUADRATURE_DEPTH: u32 = 10;
const ATOMS: usize = 8;
const FAMILY: KernelFamily = KernelFamily::SuperGaussian(Beta::One);

fn truth_of(target: &Target<f64>) -> FloatImage<f64> {
    let mut budget = coverage_budget(CELL_LIMIT);
    let (coverage, _) = match exact_coverage(target, QUADRATURE_DEPTH, &mut budget) {
        Ok(v) => v,
        Err(e) => panic!("the ground truth for {} did not compute: {e}", target.name),
    };
    coverage_image_to_colour(target, &coverage)
}

fn request(kernel: KernelFamily, max_atoms: usize) -> FitRequest {
    let mut r = FitRequest::new(kernel, SharedSettings::<f64>::prefiltered());
    r.max_atoms = max_atoms;
    r.tile = TILE;
    r
}

fn plain_energies(target: &Target<f64>, truth: &FloatImage<f64>, kernel: KernelFamily, n: usize) -> (Vec<f64>, u64, Vec<Atom2d>) {
    let mut budget = fit_budget(RENDER_LIMIT);
    match fit(target, truth, &request(kernel, n), &mut budget) {
        Ok(o) => (o.steps.iter().map(|s| s.residual_energy).collect(), o.renders, o.atoms),
        Err(e) => panic!("the plain fit of {} failed: {e}", target.name),
    }
}

fn shaped_energies(target: &Target<f64>, truth: &FloatImage<f64>, kernel: KernelFamily, n: usize) -> (Vec<f64>, u64, Vec<ShapedAtom2d>) {
    let mut budget = fit_budget(RENDER_LIMIT);
    match fit_shaped(target, truth, &request(kernel, n), &mut budget) {
        Ok(o) => (o.steps.iter().map(|s| s.residual_energy).collect(), o.renders, o.atoms),
        Err(e) => panic!("the shaped fit of {} failed: {e}", target.name),
    }
}

fn beta_four_atom_target() -> (Target<f64>, FloatImage<f64>) {
    let target = Target::<f64>::t4_curve();
    let atom = ShapedAtom2d::new(
        Atom2d::from_sigma(
            Vec2::new(16.0, 16.0),
            Vec2::new(5.0, 3.0),
            0.4,
            1.0,
            Rgb::new(0.9, 0.9, 0.9),
        ),
        CurveShape::with_beta(4.0),
    );
    let settings = SharedSettings::<f64>::prefiltered();
    let (w, h) = target.viewport;
    let image = match render_shaped_atoms(&[atom], &camera(w, h), FAMILY, &settings, TILE, Collect::Pixels) {
        Ok(o) => o.image,
        Err(e) => panic!("the beta-4 truth did not render: {e}"),
    };
    (target, image)
}

#[test]
fn the_beta_knob_stays_inside_the_swept_range() {
    assert_eq!(nudge_beta(1.0, 0.0), 1.0);
    assert_eq!(nudge_beta(4.0, 10.0), MAX_BETA);
    assert_eq!(nudge_beta(1.0, -10.0), MIN_BETA);
    assert!((nudge_beta(1.0, 2.0f64.ln()) - 2.0).abs() < 1e-12);
}

#[test]
fn a_shaped_fit_under_a_curve_that_ignores_beta_is_the_plain_fit() {
    let target = Target::<f64>::t4_curve();
    let truth = truth_of(&target);
    let (plain_e, plain_r, plain_atoms) = plain_energies(&target, &truth, KernelFamily::Gaussian, 3);
    let (shaped_e, shaped_r, shaped_atoms) = shaped_energies(&target, &truth, KernelFamily::Gaussian, 3);
    assert_eq!(plain_atoms.len(), shaped_atoms.len());
    for (p, s) in plain_atoms.iter().zip(shaped_atoms.iter()) {
        assert_eq!(p.bits(), s.atom.bits(), "the nine scalars moved when only a beta knob was added");
        assert_eq!(s.shape.beta, 1.0, "beta moved under a kernel that does not read it");
    }
    assert_eq!(plain_e, shaped_e);
    assert!(shaped_r > plain_r, "the shaped ladder spent no renders on its tenth knob");
}

#[test]
fn the_beta_ladder_recovers_a_target_that_is_one_beta_four_atom() {
    let (target, truth) = beta_four_atom_target();
    let (plain_e, _, _) = plain_energies(&target, &truth, FAMILY, 1);
    let (shaped_e, _, shaped_atoms) = shaped_energies(&target, &truth, FAMILY, 1);
    let beta = shaped_atoms[0].shape.beta;
    println!("beta-4 control: plain {:.6} shaped {:.6} recovered beta {:.4}", plain_e[0], shaped_e[0], beta);
    assert!(beta > 2.0, "the ladder did not climb toward beta 4, it stopped at {beta}");
    assert!(shaped_e[0] < plain_e[0] * 0.5, "a tuned beta did not beat a gaussian on its own curve");
}

#[test]
fn a_tuned_beta_beats_a_tuned_gaussian_on_every_target() {
    let mut wins = 0usize;
    let mut cells = 0usize;
    for target in Target::<f64>::all() {
        let truth = truth_of(&target);
        let (plain_e, _, _) = plain_energies(&target, &truth, FAMILY, ATOMS);
        let (shaped_e, _, _) = shaped_energies(&target, &truth, FAMILY, ATOMS);
        for (p, s) in plain_e.iter().zip(shaped_e.iter()) {
            cells += 1;
            if s < p {
                wins += 1;
            }
        }
    }
    println!("shaped beats the tuned gaussian in {wins} of {cells} cells");
    assert!(cells >= 40, "only {cells} rungs were compared");
    assert!(wins >= 40, "a per-atom beta beat the tuned gaussian in only {wins} of {cells} rungs");
}

#[test]
fn a_tuned_beta_against_the_best_fixed_family() {
    println!("target\tn\tplain_energy\tshaped_energy\tratio\tbest_fixed_energy\tbest_fixed_beta\tshaped_over_fixed\tplain_renders\tshaped_renders\tbetas");
    let mut wins = 0usize;
    let mut fixed_wins = 0usize;
    let mut cells = 0usize;
    for target in Target::<f64>::all() {
        let truth = truth_of(&target);
        let (plain_e, plain_r, _) = plain_energies(&target, &truth, FAMILY, ATOMS);
        let (shaped_e, shaped_r, shaped_atoms) = shaped_energies(&target, &truth, FAMILY, ATOMS);
        let fixed: Vec<(Beta, Vec<f64>)> = Beta::SWEEP
            .iter()
            .map(|&b| (b, plain_energies(&target, &truth, KernelFamily::SuperGaussian(b), ATOMS).0))
            .collect();
        let betas: Vec<String> = shaped_atoms.iter().map(|a| format!("{:.2}", a.shape.beta)).collect();
        let rungs = plain_e.len().min(shaped_e.len());
        for n in 0..rungs {
            let mut best = (Beta::One, plain_e[n]);
            for (b, e) in fixed.iter() {
                match e.get(n) {
                    Some(&v) if v < best.1 => best = (*b, v),
                    Some(_) => (),
                    None => (),
                }
            }
            cells += 1;
            if shaped_e[n] < plain_e[n] {
                wins += 1;
            }
            if shaped_e[n] < best.1 {
                fixed_wins += 1;
            }
            println!(
                "{}\t{}\t{:.6}\t{:.6}\t{:.4}\t{:.6}\t{}\t{:.4}\t{}\t{}\t{}",
                target.name,
                n + 1,
                plain_e[n],
                shaped_e[n],
                shaped_e[n] / plain_e[n],
                best.1,
                best.0.label(),
                shaped_e[n] / best.1,
                plain_r,
                shaped_r,
                betas[..=n].join(",")
            );
        }
    }
    println!("shaped beats the tuned gaussian in {wins} of {cells} cells");
    println!("shaped beats the best fixed-beta family in {fixed_wins} of {cells} cells");
    assert!(cells > 0);
}
