#![cfg(feature = "fit")]
use lights::error::LightsError;
use lights::fit::atom2d::Atom2d;
use lights::fit::backward::{Backward, RENDERS_PER_ANALYTIC_GRADIENT, analytic_gradient};
use lights::fit::coverage::{coverage_budget, coverage_image_to_colour, exact_coverage};
use lights::fit::descent::{
    CHOSEN_STEP, GradientSource, P, RENDERS_PER_GRADIENT, descent_budget, fit_budget as descent_fit_budget,
    fit_with, gradient, norm, parameter_name,
};
use lights::fit::descent::{LINE_SEARCH_HALVINGS, descend_with};
use lights::fit::greedy::{
    Context, FitRequest, background_image, evaluate, fit, fit_budget, refine, render_or_background,
    seed_atom_scaled, seed_pixel,
};
use lights::fit::target::{EDGE_DEGREES, Target};
use lights::guard::Budget;
use lights::image::FloatImage;
use lights::kernel::{Beta, KernelFamily};
use lights::numerics::vector::{Rgb, Vec2};
use lights::settings::SharedSettings;

const TILE: u32 = 16;
const CELL_LIMIT: u64 = 64_000_000;
const RENDER_LIMIT: u64 = 4_000_000;
const QUADRATURE_DEPTH: u32 = 10;
const AGREEMENT: f64 = 1e-7;
const FORWARD_AGREEMENT: f64 = 1e-12;
const COMPARISON_ATOMS: usize = 4;
const PRICE_REPS: usize = 5;

fn truth_of(target: &Target<f64>) -> FloatImage<f64> {
    let mut budget = coverage_budget(CELL_LIMIT);
    let (coverage, _) = match exact_coverage(target, QUADRATURE_DEPTH, &mut budget) {
        Ok(v) => v,
        Err(e) => panic!("the ground truth for {} did not compute: {e}", target.name),
    };
    coverage_image_to_colour(target, &coverage)
}

fn request_of(kernel: KernelFamily, max_atoms: usize) -> FitRequest {
    let mut request = FitRequest::new(kernel, SharedSettings::<f64>::prefiltered());
    request.max_atoms = max_atoms;
    request.tile = TILE;
    request
}

fn context<'a>(truth: &'a FloatImage<f64>, request: &FitRequest, viewport: (u32, u32)) -> Context<'a> {
    match Context::new(truth, request, viewport) {
        Ok(c) => c,
        Err(e) => panic!("the fit context did not build: {e}"),
    }
}

fn interior_atom() -> Atom2d {
    Atom2d::from_sigma(
        Vec2::new(31.3, 32.6),
        Vec2::new(6.0, 1.2),
        EDGE_DEGREES.to_radians() + 0.2,
        0.8,
        Rgb::new(0.7, 0.75, 0.65),
    )
}

fn occluder() -> Atom2d {
    Atom2d::from_sigma(
        Vec2::new(29.4, 34.1),
        Vec2::new(3.5, 2.2),
        -0.6,
        0.55,
        Rgb::new(0.2, 0.35, 0.9),
    )
}

fn fitted_atom(target: &Target<f64>, truth: &FloatImage<f64>, request: &FitRequest) -> Atom2d {
    let mut budget = fit_budget(RENDER_LIMIT);
    let outcome = match fit(target, truth, request, &mut budget) {
        Ok(o) => o,
        Err(e) => panic!("the one-atom fit of {} failed: {e}", target.name),
    };
    assert_eq!(outcome.atoms.len(), 1, "the fit must hand back exactly one atom");
    outcome.atoms[0]
}

fn backward_of(atoms: &[Atom2d], index: usize, ctx: &Context<'_>, budget: &mut Budget) -> Backward {
    match analytic_gradient(atoms, index, ctx, budget) {
        Ok(b) => b,
        Err(e) => panic!("the analytic gradient did not compute: {e}"),
    }
}

fn difference_of(atoms: &mut [Atom2d], index: usize, ctx: &Context<'_>, budget: &mut Budget) -> [f64; P] {
    match gradient(atoms, index, CHOSEN_STEP, ctx, budget) {
        Ok(g) => g,
        Err(e) => panic!("the finite-difference gradient did not compute: {e}"),
    }
}

fn assert_agrees(label: &str, atoms: &mut [Atom2d], index: usize, ctx: &Context<'_>) -> f64 {
    let mut budget = descent_budget(RENDER_LIMIT);
    let analytic = backward_of(atoms, index, ctx, &mut budget);
    assert!(analytic.visible, "{label}: the atom under test was culled, so nothing is measured");
    let fd = difference_of(atoms, index, ctx, &mut budget);
    let scale = norm(&fd);
    assert!(scale > 0.0, "{label}: the finite-difference gradient is zero, so no agreement can be read");
    let mut worst = 0.0f64;
    for i in 0..P {
        let gap = (analytic.gradient[i] - fd[i]).abs() / scale;
        println!(
            "backward: {label:<22} {:<12} analytic {:>+15.8e}  difference {:>+15.8e}  gap/|g| {gap:.3e}",
            parameter_name(i),
            analytic.gradient[i],
            fd[i]
        );
        assert!(
            fd[i] != 0.0,
            "{label}: the partial with respect to {} is exactly zero, so an analytic gradient \
             that skipped that parameter would pass",
            parameter_name(i)
        );
        assert!(
            gap < AGREEMENT,
            "{label}: the analytic partial with respect to {} is {:+.8e} and the central \
             difference at eps {CHOSEN_STEP:e} is {:+.8e}, a gap of {gap:.3e} of |g| \
             (tolerance {AGREEMENT:e})",
            parameter_name(i),
            analytic.gradient[i],
            fd[i]
        );
        worst = worst.max(gap);
    }
    worst
}

#[test]
fn the_reverse_pass_reproduces_the_renderers_image_and_loss() {
    let target = Target::<f64>::t5_thin_stroke();
    let truth = truth_of(&target);
    let request = request_of(KernelFamily::Gaussian, 2);
    let ctx = context(&truth, &request, target.viewport);
    let atoms = [interior_atom(), occluder()];
    let mut budget = descent_budget(RENDER_LIMIT);
    let backward = backward_of(&atoms, 0, &ctx, &mut budget);
    let rendered = match render_or_background(&atoms, &ctx.spec, ctx.kernel, &ctx.settings, ctx.tile) {
        Ok(i) => i,
        Err(e) => panic!("the renderer failed: {e}"),
    };
    let mut worst = 0.0f64;
    for (a, b) in backward.image.pixels.iter().zip(rendered.pixels.iter()) {
        let d = *a - *b;
        worst = worst.max(d.x.abs()).max(d.y.abs()).max(d.z.abs());
    }
    println!("backward: forward replica against the tile renderer, worst channel gap {worst:.3e}");
    assert!(
        worst <= FORWARD_AGREEMENT,
        "the reverse pass's own forward image differs from the renderer's by {worst:.3e}; a \
         gradient of a different image is a gradient of a different loss"
    );
    let loss = match evaluate(&atoms, &ctx, &mut budget) {
        Ok(e) => e,
        Err(e) => panic!("the energy did not evaluate: {e}"),
    };
    assert!(
        (backward.loss - loss).abs() <= FORWARD_AGREEMENT * loss.max(1.0),
        "the reverse pass reports loss {} and evaluate() reports {loss}",
        backward.loss
    );
}

#[test]
fn the_analytic_gradient_agrees_with_the_central_difference() {
    let target = Target::<f64>::t5_thin_stroke();
    let truth = truth_of(&target);
    let request = request_of(KernelFamily::Gaussian, 1);
    let ctx = context(&truth, &request, target.viewport);

    let mut interior = [interior_atom()];
    let w1 = assert_agrees("interior", &mut interior, 0, &ctx);

    let mut fitted = [fitted_atom(&target, &truth, &request)];
    let w2 = assert_agrees("fitted", &mut fitted, 0, &ctx);

    let two = request_of(KernelFamily::Gaussian, 2);
    let ctx2 = context(&truth, &two, target.viewport);
    let mut behind = [interior_atom(), occluder()];
    let w3 = assert_agrees("behind an occluder", &mut behind, 0, &ctx2);
    let w4 = assert_agrees("the occluder", &mut behind, 1, &ctx2);
    println!("backward: worst gap/|g| interior {w1:.3e} fitted {w2:.3e} behind {w3:.3e} occluder {w4:.3e}");
}

#[test]
fn an_analytic_gradient_bills_one_render_to_the_callers_site() {
    let target = Target::<f64>::t2_straight_edge();
    let truth = truth_of(&target);
    let request = request_of(KernelFamily::Gaussian, 1);
    let ctx = context(&truth, &request, target.viewport);
    let mut budget = descent_budget(RENDER_LIMIT);
    let before = budget.used();
    let _ = backward_of(&[interior_atom()], 0, &ctx, &mut budget);
    assert_eq!(budget.used() - before, RENDERS_PER_ANALYTIC_GRADIENT);
}

#[test]
fn a_kernel_without_a_reverse_pass_is_refused_rather_than_differentiated_as_a_gaussian() {
    let target = Target::<f64>::t2_straight_edge();
    let truth = truth_of(&target);
    let mut refused = 0usize;
    for kernel in [
        KernelFamily::GaussianNormalised,
        KernelFamily::SuperGaussian(Beta::Two),
        KernelFamily::CompactBump,
    ] {
        let request = request_of(kernel, 1);
        let ctx = context(&truth, &request, target.viewport);
        let mut budget = descent_budget(RENDER_LIMIT);
        match analytic_gradient(&[interior_atom()], 0, &ctx, &mut budget) {
            Err(LightsError::GradientUnsupported { .. }) => refused += 1,
            Err(e) => panic!("{}: refused for the wrong reason: {e}", kernel.name()),
            Ok(_) => panic!("{}: differentiated a kernel the reverse pass does not carry", kernel.name()),
        }
    }
    assert_eq!(refused, 3);
}

fn every_class() -> [Target<f64>; 6] {
    [
        Target::<f64>::t1_flat_fill(),
        Target::<f64>::t2_straight_edge(),
        Target::<f64>::t3_right_angle_corner(),
        Target::<f64>::t4_curve(),
        Target::<f64>::t5_thin_stroke(),
        Target::<f64>::t6_glyph_like(),
    ]
}

fn seconds<F: FnMut()>(mut f: F) -> f64 {
    let start = std::time::Instant::now();
    f();
    start.elapsed().as_secs_f64()
}

#[test]
fn the_price_of_an_analytic_gradient_is_measured_in_renders() {
    println!("price: class            n   rep   render_s        analytic_s      renders_per_gradient");
    let mut ratios: Vec<f64> = Vec::new();
    for target in [Target::<f64>::t5_thin_stroke(), Target::<f64>::t6_glyph_like()] {
        let truth = truth_of(&target);
        let request = request_of(KernelFamily::Gaussian, COMPARISON_ATOMS);
        let mut budget = fit_budget(RENDER_LIMIT);
        let fitted = match fit(&target, &truth, &request, &mut budget) {
            Ok(o) => o,
            Err(e) => panic!("the fit of {} failed: {e}", target.name),
        };
        let ctx = context(&truth, &request, target.viewport);
        let atoms = fitted.atoms;
        let last = atoms.len() - 1;
        for rep in 0..PRICE_REPS {
            let mut b = descent_budget(RENDER_LIMIT);
            let render_s = seconds(|| {
                let _ = evaluate(&atoms, &ctx, &mut b);
            });
            let analytic_s = seconds(|| {
                let _ = backward_of(&atoms, last, &ctx, &mut b);
            });
            let ratio = analytic_s / render_s;
            println!(
                "price: {:<16} {:>2}   {rep:>3}   {render_s:>13.6e}   {analytic_s:>13.6e}   {ratio:>8.3}",
                target.name,
                atoms.len()
            );
            ratios.push(ratio);
        }
    }
    let lo = ratios.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = ratios.iter().cloned().fold(0.0f64, f64::max);
    println!(
        "price: one analytic gradient costs {lo:.3}x to {hi:.3}x one render over {} reps, \
         against {RENDERS_PER_GRADIENT} renders for the central difference",
        ratios.len()
    );
    assert_eq!(ratios.len(), 2 * PRICE_REPS);
    assert!(
        hi < RENDERS_PER_GRADIENT as f64,
        "the analytic gradient costs {hi:.3}x a render at worst, which is no cheaper than the \
         {RENDERS_PER_GRADIENT} renders of the central difference it replaces"
    );
}

#[test]
fn the_analytic_descent_is_compared_with_the_ladder_at_equal_renders() {
    assert_eq!(GradientSource::CentralDifference.adam_steps(), 24);
    assert_eq!(GradientSource::Analytic.adam_steps(), 228);
    let mut cells: Vec<(String, usize, f64, f64, u64, u64)> = Vec::new();
    println!(
        "compare: class            n   greedy_energy    analytic_energy  analytic_gain  \
         greedy_renders  analytic_renders"
    );
    for target in every_class() {
        let truth = truth_of(&target);
        let request = request_of(KernelFamily::Gaussian, COMPARISON_ATOMS);
        let mut greedy_budget = fit_budget(RENDER_LIMIT);
        let greedy = match fit(&target, &truth, &request, &mut greedy_budget) {
            Ok(o) => o,
            Err(e) => panic!("the greedy fit of {} failed: {e}", target.name),
        };
        let mut budget = descent_fit_budget(RENDER_LIMIT);
        let descent = match fit_with(GradientSource::Analytic, &target, &truth, &request, &mut budget) {
            Ok(o) => o,
            Err(e) => panic!("the analytic descent fit of {} failed: {e}", target.name),
        };
        let mut fd_budget = descent_fit_budget(RENDER_LIMIT);
        let difference = match fit_with(
            GradientSource::CentralDifference,
            &target,
            &truth,
            &request,
            &mut fd_budget,
        ) {
            Ok(o) => o,
            Err(e) => panic!("the central-difference descent fit of {} failed: {e}", target.name),
        };
        assert_eq!(descent.steps.len(), difference.steps.len());
        for (a, b) in descent.steps.iter().zip(difference.steps.iter()) {
            let gap = (a.residual_energy - b.residual_energy).abs() / b.residual_energy.max(1.0);
            assert!(
                gap < 1e-9,
                "{}: Adam fed the analytic gradient reached {} and fed the central difference \
                 reached {}. BURNDOWN L-147 measured the two IDENTICAL, because Adam's first \
                 bias-corrected step is sign(g) times the rate and every improvement it ever \
                 made fell in steps 1-5. Adam now reads the gradient's magnitude, so L-147's \
                 finding is stale: re-measure both tables before quoting either",
                target.name,
                a.residual_energy,
                b.residual_energy
            );
        }
        if descent.atoms.len() != greedy.atoms.len() {
            println!(
                "compare: {:<16} COUNTS DIFFER: greedy placed {}, analytic placed {}",
                target.name,
                greedy.atoms.len(),
                descent.atoms.len()
            );
            continue;
        }
        let n = descent.atoms.len();
        let g_energy = match greedy.steps.last() {
            Some(s) => s.residual_energy,
            None => panic!("{}: greedy recorded no step", target.name),
        };
        let d_energy = match descent.steps.last() {
            Some(s) => s.residual_energy,
            None => panic!("{}: descent recorded no step", target.name),
        };
        let gain = (g_energy - d_energy) / g_energy;
        println!(
            "compare: {:<16} {n:>3}   {g_energy:>13.6}   {d_energy:>14.6}   {gain:>+11.3e}   \
             {:>14}  {:>15}",
            target.name, greedy.renders, descent.renders
        );
        cells.push((target.name.to_string(), n, g_energy, d_energy, greedy.renders, descent.renders));
    }
    assert!(
        cells.len() >= 4,
        "only {} classes reached an equal-count comparison",
        cells.len()
    );
    let wins = cells.iter().filter(|c| c.3 < c.2).count();
    let losses = cells.iter().filter(|c| c.3 > c.2).count();
    let renders_greedy: u64 = cells.iter().map(|c| c.4).sum();
    let renders_descent: u64 = cells.iter().map(|c| c.5).sum();
    let ratio = renders_descent as f64 / renders_greedy as f64;
    println!(
        "compare: {} classes at equal atom count; analytic lower in {wins}, greedy lower in \
         {losses}; renders {renders_greedy} greedy against {renders_descent} analytic ({ratio:.2}x)",
        cells.len()
    );
    assert!(
        ratio < 1.25 && ratio > 0.9,
        "the analytic descent spent {ratio:.2}x greedy's renders, so the comparison is not at \
         equal budget"
    );
    assert_eq!(
        losses,
        cells.len(),
        "analytic Adam is no longer lower-bounded by the ladder on every class; L-147 recorded \
         0 wins; re-measure"
    );
}

#[test]
fn a_line_search_fed_the_exact_gradient_is_never_worse_than_one_fed_the_difference() {
    let mut beats_ladder: Vec<String> = Vec::new();
    println!(
        "linesearch: class                 seed_energy   ladder_energy  cd_energy      analytic_energy  \
         ladder_r  cd_r  analytic_r"
    );
    for target in every_class() {
        let truth = truth_of(&target);
        let request = request_of(KernelFamily::Gaussian, 1);
        let ctx = context(&truth, &request, target.viewport);
        let background = background_image(target.viewport, ctx.settings.background);
        let residual = match truth.difference(&background) {
            Ok(r) => r,
            Err(e) => panic!("{}: residual failed: {e}", target.name),
        };
        let seed = match seed_pixel(&residual) {
            Some(s) => s,
            None => panic!("{} has no seed pixel", target.name),
        };
        let seeded = seed_atom_scaled(
            &truth,
            &residual,
            seed,
            request.initialisation,
            &ctx.settings,
            request.step_scale,
        );
        let mut ladder_atoms = [seeded];
        let mut ladder_budget = descent_fit_budget(RENDER_LIMIT);
        let start = energy_or_panic(&ladder_atoms, &ctx, &mut ladder_budget);
        let ladder = match refine(
            &mut ladder_atoms,
            0,
            request.refine_iterations,
            start,
            &ctx,
            &mut ladder_budget,
        ) {
            Ok(e) => e,
            Err(e) => panic!("{}: the ladder failed: {e}", target.name),
        };
        let ladder_renders = ladder_budget.used();
        let mut finals = [0.0f64; 2];
        let mut renders = [0u64; 2];
        for (k, source) in GradientSource::ALL.iter().enumerate() {
            let per_step = source.renders_per_gradient() + LINE_SEARCH_HALVINGS as u64 + 2;
            let steps = (ladder_renders / per_step).max(1) as u32;
            let mut atoms = [seeded];
            let mut budget = descent_fit_budget(RENDER_LIMIT);
            let trace = match descend_with(*source, &mut atoms, 0, steps, CHOSEN_STEP, &ctx, &mut budget) {
                Ok(t) => t,
                Err(e) => panic!("{}: the {} line search failed: {e}", target.name, source.name()),
            };
            finals[k] = trace.losses.last().cloned().unwrap_or(trace.start);
            renders[k] = trace.renders;
            assert!(
                trace.renders <= ladder_renders,
                "{}: the {} line search spent {} renders against the ladder's {ladder_renders}",
                target.name,
                source.name(),
                trace.renders
            );
        }
        println!(
            "linesearch: {:<22} {start:>12.4}  {ladder:>12.4}  {:>12.4}  {:>14.4}  {ladder_renders:>8}  {:>4}  {:>10}",
            target.name, finals[0], finals[1], renders[0], renders[1]
        );
        assert!(
            finals[1] <= finals[0] * (1.0 + 1e-12),
            "{}: the line search fed the exact gradient ended at {} and fed the central \
             difference at {}; the exact gradient must not do worse at more steps for the same \
             renders",
            target.name,
            finals[1],
            finals[0]
        );
        if finals[1] < ladder {
            beats_ladder.push(target.name.to_string());
        }
    }
    println!("linesearch: the exact-gradient line search beats the ladder on {beats_ladder:?}");
    assert_eq!(
        beats_ladder,
        vec!["t4-curve".to_string(), "t6-glyph-like".to_string()],
        "BURNDOWN L-147 measured the exact-gradient line search beating the ladder on t4-curve \
         and t6-glyph-like only, at equal renders from the same seed; the set moved, so re-measure \
         and rewrite the record before quoting it"
    );
}

fn energy_or_panic(atoms: &[Atom2d], ctx: &Context<'_>, budget: &mut Budget) -> f64 {
    match evaluate(atoms, ctx, budget) {
        Ok(e) => e,
        Err(e) => panic!("the energy did not evaluate: {e}"),
    }
}
