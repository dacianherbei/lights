#![cfg(feature = "fit")]
use lights::fit::atom2d::Atom2d;
use lights::fit::coverage::{coverage_budget, coverage_image_to_colour, exact_coverage};
use lights::fit::descent::{
    CHOSEN_STEP, DIRECTION_SEED, LINE_SEARCH_HALVINGS, P, RENDERS_PER_GRADIENT, STEP_SWEEP,
    box_of, descend, descent_budget, directional_difference, dot, gradient, line_search, norm,
    pack, parameter_name, projected_descent, step_sweep, unit_direction, unpack,
};
use lights::fit::descent::{
    ADAM_DECAY_LEVELS, ADAM_RATE_SWEEP, ADAM_STEPS, AdamState, adam, adam_at_gain,
    fit as descent_fit, fit_budget as descent_fit_budget, parameter_scales,
};
use lights::fit::greedy::{
    Context, FitRequest, Initialisation, background_image, evaluate, fit, fit_budget, refine,
    seed_atom_scaled, seed_pixel,
};
use lights::fit::target::{EDGE_DEGREES, Target};
use lights::guard::Budget;
use lights::image::FloatImage;
use lights::kernel::KernelFamily;
use lights::numerics::vector::{Rgb, Vec2};
use lights::settings::SharedSettings;

const TILE: u32 = 16;
const CELL_LIMIT: u64 = 64_000_000;
const RENDER_LIMIT: u64 = 4_000_000;
const QUADRATURE_DEPTH: u32 = 10;
const IDENTITY_TOLERANCE: f64 = 1e-3;
const FLAT_MIDDLE: f64 = 1e-6;
const SMALLEST_DIRECTION_COMPONENT: f64 = 0.05;
const LADDER_CONVERGENCE_SWEEPS: u32 = 64;
const DESCENT_STEPS: u32 = 20;
const DESCENT_GAIN_FLOOR: f64 = 1e-3;

fn truth_of(target: &Target<f64>) -> FloatImage<f64> {
    let mut budget = coverage_budget(CELL_LIMIT);
    let (coverage, _) = match exact_coverage(target, QUADRATURE_DEPTH, &mut budget) {
        Ok(v) => v,
        Err(e) => panic!("the ground truth for {} did not compute: {e}", target.name),
    };
    coverage_image_to_colour(target, &coverage)
}

fn one_atom_request() -> FitRequest {
    let mut request = FitRequest::new(
        KernelFamily::Gaussian,
        SharedSettings::<f64>::prefiltered(),
    );
    request.max_atoms = 1;
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

fn seeded_atom(
    target: &Target<f64>,
    truth: &FloatImage<f64>,
    ctx: &Context<'_>,
    budget: &mut Budget,
) -> Atom2d {
    let background = background_image(target.viewport, ctx.settings.background);
    let residual = match truth.difference(&background) {
        Ok(r) => r,
        Err(e) => panic!("the residual of {} did not compute: {e}", target.name),
    };
    let seed = match seed_pixel(&residual) {
        Some(s) => s,
        None => panic!("{} has no seed pixel: the truth equals the background", target.name),
    };
    let _ = budget;
    seed_atom_scaled(
        truth,
        &residual,
        seed,
        Initialisation::StructureTensor,
        &ctx.settings,
        1.0,
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

fn energy_of(atoms: &[Atom2d], ctx: &Context<'_>, budget: &mut Budget) -> f64 {
    match evaluate(atoms, ctx, budget) {
        Ok(e) => e,
        Err(e) => panic!("the energy did not evaluate: {e}"),
    }
}

fn converge_the_ladder(atoms: &mut [Atom2d], ctx: &Context<'_>, budget: &mut Budget) -> (f64, u32) {
    let mut energy = energy_of(atoms, ctx, budget);
    let mut sweeps = 0u32;
    for _ in 0..LADDER_CONVERGENCE_SWEEPS {
        let after = match refine(atoms, 0, 1, energy, ctx, budget) {
            Ok(e) => e,
            Err(e) => panic!("the ladder sweep failed: {e}"),
        };
        sweeps += 1;
        if !(after < energy) {
            break;
        }
        energy = after;
    }
    let again = match refine(atoms, 0, 1, energy, ctx, budget) {
        Ok(e) => e,
        Err(e) => panic!("the ladder sweep failed: {e}"),
    };
    assert_eq!(
        again, energy,
        "the ladder still moved after {sweeps} sweeps: it is not stuck, so the \
         discriminating case has not been reached and LADDER_CONVERGENCE_SWEEPS \
         ({LADDER_CONVERGENCE_SWEEPS}) is too small"
    );
    (energy, sweeps)
}

fn gradient_of(atoms: &mut [Atom2d], eps: f64, ctx: &Context<'_>, budget: &mut Budget) -> [f64; P] {
    match gradient(atoms, 0, eps, ctx, budget) {
        Ok(g) => g,
        Err(e) => panic!("the gradient did not compute: {e}"),
    }
}

fn identity_error(atoms: &mut [Atom2d], eps: f64, ctx: &Context<'_>, budget: &mut Budget) -> (f64, [f64; P]) {
    let d = unit_direction(DIRECTION_SEED);
    let g = gradient_of(atoms, eps, ctx, budget);
    let along = match directional_difference(atoms, 0, &d, eps, ctx, budget) {
        Ok(v) => v,
        Err(e) => panic!("the directional difference did not compute: {e}"),
    };
    let scale = norm(&g);
    assert!(scale > 0.0, "the gradient is identically zero, so no identity can be checked");
    ((along - dot(&g, &d)).abs() / scale, g)
}

#[test]
fn pack_and_unpack_are_inverse_and_every_parameter_has_a_name() {
    let a = interior_atom();
    let v = pack(&a);
    let back = unpack(&v);
    assert_eq!(back.bits(), a.bits(), "pack then unpack must be the identity, bit for bit");
    let mut names: Vec<&str> = Vec::with_capacity(P);
    for i in 0..P {
        let name = parameter_name(i);
        assert!(!names.contains(&name), "parameter {i} shares the name {name} with an earlier one");
        names.push(name);
    }
    assert_eq!(names.len(), P);
    assert_eq!(v[8], a.colour.z, "the last slot of the vector is colour.b");
}

#[test]
fn the_direction_used_by_the_identity_check_touches_every_parameter() {
    let d = unit_direction(DIRECTION_SEED);
    assert!((norm(&d) - 1.0).abs() < 1e-12, "the direction is not unit length: {}", norm(&d));
    for i in 0..P {
        assert!(
            d[i].abs() >= SMALLEST_DIRECTION_COMPONENT,
            "component {} ({}) of the check direction is {:.4}: a partial missing there \
             would hide inside the tolerance, so change DIRECTION_SEED",
            i,
            parameter_name(i),
            d[i]
        );
    }
}

#[test]
fn a_gradient_costs_eighteen_renders_and_bills_its_own_site() {
    let target = Target::<f64>::t5_thin_stroke();
    let truth = truth_of(&target);
    let request = one_atom_request();
    let ctx = context(&truth, &request, target.viewport);
    let mut atoms = [interior_atom()];
    let mut budget = descent_budget(RENDER_LIMIT);
    assert_eq!(budget.site(), "fit::descent::gradient");
    let before = budget.used();
    let _ = gradient_of(&mut atoms, CHOSEN_STEP, &ctx, &mut budget);
    assert_eq!(
        budget.used() - before,
        RENDERS_PER_GRADIENT,
        "one gradient must cost exactly two renders per parameter"
    );
    assert_eq!(RENDERS_PER_GRADIENT, 2 * P as u64);
}

#[test]
fn the_directional_derivative_identity_holds_at_the_chosen_step() {
    let target = Target::<f64>::t5_thin_stroke();
    let truth = truth_of(&target);
    let request = one_atom_request();
    let ctx = context(&truth, &request, target.viewport);
    let mut budget = descent_budget(RENDER_LIMIT);

    let mut interior = [interior_atom()];
    let (err_interior, g_interior) = identity_error(&mut interior, CHOSEN_STEP, &ctx, &mut budget);
    for i in 0..P {
        assert!(
            g_interior[i] != 0.0,
            "the partial with respect to {} is exactly zero on the interior atom, so a \
             gradient that skipped it would pass this check",
            parameter_name(i)
        );
    }
    println!(
        "descent: identity error on the interior atom at eps {CHOSEN_STEP:e} is {err_interior:.3e} \
         relative to |g| = {:.6e}",
        norm(&g_interior)
    );
    assert!(
        err_interior < IDENTITY_TOLERANCE,
        "the directional derivative along d disagrees with g.d by {err_interior:.3e} of |g| \
         on the interior atom (tolerance {IDENTITY_TOLERANCE:e})"
    );

    let mut fitted = [fitted_atom(&target, &truth, &request)];
    let (err_fitted, g_fitted) = identity_error(&mut fitted, CHOSEN_STEP, &ctx, &mut budget);
    println!(
        "descent: identity error on the fitted atom at eps {CHOSEN_STEP:e} is {err_fitted:.3e} \
         relative to |g| = {:.6e}",
        norm(&g_fitted)
    );
    assert!(
        err_fitted < IDENTITY_TOLERANCE,
        "the directional derivative along d disagrees with g.d by {err_fitted:.3e} of |g| \
         on the fitted atom (tolerance {IDENTITY_TOLERANCE:e})"
    );
}

#[test]
fn the_step_sweep_measures_a_flat_middle_and_the_chosen_step_names_its_row() {
    let target = Target::<f64>::t5_thin_stroke();
    let truth = truth_of(&target);
    let request = one_atom_request();
    let ctx = context(&truth, &request, target.viewport);
    let mut budget = descent_budget(RENDER_LIMIT);
    let mut atoms = [fitted_atom(&target, &truth, &request)];
    let d = unit_direction(DIRECTION_SEED);
    let rows = match step_sweep(&mut atoms, 0, &STEP_SWEEP, &d, &ctx, &mut budget) {
        Ok(r) => r,
        Err(e) => panic!("the step sweep did not run: {e}"),
    };
    assert_eq!(rows.len(), STEP_SWEEP.len());
    println!("descent: step sweep on the fitted atom of {} ({} renders)", target.name, budget.used());
    println!("descent:   eps        |g|             change_vs_prev   identity_err");
    for r in rows.iter() {
        println!(
            "descent:   {:<9.0e}  {:<14.6e}  {:<15.3e}  {:.3e}",
            r.eps, r.gradient_norm, r.change_from_previous, r.identity_error
        );
    }
    let mut header = String::from("descent:   eps      ");
    for i in 0..P {
        header.push_str(&format!("  {:>12}", parameter_name(i)));
    }
    println!("{header}");
    for r in rows.iter() {
        let mut line = format!("descent:   {:<9.0e}", r.eps);
        for i in 0..P {
            line.push_str(&format!("  {:>12.5e}", r.gradient[i]));
        }
        println!("{line}");
    }
    assert!(
        rows[0].identity_error > IDENTITY_TOLERANCE,
        "at the largest step ({:e}) the two central estimates still agree to {:.3e}: the \
         sweep did not reach the curvature side, where they must disagree",
        rows[0].eps,
        rows[0].identity_error
    );
    let chosen = match rows.iter().position(|r| r.eps == CHOSEN_STEP) {
        Some(k) => k,
        None => panic!("CHOSEN_STEP {CHOSEN_STEP:e} is not a row of STEP_SWEEP"),
    };
    assert!(
        chosen > 0 && chosen + 1 < rows.len(),
        "the chosen step is an end of the sweep, so nothing bounds it on that side"
    );
    let row = rows[chosen];
    assert!(
        row.change_from_previous < FLAT_MIDDLE,
        "the gradient at the chosen step differs from the one a decade larger by \
         {:.3e} of its norm: that is not a flat middle (limit {FLAT_MIDDLE:e})",
        row.change_from_previous
    );
    assert!(
        row.identity_error < IDENTITY_TOLERANCE,
        "the identity error at the chosen step is {:.3e} (tolerance {IDENTITY_TOLERANCE:e})",
        row.identity_error
    );
    let first = rows[0];
    let last = rows[rows.len() - 1];
    assert!(
        rows[1].change_from_previous > row.change_from_previous,
        "the largest step ({:e}) agrees with the next decade as well as the chosen one does \
         ({:.3e} against {:.3e}): the sweep did not reach the curvature side",
        first.eps,
        rows[1].change_from_previous,
        row.change_from_previous
    );
    assert!(
        last.change_from_previous > row.change_from_previous,
        "the smallest step ({:e}) agrees with the previous decade as well as the chosen one \
         does ({:.3e} against {:.3e}): the sweep did not reach the cancellation side",
        last.eps,
        last.change_from_previous,
        row.change_from_previous
    );
}

#[test]
fn the_ladder_stops_where_the_descent_direction_still_lowers_the_energy() {
    let target = Target::<f64>::t5_thin_stroke();
    let truth = truth_of(&target);
    let request = one_atom_request();
    let ctx = context(&truth, &request, target.viewport);
    let mut budget = descent_budget(RENDER_LIMIT);
    let mut atoms = [fitted_atom(&target, &truth, &request)];

    let (stuck, sweeps) = converge_the_ladder(&mut atoms, &ctx, &mut budget);
    let stuck_atom = atoms[0];
    let g = gradient_of(&mut atoms, CHOSEN_STEP, &ctx, &mut budget);
    let x = pack(&atoms[0]);
    let bounds = box_of(&ctx);
    let direction = projected_descent(&g, &x, &bounds);
    println!(
        "descent: the ladder is stuck at energy {stuck:.6} after {sweeps} extra sweeps; \
         |g| = {:.6e}, |projected direction| = {:.6e}",
        norm(&g),
        norm(&direction)
    );
    for i in 0..P {
        println!(
            "descent:   {:<12} x = {:>12.6}  dL/dx = {:>14.6e}  direction = {:>14.6e}",
            parameter_name(i),
            x[i],
            g[i],
            direction[i]
        );
    }
    assert!(
        norm(&direction) > 0.0,
        "every component of the descent direction is zero or points out of the box: \
         the stuck point is a box-constrained stationary point and the entry's mechanism \
         does not apply here"
    );
    let step = match line_search(&mut atoms, 0, &direction, LINE_SEARCH_HALVINGS, &bounds, &ctx, &mut budget) {
        Ok(Some(s)) => s,
        Ok(None) => panic!(
            "no step along the projected gradient, from length 1 down to 2^-{LINE_SEARCH_HALVINGS}, \
             lowers the energy below the ladder's {stuck:.6}: the ladder's stop is a real minimum \
             along the gradient too"
        ),
        Err(e) => panic!("the line search failed: {e}"),
    };
    let one_step_gain = (stuck - step.loss) / stuck;
    println!(
        "descent: one line-searched step of length {:e} lowers the energy to {:.6}, \
         a gain of {:.3e} relative; the ladder's own smallest centre step is 8 * 2^-7 = 0.0625",
        step.length, step.loss, one_step_gain
    );

    atoms[0] = stuck_atom;
    let trace = match descend(&mut atoms, 0, DESCENT_STEPS, CHOSEN_STEP, &ctx, &mut budget) {
        Ok(t) => t,
        Err(e) => panic!("the descent from the stuck point failed: {e}"),
    };
    let last = match trace.losses.last() {
        Some(l) => *l,
        None => panic!("descend took no step from a point the line search just moved from"),
    };
    let gain = (trace.start - last) / trace.start;
    println!(
        "descent: {} steps of steepest descent from the stuck point: energy {:.6} -> {:.6}, \
         gain {:.3e} relative, {} renders",
        trace.losses.len(),
        trace.start,
        last,
        gain,
        trace.renders
    );
    let moved = pack(&atoms[0]);
    for i in 0..P {
        println!(
            "descent:   {:<12} {:>12.6} -> {:>12.6}",
            parameter_name(i),
            x[i],
            moved[i]
        );
    }
    assert_eq!(trace.start, stuck, "descend must start from the ladder's stuck energy");
    assert!(
        last < step.loss,
        "{} steps of descent ended at {last:.6}, no lower than the single line-searched \
         step's {:.6}: descend is not following the direction it computes",
        trace.losses.len(),
        step.loss
    );
    assert!(
        gain > DESCENT_GAIN_FLOOR,
        "descent from the ladder's stuck point gained only {gain:.3e} relative in \
         {DESCENT_STEPS} steps (floor {DESCENT_GAIN_FLOOR:e}): the ladder's stop was \
         not materially early on this target"
    );
}

const COMPARISON_ATOMS: usize = 4;

const FLAT_FILL_ADAM_TOLERANCE: f64 = 1e-2;

#[test]
fn the_rate_sweep_finds_no_single_gain_and_one_class_where_no_gain_moves_at_all() {
    let mut best_per_class: Vec<(String, f64, f64, f64, f64)> = Vec::new();
    for target in [
        Target::<f64>::t1_flat_fill(),
        Target::<f64>::t2_straight_edge(),
        Target::<f64>::t5_thin_stroke(),
    ] {
        let truth = truth_of(&target);
        let request = one_atom_request();
        let ctx = context(&truth, &request, target.viewport);
        let mut budget = descent_fit_budget(RENDER_LIMIT);
        let seed = seeded_atom(&target, &truth, &ctx, &mut budget);
        let start = energy_of(&[seed], &ctx, &mut budget);
        let seed_gradient = {
            let mut atoms = [seed];
            norm(&gradient_of(&mut atoms, CHOSEN_STEP, &ctx, &mut budget))
        };
        let ladder = {
            let mut atoms = [seed];
            match refine(&mut atoms, 0, request.refine_iterations, start, &ctx, &mut budget) {
                Ok(e) => e,
                Err(e) => panic!("the ladder refine failed: {e}"),
            }
        };
        println!(
            "rate: {} seed energy {start:.6}, |g| at the seed {seed_gradient:.6e}, ladder \
             reaches {ladder:.6}",
            target.name
        );
        let mut best = (f64::INFINITY, 0.0f64);
        for gain in ADAM_RATE_SWEEP {
            let mut atoms = [seed];
            let reached = match adam_at_gain(
                &mut atoms,
                0,
                ADAM_STEPS,
                CHOSEN_STEP,
                gain,
                start,
                &ctx,
                &mut budget,
            ) {
                Ok(e) => e,
                Err(e) => panic!("adam at gain {gain} failed: {e}"),
            };
            println!(
                "rate:   gain {gain:<6} reaches {reached:>14.6}   moved {:>+9.3e} of the seed   \
                 against the ladder {:>+9.3e}",
                (start - reached) / start,
                (ladder - reached) / ladder
            );
            if reached < best.0 {
                best = (reached, gain);
            }
        }
        println!(
            "rate: {} best gain {} at energy {:.6}, ladder {ladder:.6}",
            target.name, best.1, best.0
        );
        best_per_class.push((target.name.to_string(), best.1, best.0, ladder, start));
    }
    assert_eq!(best_per_class.len(), 3);
    let gains: Vec<f64> = best_per_class.iter().map(|c| c.1).collect();
    assert!(
        gains[0] != gains[1] || gains[1] != gains[2],
        "all three classes chose the same gain ({:?}). That contradicts the measurement this \
         test was written on, where they chose three different rows; re-measure and rewrite \
         BURNDOWN L-132 rather than loosening this",
        gains
    );

    let stroke = match best_per_class.iter().find(|c| c.0 == "t5-thin-stroke") {
        Some(c) => c,
        None => panic!("the thin stroke is not in the sweep"),
    };
    assert_eq!(
        stroke.2, stroke.4,
        "on the thin stroke the best of {} gains reached {:.6} against a seed energy of {:.6}: \
         adam moved. The finding L-132 records is that NO gain across a 40x range improves on \
         the seed there while the ladder reaches {:.6}, so if this now moves the entry is stale",
        ADAM_RATE_SWEEP.len(),
        stroke.2,
        stroke.4,
        stroke.3
    );

    let fill = match best_per_class.iter().find(|c| c.0 == "t1-flat-fill") {
        Some(c) => c,
        None => panic!("the flat fill is not in the sweep"),
    };
    let behind = (fill.2 - fill.3) / fill.3;
    println!(
        "rate: on the flat fill adam's best is {behind:+.3e} behind the ladder, which is the \
         guard that the implementation is not simply broken"
    );
    assert!(
        behind.abs() < FLAT_FILL_ADAM_TOLERANCE,
        "on the flat fill adam's best gain reaches {:.6} against the ladder's {:.6}, \
         {behind:+.3e} apart (tolerance {FLAT_FILL_ADAM_TOLERANCE:e}). The thin-stroke finding \
         only means something if adam CAN match the ladder somewhere; if it cannot, the \
         implementation is the suspect and not the method",
        fill.2,
        fill.3
    );
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

fn comparison_request(max_atoms: usize) -> FitRequest {
    let mut request = FitRequest::new(
        KernelFamily::Gaussian,
        SharedSettings::<f64>::prefiltered(),
    );
    request.max_atoms = max_atoms;
    request.tile = TILE;
    request
}

#[test]
fn the_adam_update_is_scaled_per_parameter_and_bias_corrected() {
    let scales = parameter_scales(1.0);
    assert!(
        scales[0] > scales[6] * 4.0,
        "the centre scale ({}) is not far above the colour scale ({}); one learning rate over \
         nine parameters of different units is the trap this vector exists to avoid",
        scales[0],
        scales[6]
    );
    let mut state = AdamState::new(ADAM_STEPS);
    let mut g = [0.0f64; P];
    for i in 0..P {
        g[i] = 1.0;
    }
    let rate = parameter_scales(1.0);
    let first = state.update(&g, &rate);
    assert_eq!(state.steps, 1);
    for i in 0..P {
        assert!(
            first[i] < 0.0,
            "the first step along a positive gradient must go downhill in {} and it is {:.6e}",
            parameter_name(i),
            first[i]
        );
        let relative = (first[i].abs() - rate[i]).abs() / rate[i];
        assert!(
            relative < 1e-6,
            "with bias correction the first step of Adam under a unit gradient must be exactly \
             the rate in {}; it is {:.6e} against a rate of {:.6e} ({relative:.3e} off), which \
             is what a missing 1 - beta^t denominator looks like",
            parameter_name(i),
            first[i].abs(),
            rate[i]
        );
    }

    let mut schedule = AdamState::new(ADAM_STEPS);
    let mut seen: Vec<f64> = Vec::new();
    for _ in 0..ADAM_STEPS {
        seen.push(schedule.decay());
        schedule.steps += 1;
    }
    println!("adam: the decay schedule over {ADAM_STEPS} steps is {seen:?}");
    assert_eq!(seen[0], 1.0, "the schedule must open at a full rate, for reach");
    let last = seen[seen.len() - 1];
    let floor = 0.5f64.powi((ADAM_DECAY_LEVELS - 1) as i32);
    assert_eq!(
        last, floor,
        "the schedule must close at 2^-({ADAM_DECAY_LEVELS} - 1) = {floor}, for precision; it \
         closes at {last}"
    );
    let levels = {
        let mut distinct: Vec<f64> = Vec::new();
        for v in seen.iter() {
            if distinct.last() != Some(v) {
                distinct.push(*v);
            }
        }
        distinct
    };
    assert_eq!(
        levels.len() as u32,
        ADAM_DECAY_LEVELS,
        "the schedule visits {} distinct rates over {ADAM_STEPS} steps and ADAM_DECAY_LEVELS is \
         {ADAM_DECAY_LEVELS}: {levels:?}. A schedule that never decays keeps the ladder's reach \
         and loses its precision, and a schedule that decays at once keeps neither",
        levels.len()
    );
    for pair in levels.windows(2) {
        assert_eq!(
            pair[1] * 2.0,
            pair[0],
            "the schedule steps from {} to {} and that is not a halving, so it is not the \
             ladder's own resolution schedule any more",
            pair[0],
            pair[1]
        );
    }
}

#[test]
fn adam_never_leaves_an_atom_worse_than_it_found_it() {
    let target = Target::<f64>::t5_thin_stroke();
    let truth = truth_of(&target);
    let request = one_atom_request();
    let ctx = context(&truth, &request, target.viewport);
    let mut budget = descent_fit_budget(RENDER_LIMIT);
    let mut atoms = [interior_atom()];
    let before = energy_of(&atoms, &ctx, &mut budget);
    let after = match adam(&mut atoms, 0, ADAM_STEPS, CHOSEN_STEP, before, &ctx, &mut budget) {
        Ok(e) => e,
        Err(e) => panic!("adam failed: {e}"),
    };
    let redrawn = energy_of(&atoms, &ctx, &mut budget);
    println!(
        "adam: energy {before:.6} -> {after:.6} over {ADAM_STEPS} steps; the atom it left \
         redraws at {redrawn:.6}"
    );
    assert!(
        after <= before,
        "adam returned {after:.6}, worse than the {before:.6} it was given; it keeps the best \
         point it saw, so this cannot happen unless that bookkeeping is gone"
    );
    assert_eq!(
        redrawn, after,
        "the atom adam left in place does not render to the energy it reported: the returned \
         number and the returned atom are from different points"
    );
}

#[test]
fn descent_fit_matches_greedys_contract_and_is_compared_at_equal_atom_count() {
    let mut cells: Vec<(String, usize, f64, f64, u64, u64)> = Vec::new();
    println!(
        "compare: class            n   greedy_energy    descent_energy   descent_gain   \
         greedy_renders  descent_renders"
    );
    for target in every_class() {
        let truth = truth_of(&target);
        let request = comparison_request(COMPARISON_ATOMS);

        let mut greedy_budget = fit_budget(RENDER_LIMIT);
        let greedy = match fit(&target, &truth, &request, &mut greedy_budget) {
            Ok(o) => o,
            Err(e) => panic!("the greedy fit of {} failed: {e}", target.name),
        };

        let mut adam_budget = descent_fit_budget(RENDER_LIMIT);
        assert_eq!(
            adam_budget.site(),
            "fit::descent::fit",
            "the descent fitter must bill its own site, or a budget spent here is reported as \
             greedy's"
        );
        let descent = match descent_fit(&target, &truth, &request, &mut adam_budget) {
            Ok(o) => o,
            Err(e) => panic!("the descent fit of {} failed: {e}", target.name),
        };

        assert_eq!(
            descent.render_threads, greedy.render_threads,
            "{}: the two fitters disagree about the thread pin, so their render counts are not \
             comparable",
            target.name
        );
        assert_eq!(
            descent.steps.len(),
            descent.atoms.len(),
            "{}: descent recorded {} steps for {} atoms; FitStep is one per placement",
            target.name,
            descent.steps.len(),
            descent.atoms.len()
        );
        for a in descent.atoms.iter() {
            assert!(
                a.is_finite(),
                "{}: descent emitted a non-finite atom",
                target.name
            );
        }
        assert_eq!(
            descent.scene.len(),
            descent.atoms.len(),
            "{}: descent's scene and atom list disagree, so to_scene dropped one",
            target.name
        );

        if descent.atoms.len() != greedy.atoms.len() {
            println!(
                "compare: {:<16} COUNTS DIFFER: greedy placed {}, descent placed {}; \
                 exhausted {} against {}",
                target.name,
                greedy.atoms.len(),
                descent.atoms.len(),
                greedy.improvement_exhausted,
                descent.improvement_exhausted
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
        cells.push((
            target.name.to_string(),
            n,
            g_energy,
            d_energy,
            greedy.renders,
            descent.renders,
        ));
    }
    assert!(
        cells.len() >= 4,
        "only {} classes reached an equal-count comparison; below that the verdict is about \
         the harness and not about the two searches",
        cells.len()
    );
    let wins = cells.iter().filter(|c| c.3 < c.2).count();
    let losses = cells.iter().filter(|c| c.3 > c.2).count();
    let renders_greedy: u64 = cells.iter().map(|c| c.4).sum();
    let renders_descent: u64 = cells.iter().map(|c| c.5).sum();
    println!(
        "compare: {} classes at equal atom count; descent lower in {wins}, greedy lower in \
         {losses}; renders {renders_greedy} greedy against {renders_descent} descent \
         ({:.2}x)",
        cells.len(),
        renders_descent as f64 / renders_greedy as f64
    );
    for c in cells.iter() {
        if c.3 > c.2 {
            println!(
                "compare:   DESCENT LOSES: {} n={} greedy {:.6} descent {:.6} ({:+.3e})",
                c.0,
                c.1,
                c.2,
                c.3,
                (c.2 - c.3) / c.2
            );
        }
    }
    let ratio = renders_descent as f64 / renders_greedy as f64;
    assert!(
        ratio < 1.25 && ratio > 0.9,
        "descent spent {ratio:.2}x greedy's renders. The comparison below is only about the two \
         SEARCHES if the two fitters are otherwise the same fitter, so the budgets must be \
         within a tenth of each other; a ratio far below 1 means descent skipped work greedy \
         does, which is a different fitter and not a faster one"
    );
    assert_eq!(
        wins, 0,
        "descent is now lower than greedy on {wins} of {} classes at equal atom count and equal \
         budget. BURNDOWN L-132 records the opposite -- 0 wins, 6 losses -- so the entry is \
         stale and must be rewritten with the new table. Do not loosen this assertion to make \
         it pass.",
        cells.len()
    );
    assert_eq!(
        losses,
        cells.len(),
        "descent matched greedy exactly on {} of {} classes, which neither the win nor the loss \
         branch describes; re-measure before reading either",
        cells.len() - wins - losses,
        cells.len()
    );
}
