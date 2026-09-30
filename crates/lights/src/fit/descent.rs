use crate::error::Result;
use crate::fit::atom2d::{Atom2d, to_scene};
use crate::fit::backward::{RENDERS_PER_ANALYTIC_GRADIENT, analytic_gradient};
use crate::fit::greedy::{
    ANGLE_STEP, COLOUR_STEP, Context, FitOutcome, FitRequest, FitStep, LOG_SCALE_STEP, MAX_OPACITY,
    OPACITY_STEP, background_image, centre_step, evaluate, render_or_background, residual_energy,
    residual_peak, seed_atom_scaled, seed_pixel,
};
use crate::fit::target::Target;
use crate::guard::Budget;
use crate::image::FloatImage;
use crate::numerics::vector::{Rgb, Vec2};

pub const P: usize = 9;
pub const STEP_SWEEP: [f64; 10] = [
    1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10,
];
pub const CHOSEN_STEP: f64 = 1e-6;
pub const DIRECTION_SEED: u64 = 20260915;
pub const LINE_SEARCH_HALVINGS: u32 = 16;
pub const RENDERS_PER_GRADIENT: u64 = 2 * P as u64;

pub fn descent_budget(limit: u64) -> Budget {
    Budget::new("fit::descent::gradient", limit)
}

pub fn pack(a: &Atom2d) -> [f64; P] {
    [
        a.centre.x,
        a.centre.y,
        a.log_scale.x,
        a.log_scale.y,
        a.angle,
        a.opacity,
        a.colour.x,
        a.colour.y,
        a.colour.z,
    ]
}

pub fn unpack(v: &[f64; P]) -> Atom2d {
    Atom2d {
        centre: Vec2::new(v[0], v[1]),
        log_scale: Vec2::new(v[2], v[3]),
        angle: v[4],
        opacity: v[5],
        colour: Rgb::new(v[6], v[7], v[8]),
    }
}

pub fn parameter_name(i: usize) -> &'static str {
    match i {
        0 => "centre.x",
        1 => "centre.y",
        2 => "log_scale.x",
        3 => "log_scale.y",
        4 => "angle",
        5 => "opacity",
        6 => "colour.r",
        7 => "colour.g",
        8 => "colour.b",
        _ => crate::guard::unreachable(
            "fit::descent::parameter_name",
            &format!("index {i} is outside the {P}-vector"),
        ),
    }
}

pub fn dot(a: &[f64; P], b: &[f64; P]) -> f64 {
    let mut acc = 0.0f64;
    for i in 0..P {
        acc += a[i] * b[i];
    }
    acc
}

pub fn norm(a: &[f64; P]) -> f64 {
    dot(a, a).sqrt()
}

pub fn scaled(a: &[f64; P], s: f64) -> [f64; P] {
    let mut out = [0.0f64; P];
    for i in 0..P {
        out[i] = a[i] * s;
    }
    out
}

pub fn added(a: &[f64; P], b: &[f64; P]) -> [f64; P] {
    let mut out = [0.0f64; P];
    for i in 0..P {
        out[i] = a[i] + b[i];
    }
    out
}

pub fn unit_direction(seed: u64) -> [f64; P] {
    let mut state = seed;
    let mut d = [0.0f64; P];
    for slot in d.iter_mut() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (state >> 11) as f64 / (1u64 << 53) as f64;
        *slot = u * 2.0 - 1.0;
    }
    let n = norm(&d);
    if n > 0.0 { scaled(&d, 1.0 / n) } else { d }
}

pub fn loss_at(
    atoms: &mut [Atom2d],
    index: usize,
    v: &[f64; P],
    ctx: &Context<'_>,
    budget: &mut Budget,
) -> Result<f64> {
    let saved = atoms[index];
    atoms[index] = unpack(v);
    let energy = evaluate(atoms, ctx, budget);
    atoms[index] = saved;
    energy
}

pub fn gradient(
    atoms: &mut [Atom2d],
    index: usize,
    eps: f64,
    ctx: &Context<'_>,
    budget: &mut Budget,
) -> Result<[f64; P]> {
    let x = pack(&atoms[index]);
    let mut g = [0.0f64; P];
    for i in 0..P {
        let mut plus = x;
        plus[i] += eps;
        let mut minus = x;
        minus[i] -= eps;
        let up = loss_at(atoms, index, &plus, ctx, budget)?;
        let down = loss_at(atoms, index, &minus, ctx, budget)?;
        g[i] = (up - down) / (2.0 * eps);
    }
    Ok(g)
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GradientSource {
    CentralDifference,
    Analytic,
}

impl GradientSource {
    pub const ALL: [GradientSource; 2] = [GradientSource::CentralDifference, GradientSource::Analytic];

    pub fn renders_per_gradient(self) -> u64 {
        match self {
            GradientSource::CentralDifference => RENDERS_PER_GRADIENT,
            GradientSource::Analytic => RENDERS_PER_ANALYTIC_GRADIENT,
        }
    }

    pub fn adam_steps(self) -> u32 {
        let per_step = RENDERS_PER_GRADIENT + 1;
        let own = self.renders_per_gradient() + 1;
        (ADAM_STEPS as u64 * per_step / own) as u32
    }

    pub fn name(self) -> &'static str {
        match self {
            GradientSource::CentralDifference => "central-difference",
            GradientSource::Analytic => "analytic",
        }
    }
}

pub fn gradient_from(
    source: GradientSource,
    atoms: &mut [Atom2d],
    index: usize,
    eps: f64,
    ctx: &Context<'_>,
    budget: &mut Budget,
) -> Result<[f64; P]> {
    match source {
        GradientSource::CentralDifference => gradient(atoms, index, eps, ctx, budget),
        GradientSource::Analytic => Ok(analytic_gradient(atoms, index, ctx, budget)?.gradient),
    }
}

pub fn directional_difference(
    atoms: &mut [Atom2d],
    index: usize,
    d: &[f64; P],
    eps: f64,
    ctx: &Context<'_>,
    budget: &mut Budget,
) -> Result<f64> {
    let x = pack(&atoms[index]);
    let plus = added(&x, &scaled(d, eps));
    let minus = added(&x, &scaled(d, -eps));
    let up = loss_at(atoms, index, &plus, ctx, budget)?;
    let down = loss_at(atoms, index, &minus, ctx, budget)?;
    Ok((up - down) / (2.0 * eps))
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SweepRow {
    pub eps: f64,
    pub gradient_norm: f64,
    pub change_from_previous: f64,
    pub identity_error: f64,
    pub gradient: [f64; P],
}

pub fn relative_gap(a: f64, b: f64, scale: f64) -> f64 {
    let floor = f64::EPSILON * scale.max(1.0);
    (a - b).abs() / scale.abs().max(floor)
}

pub fn step_sweep(
    atoms: &mut [Atom2d],
    index: usize,
    steps: &[f64],
    d: &[f64; P],
    ctx: &Context<'_>,
    budget: &mut Budget,
) -> Result<Vec<SweepRow>> {
    let mut rows: Vec<SweepRow> = Vec::with_capacity(steps.len());
    let mut previous: Option<[f64; P]> = None;
    for &eps in steps {
        let g = gradient(atoms, index, eps, ctx, budget)?;
        let along = directional_difference(atoms, index, d, eps, ctx, budget)?;
        let gn = norm(&g);
        let change_from_previous = match previous {
            Some(p) => {
                let mut diff = [0.0f64; P];
                for i in 0..P {
                    diff[i] = g[i] - p[i];
                }
                norm(&diff) / gn.max(f64::EPSILON)
            }
            None => f64::NAN,
        };
        rows.push(SweepRow {
            eps,
            gradient_norm: gn,
            change_from_previous,
            identity_error: relative_gap(along, dot(&g, d), gn),
            gradient: g,
        });
        previous = Some(g);
    }
    Ok(rows)
}

pub fn box_of(ctx: &Context<'_>) -> [(f64, f64); P] {
    let wide = ctx.viewport.0 as f64 * crate::fit::greedy::CENTRE_MARGIN;
    let tall = ctx.viewport.1 as f64 * crate::fit::greedy::CENTRE_MARGIN;
    [
        (-wide, ctx.viewport.0 as f64 + wide),
        (-tall, ctx.viewport.1 as f64 + tall),
        ctx.bounds,
        ctx.bounds,
        (f64::NEG_INFINITY, f64::INFINITY),
        (ctx.floor_opacity, MAX_OPACITY),
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
    ]
}

pub fn projected_descent(g: &[f64; P], x: &[f64; P], bounds: &[(f64, f64); P]) -> [f64; P] {
    let mut d = [0.0f64; P];
    for i in 0..P {
        let downhill = -g[i];
        let at_floor = x[i] <= bounds[i].0 && downhill < 0.0;
        let at_ceiling = x[i] >= bounds[i].1 && downhill > 0.0;
        d[i] = if at_floor || at_ceiling { 0.0 } else { downhill };
    }
    d
}

pub fn clamp_to(v: &[f64; P], bounds: &[(f64, f64); P]) -> [f64; P] {
    let mut out = *v;
    for i in 0..P {
        out[i] = out[i].clamp(bounds[i].0, bounds[i].1);
    }
    out
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LineStep {
    pub length: f64,
    pub loss: f64,
    pub point: [f64; P],
}

pub fn line_search(
    atoms: &mut [Atom2d],
    index: usize,
    direction: &[f64; P],
    halvings: u32,
    bounds: &[(f64, f64); P],
    ctx: &Context<'_>,
    budget: &mut Budget,
) -> Result<Option<LineStep>> {
    let n = norm(direction);
    if !(n > 0.0) {
        return Ok(None);
    }
    let unit = scaled(direction, 1.0 / n);
    let x = pack(&atoms[index]);
    let here = loss_at(atoms, index, &x, ctx, budget)?;
    let mut best: Option<LineStep> = None;
    for level in 0..=halvings {
        let length = 0.5f64.powi(level as i32);
        let point = clamp_to(&added(&x, &scaled(&unit, length)), bounds);
        if !unpack(&point).is_finite() {
            continue;
        }
        let loss = loss_at(atoms, index, &point, ctx, budget)?;
        let better = match best {
            Some(b) => loss < b.loss,
            None => loss < here,
        };
        if better {
            best = Some(LineStep {
                length,
                loss,
                point,
            });
        }
    }
    Ok(best)
}

#[derive(Clone, Debug, PartialEq)]
pub struct DescentTrace {
    pub start: f64,
    pub losses: Vec<f64>,
    pub renders: u64,
}

pub fn descend(
    atoms: &mut [Atom2d],
    index: usize,
    steps: u32,
    eps: f64,
    ctx: &Context<'_>,
    budget: &mut Budget,
) -> Result<DescentTrace> {
    descend_with(GradientSource::CentralDifference, atoms, index, steps, eps, ctx, budget)
}

pub fn descend_with(
    source: GradientSource,
    atoms: &mut [Atom2d],
    index: usize,
    steps: u32,
    eps: f64,
    ctx: &Context<'_>,
    budget: &mut Budget,
) -> Result<DescentTrace> {
    let bounds = box_of(ctx);
    let started = budget.used();
    let x0 = pack(&atoms[index]);
    let start = loss_at(atoms, index, &x0, ctx, budget)?;
    let mut losses: Vec<f64> = Vec::with_capacity(steps as usize);
    for _ in 0..steps {
        let g = gradient_from(source, atoms, index, eps, ctx, budget)?;
        let x = pack(&atoms[index]);
        let direction = projected_descent(&g, &x, &bounds);
        match line_search(atoms, index, &direction, LINE_SEARCH_HALVINGS, &bounds, ctx, budget)? {
            Some(step) => {
                atoms[index] = unpack(&step.point);
                losses.push(step.loss);
            }
            None => break,
        }
    }
    Ok(DescentTrace {
        start,
        losses,
        renders: budget.used() - started,
    })
}

pub const ADAM_BETA1: f64 = 0.9;
pub const ADAM_BETA2: f64 = 0.999;
pub const ADAM_EPSILON: f64 = 1e-8;
pub const ADAM_RATE: f64 = 1.0;
pub const ADAM_STEPS: u32 = 24;
pub const ADAM_DECAY_LEVELS: u32 = 8;
pub const ADAM_RATE_SWEEP: [f64; 6] = [0.05, 0.125, 0.25, 0.5, 1.0, 2.0];
pub const DEFAULT_DESCENT_BUDGET: u64 = 4_000_000;

pub fn fit_budget(limit: u64) -> Budget {
    Budget::new("fit::descent::fit", limit)
}

pub fn parameter_scales(step_scale: f64) -> [f64; P] {
    let centre = centre_step(step_scale);
    [
        centre,
        centre,
        LOG_SCALE_STEP,
        LOG_SCALE_STEP,
        ANGLE_STEP,
        OPACITY_STEP,
        COLOUR_STEP,
        COLOUR_STEP,
        COLOUR_STEP,
    ]
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct AdamState {
    pub first: [f64; P],
    pub second: [f64; P],
    pub steps: u32,
    pub budget: u32,
}

impl AdamState {
    pub fn new(budget: u32) -> Self {
        Self {
            first: [0.0; P],
            second: [0.0; P],
            steps: 0,
            budget,
        }
    }

    pub fn update(&mut self, g: &[f64; P], rate: &[f64; P]) -> [f64; P] {
        self.steps += 1;
        let t = self.steps as i32;
        let bias1 = 1.0 - ADAM_BETA1.powi(t);
        let bias2 = 1.0 - ADAM_BETA2.powi(t);
        let shrink = self.decay();
        let mut delta = [0.0f64; P];
        for i in 0..P {
            self.first[i] = ADAM_BETA1 * self.first[i] + (1.0 - ADAM_BETA1) * g[i];
            self.second[i] = ADAM_BETA2 * self.second[i] + (1.0 - ADAM_BETA2) * g[i] * g[i];
            let m = self.first[i] / bias1;
            let v = self.second[i] / bias2;
            delta[i] = -rate[i] * shrink * m / (v.sqrt() + ADAM_EPSILON);
        }
        delta
    }

    pub fn decay(&self) -> f64 {
        let per_level = (self.budget.max(1) as f64) / (ADAM_DECAY_LEVELS as f64);
        let level = ((self.steps.saturating_sub(1)) as f64 / per_level).floor();
        0.5f64.powf(level.min((ADAM_DECAY_LEVELS - 1) as f64))
    }
}

impl Default for AdamState {
    fn default() -> Self {
        Self::new(ADAM_STEPS)
    }
}

pub fn adam(
    atoms: &mut [Atom2d],
    index: usize,
    steps: u32,
    eps: f64,
    best_in: f64,
    ctx: &Context<'_>,
    budget: &mut Budget,
) -> Result<f64> {
    adam_at_gain(atoms, index, steps, eps, ADAM_RATE, best_in, ctx, budget)
}

pub fn adam_at_gain(
    atoms: &mut [Atom2d],
    index: usize,
    steps: u32,
    eps: f64,
    gain: f64,
    best_in: f64,
    ctx: &Context<'_>,
    budget: &mut Budget,
) -> Result<f64> {
    adam_with(
        GradientSource::CentralDifference,
        atoms,
        index,
        steps,
        eps,
        gain,
        best_in,
        ctx,
        budget,
    )
}

pub fn adam_with(
    source: GradientSource,
    atoms: &mut [Atom2d],
    index: usize,
    steps: u32,
    eps: f64,
    gain: f64,
    best_in: f64,
    ctx: &Context<'_>,
    budget: &mut Budget,
) -> Result<f64> {
    let bounds = box_of(ctx);
    let rate = scaled(&parameter_scales(ctx.step_scale), gain);
    let mut state = AdamState::new(steps);
    let mut best = best_in;
    let mut best_point = pack(&atoms[index]);
    let mut here = best_point;
    for _ in 0..steps {
        let g = gradient_from(source, atoms, index, eps, ctx, budget)?;
        let delta = state.update(&g, &rate);
        here = clamp_to(&added(&here, &delta), &bounds);
        if !unpack(&here).is_finite() {
            break;
        }
        let energy = loss_at(atoms, index, &here, ctx, budget)?;
        if energy < best {
            best = energy;
            best_point = here;
        }
    }
    atoms[index] = unpack(&best_point);
    Ok(best)
}

pub fn fit(
    target: &Target<f64>,
    truth: &FloatImage<f64>,
    request: &FitRequest,
    budget: &mut Budget,
) -> Result<FitOutcome> {
    fit_with(GradientSource::CentralDifference, target, truth, request, budget)
}

pub fn fit_with(
    source: GradientSource,
    target: &Target<f64>,
    truth: &FloatImage<f64>,
    request: &FitRequest,
    budget: &mut Budget,
) -> Result<FitOutcome> {
    let steps_per_atom = source.adam_steps();
    let viewport = target.viewport;
    let ctx = Context::new(truth, request, viewport)?;
    let settings = ctx.settings;

    let mut atoms: Vec<Atom2d> = Vec::new();
    let mut steps: Vec<FitStep> = Vec::new();
    let started = budget.used();
    let mut current = background_image(viewport, settings.background);
    let mut energy = residual_energy(truth, &current)?;
    let mut improvement_exhausted = false;

    for _ in 0..request.max_atoms {
        let residual = truth.difference(&current)?;
        let seed = match seed_pixel(&residual) {
            Some(s) => s,
            None => {
                improvement_exhausted = true;
                break;
            }
        };
        atoms.push(seed_atom_scaled(
            truth,
            &residual,
            seed,
            request.initialisation,
            &settings,
            request.step_scale,
        ));
        let placed = atoms.len() - 1;
        let seeded = evaluate(&atoms, &ctx, budget)?;
        let mut best = adam_with(
            source,
            &mut atoms,
            placed,
            steps_per_atom,
            CHOSEN_STEP,
            ADAM_RATE,
            seeded,
            &ctx,
            budget,
        )?;
        for back in 1..=request.refine_tail {
            let earlier = match placed.checked_sub(back) {
                Some(i) => i,
                None => break,
            };
            best = adam_with(
                source,
                &mut atoms,
                earlier,
                steps_per_atom,
                CHOSEN_STEP,
                ADAM_RATE,
                best,
                &ctx,
                budget,
            )?;
        }
        if !(best < energy) {
            atoms.pop();
            improvement_exhausted = true;
            break;
        }
        budget.tick()?;
        current = render_or_background(&atoms, &ctx.spec, ctx.kernel, &ctx.settings, ctx.tile)?;
        energy = residual_energy(truth, &current)?;
        let residual_now = truth.difference(&current)?;
        steps.push(FitStep {
            atoms: atoms.len(),
            residual_energy: energy,
            residual_peak: residual_peak(&residual_now),
            placed: atoms[placed],
        });
    }

    let scene = to_scene(&atoms, viewport, settings.scale_floor)?;
    Ok(FitOutcome {
        scene,
        atoms,
        steps,
        improvement_exhausted,
        renders: budget.used() - started,
        render_threads: settings.threads,
    })
}
