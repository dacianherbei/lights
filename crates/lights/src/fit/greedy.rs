use crate::agreement::ContractViolation;
use crate::camera::CameraSpec;
use crate::dispatch::{Collect, Precision, RenderOutcome, RenderRequest, render};
use crate::error::{DegenerateReason, LightsError, Result};
use crate::fit::atom2d::{
    Atom2d, ShapedAtom2d, camera, pixel_centre, to_scene, to_shaped_scene,
};
use crate::fit::target::Target;
use crate::guard::Budget;
use crate::image::FloatImage;
use crate::kernel::{CurveShape, KernelFamily};
use crate::method::RenderMethod;
use crate::method::splat::{SortKind, SplatConfig};
use crate::numerics::vector::{Rgb, Vec2};
use crate::scene::Scene;
use crate::settings::SharedSettings;
use crate::tiling::DEFAULT_TILE;

pub const SEED_SIGMA: f64 = 0.7;
pub const ELONGATION_GAIN: f64 = 3.0;
pub const TENSOR_RADIUS: i64 = 3;
pub const LADDER_LEVELS: u32 = 8;
pub const CENTRE_STEP: f64 = 8.0;
pub const LOG_SCALE_STEP: f64 = 1.6;
pub const ANGLE_STEP: f64 = 1.6;
pub const OPACITY_STEP: f64 = 0.8;
pub const COLOUR_STEP: f64 = 0.8;
pub const MIN_LOG_SCALE: f64 = -2.0;
pub const MAX_LOG_SCALE: f64 = 5.0;
pub const MAX_OPACITY: f64 = 1.0;
pub const CENTRE_MARGIN: f64 = 1.0;
pub const DEFAULT_FIT_BUDGET: u64 = 4_000_000;
pub const DEFAULT_REFINE_ITERATIONS: u32 = 6;
pub const DEFAULT_REFINE_TAIL: usize = 3;
pub const UNIT_STEP_SCALE: f64 = 1.0;
pub const LOG_BETA_STEP: f64 = 1.0;
pub const MIN_BETA: f64 = 0.5;
pub const MAX_BETA: f64 = 8.0;

pub fn centre_step(step_scale: f64) -> f64 {
    CENTRE_STEP * step_scale
}

pub fn seed_sigma(step_scale: f64) -> f64 {
    SEED_SIGMA * step_scale
}

pub fn opacity_floor(settings: &SharedSettings<f64>) -> f64 {
    settings.blend.alpha_cutoff * 2.0
}

pub fn opacity_range_is_empty(settings: &SharedSettings<f64>) -> bool {
    !(opacity_floor(settings) <= MAX_OPACITY)
}

pub fn log_scale_bounds(step_scale: f64) -> (f64, f64) {
    let shift = step_scale.ln();
    (MIN_LOG_SCALE + shift, MAX_LOG_SCALE + shift)
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Initialisation {
    StructureTensor,
    Isotropic,
}

impl Initialisation {
    pub const ALL: [Initialisation; 2] =
        [Initialisation::StructureTensor, Initialisation::Isotropic];

    pub fn name(self) -> &'static str {
        match self {
            Initialisation::StructureTensor => "structure-tensor",
            Initialisation::Isotropic => "isotropic",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FitRequest {
    pub kernel: KernelFamily,
    pub settings: SharedSettings<f64>,
    pub max_atoms: usize,
    pub refine_iterations: u32,
    pub refine_tail: usize,
    pub initialisation: Initialisation,
    pub tile: u32,
    pub step_scale: f64,
}

impl FitRequest {
    pub fn new(kernel: KernelFamily, settings: SharedSettings<f64>) -> Self {
        Self {
            kernel,
            settings,
            max_atoms: 8,
            refine_iterations: DEFAULT_REFINE_ITERATIONS,
            refine_tail: DEFAULT_REFINE_TAIL,
            initialisation: Initialisation::StructureTensor,
            tile: DEFAULT_TILE,
            step_scale: UNIT_STEP_SCALE,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FitStep {
    pub atoms: usize,
    pub residual_energy: f64,
    pub residual_peak: f64,
    pub placed: Atom2d,
}

#[derive(Clone, Debug)]
pub struct FitOutcome {
    pub scene: Scene<f64>,
    pub atoms: Vec<Atom2d>,
    pub steps: Vec<FitStep>,
    pub improvement_exhausted: bool,
    pub renders: u64,
    pub render_threads: usize,
}

pub fn fit_budget(limit: u64) -> Budget {
    Budget::new("fit::greedy::fit", limit)
}

pub fn splat_method(tile: u32) -> RenderMethod {
    RenderMethod::SplatRaster(SplatConfig {
        tile,
        sort: SortKind::GlobalSort,
    })
}

fn refuse_tolerance_unless_gaussian(kernel: KernelFamily, collect: Collect) -> Result<()> {
    match collect {
        Collect::Pixels => (),
        Collect::PixelsAndCounters => (),
        Collect::PixelsAndCoverage => (),
        Collect::PixelsAndTolerance => {
            if !kernel.blend_is_over() {
                return Err(LightsError::ContractViolation(
                    ContractViolation::BlendClassNotOver {
                        kernel: kernel.name(),
                    },
                ));
            }
            if !kernel.curve_is_gaussian() {
                return Err(LightsError::ContractViolation(
                    ContractViolation::KernelCurveNotGaussian {
                        kernel: kernel.name(),
                    },
                ));
            }
        }
    }
    Ok(())
}

fn render_scene(
    scene: Scene<f64>,
    spec: &CameraSpec,
    kernel: KernelFamily,
    settings: &SharedSettings<f64>,
    tile: u32,
    collect: Collect,
    shapes: Option<Vec<CurveShape>>,
) -> Result<RenderOutcome> {
    render(&RenderRequest {
        precision: Precision::F64,
        kernel,
        method: splat_method(tile),
        settings: *settings,
        camera: spec.clone(),
        scene,
        collect,
        shapes,
    })
}

pub fn render_atoms(
    atoms: &[Atom2d],
    spec: &CameraSpec,
    kernel: KernelFamily,
    settings: &SharedSettings<f64>,
    tile: u32,
    collect: Collect,
) -> Result<RenderOutcome> {
    refuse_tolerance_unless_gaussian(kernel, collect)?;
    let scene = to_scene(atoms, (spec.width, spec.height), settings.scale_floor)?;
    render_scene(scene, spec, kernel, settings, tile, collect, None)
}

pub fn render_shaped_atoms(
    atoms: &[ShapedAtom2d],
    spec: &CameraSpec,
    kernel: KernelFamily,
    settings: &SharedSettings<f64>,
    tile: u32,
    collect: Collect,
) -> Result<RenderOutcome> {
    refuse_tolerance_unless_gaussian(kernel, collect)?;
    let (scene, shapes) = to_shaped_scene(atoms, (spec.width, spec.height), settings.scale_floor)?;
    match collect {
        Collect::Pixels => (),
        Collect::PixelsAndCounters => (),
        Collect::PixelsAndCoverage => (),
        Collect::PixelsAndTolerance => {
            if !shapes.iter().all(CurveShape::is_gaussian) {
                return Err(LightsError::ContractViolation(
                    ContractViolation::KernelCurveNotGaussian {
                        kernel: kernel.name(),
                    },
                ));
            }
        }
    }
    render_scene(scene, spec, kernel, settings, tile, collect, Some(shapes))
}

pub fn background_image(viewport: (u32, u32), background: Rgb<f64>) -> FloatImage<f64> {
    FloatImage::filled(viewport.0, viewport.1, background)
}

pub fn render_or_background(
    atoms: &[Atom2d],
    spec: &CameraSpec,
    kernel: KernelFamily,
    settings: &SharedSettings<f64>,
    tile: u32,
) -> Result<FloatImage<f64>> {
    if atoms.is_empty() {
        return Ok(background_image(
            (spec.width, spec.height),
            settings.background,
        ));
    }
    let outcome = render_atoms(atoms, spec, kernel, settings, tile, Collect::Pixels)?;
    Ok(outcome.image)
}

pub fn residual_energy(truth: &FloatImage<f64>, rendered: &FloatImage<f64>) -> Result<f64> {
    if truth.width != rendered.width || truth.height != rendered.height {
        return Err(LightsError::SizeMismatch {
            a: (truth.width, truth.height),
            b: (rendered.width, rendered.height),
        });
    }
    let mut acc = 0.0f64;
    for (a, b) in truth.pixels.iter().zip(rendered.pixels.iter()) {
        let p = *a - *b;
        acc += p.x * p.x + p.y * p.y + p.z * p.z;
    }
    Ok(acc)
}

pub fn residual_peak(residual: &FloatImage<f64>) -> f64 {
    let mut best = 0.0f64;
    for p in residual.pixels.iter() {
        let m = (p.x * p.x + p.y * p.y + p.z * p.z).sqrt();
        if m > best {
            best = m;
        }
    }
    best
}

fn magnitude(p: Rgb<f64>) -> f64 {
    (p.x * p.x + p.y * p.y + p.z * p.z).sqrt()
}

fn luminance(p: Rgb<f64>) -> f64 {
    (p.x + p.y + p.z) / 3.0
}

pub fn seed_pixel(residual: &FloatImage<f64>) -> Option<(u32, u32)> {
    let mut best = 0.0f64;
    let mut found: Option<(u32, u32)> = None;
    for py in 0..residual.height {
        for px in 0..residual.width {
            let m = magnitude(residual.get(px, py));
            if m > best {
                best = m;
                found = Some((px, py));
            }
        }
    }
    found
}

fn clamped_luminance(image: &FloatImage<f64>, x: i64, y: i64) -> f64 {
    let cx = x.clamp(0, image.width as i64 - 1) as u32;
    let cy = y.clamp(0, image.height as i64 - 1) as u32;
    luminance(image.get(cx, cy))
}

pub fn structure_tensor(residual: &FloatImage<f64>, seed: (u32, u32)) -> (f64, f64, f64) {
    let (sx, sy) = (seed.0 as i64, seed.1 as i64);
    let mut xx = 0.0f64;
    let mut xy = 0.0f64;
    let mut yy = 0.0f64;
    for dy in -TENSOR_RADIUS..=TENSOR_RADIUS {
        for dx in -TENSOR_RADIUS..=TENSOR_RADIUS {
            let x = sx + dx;
            let y = sy + dy;
            let gx = 0.5
                * (clamped_luminance(residual, x + 1, y) - clamped_luminance(residual, x - 1, y));
            let gy = 0.5
                * (clamped_luminance(residual, x, y + 1) - clamped_luminance(residual, x, y - 1));
            xx += gx * gx;
            xy += gx * gy;
            yy += gy * gy;
        }
    }
    (xx, xy, yy)
}

pub fn tensor_axis(xx: f64, xy: f64, yy: f64) -> (f64, f64) {
    let trace_half = (xx + yy) * 0.5;
    let deviation = (((xx - yy) * 0.5).powi(2) + xy * xy).sqrt();
    let coherence = if trace_half > 0.0 {
        (deviation / trace_half).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let across = 0.5 * (2.0 * xy).atan2(xx - yy);
    (across + core::f64::consts::FRAC_PI_2, coherence)
}

pub fn seed_atom(
    truth: &FloatImage<f64>,
    residual: &FloatImage<f64>,
    seed: (u32, u32),
    initialisation: Initialisation,
    settings: &SharedSettings<f64>,
) -> Atom2d {
    seed_atom_scaled(truth, residual, seed, initialisation, settings, UNIT_STEP_SCALE)
}

pub fn seed_atom_scaled(
    truth: &FloatImage<f64>,
    residual: &FloatImage<f64>,
    seed: (u32, u32),
    initialisation: Initialisation,
    settings: &SharedSettings<f64>,
    step_scale: f64,
) -> Atom2d {
    let sigma = seed_sigma(step_scale);
    let colour = Rgb::new(
        truth.get(seed.0, seed.1).x.clamp(0.0, 1.0),
        truth.get(seed.0, seed.1).y.clamp(0.0, 1.0),
        truth.get(seed.0, seed.1).z.clamp(0.0, 1.0),
    );
    let opacity = MAX_OPACITY.max(opacity_floor(settings)).min(MAX_OPACITY);
    match initialisation {
        Initialisation::Isotropic => Atom2d::from_sigma(
            pixel_centre(seed.0, seed.1),
            Vec2::new(sigma, sigma),
            0.0,
            opacity,
            colour,
        ),
        Initialisation::StructureTensor => {
            let (xx, xy, yy) = structure_tensor(residual, seed);
            let (along, coherence) = tensor_axis(xx, xy, yy);
            let elongation = 1.0 + ELONGATION_GAIN * coherence;
            Atom2d::from_sigma(
                pixel_centre(seed.0, seed.1),
                Vec2::new(sigma * elongation, sigma / elongation),
                along,
                opacity,
                colour,
            )
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Knob {
    CentreX,
    CentreY,
    LogScaleX,
    LogScaleY,
    Angle,
    Opacity,
    ColourR,
    ColourG,
    ColourB,
    LogBeta,
}

pub const KNOBS: [Knob; 9] = [
    Knob::CentreX,
    Knob::CentreY,
    Knob::LogScaleX,
    Knob::LogScaleY,
    Knob::Angle,
    Knob::Opacity,
    Knob::ColourR,
    Knob::ColourG,
    Knob::ColourB,
];

pub const SHAPED_KNOBS: [Knob; 10] = [
    Knob::CentreX,
    Knob::CentreY,
    Knob::LogScaleX,
    Knob::LogScaleY,
    Knob::Angle,
    Knob::Opacity,
    Knob::ColourR,
    Knob::ColourG,
    Knob::ColourB,
    Knob::LogBeta,
];

fn base_step(knob: Knob, step_scale: f64) -> f64 {
    match knob {
        Knob::CentreX => centre_step(step_scale),
        Knob::CentreY => centre_step(step_scale),
        Knob::LogScaleX => LOG_SCALE_STEP,
        Knob::LogScaleY => LOG_SCALE_STEP,
        Knob::Angle => ANGLE_STEP,
        Knob::Opacity => OPACITY_STEP,
        Knob::ColourR => COLOUR_STEP,
        Knob::ColourG => COLOUR_STEP,
        Knob::ColourB => COLOUR_STEP,
        Knob::LogBeta => LOG_BETA_STEP,
    }
}

fn wrap_angle(a: f64) -> f64 {
    let tau = core::f64::consts::TAU;
    let shifted = (a + core::f64::consts::PI).rem_euclid(tau);
    shifted - core::f64::consts::PI
}

fn nudge(
    a: &Atom2d,
    knob: Knob,
    delta: f64,
    viewport: (u32, u32),
    floor_opacity: f64,
    bounds: (f64, f64),
) -> Atom2d {
    let mut out = *a;
    let wide = viewport.0 as f64 * CENTRE_MARGIN;
    let tall = viewport.1 as f64 * CENTRE_MARGIN;
    match knob {
        Knob::CentreX => {
            out.centre.x = (a.centre.x + delta).clamp(-wide, viewport.0 as f64 + wide)
        }
        Knob::CentreY => {
            out.centre.y = (a.centre.y + delta).clamp(-tall, viewport.1 as f64 + tall)
        }
        Knob::LogScaleX => {
            out.log_scale.x = (a.log_scale.x + delta).clamp(bounds.0, bounds.1)
        }
        Knob::LogScaleY => {
            out.log_scale.y = (a.log_scale.y + delta).clamp(bounds.0, bounds.1)
        }
        Knob::Angle => out.angle = wrap_angle(a.angle + delta),
        Knob::Opacity => {
            if floor_opacity > MAX_OPACITY {
                crate::guard::unreachable(
                    "fit::greedy::nudge",
                    &format!(
                        "opacity floor {floor_opacity} above MAX_OPACITY {MAX_OPACITY}: \
                         fit() rejects this before a Context is built"
                    ),
                );
            }
            out.opacity = (a.opacity + delta).clamp(floor_opacity, MAX_OPACITY)
        }
        Knob::ColourR => out.colour.x = (a.colour.x + delta).clamp(0.0, 1.0),
        Knob::ColourG => out.colour.y = (a.colour.y + delta).clamp(0.0, 1.0),
        Knob::ColourB => out.colour.z = (a.colour.z + delta).clamp(0.0, 1.0),
        Knob::LogBeta => crate::guard::unreachable(
            "fit::greedy::nudge",
            "an Atom2d carries no curve; LogBeta is a ShapedAtom2d knob",
        ),
    }
    out
}

pub fn nudge_beta(beta: f64, delta: f64) -> f64 {
    (beta.ln() + delta).exp().clamp(MIN_BETA, MAX_BETA)
}

pub trait Tunable: Copy {
    const KNOBS: &'static [Knob];
    fn seeded(atom: Atom2d, kernel: KernelFamily) -> Self;
    fn base(&self) -> Atom2d;
    fn nudged(&self, knob: Knob, delta: f64, ctx: &Context<'_>) -> Self;
    fn render_all(atoms: &[Self], ctx: &Context<'_>) -> Result<FloatImage<f64>>;
}

impl Tunable for Atom2d {
    const KNOBS: &'static [Knob] = &KNOBS;

    fn seeded(atom: Atom2d, _kernel: KernelFamily) -> Self {
        atom
    }

    fn base(&self) -> Atom2d {
        *self
    }

    fn nudged(&self, knob: Knob, delta: f64, ctx: &Context<'_>) -> Self {
        nudge(self, knob, delta, ctx.viewport, ctx.floor_opacity, ctx.bounds)
    }

    fn render_all(atoms: &[Self], ctx: &Context<'_>) -> Result<FloatImage<f64>> {
        render_or_background(atoms, &ctx.spec, ctx.kernel, &ctx.settings, ctx.tile)
    }
}

impl Tunable for ShapedAtom2d {
    const KNOBS: &'static [Knob] = &SHAPED_KNOBS;

    fn seeded(atom: Atom2d, kernel: KernelFamily) -> Self {
        ShapedAtom2d::new(atom, kernel.shape())
    }

    fn base(&self) -> Atom2d {
        self.atom
    }

    fn nudged(&self, knob: Knob, delta: f64, ctx: &Context<'_>) -> Self {
        match knob {
            Knob::LogBeta => {
                let mut shape = self.shape;
                shape.beta = nudge_beta(shape.beta, delta);
                ShapedAtom2d::new(self.atom, shape)
            }
            other => ShapedAtom2d::new(
                nudge(&self.atom, other, delta, ctx.viewport, ctx.floor_opacity, ctx.bounds),
                self.shape,
            ),
        }
    }

    fn render_all(atoms: &[Self], ctx: &Context<'_>) -> Result<FloatImage<f64>> {
        if atoms.is_empty() {
            return Ok(background_image(ctx.viewport, ctx.settings.background));
        }
        let outcome = render_shaped_atoms(
            atoms,
            &ctx.spec,
            ctx.kernel,
            &ctx.settings,
            ctx.tile,
            Collect::Pixels,
        )?;
        Ok(outcome.image)
    }
}

pub struct Context<'a> {
    pub truth: &'a FloatImage<f64>,
    pub spec: CameraSpec,
    pub kernel: KernelFamily,
    pub settings: SharedSettings<f64>,
    pub tile: u32,
    pub viewport: (u32, u32),
    pub floor_opacity: f64,
    pub step_scale: f64,
    pub bounds: (f64, f64),
}

impl<'a> Context<'a> {
    pub fn new(
        truth: &'a FloatImage<f64>,
        request: &FitRequest,
        viewport: (u32, u32),
    ) -> Result<Self> {
        if truth.width != viewport.0 || truth.height != viewport.1 {
            return Err(LightsError::SizeMismatch {
                a: (truth.width, truth.height),
                b: viewport,
            });
        }
        let mut settings = request.settings;
        settings.threads = 1;
        if opacity_range_is_empty(&settings) {
            return Err(LightsError::DegenerateAtom {
                index: 0,
                reason: DegenerateReason::OpacityBelowCutoff,
            });
        }
        Ok(Self {
            truth,
            spec: camera(viewport.0, viewport.1),
            kernel: request.kernel,
            settings,
            tile: request.tile,
            viewport,
            floor_opacity: opacity_floor(&settings),
            step_scale: request.step_scale,
            bounds: log_scale_bounds(request.step_scale),
        })
    }
}

pub fn evaluate<A: Tunable>(atoms: &[A], ctx: &Context<'_>, budget: &mut Budget) -> Result<f64> {
    budget.tick()?;
    let image = A::render_all(atoms, ctx)?;
    residual_energy(ctx.truth, &image)
}

pub fn refine<A: Tunable>(
    atoms: &mut [A],
    index: usize,
    iterations: u32,
    best_in: f64,
    ctx: &Context<'_>,
    budget: &mut Budget,
) -> Result<f64> {
    let mut best = best_in;
    for _ in 0..iterations {
        for &knob in A::KNOBS {
            let base = base_step(knob, ctx.step_scale);
            for level in 0..LADDER_LEVELS {
                let delta = base * 0.5f64.powi(level as i32);
                let mut chosen: Option<A> = None;
                for sign in [1.0f64, -1.0f64] {
                    let trial = atoms[index].nudged(knob, sign * delta, ctx);
                    if !trial.base().is_finite() {
                        continue;
                    }
                    let saved = atoms[index];
                    atoms[index] = trial;
                    let energy = evaluate(atoms, ctx, budget)?;
                    atoms[index] = saved;
                    if energy < best {
                        best = energy;
                        chosen = Some(trial);
                    }
                }
                match chosen {
                    Some(t) => atoms[index] = t,
                    None => (),
                }
            }
        }
    }
    Ok(best)
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Reading {
    Prefix,
    OwnFit,
}

impl Reading {
    pub fn of(n: usize, placed: usize) -> Reading {
        if n < placed { Reading::Prefix } else { Reading::OwnFit }
    }

    pub fn name(self) -> &'static str {
        match self {
            Reading::Prefix => "prefix",
            Reading::OwnFit => "own-fit",
        }
    }
}

pub const PREFIX_SHORTFALL: &str = "a `prefix` row renders atoms[..n] of a longer fit whose \
     refine_tail re-tuned them for atoms the row discards; it is NOT the n-atom fit. Measured \
     2026-09-16 over six classes at five sub-full rungs: 26 of 30 cells worse than the rung's \
     own fit, mean +23.6%, worst 2.83x residual energy (t2-straight-edge, n=1); 4 of 30 better \
     by 0.9%-6.9%, all at high n. A prefix curve is pessimistic on average, not a strict bound. \
     See RESEARCH.json, experiment l142-prefix-shortfall.";

pub fn prefix<A>(atoms: &[A], n: usize) -> (&[A], Reading) {
    (&atoms[..n], Reading::of(n, atoms.len()))
}

struct Pursuit<A> {
    atoms: Vec<A>,
    steps: Vec<FitStep>,
    improvement_exhausted: bool,
    renders: u64,
}

fn pursue<A: Tunable>(
    truth: &FloatImage<f64>,
    request: &FitRequest,
    ctx: &Context<'_>,
    budget: &mut Budget,
) -> Result<Pursuit<A>> {
    let viewport = ctx.viewport;
    let settings = ctx.settings;

    let mut atoms: Vec<A> = Vec::new();
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
        atoms.push(A::seeded(
            seed_atom_scaled(
                truth,
                &residual,
                seed,
                request.initialisation,
                &settings,
                request.step_scale,
            ),
            ctx.kernel,
        ));
        let placed = atoms.len() - 1;
        let seeded = evaluate(&atoms, ctx, budget)?;
        let mut best = refine(
            &mut atoms,
            placed,
            request.refine_iterations,
            seeded,
            ctx,
            budget,
        )?;
        for back in 1..=request.refine_tail {
            let earlier = match placed.checked_sub(back) {
                Some(i) => i,
                None => break,
            };
            best = refine(&mut atoms, earlier, 1, best, ctx, budget)?;
        }
        if !(best < energy) {
            atoms.pop();
            improvement_exhausted = true;
            break;
        }
        budget.tick()?;
        current = A::render_all(&atoms, ctx)?;
        energy = residual_energy(truth, &current)?;
        let residual_now = truth.difference(&current)?;
        steps.push(FitStep {
            atoms: atoms.len(),
            residual_energy: energy,
            residual_peak: residual_peak(&residual_now),
            placed: atoms[placed].base(),
        });
    }

    Ok(Pursuit {
        atoms,
        steps,
        improvement_exhausted,
        renders: budget.used() - started,
    })
}

pub fn fit(
    target: &Target<f64>,
    truth: &FloatImage<f64>,
    request: &FitRequest,
    budget: &mut Budget,
) -> Result<FitOutcome> {
    let viewport = target.viewport;
    let ctx = Context::new(truth, request, viewport)?;
    let run: Pursuit<Atom2d> = pursue(truth, request, &ctx, budget)?;
    let scene = to_scene(&run.atoms, viewport, ctx.settings.scale_floor)?;
    Ok(FitOutcome {
        scene,
        atoms: run.atoms,
        steps: run.steps,
        improvement_exhausted: run.improvement_exhausted,
        renders: run.renders,
        render_threads: ctx.settings.threads,
    })
}

#[derive(Clone, Debug)]
pub struct ShapedFitOutcome {
    pub atoms: Vec<ShapedAtom2d>,
    pub steps: Vec<FitStep>,
    pub improvement_exhausted: bool,
    pub renders: u64,
}

pub fn fit_shaped(
    target: &Target<f64>,
    truth: &FloatImage<f64>,
    request: &FitRequest,
    budget: &mut Budget,
) -> Result<ShapedFitOutcome> {
    let ctx = Context::new(truth, request, target.viewport)?;
    let run: Pursuit<ShapedAtom2d> = pursue(truth, request, &ctx, budget)?;
    Ok(ShapedFitOutcome {
        atoms: run.atoms,
        steps: run.steps,
        improvement_exhausted: run.improvement_exhausted,
        renders: run.renders,
    })
}
