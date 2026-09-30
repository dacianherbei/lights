use crate::error::{LightsError, Result};
use crate::fit::analytic::sdf_coverage;
use crate::fit::atom2d::{Atom2d, camera};
use crate::fit::coverage::{coverage_budget, coverage_image_to_colour};
use crate::fit::edge::Criterion;
use crate::fit::greedy::{
    FitRequest, Initialisation, UNIT_STEP_SCALE, fit, fit_budget, render_or_background,
};
use crate::fit::harness::ProfileChoice;
pub use crate::fit::nterm::ATOM_SCALARS_NATIVE;
use crate::fit::nterm::{
    BITS_PER_BYTE, VALUE_BYTES, index_bits, max_abs_error, relative_l2,
};
use crate::fit::sweep::{self, Concurrency};
use crate::fit::scale::{
    Convention, Judgement, Scale, ScaleClass, judge_fwhm, judge_transition, measure_scene,
    scaled_atoms, scaled_target,
};
use crate::fit::target::Target;
use crate::fit::transform::{
    Ranking, Transform, expressible, forward, inverse, keep_largest, max_levels, weight,
};
use crate::image::FloatImage;
use crate::kernel::KernelFamily;
use crate::numerics::vector::Rgb;
use crate::settings::SharedSettings;
use crate::tiling::DEFAULT_TILE;

pub const MAGNIFY_LADDER: [(u32, u32); 4] = [(1, 1), (2, 1), (4, 1), (8, 1)];
pub const MINIFY_LADDER: [(u32, u32); 6] = [(8, 1), (4, 1), (2, 1), (1, 1), (1, 2), (1, 4)];
pub const SYMMETRIC_VALUE_BYTES: f64 = VALUE_BYTES;
pub const DEFAULT_WAVELET_TERMS: usize = 64;
pub const IDENTICAL_TOLERANCE: f64 = 0.0;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Direction {
    Magnify,
    Minify,
}

impl Direction {
    pub const ALL: [Direction; 2] = [Direction::Magnify, Direction::Minify];

    pub fn name(self) -> &'static str {
        match self {
            Direction::Magnify => "magnify",
            Direction::Minify => "minify",
        }
    }

    pub fn rungs(self) -> Vec<(u32, u32)> {
        match self {
            Direction::Magnify => MAGNIFY_LADDER.to_vec(),
            Direction::Minify => MINIFY_LADDER.to_vec(),
        }
    }

    pub fn ladder(self) -> Result<Vec<Scale>> {
        let mut out = Vec::new();
        for (n, d) in self.rungs() {
            out.push(Scale::new(n, d)?);
        }
        Ok(out)
    }

    pub fn base(self) -> Result<Scale> {
        match self.ladder()?.first() {
            Some(s) => Ok(*s),
            None => Err(LightsError::ViewportZero {
                width: 0,
                height: 0,
            }),
        }
    }

    pub fn legend(self) -> &'static str {
        match self {
            Direction::Magnify => {
                "fit at 1x and render outwards: this asks whether the description carries \
                 detail it was never asked to represent"
            }
            Direction::Minify => {
                "fit at 8x and render inwards: this asks about ALIASING, and it is where the \
                 missing pixel prefilter bites, which is why both profiles are run"
            }
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Independence {
    ContinuousFrozen,
    ContinuousRefit,
    ByConstruction,
    LatticeBound,
}

impl Independence {
    pub fn name(self) -> &'static str {
        match self {
            Independence::ContinuousFrozen => "continuous-frozen",
            Independence::ContinuousRefit => "continuous-refit",
            Independence::ByConstruction => "by-construction",
            Independence::LatticeBound => "lattice-bound",
        }
    }

    pub fn legend(self) -> &'static str {
        match self {
            Independence::ContinuousFrozen => {
                "one parametric field, frozen at the base scale and re-sampled at every other: \
                 this is the arm on trial"
            }
            Independence::ContinuousRefit => {
                "the same family fitted afresh at every scale: the honest control for the \
                 frozen arm, and a greedy FLOOR on atom quality rather than a bound above it. \
                 The gap between them IS the cost of resolution independence, and it is a \
                 MINIMUM statement: the frozen set beats the fresh fit in 30 of the 80 cells \
                 of the published p5 ladders, which a bound above could not permit"
            }
            Independence::ByConstruction => {
                "a signed distance evaluated at the device rate: resolution independent BY \
                 CONSTRUCTION and therefore the bar, not a competitor"
            }
            Independence::LatticeBound => {
                "a coefficient set on a fixed dyadic lattice: it does NOT resample for free, \
                 and a loss here is STRUCTURAL rather than a defect. It is the difference \
                 between a continuous description and a sampled one, and it is included \
                 precisely to be measured"
            }
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Arm3 {
    Frozen(KernelFamily),
    Refit(KernelFamily),
    AnalyticSdf,
    Wavelet(Transform),
}

impl Arm3 {
    pub fn name(self) -> String {
        match self {
            Arm3::Frozen(k) => format!("frozen-atoms-{}", k.name()),
            Arm3::Refit(k) => format!("refitted-atoms-{}", k.name()),
            Arm3::AnalyticSdf => "analytic-sdf".to_string(),
            Arm3::Wavelet(t) => format!("wavelet-{}", t.name()),
        }
    }

    pub fn independence(self) -> Independence {
        match self {
            Arm3::Frozen(_) => Independence::ContinuousFrozen,
            Arm3::Refit(_) => Independence::ContinuousRefit,
            Arm3::AnalyticSdf => Independence::ByConstruction,
            Arm3::Wavelet(_) => Independence::LatticeBound,
        }
    }

    pub fn honours_profile(self) -> bool {
        match self {
            Arm3::Frozen(_) => true,
            Arm3::Refit(_) => true,
            Arm3::AnalyticSdf => false,
            Arm3::Wavelet(_) => false,
        }
    }

    pub fn refits_per_scale(self) -> bool {
        match self {
            Arm3::Frozen(_) => false,
            Arm3::Refit(_) => true,
            Arm3::AnalyticSdf => false,
            Arm3::Wavelet(_) => false,
        }
    }
}

pub fn arms(kernel: KernelFamily) -> Vec<Arm3> {
    vec![
        Arm3::Frozen(kernel),
        Arm3::Refit(kernel),
        Arm3::AnalyticSdf,
        Arm3::Wavelet(Transform::Haar),
        Arm3::Wavelet(Transform::Cdf97),
    ]
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Resample {
    Bilinear,
    PixelBoxArea,
}

impl Resample {
    pub fn for_ratio(ratio: f64) -> Self {
        if ratio >= 1.0 {
            Resample::Bilinear
        } else {
            Resample::PixelBoxArea
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Resample::Bilinear => "bilinear",
            Resample::PixelBoxArea => "pixel-box-area",
        }
    }

    pub fn legend(self) -> &'static str {
        match self {
            Resample::Bilinear => {
                "magnifying a lattice: bilinear is the textbook choice and is given to the \
                 lattice arm deliberately, so its loss is a floor on its loss and not a \
                 strawman"
            }
            Resample::PixelBoxArea => {
                "minifying a lattice: area averaging is the correct minifier and is the \
                 lattice arm's own prefilter, so its loss here is not the missing atom \
                 prefilter wearing another hat"
            }
        }
    }
}

fn sample_bilinear(image: &FloatImage<f64>, u: f64, v: f64) -> Rgb<f64> {
    let last_x = (image.width - 1) as f64;
    let last_y = (image.height - 1) as f64;
    let cu = u.clamp(0.0, last_x);
    let cv = v.clamp(0.0, last_y);
    let x0 = cu.floor();
    let y0 = cv.floor();
    let fx = cu - x0;
    let fy = cv - y0;
    let ix = x0 as u32;
    let iy = y0 as u32;
    let jx = (ix + 1).min(image.width - 1);
    let jy = (iy + 1).min(image.height - 1);
    let a = image.get(ix, iy);
    let b = image.get(jx, iy);
    let c = image.get(ix, jy);
    let d = image.get(jx, jy);
    let top = a * (1.0 - fx) + b * fx;
    let bottom = c * (1.0 - fx) + d * fx;
    top * (1.0 - fy) + bottom * fy
}

fn axis_overlap(destination: u32, count: u32, source: u32) -> Vec<(u32, f64)> {
    let step = (source as f64) / (count as f64);
    let lo = (destination as f64) * step;
    let hi = lo + step;
    let first = lo.floor() as i64;
    let last = (hi.ceil() as i64 - 1).max(first);
    let mut out = Vec::new();
    for index in first..=last {
        let clamped = index.clamp(0, source as i64 - 1) as u32;
        let cell_lo = (clamped as f64).max(lo);
        let cell_hi = ((clamped + 1) as f64).min(hi);
        let width = cell_hi - cell_lo;
        if width > 0.0 {
            out.push((clamped, width / step));
        }
    }
    if out.is_empty() {
        out.push((0, 1.0));
    }
    out
}

pub fn resample(
    image: &FloatImage<f64>,
    viewport: (u32, u32),
    rule: Resample,
) -> Result<FloatImage<f64>> {
    if viewport.0 == 0 || viewport.1 == 0 || image.width == 0 || image.height == 0 {
        return Err(LightsError::ViewportZero {
            width: viewport.0,
            height: viewport.1,
        });
    }
    let mut out = FloatImage::<f64>::new(viewport.0, viewport.1);
    match rule {
        Resample::Bilinear => {
            let sx = (image.width as f64) / (viewport.0 as f64);
            let sy = (image.height as f64) / (viewport.1 as f64);
            for y in 0..viewport.1 {
                for x in 0..viewport.0 {
                    let u = ((x as f64) + 0.5) * sx - 0.5;
                    let v = ((y as f64) + 0.5) * sy - 0.5;
                    out.set(x, y, sample_bilinear(image, u, v));
                }
            }
        }
        Resample::PixelBoxArea => {
            let columns: Vec<Vec<(u32, f64)>> = (0..viewport.0)
                .map(|x| axis_overlap(x, viewport.0, image.width))
                .collect();
            let rows: Vec<Vec<(u32, f64)>> = (0..viewport.1)
                .map(|y| axis_overlap(y, viewport.1, image.height))
                .collect();
            for y in 0..viewport.1 {
                for x in 0..viewport.0 {
                    let mut acc = Rgb::zero();
                    for (sy, wy) in rows[y as usize].iter() {
                        for (sx, wx) in columns[x as usize].iter() {
                            acc = acc + image.get(*sx, *sy) * (*wx * *wy);
                        }
                    }
                    out.set(x, y, acc);
                }
            }
        }
    }
    Ok(out)
}

pub fn atom_bytes_symmetric(terms: usize) -> f64 {
    (terms as f64) * (ATOM_SCALARS_NATIVE as f64) * SYMMETRIC_VALUE_BYTES
}

pub fn coefficient_bytes_symmetric(terms: usize, width: u32, height: u32) -> f64 {
    (terms as f64) * (SYMMETRIC_VALUE_BYTES + index_bits(width, height) / BITS_PER_BYTE)
}

pub fn analytic_bytes_symmetric(target: &Target<f64>) -> f64 {
    target.description_bytes_at(SYMMETRIC_VALUE_BYTES as usize) as f64
}

#[derive(Clone, Debug, PartialEq)]
pub struct Aspect3Request {
    pub direction: Direction,
    pub ladder: Vec<Scale>,
    pub kernel: KernelFamily,
    pub profiles: Vec<ProfileChoice>,
    pub max_atoms: usize,
    pub tile: u32,
    pub refine_iterations: u32,
    pub refine_tail: usize,
    pub initialisation: Initialisation,
    pub render_limit: u64,
    pub quadrature_depth: u32,
    pub cell_limit: u64,
    pub wavelet_terms: usize,
    pub ranking: Ranking,
    pub scene_invariant_search: bool,
    pub criterion: Criterion,
    pub concurrency: Concurrency,
}

impl Aspect3Request {
    pub fn new(direction: Direction) -> Result<Self> {
        Ok(Self {
            direction,
            ladder: direction.ladder()?,
            kernel: KernelFamily::Gaussian,
            profiles: vec![ProfileChoice::Contract, ProfileChoice::Prefiltered],
            max_atoms: 8,
            tile: DEFAULT_TILE,
            refine_iterations: 6,
            refine_tail: 3,
            initialisation: Initialisation::StructureTensor,
            render_limit: 40_000_000,
            quadrature_depth: 10,
            cell_limit: 4_000_000_000,
            wavelet_terms: DEFAULT_WAVELET_TERMS,
            ranking: Ranking::Weighted,
            scene_invariant_search: true,
            criterion: Criterion::registered(),
            concurrency: Concurrency::Parallel,
        })
    }

    pub fn base(&self) -> Result<Scale> {
        match self.ladder.first() {
            Some(s) => Ok(*s),
            None => Err(LightsError::InvalidCamera(
                "an empty scale ladder has no base scale to fit at",
            )),
        }
    }

    pub fn step_scale(&self, scale: Scale) -> f64 {
        if self.scene_invariant_search {
            scale.factor()
        } else {
            UNIT_STEP_SCALE
        }
    }

    pub fn search_convention(&self) -> &'static str {
        if self.scene_invariant_search {
            "scene-invariant"
        } else {
            "device-pixel"
        }
    }
}

#[derive(Clone, Debug)]
pub struct FittedAtoms {
    pub atoms: Vec<Atom2d>,
    pub improvement_exhausted: bool,
    pub renders: u64,
}

pub fn fit_request_for(
    scale: Scale,
    settings: SharedSettings<f64>,
    request: &Aspect3Request,
) -> FitRequest {
    let mut fit_request = FitRequest::new(request.kernel, settings);
    fit_request.max_atoms = request.max_atoms;
    fit_request.tile = request.tile;
    fit_request.refine_iterations = request.refine_iterations;
    fit_request.refine_tail = request.refine_tail;
    fit_request.initialisation = request.initialisation;
    fit_request.step_scale = request.step_scale(scale);
    fit_request
}

pub fn fit_at(
    class: &ScaleClass,
    truth: &FloatImage<f64>,
    scale: Scale,
    settings: SharedSettings<f64>,
    request: &Aspect3Request,
) -> Result<FittedAtoms> {
    let scaffold = class.scaffold(scale)?;
    let fit_request = fit_request_for(scale, settings, request);
    let mut budget = fit_budget(request.render_limit);
    let outcome = fit(&scaffold, truth, &fit_request, &mut budget)?;
    Ok(FittedAtoms {
        atoms: outcome.atoms,
        improvement_exhausted: outcome.improvement_exhausted,
        renders: outcome.renders,
    })
}

pub fn render_atoms_at(
    atoms: &[Atom2d],
    viewport: (u32, u32),
    kernel: KernelFamily,
    settings: &SharedSettings<f64>,
    tile: u32,
) -> Result<FloatImage<f64>> {
    render_or_background(atoms, &camera(viewport.0, viewport.1), kernel, settings, tile)
}

pub fn analytic_at(class: &ScaleClass, scale: Scale) -> Result<Option<FloatImage<f64>>> {
    match class.analytic() {
        None => Ok(None),
        Some(target) => {
            let big = scaled_target(target, scale)?;
            Ok(Some(coverage_image_to_colour(&big, &sdf_coverage(&big))))
        }
    }
}

#[derive(Clone, Debug)]
pub struct WaveletBase {
    pub transform: Transform,
    pub terms: usize,
    pub available: usize,
    pub levels: u32,
    pub base_viewport: (u32, u32),
    pub reconstruction: FloatImage<f64>,
}

pub fn prepare_wavelet(
    truth: &FloatImage<f64>,
    transform: Transform,
    ranking: Ranking,
    terms: usize,
) -> Result<Option<WaveletBase>> {
    if !expressible(truth.width, truth.height) {
        return Ok(None);
    }
    let levels = max_levels(truth.width, truth.height);
    let coefficients = forward(transform, truth, levels)?;
    let weights = weight::measure(transform, truth.width, truth.height, levels)?;
    let available = coefficients.terms();
    let kept = keep_largest(&coefficients, &weights, ranking, terms.min(available))?;
    Ok(Some(WaveletBase {
        transform,
        terms: terms.min(available),
        available,
        levels,
        base_viewport: (truth.width, truth.height),
        reconstruction: inverse(&kept)?,
    }))
}

pub fn wavelet_at(base: &WaveletBase, viewport: (u32, u32), ratio: f64) -> Result<FloatImage<f64>> {
    resample(&base.reconstruction, viewport, Resample::for_ratio(ratio))
}

pub fn boundary_localised_reference(base_viewport: (u32, u32), viewport: (u32, u32)) -> f64 {
    let base_side = ((base_viewport.0 as f64) * (base_viewport.1 as f64)).sqrt();
    let side = ((viewport.0 as f64) * (viewport.1 as f64)).sqrt();
    if side <= 0.0 || base_side <= 0.0 {
        return f64::NAN;
    }
    (base_side / side).sqrt()
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScaleRow {
    pub class: String,
    pub direction: &'static str,
    pub profile: String,
    pub arm: String,
    pub independence: &'static str,
    pub scale: Scale,
    pub base_scale: Scale,
    pub viewport: (u32, u32),
    pub at_base: bool,
    pub absent: bool,
    pub terms: usize,
    pub description_bytes: f64,
    pub relative_l2: f64,
    pub max_abs: f64,
    pub decay: f64,
    pub boundary_reference: f64,
    pub quality_measured: bool,
    pub transition: Judgement,
    pub fwhm: Judgement,
    pub exhausted: Option<bool>,
    pub fit_renders: u64,
    pub render_nanos: u128,
    pub agrees_with_frozen_at_base: Option<bool>,
    pub resample: &'static str,
}

#[derive(Clone, Debug)]
pub struct ClassLadder {
    pub class: String,
    pub direction: &'static str,
    pub base_scale: Scale,
    pub base_viewport: (u32, u32),
    pub analytic_present: bool,
    pub rows: Vec<ScaleRow>,
    pub truths: Vec<(Scale, FloatImage<f64>)>,
    pub renders: Vec<(String, Scale, String, FloatImage<f64>)>,
    pub frozen: Vec<(String, Vec<Atom2d>)>,
}

impl ClassLadder {
    pub fn render_of(&self, arm: &str, scale: Scale, profile: &str) -> Option<&FloatImage<f64>> {
        self.renders
            .iter()
            .find(|(a, s, p, _)| a == arm && *s == scale && p == profile)
            .map(|(_, _, _, image)| image)
    }

    pub fn truth_of(&self, scale: Scale) -> Option<&FloatImage<f64>> {
        self.truths
            .iter()
            .find(|(s, _)| *s == scale)
            .map(|(_, image)| image)
    }
}

fn blank() -> Judgement {
    Judgement {
        scene: f64::NAN,
        device: f64::NAN,
        meets_scene: false,
        meets_device: false,
    }
}

pub fn pixelwise_identical(a: &FloatImage<f64>, b: &FloatImage<f64>) -> bool {
    a.width == b.width
        && a.height == b.height
        && a.pixels.iter().zip(b.pixels.iter()).all(|(p, q)| {
            p.x.to_bits() == q.x.to_bits()
                && p.y.to_bits() == q.y.to_bits()
                && p.z.to_bits() == q.z.to_bits()
        })
}

fn judgements(
    class: &ScaleClass,
    image: &FloatImage<f64>,
    scale: Scale,
    criterion: &Criterion,
) -> Result<(bool, Judgement, Judgement)> {
    match class.analytic() {
        None => Ok((false, blank(), blank())),
        Some(target) => {
            if target.probes.is_empty() {
                return Ok((false, blank(), blank()));
            }
            let metric = measure_scene(image, target, scale, criterion)?;
            Ok((
                true,
                judge_transition(&metric, scale, criterion),
                judge_fwhm(&metric, scale, criterion),
            ))
        }
    }
}

fn leading_profile(request: &Aspect3Request) -> Result<ProfileChoice> {
    match request.profiles.first() {
        Some(p) => Ok(*p),
        None => Err(LightsError::InvalidCamera(
            "aspect 3 needs at least one settings profile: the prefilter is an axis of this \
             run, not a default",
        )),
    }
}

struct Prepared {
    base: Scale,
    base_viewport: (u32, u32),
    truths: Vec<(Scale, FloatImage<f64>)>,
    base_truth: FloatImage<f64>,
    wavelets: Vec<(Transform, Option<WaveletBase>)>,
}

fn prepare(class: &ScaleClass, request: &Aspect3Request) -> Result<Prepared> {
    let base = request.base()?;
    let base_viewport = class.viewport_at(base)?;
    let mut cells = coverage_budget(request.cell_limit);

    let mut truths: Vec<(Scale, FloatImage<f64>)> = Vec::new();
    for scale in request.ladder.iter() {
        truths.push((*scale, class.truth(*scale, request.quadrature_depth, &mut cells)?));
    }
    let base_truth = match truths.iter().find(|(s, _)| *s == base) {
        Some((_, t)) => t.clone(),
        None => {
            return Err(LightsError::ViewportZero {
                width: base_viewport.0,
                height: base_viewport.1,
            });
        }
    };

    let mut wavelets = Vec::new();
    for transform in [Transform::Haar, Transform::Cdf97] {
        wavelets.push((
            transform,
            prepare_wavelet(&base_truth, transform, request.ranking, request.wavelet_terms)?,
        ));
    }
    Ok(Prepared {
        base,
        base_viewport,
        truths,
        base_truth,
        wavelets,
    })
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct FitJob {
    pub class: usize,
    pub profile: usize,
    pub rung: Option<usize>,
}

fn fits_per_profile(request: &Aspect3Request) -> usize {
    let refits = arms(request.kernel).iter().any(|a| a.refits_per_scale());
    if refits { 1 + request.ladder.len() } else { 1 }
}

pub fn fit_jobs(class_count: usize, request: &Aspect3Request) -> Vec<FitJob> {
    let per_profile = fits_per_profile(request);
    let mut jobs = Vec::with_capacity(class_count * request.profiles.len() * per_profile);
    for class in 0..class_count {
        for profile in 0..request.profiles.len() {
            jobs.push(FitJob {
                class,
                profile,
                rung: None,
            });
            for rung in 0..per_profile - 1 {
                jobs.push(FitJob {
                    class,
                    profile,
                    rung: Some(rung),
                });
            }
        }
    }
    jobs
}

fn run_fit(
    job: &FitJob,
    classes: &[ScaleClass],
    prepared: &[Prepared],
    request: &Aspect3Request,
) -> Result<FittedAtoms> {
    let class = &classes[job.class];
    let prep = &prepared[job.class];
    let settings = request.profiles[job.profile].settings();
    match job.rung {
        None => fit_at(class, &prep.base_truth, prep.base, settings, request),
        Some(rung) => {
            let (scale, truth) = &prep.truths[rung];
            fit_at(class, truth, *scale, settings, request)
        }
    }
}

pub fn ladders(classes: &[ScaleClass], request: &Aspect3Request) -> Result<Vec<ClassLadder>> {
    leading_profile(request)?;
    let prepared = sweep::map(request.concurrency, classes, |class| prepare(class, request))?;
    let jobs = fit_jobs(classes.len(), request);
    let fits = sweep::map(request.concurrency, &jobs, |job| {
        run_fit(job, classes, &prepared, request)
    })?;
    let per_class = request.profiles.len() * fits_per_profile(request);
    let mut out = Vec::with_capacity(classes.len());
    for (index, (class, prep)) in classes.iter().zip(prepared.into_iter()).enumerate() {
        let own = &fits[index * per_class..(index + 1) * per_class];
        out.push(assemble(class, request, prep, own)?);
    }
    Ok(out)
}

pub fn class_ladder(class: &ScaleClass, request: &Aspect3Request) -> Result<ClassLadder> {
    match ladders(std::slice::from_ref(class), request)?.pop() {
        Some(ladder) => Ok(ladder),
        None => Err(LightsError::InvalidCamera(
            "aspect 3 returned no ladder for the one class it was given",
        )),
    }
}

fn assemble(
    class: &ScaleClass,
    request: &Aspect3Request,
    prepared: Prepared,
    fits: &[FittedAtoms],
) -> Result<ClassLadder> {
    let leading = leading_profile(request)?;
    let Prepared {
        base,
        base_viewport,
        truths,
        base_truth: _,
        wavelets,
    } = prepared;
    let ladder = request.ladder.clone();
    let per_profile = fits_per_profile(request);
    let missing = || {
        LightsError::InvalidCamera("aspect 3 assembled a ladder without the fit it was owed")
    };

    let mut rows: Vec<ScaleRow> = Vec::new();
    let mut renders: Vec<(String, Scale, String, FloatImage<f64>)> = Vec::new();
    let mut base_errors: Vec<(String, String, f64)> = Vec::new();
    let mut frozen_at_base: Vec<(String, FloatImage<f64>)> = Vec::new();
    let mut frozen_atoms: Vec<(String, Vec<Atom2d>)> = Vec::new();

    for (profile_index, profile) in request.profiles.iter().enumerate() {
        let settings = profile.settings();
        let first = profile_index * per_profile;
        let frozen = fits.get(first).ok_or_else(missing)?;
        frozen_atoms.push((profile.name().to_string(), frozen.atoms.clone()));
        for arm in arms(request.kernel) {
            let arm_name = arm.name();
            let independence = arm.independence();
            let honours = arm.honours_profile();
            if !honours && *profile != leading {
                continue;
            }
            let profile_label = if honours {
                profile.name().to_string()
            } else {
                "profile-free".to_string()
            };
            for (rung, scale) in ladder.iter().enumerate() {
                let viewport = class.viewport_at(*scale)?;
                let ratio = scale.ratio_to(base)?;
                let truth = match truths.iter().find(|(s, _)| s == scale) {
                    Some((_, t)) => t,
                    None => {
                        return Err(LightsError::ViewportZero {
                            width: viewport.0,
                            height: viewport.1,
                        });
                    }
                };
                let started = std::time::Instant::now();
                let (image, terms, bytes, exhausted, fit_renders, resample_name, absent) =
                    match arm {
                        Arm3::Frozen(kernel) => {
                            let moved = scaled_atoms(&frozen.atoms, ratio);
                            let image =
                                render_atoms_at(&moved, viewport, kernel, &settings, request.tile)?;
                            (
                                Some(image),
                                frozen.atoms.len(),
                                atom_bytes_symmetric(frozen.atoms.len()),
                                Some(frozen.improvement_exhausted),
                                frozen.renders,
                                "none",
                                false,
                            )
                        }
                        Arm3::Refit(kernel) => {
                            let fitted = fits.get(first + 1 + rung).ok_or_else(missing)?;
                            let image = render_atoms_at(
                                &fitted.atoms,
                                viewport,
                                kernel,
                                &settings,
                                request.tile,
                            )?;
                            (
                                Some(image),
                                fitted.atoms.len(),
                                atom_bytes_symmetric(fitted.atoms.len()),
                                Some(fitted.improvement_exhausted),
                                fitted.renders,
                                "none",
                                false,
                            )
                        }
                        Arm3::AnalyticSdf => {
                            let image = analytic_at(class, *scale)?;
                            let bytes = match class.analytic() {
                                Some(t) => analytic_bytes_symmetric(t),
                                None => 0.0,
                            };
                            let primitives = match class.analytic() {
                                Some(t) => t.primitive_count(),
                                None => 0,
                            };
                            let absent = image.is_none();
                            (image, primitives, bytes, None, 0, "none", absent)
                        }
                        Arm3::Wavelet(transform) => {
                            let prepared = match wavelets.iter().find(|(t, _)| *t == transform) {
                                Some((_, w)) => w,
                                None => {
                                    return Err(LightsError::ViewportZero {
                                        width: viewport.0,
                                        height: viewport.1,
                                    });
                                }
                            };
                            match prepared {
                                None => (None, 0, 0.0, None, 0, "none", true),
                                Some(prepared) => {
                                    let image =
                                        wavelet_at(prepared, viewport, ratio.factor())?;
                                    (
                                        Some(image),
                                        prepared.terms,
                                        coefficient_bytes_symmetric(
                                            prepared.terms,
                                            base_viewport.0,
                                            base_viewport.1,
                                        ),
                                        None,
                                        0,
                                        Resample::for_ratio(ratio.factor()).name(),
                                        false,
                                    )
                                }
                            }
                        }
                    };
                let render_nanos = started.elapsed().as_nanos();
                let at_base = *scale == base;

                let (error, worst, quality, transition, fwhm, agrees) = match &image {
                    None => (
                        f64::NAN,
                        f64::NAN,
                        false,
                        blank(),
                        blank(),
                        None,
                    ),
                    Some(rendered) => {
                        let (quality, transition, fwhm) =
                            judgements(class, rendered, *scale, &request.criterion)?;
                        let agrees = match arm {
                            Arm3::Refit(_) => {
                                if at_base {
                                    frozen_at_base
                                        .iter()
                                        .find(|(p, _)| p == &profile_label)
                                        .map(|(_, f)| pixelwise_identical(f, rendered))
                                } else {
                                    None
                                }
                            }
                            Arm3::Frozen(_) => None,
                            Arm3::AnalyticSdf => None,
                            Arm3::Wavelet(_) => None,
                        };
                        (
                            relative_l2(truth, rendered)?,
                            max_abs_error(truth, rendered)?,
                            quality,
                            transition,
                            fwhm,
                            agrees,
                        )
                    }
                };

                match (&arm, &image) {
                    (Arm3::Frozen(_), Some(rendered)) => {
                        if at_base {
                            frozen_at_base.push((profile_label.clone(), rendered.clone()));
                        }
                    }
                    (Arm3::Refit(_), _) => (),
                    (Arm3::AnalyticSdf, _) => (),
                    (Arm3::Wavelet(_), _) => (),
                    (Arm3::Frozen(_), None) => (),
                }

                if at_base {
                    base_errors.push((arm_name.clone(), profile_label.clone(), error));
                }
                let anchor = base_errors
                    .iter()
                    .find(|(a, p, _)| a == &arm_name && p == &profile_label)
                    .map(|(_, _, e)| *e);
                let decay = match anchor {
                    Some(e) if e > 0.0 => error / e,
                    Some(_) => f64::NAN,
                    None => f64::NAN,
                };

                match &image {
                    Some(rendered) => renders.push((
                        arm_name.clone(),
                        *scale,
                        profile_label.clone(),
                        rendered.clone(),
                    )),
                    None => (),
                }

                rows.push(ScaleRow {
                    class: class.name().to_string(),
                    direction: request.direction.name(),
                    profile: profile_label.clone(),
                    arm: arm_name.clone(),
                    independence: independence.name(),
                    scale: *scale,
                    base_scale: base,
                    viewport,
                    at_base,
                    absent,
                    terms,
                    description_bytes: bytes,
                    relative_l2: error,
                    max_abs: worst,
                    decay,
                    boundary_reference: boundary_localised_reference(base_viewport, viewport),
                    quality_measured: quality,
                    transition,
                    fwhm,
                    exhausted,
                    fit_renders,
                    render_nanos,
                    agrees_with_frozen_at_base: agrees,
                    resample: resample_name,
                });
            }
        }
    }

    Ok(ClassLadder {
        class: class.name().to_string(),
        direction: request.direction.name(),
        base_scale: base,
        base_viewport,
        analytic_present: class.analytic().is_some(),
        rows,
        truths,
        renders,
        frozen: frozen_atoms,
    })
}

#[derive(Clone, Debug)]
pub struct Aspect3Report {
    pub direction: &'static str,
    pub convention: Convention,
    pub search_convention: &'static str,
    pub kernel: &'static str,
    pub wavelet_terms: usize,
    pub max_atoms: usize,
    pub criterion: Criterion,
    pub ladders: Vec<ClassLadder>,
}

impl Aspect3Report {
    pub fn rows(&self) -> Vec<&ScaleRow> {
        self.ladders.iter().flat_map(|l| l.rows.iter()).collect()
    }

    pub fn base_agreement_failures(&self) -> Vec<&ScaleRow> {
        self.rows()
            .into_iter()
            .filter(|r| r.agrees_with_frozen_at_base == Some(false))
            .collect()
    }

    pub fn base_agreement_checked(&self) -> usize {
        self.rows()
            .into_iter()
            .filter(|r| r.agrees_with_frozen_at_base.is_some())
            .count()
    }

    pub fn ceiling_violations(&self) -> Vec<(String, String, Scale, f64, f64)> {
        let mut out = Vec::new();
        for ladder in self.ladders.iter() {
            for row in ladder.rows.iter() {
                if !row.arm.starts_with("frozen-atoms") {
                    continue;
                }
                let refit = row.arm.replace("frozen-atoms", "refitted-atoms");
                let mate = ladder
                    .rows
                    .iter()
                    .find(|r| r.arm == refit && r.profile == row.profile && r.scale == row.scale);
                match mate {
                    Some(m) => {
                        if m.relative_l2 > row.relative_l2 {
                            out.push((
                                ladder.class.clone(),
                                row.profile.clone(),
                                row.scale,
                                row.relative_l2,
                                m.relative_l2,
                            ));
                        }
                    }
                    None => (),
                }
            }
        }
        out
    }

    pub fn decay_that_is_only_the_metric(
        &self,
        tolerance: f64,
    ) -> Vec<(String, String, String, Scale, f64, f64)> {
        let mut out = Vec::new();
        for ladder in self.ladders.iter() {
            for row in ladder.rows.iter() {
                if row.absent || row.at_base {
                    continue;
                }
                let reference = row.boundary_reference;
                if !reference.is_finite() || !row.decay.is_finite() {
                    continue;
                }
                if (reference - 1.0).abs() <= tolerance {
                    continue;
                }
                if (row.decay / reference - 1.0).abs() <= tolerance {
                    out.push((
                        ladder.class.clone(),
                        row.arm.clone(),
                        row.profile.clone(),
                        row.scale,
                        row.decay,
                        reference,
                    ));
                }
            }
        }
        out
    }

    pub fn prefilter_gap(&self) -> Vec<(String, String, Scale, f64, f64)> {
        let mut out = Vec::new();
        for ladder in self.ladders.iter() {
            for row in ladder.rows.iter() {
                if row.profile != "contract" {
                    continue;
                }
                let mate = ladder.rows.iter().find(|r| {
                    r.profile == "prefiltered" && r.arm == row.arm && r.scale == row.scale
                });
                match mate {
                    Some(m) => out.push((
                        ladder.class.clone(),
                        row.arm.clone(),
                        row.scale,
                        row.relative_l2,
                        m.relative_l2,
                    )),
                    None => (),
                }
            }
        }
        out
    }
}

pub fn run(classes: &[ScaleClass], request: &Aspect3Request) -> Result<Aspect3Report> {
    let ladders = ladders(classes, request)?;
    Ok(Aspect3Report {
        direction: request.direction.name(),
        convention: Convention::SceneUnits,
        search_convention: request.search_convention(),
        kernel: request.kernel.name(),
        wavelet_terms: request.wavelet_terms,
        max_atoms: request.max_atoms,
        criterion: request.criterion,
        ladders,
    })
}

pub const SCALE_COLUMNS: [&str; 29] = [
    "class",
    "direction",
    "profile",
    "arm",
    "independence",
    "scale",
    "base_scale",
    "viewport",
    "at_base",
    "absent",
    "terms",
    "description_bytes",
    "relative_l2",
    "max_abs_error",
    "decay_from_own_base",
    "decay_if_error_is_boundary_localised",
    "quality_measured",
    "transition_10_90_scene",
    "transition_10_90_device_px",
    "meets_transition_scene",
    "meets_transition_device_px",
    "fwhm_scene",
    "fwhm_device_px",
    "meets_fwhm_scene",
    "meets_fwhm_device_px",
    "exhausted",
    "fit_renders",
    "render_ns",
    "resample",
];

fn number(v: f64) -> String {
    if v.is_finite() {
        format!("{v:.6}")
    } else {
        "nan".to_string()
    }
}

fn flag(v: bool) -> &'static str {
    if v { "yes" } else { "no" }
}

fn probed(measured: bool, value: f64, verdict: bool) -> String {
    if !measured || !value.is_finite() {
        return "-".to_string();
    }
    flag(verdict).to_string()
}

fn optional(v: Option<bool>) -> String {
    match v {
        Some(true) => "yes".to_string(),
        Some(false) => "no".to_string(),
        None => "-".to_string(),
    }
}

pub fn row_cells(r: &ScaleRow) -> Vec<String> {
    vec![
        r.class.clone(),
        r.direction.to_string(),
        r.profile.clone(),
        r.arm.clone(),
        r.independence.to_string(),
        r.scale.label(),
        r.base_scale.label(),
        format!("{}x{}", r.viewport.0, r.viewport.1),
        flag(r.at_base).to_string(),
        flag(r.absent).to_string(),
        r.terms.to_string(),
        number(r.description_bytes),
        number(r.relative_l2),
        number(r.max_abs),
        number(r.decay),
        number(r.boundary_reference),
        flag(r.quality_measured).to_string(),
        number(r.transition.scene),
        number(r.transition.device),
        probed(r.quality_measured, r.transition.scene, r.transition.meets_scene),
        probed(r.quality_measured, r.transition.device, r.transition.meets_device),
        number(r.fwhm.scene),
        number(r.fwhm.device),
        probed(r.quality_measured, r.fwhm.scene, r.fwhm.meets_scene),
        probed(r.quality_measured, r.fwhm.device, r.fwhm.meets_device),
        optional(r.exhausted),
        r.fit_renders.to_string(),
        r.render_nanos.to_string(),
        r.resample.to_string(),
    ]
}

pub fn header_lines(report: &Aspect3Report) -> Vec<String> {
    vec![
        format!("direction\t{}", report.direction),
        format!("primary_criterion_convention\t{}", report.convention.name()),
        format!("primary_convention_legend\t{}", report.convention.legend()),
        format!(
            "secondary_convention_legend\t{}",
            Convention::DevicePixels.legend()
        ),
        format!("fitter_search_convention\t{}", report.search_convention),
        format!("kernel\t{}", report.kernel),
        format!("max_atoms\t{}", report.max_atoms),
        format!("wavelet_terms\t{}", report.wavelet_terms),
        format!("criterion.max_transition\t{}", report.criterion.max_transition),
        format!("criterion.target_fwhm\t{}", report.criterion.target_fwhm),
        format!("criterion.max_fwhm_error\t{}", report.criterion.max_fwhm_error),
        format!("byte_convention\tsymmetric-f32-provisional-see-C2"),
        format!("render_ns_convention\trender-only-fit-excluded-see-fit_renders"),
    ]
}

pub fn write_table(path: &std::path::Path, report: &Aspect3Report) -> Result<()> {
    let mut text = String::new();
    for line in header_lines(report) {
        text.push('#');
        text.push_str(&line);
        text.push('\n');
    }
    text.push_str(&SCALE_COLUMNS.join("\t"));
    text.push('\n');
    for row in report.rows() {
        text.push_str(&row_cells(row).join("\t"));
        text.push('\n');
    }
    std::fs::write(path, text)?;
    Ok(())
}
