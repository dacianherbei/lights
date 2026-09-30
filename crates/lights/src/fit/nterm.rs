use crate::error::{LightsError, Result};
use crate::fit::atom2d::camera;
use crate::fit::content::Content;
use crate::fit::coverage::{DEFAULT_CELL_LIMIT, DEFAULT_MAX_DEPTH, coverage_budget};
use crate::fit::greedy::{
    DEFAULT_FIT_BUDGET, DEFAULT_REFINE_ITERATIONS, DEFAULT_REFINE_TAIL, FitRequest, Initialisation,
    Reading, fit, fit_budget, prefix, render_or_background, residual_energy,
};
use crate::fit::transform::weight::measure;
use crate::fit::transform::{
    CHANNELS, Ranking, Transform, forward, inverse, max_levels, rank_coefficients, zeroed,
};
use crate::image::FloatImage;
use crate::kernel::{Beta, KernelFamily};
use crate::settings::SharedSettings;
use crate::tiling::DEFAULT_TILE;

pub const LADDER: [usize; 20] = [
    1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024,
];
pub const MAX_ATOMS: usize = 8;
pub const VALUE_BYTES: f64 = 4.0;
pub const BITS_PER_BYTE: f64 = 8.0;
pub const ATOM_SCALARS_CARRIED: usize = 14;
pub const ATOM_SCALARS_NATIVE: usize = 9;

pub fn ladder_upto(available: usize) -> Vec<usize> {
    let mut out: Vec<usize> = LADDER
        .iter()
        .copied()
        .filter(|n| *n > 0 && *n <= available)
        .collect();
    if available > 0 && out.last() != Some(&available) {
        out.push(available);
    }
    out
}

pub fn positions(width: u32, height: u32) -> usize {
    (width as usize) * (height as usize) * CHANNELS
}

pub fn index_bits(width: u32, height: u32) -> f64 {
    (positions(width, height) as f64).log2().ceil()
}

pub fn bytes_explicit(terms: usize, width: u32, height: u32) -> f64 {
    (terms as f64) * (VALUE_BYTES + index_bits(width, height) / BITS_PER_BYTE)
}

pub fn log2_choose(total: usize, chosen: usize) -> f64 {
    if chosen > total {
        crate::guard::unreachable(
            "fit::nterm::log2_choose",
            &format!("{chosen} retained terms out of {total} available positions"),
        );
    }
    let short = chosen.min(total - chosen);
    let mut acc = 0.0f64;
    for i in 0..short {
        acc += ((total - i) as f64).log2() - ((i + 1) as f64).log2();
    }
    acc
}

pub fn bytes_combinatorial(terms: usize, width: u32, height: u32) -> f64 {
    (terms as f64) * VALUE_BYTES
        + log2_choose(positions(width, height), terms) / BITS_PER_BYTE
}

pub fn atom_bytes(terms: usize, scalars: usize) -> f64 {
    (terms as f64) * (scalars as f64) * VALUE_BYTES
}

pub fn image_energy(image: &FloatImage<f64>) -> f64 {
    let mut acc = 0.0f64;
    for p in image.pixels.iter() {
        acc += p.x * p.x + p.y * p.y + p.z * p.z;
    }
    acc
}

pub fn relative_l2(truth: &FloatImage<f64>, rendered: &FloatImage<f64>) -> Result<f64> {
    let residual = residual_energy(truth, rendered)?;
    let energy = image_energy(truth);
    if energy > 0.0 {
        return Ok((residual / energy).sqrt());
    }
    if residual == 0.0 {
        return Ok(0.0);
    }
    Err(LightsError::NonFinite {
        what: "fit::nterm::relative_l2 against a truth image carrying no energy",
    })
}

pub fn max_abs_error(truth: &FloatImage<f64>, rendered: &FloatImage<f64>) -> Result<f64> {
    Ok(truth.difference(rendered)?.max_abs().0)
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Optimality {
    ExactNTerm,
    BasisRanked,
    GreedyLowerBound,
}

impl Optimality {
    pub const ALL: [Optimality; 3] = [
        Optimality::ExactNTerm,
        Optimality::BasisRanked,
        Optimality::GreedyLowerBound,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Optimality::ExactNTerm => "exact-n-term",
            Optimality::BasisRanked => "basis-ranked",
            Optimality::GreedyLowerBound => "greedy-lower-bound",
        }
    }

    pub fn legend(self) -> &'static str {
        match self {
            Optimality::ExactNTerm => {
                "orthonormal basis: keeping the N largest coefficients IS the optimal N-term \
                 approximation, obtained exactly and with no search"
            }
            Optimality::BasisRanked => {
                "biorthogonal basis: the N kept coefficients are the N largest under the \
                 subband-weighted ranking, which is NOT proven optimal because the synthesis \
                 functions are not mutually orthogonal"
            }
            Optimality::GreedyLowerBound => {
                "greedy matching pursuit: a lower bound on what N atoms can represent, since \
                 hand-built atoms beat it in both directions in the earlier run"
            }
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ByteRule {
    RetainedCoefficient,
    AtomParameterVector,
}

impl ByteRule {
    pub fn name(self) -> &'static str {
        match self {
            ByteRule::RetainedCoefficient => "coefficient-value-plus-index",
            ByteRule::AtomParameterVector => "atom-parameter-vector",
        }
    }

    pub fn legend(self) -> &'static str {
        match self {
            ByteRule::RetainedCoefficient => {
                "a retained coefficient is a value plus a position: the explicit column pays \
                 an uncompressed index and brackets from ABOVE, the combinatorial column pays \
                 the exact cost of naming which N of the 3*W*H positions were kept and \
                 brackets from BELOW"
            }
            ByteRule::AtomParameterVector => {
                "an atom is a fully specified parameter vector and needs no index: the two \
                 columns are the fourteen-scalar convention the earlier run carried and the \
                 nine-scalar count a 2D-native atom actually carries, both billed at the f32 \
                 value a retained coefficient pays"
            }
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Arm {
    Atoms(KernelFamily),
    Wavelet(Transform, Ranking),
}

pub const REACHABLE_ARMS: [Arm; 4] = [
    Arm::Atoms(KernelFamily::Gaussian),
    Arm::Atoms(KernelFamily::SuperGaussian(Beta::Four)),
    Arm::Wavelet(Transform::Haar, Ranking::Weighted),
    Arm::Wavelet(Transform::Cdf97, Ranking::Weighted),
];

impl Arm {
    pub fn name(self) -> String {
        match self {
            Arm::Atoms(kernel) => format!("atoms-{}", kernel.name()),
            Arm::Wavelet(transform, ranking) => {
                format!("{}-{}", transform.name(), ranking.name())
            }
        }
    }

    pub fn optimality(self) -> Optimality {
        match self {
            Arm::Atoms(_) => Optimality::GreedyLowerBound,
            Arm::Wavelet(transform, _) => {
                if transform.orthonormal() {
                    Optimality::ExactNTerm
                } else {
                    Optimality::BasisRanked
                }
            }
        }
    }

    pub fn byte_rule(self) -> ByteRule {
        match self {
            Arm::Atoms(_) => ByteRule::AtomParameterVector,
            Arm::Wavelet(_, _) => ByteRule::RetainedCoefficient,
        }
    }

    pub fn has_fitter(self) -> bool {
        match self {
            Arm::Atoms(_) => true,
            Arm::Wavelet(_, _) => false,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NTermRequest {
    pub arm: Arm,
    pub settings: SharedSettings<f64>,
    pub tile: u32,
    pub max_atoms: usize,
    pub initialisation: Initialisation,
    pub refine_iterations: u32,
    pub refine_tail: usize,
    pub render_limit: u64,
    pub quadrature_depth: u32,
    pub cell_limit: u64,
    pub levels: Option<u32>,
}

impl NTermRequest {
    pub fn new(arm: Arm) -> Self {
        Self {
            arm,
            settings: SharedSettings::<f64>::prefiltered(),
            tile: DEFAULT_TILE,
            max_atoms: MAX_ATOMS,
            initialisation: Initialisation::StructureTensor,
            refine_iterations: DEFAULT_REFINE_ITERATIONS,
            refine_tail: DEFAULT_REFINE_TAIL,
            render_limit: DEFAULT_FIT_BUDGET,
            quadrature_depth: DEFAULT_MAX_DEPTH,
            cell_limit: DEFAULT_CELL_LIMIT,
            levels: None,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct NTermPoint {
    pub terms: usize,
    pub relative_l2: f64,
    pub max_abs_error: f64,
    pub bytes_explicit: f64,
    pub bytes_combinatorial: f64,
    pub optimality: Optimality,
    pub reading: Reading,
    pub wall_nanos: u128,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NTermCurve {
    pub content: String,
    pub arm: String,
    pub viewport: (u32, u32),
    pub optimality: Optimality,
    pub byte_rule: ByteRule,
    pub control: String,
    pub available_terms: usize,
    pub improvement_exhausted: bool,
    pub truth_energy: f64,
    pub points: Vec<NTermPoint>,
}

impl NTermCurve {
    pub fn residual_energy_at(&self, terms: usize) -> Option<f64> {
        self.points
            .iter()
            .find(|p| p.terms == terms)
            .map(|p| p.relative_l2 * p.relative_l2 * self.truth_energy)
    }
}

fn coefficient_point(
    terms: usize,
    truth: &FloatImage<f64>,
    rendered: &FloatImage<f64>,
    viewport: (u32, u32),
    optimality: Optimality,
    wall_nanos: u128,
) -> Result<NTermPoint> {
    Ok(NTermPoint {
        terms,
        relative_l2: relative_l2(truth, rendered)?,
        max_abs_error: max_abs_error(truth, rendered)?,
        bytes_explicit: bytes_explicit(terms, viewport.0, viewport.1),
        bytes_combinatorial: bytes_combinatorial(terms, viewport.0, viewport.1),
        optimality,
        reading: Reading::OwnFit,
        wall_nanos,
    })
}

fn atom_point(
    terms: usize,
    reading: Reading,
    truth: &FloatImage<f64>,
    rendered: &FloatImage<f64>,
    wall_nanos: u128,
) -> Result<NTermPoint> {
    Ok(NTermPoint {
        terms,
        relative_l2: relative_l2(truth, rendered)?,
        max_abs_error: max_abs_error(truth, rendered)?,
        bytes_explicit: atom_bytes(terms, ATOM_SCALARS_CARRIED),
        bytes_combinatorial: atom_bytes(terms, ATOM_SCALARS_NATIVE),
        optimality: Optimality::GreedyLowerBound,
        reading,
        wall_nanos,
    })
}

pub fn atom_curve(
    content: &Content,
    truth: &FloatImage<f64>,
    kernel: KernelFamily,
    request: &NTermRequest,
) -> Result<NTermCurve> {
    let scaffold = content.fit_scaffold();
    let mut fit_request = FitRequest::new(kernel, request.settings);
    fit_request.max_atoms = request.max_atoms;
    fit_request.tile = request.tile;
    fit_request.refine_iterations = request.refine_iterations;
    fit_request.refine_tail = request.refine_tail;
    fit_request.initialisation = request.initialisation;
    let mut budget = fit_budget(request.render_limit);
    let outcome = fit(&scaffold, truth, &fit_request, &mut budget)?;

    let viewport = content.viewport();
    let spec = camera(viewport.0, viewport.1);
    let mut points = Vec::new();
    for n in ladder_upto(outcome.atoms.len()) {
        let (slice, reading) = prefix(&outcome.atoms, n);
        let started = std::time::Instant::now();
        let image = render_or_background(
            slice,
            &spec,
            kernel,
            &request.settings,
            request.tile,
        )?;
        let wall = started.elapsed().as_nanos();
        points.push(atom_point(n, reading, truth, &image, wall)?);
    }

    Ok(NTermCurve {
        content: content.name().to_string(),
        arm: request.arm.name(),
        viewport,
        optimality: Optimality::GreedyLowerBound,
        byte_rule: ByteRule::AtomParameterVector,
        control: content.control_label().to_string(),
        available_terms: outcome.atoms.len(),
        improvement_exhausted: outcome.improvement_exhausted,
        truth_energy: image_energy(truth),
        points,
    })
}

pub fn wavelet_curve(
    content: &Content,
    truth: &FloatImage<f64>,
    transform: Transform,
    ranking: Ranking,
    request: &NTermRequest,
) -> Result<NTermCurve> {
    let viewport = content.viewport();
    let levels = match request.levels {
        Some(l) => l,
        None => max_levels(viewport.0, viewport.1),
    };
    let coefficients = forward(transform, truth, levels)?;
    let weights = measure(transform, viewport.0, viewport.1, levels)?;
    let ranked = rank_coefficients(&coefficients, &weights, ranking)?;
    let optimality = request.arm.optimality();

    let mut kept = zeroed(transform, viewport.0, viewport.1, levels);
    let mut placed = 0usize;
    let mut points = Vec::new();
    for n in ladder_upto(coefficients.terms()) {
        for entry in ranked[placed..n].iter() {
            kept.set(entry.channel, entry.x, entry.y, entry.value);
        }
        placed = n;
        let started = std::time::Instant::now();
        let image = inverse(&kept)?;
        let wall = started.elapsed().as_nanos();
        points.push(coefficient_point(n, truth, &image, viewport, optimality, wall)?);
    }

    Ok(NTermCurve {
        content: content.name().to_string(),
        arm: request.arm.name(),
        viewport,
        optimality,
        byte_rule: ByteRule::RetainedCoefficient,
        control: content.control_label().to_string(),
        available_terms: coefficients.terms(),
        improvement_exhausted: false,
        truth_energy: image_energy(truth),
        points,
    })
}

pub fn nterm_curve(content: &Content, request: &NTermRequest) -> Result<NTermCurve> {
    let mut cells = coverage_budget(request.cell_limit);
    let truth = content.truth(request.quadrature_depth, &mut cells)?;
    match request.arm {
        Arm::Atoms(kernel) => atom_curve(content, &truth, kernel, request),
        Arm::Wavelet(transform, ranking) => {
            wavelet_curve(content, &truth, transform, ranking, request)
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ByteVerdict {
    TransformCheaper,
    AtomsCheaper,
    Bracketed,
    NeverReached,
}

impl ByteVerdict {
    pub const ALL: [ByteVerdict; 4] = [
        ByteVerdict::TransformCheaper,
        ByteVerdict::AtomsCheaper,
        ByteVerdict::Bracketed,
        ByteVerdict::NeverReached,
    ];

    pub fn name(self) -> &'static str {
        match self {
            ByteVerdict::TransformCheaper => "transform-cheaper",
            ByteVerdict::AtomsCheaper => "atoms-cheaper",
            ByteVerdict::Bracketed => "bracketed",
            ByteVerdict::NeverReached => "never-reached",
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ByteCrossing {
    pub atoms: usize,
    pub atom_relative_l2: f64,
    pub atom_bytes: f64,
    pub transform_terms_above: usize,
    pub transform_terms: Option<usize>,
    pub transform_relative_l2: Option<f64>,
    pub transform_bytes_low: Option<f64>,
    pub transform_bytes_high: Option<f64>,
    pub verdict: ByteVerdict,
}

pub fn byte_crossings(atoms: &NTermCurve, transform: &NTermCurve) -> Result<Vec<ByteCrossing>> {
    if atoms.byte_rule != ByteRule::AtomParameterVector {
        return Err(LightsError::MetricUndefined {
            what: "fit::nterm::byte_crossings was given a coefficient curve as its atom curve",
        });
    }
    if transform.byte_rule != ByteRule::RetainedCoefficient {
        return Err(LightsError::MetricUndefined {
            what: "fit::nterm::byte_crossings was given an atom curve as its transform curve",
        });
    }
    if atoms.viewport != transform.viewport {
        return Err(LightsError::SizeMismatch {
            a: atoms.viewport,
            b: transform.viewport,
        });
    }
    let (width, height) = transform.viewport;
    let mut out = Vec::with_capacity(atoms.points.len());
    for point in atoms.points.iter() {
        let target = point.relative_l2;
        let cost = atom_bytes(point.terms, ATOM_SCALARS_NATIVE);
        let mut previous = 0usize;
        let mut reached: Option<&NTermPoint> = None;
        for candidate in transform.points.iter() {
            if candidate.relative_l2 <= target {
                reached = Some(candidate);
                break;
            }
            previous = candidate.terms;
        }
        let crossing = match reached {
            Some(hit) => {
                let low_terms = previous + 1;
                let low = bytes_combinatorial(low_terms, width, height);
                let high = hit.bytes_explicit;
                let verdict = if high < cost {
                    ByteVerdict::TransformCheaper
                } else if low > cost {
                    ByteVerdict::AtomsCheaper
                } else {
                    ByteVerdict::Bracketed
                };
                ByteCrossing {
                    atoms: point.terms,
                    atom_relative_l2: target,
                    atom_bytes: cost,
                    transform_terms_above: previous,
                    transform_terms: Some(hit.terms),
                    transform_relative_l2: Some(hit.relative_l2),
                    transform_bytes_low: Some(low),
                    transform_bytes_high: Some(high),
                    verdict,
                }
            }
            None => ByteCrossing {
                atoms: point.terms,
                atom_relative_l2: target,
                atom_bytes: cost,
                transform_terms_above: previous,
                transform_terms: None,
                transform_relative_l2: None,
                transform_bytes_low: None,
                transform_bytes_high: None,
                verdict: ByteVerdict::NeverReached,
            },
        };
        out.push(crossing);
    }
    Ok(out)
}
