use crate::camera::{Camera, CameraSpec};
use crate::dispatch::Collect;
use crate::error::{LightsError, Result};
use crate::fit::analytic::{AnalyticArm, render_analytic};
use crate::fit::atom2d::{Atom2d, DZ, Z0, camera, shifted_camera, to_scene};
use crate::fit::coverage::{
    DEFAULT_CELL_LIMIT, DEFAULT_MAX_DEPTH, coverage_budget, coverage_image_to_colour,
    exact_coverage,
};
use crate::fit::edge::{
    Criterion, SDF_ARM_TRUE_FWHM, SDF_ARM_TRUE_TRANSITION, TargetMetric, measure_target,
    measure_target_shifted,
};
use crate::fit::greedy::{
    DEFAULT_FIT_BUDGET, FitRequest, Reading, fit, fit_budget, prefix, render_atoms,
    render_or_background, residual_energy, residual_peak, splat_method,
};
use crate::fit::report::{
    ATOM_ARM, BinRow, Header, NATURAL, NO_AXIS, ProbeRow, Report, Row, SensitivityRow, Sheet,
    SweepRow, TILED,
};
use crate::fit::sweep::{self, Concurrency};
use crate::fit::target::{ProbeKind, Target};
use crate::image::FloatImage;
use crate::kernel::compact::CompactBump;
use crate::kernel::gabor::{GaborColour, GaborDensity};
use crate::kernel::gaussian::{Gaussian, GaussianNormalised};
use crate::kernel::hermite::{GaussHermiteColour, GaussHermiteDensity};
use crate::kernel::supergaussian::SuperGaussian;
use crate::kernel::{CurveShape, Kernel, KernelFamily};
use crate::method::splat::SplatRaster;
use crate::method::{ImageFormation, RenderStrategy, create_strategy};
use crate::numerics::vector::Vec2;
use crate::probe::Counters;
use crate::scene::Scene;
use crate::scene::prepare::PreparedScene;
use crate::settings::{PIXEL_BOX_VARIANCE, SharedSettings};
use crate::tiling::DEFAULT_TILE;

pub const FIELD_SIZE: u32 = 512;
pub const FIELD_COPIES: usize = 40;
pub const GRID_SHIFT: f64 = 0.5;
pub const DILATION_SWEEP: [f64; 3] = [PIXEL_BOX_VARIANCE, 0.1, 0.25];
pub const CUTOFF_SWEEP: [f64; 3] = [1.0 / 255.0, 1.0 / 1024.0, 1.0 / 65536.0];
pub const CI95: f64 = 1.959963984540054;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ProfileChoice {
    Prefiltered,
    Contract,
}

impl ProfileChoice {
    pub const ALL: [ProfileChoice; 2] = [ProfileChoice::Prefiltered, ProfileChoice::Contract];

    pub fn name(self) -> &'static str {
        match self {
            ProfileChoice::Prefiltered => "prefiltered",
            ProfileChoice::Contract => "contract",
        }
    }

    pub fn settings(self) -> SharedSettings<f64> {
        match self {
            ProfileChoice::Prefiltered => SharedSettings::<f64>::prefiltered(),
            ProfileChoice::Contract => SharedSettings::<f64>::contract(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProbeRequest {
    pub targets: Vec<usize>,
    pub blends: Vec<KernelFamily>,
    pub profiles: Vec<ProfileChoice>,
    pub max_atoms: usize,
    pub tile: u32,
    pub repeats: usize,
    pub tiled_field: bool,
    pub quadrature_depth: u32,
    pub cell_limit: u64,
    pub render_limit: u64,
    pub criterion: Criterion,
    pub sensitivity: bool,
    pub concurrency: Concurrency,
}

impl ProbeRequest {
    pub fn all() -> Self {
        Self {
            targets: (0..Target::<f64>::all().len()).collect(),
            blends: KernelFamily::BLEND_PAIR.to_vec(),
            profiles: ProfileChoice::ALL.to_vec(),
            max_atoms: 8,
            tile: DEFAULT_TILE,
            repeats: 5,
            tiled_field: false,
            quadrature_depth: DEFAULT_MAX_DEPTH,
            cell_limit: DEFAULT_CELL_LIMIT,
            render_limit: DEFAULT_FIT_BUDGET,
            criterion: Criterion::registered(),
            sensitivity: true,
            concurrency: Concurrency::Parallel,
        }
    }

    pub fn cells(&self) -> usize {
        self.targets.len() * self.blends.len() * self.profiles.len()
    }

    pub fn expected_rows(&self) -> usize {
        let atom = self.cells() * if self.tiled_field { 2 } else { 1 };
        atom + self.targets.len() * AnalyticArm::ALL.len()
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct TileCost {
    pub tile_atom_pairs: u64,
    pub candidates_tested: u64,
    pub kernel_evaluations: u64,
    pub max_tile_list: u32,
    pub mean_tile_list: f64,
    pub box_over_ellipse_area: f64,
}

fn binning_waste<T: crate::numerics::float::FloatingPoint, K: Kernel<T>>(
    raster: &SplatRaster<T, K>,
) -> (u32, f64, f64) {
    let ranges = raster.ranges();
    let mut max_list = 0u32;
    let mut total = 0u64;
    for (lo, hi) in ranges.iter() {
        let n = hi.saturating_sub(*lo);
        total += n as u64;
        if n > max_list {
            max_list = n;
        }
    }
    let mean_list = if ranges.is_empty() {
        f64::NAN
    } else {
        total as f64 / ranges.len() as f64
    };
    let atoms = raster.screen_atoms();
    let mut ratio_sum = 0.0f64;
    let mut counted = 0u32;
    for a in atoms.iter() {
        let box_area = (a.half_extent.x * a.half_extent.y).to_f64() * 4.0;
        let det = a.frame.cov.determinant().to_f64();
        let ellipse = core::f64::consts::PI * a.support_sq.to_f64() * det.max(0.0).sqrt();
        if ellipse > 0.0 {
            ratio_sum += box_area / ellipse;
            counted += 1;
        }
    }
    let ratio = if counted == 0 {
        f64::NAN
    } else {
        ratio_sum / counted as f64
    };
    (max_list, mean_list, ratio)
}

fn bound_raster<K: Kernel<f64>>(
    scene: &Scene<f64>,
    spec: &CameraSpec,
    shape: &CurveShape,
    settings: &SharedSettings<f64>,
    tile: u32,
) -> Result<(u32, f64, f64, Counters)> {
    let cam = Camera::<f64>::from_spec(spec, settings.near_plane)?;
    let prepared = PreparedScene::<f64, K>::build_uniform(scene, shape, settings)?;
    let mut strategy = create_strategy::<f64, K>(&splat_method(tile), settings)?;
    strategy.build(&prepared)?;
    strategy.bind(&prepared, &cam)?;
    match strategy {
        RenderStrategy::SplatRaster(raster) => {
            let (max_list, mean_list, ratio) = binning_waste(&raster);
            Ok((max_list, mean_list, ratio, raster.bind_counters()))
        }
        RenderStrategy::RayCast(_) => Err(LightsError::InvalidCamera(
            "the P2 harness reads tile cost from the rasteriser only",
        )),
    }
}

pub fn tile_cost(
    atoms: &[Atom2d],
    spec: &CameraSpec,
    kernel: KernelFamily,
    settings: &SharedSettings<f64>,
    tile: u32,
) -> Result<TileCost> {
    if atoms.is_empty() {
        return Ok(TileCost::default());
    }
    let scene = to_scene(atoms, (spec.width, spec.height), settings.scale_floor)?;
    let shape = kernel.shape();
    let (max_list, mean_list, ratio, _) = match kernel {
        KernelFamily::Gaussian => bound_raster::<Gaussian>(&scene, spec, &shape, settings, tile)?,
        KernelFamily::GaussianNormalised => {
            bound_raster::<GaussianNormalised>(&scene, spec, &shape, settings, tile)?
        }
        KernelFamily::SuperGaussian(_) => {
            bound_raster::<SuperGaussian>(&scene, spec, &shape, settings, tile)?
        }
        KernelFamily::CompactBump => {
            bound_raster::<CompactBump>(&scene, spec, &shape, settings, tile)?
        }
        KernelFamily::GaborColour(_) => {
            bound_raster::<GaborColour>(&scene, spec, &shape, settings, tile)?
        }
        KernelFamily::GaborDensity(_) => {
            bound_raster::<GaborDensity>(&scene, spec, &shape, settings, tile)?
        }
        KernelFamily::GaussHermiteColour(_) => {
            bound_raster::<GaussHermiteColour>(&scene, spec, &shape, settings, tile)?
        }
        KernelFamily::GaussHermiteDensity(_) => {
            bound_raster::<GaussHermiteDensity>(&scene, spec, &shape, settings, tile)?
        }
    };
    let outcome = render_atoms(
        atoms,
        spec,
        kernel,
        settings,
        tile,
        Collect::PixelsAndCounters,
    )?;
    Ok(TileCost {
        tile_atom_pairs: outcome.report.bind.counters.tile_atom_pairs,
        candidates_tested: outcome.counters.candidates_tested,
        kernel_evaluations: outcome.counters.kernel_evaluations,
        max_tile_list: max_list,
        mean_tile_list: mean_list,
        box_over_ellipse_area: ratio,
    })
}

pub fn tiled_atoms(atoms: &[Atom2d], viewport: (u32, u32), copies: usize) -> Vec<Atom2d> {
    let across = (FIELD_SIZE / viewport.0.max(1)).max(1);
    let mut out = Vec::new();
    for copy in 0..copies {
        let column = (copy as u32) % across;
        let row = (copy as u32) / across;
        let offset = Vec2::new(
            (column * viewport.0) as f64,
            (row * viewport.1) as f64,
        );
        for a in atoms.iter() {
            let mut moved = *a;
            moved.centre = a.centre + offset;
            out.push(moved);
        }
    }
    out
}

fn statistics(samples: &[f64]) -> (f64, f64) {
    if samples.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    if samples.len() < 2 {
        return (mean, 0.0);
    }
    let variance = samples
        .iter()
        .map(|s| (s - mean) * (s - mean))
        .sum::<f64>()
        / (n - 1.0);
    (mean, CI95 * (variance / n).sqrt())
}

fn timed(
    atoms: &[Atom2d],
    spec: &CameraSpec,
    kernel: KernelFamily,
    settings: &SharedSettings<f64>,
    tile: u32,
    repeats: usize,
) -> Result<(f64, f64)> {
    if atoms.is_empty() || repeats == 0 {
        return Ok((f64::NAN, f64::NAN));
    }
    let mut samples = Vec::new();
    for index in 0..repeats + 1 {
        let started = std::time::Instant::now();
        let _ = render_atoms(atoms, spec, kernel, settings, tile, Collect::Pixels)?;
        let elapsed = started.elapsed().as_nanos() as f64;
        if index > 0 {
            samples.push(elapsed);
        }
    }
    Ok(statistics(&samples))
}

fn blank_row(
    target: &Target<f64>,
    blend: &str,
    profile: &str,
    arm: &str,
    field: &str,
) -> Row {
    Row {
        target: target.name.to_string(),
        blend: blend.to_string(),
        profile: profile.to_string(),
        arm: arm.to_string(),
        field: field.to_string(),
        atoms: 0,
        atoms_available: 0,
        reading: Reading::OwnFit,
        met_criterion_at: None,
        primitives: target.primitive_count(),
        description_bytes: target.description_bytes(),
        transition_10_90_px: f64::NAN,
        contrast_frac: f64::NAN,
        peak_frac: f64::NAN,
        fwhm_px: f64::NAN,
        fwhm_error_px: f64::NAN,
        meets_criterion: false,
        quality_measured: true,
        improvement_exhausted: false,
        fit_renders: 0,
        delta_to_control_px: f64::NAN,
        delta_to_control_peak_frac: f64::NAN,
        instrument_offset_px: f64::NAN,
        instrument_offset_fwhm_px: f64::NAN,
        apex_displacement_px: f64::NAN,
        interior_ripple: f64::NAN,
        residual_energy: f64::NAN,
        residual_peak: f64::NAN,
        thin_bins: 0,
        samples: 0,
        shifted_grid_transition_10_90_px: f64::NAN,
        overfit_gap_px: f64::NAN,
        tile_atom_pairs: 0,
        candidates_tested: 0,
        kernel_evaluations: 0,
        max_tile_list: 0,
        mean_tile_list: f64::NAN,
        box_over_ellipse_area: f64::NAN,
        render_ns_mean: f64::NAN,
        render_ns_ci95: f64::NAN,
    }
}

fn fill_quality(row: &mut Row, metric: &TargetMetric) {
    row.transition_10_90_px = metric.transition_10_90;
    row.contrast_frac = metric.contrast_frac;
    row.peak_frac = metric.peak_frac;
    row.fwhm_px = metric.fwhm;
    row.fwhm_error_px = metric.fwhm_error;
    row.apex_displacement_px = metric.apex_displacement;
    row.interior_ripple = metric.interior_ripple;
    row.thin_bins = metric.thin_bins;
    row.samples = metric.probes.iter().map(|p| p.samples).sum();
    row.meets_criterion = metric.meets;
}

fn kind_name(kind: ProbeKind) -> &'static str {
    match kind {
        ProbeKind::Edge => "edge",
        ProbeKind::Stroke => "stroke",
        ProbeKind::Interior => "interior",
        ProbeKind::Apex => "apex",
    }
}

fn detail(
    target: &Target<f64>,
    blend: &str,
    profile: &str,
    arm: &str,
    atoms: usize,
    metric: &TargetMetric,
    probes: &mut Vec<ProbeRow>,
    bins: &mut Vec<BinRow>,
) {
    for m in metric.probes.iter() {
        probes.push(ProbeRow {
            target: target.name.to_string(),
            blend: blend.to_string(),
            profile: profile.to_string(),
            arm: arm.to_string(),
            atoms,
            feature: m.feature.0,
            kind: kind_name(m.kind).to_string(),
            samples: m.samples,
            thin_bins: m.thin_bins,
            transition_10_90_px: m.transition_10_90,
            sdf_90: m.sdf_90,
            sdf_10: m.sdf_10,
            contrast: m.contrast,
            target_contrast: m.target_contrast,
            peak: m.peak,
            fwhm_px: m.fwhm,
            plateau_low: m.plateau_low,
            plateau_high: m.plateau_high,
            apex_displacement_px: m.apex_displacement,
            interior_ripple: m.interior_ripple,
            judged: m.judged,
            meets: m.meets,
        });
        for b in m.bins.iter() {
            bins.push(BinRow {
                target: target.name.to_string(),
                blend: blend.to_string(),
                profile: profile.to_string(),
                arm: arm.to_string(),
                feature: m.feature.0,
                sdf: b.sdf,
                mean: b.mean,
                samples: b.samples,
            });
        }
    }
}

fn hardware_banner() -> String {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    format!(
        "{} {} cores {}",
        std::env::consts::OS,
        cores,
        std::env::consts::ARCH
    )
}

fn build_profile() -> &'static str {
    if cfg!(debug_assertions) {
        "DEBUG - every wall-clock number below is meaningless as an absolute"
    } else {
        "release"
    }
}

struct Head {
    target: Target<f64>,
    truth: FloatImage<f64>,
    undecided: f64,
    rows: Vec<Row>,
    sheets: Vec<Sheet>,
    probes: Vec<ProbeRow>,
    bins: Vec<BinRow>,
    control_width: f64,
    control_peak: f64,
    sdf_arm_width: f64,
    sdf_arm_fwhm: f64,
    control_valid: bool,
}

struct Timing {
    atoms: Vec<Atom2d>,
    spec: CameraSpec,
    blend: KernelFamily,
    settings: SharedSettings<f64>,
}

struct Cell {
    head: usize,
    sweep: Vec<SweepRow>,
    row: Row,
    timing: Option<Timing>,
    field: Option<(Row, Timing)>,
    probes: Vec<ProbeRow>,
    bins: Vec<BinRow>,
    sheets: Vec<Sheet>,
    sensitivity: Vec<SensitivityRow>,
}

fn head(all: &[Target<f64>], index: usize, request: &ProbeRequest) -> Result<Head> {
    let target = match all.get(index) {
        Some(t) => t.clone(),
        None => {
            return Err(LightsError::InvalidCamera(
                "the P2 harness was given a target index that does not exist",
            ));
        }
    };
    let mut cells = coverage_budget(request.cell_limit);
    let (coverage, residual) = exact_coverage(&target, request.quadrature_depth, &mut cells)?;
    let truth = coverage_image_to_colour(&target, &coverage);
    let mut out = Head {
        undecided: residual.undecided_area,
        rows: Vec::new(),
        sheets: vec![Sheet {
            name: format!("{}-truth", target.name),
            image: truth.clone(),
            normalized: false,
        }],
        probes: Vec::new(),
        bins: Vec::new(),
        control_width: f64::NAN,
        control_peak: f64::NAN,
        sdf_arm_width: f64::NAN,
        sdf_arm_fwhm: f64::NAN,
        control_valid: true,
        truth,
        target,
    };
    for arm in AnalyticArm::ALL {
        let mut budget = coverage_budget(request.cell_limit);
        let (image, cost) =
            render_analytic(&out.target, arm, request.quadrature_depth, &mut budget)?;
        let metric = measure_target(&image, &out.target, &request.criterion)?;
        let mut row = blank_row(&out.target, NO_AXIS, NO_AXIS, arm.name(), NATURAL);
        fill_quality(&mut row, &metric);
        detail(
            &out.target,
            NO_AXIS,
            NO_AXIS,
            arm.name(),
            0,
            &metric,
            &mut out.probes,
            &mut out.bins,
        );
        row.description_bytes = cost.description_bytes;
        row.primitives = cost.primitives;
        row.kernel_evaluations = cost.sdf_evaluations;
        row.residual_energy = residual_energy(&out.truth, &image)?;
        row.residual_peak = residual_peak(&out.truth.difference(&image)?);
        row.render_ns_ci95 = 0.0;
        row.met_criterion_at = if metric.meets { Some(0) } else { None };
        match arm {
            AnalyticArm::Exact => {
                out.control_width = metric.transition_10_90;
                out.control_peak = metric.peak_frac;
                row.delta_to_control_px = 0.0;
                row.delta_to_control_peak_frac = 0.0;
                out.control_valid = metric.meets;
            }
            AnalyticArm::Sdf => {
                out.sdf_arm_width = metric.transition_10_90;
                out.sdf_arm_fwhm = metric.fwhm;
                row.delta_to_control_px = metric.transition_10_90 - out.control_width;
                row.delta_to_control_peak_frac = metric.peak_frac - out.control_peak;
            }
        }
        out.rows.push(row);
        out.sheets.push(Sheet {
            name: format!("{}-{}", out.target.name, arm.name()),
            image,
            normalized: false,
        });
    }
    Ok(out)
}

fn cell(
    heads: &[Head],
    job: &(usize, KernelFamily, ProfileChoice),
    request: &ProbeRequest,
) -> Result<Cell> {
    let (position, blend, profile) = *job;
    let h = &heads[position];
    let target = &h.target;
    let truth = &h.truth;
    let settings = profile.settings();
    let mut fit_request = FitRequest::new(blend, settings);
    fit_request.max_atoms = request.max_atoms;
    fit_request.tile = request.tile;
    let mut budget = fit_budget(request.render_limit);
    let outcome = fit(target, truth, &fit_request, &mut budget)?;
    let spec = camera(target.viewport.0, target.viewport.1);
    let shifted = shifted_camera(
        target.viewport.0,
        target.viewport.1,
        Vec2::new(GRID_SHIFT, GRID_SHIFT),
    );
    let mut out = Cell {
        head: position,
        sweep: Vec::new(),
        row: blank_row(target, blend.name(), profile.name(), ATOM_ARM, NATURAL),
        timing: None,
        field: None,
        probes: Vec::new(),
        bins: Vec::new(),
        sheets: Vec::new(),
        sensitivity: Vec::new(),
    };

    let mut first_pass: Option<usize> = None;
    let mut chosen_count: Option<usize> = None;
    let mut chosen: Option<(usize, TargetMetric, FloatImage<f64>, f64, f64)> = None;
    for n in 1..=outcome.atoms.len() {
        let (slice, reading) = prefix(&outcome.atoms, n);
        let image = render_or_background(slice, &spec, blend, &settings, request.tile)?;
        let metric = measure_target(&image, target, &request.criterion)?;
        let energy = residual_energy(truth, &image)?;
        let peak = residual_peak(&truth.difference(&image)?);
        out.sweep.push(SweepRow {
            target: target.name.to_string(),
            blend: blend.name().to_string(),
            profile: profile.name().to_string(),
            atoms: n,
            transition_10_90_px: metric.transition_10_90,
            contrast_frac: metric.contrast_frac,
            peak_frac: metric.peak_frac,
            fwhm_px: metric.fwhm,
            residual_energy: energy,
            thin_bins: metric.thin_bins,
            meets_criterion: metric.meets,
            reading,
        });
        let take = match first_pass {
            Some(_) => false,
            None => metric.meets || n == outcome.atoms.len(),
        };
        if metric.meets && first_pass.is_none() {
            first_pass = Some(n);
        }
        if take || chosen.is_none() {
            chosen_count = Some(n);
            chosen = Some((n, metric, image, energy, peak));
        }
    }

    let row = &mut out.row;
    row.atoms_available = outcome.atoms.len();
    row.met_criterion_at = first_pass;
    row.improvement_exhausted = outcome.improvement_exhausted;
    row.fit_renders = outcome.renders;
    match chosen {
        Some((n, metric, image, energy, peak)) => {
            row.atoms = n;
            fill_quality(row, &metric);
            detail(
                target,
                blend.name(),
                profile.name(),
                ATOM_ARM,
                n,
                &metric,
                &mut out.probes,
                &mut out.bins,
            );
            row.residual_energy = energy;
            row.residual_peak = peak;
            row.delta_to_control_px = metric.transition_10_90 - h.control_width;
            row.delta_to_control_peak_frac = metric.peak_frac - h.control_peak;
            let (slice, reading) = prefix(&outcome.atoms, n);
            row.reading = reading;
            let shifted_image =
                render_or_background(slice, &shifted, blend, &settings, request.tile)?;
            let shifted_metric = measure_target_shifted(
                &shifted_image,
                target,
                &request.criterion,
                Vec2::new(-GRID_SHIFT, -GRID_SHIFT),
            )?;
            row.shifted_grid_transition_10_90_px = shifted_metric.transition_10_90;
            row.overfit_gap_px =
                (metric.transition_10_90 - shifted_metric.transition_10_90).abs();
            let cost = tile_cost(slice, &spec, blend, &settings, request.tile)?;
            row.tile_atom_pairs = cost.tile_atom_pairs;
            row.candidates_tested = cost.candidates_tested;
            row.kernel_evaluations = cost.kernel_evaluations;
            row.max_tile_list = cost.max_tile_list;
            row.mean_tile_list = cost.mean_tile_list;
            row.box_over_ellipse_area = cost.box_over_ellipse_area;
            out.timing = Some(Timing {
                atoms: slice.to_vec(),
                spec: spec.clone(),
                blend,
                settings,
            });
            out.sheets.push(Sheet {
                name: format!("{}-{}-{}-atoms", target.name, blend.name(), profile.name()),
                image: image.clone(),
                normalized: false,
            });
            out.sheets.push(Sheet {
                name: format!(
                    "{}-{}-{}-difference",
                    target.name,
                    blend.name(),
                    profile.name()
                ),
                image: truth.difference(&image)?,
                normalized: true,
            });
            if request.sensitivity {
                out.sensitivity = sensitivity_rows(
                    target, slice, &spec, blend, profile, &settings, request,
                )?;
            }
        }
        None => (),
    }

    if request.tiled_field {
        let (placed, reading) = match chosen_count {
            Some(n) => prefix(&outcome.atoms, n),
            None => prefix(&outcome.atoms, outcome.atoms.len()),
        };
        let field = tiled_atoms(placed, target.viewport, FIELD_COPIES);
        let field_spec = camera(FIELD_SIZE, FIELD_SIZE);
        let mut field_row = blank_row(target, blend.name(), profile.name(), ATOM_ARM, TILED);
        field_row.atoms = field.len();
        field_row.atoms_available = field.len();
        field_row.reading = reading;
        field_row.quality_measured = false;
        field_row.improvement_exhausted = outcome.improvement_exhausted;
        field_row.fit_renders = outcome.renders;
        let cost = tile_cost(&field, &field_spec, blend, &settings, request.tile)?;
        field_row.tile_atom_pairs = cost.tile_atom_pairs;
        field_row.candidates_tested = cost.candidates_tested;
        field_row.kernel_evaluations = cost.kernel_evaluations;
        field_row.max_tile_list = cost.max_tile_list;
        field_row.mean_tile_list = cost.mean_tile_list;
        field_row.box_over_ellipse_area = cost.box_over_ellipse_area;
        out.field = Some((
            field_row,
            Timing {
                atoms: field,
                spec: field_spec,
                blend,
                settings,
            },
        ));
    }
    Ok(out)
}

fn time_row(row: &mut Row, timing: &Timing, request: &ProbeRequest) -> Result<()> {
    let (mean, ci) = timed(
        &timing.atoms,
        &timing.spec,
        timing.blend,
        &timing.settings,
        request.tile,
        request.repeats,
    )?;
    row.render_ns_mean = mean;
    row.render_ns_ci95 = ci;
    Ok(())
}

fn measure_timings(heads: &mut [Head], cells: &mut [Cell], request: &ProbeRequest) -> Result<()> {
    for h in heads.iter_mut() {
        for (row, arm) in h.rows.iter_mut().zip(AnalyticArm::ALL) {
            let mut budget = coverage_budget(request.cell_limit);
            let started = std::time::Instant::now();
            let _ = render_analytic(&h.target, arm, request.quadrature_depth, &mut budget)?;
            row.render_ns_mean = started.elapsed().as_nanos() as f64;
        }
    }
    for c in cells.iter_mut() {
        match c.timing.as_ref() {
            Some(timing) => time_row(&mut c.row, timing, request)?,
            None => (),
        }
        match c.field.as_mut() {
            Some((row, timing)) => time_row(row, timing, request)?,
            None => (),
        }
    }
    Ok(())
}

pub fn run(request: &ProbeRequest) -> Result<Report> {
    let all = Target::<f64>::all();
    let mut heads = sweep::map(request.concurrency, &request.targets, |index| {
        head(&all, *index, request)
    })?;

    let mut jobs: Vec<(usize, KernelFamily, ProfileChoice)> = Vec::with_capacity(request.cells());
    for position in 0..heads.len() {
        for blend in request.blends.iter() {
            for profile in request.profiles.iter() {
                jobs.push((position, *blend, *profile));
            }
        }
    }
    let mut cells = sweep::map(request.concurrency, &jobs, |job| cell(&heads, job, request))?;

    measure_timings(&mut heads, &mut cells, request)?;

    let mut rows: Vec<Row> = Vec::new();
    let mut sweep: Vec<SweepRow> = Vec::new();
    let mut sensitivity: Vec<SensitivityRow> = Vec::new();
    let mut sheets: Vec<Sheet> = Vec::new();
    let mut probe_rows: Vec<ProbeRow> = Vec::new();
    let mut bin_rows: Vec<BinRow> = Vec::new();
    let mut viewports: Vec<(String, u32, u32)> = Vec::new();
    let mut undecided = 0.0f64;
    let mut control_valid = true;
    let mut grouped: Vec<Vec<Cell>> = heads.iter().map(|_| Vec::new()).collect();
    for c in cells {
        let position = c.head;
        grouped[position].push(c);
    }

    for (h, owned) in heads.into_iter().zip(grouped) {
        viewports.push((h.target.name.to_string(), h.target.viewport.0, h.target.viewport.1));
        undecided += h.undecided;
        control_valid = control_valid && h.control_valid;
        let target_rows_start = rows.len();
        rows.extend(h.rows);
        sheets.extend(h.sheets);
        probe_rows.extend(h.probes);
        bin_rows.extend(h.bins);
        for c in owned {
            sweep.extend(c.sweep);
            probe_rows.extend(c.probes);
            bin_rows.extend(c.bins);
            sheets.extend(c.sheets);
            sensitivity.extend(c.sensitivity);
            rows.push(c.row);
            match c.field {
                Some((row, _)) => rows.push(row),
                None => (),
            }
        }
        let offset = h.sdf_arm_width - SDF_ARM_TRUE_TRANSITION;
        let fwhm_offset = h.sdf_arm_fwhm - SDF_ARM_TRUE_FWHM;
        for r in rows[target_rows_start..].iter_mut() {
            r.instrument_offset_px = offset;
            r.instrument_offset_fwhm_px = fwhm_offset;
        }
    }

    let header = Header {
        hardware: hardware_banner(),
        profile_build: build_profile(),
        precision: "f64",
        method: "splat-raster",
        sort: "global-sort",
        tile: request.tile,
        threads: 1,
        pixel_box_variance: PIXEL_BOX_VARIANCE,
        alpha_cutoff: SharedSettings::<f64>::contract().blend.alpha_cutoff,
        z0: Z0,
        dz: DZ,
        quadrature_depth: request.quadrature_depth,
        undecided_area: undecided,
        field_size: FIELD_SIZE,
        field_copies: FIELD_COPIES,
        criterion: request.criterion,
        viewports,
        control_arm_valid: control_valid,
    };
    Ok(Report {
        header,
        rows,
        sweep,
        sensitivity,
        probes: probe_rows,
        bins: bin_rows,
        sheets,
    })
}

fn sensitivity_rows(
    target: &Target<f64>,
    atoms: &[Atom2d],
    spec: &CameraSpec,
    blend: KernelFamily,
    profile: ProfileChoice,
    settings: &SharedSettings<f64>,
    request: &ProbeRequest,
) -> Result<Vec<SensitivityRow>> {
    let mut out = Vec::new();
    for value in DILATION_SWEEP {
        let mut swept = *settings;
        swept.screen_dilation = value;
        swept.energy_compensation = true;
        let image = render_or_background(atoms, spec, blend, &swept, request.tile)?;
        let metric = measure_target(&image, target, &request.criterion)?;
        out.push(SensitivityRow {
            target: target.name.to_string(),
            blend: blend.name().to_string(),
            profile: profile.name().to_string(),
            knob: "screen_dilation".to_string(),
            value,
            transition_10_90_px: metric.transition_10_90,
            contrast_frac: metric.contrast_frac,
            peak_frac: metric.peak_frac,
            fwhm_px: metric.fwhm,
        });
    }
    for value in CUTOFF_SWEEP {
        let mut swept = *settings;
        swept.blend.alpha_cutoff = value;
        let image = render_or_background(atoms, spec, blend, &swept, request.tile)?;
        let metric = measure_target(&image, target, &request.criterion)?;
        out.push(SensitivityRow {
            target: target.name.to_string(),
            blend: blend.name().to_string(),
            profile: profile.name().to_string(),
            knob: "alpha_cutoff".to_string(),
            value,
            transition_10_90_px: metric.transition_10_90,
            contrast_frac: metric.contrast_frac,
            peak_frac: metric.peak_frac,
            fwhm_px: metric.fwhm,
        });
    }
    Ok(out)
}
