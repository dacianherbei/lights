use crate::error::Result;
use crate::fit::edge::{Criterion, SDF_ARM_TRUE_TRANSITION};
use crate::fit::greedy::{PREFIX_SHORTFALL, Reading};
use crate::image::{FloatImage, write_png, write_png_normalized};
use std::path::Path;

pub const NATURAL: &str = "natural";
pub const TILED: &str = "tiled";
pub const ATOM_ARM: &str = "atoms";
pub const NO_AXIS: &str = "-";

#[derive(Clone, Debug, PartialEq)]
pub struct Header {
    pub hardware: String,
    pub profile_build: &'static str,
    pub precision: &'static str,
    pub method: &'static str,
    pub sort: &'static str,
    pub tile: u32,
    pub threads: usize,
    pub pixel_box_variance: f64,
    pub alpha_cutoff: f64,
    pub z0: f64,
    pub dz: f64,
    pub quadrature_depth: u32,
    pub undecided_area: f64,
    pub field_size: u32,
    pub field_copies: usize,
    pub criterion: Criterion,
    pub viewports: Vec<(String, u32, u32)>,
    pub control_arm_valid: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub target: String,
    pub blend: String,
    pub profile: String,
    pub arm: String,
    pub field: String,
    pub atoms: usize,
    pub atoms_available: usize,
    pub reading: Reading,
    pub met_criterion_at: Option<usize>,
    pub primitives: usize,
    pub description_bytes: usize,
    pub transition_10_90_px: f64,
    pub contrast_frac: f64,
    pub peak_frac: f64,
    pub fwhm_px: f64,
    pub fwhm_error_px: f64,
    pub meets_criterion: bool,
    pub quality_measured: bool,
    pub improvement_exhausted: bool,
    pub fit_renders: u64,
    pub delta_to_control_px: f64,
    pub delta_to_control_peak_frac: f64,
    pub instrument_offset_px: f64,
    pub instrument_offset_fwhm_px: f64,
    pub apex_displacement_px: f64,
    pub interior_ripple: f64,
    pub residual_energy: f64,
    pub residual_peak: f64,
    pub thin_bins: u32,
    pub samples: u32,
    pub shifted_grid_transition_10_90_px: f64,
    pub overfit_gap_px: f64,
    pub tile_atom_pairs: u64,
    pub candidates_tested: u64,
    pub kernel_evaluations: u64,
    pub max_tile_list: u32,
    pub mean_tile_list: f64,
    pub box_over_ellipse_area: f64,
    pub render_ns_mean: f64,
    pub render_ns_ci95: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SweepRow {
    pub target: String,
    pub blend: String,
    pub profile: String,
    pub atoms: usize,
    pub transition_10_90_px: f64,
    pub contrast_frac: f64,
    pub peak_frac: f64,
    pub fwhm_px: f64,
    pub residual_energy: f64,
    pub thin_bins: u32,
    pub meets_criterion: bool,
    pub reading: Reading,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SensitivityRow {
    pub target: String,
    pub blend: String,
    pub profile: String,
    pub knob: String,
    pub value: f64,
    pub transition_10_90_px: f64,
    pub contrast_frac: f64,
    pub peak_frac: f64,
    pub fwhm_px: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProbeRow {
    pub target: String,
    pub blend: String,
    pub profile: String,
    pub arm: String,
    pub atoms: usize,
    pub feature: u16,
    pub kind: String,
    pub samples: u32,
    pub thin_bins: u32,
    pub transition_10_90_px: f64,
    pub sdf_90: f64,
    pub sdf_10: f64,
    pub contrast: f64,
    pub target_contrast: f64,
    pub peak: f64,
    pub fwhm_px: f64,
    pub plateau_low: f64,
    pub plateau_high: f64,
    pub apex_displacement_px: f64,
    pub interior_ripple: f64,
    pub judged: bool,
    pub meets: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BinRow {
    pub target: String,
    pub blend: String,
    pub profile: String,
    pub arm: String,
    pub feature: u16,
    pub sdf: f64,
    pub mean: f64,
    pub samples: u32,
}

#[derive(Clone, Debug)]
pub struct Sheet {
    pub name: String,
    pub image: FloatImage<f64>,
    pub normalized: bool,
}

#[derive(Clone, Debug)]
pub struct Report {
    pub header: Header,
    pub rows: Vec<Row>,
    pub sweep: Vec<SweepRow>,
    pub sensitivity: Vec<SensitivityRow>,
    pub probes: Vec<ProbeRow>,
    pub bins: Vec<BinRow>,
    pub sheets: Vec<Sheet>,
}

pub const OVERFIT_FLAG_PX: f64 = 0.2;

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

fn at(v: Option<usize>) -> String {
    match v {
        Some(n) => n.to_string(),
        None => "never".to_string(),
    }
}

pub const ROW_COLUMNS: [&str; 39] = [
    "target",
    "blend",
    "profile",
    "arm",
    "field",
    "atoms",
    "atoms_available",
    "met_criterion_at",
    "primitives",
    "description_bytes",
    "transition_10_90_px",
    "contrast_frac",
    "peak_frac",
    "fwhm_px",
    "fwhm_error_px",
    "meets_criterion",
    "quality_measured",
    "improvement_exhausted",
    "fit_renders",
    "delta_to_control_px",
    "delta_to_control_peak_frac",
    "instrument_offset_px",
    "instrument_offset_fwhm_px",
    "apex_displacement_px",
    "interior_ripple",
    "residual_energy",
    "residual_peak",
    "thin_bins",
    "samples",
    "shifted_grid_transition_10_90_px",
    "overfit_gap_px",
    "tile_atom_pairs",
    "candidates_tested",
    "kernel_evaluations",
    "max_tile_list",
    "mean_tile_list",
    "box_over_ellipse_area",
    "render_ns_mean",
    "render_ns_ci95",
];

pub const SWEEP_COLUMNS: [&str; 12] = [
    "target",
    "blend",
    "profile",
    "atoms",
    "transition_10_90_px",
    "contrast_frac",
    "peak_frac",
    "fwhm_px",
    "residual_energy",
    "thin_bins",
    "meets_criterion",
    "reading",
];

pub const SENSITIVITY_COLUMNS: [&str; 9] = [
    "target",
    "blend",
    "profile",
    "knob",
    "value",
    "transition_10_90_px",
    "contrast_frac",
    "peak_frac",
    "fwhm_px",
];

fn row_cells(r: &Row) -> Vec<String> {
    vec![
        r.target.clone(),
        r.blend.clone(),
        r.profile.clone(),
        r.arm.clone(),
        r.field.clone(),
        r.atoms.to_string(),
        r.atoms_available.to_string(),
        if r.quality_measured { at(r.met_criterion_at) } else { "n/a".to_string() },
        r.primitives.to_string(),
        r.description_bytes.to_string(),
        number(r.transition_10_90_px),
        number(r.contrast_frac),
        number(r.peak_frac),
        number(r.fwhm_px),
        number(r.fwhm_error_px),
        flag(r.meets_criterion).to_string(),
        flag(r.quality_measured).to_string(),
        flag(r.improvement_exhausted).to_string(),
        r.fit_renders.to_string(),
        number(r.delta_to_control_px),
        number(r.delta_to_control_peak_frac),
        number(r.instrument_offset_px),
        number(r.instrument_offset_fwhm_px),
        number(r.apex_displacement_px),
        number(r.interior_ripple),
        number(r.residual_energy),
        number(r.residual_peak),
        r.thin_bins.to_string(),
        r.samples.to_string(),
        number(r.shifted_grid_transition_10_90_px),
        number(r.overfit_gap_px),
        r.tile_atom_pairs.to_string(),
        r.candidates_tested.to_string(),
        r.kernel_evaluations.to_string(),
        r.max_tile_list.to_string(),
        number(r.mean_tile_list),
        number(r.box_over_ellipse_area),
        number(r.render_ns_mean),
        number(r.render_ns_ci95),
    ]
}

fn sweep_cells(r: &SweepRow) -> Vec<String> {
    vec![
        r.target.clone(),
        r.blend.clone(),
        r.profile.clone(),
        r.atoms.to_string(),
        number(r.transition_10_90_px),
        number(r.contrast_frac),
        number(r.peak_frac),
        number(r.fwhm_px),
        number(r.residual_energy),
        r.thin_bins.to_string(),
        flag(r.meets_criterion).to_string(),
        r.reading.name().to_string(),
    ]
}

pub const PROBE_COLUMNS: [&str; 22] = [
    "target",
    "blend",
    "profile",
    "arm",
    "atoms",
    "feature",
    "kind",
    "samples",
    "thin_bins",
    "transition_10_90_px",
    "sdf_90",
    "sdf_10",
    "contrast",
    "target_contrast",
    "peak",
    "fwhm_px",
    "plateau_low",
    "plateau_high",
    "apex_displacement_px",
    "interior_ripple",
    "judged",
    "meets",
];

pub const BIN_COLUMNS: [&str; 8] = [
    "target", "blend", "profile", "arm", "feature", "sdf", "mean", "samples",
];

fn probe_cells(r: &ProbeRow) -> Vec<String> {
    vec![
        r.target.clone(),
        r.blend.clone(),
        r.profile.clone(),
        r.arm.clone(),
        r.atoms.to_string(),
        r.feature.to_string(),
        r.kind.clone(),
        r.samples.to_string(),
        r.thin_bins.to_string(),
        number(r.transition_10_90_px),
        number(r.sdf_90),
        number(r.sdf_10),
        number(r.contrast),
        number(r.target_contrast),
        number(r.peak),
        number(r.fwhm_px),
        number(r.plateau_low),
        number(r.plateau_high),
        number(r.apex_displacement_px),
        number(r.interior_ripple),
        flag(r.judged).to_string(),
        flag(r.meets).to_string(),
    ]
}

fn bin_cells(r: &BinRow) -> Vec<String> {
    vec![
        r.target.clone(),
        r.blend.clone(),
        r.profile.clone(),
        r.arm.clone(),
        r.feature.to_string(),
        number(r.sdf),
        number(r.mean),
        r.samples.to_string(),
    ]
}

fn sensitivity_cells(r: &SensitivityRow) -> Vec<String> {
    vec![
        r.target.clone(),
        r.blend.clone(),
        r.profile.clone(),
        r.knob.clone(),
        number(r.value),
        number(r.transition_10_90_px),
        number(r.contrast_frac),
        number(r.peak_frac),
        number(r.fwhm_px),
    ]
}

pub fn header_lines(h: &Header) -> Vec<String> {
    let mut out = vec![
        format!("hardware\t{}", h.hardware),
        format!("build\t{}", h.profile_build),
        format!("precision\t{}", h.precision),
        format!("method\t{}", h.method),
        format!("sort\t{}", h.sort),
        format!("tile\t{}", h.tile),
        format!("threads\t{}", h.threads),
        format!("PIXEL_BOX_VARIANCE\t{}", h.pixel_box_variance),
        format!("alpha_cutoff\t{}", h.alpha_cutoff),
        format!("Z0\t{}", h.z0),
        format!("DZ\t{}", h.dz),
        format!("quadrature_depth\t{}", h.quadrature_depth),
        format!("undecided_area\t{}", h.undecided_area),
        format!("field\t{}x{} holding {} copies", h.field_size, h.field_size, h.field_copies),
        format!("criterion.max_transition_px\t{}", h.criterion.max_transition),
        format!("criterion.min_contrast_frac\t{}", h.criterion.min_contrast_frac),
        format!("criterion.min_peak_frac\t{}", h.criterion.min_peak_frac),
        format!("criterion.target_fwhm_px\t{}", h.criterion.target_fwhm),
        format!("criterion.max_fwhm_error_px\t{}", h.criterion.max_fwhm_error),
        format!("criterion.max_interior_ripple\t{}", h.criterion.max_interior_ripple),
        format!("control_arm_valid\t{}", flag(h.control_arm_valid)),
    ];
    for (name, width, height) in h.viewports.iter() {
        out.push(format!("viewport.{name}\t{width}x{height}"));
    }
    out
}

fn table(
    path: &Path,
    header: &Header,
    columns: &[&str],
    rows: Vec<Vec<String>>,
) -> Result<()> {
    let mut text = String::new();
    for line in header_lines(header) {
        text.push('#');
        text.push_str(&line);
        text.push('\n');
    }
    text.push_str(&columns.join("\t"));
    text.push('\n');
    for cells in rows {
        text.push_str(&cells.join("\t"));
        text.push('\n');
    }
    std::fs::write(path, text)?;
    Ok(())
}

pub fn write_report(dir: &Path, report: &Report) -> Result<Vec<std::path::PathBuf>> {
    std::fs::create_dir_all(dir)?;
    let mut written: Vec<std::path::PathBuf> = Vec::new();
    for (name, columns, rows) in [
        (
            "report.tsv",
            &ROW_COLUMNS[..],
            report.rows.iter().map(row_cells).collect::<Vec<Vec<String>>>(),
        ),
        (
            "sweep.tsv",
            &SWEEP_COLUMNS[..],
            report.sweep.iter().map(sweep_cells).collect(),
        ),
        (
            "sensitivity.tsv",
            &SENSITIVITY_COLUMNS[..],
            report.sensitivity.iter().map(sensitivity_cells).collect(),
        ),
        (
            "probes.tsv",
            &PROBE_COLUMNS[..],
            report.probes.iter().map(probe_cells).collect(),
        ),
        (
            "bins.tsv",
            &BIN_COLUMNS[..],
            report.bins.iter().map(bin_cells).collect(),
        ),
    ] {
        let path = dir.join(name);
        table(&path, &report.header, columns, rows)?;
        written.push(path);
    }
    for sheet in report.sheets.iter() {
        let path = dir.join(format!("{}.png", sheet.name));
        if sheet.normalized {
            write_png_normalized(&path, &sheet.image)?;
        } else {
            write_png(&path, &sheet.image.to_clamped_u8())?;
        }
    }
    Ok(written)
}

pub fn print_header(header: &Header) {
    println!("=== P2 header: every constant below moves the numbers ===");
    for line in header_lines(header) {
        let mut parts = line.splitn(2, '\t');
        let name = parts.next().unwrap_or("");
        let value = parts.next().unwrap_or("");
        println!("{name:<34} {value}");
    }
}

pub fn print_rows(rows: &[Row]) {
    println!(
        "\n{:<15} {:<19} {:<11} {:<14} {:<7} {:>5} {:>6} {:>8} {:>7} {:>7} {:>8} {:>5} {:>9} {:>9} {:>7} {:>10} {:>8}",
        "target",
        "blend",
        "profile",
        "arm",
        "field",
        "atoms",
        "first@",
        "10-90px",
        "contr",
        "peak",
        "fwhm",
        "meets",
        "d-ctrl px",
        "d-ctrl pk",
        "overfit",
        "tile pairs",
        "reading"
    );
    for r in rows.iter() {
        println!(
            "{:<15} {:<19} {:<11} {:<14} {:<7} {:>5} {:>6} {:>8} {:>7} {:>7} {:>8} {:>5} {:>9} {:>9} {:>7} {:>10} {:>8}",
            r.target,
            r.blend,
            r.profile,
            r.arm,
            r.field,
            r.atoms,
            if r.quality_measured { at(r.met_criterion_at) } else { "n/a".to_string() },
            number(r.transition_10_90_px),
            number(r.contrast_frac),
            number(r.peak_frac),
            number(r.fwhm_px),
            if r.quality_measured { flag(r.meets_criterion) } else { "n/a" },
            number(r.delta_to_control_px),
            number(r.delta_to_control_peak_frac),
            number(r.overfit_gap_px),
            r.tile_atom_pairs,
            r.reading.name()
        );
    }
    println!(
        "\nd-ctrl is the headline, not the absolute width: nothing can beat \
         exact box coverage of the same shape. instrument_offset_px in \
         report.tsv is the analytic-sdf arm's measured width minus its closed \
         form of exactly {SDF_ARM_TRUE_TRANSITION} px, which is this target's \
         own scatter quantisation and applies to every row of that target."
    );
}

pub fn print_sweep(sweep: &[SweepRow]) {
    println!(
        "\n=== quality against atom count, both sides of the first pass ===\n{:<21} {:<19} {:<11} {:>6} {:>9} {:>8} {:>9} {:>16} {:>6} {:>8}",
        "target", "blend", "profile", "atoms", "10-90px", "contr", "fwhm", "residual energy", "meets", "reading"
    );
    for r in sweep.iter() {
        println!(
            "{:<21} {:<19} {:<11} {:>6} {:>9} {:>8} {:>9} {:>16} {:>6} {:>8}",
            r.target,
            r.blend,
            r.profile,
            r.atoms,
            number(r.transition_10_90_px),
            number(r.contrast_frac),
            number(r.fwhm_px),
            number(r.residual_energy),
            flag(r.meets_criterion),
            r.reading.name()
        );
    }
    if sweep.iter().any(|r| r.reading == Reading::Prefix) {
        println!("\n{PREFIX_SHORTFALL}");
    }
}

pub fn print_sensitivity(rows: &[SensitivityRow]) {
    println!(
        "\n=== sensitivity: a constant that sets the answer must be visible ===\n{:<21} {:<19} {:<11} {:<16} {:>12} {:>9} {:>8} {:>8}",
        "target", "blend", "profile", "knob", "value", "10-90px", "contr", "fwhm"
    );
    for r in rows.iter() {
        println!(
            "{:<21} {:<19} {:<11} {:<16} {:>12} {:>9} {:>8} {:>8}",
            r.target,
            r.blend,
            r.profile,
            r.knob,
            number(r.value),
            number(r.transition_10_90_px),
            number(r.contrast_frac),
            number(r.fwhm_px)
        );
    }
}

pub fn print_report(report: &Report) {
    print_header(&report.header);
    print_rows(&report.rows);
    print_sweep(&report.sweep);
    print_sensitivity(&report.sensitivity);
}

pub fn never_met(rows: &[Row]) -> Vec<&Row> {
    rows.iter()
        .filter(|r| {
            r.arm == ATOM_ARM && r.quality_measured && r.met_criterion_at.is_none()
        })
        .collect()
}

pub fn overfit_flagged(rows: &[Row]) -> Vec<&Row> {
    rows.iter()
        .filter(|r| r.overfit_gap_px.is_finite() && r.overfit_gap_px > OVERFIT_FLAG_PX)
        .collect()
}
