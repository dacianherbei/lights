use crate::error::{LightsError, Result};
use rayon::prelude::*;
use crate::fit::coverage::{DEFAULT_CELL_LIMIT, DEFAULT_MAX_DEPTH, coverage_budget, exact_coverage};
use crate::fit::scale::scaled_primitive;
use crate::fit::target::{
    FeatureId, GLYPH_EM, Part, Primitive, PrimitiveSet, SUBPIXEL_X, SUBPIXEL_Y, Target,
};
use crate::image::FloatImage;
use crate::numerics::vector::{Rgb, Vec2};
use crate::settings::PIXEL_BOX_VARIANCE;

pub const RECONSTRUCTION_SIGMA: f64 = 0.7;
pub const FINE_PER_UNIT: f64 = 48.0;
pub const SUPPORT_SIGMAS: f64 = 6.0;
pub const EIGHT_BIT_STEP: f64 = 1.0 / 255.0;
pub const EDGE_FLOOR: f64 = 0.041632258331775196;
pub const EM_SIZES: [u32; 13] = [8, 10, 12, 14, 16, 20, 24, 32, 48, 64, 96, 128, 200];
pub const DIDONE_STEM_HALF_WIDTH: f64 = 1.0;
pub const DIDONE_HAIRLINE_HALF_WIDTH: f64 = 0.1;
pub const PHASES: [f64; 4] = [0.0, 0.25, 0.5, 0.75];
pub const MATCH_TOLERANCE: f64 = 0.01;
pub const BROKEN_FACTOR: f64 = 3.0;
pub const MAX_FINE_RATIO: f64 = 0.25;
pub const HAIRLINE_INK_SWING: f64 = 0.3611;
pub const THICKNESS_PERCENTILE: f64 = 0.02;
pub const MIN_OPPOSITION: f64 = 0.7;
pub const MIN_GAP_OPPOSITION: f64 = 0.97;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Verdict {
    Holds,
    Marginal,
    Broken,
}

impl Verdict {
    pub fn name(self) -> &'static str {
        match self {
            Verdict::Holds => "holds",
            Verdict::Marginal => "marginal",
            Verdict::Broken => "broken",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FineAtoms {
    pub centres: Vec<Vec2<f64>>,
    pub weights: Vec<f64>,
    pub fine_per_unit: f64,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Residual {
    pub max_abs: f64,
    pub mean_abs: f64,
    pub support: usize,
    pub ink_ratio: f64,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct GlyphScaleRow {
    pub em_px: u32,
    pub px_per_unit: f64,
    pub thinnest_px: f64,
    pub inflation: f64,
    pub inflated: Residual,
    pub scaled_only: Residual,
    pub ink_low: f64,
    pub ink_high: f64,
}

fn worse(a: Option<Residual>, b: Residual) -> Residual {
    match a {
        Some(r) if r.max_abs >= b.max_abs => r,
        Some(_) | None => b,
    }
}

fn part(id: u16, shape: Primitive<f64>) -> Part<f64> {
    Part {
        id: FeatureId(id),
        shape,
    }
}

fn p(x: f64, y: f64) -> Vec2<f64> {
    Vec2::new(x + SUBPIXEL_X, y + SUBPIXEL_Y)
}

fn stroke(id: u16, a: Vec2<f64>, b: Vec2<f64>, half_width: f64) -> Part<f64> {
    part(id, Primitive::Segment { a, b, half_width })
}

pub fn didone_n() -> Target<f64> {
    let h = DIDONE_HAIRLINE_HALF_WIDTH;
    Target::ink_on_paper(
        "didone-n",
        (32, 32),
        vec![
            stroke(0, p(11.0, 10.0), p(11.0, 22.0), h),
            stroke(1, p(21.0, 10.0), p(21.0, 22.0), h),
            stroke(2, p(11.0, 10.0), p(21.0, 22.0), DIDONE_STEM_HALF_WIDTH),
            stroke(3, p(9.5, 22.0), p(12.5, 22.0), h),
            stroke(4, p(19.5, 10.0), p(22.5, 10.0), h),
        ],
        Vec::new(),
        Vec::new(),
    )
}

pub fn aligned_edge() -> Target<f64> {
    Target::ink_on_paper(
        "aligned-edge",
        (16, 16),
        vec![part(
            0,
            Primitive::half_plane(Vec2::new(8.0, 0.0), Vec2::new(-1.0, 0.0)),
        )],
        Vec::new(),
        Vec::new(),
    )
}

pub fn translated_primitive(shape: &Primitive<f64>, offset: Vec2<f64>) -> Primitive<f64> {
    match shape {
        Primitive::HalfPlane { point, normal } => Primitive::HalfPlane {
            point: *point + offset,
            normal: *normal,
        },
        Primitive::Segment { a, b, half_width } => Primitive::Segment {
            a: *a + offset,
            b: *b + offset,
            half_width: *half_width,
        },
        Primitive::Arc {
            centre,
            radius,
            from,
            to,
            half_width,
        } => Primitive::Arc {
            centre: *centre + offset,
            radius: *radius,
            from: *from,
            to: *to,
            half_width: *half_width,
        },
        Primitive::Disc { centre, radius } => Primitive::Disc {
            centre: *centre + offset,
            radius: *radius,
        },
    }
}

fn mapped_parts(parts: &[Part<f64>], scale: f64, offset: Vec2<f64>) -> Vec<Part<f64>> {
    parts
        .iter()
        .map(|q| part(q.id.0, translated_primitive(&scaled_primitive(&q.shape, scale), offset)))
        .collect()
}

pub fn mapped_target(
    target: &Target<f64>,
    scale: f64,
    offset: Vec2<f64>,
    viewport: (u32, u32),
) -> Target<f64> {
    Target {
        name: target.name,
        viewport,
        set: PrimitiveSet {
            ink: target.set.ink,
            parts: mapped_parts(&target.set.parts, scale, offset),
            holes: mapped_parts(&target.set.holes, scale, offset),
            regions: target.set.regions.iter().map(|r| r.mapped(scale, offset)).collect(),
        },
        paper: target.paper,
        probes: Vec::new(),
    }
}

pub fn viewport_at(target: &Target<f64>, px_per_unit: f64) -> (u32, u32) {
    (
        (target.viewport.0 as f64 * px_per_unit).ceil() as u32,
        (target.viewport.1 as f64 * px_per_unit).ceil() as u32,
    )
}

fn coverage(target: &Target<f64>) -> Result<FloatImage<f64>> {
    let mut budget = coverage_budget(DEFAULT_CELL_LIMIT);
    let (image, _) = exact_coverage(target, DEFAULT_MAX_DEPTH, &mut budget)?;
    Ok(image)
}

pub fn reference_at(target: &Target<f64>, px_per_unit: f64, shift_px: Vec2<f64>) -> Result<FloatImage<f64>> {
    let viewport = viewport_at(target, px_per_unit);
    coverage(&mapped_target(target, px_per_unit, shift_px, viewport))
}

pub fn fine_atoms(target: &Target<f64>, fine_per_unit: f64, margin_units: f64) -> Result<FineAtoms> {
    let margin_fine = (margin_units * fine_per_unit).ceil();
    let inner = viewport_at(target, fine_per_unit);
    let viewport = (
        inner.0 + 2 * margin_fine as u32,
        inner.1 + 2 * margin_fine as u32,
    );
    let fine = coverage(&mapped_target(
        target,
        fine_per_unit,
        Vec2::new(margin_fine, margin_fine),
        viewport,
    ))?;
    let mut centres = Vec::new();
    let mut weights = Vec::new();
    for j in 0..fine.height {
        for i in 0..fine.width {
            let c = fine.get(i, j).x;
            if c > 0.0 {
                centres.push(Vec2::new(
                    (i as f64 + 0.5 - margin_fine) / fine_per_unit,
                    (j as f64 + 0.5 - margin_fine) / fine_per_unit,
                ));
                weights.push(c);
            }
        }
    }
    Ok(FineAtoms {
        centres,
        weights,
        fine_per_unit,
    })
}

pub fn scaled_variance(atoms: &FineAtoms, px_per_unit: f64) -> f64 {
    let r = px_per_unit / atoms.fine_per_unit;
    RECONSTRUCTION_SIGMA * RECONSTRUCTION_SIGMA * r * r
}

pub fn inflation_for(atoms: &FineAtoms, px_per_unit: f64) -> Result<f64> {
    let r = px_per_unit / atoms.fine_per_unit;
    let inflation = PIXEL_BOX_VARIANCE * (1.0 + r * r) - scaled_variance(atoms, px_per_unit);
    if inflation < 0.0 {
        return Err(LightsError::SizeMismatch {
            a: (px_per_unit.ceil() as u32, atoms.fine_per_unit as u32),
            b: (1, 1),
        });
    }
    Ok(inflation)
}

pub fn predicted_at(
    atoms: &FineAtoms,
    px_per_unit: f64,
    viewport: (u32, u32),
    inflation: f64,
    shift_px: Vec2<f64>,
) -> FloatImage<f64> {
    let variance = scaled_variance(atoms, px_per_unit) + inflation;
    let r = px_per_unit / atoms.fine_per_unit;
    let cell_area = r * r;
    let norm = cell_area / (2.0 * std::f64::consts::PI * variance);
    let reach = SUPPORT_SIGMAS * variance.sqrt();
    let mut acc = vec![0.0f64; (viewport.0 * viewport.1) as usize];
    for (centre, weight) in atoms.centres.iter().zip(atoms.weights.iter()) {
        let c = *centre * px_per_unit + shift_px;
        let x0 = (c.x - reach - 0.5).floor().max(0.0) as i64;
        let x1 = ((c.x + reach - 0.5).ceil() as i64).min(viewport.0 as i64 - 1);
        let y0 = (c.y - reach - 0.5).floor().max(0.0) as i64;
        let y1 = ((c.y + reach - 0.5).ceil() as i64).min(viewport.1 as i64 - 1);
        for py in y0..=y1 {
            let dy = py as f64 + 0.5 - c.y;
            for px in x0..=x1 {
                let dx = px as f64 + 0.5 - c.x;
                let g = (-(dx * dx + dy * dy) / (2.0 * variance)).exp();
                acc[(py as u32 * viewport.0 + px as u32) as usize] += weight * norm * g;
            }
        }
    }
    FloatImage {
        width: viewport.0,
        height: viewport.1,
        pixels: acc.into_iter().map(Rgb::splat).collect(),
    }
}

pub fn residual(reference: &FloatImage<f64>, predicted: &FloatImage<f64>) -> Result<Residual> {
    if reference.width != predicted.width || reference.height != predicted.height {
        return Err(LightsError::SizeMismatch {
            a: (reference.width, reference.height),
            b: (predicted.width, predicted.height),
        });
    }
    let mut max_abs = 0.0f64;
    let mut sum_abs = 0.0f64;
    let mut support = 0usize;
    let mut ink_reference = 0.0f64;
    let mut ink_predicted = 0.0f64;
    for (a, b) in reference.pixels.iter().zip(predicted.pixels.iter()) {
        ink_reference += a.x;
        ink_predicted += b.x;
        if a.x > 0.0 || b.x > EIGHT_BIT_STEP * 0.5 {
            let d = (a.x - b.x).abs();
            max_abs = max_abs.max(d);
            sum_abs += d;
            support += 1;
        }
    }
    Ok(Residual {
        max_abs,
        mean_abs: if support > 0 { sum_abs / support as f64 } else { 0.0 },
        support,
        ink_ratio: if ink_reference > 0.0 { ink_predicted / ink_reference } else { 0.0 },
    })
}

pub fn thinnest_half_width(target: &Target<f64>) -> f64 {
    let mut best = f64::INFINITY;
    for q in target.set.parts.iter().chain(target.set.holes.iter()) {
        let h = match q.shape {
            Primitive::HalfPlane { .. } => f64::INFINITY,
            Primitive::Segment { half_width, .. } => half_width,
            Primitive::Arc { half_width, .. } => half_width,
            Primitive::Disc { radius, .. } => radius,
        };
        best = best.min(h);
    }
    best
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Side {
    Ink,
    Paper,
}

impl Side {
    pub fn min_opposition(self) -> f64 {
        match self {
            Side::Ink => MIN_OPPOSITION,
            Side::Paper => MIN_GAP_OPPOSITION,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Crossing {
    pub at: Vec2<f64>,
    pub width: f64,
    pub opposition: f64,
}

pub fn stroke_widths(target: &Target<f64>, samples_per_unit: f64) -> Vec<f64> {
    widths_on(target, samples_per_unit, Side::Ink)
}

pub fn gap_widths(target: &Target<f64>, samples_per_unit: f64) -> Vec<f64> {
    widths_on(target, samples_per_unit, Side::Paper)
}

pub fn widths_on(target: &Target<f64>, samples_per_unit: f64, side: Side) -> Vec<f64> {
    let mut out: Vec<f64> = crossings(target, samples_per_unit, side, side.min_opposition())
        .into_iter()
        .map(|c| c.width)
        .collect();
    out.sort_by(f64::total_cmp);
    out
}

pub fn crossings(target: &Target<f64>, samples_per_unit: f64, side: Side, min_opposition: f64) -> Vec<Crossing> {
    let w = (target.viewport.0 as f64 * samples_per_unit).ceil() as usize;
    let h = (target.viewport.1 as f64 * samples_per_unit).ceil() as usize;
    let feet: Vec<Option<(Vec2<f64>, Vec2<f64>)>> = (0..w * h)
        .into_par_iter()
        .map(|idx| {
            let (i, j) = (idx % w, idx / w);
            let p = Vec2::new((i as f64 + 0.5) / samples_per_unit, (j as f64 + 0.5) / samples_per_unit);
            let (signed, normal) = target.nearest_surface(p);
            if (signed < 0.0) == (side == Side::Ink) { Some((p - normal * signed, normal)) } else { None }
        })
        .collect();
    (0..w * h)
        .into_par_iter()
        .flat_map_iter(|idx| {
            let (i, j) = (idx % w, idx / w);
            let mut local = Vec::new();
            if let Some((fp, np)) = feet[idx] {
                for (di, dj) in [(1i64, 0i64), (0, 1), (1, 1), (-1, 1)] {
                    let (qi, qj) = (i as i64 + di, j as i64 + dj);
                    if qi < 0 || qi >= w as i64 || qj >= h as i64 {
                        continue;
                    }
                    let Some((fq, nq)) = feet[qj as usize * w + qi as usize] else {
                        continue;
                    };
                    if np.dot(nq) < -min_opposition {
                        let across = fp - fq;
                        local.push(Crossing {
                            at: Vec2::new(
                                (i as f64 + 0.5 + 0.5 * di as f64) / samples_per_unit,
                                (j as f64 + 0.5 + 0.5 * dj as f64) / samples_per_unit,
                            ),
                            width: 0.5 * (across.dot(np).abs() + across.dot(nq).abs()),
                            opposition: -np.dot(nq),
                        });
                    }
                }
            }
            local.into_iter()
        })
        .collect()
}

pub fn thinnest_feature(target: &Target<f64>, samples_per_unit: f64) -> f64 {
    thinnest_of(&stroke_widths(target, samples_per_unit))
}

pub fn thinnest_gap(target: &Target<f64>, samples_per_unit: f64) -> f64 {
    thinnest_of(&gap_widths(target, samples_per_unit))
}

pub fn thinnest_of(sorted: &[f64]) -> f64 {
    if sorted.is_empty() {
        return f64::INFINITY;
    }
    sorted[(THICKNESS_PERCENTILE * (sorted.len() - 1) as f64).floor() as usize]
}

pub enum ScaleProgress<'a> {
    Prepared { rungs: usize, thinnest_units: f64 },
    RungStarted { em_px: u32, index: usize, total: usize },
    PhaseDone { em_px: u32, phase: usize, phases: usize },
    PhaseStep { em_px: u32, phase: usize, what: &'static str },
    Viewport { em_px: u32, width: u32, height: u32 },
    RungDone { row: &'a GlyphScaleRow, index: usize, total: usize },
}

pub fn curve(
    target: &Target<f64>,
    em_units: f64,
    sizes: &[u32],
    fine_per_unit: f64,
) -> Result<Vec<GlyphScaleRow>> {
    curve_reporting(target, em_units, sizes, fine_per_unit, |_| {})
}

pub fn curve_reporting<F: Fn(ScaleProgress) + Send + Sync>(
    target: &Target<f64>,
    em_units: f64,
    sizes: &[u32],
    fine_per_unit: f64,
    report: F,
) -> Result<Vec<GlyphScaleRow>> {
    let atoms = fine_atoms(target, fine_per_unit, 0.0)?;
    let thinnest = thinnest_feature(target, fine_per_unit);
    report(ScaleProgress::Prepared { rungs: sizes.len(), thinnest_units: thinnest });
    let total = sizes.len();
    sizes
        .par_iter()
        .enumerate()
        .map(|(index, &em_px)| {
            report(ScaleProgress::RungStarted { em_px, index, total });
            let row = row_at(target, &atoms, em_units, em_px, thinnest, &report)?;
            report(ScaleProgress::RungDone { row: &row, index, total });
            Ok(row)
        })
        .collect()
}

fn row_at<F: Fn(ScaleProgress) + Send + Sync>(
    target: &Target<f64>,
    atoms: &FineAtoms,
    em_units: f64,
    em_px: u32,
    thinnest: f64,
    report: &F,
) -> Result<GlyphScaleRow> {
    let k = em_px as f64 / em_units;
    let viewport = viewport_at(target, k);
    report(ScaleProgress::Viewport { em_px, width: viewport.0, height: viewport.1 });
    let inflation = inflation_for(atoms, k)?;
    let pairs: Vec<(Residual, Residual)> = PHASES
        .par_iter()
        .enumerate()
        .map(|(pi, &phase)| {
            let shift = Vec2::new(phase, phase);
            let reference = reference_at(target, k, shift)?;
            report(ScaleProgress::PhaseStep { em_px, phase: pi + 1, what: "reference rasterised" });
            let a = residual(&reference, &predicted_at(atoms, k, viewport, inflation, shift))?;
            report(ScaleProgress::PhaseStep { em_px, phase: pi + 1, what: "inflated residual" });
            let b = residual(&reference, &predicted_at(atoms, k, viewport, 0.0, shift))?;
            report(ScaleProgress::PhaseDone { em_px, phase: pi + 1, phases: PHASES.len() });
            Ok((a, b))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut inflated: Option<Residual> = None;
    let mut scaled_only: Option<Residual> = None;
    let mut ink_low = f64::INFINITY;
    let mut ink_high = f64::NEG_INFINITY;
    for (a, b) in pairs {
        ink_low = ink_low.min(a.ink_ratio);
        ink_high = ink_high.max(a.ink_ratio);
        inflated = Some(worse(inflated, a));
        scaled_only = Some(worse(scaled_only, b));
    }
    let (Some(inflated), Some(scaled_only)) = (inflated, scaled_only) else {
        return Err(LightsError::SizeMismatch { a: (em_px, 0), b: (PHASES.len() as u32, 0) });
    };
    Ok(GlyphScaleRow {
        em_px,
        px_per_unit: k,
        thinnest_px: thinnest * k,
        inflation,
        inflated,
        scaled_only,
        ink_low,
        ink_high,
    })
}

pub fn glyph_curve(target: &Target<f64>) -> Result<Vec<GlyphScaleRow>> {
    curve(target, GLYPH_EM, &EM_SIZES, FINE_PER_UNIT)
}

pub fn floor_row(rows: &[GlyphScaleRow]) -> Option<&GlyphScaleRow> {
    rows.iter()
        .filter(|r| r.px_per_unit / FINE_PER_UNIT <= MAX_FINE_RATIO)
        .max_by_key(|r| r.em_px)
}

pub fn verdict(row: &GlyphScaleRow, floor: &GlyphScaleRow) -> Verdict {
    let excess = row.inflated.max_abs - floor.inflated.max_abs;
    if excess <= MATCH_TOLERANCE {
        Verdict::Holds
    } else if excess <= BROKEN_FACTOR * MATCH_TOLERANCE {
        Verdict::Marginal
    } else {
        Verdict::Broken
    }
}

pub fn matches_down_to(rows: &[GlyphScaleRow]) -> Option<u32> {
    let floor = floor_row(rows)?;
    let mut sorted: Vec<&GlyphScaleRow> = rows.iter().filter(|r| r.em_px <= floor.em_px).collect();
    sorted.sort_by(|a, b| b.em_px.cmp(&a.em_px));
    let mut answer = None;
    for row in sorted {
        if verdict(row, floor) == Verdict::Broken {
            break;
        }
        answer = Some(row.em_px);
    }
    answer
}

pub fn horizontal_hairline(phase: f64, half_width: f64) -> Target<f64> {
    Target::ink_on_paper(
        "horizontal-hairline",
        (16, 16),
        vec![stroke(
            0,
            Vec2::new(2.0, 8.0 + phase),
            Vec2::new(14.0, 8.0 + phase),
            half_width,
        )],
        Vec::new(),
        Vec::new(),
    )
}
