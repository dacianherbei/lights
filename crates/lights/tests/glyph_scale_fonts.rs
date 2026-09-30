#![cfg(feature = "fit")]
use lights::Rgb;
use lights::fit::glyph_scale::{
    Crossing, EIGHT_BIT_STEP, EM_SIZES, FINE_PER_UNIT, GlyphScaleRow, MIN_GAP_OPPOSITION, MIN_OPPOSITION, PHASES,
    Residual, ScaleProgress, Side, THICKNESS_PERCENTILE, Verdict, crossings, curve, curve_reporting, didone_n,
    fine_atoms, floor_row, inflation_for, matches_down_to, predicted_at, reference_at, residual, stroke_widths,
    thinnest_feature, thinnest_gap, thinnest_half_width, thinnest_of, verdict, viewport_at,
};
use lights::fit::target::{GLYPH_EM, Part, Primitive, SUBPIXEL_X, SUBPIXEL_Y, Target};
use lights::image::FloatImage;
use lights::numerics::vector::Vec2;
use lights::outline::{ContourBuilder, EmPoint, Filled};
use lights::primitive::{FeatureId, PrimitiveSet, Region};
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use ttf_parser::{Face, OutlineBuilder};

const INTER: &[u8] = include_bytes!("fonts/Inter-Regular-subset.ttf");
const ABRIL: &[u8] = include_bytes!("fonts/AbrilFatface-Regular.ttf");
const NOTO_JP: &[u8] = include_bytes!("fonts/NotoSansJP-Regular-subset.ttf");
const STEM_FONT: &[u8] = include_bytes!("fonts/LightsStemCalibration.ttf");
const TOLERANCE_EM: f64 = 1.0 / 4000.0;
const MARGIN_UNITS: f64 = 1.0;
const INTER_N_NECK_UNITS: f64 = 0.4619;
const INTER_N_COUNTER_UNITS: f64 = 534.0 / 2048.0 * 12.0;
const UTSU_GAP_UNITS: f64 = 16.0 / 1000.0 * 12.0;
const NOTCH_ARM: f64 = 8.0;
const LARGEST_ANCHORABLE_EM: u32 = 128;
const RULE_PX: f64 = 0.7;
const WORST_SAMPLES: usize = 12;
const NEAR_PX: f64 = 0.5;
const ENRICHMENT: f64 = 3.0;
const PUBLISHED_MATCHES_DOWN_TO: [(&str, u32); 9] = [
    ("inter-n", 8),
    ("inter-o", 10),
    ("inter-g", 8),
    ("abril-n", 48),
    ("abril-o", 48),
    ("abril-e", 32),
    ("noto-jp-ei", 8),
    ("noto-jp-sho", 12),
    ("noto-jp-utsu", 32),
];
const PUBLISHED_INK_ONLY_FAILURES: [(&str, u32); 1] = [("noto-jp-utsu", 24)];
const CORPUS: [(&[u8], char, &str); 9] = [
    (INTER, 'n', "inter-n"),
    (INTER, 'o', "inter-o"),
    (INTER, 'g', "inter-g"),
    (ABRIL, 'n', "abril-n"),
    (ABRIL, 'o', "abril-o"),
    (ABRIL, 'e', "abril-e"),
    (NOTO_JP, '\u{6c38}', "noto-jp-ei"),
    (NOTO_JP, '\u{66f8}', "noto-jp-sho"),
    (NOTO_JP, '\u{9b31}', "noto-jp-utsu"),
];
const CALIBRATION: [(&[u8], char, &str); 4] = [
    (STEM_FONT, 'I', "calibration-stem"),
    (STEM_FONT, 'l', "calibration-hairline"),
    (STEM_FONT, 'o', "calibration-ring"),
    (STEM_FONT, '+', "calibration-cross"),
];
const STEMS: [&str; 2] = ["calibration-stem", "calibration-hairline"];
const RING: &str = "calibration-ring";
const CROSS: &str = "calibration-cross";
const FONT_FILES: [(&str, &str); 4] = [
    ("inter-", "Inter-Regular-subset.ttf"),
    ("abril-", "AbrilFatface-Regular.ttf"),
    ("noto-jp-", "NotoSansJP-Regular-subset.ttf"),
    ("calibration-", "LightsStemCalibration.ttf"),
];
const ORACLE_FIXTURES: &str = "tests/fixtures/glyph_oracle";
const ORACLE_MODES: [&str; 6] =
    ["coretext", "coretext-path", "coretext-smoothed-dark", "coretext-smoothed-light", "freetype", "freetype-light"];
const CALIBRATED_MODES: [&str; 3] = ["coretext", "coretext-path", "freetype"];
const GLYPH_RENDERER: &str = "coretext";
const PATH_FILLER: &str = "coretext-path";
const FREETYPE: &str = "freetype";
const SMOOTHED_CONTROL: &str = "coretext-smoothed-dark";
const CALIBRATION_STEPS: f64 = 4.0;
const APPROXIMATION_STEPS: f64 = 20.0;
const CONTROL_STEPS: f64 = 20.0;
const CALIBRATION_LARGEST_EM: u32 = 32;
const MANIFEST_HEADER: &str = "name\tfont\tglyph_id\tem_px\tphase_index\tphase\torigin_x\torigin_y\twidth\theight";

struct Pen {
    upem: f64,
    builder: ContourBuilder,
}

impl Pen {
    fn em(&self, x: f32, y: f32) -> EmPoint {
        Vec2::new(x as f64 / self.upem, y as f64 / self.upem)
    }
}

impl OutlineBuilder for Pen {
    fn move_to(&mut self, x: f32, y: f32) {
        let p = self.em(x, y);
        self.builder.move_to(p);
    }

    fn line_to(&mut self, x: f32, y: f32) {
        let p = self.em(x, y);
        self.builder.line_to(p);
    }

    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        let (c, p) = (self.em(x1, y1), self.em(x, y));
        self.builder.quad_to(c, p);
    }

    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        let (c1, c2, p) = (self.em(x1, y1), self.em(x2, y2), self.em(x, y));
        self.builder.cubic_to(c1, c2, p);
    }

    fn close(&mut self) {
        self.builder.close();
    }
}

struct Placed {
    target: Target<f64>,
    glyph_id: u16,
    offset: Vec2<f64>,
}

fn placed(font: &[u8], ch: char, name: &'static str) -> Placed {
    let face = Face::parse(font, 0).unwrap();
    let id = face.glyph_index(ch).unwrap_or_else(|| panic!("{name}: the face has no {ch:?}"));
    let upem = face.units_per_em() as f64;
    let mut pen = Pen { upem, builder: ContourBuilder::new(TOLERANCE_EM).unwrap() };
    let rect = face.outline_glyph(id, &mut pen).unwrap_or_else(|| panic!("{name}: no outline"));
    let min = Vec2::new(rect.x_min as f64 / upem, rect.y_min as f64 / upem);
    let size = Vec2::new(rect.x_max as f64 / upem, rect.y_max as f64 / upem) - min;
    let offset = Vec2::new(
        MARGIN_UNITS + SUBPIXEL_X - min.x * GLYPH_EM,
        MARGIN_UNITS + SUBPIXEL_Y - min.y * GLYPH_EM,
    );
    let target = Target {
        name,
        viewport: (
            (size.x * GLYPH_EM + 2.0 * MARGIN_UNITS + 1.0).ceil() as u32,
            (size.y * GLYPH_EM + 2.0 * MARGIN_UNITS + 1.0).ceil() as u32,
        ),
        set: PrimitiveSet {
            ink: Rgb::new(1.0, 1.0, 1.0),
            parts: Vec::new(),
            regions: vec![Region {
                id: FeatureId(0),
                shape: Filled::new(pen.builder.finish().unwrap()).mapped(GLYPH_EM, offset),
            }],
            holes: Vec::new(),
        },
        paper: Rgb::new(0.0, 0.0, 0.0),
        probes: Vec::new(),
    };
    Placed { target, glyph_id: id.0, offset }
}

fn glyph(font: &[u8], ch: char, name: &'static str) -> Target<f64> {
    placed(font, ch, name).target
}

fn oracle_glyph(name: &str) -> Placed {
    let &(font, ch, name) = CALIBRATION
        .iter()
        .chain(CORPUS.iter())
        .find(|g| g.2 == name)
        .unwrap_or_else(|| panic!("{name} is neither a calibration glyph nor in the corpus"));
    placed(font, ch, name)
}

fn font_file(name: &str) -> &'static str {
    FONT_FILES
        .iter()
        .find(|(prefix, _)| name.starts_with(prefix))
        .map(|f| f.1)
        .unwrap_or_else(|| panic!("no vendored font file for {name}"))
}

#[derive(Clone, Debug, PartialEq)]
struct OracleRow {
    name: String,
    font: String,
    glyph_id: u16,
    em_px: u32,
    phase_index: usize,
    origin: Vec2<f64>,
    viewport: (u32, u32),
}

impl OracleRow {
    fn of(name: &str, glyph: &Placed, em_px: u32, phase_index: usize) -> OracleRow {
        let k = em_px as f64 / GLYPH_EM;
        let phase = PHASES[phase_index];
        OracleRow {
            name: name.to_string(),
            font: font_file(name).to_string(),
            glyph_id: glyph.glyph_id,
            em_px,
            phase_index,
            origin: glyph.offset * k + Vec2::new(phase, phase),
            viewport: viewport_at(&glyph.target, k),
        }
    }

    fn phase(&self) -> f64 {
        PHASES[self.phase_index]
    }

    fn line(&self) -> String {
        format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            self.name,
            self.font,
            self.glyph_id,
            self.em_px,
            self.phase_index,
            self.phase(),
            self.origin.x,
            self.origin.y,
            self.viewport.0,
            self.viewport.1
        )
    }

    fn parse(line: &str) -> OracleRow {
        let cells: Vec<&str> = line.split('\t').collect();
        assert_eq!(cells.len(), 10, "a manifest row has ten cells: {line:?}");
        let number = |i: usize| -> f64 { cells[i].parse().unwrap_or_else(|_| panic!("cell {i} of {line:?}")) };
        let row = OracleRow {
            name: cells[0].to_string(),
            font: cells[1].to_string(),
            glyph_id: number(2) as u16,
            em_px: number(3) as u32,
            phase_index: number(4) as usize,
            origin: Vec2::new(number(6), number(7)),
            viewport: (number(8) as u32, number(9) as u32),
        };
        assert_eq!(row.phase(), number(5), "{line:?}: the phase does not match its index");
        row
    }
}

fn oracle_rows(names: &[&str], sizes: &[u32]) -> Vec<OracleRow> {
    let mut rows = Vec::new();
    for &name in names {
        let glyph = oracle_glyph(name);
        for &em_px in sizes {
            for phase_index in 0..PHASES.len() {
                rows.push(OracleRow::of(name, &glyph, em_px, phase_index));
            }
        }
    }
    rows
}

fn fixture_rows() -> Vec<OracleRow> {
    let names: Vec<&str> = CALIBRATION.iter().map(|g| g.2).collect();
    let sizes: Vec<u32> = EM_SIZES.iter().copied().filter(|&s| s <= CALIBRATION_LARGEST_EM).collect();
    oracle_rows(&names, &sizes)
}

fn read_manifest(dir: &Path) -> Vec<OracleRow> {
    let path = dir.join("manifest.tsv");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let mut lines = text.lines();
    assert_eq!(lines.next(), Some(MANIFEST_HEADER), "{}: unexpected header", path.display());
    lines.filter(|l| !l.trim().is_empty()).map(OracleRow::parse).collect()
}

fn read_images(dir: &Path, mode: &str, rows: &[OracleRow]) -> Vec<FloatImage<f64>> {
    let path = dir.join(format!("{mode}.pgm"));
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let mut at = 0usize;
    let images: Vec<FloatImage<f64>> = rows
        .iter()
        .map(|row| {
            let (w, h) = row.viewport;
            let header = format!("P5\n{w} {h}\n255\n");
            assert!(
                bytes[at..].starts_with(header.as_bytes()),
                "{}: the image for {} at {} px/em phase {} is not a {w}x{h} P5 image",
                path.display(),
                row.name,
                row.em_px,
                row.phase()
            );
            at += header.len();
            assert!(at + (w * h) as usize <= bytes.len(), "{}: {} at {} px/em is cut short", path.display(), row.name, row.em_px);
            let data = &bytes[at..at + (w * h) as usize];
            at += (w * h) as usize;
            let pixels = (0..h)
                .flat_map(|j| {
                    let m = (h - 1 - j) as usize;
                    (0..w as usize).map(move |i| Rgb::splat(data[m * w as usize + i] as f64 / 255.0))
                })
                .collect();
            FloatImage { width: w, height: h, pixels }
        })
        .collect();
    assert_eq!(at, bytes.len(), "{}: bytes left over after {} images", path.display(), rows.len());
    images
}

#[test]
fn the_thickness_instrument_reads_the_stroke_widths_the_synthetic_outlines_were_drawn_with() {
    for (target, drawn) in [(Target::t6_glyph_like(), 1.0), (didone_n(), 0.2)] {
        let read = thinnest_feature(&target, FINE_PER_UNIT);
        println!(
            "{}: thinnest feature {read:.4} units, drawn {drawn}, 2 x thinnest_half_width {:.4}",
            target.name,
            2.0 * thinnest_half_width(&target)
        );
        assert!((read - drawn).abs() <= 1.0 / FINE_PER_UNIT, "{}: read {read} against {drawn}", target.name);
    }
}

fn rectangle(width: f64) -> Target<f64> {
    stems("square-cornered-stem", width, 0.0, 1)
}

fn stems(name: &'static str, width: f64, gap: f64, count: usize) -> Target<f64> {
    let mut b = ContourBuilder::new(TOLERANCE_EM).unwrap();
    let y0 = 2.0 + SUBPIXEL_Y;
    for k in 0..count {
        let x0 = 2.0 + SUBPIXEL_X + k as f64 * (width + gap);
        b.move_to(Vec2::new(x0, y0));
        b.line_to(Vec2::new(x0 + width, y0));
        b.line_to(Vec2::new(x0 + width, y0 + 10.0));
        b.line_to(Vec2::new(x0, y0 + 10.0));
        b.close();
    }
    Target {
        name,
        viewport: (16, 16),
        set: PrimitiveSet {
            ink: Rgb::new(1.0, 1.0, 1.0),
            parts: Vec::new(),
            regions: vec![Region { id: FeatureId(0), shape: Filled::new(b.finish().unwrap()) }],
            holes: Vec::new(),
        },
        paper: Rgb::new(0.0, 0.0, 0.0),
        probes: Vec::new(),
    }
}

#[test]
fn a_square_cornered_stem_reads_its_width_and_not_its_corners() {
    let read = thinnest_feature(&rectangle(1.0), FINE_PER_UNIT);
    println!("square-cornered stem 1 unit wide: thinnest feature {read:.4} units");
    assert!((read - 1.0).abs() <= 1.0 / FINE_PER_UNIT, "read {read} against 1");
}

#[test]
fn the_gap_instrument_reads_the_slot_between_two_stems_and_an_open_exterior_as_no_gap() {
    let (width, gap) = (1.0, 0.3);
    let pair = stems("two-stems", width, gap, 2);
    let (slot, ink) = (thinnest_gap(&pair, FINE_PER_UNIT), thinnest_feature(&pair, FINE_PER_UNIT));
    let lone = thinnest_gap(&rectangle(width), FINE_PER_UNIT);
    let walls = (2.0 + SUBPIXEL_X + width, 2.0 + SUBPIXEL_X + width + gap);
    let found = crossings(&pair, FINE_PER_UNIT, Side::Paper, Side::Paper.min_opposition());
    let stray = found.iter().filter(|c| c.at.x < walls.0 || c.at.x > walls.1).count();
    println!(
        "two stems {width} wide, {gap} apart: thinnest gap {slot:.4}, thinnest ink {ink:.4}; one stem alone: thinnest gap {lone}; {stray} of {} paper crossings outside the slot",
        found.len()
    );
    assert!(!found.is_empty() && stray == 0, "{stray} of {} paper crossings sit outside the slot they measure", found.len());
    assert!((slot - gap).abs() <= 1.0 / FINE_PER_UNIT, "slot read {slot} against {gap}");
    assert!((ink - width).abs() <= 1.0 / FINE_PER_UNIT, "ink read {ink} against {width}");
    assert!(lone.is_infinite(), "the paper around a convex stem has no opposing walls, read {lone}");
}

fn notch(half_angle_degrees: f64) -> Target<f64> {
    let apex = Vec2::new(8.0 + SUBPIXEL_X, 2.0 + SUBPIXEL_Y);
    let arm = |sign: f64| {
        let t = sign * half_angle_degrees.to_radians();
        Primitive::Segment { a: apex, b: apex + Vec2::new(NOTCH_ARM * t.sin(), NOTCH_ARM * t.cos()), half_width: 0.5 }
    };
    Target::ink_on_paper(
        "acute-notch",
        (16, 16),
        vec![Part { id: FeatureId(0), shape: arm(-1.0) }, Part { id: FeatureId(1), shape: arm(1.0) }],
        Vec::new(),
        Vec::new(),
    )
}

#[test]
fn an_acute_notch_is_not_a_gap_and_inter_n_reads_its_counter_rather_than_its_notch() {
    let v = thinnest_gap(&notch(15.0), FINE_PER_UNIT);
    let mouth = 2.0 * NOTCH_ARM * 15f64.to_radians().sin() - 1.0;
    let n = thinnest_gap(&glyph(INTER, 'n', "inter-n"), FINE_PER_UNIT);
    println!("30 degree notch: thinnest gap {v:.4} (the mouth between the arm ends is {mouth:.4}); inter-n: thinnest gap {n:.4}");
    assert!(v > 0.5 * mouth, "the notch apex was read as a gap of {v}");
    assert!((n - INTER_N_COUNTER_UNITS).abs() < 0.05, "Inter n's counter is {INTER_N_COUNTER_UNITS} units, read {n}");
}

#[test]
fn utsu_is_thinnest_in_its_paper_and_not_in_its_ink() {
    let target = glyph(NOTO_JP, '\u{9b31}', "noto-jp-utsu");
    let (gap, ink) = (thinnest_gap(&target, FINE_PER_UNIT), thinnest_feature(&target, FINE_PER_UNIT));
    println!("noto-jp-utsu: thinnest parallel gap {gap:.4} units, thinnest stroke {ink:.4} units");
    assert!((gap - UTSU_GAP_UNITS).abs() < 1.0 / FINE_PER_UNIT, "read {gap} against {UTSU_GAP_UNITS}");
    assert!(gap < 0.6 * ink, "gap {gap} against stroke {ink}");
}

#[test]
fn an_eccentric_ring_reads_its_thinnest_cross_section_which_is_a_saddle_of_the_field() {
    let (outer, inner, offset) = (5.0, 3.0, 0.8);
    let centre = Vec2::new(8.0 + SUBPIXEL_X, 8.0 + SUBPIXEL_Y);
    let ring = Target::ink_on_paper(
        "eccentric-ring",
        (16, 16),
        vec![Part { id: FeatureId(0), shape: Primitive::Disc { centre, radius: outer } }],
        vec![Part { id: FeatureId(1), shape: Primitive::Disc { centre: centre + Vec2::new(offset, 0.0), radius: inner } }],
        Vec::new(),
    );
    let thinnest = outer - inner - offset;
    let read = thinnest_feature(&ring, FINE_PER_UNIT);
    println!("eccentric ring: thinnest feature {read:.4} units, thinnest cross-section {thinnest}, thickest {}", outer - inner + offset);
    assert!((read - thinnest).abs() <= 0.02 * thinnest + 1.0 / FINE_PER_UNIT, "read {read} against {thinnest}");
}

#[test]
fn a_real_glyph_reports_a_finite_thinnest_feature_in_its_curve() {
    let target = glyph(INTER, 'n', "inter-n");
    let rows = curve(&target, GLYPH_EM, &[16], FINE_PER_UNIT).unwrap();
    let thinnest = thinnest_feature(&target, FINE_PER_UNIT);
    println!("inter-n: thinnest {thinnest:.4} units, {:.4} px at 16 px em", rows[0].thinnest_px);
    assert!(rows[0].thinnest_px.is_finite(), "a region's thinnest feature read {}", rows[0].thinnest_px);
    assert!((thinnest - INTER_N_NECK_UNITS).abs() < 0.03, "Inter n's shoulder neck is {INTER_N_NECK_UNITS} units, read {thinnest}");
    assert!((rows[0].thinnest_px - thinnest * 16.0 / GLYPH_EM).abs() < 1e-12);
}

fn say(name: &str, started: std::time::Instant, what: &str) {
    use std::io::Write;
    println!("[{:>6.0}s] {name}: {what}", started.elapsed().as_secs_f64());
    std::io::stdout().flush().ok();
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    sorted[(q * (sorted.len() - 1) as f64).floor() as usize]
}

struct Measured<'a> {
    name: &'a str,
    viewport: (u32, u32),
    widths: Vec<f64>,
    gap: f64,
    rows: Vec<GlyphScaleRow>,
    seconds: f64,
}

fn smallest_em_at_least(rows: &[GlyphScaleRow], units: f64) -> Option<u32> {
    rows.iter().filter(|r| units * r.px_per_unit >= RULE_PX).map(|r| r.em_px).min()
}

#[test]
#[ignore]
fn real_glyphs_over_scale() {
    let sizes: Vec<u32> = match std::env::var("L168_SIZES") {
        Ok(v) if !v.trim().is_empty() => v.split(',').filter_map(|s| s.trim().parse().ok()).collect(),
        _ => EM_SIZES.iter().copied().filter(|&s| s <= LARGEST_ANCHORABLE_EM).collect(),
    };
    let full_ladder = sizes.iter().copied().eq(EM_SIZES.iter().copied().filter(|&s| s <= LARGEST_ANCHORABLE_EM));
    let only = std::env::var("L168_ONLY").unwrap_or_default();
    let wanted: Vec<&(&[u8], char, &str)> = CORPUS
        .iter()
        .filter(|(_, _, n)| only.trim().is_empty() || only.split(',').any(|w| w.trim() == *n))
        .collect();
    println!("L168: {} of {} glyphs, {} rungs {:?}", wanted.len(), CORPUS.len(), sizes.len(), sizes);
    let results: Vec<Measured> = std::thread::scope(|scope| {
        let handles: Vec<_> = wanted
            .iter()
            .map(|&&(font, ch, name)| {
                let sizes = &sizes;
                scope.spawn(move || {
                    let started = std::time::Instant::now();
                    let total_work: f64 = sizes.iter().map(|&s| (s as f64) * (s as f64)).sum();
                    let spent = std::sync::atomic::AtomicU64::new(0);
                    let load = |s: &std::sync::atomic::AtomicU64| {
                        s.load(std::sync::atomic::Ordering::Relaxed) as f64
                    };
                    say(name, started, "started");
                    let target = glyph(font, ch, name);
                    say(name, started, "outline built");
                    let widths = stroke_widths(&target, FINE_PER_UNIT);
                    say(name, started, &format!("{} stroke widths read", widths.len()));
                    let gap = thinnest_gap(&target, FINE_PER_UNIT);
                    say(name, started, &format!("thinnest parallel gap {gap:.4} units"));
                    let rows = curve_reporting(&target, GLYPH_EM, sizes, FINE_PER_UNIT, |p| match p {
                        ScaleProgress::Prepared { rungs, thinnest_units } => say(
                            name,
                            started,
                            &format!("prepared: {rungs} rungs, thinnest {thinnest_units:.4} units"),
                        ),
                        ScaleProgress::RungStarted { em_px, index, total } => say(
                            name,
                            started,
                            &format!(
                                "rung {em_px} px/em ({}/{total}) begins - {:.0}% of the work done, this rung is {:.0}%",
                                index + 1,
                                100.0 * load(&spent) / total_work,
                                100.0 * (em_px as f64) * (em_px as f64) / total_work
                            ),
                        ),
                        ScaleProgress::Viewport { em_px, width, height } => say(
                            name,
                            started,
                            &format!("rung {em_px} px/em viewport {width}x{height} = {} px", width as u64 * height as u64),
                        ),
                        ScaleProgress::PhaseStep { em_px, phase, what } => {
                            say(name, started, &format!("rung {em_px} px/em phase {phase}/4: {what}"))
                        }
                        ScaleProgress::PhaseDone { em_px, phase, phases } => say(
                            name,
                            started,
                            &format!("rung {em_px} px/em phase {phase}/{phases} complete"),
                        ),
                        ScaleProgress::RungDone { row, index, total } => {
                            let w = (row.em_px as u64) * (row.em_px as u64);
                            spent.fetch_add(w, std::sync::atomic::Ordering::Relaxed);
                            say(
                                name,
                                started,
                                &format!(
                                    "rung {} px/em ({}/{total}) DONE - thinnest {:.2} px, inflated max {:.4} - {:.1}% of work, elapsed {:.0}s, projected total {:.0}s, ~{:.0}s left",
                                    row.em_px,
                                    index + 1,
                                    row.thinnest_px,
                                    row.inflated.max_abs,
                                    100.0 * load(&spent) / total_work,
                                    started.elapsed().as_secs_f64(),
                                    started.elapsed().as_secs_f64() * total_work / load(&spent).max(1.0),
                                    started.elapsed().as_secs_f64() * (total_work / load(&spent).max(1.0) - 1.0)
                                ),
                            )
                        }
                    })
                    .unwrap();
                    let seconds = started.elapsed().as_secs_f64();
                    say(name, started, &format!("DONE in {seconds:.0} s"));
                    Measured { name, viewport: target.viewport, widths, gap, rows, seconds }
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    println!("glyph\tem_px\tthinnest_px\tthinnest_gap_px\tmedian_stroke_px\tinflated_max_abs\tinflated_mean_abs\tink_low\tink_high\tscaled_only_max_abs\tverdict");
    for m in &results {
        let floor = floor_row(&m.rows).unwrap();
        let median = percentile(&m.widths, 0.5);
        for r in &m.rows {
            println!(
                "{}\t{}\t{:.3}\t{:.3}\t{:.3}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{}",
                m.name,
                r.em_px,
                r.thinnest_px,
                m.gap * r.px_per_unit,
                median * r.px_per_unit,
                r.inflated.max_abs,
                r.inflated.mean_abs,
                r.ink_low,
                r.ink_high,
                r.scaled_only.max_abs,
                verdict(r, floor).name()
            );
        }
    }
    println!("glyph\tviewport_units\tthinnest_units\tthinnest_gap_units\tp10_units\tmedian_units\tmatches_down_to\trule_ink_only\trule_ink_and_paper\tsmallest_em_median_ge_0.7px\tfloor_em\tfloor_max_abs\tseconds");
    let mut ink_only_failures = Vec::new();
    for m in &results {
        let floor = floor_row(&m.rows).unwrap();
        let (thin, p10, median) = (percentile(&m.widths, THICKNESS_PERCENTILE), percentile(&m.widths, 0.1), percentile(&m.widths, 0.5));
        let either = thin.min(m.gap);
        println!(
            "{}\t{}x{}\t{thin:.4}\t{:.4}\t{p10:.4}\t{median:.4}\t{:?}\t{:?}\t{:?}\t{:?}\t{}\t{:.4}\t{:.0}",
            m.name,
            m.viewport.0,
            m.viewport.1,
            m.gap,
            matches_down_to(&m.rows),
            smallest_em_at_least(&m.rows, thin),
            smallest_em_at_least(&m.rows, either),
            smallest_em_at_least(&m.rows, median),
            floor.em_px,
            floor.inflated.max_abs,
            m.seconds
        );
        for r in &m.rows {
            let broken = verdict(r, floor) == Verdict::Broken;
            assert!(
                !(broken && either * r.px_per_unit >= RULE_PX),
                "{} at {} px/em is broken with its thinnest ink or paper at {:.3} px",
                m.name,
                r.em_px,
                either * r.px_per_unit
            );
            if broken && thin * r.px_per_unit >= RULE_PX {
                ink_only_failures.push((m.name, r.em_px));
            }
        }
        if full_ladder {
            let published = PUBLISHED_MATCHES_DOWN_TO.iter().find(|(n, _)| *n == m.name).map(|p| p.1);
            assert_eq!(matches_down_to(&m.rows), published, "{} matches down to", m.name);
        }
    }
    println!("broken rungs whose thinnest INK is at least {RULE_PX} px: {ink_only_failures:?}");
    if full_ladder && results.len() == CORPUS.len() {
        assert_eq!(ink_only_failures, PUBLISHED_INK_ONLY_FAILURES.to_vec());
    }
}

fn thin_sites(target: &Target<f64>, side: Side, px_per_unit: f64) -> Vec<Vec2<f64>> {
    crossings(target, FINE_PER_UNIT, side, side.min_opposition())
        .into_iter()
        .filter(|c| c.width * px_per_unit < RULE_PX)
        .map(|c| c.at)
        .collect()
}

fn near(sites: &[Vec2<f64>], at: Vec2<f64>, px_per_unit: f64) -> bool {
    sites.iter().any(|s| (*s - at).length() * px_per_unit <= NEAR_PX)
}

#[test]
#[ignore]
fn the_worst_pixels_of_a_broken_rung_sit_on_its_thinnest_feature() {
    let cases = [
        (NOTO_JP, '\u{9b31}', "noto-jp-utsu", 20, Side::Paper),
        (NOTO_JP, '\u{9b31}', "noto-jp-utsu", 24, Side::Paper),
        (ABRIL, 'n', "abril-n", 24, Side::Ink),
        (ABRIL, 'n', "abril-n", 32, Side::Ink),
    ];
    println!("glyph\tem_px\tside\tworst_abs\tworst_near_thin_paper\tworst_near_thin_ink\ttop_near_thin_paper\ttop_near_thin_ink\tsupport\tsupport_near_thin_paper\tsupport_near_thin_ink");
    for (font, ch, name, em_px, side) in cases {
        let target = glyph(font, ch, name);
        let atoms = fine_atoms(&target, FINE_PER_UNIT, 0.0).unwrap();
        let k = em_px as f64 / GLYPH_EM;
        let viewport = viewport_at(&target, k);
        let inflation = inflation_for(&atoms, k).unwrap();
        let paper = thin_sites(&target, Side::Paper, k);
        let ink = thin_sites(&target, Side::Ink, k);
        let mut samples: Vec<(f64, Vec2<f64>)> = Vec::new();
        for &phase in PHASES.iter() {
            let shift = Vec2::new(phase, phase);
            let reference = reference_at(&target, k, shift).unwrap();
            let predicted = predicted_at(&atoms, k, viewport, inflation, shift);
            for py in 0..viewport.1 {
                for px in 0..viewport.0 {
                    let (a, b) = (reference.get(px, py).x, predicted.get(px, py).x);
                    if a > 0.0 || b > 0.5 * EIGHT_BIT_STEP {
                        let at = Vec2::new((px as f64 + 0.5 - phase) / k, (py as f64 + 0.5 - phase) / k);
                        samples.push(((a - b).abs(), at));
                    }
                }
            }
        }
        samples.sort_by(|p, q| q.0.total_cmp(&p.0));
        let share = |set: &[(f64, Vec2<f64>)], sites: &[Vec2<f64>]| {
            set.iter().filter(|s| near(sites, s.1, k)).count() as f64 / set.len() as f64
        };
        let top = &samples[..WORST_SAMPLES];
        let (worst, at) = samples[0];
        println!(
            "{name}\t{em_px}\t{side:?}\t{worst:.4}\t{}\t{}\t{:.3}\t{:.3}\t{}\t{:.3}\t{:.3}",
            near(&paper, at, k),
            near(&ink, at, k),
            share(top, &paper),
            share(top, &ink),
            samples.len(),
            share(&samples, &paper),
            share(&samples, &ink)
        );
        let sites = match side {
            Side::Paper => &paper,
            Side::Ink => &ink,
        };
        assert!(near(sites, at, k), "{name} at {em_px}: the worst pixel is not on a thin {side:?} feature");
        assert!(
            share(top, sites) >= ENRICHMENT * share(&samples, sites),
            "{name} at {em_px}: the worst pixels sit on thin {side:?} no more often than the support does"
        );
    }
}

const GATES: [f64; 8] = [0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99];
const PLATEAU: (f64, f64) = (0.93, 0.99);
const NOTCHED: [&str; 3] = ["inter-n", "abril-n", "noto-jp-ei"];

#[test]
#[ignore]
fn the_paper_gate_is_read_off_a_plateau_and_the_ink_gate_reads_a_notch() {
    println!("glyph\tside\t{}", GATES.map(|g| format!("gate_{g}")).join("\t"));
    let ladder: Vec<u32> = EM_SIZES.iter().copied().filter(|&s| s <= LARGEST_ANCHORABLE_EM).collect();
    for &(font, ch, name) in CORPUS.iter() {
        let target = glyph(font, ch, name);
        let reading = |side: Side, gate: f64| -> (f64, Vec<Crossing>) {
            let found: Vec<Crossing> =
                crossings(&target, FINE_PER_UNIT, side, MIN_OPPOSITION).into_iter().filter(|c| c.opposition > gate).collect();
            let mut w: Vec<f64> = found.iter().map(|c| c.width).collect();
            w.sort_by(f64::total_cmp);
            (thinnest_of(&w), found)
        };
        let ink_row: Vec<f64> = GATES.iter().map(|&g| reading(Side::Ink, g).0).collect();
        let paper_row: Vec<f64> = GATES.iter().map(|&g| reading(Side::Paper, g).0).collect();
        let show = |row: &[f64]| row.iter().map(|v| format!("{v:.4}")).collect::<Vec<_>>().join("\t");
        println!("{name}\tInk\t{}", show(&ink_row));
        println!("{name}\tPaper\t{}", show(&paper_row));
        let ink = reading(Side::Ink, Side::Ink.min_opposition()).0;
        let predicted = |gap: f64| ladder.iter().copied().find(|&em| ink.min(gap) * em as f64 / GLYPH_EM >= RULE_PX);
        let chosen = predicted(reading(Side::Paper, Side::Paper.min_opposition()).0);
        let plateau: Vec<f64> =
            GATES.iter().zip(paper_row.iter()).filter(|(g, _)| (PLATEAU.0..=PLATEAU.1).contains(*g)).map(|(_, &p)| p).collect();
        assert!(plateau.iter().all(|&p| predicted(p) == chosen), "{name}: the rule's size moves with the paper gate");
        let (low, high) = plateau.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &p| (a.min(p), b.max(p)));
        let binds = low < ink;
        println!("{name}\tpaper {low:.4}-{high:.4} over gates {PLATEAU:?}, ink {ink:.4}, the paper sets the rule: {binds}, predicted size {chosen:?}");
        if binds {
            assert!(high - low <= 0.05 * low, "{name}: where the paper sets the rule its reading moves {low}-{high}");
        }
        let (loose, all) = reading(Side::Paper, MIN_OPPOSITION);
        let tail: Vec<Crossing> = all.into_iter().filter(|c| c.width <= loose).collect();
        let (lo, hi) = tail.iter().fold(((f64::INFINITY, f64::INFINITY), (f64::NEG_INFINITY, f64::NEG_INFINITY)), |(lo, hi), c| {
            ((lo.0.min(c.at.x), lo.1.min(c.at.y)), (hi.0.max(c.at.x), hi.1.max(c.at.y)))
        });
        let angles = tail.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), c| {
            let d = c.opposition.min(1.0).acos().to_degrees();
            (a.min(d), b.max(d))
        });
        println!(
            "{name}\tunder the ink gate the thinnest {} paper crossings read up to {loose:.4} units, span {:.2} x {:.2} units, walls {:.0}-{:.0} degrees apart",
            tail.len(),
            hi.0 - lo.0,
            hi.1 - lo.1,
            angles.0,
            angles.1
        );
        if NOTCHED.contains(&name) {
            assert!(hi.0 - lo.0 <= 1.0 && hi.1 - lo.1 <= 1.0, "{name}: the thin paper tail is not one notch");
            assert!(tail.iter().all(|c| c.opposition < MIN_GAP_OPPOSITION), "{name}: walls {angles:?} degrees apart");
            assert!(thinnest_gap(&target, FINE_PER_UNIT) > 2.0 * loose, "{name}: the gate did not remove the notch");
        }
    }
}

fn ladder() -> Vec<u32> {
    EM_SIZES.iter().copied().filter(|&s| s <= LARGEST_ANCHORABLE_EM).collect()
}

#[test]
#[ignore]
fn the_system_rasteriser_manifest() {
    let dir = PathBuf::from(
        std::env::var("L173_DIR").unwrap_or_else(|_| panic!("L173_DIR names the directory manifest.tsv is written to")),
    );
    let only = std::env::var("L173_ONLY").unwrap_or_default();
    let rows = if only.trim() == "fixture" {
        fixture_rows()
    } else {
        let names: Vec<&str> = CALIBRATION
            .iter()
            .chain(CORPUS.iter())
            .map(|g| g.2)
            .filter(|n| only.trim().is_empty() || only.split(',').any(|w| w.trim() == *n))
            .collect();
        let sizes: Vec<u32> = match std::env::var("L173_SIZES") {
            Ok(v) if !v.trim().is_empty() => v.split(',').filter_map(|s| s.trim().parse().ok()).collect(),
            _ => ladder(),
        };
        oracle_rows(&names, &sizes)
    };
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("cannot create {}: {e}", dir.display()));
    let text: String =
        std::iter::once(MANIFEST_HEADER.to_string()).chain(rows.iter().map(OracleRow::line)).map(|l| l + "\n").collect();
    let path = dir.join("manifest.tsv");
    std::fs::write(&path, text).unwrap_or_else(|e| panic!("cannot write {}: {e}", path.display()));
    println!("{} rows -> {}", rows.len(), path.display());
}

#[test]
fn the_system_rasterisers_draw_a_square_cornered_stem_where_box_coverage_puts_it() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(ORACLE_FIXTURES);
    let rows = read_manifest(&dir);
    assert_eq!(
        rows,
        fixture_rows(),
        "{ORACLE_FIXTURES} was rendered at placements the harness no longer uses; regenerate it with the oracle harness"
    );
    let glyphs: Vec<(&str, Target<f64>)> = CALIBRATION.iter().map(|&(font, ch, name)| (name, glyph(font, ch, name))).collect();
    let references: Vec<FloatImage<f64>> = rows
        .iter()
        .map(|row| {
            let target = &glyphs.iter().find(|g| g.0 == row.name).unwrap().1;
            reference_at(target, row.em_px as f64 / GLYPH_EM, Vec2::new(row.phase(), row.phase())).unwrap()
        })
        .collect();
    for mode in CALIBRATED_MODES.iter().copied().chain(std::iter::once(SMOOTHED_CONTROL)) {
        let images = read_images(&dir, mode, &rows);
        for &(_, _, name) in CALIBRATION.iter() {
            let mut worst = (0.0f64, 0u32, 0.0f64);
            let (mut low, mut high) = (f64::INFINITY, f64::NEG_INFINITY);
            for ((row, reference), image) in rows.iter().zip(&references).zip(&images).filter(|((r, _), _)| r.name == name) {
                let r = residual(reference, image).unwrap();
                if r.max_abs > worst.0 {
                    worst = (r.max_abs, row.em_px, row.phase());
                }
                low = low.min(r.ink_ratio);
                high = high.max(r.ink_ratio);
            }
            let steps = worst.0 / EIGHT_BIT_STEP;
            println!(
                "{mode}\t{name}\tworst |system - box| {steps:.2} 8-bit steps at {} px/em phase {}\tsystem/box ink {low:.4}-{high:.4}",
                worst.1, worst.2
            );
            match expectation(mode, name) {
                Expect::Within(bound) => assert!(
                    steps <= bound,
                    "{mode} {name}: the system draws the calibration glyph {steps:.2} 8-bit steps from box coverage, over {bound}"
                ),
                Expect::Beyond(bound) => assert!(
                    steps > bound,
                    "{mode} {name}: the fixture no longer records the glyph renderer's departure ({steps:.2} steps); if it was regenerated on a system that changed it, the published finding is stale"
                ),
                Expect::Darker(bound) => assert!(
                    steps >= bound && low > 1.0,
                    "{mode} {name}: font smoothing darkens every glyph, and this comparison read {steps:.2} steps with ink {low:.4}-{high:.4}"
                ),
            }
        }
    }
}

enum Expect {
    Within(f64),
    Beyond(f64),
    Darker(f64),
}

fn expectation(mode: &str, name: &str) -> Expect {
    match (mode, name) {
        (SMOOTHED_CONTROL, _) => Expect::Darker(CONTROL_STEPS),
        (GLYPH_RENDERER, RING) | (GLYPH_RENDERER, CROSS) => Expect::Beyond(APPROXIMATION_STEPS),
        (FREETYPE, RING) | (FREETYPE, CROSS) | (PATH_FILLER, RING) => Expect::Within(APPROXIMATION_STEPS),
        _ => Expect::Within(CALIBRATION_STEPS),
    }
}

fn worst_of(a: Option<Residual>, b: Residual) -> Residual {
    match a {
        Some(r) if r.max_abs >= b.max_abs => r,
        Some(_) | None => b,
    }
}

struct PhaseBeside {
    em_px: u32,
    inflated: Residual,
    scaled_only: Residual,
    box_system: Vec<Residual>,
    system_inflated: Vec<Residual>,
}

struct Beside {
    row: GlyphScaleRow,
    box_system: Vec<Residual>,
    system_inflated: Vec<Residual>,
}

fn steps(r: &Residual) -> f64 {
    r.max_abs / EIGHT_BIT_STEP
}

fn spread(values: &[f64]) -> (f64, f64, f64) {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    let median = if n % 2 == 1 { sorted[n / 2] } else { 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]) };
    (sorted[0], median, sorted[n - 1])
}

#[test]
#[ignore]
fn real_glyphs_beside_the_system_rasterisers() {
    let dir = PathBuf::from(
        std::env::var("L173_DIR").unwrap_or_else(|_| panic!("L173_DIR names the directory the system renders are in")),
    );
    let rows = read_manifest(&dir);
    let modes: Vec<&str> = ORACLE_MODES.iter().copied().filter(|m| dir.join(format!("{m}.pgm")).exists()).collect();
    assert!(!modes.is_empty(), "no system renders in {}", dir.display());
    let images: Vec<Vec<FloatImage<f64>>> = modes.iter().map(|m| read_images(&dir, m, &rows)).collect();
    let names: Vec<&'static str> =
        CALIBRATION.iter().chain(CORPUS.iter()).map(|g| g.2).filter(|n| rows.iter().any(|r| r.name == *n)).collect();
    println!("L173: {} glyphs, {} manifest rows, modes {modes:?}", names.len(), rows.len());
    let started = std::time::Instant::now();
    let results: Vec<(&str, f64, Vec<Beside>)> = names
        .par_iter()
        .map(|&name| {
            let glyph = oracle_glyph(name);
            let target = &glyph.target;
            say(name, started, "started");
            let atoms = fine_atoms(target, FINE_PER_UNIT, 0.0).unwrap();
            let thinnest = thinnest_feature(target, FINE_PER_UNIT);
            let gap = thinnest_gap(target, FINE_PER_UNIT);
            say(name, started, &format!("prepared: thinnest {thinnest:.4} units, thinnest gap {gap:.4} units"));
            let mine: Vec<usize> = (0..rows.len()).filter(|&i| rows[i].name == name).collect();
            let phases: Vec<PhaseBeside> = mine
                .par_iter()
                .map(|&index| {
                    let row = &rows[index];
                    let k = row.em_px as f64 / GLYPH_EM;
                    let viewport = viewport_at(target, k);
                    assert_eq!(row.viewport, viewport, "{name} at {} px/em", row.em_px);
                    let inflation = inflation_for(&atoms, k).unwrap();
                    let shift = Vec2::new(row.phase(), row.phase());
                    let reference = reference_at(target, k, shift).unwrap();
                    let predicted = predicted_at(&atoms, k, viewport, inflation, shift);
                    let beside = PhaseBeside {
                        em_px: row.em_px,
                        inflated: residual(&reference, &predicted).unwrap(),
                        scaled_only: residual(&reference, &predicted_at(&atoms, k, viewport, 0.0, shift)).unwrap(),
                        box_system: images.iter().map(|set| residual(&reference, &set[index]).unwrap()).collect(),
                        system_inflated: images.iter().map(|set| residual(&set[index], &predicted).unwrap()).collect(),
                    };
                    say(name, started, &format!("rung {} px/em phase {} done", row.em_px, row.phase()));
                    beside
                })
                .collect();
            let mut sizes: Vec<u32> = phases.iter().map(|p| p.em_px).collect();
            sizes.sort_unstable();
            sizes.dedup();
            let besides: Vec<Beside> = sizes
                .iter()
                .map(|&em_px| {
                    let k = em_px as f64 / GLYPH_EM;
                    let (mut inflated, mut scaled_only) = (None, None);
                    let (mut ink_low, mut ink_high) = (f64::INFINITY, f64::NEG_INFINITY);
                    let mut box_system: Vec<Option<Residual>> = vec![None; modes.len()];
                    let mut system_inflated: Vec<Option<Residual>> = vec![None; modes.len()];
                    for p in phases.iter().filter(|p| p.em_px == em_px) {
                        ink_low = ink_low.min(p.inflated.ink_ratio);
                        ink_high = ink_high.max(p.inflated.ink_ratio);
                        inflated = Some(worst_of(inflated, p.inflated));
                        scaled_only = Some(worst_of(scaled_only, p.scaled_only));
                        for m in 0..modes.len() {
                            box_system[m] = Some(worst_of(box_system[m], p.box_system[m]));
                            system_inflated[m] = Some(worst_of(system_inflated[m], p.system_inflated[m]));
                        }
                    }
                    Beside {
                        row: GlyphScaleRow {
                            em_px,
                            px_per_unit: k,
                            thinnest_px: thinnest * k,
                            inflation: inflation_for(&atoms, k).unwrap(),
                            inflated: inflated.unwrap(),
                            scaled_only: scaled_only.unwrap(),
                            ink_low,
                            ink_high,
                        },
                        box_system: box_system.into_iter().map(Option::unwrap).collect(),
                        system_inflated: system_inflated.into_iter().map(Option::unwrap).collect(),
                    }
                })
                .collect();
            say(name, started, &format!("DONE, {} rungs", besides.len()));
            (name, gap, besides)
        })
        .collect();
    println!("glyph\tem_px\tthinnest_px\tthinnest_gap_px\tverdict\tbox_inflated_max\tmode\tbox_system_max\tbox_system_mean\tsystem_over_box_ink\tsystem_inflated_max");
    for (name, gap, besides) in &results {
        let rows: Vec<GlyphScaleRow> = besides.iter().map(|b| b.row).collect();
        let floor = floor_row(&rows).unwrap();
        for b in besides {
            for (m, mode) in modes.iter().enumerate() {
                println!(
                    "{name}\t{}\t{:.3}\t{:.3}\t{}\t{:.4}\t{mode}\t{:.4}\t{:.4}\t{:.4}\t{:.4}",
                    b.row.em_px,
                    b.row.thinnest_px,
                    gap * b.row.px_per_unit,
                    verdict(&b.row, floor).name(),
                    b.row.inflated.max_abs,
                    b.box_system[m].max_abs,
                    b.box_system[m].mean_abs,
                    b.box_system[m].ink_ratio,
                    b.system_inflated[m].max_abs
                );
            }
        }
    }
    let sizes: Vec<u32> = results.first().map(|r| r.2.iter().map(|b| b.row.em_px).collect()).unwrap_or_default();
    let header: Vec<String> = sizes.iter().map(|s| s.to_string()).collect();
    for (title, against_inflation) in [("|x - box|", false), ("|x - inflated|", true)] {
        println!("{title}, worst pixel over four phases, in 8-bit steps");
        println!("glyph\tx\t{}", header.join("\t"));
        for (name, _, besides) in &results {
            if !against_inflation {
                let line: Vec<String> = besides.iter().map(|b| format!("{:.1}", steps(&b.row.inflated))).collect();
                println!("{name}\tinflated\t{}", line.join("\t"));
            }
            for (m, mode) in modes.iter().enumerate() {
                let line: Vec<String> = besides
                    .iter()
                    .map(|b| format!("{:.1}", steps(if against_inflation { &b.system_inflated[m] } else { &b.box_system[m] })))
                    .collect();
                println!("{name}\t{mode}\t{}", line.join("\t"));
            }
        }
    }
    let corpus: Vec<(&str, GlyphScaleRow, &Beside)> = results
        .iter()
        .filter(|r| !r.0.starts_with("calibration-"))
        .flat_map(|(name, _, besides)| {
            let rows: Vec<GlyphScaleRow> = besides.iter().map(|b| b.row).collect();
            let floor = *floor_row(&rows).unwrap();
            besides.iter().map(move |b| (*name, floor, b))
        })
        .collect();
    println!("the corpus in bands: worst pixel over four phases in 8-bit steps, and how often a system departs from box further than inflation does");
    println!("band\tx\trows\tlow\tmedian\thigh\tfurther_than_inflation\tratio_low\tratio_median\tratio_high\tink_low\tink_high");
    for band in ["8-32 px/em", "every rung", "rungs L-168 calls broken"] {
        let chosen: Vec<&(&str, GlyphScaleRow, &Beside)> = corpus
            .iter()
            .filter(|(_, floor, b)| match band {
                "8-32 px/em" => b.row.em_px <= 32,
                "every rung" => true,
                _ => verdict(&b.row, floor) == Verdict::Broken,
            })
            .collect();
        if chosen.is_empty() {
            continue;
        }
        let inflation: Vec<f64> = chosen.iter().map(|c| steps(&c.2.row.inflated)).collect();
        let (low, median, high) = spread(&inflation);
        println!("{band}\tinflated\t{}\t{low:.1}\t{median:.1}\t{high:.1}", chosen.len());
        for (m, mode) in modes.iter().enumerate() {
            let away: Vec<f64> = chosen.iter().map(|c| steps(&c.2.box_system[m])).collect();
            let ratio: Vec<f64> = chosen.iter().map(|c| steps(&c.2.box_system[m]) / steps(&c.2.row.inflated)).collect();
            let further = chosen.iter().filter(|c| c.2.box_system[m].max_abs > c.2.row.inflated.max_abs).count();
            let ink = chosen.iter().map(|c| c.2.box_system[m].ink_ratio).fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), v| (a.min(v), b.max(v)));
            let (low, median, high) = spread(&away);
            let (r_low, r_median, r_high) = spread(&ratio);
            println!(
                "{band}\t{mode}\t{}\t{low:.1}\t{median:.1}\t{high:.1}\t{further}\t{r_low:.2}\t{r_median:.2}\t{r_high:.2}\t{:.3}\t{:.3}",
                chosen.len(),
                ink.0,
                ink.1
            );
        }
    }
    if let (Some(plain), Some(light)) =
        (modes.iter().position(|m| *m == "freetype"), modes.iter().position(|m| *m == "freetype-light"))
    {
        let differ = (0..rows.len()).filter(|&i| images[plain][i].pixels != images[light][i].pixels).count();
        println!("freetype-light differs from freetype in {differ} of {} images", rows.len());
    }
    for (name, _, besides) in results.iter().filter(|r| STEMS.contains(&r.0)) {
        for mode in CALIBRATED_MODES {
            let Some(m) = modes.iter().position(|x| *x == mode) else { continue };
            let worst = besides.iter().map(|b| steps(&b.box_system[m])).fold(0.0, f64::max);
            assert!(worst <= CALIBRATION_STEPS, "{name}: {mode} is {worst:.2} steps from box coverage on the full ladder");
        }
    }
    let full = results.iter().filter(|r| !r.0.starts_with("calibration-")).count() == CORPUS.len() && sizes == ladder();
    if full {
        for (name, _, besides) in results.iter().filter(|r| !r.0.starts_with("calibration-")) {
            let rows: Vec<GlyphScaleRow> = besides.iter().map(|b| b.row).collect();
            let published = PUBLISHED_MATCHES_DOWN_TO.iter().find(|(n, _)| n == name).map(|p| p.1);
            assert_eq!(matches_down_to(&rows), published, "{name}: the oracle run no longer reproduces L-168's curve");
        }
    }
    println!("reproduces L-168's matches_down_to: {}", if full { "asserted" } else { "not asserted, partial run" });
}
