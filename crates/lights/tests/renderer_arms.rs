#![cfg(feature = "fit")]
use lights::fit::Target;
use lights::{Capture, Clip, DrawList, DrawOp, DrawShape, Ink, Rgb, Similarity2, Stand, Vec2};
use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU32;
use std::path::PathBuf;

#[path = "../examples/support/tiny_arm.rs"]
mod tiny_arm;

use tiny_arm::{QualityArm, TinyList};

const ARMS: &str = "docs/lights/benchmarks/l176/arms.tsv";
const QUALITY: &str = "docs/lights/benchmarks/l176/quality.tsv";
const DRAWLIST: &str = "docs/lights/benchmarks/l175/drawlist.tsv";
const BASELINE: &str = "docs/lights/benchmarks/l174/baseline.tsv";
const OPS: &str = "docs/lights/benchmarks/l175/ops";
const SLUG: &str = "l176-renderer-arms";
const COMMAND: &str = "cargo run -p lights --release --all-features --example renderer_arms -- --ops docs/lights/benchmarks/l175/ops --out docs/lights/benchmarks/l176";
const SCENES: [&str; 11] = [
    "overview", "pipeline", "hit_test", "focus", "catalog", "layout", "tree", "container", "graph", "canvas", "timeline",
];
const NAMED: [&str; 3] = ["tree", "canvas", "overview"];
const BUDGET_MS: f64 = 16.67;

type Row = BTreeMap<String, String>;

struct Table {
    header: BTreeMap<String, String>,
    rows: Vec<Row>,
}

fn path(rel: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p.push(rel);
    p
}

fn read(rel: &str) -> String {
    let p = path(rel);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()))
}

fn parse_table(text: &str, what: &str) -> Table {
    let mut header = BTreeMap::new();
    let mut body = Vec::new();
    for line in text.lines() {
        match line.strip_prefix('#') {
            Some(rest) => {
                let (k, v) = rest.split_once('\t').unwrap_or((rest, ""));
                header.entry(k.to_string()).or_insert_with(|| v.to_string());
            }
            None => body.push(line),
        }
    }
    let columns: Vec<&str> = body.first().unwrap_or_else(|| panic!("{what} has no column header")).split('\t').collect();
    let rows = body[1..]
        .iter()
        .map(|l| {
            let cells: Vec<&str> = l.split('\t').collect();
            assert_eq!(cells.len(), columns.len(), "{what}: a row of {} cells under {} columns", cells.len(), columns.len());
            columns.iter().zip(cells).map(|(c, v)| (c.to_string(), v.to_string())).collect()
        })
        .collect();
    Table { header, rows }
}

fn num(row: &Row, k: &str) -> f64 {
    row.get(k).and_then(|v| v.parse::<f64>().ok()).unwrap_or_else(|| panic!("{k} is {:?}, not a number, in {row:?}", row.get(k)))
}

fn record() -> String {
    let text = read("docs/lights/RESEARCH.json");
    let key = format!("\"slug\": \"{SLUG}\"");
    let start = text.find(&key).unwrap_or_else(|| panic!("docs/lights/RESEARCH.json has no record {SLUG}"));
    let rest = &text[start + key.len()..];
    let end = rest.find("\"slug\": ").map_or(text.len(), |i| start + key.len() + i);
    text[start..end].to_string()
}

fn v(x: f32, y: f32) -> Vec2<f32> {
    Vec2::new(x, y)
}

fn ink() -> Ink<f32> {
    Ink::straight(Rgb::new(0.2, 0.6, 0.9), 0.8)
}

fn shapes() -> Vec<(&'static str, DrawShape<f32>)> {
    let star: Vec<Vec2<f32>> = (0..5)
        .map(|i| {
            let t = -std::f32::consts::FRAC_PI_2 + i as f32 * 4.0 * std::f32::consts::PI / 5.0;
            v(40.3 + 22.0 * t.cos(), 33.7 + 22.0 * t.sin())
        })
        .collect();
    vec![
        ("rect", DrawShape::Rect { min: v(10.3, 12.6), max: v(61.8, 40.2), corner_radius: 0.0 }),
        ("rounded rect", DrawShape::Rect { min: v(20.25, 15.5), max: v(50.25, 45.5), corner_radius: 15.0 }),
        ("outline", DrawShape::RectOutline { min: v(9.4, 8.7), max: v(70.2, 50.9), corner_radius: 6.0, half_width: 1.5 }),
        ("sharp outline", DrawShape::RectOutline { min: v(12.5, 11.25), max: v(66.0, 44.0), corner_radius: 0.0, half_width: 2.0 }),
        ("segment", DrawShape::Segment { a: v(8.2, 9.9), b: v(70.6, 52.3), half_width: 2.5 }),
        ("disc", DrawShape::Disc { centre: v(40.4, 30.6), radius: 17.3 }),
        ("ring", DrawShape::Ring { centre: v(39.6, 31.2), radius: 16.0, half_width: 2.25 }),
        ("star polygon", DrawShape::Polygon { points: star }),
        ("polyline", DrawShape::Polyline { points: vec![v(8.0, 50.0), v(30.5, 10.5), v(52.0, 48.0), v(74.0, 12.0)], closed: false, half_width: 1.75 }),
    ]
}

fn one(shape: DrawShape<f32>) -> DrawList<f32> {
    let mut list = DrawList::new();
    list.push(DrawOp::new(shape, ink())).expect("a valid op");
    list
}

#[test]
fn tiny_skia_draws_each_shape_with_the_ink_lights_draws() {
    let mut checked = 0;
    for (name, shape) in shapes() {
        for scale in [1.0f32, 2.0] {
            let (w, h) = ((84.0 * scale) as u32, (64.0 * scale) as u32);
            let view = Similarity2::uniform(scale);
            let list = one(shape.clone());
            let lights = tiny_arm::layer_ink(&list.render_through(view, w, h).expect("lights renders"));
            let tiny_list = TinyList::from_list(&list, view).expect("an axis-aligned view");
            assert_eq!(tiny_list.len(), 1, "{name}: the converter dropped the op");
            let tiny = tiny_arm::pixmap_ink(&tiny_arm::render(&tiny_list, w, h, None).expect("a pixmap"));
            let tolerance = 0.01 * lights + 2.0;
            assert!(
                (tiny - lights).abs() <= tolerance,
                "{name} @{scale}x: tiny-skia inks {tiny:.2} px and lights {lights:.2} px, beyond {tolerance:.2}"
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 18, "nine shapes at two scales");
}

#[test]
fn a_figure_that_crosses_its_clip_is_drawn_only_inside_it() {
    let clip = [20.5f32, 14.25, 48.0, 40.0];
    for (name, shape) in shapes() {
        let (w, h) = (84u32, 64u32);
        let mut list = DrawList::new();
        list.push(DrawOp::new(shape.clone(), Ink::opaque(Rgb::splat(1.0))).clipped(Clip::new(v(clip[0], clip[1]), v(clip[2], clip[3]))))
            .expect("a valid op");
        let view = Similarity2::uniform(1.0f32);
        let tiny_list = TinyList::from_list(&list, view).expect("an axis-aligned view");
        let Some(pixmap) = tiny_arm::render(&tiny_list, w, h, None) else {
            panic!("{name}: no pixmap");
        };
        let mut whole = DrawList::new();
        whole.push(DrawOp::new(shape.clone(), Ink::opaque(Rgb::splat(1.0)))).expect("a valid op");
        let unclipped = tiny_arm::render(&TinyList::from_list(&whole, view).expect("view"), w, h, None).expect("a pixmap");
        let alpha = |p: &tiny_skia::Pixmap, x: u32, y: u32| p.data()[((y * w + x) * 4 + 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let (fx, fy) = (x as f32, y as f32);
                let outside = fx + 1.0 <= clip[0] || fx >= clip[2] || fy + 1.0 <= clip[1] || fy >= clip[3];
                if outside {
                    assert_eq!(alpha(&pixmap, x, y), 0, "{name}: pixel ({x}, {y}) lies outside the clip and is inked");
                }
                let interior = fx >= 21.0 && fx + 1.0 <= 48.0 && fy >= 15.0 && fy + 1.0 <= 40.0;
                if interior {
                    let (a, b) = (alpha(&pixmap, x, y), alpha(&unclipped, x, y));
                    assert!(a.abs_diff(b) <= 1, "{name}: pixel ({x}, {y}) inside the clip reads {a} clipped and {b} unclipped");
                }
            }
        }
        let edge = (15..40).filter(|y| alpha(&unclipped, 20, *y) == 255).map(|y| alpha(&pixmap, 20, y)).collect::<Vec<u8>>();
        for a in edge {
            assert!((126..=129).contains(&a), "{name}: a pixel half inside the clip at x = 20.5 reads {a}, not half");
        }
    }
}

#[test]
fn tiny_skias_bands_keep_the_frames_ink_and_lose_hairlines_at_their_edges() {
    let text = read(&format!("{OPS}/ops_canvas_1.tsv"));
    let capture = Capture::<f32>::parse(&text).expect("a capture");
    let (list, _) = capture.draw_list(&[Stand::Native, Stand::Proxy]);
    let view = Similarity2::uniform(1.0f32);
    let tiny = TinyList::from_list(&list, view).expect("view");
    let (w, h) = (1920u32, 1080u32);
    let serial = tiny_arm::render(&tiny, w, h, None).expect("serial");
    let ink = tiny_arm::pixmap_ink(&serial);
    assert!(ink > 1.0e6, "the canvas frame inks almost nothing, so this compared nothing");
    for rows in [7u32, 16, 64] {
        let banded = tiny_arm::render(&tiny, w, h, NonZeroU32::new(rows)).expect("banded");
        let (max, px) = tiny_arm::pixmap_difference(&serial, &banded);
        let band_ink = tiny_arm::pixmap_ink(&banded);
        println!("{rows}-row bands: {max} levels at {px} pixels, ink {band_ink:.1} against {ink:.1}");
        assert!((band_ink - ink).abs() <= 0.01 * ink, "{rows}-row bands ink {band_ink} against {ink} serially");
        assert!(px > 0, "{rows}-row bands drew tiny-skia's serial frame exactly, which this record says they do not");
        assert!(px <= 10_000, "{rows}-row bands depart from the serial frame at {px} pixels, far beyond a band edge's");
    }
    let mut thin = TinyList::from_list(&list, view).expect("view");
    thin.ops.retain(|o| matches!(&o.figure, tiny_arm::Figure::Stroke(_, s) if s.width <= 1.0) && !o.masked);
    assert!(thin.len() >= 50, "only {} unmasked hairlines in the canvas frame", thin.len());
    let alone = tiny_arm::pixmap_ink(&tiny_arm::render(&thin, w, h, None).expect("serial"));
    let one_row = tiny_arm::pixmap_ink(&tiny_arm::render(&thin, w, h, NonZeroU32::new(1)).expect("banded"));
    println!("{} hairlines: {alone:.1} px of ink serially, {one_row:.1} in one-row bands", thin.len());
    assert!(alone > 1.0e4 && one_row < 0.1 * alone, "hairlines ink {alone} serially and {one_row} in one-row bands");
}

#[test]
fn the_parallel_pack_writes_the_bytes_the_serial_pack_writes() {
    let mut list = DrawList::new();
    for (_, shape) in shapes() {
        list.push(DrawOp::new(shape, ink())).expect("a valid op");
    }
    let layer = list.render_through(Similarity2::uniform(1.0), 84, 64).expect("a render");
    let serial = tiny_arm::pack_rgba(&layer);
    assert_eq!(serial.len(), 84 * 64 * 4, "four bytes a pixel");
    assert!(serial.chunks_exact(4).any(|p| p[3] > 0 && p[3] < 255), "no partial alpha, so the alpha channel went unchecked");
    for rows in [1u32, 16, 100] {
        assert_eq!(tiny_arm::pack_rgba_parallel(&layer, NonZeroU32::new(rows).expect("rows")), serial, "{rows}-row bands");
    }
}

#[test]
fn every_target_converts_to_a_list_both_renderers_draw() {
    for target in Target::<f64>::all() {
        let list = tiny_arm::target_list(&target).unwrap_or_else(|| panic!("{} has no DrawList", target.name));
        assert_eq!(list.len(), target.set.parts.len() + target.set.holes.iter().filter(|h| matches!(h.shape, lights::fit::Primitive::Disc { .. })).count(), "{}", target.name);
        for arm in [QualityArm::Lights, QualityArm::Tiny] {
            let image = tiny_arm::quality_image(arm, &target, 1.0).unwrap_or_else(|| panic!("{} {}", arm.name(), target.name));
            let ink: f64 = image.pixels.iter().map(|p| p.x).sum();
            assert!(ink > 3.0, "{} drew {} with {ink} px of ink", arm.name(), target.name);
        }
    }
}

#[test]
fn the_quality_table_is_what_each_arm_renders_today() {
    let t = parse_table(&read(QUALITY), QUALITY);
    let targets = Target::<f64>::all();
    assert_eq!(t.rows.len(), targets.len() * 5, "six targets: two arms at two scales plus the exact-area row");
    let mut seen = 0;
    for r in t.rows.iter().filter(|r| r["arm"] != "exact-area") {
        let target = targets.iter().find(|x| x.name == r["target"]).unwrap_or_else(|| panic!("no target {}", r["target"]));
        let arm = if r["arm"] == "tiny-skia" { QualityArm::Tiny } else { QualityArm::Lights };
        let m = tiny_arm::quality(arm, target, num(r, "ppp")).unwrap_or_else(|| panic!("{} {}", r["arm"], r["target"]));
        let at = format!("{} {} @{}x", r["arm"], r["target"], r["ppp"]);
        assert_eq!(r["meets"], if m.meets { "yes" } else { "no" }, "{at}: the table's verdict is not today's");
        assert_eq!(num(r, "judged"), f64::from(m.judged), "{at}: judged");
        for (k, now) in [
            ("transition_10_90", m.transition_10_90),
            ("contrast_frac", m.contrast_frac),
            ("peak_frac", m.peak_frac),
            ("fwhm", m.fwhm),
            ("apex_displacement", m.apex_displacement),
            ("interior_ripple", m.interior_ripple),
        ] {
            let was = r[k].as_str();
            if was == "nan" {
                assert!(!now.is_finite(), "{at}: {k} was not measured and now reads {now}");
            } else {
                let was: f64 = was.parse().unwrap_or_else(|_| panic!("{at}: {k} = {was}"));
                assert!((was - now).abs() <= 1.0e-4, "{at}: {k} was {was} and is {now}");
            }
        }
        seen += 1;
    }
    assert_eq!(seen, targets.len() * 4);
}

#[test]
fn the_arms_table_is_a_release_run_of_the_command_the_record_quotes() {
    let t = parse_table(&read(ARMS), ARMS);
    assert_eq!(t.header.get("build").map(String::as_str), Some("release"), "L-176 times a release build");
    let args = COMMAND.split_once(" -- ").map(|(_, a)| a).unwrap_or_default();
    let invocation = t.header.get("invocation").cloned().unwrap_or_default();
    assert!(invocation.ends_with(&format!("/renderer_arms {args}")), "{ARMS} was written by {invocation:?}");
    assert!(record().contains(COMMAND), "{SLUG} does not quote {COMMAND:?}");
    for r in &t.rows {
        assert!(num(r, "reps") >= 5.0, "{} @{}x has {} reps", r["scene"], r["ppp"], r["reps"]);
    }
}

#[test]
fn both_renderers_drew_the_same_frames_and_the_lights_arm_is_l175s() {
    let t = parse_table(&read(ARMS), ARMS);
    let key = |r: &Row| (r["scene"].clone(), r["ppp"].clone(), r["list"].clone(), r["renderer"].clone(), r["mode"].clone());
    let keys: BTreeSet<_> = t.rows.iter().map(key).collect();
    assert_eq!(keys.len(), t.rows.len(), "a job appears twice");
    assert_eq!(t.rows.len(), SCENES.len() * 2 * 2 * 2 * 2, "eleven scenes, two scales, two lists, two renderers, two modes");
    let old = parse_table(&read(DRAWLIST), DRAWLIST);
    for r in t.rows.iter().filter(|r| r["renderer"] == "lights") {
        let at = format!("{} @{}x {} {}", r["scene"], r["ppp"], r["list"], r["mode"]);
        let twin = old
            .rows
            .iter()
            .find(|x| x["precision"] == "f32" && (&x["scene"], &x["ppp"], &x["list"], &x["mode"]) == (&r["scene"], &r["ppp"], &r["list"], &r["mode"]))
            .unwrap_or_else(|| panic!("{DRAWLIST} has no {at}"));
        assert_eq!(r["frame_hash"], twin["frame_hash"], "{at}: the lights arm drew other bits than L-175 did");
        assert_eq!(r["ops"], twin["ops"], "{at}: ops");
        let tiny = t
            .rows
            .iter()
            .find(|x| x["renderer"] == "tiny-skia" && (&x["scene"], &x["ppp"], &x["list"], &x["mode"]) == (&r["scene"], &r["ppp"], &r["list"], &r["mode"]))
            .unwrap_or_else(|| panic!("{at} has no tiny-skia row"));
        let (li, ti) = (num(r, "paint_px"), num(tiny, "paint_px"));
        assert!((ti - li).abs() <= 0.01 * li, "{at}: tiny-skia paints {ti} px and lights {li} px");
        assert!((num(tiny, "band_paint_px") - ti).abs() <= 0.01 * ti, "{at}: tiny-skia's bands paint {} against {ti}", tiny["band_paint_px"]);
        assert_eq!(r["band_paint_px"], r["paint_px"], "{at}: lights' bands are its serial frame");
        assert_eq!(num(r, "ink_px"), num(r, "width_px") * num(r, "height_px"), "{at}: an optical frame is opaque, so its ink is its area");
        assert!(num(tiny, "drawn") <= num(tiny, "ops"), "{at}: drawn");
        for x in [r, tiny] {
            let (fixed, opsx, full) = (num(x, "fixed_ms"), num(x, "opsx_ms"), num(x, "full_ms"));
            let per_op = (opsx - fixed) * 1e6 / (32.0 * num(x, "ops"));
            assert!((num(x, "per_op_ns") - per_op).abs() <= 1e-4 * 1e6 / (32.0 * num(x, "ops")) + 0.05 + 1e-9, "{at}: per_op_ns");
            let px = full - fixed - num(x, "ops_ms");
            assert!((num(x, "px_ms") - px).abs() <= 3e-4, "{at}: px_ms");
            assert!((num(x, "page_ms") - (num(x, "touch_ms") - fixed)).abs() <= 2e-4, "{at}: page_ms");
            let pixels = num(x, "width_px") * num(x, "height_px");
            let per_px = if x["renderer"] == "lights" { 24.0 } else { 4.0 };
            assert_eq!(num(x, "frame_bytes"), pixels * per_px, "{at}: frame_bytes");
        }
        assert!(num(r, "pack_ms") > 0.0 && num(tiny, "pack_ms") == 0.0, "{at}: only lights pays a pack");
    }
}

fn kill_condition(t: &Table, base: &Table) -> (bool, String) {
    let mut lines = Vec::new();
    let mut holds = true;
    for scene in NAMED {
        for ppp in ["1", "2"] {
            let row = |renderer: &str| {
                t.rows
                    .iter()
                    .find(|r| r["scene"] == scene && r["ppp"] == ppp && r["list"] == "framed" && r["renderer"] == renderer && r["mode"] == "parallel")
                    .unwrap_or_else(|| panic!("no {renderer} {scene} @{ppp}x"))
            };
            let (l, s) = (row("lights"), row("tiny-skia"));
            let lights_ms = num(l, "full_ms") + num(l, "pack_ms");
            let tiny_ms = num(s, "full_ms");
            let b = base
                .rows
                .iter()
                .find(|r| r["scene"] == scene && r["ppp"] == ppp)
                .unwrap_or_else(|| panic!("{BASELINE} has no {scene} @{ppp}x"));
            let epaint = num(b, "tess_ms") + num(b, "prep_ms") + num(b, "encode_ms");
            let near = lights_ms.max(tiny_ms) / lights_ms.min(tiny_ms) <= 1.3;
            let both_lose = lights_ms - epaint > BUDGET_MS && tiny_ms - epaint > BUDGET_MS;
            holds = holds && near && both_lose;
            lines.push(format!("{scene}@{ppp}x lights {lights_ms:.4} tiny {tiny_ms:.4} epaint {epaint:.4} near {near} both_lose {both_lose}"));
        }
    }
    (holds, lines.join("; "))
}

#[test]
fn the_kill_condition_is_decided_by_the_tables_and_the_record_says_which_way() {
    let t = parse_table(&read(ARMS), ARMS);
    let base = parse_table(&read(BASELINE), BASELINE);
    for k in ["tess_ms", "prep_ms", "encode_ms"] {
        assert!(base.rows.iter().all(|r| r.contains_key(k)), "{BASELINE} lost its {k} column");
    }
    let (holds, why) = kill_condition(&t, &base);
    let rec = record();
    let (said, unsaid) = if holds {
        ("KILL CONDITION HOLDS", "KILL CONDITION DOES NOT HOLD")
    } else {
        ("KILL CONDITION DOES NOT HOLD", "KILL CONDITION HOLDS")
    };
    assert!(rec.contains(said), "the tables say {said:?} ({why}) and {SLUG} does not");
    assert!(!rec.contains(unsaid), "the tables say {said:?} ({why}) and {SLUG} also says {unsaid:?}");
}

#[test]
fn the_research_record_quotes_the_tables_it_was_written_from() {
    let t = parse_table(&read(ARMS), ARMS);
    let rec = record();
    for scene in NAMED {
        for r in t.rows.iter().filter(|r| r["scene"] == scene && r["list"] == "framed" && r["mode"] == "parallel") {
            assert!(rec.contains(&r["full_ms"]), "{SLUG} never quotes {scene} @{}x {} full_ms = {}", r["ppp"], r["renderer"], r["full_ms"]);
        }
    }
}
