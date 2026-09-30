use lights::{Capture, Similarity2, Stand};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::process::Command;

const OPS: &str = "docs/lights/benchmarks/l175/ops";
const TABLE: &str = "docs/lights/benchmarks/l175/drawlist.tsv";
const BASELINE: &str = "docs/lights/benchmarks/l174/baseline.tsv";
const SLUG: &str = "l175-drawlist-ui-frame";
const COMMAND: &str = "cargo run -p lights --release --all-features --bin lights-ui-frame -- --ops docs/lights/benchmarks/l175/ops --out docs/lights/benchmarks/l175/drawlist.tsv";
const CAPTURE: &str = "cargo run -p optical --release --bin unified_showcase -- --frame-bench --reps 1 --out /tmp/l175_frame_scratch --ops-out docs/lights/benchmarks/l175/ops";
const SCENES: [&str; 11] = [
    "overview", "pipeline", "hit_test", "focus", "catalog", "layout", "tree", "container", "graph", "canvas", "timeline",
];
const NAMED: [&str; 3] = ["tree", "canvas", "overview"];
const GROWING: &str = "timeline";
const REPEAT: f64 = 32.0;

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

fn census(text: &str) -> BTreeMap<String, f64> {
    let line = text.lines().find_map(|l| l.strip_prefix("#census\t")).expect("a #census line");
    line.split(' ')
        .filter_map(|kv| kv.split_once('='))
        .map(|(k, v)| (k.to_string(), v.parse::<f64>().unwrap_or_else(|_| panic!("#census {k}={v}"))))
        .collect()
}

fn header_line<'a>(text: &'a str, key: &str) -> &'a str {
    text.lines().find_map(|l| l.strip_prefix(&format!("#{key}\t"))).unwrap_or_else(|| panic!("no #{key} line"))
}

fn record() -> String {
    let text = read("docs/lights/RESEARCH.json");
    let key = format!("\"slug\": \"{SLUG}\"");
    let start = text.find(&key).unwrap_or_else(|| panic!("docs/lights/RESEARCH.json has no record {SLUG}"));
    let rest = &text[start + key.len()..];
    let end = rest.find("\"slug\": ").map_or(text.len(), |i| start + key.len() + i);
    text[start..end].to_string()
}

fn printed(ops: f64) -> f64 {
    1e-4 * 1e6 / (REPEAT * ops) + 0.05 + 1e-9
}

fn capture_text(scene: &str, ppp: u32) -> String {
    read(&format!("{OPS}/ops_{scene}_{ppp}.tsv"))
}

#[test]
fn every_scene_was_captured_at_both_scales_from_the_last_frame_of_rep_zero() {
    let mut names: Vec<String> = std::fs::read_dir(path(OPS))
        .expect("the capture directory")
        .map(|e| e.expect("an entry").file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    let mut want: Vec<String> = SCENES.iter().flat_map(|s| [1, 2].map(|p| format!("ops_{s}_{p}.tsv"))).collect();
    want.sort();
    assert_eq!(names, want, "one capture per scene and scale, and nothing else");
    for scene in SCENES {
        for ppp in [1u32, 2] {
            let at = format!("{scene} @{ppp}x");
            let text = capture_text(scene, ppp);
            assert_eq!(header_line(&text, "child"), format!("--frame-bench-child {scene} {ppp} 0 30 60 1920 1080"), "{at}");
            assert!(header_line(&text, "frame").starts_with("index=89 of=90 "), "{at}: not the last frame of the script");
            let walked: f64 = header_line(&text, "walked").parse().expect("a #walked count");
            assert_eq!(walked, census(&text)["ops"], "{at}: the dump and the census walked different shapes");
            for c in [Capture::<f64>::parse(&text).map(|c| c.ops.iter().map(|o| o.shape).max()), Capture::<f32>::parse(&text).map(|c| c.ops.iter().map(|o| o.shape).max())] {
                let last = c.unwrap_or_else(|e| panic!("{at}: {e}")).unwrap_or_else(|| panic!("{at}: no op"));
                assert!((last as f64) < walked, "{at}: shape {last} of {walked}");
            }
        }
    }
}

#[test]
fn each_capture_is_a_frame_l174_measured() {
    let base = parse_table(&read(BASELINE), BASELINE);
    for scene in SCENES {
        for ppp in [1u32, 2] {
            let at = format!("{scene} @{ppp}x");
            let b = base
                .rows
                .iter()
                .find(|r| r["scene"] == scene && num(r, "ppp") == f64::from(ppp))
                .unwrap_or_else(|| panic!("{BASELINE} has no {at}"));
            let c = census(&capture_text(scene, ppp));
            if scene == GROWING {
                assert!(c["ops"] >= num(b, "ops"), "{at}: the last frame of a growing scene has {} ops, fewer than the mean {}", c["ops"], b["ops"]);
                continue;
            }
            for k in ["ops", "visible", "rect", "circle", "line", "path", "text", "mesh", "glyphs"] {
                let mean = num(b, k);
                assert!((c[k] - mean).abs() <= (0.01 * mean).max(2.0), "{at}: {k} {} in the capture against a mean of {mean} in L-174", c[k]);
            }
        }
    }
}

#[test]
fn the_table_is_a_release_run_of_the_command_the_record_quotes() {
    let t = parse_table(&read(TABLE), TABLE);
    assert_eq!(t.header.get("build").map(String::as_str), Some("release"), "L-175 times a release build");
    let args = COMMAND.split_once(" -- ").map(|(_, a)| a).unwrap_or_default();
    let invocation = t.header.get("invocation").cloned().unwrap_or_default();
    assert!(invocation.ends_with(&format!("/lights-ui-frame {args}")), "{TABLE} was written by {invocation:?}");
    let rec = record();
    assert!(rec.contains(COMMAND), "{SLUG} does not quote {COMMAND:?}");
    assert!(rec.contains(CAPTURE), "{SLUG} does not quote the capture command {CAPTURE:?}");
    for r in &t.rows {
        assert!(num(r, "reps") >= 5.0, "{} @{}x has {} reps", r["scene"], r["ppp"], r["reps"]);
    }
}

#[test]
fn every_capture_was_timed_in_both_modes_and_the_modes_drew_the_same_bits() {
    let t = parse_table(&read(TABLE), TABLE);
    let key = |r: &Row| (r["scene"].clone(), r["ppp"].clone(), r["precision"].clone(), r["list"].clone(), r["mode"].clone());
    let keys: BTreeSet<_> = t.rows.iter().map(key).collect();
    assert_eq!(keys.len(), t.rows.len(), "a job appears twice");
    assert_eq!(t.rows.len(), SCENES.len() * 2 * 2 * 2 * 2, "eleven scenes, two scales, two precisions, two lists, two modes");
    let points = [1920.0, 1080.0];
    for r in t.rows.iter().filter(|r| r["mode"] == "serial") {
        let at = format!("{} @{}x {} {}", r["scene"], r["ppp"], r["precision"], r["list"]);
        let twin = t
            .rows
            .iter()
            .find(|x| x["mode"] == "parallel" && (&x["scene"], &x["ppp"], &x["precision"], &x["list"]) == (&r["scene"], &r["ppp"], &r["precision"], &r["list"]))
            .unwrap_or_else(|| panic!("{at} has no parallel row"));
        assert_eq!(r["frame_hash"], twin["frame_hash"], "{at}: the modes drew different bits");
        assert!(num(twin, "threads") > 1.0, "{at}: the parallel row ran on one thread");
        let ppp = num(r, "ppp");
        let text = capture_text(&r["scene"], ppp as u32);
        let admit: &[Stand] = if r["list"] == "native" { &[Stand::Native] } else { &[Stand::Native, Stand::Proxy] };
        let (w, h) = ((points[0] * ppp) as u32, (points[1] * ppp) as u32);
        let (ops, visited) = if r["precision"] == "f32" {
            let (list, _) = Capture::<f32>::parse(&text).unwrap().draw_list(admit);
            let view = Similarity2::uniform(ppp as f32);
            (list.len(), list.ops().iter().filter_map(|o| o.footprint(view, w, h)).map(|[x0, y0, x1, y1]| u64::from(x1 - x0) * u64::from(y1 - y0)).sum::<u64>())
        } else {
            let (list, _) = Capture::<f64>::parse(&text).unwrap().draw_list(admit);
            let view = Similarity2::uniform(ppp);
            (list.len(), list.ops().iter().filter_map(|o| o.footprint(view, w, h)).map(|[x0, y0, x1, y1]| u64::from(x1 - x0) * u64::from(y1 - y0)).sum::<u64>())
        };
        for x in [r, twin] {
            assert_eq!(num(x, "ops"), ops as f64, "{at}: the table timed a list of {} ops and the capture builds {ops}", x["ops"]);
            assert_eq!(num(x, "visited_px"), visited as f64, "{at}: visited_px is not the sum of the footprints");
            let (fixed, opsx, full) = (num(x, "fixed_ms"), num(x, "opsx_ms"), num(x, "full_ms"));
            let per_op = (opsx - fixed) * 1e6 / (REPEAT * ops as f64);
            assert!(
                (num(x, "per_op_ns") - per_op).abs() <= printed(ops as f64),
                "{at} {}: per_op_ns {} is not (opsx - fixed) / (32 ops) = {per_op}",
                x["mode"],
                x["per_op_ns"]
            );
            let px = full - fixed - num(x, "ops_ms");
            assert!((num(x, "px_ms") - px).abs() <= 3e-4, "{at} {}: px_ms {} against {px}", x["mode"], x["px_ms"]);
            assert!((num(x, "page_ms") - (num(x, "touch_ms") - fixed)).abs() <= 2e-4, "{at} {}: page_ms", x["mode"]);
        }
        assert!(num(r, "px_share") >= 0.95, "{at}: the pixel term is {} of the serial frame", r["px_share"]);
    }
}

#[test]
fn the_research_record_quotes_the_table_it_was_written_from() {
    let t = parse_table(&read(TABLE), TABLE);
    let rec = record();
    for scene in NAMED {
        for r in t.rows.iter().filter(|r| r["scene"] == scene && r["precision"] == "f32" && r["list"] == "native") {
            assert!(rec.contains(&r["full_ms"]), "{SLUG} never quotes {scene} @{}x {} full_ms = {}", r["ppp"], r["mode"], r["full_ms"]);
        }
    }
}

#[test]
fn the_bench_decomposes_a_tiny_capture_by_the_formulas_its_header_states() {
    let dir = tempfile::Builder::new().prefix("lights-ui-frame-").tempdir().expect("a scratch dir");
    let ops = dir.path().join("ops");
    std::fs::create_dir_all(&ops).expect("ops dir");
    let text = capture_text("tree", 1);
    let rows: Vec<&str> = text.lines().filter(|l| !l.starts_with('#')).take(40).collect();
    std::fs::write(ops.join("ops_mini_1.tsv"), rows.join("\n") + "\n").expect("a mini capture");
    let out = dir.path().join("mini.tsv");
    let run = Command::new(env!("CARGO_BIN_EXE_lights-ui-frame"))
        .args(["--points", "96x48", "--reps", "5", "--band", "7", "--ops"])
        .arg(&ops)
        .arg("--out")
        .arg(&out)
        .output()
        .expect("the bench runs");
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stdout));
    let t = parse_table(&std::fs::read_to_string(&out).expect("the table"), "mini.tsv");
    assert_eq!(t.rows.len(), 8, "one scene, one scale, two precisions, two lists, two modes");
    assert!(t.header["arms"].contains(&format!("repeated {} times", REPEAT as u32)), "{}", t.header["arms"]);
    for r in &t.rows {
        assert_eq!((r["width_px"].as_str(), r["height_px"].as_str()), ("96", "48"));
        let (fixed, opsx) = (num(r, "fixed_ms"), num(r, "opsx_ms"));
        let per_op = (opsx - fixed) * 1e6 / (REPEAT * num(r, "ops"));
        assert!((num(r, "per_op_ns") - per_op).abs() <= printed(num(r, "ops")), "per_op_ns {} against {per_op}", r["per_op_ns"]);
        assert_eq!(r["band_rows"], if r["mode"] == "parallel" { "7" } else { "48" });
    }
    let refused = Command::new(env!("CARGO_BIN_EXE_lights-ui-frame")).args(["--ops", "x", "--out", "y", "--frobnicate", "1"]).output().expect("runs");
    assert_eq!(refused.status.code(), Some(2), "an unknown flag is refused");
    let missing = Command::new(env!("CARGO_BIN_EXE_lights-ui-frame")).args(["--reps", "5"]).output().expect("runs");
    assert_eq!(missing.status.code(), Some(2), "--ops and --out are required");
}
