use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

const BASELINE: &str = "docs/lights/benchmarks/l174/baseline.tsv";
const REPS: &str = "docs/lights/benchmarks/l174/reps.tsv";
const SLUG: &str = "l174-optical-egui-baseline";
const COMMAND: &str =
    "cargo run -p optical --release --bin unified_showcase -- --frame-bench --out docs/lights/benchmarks/l174";
const ROLES: [&str; 3] = ["text-heavy", "fill-heavy", "op-heavy"];
const POINTS: [f64; 2] = [1920.0, 1080.0];

type Row = BTreeMap<String, String>;

struct Table {
    header: BTreeMap<String, String>,
    rows: Vec<Row>,
}

fn repo(rel: &str) -> String {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p.push(rel);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()))
}

fn table(rel: &str) -> Table {
    let text = repo(rel);
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
    let columns: Vec<&str> = body.first().unwrap_or_else(|| panic!("{rel} has no column header")).split('\t').collect();
    let rows = body[1..]
        .iter()
        .map(|l| {
            let cells: Vec<&str> = l.split('\t').collect();
            assert_eq!(cells.len(), columns.len(), "{rel}: a row of {} cells under {} columns", cells.len(), columns.len());
            columns.iter().zip(cells).map(|(c, v)| (c.to_string(), v.to_string())).collect()
        })
        .collect();
    Table { header, rows }
}

fn num(row: &Row, k: &str) -> f64 {
    row.get(k)
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or_else(|| panic!("{} @{}x: {k} is {:?}, not a number", row["scene"], row["ppp"], row.get(k)))
}

fn declared_roles(t: &Table) -> BTreeMap<String, String> {
    let line = t.header.get("roles").unwrap_or_else(|| panic!("{BASELINE} names no roles"));
    line.split('\t')
        .map(|pair| {
            let (role, scene) = pair.split_once('=').unwrap_or_else(|| panic!("#roles cell {pair:?}"));
            (role.to_string(), scene.to_string())
        })
        .collect()
}

fn record() -> String {
    let text = repo("docs/lights/RESEARCH.json");
    let key = format!("\"slug\": \"{SLUG}\"");
    let start = text.find(&key).unwrap_or_else(|| panic!("docs/lights/RESEARCH.json has no record {SLUG}"));
    let rest = &text[start + key.len()..];
    let end = rest.find("\"slug\": ").map_or(text.len(), |i| start + key.len() + i);
    text[start..end].to_string()
}

#[test]
fn the_optical_baseline_is_a_release_run_timed_by_a_pass_timestamp_over_five_reps() {
    let t = table(BASELINE);
    assert_eq!(t.header.get("build").map(String::as_str), Some("release"), "L-174's close bar is release only");
    let clock = t.header.get("gpu_clock").cloned().unwrap_or_default();
    assert!(clock.starts_with("timestamp-query"), "the GPU column must come from a timestamp query, and it says {clock:?}");
    assert!(!t.rows.is_empty(), "{BASELINE} holds no rows");
    for r in &t.rows {
        assert!(num(r, "reps") >= 5.0, "{} @{}x has {} reps; five is the floor", r["scene"], r["ppp"], r["reps"]);
    }
    let args = COMMAND.split_once(" -- ").map(|(_, a)| a).unwrap_or_default();
    let invocation = t.header.get("invocation").cloned().unwrap_or_default();
    assert!(
        invocation.ends_with(&format!("/unified_showcase {args}")),
        "{BASELINE} was written by {invocation:?}, not by the command {SLUG} quotes"
    );
    assert!(record().contains(COMMAND), "{SLUG}'s command is not {COMMAND:?}");
}

#[test]
fn every_scene_was_measured_at_both_scales_and_every_rep_drew_the_same_frames() {
    let t = table(BASELINE);
    let reps = table(REPS);
    let scenes: BTreeSet<&str> = t.rows.iter().map(|r| r["scene"].as_str()).collect();
    assert!(scenes.len() >= 3, "three named scenes need at least three scenes, found {scenes:?}");
    for scene in &scenes {
        for ppp in [1.0, 2.0] {
            let r = t
                .rows
                .iter()
                .find(|r| r["scene"] == *scene && num(r, "ppp") == ppp)
                .unwrap_or_else(|| panic!("{scene} was not measured at {ppp}x"));
            let at = format!("{scene} @{ppp}x");
            assert_eq!(r["deterministic"], "yes", "{at}: the reps drew different frames");
            assert_eq!(num(r, "width_px"), POINTS[0] * ppp, "{at}");
            assert_eq!(num(r, "height_px"), POINTS[1] * ppp, "{at}");
            assert!(num(r, "ink_px") > 0.0, "{at}: the timed pass drew nothing");
            assert_eq!(num(r, "upload_bytes"), 0.0, "{at}: the font atlas still grew after warm-up");
            let own: Vec<&Row> = reps.rows.iter().filter(|x| x["scene"] == *scene && num(x, "ppp") == ppp).collect();
            assert_eq!(own.len() as f64, num(r, "reps"), "{at}: the summary counts reps that {REPS} does not hold");
            for x in own {
                assert_eq!(x["counters_hash"], r["counters_hash"], "{at} rep {}: its frames differ from the summary's", x["rep"]);
            }
        }
    }
}

#[test]
fn the_three_named_scenes_are_the_ones_the_role_rule_picks() {
    let t = table(BASELINE);
    let declared = declared_roles(&t);
    let base: Vec<&Row> = t.rows.iter().filter(|r| num(r, "ppp") == 1.0).collect();
    let rules: [fn(&Row) -> f64; 3] = [
        |r| num(r, "text_vertices") / num(r, "vertices"),
        |r| num(r, "op_area_px"),
        |r| num(r, "visible"),
    ];
    let mut taken: Vec<String> = Vec::new();
    for (role, score) in ROLES.iter().zip(rules) {
        let best = base
            .iter()
            .filter(|r| !taken.contains(&r["scene"]))
            .max_by(|a, b| score(a).total_cmp(&score(b)))
            .unwrap_or_else(|| panic!("no scene left for {role}"));
        assert_eq!(declared.get(*role), Some(&best["scene"]), "{role}: the rule picks {}, the file names {:?}", best["scene"], declared.get(*role));
        taken.push(best["scene"].clone());
    }
    assert_eq!(declared.len(), ROLES.len(), "{declared:?}");
    for r in &t.rows {
        let named = declared.iter().find(|(_, s)| **s == r["scene"]).map_or("-", |(role, _)| role.as_str());
        assert_eq!(r["role"], named, "{} @{}x carries role {}", r["scene"], r["ppp"], r["role"]);
    }
}

#[test]
fn the_research_record_quotes_the_baseline_it_was_written_from() {
    let t = table(BASELINE);
    let rec = record();
    for scene in declared_roles(&t).values() {
        for r in t.rows.iter().filter(|r| r["scene"] == *scene) {
            for k in ["cpu_ms", "gpu_ms"] {
                assert!(rec.contains(&r[k]), "{SLUG} never quotes {scene} @{}x {k} = {}", r["ppp"], r[k]);
            }
        }
    }
}
