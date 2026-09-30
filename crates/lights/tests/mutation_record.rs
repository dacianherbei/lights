use std::path::{Path, PathBuf};
use std::process::Command;

const RECORD: &str = "docs/lights/mutations/results.tsv";

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn record_text() -> String {
    let path = repo_root().join(RECORD);
    match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => panic!(
            "{} is missing: {e}. The mutation manifest is the only proof a test \
             can fail, and without a committed record of a run, 'the manifest is \
             green' is a claim that costs a full re-run to check. Produce it with \
             `python3 crates/lights/scripts/mutate_par.py dev 12 \
             /Volumes/RamBuild/mutpool_full`",
            path.display()
        ),
    }
}

fn header(text: &str, key: &str) -> String {
    let want = format!("#{key}\t");
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix(want.as_str()) {
            return rest.to_string();
        }
    }
    panic!("the record carries no #{key} line");
}

fn count(text: &str, key: &str) -> usize {
    let raw = header(text, key);
    match raw.trim().parse::<usize>() {
        Ok(v) => v,
        Err(e) => panic!("#{key} is {raw}, which is not a count: {e}"),
    }
}

struct Row {
    verdict: String,
    gates: String,
    run: String,
    file: String,
    test: String,
    name: String,
}

fn rows(text: &str) -> Vec<Row> {
    let mut columns: Vec<String> = Vec::new();
    let mut out = Vec::new();
    for line in text.lines() {
        if line.starts_with('#') {
            continue;
        }
        let cells: Vec<&str> = line.split('\t').collect();
        if columns.is_empty() {
            columns = cells.iter().map(|c| c.to_string()).collect();
            continue;
        }
        let at = |name: &str| -> String {
            let index = match columns.iter().position(|c| c == name) {
                Some(i) => i,
                None => panic!("the record has no {name} column"),
            };
            match cells.get(index) {
                Some(v) => (*v).to_string(),
                None => panic!("a row is short of the {name} column"),
            }
        };
        out.push(Row {
            verdict: at("verdict"),
            gates: at("gates"),
            run: at("run"),
            file: at("file"),
            test: at("test"),
            name: at("name"),
        });
    }
    out
}

fn manifest_rows() -> Vec<(String, String, String)> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts").join("mutations.json");
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => panic!("{}: {e}", path.display()),
    };
    let mut out = Vec::new();
    let mut file = String::new();
    let mut test = String::new();
    let mut name = String::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(v) = json_string(trimmed, "\"name\":") {
            if !name.is_empty() {
                out.push((file.clone(), test.clone(), name.clone()));
                file.clear();
                test.clear();
            }
            name = v;
        }
        if let Some(v) = json_string(trimmed, "\"file\":") {
            file = v;
        }
        if let Some(v) = json_string(trimmed, "\"test\":") {
            test = v;
        }
    }
    if !name.is_empty() {
        out.push((file, test, name));
    }
    out
}

fn json_string(line: &str, key: &str) -> Option<String> {
    let rest = line.strip_prefix(key)?;
    let rest = rest.trim();
    let inner = rest.strip_prefix('"')?;
    let end = find_unescaped_quote(inner)?;
    Some(unescape(&inner[..end]))
}

fn find_unescaped_quote(text: &str) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut i = 0usize;
    let mut escaped = false;
    for b in bytes.iter() {
        if escaped {
            escaped = false;
        } else if *b == b'\\' {
            escaped = true;
        } else if *b == b'"' {
            return Some(i);
        }
        i += 1;
    }
    None
}

fn unescape(text: &str) -> String {
    let mut units: Vec<u16> = Vec::new();
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            let mut buf = [0u16; 2];
            units.extend_from_slice(c.encode_utf16(&mut buf));
            continue;
        }
        match chars.next() {
            Some('n') => units.push(u16::from(b'\n')),
            Some('t') => units.push(u16::from(b'\t')),
            Some('r') => units.push(u16::from(b'\r')),
            Some('u') => {
                let hex: String = chars.by_ref().take(4).collect();
                match u16::from_str_radix(&hex, 16) {
                    Ok(unit) => units.push(unit),
                    Err(e) => panic!("\\u{hex} in the manifest is not a UTF-16 unit: {e}"),
                }
            }
            Some(other) => {
                let mut buf = [0u16; 2];
                units.extend_from_slice(other.encode_utf16(&mut buf));
            }
            None => (),
        }
    }
    String::from_utf16_lossy(&units)
}

#[test]
fn the_committed_record_is_of_a_full_run_and_says_so_in_its_own_header() {
    let text = record_text();
    assert!(
        header(&text, "invocation").contains("mutate_par.py"),
        "the record must name the command that produced it"
    );
    let declared = count(&text, "manifest_rows");
    let held = count(&text, "rows");
    assert_eq!(
        declared, held,
        "the record holds {held} of the manifest's {declared} rows. Every manifest \
         row must be in it — as a fresh result, or carried forward from an earlier \
         run under its own date — because a row the record does not name is a row \
         nothing has ever proven can fail"
    );
    assert!(declared > 0, "a record of nothing proves nothing");
    let evaluated = count(&text, "evaluated");
    assert!(
        evaluated <= held,
        "a run cannot have evaluated {evaluated} of {held} rows"
    );
}

#[test]
fn every_row_in_the_committed_record_was_caught_through_all_four_gates() {
    let text = record_text();
    let all = rows(&text);
    assert_eq!(
        all.len(),
        count(&text, "rows"),
        "the header's #rows must match the rows beneath it"
    );
    let caught: Vec<&Row> = all.iter().filter(|r| r.verdict == "CAUGHT").collect();
    assert_eq!(
        caught.len(),
        count(&text, "caught"),
        "the header's #caught must match the rows beneath it; a header that can \
         disagree with its own body is not a record"
    );
    for row in all.iter() {
        assert_eq!(
            row.verdict, "CAUGHT",
            "{} ({}) is {}: a mutation that does not compile tested nothing, and \
             one that escapes is a hole in the suite",
            row.name, row.file, row.verdict
        );
        assert!(
            !row.run.is_empty(),
            "{} carries no run date, so nothing says WHEN it was last proven to \
             fail; a carried-forward row that cannot be told from a fresh one is \
             how a record goes stale in place",
            row.name
        );
        assert_eq!(
            row.gates, "....",
            "{} shows gates {} — applied, compiled, ran, detected, and a dot is \
             the only clean mark. ..xx means the test filter matched nothing, \
             which is almost always a bare-function-name or a missing features \
             field rather than a driver fault",
            row.name, row.gates
        );
    }
}

#[test]
fn the_record_and_the_manifest_name_exactly_the_same_mutations() {
    let text = record_text();
    let mut recorded: Vec<(String, String, String)> = rows(&text)
        .into_iter()
        .map(|r| (r.file, r.test, r.name))
        .collect();
    let mut declared = manifest_rows();
    assert!(!declared.is_empty(), "the manifest parser read nothing");
    recorded.sort();
    declared.sort();
    let missing: Vec<&(String, String, String)> =
        declared.iter().filter(|d| !recorded.contains(d)).collect();
    let extra: Vec<&(String, String, String)> =
        recorded.iter().filter(|r| !declared.contains(r)).collect();
    assert!(
        missing.is_empty(),
        "{} manifest row(s) are in no committed result, so nothing has ever \
         proven they can fail. First: {:?}. Re-run mutate_par.py and commit the \
         record beside the manifest change",
        missing.len(),
        missing.first()
    );
    assert!(
        extra.is_empty(),
        "{} recorded row(s) are no longer in the manifest, so the record is of a \
         tree that has moved. First: {:?}",
        extra.len(),
        extra.first()
    );
}

fn fake_tree(dir: &Path) -> PathBuf {
    let scripts = dir.join("crates").join("lights").join("scripts");
    if let Err(e) = std::fs::create_dir_all(&scripts) {
        panic!("cannot build the fake tree: {e}");
    }
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("scripts")
        .join("mutate_par.py");
    let copy = scripts.join("mutate_par.py");
    if let Err(e) = std::fs::copy(&source, &copy) {
        panic!("cannot copy mutate_par.py: {e}");
    }
    if let Err(e) = std::fs::write(scripts.join("mutations.json"), "{\"mutations\": []}\n") {
        panic!("cannot write the empty manifest: {e}");
    }
    let crate_dir = dir.join("crates").join("lights");
    if let Err(e) = std::fs::write(
        crate_dir.join("Cargo.toml"),
        "[package]\nname = \"lights\"\nversion = \"0.0.0\"\n",
    ) {
        panic!("cannot write the fake manifest: {e}");
    }
    copy
}

#[test]
fn the_driver_writes_its_record_even_when_it_evaluated_nothing() {
    let dir = match tempfile::Builder::new().prefix("lights-mutrecord-").tempdir() {
        Ok(d) => d,
        Err(e) => panic!("tempdir: {e}"),
    };
    let script = fake_tree(dir.path());
    let mut command = Command::new("python3");
    command.arg(&script);
    command.arg("dev");
    command.arg("1");
    command.arg(dir.path().join("pool"));
    let output = match command.output() {
        Ok(o) => o,
        Err(e) => panic!("running the driver: {e}"),
    };
    let said = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.status.code().unwrap_or(-1), 0, "{said}");
    let written = dir.path().join(RECORD);
    assert!(
        written.is_file(),
        "the driver must write its record on EVERY run, not only on a run that \
         caught something — a verdict that exists only in a terminal is the \
         defect this record was added for. It said: {said}"
    );
    let text = match std::fs::read_to_string(&written) {
        Ok(t) => t,
        Err(e) => panic!("{}: {e}", written.display()),
    };
    assert_eq!(count(&text, "evaluated"), 0, "{text}");
    assert_eq!(count(&text, "caught"), 0, "{text}");
    assert_eq!(count(&text, "manifest_rows"), 0, "{text}");
    assert_eq!(count(&text, "rows"), 0, "{text}");
    assert!(
        text.contains("\tverdict\tgates\tseconds\trun\t"),
        "the column header must be present even with no rows under it, and it \
         must carry the per-row run date: without it a row carried forward from \
         an earlier run is indistinguishable from one just proven: {text}"
    );
    assert!(said.contains("result written to"), "{said}");
}

fn anchor_faults(root: &Path, manifest: &str) -> Vec<(String, String, usize)> {
    let mut rows: Vec<(String, String, String)> = Vec::new();
    for line in manifest.lines() {
        let trimmed = line.trim();
        if let Some(v) = json_string(trimmed, "\"name\":") {
            rows.push((v, String::new(), String::new()));
        }
        if let Some(v) = json_string(trimmed, "\"file\":") {
            if let Some(last) = rows.last_mut() {
                last.1 = v;
            }
        }
        if let Some(v) = json_string(trimmed, "\"before\":") {
            if let Some(last) = rows.last_mut() {
                last.2 = v;
            }
        }
    }
    let mut faults = Vec::new();
    for (name, file, before) in rows {
        let text = std::fs::read_to_string(root.join(&file)).unwrap_or_default();
        let hits = if before.is_empty() { 0 } else { text.matches(before.as_str()).count() };
        if hits != 1 {
            faults.push((name, file, hits));
        }
    }
    faults
}

#[test]
fn every_anchor_in_the_manifest_matches_its_file_exactly_once() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts").join("mutations.json");
    let manifest = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => panic!("{}: {e}", path.display()),
    };
    assert!(!manifest_rows().is_empty(), "the manifest parser read nothing");
    let faults = anchor_faults(&repo_root(), &manifest);
    assert!(
        faults.is_empty(),
        "{} manifest row(s) have a `before` that does not match its file exactly \
         once. An absent anchor is refused by the driver, so the committed CAUGHT \
         is of a tree that has moved; a repeated one lands on whichever copy comes \
         first in the file. Each (name, file, matches): {:#?}",
        faults.len(),
        faults
    );
}

#[test]
fn the_anchor_check_reports_an_absent_and_a_repeated_anchor_and_passes_a_unique_one() {
    let dir = match tempfile::Builder::new().prefix("lights-anchors-").tempdir() {
        Ok(d) => d,
        Err(e) => panic!("tempdir: {e}"),
    };
    if let Err(e) = std::fs::write(dir.path().join("a.rs"), "one\ntwo\ntwo\nfour \u{2014} \u{1F600}\n") {
        panic!("cannot write the fake source: {e}");
    }
    let manifest = "{\"mutations\": [\n  {\n   \"name\": \"unique\",\n   \"file\": \"a.rs\",\n   \"before\": \"one\\ntwo\"\n  },\n  {\n   \"name\": \"repeated\",\n   \"file\": \"a.rs\",\n   \"before\": \"two\"\n  },\n  {\n   \"name\": \"absent\",\n   \"file\": \"a.rs\",\n   \"before\": \"three\"\n  },\n  {\n   \"name\": \"escaped\",\n   \"file\": \"a.rs\",\n   \"before\": \"four \\u2014 \\ud83d\\ude00\"\n  }\n]}\n";
    let faults = anchor_faults(dir.path(), manifest);
    assert_eq!(
        faults,
        vec![
            ("repeated".to_string(), "a.rs".to_string(), 2),
            ("absent".to_string(), "a.rs".to_string(), 0),
        ]
    );
}
