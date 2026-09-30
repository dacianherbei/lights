#![cfg(feature = "fit")]
use std::path::PathBuf;

const RUN_8: &str = "run-8";
const RUN_64: &str = "run-64";

const COLUMNS: [&str; 9] = [
    "content",
    "viewport_width",
    "viewport_height",
    "control",
    "shared_ceiling",
    "requested",
    "zoom",
    "columns",
    "png",
];

const CLASSES: [&str; 8] = [
    "t2-straight-edge",
    "t3-right-angle-corner",
    "t5-thin-stroke",
    "t6-glyph-like",
    "n1-colour-field",
    "n2-soft-shadow",
    "n3-band-limited-noise",
    "n4-broadband-patch",
];

const PUBLISHED_CEILINGS_AT_64: [(&str, u32); 8] = [
    ("t2-straight-edge", 11),
    ("t3-right-angle-corner", 15),
    ("t5-thin-stroke", 13),
    ("t6-glyph-like", 10),
    ("n1-colour-field", 61),
    ("n2-soft-shadow", 50),
    ("n3-band-limited-noise", 64),
    ("n4-broadband-patch", 31),
];

fn run_dir(run: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("docs")
        .join("lights")
        .join("p4")
        .join(run)
}

struct Table {
    run: String,
    invocation: String,
    columns: Vec<String>,
    rows: Vec<Vec<String>>,
}

impl Table {
    fn cell(&self, row: &[String], name: &str) -> String {
        let index = match self.columns.iter().position(|c| c == name) {
            Some(i) => i,
            None => panic!("{}/report.tsv has no column named {name}", self.run),
        };
        match row.get(index) {
            Some(v) => v.clone(),
            None => panic!("{}: row is short of column {name}", self.run),
        }
    }

    fn number(&self, row: &[String], name: &str) -> u32 {
        let text = self.cell(row, name);
        match text.parse::<u32>() {
            Ok(v) => v,
            Err(e) => panic!("{}: column {name} holds {text}: {e}", self.run),
        }
    }
}

fn read(run: &str) -> Table {
    let path = run_dir(run).join("report.tsv");
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => panic!(
            "{} is missing: {e}. Aspect 2 published its rel-L2 and byte figures \
             from harnesses that were never committed; this table is the only \
             thing on disk that any of its claims can be checked against. \
             Regenerate with the command the file's own #invocation carries.",
            path.display()
        ),
    };
    let mut invocation = String::new();
    let mut columns: Vec<String> = Vec::new();
    let mut rows: Vec<Vec<String>> = Vec::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("#invocation\t") {
            invocation = rest.to_string();
            continue;
        }
        if line.starts_with('#') {
            continue;
        }
        let cells: Vec<String> = line.split('\t').map(|c| c.to_string()).collect();
        if columns.is_empty() {
            columns = cells;
        } else {
            rows.push(cells);
        }
    }
    assert!(!rows.is_empty(), "{} carries no data rows", path.display());
    Table {
        run: run.to_string(),
        invocation,
        columns,
        rows,
    }
}

fn row_for(table: &Table, class: &str, zoom: u32) -> Vec<String> {
    let found: Vec<&Vec<String>> = table
        .rows
        .iter()
        .filter(|r| table.cell(r, "content") == class && table.number(r, "zoom") == zoom)
        .collect();
    assert_eq!(
        found.len(),
        1,
        "{}: {class} at zoom {zoom} must name exactly one row, found {}",
        table.run,
        found.len()
    );
    found[0].clone()
}

fn a_fresh_tiny_run() -> (tempfile::TempDir, Table) {
    let held = match tempfile::Builder::new().prefix("lights-p4-columns-").tempdir() {
        Ok(d) => d,
        Err(e) => panic!("tempdir: {e}"),
    };
    let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_lights-p4"));
    command.args([
        "--classes",
        "1",
        "--terms",
        "shared",
        "--max-atoms",
        "1",
        "--out-dir",
    ]);
    command.arg(held.path());
    let out = match command.output() {
        Ok(o) => o,
        Err(e) => panic!("running lights-p4: {e}"),
    };
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(out.status.code().unwrap_or(-1), 0, "{text}");
    let path = held.path().join("report.tsv");
    let body = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => panic!("{}: {e}", path.display()),
    };
    let mut invocation = String::new();
    let mut columns: Vec<String> = Vec::new();
    let mut rows: Vec<Vec<String>> = Vec::new();
    for line in body.lines() {
        if let Some(rest) = line.strip_prefix("#invocation\t") {
            invocation = rest.to_string();
            continue;
        }
        if line.starts_with('#') {
            continue;
        }
        let cells: Vec<String> = line.split('\t').map(|c| c.to_string()).collect();
        if columns.is_empty() {
            columns = cells;
        } else {
            rows.push(cells);
        }
    }
    assert!(!rows.is_empty(), "a fresh run wrote no data rows");
    let fresh = Table {
        run: "fresh".to_string(),
        invocation,
        columns,
        rows,
    };
    (held, fresh)
}

#[test]
fn both_committed_p4_runs_carry_the_columns_todays_binary_writes() {
    let (_held, fresh) = a_fresh_tiny_run();
    let expected: Vec<String> = COLUMNS.iter().map(|c| c.to_string()).collect();
    assert_eq!(
        fresh.columns, expected,
        "today's lights-p4 writes a different column set than this test names; \
         the test's list is a copy because the binary's REPORT_COLUMNS is \
         private to a binary and no integration test can import it, so this \
         assertion is what keeps the copy honest"
    );
    for run in [RUN_8, RUN_64] {
        let table = read(run);
        assert_eq!(
            table.columns, fresh.columns,
            "{run}/report.tsv was written by a different shape of lights-p4's \
             REPORT_COLUMNS than the binary emits today, so every figure read \
             out of it by position is reading the wrong column"
        );
    }
}

#[test]
fn todays_binary_writes_a_strip_name_and_not_a_path_into_its_own_machine() {
    let (_held, fresh) = a_fresh_tiny_run();
    let mut seen = 0usize;
    for row in fresh.rows.iter() {
        let png = fresh.cell(row, "png");
        assert!(
            !png.contains('/'),
            "lights-p4 wrote {png:?} into the png column. A published table that \
             carries an absolute path names one machine, and the two committed \
             runs are read by anyone."
        );
        assert!(!png.is_empty(), "the png column must name the strip");
        seen += 1;
    }
    assert!(seen > 0, "a run that writes no row cannot fail this check");
}

#[test]
fn each_committed_p4_run_names_the_invocation_and_the_atom_budget_that_produced_it() {
    for (run, budget) in [(RUN_8, "--max-atoms 8"), (RUN_64, "--max-atoms 64")] {
        let table = read(run);
        assert!(
            table.invocation.contains("lights-p4"),
            "{run}: {}",
            table.invocation
        );
        assert!(
            table.invocation.contains(budget),
            "{run} is the {budget} run and its own first line must say so: {}",
            table.invocation
        );
        assert!(
            table.invocation.contains("--classes all"),
            "{run} must cover every content class: {}",
            table.invocation
        );
    }
}

#[test]
fn the_published_shared_ceilings_are_what_the_binary_records_at_a_64_atom_budget() {
    let table = read(RUN_64);
    let mut checked = 0usize;
    for (class, published) in PUBLISHED_CEILINGS_AT_64.iter() {
        let row = row_for(&table, class, 1);
        let measured = table.number(&row, "shared_ceiling");
        assert_eq!(
            measured, *published,
            "{class}: the research record publishes a shared ceiling of \
             {published} at a 64-atom budget and the committed table says \
             {measured}"
        );
        assert_eq!(
            table.number(&row, "requested"),
            measured,
            "{class}: a shared-terms run renders AT the ceiling, so requested \
             and shared_ceiling must agree or the strip is not the figure the \
             row describes"
        );
        checked += 1;
    }
    assert_eq!(
        checked,
        PUBLISHED_CEILINGS_AT_64.len(),
        "every published ceiling must have been checked"
    );

    let eight = read(RUN_8);
    for class in CLASSES.iter() {
        let row = row_for(&eight, class, 1);
        assert_eq!(
            eight.number(&row, "shared_ceiling"),
            8,
            "{class}: at a budget of 8 every class is capped by the budget, \
             which is what makes the 64-atom run the one that measures reach"
        );
    }
}

#[test]
fn the_analytic_control_is_absent_on_exactly_the_four_continuous_classes() {
    for run in [RUN_8, RUN_64] {
        let table = read(run);
        let mut absent = 0usize;
        let mut present = 0usize;
        for class in CLASSES.iter() {
            let row = row_for(&table, class, 1);
            let control = table.cell(&row, "control");
            if class.starts_with('n') {
                assert_eq!(
                    control, "ABSENT",
                    "{run} {class}: no signed distance describes this class, so \
                     the control must be declared absent rather than replaced by \
                     an invented arm"
                );
                absent += 1;
            } else {
                assert_eq!(control, "analytic-exact", "{run} {class}");
                present += 1;
            }
        }
        assert_eq!((absent, present), (4, 4), "{run}");
    }
}

#[test]
fn the_glyph_is_the_only_class_whose_viewport_forces_a_shallower_transform() {
    for run in [RUN_8, RUN_64] {
        let table = read(run);
        let mut small = Vec::new();
        for class in CLASSES.iter() {
            let row = row_for(&table, class, 1);
            let w = table.number(&row, "viewport_width");
            let h = table.number(&row, "viewport_height");
            if (w, h) == (32, 32) {
                small.push(*class);
            } else {
                assert_eq!(
                    (w, h),
                    (64, 64),
                    "{run} {class}: aspect 2 ran every class at 64x64 except the \
                     glyph, and the record's level-count finding rests on that"
                );
            }
        }
        assert_eq!(
            small,
            vec!["t6-glyph-like"],
            "{run}: the record says t6-glyph-like ran at three transform levels \
             where every other class ran at four, because max_levels(32x32) is 3 \
             and max_levels(64x64) is 4. That claim is about this column."
        );
    }
}

#[test]
fn every_strip_the_table_names_is_on_disk_and_every_strip_on_disk_is_named() {
    for run in [RUN_8, RUN_64] {
        let table = read(run);
        let dir = run_dir(run);
        let mut named: Vec<String> = Vec::new();
        for row in table.rows.iter() {
            let png = table.cell(row, "png");
            assert!(
                !png.contains('/'),
                "{run}: the png column must be a name relative to the run \
                 directory, and it holds {png:?}. An absolute path in a \
                 published artefact names this machine and nobody else's."
            );
            assert!(
                dir.join(&png).exists(),
                "{run}: the table names {png} and it is not on disk"
            );
            named.push(png);
        }
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            Err(e) => panic!("{}: {e}", dir.display()),
        };
        let mut on_disk: Vec<String> = Vec::new();
        for entry in entries {
            let path = match entry {
                Ok(e) => e.path(),
                Err(e) => panic!("{e}"),
            };
            if path.extension().map(|e| e == "png").unwrap_or(false) {
                match path.file_name() {
                    Some(n) => on_disk.push(n.to_string_lossy().to_string()),
                    None => panic!("a png with no file name"),
                }
            }
        }
        named.sort();
        named.dedup();
        on_disk.sort();
        assert!(!named.is_empty(), "{run} names no strip at all");
        assert_eq!(
            on_disk, named,
            "{run}: the strips on disk and the strips the table names must be \
             the same set, or the published figure and the published picture \
             came from different runs"
        );
    }
}
