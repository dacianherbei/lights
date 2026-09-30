#![cfg(feature = "fit")]
use lights::fit::aspect3::{
    Aspect3Request, Direction, SCALE_COLUMNS, analytic_at,
};
use lights::fit::coverage::coverage_budget;
use lights::fit::nterm::{max_abs_error, relative_l2};
use lights::fit::scale::{Scale, ScaleClass};
use lights::fit::target::Target;
use std::path::PathBuf;

const MAGNIFY: &str = "ladder-magnify.tsv";
const MINIFY: &str = "ladder-minify.tsv";

fn artefact(file: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("docs")
        .join("lights")
        .join("p5")
        .join(file)
}

struct Table {
    file: String,
    invocation: String,
    columns: Vec<String>,
    rows: Vec<Vec<String>>,
}

impl Table {
    fn cell(&self, row: &[String], name: &str) -> String {
        let index = match self.columns.iter().position(|c| c == name) {
            Some(i) => i,
            None => panic!("{} has no column named {name}", self.file),
        };
        match row.get(index) {
            Some(v) => v.clone(),
            None => panic!("{}: row is short of column {name}", self.file),
        }
    }

    fn number(&self, row: &[String], name: &str) -> f64 {
        let text = self.cell(row, name);
        match text.parse::<f64>() {
            Ok(v) => v,
            Err(e) => panic!("{}: column {name} holds {text}: {e}", self.file),
        }
    }
}

fn read(file: &str) -> Table {
    let path = artefact(file);
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => panic!(
            "{} is missing: {e}. Aspect 3's published ladder has no artefact \
             without it. Regenerate with the command the file's own \
             #invocation line carries.",
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
        file: file.to_string(),
        invocation,
        columns,
        rows,
    }
}

fn class_named(name: &str) -> ScaleClass {
    match name {
        "t2-straight-edge" => ScaleClass::Analytic(Target::<f64>::t2_straight_edge()),
        "t3-right-angle-corner" => ScaleClass::Analytic(Target::<f64>::t3_right_angle_corner()),
        "t6-glyph-like" => ScaleClass::Analytic(Target::<f64>::t6_glyph_like()),
        "n1-colour-field" => ScaleClass::colour_field(),
        other => panic!(
            "the committed table carries class {other}, which this test cannot \
             rebuild; add it to class_named or the recomputation is silently \
             skipping a class"
        ),
    }
}

fn scale_labelled(label: &str, direction: Direction) -> Scale {
    let ladder = match direction.ladder() {
        Ok(l) => l,
        Err(e) => panic!("{} has no ladder: {e}", direction.name()),
    };
    match ladder.into_iter().find(|s| s.label() == label) {
        Some(s) => s,
        None => panic!(
            "no rung of the {} ladder is labelled {label}",
            direction.name()
        ),
    }
}

struct Recomputed {
    relative_l2: f64,
    max_abs_error: f64,
}

fn recompute_analytic(class_name: &str, scale_label: &str, direction: Direction) -> Recomputed {
    let class = class_named(class_name);
    let scale = scale_labelled(scale_label, direction);
    let request = match Aspect3Request::new(direction) {
        Ok(r) => r,
        Err(e) => panic!("{e}"),
    };
    let mut cells = coverage_budget(request.cell_limit);
    let truth = match class.truth(scale, request.quadrature_depth, &mut cells) {
        Ok(t) => t,
        Err(e) => panic!("{class_name} at {scale_label} has no truth: {e}"),
    };
    let image = match analytic_at(&class, scale) {
        Ok(Some(i)) => i,
        Ok(None) => panic!(
            "{class_name} has no analytic arm, so this cell cannot be recomputed"
        ),
        Err(e) => panic!("{class_name} at {scale_label}: {e}"),
    };
    let rel = match relative_l2(&truth, &image) {
        Ok(v) => v,
        Err(e) => panic!("{e}"),
    };
    let max = match max_abs_error(&truth, &image) {
        Ok(v) => v,
        Err(e) => panic!("{e}"),
    };
    Recomputed {
        relative_l2: rel,
        max_abs_error: max,
    }
}

fn analytic_row(table: &Table, class: &str, scale: &str) -> Vec<String> {
    let found: Vec<&Vec<String>> = table
        .rows
        .iter()
        .filter(|r| {
            table.cell(r, "arm") == "analytic-sdf"
                && table.cell(r, "class") == class
                && table.cell(r, "scale") == scale
                && table.cell(r, "absent") == "no"
        })
        .collect();
    assert_eq!(
        found.len(),
        1,
        "{} must carry exactly one present analytic-sdf row for {class} at \
         {scale}, found {}",
        table.file,
        found.len()
    );
    found[0].clone()
}

#[test]
fn both_committed_ladders_carry_the_columns_this_crate_writes_today() {
    for file in [MAGNIFY, MINIFY] {
        let table = read(file);
        let expected: Vec<String> = SCALE_COLUMNS.iter().map(|c| c.to_string()).collect();
        assert_eq!(
            table.columns, expected,
            "{file} was written by a different shape of SCALE_COLUMNS than this \
             crate emits, so every figure read out of it by position is reading \
             the wrong column"
        );
    }
}

#[test]
fn each_committed_ladder_names_the_invocation_that_produced_it() {
    for file in [MAGNIFY, MINIFY] {
        let table = read(file);
        assert!(
            !table.invocation.is_empty(),
            "{file} carries no #invocation line, so its figures name no command \
             that reproduces them"
        );
        assert!(
            table.invocation.contains("lights-p5"),
            "{file}: {}",
            table.invocation
        );
        assert!(
            table.invocation.contains("--direction both"),
            "{file} is one half of a run that produced both ladders, and the \
             invocation must say so: {}",
            table.invocation
        );
    }
}

#[test]
fn the_committed_analytic_cells_are_what_todays_coverage_function_computes() {
    let table = read(MAGNIFY);
    let classes = [
        "t2-straight-edge",
        "t3-right-angle-corner",
        "t6-glyph-like",
    ];
    let rungs = ["1x", "2x", "4x", "8x"];
    let checked: Vec<(&str, &str)> = classes
        .iter()
        .flat_map(|c| rungs.iter().map(move |s| (*c, *s)))
        .collect();
    let mut seen = 0usize;
    for (class, scale) in checked.iter() {
        let row = analytic_row(&table, class, scale);
        let published_rel = table.number(&row, "relative_l2");
        let published_max = table.number(&row, "max_abs_error");
        let measured = recompute_analytic(class, scale, Direction::Magnify);
        assert!(
            (published_rel - measured.relative_l2).abs() < 1e-6,
            "{class} at {scale}: the committed table says relative_l2 \
             {published_rel:.6} and today's sdf_coverage computes \
             {:.6}. The published column was measured by code that is gone.",
            measured.relative_l2
        );
        assert!(
            (published_max - measured.max_abs_error).abs() < 1e-6,
            "{class} at {scale}: the committed table says max_abs_error \
             {published_max:.6} and today's sdf_coverage computes \
             {:.6}.",
            measured.max_abs_error
        );
        seen += 1;
    }
    assert_eq!(
        seen,
        checked.len(),
        "every named cell must have been recomputed; a test that checks no cell \
         passes by measuring nothing"
    );
}

#[test]
fn the_lower_bound_and_the_real_field_part_company_exactly_where_primitives_meet() {
    let one = Target::<f64>::t2_straight_edge();
    let several = Target::<f64>::t3_right_angle_corner();
    assert_eq!(one.primitive_count(), 1, "t2 is the single-primitive control");
    assert!(
        several.primitive_count() > 1,
        "t3 must carry several primitives for a lower bound to be wrong about"
    );

    let mut disagreements_on_one = 0usize;
    let mut disagreements_on_several = 0usize;
    let mut probed = 0usize;
    for py in 0..one.viewport.1 {
        for px in 0..one.viewport.0 {
            let p = lights::numerics::vector::Vec2::new(px as f64 + 0.5, py as f64 + 0.5);
            if (one.sdf(p) - one.combined(p)).abs() > 1e-12 {
                disagreements_on_one += 1;
            }
            if (several.sdf(p) - several.combined(p)).abs() > 1e-12 {
                disagreements_on_several += 1;
            }
            probed += 1;
        }
    }
    assert!(probed > 0, "the probe must visit at least one pixel");
    assert_eq!(
        disagreements_on_one, 0,
        "with one primitive the lower bound IS the field; {disagreements_on_one} \
         of {probed} centres disagree, so the control is not a control"
    );
    assert!(
        disagreements_on_several > 0,
        "with several primitives the lower bound must depart from the field \
         somewhere, or the defect L-144 records could not have happened"
    );
}

#[test]
fn every_frozen_base_row_has_a_checkpoint_beside_it_and_the_two_agree() {
    let mut visited = 0usize;
    let mut files_named: Vec<PathBuf> = Vec::new();
    for file in [MAGNIFY, MINIFY] {
        let table = read(file);
        let base_rows: Vec<Vec<String>> = table
            .rows
            .iter()
            .filter(|r| {
                table.cell(r, "arm").starts_with("frozen-atoms")
                    && table.cell(r, "at_base") == "yes"
            })
            .cloned()
            .collect();
        assert!(
            !base_rows.is_empty(),
            "{file} carries no frozen base row, so this check would pass by \
             visiting nothing"
        );
        for row in base_rows.iter() {
            let class = table.cell(row, "class");
            let direction = table.cell(row, "direction");
            let profile = table.cell(row, "profile");
            let path = artefact(&format!("frozen-{class}-{direction}-{profile}.fit"));
            let (atoms, viewport) = match lights::fit::atoms_io::read_atoms(&path) {
                Ok(v) => v,
                Err(e) => panic!(
                    "{}: {e}. The frozen arm of aspect 3 is unreproducible without \
                     a refit unless this checkpoint is committed beside the table.",
                    path.display()
                ),
            };
            let terms: usize = match table.cell(row, "terms").parse() {
                Ok(v) => v,
                Err(e) => panic!("terms is not a count: {e}"),
            };
            assert_eq!(
                atoms.len(),
                terms,
                "{} holds {} atoms and its row says the frozen set is {terms}",
                path.display(),
                atoms.len()
            );
            assert_eq!(
                format!("{}x{}", viewport.0, viewport.1),
                table.cell(row, "viewport"),
                "{} was fitted at a different viewport than its row reports",
                path.display()
            );
            files_named.push(path);
            visited += 1;
        }
    }
    assert!(
        visited >= 8,
        "only {visited} frozen base rows were visited; the published run has four \
         classes under two profiles in each direction, so anything under eight \
         means the table or this filter has moved"
    );

    let mut on_disk: Vec<PathBuf> = Vec::new();
    let dir = artefact("");
    let entries = match std::fs::read_dir(&dir) {
        Ok(e) => e,
        Err(e) => panic!("{}: {e}", dir.display()),
    };
    for entry in entries {
        let path = match entry {
            Ok(e) => e.path(),
            Err(e) => panic!("{e}"),
        };
        if path.extension().map(|e| e == "fit").unwrap_or(false) {
            on_disk.push(path);
        }
    }
    files_named.sort();
    on_disk.sort();
    assert_eq!(
        on_disk, files_named,
        "every checkpoint on disk must be named by a frozen base row and every \
         such row must have one; a file nobody's row points at is a checkpoint of \
         a run that is no longer published"
    );
}
