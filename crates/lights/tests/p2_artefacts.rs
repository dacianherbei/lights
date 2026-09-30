#![cfg(feature = "fit")]
use lights::fit::atom2d::{Atom2d, camera};
use lights::fit::harness::tile_cost;
use lights::fit::report::ROW_COLUMNS;
use lights::kernel::KernelFamily;
use lights::settings::SharedSettings;
use lights::numerics::vector::{Rgb, Vec2};
use std::path::PathBuf;

const RUN_48: &str = "run-48";
const RUN_8_TILED: &str = "run-8-tiled";

fn artefact(run: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("docs")
        .join("lights")
        .join("p2")
        .join(run)
        .join("report.tsv")
}

struct Table {
    invocation: String,
    columns: Vec<String>,
    rows: Vec<Vec<String>>,
}

impl Table {
    fn cell(&self, row: &[String], name: &str) -> String {
        let index = match self.columns.iter().position(|c| c == name) {
            Some(i) => i,
            None => panic!("no column named {name} in the committed table"),
        };
        match row.get(index) {
            Some(v) => v.clone(),
            None => panic!("row is short of column {name}"),
        }
    }

    fn number(&self, row: &[String], name: &str) -> f64 {
        let text = self.cell(row, name);
        match text.parse::<f64>() {
            Ok(v) => v,
            Err(e) => panic!("column {name} holds {text}, which is not a number: {e}"),
        }
    }
}

fn read(run: &str) -> Table {
    let path = artefact(run);
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => panic!(
            "{} is missing: {e}. P2's published figures are prose-only without \
             it, which is the condition BURNDOWN L-48 was raised against. \
             Regenerate with the invocation the file's own first line carries.",
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
    assert!(
        !invocation.is_empty(),
        "{} carries no #invocation line, so the figures in it name no command \
         that reproduces them",
        path.display()
    );
    assert!(!rows.is_empty(), "{} carries no data rows", path.display());
    Table { invocation, columns, rows }
}

fn atom_cells(table: &Table) -> Vec<Vec<String>> {
    table
        .rows
        .iter()
        .filter(|r| table.cell(r, "arm") == "atoms" && table.cell(r, "field") == "natural")
        .cloned()
        .collect()
}

fn tiled_cells(table: &Table) -> Vec<Vec<String>> {
    table
        .rows
        .iter()
        .filter(|r| table.cell(r, "field") == "tiled")
        .cloned()
        .collect()
}

fn pick(table: &Table, rows: &[Vec<String>], target: &str, blend: &str, profile: &str) -> Vec<String> {
    let found: Vec<&Vec<String>> = rows
        .iter()
        .filter(|r| {
            table.cell(r, "target") == target
                && table.cell(r, "blend") == blend
                && table.cell(r, "profile") == profile
        })
        .collect();
    assert_eq!(
        found.len(),
        1,
        "{target} {blend} {profile} must name exactly one cell, found {}",
        found.len()
    );
    found[0].clone()
}

#[test]
fn both_committed_tables_carry_the_columns_this_crate_writes_today() {
    for run in [RUN_48, RUN_8_TILED] {
        let table = read(run);
        let expected: Vec<String> = ROW_COLUMNS.iter().map(|c| c.to_string()).collect();
        assert_eq!(
            table.columns, expected,
            "{run}/report.tsv was written by a different shape of ROW_COLUMNS \
             than this crate emits, so every figure read out of it by position \
             is reading the wrong column"
        );
    }
}

#[test]
fn each_committed_table_names_the_invocation_that_produced_it() {
    let forty_eight = read(RUN_48);
    assert!(
        forty_eight.invocation.contains("lights-p2"),
        "{}",
        forty_eight.invocation
    );
    assert!(
        forty_eight.invocation.contains("--max-atoms 48"),
        "the headline run is --max-atoms 48: {}",
        forty_eight.invocation
    );
    assert!(
        !forty_eight.invocation.contains("--tiled-field"),
        "the headline run does NOT tile the field; the tile-binning figures come \
         from a second, separate invocation: {}",
        forty_eight.invocation
    );

    let tiled = read(RUN_8_TILED);
    assert!(
        tiled.invocation.contains("--max-atoms 8"),
        "the tile-binning run is --max-atoms 8, and 8 atoms x 40 copies is what \
         makes its scene population 320: {}",
        tiled.invocation
    );
    assert!(
        tiled.invocation.contains("--tiled-field"),
        "{}",
        tiled.invocation
    );
}

#[test]
fn the_published_binning_ratio_is_the_cross_cell_maximum_and_it_is_123_4() {
    let table = read(RUN_8_TILED);
    let rows = tiled_cells(&table);
    assert_eq!(rows.len(), 24, "6 targets x 2 blends x 2 profiles");
    let mut best = rows[0].clone();
    for row in rows.iter() {
        if table.number(row, "box_over_ellipse_area") > table.number(&best, "box_over_ellipse_area") {
            best = row.clone();
        }
    }
    assert_eq!(table.cell(&best, "target"), "t5-thin-stroke");
    assert_eq!(table.cell(&best, "blend"), "gaussian-normalised");
    assert_eq!(table.cell(&best, "profile"), "contract");
    let ratio = table.number(&best, "box_over_ellipse_area");
    assert!(
        (ratio - 123.406101).abs() < 5e-6,
        "P2's headline 123x is this cross-cell maximum and it reproduced at \
         123.406101; the committed table says {ratio}"
    );
}

#[test]
fn the_average_tile_no_longer_holds_most_of_the_scene_and_the_published_232_named_a_binner_that_is_gone() {
    let table = read(RUN_8_TILED);
    let rows = tiled_cells(&table);
    let edge = pick(&table, &rows, "t2-straight-edge", "gaussian", "prefiltered");
    assert_eq!(
        table.cell(&edge, "atoms"),
        "320",
        "8 fitted atoms x 40 copies is the scene P2 quoted"
    );
    let mean = table.number(&edge, "mean_tile_list");
    assert!(
        (mean - 25.747070).abs() < 5e-6,
        "the committed table says {mean}; the figure P2 published for this cell \
         is 232.0234, which this crate reproduces ONLY with the per-tile \
         Mahalanobis rejection in src/method/splat/bin.rs removed. That \
         rejection landed in 6b1b9ab8 on 2026-09-14, after P2 ran, and it \
         prunes this cell 9.01x"
    );
    assert!(
        mean * 9.0 < 320.0,
        "the claim P2 drew from 232 of 320 was that the average tile holds \
         essentially the whole scene; at {mean} of 320 it does not"
    );
    let ratio = table.number(&edge, "box_over_ellipse_area");
    assert!(
        ratio > 80.0,
        "the AREA ratio is unchanged at {ratio} and that is the point: the atoms \
         are the same atoms, the binner is not the same binner"
    );
}

#[test]
fn the_binner_rejects_tiles_the_axis_aligned_box_hands_it() {
    let field = 128u32;
    let tile = 16u32;
    let spec = camera(field, field);
    let settings = SharedSettings::<f64>::contract();
    let needle = Atom2d::from_sigma(
        Vec2::new(64.0, 64.0),
        Vec2::new(26.0, 0.35),
        std::f64::consts::FRAC_PI_4,
        1.0,
        Rgb::new(1.0, 1.0, 1.0),
    );
    let cost = match tile_cost(&[needle], &spec, KernelFamily::Gaussian, &settings, tile) {
        Ok(c) => c,
        Err(e) => panic!("binning one needle: {e}"),
    };
    let across = field / tile;
    let all_tiles = u64::from(across * across);
    assert!(
        cost.box_over_ellipse_area > 8.0,
        "a needle at 45 degrees is the shape whose axis-aligned box is much \
         larger than its ellipse; this one measures {}",
        cost.box_over_ellipse_area
    );
    assert!(
        cost.tile_atom_pairs < all_tiles,
        "the needle's axis-aligned box spans the whole {field}x{field} field, so \
         a binner that trusts the box alone pays {all_tiles} pairs; this one \
         paid {}. Every P2 tile figure republished after 2026-09-14 rests on \
         that rejection existing",
        cost.tile_atom_pairs
    );
    assert!(
        cost.tile_atom_pairs > 0,
        "an atom at the centre of the field must reach at least one tile"
    );
}

#[test]
fn twenty_two_of_the_twenty_four_cells_stopped_before_the_budget_did() {
    let table = read(RUN_48);
    let rows = atom_cells(&table);
    assert_eq!(rows.len(), 24, "6 targets x 2 blends x 2 profiles");
    let stopped: Vec<&Vec<String>> = rows
        .iter()
        .filter(|r| table.cell(r, "improvement_exhausted") == "yes")
        .collect();
    assert_eq!(
        stopped.len(),
        22,
        "P2 published 'in 20 of 24 cells the fitter stopped, not the budget'. \
         The committed table says {} of 24, and the two that did reach \
         --max-atoms 48 are both the right-angle corner under normalised \
         blending",
        stopped.len()
    );
    let at_budget: Vec<&Vec<String>> = rows
        .iter()
        .filter(|r| table.cell(r, "improvement_exhausted") != "yes")
        .collect();
    for row in at_budget.iter() {
        assert_eq!(table.cell(row, "target"), "t3-right-angle-corner");
        assert_eq!(table.cell(row, "blend"), "gaussian-normalised");
        assert_eq!(table.cell(row, "atoms_available"), "48");
    }
}

#[test]
fn the_glyph_atom_counts_the_write_up_tabulates_are_in_the_committed_table() {
    let table = read(RUN_48);
    let rows = atom_cells(&table);
    let expected = [
        ("gaussian", "contract", "15"),
        ("gaussian", "prefiltered", "22"),
        ("gaussian-normalised", "contract", "12"),
        ("gaussian-normalised", "prefiltered", "12"),
    ];
    for (blend, profile, count) in expected {
        let row = pick(&table, &rows, "t6-glyph-like", blend, profile);
        assert_eq!(
            table.cell(&row, "atoms_available"),
            count,
            "P2's glyph table says {blend} {profile} cost {count} atoms"
        );
        assert_eq!(
            table.cell(&row, "meets_criterion"),
            "no",
            "every glyph cell is a cost of FAILING; none of them met the criterion"
        );
    }
}
