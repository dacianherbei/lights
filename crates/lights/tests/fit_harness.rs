#![cfg(feature = "fit")]
use lights::dispatch::Collect;
use lights::error::LightsError;
use lights::fit::atom2d::camera;
use lights::fit::edge::Criterion;
use lights::fit::greedy::render_atoms;
use lights::fit::edge::SDF_ARM_TRUE_TRANSITION;
use lights::fit::harness::{FIELD_COPIES, ProbeRequest, ProfileChoice, run};
use lights::fit::report::{
    ATOM_ARM, NATURAL, ROW_COLUMNS, TILED, header_lines, never_met, write_report,
};
use lights::fit::target::Target;
use lights::kernel::KernelFamily;
use lights::settings::SharedSettings;
use std::path::PathBuf;
use std::process::Command;

fn small(tiled: bool) -> ProbeRequest {
    let mut request = ProbeRequest::all();
    request.targets = vec![0, 1];
    request.blends = vec![KernelFamily::GaussianNormalised];
    request.profiles = vec![ProfileChoice::Prefiltered];
    request.max_atoms = 2;
    request.repeats = 1;
    request.sensitivity = false;
    request.tiled_field = tiled;
    request
}

fn scratch(name: &str) -> tempfile::TempDir {
    match tempfile::Builder::new().prefix(&format!("lights-p2-{name}-")).tempdir() {
        Ok(d) => d,
        Err(e) => panic!("tempdir: {e}"),
    }
}

fn binary() -> Command {
    Command::new(env!("CARGO_BIN_EXE_lights-p2"))
}

fn code(mut command: Command) -> (i32, String) {
    let output = match command.output() {
        Ok(o) => o,
        Err(e) => panic!("running lights-p2: {e}"),
    };
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    (output.status.code().unwrap_or(-1), text)
}

fn read(path: &std::path::Path) -> String {
    match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => panic!("cannot read {}: {e}", path.display()),
    }
}

#[test]
fn the_report_has_a_row_for_every_cell_with_none_silently_missing() {
    for tiled in [false, true] {
        let mut request = small(tiled);
        request.targets = vec![0];
        request.blends = KernelFamily::FIT_SWEEP.to_vec();
        request.profiles = ProfileChoice::ALL.to_vec();
        let report = match run(&request) {
            Ok(r) => r,
            Err(e) => panic!("the harness failed with tiled_field {tiled}: {e}"),
        };
        assert_eq!(
            report.rows.len(),
            request.expected_rows(),
            "the matrix has {} target-blend-profile cells and {} analytic rows \
             per target, so with tiled_field {tiled} it must produce {} rows \
             and it produced {}. A missing cell is how a matrix comes to LOOK \
             like coverage without being it, so the count is asserted against \
             the product of the axes rather than against a number typed here",
            request.cells(),
            2,
            request.expected_rows(),
            report.rows.len()
        );
        let atom_rows = report
            .rows
            .iter()
            .filter(|r| r.arm == ATOM_ARM && r.field == NATURAL)
            .count();
        assert_eq!(
            atom_rows,
            request.cells(),
            "one natural atom row per cell, got {atom_rows}"
        );
        let tiled_rows = report.rows.iter().filter(|r| r.field == TILED).count();
        assert_eq!(
            tiled_rows,
            if tiled { request.cells() } else { 0 },
            "tiled_field {tiled} produced {tiled_rows} tiled rows"
        );
        for target in request.targets.iter() {
            let name = Target::<f64>::all()[*target].name;
            let analytic = report
                .rows
                .iter()
                .filter(|r| r.target == name && r.arm != ATOM_ARM)
                .count();
            assert_eq!(
                analytic, 2,
                "{name} carries {analytic} analytic rows. The control arm has \
                 NO blend rule and NO prefilter, so crossing it with those axes \
                 would print the same numbers four times and invite the reader \
                 to believe they were measured four times"
            );
        }
        for row in never_met(&report.rows) {
            assert_eq!(
                row.field, NATURAL,
                "{} {} is a {} row and it reached the never-met list. A tiled \
                 row measures COST and nothing else; counting it as a quality \
                 failure would make the exit code a verdict on an axis that \
                 was never measured",
                row.target, row.blend, row.field
            );
        }
        for row in report.rows.iter() {
            if row.arm != ATOM_ARM || row.field != NATURAL {
                continue;
            }
            let cells = report
                .sweep
                .iter()
                .filter(|s| {
                    s.target == row.target
                        && s.blend == row.blend
                        && s.profile == row.profile
                })
                .count();
            assert_eq!(
                cells, row.atoms_available,
                "{} {} placed {} atoms but the sweep carries {cells} rows. The \
                 curve on BOTH sides of the first pass is the point: a single \
                 first-pass count would be reporting the search",
                row.target, row.blend, row.atoms_available
            );
        }
    }
}

#[test]
fn the_harness_never_requests_the_tolerance_collect_on_the_normalised_arm() {
    let source = read(&PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/fit/harness.rs"));
    assert!(
        !source.contains(concat!("Pixels", "AndTolerance")),
        "src/fit/harness.rs names the tolerance collect. Probe::tolerance \
         computes ONE minus the cell coverage as a transmittance, and a \
         normalised accumulator has no transmittance at all, so the image it \
         produces is plausible-looking and meaningless"
    );
    let atoms = [lights::fit::atom2d::Atom2d::from_sigma(
        lights::numerics::vector::Vec2::new(8.0, 8.0),
        lights::numerics::vector::Vec2::new(1.0, 1.0),
        0.0,
        1.0,
        lights::numerics::vector::Rgb::splat(1.0),
    )];
    let settings = SharedSettings::<f64>::prefiltered();
    let refused = render_atoms(
        &atoms,
        &camera(16, 16),
        KernelFamily::GaussianNormalised,
        &settings,
        16,
        Collect::PixelsAndTolerance,
    );
    match refused {
        Err(LightsError::ContractViolation(_)) => (),
        other => panic!(
            "the one function every arm renders through must refuse that \
             combination at the source rather than leaving the refusal to a \
             call site nobody wrote, got {:?}",
            other.map(|o| o.image.width)
        ),
    }
    let report = match run(&small(false)) {
        Ok(r) => r,
        Err(e) => panic!("the normalised arm did not survive a harness run: {e}"),
    };
    assert!(
        !report.rows.is_empty(),
        "the harness produced no rows, so it never exercised the path"
    );
}

#[test]
fn the_header_carries_every_constant_that_moves_the_numbers() {
    let dir = scratch("header");
    let report = match run(&small(false)) {
        Ok(r) => r,
        Err(e) => panic!("{e}"),
    };
    match write_report(dir.path(), &report) {
        Ok(written) => assert_eq!(
            written.len(),
            5,
            "write_report must name every table it wrote, because that list is \
             what gets stamped with the invocation"
        ),
        Err(e) => panic!("writing the report: {e}"),
    }
    let lines = header_lines(&report.header).join("\n");
    for needed in [
        "tile\t",
        "threads\t",
        "PIXEL_BOX_VARIANCE\t",
        "alpha_cutoff\t",
        "Z0\t",
        "DZ\t",
        "quadrature_depth\t",
        "undecided_area\t",
        "precision\t",
        "criterion.max_transition_px\t",
        "criterion.min_contrast_frac\t",
        "viewport.t2-straight-edge\t",
        "control_arm_valid\t",
    ] {
        assert!(
            lines.contains(needed),
            "the header is missing `{}`. Every one of these changes the numbers \
             in the table, and a table without them is unreproducible:\n{lines}",
            needed.trim_end()
        );
    }
    for name in ["report.tsv", "sweep.tsv", "sensitivity.tsv", "probes.tsv", "bins.tsv"] {
        let text = read(&dir.path().join(name));
        assert!(
            text.starts_with("#hardware\t"),
            "{name} does not carry the header block, so a file re-analysed \
             later would not know what produced it"
        );
        assert!(
            text.contains("#PIXEL_BOX_VARIANCE\t"),
            "{name} carries rows without the constants that set them"
        );
    }
    let table = read(&dir.path().join("report.tsv"));
    let head = match table.lines().find(|l| !l.starts_with('#')) {
        Some(l) => l,
        None => panic!("report.tsv has no column line"),
    };
    assert_eq!(
        head.split('\t').count(),
        ROW_COLUMNS.len(),
        "report.tsv names {} columns against {} declared",
        head.split('\t').count(),
        ROW_COLUMNS.len()
    );
    for row in table.lines().filter(|l| !l.starts_with('#')).skip(1) {
        assert_eq!(
            row.split('\t').count(),
            ROW_COLUMNS.len(),
            "a data row carries {} fields against {} columns: {row}",
            row.split('\t').count(),
            ROW_COLUMNS.len()
        );
    }
}

#[test]
fn the_shifted_grid_column_is_measured_rather_than_copied() {
    let mut request = small(false);
    request.targets = vec![1];
    request.max_atoms = 3;
    let report = match run(&request) {
        Ok(r) => r,
        Err(e) => panic!("{e}"),
    };
    let mut compared = 0usize;
    let mut differing = 0usize;
    for row in report.rows.iter() {
        if row.arm != ATOM_ARM || !row.transition_10_90_px.is_finite() {
            continue;
        }
        compared += 1;
        assert!(
            row.shifted_grid_transition_10_90_px.is_finite(),
            "{} {} reports a fitted-grid width but no shifted-grid width. \
             Without the shifted re-measure the unprefiltered column would \
             flatter the atoms: a fitter optimising a point-sampled residual \
             places atoms that cancel aliasing AT THE SAMPLED POSITIONS",
            row.target,
            row.blend
        );
        let gap = (row.transition_10_90_px - row.shifted_grid_transition_10_90_px).abs();
        assert!(
            (gap - row.overfit_gap_px).abs() < 1.0e-12,
            "{} {}: the overfit gap column says {} where the two widths differ \
             by {gap}",
            row.target,
            row.blend,
            row.overfit_gap_px
        );
        if gap > 1.0e-9 {
            differing += 1;
        }
    }
    assert!(compared > 0, "no cell produced a width to compare");
    assert!(
        differing > 0,
        "{compared} cells were re-measured through a half-pixel-shifted camera \
         and not one width moved. That would mean the shifted camera renders \
         the same samples as the fitted one, i.e. the whole overfit column is a \
         copy of the column beside it"
    );
}

#[test]
fn the_p2_binary_refuses_to_write_to_the_repository_root() {
    let (status, text) = code(binary());
    assert_eq!(
        status, 2,
        "--out-dir must be REQUIRED so the binary cannot write to the \
         repository root by accident: {text}"
    );
    assert!(text.contains("--out-dir is REQUIRED"), "{text}");
}

#[test]
fn the_binary_exits_three_when_the_control_arm_fails_its_own_criterion() {
    let dir = scratch("invalid");
    let mut command = binary();
    command.args([
        "--targets", "2", "--blends", "normalised", "--profiles", "prefiltered",
        "--max-atoms", "1", "--no-sensitivity", "--max-width", "0.01", "--out-dir",
    ]);
    command.arg(dir.path());
    let (status, text) = code(command);
    assert_eq!(
        status, 3,
        "a criterion the exact-coverage control arm cannot meet is a verdict on \
         the THRESHOLD, not on the atoms: {text}"
    );
    assert!(text.contains("CONTROL ARM FAILED"), "{text}");
    assert!(
        !text.contains("first@"),
        "the atom table was printed anyway. The failure mode this exit code \
         exists for is an invalid criterion silently producing a confident \
         table of atom counts:\n{text}"
    );
    let table = read(&dir.path().join("report.tsv"));
    assert!(
        table.contains("#control_arm_valid\tno"),
        "report.tsv must record that the control arm failed, so the rows it \
         does carry cannot be read as a clean run"
    );
}

#[test]
fn the_binary_exits_one_when_a_target_never_meets_its_criterion() {
    let dir = scratch("never");
    let mut command = binary();
    command.args([
        "--targets", "2", "--blends", "normalised", "--profiles", "contract",
        "--max-atoms", "1", "--no-sensitivity", "--out-dir",
    ]);
    command.arg(dir.path());
    let (status, text) = code(command);
    assert_eq!(
        status, 1,
        "A GATE THAT CANNOT FAIL IS NOT A GATE. One atom cannot resolve a \
         straight edge to under a pixel, and that is a RESULT to state rather \
         than a budget to extend until something passes: {text}"
    );
    assert!(text.contains("NEVER MET THE CRITERION"), "{text}");
}

#[test]
fn the_binary_exits_zero_and_writes_every_table_and_sheet() {
    let dir = scratch("clean");
    let mut command = binary();
    command.args([
        "--targets", "1", "--blends", "normalised", "--profiles", "prefiltered",
        "--max-atoms", "2", "--no-sensitivity", "--out-dir",
    ]);
    command.arg(dir.path());
    let (status, text) = code(command);
    assert_eq!(
        status, 0,
        "the flat fill is reproduced exactly by two normalised atoms, which is \
         the partition of unity reaching a rendered pixel: {text}"
    );
    assert!(text.contains("EVERY CELL MET ITS CRITERION"), "{text}");
    for name in ["report.tsv", "sweep.tsv", "sensitivity.tsv", "probes.tsv", "bins.tsv"] {
        assert!(dir.path().join(name).is_file(), "{name} was not written");
    }
    for name in [
        "t1-flat-fill-truth.png",
        "t1-flat-fill-analytic-exact.png",
        "t1-flat-fill-analytic-sdf.png",
        "t1-flat-fill-gaussian-normalised-prefiltered-atoms.png",
        "t1-flat-fill-gaussian-normalised-prefiltered-difference.png",
    ] {
        assert!(
            dir.path().join(name).is_file(),
            "{name} was not written. The difference image is normalised and \
             MUST ship beside the truth and the render, or a reader looking at \
             it alone will read a scale of 1e-16 as a defect"
        );
    }
}

#[test]
fn the_criterion_the_report_prints_is_the_criterion_it_judged_by() {
    let mut request = small(false);
    request.criterion = Criterion {
        max_transition: 0.5,
        ..Criterion::registered()
    };
    let report = match run(&request) {
        Ok(r) => r,
        Err(e) => panic!("{e}"),
    };
    assert!(
        (report.header.criterion.max_transition - 0.5).abs() < 1.0e-12,
        "the header printed a criterion of {} while the run judged by 0.5. A \
         header that names constants the run did not use is worse than no \
         header",
        report.header.criterion.max_transition
    );
    assert!(
        !report.header.control_arm_valid,
        "at a 0.5 px threshold the exact-coverage arm cannot pass on the 30 \
         degree edge, whose own box-filter floor is about 0.78 px, so \
         control_arm_valid must be false and this test would be vacuous if it \
         were not"
    );
}

#[test]
fn the_tile_cost_columns_are_read_from_the_rasteriser_and_not_invented() {
    let mut request = small(true);
    request.targets = vec![4];
    request.blends = vec![KernelFamily::Gaussian];
    request.profiles = vec![ProfileChoice::Contract];
    request.max_atoms = 4;
    let report = match run(&request) {
        Ok(r) => r,
        Err(e) => panic!("{e}"),
    };
    let mut checked = 0usize;
    for row in report.rows.iter() {
        if row.arm != ATOM_ARM || row.atoms == 0 {
            continue;
        }
        checked += 1;
        assert!(
            row.tile_atom_pairs > 0 && row.kernel_evaluations > 0,
            "{} {} {} placed {} atoms and reports {} tile-atom pairs over {} \
             kernel evaluations. tile_atom_pairs is filled during BIND and the \
             kernel count during RENDER, so a zero here means one of the two \
             was read off the wrong report and the whole cost column is a \
             constant. candidates_tested is NOT checked here and is expected to \
             be zero: only the ray caster increments it, and the P2 harness is \
             fixed to the rasteriser. It ships as a column so the file has the \
             same shape whichever method a later run uses",
            row.target,
            row.blend,
            row.field,
            row.atoms,
            row.tile_atom_pairs,
            row.kernel_evaluations
        );
        assert!(
            row.max_tile_list as usize <= row.atoms,
            "{} {} {}: one tile holds {} candidates out of {} atoms in the \
             scene. A candidate list longer than the scene means the range \
             arithmetic over SplatRaster::ranges() is adding its endpoints \
             rather than subtracting them",
            row.target,
            row.blend,
            row.field,
            row.max_tile_list,
            row.atoms
        );
        assert!(
            row.mean_tile_list <= row.max_tile_list as f64 + 1.0e-12,
            "{} {} {}: mean list {} above max list {}",
            row.target,
            row.blend,
            row.field,
            row.mean_tile_list,
            row.max_tile_list
        );
        let floor = 4.0 / core::f64::consts::PI;
        assert!(
            row.box_over_ellipse_area >= floor - 1.0e-9,
            "{} {} {}: the axis-aligned bound is {} times the ellipse it \
             encloses. The box is 4 r^2 sqrt(cxx cyy) and the ellipse is pi r^2 \
             sqrt(det), and det <= cxx cyy always, so the ratio can never fall \
             below 4/pi = {floor}. A value under it means the binning waste is \
             being computed from the wrong covariance",
            row.target,
            row.blend,
            row.field,
            row.box_over_ellipse_area
        );
    }
    assert!(checked >= 2, "only {checked} rows carried a tile cost to check");

    for target in request.targets.iter() {
        let name = Target::<f64>::all()[*target].name;
        let natural = report
            .rows
            .iter()
            .find(|r| r.target == name && r.arm == ATOM_ARM && r.field == NATURAL);
        let tiled = report
            .rows
            .iter()
            .find(|r| r.target == name && r.field == TILED);
        match (natural, tiled) {
            (Some(n), Some(t)) => {
                assert!(
                    n.atoms < n.atoms_available,
                    "{name} met its criterion at {} of the {} atoms it placed. \
                     This test needs a cell where those two DIFFER, or tiling \
                     the whole fit and tiling the reported fit are the same \
                     thing and the assertion below is vacuous",
                    n.atoms,
                    n.atoms_available
                );
                assert_eq!(
                    t.atoms,
                    n.atoms * FIELD_COPIES,
                    "{name} tiles {} copies of a {}-atom fit into {} atoms. The \
                     tiled row must carry the SAME atoms as the natural row or \
                     the two cost numbers are not comparable and the tile-cost \
                     comparison measures the atom count instead",
                    FIELD_COPIES,
                    n.atoms,
                    t.atoms
                );
            }
            _ => panic!("{name} is missing a natural or a tiled row"),
        }
    }
}

#[test]
fn the_instrument_offset_comes_from_the_arm_whose_width_has_a_closed_form() {
    let mut request = small(false);
    request.targets = vec![1, 3];
    request.max_atoms = 1;
    let report = match run(&request) {
        Ok(r) => r,
        Err(e) => panic!("{e}"),
    };
    for row in report.rows.iter() {
        if row.target == "t2-straight-edge" {
            assert!(
                row.instrument_offset_px.abs() < 1.0e-12,
                "{} {} reports an instrument offset of {}. The offset is the \
                 analytic-SDF arm's measured width minus its closed form of \
                 exactly {SDF_ARM_TRUE_TRANSITION} px, and on the 30 degree \
                 edge that arm measures the closed form to the last bit. A \
                 non-zero value here means the offset was taken from the \
                 exact-coverage arm, which has no closed form and whose \
                 departure from {SDF_ARM_TRUE_TRANSITION} px is a property of \
                 the box filter rather than of the instrument",
                row.target,
                row.arm,
                row.instrument_offset_px
            );
        }
        if row.target == "t4-curve" {
            assert!(
                row.instrument_offset_px > 0.005,
                "t4 reports an instrument offset of {}. A 3 px radius circle \
                 carries about nineteen pixel centres per unit of signed \
                 distance, so its crossings are bracketed across gaps and the \
                 offset is real. Reporting zero there would hide the quantity \
                 that makes a glyph-scale width readable",
                row.instrument_offset_px
            );
        }
    }
}

#[test]
fn the_delta_to_control_column_is_a_measured_difference_and_not_a_constant() {
    let mut request = small(false);
    request.targets = vec![1];
    request.blends = vec![KernelFamily::Gaussian];
    request.max_atoms = 4;
    let report = match run(&request) {
        Ok(r) => r,
        Err(e) => panic!("{e}"),
    };
    let control = match report
        .rows
        .iter()
        .find(|r| r.arm == "analytic-exact" && r.target == "t2-straight-edge")
    {
        Some(r) => r.transition_10_90_px,
        None => panic!("the exact-coverage row the deltas are taken against is missing"),
    };
    let mut checked = 0usize;
    let mut moved = 0usize;
    for row in report.rows.iter() {
        if row.arm != ATOM_ARM || !row.transition_10_90_px.is_finite() {
            continue;
        }
        let expected = row.transition_10_90_px - control;
        assert!(
            (row.delta_to_control_px - expected).abs() < 1.0e-12,
            "{} {} {} reports delta_to_control {} while its own width {} minus \
             the exact-coverage width {} is {expected}. The report names this \
             column the headline - nothing can render a shape's boundary \
             sharper than its exact box coverage, so the absolute width means \
             little and the gap to the control means everything - and a column \
             hard-set to zero reads as ATOMS MATCH THE CONTROL EXACTLY, which \
             is the most flattering lie the table can tell",
            row.target,
            row.blend,
            row.profile,
            row.delta_to_control_px,
            row.transition_10_90_px,
            control
        );
        checked += 1;
        if row.delta_to_control_px.abs() > 1.0e-9 {
            moved += 1;
        }
    }
    assert!(
        checked > 0 && moved > 0,
        "{checked} atom rows carried a finite width and {moved} of them a \
         non-zero delta. With none the assertion above is satisfied by a \
         column of zeros against a control of zero and proves nothing"
    );
    let sdf = match report
        .rows
        .iter()
        .find(|r| r.arm == "analytic-sdf" && r.target == "t2-straight-edge")
    {
        Some(r) => r,
        None => panic!("the analytic-sdf row is missing, so its delta was never checked"),
    };
    let expected = sdf.transition_10_90_px - control;
    assert!(
        expected.is_finite() && expected.abs() > 1.0e-9,
        "the analytic-sdf width {} equals the exact width {control}, so a zeroed \
         delta could not be told from a measured one",
        sdf.transition_10_90_px
    );
    assert!(
        (sdf.delta_to_control_px - expected).abs() < 1.0e-12,
        "the analytic-sdf row reports delta_to_control {} while its own width {} \
         minus the exact-coverage width {control} is {expected}",
        sdf.delta_to_control_px,
        sdf.transition_10_90_px
    );
}

fn header_value(text: &str, name: &str) -> String {
    let wanted = format!("#{name}\t");
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix(wanted.as_str()) {
            return rest.to_string();
        }
    }
    panic!("no header line named {name} in:\n{text}");
}

fn cheap_binary_run(dir: &std::path::Path) -> (i32, String) {
    let mut command = binary();
    command.args([
        "--targets", "1", "--blends", "normalised", "--profiles", "prefiltered",
        "--max-atoms", "2", "--no-sensitivity", "--out-dir",
    ]);
    command.arg(dir);
    code(command)
}

fn threads_the_render_actually_used() -> usize {
    let configured = ProfileChoice::Prefiltered.settings().threads;
    if configured == 0 {
        return match std::thread::available_parallelism() {
            Ok(n) => n.get(),
            Err(_) => 1,
        };
    }
    configured
}

#[test]
fn the_reported_thread_count_is_the_one_the_timing_render_actually_ran_on() {
    let dir = scratch("threads");
    let (status, text) = cheap_binary_run(dir.path());
    assert_eq!(status, 0, "{text}");
    let table = read(&dir.path().join("report.tsv"));
    let stated = header_value(&table, "threads");
    let actual = threads_the_render_actually_used();
    assert_eq!(
        stated,
        actual.to_string(),
        "the header states `threads {stated}` while the profile it ran hands \
         the rasteriser `threads = {}`, which the splat back end resolves to \
         {actual} rayon workers. A header that names a configuration the run \
         did not use invalidates every render_ns column in the file, and it is \
         the one defect that cannot be caught by re-reading the numbers",
        ProfileChoice::Prefiltered.settings().threads
    );
}

#[test]
fn the_p2_report_records_the_command_that_produced_it() {
    let dir = scratch("invocation");
    let (status, text) = cheap_binary_run(dir.path());
    assert_eq!(status, 0, "{text}");
    for name in ["report.tsv", "sweep.tsv", "probes.tsv", "bins.tsv"] {
        let table = read(&dir.path().join(name));
        let first = match table.lines().next() {
            Some(l) => l,
            None => panic!("{name} is empty"),
        };
        assert!(
            first.starts_with("#invocation\t"),
            "{name} does not open with the argv that produced it. Without it \
             no published figure is reproducible from the artefact alone: \
             {first}"
        );
        assert!(
            first.contains("--max-atoms") && first.contains("normalised"),
            "{name} records an invocation that is missing arguments the run \
             was actually given: {first}"
        );
    }
}
