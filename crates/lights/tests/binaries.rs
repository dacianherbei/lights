use std::process::Command;
use lights::method::RenderMethod;

fn scratch(name: &str) -> tempfile::TempDir {
    match tempfile::Builder::new().prefix(&format!("lights-binaries-{name}-")).tempdir() {
        Ok(d) => d,
        Err(e) => panic!("tempdir: {e}"),
    }
}

#[cfg(all(feature = "method-splat", feature = "method-raycast"))]
fn agreement() -> Command {
    Command::new(env!("CARGO_BIN_EXE_lights-agreement"))
}

fn code(mut command: Command) -> (i32, String) {
    let output = match command.output() {
        Ok(o) => o,
        Err(e) => panic!("running the binary: {e}"),
    };
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    (output.status.code().unwrap_or(-1), text)
}

#[test]
#[cfg(all(feature = "method-splat", feature = "method-raycast"))]
fn the_agreement_binary_refuses_to_write_to_the_repository_root() {
    let (status, text) = code(agreement());
    assert_eq!(
        status, 2,
        "--out-dir must be REQUIRED so the binary cannot write to the repository \
         root by accident: {text}"
    );
    assert!(text.contains("--out-dir is REQUIRED"), "{text}");
}

#[test]
#[cfg(all(feature = "method-splat", feature = "method-raycast"))]
fn the_agreement_binary_exits_clean_and_writes_four_pngs() {
    let dir = scratch("clean");
    let mut command = agreement();
    command.args([
        "--scene",
        "spheres-anisotropic-no-ground",
        "--width",
        "128",
        "--height",
        "72",
        "--out-dir",
    ]);
    command.arg(dir.path());
    let (status, text) = code(command);
    assert_eq!(status, 0, "a clean pair must exit zero: {text}");
    assert!(text.contains("AGREEMENT CLEAN"), "{text}");
    for name in ["splat.png", "raycast.png", "difference.png", "tolerance.png"] {
        assert!(
            dir.path().join(name).is_file(),
            "{name} was not written. Anyone looking at the difference image \
             without the tolerance image beside it will conclude the renderer is \
             broken at the periphery; writing BOTH is the point"
        );
    }
}

#[test]
#[cfg(all(feature = "method-splat", feature = "method-raycast"))]
fn the_agreement_binary_exits_nonzero_when_pixels_actually_fail() {
    let dir = scratch("loose");
    let mut command = agreement();
    command.args([
        "--scene",
        "needles",
        "--camera",
        "narrow",
        "--mode",
        "loose",
        "--width",
        "192",
        "--height",
        "144",
        "--out-dir",
    ]);
    command.arg(dir.path());
    let (status, text) = code(command);
    assert_eq!(
        status, 1,
        "A CI GATE THAT CANNOT FAIL IS NOT A GATE. In loose mode the ray caster \
         sorts by t* and the rasteriser by centre depth, so pixels genuinely \
         exceed the strict-mode tolerance and the binary must exit nonzero: \
         {text}"
    );
    assert!(text.contains("AGREEMENT FAILED"), "{text}");
    assert!(text.contains("failing pixels"), "{text}");
}

#[test]
#[cfg(all(feature = "method-splat", feature = "method-raycast"))]
fn the_agreement_binary_refuses_a_scene_that_violates_the_contract() {
    let dir = scratch("contract");
    let mut command = agreement();
    command.args(["--scene", "spheres", "--width", "128", "--height", "72", "--out-dir"]);
    command.arg(dir.path());
    let (status, text) = code(command);
    assert_eq!(
        status, 3,
        "the full sphere scene carries the ground atom, whose Jacobian clamp \
         has no ray-casting counterpart: {text}"
    );
    assert!(text.contains("CONTRACT VIOLATED"), "{text}");
    assert!(text.contains("ClampedAtomsPresent"), "{text}");
}

#[test]
fn the_benchmark_times_each_phase_around_the_phase_it_names() {
    let dir = scratch("bench-phases");
    let mut command = Command::new(env!("CARGO_BIN_EXE_lights-bench"));
    command.args(["--smoke", "--out-dir"]);
    command.arg(dir.path());
    let (status, text) = code(command);
    assert_eq!(status, 0, "{text}");
    assert!(text.contains("hardware:"), "the table must name the hardware: {text}");
    assert!(
        text.contains("aarch64") || text.contains("x86_64") || text.contains("arm"),
        "{text}"
    );

    let mut rows = 0usize;
    for line in text.lines() {
        if !line.starts_with("splat-raster") && !line.starts_with("ray-cast") {
            continue;
        }
        let phases: Vec<f64> = line
            .split_whitespace()
            .filter_map(|token| token.split("+-").next())
            .filter(|left| !left.is_empty())
            .filter_map(|left| left.parse::<f64>().ok())
            .collect();
        assert!(
            phases.len() >= 7,
            "unparsable row (the first parse of this table read the ATOM COUNT \
             and called it a render time, which is why the columns are now \
             split on their own +- markers): {line}"
        );
        let build_ms = phases[phases.len() - 6];
        let bind_ms = phases[phases.len() - 5];
        let clear_ms = phases[phases.len() - 4];
        let render_ms = phases[phases.len() - 3];
        let frame_ms = phases[phases.len() - 2];
        assert!(
            render_ms > 0.01,
            "a render phase measured at {render_ms} ms means the clock is not \
             around the render call: {line}"
        );
        assert!(
            build_ms > 1.0e-4,
            "a build phase measured at {build_ms} ms means the clock start was \
             moved off the phase it names. Building 2000 atoms takes tenths of \
             a millisecond even in this debug smoke build, so a floor of 1e-4 \
             ms cannot fire on a real measurement - and moving `started` past \
             PreparedScene::build makes the render > 2*build assertion EASIER \
             to satisfy, so that one actively rewards the bug: {line}"
        );
        assert!(
            bind_ms > 1.0e-4,
            "a bind phase measured at {bind_ms} ms means the same for \
             projection and sorting: {line}"
        );
        assert!(
            clear_ms > 1.0e-5,
            "the framebuffer clear is a real memset - up to 66 MB at 1080p in \
             f64 - and it is NOT part of the render phase. It gets its own \
             clock so that a render clock started before it is visible as a \
             double count: {line}"
        );
        assert!(
            render_ms > 2.0 * build_ms,
            "in the smoke case the render phase dominates the build phase by an \
             order of magnitude. build {build_ms} render {render_ms} means the \
             clock start was moved off the phase it names: {line}"
        );
        let summed = build_ms + bind_ms + clear_ms + render_ms;
        let residual = (frame_ms - summed).abs();
        assert!(
            residual < 0.5 * clear_ms && residual < 0.05 * frame_ms,
            "the per-frame column is measured by its OWN clock around the whole \
             block, so it is an INDEPENDENT check rather than a restatement of \
             the columns beside it: build {build_ms} + bind {bind_ms} + clear \
             {clear_ms} + render {render_ms} = {summed} against frame \
             {frame_ms}, residual {residual}. The tolerance is half the clear \
             phase precisely so that a render clock which has swallowed the \
             memset - the mutation that every other assertion in this test \
             passes through - fails here: {line}"
        );
        rows += 1;
    }
    assert_eq!(
        rows,
        usize::from(cfg!(feature = "method-splat"))
            + usize::from(cfg!(feature = "method-raycast")),
        "one row per compiled method: {text}"
    );
}

fn bench() -> Command {
    Command::new(env!("CARGO_BIN_EXE_lights-bench"))
}

#[test]
fn the_benchmark_runs_only_the_atom_ladder_it_is_given() {
    let dir = scratch("bench-atoms");
    let mut command = bench();
    command.args(["--quick", "--atoms", "300,700", "--out-dir"]);
    command.arg(dir.path());
    let (status, text) = code(command);
    assert_eq!(status, 0, "{text}");
    let report = read(&dir.path().join("report.tsv"));
    let names = columns(&report);
    let sweep = names.iter().position(|c| c == "sweep").unwrap_or(usize::MAX);
    let atoms = names.iter().position(|c| c == "atoms").unwrap_or(usize::MAX);
    let precision = names.iter().position(|c| c == "precision").unwrap_or(usize::MAX);
    assert!(sweep < names.len() && atoms < names.len() && precision < names.len(), "{report}");
    let mut seen = Vec::new();
    for line in report.lines().filter(|l| !l.starts_with('#')).skip(1) {
        let cells: Vec<&str> = line.split('\t').collect();
        assert_eq!(cells[sweep], "atom-count", "only the atom axis runs: {line}");
        seen.push((cells[atoms].to_string(), cells[precision].to_string()));
    }
    let methods = RenderMethod::implemented_for_timing().len();
    assert!(methods >= 1, "no method compiled in, so this test would measure nothing");
    assert_eq!(seen.len(), 2 * 2 * methods, "two counts x two precisions x methods: {report}");
    for count in ["300", "700"] {
        for prec in ["f32", "f64"] {
            assert!(
                seen.iter().any(|(a, p)| a == count && p == prec),
                "no {count}-atom {prec} row: {report}"
            );
        }
    }

    let mut refused = bench();
    refused.args(["--atoms", "300,zero", "--out-dir"]);
    refused.arg(dir.path());
    let (status, text) = code(refused);
    assert_eq!(status, 2, "a count that is not a number is refused: {text}");
}

#[cfg(feature = "fit")]
fn p3() -> Command {
    Command::new(env!("CARGO_BIN_EXE_lights-p3"))
}

fn read(path: &std::path::Path) -> String {
    match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => panic!("cannot read {}: {e}", path.display()),
    }
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

fn columns(text: &str) -> Vec<String> {
    for line in text.lines() {
        if !line.starts_with('#') {
            return line.split('\t').map(|s| s.to_string()).collect();
        }
    }
    panic!("no column line in:\n{text}");
}

fn data_rows(text: &str) -> Vec<Vec<String>> {
    text.lines()
        .filter(|l| !l.starts_with('#'))
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split('\t').map(|s| s.to_string()).collect())
        .collect()
}

fn cell(names: &[String], row: &[String], wanted: &str) -> String {
    let index = match names.iter().position(|n| n == wanted) {
        Some(i) => i,
        None => panic!("no column named {wanted} in {names:?}"),
    };
    match row.get(index) {
        Some(v) => v.clone(),
        None => panic!("row {row:?} is short of column {wanted}"),
    }
}

#[test]
fn the_benchmark_refuses_an_unknown_flag_instead_of_running_the_full_sweep() {
    let dir = scratch("bench-typo");
    let mut command = bench();
    command.args(["--smoke", "--smoek", "--out-dir"]);
    command.arg(dir.path());
    let (status, text) = code(command);
    assert_eq!(
        status, 2,
        "a mistyped flag must be REFUSED by name. `--smoke` rides along here \
         ONLY so that a regression cannot cost this test the twenty-minute full \
         sweep: drop `--smoke` and a binary that ignores `--smoek` measures \
         1000, 10k and 100k atoms at 1080p before failing. Silently accepting a \
         flag means the run that comes back is not the run that was asked for: \
         {text}"
    );
    assert!(
        text.contains("--smoek"),
        "the refusal must NAME the flag it did not understand: {text}"
    );
}

#[test]
fn the_benchmark_refuses_to_write_to_the_repository_root() {
    let mut command = bench();
    command.arg("--smoke");
    let (status, text) = code(command);
    assert_eq!(
        status, 2,
        "--out-dir must be REQUIRED so the binary cannot write to the \
         repository root by accident: {text}"
    );
    assert!(text.contains("--out-dir is REQUIRED"), "{text}");
}

#[test]
fn the_benchmark_writes_a_table_that_records_the_command_that_produced_it() {
    let dir = scratch("bench-tsv");
    let mut command = bench();
    command.args(["--smoke", "--out-dir"]);
    command.arg(dir.path());
    let (status, text) = code(command);
    assert_eq!(status, 0, "{text}");

    let table = read(&dir.path().join("report.tsv"));
    let first = match table.lines().next() {
        Some(l) => l,
        None => panic!("report.tsv is empty"),
    };
    assert!(
        first.starts_with("#invocation\t"),
        "the FIRST line of the report header must be the full argv that \
         produced it, or no figure in this file can ever be re-run: {first}"
    );
    assert!(
        first.contains("--smoke") && first.contains("--out-dir"),
        "the recorded invocation must carry every argument actually passed, \
         not a reconstruction: {first}"
    );

    let names = columns(&table);
    let rows = data_rows(&table);
    assert_eq!(
        rows.len(),
        usize::from(cfg!(feature = "method-splat"))
            + usize::from(cfg!(feature = "method-raycast")),
        "one machine-readable row per compiled method, matching the printed \
         table exactly: {table}"
    );
    for row in rows.iter() {
        assert_eq!(
            row.len(),
            names.len(),
            "a data row carries {} fields against {} columns: {row:?}",
            row.len(),
            names.len()
        );
        let render: f64 = match cell(&names, row, "render_ms_mean").parse() {
            Ok(v) => v,
            Err(e) => panic!("render_ms_mean is not a number: {e}"),
        };
        assert!(
            render > 0.0,
            "a TSV row that reports a zero render time is reporting a clock \
             that was never started: {row:?}"
        );
        let atoms: usize = match cell(&names, row, "atoms").parse() {
            Ok(v) => v,
            Err(e) => panic!("atoms is not a number: {e}"),
        };
        assert_eq!(atoms, 2_000, "the smoke case is 2000 atoms: {row:?}");
    }
    assert!(
        !header_value(&table, "hardware").is_empty(),
        "the table must name the hardware it was measured on"
    );
}

#[test]
#[cfg(all(feature = "method-splat", feature = "method-raycast"))]
fn the_agreement_binary_writes_a_table_that_records_the_command_that_produced_it() {
    let dir = scratch("agreement-tsv");
    let mut command = agreement();
    command.args([
        "--scene",
        "spheres-anisotropic-no-ground",
        "--width",
        "128",
        "--height",
        "72",
        "--out-dir",
    ]);
    command.arg(dir.path());
    let (status, text) = code(command);
    assert_eq!(status, 0, "{text}");

    let table = read(&dir.path().join("report.tsv"));
    let first = match table.lines().next() {
        Some(l) => l,
        None => panic!("report.tsv is empty"),
    };
    assert!(
        first.starts_with("#invocation\t"),
        "the FIRST line of the report header must be the full argv: {first}"
    );
    assert!(
        first.contains("spheres-anisotropic-no-ground"),
        "the recorded invocation must carry the scene that was measured: {first}"
    );

    let names = columns(&table);
    let rows = data_rows(&table);
    assert_eq!(rows.len(), 1, "one row per comparison: {table}");
    let row = &rows[0];
    assert_eq!(cell(&names, row, "failing_pixels"), "0", "{table}");
    assert_eq!(
        cell(&names, row, "pixels"),
        "9216",
        "128x72 is 9216 pixels and the table must say so: {table}"
    );
    let ratio: f64 = match cell(&names, row, "worst_ratio").parse() {
        Ok(v) => v,
        Err(e) => panic!("worst_ratio is not a number: {e}"),
    };
    assert!(
        ratio > 0.0 && ratio <= 1.0,
        "a clean run has every pixel inside its tolerance, so the worst ratio \
         is in (0, 1]: {ratio}"
    );
}

#[test]
#[cfg(feature = "fit")]
fn the_p3_binary_refuses_to_write_to_the_repository_root() {
    let (status, text) = code(p3());
    assert_eq!(
        status, 2,
        "--out-dir must be REQUIRED so the binary cannot write to the \
         repository root by accident: {text}"
    );
    assert!(text.contains("--out-dir is REQUIRED"), "{text}");
}

#[test]
#[cfg(feature = "fit")]
fn the_p3_binary_writes_an_atom_row_for_the_kernel_it_was_asked_to_sweep() {
    let dir = scratch("p3-clean");
    let mut command = p3();
    command.args([
        "--targets",
        "1",
        "--kernels",
        "gaussian",
        "--max-atoms",
        "2",
        "--repeats",
        "1",
        "--no-strips",
        "--out-dir",
    ]);
    command.arg(dir.path());
    let (status, text) = code(command);
    assert!(
        status == 0 || status == 1,
        "a cell that never meets its criterion is a RESULT (exit 1), not a \
         crash; anything else is a failure of the run: status {status}\n{text}"
    );

    let table = read(&dir.path().join("report.tsv"));
    let first = match table.lines().next() {
        Some(l) => l,
        None => panic!("report.tsv is empty"),
    };
    assert!(
        first.starts_with("#invocation\t"),
        "the FIRST line of the report header must be the full argv: {first}"
    );
    assert!(
        first.contains("--kernels") && first.contains("gaussian"),
        "the recorded invocation must carry the axis that was swept: {first}"
    );

    let names = columns(&table);
    let rows = data_rows(&table);
    let atom_rows: Vec<&Vec<String>> = rows
        .iter()
        .filter(|r| cell(&names, r, "arm") == "atoms")
        .collect();
    assert_eq!(
        atom_rows.len(),
        1,
        "one target crossed with one kernel is exactly one atom row; a \
         different count means the binary did not run the axis it was given: \
         {table}"
    );
    let row = atom_rows[0];
    assert_eq!(cell(&names, row, "blend"), "gaussian", "{table}");
    assert_eq!(cell(&names, row, "profile"), "contract", "{table}");
    let available: usize = match cell(&names, row, "atoms_available").parse() {
        Ok(v) => v,
        Err(e) => panic!("atoms_available is not a number: {e}"),
    };
    assert_eq!(
        available, 2,
        "--max-atoms 2 must reach the fitter as a budget of 2: {table}"
    );
    assert!(
        rows.iter().any(|r| cell(&names, r, "arm").starts_with("analytic")),
        "the analytic control arm must be carried alongside the swept kernel, \
         or the kernel column has nothing to be read against: {table}"
    );
}

fn python() -> Command {
    Command::new("python3")
}

fn fake_tree(dir: &std::path::Path) -> std::path::PathBuf {
    let scripts = dir.join("crates").join("lights").join("scripts");
    if let Err(e) = std::fs::create_dir_all(&scripts) {
        panic!("cannot build the fake tree: {e}");
    }
    let source = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("scripts")
        .join("mutate.py");
    let copy = scripts.join("mutate.py");
    if let Err(e) = std::fs::copy(&source, &copy) {
        panic!("cannot copy mutate.py: {e}");
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
fn the_serial_mutation_driver_refuses_to_mutate_a_tree_it_was_not_given() {
    let dir = scratch("mutate-no-tree");
    let script = fake_tree(dir.path());
    let mut command = python();
    command.arg(&script);
    let (status, text) = code(command);
    assert_eq!(
        status, 2,
        "mutate.py edits files IN PLACE. Run against the tree it lives in, a \
         crash between apply and revert leaves the user's working copy \
         mutated, so it must refuse to start without an isolated tree: \
         status {status}\n{text}"
    );
    assert!(
        text.contains("no isolated tree given"),
        "the refusal must say WHICH guard fired, so that losing this one and \
         landing on a neighbouring guard is visible rather than silent: {text}"
    );
    assert!(
        text.contains("--tree") && text.contains("--snapshot"),
        "the refusal must say how to give it an isolated tree: {text}"
    );
}

#[test]
fn the_serial_mutation_driver_refuses_the_tree_it_lives_in() {
    let dir = scratch("mutate-self");
    let script = fake_tree(dir.path());
    let mut command = python();
    command.arg(&script);
    command.arg("--tree");
    command.arg(dir.path());
    let (status, text) = code(command);
    assert_eq!(
        status, 2,
        "naming its OWN tree is the exact hazard the guard exists for, so \
         pointing --tree at it must be refused too: status {status}\n{text}"
    );
    assert!(
        text.contains("overlaps the tree"),
        "the refusal must say WHICH guard fired. The fake tree here carries a \
         crates/lights/Cargo.toml precisely so that a driver whose overlap \
         guard is gone runs on to completion instead of being caught by the \
         next check for the wrong reason: {text}"
    );
    assert!(
        text.contains("mutate_par.py"),
        "the refusal must point at the isolated driver: {text}"
    );
}

#[test]
#[cfg(feature = "fit")]
fn the_p5_help_names_every_class_in_its_catalogue_and_both_spellings() {
    let mut command = Command::new(env!("CARGO_BIN_EXE_lights-p5"));
    command.arg("--help");
    let (_, help) = code(command);

    let mut named = 0usize;
    for name in [
        "t2-straight-edge",
        "t3-right-angle-corner",
        "n1-colour-field",
        "t6-glyph-like",
        "t5-thin-stroke",
        "n3-band-limited-noise",
        "n2-soft-shadow",
    ] {
        assert!(
            help.contains(name),
            "the p5 help does not mention the class {name}, so a class that exists \
             and can be asked for by name is undiscoverable from the binary itself. \
             This is how the help came to say `1..6` against a catalogue of seven: \
             n2-soft-shadow was added to the ladder and the hand-written usage \
             string was not. Help:\n{help}"
        );
        named += 1;
    }
    assert!(named >= 1);

    assert!(
        help.contains(&format!("1..{named}")),
        "the p5 help must state the index range that matches its catalogue of \
         {named}, or `--classes 7` reads as a bad flag when it is a real class. \
         Help:\n{help}"
    );
    assert!(
        help.contains("`every`"),
        "`every` is the ONLY spelling that runs the whole catalogue -- `all` runs \
         the default four -- so a help that does not name it leaves no documented \
         way to run every class. Help:\n{help}"
    );
    assert!(
        help.contains("NOT the whole catalogue"),
        "`all` running four of {named} is the trap this entry exists for, and the \
         help has to say so in as many words rather than leaving the reader to \
         infer it from a parenthesised default. Help:\n{help}"
    );
}

#[test]
#[cfg(feature = "fit")]
fn the_p5_ladder_table_records_the_command_that_produced_it() {
    let dir = scratch("p5-invocation");
    let mut command = Command::new(env!("CARGO_BIN_EXE_lights-p5"));
    command.args([
        "--classes",
        "t6",
        "--direction",
        "magnify",
        "--profile",
        "contract",
        "--max-atoms",
        "1",
        "--terms",
        "8",
        "--no-search-probe",
        "--no-strips",
        "--out-dir",
    ]);
    command.arg(dir.path());
    let (status, text) = code(command);
    assert_eq!(status, 0, "{text}");
    let table = read(&dir.path().join("ladder-magnify.tsv"));
    let first = match table.lines().next() {
        Some(l) => l,
        None => panic!("ladder-magnify.tsv is empty"),
    };
    assert!(
        first.starts_with("#invocation\t"),
        "the FIRST line of the ladder table must be the full argv that \
         produced it: {first}"
    );
    assert!(
        first.contains("--terms") && first.contains("magnify"),
        "the recorded invocation must carry the rung count and the direction, \
         which are the two knobs that decide what the ladder measures: {first}"
    );
}

fn demo() -> Command {
    Command::new(env!("CARGO_BIN_EXE_lights-demo"))
}

#[test]
fn the_demo_refuses_to_write_to_the_repository_root() {
    let (status, text) = code(demo());
    assert_eq!(
        status, 2,
        "--out-dir must be REQUIRED so the binary cannot write to the \
         repository root by accident: {text}"
    );
    assert!(text.contains("--out-dir is REQUIRED"), "{text}");
}

#[test]
fn the_demo_draws_one_panel_for_every_method_that_is_built() {
    let dir = scratch("demo");
    let mut command = demo();
    command.args([
        "--out-dir",
        &dir.path().display().to_string(),
        "--width",
        "96",
        "--height",
        "64",
        "--frames",
        "2",
        "--reps",
        "2",
    ]);
    let (status, text) = code(command);
    assert_eq!(status, 0, "{text}");

    let table = read(&dir.path().join("demo.tsv"));
    let first = match table.lines().next() {
        Some(l) => l,
        None => panic!("demo.tsv is empty"),
    };
    assert!(
        first.starts_with("#invocation\t"),
        "the FIRST line of the manifest must be the argv that produced it, or \
         the page it feeds states numbers no one can re-run: {first}"
    );

    let mut columns: Vec<&str> = Vec::new();
    let mut rows: Vec<Vec<&str>> = Vec::new();
    for line in table.lines() {
        if line.starts_with('#') {
            continue;
        }
        let cells: Vec<&str> = line.split('\t').collect();
        if columns.is_empty() {
            columns = cells;
            continue;
        }
        rows.push(cells);
    }
    let column = |name: &str| -> usize {
        match columns.iter().position(|c| *c == name) {
            Some(i) => i,
            None => panic!("the manifest has no {name} column: {columns:?}"),
        }
    };
    let (kind, group, file) = (column("kind"), column("group"), column("file"));

    let built = lights::method::RenderMethod::implemented();
    assert!(
        !built.is_empty(),
        "no method is built into this test binary, so this case would pass by \
         measuring nothing"
    );
    let method_panels = rows
        .iter()
        .filter(|r| r[kind] == "panel" && r[group] == "the method axis")
        .count();
    assert_eq!(
        method_panels,
        built.len(),
        "the demo must draw ONE panel per method RenderMethod::implemented() \
         returns — {} built, {method_panels} drawn. A panel that is not drawn \
         is a feature nobody can see, which is the whole point of this binary",
        built.len()
    );

    let frames = rows.iter().filter(|r| r[kind] == "frame").count();
    assert_eq!(
        frames, 2,
        "--frames 2 must produce exactly two orbit frames, or the page's \
         moving panel is not showing the run it names"
    );

    for row in rows.iter() {
        if row[file].is_empty() {
            continue;
        }
        let path = dir.path().join(row[file]);
        assert!(
            path.exists(),
            "the manifest names {} and the file is not there, so the page it \
             feeds would carry an empty frame",
            row[file]
        );
    }
}

#[test]
#[cfg(feature = "method-raycast")]
fn the_demo_times_the_ray_caster_through_its_tile_index() {
    let dir = scratch("demo-accel");
    let mut command = demo();
    command.args([
        "--out-dir",
        &dir.path().display().to_string(),
        "--width",
        "96",
        "--height",
        "64",
        "--frames",
        "1",
        "--reps",
        "1",
    ]);
    let (status, text) = code(command);
    assert_eq!(status, 0, "{text}");

    let table = read(&dir.path().join("demo.tsv"));
    let mut columns: Vec<&str> = Vec::new();
    let mut ray_cast = 0usize;
    for line in table.lines() {
        if line.starts_with('#') {
            continue;
        }
        let cells: Vec<&str> = line.split('\t').collect();
        if columns.is_empty() {
            columns = cells;
            continue;
        }
        let at = |name: &str| -> &str {
            match columns.iter().position(|c| *c == name) {
                Some(i) => cells[i],
                None => panic!("the manifest has no {name} column: {columns:?}"),
            }
        };
        if at("kind") != "panel" || at("method") != "ray-cast" || at("atoms") == "0" {
            continue;
        }
        ray_cast += 1;
        assert_ne!(
            at("tile_pairs"),
            "0",
            "panel {} times the ray caster with NO tile index: tile_pairs is 0 \
             for {} atoms, which is AccelKind::BruteForce, the contract config \
             RenderMethod::implemented() hands out. Its ms_min is O(pixels x \
             atoms) and says nothing about the renderer anyone would ship",
            at("id"),
            at("atoms")
        );
    }
    assert!(
        ray_cast > 0,
        "the ray caster is built and no panel drew through it, so this case \
         would pass by measuring nothing"
    );
}

#[cfg(feature = "fit")]
fn atom_curve() -> Command {
    Command::new(env!("CARGO_BIN_EXE_lights-p3"))
}

#[test]
#[cfg(feature = "fit")]
fn the_atom_curve_binary_refuses_to_write_to_the_repository_root() {
    let (status, text) = code(atom_curve());
    assert_eq!(
        status, 2,
        "lights-p3 is the runner behind the aspect-1 kernel-curve result and \
         nothing in this crate ran it until 2026-09-15: {text}"
    );
    assert!(text.contains("--out-dir is REQUIRED"), "{text}");
}

#[cfg(feature = "fit")]
fn tables_named_in(usage: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for piece in usage.split(|c: char| !(c.is_ascii_alphanumeric() || c == '.' || c == '_')) {
        if piece.ends_with(".tsv") && !out.iter().any(|k| k == piece) {
            out.push(piece.to_string());
        }
    }
    out
}

#[test]
#[cfg(feature = "fit")]
fn the_atom_curve_binary_writes_every_table_its_own_usage_names() {
    let mut help = atom_curve();
    help.arg("--help");
    let (help_status, usage) = code(help);
    assert_eq!(help_status, 2, "{usage}");
    let named = tables_named_in(&usage);
    assert!(
        named.len() >= 4,
        "the usage block must name the tables it writes, so this test measures \
         the binary's own promise rather than a list that drifts beside it. \
         Found {named:?} in: {usage}"
    );

    let dir = scratch("atom-curve");
    let mut command = atom_curve();
    command.args([
        "--targets",
        "2",
        "--kernels",
        "gaussian",
        "--max-atoms",
        "2",
        "--repeats",
        "1",
        "--no-strips",
        "--out-dir",
    ]);
    command.arg(dir.path());
    let (status, text) = code(command);
    assert_eq!(
        status, 1,
        "a cell that never meets its criterion inside --max-atoms is a RESULT and \
         the binary documents exit 1 for it; at two atoms on the edge that is the \
         expected outcome: {text}"
    );
    assert!(text.contains("NEVER MET THE CRITERION"), "{text}");

    for table in named.iter() {
        let path = dir.path().join(table);
        assert!(
            path.is_file(),
            "the usage block names {table} and the run did not write it: {text}"
        );
        let body = match std::fs::read_to_string(&path) {
            Ok(b) => b,
            Err(e) => panic!("{}: {e}", path.display()),
        };
        assert!(
            body.starts_with("#invocation\t"),
            "{table} carries no invocation stamp, so its numbers name no command \
             that reproduces them"
        );
        assert!(
            body.lines().count() > 1,
            "{table} is a header and nothing else"
        );
    }

    let mut pngs = 0usize;
    let entries = match std::fs::read_dir(dir.path()) {
        Ok(e) => e,
        Err(e) => panic!("reading the out dir: {e}"),
    };
    for entry in entries {
        let path = match entry {
            Ok(v) => v.path(),
            Err(e) => panic!("bad entry: {e}"),
        };
        if path.extension().map(|x| x == "png").unwrap_or(false) {
            pngs += 1;
        }
    }
    assert!(pngs >= 3, "the per-cell images must land beside the tables, got {pngs}");
}
