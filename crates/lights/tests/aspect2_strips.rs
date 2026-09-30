#![cfg(feature = "fit")]
use lights::fit::analytic::AnalyticArm;
use lights::fit::aspect2::{
    ABSENT_GROUND, ABSENT_HATCH_PERIOD, ABSENT_INK, SEPARATOR, TRUTH_LABEL, TermChoice, TermCount,
    absent_panel, caption, class_panels, control_panel, separator, strip_for_class, truth_panel,
};
use lights::fit::content::{CONTROL_ABSENT, Content};
use lights::fit::coverage::{DEFAULT_CELL_LIMIT, DEFAULT_MAX_DEPTH, coverage_budget};
use lights::fit::nterm::{Arm, NTermRequest, REACHABLE_ARMS, positions};
use lights::fit::strip::upscale;
use lights::image::FloatImage;
use lights::numerics::vector::Rgb;

fn truth_of(content: &Content) -> FloatImage<f64> {
    let mut budget = coverage_budget(DEFAULT_CELL_LIMIT);
    match content.truth(DEFAULT_MAX_DEPTH, &mut budget) {
        Ok(t) => t,
        Err(e) => panic!("{} produced no truth image: {e}", content.name()),
    }
}

fn request(max_atoms: usize) -> NTermRequest {
    let mut r = NTermRequest::new(REACHABLE_ARMS[0]);
    r.max_atoms = max_atoms;
    r
}

fn bits(colour: Rgb<f64>) -> (u64, u64, u64) {
    (colour.x.to_bits(), colour.y.to_bits(), colour.z.to_bits())
}

fn same_bits(a: &FloatImage<f64>, b: &FloatImage<f64>) -> bool {
    if a.width != b.width || a.height != b.height {
        return false;
    }
    a.pixels
        .iter()
        .zip(b.pixels.iter())
        .all(|(p, q)| bits(*p) == bits(*q))
}

#[test]
fn the_analytic_column_is_absent_on_every_new_class_and_the_truth_itself_on_every_carried_forward_one()
 {
    let request = request(1);
    let mut absent_seen = 0usize;
    let mut present_seen = 0usize;
    for content in Content::all() {
        let truth = truth_of(&content);
        let panel = match control_panel(&content, &truth, &request) {
            Ok(p) => p,
            Err(e) => panic!("{}: {e}", content.name()),
        };
        let (width, height) = content.viewport();
        match content {
            Content::Raster(_) => {
                absent_seen += 1;
                assert!(
                    panel.absent,
                    "{} has no SDF description, so its control column must declare itself \
                     ABSENT rather than carry an image",
                    content.name()
                );
                assert_eq!(panel.label, CONTROL_ABSENT, "{}", content.name());
                assert_eq!(content.control_label(), CONTROL_ABSENT, "{}", content.name());
                assert!(
                    same_bits(&panel.image, &absent_panel(width, height)),
                    "{}: the control slot must be the marker itself, bit for bit. Putting the \
                     truth image there would say an analytic arm reproduced this class exactly",
                    content.name()
                );
                assert!(
                    !same_bits(&panel.image, &truth),
                    "{}: the marker must not be the truth image",
                    content.name()
                );
                assert_eq!(panel.relative_l2, None, "{}", content.name());
            }
            Content::Analytic(_) => {
                present_seen += 1;
                assert!(!panel.absent, "{}", content.name());
                assert_eq!(panel.label, AnalyticArm::Exact.name(), "{}", content.name());
                assert!(
                    same_bits(&panel.image, &truth),
                    "{}: exact box coverage of the description IS how the truth image is \
                     built, so the control column must coincide with it bit for bit. A \
                     difference means one of the two paths quietly changed",
                    content.name()
                );
                assert!(
                    !same_bits(&panel.image, &absent_panel(width, height)),
                    "{}: a class that HAS a control must not be marked absent",
                    content.name()
                );
                assert_eq!(panel.relative_l2, Some(0.0), "{}", content.name());
            }
        }
    }
    assert_eq!(absent_seen, 4, "four new classes carry no analytic control");
    assert_eq!(present_seen, 4, "four carried-forward classes do carry one");
}

#[test]
fn the_absent_marker_is_a_diagonal_hatch_no_content_class_can_produce() {
    let panel = absent_panel(64, 64);
    let ink = Rgb::new(ABSENT_INK.0, ABSENT_INK.1, ABSENT_INK.2);
    let ground = Rgb::new(ABSENT_GROUND.0, ABSENT_GROUND.1, ABSENT_GROUND.2);
    assert_ne!(bits(ink), bits(ground), "the two hatch colours must differ");

    let mut inks = 0usize;
    let mut grounds = 0usize;
    for p in panel.pixels.iter() {
        if bits(*p) == bits(ink) {
            inks += 1;
        } else if bits(*p) == bits(ground) {
            grounds += 1;
        } else {
            panic!("the marker carries a third colour {p:?}");
        }
    }
    let total = panel.pixels.len();
    assert!(
        inks * 4 > total && grounds * 4 > total,
        "a hatch is two colours in comparable amounts, not a flat fill: {inks} ink, \
         {grounds} ground of {total}"
    );

    let mut horizontal_changes = 0usize;
    for y in 0..panel.height {
        for x in 0..panel.width - 1 {
            if bits(panel.get(x, y)) != bits(panel.get(x + 1, y)) {
                horizontal_changes += 1;
            }
        }
    }
    assert!(
        horizontal_changes > (panel.width * panel.height / (2 * ABSENT_HATCH_PERIOD)) as usize,
        "the ruling must cross every row about once per {ABSENT_HATCH_PERIOD} px, got \
         {horizontal_changes} changes"
    );

    for y in 1..panel.height {
        for x in 0..panel.width - 1 {
            assert_eq!(
                bits(panel.get(x, y)),
                bits(panel.get(x + 1, y - 1)),
                "the hatch runs along the anti-diagonal, so ({x},{y}) and ({},{}) are one band",
                x + 1,
                y - 1
            );
        }
    }

    for content in Content::all() {
        let truth = truth_of(&content);
        for p in truth.pixels.iter() {
            assert_ne!(
                bits(*p),
                bits(ink),
                "{} produces the marker ink exactly, so a reader could mistake the ABSENT \
                 column for content",
                content.name()
            );
        }
    }
    assert_ne!(
        bits(ink),
        bits(Rgb::new(SEPARATOR.0, SEPARATOR.1, SEPARATOR.2)),
        "the marker must not be the separator colour either"
    );
}

#[test]
fn every_arm_gets_a_column_and_the_column_order_is_the_same_on_every_class() {
    let arms = REACHABLE_ARMS.to_vec();
    let mut expected: Vec<String> = vec![TRUTH_LABEL.to_string(), "control".to_string()];
    for arm in arms.iter() {
        expected.push(arm.name());
    }

    let mut orders = Vec::new();
    for content in [Content::Analytic(
        lights::fit::target::Target::t2_straight_edge(),
    ), Content::n3_band_limited_noise()] {
        let class = match class_panels(&content, &arms, TermChoice::SharedCeiling, &request(2)) {
            Ok(c) => c,
            Err(e) => panic!("{}: {e}", content.name()),
        };
        assert_eq!(
            class.panels.len(),
            arms.len() + 2,
            "{}: truth, control and one column per arm",
            content.name()
        );
        assert_eq!(class.column_labels(), expected, "{}", content.name());
        for panel in class.panels.iter().skip(2) {
            let error = match panel.relative_l2 {
                Some(e) => e,
                None => panic!("{} {}: an arm column must carry its own error", content.name(), panel.label),
            };
            assert!(
                error > 0.0,
                "{} {}: no arm reproduces a class exactly at the shared ceiling, so an error \
                 of zero means the error is being measured against the render itself",
                content.name(),
                panel.label
            );
            assert!(error.is_finite(), "{} {}", content.name(), panel.label);
        }
        orders.push(class.column_labels());
    }
    assert_eq!(
        orders[0], orders[1],
        "a column may only be compared across classes if it is in the same place on both"
    );
}

#[test]
fn the_shared_ceiling_is_the_smallest_in_the_strip_and_it_is_the_atom_ceiling() {
    let content = Content::n3_band_limited_noise();
    let arms = REACHABLE_ARMS.to_vec();
    let class = match class_panels(&content, &arms, TermChoice::SharedCeiling, &request(2)) {
        Ok(c) => c,
        Err(e) => panic!("{e}"),
    };
    let (width, height) = content.viewport();
    let coefficients = positions(width, height);

    let mut atom_ceiling = usize::MAX;
    let mut wavelet_ceiling = 0usize;
    for panel in class.panels.iter().skip(2) {
        let terms = match panel.terms {
            Some(t) => t,
            None => panic!("{}: an arm column must count its terms", panel.label),
        };
        if panel.has_fitter {
            atom_ceiling = atom_ceiling.min(terms.available);
        } else {
            wavelet_ceiling = wavelet_ceiling.max(terms.available);
        }
        assert_eq!(terms.rendered, class.requested, "{}", panel.label);
    }
    assert_eq!(wavelet_ceiling, coefficients, "a transform reaches 3*W*H terms");
    assert_eq!(coefficients, 12288, "n3 is 64x64 in three colour planes");
    assert!(
        atom_ceiling <= 2,
        "the fitter was given two atoms, so the atom ceiling is at most two, got {atom_ceiling}"
    );
    assert_eq!(
        class.shared_ceiling, atom_ceiling,
        "the shared count is the SMALLEST ceiling in the strip, which is the atom ceiling: \
         that gap, {atom_ceiling} against {coefficients}, IS the asymmetry the banner states"
    );
    assert_eq!(class.requested, class.shared_ceiling);
}

#[test]
fn a_request_beyond_an_arm_ceiling_renders_the_ceiling_and_the_caption_says_so() {
    let content = Content::n3_band_limited_noise();
    let arms = REACHABLE_ARMS.to_vec();
    let asked = 64usize;
    let class = match class_panels(&content, &arms, TermChoice::Explicit(asked), &request(2)) {
        Ok(c) => c,
        Err(e) => panic!("{e}"),
    };
    assert_eq!(class.requested, asked);
    assert!(class.any_at_ceiling(), "the atom columns cannot reach {asked} terms");

    let mut clipped = 0usize;
    let mut full = 0usize;
    for panel in class.panels.iter().skip(2) {
        let terms = match panel.terms {
            Some(t) => t,
            None => panic!("{}: an arm column must count its terms", panel.label),
        };
        assert_eq!(terms.requested, asked, "{}", panel.label);
        if panel.has_fitter {
            clipped += 1;
            assert!(terms.at_ceiling(), "{}", panel.label);
            assert_eq!(terms.rendered, terms.available, "{}", panel.label);
            assert!(terms.rendered < asked, "{}", panel.label);
            let text = panel.caption();
            assert!(
                text.contains("CEILING") && text.contains(&format!("{asked} requested")),
                "{}: a clipped column must say so, got `{text}`",
                panel.label
            );
        } else {
            full += 1;
            assert!(!terms.at_ceiling(), "{}", panel.label);
            assert_eq!(
                terms.rendered, asked,
                "{}: a transform has 12288 coefficients and must keep exactly the {asked} \
                 that were asked for",
                panel.label
            );
            assert!(!panel.caption().contains("CEILING"), "{}", panel.label);
        }
    }
    assert_eq!(clipped, 2, "the two atom columns are the clipped ones");
    assert_eq!(full, 2, "the two transform columns are not");

    let line = caption(&class, 1);
    assert!(line.contains("CEILING"), "{line}");
    assert_eq!(line.lines().count(), 1, "a caption is one line: {line}");
}

#[test]
fn a_ceiling_column_is_not_the_same_picture_as_the_column_below_it() {
    let content = Content::n3_band_limited_noise();
    let arms = vec![Arm::Wavelet(
        lights::fit::transform::Transform::Haar,
        lights::fit::transform::Ranking::Weighted,
    )];
    let low = match class_panels(&content, &arms, TermChoice::Explicit(4), &request(1)) {
        Ok(c) => c,
        Err(e) => panic!("{e}"),
    };
    let high = match class_panels(&content, &arms, TermChoice::Explicit(256), &request(1)) {
        Ok(c) => c,
        Err(e) => panic!("{e}"),
    };
    let a = &low.panels[2];
    let b = &high.panels[2];
    assert!(
        !same_bits(&a.image, &b.image),
        "keeping 4 coefficients and keeping 256 must not render the same image, or the term \
         count is not reaching the reconstruction"
    );
    let (ea, eb) = match (a.relative_l2, b.relative_l2) {
        (Some(x), Some(y)) => (x, y),
        _ => panic!("both columns must carry an error"),
    };
    assert!(
        eb < ea,
        "256 coefficients must beat 4 on the same class: {eb} against {ea}"
    );
    assert!(
        ea > 0.5,
        "four coefficients of 12288 cannot reproduce a noise field; an error of {ea} suggests \
         the truncation is not truncating"
    );
}

#[test]
fn the_strip_is_the_panels_side_by_side_over_a_separator_ground() {
    let content = Content::n1_colour_field();
    let arms = REACHABLE_ARMS.to_vec();
    let class = match class_panels(&content, &arms, TermChoice::SharedCeiling, &request(2)) {
        Ok(c) => c,
        Err(e) => panic!("{e}"),
    };
    let gap = 3u32;
    let strip = match strip_for_class(&class, gap) {
        Ok(s) => s,
        Err(e) => panic!("{e}"),
    };
    let (width, height) = content.viewport();
    let columns = class.panels.len() as u32;
    assert_eq!(strip.height, height);
    assert_eq!(strip.width, columns * width + (columns - 1) * gap);

    let ground = separator();
    let mut cursor = 0u32;
    for panel in class.panels.iter() {
        for y in 0..height {
            for x in 0..width {
                assert_eq!(
                    bits(strip.get(cursor + x, y)),
                    bits(panel.image.get(x, y)),
                    "{} moved when it was placed in the strip",
                    panel.label
                );
            }
        }
        cursor += width;
        if cursor < strip.width {
            for y in 0..height {
                for g in 0..gap {
                    assert_eq!(
                        bits(strip.get(cursor + g, y)),
                        bits(ground),
                        "the bar between panels must be the separator colour and nothing else"
                    );
                }
            }
        }
        cursor += gap;
    }

    let zoomed = match upscale(&strip, 4) {
        Ok(z) => z,
        Err(e) => panic!("{e}"),
    };
    assert_eq!(zoomed.width, strip.width * 4);
    assert_eq!(zoomed.height, strip.height * 4);
    for y in 0..zoomed.height {
        for x in 0..zoomed.width {
            assert_eq!(
                bits(zoomed.get(x, y)),
                bits(strip.get(x / 4, y / 4)),
                "the 4x strip must repeat pixels and invent none"
            );
        }
    }
}

#[test]
fn the_truth_column_is_the_truth_and_carries_no_term_count() {
    let content = Content::n2_soft_shadow();
    let truth = truth_of(&content);
    let panel = truth_panel(&truth);
    assert!(same_bits(&panel.image, &truth));
    assert_eq!(panel.terms, None);
    assert_eq!(panel.relative_l2, None);
    assert!(!panel.absent);
    assert_eq!(panel.label, TRUTH_LABEL);
}

#[test]
fn the_term_count_clips_to_what_an_arm_has_and_reports_both_numbers() {
    let cases: [(usize, usize, usize, bool); 5] = [
        (1, 12288, 1, false),
        (64, 12288, 64, false),
        (64, 2, 2, true),
        (2, 2, 2, false),
        (12288, 12288, 12288, false),
    ];
    for (requested, available, rendered, at_ceiling) in cases {
        let terms = TermCount::resolve(requested, available);
        assert_eq!(terms.requested, requested);
        assert_eq!(terms.available, available);
        assert_eq!(
            terms.rendered, rendered,
            "asking {requested} of {available} renders {rendered}"
        );
        assert_eq!(
            terms.at_ceiling(),
            at_ceiling,
            "asking {requested} of {available} is at the ceiling: {at_ceiling}"
        );
    }
}

fn strip_binary() -> std::process::Command {
    std::process::Command::new(env!("CARGO_BIN_EXE_lights-p4"))
}

fn run(mut command: std::process::Command) -> (i32, String) {
    let output = match command.output() {
        Ok(o) => o,
        Err(e) => panic!("running lights-p4: {e}"),
    };
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    (output.status.code().unwrap_or(-1), text)
}

fn scratch(name: &str) -> tempfile::TempDir {
    match tempfile::Builder::new().prefix(&format!("lights-p4-{name}-")).tempdir() {
        Ok(d) => d,
        Err(e) => panic!("tempdir: {e}"),
    }
}

#[test]
fn the_strip_binary_refuses_to_write_to_the_repository_root() {
    let (status, text) = run(strip_binary());
    assert_eq!(status, 2, "--out-dir must be REQUIRED: {text}");
    assert!(text.contains("--out-dir is REQUIRED"), "{text}");
}

#[test]
fn the_strip_binary_writes_both_zooms_and_prints_a_manifest_of_absolute_paths() {
    let dir = scratch("both-zooms");
    if let Err(e) = std::fs::create_dir_all(dir.path()) {
        panic!("{dir:?}: {e}");
    }
    let mut command = strip_binary();
    command.current_dir(dir.path());
    command.args(["--classes", "n3", "--max-atoms", "2", "--out-dir", "."]);
    let (status, text) = run(command);
    assert_eq!(status, 0, "a clean run exits zero: {text}");

    assert!(
        text.contains("ABSENT") && text.contains("no fitter"),
        "the banner must carry the ABSENT column and the fitter asymmetry: {text}"
    );
    let manifest = match text.split("=== IMAGE MANIFEST ===").nth(1) {
        Some(m) => m,
        None => panic!("no manifest was printed: {text}"),
    };
    let paths: Vec<&str> = manifest
        .lines()
        .map(|l| l.trim_end())
        .filter(|l| l.ends_with(".png"))
        .collect();
    assert_eq!(paths.len(), 2, "one class produces a 1x and a 4x strip: {manifest}");
    let mut captions = Vec::new();
    for line in manifest.lines() {
        if line.starts_with("    ") {
            captions.push(line.trim());
        }
    }
    assert_eq!(captions.len(), 2, "every path carries exactly one caption line: {manifest}");
    for caption_line in captions.iter() {
        assert!(
            caption_line.contains("n3-band-limited-noise") && caption_line.contains("ABSENT"),
            "a caption names its class and says the control is absent: {caption_line}"
        );
    }

    let mut images = Vec::new();
    for path in paths.iter() {
        let path = std::path::Path::new(path);
        assert!(
            path.is_absolute(),
            "the manifest prints ABSOLUTE paths. This run was given --out-dir `.` from inside \
             the scratch directory precisely so that echoing the argument back would fail \
             here instead of passing by accident: {path:?}"
        );
        assert!(path.is_file(), "the manifest names a file that exists: {path:?}");
        match lights::image::read_png(path) {
            Ok(image) => images.push(image),
            Err(e) => panic!("{path:?}: {e}"),
        }
    }
    let (one, four) = (&images[0], &images[1]);
    assert_eq!(four.width, one.width * 4, "the second strip is the 4x one");
    assert_eq!(four.height, one.height * 4);
    for y in 0..four.height {
        for x in 0..four.width {
            let a = four.pixels[(y as usize) * (four.width as usize) + (x as usize)];
            let source = one.pixels
                [((y / 4) as usize) * (one.width as usize) + ((x / 4) as usize)];
            assert_eq!(
                a, source,
                "the 4x file must be the 1x file with every pixel repeated, not a second \
                 render and not a resampled copy"
            );
        }
    }
}

#[test]
fn the_strip_binary_names_the_term_count_in_the_file_it_writes() {
    let dir = scratch("term-count");
    let mut command = strip_binary();
    command.args(["--classes", "n1", "--terms", "48", "--max-atoms", "2", "--out-dir"]);
    command.arg(dir.path());
    let (status, text) = run(command);
    assert_eq!(status, 0, "{text}");
    assert!(
        dir.path().join("strip-n1-colour-field-n48.png").is_file(),
        "a strip carries its term count in its name so two term counts cannot overwrite each \
         other: {text}"
    );
    assert!(dir.path().join("strip-n1-colour-field-n48-4x.png").is_file(), "{text}");
    assert!(
        text.contains("CEILING"),
        "at 48 terms the atom columns are pinned at their ceiling and the caption must say \
         so: {text}"
    );
}

#[test]
fn an_atom_column_renders_the_prefix_its_caption_claims_and_not_the_whole_fitted_set() {
    let content = Content::n2_soft_shadow();
    let arms = vec![Arm::Atoms(lights::kernel::KernelFamily::Gaussian)];
    let base = request(3);
    let one = match class_panels(&content, &arms, TermChoice::Explicit(1), &base) {
        Ok(c) => c,
        Err(e) => panic!("{e}"),
    };
    let three = match class_panels(&content, &arms, TermChoice::Explicit(3), &base) {
        Ok(c) => c,
        Err(e) => panic!("{e}"),
    };
    let a = &one.panels[2];
    let b = &three.panels[2];
    let (ta, tb) = match (a.terms, b.terms) {
        (Some(x), Some(y)) => (x, y),
        _ => panic!("an atom column must count its terms"),
    };
    assert_eq!(ta.rendered, 1);
    assert!(
        tb.rendered > ta.rendered,
        "the fitter was given three atoms and placed {}, so this test can only see the \
         difference if it placed more than one",
        tb.available
    );
    assert!(
        !same_bits(&a.image, &b.image),
        "a column captioned n=1 must render ONE atom. Rendering the whole fitted set at every \
         requested count would put the same picture under two different numbers"
    );
}

#[test]
fn an_atom_column_says_whether_its_count_is_the_budget_or_the_fitter_and_a_transform_says_neither() {
    let content = Content::n3_band_limited_noise();
    let arms = REACHABLE_ARMS.to_vec();
    let base = request(2);
    let class = match class_panels(&content, &arms, TermChoice::SharedCeiling, &base) {
        Ok(c) => c,
        Err(e) => panic!("{e}"),
    };

    let truth = truth_of(&content);
    let scaffold = content.fit_scaffold();
    let mut fit_request =
        lights::fit::greedy::FitRequest::new(lights::kernel::KernelFamily::Gaussian, base.settings);
    fit_request.max_atoms = base.max_atoms;
    fit_request.tile = base.tile;
    fit_request.refine_iterations = base.refine_iterations;
    fit_request.refine_tail = base.refine_tail;
    fit_request.initialisation = base.initialisation;
    let mut budget = lights::fit::greedy::fit_budget(base.render_limit);
    let outcome = match lights::fit::greedy::fit(&scaffold, &truth, &fit_request, &mut budget) {
        Ok(o) => o,
        Err(e) => panic!("{e}"),
    };

    let mut atom_columns = 0usize;
    let mut transform_columns = 0usize;
    for panel in class.panels.iter().skip(2) {
        if panel.has_fitter {
            atom_columns += 1;
            assert!(
                panel.at_own_ceiling(),
                "{}: at the shared count every atom column sits on its own ceiling",
                panel.label
            );
            if panel.label.contains("atoms-gaussian") {
                assert_eq!(
                    panel.exhausted,
                    Some(outcome.improvement_exhausted),
                    "{}: the column must report the fitter's OWN stop reason, not a constant",
                    panel.label
                );
            }
            let stop = match panel.exhausted {
                Some(e) => e,
                None => panic!("{}: an atom column knows why it stopped", panel.label),
            };
            let text = panel.caption();
            if stop {
                assert!(text.contains("fitter exhausted"), "{text}");
                assert!(!panel.budget_stop(), "{text}");
            } else {
                assert!(
                    text.contains("BUDGET STOP"),
                    "{}: {} atoms is --max-atoms, and the fitter had not run out of \
                     improvements, so the count is the BUDGET and the caption must say so: \
                     {text}",
                    panel.label,
                    base.max_atoms
                );
                assert!(panel.budget_stop(), "{text}");
            }
        } else {
            transform_columns += 1;
            assert_eq!(
                panel.exhausted, None,
                "{}: a transform has no fitter, so it has no stop reason to report",
                panel.label
            );
            assert!(!panel.budget_stop(), "{}", panel.label);
            let text = panel.caption();
            assert!(!text.contains("BUDGET STOP"), "{text}");
            assert!(!text.contains("fitter exhausted"), "{text}");
        }
    }
    assert_eq!(atom_columns, 2);
    assert_eq!(transform_columns, 2);
    assert_eq!(
        class.any_budget_stop(),
        !outcome.improvement_exhausted,
        "the strip-level flag must follow the columns"
    );
    if class.any_budget_stop() {
        assert!(caption(&class, 1).contains("BUDGET STOP"), "{}", caption(&class, 1));
    }
}

#[test]
fn the_strip_binary_writes_a_manifest_table_that_records_its_own_invocation() {
    let dir = scratch("manifest-table");
    let mut command = strip_binary();
    command.args(["--classes", "n3", "--max-atoms", "2", "--zoom", "2", "--out-dir"]);
    command.arg(dir.path());
    let (status, text) = run(command);
    assert_eq!(status, 0, "a clean run exits zero: {text}");

    let table = match std::fs::read_to_string(dir.path().join("report.tsv")) {
        Ok(s) => s,
        Err(e) => panic!("report.tsv: {e}. The strips are the whole output of \
                          this run and nothing machine-readable said which \
                          arguments drew them"),
    };
    let first = match table.lines().next() {
        Some(l) => l,
        None => panic!("report.tsv is empty"),
    };
    assert!(
        first.starts_with("#invocation\t"),
        "the FIRST line of the report header must be the full argv that \
         produced the PNGs beside it: {first}"
    );
    assert!(
        first.contains("n3") && first.contains("--max-atoms"),
        "the recorded invocation must carry the class and the budget, which \
         are exactly the two knobs that change what the strip shows: {first}"
    );

    let names: Vec<&str> = match table.lines().find(|l| !l.starts_with('#')) {
        Some(l) => l.split('\t').collect(),
        None => panic!("report.tsv has no column line"),
    };
    let rows: Vec<Vec<&str>> = table
        .lines()
        .filter(|l| !l.starts_with('#'))
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split('\t').collect())
        .collect();
    assert_eq!(
        rows.len(),
        2,
        "one class at two zooms is two PNGs, so the table carries two rows or \
         it is not a manifest of what was written: {table}"
    );
    let index = |wanted: &str| -> usize {
        match names.iter().position(|n| *n == wanted) {
            Some(i) => i,
            None => panic!("no column named {wanted} in {names:?}"),
        }
    };
    let zoom = index("zoom");
    let png = index("png");
    let content = index("content");
    let mut zooms: Vec<&str> = rows.iter().map(|r| r[zoom]).collect();
    zooms.sort();
    assert_eq!(zooms, vec!["1", "2"], "{table}");
    for row in rows.iter() {
        assert_eq!(row.len(), names.len(), "short row {row:?}");
        assert!(row[content].contains("n3"), "{table}");
        assert!(
            !row[png].contains('/'),
            "the png column names a strip RELATIVE to the run directory, so the \
             committed table does not carry one machine's paths; it holds {}",
            row[png]
        );
        let path = dir.path().join(row[png]);
        assert!(
            path.is_file(),
            "the table names {} which was never written: a manifest that \
             points at nothing is worse than no manifest",
            row[png]
        );
    }
}
