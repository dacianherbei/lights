use std::path::{Path, PathBuf};

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => panic!("cannot read {}: {e}", dir.display()),
    };
    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => panic!("bad entry under {}: {e}", dir.display()),
        };
        let path = entry.path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().map(|e| e == "rs").unwrap_or(false) {
            out.push(path);
        }
    }
}

fn source_files() -> Vec<PathBuf> {
    let mut out = Vec::new();
    rust_files(&crate_root().join("src"), &mut out);
    rust_files(&crate_root().join("examples"), &mut out);
    out.sort();
    assert!(out.len() > 10, "the source scan found nothing to check");
    out
}

fn all_files() -> Vec<PathBuf> {
    let mut out = source_files();
    rust_files(&crate_root().join("tests"), &mut out);
    out.sort();
    out
}

fn read(path: &Path) -> String {
    match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => panic!("cannot read {}: {e}", path.display()),
    }
}

fn relative(path: &Path) -> String {
    match path.strip_prefix(crate_root()) {
        Ok(p) => p.display().to_string(),
        Err(_) => path.display().to_string(),
    }
}

fn word_present(haystack: &str, needle: &str) -> bool {
    let bytes = haystack.as_bytes();
    let mut from = 0usize;
    for _ in 0..bytes.len() {
        let found = match haystack[from..].find(needle) {
            Some(i) => from + i,
            None => return false,
        };
        let before_ok = found == 0 || !is_ident_byte(bytes[found - 1]);
        let after = found + needle.len();
        let after_ok = after >= bytes.len() || !is_ident_byte(bytes[after]);
        if before_ok && after_ok {
            return true;
        }
        from = found + 1;
        if from >= bytes.len() {
            return false;
        }
    }
    false
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

#[test]
fn no_comments_in_src() {
    let line_comment = concat!("/", "/");
    let block_comment = concat!("/", "*");
    let pointer_marker = concat!("_comment", "/");

    for path in all_files() {
        let text = read(&path);
        let name = relative(&path);
        let comment_lines: Vec<&str> = text
            .lines()
            .filter(|l| l.contains(line_comment))
            .collect();
        assert!(
            comment_lines.len() <= 1,
            "{name} has {} comment lines; standing rule 1 allows at most ONE \
             pointer line per file",
            comment_lines.len()
        );
        if let Some(line) = comment_lines.first() {
            assert!(
                line.contains(pointer_marker),
                "{name}: the only permitted comment is a pointer to a sibling \
                 prose folder, got: {line}"
            );
            let pointed = line
                .trim()
                .trim_start_matches(line_comment)
                .trim();
            let folder = crate_root().join(pointed);
            assert!(
                folder.is_dir(),
                "{name} points at {pointed}, which does not exist"
            );
        }
        assert!(
            !text.contains(block_comment),
            "{name} contains a block comment"
        );
    }
}

#[test]
fn no_panic_paths_outside_guard() {
    let forbidden = [
        concat!("panic", "!"),
        concat!(".unwrap", "()"),
        concat!(".expect", "("),
        concat!("unreachable", "!("),
        concat!("todo", "!("),
        concat!("unimplemented", "!("),
    ];
    for path in source_files() {
        let name = relative(&path);
        if name.ends_with("guard.rs") {
            continue;
        }
        let text = read(&path);
        for token in forbidden {
            assert!(
                !text.contains(token),
                "{name} contains `{token}`: standing rules 2 and 6 put every hard \
                 failure behind guard::unreachable"
            );
        }
    }
    let guard = read(&crate_root().join("src/guard.rs"));
    assert!(
        guard.contains(concat!("panic", "!")),
        "guard.rs is where the one shared panic lives"
    );
}

fn wildcard_scope() -> Vec<PathBuf> {
    let mut paths = vec![
        crate_root().join("src/dispatch.rs"),
        crate_root().join("src/method/mod.rs"),
        crate_root().join("src/kernel/mod.rs"),
    ];
    rust_files(&crate_root().join("src/bin"), &mut paths);
    paths.sort();
    paths
}

#[test]
fn every_scan_reads_the_files_it_names() {
    let scanned: Vec<String> = all_files().iter().map(|p| relative(p)).collect();
    for file in NEVER_EXCEPTED {
        assert!(
            scanned.iter().any(|s| s == file),
            "{file} is named as a carrier of the path-traced reference, and the \
             photometric scan never reads it, so the ban on it holds nothing"
        );
    }
    let mut examples = Vec::new();
    rust_files(&crate_root().join("examples"), &mut examples);
    assert!(!examples.is_empty(), "the crate has no example to scan");
    for path in examples {
        let name = relative(&path);
        assert!(
            scanned.contains(&name),
            "{name} ships with the crate and is outside every file scan"
        );
    }
    let wildcard: Vec<String> = wildcard_scope().iter().map(|p| relative(p)).collect();
    assert!(
        wildcard.iter().any(|s| s == "src/kernel/mod.rs"),
        "the kernel axis is pruned by the same exhaustive matches as the method \
         axis, and src/kernel/mod.rs is outside the wildcard scan"
    );
}

#[test]
fn no_wildcard_match_arm_in_dispatch_or_forwarding() {
    let paths = wildcard_scope();
    assert!(paths.len() >= 4, "the wildcard scan found only {} files", paths.len());
    for path in paths {
        let relative_path = relative(&path);
        let text = read(&path);
        for (n, line) in text.lines().enumerate() {
            assert!(
                !line.contains("_ =>"),
                "{relative_path}:{}: a wildcard arm stops the compiler enumerating \
                 the pruning worklist, which is the single line that would \
                 silently destroy the prunability argument",
                n + 1
            );
        }
    }
}

#[test]
fn no_method_file_imports_the_agreement_harness() {
    let method_root = crate_root().join("src/method");
    let mut files = Vec::new();
    rust_files(&method_root, &mut files);
    assert!(files.len() >= 5, "the agreement-import scan found {} files", files.len());
    for path in files {
        let name = relative(&path);
        let text = read(&path);
        assert!(
            !text.contains("crate::agreement"),
            "{name} imports crate::agreement, which imports crate::dispatch and \
             crate::method: that closes a cycle and makes the harness \
             unprunable. The per-atom divergence bound lives in crate::bound, \
             which depends on numerics alone"
        );
    }
    let bound = read(&crate_root().join("src/bound.rs"));
    for upper in ["crate::dispatch", "crate::method", "crate::camera", "crate::agreement"] {
        assert!(
            !bound.contains(upper),
            "src/bound.rs references {upper}: it must stay below both renderers"
        );
    }
}

#[test]
fn the_pruned_methods_and_the_response_axis_stay_pruned() {
    let mut files = Vec::new();
    rust_files(&crate_root().join("src"), &mut files);
    assert!(
        files.len() >= 20,
        "the source scan found only {} files, so it cannot fail",
        files.len()
    );
    let pruned = [
        "RayMarch",
        "VoxelDda",
        "ConeBeam",
        "raymarch",
        "voxel_dda",
        "cone_beam",
        "SegmentIntegral",
        "ResponseModel",
        "ResponseMismatch",
    ];
    let mut found: Vec<String> = Vec::new();
    for path in files.iter() {
        let text = read(path);
        for token in pruned {
            if text.contains(token) {
                found.push(format!("{}: {token}", relative(path)));
            }
        }
    }
    assert!(
        found.is_empty(),
        "L-117, PM ruling 2026-09-28: the three parked methods and the segment-integral \
         response were pruned, and ResponseModel went with them because one value \
         compared with itself cannot fire. They are back in the source: {found:?}. \
         A method re-enters with its implementation and a consumer, not as a stub \
         returning MethodNotImplemented"
    );
    let mut methods: Vec<String> = match std::fs::read_dir(crate_root().join("src/method")) {
        Ok(entries) => entries
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect(),
        Err(e) => panic!("src/method: {e}"),
    };
    methods.sort();
    assert_eq!(
        methods,
        ["mod.rs", "raycast", "splat"],
        "src/method holds exactly the two built methods and the axis module"
    );
}

#[test]
fn no_cross_method_imports() {
    let method_root = crate_root().join("src/method");
    let mut files = Vec::new();
    rust_files(&method_root, &mut files);
    let siblings = ["splat", "raycast"];

    let mut checked = 0usize;
    for path in files {
        let name = relative(&path);
        let owner = siblings.iter().find(|s| {
            name == format!("src/method/{s}.rs") || name.starts_with(&format!("src/method/{s}/"))
        });
        let owner = match owner {
            Some(o) => *o,
            None => continue,
        };
        checked += 1;
        let text = read(&path);
        for other in siblings {
            if other == owner {
                continue;
            }
            let import = format!("crate::method::{other}");
            assert!(
                !text.contains(&import),
                "{name} imports {import}: pruning a losing method must stay a \
                 `git rm` of one directory"
            );
        }
    }
    assert!(checked >= 5, "the cross-method scan found only {checked} files");
}

#[test]
fn lower_layers_are_camera_free() {
    let lower = [
        "src/numerics/",
        "src/kernel/",
        "src/blend.rs",
        "src/image.rs",
        "src/tiling.rs",
        "src/probe.rs",
    ];
    let upper = ["crate::camera", "crate::scene", "crate::method"];

    let mut checked = 0usize;
    for path in source_files() {
        let name = relative(&path);
        if !lower.iter().any(|l| name.starts_with(l)) {
            continue;
        }
        checked += 1;
        let text = read(&path);
        for import in upper {
            assert!(
                !text.contains(import),
                "{name} references {import}: this seam is what lets the future \
                 camera-free 2D scalar-field path reuse these layers unchanged"
            );
        }
    }
    assert!(checked >= 9, "the lower-layer scan found only {checked} files");
}

#[test]
fn bounded_loops_carry_a_budget() {
    for path in source_files() {
        let name = relative(&path);
        let text = read(&path);
        let lines: Vec<&str> = text.lines().collect();
        for (n, line) in lines.iter().enumerate() {
            let unbounded = word_present(line, "while") || word_present(line, "loop");
            if !unbounded {
                continue;
            }
            let lo = n.saturating_sub(3);
            let hi = (n + 4).min(lines.len());
            let near_budget = lines[lo..hi].iter().any(|l| l.contains("Budget"));
            assert!(
                near_budget,
                "{name}:{}: standing rule 3 requires every loop that is not \
                 iterating a finite collection to carry a bounded counter: {line}",
                n + 1
            );
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Photometric {
    Clean,
    Forbidden,
    ReachesTheReference,
}

fn photometric_tokens() -> [&'static str; 3] {
    [concat!("ps", "nr"), concat!("ss", "im"), concat!("m", "se")]
}

const NEVER_EXCEPTED: [&str; 4] = [
    "src/agreement.rs",
    "src/scene/demo.rs",
    "src/bin/lights_agreement.rs",
    "examples/render_demo.rs",
];

const REFERENCE_CARRIERS: [&str; 5] = [
    "::agreement",
    "::scene::demo",
    "SPHERE_SCENE",
    "spheres_no_ground",
    "rustray",
];

const PHOTOMETRIC_EXCEPTIONS: [(&str, &str); 4] = [
    (
        "src/fit/metric.rs",
        "fit::coverage::exact_coverage, an adaptive quadrature of an SDF, whose \
         undecided area is measured and refused above EXACTNESS_LIMIT; or an \
         8-bit image::Image, whose quantisation floor is stated and a candidate \
         inside it refused",
    ),
    (
        "src/bin/lights_demo.rs",
        "the same exact raster, shown as the demo page's distortion panels: the \
         figure is printed beside the picture it scores, and the binary cannot \
         reach the sphere scene",
    ),
    (
        "tests/fit_metric.rs",
        "the same exact raster, built from Target::t2_straight_edge, plus the \
         refusal cases that prove the exactness bar is enforced, plus a \
         synthetic 8-bit raster and the quantisation-floor refusal",
    ),
    (
        "tests/fit_imagegs.rs",
        "six 8-bit CLIC2020 crops decoded by read_png and entered through \
         ExactReference::from_raster, the door the PM's L-43 ruling opened; no \
         render of the sphere scene is reachable from it",
    ),
];

fn photometric_verdict(name: &str, text: &str, exceptions: &[(&str, &str)]) -> Photometric {
    let lowered = text.to_ascii_lowercase();
    let carries = photometric_tokens()
        .iter()
        .any(|token| word_present(&lowered, token));
    if !carries {
        return Photometric::Clean;
    }
    let listed = exceptions.iter().any(|(file, _)| *file == name);
    let banned = NEVER_EXCEPTED.iter().any(|file| *file == name);
    if !listed || banned {
        return Photometric::Forbidden;
    }
    if REFERENCE_CARRIERS.iter().any(|c| text.contains(c)) {
        return Photometric::ReachesTheReference;
    }
    Photometric::Clean
}

#[test]
fn no_photometric_comparison_against_a_path_traced_reference() {
    assert!(
        !PHOTOMETRIC_EXCEPTIONS.is_empty(),
        "an empty exception table means this guard has never been seen to permit \
         anything, and a narrowing nobody exercised is a crate-wide ban wearing a \
         new shape"
    );
    for (file, ground_truth) in PHOTOMETRIC_EXCEPTIONS {
        assert!(
            !NEVER_EXCEPTED.contains(&file),
            "{file} is one of the four files that carry the path-traced reference \
             (tests/demo_smoke.rs:178-183) and can never be excepted"
        );
        let path = crate_root().join(file);
        assert!(
            path.is_file(),
            "{file} is listed as a photometric exception and does not exist"
        );
        let text = read(&path).to_ascii_lowercase();
        assert!(
            photometric_tokens()
                .iter()
                .any(|token| word_present(&text, token)),
            "{file} is listed as a photometric exception and uses none of the \
             tokens: delete the row rather than leaving a hole open for nothing"
        );
        assert!(
            ground_truth.len() > 20,
            "{file} must state the ground truth it compares against"
        );
    }

    let mut scanned = 0usize;
    for path in all_files() {
        let name = relative(&path);
        let text = read(&path);
        scanned += 1;
        let verdict = photometric_verdict(&name, &text, &PHOTOMETRIC_EXCEPTIONS);
        assert_ne!(
            verdict,
            Photometric::Forbidden,
            "{name} names a distortion metric and is not on the exception table. \
             A photometric comparison against the path-traced reference is \
             meaningless here: the two solve different transport equations, that \
             render is about 2x overbright from a verified double-count bug, its \
             entire sky is clipped to pure white, and its 121.3 degree horizontal \
             field of view breaks the EWA local-affine approximation outright. \
             Against an EXACT raster the same number is sound, which is what the \
             exception table is for: add the file there, state its ground truth, \
             and keep it unable to reach the reference"
        );
        assert_ne!(
            verdict,
            Photometric::ReachesTheReference,
            "{name} is a photometric exception AND names a carrier of the \
             path-traced reference. The exception is granted on the ground truth \
             being exact; a file that can reach the rustray scene has no such \
             guarantee"
        );
    }
    assert!(
        scanned > 10,
        "the photometric scan found {scanned} files: it must not be able to pass \
         by looking at nothing"
    );
}

#[test]
fn the_transform_arm_is_not_a_kernel() {
    let root = crate_root().join("src/fit/transform");
    let mut files = Vec::new();
    rust_files(&root, &mut files);
    files.sort();
    assert!(
        files.len() >= 4,
        "the transform scan found {} files: this guard must not be able to pass by looking \
         at nothing",
        files.len()
    );
    let forbidden = [
        "crate::kernel",
        "crate::scene",
        "crate::camera",
        "crate::method",
        "crate::blend",
        concat!("Kernel", " for"),
    ];
    let mut checked = 0usize;
    for path in files {
        let name = relative(&path);
        let text = read(&path);
        for token in forbidden {
            assert!(
                !text.contains(token),
                "{name} references `{token}`. A wavelet is a fixed dyadic lattice applied as \
                 a separable filter bank with downsampling; a Kernel is a free-floating \
                 anisotropic atom evaluated at an arbitrary subpixel offset. Nothing that \
                 takes a ScreenFrame and returns a scalar can express the first, and bending \
                 it into the trait would hand the transform arm the free placement and free \
                 anisotropy that the atom arms are the ones on trial for"
            );
        }
        checked += 1;
    }
    assert_eq!(checked, 4, "the transform arm is four files: mod, haar, cdf97, weight");
}

#[test]
fn the_layering_instrument_can_actually_fail() {
    let line_comment = concat!("/", "/");
    let with_two_comments = format!("{line_comment} one\n{line_comment} two\n");
    assert_eq!(
        with_two_comments
            .lines()
            .filter(|l| l.contains(line_comment))
            .count(),
        2,
        "the comment detector must see a comment when one is present"
    );

    assert!(word_present("match x { _ => 1 }", "_"));
    assert!(word_present("    loop {", "loop"));
    assert!(word_present("while budget.tick().is_ok()", "while"));
    assert!(
        !word_present("developer_loops_here", "loop"),
        "the loop detector must not fire on an identifier containing the word"
    );
    assert!(
        !word_present("meanwhile", "while"),
        "the loop detector must not fire inside a longer word"
    );
    assert!(!word_present("themselves", concat!("m", "se")));
    let photometric = concat!("ps", "nr");
    let embedded = format!("compute_{photometric}(a, b)");
    let bare = format!("let {photometric} = 1.0;");
    assert!(!word_present(&embedded, photometric));
    assert!(word_present(&bare, photometric));

    let table = [("src/fit/metric.rs", "an exact raster")];
    let poisoned = [("src/agreement.rs", "a claim this guard must not believe")];
    assert_eq!(
        photometric_verdict("src/fit/greedy.rs", &bare, &table),
        Photometric::Forbidden,
        "an unlisted file naming the token must be refused"
    );
    assert_eq!(
        photometric_verdict("src/fit/metric.rs", &bare, &table),
        Photometric::Clean,
        "a listed file naming the token must be permitted"
    );
    assert_eq!(
        photometric_verdict("src/agreement.rs", &bare, &poisoned),
        Photometric::Forbidden,
        "a file that carries the path-traced reference must be refused even when \
         somebody puts it on the exception table"
    );
    let reaching = format!("{bare}\nuse crate::scene::demo::SPHERE_SCENE;");
    assert_eq!(
        photometric_verdict("src/fit/metric.rs", &reaching, &table),
        Photometric::ReachesTheReference,
        "a listed file that reaches the reference must be refused"
    );
    assert_eq!(
        photometric_verdict("src/fit/metric.rs", &embedded, &table),
        Photometric::Clean,
        "the token detector must not fire inside a longer identifier"
    );

    for (fragment, what) in platform_fragments() {
        assert_eq!(
            platform_code(&format!("let x = {fragment};")),
            Some((fragment, what)),
            "the platform detector must see {what}"
        );
    }
    let gated = format!("#[{}(all(feature = \"fit\", any(unix, windows)))]", "cfg");
    assert_eq!(
        platform_code(&gated),
        Some(("unix", "a cfg on a target")),
        "a platform nested inside a cfg predicate must be seen"
    );
    let targeted = format!("if {}!({} = \"macos\") {{}}", "cfg", concat!("target", "_os"));
    assert_eq!(
        platform_code(&targeted).map(|hit| hit.1),
        Some("a cfg on a target"),
        "a target key inside cfg! must be seen"
    );
    let gate = format!("#[{}(enable = \"neon\")]", concat!("target", "_feature"));
    assert_eq!(
        platform_code(&gate).map(|hit| hit.1),
        Some("a gate on a CPU feature"),
        "a feature gate on a function must be seen"
    );
    let feature_only = format!("#[{}(feature = \"fit\")]", "cfg");
    for innocent in [
        feature_only.as_str(),
        "for pair in edges.windows(2) {",
        "let target_dir = 1; let target_envelope = 2;",
        "if cfg!(debug_assertions) {}",
        "let os = std::env::consts::OS;",
        "text.contains(\"aarch64\")",
    ] {
        assert_eq!(
            platform_code(innocent),
            None,
            "the platform detector fired on portable code: `{innocent}`"
        );
    }
    let program = platform_programs()[0];
    assert_eq!(platform_program(&format!("Command::new({program})")), Some(program));
    assert_eq!(platform_program("Command::new(\"git\")"), None);
    for name in [concat!("ob", "jc2-metal"), "cocoa", "core-foundation-sys", "metal"] {
        assert!(platform_dependency(name), "{name} binds an Apple framework");
    }
    for name in ["rayon", "png", "thiserror", "tiny-skia", "ttf-parser", "tempfile"] {
        assert!(!platform_dependency(name), "{name} is portable and was refused");
    }

    let src = crate_root().join("src/guard.rs");
    assert!(src.is_file(), "the scanner is looking at the wrong directory");
    assert!(
        read(&src).contains("Budget"),
        "guard.rs must define the shared budget"
    );
}

#[test]
fn the_fitter_never_reaches_back_into_the_core() {
    let reach = concat!("crate", "::fit");
    let allowed = ["src/lib.rs", "src/bin/lights_p2.rs"];

    let fit_root = crate_root().join("src/fit");
    let mut fit_files = Vec::new();
    rust_files(&fit_root, &mut fit_files);
    assert!(
        fit_files.len() >= 3,
        "the fitter tree holds only {} files: this scan must not be able to \
         pass by looking at nothing",
        fit_files.len()
    );

    let mut checked = 0usize;
    for path in source_files() {
        let name = relative(&path);
        if name.starts_with("src/fit/") || allowed.contains(&name.as_str()) {
            continue;
        }
        checked += 1;
        let text = read(&path);
        assert!(
            !text.contains(reach),
            "{name} references {reach}: the arrow runs one way only. The core \
             must build, and must be prunable, without an optimiser in it, \
             which is why `fit` is a non-default feature rather than a module"
        );
    }
    assert!(
        checked >= 20,
        "the fitter-arrow scan found only {checked} files outside src/fit/"
    );

    let manifest = read(&crate_root().join("Cargo.toml"));
    let default_line = match manifest
        .lines()
        .find(|l| l.trim_start().starts_with("default ="))
    {
        Some(l) => l,
        None => panic!("Cargo.toml has no `default =` feature list to check"),
    };
    assert!(
        !word_present(default_line, "fit"),
        "`fit` appears in the default feature list ({default_line}): \
         `cargo build -p lights` must not compile the fitter"
    );
    assert!(
        manifest.lines().any(|l| l.trim_start().starts_with("fit =")),
        "Cargo.toml declares no `fit` feature, so the gate the rest of this \
         test checks does not exist"
    );
}

#[test]
fn the_fit_feature_declares_every_method_it_cannot_compile_without() {
    let manifest = read(&crate_root().join("Cargo.toml"));
    let fit_line = match manifest
        .lines()
        .find(|l| l.trim_start().starts_with("fit ="))
    {
        Some(l) => l,
        None => panic!("Cargo.toml declares no `fit` feature line to check"),
    };

    let mut files = Vec::new();
    rust_files(&crate_root().join("src/fit"), &mut files);
    assert!(
        files.len() >= 3,
        "the fitter tree holds only {} files: this scan must not be able to \
         pass by looking at nothing",
        files.len()
    );

    let siblings = ["splat", "raycast"];
    let mut required: Vec<&str> = Vec::new();
    for path in files.iter() {
        let text = read(path);
        for sibling in siblings {
            let import = format!("crate::method::{sibling}");
            if text.contains(&import) && !required.contains(&sibling) {
                required.push(sibling);
            }
        }
    }
    assert!(
        !required.is_empty(),
        "the scan found no method dependency anywhere under src/fit, so it \
         cannot fail and is measuring nothing"
    );

    for sibling in required {
        let feature = format!("method-{}", sibling.replace('_', "-"));
        assert!(
            fit_line.contains(&feature),
            "src/fit references crate::method::{sibling} but the manifest line \
             `{fit_line}` does not make `fit` require `{feature}`. A feature \
             that silently needs another is a trap: \
             `cargo build -p lights --no-default-features --features fit` then \
             fails inside the absent-method stub with a type error that reads \
             as a bug in the fitter"
        );
    }
}

fn prose_files() -> Vec<PathBuf> {
    let mut out = Vec::new();
    collect_markdown(&crate_root(), &mut out);
    let docs = crate_root().join("../../docs/lights");
    if docs.is_dir() {
        collect_markdown(&docs, &mut out);
    }
    out.sort();
    out
}

fn collect_markdown(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => panic!("cannot read {}: {e}", dir.display()),
    };
    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => panic!("bad entry under {}: {e}", dir.display()),
        };
        let path = entry.path();
        if path.is_dir() {
            if path.file_name().map(|n| n == "target").unwrap_or(false) {
                continue;
            }
            collect_markdown(&path, out);
        } else if path.extension().map(|e| e == "md").unwrap_or(false) {
            out.push(path);
        }
    }
}

#[test]
fn no_prose_in_this_crate_claims_a_ci_that_does_not_exist() {
    let repo = crate_root().join("..").join("..");
    let configs = [
        repo.join(".github").join("workflows"),
        repo.join(".gitlab-ci.yml"),
        repo.join(".circleci"),
        repo.join("azure-pipelines.yml"),
        repo.join("Jenkinsfile"),
    ];
    let ci_exists = configs.iter().any(|p| p.exists());

    let claims = ["CI check", "CI gate", "CI job", "CI workflow", "CI run", "in CI"];
    let files = prose_files();
    assert!(
        files.len() > 5,
        "the prose scan found only {} markdown files, so it cannot fail",
        files.len()
    );
    let mut checked = 0usize;
    for path in files {
        let name = relative(&path);
        let text = read(&path);
        checked += 1;
        for claim in claims {
            let found = text.contains(claim);
            assert!(
                !found || ci_exists,
                "{name} says `{claim}` but this repository has no continuous \
                 integration at all: no workflow directory, no pipeline file, \
                 nothing that runs on a push. A command printed under the words \
                 `a green CI check` is a belief with a command under it, and the \
                 belief was carried for months without anything running it. \
                 Either wire the check into scripts/run_tests.sh and say so, or \
                 say plainly that it is a manual check"
            );
        }
    }
    assert!(checked > 5, "the prose scan checked only {checked} files");

    let script = read(&crate_root().join("scripts/run_tests.sh"));
    assert!(
        script.contains("--detachability"),
        "the detachability builds have no home in the one test script, so the \
         only thing standing behind `either method is detachable today` is a \
         paragraph"
    );
    for feature in ["method-splat", "method-raycast", "fit"] {
        assert!(
            script.contains(&format!("--features {feature}")),
            "scripts/run_tests.sh --detachability does not build \
             `--no-default-features --features {feature}`, so that feature's \
             detachability is still unchecked"
        );
    }
}

fn manifest_dependency_names(manifest: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut in_deps = false;
    for raw in manifest.lines() {
        let line = raw.trim();
        if line.starts_with('[') {
            let header = line.trim_matches(|c| c == '[' || c == ']');
            let table_dep = ["dependencies.", "dev-dependencies.", "build-dependencies."]
                .iter()
                .find_map(|p| header.strip_prefix(p).or_else(|| header.split_once(&format!(".{p}")).map(|(_, n)| n)));
            if let Some(name) = table_dep {
                names.push(name.to_string());
                in_deps = false;
            } else {
                in_deps = header.ends_with("dependencies");
            }
            continue;
        }
        if in_deps {
            if let Some((key, _)) = line.split_once('=') {
                let key = key.trim();
                if !key.is_empty() && !key.starts_with('#') {
                    names.push(key.to_string());
                }
            }
        }
    }
    names
}

#[test]
fn lights_takes_no_window_gpu_input_or_frame_clock_dependency() {
    let manifest = read(&crate_root().join("Cargo.toml"));
    let names = manifest_dependency_names(&manifest);
    assert!(
        names.iter().any(|n| n == "rayon") && names.iter().any(|n| n == "png"),
        "the dependency scan did not find rayon and png ({names:?}), so it \
         could pass by reading nothing"
    );
    let forbidden = [
        "winit", "wgpu", "raw-window-handle", "sdl2", "sdl3", "sdl3-sys", "glfw",
        "glutin", "glow", "eframe", "egui", "softbuffer", "minifb", "pixels", "ash",
        "vulkano", "metal", "gilrs",
    ];
    let hits: Vec<&String> = names.iter().filter(|n| forbidden.contains(&n.as_str())).collect();
    assert!(
        hits.is_empty(),
        "lights depends on {hits:?}: PM ruling 2026-09-16 (L-121) keeps lights \
         headless, because a render is a pure function of a scene and that is what \
         makes it bit-identical across thread counts; the window, input and frame \
         clock belong in a separate host crate that depends on lights"
    );
}

const PLATFORM_HOST: &str = "../lights-mac";

const PLATFORM_PROGRAM_EXCEPTIONS: [(&str, &str); 3] = [
    (
        "src/bin/lights_bench.rs",
        "the hardware line of every bench report: the CPU brand the kernel \
         reports, falling back to the architecture where the program is missing",
    ),
    (
        "src/bin/lights_ui_frame.rs",
        "the hardware line of L-175's report: the CPU brand and the macOS version, \
         a label printed beside numbers measured on this machine",
    ),
    (
        "examples/renderer_arms.rs",
        "the hardware line of L-176's report, from the same two programs as L-175's",
    ),
];

fn platform_fragments() -> [(&'static str, &'static str); 15] {
    [
        (concat!("ob", "jc"), "the Objective-C runtime or a binding to an Apple framework"),
        (concat!("msg", "_send"), "an Objective-C message send"),
        (concat!("sel", "!("), "an Objective-C selector"),
        (concat!("sel", "_registerName"), "an Objective-C selector"),
        (concat!("kind = \"frame", "work\""), "a link to an Apple framework"),
        (concat!("=frame", "work="), "a link to an Apple framework"),
        (concat!("core", "::arch"), "an architecture intrinsic"),
        (concat!("std", "::arch"), "an architecture intrinsic"),
        (concat!("as", "m!"), "inline assembly"),
        (concat!("_feature", "_detected"), "a probe for a CPU feature"),
        (concat!("os", "::unix"), "an OS-only module of std"),
        (concat!("os", "::macos"), "an OS-only module of std"),
        (concat!("os", "::darwin"), "an OS-only module of std"),
        (concat!("os", "::linux"), "an OS-only module of std"),
        (concat!("os", "::windows"), "an OS-only module of std"),
    ]
}

fn platform_cfg_words() -> [&'static str; 8] {
    [
        "unix",
        "windows",
        concat!("target", "_arch"),
        concat!("target", "_os"),
        concat!("target", "_vendor"),
        concat!("target", "_family"),
        concat!("target", "_env"),
        concat!("target", "_abi"),
    ]
}

fn platform_programs() -> [&'static str; 2] {
    [concat!("\"sys", "ctl\""), concat!("\"sw_", "vers\"")]
}

fn cfg_bodies(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    for (at, _) in text.match_indices("cfg") {
        if at > 0 && is_ident_byte(bytes[at - 1]) {
            continue;
        }
        let rest = &text[at + 3..];
        let Some(open) = ["(", "!(", "_attr("].iter().find(|p| rest.starts_with(**p)) else {
            continue;
        };
        let body = &rest[open.len()..];
        let mut depth = 1usize;
        let mut end = body.len();
        for (i, b) in body.bytes().enumerate() {
            if b == b'(' {
                depth += 1;
            }
            if b == b')' {
                depth -= 1;
                if depth == 0 {
                    end = i;
                    break;
                }
            }
        }
        out.push(&body[..end]);
    }
    out
}

fn platform_code(text: &str) -> Option<(&'static str, &'static str)> {
    if let Some(hit) = platform_fragments().into_iter().find(|(f, _)| text.contains(f)) {
        return Some(hit);
    }
    let gate = concat!("target", "_feature");
    if word_present(text, gate) {
        return Some((gate, "a gate on a CPU feature"));
    }
    for body in cfg_bodies(text) {
        if let Some(word) = platform_cfg_words().into_iter().find(|w| word_present(body, w)) {
            return Some((word, "a cfg on a target"));
        }
    }
    None
}

fn platform_program(text: &str) -> Option<&'static str> {
    platform_programs().into_iter().find(|p| text.contains(p))
}

fn platform_dependency(name: &str) -> bool {
    [concat!("ob", "jc"), "block2", "cocoa", "core-foundation", "core-graphics", "metal"]
        .iter()
        .any(|p| name.starts_with(p))
}

#[test]
fn platform_code_lives_in_the_host_crate_and_never_in_lights() {
    let host = crate_root().join(PLATFORM_HOST);
    assert!(
        host.join("Cargo.toml").is_file(),
        "{PLATFORM_HOST} is where the platform code lives (L-186 built it, L-193 \
         made it the one place), and it is not there: move this constant with it"
    );
    let mut host_files = Vec::new();
    rust_files(&host.join("src"), &mut host_files);
    let carriers = host_files
        .iter()
        .filter(|p| platform_code(&read(p)).is_some())
        .count();
    assert!(
        carriers > 0,
        "none of the {} files under {PLATFORM_HOST}/src carries platform code, so \
         this detector cannot see the shape platform code takes in this tree, and a \
         detector that sees nothing refuses nothing",
        host_files.len()
    );
    let own = manifest_dependency_names(&read(&crate_root().join("Cargo.toml")));
    let host_only: Vec<String> = manifest_dependency_names(&read(&host.join("Cargo.toml")))
        .into_iter()
        .filter(|n| n != "lights" && !own.contains(n))
        .collect();
    assert!(
        !host_only.is_empty(),
        "the host takes no dependency lights does not, so the manifest rule below \
         is checked against nothing"
    );
    for name in &host_only {
        assert!(
            platform_dependency(name),
            "the macOS host depends on {name}, and the rule that keeps platform \
             dependencies out of lights does not refuse it. If {name} binds a \
             platform, the rule has drifted from the host the way the L-121 list did, \
             which named metal and none of the five bindings the host takes: add it \
             to platform_dependency. If it is portable, lights may take it, and this \
             check has to learn to tell the two apart"
        );
    }

    let mut files = all_files();
    let build_script = crate_root().join("build.rs");
    if build_script.is_file() {
        files.push(build_script);
    }
    for path in &files {
        let name = relative(path);
        let text = read(path);
        if let Some((token, what)) = platform_code(&text) {
            panic!(
                "{name} carries {what} (`{token}`). Platform code lives in the host \
                 crate, crates/lights-mac, and never in lights (L-193). PM, \
                 2026-09-29: a Windows or Linux port is written from scratch and its \
                 shared part is isolated from two working implementations, never \
                 guessed from one, so the macOS code has to be findable in one place \
                 rather than woven through the core; and L-121 (PM, 2026-09-16) keeps \
                 lights headless. An architecture intrinsic cannot land in lights \
                 without a cfg arm for a platform that does not exist or a core that \
                 stops compiling off aarch64, and the portable arithmetic is the \
                 definition of correct that tests/determinism.rs pins, so a first one \
                 is a PM ruling and not a context's"
            );
        }
        let excepted = PLATFORM_PROGRAM_EXCEPTIONS.iter().any(|(file, _)| *file == name);
        if let Some(program) = platform_program(&text) {
            assert!(
                excepted,
                "{name} runs {program}, which answers only on macOS, and is not on \
                 PLATFORM_PROGRAM_EXCEPTIONS. Platform code lives in crates/lights-mac \
                 (L-193, PM 2026-09-29), and the only OS-only lines lights may carry \
                 are the hardware labels on that table, listed by file so that a port \
                 finds every one of them in one place: add the file there with its \
                 reason, or move the call into the host"
            );
        }
    }
    assert!(
        files.len() > 50,
        "the platform scan read {} files: it must not be able to pass by reading \
         nothing",
        files.len()
    );

    for (file, reason) in PLATFORM_PROGRAM_EXCEPTIONS {
        let path = crate_root().join(file);
        assert!(
            path.is_file(),
            "{file} is listed as running an OS-only program and does not exist"
        );
        assert!(
            platform_program(&read(&path)).is_some(),
            "{file} is listed as running an OS-only program and runs none: delete \
             the row rather than leaving a hole open for nothing"
        );
        assert!(reason.len() > 20, "{file} must state why it runs one");
    }

    let manifest = read(&crate_root().join("Cargo.toml"));
    assert!(
        own.iter().any(|n| n == "rayon"),
        "the dependency scan did not find rayon ({own:?}), so it could pass by \
         reading nothing"
    );
    let taken: Vec<&String> = own.iter().filter(|n| platform_dependency(n)).collect();
    assert!(
        taken.is_empty(),
        "lights depends on {taken:?}, a binding to an Apple framework: platform code \
         lives in crates/lights-mac and never in lights (L-193, PM 2026-09-29), and \
         L-121 keeps the window and the GPU out of it besides"
    );
    assert!(
        !manifest.lines().any(|l| l.trim_start().starts_with("[target.")),
        "lights' manifest carries a [target.] table: a dependency or a setting per \
         target is a cfg arm in the manifest, and platform code lives in \
         crates/lights-mac (L-193, PM 2026-09-29)"
    );
}
