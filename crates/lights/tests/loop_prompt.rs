use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p
}

fn prompt() -> String {
    let path = repo_root().join("docs/lights/BURNDOWN_LOOP.md");
    match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => panic!(
            "docs/lights/BURNDOWN_LOOP.md is the prompt every iteration of this \
             loop is driven from and it could not be read at {}: {e}. If this \
             fired inside a mutation worker under /Volumes/RamBuild/mutpool*, \
             the cause is that scripts/mutate_par.py's OUTSIDE list no longer \
             copies this file into the worker tree (it does since L-152, \
             2026-09-28). A row may name a file outside the crate only if \
             OUTSIDE copies it; before L-152 nothing was copied, and two such \
             rows were deleted on 2026-09-15 for that reason",
            path.display()
        ),
    }
}

fn artifact_urls(text: &str) -> Vec<String> {
    let needle = concat!("https:", "/", "/", "claude.ai/code/artifact/");
    let mut out = Vec::new();
    let mut rest = text;
    let mut budget = 64usize;
    while let Some(at) = rest.find(needle) {
        match budget.checked_sub(1) {
            Some(n) => budget = n,
            None => panic!("more artifact urls than this scan budgets for"),
        }
        let tail = &rest[at + needle.len()..];
        let end = tail
            .find(|c: char| !(c.is_ascii_hexdigit() || c == '-'))
            .unwrap_or(tail.len());
        let id = &tail[..end];
        if !out.iter().any(|u: &String| u == id) {
            out.push(id.to_string());
        }
        rest = &tail[end..];
    }
    out
}

#[test]
fn every_file_the_loop_prompt_tells_a_context_to_run_exists() {
    let text = prompt();
    assert!(
        text.len() > 4000,
        "the prompt is {} bytes, which is far short of the document this test \
         is pointed at; it would pass by inspecting the wrong file",
        text.len()
    );

    let named = [
        "crates/lights/head_open.toml",
        "crates/lights/tail.toml",
        "crates/lights/scripts/split_register.py",
        "crates/lights/scripts/burndown.py",
        "crates/lights/scripts/burndown_selftest.py",
        "crates/lights/scripts/run_tests.sh",
        "crates/lights/scripts/mutate_par.py",
        "crates/lights/scripts/mutations.json",
        "crates/lights/scripts/research.py",
        "crates/lights/scripts/demo_page.py",
        "crates/lights/src/bin/lights_demo.rs",
        "crates/lights/tests/layering.rs",
        "crates/lights/scripts/burndown_loop.sh",
        "crates/lights/scripts/loop_usage.py",
        "crates/lights/scripts/burndown_host.py",
        "crates/lights/scripts/burndown_watch.py",
        "crates/lights/scripts/burndown_host_selftest.py",
        "crates/lights/scripts/loop_usage_selftest.py",
        "crates/lights/scripts/burndown_tasks.json",
        "docs/lights/RESEARCH.json",
        "docs/lights/burndown_history.json",
        "docs/lights/mutations/results.tsv",
    ];
    let root = repo_root();
    let mut checked = 0usize;
    for path in named {
        assert!(
            text.contains(path) || text.contains(path.trim_start_matches("crates/lights/")),
            "{path} is asserted here as something the prompt names, and the \
             prompt does not name it. Either the prompt dropped it or this \
             list is stale — both are findings"
        );
        assert!(
            root.join(path).exists(),
            "the prompt tells a fresh context to run or read {path} and no such \
             file exists. That is the shape of L-63: a mandatory step naming \
             something that cannot be executed"
        );
        checked += 1;
    }
    assert_eq!(checked, 22, "the sweep asserted nothing if it ran no rows");
}

#[test]
fn the_prompt_carries_three_distinct_artifact_urls_and_no_placeholder() {
    let text = prompt();
    let urls = artifact_urls(&text);
    assert_eq!(
        urls.len(),
        3,
        "the prompt names the research record, the burn-down report and the \
         demo, and each is a page the publishing session deploys the file \
         under docs/lights/pages to (L-148). It carries {} distinct artifact \
         urls: {urls:?}",
        urls.len()
    );
    for url in urls.iter() {
        assert_eq!(
            url.len(),
            36,
            "{url} is not a uuid, so a context pasting it would publish a \
             SEPARATE artifact and hand the PM a link nobody is watching"
        );
        assert_eq!(
            url.chars().filter(|c| *c == '-').count(),
            4,
            "{url} is not shaped like a uuid"
        );
    }

    for token in ["ARTIFACT_URL", "REPORT_URL", "DEMO_URL", "RESEARCH_URL", "TODO", "TBD"] {
        assert!(
            !text.contains(token),
            "the prompt contains the literal token {token}. L-63 was exactly \
             this: `ARTIFACT_URL` sat in the highest-priority step of the \
             loop, so every iteration met an instruction it could not carry \
             out, and the close bar that step gates could never be cleared"
        );
    }
}

#[test]
fn the_prompts_parallelism_figures_agree_with_the_consolidated_research_record() {
    let text = prompt();
    let record = match std::fs::read_to_string(repo_root().join("docs/lights/RESEARCH.md")) {
        Ok(t) => t,
        Err(e) => panic!(
            "docs/lights/RESEARCH.md is the one consolidated research document \
             and it could not be read: {e}"
        ),
    };

    let mut checked = 0usize;
    for figure in ["6.01", "7.00", "8.83"] {
        assert!(
            record.contains(figure),
            "{figure} is asserted here as a figure the record carries and the \
             record does not carry it; this list is stale"
        );
        assert!(
            text.contains(figure),
            "the prompt quotes a parallelism figure the record does not \
             support, or dropped {figure}. Until 2026-09-15 it said 5.7-7.0x \
             end to end and a 2-11x tile slowdown, and NEITHER number appears \
             in any tree: the record has 6.01x/7.00x/6.59x against an 8.83x \
             ceiling, and 1.7x-4.3x. That is L-64"
        );
        checked += 1;
    }
    for withdrawn in ["5.7", "2\u{2013}11", "2-11x"] {
        for line in text.lines() {
            if !line.contains(withdrawn) {
                continue;
            }
            assert!(
                line.contains("wrong here until") || line.contains("L-64"),
                "the prompt carries the withdrawn figure {withdrawn} on a line \
                 that does not mark it as withdrawn: {line}. A number in the \
                 prompt is quoted by every context this loop starts, so a wrong \
                 one here propagates faster than one in a results file. The old \
                 sentence may STAY - this crate keeps withdrawals visible - but \
                 only beside the words that withdraw it"
            );
        }
        checked += 1;
    }
    assert_eq!(checked, 6, "the sweep asserted nothing if it ran no rows");
}

fn run_tests_sh_commands() -> Vec<String> {
    let path = repo_root().join("crates/lights/scripts/run_tests.sh");
    let script = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => panic!("{} could not be read: {e}", path.display()),
    };
    let mut out: Vec<String> = Vec::new();
    for line in script.lines() {
        if line.trim_start().starts_with("echo ") {
            continue;
        }
        let Some(at) = line.find("cargo ") else { continue };
        let tail = &line[at..];
        if !(tail.starts_with("cargo test -p lights") || tail.starts_with("cargo build -p lights")) {
            continue;
        }
        let end = [" >", " 2>&1", ";", " |"]
            .iter()
            .filter_map(|stop| tail.find(stop))
            .min()
            .unwrap_or(tail.len());
        let command = tail[..end]
            .replace("$FEATURES", "--all-features")
            .replace(" \"${ARGS[@]}\"", "");
        if !out.contains(&command) {
            out.push(command);
        }
    }
    out
}

fn close_bar_fallback_commands(text: &str) -> Vec<String> {
    let start = match text.find("## The close bar") {
        Some(at) => at,
        None => panic!("the prompt has no close bar section"),
    };
    let section = &text[start..];
    let section = match section[3..].find("\n## ") {
        Some(end) => &section[..end + 3],
        None => section,
    };
    let marker = "its constituents, each run and printed on its own";
    let from = match section.find(marker) {
        Some(at) => at,
        None => panic!(
            "the close bar names `bash crates/lights/scripts/run_tests.sh` as its \
             only test command and carries no fallback for a session that is \
             refused it. L-152 measured that a driven context IS refused it, \
             even `bash -n`, while every constituent cargo command runs; the PM \
             ruled on 2026-09-16 (option b) that the bar names those \
             constituents, so a context meets the substance of the bar and says \
             so instead of lowering it silently. Expected the phrase {marker:?}"
        ),
    };
    let mut out = Vec::new();
    for line in section[from..].lines().skip(1) {
        let trimmed = line.trim();
        if trimmed.starts_with("cargo ") && line.starts_with("      ") {
            out.push(trimmed.to_string());
        } else if !out.is_empty() && !trimmed.is_empty() {
            break;
        }
    }
    out
}

#[test]
fn the_close_bar_names_every_command_run_tests_sh_issues_as_its_fallback() {
    let script = run_tests_sh_commands();
    assert!(
        script.len() >= 2,
        "found {} cargo commands in run_tests.sh, so this scan read nothing: \
         {script:?}",
        script.len()
    );
    assert!(
        script.iter().any(|c| c == "cargo test -p lights --all-features"),
        "run_tests.sh no longer runs the --all-features suite, or this scan \
         stopped reading it: {script:?}"
    );
    let named = close_bar_fallback_commands(&prompt());
    for command in script.iter() {
        assert!(
            named.contains(command),
            "run_tests.sh issues `{command}` and the close bar's fallback does \
             not name it, so a context that cannot run the script meets a bar \
             lower than the one it reports. Named: {named:?}"
        );
    }
    for command in named.iter() {
        assert!(
            script.contains(command),
            "the close bar's fallback names `{command}` and run_tests.sh does \
             not issue it, so the fallback has drifted from the documented \
             command. The script issues: {script:?}"
        );
    }
}

fn head_header_literal(script: &str) -> String {
    let start = match script.find("HEAD_HEADER = (") {
        Some(at) => at,
        None => panic!("scripts/split_register.py no longer defines HEAD_HEADER"),
    };
    let body = &script[start..];
    let end = match body.find("\n)") {
        Some(at) => at,
        None => panic!("HEAD_HEADER in scripts/split_register.py is not closed"),
    };
    body[..end].to_string()
}

fn leading_comment_block(toml: &str) -> String {
    toml.lines()
        .take_while(|l| !l.starts_with("[[entry]]"))
        .filter(|l| l.starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn every_place_a_reader_starts_from_states_the_mission_and_the_surviving_target() {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let read = |p: &Path| match std::fs::read_to_string(p) {
        Ok(t) => t,
        Err(e) => panic!("could not read {}: {e}", p.display()),
    };
    let needle = "simple 3D game engine";
    let mission = "fulgor/nexus paradigm";
    let script = read(&crate_dir.join("scripts/split_register.py"));
    let places = [
        (
            "the register header split_register.py writes into head_open.toml",
            head_header_literal(&script),
        ),
        (
            "head_open.toml's own header",
            leading_comment_block(&read(&crate_dir.join("head_open.toml"))),
        ),
        ("docs/lights/BURNDOWN_LOOP.md", prompt()),
        (
            "docs/plan/TWO_HORIZON_PLAN.md",
            read(&repo_root().join("docs/plan/TWO_HORIZON_PLAN.md")),
        ),
    ];
    for (name, text) in places.iter() {
        assert!(
            text.contains("optical") && text.contains(needle) && text.contains(mission),
            "{name} does not state the mission ({mission}) and the surviving target (a \
             {needle}). PM, 2026-09-09 (L-120) ruled that the requirement is written \
             in every place a reader starts from; the register header lost it once \
             already, when BURNDOWN.toml was split into head_open.toml and tail.toml. \
             PM, 2026-09-29: the optical half of L-120 is withdrawn, so a reader must \
             be told the MISSION as well as the surviving target"
        );
    }
}

#[test]
fn a_long_running_host_rolls_with_the_header_on_disk_not_the_one_it_imported() {
    let scripts = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts");
    let code = "import sys; sys.path.insert(0, '.'); \
                import burndown_host as H, split_register as S; \
                S.HEAD_HEADER = '# a header imported before the fix landed'; \
                H.fresh_modules(); \
                print(S.HEAD_HEADER)";
    let out = std::process::Command::new("python3")
        .current_dir(&scripts)
        .arg("-c")
        .arg(code)
        .output()
        .unwrap_or_else(|e| panic!("python3 could not be run for burndown_host.fresh_modules: {e}"));
    assert!(
        out.status.success(),
        "burndown_host.fresh_modules exited {:?} with stderr {}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    let header = String::from_utf8_lossy(&out.stdout);
    assert!(
        header.contains("optical") && header.contains("simple 3D game engine"),
        "the host rolled head_open.toml with the header it imported at start-up rather than \
         the one on disk: {header}. On 2026-09-28 a host started at 02:18 rolled the register \
         at 13:05 with the pre-L-120 header and undid L-120's fix on disk"
    );
}
