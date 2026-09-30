use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

const SECTIONS: [&str; 6] = ["question", "measured", "paradigm", "comparison", "stands", "open"];
const UI_VERDICT: [&str; 3] = [
    "l174-optical-egui-baseline",
    "l175-drawlist-ui-frame",
    "l176-renderer-arms",
];
const CONTRACT: [&str; 2] = ["arm:method/splat-raster", "arm:method/ray-cast"];
const OUTPUT_FLAGS: [&str; 5] = ["--out-dir", "--out", "--ops-out", "--json-out", ">"];
const SEGMENT_ENDS: [&str; 6] = ["&&", ";", "|", "do", "done", "then"];

const READER: &str = r#"
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
def flat(s):
    return str(s).replace("\t", " ").replace("\n", " ").replace("\r", " ")
for r in d.get("records", []):
    print("record\t%s\t%s" % (r.get("slug", ""), flat(r.get("command", ""))))
    print("text\t%s\t%s" % (r.get("slug", ""), flat(json.dumps(r, ensure_ascii=False))))
for a in d.get("arms", []):
    print("text\tarm:%s\t%s" % (a.get("arm", ""), flat(json.dumps(a, ensure_ascii=False))))
rep = d.get("report") or {}
for s in rep.get("sections", []):
    print("section\t%s\t%s" % (s.get("key", ""), flat(s.get("heading", ""))))
    for i, c in enumerate(s.get("claims", [])):
        print("claim\t%s\t%d\t%s" % (s.get("key", ""), i, flat(c.get("text", ""))))
        for x in c.get("cites", []):
            print("cite\t%s\t%d\t%s" % (s.get("key", ""), i, x))
for i, r in enumerate(rep.get("reproduced", [])):
    print("repro\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % (i, r.get("record", ""), flat(r.get("invocation", "")),
          flat(r.get("ran") or ""), r.get("how", ""), r.get("exit"), r.get("expected"), r.get("at") or ""))
"#;

const RENDER: &str = r####"
import sys
sys.path.insert(0, sys.argv[1])
import research
doc = research.load()
md, page = research.markdown(doc), research.report(doc)
rep = doc.get("report") or {}
print("front\t%d\t%d" % (md.find("## " + rep.get("title", "\u0000")), md.find("## representation")))
for s in rep.get("sections", []):
    print("heading\t%s\t%d\t%d" % (s.get("key", ""), ("### " + s.get("heading", "")) in md,
          ('data-id="report--%s"' % s.get("key", "")) in page and ('id="report--%s"' % s.get("key", "")) in page))
    for i, c in enumerate(s.get("claims", [])):
        print("claimtext\t%s\t%d\t%d" % (s.get("key", ""), i, c.get("text", "\u0000") in md))
print("ledger\t%d\t%d" % ("### How each cited experiment was reproduced" in md, 'id="report--reproduced"' in page))
"####;

struct Claim {
    section: String,
    text: String,
    cites: Vec<String>,
}

struct Repro {
    record: String,
    invocation: String,
    ran: String,
    how: String,
    exit: String,
    expected: String,
    at: String,
}

struct Report {
    sections: Vec<(String, String)>,
    claims: Vec<Claim>,
    repros: Vec<Repro>,
    commands: BTreeMap<String, String>,
    texts: BTreeMap<String, String>,
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("the repository root must resolve from the crate manifest")
}

fn python(script: &str, arg: &Path) -> String {
    let out = Command::new("python3")
        .arg("-c")
        .arg(script)
        .arg(arg)
        .current_dir(repo_root())
        .output()
        .unwrap_or_else(|e| panic!("python3 could not run: {e}"));
    assert!(
        out.status.success(),
        "python3 failed on {}: {}",
        arg.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("python prints utf-8")
}

fn report() -> Report {
    let text = python(READER, &repo_root().join("docs/lights/RESEARCH.json"));
    let mut r = Report {
        sections: Vec::new(),
        claims: Vec::new(),
        repros: Vec::new(),
        commands: BTreeMap::new(),
        texts: BTreeMap::new(),
    };
    for line in text.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        match f[0] {
            "record" => {
                r.commands.insert(f[1].to_string(), f[2].to_string());
            }
            "text" => {
                r.texts.insert(f[1].to_string(), f[2].to_string());
            }
            "section" => r.sections.push((f[1].to_string(), f[2].to_string())),
            "claim" => r.claims.push(Claim {
                section: f[1].to_string(),
                text: f[3].to_string(),
                cites: Vec::new(),
            }),
            "cite" => {
                let at = r
                    .claims
                    .iter()
                    .rposition(|c| c.section == f[1])
                    .unwrap_or_else(|| panic!("a cite printed before its claim: {line}"));
                r.claims[at].cites.push(f[3].to_string());
            }
            "repro" => r.repros.push(Repro {
                record: f[2].to_string(),
                invocation: f[3].to_string(),
                ran: f[4].to_string(),
                how: f[5].to_string(),
                exit: f[6].to_string(),
                expected: f[7].to_string(),
                at: f[8].to_string(),
            }),
            other => panic!("the reader printed a line kind this test does not define: {other}"),
        }
    }
    r
}

fn squash(s: &str) -> String {
    s.split_whitespace()
        .filter(|t| *t != "\\")
        .collect::<Vec<_>>()
        .join(" ")
}

fn decimals(s: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for token in s.split(|c: char| !(c.is_ascii_digit() || c == '.')) {
        let whole: String = token.chars().take_while(char::is_ascii_digit).collect();
        if whole.is_empty() || !token[whole.len()..].starts_with('.') {
            continue;
        }
        let frac: String = token[whole.len() + 1..].chars().take_while(char::is_ascii_digit).collect();
        if !frac.is_empty() {
            out.insert(format!("{whole}.{frac}"));
        }
    }
    out
}

fn cited_records(r: &Report) -> BTreeSet<String> {
    r.claims
        .iter()
        .flat_map(|c| c.cites.iter())
        .filter(|x| !x.starts_with("arm:"))
        .cloned()
        .collect()
}

fn placeholder_match(pattern: &str, s: &str) -> bool {
    let mut literals: Vec<&str> = Vec::new();
    let mut rest = pattern;
    for _ in 0..pattern.len() {
        match (rest.find('<'), rest.find('>')) {
            (Some(a), Some(b)) if a < b => {
                literals.push(&rest[..a]);
                rest = &rest[b + 1..];
            }
            _ => break,
        }
    }
    literals.push(rest);
    if literals.len() == 1 {
        return pattern == s;
    }
    let (first, last) = (literals[0], literals[literals.len() - 1]);
    if !s.starts_with(first) || !s.ends_with(last) || s.len() < first.len() + last.len() {
        return false;
    }
    let mut middle = &s[first.len()..s.len() - last.len()];
    for lit in &literals[1..literals.len() - 1] {
        match middle.find(lit) {
            Some(at) => middle = &middle[at + lit.len()..],
            None => return false,
        }
    }
    true
}

fn same_run(invocation: &str, ran: &str) -> Result<(), String> {
    let a: Vec<&str> = invocation.split_whitespace().filter(|t| *t != "\\").collect();
    let b: Vec<&str> = ran.split_whitespace().filter(|t| *t != "\\").collect();
    if a.len() != b.len() {
        return Err(format!("{} tokens recorded against {} run", a.len(), b.len()));
    }
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        let output = i > 0 && OUTPUT_FLAGS.contains(&a[i - 1]);
        if x != y && !output && !placeholder_match(x, y) {
            return Err(format!("token {i}: recorded {x:?}, run {y:?}"));
        }
    }
    Ok(())
}

enum Tree {
    Work(PathBuf),
    At(String),
}

impl Tree {
    fn of(at: &str) -> Tree {
        if at.is_empty() {
            Tree::Work(repo_root())
        } else {
            Tree::At(at.to_string())
        }
    }

    fn git(args: &[&str]) -> Option<String> {
        let out = Command::new("git").args(args).current_dir(repo_root()).output().ok()?;
        out.status.success().then(|| String::from_utf8_lossy(&out.stdout).into_owned())
    }

    fn read(&self, rel: &str) -> Option<String> {
        match self {
            Tree::Work(root) => std::fs::read_to_string(root.join(rel)).ok(),
            Tree::At(rev) => Tree::git(&["show", &format!("{rev}:{rel}")]),
        }
    }

    fn exists(&self, rel: &str) -> bool {
        match self {
            Tree::Work(root) => root.join(rel).exists(),
            Tree::At(rev) => Tree::git(&["cat-file", "-e", &format!("{rev}:{rel}")]).is_some(),
        }
    }

    fn list(&self, dir: &str) -> Vec<String> {
        match self {
            Tree::Work(root) => std::fs::read_dir(root.join(dir))
                .map(|d| d.flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect())
                .unwrap_or_default(),
            Tree::At(rev) => Tree::git(&["ls-tree", "--name-only", rev, &format!("{dir}/")])
                .map(|s| s.lines().map(|l| l.rsplit('/').next().unwrap_or(l).to_string()).collect())
                .unwrap_or_default(),
        }
    }

    fn mentions(&self, dir: &str, needle: &str) -> bool {
        match self {
            Tree::Work(root) => contains_under(&root.join(dir), needle),
            Tree::At(rev) => Tree::git(&["grep", "-q", "-F", "-e", needle, rev, "--", dir]).is_some(),
        }
    }
}

fn contains_under(path: &Path, needle: &str) -> bool {
    if path.is_file() {
        return std::fs::read_to_string(path).is_ok_and(|t| t.contains(needle));
    }
    let Ok(entries) = std::fs::read_dir(path) else {
        return false;
    };
    entries.flatten().any(|e| {
        let p = e.path();
        let name = e.file_name().to_string_lossy().into_owned();
        !name.starts_with('.') && name != "target" && contains_under(&p, needle)
    })
}

struct Package {
    dir: String,
    manifest: String,
}

impl Package {
    fn at(tree: &Tree, manifest_path: &str) -> Option<Package> {
        let manifest = tree.read(manifest_path)?;
        let dir = Path::new(manifest_path).parent()?.to_string_lossy().into_owned();
        Some(Package { dir, manifest })
    }

    fn all(tree: &Tree) -> Vec<Package> {
        tree.list("crates")
            .iter()
            .filter_map(|e| Package::at(tree, &format!("crates/{e}/Cargo.toml")))
            .collect()
    }

    fn named(tree: &Tree, name: &str) -> Option<Package> {
        Package::all(tree)
            .into_iter()
            .find(|p| package_name(&p.manifest).as_deref() == Some(name))
    }

    fn declared(&self, table: &str) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        let mut inside = false;
        for line in self.manifest.lines().map(str::trim) {
            if line.starts_with('[') {
                inside = line == format!("[[{table}]]");
                continue;
            }
            if inside && line.starts_with("name") {
                if let Some((_, v)) = line.split_once('=') {
                    out.insert(v.trim().trim_matches('"').to_string());
                }
            }
        }
        out
    }

    fn has_bin(&self, tree: &Tree, name: &str) -> bool {
        self.declared("bin").contains(name)
            || tree.exists(&format!("{}/src/bin/{name}.rs", self.dir))
            || (package_name(&self.manifest).as_deref() == Some(name)
                && tree.exists(&format!("{}/src/main.rs", self.dir)))
    }

    fn has_test(&self, tree: &Tree, name: &str) -> bool {
        tree.exists(&format!("{}/tests/{name}.rs", self.dir)) || self.declared("test").contains(name)
    }

    fn has_example(&self, tree: &Tree, name: &str) -> bool {
        tree.exists(&format!("{}/examples/{name}.rs", self.dir)) || self.declared("example").contains(name)
    }

    fn has_feature(&self, name: &str) -> bool {
        let mut inside = false;
        for line in self.manifest.lines().map(str::trim) {
            if line.starts_with('[') {
                inside = line == "[features]";
                continue;
            }
            if inside && line.split('=').next().map(str::trim) == Some(name) {
                return true;
            }
        }
        false
    }

    fn test_fns(&self, tree: &Tree, tests: &[String]) -> BTreeSet<String> {
        let files: Vec<String> = if tests.is_empty() {
            tree.list(&format!("{}/tests", self.dir))
                .into_iter()
                .filter(|n| n.ends_with(".rs"))
                .map(|n| format!("{}/tests/{n}", self.dir))
                .collect()
        } else {
            tests.iter().map(|t| format!("{}/tests/{t}.rs", self.dir)).collect()
        };
        let mut out = BTreeSet::new();
        for f in files {
            let text = tree.read(&f).unwrap_or_default();
            for piece in text.split("fn ").skip(1) {
                let name: String = piece.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
                if !name.is_empty() && piece[name.len()..].trim_start().starts_with(['(', '<']) {
                    out.insert(name);
                }
            }
        }
        out
    }
}

fn package_name(manifest: &str) -> Option<String> {
    let mut inside = false;
    for line in manifest.lines().map(str::trim) {
        if line.starts_with('[') {
            inside = line == "[package]";
            continue;
        }
        if inside && line.starts_with("name") {
            return line.split_once('=').map(|(_, v)| v.trim().trim_matches('"').to_string());
        }
    }
    None
}

fn clean(t: &str) -> &str {
    t.trim_matches(|c: char| c == '\'' || c == '"' || c == '`' || c == ';')
}

fn flags_after(tokens: &[&str]) -> Vec<String> {
    let mut out = Vec::new();
    for raw in tokens {
        if SEGMENT_ENDS.contains(raw) {
            break;
        }
        let t = clean(raw);
        if t.starts_with("--") && t.len() > 2 && !t.contains('=') {
            out.push(t.to_string());
        }
        if raw.ends_with(';') {
            break;
        }
    }
    out
}

fn resolve(invocation: &str, at: &str) -> Vec<String> {
    let tree = Tree::of(at);
    let tokens: Vec<&str> = invocation.split_whitespace().collect();
    let mut problems = Vec::new();
    let mut i = 0;
    let n = tokens.len();
    for _ in 0..n {
        if i >= n {
            break;
        }
        let t = clean(tokens[i]);
        if t == "python3" && i + 1 < n {
            let script = clean(tokens[i + 1]);
            if !script.starts_with('-') {
                if !tree.exists(script) {
                    problems.push(format!("python3 runs {script}, which does not exist"));
                } else {
                    for flag in flags_after(&tokens[i + 2..]) {
                        if !tree.mentions(script, &format!("\"{flag}\"")) {
                            problems.push(format!("{script} is passed {flag}, which it does not parse"));
                        }
                    }
                }
            }
            i += 2;
            continue;
        }
        if t != "cargo" {
            i += 1;
            continue;
        }
        let sub = tokens.get(i + 1).map(|s| clean(s)).unwrap_or("");
        let mut j = i + 2;
        let (mut package, mut manifest) = (None::<String>, None::<String>);
        let (mut bins, mut tests, mut examples, mut features) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut harness, mut exact, mut filters, mut inputs, mut program) =
            (false, false, Vec::new(), Vec::new(), Vec::new());
        for _ in j..n {
            if j >= n {
                break;
            }
            let raw = tokens[j];
            let tok = clean(raw);
            if SEGMENT_ENDS.contains(&raw) {
                break;
            }
            let value = tokens.get(j + 1).map(|s| clean(s).to_string());
            let mut step = 1;
            if harness {
                match tok {
                    "--test-threads" if sub == "test" => step = 2,
                    "--exact" if sub == "test" => exact = true,
                    "--ops" => {
                        inputs.extend(value.clone());
                        program.push(tok.to_string());
                        step = 2;
                    }
                    t if t.starts_with("--") && sub == "run" && t.len() > 2 && !t.contains('=') => {
                        program.push(t.to_string())
                    }
                    t if t.starts_with('-') => {}
                    t if sub == "test" && !t.contains('/') && !t.contains('$') => filters.push(t.to_string()),
                    _ => {}
                }
            } else {
                match tok {
                    "-p" | "--package" => {
                        package = value.clone();
                        step = 2;
                    }
                    "--manifest-path" => {
                        manifest = value.clone();
                        step = 2;
                    }
                    "--bin" => {
                        bins.extend(value.clone());
                        step = 2;
                    }
                    "--test" => {
                        tests.extend(value.clone());
                        step = 2;
                    }
                    "--example" => {
                        examples.extend(value.clone());
                        step = 2;
                    }
                    "--features" => {
                        features.extend(value.iter().flat_map(|v| v.split(',').map(str::to_string)));
                        step = 2;
                    }
                    "--target-dir" | "--config" | "--profile" => step = 2,
                    "--" => harness = true,
                    _ => {}
                }
            }
            let ends = raw.ends_with(';');
            j += step;
            if ends {
                break;
            }
        }
        let candidates: Vec<Package> = match (&package, &manifest) {
            (_, Some(m)) => Package::at(&tree, m).into_iter().collect(),
            (Some(p), None) => Package::named(&tree, p).into_iter().collect(),
            (None, None) => Package::all(&tree),
        };
        if candidates.is_empty() {
            problems.push(format!(
                "cargo {sub} names package {package:?} / manifest {manifest:?}, which the tree does not hold"
            ));
        }
        let fits = |p: &Package| {
            bins.iter().all(|b| p.has_bin(&tree, b))
                && tests.iter().all(|t| p.has_test(&tree, t))
                && examples.iter().all(|e| p.has_example(&tree, e))
        };
        match candidates.iter().find(|&p| fits(p)) {
            None if !candidates.is_empty() => problems.push(format!(
                "cargo {sub} names bins {bins:?}, tests {tests:?}, examples {examples:?}, and no package \
                 it could mean holds them all"
            )),
            None => {}
            Some(p) => {
                for f in &features {
                    if !p.has_feature(f) {
                        problems.push(format!("feature {f} is not declared by {}", p.dir));
                    }
                }
                let fns = p.test_fns(&tree, &tests);
                for f in &filters {
                    let found = if exact { fns.contains(f) } else { fns.iter().any(|x| x.contains(f.as_str())) };
                    if !found {
                        problems.push(format!("the test filter {f} (exact: {exact}) matches no test function in {tests:?}"));
                    }
                }
                for input in &inputs {
                    if !tree.exists(input) {
                        problems.push(format!("the input {input} does not exist"));
                    }
                }
                for flag in &program {
                    if !tree.mentions(&p.dir, &format!("\"{flag}\"")) {
                        problems.push(format!(
                            "cargo {sub} passes {flag} to a program in {} that does not parse it",
                            p.dir
                        ));
                    }
                }
            }
        }
        i = j;
    }
    problems
}

#[test]
fn the_report_answers_what_it_was_written_to_answer_in_order() {
    let r = report();
    let keys: Vec<&str> = r.sections.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(
        keys, SECTIONS,
        "docs/lights/RESEARCH.json `report` must answer, in this order: what lights set out to prove, \
         what it measured, what the measurements say about the paradigm, the comparison, what stands, \
         and what goes or waits"
    );
    for key in SECTIONS {
        let claims: Vec<&Claim> = r.claims.iter().filter(|c| c.section == key).collect();
        assert!(!claims.is_empty(), "section {key} makes no claim");
        for c in claims {
            assert!(
                !c.cites.is_empty(),
                "a claim in {key} cites nothing, so nobody can trace it: {:?}",
                &c.text[..c.text.len().min(80)]
            );
        }
    }
}

#[test]
fn every_citation_names_a_record_or_an_arm_row() {
    let r = report();
    assert!(!r.claims.is_empty(), "the report makes no claims, so this check would pass by testing nothing");
    let mut ghosts = Vec::new();
    for c in &r.claims {
        for x in &c.cites {
            if !r.texts.contains_key(x) {
                ghosts.push(format!("{} cites {x}", c.section));
            }
        }
    }
    assert!(ghosts.is_empty(), "these citations name nothing in the record: {ghosts:?}");
}

#[test]
fn every_figure_a_claim_quotes_is_in_a_record_it_cites() {
    let r = report();
    let mut quoted = 0;
    let mut stray = Vec::new();
    for c in &r.claims {
        let pool: BTreeSet<String> = c
            .cites
            .iter()
            .filter_map(|x| r.texts.get(x))
            .flat_map(|t| decimals(t))
            .collect();
        for d in decimals(&c.text) {
            quoted += 1;
            if !pool.contains(&d) {
                stray.push(format!("{}: {d}", c.section));
            }
        }
    }
    assert!(quoted > 0, "the report quotes no figure at all, so this check would pass by testing nothing");
    assert!(
        stray.is_empty(),
        "the report is written from the record, so every figure it quotes must appear in a record or arm \
         row the same claim cites; these appear in none of them: {stray:?}"
    );
}

#[test]
fn the_comparison_cites_what_cost_lights_the_ui_target_and_what_stands_cites_both_methods() {
    let r = report();
    let cites_of = |key: &str| -> BTreeSet<String> {
        r.claims.iter().filter(|c| c.section == key).flat_map(|c| c.cites.iter().cloned()).collect()
    };
    let comparison = cites_of("comparison");
    let missing: Vec<&str> = UI_VERDICT.iter().copied().filter(|x| !comparison.contains(*x)).collect();
    assert!(
        missing.is_empty(),
        "the comparison must stand on the three measurements that withdrew the optical target: {missing:?}"
    );
    let stands = cites_of("stands");
    let absent: Vec<&str> = CONTRACT.iter().copied().filter(|x| !stands.contains(*x)).collect();
    assert!(
        absent.is_empty(),
        "what stands must cite both method arms, which the agreement contract is made of: {absent:?}"
    );
}

#[test]
fn every_cited_record_was_reproduced_from_its_own_command() {
    let r = report();
    let cited = cited_records(&r);
    assert!(!cited.is_empty(), "the report cites no record, so this check would pass by testing nothing");
    let mut problems = Vec::new();
    for slug in &cited {
        if !r.repros.iter().any(|p| &p.record == slug) {
            problems.push(format!("{slug} is cited and was not reproduced"));
        }
    }
    for p in &r.repros {
        let Some(command) = r.commands.get(&p.record) else {
            problems.push(format!("a reproduction names {}, which is not a record", p.record));
            continue;
        };
        if !squash(command).contains(&squash(&p.invocation)) {
            problems.push(format!("{}: {:?} is not in its record's own command", p.record, p.invocation));
        }
        if !p.ran.is_empty() {
            if let Err(why) = same_run(&p.invocation, &p.ran) {
                problems.push(format!("{}: what was run is not the recorded invocation ({why})", p.record));
            }
        }
        if p.how != "full" && p.how != "form" {
            problems.push(format!("{}: how {:?} is neither full nor form", p.record, p.how));
        }
        if p.exit != p.expected {
            problems.push(format!("{}: exited {} where its record expects {}", p.record, p.exit, p.expected));
        }
    }
    assert!(problems.is_empty(), "{problems:#?}");
}

#[test]
fn every_reproduced_invocation_resolves_on_the_tree() {
    let r = report();
    let cargo_rows = r.repros.iter().filter(|p| p.invocation.contains("cargo ")).count();
    assert!(cargo_rows > 0, "no reproduction runs cargo, so this check would pass by testing nothing");
    let history = Tree::git(&["rev-parse", "--git-dir"]).is_some();
    let mut problems = Vec::new();
    for p in &r.repros {
        if !p.at.is_empty() && !history {
            println!("{}: pinned to {} and this checkout holds no history, so it is resolved only in one that does", p.record, p.at);
            continue;
        }
        for why in resolve(&p.invocation, &p.at) {
            let at = if p.at.is_empty() { "the working tree".to_string() } else { format!("commit {}", p.at) };
            problems.push(format!("{} at {at}: {why}", p.record));
        }
    }
    assert!(
        problems.is_empty(),
        "a record whose command names a target the tree no longer holds is not a record; amend the record \
         in the commit that removes the target: {problems:#?}"
    );
}

#[test]
fn the_report_renders_at_the_front_of_both_generated_forms() {
    let r = report();
    let out = python(RENDER, &repo_root().join("crates/lights/scripts"));
    let mut headings = 0;
    for line in out.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        match f[0] {
            "front" => {
                let (report_at, records_at): (i64, i64) = (f[1].parse().unwrap(), f[2].parse().unwrap());
                assert!(
                    report_at >= 0 && (records_at < 0 || report_at < records_at),
                    "RESEARCH.md must open on the report, above the records: report at {report_at}, records at {records_at}"
                );
            }
            "heading" => {
                headings += 1;
                assert_eq!((f[2], f[3]), ("1", "1"), "section {} is missing from the markdown or the page", f[1]);
            }
            "claimtext" => assert_eq!(f[3], "1", "claim {} of {} is not in the markdown", f[2], f[1]),
            "ledger" => assert_eq!((f[1], f[2]), ("1", "1"), "the reproduction ledger is not rendered"),
            other => panic!("the renderer printed a line kind this test does not define: {other}"),
        }
    }
    assert_eq!(headings, r.sections.len(), "every section must be rendered, and this counted {headings}");
}
