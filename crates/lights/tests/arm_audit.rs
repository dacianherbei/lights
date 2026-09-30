use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use lights::{KernelFamily, MethodId, Precision};

const VERDICTS: [&str; 3] = ["KEEP", "PRUNE", "RE-JUDGE"];
const EXPERIMENT_FIELDS: [&str; 5] = ["question", "arms", "criterion", "cost", "prunes"];
const STRATEGY_AXES: [&str; 3] = ["kernel/", "method/", "precision/"];

const READER: &str = r#"
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
def flat(s):
    return str(s).replace("\t", " ").replace("\n", " ").replace("\r", " ")
for r in d.get("records", []):
    print("record\t%s\t%s" % (r.get("slug", ""), flat(r.get("command", ""))))
for a in d.get("arms", []):
    key = a.get("arm", "")
    print("arm\t%s\t%s\t%s\t%s" % (key, a.get("verdict", ""), a.get("provenance", ""), a.get("transfers", "")))
    for w in a.get("where", []):
        print("where\t%s\t%s" % (key, w))
    for s in a.get("records", []):
        print("cites\t%s\t%s" % (key, s))
    for k, v in (a.get("experiment") or {}).items():
        print("experiment\t%s\t%s\t%s" % (key, k, flat(v)))
"#;

struct Arm {
    key: String,
    verdict: String,
    provenance: String,
    transfers: String,
    records: Vec<String>,
    places: Vec<String>,
    experiment: BTreeMap<String, String>,
}

struct Table {
    arms: Vec<Arm>,
    commands: BTreeMap<String, String>,
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("the repository root must resolve from the crate manifest")
}

fn arm_named<'a>(arms: &'a mut [Arm], key: &str) -> &'a mut Arm {
    arms.iter_mut()
        .find(|a| a.key == key)
        .unwrap_or_else(|| panic!("the reader named {key} before printing its row"))
}

fn table() -> Table {
    let record = repo_root().join("docs/lights/RESEARCH.json");
    let out = Command::new("python3")
        .arg("-c")
        .arg(READER)
        .arg(&record)
        .output()
        .unwrap_or_else(|e| panic!("python3 could not read {}: {e}", record.display()));
    assert!(
        out.status.success(),
        "{} did not parse: {}",
        record.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8(out.stdout).expect("the reader prints utf-8");
    let mut arms: Vec<Arm> = Vec::new();
    let mut commands = BTreeMap::new();
    for line in text.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        match f[0] {
            "record" => {
                commands.insert(f[1].to_string(), f[2].to_string());
            }
            "arm" => arms.push(Arm {
                key: f[1].to_string(),
                verdict: f[2].to_string(),
                provenance: f[3].to_string(),
                transfers: f[4].to_string(),
                records: Vec::new(),
                places: Vec::new(),
                experiment: BTreeMap::new(),
            }),
            "where" => arm_named(&mut arms, f[1]).places.push(f[2].to_string()),
            "cites" => arm_named(&mut arms, f[1]).records.push(f[2].to_string()),
            "experiment" => {
                arm_named(&mut arms, f[1])
                    .experiment
                    .insert(f[2].to_string(), f[3].to_string());
            }
            other => panic!("the reader printed a line kind this test does not define: {other}"),
        }
    }
    Table { arms, commands }
}

fn declared_strategy_arms() -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for k in KernelFamily::ALL {
        out.insert(format!("kernel/{}", k.kernel_name()));
    }
    for m in MethodId::ALL {
        out.insert(format!("method/{}", m.name()));
    }
    for p in Precision::ALL {
        out.insert(format!("precision/{}", p.name()));
    }
    out
}

fn bin_paths() -> BTreeMap<String, String> {
    let manifest = std::fs::read_to_string(repo_root().join("crates/lights/Cargo.toml"))
        .expect("crates/lights/Cargo.toml must be readable");
    let mut out = BTreeMap::new();
    let mut in_bin = false;
    let mut name: Option<String> = None;
    for line in manifest.lines().map(str::trim) {
        if line.starts_with('[') {
            in_bin = line == "[[bin]]";
            name = None;
            continue;
        }
        if !in_bin {
            continue;
        }
        let value = line.split_once('=').map(|(_, v)| v.trim().trim_matches('"').to_string());
        if line.starts_with("name") {
            name = value;
        } else if line.starts_with("path") {
            if let (Some(n), Some(p)) = (name.clone(), value) {
                out.insert(format!("crates/lights/{p}"), n);
            }
        }
    }
    out
}

fn target_of(place: &str, bins: &BTreeMap<String, String>) -> Option<String> {
    let stem = |prefix: &str| {
        place
            .strip_prefix(prefix)
            .and_then(|rest| rest.strip_suffix(".rs"))
            .filter(|rest| !rest.contains('/'))
            .map(str::to_string)
    };
    if let Some(name) = bins.get(place) {
        return Some(format!("bin:{name}"));
    }
    if let Some(t) = stem("crates/lights/tests/") {
        return Some(format!("test:{t}"));
    }
    stem("crates/lights/examples/").map(|e| format!("example:{e}"))
}

fn targets_named_by(command: &str) -> BTreeSet<String> {
    let tokens: Vec<&str> = command.split_whitespace().collect();
    let mut out = BTreeSet::new();
    for pair in tokens.windows(2) {
        let kind = match pair[0] {
            "--test" => "test",
            "--bin" => "bin",
            "--example" => "example",
            _ => continue,
        };
        let name = pair[1].trim_matches(|c: char| c == ';' || c == '\'' || c == '"');
        out.insert(format!("{kind}:{name}"));
    }
    out
}

#[test]
fn every_strategy_arm_in_the_code_has_exactly_one_row() {
    let t = table();
    let declared = declared_strategy_arms();
    for axis in STRATEGY_AXES {
        assert!(
            declared.iter().any(|d| d.starts_with(axis)),
            "the code declares no {axis} arm, so this coverage check would pass by testing nothing"
        );
    }
    let mut rows: BTreeMap<&str, usize> = BTreeMap::new();
    for a in t.arms.iter().filter(|a| STRATEGY_AXES.iter().any(|x| a.key.starts_with(x))) {
        *rows.entry(a.key.as_str()).or_default() += 1;
    }
    let missing: Vec<&str> = declared
        .iter()
        .map(String::as_str)
        .filter(|d| !rows.contains_key(d))
        .collect();
    let phantom: Vec<&str> = rows.keys().copied().filter(|k| !declared.contains(*k)).collect();
    let twice: Vec<&str> = rows.iter().filter(|(_, n)| **n > 1).map(|(k, _)| *k).collect();
    assert!(
        missing.is_empty() && phantom.is_empty() && twice.is_empty(),
        "docs/lights/RESEARCH.json `arms` must give every kernel, method and precision the code \
         declares exactly one verdict. declared by the code and absent from the table: {missing:?}; \
         in the table and declared nowhere: {phantom:?}; listed more than once: {twice:?}"
    );
}

#[test]
fn the_audit_never_proposes_fewer_than_two_method_arms() {
    let t = table();
    let methods: Vec<&Arm> = t.arms.iter().filter(|a| a.key.starts_with("method/")).collect();
    assert_eq!(
        methods.len(),
        MethodId::ALL.len(),
        "every method arm needs a row before its verdict can be checked"
    );
    let pruned: Vec<&str> = methods
        .iter()
        .filter(|a| a.verdict == "PRUNE")
        .map(|a| a.key.as_str())
        .collect();
    let kept = methods.iter().filter(|a| a.verdict == "KEEP").count();
    assert!(
        pruned.is_empty() && kept >= 2,
        "the agreement contract between two independently written methods IS the paradigm case in \
         miniature, and L-198 forbids an audit that cuts below two method arms: pruned {pruned:?}, \
         kept {kept} of {}",
        methods.len()
    );
}

#[test]
fn no_arm_is_pruned_on_evidence_that_does_not_transfer() {
    let t = table();
    let pruned: Vec<&Arm> = t.arms.iter().filter(|a| a.verdict == "PRUNE").collect();
    assert!(
        !pruned.is_empty(),
        "the table prunes nothing, so the transfer rule would pass by testing nothing"
    );
    let offending: Vec<String> = pruned
        .iter()
        .filter(|a| a.transfers != "yes")
        .map(|a| format!("{} (provenance {}, transfers {})", a.key, a.provenance, a.transfers))
        .collect();
    assert!(
        offending.is_empty(),
        "a PRUNE must rest on evidence that carries to the surviving mission; an arm that lost on a \
         target that no longer exists is RE-JUDGE, never PRUNE: {offending:?}"
    );
}

#[test]
fn every_command_a_prune_would_delete_is_named_by_it() {
    let t = table();
    let bins = bin_paths();
    assert!(
        bins.values().any(|b| b == "lights-demo"),
        "the manifest reader found no [[bin]] lights-demo, so it maps no binary at all: {bins:?}"
    );
    let mut checked = 0usize;
    let mut unnamed = Vec::new();
    for a in t.arms.iter().filter(|a| a.verdict == "PRUNE") {
        for place in &a.places {
            let Some(target) = target_of(place, &bins) else {
                continue;
            };
            for (slug, command) in &t.commands {
                if !targets_named_by(command).contains(&target) {
                    continue;
                }
                checked += 1;
                if !a.records.contains(slug) {
                    unnamed.push(format!("{} deletes {target} ({place}), which {slug}'s command runs", a.key));
                }
            }
        }
    }
    assert!(
        checked > 0,
        "no PRUNE deletes a target any record's command runs, so this rule tested nothing"
    );
    assert!(
        unnamed.is_empty(),
        "L-198 may not propose a deletion that leaves a RESEARCH.json command unable to run, so a \
         PRUNE must name every record whose command runs what it deletes; L-200 then pins each of \
         those commands to the commit before the removal: {unnamed:?}"
    );
}

#[test]
fn every_rejudge_states_the_experiment_that_would_decide_it() {
    let t = table();
    let rejudged: Vec<&Arm> = t.arms.iter().filter(|a| a.verdict == "RE-JUDGE").collect();
    assert!(
        !rejudged.is_empty(),
        "the table re-judges nothing, so this rule would pass by testing nothing"
    );
    let mut silent = Vec::new();
    for a in rejudged {
        for field in EXPERIMENT_FIELDS {
            let stated = a.experiment.get(field).is_some_and(|v| !v.trim().is_empty());
            if !stated {
                silent.push(format!("{} states no {field}", a.key));
            }
        }
    }
    assert!(
        silent.is_empty(),
        "an audit that finds a question needing an experiment states the question, the arms, the \
         criterion, the cost and what it would prune, and stops: {silent:?}"
    );
}

#[test]
fn the_table_cites_only_paths_and_records_that_exist() {
    let t = table();
    assert!(!t.arms.is_empty(), "docs/lights/RESEARCH.json carries no `arms` table");
    let root = repo_root();
    let mut wrong = Vec::new();
    for a in &t.arms {
        if !VERDICTS.contains(&a.verdict.as_str()) {
            wrong.push(format!("{} has verdict {}", a.key, a.verdict));
        }
        if a.places.is_empty() {
            wrong.push(format!("{} names no code", a.key));
        }
        for place in &a.places {
            if !root.join(place).exists() {
                wrong.push(format!("{} names {place}, which is not in the tree", a.key));
            }
        }
        for slug in &a.records {
            if !t.commands.contains_key(slug) {
                wrong.push(format!("{} cites {slug}, which is not a record", a.key));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "a verdict about code that is gone, or backed by a record that is not there, is a sentence \
         that WAS true; L-200 deletes files, so it amends this table in the same commit: {wrong:?}"
    );
}
