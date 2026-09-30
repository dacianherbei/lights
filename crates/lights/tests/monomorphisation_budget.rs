use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn symbol(method: &str, n: usize) -> Vec<u8> {
    format!(
        "_ZN6lights6method{}{method}7execute11render_tile17h{n:016x}E",
        method.len()
    )
    .into_bytes()
}

fn fake_binary(target: &Path, profile: &str, name: &str, splat: usize, raycast: usize) {
    let at = target.join(profile);
    std::fs::create_dir_all(&at).expect("the fake profile directory is created");
    let mut bytes = b"not-an-executable-format\0".to_vec();
    for (method, count) in [("splat", splat), ("raycast", raycast)] {
        for i in 0..count {
            bytes.extend(symbol(method, i));
            bytes.push(0);
        }
    }
    let path = at.join(name);
    std::fs::write(&path, bytes).expect("the fake binary is written");
    let status = Command::new("chmod")
        .arg("+x")
        .arg(&path)
        .status()
        .expect("chmod runs");
    assert!(status.success(), "chmod +x {} failed", path.display());
}

fn budget(target: &Path) -> (Option<i32>, String) {
    let out: Output = Command::new("python3")
        .arg(crate_root().join("scripts/monomorphisation_budget.py"))
        .arg("--target-dir")
        .arg(target)
        .output()
        .expect("python3 runs the budget script");
    (out.status.code(), String::from_utf8_lossy(&out.stdout).into_owned())
}

fn predicted_per_method() -> usize {
    let dir = tempfile::tempdir().expect("a scratch target directory");
    fake_binary(dir.path(), "debug", "lights-probe", 1, 0);
    let (_, text) = budget(dir.path());
    let line = text
        .lines()
        .find(|l| l.starts_with("source:"))
        .unwrap_or_else(|| panic!("the script printed no source line:\n{text}"));
    let n: usize = line
        .rsplit('=')
        .next()
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| panic!("no per-method prediction in {line:?}"));
    assert!(
        n > 0,
        "the source predicts no instantiation at all, so every check here would pass by testing nothing"
    );
    n
}

#[test]
fn a_release_binary_carrying_cross_crate_copies_does_not_move_the_verdict() {
    let n = predicted_per_method();
    let dir = tempfile::tempdir().expect("a scratch target directory");
    fake_binary(dir.path(), "debug", "lights-demo", n, n);
    fake_binary(dir.path(), "release", "lights-demo", n + 2, n + 2);
    let (code, text) = budget(dir.path());
    assert_eq!(
        code,
        Some(0),
        "a release binary whose crate re-instantiates the generic renderer carries copies of \
         instantiations the library already holds - release builds do not share generics across \
         crates - so it must not decide the verdict; the debug build of the same binary holds \
         exactly the predicted {n} per method:\n{text}"
    );
    assert!(
        text.contains(&format!(
            "MEASURED ceiling over 1 debug shipping binaries: {} ",
            2 * n
        )),
        "the ceiling must be read off the one debug shipping binary:\n{text}"
    );
}

#[test]
fn a_target_directory_holding_only_release_binaries_is_refused_rather_than_measured() {
    let n = predicted_per_method();
    let dir = tempfile::tempdir().expect("a scratch target directory");
    fake_binary(dir.path(), "release", "lights-agreement", n, n);
    let (code, text) = budget(dir.path());
    assert_eq!(
        code,
        Some(3),
        "a release count is what survived inlining plus what cross-crate copies added, so a \
         directory with no debug binary has nothing that measures instantiations:\n{text}"
    );
    assert!(
        text.contains("refusing: no debug shipping binary"),
        "the refusal must say why:\n{text}"
    );
}

#[test]
fn a_debug_binary_with_an_instantiation_the_source_does_not_predict_is_still_a_mismatch() {
    let n = predicted_per_method();
    let dir = tempfile::tempdir().expect("a scratch target directory");
    fake_binary(dir.path(), "debug", "lights-agreement", n + 1, n);
    let (code, text) = budget(dir.path());
    assert_eq!(
        code,
        Some(1),
        "an unpredicted instantiation in a debug shipping binary is the defect this script exists to \
         catch:\n{text}"
    );
    assert!(
        text.contains(&format!("MISMATCH splat: measured {}, predicted {n}", n + 1)),
        "the mismatch must name the method and both counts:\n{text}"
    );
}
