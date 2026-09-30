use std::path::PathBuf;
use std::process::Command;

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn python(script: &str, args: &[&str]) -> (bool, String) {
    let o = Command::new("python3")
        .arg(crate_root().join("scripts").join(script))
        .args(args)
        .current_dir(crate_root())
        .output()
        .unwrap_or_else(|e| panic!("python3 {script} did not start: {e}"));
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    );
    (o.status.success(), text)
}

#[test]
fn the_register_checker_reports_an_entry_whose_blocker_has_closed() {
    let (ok, text) = python("burndown_selftest.py", &[]);
    assert!(
        ok,
        "burndown_selftest.py failed. L-158: L-147 sat BLOCKED ON L-146 for thirteen days after \
         L-146 closed, and --next printed REGISTER DRY, because nothing resolved the id the \
         prose named. {text}"
    );
    assert!(
        text.contains("deliberate breaks, all caught"),
        "the selftest exited 0 without its summary line, so it may have tested nothing: {text}"
    );
}

#[test]
fn no_open_entry_in_the_live_register_waits_on_a_closed_blocker() {
    let (_, text) = python("burndown.py", &["--check"]);
    let stale: Vec<&str> = text
        .lines()
        .filter(|l| l.contains(" is blocked on ") && l.contains("unblock it"))
        .collect();
    assert!(
        stale.is_empty(),
        "an open entry is hidden from --next behind a blocker that is closed or absent: {stale:?}"
    );
}
