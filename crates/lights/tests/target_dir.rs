use std::path::PathBuf;
use std::process::Command;

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn chooser() -> PathBuf {
    crate_root().join("scripts/target_dir.py")
}

fn choose(free_bytes: u64) -> String {
    let script = chooser();
    let out = Command::new("python3")
        .arg(&script)
        .arg("--configured")
        .arg("/scratch/cargo-target")
        .arg("--fallback")
        .arg("/on-disk/target")
        .arg("--free-bytes")
        .arg(free_bytes.to_string())
        .output()
        .unwrap_or_else(|e| {
            panic!(
                "scripts/target_dir.py is what keeps this crate compiling when the \
                 shared scratch volume fills, and it could not be run at {}: {e}",
                script.display()
            )
        });
    assert!(
        out.status.success(),
        "target_dir.py exited {:?} with stderr {}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

const GIB: u64 = 1024 * 1024 * 1024;

#[test]
fn a_full_scratch_volume_is_abandoned_for_the_on_disk_target() {
    assert_eq!(
        choose(GIB / 32),
        "/on-disk/target",
        "with 32 MiB free on the configured target dir the build cannot link, so \
         the chooser must hand back the on-disk fallback; three burn-down contexts \
         concluded the crate could not be compiled at all because nothing did this"
    );
}

#[test]
fn a_roomy_scratch_volume_is_kept() {
    assert_eq!(
        choose(20 * GIB),
        "/scratch/cargo-target",
        "the ram disk is the reason builds are fast here and the fallback is a \
         last resort, so room above the floor must keep the configured dir"
    );
}

#[test]
fn the_floor_is_crossed_in_the_right_direction() {
    let floor = 8 * GIB;
    assert_eq!(choose(floor - 1), "/on-disk/target");
    assert_eq!(choose(floor), "/scratch/cargo-target");
}

#[test]
fn run_tests_sh_consults_the_chooser_and_exports_what_it_says() {
    let path = crate_root().join("scripts/run_tests.sh");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{} could not be read: {e}", path.display()));
    assert!(
        text.contains("scripts/target_dir.py"),
        "run_tests.sh is the one documented test command and it must ask \
         scripts/target_dir.py where to build, or a full scratch volume makes \
         every close bar in the register unreachable"
    );
    assert!(
        text.contains("export CARGO_TARGET_DIR="),
        "asking the chooser is useless unless the answer is exported to cargo"
    );
    let ask = text.find("scripts/target_dir.py").unwrap();
    let export = text.find("export CARGO_TARGET_DIR=").unwrap();
    assert!(
        ask < export,
        "the chooser must be consulted before its answer is exported"
    );
    let first_cargo = text.find("cargo ").unwrap_or(usize::MAX);
    assert!(
        export < first_cargo,
        "CARGO_TARGET_DIR must be exported before the first cargo invocation, \
         or the detachability builds still land on the full volume"
    );
}

#[test]
fn the_chooser_path_run_tests_sh_names_exists_from_the_directory_it_changes_into() {
    let script = crate_root().join("scripts/run_tests.sh");
    let text = std::fs::read_to_string(&script)
        .unwrap_or_else(|e| panic!("{} could not be read: {e}", script.display()));
    let cd = text
        .lines()
        .find_map(|l| l.trim().strip_prefix("cd \"$(dirname \"$0\")/"))
        .and_then(|rest| rest.strip_suffix('"'))
        .unwrap_or_else(|| panic!("run_tests.sh no longer changes directory relative to itself"));
    let cwd = crate_root().join("scripts").join(cd);
    let named = text
        .split_whitespace()
        .find(|w| w.ends_with("target_dir.py"))
        .unwrap_or_else(|| panic!("run_tests.sh names no target_dir.py"));
    let resolved = cwd.join(named);
    assert!(
        resolved.is_file(),
        "run_tests.sh runs `python3 {named}` from {} and that is not a file, so the \
         chooser never answers and a full scratch volume is never routed around",
        cwd.display()
    );
    assert!(
        !text.contains("target_dir.py --explain || true"),
        "a chooser that cannot be found must stop the run, not be swallowed"
    );
}

fn default_pool(free_bytes: u64) -> String {
    let code = format!(
        "import sys; sys.path.insert(0, 'scripts'); import mutate_par; print(mutate_par.default_pool_root({free_bytes}))"
    );
    let out = Command::new("python3")
        .current_dir(crate_root())
        .arg("-c")
        .arg(&code)
        .output()
        .unwrap_or_else(|e| panic!("python3 could not be run for mutate_par.default_pool_root: {e}"));
    assert!(
        out.status.success(),
        "mutate_par.default_pool_root exited {:?} with stderr {}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

#[test]
fn the_mutation_pool_leaves_a_tight_scratch_volume_for_the_ssd() {
    assert_eq!(
        default_pool(6 * GIB + GIB / 5),
        "/tmp/mutpool",
        "with 6.2 GiB free on the 16 GiB scratch volume a 4- and then a 2-worker pool \
         died with ENOSPC on 2026-09-28, so the default pool must move to the SSD \
         below the shared floor exactly as the cargo target dir does"
    );
    assert_eq!(
        default_pool(20 * GIB),
        "/Volumes/RamBuild/mutpool",
        "room above the floor must keep the pool on the ram disk"
    );
}
