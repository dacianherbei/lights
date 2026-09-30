import importlib.util
import pathlib
import shutil
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent


def load():
    spec = importlib.util.spec_from_file_location("mutate_par", HERE / "mutate_par.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_pool(root, workers, foreign=None):
    for w in range(workers):
        (root / ("w%d" % w) / "crates" / "lights").mkdir(parents=True)
        (root / ("w%d" % w) / "Cargo.toml").write_text("x", encoding="utf-8")
        (root / ("t%d" % w) / "debug").mkdir(parents=True)
        (root / ("t%d" % w) / "debug" / "big.rlib").write_text("x" * 64, encoding="utf-8")
    if foreign:
        (root / foreign).write_text("not ours", encoding="utf-8")


def case_a_clean_run_leaves_nothing(mod, fails):
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp) / "mutpool"
        root.mkdir()
        make_pool(root, 3)
        mod.remove_pool(root, 3)
        if root.exists():
            fails.append("case A: a pool whose every child was removed must not leave the root behind")


def case_b_a_foreign_file_survives(mod, fails):
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp) / "mutpool"
        root.mkdir()
        make_pool(root, 2, foreign="NOTES.md")
        mod.remove_pool(root, 2)
        if not (root / "NOTES.md").exists():
            fails.append("case B: remove_pool deleted a file the run did not create")
        if (root / "w0").exists() or (root / "t1").exists():
            fails.append("case B: the run's own worker trees must go even when the root stays")


def case_c_a_wider_earlier_pool_survives(mod, fails):
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp) / "mutpool"
        root.mkdir()
        make_pool(root, 6)
        mod.remove_pool(root, 2)
        for name in ("w0", "t0", "w1", "t1"):
            if (root / name).exists():
                fails.append("case C: %s belongs to this run and must be removed" % name)
        for name in ("w2", "w5", "t5"):
            if not (root / name).exists():
                fails.append("case C: %s was left by a wider earlier run and must not be touched" % name)


def case_d_an_absent_pool_is_not_an_error(mod, fails):
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp) / "never-made"
        try:
            mod.remove_pool(root, 4)
        except Exception as e:
            fails.append("case D: removing a pool that was never created raised %r" % e)


def case_e_the_removal_is_reported(mod, fails):
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp) / "mutpool"
        root.mkdir()
        make_pool(root, 2)
        gone = mod.remove_pool(root, 2)
        if len(gone) != 5:
            fails.append(
                "case E: four worker entries plus the root is five, and the count is what "
                "the run prints; got %d (%r)" % (len(gone), gone)
            )


def case_f_the_finally_arm_exists(mod, fails):
    source = (HERE / "mutate_par.py").read_text(encoding="utf-8")
    if "finally:" not in source:
        fails.append(
            "case F: a pool must be removed on a run that FAILS as well as one that "
            "succeeds, and the disk already holds the residue of a run that died with "
            "Errno 28 -- so the call has to sit in a finally arm"
        )
    if "MUT_KEEP_POOL" not in source:
        fails.append("case F: a failed mutation often needs its worker tree read, so keeping it must be possible")


def case_g_the_snapshot_carries_what_the_tests_read_outside_the_crate(mod, fails):
    with tempfile.TemporaryDirectory() as tmp:
        root, tree = pathlib.Path(tmp) / "repo", pathlib.Path(tmp) / "w0"
        for rel, mode in mod.OUTSIDE:
            if mode == "copy":
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / rel).write_text("body of " + rel, encoding="utf-8")
            elif mode == "tsv":
                (root / rel / "run-8").mkdir(parents=True, exist_ok=True)
                (root / rel / "run-8" / "report.tsv").write_text("cells of " + rel, encoding="utf-8")
                (root / rel / "big.png").write_bytes(b"x" * 64)
            else:
                (root / rel).mkdir(parents=True, exist_ok=True)
                (root / rel / "paper.pdf").write_bytes(b"x" * 64)
        mod.snapshot_outside(root, tree)
        for rel, mode in mod.OUTSIDE:
            if mode == "copy":
                if (tree / rel).read_text(encoding="utf-8") != "body of " + rel:
                    fails.append("case G: %s was not copied byte for byte" % rel)
            elif mode == "tsv":
                if not (tree / rel / "run-8" / "report.tsv").is_file() or (
                    tree / rel / "run-8" / "report.tsv"
                ).read_text(encoding="utf-8") != "cells of " + rel:
                    fails.append("case G: %s's nested tsv was not copied byte for byte" % rel)
                if (tree / rel / "big.png").exists():
                    fails.append("case G: %s copied a png the tests never read" % rel)
            elif [e.name for e in (tree / rel).iterdir()] != ["paper.pdf"]:
                fails.append("case G: %s does not carry the shelf's names" % rel)
        try:
            mod.snapshot_outside(pathlib.Path(tmp) / "absent", pathlib.Path(tmp) / "w1")
        except OSError as e:
            fails.append("case G: a tree without the outside files must snapshot, not raise: %s" % e)
        if not any(rel == "CLAUDE.md" for rel, _ in mod.OUTSIDE):
            fails.append("case G: prose_pointers.rs reads CLAUDE.md and the pool would not carry it")
        for shelf in ("docs/lights/p2", "docs/lights/p4", "docs/lights/p5"):
            if (shelf, "tsv") not in mod.OUTSIDE:
                fails.append("case G: tests/%s_artefacts.rs reads %s/**.tsv and the pool would not carry it" % (shelf[-2:], shelf))
        for need in (("docs/lights/benchmarks/l174", "tsv"), ("docs/lights/RESEARCH.json", "copy")):
            if need not in mod.OUTSIDE:
                fails.append("case G: tests/optical_baseline.rs reads %s and the pool would not carry it" % need[0])
        if ("docs/lights/benchmarks/l175", "tsv") not in mod.OUTSIDE:
            fails.append("case G: tests/ui_frame.rs reads docs/lights/benchmarks/l175/**.tsv and the pool would not carry it")
        if ("docs/lights/benchmarks/l176", "tsv") not in mod.OUTSIDE:
            fails.append("case G: tests/renderer_arms.rs reads docs/lights/benchmarks/l176/**.tsv and the pool would not carry it")
        if ("docs/lights/PLATFORM_CONTRACT.md", "copy") not in mod.OUTSIDE:
            fails.append("case G: tests/platform_contract.rs reads docs/lights/PLATFORM_CONTRACT.md and the pool would not carry it")


def case_h_a_red_test_is_read_from_the_harness_output(mod, fails):
    out = "running 3 tests\ntest a ... ok\ntest b ... FAILED\ntest c::d ... FAILED\ntest e ... ignored\n"
    if mod.red_tests(out) != {"b", "c::d"}:
        fails.append("case H: red_tests read %r" % sorted(mod.red_tests(out)))


def case_i_a_row_red_on_the_clean_tree_is_refused_untouched(mod, fails):
    with tempfile.TemporaryDirectory() as tmp:
        tree = pathlib.Path(tmp)
        (tree / "f.rs").write_text("alpha", encoding="utf-8")
        m = {"file": "f.rs", "test": "t", "features": [], "before": "alpha", "after": "beta"}
        verdict, _, gates = mod.evaluate(m, "dev", tree, tree / "target", {("lights", (), "t")})
        if verdict != "REFUSED" or gates:
            fails.append("case I: a baseline-red row scored %s %r" % (verdict, gates))
        if (tree / "f.rs").read_text(encoding="utf-8") != "alpha":
            fails.append("case I: a refused row still mutated the file")


def case_j_the_drivers_own_controls_do_not_reach_the_tests(mod, fails):
    e = mod.child_env({"MUT_ONLY": "x", "MUT_MERGE": "1", "PATH": "/bin"})
    if e != {"PATH": "/bin"}:
        fails.append("case J: a test that runs the driver would inherit %r" % sorted(e))

def case_k_a_sibling_package_is_snapshotted_and_named_by_its_rows(mod, fails):
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp) / "repo"
        expected = [("crates/lights", "lights"), ("crates/lights-mac", "lights-mac")]
        for rel, name in expected:
            (root / rel / "src").mkdir(parents=True)
            (root / rel / "Cargo.toml").write_text('[package]\nname = "%s"\n' % name, encoding="utf-8")
            (root / rel / "target").mkdir()
        tree = pathlib.Path(tmp) / "w0"
        members = mod.snapshot_packages(root, tree)
        if members != [rel for rel, _ in expected]:
            fails.append("case K: the pool carried %r of %r" % (members, expected))
        listed = (tree / "Cargo.toml").read_text(encoding="utf-8")
        for rel, _ in expected:
            if '"%s"' % rel not in listed:
                fails.append("case K: the pool workspace does not list %s" % rel)
            if (tree / rel / "target").exists():
                fails.append("case K: %s's target directory was copied into the pool" % rel)
        if ("crates/lights-mac", "lights-mac") not in mod.PACKAGES:
            fails.append("case K: tests/host.rs in crates/lights-mac has mutation rows and the pool would not carry the crate")
        shutil.rmtree(root / "crates" / "lights-mac")
        alone = mod.snapshot_packages(root, pathlib.Path(tmp) / "w1")
        if alone != ["crates/lights"]:
            fails.append("case K: without the sibling on disk the pool carried %r" % alone)
        if mod.package_of({}) != "lights" or mod.package_of({"package": "lights-mac"}) != "lights-mac":
            fails.append("case K: a row's package is not read, or does not default to lights")
        (tree / "f.rs").write_text("alpha", encoding="utf-8")
        m = {"file": "f.rs", "test": "t", "package": "lights-mac", "before": "alpha", "after": "beta"}
        verdict, _, gates = mod.evaluate(m, "dev", tree, tree / "target", {("lights-mac", (), "t")})
        if verdict != "REFUSED" or gates:
            fails.append("case K: a lights-mac row red on the clean tree scored %s %r" % (verdict, gates))


CASES = [
    case_a_clean_run_leaves_nothing,
    case_b_a_foreign_file_survives,
    case_c_a_wider_earlier_pool_survives,
    case_d_an_absent_pool_is_not_an_error,
    case_e_the_removal_is_reported,
    case_f_the_finally_arm_exists,
    case_g_the_snapshot_carries_what_the_tests_read_outside_the_crate,
    case_h_a_red_test_is_read_from_the_harness_output,
    case_i_a_row_red_on_the_clean_tree_is_refused_untouched,
    case_j_the_drivers_own_controls_do_not_reach_the_tests,
    case_k_a_sibling_package_is_snapshotted_and_named_by_its_rows,
]


def main():
    mod = load()
    fails = []
    for case in CASES:
        case(mod, fails)
    if fails:
        for f in fails:
            print("FAIL %s" % f)
        print("%d of %d checks failed" % (len(fails), len(CASES)))
        return 1
    print("mutate_par pool cleanup: %d checks green" % len(CASES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
