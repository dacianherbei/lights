import json
import os
import pathlib
import shutil
import subprocess
import tempfile
import sys

HOME = pathlib.Path(__file__).resolve().parents[3]
CRATE = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = CRATE / "scripts" / "mutations.json"
MAX_MUTATIONS = 512
BACKUPS = pathlib.Path(tempfile.gettempdir()) / "lights-mutate-backups"
WORKSPACE = '[workspace]\nmembers = ["crates/lights"]\nresolver = "3"\n'

USAGE = """\
mutate.py [profile] (--tree DIR | --snapshot DIR)

  --tree DIR      an ALREADY isolated checkout to mutate. Must not be the tree
                  this script lives in.
  --snapshot DIR  make that isolated checkout first, by copying crates/lights
                  into DIR/crates/lights, then mutate it.

This driver edits files IN PLACE. Run against the working tree it lives in, a
crash or a kill between apply and revert leaves the user's source mutated with
nothing to say so. Point it at a scratch tree, or use mutate_par.py, which
snapshots into worker trees by construction."""

APPLIED, COMPILED, RAN, DETECTED = "applied", "compiled", "ran", "detected"


def resolve_tree(argv):
    """Return (profile, tree, target) or (None, message, None) to refuse."""
    profile = "release"
    tree = None
    make = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--help", "-h"):
            return None, USAGE, None
        if a == "--tree":
            if i + 1 >= len(argv):
                return None, "--tree needs a value\n\n" + USAGE, None
            tree = pathlib.Path(argv[i + 1])
            i += 2
            continue
        if a == "--snapshot":
            if i + 1 >= len(argv):
                return None, "--snapshot needs a value\n\n" + USAGE, None
            make = pathlib.Path(argv[i + 1])
            i += 2
            continue
        if a.startswith("-"):
            return None, "unknown flag " + a + "\n\n" + USAGE, None
        profile = a
        i += 1

    if tree is not None and make is not None:
        return None, "--tree and --snapshot are alternatives\n\n" + USAGE, None
    chosen = tree if tree is not None else make
    if chosen is None:
        return None, "no isolated tree given\n\n" + USAGE, None

    chosen = chosen.expanduser().resolve()
    if chosen == HOME or HOME == chosen or HOME in chosen.parents or chosen in HOME.parents:
        return (
            None,
            "REFUSED: " + str(chosen) + " overlaps the tree mutate.py lives in ("
            + str(HOME) + "). This driver edits files in place; a crash between "
            "apply and revert would leave that tree mutated. Use --snapshot with "
            "a path outside it, or run mutate_par.py.\n\n" + USAGE,
            None,
        )

    if make is not None:
        dest = chosen / "crates" / "lights"
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(CRATE, dest, ignore=shutil.ignore_patterns("target"))
        (chosen / "Cargo.toml").write_text(WORKSPACE, encoding="utf-8")
    if not (chosen / "crates" / "lights" / "Cargo.toml").is_file():
        return (
            None,
            "REFUSED: " + str(chosen) + " holds no crates/lights to mutate. Pass "
            "--snapshot instead of --tree to build it.\n\n" + USAGE,
            None,
        )
    return profile, chosen, chosen / "mutate-target"


def run(args, tree, target):
    env = dict(os.environ)
    env["CARGO_TARGET_DIR"] = str(target)
    return subprocess.run(args, cwd=tree, capture_output=True, text=True, env=env)


def snapshot(rel, tree):
    path = tree / rel
    if not path.is_file():
        return None, "no file at " + rel + ", so there is nothing to mutate"
    original = path.read_bytes()
    BACKUPS.mkdir(parents=True, exist_ok=True)
    keep = BACKUPS / rel.replace("/", "%")
    keep.write_bytes(original)
    return original, ""


def apply(rel, before, after, tree):
    path = tree / rel
    text = path.read_text(encoding="utf-8")
    if before not in text:
        return False, "pattern absent from the file"
    if before == after:
        return False, "pattern and replacement are identical"
    mutated = text.replace(before, after, 1)
    if mutated == text:
        return False, "replacement left the file byte-identical"
    path.write_text(mutated, encoding="utf-8")
    return True, ""


def revert(rel, original, tree):
    path = tree / rel
    path.write_bytes(original)
    return path.read_bytes() == original


def feature_args(m):
    features = m.get("features", [])
    return ["--features", ",".join(features)] if features else []


def evaluate(m, profile, tree, target):
    rel, test = m["file"], m["test"]
    gates_features = feature_args(m)
    original, why = snapshot(rel, tree)
    if original is None:
        return "REFUSED", why, {}

    ok, why = apply(rel, m["before"], m["after"], tree)
    if not ok:
        revert(rel, original, tree)
        return "REFUSED", why, {APPLIED: False}

    gates = {APPLIED: True, COMPILED: False, DETECTED: False}
    build = run(
        ["cargo", "build", "-p", "lights", "--profile", profile, "--tests"] + gates_features,
        tree,
        target,
    )
    if build.returncode != 0:
        revert(rel, original, tree)
        first = next(
            (l for l in build.stderr.splitlines() if l.startswith("error")), "build failed"
        )
        return "NOT-A-RESULT", "mutation did not compile: " + first, gates

    gates[COMPILED] = True
    out = run(
        ["cargo", "test", "-p", "lights", "--profile", profile]
        + gates_features
        + ["-q", "--", "--exact", test],
        tree,
        target,
    )
    executed = 0
    for line in out.stdout.splitlines():
        if line.startswith("test result:"):
            parts = line.replace(";", "").split()
            for i, tok in enumerate(parts):
                if tok in ("passed", "failed") and i > 0 and parts[i - 1].isdigit():
                    executed += int(parts[i - 1])
    gates[RAN] = executed > 0
    detected = out.returncode != 0
    gates[DETECTED] = detected
    reverted = revert(rel, original, tree)

    if not reverted:
        return "DIRTY", "could not restore " + rel + " from " + str(BACKUPS), gates
    if executed == 0:
        return "REFUSED", "the filter matched no test named " + test, gates
    if not detected:
        return "ESCAPED", "the suite still passes with this mutation applied", gates
    reason = next(
        (l.strip() for l in out.stdout.splitlines() if "panicked at" in l),
        "test failed",
    )
    return "CAUGHT", reason, gates


def main():
    profile, tree, target = resolve_tree(sys.argv[1:])
    if profile is None:
        print(tree)
        return 2
    print("mutating the isolated tree at %s" % tree, flush=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    mutations = manifest["mutations"]
    if len(mutations) > MAX_MUTATIONS:
        print("manifest exceeds MAX_MUTATIONS")
        return 2

    verdicts = []
    for i, m in enumerate(mutations):
        verdict, reason, gates = evaluate(m, profile, tree, target)
        verdicts.append(verdict)
        marks = "".join(
            "." if gates.get(g) else ("x" if g in gates else "-")
            for g in (APPLIED, COMPILED, RAN, DETECTED)
        )
        print("[{}/{}] {:<14} {:<12} {}".format(i + 1, len(mutations), verdict, marks, m["name"]))
        print("         " + reason[:160])

    bad = [v for v in verdicts if v != "CAUGHT"]
    print()
    print("{} caught, {} not".format(len(verdicts) - len(bad), len(bad)))
    if bad:
        print("a mutation that does not COMPILE is not a tested mutation, and a")
        print("mutation that ESCAPES is a hole in the suite. Neither counts as a pass.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
