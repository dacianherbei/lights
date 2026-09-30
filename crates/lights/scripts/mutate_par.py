import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[3]
CRATE = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = CRATE / "scripts" / "mutations.json"
RESULTS = ROOT / "docs" / "lights" / "mutations" / "results.tsv"
COLUMNS = ["index", "verdict", "gates", "seconds", "run", "features", "file", "test", "name", "reason"]
MAX_MUTATIONS = 512
MAX_WORKERS = 16
RAM_POOL = pathlib.Path("/Volumes/RamBuild/mutpool")
SSD_POOL = pathlib.Path("/tmp/mutpool")

APPLIED, COMPILED, RAN, DETECTED = "applied", "compiled", "ran", "detected"

PACKAGES = [
    ("crates/lights", "lights"),
    ("crates/lights-mac", "lights-mac"),
]

OUTSIDE = [
    ("CLAUDE.md", "copy"),
    ("docs/lights/CORPUS_READING_MAP.md", "copy"),
    ("docs/lights/BURNDOWN_LOOP.md", "copy"),
    ("docs/lights/TEXT_LAYERS.md", "copy"),
    ("docs/lights/PLATFORM_CONTRACT.md", "copy"),
    ("docs/war_map.toml", "copy"),
    ("docs/plan/TWO_HORIZON_PLAN.md", "copy"),
    ("docs/gaussian_splatting", "names"),
    ("docs/lights/p2", "tsv"),
    ("docs/lights/p4", "tsv"),
    ("docs/lights/p5", "tsv"),
    ("docs/lights/benchmarks/l174", "tsv"),
    ("docs/lights/benchmarks/l175", "tsv"),
    ("docs/lights/benchmarks/l176", "tsv"),
    ("docs/lights/RESEARCH.json", "copy"),
    ("docs/lights/benchmarks/l186/winit_arm/Cargo.toml", "copy"),
]


def package_of(m):
    return m.get("package", "lights")


def snapshot_packages(root, tree):
    members = []
    for rel, _ in PACKAGES:
        src, dest = root / rel, tree / rel
        if not (src / "Cargo.toml").exists():
            continue
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dest, ignore=shutil.ignore_patterns("target"))
        members.append(rel)
    listed = ", ".join('"%s"' % m for m in members)
    (tree / "Cargo.toml").write_text(
        '[workspace]\nmembers = [%s]\nresolver = "3"\n' % listed, encoding="utf-8"
    )
    return members


def snapshot_outside(root, tree):
    for rel, mode in OUTSIDE:
        src, dest = root / rel, tree / rel
        if not src.exists():
            continue
        if mode == "copy":
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
        elif mode == "tsv":
            for found in sorted(src.rglob("*.tsv")):
                target = dest / found.relative_to(src)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(found, target)
        else:
            dest.mkdir(parents=True, exist_ok=True)
            for entry in src.iterdir():
                (dest / entry.name).write_bytes(b"")


def red_tests(stdout):
    red = set()
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0] == "test" and parts[2] == "..." and parts[3] == "FAILED":
            red.add(parts[1])
    return red


def baseline_red(rows, profile, tree, target):
    env = {"CARGO_TARGET_DIR": str(target)}
    groups = {}
    for m in rows:
        key = (package_of(m), tuple(m.get("features", [])))
        groups.setdefault(key, set()).add(m["test"])
    red = set()
    for (package, features), tests in groups.items():
        fa = feature_args({"features": list(features)})
        out = run(
            ["cargo", "test", "-p", package, "--profile", profile]
            + fa
            + ["--no-fail-fast", "--", "--exact"]
            + sorted(tests),
            tree,
            env,
        )
        red |= {(package, features, t) for t in red_tests(out.stdout)}
    return red


def child_env(environ):
    return {k: v for k, v in environ.items() if not k.startswith("MUT_")}


def run(args, cwd, env=None):
    e = child_env(os.environ)
    if env:
        e.update(env)
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, env=e)


def readable(rel, tree):
    p = tree / rel
    if not p.exists():
        return False, "file absent from the worker snapshot"
    return True, ""


def apply(rel, before, after, tree):
    path = tree / rel
    text = path.read_text(encoding="utf-8")
    ORIGINAL[(str(tree), rel)] = text
    if before not in text:
        return False, "pattern absent from the file"
    if before == after:
        return False, "pattern and replacement are identical"
    mutated = text.replace(before, after, 1)
    if mutated == text:
        return False, "replacement left the file byte-identical"
    path.write_text(mutated, encoding="utf-8")
    return True, ""


ORIGINAL = {}


def revert(rel, tree):
    """Restore the exact bytes saved at apply time. Independent of git, so an
    UNCOMMITTED working tree is safe to snapshot and mutate."""
    key = (str(tree), rel)
    original = ORIGINAL.pop(key, None)
    if original is None:
        return True
    path = tree / rel
    path.write_text(original, encoding="utf-8")
    return path.read_text(encoding="utf-8") == original


def feature_args(m):
    features = m.get("features", [])
    return ["--features", ",".join(features)] if features else []


def evaluate(m, profile, tree, target, red=frozenset()):
    rel, test, package = m["file"], m["test"], package_of(m)
    env = {"CARGO_TARGET_DIR": str(target)}
    gates_features = feature_args(m)
    if (package, tuple(m.get("features", [])), test) in red:
        return "REFUSED", "the test is red on the UNMUTATED snapshot, so a failure proves nothing", {}
    ok, why = readable(rel, tree)
    if not ok:
        return "REFUSED", why, {}

    ok, why = apply(rel, m["before"], m["after"], tree)
    if not ok:
        revert(rel, tree)
        return "REFUSED", why, {APPLIED: False}

    gates = {APPLIED: True, COMPILED: False, RAN: False, DETECTED: False}
    build = run(
        ["cargo", "build", "-p", package, "--profile", profile, "--tests"] + gates_features,
        tree,
        env,
    )
    if build.returncode != 0:
        revert(rel, tree)
        first = next(
            (l for l in build.stderr.splitlines() if l.startswith("error")), "build failed"
        )
        return "NOT-A-RESULT", "mutation did not compile: " + first, gates

    gates[COMPILED] = True
    out = run(
        ["cargo", "test", "-p", package, "--profile", profile]
        + gates_features
        + ["-q", "--", "--exact", test],
        tree,
        env,
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
    reverted = revert(rel, tree)

    if not reverted:
        return "DIRTY", "could not restore " + rel + " - inspect the tree", gates
    if executed == 0:
        return "REFUSED", "the filter matched no test named " + test, gates
    if not detected:
        return "ESCAPED", "the suite still passes with this mutation applied", gates
    reason = next(
        (l.strip() for l in out.stdout.splitlines() if "panicked at" in l), "test failed"
    )
    return "CAUGHT", reason, gates


def marks(gates):
    return "".join(
        "." if gates.get(g) else ("x" if g in gates else "-")
        for g in (APPLIED, COMPILED, RAN, DETECTED)
    )


def cell(text):
    return str(text).replace("\t", " ").replace("\n", " ").replace("\r", " ")


def key_of(m):
    return (m["file"], m["test"], m["name"])


def read_results(path):
    """Index an existing record by the same key the manifest is read with, so a
    partial run can be merged into it without pretending it ran the whole thing.
    A record written before the per-row `run` column existed carries one date for
    the whole file; that IS the date its rows were evaluated, so it is read out of
    the header rather than left blank."""
    if not path.is_file():
        return {}
    columns = []
    out = {}
    fallback = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#date\t"):
            fallback = line.split("\t", 1)[1]
        if line.startswith("#"):
            continue
        cells = line.split("\t")
        if not columns:
            columns = cells
            continue
        row = dict(zip(columns, cells))
        if "file" in row and "test" in row and "name" in row:
            for c in COLUMNS:
                if c not in row:
                    row[c] = fallback if c == "run" else ""
            out[(row["file"], row["test"], row["name"])] = row
    return out


def write_results(mutations, results, seconds, profile, workers, elapsed, only):
    target = pathlib.Path(os.environ.get("MUT_RESULTS", str(RESULTS)))
    target.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    merging = os.environ.get("MUT_MERGE", "") == "1"
    previous = read_results(target) if merging else {}

    fresh = {}
    for i, m in enumerate(mutations):
        verdict, reason, gates = results[i] if results[i] else ("NOT-RUN", "the driver never reached this row", {})
        fresh[key_of(m)] = {
            "verdict": verdict,
            "gates": marks(gates),
            "seconds": "%.1f" % seconds[i],
            "run": stamp,
            "features": ",".join(m.get("features", [])),
            "file": m["file"],
            "test": m["test"],
            "name": m["name"],
            "reason": reason[:200],
        }

    declared = json.loads(MANIFEST.read_text(encoding="utf-8"))["mutations"]
    body = []
    for m in declared:
        k = key_of(m)
        row = fresh.get(k) or previous.get(k)
        if row is None:
            row = {
                "verdict": "NOT-RUN",
                "gates": "----",
                "seconds": "0.0",
                "run": "",
                "features": ",".join(m.get("features", [])),
                "file": m["file"],
                "test": m["test"],
                "name": m["name"],
                "reason": "no run has ever evaluated this manifest row",
            }
        body.append(row)

    caught = sum(1 for r in body if r["verdict"] == "CAUGHT")
    lines = [
        "#invocation\t" + " ".join(sys.argv),
        "#date\t" + stamp,
        "#profile\t" + profile,
        "#workers\t" + str(workers),
        "#prefix_limit\t" + str(only),
        "#merged\t" + ("1" if merging else "0"),
        "#manifest\t" + str(MANIFEST.relative_to(ROOT)),
        "#manifest_rows\t" + str(len(declared)),
        "#rows\t" + str(len(body)),
        "#evaluated\t" + str(len(mutations)),
        "#caught\t" + str(caught),
        "#not_caught\t" + str(len(body) - caught),
        "#seconds\t%.1f" % elapsed,
        "\t".join(COLUMNS),
    ]
    for i, row in enumerate(body):
        row["index"] = i
        lines.append("\t".join(cell(row[c]) for c in COLUMNS))
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def remove_pool(pool_root, workers):
    """Removes EXACTLY the w<i>/t<i> children this run makes, and the root only
    once it is empty. A caller can pass any path as the third argument, so a
    blind rmtree of the root would be a loaded gun pointed at whatever that path
    happens to be. A w<i> left by an earlier, wider run therefore survives: not
    deleting what we did not create is the property worth keeping."""
    removed = []
    for w in range(workers):
        for name in ("w%d" % w, "t%d" % w):
            child = pool_root / name
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
                removed.append(name)
    if pool_root.is_dir() and not any(pool_root.iterdir()):
        pool_root.rmdir()
        removed.append(str(pool_root))
    return removed


def default_pool_root(free=None):
    """The RAM disk when it has room above the shared floor, the SSD otherwise.
    Same rule and same floor as target_dir.py, because the pool is a tenant of
    the same volume and takes the same headroom cargo needs."""
    import target_dir
    if free is None:
        free = target_dir.free_bytes(str(RAM_POOL))
    floor = int(target_dir.FLOOR_GIB * target_dir.GIB)
    _, chosen = target_dir.choose(str(RAM_POOL), free, floor, str(SSD_POOL))
    return pathlib.Path(chosen)


def main():
    profile = sys.argv[1] if len(sys.argv) > 1 else "release"
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    pool_root = pathlib.Path(sys.argv[3]) if len(sys.argv) > 3 else default_pool_root()
    only = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    if workers > MAX_WORKERS:
        print("workers exceeds MAX_WORKERS")
        return 2
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    mutations = manifest["mutations"]
    needles = [n for n in os.environ.get("MUT_ONLY", "").split(",") if n]
    if needles:
        mutations = [
            m for m in mutations if any(n in m["test"] or n in m["name"] for n in needles)
        ]
        if not mutations:
            print("MUT_ONLY matched no row; the manifest is left alone")
            return 2
        print("MUT_ONLY selected %d of %d rows" % (len(mutations), len(manifest["mutations"])))
    if only:
        mutations = mutations[:only]
    if len(mutations) > MAX_MUTATIONS:
        print("manifest exceeds MAX_MUTATIONS")
        return 2

    trees, targets = [], []
    pool_root.mkdir(parents=True, exist_ok=True)
    for w in range(workers):
        t = pool_root / ("w%d" % w)
        g = pool_root / ("t%d" % w)
        t.mkdir(parents=True, exist_ok=True)
        snapshot_packages(ROOT, t)
        snapshot_outside(ROOT, t)
        trees.append(t)
        targets.append(g)
    print("snapshotted the current crate into %d workers under %s" % (workers, pool_root), flush=True)
    keep = os.environ.get("MUT_KEEP_POOL", "") != ""

    results = [None] * len(mutations)
    seconds = [0.0] * len(mutations)
    counter = {"done": 0}
    emit = __import__("threading").Lock()

    def worker(w):
        """One TREE per WORKER THREAD, never per task index. Two tasks that map
        to the same tree must never run concurrently in it, or one sees the
        other's dirty file and the clean-tree guard refuses it."""
        mine = range(w, len(mutations), workers)
        red = baseline_red([mutations[i] for i in mine], profile, trees[w], targets[w])
        for i in mine:
            started = time.time()
            res = evaluate(mutations[i], profile, trees[w], targets[w], red)
            seconds[i] = time.time() - started
            results[i] = res
            with emit:
                counter["done"] += 1
                verdict, reason, gates = res
                print(
                    "[%d/%d] w%d %-14s %-5s %s"
                    % (counter["done"], len(mutations), w, verdict, marks(gates), mutations[i]["name"][:64]),
                    flush=True,
                )

    try:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(worker, range(workers)))
        elapsed = time.time() - t0

        bad = [r for r in results if r and r[0] != "CAUGHT"]
        where = write_results(mutations, results, seconds, profile, workers, elapsed, only)
        print()
        print("%d caught, %d not, in %.1f s on %d workers" % (len(results) - len(bad), len(bad), elapsed, workers))
        print("result written to %s" % where)
        for i, r in enumerate(results):
            if r and r[0] != "CAUGHT":
                print("  %-14s %s :: %s" % (r[0], mutations[i]["name"][:56], r[1][:90]))
        if bad:
            print("a mutation that does not COMPILE is not a tested mutation, and a")
            print("mutation that ESCAPES is a hole in the suite. Neither counts as a pass.")
        return 1 if bad else 0
    finally:
        if keep:
            print("MUT_KEEP_POOL is set, so %s stays on disk for inspection" % pool_root)
        else:
            gone = remove_pool(pool_root, workers)
            print("freed %d pool entries under %s" % (len(gone), pool_root))


if __name__ == "__main__":
    sys.exit(main())
