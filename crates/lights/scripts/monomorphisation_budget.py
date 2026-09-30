import argparse
import os
import re
import sys

SYMBOL = re.compile(
    rb"_?_ZN[0-9A-Za-z_$.]*?(?P<method>splat|raycast)7execute11render_tile[0-9A-Za-z_$.]*"
)

KERNEL_IMPL = re.compile(r"^\s*impl(?:<[^>]*>)?\s+Kernel<T>\s+for\s+(\w+)", re.M)
PROBE_IMPL = re.compile(r"^\s*impl(?:<[^>]*>)?\s+Probe<T>\s+for\s+(\w+)", re.M)
PRECISION_ALL = re.compile(r"pub const ALL:\s*\[Precision;\s*(\d+)\]")


def source_prediction(src_dir):
    kernels, probes = set(), set()
    for root, _, files in os.walk(src_dir):
        for name in files:
            if not name.endswith(".rs"):
                continue
            text = open(os.path.join(root, name), encoding="utf-8").read()
            kernels.update(KERNEL_IMPL.findall(text))
            probes.update(PROBE_IMPL.findall(text))
    dispatch = open(os.path.join(src_dir, "dispatch.rs"), encoding="utf-8").read()
    found = PRECISION_ALL.search(dispatch)
    if not found:
        raise SystemExit("refusing: Precision::ALL not found in src/dispatch.rs")
    return sorted(kernels), sorted(probes), int(found.group(1))


def scan(path):
    raw = open(path, "rb").read()
    per = {"splat": set(), "raycast": set()}
    for match in SYMBOL.finditer(raw):
        per[match.group("method").decode()].add(match.group(0).decode())
    return os.path.getsize(path), per


def executables(target_dir):
    out = []
    for profile in ("debug", "release"):
        base = os.path.join(target_dir, profile)
        for where, shipped in ((base, True), (os.path.join(base, "deps"), False)):
            if not os.path.isdir(where):
                continue
            for name in sorted(os.listdir(where)):
                path = os.path.join(where, name)
                if not os.path.isfile(path) or not os.access(path, os.X_OK):
                    continue
                if name.endswith((".d", ".rlib", ".rmeta", ".dSYM")):
                    continue
                out.append((path, shipped, profile))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-dir", default="/Volumes/RamBuild/cargo-target")
    ap.add_argument("--src-dir", default=os.path.join(os.path.dirname(__file__), "..", "src"))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    kernels, probes, precisions = source_prediction(os.path.normpath(args.src_dir))
    predicted_per_method = len(kernels) * len(probes) * precisions

    rows, ceiling = [], {"splat": 0, "raycast": 0}
    shipped_seen = 0
    for path, shipped, profile in executables(args.target_dir):
        size, per = scan(path)
        if len(per["splat"]) + len(per["raycast"]) == 0:
            continue
        rows.append((os.path.basename(path), size, len(per["splat"]),
                     len(per["raycast"]), shipped, profile))
        if not shipped or profile != "debug":
            continue
        shipped_seen += 1
        for method in ceiling:
            ceiling[method] = max(ceiling[method], len(per[method]))

    if shipped_seen == 0:
        print(f"refusing: no debug shipping binary under {args.target_dir} "
              f"contains a render_tile symbol")
        return 3

    if not args.quiet:
        print(f"source: {len(kernels)} Kernel impls x {len(probes)} Probe impls "
              f"x {precisions} precisions = {predicted_per_method} per method")
        print(f"{'artefact':46} {'bytes':>10} {'splat':>6} {'raycast':>8}  where")
        for name, size, sp, rc, shipped, profile in sorted(rows, key=lambda r: -(r[2] + r[3])):
            where = ("bin " if shipped else "test ") + profile
            print(f"{name:46} {size:10d} {sp:6d} {rc:8d}  {where}")

    built = [m for m, n in ceiling.items() if n > 0]
    measured = sum(ceiling.values())
    print(f"MEASURED ceiling over {shipped_seen} debug shipping binaries: {measured} "
          f"render_tile instantiations across {len(built)} built method(s) {built}")
    print(f"PREDICTED       : {predicted_per_method * len(built)}")

    bad = [m for m in built if ceiling[m] != predicted_per_method]
    if bad:
        for m in bad:
            print(f"MISMATCH {m}: measured {ceiling[m]}, predicted {predicted_per_method}")
        return 1
    print("agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
