#!/usr/bin/env python3
import argparse
import os
import shutil
import sys

FLOOR_GIB = 8.0
GIB = 1024 ** 3

CONFIGURED = "configured"
FALLBACK = "fallback"
UNMEASURED = "unmeasured"


def repo_root():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", ".."))


def cargo_home():
    return os.environ.get("CARGO_HOME") or os.path.join(os.path.expanduser("~"), ".cargo")


def configured_target_dir():
    from_env = os.environ.get("CARGO_TARGET_DIR")
    if from_env:
        return os.path.abspath(from_env)
    config = os.path.join(cargo_home(), "config.toml")
    try:
        with open(config) as handle:
            lines = handle.read().splitlines()
    except OSError:
        return None
    section = None
    for line in lines:
        stripped = line.split("#", 1)[0].strip()
        if not stripped:
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1].strip()
            continue
        if section != "build" or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() != "target-dir":
            continue
        value = value.strip().strip('"').strip("'")
        if value:
            return os.path.abspath(os.path.expanduser(value))
    return None


def free_bytes(path):
    probe = path
    while probe and not os.path.isdir(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            return None
        probe = parent
    if not probe:
        return None
    try:
        return shutil.disk_usage(probe).free
    except OSError:
        return None


def choose(configured, free, floor, fallback):
    if configured is None:
        return FALLBACK, fallback
    if free is None:
        return UNMEASURED, configured
    if free < floor:
        return FALLBACK, fallback
    return CONFIGURED, configured


def main(argv):
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--floor-gib", type=float, default=FLOOR_GIB)
    parser.add_argument("--free-bytes", type=int, default=None)
    parser.add_argument("--configured", default=None)
    parser.add_argument("--fallback", default=None)
    parser.add_argument("--explain", action="store_true")
    args = parser.parse_args(argv)

    configured = args.configured if args.configured is not None else configured_target_dir()
    if configured == "":
        configured = None
    fallback = args.fallback or os.path.join(repo_root(), "target")
    free = args.free_bytes if args.free_bytes is not None else (
        free_bytes(configured) if configured else None
    )
    floor = int(args.floor_gib * GIB)

    reason, chosen = choose(configured, free, floor, fallback)

    if args.explain:
        seen = "unmeasured" if free is None else "%.2f GiB" % (free / GIB)
        sys.stderr.write(
            "configured=%s free=%s floor=%.2f GiB -> %s\n"
            % (configured or "none", seen, floor / GIB, reason)
        )
    print(chosen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
