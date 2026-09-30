import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

FIELDS = 'title = "t%(n)d"\narea = "core"\nkind = "%(kind)s"\nstate = "absent"\n' \
         'status = "%(status)s"\nraised = "2026-09-01"\nsize = "small"\n' \
         'where = "src/x.rs"\nwhy = "w."\nnext = "%(nxt)s"\n'


def register(count, open_count):
    out = ["# synthetic", ""]
    for n in range(1, count + 1):
        opened = n <= open_count
        out.append("[[entry]]")
        out.append('id = "L-%d"' % n)
        out.append(FIELDS % {
            "n": n,
            "kind": "defect" if n % 7 == 0 else "owed",
            "status": "open" if opened else "closed",
            "nxt": "do the thing." if opened else "BUILT 2026-09-01 - done.",
        })
    return "\n".join(out) + "\n"


def tool(tree, *args, **kw):
    env = dict(os.environ)
    env.update(kw.pop("env", {}))
    done = subprocess.run(
        [sys.executable, os.path.join(tree, "crates", "lights", "scripts", "split_register.py")] + list(args),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, env=env,
    )
    return done.returncode, done.stdout


def ids(path):
    sys.path.insert(0, os.path.join(os.path.dirname(path), "scripts"))
    import burndown
    if not os.path.exists(path):
        return set()
    return set(e["id"] for e in burndown.parse(open(path).read())[0])


def build(total, opens):
    tree = tempfile.mkdtemp()
    scripts = os.path.join(tree, "crates", "lights", "scripts")
    os.makedirs(scripts)
    for name in ("burndown.py", "split_register.py"):
        shutil.copy(os.path.join(HERE, name), scripts)
    with open(os.path.join(tree, "crates", "lights", "BURNDOWN.toml"), "w") as fh:
        fh.write(register(total, opens))
    return tree


def close_in_head(tree, how_many):
    sys.path.insert(0, os.path.join(tree, "crates", "lights", "scripts"))
    import burndown
    path = os.path.join(tree, "crates", "lights", "head_open.toml")
    entries, _ = burndown.parse(open(path).read())
    for e in entries[:how_many]:
        e["status"] = "closed"
        e["next"] = "BUILT 2026-09-01 - closed by the selftest."
    open(path, "w").write(burndown.dump(entries, "# head"))


def main():
    failures = []

    def want(name, got, expected):
        if got != expected:
            failures.append("%s: expected %r, got %r" % (name, expected, got))

    # --- the happy path, and conservation across it -------------------------
    tree = build(40, 25)
    legacy = ids(os.path.join(tree, "crates", "lights", "BURNDOWN.toml"))
    code, _ = tool(tree, "--split")
    want("split exits 0", code, 0)
    head = os.path.join(tree, "crates", "lights", "head_open.toml")
    tail = os.path.join(tree, "crates", "lights", "tail.toml")
    want("head holds one iteration", len(ids(head)), 10)
    want("nothing is lost by the split", ids(head) | ids(tail), legacy)
    want("nothing is in both files", ids(head) & ids(tail), set())
    code, _ = tool(tree, "--check")
    want("check is clean after split", code, 0)

    code, out = tool(tree, "--split")
    want("a second --split refuses", code, 1)

    close_in_head(tree, 4)
    code, out = tool(tree, "--roll")
    want("roll exits 0", code, 0)
    want("roll retires exactly the closed ones", "retired 4 closed" in out, True)
    want("roll promotes the same number", "promoted 4 from" in out, True)
    want("head is refilled to the iteration size", len([i for i in ids(head)]), 10)
    want("nothing is lost by the roll", ids(head) | ids(tail), legacy)
    want("nothing is duplicated by the roll", ids(head) & ids(tail), set())
    code, _ = tool(tree, "--check")
    want("check is clean after roll", code, 0)

    # --- rolling with nothing closed must be a no-op -------------------------
    before_head, before_tail = ids(head), ids(tail)
    code, out = tool(tree, "--roll")
    want("an idle roll exits 0", code, 0)
    want("an idle roll moves nothing out", ids(head), before_head)
    want("an idle roll moves nothing in", ids(tail), before_tail)

    # --- draining to dry -----------------------------------------------------
    small = build(12, 12)
    tool(small, "--split", env={"BURNDOWN_HEAD_SIZE": "10"})
    close_in_head(small, 10)
    code, out = tool(small, "--roll", env={"BURNDOWN_HEAD_SIZE": "10"})
    want("the last two are promoted", "promoted 2 from" in out, True)
    close_in_head(small, 2)
    code, out = tool(small, "--roll", env={"BURNDOWN_HEAD_SIZE": "10"})
    want("an empty register announces itself", "REGISTER DRY" in out, True)

    # --- the breaks the checker must catch -----------------------------------
    dup = build(20, 15)
    tool(dup, "--split")
    d_head = os.path.join(dup, "crates", "lights", "head_open.toml")
    d_tail = os.path.join(dup, "crates", "lights", "tail.toml")
    with open(d_tail, "a") as fh:
        fh.write(open(d_head).read().split("[[entry]]", 1)[1].join(["[[entry]]", ""]))
    code, out = tool(dup, "--check")
    want("an entry in BOTH files is refused", code, 1)
    want("and it says which", "appears in both" in out, True)

    lost = build(20, 15)
    tool(lost, "--split")
    l_tail = os.path.join(lost, "crates", "lights", "tail.toml")
    text = open(l_tail).read()
    cut = text.rindex("[[entry]]")
    open(l_tail, "w").write(text[:cut])
    code, out = tool(lost, "--check")
    want("an entry dropped from tail is refused", code, 1)
    want("and it says it is lost", "in neither split file" in out, True)

    over = build(40, 30)
    tool(over, "--split")
    o_head = os.path.join(over, "crates", "lights", "head_open.toml")
    o_tail = os.path.join(over, "crates", "lights", "tail.toml")
    parts = open(o_tail).read().split("[[entry]]")
    with open(o_head, "a") as fh:
        for chunk in parts[1:6]:
            fh.write("[[entry]]" + chunk)
    code, out = tool(over, "--check")
    want("an oversized head is refused", code, 1)
    want("and it says the iteration size", "iteration size" in out, True)

    missing = build(20, 15)
    code, out = tool(missing, "--roll")
    want("--roll before --split refuses", code, 2)
    code, out = tool(missing, "--check")
    want("--check before --split refuses", code, 2)

    for tmp in (tree, small, dup, lost, over, missing):
        shutil.rmtree(tmp, ignore_errors=True)

    if failures:
        for line in failures:
            print("FAIL " + line)
        return 1
    print("split register: 26 deliberate cases, all correct")
    return 0


if __name__ == "__main__":
    sys.exit(main())
