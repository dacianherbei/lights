import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CRATE = os.path.dirname(HERE)

sys.path.insert(0, HERE)
import burndown

HEAD = os.path.join(CRATE, "head_open.toml")
TAIL = os.path.join(CRATE, "tail.toml")
LEGACY = os.path.join(CRATE, "BURNDOWN.toml")
HEAD_SIZE = int(os.environ.get("BURNDOWN_HEAD_SIZE", "10"))

HEAD_HEADER = (
    "# THE CURRENT ITERATION. Generated and maintained by scripts/split_register.py --roll;\n"
    "# edit an entry here freely, but do not add or remove [[entry]] blocks by hand.\n"
    "# Everything else - the queue that has not been scheduled yet and every closed entry -\n"
    "# is in tail.toml, which no context should ever need to open.\n"
    "#\n"
    "# LIGHTS IS AN EXPERIMENTAL PROVING GROUND FOR THE FULGOR/NEXUS PARADIGM (PM, 2026-09-29).\n"
    "# It is NOT the renderer under `optical`. That half of L-120's two-target ruling was\n"
    "# WITHDRAWN on 2026-09-29: L-174 measured egui's whole CPU frame at 0.165-0.979 ms, L-175\n"
    "# the DrawList at 2.4-8.3x that on 18 threads against egui's one, and L-176 tiny-skia at\n"
    "# 2.2-10.2x faster than lights from a sixth of the memory - so the port would have made\n"
    "# optical slower on more cores for no behaviour gain. optical's source is off limits.\n"
    "# WHAT STANDS is the 3D target - a simple 3D game engine, the atom renderer no other crate\n"
    "# offers - and the fulgor/nexus paradigm case it carries: interchangeable strategies, with an\n"
    "# agreement contract between independent implementations. Judge every entry against that."
)
TAIL_HEADER = (
    "# THE QUEUE AND THE ARCHIVE. Open entries here are waiting for a future iteration, in\n"
    "# priority order; closed entries are the record. scripts/split_register.py --roll moves\n"
    "# entries between this file and head_open.toml. Do not read this file whole: it is large\n"
    "# on purpose and burndown.py loads both files for you."
)


def read(path):
    if not os.path.exists(path):
        return [], []
    entries, bad = burndown.parse(open(path).read())
    return entries, bad


def write(path, entries, header):
    open(path, "w").write(burndown.dump(entries, header))


def is_open(e):
    return e.get("status") == "open"


def split_files_exist():
    return os.path.exists(HEAD) and os.path.exists(TAIL)


def load_all():
    """Every entry, whichever layout is on disk. head first so --next prefers it."""
    if split_files_exist():
        head, bad_h = read(HEAD)
        tail, bad_t = read(TAIL)
        return head + tail, bad_h + bad_t
    return read(LEGACY)


def queue_order(entries):
    return sorted([e for e in entries if is_open(e)], key=burndown.rank)


def split_blocks(text):
    """Split a register file into (prelude, [(id, raw_text)]) WITHOUT parsing the fields.

    reconcile() moves entries between files by moving their RAW TEXT, so an entry that
    travels is byte-identical on the other side. The parse-and-redump path in do_split /
    do_roll reformats every entry it touches, which is fine for a one-off but would make
    every context's diff include entries it never edited.
    """
    pre, blocks, cur = [], [], None
    for line in text.splitlines(keepends=True):
        if line.strip() == "[[entry]]":
            cur = [line]
            blocks.append(cur)
            continue
        (pre if cur is None else cur).append(line)
    out = []
    for b in blocks:
        raw = "".join(b).rstrip("\n") + "\n"
        m = re.search(r'^id = "([^"]+)"', raw, re.M)
        out.append((m.group(1) if m else None, raw))
    return "".join(pre).rstrip("\n") + "\n", out


def workable(entries):
    """Open, not blocked, in the order --next would serve them."""
    live = [e for e in entries if e.get("status") == "open" and not burndown.is_blocked(e)]
    return sorted(live, key=burndown.rank)


def reconcile(budget, served=()):
    """Make head hold EXACTLY `budget` workable entries that are not already served.

    Called once per context by the host, before the context starts. Unlike --roll this is
    budget-aware and served-aware: an entry the current run has already worked does not come
    back, and a blocked entry never enters head at all, so a context is never handed work it
    cannot do. Closed entries in head go to the tail. Returns what moved.
    """
    if not os.path.isfile(TAIL):
        raise SystemExit("[split] %s is missing - nothing to fill the head from" % TAIL)
    tpre, tblocks = split_blocks(open(TAIL, encoding="utf-8").read())
    hblocks = []
    if os.path.isfile(HEAD):
        hblocks = split_blocks(open(HEAD, encoding="utf-8").read())[1]

    entries, bad = burndown.parse("".join(raw for _, raw in hblocks + tblocks))
    if bad:
        raise SystemExit("[split] the register could not be read: %s" % bad[:3])

    work = [e["id"] for e in workable(entries)]
    served = set(served)
    budget = int(budget)

    keep, out = [], []
    for eid, raw in hblocks:
        if eid in work and eid not in served and len(keep) < budget:
            keep.append((eid, raw))
        else:
            out.append((eid, raw))

    have = {eid for eid, _ in keep}
    tail_raw = dict(tblocks)
    moved_in = []
    for eid in work:
        if len(keep) >= budget:
            break
        if eid in have or eid in served or eid not in tail_raw:
            continue
        keep.append((eid, tail_raw[eid]))
        moved_in.append(eid)

    promoted = set(moved_in)
    new_tail = [(eid, raw) for eid, raw in tblocks if eid not in promoted] + out

    with open(HEAD, "w", encoding="utf-8") as fh:
        fh.write(HEAD_HEADER.rstrip("\n") + "\n\n" + "\n".join(raw for _, raw in keep))
    with open(TAIL, "w", encoding="utf-8") as fh:
        fh.write(tpre.rstrip("\n") + "\n\n" + "\n".join(raw for _, raw in new_tail))

    return {"head": [eid for eid, _ in keep], "promoted": moved_in,
            "retired": [eid for eid, _ in out], "queued": len(work) - len(served) - len(keep)}


def do_split():
    if split_files_exist():
        print("head_open.toml and tail.toml already exist; use --roll")
        return 1
    entries, bad = read(LEGACY)
    if bad:
        print("%s has %d unparseable line(s); refusing to split a register that does not parse"
              % (LEGACY, len(bad)))
        return 2
    if not entries:
        print("%s holds no [[entry]] block" % LEGACY)
        return 2

    ordered = queue_order(entries)
    head_ids = set(e["id"] for e in ordered[:HEAD_SIZE])
    head = [e for e in entries if e["id"] in head_ids]
    tail = [e for e in entries if e["id"] not in head_ids]

    write(HEAD, sorted(head, key=burndown.rank), HEAD_HEADER)
    write(TAIL, tail, TAIL_HEADER)
    print("split %d entries: %d into head_open.toml, %d into tail.toml"
          % (len(entries), len(head), len(tail)))
    print("BURNDOWN.toml is left in place; delete it once you trust the split")
    return 0


def do_roll():
    if not split_files_exist():
        print("head_open.toml or tail.toml is missing; run --split first")
        return 2
    head, bad_h = read(HEAD)
    tail, bad_t = read(TAIL)
    if bad_h or bad_t:
        print("refusing to roll: %d unparseable line(s)" % (len(bad_h) + len(bad_t)))
        return 2

    retired = [e for e in head if not is_open(e)]
    staying = [e for e in head if is_open(e)]
    want = HEAD_SIZE - len(staying)

    promoted = []
    if want > 0:
        for e in queue_order(tail):
            if len(promoted) == want:
                break
            promoted.append(e)
    promoted_ids = set(e["id"] for e in promoted)

    new_head = sorted(staying + promoted, key=burndown.rank)
    new_tail = [e for e in tail if e["id"] not in promoted_ids] + retired

    write(HEAD, new_head, HEAD_HEADER)
    write(TAIL, new_tail, TAIL_HEADER)

    print("retired %d closed to tail.toml%s"
          % (len(retired), (": " + ", ".join(e["id"] for e in retired)) if retired else ""))
    print("promoted %d from tail.toml%s"
          % (len(promoted), (": " + ", ".join(e["id"] for e in promoted)) if promoted else ""))
    print("head_open.toml now holds %d entries, %d open"
          % (len(new_head), len([e for e in new_head if is_open(e)])))
    left = len([e for e in new_tail if is_open(e)])
    print("tail.toml holds %d entries, %d still queued" % (len(new_tail), left))
    if not new_head and not left:
        print("REGISTER DRY")
    return 0


def do_check():
    if not split_files_exist():
        print("head_open.toml or tail.toml is missing; the register is not split")
        return 2
    head, bad_h = read(HEAD)
    tail, bad_t = read(TAIL)
    problems = []
    if bad_h:
        problems.append("head_open.toml has %d unparseable line(s)" % len(bad_h))
    if bad_t:
        problems.append("tail.toml has %d unparseable line(s)" % len(bad_t))

    seen = {}
    for where, group in (("head_open.toml", head), ("tail.toml", tail)):
        for e in group:
            key = e.get("id")
            if key in seen:
                problems.append("%s appears in both %s and %s; an entry must live in exactly one"
                                % (key, seen[key], where))
            seen[key] = where

    open_head = len([e for e in head if is_open(e)])
    if open_head > HEAD_SIZE:
        problems.append("head_open.toml holds %d open entries and the iteration size is %d; "
                        "head is meant to be one iteration, not a second register"
                        % (open_head, HEAD_SIZE))

    if os.path.exists(LEGACY):
        legacy, _ = read(LEGACY)
        lost = set(e["id"] for e in legacy) - set(seen)
        if lost:
            problems.append("%d id(s) present in BURNDOWN.toml and in neither split file: %s"
                            % (len(lost), ", ".join(sorted(lost))))

    print("\n".join(problems) if problems else "split clean")
    print("head %d (%d open) · tail %d (%d queued)"
          % (len(head), open_head, len(tail), len([e for e in tail if is_open(e)])))
    return 1 if problems else 0


def main(argv):
    if "--split" in argv:
        return do_split()
    if "--roll" in argv:
        return do_roll()
    if "--check" in argv:
        return do_check()
    sys.stderr.write("usage: split_register.py --split | --roll | --check\n")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
