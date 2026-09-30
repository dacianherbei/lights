import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import burndown

GOOD = '''
[[entry]]
id = "L-1"
title = "a thing that is not there"
area = "kernel"
kind = "defect"
state = "absent"
status = "open"
raised = "2026-09-09"
size = "small"
where = "src/kernel/mod.rs"
why = "measured at src/kernel/mod.rs:12 today."
next = "read src/kernel/mod.rs:12."

[[entry]]
id = "L-2"
title = "a thing that was built"
area = "method"
kind = "feature"
state = "built"
status = "closed"
raised = "2026-09-09"
size = "small"
where = "src/method/mod.rs"
why = "built at src/method/mod.rs:40."
next = "BUILT 2026-09-09, covered by tests/method_matrix.rs."
'''

BREAKS = [
    ("duplicate id", GOOD.replace('id = "L-2"', 'id = "L-1"'), "duplicate id"),
    ("bad id shape", GOOD.replace('id = "L-2"', 'id = "O-2"'), "is not L-<n>"),
    ("missing field", GOOD.replace('why = "measured at src/kernel/mod.rs:12 today."\n', ""), "is missing"),
    ("unknown kind", GOOD.replace('kind = "defect"', 'kind = "bug"'), "kind ="),
    ("unknown area", GOOD.replace('area = "kernel"', 'area = "renderer"'), "area ="),
    ("unknown state", GOOD.replace('state = "absent"', 'state = "shipped"'), "state ="),
    ("unknown size", GOOD.replace('size = "small"', 'size = "tiny"'), "size ="),
    ("future date", GOOD.replace('raised = "2026-09-09"', 'raised = "2099-01-01"'), "in the future"),
    ("bad date", GOOD.replace('raised = "2026-09-09"', 'raised = "9 Sep"'), "not YYYY-MM-DD"),
    ("closed without evidence word", GOOD.replace('next = "BUILT 2026-09-09, covered by tests/method_matrix.rs."',
                                                  'next = "it works now."'), "does not open with"),
    ("truncated field", GOOD.replace('why = "measured at src/kernel/mod.rs:12 today."',
                                     'why = "%s and then the sentence stops in the middle of a wo"' % ("x" * 100)),
     "stops mid-sentence"),
    ("unreadable line", GOOD + "\nthis is not a pair\n", "could not be read"),
    ("blocked buried in next", GOOD.replace(
        'next = "read src/kernel/mod.rs:12."',
        'next = "Write the ruling first. That ruling is BLOCKED on the primitive-set entry below."'),
     "invisible to is_blocked"),
    ("blocker has closed", GOOD.replace(
        'next = "read src/kernel/mod.rs:12."',
        'next = "BLOCKED ON L-2, which adds the derivative this consumes."'), "L-1 is blocked on L-2, which is closed"),
    ("blocker never existed", GOOD.replace(
        'next = "read src/kernel/mod.rs:12."',
        'next = "BLOCKED 2026-09-15 by L-7 and L-2."'), "L-1 is blocked on L-7, which is not in the register"),
    ("duplicate title", GOOD.replace('title = "a thing that was built"',
                                     'title = "A thing that is not there!"'), "word for word"),
    ("cluster of one", GOOD.replace('size = "small"\nwhere = "src/kernel/mod.rs"',
                                    'size = "small"\ncluster = "alone"\nwhere = "src/kernel/mod.rs"'),
     "carried by one entry"),
]


SECOND = '''
[[entry]]
id = "L-900"
title = "an entry that lives in the tail and must never reach the report"
area = "core"
kind = "owed"
state = "absent"
status = "open"
raised = "2026-09-01"
size = "small"
where = "src/x.rs"
why = "w."
next = "do it."

[[entry]]
id = "L-901"
title = "a closed entry that lives in the tail"
area = "core"
kind = "owed"
state = "built"
status = "closed"
raised = "2026-09-01"
size = "small"
where = "src/x.rs"
why = "w."
next = "BUILT 2026-09-01 - done."
'''


def main():
    entries, bad = burndown.parse(GOOD)
    problems = burndown.check(entries, bad)
    failures = []
    if problems:
        failures.append("the good register was refused: %s" % problems)
    if burndown.standing(entries) != (1, 2):
        failures.append("standing miscounted: %s" % (burndown.standing(entries),))
    if burndown.oldest_open(entries)["id"] != "L-1":
        failures.append("oldest_open picked the wrong entry")
    waiting = GOOD.replace('next = "read src/kernel/mod.rs:12."',
                           'next = "BLOCKED on a PM ruling: is it owed at all?"')
    unblocked = waiting.replace('status = "closed"', 'status = "open"').replace(
        'next = "BUILT 2026-09-09, covered by tests/method_matrix.rs."', 'next = "build it."')
    e3, _ = burndown.parse(unblocked)
    picked = burndown.oldest_open(e3)
    if picked is None or picked["id"] != "L-2":
        failures.append("oldest_open handed out a BLOCKED entry instead of skipping to L-2: %r"
                        % (picked and picked["id"]))
    if [b["id"] for b in burndown.blocked_open(e3)] != ["L-1"]:
        failures.append("blocked_open did not list exactly L-1")
    e4, _ = burndown.parse(waiting)
    if burndown.oldest_open(e4) is not None:
        failures.append("a register whose only open entry is BLOCKED is not dry")
    stated = GOOD.replace('state = "absent"', 'state = "blocked"')
    e5, _ = burndown.parse(stated)
    if burndown.oldest_open(e5) is not None:
        failures.append("state = blocked was not honoured by oldest_open")
    mention = GOOD.replace('next = "read src/kernel/mod.rs:12."',
                           'next = "FIX the predicate that treats BLOCKED anywhere as a block."')
    e6, _ = burndown.parse(mention)
    picked = burndown.oldest_open(e6)
    if picked is None or picked["id"] != "L-1":
        failures.append("an entry that merely MENTIONS the word blocked was skipped: %r" % (picked and picked["id"]))
    capability = GOOD.replace('next = "read src/kernel/mod.rs:12."',
                              'next = "BLOCKED 2026-09-16 on the same capability as L-2, not on a ruling."')
    e7, b7 = burndown.parse(capability)
    if burndown.check(e7, b7):
        failures.append("an entry blocked on a CAPABILITY that merely mentions a closed id was "
                        "reported as waiting on it: %s" % burndown.check(e7, b7))
    round_trip, _ = burndown.parse(burndown.dump(entries, "# header"))
    if [e["why"] for e in round_trip] != [e["why"] for e in entries]:
        failures.append("dump/parse is not a round trip")
    for name, text, want in BREAKS:
        e2, b2 = burndown.parse(text)
        got = burndown.check(e2, b2)
        if not any(want in p for p in got):
            failures.append("BREAK NOT CAUGHT: %s (wanted %r, got %s)" % (name, want, got or "clean"))
    import tempfile, os, io, contextlib

    def refused_by_check(want, register, head, tail):
        """Run --check with the module's file constants repointed, and require a refusal."""
        saved = (burndown.REGISTER, burndown.HEAD, burndown.TAIL)
        burndown.REGISTER, burndown.HEAD, burndown.TAIL = register, head, tail
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                code = burndown.main(["--check"])
        finally:
            burndown.REGISTER, burndown.HEAD, burndown.TAIL = saved
        return code == 1 and want in buf.getvalue(), buf.getvalue().strip()

    tmp = tempfile.mkdtemp()
    empty = os.path.join(tmp, "BURNDOWN.toml")
    open(empty, "w").write("# a register with no entries\n")
    gone = os.path.join(tmp, "nothing-here")

    # the single-file layout, which survives for anyone who has not split yet
    for label, register, want in (
        ("absent", gone, "nor the head/tail pair exists"),
        ("empty", empty, "holds no [[entry]] block"),
    ):
        ok, saw = refused_by_check(want, register, gone + ".head", gone + ".tail")
        if not ok:
            failures.append("a %s single-file register was not refused: %r" % (label, saw))

    # the split layout: an empty PAIR must be refused just as loudly, or a lost
    # head and a lost tail would read as a clean register with nothing in it
    head = os.path.join(tmp, "head_open.toml")
    tail = os.path.join(tmp, "tail.toml")
    open(head, "w").write("# no entries\n")
    open(tail, "w").write("# no entries\n")
    ok, saw = refused_by_check("holds no [[entry]] block", gone, head, tail)
    if not ok:
        failures.append("an empty head/tail PAIR was not refused: %r" % saw)

    # and the pair must be preferred over a stale single file that still exists
    open(head, "w").write(GOOD)
    saved = (burndown.REGISTER, burndown.HEAD, burndown.TAIL)
    burndown.REGISTER, burndown.HEAD, burndown.TAIL = empty, head, tail
    try:
        picked, _, head_ids = burndown.load()
    finally:
        burndown.REGISTER, burndown.HEAD, burndown.TAIL = saved
    if not picked or head_ids is None:
        failures.append("the split pair was not preferred over a single register that still exists")

    # THE LOADER MUST RETURN BOTH FILES. Dropping `+ tail` leaves --next and
    # --clusters blind to everything not in the current iteration, and the only
    # visible symptom is that the standing count quietly collapses to head's.
    open(head, "w").write(GOOD)
    open(tail, "w").write(SECOND)
    saved = (burndown.REGISTER, burndown.HEAD, burndown.TAIL)
    burndown.REGISTER, burndown.HEAD, burndown.TAIL = gone, head, tail
    try:
        both, _, head_ids = burndown.load()
        in_head, in_tail = burndown.parse(GOOD)[0], burndown.parse(SECOND)[0]
        report_html = burndown.report(both, [], head_ids=head_ids)
    finally:
        burndown.REGISTER, burndown.HEAD, burndown.TAIL = saved

    if len(both) != len(in_head) + len(in_tail):
        failures.append("load() returned %d entries from a %d + %d split; it must return BOTH files"
                        % (len(both), len(in_head), len(in_tail)))
    if head_ids != set(e["id"] for e in in_head):
        failures.append("load() reported the wrong head ids: %r" % (head_ids,))

    # THE REPORT MUST RENDER HEAD ONLY. Dropping the trim puts the whole
    # register back on the page and nothing else changes, so no count moves.
    for e in in_head:
        if e.get("status") != "open":
            continue
        if e["id"] not in report_html:
            failures.append("the report dropped %s, which is open in the current iteration"
                            % e["id"])
    for e in in_tail:
        if e.get("status") == "open" and ('>%s<' % e["id"]) in report_html:
            failures.append("the report rendered %s, which is in tail.toml and is NOT this "
                            "iteration" % e["id"])
    if "Closed \u2014 what the evidence was" in report_html:
        failures.append("the split report still carries the closed-entry section; it must stop "
                        "after the open list")
    html = burndown.report(entries, [{"date": "2026-09-09", "open": 1, "total": 2, "note": "x"},
                                     {"date": "2026-09-10", "open": 0, "total": 2, "note": "y"}])
    for want in ("<title>lights burn-down</title>", "L-1", "polyline", "prefers-color-scheme",
                 "state-absent"):
        if want not in html:
            failures.append("report is missing %r" % want)
    if failures:
        print("\n".join(failures))
        return 1
    print("burndown checker: %d deliberate breaks, all caught" % len(BREAKS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
