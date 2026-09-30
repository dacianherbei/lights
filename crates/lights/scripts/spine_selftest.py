import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import spine  # noqa: E402

REC = {"records": [
    {"slug": "zz-later-same-day", "date": "2026-01-01", "title": "second",
     "refuted": ["b2"], "verdict": "v2", "findings": ["f2"], "numbers": [], "command": "x"},
    {"slug": "aa-first-same-day", "date": "2026-01-01", "title": "first",
     "refuted": ["b1"], "verdict": "v1", "findings": ["f1"], "numbers": [], "command": "x"},
    {"slug": "l999-links-an-entry", "date": "2026-01-02", "title": "third",
     "refuted": ["b3"], "verdict": "v3", "findings": [], "numbers": [], "command": "x"},
    {"slug": "p2-must-not-become-L-2", "date": "2026-01-03", "title": "fourth",
     "refuted": ["b4"], "verdict": "v4", "findings": [], "numbers": [], "command": "x"},
    {"slug": "nothing-refuted", "date": "2026-01-04", "title": "fifth",
     "refuted": [], "verdict": "v5", "findings": [], "numbers": [], "command": "x"},
]}

TSV = ("kind\tgroup\tid\tframe\ttitle\tshows\tregisters\tfile\n"
       "panel\tg\tpanel-a\t\tA panel\ts\tL-999\ta.png\n"
       "panel\tg\tpanel-b\t\tB panel\ts\tL-2\tb.png\n"
       "frame\tg\tf\t0\t\t\t\tf.png\n")


def main():
    fails = []

    def want(name, got, exp):
        if got != exp:
            fails.append("%s: expected %r, got %r" % (name, exp, got))

    tmp = tempfile.mkdtemp()
    tsv = os.path.join(tmp, "demo.tsv")
    open(tsv, "w").write(TSV)

    st = spine.steps(REC, tsv)

    want("a record that refuted nothing is not a step", len(st), 4)
    want("the record's own order breaks a date tie, not the slug",
         [s["slug"] for s in st[:2]], ["zz-later-same-day", "aa-first-same-day"])
    want("dates still order the spine", st[2]["date"], "2026-01-02")

    # THE FALSE LINK THIS TEST EXISTS FOR
    want("p2-… must NOT be read as entry L-2", st[3]["entry"], None)
    want("and so it reaches no panel", st[3]["panels"], [])
    want("an l<nn>- slug does link", spine.entry_of("l999-links-an-entry"), "L-999")
    want("a p<n>- slug does not", spine.entry_of("p2-crisp-ui"), None)
    want("an a<n>- slug does not", spine.entry_of("a1-atom-curve"), None)

    # panels carry (id, title) so the page can emit a working data-id
    p = spine.panels_by_entry(tsv)
    want("panels are keyed by entry", sorted(p), ["L-2", "L-999"])
    want("and carry the id the picker needs", p["L-999"], [("panel-a", "A panel")])
    want("a frame row is not a panel", all(k != "frame" for k in p), True)
    want("a missing tsv yields no panels, not a crash", spine.panels_by_entry(None), {})

    problems, steps_, silent = spine.check(REC, tsv)
    want("a sound record checks clean", problems, [])
    want("and the silent record is counted", silent, 1)

    bad = {"records": [{"slug": "l1-x", "date": "2026-01-01", "title": "t",
                        "refuted": ["b"], "verdict": "", "findings": [], "numbers": []}]}
    problems, _, _ = spine.check(bad, tsv)
    want("a step with no verdict is refused", any("verdict" in p for p in problems), True)

    empty, _, _ = spine.check({"records": []}, tsv)
    want("a record with no steps at all is refused", any("NO steps" in p for p in empty), True)

    if fails:
        for f in fails:
            print("FAIL " + f)
        return 1
    print("spine: 15 deliberate cases, all correct")
    return 0


if __name__ == "__main__":
    sys.exit(main())
