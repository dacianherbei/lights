"""THE SPINE - what the crate believed, what it measured, and what that changed.

The demo page shows 51 conclusions. This derives the ARGUMENT under them: every experiment
in docs/lights/RESEARCH.json refutes something, most often the project's own prior belief,
and each refutation is why some part of the renderer looks the way it does.

Nothing here is authored. Every field is lifted from the record, and a step the record cannot
support is REFUSED rather than filled in - see `--check`.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CRATE = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CRATE))
sys.path.insert(0, HERE)
import burndown  # noqa: E402

RECORD = os.path.join(ROOT, "docs", "lights", "RESEARCH.json")
# A slug links to an entry ONLY in the form l<digits>-. `p2-crisp-ui` does NOT mean L-2:
# that regex was written, it matched, and it invented a link to an unrelated entry. The
# seven experiments with no id in their slug carry no entry rather than a wrong one.
ENTRY_IN_SLUG = re.compile(r"^l(\d+)-")


def entry_of(slug):
    m = ENTRY_IN_SLUG.match(slug)
    return ("L-%s" % m.group(1)) if m else None


def first_sentence(text, limit=260):
    t = " ".join((text or "").split())
    if len(t) <= limit:
        return t
    cut = t.rfind(". ", 0, limit)
    return t[:cut + 1] if cut > 80 else t[:limit].rstrip() + "…"


def panels_by_entry(tsv_path):
    """entry id -> [panel title]. The demo already carries the link; nothing new is invented."""
    out = {}
    if not tsv_path or not os.path.isfile(tsv_path):
        return out
    rows = [l.split("\t") for l in open(tsv_path).read().splitlines() if not l.startswith("#")]
    if not rows:
        return out
    hdr = rows[0]
    for col in ("registers", "kind", "title", "id"):
        if col not in hdr:
            return out
    if "id" not in hdr:
        return out
    ri, ki, ti, ii = (hdr.index("registers"), hdr.index("kind"),
                      hdr.index("title"), hdr.index("id"))
    for r in rows[1:]:
        if len(r) <= max(ri, ki, ti, ii) or r[ki] != "panel":
            continue
        for x in r[ri].replace(";", ",").split(","):
            x = x.strip()
            if x:
                out.setdefault(x, []).append((r[ii], r[ti]))
    return out


def steps(record=None, tsv_path=None):
    doc = record or json.load(open(RECORD, encoding="utf-8"))
    entries, _, _ = burndown.load()
    by_id = {e.get("id"): e for e in entries}
    panels = panels_by_entry(tsv_path)

    out = []
    # Date first, then the record's OWN order - not the slug. Alphabetical put a1-atom-curve
    # before p2-crisp-ui on 2026-09-08, and P2 is the premise refutation the rest follows from.
    ordered = sorted(enumerate(doc.get("records", [])),
                     key=lambda pair: (pair[1].get("date", ""), pair[0]))
    for _, rec in ordered:
        refuted = [r for r in (rec.get("refuted") or []) if r and r.strip()]
        if not refuted:
            continue                      # a step with nothing refuted is not an argument
        eid = entry_of(rec.get("slug", ""))
        entry = by_id.get(eid) if eid else None
        out.append({
            "date": rec.get("date", ""),
            "slug": rec.get("slug", ""),
            "title": rec.get("title", ""),
            "believed": first_sentence(refuted[0], 300),
            "also_refuted": [first_sentence(r, 180) for r in refuted[1:4]],
            "measured": first_sentence(rec.get("verdict") or rec.get("answer"), 340),
            "numbers": [(n.get("label", ""), n.get("value", ""))
                        for n in (rec.get("numbers") or [])[:3]],
            "changed": [first_sentence(f, 200) for f in (rec.get("findings") or [])[:2]],
            "entry": eid if entry else None,
            "entry_title": entry.get("title") if entry else None,
            "entry_status": entry.get("status") if entry else None,
            "panels": panels.get(eid, []) if eid else [],
            "command": bool((rec.get("command") or "").strip()),
        })
    return out


def check(record=None, tsv_path=None):
    problems = []
    doc = record or json.load(open(RECORD, encoding="utf-8"))
    st = steps(doc, tsv_path)
    if not st:
        problems.append("the record yielded NO steps - a spine with no argument is not a spine")
    for s in st:
        if not s["believed"]:
            problems.append("%s: a step with no refuted belief" % s["slug"])
        if not s["measured"]:
            problems.append("%s: a step with no verdict - what was measured must be stated"
                            % s["slug"])
        if s["entry"] and not s["entry_title"]:
            problems.append("%s: names entry %s, which is not in the register" % (s["slug"], s["entry"]))
    silent = len(doc.get("records", [])) - len(st)
    return problems, st, silent


def main(argv):
    tsv = None
    for i, a in enumerate(argv):
        if a == "--demo" and i + 1 < len(argv):
            tsv = argv[i + 1]
    problems, st, silent = check(tsv_path=tsv)
    if "--check" in argv:
        print("\n".join(problems) if problems else "clean")
        print("%d step(s) of argument; %d experiment(s) refuted nothing and are not steps"
              % (len(st), silent))
        with_entry = len([s for s in st if s["entry"]])
        with_panel = len([s for s in st if s["panels"]])
        print("%d step(s) name a register entry; %d reach a demo panel" % (with_entry, with_panel))
        return 1 if problems else 0
    if "--json" in argv:
        print(json.dumps(st, indent=1))
        return 0
    for s in st:
        print("%s  %s" % (s["date"], s["slug"]))
        print("   BELIEVED  %s" % s["believed"][:150])
        print("   MEASURED  %s" % s["measured"][:150])
        if s["entry"]:
            print("   ENTRY     %s (%s)" % (s["entry"], s["entry_status"]))
        if s["panels"]:
            print("   PANELS    %s" % ", ".join(t[:40] for _, t in s["panels"]))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
