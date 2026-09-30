import datetime
import html
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CRATE = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CRATE))
REGISTER = os.path.join(CRATE, "BURNDOWN.toml")
HEAD = os.path.join(CRATE, "head_open.toml")
TAIL = os.path.join(CRATE, "tail.toml")
HISTORY = os.path.join(ROOT, "docs", "lights", "burndown_history.json")

REQUIRED = ("id", "title", "area", "kind", "state", "status", "raised", "size", "why", "next", "where")
KINDS = ("feature", "defect", "owed", "spec", "research", "cleanup", "decision", "walk")
STATUS = ("open", "closed")
STATES = ("built", "partial", "absent", "pruned", "blocked", "unverified")
SIZES = ("small", "medium", "large")
AREAS = ("kernel", "method", "fit", "core", "instrument", "tooling", "unlanded", "docs", "gap",
         "backlog")
CLOSED_WORDS = ("FIXED", "BUILT", "CLOSED", "DROPPED", "MEASURED")
KIND_ORDER = {k: i for i, k in enumerate(("defect", "owed", "research", "spec", "feature", "cleanup", "walk", "decision"))}
FIELD_ENDS = '.!?"\')]:'
ID = re.compile(r"^L-(\d+)$")
SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{1,40}$")
PAIR = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"(.*)"\s*$')
BLOCKER = re.compile(r"\bBLOCKED(?:\s+\d{4}-\d{2}-\d{2})?\s+(?:ON|BY|BEHIND)\s+((?:L-\d+(?:\s*(?:,|and|or|\+)\s*)?)+)",
                     re.IGNORECASE)


def unescape(v):
    out, i = [], 0
    while i < len(v):
        c = v[i]
        if c == "\\" and i + 1 < len(v):
            n = v[i + 1]
            out.append({"n": "\n", "t": "\t", '"': '"', "\\": "\\"}.get(n, "\\" + n))
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


def escape(v):
    return v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t")


def parse(text):
    entries, bad, cur = [], [], None
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line == "[[entry]]":
            cur = {"_line": n}
            entries.append(cur)
            continue
        if line.startswith("[") and line.endswith("]"):
            cur = None
            continue
        m = PAIR.match(line)
        if m and cur is not None:
            cur[m.group(1)] = unescape(m.group(2))
        else:
            bad.append((n, line))
    return entries, bad


def dump(entries, header):
    order = ("id", "title", "area", "kind", "state", "status", "raised", "size", "cluster", "where", "evidence", "why", "next")
    out = [header.rstrip(), ""]
    for e in entries:
        out.append("[[entry]]")
        for k in order:
            if e.get(k):
                out.append('%s = "%s"' % (k, escape(e[k])))
        for k in sorted(e):
            if k not in order and not k.startswith("_"):
                out.append('%s = "%s"' % (k, escape(e[k])))
        out.append("")
    return "\n".join(out) + "\n"


def check(entries, bad, today=None):
    problems, seen, titles = [], {}, {}
    today = today or datetime.date.today()
    for e in entries:
        eid = e.get("id", "?")
        at = e.get("_line", 0)
        if not ID.match(eid or ""):
            problems.append("line %d: id %r is not L-<n>" % (at, eid))
        if eid in seen:
            problems.append("duplicate id %s (lines %d and %d)" % (eid, seen[eid], at))
        seen[eid] = at
        key = re.sub(r"[^a-z0-9 ]", "", (e.get("title") or "").lower()).strip()
        if key and key in titles:
            problems.append("%s repeats the title of %s word for word; two ids for one fact is paid for twice"
                            % (eid, titles[key]))
        titles[key] = eid
        missing = [f for f in REQUIRED if not e.get(f)]
        if missing:
            problems.append("%s (line %d) is missing: %s" % (eid, at, ", ".join(missing)))
        for field, allowed in (("kind", KINDS), ("status", STATUS), ("state", STATES), ("size", SIZES),
                               ("area", AREAS)):
            v = e.get(field)
            if v and v not in allowed:
                problems.append("%s: %s = %r is not one of %s" % (eid, field, v, "|".join(allowed)))
        try:
            raised = datetime.date.fromisoformat(e.get("raised", ""))
            if raised > today:
                problems.append("%s: raised %s is in the future" % (eid, raised))
        except ValueError:
            problems.append("%s: raised %r is not YYYY-MM-DD" % (eid, e.get("raised")))
        if e.get("cluster") and not SLUG.match(e["cluster"]):
            problems.append("%s: cluster %r is not a slug" % (eid, e["cluster"]))
        nxt = e.get("next", "")
        if e.get("status") == "closed" and not nxt.startswith(CLOSED_WORDS):
            problems.append("%s is closed but next does not open with %s" % (eid, "/".join(CLOSED_WORDS)))
        if (
            e.get("status") == "open"
            and "BLOCKED ON" in nxt.upper()
            and not is_blocked(e)
        ):
            problems.append(
                "%s says BLOCKED ON in its next but is invisible to is_blocked(): set "
                "state = \"blocked\" or open next with BLOCKED, or --next hands this "
                "entry to every context in turn" % eid
            )
        for field in ("why", "next"):
            v = e.get(field, "")
            if len(v) > 120 and v[-1] not in FIELD_ENDS:
                problems.append("%s: %s stops mid-sentence (%r)" % (eid, field, v[-40:]))
    for e, ref, why in stale_blockers(entries):
        problems.append("%s is blocked on %s, which %s; unblock it or name what it waits on now"
                        % (e.get("id", "?"), ref, why))
    for slug, members in clusters(entries).items():
        if len(members) < 2:
            problems.append("cluster %r is carried by one entry (%s); a cluster of one is a label" % (slug, members[0]["id"]))
    for n, line in bad:
        problems.append("line %d could not be read: %s" % (n, line[:70]))
    return problems


def blockers(e):
    out = []
    for m in BLOCKER.finditer(e.get("next", "")):
        for ref in re.findall(r"L-\d+", m.group(1)):
            if ref not in out:
                out.append(ref)
    return out


def stale_blockers(entries):
    by_id = {e.get("id"): e for e in entries}
    out = []
    for e in entries:
        if e.get("status") != "open" or not is_blocked(e):
            continue
        for ref in blockers(e):
            target = by_id.get(ref)
            if target is None:
                out.append((e, ref, "is not in the register"))
            elif target.get("status") == "closed":
                out.append((e, ref, "is closed (%s)" % target.get("state", "?")))
    return out


def clusters(entries):
    out = {}
    for e in entries:
        if e.get("cluster"):
            out.setdefault(e["cluster"], []).append(e)
    return out


def standing(entries):
    opens = [e for e in entries if e.get("status") == "open"]
    return len(opens), len(entries)


def rank(e):
    m = ID.match(e.get("id", "L-0"))
    return (KIND_ORDER.get(e.get("kind"), 99), int(m.group(1)) if m else 0)


def is_blocked(e):
    return e.get("state") == "blocked" or e.get("next", "").lstrip().upper().startswith("BLOCKED")


def blocked_open(entries):
    return sorted([e for e in entries if e.get("status") == "open" and is_blocked(e)], key=rank)


def oldest_open(entries):
    opens = sorted([e for e in entries if e.get("status") == "open" and not is_blocked(e)], key=rank)
    return opens[0] if opens else None


def load_history():
    try:
        with open(HISTORY) as fh:
            return json.load(fh)
    except (IOError, ValueError):
        return []


def append_history(entries, note, today=None):
    rows = load_history()
    o, t = standing(entries)
    day = (today or datetime.date.today()).isoformat()
    rows.append({"date": day, "open": o, "total": t, "note": note})
    with open(HISTORY, "w") as fh:
        json.dump(rows, fh, indent=1)
        fh.write("\n")
    return o, t


CSS = """
:root{
  --paper:#EEF1F4; --card:#FFFFFF; --ink:#141A20; --muted:#5C6874; --line:#D6DDE4;
  --open:#0E6E86; --closed:#657282; --defect:#A33221; --blocked:#9A5B08; --grid:#E2E7EC;
  --shadow:0 1px 0 rgba(20,26,32,.05);
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --paper:#0F141A; --card:#161D25; --ink:#E3EAF1; --muted:#93A2B0; --line:#26313C;
    --open:#4FB6D0; --closed:#7C8B9B; --defect:#E0765F; --blocked:#D69A3C; --grid:#1E2833;
    --shadow:none;
  }
}
:root[data-theme="dark"]{
  --paper:#0F141A; --card:#161D25; --ink:#E3EAF1; --muted:#93A2B0; --line:#26313C;
  --open:#4FB6D0; --closed:#7C8B9B; --defect:#E0765F; --blocked:#D69A3C; --grid:#1E2833;
  --shadow:none;
}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
  font-family:"IBM Plex Sans","Helvetica Neue",Arial,sans-serif;font-size:15px;line-height:1.55}
.wrap{max-width:1080px;margin:0 auto;padding:40px 24px 96px}
.eyebrow{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11px;letter-spacing:.18em;
  text-transform:uppercase;color:var(--muted)}
h1{font-size:34px;line-height:1.1;margin:6px 0 2px;font-weight:600;letter-spacing:-.01em;text-wrap:balance}
h2{font-size:13px;font-family:"IBM Plex Mono",ui-monospace,monospace;letter-spacing:.14em;
  text-transform:uppercase;color:var(--muted);font-weight:500;margin:44px 0 12px}
.sub{color:var(--muted);max-width:66ch;margin:8px 0 0}
.count{font-family:"IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums;
  font-size:44px;font-weight:600;letter-spacing:-.02em}
.count small{font-size:15px;font-weight:400;color:var(--muted);letter-spacing:0}
.head{display:flex;gap:32px;align-items:flex-end;flex-wrap:wrap;justify-content:space-between;
  border-bottom:1px solid var(--line);padding-bottom:20px}
.panel{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:18px 20px;
  box-shadow:var(--shadow)}
.chart{overflow-x:auto}
.bars{display:flex;flex-direction:column;gap:7px;margin:0;padding:0;list-style:none}
.bars li{display:grid;grid-template-columns:110px 1fr 78px;gap:12px;align-items:center}
.bars .nm{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12px;color:var(--muted)}
.track{height:11px;background:var(--grid);border-radius:2px;overflow:hidden;display:flex}
.track i{display:block;height:100%}
.track .o{background:var(--open)}
.track .c{background:var(--closed);opacity:.45}
.bars .no{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12px;
  font-variant-numeric:tabular-nums;color:var(--muted);text-align:right}
.rows{display:flex;flex-direction:column;gap:0;border-top:1px solid var(--line)}
details.row{border-bottom:1px solid var(--line);padding:0}
details.row>summary{list-style:none;cursor:pointer;display:grid;
  grid-template-columns:52px 1fr auto;gap:14px;align-items:baseline;padding:11px 4px}
details.row>summary::-webkit-details-marker{display:none}
details.row[open]>summary{background:transparent}
.id{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12px;color:var(--muted);
  font-variant-numeric:tabular-nums}
.ttl{font-weight:500}
details.row.closed .ttl{color:var(--muted);text-decoration:line-through;text-decoration-color:var(--line)}
.tags{display:flex;gap:6px;align-items:center;flex-wrap:wrap;justify-content:flex-end}
.chip{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:10.5px;letter-spacing:.06em;
  text-transform:uppercase;padding:2px 7px;border-radius:3px;border:1px solid var(--line);color:var(--muted)}
.chip.defect{color:var(--defect);border-color:var(--defect)}
.chip.blocked{color:var(--blocked);border-color:var(--blocked)}
.chip.done{color:var(--closed);border-color:var(--closed)}
.chip.state-absent{color:var(--defect);border-color:var(--defect);opacity:.85}
.chip.state-partial{color:var(--blocked);border-color:var(--blocked);opacity:.85}
.chip.state-built{color:var(--open);border-color:var(--open);opacity:.85}
.body{padding:2px 4px 18px 66px;display:flex;flex-direction:column;gap:10px;
  max-width:78ch;color:var(--ink)}
.body b{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:10.5px;letter-spacing:.1em;
  text-transform:uppercase;color:var(--muted);font-weight:500;display:block;margin-bottom:2px}
.body code,.mono{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12.5px}
.body p{margin:0}
.foot{margin-top:56px;padding-top:18px;border-top:1px solid var(--line);color:var(--muted);font-size:13px}
a{color:var(--open)}
@media (max-width:620px){
  .bars li{grid-template-columns:84px 1fr 62px}
  .body{padding-left:4px}
  details.row>summary{grid-template-columns:46px 1fr;grid-template-rows:auto auto}
  .tags{grid-column:1/-1;justify-content:flex-start}
}
"""


def esc(v):
    return html.escape(v or "")


def linkify(v):
    return re.sub(r"(`[^`]+`)", lambda m: "<code>%s</code>" % esc(m.group(1)[1:-1]), esc(v)).replace("&#x27;", "'")


def chart(rows):
    if len(rows) < 2:
        return '<p class="sub">The burn-down line needs two readings; there is %d.</p>' % len(rows)
    w, h, pad_l, pad_r, pad_t, pad_b = 960, 220, 44, 16, 14, 34
    top = max(max(r["total"] for r in rows), 1)
    n = len(rows)
    x = lambda i: pad_l + (w - pad_l - pad_r) * (i / (n - 1) if n > 1 else 0)
    y = lambda v: pad_t + (h - pad_t - pad_b) * (1 - v / top)
    parts = ['<svg viewBox="0 0 %d %d" width="100%%" style="min-width:640px;display:block" role="img" aria-label="open entries over time">' % (w, h)]
    for k in range(5):
        v = top * k / 4.0
        parts.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="var(--grid)" stroke-width="1"/>' % (pad_l, y(v), w - pad_r, y(v)))
        parts.append('<text x="%d" y="%.1f" fill="var(--muted)" font-size="11" font-family="IBM Plex Mono,monospace" text-anchor="end">%d</text>' % (pad_l - 8, y(v) + 4, round(v)))
    total_pts = " ".join("%.1f,%.1f" % (x(i), y(r["total"])) for i, r in enumerate(rows))
    open_pts = " ".join("%.1f,%.1f" % (x(i), y(r["open"])) for i, r in enumerate(rows))
    parts.append('<polyline points="%s" fill="none" stroke="var(--closed)" stroke-width="1.5" stroke-dasharray="3 3"/>' % total_pts)
    parts.append('<polyline points="%s" fill="none" stroke="var(--open)" stroke-width="2.5"/>' % open_pts)
    for i, r in enumerate(rows):
        parts.append('<circle cx="%.1f" cy="%.1f" r="3" fill="var(--open)"/>' % (x(i), y(r["open"])))
    for i in (0, n - 1):
        anchor = "start" if i == 0 else "end"
        parts.append('<text x="%.1f" y="%d" fill="var(--muted)" font-size="11" font-family="IBM Plex Mono,monospace" text-anchor="%s">%s</text>' % (x(i), h - 10, anchor, esc(rows[i]["date"])))
    parts.append('<text x="%d" y="%d" fill="var(--closed)" font-size="11" font-family="IBM Plex Mono,monospace" text-anchor="middle">total %d \u00b7 open %d</text>' % (w // 2, h - 10, rows[-1]["total"], rows[-1]["open"]))
    parts.append("</svg>")
    return '<div class="chart">%s</div>' % "".join(parts)


def area_bars(entries):
    rows = []
    for area in AREAS:
        mine = [e for e in entries if e.get("area") == area]
        if not mine:
            continue
        o = len([e for e in mine if e.get("status") == "open"])
        rows.append((area, o, len(mine)))
    widest = max([t for _, _, t in rows] or [1])
    out = ['<ul class="bars">']
    for area, o, t in sorted(rows, key=lambda r: -r[1]):
        out.append('<li><span class="nm">%s</span><span class="track">'
                   '<i class="o" style="width:%.2f%%"></i><i class="c" style="width:%.2f%%"></i>'
                   '</span><span class="no">%d of %d</span></li>'
                   % (esc(area), 100.0 * o / widest, 100.0 * (t - o) / widest, o, t))
    out.append("</ul>")
    return "".join(out)


def row_html(e):
    kind = e.get("kind", "")
    closed = e.get("status") == "closed"
    blocked = is_blocked(e)
    chips = ['<span class="chip %s">%s</span>' % ("defect" if kind == "defect" else "", esc(kind))]
    chips.append('<span class="chip state-%s">%s</span>' % (esc(e.get("state", "")), esc(e.get("state", ""))))
    chips.append('<span class="chip">%s</span>' % esc(e.get("size", "")))
    if e.get("cluster"):
        chips.append('<span class="chip">%s</span>' % esc(e["cluster"]))
    if blocked and not closed:
        chips.append('<span class="chip blocked">blocked</span>')
    if closed:
        chips.append('<span class="chip done">closed</span>')
    body = []
    for label, field in (("why", "why"), ("next", "next"), ("where", "where"), ("evidence", "evidence")):
        if e.get(field):
            body.append("<p><b>%s</b>%s</p>" % (label, linkify(e[field])))
    return ('<details class="row%s"><summary><span class="id">%s</span>'
            '<span class="ttl">%s</span><span class="tags">%s</span></summary>'
            '<div class="body">%s</div></details>'
            % (" closed" if closed else "", esc(e.get("id", "")), esc(e.get("title", "")),
               "".join(chips), "".join(body)))


def report(entries, rows, today=None, head_ids=None):
    o, t = standing(entries)
    day = (today or datetime.date.today()).isoformat()
    opens = sorted([e for e in entries if e.get("status") == "open"], key=rank)
    dones = sorted([e for e in entries if e.get("status") == "closed"], key=rank)
    if head_ids is not None:
        opens = [e for e in opens if e.get("id") in head_ids]
        dones = []
    out = ["<title>lights burn-down</title>",
           '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>',
           '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">',
           "<style>%s</style>" % CSS,
           '<div class="wrap">',
           '<div class="head"><div><p class="eyebrow">crates/lights \u00b7 %s</p><h1>The burn-down</h1>'
           '<p class="sub">Every feature of the renderer, what state it is measured to be in, and the '
           'first act that moves it. The register is <code>crates/lights/head_open.toml</code> (this '
           'iteration) plus <code>crates/lights/tail.toml</code> (the queue and the archive); this '
           'page is generated from them and is redeployed the moment an entry changes state.</p></div>'
           '<div><p class="count">%d<small> open</small></p><p class="eyebrow">of %d entries</p></div></div>'
           % (day, o, t),
           '<h2>Open entries over time</h2><div class="panel">%s</div>' % chart(rows),
           '<h2>By area \u2014 open, then closed</h2><div class="panel">%s</div>' % area_bars(entries),
           "<h2>Open \u2014 defects first, then oldest</h2>",
           ('<p class="sub">This is the CURRENT ITERATION only — %d of the %d open entries. '
            'The rest of the queue and every closed entry are in '
            '<code>crates/lights/tail.toml</code>, which this page deliberately does not '
            'render and no context needs to open. <code>scripts/split_register.py --roll</code> '
            'refills this list when an iteration ends.</p>' % (len(opens), o))
           if head_ids is not None else "",
           '<div class="rows">%s</div>' % ("".join(row_html(e) for e in opens) or '<p class="sub">Nothing open.</p>')]
    if dones:
        out.append("<h2>Closed \u2014 what the evidence was</h2>")
        out.append('<div class="rows">%s</div>' % "".join(row_html(e) for e in dones))
    out.append('<p class="foot">Generated by <code>crates/lights/scripts/burndown.py</code> from '
               '<code>%s</code> at %s. A closed entry whose report was not redeployed is not '
               'closed.</p>'
               % ("head_open.toml + tail.toml" if head_ids is not None else "BURNDOWN.toml", day))
    out.append("</div>")
    return "\n".join(out)


def is_split():
    return os.path.exists(HEAD) and os.path.exists(TAIL)


def load():
    """Every entry, from the split pair when it exists and from the single
    register otherwise. head comes first so ranking ties resolve toward the
    current iteration. The second value is the set of ids in head, or None when
    the register is not split."""
    if is_split():
        head, bad_h = parse(open(HEAD).read())
        tail, bad_t = parse(open(TAIL).read())
        return head + tail, bad_h + bad_t, set(e.get("id") for e in head)
    if not os.path.exists(REGISTER):
        return [], [], None
    entries, bad = parse(open(REGISTER).read())
    return entries, bad, None


def main(argv):
    present = is_split() or os.path.exists(REGISTER)
    entries, bad, head_ids = load()
    if "--check" in argv:
        if not present:
            print("neither %s nor the head/tail pair exists — that is not an empty "
                  "register and it is not a finished one" % REGISTER)
            return 1
        if not entries:
            print("the register holds no [[entry]] block — an empty register is not a "
                  "finished one")
            return 1
        problems = check(entries, bad)
        o, t = standing(entries)
        print("\n".join(problems) if problems else "clean")
        print("%d open of %d" % (o, t))
        return 1 if problems else 0
    if "--next" in argv:
        e = oldest_open(entries)
        waiting = blocked_open(entries)
        if not e:
            print("REGISTER DRY")
            if waiting:
                print("blocked   %d open %s wait on a PM ruling: %s"
                      % (len(waiting), "entry" if len(waiting) == 1 else "entries",
                         ", ".join(b["id"] for b in waiting)))
            return 0
        for k in ("id", "title", "area", "kind", "size", "cluster", "where", "evidence", "why", "next"):
            if e.get(k):
                print("%-9s %s" % (k, e[k]))
        if waiting:
            print("skipped   %s (blocked, waiting on a PM ruling)" % ", ".join(b["id"] for b in waiting))
        return 0
    if "--clusters" in argv:
        cl = clusters(entries)
        if not cl:
            print("no clusters")
        for slug, members in sorted(cl.items()):
            live = [m for m in members if m.get("status") == "open"]
            print("%s \u2014 %d members, %d live" % (slug, len(members), len(live)))
            for m in sorted(members, key=rank):
                print("    %-6s %-6s %s" % (m["id"], m.get("status"), m.get("title")))
        return 0
    if "--history" in argv:
        note = argv[argv.index("--history") + 1] if len(argv) > argv.index("--history") + 1 else "no note"
        o, t = append_history(entries, note)
        print("%s: %d open of %d" % (HISTORY, o, t))
        return 0
    sys.stdout.write(report(entries, load_history(), head_ids=head_ids))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
