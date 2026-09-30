#!/usr/bin/env python3
"""Emit a linear, single-file, print-ready document from docs/lights/RESEARCH.json.

Destination in the repo: crates/lights/scripts/build_research_document.py
Output:                  docs/lights/RESEARCH_DOCUMENT.html

Why a separate form at all: docs/lights/pages/research.html is a split-pane
application with a tree and a detail pane, driven by JavaScript and remembered in
localStorage. It is the right shape on a screen and the wrong shape on paper - only
the selected record is visible, so printing it yields one experiment. This emits
every record in reading order with no JavaScript, so a browser's Save as PDF
produces a document rather than a screenshot.

Single source of truth is RESEARCH.json, the same file the interactive page and
RESEARCH.md are generated from, so the three cannot disagree.
"""
import html
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("LIGHTS_ROOT",
                      os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
RECORD = os.path.join(ROOT, "docs", "lights", "RESEARCH.json")
OUT = os.path.join(ROOT, "docs", "lights", "RESEARCH_DOCUMENT.html")

sys.path.insert(0, os.path.join(ROOT, "crates", "lights", "scripts"))
import research

copy_pasteable = research.copy_pasteable

VERDICT_ORDER = {"PRUNE": 0, "RE-JUDGE": 1, "KEEP": 2}


def esc(v):
    return html.escape("" if v is None else str(v))


def text_of(node, *keys):
    """A section or finding may name its parts differently across revisions."""
    if isinstance(node, str):
        return node
    if isinstance(node, dict):
        for k in keys:
            if node.get(k):
                return node[k]
    return ""


def as_list(v):
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def bullets(items, cls=""):
    items = [i for i in (text_of(x, "text", "detail", "body", "note") or x for x in as_list(items)) if i]
    if not items:
        return ""
    lis = "".join("<li>%s</li>" % esc(i) for i in items)
    return '<ul class="f %s">%s</ul>' % (cls, lis)


def numbers_table(rows):
    rows = as_list(rows)
    if not rows:
        return ""
    body = "".join(
        "<tr><td>%s</td><td class=v>%s</td><td class=n>%s</td></tr>"
        % (esc(r.get("label")), esc(r.get("value")), esc(r.get("note")))
        for r in rows if isinstance(r, dict)
    )
    return ("<table class=nums><thead><tr><th>measurement</th><th>value</th>"
            "<th>where it comes from</th></tr></thead><tbody>%s</tbody></table>" % body)


def field(label, body):
    return '<div class="field"><b>%s</b>%s</div>' % (esc(label), body) if body else ""


def para(s):
    return "<p>%s</p>" % esc(s) if s else ""


def arms_section(arms):
    rows = as_list(arms) if not isinstance(arms, dict) else list(arms.values())
    if not rows:
        return ""
    rows.sort(key=lambda a: (VERDICT_ORDER.get(a.get("verdict"), 9), str(a.get("arm"))))
    tally = {}
    for a in rows:
        tally[a.get("verdict")] = tally.get(a.get("verdict"), 0) + 1
    head = "".join(
        "<tr><td class=mono>%s</td><td class='v verdict-%s'>%s</td><td>%s</td><td>%s</td></tr>"
        % (esc(a.get("arm")), esc(str(a.get("verdict")).lower().replace("-", "")),
           esc(a.get("verdict")), esc(a.get("provenance")), esc(a.get("transfers")))
        for a in rows
    )
    detail = "".join(
        "<div class=arm><h3>%s <span class='chip verdict-%s'>%s</span></h3>"
        "<p class=meta>provenance %s · transfers %s · %s</p>%s%s</div>"
        % (esc(a.get("arm")), esc(str(a.get("verdict")).lower().replace("-", "")), esc(a.get("verdict")),
           esc(a.get("provenance")), esc(a.get("transfers")),
           esc(", ".join(as_list(a.get("where")))),
           para(a.get("evidence")),
           field("records", esc(", ".join(as_list(a.get("records"))))))
        for a in rows
    )
    summary = " · ".join("%s %d" % (k, v) for k, v in sorted(tally.items(), key=lambda kv: VERDICT_ORDER.get(kv[0], 9)))
    return ('<h2>The arms, reviewed</h2><p class=lead>%s. A verdict of PRUNE is reserved for a reason that '
            'holds on the surviving target; RE-JUDGE marks an arm whose only killing evidence was measured '
            'against a target this crate no longer has.</p>'
            '<table class=nums><thead><tr><th>arm</th><th>verdict</th><th>provenance</th>'
            '<th>transfers</th></tr></thead><tbody>%s</tbody></table>%s' % (esc(summary), head, detail))


def record_section(r):
    return (
        '<article class=record><h3>%s</h3><p class=meta>%s · %s · %s</p>%s%s%s%s%s%s%s%s%s</article>'
        % (esc(r.get("title")), esc(r.get("group")), esc(r.get("date")), esc(r.get("slug")),
           field("the answer", para(r.get("verdict"))),
           field("the question it was built to answer", para(r.get("question"))),
           field("criterion, and its convention", para(r.get("criterion"))),
           field("reproduce it", "<pre class=cmd>%s</pre>" % esc(r.get("command")) if r.get("command") else ""),
           field("what it measured", numbers_table(r.get("numbers"))),
           field("what it settled", bullets(r.get("findings"))),
           field("what it refuted", bullets(r.get("refuted"), "refuted")),
           field("what it does not settle", bullets(r.get("not_settled"), "caveat")),
           field("sources", '<p class=srcs>%s</p>' % esc(" · ".join(as_list(r.get("sources")))))))


CSS = """
@page { size: A4; margin: 18mm 16mm 20mm 16mm; }
* { box-sizing: border-box }
body { font: 10.5pt/1.5 "Iowan Old Style", Georgia, "Times New Roman", serif;
       color: #111; background: #fff; margin: 0; padding: 0 }
.wrap { max-width: 190mm; margin: 0 auto; padding: 8mm 0 }
h1 { font-size: 24pt; line-height: 1.15; margin: 0 0 2mm; letter-spacing: -.01em }
h2 { font-size: 15pt; margin: 0 0 4mm; padding-bottom: 2mm; border-bottom: 1.5pt solid #111;
     page-break-before: always; page-break-after: avoid }
h3 { font-size: 11.5pt; margin: 0 0 1mm; page-break-after: avoid }
.cover { page-break-after: always; padding-top: 30mm }
.cover .sub { font-size: 12pt; color: #333; max-width: 130mm }
.cover dl { margin: 12mm 0 0; font-size: 9.5pt; color: #333 }
.cover dt { font-weight: 600; float: left; width: 34mm; clear: left }
.cover dd { margin: 0 0 1.5mm 34mm }
.lead { color: #333; max-width: 150mm }
.claim { margin: 0 0 3.5mm; max-width: 165mm; page-break-inside: avoid }
.cites { font-family: ui-monospace, Menlo, monospace; font-size: 7.5pt; color: #666;
         white-space: nowrap }
.meta { font-size: 8.5pt; color: #555; margin: 0 0 3mm; font-family: ui-monospace, "SF Mono", Menlo, monospace }
.mono, .srcs { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 8pt; word-break: break-word }
.field { margin: 0 0 3mm; page-break-inside: avoid }
.field > b { display: block; font-family: ui-monospace, "SF Mono", Menlo, monospace;
             font-size: 7.5pt; letter-spacing: .09em; text-transform: uppercase; color: #666;
             font-weight: 600; margin-bottom: .8mm }
.field p { margin: 0 }
pre.cmd { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 8pt; line-height: 1.4;
          background: #f4f4f2; border: .5pt solid #ddd; border-radius: 2pt; padding: 2.5mm 3mm;
          white-space: pre-wrap; word-break: break-word; margin: 0 }
table.nums { border-collapse: collapse; width: 100%; font-size: 8.5pt; margin: 0 }
table.nums th { text-align: left; font-family: ui-monospace, Menlo, monospace; font-size: 7.5pt;
                text-transform: uppercase; letter-spacing: .07em; color: #666; font-weight: 600;
                border-bottom: 1pt solid #111; padding: 1.5mm 2mm }
table.nums td { border-bottom: .5pt solid #ddd; padding: 1.5mm 2mm; vertical-align: top }
table.nums td.v { font-family: ui-monospace, Menlo, monospace; font-weight: 600 }
table.nums td.n { color: #444; font-size: 8pt }
table.nums tr { page-break-inside: avoid }
ul.f { margin: 0; padding-left: 5mm } ul.f li { margin: 0 0 1.2mm }
ul.f.refuted li::marker { color: #a33221 } ul.f.caveat li::marker { color: #8a6212 }
.chip { font-family: ui-monospace, Menlo, monospace; font-size: 7.5pt; border: .5pt solid #999;
        border-radius: 2pt; padding: .3mm 1.5mm; vertical-align: middle; color: #444 }
.verdictprune { color: #a33221; border-color: #a33221 }
.verdictrejudge { color: #8a6212; border-color: #8a6212 }
.verdictkeep { color: #2e6b3f; border-color: #2e6b3f }
.record, .arm { page-break-inside: auto; margin: 0 0 7mm }
.record + .record { border-top: .5pt solid #ddd; padding-top: 5mm }
.foot { margin-top: 8mm; padding-top: 3mm; border-top: .5pt solid #bbb; font-size: 8.5pt; color: #555 }
@media screen { body { padding: 6mm } .wrap { box-shadow: 0 0 0 1px #eee } }
"""


def build():
    if not os.path.isfile(RECORD):
        sys.exit("the research record is missing at %s" % RECORD)
    r = json.load(open(RECORD, encoding="utf-8"))
    records = as_list(r.get("records"))
    audit = as_list(r.get("audit"))
    arms = r.get("arms")
    report = r.get("report") or {}

    reproducible = sum(1 for x in records if copy_pasteable(x))
    n_arms = len(as_list(arms) if not isinstance(arms, dict) else list(arms.values()))

    cover = (
        '<section class=cover><h1>%s</h1><p class=sub>%s</p><dl>'
        '<dt>date</dt><dd>%s</dd><dt>tree</dt><dd class=mono>%s</dd>'
        '<dt>experiments</dt><dd>%d, of which %d carry a copy-pasteable command</dd>'
        '<dt>arms reviewed</dt><dd>%d</dd>'
        '<dt>cross-experiment</dt><dd>%d findings the records make about each other</dd>'
        '<dt>generated from</dt><dd class=mono>docs/lights/RESEARCH.json</dd></dl></section>'
        % (esc(report.get("title") or "The lights research record"),
           esc(text_of(report, "summary", "lead", "subtitle")
               or "Every experiment this crate has run, each with the command that reproduces it, "
                  "the criterion it was judged by, what it refuted, and what it does not settle."),
           esc(report.get("date") or ""), esc(report.get("tree") or ""),
           len(records), reproducible, n_arms, len(audit)))

    sections = ""
    for s in as_list(report.get("sections")):
        heading = text_of(s, "heading", "title", "name")
        body = ""
        for c in as_list(s.get("claims")):
            t = text_of(c, "text", "body")
            if not t:
                continue
            cites = as_list(c.get("cites"))
            body += '<p class=claim>%s%s</p>' % (
                esc(t),
                ' <span class=cites>%s</span>' % esc(" · ".join(str(x) for x in cites)) if cites else "")
        if not body:
            body = "".join(para(x) for x in str(text_of(s, "body", "text", "prose")).split("\n\n"))
        if heading or body:
            sections += "<h2>%s</h2>%s" % (esc(heading), body)

    recs = "<h2>The experiments</h2>" + "".join(record_section(x) for x in records)
    aud = ("<h2>What the records say about each other</h2>" +
           "".join('<article class=record><h3>%s</h3><p class=meta>%s · %s</p>%s</article>'
                   % (esc(a.get("title")), esc(a.get("kind")),
                      esc(", ".join(as_list(a.get("experiments")))), para(a.get("detail")))
                   for a in audit)) if audit else ""

    doc = ("<title>%s</title><style>%s</style><div class=wrap>%s%s%s%s%s<p class=foot>"
           "Generated by crates/lights/scripts/build_research_document.py from "
           "docs/lights/RESEARCH.json. The interactive form is docs/lights/pages/research.html; "
           "the reading form is docs/lights/RESEARCH.md. All three come from that one file.</p></div>"
           % (esc(report.get("title") or "The lights research record"), CSS,
              cover, sections, arms_section(arms), recs, aud))

    open(OUT, "w", encoding="utf-8").write(doc)
    print("wrote %s (%d bytes)" % (OUT, len(doc)))
    print("  %d experiments, %d reproducible, %d arms, %d cross-experiment findings"
          % (len(records), reproducible, n_arms, len(audit)))
    if not report:
        print("  NOTE: the record carries no `report` section yet, so the cover uses a default lead")
    missing = [x.get("slug") for x in records if not copy_pasteable(x)]
    if missing:
        print("  without a copy-pasteable command: %s" % ", ".join(str(m) for m in missing))


if __name__ == "__main__":
    build()
