import datetime
import html
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CRATE = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CRATE))
RECORD = os.path.join(ROOT, "docs", "lights", "RESEARCH.json")
MARKDOWN = os.path.join(ROOT, "docs", "lights", "RESEARCH.md")

REQUIRED = ("slug", "title", "group", "date", "verdict", "question", "command", "criterion",
            "numbers", "findings", "refuted", "not_settled", "artefacts", "sources")
LISTS = ("findings", "refuted", "not_settled", "artefacts", "sources")
GROUPS = ("representation", "transport", "prior-art", "engineering")
AUDIT_KINDS = ("contradiction", "not-comparable", "shared-confound", "unreproducible", "owed")
ARM_VERDICTS = ("RE-JUDGE", "PRUNE", "KEEP")
ARM_PROVENANCE = ("3d-scene", "target-free", "ui-crisp", "optical-frame", "glyph", "image", "none")
ARM_TRANSFERS = ("yes", "partly", "no")
ON_TARGET = ("3d-scene", "target-free")
EXPERIMENT_FIELDS = ("question", "arms", "criterion", "cost", "prunes")
ARM_KEY = re.compile(r"^[a-z0-9]+(/[a-z0-9][a-z0-9-]*)+$")
MANIFEST = os.path.join(CRATE, "scripts", "mutations.json")
SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{1,48}$")
NO_COMMAND = "NONE"
FIELD_ENDS = '.!?"\')]:'
REPORT_SECTIONS = ("question", "measured", "paradigm", "comparison", "stands", "open")
REPRO_HOW = ("full", "form")


def load(path=None):
    with open(path or RECORD) as fh:
        return json.load(fh)


def check(doc):
    problems, seen = [], {}
    today = datetime.date.today()
    records = doc.get("records", [])
    if not records:
        problems.append("the record holds no experiments — an empty consolidated document is not a finished one")
    for r in records:
        slug = r.get("slug", "?")
        if not SLUG.match(slug or ""):
            problems.append("%r is not a slug" % slug)
        if slug in seen:
            problems.append("duplicate slug %s — two ids for one experiment is paid for twice" % slug)
        seen[slug] = True
        for f in REQUIRED:
            if r.get(f) in (None, "", [], {}):
                problems.append("%s is missing: %s" % (slug, f))
        if r.get("group") and r["group"] not in GROUPS:
            problems.append("%s: group %r is not one of %s" % (slug, r["group"], "|".join(GROUPS)))
        try:
            when = datetime.date.fromisoformat(r.get("date", ""))
            if when > today:
                problems.append("%s: date %s is in the future" % (slug, when))
        except ValueError:
            problems.append("%s: date %r is not YYYY-MM-DD" % (slug, r.get("date")))
        for f in LISTS:
            v = r.get(f)
            if v is not None and not isinstance(v, list):
                problems.append("%s: %s must be a list" % (slug, f))
        if not r.get("sources"):
            problems.append("%s cites no source — a record nobody can trace is not a record" % slug)
        cmd = (r.get("command") or "").strip()
        if cmd and not cmd.startswith(NO_COMMAND) and "lights" not in cmd and "cargo" not in cmd:
            problems.append("%s: command %r names no binary; say %s plus the reason if there is none"
                            % (slug, cmd[:40], NO_COMMAND))
        for f in ("verdict", "question", "criterion"):
            v = (r.get(f) or "").strip()
            if len(v) > 120 and v[-1] not in FIELD_ENDS:
                problems.append("%s: %s stops mid-sentence (%r)" % (slug, f, v[-40:]))
        for n in r.get("numbers", []):
            for k in ("label", "value", "note"):
                if not n.get(k):
                    problems.append("%s: a number is missing its %s" % (slug, k))
    for a in doc.get("audit", []):
        if a.get("kind") not in AUDIT_KINDS:
            problems.append("audit %r: kind %r is not one of %s"
                            % (a.get("title", "?"), a.get("kind"), "|".join(AUDIT_KINDS)))
        for name in a.get("experiments", []):
            if name not in seen:
                problems.append("audit %r names %r, which is not an experiment in this record"
                                % (a.get("title", "?"), name))
    problems += check_arms(doc.get("arms", []), seen)
    problems += check_report(doc, seen)
    return problems


def squash(v):
    return " ".join((v or "").replace("\\\n", " ").split())


def check_report(doc, seen):
    rep = doc.get("report")
    if not rep:
        return []
    problems, cited, ran = [], set(), set()
    arms = {"arm:" + a.get("arm", "") for a in doc.get("arms", [])}
    keys = [s.get("key") for s in rep.get("sections", [])]
    if keys != list(REPORT_SECTIONS):
        problems.append("report sections are %s; the report answers %s, in that order"
                        % (keys, "|".join(REPORT_SECTIONS)))
    for s in rep.get("sections", []):
        if not s.get("claims"):
            problems.append("report section %s makes no claim" % s.get("key"))
        for c in s.get("claims", []):
            if not c.get("cites"):
                problems.append("report section %s has a claim that cites nothing: %r"
                                % (s.get("key"), (c.get("text") or "")[:40]))
            for x in c.get("cites", []):
                if x in seen:
                    cited.add(x)
                elif x not in arms:
                    problems.append("report cites %r, which is neither an experiment nor an arm in this record" % x)
    commands = {r.get("slug"): squash(r.get("command")) for r in doc.get("records", [])}
    for row in rep.get("reproduced", []):
        slug = row.get("record")
        if slug not in commands:
            problems.append("report reproduces %r, which is not an experiment in this record" % slug)
            continue
        ran.add(slug)
        if squash(row.get("invocation")) not in commands[slug]:
            problems.append("report reproduces %s with %r, which its own command does not contain"
                            % (slug, (row.get("invocation") or "")[:50]))
        if row.get("how") not in REPRO_HOW:
            problems.append("report reproduces %s as %r, which is not one of %s"
                            % (slug, row.get("how"), "|".join(REPRO_HOW)))
        if row.get("exit") != row.get("expected"):
            problems.append("report reproduces %s and it exited %r where its record expects %r"
                            % (slug, row.get("exit"), row.get("expected")))
    for slug in sorted(cited - ran):
        problems.append("report cites %s and records no reproduction of it on this tree" % slug)
    return problems


def check_arms(arms, seen):
    problems, keys = [], set()
    for a in arms:
        key = a.get("arm") or "?"
        if not ARM_KEY.match(key):
            problems.append("arm %r is not an axis/name key" % key)
        if key in keys:
            problems.append("arm %s is listed twice — one arm, one verdict" % key)
        keys.add(key)
        for field, allowed in (("verdict", ARM_VERDICTS), ("provenance", ARM_PROVENANCE),
                               ("transfers", ARM_TRANSFERS)):
            if a.get(field) not in allowed:
                problems.append("arm %s: %s %r is not one of %s"
                                % (key, field, a.get(field), "|".join(allowed)))
        if a.get("provenance") in ON_TARGET and a.get("transfers") != "yes":
            problems.append("arm %s: evidence taken on %s transfers by definition, so transfers must be yes"
                            % (key, a.get("provenance")))
        if not (a.get("evidence") or "").strip():
            problems.append("arm %s gives no evidence" % key)
        if not a.get("where"):
            problems.append("arm %s names no code" % key)
        for rel in a.get("where", []):
            if not os.path.exists(os.path.join(ROOT, rel)):
                problems.append("arm %s names %s, which does not exist" % (key, rel))
        for name in a.get("records", []):
            if name not in seen:
                problems.append("arm %s cites %r, which is not an experiment in this record" % (key, name))
        experiment = a.get("experiment") or {}
        if a.get("verdict") == "RE-JUDGE":
            for field in EXPERIMENT_FIELDS:
                if not (experiment.get(field) or "").strip():
                    problems.append("arm %s is RE-JUDGE and its experiment states no %s" % (key, field))
        elif experiment:
            problems.append("arm %s is %s and carries an experiment; only a RE-JUDGE waits on one"
                            % (key, a.get("verdict")))
    return problems


def arm_cost(a, manifest):
    lines, rows = 0, 0
    for rel in a.get("where", []):
        path = os.path.join(ROOT, rel)
        files = [path] if os.path.isfile(path) else [
            os.path.join(d, f) for d, _, names in os.walk(path) for f in names]
        for f in files:
            if f.endswith(".rs"):
                with open(f, encoding="utf-8", errors="replace") as fh:
                    lines += sum(1 for _ in fh)
        prefix = rel.rstrip("/") + "/"
        rows += sum(1 for m in manifest if m.get("file") == rel or m.get("file", "").startswith(prefix))
    return lines, rows


def load_manifest():
    if not os.path.exists(MANIFEST):
        return []
    with open(MANIFEST) as fh:
        return json.load(fh).get("mutations", [])


def copy_pasteable(r):
    first = (r.get("command") or "").strip().splitlines()
    head = first[0].strip() if first else ""
    return head.startswith(("cargo ", "python3 ", "bash ", "/", "./"))


def standing(doc):
    records = doc.get("records", [])
    return len(records), len([r for r in records if copy_pasteable(r)]), len(doc.get("audit", []))


def esc(v):
    return html.escape(v or "")


def linkify(v):
    return re.sub(r"(`[^`]+`)", lambda m: "<code>%s</code>" % esc(m.group(1)[1:-1]), esc(v)).replace("&#x27;", "'")


def markdown(doc, today=None):
    day = (today or datetime.date.today()).isoformat()
    total, runnable, audits = standing(doc)
    out = ["# lights — the research record", "",
           "Every experiment this crate has run, in one place, with the command that reproduces it.",
           "Generated from `docs/lights/RESEARCH.json` by `crates/lights/scripts/research.py`; edit the",
           "JSON, never this file. Interactive form, with the tree and the reading pane:",
           "<%s>" % doc.get("page", "unpublished"), "",
           "**%d experiments · %d with a command you can paste into a shell · %d cross-experiment "
           "findings · %s**"
           % (total, runnable, audits, day), "",
           "---", ""]
    if doc.get("report"):
        out += report_markdown(doc["report"])
    for group in GROUPS:
        mine = [r for r in doc.get("records", []) if r.get("group") == group]
        if not mine:
            continue
        out += ["## %s" % group, ""]
        for r in sorted(mine, key=lambda x: x.get("date", "")):
            out += ["### %s — %s" % (r.get("slug"), r.get("title")), "",
                    "**%s**" % r.get("verdict", ""), "",
                    "- **asked:** %s" % r.get("question", ""),
                    "- **date:** %s" % r.get("date", ""),
                    "- **criterion:** %s" % r.get("criterion", ""), "",
                    "```", (r.get("command") or "").strip(), "```", ""]
            if r.get("numbers"):
                out += ["| measurement | value | where it comes from |", "|---|---|---|"]
                for n in r["numbers"]:
                    out.append("| %s | **%s** | %s |" % (n.get("label"), n.get("value"), n.get("note")))
                out.append("")
            for label, field in (("What it settled", "findings"), ("What it refuted", "refuted"),
                                 ("What it does NOT settle", "not_settled")):
                if r.get(field):
                    out += ["**%s**" % label, ""]
                    out += ["- %s" % x for x in r[field]]
                    out.append("")
            if r.get("artefacts"):
                out += ["**Artefacts:** " + ", ".join("`%s`" % a for a in r["artefacts"]), ""]
            out += ["**Read from:** " + ", ".join("`%s`" % a for a in r.get("sources", [])), "", "---", ""]
    if doc.get("audit"):
        out += ["## What the experiments say about each other", "",
                "Found by reading the records against one another, which no single run could do.", ""]
        for a in doc["audit"]:
            out += ["### %s — %s" % (a.get("kind"), a.get("title")), "",
                    "*%s*" % ", ".join(a.get("experiments", [])), "", a.get("detail", ""), ""]
    if doc.get("arms"):
        out += arms_markdown(doc["arms"], load_manifest())
    return "\n".join(out) + "\n"


def report_markdown(rep):
    out = ["## %s" % rep.get("title", ""), "",
           "*%s, on tree %s. Every claim cites the experiments or arm rows it rests on, and every "
           "experiment it cites was re-run on that tree; the last table says how.*"
           % (rep.get("date", ""), rep.get("tree", "")), ""]
    for s in rep.get("sections", []):
        out += ["### %s" % s.get("heading", ""), ""]
        for c in s.get("claims", []):
            out += [c.get("text", ""), "", "*cites: %s*" % ", ".join("`%s`" % x for x in c.get("cites", [])), ""]
    rows = rep.get("reproduced", [])
    if rows:
        out += ["### How each cited experiment was reproduced", "",
                "| experiment | as run | exit | wall s | what was compared |", "|---|---|---|---|---|"]
        for r in rows:
            out.append("| %s | `%s` | %s | %s | %s |"
                       % (r.get("record"), (r.get("ran") or r.get("invocation") or "").replace("|", "\\|"),
                          r.get("exit"), r.get("wall_s"), (r.get("result") or "").replace("|", "\\|")))
        out.append("")
    return out + ["---", ""]


def arms_markdown(arms, manifest):
    out = ["## The arm audit — what stays, what goes, what is re-judged", "",
           "One row per arm. **Provenance** is the target the deciding measurement was taken against, and "
           "**transfers** says whether that evidence carries to the surviving mission — the 3D engine and "
           "the paradigm case. No arm is pruned on evidence that does not transfer, and a RE-JUDGE states "
           "the experiment that would decide it instead of running it. The two cost columns are counted "
           "from the tree when this file is generated: Rust lines under the row's paths (a file shared by "
           "two rows counts in both) and mutation rows aimed at them.", "",
           "| arm | verdict | provenance | transfers | Rust lines | mutation rows | records |",
           "|---|---|---|---|---|---|---|"]
    for verdict in ARM_VERDICTS:
        for a in [x for x in arms if x.get("verdict") == verdict]:
            lines, rows = arm_cost(a, manifest)
            out.append("| `%s` | **%s** | %s | %s | %d | %d | %s |"
                       % (a.get("arm"), verdict, a.get("provenance"), a.get("transfers"), lines, rows,
                          ", ".join(a.get("records", [])) or "—"))
    out.append("")
    for verdict in ARM_VERDICTS:
        for a in [x for x in arms if x.get("verdict") == verdict]:
            out += ["### `%s` — %s" % (a.get("arm"), verdict), "", a.get("evidence", ""), "",
                    "**Where:** " + ", ".join("`%s`" % w for w in a.get("where", [])), ""]
            experiment = a.get("experiment") or {}
            if experiment:
                out += ["**The experiment that would decide it:**", ""]
                out += ["- **%s:** %s" % (f, experiment.get(f, "")) for f in EXPERIMENT_FIELDS]
                out.append("")
    return out

CSS = """
:root{
  --paper:#F2F3F7; --card:#FFFFFF; --ink:#15181F; --muted:#5B6274; --line:#DBDEE7;
  --open:#3A4E93; --closed:#6A7188; --defect:#A33221; --blocked:#8A6212; --grid:#E5E7EE;
  --ok:#2E7D4F; --shadow:0 1px 0 rgba(20,24,40,.05);
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --paper:#0F1117; --card:#171A22; --ink:#E5E8F0; --muted:#949BB0; --line:#272B37;
    --open:#8FA6EE; --closed:#7E859B; --defect:#E0765F; --blocked:#D6A63C; --grid:#1E212B;
    --ok:#5CBB84; --shadow:none;
  }
}
:root[data-theme="dark"]{
  --paper:#0F1117; --card:#171A22; --ink:#E5E8F0; --muted:#949BB0; --line:#272B37;
  --open:#8FA6EE; --closed:#7E859B; --defect:#E0765F; --blocked:#D6A63C; --grid:#1E212B;
  --ok:#5CBB84; --shadow:none;
}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
  font-family:"IBM Plex Sans","Helvetica Neue",Arial,sans-serif;font-size:15px;line-height:1.58}
.wrap{max-width:1220px;margin:0 auto;padding:40px 24px 80px}
.eyebrow{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11px;letter-spacing:.18em;
  text-transform:uppercase;color:var(--muted);margin:0}
h1{font-size:34px;line-height:1.1;margin:6px 0 2px;font-weight:600;letter-spacing:-.012em;text-wrap:balance}
h2{font-size:13px;font-family:"IBM Plex Mono",ui-monospace,monospace;letter-spacing:.14em;
  text-transform:uppercase;color:var(--muted);font-weight:500;margin:40px 0 12px}
.sub{color:var(--muted);max-width:68ch;margin:10px 0 0}
.head{display:flex;gap:32px;align-items:flex-end;flex-wrap:wrap;justify-content:space-between;
  border-bottom:1px solid var(--line);padding-bottom:20px;margin-bottom:8px}
.count{font-family:"IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums;
  font-size:42px;font-weight:600;letter-spacing:-.02em;margin:0}
.count small{font-size:14px;font-weight:400;color:var(--muted);letter-spacing:0}
.chip{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:10.5px;letter-spacing:.06em;
  text-transform:uppercase;padding:2px 7px;border-radius:3px;border:1px solid var(--line);
  color:var(--muted);white-space:nowrap}
.chip.refuted{color:var(--defect);border-color:var(--defect)}
.chip.ok{color:var(--ok);border-color:var(--ok)}
.chip.warn{color:var(--blocked);border-color:var(--blocked)}
.hint{color:var(--muted);font-size:13.5px;margin:0 0 14px;max-width:78ch}
table.nums{border-collapse:collapse;width:100%;font-size:13px;margin:0 0 16px}
table.nums th{text-align:left;font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:10.5px;
  letter-spacing:.08em;text-transform:uppercase;color:var(--muted);font-weight:500;
  padding:6px 10px;border-bottom:1px solid var(--line)}
table.nums td{padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}
table.nums td.v{font-family:"IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums;
  font-weight:600;white-space:nowrap}
table.nums td.n{color:var(--muted);font-size:12.5px}
pre.cmd{background:var(--grid);border:1px solid var(--line);border-radius:4px;padding:11px 13px;
  overflow-x:auto;font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12.5px;margin:0 0 16px}
pre.cmd.none{color:var(--muted);background:transparent;border-style:dashed}
ul.f{margin:0 0 16px;padding-left:20px}
ul.f li{margin:0 0 7px}
ul.f.refuted li::marker{color:var(--defect)}
ul.f.caveat li::marker{color:var(--blocked)}
.srcs{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11.5px;color:var(--muted);
  margin:0;word-break:break-word}
.foot{margin-top:44px;padding-top:18px;border-top:1px solid var(--line);color:var(--muted);font-size:13px}
a{color:var(--open)}

.split{display:grid;grid-template-columns:var(--tree,320px) 7px minmax(0,1fr);
  border:1px solid var(--line);border-radius:6px;background:var(--card);
  box-shadow:var(--shadow);overflow:hidden;min-height:560px}
.tree{overflow:auto;max-height:76vh;padding:14px 6px 18px 14px;border-right:0}
.tree details{margin:0 0 2px}
.tree summary{list-style:none;cursor:pointer;padding:5px 6px;border-radius:4px;
  font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11.5px;letter-spacing:.06em;
  text-transform:uppercase;color:var(--muted);display:flex;gap:8px;align-items:center}
.tree summary::-webkit-details-marker{display:none}
.tree summary::before{content:"\25B8";font-size:9px;transition:transform .12s;display:inline-block}
.tree details[open]>summary::before{transform:rotate(90deg)}
.tree summary:hover{background:var(--grid);color:var(--ink)}
.tree .n{margin-left:auto;font-variant-numeric:tabular-nums;opacity:.7}
.tree .kids{margin:0 0 6px 12px;padding-left:8px;border-left:1px solid var(--line)}
.tree .leaf{display:grid;grid-template-columns:44px 1fr;gap:8px;width:100%;text-align:left;
  background:none;border:0;padding:5px 6px;border-radius:4px;cursor:pointer;color:var(--ink);
  font:inherit;font-size:12.5px;line-height:1.35;align-items:baseline}
.tree .leaf:hover{background:var(--grid)}
.tree .leaf .lid{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11px;color:var(--muted)}
.tree .leaf.sel{background:color-mix(in srgb,var(--open) 15%,transparent)}
.tree .leaf.sel .lid{color:var(--open);font-weight:600}
.tree .leaf.parked .lt{color:var(--muted);text-decoration:line-through;
  text-decoration-color:var(--line)}
.tree .leaf.refuted .lid{color:var(--defect)}
.tree .leaf:focus-visible,.tree summary:focus-visible{outline:2px solid var(--open);outline-offset:1px}
.bar{background:var(--line);cursor:col-resize;position:relative}
.bar::after{content:"";position:absolute;inset:0 -4px;}
.bar:hover,.bar:focus-visible{background:var(--open);outline:none}
.pane{overflow:auto;max-height:76vh;padding:22px 26px 34px}
.pane article{max-width:78ch}
.pane h3{font-size:17px;line-height:1.3;margin:0 0 4px;font-weight:600;text-wrap:balance}
.pane .meta{display:flex;gap:6px;flex-wrap:wrap;margin:0 0 16px}
.pane .field{margin:0 0 14px}
.pane .field b{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:10.5px;
  letter-spacing:.1em;text-transform:uppercase;color:var(--muted);font-weight:500;
  display:block;margin-bottom:3px}
.pane .field p{margin:0}
.pane code{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12.5px;
  background:var(--grid);padding:1px 4px;border-radius:3px}
.hint{color:var(--muted);font-size:13px;margin:0 0 14px}
@media (max-width:760px){
  .split{grid-template-columns:1fr;min-height:0}
  .bar{display:none}
  .tree{max-height:300px;border-bottom:1px solid var(--line)}
  .pane{max-height:none}
}
"""
SPLIT_JS = """
(function(){
  var root=document.getElementById('split'), bar=document.getElementById('bar');
  var leaves=[].slice.call(document.querySelectorAll('.leaf'));
  var panes=[].slice.call(document.querySelectorAll('.entry'));
  function get(k,d){try{var v=localStorage.getItem(k);return v===null?d:v}catch(e){return d}}
  function set(k,v){try{localStorage.setItem(k,v)}catch(e){}}
  function show(id){
    var found=false;
    panes.forEach(function(p){var on=p.id===id;p.hidden=!on;if(on)found=true});
    leaves.forEach(function(l){
      var on=l.getAttribute('data-id')===id;
      l.classList.toggle('sel',on);
      if(on){l.setAttribute('aria-current','true')}else{l.removeAttribute('aria-current')}
    });
    if(found){set('lights.research.entry',id);document.querySelector('.pane').scrollTop=0}
    return found;
  }
  leaves.forEach(function(l){l.addEventListener('click',function(){show(l.getAttribute('data-id'))})});
  var want=(location.hash||'').replace('#','')||get('lights.research.entry','');
  if(!want||!show(want)){if(leaves.length)show(leaves[0].getAttribute('data-id'))}
  var w=parseInt(get('lights.research.tree','320'),10);
  if(w>=180&&w<=760){root.style.setProperty('--tree',w+'px')}
  var dragging=false;
  function width(px){
    px=Math.max(200,Math.min(px,Math.min(720,root.clientWidth-260)));
    root.style.setProperty('--tree',px+'px');set('lights.research.tree',String(px));
  }
  bar.addEventListener('pointerdown',function(ev){
    dragging=true;bar.setPointerCapture(ev.pointerId);ev.preventDefault();
  });
  bar.addEventListener('pointermove',function(ev){
    if(!dragging)return;width(ev.clientX-root.getBoundingClientRect().left);
  });
  bar.addEventListener('pointerup',function(ev){dragging=false;bar.releasePointerCapture(ev.pointerId)});
  bar.addEventListener('keydown',function(ev){
    var cur=root.getBoundingClientRect(),now=parseInt(getComputedStyle(root).getPropertyValue('--tree'),10)||320;
    if(ev.key==='ArrowLeft'){width(now-24);ev.preventDefault()}
    if(ev.key==='ArrowRight'){width(now+24);ev.preventDefault()}
  });
})();
"""


def leaf(key, label, cls="", lid=None):
    return ('<button class="leaf %s" data-id="%s" type="button">'
            '<span class="lid">%s</span><span class="lt">%s</span></button>'
            % (cls, esc(key), esc(lid if lid is not None else key.split("--")[-1][:9]), esc(label)))


VERDICT_CHIP = {"PRUNE": "refuted", "RE-JUDGE": "warn", "KEEP": "ok"}


def tree_html(doc):
    out = []
    rep = doc.get("report")
    if rep:
        sections = rep.get("sections", [])
        leaves = "".join(leaf("report--%s" % s.get("key", ""), s.get("heading", ""), "", str(i + 1))
                         for i, s in enumerate(sections))
        leaves += leaf("report--reproduced", "how each cited experiment was reproduced", "", "R")
        out.append('<details open><summary>final report<span class="n">%d</span></summary>'
                   '<div class="kids">%s</div></details>' % (len(sections), leaves))
    for group in GROUPS:
        mine = sorted([r for r in doc.get("records", []) if r.get("group") == group],
                      key=lambda x: x.get("date", ""))
        if not mine:
            continue
        kids = "".join(leaf(r["slug"], r.get("title", ""), "" if copy_pasteable(r) else "parked")
                       for r in mine)
        out.append('<details open><summary>%s<span class="n">%d</span></summary>'
                   '<div class="kids">%s</div></details>' % (esc(group), len(mine), kids))
    audit = doc.get("audit", [])
    if audit:
        inner = []
        for kind in AUDIT_KINDS:
            these = [a for a in audit if a.get("kind") == kind]
            if not these:
                continue
            leaves = "".join(leaf("audit--%d" % audit.index(a), a.get("title", ""),
                                  "refuted" if kind == "contradiction" else "")
                             for a in these)
            inner.append('<details open><summary>%s<span class="n">%d</span></summary>'
                         '<div class="kids">%s</div></details>' % (esc(kind), len(these), leaves))
        out.append('<details open><summary>cross-experiment<span class="n">%d</span></summary>'
                   '<div class="kids">%s</div></details>' % (len(audit), "".join(inner)))
    arms = doc.get("arms", [])
    if arms:
        inner = []
        for verdict in ARM_VERDICTS:
            these = [a for a in arms if a.get("verdict") == verdict]
            if not these:
                continue
            leaves = "".join(leaf("arm--%s" % a.get("arm"), a.get("arm", ""),
                                  "refuted" if verdict == "PRUNE" else "", a.get("provenance", ""))
                             for a in these)
            inner.append('<details open><summary>%s<span class="n">%d</span></summary>'
                         '<div class="kids">%s</div></details>' % (esc(verdict), len(these), leaves))
        out.append('<details open><summary>arm audit<span class="n">%d</span></summary>'
                   '<div class="kids">%s</div></details>' % (len(arms), "".join(inner)))
    return "".join(out)


def record_pane(r):
    cmd = (r.get("command") or "").strip()
    runnable = copy_pasteable(r)
    chips = ['<span class="chip">%s</span>' % esc(r.get("group", "")),
             '<span class="chip">%s</span>' % esc(r.get("date", "")),
             '<span class="chip %s">%s</span>' % ("ok" if runnable else "warn",
                                                  "copy-pasteable command" if runnable
                                                  else "no command on record")]
    if r.get("refuted"):
        chips.append('<span class="chip refuted">%d refuted</span>' % len(r["refuted"]))
    parts = ['<article class="entry" id="%s"><h3>%s</h3><div class="meta">%s</div>'
             % (esc(r.get("slug", "")), esc(r.get("title", "")), "".join(chips)),
             '<div class="field"><b>the answer</b><p>%s</p></div>' % linkify(r.get("verdict", "")),
             '<div class="field"><b>the question it was built to answer</b><p>%s</p></div>'
             % linkify(r.get("question", "")),
             '<div class="field"><b>criterion, and its convention</b><p>%s</p></div>'
             % linkify(r.get("criterion", "")),
             '<div class="field"><b>reproduce it</b><pre class="cmd%s">%s</pre></div>'
             % ("" if runnable else " none", esc(cmd))]
    if r.get("numbers"):
        rows = "".join('<tr><td>%s</td><td class="v">%s</td><td class="n">%s</td></tr>'
                       % (linkify(n.get("label", "")), esc(n.get("value", "")), linkify(n.get("note", "")))
                       for n in r["numbers"])
        parts.append('<div class="field"><b>what it measured</b>'
                     '<table class="nums"><thead><tr><th>measurement</th><th>value</th>'
                     '<th>where it comes from</th></tr></thead><tbody>%s</tbody></table></div>' % rows)
    for label, field, cls in (("what it settled", "findings", ""),
                              ("what it refuted", "refuted", "refuted"),
                              ("what it does NOT settle", "not_settled", "caveat")):
        if r.get(field):
            parts.append('<div class="field"><b>%s</b><ul class="f %s">%s</ul></div>'
                         % (label, cls, "".join("<li>%s</li>" % linkify(x) for x in r[field])))
    if r.get("artefacts"):
        parts.append('<div class="field"><b>artefacts on disk</b><p class="srcs">%s</p></div>'
                     % esc(" \u00b7 ".join(r["artefacts"])))
    parts.append('<div class="field"><b>read from</b><p class="srcs">%s</p></div>'
                 % esc(" \u00b7 ".join(r.get("sources", []))))
    parts.append("</article>")
    return "".join(parts)


def audit_pane(i, a):
    return ('<article class="entry" id="audit--%d"><h3>%s</h3>'
            '<div class="meta"><span class="chip %s">%s</span>%s</div>'
            '<div class="field"><b>what it says</b><p>%s</p></div></article>'
            % (i, esc(a.get("title", "")),
               "refuted" if a.get("kind") == "contradiction" else "",
               esc(a.get("kind", "")),
               "".join('<span class="chip">%s</span>' % esc(x) for x in a.get("experiments", [])),
               linkify(a.get("detail", ""))))


def report_panes(rep):
    panes = []
    for s in rep.get("sections", []):
        body = "".join('<div class="field"><p>%s</p><p class="srcs">cites %s</p></div>'
                       % (linkify(c.get("text", "")), esc(" · ".join(c.get("cites", []))))
                       for c in s.get("claims", []))
        panes.append('<article class="entry" id="report--%s"><h3>%s</h3><div class="meta">'
                     '<span class="chip">final report</span><span class="chip">%s</span>'
                     '<span class="chip">tree %s</span></div>%s</article>'
                     % (esc(s.get("key", "")), esc(s.get("heading", "")), esc(rep.get("date", "")),
                        esc(rep.get("tree", "")), body))
    rows = "".join('<tr><td>%s</td><td class="v"><code>%s</code></td><td>%s</td><td class="n">%s</td></tr>'
                   % (esc(r.get("record", "")), esc(r.get("ran") or r.get("invocation") or ""),
                      esc(str(r.get("exit"))), linkify(r.get("result", "")))
                   for r in rep.get("reproduced", []))
    panes.append('<article class="entry" id="report--reproduced"><h3>How each cited experiment was '
                 'reproduced</h3><div class="meta"><span class="chip">tree %s</span></div>'
                 '<div class="field"><table class="nums"><thead><tr><th>experiment</th><th>as run</th>'
                 '<th>exit</th><th>what was compared</th></tr></thead><tbody>%s</tbody></table></div>'
                 '</article>' % (esc(rep.get("tree", "")), rows))
    return "".join(panes)


def arm_pane(a, manifest):
    lines, rows = arm_cost(a, manifest)
    verdict = a.get("verdict", "")
    chips = ['<span class="chip %s">%s</span>' % (VERDICT_CHIP.get(verdict, ""), esc(verdict)),
             '<span class="chip">provenance: %s</span>' % esc(a.get("provenance", "")),
             '<span class="chip">transfers: %s</span>' % esc(a.get("transfers", "")),
             '<span class="chip">%d Rust lines · %d mutation rows</span>' % (lines, rows)]
    chips += ['<span class="chip">%s</span>' % esc(x) for x in a.get("records", [])]
    parts = ['<article class="entry" id="arm--%s"><h3>%s</h3><div class="meta">%s</div>'
             % (esc(a.get("arm", "")), esc(a.get("arm", "")), "".join(chips)),
             '<div class="field"><b>why this verdict</b><p>%s</p></div>' % linkify(a.get("evidence", ""))]
    experiment = a.get("experiment") or {}
    if experiment:
        parts.append('<div class="field"><b>the experiment that would decide it</b><ul class="f caveat">%s</ul></div>'
                     % "".join("<li><b>%s</b> %s</li>" % (esc(f), linkify(experiment.get(f, "")))
                               for f in EXPERIMENT_FIELDS))
    parts.append('<div class="field"><b>where</b><p class="srcs">%s</p></div>'
                 % esc(" · ".join(a.get("where", []))))
    parts.append("</article>")
    return "".join(parts)


def report(doc, today=None):
    day = (today or datetime.date.today()).isoformat()
    total, runnable, audits = standing(doc)
    panes = report_panes(doc["report"]) if doc.get("report") else ""
    panes += "".join(record_pane(r) for r in doc.get("records", []))
    panes += "".join(audit_pane(i, a) for i, a in enumerate(doc.get("audit", [])))
    manifest = load_manifest() if doc.get("arms") else []
    panes += "".join(arm_pane(a, manifest) for a in doc.get("arms", []))
    return "\n".join([
        "<title>The lights research record</title>",
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>',
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
        'family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">',
        "<style>%s</style>" % CSS,
        '<div class="wrap">',
        '<div class="head"><div><p class="eyebrow">crates/lights \u00b7 %s</p>'
        '<h1>The research record</h1>'
        '<p class="sub">Every experiment this crate has run, in one place, each with the command that '
        'reproduces it, the criterion it was judged by, what it refuted, and what it does not settle. '
        'Generated from <code>docs/lights/RESEARCH.json</code>; the readable form is '
        '<code>docs/lights/RESEARCH.md</code>.</p></div>'
        '<div><p class="count">%d<small> experiments</small></p>'
        '<p class="eyebrow">%d with a copy-pasteable command \u00b7 %d cross-experiment findings</p>'
        '</div></div>'
        % (day, total, runnable, audits),
        '<h2>The record</h2>',
        '<p class="hint">A dimmed leaf is an experiment whose command cannot be pasted into a shell as '
        'recorded \u2014 the record explains what is known about how it was run, and that gap is what this '
        'document exists to close. The tree groups experiments by what they were asking about; the '
        'cross-experiment node holds what the records say about <em>each other</em> \u2014 contradictions, '
        'tables that are not in one currency, and confounds shared across runs \u2014 and the arm audit gives '
        'every arm one verdict, grouped by it, with the target its evidence was taken against as the leaf '
        'label. Drag the divider to resize; it and your last '
        'selection are remembered in this browser. Without JavaScript every record is shown in the '
        'right pane in the same order.</p>',
        '<div class="split" id="split"><div class="tree" role="tree">%s</div>'
        '<div class="bar" id="bar" role="separator" aria-orientation="vertical" tabindex="0" '
        'aria-label="resize the tree"></div><div class="pane">%s</div></div>'
        % (tree_html(doc), panes or '<p class="hint">The record is empty.</p>'),
        '<p class="foot">Generated by <code>crates/lights/scripts/research.py</code> from '
        '<code>RESEARCH.json</code> at %s. An experiment is not finished when its numbers exist \u2014 '
        'it is finished when they are here, with the command that reproduces them.</p>' % day,
        "</div>",
        "<script>%s</script>" % SPLIT_JS,
    ])


def main(argv):
    if not os.path.exists(RECORD):
        print("%s is ABSENT \u2014 that is not an empty record and it is not a finished one" % RECORD)
        return 1
    doc = load()
    if "--check" in argv:
        problems = check(doc)
        print("\n".join(problems) if problems else "clean")
        t, r, a = standing(doc)
        print("%d experiments, %d reproducible, %d cross-experiment findings" % (t, r, a))
        arms = doc.get("arms", [])
        if arms:
            print("%d arms: %s" % (len(arms), ", ".join(
                "%d %s" % (len([x for x in arms if x.get("verdict") == v]), v) for v in ARM_VERDICTS)))
        return 1 if problems else 0
    if "--md" in argv:
        with open(MARKDOWN, "w") as fh:
            fh.write(markdown(doc))
        print("wrote %s" % MARKDOWN)
        return 0
    sys.stdout.write(report(doc))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
