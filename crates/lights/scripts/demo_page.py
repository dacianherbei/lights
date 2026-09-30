import base64
import datetime
import os
import sys

CSS = """
.spine{display:flex;flex-direction:column;gap:0;border-top:1px solid var(--line);margin-bottom:8px}
details.step{border-bottom:1px solid var(--line)}
details.step>summary{list-style:none;cursor:pointer;display:grid;
  grid-template-columns:96px 1fr 40px;gap:14px;align-items:baseline;padding:12px 4px}
details.step>summary::-webkit-details-marker{display:none}
details.step .sd{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11.5px;color:var(--muted)}
details.step .st{font-weight:500;line-height:1.35}
details.step .sn{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11px;
  color:var(--muted);text-align:right;font-variant-numeric:tabular-nums}
details.step[open] .st{color:var(--accent)}
.sbody{padding:2px 4px 20px 110px;display:flex;flex-direction:column;gap:12px;max-width:84ch}
.sbody .believed{border-left:2px solid var(--defect);padding-left:12px;color:var(--ink)}
.sbody ul.f{margin:0;padding-left:18px}
.sbody ul.f li{margin:0 0 5px}
.spinelinks{display:flex;gap:8px;flex-wrap:wrap}
.spinelinks .spinep{width:auto;border:1px solid var(--open);border-radius:4px;color:var(--open);
  padding:4px 10px;font-size:12px;background:none;cursor:pointer}
.spinelinks .spinep:hover{background:color-mix(in srgb,var(--open) 14%,transparent)}
@media (max-width:760px){
  details.step>summary{grid-template-columns:1fr 36px}
  details.step .sd{grid-column:1/-1}
  .sbody{padding-left:4px}
}

:root{
  --paper:#0C0E13; --card:#12151C; --ink:#E6EAF2; --muted:#8E97A9; --line:#232936;
  --open:#6FA8FF; --closed:#7E859B; --accent:#F2B84B; --grid:#171B24;
  --ok:#5CBB84; --shadow:none;
}
@media (prefers-color-scheme:light){
  :root:not([data-theme="dark"]){
    --paper:#EDEFF4; --card:#FFFFFF; --ink:#151821; --muted:#5B6274; --line:#D9DDE6;
    --open:#2F4FA8; --closed:#6A7188; --accent:#8A5E06; --grid:#E4E7EE;
    --ok:#2E7D4F; --shadow:0 1px 0 rgba(20,24,40,.05);
  }
}
:root[data-theme="light"]{
  --paper:#EDEFF4; --card:#FFFFFF; --ink:#151821; --muted:#5B6274; --line:#D9DDE6;
  --open:#2F4FA8; --closed:#6A7188; --accent:#8A5E06; --grid:#E4E7EE;
  --ok:#2E7D4F; --shadow:0 1px 0 rgba(20,24,40,.05);
}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
  font-family:"IBM Plex Sans","Helvetica Neue",Arial,sans-serif;font-size:15px;line-height:1.58}
.wrap{max-width:1280px;margin:0 auto;padding:40px 24px 80px}
.eyebrow{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11px;letter-spacing:.18em;
  text-transform:uppercase;color:var(--muted);margin:0}
h1{font-size:34px;line-height:1.1;margin:6px 0 2px;font-weight:600;letter-spacing:-.012em}
h2{font-size:13px;font-family:"IBM Plex Mono",ui-monospace,monospace;letter-spacing:.14em;
  text-transform:uppercase;color:var(--muted);font-weight:500;margin:40px 0 12px}
.sub{color:var(--muted);max-width:70ch;margin:10px 0 0}
.head{display:flex;gap:32px;align-items:flex-end;flex-wrap:wrap;justify-content:space-between;
  border-bottom:1px solid var(--line);padding-bottom:20px;margin-bottom:8px}
.count{font-family:"IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums;
  font-size:42px;font-weight:600;letter-spacing:-.02em;margin:0}
.count small{font-size:14px;font-weight:400;color:var(--muted);letter-spacing:0}
.chip{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:10.5px;letter-spacing:.06em;
  text-transform:uppercase;padding:2px 7px;border-radius:3px;border:1px solid var(--line);
  color:var(--muted);white-space:nowrap}
.chip.reg{color:var(--open);border-color:var(--open)}
.chip.live{color:var(--accent);border-color:var(--accent)}
.hint{color:var(--muted);font-size:13.5px;margin:0 0 14px;max-width:80ch}
.split{display:grid;grid-template-columns:var(--tree,300px) 7px minmax(0,1fr);
  border:1px solid var(--line);border-radius:6px;background:var(--card);
  box-shadow:var(--shadow);overflow:hidden;min-height:620px}
.tree{overflow:auto;max-height:80vh;padding:14px 6px 18px 14px}
.tree details{margin:0 0 2px}
.tree summary{list-style:none;cursor:pointer;padding:5px 6px;border-radius:4px;
  font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11.5px;letter-spacing:.06em;
  text-transform:uppercase;color:var(--muted);display:flex;gap:8px;align-items:center}
.tree summary::-webkit-details-marker{display:none}
.tree summary::before{content:"\\25B8";font-size:9px;transition:transform .12s;display:inline-block}
.tree details[open]>summary::before{transform:rotate(90deg)}
.tree summary:hover{background:var(--grid);color:var(--ink)}
.tree .n{margin-left:auto;font-variant-numeric:tabular-nums;opacity:.7}
.tree .kids{margin:0 0 6px 12px;padding-left:8px;border-left:1px solid var(--line)}
.tree .leaf{display:block;width:100%;text-align:left;background:none;border:0;padding:5px 6px;
  border-radius:4px;cursor:pointer;color:var(--ink);font:inherit;font-size:12.5px;line-height:1.35}
.tree .leaf:hover{background:var(--grid)}
.tree .leaf.sel{background:color-mix(in srgb,var(--open) 18%,transparent)}
.tree .leaf:focus-visible,.tree summary:focus-visible{outline:2px solid var(--open);outline-offset:1px}
.bar{background:var(--line);cursor:col-resize;position:relative}
.bar:hover,.bar:focus-visible{background:var(--open);outline:none}
.pane{overflow:auto;max-height:80vh;padding:22px 26px 34px}
.pane article{max-width:86ch}
.pane h3{font-size:19px;line-height:1.3;margin:0 0 6px;font-weight:600}
.pane .meta{display:flex;gap:6px;flex-wrap:wrap;margin:0 0 16px}
.shot{display:block;width:100%;height:auto;border:1px solid var(--line);border-radius:5px;
  background:#000;margin:0 0 14px}
.pane .field{margin:0 0 14px}
.pane .field b{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:10.5px;
  letter-spacing:.1em;text-transform:uppercase;color:var(--muted);font-weight:500;
  display:block;margin-bottom:3px}
.pane .field p{margin:0}
table.nums{border-collapse:collapse;width:100%;font-size:13px;margin:0 0 16px}
table.nums th{text-align:left;font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:10.5px;
  letter-spacing:.08em;text-transform:uppercase;color:var(--muted);font-weight:500;
  padding:6px 10px;border-bottom:1px solid var(--line)}
table.nums td{padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}
table.nums td.v{font-family:"IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums;
  font-weight:600;white-space:nowrap}
pre.cmd{background:var(--grid);border:1px solid var(--line);border-radius:4px;padding:11px 13px;
  overflow-x:auto;font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12.5px;margin:0 0 16px}
.player{display:flex;gap:10px;align-items:center;margin:0 0 14px;flex-wrap:wrap}
.player button{font:inherit;font-size:13px;padding:5px 14px;border-radius:4px;cursor:pointer;
  border:1px solid var(--line);background:var(--grid);color:var(--ink)}
.player button:hover{border-color:var(--open)}
.player button:focus-visible,.player input:focus-visible{outline:2px solid var(--open);outline-offset:2px}
@media (prefers-reduced-motion:reduce){.shot{transition:none}}
.player input{flex:1;min-width:140px}
.player .read{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12px;color:var(--muted);
  font-variant-numeric:tabular-nums}
code{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12.5px;
  background:var(--grid);padding:1px 4px;border-radius:3px}
.foot{margin-top:44px;padding-top:18px;border-top:1px solid var(--line);color:var(--muted);font-size:13px}
a{color:var(--open)}
@media (max-width:760px){
  .split{grid-template-columns:1fr;min-height:0}
  .bar{display:none}
  .tree{max-height:280px;border-bottom:1px solid var(--line)}
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
    if(found){set('lights.demo.entry',id);document.querySelector('.pane').scrollTop=0}
    return found;
  }
  leaves.forEach(function(l){l.addEventListener('click',function(){show(l.getAttribute('data-id'))})});
  var want=(location.hash||'').replace('#','')||get('lights.demo.entry','');
  if(!want||!show(want)){if(leaves.length)show(leaves[0].getAttribute('data-id'))}
  var w=parseInt(get('lights.demo.tree','300'),10);
  if(w>=180&&w<=760){root.style.setProperty('--tree',w+'px')}
  var dragging=false;
  function width(px){
    px=Math.max(200,Math.min(px,Math.min(720,root.clientWidth-260)));
    root.style.setProperty('--tree',px+'px');set('lights.demo.tree',String(px));
  }
  bar.addEventListener('pointerdown',function(ev){
    dragging=true;bar.setPointerCapture(ev.pointerId);ev.preventDefault();
  });
  bar.addEventListener('pointermove',function(ev){
    if(!dragging)return;width(ev.clientX-root.getBoundingClientRect().left);
  });
  bar.addEventListener('pointerup',function(ev){dragging=false;bar.releasePointerCapture(ev.pointerId)});
  bar.addEventListener('keydown',function(ev){
    var now=parseInt(getComputedStyle(root).getPropertyValue('--tree'),10)||300;
    if(ev.key==='ArrowLeft'){width(now-24);ev.preventDefault()}
    if(ev.key==='ArrowRight'){width(now+24);ev.preventDefault()}
  });
})();
(function(){
  var img=document.getElementById('orbit-frame');
  if(!img||!window.ORBIT||!ORBIT.length)return;
  var play=document.getElementById('orbit-play');
  var slider=document.getElementById('orbit-seek');
  var read=document.getElementById('orbit-read');
  var at=0, timer=null, hz=18;
  slider.max=String(ORBIT.length-1);
  function draw(){
    img.src=ORBIT[at];
    slider.value=String(at);
    read.textContent='frame '+(at+1)+' of '+ORBIT.length+' \\u00b7 '+hz+' fps playback';
  }
  function tick(){at=(at+1)%ORBIT.length;draw()}
  function start(){if(timer)return;timer=setInterval(tick,1000/hz);play.textContent='Pause'}
  function stop(){if(!timer)return;clearInterval(timer);timer=null;play.textContent='Play'}
  play.addEventListener('click',function(){if(timer){stop()}else{start()}});
  slider.addEventListener('input',function(){stop();at=parseInt(slider.value,10)||0;draw()});
  ORBIT.forEach(function(src){var p=new Image();p.src=src});
  draw();
  var still=false;
  try{still=window.matchMedia('(prefers-reduced-motion: reduce)').matches}catch(e){}
  if(still){play.textContent='Play'}else{start()}
})();
"""


def spine_html(directory):
    """The argument the panels are the conclusions of. Rendered ABOVE them, because a
    catalogue of 51 working features does not say why the crate has that shape."""
    try:
        import spine
        steps = spine.steps(tsv_path=os.path.join(directory, "demo.tsv"))
    except Exception as exc:                       # the spine must never break the demo
        return '<p class="hint">the spine could not be derived: %s</p>' % esc(str(exc))
    if not steps:
        return ""
    out = []
    for n, st in enumerate(steps, 1):
        # data-id, not data-title: the page's own picker listens on .leaf[data-id], and a
        # .leaf without one would be collected by querySelectorAll and select nothing.
        panels = "".join(
            '<button class="leaf spinep" data-id="%s" type="button">%s</button>'
            % (esc(pid), esc(title)) for pid, title in st["panels"])
        nums = "".join('<tr><td>%s</td><td class="v">%s</td></tr>'
                       % (esc(l), esc(v)) for l, v in st["numbers"] if l)
        also = "".join("<li>%s</li>" % esc(a) for a in st["also_refuted"])
        changed = "".join("<li>%s</li>" % esc(c) for c in st["changed"])
        out.append(
            '<details class="step"%s><summary>'
            '<span class="sd">%s</span>'
            '<span class="st">%s</span>'
            '<span class="sn">%02d</span></summary>'
            '<div class="sbody">'
            '<div class="field"><b>it was believed</b><p class="believed">%s</p></div>'
            '%s'
            '<div class="field"><b>then it was measured</b><p>%s</p></div>'
            '%s%s%s%s'
            '</div></details>'
            % (" open" if n <= 2 else "",
               esc(st["date"]), esc(st["title"] or st["slug"]), n,
               esc(st["believed"]),
               ('<div class="field"><b>and so were</b><ul class="f">%s</ul></div>' % also) if also else "",
               esc(st["measured"]),
               ('<table class="nums">%s</table>' % nums) if nums else "",
               ('<div class="field"><b>which changed the crate</b><ul class="f">%s</ul></div>'
                % changed) if changed else "",
               ('<div class="field"><b>the entry it settled</b><p><code>%s</code> %s <span '
                'class="chip">%s</span></p></div>'
                % (esc(st["entry"]), esc(st["entry_title"] or ""), esc(st["entry_status"] or "")))
               if st["entry"] else "",
               ('<div class="field"><b>you can see it here</b><div class="spinelinks">%s</div></div>'
                % panels) if panels else ""))
    return '<div class="spine">%s</div>' % "".join(out)


def esc(v):
    return (str(v).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def read_manifest(directory):
    path = os.path.join(directory, "demo.tsv")
    if not os.path.exists(path):
        raise SystemExit("%s is absent — run lights-demo --out-dir %s first"
                         % (path, directory))
    invocation, header, columns, rows = "", {}, [], []
    for line in open(path, encoding="utf-8").read().splitlines():
        if line.startswith("#invocation\t"):
            invocation = line.split("\t", 1)[1]
            continue
        if line.startswith("#"):
            parts = line[1:].split("\t", 1)
            if len(parts) == 2:
                header[parts[0]] = parts[1]
            continue
        cells = line.split("\t")
        if not columns:
            columns = cells
            continue
        rows.append(dict(zip(columns, cells + [""] * (len(columns) - len(cells)))))
    return invocation, header, rows


def data_uri(directory, name):
    path = os.path.join(directory, name)
    if not name or not os.path.exists(path):
        return ""
    with open(path, "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode("ascii")


def facts(row):
    pairs = [
        ("method", row.get("method", "")),
        ("kernel", row.get("kernel", "")),
        ("precision", row.get("precision", "")),
        ("atoms in the scene", row.get("atoms", "")),
        ("tile-atom pairs", row.get("tile_pairs", "")),
        ("kernel evaluations", row.get("kernel_evals", "")),
        ("fastest of the reps", (row.get("ms_min", "") + " ms").strip()),
    ]
    live = [(k, v) for k, v in pairs if v and v != "ms"]
    if not live:
        return ""
    body = "".join(
        '<tr><td>%s</td><td class="v">%s</td></tr>' % (esc(k), esc(v)) for k, v in live)
    return ('<table class="nums"><thead><tr><th>measured</th><th>value</th></tr></thead>'
            '<tbody>%s</tbody></table>' % body)


def chips(row, header):
    out = []
    for reg in [r.strip() for r in row.get("registers", "").split(",") if r.strip()]:
        out.append('<span class="chip reg">%s</span>' % esc(reg))
    size = "%sx%s" % (row.get("width", ""), row.get("height", ""))
    if size != "x":
        out.append('<span class="chip">%s</span>' % esc(size))
    out.append('<span class="chip">%s threads</span>' % esc(header.get("threads", "?")))
    return '<div class="meta">%s</div>' % "".join(out)


def player(frames):
    return (
        '<div class="player">'
        '<button id="orbit-play" type="button">Pause</button>'
        '<input id="orbit-seek" type="range" min="0" max="%d" value="0" '
        'aria-label="orbit frame">'
        '<span class="read" id="orbit-read"></span>'
        '</div>' % max(0, len(frames) - 1))


def pane(row, directory, header, orbit_frames):
    ident = row["id"]
    moving = ident == "orbit" and orbit_frames
    image = data_uri(directory, row.get("file", ""))
    shot = ''
    if image and moving:
        shot = ('<img class="shot" id="orbit-frame" src="%s" alt="%s">'
                % (orbit_frames[0], esc(row.get("title", ident))))
    elif image:
        shot = '<img class="shot" src="%s" alt="%s">' % (image, esc(row.get("title", ident)))
    detail = row.get("ms_detail", "")
    return (
        '<article class="entry" id="%s" hidden>'
        '<h3>%s</h3>%s%s%s'
        '<div class="field"><b>what it shows</b><p>%s</p></div>'
        '%s'
        '%s'
        '</article>'
    ) % (
        esc(ident),
        esc(row.get("title", ident)),
        chips(row, header),
        shot,
        player(orbit_frames) if moving else "",
        esc(row.get("shows", "")),
        facts(row),
        ('<div class="field"><b>how it was measured</b><p>%s</p></div>' % esc(detail))
        if detail else "",
    )


def tree(groups):
    out = []
    for group, rows in groups:
        leaves = "".join(
            '<button class="leaf" data-id="%s" type="button">%s</button>'
            % (esc(r["id"]), esc(r.get("title", r["id"])))
            for r in rows)
        out.append('<details open><summary>%s<span class="n">%d</span></summary>'
                   '<div class="kids">%s</div></details>'
                   % (esc(group), len(rows), leaves))
    return "".join(out)


def page(directory, today=None):
    day = (today or datetime.date.today()).isoformat()
    invocation, header, rows = read_manifest(directory)
    frames = [data_uri(directory, r["file"])
              for r in rows if r.get("kind") == "frame" and r.get("file")]
    panels = [r for r in rows if r.get("kind") == "panel"]
    groups = []
    for row in panels:
        if not groups or groups[-1][0] != row.get("group", ""):
            groups.append((row.get("group", ""), []))
        groups[-1][1].append(row)
    panes = "".join(pane(r, directory, header, frames) for r in panels)
    orbit_js = "window.ORBIT=[%s];" % ",".join('"%s"' % f for f in frames)
    warning = header.get("warning", "")
    return "\n".join([
        "<title>The lights demo</title>",
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>',
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
        'family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">',
        "<style>%s</style>" % CSS,
        '<div class="wrap">',
        '<div class="head"><div><p class="eyebrow">crates/lights · %s</p>'
        '<h1>The demo</h1>'
        '<p class="sub">One panel per feature of the renderer, each rendered by the crate itself '
        'and carrying what it measured. Landing a feature means adding a panel here, so this page '
        'is what it currently means to say the renderer works. Generated from '
        '<code>demo.tsv</code> by <code>crates/lights/scripts/demo_page.py</code>.</p></div>'
        '<div><p class="count">%d<small> panels</small></p>'
        '<p class="eyebrow">%d orbit frames · %s threads</p></div></div>'
        % (day, len(panels), len(frames), esc(header.get("threads", "?"))),
        '<h2>How the crate got its shape</h2>',
        '<p class="hint">Every experiment this crate has run <b>refuted something</b>, and most '
        'often it was the project\'s own prior belief. That is the argument the panels below are '
        'the conclusions of, in the order it was made — derived from '
        '<code>docs/lights/RESEARCH.json</code> by <code>scripts/spine.py</code>, never authored. '
        'A step whose belief or verdict the record cannot supply is refused rather than filled in. '
        'The first two are open; click any step.</p>',
        spine_html(directory),
        '<h2>The panels</h2>',
        '<p class="hint">Every image on this page came out of <code>lights</code> itself on the run '
        'named below — nothing is an illustration. Pick a panel on the left; drag the divider to '
        'resize. <b>The orbit under &ldquo;in motion&rdquo; plays on its own</b>: it is the only panel '
        'that can show the renderer being driven frame by frame, and it is where this crate’s '
        'first frame-rate number comes from.</p>',
        '<div class="split" id="split"><div class="tree" role="tree">%s</div>'
        '<div class="bar" id="bar" role="separator" aria-orientation="vertical" tabindex="0" '
        'aria-label="resize the tree"></div><div class="pane">%s</div></div>'
        % (tree(groups), panes or '<p class="hint">The manifest holds no panels.</p>'),
        '<h2>The run behind this page</h2>',
        '<pre class="cmd">%s</pre>' % esc(invocation or "no invocation recorded"),
        '<p class="hint">%s</p>' % esc(warning),
        '<p class="foot">Generated %s. A panel is a claim the code has to keep: every number beside '
        'an image came from the render above it, on this run, on this machine.</p>' % day,
        "</div>",
        "<script>%s%s</script>" % (orbit_js, SPLIT_JS),
    ])


def main(argv):
    if not argv:
        print("usage: demo_page.py <out-dir>   (the directory lights-demo wrote)")
        return 2
    sys.stdout.write(page(argv[0]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
