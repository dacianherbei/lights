import base64, csv, os, io, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pngdiff import read_png, maxdiff, encode

SP = "<session scratchpad>/959574de-b7ef-49fa-8b42-9a8e0e211b3c/scratchpad/p2_run_48"
OUT = "<session scratchpad>/959574de-b7ef-49fa-8b42-9a8e0e211b3c/scratchpad/p2_results.html"

meta, rows, hdr = {}, [], None
for line in open(os.path.join(SP, "report.tsv")):
    line = line.rstrip("\n")
    if line.startswith("#"):
        p = line[1:].split("\t")
        if len(p) >= 2: meta[p[0]] = p[1]
        continue
    parts = line.split("\t")
    if hdr is None: hdr = parts; continue
    rows.append(dict(zip(hdr, parts)))

def img(name):
    p = os.path.join(SP, name)
    if not os.path.exists(p): return None
    return "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()

TARGETS = [
    ("t1-flat-fill", "T1", "Flat fill", "interior ripple &le; 1/255",
     "The only class splats win. Under normalised blending two atoms give ripple of exactly 0.000000 &mdash; a partition of unity reaching a rendered pixel. Alpha compositing provably cannot: it leaves 0.0002. It is also a filled rectangle, which the analytic arm renders exactly, for free."),
    ("t2-straight-edge", "T2", "Straight edge, 30&deg;", "10&ndash;90 width &lt; 1 px, contrast &ge; 0.90",
     "Chosen at 30&deg; deliberately: an irrational tangent makes the pixel-grid offsets equidistribute, so the integer grid hands you a dense sub-pixel scatter of the edge profile for free. A rational angle would sample it periodically and coarsely."),
    ("t3-right-angle-corner", "T3", "Right-angle corner", "10&ndash;90 width &lt; 1 px",
     "The corner is where the cost concentrates. Two analytic primitives against 25 atoms &mdash; and the normalised cells were the only ones in the whole run that ran out of <em>budget</em> rather than stalling."),
    ("t4-curve", "T4", "Curve, r = 3 px", "10&ndash;90 width &lt; 1 px",
     "Glyph-scale curvature. Passes in exactly one of four splat configurations, and only under the unprefiltered profile."),
    ("t5-thin-stroke", "T5", "Thin stroke, 1 px", "peak &ge; 0.90, |FWHM &minus; 1| &le; 0.5",
     "The verify pass broke this cell open: one hand-placed needle meets the criterion where the fitter needed twelve. The fitter minimises L2 energy, and the L2-optimal stroke does not meet the criterion while a deliberately-too-thin needle does."),
    ("t6-glyph-like", "T6", "Glyph, 12 px em", "peak &ge; 0.90 on stem and bowl",
     "Never met, in all four splat configurations. This is the class the whole question was about, and the splat arm did not produce a number at any budget."),
]

CELLS = [
    ("gaussian-contract",              "gaussian",            "contract",    "Over",       "no prefilter"),
    ("gaussian-prefiltered",           "gaussian",            "prefiltered", "Over",       "prefiltered"),
    ("gaussian-normalised-contract",   "gaussian-normalised", "contract",    "Normalised", "no prefilter"),
    ("gaussian-normalised-prefiltered","gaussian-normalised", "prefiltered", "Normalised", "prefiltered"),
]

def find(t, blend, profile, arm="atoms"):
    for r in rows:
        if r["target"]==t and r["arm"]==arm and r.get("blend")==blend and r.get("profile")==profile:
            return r
    return None

def analytic(t, arm):
    for r in rows:
        if r["target"] == t and r["arm"] == arm: return r
    return None

def num(r, k, nd=3, dash="&mdash;"):
    if not r: return dash
    v = r.get(k, "")
    if v in ("", "nan", "-"): return dash
    try: return ("%%.%df" % nd) % float(v)
    except Exception: return v

def us(r):
    if not r: return "&mdash;"
    v = r.get("render_ns_mean","")
    try: return "%.1f&thinsp;&micro;s" % (float(v)/1000.0)
    except Exception: return "&mdash;"

ATOM_SCALARS = 14
ATOM_B = {"f64": 14*8, "f32": 14*4, "f16": 14*2}

def count_of(r, arm):
    """How many primitives the arm produced. One question, one column."""
    if r is None: return "&mdash;"
    key = "primitives" if arm != "atoms" else "atoms"
    v = (r.get(key) or "").strip()
    if v in ("","-","nan"): return "&mdash;"
    return v + ("&nbsp;prim" if arm != "atoms" else "&nbsp;atoms")

def meets_at_of(r):
    """The atom count at which the criterion was FIRST met. A different question
       from how many were fitted, so it gets its own column."""
    if r is None: return "&mdash;"
    v = (r.get("met_criterion_at") or "").strip()
    if v in ("","-","nan","-1","never"): return "&mdash;"
    return v

def size_of(r, arm):
    """Description SIZE in bytes. Analytic: the stored primitive set.
       Splats: atoms x 112 B, an Atom<f64> being 14 scalars."""
    if r is None: return "&mdash;"
    if arm != "atoms":
        v = r.get("description_bytes","").strip()
        try: return "%d&thinsp;B" % int(float(v))
        except Exception: return "&mdash;"
    a = (r.get("met_criterion_at") or "").strip()
    if a in ("","-","nan","-1","never"): a = (r.get("atoms") or "").strip()
    try: n = int(float(a))
    except Exception: return "&mdash;"
    return "%d&thinsp;B" % (n * ATOM_B["f64"])

def size16_of(r, arm):
    if r is None or arm != "atoms": return ""
    a = (r.get("met_criterion_at") or "").strip()
    if a in ("","-","nan","-1","never"): a = (r.get("atoms") or "").strip()
    try: n = int(float(a))
    except Exception: return ""
    return "%d&thinsp;B" % (n * ATOM_B["f16"])

def bytes_of(r):
    if not r: return "&mdash;"
    v = r.get("description_bytes","")
    try: return "%d&thinsp;B" % int(float(v))
    except Exception: return "&mdash;"

def atoms_of(r):
    if not r: return "&mdash;"
    v = (r.get("met_criterion_at") or "").strip()
    if v in ("","-","nan","-1"): return "never"
    return v

def meets_of(r):
    return bool(r) and r.get("meets_criterion","").strip().lower() in ("yes","true","1")

def bound_of(r):
    if not r: return "&mdash;"
    return "fitter" if r.get("improvement_exhausted","").strip().lower() in ("yes","true","1") else "budget"

plates = []
for key, tid, title, crit, note in TARGETS:
    ax  = analytic(key, "analytic-exact")
    asd = analytic(key, "analytic-sdf")
    shots = []
    t = img(key+"-truth.png")
    if t: shots.append(("truth","ground truth",t,"",""))
    a = img(key+"-analytic-sdf.png")
    if a: shots.append(("analytic-sdf","analytic sdf",a,"pass" if meets_of(asd) else "fail",
                        (asd or {}).get("primitives","") + " primitives"))
    cellrows = []
    for cid, blend, prof, blabel, plabel in CELLS:
        r  = find(key, blend, prof)
        im = img(key+"-"+cid+"-atoms.png")
        n = (r or {}).get("atoms","").strip()
        lab = ("%s atoms" % n) if n not in ("","-","nan") else ""
        if im: shots.append((cid, blabel+"<br>"+plabel, im, "pass" if meets_of(r) else "fail", lab))
        cellrows.append((blabel, plabel, count_of(r,"atoms"), meets_at_of(r), meets_of(r), bound_of(r),
                         size_of(r,"atoms"),
                         num(r,"transition_10_90_px"), num(r,"contrast_frac"), num(r,"peak_frac"),
                         num(r,"interior_ripple",6), num(r,"box_over_ellipse_area",1),
                         num(r,"mean_tile_list",0), us(r)))
    plates.append((tid,title,crit,note,shots,cellrows,asd,ax))


DIFFS = {}
def diffs_for(key, variants):
    """variants: list of (id, filename). Returns {id: datauri}, shared_peak."""
    if key in DIFFS: return DIFFS[key]
    tp = os.path.join(SP, key + "-truth.png")
    if not os.path.exists(tp):
        DIFFS[key] = ({}, 0); return DIFFS[key]
    truth = read_png(tp)
    fields, peak = {}, 0
    for vid, fn in variants:
        fp = os.path.join(SP, fn)
        if not os.path.exists(fp): continue
        try:
            f, pk = maxdiff(truth, read_png(fp))
        except Exception:
            continue
        fields[vid] = f
        peak = max(peak, pk)
    w, h = truth[0], truth[1]
    out = {vid: encode(f, w, h, peak) for vid, f in fields.items()}
    DIFFS[key] = (out, peak)
    return DIFFS[key]

def variant_list(key):
    v = [("analytic-exact", key + "-analytic-exact.png"),
         ("analytic-sdf",   key + "-analytic-sdf.png")]
    for cid, _b, _p, _bl, _pl in CELLS:
        v.append((cid, key + "-" + cid + "-atoms.png"))
    return v

def shot_html(s, dmap=None):
    cid, cap, src, verdict, val = s
    badge = ""
    if verdict == "pass": badge = '<span class="v ok">meets</span>'
    elif verdict == "fail": badge = '<span class="v no">fails</span>'
    v = ('<div class="val">%s</div>' % val) if val else ""
    d = ""
    if dmap and cid in dmap:
        d = ('<img class="dif" src="%s" alt="difference from ground truth" width="64" height="64">'
             '<span class="dlab">&Delta; truth</span>') % dmap[cid]
    return ('<figure class="shot"><img src="%s" alt="%s" width="64" height="64">%s'
            '<figcaption><span class="cap">%s</span>%s%s</figcaption></figure>') % (src, cid, d, cap, badge, v)



# ---- ASPECT 2: signed atoms and wavelet transforms ----------------------
A2 = "<session scratchpad>/959574de-b7ef-49fa-8b42-9a8e0e211b3c/scratchpad/aspect2-strips"
A2_STRIPS = [
 ("strip-t2-straight-edge-n512-4x.png",
  "30&deg; straight edge, N=512 &mdash; Haar reaches this edge <strong>exactly</strong> at 1024 terms (rel L2 0.0000, max-abs 0.0000) where CDF 9/7 is still at 0.0112. The prediction that Haar would be poor on a non-axis-aligned edge was wrong."),
 ("strip-n1-colour-field-n8-4x.png",
  "Colour field, N=8 &mdash; at low term counts a transform has too few coefficients to place even a constant colour (every transform arm sits at 0.95&ndash;0.99 relative L2 at N=1). This end flatters the atoms."),
 ("strip-n1-colour-field-n512-4x.png",
  "Colour field, N=512 &mdash; at high counts the atom arms are pinned at their fitter ceiling while the transforms keep going. This end flatters the transforms. Read both or neither."),
 ("strip-n2-soft-shadow-n512-4x.png",
  "Soft shadow, N=512 &mdash; the greedy fitter exhausted at 50 atoms and cannot go below rel L2 0.0205; CDF 9/7 reaches 0.0058 at 1024 coefficients. That is a fitter result, not a representation result."),
 ("strip-n3-band-limited-noise-n512-4x.png",
  "Band-limited noise, N=512 &mdash; CDF 9/7's best case, and the one the verify pass found was <em>dyadically aligned</em>: the noise lattice periods 16/8/4 anchor at pixel 0, worth 1.69&times; to the transform."),
 ("strip-n4-broadband-patch-n512-4x.png",
  "Broadband patch, N=512 &mdash; the closest class to a photograph, and the only one where no arm is clearly ahead on both axes."),
]

def a2_strips():
    out = []
    for fn, cap in A2_STRIPS:
        fp = os.path.join(A2, fn)
        if not os.path.exists(fp): continue
        uri = "data:image/png;base64," + base64.b64encode(open(fp,"rb").read()).decode()
        out.append('<figure class="stripfig"><img src="%s" alt="%s"><figcaption>%s</figcaption></figure>' % (uri, fn, cap))
    return "".join(out)

a2_img = a2_strips()

# ---- ASPECT 1: the atom curve -------------------------------------------
A1 = "<session scratchpad>/959574de-b7ef-49fa-8b42-9a8e0e211b3c/scratchpad/p3_atom_curve"
from collections import defaultdict as _dd

def a1_rows():
    fp = os.path.join(A1, "report.tsv")
    if not os.path.exists(fp): return []
    ls = [l.rstrip("\n") for l in open(fp) if not l.startswith("#")]
    h = ls[0].split("\t")
    return [dict(zip(h, l.split("\t"))) for l in ls[1:]]

A1_ORDER = [("super-gaussian-b0.5","&beta; 0.5"), ("gaussian","gaussian (&beta;=1)"),
            ("super-gaussian-b2","&beta; 2"), ("super-gaussian-b4","&beta; 4"),
            ("super-gaussian-b8","&beta; 8"), ("compact-bump","compact bump")]
A1_VERDICT = {"super-gaussian-b0.5":("no","eliminated"), "gaussian":("","control"),
              "super-gaussian-b2":("ok","survives"), "super-gaussian-b4":("ok","headline"),
              "super-gaussian-b8":("no","eliminated"), "compact-bump":("ok","carried")}

def a1_table():
    rows = a1_rows()
    if not rows: return "", ""
    meets, tot, corner, kev, ovf = _dd(int), _dd(int), {}, _dd(int), {}
    for r in rows:
        k = r.get("blend","")
        if r.get("arm") != "atoms" or k in ("","-"): continue
        tot[k] += 1
        if r.get("meets_criterion","").lower() == "yes": meets[k] += 1
        if r.get("target") != "t1-flat-fill":
            try: kev[k] += int(float(r.get("kernel_evaluations") or 0))
            except Exception: pass
        if r.get("target") == "t3-right-angle-corner":
            corner[k] = r.get("delta_to_control_px","")
            ovf[k] = r.get("overfit_gap_px","")
    base = kev.get("gaussian", 0)
    out = []
    for key, label in A1_ORDER:
        if key not in tot: continue
        cls, note = A1_VERDICT.get(key, ("",""))
        rel = ("%.2f&times;" % (base / kev[key])) if kev.get(key) else "&mdash;"
        try: cd = "%.3f&nbsp;px" % float(corner.get(key,"nan"))
        except Exception: cd = "&mdash;"
        try:
            ov = float(ovf.get(key,"nan")); ovs = "%.3f&nbsp;px" % ov
            ovc = ' class="m warn"' if ov > 0.4 else ' class="m"'
        except Exception: ovs, ovc = "&mdash;", ' class="m"'
        out.append('<tr><td class="m">%s</td><td class="m atoms %s">%d of %d</td>'
                   '<td class="m">%s</td><td class="m">%s</td><td%s>%s</td>'
                   '<td class="m"><span class="v %s">%s</span></td></tr>'
                   % (label, ("ok" if meets[key] >= 4 else ""), meets[key], tot[key],
                      cd, rel, ovc, ovs, cls, note))
    return "".join(out), str(base)

A1_STRIPS = [("t3-right-angle-corner","Right-angle corner &mdash; where &beta;=4 wins outright"),
             ("t6-glyph-like","12&nbsp;px glyph &mdash; no kernel meets, at any budget"),
             ("t2-straight-edge","30&deg; straight edge"),
             ("t5-thin-stroke","1&nbsp;px stroke &mdash; the ranking reverses here"),
             ("t4-curve","Curve, r = 3&nbsp;px"),
             ("t1-flat-fill","Flat fill")]

def a1_strips():
    out = []
    for key, cap in A1_STRIPS:
        fp = os.path.join(A1, "strip-%s-4x.png" % key)
        if not os.path.exists(fp): continue
        uri = "data:image/png;base64," + base64.b64encode(open(fp,"rb").read()).decode()
        out.append('<figure class="stripfig"><img src="%s" alt="%s"><figcaption>%s</figcaption></figure>'
                   % (uri, key, cap))
    return "".join(out)

a1_body, a1_base = a1_table()
a1_img = a1_strips()


# ---- ASPECT 3: the zoom ladder ------------------------------------------
A3 = "<repo>/docs/lights/p5"
A3_CACHE = os.path.join(SP, "a3_small")

def a3_load(name):
    fp = os.path.join(A3, name)
    if not os.path.exists(fp): return [], {}
    head, body, meta3 = None, [], {}
    for line in open(fp):
        line = line.rstrip("\n")
        if line.startswith("#"):
            q = line[1:].split("\t")
            if len(q) >= 2: meta3[q[0]] = q[1]
            continue
        parts = line.split("\t")
        if head is None: head = parts; continue
        body.append(dict(zip(head, parts)))
    return body, meta3

def a3_verdict(r, meta3):
    pairs = [("transition_10_90_scene", "meets_transition_scene", "meets_scene"),
             ("fwhm_scene", "meets_fwhm_scene", None)]
    for value_key, new_key, old_key in pairs:
        raw = r.get(value_key, "nan")
        try: finite = float(raw) == float(raw) and abs(float(raw)) != float("inf")
        except Exception: finite = False
        if not finite: continue
        if new_key in r and r[new_key] in ("yes", "no"): return r[new_key]
        if old_key and r.get(old_key) in ("yes", "no"): return r[old_key]
        if value_key.startswith("fwhm"):
            try:
                target = float(meta3.get("criterion.target_fwhm", 1.0))
                tol = float(meta3.get("criterion.max_fwhm_error", 0.5))
                return "yes" if abs(float(raw) - target) <= tol else "no"
            except Exception: return "n/a"
        try:
            return "yes" if float(raw) < float(meta3.get("criterion.max_transition", 1.0)) else "no"
        except Exception: return "n/a"
    return "n/a"

A3_CLASSES = [("t2-straight-edge", "T2 straight edge <span class=\'probe\'>judged on 10&ndash;90 transition &lt; 1 scene unit</span>"),
              ("t3-right-angle-corner", "T3 right-angle corner <span class=\'probe\'>judged on 10&ndash;90 transition &lt; 1 scene unit</span>"),
              ("t6-glyph-like", "T6 glyph, 12 px <span class=\'probe\'>judged on FWHM, &plusmn;0.5 of a 1-unit stem &mdash; a loose gate; read rel L2 beside it</span>"),
              ("n1-colour-field", "N1 colour field <span class=\'probe\'>carries neither probe &mdash; rel L2 only</span>")]
A3_ARMS = [("frozen-atoms-gaussian", "contract", "frozen atoms &middot; no prefilter"),
           ("frozen-atoms-gaussian", "prefiltered", "frozen atoms &middot; prefiltered"),
           ("refitted-atoms-gaussian", "contract", "refitted atoms (control)"),
           ("analytic-sdf", "profile-free", "analytic SDF (bar)"),
           ("wavelet-haar", "profile-free", "Haar"),
           ("wavelet-cdf97", "profile-free", "CDF 9/7")]

def a3_table(name, rung_order):
    body, meta3 = a3_load(name)
    if not body: return ""
    out = []
    for key, label in A3_CLASSES:
        out.append('<tr class="a3head"><td colspan="%d">%s</td></tr>' % (len(rung_order) + 2, label))
        for arm, profile, arm_label in A3_ARMS:
            cells = {r["scale"]: r for r in body
                     if r["class"] == key and r["arm"] == arm and r["profile"] == profile}
            if not cells: continue
            tds = []
            for rung in rung_order:
                r = cells.get(rung)
                if not r: tds.append('<td class="m">&mdash;</td>'); continue
                v = a3_verdict(r, meta3)
                if r.get("absent") == "yes":
                    tds.append('<td class="m">absent</td>'); continue
                mark = {"yes": '<span class="v ok">meets</span>',
                        "no": '<span class="v no">no</span>'}.get(v, '<span class="m">n/a</span>')
                base = ' style="font-weight:600"' if r.get("at_base") == "yes" else ''
                tds.append('<td class="m"%s>%s</td>' % (base, mark))
            first_row = cells.get(rung_order[0], {})
            last_row = cells.get(rung_order[-1], {})
            def f3(v):
                try:
                    x = float(v)
                    return "%.3f" % x if x == x else "&mdash;"
                except Exception:
                    return "&mdash;"
            if first_row.get("absent") == "yes":
                tds.append('<td class="m">absent</td>')
            else:
                tds.append('<td class="m">%s &rarr; %s</td>'
                           % (f3(first_row.get("relative_l2", "")), f3(last_row.get("relative_l2", ""))))
            out.append('<tr><td>%s</td>%s</tr>' % (arm_label, "".join(tds)))
    return "".join(out)

A3_STRIPS = [("ladder-t2-straight-edge-magnify-contract-frozen.png",
              "T2 edge, magnify 1&times;&rarr;8&times;, frozen atoms &mdash; the one class a frozen description carries the whole way"),
             ("ladder-t2-straight-edge-minify-contract-frozen.png",
              "T2 edge, minify 8&times;&rarr;0.25&times; &mdash; frozen still meets where a fresh fit stops"),
             ("ladder-t3-right-angle-corner-magnify-contract-frozen.png",
              "T3 corner, magnify &mdash; atoms meet at no rung in either direction"),
             ("ladder-t6-glyph-like-magnify-contract-frozen.png",
              "T6 glyph, magnify &mdash; the frozen set decays 0.223&rarr;0.458 rel L2 while the analytic bar improves 0.257&rarr;0.092"),
             ("ladder-n1-colour-field-magnify-contract-frozen.png",
              "N1 colour field, magnify &mdash; smooth content, where freezing costs least"),
             ("extremes-t6-glyph-like-magnify-contract-s8.png",
              "T6 glyph at 8&times;: truth, frozen, refitted, analytic side by side")]

def a3_strips():
    if not os.path.isdir(A3): return ""
    try: os.makedirs(A3_CACHE, exist_ok=True)
    except Exception: pass
    out = []
    for fn, cap in A3_STRIPS:
        src = os.path.join(A3, fn)
        if not os.path.exists(src): continue
        small = os.path.join(A3_CACHE, fn)
        if not os.path.exists(small) or os.path.getmtime(small) < os.path.getmtime(src):
            os.system('sips -Z 1200 "%s" --out "%s" >/dev/null 2>&1' % (src, small))
        use = small if os.path.exists(small) else src
        uri = "data:image/png;base64," + base64.b64encode(open(use, "rb").read()).decode()
        out.append('<figure class="stripfig"><img src="%s" alt="%s"><figcaption>%s</figcaption></figure>'
                   % (uri, fn, cap))
    return "".join(out)

a3_mag = a3_table("ladder-magnify.tsv", ["1x", "2x", "4x", "8x"])
a3_min = a3_table("ladder-minify.tsv", ["8x", "4x", "2x", "1x", "0.5x", "0.25x"])
a3_img = a3_strips()

key_of = {t[1]: t[0] for t in TARGETS}
plate_html = []
for tid, title, crit, note, shots, cellrows, asd, ax in plates:
    dmap, dpeak = diffs_for(key_of[tid], variant_list(key_of[tid]))
    strip = "".join(shot_html(x, dmap) for x in shots)
    dnote = ('<p class="dnorm">All &Delta; panels for this target share one normalisation: '
             'white = %d/255 of maximum channel error, the worst pixel across every variant here. '
             'Independently scaled panels would make a tiny error look identical to a large one.</p>') % dpeak
    head_extra = ('<tr class="ref"><td class="m">analytic</td><td class="m">sdf</td>'
                  '<td class="m">%s</td><td class="m atoms ok">exact</td>'
                  '<td class="m">&mdash;</td><td class="m">%s</td>'
                  '<td class="m">%s</td><td class="m">%s</td><td class="m">%s</td>'
                  '<td class="m">%s</td><td class="m">&mdash;</td><td class="m">&mdash;</td>'
                  '<td class="m">%s</td></tr>') % (
        count_of(asd,"analytic-sdf"), size_of(asd,"analytic-sdf"),
        num(asd,"transition_10_90_px"), num(asd,"contrast_frac"),
        num(asd,"peak_frac"), num(asd,"interior_ripple",6), us(asd))
    trs = "".join(
        '<tr><td class="m">%s</td><td class="m">%s</td><td class="m">%s</td>'
        '<td class="m atoms %s">%s</td><td class="m b-%s">%s</td><td class="m">%s</td>'
        '<td class="m">%s</td><td class="m">%s</td><td class="m">%s</td>'
        '<td class="m">%s</td><td class="m">%s</td><td class="m">%s</td>'
        '<td class="m">%s</td></tr>'
        % (b,p,a,("ok" if ok else "no"),ma,bd,bd,sz,w,c,pk,rp,box,mt,tm)
        for (b,p,a,ma,ok,bd,sz,w,c,pk,rp,box,mt,tm) in cellrows)
    plate_html.append("""
<section class="plate">
  <header class="plate-h"><span class="tid">%s</span><h3>%s</h3><span class="crit">%s</span></header>
  <div class="strip">%s</div>
  <p class="note">%s</p>\n  %s
  <div class="tw"><table>
    <thead><tr><th>blend</th><th>profile</th><th>atoms&nbsp;fitted</th><th>meets&nbsp;at</th><th>stopped&nbsp;by</th>
      <th>size&nbsp;f64</th><th>10&ndash;90&nbsp;px</th><th>contrast</th><th>peak</th><th>ripple</th>
      <th>box&divide;ell</th><th>tile&nbsp;list</th><th>render</th></tr></thead>
    <tbody>%s%s</tbody>
  </table></div>
</section>""" % (tid,title,crit,strip,note,dnote,head_extra,trs))

# THE GLYPH PLANE - every strategy on one content, with cost
gk = "t6-glyph-like"
plane = []
gax, gsd = analytic(gk,"analytic-exact"), analytic(gk,"analytic-sdf")
plane.append(("truth","ground truth", img(gk+"-truth.png"), "&mdash;","&mdash;","&mdash;","",""))
plane.append(("analytic-exact","analytic exact", img(gk+"-analytic-exact.png"),
              count_of(gax,"analytic-exact"), size_of(gax,"analytic-exact"), us(gax)+" &dagger;",
              "pass" if meets_of(gax) else "fail", ""))
plane.append(("analytic-sdf","analytic sdf", img(gk+"-analytic-sdf.png"),
              count_of(gsd,"analytic-sdf"), size_of(gsd,"analytic-sdf"), us(gsd),
              "pass" if meets_of(gsd) else "fail", ""))
for cid, blend, prof, blabel, plabel in CELLS:
    r = find(gk, blend, prof)
    plane.append((cid, blabel+" &middot; "+plabel, img(gk+"-"+cid+"-atoms.png"),
                  count_of(r,"atoms"), size_of(r,"atoms"), us(r),
                  "pass" if meets_of(r) else "fail", size16_of(r,"atoms")))

gmap, gpeak = diffs_for(gk, variant_list(gk))
plane_html = "".join(
    '<figure class="pcell"><img src="%s" alt="%s" width="64" height="64">%s'
    '<figcaption><span class="pname">%s</span>'
    '<span class="prow"><span class="pk">count</span><span>%s</span></span>'
    '<span class="prow"><span class="pk">size f64</span><span>%s</span></span>'
    '<span class="prow"><span class="pk">render</span><span>%s</span></span>'
    '<span class="prow"><span class="pk">size f16</span><span>%s</span></span>'
    '<span class="prow verdict-row"><span class="pk">&nbsp;</span><span class="v %s">%s</span></span>'
    '</figcaption></figure>'
    % (src or "", name,
       ('<img class="dif" src="%s" alt="difference" width="64" height="64"><span class="dlab">&Delta; truth</span>' % gmap[vid]) if vid in gmap else "",
       name, atoms, byt, tm, (peak if peak else "&mdash;"), ("ok" if verdict=="pass" else "no" if verdict else ""),
       ("meets" if verdict=="pass" else "fails" if verdict else "&mdash;"))
    for (vid, name, src, atoms, byt, tm, verdict, peak) in plane if src)

html = """<title>What Crisp Pixels Cost in Gaussians</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Serif:ital,wght@0,400;0,500;1,400&display=swap">
<style>
:root{
  --ground:#f7f8fa; --panel:#ffffff; --ink:#171a21; --muted:#5c6373; --rule:#dfe3ea;
  --accent:#0d7d8a; --ok:#2e7d4f; --no:#b3402f; --warn:#96701d; --chip:#eef1f5;
  --shadow:0 1px 2px rgba(20,26,40,.06),0 8px 24px -16px rgba(20,26,40,.28);
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#12141a; --panel:#171a21; --ink:#e8eaef; --muted:#8b93a5; --rule:#272b36;
  --accent:#4dc4d0; --ok:#5cbb84; --no:#e0765f; --warn:#d0a642; --chip:#1d212a;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px -16px rgba(0,0,0,.8);
}}
:root[data-theme="dark"]{
  --ground:#12141a; --panel:#171a21; --ink:#e8eaef; --muted:#8b93a5; --rule:#272b36;
  --accent:#4dc4d0; --ok:#5cbb84; --no:#e0765f; --warn:#d0a642; --chip:#1d212a;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px -16px rgba(0,0,0,.8);
}
*{box-sizing:border-box}
body{background:var(--ground);color:var(--ink);
  font-family:"IBM Plex Serif",Georgia,serif;font-size:16.5px;line-height:1.62;
  padding:0 20px 96px;-webkit-font-smoothing:antialiased}
.wrap{max-width:1180px;margin:0 auto}
.col{max-width:66ch}
h1,h2,h3,.ui{font-family:"IBM Plex Sans",system-ui,sans-serif;text-wrap:balance}
h1{font-size:clamp(2rem,4.6vw,3.1rem);font-weight:700;letter-spacing:-.021em;line-height:1.08;margin:0 0 14px}
h2{font-size:1.42rem;font-weight:600;letter-spacing:-.011em;margin:0 0 10px}
h3{font-size:1.06rem;font-weight:600;margin:0}
.m,.mono,code,table{font-family:"IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums}
header.top{border-bottom:2px solid var(--ink);padding:52px 0 22px;margin-bottom:34px}
.eyebrow{font-family:"IBM Plex Mono",monospace;font-size:.7rem;letter-spacing:.15em;
  text-transform:uppercase;color:var(--accent);margin:0 0 16px;font-weight:500}
.lede{font-size:1.16rem;color:var(--muted);margin:0;max-width:64ch}
.rig{display:flex;flex-wrap:wrap;gap:5px 22px;margin-top:22px;
  font-family:"IBM Plex Mono",monospace;font-size:.735rem;color:var(--muted)}
.rig b{color:var(--ink);font-weight:500}
.verdict{background:var(--panel);border:1px solid var(--rule);border-left:4px solid var(--accent);
  padding:26px 30px;margin:0 0 16px;box-shadow:var(--shadow)}
.verdict p{margin:0 0 12px}.verdict p:last-child{margin:0}
.alarm{background:var(--panel);border:1px solid var(--rule);border-left:4px solid var(--no);
  padding:26px 30px;margin:0 0 44px;box-shadow:var(--shadow)}
.alarm h2{color:var(--no)}
.alarm p{margin:0 0 12px}.alarm p:last-child{margin:0}
.plate{margin:0 0 40px;padding:0 0 30px;border-bottom:1px solid var(--rule)}
.plate-h{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin-bottom:16px}
.tid{font-family:"IBM Plex Mono",monospace;font-size:.76rem;font-weight:600;color:var(--ground);
  background:var(--ink);padding:3px 8px;letter-spacing:.04em}
.crit{font-family:"IBM Plex Mono",monospace;font-size:.74rem;color:var(--muted);margin-left:auto}
.strip{display:flex;gap:14px;flex-wrap:wrap;margin:0 0 16px;padding:18px;
  background:var(--panel);border:1px solid var(--rule)}
.shot{margin:0;text-align:center;width:132px}
.shot img{width:128px;height:128px;image-rendering:pixelated;
  border:1px solid var(--rule);background:#000;display:block}
figcaption{font-family:"IBM Plex Mono",monospace;font-size:.66rem;color:var(--muted);
  margin-top:7px;line-height:1.42}
.cap{display:block}
.v{display:inline-block;margin-top:4px;padding:1px 6px;font-size:.62rem;font-weight:600;letter-spacing:.03em}
.v.ok{color:var(--ok);background:color-mix(in srgb,var(--ok) 13%,transparent)}
.v.no{color:var(--no);background:color-mix(in srgb,var(--no) 13%,transparent)}
.val{color:var(--ink);font-weight:500;margin-top:3px}
.note{max-width:66ch;color:var(--muted);margin:0 0 18px;font-size:.97rem}
.tw{overflow-x:auto;border:1px solid var(--rule);background:var(--panel)}
table{border-collapse:collapse;width:100%;font-size:.76rem}
th{text-align:left;padding:9px 12px;border-bottom:1px solid var(--rule);color:var(--muted);
  font-weight:500;white-space:nowrap;font-size:.7rem;letter-spacing:.03em}
td{padding:8px 12px;border-bottom:1px solid var(--rule);white-space:nowrap}
tr:last-child td{border-bottom:none}
.atoms{font-weight:600}
.atoms.no{color:var(--no)}
.b-fitter{color:var(--warn)}
.b-budget{color:var(--muted)}
.plane{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:2px;
  background:var(--rule);border:1px solid var(--rule);margin:0 0 14px}
.pcell{margin:0;background:var(--panel);padding:16px 14px 14px;text-align:center}
.pcell img{width:132px;height:132px;image-rendering:pixelated;border:1px solid var(--rule);
  background:#000;display:block;margin:0 auto 10px}
.dif{width:128px;height:128px;image-rendering:pixelated;border:1px solid var(--rule);
  background:#000;display:block;margin:6px 0 0}
.pcell .dif{width:132px;height:132px;margin:6px auto 0}
.dlab{display:block;font-family:"IBM Plex Mono",monospace;font-size:.6rem;color:var(--muted);
  letter-spacing:.05em;margin-top:3px}
.dnorm{max-width:66ch;font-family:"IBM Plex Mono",monospace;font-size:.68rem;color:var(--muted);
  margin:0 0 16px;line-height:1.55}
.pname{display:block;font-family:"IBM Plex Sans",sans-serif;font-size:.76rem;font-weight:600;
  color:var(--ink);margin-bottom:7px;line-height:1.3}
.pk{color:var(--muted);opacity:.75}
.verdict-row{margin-top:3px}
.prow{display:flex;justify-content:space-between;gap:8px;font-family:"IBM Plex Mono",monospace;
  font-size:.66rem;color:var(--muted);padding:2px 0}
.probe{font-family:'IBM Plex Mono',monospace;font-size:.66rem;font-weight:400;color:var(--muted);text-transform:none;letter-spacing:0;margin-left:10px}
.stripfig{margin:0 0 18px}
.stripfig img{width:100%;max-width:1066px;height:auto;image-rendering:pixelated;
  border:1px solid var(--rule);background:#000;display:block}
.stripfig figcaption{font-family:"IBM Plex Mono",monospace;font-size:.7rem;color:var(--muted);
  margin-top:7px}
.striprail{overflow-x:auto;padding:18px;background:var(--panel);border:1px solid var(--rule);margin:0 0 16px}
.warn{color:var(--warn);font-weight:600}
.acct{background:var(--panel);border:1px solid var(--rule);padding:22px 26px;margin:0 0 44px;max-width:74ch}
.acct h3{margin:0 0 10px;font-size:.98rem}
.acct p{margin:0 0 10px;font-size:.93rem;color:var(--muted)}
.acct p:last-child{margin:0}
.acct strong{color:var(--ink);font-weight:600}
.warn-note{border-left:3px solid var(--warn);padding-left:14px}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(272px,1fr));gap:18px;margin:26px 0 44px}
.card{background:var(--panel);border:1px solid var(--rule);padding:20px 22px}
.card h3{margin:0 0 8px;font-size:.95rem}
.card p{margin:0;font-size:.92rem;color:var(--muted)}
.k{font-family:"IBM Plex Mono",monospace;font-size:1.42rem;font-weight:600;
  color:var(--accent);display:block;margin-bottom:5px}
footer{margin-top:52px;padding-top:22px;border-top:1px solid var(--rule);
  font-family:"IBM Plex Mono",monospace;font-size:.72rem;color:var(--muted)}
em{font-style:italic}
a{color:var(--accent)}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>
<div class="wrap">
<header class="top">
  <p class="eyebrow">lights &middot; experiment P2</p>
  <h1>What Crisp Pixels Cost in Gaussians</h1>
  <p class="lede">Six UI content classes, fitted with anisotropic gaussian atoms under two blend
  rules and two prefilter profiles, measured against an analytic renderer drawing the same
  shapes from the same definition &mdash; then three follow-up aspects: is the gaussian the wrong
  <em>curve</em>, are wavelets a better <em>dictionary</em>, and does a fitted description survive a
  <em>zoom</em>.</p>
  <div class="rig">
    <span><b>hardware</b> @@HARDWARE@@</span><span><b>build</b> @@BUILD@@</span>
    <span><b>precision</b> @@PRECISION@@</span><span><b>method</b> @@METHOD@@</span>
    <span><b>tile</b> @@TILE@@</span><span><b>threads</b> @@THREADS@@</span>
    <span><b>pixel box variance</b> 1/12</span><span><b>alpha cutoff</b> 1/255</span>
  </div>
</header>

<div class="verdict col">
  <h2>Analytic wins every crisp class, and it is not close</h2>
  <p>One <code>clamp(0.5 &minus; sdf)</code> per pixel meets every criterion on all six targets,
  at 10&ndash;39&nbsp;&micro;s per frame from an 83&ndash;274 byte description, with no fitter and no
  failure mode. The splat arm never beats it and matches it once.</p>
  <p>Cost scales with <em>corners and curvature</em>, not area: 2 atoms for a fill, 8&ndash;10 for a
  straight edge, 14 for a 3&nbsp;px curve, 12 for a 1&nbsp;px stroke, 17&ndash;27 for a corner &mdash;
  and never, at any budget, for a 12&nbsp;px glyph.</p>
</div>

<div class="alarm col">
  <h2>The atom counts below must not be quoted</h2>
  <p>The harness is sound: every judged column is re-measured from a fresh render of the emitted
  atoms and never read from the fitter's own bookkeeping, so a broken fitter cannot lie to it.
  Deliberate breaks produced 3.6&nbsp;px, contrast 0.24 and <code>nan</code> &mdash; not plausible numbers.</p>
  <p>But in <strong>20 of 24 cells the fitter stopped, not the budget</strong>, and hand-built
  solutions beat it in both directions. One hand-placed needle meets the T5 criterion where the
  fitter needed twelve, collapsing that row from <span class="mono">never/12/never/never</span> to
  <span class="mono">2/1/2/1</span>. And a four-atom construction <em>passes</em> T6 while sitting at
  residual energy 68.19 against an empty scene's 50.74 &mdash; it passes while being worse than
  painting nothing.</p>
  <p>So every count here is an upper bound on the cost of the content class, and a statement about
  greedy matching pursuit rather than about gaussians.</p>
</div>

<div class="cols">
  <div class="card"><span class="k">1 of 6</span><h3>Classes splats win</h3>
    <p>Flat fill, where normalised blending gives interior ripple of exactly zero &mdash; and which
    analytic renders exactly, for free.</p></div>
  <div class="card"><span class="k">123&times;</span><h3>Binning waste</h3>
    <p>The axis-aligned footprint box against the true ellipse area. The needles that make an edge
    cheap in atoms are catastrophic in tile work.</p></div>
  <div class="card"><span class="k">73%</span><h3>Tile saturation</h3>
    <p>On a 512&times;512 field of 40 copies, the average tile's candidate list holds 232 of the
    320 atoms in the entire scene.</p></div>
  <div class="card"><span class="k">never</span><h3>The glyph</h3>
    <p>Not met in any of the four splat configurations at any budget. The class the question was
    about produced no number at all.</p></div>
</div>

<h2 class="col" style="margin-bottom:6px">One glyph, every strategy</h2>
<p class="note">&dagger; <em>analytic exact</em> is the offline quadrature that defines the quality
ceiling, not a contestant &mdash; at 23&nbsp;ms it is three orders slower than everything else here.
<em>analytic sdf</em> is the one a real renderer would ship. The 12&nbsp;px glyph rendered by each strategy in turn, with what it cost to store and
to draw. This is the class the whole question was about, and the four splat configurations
produced no passing result at any budget &mdash; the two analytic arms are the only ones that meet
the criterion at all. Render times are single-threaded f64 on a 64&times;64 field; treat them as
relative, not as a frame budget.</p>
<div class="plane">@@PLANE@@</div>
<div class="acct">
<h3>How count and size are accounted</h3>
<p><strong>Count and size are different quantities and never share a column.</strong> The count is
primitives for the analytic arm and atoms for the splat arm &mdash; different objects, so the unit is
named in every cell rather than left implied.</p>
<p><strong>Analytic size</strong> is the stored primitive set: 6 primitives, 274&nbsp;B for this glyph.
<strong>Splat size</strong> is atoms &times; 112&nbsp;B, because <code>Atom&lt;f64&gt;</code> is 14 scalars &mdash;
mean&nbsp;3, rotation quaternion&nbsp;4, scale&nbsp;3, opacity&nbsp;1, colour&nbsp;3. That is what this run
actually measured. The f16 row is what a shipping format would cost at 28&nbsp;B per atom.</p>
<p class="warn-note">Two caveats that both favour splats, stated so the comparison is not read as
harsher than it is. This is a <em>3D</em> atom carrying a quaternion and a z coordinate that these
2D targets never use; a 2D-native atom would be about 8 scalars. And a fair like-for-like read is
the f16 row, not f64. Even there, the smallest fit that <em>fails</em> the criterion costs more than the
analytic description that <em>meets</em> it.</p>
</div>

<h2 class="col" style="margin-bottom:6px">Aspect 2 &mdash; signed atoms, and wavelets as a rival</h2>
<p class="note">Four arms on eight classes. <strong>Gabor</strong> and <strong>Gauss-Hermite</strong> as signed,
oscillatory atoms through the <code>Kernel</code> trait; <strong>Haar</strong> and <strong>CDF&nbsp;9/7</strong> as
transforms deliberately kept <em>out</em> of it &mdash; a wavelet is a basis on a fixed dyadic lattice,
not a free-floating atom, and forcing it through a kernel interface would have made the comparison
meaningless. The one axis that makes them comparable is <strong>N-term nonlinear approximation</strong>:
keep the N largest coefficients, or keep N atoms, and plot error against N.</p>

<div class="acct">
<h3>Three answers</h3>
<p><strong>1. On content analytic cannot express, there is no clear winner &mdash; the two honest axes
point opposite ways.</strong> On <em>terms</em> the atom wins all four classes, needing 3&times;&ndash;32&times;
fewer at equal error (noise at N=64: gaussian 0.0421 against CDF 0.1097 and Haar 0.1573). On
<em>bytes</em> the transform wins all four by 3.1&times;&ndash;9.7&times;. And on <em>reach</em> the transform wins
outright: the greedy fitter <em>exhausted</em> at 61 and 50 atoms and cannot go below rel L2 0.0077 and
0.0205, while CDF&nbsp;9/7 continues to 0.0005 at 1024 coefficients &mdash; which is a fitter result, not
a representation result.</p>
<p><strong>2. The signed atom earns nothing, and how it fails is the informative part.</strong> Given a
shape set including frequency zero &mdash; making Gabor a strict superset of the gaussian &mdash; the
optimiser drove the carrier frequency <strong>to zero</strong> on all four new classes (0.000&ndash;0.006).
At f=0 the crate's own test asserts the Gabor <em>is</em> the gaussian, bit for bit. Where a real carrier
survived, at pixel scale, Gabor came out <em>worse</em> than the gaussian (0.4758 against 0.2221) &mdash;
which a superset cannot do, so it marks a <strong>search failure, not a representational one</strong>.
Gauss-Hermite lost on all eight classes at every N. <strong>Sign placement ruled: colour</strong>, which
keeps alpha non-negative so <code>Over</code> stays valid, the set stays sortable and the early-out survives.</p>
<p><strong>3. The prediction was half right, and the failure is the finding.</strong> CDF&nbsp;9/7 beats Haar
2&ndash;8&times; on smooth content, as expected, and rings at edges as expected &mdash; its peak error reaches
<strong>1.5377 against a truth whose range is at most 1.0</strong>, i.e. it overshoots, while Haar's peak
error is exactly 1.0000 and never exceeds the step. Relative L2 alone would have hidden that entirely.
But <em>"Haar should be poor on a 30&deg; edge"</em> is <strong>false</strong>: Haar leads there at N=8, 16 and
512, and reaches it <strong>exactly</strong> at 1024 terms. A smooth basis has no finite exact
representation of a step at any angle; a piecewise-constant one does.</p>
</div>

<div class="acct">
<h3>What the verify pass then demolished</h3>
<p class="warn-note">This run was built to guard against flattering the <em>transforms</em>, since they
get their N-term optimum for free. The audit found three ways they were flattered that the run never
named &mdash; the guard was pointed in the right direction and still missed what it caught.</p>
<p><strong>The byte axis rests on an unpinned convention.</strong> A coefficient is billed at 4 bytes and
an atom parameter at 8, with nothing fixing it &mdash; mutating the constant passes the whole suite.
Bill both at f32 and the atom at its <em>native nine scalars</em>, and "transforms win by
3.1&times;&ndash;9.7&times;" becomes 3.6&times;, 2.4&times;, 1.8&times;, and the soft shadow <strong>flips to the
atoms</strong>. On the carried-forward classes the atoms then sweep all eight cells.</p>
<p><strong>Half the new content was dyadically aligned.</strong> The noise classes use a lattice with
periods 16, 8 and 4 anchored at pixel 0, so cell boundaries land on dyadic block boundaries. Under a
controlled shift test CDF&nbsp;9/7 is <strong>1.69&times; better aligned than misaligned</strong>, groups
non-overlapping, while the atom arm is indifferent.</p>
<p><strong>Every transform number rides on one undocumented constant.</strong>
<code>MIN_COARSE_SIDE = 4</code> gives four decomposition levels at 64&times;64. The colour field's CDF
error at N=64 is 0.0303 at four levels, 0.6390 at three, 0.9720 at one &mdash; a <strong>21&times;
swing</strong> that no reported number states.</p>
<p>What survives unconditionally: both controls hold on independent re-measurement &mdash; a
zero-frequency Gabor is bit-identical to the gaussian through the full render path, and both
transforms round-trip to a worst max-abs of 3.55e&minus;15. <strong>Haar's exact-optimum label is
earned</strong> (energy preserved to 2.1e&minus;14); <strong>CDF&nbsp;9/7's is not</strong> (coefficient energy
off by up to 262&times;), and the run was right to refuse it.</p>
</div>

<p class="note">Strip columns, left to right: <strong>truth &middot; analytic &middot; atoms-gaussian &middot;
atoms-&beta;4 &middot; Haar &middot; CDF&nbsp;9/7</strong>. On the four new classes the analytic column is drawn as a
magenta hatch and marked <strong>ABSENT</strong> &mdash; no SDF describes a Shepard colour field, a blurred
shadow, value noise or a broadband patch, and no weak stand-in was invented for it.</p>
<p class="dnorm">The Gabor and Gauss-Hermite columns are <strong>not drawn</strong>, and that is a fitter
limit rather than a result: both are <code>KernelFamily</code> variants and the greedy fitter accepts
them, but each at a fixed carrier or lobe the crate chose, because the ladder carries no frequency,
orientation or phase knob. A column fitted that way would measure the carrier, not the atom, which is
the invented arm this run refuses everywhere else.</p>
<div class="striprail">@@A2IMG@@</div>

<h2 class="col" style="margin-bottom:6px">Aspect 1 &mdash; is the gaussian the wrong curve?</h2>
<p class="note">The atom shape had never been varied: both original kernels were
<code>exp(&minus;&frac12;&#8214;u&#8214;&sup2;)</code>, differing only in blend. A super-gaussian
<code>exp(&minus;(&frac12;&#8214;u&#8214;&sup2;)<sup>&beta;</sup>)</code> makes the shape a swept parameter,
and <strong>&beta;=1 is the gaussian</strong> &mdash; byte-identical end to end, which is the control every
other row is read against. In the table below gaussian and &beta;=1 agree to six decimals on both
columns, which is that control showing up in the data.</p>
<div class="tw"><table>
  <thead><tr><th>kernel</th><th>meets</th><th>corner gap to exact</th>
    <th>cost vs gaussian</th><th>grid overfit gap</th><th>verdict</th></tr></thead>
  <tbody>@@A1BODY@@</tbody>
</table></div>
<p class="dnorm">Cost is kernel evaluations summed over T2&ndash;T6, relative to the gaussian's
@@A1BASE@@. <em>Corner gap</em> is distance from exact box coverage on T3 &mdash; lower is better.
<em>Grid overfit gap</em> is the difference between the fitted and half-pixel-shifted grids: &beta;=8
is the only kernel above 0.4&nbsp;px, so its cheapness is aliasing cancellation at sampled pixel
positions rather than representation, and it is eliminated on that basis rather than on its
headline numbers.</p>
<div class="acct">
<h3>What the sweep settles</h3>
<p><strong>&beta; is not inert, so the kernel axis cannot be fixed at the gaussian and dropped.</strong>
On extended edges and corners a flatter shoulder wins decisively: &beta;=4 lands 0.205&nbsp;px from
exact coverage where the gaussian sits 2.365&nbsp;px away &mdash; an <strong>11&times; reduction in the
gap to the ceiling</strong> &mdash; and it is the only kernel meeting there at all.</p>
<p><strong>At pixel scale the ranking reverses.</strong> On the 1&nbsp;px stroke the gaussian and the
compact bump meet at 2 atoms while every &beta;&nbsp;&ge;&nbsp;2 fails on peak, 0.78&ndash;0.80 against a
0.9 floor. And <strong>no kernel meets the glyph at any budget</strong>, which leaves the conclusion
above standing: a different atom curve does not rescue crisp content.</p>
<p class="warn-note">The confound cuts the right way here. The greedy search is gaussian-tuned &mdash;
single-atom recovery 0.93/255 for the gaussian against 1.38/255 for &beta;=4 &mdash; so it
<em>handicaps</em> &beta;=4, and &beta;=4 still wins the corner by 2.16&nbsp;px. That makes the shape
advantage a lower bound.</p>
<p><strong>One hypothesis refuted.</strong> Exact finite support was expected to collapse the 123&times;
binning waste. It rose instead, 48.9 against 14.4 on the corner. The ratio is
4r&sup2;&radic;(s<sub>xx</sub>s<sub>yy</sub>) / (&pi;r&sup2;&radic;det) &mdash; <strong>r&sup2; cancels</strong>, so it
measures fitted-covariance anisotropy and no atom curve can move it. The waste is the
axis-aligned box against anisotropic atoms; the fix is an OBB test, not a kernel.</p>
</div>
<p class="note">Each strip below is one target under all seven kernels in sweep order, with
analytic-exact last, at 4&times;.</p>
<div class="striprail">@@A1IMG@@</div>

<h2 class="col" style="margin-bottom:6px">Aspect 3 &mdash; does a fitted description survive a zoom?</h2>
<p class="note">The last untested argument for atoms. Every number before this one was measured at a
single sampling rate, so <em>resolution independence</em> &mdash; fit once, render at any scale &mdash; had
never been checked. The experiment freezes a description and renders it up a ladder of scales, against
three references: the same fitter <strong>refitted at every rung</strong> (the control), the
<strong>analytic SDF</strong> (the bar), and two <strong>lattice-bound transforms</strong>, which are
structurally disadvantaged here because a wavelet basis is nailed to a dyadic grid that a zoom moves.
<span class="mono">max_atoms&nbsp;=&nbsp;8</span>, <span class="mono">wavelet_terms&nbsp;=&nbsp;64</span>,
gaussian kernel, f64.</p>

<p class="note"><strong>The criterion convention is ruled before any number: scene units primary,
device pixels secondary, both printed on every row.</strong> The scene does not change when the camera
zooms, so a scene criterion asks one question at every rung and a difference along the ladder belongs
to the description. A pixel criterion re-asks a four-times-stricter question at 4&times;. The analytic
bar proves the ruling is load-bearing rather than cosmetic: its 10&ndash;90 transition is <em>flat</em>
at 0.80&ndash;0.96 <strong>device px</strong> across a 32&times; span and runs 0.12&nbsp;&rarr;&nbsp;3.38 in
<strong>scene units</strong>. Same measurement, two conventions, opposite stories.</p>

<h3 class="col" style="margin:26px 0 6px;font-size:.98rem">Magnify &mdash; fitted at 1&times;, rendered out to 8&times;</h3>
<div class="tw"><table>
  <thead><tr><th>arm</th><th>1&times;</th><th>2&times;</th><th>4&times;</th><th>8&times;</th><th>rel L2, 1&times; &rarr; 8&times;</th></tr></thead>
  <tbody>@@A3MAG@@</tbody>
</table></div>

<h3 class="col" style="margin:26px 0 6px;font-size:.98rem">Minify &mdash; fitted at 8&times;, rendered in to a quarter</h3>
<div class="tw"><table>
  <thead><tr><th>arm</th><th>8&times;</th><th>4&times;</th><th>2&times;</th><th>1&times;</th><th>0.5&times;</th><th>0.25&times;</th><th>rel L2, 8&times; &rarr; 0.25&times;</th></tr></thead>
  <tbody>@@A3MIN@@</tbody>
</table></div>
<p class="dnorm">The bold cell in each row is the rung the description was fitted at. <span class="mono">n/a</span>
is a probe the target does not carry &mdash; the edge and corner classes are judged on the 10&ndash;90
transition and the stroke and glyph classes on FWHM, and neither is invented for a target with no probe
of that kind. The colour field carries neither and is read on rel L2 alone.</p>

<p class="warn-note"><strong>The glyph gate is loose, and the table shows it.</strong> FWHM within
&plusmn;0.5 of a one-unit stem passes a Haar arm sitting at rel L2 <strong>0.676</strong> beside an
analytic arm at <strong>0.092</strong>. A stroke can be the right <em>width</em> and still be the wrong
picture. So the glyph conclusion here is drawn from the rel L2 ratio &mdash; frozen atoms
2.051&times; worse across the ladder while the bar improves 2.8&times; &mdash; and never from the verdict
column, which is why both are printed side by side.</p>
<div class="acct">
<h3>What the ladder settles</h3>
<p><strong>Resolution independence is real, and it is narrow.</strong> One frozen 8-atom description of
the 30&deg; edge meets its criterion at <em>every rung of both ladders</em> &mdash; a 32&times; span from
0.25&times; to 8&times; &mdash; and under the unprefiltered profile it does so <em>where a fresh fit at
that rung fails</em>, at 0.5&times; and 0.25&times;. The control is not a ceiling: it is one greedy
trajectory, and freezing beat it.</p>
<p><strong>The glyph is the product question, and frozen atoms lose it.</strong> Across magnification
the frozen set decays 0.223&nbsp;&rarr;&nbsp;0.458 rel L2, a ratio of <strong>2.051</strong>, while the
analytic bar goes the other way, 0.257&nbsp;&rarr;&nbsp;0.092. Text drawn from a frozen atom set gets
worse exactly where a user zooms in to read it.</p>
<p><strong>The corner is analytic-only at every scale.</strong> No atom arm meets it at any rung, in
either direction, under either profile &mdash; consistent with aspect 1, where only &beta;=4 met it at
a single scale.</p>
<p><strong>Analytic is not automatically minification-safe.</strong> The SDF bar meets from 1&times; up
and <em>fails</em> at 0.5&times; and 0.25&times; on the edge. Resolution independence is a property of a
description <em>plus</em> a resampling story, and the analytic arm has no prefilter either.</p>
<p class="warn-note">And <strong>the prefiltered profile is worse, not better, under minification</strong>
&mdash; frozen atoms lose 1&times; and everything below it. That is the opposite of what a pixel
prefilter is for, and it is the strongest evidence yet for the parked item that says
<span class="mono">screen_dilation = 0.3</span> is being applied as one constant where Mip-Splatting
states it is a sum of a screen half and an object half.</p>
</div>

<div class="acct">
<h3>Two ways this table could have lied, one of which it did</h3>
<p class="warn-note"><strong>The machine-readable file disagreed with the printed one, and the
machine-readable file was wrong.</strong> The TSV emitted a single <span class="mono">meets_scene</span>
column carrying the <em>transition</em> verdict only, gated on nothing but "was quality measured".
For a class judged on FWHM the transition is <span class="mono">nan</span>, and
<span class="mono">nan&nbsp;&lt;&nbsp;max_transition</span> is <em>false</em> in Rust &mdash; so every
glyph row read <strong>no</strong> where the honest answer is <em>this probe does not apply</em>, and the
FWHM verdict, which the code computes, was never written to the file at all. Read straight, the table
said analytic SDF never meets the glyph at any rung, contradicting the P2 result above. Computed from
the right probe it meets at every rung: FWHM 0.937, 0.947, 0.933, 0.950 against a target of 1 with a
tolerance of 0.5. Fixed before this page was built &mdash; the column is now
<span class="mono">meets_transition_*</span>, the FWHM verdict has its own column, an absent probe
prints a dash, and a regression case drives a glyph ladder and an edge ladder and asserts both, with a
mutation that makes the verdict ignore probe existence proving the case can fail.</p>
<p><strong>The decay column has a denominator, and the denominator moves.</strong> Relative L2 divides
by the norm of the truth <em>at that rung</em>, so where an arm's error lives decides how its decay
reads even when the error never changes. An error confined to a boundary of fixed device-pixel width
covers a shrinking share of a growing image and decays at &radic;(base/side) for free. The analytic bar
proves this is not pedantry: its decay sits <em>on</em> that boundary reference to within two percent at
every rung and its max&nbsp;|e| never moves at all &mdash; so all of its apparent improvement under
magnification and all of its apparent collapse under minification are the denominator and none of it is
the description. Every row therefore prints both references.</p>
<p><strong>Do not cross-quote this table with the one at the top of this page.</strong> P2 ran
<span class="mono">--max-atoms 48</span> against a pass criterion; this ran <span class="mono">8</span>
against probe widths. An arm that "meets" here has met a narrower question.</p>
</div>

<p class="note">Each strip is one class across the whole ladder, frozen description, unprefiltered
profile. Images are downscaled for the page; the originals are in
<span class="mono">docs/lights/p5/</span>.</p>
<div class="striprail">@@A3IMG@@</div>

<h2 class="col" style="margin-bottom:6px">The six targets</h2>
<p class="note">Images are 64&times;64, shown at 2&times; with hard pixel edges &mdash; at this scale the
measurement is sub-pixel, so smoothing them would hide the thing being measured.
<span class="mono">atoms&nbsp;fitted</span> is how many the fitter produced;
<span class="mono">meets&nbsp;at</span> is the count at which the criterion was first satisfied, and
a dash there means it never was at any budget. They are different questions and never share a cell.
<span class="mono">stopped&nbsp;by</span> is the column that exposed the flaw above.
One cell needs a caveat the report cannot express: T5 under <em>Over &middot; no prefilter</em> meets
at 2, but the sweep showed it does not <em>hold</em> from 2 onward &mdash; it is a crossing, not a
stable pass.</p>
@@PLATES@@

<footer>
  Generated from <span class="mono">report.tsv</span>, 36 measured rows, run
  <span class="mono">lights-p2 --max-atoms 48</span>. Quadrature depth 10, undecided area
  0.508&nbsp;px&sup2;; paint order frozen at <span class="mono">z = 64 &minus; i/64</span>.
  PSNR, SSIM and MSE appear nowhere by design &mdash; a 12&nbsp;px glyph is a rounding error in any
  image-wide error budget. Aspects 1&ndash;3 append their own runs below: the atom-curve sweep
  (<span class="mono">lights-p3</span>), the dictionary comparison (<span class="mono">lights-p4</span>)
  and the zoom ladder (<span class="mono">lights-p5 --classes t2,t3,n1,t6 --direction both --profile
  both</span>, <span class="mono">max_atoms&nbsp;=&nbsp;8</span>). Each names its own criterion; the
  four tables are not in one currency and are not to be cross-quoted.
</footer>
</div>
"""
subs = {"@@HARDWARE@@": meta.get("hardware","&mdash;"), "@@BUILD@@": meta.get("build","&mdash;"),
        "@@PRECISION@@": meta.get("precision","&mdash;"), "@@METHOD@@": meta.get("method","&mdash;"),
        "@@TILE@@": meta.get("tile","&mdash;"), "@@THREADS@@": meta.get("threads","&mdash;"),
        "@@PLATES@@": "".join(plate_html), "@@PLANE@@": plane_html, "@@A1BODY@@": a1_body, "@@A1BASE@@": a1_base, "@@A1IMG@@": a1_img, "@@A2IMG@@": a2_img,
        "@@A3MAG@@": a3_mag, "@@A3MIN@@": a3_min, "@@A3IMG@@": a3_img}
for k, v in subs.items():
    html = html.replace(k, v)

io.open(OUT, "w", encoding="utf-8").write(html)
print("wrote", OUT, os.path.getsize(OUT), "bytes,", len(rows), "data rows")
