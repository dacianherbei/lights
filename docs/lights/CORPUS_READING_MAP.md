# The gaussian_splatting corpus — what each paper is for

**Date:** 2026-09-08
**Method:** three verification workflows. The first judged the 13 PDFs in
`docs/gaussian_splatting/` against the `lights` renderer phases alone and
concluded the corpus was "assembled for a different project". That was a
framing error — see §2. The corpus was re-judged against three real
objectives, and a critic was pointed specifically at over-correction.

**Verdict on the collection: it was well chosen.** Eleven of thirteen track
one coherent thesis — *an anisotropic gaussian field as an editable,
resolution-independent, compact representation* — approached from every side
a person actually needs: fit it (Image-GS), budget it (MEGS²), segment it
(GPSToken), label it (VALA), constrain its shape (GeoSplat), isolate a part
(CoRe-GS), condition generation on it (G2Editor), render it fast (GS-TG),
store it (AVS), and one control showing what dense-and-wrong looks like
(inpainting). A deliberate sweep, not an arXiv trawl.

---

## 1. The table

| Paper | lights | iris | compression | other |
|---|---|---|---|---|
| **Image-GS** (SIGGRAPH 2025) | **USE** | **USE** | **USE** | vision-safe compression of references |
| GPSToken | SKIP | REFERENCE — entropy partition (Eq. 4) only | SKIP | generative variants (D) |
| 2DGS Inpainting | REFERENCE — Cholesky covariance | **SKIP** | **SKIP** | robustness to degraded refs (E) |
| **VALA** | **USE** | REFERENCE — streaming cosine median | REFERENCE | per-atom attribution as debug |
| CoRe-GS | SKIP | REFERENCE — Table III budget split | SKIP | matte-key utility |
| G2Editor | SKIP | REFERENCE | SKIP | generative variants (D) |
| SWAGSplatting | NOTE, DO NOT CITE | REFERENCE — freeze-then-refit schedule | SKIP | depth-dependent formation |
| AVS Geometry Compression | SKIP | SKIP | **SPLIT** — skip for themes, later for the canvas scene | streaming a time-varying scene (F) |
| **MEGS²** | REFERENCE | **USE** | **USE** | κ is the LOD knob |
| GeoSplat | SKIP | REFERENCE — Prop 3.1 area-invariance | SKIP | fitting from few observations |
| **GS-TG** | **USE** | SKIP | SKIP | canvas-scale tiling |
| Particle Merging-and-Splitting | REFERENCE — under the *animation* objective | SKIP | SKIP | **merge/split over time — sole entry** |
| Distributed Isosurface | REFERENCE — §III protocol only | SKIP | SKIP | probe methodology; bibliography node |

**Six SKIPs survived full scrutiny and are not softened:** inpainting for
iris and compression (a uniform grid at 1.27 atoms/pixel, ~2,500× denser
than Image-GS, no compactness term in the loss, nothing extrapolates past
the input canvas); AVS for theme storage (a ~25% ceiling by its own law);
GS-TG for iris; Particle Merging for iris and compression; the Distributed
abstract for iris and compression.

---

## 2. What the first triage got wrong

It was asked *"which renderer phase does this paper serve?"* An analysis /
inverse paper serves no rendering phase, so the frame could only return
SKIP. The objective set handed to the reader was incomplete — and two of
the missing objectives are written down in this repository.

- **Image-GS**, marked peripheral, is the centre of the corpus and USE on
  all three objectives at once. The renderer frame asked what it *renders*;
  it renders nothing new — it **fits**, which is the whole of the iris
  objective.
- **VALA**, dismissed as a "language" paper. Its two mechanisms contain no
  language content; CLIP merely supplies the vectors being aggregated. A
  gate over `alpha*T` and a streaming geometric median are renderer and
  estimator mathematics.
- **MEGS², CoRe-GS, GeoSplat, GPSToken** — all filtered on *domain* when
  their *mechanisms* (pruning to a byte budget, a coarse-then-focused
  schedule, shape priors that pay off when observations are few, a
  non-neural entropy partition over a Sobel map) are exactly the
  segmentation, allocation and compaction machinery iris needs.
- **Particle Merging-and-Splitting** — missed by *both* runs. It is the only
  paper of thirteen about primitives changing over time, against a settled
  plan requirement that every decoration declare `static|animated`.

**Domain was never the right filter for any paper here.** Every one of them
manipulates a fitted anisotropic gaussian field, which is the object both
objectives need.

---

## 3. The Image-GS edge result, and the claims it refuted

Image-GS **does not address** hard edges or text — it never uses the word
"edge" (0 occurrences), "sharp" (0), "glyph" (0), "font" (0), and has no
text image in its evaluation. But its Eq. (5) settles the question anyway.

**It normalises rather than alpha-composites**: the top-K weighted colour
sum is divided by the sum of the weights — a partition of unity. Under that
rule:

- a straight edge costs **two atoms**; in the limit of separation, exactly a
  half-plane
- 10–90% transition width = **4.394·σ²/d**, verified numerically to 4 dp
- with anisotropy the perpendicular profile is independent of the along-edge
  scale, so an edge of **any length** is still two atoms

**The corrected statement:** gaussians are cheap on flat fills and on
straight boundaries of any length; they get expensive per unit of
**curvature**. A 12px glyph is ~20 atoms by the flattening bound (terminals,
junctions, 2–4px counters). A 1px panel border is 2–5 atoms and
length-independent.

Claims this refuted, recorded so they are not repeated:

| Claim | Verdict |
|---|---|
| gaussians terrible at hard edges | **split** — true only of Image-GS *as configured* (σ∈[5,10], fp16 means) |
| a step costs unbounded atoms | **refuted** |
| 30dB/160KB is "on photographs, the favourable case" | **refuted** — it is a stylised illustration; photographs are the paper's *unfavourable* case |
| a UI is ~90% crisp chrome | no source; by area a node graph is mostly flat fill, the cheapest content there is |
| hybrid split at "text, borders, 1px strokes" | right conclusion, **wrong split line** — borders are cheap; text is the expensive one |
| authored rects don't need a splat renderer | **confirmed** — the strongest of the set, and untouched |

**Consequence for P2:** atom counts are renderer-dependent by ~10×, so the
blend rule must be settled before any count means anything, and the content
classes must be measured separately.

---

## 4. Citation hygiene

- **SWAGSplatting's** headline (+2.70/+3.48 dB, +11.42% SSIM, −34.80% LPIPS)
  is computed against the arithmetic **mean of six baselines**, not the best.
  Against the method it extends: +0.34 and +0.28 dB. Its own ablation prices
  its headline mechanism at 0.91% PSNR. Its formation model is standard
  exponential fog — cite the Schechner lineage, not a five-page preprint.
- **AVS Geometry Compression** claims 40% against a measured 25.64%, and its
  prose disagrees with its own Table 1.

---

## 5. Acquisition list — verified

A search across four fronts returned **24 papers with exactly ONE overlap**
with the 13 already owned (Image-GS). The collections are complementary, not
redundant. Every entry below was verified against a primary source; the
unverified ones are named in §7 rather than listed here.

### Read two you already own, first

- **Particle Merging-and-Splitting** — this *is* the merge/split machinery
  for the LOD gap. Read before acquiring Runnalls or 3DGS-MCMC.
- **GS-TG** — read against StopThePop's tile culling before implementing
  either.

### Top three

| Paper | id | What it settles |
|---|---|---|
| **Mip-Splatting** — **not on disk** | arXiv:2311.16493 | Splits a 0.3 dilation into 0.1 screen-space + 0.2 object-space. Its Eq. 10 prefactor `sqrt(\|Σ\|/\|Σ+sI\|)` is character-for-character the `gain` computed at `kernel/gaussian.rs` by `moment_prefilter` and applied at `method/splat/project.rs` under `if settings.energy_compensation {` |
| **EWA Volume Splatting** (Zwicker, VIS 2001) — **not on disk** | ACM 10.5555/601671.601674 | The prefilter is **integral-preserving, not peak-preserving** — so `energy_compensation` is the framework, not an option. `V_h = I` is the derived low-pass. One resample filter covers the whole zoom range: **no mip pyramid, no min/mag branch** |
| **GaussianImage** (ECCV 2024) — **not on disk** | arXiv:2403.08551 | Accumulated summation instead of alpha compositing — permutation-invariant, so **the per-tile depth sort is deleted**. A whole-pipeline fork for the 2D target, expressible through the existing blend-mode seam. 8 floats/atom |

**Withdrawn 2026-09-28 (L-101).** The Mip-Splatting row first said the gain is
"discarded at `project.rs`" and that the paper "decomposes our `screen_dilation = 0.3`".
Both were true of `reference()` and `contract()` when this map landed (47946dca, 8d6bda7a,
2026-09-08) and both stopped being true the same day: 6ac92f33 added `prefiltered()`
(`src/settings.rs`, `screen_dilation: PIXEL_BOX_VARIANCE`, `energy_compensation: true`),
and from then on the gain is applied, not discarded. No selectable profile renders at 0.3
(`tests/prefilter.rs::no_selectable_profile_reaches_the_reference_dilation`), and the crate
builds the screen half only, so there is nothing of ours to decompose (L-37). The sentence
was copied into `docs/plan/TWO_HORIZON_PLAN.md` a day after it went false; L-37 corrected
that copy. None of the three papers above is in `docs/gaussian_splatting/` — each row says
so, and `tests/prose_pointers.rs` keeps the mark in step with the folder.

### This month — these change architecture

| # | Paper | id | For |
|---|---|---|---|
| 4 | Hierarchical 3D Gaussian Representation | arXiv:2406.12080 | LOD; `TWO_HORIZON_PLAN` §5 asks for this by name |
| 5 | StopThePop | arXiv:2402.00525 | **3D only** — see the ruling in §6 |
| 6 | Analytic-Splatting | arXiv:2403.11056 | Replaces the point sample with a closed-form pixel-area integral. Both our `PixelConvention` variants are point samples |
| 7 | Instant GaussianImage | arXiv:2506.23479 | Fit speed for iris *(venue partially verified)* |
| 8 | GES: Generalized Exponential Splatting | arXiv:2402.10128 | The super-gaussian atom, on the Kernel axis |
| 9 | DARB-Splatting | arXiv:2501.12369 | Decaying anisotropic radial bases *(preprint)* |

### When you hit that problem

Sequential Point Trees (10.1145/882262.882321) · Gaussian mixture reduction:
Salmond (Xplore 1255321) and the KL approach (10.1109/TAES.2007.438358) ·
Deformable Beta Splatting (arXiv:2501.18630) · **3DGabSplat**
(arXiv:2508.05343, *author list uncaptured*) · Surface Splatting
(10.1145/383259.383300) · Anisotropic Kuwahara
(10.1111/j.1467-8659.2009.01574.x) · Weighted Voronoi Stippling
(10.1145/508530.508537) · GaussianImage++ (arXiv:2512.19108)

### The control arm — P2 cannot conclude without it

MSDF improved corners (10.1111/cgf.13265) · GPU font rendering from glyph
outlines (JCGT) · Real-time analytic antialiased text (10.1111/cgf.13757) ·
Analytic rasterization with polynomial filters (10.1111/cgf.12070)

### iris route

Palette-based decomposition and recoloring (10.1145/3272127.3275054 — top
pick) · Diffusion Curves (10.1145/1360612.1360691)

---

## 6. Two rulings the new papers settle

**The signed/alpha rule is narrower than this project had assumed.**
3DGabSplat and NegGS both put the sign in the **colour/emission** while
opacity stays non-negative, so standard front-to-back alpha compositing
remains valid *and sortable*. Only sign in the **density/kernel weight**
(true matching-pursuit dictionaries) forces an additive signed buffer. The
crate's `Additive` path is therefore needed for the **iris dictionary-fitting**
target, not necessarily for the UI target.

**StopThePop does not apply to 2D orthographic pan/zoom.** Its popping
mechanism is order inversion along a ray under camera *rotation*; under
orthographic projection all rays are parallel, so centre-depth order equals
per-ray order at every pixel. **`OrderingModel::CentreDepth` is provably
sufficient for the node-graph target** — assert it as a theorem in the suite
rather than paying for a resort window.

---

## 7. Gaps — measure, do not read

**No head-to-head of gaussians versus SDF/vector on text or UI content
exists.** Searched five ways across two independent passes. Every 2D gaussian
paper benchmarks against neural fields, learned codecs, JPEG or block texture
compressors on photographic or artistic imagery; no font-rendering paper has
a gaussian arm. **P2, run properly, is a first measurement, not a
replication.**

Also unanswered anywhere, each stated as a measurement:

- No prefiltering or LOD theory for a purely 2D orthographic gaussian
  decomposition re-rendered at another size (the iris case). The 2D
  specialisation `Σ_screen = s²Σ + σ²I` is a corollary nobody states.
- Nothing establishes what, if anything, pops under 2D orthographic
  pan/zoom. Write the centre-depth sufficiency claim as a test.
- No LOD or mixture reduction for a **signed** atom field — every LOD paper
  found assumes non-negative opacity.
- No analysis of clamping or order-dependence hazards when negative values
  pass through front-to-back alpha compositing.
- No benchmark of fitting methods at a fixed atom budget. Every speed number
  quoted is from a different paper on a different GPU; since `lights` is CPU,
  these are ordering hints, not budgets.

### Citation hygiene

**DO NOT CITE** — "Prefiltering Antialiasing for General Vector Graphics"
(on-topic abstract on Academia/ResearchGate, but authors, venue and DOI
unconfirmable; Manson & Schaefer CGF 2013 covers the same ground, peer
reviewed and open) and "Generalized non-exponential Gaussian splatting"
(arXiv:2603.02887, seen only as a search-result string).

**Partially verified, cite with care:** 3DGabSplat (equations read, authors
not captured), Greene & Heckbert 1986 (record confirmed, text unread),
Heckbert's 1989 thesis (report number unverified), Bergeaud & Mallat 1995 and
Figueras i Ventura 2006 (DOIs inferred), Octree-GS (published venue
unconfirmed).

**Note:** the Discontinuity-Aware 2D Neural Fields project page prints a DOI
belonging to a different paper; the correct one is 10.1145/3618379.

**Refused rather than invented:** no canonical paper fits a gaussian mixture
to an image by EM treating pixel intensity as a 2D spatial density. The
verified substitute is Secord's weighted Lloyd relaxation for positions plus
Kyprianidis's structure-tensor eigendecomposition for covariance.
