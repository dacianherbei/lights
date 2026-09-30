# P2 — what crisp UI content costs in gaussian atoms

> **This is one run's long-form write-up, not the record.** The consolidated
> research record is `docs/lights/RESEARCH.md` (source `RESEARCH.json`, read as a
> tree at <https://claude.ai/code/artifact/aaf71a60-76fb-4964-aad3-a91a3e9be68f>),
> and it carries this run's corrections in full. Four figures below are corrected
> there and must not be quoted from this file alone: the analytic SDF arm painted
> coverage from a distance LOWER BOUND; every `render_ns` here was measured on
> 18 cores under a header that says `threads 1`; the tile-occupancy figure named a
> binner this crate no longer has; and the stop count was 22 of 24, not 20.
>
> **This run now has artefacts.** `docs/lights/p2/run-48/` and
> `docs/lights/p2/run-8-tiled/` carry the two invocations' tables, each stamped
> with the command that produced it, and `crates/lights/tests/p2_artefacts.rs`
> reads them. Every corrected figure below was re-derived from them on 2026-09-15.


**Date:** 2026-09-08
**Runs — there are TWO, and no single one produced every figure below.** Both
need `--out-dir`; the binary exits 2 without it, so the line this file carried
until 2026-09-15 (`lights-p2 --max-atoms 48`) could not be pasted into a shell:

```
cargo run -p lights --all-features --bin lights-p2 -- --out-dir docs/lights/p2/run-48 --max-atoms 48
cargo run -p lights --all-features --bin lights-p2 -- --out-dir docs/lights/p2/run-8-tiled --max-atoms 8 --tiled-field
```

The first is the headline and the atom counts; the second is every tile figure,
and its 8 atoms × 40 copies is where the population of 320 comes from. Both exit
1 — a cell never met its criterion inside `--max-atoms`, which is a RESULT.
macOS aarch64, f64, splat-raster, global sort, tile 16, `PIXEL_BOX_VARIANCE = 1/12`,
`alpha_cutoff = 1/255`, paint order frozen at `z = 64 − i/64`. The 2026-09-08 run
was release and the committed tables are debug. That the two agree on every
deterministic column is MEASURED, not assumed: `box_over_ellipse_area` at
123.406101, the four glyph atom counts, and — with the 2026-09-08 binner restored
— `mean_tile_list` at 232.0234 all reproduce bit-exactly against the release
figures published here. Wall-clock columns do not carry over, and the tables say
so themselves in their `#build` header.
**Interactive page** with all images and per-pixel error maps:
<https://claude.ai/code/artifact/3577a16a-ad94-4f7f-bea6-3727dbe65bed>
That page now also carries aspects 1, 2 and 3. **Aspect 3 — resolution
independence** is written up in `docs/lights/P5_RESULTS.md`: one frozen atom
description of a 30° edge holds its criterion across a 32× zoom span and beats a
fresh fit at the small rungs, the corner is analytic-only at every scale, and the
analytic bar itself fails below 0.5×. The four tables on that page are NOT in one
currency — P2 ran `--max-atoms 48` against a pass criterion, aspect 3 ran 8
against probe widths — and must not be cross-quoted.

---

## The answer

**Analytic rendering beats gaussian splatting on every crisp UI content
class, and it is not close.** One `clamp(0.5 − sdf)` per pixel meets every
criterion on all six targets at 10–39 µs per frame from an 83–274 byte
description, with no fitter and no failure mode.

The glyph, which is the class the whole question was about:

| arm | count | size f64 | size f16 | render | meets |
|---|---|---|---|---|---|
| **analytic sdf** | 6 prim | **274 B** | — | **38.9 µs** | **yes** |
| Over · no prefilter | 15 atoms | 1,680 B | 420 B | 81.5 µs | no |
| Over · prefiltered | 22 atoms | 2,464 B | 616 B | 96.1 µs | no |
| Normalised · no prefilter | 12 atoms | 1,344 B | 336 B | 54.9 µs | no |
| Normalised · prefiltered | 12 atoms | 1,344 B | 336 B | 120.7 µs | no |

Splat sizes are atoms × 112 B, an `Atom<f64>` being 14 scalars: mean 3,
rotation quaternion 4, scale 3, opacity 1, colour 3. Those are the costs of
*failing*; none of those fits meet the criterion.

**Splats win exactly one class.** Flat fill under normalised blending: 2
atoms, interior ripple **exactly 0.000000** — a partition of unity reaching a
rendered pixel, which alpha compositing provably cannot do (`Over` leaves
0.0002). It is also a filled rectangle, which analytic renders exactly for free.

**Cost scales with corners and curvature, not area:** 2 atoms (fill) → 8–10
(straight edge) → 14 (3 px curve) → 12 (1 px stroke) → 17–27 (corner) →
never (glyph). A single "atoms per glyph" figure would have hidden a ~10×
spread.

**The binning result inverts the atom-count win — and the fix has since been
built.** The axis-aligned footprint box reaches **123×** the true ellipse area.
That figure is the cross-cell maximum of a per-cell mean over atoms, it lands on
t5-thin-stroke under normalised blending and the contract profile, and it
reproduced bit-exactly at **123.406101** on 2026-09-15 from
`docs/lights/p2/run-8-tiled/report.tsv`. Aspect 1 later proved it is not a kernel
property: the ratio is `4r²√(sxx·syy) / (πr²√det)`, `r²` cancels, so it measures
fitted-covariance anisotropy.

**WITHDRAWN 2026-09-15 — "the *average* tile's candidate list holds 232 of the
320 atoms in the scene", and with it "an 8-atom edge is 8 atoms tested against
essentially every pixel".** Both sentences are true only of the binner this crate
had on 2026-09-08. Commit 6b1b9ab8 (2026-09-14) gave `bin_and_sort` a per-tile
rejection against the atom's own whitened quadratic form: `span_of_box` still
supplies the candidate rectangle, but each tile in it is then dropped unless the
ellipse actually reaches the tile. The same cell that published 232 of 320 —
t2-straight-edge, `Over`, prefiltered — now measures **25.747070 of 320**, a
**9.01× prune**, and the whole 24-cell tiled sweep peaks at 30.58 of 320. The old
number is reproducible: delete that one rejection and the cell returns **232.0234**.

The area ratio did not move and the tile work did, which is the useful half of
the correction: `box_over_ellipse_area` measures the ATOMS, so it is unchanged at
80.691765 on that very cell, while the pairs it used to imply are gone. Quoting
123× as a statement about tile cost is the mistake this paragraph now exists to
prevent.

**The Image-GS premise did not reproduce.** This crate's `Normalised` is a
coverage *mix*, `background·(1−c) + (weighted/weight)·c` with `c = min(weight,1)`
— not Eq. 5's pure top-K ratio. It reduces to Eq. 5 only where total weight
reaches one, and an edge sharp enough to pass carries midpoint weight far
below a half, so the mix fades toward the background exactly at the edge.
Normalised lost to `Over` on all five non-fill targets.

---

## Why the atom counts must not be quoted

The harness is sound — every judged column is re-measured from a fresh render
of the emitted atoms, never read from the fitter's bookkeeping, so a broken
fitter cannot lie to it. Deliberate breaks produced 3.6 px, contrast 0.24 and
`nan`, not plausible numbers.

But **in 22 of 24 cells the fitter stopped, not the budget** — corrected
2026-09-15 from 20, which no reading of `docs/lights/p2/run-48/report.tsv`
produces; the two cells that did reach `--max-atoms 48` are the right-angle
corner under normalised blending, both profiles — and hand-built solutions beat
it in both directions:

- **Downward** — one hand-placed needle meets the 1 px stroke criterion where
  the fitter needed twelve; all four stroke cells collapse from
  `never/12/never/never` to `2/1/2/1`. The fitter minimises L2 energy, and the
  L2-optimal stroke does *not* meet the criterion while a deliberately-too-thin
  needle does.
- **Upward** — a four-atom construction *passes* the glyph criterion while
  sitting at residual energy 68.19 against an empty scene's 50.74. It passes
  while being worse than painting nothing.

So every count is an upper bound, and a statement about greedy matching
pursuit rather than about gaussians. Closing that needs joint gradient
descent, which the crate does not have — now a backlog item in
`docs/plan/TWO_HORIZON_PLAN.md` §8, with the calibration measurement that
would turn this confound into a number.

---

## Aspect 1 — the atom curve, and it is not inert

β=1 of the super-gaussian is SHA-256 byte-identical to the gaussian
end-to-end, which is the control everything below is read against.

| kernel | meets /6 | corner gap to exact | cost vs gaussian |
|---|---|---|---|
| β=0.5 | 1 | +2.850 px | 1.75× more — **eliminated** |
| gaussian (β=1) | 2 | +2.365 px | 1.00× |
| β=2 | 2 | +0.757 px | 1.82× cheaper |
| **β=4** | **4** | **+0.205 px** | **2.39× cheaper** |
| β=8 | 2 | +0.808 px | 4.00× cheaper — **eliminated** |
| compact bump | 2 | +1.907 px | 1.40× cheaper |

β=4 closes the corner gap **11×** and is the only kernel meeting there. The
fitter is gaussian-tuned (single-atom recovery 0.93/255 vs 1.38/255 for β=4),
so it *handicaps* β=4 — making that a lower bound on the shape advantage.

β=8 was eliminated on evidence rather than headline numbers: it is the only
kernel tripping the sample-grid overfit flag at 0.51 px between fitted and
half-pixel-shifted grids, so its wins are aliasing cancellation at sampled
pixel positions, not representation.

**At pixel scale the ranking reverses** — on the 1 px stroke the gaussian and
the compact bump meet at 2 atoms while every β ≥ 2 fails on peak (0.78–0.80
against a 0.9 floor). And no kernel meets the glyph at any budget.

---

## What this settles, and what it does not

**Settled:** analytic renders crisp authored UI chrome smaller, faster and
correctly. `lights` should not draw fills, borders, rules, wires or text.
For the role that survives, **β=4 is the default, not the gaussian** — on
everything larger than a pixel.

**Not settled, and it is the half that matters for `lights` and `iris`:**
every target tested was a shape an SDF renders exactly, from a description
you already hold. The content splats exist for — photographs, soft shadows,
band-limited noise, fitted textures, continuous fields — has **no analytic
arm at all** and is entirely unmeasured. This work bounds where splats lose.
It says nothing about where they win.

**Also unmeasured:** resolution independence. Everything here is at one scale,
and one fitted description surviving a zoom range is the property a
per-resolution rasteriser lacks.

---

## Correction, 2026-09-14: the analytic arm's residual columns were wrong on two targets

Found while closing register defect L-6. **Every quality conclusion on this page stands. Two residual
figures and two strip images do not.**

`src/fit/analytic.rs` painted coverage as `clamp(0.5 - target.sdf(centre), 0, 1)`, and `Target::sdf`
is `sign(classify(p)) * clearance(p)` where `clearance` is `min_i |sdf_i|` over the **primitives** —
which `target_rs_comment/one_definition.md` calls *"a certified LOWER bound on the distance to the
combined boundary"* and closes with *"the sign is a fact, the distance is a bound… conflating them is
the mistake this file exists to prevent."* The arm conflated them. Because the bound understates the
distance, `0.5 - d` **overstates** coverage: the arm painted ink on open paper.

The witness: on the right-angle corner, the first edge's *line* runs past the apex through territory
outside the wedge, so `clearance` reads ~0 all along it. Pixel (26,41) — at least **10.93 px outside
the shape** by brute-force distance to a dense boundary sample — was painted **0.487** coverage. A
half-lit ray clean across a 64×64 image.

The fix is one line: read the shape's own CSG value, `target.combined(centre)`.

**Residuals that change** (measured through the real `measure_target` / `residual_energy` path):

| target | residual energy before → after | residual peak before → after |
|---|---|---|
| t3 right-angle corner | 18.536 → **0.132** (140×) | 0.843 → 0.058 |
| t6 glyph | 3.343 → **0.382** (8.8×) | 0.856 → 0.308 |

T1, T2, T4 and T5 cannot move: one primitive and no hole means `combined` and `sign*clearance` are the
same number everywhere. **The two strip PNGs for t3 and t6 made before today are wrong.**

**What does NOT change, and this is the other half of the finding.** `transition_10_90`, `fwhm`,
`peak_frac`, `contrast_frac`, `interior_ripple` and every `meets` verdict are **identical on all six
targets** before and after — because every probe window sits on a real edge, where the nearest
primitive surface *is* the boundary and the two distances agree. So the criterion, the calibration
constants, and **every atoms-versus-analytic quality conclusion on this page stand unchanged**. What
was wrong was the arm's picture and its residual, not its measured edge.

`the_two_control_arms_differ_so_the_cheap_one_is_not_the_ceiling` still holds — the arms still differ
by more than 0.02 on every target with a boundary — so the cost comparison is still against a real arm
and not a straw man.

## Correction, 2026-09-14: the `threads 1` in this page's rig line was not what the run used

Found while closing register defect L-34. The header literal is `threads: 1` at
`src/fit/harness.rs:737`. But `ProfileChoice::settings()` returns `contract()` or `prefiltered()`,
both of which inherit `threads: 0` from `reference()`, and `src/method/splat/implementation.rs` reads
**0 as the rayon global pool** — 18 workers on this machine — while only 1 means serial.

**So every `render_ns` column in this run was measured on all 18 cores under a header that said 1.**
The per-frame microsecond figures quoted on this page are multi-core numbers. Their *ranking* is
unaffected (every arm was measured the same way), but none of them is the single-threaded cost the
header claims.

`lights-p2` now computes the count from the settings the run actually passed, resolving 0 through
`available_parallelism()`, and writes it into both the file and the printed header — never a literal.
The same literal in `harness.rs` is recorded as owed.
