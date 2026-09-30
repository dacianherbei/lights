# Aspect 3 — resolution independence: what a frozen atom set is worth under zoom

> **This is one run's long-form write-up, not the record.** The consolidated
> research record is `docs/lights/RESEARCH.md` (source `RESEARCH.json`, read as a
> tree at <https://claude.ai/code/artifact/aaf71a60-76fb-4964-aad3-a91a3e9be68f>).
> §C3 below is corrected there: aspect 3 never rendered at `screen_dilation = 0.3`
> — it ran at `PIXEL_BOX_VARIANCE = 1/12` — and the blur it measured cannot be
> "the object half", because the crate has no object-space term at all.


**Date:** 2026-09-09; **artefacts re-measured 2026-09-15** (BURNDOWN L-144).
**Run, as it can be pasted:**

```
cargo run -p lights --all-features --release --bin lights-p5 -- \
  --out-dir <REPO>/docs/lights/p5 --classes t2,t3,n1,t6 --direction both --profile both
```

macOS aarch64 (M5 Max), release, f64, gaussian kernel, `max_atoms = 8`,
`wavelet_terms = 64`, tile 16, exact-coverage depth 10, fitter search
**scene-invariant**.
**Artefacts:** `docs/lights/p5/ladder-magnify.tsv`, `ladder-minify.tsv`, 48 PNG strips,
8 `frozen-*.fit` checkpoints, manifest with absolute paths and captions printed by
the run itself. Each table's own first line carries the `#invocation` that wrote it.

> **The 2026-09-09 tables were stale and are replaced.** `src/fit/analytic.rs:44`
> changed on 2026-09-14 and the analytic reference arm moved with it. Twenty rows
> across the two ladders differ from the 2026-09-09 artefacts on the 26
> deterministic columns — eight magnifying, twelve minifying — and every one of
> them is `analytic-sdf` on t3 or t6. Every atom arm, every wavelet arm, every
> criterion verdict and both single-primitive classes are bit-identical, which is
> the control. The corrections are marked inline below and none of them was made
> by editing a number: the whole run was re-executed.
>
> **The `render_ns` column is this session's clock and not 2026-09-09's**, which
> is the price of re-running over the published directory. The deterministic
> counter beside it, `fit_renders`, is identical on **all 280 rows** of both
> ladders — that is the control saying the work done is the same work. `render_ns`
> moved by a median of **0.90x** magnifying and **0.85x** minifying over 272
> paired rows, with a spread from **0.10x to 3.98x**. Read that as the machine and
> not the code: it is the same single-run scheduler noise the register's
> measurement rules already forbid publishing a single sample of, now measured
> across 272 pairs rather than asserted.
>
> **From 2026-09-28 (L-95) a refitted row's `render_ns` no longer includes its
> fit.** The fits now run in parallel before any row is assembled, so every arm's
> `render_ns` times the render alone and the fit's cost is `fit_renders`. A table
> carrying the header line `render_ns_convention	render-only-fit-excluded-see-fit_renders`
> is the new reading; every table above predates it, and its refitted `render_ns`
> includes a whole fit. See `crates/lights/src/fit/aspect3_rs_comment/fits_are_parallel_the_ladder_is_not.md`.

---

## The criterion convention, ruled before any number

**PRIMARY: SCENE UNITS. SECONDARY: DEVICE PIXELS. Both printed on every row.**

This experiment holds the DESCRIPTION fixed and varies only the sampling rate.
The scene does not change when the camera zooms, so a scene criterion asks one
question at every rung and a difference along the ladder is attributable to the
description. A pixel criterion re-asks a four-times-stricter question at 4x, and
a decay curve measured that way cannot be separated from the goalposts moving.

The pixel column is kept because the registered thresholds are DISPLAY criteria
and "is it sharp on the monitor" is a genuine pixel question. The analytic bar
proves the ruling is load-bearing rather than cosmetic: its 10–90 transition is
FLAT at 0.80–0.96 **device px** across a 32x span of scale and runs 0.12 → 3.38
in **scene units**. Same measurement, two conventions, opposite stories. A zoom
result that does not name its convention is uninterpretable.

---

## The answer

**Resolution independence is real for atoms on smooth and near-smooth content,
marginal on crisp edges, and it fails on text.** The gap between the frozen arm
and a fresh fit is the cost of freezing, and it is small or negative everywhere
except the glyph — but see C1 below, because the "ceiling" that gap is measured
against was breached in **13 of 24 magnification cells**.

Magnification, contract profile, each arm against its own base:

| class | frozen decay | frozen relL2 base → 8x | analytic relL2 8x | verdict |
|---|---|---|---|---|
| n1 colour field | **1.002** | 0.03786 → 0.03793 | *absent* | frozen atoms are essentially perfect |
| t3 corner | 1.122 | 0.17796 → 0.19959 | 0.00236 | mild decay, analytic 84x better |
| t2 straight edge | 1.359 | 0.05877 → 0.07985 | 0.00167 | real decay, analytic 48x better |
| t6 glyph, 12 px | **2.051** | 0.22339 → 0.45811 | 0.01352 | **frozen atoms fail** |

> **CORRECTED 2026-09-15 (BURNDOWN L-144). The `analytic relL2 8x` column above
> USED to read 0.02758 on the corner and 0.09216 on the glyph, and the verdicts
> USED to say "analytic 7x better" and "analytic 5x better".** Those figures were
> measured on 2026-09-09 by a `sdf_coverage` that painted from `Target::sdf`, a
> distance LOWER BOUND over the primitives; `src/fit/analytic.rs:44` changed to
> the real field `Target::combined` on 2026-09-14 and the table was not
> re-measured until today. The bound is wrong only where two primitives are both
> near, so the correction touches exactly the multi-primitive classes — t3 (2
> primitives) and t6 (6) — and leaves t2 and n1 bit-identical. **It moved every
> figure in the analytic arm's favour**, which is the uncomfortable direction:
> every "atoms come within X of analytic" sentence in this document was drawn
> against a handicapped opponent. Nothing else in the table moved: of 26
> deterministic columns across both ladders, only `relative_l2`,
> `max_abs_error` and the two decay columns changed, and only on the
> `analytic-sdf` rows of t3 and t6. Every atom arm, every wavelet arm and every
> criterion verdict — including the 10–90 and FWHM columns — is unchanged.
> `tests/p5_artefacts.rs` now recomputes twelve analytic cells from the code and
> fails if the committed table disagrees, so this cannot sit unnoticed again.

**The colour field is the strongest pro-atom result in the project and it must
not be oversold.** A frozen set decays 1.0000 → 1.0019 over an 8x zoom: the
continuous parametric field claim is literally true. But CDF 9/7 holds equally
flat (1.000) and beats the atoms absolutely (0.03026 against 0.03793), because
band-limited content has no detail above the base lattice for a lattice-bound
arm to lose. Resolution independence only DISCRIMINATES on content with
structure above the base lattice. The n1 result proves atoms do not degrade; it
does not prove they win.

**The glyph is the product question and frozen atoms lose it.** A frozen 12 px
fit carries no detail to fill 8x. Note the trap: the registered FWHM criterion
still reads *met* in scene units at every rung while whole-field relL2 doubles.
The probe is local and the error is not. Criterion and relL2 must be read
together or the glyph looks fine when it is not.

---

## The eighth artefact, found by this run: the decay column has a denominator

This is the finding that changes how every other number in the table is read,
and it was not in the design.

`decay` is an arm's relative L2 at a rung over its own at the base. Relative L2
divides by the norm of the truth AT THAT RUNG. So where an arm's error LIVES
decides how its decay reads, even when the error never changes. There are two
references:

- **full-field reference = 1.000** — an error spread over the whole image keeps
  its share of the image at every rung;
- **boundary reference = `sqrt(base_side / side)`** — an error confined to a
  boundary of fixed DEVICE-pixel width covers a shrinking share of a growing
  image: 0.354 at 8x magnification, 5.657 at 0.25x minification.

**The analytic SDF arm — the one arm that is resolution-independent BY
CONSTRUCTION — posts the most extreme decay in the table in both directions.**
On the two EDGE classes its decay sits on the boundary reference: over t2 and t3,
both ladders, the ratio `decay / bdry ref` stays inside **0.964–1.023**, median
**1.010 magnifying and 0.997 minifying**, and its `max|e|` barely moves
(0.0335–0.0383 on t2 and 0.0324–0.0383 on t3, across the whole ladder). For those
two classes the reading stands: their apparent improvement under magnification
and their apparent collapse under minification are the denominator, not the
description.

**On the glyph it is not the denominator, and the table used to say it was.**
Corrected (L-144), t6's analytic decay runs **0.156 magnifying** against a
boundary reference of 0.354 — 2.3x better than the metric alone predicts — and
**67.07 at 0.25x minifying** against 5.66, which is 11.9x worse. It departs at
every non-base rung of both ladders: ratio 0.89 / 0.59 / 0.44 magnifying at
2x / 4x / 8x, and 1.34 / 2.03 / 2.27 / 3.54 / 11.86 minifying from 4x down to
0.25x. A glyph described by signed distance genuinely gets better with pixels and
genuinely dies without them, and that is a statement about the description.

Every other arm tracks the full-field reference instead: frozen atoms 1.145 and
0.974, CDF 9/7 1.064 and 0.938 (medians against 1.000).

> **CORRECTED 2026-09-15 (BURNDOWN L-144). The two paragraphs above USED to read:
> "It reads 0.353 magnifying and 5.787 minifying. Measured over three classes and
> both ladders, its decay sits on the boundary reference with a median ratio of
> 1.002 magnifying and 1.000 minifying … All of its apparent improvement under
> magnification and all of its apparent collapse under minification are the
> denominator. None of it is the description."** That sentence was true of the
> arm as it was then MEASURED and false of the arm. The coverage function painted
> from a distance lower bound, whose error is a fixed-width artefact along the
> seam where two primitives meet — which is to say the bug was *itself*
> boundary-localised, and so it manufactured, on the multi-primitive classes,
> exactly the behaviour the finding cited them as evidence for. With the bug gone
> the evidence is t2 and t3 only, and it is still good evidence: t2 is a
> single-primitive class the bug could not touch, and it tracks the boundary
> reference at 0.964–1.022 over eleven rungs. **The finding survives; one of its
> three witnesses turned out to be the defect testifying about itself.** Note
> also that the corrected t3 tracks the reference slightly LESS tightly than the
> broken one did (1.000–1.020 against 0.998–1.002) — a fit that looks too good is
> worth suspecting.

So the spec's own reconnaissance reading — *"the analytic bar IMPROVES 1.00 →
0.36"* on the glyph — **was refuted on the broken column and is reinstated on the
corrected one.** The glyph's analytic decay is 0.156, not 0.36; `1/sqrt(8)` is
0.354, so 0.156 is 2.3x past what the denominator alone gives and the improvement
is real. The other half of the claim survives and grows: analytic ending better
than frozen atoms in absolute error at the same rung is now **34x** (0.01352
against 0.45811), where the broken column made it 5x.

> **WITHDRAWN 2026-09-15 (L-144):** *"So the spec's own reconnaissance reading —
> 'the analytic bar IMPROVES 1.00 → 0.36' on the glyph — is the metric, not a
> representational gain: 0.36 is 1/sqrt(8) = 0.354."* The number 0.36 was the
> broken arm's; the arm's own decay is 0.156. The refutation was correct
> arithmetic applied to a stale figure, which is the failure mode worth naming:
> a check that is sound in form can still be pointed at the wrong number.

The run now prints a `bdry ref` column immediately beside `decay`, and a third
self-audit, `decay_that_is_only_the_metric`, names the rows where the decay is
the denominator. It deliberately does NOT correct the decay: a single corrected
column would be a third wrong number, making a perfectly flat colour-field
result look like it decayed 2.8x.

**Where the boundary reference stops predicting is the real signal.** On the
glyph the analytic arm departs from it at every non-base rung of both ladders —
ratio 3.54 at 16 px and 11.86 at 8 px minifying, and 0.44 at 8x magnifying — and
`max|e|` moves with it (0.1756 at 8x up to 0.2917 at 4x and down to 0.1295 at
0.5x as the glyph vanishes). That is a genuine representational signal, visible
only because the geometric part was accounted for first.

> **CORRECTED 2026-09-15 (L-144). This paragraph USED to read "ratio 1.26 at
> 16 px and 1.74 at 8 px … and `max|e|` finally falls as the glyph vanishes",
> and to place the departure on the minifying tail alone.** Both figures came
> from the broken column. The corrected arm departs by 3.54 and 11.86 at those
> two rungs, departs under MAGNIFICATION as well, and its `max|e|` does not sit
> pinned near 0.49 and then fall — it moves across the whole ladder, because it
> is no longer dominated by a constant seam artefact of size 0.494.

---

## C1 — the refitted ceiling is not a ceiling

**The frozen set beat a fresh fit in 13 of 24 magnification cells.** On the
corner it beat it at EVERY non-base rung under BOTH profiles (6 of 6):

| cell | frozen | refitted | frozen better by |
|---|---|---|---|
| t3 corner, contract, 8x | 0.19959 | 0.27913 | 0.07954 |
| t3 corner, prefiltered, 4x | 0.19613 | 0.29238 | 0.09626 |
| t2 edge, contract, 8x | 0.07985 | 0.09603 | 0.01617 |

Greedy matching pursuit is a FLOOR, not an optimum. Where the frozen set wins,
the frozen-to-refitted gap is a MINIMUM statement about the cost of resolution
independence and possibly zero. It must not be read as a measurement of the
representation.

---

## C3 — the missing prefilter, sized rather than asserted

`SharedSettings::contract()` ships `screen_dilation = 0`, meaning no pixel
prefilter at all. Under magnification the prefilter is a net cost at the base
rung and a small win away from it (t2 frozen: −0.00101 at 1x, then +0.00191,
+0.00214, +0.00206). Under minification it is the whole story: the frozen arm's
error under `contract()` turns UP at the small rungs while under `prefiltered()`
it stays flat or improves.

**The apparent minification failure of atoms is the missing prefilter, not the
representation.** Minification reported under `contract()` alone would be
aliasing misreported as a representational loss — exactly the confound C3 warns
of, now measured.

### AMENDED 2026-09-09, AFTER THE VERDICT COLUMNS WERE FIXED

The sentence above is true of ERROR and false of SHARPNESS, and the run had no
way to say so because the criterion verdict for this class was not being emitted
correctly (see the correction note at the end of this document). Re-measured on
the regenerated `ladder-minify.tsv`, frozen atoms on t2:

| profile | 8x | 4x | 2x | 1x | 0.5x | 0.25x |
|---|---|---|---|---|---|---|
| `contract()` rel L2 | 0.0960 | 0.0934 | 0.0903 | 0.0899 | 0.0963 | 0.1132 |
| `contract()` 10–90 | 0.391 | 0.391 | 0.391 | 0.383 | 0.423 | 0.573 |
| `contract()` meets | yes | yes | yes | yes | yes | yes |
| `prefiltered()` rel L2 | 0.0676 | 0.0639 | 0.0594 | 0.0571 | 0.0555 | 0.0564 |
| `prefiltered()` 10–90 | 0.629 | 0.642 | 0.711 | **1.419** | **1.816** | **2.944** |
| `prefiltered()` meets | yes | yes | yes | **no** | **no** | **no** |

**The prefilter buys error and sells sharpness.** It lowers rel L2 at every rung
and holds it FLAT into the small scales — which is the C3 result and it stands —
while widening the 10–90 transition from 0.38–0.57 scene units to 0.63–2.94, so
the edge stops meeting a sub-unit sharpness criterion from 1x down. An arm can be
closer to the truth in L2 and further from the product requirement at the same
time, and this run reports both because reporting either alone is a different
claim than it appears to be.

This is the sharpest available evidence for the parked item *decompose the
prefilter*: `screen_dilation = 0.3` is applied here as one constant, where
Mip-Splatting (arXiv:2311.16493) states it is a SUM of 0.1 screen-space and 0.2
object-space. A 2D UI wants only the screen half, and it is plausibly the object
half that is doing this blurring. Nothing here measures that split — it is the
next experiment, not a conclusion of this one.

> **CORRECTED 2026-09-15 (BURNDOWN L-37). The paragraph above is wrong in its
> premise and wrong in its guess, and it is kept so the withdrawal is visible.**
> This run did not apply 0.3. `ProfileChoice::Prefiltered` resolves to
> `SharedSettings::prefiltered()`, whose `screen_dilation` is
> `PIXEL_BOX_VARIANCE = 1/12` (crates/lights/src/settings.rs:72,137); 0.3 is
> `REFERENCE_DILATION`, reachable only through `reference()`, which no profile
> selector offers (crates/lights/src/fit/harness.rs:47-50). Every `prefiltered`
> row in the tables above was rendered at 1/12.
> Nor can the blur be the object half: the crate has no object-space term at
> all. `screen_dilation` has exactly one consumer,
> `K::prefilter(params, cov, pixel_variance)` at
> crates/lights/src/method/splat/project.rs:140, and a grep for any
> object-space or volume prefilter over `src/` returns nothing. The widening
> from 0.63 to 2.94 was produced by the screen-space pixel box alone. The
> measured C3 result — the prefilter lowers relative L2 at every rung and widens
> the transition — is unaffected; what changes is which knob to reach for next,
> and it is not a decomposition of a constant this crate never used.

---

## The search convention is a larger effect than the thing under test

Refitting at the far rung under both search conventions:

| class, direction, far rung | scene-invariant | pixel search | difference |
|---|---|---|---|
| t2 magnify at 8x | 0.09603 | 0.18628 | **0.09025** |
| t2 minify at 0.25x | 0.02932 | 0.02433 | 0.00499 |

At 8x the search convention moves the refitted ceiling by **more than the entire
frozen-to-refitted gap being measured**. The fitter's centre step, seed sigma
and both log-scale clamps are pixel constants; this run holds them
scene-invariant. No setting can equalise the seed LATTICE — a refit at 8x
chooses among 64x more distinct seed positions — so the ceiling is genuinely
stronger at large scales for a reason about the SEARCH, not about resolution
independence.

---

## The wavelet arms, and the trap in reading them

The wavelet arms post decays near 1.00 in both directions — among the flattest
in the table — while carrying relL2 0.19 against the atoms' 0.06 on t2 and
failing the criterion at every single rung. **A flat decay from a bad base is
not resolution independence.** Read the base row's absolute relL2 before the
decay column means anything.

Their loss is STRUCTURAL, not a defect: a coefficient set lives on a fixed
dyadic lattice and does not resample for free. Each arm is handed the
textbook-correct resampler in its direction (bilinear magnifying, pixel-box-area
minifying) so the loss is a floor rather than a strawman.

---

## What this run does NOT settle

- **C2, the byte comparison, remains open.** Every arm is billed in one
  symmetric f32 currency so the columns are at least in the same money, and the
  header is stamped `symmetric-f32-provisional-see-C2`. **No byte ranking is
  claimed from this run.**
- **Absolute error is not comparable to aspect 2.** The truth here is the
  pixel-box INTEGRAL of one fixed continuous field; aspect 2 used the POINT
  SAMPLE. Only ratios along a ladder carry across.
- **The base rung privileges three of four arms.** At base the frozen arm IS the
  refitted arm and the wavelet lattice IS the truth lattice. Mitigated by
  running both ladders from both ends and scoring each arm against its own base.

---

## Correction, 2026-09-09: the verdict columns did not say which probe they meant

The artefacts this document cites were regenerated after a defect was found in
the TSV emitter, and any copy of `ladder-*.tsv` older than this note is wrong in
one column.

`row_cells` emitted a single `meets_scene` / `meets_device_px` pair carrying the
**transition** verdict, gated only on whether quality was measured. A class judged
on FWHM rather than on a 10–90 transition carries `transition_10_90 = nan`, and
`nan < criterion.max_transition` is **false** in Rust — so every glyph row read
`no` where the honest answer is *this target carries no transition probe*. The
FWHM verdict, which `judge_fwhm` computes at `src/fit/scale.rs:459-461`, was never
written to the file at all.

The printed table was right the whole time: `met()` at `src/bin/lights_p5.rs:428`
returns `-` when the value is not finite and prints all four verdicts. So the
machine-readable artefact and the human-readable run disagreed, and the one a
report generator consumes was the wrong one.

Read straight, the table said the analytic SDF never meets the glyph at any rung —
contradicting P2 on the same page. Computed from the right probe it meets at every
rung: FWHM 0.937, 0.947, 0.933, 0.950 against a target of 1 with a tolerance of 0.5.

**Fixed.** `SCALE_COLUMNS` is 29: `meets_transition_scene` /
`meets_transition_device_px` name their probe, `meets_fwhm_scene` /
`meets_fwhm_device_px` were added, and `probed()` prints `-` where a probe is
absent. Regression case
`the_tsv_names_which_probe_each_verdict_belongs_to_and_says_nothing_where_a_probe_is_absent`
drives a glyph ladder and an edge ladder, asserts every finite probe is judged and
every absent one dashed, and refuses to pass unless it saw both kinds. It was RED
on the unfixed tree; a mutation making the verdict ignore probe existence makes it
red on the fixed tree, recorded in `scripts/mutations.json`. `tests/aspect3.rs`:
31 passed, 0 failed.

**And the corrected column carries a warning of its own.** With the right probe,
Haar *meets* the glyph criterion while sitting at rel L2 0.676, beside the analytic
arm at 0.0135 (0.092 before the L-144 correction above, which makes the gap wider,
not narrower). FWHM within ±0.5 of a one-unit stem is a loose gate — a stroke can be
the right width and the wrong picture. Every glyph conclusion in this document is
drawn from rel L2 ratios and never from that column.

Register entry: **L-119**, in `crates/lights/BURNDOWN.toml`.
Interactive tables and strips: <https://claude.ai/code/artifact/3577a16a-ad94-4f7f-bea6-3727dbe65bed>
