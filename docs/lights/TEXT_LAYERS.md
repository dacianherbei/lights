# Text in lights: which font layers it owns, and the three tiers of a glyph

BURNDOWN L-160, cluster `text-as-atoms`, written 2026-09-28. This is a **proposed
ruling**: it takes the entry's defensible answer, checks each part against the tree
as it stands today, and says where the tree overruled the entry. The PM can veto any
section; until then L-161 (the first converter) builds against it.

`tests/text_layers.rs` pins the three lines below that a future change can silently
break. Change any one of them only by amending this document in the same commit.

    lights links: png, rayon, thiserror
    tier 2 precedent: src/scene/io.rs
    glyph partition: one Region, holes empty

## What was measured before ruling

Every claim here was re-read from the tree on 2026-09-28, not taken from the entry.

| claim | where it stands | how to re-check |
|---|---|---|
| lights links three libraries | `rayon`, `png`, `thiserror`; two dev-dependencies, `tempfile` and (since L-171, 2026-09-29) `ttf-parser` — it was one, `tempfile`, when this was written | `sed -n '/\[dependencies\]/,/^\[/p' crates/lights/Cargo.toml` |
| `fit` is not a default feature | `default = ["method-splat", "method-raycast"]`; `fit = ["method-splat"]` | same file, `[features]` |
| nothing outside `src/fit/` may name `crate::fit` | `tests/layering.rs::the_fitter_never_reaches_back_into_the_core` | read the test |
| the parts/holes partition is core, not fit | `PrimitiveSet { ink, parts, holes }` in `src/primitive.rs`, evaluated as union(parts) minus union(holes) by `combined` | `grep -n "pub struct PrimitiveSet" -A5 crates/lights/src/primitive.rs` |
| the core has **no filled region bounded by curves** | `Primitive` is `HalfPlane`, `Segment`, `Arc`, `Disc`, and `Segment` and `Arc` carry a `half_width`: they are STROKES. `DrawShape::Polygon` in `src/draw.rs` is filled but straight-edged, one contour, EVEN-ODD | `grep -n "Polygon\|half_width" crates/lights/src/primitive.rs crates/lights/src/draw.rs` |
| the core already has a versioned, refusing file format | `SCENE_MAGIC = "lights-scene"`, `SCENE_VERSION = 1`, refuses a different version, in `src/scene/io.rs` | `grep -n "MAGIC\|VERSION\|different format" crates/lights/src/scene/io.rs` |
| `atoms_io`'s format is fit-only | `src/fit/atoms_io.rs` stores `Atom2d` and is compiled only under `fit` | `sed -n 1,9p crates/lights/src/fit/atoms_io.rs` |
| text already reaches lights as coverage | `RasterLayer::from_coverage8` in `src/image.rs` (L-75) takes a glyph bitmap rasterised elsewhere | `grep -n "fn from_coverage8" crates/lights/src/image.rs` |
| layer 1 is already taken from outside | optical calls `font_kit::source::SystemSource` in `src/unified/picker/font_picker.rs` | `grep -rn font_kit crates/optical/src` |
| a pure-Rust parser is already in the workspace | `ttf-parser 0.25.1` is in `Cargo.lock`, pulled in by eframe's `ab_glyph`; `rustybuzz` and `fontdb` are not | `grep -n '^name = "ttf-parser"\|^name = "rustybuzz"\|^name = "fontdb"' Cargo.lock` |

Two of these overrule the entry as written:

1. **The entry's `next` names `atoms_io` as the precedent for Tier 2, and its own `why`
   forbids that.** `atoms_io` lives under `src/fit/`, so a derived-atom format copied
   from it, or written through it, would make text opt-in, which is the exact failure
   the `why` names. The precedent is `scene::io`, which carries the same MAGIC, VERSION
   and refuse-on-mismatch and is core.
2. **"Tier 0 in core terms" is not free.** The core vocabulary is strokes and one
   straight-edged even-odd polygon. A TrueType or CFF glyph is several closed contours
   of lines and curves under the NON-ZERO rule. Tier 0 therefore needs one new core
   shape, and naming it is part of this ruling (below), not a detail left to L-161.

## (a) The layers lights owns

Font rendering is four layers: (1) discovery and fallback, (2) parsing, (3) shaping,
(4) rasterisation. **lights owns layer 4 only.** Layers 1 to 3 live in the consumer,
which today is optical, and lights never links a font library. That is what the
`lights links:` line above pins.

The two candidate stacks for layers 1 to 3, weighed:

| | pure Rust: `fontdb` or `font-kit`, `ttf-parser`, `rustybuzz` | C: fontconfig, FreeType, HarfBuzz |
|---|---|---|
| build | cargo only | three C libraries and a system package per platform |
| layer 1 on macOS | `font-kit` already wraps Core Text and is already in optical | fontconfig is not the platform's own discovery mechanism on macOS |
| layer 2 | `ttf-parser` is already compiled in this workspace | FreeType |
| layer 3 coverage | `rustybuzz` is a port of HarfBuzz that tracks its shaping tests; it lags HarfBuzz by releases, not by design | HarfBuzz, the reference |
| what it brings that lights does not want | nothing | FreeType's hinting and rasteriser are layer 4, the layer lights replaces |

**Ruled: the pure-Rust stack, in optical.** `font-kit` stays for discovery because it
is already there; `ttf-parser` for parsing; `rustybuzz` for shaping when optical first
needs more than one glyph per codepoint. The only thing the C stack buys that the
Rust one does not is shaping completeness at the edges of complex scripts, and that
is a reason to revisit layer 3 alone, later, without touching lights.

Consequences for the sibling entries, which this ruling rewrites:

- **L-161 does not write an SFNT reader.** Its step (1), table directory, cmap, loca,
  glyf, composites and the ttc header, is exactly what `ttf-parser` already does, and
  re-implementing layer 2 inside lights contradicts owning layer 4 only. The converter
  splits in two: the curve stream comes from `ttf-parser`'s `OutlineBuilder`
  (`move_to`, `line_to`, `quad_to`, `curve_to`, `close`) in the consumer or a test, and
  lights owns what is actually hard, the quadratic-to-arc conversion with an em-space
  tolerance and the contour-direction-to-partition step, as a core function over that
  stream. `ttf-parser` may be a lights DEV-dependency, so L-161's and L-168's tests
  can read a vendored face, without appearing in the `lights links:` line.
- **The interim path stays.** Until Tier 0 is built, optical draws text by
  rasterising elsewhere and handing lights coverage through `from_coverage8`. Nothing
  here retires it; it is also the appearance oracle L-168 will compare against.

## (b) The three tiers, by what makes each one change

### Tier 0 — design geometry. Changes when the type designer changes the design.

Core types only, compiled in the default build, so the layering guard stays green and
text is never opt-in.

- **Outline.** One new core shape: a closed CONTOUR whose edges are line segments and
  circular arcs (`Line { a, b }`, `Arc { centre, radius, from, to }`), with no
  `half_width`. It is the filled counterpart of the stroked `Primitive::Arc`.
- **Partition — AMENDED 2026-09-29 by a PM ruling (L-172): a glyph is ONE `Region`.**
  Not parts and holes. A `Region` is a single solid whose membership *is* the non-zero
  winding of all the glyph's contours, and whose distance is taken to the region's true
  boundary, so the invariant below holds **by construction** — for any overlap, partial
  cancellation, self-intersection or nesting depth — with `holes` empty and no overlap
  resolution at all. The converter emits one `Region` per glyph and owns no partition.

  **WITHDRAWN, and kept because the withdrawal says more than the correction does.** This
  bullet used to read: *"A glyph is parts and holes, reusing the semantics `PrimitiveSet`
  already evaluates: union of parts minus union of holes. The converter owns the invariant
  that makes this honest: the parts-minus-holes region equals the source's non-zero
  region, within the stated tolerance. That is not automatic … so the converter must
  resolve overlaps before it emits."* It was written on 2026-09-28, before `Region`
  existed, and it is withdrawn for two measured reasons.

  First, a `Region` meets that invariant exactly. `tests/outline.rs` measures it against a
  sampled reference on five inputs — a ring, two overlapping +1 contours, +1 +1 −1, a
  self-intersecting pentagram of winding 2, and a bullseye with a dot in its counter — at
  a worst disagreement of 1.9e-4 em, which is 0.95 of the reference's own sample spacing.

  Second, the obvious reading of the old rule is provably wrong, and the same test pins
  that too: taking every clockwise contour as a hole **carves the centre** of `+1 +1 −1`
  and of a bullseye whose dot sits inside a counter — the shapes of `◎`, `®` and `回`. A
  correct one-level partition of those exists only by cutting annuli into simple pieces, a
  boolean-and-decompose kernel on lines and circles whose sole purpose would be to
  reproduce what `Region` already computes.

  What the partition alone could still offer is a per-contour `FeatureId`, and no consumer
  reads one. If one ever does, emit a partition *beside* the `Region` rather than instead
  of it.
- **No ink.** `PrimitiveSet` carries `ink`; a glyph does not. Colour belongs to the run
  that draws it, and a glyph becomes a `PrimitiveSet` only at draw time.
- **Units.** Em space, 1.0 = one em, y up from the baseline. Font units stop at the
  converter. The arc tolerance is stated in em, as L-161 requires.
- **Metrics.** Per glyph: advance and bounds. Per face: ascender, descender, line gap,
  and the axis coordinates the outlines were taken at.
- **Design axes, including `opsz`.** Stored as INSTANCES in the first version: each
  outline set is tagged with the full axis tuple it was converted at, and a different
  tuple is a different Tier 0 set. Continuous interpolation is deferred, and the reason
  is specific to arcs: two masters converted independently produce different arc
  counts, because biarc split points depend on the curve, so their arcs cannot be
  interpolated edge for edge. Interpolation opens only once one glyph has been
  converted COMPATIBLY across two masters (shared split parameters) and the extra arc
  count measured.

### Tier 1 — rendering intent. Changes when the designer states intent about rendering.

Declarative, from a bounded vocabulary, and optional: a Tier 0 set without Tier 1 is
complete.

- **Alignment zones**: baseline, x-height, cap height, ascender, descender, each a
  position and an overshoot, in em.
- **Standard stem widths**, horizontal and vertical, in em.
- The vocabulary is the one CFF already uses (`BlueValues`, `StdHW`, `StdVW`), which is
  the precedent that it can be stated without code.

**Never instructions.** No bytecode, no conditionals, nothing a reader must execute.
TrueType's `fpgm`, `prep` and glyph programs are NOT converted. The last format that
tried total adaptation was TrueType hinting, a Turing-complete VM in every file, and
the industry spent twenty-five years retreating from it toward declarative hints and
then continuous axes.

Nothing in lights consumes Tier 1 yet. L-159 measured the small-size failure as a
FILTER error (a gaussian of the box variance is not a partition of unity), which no
alignment zone repairs, so whether Tier 1 buys anything is unmeasured. It is specified
so a converter does not discard it; it is not built until an evaluator reads it.

### Tier 2 — derived atoms. Changes whenever the converter, the kernel or the device class does.

- **Volatile and regenerable**, never authoritative: deleting every Tier 2 file loses
  time, never information.
- **Core format**, copying `scene::io`'s pattern (the `tier 2 precedent:` line above):
  its own magic, a version, and a refusal on mismatch. NOT `atoms_io`, for the reason in
  the measured section.
- **Keyed by** converter version, a hash of the Tier 0 set, kernel family, and device
  class. **Not by scale alone**: L-159 measured that inflating covariances carries one
  fine fit down to where the thinnest feature is about 0.7 px, so scale enters the key
  only as the band of any low-scale correction below that, and that band is a property
  of the glyph's thinnest feature, not of its em size.
- **Produced only under `fit`.** The fitter is a non-default feature, so a default build
  can LOAD Tier 2 and cannot regenerate it. A cache miss in the default build therefore
  falls back to evaluating Tier 0 directly, which the core already knows how to do for
  a parts-and-holes set. That is what keeps text working in the default build, and it
  is a ruling, not an accident: if the fallback is ever removed, text becomes opt-in.

## (c) What may never be stored in Tier 0 or Tier 1

Anything that depends on the device:

- a pixel size or ppem, and anything computed for one: hinted outlines, `DELTAP`
  deltas, `gasp` ranges, grid-fitted positions;
- embedded bitmap strikes (`EBDT`, `CBDT`, `sbix`);
- subpixel layout, transfer function or gamma, stem darkening, and the pixel filter.

Tier 2 may be keyed by device class because it is a cache. The payoff of the scheme is
that a new display is one new evaluator and never a reconversion, and that holds only
if nothing device-specific reaches Tier 0 or Tier 1.

## What this does not settle

- Colour glyphs (`COLR`, `SVG`). **Checked on paper by L-171 (2026-09-29), not built.**
  `COLR` v0 is exactly the obvious shape: a base glyph lists (layer glyph, `CPAL` index)
  pairs, each layer is an ordinary `glyf` outline, so each layer is one Tier 0 `Filled`
  that becomes one `PrimitiveSet` with a solid ink at draw time, laid over the last with
  `FloatImage::composite_over` (L-38). Nothing new is needed. `COLR` v1 breaks it in two
  places, and both are findings about Tier 0 rather than about colour:
  1. **Tier 0 is not closed under an affine map.** `PaintTransform`, `PaintSkew` and
     non-uniform `PaintScale` map a layer's outline by a general affine. A quadratic
     stays a quadratic under that map; a circular ARC becomes an elliptical one, which
     is not in the line+arc vocabulary. `Edge::mapped` takes one scale and an offset —
     similarity only — and that is not an oversight to patch. So a transformed paint
     must be converted AFTER its transform is applied to the source curves: Tier 0 for
     a v1 glyph is keyed by the transform, not only by the axis tuple. The same holds
     for a synthetic oblique of any face.
  2. **An ink is one colour.** `PaintLinearGradient`, `PaintRadialGradient` and
     `PaintSweepGradient` fill a `PaintGlyph` with a colour that varies per pixel;
     `PrimitiveSet { ink: Rgb }` has nowhere to hold it. A gradient layer needs either
     a new ink kind or the raster path (L-73), and which one is the design question a
     v1 consumer has to open. `PaintComposite`'s modes beyond source-over are a third,
     smaller gap.
  `SVG` glyphs are not outlines at all and stay on the interim coverage path.
- CFF (`curve_to`). Built by L-171 as `ContourBuilder::cubic_to`, the same bisection
  with a cubic sag bound (3/4 of the larger second difference), measured two-sided at
  0.9989x the tolerance or less over 48 seeded cubics. **No real CFF glyph has been
  converted**: no licence-clean CFF face was available offline, and the round trip in
  `tests/font_roundtrip.rs` is TrueType only, where a cubic is refused as a bug.
- Overlapping contours from a real font. Both faces examined for L-171 (Hack and
  Ubuntu Light) ship overlap-removed: none of the 8 Hack glyphs or the 9 Ubuntu Light
  glyphs converted has winding >= 2 anywhere on a 90x90 grid over its bounds.
  The overlap cases are covered only by synthetic inputs in `tests/outline.rs`; a
  variable font's unmerged masters are the likely real source.
- Whether the direct Tier 0 evaluator is fast enough to be the default-build text path
  at UI sizes. It is correct by construction and unmeasured for speed.
- Whether `PrimitiveSet::combined`'s min/max distance is a close enough coverage for
  text: it is a distance BOUND, not an exact distance, the same class of error corrected
  in `docs/lights/P2_RESULTS.md`. L-159's exact box-coverage oracle is the reference to
  measure it against.
