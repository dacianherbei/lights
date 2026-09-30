# lights

A CPU-only anisotropic-atom renderer for macOS on Apple silicon, and an experimental
proving ground for a way of building software: **interchangeable strategies, measured, with
a correctness contract between independent implementations.**

It renders 3D scenes made of anisotropic gaussian atoms and surfels — no triangles, no
GPU rasteriser — along three interchangeable axes: **kernel** (atom shape), **method**
(image formation), **precision**. `crates/lights-mac` is the macOS host: an AppKit window,
its pointer and key events, and a `CAMetalLayer` that presents a frame lights rendered.

```
cargo run -p lights-mac --release --bin lights-gallery
```

A 960×600 window, three scenes, one module each. Drag to orbit. `R` toggles the splat
rasteriser against the ray caster. Keys 1–3 switch scenes.

## Read the research first

This crate's output is not only code. **41 experiments, 38 of them with a command you can
paste**, every figure naming where it came from, and 31 cross-experiment findings the
records make about *each other* — contradictions, tables not in one currency, confounds
shared across runs, and three findings of kind `unreproducible` where the evidence is gone
and the record says so.

| | |
|---|---|
| `docs/lights/RESEARCH.json` | the single source of truth |
| `docs/lights/RESEARCH.md` | the reading form |
| `docs/lights/RESEARCH_DOCUMENT.html` | a printable linear document — open it and Save as PDF |
| `docs/lights/pages/research.html` | an interactive split-pane form |
| `crates/lights/tail.toml` + `head_open.toml` | the register: 202 entries, every feature and its measured state |

All four forms are generated from the JSON, and the document imports the record's own
`copy_pasteable` predicate rather than reimplementing it, so they cannot disagree about how
many experiments reproduce.

## What this crate does not do, and how we know

**It is not a UI renderer, and it was one until it was measured.** The display list drew
crisp interface chrome and lost:

- egui's whole CPU frame over eleven real interface tabs is **0.165–0.979 ms**, so the bar
  was about one millisecond and not a 16.67 ms frame budget
- replaying the identical op sets, this crate's `DrawList` is **2.4–8.3× that on 18 threads
  against egui's one**, and **97.8–99.2% of the frame is pixel traffic** — memory
  bandwidth, which no choice of kernel, method or precision touches
- tiny-skia draws those same op sets **2.22–10.16× faster from a sixth of the memory**,
  with no pack at all

So the UI target was withdrawn. What survives that comparison is narrow and real: lights
passes the registered quality gate at 1× where tiny-skia fails the glyph-like target,
drawing a 1 px stroke as a hairline at peak 0.6157 against a 0.90 bar where lights reads
0.9369, and its parallel bands are bit-exact where tiny-skia's depart by up to 191 levels.
It was not enough to keep the target. The full argument is the report's section *"The
comparison, stated plainly"*.

**The design's own founding rationale did not survive either.** Static dispatch over
precision, kernel and probe exists to preserve pixel-parallel vector width; that width is
**0.1–0.4% of the render loops' work**, across 96 render-loop bodies. Measured, written
down, left in.

## What the measuring did earn

The strongest evidence here is **agreement, not speed**. Two independently written
renderers are certified against each other pixel by pixel inside a derived bound, and the
ray caster's production binning must reproduce its brute-force oracle bit for bit — which
is how its tile pairs fell from 109,188 to 46,717 without moving an image. The same shape
recurs wherever the crate holds two implementations of one thing.

Every arm carries one verdict, with the target its deciding evidence was taken against and
whether that evidence transfers: **16 KEEP, 6 PRUNE, 9 RE-JUDGE**, and no arm is pruned on
evidence that does not transfer to the surviving target. That table is `arms` in the
research record.

## Building and testing

```
bash crates/lights/scripts/run_tests.sh
```

That is the one documented test command. `cargo test` alone silently skips the entire `fit`
arm, because `fit` is not a default feature. The script runs `--all-features`, sums the
per-target result lines itself, and then builds **and runs** each method feature alone, so
a feature that cannot stand on its own fails loudly.

Requires macOS on Apple silicon and a recent Rust toolchain. The library itself links three
crates — `rayon`, `png`, `thiserror` — and a test refuses a fourth without an amendment to
`docs/lights/TEXT_LAYERS.md`. `crates/lights-mac` is the only place platform code lives,
and a test refuses an Apple framework or an `aarch64` intrinsic anywhere else.

## A few conventions worth knowing before reading the code

- **No comments in the code.** `tests/layering.rs::no_comments_in_src` allows at most one line
  per file and a guard enforces it. In the monorepo this crate came from, the reasoning lives in
  sibling `_comment/` folders; those are not carried here, so the code is bare and the arguments
  are in `docs/lights/` and in the research record instead.
- **Every fix carries a test that was seen RED before the fix**, plus a row in
  `crates/lights/scripts/mutations.json` proving that test can fail — applied, compiled,
  ran, detected. 667 rows, all CAUGHT.
- **Withdrawals are never silent.** When a ruling changes, the old sentence stays quoted
  with the date it stopped being true, because the withdrawal usually says more than the
  correction: it says which way the tree moved.
- `docs/lights/parked/` holds regression tests written test-first and seen red, waiting for
  the change they test. The test is the expensive half.

## Provenance

Extracted from the fulgor monorepo as a single commit with no history. Two files are kept
verbatim rather than rewritten, because this crate's own guards quote them: `CLAUDE.md`
(which describes the wider project, not just this crate) and
`docs/plan/TWO_HORIZON_PLAN.md`. `docs/gaussian_splatting/` cites thirteen third-party
papers and carries none of them. One further omission is stated in the record itself: the harness
behind the `session-isolation-cost` experiment is local automation of that monorepo and is not
carried here, so that experiment's command cannot be pasted from this repository — which is why
the reproducible count is 38 of 41 rather than 39.

Licensed under either of **Apache-2.0** or **MIT** at your option. The vendored fonts under
`crates/lights/tests/fonts/` keep their own licences beside them: `OFL-Inter.txt`,
`OFL-AbrilFatface.txt`, `OFL-NotoSansJP.txt` and `UFL.txt` for Ubuntu Light.
`LightsStemCalibration.ttf` was built for this crate.
