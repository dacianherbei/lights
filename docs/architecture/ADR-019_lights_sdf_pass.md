# ADR-019: The analytic SDF champion is a composited pass, not a RenderMethod

**Date**: 2026-09-28
**Status**: Accepted (burndown loop, under the PM's L-73 ruling (iii); the PM may veto)
**Deciders**: burndown loop context for L-118; rulings by the PM on L-73 and L-75
**Related**: BURNDOWN L-118 (this decision), L-73 (one primitive kind per scene), L-75 (text arrives as coverage), L-144 (the SDF lower-bound correction), L-156 (depthless composite on a resting object); `docs/lights/P2_RESULTS.md`; ADR-018

---

## Context

P2 ruled the analytic SDF arm the winner on every crisp UI content class. The glyph row
(`docs/lights/P2_RESULTS.md:62`) is 6 primitives, 274 B, criterion met, against 12 to 22
gaussian atoms that all fail. `optical` is the crate that has to consume that win.

Measured on 2026-09-28:

- The whole SDF path lives in `crates/lights/src/fit/` (7,973 lines), behind the non-default
  `fit` feature. `grep -rnE 'analytic|Analytic|sdf|Sdf|SDF' src/method/ src/dispatch.rs
  src/scene/ src/kernel/ src/settings.rs src/blend.rs src/agreement.rs` exits 1.
- The core never reaches into `fit` (`grep -rn 'crate::fit' src | grep -v '^src/fit/'` exits 1),
  enforced by `tests/layering.rs::the_fitter_never_reaches_back_into_the_core`.
- `MethodId::ALL` has five members and none is analytic. `RenderMethod` consumes a
  `PreparedScene<T, K>` of atoms under a kernel `K` and a 3D camera.
- No crate in the workspace depends on `lights`: `grep -rn lights --include=Cargo.toml .`
  outside `crates/lights/` finds the workspace member line and an unrelated aegis comment.
- What the SDF pass actually needs is small and already core-clean: `src/fit/target.rs`
  (547 lines) imports only `crate::numerics`; `Similarity2` already lives in
  `crate::numerics::similarity`; `sdf_coverage_through` in `src/fit/analytic.rs` is about
  twenty lines over `Target::combined`.

Two rulings shape the answer. **L-73 (iii)**: one scene holds one primitive kind, and kinds
mix by compositing two renders. **L-75 (B)**: text arrives as pre-rasterised coverage through
`RasterLayer<T>` (`src/image.rs`), so text does not need an outline or analytic path and is
not an argument either way.

---

## Decision

**We will** lift the primitive set and its SDF coverage out of `fit` into a core module of
the default build, producing a `RasterLayer<T>` that is laid over any render with
`RasterLayer::over` / `FloatImage::composite_over`. It will **not** be a `MethodId` or a
`RenderMethod`.

**This means**:
- A core module (working name `lights::primitive`) owns `Primitive<T>`, `Part<T>` and a
  primitive set of parts, holes and ink. The fit-only fields of `Target` (`name`, `paper`,
  `probes`, `FeatureId`, `EdgeProbe`) stay in `fit`, and `fit::target::Target` wraps the core
  set. The dependency direction stays fit → core, so the layering guard is untouched.
- The pass renders through `Similarity2` and a viewport, not a `CameraSpec`: it is a 2D pass.
- Coverage uses `Target::combined` (the L-144 correction), never the clearance-based
  `Target::sdf` lower bound. The residual that choice carries is published: max|e| 0.0335
  on t3 and 0.1775 on t6 against the exact arm at 1x (RESEARCH, `analytic-sdf lower bound`).
  The exact arm (`exact_coverage`) moves with it or stays in `fit` as the reference.
- `fit::analytic::render_analytic` becomes a caller of the core pass, so every published
  analytic column is produced by the same code optical will run.
- Wiring `optical` to depend on `lights` is a separate step, owned by the optical side.
  This ADR only removes the reason it cannot: today it would have to enable `fit`.

---

## Rationale

1. **The pass has nothing a RenderMethod consumes.** A primitive set has no kernel `K`, no
   atoms and no 3D camera. Making it a `RenderMethod` means a `Scene` that carries
   primitives, which is exactly option (ii) that L-73 ruled must be earned and has not been.
2. **The precedent is already in the tree.** L-75 built `RasterLayer<T>` as a composited pass
   in `src/image.rs`, default build, not a method. The SDF pass has the same shape (colour
   plus coverage, laid over a render) and should sit beside it.
3. **Optical should not build the fitter.** Consuming the pass fit-side means enabling a
   non-default feature and compiling 7,973 lines of experiment code into a UI library to
   reach roughly 570 lines that depend only on `numerics`.

### Why NOT the alternatives

**(A) `MethodId::AnalyticSdf` + `RenderMethod::AnalyticSdf(..)`.** Rejected. Beyond reason 1,
every structure that walks `RenderMethod::implemented()` assumes a member renders the atom
scene it is handed: the agreement harness compares methods on the same atoms, the demo draws
one panel per method, and the detachability build runs each method alone. An SDF member
would refuse every one of those scenes.

**(B) Leave it fit-side and have optical drive `fit::analytic` directly.** Rejected by
reason 3, and because the published analytic numbers and the shipped renderer would then be
the same code only by accident of feature flags.

---

## Consequences

- One more default-build module in `lights`; `fit` shrinks by the lifted part.
- The composite is depthless. For 2D UI chrome that is the intended semantics. L-156 shows a
  depthless composite is wrong for a 3D object resting on a floor; that question is L-156's,
  and it does not reach a 2D primitive pass.
- Not settled here: whether primitives beyond `HalfPlane`, `Segment`, `Arc` and `Disc`
  (rounded rect, bezier) are needed by optical. The lift carries the four that exist.
