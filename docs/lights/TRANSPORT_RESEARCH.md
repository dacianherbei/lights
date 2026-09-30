# Lights — Light Transport Research Record

**Date:** 2026-09-08
**Provenance:** Session on branch `atrium/rebuild-loop`. Dacian proposed a
splat renderer demo for the `lights` crate, then successively proposed an
event-driven (neuromorphic-style) transport layer, purpose-built hardware
for it, a chip-partitioning algorithm, a patch-graph reformulation, and a
bit-quantised visibility scheme. Each was put through an independent
verification workflow before being accepted or rejected.
**Status of everything below: VERIFIED unless marked ESTIMATE or OPEN.**

---

## 1. Verdict, up front

The transport experiment **changes the algorithm, not the project. Build
nothing yet.**

The reason is physical and specific to the target workload, not a
difficulty argument:

**In a 2.5D card canvas there is almost no light transport to compute.**
All card fronts face +z, so `cos θ_i · cos θ_j < 0` for every front-to-front
pair and co-facing surfaces exchange exactly zero. On a visible face
`F_sky > 0.99`, so `‖M‖∞ ≈ 0.01` and the second bounce is ≈ 1e-4 — four
orders of magnitude below the 1/256 display quantum. The only strongly
coupled block is the sealed ~4px cavity between a card's back face and the
canvas (`F ≈ 0.98`, `ρ(M) ≈ 0.88`), which nobody can see into.

After honest culling the surviving signal is a rim-local contact shadow
plus colour bleed from a card onto the canvas behind it — which is what a
screen-space pass produces in microseconds with no graph. **No phenomenon
has been named that the transport solve delivers and SSAO cannot.**

This is fatal to the transport crate independently of every other finding,
and no algorithmic reorganisation touches it.

It does **not** bear on the splat renderer itself, which is a separate
deliverable under Horizon 2 (`docs/plan/TWO_HORIZON_PLAN.md` §1).

---

## 2. Prior art — do not re-derive

| Work | What it is | Relation |
|---|---|---|
| Buckalew & Fussell '89, *Illumination Networks* (SIGGRAPH, CG 23(3):89–98) | persistent link network | the true origin of negative-energy shooting |
| Puech, Sillion & Vedel '90 (I3D, CG 24(2):51–57) | interactive radiosity updates | cited by both 1990 papers as the negative-energy source |
| Cohen/Chen/Wallace/Greenberg '88 | progressive refinement, shooting | "first photon triggers photons toward connected points" |
| Chen '90, *Incremental Radiosity* (SIGGRAPH) | geometry queue + per-patch counter | Eq. (8) subtracts old contribution, Eq. (10) re-shoots |
| George, Sillion & Greenberg '90 | radiosity redistribution | "shooting positive and negative energy" |
| Hanrahan, Salzman & Aupperle '91 | hierarchical radiosity | error-bounded refinement; 98 polys, 1.0e9 potential interactions → 15,526 actual |
| Smits, Arvo & Salesin '94 | clustering | O(s² + p) → O(s log s + p) |
| Drettakis & Sillion '97 | line-space hierarchy, shaft index | the dynamic-hierarchy reference |
| Keller '97, *Instant Radiosity* | VPLs regenerated per frame | **not** a persistent graph |
| Laine et al. '07, *Incremental Instant Radiosity* | VPL set kept across frames | 45× on the stage, **6.8× on the frame** |
| Sloan et al. '02, PRT | precomputed transport, SH basis | the operator/source split |
| **Geomerics Enlighten** (shipped ~2011) | cluster-to-cluster form factors precomputed; direct light, sky cubemap, albedo and emissive **streamed** per frame | **the patch-graph proposal, already a product** |
| Bitterli et al. '20 / Ouyang et al. '21, ReSTIR | reservoir spatiotemporal reuse | what actually shipped in real-time path tracers |
| **RadiosityGS** (SIGGRAPH Asia 2025, arXiv:2509.18497, code) | radiosity over Gaussian surfels | **splats as transport nodes, already published** |
| PRTGS (ACM MM 2024) | per-splat transfer | ditto |
| Teller & Séquin '91 | cell-and-portal, PVS bit vectors | 1-bit visibility at cell granularity |
| GS-TG (in `docs/gaussian_splatting/`) | per-gaussian tile bitmask | the bitmask idea, in our own PDF folder |

**Naming:** do not write "event-based" — that phrase plus "gaussian
splatting" already means event *cameras* as input across 8+ papers. Use
*incremental* or *signed-delta*.

---

## 3. Findings that decide it

**F1 — a gaussian splat is a bad transport node.** No unique normal, no
area, no front/back, so operator row sums exceed 1 and a signed solve
**diverges** (the scene slowly brightens; it presents as a tone-mapping bug
and gets misdiagnosed). SH colour is view-dependent by construction.
*Repaired* by patch nodes (`Σ_j F_ij = 1` by construction, `‖M‖∞ ≤ ρ_max < 1`)
— but a one-sided row clamp on the splat operator does the same in one line.

**F2 — edge DISCOVERY, not update, is the cost.** Drettakis & Sillion
measured 0.08 s identifying affected links out of a 0.53 s update (15%) and
0.40 s applying it (75%). The expense is building the index and
re-evaluating visibility.

**F3 — hierarchical clustering is a precondition, not an optimisation.**
Without it the far field is dense and no partition is good (cut → 1 − 1/k
= 94% at k=16). But clustering is an **approximation**, which destroys the
exactness that was supposed to distinguish this from ReSTIR. *Exactness was
never deliverable*: the graph is exact only w.r.t. its own truncation, a
systematic bias that does not average away, while resampling noise does.

**F4 — inverse-square falloff does NOT make transport local.** It makes it
*scale-invariant*: r² growth in partner count exactly cancels 1/r² falloff,
so energy is equal per distance octave. **Occlusion is the sole source of
locality**, giving energy per shell ~ e^(−r/λ):

| | within λ | 2λ | 3λ |
|---|---|---|---|
| energy | 63% | 86% | 95% |
| link count | 8% | 32% | 58% |

Links are mostly long-and-weak; energy is mostly short-range. Weight any
partition by **event volume**, never by edge count.

**F5 — structural plasticity was the hardware disqualifier.** Required
1e7–1e8 edge creations/s vs 1e1–1e3/s demonstrated, and only ever on
platforms whose cores are small CPUs (SpiNNaker, BrainScaleS PPU), never on
Loihi's fixed synapse pipeline. *Dissolved* by static topology — but
self-defeatingly, since zero edge-events during relighting is permanently
below the fabric's own break-even and the kernel becomes SpMV.

**F6 — a dragged node churns its whole incident edge set every frame**, and
dragging is the most common interaction. Worse than first assessed:
tooltips, dropdowns, context menus, drag ghosts, hover scrollbars and
resize handles add and delete occluders **at hover frequency**, in exactly
the 0–10px z-band where all surviving transport lives. Rim-band links scale
with total **perimeter**, not card count (≈3.3× before glyphs), and text is
the highest-perimeter geometry in a UI.

**F7 — purpose-built silicon is not worth building.** See §5.

**F8 — the discretisation is self-contradictory at the only scale that
matters.** The affordable arithmetic used 20px patches; resolving the 4px
contact gradient needs ≤3px. Refine and the band alone becomes ~8e4 patches
(3.5× the whole estimated scene), links ~5e6 (≈25×), drag path 12–25 ms
against a 16.6 ms budget. And patch size is under two incompatible
criteria: scene-fixed (or canvas zoom churns the graph) and pixel-tracking
(or it bands visibly). Zoom is continuous.

**F9 — the scope premise is a scissors.** Patches are near-free only for
authored parametric rectangles — which need no splat renderer, because
`egui::Painter` already draws them. Content that would justify a splat
substrate is reconstructed and unparameterised, where meshing returns
whole, diffuse-albedo extraction from fitted SH is ill-posed, and
non-partition is structural (overlap multiplicity 3–10, so `Σ_j F_ij ≫ 1`).
**No scope justifies the splat substrate and the patch transport at once.**

---

## 4. Correction log — predictions made and refuted

Recorded because the same errors are easy to make again.

| Claimed | Actual |
|---|---|
| Negative-energy shooting is Chen '90 | Predates it — Buckalew & Fussell '89, Puech/Sillion/Vedel '90 |
| Negative first, then re-shoot | Both signs fire together; **positive** erases a stale shadow, **negative** creates a new one |
| Finding dirty edges dominates | 15% find, 75% apply (Drettakis & Sillion) |
| ~1.6 GB at 100 links/node | Presumes a refinement oracle the design lacks; without it 1M nodes is **16 TB**. Realistically 6–13 GB — and 1M nodes is 2–3 orders too many; at 1e4 surfaces the *dense* matrix is ~1.2 GB |
| 5–20× realized speedup | vs a full re-solve, which nobody ships. Against dirty-rect: **1.5–3×, 0-to-negative on pan/zoom** |
| Globally dynamic is slower | Chen measured 1.15× and 1.16× — still faster. Negative only when paying graph maintenance |
| Diffuse-only killed radiosity | **Meshing** killed it first — and splats delete that cause |
| Catastrophic cancellation is the hazard | Avoidable by construction (per-link `last_contribution`, write new−old: order-free, idempotent). Real hazard is Chen's Fig. 5 ordering bug → needs per-node version stamps |
| Splats-as-transport-nodes is novel | RadiosityGS (2025) and PRTGS (2024) already do it |
| A splat is a view-independent point of light | It is not — see F1 |
| Exactness differentiates it from ReSTIR | Not deliverable (F3). Real differentiators: determinism on thin features, zero idle cost, provenance |
| 99% static fraction favours it | Conflates static geometry / illumination / screen; distribution is bimodal with little mass where a graph wins. `draw_node` at 3–8 ms means the static fraction of *work* is 0% |
| 1–10 pJ/bit crossing cost | That is **long-reach SerDes**. Die-to-die is 0.25–2 pJ/bit |

---

## 5. Hardware assessment

**Energy, isolated:** 10–30 pJ per RGB edge-event on custom 7nm SRAM-adjacent
silicon (anchored: TrueNorth 26 pJ, Loihi 23.6 pJ, ODIN 12.7 pJ — three
measured chips) vs ~0.5–3 nJ per irregular edge-visit on one M5 Max P-core =
**30–100×**, central ~50×, ±3×.

**The crossover that decides it:** at IDE scale (N=1e5, K=32, 12 B/edge =
38 MB) SRAM retention leakage is ~9 mW = 153 µJ/frame at 60 Hz. The fabric
beats one CPU core only **above ~1e5 edge-events/frame**: at 1e5 they are
even; at 1e6 the fabric wins 8.7×; at 1e7, 42×. **The high static fraction
is the fabric's worst case, not its best.**

**System level:** repair : propagation = 8 nJ : 40 pJ = **200:1, independent
of the dirty fraction f**. Amdahl cap 1.005× on transport, ~1.001× on frame
energy. A full custom system moves IDE-scale transport from ~58 mW to
~15 mW — **20–60 mW saved on a 5–15 W machine (0.2–1.2%)**.

**Multi-chip:** bandwidth is never binding (one 5 Gb/s lane carries 840k
12-byte events per frame). What kills it is **SerDes static power** —
100–400 mW/lane, link-up regardless of traffic, ungateable at 60 Hz. Four
lanes = 0.4–1.6 W against 1–10 mW of dynamic event energy: a **400–1600×
regression per added unit**. Die-to-die on an interposer (UCIe-Advanced
~0.25 pJ/bit) would invert this, but that requires custom silicon, so the
FPGA prototype path is unrepresentative in exactly the dimension of interest.

**FPGA capacity** (12 B/edge, 32 B/node, degree 32, 80% usable): ECP5-85F
**921 nodes**; Artix-7 200T 3,225; ZU7EV 9,578; Alveo U250 108,890. The
largest fully-open part holds a unit test; the smallest part fitting 1e5
splats idles at 30–40 W.

**Toolchain (verified on this machine, 2026-09-08):** nothing HDL-related
installed; Homebrew present. Verilator / Icarus / GHDL / cocotb / Yosys /
nextpnr install natively on arm64 macOS and cover Lattice iCE40 and ECP5
end to end. Vivado and Quartus do **not** run on macOS; Vivado is x86-64
only. Machine is an Apple M5 Max, 137 GB — `CLAUDE.md`'s "2014 MacBook Pro,
16GB" is stale and misled part of the analysis.

**Partitioning:** cut law `c ≈ 1.1·R/L`; with `L = s·N_chip^(1/d)` the
controlling parameter is the **effective dimension d** of the splat cloud:
d=2.0 → c=2.5%; d=2.5 → 9.8%; d=3.0 → **24%**. Box-countable from an
existing `.ply` in an afternoon. Note c=10% is a 1.9× penalty — the target
is c ≤ 3%. METIS buys ~0.15× energy over a geometric partition and costs
table-based routing; use it once offline as an instrument, not as a
component. Better: portal-aware cuts (Teller & Séquin) put the boundary
where occlusion already is.

---

## 6. Design contributions from the discussion

**Operator/source split (Dacian).** `B = (I − RF)⁻¹E`. `F` depends only on
positions, normals, areas and visibility; `R = diag(ρ)` only on materials;
lights appear **only** in `E`. Verified exact, and exact for *any* BRDF
(`L = (I − KG)⁻¹Lₑ`) — non-diffuse breaks the operator's *compactness*, not
the separation. **This is Enlighten.** Keep it as a self-imposed invariant —
*topology is a function of geometry only; never re-mesh in response to
lighting* — which is what permanently keeps F2's measured killer out of any
future design. State it as a discipline, never as a discovery.

The split lands the wrong way round for an IDE: recolour, theme, selection
and hover become free (they are `diag(ρ)` and emissive, streamed inputs) —
**and they were already free**. Drag, resize, expand/collapse, zoom and the
popover layer all touch the operator, and they are what an IDE does.

**Bit-quantised visibility + activity mask (Dacian).** `F = (analytic
geometric term) × (visibility term)`: store only the non-analytic half.
A second bit separating *potential* from *active* topology turns edge
creation into a **bit flip in a preallocated mask** — 1e8 bit flips/s is
trivial where 1e8 allocations/s is impossible. Sound, and the right
decomposition. ESTIMATE: 38 MB → ~4 MB, leakage 9 mW → ~1 mW. Caveat:
patch-to-patch visibility is usually *partial*; 1 bit gives hard shadow
boundaries where classical radiosity carries a fractional term. 2–4 bits
still buys 8–16×.

**LOD on patch size (Dacian).** This is the hierarchical refinement oracle
that F3 says is mandatory, arrived at independently. The stated worry about
"jitter" correctly names the real artifact — **popping** at level
transitions — and the C^∞ gaussian substrate is the mitigation (smooth
cross-fade where polygonal patches snap). Blocked by F8's contradiction.

**Post-denoising (Dacian) — needs reframing.** OptiX, OIDN and DLSS Ray
Reconstruction remove **zero-mean Monte Carlo noise**. Coarse-LOD and 1-bit
visibility produce **structured bias** — banding, hard shadow terraces —
which a denoiser smooths into plausible wrongness rather than removes.
Reframe as a *learned corrector* trained on this error distribution: pairs
of (coarse solve, converged reference). The `lights` ray-cast method is a
free unlimited reference generator, which is an unusually strong position.

**Rigid-object decomposition.** Internal patch-to-patch form factors are
invariant under rigid motion. For a **convex** card the internal operator is
identically **zero** — the segment between two points on a convex opaque
surface passes through the body. Only concave relief (bevels, insets, ~5% of
area) contributes, and that block is small enough to recompute.

---

## 7. If it were built anyway — the honest shape

A bounded, non-divergent, deterministic **single-bounce** indirect term over
a geometry-derived link set, recomputed from scratch each frame: no cached
solution, no signed deltas, no fabric, no silicon. Transport in
**irradiance**, not radiosity, so albedo stays out of the solved quantity
and a theme change is a free multiply. Push coarse→fine by **replication**
(irradiance is a density, never divide among children); pull fine→coarse as
the area-weighted mean. The coarse graph carries only bounce ≥ 1; direct is
computed at fine resolution, or light-leak halos appear wherever a shadow
crosses a coarse patch.

Never ship the unoccluded form factor "as an optimisation": `F` must always
be `F_unoccluded × V̄`, because the unoccluded factor is exactly wrong at the
shadow boundary where the entire visible signal lives, and it fails silently
as a plausible, energy-conserving, wrong image.

**That is a shader.** A page of code plus an R-tree, living in `optical`
next to the painter — not a crate, and not a revived `lights`.

---

## 8. The measurement ladder

**Do this first, before anything else. One day, no crate.**

> **RUN 2026-09-28 (BURNDOWN L-107).** Rung 1 passes on a light canvas only (coloured contact bleed,
> up to 17 CIEDE2000; invisible on a dark canvas); rung 2 kills the line (the bleed halves within
> 2–4 px of the rim at card elevations, so the patches are ≤3 px). Numbers, command and what is not
> settled: `docs/lights/RESEARCH.json`, slug `l107-transport-gate`.

Render one real atrium canvas — ~60 node cards at real z-offsets over the
real background, real albedos — twice:

1. an offline path-traced reference (an unattended overnight run), and
2. SSAO plus a contact-hardened drop shadow in `egui::Painter`, no
   transport graph at all.

Put them side by side at 100% and 200%.

**Falsification:** if you cannot tell them apart, or the only difference is
a rim tint you would not notice unpaired, the line is dead and it cost a day
instead of three weeks. If you *can* tell them apart, you have the one thing
this analysis never had — **a named visual phenomenon that transport
delivers and a screen-space pass cannot** — and only then does anything else
become worth measuring.

Every other measurement presumes the effect is worth having, and that
presumption is the one that was never tested.

**Second, only if the first passes:** measure the contact-shadow gradient in
pixels at real Δz and confirm the patch size that resolves it is affordable.
At ≤3px it is not (F8).

**Then, only if that passes:** log the per-frame changed fraction across
real sessions (the histogram, not the mean); microbenchmark scatter vs
stream; box-count `d` on a `.ply`; freeze the baseline in writing as the
traditional renderer *with* dirty-rect, idle-skip and viewport culling.

**Kill criteria:** indistinguishable images; or required patch size ≤3px; or
measured out-degree 4–8 (the design collapses into ordinary dirty-tile
invalidation); or IDE renderer power delta < 2 W.

---

## 9. What survives

**The work bound.** `total firings ≤ |E_injected| / (ε·(1−ρ))`, converging in
~8 generations at mean reflectance 0.5. A directly dialable per-frame work
cap — exactly what a 16.6 ms budget needs, and rate-coded spiking networks
have no equivalent. Threshold-and-re-emit also removes progressive
radiosity's global max-residual scan outright. No hardware required.

**The 2.5D closed forms.** For axis-aligned rectangles at discrete depths the
unoccluded form factor is closed-form (Hottel algebra) and occlusion is
rectangle boolean intersection, so discovery is an R-tree range query
`O(log N + c)`. Breaks on rotation, rounded corners, the hex layout already
in this codebase, and curved Bezier wires. Salvage this much.

**The framing that outlives the renderer.** The transport graph is a
dataflow graph with signed deltas, and fulgor *is* a dataflow execution
engine. One shared invalidation spine serving layout, hit-testing,
provenance **and** light — an IDE rendering itself through its own execution
model — stands whether or not the renderer beats a screen-space AO pass.

**A note on the substrate.** `docs/war_map.toml:66` already marks `lights`
`group = "Legacy"`, `status = "Deferred"`, "Archived — not carried forward".
This line of work is a reformulation of an experiment the project's own war
map had already retired. That is not an argument against the splat renderer,
which is a separate Horizon 2 deliverable — but it is worth knowing.

---

## 10. Provenance

Five verification workflows, ~2.5M subagent tokens. Raw per-agent returns
are in the session transcript directory under
`subagents/workflows/wf_*/journal.jsonl` — they are not durable across
machines, so the conclusions above are the record. Confidence is marked
where it is not high; anything labelled ESTIMATE is arithmetic, not
measurement.
