# Two-Horizon Plan — Fulgor Development

**Date:** 2026-07-05
**Provenance:** Planning discussion between Dacian and Claude (Fable 5
session) on branch `wiring/spiral-unify-layout`, immediately after the
spiral inner-wire routing work closed (commits `e9d1106`, `92bf84e`,
1341 optical lib tests passing). Written in full detail for continuity
across model/session switches — the next session should be able to act
from this document alone.

**Status of every decision below: SETTLED unless marked OPEN.**

---

## 1. The Two Horizons

**Horizon 1 — finish fulgor.** Atrium integrated with forge/nexus, proofed
by experiments. User: *"optical was a nice detour but we need to finish
fulgor."* Scope gate, verbatim intent: **no new optical feature unless a
track's deliverable demands it.**

**Horizon 2 — rendering evolution.** Optical eventually runs on a new
engine (`lights` crate) based on the fulgor/nexus runtime. Experimental
basis: **gaussian splatting** (extensive docs in `docs/gaussian_splatting/`,
esp. "Efficient Geometry Compression and Communication for 3D Gaussian
Splatting Point Clouds") — high compression, potentially faster rendering,
and a path to 3D. Concept: **points in space as lights which emit events
that change over time as the geometry creating them changes.** Long-term
goals FRAME the short-term ones; they are not current work items.

> **AMENDED 2026-09-29: ONE TARGET, AND A MISSION.** `lights` is an experimental proving ground
> for the **fulgor/nexus paradigm**, and its renderer target is **a simple 3D game engine**. The
> half of L-120 that made it the renderer under `optical` is WITHDRAWN — L-174, L-175 and L-176
> measured that the port would make optical slower on more cores for no behaviour gain, and
> optical's source is off limits. The paragraph below stands for the 3D half only.
> **(PM, 2026-09-09, superseded in its `optical` half.)** `lights` is the renderer under
> `optical` **and a simple 3D game engine**. The 3D half is not a later phase: `Atom` is already a
> 3D primitive with a quaternion and an anisotropic scale, the camera already carries a pinhole
> projection, and both image-formation methods already resolve occlusion. What the second target
> changes is which absences count as defects — a scene that cannot be mutated or saved, a primitive
> that cannot carry a raster, an atom without a normal, and the fact that nothing owns a frame are
> survivable in a UI and blocking in a game. Recorded and re-weighted in
> `crates/lights/BURNDOWN.toml` as L-120 and the `three-d-game-engine` cluster.

**The architectural contract binding both horizons:** every feature
produces **engine-agnostic SCENE DATA plus a thin egui painter — never
egui-entangled logic.** Routes, bundles, elevation fields, LOD decisions,
selection state, animation timelines, decorations: all data. egui only
strokes it. That is the entire `lights` migration path: the data model
survives, the painter is swapped.

> **AMENDED 2026-09-08 BY MEASUREMENT.** "The painter is swapped" is now
> contradicted for *crisp chrome*, and the crate that was built to do the
> swapping is what disproved it. Experiment P2 fitted six UI content classes
> with gaussian atoms under two blend rules and two prefilter profiles, and
> compared them against an analytic renderer drawing the same shapes from the
> same definition. Analytic won every crisp class. On the 12 px glyph it is
> the **only** arm that meets the criterion at any budget, at **274 B and
> 38.9 µs** against 1,344–2,464 B and 54.9–120.7 µs for fits that all fail.
> Splats matched analytic on exactly one class — flat fill — which analytic
> renders exactly, for free.
>
> Aspect 1 then swept the atom curve and found β is **not** inert: a
> super-gaussian at β=4 closes the corner gap to the ceiling 11× and costs
> 2.39× less than the gaussian. But no curve meets the glyph criterion, and at
> pixel scale the ranking reverses — the gaussian is the only kernel meeting
> on a 1 px stroke.
>
> **What survives, and it is narrower but real.** Analytic renders everything
> you already hold as geometry: fills, borders, rules, wires, text. `lights`
> renders what has no analytic description to hold — continuous fields (the
> §2a elevation work), photographic and *fitted* content, 3D. And β=4, not the
> gaussian, is the default for that role. The same split was reached
> independently by the transport research from unrelated evidence
> (`docs/lights/TRANSPORT_RESEARCH.md`).
>
> **Caveat that keeps this honest.** Every content class tested was one an SDF
> renders exactly. The content splats exist for — photographs, soft shadows,
> fitted textures — has *no analytic arm at all* and is still unmeasured, so
> this bounds where splats lose, not where they win. Full table, images and
> error maps: `docs/lights/P2_RESULTS.md`.

---

## 2. Settled Technical Decisions

### 2a. Elevation semantics (contours) = NESTING DEPTH
Node/ChildrenLayout backgrounds painted as map elevation levels.
Implementation: elevation is a **scalar field** — one smooth gaussian bump
per node, weighted by nesting depth — with iso-lines extracted by marching
squares. All the user's cartographic contour rules then hold
*automatically* (iso-lines of a scalar field cannot cross, merge, or
split; spacing encodes slope; closed concentric loops = hills; hachures on
depressions by sign; wires carved as valleys give the V-points-upstream
pattern from the gradient).
**Convergence:** a field of gaussian bumps IS morally a 2D gaussian splat
scene — the per-node bump data (position, amplitude, covariance) is
directly what `lights` renders natively later, and node events modulating
amplitude over time is the points-as-lights model. Contours, done as a
field, are simultaneously a visual feature and splat experiment #1.

### 2b. Lanes vs rivers = an LOD feature, not a user aesthetic
Two wire looks, shared across all three layouts once built: **lane-based**
(dense, detailed) and **S-3 rivers as bordered bundles** (lighter, faster).
Three-layer control:
1. **LOD algorithm** (default ON in atrium) picks lanes vs rivers by
   zoom/budget automatically.
2. **Debug overrides**: force-lanes / force-rivers.
3. **Kill switch**: disable the detailed lane mode entirely → rivers-only
   for slow systems.
In the **optical showcase**: a manual toggle between modes (for testing).
In **atrium**: LOD auto is the default.
Selection-based highlighting is **orthogonal to the mode** — active
elements read as active in both looks, which is what makes automatic
switching acceptable UX.
**Cost containment:** rivers are a bundle-envelope PRESENTATION over the
same `RadialParentRoute` chains as lanes — ONE routing pass, two stroke
passes; swapping painters is nearly free. LOD threshold gets hysteresis so
buffer caches don't thrash. The slow-system tier saves painting, not
routing; only swap in cheap taut centerlines (existing `river_routing`)
behind the same interface if profiling demands it — measure first.

### 2c. Settings live APP-LEVEL (not per-graph ViewState)
User rationale: as fulgor bootstraps itself (applications as graphs of
activated functions), performance will be fed back **by forge directly
into the running engine** (aligns with the PGO/DeploymentPlan vision in
`docs/NEXUS_SERVICE_VISION.md`).

### 2d. rkyv migration: OPPORTUNISTIC, not frozen
Adopted when a track touches serialization anyway; serde dual-coding is
fine where it aids experimentation. No standalone migration push.

### 2e. Tiers: MVP demos on Inline; ALL tiers develop in parallel
"MVP uses inline due to its simplicity" is a **demo constraint, not a
development freeze** — isolated and resilient continue advancing in
Track 1.

### 2f. Grid layout standardized initially
Plan display (and initially everything in atrium) uses **grid layout for
compactness**; radial/spiral support added later. The
plan-overlay-over-project-display idea (radial link lines connecting plan
nodes to code nodes) is a later slice (Track 0 v2).

---

## 3. Parallel Track Structure

The user runs **parallel Claude sessions** (budget allows). The plan is
parallel tracks drawn on **CRATE BOUNDARIES** — merge surface is the
constraint, not Claude throughput. The user's review bandwidth is the
global limit; Track 0 exists to raise it.

| Track | Content | Crates touched | First deliverable |
|---|---|---|---|
| **0. Review interface** | Gerrit-like flow in atrium (KEYSTONE — see §4) | atrium (+ optical text widgets) | plan panel + timeline (v0) |
| **1. Forge/nexus proofing** | Experiments-backed hardening; isolated + resilient advance here | forge, nexus, running_connections, experiments | experiment backlog from B.10 gaps (Resilient completion path) |
| **2. Graph MVP** | Syn parser, RuntimeState display (V2/Q-042 lands via the blocking list, §5) | optical/node_graph, atrium | one-shot syn parser → V2 model |
| **3. Agora** | Code-market crate — DESIGN FIRST | new crate (design only) | design doc + market mechanics |
| **4. UI look & feel** | `docs/theme_design_research` continuation; theme metrics/baseline (calligraphy) | optical/theme | metric/baseline rules |
| **5. Decoration system** | Widgets as materials, animated vines, … | optical (painter layer) | data-model design |

**Interference boundaries:** Tracks 4 and 5 both touch optical
theme/painting — stagger or split files explicitly. Track 2 owns
`optical/node_graph`. Track 0 owns atrium panels. Tracks 1 and 3 are
clean. Pause any track if interference emerges.
**Track 4 → Track 0 synergy (review pass 3):** the diff/comment panels are
text-layout-heavy — the theme metric/baseline (calligraphy) work's FIRST
application is Track 0's text panels, so Track 4 feeds Track 0 instead of
competing with it for optical files.

**Agora staging (pushback, accepted):** its exchange artifact ("what is a
tradable unit of code?") depends on the forge binary + metadata contracts
that Track 1 is still proofing. Design + market-mechanics research
(content addressing, trust/verification, the NexusService
closed-source-distribution story) proceeds now; **crate code is held**
until the metadata contract stops moving.

**Decoration constraint (pushback, accepted):** every decoration declares
**static|animated from day one**, so the future buffer system (Imp 7) can
bake static decorations into cached buffers and run animated ones on the
dynamic overlay. Decorations are data layers per the scene-data contract.
Animated decorations are otherwise the exact enemy of
"paint-inactive-parts-into-buffers".

**Review-load balancing:** ≤2–3 tracks active per day, mixed review types
(code / design-doc / visual) — visual review via showcase screenshots is
much cheaper for the user than diff review.

---

## 4. Track 0 — The Review Interface (KEYSTONE)

### Why it outranks the graph MVP in atrium
1. User's review bandwidth is the bottleneck of the whole parallel plan;
   this tool raises it. It is the force multiplier for every other track.
2. User requirement: *"if I don't know where we are, what we did and what
   is the aim I can review wrongly your commits"* — plan context is a
   prerequisite for correct review.
3. **It is aegis's front door.** Review comments, flagged hunks, questions
   asked, role switches, thread resolution patterns — that is exactly the
   behavioral corpus aegis (the system that analyses user behavior to help
   them build software with fulgor) will learn from. Persist every review
   thread as append-only JSONL from day one.

### The four components + the consistency mechanism
1. **Plan display as a graph** — FIRST. Renders the canonical plan file
   (see below) as a NodeGraph, grid layout. Later (v2): overlay over the
   fulgor project display with radial link lines.
2. **Commit timeline widget** — AUTHORITATIVE SOURCE = `git log` (review
   pass 1: `docs/timeline/timeline_raw.jsonl` is deliberately UNTRACKED
   per standing directive, hence local-only, unshared, unbounded — it may
   serve as optional local enrichment, never as the source of truth).
   Implementation choice (git2 vs gitoxide vs shelling out) is v0's first
   design decision.
3. **Diff tool with comment + edit** — commit → files → hunks, syntax
   highlighting; select range → comment; discussion threads.
4. **Chat interface** — in-panel coworker-style discussion (v2).

**Consistency mechanism (user requirement: "my view and your view remain
consistent, or divergence can be corrected"):**
- **One canonical PLAN FILE in the repo**: `docs/plan/` (this directory),
  TOML (matches the `open_questions.toml` precedent), data model =
  **extension of the narration system's Epic→Feature→Plan→Items hierarchy**
  (iteration 1 already done; V2 narrative widgets already in atrium).
  Atrium RENDERS it; Claude READS/EDITS it via ordinary commits — so plan
  changes arrive as reviewable diffs through the very UI that displays
  them. **RULE: the plan file is authoritative for plan state; Claude's
  memory only points to it** (prevents dual-maintenance drift).
  The TOML additionally encodes, per track: its **crate-set (ownership)**
  and a **status/current_activity field** — so the plan panel can (a)
  auto-warn on crate overlaps when a new task starts (tool support for the
  conflict pre-discussion) and (b) act as the status heartbeat across
  parallel sessions the user cannot watch simultaneously.
- **`view_state.json`** — atrium continuously exports what the user is
  seeing (current tab, selected plan nodes, visible commit range, open
  threads, viewport). Claude reads it in-session. Belongs in **v0**: it is
  cheap and closes the alignment loop before chat exists (the ordinary
  Claude session + view_state.json IS the chat until v2). Also solves the
  "third node from the left" grounding problem permanently.
  **Location: `target/view_state.json`** (already gitignored; precedent:
  `target/graph_config.json`); one file per atrium instance if multiple
  run.

### Roles and write ownership (SETTLED — resolves the direct-edit question)
- **Designer role** (user reviews Claude's work — optical, atrium): Claude
  writes the tree; user comments + attaches **suggested-edit patches**
  (Gerrit-style) that Claude applies.
- **Student role** (user learns by writing — **forge and nexus**, user:
  "there I have a lot to learn"): **the USER writes the tree directly;
  Claude reviews/teaches** — comments and suggested edits flow the other
  way.
- **Role = single writer per TRACK'S CRATE-SET at any moment** (review
  pass 1: "per tree" was too coarse for a monorepo — student mode on
  forge/nexus must not block Claude committing plan edits in docs/plan/).
  `docs/plan/` and `docs/reviews/` are SHARED-WRITE zones (small text
  files; conflicts trivial; protected by the review flow itself). No file
  locks needed; role handoffs go through the conflict-discussion
  protocol (§6).
- **Thread schema is SYMMETRIC from day one**: `author` and `reviewer` are
  fields, not assumptions. Same widgets serve both modes. Bonus: the
  corpus then captures both reviewing AND learning behavior for aegis.
- Threads carry **role tags** (designer-review / student-question) per the
  teaching methodology used to build fulgor.

### Track 0 slices (re-sliced by dependency, review pass 3)
- **v0a — NO optical dependency (atrium crate only):** canonical plan TOML
  format + commit timeline (git-log-based) + `view_state.json` export.
- **v0b — NO optical dependency:** read-only diff viewer (commit → files →
  hunks, syntax highlighting).
- **v0c — needs the optical blocking list (§5):** plan graph panel
  (grid-layout NodeGraph populated from the plan TOML via add_node).
- **v1:** comment threads + suggested-edit patches; Claude reads + applies
  in-session. **Storage: committed, append-only JSONL in `docs/reviews/`**
  (durable + shared = usable as the aegis corpus; a Gerrit-NoteDB-style
  git-ref store is a later refinement). **Thread schema v1 (fixed now to
  spare the corpus a migration):** `{id, commit, file, range, author,
  reviewer, role_tag, body, status, parent_id}` — symmetric by
  construction.
- **v2:** in-panel chat (design question: aegis subprocess vs file
  handshake — beware the historical kill-terminal-on-test problem that
  deferred atrium chat before), resolve states, plan-overlay-over-project
  display.

### Track 0 acceptance gates (added, review pass 2)
- **v0 gate:** user opens atrium and sees the plan graph reflecting the
  docs/plan TOML + the commit timeline of the current branch; Claude,
  reading `target/view_state.json` alone, correctly states what the user
  is currently viewing.
- **v1 gate:** one full review round-trip entirely through the files —
  user comments on a hunk → Claude reads the thread in-session → applies a
  suggested edit → thread marked resolved.

---

## 5. Optical Blocking List (FULL CUT — confirmed)

User: *"finish optical blocking stuff first before moving to atrium."*
Full cut confirmed over minimal cut. In order:

1. **Merge `wiring/spiral-unify-layout`** — spiral inner-wire routing done
   and user-confirmed (see `docs/wiring/spiral_requirements.md` + memory
   `project_inner_wire_routing_model`); sibling wires verified.
   **Merge order (review pass 2): spiral → main FIRST, then rebase the
   grid branch onto main — item 2 verifies the REBASED state, never the
   stale branch** (both branches touch the same wiring crates).
2. **Verify + merge `wiring/grid-show-wires`** — grid corridor
   architecture landed 2026-07-03, awaiting the user's visual gate. Needed
   because atrium standardizes on grid layout. (Same gate workflow as the
   spiral session: showcase + jitter-dbg + one fix at a time.)
   **Done-ness (review pass 2): merge = wires render correctly + no
   regressions. The known open polish (hand-off tail, G-EDIT, radial
   palette port, child title-bar buttons, place-oscillation) is explicitly
   PARKED to Horizon 2 — it does not gate the merge.**
3. **Improvement 1 — unified handoff connector**: remove handoff points
   from radial and grid ChildrenLayout shapes; use the spiral connector
   (cone/chord blocker + direct biarc with S/simple + L-cap + cusp-free
   direction-following round-around) to connect to the SAME computed
   points. Point computation unchanged; only connection geometry swaps.
   Radial was signed off 2026-07-02 — keep that sign-off intact.
4. **Task 3 — consolidation**: port the wiring algorithms from the Graph
   tab to the shared **NodeView layer** (review pass 1: the plan panel is
   read-only — NodeView, not NodeEditor alone, must receive the port);
   retire all other wiring versions. Every atrium panel builds on the
   consolidated path ONCE. (Rivers are parked ⇒ wiring is stable after
   Imp 1 ⇒ consolidate immediately, no detour.)
   **Pre-step (review pass 2, mandatory): a half-day RETIREMENT INVENTORY
   — enumerate every wiring path to port or retire (route_channel, D-004
   radial painter, spiral_route canvas usage, river legs,
   corridor_routing, cell_wire_routing, …) BEFORE any porting. This item
   is the biggest unknown in the list; do not start it unsized.**
5. **Q-042 — populate V2 canvas via `add_node()`** — MOVED UP from
   Track 2: the plan graph panel IS a programmatically-populated NodeGraph.
   **Data source = the plan TOML itself (review pass 3), not
   war_map.toml — proving add_node and rendering the real plan become the
   same act; this item IS Track 0 v0c's core.**

Then atrium Track 0 v0c begins (v0a/v0b have no optical dependency — see
Open Question O-1 on whether they may start in parallel).

### 5b. NodeEditor→mlp port (settled 2026-07-13 — successor to Task 3 P2/P4)

Status update on the list above, post-mlp (ADR-018): items 1 and 2 are merged
to main; Task 3 P0/P1/P3-partial live on the mlp branch
(`layout/measure-layout-paint`, contains `wiring/task3-consolidation`);
**Imp 1 (item 3) is superseded** — the mlp pipeline replaced handoff points
entirely (radial projections, grid outer roads, spiral hub fans).

User decision 2026-07-13: **the next big step is porting the NodeEditor/Canvas
tab to the mlp pipeline and cleaning up the legacy implementation.** Settled
design (drag/drop + affinity spec: `docs/wiring/DRAG_PREVIEW_AFFINITY.md`):

- **Root = mlp container in structured canvas modes** (Grid/Radial/Spiral):
  mlp lays out the top level like any node; dragging is already
  swap/cell-targeting there. **Manual/free mode:** host positions pass
  through, wires via `route_channel`. **Tree + RouteV2/Sugiyama keep their
  existing paths** (no mlp equivalent; `route_v2_edited` untouched).
- **Wire-model migration:** `UnifiedMapV2.wire_connections` → root sibling
  wires (`children_wires`); map-decorator adjacency reads the root's wires;
  rkyv snapshot converter (or accepted regeneration); host mirrors top-level
  selection into root node flags for pulse parity.
- **Decorator framework KEPT:** un-suppress the chain under `use_mlp`; only
  `InnerWiringDecorator` retires (mlp IS its replacement); border/visibility/
  intent/debug-trace/on-demand decorators run over mlp-bound geometry.
- **Editing API + GEF CommandStack KEPT**; every drop = one atomic reversible
  Command. Zoom/pan: already flows into mlp as `(center, scale)`; Graph tab
  gains pan/zoom (currently hardcoded 1.0).
- **Reroute on drop**; `ChromeCache` on the Canvas path (per-frame rebuild is
  a perf trap); gate with `MLP_PERF` on the 2014 MBP reference machine.
- **Drag landing preview** (RTS formation markers) view-first, algorithm
  second. Grid/radial gain **explicit slot addressing** (ADR-007
  `Placement<T>`, auto-shrink of empty strips); spiral gains the **affinity
  model** (flat symmetric matrix, no hubs — hub-inheritance considered and
  rejected; layout consumes affinity instead of connections).

Slices: (1) root container + structured-mode layout at root, (2) wire-model
migration, (3) decorator un-suppression + InnerWiring retirement, (4) flip
`use_mlp` default + visual gates per canvas mode + retire the legacy
`paint_wires` residue (Tree/RouteV2 survive), (5) drag preview view-first,
(6) `Placement<T>` slots + affinity model. Q-042/blocking item 5 still
precedes atrium Track 0 v0c.

---

## 6. Working Protocol (user commitment)

- **≤2–3 tasks running simultaneously.**
- **Every new task is preceded by a conflict discussion** against the
  running tasks (crate-boundary table in §3 is the early-warning map).
- **A paused task keeps ownership of its crates** unless explicitly
  released — pausing-to-think-deeper is a designed-for pattern, and
  ownership prevents the pause from inviting conflicts.
- Role handoffs (designer ↔ student on a track) go through the same
  conflict discussion.

---

## 7. The Aegis Record

The user challenged a perceived pattern of deferring aegis ("a system that
analyses the behavior of users and builds a system that helps him or her
better build software with fulgor"), suspecting an external directive to
slow AI-improvement work. For the record: **no such directive exists**;
the 2026-07-05 "defer the aegis pipeline" suggestion was parser scope
discipline for the MVP critical path only (one-shot syn parse vs the
6-milestone incremental pipeline), and prior work (iris ClaudeCodeAgent
chain, 3-tier narrative classifier, L2 Candle plan) is all aegis-family.
**Aegis is strategic. It is never to be quietly deferred.** Track 0's
JSONL review corpus is deliberately designed as aegis's first behavioral
data source; the one-shot parser's output type is deliberately the future
aegis-pipeline contract.

---

## 8. Horizon 2 — Parked Inventory (unlocked by gates, not dates)

> **Every `lights` experiment is in ONE consolidated record**, and it is the place
> to look before quoting any figure below: `docs/lights/RESEARCH.md`, generated
> from `docs/lights/RESEARCH.json`, read as a tree at
> <https://claude.ai/code/artifact/aaf71a60-76fb-4964-aad3-a91a3e9be68f>. It
> carries all four aspect runs — including **aspect 2**, whose findings had no
> document of their own and lived only in commit `251381a7`'s message — with each
> experiment's reproducing command, its criterion AND the convention that
> criterion uses, what it refuted, what it does NOT settle, and a standing audit
> section. `P2_RESULTS.md` and `P5_RESULTS.md` remain as the long-form write-ups
> of two of those runs; they are sections of the record, not the record, and both
> carry published-number corrections that the record states in full.

- **Rivers (Imp 3/4)**: S-3 bordered bundles; radial first, then grid +
  spiral; shared bundle-envelope renderer over shared routes.
- **Radial lane-based wiring (Imp 2)**: second display mode, from the
  post-Imp-1 state.
- **Wire animations (Imp 5)**: selection-driven (pin/node/multi-node);
  must live on the dynamic layer above buffers.
- **Contours (Imp 6)**: per §2a; implementation waits for the Imp 7
  layering decision (terrain is the ideal cached background).
- **Buffers/LOD (Imp 7)**: research may run early (it GATES 5 and 6).
  Research questions: egui texture-buffer path; invalidation policy
  (hovered/selected/animated/edited = live, rest cached); LOD tiers
  full → simplified → sprite → dot; calligraphy tie-in — LOD preserves the
  proportion hierarchy (silhouette before interior detail, minimum stroke
  width in SCREEN pixels).
- **Spatial index + culling**: dual-purpose flagship — quadtree culling
  for egui now, splat LOD/streaming octree later. MAY jump the queue if
  125-node profiling demands it. Measure first (draw_node 3–8ms is the
  known bottleneck).
- **Edge aggregation at scale**: rivers beautify ~50 wires; 5812 (the HEB
  lesson) needs data-level bundling (aggregate parallel edges between
  containers into one river + count badge).
- **Contrast management**: once terrain tints backgrounds, wire palette
  needs a luminance floor defined relative to terrain, not absolute RGB.
- **Semantic zoom**: LOD by meaning (far zoom → module containers only).
- **Full edit-mode hardening** (all 3 layouts) + shared EditableLayout
  trait (fix drag/edit bugs once, not three times).
- **Direct working-tree editing from atrium** outside the role protocol —
  needs ownership/locking design.
- **`lights`/splat experiments**: one validation question each (the
  B.4–B.8 pattern); first candidates — splat-render the same scene data
  optical produces; test the compression paper's claims on a real graph
  scene. **REVISED 2026-09-28 (BURNDOWN L-106).** The pattern was built,
  but not in `experiments/`: it lives as four `fit` binaries inside the
  crate, `crates/lights/src/bin/lights_p{2,3,4,5}.rs`, and every result is
  in `docs/lights/RESEARCH.json`. `experiments/` holds no lights work. The
  first candidate is answered by a display list, not a scene exporter:
  `lights::draw::DrawList` (L-76), with one optical widget, `rating.rs`,
  ported onto it in `crates/lights/tests/draw_list.rs::rating_walk`. There
  is no crate dependency in either direction. The second candidate has
  **not been run**. The seam for it now exists (`DrawList::render` →
  `RasterLayer::over` a canvas → `FloatImage::to_clamped_u8` →
  `ExactReference::from_raster`, L-155; composed here, never called),
  but no fit has been scored against an optical canvas.
- **Patch light transport (`lights`)**: nodes as surface patches, the
  link graph a function of geometry, lights streamed onto it. That is
  Geomerics Enlighten (2010); it fixes divergence (Σ F ≤ 1 by
  construction), but so does a one-sided row clamp on the splat
  operator. One question — render one canvas twice, path-traced
  reference vs SSAO plus a contact-hardened drop shadow; if they are
  indistinguishable the crate is dead. Prior art, do not re-derive —
  Enlighten cluster form factors; PRT (Sloan '02); Drettakis & Sillion
  '97; RadiosityGS (SIGGRAPH Asia 2025). Deciders that changed:
  co-facing fronts in a 2.5D canvas exchange nothing, so ‖M‖∞ ≈ 0.01
  and no second bounce; the 20px patch that makes N small cannot
  resolve a 4px contact shadow; tooltips churn occluders at hover
  frequency, so topology is not geometry-static. GATE — unchanged,
  plus: name the effect SSAO cannot produce. Salvage only the 2.5D
  closed-form factors and boolean occlusion — a page of code. Full
  record and correction log: `docs/lights/TRANSPORT_RESEARCH.md`.
  **GATE RUN 2026-09-28 (BURNDOWN L-107) — THE PATCH LINE IS DEAD AT
  RUNG 2, NOT RUNG 1.** The two renders are NOT indistinguishable on a
  light canvas: transport adds a coloured contact bleed (a card's colour
  reflected off its underside onto the canvas, up to 17 CIEDE2000, a
  fifth of the visible canvas over the JND) that neither SSAO + drop
  shadow nor GTAO multi-bounce reproduces — that is the named effect.
  On a dark canvas it is invisible (max 1.17). But the bleed halves
  within 2 px of the rim at 2–4 px elevation and 4 px at 8 px, so the
  patches that resolve it are the ≤3 px the ladder names as its kill.
  Build no transport graph. The two things NOT measured are the cheap
  way to deliver the effect (a screen-space colour-bleed pass, SSDO
  class) and anything about the 3D-game target. The canvas was crafted,
  not captured from a real atrium session. Record:
  `docs/lights/RESEARCH.json` slug `l107-transport-gate`; command
  `python3 crates/lights/scripts/transport_gate.py --out
  docs/lights/transport_gate`.
- **Decompose the prefilter (`lights`)** — **THE FIRST TWO CLAIMS OF THIS ITEM
  WERE WRONG AND ARE WITHDRAWN, 2026-09-15 (BURNDOWN L-37). What they said is
  kept below because the withdrawal says more than the correction does.**
  They said: *"`settings.rs` ships `screen_dilation = 0.3` as one constant"* and
  *"`project.rs` DISCARDS [the gain] unless `energy_compensation` is set, and it
  is false in both shipped profiles"*.
  Measured against the tree: `REFERENCE_DILATION = 0.3` (src/settings.rs:138)
  reaches exactly one profile, `reference()` (:51), and `ProfileChoice` — the
  only profile selector any experiment has — offers `Prefiltered` and `Contract`
  and nothing else (src/fit/harness.rs:47-50). So **no experiment in this crate
  has ever rendered with 0.3**; every published number was measured at
  `screen_dilation = 0` (contract) or at `PIXEL_BOX_VARIANCE = 1/12`
  (prefiltered, src/settings.rs:72,137). 0.3 survives only in
  tests/splat_projection.rs. And `prefiltered()` sets
  `energy_compensation: true` (src/settings.rs:73-77), which
  src/method/splat/project.rs:151-155 then applies — true since commit 6ac92f33
  on 2026-09-08, the day BEFORE this item was written at b3d82ae4, so the
  sentence was already false when it was committed. It was inherited from
  docs/lights/CORPUS_READING_MAP.md:147 rather than measured.
  **There is also nothing to DECOMPOSE.** `screen_dilation` has exactly one
  consumer in the whole crate, `K::prefilter(params, cov, pixel_variance)` at
  src/method/splat/project.rs:140, and the trait names that parameter
  `pixel_variance` because that is what it is. A grep for any object-space or
  volume prefilter over `src/` returns nothing. So the crate does not apply a
  0.1 + 0.2 sum that needs splitting: it applies the **screen half only**, and
  its value is not half of anything — 1/12 is the exact variance of a box one
  pixel wide. The object half was never built. The real question is therefore
  not "split the constant" but **"should an object-space term exist at all, now
  that lights is also a simple 3D game engine (L-120)?"**
  What survives unchanged: Mip-Splatting (arXiv:2311.16493) Eq. 10 carries the
  prefactor `sqrt(|Σ|/|Σ+sI|)`, which `kernel/gaussian.rs` computes as `gain`;
  for a 2D orthographic UI only a screen-space term applies, and a 3D target is
  where an object-space term would earn its keep.
  Zwicker's EWA Volume Splatting (VIS 2001) Eq. 22 settles which is right: the
  resample filter carries the same prefactor as the unconvolved footprint, so
  the INTEGRAL is preserved, not the peak. The crate's own
  `agreement_rs_comment/preconditions.md` already measures the mismatch at ~3%
  for a 3px atom and calls it "the number-one false bug". One question — **with
  the filter decomposed and compensation on, does `prefiltered()` become the
  default, and what does that move in the zoom and minification columns?**
  Until then, any minification result measured under `contract()` — which has
  NO pixel prefilter at all — is a statement about the missing filter rather
  than about the representation.
  **MEASURED 2026-09-09 by aspect 3** (`docs/lights/P5_RESULTS.md`, §C3 as
  amended): the prefilter **buys error and sells sharpness**. On a 30° edge
  minified from 8× to a quarter, `prefiltered()` holds rel L2 flat at
  0.056–0.068 where `contract()` turns up to 0.113 — the C3 result, and it
  stands — while widening the 10–90 transition from 0.38–0.57 scene units to
  0.63–**2.94**, so the edge stops meeting a sub-unit sharpness criterion from
  1× down. An arm can be closer to the truth in L2 and further from the product
  requirement at the same time. That makes the decomposition question sharper,
  not weaker: a 2D UI wants only the screen half, and the object half is the
  plausible source of that blur. Nothing measured the split yet.
  **CORRECTED 2026-09-15 (L-37): it cannot be the object half, because there is
  no object half.** The prefiltered arm ran at `PIXEL_BOX_VARIANCE = 1/12`, a
  screen-space pixel-box variance and the only non-zero dilation any experiment
  has used; the widening from 0.63 to 2.94 was produced by the screen term
  alone, applied at every rung of the ladder including the ones where the atom
  is already smaller than a pixel. A decomposition cannot recover that sharpness
  because there is nothing to take out.
  THE OTHER HALF OF THE QUESTION IS BLOCKED, NOT UNMEASURED: whether
  `prefiltered()` becomes the default cannot be certified by this crate's own
  cross-method instrument, because src/agreement.rs:49-54 REFUSES any non-zero
  dilation or compensation outright (ContractViolation::DilationEnabled /
  EnergyCompensationMismatch), and the ray-cast path refuses them too
  (src/method/raycast/implementation.rs:42-52). A PM ruling is needed on which
  of the two the contract follows before the experiment can be built.
- **Re-tabulate the byte axis symmetrically (`lights`)** — cheap, and it
  decides whether a published conclusion is a result or an artefact. Aspect 2
  billed a wavelet coefficient at 4 bytes and an atom parameter at 8, with
  nothing pinning it: mutating the constant passes the entire suite. Bill both
  at f32 and the atom at its NATIVE nine scalars — which is what
  `Atom2d::bits()` actually returns, against the 14-scalar 3D atom the run
  carried forward — and the headline "transforms win on all four classes by
  3.1× to 9.7×" becomes 3.6×, 2.4×, 1.8×, with the soft shadow flipping to the
  atoms; on the carried-forward classes the atoms then sweep all eight cells.
  One question — **under a symmetric convention and the native atom width,
  does any transform byte-win survive?** This is a re-tabulation of data
  already on disk, not a new experiment, and it should be done before anyone
  cites the aspect 2 byte column. **REVISED 2026-09-28 (BURNDOWN L-108).** The
  currency half is built: `atom_bytes` bills at the same f32 `VALUE_BYTES` a
  coefficient pays (36 B a native atom, not 72), aspect 3's
  `SYMMETRIC_VALUE_BYTES` is that same constant, and
  `tests/nterm.rs::an_atom_scalar_and_a_coefficient_value_are_billed_in_one_currency`
  pins it. The "data already on disk" clause is WITHDRAWN: `docs/lights/p4/`
  holds no error or byte column, the harness that produced 3.6×/2.4×/1.8× is
  gone, and no binary calls `nterm_curve`. The question is unanswered and needs
  one new committed run. **ANSWERED 2026-09-28 (L-108): YES, on three classes,
  for one transform.** `lights-p4 --nterm --max-atoms 64` wrote
  `docs/lights/p4/nterm-64/crossings.tsv`. CDF 9/7 is cheaper than the atoms at
  equal relative L2 on n1, n3 and n4, by 1.04× to 3.13×, and that holds even on
  its pessimistic bracket. The atoms are cheaper on n2 and on every T-class rung
  past the first two. Haar wins 1 rung of 160. The 3.6×/2.4×/1.8× figures are
  not reproduced as a statistic, but their order n1 > n3 > n4 and the n2 flip
  both reappear. Every transform win is against a greedy atom lower bound. The
  full record is in `docs/lights/RESEARCH.json`
  (`p4-aspect2-signed-atoms-and-wavelets`).
- **A gradient-descent fitter for `lights`** — the highest-value item on
  this list, because it bounds every atom number the project has produced.
  The crate has no gradient path: greedy matching pursuit stopped 20 of 24
  cells in P2 and *exhausted* on two of four classes in aspect 2, and
  hand-built atoms beat it in both directions — one needle met a criterion
  where the fitter needed twelve, and a four-atom construction passed while
  sitting at higher residual energy than an empty scene. So every atom count
  is an upper bound on cost and a lower bound on quality, and several
  published conclusions rest on that. One question — **fit a standard image
  and compare the rate-distortion curve against Image-GS's published points
  at equal bits-per-pixel; how many dB behind is our fitter?** That converts
  the confound from a caveat into a number. Recipe is public (Adam on
  L1 + 0.1·SSIM, ~5k steps, densification sampled from reconstruction
  error), and 95% of final quality lands inside 400 steps. PREREQUISITE:
  `tests/layering.rs` forbids the identifiers psnr/ssim/mse crate-wide; that
  guard must be NARROWED, not removed — its stated reason is comparison
  against a *path-traced* reference and says nothing about an exact raster
  ground truth, which is what rate-distortion needs. What it decides: if a
  real fitter closes the gap, aspect 2's "transforms win on reach" evaporates
  (that was explicitly a fitter result, not a representation one) and the P2
  atom counts fall; if it does not, every conclusion so far hardens instead.
  **MEASURED 2026-09-28 (BURNDOWN L-43), and the recipe as written cannot
  be run.** No image-gs number exists on an image we can obtain: its 45 2K
  images are behind an authenticated link, its CLIC2020 run is a figure
  only, and it needs CUDA. Across datasets, at image-gs's own float16
  billing on six 64×64 CLIC2020 crops, greedy scores 21.24 dB at 0.122 bpp
  and 24.48 dB at 0.366 bpp against its published 29.20 and 32.99: about
  **8 dB behind**, under two of its standard deviations. What remains is
  internal — does a better fitter close that on the same committed crops
  (`tests/fit_imagegs.rs`)? Record: `docs/lights/RESEARCH.json`
  `l43-greedy-at-imagegs-bitrates`.
- **Agora crate code** (design proceeds in Track 3 now).

---

## 9. Original Task/Improvement Numbering (traceability)

- Task 1 (evaluate optical features needed for atrium) → §3 + §5.
- Task 2 (test edit mode, all 3 layouts) → MVP slice only
  (drag/select/persist) in Track 2; full hardening parked (§8).
- Task 3 (port Graph algorithms to NodeEditor, retire the rest) →
  blocking list item 4.
- Imp 1 (remove radial/grid handoff points, use the spiral connector) →
  blocking list item 3.
- Imp 2 (radial lane wiring) → parked (§8).
- Imp 3/4 (rivers radial, then grid+spiral) → parked (§8).
- Imp 5 (wire animations on selection) → parked (§8).
- Imp 6 (elevation contours) → §2a, parked for implementation.
- Imp 7 (buffers/LOD research) → §8, research may run early.

---

## 10. Immediate Next Actions

1. Merge `wiring/spiral-unify-layout` (blocking item 1 — done+confirmed,
   just needs the merge).
2. Grid verification session (blocking item 2) — schedule per the working
   protocol.
3. Plan-file TOML format design (extends narration model) — can run as a
   design discussion in parallel with 1–2.
4. Acceptance gate for Track 2 (unchanged in substance): atrium starts
   <3s **on the 2014 MacBook Pro reference machine (repo principle:
   always note hardware)**; fulgor's own items appear as nodes; one
   function compiles via forge and executes via nexus from the GUI; pod
   state transitions visibly update.

---

## 11. Open Questions (need user ruling)

- **O-1 — RESOLVED (2026-07-06): fully serial stands.** User rationale:
  optical code does not need the review tool — the user reviews optical
  diffs comfortably in another editor. The review tool's value is
  concentrated on forge/nexus (student-mode work) and atrium, which come
  after the optical blocking list anyway. So "optical first" is
  consistent, not contradictory: the tool arrives exactly when the work
  that needs it begins.
- **O-2 — RESOLVED (2026-07-06):** commit trailer is
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` from now on.
- **O-3 — RESOLVED (2026-07-06): shell-out to `git` behind a `GitSource`
  trait** (data model: CommitInfo/FileDiff/Hunk; first impl = subprocess
  with `-z`/`--format` porcelain parsing). Rationale: fastest to build,
  exact git behavior for a developer tool on a machine that has git by
  definition, zero build deps; swap to git2/gitoxide later behind the
  same trait if profiling demands in-process performance (gitoxide is the
  long-term bet if aegis's incremental pipeline ever indexes thousands of
  commits in-process).
