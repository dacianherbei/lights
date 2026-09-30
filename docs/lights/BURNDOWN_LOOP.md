# The lights burn-down loop — prompt for a fresh Claude Code context

You are finishing the `lights` crate, one register entry at a time. State lives ON DISK, not in your
memory: `crates/lights/BURNDOWN.toml` is the register, `docs/lights/burndown_history.json` is the day
log, git is the history, and the report page is what the PM reads. You start with no memory of
earlier iterations, and that is by design.

`lights` is a CPU-only anisotropic-atom renderer, and since 2026-09-29 it is an **experimental
proving ground for the fulgor/nexus paradigm** — interchangeable strategies, measured, with a
correctness contract between independent implementations. Its surviving renderer target is **a
simple 3D game engine**: the atom renderer no other crate offers. It has three interchangeable
axes — kernel (atom shape), method (image formation), precision — so that a champion can be
measured and the losers pruned.

**THE `optical` HALF OF L-120 IS WITHDRAWN (PM, 2026-09-29).** lights is NOT the renderer under
`optical` and may not touch optical's source. Three measurements retired it: L-174, egui's whole
CPU frame over eleven real optical tabs is 0.165–0.979 ms; L-175, the DrawList replaying the
identical op sets is 2.4–8.3× that on 18 threads against egui's one, 97.8–99.2% of it pixel
traffic rather than algorithm; L-176, tiny-skia draws the same op sets 2.2–10.2× faster from a
sixth of the memory with no pack. A port would have made optical slower on more cores for no
behaviour gain. Judge every entry against the 3D target and the paradigm case, never against a
UI consumer.

## Read first, in this order — and read nothing else until you have picked an entry

1. `python3 crates/lights/scripts/burndown.py --check` — schema and standing. It must print `clean`.
   It reads `head_open.toml` + `tail.toml`; if you ever need to see the split itself,
   `python3 crates/lights/scripts/split_register.py --check`.
2. `python3 crates/lights/scripts/burndown.py --next` — the entry to work. This function decides the
   order, not your reading of the file.
3. `git log --oneline -8` — what the previous context did, in its own words.
4. `tail -c 1200 docs/lights/burndown_history.json` — where the count stood.
5. `python3 crates/lights/scripts/burndown.py --clusters` — which entries share one cause.
5b. `python3 crates/lights/scripts/research.py --check` — the research record's standing. Any
   experiment you run or re-run is written into `docs/lights/RESEARCH.json` in the same commit.
6. Then, and only then, the entry's own `where` and `evidence`. Do not read the whole crate; 38k
   lines will not fit and the entry exists so that you do not have to.

## The order, every iteration

`--next` picks: **defects first**, then `owed`, `research`, `spec`, `feature`, `cleanup`, `walk`,
`decision`, and within a kind the lowest id. A defect outranks everything because a defect is a
measurement that is already wrong, and every number built on top of it inherits the error.

Never skip an entry because it looks large. Split it into smaller entries in `BURNDOWN.toml`, each
with its own `why`/`next`/`size`, and work the first piece.

**THE UNIT OF WORK IS THE CLUSTER, WHEN THERE IS ONE.** If the entry carries a `cluster` slug, run
`--clusters` and read every live member before touching code: one cause with four ids is fixed once,
and a fix written from one entry's `next` alone reproduces the other three.

## Measure the entry's own `why` before you build it

> PM, 2026-09-07: *watch out of issues description. They might no longer be compatible with the
> latest design so some of them need to be closed or rewritten or debated.*

An entry's `why` is the field you size the work from, and a sentence that WAS true reads exactly like
one that is. This register was built on 2026-09-09 by measuring the tree; anything written before
that, and anything carried over from `docs/plan/TWO_HORIZON_PLAN.md` §8, is a claim about a tree that
has since moved. Check the `evidence` field's `path:line` first — line numbers drift — and check the
claim, not just the line.

## The five answers. Say which one you chose, in the entry.

- **FIX / BUILD** — build it, to the close bar below. Flip `status = "closed"` and open `next` with
  `FIXED` or `BUILT`, the date, and the evidence.
- **MEASURE** — for a `research` entry. Run the experiment, write the number and the exact command
  that reproduces it into `next` opening with `MEASURED`, and then **prune**: close every entry the
  result kills, each citing the number that killed it. A measurement that prunes nothing is either
  not finished or was not worth running; say which. **AND WRITE IT INTO THE CONSOLIDATED RESEARCH
  DOCUMENT IN THE SAME COMMIT** (PM, 2026-09-09: *all research and experiments must be documented in
  one consolidated document*). Never start a new results file. The record is
  **`docs/lights/RESEARCH.json`** — edit that, never the markdown, then regenerate both forms:

      python3 crates/lights/scripts/research.py --check   # schema; it refuses a command that names no binary
      python3 crates/lights/scripts/research.py --md      # rewrites docs/lights/RESEARCH.md
      python3 crates/lights/scripts/burndown_host.py pages   # writes docs/lights/pages/research.html

  The published page is <https://claude.ai/code/artifact/aaf71a60-76fb-4964-aad3-a91a3e9be68f>; you do
  not publish it (see *The moment an entry is created or closed*), you regenerate the file and commit
  it. Per experiment the record carries the question, the
  exact reproducing command, the criterion AND its convention, a numbers table where every figure names
  where it came from, what it refuted, what it does not settle, and every source read. A finding that
  lives only in a commit message or a published page is not documented.

  **The command field is the one that gets lied to.** The record's own front page counts how many
  experiments have a command that can be pasted into a shell, and today that is ONE of seven (L-126).
  Write an invocation you have RUN, not one you reconstructed — a plausible command that was never
  executed is the same defect wearing a fix.
- **CLOSE** — no longer valid, superseded, or already built under another id. Say why in `next`
  opening with `CLOSED` or `DROPPED`, and cite the code or the ruling that replaced it. Never close
  on "it looks fine": run the instrument first.
- **REWRITE** — the title still names something real and the prose behind it does not. Correct `why`
  to the reading you measured today, and say what you withdrew and when it stopped being true. Never
  delete the old sentence silently — the withdrawal usually says more than the correction, because it
  says which way the tree moved. A rewrite is not a licence to weaken: the test is whether the new
  sentence is one you measured today.
- **BLOCKED** — it needs a PM ruling the design does not give (a semantics choice, two rulings in
  conflict, a scope call). Write the question, the options and your recommendation into `next`, leave
  it open, and move to the next entry. Do not stop the loop for it.

**A BUG FOUND WHILE SOLVING AN ENTRY IS SOLVED AS PART OF THAT ENTRY, NEVER FILED** (PM, 2026-09-09:
*if bugs appear during solving of an issues do not file them. just solve them as part of the issue*).
It gets its regression test and its measurement like anything else, and all of it is written into the
PARENT entry's `next` — withdrawing the filing is not withdrawing the fact. Raise a separate id only
for something genuinely outside the entry's reach: a PM ruling, or work that cannot be finished in
the sitting. Say why, in both places.

**COMMIT THE RE-WORK** (PM, 2026-09-09: *Commit this re-work so that it is not done again by future
iterations unless necessary*). A measured correction left uncommitted is re-derived by the next
context from the same stale sentence.

## The close bar — never lowered to move a count

- **A PERFORMANCE AND SIMPLICITY AUDIT of what you just landed** (PM, 2026-09-29). Answer each
  with a figure or an explicit *not measured*, never a judgement. PERFORMANCE: cost per pixel, per
  op and per frame at the size the feature is actually used at; allocations per call (a fresh plane
  per frame is this crate's recurring defect); does it add a pass over the framebuffer or fuse into
  one that exists; does it survive `tests/determinism.rs` bit-identity across thread counts; is it
  on the deadline path (6 P-cores at USER_INTERACTIVE) or background (12 E-cores). SIMPLICITY:
  public items added; can an existing primitive already express it; does it add a type parameter, a
  trait, a feature gate or a dependency, each of which is permanent; lines added against lines it
  lets you delete. **If the audit finds a question that needs an EXPERIMENT, state it and stop** —
  the question, the arms, the criterion and its convention, the wall-clock cost, and what it would
  prune — and leave it for the PM. An experiment already written into a filed `research` entry
  proceeds as normal; this gate is for the ones an audit newly discovers.

- A regression test in `crates/lights/tests/` that goes **RED on the unfixed tree**. Say which test
  and that you saw it red; a test written after the fix and never seen red is not evidence.
- A mutation in `crates/lights/scripts/mutations.json` proving that test can fail, run through
  `python3 crates/lights/scripts/mutate_par.py` and passing all four gates — applied, compiled, ran,
  detected. A mutation that fails to compile tested nothing.
- **`bash crates/lights/scripts/run_tests.sh` green.** That is the one documented command: it runs
  `--all-features`, sums the per-target result lines itself, prints the log path it read, and then
  builds AND RUNS each method feature alone. Never `cargo test` bare — `fit` is not a default
  feature, so plain `cargo test` silently skips 157 of 409 tests, the entire fit arm. PRINT both
  totals from the run you just did; never carry a number forward from another context. Build once
  per context — see *Build once per context* below.
  **When the session refuses the script** — a driven context has been refused `bash run_tests.sh`
  and even `bash -n` on it, while every cargo command it issues runs (L-152, PM ruling 2026-09-16,
  option (b)) — the bar is met by its constituents, each run and printed on its own, from the repo
  root, with `CARGO_TARGET_DIR` exported to whatever `python3 crates/lights/scripts/target_dir.py
  --explain` prints (unset it if that prints nothing):

      cargo test -p lights --no-default-features --features method-splat
      cargo test -p lights --no-default-features --features method-raycast
      cargo build -p lights --no-default-features --features fit
      cargo test -p lights --all-features
      cargo test -p lights-mac

  Sum the `test result:` lines of the last two yourself and print targets and passed, and SAY in
  the entry that the constituents stood in for the script. The two feature-alone runs are the last
  thing the context does (they switch feature sets). `tests/loop_prompt.rs` asserts this list
  equals the commands `run_tests.sh` issues, in both directions, so the fallback cannot drift below
  the script.
- `python3 crates/lights/scripts/burndown.py --check` clean and
  `python3 crates/lights/scripts/burndown_selftest.py` green. If you touched the driver or its
  parser, `python3 crates/lights/scripts/loop_usage_selftest.py` too — it is what stands between the
  loop and never stopping.
- No comments in generated code. `tests/layering.rs::no_comments_in_src` allows at most one line
  per file. In the published package the sibling prose folders this rule used to point at have been
  removed, so the permitted count is zero.
- Any number you publish carries the exact command that reproduces it from a clean checkout.
- Say plainly what you did NOT verify.

## Choosing direction — this is the part that is yours to decide

When several entries are equally old, prefer, in this order:

1. the one that can **falsify something already published**. Every atom count in `docs/lights/` rests
   on a greedy fitter that is a known confound; a result that would overturn a conclusion is worth
   more than one that would decorate it;
2. the **instrument before the feature it measures** — this crate's recurring lesson is that
   instruments lie, and four of them have reported success while measuring nothing;
3. the one that **unblocks the most other entries** (`--clusters` shows the sharing);
4. the cheapest to measure, last.

**One axis at a time.** Finish one axis — kernel, or method, or fitter — before starting the next.
A matrix of combinations is what this project deliberately stopped building.

## Measurement rules paid for here — read these before you time anything

- Single-thread wall time on this machine has a **2.4× spread** over identical calls as the scheduler
  moves work between P and E cores. Never publish a single-run time. Five reps minimum, report the
  spread, and treat a 1.3× difference as noise.
- The fit **job sweep** is the parallel win: **6.01×–7.00×** end to end against an **8.83×**
  critical-path ceiling. Tile parallelism *inside* a fit is a **1.7×–4.3× slowdown** at the 32×32 to
  64×64 viewports the fitter works at; `--threads 1` is correct for fit experiments.
  **Both figures were wrong here until 2026-09-15 (L-64)** — this bar said 5.7–7.0× and 2–11×, and
  neither number appears in any tree. The ones above are the consolidated record's, with their
  sources: `docs/lights/RESEARCH.md` rows for sweep speedup (7.00× on lights-p2, 6.01× on lights-p4,
  6.59× on the 32 aspect-2 jobs) and for tile parallelism (0.23×–0.58× of serial). Quote the record,
  never this prompt, and `tests/loop_prompt.rs` now asserts the two agree.
- The cross-method agreement harness **refuses** rather than compares when its preconditions are
  violated (exit 3). A refusal is a result. Do not relax a precondition to get a comparison.
- `crates/lights/tests/layering.rs` forbids the identifiers `psnr`/`ssim`/`mse` crate-wide. Narrowing
  that guard is its own entry with its own reasoning — never a side effect of another one.
- Builds go to the RAM disk (`/Volumes/RamBuild`), so `cargo clean` cleans that and not `target/`.

## BUILD ONCE PER CONTEXT — measured 2026-09-09, and it is the largest cost in this loop

Fifteen agents inventorying this crate spent **3,931 seconds inside 101 cargo calls** — 77% of all
their tool time in 10% of their calls — against 1,185 s across 936 other shell calls. One agent alone
spent 36 of its 43 minutes inside eight `cargo test` invocations. A cold
`cargo test -p lights --all-features` takes **416 s** on this machine.

The cause was not the crate and not the lock (exactly one `Blocking waiting for file lock` line in
fifteen transcripts). It was **feature switching**: 92 invocations across FIVE build graphs —
`--all-features` (54), release (16), default features (9), `--features fit` + release (9),
`--features fit` (4). `--all-features` turns on `fit`, which the default set leaves off, so every
alternation relinks every test binary.

**THE CACHE IS NOT IN THE CONTEXT — IT IS ON THE RAM DISK.** `~/.cargo/config.toml` sets
`build.target-dir` globally to `/Volumes/RamBuild/cargo-target`, so the build cache survives a
session boundary intact. Ending a context and starting a fresh one costs you nothing in build time;
what costs 416 s is a *cold* target directory, which happens after a reboot clears the RAM disk, and
what costs a relink is *switching feature sets*. Read the rule as **one feature set per RUN**, not
one feature set per context. (`scripts/mutate_par.py` is unaffected either way: each worker carries
its own `CARGO_TARGET_DIR`.)

So, first thing in the context, once:

    cargo test -p lights --all-features --no-run 2>&1 | tail -5

and from then on **stay on that one feature set for the whole context**. Run a single test through
the already-built binary rather than through cargo:

    ls -t /Volumes/RamBuild/cargo-target/debug/deps/aspect3-* | head -1     # then run it directly
    cargo test -p lights --all-features --test aspect3 -- <name> --nocapture   # only if you must

Do not add `--release`, do not drop to `--features fit`, and do not build a private mutation pool:
`scripts/mutate_par.py` already keeps one worktree per worker with its own `CARGO_TARGET_DIR`, which
is the only sanctioned way to compile the crate more than once at a time. If an entry genuinely needs
a second feature set, that is the LAST thing the context does, after everything else is committed.

## THE DETACHABILITY CHECK RUNS BOTH DIRECTIONS, AND IT MUST RUN THE TESTS

`run_tests.sh` ends by building and running `--no-default-features --features method-splat` and the
same for `method-raycast`. Two things about it were paid for on 2026-09-14 and must not be undone:

- **`--no-run` cannot see this class of failure.** `absent.rs` is a *compiling* stub: a method that
  is not built still type-checks and returns `MethodNotImplemented` **at runtime**. A test hardcoded
  to the absent method compiles perfectly and fails when run, so the check must RUN the tests. An
  earlier `--no-run` version reported "every listed build is green" against a genuinely broken tree.
- **Running only one direction is how a regression shipped.** The goal is symmetric; the check has
  to be.

When you add a test: if it is genuinely about one method, carry `#[cfg(feature = "method-splat")]`
and say so. If it is about something else, walk `RenderMethod::implemented()` **with a guard that
fails when the list is empty**, so it cannot pass by testing nothing. A vacuity guard hardcoded to
`>= 2` is wrong — it cannot hold when one method is built; assert coverage *equals*
`implemented().len()` and is at least 1.

## THE MUTATION MANIFEST HIDES ROWS THAT RUN NOTHING — read the gate marks

`scripts/mutations.json` is the only proof a test can fail, and on 2026-09-14 a full re-run found
**eleven rows that ran nothing while looking present**. Three shapes, all of which you can create by
accident:

- **A stale anchor.** `before` must match the file **exactly once**. Check with
  `grep -ocF "<before>" <file>` before you commit a row — an anchor matching **zero** lines or
  **six** is refused, and the refusal reads like a driver bug rather than a stale row. Any fix you
  make can move someone else's anchor; re-run the manifest after a sweep.
- **A malformed `test` field.** It is passed to `cargo test -- --exact`, so it must be the **bare
  function name**. `tests/foo.rs::name` matches nothing and reports `..xx` — applied, compiled,
  **DID NOT RUN**, not detected. Read the four gate marks; `....` is the only clean one.
- **A row that is not a source mutation.** A `before` written as prose, or a `test` that is a shell
  command, cannot be applied. Do not add one; delete it if you find one.

**THE RESULT LIVES ON DISK AND A PARTIAL RUN WILL WIPE IT.**
`docs/lights/mutations/results.tsv` is the committed record, and
`tests/mutation_record.rs` asserts that EVERY row of the manifest appears there scored CAUGHT
through all four gates. A `MUT_ONLY` run **replaces** that file with only the rows it selected, so
every other row becomes `NOT-RUN` and the suite goes red — under the detachability build, not under
`--all-features`, so you find out late. **Always pass `MUT_MERGE=1` with `MUT_ONLY`**, which merges
the new verdicts into the existing record instead of replacing it. It cost this loop a round trip on
2026-09-15. If you have already wiped it, `git show HEAD:docs/lights/mutations/results.tsv >
docs/lights/mutations/results.tsv` restores the committed one; then re-run your rows with
`MUT_MERGE=1`.

`python3 scripts/mutate_par.py <profile> <workers> <pool>` re-runs everything. **Its fourth argument
is a PREFIX LIMIT (`mutations[:n]`), not a row index** — to re-run a few rows, filter the manifest to
those rows, run it, and restore the full file afterwards. Filter by the `test` name, not by `file`:
several rows can share a file and the count assertion then trips (it did, 2026-09-15) and the run
starts on the FULL manifest — kill it by pid and delete the pool. `dev` is the right profile for a
two-row check: 2 workers, ~30 s.

**WHEN A MUTATION ESCAPES, CLOSE THE HOLE OR REPLACE THE ROW — NEVER KEEP IT GREEN WITH A CONTRIVED
ASSERTION.** Two escaped on 2026-09-15 and they resolved in opposite directions, which is the point:

- One was a REAL hole. The test asserted colours by calling the sampler function directly, so the
  `Sampler::SoftShadow` dispatch arm was never exercised and a mutation routing the class to n1's
  colour field passed. Fixed by asserting the CLASS's own truth carries n2's ink.
- One was NOT a hole, and saying so was the honest answer. Letting `Renderer::set_scene` skip
  `strategy.build` is undetectable today, because both methods' `build` only resets state that
  `bind` rewrites every frame and the empty-scene refusal is already done one layer down by
  `PreparedScene::build`. The row was REPLACED with one that mutates `set_scene` into keeping the
  old scene — a real defect, caught — and the finding was written into the comment folder, because a
  future method whose `build` carries state would break silently.

**AND CHECK THAT YOUR TEST MEASURES THE QUANTITY IT NAMES.** L-42's first version scanned
darkest-to-brightest across a whole row, which spanned the shadow core AND the card, and reported
ratios of 2.07 and 4.16 — passing beautifully, and it would have passed with the exact defect it was
written to catch, because the quantity it actually measured was dominated by something else. A test
whose numbers look right is not evidence that it is pointed at the right thing.

## THE MOMENT an entry is created or closed — before anything else

Not a step at the end of the iteration. At the instant `BURNDOWN.toml` changes state, every time,
including for an entry you created on the way to another:

1. `python3 crates/lights/scripts/burndown_host.py pages` — it regenerates the PM's pages to fixed
   files, `docs/lights/pages/burndown.html` and `docs/lights/pages/research.html`, and prints each
   path it wrote. Commit them with the entry and NAME THE FILE in the entry's `next`. **You do not
   publish.** A `claude -p` context has no artifact tool (L-148, PM ruling 2026-09-16, option (b)):
   a separate interactive session publishes the files in one pass to the report
   **<https://claude.ai/code/artifact/7f564802-833c-4be0-8acb-e954ca6a4eda>** and the research page,
   and a page that has not been published yet is visible as a diff between the file and the page.
   The host also regenerates both after every context and commits them with its roll.
2. `python3 crates/lights/scripts/burndown.py --history "L-nn closed: what moved"`.
3. Commit **by path** — never `git add -A`. One commit per entry, the message naming the id and the
   answer (FIXED / BUILT / MEASURED / CLOSED / REWRITTEN / SPLIT / BLOCKED).
   **DO NOT PUSH** — PM, 2026-09-15: the git server is down. Commit locally and say so.
4. If the entry changed anything the renderer can DRAW, re-run the demo and regenerate its page to
   `docs/lights/pages/demo.html` — see *A feature is demoed as it lands* below. The host does not do
   this one for you: it needs a render of the demo binary.

A closed entry whose pages were not regenerated and named is not closed.

## A FEATURE IS DEMOED AS IT LANDS — standing, PM 2026-09-15

> *Focus on getting a visual demo running to which features are demoed as they get implemented.*

The demo exists and it is the crate's front door:
**<https://claude.ai/code/artifact/640f8902-42c8-4414-a41f-0b5f80a81ff4>**. Its unit is a PANEL, one
per feature, so landing a feature means adding a panel — and a feature that cannot be shown is
visibly missing from the page.

    cargo run -p lights --all-features --bin lights-demo -- \
      --out-dir <DIR> --width 640 --height 360 --frames 32 --reps 5
    python3 crates/lights/scripts/demo_page.py <DIR> > docs/lights/pages/demo.html
    # commit it and name it; the interactive session publishes it to the URL above

`src/bin/lights_demo.rs` renders the panels and writes `demo.tsv` (the invocation on its first line,
one row per panel and per orbit frame, every deterministic counter beside every wall clock);
`scripts/demo_page.py` embeds the PNGs as data URIs and emits one self-contained page in the same
tree-left/content-right shape as the register report.

Three things the demo has already had to learn:

- **Walk `RenderMethod::implemented()`, never a hardcoded list.** `run_tests.sh` builds each method
  alone, and `tests/binaries.rs` asserts the panel count EQUALS that list's length and that the list
  is non-empty. A group that only one method can serve — the prefilter, which the ray caster refuses
  — is rendered through that method and is ABSENT under a build without it, rather than quietly wrong.
- **A panel that shows an artifact must caption it.** The `damage` group deliberately shows a seam,
  because the rule it exists to teach is that a damaged rect must cover the union of the old and new
  footprints. An uncaptioned artifact reads as a bug.
- **The page's playback rate is not the renderer's frame rate.** The orbit plays at a rate the page
  chooses; the measured milliseconds are in the panel's own table, and the two must never be
  confused.

## How the record is read: tree left, content right, draggable splitter

PM, 2026-09-09: *the list of these issues should be listed in a tree if they need to be grouped on the
left side and content should shown on the right side while being able to resize the splitter between
them.*

The generated report already does this and you get it for free — the tree groups by area and, inside
an area, by `cluster` where entries share one cause, so **giving an entry a cluster is what makes the
grouping appear**. Closed entries fold into their own node. The same layout is required of the
consolidated research document (L-125). If you add a view of this register anywhere else, it takes the
same shape.

## WHAT A CONTEXT COSTS, AND THE TWO FILES THAT WOULD BLOW IT

Measured 2026-09-15 with the session-cost harness (not carried in this repository) and `wc -c`.
Loaded whether you want it or not:

| paid every session | bytes | approx tokens |
|---|---|---|
| `CLAUDE.md` | 11,894 | ~2,973 |
| this file, pasted whole as the `-p` prompt by the driver | 28,413 | ~7,103 |

**NEVER READ THESE WHOLE. That is the single most expensive mistake available to you:**

| never `cat` this | bytes | approx tokens |
|---|---|---|
| `crates/lights/tail.toml` | ~488,000 | **~122,000** |
| `docs/lights/RESEARCH.md` | 286,407 | **~71,601** |

**THE REGISTER IS TWO FILES NOW (2026-09-15).** `crates/lights/head_open.toml` is the CURRENT
ITERATION — ten open entries, ~29 KB — and it is the one you read and edit. `crates/lights/tail.toml`
holds the queue that is not scheduled yet plus every closed entry, and **nothing should ever open
it**: `burndown.py` loads both for you, so `--check`, `--next`, `--clusters` and `--history` all see
the whole register exactly as before. `scripts/split_register.py --roll` retires the entries you
closed and promotes the same number from the queue; the driver runs it before every context, so head
is always one iteration. The generated report renders head only and stops after the open list.

`burndown.py --next` prints ONE entry and `research.py --check` prints three numbers, which is
exactly why the loop has never paid those 195k tokens. Reach into either file with a targeted `grep`
or an `awk` range and never with `cat`, `Read` without a range, or a subagent told to "read the
register". The same applies to `docs/lights/mutations/results.tsv` (342 rows) — count it, do not
print it.

Two MCP servers are configured for this repository in `.mcp.json`, `claude-flow` and `ruv-swarm`.
**Nothing under `crates/`, `docs/lights/` or `scripts/` references either** (verified by grep,
2026-09-15), and `ruv-swarm` failed to connect in the session that measured this. Their tool names
are in every system prompt regardless. Removing them is the PM's call, not a context's, so this is
recorded rather than acted on — see L-137.

## HOW THIS LOOP IS DRIVEN, AND WHAT IS NOT YET MEASURED ABOUT IT

Two drivers exist and they answer different questions.

    python3 crates/lights/scripts/burndown_host.py run --budget 1     # what runs the loop

The HOST owns the budget and substitutes it into this document's budget marker before you see
it, and it fills `head_open.toml` with exactly that many WORKABLE entries first — so the head IS your
budget and a blocked entry never reaches you. It heartbeats every five minutes, ends a context with a
wall that reaps the whole process group, recovers the usage from the transcript when the result JSON
is lost, and writes one row per context to `docs/lights/benchmarks/loop_usage.json` naming which
entries moved and what was committed. `burndown_host.py report` compares budgets on real work;
`inspect` says what a context pays before it acts, and
`python3 crates/lights/scripts/burndown_host_selftest.py` proves the parts that decide whether it
stops. The observing half — heartbeat, process-group reaping, transcript usage recovery — lives in
`crates/lights/scripts/burndown_watch.py`.

The older shell driver is kept and still works, but it hard-codes a budget of ten in prose, measures
a context only if it finishes cleanly, cannot end a grandchild, and prints nothing for the 40-70
minutes one is running:

    bash crates/lights/scripts/burndown_loop.sh          # superseded, still valid

starts a fresh `claude -p` per context, ten entries each, and stops on `REGISTER DRY`. Since
2026-09-15 (L-139) it also **records what each context cost**: `--output-format json`, the raw
document kept per context under `docs/lights/benchmarks/loop/`, and one row per context appended to
`docs/lights/benchmarks/loop_cost.tsv` carrying turns, the four token columns, cost, wall seconds and
the register's open/total afterwards. Three things about it are load-bearing and are explained in
`crates/lights/scripts/burndown_loop_sh_comment/`: stdout and stderr are captured SEPARATELY (one
warning merged into the JSON makes it unparseable); the stop signal is an exit code from
`crates/lights/scripts/loop_usage.py` rather than a grep, and it matches `REGISTER DRY` as a WHOLE
LINE so that a context merely mentioning the sentinel does not stop the run; and the driver drops
`ANTHROPIC_API_KEY` from the child environment, without which every context bills to an API account
and returns `Credit balance is too low`. `python3 crates/lights/scripts/loop_usage_selftest.py`
proves the stop signal, fourteen cases, and it was validated by breaking the parser three ways.

    (the session-cost harness is local automation of the monorepo and is not carried in this
     repository; crates/lights/scripts/burndown_tasks.json, the task list it read, is here)

is the instrumented host: one task per session, a model fixed for the whole run, and the usage the
`json` output reports collected per task. Its `verify` runs the same sequence twice — a fresh session
per task against one session resumed — and prints the measured difference.

**MEASURED 2026-09-15, AND ISOLATION LOST (L-138).** Five read-only tasks under sonnet: a fresh
session per task cost **$1.209**, one resumed session cost **$0.458** — isolation was **2.64x more
expensive** and 30.8% slower. The mechanism is that a session start re-writes about **40,000
cache-creation tokens** before any work happens, and a cache WRITE is roughly an order of magnitude
dearer per token than a cache READ. The accumulation isolation avoids is real and visible (+31.9%
cache reads) and it is simply the smaller term at this scale. The full record is the
`session-isolation-cost` experiment in `docs/lights/RESEARCH.json`.

**So do not end a context to save tokens. It costs more, not less.** End one when the context is
genuinely stale or exhausted, which is a different reason and still a good one.

Two things that measurement does NOT settle, and neither may be assumed away. Isolation must win
eventually, because cache-read cost grows with accumulated context while the ~40k floor is constant;
those five tasks peaked at 770,840 cache reads and persistence still won, so the crossover is above
that and was **not** found. And the proxy was five short READS — the real loop edits, runs cargo,
commits and regenerates pages, so a ten-entry context is far heavier and may sit on the other side. **The
budget of ten entries below therefore remains a working assumption.** `burndown_tasks.json` is
deliberately READ-ONLY so `verify`'s two phases start from the same tree; measuring the write half
needs a disposable git worktree per phase and has not been done.

Whatever you run, write it into `docs/lights/RESEARCH.json` like any other measurement, and correct
the budget only if a number says to. One check first, learned the hard way: `ANTHROPIC_API_KEY` is
set in this shell and the host copies the environment to the child, so **without `--unset-env
ANTHROPIC_API_KEY` every spawned session bills to an API account and returns `Credit balance is too
low` with zero usage** — a run of five clean-looking failures and a $0.00 total. Smoke-test one task
before spending ten.

## NO PAUSES, NO QUESTIONS

- Go from one entry straight to the next in the same turn. Do not schedule a wake-up, do not wait for
  an interval, do not stop to summarise between entries.
- Never ask the PM anything mid-loop. A decision only the PM can make is a BLOCKED entry.
- Waiting on a long build or a mutation run is done by polling its log in a bounded loop, never by
  ending the turn.

## Rules that have destroyed work here

- Never run state-changing git against the PM's other repositories, and never touch the worktrees
  under `/Volumes/RamBuild/mutpool*` while a mutation run is using them.
- Kill only processes you started, by pid.
- Never edit `crates/lights/scripts/mutations.json` with a pattern that can span entries; assert the
  count after any bulk edit.
- A green run may be a rescue. If two runs disagree, report the one that disagrees.
- `grep -c` counts lines, not occurrences. Use `grep -o | wc -l`.
- **THE WORD `while` OR `loop` INSIDE A PROSE STRING TRIPS A GUARD.**
  `tests/layering.rs::bounded_loops_carry_a_budget` matches WORDS, not syntax, and demands a
  `Budget` within three lines. It cost this loop three round trips on 2026-09-15 — twice in panel
  text, once in a `println!` that said `warm loop`. Write *and*, *whereas*, *cycle*, *path*.
  **It only fires outside the default feature set**, so `--all-features` stays green and only the
  detachability run catches it: if you wrote prose in a string, run
  `cargo test -p lights --no-default-features --features method-splat --test layering` before the
  full suite and save yourself the round trip.
- **`_ =>` is banned in ALL of `src/bin/`**, not only in `src/dispatch.rs` and `src/method/mod.rs`.
  `no_wildcard_match_arm_in_dispatch_or_forwarding` scans the whole binary directory. Enumerate the
  arms, or replace the match with arithmetic.

## WHERE THE WORK STANDS, 2026-09-15 (third sitting) — so you do not re-derive it

**OPEN DEFECTS ARE ZERO.** 36 -> 1 -> 0 over three sittings. L-20, the last one, is closed: the ray
caster no longer bounds an atom by a sphere, it bins the EXACT SILHOUETTE CONIC of the support
ellipsoid, computed in the new `src/extent.rs` which sits below both renderers (the layering guard
forbids raycast importing splat's EWA conic, and that is why the function lives at crate root).
Tile pairs went 109,188 -> 46,717 against splat's 46,704 on the same scene, so the 2.15x this
register was built around is now 1.0003x — **the two methods' tile-pair columns are comparable for
the first time**, and `lights_bench.rs`'s footer saying they are not was stale and is corrected.

Eight more closed in the same sitting, and what they add up to is that **lights can now be driven**:

- **L-38** — `render()` hands out premultiplied colour and coverage under `Collect::PixelsAndCoverage`,
  and `FloatImage::composite_over` lays a render over a canvas it never saw. The identity
  `resolve(acc, bg) == resolve(acc, 0) + bg*(1 - coverage)` holds for ALL THREE blend modes.
- **L-40** — a scene that CHANGES: `Quat::mul/conjugate/inverse/rotate`, and
  `Scene::set/remove/clear` through one shared private `ingest`, so there is exactly one path that
  clamps the floor, normalises the rotation and validates. NOT `atoms_mut`, deliberately.
- **L-41** — a damaged rect is a CAMERA (`CameraSpec::sub_rect` shifts the principal point), not a
  clip, and `FloatImage::blit_from` puts it back. Exact for ray-cast, one ULP for splat.
- **L-129** — `lights::Renderer<T, K>` is the warm loop: it holds strategy, prepared scene and
  framebuffer. `dispatch::render()` is UNCHANGED and bit-identical to it. 1.23x at 1920x1080 f32.
- **L-39** — the framebuffer states its size, and it is 93-99% of the renderer's resident memory at
  1080p (63.75 MiB at f64) — a number the bench's MiB column did not contain before.
- **L-37, L-42** — the prefilter backlog item was wrong three ways (there is no object-space term to
  decompose at all), and the soft shadow can now enter the zoom ladder as an analytic sampler.
- **L-128** — the demo, which is the crate's front door now. See the demo section above.

What is open is **owed**, **spec**, **research** and **decision** work. The `gap` area is still the
largest and it is where the two targets meet. `--next` gives **L-43** (no gradient fitter; its own
`next` says the prerequisite is NARROWING the crate-wide psnr/ssim/mse ban, which is its own entry).
**L-123** — no frame rate at 1080p — is now cheap: the demo already measures one at 480x270 and
prints it; the only thing missing is a release build, and the prompt's build-once rule says that is
the LAST thing a context does.

**Three published-number corrections are live.** Two are in `docs/lights/P2_RESULTS.md` (the analytic
SDF arm painted coverage from a distance *lower bound*; every `render_ns` in P2 was measured on 18
cores under a header that said `threads 1`). The third is in `docs/lights/P5_RESULTS.md` §C3 and in
`RESEARCH.json`'s audit 8: aspect 3 never rendered at `screen_dilation = 0.3` — it ran at
`PIXEL_BOX_VARIANCE = 1/12` — and the blur it measured cannot be "the object half" because the crate
has no object-space term. Do not quote those columns without reading the corrections.

**The PM said the git server is down: commit, do not push.**

## The budget of this context: {{BUDGET}} entries

**The host sets that number, not you, and it is substituted into this line before you ever see it.**
`crates/lights/scripts/burndown_host.py` fills `head_open.toml` with exactly that many workable
entries before starting you, so the head IS your budget: when head is empty you are done, and you do
not need to count. A blocked entry never enters head, so you will not be handed work you cannot do.

Count every entry you CLOSE, FIX, MEASURE, REWRITE or CREATE. When the count reaches the number
above, finish the entry in hand — its commit, its report regeneration — write a four-line handover as
the last commit message body (what closed, what is mid-flight and where, what the next context should
pick up), and STOP. Do not start one more.

The host restarts you with a clean context and a refilled head. End with exactly this line, which is
the word the host reads to know you finished rather than died:
`BUDGET SPENT — run /clear, then paste the start prompt again`

If no open entry remains that is not BLOCKED, print `REGISTER DRY` on its own line and stop.
