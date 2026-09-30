import copy
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import research

GOOD = {
    "page": "https://example.invalid/x",
    "records": [
        {
            "slug": "p2-legibility",
            "title": "what crisp ui content costs in gaussian atoms",
            "group": "representation",
            "date": "2026-09-08",
            "verdict": "analytic beats splats on every crisp class.",
            "question": "how many atoms does a legible glyph cost?",
            "command": "cargo run --release -p lights --features fit --bin lights-p2 -- --max-atoms 48",
            "criterion": "10-90 transition below one device pixel, contrast at least 0.90.",
            "numbers": [{"label": "glyph, analytic", "value": "274 B", "note": "P2_RESULTS.md table"}],
            "findings": ["the analytic arm met every criterion."],
            "refuted": ["the premise that splats win on text."],
            "not_settled": ["every atom count rides on a greedy fitter."],
            "artefacts": ["docs/lights/P2_RESULTS.md"],
            "sources": ["docs/lights/P2_RESULTS.md", "git show 6ac92f33"],
        },
        {
            "slug": "transport-study",
            "title": "light transport on splats",
            "group": "transport",
            "date": "2026-09-07",
            "verdict": "the direction is gated on one experiment nobody has run.",
            "question": "can splats carry light transport?",
            "command": "NONE — this is a literature and reasoning study, not a run.",
            "criterion": "no numeric criterion; the gate is a named experiment.",
            "numbers": [{"label": "operator norm", "value": "0.01", "note": "TRANSPORT_RESEARCH.md"}],
            "findings": ["a splat has no normal and no area."],
            "refuted": ["the claim that splats-as-transport-nodes is novel."],
            "not_settled": ["the gate experiment has not been run."],
            "artefacts": ["docs/lights/TRANSPORT_RESEARCH.md"],
            "sources": ["docs/lights/TRANSPORT_RESEARCH.md"],
        },
    ],
    "audit": [
        {"kind": "shared-confound", "title": "the greedy fitter bounds every atom count",
         "experiments": ["p2-legibility"], "detail": "every count is an upper bound."},
    ],
    "arms": [
        {"arm": "kernel/gaussian", "verdict": "KEEP", "provenance": "3d-scene", "transfers": "yes",
         "records": ["p2-legibility"], "where": ["crates/lights/src/lib.rs"],
         "evidence": "the only kernel the certified path accepts."},
        {"arm": "kernel/density", "verdict": "PRUNE", "provenance": "target-free", "transfers": "yes",
         "records": [], "where": ["crates/lights/src/lib.rs"],
         "evidence": "an additive resolve has no occlusion."},
        {"arm": "kernel/carrier", "verdict": "RE-JUDGE", "provenance": "ui-crisp", "transfers": "no",
         "records": ["p2-legibility"], "where": ["crates/lights/src/lib.rs"],
         "evidence": "judged only on crisp UI content, by a search that could not tune it.",
         "experiment": {"question": "does it beat the gaussian on held-out views?",
                        "arms": "carrier against gaussian.", "criterion": "1.3x either way decides.",
                        "cost": "a build first.", "prunes": "the carrier, if it ties."}},
    ],
    "report": {
        "title": "what it set out to prove",
        "date": "2026-09-29",
        "tree": "abc12345",
        "sections": [{"key": k, "heading": "heading " + k,
                      "claims": [{"text": "claim for %s." % k, "cites": ["p2-legibility", "arm:kernel/gaussian"]}]}
                     for k in research.REPORT_SECTIONS],
        "reproduced": [{"record": "p2-legibility",
                        "invocation": "cargo run --release -p lights --features fit --bin lights-p2 -- --max-atoms 48",
                        "ran": None, "how": "full", "exit": 0, "expected": 0, "wall_s": 1.0,
                        "result": "identical."}],
    },
}


def broken(fn):
    d = copy.deepcopy(GOOD)
    fn(d)
    return d


def drop_field(d):
    del d["records"][0]["criterion"]


def bad_slug(d):
    d["records"][0]["slug"] = "P2 Legibility"


def dup_slug(d):
    d["records"][1]["slug"] = d["records"][0]["slug"]


def bad_group(d):
    d["records"][0]["group"] = "misc"


def future(d):
    d["records"][0]["date"] = "2099-01-01"


def malformed_date(d):
    d["records"][0]["date"] = "8 Sep"


def not_a_list(d):
    d["records"][0]["findings"] = "one finding"


def no_sources(d):
    d["records"][0]["sources"] = []


def no_binary(d):
    d["records"][0]["command"] = "ran it on the machine and looked at the pictures"


def truncated(d):
    d["records"][0]["verdict"] = "x" * 130 + " and then the sentence stops in the middle of a wo"


def number_without_note(d):
    d["records"][0]["numbers"][0]["note"] = ""


def bad_audit_kind(d):
    d["audit"][0]["kind"] = "interesting"


def audit_names_ghost(d):
    d["audit"][0]["experiments"] = ["an-experiment-that-does-not-exist"]


def empty(d):
    d["records"] = []


def arm_verdict(d):
    d["arms"][0]["verdict"] = "MAYBE"


def arm_provenance(d):
    d["arms"][0]["provenance"] = "vibes"


def arm_on_target_not_transferring(d):
    d["arms"][1]["transfers"] = "no"


def arm_ghost_record(d):
    d["arms"][0]["records"] = ["an-experiment-that-does-not-exist"]


def arm_absent_code(d):
    d["arms"][0]["where"] = ["crates/lights/src/no_such_arm.rs"]


def arm_rejudge_without_cost(d):
    d["arms"][2]["experiment"]["cost"] = ""


def arm_keep_with_experiment(d):
    d["arms"][0]["experiment"] = dict(d["arms"][2]["experiment"])


def arm_twice(d):
    d["arms"][1]["arm"] = d["arms"][0]["arm"]


def arm_without_evidence(d):
    d["arms"][0]["evidence"] = " "


def report_section_missing(d):
    d["report"]["sections"].pop()


def report_claim_cites_nothing(d):
    d["report"]["sections"][0]["claims"][0]["cites"] = []


def report_cites_ghost(d):
    d["report"]["sections"][0]["claims"][0]["cites"] = ["no-such-experiment"]


def report_invented_invocation(d):
    d["report"]["reproduced"][0]["invocation"] = "cargo run -p lights --bin lights-p9"


def report_exit_mismatch(d):
    d["report"]["reproduced"][0]["exit"] = 3


def report_unreproduced_citation(d):
    d["report"]["reproduced"] = []


def report_bad_how(d):
    d["report"]["reproduced"][0]["how"] = "vibes"


BREAKS = [
    ("missing field", drop_field, "is missing: criterion"),
    ("bad slug", bad_slug, "is not a slug"),
    ("duplicate slug", dup_slug, "duplicate slug"),
    ("unknown group", bad_group, "is not one of representation"),
    ("future date", future, "is in the future"),
    ("malformed date", malformed_date, "not YYYY-MM-DD"),
    ("list field is a string", not_a_list, "must be a list"),
    ("no sources", no_sources, "cites no source"),
    ("command names no binary", no_binary, "names no binary"),
    ("verdict stops mid-sentence", truncated, "stops mid-sentence"),
    ("number without a note", number_without_note, "is missing its note"),
    ("unknown audit kind", bad_audit_kind, "is not one of"),
    ("audit names a ghost", audit_names_ghost, "which is not an experiment"),
    ("empty record", empty, "holds no experiments"),
    ("arm verdict outside the three", arm_verdict, "verdict 'MAYBE' is not one of"),
    ("arm provenance unknown", arm_provenance, "provenance 'vibes' is not one of"),
    ("on-target evidence said not to transfer", arm_on_target_not_transferring, "transfers by definition"),
    ("arm cites a ghost", arm_ghost_record, "cites 'an-experiment-that-does-not-exist'"),
    ("arm names absent code", arm_absent_code, "no_such_arm.rs, which does not exist"),
    ("RE-JUDGE without its cost", arm_rejudge_without_cost, "experiment states no cost"),
    ("KEEP carrying an experiment", arm_keep_with_experiment, "only a RE-JUDGE waits on one"),
    ("arm listed twice", arm_twice, "listed twice"),
    ("arm without evidence", arm_without_evidence, "gives no evidence"),
    ("report section missing", report_section_missing, "report sections are"),
    ("report claim cites nothing", report_claim_cites_nothing, "cites nothing"),
    ("report cites a ghost", report_cites_ghost, "neither an experiment nor an arm"),
    ("report invents an invocation", report_invented_invocation, "its own command does not contain"),
    ("report run exited unexpectedly", report_exit_mismatch, "exited 3 where its record expects 0"),
    ("report cites an unreproduced record", report_unreproduced_citation, "records no reproduction"),
    ("report run of an unknown kind", report_bad_how, "is not one of full|form"),
]


def main():
    failures = []
    if research.check(GOOD):
        failures.append("the good record was refused: %s" % research.check(GOOD))
    if research.standing(GOOD) != (2, 1, 1):
        failures.append("standing miscounted: %s" % (research.standing(GOOD),))
    for name, fn, want in BREAKS:
        got = research.check(broken(fn))
        if not any(want in p for p in got):
            failures.append("BREAK NOT CAUGHT: %s (wanted %r, got %s)" % (name, want, got or "clean"))
    md = research.markdown(GOOD)
    for want in ("# lights — the research record", "p2-legibility", "```", "| measurement |",
                 "What it does NOT settle", "each other", "## The arm audit",
                 "| `kernel/carrier` | **RE-JUDGE** | ui-crisp | no |", "- **cost:** a build first.",
                 "## what it set out to prove", "### heading measured", "claim for stands.",
                 "### How each cited experiment was reproduced"):
        if want not in md:
            failures.append("markdown is missing %r" % want)
    page = research.report(GOOD)
    for want in ("<title>The lights research record</title>", 'id="split"', 'id="bar"', 'class="leaf',
                 'class="entry"', "localStorage", "col-resize", "prefers-color-scheme",
                 "no command on record", "copy-pasteable command", "arm audit",
                 'data-id="arm--kernel/density"', 'id="arm--kernel/density"',
                 "the experiment that would decide it", "final report", 'data-id="report--question"',
                 'id="report--reproduced"', "claim for open."):
        if want not in page:
            failures.append("page is missing %r" % want)
    missing = os.path.join(os.path.dirname(research.RECORD), "no-such-record.json")
    saved, research.RECORD = research.RECORD, missing
    buf = io.StringIO()
    stdout, sys.stdout = sys.stdout, buf
    code = research.main(["--check"])
    sys.stdout = stdout
    research.RECORD = saved
    if code != 1 or "is ABSENT" not in buf.getvalue():
        failures.append("an absent record was not refused: %r" % buf.getvalue().strip())
    if failures:
        print("\n".join(failures))
        return 1
    print("research record checker: %d deliberate breaks, all caught" % len(BREAKS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
