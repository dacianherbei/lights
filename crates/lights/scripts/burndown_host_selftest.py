import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import burndown                                    # noqa: E402
import burndown_host as H                          # noqa: E402
import split_register as S                         # noqa: E402

ENTRY = ('[[entry]]\nid = "L-%d"\ntitle = "t%d"\narea = "core"\nkind = "%s"\nstate = "%s"\n'
         'status = "%s"\nraised = "2026-09-01"\nsize = "small"\nwhere = "src/x.rs"\n'
         'why = "w."\nnext = "%s"\n')


def register(n_open, blocked=(), closed=()):
    out = []
    for i in range(1, n_open + 1):
        st = "closed" if i in closed else "open"
        out.append(ENTRY % (i, i, "owed", "blocked" if i in blocked else "absent", st,
                            "BUILT 2026-09-01 - done." if st == "closed" else "do it."))
    return "# synthetic\n\n" + "\n".join(out)


def main():
    fails = []

    def want(name, got, exp):
        if got != exp:
            fails.append("%s: expected %r, got %r" % (name, exp, got))

    # ---- classify: the two cases a stop-word reader alone gets wrong ----------
    want("a wall is ABANDONED", H.classify({"parsed": True, "result": ""}, None), "ABANDONED")
    want("unparsed with nothing recovered is ERROR",
         H.classify({"parsed": False, "usage_source": "unmeasured"}, 0), "ERROR")
    want("unparsed but recovered from the transcript is ABANDONED, not ERROR",
         H.classify({"parsed": False, "usage_source": "transcript"}, 0), "ABANDONED")
    want("is_error is ERROR", H.classify({"parsed": True, "result": "x", "is_error": True}, 0), "ERROR")
    want("the sentinel on its own line", H.classify({"parsed": True, "result": "a\nREGISTER DRY"}, 0),
         "REGISTER DRY")
    want("a prose mention is NOT dry",
         H.classify({"parsed": True, "result": "I did not print REGISTER DRY here"}, 0), "OTHER")
    want("budget spent", H.classify({"parsed": True, "result": "BUDGET SPENT - run /clear"}, 0),
         "BUDGET SPENT")

    # ---- exhausted: a credit wall is not a register result -------------------
    want("429 is exhausted", H.exhausted({"api_error_status": 429, "result": ""}), True)
    want("the words alone are enough", H.exhausted({"result": "Credit balance is too low"}), True)
    want("a 500 is not exhausted", H.exhausted({"api_error_status": 500, "result": "boom"}), False)

    # ---- the budget marker is the host's contract with the prompt ------------
    tmp = tempfile.mkdtemp()
    good = os.path.join(tmp, "p.md")
    open(good, "w").write("budget is {{BUDGET}} entries\n")
    want("one marker substitutes", H.prompt_text(good, 7), "budget is 7 entries\n")
    for name, text in (("no marker", "no marker here\n"),
                       ("two markers", "{{BUDGET}} and {{BUDGET}}\n")):
        p = os.path.join(tmp, name.replace(" ", "_") + ".md")
        open(p, "w").write(text)
        try:
            H.prompt_text(p, 1)
            fails.append("%s: the host accepted a prompt it must refuse" % name)
        except SystemExit:
            pass

    # ---- scrub_env: BOTH credentials, and the wall is passed through ---------
    env = {"ANTHROPIC_API_KEY": "k", "ANTHROPIC_AUTH_TOKEN": "t", "PATH": "/bin"}
    want("both credentials are dropped", sorted(H.scrub_env(env)), ["PATH"])
    want("--keep-api-key keeps them", sorted(H.scrub_env(env, True)),
         ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "PATH"])
    want("the background wait ceiling is set from the wall",
         H.child_env(env, False, 30)[H.BG_WAIT_VAR], "30000")

    # ---- what_moved names entries, which a count cannot ---------------------
    before = {"L-1": {"status": "open", "kind": "owed", "state": "absent", "title": "a"}}
    after = {"L-1": {"status": "closed", "kind": "owed", "state": "built", "title": "a"},
             "L-2": {"status": "open", "kind": "owed", "state": "absent", "title": "b"}}
    m = H.what_moved(before, after)
    want("closed is named", m["closed"], ["L-1"])
    want("raised is named", m["raised"], ["L-2"])
    want("a net-zero sitting is still visible", (len(m["closed"]), len(m["raised"])), (1, 1))
    m2 = H.what_moved({"L-1": {"status": "closed", "kind": "owed", "state": "built", "title": "a"}},
                      {"L-1": {"status": "open", "kind": "owed", "state": "built", "title": "a"}})
    want("a reopen is not counted as a close", (m2["closed"], m2["reopened"]), ([], ["L-1"]))

    # ---- read_result on the shapes a run actually produces -------------------
    want("garbage never reads as measured", H.read_result("not json")["usage_source"], "unmeasured")
    doc = {"usage": {"input_tokens": 2, "cache_read_input_tokens": 5}, "result": "ok",
           "num_turns": 3, "session_id": "s", "total_cost_usd": 1.5, "is_error": False}
    r = H.read_result(json.dumps(doc))
    want("usage is read from one level down", (r["input_tokens"], r["cache_read_input_tokens"]), (2, 5))
    want("a real result is sourced to the result", r["usage_source"], "result")
    want("a zero-usage document is not 'measured'",
         H.read_result(json.dumps({"usage": {}, "result": "x"}))["usage_source"], "unmeasured")

    # ---- reconcile: budget, served, blocked, and byte preservation ----------
    tree = tempfile.mkdtemp()
    os.makedirs(os.path.join(tree, "crates", "lights"))
    hp = os.path.join(tree, "crates", "lights", "head_open.toml")
    tp = os.path.join(tree, "crates", "lights", "tail.toml")
    open(hp, "w").write("# head\n")
    open(tp, "w").write(register(8, blocked=(2, 3)))
    saved = (S.HEAD, S.TAIL)
    S.HEAD, S.TAIL = hp, tp
    try:
        raw_before = {i: r for i, r in S.split_blocks(open(tp).read())[1]}
        got = S.reconcile(2)
        want("head is filled to the budget", len(got["head"]), 2)
        want("a blocked entry never enters head", [i for i in got["head"] if i in ("L-2", "L-3")], [])
        second = S.reconcile(2, served=got["head"])
        want("served entries do not come back", sorted(set(second["head"]) & set(got["head"])), [])
        raw_after = {}
        for p in (hp, tp):
            for i, r in S.split_blocks(open(p).read())[1]:
                raw_after[i] = r
        want("ids are conserved", sorted(raw_after) == sorted(raw_before), True)
        drift = [i for i in raw_before if raw_before[i] != raw_after.get(i)]
        want("no entry's text drifts when it travels", drift, [])
        big = S.reconcile(99)
        want("asking for more than exists is not an error", len(big["head"]), 6)
    finally:
        S.HEAD, S.TAIL = saved

    # ---- --extra-arg, and the flag pair that contradicts itself ------------
    base = H.claude_command("p", model="m", permission_mode="acceptEdits", session_id="s")
    want("the default passes a permission mode", "--permission-mode" in base, True)
    want("the session id is the host's", base[base.index("--session-id") + 1], "s")
    for form in ("--dangerously-skip-permissions", "--extra-arg=--dangerously-skip-permissions"):
        c = H.claude_command("p", permission_mode="acceptEdits", extra=[form])
        want("the skip flag reaches claude (%s)" % form[:14], form in c, True)
        want("and it suppresses --permission-mode (%s)" % form[:14], "--permission-mode" in c, False)
    passthrough = H.claude_command("p", extra=["--verbose", "--foo=bar"])
    want("arbitrary extra args are passed in order",
         passthrough[-2:], ["--verbose", "--foo=bar"])

    # ---- --effort, and the three ways it could be got wrong ----------------
    want("no effort means the flag is ABSENT, not empty",
         "--effort" in H.claude_command("p", model="m"), False)
    eff = H.claude_command("p", model="m", effort="xhigh")
    want("an effort reaches claude as a flag pair",
         eff[eff.index("--effort") + 1], "xhigh")
    want("effort does not disturb the model that precedes it",
         eff[eff.index("--model") + 1], "m")
    want("effort SURVIVES the skip-permissions suppression, unlike permission mode",
         "--effort" in H.claude_command("p", effort="max", permission_mode="acceptEdits",
                                        extra=["--dangerously-skip-permissions"]), True)
    parsed = H.build_parser().parse_args(["run", "--effort", "high"])
    want("the parser accepts a listed level", parsed.effort, "high")
    want("and defaults to None so the session default stands",
         H.build_parser().parse_args(["run"]).effort, None)
    try:
        H.build_parser().parse_args(["run", "--effort", "enormous"])
        want("an unlisted level is REFUSED rather than passed through", "accepted", "refused")
    except SystemExit:
        want("an unlisted level is REFUSED rather than passed through", "refused", "refused")

    # ---- the loop, and where --continuous turns a stop into a continuation ----
    waits = []
    real_sleep, H.sleep = H.sleep, lambda n: waits.append(n)
    real_run = H.run_context

    class Args(object):
        extra_arg = []
        prompt = os.path.join(HERE, "..", "..", "..", "docs", "lights", "BURNDOWN_LOOP.md")
        budget = 1; model = "m"; permission_mode = None; max_budget_usd = None
        keep_api_key = False; timeout_seconds = 10; heartbeat_seconds = 0; no_settle = True
        continuous = False; idle_seconds = 7; stop_after_errors = 2; max_runs = None

    class Done(Exception):
        pass

    def drive(script, **kw):
        a = Args()
        for k, v in kw.items():
            setattr(a, k, v)
        seq, seen = list(script), []

        def fake(args, i, served):
            if not seq:
                raise Done()          # the TEST's bound; production polls forever on purpose
            out = seq.pop(0)
            seen.append((i, out["outcome"], tuple(sorted(served))))
            return out
        H.run_context = fake
        del waits[:]
        try:
            rc = H.command_run(a)
        except Done:
            rc = "script-exhausted"
        return rc, seen, list(waits)

    def ok(ids):
        return {"outcome": "BUDGET SPENT", "row": {}, "served": ids, "head": ids,
                "workable": 5, "stderr": "", "exhausted": False}
    err = {"outcome": "ERROR", "row": {}, "served": ["L-9"], "head": ["L-9"], "workable": 5,
           "stderr": "boom", "exhausted": False}
    dry_work = {"outcome": "REGISTER DRY", "row": None, "served": [], "workable": 3, "stderr": ""}
    dry_none = {"outcome": "REGISTER DRY", "row": None, "served": [], "workable": 0, "stderr": ""}

    try:
        rc, seen, w = drive([ok(["L-1"]), dry_work])
        want("a dry register stops by default", (rc, w), (0, []))

        rc, seen, w = drive([err, err, ok(["L-3"])])
        want("two bad contexts in a row exit 1 by default", rc, 1)
        want("and they do not wait", w, [])

        rc, seen, w = drive([ok(["L-1"]), dry_work, ok(["L-1"])], continuous=True)
        want("continuous clears the served set for another pass",
             [s[2] for s in seen], [(), ("L-1",), ()])
        want("clearing the set does not wait", w, [])

        rc, seen, w = drive([dry_none, dry_none], continuous=True)
        want("nothing workable waits idle_seconds each poll", w, [7, 7])

        rc, seen, w = drive([err, err, ok(["L-4"])], continuous=True)
        want("an error streak waits and continues instead of exiting",
             ([s[1] for s in seen], w), (["ERROR", "ERROR", "BUDGET SPENT"], [7]))

        rc, seen, w = drive([ok(["L-1"]), ok(["L-2"]), ok(["L-3"])], continuous=True, max_runs=2)
        want("an explicit --max-runs still caps continuous", (rc, len(seen)), (0, 2))

        rc, seen, w = drive([dry_work, ok(["L-1"])], continuous=True)
        want("a dry pass does not consume a context number", [s[0] for s in seen], [1, 1])
    finally:
        H.sleep, H.run_context = real_sleep, real_run

    for d in (tmp, tree):
        shutil.rmtree(d, ignore_errors=True)

    if fails:
        for f in fails:
            print("FAIL " + f)
        return 1
    print("burndown host: 52 deliberate cases, all correct")
    return 0


if __name__ == "__main__":
    sys.exit(main())
