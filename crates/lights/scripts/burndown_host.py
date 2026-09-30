#!/usr/bin/env python3
"""THE LIGHTS BURN-DOWN HOST - one register entry, one Claude Code context, every context measured.

Adapted 2026-09-28 from codeforge's scripts/burndown_host.py, which is the same idea grown up:
the host owns the BUDGET and injects it into the prompt, fills the head to exactly that budget
before each context, gives the context its own session id so what it leaves behind can be found,
watches it with a heartbeat, ends it with a wall that reaps the whole process group, recovers the
usage from the transcript when the result JSON never arrives, and appends one row per context.

It replaces burndown_loop.sh, which measured a context only if it finished cleanly, printed nothing
for the 40-70 minutes one was running, could not end a grandchild, and hard-coded a budget of ten.
"""
import argparse
import importlib
import itertools
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
import uuid

HERE = pathlib.Path(__file__).resolve().parent
CRATE = HERE.parent
ROOT = CRATE.parent.parent
sys.path.insert(0, str(HERE))

import burndown                                              # noqa: E402
import split_register                                        # noqa: E402
from burndown_watch import (GIT_SECONDS, HEARTBEAT_SECONDS, PIPE_DRAIN_SECONDS,  # noqa: E402
                           USAGE_FIELDS, git_run, heartbeat_loop, kill_and_drain,
                           reap_group, recover_usage, source_of, stamp, wall_note)

PROMPT = ROOT / "docs" / "lights" / "BURNDOWN_LOOP.md"
USAGE_LOG = ROOT / "docs" / "lights" / "benchmarks" / "loop_usage.json"
PAGES_DIR = ROOT / "docs" / "lights" / "pages"
PAGES = (
    ("burndown", HERE / "burndown.py", CRATE / "head_open.toml"),
    ("research", HERE / "research.py", ROOT / "docs" / "lights" / "RESEARCH.json"),
)
DEFAULT_MODEL = "claude-opus-5"
# `claude --effort <level>`. None means "say nothing and let the session decide",
# which is the only default that cannot change behaviour by being added.
EFFORTS = ("low", "medium", "high", "xhigh", "max")
DEFAULT_MAX_RUNS = 40
IDLE_SECONDS = 600
sleep = time.sleep  # a name so a test can replace it and drive the loop without waiting
BUDGET_MARK = "{{BUDGET}}"
AUTH_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
BG_WAIT_VAR = "CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS"
OUTCOMES = ("BUDGET SPENT", "REGISTER DRY", "ERROR", "ABANDONED", "OTHER")


def scrub_env(environ, keep_api_key=False):
    """Both credentials, not just the key. Measured 2026-09-15: a spawned context inherits
    ANTHROPIC_API_KEY and bills an API account, which fails as a clean $0.00 success."""
    if keep_api_key:
        return dict(environ)
    return {k: v for k, v in environ.items() if k not in AUTH_VARS}


def child_env(environ, keep_api_key, wall_seconds):
    env = scrub_env(environ, keep_api_key)
    env[BG_WAIT_VAR] = str(int(wall_seconds) * 1000)
    return env


def prompt_text(path, budget):
    """The budget is the HOST's to set, through exactly one marker."""
    text = pathlib.Path(path).read_text(encoding="utf-8")
    n = text.count(BUDGET_MARK)
    if n != 1:
        raise SystemExit("[host] %s carries %s %d times; the host sets the budget through exactly "
                         "one" % (path, BUDGET_MARK, n))
    return text.replace(BUDGET_MARK, str(int(budget)))


SKIP_PERMISSIONS = "--dangerously-skip-permissions"


def claude_command(prompt, model=None, effort=None, permission_mode=None,
                   max_budget_usd=None, session_id=None, extra=()):
    """--dangerously-skip-permissions and --permission-mode contradict each other, so the
    host passes only one. The explicit skip wins, because a caller who typed it meant it."""
    extra = list(extra)
    if any(a == SKIP_PERMISSIONS or a.endswith("=" + SKIP_PERMISSIONS) for a in extra):
        permission_mode = None
    cmd = ["claude", "-p", prompt, "--output-format", "json"]
    if session_id:
        cmd += ["--session-id", session_id]
    if model:
        cmd += ["--model", model]
    if effort:
        cmd += ["--effort", effort]
    if permission_mode:
        cmd += ["--permission-mode", permission_mode]
    if max_budget_usd is not None:
        cmd += ["--max-budget-usd", str(max_budget_usd)]
    return cmd + list(extra)


def read_result(stdout):
    r = {f: 0 for f in USAGE_FIELDS}
    r.update({"parsed": False, "is_error": None, "num_turns": None, "session_id": None,
              "total_cost_usd": None, "duration_ms": None, "result": "",
              "api_error_status": None, "usage_source": "unmeasured"})
    try:
        doc = json.loads(stdout)
    except ValueError:
        return r
    if not isinstance(doc, dict):
        return r
    usage = doc.get("usage") if isinstance(doc.get("usage"), dict) else {}
    for f in USAGE_FIELDS:
        v = usage.get(f)
        r[f] = int(v) if isinstance(v, (int, float)) else 0
    for k in ("is_error", "num_turns", "session_id", "total_cost_usd", "duration_ms",
              "api_error_status"):
        r[k] = doc.get(k)
    r["result"] = doc.get("result") or ""
    r["parsed"] = True
    r["usage_source"] = "result" if any(r[f] for f in USAGE_FIELDS) else "unmeasured"
    return r


def outcome_of(result_text, is_error=False):
    if is_error:
        return "ERROR"
    for line in reversed([l.strip() for l in (result_text or "").splitlines() if l.strip()]):
        if line.startswith("BUDGET SPENT"):
            return "BUDGET SPENT"
        if line == "REGISTER DRY":
            return "REGISTER DRY"
    return "OTHER"


def classify(r, exit_code):
    """The outcome, and the two cases a stop-word reader alone gets wrong.

    An unreadable result is NOT "OTHER". If nothing parsed and no usage could be recovered,
    the host learned nothing and must not start another context on the assumption that this
    one was fine - that is the behaviour burndown_loop.sh had and this host briefly lost.
    If the result never parsed but the TRANSCRIPT shows the context did work, it ran and its
    answer was lost, which is ABANDONED rather than ERROR: a different fact, worth a
    different word, and the usage row is real either way.
    """
    if exit_code is None:
        return "ABANDONED"
    if not r.get("parsed"):
        return "ABANDONED" if r.get("usage_source") == "transcript" else "ERROR"
    return outcome_of(r.get("result"), r.get("is_error"))


def exhausted(r):
    """A credit/rate wall is not a register result and retrying cannot help."""
    words = ("usage credits", "usage limit", "rate limit", "credit balance")
    low = (r.get("result") or "").lower()
    return r.get("api_error_status") == 429 or any(w in low for w in words)


def register_snapshot():
    """Every entry by id, BEFORE the context. A count cannot name an entry, and an entry
    raised and closed in one sitting nets to zero and is invisible to every number."""
    entries, _, _ = burndown.load()
    return {e.get("id"): {"kind": e.get("kind"), "status": e.get("status"),
                          "state": e.get("state"), "title": e.get("title")} for e in entries}


def what_moved(before, after):
    closed = sorted(i for i, a in after.items()
                    if a["status"] == "closed" and before.get(i, {}).get("status") == "open")
    raised = sorted(i for i in after if i not in before)
    reopened = sorted(i for i, a in after.items()
                      if a["status"] == "open" and before.get(i, {}).get("status") == "closed")
    edited = sorted(i for i, a in after.items()
                    if i in before and a != before[i] and i not in closed and i not in reopened)
    return {"closed": closed, "raised": raised, "reopened": reopened, "edited": edited}


def standing():
    entries, bad, _ = burndown.load()
    problems = burndown.check(entries, bad)
    o, t = burndown.standing(entries)
    return o, t, problems


def context_commits(before_sha):
    """What the context committed, read from OUTSIDE it rather than from what it claims."""
    p = git_run(str(ROOT), ["log", "--oneline", "%s..HEAD" % before_sha])
    if p.returncode != 0:
        return []
    return [l.strip() for l in (p.stdout or "").splitlines() if l.strip()]


def head_sha():
    p = git_run(str(ROOT), ["rev-parse", "HEAD"])
    return (p.stdout or "").strip() if p.returncode == 0 else ""


def append_row(row):
    USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    rows.append(row)
    USAGE_LOG.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")


def load_rows():
    if not USAGE_LOG.exists():
        return []
    try:
        rows = json.loads(USAGE_LOG.read_text(encoding="utf-8"))
    except ValueError:
        return []
    return rows if isinstance(rows, list) else []


def spawn_context(cmd, env, timeout, session_id=None, heartbeat=0, label="context"):
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True, env=env, start_new_session=True)
    stop = threading.Event()
    if heartbeat > 0:
        threading.Thread(target=heartbeat_loop,
                         args=(proc, session_id, stop, heartbeat, timeout // heartbeat + 2, label),
                         daemon=True).start()
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return stdout or "", stderr or "", proc.returncode, reap_group(proc.pid)
    except subprocess.TimeoutExpired:
        stdout, stderr, reaped = kill_and_drain(proc, PIPE_DRAIN_SECONDS)
        return stdout, wall_note(stderr, timeout), None, reaped
    finally:
        stop.set()


def fresh_modules():
    importlib.reload(burndown)
    importlib.reload(split_register)


def run_context(args, i, served=()):
    fresh_modules()
    before_open, before_total, problems = standing()
    if problems:
        raise SystemExit("[host] the register does not check out before context %d: %s"
                         % (i, problems))
    before_reg = register_snapshot()
    before_sha = head_sha()

    fill = split_register.reconcile(args.budget, served)
    if not fill["head"]:
        entries, _, _ = burndown.load()
        workable = len(split_register.workable(entries))
        stamp("nothing to hand out: %d workable entr%s in the register, %d already served this run"
              % (workable, "y" if workable == 1 else "ies", len(set(served))))
        return {"outcome": "REGISTER DRY", "row": None, "served": [],
                "workable": workable, "stderr": ""}

    sid = str(uuid.uuid4())
    prompt = prompt_text(args.prompt, args.budget)
    stamp("context %d starts · budget %d · model %s · effort %s · %d open of %d · head %s · session %s"
          % (i, args.budget, args.model or "(default)",
             getattr(args, "effort", None) or "(session default)", before_open, before_total,
             ",".join(fill["head"]), sid[:8]))
    if fill["promoted"]:
        stamp("  promoted into head: %s" % ", ".join(fill["promoted"]))

    cmd = claude_command(prompt, model=args.model, effort=getattr(args, "effort", None),
                         permission_mode=args.permission_mode,
                         max_budget_usd=args.max_budget_usd, session_id=sid,
                         extra=getattr(args, "extra_arg", ()))
    t0 = time.monotonic()
    stdout, stderr, code, reaped = spawn_context(
        cmd, child_env(os.environ, args.keep_api_key, args.timeout_seconds),
        args.timeout_seconds, session_id=sid, heartbeat=args.heartbeat_seconds,
        label="context %d" % i)
    wall = time.monotonic() - t0

    r = recover_usage(read_result(stdout), sid, args.model or DEFAULT_MODEL)
    outcome = classify(r, code)
    after_open, after_total, _ = standing()
    moved = what_moved(before_reg, register_snapshot())
    commits = context_commits(before_sha) if before_sha else []

    row = {"iso": time.strftime("%Y-%m-%dT%H:%M:%S"), "context": i, "budget": args.budget,
           "model": args.model, "effort": getattr(args, "effort", None),
           "session_id": r["session_id"], "outcome": outcome,
           "exit_code": code, "wall_seconds": round(wall, 1), "usage_source": source_of(r),
           "num_turns": r["num_turns"], "total_cost_usd": r["total_cost_usd"],
           "open_before": before_open, "open_after": after_open, "total_after": after_total,
           "head": fill["head"], "moved": moved, "commits": commits,
           "reaped": reaped, "stderr_tail": (stderr or "")[-600:]}
    row.update({f: r[f] for f in USAGE_FIELDS})
    append_row(row)
    render_pages()

    ctx = sum(r[f] for f in USAGE_FIELDS)
    changed = len(moved["closed"]) + len(moved["raised"]) + len(moved["reopened"])
    stamp("context %d ends · %s · %.0fm · %d turns · %s ctx tokens · %s · %d open of %d · "
          "%d entries moved · %d commit(s)"
          % (i, outcome, wall / 60.0, r["num_turns"] or 0, "{:,}".format(ctx),
             "cost unmeasured" if r["total_cost_usd"] is None else "$%.2f" % r["total_cost_usd"],
             after_open, after_total, changed, len(commits)))
    for label in ("closed", "raised", "reopened"):
        if moved[label]:
            stamp("  %-8s %s" % (label, ", ".join(moved[label])))
    if reaped:
        stamp("  the wall reaped %d process(es) the context left behind" % len(reaped))
    if exhausted(r):
        stamp("  THE ACCOUNT IS OUT OF BUDGET, NOT THE REGISTER: %s" % r["result"][:200].strip())
        outcome = "ERROR"

    return {"outcome": outcome, "row": row, "served": fill["head"], "exhausted": exhausted(r),
            "head": fill["head"], "workable": fill["queued"] + len(fill["head"]),
            "stderr": stderr or ""}


def render_pages(out_dir=PAGES_DIR):
    """L-148: a `claude -p` context has no artifact tool, so the host writes the PM's pages to a
    fixed path where a stale page is a git diff. Returns (written paths, skipped names, failures)."""
    out_dir = pathlib.Path(out_dir)
    written, skipped, failed = [], [], []
    for name, script, source in PAGES:
        if not source.exists():
            skipped.append(name)
            stamp("skipped %s: %s is not in this tree" % (name, source))
            continue
        p = subprocess.run([sys.executable, str(script)], cwd=str(ROOT), capture_output=True,
                           text=True, timeout=GIT_SECONDS * 4)
        if p.returncode != 0 or "<title>" not in p.stdout:
            failed.append(name)
            stamp("FAILED %s: %s exited %d - %s" % (name, script.name, p.returncode,
                                                    (p.stderr or "")[-300:].strip()))
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / (name + ".html")
        tmp = dest.with_suffix(".html.tmp")
        tmp.write_text(p.stdout, encoding="utf-8")
        os.replace(tmp, dest)
        written.append(dest)
        stamp("wrote %s (%d bytes)" % (dest, dest.stat().st_size))
    return written, skipped, failed


def command_pages(args):
    _, _, failed = render_pages(args.out_dir)
    return 1 if failed else 0


def settle_commit(args, n):
    """The host's own move is committed by the host. A context cannot commit the register
    roll that preceded it, because the roll happened before it started."""
    render_pages()
    paths = ["crates/lights/head_open.toml", "crates/lights/tail.toml",
             str(USAGE_LOG.relative_to(ROOT)), str(PAGES_DIR.relative_to(ROOT))]
    p = git_run(str(ROOT), ["status", "--porcelain"] + paths)
    if p.returncode != 0 or not (p.stdout or "").strip():
        return
    git_run(str(ROOT), ["add"] + paths)
    git_run(str(ROOT), ["commit", "-q", "-m",
                        "lights host: the roll, the usage log and the pages for %d context(s)" % n],
            seconds=GIT_SECONDS)
    stamp("the host committed its own move (the roll, the usage log and the pages)")


def command_run(args):
    """The loop, and the three places it stops - or does not, under --continuous."""
    if not pathlib.Path(args.prompt).is_file():
        raise SystemExit("[host] %s is missing - nothing to drive" % args.prompt)
    if any(k in os.environ for k in AUTH_VARS) and not args.keep_api_key:
        stamp("note: %s is set in this shell and is NOT passed to the contexts - they run on the "
              "claude.ai login; --keep-api-key passes it"
              % ", ".join(k for k in AUTH_VARS if k in os.environ))

    cap = args.max_runs if args.max_runs is not None else (None if args.continuous else DEFAULT_MAX_RUNS)
    served, errors, i, ran = set(), 0, 1, 0
    for _ in itertools.count():
        if cap is not None and i > cap:
            stamp("stopped after %d context(s) (--max-runs)" % cap)
            break

        out = run_context(args, i, served)

        if out["outcome"] == "REGISTER DRY":
            if not args.continuous:
                stamp("the register is dry after %d context(s)" % ran)
                break
            # A served-but-unsolved entry stays OPEN, so "served everything once" is not "done".
            if served and out.get("workable"):
                stamp("continuous: every workable entry was served once (%d) - clearing the served "
                      "set, another pass starts" % len(served))
                served = set()
            else:
                n = out.get("workable") or 0
                stamp("continuous: %s - looking again in %ds. A ruling or a new entry may arrive "
                      "while it waits."
                      % ("nothing workable in the register" if not n else
                         "%d workable entr%s the head could not be filled from"
                         % (n, "y" if n == 1 else "ies"),
                         args.idle_seconds))
                sleep(args.idle_seconds)
            continue  # a dry pass does not consume a context number

        served |= set(out.get("head") or [])
        ran = i
        errors = errors + 1 if out["outcome"] in ("ERROR", "ABANDONED") else 0
        if errors >= args.stop_after_errors:
            tail = (out.get("stderr") or "")[-200:]
            if not args.continuous:
                stamp("stopping: %d context(s) in a row ended badly - last stderr: %s" % (errors, tail))
                if not args.no_settle:
                    settle_commit(args, ran)
                return 1
            stamp("continuous: %d context(s) in a row ended badly - waiting %ds, then continuing. A "
                  "usage limit is the usual cause and it lifts with time. Last stderr: %s"
                  % (errors, args.idle_seconds, tail))
            sleep(args.idle_seconds)
            errors = 0
        i += 1

    if not args.no_settle:
        settle_commit(args, ran)
    return 0


def column_of(row):
    return (int(row.get("budget") or 0), str(row.get("model") or "(unrecorded)"))


def command_report(args):
    rows = [r for r in load_rows() if r.get("usage_source") != "unmeasured"]
    if not rows:
        print("no measured context in %s yet" % USAGE_LOG)
        return 1
    cols = {}
    for r in rows:
        cols.setdefault(column_of(r), []).append(r)
    print("%-26s %7s %9s %11s %10s %9s %8s" %
          ("budget / model", "ctxs", "entries", "ctx tokens", "cost", "per entry", "min"))
    for key in sorted(cols):
        g = cols[key]
        ent = sum(len(r.get("moved", {}).get("closed", [])) + len(r.get("moved", {}).get("raised", []))
                  + len(r.get("moved", {}).get("reopened", [])) for r in g)
        ctx = sum(sum(int(r.get(f) or 0) for f in USAGE_FIELDS) for r in g)
        cost = sum(float(r["total_cost_usd"]) for r in g if r.get("total_cost_usd") is not None)
        mins = sum(float(r.get("wall_seconds") or 0) for r in g) / 60.0
        print("%-26s %7d %9d %11s %10s %9s %8.0f"
              % ("%d / %s" % key, len(g), ent, "{:,}".format(ctx), "$%.2f" % cost,
                 "$%.2f" % (cost / ent) if ent else "n/a", mins))
    print("\nThe number that decides is COST PER ENTRY CHANGED, not cost per context: a smaller "
          "budget pays the per-session floor more often and re-reads a smaller context each turn. "
          "A column is one budget on ONE model - two models are two experiments.")
    return 0


def command_inspect(args):
    o, t, problems = standing()
    print("register: %d open of %d · %s" % (o, t, "clean" if not problems else problems[:2]))
    print("prompt:   %s · %d bytes · budget marker %s"
          % (args.prompt, pathlib.Path(args.prompt).stat().st_size,
             "present once" if pathlib.Path(args.prompt).read_text().count(BUDGET_MARK) == 1
             else "MISSING OR REPEATED"))
    for name in ("head_open.toml", "tail.toml"):
        p = CRATE / name
        if p.exists():
            print("%-9s %s bytes (~%dk tokens)" % (name + ":", "{:,}".format(p.stat().st_size),
                                                   p.stat().st_size / 4000))
    present = [v for v in AUTH_VARS if v in os.environ]
    print("credentials in this shell that a child would inherit: %s"
          % (", ".join(present) if present else "none"))
    print("  (the host scrubs them unless --keep-api-key; without that a context bills an API "
          "account and fails as a clean $0.00)")
    rows = load_rows()
    print("usage log: %s · %d row(s)" % (USAGE_LOG, len(rows)))
    return 0


def build_parser():
    p = argparse.ArgumentParser(description="drive the lights burn-down, one entry per context")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(q):
        q.add_argument("--prompt", default=str(PROMPT))
        q.add_argument("--model", default=os.environ.get("BURNDOWN_MODEL") or DEFAULT_MODEL)
        q.add_argument("--effort", choices=EFFORTS,
                       default=os.environ.get("BURNDOWN_EFFORT") or None,
                       help="reasoning effort for the context (%s). Omitted entirely when "
                            "unset, so the session default stands; also settable as "
                            "BURNDOWN_EFFORT." % ", ".join(EFFORTS))
        q.add_argument("--permission-mode", default="acceptEdits")
        q.add_argument("--max-budget-usd", type=float,
                       default=float(os.environ["BURNDOWN_MAX_USD"])
                       if os.environ.get("BURNDOWN_MAX_USD") else None,
                       help="Claude Code's own dollar cap per context")
        q.add_argument("--keep-api-key", action="store_true")
        q.add_argument("--extra-arg", action="append", default=[], metavar="FLAG",
                       help="pass FLAG straight to `claude`; repeat as needed. "
                            "--extra-arg=--dangerously-skip-permissions is the usual one, and it "
                            "SUPPRESSES --permission-mode because the two contradict each other.")

    r = sub.add_parser("run", help="work the register, one context per --budget entries")
    r.add_argument("--budget", type=int, default=int(os.environ.get("BURNDOWN_BUDGET") or 1),
                   help="register entries per context (default 1)")
    r.add_argument("--max-runs", type=int, default=None,
                   help="stop after this many contexts (default %d; NO CAP under --continuous)"
                        % DEFAULT_MAX_RUNS)
    r.add_argument("--continuous", action="store_true",
                   default=os.environ.get("BURNDOWN_CONTINUOUS") == "1",
                   help="never stop: a dry register waits and looks again, an error streak waits "
                        "and continues")
    r.add_argument("--idle-seconds", type=int, default=IDLE_SECONDS,
                   help="under --continuous, the wait before looking again (default %d)" % IDLE_SECONDS)
    r.add_argument("--stop-after-errors", type=int, default=2,
                   help="how many bad contexts in a row before stopping (or waiting, under "
                        "--continuous)")
    r.add_argument("--timeout-seconds", type=int, default=7200,
                   help="the wall for one context; it reaps the whole process group")
    r.add_argument("--heartbeat-seconds", type=int, default=HEARTBEAT_SECONDS)
    r.add_argument("--no-settle", action="store_true", help="do not commit the host's own move")
    common(r)
    r.set_defaults(func=command_run)

    q = sub.add_parser("report", help="compare the recorded contexts by budget")
    q.set_defaults(func=command_report)

    g = sub.add_parser("pages", help="regenerate the PM's pages to fixed files (L-148)")
    g.add_argument("--out-dir", default=str(PAGES_DIR))
    g.set_defaults(func=command_pages)

    i = sub.add_parser("inspect", help="what a context pays before it acts")
    i.add_argument("--prompt", default=str(PROMPT))
    i.set_defaults(func=command_inspect)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        stamp("interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
