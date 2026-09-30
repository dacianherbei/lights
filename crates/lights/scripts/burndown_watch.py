import glob
import json
import os
import pathlib
import signal
import subprocess
import time

HERE = pathlib.Path(__file__).resolve().parent
CRATE = HERE.parent
ROOT = CRATE.parent.parent

USAGE_FIELDS = ("input_tokens", "output_tokens", "cache_creation_input_tokens",
                "cache_read_input_tokens")
USAGE_SOURCES = ("result", "transcript", "unmeasured")
PIPE_DRAIN_SECONDS = 30
GIT_SECONDS = 60
HEARTBEAT_SECONDS = 300
TRANSCRIPTS = "~/.claude/projects"

# input, output, cache-write, cache-read in USD per million tokens.
PRICE_USD_PER_MTOK = {
    "claude-opus-5": (5.0, 25.0, 6.25, 0.5),
    "claude-sonnet-5": (3.0, 15.0, 3.75, 0.3),
}


class LoopLimit(Exception):
    pass


class Bound(object):
    def __init__(self, limit, what):
        self.limit, self.what, self.n = int(limit), what, 0

    def __call__(self):
        self.n += 1
        if self.n > self.limit:
            raise LoopLimit("%s exceeded its bound of %d" % (self.what, self.limit))
        return self.n

    def __iter__(self):
        while self.n <= self.limit:
            yield self()


def bounded(limit, what):
    return Bound(limit, what)


def stamp(msg):
    print("%s  %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def ps_read(fields):
    try:
        return subprocess.run(["ps", "-Ao", fields], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, universal_newlines=True,
                              timeout=10).stdout
    except (subprocess.TimeoutExpired, OSError):
        stamp("ps did not answer in 10s - the host cannot see the process table this tick")
        return ""


def group_left(pgid):
    if not pgid:
        return []
    live = []
    for line in ps_read("pid=,pgid=,args=").splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3 or not parts[1].isdigit() or int(parts[1]) != pgid:
            continue
        if int(parts[0]) in (os.getpid(), os.getpgid(0)):
            continue
        live.append("%s %s" % (parts[0], parts[2][:90]))
    return live


def reap_group(pgid):
    left = group_left(pgid)
    if not left:
        return []
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            break
        time.sleep(2)
        if not group_left(pgid):
            break
    return left


def kill_and_drain(proc, seconds):
    # REAP THE SESSION BEFORE READING THE PIPES, NEVER AFTER. proc.kill() ends the
    # direct child only; its children inherited these pipe write ends, so a bare
    # communicate() here waits on a cargo run or an MCP server the wall was meant to end.
    reaped = reap_group(proc.pid)
    try:
        out, err = proc.communicate(timeout=seconds)
    except subprocess.TimeoutExpired:
        for pipe in (proc.stdout, proc.stderr):
            try:
                pipe.close()
            except (OSError, AttributeError):
                pass
        out, err = "", ""
    return out or "", err or "", reaped


def wall_note(err, seconds):
    return (err + "\n" if err.strip() else "") + "TIMEOUT after %ds" % seconds


def run_bounded(cmd, cwd, seconds, env=None):
    proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True, env=env, start_new_session=True)
    try:
        out, err = proc.communicate(timeout=seconds)
        return subprocess.CompletedProcess(cmd, proc.returncode, out or "", err or "")
    except subprocess.TimeoutExpired:
        out, err, _ = kill_and_drain(proc, PIPE_DRAIN_SECONDS)
        return subprocess.CompletedProcess(cmd, None, out, wall_note(err, seconds))


def git_run(root, args, seconds=GIT_SECONDS):
    p = run_bounded(["git"] + list(args), root, seconds)
    if p.returncode is None:
        stamp("git %s did not return in %ds and was ended - the host treats it as failed"
              % (" ".join(args[:2]), seconds))
    return p


def cpu_seconds(text):
    try:
        return sum(float(p) * 60 ** i for i, p in enumerate(reversed(text.split(":"))))
    except ValueError:
        return 0.0


def group_cpu(pgid):
    total, n = 0.0, 0
    for line in ps_read("pgid=,time=").splitlines():
        parts = line.split()
        if len(parts) != 2 or not parts[0].isdigit() or int(parts[0]) != pgid:
            continue
        n += 1
        total += cpu_seconds(parts[1])
    return total, n


def transcript_of(session_id):
    if not session_id:
        return None
    hits = glob.glob(os.path.join(os.path.expanduser(TRANSCRIPTS), "*", "%s.jsonl" % session_id))
    return hits[0] if hits else None


def usage_from_transcript(session_id):
    path = transcript_of(session_id)
    if not path:
        return None
    seen, total = set(), {f: 0 for f in USAGE_FIELDS}
    with open(path, errors="replace") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            m = row.get("message")
            u = m.get("usage") if isinstance(m, dict) else None
            if not isinstance(u, dict):
                continue
            key = m.get("id") or row.get("uuid")
            if key in seen:
                continue
            seen.add(key)
            for f in USAGE_FIELDS:
                v = u.get(f)
                total[f] += int(v) if isinstance(v, (int, float)) else 0
    return dict(total, messages=len(seen)) if seen else None


def cost_of(model, usage):
    card = PRICE_USD_PER_MTOK.get(model)
    if not card:
        return None
    return sum(int(usage.get(f) or 0) * rate for f, rate in zip(USAGE_FIELDS, card)) / 1e6


def recover_usage(r, session_id, model):
    r["session_id"] = r.get("session_id") or session_id
    if r.get("usage_source") != "unmeasured":
        return r
    u = usage_from_transcript(session_id)
    if not u:
        return r
    for f in USAGE_FIELDS:
        r[f] = u[f]
    r["total_cost_usd"] = cost_of(model, u)
    r["usage_source"] = "transcript"
    stamp("the result never arrived, so the usage was read back from the transcript of %s: "
          "%d messages, %s context tokens, cost %s"
          % (session_id, u["messages"], "{:,}".format(sum(u[f] for f in USAGE_FIELDS)),
             "unpriced - %s is not in the card" % model if r["total_cost_usd"] is None
             else "$%.2f" % r["total_cost_usd"]))
    return r


def source_of(row):
    src = row.get("usage_source") or ("result" if row.get("parsed") else "unmeasured")
    return src if src in USAGE_SOURCES else "unmeasured"


def heartbeat_loop(proc, session_id, stop, every, limit, label):
    t0, cpu_was, size_was = time.monotonic(), 0.0, 0
    try:
        for _ in bounded(limit, "the heartbeat of %s" % label):
            if stop.wait(every) or proc.poll() is not None:
                return
            cpu, procs = group_cpu(proc.pid)
            path = transcript_of(session_id)
            live = bool(path and os.path.exists(path))
            size = os.path.getsize(path) if live else 0
            quiet = (time.time() - os.path.getmtime(path)) if live else None
            stamp("%s · %.0fm in · %d process(es) · cpu %.0fs (+%.0fs) · transcript %s (+%s), "
                  "last write %s"
                  % (label, (time.monotonic() - t0) / 60.0, procs, cpu, cpu - cpu_was,
                     "{:,}B".format(size), "{:,}B".format(size - size_was),
                     "never" if quiet is None else "%.0fs ago" % quiet))
            cpu_was, size_was = cpu, size
    except LoopLimit as e:
        stamp("%s: %s - the heartbeat stops here. The wall above it should have fired first, so "
              "read this as the wall having failed, not as the context having finished."
              % (label, e))
