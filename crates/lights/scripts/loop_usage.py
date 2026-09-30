#!/usr/bin/env python3
import json
import sys

DRY = "REGISTER DRY"

OK_RUNNING = 0
BAD = 2
OK_DRY = 3
EXHAUSTED = 4

CREDIT_WORDS = ("usage credits", "usage limit", "rate limit", "credit balance")

COLUMNS = (
    "status",
    "session",
    "turns",
    "input",
    "cache_create",
    "cache_read",
    "output",
    "cost_usd",
    "api_ms",
)


def is_dry(result):
    if not isinstance(result, str):
        return False
    for line in result.splitlines():
        if line.strip() == DRY:
            return True
    return False


def row(doc):
    usage = doc.get("usage") or {}

    def count(name):
        value = usage.get(name)
        return int(value) if isinstance(value, (int, float)) else 0

    cost = doc.get("total_cost_usd")
    turns = doc.get("num_turns")
    api_ms = doc.get("duration_ms")
    status = "error" if doc.get("is_error") else "ok"
    return [
        status,
        str(doc.get("session_id") or "-"),
        str(turns if isinstance(turns, int) else "-"),
        str(count("input_tokens")),
        str(count("cache_creation_input_tokens")),
        str(count("cache_read_input_tokens")),
        str(count("output_tokens")),
        ("%.6f" % cost) if isinstance(cost, (int, float)) else "-",
        str(api_ms if isinstance(api_ms, int) else "-"),
    ]


def main(argv):
    if len(argv) != 2:
        sys.stderr.write("usage: loop_usage.py <raw-json-file>\n")
        return BAD

    try:
        with open(argv[1], "r") as handle:
            text = handle.read()
    except OSError as exc:
        sys.stderr.write("cannot read %s: %s\n" % (argv[1], exc))
        return BAD

    if not text.strip():
        sys.stderr.write(
            "%s is empty. Claude Code wrote no JSON at all, so neither the usage "
            "nor the REGISTER DRY signal can be read, and continuing would spend "
            "another context on a driver that cannot tell when to stop.\n" % argv[1]
        )
        return BAD

    try:
        doc = json.loads(text)
    except ValueError as exc:
        sys.stderr.write(
            "%s is not JSON (%s). The usual cause is stderr merged into stdout: "
            "capture them separately, because --output-format json writes the "
            "document to stdout alone.\n" % (argv[1], exc)
        )
        return BAD

    if not isinstance(doc, dict):
        sys.stderr.write("%s is JSON but not an object\n" % argv[1])
        return BAD

    sys.stdout.write("\t".join(row(doc)))
    sys.stdout.write("\n")

    if doc.get("is_error"):
        result = str(doc.get("result") or "")
        status = doc.get("api_error_status")
        terminal = doc.get("terminal_reason")
        detail = ", ".join(
            part for part in (
                ("api_error_status %s" % status) if status else "",
                ("terminal_reason %s" % terminal) if terminal else "",
            ) if part
        )
        lowered = result.lower()
        if status == 429 or any(word in lowered for word in CREDIT_WORDS):
            sys.stderr.write(
                "THE ACCOUNT IS OUT OF BUDGET, NOT THE REGISTER (%s).\n%s\n"
                "Retrying will not help: each further context fails in seconds "
                "for the same reason and the loop would burn its MAX_RUNS on "
                "nothing. Top up, or pass a different model with "
                "BURNDOWN_MODEL=<name> (a bare model name, not a flag pair), then start the loop again.\n"
                % (detail or "no status reported", result[:400])
            )
            return EXHAUSTED
        sys.stderr.write(
            "the run reported is_error, so its REGISTER DRY signal is not "
            "trustworthy (%s): %s\n" % (detail or "no status reported", result[:400])
        )
        return BAD

    return OK_DRY if is_dry(doc.get("result")) else OK_RUNNING


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
