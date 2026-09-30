#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PARSER = os.path.join(HERE, "loop_usage.py")
FIXTURE = os.path.join(HERE, "loop_usage_fixture.json")

OK_RUNNING = 0
BAD = 2
OK_DRY = 3
EXHAUSTED = 4


def run(text):
    handle, path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(handle, "w") as fh:
            fh.write(text)
        done = subprocess.run(
            [sys.executable, PARSER, path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        return done.returncode, done.stdout.strip()
    finally:
        os.unlink(path)


def with_result(base, result):
    doc = json.loads(base)
    doc["result"] = result
    return json.dumps(doc)


def main():
    with open(FIXTURE, "r") as fh:
        base = fh.read()

    cases = [
        ("the fixture as recorded, whose result is exactly the sentinel", base, OK_DRY),
        (
            "the sentinel on its own line at the end of a real handover",
            with_result(base, "L-99 closed: the thing.\n\nREGISTER DRY"),
            OK_DRY,
        ),
        (
            "the sentinel on its own line with trailing whitespace",
            with_result(base, "done\nREGISTER DRY   \n"),
            OK_DRY,
        ),
        (
            "THE ONE THAT MATTERS: the sentinel mentioned inside prose must NOT stop the loop",
            with_result(
                base,
                "I checked and the register is not dry, so I did not print "
                "REGISTER DRY on its own line.",
            ),
            OK_RUNNING,
        ),
        (
            "a sentinel that is only part of a longer line",
            with_result(base, "REGISTER DRY RUN COMPLETE"),
            OK_RUNNING,
        ),
        (
            "ordinary work output",
            with_result(base, "L-56 BUILT. Suite 54 targets, 510 passed."),
            OK_RUNNING,
        ),
        ("an empty file", "", BAD),
        ("whitespace only", "   \n\t\n", BAD),
        ("stderr merged into stdout, the documented corruption", "some warning\n" + base, BAD),
        ("truncated json", base[: len(base) // 2], BAD),
        ("json that is not an object", "[1, 2, 3]", BAD),
        (
            "a run that reported is_error must never be read as dry",
            json.dumps(dict(json.loads(base), is_error=True, result="REGISTER DRY")),
            BAD,
        ),
        (
            "a 429 is the ACCOUNT running out, not the register, and says so distinctly",
            json.dumps(dict(json.loads(base), is_error=True, api_error_status=429,
                            result="You're out of usage credits.")),
            EXHAUSTED,
        ),
        (
            "out of credits without a status is still recognised by its words",
            json.dumps(dict(json.loads(base), is_error=True,
                            result="Credit balance is too low")),
            EXHAUSTED,
        ),
        (
            "an ordinary error is NOT reported as an exhausted account",
            json.dumps(dict(json.loads(base), is_error=True, api_error_status=500,
                            result="internal error")),
            BAD,
        ),
        (
            "a missing result field is not dry",
            json.dumps({k: v for k, v in json.loads(base).items() if k != "result"}),
            OK_RUNNING,
        ),
        (
            "a result that is not a string is not dry",
            with_result(base, {"nested": "REGISTER DRY"}),
            OK_RUNNING,
        ),
    ]

    failures = []
    for name, text, expected in cases:
        code, out = run(text)
        if code != expected:
            failures.append("%s: expected exit %d, got %d" % (name, expected, code))
        if expected != BAD and not out:
            failures.append("%s: exited %d and printed no row" % (name, code))

    code, out = run(base)
    fields = out.split("\t")
    if len(fields) != 9:
        failures.append("the row has %d fields, expected 9: %r" % (len(fields), out))

    if failures:
        for line in failures:
            print("FAIL " + line)
        return 1

    print("loop usage parser: %d deliberate cases, all correct" % len(cases))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
