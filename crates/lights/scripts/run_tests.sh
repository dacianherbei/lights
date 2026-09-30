#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/../.."
FEATURES="${FEATURES:---all-features}"

LIGHTS_TARGET_DIR="$(python3 lights/scripts/target_dir.py --explain)" || {
  echo "run_tests.sh: lights/scripts/target_dir.py did not answer; refusing to build blind"
  exit 2
}
if [ -n "$LIGHTS_TARGET_DIR" ]; then
  export CARGO_TARGET_DIR="$LIGHTS_TARGET_DIR"
fi

DETACH="${DETACH:-1}"
ARGS=()
for a in "$@"; do
  case "$a" in
    --detachability) DETACH=1 ;;
    --no-detachability) DETACH=0 ;;
    *) ARGS+=("$a") ;;
  esac
done

if [ "$DETACH" = "1" ]; then
  echo "== feature detachability =="
  DETACH_STATUS=0
  if ! cargo test -p lights --no-default-features --features method-splat >/dev/null 2>&1; then
    DETACH_STATUS=1
    echo "DETACHABILITY BROKEN: --features method-splat does not BUILD AND PASS alone."
    echo "Pruning the other method is no longer a git rm of one directory."
  fi
  if ! cargo test -p lights --no-default-features --features method-raycast >/dev/null 2>&1; then
    DETACH_STATUS=1
    echo "DETACHABILITY BROKEN: --features method-raycast does not BUILD AND PASS alone."
    echo "Pruning the other method is no longer a git rm of one directory."
  fi
  if ! cargo build -p lights --no-default-features --features fit; then
    DETACH_STATUS=1
    echo "FEATURE TRAP: --features fit does not BUILD AND PASS alone. A feature that"
    echo "silently needs another must declare it in [features], not leave the"
    echo "caller to discover it through a type error in a stub module."
  fi
  if [ "$DETACH_STATUS" != "0" ]; then
    exit "$DETACH_STATUS"
  fi
  echo "detachability: every listed build is green"
  echo
fi

echo "== cargo test -p lights $FEATURES =="
LOG="${TEST_LOG:-$(mktemp -t lights_tests)}"
if [ "${#ARGS[@]}" -eq 0 ]; then
  cargo test -p lights $FEATURES 2>&1 | tee "$LOG"
else
  cargo test -p lights $FEATURES "${ARGS[@]}" 2>&1 | tee "$LOG"
fi
STATUS=${PIPESTATUS[0]}
echo
echo "== cargo test -p lights-mac, the macOS host =="
cargo test -p lights-mac 2>&1 | tee -a "$LOG"
HOST_STATUS=${PIPESTATUS[0]}
if [ "$HOST_STATUS" != "0" ]; then
  STATUS=$HOST_STATUS
fi
echo
python3 - "$LOG" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
rows = re.findall(r"test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored", text)
if not rows:
    print("NO TEST RESULT LINES — the run did not reach a single target")
    raise SystemExit(1)
passed = sum(int(r[1]) for r in rows)
failed = sum(int(r[2]) for r in rows)
ignored = sum(int(r[3]) for r in rows)
print("%d targets · %d passed · %d failed · %d ignored" % (len(rows), passed, failed, ignored))
print("log: %s" % sys.argv[1])
print("Quote BOTH numbers, always, and only from the run you just did.")
PY
exit $STATUS
