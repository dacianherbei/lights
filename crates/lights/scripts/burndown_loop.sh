#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/../../.."

MAX_RUNS="${MAX_RUNS:-40}"
LOG="${BURNDOWN_LOG:-docs/lights/burndown_loop.log}"
PROMPT="${BURNDOWN_PROMPT:-docs/lights/BURNDOWN_LOOP.md}"
PERM_FLAGS="${PERM_FLAGS:---permission-mode acceptEdits}"
# BURNDOWN_MODEL and BURNDOWN_EFFORT are BARE VALUES here -- "claude-opus-5",
# "high" -- exactly as burndown_host.py reads them. This file used to want a whole
# flag pair ("--model claude-opus-5"), so ONE NAME MEANT TWO THINGS across the two
# drivers: a host-shaped value landed here as a positional argument, the run used
# the DEFAULT model, and nothing said so. A flag-shaped value is refused rather
# than guessed at, and an unlisted effort is refused the way the host's parser
# refuses it -- failing here costs a message, failing later costs a context.
MODEL="${BURNDOWN_MODEL:-claude-opus-5}"
EFFORT="${BURNDOWN_EFFORT:-}"

case "$MODEL" in
  --*) printf 'BURNDOWN_MODEL is now a BARE model name, not a flag pair.\n  got:  %s\n  want: %s\n' \
         "$MODEL" "${MODEL#--model }" >&2; exit 2 ;;
esac
MODEL_FLAGS="--model $MODEL"

EFFORT_FLAGS=""
if [ -n "$EFFORT" ]; then
  case "$EFFORT" in
    low|medium|high|xhigh|max) EFFORT_FLAGS="--effort $EFFORT" ;;
    *) printf 'BURNDOWN_EFFORT=%s is not one of: low medium high xhigh max\n' "$EFFORT" >&2
       exit 2 ;;
  esac
fi
COST="${BURNDOWN_COST:-docs/lights/benchmarks/loop_cost.tsv}"
RAW_DIR="${BURNDOWN_RAW:-docs/lights/benchmarks/loop}"
KEEP_API_KEY="${BURNDOWN_KEEP_API_KEY:-0}"

if [ ! -r "$PROMPT" ]; then
  printf 'the prompt %s cannot be read; nothing to drive\n' "$PROMPT" >&2
  exit 2
fi

mkdir -p "$(dirname "$COST")" "$RAW_DIR" "$(dirname "$LOG")"
if [ ! -s "$COST" ]; then
  printf 'started\tcontext\tstatus\tsession\tturns\tinput\tcache_create\tcache_read\toutput\tcost_usd\tapi_ms\twall_s\topen\ttotal\n' > "$COST"
fi

drive() {
  if [ "$KEEP_API_KEY" = "1" ]; then
    claude -p "$1" $PERM_FLAGS $MODEL_FLAGS $EFFORT_FLAGS --output-format json
  else
    env -u ANTHROPIC_API_KEY claude -p "$1" $PERM_FLAGS $MODEL_FLAGS $EFFORT_FLAGS --output-format json
  fi
}

for i in $(seq 1 "$MAX_RUNS"); do
  started="$(date +%H:%M:%S)"
  raw="$RAW_DIR/context-$(printf '%03d' "$i").json"
  printf '%s  context %d of %d starts\n' "$started" "$i" "$MAX_RUNS" | tee -a "$LOG"

  python3 crates/lights/scripts/split_register.py --roll >> "$LOG" 2>&1

  begin="$(date +%s)"
  drive "$(cat "$PROMPT")" > "$raw" 2>> "$LOG"
  wall=$(( $(date +%s) - begin ))

  err="$RAW_DIR/context-$(printf '%03d' "$i").err"
  row="$(python3 crates/lights/scripts/loop_usage.py "$raw" 2> "$err")"
  verdict=$?
  cat "$err" >> "$LOG"
  if [ -z "$row" ]; then
    row="$(printf 'unreadable\t-\t-\t0\t0\t0\t0\t-\t-')"
  fi

  python3 crates/lights/scripts/burndown_host.py pages >> "$LOG" 2>&1
  standing="$(python3 crates/lights/scripts/burndown.py --check | tail -1)"
  open_count="$(printf '%s' "$standing" | awk '{print $1}')"
  total_count="$(printf '%s' "$standing" | awk '{print $4}')"

  printf '%s\t%d\t%s\t%d\t%s\t%s\n' \
    "$started" "$i" "$row" "$wall" "${open_count:--}" "${total_count:--}" >> "$COST"

  printf '%s  context %d ends: %s | %s | wall %ds\n' \
    "$(date +%H:%M:%S)" "$i" "$standing" "$row" "$wall" | tee -a "$LOG"

  if [ "$verdict" = "3" ]; then
    printf '%s  the register is dry after %d context(s)\n' "$(date +%H:%M:%S)" "$i" | tee -a "$LOG"
    exit 0
  fi

  if [ "$verdict" = "4" ]; then
    printf '%s  context %d could not run and neither can the next one:\n' \
      "$(date +%H:%M:%S)" "$i" | tee -a "$LOG"
    cat "$err"
    exit 4
  fi

  if [ "$verdict" != "0" ]; then
    printf '%s  context %d produced no trustworthy result (%s); stopping rather than spending another\n' \
      "$(date +%H:%M:%S)" "$i" "$raw" | tee -a "$LOG"
    cat "$err"
    exit 1
  fi
done

printf '%s  stopped after %d contexts (MAX_RUNS)\n' "$(date +%H:%M:%S)" "$MAX_RUNS" | tee -a "$LOG"
