#!/bin/bash
# 50 distinct AI Personal Stylist scenarios on the booted iOS simulator.
# Requires: Metro on :8081, Expo Go logged in, Personal Stylist reachable.
# Usage: ./.maestro/run_stylist_50.sh
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR/.."
M="$HOME/.maestro/bin/maestro"
OUT="$DIR/verify-shots"
mkdir -p "$OUT"
LOG="$OUT/s50_run.log"
: > "$LOG"

# id|comment  — comments must parse to unique style × occasion (see stylistIntent.ts)
SCENARIOS=(
  "01|work, old money"
  "02|work, classic"
  "03|work, semi-classic"
  "04|work, minimalist"
  "05|work, casual"
  "06|weekend casual brunch, old money"
  "07|weekend casual brunch, classic"
  "08|weekend casual brunch, semi-classic"
  "09|weekend casual brunch, minimalist"
  "10|weekend casual brunch"
  "11|date night, old money"
  "12|date night, classic"
  "13|date night, semi-classic"
  "14|date night, minimalist"
  "15|date night, casual"
  "16|formal wedding, old money"
  "17|formal wedding, classic"
  "18|formal wedding, semi-classic"
  "19|formal gala, minimalist"
  "20|formal wedding, casual"
  "21|gym workout, old money"
  "22|gym workout, classic"
  "23|gym workout, semi-classic"
  "24|gym workout, minimalist"
  "25|gym workout, casual"
  "26|travel airport, old money"
  "27|travel airport, classic"
  "28|travel airport, semi-classic"
  "29|travel airport, minimalist"
  "30|travel airport, casual"
  "31|work, old money, \$250"
  "32|work, minimalist, under \$100"
  "33|office interview, classic"
  "34|corporate meeting, old money"
  "35|romantic dinner, old money"
  "36|black tie wedding, classic"
  "37|quiet luxury weekend casual"
  "38|everyday casual, minimalist"
  "39|airport trip, semi-classic"
  "40|flight travel, casual"
  "41|sport run, casual"
  "42|business casual work"
  "43|modern professional office"
  "44|date night classic"
  "45|old money work"
  "46|minimalist date night"
  "47|classic work, \$200"
  "48|semi-classic brunch weekend"
  "49|formal old money"
  "50|casual everyday, smart casual"
)

pass=0
fail=0
n=0
# START_FROM=NN resumes a partial run (skips scenarios with a smaller id).
START_FROM="${START_FROM:-1}"
for row in "${SCENARIOS[@]}"; do
  n=$((n + 1))
  id="${row%%|*}"
  comment="${row#*|}"
  if (( id < START_FROM )); then continue; fi
  shot="s50_${id}"
  echo "=== ${id}/50 ${comment} ===" | tee -a "$LOG"
  ok=0
  for attempt in 1 2; do
    out="$($M test ".maestro/sim_stylist_50_one.yaml" --no-reinstall-driver \
      --test-output-dir "$OUT" \
      -e "COMMENT=${comment}" -e "SHOT=${shot}" 2>&1)" || true
    echo "$out" >> "$LOG"
    if echo "$out" | grep -qE "FAILED|Could not|Timed out|Assertion|Debug output"; then
      echo "  attempt ${attempt} failed" | tee -a "$LOG"
      sleep 5
    else
      ok=1
      break
    fi
  done
  if (( ok )); then
    pass=$((pass + 1))
    echo "  PASS" | tee -a "$LOG"
  else
    fail=$((fail + 1))
    echo "  FAIL" | tee -a "$LOG"
    xcrun simctl io booted screenshot "$OUT/${shot}_fail.png" >/dev/null 2>&1 || true
  fi
  # Consolidate the newest named screenshot to the top level (macOS find: -mmin).
  f=$(find "$OUT" -name "${shot}.png" -mmin -10 2>/dev/null | xargs ls -t 2>/dev/null | head -1)
  if [ -n "${f:-}" ]; then cp "$f" "$OUT/${shot}.png" 2>/dev/null || true; fi
  # Per-run dumps (screen recordings + hierarchies) run hundreds of MB and
  # filled the disk mid-suite once — the consolidated PNG is all we keep.
  find "$OUT" -maxdepth 1 -type d -name "20*" -exec rm -rf {} + 2>/dev/null || true
  # Every maestro invocation also leaks a ~190 MB xcodebuild log dir into
  # $TMPDIR (47 GB accumulated over a day of runs) — purge stale ones.
  find "${TMPDIR:-/tmp}" -maxdepth 1 -name "maestro_xctestrunner_*" -mmin +3 \
    -exec rm -rf {} + 2>/dev/null || true
done

echo "=== done pass=${pass} fail=${fail} ===" | tee -a "$LOG"
