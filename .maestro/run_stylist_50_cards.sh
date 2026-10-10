#!/bin/bash
# Deep run: all 50 AI Personal Stylist scenarios, capturing EVERY outfit card
# (first card + 2 scrolled cards) after generation.
# Requires: Metro on :8081, Expo Go logged in, Personal Stylist reachable.
# Usage: ./.maestro/run_stylist_50_cards.sh
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR/.."
M="$HOME/.maestro/bin/maestro"
OUT="$DIR/verify-shots"
mkdir -p "$OUT"
LOG="$OUT/s50c_run.log"

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

# ONLY="45" (comma-separated ids) restricts the run to those scenarios.
if [ -n "${ONLY:-}" ]; then
  filtered=()
  for row in "${SCENARIOS[@]}"; do
    id="${row%%|*}"
    case ",${ONLY}," in *",${id},"*) filtered+=("$row");; esac
  done
  if [ ${#filtered[@]} -eq 0 ]; then
    echo "ONLY=${ONLY} matched no scenarios" >&2
    exit 2
  fi
  SCENARIOS=("${filtered[@]}")
  echo "=== ONLY run [${ONLY}] $(date '+%F %T') ===" >> "$LOG"
else
  : > "$LOG"
fi

MQ=/tmp/expo_metro.log
pass=0
fail=0
n=0
for row in "${SCENARIOS[@]}"; do
  n=$((n + 1))
  id="${row%%|*}"
  comment="${row#*|}"
  shot="s50c_${id}"
  rm -f "$OUT/${shot}.png" "$OUT/${shot}_c2.png" "$OUT/${shot}_c3.png" "$OUT/${shot}_c4.png" "$OUT/${shot}_c5.png" "$OUT/${shot}_c6.png" "$OUT/${shot}_c7.png" "$OUT/${shot}_fail.png"
  echo "=== ${id}/50 ${comment} ===" | tee -a "$LOG"
  ok=0
  pre_shown=$(grep -c "edge=5 shown=" "$MQ" 2>/dev/null || true)
  pre_shown=${pre_shown:-0}
  for attempt in 1 2 3 4 5; do
    out="$($M test ".maestro/sim_stylist_50_cards.yaml" --no-reinstall-driver \
      --test-output-dir "$OUT" \
      -e "COMMENT=${comment}" -e "SHOT=${shot}" 2>&1)" || true
    echo "$out" >> "$LOG"
    deck_ok=$(sed -n "$((pre_shown + 1)),\$p" "$MQ" 2>/dev/null | grep -c "edge=5 shown=3" || true)
    if echo "$out" | grep -qE "FAILED|Could not|Timed out|Assertion|Debug output|No space left"; then
      echo "  attempt ${attempt} failed (maestro)" | tee -a "$LOG"
      sleep 5
    elif [ "${deck_ok:-0}" -lt 1 ]; then
      echo "  attempt ${attempt} failed (deck not shown=3)" | tee -a "$LOG"
      sleep 5
    else
      ok=1
      break
    fi
  done
  # Consolidate the newest named screenshots to the top level.
  for suffix in "" "_c2" "_c3" "_c4" "_c5" "_c6" "_c7"; do
    f=$(find "$OUT" -name "${shot}${suffix}.png" -mmin -10 2>/dev/null | xargs ls -t 2>/dev/null | head -1)
    if [ -n "${f:-}" ]; then cp "$f" "$OUT/${shot}${suffix}.png" 2>/dev/null || true; fi
  done
  if (( ok )) && [ -f "$OUT/${shot}.png" ]; then
    pass=$((pass + 1))
    echo "  PASS (shots $(ls "$OUT" | grep -c "^${shot}"))" | tee -a "$LOG"
  else
    fail=$((fail + 1))
    echo "  FAIL (no screenshot found)" | tee -a "$LOG"
    xcrun simctl io booted screenshot "$OUT/${shot}_fail.png" >/dev/null 2>&1 || true
  fi
done

echo "=== cards run done pass=${pass} fail=${fail} ===" | tee -a "$LOG"
