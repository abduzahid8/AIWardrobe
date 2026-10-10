#!/bin/bash
# Resume run: scenarios 39-50 of the 50 AI Personal Stylist suite.
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR/.."
M="$HOME/.maestro/bin/maestro"
OUT="$DIR/verify-shots"
mkdir -p "$OUT"
LOG="$OUT/s50_run.log"

SCENARIOS=(
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
for row in "${SCENARIOS[@]}"; do
  id="${row%%|*}"
  comment="${row#*|}"
  shot="s50_${id}"
  rm -f "$OUT/${shot}.png" "$OUT/${shot}_fail.png"
  echo "=== RESUME ${id}/50 ${comment} ===" | tee -a "$LOG"
  ok=0
  for attempt in 1 2 3; do
    out="$($M test ".maestro/sim_stylist_50_one.yaml" --no-reinstall-driver \
      --test-output-dir "$OUT" \
      -e "COMMENT=${comment}" -e "SHOT=${shot}" 2>&1)" || true
    echo "$out" >> "$LOG"
    if echo "$out" | grep -qE "No space left|FAILED|Could not|Timed out|Assertion"; then
      echo "  attempt ${attempt} failed" | tee -a "$LOG"
      sleep 5
    else
      ok=1
      break
    fi
  done
  # Verify the screenshot actually landed before trusting PASS.
  f=$(find "$OUT" -name "${shot}.png" -mmin -10 2>/dev/null | xargs ls -t 2>/dev/null | head -1)
  if [ -n "${f:-}" ]; then cp "$f" "$OUT/${shot}.png" 2>/dev/null || true; fi
  if (( ok )) && [ -f "$OUT/${shot}.png" ]; then
    pass=$((pass + 1))
    echo "  PASS (verified screenshot)" | tee -a "$LOG"
  else
    fail=$((fail + 1))
    echo "  FAIL (no screenshot found)" | tee -a "$LOG"
    xcrun simctl io booted screenshot "$OUT/${shot}_fail.png" >/dev/null 2>&1 || true
  fi
done

echo "=== resume done pass=${pass} fail=${fail} ===" | tee -a "$LOG"
