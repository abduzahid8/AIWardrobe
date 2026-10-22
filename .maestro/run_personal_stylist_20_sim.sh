#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."

UDID="${UDID:-A803D6CC-0840-40F3-95EF-70546B2E0ED0}"
M="$HOME/.maestro/bin/maestro"
FLOW="${FLOW:-.maestro/sim_personal_stylist_20.yaml}"
OUT="${OUT:-.maestro/verify-shots/sim20}"
LOG="$OUT/run${ONLY:+_${ONLY//,/_}}.log"
mkdir -p "$OUT"

SCENARIOS=(
  "01|work, old money"
  "02|work, classic, under \$250"
  "03|job interview, business casual"
  "04|corporate conference, semi-classic"
  "05|date night, old money"
  "06|romantic dinner, minimalist"
  "07|date night, casual"
  "08|weekend brunch, classic"
  "09|everyday casual, minimalist"
  "10|formal wedding, old money"
  "11|black tie wedding, classic"
  "12|formal gala, semi-classic"
  "13|gym workout, casual"
  "14|sport run, minimalist"
  "15|travel airport, classic"
  "16|flight travel, semi-classic"
  "17|quiet luxury weekend, old money"
  "18|modern professional office"
  "19|офис, олд мани"
  "20|sayohat uchun, oddiy"
)

if [ -n "${ONLY:-}" ]; then
  filtered=()
  for row in "${SCENARIOS[@]}"; do
    id="${row%%|*}"
    case ",${ONLY}," in *",${id},"*) filtered+=("$row");; esac
  done
  SCENARIOS=("${filtered[@]}")
fi

: > "$LOG"
pass=0
fail=0

for row in "${SCENARIOS[@]}"; do
  id="${row%%|*}"
  comment="${row#*|}"
  shot="sim20_${id}"
  ok=0
  echo "=== ${id} ${comment} ===" | tee -a "$LOG"

  for attempt in 1 2 3; do
    attempt_dir="$OUT/attempts/${id}_${attempt}"
    rm -rf "$attempt_dir"
    out=$("$M" --udid "$UDID" test "$FLOW" \
      --no-reinstall-driver --test-output-dir "$attempt_dir" \
      -e "COMMENT=${comment}" -e "SHOT=${shot}" 2>&1) || true
    rm -rf "$HOME/Library/Application Support/ai.opencode.desktop/maestro/tests"
    rm -rf "${TMPDIR:-/tmp/}"maestro_xctestrunner_xcodebuild_output* 2>/dev/null || true
    echo "$out" >> "$LOG"

    shot_file=$(find "$attempt_dir" -name "${shot}.png" -type f 2>/dev/null | head -1)
    if echo "$out" | grep -qE "FAILED|Could not|Timed out|Assertion|No space left" || [ -z "$shot_file" ]; then
      echo "  attempt ${attempt} failed" | tee -a "$LOG"
      mkdir -p "$OUT/${id}"
      fail_shot=$(find "$attempt_dir" -name '*.png' -type f 2>/dev/null | sort | tail -1)
      [ -n "$fail_shot" ] && cp "$fail_shot" "$OUT/${id}/${shot}_fail.png"
      rm -rf "$attempt_dir"
      sleep 5
      continue
    fi

    mkdir -p "$OUT/${id}"
    for suffix in "" "_c2" "_c3" "_c4"; do
      f=$(find "$attempt_dir" -name "${shot}${suffix}.png" -type f 2>/dev/null | head -1)
      [ -n "$f" ] && cp "$f" "$OUT/${id}/${shot}${suffix}.png"
    done
    ok=1
    rm -rf "$attempt_dir"
    break
  done

  if (( ok )); then
    pass=$((pass + 1))
    echo "  PASS" | tee -a "$LOG"
  else
    fail=$((fail + 1))
    xcrun simctl io "$UDID" screenshot "$OUT/${id}/${shot}_fail.png" >/dev/null 2>&1 || true
    echo "  FAIL" | tee -a "$LOG"
  fi
done

echo "=== simulator run done pass=${pass} fail=${fail} ===" | tee -a "$LOG"
