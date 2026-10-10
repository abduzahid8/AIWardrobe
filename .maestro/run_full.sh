#!/bin/bash
# Full-app E2E orchestration: self-contained flows plus deep-link entries
# that YAML cannot open itself. Read-only (never purchases, deletes,
# submits forms, signs out, or changes language).
# REQUIRES: Metro on :8081, Expo Go on the booted simulator, logged-IN session.
# Usage: ./.maestro/run_full.sh
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR/.."
DEV=$(cat /tmp/e2e_probe/new_udid.txt 2>/dev/null || echo 3776E6DF-F93F-40F4-AA8B-739E4B67AF42)
M="$HOME/.maestro/bin/maestro"
BASE_URL="exp://127.0.0.1:8081"
OUT=".maestro/artifacts/full_$(date +%s)"
mkdir -p "$OUT"
FAILS=()
run() {
  local out n
  out="$($M test ".maestro/$1" --no-reinstall-driver 2>&1)"
  printf '%s\n' "$out" > "$OUT/${1%.yaml}.log"
  n=$(printf '%s\n' "$out" | grep -cE "COMPLETED")
  echo "  steps completed: $n"
  printf '%s\n' "$out" | grep -E "FAILED|Invalid" | head -5
  # 0 completed steps means Maestro never ran the flow (driver/launch error) —
  # its output has no FAILED line either, so count it as a failure explicitly.
  if printf '%s\n' "$out" | grep -qE "FAILED|Invalid" || [ "$n" -eq 0 ]; then
    FAILS+=("$1")
    [ "$n" -eq 0 ] && printf '%s\n' "$out" | tail -8 | sed 's/^/      /'
  else
    echo "  $1: PASS"
  fi
}
reopen() {
  xcrun simctl terminate "$DEV" host.exp.Exponent >/dev/null 2>&1 || true
  sleep 2
  xcrun simctl openurl "$DEV" "$BASE_URL" >/dev/null 2>&1 || true
  sleep 22
}
# A reopen between flows isolates failures: a flow that dies mid-way leaves
# the app on an arbitrary screen (calendar sheet, edit modal, paywall) which
# would otherwise poison every subsequent flow's entry taps.
echo "=== 1/8 calendar round trip ==="
reopen
run full_calendar.yaml
echo "=== 2/8 trips segment ==="
reopen
run full_trips.yaml
echo "=== 3/8 edit sheet (cancel, no mutation) ==="
reopen
run full_edit_sheet.yaml
echo "=== 4/8 privacy via deep link ==="
reopen
xcrun simctl openurl "$DEV" "$BASE_URL/--/privacy"
sleep 10
run full_legal_privacy.yaml
echo "=== 5/8 terms via deep link ==="
reopen
xcrun simctl openurl "$DEV" "$BASE_URL/--/terms"
sleep 10
run full_legal_terms.yaml
echo "=== 6/8 stylist setup (read-only, never generates) ==="
reopen
run full_stylist_setup.yaml
echo "=== 7/8 AI source modes (wardrobe ↔ new, read-only) ==="
reopen
run full_ai_modes.yaml
echo "=== 8/8 paywall renders (no purchase) + restore ==="
reopen
run full_paywall.yaml
reopen
echo "=== full suite finished (app left on Home) ==="
echo "logs: $OUT"
if [ ${#FAILS[@]} -eq 0 ]; then
  echo "ALL 8 PASS"
else
  echo "FAILED (${#FAILS[@]}): ${FAILS[*]}"
  exit 1
fi
