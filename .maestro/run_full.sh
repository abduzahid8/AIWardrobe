#!/bin/bash
# Full-app E2E orchestration: self-contained flows plus deep-link entries
# that YAML cannot open itself. Read-only (never purchases, deletes,
# submits forms, signs out, or changes language).
# REQUIRES: Metro on :8081, Expo Go on the booted simulator, logged-IN session.
# Usage: ./.maestro/run_full.sh
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR/.."
DEV=3776E6DF-F93F-40F4-AA8B-739E4B67AF42
M="$HOME/.maestro/bin/maestro"
BASE_URL="exp://127.0.0.1:8081"
run() {
  local out
  out="$($M test ".maestro/$1" --no-reinstall-driver 2>&1)"
  echo "$out" | grep -cE "COMPLETED" | xargs echo "  steps completed:"
  echo "$out" | grep -E "FAILED|Invalid" | head -5
  echo "$out" | grep -qE "FAILED|Invalid" || echo "  $1: PASS"
}
reopen() {
  xcrun simctl terminate "$DEV" host.exp.Exponent >/dev/null 2>&1 || true
  sleep 2
  xcrun simctl openurl "$DEV" "$BASE_URL" >/dev/null 2>&1 || true
  sleep 22
}
echo "=== 1/9 calendar round trip ==="
run full_calendar.yaml
echo "=== 2/9 insights round trip ==="
run full_insights.yaml
echo "=== 3/9 trips segment ==="
run full_trips.yaml
echo "=== 4/9 edit sheet (cancel, no mutation) ==="
run full_edit_sheet.yaml
echo "=== 5/9 privacy via deep link ==="
xcrun simctl openurl "$DEV" "$BASE_URL/--/privacy"
sleep 10
run full_legal_privacy.yaml
echo "=== 6/9 terms via deep link ==="
xcrun simctl openurl "$DEV" "$BASE_URL/--/terms"
sleep 10
run full_legal_terms.yaml
echo "=== 7/9 stylist setup (read-only, never generates) ==="
run full_stylist_setup.yaml
echo "=== 8/9 AI source modes (wardrobe ↔ new, read-only) ==="
run full_ai_modes.yaml
echo "=== 9/9 paywall renders (no purchase) + restore ==="
reopen
run full_paywall.yaml
reopen
echo "=== full suite finished (app left on Home) ==="
