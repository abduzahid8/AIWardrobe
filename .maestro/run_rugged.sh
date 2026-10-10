#!/bin/bash
# Rugged E2E orchestration: maestro flows + system-level steps that YAML
# cannot do (kill/reopen, deep-link open, appearance toggle, screenshots).
# System flows intentionally end in states YAML cannot leave (detail screen
# with no back UI, open keyboard covering the tab bar) — recovery between
# flows is terminate+reopen, which doubles as cold-start coverage.
# REQUIRES: Metro on :8081, Expo Go on the booted simulator, logged-IN session.
# Usage: ./.maestro/run_rugged.sh
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR/.."
DEV=$(cat /tmp/e2e_probe/new_udid.txt 2>/dev/null || echo 3776E6DF-F93F-40F4-AA8B-739E4B67AF42)
M="$HOME/.maestro/bin/maestro"
shot() { sleep 2; xcrun simctl io "$DEV" screenshot "/tmp/rugged_$1.png"; }
reopen() {
  xcrun simctl terminate "$DEV" host.exp.Exponent >/dev/null 2>&1 || true
  sleep 2
  xcrun simctl openurl "$DEV" "exp://127.0.0.1:8081" >/dev/null 2>&1 || true
  sleep 22
}
FAILS=()
run() {
  local out n
  out="$($M test ".maestro/$1" --no-reinstall-driver 2>&1)"
  n=$(printf '%s\n' "$out" | grep -cE "COMPLETED")
  echo "  steps completed: $n"
  printf '%s\n' "$out" | grep -E "FAILED|Invalid" | head -5
  if printf '%s\n' "$out" | grep -qE "FAILED|Invalid" || [ "$n" -eq 0 ]; then
    FAILS+=("$1")
    [ "$n" -eq 0 ] && printf '%s\n' "$out" | tail -8 | sed 's/^/      /'
  else
    echo "  $1: PASS"
  fi
}
echo "=== 1/7 detail_push (push + header-chevron back, ends on Closet) ==="
reopen
run rugged_detail_push.yaml
shot detail_pushed
echo "=== 2/7 recover + search (ends with keyboard up) ==="
reopen
run rugged_search.yaml
shot search_done
echo "=== 3/7 recover + same-tab mash ==="
reopen
run rugged_mash.yaml
echo "=== 4/7 dark appearance ==="
xcrun simctl ui "$DEV" appearance dark
sleep 4
shot dark_mode
run rugged_dark_mode.yaml
xcrun simctl ui "$DEV" appearance light
sleep 4
run rugged_dark_mode.yaml
shot light_restored
echo "=== 5/7 password-reset deep link ==="
xcrun simctl openurl "$DEV" "exp://127.0.0.1:8081/--/reset-password"
sleep 10
run rugged_deeplink.yaml
shot deeplink
echo "=== 6/7 kill + cold reopen persistence ==="
reopen
run rugged_relaunch.yaml
shot relaunched
echo "=== 7/7 AI setup mash (source toggle chaos, read-only) ==="
reopen
run rugged_ai_mash.yaml
shot ai_mashed
echo "=== rugged suite finished ==="
if [ ${#FAILS[@]} -eq 0 ]; then
  echo "ALL 7 PASS"
else
  echo "FAILED (${#FAILS[@]}): ${FAILS[*]}"
  exit 1
fi
