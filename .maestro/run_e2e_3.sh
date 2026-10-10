#!/usr/bin/env bash
# Chain runner · signup → delete → signup-again (same unique credentials).
#
# Each of the three tests is split into short Maestro flows because the iOS
# accessibility server degrades after ~5-9 hierarchy queries per flow and
# after ~20 runs per boot (see .maestro/README.md). This runner:
#   1. health-checks the driver (reboots the sim + relaunches Expo Go if dead)
#   2. generates the sign-up fill flow with this run's username
#   3. pbcopy's the unique email to the SIMULATOR clipboard before every flow
#      (the email field cannot be typed into — iOS keeps inserting domains)
#   4. runs the chain, stopping at the first failure with logs + artifacts
#
# Usage: .maestro/run_e2e_3.sh
set -u
cd "$(dirname "$0")/.."

MAESTRO="$HOME/.maestro/bin/maestro"
RUN=$(date +%s)
EMAIL="e2e.cycle${RUN}@aiwardrobe.club"
PASSWORD="Abcdef123!"
USER="e2ecyc${RUN}"
OUT=".maestro/artifacts/e2e3_${RUN}"
SCEN=".maestro/scenarios"

UDID=$(cat /tmp/e2e_probe/new_udid.txt 2>/dev/null || true)
if [ -z "${UDID:-}" ]; then
  UDID=$(xcrun simctl list devices booted -j | python3 -c 'import json,sys;d=json.load(sys.stdin);print(next(dev["udid"] for devs in d["devices"].values() for dev in devs if dev["state"]=="Booted"))' 2>/dev/null || true)
fi
if [ -z "${UDID:-}" ]; then echo "FATAL: no booted simulator"; exit 1; fi

mkdir -p "$OUT"
printf 'RUN=%s\nEMAIL=%s\nUSERNAME=%s\nPASSWORD=%s\n' "$RUN" "$EMAIL" "$USER" "$PASSWORD" > /tmp/e2e_probe/last_account.txt
chmod 600 /tmp/e2e_probe/last_account.txt

# Generate the fill flows with this run's username (env interpolation is
# broken in Maestro 2.8.0 — values must be baked into the YAML). Two halves:
# A types the username and pastes the PASSWORD, B pastes the EMAIL — the
# clipboard can only hold one value, and neither email nor password may be
# typed (domain-insert bug / lossy secure-field inputText).
sed -e "s/__USER__/${USER}/" -e "s/__PASSWORD__/${PASSWORD}/" \
    "$SCEN/_e2e_signup_fill_a.template.yaml" > "$SCEN/_e2e_signup_fill_a.yaml"
sed -e "s/__USER__/${USER}/" -e "s/__EMAIL__/${EMAIL}/" \
    "$SCEN/_e2e_signup_fill_b.template.yaml" > "$SCEN/_e2e_signup_fill_b.yaml"

# Write BOTH pasteboards: the simulator re-syncs the macOS pasteboard over
# the sim one on focus changes, which once clobbered our address with a note
# the user had copied on the host mid-run.
pbcopy_val() {
  printf '%s' "$1" | pbcopy
  printf '%s' "$1" | xcrun simctl pbcopy "$UDID"
}
pbcopy_email() { pbcopy_val "$EMAIL"; }

run_flow() {  # $1 = label, $2 = flow path, $3 = optional clipboard value
  pbcopy_val "${3:-$EMAIL}"
  "$MAESTRO" --udid "$UDID" test "$2" \
    --no-reinstall-driver --test-output-dir "$OUT/$1" \
    >"$OUT/$1.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "FAIL  $1  (log: $OUT/$1.log)"
    tail -12 "$OUT/$1.log" | sed 's/^/      /'
    return 1
  fi
  echo "PASS  $1"
  return 0
}

reboot_sim() {
  echo "      driver dead → rebooting simulator $UDID"
  xcrun simctl shutdown "$UDID" >/dev/null 2>&1
  xcrun simctl boot "$UDID" >/dev/null 2>&1
  xcrun simctl bootstatus "$UDID" -b >/dev/null 2>&1
  sleep 5
  xcrun simctl openurl "$UDID" "exp://127.0.0.1:8081" >/dev/null 2>&1
  sleep 40
  # Fresh Expo Go boots can land on the developer menu (intro tooltip) which
  # blanks the app hierarchy for the health assert — dismiss it if present.
  "$MAESTRO" --udid "$UDID" test "$SCEN/_e2e_devmenu.yaml" \
    --no-reinstall-driver --test-output-dir "$OUT/devmenu" >/dev/null 2>&1 || true
}

# ── Health check ──────────────────────────────────────────────────────────────
# Forced reboot FIRST: keystroke delivery degrades with runs-per-boot
# (observed 13/13 → 6/13 → 1/13 chars) — signup typing must run early
# in every boot window.
reboot_sim
if ! run_flow health "$SCEN/e2e_health.yaml"; then
  reboot_sim
  run_flow health "$SCEN/e2e_health.yaml" || { echo "FATAL: driver unhealthy after reboot"; exit 1; }
fi

# ── Test 01 · sign up ─────────────────────────────────────────────────────────
run_flow 01a_prepare   "$SCEN/e2e_prepare_signin.yaml"   || exit 1
run_flow 01b_signup_a  "$SCEN/e2e_signup_form_a.yaml" "$PASSWORD" || exit 1
if ! run_flow 01b_signup_b  "$SCEN/e2e_signup_form_b.yaml"; then
  echo "retry 01b_signup_b with fresh pbcopy"
  run_flow 01b_signup_b  "$SCEN/e2e_signup_form_b.yaml" || exit 1
fi
run_flow 01c_quiz_a    "$SCEN/e2e_quiz_a.yaml"           || exit 1
run_flow 01d_quiz_b    "$SCEN/e2e_quiz_b.yaml"           || exit 1

# Fresh boot before test 03 — its signup typing must also run early.
reboot_sim
run_flow health "$SCEN/e2e_health.yaml" || { echo "FATAL: unhealthy mid-chain"; exit 1; }

# ── Test 02 · delete account + reject old credentials ─────────────────────────
run_flow 02_delete     "$SCEN/e2e_02_delete.yaml"        || exit 1
run_flow 02b_prepare   "$SCEN/e2e_02b_prepare.yaml"        || exit 1
# Paste-window retry: pbcopy lands right before the flow, but the host
# pasteboard is user-contended — one fresh attempt if content-assert fails.
if ! run_flow 02b_reject "$SCEN/e2e_02b_reject_login.yaml"; then
  echo "retry 02b_reject with fresh pbcopy"
  run_flow 02b_reject "$SCEN/e2e_02b_reject_login.yaml" || exit 1
fi

# 02b deliberately triggers console.errors (rejected login) which queue RN
# LogBox notifications that overlay footer taps in test 03 — a full app
# reload resets the JS context and empties that queue.
xcrun simctl terminate "$UDID" host.exp.Exponent >/dev/null 2>&1 || true
sleep 2
xcrun simctl openurl "$UDID" "exp://127.0.0.1:8081" >/dev/null 2>&1 || true
sleep 22

# ── Test 03 · sign up again with the SAME credentials ─────────────────────────
run_flow 03a_prepare   "$SCEN/e2e_prepare_signin.yaml"   || exit 1
run_flow 03b_signup_a  "$SCEN/e2e_signup_form_a.yaml" "$PASSWORD" || exit 1
if ! run_flow 03b_signup_b  "$SCEN/e2e_signup_form_b.yaml"; then
  echo "retry 03b_signup_b with fresh pbcopy"
  run_flow 03b_signup_b  "$SCEN/e2e_signup_form_b.yaml" || exit 1
fi
run_flow 03c_quiz_a    "$SCEN/e2e_quiz_a.yaml"           || exit 1
run_flow 03d_quiz_b    "$SCEN/e2e_quiz_b.yaml"           || exit 1

echo "────────────────────────────────────────────"
echo "CHAIN GREEN · RUN=$RUN · artifacts: $OUT"
echo "credentials: /tmp/e2e_probe/last_account.txt"
