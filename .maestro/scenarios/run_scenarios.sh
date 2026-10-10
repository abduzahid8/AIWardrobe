#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# AIWardrobe — 5 full-app E2E scenarios (Maestro, Expo Go).
#
#   ./run_scenarios.sh                 run all 5 scenarios
#   ./run_scenarios.sh --scenario 4    run a single scenario
#   ./run_scenarios.sh --no-reset      keep local storage (no fresh-install state)
#   ./run_scenarios.sh --list          list the scenarios and their steps
#
# Each scenario is a short list of flow files run as SEPARATE `maestro test`
# invocations — the iOS accessibility server degrades inside long flows, so
# splitting is what makes this reliable (see .maestro/README.md).
#
# Before the suite the runner wipes this app's AsyncStorage inside Expo Go
# (language choice, style-quiz state, Supabase session). That gives every run
# the same deterministic starting point: first-run language gate → Sign In →
# style quiz. The Supabase session and trial date are restored by the app
# itself on the next sign-in, so nothing server-side is lost.
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

UDID="${MAESTRO_UDID:-3776E6DF-F93F-40F4-AA8B-739E4B67AF42}"
EXP_URL="${EXP_URL:-exp://127.0.0.1:8081}"
METRO_STATUS_URL="${METRO_STATUS_URL:-http://localhost:8081/status}"
BUNDLE_ID="host.exp.Exponent"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ART="$SCRIPT_DIR/artifacts"
LOGS="$ART/logs"

# ── Scenario definitions (step = flow:<file> | link:<deep-link-path>) ────────
SCENARIO_NAMES=(
  "1 · Auth: validation, links, sign-in"
  "2 · Onboarding: style quiz → Home"
  "3 · Core tabs: Home/Closet/Inspo/Profile"
  "4 · AI stylist: setup + 2 real generations"
  "5 · Profile: paywall, insights, legal, sign-out"
)
SCENARIO_STEPS=(
  "flow:s1_auth_validation.yaml flow:s1_auth_links.yaml flow:s1_sign_in.yaml"
  "flow:s2_quiz_progress.yaml flow:s2_quiz_finish.yaml"
  "flow:s3_tabs_closet.yaml flow:s3_tabs_inspo_profile.yaml"
  "flow:s4_stylist_setup.yaml flow:s4_stylist_generate.yaml flow:s4_home_generate.yaml"
  "flow:s5_paywall_insights.yaml link:privacy flow:s5_legal_privacy.yaml link:terms flow:s5_legal_terms.yaml flow:s5_sign_out.yaml"
)

SCENARIOS_TO_RUN=()
RESET=1

usage() {
  sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  echo "Scenarios:"
  for i in "${!SCENARIO_NAMES[@]}"; do
    echo "  $((i + 1)). ${SCENARIO_NAMES[$i]}"
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scenario) SCENARIOS_TO_RUN+=("$2"); shift 2 ;;
    --scenario=*) SCENARIOS_TO_RUN+=("${1#*=}"); shift ;;
    --no-reset) RESET=0; shift ;;
    --list)
      for i in "${!SCENARIO_NAMES[@]}"; do
        echo "$((i + 1)). ${SCENARIO_NAMES[$i]}"
        echo "   steps: ${SCENARIO_STEPS[$i]}"
      done
      exit 0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

[[ ${#SCENARIOS_TO_RUN[@]} -eq 0 ]] && SCENARIOS_TO_RUN=(1 2 3 4 5)

# ── Preflight ────────────────────────────────────────────────────────────────
command -v maestro >/dev/null 2>&1 || { echo "FATAL: maestro CLI not found" >&2; exit 2; }
curl -sf "$METRO_STATUS_URL" >/dev/null 2>&1 || {
  echo "FATAL: Metro bundler is not reachable at $METRO_STATUS_URL" >&2
  echo "       start it with: npx expo start --port 8081" >&2
  exit 2
}
xcrun simctl list devices booted | grep -q "$UDID" || {
  echo "FATAL: simulator $UDID is not booted" >&2
  exit 2
}

mkdir -p "$LOGS"

# ── Fresh-install state ──────────────────────────────────────────────────────
# Only this app's AsyncStorage is removed (detected by this app's own keys),
# so other Expo Go projects on the simulator keep their data.
# Order matters: terminate FIRST, otherwise the running app flushes its
# in-memory state back to disk after the wipe.
reset_state() {
  echo "── Resetting local app state (AsyncStorage wipe) …"
  local data
  data="$(xcrun simctl get_app_container "$UDID" "$BUNDLE_ID" data 2>/dev/null || true)"
  if [[ -z "$data" || ! -d "$data" ]]; then
    echo "   WARNING: could not resolve Expo Go data container — skipping wipe"
    return
  fi
  xcrun simctl terminate "$UDID" "$BUNDLE_ID" >/dev/null 2>&1 || true
  sleep 2
  # A freshly-reset install does NOT re-create every key (e.g. `style-preferences`
  # only appears once the quiz has run), so match on any key this app writes.
  local manifest wiped=0
  for manifest in "$data"/Documents/ExponentExperienceData/*/*/RCTAsyncLocalStorage/manifest.json; do
    [[ -f "$manifest" ]] || continue
    if grep -qE '(@app_language|style-preferences|wardrobe-storage|sb-)' "$manifest" 2>/dev/null; then
      rm -rf "$(dirname "$manifest")"
      echo "   wiped: ${manifest#"$data"/Documents/ExponentExperienceData/}"
      wiped=1
    fi
  done
  [[ $wiped -eq 0 ]] && echo "   nothing to wipe (already a fresh install)"
  sleep 1
  xcrun simctl openurl "$UDID" "$EXP_URL" >/dev/null
  echo "   relaunching $EXP_URL …"
  sleep 30
}

# ── Steps ────────────────────────────────────────────────────────────────────
run_flow() {
  local file="$1" scenario="$2" tag="$3"
  # --output takes a FILE path for JUNIT reports, not a directory.
  local out="$ART/$scenario/${tag}_report.xml" log="$LOGS/${scenario}_${tag}.log"
  mkdir -p "$ART/$scenario"
  echo "   ▸ flow   $file"
  ( cd "$SCRIPT_DIR" && maestro --udid "$UDID" test "$file" \
      --no-reinstall-driver --format JUNIT --output "$out" \
      --test-output-dir "$ART/$scenario/$tag" ) >"$log" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "     FAILED (exit $rc) — see $log"
    return 1
  fi
  echo "     ok"
  return 0
}

open_link() {
  local path="$1" scenario="$2" tag="$3"
  echo "   ▸ link   $EXP_URL/--/$path"
  xcrun simctl openurl "$UDID" "$EXP_URL/--/$path" >/dev/null
  sleep 15
  return 0
}

# ── Main ─────────────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════════"
echo " AIWardrobe E2E · scenarios: ${SCENARIOS_TO_RUN[*]}"
echo " device: $UDID · target: $EXP_URL · reset: $RESET"
echo "════════════════════════════════════════════════════════════════"

[[ $RESET -eq 1 ]] && reset_state

FAILURES=()
RESULTS=()

for n in "${SCENARIOS_TO_RUN[@]}"; do
  idx=$((n - 1))
  [[ $idx -lt 0 || $idx -ge ${#SCENARIO_NAMES[@]} ]] && { echo "Unknown scenario: $n" >&2; exit 2; }
  echo
  echo "── Scenario $n · ${SCENARIO_NAMES[$idx]}"
  ok=1
  step_no=0
  for step in ${SCENARIO_STEPS[$idx]}; do
    step_no=$((step_no + 1))
    kind="${step%%:*}"
    value="${step#*:}"
    tag="$(printf '%02d_%s' "$step_no" "$(basename "$value" .yaml 2>/dev/null || echo "$value")")"
    if [[ "$kind" == "flow" ]]; then
      run_flow "$value" "$n" "$tag" || { ok=0; FAILURES+=("scenario $n · $value"); }
    elif [[ "$kind" == "link" ]]; then
      open_link "$value" "$n" "$tag"
    else
      echo "   UNKNOWN step kind: $step" >&2; ok=0
    fi
  done
  if [[ $ok -eq 1 ]]; then
    RESULTS+=("PASS  scenario $n · ${SCENARIO_NAMES[$idx]}")
  else
    RESULTS+=("FAIL  scenario $n · ${SCENARIO_NAMES[$idx]}")
  fi
done

echo
echo "════════════════════════════════════════════════════════════════"
echo " SUMMARY"
echo "════════════════════════════════════════════════════════════════"
for r in "${RESULTS[@]}"; do echo "  $r"; done
if [[ ${#FAILURES[@]} -gt 0 ]]; then
  echo
  echo " Failing steps:"
  for f in "${FAILURES[@]}"; do echo "   - $f"; done
  echo
  echo " Logs: $LOGS"
  exit 1
fi
echo
echo " All green. Artifacts: $ART"
