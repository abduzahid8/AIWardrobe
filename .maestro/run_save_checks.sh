#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# AIWardrobe — save-logic E2E (Maestro, Expo Go).
#
# Verifies, end to end on the iOS simulator:
#   1. SAVE OUTFIT   Home heart → addOutfit + saveOutfit('saved','home')
#                    → appears in Closet → Collections → Saved
#   2. UNSAVE OUTFIT modal Remove → unsaveOutfit → leaves the Saved list
#                    → outfit.saved flips false in AsyncStorage
#   3. SAVE CLOTHING Closet Upload → AI Studio Photo → classify (live
#                    ai-process edge fn) → editor Save → addItem
#                    → clothing_items row on Supabase + item in the grid
#   4. PERSISTENCE   cold relaunch → clothing item still in the closet
#
# "Logically correct" is proven at three layers:
#   - UI:  the card appears / disappears, the alert fires, the grid fills
#   - DB:  clothing_items count delta via PostgREST (RLS-scoped user token)
#   - Storage: outfit.saved flag inside wardrobe-storage after app death
#
# REQUIRES: Metro on :8081, Expo Go on the booted simulator, signed-in
# e2e.stylist session. The runner does NOT sign in — do that once via
# .maestro/scenarios/_sign_in.yaml (or the AsyncStorage session-inject
# used during initial setup).
#
# STEP 0 seeds 5 catalog essentials when the account has fewer, because Home
# only swaps the essentials grid for the swipe deck at items.length >= 5.
#
# Usage: ./.maestro/run_save_checks.sh
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR/.."

MAESTRO="$HOME/.maestro/bin/maestro"
BUNDLE_ID="host.exp.Exponent"
EXP_URL="exp://127.0.0.1:8081"
RUN=$(date +%s)
OUT=".maestro/artifacts/save_checks_${RUN}"
mkdir -p "$OUT"

# ── Device ───────────────────────────────────────────────────────────────────
UDID=$(cat /tmp/e2e_probe/new_udid.txt 2>/dev/null || true)
if [[ -z "${UDID:-}" ]]; then
  UDID=$(xcrun simctl list devices booted -j 2>/dev/null | python3 -c '
import json,sys
d=json.load(sys.stdin)
for devs in d["devices"].values():
    for dev in devs:
        if dev["state"]=="Booted":
            print(dev["udid"]); sys.exit()
' 2>/dev/null || true)
fi
[[ -z "${UDID:-}" ]] && { echo "FATAL: no booted simulator"; exit 1; }

# ── Preflight ────────────────────────────────────────────────────────────────
curl -sf http://localhost:8081/status >/dev/null 2>&1 || {
  echo "FATAL: Metro not reachable on :8081 — start it with: npx expo start --port 8081"
  exit 1
}

echo "════════════════════════════════════════════════════════════════"
echo " AIWardrobe save-logic E2E"
echo " device: $UDID · target: $EXP_URL · artifacts: $OUT"
echo "════════════════════════════════════════════════════════════════"

# ── Helpers ──────────────────────────────────────────────────────────────────
screenshot() { xcrun simctl io "$UDID" screenshot "$OUT/$1.png" >/dev/null 2>&1 || true; }

reopen() {
  xcrun simctl terminate "$UDID" "$BUNDLE_ID" >/dev/null 2>&1 || true
  sleep 2
  xcrun simctl openurl "$UDID" "$EXP_URL" >/dev/null 2>&1 || true
  sleep 25
  # Fresh Expo Go boots can land on the developer menu — dismiss if present.
  "$MAESTRO" --udid "$UDID" test .maestro/scenarios/_e2e_devmenu.yaml \
    --no-reinstall-driver >/dev/null 2>&1 || true
}

FAILS=()
run_flow() {
  local file="$1" tag="$2"
  echo "── $tag ($file)"
  local log="$OUT/${tag}.log"
  "$MAESTRO" --udid "$UDID" test ".maestro/$file" \
    --no-reinstall-driver --test-output-dir "$OUT/${tag}_shots" >"$log" 2>&1
  local rc=$?
  local n
  n=$(grep -c "COMPLETED" "$log" 2>/dev/null || echo 0)
  echo "   steps completed: $n"
  if [[ $rc -ne 0 ]] || [[ "$n" -eq 0 ]]; then
    echo "   FAILED — last lines:"
    grep -E "FAILED|Invalid|Error" "$log" | head -5 | sed 's/^/     /'
    tail -6 "$log" | sed 's/^/     /'
    FAILS+=("$tag")
    return 1
  fi
  echo "   PASS"
  return 0
}

# ── Storage / DB probes ──────────────────────────────────────────────────────
# outfit.saved lives ONLY in AsyncStorage (the outfits array is not synced —
# saved_outfits is a separate OOTD table). Read it straight off the container.
storage_dir() {
  local data
  data=$(xcrun simctl get_app_container "$UDID" "$BUNDLE_ID" data 2>/dev/null || true)
  [[ -z "$data" ]] && return 1
  # Expo Go may nest under @scope/name or a bare project dir.
  local dir
  dir=$(find "$data/Documents/ExponentExperienceData" -path "*RCTAsyncLocalStorage" -type d 2>/dev/null | head -1)
  [[ -z "$dir" ]] && return 1
  printf '%s' "$dir"
}

outfit_saved_counts() {
  local dir
  dir=$(storage_dir) || { echo "ERR no-storage"; return; }
  python3 - "$dir/manifest.json" <<'PY'
import json, sys
try:
    m = json.load(open(sys.argv[1]))
except Exception as e:
    print("ERR", e); sys.exit()
raw = m.get("wardrobe-storage")
if not raw:
    print("0 0"); sys.exit()
try:
    st = json.loads(raw).get("state", {})
except Exception:
    print("0 0"); sys.exit()
outfits = st.get("outfits") or []
saved = [o for o in outfits if o.get("saved")]
print(f"{len(outfits)} {len(saved)}")
PY
}

clothing_count() {
  python3 - "$1" <<'PY'
import json, sys, urllib.request
env = {}
for line in open('.env'):
    line = line.strip()
    if line and not line.startswith('#') and '=' in line:
        k, _, v = line.partition('=')
        env[k.strip()] = v.strip()
URL = env.get('EXPO_PUBLIC_SUPABASE_URL', '')
KEY = env.get('EXPO_PUBLIC_SUPABASE_ANON_KEY', '')
try:
    auth = json.load(open('/tmp/save_checks/auth_stylist.json'))
except Exception:
    print('NO_AUTH'); sys.exit()
TOKEN = auth['access_token']
req = urllib.request.Request(
    URL + '/rest/v1/clothing_items?select=id',
    headers={'apikey': KEY, 'Authorization': 'Bearer ' + TOKEN,
             'Prefer': 'count=exact', 'Range-Unit': 'items'},
)
try:
    with urllib.request.urlopen(req, timeout=30) as r:
        cr = r.headers.get('content-range', '')
        print(int(cr.split('/')[1]) if '/' in cr else -1)
except Exception as e:
    print('ERR', e)
PY
}

# ═════════════════════════════════════════════════════════════════════════════
# 0 · Seed — Home renders Wardrobe Essentials instead of the swipe deck while
#      items.length < 5 (HomeScreen.tsx:2041), so "Save outfit" cannot exist.
#      The catalog holds exactly 5 essentials; seed_essentials_a/b add them.
# ═════════════════════════════════════════════════════════════════════════════
echo "═══ 0/6 seed essentials ═══"
PRE_ITEMS=$(clothing_count)
echo "   clothing_items before seed: $PRE_ITEMS"
if [[ "$PRE_ITEMS" =~ ^[0-9]+$ ]] && [[ "$PRE_ITEMS" -lt 5 ]]; then
  echo "   fewer than 5 items — the deck is gated off, seeding..."
  reopen   # fresh boot → Home at the top (seed flows scroll DOWN only)
  run_flow seed_essentials_a.yaml 00_seed_a || true
  run_flow seed_essentials_b.yaml 00_seed_b || true
  sleep 5
  SEED_ITEMS=$(clothing_count)
  echo "   clothing_items after seed: $SEED_ITEMS"
  if ! [[ "$SEED_ITEMS" =~ ^[0-9]+$ ]] || [[ "$SEED_ITEMS" -lt 5 ]]; then
    echo "FATAL: seeding left $SEED_ITEMS items — the swipe deck (and its"
    echo "       Save outfit heart) cannot render with fewer than 5."
    exit 1
  fi
  echo "   PASS: deck prerequisites in place"
else
  echo "   already >= 5 items — skipping seed"
fi

# ═════════════════════════════════════════════════════════════════════════════
# 1 · Baseline
# ═════════════════════════════════════════════════════════════════════════════
echo "═══ 1/6 baseline ═══"
# The app must be dead before we read storage, or a late persist write
# can race the snapshot.
xcrun simctl terminate "$UDID" "$BUNDLE_ID" >/dev/null 2>&1 || true
sleep 3
BASE_OUTFITS=$(outfit_saved_counts)
BASE_ITEMS=$(clothing_count)
echo "   outfits(total saved): $BASE_OUTFITS"
echo "   clothing_items:       $BASE_ITEMS"
[[ "$BASE_ITEMS" == NO_AUTH || "$BASE_ITEMS" == ERR* ]] && {
  echo "FATAL: cannot reach Supabase as the e2e user (see /tmp/save_checks/auth_stylist.json)"
  exit 1
}
read -r BASE_OUTFIT_N BASE_SAVED_N <<< "$BASE_OUTFITS"
reopen

# ═════════════════════════════════════════════════════════════════════════════
# 2 · Save outfit from Home
# ═════════════════════════════════════════════════════════════════════════════
run_flow save_outfit_home.yaml 02_save_outfit || true
screenshot after_save_outfit

# Storage check: a new saved=true outfit must exist.
xcrun simctl terminate "$UDID" "$BUNDLE_ID" >/dev/null 2>&1 || true
sleep 3
AFTER_OUTFITS=$(outfit_saved_counts)
read -r AFTER_OUTFIT_N AFTER_SAVED_N <<< "$AFTER_OUTFITS"
echo "── storage after save: total=$AFTER_OUTFIT_N saved=$AFTER_SAVED_N (was $BASE_OUTFIT_N/$BASE_SAVED_N)"
if [[ "$AFTER_SAVED_N" -le "$BASE_SAVED_N" ]]; then
  echo "   FAIL: no new saved=true outfit in wardrobe-storage"
  FAILS+=("storage_saved_outfit")
else
  echo "   PASS: saved outfit persisted locally"
fi
reopen

# ═════════════════════════════════════════════════════════════════════════════
# 3 · It shows up in Collections → Saved
# ═════════════════════════════════════════════════════════════════════════════
run_flow verify_saved_outfit.yaml 03_verify_saved || true
screenshot after_verify_saved

# ═════════════════════════════════════════════════════════════════════════════
# 4 · Unsave it again
# ═════════════════════════════════════════════════════════════════════════════
run_flow unsave_outfit.yaml 04_unsave_outfit || true
screenshot after_unsave

xcrun simctl terminate "$UDID" "$BUNDLE_ID" >/dev/null 2>&1 || true
sleep 3
POST_UNSURE=$(outfit_saved_counts)
read -r POST_UNSURE_N POST_SAVED_N <<< "$POST_UNSURE"
echo "── storage after unsave: total=$POST_UNSURE_N saved=$POST_SAVED_N"
# The outfit row stays (unsave only flips the flag) — saved count must drop.
if [[ "$POST_SAVED_N" -ge "$AFTER_SAVED_N" ]]; then
  echo "   FAIL: saved count did not drop after unsave ($POST_SAVED_N >= $AFTER_SAVED_N)"
  FAILS+=("storage_unsave_outfit")
else
  echo "   PASS: outfit flag flipped to saved=false"
fi
reopen

# ═════════════════════════════════════════════════════════════════════════════
# 5 · Save a clothing item via the real AI Studio pipeline
# ═════════════════════════════════════════════════════════════════════════════
run_flow save_clothing_photo.yaml 05_save_clothing || true
screenshot after_save_clothing

# Give the Supabase insert a moment (addItem is optimistic; the insert is
# awaited inside handleSave but RLS/replication lag can trail the alert).
sleep 8
NEW_ITEMS=$(clothing_count)
echo "── clothing_items after save: $NEW_ITEMS (was $BASE_ITEMS)"
if [[ "$NEW_ITEMS" =~ ^[0-9]+$ ]] && [[ "$BASE_ITEMS" =~ ^[0-9]+$ ]]; then
  if [[ "$NEW_ITEMS" -le "$BASE_ITEMS" ]]; then
    echo "   FAIL: no new clothing_items row on Supabase"
    FAILS+=("db_clothing_insert")
  else
    echo "   PASS: +$((NEW_ITEMS - BASE_ITEMS)) clothing_items row(s)"
  fi
else
  echo "   WARN: non-numeric counts ($BASE_ITEMS -> $NEW_ITEMS) — skipping delta check"
  FAILS+=("db_clothing_count_unreadable")
fi

# ═════════════════════════════════════════════════════════════════════════════
# 6 · Cold relaunch — clothing must still be there
# ═════════════════════════════════════════════════════════════════════════════
reopen
run_flow verify_clothing_persist.yaml 06_verify_persist || true
screenshot after_verify_persist

# ═════════════════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════════════════
echo "════════════════════════════════════════════════════════════════"
echo " clothing_items: $BASE_ITEMS -> $NEW_ITEMS"
echo " saved outfits:  $BASE_SAVED_N -> $AFTER_SAVED_N -> $POST_SAVED_N"
echo " artifacts: $OUT"
if [[ ${#FAILS[@]} -eq 0 ]]; then
  echo " ALL CHECKS PASS"
  exit 0
else
  echo " FAILED (${#FAILS[@]}): ${FAILS[*]}"
  exit 1
fi
