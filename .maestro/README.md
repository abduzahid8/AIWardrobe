# Maestro E2E Testing Guide

## Canonical runner

Maestro 2.8.0 is the canonical E2E runner for this repo (`npm run test:e2e` /
`npm run test:maestro`). WebdriverIO (`wdio.conf.ts`, `npm run test:wdio`) is
legacy only — it points at a `test/specs/e2e.placeholder.ts` stub so the
command doesn't crash; real coverage lives in `.maestro/*.yaml`.

No `JAVA_HOME` setup is needed: Maestro 2.8.0 works with it unset. If you do
set it, point at an installed JDK (`openjdk` / `openjdk@26` on this machine —
`openjdk@17` is NOT installed). `run_all.sh` auto-detects and otherwise
leaves it unset.

## Setup (Expo Go)

```bash
# 1. Start Metro bundler
npx expo start --port 8081

# 2. Pre-load the app in Expo Go via the exp:// URL (NO launchApp in flows)
# 3. Run a flow — always with --no-reinstall-driver
maestro test .maestro/expo_go_smoke.yaml --no-reinstall-driver
```

Or run the short tab halves back-to-back:

```bash
maestro test .maestro/expo_go_tabs_a.yaml .maestro/expo_go_tabs_b.yaml --no-reinstall-driver
```

Or use the legacy full runner (native bundle id, builds + runs):

```bash
./scripts/run_maestro_tests.sh
```

## Expo Go flows (current stack: iOS 27 + RN 0.86, Maestro 2.8.0)

| Flow | What it does | Precondition | Notes |
|------|-------------|--------------|-------|
| `expo_go_smoke.yaml` | Sign In renders | Logged OUT | Anchor: `Continue with Google` (the `Sign In` title also matches the iOS back-button label; the footer muted prefix `Don't have an account? ` is a nested raw-string Text node iOS does not expose since the Sign In redesign) |
| `expo_go_auth_validation.yaml` | Empty-submit shows native alert | Logged OUT, empty fields | Expects native alert `Please enter email and password` + `OK` — NOT inline errors |
| `expo_go_auth_navigation.yaml` | Sign In → Forgot Password → back → Sign Up | Logged OUT | Back chevron label is `Go back`; do NOT tap the top-left status-bar corner (iOS "back to Safari" exits Expo Go) |
| `expo_go_quiz_complete.yaml` | Quiz summary → main app | Quiz summary visible (`style plan is ready`) | Helper: taps `Unlock my plan` |
| `expo_go_tabs_a.yaml` | Home → Closet | Logged IN | Ends on Closet (`Tops`) |
| `expo_go_tabs_b.yaml` | Inspo → Profile → Home | Logged IN | Ends on Home (`Good morning/afternoon/evening/night`) |
| `expo_go_language_selection.yaml` | First-run gate → Sign In | Fresh install (gate visible) | Tap-to-advance (no Continue button); selects English; no-ops if gate already completed; dismisses ATT dialog conditionally |
| `language_selection.yaml` | Gate exclusivity + tap-to-advance to RU Sign In (native) | `clearState: true` (fresh) | Asserts Sign In NOT visible on gate; taps `Русский`, lands on Russian Sign In |

Logged-out vs logged-in state matters: if a previous run left an account
signed in (e.g. a manual registration mid-session), sign out first
(Profile tab → Sign Out) or reinstall Expo Go on the simulator before
re-running the auth flows.

## Working Flows (legacy native bundle)

| Flow | What it does | Notes |
|------|-------------|-------|
| `nav_home.yaml` | Launch + tap Home tab | Requires app running or will hang |
| `nav_closet.yaml` | Launch + tap Closet tab | Same |
| `nav_inspo.yaml` | Launch + tap Inspo tab | Same |
| `nav_profile.yaml` | Launch + tap Profile tab | Same |
| `nav_closet_inspo.yaml` | Home → Closet → Inspo | 2 taps in one flow |
| `nav_3tabs.yaml` | Home → Closet → Inspo → Profile | 3 taps |
| `nav_4tabs_nolaunch.yaml` | All 4 tabs, no launchApp | Preferred - app must be pre-launched |
| `nav_4tabs_screenshots.yaml` | All 4 tabs + screenshots | Captures each tab state |

## How It Works

### Current stack (iOS 27 + RN 0.86): text selectors work

On the old stack (iOS 26.1 + RN 0.81.5) the Accessibility Server returned
invalid element frames (`kAXErrorInvalidUIElement`) for RN views, which
crashed Maestro's `viewHierarchy` queries — so all interactions had to use
percentage coordinates and `launchApp` on cold start would hang the XCUITest
driver.

That no longer reproduces on iOS 27 + RN 0.86: the `expo_go_*` runs prove
`tapOn: "<text>"` / `assertVisible: ".*...*"` work reliably. New flows should
prefer text selectors; the legacy `nav_*` coordinate flows are kept for
reference only.

### Stability limit (still real)

Approximately **5 hierarchy queries per flow** before the accessibility
server degrades. A 20-step sweep failed on its last tap, while the same steps
split into halves (`expo_go_tabs_a` ~14s, `expo_go_tabs_b` ~9s) pass
reliably. Keep flows short; prefer two small flows over one long sweep.
After ~20 runs the driver degrades globally (even visible text stops
matching) — `xcrun simctl shutdown + boot` restores it.

### Direction swipes don't scroll (harness limit, not app bug)

`swipe: {direction, duration}` COMPLETES but never moves `ScrollView` /
`FlatList` content on this stack (verified on Home + Profile, all durations
200–1500ms, including after a sim reboot with a fresh driver). Taps,
`inputText` and text assertions work fine. Consequences for flow design:
- Never assert post-swipe positions; use swipes only as no-crash smoke.
- Reach below-fold screens via deep links instead (see `run_full.sh` +
  `privacy`/`terms` linking map entries in `App.tsx`), or via visible
  entry points (banners, menu buttons).
- text `tapOn` finds on-screen elements reliably; offscreen rows in long
  lists may be absent from the tree — prefer labeled back chevrons
  (`common.back` = "Назад"/"Back") for returns.

### Disk space

A full macOS disk breaks the harness first (`No space left on device`
in driver extraction) and degrades the sim second. Before a session,
clear your own footprint: old `~/.maestro/tests/*` runs and `/tmp` shots.
Never touch other projects' `DerivedData`. This repo's Xcode
`DerivedData/AIWardrobe-*` + `ModuleCache.noindex` are regenerable.

### Tab coordinates (legacy coordinate flows only)

All legacy interactions use percentage-based coordinates:

```yaml
- tapOn:
    point: "37%,94%"
```

Tab positions on iPhone 17 Pro (402×874pt):
| Tab | X% | Y% |
|-----|----|----|
| Home | 12% | 94% |
| Closet | 37% | 94% |
| Inspo | 62% | 94% |
| Profile | 87% | 94% |

### Stability Limit (legacy note — see "Stability limit (still real)" above)

Approximately **5 `waitForAnimationToEnd` calls per flow** before the accessibility server corrupts. Beyond that, simulator needs a full erase + reboot.

### Key Rule: Pre-launch the App (legacy native flows)

`launchApp` in Maestro calls `XCUIApplication().launch()`, which queries the accessibility hierarchy and **hangs** when the app isn't running. To avoid this:

1. Launch the app manually: `xcrun simctl launch booted org.name.AIWardrobe`
2. Use flows **without** `launchApp` (e.g., `nav_4tabs_nolaunch.yaml`)
3. Or use flows WITH `launchApp` but only after the app is already running

## What Can't Be Tested (legacy native flows)

- **`launchApp` on cold start**: hangs the XCUITest driver
- **More than ~5 hierarchy queries per flow**: accessibility server corruption
- **Scrolling**: requires element detection to know when scroll completes

(On the current Expo Go stack, text-based matching works — see above. The
legacy `nav_*` coordinate flows predate that fix.)

## Quick Start

```bash
# Terminal 1: Start Metro
npx expo start --port 8081

# Wait for "Waiting on http://localhost:8081"
# Terminal 2: Launch app
xcrun simctl launch booted org.name.AIWardrobe

# Wait 10s for app to render
# Terminal 3: Run tests
cd .maestro
maestro test nav_4tabs_screenshots.yaml --no-reinstall-driver
```

## Tab Coordinate Verification

If the app layout changes (e.g., different device, tab bar redesign), re-verify coordinates:
1. Take a screenshot: `xcrun simctl io booted screenshot screenshot.png`
2. Open in Preview, measure the tab bar area
3. Update coordinates in the flows

## Session learnings (2026-10-09)

- **iOS "Save Password?" sheet hides the app AX tree entirely** — its text
  never appears in the hierarchy, so `when visible "Save Password"` can
  never fire while it blocks. Dismiss with a blind coordinate tap
  (`32%,63%` = "Not Now" on iPhone 16 Pro) while the quiz is not visible —
  see the dismiss rounds in `_e2e_signup_fill_b.template.yaml`.
- **Never type the password and never trust a runner-side `pbcopy`** — the
  user's host pasteboard syncs into the simulator and wins races (observed:
  user-copied prose pasted into the email field). Both fills use the in-flow
  `setClipboard` command (`copyTextToClipboard` does not exist in 2.8.0)
  immediately before the long-press, and the runner retries `signup_b` once.
- **Casual occasion is collapsed** behind "More occasions" (testID
  `stylistMoreOccasions`, label flips to "Less" when expanded) — expand
  once before tapping Casual (`full_ai_modes.yaml`, `rugged_ai_mash.yaml`).
- **Every `maestro test` invocation leaks a ~190 MB
  `$TMPDIR/maestro_xctestrunner_*` dir** (47 GB accumulated over a day).
  The stylist-50 runner purges stale ones per iteration; purge manually if
  disk fills: `find "$TMPDIR" -maxdepth 1 -name 'maestro_xctestrunner_*' -mmin +3 -exec rm -rf {} +`
- **Fresh accounts have an empty closet** — `rugged_detail_push` /
  `rugged_search` detect "Your closet is empty" and assert the empty state
  instead of pushing/searching; full data coverage needs seeded items.
- **Reboots can land Expo Go on the developer menu** (intro tooltip) which
  blanks the app hierarchy — `run_e2e_3.sh` runs `_e2e_devmenu.yaml` after
  each reboot.
- **Treat "0 steps completed" as a failure** — Maestro output without a
  `FAILED` line usually means the driver/device was gone, not a pass
  (`run_full.sh`, `run_rugged.sh`).
- Weekly Insights screen ("Style Insights") was **removed** on 2026-10-09
  (screen, route, Profile button, notification payloads). `full_insights.yaml`
  is gone; the full suite is 8 flows.
