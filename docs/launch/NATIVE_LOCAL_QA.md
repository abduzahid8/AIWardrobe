# Isolated native simulator evidence

This supplement exercises a dedicated Release entry, not the normal AIWardrobe application root. It must not close the four-device Tier A QA, normal-app startup, auth, billing, provider, or upgrade gates.

A new disposable iPhone 17 Pro simulator on iOS 26.5 was created for this task. Bundle ID `com.aiwardrobe.rclocal` isolates its app data from `com.aiwardrobe`. The harness imports the real beta-feedback component, BuyCheck, deterministic outfit enumerator and four-slot validator. It imports no App root, account/backend setup, billing service or analytics sink. It denies JS fetch/XHR and uses bundled local images. No credential or production session was copied into the simulator.

The arm64 simulator Release build succeeded with signing disabled. Native binary, Hermes bundle, entry and feedback-component hashes are in `evidence/native-harness-session.json`. Build command:

```sh
EXPO_NO_DOTENV=1 EXPO_OFFLINE=1 RCT_NO_LAUNCH_PACKAGER=1 \
ENTRY_FILE=/Users/v.zahid/AIWardrobe/scripts/validation/launch-rc-native-entry.tsx \
xcodebuild -workspace ios/AIWardrobe.xcworkspace -scheme AIWardrobe \
  -configuration Release -sdk iphonesimulator -destination 'generic/platform=iOS Simulator' \
  -derivedDataPath /tmp/aiwardrobe-rc-native CODE_SIGNING_ALLOWED=NO \
  CODE_SIGNING_REQUIRED=NO ONLY_ACTIVE_ARCH=YES ARCHS=arm64 \
  PRODUCT_BUNDLE_IDENTIFIER=com.aiwardrobe.rclocal build
```

Only install that isolated bundle on the task's disposable simulator, never over a user's normal app. `launch-rc-native-observe.py` refuses the normal bundle ID and checks the CoreSimulator container. Its benchmark measures process launch → first-layout JSON receipt, including CLI/receipt IO, with OS caches retained. Ten runs on the keyboard-fixed bundle are recorded in `evidence/native-isolated-startup.json`. These are isolated-fixture measurements, not AIWardrobe cold/warm startup or production claims. Each native launch receipt must report Release mode and zero JS network attempts.

```sh
python3 scripts/validation/launch-rc-native-observe.py benchmark \
  AD2C6D1C-096E-4D89-86B8-22D600E30C86 --runs 10 \
  --output docs/launch/evidence/native-isolated-startup.json
```

Native interaction initially waited for a manual Mac unlock. Once unlocked, the actual feedback form reproduced a keyboard obstruction: correction fields were below the keyboard. `KeyboardAvoidingView`, drag dismissal, Done actions and explicit dismissal on review/close fixed it. Native AX state and screenshots verified both before/after fields and Review remained visible while typing; review showed exact sanitized JSON; undo retained before/after with revertedAt; close/reopen cleared the draft and disabled Review until a helpful choice. No share action was invoked. Evidence: `native-feedback-ui.json`, `feedback-keyboard-before.png`, `feedback-keyboard-fixed.png`. Unit coverage handles account change while modal open; that specific native interaction was not simulated.

The native stress button was executed before and after the fix. The current bundle's raw `native-isolated-stress.json` records: 30 local original-JPEG copies; a 220-item native FlatList with repeated scrolling; 25 actual BuyChecks that rejected a hard-budget conflict and retained low fit confidence; 25 deterministic production outfitChoices calls validating four slots; 24 reference and 20 synthetic VTON-history image entries; native cache/file cleanup; zero decode errors and zero JS network attempts. The physical image-picker UI, production wardrobe screen/media-owner wrapper/thumbnail route and actual VTON provider/history component were not exercised. Local fixture imports are file copies, not a claim that the picker permission flow passed.

`native-isolated-memory.json` samples host RSS of the simulator app process at one-second intervals for 60 seconds. Peak RSS is high (about 719 MiB in the repeat run); final RSS fell below baseline after cleanup. This does not establish absence of leaks, physical-device safety, a production memory regression, or a comparative before/after performance improvement. The workload deliberately decodes original JPEG copies through the underlying Expo image library. Validate the production thumbnail route on an older iPhone and low/mid-range Android before GO. The initial and repeat raw receipts are retained separately; do not combine their timings without accounting for build and process state.

One native Home → App Library → same-process foreground round trip retained the successful stress receipt; appStates recorded inactive/background/active and no network attempt. Ten-cycle stability, physical OS pressure, VoiceOver speech, Android, upgrade, real auth, billing and provider paths remain manual. No memory-warning injection was available in the inspected Simulator Device/Features menus, so no OS memory-warning PASS is claimed.

For another run, inspect the raw receipt rather than trusting a prepared harness. The optional observer's memory mode requires the actual harness PID and records RSS with explicit simulator/heap limitations. Never share a native system share sheet from automated QA; review/cancel or use a separately authorized destination. Preserve raw receipts and screenshots before replacing a build.
