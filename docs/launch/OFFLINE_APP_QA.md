# Actual App root: signed-out offline simulator QA

This uses the actual `App.tsx`/RootNavigator/LanguageSelection/SignIn code in a separate Release app (`com.aiwardrobe.rcoffline`) on the task's iPhone 17 Pro / iOS 26.5 simulator. It is not an authenticated cohort, real Supabase session, production configuration, billing test or physical-device certification.

The explicit QA entry is `scripts/validation/launch-rc-offline-app-entry.tsx`. Before requiring the App it points backend config at loopback public sentinels, denies HTTP fetch/XHR (with a normal network-failure error), permits bundled file/asset/data reads, and disables native purchase configuration/purchase/restore, notification registration, analytics sending and tab prewarming. No account is seeded or created. Some app services attempt external fetches; the guard rejects them before invoking the original transport. Attempt counts are recorded without URLs, request bodies, passwords or headers. Do not interpret blocked attempts as outbound requests or production failures. Native store validation was deliberately not attempted.

Build with the isolated bundle ID and no dotenv:

```sh
EXPO_NO_DOTENV=1 EXPO_OFFLINE=1 EXPO_PUBLIC_MEASURE_STARTUP=true \
EXPO_PUBLIC_SUPABASE_URL=http://127.0.0.1:54321 \
EXPO_PUBLIC_SUPABASE_ANON_KEY=local-public-qa-fixture \
EXPO_PUBLIC_REVENUECAT_IOS_API_KEY=appl_local_qa_stub \
EXPO_PUBLIC_API_URL=http://127.0.0.1:54321 RCT_NO_LAUNCH_PACKAGER=1 \
ENTRY_FILE=/Users/v.zahid/AIWardrobe/scripts/validation/launch-rc-offline-app-entry.tsx \
xcodebuild -workspace ios/AIWardrobe.xcworkspace -scheme AIWardrobe \
 -configuration Release -sdk iphonesimulator -destination 'generic/platform=iOS Simulator' \
 -derivedDataPath /tmp/aiwardrobe-rc-native CODE_SIGNING_ALLOWED=NO \
 CODE_SIGNING_REQUIRED=NO ONLY_ACTIVE_ARCH=YES ARCHS=arm64 \
 PRODUCT_BUNDLE_IDENTIFIER=com.aiwardrobe.rcoffline build
```

Only install the verified isolated ID on the disposable simulator. The binary/entry/Hermes checksums are in `evidence/offline-app-build.json`. This build directory is shared with the earlier isolated component harness, so receipts point to their installed copies when preserving provenance. Neither artifact is a signed shipping client.

Observed native UI: fresh signed-out launch reached the real language chooser, not a spinner/config-error screen; selecting English reached SignIn; a synthetic unregistered email advanced to password entry; the denied authentication request showed Login Failed / Network request failed and returned to usable controls. No Apple/Google authentication, registration, purchase, restore or external share was invoked. The account remained unauthenticated. The local language choice survived process-cold restarts; sign-in inputs did not persist across them. Screenshots and `offline-app-ui.json` document these limited assertions. Initial stricter-guard observations are retained separately and are not substituted for the refined build's measurements.

The repeatable observer targets only this isolated ID:

```sh
python3 scripts/validation/launch-rc-offline-observe.py \
 AD2C6D1C-096E-4D89-86B8-22D600E30C86 --runs 10 \
 --output docs/launch/evidence/offline-app-startup.json
```

Ten process-cold launches preserve the language choice and no authenticated session. `offline-app-startup.json` contains each sample: native RN app-start → revealed-screen p50 429 ms, range 386–663 ms, all under the preexisting local 1000 ms diagnostic budget. External process launch → file receipt p50 1848 ms includes CLI/IO and is a different boundary. Do not mix the two or claim production/authenticated startup. Background service stubs and local configuration are material limitations.

Remaining: successful sign-up/sign-in, cached signed-in session restore, wardrobe/add/generate/save/browse, real entitlement recovery, network reconnection, other supported devices, approved-config native builds and installed old/new/backend combinations. These gates stay MANUAL REQUIRED. An offline signed-out error path cannot prove billing, RLS or an authenticated first-value journey.
