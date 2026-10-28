# Current publication readiness — 28 October 2026

**Decision: hold publication.** The app is not certified for 10,000 daily active users, and the Apple Sandbox payment lifecycle has not been completed. The current source line is version 2.1.0 with iOS build 61 and Android versionCode 60.

## Verified

- Full Jest regression suite: **2,455 tests across 193 suites passed** after enforcing the mandatory four-piece outfit and budget-snapshot validation fixes.
- TypeScript passed. The 47 focused payment/backend tests passed after upgrading `react-native-purchases` to 10.11.0 / iOS SDK 5.92.0; `npm ci --dry-run` with EAS's default peer resolution passes.
- Offline load-harness: **38/38 passed**. It validates the 10,000-DAU workload assumptions and load-test safeguards; it sends no load to production and does not certify capacity.
- The last native paywall check on simulator build 58 retained Payment QA and showed Lite $2.99/month, Pro $9.99/month, and Pro Yearly $99.99/year. Lite Continue passed checkout preflight, but opened a generic Apple Account prompt in an ad-hoc-signed Simulator build. This is not a Sandbox transaction. The test profile remains Free with 0 payment rows and 0 active subscriptions. [Build 58 check](readiness-evidence/oct27-payment-build58-live-check.json).
- On that same simulator, the expired-trial Create Outfit route opened the paywall, and Continue with Free returned to Home; this checked gating/navigation only, not a purchase. The free allowance counters remained untouched. [Free-plan gate flow](readiness-evidence/oct28-payment-free-plan-gate.json).
- EAS iOS build 60 and Android build 59 completed with store distribution after the RevenueCat SDK upgrade, but predate the final outfit-layer fix. Both current source builds are complete: iOS build 61 passed `codesign --verify`, and App Store Connect reports it as `VALID` / `IN_BETA_TESTING`; EAS's submit job reports duplicate build number even though ASC confirms the build is present. Android versionCode 60 completed as a store AAB and passed `jarsigner` verification. [Build evidence](readiness-evidence/oct28-final-candidate-builds.json).

## Release blockers

- App Store Connect still shows the old 1.4.3 binary (build 57) as Invalid Binary, while build 61 is in TestFlight beta. The Account Holder must review the updated Apple Developer Program agreement before an App Store review submission.
- A real Sandbox purchase, signed transaction/webhook propagation, server entitlement activation, restore after reinstall, renewal, cancellation, refund, and Android purchase lifecycle are not verified. The physical iPhone 14 Pro Max is paired but unavailable; Developer Mode is disabled. The Simulator cannot prove a real Sandbox transaction.
- RevenueCat has an App Store app configuration but no Google Play app/product configuration. Android purchase setup remains incomplete.
- No current hosted 40-RPS representative load, 10,000-user cardinality test, backup restore, or recovery/alerting exercise has been run. The 40-RPS estimate is based on unverified assumptions (10,000 DAU × 2 sessions/day × 12 requests/session, 20% of daily requests in peak hour, 3× headroom). Production data was not load-tested. The connected Supabase project is below Pro; branch creation was rejected because branching requires Pro or above. Local Docker is installed but the daemon is unavailable, local PostgreSQL is not running, and Deno is not installed.
- Re-auditing after the RevenueCat update still reports three high-severity production dependency advisories in Expo CLI/Zod; npm recommends an incompatible Expo 54 downgrade, so no forced downgrade was applied. Edge Function Deno typechecking and current store/device journeys remain outstanding.

See the [publication gate](PUBLICATION_GATE.md), [payment evidence](readiness-evidence/oct27-payment-build58-live-check.json), [free-plan gate flow](readiness-evidence/oct28-payment-free-plan-gate.json), and [current store build evidence](readiness-evidence/oct28-final-candidate-builds.json).
