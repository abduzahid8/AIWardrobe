# Startup performance — 2026-10-31

> Superseded on 2026-11-01: the deferred hero/poster and deferred dialogs were removed to restore the original presentation. These timings describe the earlier implementation and do not validate the restored design or animation smoothness. The native-service scheduling bug is covered by the newer startup tests.

Release simulator measurements meet the 1,000 ms opening budget in the tested scenarios.

| Scenario | Process-cold runs | Minimum | Maximum |
| --- | ---: | ---: | ---: |
| Fresh install, language chooser | 5 | 284 ms | 495 ms |
| Saved language, signed out, Sign In | 5 | 343 ms | 408 ms |
| Synthetic cached session, Home, unavailable test services | 5 | 276 ms | 352 ms |

Raw measurements: [fresh install](startup-fresh-install-20261031.json),
[signed out](startup-signed-out-20261031.json),
[cached-session fixture](startup-cached-session-fixture-20261031.json).

## Changes

- Language and authentication restoration run concurrently; both still gate routing.
- Account cleanup/ownership checks and local usage hydration remain before routing.
- Authentication screens load when their route is opened.
- Home imports its required UI components directly, avoiding the full UI barrel.
- Home shows a local poster immediately; video module/decoder creation starts after focus settles. Blur releases the player.
- Sharing/shopping dialogs load on first opening and remain mounted afterward, preserving drafts and close animations.
- Analytics, purchase initialization, notification setup and cloud reconciliation start after the first frame.
- Restored purchase identification runs through the existing auth operation queue after reveal. Identity changes remain serialized, including native operations that outlive their UI deadline. Queued work for a superseded account is skipped.
- Purchase SDK configuration is shared between concurrent initialization/identification requests.
- Splash handoff waits for navigator layout. The existing short branded fade remains.

## Measurement and reproduction

Built `AIWardrobe`, Release, Hermes, arm64, iOS 26.5 / iPhone 17 Pro simulator on an arm64 Mac. No Metro development server was used.

`EXPO_PUBLIC_MEASURE_STARTUP=true` enables an opt-in measurement from React Native's native app-start timestamp (`performance.rnStartupTiming.startTime`) through the completed splash fade. The result is written **after** measurement to the simulator app's `Documents/startup-timing.json`. Production builds without the flag do not write this file or log these results.

After installing an instrumented Release build:

```sh
python3 scripts/validation/startup-benchmark.py SIMULATOR_UDID --runs 5 --output /tmp/startup-results.json
```

The script terminates the process between launches, preserves persisted app data, and fails if a measured launch exceeds 1,000 ms. Screenshots were inspected to confirm language selection, Sign In and Home actually rendered.

The Home fixture uses a synthetic cached session and matching cache-owner stamp with completed onboarding and an empty wardrobe. Its Supabase/API endpoints use `https://startup-fixture.invalid` and its purchase key is invalid. No real credentials or purchase identity were used. This tests opening Home without depending on network availability; it does not authenticate a live account. The final fixture Hermes bundle SHA-256 is `734f86f4b8132fadbcc3603cd8d0138ac6ce1364d23987eeae8ca48e8a2b6d59`.

The fresh-install and signed-out measurements used the normal environment before the final purchase-initialization concurrency guard; that guard runs after reveal. The cached-session measurements were repeated after that guard was added.

## Validation and limits

159 tests passed across nine suites covering auth, IAP, subscription authority, language, routing gates, screen preloads, Home visibility, video deferral and dialog draft preservation. TypeScript passed. Targeted ESLint completed with no errors and existing/style warnings.

These are process-cold simulator launches, with OS caches retained between runs. They are not physical-device, Android, reboot-cold, large-wardrobe or live expired-session measurements. Remote catalog/profile images can continue loading after opening. The five-second hung-auth recovery guard remains: an expired session requiring server refresh can still exceed one second. The measurements establish the target for the tested launch paths, not an unconditional one-second guarantee under every device/network condition.
