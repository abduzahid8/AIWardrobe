# Design restoration — 2026-11-01

Restored the presentation changed by the startup optimization work:

- Home creates the original `expo-video` player during its initial mount, with the original focus-based playback and pause behavior.
- Removed the static hero poster and delayed player replacement, including their unused components, asset and tests.
- Restored direct sharing/shopping dialog components and their original mount behavior.
- Restored the previous floating glass tab bar on iOS, replacing the newer standard labeled bar. Removed the standard bar's duplicate tab-press handler; the custom bar retains its original navigation, haptics and prewarming.
- Preserved layout, typography, colors, card styling and splash animation; unrelated workspace edits were left in place.

Fixed a startup scheduling error: `requestIdleCallback`'s `timeout` is a deadline, not a minimum delay. Previously the notification and purchase jobs could both run on the first available idle callback after the 250 ms startup timer. They now wait their actual requested delays, require an idle budget, and cancel pending work on navigator unmount. Screen preloads use the same scheduler and start no earlier than four seconds after readiness, one module at a time.

Wardrobe synchronization still starts promptly. Auth cleanup, purchase identity serialization and shared purchase SDK initialization are retained.

The 2026-10-31 timing report is marked superseded. Time to splash dismissal does not establish smoothness after opening, and its measurements used the deferred-video presentation that has now been removed.

Validation covers the real delay even when an idle callback runs immediately, postponement during busy frames, cancellation, the preload quiet period, startup routing, auth/IAP safety and Home occasion visibility.

142 targeted tests across six suites passed, along with TypeScript and the scheduler's ESLint checks. An arm64 Release build succeeded for the iOS 26.5 simulator. The native UI check opens Home, switches to Closet, verifies the clothing tab, returns to Home and verifies Today's Look.

The native UI check passed and its screenshot was inspected: Home displays the original video and the floating icon-only tab bar. Three process-cold launches of this final restored build measured **472, 453 and 465 ms** from native app startup through splash fade. [Raw timing evidence](startup-restored-design-fixture-20261101.json).

The launch test uses the existing synthetic cached session, matching owner stamp, completed onboarding and empty wardrobe against unreachable `.invalid` test services. It does not use a live user account. These timings measure opening, not frame pacing. Physical-device smoothness, live session refresh, and large wardrobes remain unverified; no claim that all device/network stutter is eliminated is made.
