# App Store Privacy Label Guide (AIWardrobe)

This is a practical cheat‑sheet for filling **App Store Connect → App Privacy**.
It’s based on what the code currently does in this repo (Supabase auth + cloud sync, AI photo processing, optional analytics/crash reporting, subscriptions).

## Data types the app processes

### Contact info
- **Email address**: used for account sign‑in and support.
- **Name** (username): used for profile display.

### User content
- **Photos / Videos**: user-selected media for wardrobe scans, item uploads, outfit share images, AI try‑on inputs/outputs.

### Purchases
- **Purchase history / subscription status**: used to unlock paid features and restore access.

### Location
- **Coarse/precise location (When In Use)**: optional; used to fetch local weather for outfit suggestions (not stored).

### Identifiers
- **User ID**: Supabase user ID used for account + syncing.
- **Device / session identifiers**: may be used by crash/diagnostic SDKs and for analytics session grouping.

### Diagnostics
- **Crash data** + **performance data**: used to improve stability (Sentry when configured).

## “Linked to you” vs “Not linked”
Typical, conservative mapping:
- **Linked to you**: Email, User ID, Photos/Videos you upload (because they live under your account).
- **Not linked**: Aggregated analytics events if you do not attach them to identity (note: this app currently can attach userId to analytics events when analytics sharing is enabled).

## Tracking
- The app does **not** implement App Tracking Transparency (no IDFA usage found) and `PrivacyInfo.xcprivacy` declares `NSPrivacyTracking=false`.
- If you ever add cross-app tracking/ads, you must update both your app privacy answers and ATT behavior.

## Feature-by-feature notes (helps reviewers)
- **AI processing**: user-selected images may be sent to the backend and processed by third-party AI providers to generate classification/cutouts/try-on renders.
- **Analytics**: optional; user can disable “Analytics sharing” inside Profile → Preferences.
- **Affiliate links**: tapping shopping links opens external product pages for physical goods; the app may record a “product clicked” analytics event.

## Suggested App Privacy answers (high-level)
You should expect to answer “Yes” for collection of:
- Contact info (email, username)
- User content (photos/videos)
- Purchases
- Diagnostics
- Identifiers
- Location (optional)

And “No” for:
- Tracking (unless you add ATT/IDFA or cross-app tracking in the future)

