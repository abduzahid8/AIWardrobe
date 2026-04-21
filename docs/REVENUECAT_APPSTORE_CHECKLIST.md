# RevenueCat + App Store Submission Checklist

Use this checklist before submitting the first iOS version with subscriptions.

## 1) App Store Connect (must be done first)

- Open your app in App Store Connect.
- Go to `Features > Subscriptions`.
- Create a subscription group (you already have one).
- Create at least one subscription product:
  - `com.aiwardrobe.premium.monthly`
- Fill all required metadata on the subscription:
  - Localization (display name + description)
  - Price
  - Review screenshot
- Ensure Agreements, Tax, and Banking are fully complete.

## 2) RevenueCat Dashboard

- Create iOS app in RevenueCat with the same bundle ID as your app.
- Add App Store product IDs to RevenueCat:
  - `com.aiwardrobe.premium.monthly`
  - (Optional) `com.aiwardrobe.vip.monthly`
  - (Optional) `com.aiwardrobe.vip.yearly`
- Create entitlements:
  - `premium` (or `pro`)
  - `vip` (if you sell Max plans)
- Attach each product to the matching entitlement.
- Create an offering named `default` and add packages.
- Copy the iOS Public SDK Key (`appl_...`).

## 3) App Environment Configuration

- Set `EXPO_PUBLIC_REVENUECAT_API_KEY` in `.env` locally.
- Set the same env var in EAS Secrets for production builds.
- Build a native iOS binary (`expo-go` is not enough for IAP testing).

## 4) App Code Requirements (already integrated in this project)

- RevenueCat SDK initializes at app startup.
- Purchases and restores are wired in paywall screens.
- Active entitlement sync updates local subscription tier.
- Restore Purchases button exists (App Review requirement).
- User identity is linked on login and cleared on logout.

## 5) Sandbox + TestFlight Validation

- Create Sandbox tester account in App Store Connect.
- Test purchase on real iPhone with dev/TestFlight build.
- Validate:
  - First purchase unlocks immediately.
  - App restart keeps subscription unlocked.
  - Restore Purchases unlocks after reinstall.
  - Trial-expired gate unlocks after successful purchase.

## 6) First Submission Rule (important)

- First subscription must be submitted with a new app version.
- In `App Store > iOS App > Prepare for Submission`, attach your subscription in:
  - `In-App Purchases and Subscriptions`
- Submit the app version and first subscription together.

