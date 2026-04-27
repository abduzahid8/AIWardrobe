# Subscription & Payment Integration Setup

This document serves as the source of truth for the integration between **App Store Connect**, **RevenueCat**, and **Supabase**.

## 1. Core Identifiers (MUST MATCH)

| Entity | Identifier in Dashboards | Identifier in Code |
| :--- | :--- | :--- |
| **Bundle ID** | `com.aiwardrobe` | `app.json` -> `bundleIdentifier` |
| **Monthly Product** | `com.aiwardrobe.premium.monthly` | `subscriptionStore.ts` -> `SUBSCRIPTION_PRICING.premium` |
| **Yearly Product** | `com.aiwardrobe.premium.yearly` | `subscriptionStore.ts` -> `SUBSCRIPTION_PRICING.vip` |
| **Entitlement ID** | `AIWardrobe Pro` | `iapService.ts` -> `findEntitlement(['AIWardrobe Pro', ...])` |
| **Offering ID** | `default` | `iapService.ts` -> `getAvailablePackages()` |

## 2. RevenueCat Configuration

### API Keys
- **Public SDK Key:** Starts with `appl_` (iOS) or `test_` (Sandbox/Dev). 
- **Local Env:** `EXPO_PUBLIC_REVENUECAT_API_KEY` in `.env`.

### Webhook Setup
- **URL:** `https://[your-project-ref].supabase.co/functions/v1/revenuecat-webhook`
- **Events:** Enable all (INITIAL_PURCHASE, RENEWAL, CANCELLATION, etc.).
- **Purpose:** Automatically updates the `profiles` table and `subscriptions` table in Supabase when a payment event occurs server-side.

## 3. Supabase Mapping Logic

The `revenuecat-webhook` Edge Function maps products to tiers as follows:
- **VIP Tier:** Any product ID containing `vip` or `yearly`.
- **Premium Tier:** Any product ID containing `premium` or `pro`.
- **Free Tier:** Fallback for all other IDs.

## 4. App Store Connect Readiness

- **Agreements:** "Paid Apps" agreement must be **Active** in Business section.
- **Banking/Tax:** Must be **Active**.
- **Sandbox Testing:** Create a Sandbox Tester in "Users and Access" to test payments on a real device.

## 5. Testing Checklist

1. [ ] Use a physical iOS device (IAP does not work on Simulator).
2. [ ] Ensure `.env` contains the correct `EXPO_PUBLIC_REVENUECAT_API_KEY`.
3. [ ] Run `supabase functions deploy revenuecat-webhook` after any logic changes.
4. [ ] Verify "Pro" features unlock immediately after purchase (check `effectiveTier` in `subscriptionStore`).
