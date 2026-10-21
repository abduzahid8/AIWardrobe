# Subscription and payment integration

## Product and entitlement mapping

| Product | Tier |
|---|---|
| `2.99`, `com.aiwardrobe.lite.monthly` | Lite |
| `com.aiwardrobe.premium.monthly` | Pro (`premium`) |
| `com.aiwardrobe.premium.yearly` | Max (`vip`) |

The bundle ID is `com.aiwardrobe`. The native SDK uses the project's configured offering. Server mappings live in `backend/supabase/functions/_shared/revenuecatSnapshot.ts`, including supported entitlement aliases. An unknown active entitlement raises a verification error; it does not silently grant Pro or downgrade a paying user.

## Configuration

The mobile app receives platform-specific public SDK keys: `EXPO_PUBLIC_REVENUECAT_IOS_API_KEY` (`appl_…`) and `EXPO_PUBLIC_REVENUECAT_ANDROID_API_KEY` (`goog_…`). Set both in the corresponding build environment. The legacy `EXPO_PUBLIC_REVENUECAT_API_KEY` is accepted only on its matching platform; an iOS key is never reused for Google Play. An invalid explicit platform key leaves purchases unavailable rather than silently selecting another key. RevenueCat test-store keys remain selectable for testing; the existing production checkout guard still rejects test-store payments. See [RevenueCat SDK configuration](https://www.revenuecat.com/docs/getting-started/configuring-sdk).

Set these **server-side Supabase function secrets**, not `EXPO_PUBLIC_` variables:

- `REVENUECAT_WEBHOOK_SECRET`: dedicated random shared secret. Configure the RevenueCat webhook Authorization header as `Bearer <secret>`. The API/SDK key is not accepted as the webhook secret.
- `REVENUECAT_SECRET_API_KEY`: RevenueCat API credential authorized to read Customer Info. `REVENUECAT_API_KEY` remains a legacy environment-variable alias for the API credential.
- `REVENUECAT_ENVIRONMENT`: `PRODUCTION` or `SANDBOX`; defaults to `PRODUCTION`. Use a separate staging project and sandbox integration for release tests.

Webhook URL: `https://<project>.supabase.co/functions/v1/revenuecat-webhook`. Configure lifecycle events and transfers. The function handles dashboard `TEST` without modifying entitlements. Events for another configured environment are acknowledged without changing access.

## Server authority and deployment order

1. Confirm the existing hosted migration ledger. There are duplicate historical `022` filenames in this repository; do not blindly push or reset production migrations.
2. Apply the reviewed schema/protection changes through `029_subscription_status.sql` in the agreed migration process.
3. Configure server secrets and deploy the reviewed Edge Functions, including `sync-subscription` and `revenuecat-webhook`.
4. Ship the matching mobile client. It no longer writes authoritative subscription/profile fields and does not report a mock development payment as successful.
5. Verify the hosted integration before enabling paid traffic.

Both webhook delivery and authenticated checkout/restore synchronization fetch current Customer Info from RevenueCat. The caller cannot supply a user ID or tier to `sync-subscription`; it resolves the JWT owner. Cold-start status reads use the authenticated-owner `get_subscription_status` RPC, which ranks the highest valid tier and preserves lifetime expiry. The server normalizes active entitlements, expiration/grace periods and environment, then commits subscriptions, profile, watermark, payment metadata and event deduplication in one SQL transaction. RevenueCat recommends syncing Customer Info after webhook receipt; see its [webhook guidance](https://www.revenuecat.com/docs/integrations/webhooks) and [Customer Info schema](https://www.revenuecat.com/docs/api-v1/customer-info-model).

A failed transaction leaves the event retryable. A stale snapshot cannot revoke a newer state. Duplicate payment transactions are recognized separately from event IDs. Transfers reconcile both known owners together. An anonymous identity is resolved only through explicit signed aliases and known profiles; no "recent purchaser of this product" heuristic is used. Ambiguous or unresolved identities produce a retryable error and require investigation.

Older deployments may have already marked failed events as processed. Historical event/receipt reconciliation must be reviewed explicitly; this migration cannot infer which old acknowledgements were false. Preserve the audit history when repairing affected accounts.

## Client behavior

- Checkout requires successful RevenueCat identification as the current signed-in user before any payment begins.
- Paid access activates after server verification, including on restore and SDK customer-info updates.
- When the store payment completes but verification is unavailable, the paywall shows a translated pending state and automatically retries while visible and active. The purchase button stays disabled; users can still leave the page.
- Verification responses arriving after logout/account switching cannot reactivate the previous account.
- A successful authoritative refund/free read overrides cached paid access. A database read failure is not treated as a verified downgrade.
- Cancellation preserves paid access until verified expiry. Explicit lifetime entitlements preserve a null expiry rather than inventing a monthly expiration.

## Validation required before release

Local regression coverage includes ownership, API/database failure, identity failure before checkout, pending purchase/restore, logout races, cancellation expiry, duplicate events/payments, stale delivery, failed-event retries, transferred refunds and concurrent duplicate delivery.

Hosted release validation still must cover a real sandbox purchase and restore, renewal, billing grace, cancellation, expiration, refund, upgrade/downgrade timing and account transfer. Use a physical device as part of release validation. Confirm the dashboard offering/products, entitlement aliases, server API permissions, webhook Authorization/environment, immediate checkout sync, operational alerts and retry handling. Local mocks and SQL assertions do not prove the hosted integration is configured correctly.
