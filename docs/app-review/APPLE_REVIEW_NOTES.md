Must Fix Before Submit
i18n/locales/en.json: missing paywall.termsText while screens/PaywallScreen.tsx renders it; your required auto-renew disclosure can be blank/wrong in English.
store/auth.ts: logs live session.access_token via console.error; remove immediately (security/privacy red flag in production).
src/lib/persistence.ts: logout/delete cleanup does not remove promo keys promo_redeemed_v1 and promo_skipped_v1 from store/promoCodeStore.ts; causes cross-user state leakage on shared devices.
store/auth.ts comment/flow says promo is required for trial, but store/subscriptionStore.ts hard-disables promo gate (needsPromoCode = false); this business-logic mismatch can confuse review and metadata claims.
screens/PaywallScreen.tsx legal/commercial text depends on localization completeness; ensure all storefront languages have identical required subscription disclosures (renewal, billing timing, cancellation path, trial forfeiture wording if applicable).
High-Risk Consistency Issues
ios/AIWardrobe/PrivacyInfo.xcprivacy has empty NSPrivacyCollectedDataTypes; your app clearly processes account data/photos/purchases/diagnostics per i18n/locales/en.json privacy text and docs. Keep privacy manifest and App Store privacy answers aligned.
supabase/functions/revenuecat-webhook/index.ts: webhook auth validation is permissive (warns on missing auth, may still continue). Harden authentication to reduce fraud/security risk.
store/subscriptionStore.ts + promo system files: promo-based trial entitlement exists in production code (screens/PromoCodeScreen.tsx, store/promoCodeStore.ts, supabase/functions/redeem-promo/index.ts). If exposed in review, this can trigger payment-policy scrutiny under digital entitlement rules.
Medium Priority (Should Fix)
docs/SUBSCRIPTION_SETUP.md appears stale/inconsistent with live code comments and entitlement naming; align docs with actual RC product/entitlement mapping to avoid App Review note mistakes.
api/routes/subscription.js contains TODO/mock receipt validation paths. If this backend is live/reachable, harden or disable unused endpoints to avoid trust/security concerns.
docs/APP_REVIEW_NOTES_TEMPLATE.md should be updated to exactly match current gating and trial behavior (especially promo/trial wording) before copy-pasting to App Store Connect.


1) Subscription / IAP compliance (3.1.1, 3.1.2, IAP submission docs)
✅ You use StoreKit/RevenueCat for paid digital features.
✅ Restore + Manage Subscription UI exists.
❗ English paywall legal disclosure missing: paywall.termsText absent in en.json, while screen renders it.
❗ Promo-code trial unlock path still exists in codebase; if discoverable/reachable, reviewer can treat it as non-StoreKit digital unlock risk.
❗ App Store Connect process requirement: first subscription/new type must be submitted with new app version, status Ready to Submit, review screenshot, review notes. (Operational checklist, not code.)
2) App Completeness / Accurate Metadata (2.1, 2.3)
❗ In-app behavior/comments conflict:
auth comments say promo required for trial,
subscription store currently disables promo gate.
Any mismatch between app behavior and metadata/review notes is a classic 2.3 rejection trigger.
✅ You already have review-notes/privacy docs, which is good, but they must be updated to match current behavior exactly.
3) Privacy / Security (1.6, 5.1)
❗ store/auth.ts logs session.access_token (security issue).
❗ PrivacyInfo.xcprivacy has empty NSPrivacyCollectedDataTypes while app clearly processes account info/photos/purchases/diagnostics.
✅ In-app privacy policy screen exists; includes retention/deletion text and contact email.
4) Account / Data deletion expectations (5.1.1)
✅ In-app account deletion exists and calls backend function.
❗ Local persistence wipe misses promo keys (promo_redeemed_v1, promo_skipped_v1), causing cross-user leakage on shared device.
5) App Review operational requirements (App Review + ASC Help)
Must provide:
valid reviewer access (demo account if needed),
complete app review notes for non-obvious flows (IAP/trial/restore),
all links functional,
backend services live during review.
These are frequent fail points per Apple’s “Avoiding common issues.”


Absolute blockers / highest priority
Apple SDK deadline risk (very high)

Apple’s Submitting page says from Apr 28, 2026 iOS apps must be built with iOS 26 SDK+.
Your stack is expo ~54 now; this may block upload/review unless your build chain already targets the required SDK.
This is a platform requirement, not just a guideline preference.
Missing English subscription legal disclosure

screens/PaywallScreen.tsx renders paywall.termsText.
i18n/locales/en.json does not contain this key (ru/uz do).
Risk: missing/incorrect mandatory auto-renew subscription disclosure in English storefront.
Access token logging in auth flow

store/auth.ts logs session.access_token.
Even if Babel strips console in production, this is still a critical security hygiene issue and should be removed at source.
High risk (likely rejection or severe review friction)
Privacy manifest mismatch

ios/AIWardrobe/PrivacyInfo.xcprivacy has empty NSPrivacyCollectedDataTypes.
Your app clearly collects/processes account info, photos/videos, purchases, identifiers, diagnostics (per code + privacy copy).
Mismatch can trigger compliance questions and review delays.
Promo/trial policy ambiguity

Promo redemption and trial grant system still exists (PromoCodeScreen, promoCodeStore, redeem-promo function).
subscriptionStore says promo gate disabled for App Store submission, but route is still present in navigator.
If reviewer reaches promo unlock flow, it may raise 3.1.1 payment-policy scrutiny.
App behavior/documentation mismatch

Auth comments say promo code needed for trial, but runtime gate is disabled and route behavior differs.
Apple heavily enforces accurate metadata/review notes (2.1, 2.3).
Medium risk / must harden before submit
Local data wipe incomplete on logout/delete

clearAllPersistedUserData() does not clear promo_redeemed_v1 / promo_skipped_v1.
Cross-user residue on shared devices (privacy trust issue).
useSessionGuard contains invalid hook usage

Calls useTranslation() inside async callback, not at component/hook top level.
This is a React hook rule violation and potential crash path (2.1 crash risk).
Guide CTA URL open without guard

GuideScreen uses Linking.openURL(url) without canOpenURL/catch flow.
Broken/malformed URL from remote content may degrade review experience.
RevenueCat webhook auth posture is permissive

revenuecat-webhook warns on missing auth headers and does not strictly reject all unauthenticated requests.
Security hardening recommended before production review.
Potential “incomplete feature” perception in some flows

MagicMirrorScreen can show demo/mock mode message if AI backend not configured.
If production build shows this to reviewer, may be seen as incomplete functionality (2.1).
Low risk / informational
External affiliate links

Present in shopping/flash-sale components.
This is generally okay for physical goods; not a direct 3.1.1 problem by itself.
Console warning suppression in index.ts

Broad warning suppression may hide useful QA signals during pre-submit testing.
PriceTracker fallback image uses placeholder URL

Not a direct policy violation, but avoid obviously placeholder-ish runtime content in review demos.
Already good (keep)
StoreKit/RevenueCat purchase path exists.
Restore purchases exists.
Manage subscription path exists.
Privacy Policy and Terms are accessible in-app.
In-app account deletion exists.
Sign in with Apple exists.
Final fix order (recommended)
SDK/toolchain compliance for iOS 26 requirement.
Add paywall.termsText in English (and verify all locales).
Remove token logging from auth code.
Align privacy manifest with real collected data.
Resolve promo/trial ambiguity (fully disable/remove for release, or make policy-safe and clearly documented).
Clear promo keys on logout/delete.
Fix useSessionGuard hook misuse.
Harden URL opening and webhook auth checks.
Refresh App Review Notes + metadata to exactly match actual behavior.
If you want, I can now convert this into a file-by-file patch plan and apply the code fixes directly.