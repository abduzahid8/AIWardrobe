# Phase 6 — Activation Guide (LOCAL ONLY)

All Phase 6 code runs locally/offline. The migration
`backend/supabase/migrations/20261106120000_phase6_growth.sql` is prepared
locally and NOT applied. No remote mutation is authorized by the Phase 6 prompt.

## What works without remote activation

- Activation state, onboarding routing, home prioritization, empty states,
  streaks — pure local logic (`src/services/growth/activation.ts`, `home.ts`).
- Notification policy evaluation — local only; nothing is sent
  (`notifications.ts` has no transport).
- Entitlement matrix + lifecycle + quotas against the existing local
  tier system; `MockEntitlementProvider` for tests/dev.
- Paywall context building, plan models (prices null until store provides).
- Savings ledger validation (verified-only), utilization, cohorts, education.
- Growth analytics definitions, KPI versioning, feedback domain.
- Affiliate/sponsorship guardrails, compliance checklist, server verifier
  interface (`AllowlistServerVerifier` as local stand-in).

## What requires remote activation (each needs explicit approval)

| Object | Purpose | User impact | Risk | Rollback | Success metric |
|---|---|---|---|---|---|
| Hosted migration apply | Persist activation/prefs/referrals/savings/feedback | None until clients read it | RLS misconfig → cross-user leakage | Roll back migration; tables unused by clients yet | Migration replay clean; RLS tests pass |
| RevenueCat products/offerings | Real plans, trials, prices | Paywall becomes purchasable | Wrong price/trial terms | Revert offering in dashboard | Trial start rate; billing-issue rate flat |
| Store (App/Play) pricing | Localized prices | Real charges | Price mismatch | New version / price change | Purchase success rate |
| Push/email service | Deliver value-based messages | Notifications arrive | Spam/quiet-hours violation | Disable sender; policy engine already gates | Dismiss rate; disable rate |
| Referral backend rewards | Credit inviters | Rewards granted | Fraud, double-credit | Revoke campaign; idempotency keys bound | Reward duplication = 0 |
| Live experiments (paywall/onboarding/retention) | Optimize timing/messaging | Variant UX | Deceptive disclosure | Kill-switch to control via Phase 5 flags | Aha completion up, value metrics not down |

## Guardrails before any activation

1. Migration replay + RLS review on a staging project.
2. Store compliance checklist (`STORE_COMPLIANCE_CHECKLIST`) reviewed.
3. Trial disclosure copy approved; restore/manage links verified.
4. Notification policy thresholds reviewed (cooldowns, quiet hours).
5. Value-vs-revenue guardrail wired into every experiment (6AZ).
