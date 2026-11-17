> This older overview is superseded for the current RC by [the exact ordered activation packet](launch/REMOTE_ACTIVATION.md), [inventory](launch/activation-config.json) and [rollback map](launch/ROLLBACK.md). No remote action is authorized or executed.

# REMOTE_ACTIVATION.md (Phase 7CP) — PACKET, NOT EXECUTION

Every remote step in dependency order. Nothing below has been executed;
each needs explicit approval with the 10-point briefing (action, why,
impact, services, migration risk, rollback, validation, metrics, kill
switch, success/failure criteria).

## 1. Supabase

| # | Action | Depends on | Rollback | Verify |
|---|---|---|---|---|
| 1 | Apply Phase 1–4 migrations (hosted) | staging backup | roll back additive migration | replay clean, RLS probes |
| 2 | Apply Phase 5 platform migration | #1 | same | 3 tables + RLS |
| 3 | Apply Phase 6 growth migration | #1 | same | 5 tables + RLS, no-self + idempotency |
| 4 | Deploy compatible Edge Functions | #1–3 | redeploy previous | version endpoint |
| 5 | Storage policies (private buckets) | #4 | revert policy | upload/auth probes |
| 6 | Secrets (provider keys) | — | rotate when authorized | presence w/o value logging |

## 2. RevenueCat

| # | Action | Depends on | Rollback | Verify |
|---|---|---|---|---|
| 7 | Entitlements + offerings + products | sandbox QA | revert offering | purchase/restore sandbox |
| 8 | Webhooks (signed, idempotent) | #7 | disable endpoint | replay + out-of-order test |

## 3. Notifications / jobs

| # | Action | Depends on | Rollback | Verify |
|---|---|---|---|---|
| 9 | Sender + channels (Android) / categories (iOS) | prefs UI live | disable sender | quiet-hour + pref probes |
| 10 | Schedulers (price checks, digests) | #1–3, budgets | pause job | idempotency + rate caps |

## 4. AI providers

| # | Action | Depends on | Rollback | Verify |
|---|---|---|---|---|
| 11 | Enable vision/LLM keys + limits | kill switches live | kill switch | fallback drills |
| 12 | VTON enablement (best-effort) | #11 | disable entry | history intact |

## 5. Stores

| # | Action | Depends on | Rollback | Verify |
|---|---|---|---|---|
| 13 | Signing + build upload | #4–8 green internally | halt rollout | staged rollout pause |
| 14 | Listing + privacy links | legal review | update listing | review-risk checklist |

## Kill switches (live from day one)

`kill:remote_vision_provider`, per-flag kill switches, cost-limiter.
Core wardrobe has no kill switch — it is never killed.
