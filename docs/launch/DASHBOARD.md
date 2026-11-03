# First-week dashboard specification v1

All baseline/target values are UNKNOWN until observed. Show count, denominator, cohort, RC/config version, time window, sample size and missing-coverage label. No data is not zero. Deduplicate transport retries by eventId/requestId; pseudonymous subject/cohort attribution must be consented and owner-scoped. Do not log raw wardrobe/body/image/URL/prompt/free-comment content. Current analytics endpoint is a log/ack path, not an installed dashboard; this is a specification, with partial local event coverage. Live monitoring destinations and alerts must be explicitly approved and tested before activation.

| Domain / metric | Definition and denominator | Source / interpretation |
|---|---|---|
| Reliability: crash-free sessions | 1 - sessions with native/JS fatal crash / observed sessions | Native crash reporter + session IDs; crashes before session creation require launch-attempt denominator |
| Startup failures | launch attempts without interactive core ready or with fatal error / observed launch attempts | Native launch/startup ready instrumentation; device validation pending |
| Auth failures | terminal auth/refresh failures / auth attempts; split network vs invalid session | Auth error classes; exclude deliberate signout |
| Catalog failures | failed catalog requests / requests, split RPC/window/cache | Request/error/strategy; missing RPC fallback is capability, not automatically a user failure |
| Provider failures | failed/timeout/malformed calls / provider attempts, per feature/provider | Server usage ledger; fallback may preserve user success |
| Activated users | unique users with confirmed outfit saved/worn, helpful BuyCheck or another confirmed value action / eligible exposed users | Open/view/generate-only do not qualify; v1 definition here is stricter than broad milestone helpers |
| Outfit saved/worn | unique outfit outcome IDs saved or wear-confirmed; dedupe save vs wear when counting value users | Existing saved-outfit/calendar state + outcome events; UI event coverage needs audit |
| BuyCheck completed/helpful | completed decisions / started attempts; explicit helpful yes / rated decisions | Completed ≠ helpful; feedback export supports helpful intent, no implied purchase avoidance |
| Recreated look saved | reference-derived outfit saved and user confirmed usable / recreate attempts | Domain and UI linkage must be proven; unavailable if no exposure |
| Wardrobe utilization | distinct owned active item IDs wear-confirmed in window / active owned items eligible in window | Aggregate counts server-side; never export garment arrays to analytics |
| Fallback/low confidence | fallback results / decisions; low-confidence results / confidence-labelled decisions | Per-feature counts, distinguish intentional local primary from degraded fallback |
| Corrections | non-reverted corrections / recommendations with correction UI exposure; type breakdown | Voluntary reports and product edits; reverse events do not erase audit |
| Hard-rule violations | validated violating recommendations / validated recommendations; also show absolute count | Any confirmed Tier A violation stops release regardless of small rate |
| Duplicate recommendation rate | repeated physical-product recommendation within defined task/window / recommendations | Define physical-product vs color variant and intentional repeat; no raw product list in analytics |
| Monetization: paywall views | unique view events per session and eligible user | Exposure only, not value |
| Purchase starts/purchases | starts; server-verified purchased transaction IDs; verified / started | Split cancellation, pending and failure; no client-forged premium conversions |
| Restores | restore attempts, verified restoration success and error; successful / attempts | Sandbox first; unknown not free/premium guessed |
| Subscription state failures | unresolved/failed server state refresh / refresh attempts | Split grace/billing issue/expired from infrastructure errors |
| Cost: AI/VTON calls | actual billable attempts, per request/feature/provider/model | Include failed billable calls and retries; do not count only successful screens |
| Cache hit rate | served valid cache hits / eligible cache lookups | Do not count lookup failures as misses silently |
| Cost estimate by feature | sum observed units × versioned rate; currency and rate effective date | Missing usage/rate => UNKNOWN, not $0; reconcile against provider invoice |

First-week views: daily counts plus cohort D1/D7 return-to-plan outcomes, failure/correction class ranking, latency distributions from measured client/server boundaries, spend by feature, and user-selected aha moments from interviews. Show least-used complexity only with verified feature exposure. A user who does not buy after BuyCheck has not necessarily avoided a duplicate.

Monitor owners: mobile (crash/startup/UI latency), backend (auth/catalog/provider/SQL), billing (transaction/restore), product (confirmed value/corrections), release/on-call (cohort decision). Names, destinations and prospective numeric thresholds are unassigned; monitoring readiness is not PASS. Immediate stop conditions: any cross-user exposure/data loss/forged premium/confirmed hard-rule violation, recurring startup crash, runaway spend beyond approved cap, or restore regression. Relative rate alerts need the actual Stage 0 baseline before a threshold is chosen; no fake production baseline is supplied.
