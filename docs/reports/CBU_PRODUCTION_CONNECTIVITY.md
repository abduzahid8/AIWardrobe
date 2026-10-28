# Production CBU connectivity incident — 25 October 2026

Evening recheck,17:34UTC: a fresh authenticated production request still failedHTTP503/`budget_rates_unavailable` after12,517ms inFrankfurt. Its disposable user/profile/session were independently confirmed absent. A separate database-network diagnostic then reachedGoogle over verifiedTLS (HTTP200,159ms), whileCBU port443 hit the explicitly configured4-second connection deadline. The temporary `http` extension and private diagnostic schema existed only inside rolled-back transactions and are independently absent. No ingestion job or rate snapshot was installed. This rules out a general lack of database HTTPS connectivity; it does not identify the party responsible for theCBU connection failure. [Rate recheck](readiness-evidence/oct25-evening-budget-refresh-recheck.json) · [Database network evidence](readiness-evidence/oct25-database-cbu-network-probe.json).

Budgeted outfit generation is currently blocked. The shared reference-rate snapshot expired, and the production rate refresh cannot establish a connection to the official feed. The unbudgeted v131 outfit repair remains separately verified.

At 02:15 UTC, an administrator-only temporary function tested Frankfurt (`eu-central-1`) and Mumbai (`ap-south-1`):

- DNS returned `185.203.237.70` in 1–2 ms.
- TCP to port443 timed out at5 seconds in both regions.
- HTTPS timed out at4.5 seconds in both regions.
- The real rate service independently exhausted its8-second upstream deadline, reporting `upstream_fetch`, aborted=true, HTTP503.

The same dated CBU endpoint returned HTTP200 from the developer Mac; its payload passed the application's actual parser, with reference date24October. CBU's `www` hostname resolves to the same address and also failed from production. Changing the rate-fetch region to Frankfurt did not fix it. These observations identify the failed connection path; they do not establish whether filtering/routing originates at CBU, its host, AWS or Supabase.

The canonical endpoint remains `https://cbu.uz/uz/arkhiv-kursov-valyut/json/all/YYYY-MM-DD/`, following [CBU's documented dated API](https://www.cbu.uz/en/arkhiv-kursov-valyut/veb-masteram/). Regional probes used [Supabase's documented region controls](https://supabase.com/docs/guides/functions/regional-invocation). No provider credential or user data was included in diagnostic output.

A network/ingestion repair must demonstrate an actual expired-cache refresh in production within the15-second client deadline, persist a correctly dated and validated official snapshot, and then pass category/budget and three-card deck tests. Manually seeding today's cache or accepting expired/guessed rates would conceal the failure and does not satisfy this requirement.

The initial diagnostic production `budget-rates` v5 preserved the original URL, validation and hourly freshness contract and adds sanitized stage logging. Its four files match the prepared deployment. The temporary diagnostic function was deleted. Independent SQL confirms zero remaining users, profiles or sessions for all six fixtures. Cache regression31/31 and partial-ranking handler90/90 pass; no additional full-suite claim is made after these diagnostic-only changes.

[Machine-readable evidence](readiness-evidence/oct25-budget-rates-network-blocker.json) · [Regional probe](readiness-evidence/oct25-cbu-network-probe.json)

## Shared refresh repair — later25October

Current `budget-rates` v6 and `generate-outfits` v132 additionally coordinate refreshes through a service-only database lease. This removes duplicate cold-instance refreshes and prevents late workers from publishing. A shared failure cooldown limits retries while the network path is down. Rate source, freshness and validation rules remain unchanged.

The ten-isolate simulation changed from10 upstream calls/writes to1. Twenty simultaneous real lease claims produced one owner; expiry, replacement fencing and atomic publication were checked in a transaction that was rolled back. Five live authenticated rate reads during the outage generated one upstream attempt, then failed closed; no current snapshot was manually inserted. A post-deployment unbudgeted three-look request passed. The actual CBU reachability blocker remains unresolved.

[Shared refresh evidence](readiness-evidence/oct25-shared-budget-refresh-repair.json). The service-only table intentionally has RLS enabled with no client policy and no anonymous/authenticated grants; its informational [advisor notice](https://supabase.com/docs/guides/database/database-linter?lint=0008_rls_enabled_no_policy) does not call for granting client access.
