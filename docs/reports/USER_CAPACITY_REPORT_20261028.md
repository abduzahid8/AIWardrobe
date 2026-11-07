# AIWardrobe user capacity report

Date: 28 October 2026. Assessment includes bounded live production tests described below and earlier dated evidence. Current uncommitted source changes and production configuration may differ from the tested versions.

## Decision

There is no defensible maximum user count for the complete application today. **None of the four user levels is certified:** recent bounded tests found a scan failure at two overlapping requests, image-read timeouts, and elevated authenticated-read latency. These results justify holding a reliability claim; they do not prove that any particular number of daily users will fail. A 50-person pilot remains a product decision with known risks, not a verified capacity guarantee.

For comparison, the numbers below mean **daily active users (DAU)**: unique people using the app on a day, not registered accounts and not people all using it simultaneously. If the target means simultaneous users, these conclusions do not apply; that requires a separate concurrency test.

| Daily active users | Assessment | Why |
| --- | --- | --- |
| 50 | Not certified; monitored pilot only with known risks | A live scan failed at two overlapping requests, and read and media tests missed their thresholds. A small daily population can still generate simultaneous requests. |
| 500 | Not certified | A 2-RPS authenticated read test completed without errors but failed latency targets; AI and media also failed separate bounded tests. The 2-RPS planning figure is an assumption, not proof of a 500-user workload. |
| 1,000 | Not demonstrated | Same unresolved AI, image delivery, database, and recovery limits, with greater exposure to bursts; no 1,000-user representative test. |
| 10,000 | Not ready to certify | The current readiness report explicitly identifies missing current hosted representative load, 10,000-user cardinality, recovery, and alerting tests. The workload assumptions have not been validated. |

A number of registered users can be much larger than DAU without creating the same traffic. Conversely, 50 people all scanning at once could be harder than 500 users spread across a day.

## Evidence

- A historical production read soak completed 24,000 requests in 600 seconds at approximately 40 requests/second with no errors or drops and p95 443 ms, using 20 temporary users with synthetic wardrobes. It covered selected read paths, not AI, uploads, image delivery, or native rendering. It does not prove current full-application capacity. See [10,000-DAU readiness report](READINESS_10000_DAU.md).
- A 40 requests/second write test initially failed with 108 transport/decode failures and 647 dropped requests; a later 120-second repeat completed 4,800 writes with zero failures or drops. The first failure remains unexplained. The later result is encouraging but does not prove sustained reliability.
- A 25 October production scan burst using the same photo returned two valid HTTP responses, but only one of two selected the expected garment and color; the test stopped at two concurrent scans. This is a quality failure in a small sample, not proof of a fixed provider concurrency limit. [Evidence](readiness-evidence/oct25-scan-concurrency-baseline.json). A separate experimental direct-provider configuration failed both requests at concurrency two; it was not deployed. [Evidence](readiness-evidence/oct25-scan-loweffort-concurrency.json).
- A short private-image diagnostic achieved 170/170 reads per region at its stated workload, but Frankfurt p95 was approximately 2.2 seconds; it was a single-owner synthetic probe, not an end-to-end media capacity test. Earlier failures remain unexplained. See [readiness report](READINESS_10000_DAU.md).
- A local database benchmark seeded 10,000 synthetic users, but a local database is not the hosted app and did not exercise AI, uploads, internet, or full production operations.
- The [current publication report](READINESS_CURRENT.md) records additional release blockers, including an incomplete real payment lifecycle and Android purchase configuration, plus missing restore and alerting exercises. These affect launch readiness independently of user count.

## Planning estimate, not measured capacity

Using the existing harness assumptions: 2 sessions per DAU per day, 12 ordinary requests per session, 20% of daily requests in the busiest hour, and 3× peak headroom. The resulting targets are proportional estimates; they exclude AI calls, image delivery, uploads and retries and must be calibrated against real usage.

| DAU | Ordinary requests/day | Estimated busiest-hour average | Test target with 3× headroom |
| ---: | ---: | ---: | ---: |
| 50 | 1,200 | 0.067 requests/sec | 0.2 requests/sec |
| 500 | 12,000 | 0.67 requests/sec | 2 requests/sec |
| 1,000 | 24,000 | 1.33 requests/sec | 4 requests/sec |
| 10,000 | 240,000 | 13.33 requests/sec | 40 requests/sec |

These averages do not capture bursts. Run a separate synchronized concurrency scenario for scanning and uploads. The historic 40-RPS read test covers a subset of the estimated traffic only; it cannot be added to other independent results to claim the whole app passed.

## Main constraints and actions

1. **AI scanning and generation:** test realistic photos and simultaneous requests, tracking correctness, latency, provider 429/5xx and costs. Establish provider quotas and a bounded queue, retry policy, and user-visible failure/retry behavior where needed. Verify quality after any model changes.
2. **Database and API:** test realistic mixed reads and writes with representative distinct users and data volumes. Observe CPU, memory, connections, slow queries, errors and latency; investigate the unexplained write failure before relying on a repeat. Supabase connection limits depend on compute size, and pooler clients are not equivalent to app users. [Supabase compute and connection limits](https://supabase.com/docs/guides/platform/compute-and-disk).
3. **Media:** test native uploads and private image reads across realistic image sizes, networks, regions and bursts. Track latency, bandwidth, storage usage, failures and cost; investigate prior failures.
4. **Operations and payments:** establish dashboards and alerts, test backups and restoration, verify payment lifecycle and store configurations, and exercise recovery. Document the exact deployed versions and provider plans before testing.
5. **Scaling configuration:** `render.yaml` describes a Starter Express service, but the file alone does not confirm the live plan or instance count. Its rate limiter uses per-process memory; if multiple API instances are introduced, use a shared limiter or another coordinated policy, then retest. Not every app request goes through this Express API.

## Suggested acceptance process

Use isolated staging with production-like configuration, realistic datasets and ordinary user sessions. Begin with measured pilot behavior and run mixed load for each intended tier, including peak bursts and longer soak periods. Test scanning, outfit generation, media, login and writes separately as well as together. Define acceptable p95/p99 latency, error and correctness thresholds, cost limits and recovery objectives before testing. Check dashboards and provider quotas during each run, then fix and retest failures. Avoid production load without an agreed bounded plan. Reassess each tier only after its representative workload and recovery tests pass.

## Live retest on 28 October

A bounded production public catalog read test started at 0.2 requests/second, with five requests spaced five seconds apart. All 5 returned HTTP 200 and valid nonempty catalog data; no requests were dropped. Measured latencies had a median of 1,069 ms and maximum of 2,147 ms. With five samples the calculated p95 and p99 are both the maximum; these are too few observations to characterize typical tail latency. The stage failed the predetermined p95 <1,000 ms and p99 <2,000 ms criteria, and the test stopped before 2, 4 or 40 requests/second. [Raw test results](readiness-evidence/oct28-public-catalog-capacity.json). This is a public catalog-only check from one client, not an end-to-end capacity test or proof of a server-side bottleneck. No users or production records were created by this check.

The 38 load-harness unit tests passed. A separate live AI scan test did not start because `SUPABASE_SERVICE_KEY` is empty in the local environment; no test accounts were created. The connected organization currently reports the Free plan. A database snapshot showed 13 current connections out of a configured 60, which is a momentary observation, not a user limit. Docker's daemon was unavailable, and no separate staging branch was visible. A complete yes/no result for 50, 500, 1,000 or 10,000 DAU remains unproven. Supply an isolated staging project with test credentials, or configure the existing project's service credential locally, to continue authenticated, AI, upload and mixed-workload tests.

## Authenticated production scan retest — 28 October

A bounded live test used ordinary disposable accounts and the same known garment photo. At two simultaneous scan requests, one succeeded in 8.85 seconds and identified the expected jacket and color; the other returned HTTP 503 (`ai_classification_unavailable`) after 50.68 seconds. The two-request stage therefore failed and the planned five- and ten-request stages were not run. All ten created users were reported deleted, and an independent database count found zero users remaining for this test run. [Test evidence](readiness-evidence/oct28-capacity-scan-retest.json).

**Decision:** Do not claim reliable end-to-end capacity at any requested user level on the basis of this failed scan test. Even a 50-DAU pilot could experience simultaneous scans, though this two-request sample cannot establish a failure rate, a hard two-user limit, or a maximum DAU. The underlying cause of the 503 is not established. Investigate provider/function logs and retest using a broader photo set and sustained mixed workloads before assessing higher tiers. The public catalog and prior scoped database tests cannot compensate for this failure.

## Outfit burst and scan routing follow-up

A separate live outfit-generation test passed at two simultaneous requests (2/2 AI outfits). At five simultaneous requests, all five returned valid outfits, but one used deterministic fallback instead of AI, so the strict AI-success stage failed (4/5 AI); the ten-request stage was not started. This is a small, same-shape test and does not establish a provider concurrency limit. The test reported all seven temporary accounts deleted and sessions revoked. [Evidence](readiness-evidence/outfit-concurrency-12826b6bd5f6.json).

The repository's scan Edge Function calls NVIDIA directly for classification, with a 45-second provider timeout; Replicate supports other image operations, not classification failover in that path. A separate Express API has a multi-provider failover implementation, which is not the scan Edge Function's route. Function logs during the failed scan confirm configuration was loaded, but do not identify the exact cause of the 503. Source inspection alone does not establish exact deployed source equivalence.

## Private media retest — 28 October

A test-helper dependency was outdated, causing the first attempt to stop during setup without exercising Storage. The helper was updated to load the current request-deadline module. The second run used two disposable owners, a 2.92 MB original PNG and a 224,579-byte preview created with macOS `sips` from that image (not a native-app generated preview). Owner uploads and access checks succeeded; 6/6 sequential original reads matched bytes (maximum 7.28 seconds). At 2 preview reads/second for 10 seconds, 20/20 succeeded with no drops (p95 2.88 seconds). At 5 reads/second for 10 seconds, 35 of 50 scheduled requests started, 15 were dropped by the 20-request client concurrency cap, and six started reads failed with 15-second timeouts; the test stopped before 10 RPS. These results show a failed end-to-end test from this client and network, not proof that Storage alone is the bottleneck, a user limit, or native-device capacity. It uses two repeated object URLs, not a realistic diverse-user cohort. All test users and objects were independently verified removed. [Raw evidence](readiness-evidence/production-media-delivery-2026-10-28T18-08-28-964Z.json).

## Authenticated populated reads — 28 October

The initial fixture setup attempted 50 garments per free account and was rejected by the current 20-item free-plan limit; it did not run load. The harness was updated to use 20 items per free account, with a configurable bounded rate and a separate result filename. At 2 requests/second for 120 seconds, all 240 reads succeeded with zero drops across 20 temporary users, 400 garments and 100 outfits; median latency was 464 ms, p95 2,062 ms and p99 4,034 ms. The workload **failed** the harness latency criteria of p95 below 1 second and p99 below 2 seconds. All accounts were reported deleted; an independent query found no remaining tagged accounts created during these attempts. [Raw results](readiness-evidence/oct28-populated-free-read-2rps.json). This tests app-shaped HTTP reads from one network, not image bytes, AI, writes, or a 500-DAU population; it does not isolate server latency from network effects.

In the separate 5-RPS media stage, all six timed-out reads had already received HTTP 200 headers and failed while receiving their response bodies. The evidence does not distinguish a network/client bottleneck from the storage delivery path.

### Read latency diagnostics

In the 240-request read run, client-observed time to headers was p95 2,061 ms and p99 4,031 ms; body consumption was p95 2.5 ms and p99 28.9 ms. In a matching time window, gateway logs for the four tested GET paths and Node user agent contained 242 records, so they are not an exact one-to-one correlation to the 240 requests. The recorded upstream-service-time p95 was 15–35 ms by path, with a maximum of 96 ms. This indicates the measured database/API upstream processing was much shorter than the client-observed header wait in this window; gateway/client networking and other outside-upstream delays remain candidates, not proven causes. No production configuration was changed.

The five-request outfit test's fallback result still recorded an NVIDIA provider attempt as `ok`, and the function logs show an NVIDIA response was parsed before the deterministic result was produced. Therefore that fallback is not evidence of provider outage or provider rate limiting. The logs available here do not establish the precise reason the app selected fallback; the test failed only its stricter requirement for AI-sourced output, while all five returned valid outfits.

### Deployed scan routing verification

The Supabase connector retrieved active `ai-process` version 52 after the test. Its deployed source directly calls the NVIDIA classifier for `classify` and `classify_frames`, applies a 45-second timeout, and returns `ai_classification_unavailable` when classification is unavailable. The deployed classification path has no alternate provider call. Replicate configuration is present for other operations. This verifies routing for that deployed version; it does not determine why the individual request failed.

The deployed scanner converts several distinct conditions into the same `ai_classification_unavailable` 503: a non-success provider response, timeout/network exception, or missing/invalid classification data. It logs the status for a non-success HTTP response but catches other exceptions without recording their cause. The absence of an error-status log in this test does not identify which unlogged condition occurred. Add privacy-safe reason and timing telemetry before the next diagnostic if root-cause attribution is required; do not log images, keys or full provider responses.

### Lower-rate authenticated reads

At the hypothetical 50-DAU planning target of 0.2 app reads/second, 24/24 requests succeeded with zero drops, median 709 ms, p95 1,290 ms and p99 1,323 ms. The test failed its p95 <1,000 ms criterion. Only six distinct authenticated users were exercised by the short 24-request mix, despite preparing 20 populated accounts; this is not a 50-user simulation and cannot establish a population limit. All 20 accounts were reported deleted, and an independent query found no remaining tagged test users for the run. [Evidence](readiness-evidence/oct28-populated-free-read-02rps.json).

### Earlier regional comparison

An 18 October controlled Edge-to-Storage JPEG comparison found p95 446–680 ms in Tokyo versus 2,194–2,214 ms in Frankfurt at 2/5/10 reads per second. With a 40-read generator cap, all 170 requests per region succeeded; an earlier 20-read cap caused Frankfurt-side drops. This supports geographic path and generator concurrency as possible contributors to observed performance, not a proven explanation of the 28 October direct-client timeouts or a hard storage limit. It used one owner, fixed region order, short bursts, and different images/client locations from the current test. [Prior regional evidence](readiness-evidence/native-jpeg-cap40-summary-82535b2d-1ad4-4a71-b204-5cdd6531fb69.json).

Gateway logs for the 28 October media window include 61 HTTP 200 authenticated-object GET entries and two HTTP 400 entries (the run included negative access controls). The upstream-service-time field was unavailable on these entries, so the gateway data cannot attribute the six client body timeouts to Storage or to the network, and the 400s cannot be attributed to load without request-level correlation.

## Capacity verification matrix

| Target | Current test evidence | Missing before a reliable yes |
| --- | --- | --- |
| 50 DAU | 0.2-RPS reads: 24 successes, p95 1.29 seconds (fails chosen target); six authenticated identities. Scan: 1/2 success. | Representative 50-user activity and bursts, repeated scan quality/reliability, media, writes and real-device checks. |
| 500 DAU | 2-RPS reads: 240 successes, p95 2.06 seconds (fails chosen target); 20 identities. | Actual 500-user behavior and cardinality, mixed AI/media/write workload and sustained performance. |
| 1,000 DAU | A 4-RPS, two-minute read-only run passed for 20 accounts; AI, media, writes and the wider population were excluded. | Representative 1,000-user population, mixed 4-RPS-plus bursts and independent AI/media workloads. |
| 10,000 DAU | Historical 40-RPS read soak passed, but used 20 accounts and excluded AI/media/writes; the latest lower-rate tests missed latency criteria. | Production-like 10,000-user cardinality, mixed sustained load, provider and media peaks, recovery and operational verification. |

The RPS figures derive from the assumptions above, not measured usage. The lower-rate failures justify investigating before escalating production load; they do not prove the app cannot serve those daily populations. The stricter outfit-burst criterion requires AI source; a deterministic fallback may still produce a usable outfit and should be judged separately from outright user-visible failure.

The read harness's throughput calculation was corrected after the low-rate run: it now divides by at least the nominal test window when the last response finishes early. The recorded 0.207 RPS for that run overstated its nominal-window rate; 24 scheduled/completed reads over the 120-second nominal window is 0.2 RPS. Latencies, failures and the failed p95 result are unchanged. A regression check and all 39 offline load-harness checks pass; the existing evidence file remains unchanged.

## Wear-log writes at 2 RPS — 28 October

A separate 120-second production test sent 240 wear-log upserts at 2 requests/second across 20 disposable owners, with paired writes to check idempotency. All 240 responses succeeded and no requests were dropped. Every owner's six distinct logs and wear counters reconciled correctly. Median latency was 664 ms, p95 1,816 ms and p99 2,326 ms; the stage **failed** the existing p95 <1 second and p99 <2 seconds criteria. It was a write-only workload from one client, not a full 500-DAU or mixed test. All 20 accounts were reported deleted and a separate query found no remaining tagged accounts created during the run. [Evidence](readiness-evidence/oct28-wear-load-2rps.json). The write harness was updated for this bounded rate and to avoid overstating throughput when the last request completes early; all 39 offline load-harness checks passed.

The write test's client-observed time to headers was p95 1,816 ms; body consumption p95 was 1.94 ms. Gateway logs for the matching window contained only 230 of the 240 write requests, with recorded upstream p95 39.65 ms and maximum 313 ms. The test client received an upstream timing header on all 240, whose maximum was also 313 ms. As with reads, upstream processing was much shorter than client-observed wait, but this does not identify the exact network/gateway delay or prove performance from another client location.

A subsequent five-request unauthenticated network probe to `/auth/v1/health` returned HTTP 401 as expected without an API key. Fresh curl connections observed DNS 1.8–3.0 ms, TCP 91–115 ms, TLS completion 201–365 ms and first byte 351–531 ms. This is a small different-endpoint, different-client sample; it only shows that connection setup was measurable at that moment and cannot explain the earlier tail latencies or prove authenticated capacity.

## Local PostgreSQL benchmark — 29 October

An isolated local PostgreSQL 16 database was seeded with 10,000 synthetic users, 500,000 wardrobe items and 50,000 outfits. A 60-second, 40-client `capacity.pgbench` workload offered 200 transactions/second and completed 12,081 transactions with zero failures, 201.42 TPS and 4.10 ms mean transaction latency. [Raw result](readiness-evidence/oct29-local-pgbench.log). This tests local owner-scoped queries and a rate-counter operation only. It does not test the hosted database, HTTP, network, actual Auth, images, AI, devices, or concurrent active users, and is not evidence of end-to-end 10,000-user readiness.

A clean replay of the current numbered migrations failed: the foreign-key index migration expects `admin_permissions` and `influencer_referrals`, which the numbered replay did not create. The first table exists in an unnumbered script; the latter was represented by a minimal test-only local stand-in based on observed production columns. A subsequent migration references `public.update_updated_at_column()`, absent from the replay; that migration was skipped for the benchmark. The Auth stub also needed `raw_app_meta_data`. Therefore the benchmark uses a reconstructed, non-production-equivalent schema and does not prove a clean installation or staging parity. The connected Free-plan project rejected a staging branch because branching requires Pro.

## Observed traffic caveat — 29 October

A 24-hour aggregate gateway-log sample included at least 795 catalog reads and 305 wear-log requests, with no gateway 429/5xx among those two path groups. It includes the disposable-account capacity tests and may include other requests, while log retention/coverage and account attribution are not established. Therefore these counts are not a count of real customers, actual DAU or an unbiased peak-demand estimate, and they cannot validate the workload assumptions or certify capacity.

A read-only production aggregate on 29 October found 130 clothing rows across 35 owners, including 18 inline `data:` image URLs; the largest image URL field was 4,047,638 bytes, and two rows had a nonempty thumbnail URL. These are field/row counts, not DAU or evidence that those objects can be displayed on a particular device. The current populated read tests used short synthetic image URLs, so their response sizes do not represent the largest existing inline payloads. Image migration and real-payload tests remain necessary for a capacity conclusion.

## Inline-image fixture diagnostic — 29 October

The current app wardrobe sync selects both `image_url` and `thumbnail_url`. A small production diagnostic attempted to insert one synthetic approximately 3.89 MB base64 image into a disposable owner's wardrobe before making five sequential reads. The insertion attempt timed out after the 20-second request deadline; no read stage ran, and it is unknown whether the backend received or committed the request before cancellation. The account was deleted and a separate query found no remaining tagged test account. This single client attempt is an inconclusive setup failure, not a measured read result or proof of a server capacity limit. [Evidence](readiness-evidence/oct29-inline-read-probe.json). The script wrote its failure in the report despite exiting zero; use the report fields rather than exit status for this run.

## Scan repeat — 29 October

The synchronized production scan test was repeated with the same known photo and two ordinary disposable users. Both returned HTTP 200, but one correctly selected a beige jacket in 11.7 seconds and the other selected blue jeans in 35.4 seconds. This failed the expected-category/color criterion and stopped before the five- and ten-request stages. All ten setup accounts were reported deleted and an independent query found zero remaining for the run. The prior two-request 503 did not recur in this small repeat; this run instead reproduces the earlier known-photo selection error. Neither failure establishes a fixed concurrency limit or broad accuracy rate. [Evidence](readiness-evidence/oct29-capacity-scan-repeat.json).

## Authenticated reads at 4 RPS — 29 October

A two-minute production diagnostic sent 480 app-shaped reads at 4 requests/second across 20 disposable users (400 synthetic garments, 100 outfits). All 480 succeeded with zero drops; p50 407 ms, p95 875 ms, p99 1,571 ms, achieved 3.99 RPS. This passed the read harness's thresholds. All 20 accounts were reported deleted, and an independent query found no remaining tagged test users. [Evidence](readiness-evidence/oct29-populated-free-read-4rps.json). This later, short, read-only success does not erase the previous 0.2- and 2-RPS latency failures, exercise 1,000 distinct users, or test AI, media, writes or native rendering. The 4-RPS connection to 1,000 DAU is an assumption, not measured behavior.

## Authenticated reads at 40 RPS — 29 October

A two-minute production read-only diagnostic used 20 disposable free accounts with 400 garments and 100 outfits. All 4,800 app-shaped reads succeeded, with no drops or failed validations; 39.88 RPS achieved, p50 394 ms, p95 464 ms and p99 1,051 ms. This passes the harness's scoped read thresholds. All 20 accounts were reported deleted and a separate query found no remaining tagged accounts created during the run. [Evidence](readiness-evidence/oct29-populated-free-read-40rps.json). This result excludes writes, image downloads, AI, actual large inline images, 10,000 distinct hosted users and sustained mixed load. It does not certify 10,000 DAU and does not erase prior low-rate tail-latency failures. The 40-RPS target rests on unvalidated usage assumptions.

## Concurrent reads and wear writes at 40 RPS — 29 October

A two-minute production test combined 32 app-shaped reads/second and 8 wear-log upserts/second across 20 disposable free users, with 400 synthetic garments and 100 outfits. All 3,840 reads and 960 writes completed with zero failures or drops. Read p95/p99 was 427/648 ms; write p95/p99 was 447/666 ms. All 20 users' stored log counts and wear counters reconciled, and all 20 accounts were deleted; an independent query found no tagged users remaining. [Evidence](readiness-evidence/oct29-mixed-read-write-40rps.json). This passed the scoped test criteria but used 20 accounts, repeated synthetic data and one load generator, and excluded AI, image delivery, uploads, subscriptions and native rendering. It cannot certify the complete application or 10,000 distinct hosted users; earlier scan and media failures still apply.

## Interrupted mixed-load soak — 29 October

A planned 10-minute production test at 32 read RPS + 8 wear-write RPS was interrupted after errors and read-generator drops rose sharply. At the last 180-second progress checkpoint the client reported 917 failed reads, 605 dropped scheduled reads and 352 failed writes. These are cumulative progress snapshots, not final totals; they include client transport/timeout and validation failures and do not identify their source. The runner did not produce a final JSON report because the interruption terminated it before cleanup. The 20 explicitly tagged disposable accounts were subsequently deleted using a count-guarded exact run-ID predicate, and a follow-up query found zero tagged accounts and zero orphan fixture clothing rows. [Progress log](readiness-evidence/oct29-mixed-10min-interrupted-progress.log).

A same-window gateway aggregate for the tested paths showed HTTP 200/201 records only, but coverage and one-to-one correlation are not established; it cannot explain or disprove the client-side failures. The ten-minute soak failed and does not establish a server throughput ceiling. The earlier two-minute mixed pass does not outweigh this longer-run failure. Investigate transport and generator health, network, and service telemetry before another long run.

After the interrupted soak, the mixed harness was updated to record each created test user in its report as setup proceeds, respond to SIGINT/SIGTERM, stop both streams upon a reported request failure or drop, and save completed stream results before verification/cleanup. Controlled offline tests confirm that read and write failure callbacks can stop further scheduling; all 41 load-harness tests pass. This does not guarantee recovery from a hard process kill or missing fixture-creation response, and the updated mixed runner has not yet been validated in a live abort/cleanup exercise. The user declined creating a separate staging project in the connected organization.

A controlled live SIGTERM exercise stopped the updated mixed harness during account setup after it recorded eight disposable accounts. It returned a failed/aborted report, deleted all eight, and an independent query found no remaining tagged users. [Evidence](readiness-evidence/oct29-mixed-signal-cleanup.json). This validates graceful termination during setup only; it does not prove traffic-stage abort cleanup, hard-kill recovery, or solve the prior soak failure.

A second controlled SIGTERM exercise stopped the updated runner during mixed traffic after 1,229 read and 308 write attempts. Both streams recorded `aborted: true` and failed as intended; all 20 tagged fixture accounts were deleted and independently verified absent. [Evidence](readiness-evidence/oct29-mixed-traffic-stop.json). This verifies graceful stop and account cleanup during active traffic, not hard-kill handling, production capacity, or a successful soak.

## Guarded mixed soak repeat — 29 October

The guarded 10-minute production repeat stopped after approximately 268 seconds when the read generator hit its 200-in-flight cap and dropped one scheduled request. The abort then cancelled 200 in-flight reads and 51 in-flight writes; the report counts these cancellations as failures, so they must not be interpreted as 251 independent backend errors. The exact reason requests accumulated is not established. Both streams were marked aborted and the 10-minute target failed. All 20 tagged accounts were deleted and independently verified absent. [Evidence](readiness-evidence/oct29-mixed-soak-guarded.json). This repeat supports a reproducible end-to-end/client-path stall under the specific generator and network conditions, but does not establish the cause or a server capacity ceiling. It excludes AI, image bytes, uploads and native rendering.

For the guarded soak, successful/received read headers had p95 1,573 ms and p99 3,185 ms; body-consumption p95 was 5.1 ms. The write generator's observed event-loop delay maximum was 33 ms and maximum scheduling lag 14 ms, evidence against a long local JavaScript event-loop pause but not against transport, OS/network or gateway contention. Aggregated gateway records across the five tested paths fell from 111 to 43 per five-second bucket between 04:04:30 and 04:04:55 UTC; those records reported no 5xx. These are response log counts, not an exact request-by-request trace or proof that requests did not reach the service. The cause remains unconfirmed.

## Lower-rate mixed soak attempts — 29 October

A 4-RPS mixed run (3.2 reads and 0.8 writes/second) could not complete. The first runner exited with Node's unresolved top-level-await exit code 13 before writing a final report; 20 tagged accounts were recovered and deleted using a count-guarded exact run-ID cleanup. The runner was adjusted to keep Node alive while outstanding transport promises settle. On the second attempt, the guard stopped traffic after approximately 175 seconds when a profile fetch reported `TypeError` / `ETIMEDOUT`; subsequent aborted requests are cancellations, not independent server errors. The runner finished but all its HTTP cleanup calls failed during the same period. The 20 exact tagged accounts were removed separately with a count-guarded database cleanup; a follow-up query found no users from either run or orphan fixture items. [Partial initial evidence](readiness-evidence/oct29-mixed-soak-4rps.json), [guarded retry](readiness-evidence/oct29-mixed-soak-4rps-retry.json).

This is a failure of the complete generator-to-service path for that run, not proof the server itself timed out or a precise user limit. Automatic cleanup still depends on network reachability; an independent cleanup route and durable run-ID ledger remain necessary. No 10-minute 4-RPS mixed pass was obtained.

A 29 October low-rate, non-concurrent paired network check sent 12 fresh curl connections each to the Supabase unauthenticated health endpoint and a public control endpoint over approximately 110 seconds. All Supabase responses were the expected unauthenticated 401, all control responses were 200; Supabase first-byte times ranged 248–472 ms and control 245–336 ms. [Evidence](readiness-evidence/oct29-network-paired-probe.json). This was after the mixed soak failure, not simultaneous with it, uses curl rather than the Node generator and cannot rule out intermittent local/network or service issues during the earlier run.
