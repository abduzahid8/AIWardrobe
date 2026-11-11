# Stylist 6 cloud staging — current acceptance evidence

Status: **STAGING NO-GO**, the single iOS attempt failed before quota admission. No production cutover or merge is authorized.

## Applied infrastructure

- Supabase project `fyqpifmrsftsfqibhwhy`: migration **20261111154741_stylist6_idempotency** is applied. Do not run blanket `db push`.
- Isolated Edge Function `aiw-stylist6-node-staging` is deployed with gateway JWT verification enabled. Its canonical proxy and signature module boot successfully.
- Production and S5 staging code bundles remain unchanged; function counters are now 209 and 6 (details below). S5 HMAC, entitlements and subscription records were not changed.
- Durable claims use `(user_id, request_id)` plus the canonical SHA-256 request fingerprint. RLS and service-role-only RPC/table access were verified. Overlapping SQL claims admit one winner; conflicts and terminal replay were verified and probe rows cleaned.
- Claims and quota consumption are separate transactions: at-most-once admission, not atomic exactly-once billing or delivery. An ambiguous dispatch must never be retried with a new ID.

## Credential and transport investigation

Supabase CLI 3.1.0 read checks succeeded: identity in 15 seconds and secret-name listing in 9.22 seconds. Initial sandbox execution separately failed local telemetry-file permissions and DNS; these are not service-side authentication errors. No CLI authentication rejection was observed. Earlier writes timed out without an HTTP error, so those timeouts alone do not establish a service-side failure.

This continuation successfully rotated and paired the S6-only HMAC through official CLI secure inputs. Vercel received it only for Preview branch `codex/stylist6-visual-first-20261110`; Supabase received `STYLIST6_NODE_SHARED_SECRET`. No value was printed or committed. Temporary key input was deleted.

Automatic approval review initially rejected a project-wide automation bypass as broader than staging. The owner explicitly approved that scope. Vercel accepted a new 32-character project automation credential. The first 60-second Supabase write timed out and that Vercel credential was revoked; a second bounded 180-second attempt succeeded. `STYLIST_VERCEL_BYPASS_SECRET` is now configured through secure input. Deployment Protection remains enabled. This bypass is project-wide, not a deployment-only grant; S5 application HMAC is still independent.

Real non-metered Supabase probes: missing Authorization 401; malformed JWT 401; a public project JWT passes the gateway and the proxy reaches protected S6 compute. The compute returns application JSON `stylist_compute_unconfigured` with exact version `stylist-6-cloud-staging-1`. Because that route checks HMAC before checking its Supabase configuration, this proves protected transport and matching HMAC but is a failed compute-readiness check.

Owner-only presence diagnostics identified **only the Supabase service-role variable missing at runtime**; URL and anon configuration were present. The existing authorized project service-role key was retrieved through the official Supabase key-management mechanism, kept in process memory, and restored only to the S6 Preview branch via stdin. It was neither displayed nor persisted. The branch server-credential update returned exit 0. The rebuilt Preview reached successful non-metered S6 admission as recorded below.

## Single-request safeguards

`backend/stylist-node/scripts/stylist6-staging-smoke.mjs` now requires `--single-request`; the old three-scenario invocation is disabled. It requires an explicit stable request ID and dedicated user ID matching validated login, checks the legitimate server entitlement, snapshots the exact `edge:aiOutfits` counter, verifies non-metered protected admission, and reads durable claims before dispatch. An exclusive local dispatch ledger survives timeout/failure and prevents rerun. There are no automatic retries. Optional replay is allowed only after a successful outcome and matching persisted fingerprint/status, with the identical request.

The owner approved the existing signed-in Simulator profile as the dedicated account. Server evidence before dispatch: legitimate effective Premium tier, one active Auth session, initial `edge:aiOutfits` usage **1** on **2026-11-11 UTC**. No entitlement writes were made. Private account identifiers are omitted here.

## Checks and native record reconciliation

- 131 S6/Node tests pass, including four offline single-request safety tests: default mode refusal, exactly one paid dispatch, timeout fuse surviving rerun, and failed admission/existing-claim refusal.
- TypeScript passes. No paid live smoke or NVIDIA inference is counted as a passing test.
- Earlier native attempts failed with disk exhaustion and one observed ExpoAsset redbox. A later third native build succeeded; its actual build log ends `BUILD SUCCEEDED`, the app was installed and launched, Home loaded photographs, and Profile opened. Those later successful observations supersede the earlier failure-only report; neither proves cloud acceptance.
- Fresh Simulator inspection in this continuation again opened Home and Profile without the reported ProfileScreen crash. It showed the signed-in test profile and free-trial UI. Server entitlement was checked separately.
- Client diagnostic identifier remains `41c04d4bfc02` on the existing Metro session. This is not a claim that its JavaScript bundle is at the latest server commit. No new native build was run in this continuation.

## General CI, separate from S6

The earlier dedicated S6 workflow passed at e431dbb. General CI did not pass. Its quality job hit the 15-minute limit after real failing suites; cancellation is not success and cannot all be explained by PR concurrency. The database job and retry both failed Ubuntu package-mirror DNS before database assertions ran. The run-level conclusion later appeared cancelled; the database job failures remain material.

At exact pre-completion SHA `3d7af3dfbf81f94be0201204b84a4950375eb488`, `aiOccasion.e2e.test.ts` and `outfitVisualDiversity.test.ts` also failed: 19 failed, 101 passed. Only these two failures are established as pre-existing; other general failures remain unclassified. No assertions or S5 fashion policy were weakened.

## Remaining acceptance / rollback

No successful authenticated generation, exact admitted-request quota delta, replay HTTP result, or rendered cloud outfit screenshot is asserted until recorded below. One paid generation remains the hard cap. No automatic retry is allowed after dispatch.

Rollback: stop/delete only the isolated S6 function or revert its staging pin/config; preserve S5. Retain the inert RLS-protected claim history to avoid re-admission. Any schema removal requires traffic stopped and claim/quota reconciliation first. No destructive rollback was performed.

## Verified repaired transport (continuation)

Final repaired compute: `dpl_3D87thVhyGCi5xBsNQj81UiywcpJ`, READY Preview, immutable URL `https://aiwardrobe-stylist-compute-j0r1t0yea-abduzahid8s-projects.vercel.app`, compute SHA `52c82c79082d4495c878e7b4884c540ddf6fc3dd`. GET returns application-level 405 with `stylist-6-cloud-staging-1`.

Current Supabase S6 Edge Function: **v8 ACTIVE**, `verify_jwt=true`, bundle SHA-256 `81a308955af6be4da8494c5b2be8149c3151c3adbc44529608d40a794d1b3d46`. Proxy pins the repaired immutable Preview above. The redeploys were limited to this isolated proxy and required by verified credential/environment failures and immutable-target updates; no migration was reapplied.

Real Supabase-to-Vercel malformed-body probe now returns **400** `invalid_stylist6_request`, stage `stylist6_admission`, exact `stylist-6-cloud-staging-1`. It passes the gateway using the public legacy project JWT, protection and S6 HMAC, and reaches the configured handler before rejecting the body. This proves server transport/configuration; it does not prove a valid end-user generation.

Read-only account recheck: legitimate Premium, usage remains **1**, S6 claims **0**. No metered generation or NVIDIA inference has occurred. The Mac was initially locked; the owner subsequently unlocked it and the single iOS attempt below was performed. No second generation request is permitted after the single attempt recorded below.

Latest general CI on `52c82c7`: run `43064683845`. The database/RLS/concurrent-quota job **passed**, including migration replay, ownership, entitlement, quota and webhook assertions. This supersedes the earlier package-DNS blocker for this run. The quality job passed Edge typecheck, TypeScript and ESLint and is still running unit tests; no overall pass is claimed.

## Production preservation correction

Final readback reports `generate-outfits` **v209** and S5 proxy **v6**, both ACTIVE. Their bundle hashes remain byte-identical to the initial v207/v4 baseline: production `9bb8efb4580760691d9df7f95cdf3aa8f203e743604d369e5402eaee62ec650d`; S5 proxy `778a08ed2f75ecddc28ed74d374abb1a87dd931ab2d58b83ff5f01172ed8dd46`. Their code-deployment timestamps and entrypoints also remain unchanged. The counter increments coincide with project-wide secret configuration updates, consistent with function refresh rather than a new S5 code deployment. S5 HMAC was never written, and production code was not promoted or replaced. It would be inaccurate to claim the counters themselves stayed at 207/4.

## Final single iOS attempt — FAILED, no retry

The owner unlocked the Mac, explicitly authorized the existing Simulator account, and explicitly approved public disclosure of staging deployment/configuration status, aggregate quota counts and iOS results. No private account identifiers or credential values are included.

Fresh pre-dispatch server check: legitimate Premium entitlement, daily `edge:aiOutfits` usage **1**, and **0 S6 claims**. The actual native app selected **Stylist 6 cloud staging**, catalog clothing, Classic / Work, hard cap **25,000,000 UZS**, NVIDIA unchecked. One Generate action was performed at approximately **17:35 UTC / 22:35 Tashkent on 2026-11-11**. An exclusive local dispatch record was written before the action. The existing three-scenario script was not run.

The app showed its loading state and then **“Edge Function returned a non-2xx status code.”** It returned no outfit and no fallback. Supabase logs identify one S6 POST returning **502**, duration **4103 ms**, authenticated role, and the correct test-account identity (verified privately). This is a failed test, not successful cloud generation. No second Generate action, retry, or replay request was submitted.

The server counter remains **1 → 1, delta 0**; no durable S6 claim was created. Thus no admitted paid generation occurred, and exact-one-debit/replay acceptance remains unproven. The single authorized attempt is finished; zero debit does not authorize an automatic second attempt. No catalog-SKU/image/price/budget rendering acceptance or successful cloud screenshot can be claimed because no outfit was received. NVIDIA was disabled and execution did not reach inference.

## Catalog contract root cause and bounded repair

The failed request reached real Supabase Auth (`GET /auth/v1/user` 200 from Node), rate admission (200), and all four catalog discovery calls (200). The deployed discovery RPC returns 100 candidates per slot, but its JSON projection **omits `is_active`**. Its current SQL explicitly selects only active catalog rows in both source paths. The S6 general catalog adapter requires `is_active === true` and therefore rejects the entire unnormalized RPC projection, yielding an empty server pool before claim/debit. This reproduces the observed 502/no-claim boundary; the client error itself did not expose the detailed backend body.

Fixed only the S6 trusted server-RPC boundary to restore the active-row contract fact when that field is omitted. Explicit false stays false. The general adapter still rejects raw input without an active flag; client inventory is still ignored. No SQL migration, production S5 code, entitlement or provider change is needed. A regression test covers actual RPC projection shape, usable normalized rows, explicit inactive rejection and strict raw-adapter behavior. **132 S6/Node tests pass.** No second authenticated generation was used to validate this repair.

Latest general CI quality job `130676343837` ended **cancelled** after multiple actual failing suites and its 15-minute limit; database/RLS/concurrent quotas job `130676344460` passed. Dedicated S6 workflow `43064675424` passed on `52c82c7`. Full general CI remains a release blocker.

**STAGING NO-GO**: the first authenticated iOS request failed; no successful remote outfit, admitted quota delta of one, HTTP replay behavior or rendered cloud result has been demonstrated. A future metered attempt requires fresh explicit authorization, after the repaired server artifact is deployed and non-metered checks pass.

A non-metered live metadata-only check fetched 400 discovery rows: the original projection produced **0 items / 400 inactive skips**; the repaired server normalization produced **400 items / 0 skips**. It performed no generation, image fetch or inference.
