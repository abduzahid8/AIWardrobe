# RELEASE CANDIDATE PREPARATION COMPLETE

Local preparation and limited native checks are complete for review. Full release acceptance remains blocked. No remote mutation, push, merge, store upload, real account creation or paid-provider activation was performed.

1. **RC identity:** version 3.0.2, iOS 65, Android 63; base commit `5784fc5cff24381cdae982e937d841fca611c7cd`; dirty worktree SHA-256 `3df0c34ea3530b2edc13d60aa48811e116d8e02f2eb99b0599dba41f1ce974d6`. Exact config/migration/Edge/entitlement/schema fingerprints: release-candidate.json. Store build-number availability is unverified.

2. **Launch blockers:** RC-001, RC-002 OPEN. RC-003 historical credential classification/rotation and RC-007 effective flag/kill coverage remain unresolved. Locally closed: RC-004, RC-005, RC-006, RC-008, RC-009, RC-010. See ../BLOCKERS.md.

3. **Critical journey status:** 15 automated release checks PASS, zero FAIL; 268 Jest suites / 3,004 tests; 239 focused critical tests; 21 dependency security checks; six beta-review-tool checks. Lint: 0 errors / 1,418 warnings. Full installed Tier A/B journeys remain MANUAL REQUIRED; critical-journey-status.json lists all 25 named flows individually.

4. **Device QA:** one iPhone 17 Pro / iOS 26.5 simulator exercised isolated feedback/image/BuyCheck and actual signed-out App-root subsets. The four required complete device-class scripts remain UNTESTED. Native billing, notifications, analytics sending and prewarming were disabled in the signed-out probe. Those native receipts are tied to the earlier 16bbfdac baseline; the later server stop and 503 decoder are locally contract-tested, not newly device-certified. See ../NATIVE_LOCAL_QA.md, ../OFFLINE_APP_QA.md and device-results-supplement.csv.

5. **Performance/stress:** signed-out actual App root: native revealed-screen p50 429 ms, range 386–663 ms over 10 cold processes; external process-to-receipt p50 1848 ms includes CLI/IO. Isolated native hard-budget BuyCheck case p50 14.51 ms; raw-image fixture peak RSS 718.77 MiB with no physical/leak conclusion. Host BuyCheck p50 31.02 ms is a different test case/runtime. Native fixture covered 30 copies/220 items/24 references/20 mock history images and cache cleanup. Authenticated screen/performance and low-memory-device gates remain open. Both normal JS exports and two separate nonshipping simulator builds passed.

6. **Migration dry-run:** six pending local migrations PASS on disposable PostgreSQL 16.15. Checks include catalog/RLS/grants/indexes, service-only reward writes, cross-owner visual links/update/supersession, owner-preserving asset deletion and reverse access-disable/data retention. Hosted schema/backup recovery are unverified. See ../MIGRATIONS.md.

7. **Compatibility:** missing/present RPC and baseline/new SQL contracts pass. Installed old/old, new/old, old/new and new/new complete authenticated combinations remain unverified; native signed-out offline startup does not replace them.

8. **Flags:** core routes remain available; advanced platform defaults remain conservative. Six metered Edge entries now share operator-only global/per-endpoint stop controls, prepared and locally verified but NOT deployed/configured remotely. Billing recovery remains available; no duplicate quota charge was added. Hosted/legacy coverage and provider caps still need an approved drill. See ../PAID_WORK_CONTROLS.md and ../FEATURES.md.

9. **Remote activation order:** target baseline/backup → six exact migrations → RPC/grants → reviewed Edge set → storage/security → authorized RevenueCat sandbox → provider limits/deny drill → pinned client → developer-only cohort. Exact files/functions and stop checks: ../REMOTE_ACTIVATION.md; inverses: ../ROLLBACK.md. None executed remotely. The stop posture must be approved and verified for the newly guarded deployments before core paid work is unpaused.

10. **Beta plan:** developer → five trusted → 20–50 → 100–250 → broader beta, with entry/exit evidence, feedback and rollback per stage. No cohort activated. See ../BETA_PLAN.md.

11. **Feedback instrumentation:** optional Profile/BuyCheck/outfit report; explicit helpful/reason; optional comment; sanitized available diagnostics; reviewed native sharing. Native correction input/review/undo/discard verified and keyboard obstruction fixed. No external share invoked. Seven correction-report categories are session-local export assertions; they do not automatically repair catalog data. See ../FEEDBACK.md and native-feedback-ui.json.

12. **First-week metrics:** reliability, confirmed value, AI quality, monetization and provider cost definitions/denominators prepared; no invented baselines. Local observation structures support usage/timing/value/cost summaries; real users/live monitoring destinations remain absent. See ../DASHBOARD.md.

13. **First seven days:** day 0–7 operating checklist, stop criteria and evidence → hypothesis → minimal change → flag → metric → approved rollout → result → keep/revert loop prepared. See ../FIRST_7_DAYS.md.

14. **Deferred scope:** social/feed/community, marketplace rewrite, novelty providers, crypto, gamification, major redesign/framework/microservices and unsupported advanced expansion remain deferred. See ../../DEFERRED_FEATURES.md.

15. **Still required:** named owners; successful/cached authenticated journeys; four-class real/manual QA and production-thumbnail low-memory tests; authorized sandbox restore; complete installed client/backend combinations; target schema/backup rollback and paid-route kill/budget drills; historical credential classification/rotation if genuine; monitoring verification and actual user outcomes. The requirement audit marks these incomplete rather than assigning a fabricated PASS.

RELEASE CANDIDATE BLOCKED

REMOTE ACTIVATION PENDING EXPLICIT APPROVAL
