# AIWardrobe release candidate preparation

Local preparation dated 2026-11-03 (Asia/Tashkent). Feature development is frozen. No push, deployment, hosted SQL, publishing, configuration change, cohort activation or provider enablement is authorized or performed by this packet.

Run `npm run release:check`. It writes the RC identity, checks, raw logs and source hashes into `docs/launch/evidence/`. Exit codes: 0 PASS, 1 FAIL, 2 MANUAL REQUIRED. A nonzero result must never be represented as release-ready. Reports identify the base commit and the dirty worktree fingerprint; a commit alone does not describe these changes. Re-run after any source/config change. Build numbers remain iOS 65 and Android 63; store uniqueness is unverified, not reserved by this work.

| Requirement | Reviewable artifact |
|---|---|
| RC identity and checks | evidence/release-candidate.json, evidence/source-snapshot.json |
| Blocker triage | BLOCKERS.md, blockers.json |
| Tier A/B/C journeys and flags/fallbacks | FEATURES.md |
| Manual scripts, device matrix, perf/memory gates | BETA_QA.md, device-results.csv, performance-results.csv |
| Isolated native Release build/startup supplement | NATIVE_LOCAL_QA.md, evidence/native-harness-session.json |
| Actual App-root signed-out/offline supplement | OFFLINE_APP_QA.md, evidence/offline-app-startup.json |
| Synthetic personas | `npm run beta:personas` (stdout only) |
| Migration replay and limitations | MIGRATIONS.md, evidence/migration_tests.log |
| Ordered activation and exact local inventory | REMOTE_ACTIVATION.md, activation-config.json |
| Recovery | ROLLBACK.md, backend/supabase/rollback/ |
| Prepared server paid-work stop and recovery | PAID_WORK_CONTROLS.md |
| Cohorts, acceptance, user value | BETA_PLAN.md |
| Feedback/correction schema and evidence analysis | FEEDBACK.md, `npm run beta:review -- /path/to/export.json` |
| First-week dashboard | DASHBOARD.md |
| Days 0–7 operations and evidence-driven iteration | FIRST_7_DAYS.md |
| Limits, debt, deferred scope | KNOWN_LIMITATIONS.md, QUALITY_DEBT.md, ../DEFERRED_FEATURES.md |

The report deliberately remains BLOCKED while required device, billing, installed-client compatibility and remote rollback/kill-switch evidence is absent. Passing isolated tests does not establish those claims. Final approval concerns a pinned, reviewable packet; approval of one remote step never authorizes the remaining steps implicitly.
