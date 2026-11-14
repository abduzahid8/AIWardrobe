# First seven days after a separately approved activation

No action here has run remotely. Every day: named owner reviews the pinned RC/cohort, actual evidence, open BLOCKER/HIGH issues and spend cap; captures a decision and next review time. If an owner, effective kill, monitoring feed or recovery artifact is missing, do not start activation. Use docs/RUNBOOKS.md for incident roles and launch/ROLLBACK.md for this RC's exact inverse files.

| Day | Operating checklist | Decision/evidence |
|---|---|---|
| 0 | Approve internal-only Stage 0; verify exact migration ledger/grants/RLS, pinned provider config and server billing; run startup/offline/core/restore; exercise kill and recovery; inspect early failures | Record approvals and before/after hashes; no public invitations; stop for any BLOCKER |
| 1 | Review crash/startup/auth/billing and session-owner isolation; execute remaining Tier A on four device classes; inspect restore/unknown entitlement paths | Counts/denominators and manual evidence; no expansion while critical path incomplete |
| 2 | Review outfit quality and BuyCheck hard constraints, duplicate/fit/family/price corrections; interview first users about saved/worn/helpful outcomes | Triage top damaging failures; choose minimal corrective hypothesis |
| 3 | Review actual client/server latency, memory/scroll findings, provider calls/spend, fallback/low confidence and cache hits | Enforce approved cost cap; disable harmful path; no provider added just for novelty |
| 4 | Observe onboarding and first-value task; inspect confusion, permission denial and empty/sparse wardrobes | Fix highest-impact activation friction from recordings/voluntary reports |
| 5 | Review return-to-plan/wear outcomes on another day; revisit earlier negative feedback and confirm resolution | Separate no exposure, no value and no return; do not substitute opens for retention |
| 6 | Fix high-impact reproducible issues only; rerun targeted + required RC checks; update known limits and support scripts | Evidence → hypothesis → minimal patch → flag → metric → approved rollout → observed result → keep/revert |
| 7 | Run go/no-go against zero BLOCKER, required device/billing/compatibility/rollback gates; inspect cohort outcomes, costs and team capacity | Signed next-cohort decision with evidence and uncertainty; otherwise hold/revert. Never auto-expand |

Incident checklist: preserve sanitized evidence/request IDs; stop affected route through verified control; confirm calls/writes stopped; communicate through the explicitly authorized channel; restore pinned prior state when approved; verify owner isolation and data counts; record incident and avoid speculative new work. Never clean production by deleting user data as an emergency shortcut.

Weekly roadmap input template: Problem; affected cohort/RC; evidence/sample size and missing data; hypothesis; minimal change; effective flag and inverse; prospective outcome metric/guardrail; approved exposure; measured result; keep/revert. Answer all eight product-scope questions in FEEDBACK.md. New major features stay deferred until repeated user outcomes support them.
