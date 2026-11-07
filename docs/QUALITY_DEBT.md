> Current launch gate debt is recorded in [launch/QUALITY_DEBT.md](launch/QUALITY_DEBT.md).

# QUALITY_DEBT.md (Phase 7BP) — living register

## Must fix before launch

(none currently marked — launch blockers in
`src/services/launch/release.ts` `launchBlockers()` are the gate, not this list)

## Fix within 7 days

- Manual QA matrix (`MANUAL_QA_MATRIX`): device hands-on pending.
- Sandbox billing QA (purchase/restore/expired/grace/offline/mismatch).

## Fix within 30 days

- QD-full-suite-flakes: 5 pre-existing failing suites (calendar/occasion,
  pipeline/lifecycle/stylist) predate Phase 5 and are unrelated to launch
  domains — triage or fix post-launch, never by deleting tests.

## Acceptable debt

- Full-suite pre-existing failures documented in Phase 5 report.
- Speculative interfaces from Phases 1–6 kept only where tests prove use
  (7CY); unused ones to be classified here as found.
