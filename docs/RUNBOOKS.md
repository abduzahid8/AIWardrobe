# RUNBOOKS.md (Phase 7AX–7AZ, 7CW)

Registry: `src/services/launch/operations.ts` (`RUNBOOKS`,
`INCIDENT_RESPONSE`, `ALERT_TO_RUNBOOK`). Documentation only — no live
actions in this phase.

## Severity

- **SEV0** data/security: on-call + security own it. Stop risky feature,
  preserve logs, communicate, rollback first.
- **SEV1** widespread core failure: on-call. Kill switch / rollback.
- **SEV2** degraded major feature: feature owner. Disable + fallback.
- **SEV3** minor: file and schedule.

## Runbook index

Supabase / auth / catalog / AI provider / VTON / RevenueCat outages, bad
migration, data-leak suspicion (SEV0: stop, preserve logs, scope, disable),
runaway cost (kill switch, lower limits, cache/fallback — wardrobe stays),
broken paywall (sandbox repro, restore first), price corruption (hide
history, label stale).

## Security incident (7AY)

Stop exposure → preserve logs → rotate credentials when authorized →
scope → disable access → notify per policy. Prepare only.

## Operating cadence (7CW)

Daily (early): crashes, auth, provider failure, billing, hard-rule
violations. Weekly: activation, retention, recommendation outcomes, cost,
quality debt. Monthly: pricing, roadmap, provider comparison, architecture
debt, privacy/security review.
