# Phase 5 Remote Activation — PREPARATION ONLY (do not apply)

This document lists every remote action that would be required to make the
Phase 5 production platform fully operational. **No action here is authorized.**
Each item needs explicit user approval with the 10-point briefing (§0.2).

## 1. Hosted migration apply

- **Exact action:** apply
  `backend/supabase/migrations/20261105120000_phase5_platform.sql`
  (plus Phase 3 `20261103120000_phase3_style_os.sql` and Phase 4
  `20261104120000_phase4_visual_os.sql` if not yet applied).
- **Purpose:** durable storage for sanitized decision events, experiment
  assignments, and recommendation outcomes.
- **User impact:** none until features write; tables stay empty otherwise.
- **Engineering benefit:** reproducible recommendations, measurable funnels,
  attribution windows.
- **Risk:** policy misconfiguration could expose rows; mitigated by the
  `auth.uid() = user_id` pattern used by sibling migrations.
- **Rollback:** drop the three tables (analytics/metadata only; wardrobe,
  fit, style, and shopping cores untouched).
- **Metrics to monitor:** migration apply success, RLS policy count, table growth.
- **Success criteria:** migration replays cleanly on the CI postgres service;
  `__tests__/platform/` green.
- **Failure criteria:** any RLS or replay error → do not promote.
- **Without activation:** app works fully offline/local-first; events stay
  in-memory or on-device.

## 2. Remote experiment / flag service (NOT implemented remotely)

- **Purpose:** would allow server-driven experiments and flags.
- **Status:** local deterministic system only (`src/services/platform/experimentation.ts`).
  Activation would require a new service design + approval; no code path
  calls a remote flag service today.
- **Risk if activated later:** inconsistent assignment, privacy leakage.
- **Rollback:** kill switches force local defaults.
- **Without activation:** full local behavior, no loss.

## 3. Production monitoring / analytics backend

- **Purpose:** receive sanitized events from the existing analytics queue.
- **Status:** not configured; event schemas and registry are local
  (`src/services/platform/decisions.ts`, `analytics.ts`).
- **Risk:** PII leakage if a backend accepted unvalidated payloads —
  prevented by the registry + `sanitizeEvent` allowlists.
- **Rollback:** disable sink; queue stays on-device.
- **Without activation:** all evaluation/scorecard/funnel logic runs locally
  in tests and dev tooling.

## 4. Paid provider enablement (unchanged from Phase 4)

- Still gated behind the Edge Function proxy; cost policies
  (`src/services/platform/budgets.ts`) bind calls/images/resolution.
- No keys client-side; no activation performed here.
