# Phase 4 Remote Activation — PREPARATION ONLY (do not apply)

This document lists every remote action that would be required to make the
Phase 4 multimodal layer fully operational. **No action here is authorized.**
Each item needs explicit user approval with the 10-point briefing (§0.2).

## 1. Hosted migration apply

- **Exact action:** apply
  `backend/supabase/migrations/20261104120000_phase4_visual_os.sql`
  (plus Phase 3 `20261103120000_phase3_style_os.sql` if not yet applied).
- **Why local-only is insufficient:** durable cross-device storage for visual
  assets metadata, evidence, reference looks, cache ledger, try-on requests,
  stylist plans.
- **Objects:** 6 new tables (`visual_assets`, `visual_evidence`,
  `reference_looks`, `inference_cache_ledger`, `tryon_requests`,
  `stylist_plans`) + 6 owner-scoped RLS policies. No changes to existing tables.
- **Privacy:** metadata and parsimonious references only; raw user images stay
  on-device unless a provider flow (item 2) is separately approved.
- **Cost/latency:** storage-only; negligible.
- **Failure modes:** policy misconfiguration could expose rows; mitigated by
  the `auth.uid() = user_id` pattern used by sibling migrations.
- **Rollback:** `drop table` the six tables (data loss of visual metadata only;
  wardrobe core untouched).
- **Tests:** migration file is additive and idempotent (`if not exists`);
  local suites in `__tests__/visualOS/` cover the domain layer.
- **Without activation:** app works fully offline/local-first; persistence
  falls back to on-device stores.

## 2. External AI provider enablement (vision / multimodal / VTON)

- **Exact action:** configure a provider behind the secured `ai-process` Edge
  Function proxy (secrets server-side only, never in the app bundle).
- **Why:** live garment/outfit parsing, reference interpretation, try-on
  renders cannot run on-device in this architecture.
- **Affected:** `ai-process` config + `GarmentVisionProvider` binding only;
  deterministic validators stay authoritative and unchanged.
- **Data sent:** single task-scoped image reference + allow-listed metadata
  (see `sanitizeProviderRequest`); never unrelated images, faces, locations,
  or body data.
- **Privacy implications:** image leaves device per request; payloads not
  retained; no training use without explicit opt-in.
- **Cost:** per-call vision pricing; bounded by `taskPolicy` caps
  (max calls/images/resolution per task, cache-first).
- **Latency:** one proxied round-trip per uncached parse; cached parses are local.
- **Failure modes:** provider outage → deterministic fallback (structured
  metadata only); malformed output → golden-contract recovery.
- **Rollback:** revert provider binding; fallback is the default.
- **Tests:** `MockVisionProvider` + fixture harness in `__tests__/visualOS/`;
  golden contracts pin JSON schema/enums/confidence ranges.
- **Without activation:** deterministic fallback remains fully usable.

## 3. Not required

No Edge Function deploy, no secret changes, no production backfill, no
scheduled jobs, no notifications, no storage uploads, no publishing. The
Phase 4 domain layer ships no remote calls of its own.
