# DATA_MODEL.md — AIWardrobe Canonical Data Model (Phase 7CZ)

Authoritative entity reference. Identity rules from Phases 1–4 are frozen;
this doc maps them, it does not redefine them.

## Identity (never collapse)

| Entity | Identity key | Rule |
|---|---|---|
| Product | product identity (Phase 1) | Physical product ≠ listing; listings reference products |
| Family | family id | Variants (color/size) share family, keep own identity |
| Variant | variant id (family + attributes) | Selected variant never silently swapped |
| Wardrobe item | local id + provenance | User-owned; owner-scoped RLS |
| FitEvidence / StyleEvidence | evidence id + source | Deterministic, versioned aggregation |
| VisualEvidence | parse id + image hash ref | Never stores raw private images in logs |
| Decision event | requestId + traceId (Phase 5) | Reproducible via provenance envelope |

## Ownership

All user tables: `owner_id = auth.uid()`, RLS `FOR ALL USING/WITH CHECK`.
Service-role paths exist only in Edge Functions, never in client code.

## Versioned aggregates

Fit DNA, Style DNA, ranking, BuyCheck, visual parsing carry domain versions
(Phase 5 contracts). Old versions remain comparable via golden fixtures.

## Growth tables (Phase 6, unapplied)

`growth_activation_state`, `growth_notification_prefs`, `growth_referrals`
(unique idempotency key, no-self check), `growth_savings_ledger`
(amount > 0, provenance required), `growth_feedback`.
