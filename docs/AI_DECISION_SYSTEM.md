# AI_DECISION_SYSTEM.md (Phase 7CZ)

How AIWardrobe recommendations are produced, bounded, and reproduced.

## Authority order (non-negotiable)

1. Deterministic hard rules (budget, activity, weather, variant integrity).
2. Evidence aggregates (FitEvidence, StyleEvidence, VisualEvidence).
3. Model output — treated as **evidence, never verdict**.

Vision/LLM output is advisory. Validators are authoritative; a failed
validation falls back to deterministic behavior and is labeled as such.

## Provenance envelope (Phase 5)

Every important decision carries: rulesVersion, catalogVersion,
wardrobeSnapshotVersion, style/fitProfileVersion, modelProvider/modelVersion,
prompt/schemaVersion, experimentId, createdAt. A recommendation from today
must remain explainable after code changes.

## Quality gates

- No hallucinated price history; unavailable history is stated.
- No fit guarantee; low evidence yields insufficient-evidence states.
- User visual correction always wins over parser output.
- No-buy recommendations are never weakened by shopping/affiliate incentives.
- Structured output validation on all provider JSON; prompt-injection
  defenses on untrusted product/merchant content.

## Cost safety

Per-feature budgets, caches, concurrency limits, kill switches
(remote vision, VTON, proactive rankers). Cost control never breaks core
wardrobe access.
